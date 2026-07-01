//! KV sequence-slot free-list — CFIE Feature 1's runtime half.
//!
//! The compile-time KV pass sizes one contiguous device buffer as
//! `slot_count x per_slot_tokens` and bakes the per-slot stride into
//! the decode kernel.  What the serve scheduler needs at runtime is
//! therefore *not* a block allocator: every sequence gets a whole
//! pre-sized slot, so a LIFO free-list of slot ids is sufficient.
//!
//! This module never touches CUDA — the buffer is allocated by the
//! runtime's GPU allocator and handed in via
//! [`KvSlotAllocator::attach_device_buffer`], keeping the allocator
//! pure and unit-testable on CPU-only builds.

/// Free-list allocator over fixed-size KV sequence slots.
pub struct KvSlotAllocator {
    slot_count: u32,
    per_slot_tokens: u32,
    /// LIFO stack of free slot ids — hottest slot is reused first so
    /// its KV pages stay warm in L2.
    free: Vec<u32>,
    active: Vec<bool>,
    /// Tokens appended so far, indexed by slot; meaningful only while
    /// the slot is active.
    seq_lens: Vec<u32>,
    /// Raw device pointer of the pre-allocated KV buffer; 0 when
    /// unattached (CPU-only builds and tests).
    device_base: u64,
    device_bytes: u64,
}

impl KvSlotAllocator {
    pub fn new(slot_count: u32, per_slot_tokens: u32) -> Self {
        // Reverse so the first `acquire` hands out slot 0.
        let free: Vec<u32> = (0..slot_count).rev().collect();
        Self {
            slot_count,
            per_slot_tokens,
            free,
            active: vec![false; slot_count as usize],
            seq_lens: vec![0; slot_count as usize],
            device_base: 0,
            device_bytes: 0,
        }
    }

    pub fn slot_count(&self) -> u32 {
        self.slot_count
    }

    pub fn per_slot_tokens(&self) -> u32 {
        self.per_slot_tokens
    }

    /// Claim a free slot for a new sequence.  Returns `None` when all
    /// slots are occupied — the scheduler back-pressures the request
    /// ring rather than blocking.
    pub fn acquire(&mut self) -> Option<u32> {
        let slot = self.free.pop()?;
        self.active[slot as usize] = true;
        self.seq_lens[slot as usize] = 0;
        Some(slot)
    }

    /// Return a slot to the free-list.  Safe against double-release
    /// and out-of-range ids: both return `false` without corrupting
    /// the free-list.
    pub fn release(&mut self, slot: u32) -> bool {
        if slot >= self.slot_count || !self.active[slot as usize] {
            return false;
        }
        self.active[slot as usize] = false;
        self.seq_lens[slot as usize] = 0;
        self.free.push(slot);
        true
    }

    /// Record `n_tokens` appended to the slot's KV region, returning
    /// the new sequence length.  Returns `None` when the slot is
    /// inactive or the append would exceed `per_slot_tokens` — the
    /// caller must learn about overflow *before* writing, so the
    /// length is left unchanged on refusal.
    pub fn advance(&mut self, slot: u32, n_tokens: u32) -> Option<u32> {
        if slot >= self.slot_count || !self.active[slot as usize] {
            return None;
        }
        let cur = self.seq_lens[slot as usize];
        let new = cur.checked_add(n_tokens)?;
        if new > self.per_slot_tokens {
            return None;
        }
        self.seq_lens[slot as usize] = new;
        Some(new)
    }

    /// Un-append `n_tokens` (speculative-decoding rejection).  Returns
    /// the new sequence length, or `None` when the slot is inactive or
    /// `n_tokens` exceeds the current length — rolling back tokens
    /// that were never appended is a scheduler bug, not a floor-to-0.
    pub fn rollback(&mut self, slot: u32, n_tokens: u32) -> Option<u32> {
        if slot >= self.slot_count || !self.active[slot as usize] {
            return None;
        }
        let cur = self.seq_lens[slot as usize];
        if n_tokens > cur {
            return None;
        }
        let new = cur - n_tokens;
        self.seq_lens[slot as usize] = new;
        Some(new)
    }

    /// Current sequence length of an active slot.
    pub fn seq_len(&self, slot: u32) -> Option<u32> {
        if slot >= self.slot_count || !self.active[slot as usize] {
            return None;
        }
        Some(self.seq_lens[slot as usize])
    }

    pub fn active_count(&self) -> u32 {
        self.slot_count - self.free.len() as u32
    }

    /// Record the externally-allocated device buffer backing the
    /// slots.  The GPU allocator owns the allocation and its lifetime;
    /// this only stores the address for the decode kernel's launch
    /// parameters.
    pub fn attach_device_buffer(&mut self, base: u64, bytes: u64) {
        self.device_base = base;
        self.device_bytes = bytes;
    }

    pub fn device_base(&self) -> u64 {
        self.device_base
    }

    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_until_exhausted_then_release_recycles_lifo() {
        let mut a = KvSlotAllocator::new(3, 16);
        assert_eq!(a.acquire(), Some(0));
        assert_eq!(a.acquire(), Some(1));
        assert_eq!(a.acquire(), Some(2));
        assert_eq!(a.acquire(), None);
        assert_eq!(a.active_count(), 3);

        assert!(a.release(0));
        assert!(a.release(2));
        // LIFO: last released comes back first.
        assert_eq!(a.acquire(), Some(2));
        assert_eq!(a.acquire(), Some(0));
        assert_eq!(a.acquire(), None);
    }

    #[test]
    fn double_release_and_out_of_range_are_safe() {
        let mut a = KvSlotAllocator::new(2, 16);
        let s = a.acquire().unwrap();
        assert!(a.release(s));
        assert!(!a.release(s));
        assert!(!a.release(99));
        // Free-list must not have grown from the bad releases.
        assert_eq!(a.acquire(), Some(s));
        assert_eq!(a.acquire(), Some(1));
        assert_eq!(a.acquire(), None);
    }

    #[test]
    fn advance_refuses_overflow_at_exact_budget() {
        let mut a = KvSlotAllocator::new(1, 8);
        let s = a.acquire().unwrap();
        assert_eq!(a.advance(s, 7), Some(7));
        assert_eq!(a.advance(s, 2), None);
        // Refusal must leave the length untouched.
        assert_eq!(a.seq_len(s), Some(7));
        assert_eq!(a.advance(s, 1), Some(8));
        assert_eq!(a.advance(s, 1), None);
        assert_eq!(a.seq_len(s), Some(8));
    }

    #[test]
    fn advance_rejects_inactive_and_out_of_range_slots() {
        let mut a = KvSlotAllocator::new(2, 8);
        assert_eq!(a.advance(0, 1), None);
        assert_eq!(a.advance(5, 1), None);
        let s = a.acquire().unwrap();
        a.release(s);
        assert_eq!(a.advance(s, 1), None);
        assert_eq!(a.seq_len(s), None);
    }

    #[test]
    fn rollback_subtracts_and_errors_past_zero() {
        let mut a = KvSlotAllocator::new(1, 16);
        let s = a.acquire().unwrap();
        assert_eq!(a.advance(s, 5), Some(5));
        assert_eq!(a.rollback(s, 2), Some(3));
        assert_eq!(a.rollback(s, 3), Some(0));
        assert_eq!(a.rollback(s, 1), None);
        assert_eq!(a.seq_len(s), Some(0));
        assert_eq!(a.rollback(99, 0), None);
    }

    #[test]
    fn release_resets_seq_len_for_next_tenant() {
        let mut a = KvSlotAllocator::new(1, 8);
        let s = a.acquire().unwrap();
        assert_eq!(a.advance(s, 4), Some(4));
        assert!(a.release(s));
        let s2 = a.acquire().unwrap();
        assert_eq!(s2, s);
        assert_eq!(a.seq_len(s2), Some(0));
    }

    #[test]
    fn active_count_tracks_acquire_release() {
        let mut a = KvSlotAllocator::new(4, 8);
        assert_eq!(a.active_count(), 0);
        let s0 = a.acquire().unwrap();
        let s1 = a.acquire().unwrap();
        assert_eq!(a.active_count(), 2);
        a.release(s0);
        assert_eq!(a.active_count(), 1);
        a.release(s0); // double-release must not change the count
        assert_eq!(a.active_count(), 1);
        a.release(s1);
        assert_eq!(a.active_count(), 0);
    }

    #[test]
    fn device_buffer_attach_records_handle() {
        let mut a = KvSlotAllocator::new(1, 8);
        assert_eq!(a.device_base(), 0);
        a.attach_device_buffer(0x7f00_0000_1000, 4096);
        assert_eq!(a.device_base(), 0x7f00_0000_1000);
        assert_eq!(a.device_bytes(), 4096);
    }

    #[test]
    fn zero_slot_allocator_never_acquires() {
        let mut a = KvSlotAllocator::new(0, 8);
        assert_eq!(a.acquire(), None);
        assert_eq!(a.active_count(), 0);
        assert!(!a.release(0));
    }
}
