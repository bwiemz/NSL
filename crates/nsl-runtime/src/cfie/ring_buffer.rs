//! Lock-free SPSC ring buffer used by CFIE's persistent decode kernel.
//!
//! The host (single producer) pushes new-request descriptors at the
//! tail; the GPU-side scheduler (single consumer) pops them from the
//! head.  We use **sequenced monotonic atomics** so the GPU can read
//! the tail index without coordinating a fence with the host — the
//! index never goes backwards and stale reads just delay the pickup
//! by one iteration.
//!
//! On a multi-GPU deployment the actual GPU-side consumer lives in
//! PTX; on single-GPU test builds the consumer is a simple CPU-side
//! helper used for validation.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Per-request metadata the host enqueues.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[derive(Default)]
pub struct RequestSlot {
    /// Opaque sequence identifier returned to the host as the request
    /// progresses.
    pub sequence_id: u64,
    /// Pointer (as `u64`) to the prompt token buffer in GPU memory.
    pub prompt_ptr: u64,
    /// Number of prompt tokens.
    pub prompt_len: u32,
    /// Requested maximum decode length.
    pub max_new_tokens: u32,
    /// Grammar DFA start state — 0 when no grammar is active.
    pub grammar_start_state: u32,
    /// Sampling params packed: low 16 bits = top_k, high 16 bits =
    /// temperature×1000 rounded.  The kernel decodes this in-place.
    pub sampling_packed: u32,
}


/// Fixed-capacity ring buffer.  `CAPACITY` must be a power of two.
pub struct RingBuffer {
    capacity_mask: usize,
    slots: Box<[RequestSlot]>,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl RingBuffer {
    /// Construct a new buffer with the given capacity (rounded up to the
    /// next power of two).
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two().max(2);
        let slots = vec![RequestSlot::default(); capacity].into_boxed_slice();
        Self {
            capacity_mask: capacity - 1,
            slots,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity_mask + 1
    }

    pub fn len(&self) -> usize {
        self.tail
            .load(Ordering::Acquire)
            .wrapping_sub(self.head.load(Ordering::Acquire))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }

    /// Enqueue a new request.  Returns `false` when the buffer is
    /// full — callers are expected to back off rather than block.
    pub fn push(&mut self, slot: RequestSlot) -> bool {
        if self.is_full() {
            return false;
        }
        let tail = self.tail.load(Ordering::Relaxed);
        let idx = tail & self.capacity_mask;
        self.slots[idx] = slot;
        // Release so the consumer sees the written slot before the
        // bumped tail index.
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        true
    }

    /// Dequeue the next request.  Returns `None` when the buffer is
    /// empty (the typical case on the kernel's hot path — the
    /// scheduler just keeps looping).
    pub fn pop(&mut self) -> Option<RequestSlot> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let idx = head & self.capacity_mask;
        let slot = self.slots[idx];
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Some(slot)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_slot(id: u64) -> RequestSlot {
        RequestSlot {
            sequence_id: id,
            prompt_ptr: 0xDEAD_BEEF,
            prompt_len: 32,
            max_new_tokens: 128,
            grammar_start_state: 0,
            sampling_packed: 0x0032_0032, // top_k=50, temp×1000=50
        }
    }

    #[test]
    fn push_pop_single_element() {
        let mut rb = RingBuffer::new(4);
        assert!(rb.is_empty());
        assert!(rb.push(make_slot(1)));
        assert_eq!(rb.len(), 1);
        let s = rb.pop().unwrap();
        assert_eq!(s.sequence_id, 1);
        assert!(rb.is_empty());
    }

    #[test]
    fn push_fills_and_pop_drains_fifo() {
        let mut rb = RingBuffer::new(4);
        for i in 0..4 {
            assert!(rb.push(make_slot(i)));
        }
        assert!(rb.is_full());
        assert!(!rb.push(make_slot(99)));
        for i in 0..4 {
            assert_eq!(rb.pop().unwrap().sequence_id, i);
        }
        assert!(rb.is_empty());
    }

    #[test]
    fn capacity_rounds_up_to_power_of_two() {
        assert_eq!(RingBuffer::new(5).capacity(), 8);
        assert_eq!(RingBuffer::new(8).capacity(), 8);
        assert_eq!(RingBuffer::new(0).capacity(), 2);
    }

    #[test]
    fn wraparound_preserves_order() {
        let mut rb = RingBuffer::new(4);
        // Push 4, pop 2, push 2, verify sequence.
        for i in 0..4 {
            rb.push(make_slot(i));
        }
        for _ in 0..2 {
            rb.pop();
        }
        rb.push(make_slot(100));
        rb.push(make_slot(101));
        assert_eq!(rb.pop().unwrap().sequence_id, 2);
        assert_eq!(rb.pop().unwrap().sequence_id, 3);
        assert_eq!(rb.pop().unwrap().sequence_id, 100);
        assert_eq!(rb.pop().unwrap().sequence_id, 101);
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut rb = RingBuffer::new(4);
        assert!(rb.pop().is_none());
    }
}
