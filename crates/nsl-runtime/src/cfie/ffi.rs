//! CFIE C-ABI surface: the KV sequence-slot free-list and the host-side
//! grammar-transition lookup.  The compiled serve binary calls the slot
//! functions from the serve-init wiring emitted by `serve.rs`; the decode
//! loop itself is in `engine.rs`.

use std::sync::{Mutex, OnceLock};

/// Query a compiled DFA state transition: a host-side lookup for tests
/// of the compile-time grammar tables.  No generation driver calls it.
///
/// `table_ptr` / `num_states` / `vocab_size` describe the flattened
/// `[num_states][vocab_size]` transition table.  Returns the next
/// state or `-1` for a reject.
///
/// # Safety
/// `table_ptr` must point to an array of at least
/// `num_states * vocab_size` `u32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nsl_cfie_grammar_transition(
    table_ptr: *const u32,
    num_states: i64,
    vocab_size: i64,
    state: i64,
    token: i64,
) -> i64 {
    if table_ptr.is_null()
        || num_states <= 0
        || vocab_size <= 0
        || state < 0
        || state >= num_states
        || token < 0
        || token >= vocab_size
    {
        return -1;
    }
    let idx = (state as usize) * (vocab_size as usize) + (token as usize);
    let next = unsafe { *table_ptr.add(idx) };
    if next == u32::MAX {
        -1
    } else {
        next as i64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_transition_respects_reject_sentinel() {
        let table: Vec<u32> = vec![1, u32::MAX, 2, u32::MAX];
        let ptr = table.as_ptr();
        // (state 0, token 0) → 1
        let r = unsafe { nsl_cfie_grammar_transition(ptr, 2, 2, 0, 0) };
        assert_eq!(r, 1);
        // (state 0, token 1) → reject.
        let r = unsafe { nsl_cfie_grammar_transition(ptr, 2, 2, 0, 1) };
        assert_eq!(r, -1);
        // Out-of-range state → -1.
        let r = unsafe { nsl_cfie_grammar_transition(ptr, 2, 2, 5, 0) };
        assert_eq!(r, -1);
    }

    #[test]
    fn grammar_null_pointer_rejects_safely() {
        let r = unsafe { nsl_cfie_grammar_transition(std::ptr::null(), 1, 1, 0, 0) };
        assert_eq!(r, -1);
    }
}

// ---------------------------------------------------------------------------
// KV slot allocator (Feature 1 runtime half)
// ---------------------------------------------------------------------------

use crate::cfie::kv_slots::KvSlotAllocator;

static GLOBAL_KV_SLOTS: OnceLock<Mutex<Option<KvSlotAllocator>>> = OnceLock::new();

/// Shared with `cfie::engine` — the KV-pool alloc/destroy FFIs attach
/// and detach the device buffer, and the launch FFIs inject
/// `device_base()` as the kernels' `kv_base` parameter.
pub(crate) fn kv_slots_global() -> &'static Mutex<Option<KvSlotAllocator>> {
    GLOBAL_KV_SLOTS.get_or_init(|| Mutex::new(None))
}

fn slot_arg(v: i64) -> Option<u32> {
    u32::try_from(v).ok()
}

/// (Re)create the global KV slot allocator.  Called once at server
/// startup by the CFIE-compiled binary from the compile-time
/// `slot_count x per_slot_tokens` envelope; drops any prior state.
/// Returns 0 on success, -1 on non-positive / out-of-range arguments.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_cfie_kv_slots_init(slot_count: i64, per_slot_tokens: i64) -> i64 {
    if slot_count <= 0 || per_slot_tokens <= 0 {
        return -1;
    }
    let (slots, tokens) = match (slot_arg(slot_count), slot_arg(per_slot_tokens)) {
        (Some(s), Some(t)) => (s, t),
        _ => return -1,
    };
    // Mirror the decode kernel's invariant: the global token pool
    // (slot_count * per_slot_tokens) must fit the kernel's 32-bit
    // token index.  Also bounds the eager per-slot Vec allocations.
    if (slots as u64) * (tokens as u64) > u32::MAX as u64 {
        return -1;
    }
    let mut g = match kv_slots_global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    *g = Some(KvSlotAllocator::new(slots, tokens));
    0
}

/// Claim a slot for a new sequence.  Returns the slot id, or -1 when
/// all slots are occupied or the allocator is uninitialized — the
/// scheduler back-pressures the request ring on -1.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_cfie_kv_slot_acquire() -> i64 {
    let mut g = match kv_slots_global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match g.as_mut().and_then(|a| a.acquire()) {
        Some(slot) => slot as i64,
        None => -1,
    }
}

/// Return a finished sequence's slot to the free-list.  Returns 1 on
/// success, 0 when the slot is not active (double-release or bad id),
/// -1 on internal error / uninitialized allocator.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_cfie_kv_slot_release(slot: i64) -> i64 {
    let slot = match slot_arg(slot) {
        Some(s) => s,
        None => return 0,
    };
    let mut g = match kv_slots_global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match g.as_mut() {
        Some(a) => {
            if a.release(slot) {
                1
            } else {
                0
            }
        }
        None => -1,
    }
}

/// Record `n_tokens` appended to the slot, returning the new sequence
/// length.  Returns -1 when the append would exceed the per-slot
/// budget or the slot is inactive — the compiled decode loop checks
/// this *before* writing KV entries, so a refusal never corrupts the
/// buffer.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_cfie_kv_slot_advance(slot: i64, n_tokens: i64) -> i64 {
    let (slot, n) = match (slot_arg(slot), slot_arg(n_tokens)) {
        (Some(s), Some(n)) => (s, n),
        _ => return -1,
    };
    let mut g = match kv_slots_global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match g.as_mut().and_then(|a| a.advance(slot, n)) {
        Some(new_len) => new_len as i64,
        None => -1,
    }
}

/// Un-append `n_tokens` after a speculative-decoding rejection,
/// returning the new sequence length.  Returns -1 when `n_tokens`
/// exceeds the current length or the slot is inactive.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_cfie_kv_slot_rollback(slot: i64, n_tokens: i64) -> i64 {
    let (slot, n) = match (slot_arg(slot), slot_arg(n_tokens)) {
        (Some(s), Some(n)) => (s, n),
        _ => return -1,
    };
    let mut g = match kv_slots_global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match g.as_mut().and_then(|a| a.rollback(slot, n)) {
        Some(new_len) => new_len as i64,
        None => -1,
    }
}

/// Number of currently active sequence slots — diagnostic hook for
/// host-side monitoring and tests.  Returns -1 before init.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_cfie_kv_slots_active() -> i64 {
    let g = match kv_slots_global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match g.as_ref() {
        Some(a) => a.active_count() as i64,
        None => -1,
    }
}

/// Record the device buffer backing the KV slots.  The runtime's GPU
/// allocator performs (and owns) the actual allocation; this only
/// stores the address the decode kernel receives as a launch
/// parameter.  `base` of 0 detaches (CPU-only builds).  Returns 0 on
/// success, -1 on negative `base`/`bytes` or uninitialized allocator.
/// (Device VAs fit in the positive i64 range; a negative value here is
/// a caller bug, not a pointer.)
#[unsafe(no_mangle)]
pub extern "C" fn nsl_cfie_kv_attach_device(base: i64, bytes: i64) -> i64 {
    if base < 0 || bytes < 0 {
        return -1;
    }
    let mut g = match kv_slots_global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match g.as_mut() {
        Some(a) => {
            a.attach_device_buffer(base as u64, bytes as u64);
            0
        }
        None => -1,
    }
}

#[cfg(test)]
mod kv_slot_tests {
    use super::*;
    use std::sync::MutexGuard;

    // These tests share GLOBAL_KV_SLOTS with the engine tests in
    // engine.rs — both modules must serialize under the SAME lock or
    // `cargo test --lib cfie` interleaves slot-allocator mutations
    // across modules (reproduced flake).
    fn kv_serial_lock() -> MutexGuard<'static, ()> {
        crate::cfie::test_serial_lock()
    }

    fn deinit_kv_slots() {
        *kv_slots_global().lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    #[test]
    fn calls_before_init_return_error() {
        let _serial = kv_serial_lock();
        deinit_kv_slots();
        assert_eq!(nsl_cfie_kv_slot_acquire(), -1);
        assert_eq!(nsl_cfie_kv_slot_release(0), -1);
        assert_eq!(nsl_cfie_kv_slot_advance(0, 1), -1);
        assert_eq!(nsl_cfie_kv_slot_rollback(0, 1), -1);
        assert_eq!(nsl_cfie_kv_slots_active(), -1);
        assert_eq!(nsl_cfie_kv_attach_device(0x1000, 64), -1);
    }

    #[test]
    fn init_rejects_bad_args() {
        let _serial = kv_serial_lock();
        assert_eq!(nsl_cfie_kv_slots_init(0, 8), -1);
        assert_eq!(nsl_cfie_kv_slots_init(8, 0), -1);
        assert_eq!(nsl_cfie_kv_slots_init(-1, 8), -1);
        assert_eq!(nsl_cfie_kv_slots_init(8, -1), -1);
        assert_eq!(nsl_cfie_kv_slots_init(i64::MAX, 8), -1);
    }

    #[test]
    fn acquire_advance_release_roundtrip() {
        let _serial = kv_serial_lock();
        assert_eq!(nsl_cfie_kv_slots_init(2, 4), 0);
        let s0 = nsl_cfie_kv_slot_acquire();
        let s1 = nsl_cfie_kv_slot_acquire();
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(nsl_cfie_kv_slot_acquire(), -1); // exhausted
        assert_eq!(nsl_cfie_kv_slots_active(), 2);

        assert_eq!(nsl_cfie_kv_slot_advance(s0, 3), 3);
        assert_eq!(nsl_cfie_kv_slot_advance(s0, 1), 4);
        assert_eq!(nsl_cfie_kv_slot_advance(s0, 1), -1); // past budget
        assert_eq!(nsl_cfie_kv_slot_rollback(s0, 2), 2);
        assert_eq!(nsl_cfie_kv_slot_rollback(s0, 3), -1); // n > seq_len

        assert_eq!(nsl_cfie_kv_slot_release(s0), 1);
        assert_eq!(nsl_cfie_kv_slot_release(s0), 0); // double release
        assert_eq!(nsl_cfie_kv_slot_release(99), 0); // out of range
        assert_eq!(nsl_cfie_kv_slot_release(-3), 0); // negative id
        assert_eq!(nsl_cfie_kv_slots_active(), 1);
        // LIFO recycle hands the released slot back.
        assert_eq!(nsl_cfie_kv_slot_acquire(), s0);
    }

    #[test]
    fn advance_rejects_negative_and_inactive() {
        let _serial = kv_serial_lock();
        assert_eq!(nsl_cfie_kv_slots_init(1, 8), 0);
        assert_eq!(nsl_cfie_kv_slot_advance(-1, 1), -1);
        assert_eq!(nsl_cfie_kv_slot_advance(0, -1), -1);
        assert_eq!(nsl_cfie_kv_slot_advance(0, 1), -1); // never acquired
    }

    #[test]
    fn attach_device_records_after_init() {
        let _serial = kv_serial_lock();
        assert_eq!(nsl_cfie_kv_slots_init(1, 8), 0);
        assert_eq!(nsl_cfie_kv_attach_device(0x1000, 4096), 0);
        assert_eq!(nsl_cfie_kv_attach_device(0, 0), 0); // detach
        assert_eq!(nsl_cfie_kv_attach_device(0x1000, -1), -1);
    }
}
