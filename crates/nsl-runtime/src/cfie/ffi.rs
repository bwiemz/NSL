//! CFIE C-ABI surface.
//!
//! The compiled persistent-decode kernel calls these from PTX / CPU
//! fallback.  Single-GPU builds ship working Rust implementations;
//! multi-GPU deployments will replace them with CUDA-specific paths.

use std::sync::{Mutex, OnceLock};

use crate::cfie::ring_buffer::{RequestSlot, RingBuffer};

static GLOBAL_RING: OnceLock<Mutex<RingBuffer>> = OnceLock::new();

fn global() -> &'static Mutex<RingBuffer> {
    GLOBAL_RING.get_or_init(|| Mutex::new(RingBuffer::new(64)))
}

/// Resize the global ring buffer.  Drops any queued requests.  Called
/// once at server startup by the CFIE-compiled binary based on the
/// compile-time `Scheduler::max_active` value.
///
/// # Safety
/// Must be called *before* the persistent kernel launches.
#[no_mangle]
pub extern "C" fn nsl_cfie_ring_init(capacity: i64) -> i64 {
    if capacity <= 0 {
        return -1;
    }
    let mut rb = match global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    *rb = RingBuffer::new(capacity as usize);
    0
}

/// Host pushes a new request.  Returns 1 on success, 0 on full-buffer
/// back-pressure, -1 on internal error.
///
/// # Safety
/// All pointer fields are interpreted as opaque `u64`s; the compiled
/// kernel is responsible for validating them.
#[no_mangle]
pub extern "C" fn nsl_cfie_ring_push(
    sequence_id: i64,
    prompt_ptr: i64,
    prompt_len: i64,
    max_new_tokens: i64,
    grammar_start_state: i64,
    sampling_packed: i64,
) -> i64 {
    let slot = RequestSlot {
        sequence_id: sequence_id as u64,
        prompt_ptr: prompt_ptr as u64,
        prompt_len: prompt_len.max(0) as u32,
        max_new_tokens: max_new_tokens.max(0) as u32,
        grammar_start_state: grammar_start_state.max(0) as u32,
        sampling_packed: sampling_packed as u32,
    };
    let mut rb = match global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    if rb.push(slot) {
        1
    } else {
        0
    }
}

/// GPU-side scheduler pops a request.  On single-GPU test builds the
/// caller is the CPU fallback; on multi-GPU builds this is a stub
/// since the GPU reads the ring directly from pinned memory.
/// Returns 1 on success with `out_*` filled; 0 when the ring is empty.
///
/// # Safety
/// Out-parameter pointers must reference valid `i64` storage.
#[no_mangle]
pub unsafe extern "C" fn nsl_cfie_ring_pop(
    out_sequence_id: *mut i64,
    out_prompt_ptr: *mut i64,
    out_prompt_len: *mut i64,
    out_max_new_tokens: *mut i64,
    out_grammar_start_state: *mut i64,
    out_sampling_packed: *mut i64,
) -> i64 {
    if out_sequence_id.is_null()
        || out_prompt_ptr.is_null()
        || out_prompt_len.is_null()
        || out_max_new_tokens.is_null()
        || out_grammar_start_state.is_null()
        || out_sampling_packed.is_null()
    {
        return -1;
    }
    let mut rb = match global().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match rb.pop() {
        Some(slot) => {
            unsafe {
                *out_sequence_id = slot.sequence_id as i64;
                *out_prompt_ptr = slot.prompt_ptr as i64;
                *out_prompt_len = slot.prompt_len as i64;
                *out_max_new_tokens = slot.max_new_tokens as i64;
                *out_grammar_start_state = slot.grammar_start_state as i64;
                *out_sampling_packed = slot.sampling_packed as i64;
            }
            1
        }
        None => 0,
    }
}

/// Current in-flight request count — diagnostic hook for host-side
/// monitoring and tests.  (CLI surface today: `nsl build --cfie` /
/// `--cfie-report`; a live-status subcommand is future work.)
#[no_mangle]
pub extern "C" fn nsl_cfie_ring_len() -> i64 {
    match global().lock() {
        Ok(rb) => rb.len() as i64,
        Err(_) => -1,
    }
}

/// Query a compiled DFA state transition.  The compiled kernel walks
/// the transition table embedded as a PTX constant; this host-side
/// wrapper exists only so tests and Python clients can simulate the
/// grammar.
///
/// `table_ptr` / `num_states` / `vocab_size` describe the flattened
/// `[num_states][vocab_size]` transition table.  Returns the next
/// state or `-1` for a reject.
///
/// # Safety
/// `table_ptr` must point to an array of at least
/// `num_states * vocab_size` `u32`s.
#[no_mangle]
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
    use std::sync::{Mutex, MutexGuard, OnceLock};

    // Tests that call nsl_cfie_ring_init / push / pop race on GLOBAL_RING
    // when run in parallel.  Serialize them with a module-local guard.
    fn cfie_serial_lock() -> MutexGuard<'static, ()> {
        static SERIAL: OnceLock<Mutex<()>> = OnceLock::new();
        let m = SERIAL.get_or_init(|| Mutex::new(()));
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn init_accepts_positive_capacity() {
        let _serial = cfie_serial_lock();
        assert_eq!(nsl_cfie_ring_init(32), 0);
        assert_eq!(nsl_cfie_ring_init(0), -1);
        assert_eq!(nsl_cfie_ring_init(-1), -1);
    }

    #[test]
    fn push_and_pop_roundtrip_through_ffi() {
        let _serial = cfie_serial_lock();
        assert_eq!(nsl_cfie_ring_init(16), 0);
        assert_eq!(
            nsl_cfie_ring_push(42, 0x1000, 8, 128, 0, 0x0032_0032),
            1
        );
        assert_eq!(nsl_cfie_ring_len(), 1);

        let mut id = 0i64;
        let mut ptr = 0i64;
        let mut plen = 0i64;
        let mut mnt = 0i64;
        let mut gs = 0i64;
        let mut sp = 0i64;
        let ok = unsafe {
            nsl_cfie_ring_pop(&mut id, &mut ptr, &mut plen, &mut mnt, &mut gs, &mut sp)
        };
        assert_eq!(ok, 1);
        assert_eq!(id, 42);
        assert_eq!(ptr, 0x1000);
        assert_eq!(plen, 8);
        assert_eq!(mnt, 128);
    }

    #[test]
    fn pop_returns_zero_on_empty_ring() {
        let _serial = cfie_serial_lock();
        assert_eq!(nsl_cfie_ring_init(16), 0);
        // Drain anything left from earlier tests.
        while nsl_cfie_ring_len() > 0 {
            let mut id = 0i64;
            let mut a = 0i64;
            let mut b = 0i64;
            let mut c = 0i64;
            let mut d = 0i64;
            let mut e = 0i64;
            unsafe {
                nsl_cfie_ring_pop(&mut id, &mut a, &mut b, &mut c, &mut d, &mut e);
            }
        }
        let mut id = 0i64;
        let mut a = 0i64;
        let mut b = 0i64;
        let mut c = 0i64;
        let mut d = 0i64;
        let mut e = 0i64;
        let ok = unsafe {
            nsl_cfie_ring_pop(&mut id, &mut a, &mut b, &mut c, &mut d, &mut e)
        };
        assert_eq!(ok, 0);
    }

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
