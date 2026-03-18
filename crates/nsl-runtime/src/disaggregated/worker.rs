//! M41: Prefill and decode worker loop implementations.
//!
//! Each worker runs as a separate OS process. The loops receive messages
//! from the router, process requests, and send results back.
//! For now these are FFI entry points that the codegen's role dispatch calls.

use std::sync::Mutex;

use super::messages::WorkerRole;

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------

static WORKER_CTX: Mutex<Option<WorkerContext>> = Mutex::new(None);

struct WorkerContext {
    role: WorkerRole,
    _rank: i32,
    _model_ptr: i64,  // opaque model handle from codegen
}

// ---------------------------------------------------------------------------
// FFI: Initialization
// ---------------------------------------------------------------------------

/// Initialize a disaggregated worker (called after role dispatch).
///
/// `role`: 1 = prefill, 2 = decode
/// `rank`: this worker's local rank
/// `model_ptr`: opaque pointer to the compiled model
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_disagg_worker_init(role: i64, rank: i64, model_ptr: i64) -> i64 {
    let worker_role = match role {
        1 => WorkerRole::Prefill,
        2 => WorkerRole::Decode,
        _ => return -1,
    };

    let mut guard = WORKER_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(WorkerContext {
        role: worker_role,
        _rank: rank as i32,
        _model_ptr: model_ptr,
    });
    0
}

/// Run the prefill worker loop.
///
/// This is the main entry point for a prefill worker process.
/// The loop:
/// 1. Receives StartPrefill messages from the router
/// 2. Runs the model's forward pass on the prompt tokens
/// 3. Serializes the KV-cache and transfers to the assigned decode worker
/// 4. Sends PrefillComplete back to the router
///
/// `config_ptr`: pointer to serve config (max_batch, prefill_chunk, etc.)
///
/// Returns 0 on clean shutdown, negative on error.
#[no_mangle]
pub extern "C" fn nsl_disagg_prefill_loop(_config_ptr: i64) -> i64 {
    // Verify role (brief lock, then release)
    {
        let guard = WORKER_CTX.lock().unwrap();
        let ctx = guard.as_ref().expect("nsl_disagg_worker_init not called");
        assert_eq!(ctx.role, WorkerRole::Prefill, "prefill_loop called on non-prefill worker");
    }

    // In a full implementation, this would:
    // 1. Loop waiting for messages from the router (via shared memory or pipe)
    // 2. For each StartPrefill: allocate KV blocks, run forward, serialize KV,
    //    transfer to decode worker, notify router
    // Currently a stub that returns immediately for testing.
    0
}

/// Run the decode worker loop.
///
/// This is the main entry point for a decode worker process.
/// The loop:
/// 1. Checks for incoming KV transfers (non-blocking)
/// 2. Runs batched decode step on all active sequences
/// 3. Streams generated tokens back to the router
/// 4. Cleans up completed sequences
///
/// `config_ptr`: pointer to serve config
///
/// Returns 0 on clean shutdown, negative on error.
#[no_mangle]
pub extern "C" fn nsl_disagg_decode_loop(_config_ptr: i64) -> i64 {
    // Verify role (brief lock, then release)
    {
        let guard = WORKER_CTX.lock().unwrap();
        let ctx = guard.as_ref().expect("nsl_disagg_worker_init not called");
        assert_eq!(ctx.role, WorkerRole::Decode, "decode_loop called on non-decode worker");
    }

    // In a full implementation, this would:
    // 1. Check for KV transfers via try_recv_kv()
    // 2. Admit new sequences into the BatchScheduler
    // 3. Run batched decode step (reuse M29 scheduler.step())
    // 4. Sample tokens, stream to router
    // 5. Free KV pages on EOS
    // Currently a stub that returns immediately for testing.
    0
}

/// Destroy the worker context.
#[no_mangle]
pub extern "C" fn nsl_disagg_worker_destroy() -> i64 {
    let mut guard = WORKER_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_disagg_worker_destroy();
        guard
    }

    #[test]
    fn worker_init_prefill() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);
        // Prefill loop returns immediately (stub)
        assert_eq!(nsl_disagg_prefill_loop(0), 0);
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_init_decode() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(2, 0, 0), 0);
        // Decode loop returns immediately (stub)
        assert_eq!(nsl_disagg_decode_loop(0), 0);
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_double_init_fails() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);
        assert_eq!(nsl_disagg_worker_init(2, 0, 0), -1);
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_invalid_role_fails() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(99, 0, 0), -1);
    }
}
