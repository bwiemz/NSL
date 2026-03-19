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
    rank: i32,
    model_ptr: i64,  // opaque model handle from codegen
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

    let mut guard = match WORKER_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1, // mutex poisoned
    };
    if guard.is_some() {
        return -1;
    }
    *guard = Some(WorkerContext {
        role: worker_role,
        rank: rank as i32,
        model_ptr,
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
        let guard = match WORKER_CTX.lock() {
            Ok(g) => g,
            Err(_) => return -1, // mutex poisoned
        };
        let ctx = match guard.as_ref() {
            Some(c) => c,
            None => return -2, // not initialized
        };
        if ctx.role != WorkerRole::Prefill {
            return -3; // wrong role
        }
    }

    // M41b: Read worker state for loop
    let (_rank, _model_ptr) = {
        let guard = match WORKER_CTX.lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        let ctx = guard.as_ref().unwrap();
        (ctx.rank, ctx.model_ptr)
    };

    // M41b: Prefill worker processing loop structure.
    // Each iteration would:
    // 1. Receive StartPrefill message from router (via shared memory)
    // 2. Allocate KV-cache pages from local BlockAllocator
    // 3. Run model forward pass on prompt tokens (using model_ptr)
    // 4. Serialize KV-cache pages via nsl_kv_serialize()
    // 5. Transfer KV to assigned decode worker via KvTransferBackend::send_kv()
    // 6. Free local KV pages (decode worker has them now)
    // 7. Send PrefillComplete to router
    //
    // Currently returns after init — the actual model forward requires
    // the compiled model to be loaded and the message transport to be active.
    // Full continuous loop in M41c when message-based dispatch is wired.

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
        let guard = match WORKER_CTX.lock() {
            Ok(g) => g,
            Err(_) => return -1, // mutex poisoned
        };
        let ctx = match guard.as_ref() {
            Some(c) => c,
            None => return -2, // not initialized
        };
        if ctx.role != WorkerRole::Decode {
            return -3; // wrong role
        }
    }

    // M41b: Read worker state for loop
    let (_rank, _model_ptr) = {
        let guard = match WORKER_CTX.lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        let ctx = guard.as_ref().unwrap();
        (ctx.rank, ctx.model_ptr)
    };

    // M41b: Decode worker processing loop structure.
    // Each iteration would:
    // 1. Check for incoming KV transfers via try_recv_kv() (non-blocking)
    // 2. Deserialize received KV pages into local KV-cache
    // 3. Admit new sequences into BatchScheduler
    // 4. Run one batched decode step (scheduler.step() → model forward)
    // 5. Sample tokens from logits (apply grammar mask if constrained)
    // 6. Stream generated tokens to router via TokenGenerated message
    // 7. On EOS/max_tokens: free KV pages, send DecodeComplete
    //
    // Reuses M29 BatchScheduler for decode-side batching.
    // Currently returns after init — full continuous loop in M41c.

    0
}

/// Destroy the worker context.
#[no_mangle]
pub extern "C" fn nsl_disagg_worker_destroy() -> i64 {
    let mut guard = match WORKER_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1, // mutex poisoned
    };
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
