//! M43: Pipeline send/recv communication stubs.
//!
//! Point-to-point send/recv for activations and gradients between adjacent
//! pipeline stages. All functions are stubs returning 0 for M43a; real
//! implementations will use SharedMem (matching M30's SimulatedBackend pattern)
//! or NCCL point-to-point in M43b.

use std::sync::Mutex;

static PIPELINE_CTX: Mutex<Option<PipelineContext>> = Mutex::new(None);

struct PipelineContext {
    _num_stages: usize,
    _num_micro_batches: usize,
}

/// Initialize the pipeline communication context.
/// `schedule_type`: 0 = 1F1B, 1 = GPipe.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_pipeline_init(
    num_stages: i64,
    schedule_type: i64,
    num_micro_batches: i64,
) -> i64 {
    let _ = schedule_type; // used in M43b for schedule selection
    let mut guard = PIPELINE_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(PipelineContext {
        _num_stages: num_stages as usize,
        _num_micro_batches: num_micro_batches as usize,
    });
    0
}

/// Send a tensor to the next pipeline stage. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_pipeline_send(
    _tensor_ptr: i64,
    _dst_rank: i64,
    _tag: i64,
    _stream: i64,
) -> i64 {
    0
}

/// Receive a tensor from the previous pipeline stage. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_pipeline_recv(
    _shape_ptr: i64,
    _ndim: i64,
    _dtype: i64,
    _src_rank: i64,
    _tag: i64,
    _stream: i64,
) -> i64 {
    0
}

/// Send gradients to the previous pipeline stage. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_pipeline_send_grad(
    _grad_ptr: i64,
    _dst_rank: i64,
    _tag: i64,
    _stream: i64,
) -> i64 {
    0
}

/// Receive gradients from the next pipeline stage. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_pipeline_recv_grad(
    _shape_ptr: i64,
    _ndim: i64,
    _dtype: i64,
    _src_rank: i64,
    _tag: i64,
    _stream: i64,
) -> i64 {
    0
}

/// Pipeline barrier — synchronize all stages. Stub: returns 0.
#[no_mangle]
pub extern "C" fn nsl_pipeline_barrier() -> i64 {
    0
}

/// Destroy the pipeline communication context.
/// Returns 0 on success, -1 if not initialized.
#[no_mangle]
pub extern "C" fn nsl_pipeline_destroy() -> i64 {
    let mut guard = PIPELINE_CTX.lock().unwrap();
    if guard.is_none() {
        return -1;
    }
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_init_destroy_lifecycle() {
        // Ensure clean state
        {
            let mut guard = PIPELINE_CTX.lock().unwrap();
            *guard = None;
        }

        // First init succeeds
        assert_eq!(nsl_pipeline_init(4, 0, 8), 0);
        // Double init fails
        assert_eq!(nsl_pipeline_init(4, 0, 8), -1);
        // Destroy succeeds
        assert_eq!(nsl_pipeline_destroy(), 0);
        // Double destroy fails
        assert_eq!(nsl_pipeline_destroy(), -1);
        // Re-init after destroy succeeds
        assert_eq!(nsl_pipeline_init(2, 1, 4), 0);
        // Cleanup
        assert_eq!(nsl_pipeline_destroy(), 0);
    }

    #[test]
    fn test_pipeline_barrier_returns_zero() {
        assert_eq!(nsl_pipeline_barrier(), 0);
    }
}
