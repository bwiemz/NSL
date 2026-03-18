//! M39b: Runtime assertions for vmap batch-size checking.

/// Check that a tensor's batch dimension matches the expected batch size.
///
/// Called at the entry of batched functions to validate all batch-variant
/// arguments have consistent batch sizes.
///
/// Parameters (all i64 for Cranelift):
/// - tensor_ptr: pointer to NslTensor
/// - expected_batch: expected batch size (from first variant arg)
/// - batch_dim: which dimension is the batch dimension
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_vmap_check_batch(
    tensor_ptr: i64,
    expected_batch: i64,
    batch_dim: i64,
) -> i64 {
    if tensor_ptr == 0 {
        return -1;
    }
    let tensor = unsafe { &*(tensor_ptr as *const crate::tensor::NslTensor) };
    let dim = batch_dim as usize;
    if dim >= tensor.ndim as usize {
        eprintln!(
            "vmap: batch_dim {} out of range for tensor with ndim {}",
            dim, tensor.ndim
        );
        return -1;
    }
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) };
    let actual = shape[dim];
    if actual != expected_batch {
        eprintln!(
            "vmap batch size mismatch: expected {} at dim {}, got {}",
            expected_batch, dim, actual
        );
        return -1;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_tensor_returns_error() {
        assert_eq!(nsl_vmap_check_batch(0, 32, 0), -1);
    }
}
