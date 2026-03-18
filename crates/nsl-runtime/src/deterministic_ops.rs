//! M46: Deterministic runtime operation variants (stubs).
//!
//! These are called when --deterministic is active, replacing the default
//! non-deterministic GPU kernels. Full implementations (sort-based reduction
//! PTX) deferred to M46b.

/// Deterministic reduce_sum — uses sort-based reduction instead of atomicAdd.
/// Stub: delegates to existing nsl_tensor_sum_dim (CPU path is already deterministic).
/// Full sort-based GPU PTX in M46b.
///
/// NOTE: Signature matches nsl_tensor_sum_dim(tensor_ptr, dim, keepdim) = 3 params.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_sum_deterministic(input: i64, dim: i64, keepdim: i64) -> i64 {
    // Delegate to standard path — CPU reductions are already deterministic.
    // GPU sort-based reduction PTX deferred to M46b.
    crate::tensor::nsl_tensor_sum_dim(input, dim, keepdim)
}

/// Deterministic reduce_mean — sort-based reduction.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_mean_deterministic(input: i64, dim: i64, keepdim: i64) -> i64 {
    crate::tensor::nsl_tensor_mean_dim(input, dim, keepdim)
}

/// Deterministic scatter_add — sort indices then sequential accumulate.
/// SAFETY: Stub — returns 0 (null). Must not be called until M46b implements.
#[no_mangle]
pub extern "C" fn nsl_tensor_scatter_add_deterministic(
    _input: i64,
    _indices: i64,
    _src: i64,
) -> i64 {
    // TODO M46b: implement sort-indices-then-sequential-accumulate
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_scatter_stub_returns_zero() {
        // Stub — just verify it doesn't panic
        assert_eq!(nsl_tensor_scatter_add_deterministic(0, 0, 0), 0);
    }
}
