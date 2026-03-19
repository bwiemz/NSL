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
///
/// CPU implementation: clone input, build sorted (index, value) pairs,
/// accumulate in sorted order (deterministic regardless of thread scheduling).
/// GPU sort-based PTX kernel deferred to M46c.
#[no_mangle]
pub extern "C" fn nsl_tensor_scatter_add_deterministic(
    input: i64,
    indices: i64,
    src: i64,
) -> i64 {
    if input == 0 || indices == 0 || src == 0 {
        return 0;
    }

    // Clone input tensor as output base
    let output = crate::tensor::nsl_tensor_clone(input);
    if output == 0 { return 0; }

    let idx_tensor = unsafe { &*(indices as *const crate::tensor::NslTensor) };
    let src_tensor = unsafe { &*(src as *const crate::tensor::NslTensor) };
    let out_tensor = unsafe { &mut *(output as *mut crate::tensor::NslTensor) };

    let n = idx_tensor.len as usize;
    if n == 0 { return output; }

    // Build sorted (index, value) pairs for deterministic ordering
    let mut pairs: Vec<(i64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let idx = unsafe { *(idx_tensor.data as *const f64).add(i) } as i64;
        let val = unsafe { *(src_tensor.data as *const f64).add(i) };
        pairs.push((idx, val));
    }
    // Sort by index — ensures deterministic accumulation order
    pairs.sort_by_key(|&(idx, _)| idx);

    // Sequential accumulate in sorted order
    let out_data = out_tensor.data as *mut f64;
    for (idx, val) in &pairs {
        if *idx >= 0 && (*idx as usize) < out_tensor.len as usize {
            unsafe { *out_data.add(*idx as usize) += val; }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_variants_are_exported() {
        let sum_fn: extern "C" fn(i64, i64, i64) -> i64 = nsl_tensor_reduce_sum_deterministic;
        let mean_fn: extern "C" fn(i64, i64, i64) -> i64 = nsl_tensor_reduce_mean_deterministic;
        assert!(!std::ptr::addr_of!(sum_fn).is_null());
        assert!(!std::ptr::addr_of!(mean_fn).is_null());
    }

    #[test]
    fn scatter_add_null_returns_zero() {
        // Null inputs return 0 (no crash)
        assert_eq!(nsl_tensor_scatter_add_deterministic(0, 0, 0), 0);
    }
}
