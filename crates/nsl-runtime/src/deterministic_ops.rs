//! M46/M46b: Deterministic runtime operation variants.
//!
//! These are called when --deterministic is active, replacing the default
//! non-deterministic GPU kernels with bit-reproducible alternatives.
//!
//! GPU deterministic kernels (M46b):
//! - **Sum/Mean**: Sequential single-thread PTX kernels (`nsl_det_global_sum_f32`,
//!   `nsl_det_sum_dim_f32`) that accumulate in fixed ascending order.
//! - **Scatter_add**: CPU fallback (sort indices, sequential accumulate).
//!   GPU-native sort-based PTX deferred to M46c.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Global deterministic mode flag — set at program start when --deterministic is active.
static DETERMINISTIC_MODE: AtomicBool = AtomicBool::new(false);

/// Global RNG seed — set when --deterministic is active to ensure reproducible RNG.
static RNG_SEED: AtomicU64 = AtomicU64::new(42);

/// Set global deterministic mode flag.
/// Called from compiled main() when --deterministic is active.
#[no_mangle]
pub extern "C" fn nsl_set_deterministic(mode: i64) -> i64 {
    DETERMINISTIC_MODE.store(mode != 0, Ordering::SeqCst);
    if mode != 0 {
        eprintln!("[nsl] deterministic mode enabled");
    }
    0
}

/// Returns true if deterministic mode is currently active.
pub fn is_deterministic() -> bool {
    DETERMINISTIC_MODE.load(Ordering::Relaxed)
}

/// Seed all RNG sources for reproducibility.
/// Called from compiled main() when --deterministic is active.
#[no_mangle]
pub extern "C" fn nsl_rng_seed(seed: i64) -> i64 {
    RNG_SEED.store(seed as u64, Ordering::SeqCst);
    eprintln!("[nsl] deterministic RNG seed set to {seed}");
    0
}

/// Retrieve the current RNG seed (used by stochastic ops to thread the seed).
pub fn get_rng_seed() -> u64 {
    RNG_SEED.load(Ordering::Relaxed)
}

/// Deterministic reduce_sum — uses sequential single-thread GPU kernels for GPU tensors,
/// or delegates to CPU path (already deterministic) for CPU tensors.
///
/// M46b: GPU tensors use `nsl_det_global_sum_f32` / `nsl_det_sum_dim_f32` PTX kernels
/// that accumulate in fixed ascending order (no parallelism = bit-reproducible).
///
/// NOTE: Signature matches nsl_tensor_sum_dim(tensor_ptr, dim, keepdim) = 3 params.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_sum_deterministic(input: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = crate::tensor::NslTensor::from_ptr(input);

    // GPU path: use deterministic sequential kernels
    if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let keepdim_bool = keepdim != 0;
            let c_ptr = crate::tensor::nsl_tensor_contiguous(input);
            let result = if dim == -1 {
                // Global deterministic sum: single-thread sequential accumulation
                crate::cuda::gpu_det_global_sum_f32(c_ptr)
            } else {
                let ndim = tensor.ndim as usize;
                let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
                // Per-dim deterministic sum: one thread per output, sequential inner loop
                crate::cuda::gpu_det_sum_dim_f32(c_ptr, d, keepdim_bool)
            };
            if c_ptr != input { crate::tensor::nsl_tensor_free(c_ptr); }
            return result;
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Fallback: transfer to CPU, compute deterministically, transfer back
            let cpu_t = crate::tensor::nsl_tensor_to_device(input, 0);
            let result = crate::tensor::nsl_tensor_sum_dim(cpu_t, dim, keepdim);
            let gpu_result = crate::tensor::nsl_tensor_to_device(result, tensor.device as i64);
            crate::tensor::nsl_tensor_free(cpu_t);
            crate::tensor::nsl_tensor_free(result);
            return gpu_result;
        }
    }

    // CPU path: already deterministic (sequential loop)
    crate::tensor::nsl_tensor_sum_dim(input, dim, keepdim)
}

/// Deterministic reduce_mean — uses deterministic sum + divide for GPU tensors,
/// or delegates to CPU path (already deterministic) for CPU tensors.
///
/// M46b: GPU tensors use the deterministic sum kernel then divide by element count.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_mean_deterministic(input: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = crate::tensor::NslTensor::from_ptr(input);

    // GPU path: deterministic sum then scalar divide
    if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let keepdim_bool = keepdim != 0;
            let c_ptr = crate::tensor::nsl_tensor_contiguous(input);

            if dim == -1 {
                // Global deterministic mean: det_sum / total_elements
                let num_elements = tensor.len;
                let sum_ptr = crate::cuda::gpu_det_global_sum_f32(c_ptr);
                if c_ptr != input { crate::tensor::nsl_tensor_free(c_ptr); }
                let inv = 1.0_f32 / num_elements as f32;
                crate::cuda::gpu_scalar_op_inplace(
                    sum_ptr, inv,
                    crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0",
                );
                return sum_ptr;
            }

            let ndim = tensor.ndim as usize;
            let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
            let shape_slice = unsafe { std::slice::from_raw_parts(tensor.shape, ndim) };
            let dim_size = shape_slice[d];
            let sum_ptr = crate::cuda::gpu_det_sum_dim_f32(c_ptr, d, keepdim_bool);
            if c_ptr != input { crate::tensor::nsl_tensor_free(c_ptr); }
            let inv = 1.0_f32 / dim_size as f32;
            crate::cuda::gpu_scalar_op_inplace(
                sum_ptr, inv,
                crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0",
            );
            return sum_ptr;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let cpu_t = crate::tensor::nsl_tensor_to_device(input, 0);
            let result = crate::tensor::nsl_tensor_mean_dim(cpu_t, dim, keepdim);
            let gpu_result = crate::tensor::nsl_tensor_to_device(result, tensor.device as i64);
            crate::tensor::nsl_tensor_free(cpu_t);
            crate::tensor::nsl_tensor_free(result);
            return gpu_result;
        }
    }

    // CPU path: already deterministic (sequential loop)
    crate::tensor::nsl_tensor_mean_dim(input, dim, keepdim)
}

/// Deterministic scatter_add — sort indices then sequential accumulate.
///
/// CPU: clone input, build sorted (index, value) pairs, accumulate in sorted order
/// (deterministic regardless of thread scheduling).
///
/// GPU (M46b): transfer to CPU, run sort-based scatter_add, transfer back.
/// A full GPU-native sort-based PTX kernel (bitonic sort + sequential accumulate)
/// is deferred to M46c — the CPU fallback is correct and sufficient for now.
#[no_mangle]
pub extern "C" fn nsl_tensor_scatter_add_deterministic(
    input: i64,
    indices: i64,
    src: i64,
) -> i64 {
    if input == 0 || indices == 0 || src == 0 {
        return 0;
    }

    let input_tensor = crate::tensor::NslTensor::from_ptr(input);

    // GPU path: transfer to CPU, run deterministic scatter_add, transfer back
    if input_tensor.device > 0 {
        let device = input_tensor.device as i64;
        let cpu_input = crate::tensor::nsl_tensor_to_device(input, 0);
        let cpu_indices = crate::tensor::nsl_tensor_to_device(indices, 0);
        let cpu_src = crate::tensor::nsl_tensor_to_device(src, 0);

        // Run the CPU sort-based scatter_add
        let cpu_result = nsl_tensor_scatter_add_deterministic_cpu(cpu_input, cpu_indices, cpu_src);

        // Transfer result back to GPU
        let gpu_result = crate::tensor::nsl_tensor_to_device(cpu_result, device);

        // Cleanup CPU temporaries
        crate::tensor::nsl_tensor_free(cpu_input);
        crate::tensor::nsl_tensor_free(cpu_indices);
        crate::tensor::nsl_tensor_free(cpu_src);
        crate::tensor::nsl_tensor_free(cpu_result);

        return gpu_result;
    }

    // CPU path: sort-based deterministic scatter_add
    nsl_tensor_scatter_add_deterministic_cpu(input, indices, src)
}

/// CPU implementation of sort-based deterministic scatter_add.
fn nsl_tensor_scatter_add_deterministic_cpu(
    input: i64,
    indices: i64,
    src: i64,
) -> i64 {
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

    #[test]
    fn deterministic_mode_flag() {
        // Default off
        assert!(!is_deterministic());
        // Enable
        nsl_set_deterministic(1);
        assert!(is_deterministic());
        // Disable
        nsl_set_deterministic(0);
        assert!(!is_deterministic());
    }

    #[test]
    fn rng_seed_roundtrip() {
        nsl_rng_seed(123);
        assert_eq!(get_rng_seed(), 123);
        nsl_rng_seed(42);
        assert_eq!(get_rng_seed(), 42);
    }

    #[test]
    fn cpu_deterministic_sum_matches_standard() {
        // CPU path should produce identical results for deterministic and standard sum
        use crate::tensor::{nsl_tensor_sum_dim, NslTensor};

        // Create a small CPU tensor [1.0, 2.0, 3.0, 4.0]
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let shape = vec![4_i64];
        let strides = vec![1_i64];
        let t = Box::new(NslTensor::new(
            data.as_ptr() as *mut std::ffi::c_void,
            shape.as_ptr() as *mut i64,
            strides.as_ptr() as *mut i64,
            1,
            4,
            0, // CPU
            0, // f64
            1,
            0,
        ));
        std::mem::forget(data);
        std::mem::forget(shape);
        std::mem::forget(strides);
        let ptr = Box::into_raw(t) as i64;

        let std_result = nsl_tensor_sum_dim(ptr, -1, 0);
        let det_result = nsl_tensor_reduce_sum_deterministic(ptr, -1, 0);

        let std_t = NslTensor::from_ptr(std_result);
        let det_t = NslTensor::from_ptr(det_result);

        let std_val = unsafe { *std_t.data_f64() };
        let det_val = unsafe { *det_t.data_f64() };

        assert!((std_val - det_val).abs() < 1e-10,
            "CPU deterministic sum ({}) should match standard sum ({})", det_val, std_val);
    }
}
