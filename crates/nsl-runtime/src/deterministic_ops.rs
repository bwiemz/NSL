//! M46/M46b/M46c: Deterministic runtime operation variants.
//!
//! These are called when --deterministic is active, replacing the default
//! non-deterministic GPU kernels with bit-reproducible alternatives.
//!
//! GPU deterministic kernels (M46b):
//! - **Sum/Mean**: Sequential single-thread PTX kernels (`nsl_det_global_sum_f32`,
//!   `nsl_det_sum_dim_f32`) that accumulate in fixed ascending order.
//!
//! GPU deterministic kernels (M46c):
//! - **Scatter_add**: Output-centric GPU kernel (`nsl_det_scatter_add_f32`).
//!   One thread per (output_row, col) pair; each thread scans ALL input indices
//!   sequentially, accumulating matching values. No atomics, no sorting —
//!   deterministic because each output element is owned by exactly one thread.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Global deterministic mode flag — set at program start when --deterministic is active.
static DETERMINISTIC_MODE: AtomicBool = AtomicBool::new(false);

/// Value of [`RNG_SEED`] when no `--seed` was given. Consumers that need to
/// distinguish "seeded run" from "default" (e.g. the DataLoader's shuffle key)
/// compare against this rather than hard-coding 42 in a second place.
pub const RNG_SEED_DEFAULT: u64 = 42;

/// Global RNG seed — set when --deterministic is active to ensure reproducible RNG.
static RNG_SEED: AtomicU64 = AtomicU64::new(RNG_SEED_DEFAULT);

/// Whether [`nsl_rng_seed`] actually ran. Consumers that only want to change
/// behavior for an EXPLICITLY seeded run must not infer that from the seed
/// VALUE: `--seed 42` is indistinguishable from the default by value alone,
/// so `--seed 42` would silently behave differently from `--seed 43`.
static RNG_SEED_SET: AtomicBool = AtomicBool::new(false);

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
/// Called from compiled main() when --deterministic and/or --seed is active.
#[no_mangle]
pub extern "C" fn nsl_rng_seed(seed: i64) -> i64 {
    RNG_SEED.store(seed as u64, Ordering::SeqCst);
    RNG_SEED_SET.store(true, Ordering::SeqCst);
    // P0 certification (--seed): the SAMPLING thread-local RNG feeds
    // randn/rand model init, but it was only reseedable via
    // nsl_manual_seed — "seed all RNG sources" silently excluded the one
    // that determines the initial weights, so every --seed produced the
    // SAME init. Reseed it here (program start runs on the thread that
    // executes model init).
    crate::sampling::nsl_manual_seed(seed);
    // Same omission on the GPU side: `gpu_dropout_f32`'s counter started at a
    // hard-coded 42 regardless of --seed, so GPU dropout masks were identical
    // across seeds AND unreachable by "seed all RNG sources".
    //
    // MIXED, not offset. A bare `seed + 42` would land `--seed 0` exactly on
    // the unseeded default, so the flag would provably do nothing for that
    // one value — the same "indistinguishable from the default by value
    // alone" trap RNG_SEED_SET exists for a few lines above.
    //
    // Adjacent seeds are NOT a correlation hazard either way: the kernel's
    // per-element dice is a bijective avalanche mix of `base + idx`
    // (cuda/fused_kernels.rs), whose defining property is that neighbouring
    // counters decorrelate — measured at 0.8204 same-position mask agreement
    // for p=0.1 across offsets 1..4096, against an independent baseline of
    // (1-p)^2 + p^2 = 0.8200. Mixing is for the --seed 0 collision, not to
    // repair a correlation that was never there.
    crate::rng_state::set_gpu_dropout_counter(crate::sr_bf16::sr_mix64(
        seed as u64,
        crate::rng_state::GPU_DROPOUT_SEED_DEFAULT,
    ));
    eprintln!("[nsl] deterministic RNG seed set to {seed}");
    0
}

/// Retrieve the current RNG seed (used by stochastic ops to thread the seed).
pub fn get_rng_seed() -> u64 {
    RNG_SEED.load(Ordering::Relaxed)
}

/// The seed only if the program explicitly set one — see [`RNG_SEED_SET`].
pub fn explicit_rng_seed() -> Option<u64> {
    RNG_SEED_SET
        .load(Ordering::Relaxed)
        .then(|| RNG_SEED.load(Ordering::Relaxed))
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

    // GPU path: use deterministic output-centric kernel (M46c) — no atomics, no CPU fallback
    if input_tensor.device > 0 {
        #[cfg(feature = "cuda")]
        {
            return crate::cuda::gpu_det_scatter_add_f32(input, indices, src);
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Non-CUDA builds: transfer to CPU, run deterministic scatter_add, transfer back
            let device = input_tensor.device as i64;
            let cpu_input = crate::tensor::nsl_tensor_to_device(input, 0);
            let cpu_indices = crate::tensor::nsl_tensor_to_device(indices, 0);
            let cpu_src = crate::tensor::nsl_tensor_to_device(src, 0);

            let cpu_result = nsl_tensor_scatter_add_deterministic_cpu(cpu_input, cpu_indices, cpu_src);
            let gpu_result = crate::tensor::nsl_tensor_to_device(cpu_result, device);

            crate::tensor::nsl_tensor_free(cpu_input);
            crate::tensor::nsl_tensor_free(cpu_indices);
            crate::tensor::nsl_tensor_free(cpu_src);
            crate::tensor::nsl_tensor_free(cpu_result);

            return gpu_result;
        }
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
        // Item 8: nsl_rng_seed now also writes the process-global GPU dropout
        // counter, which rng_state's tests assert absolute values on. Share
        // their lock so cargo's parallel threads cannot interleave.
        let _guard = crate::rng_state::GLOBAL_RNG_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        nsl_rng_seed(123);
        assert_eq!(get_rng_seed(), 123);
        nsl_rng_seed(42);
        assert_eq!(get_rng_seed(), 42);
        // Distinct seeds must give distinct GPU dropout streams, not offsets
        // into one tape: `hash32(base + idx)` over a base advanced by the
        // launch length means a `seed + K` base would make seed s+1's masks a
        // one-element shift of seed s's.
        nsl_rng_seed(1);
        let c1 = crate::rng_state::gpu_dropout_counter();
        nsl_rng_seed(2);
        let c2 = crate::rng_state::gpu_dropout_counter();
        assert!(
            c1 != c2,
            "adjacent seeds must give different GPU dropout bases ({c1})"
        );
        nsl_rng_seed(0);
        assert_ne!(
            crate::rng_state::gpu_dropout_counter(),
            crate::rng_state::GPU_DROPOUT_SEED_DEFAULT,
            "--seed 0 must not land on the unseeded default, or the flag \
             provably does nothing for that value"
        );
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
