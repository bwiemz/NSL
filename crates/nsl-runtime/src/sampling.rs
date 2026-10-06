//! Sampling primitives for NSL: topk, multinomial, argmax, cumsum, lt_scalar, RNG.

use std::cell::RefCell;

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::autodiff;
use crate::cpu::{create_tensor_with_shape_rs, create_tensor_with_shape_rs_dtype, get_shape_vec, get_strides_vec};
use crate::dict::{nsl_dict_new, nsl_dict_set_str};
use crate::string::nsl_str_from_rust;
use crate::tensor::{nsl_tensor_free, nsl_tensor_to_device, NslTensor};

// ---------------------------------------------------------------------------
// GPU-input redirect
// ---------------------------------------------------------------------------

/// CPU redirect shared by the sampling FFIs below. Every one of them is a
/// host-side implementation, and a GPU tensor's `data` is a device pointer
/// (`cuda::inner::alloc_managed`) that must never be dereferenced on the
/// host. Stages the input on the CPU, re-enters the op there, and hands the
/// result back on the input's device — mirroring the non-dim-0 arm of
/// `nsl_tensor_gather`.
///
/// Index outputs (argmax, multinomial, topk indices) come back in the
/// input's dtype -- see [`index_output_in_operand_dtype`] -- so the upload
/// here is a byte copy, not a narrowing.
fn redirect_gpu_input_to_host(tensor_ptr: i64, op: impl FnOnce(i64) -> i64) -> i64 {
    let device = NslTensor::from_ptr(tensor_ptr).device;
    // Pause the tape across the CPU redirect (see nsl_tensor_stack).
    let _pause = autodiff::TapePause::new();
    let cpu_in = nsl_tensor_to_device(tensor_ptr, 0);
    let cpu_out = op(cpu_in);
    let dev_out = nsl_tensor_to_device(cpu_out, device as i64);
    nsl_tensor_free(cpu_in);
    nsl_tensor_free(cpu_out);
    dev_out
}

/// An index output in its operand's dtype: f64 only for an f64 input, f32
/// for anything else (C5: helpers mint their operand's dtype). The ops build
/// indices in f64 and convert once here; f32 is exact below 2^24, far above
/// any vocab. They used to return f64 whatever the input, which an upload to
/// the input's GPU then narrowed, and which an f64-refusing upload cannot.
fn index_output_in_operand_dtype(f64_ptr: i64, in_dtype: u16) -> i64 {
    if in_dtype == 0 {
        return f64_ptr;
    }
    let out = crate::tensor::precision_cast::convert_untaped(f64_ptr, 1);
    nsl_tensor_free(f64_ptr);
    out
}

// ---------------------------------------------------------------------------
// Thread-local RNG
// ---------------------------------------------------------------------------

// The training RNG. Spelled as the CONCRETE `ChaCha12Rng` rather than
// `rand::rngs::StdRng`, which in rand 0.9 *is* `ChaCha12Rng` — so the stream
// is unchanged bit-for-bit — because only the concrete type exposes
// `get_word_pos`/`set_word_pos`. Item 8 needs an O(1) snapshot/restore of
// this stream for checkpoint resume; replaying draws to catch up is
// O(elements dropped out so far), i.e. billions of draws at 50M scale.
thread_local! {
    static RNG: RefCell<ChaCha12Rng> = RefCell::new(ChaCha12Rng::seed_from_u64(0));
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_manual_seed(seed: i64) {
    RNG.with(|r| {
        *r.borrow_mut() = ChaCha12Rng::seed_from_u64(seed as u64);
    });
}

/// Generate a uniform random f64 in [0, 1) using the thread-local seeded RNG.
pub fn rng_f64() -> f64 {
    RNG.with(|r| r.borrow_mut().random::<f64>())
}

/// Capture this thread's RNG as (seed bytes, word position) — see
/// [`crate::rng_state::RngSnapshot`]. The seed alone is not the state: the
/// stream has advanced, and resuming from the seed would re-draw masks the
/// pre-crash run already used.
pub fn rng_snapshot() -> ([u8; 32], u128) {
    RNG.with(|r| {
        let rng = r.borrow();
        (rng.get_seed(), rng.get_word_pos())
    })
}

/// Restore this thread's RNG to a captured (seed, position).
pub fn rng_restore(seed: [u8; 32], pos: u128) {
    RNG.with(|r| {
        let mut rng = ChaCha12Rng::from_seed(seed);
        rng.set_word_pos(pos);
        *r.borrow_mut() = rng;
    });
}

// ---------------------------------------------------------------------------
// topk
// ---------------------------------------------------------------------------

/// Returns a dict with keys "values" and "indices" (both tensors).
/// `dim` supports negative indexing. Output shape = input shape with `dim`
/// replaced by `k`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_topk(tensor_ptr: i64, k: i64, dim: i64) -> i64 {
    // Host-side implementation: stage GPU inputs on the CPU and upload both
    // result tensors before the dict is built. The dict wraps two tensors,
    // so this keeps the staging inline instead of re-entering through
    // redirect_gpu_input_to_host and unpacking the CPU dict.
    let in_device = NslTensor::from_ptr(tensor_ptr).device;
    // Pause the tape across the CPU redirect (see nsl_tensor_stack).
    let _pause = (in_device != 0).then(autodiff::TapePause::new);
    let staged_ptr = if in_device != 0 {
        nsl_tensor_to_device(tensor_ptr, 0)
    } else {
        tensor_ptr
    };
    let tensor = NslTensor::from_ptr(staged_ptr);
    let shape = get_shape_vec(tensor);
    let strides = get_strides_vec(tensor);
    let ndim = shape.len();
    let in_dtype = tensor.dtype;

    // Helper: read tensor element as f64 regardless of dtype
    let read_val = |idx: usize| -> f64 {
        if in_dtype == 1 { unsafe { *tensor.data_f32().add(idx) as f64 } }
        else { unsafe { *tensor.data_f64().add(idx) } }
    };

    // Resolve negative dim
    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    assert!(d < ndim, "topk: dim {} out of range for ndim {}", dim, ndim);
    let dim_size = shape[d] as usize;
    let k = k as usize;
    assert!(k <= dim_size, "topk: k ({}) > dim size ({})", k, dim_size);

    // Build output shape
    let mut out_shape: Vec<i64> = shape.clone();
    out_shape[d] = k as i64;

    // values output matches input dtype; indices are built in f64 and take
    // the input's dtype at the end (`index_output_in_operand_dtype`)
    let values_ptr = create_tensor_with_shape_rs_dtype(&out_shape, in_dtype);
    let indices_ptr = create_tensor_with_shape_rs(&out_shape);
    let values_tensor = NslTensor::from_ptr(values_ptr);
    let indices_tensor = NslTensor::from_ptr(indices_ptr);
    let idx_data = indices_tensor.data_f64();
    let out_strides = get_strides_vec(values_tensor);

    let write_val = |out_idx: usize, val: f64| {
        if in_dtype == 1 {
            unsafe { *NslTensor::from_ptr(values_ptr).data_f32().add(out_idx) = val as f32 };
        } else {
            unsafe { *NslTensor::from_ptr(values_ptr).data_f64().add(out_idx) = val };
        }
    };

    // Number of slices = product of all dims except d
    let num_slices: usize = shape.iter().enumerate()
        .filter(|&(i, _)| i != d)
        .map(|(_, &s)| s as usize)
        .product();

    if ndim == 1 {
        // Simple 1D case
        let mut pairs: Vec<(f64, usize)> = (0..dim_size)
            .map(|i| (read_val(i), i))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (j, &(val, orig_idx)) in pairs.iter().enumerate().take(k) {
            write_val(j, val);
            unsafe { *idx_data.add(j) = orig_idx as f64; }
        }
    } else {
        // nD case: iterate over slices perpendicular to dim d
        let outer_dims: Vec<usize> = (0..ndim).filter(|&i| i != d).collect();
        let outer_sizes: Vec<usize> = outer_dims.iter().map(|&i| shape[i] as usize).collect();

        for slice_idx in 0..num_slices {
            let mut remaining = slice_idx;
            let mut outer_coords: Vec<usize> = vec![0; outer_dims.len()];
            for i in (0..outer_dims.len()).rev() {
                outer_coords[i] = remaining % outer_sizes[i];
                remaining /= outer_sizes[i];
            }

            let mut base_offset: usize = 0;
            for (oi, &od) in outer_dims.iter().enumerate() {
                base_offset += outer_coords[oi] * strides[od];
            }
            let dim_stride = strides[d];

            let mut pairs: Vec<(f64, usize)> = (0..dim_size)
                .map(|i| {
                    let offset = base_offset + i * dim_stride;
                    (read_val(offset), i)
                })
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut out_base: usize = 0;
            for (oi, &od) in outer_dims.iter().enumerate() {
                out_base += outer_coords[oi] * out_strides[od];
            }
            let out_dim_stride = out_strides[d];
            for (j, &(val, orig_idx)) in pairs.iter().enumerate().take(k) {
                let out_offset = out_base + j * out_dim_stride;
                write_val(out_offset, val);
                unsafe { *idx_data.add(out_offset) = orig_idx as f64; }
            }
        }
    }

    let indices_ptr = index_output_in_operand_dtype(indices_ptr, in_dtype);

    // Hand the results back on the input's device and drop the CPU staging
    // copies — the dict must hold the device-resident tensors.
    let (values_ptr, indices_ptr) = if in_device != 0 {
        let values_dev = nsl_tensor_to_device(values_ptr, in_device as i64);
        let indices_dev = nsl_tensor_to_device(indices_ptr, in_device as i64);
        nsl_tensor_free(values_ptr);
        nsl_tensor_free(indices_ptr);
        nsl_tensor_free(staged_ptr);
        (values_dev, indices_dev)
    } else {
        (values_ptr, indices_ptr)
    };

    // Return dict with "values" and "indices"
    let dict = nsl_dict_new();
    let key_values = nsl_str_from_rust("values");
    let key_indices = nsl_str_from_rust("indices");
    nsl_dict_set_str(dict, key_values, values_ptr);
    nsl_dict_set_str(dict, key_indices, indices_ptr);
    dict
}

// ---------------------------------------------------------------------------
// multinomial
// ---------------------------------------------------------------------------

/// Sample from a 1D or 2D probability tensor. Probabilities need not sum to 1.
/// Returns a tensor of sampled indices (as f64).
/// For 1D input of shape [n], returns shape [num_samples].
/// For 2D input of shape [batch, n], returns shape [batch, num_samples].
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_multinomial(tensor_ptr: i64, num_samples: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.device != 0 {
        return redirect_gpu_input_to_host(tensor_ptr, |cpu| {
            nsl_tensor_multinomial(cpu, num_samples)
        });
    }
    let shape = get_shape_vec(tensor);
    let in_dtype = tensor.dtype;
    let ndim = shape.len();

    let read_val = |idx: usize| -> f64 {
        if in_dtype == 1 { unsafe { *tensor.data_f32().add(idx) as f64 } }
        else { unsafe { *tensor.data_f64().add(idx) } }
    };

    assert!(ndim == 1 || ndim == 2, "multinomial: input must be 1D or 2D");
    let num_samples = num_samples as usize;

    let (batch_size, num_categories) = if ndim == 1 {
        (1_usize, shape[0] as usize)
    } else {
        (shape[0] as usize, shape[1] as usize)
    };

    let out_shape: Vec<i64> = if ndim == 1 {
        vec![num_samples as i64]
    } else {
        vec![batch_size as i64, num_samples as i64]
    };
    // indices are built in f64 and take the input's dtype at the end
    let result_ptr = create_tensor_with_shape_rs(&out_shape);
    let result_tensor = NslTensor::from_ptr(result_ptr);
    let result_data = result_tensor.data_f64();

    for b in 0..batch_size {
        let row_offset = b * num_categories;

        // Build CDF with negative clamping
        let mut cdf = Vec::with_capacity(num_categories);
        let mut running = 0.0_f64;
        for i in 0..num_categories {
            let val = read_val(row_offset + i);
            let clamped = if val < 0.0 { 0.0 } else { val };
            running += clamped;
            cdf.push(running);
        }
        let total = running;
        assert!(total > 0.0, "multinomial: sum of probabilities must be > 0");

        let out_row_offset = b * num_samples;
        RNG.with(|r| {
            let mut rng = r.borrow_mut();
            for s in 0..num_samples {
                let u: f64 = rng.random::<f64>() * total;
                // Binary search for first CDF entry >= u
                let mut lo = 0_usize;
                let mut hi = num_categories;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if cdf[mid] < u {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                // Clamp to valid range
                let idx = lo.min(num_categories - 1);
                unsafe {
                    *result_data.add(out_row_offset + s) = idx as f64;
                }
            }
        });
    }

    index_output_in_operand_dtype(result_ptr, in_dtype)
}

// ---------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------

/// Returns the index of the maximum value along dimension `dim`.
/// Output shape = input shape with dim d removed.
/// For 1D input, returns shape [1].
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_argmax(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.device != 0 {
        return redirect_gpu_input_to_host(tensor_ptr, |cpu| nsl_tensor_argmax(cpu, dim));
    }
    let shape = get_shape_vec(tensor);
    let strides = get_strides_vec(tensor);
    let ndim = shape.len();
    let in_dtype = tensor.dtype;

    let read_val = |idx: usize| -> f64 {
        if in_dtype == 1 { unsafe { *tensor.data_f32().add(idx) as f64 } }
        else { unsafe { *tensor.data_f64().add(idx) } }
    };

    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    assert!(d < ndim, "argmax: dim {} out of range for ndim {}", dim, ndim);
    let dim_size = shape[d] as usize;

    // Output shape: input shape with dim d removed (built in f64; the result
    // takes the input's dtype)
    let out_shape: Vec<i64> = if ndim == 1 {
        vec![1]
    } else {
        shape.iter().enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &s)| s)
            .collect()
    };

    let result_ptr = create_tensor_with_shape_rs(&out_shape);
    let result_tensor = NslTensor::from_ptr(result_ptr);
    let result_data = result_tensor.data_f64();

    if ndim == 1 {
        let mut best_idx = 0_usize;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..dim_size {
            let v = read_val(i);
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        unsafe { *result_data = best_idx as f64; }
    } else {
        let outer_dims: Vec<usize> = (0..ndim).filter(|&i| i != d).collect();
        let outer_sizes: Vec<usize> = outer_dims.iter().map(|&i| shape[i] as usize).collect();
        let num_slices: usize = outer_sizes.iter().product();

        for slice_idx in 0..num_slices {
            let mut remaining = slice_idx;
            let mut outer_coords: Vec<usize> = vec![0; outer_dims.len()];
            for i in (0..outer_dims.len()).rev() {
                outer_coords[i] = remaining % outer_sizes[i];
                remaining /= outer_sizes[i];
            }

            let mut base_offset: usize = 0;
            for (oi, &od) in outer_dims.iter().enumerate() {
                base_offset += outer_coords[oi] * strides[od];
            }
            let dim_stride = strides[d];

            let mut best_idx = 0_usize;
            let mut best_val = f64::NEG_INFINITY;
            for i in 0..dim_size {
                let v = read_val(base_offset + i * dim_stride);
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            unsafe { *result_data.add(slice_idx) = best_idx as f64; }
        }
    }

    index_output_in_operand_dtype(result_ptr, in_dtype)
}

// ---------------------------------------------------------------------------
// cumsum
// ---------------------------------------------------------------------------

/// Cumulative sum along dimension `dim`. Output same shape as input.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_cumsum(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.device != 0 {
        return redirect_gpu_input_to_host(tensor_ptr, |cpu| nsl_tensor_cumsum(cpu, dim));
    }
    let shape = get_shape_vec(tensor);
    let strides = get_strides_vec(tensor);
    let ndim = shape.len();
    let in_dtype = tensor.dtype;

    let read_val = |idx: usize| -> f64 {
        if in_dtype == 1 { unsafe { *tensor.data_f32().add(idx) as f64 } }
        else { unsafe { *tensor.data_f64().add(idx) } }
    };

    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    assert!(d < ndim, "cumsum: dim {} out of range for ndim {}", dim, ndim);
    let dim_size = shape[d] as usize;

    let result_ptr = create_tensor_with_shape_rs_dtype(&shape, in_dtype);
    let result_tensor = NslTensor::from_ptr(result_ptr);
    let out_strides = get_strides_vec(result_tensor);

    let write_val = |idx: usize, val: f64| {
        if in_dtype == 1 {
            unsafe { *NslTensor::from_ptr(result_ptr).data_f32().add(idx) = val as f32 };
        } else {
            unsafe { *NslTensor::from_ptr(result_ptr).data_f64().add(idx) = val };
        }
    };

    if ndim == 1 {
        let mut running = 0.0_f64;
        for i in 0..dim_size {
            running += read_val(i);
            write_val(i, running);
        }
    } else {
        let outer_dims: Vec<usize> = (0..ndim).filter(|&i| i != d).collect();
        let outer_sizes: Vec<usize> = outer_dims.iter().map(|&i| shape[i] as usize).collect();
        let num_slices: usize = outer_sizes.iter().product();

        for slice_idx in 0..num_slices {
            let mut remaining = slice_idx;
            let mut outer_coords: Vec<usize> = vec![0; outer_dims.len()];
            for i in (0..outer_dims.len()).rev() {
                outer_coords[i] = remaining % outer_sizes[i];
                remaining /= outer_sizes[i];
            }

            let mut in_base: usize = 0;
            let mut out_base: usize = 0;
            for (oi, &od) in outer_dims.iter().enumerate() {
                in_base += outer_coords[oi] * strides[od];
                out_base += outer_coords[oi] * out_strides[od];
            }
            let in_stride = strides[d];
            let out_stride = out_strides[d];

            let mut running = 0.0_f64;
            for i in 0..dim_size {
                running += read_val(in_base + i * in_stride);
                write_val(out_base + i * out_stride, running);
            }
        }
    }

    result_ptr
}

// ---------------------------------------------------------------------------
// lt_scalar
// ---------------------------------------------------------------------------

/// Element-wise `< scalar` comparison. Returns 1.0 where true, 0.0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_lt_scalar(tensor_ptr: i64, scalar: f64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.device != 0 {
        // The CPU f32 arm below compares in f32 against the ROUNDED
        // threshold (`scalar as f32`). When the staging download promoted the
        // data f32→f64 (before C5 step 2a it did) the f64 arm compared against
        // the UNROUNDED threshold, and at an f32-representability boundary the
        // masks differed (measured: `lt_scalar(x, 0.9)` on x == 0.9f32 flipped
        // per device, review M1 on 1070c53b). The download now keeps f32;
        // pre-rounding the threshold keeps the two arms identical regardless.
        let s = if tensor.dtype == 1 {
            (scalar as f32) as f64
        } else {
            scalar
        };
        return redirect_gpu_input_to_host(tensor_ptr, |cpu| nsl_tensor_lt_scalar(cpu, s));
    }
    let shape = get_shape_vec(tensor);
    let in_dtype = tensor.dtype;
    let len = tensor.len as usize;

    // Output is a boolean mask — same dtype as input
    let result_ptr = create_tensor_with_shape_rs_dtype(&shape, in_dtype);
    let result_tensor = NslTensor::from_ptr(result_ptr);

    if in_dtype == 1 {
        let data = tensor.data_f32();
        let result_data = result_tensor.data_f32();
        let s_f32 = scalar as f32;
        for i in 0..len {
            unsafe {
                let v = *data.add(i);
                *result_data.add(i) = if v < s_f32 { 1.0_f32 } else { 0.0_f32 };
            }
        }
    } else {
        let data = tensor.data_f64();
        let result_data = result_tensor.data_f64();
        for i in 0..len {
            unsafe {
                let v = *data.add(i);
                *result_data.add(i) = if v < scalar { 1.0 } else { 0.0 };
            }
        }
    }

    result_ptr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::create_tensor_with_shape_rs;
    use crate::dict::nsl_dict_get_str;

    /// Helper: create a 1D tensor from a slice of f64.
    fn make_1d_tensor(values: &[f64]) -> i64 {
        let shape = [values.len() as i64];
        let ptr = create_tensor_with_shape_rs(&shape);
        let t = NslTensor::from_ptr(ptr);
        let data = t.data_f64();
        for (i, &v) in values.iter().enumerate() {
            unsafe { *data.add(i) = v; }
        }
        ptr
    }

    /// Helper: read 1D tensor data as Vec<f64>.
    fn read_1d(ptr: i64) -> Vec<f64> {
        let t = NslTensor::from_ptr(ptr);
        let data = t.data_f64();
        let len = t.len as usize;
        (0..len).map(|i| unsafe { *data.add(i) }).collect()
    }

    /// Index outputs take their operand's dtype (C5): f32 logits give f32
    /// indices -- which the GPU path can upload as a byte copy -- and f64
    /// logits keep f64. They used to be f64 whatever the input.
    #[test]
    fn test_index_outputs_take_the_operand_dtype() {
        let f32_logits = crate::tensor::precision_cast::convert_untaped(
            make_1d_tensor(&[0.5, 2.5, -1.0, 2.0]),
            1,
        );
        let f64_logits = make_1d_tensor(&[0.5, 2.5, -1.0, 2.0]);
        let tag = |p: i64| NslTensor::from_ptr(p).dtype;
        let key_i = nsl_str_from_rust("indices");

        let am32 = nsl_tensor_argmax(f32_logits, 0);
        assert_eq!(tag(am32), 1);
        assert_eq!(unsafe { *NslTensor::from_ptr(am32).data_f32() }, 1.0);
        assert_eq!(tag(nsl_tensor_argmax(f64_logits, 0)), 0);

        let tk32 = nsl_dict_get_str(nsl_tensor_topk(f32_logits, 2, 0), key_i);
        assert_eq!(tag(tk32), 1);
        let t = NslTensor::from_ptr(tk32);
        let got: Vec<f32> = (0..2).map(|i| unsafe { *t.data_f32().add(i) }).collect();
        assert_eq!(got, vec![1.0, 3.0]);
        assert_eq!(tag(nsl_dict_get_str(nsl_tensor_topk(f64_logits, 2, 0), key_i)), 0);

        let probs32 = crate::tensor::precision_cast::convert_untaped(
            make_1d_tensor(&[0.0, 1.0, 0.0]),
            1,
        );
        let mn32 = nsl_tensor_multinomial(probs32, 3);
        assert_eq!(tag(mn32), 1);
        let m = NslTensor::from_ptr(mn32);
        assert!((0..3).all(|i| unsafe { *m.data_f32().add(i) } == 1.0));
        assert_eq!(tag(nsl_tensor_multinomial(make_1d_tensor(&[0.0, 1.0]), 1)), 0);
    }

    #[test]
    fn test_topk_basic() {
        let t = make_1d_tensor(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let dict = nsl_tensor_topk(t, 3, 0);

        let key_v = nsl_str_from_rust("values");
        let key_i = nsl_str_from_rust("indices");
        let values_ptr = nsl_dict_get_str(dict, key_v);
        let indices_ptr = nsl_dict_get_str(dict, key_i);

        let values = read_1d(values_ptr);
        let indices = read_1d(indices_ptr);

        assert_eq!(values, vec![9.0, 6.0, 5.0]);
        assert_eq!(indices, vec![5.0, 7.0, 4.0]);
    }

    #[test]
    fn test_multinomial_deterministic() {
        let probs = make_1d_tensor(&[0.1, 0.2, 0.3, 0.4]);

        nsl_manual_seed(42);
        let r1 = read_1d(nsl_tensor_multinomial(probs, 5));

        nsl_manual_seed(42);
        let r2 = read_1d(nsl_tensor_multinomial(probs, 5));

        assert_eq!(r1, r2);
    }

    #[test]
    fn test_multinomial_unnormalized() {
        // Probabilities that don't sum to 1 — should not panic
        let probs = make_1d_tensor(&[10.0, 20.0, 30.0]);
        nsl_manual_seed(123);
        let result = read_1d(nsl_tensor_multinomial(probs, 4));
        assert_eq!(result.len(), 4);
        for &idx in &result {
            assert!((0.0..3.0).contains(&idx));
        }
    }

    #[test]
    fn test_argmax() {
        let t = make_1d_tensor(&[1.0, 5.0, 3.0, 2.0]);
        let result = nsl_tensor_argmax(t, 0);
        let data = read_1d(result);
        assert_eq!(data, vec![1.0]);
    }

    #[test]
    fn test_cumsum() {
        let t = make_1d_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let result = nsl_tensor_cumsum(t, 0);
        let data = read_1d(result);
        assert_eq!(data, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_lt_scalar() {
        let t = make_1d_tensor(&[0.1, 0.5, 0.9, 0.3]);
        let result = nsl_tensor_lt_scalar(t, 0.5);
        let data = read_1d(result);
        assert_eq!(data, vec![1.0, 0.0, 0.0, 1.0]);
    }
}
