//! Sampling primitives for NSL: topk, multinomial, argmax, cumsum, lt_scalar, RNG.

use std::cell::RefCell;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::cpu::{create_tensor_with_shape_rs, get_shape_vec, get_strides_vec};
use crate::dict::{nsl_dict_new, nsl_dict_set_str};
use crate::string::nsl_str_from_rust;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// Thread-local RNG
// ---------------------------------------------------------------------------

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}

#[no_mangle]
pub extern "C" fn nsl_manual_seed(seed: i64) {
    RNG.with(|r| {
        *r.borrow_mut() = StdRng::seed_from_u64(seed as u64);
    });
}

/// Generate a uniform random f64 in [0, 1) using the thread-local seeded RNG.
pub fn rng_f64() -> f64 {
    RNG.with(|r| r.borrow_mut().gen::<f64>())
}

// ---------------------------------------------------------------------------
// topk
// ---------------------------------------------------------------------------

/// Returns a dict with keys "values" and "indices" (both tensors).
/// `dim` supports negative indexing. Output shape = input shape with `dim`
/// replaced by `k`.
#[no_mangle]
pub extern "C" fn nsl_tensor_topk(tensor_ptr: i64, k: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = get_shape_vec(tensor);
    let strides = get_strides_vec(tensor);
    let ndim = shape.len();
    let data = tensor.data_f64();

    // Resolve negative dim
    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    assert!(d < ndim, "topk: dim {} out of range for ndim {}", dim, ndim);
    let dim_size = shape[d] as usize;
    let k = k as usize;
    assert!(k <= dim_size, "topk: k ({}) > dim size ({})", k, dim_size);

    // Build output shape
    let mut out_shape: Vec<i64> = shape.clone();
    out_shape[d] = k as i64;

    let values_ptr = create_tensor_with_shape_rs(&out_shape);
    let indices_ptr = create_tensor_with_shape_rs(&out_shape);
    let values_tensor = NslTensor::from_ptr(values_ptr);
    let indices_tensor = NslTensor::from_ptr(indices_ptr);
    let val_data = values_tensor.data_f64();
    let idx_data = indices_tensor.data_f64();
    let out_strides = get_strides_vec(values_tensor);

    // Number of slices = product of all dims except d
    let num_slices: usize = shape.iter().enumerate()
        .filter(|&(i, _)| i != d)
        .map(|(_, &s)| s as usize)
        .product();

    if ndim == 1 {
        // Simple 1D case
        let mut pairs: Vec<(f64, usize)> = (0..dim_size)
            .map(|i| (unsafe { *data.add(i) }, i))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for j in 0..k {
            unsafe {
                *val_data.add(j) = pairs[j].0;
                *idx_data.add(j) = pairs[j].1 as f64;
            }
        }
    } else {
        // nD case: iterate over slices perpendicular to dim d
        // Build "outer" coordinate iteration (all dims except d)
        let outer_dims: Vec<usize> = (0..ndim).filter(|&i| i != d).collect();
        let outer_sizes: Vec<usize> = outer_dims.iter().map(|&i| shape[i] as usize).collect();

        for slice_idx in 0..num_slices {
            // Decompose slice_idx into outer coordinates
            let mut remaining = slice_idx;
            let mut outer_coords: Vec<usize> = vec![0; outer_dims.len()];
            for i in (0..outer_dims.len()).rev() {
                outer_coords[i] = remaining % outer_sizes[i];
                remaining /= outer_sizes[i];
            }

            // Compute base offset in input (with dim d = 0)
            let mut base_offset: usize = 0;
            for (oi, &od) in outer_dims.iter().enumerate() {
                base_offset += outer_coords[oi] * strides[od];
            }
            let dim_stride = strides[d];

            // Collect (value, index) pairs along dim d
            let mut pairs: Vec<(f64, usize)> = (0..dim_size)
                .map(|i| {
                    let offset = base_offset + i * dim_stride;
                    (unsafe { *data.add(offset) }, i)
                })
                .collect();
            pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            // Write to output
            let mut out_base: usize = 0;
            for (oi, &od) in outer_dims.iter().enumerate() {
                out_base += outer_coords[oi] * out_strides[od];
            }
            let out_dim_stride = out_strides[d];
            for j in 0..k {
                let out_offset = out_base + j * out_dim_stride;
                unsafe {
                    *val_data.add(out_offset) = pairs[j].0;
                    *idx_data.add(out_offset) = pairs[j].1 as f64;
                }
            }
        }
    }

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
#[no_mangle]
pub extern "C" fn nsl_tensor_multinomial(tensor_ptr: i64, num_samples: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = get_shape_vec(tensor);
    let data = tensor.data_f64();
    let ndim = shape.len();

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
    let result_ptr = create_tensor_with_shape_rs(&out_shape);
    let result_tensor = NslTensor::from_ptr(result_ptr);
    let result_data = result_tensor.data_f64();

    for b in 0..batch_size {
        let row_offset = b * num_categories;

        // Build CDF with negative clamping
        let mut cdf = Vec::with_capacity(num_categories);
        let mut running = 0.0_f64;
        for i in 0..num_categories {
            let val = unsafe { *data.add(row_offset + i) };
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
                let u: f64 = rng.gen::<f64>() * total;
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

    result_ptr
}

// ---------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------

/// Returns the index of the maximum value along dimension `dim`.
/// Output shape = input shape with dim d removed.
/// For 1D input, returns shape [1].
#[no_mangle]
pub extern "C" fn nsl_tensor_argmax(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = get_shape_vec(tensor);
    let strides = get_strides_vec(tensor);
    let ndim = shape.len();
    let data = tensor.data_f64();

    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    assert!(d < ndim, "argmax: dim {} out of range for ndim {}", dim, ndim);
    let dim_size = shape[d] as usize;

    // Output shape: input shape with dim d removed
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
            let v = unsafe { *data.add(i) };
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
                let v = unsafe { *data.add(base_offset + i * dim_stride) };
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            unsafe { *result_data.add(slice_idx) = best_idx as f64; }
        }
    }

    result_ptr
}

// ---------------------------------------------------------------------------
// cumsum
// ---------------------------------------------------------------------------

/// Cumulative sum along dimension `dim`. Output same shape as input.
#[no_mangle]
pub extern "C" fn nsl_tensor_cumsum(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = get_shape_vec(tensor);
    let strides = get_strides_vec(tensor);
    let ndim = shape.len();
    let data = tensor.data_f64();

    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    assert!(d < ndim, "cumsum: dim {} out of range for ndim {}", dim, ndim);
    let dim_size = shape[d] as usize;

    let result_ptr = create_tensor_with_shape_rs(&shape);
    let result_tensor = NslTensor::from_ptr(result_ptr);
    let result_data = result_tensor.data_f64();
    let out_strides = get_strides_vec(result_tensor);

    if ndim == 1 {
        let mut running = 0.0_f64;
        for i in 0..dim_size {
            running += unsafe { *data.add(i) };
            unsafe { *result_data.add(i) = running; }
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
                running += unsafe { *data.add(in_base + i * in_stride) };
                unsafe { *result_data.add(out_base + i * out_stride) = running; }
            }
        }
    }

    result_ptr
}

// ---------------------------------------------------------------------------
// lt_scalar
// ---------------------------------------------------------------------------

/// Element-wise `< scalar` comparison. Returns 1.0 where true, 0.0 otherwise.
#[no_mangle]
pub extern "C" fn nsl_tensor_lt_scalar(tensor_ptr: i64, scalar: f64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = get_shape_vec(tensor);
    let data = tensor.data_f64();
    let len = tensor.len as usize;

    let result_ptr = create_tensor_with_shape_rs(&shape);
    let result_tensor = NslTensor::from_ptr(result_ptr);
    let result_data = result_tensor.data_f64();

    for i in 0..len {
        unsafe {
            let v = *data.add(i);
            *result_data.add(i) = if v < scalar { 1.0 } else { 0.0 };
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
            assert!(idx >= 0.0 && idx < 3.0);
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
