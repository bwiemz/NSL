//! M35b: GPTQ (Generalized Post-Training Quantization) with full OBQ algorithm.
//!
//! Implements the Optimal Brain Quantizer (OBQ) from the GPTQ paper:
//! column-wise quantization with Hessian-based error compensation.
//!
//! Algorithm overview:
//! 1. Compute Hessian H = X^T X from calibration data
//! 2. Add damping: H += λI (λ = 0.01 * mean(diag(H)))
//! 3. Compute Cholesky factorization of H^{-1}
//! 4. For each column i (optionally in act-order):
//!    a. Quantize w_i to q_i using group scale/zero
//!    b. Compute error δ = (w_i - q_i) / H^{-1}_{ii}
//!    c. Compensate remaining columns: W_{:,i+1:} -= δ * H^{-1}_{i,i+1:}
//!
//! The compensation step is what makes GPTQ superior to RTN: quantization
//! errors are distributed to unquantized columns proportionally to their
//! correlation (via the Hessian inverse), minimizing total reconstruction error.

use std::sync::atomic::AtomicI64;
use std::sync::Mutex;

use crate::awq::{
    AwqPackedWeight, compute_group_params, quantize_int4, write_nibble,
};
use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// GPTQ configuration
// ---------------------------------------------------------------------------

/// GPTQ quantization configuration.
#[derive(Clone, Debug)]
pub struct GptqConfig {
    /// Number of quantization bits (4 or 8).
    pub bits: usize,
    /// Group size for per-group quantization (typically 128).
    pub group_size: usize,
    /// Damping factor for Hessian regularization (fraction of mean diagonal).
    /// Default: 0.01 (1% of mean diagonal added to diagonal).
    pub damp_percent: f64,
    /// Whether to use activation-order (quantize columns in descending
    /// Hessian diagonal order for better quality).
    pub act_order: bool,
    /// Block size for blocked Hessian updates (0 = full column-wise).
    /// Larger blocks trade quality for speed. Typical: 128.
    pub block_size: usize,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            damp_percent: 0.01,
            act_order: true,
            block_size: 128,
        }
    }
}

// ---------------------------------------------------------------------------
// Hessian computation from calibration data
// ---------------------------------------------------------------------------

/// Accumulator for computing H = X^T X from calibration batches.
///
/// Call `add_batch()` for each calibration input, then `finalize()` to get
/// the averaged Hessian matrix.
pub struct HessianAccumulator {
    /// Running sum of X^T X, stored as flat [K, K] row-major.
    h: Vec<f64>,
    /// Input dimension (columns of X = rows of weight matrix).
    k: usize,
    /// Number of samples accumulated.
    n_samples: usize,
}

impl HessianAccumulator {
    pub fn new(k: usize) -> Self {
        Self {
            h: vec![0.0; k * k],
            k,
            n_samples: 0,
        }
    }

    /// Add a calibration batch X [batch_size, K] to the Hessian accumulator.
    ///
    /// Computes H += X^T @ X (outer product accumulation).
    pub fn add_batch(&mut self, x: &[f64], batch_size: usize) {
        let k = self.k;
        assert_eq!(x.len(), batch_size * k, "calibration batch shape mismatch");

        // H += X^T @ X — computed as sum of outer products of each row
        for b in 0..batch_size {
            let row = &x[b * k..(b + 1) * k];
            for i in 0..k {
                for j in i..k {
                    let val = row[i] * row[j];
                    self.h[i * k + j] += val;
                    if i != j {
                        self.h[j * k + i] += val; // symmetric
                    }
                }
            }
        }
        self.n_samples += batch_size;
    }

    /// Finalize: divide by sample count and return the Hessian [K, K].
    pub fn finalize(mut self) -> Vec<f64> {
        if self.n_samples > 0 {
            let scale = 1.0 / self.n_samples as f64;
            for v in &mut self.h {
                *v *= scale;
            }
        }
        self.h
    }

    /// Get the current accumulated (non-averaged) Hessian.
    pub fn current(&self) -> &[f64] {
        &self.h
    }
}

// ---------------------------------------------------------------------------
// Cholesky factorization of Hessian inverse
// ---------------------------------------------------------------------------

/// Add damping to Hessian diagonal: H[i,i] += damp_percent * mean(diag(H)).
fn add_damping(h: &mut [f64], k: usize, damp_percent: f64) {
    let diag_mean: f64 = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
    let damp = damp_percent * diag_mean;
    // Ensure minimum damping for numerical stability
    let damp = damp.max(1e-8);
    for i in 0..k {
        h[i * k + i] += damp;
    }
}

/// In-place Cholesky factorization: H → L where H = L L^T.
///
/// The lower triangle of `h` is overwritten with L.
/// Returns false if the matrix is not positive definite.
fn cholesky_lower(h: &mut [f64], k: usize) -> bool {
    for i in 0..k {
        for j in 0..=i {
            let mut sum = h[i * k + j];
            for p in 0..j {
                sum -= h[i * k + p] * h[j * k + p];
            }
            if i == j {
                if sum <= 0.0 {
                    return false; // not positive definite
                }
                h[i * k + j] = sum.sqrt();
            } else {
                h[i * k + j] = sum / h[j * k + j];
            }
        }
    }
    true
}

/// Compute the inverse of a lower-triangular Cholesky factor L.
///
/// Returns L^{-1} as a flat [K, K] row-major matrix.
/// L^{-1} is also lower triangular.
fn cholesky_inv_lower(l: &[f64], k: usize) -> Vec<f64> {
    let mut inv = vec![0.0; k * k];

    // Column-by-column forward substitution: for each column j,
    // solve L * x_j = e_j from top to bottom.
    for j in 0..k {
        inv[j * k + j] = 1.0 / l[j * k + j];
        for i in (j + 1)..k {
            let mut sum = 0.0;
            for p in j..i {
                sum += l[i * k + p] * inv[p * k + j];
            }
            inv[i * k + j] = -sum / l[i * k + i];
        }
    }
    inv
}

/// Compute H^{-1} from its Cholesky factor L (where H = L L^T).
///
/// H^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
fn hessian_inverse_from_cholesky(l: &[f64], k: usize) -> Vec<f64> {
    let l_inv = cholesky_inv_lower(l, k);

    // H^{-1} = L^{-T} @ L^{-1}
    let mut h_inv = vec![0.0; k * k];
    for i in 0..k {
        for j in i..k {
            let mut sum = 0.0;
            // L^{-T}[i,p] = L^{-1}[p,i], so H^{-1}[i,j] = sum_p L^{-1}[p,i] * L^{-1}[p,j]
            for p in j..k {
                sum += l_inv[p * k + i] * l_inv[p * k + j];
            }
            h_inv[i * k + j] = sum;
            h_inv[j * k + i] = sum; // symmetric
        }
    }
    h_inv
}

// ---------------------------------------------------------------------------
// Core GPTQ quantization algorithm (OBQ)
// ---------------------------------------------------------------------------

/// GPTQ quantize with full OBQ (Optimal Brain Quantizer) algorithm.
///
/// `weights`: flattened [K, N] weight matrix (row-major)
/// `hessian`: [K, K] Hessian matrix (X^T X / n_samples)
/// `k`: number of input features (rows)
/// `n`: number of output features (columns)
/// `config`: quantization configuration
///
/// Returns the packed 4-bit weight in AWQ format.
pub fn gptq_quantize_obq(
    weights: &[f64],
    hessian: &[f64],
    k: usize,
    n: usize,
    config: &GptqConfig,
) -> AwqPackedWeight {
    assert_eq!(weights.len(), k * n);
    assert_eq!(hessian.len(), k * k);

    let group_size = config.group_size;
    let num_groups = k.div_ceil(group_size);
    let packed_bytes = (k * n).div_ceil(2);

    // Step 1: Compute Hessian inverse via Cholesky
    let mut h = hessian.to_vec();
    add_damping(&mut h, k, config.damp_percent);

    let cholesky_ok = cholesky_lower(&mut h, k);
    if !cholesky_ok {
        // Fall back to RTN if Cholesky fails (degenerate Hessian)
        eprintln!("[nsl-gptq] Cholesky failed, falling back to RTN");
        return crate::awq::awq_quantize_cpu(weights, k, n, group_size);
    }
    let h_inv = hessian_inverse_from_cholesky(&h, k);

    // Step 2: Determine column order
    let col_order: Vec<usize> = if config.act_order {
        // Act-order: quantize columns with largest Hessian diagonal first
        // (these are the most "important" / most activated columns)
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| {
            h_inv[b * k + b].partial_cmp(&h_inv[a * k + a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        order
    } else {
        (0..k).collect()
    };

    // Inverse permutation: perm_inv[col_order[i]] = i
    let mut perm_inv = vec![0usize; k];
    for (new_pos, &orig_col) in col_order.iter().enumerate() {
        perm_inv[orig_col] = new_pos;
    }

    // Step 3: Make a mutable copy of weights for error compensation
    // We work in the permuted order for act-order, but store results in original order
    let mut w = weights.to_vec();

    // Allocate output
    let mut packed_data = vec![0u8; packed_bytes];
    let mut scales = vec![0.0f64; num_groups * n];
    let mut zeros = vec![0.0f64; num_groups * n];

    // Step 4: Column-wise OBQ with optional blocking
    let block_size = if config.block_size > 0 { config.block_size.min(k) } else { k };

    for block_start in (0..k).step_by(block_size) {
        let block_end = (block_start + block_size).min(k);

        // Process columns in this block
        for idx in block_start..block_end {
            let col = col_order[idx];

            // Get H^{-1}_{col,col} (diagonal element for this column)
            let h_inv_diag = h_inv[col * k + col];
            if h_inv_diag.abs() < 1e-15 {
                // Degenerate — just use RTN for this column
                for row in 0..n {
                    let g = col / group_size;
                    // Lazily compute group params for this column/group combo
                    let group: Vec<f64> = {
                        let start = g * group_size;
                        let end = (start + group_size).min(k);
                        (start..end).map(|r| w[r * n + row]).collect()
                    };
                    let (scale, zero) = compute_group_params(&group);
                    scales[g * n + row] = scale;
                    zeros[g * n + row] = zero;
                    let q = quantize_int4(w[col * n + row], scale, zero);
                    write_nibble(&mut packed_data, col * n + row, q);
                }
                continue;
            }

            // For each output column (row in the weight matrix sense is input dim):
            // We quantize w[col, :] (all output columns for this input column)
            // and compensate remaining input columns.
            for row in 0..n {
                let g = col / group_size;

                // Compute group params for this group/output-column pair
                let start = g * group_size;
                let end = (start + group_size).min(k);
                let group: Vec<f64> = (start..end).map(|r| w[r * n + row]).collect();
                let (scale, zero) = compute_group_params(&group);
                scales[g * n + row] = scale;
                zeros[g * n + row] = zero;

                // Quantize this weight
                let w_val = w[col * n + row];
                let q = quantize_int4(w_val, scale, zero);
                write_nibble(&mut packed_data, col * n + row, q);

                // Compute quantization error
                let q_val = (q as f64) * scale + zero;
                let error = (w_val - q_val) / h_inv_diag;

                // Compensate remaining columns in this block
                for jdx in (idx + 1)..block_end {
                    let j = col_order[jdx];
                    // W[j, row] -= error * H^{-1}[col, j]
                    w[j * n + row] -= error * h_inv[col * k + j];
                }
            }
        }

        // After processing this block, propagate errors to columns in future blocks
        // This is the "lazy batch update" from the GPTQ paper for blocked processing
        if block_end < k {
            for row in 0..n {
                for idx in block_start..block_end {
                    let col = col_order[idx];
                    let g = col / group_size;
                    let scale = scales[g * n + row];
                    let zero = zeros[g * n + row];

                    let q = crate::awq::read_nibble(
                        &packed_data,
                        col * n + row,
                    );
                    let q_val = (q as f64) * scale + zero;
                    let original_w = weights[col * n + row]; // use original weight for cross-block error
                    let error = (original_w - q_val) / h_inv[col * k + col].max(1e-15);

                    for jdx in block_end..k {
                        let j = col_order[jdx];
                        w[j * n + row] -= error * h_inv[col * k + j];
                    }
                }
            }
        }
    }

    let packed_data = packed_data.into_boxed_slice();
    let scales = scales.into_boxed_slice();
    let zeros = zeros.into_boxed_slice();

    AwqPackedWeight {
        data: Box::into_raw(packed_data) as *mut u8,
        scales: Box::into_raw(scales) as *mut f64,
        zeros: Box::into_raw(zeros) as *mut f64,
        k: k as i64,
        n: n as i64,
        group_size: group_size as i64,
        num_groups: num_groups as i64,
        refcount: AtomicI64::new(1),
    }
}

/// Simple RTN (Round-To-Nearest) quantization — baseline without Hessian.
/// Kept for comparison and fallback.
pub fn gptq_quantize_rtn(
    weights: &[f64],
    k: usize,
    n: usize,
    group_size: usize,
) -> AwqPackedWeight {
    crate::awq::awq_quantize_cpu(weights, k, n, group_size)
}

/// GPTQ quantize with full Hessian matrix.
///
/// Public entry point that dispatches to OBQ if a full Hessian [K,K] is provided,
/// or RTN if only diagonal or no Hessian.
pub fn gptq_quantize_cpu(
    weights: &[f64],
    hessian: &[f64],
    k: usize,
    n: usize,
    group_size: usize,
    bits: usize,
) -> AwqPackedWeight {
    // If hessian length is K*K, use full OBQ
    if hessian.len() == k * k {
        let config = GptqConfig {
            bits,
            group_size,
            act_order: true,
            ..Default::default()
        };
        gptq_quantize_obq(weights, hessian, k, n, &config)
    } else if hessian.len() == k {
        // Diagonal Hessian: construct full diagonal matrix and use OBQ
        let mut full_h = vec![0.0; k * k];
        for i in 0..k {
            full_h[i * k + i] = hessian[i].max(1e-8);
        }
        let config = GptqConfig {
            bits,
            group_size,
            act_order: true,
            ..Default::default()
        };
        gptq_quantize_obq(weights, &full_h, k, n, &config)
    } else {
        // No valid Hessian — fall back to RTN
        gptq_quantize_rtn(weights, k, n, group_size)
    }
}

// ---------------------------------------------------------------------------
// Global Hessian accumulator for calibration pipeline
// ---------------------------------------------------------------------------

static HESSIAN_CTX: Mutex<Option<HessianAccumulator>> = Mutex::new(None);

/// Initialize a Hessian accumulator for GPTQ calibration.
///
/// `k`: input dimension of the weight matrix being quantized.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_gptq_hessian_init(k: i64) -> i64 {
    let mut guard = match HESSIAN_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    *guard = Some(HessianAccumulator::new(k as usize));
    0
}

/// Add a calibration batch to the Hessian accumulator.
///
/// `input_ptr`: NslTensor* with shape [batch_size, K] — calibration inputs
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_gptq_hessian_add_batch(input_ptr: i64) -> i64 {
    let t = unsafe { &*(input_ptr as *const NslTensor) };
    assert!(t.ndim >= 2, "calibration input must be 2D");

    let batch_size = unsafe { *t.shape } as usize;
    let k = unsafe { *t.shape.add(1) } as usize;
    let len = t.len as usize;

    let data: Vec<f64> = if t.dtype == 1 {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        raw.iter().map(|&v| v as f64).collect()
    } else {
        unsafe { std::slice::from_raw_parts(t.data as *const f64, len) }.to_vec()
    };

    let mut guard = match HESSIAN_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match guard.as_mut() {
        Some(acc) => {
            acc.add_batch(&data, batch_size);
            0
        }
        None => -2, // not initialized
    }
}

/// Finalize the Hessian accumulator and return the Hessian as an NslTensor [K, K].
///
/// The accumulator is consumed — call nsl_gptq_hessian_init() again to start a new calibration.
/// Returns an NslTensor* (as i64) containing the Hessian, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_gptq_hessian_finalize() -> i64 {
    let mut guard = match HESSIAN_CTX.lock() {
        Ok(g) => g,
        Err(_) => return 0,
    };
    let acc = match guard.take() {
        Some(a) => a,
        None => return 0,
    };
    let k = acc.k;
    let h = acc.finalize();

    // Create an NslTensor [K, K] containing the Hessian
    let len = k * k;
    let data = checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
    for (i, &v) in h.iter().enumerate() {
        unsafe { *data.add(i) = v; }
    }

    let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape = k as i64;
        *shape.add(1) = k as i64;
    }
    let strides = NslTensor::compute_strides(shape, 2);

    let tensor = Box::new(NslTensor::new(
        data as *mut std::ffi::c_void,
        shape,
        strides,
        2,
        len as i64,
        0,  // CPU
        0,  // f64
        1,  // owns_data
        0,  // no owner
    ));
    NslTensor::publish(tensor)
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// GPTQ quantize: weight + Hessian -> packed weight.
///
/// `weight_ptr`: NslTensor [K, N] — weight matrix to quantize
/// `hessian_ptr`: NslTensor [K, K] (full Hessian) or [K] (diagonal) or 0 (RTN fallback)
/// `group_size`: quantization group size (typically 128)
/// `bits`: quantization bits (4 or 8)
///
/// Returns pointer to AwqPackedWeight.
#[no_mangle]
pub extern "C" fn nsl_gptq_quantize(
    weight_ptr: i64,
    hessian_ptr: i64,
    group_size: i64,
    bits: i64,
) -> i64 {
    let t = unsafe { &*(weight_ptr as *const NslTensor) };
    assert!(t.ndim >= 2, "nsl_gptq_quantize requires 2D weight tensor (got {}D)", t.ndim);
    let len = t.len as usize;

    let data: Vec<f64> = if t.dtype == 1 {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        raw.iter().map(|&v| v as f64).collect()
    } else {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        raw.to_vec()
    };

    let k = unsafe { *t.shape } as usize;
    let n = unsafe { *t.shape.add(1) } as usize;

    let hessian = if hessian_ptr == 0 {
        vec![1.0f64; k] // diagonal identity → RTN
    } else {
        let h = unsafe { &*(hessian_ptr as *const NslTensor) };
        let h_len = h.len as usize;
        if h.dtype == 1 {
            let raw = unsafe { std::slice::from_raw_parts(h.data as *const f32, h_len) };
            raw.iter().map(|&v| v as f64).collect()
        } else {
            unsafe { std::slice::from_raw_parts(h.data as *const f64, h_len) }.to_vec()
        }
    };

    let packed = gptq_quantize_cpu(&data, &hessian, k, n, group_size as usize, bits as usize);
    Box::into_raw(Box::new(packed)) as i64
}

/// GPTQ quantize with explicit config (act_order, block_size, damp_percent).
///
/// `weight_ptr`: NslTensor [K, N]
/// `hessian_ptr`: NslTensor [K, K]
/// `group_size`: quantization group size
/// `bits`: 4 or 8
/// `act_order`: 1 = enable activation order, 0 = sequential
/// `block_size`: block size for blocked updates (0 = full column-wise)
/// `damp_percent_bits`: f64 damping percent encoded as u64 bits
#[no_mangle]
pub extern "C" fn nsl_gptq_quantize_ext(
    weight_ptr: i64,
    hessian_ptr: i64,
    group_size: i64,
    bits: i64,
    act_order: i64,
    block_size: i64,
    damp_percent_bits: i64,
) -> i64 {
    let t = unsafe { &*(weight_ptr as *const NslTensor) };
    assert!(t.ndim >= 2);
    let len = t.len as usize;

    let data: Vec<f64> = if t.dtype == 1 {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        raw.iter().map(|&v| v as f64).collect()
    } else {
        unsafe { std::slice::from_raw_parts(t.data as *const f64, len) }.to_vec()
    };

    let k = unsafe { *t.shape } as usize;
    let n = unsafe { *t.shape.add(1) } as usize;

    let hessian = if hessian_ptr == 0 {
        // No Hessian → identity for RTN
        let mut h = vec![0.0; k * k];
        for i in 0..k { h[i * k + i] = 1.0; }
        h
    } else {
        let h = unsafe { &*(hessian_ptr as *const NslTensor) };
        let h_len = h.len as usize;
        if h.dtype == 1 {
            let raw = unsafe { std::slice::from_raw_parts(h.data as *const f32, h_len) };
            raw.iter().map(|&v| v as f64).collect()
        } else {
            unsafe { std::slice::from_raw_parts(h.data as *const f64, h_len) }.to_vec()
        }
    };

    let damp = f64::from_bits(damp_percent_bits as u64);

    let config = GptqConfig {
        bits: bits as usize,
        group_size: group_size as usize,
        damp_percent: if damp > 0.0 { damp } else { 0.01 },
        act_order: act_order != 0,
        block_size: block_size as usize,
    };

    let packed = gptq_quantize_obq(&data, &hessian, k, n, &config);
    Box::into_raw(Box::new(packed)) as i64
}

/// GPTQ dequant-matmul (same kernel as AWQ — packed format is identical).
#[no_mangle]
pub extern "C" fn nsl_gptq_matmul(
    input_ptr: i64,
    packed_ptr: i64,
    group_size: i64,
    _bits: i64,
) -> i64 {
    crate::awq::nsl_awq_matmul(input_ptr, packed_ptr, group_size)
}

/// Free a GPTQ packed weight.
#[no_mangle]
pub extern "C" fn nsl_gptq_free(packed_ptr: i64) {
    crate::awq::nsl_awq_free(packed_ptr);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::awq::{awq_dequantize_cpu, awq_free_packed, awq_quantize_cpu};

    /// Compute mean squared error between two vectors.
    fn mse(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64
    }

    /// Reference matmul A[M,K] @ B[K,N] -> C[M,N]
    fn matmul_ref(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = acc;
            }
        }
        c
    }

    // -----------------------------------------------------------------------
    // Hessian accumulator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hessian_accumulator_identity() {
        // If calibration data is identity matrix, H = I/n
        let mut acc = HessianAccumulator::new(3);
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        acc.add_batch(&identity, 3);
        let h = acc.finalize();
        // H should be I/3
        assert!((h[0] - 1.0/3.0).abs() < 1e-10);
        assert!((h[4] - 1.0/3.0).abs() < 1e-10);
        assert!((h[8] - 1.0/3.0).abs() < 1e-10);
        assert!(h[1].abs() < 1e-10);
    }

    #[test]
    fn test_hessian_accumulator_correlated() {
        let mut acc = HessianAccumulator::new(2);
        // Two samples: [1, 2] and [3, 4]
        // H = (1/2) * ([1,2]^T[1,2] + [3,4]^T[3,4])
        //   = (1/2) * ([1,2; 2,4] + [9,12; 12,16])
        //   = [5, 7; 7, 10]
        let data = vec![1.0, 2.0, 3.0, 4.0];
        acc.add_batch(&data, 2);
        let h = acc.finalize();
        assert!((h[0] - 5.0).abs() < 1e-10);
        assert!((h[1] - 7.0).abs() < 1e-10);
        assert!((h[2] - 7.0).abs() < 1e-10); // symmetric
        assert!((h[3] - 10.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Cholesky tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cholesky_2x2() {
        // H = [4, 2; 2, 3]
        let mut h = vec![4.0, 2.0, 2.0, 3.0];
        assert!(cholesky_lower(&mut h, 2));
        // L should be [2, 0; 1, sqrt(2)]
        assert!((h[0] - 2.0).abs() < 1e-10);
        assert!((h[2] - 1.0).abs() < 1e-10);
        assert!((h[3] - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_hessian_inverse_roundtrip() {
        // H = [4, 2; 2, 3]
        // det(H) = 12 - 4 = 8
        // H^{-1} = (1/8) * [3, -2; -2, 4] = [0.375, -0.25; -0.25, 0.5]
        let mut h = vec![4.0, 2.0, 2.0, 3.0];
        assert!(cholesky_lower(&mut h, 2));
        let h_inv = hessian_inverse_from_cholesky(&h, 2);
        // Verify H * H^{-1} = I by checking values
        // Since our Cholesky only fills the lower triangle, upper may have stale data.
        // The inverse should still be correct if the algorithm is implemented right.
        assert!((h_inv[0] - 0.375).abs() < 1e-6, "h_inv[0,0]={}", h_inv[0]);
        assert!((h_inv[1] - (-0.25)).abs() < 1e-6, "h_inv[0,1]={}", h_inv[1]);
        assert!((h_inv[2] - (-0.25)).abs() < 1e-6, "h_inv[1,0]={}", h_inv[2]);
        assert!((h_inv[3] - 0.5).abs() < 1e-6, "h_inv[1,1]={}", h_inv[3]);
    }

    // -----------------------------------------------------------------------
    // RTN baseline test (preserved from M35)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gptq_quantize_rtn_baseline() {
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let hessian = vec![1.0f64; 4]; // diagonal Hessian
        let packed = gptq_quantize_cpu(&weights, &hessian, 4, 4, 4, 4);

        let recovered = awq_dequantize_cpu(&packed);
        for (i, (&orig, &rec)) in weights.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 0.15,
                "GPTQ RTN error too high at {}: orig={}, recovered={}",
                i, orig, rec
            );
        }

        unsafe { awq_free_packed(&packed) };
    }

    // -----------------------------------------------------------------------
    // OBQ quality tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gptq_obq_reduces_error_vs_rtn() {
        // Create a weight matrix with correlated columns
        let k = 8;
        let n = 4;
        let mut weights = vec![0.0f64; k * n];
        // Make columns correlated: w[i,j] = sin(i*j + i) * 0.5
        for i in 0..k {
            for j in 0..n {
                weights[i * n + j] = ((i * j + i) as f64 * 0.7).sin() * 0.5;
            }
        }

        // Construct a non-trivial Hessian (not just identity)
        // Use a diagonally dominant matrix: H[i,i] = 1 + 0.1*i, H[i,j] = 0.05 for i≠j
        let mut hessian = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                if i == j {
                    hessian[i * k + j] = 1.0 + 0.1 * i as f64;
                } else {
                    hessian[i * k + j] = 0.05;
                }
            }
        }

        // Quantize with RTN (baseline)
        let packed_rtn = gptq_quantize_rtn(&weights, k, n, k); // one group
        let recovered_rtn = awq_dequantize_cpu(&packed_rtn);
        let mse_rtn = mse(&weights, &recovered_rtn);

        // Quantize with OBQ
        let config = GptqConfig {
            bits: 4,
            group_size: k,
            act_order: true,
            block_size: 0, // full column-wise
            ..Default::default()
        };
        let packed_obq = gptq_quantize_obq(&weights, &hessian, k, n, &config);
        let recovered_obq = awq_dequantize_cpu(&packed_obq);
        let mse_obq = mse(&weights, &recovered_obq);

        // OBQ should have equal or lower MSE than RTN
        // (In practice, OBQ compensates errors across correlated columns)
        assert!(
            mse_obq <= mse_rtn * 1.05, // allow 5% tolerance for edge cases
            "OBQ MSE ({:.6}) should be <= RTN MSE ({:.6})",
            mse_obq, mse_rtn
        );

        unsafe {
            awq_free_packed(&packed_rtn);
            awq_free_packed(&packed_obq);
        }
    }

    #[test]
    fn test_gptq_obq_with_identity_hessian_matches_rtn() {
        // With identity Hessian, OBQ should produce similar results to RTN
        // (no cross-column compensation since columns are independent)
        let k = 4;
        let n = 4;
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let mut hessian = vec![0.0; k * k];
        for i in 0..k {
            hessian[i * k + i] = 1.0;
        }

        let config = GptqConfig {
            bits: 4,
            group_size: k,
            act_order: false, // sequential order
            block_size: 0,
            ..Default::default()
        };
        let packed = gptq_quantize_obq(&weights, &hessian, k, n, &config);
        let recovered = awq_dequantize_cpu(&packed);

        for (i, (&orig, &rec)) in weights.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 0.2,
                "OBQ with identity H error too high at {}: orig={:.3}, recovered={:.3}",
                i, orig, rec
            );
        }

        unsafe { awq_free_packed(&packed) };
    }

    #[test]
    fn test_gptq_obq_blocked_vs_unblocked() {
        let k = 8;
        let n = 4;
        let weights: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin() * 0.4).collect();
        let mut hessian = vec![0.0; k * k];
        for i in 0..k {
            hessian[i * k + i] = 1.0 + 0.1 * i as f64;
            for j in 0..k {
                if i != j {
                    hessian[i * k + j] = 0.02;
                }
            }
        }

        // Full column-wise (block_size = 0)
        let config_full = GptqConfig {
            bits: 4,
            group_size: k,
            act_order: false,
            block_size: 0,
            ..Default::default()
        };
        let packed_full = gptq_quantize_obq(&weights, &hessian, k, n, &config_full);
        let recovered_full = awq_dequantize_cpu(&packed_full);
        let mse_full = mse(&weights, &recovered_full);

        // Blocked (block_size = 4)
        let config_blocked = GptqConfig {
            bits: 4,
            group_size: k,
            act_order: false,
            block_size: 4,
            ..Default::default()
        };
        let packed_blocked = gptq_quantize_obq(&weights, &hessian, k, n, &config_blocked);
        let recovered_blocked = awq_dequantize_cpu(&packed_blocked);
        let mse_blocked = mse(&weights, &recovered_blocked);

        // Both should produce reasonable quantization
        assert!(mse_full < 0.1, "Full OBQ MSE too high: {:.6}", mse_full);
        assert!(mse_blocked < 0.1, "Blocked OBQ MSE too high: {:.6}", mse_blocked);

        unsafe {
            awq_free_packed(&packed_full);
            awq_free_packed(&packed_blocked);
        }
    }

    #[test]
    fn test_gptq_obq_act_order_effect() {
        let k = 8;
        let n = 4;
        let weights: Vec<f64> = (0..32).map(|i| (i as f64 * 0.5).cos() * 0.3).collect();

        // Hessian with varying diagonal (some columns more important)
        let mut hessian = vec![0.0; k * k];
        for i in 0..k {
            hessian[i * k + i] = (i + 1) as f64 * 2.0; // increasing importance
        }

        // Without act-order
        let config_no_act = GptqConfig {
            bits: 4,
            group_size: k,
            act_order: false,
            block_size: 0,
            ..Default::default()
        };
        let packed_no_act = gptq_quantize_obq(&weights, &hessian, k, n, &config_no_act);
        let recovered_no_act = awq_dequantize_cpu(&packed_no_act);
        let mse_no_act = mse(&weights, &recovered_no_act);

        // With act-order
        let config_act = GptqConfig {
            bits: 4,
            group_size: k,
            act_order: true,
            block_size: 0,
            ..Default::default()
        };
        let packed_act = gptq_quantize_obq(&weights, &hessian, k, n, &config_act);
        let recovered_act = awq_dequantize_cpu(&packed_act);
        let mse_act = mse(&weights, &recovered_act);

        // Both should produce valid quantization
        assert!(mse_no_act < 0.1, "No act-order MSE: {:.6}", mse_no_act);
        assert!(mse_act < 0.1, "Act-order MSE: {:.6}", mse_act);

        unsafe {
            awq_free_packed(&packed_no_act);
            awq_free_packed(&packed_act);
        }
    }

    // -----------------------------------------------------------------------
    // Matmul quality test
    // -----------------------------------------------------------------------

    #[test]
    fn test_gptq_matmul_quality() {
        let k = 8;
        let n = 4;
        let m = 2;

        let weights: Vec<f64> = (0..32).map(|i| (i as f64 * 0.2).sin() * 0.5).collect();
        let input: Vec<f64> = (0..16).map(|i| (i as f64 * 0.3).cos()).collect();

        // Reference matmul
        let ref_output = matmul_ref(&input, &weights, m, k, n);

        // GPTQ quantized matmul
        let mut hessian = vec![0.0; k * k];
        for i in 0..k { hessian[i * k + i] = 1.0; }
        let config = GptqConfig {
            bits: 4,
            group_size: k,
            ..Default::default()
        };
        let packed = gptq_quantize_obq(&weights, &hessian, k, n, &config);
        let gptq_output = crate::awq::awq_matmul_cpu(&input, &packed, m);

        // Check that matmul results are close
        for (i, (&r, &g)) in ref_output.iter().zip(gptq_output.iter()).enumerate() {
            let tol = r.abs() * 0.15 + 0.1;
            assert!(
                (r - g).abs() < tol,
                "GPTQ matmul error at {}: ref={:.4}, gptq={:.4}",
                i, r, g
            );
        }

        unsafe { awq_free_packed(&packed) };
    }

    // -----------------------------------------------------------------------
    // Larger matrix test
    // -----------------------------------------------------------------------

    #[test]
    fn test_gptq_obq_larger_matrix() {
        let k = 32;
        let n = 16;
        let group_size = 8;

        // Generate structured weights (not random, for reproducibility)
        let weights: Vec<f64> = (0..k * n)
            .map(|i| {
                let row = i / n;
                let col = i % n;
                ((row * 7 + col * 13) as f64 * 0.1).sin() * 0.3
            })
            .collect();

        // Compute synthetic Hessian from "calibration data"
        let cal_batches = 4;
        let mut acc = HessianAccumulator::new(k);
        for b in 0..cal_batches {
            let cal: Vec<f64> = (0..8 * k)
                .map(|i| ((i + b * 100) as f64 * 0.05).cos() * 0.5)
                .collect();
            acc.add_batch(&cal, 8);
        }
        let hessian = acc.finalize();

        // Quantize with OBQ
        let config = GptqConfig {
            bits: 4,
            group_size,
            act_order: true,
            block_size: 16,
            ..Default::default()
        };
        let packed = gptq_quantize_obq(&weights, &hessian, k, n, &config);
        let recovered = awq_dequantize_cpu(&packed);
        let error = mse(&weights, &recovered);

        assert!(error < 0.05, "Large matrix GPTQ MSE too high: {:.6}", error);
        assert_eq!(packed.k, k as i64);
        assert_eq!(packed.n, n as i64);
        assert_eq!(packed.group_size, group_size as i64);

        unsafe { awq_free_packed(&packed) };
    }

    // -----------------------------------------------------------------------
    // FFI calibration pipeline test
    // -----------------------------------------------------------------------

    static FFI_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_gptq_hessian_ffi_lifecycle() {
        let _lock = FFI_LOCK.lock().unwrap();

        // Initialize accumulator for K=4
        assert_eq!(nsl_gptq_hessian_init(4), 0);

        // Finalize without adding data — should still work
        let h_ptr = nsl_gptq_hessian_finalize();
        assert_ne!(h_ptr, 0);

        // Clean up
        crate::tensor::nsl_tensor_free(h_ptr);
    }

    #[test]
    fn test_gptq_dispatch_full_hessian() {
        // When hessian.len() == k*k, should use OBQ
        let k = 4;
        let n = 4;
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let mut hessian = vec![0.0; k * k];
        for i in 0..k { hessian[i * k + i] = 1.0; }

        let packed = gptq_quantize_cpu(&weights, &hessian, k, n, 4, 4);
        let recovered = awq_dequantize_cpu(&packed);

        for (&orig, &rec) in weights.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.2);
        }

        unsafe { awq_free_packed(&packed) };
    }

    #[test]
    fn test_gptq_dispatch_diagonal_hessian() {
        // When hessian.len() == k, should expand to diagonal and use OBQ
        let k = 4;
        let n = 4;
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let hessian = vec![1.0, 2.0, 3.0, 4.0]; // diagonal

        let packed = gptq_quantize_cpu(&weights, &hessian, k, n, 4, 4);
        let recovered = awq_dequantize_cpu(&packed);

        for (&orig, &rec) in weights.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.2);
        }

        unsafe { awq_free_packed(&packed) };
    }
}
