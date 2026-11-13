//! M35b: GPTQ (Frantar et al., "GPTQ: Accurate Post-Training Quantization
//! for Generative Pre-trained Transformers") for a 4-bit packed weight.
//!
//! Nothing in the compiler emits a call to this module: a `quant` block
//! with `default: gptq4` is refused by the checker. It is a library routine
//! with FFI entry points and unit tests.
//!
//! The weight is `W[K, N]` (K inputs, N outputs, `y = x @ W`), quantized in
//! groups of `group_size` inputs per output, each group with its own
//! asymmetric 4-bit scale and zero. GPTQ minimises the OUTPUT error
//! `tr(D^T H D)`, `D = W - Q`, not the weight error; a correct run raises
//! the weight mean-squared error above round-to-nearest.
//!
//! Algorithm:
//! 1. `H = X^T X / samples` from calibration inputs; damp it,
//!    `H += 0.01 * mean(diag(H)) * I`.
//! 2. Fix every group's scale and zero from the ORIGINAL weights.
//! 3. Order the inputs (optionally by descending `H[i,i]`), permute H to
//!    match, invert it, and take the upper Cholesky factor `U` of the
//!    inverse (`H^-1 = U^T U`). Row `i` of `U` is what the inverse over the
//!    not-yet-quantized inputs `i..K` contributes, so no per-step update of
//!    the inverse is needed.
//! 4. For each input `i` in that order: quantize row `i` of W with its
//!    group's fixed parameters, form `e = (w_i - q_i) / U[i,i]`, and
//!    subtract `e * U[i,j]` from every later row `j`.

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
    /// Group size for per-group quantization (typically 128).
    pub group_size: usize,
    /// Damping factor for Hessian regularization (fraction of mean diagonal).
    /// Default: 0.01 (1% of mean diagonal added to diagonal).
    pub damp_percent: f64,
    /// Whether to use activation-order (quantize columns in descending
    /// Hessian diagonal order for better quality).
    pub act_order: bool,
    /// Block size for the lazy batched update (0 = one block). Updates to
    /// inputs outside the current block are deferred to the block's end;
    /// the result is the same up to floating-point rounding, only the
    /// order of the subtractions changes.
    pub block_size: usize,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
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
// Damping, Cholesky factorization and the inverse
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
// Core GPTQ quantization algorithm
// ---------------------------------------------------------------------------

/// Why a GPTQ request cannot be carried out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GptqError {
    /// The packed format holds one nibble per weight; no other width exists.
    UnsupportedBits(usize),
    /// The Hessian is neither `[K, K]` nor a `[K]` diagonal.
    HessianShape { len: usize, k: usize },
    /// `group_size` is zero.
    ZeroGroupSize,
}

impl std::fmt::Display for GptqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedBits(bits) => write!(
                f,
                "GPTQ is implemented for 4-bit codes only (got bits={bits}): the packed weight \
                 stores one nibble per element"
            ),
            Self::HessianShape { len, k } => write!(
                f,
                "GPTQ Hessian has {len} elements; expected {} ([K, K]) or {k} (a [K] diagonal) for K={k}",
                k * k
            ),
            Self::ZeroGroupSize => write!(f, "GPTQ group size must be positive"),
        }
    }
}

impl std::error::Error for GptqError {}

/// Each group's 4-bit scale and zero, from the weights as given:
/// `[num_groups * n]`, indexed `g * n + output`.
fn static_group_params(weights: &[f64], k: usize, n: usize, group_size: usize) -> (Vec<f64>, Vec<f64>) {
    let num_groups = k.div_ceil(group_size);
    let mut scales = vec![0.0f64; num_groups * n];
    let mut zeros = vec![0.0f64; num_groups * n];
    let mut group: Vec<f64> = Vec::with_capacity(group_size);
    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(k);
        for out in 0..n {
            group.clear();
            group.extend((start..end).map(|r| weights[r * n + out]));
            let (scale, zero) = compute_group_params(&group);
            scales[g * n + out] = scale;
            zeros[g * n + out] = zero;
        }
    }
    (scales, zeros)
}

/// GPTQ: quantize `weights` to 4 bits, compensating each input's rounding
/// error on the inputs not yet quantized.
///
/// `weights`: flattened [K, N] weight matrix (row-major)
/// `hessian`: [K, K] Hessian matrix (X^T X / n_samples)
/// `k`: number of input features (rows)
/// `n`: number of output features (columns)
/// `config`: quantization configuration
///
/// Returns the packed 4-bit weight; `awq_dequantize_cpu` decodes it.
///
/// Two things here differ from the code this replaced, and each mattered
/// more than the choice of algorithm. Group parameters are computed once,
/// from the original weights: they used to be recomputed from the
/// already-compensated weights at every input and overwritten, so codes
/// written earlier were decoded with a later scale. And the error of input
/// `i` is spread with row `i` of the Cholesky factor of the inverse, which
/// is the inverse restricted to the inputs still open; the full inverse was
/// used unchanged throughout. With both wrong the output error was above
/// plain rounding.
pub fn gptq_quantize_obq(
    weights: &[f64],
    hessian: &[f64],
    k: usize,
    n: usize,
    config: &GptqConfig,
) -> AwqPackedWeight {
    assert_eq!(weights.len(), k * n);
    assert_eq!(hessian.len(), k * k);
    assert!(config.group_size > 0, "GPTQ group size must be positive");

    let group_size = config.group_size;
    let num_groups = k.div_ceil(group_size);
    let packed_bytes = (k * n).div_ceil(2);

    // Step 1: the order in which inputs are quantized. Act-order takes the
    // inputs with the largest Hessian diagonal first (a stable sort, so
    // equal diagonals keep their natural order).
    let order: Vec<usize> = if config.act_order {
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| {
            hessian[b * k + b].partial_cmp(&hessian[a * k + a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        order
    } else {
        (0..k).collect()
    };

    // Step 2: damped Hessian in that order, its inverse, and the Cholesky
    // factor of the inverse. `cholesky_lower` leaves L in the lower
    // triangle with `H^-1 = L L^T`; the upper factor is its transpose,
    // `U[i][j] = L[j][i]`.
    let mut damped = hessian.to_vec();
    add_damping(&mut damped, k, config.damp_percent);
    let mut hp = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            hp[i * k + j] = damped[order[i] * k + order[j]];
        }
    }
    let factor = if cholesky_lower(&mut hp, k) {
        let mut h_inv = hessian_inverse_from_cholesky(&hp, k);
        cholesky_lower(&mut h_inv, k).then_some(h_inv)
    } else {
        None
    };
    let Some(l_of_inverse) = factor else {
        // A Hessian that is not positive definite even after damping has
        // no inverse to compensate with. Round to nearest, and say so.
        crate::nsl_log!(
            WARN,
            "nsl-gptq",
            "[nsl-gptq] Hessian is not positive definite after damping; quantizing by round-to-nearest with no error compensation"
        );
        return crate::awq::awq_quantize_cpu(weights, k, n, group_size);
    };
    let u = |i: usize, j: usize| l_of_inverse[j * k + i];

    // Step 3: group parameters, fixed from the original weights.
    let (scales, zeros) = static_group_params(weights, k, n, group_size);

    // Step 4: quantize input by input, pushing each rounding error onto the
    // inputs that are still open.
    let mut w = weights.to_vec();
    let mut packed_data = vec![0u8; packed_bytes];
    let block_size = if config.block_size > 0 { config.block_size.min(k) } else { k };
    let mut block_err = vec![0.0f64; block_size * n];

    for block_start in (0..k).step_by(block_size.max(1)) {
        let block_end = (block_start + block_size).min(k);

        for i in block_start..block_end {
            let row = order[i];
            let g = row / group_size;
            let d = u(i, i);
            for out in 0..n {
                let scale = scales[g * n + out];
                let zero = zeros[g * n + out];
                let w_val = w[row * n + out];
                let q = quantize_int4(w_val, scale, zero);
                write_nibble(&mut packed_data, row * n + out, q);
                let err = (w_val - crate::awq::dequantize_int4(q, scale, zero)) / d;
                block_err[(i - block_start) * n + out] = err;
            }
            // Within the block the update is applied at once.
            for j in (i + 1)..block_end {
                let u_ij = u(i, j);
                let later = order[j];
                for out in 0..n {
                    w[later * n + out] -= block_err[(i - block_start) * n + out] * u_ij;
                }
            }
        }

        // Inputs beyond the block receive the block's errors together.
        for j in block_end..k {
            let later = order[j];
            for i in block_start..block_end {
                let u_ij = u(i, j);
                for out in 0..n {
                    w[later * n + out] -= block_err[(i - block_start) * n + out] * u_ij;
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

/// GPTQ with a full `[K, K]` Hessian or a `[K]` diagonal.
///
/// `bits` must be 4. It used to be accepted and ignored, so a request for
/// 8 bits returned 4-bit codes; the packed format has no other width, and
/// the request is now an error. A Hessian of any other length used to fall
/// back to round-to-nearest without a word; it is an error too.
pub fn gptq_quantize_cpu(
    weights: &[f64],
    hessian: &[f64],
    k: usize,
    n: usize,
    group_size: usize,
    bits: usize,
) -> Result<AwqPackedWeight, GptqError> {
    if bits != 4 {
        return Err(GptqError::UnsupportedBits(bits));
    }
    if group_size == 0 {
        return Err(GptqError::ZeroGroupSize);
    }
    let config = GptqConfig {
        group_size,
        act_order: true,
        ..Default::default()
    };
    if hessian.len() == k * k {
        Ok(gptq_quantize_obq(weights, hessian, k, n, &config))
    } else if hessian.len() == k {
        // Diagonal Hessian: no input is correlated with another, so there
        // is nothing to compensate and the result is round-to-nearest.
        let mut full_h = vec![0.0; k * k];
        for i in 0..k {
            full_h[i * k + i] = hessian[i].max(1e-8);
        }
        Ok(gptq_quantize_obq(weights, &full_h, k, n, &config))
    } else {
        Err(GptqError::HessianShape { len: hessian.len(), k })
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
#[unsafe(no_mangle)]
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
/// `input_ptr`: NslTensor* whose last dimension is K — calibration inputs.
/// Every leading dimension is a batch dimension: a `[B, T, K]` activation is
/// `B*T` samples of K inputs.
///
/// Returns 0 on success, -1 if the accumulator is poisoned, -2 if it was not
/// initialized, -3 if the tensor's last dimension is not the accumulator's K
/// (this used to trip an assertion inside an `extern "C"` function).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gptq_hessian_add_batch(input_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr_ref(input_ptr);
    if t.ndim < 1 {
        return -3;
    }
    let last_dim = unsafe { *t.shape.add(t.ndim as usize - 1) } as usize;

    let mut guard = match HESSIAN_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match guard.as_mut() {
        Some(acc) => {
            if acc.k == 0 || last_dim != acc.k {
                return -3;
            }
            let data: Vec<f64> = crate::quantize::logical_values_f64(input_ptr);
            acc.add_batch(&data, data.len() / acc.k);
            0
        }
        None => -2, // not initialized
    }
}

/// Finalize the Hessian accumulator and return the Hessian as an NslTensor [K, K].
///
/// The accumulator is consumed — call nsl_gptq_hessian_init() again to start a new calibration.
/// Returns an NslTensor* (as i64) containing the Hessian, or 0 on error.
#[unsafe(no_mangle)]
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

/// The `[K, N]` weight behind `weight_ptr`, in logical order on the host.
/// Dies with a message when the tensor is not a matrix: the shape is read
/// as `[K, N]` below, and a higher-rank tensor used to be quantized as its
/// first two dimensions over a longer buffer.
fn weight_matrix(weight_ptr: i64, who: &str) -> (Vec<f64>, usize, usize) {
    let t = NslTensor::from_ptr_ref(weight_ptr);
    if t.ndim != 2 {
        crate::fatal::die(
            crate::fatal::Fatal::ShapeMismatch,
            &format!("{who}: the weight must be a 2-D [K, N] tensor (got {}-D)", t.ndim),
        );
    }
    let k = unsafe { *t.shape } as usize;
    let n = unsafe { *t.shape.add(1) } as usize;
    (crate::quantize::logical_values_f64(weight_ptr), k, n)
}

fn die_on(err: GptqError, who: &str) -> ! {
    crate::fatal::die(crate::fatal::Fatal::Unsupported, &format!("{who}: {err}"))
}

/// GPTQ quantize: weight + Hessian -> packed weight.
///
/// `weight_ptr`: NslTensor [K, N] — weight matrix to quantize
/// `hessian_ptr`: NslTensor [K, K] (full Hessian) or [K] (diagonal), or 0 for
///   round-to-nearest (an identity Hessian)
/// `group_size`: quantization group size (typically 128)
/// `bits`: must be 4; any other width is a fatal error
///
/// Returns pointer to AwqPackedWeight.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gptq_quantize(
    weight_ptr: i64,
    hessian_ptr: i64,
    group_size: i64,
    bits: i64,
) -> i64 {
    if weight_ptr == 0 { crate::nsl_log!(WARN, "gptq", "nsl_gptq_quantize: null weight tensor"); return 0; }
    let (data, k, n) = weight_matrix(weight_ptr, "nsl_gptq_quantize");

    let hessian = if hessian_ptr == 0 {
        vec![1.0f64; k] // diagonal identity → RTN
    } else {
        crate::quantize::logical_values_f64(hessian_ptr)
    };

    let group_size = usize::try_from(group_size).unwrap_or(0);
    let bits = usize::try_from(bits).unwrap_or(0);
    match gptq_quantize_cpu(&data, &hessian, k, n, group_size, bits) {
        Ok(packed) => Box::into_raw(Box::new(packed)) as i64,
        Err(e) => die_on(e, "nsl_gptq_quantize"),
    }
}

/// GPTQ quantize with explicit config (act_order, block_size, damp_percent).
///
/// `weight_ptr`: NslTensor [K, N]
/// `hessian_ptr`: NslTensor [K, K], or 0 for an identity Hessian
/// `group_size`: quantization group size
/// `bits`: must be 4; any other width is a fatal error
/// `act_order`: 1 = enable activation order, 0 = sequential
/// `block_size`: block size for blocked updates (0 = one block)
/// `damp_percent_bits`: f64 damping percent encoded as u64 bits
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gptq_quantize_ext(
    weight_ptr: i64,
    hessian_ptr: i64,
    group_size: i64,
    bits: i64,
    act_order: i64,
    block_size: i64,
    damp_percent_bits: i64,
) -> i64 {
    const WHO: &str = "nsl_gptq_quantize_ext";
    if weight_ptr == 0 { crate::nsl_log!(WARN, "gptq", "nsl_gptq_quantize_ext: null weight tensor"); return 0; }
    let (data, k, n) = weight_matrix(weight_ptr, WHO);

    let hessian = if hessian_ptr == 0 {
        // No Hessian → identity for RTN
        let mut h = vec![0.0; k * k];
        for i in 0..k { h[i * k + i] = 1.0; }
        h
    } else {
        crate::quantize::logical_values_f64(hessian_ptr)
    };

    if bits != 4 {
        die_on(GptqError::UnsupportedBits(usize::try_from(bits).unwrap_or(0)), WHO);
    }
    if group_size <= 0 {
        die_on(GptqError::ZeroGroupSize, WHO);
    }
    if hessian.len() != k * k {
        die_on(GptqError::HessianShape { len: hessian.len(), k }, WHO);
    }

    let damp = f64::from_bits(damp_percent_bits as u64);

    let config = GptqConfig {
        group_size: group_size as usize,
        damp_percent: if damp > 0.0 { damp } else { 0.01 },
        act_order: act_order != 0,
        block_size: usize::try_from(block_size).unwrap_or(0),
    };

    let packed = gptq_quantize_obq(&data, &hessian, k, n, &config);
    Box::into_raw(Box::new(packed)) as i64
}

/// GPTQ dequant-matmul (same kernel as AWQ — packed format is identical).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_gptq_matmul(
    input_ptr: i64,
    packed_ptr: i64,
    group_size: i64,
    _bits: i64,
) -> i64 {
    crate::awq::nsl_awq_matmul(input_ptr, packed_ptr, group_size)
}

/// Free a GPTQ packed weight.
#[unsafe(no_mangle)]
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
    // Reference problems
    //
    // Every number below marked "reference" was produced by a NumPy
    // implementation of GPTQ written separately from this file (Algorithm 1
    // of the paper: upper Cholesky factor of the inverse of the damped,
    // permuted Hessian; group parameters from the original weights). The
    // inputs come from a 64-bit linear congruential generator so that this
    // test and that script build the same matrices without sharing code.
    // -----------------------------------------------------------------------

    struct Lcg(u64);

    impl Lcg {
        /// Uniform in [-0.5, 0.5): the top 53 bits of the state, exactly.
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    /// A `[K, N]` weight and the Hessian of `M` correlated calibration rows
    /// whose first K/16 inputs carry 8x the energy of the rest.
    fn problem(k: usize, n: usize, m: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut g = Lcg(seed);
        let z: Vec<f64> = (0..m * k).map(|_| g.next()).collect();
        let c: Vec<f64> = (0..m).map(|_| g.next()).collect();
        let mut x = vec![0.0f64; m * k];
        for row in 0..m {
            for j in 0..k {
                let mut v = z[row * k + j]
                    + 0.8 * z[row * k + (j + 1) % k]
                    + 0.6 * z[row * k + (j + 2) % k]
                    + 1.5 * c[row];
                if j < k / 16 {
                    v *= 8.0;
                }
                x[row * k + j] = v;
            }
        }
        let w: Vec<f64> = (0..k * n).map(|_| g.next() * 0.1).collect();
        let mut acc = HessianAccumulator::new(k);
        acc.add_batch(&x, m);
        (w, acc.finalize())
    }

    /// GPTQ's objective: tr(D^T H D) with D = W - Q, the squared error of
    /// the layer's output over the calibration set.
    fn output_error(w: &[f64], q: &[f64], h: &[f64], k: usize, n: usize) -> f64 {
        let d: Vec<f64> = w.iter().zip(q).map(|(a, b)| a - b).collect();
        let mut total = 0.0;
        for out in 0..n {
            for i in 0..k {
                let mut hd = 0.0;
                for j in 0..k {
                    hd += h[i * k + j] * d[j * n + out];
                }
                total += d[i * n + out] * hd;
            }
        }
        total
    }

    fn close(got: f64, want: f64, rel: f64) -> bool {
        (got - want).abs() <= rel * want.abs()
    }

    fn obq(w: &[f64], h: &[f64], k: usize, n: usize, gs: usize, act_order: bool, block: usize) -> Vec<f64> {
        let config = GptqConfig { group_size: gs, act_order, block_size: block, ..Default::default() };
        let packed = gptq_quantize_obq(w, h, k, n, &config);
        let q = awq_dequantize_cpu(&packed);
        unsafe { awq_free_packed(&packed) };
        q
    }

    fn rtn(w: &[f64], k: usize, n: usize, gs: usize) -> Vec<f64> {
        let packed = awq_quantize_cpu(w, k, n, gs);
        let q = awq_dequantize_cpu(&packed);
        unsafe { awq_free_packed(&packed) };
        q
    }

    /// The generator here and the one in the reference script agree, or no
    /// later comparison means anything.
    #[test]
    fn the_test_problem_is_the_reference_problem() {
        let (w, h) = problem(64, 32, 512, 1);
        assert_eq!(w[0], -0.011088314501774332);
        assert_eq!(w[64 * 32 - 1], -0.018477066318992076);
        assert!(close(h[0], 25.830046560969617, 1e-12), "h[0,0] = {}", h[0]);
        assert!(close(h[63 * 64], 2.6185245421349674, 1e-12), "h[63,0] = {}", h[63 * 64]);
    }

    /// Output error against the reference, on three shapes: several groups
    /// per output (64/16, 128/32) and one group spanning every input (48/48).
    /// The code this replaced gave 1.05x to 1.25x the round-to-nearest error
    /// on problems like these; the reference gives about a fifth of it.
    #[test]
    fn output_error_matches_the_reference_implementation() {
        // (k, n, m, group, seed, reference RTN, reference act-order, reference sequential)
        let cases = [
            (64, 32, 512, 16, 1u64, 0.010142034764278201, 0.002045743846450716, 0.0020632745621537475),
            (128, 64, 1024, 32, 2, 0.04740712121858101, 0.010810898970812638, 0.009767283733525332),
            (48, 8, 256, 48, 3, 0.003007551342691984, 0.0004504171756264765, 0.00047619245053740514),
        ];
        for (k, n, m, gs, seed, want_rtn, want_act, want_seq) in cases {
            let (w, h) = problem(k, n, m, seed);
            let got_rtn = output_error(&w, &rtn(&w, k, n, gs), &h, k, n);
            let got_act = output_error(&w, &obq(&w, &h, k, n, gs, true, 0), &h, k, n);
            let got_seq = output_error(&w, &obq(&w, &h, k, n, gs, false, 0), &h, k, n);
            assert!(close(got_rtn, want_rtn, 1e-7), "k={k}: RTN {got_rtn} vs reference {want_rtn}");
            assert!(close(got_act, want_act, 1e-7), "k={k}: act-order {got_act} vs reference {want_act}");
            assert!(close(got_seq, want_seq, 1e-7), "k={k}: sequential {got_seq} vs reference {want_seq}");
        }
    }

    /// Individual decoded weights, against the reference.
    #[test]
    fn decoded_weights_match_the_reference_implementation() {
        let (k, n, gs) = (64, 32, 16);
        let (w, h) = problem(k, n, 512, 1);
        let q = obq(&w, &h, k, n, gs, true, 0);
        assert!(close(q[0], -0.012091352780624478, 1e-9), "q[0,0] = {}", q[0]);
        assert!(close(q[k * n - 1], -0.028749887656857216, 1e-9), "q[k-1,n-1] = {}", q[k * n - 1]);
        let mid = (k / 2) * n + n / 2;
        assert!(close(q[mid], 0.0027940373754497747, 1e-9), "q[k/2,n/2] = {}", q[mid]);
    }

    /// What GPTQ is for, and what it costs: the output error falls well
    /// below rounding, and the weight error RISES (reference: 2.56e-6 for
    /// rounding, 7.34e-6 for GPTQ). A test that asks GPTQ for a lower weight
    /// error than rounding is asking for the wrong algorithm.
    #[test]
    fn output_error_falls_and_weight_error_rises() {
        let (k, n, gs) = (64, 32, 16);
        let (w, h) = problem(k, n, 512, 1);
        let q_rtn = rtn(&w, k, n, gs);
        let q_gptq = obq(&w, &h, k, n, gs, true, 0);
        let e_rtn = output_error(&w, &q_rtn, &h, k, n);
        let e_gptq = output_error(&w, &q_gptq, &h, k, n);
        assert!(e_gptq < 0.25 * e_rtn, "output error {e_gptq} is not well below rounding's {e_rtn}");
        let (mse_rtn, mse_gptq) = (mse(&w, &q_rtn), mse(&w, &q_gptq));
        assert!(close(mse_rtn, 2.5560748021833003e-06, 1e-7), "RTN weight MSE {mse_rtn}");
        assert!(close(mse_gptq, 7.343155946651385e-06, 1e-7), "GPTQ weight MSE {mse_gptq}");
    }

    /// Deferring updates to the end of a block changes the order of the
    /// subtractions and nothing else.
    #[test]
    fn a_blocked_run_gives_the_unblocked_result() {
        let (k, n, gs) = (64, 32, 16);
        let (w, h) = problem(k, n, 512, 1);
        let whole = output_error(&w, &obq(&w, &h, k, n, gs, true, 0), &h, k, n);
        for block in [1, 8, 24, 64, 1000] {
            let blocked = output_error(&w, &obq(&w, &h, k, n, gs, true, block), &h, k, n);
            assert!(close(blocked, whole, 1e-9), "block {block}: {blocked} vs {whole}");
        }
    }

    /// Group parameters are those of the weights as given. They used to be
    /// recomputed from compensated weights and overwritten as the run went.
    #[test]
    fn group_parameters_come_from_the_original_weights() {
        let (k, n, gs) = (64, 32, 16);
        let (w, h) = problem(k, n, 512, 1);
        let config = GptqConfig { group_size: gs, ..Default::default() };
        let packed = gptq_quantize_obq(&w, &h, k, n, &config);
        let groups = k / gs;
        let scales = unsafe { std::slice::from_raw_parts(packed.scales, groups * n) };
        let zeros = unsafe { std::slice::from_raw_parts(packed.zeros, groups * n) };
        for g in 0..groups {
            for out in 0..n {
                let column: Vec<f64> = (g * gs..(g + 1) * gs).map(|r| w[r * n + out]).collect();
                let lo = column.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                assert_eq!(zeros[g * n + out], lo, "zero of group {g}, output {out}");
                assert_eq!(scales[g * n + out], (hi - lo) / 15.0, "scale of group {g}, output {out}");
            }
        }
        unsafe { awq_free_packed(&packed) };
    }

    /// With a diagonal Hessian no input is correlated with another, the
    /// Cholesky factor is diagonal, and nothing is compensated: the codes
    /// are exactly those of rounding.
    #[test]
    fn a_diagonal_hessian_gives_round_to_nearest() {
        let (k, n, gs) = (16, 8, 4);
        let mut g = Lcg(7);
        let w: Vec<f64> = (0..k * n).map(|_| g.next()).collect();
        let diag: Vec<f64> = (0..k).map(|i| 1.0 + i as f64).collect();
        let packed = gptq_quantize_cpu(&w, &diag, k, n, gs, 4).expect("diagonal Hessian");
        let q = awq_dequantize_cpu(&packed);
        unsafe { awq_free_packed(&packed) };
        assert_eq!(q, rtn(&w, k, n, gs));
    }

    #[test]
    fn a_bit_width_other_than_four_is_refused() {
        let w = vec![0.1f64; 16];
        let h = vec![1.0f64; 4];
        for bits in [0, 2, 3, 8, 16] {
            match gptq_quantize_cpu(&w, &h, 4, 4, 4, bits) {
                Err(GptqError::UnsupportedBits(b)) => assert_eq!(b, bits),
                Err(other) => panic!("bits={bits}: {other}"),
                Ok(_) => panic!("bits={bits} was accepted; the codes would be 4-bit"),
            }
        }
    }

    #[test]
    fn a_hessian_of_the_wrong_size_is_refused() {
        let w = vec![0.1f64; 16];
        for len in [0, 3, 5, 15, 17] {
            let h = vec![1.0f64; len];
            assert_eq!(
                gptq_quantize_cpu(&w, &h, 4, 4, 4, 4).err(),
                Some(GptqError::HessianShape { len, k: 4 }),
                "a Hessian of {len} elements for K=4"
            );
        }
        assert_eq!(gptq_quantize_cpu(&w, &[1.0; 4], 4, 4, 0, 4).err(), Some(GptqError::ZeroGroupSize));
    }

    /// A Hessian with a negative eigenvalue larger than the damping has no
    /// Cholesky factor; the run falls back to rounding instead of dividing
    /// by garbage.
    #[test]
    fn an_indefinite_hessian_falls_back_to_rounding() {
        let (k, n, gs) = (4, 4, 4);
        let w: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let mut h = vec![0.0f64; k * k];
        for i in 0..k {
            h[i * k + i] = 1.0;
        }
        h[1] = 5.0;
        h[k] = 5.0;
        assert_eq!(obq(&w, &h, k, n, gs, false, 0), rtn(&w, k, n, gs));
    }

    // -----------------------------------------------------------------------
    // Packed matmul
    // -----------------------------------------------------------------------

    /// The dequantizing matmul computes `x @ Q` for the decoded Q.
    #[test]
    fn packed_matmul_is_the_matmul_of_the_decoded_weight() {
        let (k, n, m) = (8, 4, 2);
        let weights: Vec<f64> = (0..32).map(|i| (i as f64 * 0.2).sin() * 0.5).collect();
        let input: Vec<f64> = (0..16).map(|i| (i as f64 * 0.3).cos()).collect();
        let mut hessian = vec![0.0; k * k];
        for i in 0..k {
            hessian[i * k + i] = 1.0 + i as f64;
            for j in 0..k {
                if i != j {
                    hessian[i * k + j] = 0.3;
                }
            }
        }
        let config = GptqConfig { group_size: k, ..Default::default() };
        let packed = gptq_quantize_obq(&weights, &hessian, k, n, &config);
        let decoded = awq_dequantize_cpu(&packed);
        let want = matmul_ref(&input, &decoded, m, k, n);
        let got = crate::awq::awq_matmul_cpu(&input, &packed, m);
        for (i, (&r, &g)) in want.iter().zip(got.iter()).enumerate() {
            assert!((r - g).abs() < 1e-12, "element {i}: {g} vs {r}");
        }
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

    /// The FFI entry point reads a strided weight through its strides: a
    /// transposed view quantizes to the same codes as the matrix it shows.
    #[test]
    fn the_ffi_entry_point_reads_a_transposed_weight_in_logical_order() {
        let (k, n) = (6usize, 4usize);
        let mut g = Lcg(11);
        let logical: Vec<f64> = (0..k * n).map(|_| g.next()).collect();

        // Store the transpose, [N, K], and hand over its transposed view.
        let stored = crate::cpu::create_tensor_with_shape_rs_dtype(&[n as i64, k as i64], 0);
        let st = NslTensor::from_ptr(stored);
        for i in 0..k {
            for j in 0..n {
                unsafe { *st.data_f64().add(j * k + i) = logical[i * n + j] };
            }
        }
        let view = crate::tensor::shape_ops::nsl_tensor_transpose(stored, 0, 1);

        let packed_ptr = nsl_gptq_quantize(view, 0, 3, 4);
        let packed = unsafe { &*(packed_ptr as *const AwqPackedWeight) };
        assert_eq!((packed.k, packed.n), (k as i64, n as i64));
        assert_eq!(awq_dequantize_cpu(packed), rtn(&logical, k, n, 3));

        nsl_gptq_free(packed_ptr);
        crate::tensor::nsl_tensor_free(view);
        crate::tensor::nsl_tensor_free(stored);
    }

    /// A `[B, T, K]` activation is B*T samples; a last dimension that is not
    /// K is refused with a status instead of an assertion.
    #[test]
    fn the_hessian_accumulator_takes_the_last_dimension_as_k() {
        let _lock = FFI_LOCK.lock().unwrap();
        assert_eq!(nsl_gptq_hessian_init(2), 0);
        let x = crate::cpu::create_tensor_with_shape_rs_dtype(&[1, 2, 2], 0);
        let xt = NslTensor::from_ptr(x);
        for (i, v) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            unsafe { *xt.data_f64().add(i) = *v };
        }
        assert_eq!(nsl_gptq_hessian_add_batch(x), 0);
        let wrong = crate::cpu::create_tensor_with_shape_rs_dtype(&[2, 3], 0);
        assert_eq!(nsl_gptq_hessian_add_batch(wrong), -3);
        let h_ptr = nsl_gptq_hessian_finalize();
        let h = NslTensor::from_ptr(h_ptr);
        // (1/2) * ([1,2]^T[1,2] + [3,4]^T[3,4]) = [5, 7; 7, 10]
        let got: Vec<f64> = (0..4).map(|i| unsafe { *h.data_f64().add(i) }).collect();
        assert_eq!(got, vec![5.0, 7.0, 7.0, 10.0]);
        crate::tensor::nsl_tensor_free(h_ptr);
        crate::tensor::nsl_tensor_free(wrong);
        crate::tensor::nsl_tensor_free(x);
    }
}
