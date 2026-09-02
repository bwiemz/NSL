//! M35: FP8 scale management, cast operations, and matmul FFI.

use std::collections::HashMap;
use std::ffi::c_void;

use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// FP8 constants
// ---------------------------------------------------------------------------

/// Maximum representable value for E4M3 format.
pub const FP8E4M3_MAX: f32 = 448.0;
/// Maximum representable value for E5M2 format.
pub const FP8E5M2_MAX: f32 = 57344.0;

/// FP8 format identifier for FFI.
pub const FP8_FORMAT_E4M3: i64 = 0;
pub const FP8_FORMAT_E5M2: i64 = 1;

// ---------------------------------------------------------------------------
// Thread-local scale table
// ---------------------------------------------------------------------------

/// Global per-tensor scale factors, keyed by tensor pointer (as i64).
/// Uses a Mutex-protected HashMap instead of thread_local to ensure scales
/// set on one thread (e.g., a worker) are visible on another (e.g., main thread
/// running backward pass).
static FP8_SCALES: std::sync::LazyLock<std::sync::Mutex<HashMap<i64, f32>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Register scale for an FP8 tensor.
pub fn set_fp8_scale(tensor_ptr: i64, scale: f32) {
    FP8_SCALES.lock().unwrap().insert(tensor_ptr, scale);
}

/// Retrieve scale (returns 1.0 if unregistered — safe default).
pub fn get_fp8_scale(tensor_ptr: i64) -> f32 {
    *FP8_SCALES.lock().unwrap().get(&tensor_ptr).unwrap_or(&1.0)
}

/// Remove scale entry (on tensor free).
pub fn remove_fp8_scale(tensor_ptr: i64) {
    FP8_SCALES.lock().unwrap().remove(&tensor_ptr);
}

// ---------------------------------------------------------------------------
// FP8 quantize / dequantize helpers
// ---------------------------------------------------------------------------

/// Compute optimal scale factor: max(abs(tensor)) / fp8_max.
pub fn compute_scale(data: &[f64], fp8_format: i64) -> f64 {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX as f64,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX as f64,
        _ => FP8E4M3_MAX as f64,
    };
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / fp8_max
    }
}

fn compute_scale_f32(data: &[f32], fp8_format: i64) -> f64 {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX,
        _ => FP8E4M3_MAX,
    };
    if max_abs == 0.0 {
        1.0
    } else {
        (max_abs / fp8_max) as f64
    }
}

/// Quantize a single f64 value to FP8 (simulated as clamped+scaled f64).
pub fn quantize_fp8(value: f64, scale: f64, fp8_format: i64) -> f64 {
    if scale == 0.0 {
        return 0.0;
    }
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX as f64,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX as f64,
        _ => FP8E4M3_MAX as f64,
    };
    let scaled = value / scale;
    let clamped = scaled.clamp(-fp8_max, fp8_max);
    let precision = match fp8_format {
        FP8_FORMAT_E4M3 => 0.125,
        FP8_FORMAT_E5M2 => 0.5,
        _ => 0.125,
    };
    (clamped / precision).round() * precision
}

/// Dequantize a simulated FP8 value back to f64.
pub fn dequantize_fp8(fp8_value: f64, scale: f64) -> f64 {
    fp8_value * scale
}

// ---------------------------------------------------------------------------
// Host staging guard
// ---------------------------------------------------------------------------

/// Stage a tensor for a host-side flat read (`from_raw_parts` over `len`
/// elements). Returns `(readable_ptr, owned)`; when `owned` is true the
/// caller must free the returned tensor.
///
/// Every FFI in this file walks `t.data` as CPU memory in row-major order.
/// Two tensor states break that:
///  * `device != 0` — `t.data` is a CUDA device pointer, so the read is a
///    segfault (the tape-AD backward for `@fp8_compute` hands GPU-resident
///    gradients straight into `calibrate_gradient_scale` and
///    `nsl_fp8_cast`; same hazard class as the moe/ffi.rs device guard).
///  * non-contiguous views — `nsl_tensor_transpose` swaps strides over
///    shared storage, so a flat walk visits elements in STORAGE order while
///    the output gets fresh row-major strides stamped onto the view's
///    shape. That turns a transpose into a reshape: the E5M2 backward
///    quantized `B^T` in the wrong element order, so the tape-AD fp8 path
///    computed gradients from misplaced operands — wrong values, not
///    reduced precision.
///
/// Both are fixed by materializing: a D2H transfer for device tensors
/// (`nsl_tensor_to_device` materializes views itself before copying), an
/// explicit `nsl_tensor_contiguous` copy for CPU views.
///
/// dtypes outside {0=f64, 1=f32} would be read at the wrong element stride
/// (a heap OOB read for the 2-byte dtypes), and every caller feeds the
/// result into training arithmetic, so that case refuses loudly instead of
/// guessing.
fn stage_for_host_read(tensor_ptr: i64, ctx: &str) -> (i64, bool) {
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    if t.dtype != 0 && t.dtype != 1 {
        eprintln!(
            "nsl: {ctx}: unsupported dtype {} — the fp8 host path reads only f32 (1) and f64 (0)",
            t.dtype
        );
        std::process::abort();
    }
    // A zero-length read never dereferences, so nothing below applies.
    if t.len == 0 {
        return (tensor_ptr, false);
    }
    // `is_contiguous()` is the runtime-wide predicate, but it returns true
    // for EVERY rank-<=1 tensor regardless of strides, so a 1-D expand view
    // ([N] with stride 0 over a 1-element buffer) would pass straight
    // through to the flat read — an OOB read, the exact hazard this guard
    // exists to close. The rank-1 stride is checked explicitly here;
    // `nsl_tensor_contiguous` compares strides directly (not via
    // `is_contiguous`) and its GPU path is a stride-walking kernel, so it
    // materializes such views correctly on both devices. It must run BEFORE
    // any D2H transfer: `nsl_tensor_to_device` shares the blind predicate
    // and would flat-copy the same OOB range out of device memory.
    let flat_read_unsafe =
        !t.is_contiguous() || (t.ndim == 1 && t.len > 1 && unsafe { *t.strides } != 1);
    if flat_read_unsafe {
        let contig = crate::tensor::nsl_tensor_contiguous(tensor_ptr);
        let c = unsafe { &*(contig as *const NslTensor) };
        if c.device != 0 {
            let cpu = crate::tensor::nsl_tensor_to_device(contig, 0);
            crate::tensor::nsl_tensor_free(contig);
            return (cpu, true);
        }
        return (contig, true);
    }
    if t.device != 0 {
        return (crate::tensor::nsl_tensor_to_device(tensor_ptr, 0), true);
    }
    (tensor_ptr, false)
}

// ---------------------------------------------------------------------------
// FFI: FP8 cast
// ---------------------------------------------------------------------------

/// Cast a tensor to FP8 with given scale. If scale=0.0, auto-compute.
/// Returns a new tensor pointer with FP8-quantized-then-dequantized values,
/// on the same device as the input.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_cast(tensor_ptr: i64, target_dtype: i64, scale: f64) -> i64 {
    let orig_device = unsafe { (*(tensor_ptr as *const NslTensor)).device };
    let (src_ptr, staged) = stage_for_host_read(tensor_ptr, "nsl_fp8_cast");
    let t = unsafe { &*(src_ptr as *const NslTensor) };
    let len = t.len as usize;

    // Read source data as f32 (default runtime dtype)
    let src: Vec<f64> = if t.dtype == 1 {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        data.iter().map(|&v| v as f64).collect()
    } else {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        data.to_vec()
    };

    let actual_scale = if scale == 0.0 {
        compute_scale(&src, target_dtype)
    } else {
        scale
    };

    // Quantize to FP8 and dequantize back — simulates FP8 precision loss
    // Output as f32 to match runtime convention
    let result_data = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
    for (i, &v) in src.iter().enumerate() {
        let fp8 = quantize_fp8(v, actual_scale, target_dtype);
        let deq = dequantize_fp8(fp8, actual_scale);
        unsafe { *result_data.add(i) = deq as f32 };
    }

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    // Built on host (device 0); moved to the input's device below. The old
    // code stamped `t.device` directly onto this host allocation, producing
    // a GPU-labeled tensor whose data pointer was a CPU heap pointer.
    let out = Box::new(NslTensor::new(
        result_data as *mut c_void,
        shape,
        strides,
        t.ndim,
        t.len,
        0,
        1,
        1,
        0,
    ));
    let host_ptr = Box::into_raw(out) as i64;
    if staged {
        crate::tensor::nsl_tensor_free(src_ptr);
    }

    let out_ptr = move_to_original_device(host_ptr, orig_device);

    // Register scale for this tensor
    set_fp8_scale(out_ptr, actual_scale as f32);

    out_ptr
}

/// Compute optimal scale factor for FP8 conversion.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_compute_scale(tensor_ptr: i64, fp8_dtype: i64) -> f64 {
    let (src_ptr, staged) = stage_for_host_read(tensor_ptr, "nsl_fp8_compute_scale");
    let t = unsafe { &*(src_ptr as *const NslTensor) };
    let len = t.len as usize;
    let scale = if t.dtype == 1 {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        compute_scale_f32(data, fp8_dtype)
    } else {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        compute_scale(data, fp8_dtype)
    };
    if staged {
        crate::tensor::nsl_tensor_free(src_ptr);
    }
    scale
}

// ---------------------------------------------------------------------------
// FFI: FP8 matmul (CPU fallback)
// ---------------------------------------------------------------------------

/// CPU fallback for FP8 matmul: A[M,K] @ B[K,N] -> C[M,N]
/// Accumulates in f64 (simulating f32 accumulation on GPU).
pub fn fp8_matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// FP8 matmul FFI: both inputs are FP8-cast tensors. Output is f32 on CPU.
/// scale_a and scale_b are passed but on CPU, scale is already applied at cast time.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_matmul(
    a_ptr: i64,
    b_ptr: i64,
    _scale_a: f64,
    _scale_b: f64,
) -> i64 {
    // On CPU, tensors are already dequantized f32. Delegate to standard matmul.
    crate::tensor::nsl_tensor_matmul(a_ptr, b_ptr, 0)
}

/// FP8 matmul for training: performs the matmul and records TapeOp::Fp8MatMul
/// with scale factors so backward can use E5M2 GPU dispatch.
///
/// Called by codegen when @fp8_compute is used inside a grad() scope.
/// Auto-retrieves scales from FP8_SCALES table and computes K from tensor shape.
/// Same 2-arg signature as nsl_tensor_matmul for easy codegen swap.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_matmul_training(
    a_ptr: i64,
    b_ptr: i64,
    flags: u8,
) -> i64 {
    use crate::autodiff;
    use crate::tensor::fbip_flags::{relinquish_a, relinquish_b};
    use crate::tensor::NslTensor;
    use std::sync::atomic::Ordering;

    let relinq_a = relinquish_a(flags);
    let relinq_b = relinquish_b(flags);

    let is_recording = autodiff::is_recording();

    // If recording, temporarily pause to prevent nsl_tensor_matmul from
    // recording a TapeOp::MatMul (we'll record Fp8MatMul instead).
    if is_recording {
        autodiff::TAPE.with(|t| t.borrow_mut().pause_depth += 1);
    }

    // Do NOT forward the relinquish flags into nsl_tensor_matmul: we still
    // need to read shapes/refcounts from a_ptr/b_ptr for tape recording
    // below. We free A/B ourselves at the end.
    let result = crate::tensor::nsl_tensor_matmul(a_ptr, b_ptr, 0);

    if is_recording {
        // Resume recording
        autodiff::TAPE.with(|t| t.borrow_mut().pause_depth -= 1);

        let a_t = NslTensor::from_ptr(a_ptr);
        let b_t = NslTensor::from_ptr(b_ptr);

        // Auto-retrieve scales from the FP8 scale table
        let scale_a = get_fp8_scale(a_ptr);
        let scale_b = get_fp8_scale(b_ptr);

        // Compute K dimension from tensor shape: A is [..., M, K]
        let k_dim = if a_t.ndim >= 2 {
            unsafe { *a_t.shape.add(a_t.ndim as usize - 1) }
        } else {
            a_t.len
        };

        // Bump refcounts on saved tensors
        a_t.refcount.fetch_add(1, Ordering::SeqCst);
        b_t.refcount.fetch_add(1, Ordering::SeqCst);

        let device = a_t.device;

        autodiff::maybe_record(autodiff::TapeOp::Fp8MatMul {
            a: a_ptr,
            b: b_ptr,
            out: result,
            saved_a: a_ptr,
            saved_b: b_ptr,
            scale_a,
            scale_b,
            k_dim,
            device,
        });
    }

    // FBIP-3: fp8 matmul has no in-place path; the flag only enables eager
    // freeing of A and B after the matmul (and after tape recording has
    // consumed their metadata).
    if relinq_a { crate::tensor::nsl_tensor_free(a_ptr); }
    if relinq_b { crate::tensor::nsl_tensor_free(b_ptr); }
    result
}

// ---------------------------------------------------------------------------
// FP8 E5M2 backward: quantization and gradient scale
// ---------------------------------------------------------------------------

/// Compute fresh per-tensor scale for gradient tensors (no EMA).
/// Gradients change dramatically between steps, so we compute scale fresh each time.
/// Returns amax / FP8E5M2_MAX, or 1.0 if all zeros.
pub fn calibrate_gradient_scale(tensor_ptr: i64) -> f32 {
    // Guard first so the dtype refusal fires for empty tensors too (the
    // guard's own len==0 short-circuit makes the staging free); an empty
    // tensor then takes the early return below with `staged == false`.
    let (src_ptr, staged) = stage_for_host_read(tensor_ptr, "calibrate_gradient_scale");
    let t = NslTensor::from_ptr(src_ptr);
    let len = t.len as usize;
    if len == 0 {
        return 1.0;
    }
    let amax: f32 = if t.dtype == 1 {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        data.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
    } else {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        data.iter().map(|v| v.abs() as f32).fold(0.0f32, f32::max)
    };
    if staged {
        crate::tensor::nsl_tensor_free(src_ptr);
    }
    if amax == 0.0 { 1.0 } else { amax / FP8E5M2_MAX }
}

/// Quantize a tensor to E5M2 format for backward pass.
/// Input: f32/f64 tensor pointer, per-tensor scale.
/// Output: new f32 tensor with E5M2-quantized-then-dequantized values (simulating precision loss).
/// If scale is 0.0, auto-computes scale from data.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_quantize_e5m2(tensor_ptr: i64, scale: f64) -> i64 {
    nsl_fp8_cast(tensor_ptr, FP8_FORMAT_E5M2, scale)
}

/// Compute gradient scale for E5M2 backward quantization.
/// Returns scale = amax / FP8E5M2_MAX. No EMA — fresh per step.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_gradient_scale(tensor_ptr: i64) -> f64 {
    calibrate_gradient_scale(tensor_ptr) as f64
}

/// FP8 E5M2 matmul for backward pass.
/// Quantizes both inputs to E5M2 precision, then performs f32 matmul.
/// On CPU: quantize→dequantize→standard matmul (simulates E5M2 precision loss).
/// On GPU (sm_90+): would launch E5M2 MMA kernel (future: kernel caching).
///
/// Returns f32 output tensor (dequantized).
pub fn fp8_matmul_e5m2_backward(
    a_ptr: i64,
    b_ptr: i64,
    scale_a: f32,
    scale_b: f32,
) -> i64 {
    // Quantize both inputs to E5M2 precision (simulates precision loss)
    let a_e5m2 = nsl_fp8_cast(a_ptr, FP8_FORMAT_E5M2, scale_a as f64);
    let b_e5m2 = nsl_fp8_cast(b_ptr, FP8_FORMAT_E5M2, scale_b as f64);

    // Perform matmul on E5M2-quantized tensors (CPU: already dequantized f32)
    let result = crate::tensor::nsl_tensor_matmul(a_e5m2, b_e5m2, 0);

    // Free intermediate quantized tensors
    crate::tensor::nsl_tensor_free(a_e5m2);
    crate::tensor::nsl_tensor_free(b_e5m2);

    result
}

// ---------------------------------------------------------------------------
// FP8 kernel PTX cache (GPU path)
// ---------------------------------------------------------------------------

use std::collections::HashMap as StdHashMap;
use std::sync::{Mutex, OnceLock};

/// Cache key: (K dimension, format: 0=E4M3, 1=E5M2)
type Fp8KernelKey = (usize, i64);

/// Cache of compiled FP8 PTX strings, keyed by (K, format).
/// Backward K matches forward K, so kernels are reused across iterations.
static FP8_PTX_CACHE: OnceLock<Mutex<StdHashMap<Fp8KernelKey, String>>> = OnceLock::new();

/// Register a compiled FP8 PTX kernel in the cache.
/// Called by codegen/CLI when compiling an FP8 matmul.
#[allow(dead_code)]
pub fn cache_fp8_ptx(k: usize, format: i64, ptx: String) {
    let cache = FP8_PTX_CACHE.get_or_init(|| Mutex::new(StdHashMap::new()));
    cache.lock().unwrap().insert((k, format), ptx);
}

/// Retrieve a cached FP8 PTX kernel, if available.
#[allow(dead_code)]
pub fn get_cached_fp8_ptx(k: usize, format: i64) -> Option<String> {
    let cache = FP8_PTX_CACHE.get_or_init(|| Mutex::new(StdHashMap::new()));
    cache.lock().unwrap().get(&(k, format)).cloned()
}

/// FFI: Register compiled E5M2 backward PTX from codegen.
/// k_dim: inner dimension, ptx_ptr: null-terminated C string.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_cache_e5m2_ptx(k_dim: i64, ptx_ptr: i64) {
    let ptx_cstr = unsafe { std::ffi::CStr::from_ptr(ptx_ptr as *const std::ffi::c_char) };
    let ptx = ptx_cstr.to_string_lossy().into_owned();
    cache_fp8_ptx(k_dim as usize, FP8_FORMAT_E5M2, ptx);
}

// ---------------------------------------------------------------------------
// FP8 Calibration — running max with EMA
// ---------------------------------------------------------------------------

/// Calibration state for FP8 quantization.
/// Tracks running absolute maximum via exponential moving average (EMA)
/// to compute optimal per-tensor scale factors over multiple batches.
pub struct Fp8CalibrationState {
    /// Running maximum absolute value (EMA-smoothed)
    pub running_amax: f32,
    /// EMA decay factor (0.0-1.0, higher = more weight on history)
    pub ema_decay: f32,
    /// Number of batches seen
    pub num_samples: u64,
}

impl Fp8CalibrationState {
    pub fn new(ema_decay: f32) -> Self {
        Self {
            running_amax: 0.0,
            ema_decay,
            num_samples: 0,
        }
    }

    /// Update running max with a new batch of values.
    pub fn update(&mut self, data: &[f32]) {
        let batch_amax = data.iter()
            .map(|x| x.abs())
            .fold(0.0_f32, f32::max);

        if self.num_samples == 0 {
            self.running_amax = batch_amax;
        } else {
            self.running_amax = self.ema_decay * self.running_amax
                + (1.0 - self.ema_decay) * batch_amax;
        }
        self.num_samples += 1;
    }

    /// Compute the scale factor for quantization to FP8.
    /// scale = amax / fp8_max  (to be used as: quantized = value / scale)
    pub fn compute_scale_e4m3(&self) -> f32 {
        if self.running_amax == 0.0 { 1.0 } else { self.running_amax / FP8E4M3_MAX }
    }

    pub fn compute_scale_e5m2(&self) -> f32 {
        if self.running_amax == 0.0 { 1.0 } else { self.running_amax / FP8E5M2_MAX }
    }
}

/// Update calibration running max (EMA). Returns updated running_max as f64.
///
/// `tensor_ptr`: i64 pointer to an NslTensor (f32 data)
/// `running_max_ptr`: i64 pointer to a scalar tensor holding the running max
/// `momentum`: EMA decay factor (e.g., 0.999)
#[unsafe(no_mangle)]
pub extern "C" fn nsl_fp8_update_calibration(
    tensor_ptr: i64,
    running_max_ptr: i64,
    momentum: f64,
) -> f64 {
    // Guard first (dtype refusal must fire for empty tensors too; the
    // guard short-circuits len==0 without staging).
    let (src_ptr, staged) = stage_for_host_read(tensor_ptr, "nsl_fp8_update_calibration");
    let tensor = NslTensor::from_ptr(src_ptr);
    let len = tensor.len as usize;
    if len == 0 {
        return 0.0;
    }

    // Compute batch amax
    let mut batch_amax: f64 = 0.0;
    if tensor.dtype == 1 {
        for i in 0..len {
            let val = unsafe { *tensor.data_f32().add(i) };
            batch_amax = batch_amax.max(val.abs() as f64);
        }
    } else {
        for i in 0..len {
            let val = unsafe { *tensor.data_f64().add(i) };
            batch_amax = batch_amax.max(val.abs());
        }
    }
    if staged {
        crate::tensor::nsl_tensor_free(src_ptr);
    }

    // Update running max via EMA
    if running_max_ptr != 0 {
        let rm = NslTensor::from_ptr(running_max_ptr);
        // The running max is read AND written in place through host pointers.
        // A device-resident or non-{f32,f64} state tensor has no correct
        // host-side read-modify-write here; a raw dereference would segfault
        // (device) or read at the wrong stride (dtype). No in-tree caller
        // creates one — refuse loudly rather than corrupt calibration.
        if rm.device != 0 || (rm.dtype != 0 && rm.dtype != 1) {
            eprintln!(
                "nsl: nsl_fp8_update_calibration: running-max state must be a CPU f32/f64 \
                 tensor (device={}, dtype={}); keep FP8 calibration state on the host",
                rm.device, rm.dtype
            );
            std::process::abort();
        }
        let old_max = if rm.dtype == 1 {
            unsafe { *rm.data_f32() as f64 }
        } else {
            unsafe { *rm.data_f64() }
        };

        let new_max = if old_max == 0.0 {
            batch_amax
        } else {
            momentum * old_max + (1.0 - momentum) * batch_amax
        };

        // Write back
        if rm.dtype == 1 {
            unsafe { *rm.data_f32() = new_max as f32 };
        } else {
            unsafe { *rm.data_f64() = new_max };
        }

        new_max
    } else {
        batch_amax
    }
}

// ===========================================================================
// MXFP8: Per-block scaling with E8M0 scale factors (Blackwell)
// ===========================================================================

/// MXFP8 block size: 32 contiguous elements per scale factor.
pub const MXFP8_BLOCK_SIZE: usize = 32;

/// NVFP4 block size: 256 elements per scale factor (16x16).
pub const NVFP4_BLOCK_SIZE: usize = 256;

/// FP4 E2M1 max representable value.
pub const FP4E2M1_MAX: f32 = 6.0;

/// Scaling mode for FP8/FP4 quantization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fp8ScalingMode {
    /// Single scale for the entire tensor (Hopper H100).
    PerTensor,
    /// One E8M0 scale per block of `block_size` elements (Blackwell).
    PerBlock { block_size: usize },
    /// One scale per output channel.
    PerChannel,
}

/// FP4 format identifier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fp4Format {
    /// 2-bit exponent, 1-bit mantissa (Blackwell NVFP4).
    E2M1,
}

/// Layout constraint for FP8/FP4 operands in wgmma.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fp8Layout {
    /// K dimension contiguous — required for Blackwell wgmma.
    KMajor,
    /// M/N dimension contiguous — standard row-major.
    MnMajor,
}

// ---------------------------------------------------------------------------
// E8M0 scale factor encoding (8-bit exponent, no mantissa)
// ---------------------------------------------------------------------------

/// Encode a positive scale factor as E8M0 (pure power-of-2).
/// E8M0 stores only the IEEE-754 exponent byte: value = 2^(e - 127).
pub fn encode_e8m0(scale: f32) -> u8 {
    if scale <= 0.0 || !scale.is_finite() {
        return 127; // E8M0 for 1.0 (2^(127-127) = 2^0 = 1)
    }
    let bits = scale.to_bits();
    ((bits >> 23) & 0xFF) as u8
}

/// Decode an E8M0 scale byte back to f32.
/// value = 2^(e - 127).
pub fn decode_e8m0(e8m0: u8) -> f32 {
    f32::from_bits((e8m0 as u32) << 23)
}

// ---------------------------------------------------------------------------
// MXFP8 per-block quantization
// ---------------------------------------------------------------------------

/// Result of MXFP8 block quantization.
pub struct MxFp8Quantized {
    /// Quantized FP8 values (simulated as f32 with FP8 precision loss).
    pub data: Vec<f32>,
    /// E8M0 scale factors, one per block of MXFP8_BLOCK_SIZE elements.
    pub scales: Vec<u8>,
    /// Number of elements.
    pub len: usize,
}

/// Quantize a tensor using MXFP8 per-block scaling.
/// Each block of `block_size` contiguous elements gets its own E8M0 scale.
pub fn quantize_mxfp8(data: &[f32], block_size: usize, fp8_format: i64) -> MxFp8Quantized {
    let n = data.len();
    let num_blocks = n.div_ceil(block_size);
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX,
        _ => FP8E4M3_MAX,
    };
    let precision = match fp8_format {
        FP8_FORMAT_E4M3 => 0.125f32,
        FP8_FORMAT_E5M2 => 0.5f32,
        _ => 0.125f32,
    };

    let mut quantized = vec![0.0f32; n];
    let mut scales = Vec::with_capacity(num_blocks);

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n);
        let block = &data[start..end];

        // Compute block-local absmax
        let amax = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / fp8_max };

        // Encode as E8M0 (power-of-2 approximation)
        let e8m0 = encode_e8m0(scale);
        let actual_scale = decode_e8m0(e8m0);
        scales.push(e8m0);

        // Quantize each element in the block
        for (i, &v) in block.iter().enumerate() {
            let scaled = v / actual_scale;
            let clamped = scaled.clamp(-fp8_max, fp8_max);
            let quantized_val = (clamped / precision).round() * precision;
            quantized[start + i] = quantized_val * actual_scale;
        }
    }

    MxFp8Quantized { data: quantized, scales, len: n }
}

/// FFI: Quantize tensor with MXFP8 per-block scaling.
/// Returns a new tensor with quantized-then-dequantized values (simulates precision loss).
/// The per-block E8M0 scale factors are computed internally and DROPPED at
/// this boundary — only the dequantized data survives. (This comment used to
/// claim an `nsl_mxfp8_get_scales` accessor; no such symbol has ever existed.)
#[unsafe(no_mangle)]
pub extern "C" fn nsl_mxfp8_quantize(tensor_ptr: i64, block_size: i64, fp8_format: i64) -> i64 {
    let orig_device = NslTensor::from_ptr(tensor_ptr).device;
    let (src_ptr, staged) = stage_for_host_read(tensor_ptr, "nsl_mxfp8_quantize");
    let t = NslTensor::from_ptr(src_ptr);
    let len = t.len as usize;
    let bs = block_size as usize;

    let src: Vec<f32> = if t.dtype == 0 {
        unsafe { std::slice::from_raw_parts(t.data as *const f64, len) }
            .iter().map(|&v| v as f32).collect()
    } else {
        unsafe { std::slice::from_raw_parts(t.data as *const f32, len) }.to_vec()
    };

    let result = quantize_mxfp8(&src, bs, fp8_format);

    // Store quantized data as f32 tensor
    let out_data = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
    for (i, &v) in result.data.iter().enumerate() {
        unsafe { *out_data.add(i) = v };
    }

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let out = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape,
        strides,
        t.ndim,
        t.len,
        0,
        1,
        1,
        0,
    ));
    let host_ptr = Box::into_raw(out) as i64;
    if staged {
        crate::tensor::nsl_tensor_free(src_ptr);
    }
    move_to_original_device(host_ptr, orig_device)
}

/// H2D-move a freshly built host result back to the input's device (no-op
/// for CPU inputs). Keeps the quantizer FFIs from stamping a device label
/// onto a host allocation.
fn move_to_original_device(host_ptr: i64, orig_device: u8) -> i64 {
    if orig_device == 0 {
        return host_ptr;
    }
    let moved = crate::tensor::nsl_tensor_to_device(host_ptr, orig_device as i64);
    crate::tensor::nsl_tensor_free(host_ptr);
    moved
}

// ===========================================================================
// NVFP4: 4-bit E2M1 with Hadamard preprocessing (Blackwell)
// ===========================================================================

// ---------------------------------------------------------------------------
// Fast Walsh-Hadamard Transform (FWHT)
// ---------------------------------------------------------------------------

/// In-place Fast Walsh-Hadamard Transform.
/// Input length must be a power of 2. Normalizes by 1/sqrt(n).
pub fn fast_hadamard_transform(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "FWHT requires power-of-2 length, got {n}");

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }

    // Normalize
    let norm = 1.0 / (n as f32).sqrt();
    x.iter_mut().for_each(|v| *v *= norm);
}

/// In-place inverse Fast Walsh-Hadamard Transform.
/// FWHT is self-inverse (orthogonal), so inverse = forward (with same normalization).
pub fn inverse_hadamard_transform(x: &mut [f32]) {
    fast_hadamard_transform(x);
}

// ---------------------------------------------------------------------------
// FP4 E2M1 quantization
// ---------------------------------------------------------------------------

/// Quantize a single f32 value to FP4 E2M1 (simulated as f32 with E2M1 precision loss).
/// E2M1 can represent: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives).
fn quantize_fp4_e2m1(value: f32, scale: f32) -> f32 {
    if scale == 0.0 { return 0.0; }
    let scaled = value / scale;
    let clamped = scaled.clamp(-FP4E2M1_MAX, FP4E2M1_MAX);
    // E2M1 representable values (positive): 0, 0.5, 1, 1.5, 2, 3, 4, 6
    // Round to nearest representable value
    let abs_val = clamped.abs();
    let sign = clamped.signum();
    let quantized = if abs_val < 0.25 { 0.0 }
        else if abs_val < 0.75 { 0.5 }
        else if abs_val < 1.25 { 1.0 }
        else if abs_val < 1.75 { 1.5 }
        else if abs_val < 2.5 { 2.0 }
        else if abs_val < 3.5 { 3.0 }
        else if abs_val < 5.0 { 4.0 }
        else { 6.0 };
    sign * quantized * scale
}

/// Result of NVFP4 quantization.
pub struct NvFp4Quantized {
    /// Quantized FP4 values (simulated as f32 with E2M1 precision loss).
    pub data: Vec<f32>,
    /// Scale factors, one per block of NVFP4_BLOCK_SIZE elements.
    pub block_scales: Vec<f32>,
    /// Number of elements.
    pub len: usize,
    /// Whether Hadamard preprocessing was applied.
    pub hadamard_applied: bool,
}

/// Quantize a tensor using NVFP4 (E2M1) with optional Hadamard preprocessing.
///
/// Pipeline:
/// 1. Apply Hadamard transform (if enabled) — smears outliers across dimensions
/// 2. Compute per-block scale factors (blocks of NVFP4_BLOCK_SIZE)
/// 3. Quantize each element to E2M1
pub fn quantize_nvfp4(data: &[f32], block_size: usize, apply_hadamard: bool) -> NvFp4Quantized {
    let n = data.len();
    let mut working = data.to_vec();

    // Step 1: Hadamard preprocessing (optional)
    let hadamard_applied = if apply_hadamard && n.is_power_of_two() && n >= 2 {
        fast_hadamard_transform(&mut working);
        true
    } else {
        false
    };

    // Step 2: Per-block quantization
    let num_blocks = n.div_ceil(block_size);
    let mut quantized = vec![0.0f32; n];
    let mut block_scales = Vec::with_capacity(num_blocks);

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(n);
        let block = &working[start..end];

        let amax = block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if amax == 0.0 { 1.0 } else { amax / FP4E2M1_MAX };
        block_scales.push(scale);

        for (i, &v) in block.iter().enumerate() {
            quantized[start + i] = quantize_fp4_e2m1(v, scale);
        }
    }

    NvFp4Quantized { data: quantized, block_scales, len: n, hadamard_applied }
}

/// Dequantize NVFP4 data back to f32.
/// If Hadamard was applied during quantization, applies inverse Hadamard.
pub fn dequantize_nvfp4(quantized: &NvFp4Quantized) -> Vec<f32> {
    let mut result = quantized.data.clone();
    if quantized.hadamard_applied && result.len().is_power_of_two() {
        inverse_hadamard_transform(&mut result);
    }
    result
}

/// FFI: Quantize tensor with NVFP4 (E2M1) and optional Hadamard preprocessing.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_nvfp4_quantize(tensor_ptr: i64, block_size: i64, apply_hadamard: i64) -> i64 {
    let orig_device = NslTensor::from_ptr(tensor_ptr).device;
    let (src_ptr, staged) = stage_for_host_read(tensor_ptr, "nsl_nvfp4_quantize");
    let t = NslTensor::from_ptr(src_ptr);
    let len = t.len as usize;
    let bs = block_size as usize;

    let src: Vec<f32> = if t.dtype == 0 {
        unsafe { std::slice::from_raw_parts(t.data as *const f64, len) }
            .iter().map(|&v| v as f32).collect()
    } else {
        unsafe { std::slice::from_raw_parts(t.data as *const f32, len) }.to_vec()
    };

    let result = quantize_nvfp4(&src, bs, apply_hadamard != 0);

    let out_data = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
    for (i, &v) in result.data.iter().enumerate() {
        unsafe { *out_data.add(i) = v };
    }

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let out = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape,
        strides,
        t.ndim,
        t.len,
        0,
        1,
        1,
        0,
    ));
    let host_ptr = Box::into_raw(out) as i64;
    if staged {
        crate::tensor::nsl_tensor_free(src_ptr);
    }
    move_to_original_device(host_ptr, orig_device)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_table_set_get() {
        set_fp8_scale(42, 0.5);
        assert_eq!(get_fp8_scale(42), 0.5);
        assert_eq!(get_fp8_scale(999), 1.0); // default
        remove_fp8_scale(42);
        assert_eq!(get_fp8_scale(42), 1.0); // removed
    }

    #[test]
    fn test_compute_scale_e4m3() {
        let data = vec![100.0, -200.0, 50.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);
        assert!((scale - 200.0 / 448.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_scale_e5m2() {
        let data = vec![100.0, -200.0, 50.0];
        let scale = compute_scale(&data, FP8_FORMAT_E5M2);
        assert!((scale - 200.0 / 57344.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_scale_zero() {
        let data = vec![0.0, 0.0, 0.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let value = 1.5;
        let scale = 0.01;
        let fp8 = quantize_fp8(value, scale, FP8_FORMAT_E4M3);
        let recovered = dequantize_fp8(fp8, scale);
        assert!((recovered - value).abs() < scale * 0.125 + 1e-10);
    }

    #[test]
    fn test_clamping() {
        let value = 1000.0;
        let scale = 1.0;
        let fp8 = quantize_fp8(value, scale, FP8_FORMAT_E4M3);
        assert_eq!(fp8, 448.0); // clamped to E4M3 max
    }

    #[test]
    fn test_fp8_cast_ffi_auto_scale() {
        let data = vec![1.0f64, 2.0, -3.0, 4.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);

        // Quantize each element and verify round-trip
        for &v in &data {
            let quant = quantize_fp8(v, scale, FP8_FORMAT_E4M3);
            let recovered = dequantize_fp8(quant, scale);
            let rel_error = if v.abs() > 1e-10 {
                (recovered - v).abs() / v.abs()
            } else {
                0.0
            };
            assert!(
                rel_error < 0.01,
                "FP8 E4M3 relative error {} too high for value {}",
                rel_error,
                v
            );
        }
    }

    #[test]
    fn test_fp8_matmul_cpu() {
        // 2x2 matmul: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let scale_a = compute_scale(&a, FP8_FORMAT_E4M3);
        let scale_b = compute_scale(&b, FP8_FORMAT_E4M3);

        let a_fp8: Vec<f64> = a
            .iter()
            .map(|&v| dequantize_fp8(quantize_fp8(v, scale_a, FP8_FORMAT_E4M3), scale_a))
            .collect();
        let b_fp8: Vec<f64> = b
            .iter()
            .map(|&v| dequantize_fp8(quantize_fp8(v, scale_b, FP8_FORMAT_E4M3), scale_b))
            .collect();

        let result = fp8_matmul_cpu(&a_fp8, &b_fp8, 2, 2, 2);

        assert!((result[0] - 19.0).abs() < 1.0);
        assert!((result[1] - 22.0).abs() < 1.0);
        assert!((result[2] - 43.0).abs() < 2.0);
        assert!((result[3] - 50.0).abs() < 2.0);
    }

    #[test]
    fn test_fp8_calibration_running_max() {
        let mut cal = Fp8CalibrationState::new(0.9);

        // First batch: max abs = 100.0
        cal.update(&[50.0, -100.0, 25.0]);
        assert_eq!(cal.running_amax, 100.0);
        assert_eq!(cal.num_samples, 1);

        // Second batch: max abs = 200.0, EMA = 0.9 * 100 + 0.1 * 200 = 110.0
        cal.update(&[-200.0, 10.0]);
        assert!((cal.running_amax - 110.0).abs() < 1e-5);
        assert_eq!(cal.num_samples, 2);

        // Scale for E4M3: 110 / 448 = 0.2455...
        let scale = cal.compute_scale_e4m3();
        assert!((scale - 110.0 / 448.0).abs() < 1e-5);
    }

    #[test]
    fn test_fp8_calibration_zero_data() {
        let mut cal = Fp8CalibrationState::new(0.9);
        cal.update(&[0.0, 0.0, 0.0]);
        assert_eq!(cal.compute_scale_e4m3(), 1.0); // safe default
        assert_eq!(cal.compute_scale_e5m2(), 1.0);
    }

    #[test]
    fn test_fp8_calibration_single_value() {
        let mut cal = Fp8CalibrationState::new(1.0); // no decay
        cal.update(&[448.0]);
        assert_eq!(cal.compute_scale_e4m3(), 1.0); // amax == fp8_max → scale = 1.0
    }

    #[test]
    fn test_e5m2_quantize_roundtrip() {
        // E5M2 has precision 0.5, so max round-trip error per value is 0.5 * scale
        let values = vec![1.5, -3.0, 0.25, 100.0, -200.0];
        let scale = compute_scale(&values, FP8_FORMAT_E5M2);
        for &v in &values {
            let quant = quantize_fp8(v, scale, FP8_FORMAT_E5M2);
            let recovered = dequantize_fp8(quant, scale);
            let max_err = 0.5 * scale;
            assert!(
                (recovered - v).abs() <= max_err + 1e-10,
                "E5M2 round-trip error {} > max {} for value {} (scale={})",
                (recovered - v).abs(), max_err, v, scale
            );
        }
    }

    #[test]
    fn test_e5m2_wider_range() {
        // E5M2 max is 57344, much larger than E4M3's 448
        // Values near 1000 should survive E5M2 but would clamp hard in E4M3
        let value = 1000.0;
        let scale_e5m2 = 1.0; // scale = 1.0 — value is well within E5M2 range
        let quant = quantize_fp8(value, scale_e5m2, FP8_FORMAT_E5M2);
        let recovered = dequantize_fp8(quant, scale_e5m2);
        assert!((recovered - value).abs() < 1.0, "E5M2 should handle value={}", value);

        // Same value in E4M3 would clamp to 448
        let quant_e4m3 = quantize_fp8(value, 1.0, FP8_FORMAT_E4M3);
        assert_eq!(quant_e4m3, 448.0, "E4M3 should clamp 1000 to 448");
    }

    #[test]
    fn test_calibrate_gradient_scale_fresh() {
        // Gradient scale should be computed fresh (no EMA), just amax / FP8E5M2_MAX
        use std::sync::atomic::AtomicI64;
        let data = vec![0.5f32, -1.0, 0.25, 0.75];
        let data_ptr = crate::memory::checked_alloc(data.len() * std::mem::size_of::<f32>()) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            unsafe { *data_ptr.add(i) = v };
        }
        let shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = 4 };
        let strides = crate::tensor::NslTensor::compute_strides(shape, 1);
        let tensor = Box::new(crate::tensor::NslTensor::new(
            data_ptr as *mut std::ffi::c_void,
            shape,
            strides,
            1,
            4,
            0,
            1,
            1,
            0,
        ));
        let ptr = Box::into_raw(tensor) as i64;
        let scale = calibrate_gradient_scale(ptr);
        let expected = 1.0 / FP8E5M2_MAX; // amax=1.0
        assert!((scale - expected).abs() < 1e-10, "gradient scale={} expected={}", scale, expected);
        crate::tensor::nsl_tensor_free(ptr);
    }

    #[test]
    fn test_fp8_e5m2_backward_numerical_correctness() {
        // Compare FP8 E5M2 backward gradients against f32 backward.
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // Forward: C = A @ B
        // Backward: grad_A = G @ B^T, grad_B = A^T @ G (where G = ones)
        //
        // E5M2 should match f32 within tolerance (E5M2 precision = 0.5).

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let g = vec![1.0, 1.0, 1.0, 1.0]; // ones gradient

        // f32 reference backward:
        // grad_A = G @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        // grad_B = A^T @ G = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        let ref_grad_a = vec![11.0, 15.0, 11.0, 15.0];
        let ref_grad_b = vec![4.0, 4.0, 6.0, 6.0];

        // E5M2 backward (simulate precision loss)
        let scale_g = compute_scale(&g.iter().map(|&v| v as f64).collect::<Vec<_>>(), FP8_FORMAT_E5M2);
        let scale_b = compute_scale(&b.iter().map(|&v| v as f64).collect::<Vec<_>>(), FP8_FORMAT_E5M2);
        let scale_a = compute_scale(&a.iter().map(|&v| v as f64).collect::<Vec<_>>(), FP8_FORMAT_E5M2);

        // Quantize G and B^T for grad_A = G_e5m2 @ B_e5m2^T
        let g_e5m2: Vec<f64> = g.iter()
            .map(|&v| dequantize_fp8(quantize_fp8(v as f64, scale_g, FP8_FORMAT_E5M2), scale_g))
            .collect();
        let b_t = vec![b[0], b[2], b[1], b[3]]; // transpose
        let bt_e5m2: Vec<f64> = b_t.iter()
            .map(|&v| dequantize_fp8(quantize_fp8(v as f64, scale_b, FP8_FORMAT_E5M2), scale_b))
            .collect();
        let e5m2_grad_a = fp8_matmul_cpu(&g_e5m2, &bt_e5m2, 2, 2, 2);

        // Quantize A^T and G for grad_B = A_e5m2^T @ G_e5m2
        let a_t = vec![a[0], a[2], a[1], a[3]]; // transpose
        let at_e5m2: Vec<f64> = a_t.iter()
            .map(|&v| dequantize_fp8(quantize_fp8(v as f64, scale_a, FP8_FORMAT_E5M2), scale_a))
            .collect();
        let e5m2_grad_b = fp8_matmul_cpu(&at_e5m2, &g_e5m2, 2, 2, 2);

        // Check relative error ≤ 1% for small matrices
        for (i, (&ref_val, &e5m2_val)) in ref_grad_a.iter().zip(e5m2_grad_a.iter()).enumerate() {
            let rel_err = (e5m2_val - ref_val).abs() / ref_val.abs().max(1e-10);
            assert!(
                rel_err < 0.01,
                "grad_A[{}]: E5M2={} vs f32={}, rel_err={:.4}%",
                i, e5m2_val, ref_val, rel_err * 100.0
            );
        }
        for (i, (&ref_val, &e5m2_val)) in ref_grad_b.iter().zip(e5m2_grad_b.iter()).enumerate() {
            let rel_err = (e5m2_val - ref_val).abs() / ref_val.abs().max(1e-10);
            assert!(
                rel_err < 0.01,
                "grad_B[{}]: E5M2={} vs f32={}, rel_err={:.4}%",
                i, e5m2_val, ref_val, rel_err * 100.0
            );
        }
    }

    #[test]
    fn test_fp8_gradient_scale_per_step() {
        // Gradient scales should be computed fresh each step (no EMA).
        // Different gradient tensors should get different scales.
        use std::sync::atomic::AtomicI64;

        let make_tensor = |data: &[f32]| -> i64 {
            let len = data.len();
            let data_ptr = crate::memory::checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                unsafe { *data_ptr.add(i) = v };
            }
            let shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
            unsafe { *shape = len as i64 };
            let strides = crate::tensor::NslTensor::compute_strides(shape, 1);
            let tensor = Box::new(crate::tensor::NslTensor::new(
                data_ptr as *mut std::ffi::c_void,
                shape,
                strides,
                1,
                len as i64,
                0,
                1,
                1,
                0,
            ));
            Box::into_raw(tensor) as i64
        };

        // Step 1: small gradients
        let grad1 = make_tensor(&[0.1, -0.2, 0.15]);
        let scale1 = calibrate_gradient_scale(grad1);
        assert!((scale1 - 0.2 / FP8E5M2_MAX).abs() < 1e-10);

        // Step 2: larger gradients
        let grad2 = make_tensor(&[10.0, -20.0, 5.0]);
        let scale2 = calibrate_gradient_scale(grad2);
        assert!((scale2 - 20.0 / FP8E5M2_MAX).abs() < 1e-10);

        // Step 3: different gradients again
        let grad3 = make_tensor(&[1.0, -0.5, 0.75]);
        let scale3 = calibrate_gradient_scale(grad3);
        assert!((scale3 - 1.0 / FP8E5M2_MAX).abs() < 1e-10);

        // Verify all three scales are different (fresh per step)
        assert_ne!(scale1, scale2, "Step 1 and 2 should have different scales");
        assert_ne!(scale2, scale3, "Step 2 and 3 should have different scales");

        crate::tensor::nsl_tensor_free(grad1);
        crate::tensor::nsl_tensor_free(grad2);
        crate::tensor::nsl_tensor_free(grad3);
    }

    #[test]
    fn test_fp8_e5m2_matmul_backward_function() {
        // Test fp8_matmul_e5m2_backward directly
        use std::sync::atomic::AtomicI64;

        let make_2d_tensor = |data: &[f32], rows: i64, cols: i64| -> i64 {
            let len = data.len();
            let data_ptr = crate::memory::checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                unsafe { *data_ptr.add(i) = v };
            }
            let shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
            unsafe { *shape = rows; *shape.add(1) = cols };
            let strides = crate::tensor::NslTensor::compute_strides(shape, 2);
            let tensor = Box::new(crate::tensor::NslTensor::new(
                data_ptr as *mut std::ffi::c_void,
                shape,
                strides,
                2,
                len as i64,
                0,
                1,
                1,
                0,
            ));
            Box::into_raw(tensor) as i64
        };

        // A [2,3] = [[1,2,3],[4,5,6]], B [3,2] = [[7,8],[9,10],[11,12]]
        let a = make_2d_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = make_2d_tensor(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);

        let scale_a = calibrate_gradient_scale(a);
        let scale_b = calibrate_gradient_scale(b);

        let result = fp8_matmul_e5m2_backward(a, b, scale_a, scale_b);

        // Expected: A @ B = [[58,64],[139,154]]
        let r = crate::tensor::NslTensor::from_ptr(result);
        assert_eq!(r.ndim, 2);
        let rows = unsafe { *r.shape } as usize;
        let cols = unsafe { *r.shape.add(1) } as usize;
        assert_eq!(rows, 2);
        assert_eq!(cols, 2);

        // Check values (E5M2 precision loss is small for these magnitudes)
        let v00 = unsafe { *r.data_f32() } as f64;
        let v01 = unsafe { *r.data_f32().add(1) } as f64;
        let v10 = unsafe { *r.data_f32().add(2) } as f64;
        let v11 = unsafe { *r.data_f32().add(3) } as f64;

        assert!((v00 - 58.0).abs() < 2.0, "C[0,0]={} expected ~58", v00);
        assert!((v01 - 64.0).abs() < 2.0, "C[0,1]={} expected ~64", v01);
        assert!((v10 - 139.0).abs() < 3.0, "C[1,0]={} expected ~139", v10);
        assert!((v11 - 154.0).abs() < 3.0, "C[1,1]={} expected ~154", v11);

        crate::tensor::nsl_tensor_free(result);
        crate::tensor::nsl_tensor_free(a);
        crate::tensor::nsl_tensor_free(b);
    }

    #[test]
    fn test_fp8_ptx_cache() {
        // Test the kernel PTX cache
        cache_fp8_ptx(128, FP8_FORMAT_E5M2, "test_e5m2_ptx_k128".to_string());
        cache_fp8_ptx(64, FP8_FORMAT_E4M3, "test_e4m3_ptx_k64".to_string());

        assert_eq!(get_cached_fp8_ptx(128, FP8_FORMAT_E5M2), Some("test_e5m2_ptx_k128".to_string()));
        assert_eq!(get_cached_fp8_ptx(64, FP8_FORMAT_E4M3), Some("test_e4m3_ptx_k64".to_string()));
        assert_eq!(get_cached_fp8_ptx(256, FP8_FORMAT_E5M2), None); // not cached
    }

    // ── MXFP8 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_e8m0_encode_decode_roundtrip() {
        // E8M0 encodes power-of-2 scale factors
        for &scale in &[1.0f32, 2.0, 4.0, 0.5, 0.25, 128.0, 0.0078125] {
            let encoded = encode_e8m0(scale);
            let decoded = decode_e8m0(encoded);
            // E8M0 only represents powers of 2, so roundtrip is exact for powers of 2
            assert_eq!(decoded, scale, "E8M0 roundtrip failed for {scale}");
        }
    }

    #[test]
    fn test_e8m0_non_power_of_2() {
        // Non-power-of-2 values get rounded to nearest power of 2 (lower)
        let encoded = encode_e8m0(3.0);
        let decoded = decode_e8m0(encoded);
        // 3.0 has exponent 1 (2^1 = 2), so E8M0 decodes to 2.0
        assert_eq!(decoded, 2.0);
    }

    #[test]
    fn test_e8m0_edge_cases() {
        // Zero → default (1.0)
        assert_eq!(decode_e8m0(encode_e8m0(0.0)), 1.0);
        // NaN → default
        assert_eq!(decode_e8m0(encode_e8m0(f32::NAN)), 1.0);
        // Infinity → default
        assert_eq!(decode_e8m0(encode_e8m0(f32::INFINITY)), 1.0);
    }

    #[test]
    fn test_mxfp8_block_quantization_basic() {
        // 64 elements = 2 blocks of 32
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let result = quantize_mxfp8(&data, MXFP8_BLOCK_SIZE, FP8_FORMAT_E4M3);

        assert_eq!(result.len, 64);
        assert_eq!(result.scales.len(), 2); // 64/32 = 2 blocks
        assert_eq!(result.data.len(), 64);

        // Verify scales are different for different blocks
        // Block 0: max = 3.1, Block 1: max = 6.3
        // They should have different E8M0 scales
        assert_ne!(result.scales[0], result.scales[1]);
    }

    #[test]
    fn test_mxfp8_better_than_per_tensor_on_outliers() {
        // Create tensor with huge outlier that dominates per-tensor scale
        let mut data = vec![0.01f32; 64];
        data[63] = 10000.0; // huge outlier in block 1 — per-tensor scale = 10000/448 ≈ 22.3

        // Per-tensor: scale dominated by outlier, small values in block 0 crushed to zero
        let per_tensor_scale = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max) / FP8E4M3_MAX;
        let per_tensor_err: f32 = data[..32].iter()
            .map(|&v| {
                let q = (v / per_tensor_scale / 0.125).round() * 0.125 * per_tensor_scale;
                (v - q).abs()
            })
            .sum::<f32>() / 32.0;

        // MXFP8: block 0 has its own scale optimized for 0.01, unaffected by outlier
        let result = quantize_mxfp8(&data, MXFP8_BLOCK_SIZE, FP8_FORMAT_E4M3);
        let mxfp8_err: f32 = data[..32].iter().zip(result.data[..32].iter())
            .map(|(&orig, &quant)| (orig - quant).abs())
            .sum::<f32>() / 32.0;

        // MXFP8 should have lower error for block 0
        assert!(mxfp8_err < per_tensor_err,
            "MXFP8 err ({mxfp8_err}) should be less than per-tensor err ({per_tensor_err})");
    }

    #[test]
    fn test_mxfp8_partial_last_block() {
        // 50 elements = 1 full block + 1 partial block
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let result = quantize_mxfp8(&data, MXFP8_BLOCK_SIZE, FP8_FORMAT_E4M3);

        assert_eq!(result.len, 50);
        assert_eq!(result.scales.len(), 2); // ceil(50/32) = 2
    }

    // ── NVFP4 tests ─────────────────────────────────────────────────────

    #[test]
    fn test_hadamard_orthogonality() {
        // H @ H^T = I (for normalized Hadamard)
        // Applying FWHT twice should give back the original
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();

        fast_hadamard_transform(&mut data);
        // After one transform, values should be different
        assert_ne!(data, original);

        // Second transform = inverse (FWHT is self-inverse)
        fast_hadamard_transform(&mut data);

        // Should recover original within floating point tolerance
        for (a, b) in original.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-5, "Hadamard not orthogonal: {a} vs {b}");
        }
    }

    #[test]
    fn test_hadamard_preserves_norm() {
        // Hadamard transform preserves L2 norm (orthogonal matrix)
        let mut data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        fast_hadamard_transform(&mut data);
        let norm_after: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!((norm_before - norm_after).abs() < 1e-4,
            "Norm changed: {norm_before} → {norm_after}");
    }

    #[test]
    fn test_hadamard_smears_outliers() {
        // A tensor with one huge outlier should have smaller max after Hadamard
        let mut data = vec![0.0f32; 8];
        data[0] = 100.0; // single outlier

        let max_before = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        fast_hadamard_transform(&mut data);
        let max_after = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        // After Hadamard, the outlier energy is spread across all elements
        // max should decrease by roughly sqrt(n)
        assert!(max_after < max_before,
            "Hadamard should reduce max: {max_before} → {max_after}");
    }

    #[test]
    fn test_fp4_e2m1_representable_values() {
        // Verify E2M1 quantization snaps to representable values
        let scale = 1.0;
        let cases = [
            (0.0, 0.0), (0.3, 0.5), (0.7, 0.5), (1.1, 1.0),
            (1.6, 1.5), (2.2, 2.0), (3.2, 3.0), (4.5, 4.0), (5.5, 6.0),
        ];
        for (input, expected) in cases {
            let result = quantize_fp4_e2m1(input, scale);
            assert_eq!(result, expected, "FP4 E2M1: {input} should map to {expected}, got {result}");
        }
    }

    #[test]
    fn test_fp4_e2m1_clamping() {
        let scale = 1.0;
        // Values beyond 6.0 should clamp
        let result = quantize_fp4_e2m1(100.0, scale);
        assert_eq!(result, 6.0);
        let result_neg = quantize_fp4_e2m1(-100.0, scale);
        assert_eq!(result_neg, -6.0);
    }

    #[test]
    fn test_nvfp4_quantize_without_hadamard() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = quantize_nvfp4(&data, 4, false);

        assert_eq!(result.len, 4);
        assert!(!result.hadamard_applied);
        assert_eq!(result.block_scales.len(), 1); // 4/4 = 1 block
    }

    #[test]
    fn test_nvfp4_quantize_with_hadamard() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = quantize_nvfp4(&data, 4, true);

        assert!(result.hadamard_applied);

        // Dequantize should approximately recover original
        let recovered = dequantize_nvfp4(&result);
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            let err = (orig - rec).abs();
            assert!(err < 2.0, "NVFP4 roundtrip error too large: {orig} → {rec} (err={err})");
        }
    }

    // ── Host-staging guard tests (device/view/dtype hazards) ────────────

    fn make_2d(rows: i64, cols: i64, data: &[f32]) -> i64 {
        assert_eq!(data.len(), (rows * cols) as usize);
        let data_ptr =
            crate::memory::checked_alloc(data.len() * std::mem::size_of::<f32>()) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            unsafe { *data_ptr.add(i) = v };
        }
        let shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
        unsafe {
            *shape = rows;
            *shape.add(1) = cols;
        }
        let strides = crate::tensor::NslTensor::compute_strides(shape, 2);
        let t = Box::new(crate::tensor::NslTensor::new(
            data_ptr as *mut std::ffi::c_void,
            shape,
            strides,
            2,
            rows * cols,
            0,
            1,
            1,
            0,
        ));
        Box::into_raw(t) as i64
    }

    fn read_f32(ptr: i64, len: usize) -> Vec<f32> {
        let t = crate::tensor::NslTensor::from_ptr(ptr);
        assert_eq!(t.dtype, 1);
        (0..len).map(|i| unsafe { *t.data_f32().add(i) }).collect()
    }

    /// `nsl_fp8_cast` on a zero-copy transposed view must quantize the
    /// elements in LOGICAL order. Before the staging guard it read the
    /// shared storage flat and stamped fresh row-major strides onto the
    /// view's shape — reinterpreting the transpose as a reshape.
    #[test]
    fn fp8_cast_materializes_transposed_views_in_logical_order() {
        // Integers ≤ 6 are exactly representable in E4M3 at scale 1.0, so
        // quantization is the identity and any mismatch is pure ordering.
        let t = make_2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let view = crate::tensor::nsl_tensor_transpose(t, 0, 1); // [3,2] view
        let out = nsl_fp8_cast(view, FP8_FORMAT_E4M3, 1.0);
        let got = read_f32(out, 6);
        // Logical [3,2] row-major: column-major walk of the [2,3] input.
        let expect = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(
            got, expect,
            "cast of a transposed view returned storage order (reshape), not the transpose"
        );
        crate::tensor::nsl_tensor_free(out);
        crate::tensor::nsl_tensor_free(view);
        crate::tensor::nsl_tensor_free(t);
    }

    /// The rank-1 blind spot (review finding): `is_contiguous()` returns
    /// true for EVERY rank-<=1 tensor regardless of strides, so the guard's
    /// first draft passed a 1-D stride-0 expand view straight to the flat
    /// read — 1 valid element and N-1 heap-OOB reads. The guard now checks
    /// the rank-1 stride explicitly.
    #[test]
    fn fp8_cast_materializes_a_rank1_broadcast_view() {
        // A [4] stride-0 view over a 1-element buffer — what
        // `nsl_tensor_expand` legally produces.
        let base = make_2d(1, 1, &[2.0]);
        let base_t = crate::tensor::NslTensor::from_ptr(base);
        let shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = 4 };
        let strides = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = 0 };
        let view = Box::new(crate::tensor::NslTensor::new(
            base_t.data,
            shape,
            strides,
            1,
            4,
            0,
            1,
            0, // borrowed storage — the base owns the data
            0,
        ));
        let view_ptr = Box::into_raw(view) as i64;

        let out = nsl_fp8_cast(view_ptr, FP8_FORMAT_E4M3, 1.0);
        assert_eq!(
            read_f32(out, 4),
            [2.0, 2.0, 2.0, 2.0],
            "a rank-1 stride-0 view must broadcast, not flat-read past its buffer"
        );

        crate::tensor::nsl_tensor_free(out);
        crate::tensor::nsl_tensor_free(view_ptr);
        crate::tensor::nsl_tensor_free(base);
    }

    /// Same hazard for the per-block quantizer, where flat order also
    /// changes the BLOCK PARTITION (and therefore the per-block scales),
    /// not just element placement.
    #[test]
    fn mxfp8_quantize_materializes_transposed_views_in_logical_order() {
        let t = make_2d(2, 2, &[1000.0, 0.3, 0.4, 2000.0]);
        let view = crate::tensor::nsl_tensor_transpose(t, 0, 1);
        let out = nsl_mxfp8_quantize(view, 2, FP8_FORMAT_E4M3);
        let got = read_f32(out, 4);
        // Reference: the pure function on the logically-transposed data.
        let expect = quantize_mxfp8(&[1000.0, 0.4, 0.3, 2000.0], 2, FP8_FORMAT_E4M3).data;
        assert_eq!(
            got, expect,
            "block quantizer partitioned the view's storage order, not its logical order"
        );
        crate::tensor::nsl_tensor_free(out);
        crate::tensor::nsl_tensor_free(view);
        crate::tensor::nsl_tensor_free(t);
    }

    /// End-to-end tape gate: the Fp8MatMul backward transposes `saved_a` /
    /// `saved_b` as zero-copy views before handing them to
    /// `fp8_matmul_e5m2_backward`. With the stride-blind cast this computed
    /// gradients from MISPLACED operands (grad_A = G @ reshape(B) instead
    /// of G @ B^T) — wrong values on the CPU tape path, not reduced
    /// precision. Rectangular shapes (M=2, K=3, N=4) so a reshape cannot
    /// masquerade as the transpose.
    #[test]
    fn tape_fp8_backward_gradients_use_the_logical_transpose() {
        // All values are multiples of 0.5 with small magnitude, so E5M2
        // quantization at the default scale 1.0 is exact for A and B; the
        // ones-seed gradient survives its amax-based scale to ~1e-7.
        let a = make_2d(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = make_2d(
            3,
            4,
            &[1.0, -2.0, 0.5, 3.0, 2.0, 1.0, -1.0, 0.5, 0.5, 4.0, 2.0, -3.0],
        );

        let params = crate::list::nsl_list_new();
        crate::list::nsl_list_push(params, a);
        crate::list::nsl_list_push(params, b);
        crate::autodiff::nsl_tape_start(params);

        let out = nsl_fp8_matmul_training(a, b, 0);
        let grads = crate::autodiff::nsl_tape_backward(out, params);

        // Seed G = ones[2,4]:
        //   grad_A[i][k] = sum_n B[k][n]  (row sums of B)
        //   grad_B[k][n] = sum_m A[m][k]  (column sums of A)
        let grad_a = read_f32(crate::list::nsl_list_get(grads, 0), 6);
        let grad_b = read_f32(crate::list::nsl_list_get(grads, 1), 12);
        let expect_a = [2.5, 2.5, 3.5, 2.5, 2.5, 3.5];
        let expect_b = [5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0, 9.0];
        for (i, (&g, &e)) in grad_a.iter().zip(expect_a.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "grad_A[{i}]={g} expected {e} — fp8 backward used storage-order operands"
            );
        }
        for (i, (&g, &e)) in grad_b.iter().zip(expect_b.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "grad_B[{i}]={g} expected {e} — fp8 backward used storage-order operands"
            );
        }

        // nsl_list_free frees only the list storage, not the elements.
        crate::tensor::nsl_tensor_free(crate::list::nsl_list_get(grads, 0));
        crate::tensor::nsl_tensor_free(crate::list::nsl_list_get(grads, 1));
        crate::list::nsl_list_free(grads);
        crate::autodiff::nsl_tape_stop();
        crate::list::nsl_list_free(params);
        crate::tensor::nsl_tensor_free(out);
        crate::tensor::nsl_tensor_free(a);
        crate::tensor::nsl_tensor_free(b);
    }

    #[test]
    fn test_nvfp4_hadamard_smears_outlier_max() {
        // Hadamard reduces the max element, which improves block quantization scaling
        let mut data = vec![0.0f32; 8];
        data[0] = 100.0; // single outlier

        let max_before = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let mut transformed = data.clone();
        fast_hadamard_transform(&mut transformed);
        let max_after = transformed.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        // After Hadamard, the single outlier is spread across all 8 elements
        // max should decrease by sqrt(8) ≈ 2.83
        assert!(max_after < max_before * 0.5,
            "Hadamard should reduce max from {max_before} to < {}, got {max_after}",
            max_before * 0.5);
    }
}
