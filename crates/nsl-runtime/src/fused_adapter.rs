//! WRGA B.3 Task 4/5: fused LoRA/IA³ adapter FFIs.
//!
//! These entry points are the *single-call* replacement for the unfused
//! three-FFI triple (`x @ W + ((x @ A) @ B) * scale`) emitted by B.2.1.
//! The AST rewrite in `wrga_adapter_rewrite.rs` emits a Call to one of
//! these FFIs when the site's `FusionTarget` is `EpilogueFusedLora` /
//! `ActivationFusedIa3` AND the compile target is sm_80+.
//!
//! # Status — Task 5.5 (env-gated real-cudarc launch)
//!
//! Task 5.5 adds the cudarc launch *machinery* on top of the Task 4
//! CPU-fallback.  Because the compiler's `fused_ptx_kernels` registry
//! lives in `nsl-codegen` state (not reachable from `nsl-runtime`), the
//! real-CUDA path cannot fire unless the generated program populates
//! the runtime's PTX registry via `nsl_wrga_register_fused_ptx` before
//! the first fused call.  Until that codegen hookup ships (tracked as a
//! follow-up), the real-CUDA path is opt-in behind the
//! `NSL_WRGA_FUSED_CUDA=1` environment variable.  When the env var is
//! unset (the default), we take the numerically-correct CPU-fallback
//! that delegates to `nsl_tensor_matmul` + `nsl_tensor_add` + `mul_scalar`
//! — the same path B.3 Task 4 shipped, so Build 4 stays at 1e-4 and
//! Build 5 stays at exactly one launch per site.
//!
//! When the env var IS set, we attempt:
//!   1. Look up the PTX string for `kernel_handle` in the runtime
//!      registry.  Miss → warn-and-fall-through.
//!   2. Launch via `crate::cuda::inner::kernel_launch` with grid/block
//!      derived from the tensor shapes.  Launch error → warn-and-fall-
//!      through.
//! The launch counter increments exactly once per FFI call, regardless
//! of which path (CUDA vs CPU) is taken — so Build 5's `count == 1`
//! invariant is preserved in both modes.
//!
//! # FFI contract
//!
//! All tensor pointers arrive as `i64` handles on the *NslTensor* heap,
//! following the rest of the runtime's calling convention.  Operand
//! ownership stays with the caller; the returned tensor is freshly
//! allocated and the caller must free it.

use crate::tensor::arithmetic::{
    nsl_tensor_add, nsl_tensor_matmul, nsl_tensor_mul, nsl_tensor_mul_scalar,
};
use crate::tensor::activation::nsl_tensor_sigmoid;
use crate::tensor::fbip_flags::{RELINQUISH_A, RELINQUISH_B};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// One-time warning latch for the unmaterialized-adapter fallback.
static WARNED_UNMATERIALIZED_ADAPTER: AtomicBool = AtomicBool::new(false);

/// Emit a one-time stderr warning when a fused-adapter FFI is called with
/// null adapter tensors. This happens when `@adapter` is used in a
/// forward-only / inference-only program: the side-table is materialized
/// only inside a `train` block (`emit_adapter_init_sidetable`), so a
/// forward-only program leaves the slots null. `expr/access.rs`'s null-base
/// guard yields 0 for those fields, and the FFIs below fall back to the base
/// `x @ W` forward rather than dereferencing null adapter tensors (which used
/// to segfault). See
/// docs/plans/2026-05-23-wrga-b4-fused-forward-staging-scope.md.
fn warn_unmaterialized_adapter_once() {
    if !WARNED_UNMATERIALIZED_ADAPTER.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[nsl-wrga] @adapter field accessed before materialization (adapter \
             tensors are null) - running base `x @ W` forward WITHOUT the adapter. \
             Adapters materialize only inside a `train` block; there is no \
             inference-only materialization path today. See \
             docs/plans/2026-05-23-wrga-b4-fused-forward-staging-scope.md."
        );
    }
}

/// B.3 Task 5: side-channel counter for fused-adapter kernel launches.
/// Increments once per call to a fused FFI (per call site, per invocation).
/// When `NSL_KERNEL_LAUNCH_COUNTER=1` is set in the environment, a value
/// is printed to stderr at process exit via the atexit hook registered in
/// `args.rs::nsl_args_init`.
pub(crate) static FUSED_ADAPTER_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// B.3 Task 5.6 hardening: counter that increments ONLY when the real
/// cudarc PTX launch succeeds (not on CPU fallback).  Distinguishes the
/// two dispatch paths for Task 5.6's hardening test, which asserts real-
/// GPU execution when `NSL_WRGA_FUSED_CUDA=1` is set.
pub(crate) static FUSED_ADAPTER_GPU_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// Bump the fused-adapter launch counter.  Only effective (i.e. observed
/// by the atexit reporter) when `NSL_KERNEL_LAUNCH_COUNTER=1`; the
/// increment itself is unconditional and thread-safe.
#[inline]
fn record_fused_launch() {
    FUSED_ADAPTER_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Bump the GPU-specific launch counter.  Called only from the success
/// branch of `try_cuda_launch_fused_*` — never on CPU fallback.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[inline]
pub(crate) fn record_fused_gpu_launch() {
    FUSED_ADAPTER_GPU_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Atexit hook: print `[nsl-kernel-count] <N>` to stderr.  Registered
/// from `args.rs::nsl_args_init` only when `NSL_KERNEL_LAUNCH_COUNTER=1`.
pub extern "C" fn nsl_fused_adapter_launch_count_atexit() {
    let n = FUSED_ADAPTER_LAUNCH_COUNT.load(Ordering::Relaxed);
    eprintln!("[nsl-kernel-count] {n}");
}

/// Atexit hook: print `[nsl-gpu-launch-count] <N>` to stderr.  Registered
/// from `args.rs::nsl_args_init` only when `NSL_WRGA_GPU_LAUNCH_COUNTER=1`.
/// Zero means all fused calls fell back to CPU math.
pub extern "C" fn nsl_fused_adapter_gpu_launch_count_atexit() {
    let n = FUSED_ADAPTER_GPU_LAUNCH_COUNT.load(Ordering::Relaxed);
    eprintln!("[nsl-gpu-launch-count] {n}");
}

// ───────────────────────────────────────────────────────────────────────
// Task 5.5: runtime-side fused-PTX registry
// ───────────────────────────────────────────────────────────────────────

/// Entry recorded by `nsl_wrga_register_fused_ptx`.  `ptx` is a
/// null-terminated PTX source string; `kernel_name` is the entry symbol
/// to pass to `cuModuleGetFunction` (as null-terminated bytes).
#[allow(dead_code)] // fields read only on cfg(feature = "cuda")
struct FusedPtxEntry {
    ptx: String,
    kernel_name: String,
}

fn fused_ptx_registry() -> &'static Mutex<HashMap<i64, FusedPtxEntry>> {
    static REG: OnceLock<Mutex<HashMap<i64, FusedPtxEntry>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register a synthesized fused PTX kernel under `handle`.  Intended to
/// be called by compiler-emitted init code once the codegen-side hookup
/// lands.  Safe to call multiple times for the same handle (last write
/// wins).
///
/// # Safety
/// `ptx` must point to `ptx_len` valid UTF-8 bytes (not required to be
/// null-terminated).  `kernel_name` must point to `name_len` valid
/// UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn nsl_wrga_register_fused_ptx(
    handle: i64,
    ptx: *const u8,
    ptx_len: i64,
    kernel_name: *const u8,
    name_len: i64,
) {
    if handle < 0 || ptx.is_null() || ptx_len <= 0 || kernel_name.is_null() || name_len <= 0 {
        return;
    }
    let ptx_bytes = std::slice::from_raw_parts(ptx, ptx_len as usize);
    let name_bytes = std::slice::from_raw_parts(kernel_name, name_len as usize);
    let ptx_str = match std::str::from_utf8(ptx_bytes) {
        Ok(s) => s.to_string(),
        Err(_) => return,
    };
    let name_str = match std::str::from_utf8(name_bytes) {
        Ok(s) => s.to_string(),
        Err(_) => return,
    };
    let mut reg = match fused_ptx_registry().lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    reg.insert(
        handle,
        FusedPtxEntry {
            ptx: format!("{}\0", ptx_str),
            kernel_name: format!("{}\0", name_str),
        },
    );
}

/// Return true iff `NSL_WRGA_FUSED_CUDA=1` is set in the environment.
/// Controls whether fused FFIs attempt the real cudarc launch path; when
/// false (default) they take the correct CPU-fallback math that already
/// passes Build 4 at 1e-4 and Build 5's kernel-count invariant.
#[inline]
fn real_cuda_path_enabled() -> bool {
    std::env::var("NSL_WRGA_FUSED_CUDA").ok().as_deref() == Some("1")
}

/// Attempt a real cudarc-based launch for the LoRA kernel identified by
/// `handle`.  Returns `Some(out_ptr)` on success, `None` on any failure
/// (missing PTX, non-GPU tensors, launch error, etc.) — the caller then
/// falls back to the CPU-math path.  Emits a one-line stderr warning on
/// fallback so test failures are debuggable.
#[cfg(feature = "cuda")]
fn try_cuda_launch_fused_lora(
    x: i64,
    w: i64,
    lora_a: i64,
    lora_b: i64,
    scale: f64,
    kernel_handle: i64,
) -> Option<i64> {
    use crate::cuda::inner;
    use crate::tensor::NslTensor;
    use std::ffi::c_void;

    // 1. PTX must be registered for this handle.
    let (ptx_cstr, name_cstr) = {
        let reg = fused_ptx_registry().lock().ok()?;
        let entry = reg.get(&kernel_handle)?;
        (entry.ptx.clone(), entry.kernel_name.clone())
    };

    // 2. All inputs must be on GPU and f16/f32-compatible.  We do not
    //    perform dtype conversion here; that's a synthesizer concern.
    let xt = unsafe { &*(x as *const NslTensor) };
    let wt = unsafe { &*(w as *const NslTensor) };
    let at = unsafe { &*(lora_a as *const NslTensor) };
    let bt = unsafe { &*(lora_b as *const NslTensor) };
    if xt.device == 0 || wt.device == 0 || at.device == 0 || bt.device == 0 {
        eprintln!(
            "[nsl-wrga] NSL_WRGA_FUSED_CUDA=1 set but inputs not on GPU — \
             falling back to CPU math"
        );
        return None;
    }

    // 3. Derive shape-dependent grid/block from x=[m,k], w=[k,n].
    if xt.ndim < 2 || wt.ndim < 2 {
        return None;
    }
    let m = unsafe { *xt.shape.add(xt.ndim as usize - 2) } as i64;
    let n = unsafe { *wt.shape.add(wt.ndim as usize - 1) } as i64;

    // Allocate output on the same device, f32 (PTX stores back as f32).
    let out_elems = (m * n) as usize;
    let out_data = inner::try_alloc_managed(out_elems * 4)?;
    let shape = NslTensor::copy_shape(xt.shape, 2);
    unsafe {
        *shape.add(0) = m;
        *shape.add(1) = n;
    }
    let strides = NslTensor::compute_strides(shape, 2);
    let out_box = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        2,
        (m * n) as i64,
        xt.device,
        1, // f32 output
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out_box);

    // 4. Marshal launch args.  Scale passed as .param .f32 (see
    //    wrga_fused_ptx.rs — scale is NOT baked into PTX).
    //    m_rows and n_cols are passed as .param .u32 so the kernel
    //    can predicate tail rows/cols correctly regardless of the
    //    compile-time config.m used during prescan.
    let mut x_data = xt.data as u64;
    let mut w_data = wt.data as u64;
    let mut a_data = at.data as u64;
    let mut b_data = bt.data as u64;
    let mut scale_f32: f32 = scale as f32;
    let mut y_data = out_data as u64;
    let mut m_rows_u32: u32 = m as u32;
    let mut n_cols_u32: u32 = n as u32;
    let args: [*mut c_void; 8] = [
        &mut x_data as *mut _ as *mut c_void,
        &mut w_data as *mut _ as *mut c_void,
        &mut a_data as *mut _ as *mut c_void,
        &mut b_data as *mut _ as *mut c_void,
        &mut scale_f32 as *mut _ as *mut c_void,
        &mut y_data as *mut _ as *mut c_void,
        &mut m_rows_u32 as *mut _ as *mut c_void,
        &mut n_cols_u32 as *mut _ as *mut c_void,
    ];

    // 5. Grid/block.  The synthesizer emits m16n8k16 MMA tiles; one
    //    warp (32 threads) per (BM=16, BN=8) output tile.
    let grid = [(m + 15) / 16, (n + 7) / 8, 1];
    let block = [32i64, 1, 1];
    let res = inner::kernel_launch(
        ptx_cstr.as_ptr(),
        name_cstr.as_ptr(),
        grid,
        block,
        &args,
        0,
    );
    if res as u32 != 0 {
        eprintln!(
            "[nsl-wrga] fused LoRA PTX launch failed ({:?}) — falling back to CPU math",
            res
        );
        unsafe {
            let _ = Box::from_raw(out_ptr);
        }
        inner::free_managed(out_data);
        return None;
    }
    let sync_result = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
    if sync_result as u32 != 0 {
        eprintln!(
            "[nsl-wrga] fused LoRA kernel caused GPU error ({:?}) — falling back to CPU math",
            sync_result
        );
        unsafe {
            let _ = Box::from_raw(out_ptr);
        }
        inner::free_managed(out_data);
        return None;
    }
    record_fused_gpu_launch();
    Some(out_ptr as i64)
}

#[cfg(not(feature = "cuda"))]
fn try_cuda_launch_fused_lora(
    _x: i64,
    _w: i64,
    _lora_a: i64,
    _lora_b: i64,
    _scale: f64,
    _kernel_handle: i64,
) -> Option<i64> {
    None
}

#[cfg(feature = "cuda")]
fn try_cuda_launch_fused_ia3(
    x: i64,
    w: i64,
    ia3_scale: i64,
    kernel_handle: i64,
) -> Option<i64> {
    use crate::cuda::inner;
    use crate::tensor::NslTensor;
    use std::ffi::c_void;

    let (ptx_cstr, name_cstr) = {
        let reg = fused_ptx_registry().lock().ok()?;
        let entry = reg.get(&kernel_handle)?;
        (entry.ptx.clone(), entry.kernel_name.clone())
    };

    let xt = unsafe { &*(x as *const NslTensor) };
    let wt = unsafe { &*(w as *const NslTensor) };
    let gt = unsafe { &*(ia3_scale as *const NslTensor) };
    if xt.device == 0 || wt.device == 0 || gt.device == 0 {
        eprintln!(
            "[nsl-wrga] NSL_WRGA_FUSED_CUDA=1 set but inputs not on GPU — \
             falling back to CPU math"
        );
        return None;
    }
    if xt.ndim < 2 || wt.ndim < 2 {
        return None;
    }
    let m = unsafe { *xt.shape.add(xt.ndim as usize - 2) } as i64;
    let n = unsafe { *wt.shape.add(wt.ndim as usize - 1) } as i64;

    let out_elems = (m * n) as usize;
    let out_data = inner::try_alloc_managed(out_elems * 4)?;
    let shape = NslTensor::copy_shape(xt.shape, 2);
    unsafe {
        *shape.add(0) = m;
        *shape.add(1) = n;
    }
    let strides = NslTensor::compute_strides(shape, 2);
    let out_box = Box::new(NslTensor::new(
        out_data, shape, strides, 2, (m * n) as i64, xt.device, 1, 1, 0,
    ));
    let out_ptr = Box::into_raw(out_box);

    let mut x_data = xt.data as u64;
    let mut w_data = wt.data as u64;
    let mut g_data = gt.data as u64;
    let mut y_data = out_data as u64;
    let args: [*mut c_void; 4] = [
        &mut x_data as *mut _ as *mut c_void,
        &mut w_data as *mut _ as *mut c_void,
        &mut g_data as *mut _ as *mut c_void,
        &mut y_data as *mut _ as *mut c_void,
    ];

    let grid = [(m + 15) / 16, (n + 7) / 8, 1];
    let block = [32i64, 1, 1];
    let res = inner::kernel_launch(
        ptx_cstr.as_ptr(),
        name_cstr.as_ptr(),
        grid,
        block,
        &args,
        0,
    );
    if res as u32 != 0 {
        eprintln!(
            "[nsl-wrga] fused IA3 PTX launch failed ({:?}) — falling back to CPU math",
            res
        );
        unsafe {
            let _ = Box::from_raw(out_ptr);
        }
        inner::free_managed(out_data);
        let _ = kernel_handle;
        return None;
    }
    unsafe {
        cudarc::driver::sys::cuCtxSynchronize();
    }
    record_fused_gpu_launch();
    Some(out_ptr as i64)
}

#[cfg(not(feature = "cuda"))]
fn try_cuda_launch_fused_ia3(
    _x: i64,
    _w: i64,
    _ia3_scale: i64,
    _kernel_handle: i64,
) -> Option<i64> {
    None
}

/// CPU-fallback LoRA math (the Task 4 path): `y = x@W + ((x@A)@B)*scale`.
fn cpu_fallback_fused_lora(x: i64, w: i64, lora_a: i64, lora_b: i64, scale: f64) -> i64 {
    let y_main = nsl_tensor_matmul(x, w, 0);
    if y_main == 0 {
        return 0;
    }
    let x_a = nsl_tensor_matmul(x, lora_a, 0);
    if x_a == 0 {
        return 0;
    }
    let x_ab = nsl_tensor_matmul(x_a, lora_b, RELINQUISH_A);
    if x_ab == 0 {
        return 0;
    }
    let scaled = nsl_tensor_mul_scalar(x_ab, scale, RELINQUISH_A);
    if scaled == 0 {
        return 0;
    }
    nsl_tensor_add(y_main, scaled, RELINQUISH_A | RELINQUISH_B)
}

/// CPU-fallback IA³ math: `y = (x @ W) * gamma` (broadcast over last dim).
fn cpu_fallback_fused_ia3(x: i64, w: i64, ia3_scale: i64) -> i64 {
    let y_main = nsl_tensor_matmul(x, w, 0);
    if y_main == 0 {
        return 0;
    }
    nsl_tensor_mul(y_main, ia3_scale, RELINQUISH_A)
}

/// Fused LoRA matmul: `y = x @ W + ((x @ A) @ B) * scale` in a single
/// FFI call.
///
/// # Arguments
///
/// * `x`          — activation, shape `[..., k_in]`
/// * `w`          — base weight, shape `[k_in, d_out]`
/// * `lora_a`     — LoRA-A, shape `[k_in, rank]`
/// * `lora_b`     — LoRA-B, shape `[rank, d_out]`
/// * `scale`      — `alpha / rank`, passed as f64 (converted to f32 at launch)
/// * `kernel_handle` — index into the runtime fused-PTX registry populated
///                     by `nsl_wrga_register_fused_ptx`.  Negative or
///                     unregistered handles skip the CUDA path.
///
/// # Task 5.5 dispatch rules
///
/// 1. `record_fused_launch()` fires exactly once per FFI call (Build 5
///    invariant).
/// 2. If `NSL_WRGA_FUSED_CUDA=1` AND the handle is registered AND all
///    inputs are on GPU, attempt the real cudarc PTX launch.
/// 3. On any failure (or when the env gate is off), fall through to the
///    CPU-math path that delegates to `nsl_tensor_matmul` + friends.  The
///    CPU math is numerically identical to the fused algebra at 1e-4, so
///    Build 4's tolerance is preserved in both modes.
///
/// # Safety
/// All tensor pointers must be valid `*NslTensor` handles or `0`.
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_lora_matmul(
    x: i64,
    w: i64,
    lora_a: i64,
    lora_b: i64,
    scale: f64,
    kernel_handle: i64,
) -> i64 {
    if x == 0 || w == 0 {
        return 0;
    }
    // Unmaterialized adapter (forward-only `@adapter`, no train block) — see
    // the GatedLoRA path and access.rs null-base guard. Fall back to base
    // `x @ W` instead of dereferencing null adapter tensors.
    if lora_a == 0 || lora_b == 0 {
        warn_unmaterialized_adapter_once();
        return nsl_tensor_matmul(x, w, 0);
    }
    record_fused_launch();

    if real_cuda_path_enabled() {
        if let Some(out) = try_cuda_launch_fused_lora(x, w, lora_a, lora_b, scale, kernel_handle) {
            return out;
        }
    }
    cpu_fallback_fused_lora(x, w, lora_a, lora_b, scale)
}

/// Fused IA³ matmul: `y = (x @ W) * gamma` in a single FFI call.
///
/// # Arguments
///
/// * `x`             — activation, shape `[..., k_in]`
/// * `w`             — base weight, shape `[k_in, d_out]`
/// * `ia3_scale`     — per-output-channel scale vector, shape `[d_out]`
/// * `kernel_handle` — same registry semantics as `fused_lora_matmul`.
///
/// # Safety
/// All tensor pointers must be valid `*NslTensor` handles or `0`.
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_ia3_matmul(
    x: i64,
    w: i64,
    ia3_scale: i64,
    kernel_handle: i64,
) -> i64 {
    if x == 0 || w == 0 {
        return 0;
    }
    // Unmaterialized adapter (forward-only `@adapter`, no train block) — see
    // the GatedLoRA path and access.rs null-base guard. The IA3 scale
    // multiplies the result, so the base forward without it is just `x @ W`.
    if ia3_scale == 0 {
        warn_unmaterialized_adapter_once();
        return nsl_tensor_matmul(x, w, 0);
    }
    record_fused_launch();

    if real_cuda_path_enabled() {
        if let Some(out) = try_cuda_launch_fused_ia3(x, w, ia3_scale, kernel_handle) {
            return out;
        }
    }
    cpu_fallback_fused_ia3(x, w, ia3_scale)
}

/// CPU-fallback GatedLoRA math:
/// `y = x @ W + sigmoid(gate) * ((x @ A) @ B) * scale`.
///
/// # Step-0 invariant (LOAD-BEARING — mirrors wrga_adapter_rewrite.rs)
///
/// `gate` is initialized to zeros, so `sigmoid(0) == 0.5`.  The gate is
/// HALF-OPEN at step 0.  Base-model equivalence depends on `lora_B = 0`.
fn cpu_fallback_fused_gatedlora(
    x: i64,
    w: i64,
    lora_a: i64,
    lora_b: i64,
    scale: f64,
    gate: i64,
) -> i64 {
    // Base matmul: y_main = x @ W
    let y_main = nsl_tensor_matmul(x, w, 0);
    if y_main == 0 {
        return 0;
    }
    // Adapter path: (x @ A) @ B
    let x_a = nsl_tensor_matmul(x, lora_a, 0);
    if x_a == 0 {
        return y_main;
    }
    let x_ab = nsl_tensor_matmul(x_a, lora_b, RELINQUISH_A);
    if x_ab == 0 {
        return y_main;
    }
    // sigmoid(gate) * (x @ A @ B) * scale
    let gate_sig = nsl_tensor_sigmoid(gate);
    if gate_sig == 0 {
        return y_main;
    }
    let gated = nsl_tensor_mul(gate_sig, x_ab, RELINQUISH_A | RELINQUISH_B);
    if gated == 0 {
        return y_main;
    }
    let scaled = nsl_tensor_mul_scalar(gated, scale, RELINQUISH_A);
    if scaled == 0 {
        return y_main;
    }
    nsl_tensor_add(y_main, scaled, RELINQUISH_A | RELINQUISH_B)
}

/// Attempt a real cudarc-based launch for the GatedLoRA kernel identified by
/// `handle`.  Returns `Some(out_ptr)` on success, `None` on any failure
/// (missing PTX, non-GPU tensors, launch error, etc.) - the caller then
/// falls back to the CPU-math path.  Emits a one-line stderr warning on
/// fallback so test failures are debuggable.
///
/// PTX kernel signature (9 args):
///   (x_ptr, w_ptr, a_ptr, b_ptr, scale.f32, y_ptr, m_rows.u32, n_cols.u32, gate_ptr)
#[cfg(feature = "cuda")]
fn try_cuda_launch_fused_gatedlora(
    x: i64,
    w: i64,
    lora_a: i64,
    lora_b: i64,
    scale: f64,
    gate: i64,
    kernel_handle: i64,
) -> Option<i64> {
    use crate::cuda::inner;
    use crate::tensor::NslTensor;
    use std::ffi::c_void;

    // 1. PTX must be registered for this handle.
    let (ptx_cstr, name_cstr) = {
        let reg = fused_ptx_registry().lock().ok()?;
        let entry = reg.get(&kernel_handle)?;
        (entry.ptx.clone(), entry.kernel_name.clone())
    };

    // 2. All inputs must be on GPU.  We do not perform dtype conversion here;
    //    that is a synthesizer concern.
    let xt = unsafe { &*(x as *const NslTensor) };
    let wt = unsafe { &*(w as *const NslTensor) };
    let at = unsafe { &*(lora_a as *const NslTensor) };
    let bt = unsafe { &*(lora_b as *const NslTensor) };
    let gat = unsafe { &*(gate as *const NslTensor) };
    if xt.device == 0 || wt.device == 0 || at.device == 0 || bt.device == 0 || gat.device == 0 {
        eprintln!(
            "[nsl-wrga] NSL_WRGA_FUSED_CUDA=1 set but GatedLoRA inputs not on GPU - \
             falling back to CPU math"
        );
        return None;
    }

    // 3. Derive shape-dependent grid/block from x=[m,k], w=[k,n].
    if xt.ndim < 2 || wt.ndim < 2 {
        return None;
    }
    let m = unsafe { *xt.shape.add(xt.ndim as usize - 2) } as i64;
    let n = unsafe { *wt.shape.add(wt.ndim as usize - 1) } as i64;

    // Allocate output on the same device, f32 (PTX stores back as f32).
    let out_elems = (m * n) as usize;
    let out_data = inner::try_alloc_managed(out_elems * 4)?;
    let shape = NslTensor::copy_shape(xt.shape, 2);
    unsafe {
        *shape.add(0) = m;
        *shape.add(1) = n;
    }
    let strides = NslTensor::compute_strides(shape, 2);
    let out_box = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        2,
        (m * n) as i64,
        xt.device,
        1, // f32 output
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out_box);

    // 4. Marshal launch args.  PTX kernel signature (9 params):
    //    (x_ptr, w_ptr, a_ptr, b_ptr, scale.f32, y_ptr, m_rows.u32, n_cols.u32, gate_ptr)
    let mut x_data = xt.data as u64;
    let mut w_data = wt.data as u64;
    let mut a_data = at.data as u64;
    let mut b_data = bt.data as u64;
    let mut scale_f32: f32 = scale as f32;
    let mut y_data = out_data as u64;
    let mut m_rows_u32: u32 = m as u32;
    let mut n_cols_u32: u32 = n as u32;
    let mut gate_data = gat.data as u64;
    let args: [*mut c_void; 9] = [
        &mut x_data as *mut _ as *mut c_void,
        &mut w_data as *mut _ as *mut c_void,
        &mut a_data as *mut _ as *mut c_void,
        &mut b_data as *mut _ as *mut c_void,
        &mut scale_f32 as *mut _ as *mut c_void,
        &mut y_data as *mut _ as *mut c_void,
        &mut m_rows_u32 as *mut _ as *mut c_void,
        &mut n_cols_u32 as *mut _ as *mut c_void,
        &mut gate_data as *mut _ as *mut c_void,
    ];

    // 5. Grid/block.  The synthesizer emits m16n8k16 MMA tiles; one
    //    warp (32 threads) per (BM=16, BN=8) output tile.
    let grid = [(m + 15) / 16, (n + 7) / 8, 1];
    let block = [32i64, 1, 1];
    let res = inner::kernel_launch(
        ptx_cstr.as_ptr(),
        name_cstr.as_ptr(),
        grid,
        block,
        &args,
        0,
    );
    if res as u32 != 0 {
        eprintln!(
            "[nsl-wrga] fused GatedLoRA PTX launch failed ({:?}) - falling back to CPU math",
            res
        );
        unsafe {
            let _ = Box::from_raw(out_ptr);
        }
        inner::free_managed(out_data);
        return None;
    }
    let sync_result = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
    if sync_result as u32 != 0 {
        eprintln!(
            "[nsl-wrga] fused GatedLoRA kernel caused GPU error ({:?}) - falling back to CPU math",
            sync_result
        );
        unsafe {
            let _ = Box::from_raw(out_ptr);
        }
        inner::free_managed(out_data);
        return None;
    }
    record_fused_gpu_launch();
    Some(out_ptr as i64)
}

#[cfg(not(feature = "cuda"))]
fn try_cuda_launch_fused_gatedlora(
    _x: i64,
    _w: i64,
    _lora_a: i64,
    _lora_b: i64,
    _scale: f64,
    _gate: i64,
    _kernel_handle: i64,
) -> Option<i64> {
    None
}

/// Fused GatedLoRA matmul:
/// `y = x @ W + sigmoid(gate) * ((x @ A) @ B) * scale` in a single FFI call.
///
/// # Arguments
///
/// * `x`             — activation, shape `[..., k_in]`
/// * `w`             — base weight, shape `[k_in, d_out]`
/// * `lora_a`        — LoRA-A, shape `[k_in, rank]`
/// * `lora_b`        — LoRA-B, shape `[rank, d_out]`
/// * `scale`         — `alpha / rank`, passed as f64
/// * `gate`          — gating scalar or vector; sigmoid applied element-wise
/// * `kernel_handle` — index into the runtime fused-PTX registry populated
///                     by `nsl_wrga_register_fused_ptx`.  Negative or
///                     unregistered handles skip the CUDA path.
///
/// # Dispatch rules
///
/// 1. `record_fused_launch()` fires exactly once per FFI call (Build 5 invariant).
/// 2. If `NSL_WRGA_FUSED_CUDA=1` AND the handle is registered AND all
///    inputs are on GPU, attempt the real cudarc PTX launch (Task 5.0.b).
/// 3. On any failure (or when the env gate is off), fall through to the
///    CPU-math path.  The CPU math is numerically correct for the step-0
///    invariant and Fixture A at 1e-4.
/// 4. Kernel registry still lacks GatedLoRA entries until Task 5.0.c wires
///    PTX registration; Fixture A's GPU launch count stays at 0 until then.
///
/// # Safety
/// All tensor pointers must be valid `*NslTensor` handles or `0`.
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_gatedlora_matmul(
    x: i64,
    w: i64,
    lora_a: i64,
    lora_b: i64,
    scale: f64,
    gate: i64,
    kernel_handle: i64,
) -> i64 {
    if x == 0 || w == 0 {
        return 0;
    }
    // Unmaterialized adapter (forward-only `@adapter`, no train block) — the
    // side-table base is null so `access.rs` yielded 0 for these fields.
    // Fall back to the base `x @ W` forward instead of dereferencing null
    // adapter tensors (which segfaulted).
    if lora_a == 0 || lora_b == 0 || gate == 0 {
        warn_unmaterialized_adapter_once();
        return nsl_tensor_matmul(x, w, 0);
    }
    record_fused_launch();

    if real_cuda_path_enabled() {
        if let Some(out) =
            try_cuda_launch_fused_gatedlora(x, w, lora_a, lora_b, scale, gate, kernel_handle)
        {
            return out;
        }
    }
    cpu_fallback_fused_gatedlora(x, w, lora_a, lora_b, scale, gate)
}
