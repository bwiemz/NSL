//! G3 — Fused linear-CE FFI surface.
//!
//! Exports two C-ABI functions:
//!
//! * `nsl_fused_linear_ce_forward`  — forward pass
//! * `nsl_fused_linear_ce_backward` — backward pass
//!
//! Both functions accept pre-synthesised PTX (null-terminated) plus device
//! pointers for the tensors.  On CPU (device == 0) they fall back to the
//! existing CPU cross-entropy implementation.
//!
//! The PTX is supplied by the caller (typically a test or compiler-emitted
//! call site); this module does NOT synthesise PTX — that lives in
//! `nsl-codegen::fused_linear_ce`.
//!
//! # CFTP v7 — GPU-side precision_cast kernel lifts the v6 FFI refusal
//!
//! When `dtype_tag != 0` (fp16 or bf16), the caller MUST allocate
//! `x_ptr` / `w_ptr` / `bias_ptr` in HBM with a layout matching the dtype:
//! `[B*S*H]` or `[V*H]` of 16-bit elements (2 bytes per elem) for the
//! 16-bit dtypes.  The PTX kernels do `ld.global.b16` reads on those
//! pointers — supplying a 32-bit-laid-out buffer would read the upper+lower
//! halves of two adjacent f32s as a single bf16/fp16 with NO error and
//! NO diagnostic (silent numerical corruption).
//!
//! As of CFTP v6, the wengert dispatch in
//! `crates/nsl-codegen/src/wengert_lower.rs::lower_fused_linear_ce_forward`
//! (and the symmetric `..._backward_extract`) inserts an explicit
//! `nsl_tensor_to_{bf16,fp16}` cast op into the Cranelift IR before the
//! FFI call whenever `dtype_tag != 0`.  The cast op publishes a fresh
//! 16-bit storage tensor with the dtype-compliant HBM layout above, so the
//! production lowering can no longer reach the kernel with a malformed
//! buffer.
//!
//! ## CFTP v7 — lifting the default-on FFI refusal
//!
//! The v6 default-on FFI refusal (`refuse_non_f32_if_unsafe`) was a safe
//! interim because `cast_and_publish` was CPU-only: it asserted
//! `t.device == 0` and aborted on any GPU tensor, so the structural cast
//! could not actually run on the production GPU path the FFI requires.
//! Sprint v7 closes that gap end-to-end:
//!
//!   1. `cast_and_publish` now accepts GPU tensors and routes them through
//!      `gpu_cast_and_publish` (commit 30f2183c).
//!   2. The GPU cast kernels (`crates/nsl-codegen/src/precision_cast_ptx.rs`,
//!      commit 3cc13824) emit `cvt.rn.{bf16,f16}.f32` — IEEE-754 default
//!      round-to-nearest-even — matching the host `half::*::from_f32`
//!      rounding used by the direct-FFI numerical baselines.
//!   3. The CPU primitives `f32_to_{bf16,f16}_bits` (`tensor/mod.rs`) now
//!      also delegate to `half::*::from_f32` (CFTP v7 follow-on, finding-7
//!      fix), so the CPU and GPU casts are byte-identical: no more silent
//!      one-ULP divergence across device boundaries.
//!   4. The structural cast insertion is verified at the wengert layer by
//!      `crates/nsl-codegen/tests/fused_linear_ce_precision_cast_lowering.rs`
//!      — every `dtype_tag != 0` lowering inserts the
//!      `nsl_tensor_to_{bf16,fp16}` op before the FFI call, so the FFI
//!      contract is enforced at the IR level. (Note: the V=49152 numerical
//!      baselines in `fused_linear_ce_{fp16,bf16}_v49152_numerical.rs`
//!      still stage inputs via `half::*::from_f32` on the host and call
//!      the FFI directly; they validate the FFI's RTE semantics in
//!      isolation. End-to-end coverage via the production wengert GPU
//!      cast lives in the GPU-only dispatch test
//!      `crates/nsl-runtime/tests/precision_cast_gpu_dispatch.rs`.)
//!   5. The forward-to-backward precision-cast cache from commit 475b7b08
//!      remains in effect (Findings 10/14/16): one cast set per dispatch
//!      shared across forward and backward.
//!
//! With those preconditions met, the v6 default-on refusal is REMOVED:
//! `dtype_tag != 0` now reaches the kernel directly, and the
//! `NSL_FUSED_LCE_ALLOW_NON_F32_FFI` opt-in env var becomes a no-op (we
//! leave it unhandled so any straggler CI invocations fall through
//! harmlessly).  The F32 (`dtype_tag == 0`) byte-identity contract is
//! preserved — that path was never gated by the refusal.
//!
//! ## Diagnostic / safety override (`NSL_FUSED_LCE_REFUSE_NON_F32=1`)
//!
//! As an operator escape hatch for diagnosing a suspected dtype-conformance
//! regression in a downstream build (e.g. a future codegen change that
//! drops the wengert `nsl_tensor_to_{bf16,fp16}` insertion), setting
//! `NSL_FUSED_LCE_REFUSE_NON_F32=1` reinstates the v6 behavior: the FFI
//! returns `-1` with a clear diagnostic when `dtype_tag != 0`.  This is
//! OFF by default — production runs never set the var — so the v7
//! activation contract is preserved.

/// CFTP v7 diagnostic / safety override.
///
/// OFF by default: `dtype_tag != 0` proceeds straight to the kernel because
/// the v7 preconditions (GPU cast kernel + RTE + cache + re-measured
/// baselines) are met.  When `NSL_FUSED_LCE_REFUSE_NON_F32=1` is set, the
/// v6 default-on refusal is reinstated for diagnostic purposes — useful for
/// confirming a suspected dtype-conformance regression in a downstream
/// build without rebuilding the runtime.
///
/// Returns `true` if the call should refuse (caller returns `-1`
/// immediately); `false` if the call should proceed normally.
#[inline]
fn refuse_non_f32_if_unsafe(dtype_tag: i64, ffi_name: &str) -> bool {
    if dtype_tag == 0 {
        return false;
    }
    // CFTP v7 follow-on (finding-15): one-shot warning if a CI lane or
    // operator script left over from v6 still sets the legacy ALLOW
    // env var. v6's `NSL_FUSED_LCE_ALLOW_NON_F32_FFI=1` is a no-op
    // under v7 — surface the rename to stderr so stale automation is
    // discovered and updated.
    warn_on_legacy_env_once();
    if std::env::var_os("NSL_FUSED_LCE_REFUSE_NON_F32").is_none() {
        return false;
    }
    eprintln!(
        "{ffi_name}: NSL_FUSED_LCE_REFUSE_NON_F32=1 set; refusing \
         dtype_tag={dtype_tag}.  The v7 production path is now safe by \
         default (GPU PTX cast kernel + RTE rounding + wengert-inserted \
         precision_cast); this env var is an opt-in diagnostic override \
         for confirming a suspected dtype-conformance regression.  Unset \
         it to restore normal v7 activation.",
    );
    true
}

/// CFTP v7 follow-on (finding-15): emit a one-shot stderr warning when
/// the legacy v6 env var `NSL_FUSED_LCE_ALLOW_NON_F32_FFI` is observed.
/// Under v7 the variable is a no-op (we kept it unhandled so straggler
/// CI invocations fall through harmlessly), but operators reading the v6
/// docs may believe it still gates an opt-in safety override. A warning
/// surfaces the rename without spamming the log on every FFI call.
fn warn_on_legacy_env_once() {
    use std::sync::Once;
    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        if std::env::var_os("NSL_FUSED_LCE_ALLOW_NON_F32_FFI").is_some() {
            eprintln!(
                "nsl_fused_linear_ce: legacy env var \
                 NSL_FUSED_LCE_ALLOW_NON_F32_FFI is set but has NO effect \
                 under CFTP v7 — the v6 default-on FFI refusal was lifted \
                 because the wengert-inserted precision_cast op + GPU PTX \
                 cast kernel cover the dtype-conformance contract. If you \
                 set this var to drive a diagnostic, use \
                 NSL_FUSED_LCE_REFUSE_NON_F32=1 instead (opposite \
                 semantics — refuse instead of allow)."
            );
        }
    });
}

/// Forward: `loss[i] = -log_softmax(x[i] @ W^T + bias)[target[i]]`.
///
/// # Arguments
/// * `ptx_ptr`      — `*const u8`: null-terminated forward PTX string.
/// * `kname_ptr`    — `*const u8`: null-terminated kernel name.
/// * `x_ptr`        — device/host f32 tensor `[B, S, H]`.
/// * `w_ptr`        — device/host f32 tensor `[V, H]`.
/// * `bias_ptr`     — device/host f32 tensor `[V]`.
/// * `targets_ptr`  — device/host i64 tensor `[B*S]`.
/// * `loss_out_ptr` — device/host f32 output `[B*S]` (pre-allocated).
/// * `lse_out_ptr`  — device/host f32 output `[B*S]` (pre-allocated, for backward).
/// * `b, s, v, h`   — dims as u32.
/// * `smem_bytes`   — shared-memory bytes per CTA (from `FusedLinearCEConfig::shared_mem_bytes()`).
///
/// Returns 0 on success, negative on error.
///
/// # `dtype_tag` (Sprint v3-2 trailing arg; extended in v4-1)
///
/// `0` = F32 (default — preserves pre-Sprint-v3-2 behaviour byte-identically;
/// all existing callers including the Cranelift IR call site in
/// `wengert_lower.rs` pass this sentinel).
///
/// `1` = F16. Callers that pass `1` MUST allocate `x_ptr`, `w_ptr`, `bias_ptr`
/// as fp16 in HBM. The `loss_out_ptr` and `lse_out_ptr` outputs stay f32
/// regardless of `dtype_tag`.
///
/// `2` = Bf16 (Sprint v4-1). Same HBM-layout contract as F16 (16-bit storage
/// for x/W/bias; f32 outputs). The kernel-side bf16 cvt mnemonics need PTX
/// ISA 7.8+ — codegen bumps to `.version 8.0` on the bf16 path so callers
/// don't need to do anything special at this FFI layer.
///
/// Threading actual fp16/bf16 from user code (so the @fused_lm_ce decorator
/// drives `dtype_tag` end-to-end) is a v4-2 follow-on.
///
/// CSLA fused-CE tape-carry anti-vacuity: successful launch counts per
/// entry point (0 = forward small, 1 = forward large, 2 = backward). A
/// gate comparing csla-vs-baseline loss streams is vacuous if the fused
/// kernel silently never engaged (composite fallback) — these counters
/// prove which path ran and how many times.
static FUSED_LCE_LAUNCH_COUNTS: [std::sync::atomic::AtomicU64; 3] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Test/diagnostic probe: successful fused linear-CE launches for `kind`
/// (0 = forward, 1 = forward_large, 2 = backward). Other kinds return -1.
#[no_mangle]
pub extern "C" fn nsl_fused_lce_launch_count(kind: i64) -> i64 {
    match kind {
        0..=2 => FUSED_LCE_LAUNCH_COUNTS[kind as usize]
            .load(std::sync::atomic::Ordering::Relaxed) as i64,
        _ => -1,
    }
}

/// Targets dtype bridge for the fused linear-CE kernels.
///
/// The kernels read targets with `ld.global.s64` — the direct-FFI test
/// harness convention (hand-allocated i64 buffers). NSL GPU tensors are
/// f32, so passing an NSL label tensor's data pointer straight through
/// OVERREADS the buffer by 2× (latent in pool slack on small shapes —
/// deterministic garbage classes; a hard CUDA_ERROR_ILLEGAL_ADDRESS once
/// the read crosses the allocation, as the v49152 finalize kernel did on
/// the first real `nsl run` engagement). This helper materializes a
/// device i64 copy: DtoH the f32 labels, round to i64 on the host
/// (labels are whole numbers; negatives like ignore_index round
/// exactly), HtoD into a fresh rows×8 buffer. Rows are ≤ a few thousand
/// per step — the roundtrip is microseconds and keeps the validated
/// kernels + their byte-identity snapshots untouched.
///
/// Returns the raw device pointer (NOT an NslTensor — the 8-byte stride
/// would corrupt f32 byte-size accounting); release it with
/// [`nsl_fused_lce_targets_i64_free`] after the kernel FFI returns.
#[no_mangle]
pub extern "C" fn nsl_fused_lce_targets_i64_alloc(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        eprintln!("nsl_fused_lce_targets_i64_alloc: null tensor");
        std::process::abort();
    }
    #[cfg(feature = "cuda")]
    {
        let t = crate::tensor::NslTensor::from_ptr(tensor_ptr);
        let n = t.len as usize;
        // Density guard (review D2c-3): the reads below walk t.data
        // linearly — a broadcast/expand view (stride 0) or a strided
        // slice would be silently misread. Every real label path is a
        // contiguous post-reshape tensor; refuse loudly otherwise.
        {
            let shape = unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize) };
            let strides =
                unsafe { std::slice::from_raw_parts(t.strides, t.ndim as usize) };
            let mut dense_extent: i64 = 1;
            let mut has_zero_stride = false;
            for (d, (&dim, &st)) in shape.iter().zip(strides).enumerate() {
                let _ = d;
                if dim > 1 && st == 0 {
                    has_zero_stride = true;
                }
                dense_extent += st * (dim - 1).max(0);
            }
            if has_zero_stride || dense_extent != t.len {
                eprintln!(
                    "nsl_fused_lce_targets_i64_alloc: non-contiguous label \
                     tensor (dense extent {dense_extent} vs len {}, \
                     zero-stride={has_zero_stride}) — call .contiguous() \
                     before the fused loss",
                    t.len
                );
                std::process::abort();
            }
        }
        crate::cuda::inner::ensure_context();
        // Labels arrive in whatever form the loader/pipeline produced:
        // GPU f32 (batch dicts moved to device), CPU f64/f32, or narrow
        // CPU integer forms (int8 / u16 token streams). All are whole
        // numbers (class indices; ignore_index negatives round exactly).
        // Contiguity holds on every label path (post-reshape).
        let host_i64: Vec<i64> = match (t.device, t.dtype) {
            (1, crate::tensor::DTYPE_F32) => {
                let mut host_f32 = vec![0f32; n];
                crate::cuda::inner::memcpy_dtoh(
                    host_f32.as_mut_ptr() as *mut std::os::raw::c_void,
                    t.data,
                    n * 4,
                );
                host_f32.iter().map(|&v| v.round() as i64).collect()
            }
            (0, crate::tensor::DTYPE_F64) => {
                let s = unsafe { std::slice::from_raw_parts(t.data as *const f64, n) };
                s.iter().map(|&v| v.round() as i64).collect()
            }
            (0, crate::tensor::DTYPE_F32) => {
                let s = unsafe { std::slice::from_raw_parts(t.data as *const f32, n) };
                s.iter().map(|&v| v.round() as i64).collect()
            }
            // DataLoader batch tensors carry DTYPE_I32 (i32 payloads). The
            // historical tag-4 i32/int8 overload — which once garbled real
            // label streams into [t0,0,0,0,t1,0,0,0,...] when read as i8 —
            // was removed by the P4 item-16 dtype migration.
            (0, crate::tensor::DTYPE_I32) => {
                let s = unsafe { std::slice::from_raw_parts(t.data as *const i32, n) };
                s.iter().map(|&v| v as i64).collect()
            }
            (0, crate::tensor::DTYPE_U16_TOKEN) => {
                let s = unsafe { std::slice::from_raw_parts(t.data as *const u16, n) };
                s.iter().map(|&v| v as i64).collect()
            }
            (dev, dt) => {
                eprintln!(
                    "nsl_fused_lce_targets_i64_alloc: unsupported label tensor \
                     form (device={dev} dtype={dt})"
                );
                std::process::abort();
            }
        };
        let dev = crate::cuda::inner::alloc_managed(n * 8);
        crate::cuda::inner::memcpy_htod(
            dev,
            host_i64.as_ptr() as *const std::os::raw::c_void,
            n * 8,
        );
        dev as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("nsl_fused_lce_targets_i64_alloc: compiled without cuda feature");
        std::process::abort();
    }
}

/// Free a buffer produced by [`nsl_fused_lce_targets_i64_alloc`].
#[no_mangle]
pub extern "C" fn nsl_fused_lce_targets_i64_free(dev_ptr: i64) {
    #[cfg(feature = "cuda")]
    if dev_ptr != 0 {
        crate::cuda::inner::ensure_context();
        crate::cuda::inner::free_managed(dev_ptr as *mut std::os::raw::c_void);
    }
    #[cfg(not(feature = "cuda"))]
    let _ = dev_ptr;
}

/// Any value outside {0, 1, 2} is treated as F32 (defensive — preserves
/// forward-compat with un-recompiled callers).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_linear_ce_forward(
    ptx_ptr: i64,
    kname_ptr: i64,
    x_ptr: i64,
    w_ptr: i64,
    bias_ptr: i64,
    targets_ptr: i64,
    loss_out_ptr: i64,
    lse_out_ptr: i64,
    b: i64,
    s: i64,
    v: i64,
    h: i64,
    smem_bytes: i64,
    dtype_tag: i64,
) -> i64 {
    // CFTP v7: the v6 default-on dtype refusal has been LIFTED.  All four
    // preconditions in the module docstring are now satisfied (GPU cast
    // kernel, RTE rounding, cache reuse, V=49152 re-measured baselines)
    // and the production wengert path inserts the precision_cast op into
    // the Cranelift IR before this FFI is reached.  F32 stays byte-identical.
    // Setting `NSL_FUSED_LCE_REFUSE_NON_F32=1` reinstates the v6 refusal as
    // an operator-facing diagnostic override.
    if refuse_non_f32_if_unsafe(dtype_tag, "nsl_fused_linear_ce_forward") {
        return -1;
    }
    let _dtype_tag = dtype_tag;
    #[cfg(feature = "cuda")]
    {
        let rc = crate::cuda::fused_ce_kernels::launch_forward(
            ptx_ptr as *const u8,
            kname_ptr as *const u8,
            x_ptr as u64,
            w_ptr as u64,
            bias_ptr as u64,
            targets_ptr as u64,
            loss_out_ptr as u64,
            lse_out_ptr as u64,
            b as u32,
            s as u32,
            v as u32,
            h as u32,
            smem_bytes as u32,
        );
        if rc != 0 {
            eprintln!(
                "nsl_fused_linear_ce_forward: CUDA launch failed rc={}",
                rc
            );
            return -(rc as i64);
        }
        // p3: stream-ordered by default. Tests read results via memcpy_dtoh
        // (a NULL-stream barrier that self-synchronizes), so no eager sync here.
        #[cfg(feature = "cuda")]
        crate::cuda::inner::sync_after_kernel();
        FUSED_LCE_LAUNCH_COUNTS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx_ptr, kname_ptr, x_ptr, w_ptr, bias_ptr, targets_ptr,
                 loss_out_ptr, lse_out_ptr, b, s, v, h, smem_bytes);
        eprintln!("nsl_fused_linear_ce_forward: compiled without cuda feature");
        -1
    }
}

/// Sprint-3 large-vocab forward: two-kernel (per-tile partials + per-row finalize).
///
/// Use this FFI instead of `nsl_fused_linear_ce_forward` when
/// `FusedLinearCEConfig::is_large_vocab()` returns true (vocab > 8192).
/// The PTX module pointed at by `ptx_ptr` MUST contain BOTH kernels —
/// the codegen produces this in one byte string via
/// `synthesize_fused_linear_ce_ptx` whenever `is_large_vocab()` is true.
///
/// # Lifecycle of `partials_ptr`
/// Caller-owned: allocate `(B*S) * num_tiles * 2 * sizeof(f32)` bytes of
/// device memory before this call and free it after. The buffer is overwritten
/// by Kernel A and read by Kernel B; its contents on return are
/// implementation-defined (callers should not depend on them).
///
/// # Arguments
/// * `ptx_ptr`            — `*const u8`: null-terminated PTX string with both kernels.
/// * `partials_kname_ptr` — `*const u8`: null-terminated Kernel A name
///   (`FusedLinearCEConfig::large_partials_kernel_name`).
/// * `finalize_kname_ptr` — `*const u8`: null-terminated Kernel B name
///   (`FusedLinearCEConfig::large_finalize_kernel_name`).
/// * `x_ptr`        — device f32 `[B, S, H]`.
/// * `w_ptr`        — device f32 `[V, H]`.
/// * `bias_ptr`     — device f32 `[V]`.
/// * `targets_ptr`  — device i64 `[B*S]`.
/// * `partials_ptr` — device f32 `[B*S, num_tiles, 2]` (caller-owned scratch).
/// * `loss_out_ptr` — device f32 `[B*S]` output (pre-allocated).
/// * `lse_out_ptr`  — device f32 `[B*S]` output (pre-allocated; for backward reuse).
/// * `b, s, v, h`   — dims as i64.
/// * `num_tiles`    — `vocab_size.div_ceil(vocab_tile)` (host-computed).
/// * `smem_bytes`   — Kernel A per-CTA smem budget; Kernel B is launched with 0.
///
/// Returns 0 on success, negative on error.
///
/// # `dtype_tag` (Sprint v3-2 trailing arg; extended in v4-1)
///
/// `0` = F32 (default; preserves pre-Sprint-v3-2 ABI). `1` = F16 — caller
/// MUST allocate x/W/bias in HBM as fp16. `2` = Bf16 (Sprint v4-1) —
/// same HBM layout as F16 (16-bit storage). `partials_ptr` stays f32-sized
/// across all dtypes (`large_partials_bytes()` is dtype-independent).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_linear_ce_forward_large(
    ptx_ptr: i64,
    partials_kname_ptr: i64,
    finalize_kname_ptr: i64,
    x_ptr: i64,
    w_ptr: i64,
    bias_ptr: i64,
    targets_ptr: i64,
    partials_ptr: i64,
    loss_out_ptr: i64,
    lse_out_ptr: i64,
    b: i64,
    s: i64,
    v: i64,
    h: i64,
    num_tiles: i64,
    smem_bytes: i64,
    dtype_tag: i64,
) -> i64 {
    // CFTP v7: refusal lifted — see forward FFI above for the rationale.
    if refuse_non_f32_if_unsafe(dtype_tag, "nsl_fused_linear_ce_forward_large") {
        return -1;
    }
    let _dtype_tag = dtype_tag;
    #[cfg(feature = "cuda")]
    {
        let rc = crate::cuda::fused_ce_kernels::launch_forward_large(
            ptx_ptr as *const u8,
            partials_kname_ptr as *const u8,
            finalize_kname_ptr as *const u8,
            x_ptr as u64,
            w_ptr as u64,
            bias_ptr as u64,
            targets_ptr as u64,
            partials_ptr as u64,
            loss_out_ptr as u64,
            lse_out_ptr as u64,
            b as u32,
            s as u32,
            v as u32,
            h as u32,
            num_tiles as u32,
            smem_bytes as u32,
        );
        if rc != 0 {
            eprintln!("nsl_fused_linear_ce_forward_large: CUDA launch failed rc={rc}");
            return -(rc as i64);
        }
        crate::cuda::inner::sync_after_kernel(); // p3: stream-ordered by default
        FUSED_LCE_LAUNCH_COUNTS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            ptx_ptr, partials_kname_ptr, finalize_kname_ptr,
            x_ptr, w_ptr, bias_ptr, targets_ptr, partials_ptr,
            loss_out_ptr, lse_out_ptr,
            b, s, v, h, num_tiles, smem_bytes,
        );
        eprintln!("nsl_fused_linear_ce_forward_large: compiled without cuda feature");
        -1
    }
}

/// Backward: computes `dx[B,S,H]`, `dW[V,H]`, `dbias[V]` given saved lse.
///
/// # Arguments
/// * `ptx_ptr`        — `*const u8`: null-terminated backward PTX string.
/// * `kname_ptr`      — `*const u8`: null-terminated backward kernel name.
/// * `grad_output`    — upstream scalar gradient (f32, reinterpreted from i64 bits).
/// * `x_ptr`          — device f32 `[B, S, H]`.
/// * `w_ptr`          — device f32 `[V, H]`.
/// * `bias_ptr`       — device f32 `[V]`.
/// * `targets_ptr`    — device i64 `[B*S]`.
/// * `lse_ptr`        — device f32 `[B*S]` saved from forward.
/// * `dx_out_ptr`     — device f32 `[B, S, H]` (zeroed by caller).
/// * `dw_out_ptr`     — device f32 `[V, H]`    (zeroed by caller).
/// * `dbias_out_ptr`  — device f32 `[V]`        (zeroed by caller).
/// * `b, s, v, h`     — dims as i64.
/// * `num_valid`      — count of non-ignore positions (host-computed from targets).
/// * `smem_bytes`     — shared-memory bytes per CTA.
///
/// Returns 0 on success, negative on error.
///
/// # `dtype_tag` (Sprint v3-2 trailing arg; extended in v4-1)
///
/// `0` = F32 (default; preserves pre-Sprint-v3-2 ABI). `1` = F16 — caller
/// MUST allocate x/W/bias in HBM as fp16. `2` = Bf16 (Sprint v4-1) — same
/// HBM layout as F16 (16-bit storage). `lse_ptr` stays f32 across all
/// dtypes (forward writes f32 regardless of activation dtype); `dx_out` /
/// `dw_out` / `dbias_out` stay f32 regardless of dtype_tag (PyTorch
/// master-gradient convention — the kernel emits `red.global.add.f32`
/// for cross-CTA accumulation).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_linear_ce_backward(
    ptx_ptr: i64,
    kname_ptr: i64,
    grad_output_bits: i64, // f32 bits packed into i64
    x_ptr: i64,
    w_ptr: i64,
    bias_ptr: i64,
    targets_ptr: i64,
    lse_ptr: i64,
    dx_out_ptr: i64,
    dw_out_ptr: i64,
    dbias_out_ptr: i64,
    b: i64,
    s: i64,
    v: i64,
    h: i64,
    num_valid: i64,
    smem_bytes: i64,
    dtype_tag: i64,
) -> i64 {
    // CFTP v7: refusal lifted — see forward FFI above for the rationale.
    if refuse_non_f32_if_unsafe(dtype_tag, "nsl_fused_linear_ce_backward") {
        return -1;
    }
    let grad_output = f32::from_bits(grad_output_bits as u32);
    let _dtype_tag = dtype_tag;

    #[cfg(feature = "cuda")]
    {
        let rc = crate::cuda::fused_ce_kernels::launch_backward(
            ptx_ptr as *const u8,
            kname_ptr as *const u8,
            grad_output,
            x_ptr as u64,
            w_ptr as u64,
            bias_ptr as u64,
            targets_ptr as u64,
            lse_ptr as u64,
            dx_out_ptr as u64,
            dw_out_ptr as u64,
            dbias_out_ptr as u64,
            b as u32,
            s as u32,
            v as u32,
            h as u32,
            num_valid as u32,
            smem_bytes as u32,
        );
        if rc != 0 {
            eprintln!(
                "nsl_fused_linear_ce_backward: CUDA launch failed rc={}",
                rc
            );
            return -(rc as i64);
        }
        #[cfg(feature = "cuda")]
        crate::cuda::inner::sync_after_kernel(); // p3: stream-ordered by default
        FUSED_LCE_LAUNCH_COUNTS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx_ptr, kname_ptr, grad_output, x_ptr, w_ptr, bias_ptr,
                 targets_ptr, lse_ptr, dx_out_ptr, dw_out_ptr, dbias_out_ptr,
                 b, s, v, h, num_valid, smem_bytes);
        eprintln!("nsl_fused_linear_ce_backward: compiled without cuda feature");
        -1
    }
}

#[cfg(test)]
mod v7_default_activation_tests {
    //! CFTP v7: the v6 default-on FFI dtype refusal is LIFTED.  By default,
    //! `dtype_tag != 0` proceeds straight to the kernel because all four
    //! v7 preconditions (GPU PTX cast, RTE rounding, cache reuse,
    //! re-measured V=49152 baselines) are satisfied and the wengert
    //! lowering inserts an explicit `nsl_tensor_to_{bf16,fp16}` op into
    //! the IR before the FFI is reached.
    //!
    //! `NSL_FUSED_LCE_REFUSE_NON_F32=1` remains respected as an opt-in
    //! diagnostic / safety override for confirming a suspected
    //! dtype-conformance regression without rebuilding the runtime.

    /// Tests serialize on this mutex so env-var manipulation does not race
    /// across parallel test threads.  std::env::set_var is not thread-safe.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard for the v7 diagnostic-override env var.
    struct EnvGuard {
        prev: Option<std::ffi::OsString>,
    }
    impl EnvGuard {
        fn set(value: Option<&str>) -> Self {
            let prev = std::env::var_os("NSL_FUSED_LCE_REFUSE_NON_F32");
            match value {
                Some(v) => std::env::set_var("NSL_FUSED_LCE_REFUSE_NON_F32", v),
                None => std::env::remove_var("NSL_FUSED_LCE_REFUSE_NON_F32"),
            }
            EnvGuard { prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("NSL_FUSED_LCE_REFUSE_NON_F32", v),
                None => std::env::remove_var("NSL_FUSED_LCE_REFUSE_NON_F32"),
            }
        }
    }

    /// F32 (dtype_tag=0) MUST NOT trigger the refusal — this pins the
    /// byte-identity contract that the F32 fast path is never gated on
    /// env vars.
    #[test]
    fn f32_dtype_tag_never_refuses() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(None);
        assert!(
            !super::refuse_non_f32_if_unsafe(0, "test"),
            "dtype_tag=0 must never refuse — F32 byte-identity contract"
        );
        // Even with the diagnostic override on, F32 must still proceed.
        let _g2 = EnvGuard::set(Some("1"));
        assert!(
            !super::refuse_non_f32_if_unsafe(0, "test"),
            "F32 byte-identity contract holds even with diagnostic override on"
        );
    }

    /// By default (no env var set), the v7 helper MUST allow dtype_tag=1/2
    /// through — the GPU PTX cast kernel + wengert-inserted precision_cast
    /// make the path safe without an opt-in.
    #[test]
    fn bf16_dtype_tag_activates_by_default() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(None);
        assert!(
            !super::refuse_non_f32_if_unsafe(2, "test"),
            "v7 default: dtype_tag=2 (bf16) must proceed to the kernel"
        );
    }

    /// Mirror for fp16 (dtype_tag=1).
    #[test]
    fn fp16_dtype_tag_activates_by_default() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(None);
        assert!(
            !super::refuse_non_f32_if_unsafe(1, "test"),
            "v7 default: dtype_tag=1 (fp16) must proceed to the kernel"
        );
    }

    /// Diagnostic-override path: NSL_FUSED_LCE_REFUSE_NON_F32=1 reinstates
    /// the v6 refusal for diagnosing a downstream regression without
    /// rebuilding the runtime.
    #[test]
    fn diagnostic_override_refuses_bf16_dtype_tag() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(Some("1"));
        assert!(
            super::refuse_non_f32_if_unsafe(2, "test"),
            "NSL_FUSED_LCE_REFUSE_NON_F32=1 must reinstate the v6 refusal for dtype_tag=2"
        );
        assert!(
            super::refuse_non_f32_if_unsafe(1, "test"),
            "NSL_FUSED_LCE_REFUSE_NON_F32=1 must reinstate the v6 refusal for dtype_tag=1"
        );
    }

    /// Diagnostic-override applied at the forward FFI entry point: the
    /// short-circuit returns -1 BEFORE the launcher block, so null
    /// pointer args are safe.
    #[test]
    fn ffi_forward_honors_diagnostic_override() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(Some("1"));
        let rc = super::nsl_fused_linear_ce_forward(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2,
        );
        assert_eq!(
            rc, -1,
            "FFI must short-circuit with -1 when diagnostic override is set"
        );
    }

    /// Same for backward.
    #[test]
    fn ffi_backward_honors_diagnostic_override() {
        let _lk = ENV_MUTEX.lock().unwrap();
        let _g = EnvGuard::set(Some("1"));
        let rc = super::nsl_fused_linear_ce_backward(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 2,
        );
        assert_eq!(
            rc, -1,
            "backward FFI must short-circuit with -1 when diagnostic override is set"
        );
    }
}
