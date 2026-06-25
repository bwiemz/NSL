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
    // dtype_tag is currently informational; the host launcher's element-size
    // handling is implicit in the PTX bytes the caller supplied (the codegen
    // already specialized to the right dtype). The argument is plumbed so
    // future host-side hooks (e.g. SMEM cap raising for fp16) can branch on
    // it without another ABI change.
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
        // Synchronise so the test can read results immediately.
        #[cfg(feature = "cuda")]
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize();
        }
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
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
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
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize();
        }
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
