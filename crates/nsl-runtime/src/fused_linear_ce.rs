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
#[no_mangle]
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
) -> i64 {
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
#[no_mangle]
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
) -> i64 {
    let grad_output = f32::from_bits(grad_output_bits as u32);

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
