//! CPKD fused KL-CE distillation loss — public FFI surface.
//!
//! The Cranelift-lowered distill step calls these with PTX synthesised at
//! compile time by `nsl_codegen::cpkd_fused_loss` and embedded in the
//! binary's .rodata (same pattern as `fused_linear_ce.rs`).
//!
//! ## ABI lock-step contract (16-vs-18-arg lesson)
//!
//! The arg lists here, the declarations in
//! `nsl-codegen/src/builtins.rs`, and the call sites in
//! `nsl-codegen/src/wengert_lower.rs` MUST agree on count and order —
//! there is no compile-time cross-check. Forward = 20 i64 args, backward
//! = 23 i64 args. Scalars (`alpha`, `temperature`, `grad_output`) travel
//! as f32 bits in the low 32 bits of an i64.
//!
//! ## No teacher gradients (I-11)
//!
//! There are no teacher-gradient buffers anywhere in this ABI: the
//! backward writes ONLY `dxs/dws/dbs` (student). Teacher-gradient
//! accumulation (failure mode F-06) is impossible by construction.
//!
//! v1 is f32-only and GPU-only: the CPU path for distillation losses is
//! the stdlib composite `fused_kl_ce` via tape AD (outside distill
//! blocks). Calling these without the `cuda` feature is a loud error.

#[cfg(feature = "cuda")]
#[inline]
fn f32_from_bits_i64(bits: i64) -> f32 {
    f32::from_bits((bits as u64 & 0xFFFF_FFFF) as u32)
}

/// Forward: per-row loss + the three saved LSEs.
///
/// * `ptx_ptr`/`kname_ptr` — null-terminated PTX module + kernel name.
/// * `xs/ws/bs` — student hidden `[rows, HS]`, LM head `[V, HS]`, bias `[V]` (f32 device ptrs).
/// * `xt/wt/bt` — teacher analogues (`[rows, HT]`, `[V, HT]`, `[V]`).
/// * `targets`  — `[rows]` i64 device ptr.
/// * `loss_out` — `[rows]` f32 (pre-allocated).
/// * `lse_s1_out/lse_st_out/lse_tt_out` — `[rows]` f32 (pre-allocated; backward inputs).
/// * `rows/v/hs/ht` — dims.
/// * `alpha_bits/temp_bits` — f32 bits in i64.
/// * `smem_bytes` — from `FusedKlCeConfig::shared_mem_bytes()`.
///
/// Returns 0 on success, negative on error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_kl_ce_forward(
    ptx_ptr: i64,
    kname_ptr: i64,
    xs_ptr: i64,
    ws_ptr: i64,
    bs_ptr: i64,
    xt_ptr: i64,
    wt_ptr: i64,
    bt_ptr: i64,
    targets_ptr: i64,
    loss_out_ptr: i64,
    lse_s1_out_ptr: i64,
    lse_st_out_ptr: i64,
    lse_tt_out_ptr: i64,
    rows: i64,
    v: i64,
    hs: i64,
    ht: i64,
    alpha_bits: i64,
    temp_bits: i64,
    smem_bytes: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let rc = crate::cuda::fused_kl_ce_kernels::launch_forward(
            ptx_ptr as *const u8,
            kname_ptr as *const u8,
            xs_ptr as u64,
            ws_ptr as u64,
            bs_ptr as u64,
            xt_ptr as u64,
            wt_ptr as u64,
            bt_ptr as u64,
            targets_ptr as u64,
            loss_out_ptr as u64,
            lse_s1_out_ptr as u64,
            lse_st_out_ptr as u64,
            lse_tt_out_ptr as u64,
            rows as u32,
            v as u32,
            hs as u32,
            ht as u32,
            f32_from_bits_i64(alpha_bits),
            f32_from_bits_i64(temp_bits),
            smem_bytes as u32,
        );
        if rc != 0 {
            eprintln!("nsl_fused_kl_ce_forward: CUDA launch failed rc={rc}");
            return -(rc as i64);
        }
        crate::cuda::inner::sync_after_kernel(); // p3: stream-ordered by default
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            ptx_ptr,
            kname_ptr,
            xs_ptr,
            ws_ptr,
            bs_ptr,
            xt_ptr,
            wt_ptr,
            bt_ptr,
            targets_ptr,
            loss_out_ptr,
            lse_s1_out_ptr,
            lse_st_out_ptr,
            lse_tt_out_ptr,
            rows,
            v,
            hs,
            ht,
            alpha_bits,
            temp_bits,
            smem_bytes,
        );
        eprintln!(
            "nsl_fused_kl_ce_forward: compiled without the cuda feature; the fused \
             KL-CE distillation loss is GPU-only in v1 (use the stdlib composite \
             fused_kl_ce on CPU)"
        );
        -1
    }
}

/// Backward: student gradients only (`dxs/dws/dbs`; caller zero-fills —
/// the kernel accumulates via `red.global.add.f32`).
///
/// Returns 0 on success, negative on error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_fused_kl_ce_backward(
    ptx_ptr: i64,
    kname_ptr: i64,
    grad_output_bits: i64,
    xs_ptr: i64,
    ws_ptr: i64,
    bs_ptr: i64,
    xt_ptr: i64,
    wt_ptr: i64,
    bt_ptr: i64,
    targets_ptr: i64,
    lse_s1_ptr: i64,
    lse_st_ptr: i64,
    lse_tt_ptr: i64,
    dxs_out_ptr: i64,
    dws_out_ptr: i64,
    dbs_out_ptr: i64,
    rows: i64,
    v: i64,
    hs: i64,
    ht: i64,
    alpha_bits: i64,
    temp_bits: i64,
    num_valid: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        if num_valid <= 0 {
            // All rows ignored: gradients are identically zero and the
            // caller already zero-filled the buffers. Launching would
            // divide by zero in the scale computation.
            return 0;
        }
        let rc = crate::cuda::fused_kl_ce_kernels::launch_backward(
            ptx_ptr as *const u8,
            kname_ptr as *const u8,
            f32_from_bits_i64(grad_output_bits),
            xs_ptr as u64,
            ws_ptr as u64,
            bs_ptr as u64,
            xt_ptr as u64,
            wt_ptr as u64,
            bt_ptr as u64,
            targets_ptr as u64,
            lse_s1_ptr as u64,
            lse_st_ptr as u64,
            lse_tt_ptr as u64,
            dxs_out_ptr as u64,
            dws_out_ptr as u64,
            dbs_out_ptr as u64,
            rows as u32,
            v as u32,
            hs as u32,
            ht as u32,
            f32_from_bits_i64(alpha_bits),
            f32_from_bits_i64(temp_bits),
            num_valid as u32,
        );
        if rc != 0 {
            eprintln!("nsl_fused_kl_ce_backward: CUDA launch failed rc={rc}");
            return -(rc as i64);
        }
        crate::cuda::inner::sync_after_kernel(); // p3: stream-ordered by default
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            ptx_ptr,
            kname_ptr,
            grad_output_bits,
            xs_ptr,
            ws_ptr,
            bs_ptr,
            xt_ptr,
            wt_ptr,
            bt_ptr,
            targets_ptr,
            lse_s1_ptr,
            lse_st_ptr,
            lse_tt_ptr,
            dxs_out_ptr,
            dws_out_ptr,
            dbs_out_ptr,
            rows,
            v,
            hs,
            ht,
            alpha_bits,
            temp_bits,
            num_valid,
        );
        eprintln!(
            "nsl_fused_kl_ce_backward: compiled without the cuda feature; the fused \
             KL-CE distillation loss is GPU-only in v1"
        );
        -1
    }
}
