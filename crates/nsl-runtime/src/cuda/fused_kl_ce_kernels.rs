//! CPKD fused KL-CE distillation-loss kernel launchers.
//!
//! Same layering as `fused_ce_kernels.rs`: PTX strings are synthesised by
//! `nsl_codegen::cpkd_fused_loss` and arrive here as pre-computed pointers
//! through the public FFI in `crates/nsl-runtime/src/fused_kl_ce.rs`, so
//! this crate never depends on nsl-codegen.

#![allow(dead_code)]

#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// Launch the forward fused KL-CE kernel.
///
/// Grid `(rows, 1, 1)`, block `(128, 1, 1)`.
///
/// # Parameters
/// * `xs/ws/bs` — student hidden `[rows, HS]`, LM head `[V, HS]`, bias `[V]` (f32 device).
/// * `xt/wt/bt` — teacher analogues (`HT` may differ from `HS`).
/// * `targets_dev` — `[rows]` i64.
/// * `loss_out_dev` — `[rows]` f32 per-row loss (pre-allocated).
/// * `lse_s1/lse_st/lse_tt` — `[rows]` f32 saved LSEs for backward (pre-allocated).
/// * `alpha/temp` — runtime loss hyperparameters.
///
/// Returns `CUresult` as `u32` (0 = success).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_forward(
    ptx_ptr: *const u8,
    kernel_name: *const u8,
    xs_dev: u64,
    ws_dev: u64,
    bs_dev: u64,
    xt_dev: u64,
    wt_dev: u64,
    bt_dev: u64,
    targets_dev: u64,
    loss_out_dev: u64,
    lse_s1_out_dev: u64,
    lse_st_out_dev: u64,
    lse_tt_out_dev: u64,
    rows: u32,
    v: u32,
    hs: u32,
    ht: u32,
    alpha: f32,
    temp: f32,
    smem_bytes: u32,
) -> u32 {
    let mut xs_ptr = xs_dev;
    let mut ws_ptr = ws_dev;
    let mut bs_ptr = bs_dev;
    let mut xt_ptr = xt_dev;
    let mut wt_ptr = wt_dev;
    let mut bt_ptr = bt_dev;
    let mut tgt_ptr = targets_dev;
    let mut loss_ptr = loss_out_dev;
    let mut lse_s1_ptr = lse_s1_out_dev;
    let mut lse_st_ptr = lse_st_out_dev;
    let mut lse_tt_ptr = lse_tt_out_dev;
    let mut rows_val = rows;
    let mut v_val = v;
    let mut hs_val = hs;
    let mut ht_val = ht;
    let mut alpha_val = alpha;
    let mut temp_val = temp;

    let args: [*mut c_void; 17] = [
        &mut xs_ptr as *mut _ as *mut c_void,
        &mut ws_ptr as *mut _ as *mut c_void,
        &mut bs_ptr as *mut _ as *mut c_void,
        &mut xt_ptr as *mut _ as *mut c_void,
        &mut wt_ptr as *mut _ as *mut c_void,
        &mut bt_ptr as *mut _ as *mut c_void,
        &mut tgt_ptr as *mut _ as *mut c_void,
        &mut loss_ptr as *mut _ as *mut c_void,
        &mut lse_s1_ptr as *mut _ as *mut c_void,
        &mut lse_st_ptr as *mut _ as *mut c_void,
        &mut lse_tt_ptr as *mut _ as *mut c_void,
        &mut rows_val as *mut _ as *mut c_void,
        &mut v_val as *mut _ as *mut c_void,
        &mut hs_val as *mut _ as *mut c_void,
        &mut ht_val as *mut _ as *mut c_void,
        &mut alpha_val as *mut _ as *mut c_void,
        &mut temp_val as *mut _ as *mut c_void,
    ];

    let result = crate::cuda::inner::kernel_launch(
        ptx_ptr,
        kernel_name,
        [rows as i64, 1, 1],
        [128, 1, 1],
        &args,
        smem_bytes,
    );
    result as u32
}

/// Launch the backward fused KL-CE kernel.
///
/// STUDENT gradients only — the kernel ABI has no teacher-gradient
/// outputs (CPKD invariant I-11). `dxs/dws/dbs` must be zero-filled by
/// the caller (`red.global.add` accumulation).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_backward(
    ptx_ptr: *const u8,
    kernel_name: *const u8,
    grad_output: f32,
    xs_dev: u64,
    ws_dev: u64,
    bs_dev: u64,
    xt_dev: u64,
    wt_dev: u64,
    bt_dev: u64,
    targets_dev: u64,
    lse_s1_dev: u64,
    lse_st_dev: u64,
    lse_tt_dev: u64,
    dxs_out_dev: u64,
    dws_out_dev: u64,
    dbs_out_dev: u64,
    rows: u32,
    v: u32,
    hs: u32,
    ht: u32,
    alpha: f32,
    temp: f32,
    num_valid: u32,
) -> u32 {
    let mut go = grad_output;
    let mut xs_ptr = xs_dev;
    let mut ws_ptr = ws_dev;
    let mut bs_ptr = bs_dev;
    let mut xt_ptr = xt_dev;
    let mut wt_ptr = wt_dev;
    let mut bt_ptr = bt_dev;
    let mut tgt_ptr = targets_dev;
    let mut lse_s1_ptr = lse_s1_dev;
    let mut lse_st_ptr = lse_st_dev;
    let mut lse_tt_ptr = lse_tt_dev;
    let mut dxs_ptr = dxs_out_dev;
    let mut dws_ptr = dws_out_dev;
    let mut dbs_ptr = dbs_out_dev;
    let mut rows_val = rows;
    let mut v_val = v;
    let mut hs_val = hs;
    let mut ht_val = ht;
    let mut alpha_val = alpha;
    let mut temp_val = temp;
    let mut nv_val = num_valid;

    let args: [*mut c_void; 21] = [
        &mut go as *mut _ as *mut c_void,
        &mut xs_ptr as *mut _ as *mut c_void,
        &mut ws_ptr as *mut _ as *mut c_void,
        &mut bs_ptr as *mut _ as *mut c_void,
        &mut xt_ptr as *mut _ as *mut c_void,
        &mut wt_ptr as *mut _ as *mut c_void,
        &mut bt_ptr as *mut _ as *mut c_void,
        &mut tgt_ptr as *mut _ as *mut c_void,
        &mut lse_s1_ptr as *mut _ as *mut c_void,
        &mut lse_st_ptr as *mut _ as *mut c_void,
        &mut lse_tt_ptr as *mut _ as *mut c_void,
        &mut dxs_ptr as *mut _ as *mut c_void,
        &mut dws_ptr as *mut _ as *mut c_void,
        &mut dbs_ptr as *mut _ as *mut c_void,
        &mut rows_val as *mut _ as *mut c_void,
        &mut v_val as *mut _ as *mut c_void,
        &mut hs_val as *mut _ as *mut c_void,
        &mut ht_val as *mut _ as *mut c_void,
        &mut alpha_val as *mut _ as *mut c_void,
        &mut temp_val as *mut _ as *mut c_void,
        &mut nv_val as *mut _ as *mut c_void,
    ];

    let result = crate::cuda::inner::kernel_launch(
        ptx_ptr,
        kernel_name,
        [rows as i64, 1, 1],
        [128, 1, 1],
        &args,
        0, // backward uses no SMEM
    );
    result as u32
}
