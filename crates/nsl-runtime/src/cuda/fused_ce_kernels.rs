//! Fused linear-CE kernel launcher (G3, v1).
//!
//! This module provides the CUDA kernel launch wrappers for
//! `nsl_fused_linear_ce_forward` and `nsl_fused_linear_ce_backward`.
//!
//! The PTX strings are provided by the caller (synthesised via
//! `nsl_codegen::fused_linear_ce`) so that this crate does not need to
//! depend on nsl-codegen.  The public FFI functions in
//! `crates/nsl-runtime/src/fused_linear_ce.rs` receive a pre-computed PTX
//! pointer and forward it here.

// PTX launch helpers are only available when the `cuda` feature is present.
#![allow(dead_code)]

#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// Launch the forward fused linear-CE kernel.
///
/// # Parameters
/// * `ptx_ptr`       — null-terminated PTX string for the forward kernel.
/// * `kernel_name`   — null-terminated kernel name matching the PTX entry.
/// * `x_dev`         — device pointer: `x[B, S, H]` f32 row-major.
/// * `w_dev`         — device pointer: `W[V, H]`   f32 row-major.
/// * `bias_dev`      — device pointer: `bias[V]`   f32.
/// * `targets_dev`   — device pointer: `targets[B*S]` i64.
/// * `loss_out_dev`  — device pointer: output `loss[B*S]` f32.
/// * `lse_out_dev`   — device pointer: output `lse[B*S]`  f32 (for backward).
/// * `b, s, v, h`    — dims.
/// * `smem_bytes`    — shared-memory bytes per CTA.
///
/// Returns `CUresult` as `u32` (0 = success).
#[cfg(feature = "cuda")]
pub(crate) fn launch_forward(
    ptx_ptr: *const u8,
    kernel_name: *const u8,
    x_dev: u64,
    w_dev: u64,
    bias_dev: u64,
    targets_dev: u64,
    loss_out_dev: u64,
    lse_out_dev: u64,
    b: u32,
    s: u32,
    v: u32,
    h: u32,
    smem_bytes: u32,
) -> u32 {
    let mut x_ptr = x_dev;
    let mut w_ptr = w_dev;
    let mut bias_ptr = bias_dev;
    let mut tgt_ptr = targets_dev;
    let mut loss_ptr = loss_out_dev;
    let mut lse_ptr = lse_out_dev;
    let mut b_val = b;
    let mut s_val = s;
    let mut v_val = v;
    let mut h_val = h;

    let args: [*mut c_void; 10] = [
        &mut x_ptr    as *mut _ as *mut c_void,
        &mut w_ptr    as *mut _ as *mut c_void,
        &mut bias_ptr as *mut _ as *mut c_void,
        &mut tgt_ptr  as *mut _ as *mut c_void,
        &mut loss_ptr as *mut _ as *mut c_void,
        &mut lse_ptr  as *mut _ as *mut c_void,
        &mut b_val    as *mut _ as *mut c_void,
        &mut s_val    as *mut _ as *mut c_void,
        &mut v_val    as *mut _ as *mut c_void,
        &mut h_val    as *mut _ as *mut c_void,
    ];

    let grid_x = (b as i64) * (s as i64);
    let result = crate::cuda::inner::kernel_launch(
        ptx_ptr,
        kernel_name,
        [grid_x, 1, 1],
        [128, 1, 1],
        &args,
        smem_bytes,
    );
    result as u32
}

/// Launch the backward fused linear-CE kernel.
///
/// # Extra parameters vs forward
/// * `grad_output`  — scalar f32 upstream gradient.
/// * `lse_dev`      — saved lse buffer from forward (device pointer).
/// * `dx_out_dev`   — device pointer: output `dx[B, S, H]` f32 (zeroed before call).
/// * `dw_out_dev`   — device pointer: output `dW[V, H]`    f32 (zeroed before call).
/// * `dbias_out_dev`— device pointer: output `dbias[V]`    f32 (zeroed before call).
/// * `num_valid`    — count of non-ignore_index positions in targets (host-computed).
#[cfg(feature = "cuda")]
pub(crate) fn launch_backward(
    ptx_ptr: *const u8,
    kernel_name: *const u8,
    grad_output: f32,
    x_dev: u64,
    w_dev: u64,
    bias_dev: u64,
    targets_dev: u64,
    lse_dev: u64,
    dx_out_dev: u64,
    dw_out_dev: u64,
    dbias_out_dev: u64,
    b: u32,
    s: u32,
    v: u32,
    h: u32,
    num_valid: u32,
    smem_bytes: u32,
) -> u32 {
    let mut go = grad_output;
    let mut x_ptr = x_dev;
    let mut w_ptr = w_dev;
    let mut bias_ptr = bias_dev;
    let mut tgt_ptr = targets_dev;
    let mut lse_ptr = lse_dev;
    let mut dx_ptr = dx_out_dev;
    let mut dw_ptr = dw_out_dev;
    let mut dbias_ptr = dbias_out_dev;
    let mut b_val = b;
    let mut s_val = s;
    let mut v_val = v;
    let mut h_val = h;
    let mut nv_val = num_valid;

    let args: [*mut c_void; 14] = [
        &mut go       as *mut _ as *mut c_void,
        &mut x_ptr    as *mut _ as *mut c_void,
        &mut w_ptr    as *mut _ as *mut c_void,
        &mut bias_ptr as *mut _ as *mut c_void,
        &mut tgt_ptr  as *mut _ as *mut c_void,
        &mut lse_ptr  as *mut _ as *mut c_void,
        &mut dx_ptr   as *mut _ as *mut c_void,
        &mut dw_ptr   as *mut _ as *mut c_void,
        &mut dbias_ptr as *mut _ as *mut c_void,
        &mut b_val    as *mut _ as *mut c_void,
        &mut s_val    as *mut _ as *mut c_void,
        &mut v_val    as *mut _ as *mut c_void,
        &mut h_val    as *mut _ as *mut c_void,
        &mut nv_val   as *mut _ as *mut c_void,
    ];

    let grid_x = (b as i64) * (s as i64);
    let result = crate::cuda::inner::kernel_launch(
        ptx_ptr,
        kernel_name,
        [grid_x, 1, 1],
        [128, 1, 1],
        &args,
        smem_bytes,
    );
    result as u32
}
