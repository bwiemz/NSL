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

/// Launch the Sprint-3 two-kernel large-vocab forward path.
///
/// The PTX module must contain BOTH `partials_kernel_name` (Kernel A) and
/// `finalize_kernel_name` (Kernel B) as separate `.visible .entry`s. The
/// helper loads the module once, launches A on the NULL stream with grid
/// `(num_tiles, B*S, 1)`, then launches B with grid `(B*S, 1, 1)` — the
/// NULL stream's implicit serialisation guarantees B observes A's writes.
///
/// The `partials` buffer is owned and lifecycled by the caller (typically
/// allocated/freed by the FFI wrapper). Its size MUST match
/// `FusedLinearCEConfig::large_partials_bytes()` =
/// `(B*S) * num_tiles * 2 * 4`. The kernel writes 2 f32s per (row, tile)
/// slot: `[tile_max, tile_sum_unscaled]`.
///
/// `smem_bytes` is the per-CTA shared-memory budget — same value as v1's
/// `shared_mem_bytes()` is correct for Kernel A (Kernel B uses 0 smem; the
/// per-launch driver attribute is set on the larger of the two budgets so
/// the same value is safe for both launches).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_forward_large(
    ptx_ptr: *const u8,
    partials_kname_ptr: *const u8,
    finalize_kname_ptr: *const u8,
    x_dev: u64,
    w_dev: u64,
    bias_dev: u64,
    targets_dev: u64,
    partials_dev: u64,
    loss_out_dev: u64,
    lse_out_dev: u64,
    b: u32,
    s: u32,
    v: u32,
    h: u32,
    num_tiles: u32,
    smem_bytes: u32,
) -> u32 {
    // ── Kernel A: per-tile partials ───────────────────────────────────────
    {
        let mut x_ptr = x_dev;
        let mut w_ptr = w_dev;
        let mut bias_ptr = bias_dev;
        let mut tgt_ptr = targets_dev;
        let mut part_ptr = partials_dev;
        let mut b_val = b;
        let mut s_val = s;
        let mut v_val = v;
        let mut h_val = h;
        let mut nt_val = num_tiles;

        let args: [*mut c_void; 10] = [
            &mut x_ptr    as *mut _ as *mut c_void,
            &mut w_ptr    as *mut _ as *mut c_void,
            &mut bias_ptr as *mut _ as *mut c_void,
            &mut tgt_ptr  as *mut _ as *mut c_void,
            &mut part_ptr as *mut _ as *mut c_void,
            &mut b_val    as *mut _ as *mut c_void,
            &mut s_val    as *mut _ as *mut c_void,
            &mut v_val    as *mut _ as *mut c_void,
            &mut h_val    as *mut _ as *mut c_void,
            &mut nt_val   as *mut _ as *mut c_void,
        ];

        let grid_y = (b as i64) * (s as i64);
        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr,
            partials_kname_ptr,
            [num_tiles as i64, grid_y, 1],
            [128, 1, 1],
            &args,
            smem_bytes,
        );
        if result as u32 != 0 {
            return result as u32;
        }
    }

    // ── Kernel B: per-row finalize ────────────────────────────────────────
    {
        let mut x_ptr = x_dev;
        let mut w_ptr = w_dev;
        let mut bias_ptr = bias_dev;
        let mut tgt_ptr = targets_dev;
        let mut part_ptr = partials_dev;
        let mut loss_ptr = loss_out_dev;
        let mut lse_ptr = lse_out_dev;
        let mut b_val = b;
        let mut s_val = s;
        let mut v_val = v;
        let mut h_val = h;
        let mut nt_val = num_tiles;

        let args: [*mut c_void; 12] = [
            &mut x_ptr    as *mut _ as *mut c_void,
            &mut w_ptr    as *mut _ as *mut c_void,
            &mut bias_ptr as *mut _ as *mut c_void,
            &mut tgt_ptr  as *mut _ as *mut c_void,
            &mut part_ptr as *mut _ as *mut c_void,
            &mut loss_ptr as *mut _ as *mut c_void,
            &mut lse_ptr  as *mut _ as *mut c_void,
            &mut b_val    as *mut _ as *mut c_void,
            &mut s_val    as *mut _ as *mut c_void,
            &mut v_val    as *mut _ as *mut c_void,
            &mut h_val    as *mut _ as *mut c_void,
            &mut nt_val   as *mut _ as *mut c_void,
        ];

        let grid_x = (b as i64) * (s as i64);
        // Kernel B uses 0 smem (per-row finalize is single-thread). Pass 0
        // so the driver doesn't reserve unnecessary shared memory.
        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr,
            finalize_kname_ptr,
            [grid_x, 1, 1],
            [128, 1, 1],
            &args,
            0,
        );
        if result as u32 != 0 {
            return result as u32;
        }
    }

    0
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

// ─── GEMM-chunked fused linear-CE (Sprint 2.5 wiring) ───────────────────────
//
// Production path for large vocabularies. The v1 scalar kernels above are
// numerically validated but computationally disqualified at real head shapes:
// measured at rows=1024, V=49152, H=512 (Coder-50M), v1 runs 486 ms forward /
// 2222 ms backward where the cuBLAS-chunk loop below runs the same math as
// ~a dozen GEMMs (~1.4 ms fwd / ~4.3 ms bwd at TF32 rate) plus O(rows·V)
// elementwise glue — see crates/nsl-codegen/tests/fused_linear_ce_perf_probe.rs.
//
// Never materializes the full [rows, V] logits: one [rows, chunk] scratch is
// reused across chunks (chunk = 4096 columns), with per-row online-softmax
// state folded chunk by chunk (LCE_CHUNK_STATS_F32_PTX). The GEMMs route
// through `cublas_inner`, so the NSL_MATMUL_TF32 / NSL_MATMUL_BF16 math modes
// apply to the head exactly as they do to every other product.
//
// f32 storage only (dtype_tag 0): the 16-bit storage variants keep the v1
// kernels (their operands are 16-bit HBM buffers the GEMM wrappers cannot
// consume).

/// Vocab columns per chunk. 4096 keeps the logits scratch at rows*16 KiB
/// (16.8 MB at rows=1024, 134 MB at rows=8192) and the per-chunk GEMM deep
/// enough that the chunk-stats pass is a small fraction of the GEMM.
#[cfg(feature = "cuda")]
const LCE_GEMM_CHUNK: u64 = 4096;

/// Online-softmax forward over vocab chunks.
///
/// All pointers are raw DEVICE pointers: `x [rows, h]`, `w [v, h]`,
/// `bias [v]` (read only when `has_bias`), `targets [rows]` i64,
/// `loss_out`/`lse_out [rows]` f32 (pre-allocated; fully overwritten).
/// Returns 0 on success.
///
/// PRECONDITION (unenforceable here): `x`, `w` and `bias` must be f32
/// storage. Unlike `nsl_fused_linear_ce_forward`, the `_gemm` FFI pair has no
/// `dtype_tag` parameter at all — there is no 16-bit variant of this path and
/// no `NslTensor` in the signature to read a dtype from, so nothing can
/// refuse a narrower operand. The GEMMs are issued with f32 leading
/// dimensions and the logits scratch is sized `rows * chunk * 4`; a 16-bit
/// operand would be read at twice its allocated length. The check belongs at
/// `nsl_fused_linear_ce_{forward,backward}_gemm`'s caller, which is the last
/// frame that still holds tensors.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_forward(
    x_dev: u64,
    w_dev: u64,
    bias_dev: u64,
    targets_dev: u64,
    loss_out_dev: u64,
    lse_out_dev: u64,
    rows: u64,
    v: u64,
    h: u64,
    has_bias: bool,
) -> u32 {
    use crate::cuda::inner;
    let chunk = LCE_GEMM_CHUNK.min(v);
    let logits_buf = inner::alloc_managed((rows * chunk * 4) as usize) as u64;
    let mstate = inner::alloc_managed((rows * 4) as usize) as u64;
    let sstate = inner::alloc_managed((rows * 4) as usize) as u64;
    let tlstate = inner::alloc_managed((rows * 4) as usize) as u64;
    let free_all = |rc: u32| -> u32 {
        inner::free_managed(logits_buf as *mut c_void);
        inner::free_managed(mstate as *mut c_void);
        inner::free_managed(sstate as *mut c_void);
        inner::free_managed(tlstate as *mut c_void);
        rc
    };
    // m = -inf (bit pattern 0xFF800000), s = 0, tl = 0 — stream-ordered so
    // the first chunk-stats launch observes them.
    unsafe {
        inner::ensure_context();
        let r = cudarc::driver::sys::cuMemsetD32Async(
            mstate,
            0xFF80_0000u32,
            rows as usize,
            inner::current_stream(),
        );
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("[fused-lce-gemm] mstate memset failed: {r:?}");
            return free_all(1);
        }
    }
    inner::memset_d8(sstate as *mut c_void, (rows * 4) as usize);
    inner::memset_d8(tlstate as *mut c_void, (rows * 4) as usize);

    let mut chunk_start = 0u64;
    while chunk_start < v {
        let cols = chunk.min(v - chunk_start);
        // logits_chunk [rows, cols] = x @ W[chunk_start.., :]^T
        let gemm = unsafe {
            crate::cuda::cublas_inner::sgemm_row_major_t(
                x_dev as *const f32,
                (w_dev + chunk_start * h * 4) as *const f32,
                logits_buf as *mut f32,
                rows,
                cols,
                h,
                false,
                true, // W chunk is stored [cols, h]; we want its transpose
                1.0,
                0.0,
            )
        };
        if gemm.is_err() {
            eprintln!("[fused-lce-gemm] forward chunk gemm failed at {chunk_start}");
            return free_all(2);
        }
        let mut a_logits = logits_buf;
        let mut a_bias = bias_dev;
        let mut a_tgt = targets_dev;
        let mut a_m = mstate;
        let mut a_s = sstate;
        let mut a_tl = tlstate;
        let mut a_rows = rows;
        let mut a_cols = cols;
        let mut a_cs = chunk_start;
        let mut a_hb: u32 = u32::from(has_bias);
        let args: [*mut c_void; 10] = [
            &mut a_logits as *mut _ as *mut c_void,
            &mut a_bias as *mut _ as *mut c_void,
            &mut a_tgt as *mut _ as *mut c_void,
            &mut a_m as *mut _ as *mut c_void,
            &mut a_s as *mut _ as *mut c_void,
            &mut a_tl as *mut _ as *mut c_void,
            &mut a_rows as *mut _ as *mut c_void,
            &mut a_cols as *mut _ as *mut c_void,
            &mut a_cs as *mut _ as *mut c_void,
            &mut a_hb as *mut _ as *mut c_void,
        ];
        let rc = inner::kernel_launch(
            super::fused_kernels::LCE_CHUNK_STATS_F32_PTX.as_ptr(),
            b"nsl_lce_chunk_stats_f32\0".as_ptr(),
            [rows as i64, 1, 1],
            [256, 1, 1],
            &args,
            0,
        ) as u32;
        if rc != 0 {
            eprintln!("[fused-lce-gemm] chunk stats launch failed: {rc}");
            return free_all(3);
        }
        chunk_start += cols;
    }

    // Finalize: loss/lse from the folded state.
    let mut a_m = mstate;
    let mut a_s = sstate;
    let mut a_tl = tlstate;
    let mut a_tgt = targets_dev;
    let mut a_loss = loss_out_dev;
    let mut a_lse = lse_out_dev;
    let mut a_rows = rows;
    let args: [*mut c_void; 7] = [
        &mut a_m as *mut _ as *mut c_void,
        &mut a_s as *mut _ as *mut c_void,
        &mut a_tl as *mut _ as *mut c_void,
        &mut a_tgt as *mut _ as *mut c_void,
        &mut a_loss as *mut _ as *mut c_void,
        &mut a_lse as *mut _ as *mut c_void,
        &mut a_rows as *mut _ as *mut c_void,
    ];
    let grid = ((rows as i64) + 255) / 256;
    let rc = inner::kernel_launch(
        super::fused_kernels::LCE_FINALIZE_F32_PTX.as_ptr(),
        b"nsl_lce_finalize_f32\0".as_ptr(),
        [grid.max(1), 1, 1],
        [256, 1, 1],
        &args,
        0,
    ) as u32;
    if rc != 0 {
        eprintln!("[fused-lce-gemm] finalize launch failed: {rc}");
        return free_all(4);
    }
    free_all(0)
}

/// GEMM-chunked backward. `dx [rows, h]`, `dw [v, h]`, and (when `has_bias`)
/// `dbias [v]` must be ZEROED by the caller — every chunk accumulates
/// (`beta = 1` GEMMs; dbias via `red.global.add`). `scale` is
/// `grad_output / num_valid`, folded on the host. Recomputes each logits
/// chunk (one extra GEMM pass) instead of caching them — trading ~1.4 ms for
/// never holding more than one [rows, chunk] scratch.
///
/// PRECONDITION (unenforceable here): as `gemm_forward` — every pointer is a
/// bare device address and every buffer is sized at 4 bytes per element, so
/// f32 storage is required and cannot be checked from this frame.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_backward(
    x_dev: u64,
    w_dev: u64,
    bias_dev: u64,
    targets_dev: u64,
    lse_dev: u64,
    dx_dev: u64,
    dw_dev: u64,
    dbias_dev: u64,
    rows: u64,
    v: u64,
    h: u64,
    scale: f32,
    has_bias: bool,
) -> u32 {
    use crate::cuda::inner;
    let chunk = LCE_GEMM_CHUNK.min(v);
    let logits_buf = inner::alloc_managed((rows * chunk * 4) as usize) as u64;
    let free_all = |rc: u32| -> u32 {
        inner::free_managed(logits_buf as *mut c_void);
        rc
    };

    let mut chunk_start = 0u64;
    while chunk_start < v {
        let cols = chunk.min(v - chunk_start);
        let w_chunk = w_dev + chunk_start * h * 4;
        // Recompute logits chunk.
        let gemm = unsafe {
            crate::cuda::cublas_inner::sgemm_row_major_t(
                x_dev as *const f32,
                w_chunk as *const f32,
                logits_buf as *mut f32,
                rows,
                cols,
                h,
                false,
                true,
                1.0,
                0.0,
            )
        };
        if gemm.is_err() {
            eprintln!("[fused-lce-gemm] backward logits gemm failed at {chunk_start}");
            return free_all(2);
        }
        // dlogits in place (+ dbias scatter).
        let mut a_logits = logits_buf;
        let mut a_bias = bias_dev;
        let mut a_tgt = targets_dev;
        let mut a_lse = lse_dev;
        let mut a_dbias = dbias_dev;
        let mut a_rows = rows;
        let mut a_cols = cols;
        let mut a_cs = chunk_start;
        let mut a_scale = scale;
        let mut a_hb: u32 = u32::from(has_bias);
        let args: [*mut c_void; 10] = [
            &mut a_logits as *mut _ as *mut c_void,
            &mut a_bias as *mut _ as *mut c_void,
            &mut a_tgt as *mut _ as *mut c_void,
            &mut a_lse as *mut _ as *mut c_void,
            &mut a_dbias as *mut _ as *mut c_void,
            &mut a_rows as *mut _ as *mut c_void,
            &mut a_cols as *mut _ as *mut c_void,
            &mut a_cs as *mut _ as *mut c_void,
            &mut a_scale as *mut _ as *mut c_void,
            &mut a_hb as *mut _ as *mut c_void,
        ];
        let total = (rows * cols) as i64;
        let grid = ((total + 255) / 256).max(1);
        let rc = inner::kernel_launch(
            super::fused_kernels::LCE_CHUNK_DLOGITS_F32_PTX.as_ptr(),
            b"nsl_lce_chunk_dlogits_f32\0".as_ptr(),
            [grid, 1, 1],
            [256, 1, 1],
            &args,
            0,
        ) as u32;
        if rc != 0 {
            eprintln!("[fused-lce-gemm] chunk dlogits launch failed: {rc}");
            return free_all(3);
        }
        // dx += dlogits_chunk @ W_chunk       [rows, h]
        let gemm = unsafe {
            crate::cuda::cublas_inner::sgemm_row_major_t(
                logits_buf as *const f32,
                w_chunk as *const f32,
                dx_dev as *mut f32,
                rows,
                h,
                cols,
                false,
                false,
                1.0,
                1.0,
            )
        };
        if gemm.is_err() {
            eprintln!("[fused-lce-gemm] dx gemm failed at {chunk_start}");
            return free_all(4);
        }
        // dW_chunk += dlogits_chunk^T @ x     [cols, h]
        let gemm = unsafe {
            crate::cuda::cublas_inner::sgemm_wgrad_accum(
                logits_buf as *const f32, // X [rows, cols] — contraction over rows
                x_dev as *const f32,      // G [rows, h]
                (dw_dev + chunk_start * h * 4) as *mut f32,
                rows,
                cols,
                h,
                1.0,
                1.0,
            )
        };
        if gemm.is_err() {
            eprintln!("[fused-lce-gemm] dW gemm failed at {chunk_start}");
            return free_all(5);
        }
        chunk_start += cols;
    }
    free_all(0)
}
