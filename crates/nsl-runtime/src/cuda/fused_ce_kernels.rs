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
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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
// state folded chunk by chunk (`lce_chunk_stats_f32_ptx`). The GEMMs route
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

/// The LM head's GEMM operands under the BF16 math mode, each rounded to bf16
/// ONCE per call instead of once per chunk GEMM: `x` (read by every chunk's
/// logits GEMM and again by every chunk's dW GEMM), the head's weight (the
/// cast cache's image of the whole tied embedding, when it is registered),
/// and in the backward each chunk's dlogits once for both its dx and its dW
/// GEMM. The same RNE-rounded values the per-GEMM casts produced, so the
/// same arithmetic; at Coder-1B that is 36 casts of x and 24 of dlogits per
/// micro-batch fewer.
///
/// Each of the head's three GEMMs takes this route where the mode's own
/// intensity gate admits it for every chunk width, and the per-GEMM path
/// otherwise (so bf16 runs exactly where it ran before): the logits GEMM on
/// its own gate, in the forward and in the backward's recompute alike, so the
/// backward recomputes exactly the forward's logits; dx and dW each on
/// theirs (at hidden 512, Coder-50M, the logits GEMM passes and dx and dW do
/// not). `None` (the per-GEMM path for all three) unless the mode is BF16, no
/// CUDA graph is being captured, and at least one GEMM takes the route.
struct Bf16Head {
    /// `x` rounded to bf16 (0 unless the logits or the dW GEMM takes the route).
    x16: u64,
    /// The cast cache's image of the whole head weight (None: per-chunk casts).
    w16: Option<u64>,
    /// One chunk's dlogits rounded to bf16 (0 unless dx or dW takes the route).
    dl16: u64,
    logits: bool,
    dx: bool,
    dw: bool,
}

impl Bf16Head {
    fn new(x_dev: u64, w_dev: u64, rows: u64, v: u64, h: u64, chunk: u64, backward: bool) -> Option<Bf16Head> {
        use crate::cuda::cublas_inner::{bf16_gate, resolved_math_mode, CublasMathMode};
        if resolved_math_mode() != CublasMathMode::Bf16 || crate::cuda::graph_capture::enabled() {
            return None;
        }
        let tail = if v.is_multiple_of(chunk) { chunk } else { v % chunk };
        let (r, hh) = (rows as i64, h as i64);
        let widths = [chunk as i64, tail as i64];
        let logits = widths.iter().all(|&c| bf16_gate(r, c, hh, (r * hh) as usize, (c * hh) as usize));
        let dx = backward && widths.iter().all(|&c| bf16_gate(r, hh, c, (r * c) as usize, (c * hh) as usize));
        let dw = backward && widths.iter().all(|&c| bf16_gate(c, hh, r, (r * c) as usize, (r * hh) as usize));
        if !(logits || dx || dw) {
            return None;
        }
        use crate::cuda::inner;
        let x16 = if logits || dw {
            let b = inner::alloc_managed((rows * h * 2) as usize) as u64;
            crate::cuda::gpu_cast_raw_f32_to_bf16(x_dev, b, (rows * h) as usize);
            b
        } else {
            0
        };
        let w16 = if logits || dx {
            crate::cuda::bf16_cast_cache::acquire(w_dev, (v * h) as usize).map(|(buf, needs_cast)| {
                if needs_cast {
                    crate::cuda::gpu_cast_raw_f32_to_bf16(w_dev, buf, (v * h) as usize);
                }
                buf
            })
        } else {
            None
        };
        let dl16 = if dx || dw { inner::alloc_managed((rows * chunk * 2) as usize) as u64 } else { 0 };
        Some(Bf16Head { x16, w16, dl16, logits, dx, dw })
    }

    /// The bf16 rows `[chunk_start, chunk_start + cols)` of the head weight
    /// when the logits or the dx GEMM reads them: an offset into the cached
    /// image, or a scratch cast (freed with [`free_w_chunk`]).
    fn w_chunk(&self, w_dev: u64, chunk_start: u64, cols: u64, h: u64) -> Option<(u64, Option<u64>)> {
        if !(self.logits || self.dx) {
            return None;
        }
        Some(match self.w16 {
            Some(buf) => (buf + chunk_start * h * 2, None),
            None => {
                let s = crate::cuda::inner::alloc_managed((cols * h * 2) as usize) as u64;
                crate::cuda::gpu_cast_raw_f32_to_bf16(w_dev + chunk_start * h * 4, s, (cols * h) as usize);
                (s, Some(s))
            }
        })
    }

    /// `logits[rows, cols] = x @ W_chunk^T` (f32 out).
    fn logits_gemm(&self, w16c: u64, logits: u64, rows: u64, cols: u64, h: u64) {
        // SAFETY: buffers sized by the caller's shapes (row-major, as
        // `storage_gemm::matmul` maps them onto cuBLAS's column-major call).
        unsafe {
            crate::cuda::storage_gemm::gemm16(true, false, cols, rows, h, 1.0, w16c, h, self.x16, h, 0.0, logits, cols, false)
        }
    }
}

/// Free a [`Bf16Head::w_chunk`] scratch cast, if it made one.
fn free_w_chunk(w: Option<(u64, Option<u64>)>) {
    if let Some((_, Some(s))) = w {
        crate::cuda::inner::free_managed(s as *mut c_void);
    }
}

impl Drop for Bf16Head {
    fn drop(&mut self) {
        for b in [self.x16, self.dl16] {
            if b != 0 {
                crate::cuda::inner::free_managed(b as *mut c_void);
            }
        }
    }
}

/// Test hook: every logits chunk the head computes, in the forward and in the
/// backward's recompute, as f32 bits, with whether its GEMM took the bf16 head.
#[cfg(feature = "test-hooks")]
pub(crate) mod logits_probe {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    /// (backward, via the bf16 head, the chunk's logits bits)
    pub(crate) static CHUNKS: Mutex<Vec<(bool, bool, Vec<u32>)>> = Mutex::new(Vec::new());
    pub(crate) static ARMED: AtomicBool = AtomicBool::new(false);

    pub(crate) fn record(backward: bool, via_head: bool, buf: u64, n: usize) {
        if !ARMED.load(Ordering::Relaxed) {
            return;
        }
        let mut host = vec![0u32; n];
        crate::cuda::inner::memcpy_dtoh(host.as_mut_ptr() as *mut std::ffi::c_void, buf as *const std::ffi::c_void, n * 4);
        CHUNKS.lock().expect("probe lock").push((backward, via_head, host));
    }
}



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
            crate::nsl_log!(WARN, "fused-lce-gemm", "[fused-lce-gemm] mstate memset failed: {r:?}");
            return free_all(1);
        }
    }
    inner::memset_d8(sstate as *mut c_void, (rows * 4) as usize);
    inner::memset_d8(tlstate as *mut c_void, (rows * 4) as usize);

    let head = Bf16Head::new(x_dev, w_dev, rows, v, h, chunk, false);
    let mut chunk_start = 0u64;
    while chunk_start < v {
        let cols = chunk.min(v - chunk_start);
        // logits_chunk [rows, cols] = x @ W[chunk_start.., :]^T
        let via_head = head.as_ref().is_some_and(|hd| hd.logits);
        let gemm = if let Some(hd) = head.as_ref().filter(|hd| hd.logits) {
            let w = hd.w_chunk(w_dev, chunk_start, cols, h);
            let (w16c, _) = w.expect("the logits GEMM reads the bf16 weight");
            hd.logits_gemm(w16c, logits_buf, rows, cols, h);
            free_w_chunk(w);
            Ok(())
        } else {
            unsafe {
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
            }
        };
        if gemm.is_err() {
            crate::nsl_log!(WARN, "fused-lce-gemm", "[fused-lce-gemm] forward chunk gemm failed at {chunk_start}");
            return free_all(2);
        }
        #[cfg(feature = "test-hooks")]
        logits_probe::record(false, via_head, logits_buf, (rows * cols) as usize);
        #[cfg(not(feature = "test-hooks"))]
        let _ = via_head;
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
            super::fused_kernels::lce_chunk_stats_f32_ptx().as_ptr(),
            c"nsl_lce_chunk_stats_f32".as_ptr().cast(),
            [rows as i64, 1, 1],
            [256, 1, 1],
            &args,
            0,
        ) as u32;
        if rc != 0 {
            crate::nsl_log!(WARN, "fused-lce-gemm", "[fused-lce-gemm] chunk stats launch failed: {rc}");
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
        super::fused_kernels::lce_finalize_f32_ptx().as_ptr(),
        c"nsl_lce_finalize_f32".as_ptr().cast(),
        [grid.max(1), 1, 1],
        [256, 1, 1],
        &args,
        0,
    ) as u32;
    if rc != 0 {
        crate::nsl_log!(WARN, "fused-lce-gemm", "[fused-lce-gemm] finalize launch failed: {rc}");
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
/// `dw_alpha` multiplies the dW GEMMs' product: `dw += dw_alpha · dlogitsᵀ
/// x`. 1 for a zeroed `dw`; a gradient-accumulation window's `1 / N` when
/// `dw` is the weight's accumulation buffer (a tied embedding's gradient
/// accumulated in place, no zeroed scratch and no separate add).
///
/// `dx_dev == 0` skips the dx GEMMs (a caller whose x gradient nothing
/// reads); every existing caller passes a buffer.
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
    dw_alpha: f32,
) -> u32 {
    use crate::cuda::inner;
    let chunk = LCE_GEMM_CHUNK.min(v);
    let logits_buf = inner::alloc_managed((rows * chunk * 4) as usize) as u64;
    let free_all = |rc: u32| -> u32 {
        inner::free_managed(logits_buf as *mut c_void);
        rc
    };

    let head = Bf16Head::new(x_dev, w_dev, rows, v, h, chunk, true);
    let mut chunk_start = 0u64;
    while chunk_start < v {
        let cols = chunk.min(v - chunk_start);
        let w_chunk = w_dev + chunk_start * h * 4;
        // The bf16 weight rows, when the logits or the dx GEMM reads them;
        // freed on every path out of this chunk.
        let head_w = head.as_ref().and_then(|hd| hd.w_chunk(w_dev, chunk_start, cols, h));
        let fail = |what: &str, rc: u32| -> u32 {
            crate::nsl_log!(WARN, "fused-lce-gemm", "[fused-lce-gemm] {what} failed at {chunk_start}");
            free_w_chunk(head_w);
            free_all(rc)
        };
        // Recompute the logits chunk on the forward's route.
        let via_head = head.as_ref().is_some_and(|hd| hd.logits);
        let gemm = match (&head, head_w) {
            (Some(hd), Some((w16c, _))) if hd.logits => {
                hd.logits_gemm(w16c, logits_buf, rows, cols, h);
                Ok(())
            }
            _ => unsafe {
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
            },
        };
        if gemm.is_err() {
            return fail("backward logits gemm", 2);
        }
        #[cfg(feature = "test-hooks")]
        logits_probe::record(true, via_head, logits_buf, (rows * cols) as usize);
        #[cfg(not(feature = "test-hooks"))]
        let _ = via_head;
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
            super::fused_kernels::lce_chunk_dlogits_f32_ptx().as_ptr(),
            c"nsl_lce_chunk_dlogits_f32".as_ptr().cast(),
            [grid, 1, 1],
            [256, 1, 1],
            &args,
            0,
        ) as u32;
        if rc != 0 {
            return fail("chunk dlogits launch", 3);
        }
        // dlogits rounded to bf16 once for both GEMMs that take the head
        // (f32 accumulators, beta = 1).
        if let Some(hd) = head.as_ref().filter(|hd| hd.dx || hd.dw) {
            crate::cuda::gpu_cast_raw_f32_to_bf16(logits_buf, hd.dl16, (rows * cols) as usize);
        }
        // dx += dlogits_chunk @ W_chunk       [rows, h]
        let gemm = match (&head, head_w) {
            _ if dx_dev == 0 => Ok(()),
            (Some(hd), Some((w16c, _))) if hd.dx => {
                // SAFETY: buffers sized by the shapes above.
                unsafe {
                    crate::cuda::storage_gemm::gemm16(false, false, h, rows, cols, 1.0, w16c, h, hd.dl16, cols, 1.0, dx_dev, h, false);
                }
                Ok(())
            }
            _ => unsafe {
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
            },
        };
        if gemm.is_err() {
            return fail("dx gemm", 4);
        }
        // dW_chunk += dlogits_chunk^T @ x     [cols, h]
        let gemm = match &head {
            Some(hd) if hd.dw => {
                // SAFETY: buffers sized by the shapes above.
                unsafe {
                    crate::cuda::storage_gemm::gemm16(
                        false,
                        true,
                        h,
                        cols,
                        rows,
                        dw_alpha,
                        hd.x16,
                        h,
                        hd.dl16,
                        cols,
                        1.0,
                        dw_dev + chunk_start * h * 4,
                        h,
                        false,
                    );
                }
                Ok(())
            }
            _ => unsafe {
                crate::cuda::cublas_inner::sgemm_wgrad_accum(
                    logits_buf as *const f32, // X [rows, cols] — contraction over rows
                    x_dev as *const f32,      // G [rows, h]
                    (dw_dev + chunk_start * h * 4) as *mut f32,
                    rows,
                    cols,
                    h,
                    dw_alpha,
                    1.0,
                )
            },
        };
        if gemm.is_err() {
            return fail("dW gemm", 5);
        }
        free_w_chunk(head_w);
        chunk_start += cols;
    }
    free_all(0)
}

// ─── The head's gradient in its forward (row-chunked) ───────────────────────
//
// A training step whose loss is the fused head's own output knows, when the
// forward runs, the upstream gradient the backward will hand the head (the
// loss seed, 1; the compiler proves it and the backward checks it). The
// forward can then finish the head's whole backward while its logits are at
// hand, instead of the backward recomputing every logits chunk:
//
//   for each chunk of R token rows:
//     logits[R, V] = x_rows @ W^T            bf16 out, f32 accumulate (one GEMM)
//     nsl_lce_rows_grad_bf16                 per-row loss; dlogits·scale in place
//     dx_rows      = dlogits @ W             f32 out (one GEMM, K = V)
//     dW          += dw_alpha · dlogits^T x  f32 accumulate, beta 1 (K = R)
//
// Three GEMM passes over the head instead of four, no f32 dlogits and no
// cast of them, and full-width GEMMs instead of 4,096-column ones. The
// logits scratch is `R · V` bf16 (96 MiB), so the whole batch's logits are
// still never held.
//
// It needs the BF16 math mode and the head weight's persistent bf16 image
// (the cast cache, warm from the first optimizer step on). Without either it
// runs the vocabulary-chunked route both halves of the old path use (the
// forward's online softmax for the LSE, then the backward's recompute with
// the gradients), so the step's arithmetic is the old path's.

/// Bytes of the row-chunked route's bf16 logits scratch: 1,024 rows at a
/// 49,152-token vocabulary.
#[cfg(feature = "cuda")]
const LCE_ROW_CHUNK_BYTES: u64 = 96 << 20;

/// Which route [`gemm_forward_grad`] took.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GradRoute {
    /// Row chunks over the whole vocabulary, bf16 logits.
    RowChunked,
    /// The vocabulary-chunked forward, then its backward, both here.
    VocabChunked,
}

/// The row-chunked route's bf16 operands: `x` rounded once (owned), and the
/// head weight's image in the cast cache (borrowed).
#[cfg(feature = "cuda")]
struct RowHead {
    x16: u64,
    w16: u64,
}

#[cfg(feature = "cuda")]
impl RowHead {
    /// `None` (take the vocabulary-chunked route) outside the BF16 math mode,
    /// under graph capture (`gemm16` refuses it), or while the weight has no
    /// bf16 image (a full-`W` cast scratch would be `V · H` bf16, 192 MiB at
    /// Coder-1B, at the forward's activation peak).
    ///
    /// Unlike [`Bf16Head`] it does not ask `bf16_gate` per GEMM: that gate
    /// weighs a GEMM's bf16 saving against casting both operands for it, and
    /// here no GEMM casts an operand (`x` is rounded once per call, `W` is the
    /// cached image, the dlogits are written bf16 by the softmax kernel). So
    /// all three GEMMs run on bf16 operands in the BF16 mode, at every shape,
    /// and the logits are stored bf16 (as PyTorch autocast stores them); the
    /// vocabulary-chunked route keeps them f32.
    fn new(x_dev: u64, w_dev: u64, rows: u64, v: u64, h: u64) -> Option<RowHead> {
        use crate::cuda::cublas_inner::{CublasMathMode, resolved_math_mode};
        if resolved_math_mode() != CublasMathMode::Bf16 || crate::cuda::graph_capture::enabled() {
            return None;
        }
        // The cast follows the probe at once (the cache's protocol: a
        // `needs_cast` hand-off has already marked the image valid).
        let (w16, needs_cast) = crate::cuda::bf16_cast_cache::acquire(w_dev, (v * h) as usize)?;
        if needs_cast {
            crate::cuda::gpu_cast_raw_f32_to_bf16(w_dev, w16, (v * h) as usize);
        }
        let x16 = crate::cuda::inner::alloc_managed((rows * h * 2) as usize) as u64;
        crate::cuda::gpu_cast_raw_f32_to_bf16(x_dev, x16, (rows * h) as usize);
        Some(RowHead { x16, w16 })
    }
}

#[cfg(feature = "cuda")]
impl Drop for RowHead {
    fn drop(&mut self) {
        crate::cuda::inner::free_managed(self.x16 as *mut c_void);
    }
}

/// `NSL_LCE_ROW_CHUNK` (rows, read once): overrides the row chunk the byte
/// budget picks; 0 or unset keeps the budget's.
#[cfg(feature = "cuda")]
fn row_chunk_override() -> u64 {
    use std::sync::OnceLock;
    static ROWS: OnceLock<u64> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("NSL_LCE_ROW_CHUNK")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Rows per chunk: the bf16 logits of a chunk fit [`LCE_ROW_CHUNK_BYTES`]
/// (a multiple of 64 from 128 rows up), or `NSL_LCE_ROW_CHUNK`.
#[cfg(feature = "cuda")]
fn row_chunk(rows: u64, v: u64) -> u64 {
    let pick = match row_chunk_override() {
        0 => {
            let r = (LCE_ROW_CHUNK_BYTES / (v * 2)).max(1);
            if r >= 128 { r / 64 * 64 } else { r }
        }
        n => n,
    };
    pick.clamp(1, rows)
}

/// `logits[r, v] = x16[r0.., :] @ W16^T`, the f32 accumulator rounded to
/// bf16 once. cublasLt writes the bf16 logits itself. Where it does not take
/// the shape (under `--deterministic` without `--bf16-lt`, or a declined
/// plan), `gemm16` would write an f32 scratch of the whole chunk and cast it,
/// three times the logits budget; the product then runs in row blocks whose
/// f32 scratch fits half the budget.
#[cfg(feature = "cuda")]
fn logits_gemm(head: &RowHead, logits: u64, r0: u64, r: u64, v: u64, h: u64) {
    use crate::cuda::lt_matmul;
    let x16r = head.x16 + r0 * h * 2;
    crate::cuda::inner::ensure_context();
    let lt = lt_matmul::enabled() || lt_matmul::timed_lt_applies(v, r, h);
    // SAFETY (both arms): column-major views of row-major buffers sized by
    // the shapes (the logits scratch holds at least `r` rows of `v`), as
    // `gemm_backward`'s bf16 head maps them.
    if lt
        && unsafe {
            lt_matmul::matmul_bf16(
                1,
                0,
                v as i32,
                r as i32,
                h as i32,
                &1.0,
                head.w16 as *const c_void,
                h as i32,
                x16r as *const c_void,
                h as i32,
                &0.0,
                logits as *mut c_void,
                v as i32,
                true,
            )
        }
    {
        return;
    }
    let block = (LCE_ROW_CHUNK_BYTES / 2 / (v * 4)).max(1);
    let mut i = 0;
    while i < r {
        let rb = block.min(r - i);
        unsafe {
            crate::cuda::storage_gemm::gemm16(
                true,
                false,
                v,
                rb,
                h,
                1.0,
                head.w16,
                h,
                x16r + i * h * 2,
                h,
                0.0,
                logits + i * v * 2,
                v,
                true,
            )
        };
        i += rb;
    }
}

/// The fused head's forward with its whole backward (see the section
/// header): per-row losses into `loss_out [rows]`, the x gradient into `dx
/// [rows, h]` (f32, ZEROED by the caller, as `gemm_backward` accumulates
/// into it; 0 = no x gradient), and `dw += dw_alpha · dlogits^T x` into the
/// `[v, h]` f32 accumulator `dw`. `scale` is `grad_output / num_valid`, as
/// the backward folds it. Biasless heads only. Returns the CUresult-style
/// code of the first failure (0 = success) and the route taken.
///
/// PRECONDITION (unenforceable here): as [`gemm_forward`].
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_forward_grad(
    x_dev: u64,
    w_dev: u64,
    targets_dev: u64,
    loss_out_dev: u64,
    dx_dev: u64,
    dw_dev: u64,
    rows: u64,
    v: u64,
    h: u64,
    scale: f32,
    dw_alpha: f32,
    // `dw` is zero in value but not in bytes (`crate::accum_fresh`): the
    // first GEMM that writes it must overwrite.
    dw_fresh: bool,
) -> (u32, GradRoute) {
    // With no rows no GEMM writes dw: zero a fresh one here instead.
    let dw_fresh = if dw_fresh && rows == 0 {
        crate::cuda::inner::memset_d8(dw_dev as *mut c_void, (v * h * 4) as usize);
        false
    } else {
        dw_fresh
    };
    if let Some(head) = RowHead::new(x_dev, w_dev, rows, v, h) {
        let rc = row_chunked(
            &head,
            targets_dev,
            loss_out_dev,
            dx_dev,
            dw_dev,
            rows,
            v,
            h,
            scale,
            dw_alpha,
            dw_fresh,
        );
        return (rc, GradRoute::RowChunked);
    }
    if dw_fresh {
        // The vocabulary-chunked route adds each chunk's rows: zero first.
        crate::cuda::inner::memset_d8(dw_dev as *mut c_void, (v * h * 4) as usize);
    }
    let lse = crate::cuda::inner::alloc_managed((rows * 4) as usize) as u64;
    let mut rc = gemm_forward(
        x_dev,
        w_dev,
        0,
        targets_dev,
        loss_out_dev,
        lse,
        rows,
        v,
        h,
        false,
    );
    if rc == 0 {
        rc = gemm_backward(
            x_dev,
            w_dev,
            0,
            targets_dev,
            lse,
            dx_dev,
            dw_dev,
            0,
            rows,
            v,
            h,
            scale,
            false,
            dw_alpha,
        );
    }
    crate::cuda::inner::free_managed(lse as *mut c_void);
    (rc, GradRoute::VocabChunked)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn row_chunked(
    head: &RowHead,
    targets_dev: u64,
    loss_out_dev: u64,
    dx_dev: u64,
    dw_dev: u64,
    rows: u64,
    v: u64,
    h: u64,
    scale: f32,
    dw_alpha: f32,
    dw_fresh: bool,
) -> u32 {
    use crate::cuda::inner;
    use crate::cuda::storage_gemm::gemm16;
    let r_chunk = row_chunk(rows, v);
    let logits = inner::alloc_managed((r_chunk * v * 2) as usize) as u64;
    let mut r0 = 0u64;
    let mut rc = 0u32;
    while r0 < rows {
        let r = r_chunk.min(rows - r0);
        let x16r = head.x16 + r0 * h * 2;
        logits_gemm(head, logits, r0, r, v, h);
        let mut a_logits = logits;
        let mut a_tgt = targets_dev + r0 * 8;
        let mut a_loss = loss_out_dev + r0 * 4;
        let mut a_rows = r;
        let mut a_cols = v;
        let mut a_scale = scale;
        let args: [*mut c_void; 6] = [
            &mut a_logits as *mut _ as *mut c_void,
            &mut a_tgt as *mut _ as *mut c_void,
            &mut a_loss as *mut _ as *mut c_void,
            &mut a_rows as *mut _ as *mut c_void,
            &mut a_cols as *mut _ as *mut c_void,
            &mut a_scale as *mut _ as *mut c_void,
        ];
        rc = inner::kernel_launch(
            super::fused_kernels::lce_rows_grad_bf16_ptx().as_ptr(),
            c"nsl_lce_rows_grad_bf16".as_ptr().cast(),
            [r as i64, 1, 1],
            [i64::from(nsl_kir::kernels::lce_rows::LCE_ROWS_BLOCK), 1, 1],
            &args,
            0,
        ) as u32;
        if rc != 0 {
            crate::nsl_log!(
                WARN,
                "fused-lce-gemm",
                "[fused-lce-gemm] row softmax launch failed at row {r0}: {rc}"
            );
            break;
        }
        // SAFETY (the two GEMMs): column-major views of row-major buffers
        // sized by the shapes above, as `gemm_backward`'s bf16 head maps them.
        //
        // dx[r0.., :] = dlogits @ W16   [r, h]
        if dx_dev != 0 {
            unsafe {
                gemm16(
                    false,
                    false,
                    h,
                    r,
                    v,
                    1.0,
                    head.w16,
                    h,
                    logits,
                    v,
                    0.0,
                    dx_dev + r0 * h * 4,
                    h,
                    false,
                )
            };
        }
        // dW += dw_alpha · dlogits^T @ x16[r0.., :]   [v, h]; the first chunk
        // overwrites a fresh accumulator (every chunk's GEMM covers all of dW).
        let dw_beta = if dw_fresh && r0 == 0 { 0.0 } else { 1.0 };
        unsafe {
            gemm16(
                false, true, h, v, r, dw_alpha, x16r, h, logits, v, dw_beta, dw_dev, h, false,
            )
        };
        r0 += r;
    }
    inner::free_managed(logits as *mut c_void);
    rc
}
