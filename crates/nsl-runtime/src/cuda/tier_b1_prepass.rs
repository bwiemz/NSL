//! CSHA Tier B.1 production pre-pass kernels.
//!
//! Tier B.1's projection MMA expects inputs in NON-default layouts that the
//! standard CSHA pipeline doesn't produce:
//!
//!   * `csha_x_ptr` : the kernel reads it as f16 in chunks-major
//!                    `[d_model/chunk, seq, chunk]` layout, but the standard
//!                    pipeline writes f32 in row-major `[seq, d_model]`.
//!   * `csha_w*_ptr`: the kernel reads it as f16 in col-major-within-chunk
//!                    `[d_model/chunk, hd, chunk]` layout, but the standard
//!                    pipeline writes f32 in row-major `[d_model, hd]`.
//!
//! When the dispatched kernel is a Tier B.1 variant (detected via the
//! `_tier_b1` suffix on its name), `nsl_flash_attention_csha` orchestrates:
//!   1. RMSNorm + narrow + chunkify on `x` (per-step; dynamic input).
//!   2. Narrow + col-major-chunkify on `Wq/Wk/Wv` (one-time; weights are
//!      static — the orchestrator caches results keyed on the weight pointer).
//!   3. Launches the main Tier B.1 kernel with `csha.skip_rmsnorm_prologue=true`
//!      semantics, passing the chunkified scratch buffers.
//!   4. Frees the per-call scratch (x_chunkified) on completion.
//!
//! See `project_csha_tier_b1_numerical_correctness` memory for the full
//! design rationale (PR #180 closed the codegen bugs; this file closes the
//! caller-side pre-pass loop).

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// X pre-pass: RMSNorm + narrow + chunkify
//
// Grid:    (seq, 1, 1)     — one CTA per row
// Block:   (256, 1, 1)
// Inputs:
//   x_in      : f32 [seq, d_model]               (row-major)
//   gamma     : f32 [d_model]                    (RMSNorm scale)
//   x_out     : f16 [d_model/chunk, seq, chunk]  (chunks-major)
//   seq, d_model, chunk : u64
//   log2_chunk          : u32 (chunk is power of 2; from chunk_config::select)
//   eps                 : f32
//
// Each CTA processes one row of x. Pass 1: compute partial sum-of-squares
// across the row (reduction in sdata). Pass 2: each thread normalizes,
// applies gamma, narrows to f16, and writes to the chunkified output at
// `(chunk_idx, row, c) → byte (chunk_idx * seq * chunk + row * chunk + c) * 2`
// where chunk_idx = d >> log2_chunk and c = d & (chunk - 1).
// ---------------------------------------------------------------------------
pub const CSHA_TIER_B1_PREPASS_X_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry csha_tier_b1_prepass_x(\n\
    .param .u64 x_in,\n\
    .param .u64 gamma,\n\
    .param .u64 x_out,\n\
    .param .u64 seq,\n\
    .param .u64 d_model,\n\
    .param .u64 chunk,\n\
    .param .u32 log2_chunk,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<16>;\n\
    .reg .f32 %f<12>;\n\
    .reg .b16 %h1;\n\
    .reg .pred %p<5>;\n\
    .shared .f32 sdata[256];\n\
\n\
    ld.param.u64 %rd1, [x_in];\n\
    ld.param.u64 %rd2, [gamma];\n\
    ld.param.u64 %rd3, [x_out];\n\
    ld.param.u64 %rd4, [seq];\n\
    ld.param.u64 %rd5, [d_model];\n\
    ld.param.u64 %rd6, [chunk];\n\
    ld.param.u32 %r10, [log2_chunk];\n\
    ld.param.f32 %f8, [eps];\n\
\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra X_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
\n\
    // row_base_in = x_in + (row * d_model) * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd5;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
\n\
    // --- Pass 1: partial sum-of-squares ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd11, %rd8;\n\
X_SQ_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra X_SQ_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    add.u64 %rd13, %rd10, %rd12;\n\
    ld.global.f32 %f2, [%rd13];\n\
    fma.rn.f32 %f1, %f2, %f2, %f1;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra X_SQ_LOOP;\n\
X_SQ_DONE:\n\
    // Store partial in sdata[tid]\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r5, sdata;\n\
    add.u32 %r5, %r5, %r3;\n\
    st.shared.f32 [%r5], %f1;\n\
    bar.sync 0;\n\
\n\
    // Thread 0 reduces and writes rms_inv to sdata[0]\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra X_SKIP_REDUCE;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r6, %ntid.x;\n\
X_RLOOP:\n\
    setp.ge.u32 %p2, %r4, %r6;\n\
    @%p2 bra X_RDONE;\n\
    mul.lo.u32 %r7, %r4, 4;\n\
    mov.u32 %r5, sdata;\n\
    add.u32 %r5, %r5, %r7;\n\
    ld.shared.f32 %f3, [%r5];\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra X_RLOOP;\n\
X_RDONE:\n\
    cvt.rn.f32.u64 %f4, %rd5;\n\
    div.approx.f32 %f1, %f1, %f4;\n\
    add.f32 %f1, %f1, %f8;\n\
    rsqrt.approx.f32 %f1, %f1;\n\
    st.shared.f32 [sdata], %f1;\n\
X_SKIP_REDUCE:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f6, [sdata];\n\
\n\
    // --- Pass 2: normalize, scale by gamma, narrow, chunkify ---\n\
    // chunk_band_size_bytes = seq * chunk * 2\n\
    mul.lo.u64 %rd14, %rd4, %rd6;\n\
    shl.b64 %rd14, %rd14, 1;\n\
    // row_off_bytes = row * chunk * 2\n\
    mul.lo.u64 %rd15, %rd7, %rd6;\n\
    shl.b64 %rd15, %rd15, 1;\n\
    // chunk_minus_one (mask for c = d & (chunk-1))\n\
    sub.u64 %rd16, %rd6, 1;\n\
\n\
    mov.u64 %rd11, %rd8;  // d = tid\n\
X_NORM_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra X_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    // Load x[row, d]\n\
    add.u64 %rd17, %rd10, %rd12;\n\
    ld.global.f32 %f2, [%rd17];\n\
    // Load gamma[d]\n\
    add.u64 %rd18, %rd2, %rd12;\n\
    ld.global.f32 %f7, [%rd18];\n\
    // f2 = x * rms_inv * gamma\n\
    mul.f32 %f2, %f2, %f6;\n\
    mul.f32 %f2, %f2, %f7;\n\
    // Narrow to f16\n\
    cvt.rn.f16.f32 %h1, %f2;\n\
    // chunk_idx = d >> log2_chunk\n\
    shr.u64 %rd19, %rd11, %r10;\n\
    // c = d & (chunk - 1); c_bytes = c << 1\n\
    and.b64 %rd20, %rd11, %rd16;\n\
    shl.b64 %rd20, %rd20, 1;\n\
    // out_byte = chunk_idx * chunk_band_size + row_off + c_bytes\n\
    mul.lo.u64 %rd21, %rd19, %rd14;\n\
    add.u64 %rd21, %rd21, %rd15;\n\
    add.u64 %rd21, %rd21, %rd20;\n\
    add.u64 %rd21, %rd3, %rd21;\n\
    st.global.b16 [%rd21], %h1;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra X_NORM_LOOP;\n\
X_DONE:\n\
    ret;\n\
}\0";

// ---------------------------------------------------------------------------
// W pre-pass: narrow + col-major-chunkify
//
// Grid:    (ceil(d_model * hd / 256), 1, 1)
// Block:   (256, 1, 1)
// Inputs:
//   w_in       : f32 [d_model, hd]                (row-major)
//   w_out      : f16 [d_model/chunk, hd, chunk]   (col-major within each chunk band)
//   d_model, hd, chunk : u64
//   log2_hd, log2_chunk : u32 (both powers of 2)
//
// Output layout per chunk band: `[hd, chunk]` col-major →
// byte `(chunk_idx * hd * chunk + n * chunk + k_in_chunk) * 2`
// where (d_row, n) is the input position and
// d_row = chunk_idx * chunk + k_in_chunk.
//
// Indexed by input: gid → (d_row, n) where d_row = gid >> log2_hd,
// n = gid & (hd - 1). Reads coalesce (adjacent gid → adjacent n in the
// same input row). Writes are NOT coalesced (adjacent gid → adjacent n
// in output, which are `chunk * 2` bytes apart in col-major-within-chunk
// storage) — acceptable because this is a one-time conversion at model
// load, not per-step.
// ---------------------------------------------------------------------------
pub const CSHA_TIER_B1_PREPASS_W_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry csha_tier_b1_prepass_w(\n\
    .param .u64 w_in,\n\
    .param .u64 w_out,\n\
    .param .u64 d_model,\n\
    .param .u64 hd,\n\
    .param .u64 chunk,\n\
    .param .u32 log2_hd,\n\
    .param .u32 log2_chunk\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f1;\n\
    .reg .b16 %h1;\n\
    .reg .pred %p<3>;\n\
\n\
    ld.param.u64 %rd1, [w_in];\n\
    ld.param.u64 %rd2, [w_out];\n\
    ld.param.u64 %rd3, [d_model];\n\
    ld.param.u64 %rd4, [hd];\n\
    ld.param.u64 %rd5, [chunk];\n\
    ld.param.u32 %r1, [log2_hd];\n\
    ld.param.u32 %r2, [log2_chunk];\n\
\n\
    // gid = ctaid.x * 256 + tid.x\n\
    mov.u32 %r3, %ctaid.x;\n\
    mov.u32 %r4, %ntid.x;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r5, %tid.x;\n\
    add.u32 %r3, %r3, %r5;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    // total = d_model * hd\n\
    mul.lo.u64 %rd7, %rd3, %rd4;\n\
    setp.ge.u64 %p1, %rd6, %rd7;\n\
    @%p1 bra W_DONE;\n\
\n\
    // d_row = gid >> log2_hd; n = gid & (hd - 1)\n\
    shr.u64 %rd8, %rd6, %r1;\n\
    sub.u64 %rd9, %rd4, 1;\n\
    and.b64 %rd10, %rd6, %rd9;\n\
\n\
    // Load w_in[d_row, n] = w_in + gid * 4\n\
    shl.b64 %rd11, %rd6, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    cvt.rn.f16.f32 %h1, %f1;\n\
\n\
    // chunk_idx = d_row >> log2_chunk; k_in_chunk = d_row & (chunk - 1)\n\
    shr.u64 %rd12, %rd8, %r2;\n\
    sub.u64 %rd13, %rd5, 1;\n\
    and.b64 %rd14, %rd8, %rd13;\n\
\n\
    // out_byte = (chunk_idx * hd * chunk + n * chunk + k_in_chunk) * 2\n\
    // chunk_band_size = hd * chunk\n\
    mul.lo.u64 %rd15, %rd4, %rd5;\n\
    mul.lo.u64 %rd16, %rd12, %rd15;          // chunk_idx * hd * chunk\n\
    mul.lo.u64 %rd17, %rd10, %rd5;            // n * chunk\n\
    add.u64 %rd16, %rd16, %rd17;\n\
    add.u64 %rd16, %rd16, %rd14;              // + k_in_chunk\n\
    shl.b64 %rd16, %rd16, 1;                  // * 2 bytes\n\
    add.u64 %rd16, %rd2, %rd16;\n\
    st.global.b16 [%rd16], %h1;\n\
W_DONE:\n\
    ret;\n\
}\0";

// ---------------------------------------------------------------------------
// Launch wrappers + W cache
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUresult;

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

/// Process-global cache of chunkified weight tiles. Weights are static
/// across inference calls in a typical workflow (loaded once at model
/// init), so the per-call cost of running `launch_w_prepass` three
/// times (Wq/Wk/Wv) every attention layer dominates the actual
/// attention work for short sequences. The cache lets each weight
/// pointer be converted exactly once.
///
/// **Key invariant:** the input pointer is the (raw GPU device-pointer)
/// base address of the f32 weight tile. The kernel must not be called
/// with two different chunkified layouts for the same `(ctx, in_ptr,
/// dm, hd, chunk)` tuple — but in practice all four are derived from
/// the model config so there's no ambiguity.
///
/// **Context safety:** the key includes the current `CUcontext` (as
/// `u64`) so cache hits only return device pointers allocated in the
/// same CUDA context. The codebase today uses a single primary
/// context retained at startup, but if a caller ever drives the FFI
/// from a thread with a different active context, the cache will not
/// hand out a pointer that belongs to a different address space.
/// If `cuCtxGetCurrent` fails or returns null, lookups return None
/// (the caller will then allocate a fresh per-call scratch — slower
/// but correct).
///
/// **Lifetime:** entries hold owned `cuMemAlloc`'d device buffers that
/// live until process exit. This intentionally leaks: for inference
/// workloads weights are loaded once and used until the process dies,
/// so eviction would only churn the cache.
#[cfg(feature = "cuda")]
#[derive(Eq, PartialEq, Hash, Clone, Copy)]
struct WCacheKey {
    ctx: u64,
    in_ptr: u64,
    d_model: u32,
    hd: u32,
    chunk: u32,
}

#[cfg(feature = "cuda")]
fn w_cache() -> &'static Mutex<HashMap<WCacheKey, u64>> {
    static CACHE: std::sync::OnceLock<Mutex<HashMap<WCacheKey, u64>>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Read the currently-active CUDA context for the calling thread.
/// Returns `None` if no context is current, or if the driver call
/// fails. Used as part of the W cache key so the cache cannot return a
/// device pointer allocated in a different context.
#[cfg(feature = "cuda")]
fn current_cuda_context() -> Option<u64> {
    let mut ctx: cudarc::driver::sys::CUcontext = std::ptr::null_mut();
    let rc = unsafe { cudarc::driver::sys::cuCtxGetCurrent(&mut ctx) };
    if rc as u32 != 0 || ctx.is_null() {
        None
    } else {
        Some(ctx as u64)
    }
}

/// Look up `(ctx, in_ptr, dm, hd, chunk)` in the global W cache.
/// Returns the chunkified GPU pointer on hit. On miss, allocates a GPU
/// scratch buffer, launches `launch_w_prepass`, inserts into the
/// cache (when a context is current), and returns the new pointer.
/// Returns `None` only on allocation or kernel-launch failure.
///
/// When `cuCtxGetCurrent` reports no current context, lookup and
/// insertion are skipped (so the cache cannot return or remember a
/// pointer that belongs to a different address space), but the
/// allocation and pre-pass still run so the caller gets a valid
/// chunkified pointer. The result simply isn't cached — a cost only
/// paid on the (degraded) no-context path.
#[cfg(feature = "cuda")]
pub(crate) fn w_chunkified_cached(in_ptr: u64, d_model: u64, hd: u64, chunk: u64) -> Option<u64> {
    let ctx_opt = current_cuda_context();
    if let Some(ctx) = ctx_opt {
        let key = WCacheKey {
            ctx,
            in_ptr,
            d_model: d_model as u32,
            hd: hd as u32,
            chunk: chunk as u32,
        };
        if let Some(&p) = w_cache().lock().unwrap().get(&key) {
            return Some(p);
        }
    }
    let n_chunks = d_model / chunk;
    let bytes = (n_chunks * hd * chunk * 2) as usize;
    let scratch = super::inner::alloc_device(bytes);
    if scratch.is_null() {
        return None;
    }
    let rc = launch_w_prepass(in_ptr, scratch as u64, d_model, hd, chunk);
    if rc as u32 != 0 {
        unsafe {
            let _ = cudarc::driver::sys::cuMemFree_v2(scratch as cudarc::driver::sys::CUdeviceptr);
        }
        return None;
    }
    if let Some(ctx) = ctx_opt {
        let key = WCacheKey {
            ctx,
            in_ptr,
            d_model: d_model as u32,
            hd: hd as u32,
            chunk: chunk as u32,
        };
        w_cache().lock().unwrap().insert(key, scratch as u64);
    }
    Some(scratch as u64)
}

/// Launch the X pre-pass: read raw `x_in` (f32 `[seq, d_model]`), apply
/// RMSNorm with `gamma_in`, narrow to f16, and write to `x_out` in
/// `[d_model/chunk, seq, chunk]` chunks-major layout. `chunk` must be a
/// power of 2 (Tier B.1's `chunk_config::select` only emits 32/64/128).
///
/// All pointers are GPU-resident.
#[cfg(feature = "cuda")]
pub(crate) fn launch_x_prepass(
    x_in_ptr: u64,
    gamma_ptr: u64,
    x_out_ptr: u64,
    seq: u64,
    d_model: u64,
    chunk: u64,
    eps: f32,
) -> CUresult {
    debug_assert!(
        chunk.is_power_of_two(),
        "chunk must be a power of 2; got {}",
        chunk
    );
    let log2_chunk: u32 = chunk.trailing_zeros();
    let mut x_in = x_in_ptr;
    let mut gamma = gamma_ptr;
    let mut x_out = x_out_ptr;
    let mut seq_v = seq;
    let mut dm_v = d_model;
    let mut chunk_v = chunk;
    let mut log2c = log2_chunk;
    let mut eps_v = eps;
    let args: [*mut std::ffi::c_void; 8] = [
        &mut x_in as *mut _ as *mut std::ffi::c_void,
        &mut gamma as *mut _ as *mut std::ffi::c_void,
        &mut x_out as *mut _ as *mut std::ffi::c_void,
        &mut seq_v as *mut _ as *mut std::ffi::c_void,
        &mut dm_v as *mut _ as *mut std::ffi::c_void,
        &mut chunk_v as *mut _ as *mut std::ffi::c_void,
        &mut log2c as *mut _ as *mut std::ffi::c_void,
        &mut eps_v as *mut _ as *mut std::ffi::c_void,
    ];
    super::inner::kernel_launch(
        CSHA_TIER_B1_PREPASS_X_PTX.as_ptr(),
        b"csha_tier_b1_prepass_x\0".as_ptr(),
        [seq as i64, 1, 1],
        [256, 1, 1],
        &args,
        0,
    )
}

/// Launch the W pre-pass: read raw `w_in` (f32 `[d_model, hd]`), narrow
/// to f16, and write to `w_out` in `[d_model/chunk, hd, chunk]`
/// col-major-within-chunk layout. Both `chunk` and `hd` must be powers
/// of 2.
///
/// All pointers are GPU-resident. Use this on Wq/Wk/Wv. Intended as a
/// one-time conversion at model load — the orchestrator caches the
/// outputs keyed on the source weight pointer.
#[cfg(feature = "cuda")]
pub(crate) fn launch_w_prepass(
    w_in_ptr: u64,
    w_out_ptr: u64,
    d_model: u64,
    hd: u64,
    chunk: u64,
) -> CUresult {
    debug_assert!(
        chunk.is_power_of_two(),
        "chunk must be a power of 2; got {}",
        chunk
    );
    debug_assert!(hd.is_power_of_two(), "hd must be a power of 2; got {}", hd);
    let log2_chunk: u32 = chunk.trailing_zeros();
    let log2_hd: u32 = hd.trailing_zeros();
    let total: i64 = (d_model * hd) as i64;
    let grid_x = (total + 255) / 256;
    let mut w_in = w_in_ptr;
    let mut w_out = w_out_ptr;
    let mut dm_v = d_model;
    let mut hd_v = hd;
    let mut chunk_v = chunk;
    let mut log2h = log2_hd;
    let mut log2c = log2_chunk;
    let args: [*mut std::ffi::c_void; 7] = [
        &mut w_in as *mut _ as *mut std::ffi::c_void,
        &mut w_out as *mut _ as *mut std::ffi::c_void,
        &mut dm_v as *mut _ as *mut std::ffi::c_void,
        &mut hd_v as *mut _ as *mut std::ffi::c_void,
        &mut chunk_v as *mut _ as *mut std::ffi::c_void,
        &mut log2h as *mut _ as *mut std::ffi::c_void,
        &mut log2c as *mut _ as *mut std::ffi::c_void,
    ];
    super::inner::kernel_launch(
        CSHA_TIER_B1_PREPASS_W_PTX.as_ptr(),
        b"csha_tier_b1_prepass_w\0".as_ptr(),
        [grid_x, 1, 1],
        [256, 1, 1],
        &args,
        0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_ptx_is_null_terminated_and_has_entry() {
        assert!(CSHA_TIER_B1_PREPASS_X_PTX.ends_with("\0"));
        assert!(CSHA_TIER_B1_PREPASS_X_PTX.contains(".visible .entry csha_tier_b1_prepass_x"));
    }

    #[test]
    fn w_ptx_is_null_terminated_and_has_entry() {
        assert!(CSHA_TIER_B1_PREPASS_W_PTX.ends_with("\0"));
        assert!(CSHA_TIER_B1_PREPASS_W_PTX.contains(".visible .entry csha_tier_b1_prepass_w"));
    }

    #[test]
    fn x_ptx_handles_chunk_stride_with_log2_shift() {
        // Verify the kernel uses runtime shift for chunk_idx / mask for c,
        // not div / rem (which are expensive on GPU and would mean the
        // caller has to pre-compute chunk_idx/c).
        assert!(CSHA_TIER_B1_PREPASS_X_PTX.contains("shr.u64 %rd19, %rd11, %r10;"));
        assert!(CSHA_TIER_B1_PREPASS_X_PTX.contains("and.b64 %rd20, %rd11, %rd16;"));
    }

    #[test]
    fn x_ptx_emits_f32_to_f16_narrowing() {
        // Output is f16; conversion must happen before the store.
        assert!(CSHA_TIER_B1_PREPASS_X_PTX.contains("cvt.rn.f16.f32"));
        assert!(CSHA_TIER_B1_PREPASS_X_PTX.contains("st.global.b16"));
    }

    #[test]
    fn w_ptx_emits_f32_to_f16_narrowing() {
        assert!(CSHA_TIER_B1_PREPASS_W_PTX.contains("cvt.rn.f16.f32"));
        assert!(CSHA_TIER_B1_PREPASS_W_PTX.contains("st.global.b16"));
    }
}
