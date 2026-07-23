//! Muon batched Newton-Schulz engine (perf-campaign items 1/3/4).
//!
//! The per-matrix primitive (`muon.rs`) issues 15 GEMMs + 2 materialized
//! transposes + a single-block Frobenius reduction PER rank-2 matrix PER
//! optimizer step — ~2,520 GEMM launches/step for coder500m. This engine
//! processes all Muon-routed matrices of one shape TOGETHER:
//!
//! - **Shape grouping** (item 1): matrices are grouped by (rows, cols) at
//!   the call site; each group runs the whole momentum -> Newton-Schulz ->
//!   parameter-update pipeline as a fixed launch sequence, with per-matrix
//!   addressing through device pointer tables (no repacking of inputs).
//! - **No physical transposes** (item 3): tall matrices are transposed on
//!   the fly by the pack/update kernels; the Gram product x xᵀ maps to a
//!   `transa=T` strided-batched SGEMM on the workspace directly. Nothing is
//!   ever materialized for orientation.
//! - **Symmetric-operand mapping + fused polynomial** (item 4): every
//!   square NS intermediate (Gram A, A², the polynomial B and B+ns_a·I) is
//!   symmetric, so the row-major/column-major operand swap cancels and each
//!   NS iteration is exactly THREE strided-batched GEMMs plus one
//!   elementwise kernel. ns_a·x + b@x is folded into ONE gemm by adding
//!   ns_a to b's diagonal ((ns_a·I + B)@x). cuBLAS has no strided-batched
//!   SYRK/SYMM, so symmetry buys mapping simplicity rather than half-flop
//!   Gram products — the batching is where the launches go.
//! - **Persistent workspaces** (item 4): per-(r,c) workspace buffers and
//!   pointer tables live in a thread-local cache and are reused every step;
//!   steady state performs zero device allocations. Total workspace is
//!   budgeted (`NSL_MUON_BATCH_MB`, default 256 MiB) — groups larger than
//!   the budget run in chunks, the compiler's memory-budget knob the WGGO
//!   integration can later drive.
//!
//! ## Numerics contract
//! The batched path is deliberately NOT bit-identical to the sequential
//! primitive: cuBLAS strided-batched kernels reduce in a different order
//! than the single-gemm path, and the diagonal fold moves ns_a·x into the
//! GEMM's FMA reduction. The momentum update and the Frobenius sum ARE
//! bit-identical (same two-rounding elementwise math; same stride-256 +
//! tree-128 reduction order as the sequential stats kernel). Consumers gate
//! this path behind an explicit opt-in flag and validate through
//! tolerance/orthogonality differential gates, not byte equality.
//!
//! GPU-only by design: eligible params must be device-resident f32 rank-2 —
//! anything else is a loud precondition abort (the flag is a GPU perf
//! opt-in; CPU training keeps the reference path).

#![cfg(feature = "cuda")]

use std::cell::RefCell;
use std::collections::HashMap;

use crate::list::NslList;
use crate::muon_prof::{scope, Region};
use crate::tensor::NslTensor;

/// Must match muon.rs / stdlib muon.nsl exactly.
const NS_A: f64 = 3.4445;
const NS_B: f64 = -4.7750;
const NS_C: f64 = 2.0315;
const FROB_EPS: f64 = 0.000_000_1;

const BLOCK: i64 = 256;

/// Tensor-core TF32 for the batched NS GEMMs (default ON — the batch path
/// is already an explicit opt-in with tolerance-based gates, and NS is a
/// coarse polynomial approximation). `NSL_MUON_BATCH_TF32=0` forces strict
/// FP32 compute for A/B numerics debugging.
fn tf32_enabled() -> bool {
    static TF32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TF32.get_or_init(|| std::env::var("NSL_MUON_BATCH_TF32").ok().as_deref() != Some("0"))
}

fn budget_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        let mb = std::env::var("NSL_MUON_BATCH_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 16)
            .unwrap_or(256);
        mb * 1024 * 1024
    })
}

/// Persistent per-shape workspace. All pointers are raw device allocations
/// obtained through the caching allocator (`inner::alloc_managed`), plus one
/// pinned host staging block for the pointer tables.
struct GroupWs {
    /// Capacity in matrices.
    cap_k: usize,
    /// Oriented dims: rp = min(r,c), cp = max(r,c).
    rp: usize,
    cp: usize,
    /// [cap_k, rp, cp] iterate + ping-pong buffer.
    y: u64,
    y2: u64,
    /// [cap_k, rp, rp] Gram + Gram² workspaces.
    a: u64,
    aa: u64,
    /// [cap_k] per-matrix Frobenius sum-squares.
    norms: u64,
    /// Device pointer tables: m, g, p (cap_k u64 entries each).
    tab_m: u64,
    tab_g: u64,
    tab_p: u64,
    /// Pinned host staging for one table upload (cap_k u64 entries).
    stage: u64,
}

impl GroupWs {
    fn bytes_per_matrix(rp: usize, cp: usize) -> usize {
        // y + y2 + a + aa + norms entry (+ 3 table entries + staging, noise).
        (2 * rp * cp + 2 * rp * rp) * 4 + 4 + 4 * 8
    }

    fn alloc(rp: usize, cp: usize, cap_k: usize) -> GroupWs {
        let f = |elems: usize| -> u64 {
            crate::cuda::inner::set_oom_context("muon_batch_workspace");
            crate::cuda::inner::alloc_managed(elems * 4) as u64
        };
        let tab = |k: usize| -> u64 { crate::cuda::inner::alloc_managed(k * 8) as u64 };
        // Sized for THREE tables staged side by side (m, g, p).
        let stage = unsafe {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            let r = cudarc::driver::sys::cuMemAllocHost_v2(&mut p, (3 * cap_k * 8).max(8));
            assert_eq!(
                r,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "muon_batch: pinned staging alloc failed"
            );
            p as u64
        };
        GroupWs {
            cap_k,
            rp,
            cp,
            y: f(cap_k * rp * cp),
            y2: f(cap_k * rp * cp),
            a: f(cap_k * rp * rp),
            aa: f(cap_k * rp * rp),
            norms: f(cap_k),
            tab_m: tab(cap_k),
            tab_g: tab(cap_k),
            tab_p: tab(cap_k),
            stage,
        }
    }

    /// Release the workspace. The buffers came from `alloc_managed` (the
    /// caching allocator, which hands out split-block INTERIOR pointers), so
    /// they must go back through `free_managed` — raw `cuMemFree` on an
    /// interior pointer aborts, and on a segment base it double-frees the
    /// allocator's tracking (review finding H1). The caller must ensure no
    /// in-flight device work references the buffers (stream sync).
    fn free(self) {
        for p in [self.y, self.y2, self.a, self.aa, self.norms, self.tab_m, self.tab_g, self.tab_p]
        {
            crate::cuda::inner::free_managed(p as *mut std::ffi::c_void);
        }
        unsafe {
            cudarc::driver::sys::cuMemFreeHost(self.stage as *mut std::ffi::c_void);
        }
    }
}

thread_local! {
    static WS_CACHE: RefCell<HashMap<(usize, usize), GroupWs>> = RefCell::new(HashMap::new());
}

#[allow(clippy::too_many_arguments)]
fn launch(ptx: &'static str, name: &'static [u8], grid: [i64; 3], block: [i64; 3], args: &[*mut std::ffi::c_void], smem: u32) {
    let r = crate::cuda::inner::kernel_launch(ptx.as_ptr(), name.as_ptr(), grid, block, args, smem);
    assert_eq!(r as u32, 0, "muon_batch kernel {name:?} failed: {r:?}");
}

/// One chunk of one shape group: `k` matrices of logical shape [r, c].
/// The m/g/p pointer tables must already be uploaded (`upload_tables_3`).
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    ws: &GroupWs,
    k: i64,
    r: usize,
    c: usize,
    ns_steps: i64,
    momentum: f64,
    nesterov: bool,
    lr: f64,
    weight_decay: f64,
) {
    use crate::cuda::fused_kernels::{
        MUON_BATCH_MOM_F32_PTX, MUON_BATCH_PACK_F32_PTX, MUON_BATCH_POLY_F32_PTX,
        MUON_BATCH_SUMSQ_F32_PTX, MUON_BATCH_UPDATE_F32_PTX,
    };

    let n = (r * c) as i64;
    let tall = r > c;
    let (rp, cp) = (r.min(c) as i64, r.max(c) as i64);
    let tr: u32 = if tall { 1 } else { 0 };
    let grid_n = [(n + BLOCK - 1) / BLOCK, k, 1];
    let blk = [BLOCK, 1, 1];

    let mu = momentum as f32;
    let nest: u32 = if nesterov { 1 } else { 0 };

    // K1: m = mu*m + g (in place, bit-exact vs the stdlib arm).
    {
        let _p = scope(Region::MomentumUpdate);
        let mut a0 = ws.tab_m;
        let mut a1 = ws.tab_g;
        let mut a2 = mu;
        let mut a3 = n as u32;
        let args: [*mut std::ffi::c_void; 4] = [
            &mut a0 as *mut _ as *mut std::ffi::c_void,
            &mut a1 as *mut _ as *mut std::ffi::c_void,
            &mut a2 as *mut _ as *mut std::ffi::c_void,
            &mut a3 as *mut _ as *mut std::ffi::c_void,
        ];
        launch(
            MUON_BATCH_MOM_F32_PTX,
            b"nsl_muon_batch_mom_f32\0",
            grid_n,
            blk,
            &args,
            0,
        );
    }

    // K2: norms[i] = sum(u²), deterministic per-matrix tree.
    {
        let _p = scope(Region::FrobeniusReduce);
        let mut a0 = ws.tab_m;
        let mut a1 = ws.tab_g;
        let mut a2 = mu;
        let mut a3 = nest;
        let mut a4 = n as u32;
        let mut a5 = ws.norms;
        let args: [*mut std::ffi::c_void; 6] = [
            &mut a0 as *mut _ as *mut std::ffi::c_void,
            &mut a1 as *mut _ as *mut std::ffi::c_void,
            &mut a2 as *mut _ as *mut std::ffi::c_void,
            &mut a3 as *mut _ as *mut std::ffi::c_void,
            &mut a4 as *mut _ as *mut std::ffi::c_void,
            &mut a5 as *mut _ as *mut std::ffi::c_void,
        ];
        launch(
            MUON_BATCH_SUMSQ_F32_PTX,
            b"nsl_muon_batch_sumsq_f32\0",
            [k, 1, 1],
            blk,
            &args,
            256 * 4,
        );
    }

    // K3: pack normalized update into the wide-layout workspace.
    {
        let _p = scope(Region::NormalizeScale);
        let mut a0 = ws.tab_m;
        let mut a1 = ws.tab_g;
        let mut a2 = mu;
        let mut a3 = nest;
        let mut a4 = ws.norms;
        let mut a5 = ws.y;
        let mut a6 = r as u32;
        let mut a7 = c as u32;
        let mut a8 = tr;
        let mut a9 = FROB_EPS as f32;
        let args: [*mut std::ffi::c_void; 10] = [
            &mut a0 as *mut _ as *mut std::ffi::c_void,
            &mut a1 as *mut _ as *mut std::ffi::c_void,
            &mut a2 as *mut _ as *mut std::ffi::c_void,
            &mut a3 as *mut _ as *mut std::ffi::c_void,
            &mut a4 as *mut _ as *mut std::ffi::c_void,
            &mut a5 as *mut _ as *mut std::ffi::c_void,
            &mut a6 as *mut _ as *mut std::ffi::c_void,
            &mut a7 as *mut _ as *mut std::ffi::c_void,
            &mut a8 as *mut _ as *mut std::ffi::c_void,
            &mut a9 as *mut _ as *mut std::ffi::c_void,
        ];
        launch(
            MUON_BATCH_PACK_F32_PTX,
            b"nsl_muon_batch_pack_f32\0",
            grid_n,
            blk,
            &args,
            0,
        );
    }

    // NS iterations on the wide iterate Y [k, rp, cp].
    let r2 = (rp * rp) as i64;
    let mut y_cur = ws.y;
    let mut y_nxt = ws.y2;
    for _ in 0..ns_steps {
        // A = Y Yᵀ (row-major) == colmajor gemm(T, N): symmetric result.
        {
            let _p = scope(Region::GramGemm);
            let res = unsafe {
                crate::cuda::cublas_inner::sgemm_strided_batched_raw(
                    true,
                    false,
                    rp,
                    rp,
                    cp,
                    y_cur as *const f32,
                    cp,
                    rp * cp,
                    y_cur as *const f32,
                    cp,
                    rp * cp,
                    ws.a as *mut f32,
                    rp,
                    r2,
                    k,
                    tf32_enabled(),
                )
            };
            res.expect("muon_batch gram gemm failed");
        }
        // AA = A A (both symmetric, mapping cancels).
        {
            let _p = scope(Region::GramSqGemm);
            let res = unsafe {
                crate::cuda::cublas_inner::sgemm_strided_batched_raw(
                    false,
                    false,
                    rp,
                    rp,
                    rp,
                    ws.a as *const f32,
                    rp,
                    r2,
                    ws.a as *const f32,
                    rp,
                    r2,
                    ws.aa as *mut f32,
                    rp,
                    r2,
                    k,
                    tf32_enabled(),
                )
            };
            res.expect("muon_batch gram-square gemm failed");
        }
        // B' = ns_b·A + ns_c·AA + ns_a·I (in place over A), then Y' = B' Y.
        {
            let _p = scope(Region::PolyGemm);
            let mut a0 = ws.a;
            let mut a1 = ws.aa;
            let mut a2 = NS_A as f32;
            let mut a3 = NS_B as f32;
            let mut a4 = NS_C as f32;
            let mut a5 = rp as u32;
            let mut a6 = r2 as u32;
            let args: [*mut std::ffi::c_void; 7] = [
                &mut a0 as *mut _ as *mut std::ffi::c_void,
                &mut a1 as *mut _ as *mut std::ffi::c_void,
                &mut a2 as *mut _ as *mut std::ffi::c_void,
                &mut a3 as *mut _ as *mut std::ffi::c_void,
                &mut a4 as *mut _ as *mut std::ffi::c_void,
                &mut a5 as *mut _ as *mut std::ffi::c_void,
                &mut a6 as *mut _ as *mut std::ffi::c_void,
            ];
            launch(
                MUON_BATCH_POLY_F32_PTX,
                b"nsl_muon_batch_poly_f32\0",
                [(r2 + BLOCK - 1) / BLOCK, k, 1],
                blk,
                &args,
                0,
            );
            // Row-major Y' = B'·Y maps to colmajor gemm(N,N) with A := Y
            // (colmajor [cp,rp]), B := B' (symmetric — swap cancels).
            let res = unsafe {
                crate::cuda::cublas_inner::sgemm_strided_batched_raw(
                    false,
                    false,
                    cp,
                    rp,
                    rp,
                    y_cur as *const f32,
                    cp,
                    rp * cp,
                    ws.a as *const f32,
                    rp,
                    r2,
                    y_nxt as *mut f32,
                    cp,
                    rp * cp,
                    k,
                    tf32_enabled(),
                )
            };
            res.expect("muon_batch poly gemm failed");
        }
        std::mem::swap(&mut y_cur, &mut y_nxt);
    }

    // K5: p = decay*p - step*o (o unpacked from Y, transposed back if tall).
    {
        let _p = scope(Region::ParamUpdate);
        let ratio = r as f64 / c as f64;
        let scale = if ratio < 1.0 { 1.0 } else { ratio }.sqrt();
        let decay: f32 = if weight_decay > 0.0 {
            (1.0 - lr * weight_decay) as f32
        } else {
            1.0
        };
        let step: f32 = (lr * scale) as f32;
        let mut a0 = ws.tab_p;
        let mut a1 = y_cur;
        let mut a2 = r as u32;
        let mut a3 = c as u32;
        let mut a4 = tr;
        let mut a5 = decay;
        let mut a6 = step;
        let args: [*mut std::ffi::c_void; 7] = [
            &mut a0 as *mut _ as *mut std::ffi::c_void,
            &mut a1 as *mut _ as *mut std::ffi::c_void,
            &mut a2 as *mut _ as *mut std::ffi::c_void,
            &mut a3 as *mut _ as *mut std::ffi::c_void,
            &mut a4 as *mut _ as *mut std::ffi::c_void,
            &mut a5 as *mut _ as *mut std::ffi::c_void,
            &mut a6 as *mut _ as *mut std::ffi::c_void,
        ];
        launch(
            MUON_BATCH_UPDATE_F32_PTX,
            b"nsl_muon_batch_update_f32\0",
            grid_n,
            blk,
            &args,
            0,
        );
    }
}

/// Batched Muon step over the train block's full param/grad/state lists.
///
/// Handles ONLY Muon-routed (route flag 0) rank-2 params — the compiled
/// optimizer loop keeps calling the stdlib `muon_step` for AdamW-routed and
/// non-rank-2 params (and skips the ones this call owns). Eligible params
/// that are not device-resident f32 are a loud precondition abort: the
/// opt-in flag promises a GPU training run.
#[no_mangle]
pub extern "C" fn nsl_muon_step_batch(
    params_list: i64,
    grads_list: i64,
    m_list: i64,
    routes_list: i64,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    nesterov: i64,
    ns_steps: f64,
) {
    let params = NslList::from_ptr(params_list);
    let grads = NslList::from_ptr(grads_list);
    let ms = NslList::from_ptr(m_list);
    let routes = NslList::from_ptr(routes_list);
    let count = params.len as usize;
    assert_eq!(grads.len as usize, count, "muon_batch: grads len mismatch");
    assert_eq!(ms.len as usize, count, "muon_batch: m len mismatch");
    assert_eq!(routes.len as usize, count, "muon_batch: routes len mismatch");
    if !(ns_steps >= 1.0 && ns_steps.fract() == 0.0) {
        eprintln!("nsl: muon_batch ns_steps must be a positive integer (got {ns_steps})");
        std::process::abort();
    }

    // Group eligible params by shape, preserving encounter order.
    let mut group_keys: Vec<(usize, usize)> = Vec::new();
    let mut groups: HashMap<(usize, usize), (Vec<u64>, Vec<u64>, Vec<u64>)> = HashMap::new();
    for i in 0..count {
        let route = unsafe { *routes.data.add(i) };
        if route != 0 {
            continue; // AdamW-routed: stdlib arm handles it.
        }
        let p_ptr = unsafe { *params.data.add(i) };
        let p = NslTensor::from_ptr(p_ptr);
        if p.ndim != 2 {
            continue; // non-rank-2: stdlib arm's AdamW branch handles it.
        }
        let g_ptr = unsafe { *grads.data.add(i) };
        let m_ptr = unsafe { *ms.data.add(i) };
        let g = NslTensor::from_ptr(g_ptr);
        let m = NslTensor::from_ptr(m_ptr);
        for (t, what) in [(&*p, "param"), (&*g, "grad"), (&*m, "momentum")] {
            if t.device == 0 || t.dtype != 1 || !t.is_contiguous() {
                eprintln!(
                    "nsl: --muon-batch-ns requires device-resident contiguous f32 tensors; \
                     {what} #{i} is device={} dtype={} contiguous={}. Refusing (run without \
                     the flag, or move training to the GPU).",
                    t.device,
                    t.dtype,
                    t.is_contiguous()
                );
                std::process::abort();
            }
        }
        let (r, c) = unsafe { ((*p.shape) as usize, (*p.shape.add(1)) as usize) };
        if r == 0 || c == 0 {
            eprintln!("nsl: muon_batch: empty rank-2 param #{i} has no orthogonal factor");
            std::process::abort();
        }
        let key = (r, c);
        let entry = groups.entry(key).or_insert_with(|| {
            group_keys.push(key);
            (Vec::new(), Vec::new(), Vec::new())
        });
        entry.0.push(m.data as u64);
        entry.1.push(g.data as u64);
        entry.2.push(p.data as u64);
    }

    let steps = ns_steps as i64;
    for key in group_keys {
        let (m_ptrs, g_ptrs, p_ptrs) = groups.remove(&key).unwrap();
        let (r, c) = key;
        let (rp, cp) = (r.min(c), r.max(c));
        let per = GroupWs::bytes_per_matrix(rp, cp);
        // grid.y carries the matrix index — clamp to the hardware's 65535
        // cap so tiny-shape mega-groups sub-chunk instead of aborting the
        // launch (review finding).
        let chunk = (budget_bytes() / per).clamp(1, m_ptrs.len()).min(65535);
        WS_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            // Grow-only capacity per shape; reallocate when a bigger chunk
            // arrives (workspace cache is keyed by oriented shape).
            let needs_realloc = match cache.get(&(rp, cp)) {
                Some(ws) => ws.cap_k < chunk,
                None => true,
            };
            if needs_realloc {
                if let Some(old) = cache.remove(&(rp, cp)) {
                    // The previous same-oriented-shape group (e.g. [c,r] after
                    // [r,c]) may still have kernels/GEMMs and staging DMAs in
                    // flight on these buffers — quiesce the compute stream
                    // before releasing them (review finding H1).
                    unsafe {
                        crate::cuda::inner::ensure_context();
                        let r = cudarc::driver::sys::cuStreamSynchronize(
                            crate::cuda::inner::current_stream(),
                        );
                        assert_eq!(
                            r,
                            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                            "muon_batch: stream sync before workspace realloc failed"
                        );
                    }
                    old.free();
                }
                cache.insert((rp, cp), GroupWs::alloc(rp, cp, chunk));
            }
            let ws = cache.get(&(rp, cp)).unwrap();
            for start in (0..m_ptrs.len()).step_by(chunk) {
                let end = (start + chunk).min(m_ptrs.len());
                // Upload the three tables back-to-back through the shared
                // staging block; each rewrite must wait for the previous
                // async copy, so stage them at distinct offsets instead.
                upload_tables_3(
                    ws,
                    &m_ptrs[start..end],
                    &g_ptrs[start..end],
                    &p_ptrs[start..end],
                );
                run_chunk(
                    ws,
                    (end - start) as i64,
                    r,
                    c,
                    steps,
                    momentum,
                    nesterov != 0,
                    lr,
                    weight_decay,
                );
            }
        });
    }
}

/// Stage m/g/p tables at distinct offsets of one pinned block sized for
/// three tables, then enqueue three async uploads.
///
/// The stream synchronize BEFORE the host rewrite is load-bearing: the
/// previous chunk's async DMAs read this same pinned block (groups with the
/// same oriented shape share one workspace), and stream ordering only
/// orders device work — a host-side rewrite would race the in-flight copy.
/// One ~µs sync per chunk, a handful of chunks per optimizer step.
fn upload_tables_3(ws: &GroupWs, m: &[u64], g: &[u64], p: &[u64]) {
    unsafe {
        crate::cuda::inner::ensure_context();
        let r = cudarc::driver::sys::cuStreamSynchronize(crate::cuda::inner::current_stream());
        assert_eq!(
            r,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "muon_batch: stream sync before table staging failed"
        );
    }
    let k = m.len();
    unsafe {
        let base = ws.stage as *mut u64;
        std::ptr::copy_nonoverlapping(m.as_ptr(), base, k);
        std::ptr::copy_nonoverlapping(g.as_ptr(), base.add(ws.cap_k), k);
        std::ptr::copy_nonoverlapping(p.as_ptr(), base.add(2 * ws.cap_k), k);
    }
    // Enqueue on the COMPUTE stream (not memcpy_htod_async's transfer
    // stream): the batch kernels ride the compute stream and must be
    // ordered after these copies; cross-stream uploads would race them.
    let up = |table: u64, off: usize| unsafe {
        let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
            table,
            (ws.stage as usize + off * 8) as *const std::ffi::c_void,
            k * 8,
            crate::cuda::inner::current_stream(),
        );
        assert_eq!(
            r,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "muon_batch: table upload failed"
        );
    };
    up(ws.tab_m, 0);
    up(ws.tab_g, ws.cap_k);
    up(ws.tab_p, 2 * ws.cap_k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{
        nsl_tensor_free, nsl_tensor_mul_scalar, nsl_tensor_sub, nsl_tensor_to_device,
    };

    pub(super) fn make_gpu_f32(rows: i64, cols: i64, seed: f64) -> i64 {
        let n = (rows * cols) as usize;
        let data: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 0.37 + seed).sin() * 0.02)
            .collect();
        let cpu = crate::tensor::creation::create_tensor_from_f64_data(&data, &[rows, cols]);
        let gpu = nsl_tensor_to_device(cpu, 1);
        nsl_tensor_free(cpu);
        gpu
    }

    fn read_gpu_f32(ptr: i64) -> Vec<f32> {
        let cpu = nsl_tensor_to_device(ptr, 0);
        let t = NslTensor::from_ptr(cpu);
        // CPU transfer widens to f64 by convention; read whichever landed.
        let out: Vec<f32> = (0..t.len as usize)
            .map(|i| match t.dtype {
                1 => unsafe { *t.data_f32().add(i) },
                _ => (unsafe { *t.data_f64().add(i) }) as f32,
            })
            .collect();
        nsl_tensor_free(cpu);
        out
    }

    /// Sequential reference for the Muon arm using the SAME runtime FFIs the
    /// stdlib lowering emits (two-rounding momentum, per-matrix NS, scaled
    /// decayed update) — mutates m and p in place like the stdlib arm.
    fn sequential_muon_arm(p: i64, g: i64, m: i64, lr: f64, mu: f64, wd: f64, ns: f64) {
        let mm = nsl_tensor_mul_scalar(m, mu, 0);
        let m_new = crate::tensor::nsl_tensor_add(mm, g, crate::tensor::fbip_flags::RELINQUISH_A);
        crate::tensor::nsl_tensor_copy_data(m, m_new);
        nsl_tensor_free(m_new);
        let o = crate::muon::nsl_tensor_muon_orthogonalize(m, ns);
        let t = NslTensor::from_ptr(p);
        let (r, c) = unsafe { (*t.shape as f64, *t.shape.add(1) as f64) };
        let ratio = if r / c < 1.0 { 1.0 } else { r / c };
        let scale = ratio.sqrt();
        let decayed = nsl_tensor_mul_scalar(p, 1.0 - lr * wd, 0);
        let step = nsl_tensor_mul_scalar(o, lr * scale, 0);
        let p_new = nsl_tensor_sub(decayed, step, 0);
        crate::tensor::nsl_tensor_copy_data(p, p_new);
        for x in [o, decayed, step, p_new] {
            nsl_tensor_free(x);
        }
    }

    fn run_case(nesterov: bool) {
        let shapes: [(i64, i64); 3] = [(8, 8), (4, 12), (12, 4)];
        let k_per = 3usize;
        let (lr, mu, wd, ns) = (0.02f64, 0.9f64, 0.01f64, 5.0f64);

        // Two identical universes: batch and sequential.
        let mut batch: Vec<(i64, i64, i64)> = Vec::new();
        let mut seq: Vec<(i64, i64, i64)> = Vec::new();
        for (si, &(r, c)) in shapes.iter().enumerate() {
            for j in 0..k_per {
                let seed = (si * 10 + j) as f64;
                batch.push((
                    make_gpu_f32(r, c, seed),
                    make_gpu_f32(r, c, seed + 0.3),
                    make_gpu_f32(r, c, seed + 0.6),
                ));
                seq.push((
                    make_gpu_f32(r, c, seed),
                    make_gpu_f32(r, c, seed + 0.3),
                    make_gpu_f32(r, c, seed + 0.6),
                ));
            }
        }

        // Batch universe: one FFI call over the whole list.
        let params = crate::list::nsl_list_new();
        let grads = crate::list::nsl_list_new();
        let ms = crate::list::nsl_list_new();
        let routes = crate::list::nsl_list_new();
        for &(p, g, m) in &batch {
            crate::list::nsl_list_push(params, p);
            crate::list::nsl_list_push(grads, g);
            crate::list::nsl_list_push(ms, m);
            crate::list::nsl_list_push(routes, 0);
        }
        nsl_muon_step_batch(params, grads, ms, routes, lr, mu, wd, nesterov as i64, ns);
        unsafe { crate::cuda::inner::ensure_context() };
        unsafe { cudarc::driver::sys::cuCtxSynchronize() };

        // Sequential universe (non-nesterov only: the sequential helper
        // matches the stdlib non-nesterov arm; nesterov is covered by the
        // determinism assertion below).
        if !nesterov {
            for &(p, g, m) in &seq {
                sequential_muon_arm(p, g, m, lr, mu, wd, ns);
            }
            for (i, (&(pb, _, mb), &(ps, _, ms_))) in batch.iter().zip(seq.iter()).enumerate() {
                let (vb, vs) = (read_gpu_f32(pb), read_gpu_f32(ps));
                let (mvb, mvs) = (read_gpu_f32(mb), read_gpu_f32(ms_));
                // Momentum update is designed bit-exact (same two-rounding
                // elementwise math as mul_scalar + add).
                for (e, (a, b)) in mvb.iter().zip(&mvs).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "momentum diverged: matrix {i} elem {e}: {a} vs {b}"
                    );
                }
                // Params: batched NS reorders GEMM reductions — tolerance.
                let mut moved = false;
                for (e, (a, b)) in vb.iter().zip(&vs).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-4,
                        "param diverged: matrix {i} elem {e}: batch {a} vs seq {b}"
                    );
                    if (a - b).abs() > 0.0 || a.abs() > 1e-6 {
                        moved = true;
                    }
                }
                assert!(moved, "matrix {i}: params never moved (vacuous test)");
            }
        }

        // Determinism: a second identical batch universe must land bit-equal.
        let mut batch2: Vec<(i64, i64, i64)> = Vec::new();
        for (si, &(r, c)) in shapes.iter().enumerate() {
            for j in 0..k_per {
                let seed = (si * 10 + j) as f64;
                batch2.push((
                    make_gpu_f32(r, c, seed),
                    make_gpu_f32(r, c, seed + 0.3),
                    make_gpu_f32(r, c, seed + 0.6),
                ));
            }
        }
        let params2 = crate::list::nsl_list_new();
        let grads2 = crate::list::nsl_list_new();
        let ms2 = crate::list::nsl_list_new();
        let routes2 = crate::list::nsl_list_new();
        for &(p, g, m) in &batch2 {
            crate::list::nsl_list_push(params2, p);
            crate::list::nsl_list_push(grads2, g);
            crate::list::nsl_list_push(ms2, m);
            crate::list::nsl_list_push(routes2, 0);
        }
        nsl_muon_step_batch(params2, grads2, ms2, routes2, lr, mu, wd, nesterov as i64, ns);
        unsafe { cudarc::driver::sys::cuCtxSynchronize() };
        for (i, (&(pb, _, _), &(pb2, _, _))) in batch.iter().zip(batch2.iter()).enumerate() {
            let (v1, v2) = (read_gpu_f32(pb), read_gpu_f32(pb2));
            for (e, (a, b)) in v1.iter().zip(&v2).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "nondeterministic batch: matrix {i} elem {e}"
                );
            }
        }

        for &(p, g, m) in batch.iter().chain(seq.iter()).chain(batch2.iter()) {
            nsl_tensor_free(p);
            nsl_tensor_free(g);
            nsl_tensor_free(m);
        }
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn batch_matches_sequential_and_is_deterministic() {
        run_case(false);
    }

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn batch_nesterov_runs_and_is_deterministic() {
        run_case(true);
    }

    /// Review H1 regression: two groups sharing one ORIENTED workspace key
    /// with UNEQUAL counts ([4,12] x2 then [12,4] x5) force the grow-realloc
    /// branch while the first group's work may still be in flight. Under the
    /// bug this freed caching-allocator interior pointers through raw
    /// cuMemFree with no sync (abort or silent corruption); now it must
    /// quiesce, free through free_managed, and still match the sequential
    /// path.
    #[test]
    #[ignore = "requires CUDA GPU"]
    fn workspace_grow_realloc_same_oriented_key() {
        let (lr, mu, wd, ns) = (0.02f64, 0.9f64, 0.0f64, 5.0f64);
        let mut batch: Vec<(i64, i64, i64)> = Vec::new();
        let mut seq: Vec<(i64, i64, i64)> = Vec::new();
        let spec: Vec<(i64, i64)> = std::iter::repeat_n((4i64, 12i64), 2)
            .chain(std::iter::repeat_n((12, 4), 5))
            .collect();
        for (j, &(r, c)) in spec.iter().enumerate() {
            let seed = 40.0 + j as f64;
            batch.push((
                tests::make_gpu_f32(r, c, seed),
                tests::make_gpu_f32(r, c, seed + 0.3),
                tests::make_gpu_f32(r, c, seed + 0.6),
            ));
            seq.push((
                tests::make_gpu_f32(r, c, seed),
                tests::make_gpu_f32(r, c, seed + 0.3),
                tests::make_gpu_f32(r, c, seed + 0.6),
            ));
        }
        let params = crate::list::nsl_list_new();
        let grads = crate::list::nsl_list_new();
        let ms = crate::list::nsl_list_new();
        let routes = crate::list::nsl_list_new();
        for &(p, g, m) in &batch {
            crate::list::nsl_list_push(params, p);
            crate::list::nsl_list_push(grads, g);
            crate::list::nsl_list_push(ms, m);
            crate::list::nsl_list_push(routes, 0);
        }
        nsl_muon_step_batch(params, grads, ms, routes, lr, mu, wd, 0, ns);
        unsafe {
            crate::cuda::inner::ensure_context();
            cudarc::driver::sys::cuCtxSynchronize();
        }
        for &(p, g, m) in &seq {
            tests::sequential_muon_arm(p, g, m, lr, mu, wd, ns);
        }
        for (i, (&(pb, _, _), &(ps, _, _))) in batch.iter().zip(seq.iter()).enumerate() {
            let (vb, vs) = (tests::read_gpu_f32(pb), tests::read_gpu_f32(ps));
            for (e, (a, b)) in vb.iter().zip(&vs).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "grow-realloc divergence: matrix {i} elem {e}: {a} vs {b}"
                );
            }
        }
        for &(p, g, m) in batch.iter().chain(seq.iter()) {
            crate::tensor::nsl_tensor_free(p);
            crate::tensor::nsl_tensor_free(g);
            crate::tensor::nsl_tensor_free(m);
        }
    }

    #[test]
    fn workspace_budget_chunk_math() {
        // [1280,3520]-class matrix: ~49 MB workspace each; a 256 MB budget
        // must chunk a 48-matrix group rather than allocating 2.3 GB.
        let per = GroupWs::bytes_per_matrix(1280, 3520);
        assert!(per > 45_000_000 && per < 55_000_000, "per-matrix ws {per}");
        let chunk = (256usize * 1024 * 1024 / per).clamp(1, 48);
        assert!(chunk >= 4 && chunk < 48, "chunk {chunk}");
    }
}

#[cfg(test)]
mod perf_driver {
    use super::*;
    use crate::tensor::nsl_tensor_free;

    /// Perf-campaign timing driver: one coder500m step's Muon-routed load
    /// through the BATCHED engine (the sequential twin lives in
    /// muon::tests::profile_ns_500m_step; compare wall times):
    ///
    ///   cargo test -p nsl-runtime --features cuda --release --lib -- \
    ///     --ignored profile_batch_ns_500m_step --test-threads=1 --nocapture
    #[test]
    #[ignore = "requires CUDA GPU (timing driver, not a correctness gate)"]
    fn profile_batch_ns_500m_step() {
        let classes: [(i64, i64, usize); 4] =
            [(1280, 1280, 48), (1280, 640, 48), (1280, 3520, 48), (3520, 1280, 24)];
        let mut all: Vec<(i64, i64, i64)> = Vec::new();
        for (ci, &(r, c, count)) in classes.iter().enumerate() {
            for j in 0..count {
                let seed = (ci * 100 + j) as f64;
                all.push((
                    tests::make_gpu_f32(r, c, seed),
                    tests::make_gpu_f32(r, c, seed + 0.3),
                    tests::make_gpu_f32(r, c, seed + 0.6),
                ));
            }
        }
        let params = crate::list::nsl_list_new();
        let grads = crate::list::nsl_list_new();
        let ms = crate::list::nsl_list_new();
        let routes = crate::list::nsl_list_new();
        for &(p, g, m) in &all {
            crate::list::nsl_list_push(params, p);
            crate::list::nsl_list_push(grads, g);
            crate::list::nsl_list_push(ms, m);
            crate::list::nsl_list_push(routes, 0);
        }
        // Warmup primes cublas plans, workspaces and the allocator.
        nsl_muon_step_batch(params, grads, ms, routes, 0.02, 0.9, 0.01, 0, 5.0);
        unsafe {
            crate::cuda::inner::ensure_context();
            cudarc::driver::sys::cuCtxSynchronize();
        }
        let wall = std::time::Instant::now();
        nsl_muon_step_batch(params, grads, ms, routes, 0.02, 0.9, 0.01, 0, 5.0);
        unsafe { cudarc::driver::sys::cuCtxSynchronize() };
        eprintln!(
            "[muon-batch-driver] one 500M-step Muon load (168 matrices, ns=5, batched): {:.1} ms wall",
            wall.elapsed().as_secs_f64() * 1e3
        );
        crate::muon_prof::report();
        for (p, g, m) in all {
            nsl_tensor_free(p);
            nsl_tensor_free(g);
            nsl_tensor_free(m);
        }
    }
}
