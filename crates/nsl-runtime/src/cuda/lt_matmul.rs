//! cublasLt issue path for the bf16-storage GEMMs: explicit kernel selection
//! with a real workspace, instead of `cublasGemmEx(CUBLAS_GEMM_DFALT)`.
//!
//! ## Why this exists
//!
//! With the weight-cast cache amortizing the operand casts (PR #546), the
//! composed bf16 run still reaches only ~20.6% of the 200.3 TFLOPS bf16
//! roofline — the remaining gap is inside the GEMM kernels themselves. On
//! CUDA 13.x `cublasGemmEx` is a thin shim over cublasLt that asks the
//! heuristic for ONE candidate with the handle's small internal workspace and
//! takes it unseen. This module asks the same heuristic for up to
//! [`HEURISTIC_CANDIDATES`] candidates with a caller-sized workspace
//! (`NSL_MATMUL_BF16_LT_WORKSPACE_MIB`, default 64 — split-k and
//! wide-tile kernels need room DFALT never offers), times each candidate once
//! per shape on the live operands, and caches the winner for the life of the
//! process.
//!
//! ## Scope and engagement
//!
//! Opt-in: `NSL_MATMUL_BF16_LT=1` (same literal-"1" discipline as the mode
//! resolvers), consulted once. Reaches ONLY the bf16-storage arm of
//! `gemm_bf16_mode` — the FAST_TF32 low-intensity arm and every non-bf16 mode
//! keep their proven issue paths. Any decline (no heuristic candidate, a
//! failed launch, an unexpected operand op) falls back to the GemmEx call
//! mid-flight; the fallback is counted and printed once, because a run that
//! silently degraded to DFALT would fake its own A/B.
//!
//! ## Determinism
//!
//! The heuristic preference masks split-k reduction schemes to
//! `CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE` only: `INPLACE` reduces with
//! atomics on C, which is run-to-run NON-deterministic — it would silently
//! void the `--deterministic` contract for every matmul — and `OUTPUT_TYPE`
//! is excluded with it (no precision downside here since out == compute ==
//! f32, but keeping the mask minimal keeps the determinism argument
//! one-line). A cached algo replays the identical kernel every launch, so a
//! plan-cache hit cannot introduce variation.
//!
//! ## Autotune (`NSL_MATMUL_BF16_LT_TUNE`, default on)
//!
//! First encounter of a shape times each candidate: 1 warmup + min-of-3, on
//! the live bf16 operands, writing into a TRANSIENT scratch D with beta=0 —
//! never into the caller's C, whose beta may be 1 (the wgrad accumulate; a
//! timing rep that wrote C would accumulate into live gradients). The scratch
//! is freed before the real launch issues. Timing costs one `cuCtxSynchronize`
//! bracket per rep, ~a dozen launches per shape, once per process; a training
//! run has a few dozen shapes. Autotune self-disables (heuristic[0] is used,
//! untimed) when:
//!
//! * `--cuda-graphs` is active: a first-sight shape inside a real capture
//!   would synchronize mid-capture and invalidate it — the same
//!   attempt-burning failure shape the cast cache refuses graphs over. The
//!   untimed pick keeps Lt available under graphs (algo per shape is stable,
//!   so region digests are too).
//! * the scratch D would exceed 1 GiB (ldc*n*4 bytes), or its RAW driver
//!   allocation fails — tuning is optional, so a nearly-full card gets the
//!   untimed pick, never an OOM kill (the scratch and the workspace both
//!   bypass `alloc_managed` PRECISELY because that path cannot fail — it
//!   exits the process — see [`raw_device_alloc`]).
//! * `--deterministic` is active: a TIMED winner depends on machine state at
//!   first use, so two runs of one program could select different kernels
//!   and differ in bits ACROSS processes — the exact promise --deterministic
//!   makes. The untimed heuristic[0] is a pure function of (shapes, library,
//!   device, workspace cap) and keeps that promise. Within one process the
//!   cached plan makes every repeat bit-identical regardless. (An embedder
//!   that toggles the flag MID-process keeps plans tuned before the toggle;
//!   compiled programs set it before the first GEMM, so only a foreign host
//!   driving the runtime directly can reach that state.)
//!
//! Timed candidates run on the CURRENT device state: tune on a quiet card
//! for production-representative picks. A busy card yields a valid but
//! possibly suboptimal winner — correctness never depends on the timing.
//!
//! ## Interactions
//!
//! * **Cast cache**: composes freely — the cache decides what bytes feed the
//!   GEMM, this module decides which kernel consumes them. A cached image is
//!   read-only to this module.
//! * **SR rounding**: composes freely for the same reason (the dither is in
//!   the operand bytes before we see them).
//! * **cuda-graphs**: `cublasLtMatmul` with a cached algo is capture-safe
//!   (no sync, stable kernel); autotune is the only sync source and is
//!   disabled under graphs, see above. The workspace and plan handles live
//!   for the whole process, so captured nodes never outlive the addresses
//!   they reference.
//!
//! ## VRAM
//!
//! One workspace allocation (`NSL_MATMUL_BF16_LT_WORKSPACE_MIB`, default 64
//! MiB) pinned for the life of the process, shared by every Lt GEMM — safe
//! because all GPU work is single-threaded on one compute stream (process
//! invariant documented at the launch path), so uses serialize. Plus the
//! transient autotune scratch described above.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

use cudarc::cublaslt::sys as lt;

/// Candidates requested from `cublasLtMatmulAlgoGetHeuristic` per shape.
/// 8 is the customary depth (the heuristic orders by predicted speed;
/// measured winners in the wild are overwhelmingly in the first handful).
const HEURISTIC_CANDIDATES: usize = 8;

/// Autotune scratch cap: a shape whose f32 output exceeds this many bytes is
/// picked untimed (heuristic[0]) rather than risk an allocation spike.
const TUNE_SCRATCH_CAP_BYTES: usize = 1 << 30;

/// One cached issue plan: the descriptor/layout handles and the chosen algo.
/// Raw cublasLt handles are opaque host-side objects, valid until destroyed;
/// plans live for the whole process (destroyed only by [`reset_for_test`]).
struct Plan {
    desc: lt::cublasLtMatmulDesc_t,
    adesc: lt::cublasLtMatrixLayout_t,
    bdesc: lt::cublasLtMatrixLayout_t,
    cdesc: lt::cublasLtMatrixLayout_t,
    algo: lt::cublasLtMatmulAlgo_t,
    ws_bytes: usize,
}

// SAFETY: the handles are plain host allocations owned by cublasLt; nothing
// about them is thread-affine. The map's Mutex serializes all access, and all
// GPU work is single-threaded besides (process invariant).
unsafe impl Send for Plan {}

/// Shape key. `None` plans are negative entries: the heuristic (or a launch)
/// failed for this shape once, so every later call takes the GemmEx fallback
/// without re-querying per GEMM.
type Key = (i32, i32, i32, i32, i32, i32, i32, i32); // (opa, opb, m, n, k, lda, ldb, ldc)

static PLANS: LazyLock<Mutex<HashMap<Key, Option<Plan>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// Diagnostic counters (relaxed; teardown line + gates). `ISSUED` counts
// GEMMs launched through cublasLt; `TUNED` counts shapes that went through a
// timed autotune (untimed heuristic[0] picks do not count); `FALLBACKS`
// counts GEMMs that reached this module enabled and still took GemmEx.
static ISSUED: AtomicU64 = AtomicU64::new(0);
static TUNED: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// `NSL_MATMUL_BF16_LT=1` engages the path. Literal-"1" discipline, read
/// once: a typo cannot change which kernels a run measured.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::matmul_config::config().lt)
}

fn tune_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::matmul_config::config().lt_tune)
}

fn workspace_bytes_configured() -> usize {
    static MIB: OnceLock<usize> = OnceLock::new();
    *MIB.get_or_init(|| {
        // Item 4: --bf16-lt-workspace-mib through the config sink, which
        // applies the SAME 4096 clamp so the fingerprint records the effective
        // value. Still clamped here: a hand-written C host could call
        // nsl_set_matmul_config directly.
        (crate::matmul_config::config().lt_workspace_mib as usize).min(4096)
    }) << 20
}

/// Raw driver allocation for this module's two buffers (the process-lifetime
/// workspace and the transient tune scratch). Deliberately NOT
/// `alloc_managed`:
///
/// * `alloc_managed` cannot fail — every OOM path ends in `oom_fatal` ->
///   process exit. This module's whole degradation story ("skip the tune",
///   "run with zero workspace") requires an allocation that can say no.
/// * `alloc_managed`'s first stop is the transient arena's pin. A
///   plan-vs-reality drift could hand a PROCESS-LIFETIME workspace an arena
///   slot the arena re-places every step — permanent aliasing, split-k
///   scribbling over a live tensor (review finding on the first commit).
///   The raw driver path cannot intersect the arena or the caching
///   allocator's free lists at all.
///
/// Returns 0 on failure.
fn raw_device_alloc(bytes: usize) -> u64 {
    if bytes == 0 {
        return 0;
    }
    super::inner::ensure_context();
    let mut ptr: cudarc::driver::sys::CUdeviceptr = 0;
    // SAFETY: out-pointer to a live local; result checked.
    let r = unsafe { cudarc::driver::sys::cuMemAlloc_v2(&mut ptr, bytes) };
    if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        ptr as u64
    } else {
        0
    }
}

/// Free a [`raw_device_alloc`] buffer. Callers must ensure all enqueued work
/// touching it has completed (the tune loop's per-rep `cuCtxSynchronize`
/// brackets do exactly that before the scratch free).
fn raw_device_free(ptr: u64) {
    if ptr == 0 {
        return;
    }
    // SAFETY: `ptr` came from cuMemAlloc_v2 and is freed exactly once.
    unsafe {
        cudarc::driver::sys::cuMemFree_v2(ptr as cudarc::driver::sys::CUdeviceptr);
    }
}

/// (issued, tuned_shapes, fallbacks) — see the counter docs.
pub(crate) fn stats() -> (u64, u64, u64) {
    (
        ISSUED.load(Ordering::Relaxed),
        TUNED.load(Ordering::Relaxed),
        FALLBACKS.load(Ordering::Relaxed),
    )
}

struct LtHandle(lt::cublasLtHandle_t);
// SAFETY: cublasLt handles are documented thread-safe; ours is used from the
// single GPU-work thread only.
unsafe impl Send for LtHandle {}
unsafe impl Sync for LtHandle {}

/// Process-global cublasLt handle, created on first engaged GEMM. `None`
/// (creation failed) makes every call fall back — printed once below.
fn lt_handle() -> Option<lt::cublasLtHandle_t> {
    static HANDLE: OnceLock<Option<LtHandle>> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            super::inner::ensure_context();
            let mut h: lt::cublasLtHandle_t = std::ptr::null_mut();
            // SAFETY: out-pointer to a live local; checked before use.
            let st = unsafe { lt::cublasLtCreate(&mut h) };
            if st != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || h.is_null() {
                eprintln!(
                    "[nsl-matmul] cublasLtCreate failed ({st:?}) — \
                     NSL_MATMUL_BF16_LT=1 falls back to cublasGemmEx for this run"
                );
                return None;
            }
            eprintln!(
                "[nsl-matmul] bf16 GEMMs via cublasLt heuristics \
                 (NSL_MATMUL_BF16_LT=1): workspace {} MiB, autotune {}",
                workspace_bytes_configured() >> 20,
                if super::graph_capture::enabled() {
                    "off (--cuda-graphs: a mid-capture sync would invalidate \
                     the capture; heuristic[0] per shape)"
                } else if crate::deterministic_ops::is_deterministic() {
                    "off (--deterministic: a timed winner varies across runs; \
                     heuristic[0] per shape)"
                } else if tune_enabled() {
                    "on (1 warmup + min-of-3 per shape, first sight)"
                } else {
                    "off (NSL_MATMUL_BF16_LT_TUNE=0; heuristic[0] per shape)"
                },
            );
            Some(LtHandle(h))
        })
        .as_ref()
        .map(|h| h.0)
}

/// Process-global workspace: allocated once (raw driver path — see
/// [`raw_device_alloc`]), never freed (captured graph nodes may reference it
/// for the life of the process). `(0, 0)` when the allocation failed —
/// heuristics then select zero-workspace kernels, which is still at worst
/// DFALT-equivalent.
fn workspace() -> (u64, usize) {
    static WS: OnceLock<(u64, usize)> = OnceLock::new();
    *WS.get_or_init(|| {
        let bytes = workspace_bytes_configured();
        if bytes == 0 {
            return (0, 0);
        }
        let ptr = raw_device_alloc(bytes);
        if ptr == 0 {
            eprintln!(
                "[nsl-matmul] cublasLt workspace alloc ({} MiB) failed — \
                 continuing with zero workspace (kernel choice degrades)",
                bytes >> 20
            );
            (0, 0)
        } else {
            (ptr, bytes)
        }
    })
}

fn print_fallback_once(what: &str) {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[nsl-matmul] cublasLt path fell back to cublasGemmEx ({what}); \
             counted in the bf16_lt teardown counters — first occurrence only"
        );
    }
}

/// Destroy a half-built plan's handles (creation-failure cleanup path).
unsafe fn destroy_handles(
    desc: lt::cublasLtMatmulDesc_t,
    adesc: lt::cublasLtMatrixLayout_t,
    bdesc: lt::cublasLtMatrixLayout_t,
    cdesc: lt::cublasLtMatrixLayout_t,
) {
    // SAFETY (all four): each handle is either null or a live object created
    // in `build_plan`; destroy tolerates our nulls being skipped.
    unsafe {
        if !cdesc.is_null() {
            lt::cublasLtMatrixLayoutDestroy(cdesc);
        }
        if !bdesc.is_null() {
            lt::cublasLtMatrixLayoutDestroy(bdesc);
        }
        if !adesc.is_null() {
            lt::cublasLtMatrixLayoutDestroy(adesc);
        }
        if !desc.is_null() {
            lt::cublasLtMatmulDescDestroy(desc);
        }
    }
}

/// Build the descriptor/layout set for one shape, query the heuristic, and
/// (optionally) time the candidates on the live operands. Returns `None` on
/// any failure — cached as a negative entry by the caller.
#[allow(clippy::too_many_arguments)]
unsafe fn build_plan(
    handle: lt::cublasLtHandle_t,
    opa: i32,
    opb: i32,
    m: i32,
    n: i32,
    k: i32,
    a16: *const std::ffi::c_void,
    lda: i32,
    b16: *const std::ffi::c_void,
    ldb: i32,
    ldc: i32,
) -> Option<Plan> {
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let (ws_ptr, ws_bytes) = workspace();

    let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
    let mut adesc: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut bdesc: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut cdesc: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();

    // SAFETY: out-pointers to live locals; every status is checked and the
    // half-built set is destroyed on failure.
    let built = unsafe {
        'build: {
            if lt::cublasLtMatmulDescCreate(
                &mut desc,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                lt::cudaDataType_t::CUDA_R_32F,
            ) != ok
            {
                break 'build false;
            }
            // TRANSA/TRANSB hold a cublasOperation_t as int32 (0 = N, 1 = T;
            // the enum is not re-exported by the Lt bindings).
            let set_op = |attr: lt::cublasLtMatmulDescAttributes_t, v: i32| {
                lt::cublasLtMatmulDescSetAttribute(
                    desc,
                    attr,
                    &v as *const i32 as *const std::ffi::c_void,
                    std::mem::size_of::<i32>(),
                ) == ok
            };
            if !set_op(lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, opa)
                || !set_op(lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, opb)
            {
                break 'build false;
            }
            // Layout rows/cols describe the operand AS STORED (column-major
            // default order); the desc's TRANSA/TRANSB say how it is read.
            let bf16 = lt::cudaDataType_t::CUDA_R_16BF;
            let (a_rows, a_cols) = if opa == 0 { (m, k) } else { (k, m) };
            let (b_rows, b_cols) = if opb == 0 { (k, n) } else { (n, k) };
            if lt::cublasLtMatrixLayoutCreate(&mut adesc, bf16, a_rows as u64, a_cols as u64, lda as i64) != ok
                || lt::cublasLtMatrixLayoutCreate(&mut bdesc, bf16, b_rows as u64, b_cols as u64, ldb as i64) != ok
                || lt::cublasLtMatrixLayoutCreate(
                    &mut cdesc,
                    lt::cudaDataType_t::CUDA_R_32F,
                    m as u64,
                    n as u64,
                    ldc as i64,
                ) != ok
            {
                break 'build false;
            }
            true
        }
    };
    if !built {
        // SAFETY: handles are exactly as `build_plan` left them.
        unsafe { destroy_handles(desc, adesc, bdesc, cdesc) };
        return None;
    }

    // Heuristic query, workspace-capped and determinism-masked.
    let mut results: [lt::cublasLtMatmulHeuristicResult_t; HEURISTIC_CANDIDATES] =
        // SAFETY: plain-old-data out-array, fully written by the query up to
        // `returned` and only read up to `returned`.
        unsafe { std::mem::zeroed() };
    let mut returned: i32 = 0;
    // SAFETY: preference is created, attribute-set, used for one query, and
    // destroyed before any early return below.
    let query_ok = unsafe {
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        if lt::cublasLtMatmulPreferenceCreate(&mut pref) != ok {
            false
        } else {
            let ws_cap = ws_bytes;
            let red_mask: u32 =
                lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
            let pref_ok = lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_cap as *const usize as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            ) == ok
                && lt::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                    &red_mask as *const u32 as *const std::ffi::c_void,
                    std::mem::size_of::<u32>(),
                ) == ok;
            let got = pref_ok
                && lt::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    desc,
                    adesc,
                    bdesc,
                    cdesc,
                    cdesc, // D shares C's layout (in-place C = alpha*AB + beta*C)
                    pref,
                    HEURISTIC_CANDIDATES as i32,
                    results.as_mut_ptr(),
                    &mut returned,
                ) == ok;
            lt::cublasLtMatmulPreferenceDestroy(pref);
            got
        }
    };
    let candidates: Vec<&lt::cublasLtMatmulHeuristicResult_t> = if query_ok {
        results[..returned.max(0) as usize]
            .iter()
            .filter(|r| r.state == ok && r.workspaceSize <= ws_bytes)
            .collect()
    } else {
        Vec::new()
    };
    if candidates.is_empty() {
        // SAFETY: handles are live and about to be abandoned.
        unsafe { destroy_handles(desc, adesc, bdesc, cdesc) };
        return None;
    }

    // Candidate timing. Every timed launch goes to a transient scratch D
    // with beta = 0 — NEVER the caller's C (its beta may be 1: wgrad
    // accumulates, and a timing rep into it would corrupt live gradients).
    // Scratch D footprint follows the C LAYOUT, not just m*n: the D
    // descriptor carries `ld = ldc`, so a timed launch touches up to
    // `ldc*(n-1)+m` elements. Today's callers all pass ldc == m; sizing by
    // ldc keeps a future ldc>m caller from turning timing reps into device
    // OOB writes (review finding).
    let out_bytes = (ldc as usize) * (n as usize) * 4;
    // `--deterministic` disables the TIMED tune (read live, not cached: the
    // mode is set at program init, after this module's OnceLocks may exist in
    // tests): a timing-based winner is a function of machine state at first
    // use, so two runs of the same program could select different kernels and
    // produce different bits ACROSS processes — run-to-run byte
    // reproducibility is exactly what --deterministic sells. heuristic[0] is
    // a pure function of (shapes, library, device, workspace cap), so the
    // untimed pick keeps Lt available and cross-run stable.
    let tune = tune_enabled()
        && !super::graph_capture::enabled()
        && !crate::deterministic_ops::is_deterministic()
        && out_bytes <= TUNE_SCRATCH_CAP_BYTES;
    let mut chosen: usize = 0;
    if tune && candidates.len() > 1 {
        // Fallible by design: on a nearly-full card the tune is skipped
        // (untimed heuristic[0]) instead of the process dying — which is
        // what routing this through alloc_managed's oom_fatal would do.
        let scratch = raw_device_alloc(out_bytes) as *mut std::ffi::c_void;
        if !scratch.is_null() {
            let stream = super::inner::current_stream() as lt::cudaStream_t;
            let (zero, one) = (0.0f32, 1.0f32);
            let mut best = f64::INFINITY;
            let mut timings: Vec<f64> = Vec::with_capacity(candidates.len());
            for (i, cand) in candidates.iter().enumerate() {
                let mut cand_best = f64::INFINITY;
                let mut failed = false;
                for rep in 0..4 {
                    // SAFETY: all handles/pointers live; scratch is sized
                    // m*n*4 and used as both C and D with beta=0 (C is
                    // never read); ws_ptr is either null with ws 0 or a
                    // live process-lifetime block.
                    let st = unsafe {
                        cudarc::driver::sys::cuCtxSynchronize();
                        let t0 = std::time::Instant::now();
                        let st = lt::cublasLtMatmul(
                            handle,
                            desc,
                            &one as *const f32 as *const std::ffi::c_void,
                            a16,
                            adesc,
                            b16,
                            bdesc,
                            &zero as *const f32 as *const std::ffi::c_void,
                            scratch,
                            cdesc,
                            scratch,
                            cdesc,
                            &cand.algo,
                            ws_ptr as *mut std::ffi::c_void,
                            cand.workspaceSize,
                            stream,
                        );
                        cudarc::driver::sys::cuCtxSynchronize();
                        if st == ok && rep > 0 {
                            // rep 0 is the warmup: module load + Lt's own
                            // first-touch costs must not pick the winner.
                            cand_best = cand_best.min(t0.elapsed().as_secs_f64());
                        }
                        st
                    };
                    if st != ok {
                        failed = true;
                        break;
                    }
                }
                timings.push(if failed { f64::INFINITY } else { cand_best });
                if !failed && cand_best < best {
                    best = cand_best;
                    chosen = i;
                }
            }
            // Every timing rep ended in a cuCtxSynchronize, so no enqueued
            // work still reads the scratch.
            raw_device_free(scratch as u64);
            TUNED.fetch_add(1, Ordering::Relaxed);
            // The candidates were all timed anyway — under
            // NSL_MATMUL_BF16_LT_VERBOSE=1, say per shape what the untimed
            // heuristic[0] pick would have cost vs the winner. Diagnostic
            // only: heuristic[0]-with-our-workspace is NOT the same kernel
            // DFALT's tiny internal workspace selects, so this line explains
            // an A/B delta, it does not substitute for one.
            static VERBOSE: OnceLock<bool> = OnceLock::new();
            if *VERBOSE.get_or_init(|| {
                std::env::var("NSL_MATMUL_BF16_LT_VERBOSE").ok().as_deref() == Some("1")
            }) {
                let h0 = timings.first().copied().unwrap_or(f64::INFINITY);
                eprintln!(
                    "[bf16-lt] tune opa={opa} opb={opb} m={m} n={n} k={k}: \
                     winner #{chosen} {:.3} ms, heuristic[0] {:.3} ms, \
                     {} candidates",
                    best * 1e3,
                    h0 * 1e3,
                    timings.len(),
                );
            }
        }
    }

    Some(Plan {
        desc,
        adesc,
        bdesc,
        cdesc,
        algo: candidates[chosen].algo,
        ws_bytes: candidates[chosen].workspaceSize,
    })
}

/// Issue one bf16-storage GEMM through cublasLt. Returns `true` when the
/// product was launched (the caller must NOT also GemmEx it); `false` means
/// fall back — nothing was written to `c`.
///
/// Arguments are in cuBLAS column-major order, exactly as `gemm_bf16_mode`
/// holds them: `opa`/`opb` are `cublasOperation_t` as int32 (0 = N, 1 = T —
/// anything else declines), `a16`/`b16` are the bf16 operand images
/// (scratch or cast-cache), `c` is f32.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_bf16_f32(
    opa: i32,
    opb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: &f32,
    a16: *const std::ffi::c_void,
    lda: i32,
    b16: *const std::ffi::c_void,
    ldb: i32,
    beta: &f32,
    c: *mut f32,
    ldc: i32,
) -> bool {
    if !(opa == 0 || opa == 1) || !(opb == 0 || opb == 1) {
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        print_fallback_once("unexpected operand op");
        return false;
    }
    let Some(handle) = lt_handle() else {
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return false; // creation failure already printed its own line
    };

    let key: Key = (opa, opb, m, n, k, lda, ldb, ldc);
    // Plan lookup / creation. The map lock is NOT held across build_plan
    // (which synchronizes the device while timing) nor across the launch —
    // holding a lock over device-blocking work is how unrelated paths
    // (teardown atexit, test resets) end up serialized behind a GEMM.
    let hit = {
        let plans = PLANS.lock().unwrap();
        plans.get(&key).map(|p| p.as_ref().map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes)))
    };
    let (desc, adesc, bdesc, cdesc, algo, ws_need) = match hit {
        Some(Some(t)) => t,
        Some(None) => {
            // Negative entry: this shape declined once; stay declined.
            FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        None => {
            // SAFETY: forwarded pointers are the caller's live operands.
            let plan =
                unsafe { build_plan(handle, opa, opb, m, n, k, a16, lda, b16, ldb, ldc) };
            let tuple = plan
                .as_ref()
                .map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes));
            // First-writer wins: two racing builders would both insert; keep
            // the incumbent and destroy the newcomer's handles instead of
            // leaking or double-caching. (Single-threaded today; cheap
            // insurance regardless.)
            let mut plans = PLANS.lock().unwrap();
            match plans.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    if let Some(p) = plan {
                        // SAFETY: newcomer's handles, never published.
                        unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
                    }
                    match e.get() {
                        Some(p) => (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes),
                        None => {
                            FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            return false;
                        }
                    }
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(plan);
                    match tuple {
                        Some(t) => t,
                        None => {
                            FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            print_fallback_once("no heuristic candidate for a shape");
                            return false;
                        }
                    }
                }
            }
        }
    };

    let (ws_ptr, _) = workspace();
    let stream = super::inner::current_stream() as lt::cudaStream_t;
    // SAFETY: plan handles are process-lifetime; operand/output pointers are
    // the caller's live device buffers; C doubles as D (in-place accumulate,
    // the layout the heuristic was queried with).
    let st = unsafe {
        lt::cublasLtMatmul(
            handle,
            desc,
            alpha as *const f32 as *const std::ffi::c_void,
            a16,
            adesc,
            b16,
            bdesc,
            beta as *const f32 as *const std::ffi::c_void,
            c as *const std::ffi::c_void,
            cdesc,
            c as *mut std::ffi::c_void,
            cdesc,
            &algo,
            ws_ptr as *mut std::ffi::c_void,
            ws_need,
            stream,
        )
    };
    if st != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        // A failed launch writes nothing, so GemmEx can still compute the
        // product. Poison the plan so the shape does not fail-retry per GEMM,
        // destroying the displaced handles (a bare insert leaked 4 objects
        // per poisoned shape — review finding). Known residual: under
        // --deterministic a TRANSIENT failure here switches this shape
        // Lt->GemmEx mid-run, which can diverge two seeded runs — but a
        // cached, heuristic-validated algo failing transiently is itself a
        // machine-state anomaly no local choice repairs; documented rather
        // than masked.
        let displaced = PLANS.lock().unwrap().insert(key, None);
        if let Some(Some(p)) = displaced {
            // SAFETY: just removed from the map; no launch is in flight with
            // these handles (single-threaded GPU work) and cublasLt keeps no
            // reference past the failed call.
            unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
        }
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        print_fallback_once("cublasLtMatmul launch failure");
        return false;
    }
    ISSUED.fetch_add(1, Ordering::Relaxed);
    true
}

/// Test-only reset: destroy every plan, clear counters. The Lt handle and
/// workspace survive (process-lifetime by design; captured graphs and the
/// banner's once-semantics depend on them).
#[cfg(any(test, feature = "test-hooks"))]
pub(crate) fn reset_for_test() {
    let drained: Vec<Option<Plan>> = {
        let mut plans = PLANS.lock().unwrap();
        std::mem::take(&mut *plans).into_values().collect()
    };
    for p in drained.into_iter().flatten() {
        // SAFETY: handles were live and are now unreachable from the map.
        unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
    }
    ISSUED.store(0, Ordering::Relaxed);
    TUNED.store(0, Ordering::Relaxed);
    FALLBACKS.store(0, Ordering::Relaxed);
}
