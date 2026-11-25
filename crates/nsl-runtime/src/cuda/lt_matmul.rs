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
//! `--bf16-lt` / `NSL_MATMUL_BF16_LT=1` (same literal-"1" discipline as the
//! mode resolvers), consulted once; and, since November 2026, by default
//! whenever timed plan selection applies (`timed_lt_applies`: tune on, no
//! `--deterministic`, no `--cuda-graphs`), with the enumerated large-tile
//! candidates of the TF32 section added to the heuristic's. Reaches ONLY the
//! bf16-storage arm of `gemm_bf16_mode`. TF32 GEMMs (the default math mode, and the bf16 mode's
//! low-intensity FAST_TF32 arm) have their own timed plans: the TF32 section
//! at the end of this file. Any decline (no heuristic candidate, a
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
use std::sync::OnceLock;

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
/// (opa, opb, m, n, k, lda, ldb, ldc, log2 of the pointer alignment class,
/// C/D stored bf16 rather than f32).
type Key = (i32, i32, i32, i32, i32, i32, i32, i32, u32, bool);

/// Roadmap A4 step 3a: was a `PLANS` `LazyLock<Mutex<..>>` static; now one
/// map per device, parked on the CUDA context and keyed by this type. The
/// `Plan` type and the cache's contents stay here — the context names
/// neither.
#[derive(Default)]
struct PlanCache(HashMap<Key, Option<Plan>>);

/// Run `f` against this device's plan cache. Holds the context's cache mutex
/// for the duration, exactly as the `PLANS` mutex did.
fn with_plans<R>(f: impl FnOnce(&mut HashMap<Key, Option<Plan>>) -> R) -> R {
    super::context::current().with_cache(|c: &mut PlanCache| f(&mut c.0))
}

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

/// Roadmap A4 step 3a: the newtype and its `Send`/`Sync` justification moved
/// to `cuda::context`, which owns the field the handle lives in.
use super::context::LtHandle;

/// Process-global cublasLt handle, created on first engaged GEMM. `None`
/// (creation failed) makes every call fall back — printed once below.
fn lt_handle() -> Option<lt::cublasLtHandle_t> {
    // Roadmap A4 step 3a: per device, not per process — a cublasLt handle
    // binds the context current at `cublasLtCreate`. Same `OnceLock`, so the
    // creation and its one-time failure diagnostic still happen exactly once.
    let ctx = super::context::current();
    ctx.cublaslt
        .get_or_init(|| {
            super::inner::ensure_context();
            let mut h: lt::cublasLtHandle_t = std::ptr::null_mut();
            // SAFETY: out-pointer to a live local; checked before use.
            let st = unsafe { lt::cublasLtCreate(&mut h) };
            if st != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS || h.is_null() {
                crate::nsl_log!(ERROR, "nsl-matmul", 
                    "[nsl-matmul] cublasLtCreate failed ({st:?}) — \
                     NSL_MATMUL_BF16_LT=1 falls back to cublasGemmEx for this run"
                );
                return None;
            }
            crate::nsl_log!(INFO, "nsl-matmul", 
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
    // Roadmap A4 step 3a: a device pointer, so per device. Still allocated
    // once and never freed — captured graph nodes may reference it for the
    // life of the process.
    *super::context::current().lt_workspace.get_or_init(|| {
        let bytes = workspace_bytes_configured();
        if bytes == 0 {
            return (0, 0);
        }
        let ptr = raw_device_alloc(bytes);
        if ptr == 0 {
            crate::nsl_log!(WARN, "nsl-matmul", 
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
        crate::nsl_log!(INFO, "nsl-matmul", 
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
    align_log2: u32,
    c_bf16: bool,
) -> Option<Plan> {
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let (_, ws_bytes) = workspace();
    let c_type = if c_bf16 { lt::cudaDataType_t::CUDA_R_16BF } else { lt::cudaDataType_t::CUDA_R_32F };
    let align = 1u32 << align_log2;

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
                || lt::cublasLtMatrixLayoutCreate(&mut cdesc, c_type, m as u64, n as u64, ldc as i64) != ok
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
            use lt::cublasLtMatmulPreferenceAttributes_t as P;
            let set_u32 = |attr, v: &u32| {
                lt::cublasLtMatmulPreferenceSetAttribute(pref, attr, v as *const u32 as *const std::ffi::c_void, 4) == ok
            };
            let pref_ok = lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_cap as *const usize as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            ) == ok
                && set_u32(P::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red_mask)
                // The operands' real alignment (the plan is keyed by it).
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align);
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
    let mut candidates: Vec<Candidate> = if query_ok {
        results[..returned.max(0) as usize]
            .iter()
            .filter(|r| r.state == ok && r.workspaceSize <= ws_bytes)
            .map(|r| Candidate { algo: r.algo, ws: r.workspaceSize, from_heuristic: true })
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
        && out_bytes <= TUNE_SCRATCH_CAP_BYTES
        && tune_budget_allows(shapes_tuned());
    // Timed selection also tries the enumerated large-tile configurations (the
    // heuristic's few candidates rarely include the fastest kernel; see the
    // TF32 section), when the operands have the alignment `AlgoCheck` assumes.
    if tune && align >= 256 {
        // SAFETY: handle and descriptors are live.
        candidates.extend(unsafe {
            enumerate_configs(
                handle,
                desc,
                adesc,
                bdesc,
                cdesc,
                ws_bytes,
                lt::cudaDataType_t::CUDA_R_16BF,
                c_type,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            )
        });
    }
    let mut chosen: usize = 0;
    if tune && candidates.len() > 1 {
        // SAFETY: the caller's live bf16 operand images for this shape.
        if let Some((i, ms, h0)) =
            unsafe { tune_two_rounds(handle, desc, adesc, bdesc, cdesc, a16, b16, out_bytes, &candidates) }
        {
            chosen = i;
            TUNED.fetch_add(1, Ordering::Relaxed);
            log_tune("bf16-lt", opa, opb, m, n, k, &candidates[i], ms, h0, candidates.len());
        }
    }

    Some(Plan {
        desc,
        adesc,
        bdesc,
        cdesc,
        algo: candidates[chosen].algo,
        ws_bytes: candidates[chosen].ws,
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
    // SAFETY: forwarded unchanged; an f32 C/D.
    unsafe { matmul_bf16(opa, opb, m, n, k, alpha, a16, lda, b16, ldb, beta, c as *mut std::ffi::c_void, ldc, false) }
}

/// [`matmul_bf16_f32`] with the C/D buffer stored bf16 (`c_bf16`) or f32:
/// bf16 activation storage writes a GEMM's activation output as bf16, the
/// f32 accumulator rounded once by the epilogue. The plan cache keys on the
/// output type, so the two never share a plan.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_bf16(
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
    c: *mut std::ffi::c_void,
    ldc: i32,
    c_bf16: bool,
) -> bool {
    // SAFETY: forwarded unchanged; C doubles as D (in place).
    unsafe {
        matmul_bf16_into(
            opa, opb, m, n, k, alpha, a16, lda, b16, ldb, beta, c, c, ldc, c_bf16,
        )
    }
}

/// [`matmul_bf16`] writing `D = alpha * op(A) op(B) + beta * C` into `d` (out of place,
/// `d` with C's layout; `d == c` is the in-place form). The plan cache keys on neither
/// pointer's identity (only the alignment class of all buffers), so the accumulating
/// launch replays the same algorithm as the standalone product of that shape: the same
/// f32 accumulator, then the epilogue's `alpha * acc + beta * C`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_bf16_into(
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
    c: *mut std::ffi::c_void,
    d: *mut std::ffi::c_void,
    ldc: i32,
    c_bf16: bool,
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

    // Alignment class of the three buffers, capped at 256 bytes (as the TF32
    // path): an operand at an odd offset gets its own plan instead of failing,
    // and poisoning, the aligned shape's.
    let align_log2 =
        ((a16 as usize) | (b16 as usize) | (c as usize) | (d as usize) | 256).trailing_zeros();
    let key: Key = (opa, opb, m, n, k, lda, ldb, ldc, align_log2, c_bf16);
    // Plan lookup / creation. The map lock is NOT held across build_plan
    // (which synchronizes the device while timing) nor across the launch —
    // holding a lock over device-blocking work is how unrelated paths
    // (teardown atexit, test resets) end up serialized behind a GEMM.
    let hit = with_plans(|plans| {
        plans.get(&key).map(|p| p.as_ref().map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes)))
    });
    let (desc, adesc, bdesc, cdesc, algo, ws_need) = match hit {
        Some(Some(t)) => t,
        Some(None) => {
            // Negative entry: this shape declined once; stay declined.
            FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        None => {
            // The default route (no --bf16-lt) is here only to be timed: past
            // the budget a new shape goes back to GemmEx, uncached.
            if !enabled() && !tune_budget_allows(shapes_tuned()) {
                print_budget_once();
                return false;
            }
            // SAFETY: forwarded pointers are the caller's live operands.
            let plan =
                unsafe { build_plan(handle, opa, opb, m, n, k, a16, lda, b16, ldb, ldc, align_log2, c_bf16) };
            let tuple = plan
                .as_ref()
                .map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes));
            // First-writer wins: two racing builders would both insert; keep
            // the incumbent and destroy the newcomer's handles instead of
            // leaking or double-caching. (Single-threaded today; cheap
            // insurance regardless.)
            // The closure yields `None` where this block used to `return
            // false` — a closure cannot return from the enclosing fn, and the
            // counter bump and the one-time print happen inside it either way.
            let resolved = with_plans(|plans| match plans.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    if let Some(p) = plan {
                        // SAFETY: newcomer's handles, never published.
                        unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
                    }
                    match e.get() {
                        Some(p) => Some((p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes)),
                        None => {
                            FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            None
                        }
                    }
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(plan);
                    match tuple {
                        Some(t) => Some(t),
                        None => {
                            FALLBACKS.fetch_add(1, Ordering::Relaxed);
                            print_fallback_once("no heuristic candidate for a shape");
                            None
                        }
                    }
                }
            });
            match resolved {
                Some(t) => t,
                None => return false,
            }
        }
    };

    let (ws_ptr, _) = workspace();
    let stream = super::inner::current_stream() as lt::cudaStream_t;
    // SAFETY: plan handles are process-lifetime; operand/output pointers are
    // the caller's live device buffers; D has C's layout (the layout the heuristic
    // was queried with), in place (`d == c`) or out of place.
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
            d,
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
        let displaced = with_plans(|plans| plans.insert(key, None));
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

// ---------------------------------------------------------------------------
// TF32 GEMMs (f32 storage, `CUBLAS_COMPUTE_32F_FAST_TF32`) with an enumerated,
// timed plan per shape.
// ---------------------------------------------------------------------------
//
// ## Why
//
// Under the default math mode NSL issued every 2-D GEMM as `cublasSgemm_v2` on
// a `CUBLAS_TF32_TENSOR_OP_MATH` handle, which takes cuBLAS's heuristic pick
// unseen. On cuBLAS 13.8 / sm_120 that pick is a 64x64-tile
// `cutlass_80_tensorop_s1688gemm_64x64_16x6` kernel for most of a transformer's
// projections and their gradients: 48-57 TFLOPS on random data at Coder-1B's
// shapes, against 67-76 TFLOPS for the 128x128 / 128x256 / 256x128 kernels the
// same library ships for the same shapes (PyTorch's cuBLAS 12.8 heuristic picks
// those; `models/benchmarks/NSL_VS_PYTORCH_2026_11.md`). The heuristic's own
// candidate list does not help: for the NN and NT forms `cublasLtMatmulAlgoGet
// Heuristic` returns one or two candidates, both the slow kernel. The fast ones
// are reachable only by enumerating algorithm configurations
// (`cublasLtMatmulAlgoGetIds` x tile x stages) and checking each with
// `cublasLtMatmulAlgoCheck`.
//
// ## What runs
//
// First sight of a shape (key: ops, m, n, k, leading dimensions, pointer
// alignment class) builds the candidate set: the heuristic's candidates (split-k
// reductions limited to the deterministic `COMPUTE_TYPE` scheme, as on the bf16
// path) plus every enumerated configuration with one of [`ENUM_TILES`] and no
// split-k. Each candidate is timed on the live operands into a scratch output
// (one warm-up launch, then one timed launch between CUDA events), the four
// fastest again with the median of five, and the winner is cached for the life
// of the process. bf16-storage products take the same tune (`build_plan`). A plan-cache hit replays the identical kernel, so within a run
// every repeat of a shape is bit-identical.
//
// The arithmetic is the same class as `cublasSgemm` on the TF32 handle: f32
// operands rounded to TF32 inside the kernel, f32 accumulation, f32 output.
// Different kernels sum in a different order, so results move within TF32's
// precision, not beyond it.
//
// ## When it does not run (cublasSgemm on the TF32 handle, as before)
//
// * `--deterministic`: a timed winner depends on machine state, so two runs of
//   one program could pick different kernels and differ in bits; cuBLAS's
//   heuristic is a pure function of the shape, library and device.
// * `--cuda-graphs`: the tune synchronizes the device, which would invalidate a
//   capture.
// * `--no-bf16-lt-tune` (fingerprint `mmlttune=0`): the switch that turns timed
//   plan selection off, for TF32 as for bf16.
// * small products (`m*n*k` below [`TF32_LT_MIN_MNK`]): launch-bound, and the
//   tune would cost more than it can save.
// * no candidate, a failed handle, or a failed launch: the caller issues the
//   `cublasSgemm` it always did (a failed launch wrote nothing).

/// Below this many multiply-adds a GEMM stays on `cublasSgemm`: ~4 us of TF32
/// work, where the kernel choice is worth less than the tune costs.
pub(crate) const TF32_LT_MIN_MNK: u64 = 1 << 28;

/// The tiles the enumeration tries (`cublasLtMatmulTile_t` ids): 64x256 (19),
/// 128x128 (20), 256x64 (21), 128x256 (23), 256x128 (24). Every enumerated
/// winner on Coder-500M and Coder-1B's projection, dgrad and wgrad shapes, TF32
/// and bf16, was one of these (`perf/host-gemm` probes, November 2026); smaller
/// tiles never won and only multiply the tune's cost. The heuristic's own
/// candidates (any tile, including the sm_120 `nvjet` kernels) are always
/// timed alongside.
const ENUM_TILES: [i32; 5] = [19, 20, 21, 23, 24];

/// Candidates re-timed in the second round of a tune.
const TUNE_FINALISTS: usize = 4;

/// Shapes a process times, over both paths. A training step has a few dozen
/// distinct GEMM shapes; a program whose shapes keep changing (prefill over
/// prompts of every length, say) would otherwise pay a tune, and grow the plan
/// caches, for each new one. Past the budget a new TF32 shape takes
/// `cublasSgemm` and a new bf16 one GemmEx (or `--bf16-lt`'s untimed pick);
/// shapes already tuned keep their plans.
const TUNE_BUDGET_SHAPES: u64 = 128;

/// Whether one more shape may be timed, given how many have been.
fn tune_budget_allows(tuned_so_far: u64) -> bool {
    tuned_so_far < TUNE_BUDGET_SHAPES
}

/// Shapes timed so far, every path (bf16, TF32, MXFP8).
fn shapes_tuned() -> u64 {
    TUNED.load(Ordering::Relaxed)
        + TF32_TUNED.load(Ordering::Relaxed)
        + MX_TUNED.load(Ordering::Relaxed)
}

fn print_budget_once() {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        crate::nsl_log!(INFO, "nsl-matmul",
            "[nsl-matmul] {TUNE_BUDGET_SHAPES} GEMM shapes timed: new shapes from here on \
             take cuBLAS's own pick (already timed shapes keep theirs)"
        );
    }
}

static TF32_ISSUED: AtomicU64 = AtomicU64::new(0);
static TF32_TUNED: AtomicU64 = AtomicU64::new(0);
static TF32_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// (issued, tuned_shapes, fallbacks) for the TF32 path.
pub(crate) fn tf32_stats() -> (u64, u64, u64) {
    (
        TF32_ISSUED.load(Ordering::Relaxed),
        TF32_TUNED.load(Ordering::Relaxed),
        TF32_FALLBACKS.load(Ordering::Relaxed),
    )
}

/// (opa, opb, m, n, k, lda, ldb, ldc, log2 of the pointer alignment class).
type Tf32Key = (i32, i32, i32, i32, i32, i32, i32, i32, u32);

/// The TF32 plans, separate from the bf16 ones: the layouts differ in dtype, so
/// a key collision between the two would replay a plan built for other types.
#[derive(Default)]
struct Tf32PlanCache(HashMap<Tf32Key, Option<Plan>>);

fn with_tf32_plans<R>(f: impl FnOnce(&mut HashMap<Tf32Key, Option<Plan>>) -> R) -> R {
    super::context::current().with_cache(|c: &mut Tf32PlanCache| f(&mut c.0))
}

/// Does a GEMM of this size take the timed cublasLt path (TF32, and bf16-storage
/// products without `--bf16-lt`)? Checked before anything is built, so a
/// declining call costs a few loads.
pub(crate) fn timed_lt_applies(m: u64, n: u64, k: u64) -> bool {
    m.saturating_mul(n).saturating_mul(k) >= TF32_LT_MIN_MNK
        && tune_enabled()
        && !super::graph_capture::enabled()
        && !crate::deterministic_ops::is_deterministic()
}

/// One candidate algorithm and the workspace it needs.
struct Candidate {
    algo: lt::cublasLtMatmulAlgo_t,
    ws: usize,
    from_heuristic: bool,
}

/// Every configuration of every algorithm cublasLt implements for these types
/// (A and B `ab_type`, C and D f32, `ct` compute) that `cublasLtMatmulAlgoCheck`
/// accepts for this problem, restricted to [`ENUM_TILES`] and no split-k.
///
/// # Safety
/// `handle` and the four descriptors are live.
#[allow(clippy::too_many_arguments)]
unsafe fn enumerate_configs(
    handle: lt::cublasLtHandle_t,
    desc: lt::cublasLtMatmulDesc_t,
    adesc: lt::cublasLtMatrixLayout_t,
    bdesc: lt::cublasLtMatrixLayout_t,
    cdesc: lt::cublasLtMatrixLayout_t,
    ws_cap: usize,
    ab_type: lt::cudaDataType_t,
    c_type: lt::cudaDataType_t,
    ct: lt::cublasComputeType_t,
) -> Vec<Candidate> {
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let f32t = lt::cudaDataType_t::CUDA_R_32F;
    let mut ids = [0i32; 128];
    let mut nids = 0i32;
    // SAFETY: out-array of 128 ints and an out-count, as the API documents.
    if unsafe {
        lt::cublasLtMatmulAlgoGetIds(handle, ct, f32t, ab_type, ab_type, c_type, c_type, 128, ids.as_mut_ptr(), &mut nids)
    } != ok
    {
        return Vec::new();
    }
    // An attribute that is a list of i32 (tile ids, stage ids): ask its size,
    // then read it. An algorithm without the attribute yields `[0]`, the
    // "undefined, let the library choose" value.
    let cap_list = |algo: &lt::cublasLtMatmulAlgo_t, attr: lt::cublasLtMatmulAlgoCapAttributes_t| {
        let mut bytes = 0usize;
        // SAFETY: a size query (null buffer, zero size) on a live algo.
        unsafe { lt::cublasLtMatmulAlgoCapGetAttribute(algo, attr, std::ptr::null_mut(), 0, &mut bytes) };
        let n = bytes / std::mem::size_of::<i32>();
        if n == 0 {
            return vec![0i32];
        }
        let mut v = vec![0i32; n];
        let mut written = 0usize;
        // SAFETY: `v` holds exactly `bytes` bytes.
        let st = unsafe {
            lt::cublasLtMatmulAlgoCapGetAttribute(algo, attr, v.as_mut_ptr() as *mut std::ffi::c_void, bytes, &mut written)
        };
        if st != ok { vec![0i32] } else { v }
    };
    let mut out = Vec::new();
    for &id in &ids[..nids.clamp(0, 128) as usize] {
        // SAFETY: zeroed POD out-struct, filled by AlgoInit before any read.
        let mut algo: lt::cublasLtMatmulAlgo_t = unsafe { std::mem::zeroed() };
        if unsafe { lt::cublasLtMatmulAlgoInit(handle, ct, f32t, ab_type, ab_type, c_type, c_type, id, &mut algo) } != ok {
            continue;
        }
        let tiles = cap_list(&algo, lt::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_TILE_IDS);
        let stages = cap_list(&algo, lt::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_STAGES_IDS);
        for &tile in tiles.iter().filter(|t| ENUM_TILES.contains(t)) {
            for &stage in &stages {
                let mut a = algo;
                let one = 1i32;
                let set = |a: &mut lt::cublasLtMatmulAlgo_t, attr, v: &i32| {
                    // SAFETY: an i32 attribute written from a live local.
                    let st = unsafe {
                        lt::cublasLtMatmulAlgoConfigSetAttribute(a, attr, v as *const i32 as *const std::ffi::c_void, 4)
                    };
                    st == ok
                };
                use lt::cublasLtMatmulAlgoConfigAttributes_t as A;
                if !set(&mut a, A::CUBLASLT_ALGO_CONFIG_TILE_ID, &tile)
                    || !set(&mut a, A::CUBLASLT_ALGO_CONFIG_STAGES_ID, &stage)
                    || !set(&mut a, A::CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one)
                {
                    continue;
                }
                // SAFETY: zeroed POD out-struct; descriptors are live.
                let mut chk: lt::cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
                if unsafe { lt::cublasLtMatmulAlgoCheck(handle, desc, adesc, bdesc, cdesc, cdesc, &a, &mut chk) } != ok
                    || chk.workspaceSize > ws_cap
                {
                    continue;
                }
                out.push(Candidate { algo: a, ws: chk.workspaceSize, from_heuristic: false });
            }
        }
    }
    out
}

/// Time one launch of each candidate (after a warm-up launch) into `scratch`,
/// median of `reps` single launches between CUDA events. `f32::INFINITY` for a
/// candidate whose launch fails.
///
/// # Safety
/// All handles live; `scratch` holds the C layout's footprint; the operands are
/// the caller's live device buffers.
#[allow(clippy::too_many_arguments)]
unsafe fn time_candidates(
    handle: lt::cublasLtHandle_t,
    desc: lt::cublasLtMatmulDesc_t,
    adesc: lt::cublasLtMatrixLayout_t,
    bdesc: lt::cublasLtMatrixLayout_t,
    cdesc: lt::cublasLtMatrixLayout_t,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    scratch: *mut std::ffi::c_void,
    ws: (u64, usize),
    cands: &[&Candidate],
    reps: usize,
) -> Vec<f32> {
    use cudarc::driver::sys as drv;
    let stream = super::inner::current_stream();
    let (zero, one) = (0.0f32, 1.0f32);
    let launch = |c: &Candidate| {
        // SAFETY: as the function's contract; beta = 0, so `scratch` is only
        // written.
        let st = unsafe {
            lt::cublasLtMatmul(
                handle,
                desc,
                &one as *const f32 as *const std::ffi::c_void,
                a,
                adesc,
                b,
                bdesc,
                &zero as *const f32 as *const std::ffi::c_void,
                scratch,
                cdesc,
                scratch,
                cdesc,
                &c.algo,
                ws.0 as *mut std::ffi::c_void,
                c.ws,
                stream as lt::cudaStream_t,
            )
        };
        st == lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS
    };
    let mut ev = [std::ptr::null_mut(); 2];
    for e in &mut ev {
        // SAFETY: out-pointer to a live local.
        if unsafe { drv::cuEventCreate(e, 0) } != drv::CUresult::CUDA_SUCCESS {
            return vec![f32::INFINITY; cands.len()];
        }
    }
    let mut times = Vec::with_capacity(cands.len());
    for c in cands {
        if !launch(c) {
            times.push(f32::INFINITY);
            continue;
        }
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let mut ms = f32::INFINITY;
            // SAFETY: both events are live; the launch between them is on the
            // stream they are recorded on.
            unsafe {
                drv::cuEventRecord(ev[0], stream);
                let launched = launch(c);
                drv::cuEventRecord(ev[1], stream);
                drv::cuEventSynchronize(ev[1]);
                if launched {
                    drv::cuEventElapsedTime_v2(&mut ms, ev[0], ev[1]);
                }
            }
            ts.push(ms);
        }
        ts.sort_by(f32::total_cmp);
        times.push(ts[ts.len() / 2]);
    }
    for e in ev {
        // SAFETY: created above, synchronized, destroyed once.
        unsafe { drv::cuEventDestroy_v2(e) };
    }
    times
}

/// Build (and time) the plan for one TF32 shape. `None` on any failure.
///
/// # Safety
/// The pointers are the caller's live device operands for this shape.
#[allow(clippy::too_many_arguments)]
unsafe fn tf32_build_plan(
    handle: lt::cublasLtHandle_t,
    key: &Tf32Key,
    a: *const f32,
    b: *const f32,
) -> Option<Plan> {
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let &(opa, opb, m, n, k, lda, ldb, ldc, align_log2) = key;
    let (_, ws_bytes) = workspace();
    let f32t = lt::cudaDataType_t::CUDA_R_32F;
    let mut desc: lt::cublasLtMatmulDesc_t = std::ptr::null_mut();
    let mut adesc: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut bdesc: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut cdesc: lt::cublasLtMatrixLayout_t = std::ptr::null_mut();
    // SAFETY: out-pointers to live locals; statuses checked; the half-built
    // set is destroyed on failure.
    let built = unsafe {
        lt::cublasLtMatmulDescCreate(&mut desc, lt::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32, f32t) == ok
            && [
                (lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, opa),
                (lt::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, opb),
            ]
            .iter()
            .all(|(attr, v)| {
                lt::cublasLtMatmulDescSetAttribute(desc, *attr, v as *const i32 as *const std::ffi::c_void, 4) == ok
            })
            && {
                let (ar, ac) = if opa == 0 { (m, k) } else { (k, m) };
                let (br, bc) = if opb == 0 { (k, n) } else { (n, k) };
                lt::cublasLtMatrixLayoutCreate(&mut adesc, f32t, ar as u64, ac as u64, lda as i64) == ok
                    && lt::cublasLtMatrixLayoutCreate(&mut bdesc, f32t, br as u64, bc as u64, ldb as i64) == ok
                    && lt::cublasLtMatrixLayoutCreate(&mut cdesc, f32t, m as u64, n as u64, ldc as i64) == ok
            }
    };
    if !built {
        // SAFETY: handles as left above.
        unsafe { destroy_handles(desc, adesc, bdesc, cdesc) };
        return None;
    }

    // Heuristic candidates, told the operands' real alignment.
    let align = 1u32 << align_log2;
    let mut heur: [lt::cublasLtMatmulHeuristicResult_t; HEURISTIC_CANDIDATES] =
        // SAFETY: POD out-array, read only up to `returned`.
        unsafe { std::mem::zeroed() };
    let mut returned = 0i32;
    // SAFETY: the preference is created, set, used once, destroyed.
    unsafe {
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        if lt::cublasLtMatmulPreferenceCreate(&mut pref) == ok {
            use lt::cublasLtMatmulPreferenceAttributes_t as P;
            let red: u32 = lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
            let set_u32 = |attr, v: &u32| {
                lt::cublasLtMatmulPreferenceSetAttribute(pref, attr, v as *const u32 as *const std::ffi::c_void, 4) == ok
            };
            let set_ok = lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const usize as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            ) == ok
                && set_u32(P::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align);
            if !set_ok
                || lt::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    desc,
                    adesc,
                    bdesc,
                    cdesc,
                    cdesc,
                    pref,
                    HEURISTIC_CANDIDATES as i32,
                    heur.as_mut_ptr(),
                    &mut returned,
                ) != ok
            {
                returned = 0;
            }
            lt::cublasLtMatmulPreferenceDestroy(pref);
        }
    }
    let mut cands: Vec<Candidate> = heur[..returned.clamp(0, HEURISTIC_CANDIDATES as i32) as usize]
        .iter()
        .filter(|r| r.state == ok && r.workspaceSize <= ws_bytes)
        .map(|r| Candidate { algo: r.algo, ws: r.workspaceSize, from_heuristic: true })
        .collect();
    // The enumeration's `AlgoCheck` cannot be told the pointers' alignment and
    // assumes the maximum, so it only joins when the operands have it.
    if align >= 256 {
        // SAFETY: handle and descriptors are live.
        cands.extend(unsafe {
            enumerate_configs(
                handle,
                desc,
                adesc,
                bdesc,
                cdesc,
                ws_bytes,
                f32t,
                f32t,
                lt::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            )
        });
    }
    if cands.is_empty() {
        // SAFETY: live, about to be abandoned.
        unsafe { destroy_handles(desc, adesc, bdesc, cdesc) };
        return None;
    }

    let mut chosen = 0usize;
    let out_bytes = (ldc as usize) * (n as usize) * 4;
    if cands.len() > 1 && out_bytes <= TUNE_SCRATCH_CAP_BYTES {
        // SAFETY: the caller's live operands for this shape.
        let picked = unsafe {
            tune_two_rounds(handle, desc, adesc, bdesc, cdesc, a as *const _, b as *const _, out_bytes, &cands)
        };
        if let Some((i, ms, h0)) = picked {
            chosen = i;
            TF32_TUNED.fetch_add(1, Ordering::Relaxed);
            log_tune("tf32-lt", opa, opb, m, n, k, &cands[i], ms, h0, cands.len());
        }
    }
    let c = &cands[chosen];
    Some(Plan { desc, adesc, bdesc, cdesc, algo: c.algo, ws_bytes: c.ws })
}

static VERBOSE: OnceLock<bool> = OnceLock::new();

/// `NSL_MATMUL_BF16_LT_VERBOSE=1`: one line per tuned shape (both paths).
fn verbose_from_env() -> bool {
    std::env::var("NSL_MATMUL_BF16_LT_VERBOSE").ok().as_deref() == Some("1")
}

/// Time `cands` on the live operands and return the winner's index: one
/// warm-up and one timed launch each, then the [`TUNE_FINALISTS`] fastest again
/// with the median of five. `None` when the scratch cannot be allocated or no
/// candidate launches (the caller then takes the heuristic's first).
///
/// # Safety
/// As [`time_candidates`]; `out_bytes` is C's footprint.
#[allow(clippy::too_many_arguments)]
unsafe fn tune_two_rounds(
    handle: lt::cublasLtHandle_t,
    desc: lt::cublasLtMatmulDesc_t,
    adesc: lt::cublasLtMatrixLayout_t,
    bdesc: lt::cublasLtMatrixLayout_t,
    cdesc: lt::cublasLtMatrixLayout_t,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    out_bytes: usize,
    cands: &[Candidate],
) -> Option<(usize, f32, Option<f32>)> {
    // Fallible on purpose (see `raw_device_alloc`): a nearly full card gets the
    // heuristic's first candidate untimed, never an OOM exit.
    let scratch = raw_device_alloc(out_bytes) as *mut std::ffi::c_void;
    if scratch.is_null() {
        return None;
    }
    let ws = workspace();
    let all: Vec<&Candidate> = cands.iter().collect();
    // SAFETY: as the function's contract; scratch sized to C's footprint.
    let first = unsafe { time_candidates(handle, desc, adesc, bdesc, cdesc, a, b, scratch, ws, &all, 1) };
    let mut order: Vec<usize> = (0..cands.len()).filter(|&i| first[i].is_finite()).collect();
    order.sort_by(|&x, &y| first[x].total_cmp(&first[y]));
    order.truncate(TUNE_FINALISTS);
    let result = if order.is_empty() {
        None
    } else {
        let finalists: Vec<&Candidate> = order.iter().map(|&i| &cands[i]).collect();
        // SAFETY: as above.
        let second = unsafe { time_candidates(handle, desc, adesc, bdesc, cdesc, a, b, scratch, ws, &finalists, 5) };
        let best = (0..order.len()).min_by(|&x, &y| second[x].total_cmp(&second[y])).unwrap_or(0);
        let h0 = cands.iter().position(|c| c.from_heuristic).map(|i| first[i]);
        second[best].is_finite().then_some((order[best], second[best], h0))
    };
    // `time_candidates` synchronized on its last event after every launch, so
    // nothing still writes the scratch.
    raw_device_free(scratch as u64);
    result
}

/// One `[*-lt]` verbose line for a tuned shape.
#[allow(clippy::too_many_arguments)]
fn log_tune(tag: &str, opa: i32, opb: i32, m: i32, n: i32, k: i32, w: &Candidate, ms: f32, h0: Option<f32>, total: usize) {
    if !*VERBOSE.get_or_init(verbose_from_env) {
        return;
    }
    let cfg = |attr| {
        let mut v = 0i32;
        let mut written = 0usize;
        // SAFETY: an i32 attribute read into a live local.
        unsafe {
            lt::cublasLtMatmulAlgoConfigGetAttribute(&w.algo, attr, &mut v as *mut i32 as *mut std::ffi::c_void, 4, &mut written)
        };
        v
    };
    use lt::cublasLtMatmulAlgoConfigAttributes_t as A;
    crate::nsl_log!(
        INFO,
        "lt-tune",
        "[{tag}] tune opa={opa} opb={opb} m={m} n={n} k={k}: winner {ms:.3} ms \
         (algo {} tile {} stages {}{}), heuristic[0] {}, {total} candidates",
        cfg(A::CUBLASLT_ALGO_CONFIG_ID),
        cfg(A::CUBLASLT_ALGO_CONFIG_TILE_ID),
        cfg(A::CUBLASLT_ALGO_CONFIG_STAGES_ID),
        if w.from_heuristic { ", heuristic" } else { "" },
        h0.map_or("none".to_string(), |t| format!("{t:.3} ms")),
    );
}

/// Issue one TF32 GEMM (f32 operands and output, cuBLAS column-major
/// arguments exactly as `cublasSgemm_v2` takes them) through cublasLt with the
/// shape's tuned plan. `true` = launched (the caller must not also issue it);
/// `false` = declined, nothing was written to `c`.
///
/// The caller has checked [`timed_lt_applies`].
///
/// # Safety
/// `a`, `b`, `c` are live device buffers for this shape and layout.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_f32_tf32(
    opa: bool,
    opb: bool,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) -> bool {
    let Some(handle) = lt_handle() else {
        TF32_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return false;
    };
    // Alignment class of the three buffers, capped at 256 bytes: plans are
    // built for it, so a sliced operand at an odd offset gets its own plan
    // instead of failing (and poisoning) the aligned shape's.
    let addr_bits = (a as usize) | (b as usize) | (c as usize) | 256;
    let key: Tf32Key = (opa as i32, opb as i32, m, n, k, lda, ldb, ldc, addr_bits.trailing_zeros());
    let hit = with_tf32_plans(|plans| {
        plans.get(&key).map(|p| p.as_ref().map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes)))
    });
    let (desc, adesc, bdesc, cdesc, algo, ws_need) = match hit {
        Some(Some(t)) => t,
        Some(None) => {
            TF32_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        None => {
            if !tune_budget_allows(shapes_tuned()) {
                print_budget_once();
                return false;
            }
            // Built outside the cache lock: it synchronizes the device.
            // SAFETY: the caller's live operands for this shape.
            let plan = unsafe { tf32_build_plan(handle, &key, a, b) };
            let resolved = with_tf32_plans(|plans| match plans.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    if let Some(p) = plan {
                        // SAFETY: the newcomer's handles were never published.
                        unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
                    }
                    e.get().as_ref().map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes))
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    let t = plan.as_ref().map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes));
                    v.insert(plan);
                    t
                }
            });
            match resolved {
                Some(t) => t,
                None => {
                    TF32_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
            }
        }
    };
    let (ws_ptr, _) = workspace();
    let stream = super::inner::current_stream() as lt::cudaStream_t;
    // SAFETY: plan handles are process-lifetime; operands are the caller's
    // live buffers; C doubles as D (beta may be 1: the wgrad accumulate).
    let st = unsafe {
        lt::cublasLtMatmul(
            handle,
            desc,
            &alpha as *const f32 as *const std::ffi::c_void,
            a as *const std::ffi::c_void,
            adesc,
            b as *const std::ffi::c_void,
            bdesc,
            &beta as *const f32 as *const std::ffi::c_void,
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
        // A failed launch wrote nothing; poison the key so it does not
        // fail-retry per GEMM, and let the caller issue cublasSgemm.
        let displaced = with_tf32_plans(|plans| plans.insert(key, None));
        if let Some(Some(p)) = displaced {
            // SAFETY: just unpublished; no launch in flight uses them.
            unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
        }
        TF32_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        print_fallback_once("tf32 cublasLtMatmul launch failure");
        return false;
    }
    TF32_ISSUED.fetch_add(1, Ordering::Relaxed);
    true
}

/// Test-only reset: destroy every plan, clear counters. The Lt handle and
/// workspace survive (process-lifetime by design; captured graphs and the
/// banner's once-semantics depend on them).
// Gated on `test-hooks` ALONE, matching the two callers in `cuda/mod.rs`
// (`#[cfg(all(feature = "cuda", feature = "test-hooks"))]`). It used to read
// `any(test, feature = "test-hooks")`, which is a WIDER condition than its
// use: `cargo test --features cuda` (no test-hooks) compiled the function --
// `test` is on for a lib-test build -- but not the callers, so #588's
// `dead_code = deny` failed the "CUDA feature" CI job on main. The module is
// already `#[cfg(feature = "cuda")]`, so that axis needs no repeating here.
#[cfg(feature = "test-hooks")]
pub(crate) fn reset_for_test() {
    // No context, no plans — and reaching for one here would force CUDA
    // initialisation from a reset hook, which aborts where there is no driver.
    if !super::context::initialized() {
        TF32_ISSUED.store(0, Ordering::Relaxed);
        TF32_TUNED.store(0, Ordering::Relaxed);
        TF32_FALLBACKS.store(0, Ordering::Relaxed);
        ISSUED.store(0, Ordering::Relaxed);
        TUNED.store(0, Ordering::Relaxed);
        FALLBACKS.store(0, Ordering::Relaxed);
        MX_ISSUED.store(0, Ordering::Relaxed);
        MX_TUNED.store(0, Ordering::Relaxed);
        MX_FALLBACKS.store(0, Ordering::Relaxed);
        return;
    }
    // Drain inside the cache, destroy outside it: `destroy_handles` is a
    // driver call, and `with_cache` holds the mutex every device cache shares.
    let drained: Vec<Option<Plan>> =
        with_plans(|plans| std::mem::take(plans).into_values().collect());
    for p in drained.into_iter().flatten() {
        // SAFETY: handles were live and are now unreachable from the map.
        unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
    }
    let drained: Vec<Option<Plan>> =
        with_tf32_plans(|plans| std::mem::take(plans).into_values().collect());
    for p in drained.into_iter().flatten() {
        // SAFETY: as above.
        unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
    }
    let drained: Vec<Option<Plan>> =
        with_mx_plans(|plans| std::mem::take(plans).into_values().collect());
    for p in drained.into_iter().flatten() {
        // SAFETY: as above.
        unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
    }
    TF32_ISSUED.store(0, Ordering::Relaxed);
    TF32_TUNED.store(0, Ordering::Relaxed);
    TF32_FALLBACKS.store(0, Ordering::Relaxed);
    ISSUED.store(0, Ordering::Relaxed);
    TUNED.store(0, Ordering::Relaxed);
    FALLBACKS.store(0, Ordering::Relaxed);
    MX_ISSUED.store(0, Ordering::Relaxed);
    MX_TUNED.store(0, Ordering::Relaxed);
    MX_FALLBACKS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// MXFP8 GEMMs: E4M3 operands with one UE8M0 scale per 32 elements along K
// (`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`), f32 accumulation.
//
// cublasLt takes FP8 only in the "TN" form on sm_120: both operands K-major.
// In column-major terms `D[m, n] = alpha · A^T B + beta · C`, where A is
// stored as `m` rows of `k` bytes (lda = k) and B as `n` rows of `k` bytes
// (ldb = k): exactly the `[outer, k]` row-major images
// `nsl_kir::kernels::mx_quant` writes, with their scales in cublasLt's tiled
// layout. C/D are bf16 or f32 (`ldc` elements apart).
//
// Plans are keyed by `(m, n, k, ldc, c_bf16, alignment class)` in their own
// map. The scale pointers are not part of a plan: they are set on the cached
// descriptor before each launch, which reads them at enqueue time (all GPU
// work is single-threaded on one stream, as above).
// ---------------------------------------------------------------------------

/// `(m, n, k, ldc, c_bf16, log2 of the pointer alignment class)`.
type MxKey = (i32, i32, i32, i32, bool, u32);

#[derive(Default)]
struct MxPlanCache(HashMap<MxKey, Option<Plan>>);

fn with_mx_plans<R>(f: impl FnOnce(&mut HashMap<MxKey, Option<Plan>>) -> R) -> R {
    super::context::current().with_cache(|c: &mut MxPlanCache| f(&mut c.0))
}

static MX_ISSUED: AtomicU64 = AtomicU64::new(0);
static MX_TUNED: AtomicU64 = AtomicU64::new(0);
static MX_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// (issued, tuned shapes, declined) MXFP8 GEMMs.
pub(crate) fn mx_stats() -> (u64, u64, u64) {
    (
        MX_ISSUED.load(Ordering::Relaxed),
        MX_TUNED.load(Ordering::Relaxed),
        MX_FALLBACKS.load(Ordering::Relaxed),
    )
}

/// Point the descriptor's A/B block-scale attributes at `a_scales` /
/// `b_scales`. `false` when cublasLt refuses either.
unsafe fn mx_set_scale_pointers(
    desc: lt::cublasLtMatmulDesc_t,
    a_scales: *const std::ffi::c_void,
    b_scales: *const std::ffi::c_void,
) -> bool {
    use lt::cublasLtMatmulDescAttributes_t as D;
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let set = |attr, p: &*const std::ffi::c_void| {
        // SAFETY: the attribute holds a device pointer, passed by address.
        unsafe {
            lt::cublasLtMatmulDescSetAttribute(
                desc,
                attr,
                p as *const *const std::ffi::c_void as *const std::ffi::c_void,
                std::mem::size_of::<*const std::ffi::c_void>(),
            ) == ok
        }
    };
    set(D::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scales)
        && set(D::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scales)
}

/// Build the TN MXFP8 descriptor set for one shape, query the heuristic, and
/// (when tuning is on) time the candidates on the live operands.
#[allow(clippy::too_many_arguments)]
unsafe fn build_mx_plan(
    handle: lt::cublasLtHandle_t,
    m: i32,
    n: i32,
    k: i32,
    a: *const std::ffi::c_void,
    a_scales: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    b_scales: *const std::ffi::c_void,
    ldc: i32,
    align_log2: u32,
    c_bf16: bool,
) -> Option<Plan> {
    use lt::cublasLtMatmulDescAttributes_t as D;
    let ok = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
    let (_, ws_bytes) = workspace();
    let c_type = if c_bf16 {
        lt::cudaDataType_t::CUDA_R_16BF
    } else {
        lt::cudaDataType_t::CUDA_R_32F
    };
    let e4m3 = lt::cudaDataType_t::CUDA_R_8F_E4M3;
    let align = 1u32 << align_log2;
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
            let set_i32 = |attr, v: i32| {
                lt::cublasLtMatmulDescSetAttribute(
                    desc,
                    attr,
                    &v as *const i32 as *const std::ffi::c_void,
                    4,
                ) == ok
            };
            let vec32 =
                lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 as i32;
            if !set_i32(D::CUBLASLT_MATMUL_DESC_TRANSA, 1)
                || !set_i32(D::CUBLASLT_MATMUL_DESC_TRANSB, 0)
                || !set_i32(D::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, vec32)
                || !set_i32(D::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, vec32)
                || !mx_set_scale_pointers(desc, a_scales, b_scales)
            {
                break 'build false;
            }
            if lt::cublasLtMatrixLayoutCreate(&mut adesc, e4m3, k as u64, m as u64, i64::from(k))
                != ok
                || lt::cublasLtMatrixLayoutCreate(
                    &mut bdesc,
                    e4m3,
                    k as u64,
                    n as u64,
                    i64::from(k),
                ) != ok
                || lt::cublasLtMatrixLayoutCreate(
                    &mut cdesc,
                    c_type,
                    m as u64,
                    n as u64,
                    i64::from(ldc),
                ) != ok
            {
                break 'build false;
            }
            true
        }
    };
    if !built {
        // SAFETY: handles are exactly as left above.
        unsafe { destroy_handles(desc, adesc, bdesc, cdesc) };
        return None;
    }
    let mut results: [lt::cublasLtMatmulHeuristicResult_t; HEURISTIC_CANDIDATES] =
        // SAFETY: plain-old-data out-array, read only up to `returned`.
        unsafe { std::mem::zeroed() };
    let mut returned: i32 = 0;
    // SAFETY: the preference is created, set, used for one query, destroyed.
    let query_ok = unsafe {
        let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
        if lt::cublasLtMatmulPreferenceCreate(&mut pref) != ok {
            false
        } else {
            use lt::cublasLtMatmulPreferenceAttributes_t as P;
            let red_mask: u32 =
                lt::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
            let set_u32 = |attr, v: &u32| {
                lt::cublasLtMatmulPreferenceSetAttribute(
                    pref,
                    attr,
                    v as *const u32 as *const std::ffi::c_void,
                    4,
                ) == ok
            };
            let got = lt::cublasLtMatmulPreferenceSetAttribute(
                pref,
                P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes as *const usize as *const std::ffi::c_void,
                std::mem::size_of::<usize>(),
            ) == ok
                && set_u32(P::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red_mask)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align)
                && set_u32(P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align)
                && lt::cublasLtMatmulAlgoGetHeuristic(
                    handle,
                    desc,
                    adesc,
                    bdesc,
                    cdesc,
                    cdesc,
                    pref,
                    HEURISTIC_CANDIDATES as i32,
                    results.as_mut_ptr(),
                    &mut returned,
                ) == ok;
            lt::cublasLtMatmulPreferenceDestroy(pref);
            got
        }
    };
    let candidates: Vec<Candidate> = if query_ok {
        results[..returned.max(0) as usize]
            .iter()
            .filter(|r| r.state == ok && r.workspaceSize <= ws_bytes)
            .map(|r| Candidate {
                algo: r.algo,
                ws: r.workspaceSize,
                from_heuristic: true,
            })
            .collect()
    } else {
        Vec::new()
    };
    if candidates.is_empty() {
        // SAFETY: live handles, abandoned.
        unsafe { destroy_handles(desc, adesc, bdesc, cdesc) };
        return None;
    }
    // Timed selection as the bf16 path: scratch D, beta = 0, never the
    // caller's C; off under --deterministic, graph capture, or past the budget.
    let out_bytes = (ldc as usize) * (n as usize) * 4;
    let tune = tune_enabled()
        && !super::graph_capture::enabled()
        && !crate::deterministic_ops::is_deterministic()
        && out_bytes <= TUNE_SCRATCH_CAP_BYTES
        && tune_budget_allows(shapes_tuned());
    let mut chosen = 0;
    if tune && candidates.len() > 1 {
        // SAFETY: the caller's live operands; the descriptor holds their scales.
        if let Some((i, ms, h0)) = unsafe {
            tune_two_rounds(
                handle,
                desc,
                adesc,
                bdesc,
                cdesc,
                a,
                b,
                out_bytes,
                &candidates,
            )
        } {
            chosen = i;
            MX_TUNED.fetch_add(1, Ordering::Relaxed);
            log_tune(
                "mxfp8-lt",
                1,
                0,
                m,
                n,
                k,
                &candidates[i],
                ms,
                h0,
                candidates.len(),
            );
        }
    }
    Some(Plan {
        desc,
        adesc,
        bdesc,
        cdesc,
        algo: candidates[chosen].algo,
        ws_bytes: candidates[chosen].ws,
    })
}

/// Issue `D = alpha · A^T B + beta · C` (column-major, `[m, n]`, `ldc`) with
/// MXFP8 operands: `a` is `m` rows of `k` E4M3 bytes, `b` is `n` rows of
/// `k`, each with its VEC32_UE8M0 scales in cublasLt's tiled layout. `c`
/// (bf16 or f32) is read when `beta != 0`; `d` has its layout (`d == c`
/// writes in place). Returns `false` (nothing written) when cublasLt
/// declines the shape; the caller must then compute the product another way.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_mxfp8(
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const std::ffi::c_void,
    a_scales: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    b_scales: *const std::ffi::c_void,
    beta: f32,
    c: *const std::ffi::c_void,
    d: *mut std::ffi::c_void,
    ldc: i32,
    c_bf16: bool,
) -> bool {
    let decline = |why: &str| {
        MX_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        print_fallback_once(why);
        false
    };
    if m <= 0 || n <= 0 || k <= 0 || k % 32 != 0 || ldc < m {
        return decline("mxfp8: a shape outside the block-scaled contract");
    }
    // cublasLt wants 16-byte operands and scale tensors. A misaligned buffer
    // would fail at launch and poison the shape's plan for aligned callers.
    if !((a as usize)
        | (b as usize)
        | (c as usize)
        | (d as usize)
        | (a_scales as usize)
        | (b_scales as usize))
        .is_multiple_of(16)
    {
        return decline("mxfp8: an operand or scale buffer not 16-byte aligned");
    }
    let Some(handle) = lt_handle() else {
        MX_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return false;
    };
    let align_log2 =
        ((a as usize) | (b as usize) | (c as usize) | (d as usize) | 256).trailing_zeros();
    let key: MxKey = (m, n, k, ldc, c_bf16, align_log2);
    let hit = with_mx_plans(|plans| {
        plans.get(&key).map(|p| {
            p.as_ref()
                .map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes))
        })
    });
    let (desc, adesc, bdesc, cdesc, algo, ws_need) = match hit {
        Some(Some(t)) => t,
        Some(None) => {
            MX_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        None => {
            // SAFETY: forwarded pointers are the caller's live operands.
            let plan = unsafe {
                build_mx_plan(
                    handle, m, n, k, a, a_scales, b, b_scales, ldc, align_log2, c_bf16,
                )
            };
            let tuple = plan
                .as_ref()
                .map(|p| (p.desc, p.adesc, p.bdesc, p.cdesc, p.algo, p.ws_bytes));
            with_mx_plans(|plans| {
                if let Some(old) = plans.insert(key, plan).flatten() {
                    // SAFETY: displaced handles, never used again.
                    unsafe { destroy_handles(old.desc, old.adesc, old.bdesc, old.cdesc) };
                }
            });
            match tuple {
                Some(t) => t,
                None => return decline("mxfp8: no heuristic candidate for a shape"),
            }
        }
    };
    // SAFETY: the cached descriptor; this call's scales.
    if !unsafe { mx_set_scale_pointers(desc, a_scales, b_scales) } {
        return decline("mxfp8: scale pointer attribute refused");
    }
    let (ws_ptr, _) = workspace();
    let stream = super::inner::current_stream() as lt::cudaStream_t;
    // SAFETY: process-lifetime plan handles; the caller's live device buffers.
    let st = unsafe {
        lt::cublasLtMatmul(
            handle,
            desc,
            &alpha as *const f32 as *const std::ffi::c_void,
            a,
            adesc,
            b,
            bdesc,
            &beta as *const f32 as *const std::ffi::c_void,
            c,
            cdesc,
            d,
            cdesc,
            &algo,
            ws_ptr as *mut std::ffi::c_void,
            ws_need,
            stream,
        )
    };
    if st != lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        let displaced = with_mx_plans(|plans| plans.insert(key, None));
        if let Some(Some(p)) = displaced {
            // SAFETY: just removed; no launch in flight with these handles.
            unsafe { destroy_handles(p.desc, p.adesc, p.bdesc, p.cdesc) };
        }
        return decline("mxfp8: cublasLtMatmul launch failure");
    }
    MX_ISSUED.fetch_add(1, Ordering::Relaxed);
    true
}

#[cfg(test)]
mod tune_budget_tests {
    use super::*;

    #[test]
    fn the_budget_admits_exactly_its_size() {
        assert!(tune_budget_allows(0));
        assert!(tune_budget_allows(TUNE_BUDGET_SHAPES - 1));
        assert!(!tune_budget_allows(TUNE_BUDGET_SHAPES));
        assert!(!tune_budget_allows(TUNE_BUDGET_SHAPES + 7));
    }
}
