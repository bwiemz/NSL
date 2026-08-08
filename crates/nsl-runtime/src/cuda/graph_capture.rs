//! P5 item 19 — opportunistic per-region CUDA graph capture/replay.
//!
//! A REGION is one contiguous Wengert lowering (a forward CCR slice, a CSLA
//! backward layer range, or a monolithic forward/backward) bracketed by
//! `nsl_cuda_graph_region_begin/end(id)` calls that codegen emits inside
//! `compile_wengert_ops_range` under `--cuda-graphs`. Weight-stream
//! transfers, optimizer updates (their per-step bias-correction scalars would
//! never digest-stabilize) and everything else stay OUTSIDE regions.
//!
//! Per-region state machine, self-healing at every edge:
//!
//! 1. RECORD (>=2 steps): launches run eagerly while the full pseudo-op
//!    sequence is recorded — kernel (function handle, grid/block/smem, raw
//!    argument bytes via `cuFuncGetParamInfo`), row-major sgemm, memset.
//!    Two consecutive steps with identical sequences (same functions, same
//!    pointers, same scalars) prove the region is static; a readback or
//!    transfer inside the region taints it → permanently eager.
//! 2. CAPTURE (1 step): `cuStreamBeginCapture_v2` (relaxed) on the per-thread
//!    compute stream; the same host code issues the same calls, which the
//!    driver records as graph nodes instead of executing; at region end the
//!    graph is instantiated AND launched (the step still executes). The
//!    driver is the audit backstop here: any un-hooked stream interaction a
//!    raw call site performs invalidates the capture loudly — never silently.
//! 3. REPLAY (steady state): each issued pseudo-op is verified against the
//!    captured sequence and SKIPPED; at region end one `cuGraphLaunch`
//!    replays the whole region. Any mismatch eager-repairs (re-issues the
//!    matched-and-skipped prefix from the stored records — bit-identical by
//!    construction) and drops back to RECORD (or EAGER after
//!    `MAX_CAPTURE_ATTEMPTS` breaks).
//!
//! Deferred frees issued inside a region are queued host-side and their
//! completion events recorded only after the region's work is actually on
//! the stream (graph launch or eager finish) — recording them mid-replay
//! would complete before the graph ran and free live memory.
//!
//! CORRECTNESS INVARIANT (why skip-then-graph-launch is sound): host code
//! runs identically in every mode — allocations, frees and list bookkeeping
//! all happen eagerly; only GPU-op ISSUANCE is diverted. A verified-identical
//! pseudo-op sequence therefore means the graph writes exactly the buffers
//! this step's tensors point at, in the captured order, on the same stream.

#![allow(clippy::missing_safety_doc)]

#[cfg(feature = "cuda")]
mod imp {
    use cudarc::driver::sys::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    /// Two identical record steps required before a capture attempt.
    const STABLE_STREAK: u32 = 2;
    /// After this many failed capture/replay cycles a region goes eager for
    /// good (avoids capture-thrash on genuinely unstable regions).
    const MAX_CAPTURE_ATTEMPTS: u32 = 2;
    /// Regions longer than this never capture (runaway-memory guard).
    const MAX_REGION_OPS: usize = 65536;
    /// Sanity cap on a single kernel argument's size in bytes.
    const MAX_PARAM_BYTES: usize = 512;

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static LOGGING: AtomicBool = AtomicBool::new(false);
    /// Gradient-accumulation window (compile-time known). Each micro-batch
    /// phase within a window leaves the allocator in its own (per-phase
    /// self-consistent) state, so region digests are periodic with this
    /// period rather than constant — the state machine runs per
    /// (region, occurrence % window) and captures one graph per phase.
    static WINDOW: AtomicU64 = AtomicU64::new(1);

    static CAPTURES: AtomicU64 = AtomicU64::new(0);
    static REPLAYS: AtomicU64 = AtomicU64::new(0);
    static TAINTS: AtomicU64 = AtomicU64::new(0);
    static MISMATCHES: AtomicU64 = AtomicU64::new(0);
    static REPAIRED_OPS: AtomicU64 = AtomicU64::new(0);
    static EAGER_REGIONS: AtomicU64 = AtomicU64::new(0);
    static REGIONS_SEEN: AtomicU64 = AtomicU64::new(0);

    pub fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed)
    }

    fn log_on() -> bool {
        LOGGING.load(Ordering::Relaxed)
    }

    /// A pinned host staging buffer owned by a captured HtoD pseudo-op. The
    /// graph's memcpy node reads from THIS buffer at every launch, so the
    /// upload payload can be refreshed per step (host memcpy into a stable
    /// address) without the graph ever touching a caller-transient pointer.
    struct StagingBuf {
        ptr: usize,
        len: usize,
    }

    impl StagingBuf {
        fn alloc(len: usize) -> Option<std::sync::Arc<StagingBuf>> {
            let mut p: *mut c_void = std::ptr::null_mut();
            let r = unsafe { cuMemAllocHost_v2(&mut p, len.max(1)) };
            if r != CUresult::CUDA_SUCCESS || p.is_null() {
                return None;
            }
            Some(std::sync::Arc::new(StagingBuf { ptr: p as usize, len }))
        }

        fn fill_from(&self, src: *const c_void, len: usize) {
            debug_assert_eq!(len, self.len);
            unsafe {
                std::ptr::copy_nonoverlapping(src as *const u8, self.ptr as *mut u8, len);
            }
        }
    }

    impl Drop for StagingBuf {
        fn drop(&mut self) {
            unsafe { cuMemFreeHost(self.ptr as *mut c_void) };
        }
    }

    /// Which cuBLAS wrapper a recorded `GpuOp::Sgemm` came from.
    ///
    /// Replay must call the same one: the wrappers differ in operand order,
    /// transposition and leading dimensions, so reconstructing a weight-grad
    /// contraction through the row-major wrapper computes a different (and
    /// silently plausible) product.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum SgemmKind {
        /// `sgemm_row_major` / `sgemm_batched_row_major` — row-major operands
        /// with the `C^T = B^T A^T` swap.
        RowMajor,
        /// `sgemm_wgrad_accum` — `C[d,o] = alpha * X[N,d]^T @ G[N,o] + beta*C`.
        WgradAccum,
    }

    /// How the wrapper stored the operands it handed cuBLAS.
    ///
    /// `Bf16Storage` means the wrapper cast both operands f32 -> bf16 into
    /// scratch and ran `cublasGemmEx(16BF -> 32F)`; the two cast launches are
    /// ordinary hooked kernels recorded immediately BEFORE this pseudo-op,
    /// and the op's `a`/`b` remain the ORIGINAL f32 pointers (the scratch
    /// addresses are pinned by the cast ops' recorded argument bytes, so
    /// scratch stability is still verified every step). `F32` covers every
    /// f32-storage form — the handle's math mode and the explicit `FAST_TF32`
    /// low-intensity arm alike, which are the same buffers and the same
    /// numerics class.
    ///
    /// Identity-only: eager repair re-enters the recording wrapper, which
    /// re-derives the storage route deterministically (the matmul mode is
    /// resolved once per process and `bf16_storage_worthwhile` is a pure
    /// function of the recorded dims). The field exists so equality and the
    /// digest cannot conflate the two forms — the item-9 "three places must
    /// agree about op identity" lesson, applied at record time.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum GemmPrecision {
        F32,
        Bf16Storage,
    }

    /// One recorded GPU interaction. Everything needed both to compare a
    /// later step's issuance for equality and to eagerly re-issue the op
    /// during self-repair.
    #[derive(Clone)]
    enum GpuOp {
        Kernel {
            func: usize,
            grid: [u32; 3],
            block: [u32; 3],
            shared: u32,
            /// Concatenated raw argument bytes, in declaration order.
            params: Vec<u8>,
            /// Byte offset of each argument inside `params`.
            offsets: Vec<usize>,
            /// Diagnostic only (NSL_CUDA_GRAPH_LOG=1); excluded from
            /// equality and digests.
            name: String,
        },
        Sgemm {
            /// WHICH cuBLAS wrapper produced this, and therefore which one
            /// must reproduce it on replay.
            ///
            /// Not cosmetic. `sgemm_wgrad_accum` and the plain row-major path
            /// (`sgemm_row_major_t` with both ops `N`) both recorded
            /// `(a, b, c, m, n, k, alpha, beta)` through one shared hook, but
            /// they hand cuBLAS completely different operand mappings —
            /// `OP_N/OP_N` with the row-major swap versus `OP_N/OP_T` for the
            /// weight-gradient contraction. Without this field a wgrad gemm
            /// and a plain gemm over the same three buffers with the same
            /// three dims compare EQUAL, and eager repair replays either one
            /// as `sgemm_row_major`. That was true before item 9 and is why
            /// this field exists now: OP_T on the general matmul path makes
            /// the collision reachable from every model instead of only from
            /// the weight-gradient fusion.
            kind: SgemmKind,
            /// Operand storage route (f32 vs bf16-cast scratch). See the
            /// enum's docs: identity-only, repair re-derives it.
            precision: GemmPrecision,
            /// Operand transposition, for the row-major wrapper. `lda`/`ldb`
            /// are derived from these plus (m, n, k) inside the wrapper, so
            /// recording the flags pins the leading dimensions too.
            transa: bool,
            transb: bool,
            a: usize,
            b: usize,
            c: usize,
            m: u64,
            n: u64,
            k: u64,
            /// `alpha`/`beta` as raw f32 BIT PATTERNS.
            ///
            /// Stored as bits, not f32, so this stays usable in the exact
            /// equality comparison below — and, more importantly, so a FUSED
            /// accumulating gemm (`beta = 1.0`, writing into an existing
            /// buffer) can never compare equal to, or be replayed as, an
            /// overwriting one (`beta = 0.0`). Replaying the wrong beta would
            /// silently double-count or drop a gradient contribution.
            alpha_bits: u32,
            beta_bits: u32,
            /// Number of matrix slices; 1 for a plain sgemm.
            ///
            /// Item 9: batched products used to reach the graph as a
            /// `GpuOp::Kernel` — the naive `nsl_bmm_f32` launch, whose params
            /// blob carried the batch count and strides and so distinguished
            /// them automatically. Routing them to cuBLAS makes them
            /// pseudo-ops instead, so the batch geometry has to become part of
            /// THIS op's identity. Without it two products differing only in
            /// batch count or stride compare equal, and a replay reuses a
            /// captured graph built for the wrong shape.
            batch: u64,
            /// Element offsets between consecutive slices. 0 broadcasts that
            /// operand across the whole batch.
            stride_a: u64,
            stride_b: u64,
            stride_c: u64,
        },
        Memset {
            dst: usize,
            bytes: usize,
        },
        /// Host->device upload. Identity is (dst, len) ONLY — the payload is
        /// allowed to change per step (token-id batches) and flows through
        /// the owned staging buffer that the captured memcpy node reads.
        HtoD {
            dst: usize,
            len: usize,
            /// Present once captured (None during plain recording).
            staging: Option<std::sync::Arc<StagingBuf>>,
        },
        /// Device->device copy (pure stream-ordered, no host involvement).
        DtoD {
            dst: usize,
            src: usize,
            len: usize,
        },
    }

    impl PartialEq for GpuOp {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (
                    GpuOp::Kernel { func, grid, block, shared, params, offsets, .. },
                    GpuOp::Kernel {
                        func: f2,
                        grid: g2,
                        block: b2,
                        shared: s2,
                        params: p2,
                        offsets: o2,
                        ..
                    },
                ) => func == f2 && grid == g2 && block == b2 && shared == s2 && params == p2 && offsets == o2,
                (
                    GpuOp::Sgemm {
                        kind, precision, transa, transb,
                        a, b, c, m, n, k, alpha_bits, beta_bits,
                        batch, stride_a, stride_b, stride_c,
                    },
                    GpuOp::Sgemm {
                        kind: kd2, precision: pr2, transa: ta2, transb: tb2,
                        a: a2, b: b2, c: c2, m: m2, n: n2, k: k2,
                        alpha_bits: al2, beta_bits: be2,
                        batch: bt2, stride_a: sa2, stride_b: sb2, stride_c: sc2,
                    },
                ) => kind == kd2 && precision == pr2 && transa == ta2 && transb == tb2
                    && a == a2 && b == b2 && c == c2 && m == m2 && n == n2 && k == k2
                    && alpha_bits == al2 && beta_bits == be2
                    && batch == bt2 && stride_a == sa2 && stride_b == sb2 && stride_c == sc2,
                (GpuOp::Memset { dst, bytes }, GpuOp::Memset { dst: d2, bytes: n2 }) => {
                    dst == d2 && bytes == n2
                }
                // Payload deliberately excluded: uploads carry fresh data
                // through the staging buffer each step.
                (GpuOp::HtoD { dst, len, .. }, GpuOp::HtoD { dst: d2, len: l2, .. }) => {
                    dst == d2 && len == l2
                }
                (
                    GpuOp::DtoD { dst, src, len },
                    GpuOp::DtoD { dst: d2, src: s2, len: l2 },
                ) => dst == d2 && src == s2 && len == l2,
                _ => false,
            }
        }
    }

    enum RegionState {
        Record {
            prev_digest: Option<u64>,
            streak: u32,
            attempts: u32,
            /// Retained only under NSL_CUDA_GRAPH_LOG=1, for divergence diffs.
            prev_seq: Option<Vec<GpuOp>>,
        },
        Captured {
            exec: usize, // CUgraphExec
            seq: Vec<GpuOp>,
            attempts: u32,
            /// CUevent recorded after every launch of `exec` (0 = none yet).
            /// Fences two hazards: refreshing an HtoD staging buffer, and
            /// dropping the seq (freeing pinned staging), while a previous
            /// launch of this graph may still be reading them (review H1/H2).
            launch_event: usize,
        },
        Eager,
    }

    enum Mode {
        Recording { tainted: bool },
        Capturing,
        Skipping { idx: usize },
        /// A mid-region mismatch/taint already repaired the prefix; the rest
        /// of the region runs eagerly and `post` decides the next state.
        EagerRest,
    }

    struct Active {
        id: i64,
        /// occurrence % WINDOW at begin — the state-map subkey.
        phase: u64,
        mode: Mode,
        /// Recording/Capturing: the sequence being recorded.
        /// Skipping: the captured sequence being verified against.
        seq: Vec<GpuOp>,
        /// CUgraphExec while Skipping (0 otherwise).
        exec: usize,
        /// The exec's launch-fence event while Skipping (0 otherwise).
        launch_event: usize,
        /// Record-state bookkeeping carried through the pass.
        prev_digest: Option<u64>,
        streak: u32,
        attempts: u32,
        prev_seq: Option<Vec<GpuOp>>,
        /// Explicit next state set by mid-region transitions (taint/mismatch).
        post: Option<RegionState>,
        /// Deferred frees queued during the region; events are recorded at
        /// region end, after the region's work is actually on the stream.
        deferred: Vec<usize>,
    }

    thread_local! {
        static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
        static REGIONS: RefCell<HashMap<(i64, u64), RegionState>> = RefCell::new(HashMap::new());
        /// Per-region-id execution counter (phase = count % WINDOW).
        static OCCURRENCE: RefCell<HashMap<i64, u64>> = RefCell::new(HashMap::new());
        /// Depth of ignored nested begins (should not happen; belt).
        static NESTED_SKIP: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// Per-CUfunction argument sizes queried once via `cuFuncGetParamInfo`
    /// (CUDA 12.4+). `None` = query unsupported/failed → region taints.
    fn param_sizes(func: usize) -> Option<Vec<usize>> {
        static CACHE: OnceLock<Mutex<HashMap<usize, Option<Vec<usize>>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard = cache.lock().unwrap();
        if let Some(v) = guard.get(&func) {
            return v.clone();
        }
        let mut sizes = Vec::new();
        let mut ok = true;
        for i in 0..64usize {
            let mut off = 0usize;
            let mut sz = 0usize;
            let r = unsafe { cuFuncGetParamInfo(func as CUfunction, i, &mut off, &mut sz) };
            if r == CUresult::CUDA_SUCCESS {
                if sz == 0 || sz > MAX_PARAM_BYTES {
                    ok = false;
                    break;
                }
                sizes.push(sz);
            } else {
                // First failure past the last parameter is the normal
                // end-of-list signal; failure at index 0 means the driver
                // does not support the query.
                if i == 0 {
                    ok = false;
                }
                break;
            }
        }
        let entry = if ok && !sizes.is_empty() { Some(sizes) } else { None };
        guard.insert(func, entry.clone());
        entry
    }

    fn digest(seq: &[GpuOp]) -> u64 {
        // FNV-1a over a canonical byte rendering of each op.
        let mut h: u64 = 14695981039346656037;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
        };
        for op in seq {
            match op {
                GpuOp::Kernel { func, grid, block, shared, params, offsets, .. } => {
                    eat(&[1]);
                    eat(&func.to_le_bytes());
                    for d in grid.iter().chain(block.iter()) {
                        eat(&d.to_le_bytes());
                    }
                    eat(&shared.to_le_bytes());
                    eat(&(offsets.len() as u64).to_le_bytes());
                    eat(params);
                }
                GpuOp::Sgemm {
                    kind, precision, transa, transb,
                    a, b, c, m, n, k, batch, stride_a, stride_b, stride_c, ..
                } => {
                    eat(&[2]);
                    // Wrapper identity, storage route and transposition are
                    // digested for the same reason batch geometry is: two
                    // calls that produce different products must not hash
                    // alike.
                    eat(&[
                        match kind {
                            SgemmKind::RowMajor => 0u8,
                            SgemmKind::WgradAccum => 1u8,
                        },
                        match precision {
                            GemmPrecision::F32 => 0u8,
                            GemmPrecision::Bf16Storage => 1u8,
                        },
                        u8::from(*transa),
                        u8::from(*transb),
                    ]);
                    for v in [
                        *a as u64, *b as u64, *c as u64, *m, *n, *k,
                        *batch, *stride_a, *stride_b, *stride_c,
                    ] {
                        eat(&v.to_le_bytes());
                    }
                }
                GpuOp::Memset { dst, bytes } => {
                    eat(&[3]);
                    eat(&(*dst as u64).to_le_bytes());
                    eat(&(*bytes as u64).to_le_bytes());
                }
                GpuOp::HtoD { dst, len, .. } => {
                    // Payload excluded by design (fresh data every step).
                    eat(&[4]);
                    eat(&(*dst as u64).to_le_bytes());
                    eat(&(*len as u64).to_le_bytes());
                }
                GpuOp::DtoD { dst, src, len } => {
                    eat(&[5]);
                    eat(&(*dst as u64).to_le_bytes());
                    eat(&(*src as u64).to_le_bytes());
                    eat(&(*len as u64).to_le_bytes());
                }
            }
        }
        h
    }

    /// Eagerly re-issue recorded ops (bit-identical by construction: same
    /// function handles, same argument bytes, same stream). Must be called
    /// with the ACTIVE borrow RELEASED and the mode already advanced to
    /// EagerRest/none — the gemm wrappers re-enter the `on_sgemm_*` hooks.
    fn repair(ops: &[GpuOp]) {
        if ops.is_empty() {
            return;
        }
        REPAIRED_OPS.fetch_add(ops.len() as u64, Ordering::Relaxed);
        let stream = crate::cuda::inner::current_stream();
        for op in ops {
            match op {
                GpuOp::Kernel { func, grid, block, shared, params, offsets, .. } => {
                    let mut argv: Vec<*mut c_void> = offsets
                        .iter()
                        .map(|&o| unsafe { params.as_ptr().add(o) } as *mut c_void)
                        .collect();
                    let r = unsafe {
                        cuLaunchKernel(
                            *func as CUfunction,
                            grid[0], grid[1], grid[2],
                            block[0], block[1], block[2],
                            *shared,
                            stream,
                            argv.as_mut_ptr(),
                            std::ptr::null_mut(),
                        )
                    };
                    assert_eq!(
                        r,
                        CUresult::CUDA_SUCCESS,
                        "[cuda-graph] eager-repair kernel relaunch failed: {r:?}"
                    );
                }
                GpuOp::Sgemm {
                    kind, transa, transb,
                    a, b, c, m, n, k, alpha_bits, beta_bits,
                    batch, stride_a, stride_b, stride_c,
                    // Not dispatched on: re-entering the wrapper re-derives
                    // the storage route from the same dims and the
                    // once-per-process matmul mode, and `a`/`b` here are the
                    // ORIGINAL f32 pointers. For a `Bf16Storage` op this
                    // double-issues the casts — the repaired prefix already
                    // re-launched the recorded cast kernels into the
                    // recorded scratch, and the wrapper re-entry then casts
                    // again into FRESH scratch for its GemmEx. Harmless
                    // (same sources, deterministic rounding, all
                    // stream-ordered) but the extra alloc/free pair can
                    // perturb the allocator once, costing one extra record
                    // cycle before the region re-captures.
                    precision: _,
                } => {
                    let (alpha, beta) = (f32::from_bits(*alpha_bits), f32::from_bits(*beta_bits));
                    // Dispatch on `kind`, not on shape. Replaying a
                    // weight-gradient contraction through `sgemm_row_major`
                    // would recompute a DIFFERENT product from the same three
                    // buffers — no error, no crash, just a wrong gradient.
                    let r = match (kind, *batch) {
                        (SgemmKind::RowMajor, 1) => unsafe {
                            crate::cuda::cublas_inner::sgemm_row_major_t(
                                *a as *const f32,
                                *b as *const f32,
                                *c as *mut f32,
                                *m, *n, *k, *transa, *transb, alpha, beta,
                            )
                        },
                        (SgemmKind::RowMajor, _) => unsafe {
                            crate::cuda::cublas_inner::sgemm_batched_row_major(
                                *a as *const f32,
                                *b as *const f32,
                                *c as *mut f32,
                                *m, *n, *k,
                                *batch, *stride_a, *stride_b, *stride_c,
                                alpha, beta,
                            )
                        },
                        (SgemmKind::WgradAccum, _) => unsafe {
                            crate::cuda::cublas_inner::sgemm_wgrad_accum(
                                *a as *const f32,
                                *b as *const f32,
                                *c as *mut f32,
                                *m, *n, *k, alpha, beta,
                            )
                        },
                    };
                    assert!(r.is_ok(), "[cuda-graph] eager-repair sgemm failed: {r:?}");
                }
                GpuOp::Memset { dst, bytes } => {
                    let r = unsafe { cuMemsetD8_v2(*dst as CUdeviceptr, 0, *bytes) };
                    assert_eq!(
                        r,
                        CUresult::CUDA_SUCCESS,
                        "[cuda-graph] eager-repair memset failed: {r:?}"
                    );
                }
                GpuOp::HtoD { dst, len, staging } => {
                    // Repairs only replay capture/skip sequences, whose
                    // staging was refreshed with this step's payload at the
                    // moment the op was verified.
                    let staging = staging
                        .as_ref()
                        .expect("[cuda-graph] eager-repair HtoD without staging");
                    let r = unsafe {
                        cuMemcpyHtoD_v2(*dst as CUdeviceptr, staging.ptr as *const c_void, *len)
                    };
                    assert_eq!(
                        r,
                        CUresult::CUDA_SUCCESS,
                        "[cuda-graph] eager-repair HtoD failed: {r:?}"
                    );
                }
                GpuOp::DtoD { dst, src, len } => {
                    let r = unsafe {
                        cuMemcpyDtoD_v2(*dst as CUdeviceptr, *src as CUdeviceptr, *len)
                    };
                    assert_eq!(
                        r,
                        CUresult::CUDA_SUCCESS,
                        "[cuda-graph] eager-repair DtoD failed: {r:?}"
                    );
                }
            }
        }
    }

    /// Diagnostic (NSL_CUDA_GRAPH_LOG=1): describe the first differing op
    /// between two recorded passes of a region.
    fn log_first_divergence(id: i64, prev: &[GpuOp], cur: &[GpuOp]) {
        if prev.len() != cur.len() {
            eprintln!(
                "[cuda-graph] region {id}: op count changed {} -> {}",
                prev.len(),
                cur.len()
            );
            return;
        }
        for (i, (a, b)) in prev.iter().zip(cur.iter()).enumerate() {
            if a == b {
                continue;
            }
            match (a, b) {
                (
                    GpuOp::Kernel { func, params: pa, name, .. },
                    GpuOp::Kernel { func: f2, params: pb, .. },
                ) if func == f2 && pa.len() == pb.len() => {
                    let firstb = pa
                        .iter()
                        .zip(pb.iter())
                        .position(|(x, y)| x != y)
                        .unwrap_or(0);
                    eprintln!(
                        "[cuda-graph] region {id}: first divergence at op {i}: kernel '{name}' {func:#x} param byte {firstb}/{} ({:#x} vs {:#x})",
                        pa.len(),
                        u64::from_le_bytes(pa[firstb & !7..][..8].try_into().unwrap_or([0; 8])),
                        u64::from_le_bytes(pb[firstb & !7..][..8].try_into().unwrap_or([0; 8])),
                    );
                }
                (
                    GpuOp::Sgemm {
                        a: aa, b: ab, c: ac, batch: abt, kind: akd, precision: apr, ..
                    },
                    GpuOp::Sgemm {
                        a: ba, b: bb, c: bc, batch: bbt, kind: bkd, precision: bpr, ..
                    },
                ) => {
                    eprintln!(
                        "[cuda-graph] region {id}: first divergence at op {i}: sgemm ptrs \
                         ({aa:#x},{ab:#x},{ac:#x}) batch={abt} {akd:?} {apr:?} vs \
                         ({ba:#x},{bb:#x},{bc:#x}) batch={bbt} {bkd:?} {bpr:?}"
                    );
                }
                _ => {
                    let desc = |o: &GpuOp| match o {
                        GpuOp::Kernel { func, grid, block, shared, params, name, .. } => format!(
                            "kernel '{name}' {func:#x} grid{grid:?} block{block:?} sh{shared} params[{}]={:02x?}",
                            params.len(),
                            &params[..params.len().min(48)]
                        ),
                        GpuOp::Sgemm {
                            a, b, c, m, n, k, batch, kind, precision, transa, transb, ..
                        } => {
                            format!(
                                "sgemm[{kind:?} {precision:?} ta={transa} tb={transb}] a={a:#x} \
                                 b={b:#x} c={c:#x} {m}x{n}x{k} batch={batch}"
                            )
                        }
                        GpuOp::Memset { dst, bytes } => format!("memset dst={dst:#x} n={bytes}"),
                        GpuOp::HtoD { dst, len, .. } => format!("htod dst={dst:#x} n={len}"),
                        GpuOp::DtoD { dst, src, len } => {
                            format!("dtod dst={dst:#x} src={src:#x} n={len}")
                        }
                    };
                    eprintln!(
                        "[cuda-graph] region {id}: first divergence at op {i}:\n  prev: {}\n  cur:  {}",
                        desc(a),
                        desc(b)
                    );
                }
            }
            return;
        }
    }

    /// Wait for the last launch of a region's graph to retire (no-op when it
    /// never launched). MUST be called before refreshing any HtoD staging
    /// buffer of that graph and before dropping its seq — the graph's memcpy
    /// nodes DMA-read our pinned staging at execution time (review H1/H2).
    fn sync_launch_event(ev: usize) {
        if ev != 0 {
            let r = unsafe { cuEventSynchronize(ev as CUevent) };
            debug_assert_eq!(r, CUresult::CUDA_SUCCESS, "launch-event sync failed: {r:?}");
        }
    }

    fn destroy_launch_event(ev: usize) {
        if ev != 0 {
            unsafe { cuEventDestroy_v2(ev as CUevent) };
        }
    }

    fn fail_state(attempts: u32) -> RegionState {
        if attempts + 1 >= MAX_CAPTURE_ATTEMPTS {
            EAGER_REGIONS.fetch_add(1, Ordering::Relaxed);
            RegionState::Eager
        } else {
            RegionState::Record { prev_digest: None, streak: 0, attempts: attempts + 1, prev_seq: None }
        }
    }

    // ------------------------------------------------------------------
    // Region lifecycle
    // ------------------------------------------------------------------

    pub fn region_begin(id: i64) {
        if !enabled() {
            return;
        }
        let nested = ACTIVE.with(|a| a.borrow().is_some());
        if nested {
            // Should never happen (regions are straight-line lowerings) —
            // ignore the inner region entirely, but keep begin/end balanced.
            NESTED_SKIP.with(|n| n.set(n.get() + 1));
            if log_on() {
                eprintln!("[cuda-graph] nested region_begin({id}) ignored");
            }
            return;
        }
        let phase = OCCURRENCE.with(|o| {
            let mut m = o.borrow_mut();
            let c = m.entry(id).or_insert(0);
            let cur = *c;
            *c += 1;
            cur % WINDOW.load(Ordering::Relaxed).max(1)
        });
        let state = REGIONS.with(|r| r.borrow_mut().remove(&(id, phase)));
        let state = state.unwrap_or_else(|| {
            REGIONS_SEEN.fetch_add(1, Ordering::Relaxed);
            RegionState::Record { prev_digest: None, streak: 0, attempts: 0, prev_seq: None }
        });
        let active = match state {
            RegionState::Eager => {
                REGIONS.with(|r| r.borrow_mut().insert((id, phase), RegionState::Eager));
                return;
            }
            RegionState::Record { prev_digest, streak, attempts, prev_seq } => {
                if streak >= STABLE_STREAK {
                    // Stable — capture this pass.
                    let stream = crate::cuda::inner::current_stream();
                    let r = unsafe {
                        cuStreamBeginCapture_v2(
                            stream,
                            CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
                        )
                    };
                    if r != CUresult::CUDA_SUCCESS {
                        if log_on() {
                            eprintln!("[cuda-graph] region {id}: begin-capture failed: {r:?}");
                        }
                        REGIONS.with(|reg| reg.borrow_mut().insert((id, phase), fail_state(attempts)));
                        return;
                    }
                    Active {
                        id,
                        phase,
                        mode: Mode::Capturing,
                        seq: Vec::new(),
                        exec: 0,
                        launch_event: 0,
                        prev_digest,
                        streak,
                        attempts,
                        prev_seq: None,
                        post: None,
                        deferred: Vec::new(),
                    }
                } else {
                    Active {
                        id,
                        phase,
                        mode: Mode::Recording { tainted: false },
                        seq: Vec::new(),
                        exec: 0,
                        launch_event: 0,
                        prev_digest,
                        streak,
                        attempts,
                        prev_seq,
                        post: None,
                        deferred: Vec::new(),
                    }
                }
            }
            RegionState::Captured { exec, seq, attempts, launch_event } => {
                // H1 fence: the HtoD staging buffers are about to be refreshed
                // with this step's payloads — wait for the PREVIOUS launch of
                // this graph (whose memcpy nodes read those buffers) first.
                if launch_event != 0
                    && seq.iter().any(|op| matches!(op, GpuOp::HtoD { .. }))
                {
                    sync_launch_event(launch_event);
                }
                Active {
                    id,
                    phase,
                    mode: Mode::Skipping { idx: 0 },
                    seq,
                    exec,
                    launch_event,
                    prev_digest: None,
                    streak: 0,
                    attempts,
                    prev_seq: None,
                    post: None,
                    deferred: Vec::new(),
                }
            }
        };
        ACTIVE.with(|a| *a.borrow_mut() = Some(active));
    }

    pub fn region_end(id: i64) {
        if !enabled() {
            return;
        }
        let skip = NESTED_SKIP.with(|n| {
            let v = n.get();
            if v > 0 {
                // Only swallow the end that matches an ignored nested begin.
                let owns = ACTIVE.with(|a| {
                    a.borrow().as_ref().map(|act| act.id) == Some(id)
                });
                if !owns {
                    n.set(v - 1);
                    return true;
                }
            }
            false
        });
        if skip {
            return;
        }
        let Some(mut active) = ACTIVE.with(|a| a.borrow_mut().take()) else {
            return; // eager region — nothing was activated
        };
        if active.id != id {
            // Unbalanced markers (should be impossible from emitted code) —
            // fail SAFE, not silent: skipped/recorded work must still execute
            // before we disable (review M1).
            eprintln!(
                "[cuda-graph] region_end({id}) does not match active region {} — disabling",
                active.id
            );
            match active.mode {
                Mode::Skipping { idx } => {
                    sync_launch_event(active.launch_event);
                    destroy_launch_event(active.launch_event);
                    unsafe { cuGraphExecDestroy(active.exec as CUgraphExec) };
                    repair(&active.seq[..idx]);
                }
                Mode::Capturing => {
                    let stream = crate::cuda::inner::current_stream();
                    let mut graph: CUgraph = std::ptr::null_mut();
                    let _ = unsafe { cuStreamEndCapture(stream, &mut graph) };
                    if !graph.is_null() {
                        unsafe { cuGraphDestroy(graph) };
                    }
                    repair(&active.seq);
                }
                _ => {}
            }
            for ptr in active.deferred.drain(..) {
                crate::cuda::inner::defer_free_device_record(ptr as *mut c_void);
            }
            ENABLED.store(false, Ordering::Relaxed);
            return;
        }

        let next = if let Some(post) = active.post.take() {
            // A mid-region taint/mismatch already decided (and repaired).
            post
        } else {
            match active.mode {
                Mode::Recording { tainted } => {
                    if tainted || active.seq.is_empty() || active.seq.len() > MAX_REGION_OPS {
                        EAGER_REGIONS.fetch_add(1, Ordering::Relaxed);
                        RegionState::Eager
                    } else {
                        let d = digest(&active.seq);
                        let streak = if active.prev_digest == Some(d) {
                            active.streak + 1
                        } else {
                            1
                        };
                        if log_on() {
                            eprintln!(
                                "[cuda-graph] region {id}.{}: recorded {} ops, digest {d:016x}, streak {streak}",
                                active.phase,
                                active.seq.len()
                            );
                            if streak == 1 {
                                if let Some(prev) = &active.prev_seq {
                                    log_first_divergence(id, prev, &active.seq);
                                }
                            }
                        }
                        RegionState::Record {
                            prev_digest: Some(d),
                            streak,
                            attempts: active.attempts,
                            prev_seq: if log_on() {
                                Some(std::mem::take(&mut active.seq))
                            } else {
                                None
                            },
                        }
                    }
                }
                Mode::Capturing => {
                    let stream = crate::cuda::inner::current_stream();
                    let mut graph: CUgraph = std::ptr::null_mut();
                    let r = unsafe { cuStreamEndCapture(stream, &mut graph) };
                    if r != CUresult::CUDA_SUCCESS || graph.is_null() {
                        if log_on() {
                            eprintln!("[cuda-graph] region {id}: end-capture failed: {r:?}");
                        }
                        if !graph.is_null() {
                            unsafe { cuGraphDestroy(graph) };
                        }
                        // Captured launches never executed — re-issue them.
                        repair(&active.seq);
                        fail_state(active.attempts)
                    } else {
                        let mut exec: CUgraphExec = std::ptr::null_mut();
                        let ri = unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
                        unsafe { cuGraphDestroy(graph) };
                        if ri != CUresult::CUDA_SUCCESS || exec.is_null() {
                            if log_on() {
                                eprintln!("[cuda-graph] region {id}: instantiate failed: {ri:?}");
                            }
                            repair(&active.seq);
                            fail_state(active.attempts)
                        } else {
                            let rl = unsafe { cuGraphLaunch(exec, stream) };
                            if rl != CUresult::CUDA_SUCCESS {
                                unsafe { cuGraphExecDestroy(exec) };
                                repair(&active.seq);
                                fail_state(active.attempts)
                            } else {
                                CAPTURES.fetch_add(1, Ordering::Relaxed);
                                if log_on() {
                                    eprintln!(
                                        "[cuda-graph] region {id}.{}: captured ({} ops)",
                                        active.phase,
                                        active.seq.len()
                                    );
                                }
                                // Launch-fence event (H1/H2): recorded after
                                // every launch of this exec.
                                let mut ev: CUevent = std::ptr::null_mut();
                                let re = unsafe { cuEventCreate(&mut ev, 0x2) };
                                let launch_event = if re == CUresult::CUDA_SUCCESS {
                                    unsafe { cuEventRecord(ev, stream) };
                                    ev as usize
                                } else {
                                    0
                                };
                                RegionState::Captured {
                                    exec: exec as usize,
                                    seq: std::mem::take(&mut active.seq),
                                    attempts: active.attempts,
                                    launch_event,
                                }
                            }
                        }
                    }
                }
                Mode::Skipping { idx } => {
                    if idx == active.seq.len() {
                        let stream = crate::cuda::inner::current_stream();
                        let rl = unsafe { cuGraphLaunch(active.exec as CUgraphExec, stream) };
                        if rl != CUresult::CUDA_SUCCESS {
                            // Launch failed to enqueue — nothing NEW ran, but
                            // the previous launch may still be in flight and
                            // reading the seq's staging buffers (H2).
                            sync_launch_event(active.launch_event);
                            destroy_launch_event(active.launch_event);
                            unsafe { cuGraphExecDestroy(active.exec as CUgraphExec) };
                            repair(&active.seq);
                            EAGER_REGIONS.fetch_add(1, Ordering::Relaxed);
                            RegionState::Eager
                        } else {
                            REPLAYS.fetch_add(1, Ordering::Relaxed);
                            if active.launch_event != 0 {
                                unsafe {
                                    cuEventRecord(active.launch_event as CUevent, stream)
                                };
                            }
                            RegionState::Captured {
                                exec: active.exec,
                                seq: std::mem::take(&mut active.seq),
                                attempts: active.attempts,
                                launch_event: active.launch_event,
                            }
                        }
                    } else {
                        // Host issued FEWER ops than captured: the skipped
                        // prefix matched, so re-issue it and drop the graph.
                        // H2 fence before dropping the seq (pinned staging).
                        MISMATCHES.fetch_add(1, Ordering::Relaxed);
                        sync_launch_event(active.launch_event);
                        destroy_launch_event(active.launch_event);
                        unsafe { cuGraphExecDestroy(active.exec as CUgraphExec) };
                        repair(&active.seq[..idx]);
                        fail_state(active.attempts)
                    }
                }
                Mode::EagerRest => {
                    // post was consumed above — reaching here means a
                    // transition forgot to set it; fail safe.
                    RegionState::Eager
                }
            }
        };
        REGIONS.with(|r| r.borrow_mut().insert((id, active.phase), next));
        // Region work is on the stream (graph launch or eager) — the queued
        // deferred frees may record their completion events now.
        for ptr in active.deferred.drain(..) {
            crate::cuda::inner::defer_free_device_record(ptr as *mut c_void);
        }
    }

    // ------------------------------------------------------------------
    // Pseudo-op hooks (called from the primitive wrappers in cuda::inner)
    // ------------------------------------------------------------------

    /// Mid-region divergence: repair the verified prefix (Skipping) or the
    /// whole recorded sequence (Capturing, after ending the capture), switch
    /// to EagerRest, and set the region's next state.
    ///
    /// Returns the ops that must be re-issued; the caller invokes `repair`
    /// AFTER releasing the ACTIVE borrow (repair re-enters the sgemm hook).
    fn diverge(active: &mut Active, next: RegionState) -> Vec<GpuOp> {
        let ops = match active.mode {
            Mode::Capturing => {
                // Abort the capture: recorded work never executed.
                let stream = crate::cuda::inner::current_stream();
                let mut graph: CUgraph = std::ptr::null_mut();
                let r = unsafe { cuStreamEndCapture(stream, &mut graph) };
                if !graph.is_null() {
                    unsafe { cuGraphDestroy(graph) };
                }
                let _ = r; // any status: the graph is discarded either way
                std::mem::take(&mut active.seq)
            }
            Mode::Skipping { idx } => {
                // H2 fence: the previous launch of this graph may still be
                // reading the pinned staging buffers owned by the seq we are
                // about to truncate/drop.
                sync_launch_event(active.launch_event);
                destroy_launch_event(active.launch_event);
                active.launch_event = 0;
                unsafe { cuGraphExecDestroy(active.exec as CUgraphExec) };
                active.exec = 0;
                let mut pre = std::mem::take(&mut active.seq);
                pre.truncate(idx);
                pre
            }
            _ => Vec::new(),
        };
        active.mode = Mode::EagerRest;
        active.post = Some(next);
        ops
    }

    /// Hook for `kernel_launch`. Returns `false` when the launch must be
    /// SKIPPED (verified against the captured sequence; the graph launch at
    /// region end performs it).
    pub fn on_kernel(
        func: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        args: &[*mut c_void],
        name_ptr: *const u8,
    ) -> bool {
        if !enabled() {
            return true;
        }
        let outcome = ACTIVE.with(|a| {
            let mut guard = a.borrow_mut();
            let Some(active) = guard.as_mut() else { return HookOutcome::Issue };
            match active.mode {
                Mode::EagerRest => HookOutcome::Issue,
                Mode::Recording { ref mut tainted } => {
                    match read_kernel_op(func, grid, block, shared, args, name_ptr) {
                        Some(op) => {
                            active.seq.push(op);
                            HookOutcome::Issue
                        }
                        None => {
                            *tainted = true;
                            HookOutcome::Issue
                        }
                    }
                }
                Mode::Capturing => {
                    match read_kernel_op(func, grid, block, shared, args, name_ptr) {
                        Some(op) => {
                            active.seq.push(op);
                            HookOutcome::Issue
                        }
                        None => {
                            // Param query failed only NOW (never during the
                            // record steps that proved stability) — cannot
                            // happen in practice, but fail safe: abort.
                            TAINTS.fetch_add(1, Ordering::Relaxed);
                            HookOutcome::Diverged(diverge(active, fail_state(active.attempts)))
                        }
                    }
                }
                Mode::Skipping { idx } => {
                    if idx == 0 && test_diverge_arm(active.id, active.phase) {
                        // Test-only kill switch: force the op-0 divergence
                        // that motivated `HookOutcome` (see its docs), once
                        // per (region, phase) — the region then re-records,
                        // re-captures and replays normally. The gate asserts
                        // the run stays bit-exact anyway.
                        MISMATCHES.fetch_add(1, Ordering::Relaxed);
                        return HookOutcome::Diverged(diverge(
                            active,
                            fail_state(active.attempts),
                        ));
                    }
                    let matches = read_kernel_op(func, grid, block, shared, args, name_ptr)
                        .is_some_and(|op| active.seq.get(idx) == Some(&op));
                    if matches {
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return HookOutcome::Handled;
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    HookOutcome::Diverged(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match outcome {
            HookOutcome::Handled => false,
            HookOutcome::Diverged(ops) => {
                repair(&ops);
                true
            }
            HookOutcome::Issue => true,
        }
    }

    /// `NSL_CUDA_GRAPH_TEST_DIVERGE_FIRST=1` — test-only: force each
    /// (region, phase)'s first replay verification to report a mismatch at
    /// op 0. This is the exact shape that used to conflate with the skip
    /// sentinel and drop the op (see `HookOutcome`); the e2e gate keeps it
    /// covered. Returns true exactly once per (region, phase); the fired
    /// set is global (NOT on `Active`, which is rebuilt every occurrence —
    /// a per-occurrence flag would re-fire on every replay attempt and the
    /// region could never re-capture).
    fn test_diverge_arm(id: i64, phase: u64) -> bool {
        static FLAG: OnceLock<bool> = OnceLock::new();
        if !*FLAG.get_or_init(|| {
            std::env::var("NSL_CUDA_GRAPH_TEST_DIVERGE_FIRST").ok().as_deref() == Some("1")
        }) {
            return false;
        }
        static FIRED: OnceLock<Mutex<std::collections::HashSet<(i64, u64)>>> = OnceLock::new();
        FIRED
            .get_or_init(|| Mutex::new(std::collections::HashSet::new()))
            .lock()
            .unwrap()
            .insert((id, phase))
    }

    /// Verdict from a hook's ACTIVE-borrow section, deciding whether the
    /// wrapper performs its real GPU op.
    ///
    /// This is a three-way enum because two of its cases used to share one
    /// `Option<Vec<GpuOp>>` emptiness sentinel: "verified — skip" and "the
    /// region just diverged" both surfaced as a vector, and a divergence at
    /// region op 0 (or a capture abort with nothing recorded yet) produced
    /// an EMPTY repair prefix — indistinguishable from the skip sentinel.
    /// The hook then returned "skip" for an op the destroyed graph would
    /// never perform, and the step ran with a hole in it: e.g. a dropped
    /// bf16 operand cast whose GemmEx went on to read stale scratch —
    /// silent numeric corruption, no error, no taint (review 2026-08-04,
    /// HIGH). The enum makes the ambiguity unrepresentable; `Diverged`
    /// always issues the current op after repairing, even when the prefix
    /// is empty.
    enum HookOutcome {
        /// Perform the real op (recording, capturing as a graph node via
        /// the stream, eager, or no active region).
        Issue,
        /// The wrapper must NOT perform the real op: it was verified
        /// against the captured sequence and the region-end graph launch
        /// will perform it, or the capture hook already issued the async
        /// form itself.
        Handled,
        /// The region diverged at this op: eagerly re-issue the
        /// matched-and-skipped prefix (possibly empty — a mismatch at op 0
        /// has nothing to repair), then perform the current op.
        Diverged(Vec<GpuOp>),
    }

    fn read_kernel_op(
        func: usize,
        grid: [u32; 3],
        block: [u32; 3],
        shared: u32,
        args: &[*mut c_void],
        name_ptr: *const u8,
    ) -> Option<GpuOp> {
        let sizes = param_sizes(func)?;
        if sizes.len() != args.len() {
            return None;
        }
        let total: usize = sizes.iter().sum();
        let mut params = Vec::with_capacity(total + sizes.len() * 8);
        let mut offsets = Vec::with_capacity(sizes.len());
        for (i, &sz) in sizes.iter().enumerate() {
            // 8-byte-align each argument so eager-repair hands cuLaunchKernel
            // naturally-aligned pointers (review L8; Vec<u8> data is at least
            // 8-aligned from the global allocator for these sizes).
            let off = (params.len() + 7) & !7;
            params.resize(off, 0);
            offsets.push(off);
            let bytes = unsafe { std::slice::from_raw_parts(args[i] as *const u8, sz) };
            params.extend_from_slice(bytes);
        }
        let name = if log_on() && !name_ptr.is_null() {
            unsafe { std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::new()
        };
        Some(GpuOp::Kernel { func, grid, block, shared, params, offsets, name })
    }

    /// Hook for `sgemm_batched_row_major` (item 9). Returns `false` when the
    /// gemm must be skipped (verified against the captured sequence);
    /// `stride_*` are element offsets, 0 meaning broadcast.
    #[allow(clippy::too_many_arguments)]
    pub fn on_sgemm_batched(
        a: usize, b: usize, c: usize, m: u64, n: u64, k: u64, alpha: f32, beta: f32,
        batch: u64, stride_a: u64, stride_b: u64, stride_c: u64,
    ) -> bool {
        // The batched wrapper has no bf16-storage arm — always f32.
        on_sgemm_full(
            SgemmKind::RowMajor, GemmPrecision::F32, false, false,
            a, b, c, m, n, k, alpha, beta, batch, stride_a, stride_b, stride_c,
        )
    }

    /// The full hook. Every cuBLAS gemm wrapper routes here so that ops which
    /// differ in wrapper or transposition cannot compare equal.
    ///
    /// The `strides` for a `batch == 1` op are the natural per-slice extents
    /// rather than 0, so a single-slice call recorded through the batched hook
    /// and one recorded through the 2-D hook describe the same geometry.
    #[allow(clippy::too_many_arguments)]
    pub fn on_sgemm_full(
        kind: SgemmKind, precision: GemmPrecision, transa: bool, transb: bool,
        a: usize, b: usize, c: usize, m: u64, n: u64, k: u64, alpha: f32, beta: f32,
        batch: u64, stride_a: u64, stride_b: u64, stride_c: u64,
    ) -> bool {
        if !enabled() {
            return true;
        }
        let op = GpuOp::Sgemm {
            kind, precision, transa, transb,
            a, b, c, m, n, k,
            alpha_bits: alpha.to_bits(),
            beta_bits: beta.to_bits(),
            batch, stride_a, stride_b, stride_c,
        };
        let outcome = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else { return HookOutcome::Issue };
            match active.mode {
                Mode::EagerRest => HookOutcome::Issue,
                Mode::Recording { .. } | Mode::Capturing => {
                    active.seq.push(op.clone());
                    HookOutcome::Issue
                }
                Mode::Skipping { idx } => {
                    if active.seq.get(idx) == Some(&op) {
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return HookOutcome::Handled;
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    HookOutcome::Diverged(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match outcome {
            HookOutcome::Handled => false,
            HookOutcome::Diverged(ops) => {
                repair(&ops);
                true
            }
            HookOutcome::Issue => true,
        }
    }

    /// Disposition for a zero-fill memset inside/outside a region.
    pub enum MemsetAction {
        /// Run the plain synchronous memset (no active region).
        Sync,
        /// Capture in progress: run as `cuMemsetD8Async` on the compute
        /// stream so the driver records a memset node.
        AsyncOnComputeStream,
        /// Verified against the captured sequence — skip entirely.
        Skip,
    }

    pub fn on_memset(dst: usize, bytes: usize) -> MemsetAction {
        if !enabled() {
            return MemsetAction::Sync;
        }
        let op = GpuOp::Memset { dst, bytes };
        let (action, to_repair) = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else {
                return (MemsetAction::Sync, None);
            };
            match active.mode {
                Mode::EagerRest => (MemsetAction::Sync, None),
                Mode::Recording { .. } => {
                    active.seq.push(op.clone());
                    (MemsetAction::Sync, None)
                }
                Mode::Capturing => {
                    active.seq.push(op.clone());
                    (MemsetAction::AsyncOnComputeStream, None)
                }
                Mode::Skipping { idx } => {
                    if active.seq.get(idx) == Some(&op) {
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return (MemsetAction::Skip, None);
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    let ops = diverge(active, fail_state(active.attempts));
                    (MemsetAction::Sync, Some(ops))
                }
            }
        });
        if let Some(ops) = to_repair {
            repair(&ops);
        }
        action
    }

    /// Hook for `memcpy_htod`. Returns `false` when the wrapper must NOT
    /// perform its synchronous copy (the hook already issued the captured
    /// async form, or the payload was refreshed into the staging buffer the
    /// graph's memcpy node reads at the region-end launch).
    ///
    /// Upload identity is (dst, len); the PAYLOAD may legitimately change
    /// every step (token-id batches). During capture the async copy reads a
    /// pinned staging buffer WE own; during replay only the staging contents
    /// are refreshed — so the graph never dereferences a caller pointer.
    pub fn on_htod(dst: *mut c_void, src: *const c_void, len: usize) -> bool {
        if !enabled() {
            return true;
        }
        let outcome = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else { return HookOutcome::Issue };
            match active.mode {
                Mode::EagerRest => HookOutcome::Issue,
                Mode::Recording { .. } => {
                    active.seq.push(GpuOp::HtoD { dst: dst as usize, len, staging: None });
                    HookOutcome::Issue
                }
                Mode::Capturing => match StagingBuf::alloc(len) {
                    Some(staging) => {
                        staging.fill_from(src, len);
                        let stream = crate::cuda::inner::current_stream();
                        let r = unsafe {
                            cuMemcpyHtoDAsync_v2(
                                dst as CUdeviceptr,
                                staging.ptr as *const c_void,
                                len,
                                stream,
                            )
                        };
                        if r != CUresult::CUDA_SUCCESS {
                            // Node creation failed — abort the capture and
                            // let the wrapper run the plain sync copy.
                            TAINTS.fetch_add(1, Ordering::Relaxed);
                            return HookOutcome::Diverged(diverge(
                                active,
                                fail_state(active.attempts),
                            ));
                        }
                        active.seq.push(GpuOp::HtoD {
                            dst: dst as usize,
                            len,
                            staging: Some(staging),
                        });
                        HookOutcome::Handled // wrapper must not copy
                    }
                    None => {
                        TAINTS.fetch_add(1, Ordering::Relaxed);
                        HookOutcome::Diverged(diverge(active, fail_state(active.attempts)))
                    }
                },
                Mode::Skipping { idx } => {
                    let matches = matches!(
                        active.seq.get(idx),
                        Some(GpuOp::HtoD { dst: d, len: l, staging: Some(_) })
                            if *d == dst as usize && *l == len
                    );
                    if matches {
                        if let Some(GpuOp::HtoD { staging: Some(st), .. }) = active.seq.get(idx) {
                            st.fill_from(src, len);
                        }
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return HookOutcome::Handled;
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    HookOutcome::Diverged(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match outcome {
            HookOutcome::Handled => false,
            HookOutcome::Diverged(ops) => {
                repair(&ops);
                true
            }
            HookOutcome::Issue => true,
        }
    }

    /// Hook for `memcpy_dtod`. Device-to-device copies involve no host
    /// pointer, so they capture/replay like kernels: async node during
    /// capture, verify-and-skip during replay.
    pub fn on_dtod(dst: *mut c_void, src: *const c_void, len: usize) -> bool {
        if !enabled() {
            return true;
        }
        let op = GpuOp::DtoD { dst: dst as usize, src: src as usize, len };
        let outcome = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else { return HookOutcome::Issue };
            match active.mode {
                Mode::EagerRest => HookOutcome::Issue,
                Mode::Recording { .. } => {
                    active.seq.push(op.clone());
                    HookOutcome::Issue
                }
                Mode::Capturing => {
                    let stream = crate::cuda::inner::current_stream();
                    let r = unsafe {
                        cuMemcpyDtoDAsync_v2(
                            dst as CUdeviceptr,
                            src as CUdeviceptr,
                            len,
                            stream,
                        )
                    };
                    if r != CUresult::CUDA_SUCCESS {
                        TAINTS.fetch_add(1, Ordering::Relaxed);
                        return HookOutcome::Diverged(diverge(
                            active,
                            fail_state(active.attempts),
                        ));
                    }
                    active.seq.push(op.clone());
                    HookOutcome::Handled
                }
                Mode::Skipping { idx } => {
                    if active.seq.get(idx) == Some(&op) {
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return HookOutcome::Handled;
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    HookOutcome::Diverged(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match outcome {
            HookOutcome::Handled => false,
            HookOutcome::Diverged(ops) => {
                repair(&ops);
                true
            }
            HookOutcome::Issue => true,
        }
    }

    /// A synchronizing / transferring primitive fired inside a region.
    /// Recording: taints the region (never captures). Capturing/Skipping:
    /// aborts and repairs so the primitive proceeds against fully-issued
    /// work. Outside a region: no-op.
    pub fn taint(reason: &'static str) {
        if !enabled() {
            return;
        }
        let to_repair = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else { return None };
            match active.mode {
                Mode::EagerRest => None,
                Mode::Recording { ref mut tainted } => {
                    if !*tainted {
                        *tainted = true;
                        TAINTS.fetch_add(1, Ordering::Relaxed);
                        if log_on() {
                            eprintln!(
                                "[cuda-graph] region {}: tainted by {reason} — stays eager",
                                active.id
                            );
                        }
                    }
                    None
                }
                Mode::Capturing | Mode::Skipping { .. } => {
                    TAINTS.fetch_add(1, Ordering::Relaxed);
                    if log_on() {
                        eprintln!(
                            "[cuda-graph] region {}: {reason} during capture/replay — repairing",
                            active.id
                        );
                    }
                    Some(diverge(active, RegionState::Eager))
                }
            }
        });
        if let Some(ops) = to_repair {
            EAGER_REGIONS.fetch_add(1, Ordering::Relaxed);
            repair(&ops);
        }
    }

    /// Queue a deferred free while a region is active (its completion event
    /// must be recorded only after the region's work is on the stream).
    /// Returns false when no region is active — caller proceeds normally.
    pub fn queue_deferred_free(ptr: *mut c_void) -> bool {
        if !enabled() {
            return false;
        }
        ACTIVE.with(|a| {
            let mut guard = a.borrow_mut();
            match guard.as_mut() {
                Some(active) => {
                    active.deferred.push(ptr as usize);
                    true
                }
                None => false,
            }
        })
    }

    /// True while a region is active — `drain_completed_frees` must not poll
    /// events then (`cuEventQuery` is illegal during capture, and during
    /// replay the polled work may not be issued yet).
    pub fn in_region() -> bool {
        enabled() && ACTIVE.with(|a| a.borrow().is_some())
    }

    // ------------------------------------------------------------------
    // FFI-facing entry points
    // ------------------------------------------------------------------

    pub fn enable(window: i64) {
        if std::env::var("NSL_CUDA_GRAPHS").ok().as_deref() == Some("0") {
            eprintln!("[cuda-graph] disabled by NSL_CUDA_GRAPHS=0");
            return;
        }
        if crate::cuda::inner::sync_mode_enabled() {
            eprintln!("[cuda-graph] disabled: NSL_CUDA_SYNC=1 (eager sync is incompatible with capture)");
            return;
        }
        if crate::kernel_profiler::kernel_profiler_enabled() {
            eprintln!("[cuda-graph] disabled: kernel profiler active (per-launch events are incompatible with replay)");
            return;
        }
        if std::env::var("NSL_LEGACY_NULL_STREAM").ok().as_deref() == Some("1") {
            eprintln!("[cuda-graph] disabled: NSL_LEGACY_NULL_STREAM=1 (the NULL stream cannot be captured)");
            return;
        }
        if crate::cuda::inner::async_alloc_enabled() {
            // cuMemAllocAsync/cuMemFreeAsync are issued on the NULL stream
            // and the alloc path carries no taint hook — an allocation
            // inside a capture region would be recorded into the graph (or
            // fail the capture) with nothing self-healing it. Every other
            // stream-touching primitive taints; until this one does, the
            // composition is refused up front rather than left to corrupt.
            eprintln!(
                "[cuda-graph] disabled: NSL_ASYNC_ALLOC=1 (stream-ordered \
                 allocation on the NULL stream is not capture-safe)"
            );
            return;
        }
        LOGGING.store(
            std::env::var("NSL_CUDA_GRAPH_LOG").ok().as_deref() == Some("1"),
            Ordering::Relaxed,
        );
        WINDOW.store(window.max(1) as u64, Ordering::Relaxed);
        ENABLED.store(true, Ordering::Relaxed);
    }

    pub fn report() {
        if !enabled() && CAPTURES.load(Ordering::Relaxed) == 0 {
            return;
        }
        eprintln!(
            "[cuda-graph] regions={} captured={} replays={} taints={} mismatches={} repaired_ops={} eager={}",
            REGIONS_SEEN.load(Ordering::Relaxed),
            CAPTURES.load(Ordering::Relaxed),
            REPLAYS.load(Ordering::Relaxed),
            TAINTS.load(Ordering::Relaxed),
            MISMATCHES.load(Ordering::Relaxed),
            REPAIRED_OPS.load(Ordering::Relaxed),
            EAGER_REGIONS.load(Ordering::Relaxed),
        );
    }

    // ------------------------------------------------------------------
    // Op-identity gates (item 9 phase 2, added after independent review)
    // ------------------------------------------------------------------
    //
    // These live inside `imp` because `GpuOp` is private to it. They need no
    // GPU: equality and `digest` are pure functions over recorded values.
    //
    // Why they exist: the bug this commit fixed was "three places that must
    // agree about op identity, and one of them didn't". The fix added
    // `kind`/`transa`/`transb` to `GpuOp::Sgemm` and threaded them through
    // equality, the digest and the replay dispatch — but nothing pinned that
    // agreement, so dropping `kind` from the digest while keeping it in `eq`
    // (or the reverse) left the whole workspace green. An independent review
    // pointed that out. These are the gates it asked for.
    #[cfg(test)]
    mod op_identity_tests {
        use super::*;

        /// A row-major gemm and a weight-gradient gemm over the SAME three
        /// buffers with the SAME dims — the exact collision that used to make
        /// eager repair replay a wgrad contraction as a plain matmul.
        fn colliding_pair() -> (GpuOp, GpuOp) {
            let common = |kind: SgemmKind| GpuOp::Sgemm {
                kind,
                precision: GemmPrecision::F32,
                transa: false,
                transb: false,
                a: 0x1000,
                b: 0x2000,
                c: 0x3000,
                m: 64,
                n: 128,
                k: 32,
                alpha_bits: 1.0f32.to_bits(),
                beta_bits: 0.0f32.to_bits(),
                batch: 1,
                stride_a: 64 * 32,
                stride_b: 32 * 128,
                stride_c: 64 * 128,
            };
            (common(SgemmKind::RowMajor), common(SgemmKind::WgradAccum))
        }

        /// The same gemm over the same buffers in f32-storage and
        /// bf16-storage form. They enqueue different work (two cast kernels
        /// plus a 16BF GemmEx versus one f32 sgemm), so neither equality nor
        /// the digest may conflate them — a mode flip between record and a
        /// later step must diverge at this op, not replay the wrong storage
        /// route.
        fn precision_pair() -> (GpuOp, GpuOp) {
            let common = |precision: GemmPrecision| GpuOp::Sgemm {
                kind: SgemmKind::RowMajor,
                precision,
                transa: false,
                transb: false,
                a: 0x1000,
                b: 0x2000,
                c: 0x3000,
                m: 64,
                n: 128,
                k: 32,
                alpha_bits: 1.0f32.to_bits(),
                beta_bits: 0.0f32.to_bits(),
                batch: 1,
                stride_a: 64 * 32,
                stride_b: 32 * 128,
                stride_c: 64 * 128,
            };
            (common(GemmPrecision::F32), common(GemmPrecision::Bf16Storage))
        }

        #[test]
        fn f32_and_bf16_storage_are_not_equal() {
            let (f32_op, bf16_op) = precision_pair();
            assert!(
                f32_op != bf16_op,
                "an f32-storage and a bf16-storage gemm over the same \
                 buffers compared equal"
            );
        }

        #[test]
        fn f32_and_bf16_storage_digests_differ() {
            let (f32_op, bf16_op) = precision_pair();
            assert_ne!(
                digest(std::slice::from_ref(&f32_op)),
                digest(std::slice::from_ref(&bf16_op)),
                "storage route is invisible to the region digest"
            );
        }

        #[test]
        fn wgrad_and_row_major_are_not_equal() {
            let (row, wgrad) = colliding_pair();
            // `assert!(a != b)`, not `assert_ne!`: `GpuOp` has no `Debug`
            // (its `Kernel` variant carries a raw params blob).
            assert!(
                row != wgrad,
                "a weight-gradient gemm and a row-major gemm over the same \
                 buffers and dims compare EQUAL — replay would reproduce one \
                 as the other, silently computing the wrong contraction"
            );
        }

        #[test]
        fn wgrad_and_row_major_digest_differently() {
            let (row, wgrad) = colliding_pair();
            assert_ne!(
                digest(std::slice::from_ref(&row)),
                digest(std::slice::from_ref(&wgrad)),
                "the digest ignores SgemmKind, so a captured graph built for \
                 one contraction would be reused for the other. Equality and \
                 the digest must agree — disagreement between them is the \
                 exact failure class this field was added to close"
            );
        }

        /// Transposition is part of identity for the same reason: the ops
        /// select different lda/ldb, so a replay under the wrong flags reads a
        /// real, in-bounds, WRONG sub-matrix.
        #[test]
        fn transposition_flags_are_part_of_op_identity() {
            let (base, _) = colliding_pair();
            for mutate in [
                |o: &mut GpuOp| {
                    if let GpuOp::Sgemm { transa, .. } = o {
                        *transa = true;
                    }
                },
                |o: &mut GpuOp| {
                    if let GpuOp::Sgemm { transb, .. } = o {
                        *transb = true;
                    }
                },
            ] {
                let mut other = base.clone();
                mutate(&mut other);
                assert!(base != other, "transposition dropped from equality");
                assert_ne!(
                    digest(std::slice::from_ref(&base)),
                    digest(std::slice::from_ref(&other)),
                    "transposition dropped from the digest"
                );
            }
        }

        /// Sanity: two identical ops MUST still compare equal and hash alike.
        /// Without this the tests above would pass against an implementation
        /// that simply declared everything unequal.
        #[test]
        fn identical_ops_still_match() {
            let (row, _) = colliding_pair();
            let same = row.clone();
            assert!(row == same, "two identical ops compare unequal");
            assert_eq!(
                digest(std::slice::from_ref(&row)),
                digest(std::slice::from_ref(&same))
            );
        }
    }
}

// ----------------------------------------------------------------------
// FFI shims (both builds; no-ops without the `cuda` feature)
// ----------------------------------------------------------------------

/// Arm opportunistic per-region CUDA graph capture (emitted at train-block
/// start under `--cuda-graphs`). Refuses politely when an incompatible
/// diagnostic mode is active.
#[no_mangle]
pub extern "C" fn nsl_cuda_graphs_enable(accum_window: i64) {
    #[cfg(feature = "cuda")]
    imp::enable(accum_window);
    #[cfg(not(feature = "cuda"))]
    let _ = accum_window;
}

/// Begin a Wengert-lowering region (emitted by `compile_wengert_ops_range`).
#[no_mangle]
pub extern "C" fn nsl_cuda_graph_region_begin(id: i64) {
    #[cfg(feature = "cuda")]
    imp::region_begin(id);
    #[cfg(not(feature = "cuda"))]
    let _ = id;
}

/// End a Wengert-lowering region.
#[no_mangle]
pub extern "C" fn nsl_cuda_graph_region_end(id: i64) {
    #[cfg(feature = "cuda")]
    imp::region_end(id);
    #[cfg(not(feature = "cuda"))]
    let _ = id;
}

/// True when the runtime actually ARMED capture this run (as opposed to the
/// compile flag merely being set — enable() may decline). `nsl_gpu_drain_cache`
/// consults this: the per-step transient drain must be skipped only when
/// graphs are live (address stability + captured graphs reference the
/// retained segments), not merely requested (review L5).
pub fn cuda_graphs_armed() -> bool {
    #[cfg(feature = "cuda")]
    {
        imp::enabled()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Print the capture/replay counter banner (emitted at train-block teardown).
#[no_mangle]
pub extern "C" fn nsl_cuda_graphs_report() {
    #[cfg(feature = "cuda")]
    imp::report();
}

#[cfg(feature = "cuda")]
pub(crate) use imp::{
    in_region, on_dtod, on_htod, on_kernel, on_memset, on_sgemm_batched,
    on_sgemm_full, queue_deferred_free, taint,
    GemmPrecision, MemsetAction, SgemmKind,
};
