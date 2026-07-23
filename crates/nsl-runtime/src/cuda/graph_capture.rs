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
            a: usize,
            b: usize,
            c: usize,
            m: u64,
            n: u64,
            k: u64,
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
                    GpuOp::Sgemm { a, b, c, m, n, k },
                    GpuOp::Sgemm { a: a2, b: b2, c: c2, m: m2, n: n2, k: k2 },
                ) => a == a2 && b == b2 && c == c2 && m == m2 && n == n2 && k == k2,
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
                GpuOp::Sgemm { a, b, c, m, n, k } => {
                    eat(&[2]);
                    for v in [*a as u64, *b as u64, *c as u64, *m, *n, *k] {
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
    /// EagerRest/none — `sgemm_row_major` re-enters the on_sgemm hook.
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
                GpuOp::Sgemm { a, b, c, m, n, k } => {
                    let r = unsafe {
                        crate::cuda::cublas_inner::sgemm_row_major(
                            *a as *const f32,
                            *b as *const f32,
                            *c as *mut f32,
                            *m, *n, *k,
                        )
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
                (GpuOp::Sgemm { a: aa, b: ab, c: ac, .. }, GpuOp::Sgemm { a: ba, b: bb, c: bc, .. }) => {
                    eprintln!(
                        "[cuda-graph] region {id}: first divergence at op {i}: sgemm ptrs ({aa:#x},{ab:#x},{ac:#x}) vs ({ba:#x},{bb:#x},{bc:#x})"
                    );
                }
                _ => {
                    let desc = |o: &GpuOp| match o {
                        GpuOp::Kernel { func, grid, block, shared, params, name, .. } => format!(
                            "kernel '{name}' {func:#x} grid{grid:?} block{block:?} sh{shared} params[{}]={:02x?}",
                            params.len(),
                            &params[..params.len().min(48)]
                        ),
                        GpuOp::Sgemm { a, b, c, m, n, k } => {
                            format!("sgemm a={a:#x} b={b:#x} c={c:#x} {m}x{n}x{k}")
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
        let to_repair = ACTIVE.with(|a| {
            let mut guard = a.borrow_mut();
            let Some(active) = guard.as_mut() else { return None };
            match active.mode {
                Mode::EagerRest => None,
                Mode::Recording { ref mut tainted } => {
                    match read_kernel_op(func, grid, block, shared, args, name_ptr) {
                        Some(op) => {
                            active.seq.push(op);
                            None
                        }
                        None => {
                            *tainted = true;
                            None
                        }
                    }
                }
                Mode::Capturing => {
                    match read_kernel_op(func, grid, block, shared, args, name_ptr) {
                        Some(op) => {
                            active.seq.push(op);
                            None
                        }
                        None => {
                            // Param query failed only NOW (never during the
                            // record steps that proved stability) — cannot
                            // happen in practice, but fail safe: abort.
                            TAINTS.fetch_add(1, Ordering::Relaxed);
                            Some(diverge(active, fail_state(active.attempts)))
                        }
                    }
                }
                Mode::Skipping { idx } => {
                    let matches = read_kernel_op(func, grid, block, shared, args, name_ptr)
                        .is_some_and(|op| active.seq.get(idx) == Some(&op));
                    if matches {
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return Some(Vec::new()); // sentinel: SKIP the launch
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    Some(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match to_repair {
            Some(ops) if ops.is_empty() => false, // verified — skip
            Some(ops) => {
                repair(&ops);
                true
            }
            None => true,
        }
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

    /// Hook for `sgemm_row_major`. Returns `false` when the gemm must be
    /// skipped (verified against the captured sequence).
    pub fn on_sgemm(a: usize, b: usize, c: usize, m: u64, n: u64, k: u64) -> bool {
        if !enabled() {
            return true;
        }
        let op = GpuOp::Sgemm { a, b, c, m, n, k };
        let to_repair = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else { return None };
            match active.mode {
                Mode::EagerRest => None,
                Mode::Recording { .. } | Mode::Capturing => {
                    active.seq.push(op.clone());
                    None
                }
                Mode::Skipping { idx } => {
                    if active.seq.get(idx) == Some(&op) {
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return Some(Vec::new());
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    Some(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match to_repair {
            Some(ops) if ops.is_empty() => false,
            Some(ops) => {
                repair(&ops);
                true
            }
            None => true,
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
        let to_repair = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else { return None };
            match active.mode {
                Mode::EagerRest => None,
                Mode::Recording { .. } => {
                    active.seq.push(GpuOp::HtoD { dst: dst as usize, len, staging: None });
                    None
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
                            return Some(diverge(active, fail_state(active.attempts)));
                        }
                        active.seq.push(GpuOp::HtoD {
                            dst: dst as usize,
                            len,
                            staging: Some(staging),
                        });
                        Some(Vec::new()) // handled — wrapper must not copy
                    }
                    None => {
                        TAINTS.fetch_add(1, Ordering::Relaxed);
                        Some(diverge(active, fail_state(active.attempts)))
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
                        return Some(Vec::new());
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    Some(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match to_repair {
            Some(ops) if ops.is_empty() => false,
            Some(ops) => {
                repair(&ops);
                true
            }
            None => true,
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
        let to_repair = ACTIVE.with(|act| {
            let mut guard = act.borrow_mut();
            let Some(active) = guard.as_mut() else { return None };
            match active.mode {
                Mode::EagerRest => None,
                Mode::Recording { .. } => {
                    active.seq.push(op.clone());
                    None
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
                        return Some(diverge(active, fail_state(active.attempts)));
                    }
                    active.seq.push(op.clone());
                    Some(Vec::new())
                }
                Mode::Skipping { idx } => {
                    if active.seq.get(idx) == Some(&op) {
                        active.mode = Mode::Skipping { idx: idx + 1 };
                        return Some(Vec::new());
                    }
                    MISMATCHES.fetch_add(1, Ordering::Relaxed);
                    Some(diverge(active, fail_state(active.attempts)))
                }
            }
        });
        match to_repair {
            Some(ops) if ops.is_empty() => false,
            Some(ops) => {
                repair(&ops);
                true
            }
            None => true,
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
    in_region, on_dtod, on_htod, on_kernel, on_memset, on_sgemm, queue_deferred_free, taint,
    MemsetAction,
};
