//! Fast paths for `nsl_tensor_contiguous` on GPU views.
//!
//! The generic materializer (`STRIDED_COPY_F32_PTX` in `fused_kernels`) decomposes
//! every output element's flat index into N-dimensional coordinates with a
//! per-dimension `div.u64` / `rem.u64` pair. 64-bit integer division has no
//! hardware unit on any current SM, so a 5-D view costs ~10 emulated divisions
//! *per element* — the kernel is ALU-bound at roughly 140 GB/s on hardware that
//! sustains 678 GB/s, i.e. it runs about 5x slower than a plain copy.
//!
//! Almost all of the traffic in a training step does not need that generality.
//! Measured over one Coder-50M step (see `strided_copy_plan` tests for the exact
//! shapes), 69% of copied elements come from views whose innermost dimension is
//! *already* a contiguous run in both source and destination — broadcasts
//! (`expand` with stride 0) and head/batch reorderings that keep `head_dim`
//! innermost. For those, the coordinate decomposition is loop-invariant across
//! the whole run: it only has to happen once per run, on the host.
//!
//! [`plan_run`] performs that analysis:
//!   1. drop size-1 axes, then merge adjacent axes that are contiguous in BOTH
//!      the source view and the (row-major) destination — a 5-D GQA expand
//!      collapses to 2-D, a full broadcast collapses to 1-D;
//!   2. require the innermost collapsed axis to have `dst_stride == 1` and
//!      `src_stride` of either 1 (copy) or 0 (broadcast);
//!   3. precompute one source element offset per run.
//!
//! The kernel then does no index math beyond `base + i`. Views that do not fit
//! (true transposes, where the innermost axis is strided) fall back to the
//! generic kernel unchanged.

/// Smallest run length worth the per-run offset table. Below this the table
/// approaches the size of the data itself and the generic kernel is fine.
const MIN_RUN: i64 = 32;

/// Cap on the offset table (entries). 1M entries = 8 MiB of device metadata,
/// which the content-keyed cache holds persistently; anything larger falls back.
const MAX_OUTER: i64 = 1 << 20;

/// CUDA's `gridDim.y` limit. Plans with more runs than this are still valid —
/// the kernels walk `outer` with a grid-stride loop over y.
const MAX_GRID_Y: i64 = 65535;

/// Total device memory the never-freed offset tables may occupy, across all
/// cached plans. A Coder-50M step uses ~14 distinct views totalling well under
/// 1 MiB; the budget only bites on a program that materializes an unbounded
/// variety of large views, which then falls back to the generic kernel rather
/// than growing device memory without limit.
#[cfg(feature = "cuda")]
const OFFSET_TABLE_BUDGET: usize = 64 << 20;

/// A view rewritten as `outer` independent contiguous runs of `run_len` elements.
///
/// Destination offsets are implicit: run `o` writes `dst[o * run_len ..][.. run_len]`.
/// That is guaranteed by [`plan_run`], which only accepts a plan whose collapsed
/// destination strides are exactly the row-major strides over the runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RunPlan {
    /// Elements per run (>= [`MIN_RUN`]).
    pub run_len: i64,
    /// Number of runs.
    pub outer: i64,
    /// Source element offset of each run's first element. Length == `outer`.
    pub src_offsets: Vec<i64>,
    /// True when the innermost source stride is 0: every element of a run reads
    /// the SAME source element (an `expand` axis), so the kernel loads once and
    /// splats.
    pub broadcast: bool,
    /// True when the run can be moved four elements at a time with
    /// `ld.global.v4.f32` / `st.global.v4.f32`.
    pub vec4: bool,
}

impl RunPlan {
    /// Launch geometry: `(grid, block)`.
    ///
    /// `grid.x` covers one run, `grid.y` covers the runs. When a run is shorter
    /// than a full block the block is shrunk to it (rounded up to a warp) rather
    /// than launching threads that immediately exit — `run_len == 64` would
    /// otherwise idle 75% of every block.
    pub fn geometry(&self) -> ([i64; 3], [i64; 3]) {
        let units = if self.vec4 { self.run_len / 4 } else { self.run_len };
        let block = (((units + 31) / 32) * 32).clamp(32, 256);
        let grid_x = (units + block - 1) / block;
        let grid_y = self.outer.min(MAX_GRID_Y);
        ([grid_x, grid_y, 1], [block, 1, 1])
    }

    /// Entry-point name for this plan's arm.
    pub fn kernel_name(&self) -> &'static str {
        match (self.broadcast, self.vec4) {
            (false, false) => "nsl_scopy_run_f32\0",
            (false, true) => "nsl_scopy_run4_f32\0",
            (true, false) => "nsl_scopy_bcast_f32\0",
            (true, true) => "nsl_scopy_bcast4_f32\0",
        }
    }
}

/// One axis of a collapsed view: `(size, src_stride, dst_stride)`.
type Axis = (i64, i64, i64);

/// Drop size-1 axes, then merge each adjacent pair that is contiguous in both
/// the source and the destination.
///
/// Two axes `(outer_size, so, do)` and `(inner_size, si, di)` merge when
/// `so == si * inner_size` and `do == di * inner_size` — i.e. stepping the outer
/// axis once is exactly stepping the inner axis `inner_size` times, in both
/// layouts. The merged axis keeps the inner strides.
///
/// Broadcast axes participate: two stride-0 source axes merge with each other
/// (0 == 0 * n) whenever their destination axes are adjacent, so `[4, 8, 1024,
/// 64]` with source strides `[0, 0, 64, 1]` collapses to `[(32, 0, 65536),
/// (65536, 1, 1)]`.
fn collapse(shape: &[i64], src: &[i64], dst: &[i64]) -> Vec<Axis> {
    let mut out: Vec<Axis> = Vec::with_capacity(shape.len());
    for ((&size, &s), &d) in shape.iter().zip(src).zip(dst) {
        if size == 1 {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if last.1 == s.saturating_mul(size) && last.2 == d.saturating_mul(size) {
                last.0 *= size;
                last.1 = s;
                last.2 = d;
                continue;
            }
        }
        out.push((size, s, d));
    }
    out
}

/// Try to rewrite a view as contiguous runs. `None` means "use the generic kernel".
///
/// `dst` must be the row-major strides of `shape` (the materialized layout).
pub(crate) fn plan_run(shape: &[i64], src: &[i64], dst: &[i64]) -> Option<RunPlan> {
    if shape.len() != src.len() || shape.len() != dst.len() || shape.is_empty() {
        return None;
    }
    if shape.iter().any(|&s| s <= 0) {
        return None;
    }

    let axes = collapse(shape, src, dst);
    let (&(run_len, src_inner, dst_inner), outer_axes) = axes.split_last()?;

    // The run must be contiguous in the destination and either contiguous
    // (copy) or constant (broadcast) in the source.
    if dst_inner != 1 || (src_inner != 0 && src_inner != 1) {
        return None;
    }
    if run_len < MIN_RUN {
        return None;
    }

    let outer: i64 = outer_axes.iter().try_fold(1i64, |acc, a| acc.checked_mul(a.0))?;
    if outer > MAX_OUTER {
        return None;
    }

    // Destination offsets are implicit (`o * run_len`), which requires the outer
    // destination strides to be exactly row-major over the runs. A contiguous
    // destination always satisfies this; check rather than assume, because a
    // caller passing non-contiguous `dst` strides would otherwise silently write
    // to the wrong addresses.
    let mut expected = run_len;
    for axis in outer_axes.iter().rev() {
        if axis.2 != expected {
            return None;
        }
        expected = expected.checked_mul(axis.0)?;
    }

    // One source offset per run, enumerating the outer axes in row-major order.
    let mut src_offsets = Vec::with_capacity(outer as usize);
    for o in 0..outer {
        let mut rem = o;
        let mut off = 0i64;
        for axis in outer_axes.iter().rev() {
            off += (rem % axis.0) * axis.1;
            rem /= axis.0;
        }
        src_offsets.push(off);
    }

    // Vector moves need 4-element granularity on both sides. The destination is
    // a fresh allocation at `o * run_len`, so `run_len % 4 == 0` covers it. A
    // broadcast reads a single scalar, so only a real copy constrains the source.
    let vec4 = run_len % 4 == 0
        && (src_inner == 0 || outer_axes.iter().all(|a| a.1 % 4 == 0));

    Some(RunPlan {
        run_len,
        outer,
        src_offsets,
        broadcast: src_inner == 0,
        vec4,
    })
}

/// Four entry points sharing one module, selected by [`RunPlan::kernel_name`].
///
/// Common contract:
///   `src`, `dst`     — f32 device pointers
///   `offsets`        — i64[outer], source element offset of each run
///   `run_len`        — elements per run
///   `outer`          — number of runs
///   grid.x spans one run, grid.y spans the runs (grid-stride, so `outer` may
///   exceed 65535); `dst` run `o` starts at element `o * run_len`.
///
/// ISA 7.0: no `mad.lo.u32` (see `feedback_ptx_comment_ascii_only` and the
/// PTX invariants in the wiki) -- index math is `mul.lo.u32` + `add.u32`.
/// ASCII only: a non-ASCII byte anywhere in the module makes `cuModuleLoadData`
/// return CUDA_ERROR_INVALID_PTX.
pub(crate) const STRIDED_COPY_RUN_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
// dst[o*run_len + i] = src[offsets[o] + i]\n\
.visible .entry nsl_scopy_run_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    setp.ge.u32 %p1, %r3, %r4;\n\
    @%p1 bra SC_RUN_DONE;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_RUN_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_RUN_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    add.u64 %rd9, %rd9, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.f32 [%rd10], %f1;\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_RUN_LOOP;\n\
SC_RUN_DONE: ret;\n\
}\n\
\n\
// Vector form of nsl_scopy_run_f32: i counts float4 units.\n\
.visible .entry nsl_scopy_run4_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<5>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    shr.u32 %r8, %r4, 2;\n\
    setp.ge.u32 %p1, %r3, %r8;\n\
    @%p1 bra SC_RUN4_DONE;\n\
    shl.b32 %r9, %r3, 2;\n\
    cvt.u64.u32 %rd6, %r9;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_RUN4_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_RUN4_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    add.u64 %rd9, %rd9, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.v4.f32 [%rd10], {%f1, %f2, %f3, %f4};\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_RUN4_LOOP;\n\
SC_RUN4_DONE: ret;\n\
}\n\
\n\
// dst[o*run_len + i] = src[offsets[o]]  (innermost source stride 0)\n\
.visible .entry nsl_scopy_bcast_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    setp.ge.u32 %p1, %r3, %r4;\n\
    @%p1 bra SC_BC_DONE;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_BC_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_BC_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.f32 [%rd10], %f1;\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_BC_LOOP;\n\
SC_BC_DONE: ret;\n\
}\n\
\n\
// Vector form of nsl_scopy_bcast_f32: one scalar load, one v4 store.\n\
.visible .entry nsl_scopy_bcast4_f32(\n\
    .param .u64 src, .param .u64 dst, .param .u64 offsets,\n\
    .param .u64 run_len, .param .u64 outer\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<2>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r3, %r1, %r2;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [offsets];\n\
    ld.param.u64 %rd4, [run_len];\n\
    ld.param.u64 %rd5, [outer];\n\
    cvt.u32.u64 %r4, %rd4;\n\
    cvt.u32.u64 %r5, %rd5;\n\
    shr.u32 %r8, %r4, 2;\n\
    setp.ge.u32 %p1, %r3, %r8;\n\
    @%p1 bra SC_BC4_DONE;\n\
    shl.b32 %r9, %r3, 2;\n\
    cvt.u64.u32 %rd6, %r9;\n\
    mov.u32 %r6, %ctaid.y;\n\
    mov.u32 %r7, %nctaid.y;\n\
SC_BC4_LOOP:\n\
    setp.ge.u32 %p1, %r6, %r5;\n\
    @%p1 bra SC_BC4_DONE;\n\
    cvt.u64.u32 %rd7, %r6;\n\
    shl.b64 %rd8, %rd7, 3;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.u64 %rd9, [%rd8];\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd1, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    mul.wide.u32 %rd10, %r6, %r4;\n\
    add.u64 %rd10, %rd10, %rd6;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.v4.f32 [%rd10], {%f1, %f1, %f1, %f1};\n\
    add.u32 %r6, %r6, %r7;\n\
    bra SC_BC4_LOOP;\n\
SC_BC4_DONE: ret;\n\
}\0";

/// Memo key: `(device, shape, source strides)`. The destination strides are
/// derived from the shape, so they add nothing to the identity of a plan.
#[cfg(feature = "cuda")]
type PlanKey = (i32, Vec<i64>, Vec<i64>);

/// A resolved plan plus its device-resident offset table.
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(crate) struct ResidentPlan {
    pub plan: RunPlan,
    /// Device pointer to `plan.src_offsets` as i64[outer].
    pub offsets_dev: u64,
}

/// Kill switch: `NSL_STRIDED_COPY_RUN=0` routes every view back through the
/// generic kernel. Same role as `NSL_FA_FWD_MMA` for the tensor-core attention
/// forward -- if a result moves when the fast path is disabled, the planner is
/// reading source elements the generic kernel would not.
///
/// Read once; the fast path is on a per-copy hot path and `std::env::var` takes
/// a process-wide lock.
#[cfg(feature = "cuda")]
fn run_path_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NSL_STRIDED_COPY_RUN").as_deref() != Ok("0"))
}

/// Resolve `(shape, src_strides)` to a plan, memoized per device.
///
/// The memo is keyed on the view metadata (at most a couple of dozen `i64`s),
/// NOT on the offset table: a hot shape such as the `[b, s, h, hd] ->
/// [b, h, s, hd]` head transpose has 32768 runs, and rebuilding or re-hashing
/// that table on every call would cost more than the copy it is meant to
/// accelerate. `None` (the view needs the generic kernel) is memoized too.
///
/// The table is allocated from `AllocPool::Persistent` and deliberately never
/// freed, exactly like `upload_meta_i64_cached`: the cache hands the same device
/// pointer to every later call with the same view.
///
/// Because the tables are never freed, admission is capped by
/// [`OFFSET_TABLE_BUDGET`]. A training step reuses a dozen shapes, so the budget
/// is never approached in practice — it exists so that a program which
/// materializes an unbounded variety of large views degrades to the generic
/// kernel instead of consuming device memory without limit.
#[cfg(feature = "cuda")]
pub(crate) fn resident_plan(
    shape: &[i64],
    src_strides: &[i64],
    dst_strides: &[i64],
) -> Option<ResidentPlan> {
    use std::collections::HashMap;
    use std::sync::Mutex;

    static CACHE: std::sync::OnceLock<Mutex<HashMap<PlanKey, Option<ResidentPlan>>>> =
        std::sync::OnceLock::new();
    static SPENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    if !run_path_enabled() {
        return None;
    }

    let device = super::inner::current_device_ordinal();
    let key = (device, shape.to_vec(), src_strides.to_vec());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().unwrap().get(&key) {
        return hit.clone();
    }

    // Build outside the lock so a large offset table never serializes lookups.
    let built = plan_run(shape, src_strides, dst_strides).and_then(|plan| {
        let bytes = std::mem::size_of_val(&plan.src_offsets[..]);
        // Reserve before allocating; a loser of the race simply falls back.
        if SPENT.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed) + bytes
            > OFFSET_TABLE_BUDGET
        {
            SPENT.fetch_sub(bytes, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        Some(plan)
    }).map(|plan| {
        let bytes = std::mem::size_of_val(&plan.src_offsets[..]);
        let dev = {
            use super::caching_allocator::{get_alloc_pool, set_alloc_pool, AllocPool};
            let prev = get_alloc_pool();
            set_alloc_pool(AllocPool::Persistent);
            let p = super::inner::alloc_managed(bytes);
            set_alloc_pool(prev);
            p
        };
        // Immediate, not stream-ordered: under CUDA-graph capture a deferred
        // copy would publish a pointer to unwritten memory (same reasoning as
        // `upload_meta_i64_cached`).
        super::inner::memcpy_htod_immediate(
            dev,
            plan.src_offsets.as_ptr() as *const std::ffi::c_void,
            bytes,
        );
        ResidentPlan { plan, offsets_dev: dev as u64 }
    });

    let mut guard = cache.lock().unwrap();
    match guard.entry(key) {
        std::collections::hash_map::Entry::Occupied(slot) => {
            // Another thread published first; drop our duplicate table and give
            // its bytes back to the budget.
            if let Some(ours) = &built {
                super::inner::free_managed(ours.offsets_dev as *mut std::ffi::c_void);
                SPENT.fetch_sub(
                    std::mem::size_of_val(&ours.plan.src_offsets[..]),
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            slot.get().clone()
        }
        std::collections::hash_map::Entry::Vacant(slot) => {
            slot.insert(built.clone());
            built
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Row-major strides, i.e. the layout `nsl_tensor_contiguous` materializes into.
    fn contig(shape: &[i64]) -> Vec<i64> {
        let mut st = vec![1i64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            st[i] = st[i + 1] * shape[i + 1];
        }
        st
    }

    /// Reference semantics of `nsl_tensor_contiguous`: walk every output element,
    /// read through the source strides.
    fn reference(shape: &[i64], src: &[i64]) -> Vec<i64> {
        let total: i64 = shape.iter().product();
        let dst = contig(shape);
        (0..total)
            .map(|flat| {
                let mut off = 0i64;
                for d in 0..shape.len() {
                    off += ((flat / dst[d]) % shape[d]) * src[d];
                }
                off
            })
            .collect()
    }

    /// What the run kernels actually read: run `o` element `i` reads
    /// `src_offsets[o] + i * (0 if broadcast else 1)`.
    fn plan_reads(p: &RunPlan) -> Vec<i64> {
        let mut out = Vec::new();
        for &base in &p.src_offsets {
            for i in 0..p.run_len {
                out.push(if p.broadcast { base } else { base + i });
            }
        }
        out
    }

    fn assert_plan_matches_reference(shape: &[i64], src: &[i64]) -> RunPlan {
        let plan = plan_run(shape, src, &contig(shape))
            .unwrap_or_else(|| panic!("expected a run plan for shape={shape:?} src={src:?}"));
        assert_eq!(
            plan.run_len * plan.outer,
            shape.iter().product::<i64>(),
            "plan does not cover every element: {plan:?}"
        );
        assert_eq!(
            plan_reads(&plan),
            reference(shape, src),
            "run plan reads different source elements than the generic kernel"
        );
        plan
    }

    // ---- the shapes measured in a Coder-50M step (batch 4, seq 1024) ----

    #[test]
    fn gqa_kv_expand_collapses_to_two_axes() {
        // k5.expand([b, nkv, n_rep, s, hd]) -- the GQA KV replication.
        let plan = assert_plan_matches_reference(
            &[4, 4, 2, 1024, 64],
            &[262144, 65536, 0, 64, 1],
        );
        assert_eq!((plan.run_len, plan.outer), (65536, 32));
        assert!(!plan.broadcast && plan.vec4);
    }

    #[test]
    fn head_transpose_keeps_head_dim_run() {
        // q.reshape([b, s, nh, hd]).transpose(1, 2)
        let plan = assert_plan_matches_reference(&[4, 8, 1024, 64], &[524288, 64, 512, 1]);
        assert_eq!((plan.run_len, plan.outer), (64, 32768));
        assert!(!plan.broadcast && plan.vec4);
    }

    #[test]
    fn scalar_broadcast_collapses_to_one_axis() {
        // full([1], x) expanded over a whole activation.
        let plan = assert_plan_matches_reference(&[4, 1024, 512], &[0, 0, 0]);
        assert_eq!((plan.run_len, plan.outer), (2097152, 1));
        assert!(plan.broadcast && plan.vec4);
        assert_eq!(plan.src_offsets, vec![0]);
    }

    #[test]
    fn per_token_broadcast_keeps_row_run() {
        // A [b, s] reduction (RMSNorm scale) broadcast back over d_model.
        let plan = assert_plan_matches_reference(&[4, 1024, 512], &[1024, 1, 0]);
        assert_eq!((plan.run_len, plan.outer), (512, 4096));
        assert!(plan.broadcast);
        // Source offsets step by 1 -- unaligned, but a broadcast reads a scalar
        // so vectorization only constrains the destination.
        assert!(plan.vec4);
        assert_eq!(&plan.src_offsets[..3], &[0, 1, 2]);
    }

    #[test]
    fn row_vector_broadcast_over_batch_and_seq() {
        // A [d_model] bias/weight broadcast over [b, s]: the innermost axis is a
        // real contiguous read (stride 1), so this is a *copy* whose runs all
        // start at 0 -- not the broadcast arm.
        let plan = assert_plan_matches_reference(&[4, 1024, 512], &[0, 0, 1]);
        assert_eq!((plan.run_len, plan.outer), (512, 4096));
        assert!(!plan.broadcast && plan.vec4);
        assert!(plan.src_offsets.iter().all(|&o| o == 0));
    }

    #[test]
    fn leading_broadcast_axes_merge_together() {
        let plan = assert_plan_matches_reference(&[4, 8, 1024, 64], &[0, 0, 64, 1]);
        assert_eq!((plan.run_len, plan.outer), (65536, 32));
        assert!(!plan.broadcast, "innermost stride is 1, so this is a copy");
        assert!(plan.src_offsets.iter().all(|&o| o == 0));
    }

    // ---- refusals: these must keep using the generic kernel ----

    #[test]
    fn transposed_inner_axis_refuses() {
        // [4, 512, 1024] from a [4, 1024, 512] source: innermost source stride
        // is 512, so there is no contiguous run to exploit.
        assert!(plan_run(&[4, 512, 1024], &[524288, 1, 512], &contig(&[4, 512, 1024])).is_none());
    }

    #[test]
    fn two_dimensional_transpose_refuses() {
        // embed.transpose(0, 1) for the weight-tied LM head.
        assert!(plan_run(&[49152, 512], &[1, 49152], &contig(&[49152, 512])).is_none());
    }

    #[test]
    fn short_run_refuses() {
        // run_len 16 < MIN_RUN: the offset table would rival the data. Source
        // stride 32 (not 16) keeps the axes from collapsing into one run.
        assert!(plan_run(&[4096, 16], &[32, 1], &contig(&[4096, 16])).is_none());
    }

    #[test]
    fn too_many_runs_refuses() {
        let shape = [MAX_OUTER + 1, MIN_RUN];
        assert!(plan_run(&shape, &[2 * MIN_RUN, 1], &contig(&shape)).is_none());
    }

    #[test]
    fn non_contiguous_destination_refuses() {
        // Guards the implicit `o * run_len` destination offset.
        assert!(plan_run(&[8, 64], &[64, 1], &[1024, 1]).is_none());
    }

    #[test]
    fn mismatched_metadata_refuses() {
        assert!(plan_run(&[4, 64], &[64, 1], &[64]).is_none());
        assert!(plan_run(&[], &[], &[]).is_none());
        assert!(plan_run(&[4, 0], &[0, 1], &[1, 1]).is_none());
    }

    // ---- structural properties ----

    #[test]
    fn already_contiguous_view_is_a_single_run() {
        let plan = assert_plan_matches_reference(&[4, 8, 1024, 64], &contig(&[4, 8, 1024, 64]));
        assert_eq!((plan.run_len, plan.outer), (2097152, 1));
    }

    #[test]
    fn size_one_axes_are_dropped_not_merged_wrong() {
        let plan = assert_plan_matches_reference(&[1, 4, 1, 1024, 1], &[0, 1024, 0, 1, 0]);
        assert_eq!((plan.run_len, plan.outer), (4096, 1));
    }

    #[test]
    fn unaligned_source_stride_disables_vec4_for_copies() {
        // A copy (not a broadcast) whose runs start at odd offsets cannot use
        // ld.global.v4.f32.
        let plan = assert_plan_matches_reference(&[64, 64], &[65, 1]);
        assert!(!plan.vec4, "odd run starts must fall back to scalar moves");
        let aligned = assert_plan_matches_reference(&[64, 64], &[68, 1]);
        assert!(aligned.vec4);
    }

    #[test]
    fn odd_run_length_disables_vec4() {
        // A 33-wide window into a 64-wide source: the run is contiguous but not
        // a multiple of four elements.
        let plan = assert_plan_matches_reference(&[16, 33], &[64, 1]);
        assert_eq!((plan.run_len, plan.outer), (33, 16));
        assert!(!plan.vec4);
    }

    #[test]
    fn geometry_covers_every_unit_and_respects_grid_limits() {
        for (shape, src) in [
            (vec![4, 4, 2, 1024, 64], vec![262144, 65536, 0, 64, 1]),
            (vec![4, 8, 1024, 64], vec![524288, 64, 512, 1]),
            (vec![4, 1024, 512], vec![0, 0, 0]),
            (vec![4, 1024, 512], vec![1024, 1, 0]),
            (vec![16, 33], vec![64, 1]),
        ] {
            let plan = plan_run(&shape, &src, &contig(&shape)).unwrap();
            let (grid, block) = plan.geometry();
            let units = if plan.vec4 { plan.run_len / 4 } else { plan.run_len };
            assert!(grid[0] * block[0] >= units, "grid.x too small for {plan:?}");
            assert!(
                (grid[0] - 1) * block[0] < units,
                "grid.x wastes a whole block for {plan:?}"
            );
            assert!(block[0] >= 32 && block[0] <= 256 && block[0] % 32 == 0);
            assert!(grid[1] <= MAX_GRID_Y && grid[1] >= 1);
        }
    }

    #[test]
    fn ptx_is_pure_ascii_and_nul_terminated() {
        assert!(
            STRIDED_COPY_RUN_PTX.is_ascii(),
            "non-ASCII in emitted PTX makes cuModuleLoadData return CUDA_ERROR_INVALID_PTX"
        );
        assert!(STRIDED_COPY_RUN_PTX.ends_with('\0'));
        assert!(
            !STRIDED_COPY_RUN_PTX.contains("mad.lo.u32"),
            "mad.lo.u32 is not valid in PTX ISA 7.0"
        );
    }

    #[test]
    fn every_plan_arm_has_an_entry_point() {
        for (broadcast, vec4) in [(false, false), (false, true), (true, false), (true, true)] {
            let plan = RunPlan {
                run_len: 64,
                outer: 1,
                src_offsets: vec![0],
                broadcast,
                vec4,
            };
            let name = plan.kernel_name().trim_end_matches('\0');
            assert!(
                STRIDED_COPY_RUN_PTX.contains(&format!(".visible .entry {name}(")),
                "{name} has no entry point in the module"
            );
        }
    }
}
