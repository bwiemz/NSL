//! M43: ZeRO optimizer state sharding.
//!
//! Implements ZeRO-1/2/3 parameter partitioning, gradient reduction, and
//! optimizer step coordination. Stage 1 partitions optimizer states only:
//! each rank updates only its assigned parameters, then all-gathers the results.

use std::sync::Mutex;

use crate::tensor::NslTensor;

/// ZeRO optimization stage.
/// - Stage1: partition optimizer states only
/// - Stage2: partition optimizer states + gradients
/// - Stage3: partition optimizer states + gradients + parameters
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZeROStage {
    Stage1,
    Stage2,
    Stage3,
}

impl ZeROStage {
    pub fn from_i64(v: i64) -> Option<Self> {
        match v {
            1 => Some(Self::Stage1),
            2 => Some(Self::Stage2),
            3 => Some(Self::Stage3),
            _ => None,
        }
    }
}

/// Partition parameters across data-parallel ranks using round-robin.
/// Returns a vec of vec where `partitions[rank]` contains the parameter indices
/// assigned to that rank.
///
/// Index-blind fallback: with real models the parameter BYTE sizes span four
/// orders of magnitude (embedding vs bias), so round-robin can pile the heavy
/// tensors onto one rank — prefer [`partition_params_balanced`], which the
/// codegen-emitted `nsl_zero_partition_bytes` uses.
pub fn partition_params(num_params: usize, world_size: usize) -> Vec<Vec<usize>> {
    let mut partitions = vec![Vec::new(); world_size];
    for i in 0..num_params {
        partitions[i % world_size].push(i);
    }
    partitions
}

/// P4 item 13: BYTE-BALANCED ownership. Greedy LPT (longest-processing-time):
/// visit params by (bytes DESC, index ASC) and assign each to the currently
/// lightest rank (ties → lowest rank). Classic bound: max rank load ≤ 4/3 of
/// optimal, and in practice transformer param sets balance to within one
/// small tensor. DETERMINISTIC: every rank derives the identical plan from
/// the identical (sizes, world_size) inputs — the total order above has no
/// ties left undecided, so ownership, broadcast roots, and the owner-gated
/// optimizer allocation agree across the clique with no communication.
pub fn partition_params_balanced(sizes: &[u64], world_size: usize) -> Vec<Vec<usize>> {
    let ws = world_size.max(1);
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let mut partitions = vec![Vec::new(); ws];
    let mut load = vec![0u64; ws];
    for idx in order {
        let lightest = (0..ws).min_by_key(|&r| (load[r], r)).unwrap();
        load[lightest] += sizes[idx];
        partitions[lightest].push(idx);
    }
    // Keep each rank's list in ascending param order (stable iteration for
    // the owner-gated update loops and debuggability).
    for p in &mut partitions {
        p.sort_unstable();
    }
    partitions
}

/// Flatten a partition into an owner lookup: `owner_of[i]` = the rank that
/// owns param `i`. Every index is covered exactly once by construction.
pub fn owners_from_partition(partitions: &[Vec<usize>], num_params: usize) -> Vec<i32> {
    let mut owner_of = vec![0i32; num_params];
    for (rank, part) in partitions.iter().enumerate() {
        for &idx in part {
            if idx < num_params {
                owner_of[idx] = rank as i32;
            }
        }
    }
    owner_of
}

// ---------------------------------------------------------------------------
// ZeRO FFI implementation
// ---------------------------------------------------------------------------

static ZERO_CTX: Mutex<Option<ZeROContext>> = Mutex::new(None);

/// D3 anti-vacuity (review): with replicated data a silently no-op reduce
/// is invisible to the parity gate (identical grads make sum/ws == g
/// either way). These count backend collectives that ACTUALLY ran (rc==0,
/// ws>1 path only); the gate asserts exact totals so a degenerate
/// identity path or an argv-forwarding regression that drops --devices
/// cannot pass green. Reported at exit under NSL_ZERO_COUNTER=1.
pub static ZERO_ALL_REDUCE_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static ZERO_BROADCAST_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// D3 v2 memory-win proof: total optimizer-moment ELEMENTS this rank
/// actually allocated. v1 allocated full m/v on every rank (owner-gated the
/// UPDATE only); v2 gates the ALLOCATION on ownership so each rank holds
/// ~1/world_size of the moment surface. The G3 gate asserts per-rank totals
/// shrink to ~1/N — a numeric parity gate alone can't see this (owned-only
/// allocation is invisible to the loss because non-owners never read their
/// non-owned m/v). Bumped once per owned moment tensor at allocation.
pub static ZERO_OPTIM_ELEMS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Read `(rank, world_size, all_reduce_count, broadcast_count, optim_elems)`
/// for the atexit report. rank/ws are -1 when no context was initialized.
pub fn zero_counter_snapshot() -> (i64, i64, u64, u64, u64, u64) {
    let guard = ZERO_CTX.lock().unwrap();
    let (rank, ws) = guard
        .as_ref()
        .map(|c| (c.rank as i64, c.world_size as i64))
        .unwrap_or((-1, -1));
    (
        rank,
        ws,
        ZERO_ALL_REDUCE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ZERO_BROADCAST_COUNT.load(std::sync::atomic::Ordering::Relaxed),
        ZERO_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed),
        ZERO_BUCKET_MEMBERS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// D3 v2: record that an owned optimizer-moment tensor of `tensor_ptr` was
/// allocated, adding its element count to this rank's running total. Emitted
/// by codegen inside the owner-gated allocation branch (once per m and per v
/// tensor). A null pointer (the non-owned placeholder) contributes nothing,
/// so a mis-wired gate that noted non-owned slots would over-count and trip
/// the G3 gate. Returns the running total (for tests/debug).
#[no_mangle]
pub extern "C" fn nsl_zero_note_optim_alloc(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return ZERO_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed) as i64;
    }
    let numel = crate::tensor::NslTensor::from_ptr(tensor_ptr).len.max(0) as u64;
    (ZERO_OPTIM_ELEMS.fetch_add(numel, std::sync::atomic::Ordering::Relaxed) + numel) as i64
}

struct ZeROContext {
    stage: ZeROStage,
    rank: usize,
    world_size: usize,
    /// Which parameter indices this rank owns (populated by nsl_zero_partition).
    owned_params: Vec<usize>,
    /// P4 item 13: param index → owning rank, for EVERY param. The broadcast
    /// root in `nsl_zero_sync_params` MUST come from here (not `i % ws`) so
    /// the sync roots always agree with the ownership plan, whichever
    /// partitioner produced it.
    owner_of: Vec<i32>,
    /// Total number of params (set during partition).
    num_params: usize,
    /// D3: the real collective backend. `None` iff world_size == 1 (single
    /// rank: every collective degenerates to identity). P4 item 14: either
    /// the CPU-shm SimulatedBackend (default) or, with NSL_COLLECTIVES=nccl
    /// on an nccl-featured build, the CUDA-aware NcclBackend.
    backend: Option<Box<dyn crate::tensor_parallel::collective::CollectiveBackend>>,
    /// True when `backend` takes DEVICE pointers (NCCL): GPU-resident grads
    /// and params are passed directly, no host staging, no -5 refusal.
    cuda_aware: bool,
    /// Review H1: CPU-RESIDENT tensor groups always use this host (CPU-shm)
    /// backend — a CUDA-aware `backend` (NCCL) must never receive heap
    /// pointers (ncclInvalidArgument abort on real multi-GPU hardware; even
    /// GPU models carry a CPU grad group). Shares the same shm mapping, so
    /// barrier generations stay consistent (all ranks run the identical
    /// collective sequence in lockstep).
    host_backend: Option<crate::tensor_parallel::collective::SimulatedBackend>,
}

// SAFETY: the raw shm pointer inside SimulatedBackend is only touched
// under the ZERO_CTX mutex on the (single-threaded) training path — the
// same argument as TpContext's Send impl.
unsafe impl Send for ZeROContext {}

/// P4 item 14: construct the NCCL backend (nccl-featured builds only; the
/// stub refuses with the rebuild hint).
#[cfg(feature = "nccl")]
fn make_nccl_backend(
    rank: usize,
    ws: usize,
    shm_ptr: *mut u8,
    shm_len: usize,
) -> Result<Box<dyn crate::tensor_parallel::collective::CollectiveBackend>, String> {
    crate::tensor_parallel::collective::NcclBackend::new(rank as i32, ws as i32, shm_ptr, shm_len)
        .map(|b| Box::new(b) as Box<dyn crate::tensor_parallel::collective::CollectiveBackend>)
}

#[cfg(not(feature = "nccl"))]
fn make_nccl_backend(
    _rank: usize,
    _ws: usize,
    _shm_ptr: *mut u8,
    _shm_len: usize,
) -> Result<Box<dyn crate::tensor_parallel::collective::CollectiveBackend>, String> {
    Err("this runtime was built WITHOUT the nccl feature — rebuild with \
         `--features nccl` (libnccl.so on the linker/loader path), or drop \
         --collectives nccl"
        .to_string())
}

/// P4 item 14: the CUDA-aware TEST backend (device-pointer API staged through
/// the CPU-shm reduce) — validates the ZeRO GPU plumbing on one GPU.
#[cfg(feature = "cuda")]
fn make_sim_gpu_backend(
    rank: usize,
    ws: usize,
    shm_ptr: *mut u8,
    shm_len: usize,
) -> Result<Box<dyn crate::tensor_parallel::collective::CollectiveBackend>, String> {
    Ok(Box::new(
        crate::tensor_parallel::collective::GpuStagedBackend::new(
            rank as i32,
            ws as i32,
            shm_ptr,
            shm_len,
        ),
    ))
}

#[cfg(not(feature = "cuda"))]
fn make_sim_gpu_backend(
    _rank: usize,
    _ws: usize,
    _shm_ptr: *mut u8,
    _shm_len: usize,
) -> Result<Box<dyn crate::tensor_parallel::collective::CollectiveBackend>, String> {
    Err("--collectives sim-gpu needs a cuda-featured build".to_string())
}

/// Initialize ZeRO optimizer sharding (D3: REAL — builds the CPU-shm
/// SimulatedBackend from the `--devices N` spawner's env protocol).
///
/// `stage`: 1/2/3 (codegen refuses 2/3 before emitting — the guard here
/// is a belt), `world_size`: compile-time world size baked by codegen.
/// Env: NSL_LOCAL_RANK (rank, default 0), NSL_TP_SHM_PATH (required when
/// world_size > 1), NSL_SIMULATED_TP=0 refused loudly (no real transport
/// exists in this build).
/// Returns 0 on success, -1 already-initialized/invalid, -2 refused, -3
/// missing shm path.
#[no_mangle]
pub extern "C" fn nsl_zero_init(stage: i64, world_size: i64) -> i64 {
    let mut guard = ZERO_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    let Some(s) = ZeROStage::from_i64(stage) else {
        return -1;
    };
    let rank = std::env::var("NSL_LOCAL_RANK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let ws = (world_size.max(1)) as usize;
    // Review L7: a clamped out-of-range rank would silently alias the last
    // rank (duplicate segment claims, overlapping shm writes). Refuse.
    if rank >= ws {
        eprintln!(
            "nsl_zero_init: NSL_LOCAL_RANK={rank} out of range for              world_size={ws} — refusing"
        );
        return -1;
    }

    let simulated: bool = std::env::var("NSL_SIMULATED_TP")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(true);
    if !simulated && ws > 1 {
        eprintln!(
            "nsl_zero_init: NSL_SIMULATED_TP=0 requests a real collective \
             backend, but no NCCL/inter-device transport is built into this \
             runtime — refusing rather than silently simulating."
        );
        return -2;
    }

    // P4 item 14: collective-backend selection. Default = the CPU-shm
    // SimulatedBackend; NSL_COLLECTIVES=nccl requests real CUDA-aware NCCL
    // collectives (set by `nsl run --collectives nccl`, requires an
    // nccl-featured build). Anything else refuses loudly.
    let requested = std::env::var("NSL_COLLECTIVES").unwrap_or_else(|_| "sim".to_string());
    #[derive(PartialEq)]
    enum WantBackend {
        Sim,
        SimGpu,
        Nccl,
    }
    let want = match requested.as_str() {
        "sim" | "" => WantBackend::Sim,
        "sim-gpu" => WantBackend::SimGpu,
        "nccl" => WantBackend::Nccl,
        other => {
            eprintln!(
                "nsl_zero_init: unknown NSL_COLLECTIVES backend '{other}' \
                 (expected 'sim', 'sim-gpu', or 'nccl')"
            );
            return -2;
        }
    };

    let (backend, cuda_aware): (
        Option<Box<dyn crate::tensor_parallel::collective::CollectiveBackend>>,
        bool,
    ) = if ws > 1 {
        let Ok(shm_path) = std::env::var("NSL_TP_SHM_PATH") else {
            eprintln!(
                "nsl_zero_init: world_size={ws} but NSL_TP_SHM_PATH is not \
                 set — run through `nsl run --devices {ws}` (the spawner \
                 creates the shared-memory file and sets the rank env)"
            );
            return -3;
        };
        let (shm_ptr, shm_len) = crate::tensor_parallel::ffi::open_shm(&shm_path);
        match want {
            WantBackend::Nccl => match make_nccl_backend(rank, ws, shm_ptr, shm_len) {
                Ok(b) => {
                    eprintln!(
                        "[zero] rank {rank}: NCCL communicator up \
                         (ws={ws}, CUDA-aware collectives)"
                    );
                    (Some(b), true)
                }
                Err(e) => {
                    eprintln!(
                        "nsl_zero_init: NCCL backend init failed on rank {rank}: {e}"
                    );
                    return -2;
                }
            },
            WantBackend::SimGpu => match make_sim_gpu_backend(rank, ws, shm_ptr, shm_len) {
                Ok(b) => {
                    eprintln!(
                        "[zero] rank {rank}: sim-gpu staged collectives up \
                         (ws={ws}, CUDA-aware TEST backend)"
                    );
                    (Some(b), true)
                }
                Err(e) => {
                    eprintln!(
                        "nsl_zero_init: sim-gpu backend init failed on rank {rank}: {e}"
                    );
                    return -2;
                }
            },
            WantBackend::Sim => (
                Some(Box::new(
                    crate::tensor_parallel::collective::SimulatedBackend::new(
                        rank as i32,
                        ws as i32,
                        shm_ptr,
                        shm_len,
                    ),
                )
                    as Box<dyn crate::tensor_parallel::collective::CollectiveBackend>),
                false,
            ),
        }
    } else {
        // Single rank: NCCL would be a 1-rank identity clique — skip it.
        (None, false)
    };

    let host_backend = if ws > 1 {
        std::env::var("NSL_TP_SHM_PATH").ok().map(|p| {
            let (shm_ptr, shm_len) = crate::tensor_parallel::ffi::open_shm(&p);
            crate::tensor_parallel::collective::SimulatedBackend::new(
                rank as i32,
                ws as i32,
                shm_ptr,
                shm_len,
            )
        })
    } else {
        None
    };
    *guard = Some(ZeROContext {
        stage: s,
        rank,
        world_size: ws,
        owned_params: Vec::new(),
        owner_of: Vec::new(),
        num_params: 0,
        backend,
        cuda_aware,
        host_backend,
    });
    0
}

/// Partition parameters for ZeRO. Uses round-robin partitioning (index-blind
/// fallback — codegen emits `nsl_zero_partition_bytes` instead).
/// Returns the number of parameters this rank owns, or -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_partition(num_params: i64) -> i64 {
    let mut guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_mut() else {
        return -1;
    };

    let partitions = partition_params(num_params as usize, ctx.world_size);
    let my_rank = ctx.rank.min(partitions.len() - 1);
    let my_params = partitions[my_rank].clone();
    let count = my_params.len() as i64;

    ctx.owner_of = owners_from_partition(&partitions, num_params as usize);
    ctx.owned_params = my_params;
    ctx.num_params = num_params as usize;

    // L6: round-robin plans depend only on (num_params, ws) — hash them
    // anyway so a ws/param-count divergence still refuses.
    let sizes = vec![0u64; num_params.max(0) as usize];
    if !verify_plan_across_ranks(ctx, &sizes) {
        return -1;
    }

    count
}

/// P4 item 13: BYTE-BALANCED partition. Reads each parameter's byte size from
/// the runtime param list (identical on every rank — the plan needs no
/// communication) and assigns ownership by greedy LPT so per-rank optimizer
/// work and moment memory track ~1/N in BYTES, not tensor count. A null list
/// slot contributes size 0 (still assigned an owner, for total coverage).
/// Returns the number of parameters this rank owns, or -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_partition_bytes(params_list_ptr: i64, num_params: i64) -> i64 {
    let mut guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_mut() else {
        return -1;
    };
    let list_ptr = params_list_ptr as *const crate::list::NslList;
    if list_ptr.is_null() {
        return -1;
    }
    let list = unsafe { &*list_ptr };
    let n = (num_params.max(0) as usize).min(list.len as usize);

    let mut sizes = vec![0u64; num_params.max(0) as usize];
    for (i, size) in sizes.iter_mut().enumerate().take(n) {
        let tensor_raw = unsafe { *list.data.add(i) };
        if tensor_raw != 0 {
            let t = unsafe { &*(tensor_raw as *const NslTensor) };
            *size = t.data_byte_size() as u64;
        }
    }

    let partitions = partition_params_balanced(&sizes, ctx.world_size);
    let my_rank = ctx.rank.min(partitions.len() - 1);
    let my_params = partitions[my_rank].clone();
    let count = my_params.len() as i64;

    ctx.owner_of = owners_from_partition(&partitions, sizes.len());
    ctx.owned_params = my_params;
    ctx.num_params = sizes.len();

    // L6 belt: refuse loudly if any rank derived a different plan (the
    // emitted call site asserts rc >= 0, so -1 aborts the run).
    if !verify_plan_across_ranks(ctx, &sizes) {
        return -1;
    }

    count
}

/// L6 belt: verify every rank computed the IDENTICAL partition plan.
///
/// The plan is derived rank-locally (no communication) from param byte
/// sizes; if param enumeration or placement ever diverged across ranks
/// (mixed CPU/GPU placement, a rank-dependent model edit), the owner gates
/// would silently train a TORN model — each rank updating a different
/// param set against different moments. Rank 0 broadcasts an FNV-1a hash
/// of (stage, ws, num_params, sizes, owner_of); every rank compares.
/// Deterministic call point: both partition FFIs run once at train setup
/// on every rank, so the broadcast is symmetric.
fn verify_plan_across_ranks(ctx: &ZeROContext, sizes: &[u64]) -> bool {
    use crate::tensor_parallel::collective::CollectiveBackend;
    if ctx.world_size <= 1 {
        return true;
    }
    // Unit tests build contexts without a backend; the SPMD spawner always
    // provides the host backend at ws>1.
    let Some(host) = ctx.host_backend.as_ref() else {
        return true;
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |v: u64| {
        for b in v.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    mix(ctx.stage as u64);
    mix(ctx.world_size as u64);
    mix(sizes.len() as u64);
    for &s in sizes {
        mix(s);
    }
    for &o in &ctx.owner_of {
        mix(o as u64);
    }
    let local = h;
    let mut buf = local.to_le_bytes();
    let rc = host.broadcast(
        buf.as_mut_ptr() as *mut std::ffi::c_void,
        buf.len(),
        crate::tensor_parallel::collective::DTYPE_I8,
        0,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        eprintln!(
            "nsl: zero plan verification broadcast failed rc={rc} on rank {}",
            ctx.rank
        );
        return false;
    }
    let root = u64::from_le_bytes(buf);
    if root != local {
        eprintln!(
            "nsl: ZeRO partition plan MISMATCH on rank {}: local hash \
             {local:#018x} != rank-0 hash {root:#018x} — ranks disagree on \
             param sizes or ownership; refusing to train a torn model",
            ctx.rank
        );
        return false;
    }
    true
}

/// Check if this rank owns the given parameter index.
/// Returns 1 if owned, 0 if not, -1 if ZeRO not initialized.
#[no_mangle]
pub extern "C" fn nsl_zero_owns_param(param_idx: i64) -> i64 {
    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };

    if ctx.owned_params.contains(&(param_idx as usize)) {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// P4 item 15: bucketed collectives.
// ---------------------------------------------------------------------------
//
// One collective per small parameter wastes latency (shm: 2 spin-barriers
// per tensor; NCCL: a launch + watchdog'd stream sync per tensor). Instead,
// same-(dtype, device) gradients are FLATTENED in param-index order — which
// is model/layer order, so buckets align with CSLA layer ranges — into
// buckets capped at NSL_ZERO_BUCKET_MB (default 25; 0 = per-tensor), and one
// collective moves each bucket. Param sync buckets additionally by OWNER so
// each bucket broadcasts from its owning rank. Bit-exactness is preserved:
// the backend reduces element-wise in the same fixed rank order, and the
// scale-by-1/ws is applied to the same values.

/// Review H1: the backend for HOST-buffer collectives — the CPU-shm backend
/// when one exists (always, at ws>1), else the main backend (ws==1 never
/// gets here). CUDA-aware main backends must never see heap pointers.
fn host_or_main(ctx: &ZeROContext) -> &dyn crate::tensor_parallel::collective::CollectiveBackend {
    match &ctx.host_backend {
        Some(h) => h,
        None => ctx
            .backend
            .as_deref()
            .expect("ws > 1 implies a backend"),
    }
}

/// Bucket byte cap (0 disables bucketing). Fractional MB accepted — the
/// stage-2 chunking gates use sub-MB caps to force multi-sub-group and
/// intra-tensor chunk coverage on small fixtures.
fn zero_bucket_cap_bytes() -> usize {
    std::env::var("NSL_ZERO_BUCKET_MB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|mb| (mb.max(0.0) * 1024.0 * 1024.0) as usize)
        .unwrap_or(25 * 1024 * 1024)
}

/// Anti-vacuity: Σ tensors that traveled inside a bucketed collective.
pub static ZERO_BUCKET_MEMBERS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// P4 item 16 (ZeRO-2): reduce_scatter collectives that actually ran — the
/// stage-2 gate asserts this is nonzero AND all_reduce stays 0 for grads.
pub static ZERO_REDUCE_SCATTER_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// One eligible tensor (or, on the stage-2 scatter path ONLY, one byte-range
/// CHUNK of a tensor) for bucketing.
#[derive(Clone, Copy)]
struct BucketItem {
    /// Param index (owner lookup for stage-2 segmenting / owner broadcasts).
    idx: usize,
    raw: i64,
    len: usize,
    bytes: usize,
    /// Byte offset into the tensor's data. Nonzero only for stage-2 scatter
    /// chunks — the all-reduce/broadcast bucket paths always carry whole
    /// tensors (their GPU scale step is per-tensor and would double-scale
    /// a chunked one).
    off: usize,
}

/// Group `items` (already same dtype/device/owner class) into ≤cap buckets,
/// preserving order; every bucket holds ≥1 member.
fn split_buckets(items: &[BucketItem], cap: usize) -> Vec<Vec<BucketItem>> {
    let mut out: Vec<Vec<BucketItem>> = Vec::new();
    let mut cur: Vec<BucketItem> = Vec::new();
    let mut cur_bytes = 0usize;
    for &it in items {
        if !cur.is_empty() && cur_bytes + it.bytes > cap {
            out.push(std::mem::take(&mut cur));
            cur_bytes = 0;
        }
        cur_bytes += it.bytes;
        cur.push(it);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// CPU bucket all-reduce + average: pack → ONE collective → scale flat →
/// unpack. `dtype` is 0 (f64) or 1 (f32).
fn reduce_bucket_cpu(
    ctx: &ZeROContext,
    bucket: &[BucketItem],
    dtype: u16,
    inv_ws: f64,
) -> i64 {
    let backend = host_or_main(ctx);
    let total_bytes: usize = bucket.iter().map(|b| b.bytes).sum();
    let total_elems: usize = bucket.iter().map(|b| b.len).sum();
    let mut flat = vec![0u8; total_bytes];
    let mut off = 0usize;
    for b in bucket {
        unsafe {
            std::ptr::copy_nonoverlapping(
                (b.raw as *const NslTensor).read().data as *const u8,
                flat.as_mut_ptr().add(off),
                b.bytes,
            )
        };
        off += b.bytes;
    }
    let rc = backend.all_reduce_sum(
        flat.as_ptr() as *const std::ffi::c_void,
        flat.as_mut_ptr() as *mut std::ffi::c_void,
        total_elems,
        dtype as crate::tensor_parallel::collective::DtypeId,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        return -1;
    }
    ZERO_ALL_REDUCE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ZERO_BUCKET_MEMBERS.fetch_add(bucket.len() as u64, std::sync::atomic::Ordering::Relaxed);
    if dtype == 0 {
        let p = flat.as_mut_ptr() as *mut f64;
        for j in 0..total_elems {
            unsafe { *p.add(j) *= inv_ws };
        }
    } else {
        let p = flat.as_mut_ptr() as *mut f32;
        let s = inv_ws as f32;
        for j in 0..total_elems {
            unsafe { *p.add(j) *= s };
        }
    }
    let mut off = 0usize;
    for b in bucket {
        unsafe {
            std::ptr::copy_nonoverlapping(
                flat.as_ptr().add(off),
                (b.raw as *const NslTensor).read().data as *mut u8,
                b.bytes,
            )
        };
        off += b.bytes;
    }
    0
}

/// GPU (CUDA-aware) bucket all-reduce + average: DtoD pack into a device
/// bucket → ONE device collective → DtoD unpack → per-tensor on-device scale.
#[cfg(feature = "cuda")]
fn reduce_bucket_gpu(ctx: &ZeROContext, bucket: &[BucketItem], inv_ws: f64) -> i64 {
    let backend = ctx.backend.as_deref().expect("ws > 1 implies a backend");
    let total_bytes: usize = bucket.iter().map(|b| b.bytes).sum();
    let total_elems: usize = bucket.iter().map(|b| b.len).sum();
    crate::cuda::inner::ensure_context();
    let dev = crate::cuda::inner::alloc_managed(total_bytes);
    let mut off = 0usize;
    for b in bucket {
        let t = unsafe { &*(b.raw as *const NslTensor) };
        crate::cuda::inner::memcpy_dtod(
            unsafe { (dev as *mut u8).add(off) } as *mut std::ffi::c_void,
            t.data as *const std::ffi::c_void,
            b.bytes,
        );
        off += b.bytes;
    }
    let rc = backend.all_reduce_sum(
        dev as *const std::ffi::c_void,
        dev,
        total_elems,
        crate::tensor_parallel::collective::DTYPE_F32,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        crate::cuda::inner::free_managed(dev);
        return -1;
    }
    ZERO_ALL_REDUCE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ZERO_BUCKET_MEMBERS.fetch_add(bucket.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let mut off = 0usize;
    for b in bucket {
        let t = unsafe { &*(b.raw as *const NslTensor) };
        crate::cuda::inner::memcpy_dtod(
            t.data,
            unsafe { (dev as *const u8).add(off) } as *const std::ffi::c_void,
            b.bytes,
        );
        off += b.bytes;
        crate::tensor::nsl_tensor_mul_scalar_inplace(b.raw, inv_ws);
    }
    crate::cuda::inner::free_managed(dev);
    0
}

/// M3 (stage-2 shm wall): split one (dtype, device) gradient group into
/// sub-groups whose PADDED scatter buffer (`max owner segment × ws`) stays
/// ≤ `cap`. An unbucketed 1B-scale group blows past the 64MB CPU-shm slot
/// (loud -4, but a wall). Tensors larger than `cap / ws` are additionally
/// CHUNKED by byte range (chunks keep the tensor's `idx`, hence its owner),
/// so even a single 64MB embedding gradient scatters in bounded pieces.
/// Deterministic: chunking and grouping depend only on the param plan, so
/// every rank builds identical sub-groups with no communication. Bit-exact:
/// the reduction is element-wise in fixed rank order regardless of which
/// sub-group an element rides in.
fn split_scatter_subgroups(
    ctx: &ZeROContext,
    items: &[BucketItem],
    cap: usize,
    esz: usize,
) -> Vec<Vec<BucketItem>> {
    let ws = ctx.world_size;
    // Largest single chunk that still fits a padded sub-group on its own:
    // cap/ws, floored to an element boundary (minimum one element).
    let item_cap = std::cmp::max(esz, cap / ws / esz * esz);
    let mut chunks: Vec<BucketItem> = Vec::new();
    for &it in items {
        if it.bytes <= item_cap {
            chunks.push(it);
            continue;
        }
        let mut off = 0usize;
        while off < it.bytes {
            let take = item_cap.min(it.bytes - off);
            chunks.push(BucketItem {
                idx: it.idx,
                raw: it.raw,
                len: take / esz,
                bytes: take,
                off: it.off + off,
            });
            off += take;
        }
    }
    // Greedy grouping under the padded-size cap.
    let mut out: Vec<Vec<BucketItem>> = Vec::new();
    let mut cur: Vec<BucketItem> = Vec::new();
    let mut seg = vec![0usize; ws];
    for c in chunks {
        let owner = ctx.owner_of.get(c.idx).copied().unwrap_or(0) as usize;
        let owner = owner.min(ws - 1);
        let new_max = seg
            .iter()
            .enumerate()
            .map(|(r, &b)| if r == owner { b + c.bytes } else { b })
            .max()
            .unwrap_or(0);
        if !cur.is_empty() && new_max * ws > cap {
            out.push(std::mem::take(&mut cur));
            seg.iter_mut().for_each(|b| *b = 0);
            seg[owner] = c.bytes;
        } else {
            seg[owner] += c.bytes;
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// P4 item 16 (ZeRO-2): owner-segmented reduce_scatter over one (dtype,
/// device) group. Layout: `[rank0's owned grads .. pad | rank1's .. pad | …]`
/// with every rank segment padded to the max segment size, so the existing
/// `reduce_scatter_sum` (count divisible by ws, rank r receives slice r)
/// delivers each rank EXACTLY its owned gradients, summed — gradient
/// partitioning with (N-1)/N the traffic of an all-reduce. Non-owned grad
/// tensors keep their LOCAL (unsummed) values; only the owner-gated
/// optimizer reads gradients, and it reads owned slots only.
fn reduce_scatter_group_cpu(
    ctx: &ZeROContext,
    items: &[BucketItem],
    dtype: u16,
    inv_ws: f64,
) -> i64 {
    let backend = host_or_main(ctx);
    let ws = ctx.world_size;
    let esz = if dtype == 0 { 8 } else { 4 };
    // Per-rank owned member lists (ascending param order preserved).
    let mut segs: Vec<Vec<BucketItem>> = vec![Vec::new(); ws];
    for &it in items {
        let owner = ctx.owner_of.get(it.idx).copied().unwrap_or(0) as usize;
        segs[owner.min(ws - 1)].push(it);
    }
    let seg_bytes: Vec<usize> = segs.iter().map(|s| s.iter().map(|b| b.bytes).sum()).collect();
    let max_seg = seg_bytes.iter().copied().max().unwrap_or(0);
    if max_seg == 0 {
        return 0;
    }
    let total = max_seg * ws;
    let mut flat = vec![0u8; total];
    for (r, seg) in segs.iter().enumerate() {
        let mut off = r * max_seg;
        for b in seg {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ((b.raw as *const NslTensor).read().data as *const u8).add(b.off),
                    flat.as_mut_ptr().add(off),
                    b.bytes,
                )
            };
            off += b.bytes;
        }
    }
    let mut out = vec![0u8; max_seg];
    let rc = backend.reduce_scatter_sum(
        flat.as_ptr() as *const std::ffi::c_void,
        out.as_mut_ptr() as *mut std::ffi::c_void,
        total / esz,
        dtype as crate::tensor_parallel::collective::DtypeId,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        return -1;
    }
    ZERO_REDUCE_SCATTER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ZERO_BUCKET_MEMBERS.fetch_add(items.len() as u64, std::sync::atomic::Ordering::Relaxed);
    // Average + unpack THIS rank's owned segment.
    let my = ctx.rank.min(ws - 1);
    let my_elems = seg_bytes[my] / esz;
    if dtype == 0 {
        let p = out.as_mut_ptr() as *mut f64;
        for j in 0..my_elems {
            unsafe { *p.add(j) *= inv_ws };
        }
    } else {
        let p = out.as_mut_ptr() as *mut f32;
        let sc = inv_ws as f32;
        for j in 0..my_elems {
            unsafe { *p.add(j) *= sc };
        }
    }
    let mut off = 0usize;
    for b in &segs[my] {
        unsafe {
            std::ptr::copy_nonoverlapping(
                out.as_ptr().add(off),
                ((b.raw as *const NslTensor).read().data as *mut u8).add(b.off),
                b.bytes,
            )
        };
        off += b.bytes;
    }
    0
}

/// GPU (CUDA-aware) sibling of [`reduce_scatter_group_cpu`]: same
/// owner-segmented padded layout through device buffers.
#[cfg(feature = "cuda")]
fn reduce_scatter_group_gpu(ctx: &ZeROContext, items: &[BucketItem], inv_ws: f64) -> i64 {
    let backend = ctx.backend.as_deref().expect("ws > 1 implies a backend");
    let ws = ctx.world_size;
    let mut segs: Vec<Vec<BucketItem>> = vec![Vec::new(); ws];
    for &it in items {
        let owner = ctx.owner_of.get(it.idx).copied().unwrap_or(0) as usize;
        segs[owner.min(ws - 1)].push(it);
    }
    let seg_bytes: Vec<usize> = segs.iter().map(|s| s.iter().map(|b| b.bytes).sum()).collect();
    let max_seg = seg_bytes.iter().copied().max().unwrap_or(0);
    if max_seg == 0 {
        return 0;
    }
    let total = max_seg * ws;
    crate::cuda::inner::ensure_context();
    let dev = crate::cuda::inner::alloc_managed(total);
    crate::cuda::inner::memset_d8(dev, total);
    for (r, seg) in segs.iter().enumerate() {
        let mut off = r * max_seg;
        for b in seg {
            let t = unsafe { &*(b.raw as *const NslTensor) };
            crate::cuda::inner::memcpy_dtod(
                unsafe { (dev as *mut u8).add(off) } as *mut std::ffi::c_void,
                unsafe { (t.data as *const u8).add(b.off) } as *const std::ffi::c_void,
                b.bytes,
            );
            off += b.bytes;
        }
    }
    let out = crate::cuda::inner::alloc_managed(max_seg);
    let rc = backend.reduce_scatter_sum(
        dev as *const std::ffi::c_void,
        out,
        total / 4,
        crate::tensor_parallel::collective::DTYPE_F32,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        crate::cuda::inner::free_managed(dev);
        crate::cuda::inner::free_managed(out);
        return -1;
    }
    ZERO_REDUCE_SCATTER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ZERO_BUCKET_MEMBERS.fetch_add(items.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let my = ctx.rank.min(ws - 1);
    // Average on the STAGING buffer, then unpack — per-chunk correctness
    // (a per-tensor scale after unpack would double-scale a tensor whose
    // chunks landed in different sub-groups) and one kernel per group
    // instead of one per tensor. Same values × same f32 scalar → bit-exact
    // with the old order.
    let my_elems = seg_bytes[my] / 4;
    if my_elems > 0 {
        crate::cuda::gpu_scale_raw_f32(out, my_elems, inv_ws as f32);
    }
    let mut off = 0usize;
    for b in &segs[my] {
        let t = unsafe { &*(b.raw as *const NslTensor) };
        crate::cuda::inner::memcpy_dtod(
            unsafe { (t.data as *mut u8).add(b.off) } as *mut std::ffi::c_void,
            unsafe { (out as *const u8).add(off) } as *const std::ffi::c_void,
            b.bytes,
        );
        off += b.bytes;
    }
    crate::cuda::inner::free_managed(dev);
    crate::cuda::inner::free_managed(out);
    0
}

/// CPU bucket broadcast from `owner`: every rank packs its current bytes,
/// ONE collective, every rank unpacks (the root re-reads its own bytes).
fn broadcast_bucket_cpu(ctx: &ZeROContext, bucket: &[BucketItem], dtype: u16, owner: i32) -> i64 {
    let backend = host_or_main(ctx);
    let total_bytes: usize = bucket.iter().map(|b| b.bytes).sum();
    let total_elems: usize = bucket.iter().map(|b| b.len).sum();
    let mut flat = vec![0u8; total_bytes];
    let mut off = 0usize;
    for b in bucket {
        unsafe {
            std::ptr::copy_nonoverlapping(
                (b.raw as *const NslTensor).read().data as *const u8,
                flat.as_mut_ptr().add(off),
                b.bytes,
            )
        };
        off += b.bytes;
    }
    let rc = backend.broadcast(
        flat.as_mut_ptr() as *mut std::ffi::c_void,
        total_elems,
        dtype as crate::tensor_parallel::collective::DtypeId,
        owner,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        return -1;
    }
    ZERO_BROADCAST_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ZERO_BUCKET_MEMBERS.fetch_add(bucket.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let mut off = 0usize;
    for b in bucket {
        unsafe {
            std::ptr::copy_nonoverlapping(
                flat.as_ptr().add(off),
                (b.raw as *const NslTensor).read().data as *mut u8,
                b.bytes,
            )
        };
        off += b.bytes;
    }
    0
}

/// GPU (CUDA-aware) bucket broadcast from `owner` via a device bucket.
#[cfg(feature = "cuda")]
fn broadcast_bucket_gpu(ctx: &ZeROContext, bucket: &[BucketItem], owner: i32) -> i64 {
    let backend = ctx.backend.as_deref().expect("ws > 1 implies a backend");
    let total_bytes: usize = bucket.iter().map(|b| b.bytes).sum();
    let total_elems: usize = bucket.iter().map(|b| b.len).sum();
    crate::cuda::inner::ensure_context();
    let dev = crate::cuda::inner::alloc_managed(total_bytes);
    let mut off = 0usize;
    for b in bucket {
        let t = unsafe { &*(b.raw as *const NslTensor) };
        crate::cuda::inner::memcpy_dtod(
            unsafe { (dev as *mut u8).add(off) } as *mut std::ffi::c_void,
            t.data as *const std::ffi::c_void,
            b.bytes,
        );
        off += b.bytes;
    }
    let rc = backend.broadcast(
        dev,
        total_elems,
        crate::tensor_parallel::collective::DTYPE_F32,
        owner,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        crate::cuda::inner::free_managed(dev);
        return -1;
    }
    ZERO_BROADCAST_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    ZERO_BUCKET_MEMBERS.fetch_add(bucket.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let mut off = 0usize;
    for b in bucket {
        let t = unsafe { &*(b.raw as *const NslTensor) };
        crate::cuda::inner::memcpy_dtod(
            t.data,
            unsafe { (dev as *const u8).add(off) } as *const std::ffi::c_void,
            b.bytes,
        );
        off += b.bytes;
    }
    crate::cuda::inner::free_managed(dev);
    0
}

/// Reduce gradients across all DP ranks.
///
/// For Stage 1 (optimizer state sharding), all gradients are all-reduced so
/// every rank has identical gradients. The reduction averages by dividing
/// by world_size after summation.
///
/// In single-process mode (world_size=1), this is a no-op since gradients
/// are already complete.
///
/// NOTE (M43b): this path is currently DEAD in production. `nsl_zero_init`
/// is registered as a builtin but never emitted by codegen and is absent
/// from the stdlib, so `ZERO_CTX` is always `None` and this function
/// returns -1 at the context guard before touching any tensor. Wiring
/// `nsl_zero_init` emission is M43b-proper work; the body below is kept
/// correct (CPU and GPU, f64/f32) so that wiring lands on a sound runtime.
///
/// `grads_list_ptr`: pointer to an NslList of gradient tensors.
/// `num_params`: number of gradient tensors in the list.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_reduce_grads(grads_list_ptr: i64, num_params: i64) -> i64 {
    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };

    // Single-process: gradients are already complete, no communication needed.
    if ctx.world_size <= 1 {
        return 0;
    }

    // D3: REAL data-parallel gradient reduction — per tensor:
    // fixed-order all_reduce_sum across ranks (SimulatedBackend), then
    // divide by world_size. Every rank must reach every collective in the
    // same order (the spin-barrier counts arrivals), which holds because
    // codegen emits ONE reduce call over the same param-ordered list on
    // every rank.
    let list_ptr = grads_list_ptr as *const crate::list::NslList;
    if list_ptr.is_null() {
        return -1;
    }
    let list = unsafe { &*list_ptr };

    let ws = ctx.world_size as f64;
    let inv_ws = 1.0 / ws;

    // P4 item 15: bucketed fast path — flatten same-(dtype, device) grads in
    // param-index (= layer) order and run ONE collective per ≤cap bucket.
    let cap = zero_bucket_cap_bytes();
    // Review M2: stage-2 gradient partitioning lives INSIDE the bucketed
    // path; a cap of 0 would silently fall through to the all-reduce loop,
    // voiding the reduce_scatter contract and its counters. Refuse loudly.
    if cap == 0 && ctx.stage != ZeROStage::Stage1 {
        eprintln!(
            "nsl: nsl_zero_reduce_grads: --zero-stage 2 requires bucketed              collectives — do not set NSL_ZERO_BUCKET_MB=0 with stage >= 2"
        );
        return -1;
    }
    if cap > 0 {
        let mut cpu_f64: Vec<BucketItem> = Vec::new();
        let mut cpu_f32: Vec<BucketItem> = Vec::new();
        let mut gpu_f32: Vec<BucketItem> = Vec::new();
        for i in 0..num_params as usize {
            if i >= list.len as usize {
                break;
            }
            let raw = unsafe { *list.data.add(i) };
            if raw == 0 {
                continue;
            }
            let t = unsafe { &*(raw as *const NslTensor) };
            if t.device != 0 && !ctx.cuda_aware {
                eprintln!(
                    "nsl: nsl_zero_reduce_grads: GPU-resident ZeRO SPMD needs \
                     real collectives — run with --collectives nccl \
                     (nccl-featured build), or move the model to CPU / drop \
                     --devices.",
                );
                return -5;
            }
            if t.device == 0 && t.dtype > 1 {
                eprintln!(
                    "nsl: nsl_zero_reduce_grads: unsupported gradient dtype {} \
                     at index {}; only f64 (0) and f32 (1) gradients are \
                     supported",
                    t.dtype, i
                );
                return -1;
            }
            if t.device > 0 && t.dtype != 1 {
                eprintln!(
                    "nsl: nsl_zero_reduce_grads: GPU gradient at index {} has \
                     dtype {}; only the canonical GPU f32 (dtype 1) is \
                     supported",
                    i, t.dtype
                );
                return -1;
            }
            let item = BucketItem {
                idx: i,
                raw,
                len: t.len as usize,
                bytes: t.data_byte_size(),
                off: 0,
            };
            match (t.device, t.dtype) {
                (0, 0) => cpu_f64.push(item),
                (0, _) => cpu_f32.push(item),
                _ => gpu_f32.push(item),
            }
        }
        // P4 item 16 (ZeRO-2+): gradient PARTITIONING — one owner-segmented
        // reduce_scatter per (dtype, device) group instead of all-reducing
        // every gradient everywhere. Each rank receives only its owned
        // gradients (summed + averaged); non-owned grad tensors keep local
        // values that nothing reads (the optimizer update is owner-gated).
        if ctx.stage != ZeROStage::Stage1 {
            // M3: cap the padded scatter buffer (large groups sub-group,
            // giant tensors chunk) so 1B-scale gradients never hit the
            // 64MB CPU-shm slot wall.
            for (items, dtype) in [(&cpu_f64, 0u16), (&cpu_f32, 1u16)] {
                let esz = if dtype == 0 { 8 } else { 4 };
                for sub in split_scatter_subgroups(ctx, items, cap, esz) {
                    if reduce_scatter_group_cpu(ctx, &sub, dtype, inv_ws) != 0 {
                        eprintln!("nsl: nsl_zero_reduce_grads: reduce_scatter failed");
                        return -1;
                    }
                }
            }
            #[cfg(feature = "cuda")]
            for sub in split_scatter_subgroups(ctx, &gpu_f32, cap, 4) {
                if reduce_scatter_group_gpu(ctx, &sub, inv_ws) != 0 {
                    eprintln!("nsl: nsl_zero_reduce_grads: device reduce_scatter failed");
                    return -1;
                }
            }
            #[cfg(not(feature = "cuda"))]
            if !gpu_f32.is_empty() {
                eprintln!("nsl: nsl_zero_reduce_grads: GPU gradients require the cuda feature");
                return -1;
            }
            return 0;
        }

        for (items, dtype) in [(&cpu_f64, 0u16), (&cpu_f32, 1u16)] {
            for bucket in split_buckets(items, cap) {
                if reduce_bucket_cpu(ctx, &bucket, dtype, inv_ws) != 0 {
                    eprintln!("nsl: nsl_zero_reduce_grads: bucketed all_reduce failed");
                    return -1;
                }
            }
        }
        #[cfg(feature = "cuda")]
        for bucket in split_buckets(&gpu_f32, cap) {
            if reduce_bucket_gpu(ctx, &bucket, inv_ws) != 0 {
                eprintln!("nsl: nsl_zero_reduce_grads: bucketed device all_reduce failed");
                return -1;
            }
        }
        #[cfg(not(feature = "cuda"))]
        if !gpu_f32.is_empty() {
            eprintln!("nsl: nsl_zero_reduce_grads: GPU gradients require the cuda feature");
            return -1;
        }
        return 0;
    }

    for i in 0..num_params as usize {
        if i >= list.len as usize {
            break;
        }
        let tensor_raw = unsafe { *list.data.add(i) };
        let tensor_ptr = tensor_raw as *mut NslTensor;
        if tensor_ptr.is_null() {
            continue;
        }
        let tensor = unsafe { &*tensor_ptr };

        // P4 item 14: GPU-resident SPMD requires the CUDA-aware backend.
        // Under the CPU-shm SimulatedBackend the old v1 refusal stands: N
        // ranks would pile onto few devices with a host-staged reduce that
        // multiplies VRAM and PCIe traffic. With NCCL (--collectives nccl)
        // grads stay on-device and rank→device binding stripes the ranks
        // (cuda::select_device_ordinal). (Runtime, not compile-time: the
        // default --target is "cuda" even for CPU-placed models, so a
        // target-based guard would wrongly refuse the CPU SPMD path.)
        if ctx.world_size > 1 && tensor.device != 0 && !ctx.cuda_aware {
            eprintln!(
                "nsl: nsl_zero_reduce_grads: GPU-resident ZeRO SPMD needs real \
                 collectives — run with --collectives nccl (nccl-featured \
                 build), or move the model to CPU / drop --devices.",
            );
            return -5;
        }

        // Refuse loudly instead of silently skipping: a skipped tensor would
        // leave one gradient un-averaged and produce silently-wrong training.
        if tensor.device == 0 && tensor.dtype > 1 {
            eprintln!(
                "nsl: nsl_zero_reduce_grads: unsupported gradient dtype {} at \
                 index {}; only f64 (0) and f32 (1) gradients are supported",
                tensor.dtype, i
            );
            return -1;
        }
        if tensor.device > 0 && tensor.dtype != 1 {
            eprintln!(
                "nsl: nsl_zero_reduce_grads: GPU gradient at index {} has \
                 dtype {}; only the canonical GPU f32 (dtype 1) is supported",
                i, tensor.dtype
            );
            return -1;
        }

        if tensor.device == 0 {
            // CPU: in-place all-reduce (send==recv is safe — the backend
            // copies to its slot first), then average. Review H1: host
            // buffers go to the host backend.
            let backend = host_or_main(ctx);
            let rc = backend.all_reduce_sum(
                tensor.data as *const std::ffi::c_void,
                tensor.data,
                tensor.len as usize,
                tensor.dtype as crate::tensor_parallel::collective::DtypeId,
                std::ptr::null_mut(),
            );
            if rc != 0 {
                eprintln!(
                    "nsl: nsl_zero_reduce_grads: all_reduce failed rc={rc} at \
                     index {i}"
                );
                return -1;
            }
            ZERO_ALL_REDUCE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::tensor::nsl_tensor_mul_scalar_inplace(tensor_raw, inv_ws);
            continue;
        }

        // P4 item 14: CUDA-aware backend — the collective consumes the DEVICE
        // pointer directly (in-place sum), then the average runs as a native
        // on-device scale. No host staging, no PCIe round-trip.
        #[cfg(feature = "cuda")]
        if ctx.cuda_aware {
            let backend = ctx
                .backend
                .as_ref()
                .expect("world_size > 1 implies a backend");
            let rc = backend.all_reduce_sum(
                tensor.data as *const std::ffi::c_void,
                tensor.data,
                tensor.len as usize,
                crate::tensor_parallel::collective::DTYPE_F32,
                std::ptr::null_mut(),
            );
            if rc != 0 {
                eprintln!(
                    "nsl: nsl_zero_reduce_grads: device all_reduce failed \
                     rc={rc} at index {i}"
                );
                return -1;
            }
            ZERO_ALL_REDUCE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // GPU f32 contiguous: native in-place scale kernel.
            crate::tensor::nsl_tensor_mul_scalar_inplace(tensor_raw, inv_ws);
            continue;
        }

        // GPU: round-trip through the CPU with an explicit f64 -> f32
        // down-convert on the way back (same staged pattern as
        // nsl_tensor_add_inplace's GPU arm). NOT delegated to
        // nsl_tensor_mul_scalar_inplace: its GPU arm copies
        // `len * element_size()` bytes straight from the round-trip CPU
        // buffer, but `nsl_tensor_to_device(gpu_f32, 0)` upcasts to CPU f64,
        // so that copy would push raw f64 bytes into the f32 device buffer.
        #[cfg(feature = "cuda")]
        {
            let cpu_ptr = crate::tensor::nsl_tensor_to_device(tensor_raw, 0);
            let cpu = unsafe { &*(cpu_ptr as *const NslTensor) };
            let len = cpu.len as usize;
            // Convert to a host f32 staging buffer (unscaled), all-reduce
            // across ranks on the host bytes, THEN average and push back.
            let mut staged = vec![0.0f32; len];
            if cpu.dtype == 1 {
                let d = cpu.data as *const f32;
                for (j, slot) in staged.iter_mut().enumerate() {
                    *slot = unsafe { *d.add(j) };
                }
            } else {
                let d = cpu.data as *const f64;
                for (j, slot) in staged.iter_mut().enumerate() {
                    *slot = (unsafe { *d.add(j) }) as f32;
                }
            }
            {
                let backend = ctx
                    .backend
                    .as_ref()
                    .expect("world_size > 1 implies a backend");
                    let rc = backend.all_reduce_sum(
                    staged.as_ptr() as *const std::ffi::c_void,
                    staged.as_mut_ptr() as *mut std::ffi::c_void,
                    len,
                    crate::tensor_parallel::collective::DTYPE_F32,
                    std::ptr::null_mut(),
                );
                if rc != 0 {
                    eprintln!(
                        "nsl: nsl_zero_reduce_grads: GPU staged all_reduce \
                         failed rc={rc} at index {i}"
                    );
                    return -1;
                }
            }
            for slot in staged.iter_mut() {
                *slot *= inv_ws as f32;
            }
            crate::cuda::inner::memcpy_htod(
                tensor.data,
                staged.as_ptr() as *const std::ffi::c_void,
                len * std::mem::size_of::<f32>(),
            );
            crate::tensor::nsl_tensor_free(cpu_ptr);
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "nsl: nsl_zero_reduce_grads: GPU gradient at index {} requires \
                 the cuda feature",
                i
            );
            return -1;
        }
    }

    0
}

/// ZeRO optimizer step: after each rank updates only its owned parameters,
/// all-gather the updated parameters so every rank has the full model.
///
/// In single-process mode (world_size=1), this is a no-op since all params
/// are updated locally.
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_step() -> i64 {
    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };

    // Single-process: all params updated locally, no communication needed.
    if ctx.world_size <= 1 {
        return 0;
    }

    // Multi-process: each rank would broadcast its updated param slices
    // via all-gather. With actual NCCL backend:
    // 1. For each rank r, broadcast params[owned_by_r] to all ranks
    // 2. Each rank copies received params into its local model
    //
    // With simulated single-process, this is a no-op since we update all
    // owned params locally and there are no other ranks to sync with.
    0
}

/// D3 (ZeRO-1): synchronize parameters after the sharded optimizer step.
///
/// Each rank updated ONLY the params it owns (round-robin `idx % ws`);
/// this broadcasts every param from its owner so all ranks hold the full
/// updated model. EVERY rank must call this with the same param-ordered
/// list (identical collective sequence — the spin-barrier deadlocks
/// otherwise). CPU f64/f32 broadcast in place; GPU f32 stages through the
/// host. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_zero_sync_params(params_list_ptr: i64, num_params: i64) -> i64 {
    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };
    if ctx.world_size <= 1 {
        return 0;
    }
    let list_ptr = params_list_ptr as *const crate::list::NslList;
    if list_ptr.is_null() {
        return -1;
    }
    let list = unsafe { &*list_ptr };
    let backend = ctx
        .backend
        .as_ref()
        .expect("world_size > 1 implies a backend");

    // P4 item 15: bucketed fast path — group params by (OWNER, dtype, device)
    // and broadcast each ≤cap bucket from its owning rank in one collective.
    let cap = zero_bucket_cap_bytes();
    if cap > 0 {
        use std::collections::BTreeMap;
        // (owner, dtype, is_gpu) → items, in ascending param order.
        let mut groups: BTreeMap<(i32, u16, bool), Vec<BucketItem>> = BTreeMap::new();
        for i in 0..num_params as usize {
            if i >= list.len as usize {
                break;
            }
            let raw = unsafe { *list.data.add(i) };
            if raw == 0 {
                continue;
            }
            let t = unsafe { &*(raw as *const NslTensor) };
            let owner = ctx
                .owner_of
                .get(i)
                .copied()
                .unwrap_or((i % ctx.world_size) as i32);
            if t.device != 0 && !ctx.cuda_aware {
                eprintln!(
                    "nsl: nsl_zero_sync_params: GPU-resident ZeRO SPMD needs \
                     real collectives — run with --collectives nccl \
                     (nccl-featured build), or move the model to CPU / drop \
                     --devices.",
                );
                return -5;
            }
            if t.device == 0 && t.dtype > 1 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: unsupported CPU param dtype {} \
                     at index {i}",
                    t.dtype
                );
                return -1;
            }
            if t.device > 0 && t.dtype != 1 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: GPU param at index {i} has \
                     dtype {}; only f32 is supported",
                    t.dtype
                );
                return -1;
            }
            groups
                .entry((owner, t.dtype, t.device != 0))
                .or_default()
                .push(BucketItem {
                    idx: i,
                    raw,
                    len: t.len as usize,
                    bytes: t.data_byte_size(),
                    off: 0,
                });
        }
        for ((owner, dtype, is_gpu), items) in &groups {
            for bucket in split_buckets(items, cap) {
                let rc = if *is_gpu {
                    #[cfg(feature = "cuda")]
                    {
                        broadcast_bucket_gpu(ctx, &bucket, *owner)
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        eprintln!(
                            "nsl: nsl_zero_sync_params: GPU params require the \
                             cuda feature"
                        );
                        -1
                    }
                } else {
                    broadcast_bucket_cpu(ctx, &bucket, *dtype, *owner)
                };
                if rc != 0 {
                    eprintln!("nsl: nsl_zero_sync_params: bucketed broadcast failed");
                    return -1;
                }
            }
        }
        return 0;
    }

    for i in 0..num_params as usize {
        if i >= list.len as usize {
            break;
        }
        let tensor_raw = unsafe { *list.data.add(i) };
        let tensor_ptr = tensor_raw as *mut NslTensor;
        if tensor_ptr.is_null() {
            continue;
        }
        let tensor = unsafe { &*tensor_ptr };
        // P4 item 13: the broadcast root comes from the ownership PLAN. The
        // index-modulo fallback only covers a param the partitioner never saw
        // (it should not happen — partition runs before any sync).
        let owner = ctx
            .owner_of
            .get(i)
            .copied()
            .unwrap_or((i % ctx.world_size) as i32);

        // P4 item 14: GPU-resident params need the CUDA-aware backend (see
        // the matching guard in nsl_zero_reduce_grads).
        if tensor.device != 0 && !ctx.cuda_aware {
            eprintln!(
                "nsl: nsl_zero_sync_params: GPU-resident ZeRO SPMD needs real \
                 collectives — run with --collectives nccl (nccl-featured \
                 build), or move the model to CPU / drop --devices.",
            );
            return -5;
        }
        // CUDA-aware: broadcast the DEVICE pointer straight from the owner.
        #[cfg(feature = "cuda")]
        if tensor.device != 0 && ctx.cuda_aware {
            if tensor.dtype != 1 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: GPU param at index {i} has \
                     dtype {}; only f32 is supported",
                    tensor.dtype
                );
                return -1;
            }
            let rc = backend.broadcast(
                tensor.data,
                tensor.len as usize,
                crate::tensor_parallel::collective::DTYPE_F32,
                owner,
                std::ptr::null_mut(),
            );
            if rc != 0 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: device broadcast failed rc={rc} \
                     at index {i}"
                );
                return -1;
            }
            ZERO_BROADCAST_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }

        if tensor.device == 0 {
            if tensor.dtype > 1 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: unsupported CPU param dtype \
                     {} at index {i}",
                    tensor.dtype
                );
                return -1;
            }
            let rc = backend.broadcast(
                tensor.data,
                tensor.len as usize,
                tensor.dtype as crate::tensor_parallel::collective::DtypeId,
                owner,
                std::ptr::null_mut(),
            );
            if rc != 0 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: broadcast failed rc={rc} at \
                     index {i}"
                );
                return -1;
            }
            ZERO_BROADCAST_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }

        // GPU f32: stage through the host (same staged pattern as the
        // reduce above). Non-owners receive into the staging buffer and
        // push it back to the device.
        #[cfg(feature = "cuda")]
        {
            if tensor.dtype != 1 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: GPU param at index {i} has \
                     dtype {}; only f32 is supported",
                    tensor.dtype
                );
                return -1;
            }
            let len = tensor.len as usize;
            let mut staged = vec![0.0f32; len];
            crate::cuda::inner::ensure_context();
            crate::cuda::inner::memcpy_dtoh(
                staged.as_mut_ptr() as *mut std::ffi::c_void,
                tensor.data,
                len * 4,
            );
            let rc = backend.broadcast(
                staged.as_mut_ptr() as *mut std::ffi::c_void,
                len,
                crate::tensor_parallel::collective::DTYPE_F32,
                owner,
                std::ptr::null_mut(),
            );
            if rc != 0 {
                eprintln!(
                    "nsl: nsl_zero_sync_params: GPU staged broadcast failed \
                     rc={rc} at index {i}"
                );
                return -1;
            }
            crate::cuda::inner::memcpy_htod(
                tensor.data,
                staged.as_ptr() as *const std::ffi::c_void,
                len * 4,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "nsl: nsl_zero_sync_params: GPU param at index {i} requires \
                 the cuda feature"
            );
            return -1;
        }
    }
    0
}

/// Destroy ZeRO context. Returns 0 on success, -1 if not initialized.
#[no_mangle]
pub extern "C" fn nsl_zero_destroy() -> i64 {
    let mut guard = ZERO_CTX.lock().unwrap();
    if guard.is_none() {
        return -1;
    }
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// Gradient accumulation FFI
// ---------------------------------------------------------------------------

/// Accumulate gradients: dst += src element-wise.
/// Both `dst_ptr` and `src_ptr` are pointers to NslTensor.
/// `num_elems` is the number of elements to accumulate (must match tensor lengths).
///
/// Matched CPU f64/f64 and f32/f32 pairs take the original inline loops
/// (byte-identical fast path). CPU dtype mismatches convert element-wise
/// with proper casts. Any pair with a GPU-resident side — including the
/// live single-GPU FullBuffer path (GPU accum buffer += tape gradient,
/// reached by `grad_accumulation > 1` with lion/muon/soap) — migrates
/// `src` to `dst`'s device via `nsl_tensor_to_device_like` (owned ref,
/// canonical dtype: CPU=f64, GPU=f32) and delegates to the GPU-safe
/// `nsl_tensor_add_inplace` round-trip.
///
/// Non-f32/f64 dtypes refuse loudly with -1: the previous fallback promoted
/// both pointers to f64 blindly, which read out of bounds whenever `src`
/// was narrower than 8 bytes per element.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_grad_accumulate_add(
    dst_ptr: i64,
    src_ptr: i64,
    num_elems: i64,
) -> i64 {
    let dst_tensor = dst_ptr as *mut NslTensor;
    let src_tensor = src_ptr as *const NslTensor;
    if dst_tensor.is_null() || src_tensor.is_null() {
        return -1;
    }

    let dst = unsafe { &*dst_tensor };
    let src = unsafe { &*src_tensor };
    let n = num_elems as usize;

    if dst.dtype > 1 || src.dtype > 1 {
        eprintln!(
            "nsl: nsl_grad_accumulate_add: unsupported gradient dtype pair \
             (dst={}, src={}); only f64 (0) and f32 (1) are supported",
            dst.dtype, src.dtype
        );
        return -1;
    }

    if dst.device == 0 && src.device == 0 {
        match (dst.dtype, src.dtype) {
            (0, 0) => {
                // f64 += f64
                let d = dst.data as *mut f64;
                let s = src.data as *const f64;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i); }
                }
            }
            (1, 1) => {
                // f32 += f32
                let d = dst.data as *mut f32;
                let s = src.data as *const f32;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i); }
                }
            }
            (0, 1) => {
                // f64 += f32: element-wise upcast.
                let d = dst.data as *mut f64;
                let s = src.data as *const f32;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i) as f64; }
                }
            }
            (1, 0) => {
                // f32 += f64: element-wise downcast.
                let d = dst.data as *mut f32;
                let s = src.data as *const f64;
                for i in 0..n {
                    unsafe { *d.add(i) += *s.add(i) as f32; }
                }
            }
            _ => unreachable!("dtypes > 1 rejected above"),
        }
        return 0;
    }

    // At least one side is GPU-resident. The delegated nsl_tensor_add_inplace
    // always adds dst.len elements, so a partial `num_elems < dst.len` request
    // (honored by the CPU arms above) cannot be expressed here — refuse it
    // rather than silently over-adding.
    if n != dst.len as usize {
        eprintln!(
            "nsl: nsl_grad_accumulate_add: partial accumulation \
             (num_elems={} != dst len={}) is not supported on the GPU path; \
             refusing",
            n, dst.len
        );
        return -1;
    }

    // Migrate src to dst's device —
    // nsl_tensor_to_device_like returns an OWNED ref (refcount++ when the
    // devices already match, a fresh tensor otherwise) that must be freed
    // exactly once below (FBIP ownership, see stmt_fase.rs accumulate arm).
    // Migration canonicalizes dtype (CPU=f64, GPU=f32), so the delegated
    // nsl_tensor_add_inplace sees matching same-device operands and its
    // GPU arm performs the proven CPU round-trip with f64 -> f32 staging.
    let migrated = crate::tensor::nsl_tensor_to_device_like(src_ptr, dst_ptr);
    if migrated == 0 {
        return -1;
    }
    let mig = unsafe { &*(migrated as *const NslTensor) };
    if mig.dtype != dst.dtype || mig.len != dst.len {
        eprintln!(
            "nsl: nsl_grad_accumulate_add: post-migration mismatch \
             (dst dtype={} len={}, migrated src dtype={} len={}); refusing",
            dst.dtype, dst.len, mig.dtype, mig.len
        );
        crate::tensor::nsl_tensor_free(migrated);
        return -1;
    }
    crate::tensor::nsl_tensor_add_inplace(dst_ptr, migrated);
    crate::tensor::nsl_tensor_free(migrated);

    0
}

/// Zero out gradient buffer. `grad_ptr` is a pointer to NslTensor.
/// `num_elems` is the number of elements to zero.
///
/// GPU buffers (accumulation buffers are `zeros_like(param)`, so they live
/// on the param's device) are zeroed with a device memset, mirroring
/// `nsl_tensor_zero_inplace`. Zeroing is dtype-agnostic byte clearing.
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_grad_zero(grad_ptr: i64, num_elems: i64) -> i64 {
    let tensor_ptr = grad_ptr as *mut NslTensor;
    if tensor_ptr.is_null() {
        return -1;
    }

    let tensor = unsafe { &*tensor_ptr };
    let n = num_elems as usize;

    let byte_width = match tensor.dtype {
        0 => 8usize, // f64
        1 => 4,      // f32
        2 | 3 => 2,  // f16/bf16
        4..=6 => 1, // i8/fp8
        _ => 8,      // default to f64 width
    };
    let total_bytes = n * byte_width;

    if tensor.device != 0 {
        // GPU grad buffers are always f32 (`zeros_like(param)` on GPU). The
        // byte_width table above reflects CPU storage widths — e.g. dtype 4
        // is 1 byte on CPU but allocated 4 bytes/elem on GPU by
        // nsl_tensor_to_device — so any non-f32 GPU buffer would be silently
        // under-zeroed. Refuse instead (mirrors nsl_zero_reduce_grads).
        if tensor.dtype != 1 {
            eprintln!(
                "nsl: nsl_grad_zero: GPU grad buffer must be f32 (dtype 1), \
                 got dtype {}; refusing",
                tensor.dtype
            );
            return -1;
        }
        #[cfg(feature = "cuda")]
        {
            crate::cuda::inner::memset_d8(tensor.data, total_bytes);
            return 0;
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("nsl: nsl_grad_zero: GPU tensor requires the cuda feature");
            return -1;
        }
    }

    unsafe {
        std::ptr::write_bytes(tensor.data as *mut u8, 0, total_bytes);
    }

    0
}

/// All-reduce gradients across all DP ranks (legacy API, wraps nsl_zero_reduce_grads).
/// `grad_ptr` is a single gradient tensor pointer, `num_elems` is element count.
///
/// NOTE: never-emitted scaffolding — no codegen path or stdlib function
/// references this symbol today. It intentionally performs no arithmetic:
/// a single-process all-reduce is the identity, and real multi-process
/// NCCL collectives are M43b work (see nsl_zero_reduce_grads).
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_grad_all_reduce(_grad_ptr: i64, _num_elems: i64) -> i64 {
    // Single-process: no-op (gradient is already the full gradient).
    // Multi-process would call all_reduce_sum on this single tensor.
    let guard = ZERO_CTX.lock().unwrap();
    if let Some(ctx) = guard.as_ref() {
        if ctx.world_size <= 1 {
            return 0;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::sync::{Mutex, MutexGuard, OnceLock};

    // Tests that touch ZERO_CTX race on the shared global when run in
    // parallel.  Serialize them with a module-local guard (same pattern as
    // inspect_ffi.rs / health_ffi.rs) so `cargo test` is safe without
    // requiring --test-threads=1 on the command line.
    fn zero_serial_lock() -> MutexGuard<'static, ()> {
        static SERIAL: OnceLock<Mutex<()>> = OnceLock::new();
        let m = SERIAL.get_or_init(|| Mutex::new(()));
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn test_zero_stage_parsing() {
        assert_eq!(ZeROStage::from_i64(1), Some(ZeROStage::Stage1));
        assert_eq!(ZeROStage::from_i64(2), Some(ZeROStage::Stage2));
        assert_eq!(ZeROStage::from_i64(3), Some(ZeROStage::Stage3));
        assert_eq!(ZeROStage::from_i64(0), None);
        assert_eq!(ZeROStage::from_i64(4), None);
        assert_eq!(ZeROStage::from_i64(-1), None);
    }

    #[test]
    fn test_partition_params_round_robin() {
        let parts = partition_params(10, 3);
        assert_eq!(parts.len(), 3);
        // Rank 0: params 0, 3, 6, 9
        assert_eq!(parts[0], vec![0, 3, 6, 9]);
        // Rank 1: params 1, 4, 7
        assert_eq!(parts[1], vec![1, 4, 7]);
        // Rank 2: params 2, 5, 8
        assert_eq!(parts[2], vec![2, 5, 8]);

        // All params accounted for
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 10);
    }

    // ── P4 item 13: byte-balanced ownership ──────────────────────────────

    #[test]
    fn test_partition_balanced_complete_and_disjoint() {
        let sizes: Vec<u64> = vec![7, 1, 1, 1, 3, 3, 9, 2];
        let parts = partition_params_balanced(&sizes, 3);
        let mut seen = vec![false; sizes.len()];
        for p in &parts {
            for &i in p {
                assert!(!seen[i], "param {i} assigned twice");
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "every param must have an owner");
    }

    #[test]
    fn test_partition_balanced_beats_round_robin_on_skew() {
        // Transformer-shaped skew: one huge embedding, a few big matrices,
        // many tiny biases/norms. Round-robin by index piles the heavies
        // onto low ranks; LPT balances byte loads.
        let sizes: Vec<u64> = vec![4_000_000, 250_000, 250_000, 4_096, 4_096, 4_096, 4_096, 4_096];
        let ws = 2;
        let load = |parts: &Vec<Vec<usize>>| -> Vec<u64> {
            parts
                .iter()
                .map(|p| p.iter().map(|&i| sizes[i]).sum())
                .collect()
        };
        let rr = load(&{
            let mut v = vec![Vec::new(); ws];
            for i in 0..sizes.len() {
                v[i % ws].push(i);
            }
            v
        });
        let lpt = load(&partition_params_balanced(&sizes, ws));
        let spread = |l: &Vec<u64>| l.iter().max().unwrap() - l.iter().min().unwrap();
        assert!(
            spread(&lpt) < spread(&rr),
            "LPT spread {:?} must beat round-robin spread {:?}",
            lpt,
            rr
        );
        // LPT bound sanity: max load ≤ max(largest single tensor, 4/3 × ideal)
        // — no partition can undercut the biggest indivisible tensor, and away
        // from that floor LPT is within 4/3 of the perfect split.
        let total: u64 = sizes.iter().sum();
        let largest = *sizes.iter().max().unwrap();
        let bound = largest.max(total.div_ceil(ws as u64) * 4 / 3);
        assert!(*lpt.iter().max().unwrap() <= bound);
    }

    #[test]
    fn test_partition_balanced_deterministic_and_sorted() {
        let sizes: Vec<u64> = vec![5, 5, 5, 5, 8, 8, 1];
        let a = partition_params_balanced(&sizes, 4);
        let b = partition_params_balanced(&sizes, 4);
        assert_eq!(a, b, "identical inputs must give the identical plan");
        for p in &a {
            let mut s = p.clone();
            s.sort_unstable();
            assert_eq!(*p, s, "per-rank index lists are ascending");
        }
    }

    #[test]
    fn test_partition_balanced_edge_cases() {
        // ws=1 owns everything, in order.
        assert_eq!(partition_params_balanced(&[3, 1, 2], 1), vec![vec![0, 1, 2]]);
        // More ranks than params: some ranks legitimately own nothing.
        let parts = partition_params_balanced(&[10, 20], 4);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 2);
        assert_eq!(parts.len(), 4);
        // Zero-size params still get owners (coverage over balance).
        let parts = partition_params_balanced(&[0, 0, 0], 2);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_owners_from_partition_matches_membership() {
        let sizes: Vec<u64> = vec![7, 1, 9, 2, 2];
        let parts = partition_params_balanced(&sizes, 2);
        let owners = owners_from_partition(&parts, sizes.len());
        for (rank, part) in parts.iter().enumerate() {
            for &i in part {
                assert_eq!(owners[i], rank as i32);
            }
        }
    }

    #[test]
    fn test_zero_ffi_lifecycle() {
        let _serial = zero_serial_lock();
        // Ensure clean state
        {
            let mut guard = ZERO_CTX.lock().unwrap();
            *guard = None;
        }

        // D3: world_size > 1 without the spawner's shm env REFUSES (-3) —
        // the old context had no backend to build, the real one does.
        assert_eq!(nsl_zero_init(1, 4), -3);

        // world_size = 1: no backend needed; rank 0 owns everything.
        assert_eq!(nsl_zero_init(1, 1), 0);
        // Double init fails
        assert_eq!(nsl_zero_init(1, 1), -1);

        let owned = nsl_zero_partition(10);
        assert_eq!(owned, 10);
        assert_eq!(nsl_zero_owns_param(0), 1);
        assert_eq!(nsl_zero_owns_param(9), 1);

        // Reduce, step, and param sync are identity at world_size 1 (the
        // ws==1 early return fires before the null-list check).
        assert_eq!(nsl_zero_step(), 0);
        assert_eq!(nsl_zero_sync_params(0, 0), 0);

        // Destroy succeeds
        assert_eq!(nsl_zero_destroy(), 0);
        // Double destroy fails
        assert_eq!(nsl_zero_destroy(), -1);

        // Invalid stage
        assert_eq!(nsl_zero_init(5, 2), -1);
    }

    #[test]
    fn test_zero_owns_param_not_initialized() {
        let _serial = zero_serial_lock();
        {
            let mut guard = ZERO_CTX.lock().unwrap();
            *guard = None;
        }
        assert_eq!(nsl_zero_owns_param(0), -1);
    }

    #[test]
    fn test_grad_zero() {
        // Create a small f64 tensor manually
        let mut data = vec![1.0f64, 2.0, 3.0, 4.0];
        let shape = vec![4i64];
        let strides = vec![1i64];

        let tensor = NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data: data.as_mut_ptr() as *mut c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1,
            len: 4,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype: 0, // f64
            owns_data: 0, // borrowed
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        };

        let tensor_ptr = &tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_zero(tensor_ptr, 4), 0);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_grad_accumulate_add() {
        let mut dst_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let src_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let shape = vec![4i64];
        let strides = vec![1i64];

        let dst_tensor = NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data: dst_data.as_mut_ptr() as *mut c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1,
            len: 4,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype: 0,
            owns_data: 0,
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        };

        let src_tensor = NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data: src_data.as_ptr() as *mut c_void,
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            ndim: 1,
            len: 4,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype: 0,
            owns_data: 0,
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        };

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), 0);
        assert_eq!(dst_data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Build a borrowed (owns_data=0) CPU test tensor over caller-owned
    /// buffers. Used by the mixed-dtype accumulate tests below.
    fn borrowed_cpu_tensor(
        data: *mut c_void,
        shape: *mut i64,
        strides: *mut i64,
        len: i64,
        dtype: u16,
    ) -> NslTensor {
        NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data,
            shape,
            strides,
            ndim: 1,
            len,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype,
            owns_data: 0,
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        }
    }

    /// f64 dst += f32 src. The pre-fix code promoted BOTH pointers to f64,
    /// reading 8 bytes per element out of the 4-byte-per-element f32 src —
    /// an out-of-bounds read producing garbage sums. Now converts properly.
    #[test]
    fn test_grad_accumulate_add_f64_dst_f32_src_converts() {
        let mut dst_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut src_data = vec![0.5f32, 1.5, 2.5, 3.5];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let dst_tensor = borrowed_cpu_tensor(
            dst_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            0,
        );
        let src_tensor = borrowed_cpu_tensor(
            src_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            1,
        );

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), 0);
        assert_eq!(dst_data, vec![1.5, 3.5, 5.5, 7.5]);
        // src untouched
        assert_eq!(src_data, vec![0.5, 1.5, 2.5, 3.5]);
    }

    /// f32 dst += f64 src (tape-AD gradients are CPU f64; an f32 accum
    /// buffer must receive a downcast add, not a reinterpretation).
    #[test]
    fn test_grad_accumulate_add_f32_dst_f64_src_converts() {
        let mut dst_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut src_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let dst_tensor = borrowed_cpu_tensor(
            dst_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            1,
        );
        let src_tensor = borrowed_cpu_tensor(
            src_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            0,
        );

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), 0);
        assert_eq!(dst_data, vec![11.0f32, 22.0, 33.0, 44.0]);
    }

    /// Non-f32/f64 dtypes must refuse (-1) and leave the buffers untouched
    /// instead of taking the old unsound promote-to-f64 arm.
    #[test]
    fn test_grad_accumulate_add_unsupported_dtype_refuses() {
        let mut dst_data = vec![0x3C00u16, 0x4000, 0x4200, 0x4400]; // f16 bits
        let mut src_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let dst_tensor = borrowed_cpu_tensor(
            dst_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            2, // f16
        );
        let src_tensor = borrowed_cpu_tensor(
            src_data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            0,
        );

        let dst_ptr = &dst_tensor as *const NslTensor as i64;
        let src_ptr = &src_tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_accumulate_add(dst_ptr, src_ptr, 4), -1);
        assert_eq!(dst_data, vec![0x3C00, 0x4000, 0x4200, 0x4400]);
    }

    #[test]
    fn test_grad_accumulate_add_null_refuses() {
        assert_eq!(nsl_grad_accumulate_add(0, 0, 4), -1);
        assert_eq!(nsl_grad_zero(0, 4), -1);
    }

    #[test]
    fn test_grad_zero_f32() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];

        let tensor = borrowed_cpu_tensor(
            data.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
            1,
        );

        let tensor_ptr = &tensor as *const NslTensor as i64;
        assert_eq!(nsl_grad_zero(tensor_ptr, 4), 0);
        assert_eq!(data, vec![0.0f32, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_partition_single_rank() {
        let _serial = zero_serial_lock();
        // Ensure clean state
        {
            let mut guard = ZERO_CTX.lock().unwrap();
            *guard = None;
        }

        assert_eq!(nsl_zero_init(1, 1), 0);
        let owned = nsl_zero_partition(5);
        assert_eq!(owned, 5); // Single rank owns all params

        // All params owned
        for i in 0..5 {
            assert_eq!(nsl_zero_owns_param(i), 1);
        }

        // reduce_grads and step are no-ops for single rank
        assert_eq!(nsl_zero_step(), 0);

        assert_eq!(nsl_zero_destroy(), 0);
    }

    /// Backend-less context for the pure sub-group splitter (it reads only
    /// world_size + owner_of).
    fn splitter_ctx(ws: usize, owner_of: Vec<i32>) -> ZeROContext {
        ZeROContext {
            stage: ZeROStage::Stage2,
            rank: 0,
            world_size: ws,
            owned_params: Vec::new(),
            owner_of,
            num_params: 0,
            backend: None,
            cuda_aware: false,
            host_backend: None,
        }
    }

    fn item(idx: usize, bytes: usize) -> BucketItem {
        BucketItem {
            idx,
            raw: 0,
            len: bytes / 8,
            bytes,
            off: 0,
        }
    }

    /// Padded size of one sub-group under `ctx`'s owner map.
    fn padded_bytes(ctx: &ZeROContext, sub: &[BucketItem]) -> usize {
        let mut seg = vec![0usize; ctx.world_size];
        for b in sub {
            seg[ctx.owner_of[b.idx] as usize] += b.bytes;
        }
        seg.iter().copied().max().unwrap_or(0) * ctx.world_size
    }

    #[test]
    fn test_scatter_subgroups_small_group_stays_whole() {
        let ctx = splitter_ctx(2, vec![0, 1, 0, 1]);
        let items: Vec<_> = (0..4).map(|i| item(i, 1024)).collect();
        let subs = split_scatter_subgroups(&ctx, &items, 25 * 1024 * 1024, 8);
        assert_eq!(subs.len(), 1, "small group must not split");
        assert_eq!(subs[0].len(), 4);
        assert!(subs[0].iter().all(|b| b.off == 0), "no chunking expected");
    }

    #[test]
    fn test_scatter_subgroups_padded_cap_respected() {
        // 8 params × 8KB, alternating owners, cap 32KB: each sub-group's
        // padded buffer (max seg × ws) must stay ≤ cap.
        let ctx = splitter_ctx(2, vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let items: Vec<_> = (0..8).map(|i| item(i, 8 * 1024)).collect();
        let cap = 32 * 1024;
        let subs = split_scatter_subgroups(&ctx, &items, cap, 8);
        assert!(subs.len() > 1, "cap must force multiple sub-groups");
        let mut seen = 0usize;
        for sub in &subs {
            assert!(!sub.is_empty());
            assert!(
                padded_bytes(&ctx, sub) <= cap,
                "padded sub-group exceeds cap"
            );
            seen += sub.len();
        }
        assert_eq!(seen, 8, "every member exactly once");
    }

    #[test]
    fn test_scatter_subgroups_chunks_giant_tensor() {
        // M3's real case: ONE tensor larger than cap/ws (the 1B embedding
        // gradient). It must chunk into contiguous byte ranges that cover
        // the tensor exactly, each fitting a padded sub-group.
        let ctx = splitter_ctx(2, vec![0, 1]);
        let big = 100 * 1024; // owner 0
        let small = 4 * 1024; // owner 1
        let items = vec![item(0, big), item(1, small)];
        let cap = 16 * 1024; // item_cap = 8KB
        let subs = split_scatter_subgroups(&ctx, &items, cap, 8);
        let chunks: Vec<_> = subs.iter().flatten().filter(|b| b.idx == 0).collect();
        assert!(chunks.len() > 1, "giant tensor must chunk");
        // Contiguous cover: sorted-by-off chunks tile [0, big).
        let mut offs: Vec<(usize, usize)> = chunks.iter().map(|b| (b.off, b.bytes)).collect();
        offs.sort_unstable();
        let mut expect = 0usize;
        for (off, bytes) in &offs {
            assert_eq!(*off, expect, "chunk ranges must tile contiguously");
            assert_eq!(bytes % 8, 0, "chunks must stay element-aligned");
            expect = off + bytes;
        }
        assert_eq!(expect, big, "chunks must cover the whole tensor");
        for sub in &subs {
            assert!(padded_bytes(&ctx, sub) <= cap);
        }
        // Determinism: identical inputs → identical plan (ranks never talk).
        let again = split_scatter_subgroups(&ctx, &items, cap, 8);
        assert_eq!(subs.len(), again.len());
        for (a, b) in subs.iter().zip(&again) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                assert_eq!((x.idx, x.off, x.bytes), (y.idx, y.off, y.bytes));
            }
        }
    }
}
