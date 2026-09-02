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
///
/// Item C extends the same instrument to stage 3: tensor-granular sharded
/// params bump it from `emit_owner_gated_moment` exactly as stages 1/2 do,
/// and elementwise params bump it by their SLICE size from
/// `nsl_zero3_alloc_elem_moment`. REPLICATED params (tied/epilogue/
/// view-rooted) are deliberately NOT counted on any rank — counting them
/// would break the `r0 + r1 == full` partition identity the gates assert,
/// since every rank holds a full copy of those by construction.
///
/// That exclusion is correct for the identity and BLINDING for everything
/// else: it is why `assert_moment_partition` stays green no matter how large
/// the replicated remainder grows. `ZERO_REPL_OPTIM_ELEMS` below exists so
/// the excluded half is still measured.
pub static ZERO_OPTIM_ELEMS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The moment elements this rank holds as a FULL REPLICA — the half
/// `ZERO_OPTIM_ELEMS` deliberately drops.
///
/// Why a second counter instead of folding these into the first: the gates
/// assert `r0 + r1 == full` over `optim_elems`, and a replicated param
/// contributes `numel` to EVERY rank but only `numel` to the ws=1 reference,
/// so counting it there breaks the identity by exactly one replica per extra
/// rank. Keeping the two surfaces in separate counters preserves the
/// partition proof AND makes the unsharded remainder observable, which it
/// was not before: under `--zero-elementwise` the epilogue set (embedding /
/// final norm / LM head — every param with no `blocks.N` key, so not a
/// corner case) keeps full f32 m and v on every rank, and no instrument in
/// the tree reported it. On the shipped test fixture that is 8320 of 74112
/// moment elements (11%) at ws=2, and the fraction GROWS with world size
/// because the counted half shrinks as 1/ws while this one is constant.
///
/// Unlike `ZERO_OPTIM_ELEMS` this is a per-rank replica count, so the
/// identity to assert on it is `r0 == r1 == full`, not `r0 + r1 == full`.
pub static ZERO_REPL_OPTIM_ELEMS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Read `(rank, world_size, all_reduce_count, broadcast_count, optim_elems,
/// bucket_members, repl_optim_elems)` for the atexit report. rank/ws are -1
/// when no context was initialized.
pub fn zero_counter_snapshot() -> (i64, i64, u64, u64, u64, u64, u64) {
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
        ZERO_REPL_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// D3 v2: record that an owned optimizer-moment tensor of `tensor_ptr` was
/// allocated, adding its element count to this rank's running total. Emitted
/// by codegen inside the owner-gated allocation branch (once per m and per v
/// tensor). A null pointer (the non-owned placeholder) contributes nothing,
/// so a mis-wired gate that noted non-owned slots would over-count and trip
/// the G3 gate. Returns the running total (for tests/debug).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero_note_optim_alloc(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return ZERO_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed) as i64;
    }
    let numel = crate::tensor::NslTensor::from_ptr(tensor_ptr).len.max(0) as u64;
    (ZERO_OPTIM_ELEMS.fetch_add(numel, std::sync::atomic::Ordering::Relaxed) + numel) as i64
}

/// Record a moment tensor this rank holds as a FULL REPLICA (the
/// `MomentFill::Full` arm: tied / view-rooted / epilogue params).
///
/// Deliberately a SEPARATE counter from `nsl_zero_note_optim_alloc` — see
/// `ZERO_REPL_OPTIM_ELEMS` for why folding the two would break the
/// `r0 + r1 == full` partition identity. Emitted unconditionally on the
/// replicated arm (there is no owner gate to sit inside), so a null pointer
/// here means the moment allocation itself failed rather than that this rank
/// legitimately skipped it — it still contributes nothing, matching its
/// sibling's convention.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero_note_replicated_optim_alloc(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return ZERO_REPL_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed) as i64;
    }
    let numel = crate::tensor::NslTensor::from_ptr(tensor_ptr).len.max(0) as u64;
    (ZERO_REPL_OPTIM_ELEMS.fetch_add(numel, std::sync::atomic::Ordering::Relaxed) + numel) as i64
}

/// Write one pre-formatted line to stderr in a SINGLE `write_all`.
///
/// Every per-rank message in `nsl_zero_init`'s `ws > 1` arms is emitted by
/// all ranks simultaneously into the SAME inherited stderr pipe, and
/// `eprintln!` issues one write(2) per format FRAGMENT. Two ranks therefore
/// interleave mid-string. This is not theoretical — the NCCL same-device
/// refusal came out as
///
/// ```text
/// ...init failed on rank nsl_zero_init: ...failed on rank 1: 0: nccl...
/// ```
///
/// so the substring "on rank 0" did not exist anywhere in the output and a
/// gate asserting per-rank symmetry failed on a run where both ranks HAD
/// refused correctly. It is nondeterministic, so it presents as a flake.
///
/// `args.rs`'s `[zero]` counter line and the `[zero3]` teardown line already
/// solved this the same way; the init-path messages were simply never
/// converted, because until NCCL was actually exercised nothing parsed them.
/// A single `write_all` under PIPE_BUF is atomic on POSIX pipes.
fn emit_rank_line(mut line: String) {
    use std::io::Write;
    if !line.ends_with('\n') {
        line.push('\n');
    }
    let _ = std::io::stderr().lock().write_all(line.as_bytes());
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
#[unsafe(no_mangle)]
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
        emit_rank_line(format!(
            "nsl_zero_init: rank {rank}: NSL_SIMULATED_TP=0 requests a real \
             collective backend, but no NCCL/inter-device transport is built \
             into this runtime — refusing rather than silently simulating."
        ));
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
            emit_rank_line(format!(
                "nsl_zero_init: rank {rank}: unknown NSL_COLLECTIVES backend \
                 '{other}' (expected 'sim', 'sim-gpu', or 'nccl')"
            ));
            return -2;
        }
    };

    let (backend, cuda_aware): (
        Option<Box<dyn crate::tensor_parallel::collective::CollectiveBackend>>,
        bool,
    ) = if ws > 1 {
        let Ok(shm_path) = std::env::var("NSL_TP_SHM_PATH") else {
            emit_rank_line(format!(
                "nsl_zero_init: world_size={ws} but NSL_TP_SHM_PATH is not \
                 set — run through `nsl run --devices {ws}` (the spawner \
                 creates the shared-memory file and sets the rank env)"
            ));
            return -3;
        };
        let (shm_ptr, shm_len) = crate::tensor_parallel::ffi::open_shm(&shm_path);
        match want {
            WantBackend::Nccl => match make_nccl_backend(rank, ws, shm_ptr, shm_len) {
                Ok(b) => {
                    emit_rank_line(format!(
                        "[zero] rank {rank}: NCCL communicator up \
                         (ws={ws}, CUDA-aware collectives)"
                    ));
                    (Some(b), true)
                }
                Err(e) => {
                    emit_rank_line(format!(
                        "nsl_zero_init: NCCL backend init failed on rank {rank}: {e}"
                    ));
                    return -2;
                }
            },
            WantBackend::SimGpu => match make_sim_gpu_backend(rank, ws, shm_ptr, shm_len) {
                Ok(b) => {
                    emit_rank_line(format!(
                        "[zero] rank {rank}: sim-gpu staged collectives up \
                         (ws={ws}, CUDA-aware TEST backend)"
                    ));
                    (Some(b), true)
                }
                Err(e) => {
                    emit_rank_line(format!(
                        "nsl_zero_init: sim-gpu backend init failed on rank {rank}: {e}"
                    ));
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
    // Review M1: the bucket cap SHAPES the per-step collective sequence
    // (sub-group count, chunking) — a per-rank NSL_ZERO_BUCKET_MB
    // divergence would pass a sizes/owners-only hash and then deadlock the
    // spin-barrier (rank A issues 12 scatters/step, rank B issues 6).
    // Hash it with everything else that determines the sequence.
    mix(zero_bucket_cap_bytes() as u64);
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
    let matched = root == local;
    if !matched {
        eprintln!(
            "nsl: ZeRO partition plan MISMATCH on rank {}: local hash \
             {local:#018x} != rank-0 hash {root:#018x} — ranks disagree on \
             param sizes, ownership or bucket config; refusing to train a \
             torn model",
            ctx.rank
        );
    }
    // Review L2 (symmetric refusal): without this, only the MISMATCHING
    // ranks refuse — rank 0 trivially matches itself, proceeds, and hangs
    // at the next collective. All-reduce a match flag so every rank
    // (including the root) refuses together.
    let mut flag: f64 = if matched { 1.0 } else { 0.0 };
    let rc = host.all_reduce_sum(
        &flag as *const f64 as *const std::ffi::c_void,
        &mut flag as *mut f64 as *mut std::ffi::c_void,
        1,
        crate::tensor_parallel::collective::DTYPE_F64,
        std::ptr::null_mut(),
    );
    if rc != 0 {
        eprintln!(
            "nsl: zero plan verification all_reduce failed rc={rc} on rank {}",
            ctx.rank
        );
        return false;
    }
    if flag < ctx.world_size as f64 {
        if matched {
            eprintln!(
                "nsl: ZeRO partition plan MISMATCH reported by a peer rank — \
                 refusing on rank {} too (symmetric shutdown)",
                ctx.rank
            );
        }
        return false;
    }
    true
}

/// Check if this rank owns the given parameter index.
/// Returns 1 if owned, 0 if not, -1 if ZeRO not initialized.
#[unsafe(no_mangle)]
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

/// Item 8 × D3: split a FullBuffer optimizer step into the subset this rank
/// owns, so the batched launcher can be used under `--zero-stage 1/2`.
///
/// Codegen's per-param ZeRO loop does exactly two things per parameter: the
/// owner runs the update (whose kernel also zeroes `m_partial`), and a
/// NON-owner zeroes its `m_partial` and nothing else — the accumulator must
/// not survive into the next window, and a true zero-fill rather than a
/// multiply-by-0.0, because `x * 0.0` keeps a NaN/Inf alive forever.
///
/// This performs that same partition once: non-owners' accumulators are
/// zeroed here, and the returned list of owned indices feeds
/// `nsl_fase_fused_adamw_step_multi_idx`, the subset launcher that already
/// exists for CSLA layer groups. The caller owns the returned list and must
/// `nsl_list_free` it.
///
/// Why an owner SUBSET and not an owner-aware batched launcher: under stages
/// 1/2 a non-owned parameter's moment buffers are NULL placeholders
/// (`emit_owner_gated_moment` allocates nothing for them), so the batched
/// path must never see one. Handing it only owners keeps
/// `fase_multi_impl`'s null assert as a live belt instead of something to
/// weaken.
///
/// **Deliberate divergence from the loop it replaces.** `nsl_zero_owns_param`
/// returns -1 when no ZeRO context is installed, which the old loop's
/// `owns == 1` test read as "not owned" — so a program built with
/// `--zero-stage` whose `nsl_zero_init` had failed would zero every
/// accumulator, update nothing, and train silently forever. There is no
/// configuration in which that is correct, so this aborts instead.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero_owned_step_indices(accum_list: i64, num_params: i64) -> i64 {
    let n = num_params.max(0);
    // Snapshot ownership under the lock, then release it: the zero-fill below
    // reaches into tensor code and must not run with ZERO_CTX held.
    let owned: Vec<bool> = {
        let guard = ZERO_CTX.lock().unwrap();
        let Some(ctx) = guard.as_ref() else {
            if n > 0 {
                eprintln!(
                    "nsl_zero_owned_step_indices: no ZeRO context is installed, \
                     but this program was compiled with --zero-stage and is \
                     asking which of {n} parameter(s) this rank owns. \
                     nsl_zero_init must have failed (see its message above); \
                     continuing would zero every gradient accumulator and \
                     update no parameter at all — aborting instead."
                );
                std::process::abort();
            }
            return crate::list::nsl_list_new();
        };
        (0..n as usize)
            .map(|i| ctx.owned_params.contains(&i))
            .collect()
    };

    let list = crate::list::nsl_list_new();
    let mut idxs: Vec<usize> = Vec::new();
    for (i, is_owner) in owned.iter().enumerate() {
        if *is_owner {
            crate::list::nsl_list_push(list, i as i64);
            idxs.push(i);
        } else if accum_list != 0 {
            let mp = crate::list::nsl_list_get(accum_list, i as i64);
            if mp != 0 {
                crate::tensor::nsl_tensor_zero_inplace(mp);
            }
        }
    }

    // Publish the actual index SET once, under NSL_ZERO_COUNTER=1.
    //
    // A gate can otherwise only compare COUNTS, and counts cannot tell a
    // partition from an overlap plus a matching gap: ranks owning {0,1,2} and
    // {2,3,4} of five parameters sum to the same 6 as {0,1,2} and {3,4} — one
    // parameter stepped twice, one never stepped, both totals right. The sets
    // themselves are the only thing that settles it (review finding).
    //
    // Once per process, so a many-step run does not flood stderr; formatted
    // and written in one syscall because every rank shares this pipe.
    static PUBLISHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !PUBLISHED.swap(true, std::sync::atomic::Ordering::Relaxed)
        && std::env::var("NSL_ZERO_COUNTER").ok().as_deref() == Some("1")
    {
        let rank = ZERO_CTX.lock().unwrap().as_ref().map(|c| c.rank).unwrap_or(0);
        let line = format!(
            "[zero] owned-step rank={rank} n={n} idx={}\n",
            idxs.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        use std::io::Write;
        let _ = std::io::stderr().lock().write_all(line.as_bytes());
    }
    list
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
/// Item 11: the elementwise stage-3 gradient path bumps this too (its gate
/// asserts it is nonzero at stage 3, where the tensor-granular path never
/// scatters).
pub static ZERO_REDUCE_SCATTER_COUNT: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Item 11 (elementwise ZeRO-3): all_gather collectives that actually ran
/// (ws>1, rc==0 — the same counting rule as the others). The elementwise
/// gather is the only producer; the tensor-granular gather broadcasts.
pub static ZERO_ALL_GATHER_COUNT: std::sync::atomic::AtomicU64 =
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
        4..=6 => 1,  // i8/fp8
        9 => 4,      // i32 (DTYPE_I32)
        _ => 8,      // default to f64 width
    };
    let total_bytes = n * byte_width;

    if tensor.device != 0 {
        // GPU grad buffers are always f32 (`zeros_like(param)` on GPU); any
        // non-f32 GPU buffer here means a wiring bug upstream. Refuse instead
        // of guessing widths (mirrors nsl_zero_reduce_grads).
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
#[unsafe(no_mangle)]
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

// ═══════════════════════════════════════════════════════════════════════
// P3 ZeRO-3 (items 12-14): TENSOR-GRANULAR parameter sharding.
//
// Each parameter tensor is OWNED by exactly one rank (the same
// byte-balanced owner map stages 1/2 use). At rest, the owner keeps the
// full tensor device-resident and every other rank holds NOTHING
// (`data == null`) — the per-rank at-rest footprint is ~1/ws of total
// parameter bytes. Just-in-time materialization rides the CSLA
// layer-major schedule through the weight-stream residency machinery:
// codegen's per-layer upload/evict sites redirect here (no host mirrors —
// the fill primitive is a collective broadcast from the owner, the spill
// primitive a plain free). Item 14's ordering falls out of the schedule:
// layer L's backward completes (source AD knows exactly when) → L's
// gradient slots all-reduce → owner updates → release → the NEXT range
// head gathers its layer. With the CPU-shm / sim-gpu backends every
// collective is synchronous, so the ordering is validated bit-exactly;
// true comm/compute overlap needs the NCCL backend's streams (documented
// follow-up, not reachable on a 1-GPU box).
//
// Granularity note: this is the per-PARAMETER FSDP variant (owner =
// singleton group per tensor), not elementwise 1/ws sharding of every
// tensor. It reuses the stage-1/2 owner maps, keeps `muon_step` (which
// needs whole matrices) compatible, and needs no gather in the optimizer.
// Elementwise sharding would ride `CollectiveBackend::all_gather` later.
// ═══════════════════════════════════════════════════════════════════════

/// Item 12: the residency of THIS RANK's replica of a parameter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i64)]
pub enum ParameterResidency {
    /// Not sharded — full replica on every rank (view-rooted / tied /
    /// epilogue params; also everything when ZeRO-3 is off). Updated
    /// identically on all ranks from all-reduced gradients.
    Replicated = 0,
    /// This rank OWNS the tensor: its full data is the authoritative shard.
    ShardedResident = 1,
    /// Non-owner holding a temporary gathered (broadcast) copy for the
    /// current layer window; released after the layer's last use.
    GatheredTemporary = 2,
    /// Non-owner at rest: no local bytes (`data == null`).
    Evicted = 3,
}

/// Item 11: storage for one rank's persistent 1/ws shard (and its reduced
/// gradient slice). `Vec<u64>` keeps host storage 8-byte aligned for f64
/// slices; device storage is an `alloc_managed` pointer freed explicitly at
/// teardown (entries live in a static, so Drop cannot be cfg(cuda)-gated).
enum ShardBuf {
    Host(Vec<u64>),
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Dev(i64),
}

impl ShardBuf {
    fn alloc(bytes: usize, device: u8) -> ShardBuf {
        if device != 0 {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::inner::ensure_context();
                return ShardBuf::Dev(crate::cuda::inner::alloc_managed(bytes) as i64);
            }
            #[cfg(not(feature = "cuda"))]
            unreachable!("device tensor in a non-cuda build");
        }
        ShardBuf::Host(vec![0u64; bytes.div_ceil(8)])
    }

    fn ptr(&self) -> *mut std::ffi::c_void {
        match self {
            ShardBuf::Host(v) => v.as_ptr() as *mut std::ffi::c_void,
            ShardBuf::Dev(p) => *p as *mut std::ffi::c_void,
        }
    }

    fn free(self) {
        match self {
            ShardBuf::Host(_) => {}
            ShardBuf::Dev(_p) =>
            {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::ensure_context();
                    crate::cuda::inner::free_managed(_p as *mut std::ffi::c_void);
                }
            }
        }
    }
}

/// Item 11: the elementwise 1/ws shard of one parameter. Every rank holds
/// `shard = numel/ws` elements at flat offset `rank*shard`; the slice is the
/// authoritative storage (there is no owner), the full tensor is a transient
/// reconstructed by all_gather for the layer window.
struct ElemShard {
    slice: ShardBuf,
    /// The window's reduce_scattered (averaged) gradient slice. Allocated on
    /// first reduce, reused across windows (shard size never changes).
    grad: Option<ShardBuf>,
    /// Reduce/step handshake: set by `nsl_zero3_reduce_grad_slot`, consumed
    /// by `nsl_zero3_elem_adamw_step`. A step without a fresh reduce is a
    /// codegen ordering bug and aborts.
    grad_ready: bool,
    /// Elements per rank.
    shard: usize,
    /// Element size of the FULL working tensor (8 = f64, 4 = f32) — the
    /// region-offset unit for the gathered transient and the moments. The
    /// SLICE's element size can DIFFER: it is 2 (bf16) whenever
    /// `sr_param_idx` is set, `elem_bytes` otherwise. Any code sizing slice
    /// bytes must branch on `sr_param_idx`, not reuse this field.
    elem_bytes: usize,
    /// Item 16×11 composed mode (`--param-dtype bf16-sr --zero-elementwise`):
    /// the param's stable SR counter-block index. When set, the SLICE is the
    /// bf16 AUTHORITATIVE storage (2 bytes/elem, sized at carve) — the
    /// gathered full tensor stays an f32 working transient (widened at
    /// gather, never written back), and the step is the SR kernel against
    /// the slice.
    sr_param_idx: Option<u64>,
    /// Item 16×11: f32 staging for the bf16 slice's all_gather SEND buffer
    /// at ws>1 (the collective never sees a 2-byte dtype). Allocated on the
    /// first gather and reused across windows — the shard size never
    /// changes — exactly like `grad` above. A per-window alloc/free would
    /// pay two allocator-lock round trips per param per window for a buffer
    /// of constant size. Freed with the entry at teardown.
    sr_send: Option<ShardBuf>,
}

struct Zero3Entry {
    /// param_list index — the cross-rank collective ordering key for
    /// gather_all/release_all (heap pointers are per-process).
    idx: usize,
    owner: i32,
    state: ParameterResidency,
    /// Item 11: set by `nsl_zero3_mark_elementwise` (plan-driven, before the
    /// first registration); the register carves the slice.
    elem_pending: bool,
    /// Item 16×11: the plan entry's storage decision, carried by the same
    /// mark — true means the carved slice is bf16-authoritative and the
    /// update is the fused SR step. Pinned once the slice exists (a re-mark
    /// that disagrees aborts): slice bytes cannot change dtype mid-schedule.
    elem_sr: bool,
    /// Present once the first registration carved this rank's slice.
    elem: Option<ElemShard>,
}

static ZERO3_ACTIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static ZERO3_TABLE: Mutex<Option<std::collections::HashMap<i64, Zero3Entry>>> = Mutex::new(None);
static ZERO3_GATHERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ZERO3_RELEASES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Item 11 anti-vacuity: elementwise params carved / elementwise fused
/// steps executed. Reported on the teardown line so the gate can assert the
/// elementwise machinery ran (and, in the tensor-granular arm, did not).
static ZERO3_ELEM_PARAMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ZERO3_ELEM_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Item 16×11 anti-vacuity: bf16-SR-authoritative slices carved / composed
/// SR steps executed. Separate from the plain elementwise counters so the
/// composed gate can assert the SR machinery ran (a composed run whose
/// params all silently took the plain f32 arm would train without
/// stochastic rounding and read green on every other counter).
static ZERO3_SR_ELEM_PARAMS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static ZERO3_SR_ELEM_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Owner-only moments anti-vacuity: SLICE-sized optimizer moments handed out
/// by `nsl_zero3_alloc_elem_moment`. Reported on the teardown line — a fill
/// loop that silently fell back to full `zeros_like` reads 0 here while
/// loss parity stays green, which is the exact failure this counter exists
/// to catch.
static ZERO3_ELEM_MOMENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Codegen calls this once per train block when `--zero-stage 3` is baked.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_enable() -> i64 {
    ZERO3_ACTIVE.store(true, std::sync::atomic::Ordering::SeqCst);
    let mut guard = ZERO3_TABLE.lock().unwrap();
    guard.get_or_insert_with(std::collections::HashMap::new);
    eprintln!(
        "[zero3] tensor-granular parameter sharding enabled: owners keep \
         their params device-resident, non-owners hold nothing at rest and \
         gather per layer window (optimizer moments are OWNER-ONLY too — \
         allocation is deferred to the first window and every rank holds \
         m/v for its own share of the sharded set)"
    );
    0
}

/// True while a train block runs under `--zero-stage 3` — the
/// weight-stream entry points branch on this.
pub(crate) fn zero3_active() -> bool {
    ZERO3_ACTIVE.load(std::sync::atomic::Ordering::SeqCst)
}

/// Map a parameter tensor pointer to its param-list index (owners come
/// from the stage-1/2 partition). Emitted right after
/// `nsl_zero_partition_bytes` for every param.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_note_param(tensor_ptr: i64, idx: i64) -> i64 {
    if tensor_ptr == 0 || idx < 0 {
        return -1;
    }
    let owner = {
        let guard = ZERO_CTX.lock().unwrap();
        let Some(ctx) = guard.as_ref() else {
            eprintln!("nsl_zero3_note_param: ZeRO context not initialized");
            return -1;
        };
        ctx.owner_of
            .get(idx as usize)
            .copied()
            .unwrap_or((idx as usize % ctx.world_size.max(1)) as i32)
    };
    let mut guard = ZERO3_TABLE.lock().unwrap();
    guard.get_or_insert_with(std::collections::HashMap::new).insert(
        tensor_ptr,
        Zero3Entry {
            idx: idx as usize,
            owner,
            state: ParameterResidency::Replicated,
            elem_pending: false,
            elem_sr: false,
            elem: None,
        },
    );
    0
}

/// Item 11: mark a noted param for ELEMENTWISE 1/ws sharding — the plan
/// decided this at compile time (streamed, AdamW-stepped, numel divisible by
/// world_size), and codegen emits this right before the registration belt,
/// exactly like `nsl_sr_bf16_note_param`. Idempotent (the belt re-runs every
/// micro-batch). The slice itself is carved at the first registration, where
/// the admission guard has already vetted the tensor.
///
/// Item 16×11: `sr` (0/1) carries the plan entry's STORAGE decision — 1 means
/// this param's slice is bf16-authoritative and its update is the fused SR
/// step. It is an argument rather than something the carve infers, because
/// the plan is the single source of truth for storage (the same doctrine as
/// `needs_sr_note`); inferring it from the SR backend's pointer-keyed pending
/// map would couple two backends through a heap address that outlives the
/// tensor. The SR counter block is `idx` — the same value
/// `nsl_sr_bf16_note_param` records for the mirror path, which is what makes
/// a composed slice's dither stream identical to the baseline's.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_mark_elementwise(tensor_ptr: i64, idx: i64, sr: i64) -> i64 {
    let ws = {
        let guard = ZERO_CTX.lock().unwrap();
        let Some(ctx) = guard.as_ref() else {
            eprintln!("nsl_zero3_mark_elementwise: ZeRO context not initialized");
            return -1;
        };
        ctx.world_size.max(1)
    };
    let t = crate::tensor::NslTensor::from_ptr(tensor_ptr);
    let numel = t.len.max(0) as usize;
    if numel == 0 || !numel.is_multiple_of(ws) {
        // The plan's eligibility already excluded ragged params — reaching
        // here means codegen and runtime disagree about the shard layout.
        eprintln!(
            "[zero3] FATAL: param {idx} marked elementwise with numel {numel} \
             not divisible by world_size {ws} — the compile-time eligibility \
             and this runtime have drifted"
        );
        std::process::abort();
    }
    let mut guard = ZERO3_TABLE.lock().unwrap();
    let Some(e) = guard.as_mut().and_then(|m| m.get_mut(&tensor_ptr)) else {
        eprintln!(
            "[zero3] FATAL: elementwise mark of an un-noted param tensor \
             {tensor_ptr} — codegen must emit nsl_zero3_note_param first"
        );
        std::process::abort();
    };
    if !e.elem_pending {
        static ARMED: std::sync::Once = std::sync::Once::new();
        ARMED.call_once(|| {
            eprintln!(
                "[zero3] elementwise sharding armed: eligible params live as \
                 1/{ws} slices per rank, gathers ride all_gather, gradients \
                 reduce_scatter, every rank steps its own slice — and m/v are \
                 SLICE-sized too (1/{ws} of the moment surface per rank, \
                 allocated at the first window register belt)"
            );
        });
        e.elem_pending = true;
    }
    // Item 16×11: recorded on every mark (not just the arming one) so a
    // re-mark can never silently disagree with the first about storage.
    let sr = sr != 0;
    if sr && !crate::sr_bf16::srbf16_active() {
        eprintln!(
            "[zero3] FATAL: param {idx} marked bf16-sr elementwise but the SR \
             backend was never enabled — the plan's storage decision and the \
             emitted enable belt have drifted"
        );
        std::process::abort();
    }
    if e.elem.is_some() && e.elem_sr != sr {
        eprintln!(
            "[zero3] FATAL: param {idx} re-marked with storage sr={sr} after \
             its slice was carved as sr={} — the slice bytes cannot change \
             dtype under a live schedule",
            e.elem_sr
        );
        std::process::abort();
    }
    e.elem_sr = sr;
    0
}

/// Item C: allocate THIS RANK's slice-sized optimizer moment for an
/// elementwise-sharded parameter, and record its elements against the
/// `optim_elems` shrink counter.
///
/// Shape/dtype/device come from the `ElemShard` the CARVE populated
/// (`zero3_register`), never from the θ tensor's own `len` — at rest an
/// elementwise param's `data` is null and its `len` is still the FULL
/// element count, so sizing a moment off θ would hand back exactly the
/// replicated buffer this whole change exists to delete. That ordering
/// (mark → register/carve → moment fill) is load-bearing: a call before the
/// carve aborts loudly rather than silently allocating full.
///
/// Returns a fresh 1-D `[shard]` tensor owned by the caller (codegen stores
/// it in the moment list; the end-of-train free loop releases it through
/// `nsl_tensor_free` like every other moment).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_alloc_elem_moment(theta_ptr: i64, idx: i64) -> i64 {
    if theta_ptr == 0 {
        eprintln!(
            "[zero3] FATAL: elementwise moment allocation for a null \
             parameter slot (param {idx})"
        );
        std::process::abort();
    }
    let th = unsafe { &*(theta_ptr as *const crate::tensor::NslTensor) };
    let (shard, elem_bytes) = {
        let tbl = ZERO3_TABLE.lock().unwrap();
        let Some(e) = tbl.as_ref().and_then(|map| map.get(&theta_ptr)) else {
            eprintln!(
                "[zero3] FATAL: elementwise moment allocation for untracked \
                 tensor {theta_ptr} (param {idx})"
            );
            std::process::abort();
        };
        if e.idx != idx as usize {
            eprintln!(
                "[zero3] FATAL: elementwise moment param-index mismatch \
                 (tensor says {}, codegen says {idx})",
                e.idx
            );
            std::process::abort();
        }
        let Some(s) = e.elem.as_ref() else {
            eprintln!(
                "[zero3] FATAL: elementwise moment allocation for param {idx}, \
                 which was never carved — the deferred moment fill must be \
                 emitted AFTER the weight-stream register belt"
            );
            std::process::abort();
        };
        (s.shard, s.elem_bytes)
    };
    // The carve derived elem_bytes from θ's dtype; if the two disagree the
    // moment would be sized for one storage and stepped as another.
    let expect_bytes = if th.dtype == 0 { 8 } else { 4 };
    if shard == 0 || (th.dtype != 0 && th.dtype != 1) || elem_bytes != expect_bytes {
        eprintln!(
            "[zero3] FATAL: elementwise moment for param {idx} has an \
             inconsistent slice descriptor (shard={shard} elem_bytes={} \
             theta dtype={})",
            elem_bytes, th.dtype
        );
        std::process::abort();
    }
    let bytes = shard * elem_bytes;
    let shape =
        crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = shard as i64 };
    let strides = crate::tensor::NslTensor::compute_strides(shape, 1);
    let data: *mut std::ffi::c_void = if th.device != 0 {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::inner::ensure_context();
            let p = crate::cuda::inner::alloc_managed(bytes);
            crate::cuda::inner::memset_d8(p, bytes);
            p
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!(
                "[zero3] FATAL: device elementwise moment in a non-cuda build"
            );
            std::process::abort();
        }
    } else {
        crate::memory::checked_alloc_zeroed(bytes) as *mut std::ffi::c_void
    };
    let t = Box::new(crate::tensor::NslTensor::new(
        data,
        shape,
        strides,
        1,
        shard as i64,
        th.device,
        th.dtype,
        1,
        0,
    ));
    let ptr = crate::tensor::NslTensor::publish(t);
    ZERO3_ELEM_MOMENTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // The note lives HERE, not at the call site: `optim_elems` is the only
    // instrument that distinguishes owner-only moments from replicated ones
    // (loss parity is invariant to the difference), so it must not be
    // possible to add an allocation arm and forget the counter.
    nsl_zero_note_optim_alloc(ptr);
    ptr
}

/// Item 12 introspection (tests/diagnostics): residency of this rank's
/// replica as the enum's i64 value; -1 = untracked pointer.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_residency(tensor_ptr: i64) -> i64 {
    let guard = ZERO3_TABLE.lock().unwrap();
    guard
        .as_ref()
        .and_then(|t| t.get(&tensor_ptr))
        .map(|e| e.state as i64)
        .unwrap_or(-1)
}

/// Gather / release counters (gate anti-vacuity probes).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_gather_count() -> i64 {
    ZERO3_GATHERS.load(std::sync::atomic::Ordering::Relaxed) as i64
}
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_release_count() -> i64 {
    ZERO3_RELEASES.load(std::sync::atomic::Ordering::Relaxed) as i64
}

/// First registration under ZeRO-3 (weight-stream `register` redirect):
/// the owner keeps its device data (ShardedResident); every other rank
/// frees its replica (Evicted). Re-registration at window start re-evicts
/// non-owners (idempotent for the owner). No host mirror is ever created.
// Referenced from weight_stream's cuda-gated register redirect only.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn zero3_register(tensor_ptr: i64) {
    let t = crate::tensor::NslTensor::from_ptr(tensor_ptr);
    let (rank, ws) = {
        let guard = ZERO_CTX.lock().unwrap();
        guard
            .as_ref()
            .map(|c| (c.rank, c.world_size.max(1)))
            .unwrap_or((0, 1))
    };
    let mut guard = ZERO3_TABLE.lock().unwrap();
    let table = guard.get_or_insert_with(std::collections::HashMap::new);
    let Some(e) = table.get_mut(&tensor_ptr) else {
        eprintln!(
            "[zero3] FATAL: register of an un-noted param tensor {tensor_ptr} \
             — codegen must emit nsl_zero3_note_param for every param"
        );
        std::process::abort();
    };
    // First-registration admission (parity with weight-stream's #397 guard):
    // only plain owning non-slab tensors may shard — a view/slab param would
    // corrupt on free-and-regather.
    if e.state == ParameterResidency::Replicated
        && (t.owns_data == 0 || t.data_owner != 0 || t.slab_managed != 0)
    {
        eprintln!(
            "[zero3] refusing to shard tensor {tensor_ptr}: owns_data={} \
             data_owner={} slab_managed={} — only plain owning tensors shard",
            t.owns_data, t.data_owner, t.slab_managed
        );
        std::process::abort();
    }
    // Item 11: elementwise registration — carve this rank's slice out of the
    // initial full replica, then drop the full bytes on EVERY rank (there is
    // no owner; the slice is the authoritative storage). Idempotent once
    // carved: re-registration at window start only needs the eviction reset
    // below the tensor-granular arms never see for elementwise entries.
    if e.elem_pending {
        if e.elem.is_none() {
            let numel = t.len.max(0) as usize;
            // The one dtype guard on the elem path that runs BEFORE bytes
            // move: a 2-byte dtype (bf16, id >= 256) would mis-slice here,
            // and gather's f32-only check fires only at ws>1, after the
            // carve already copied. The PARAM tensor is always f64/f32 —
            // under composed bf16-sr the tensor stays a plain f32 working
            // transient and only the SLICE storage below is bf16.
            if t.dtype != 0 && t.dtype != 1 {
                eprintln!(
                    "[zero3] FATAL: elementwise carve of dtype {} (param {}) — \
                     only f64/f32 parameters shard elementwise",
                    t.dtype, e.idx
                );
                std::process::abort();
            }
            // Item 16×11: under composed `--param-dtype bf16-sr`, the slice
            // is the bf16 AUTHORITATIVE storage. Carve by the same f32→bf16
            // cast `srbf16_register` applies to the full mirror, so every
            // rank's slice equals the single-rank mirror's region
            // bit-for-bit. GPU/f32-only, exactly like the mirror backend.
            //
            // The decision (and the counter block) come from the plan via
            // the mark, NOT from the SR backend's pointer-keyed pending map:
            // that map outlives its tensors, so a recycled heap address in a
            // later train block could hand this carve another parameter's
            // dither stream. `e.idx` is the same index the mirror path's
            // note records, which is what keeps the streams identical.
            let sr_param_idx = if e.elem_sr { Some(e.idx as u64) } else { None };
            if sr_param_idx.is_some() {
                if t.device == 0 || t.dtype != 1 {
                    eprintln!(
                        "[zero3] FATAL: bf16-sr elementwise carve of param {} \
                         requires a GPU-resident f32 tensor (device={}, \
                         dtype={}) — the SR backend is GPU-only",
                        e.idx, t.device, t.dtype
                    );
                    std::process::abort();
                }
                if numel as u64 >= 1u64 << crate::sr_bf16::SR_PARAM_SHIFT {
                    // The rank offset rides inside the param's counter
                    // block; an oversized param would alias the next
                    // param's dither stream (same guard as register's).
                    eprintln!(
                        "[zero3] FATAL: bf16-sr elementwise param {} has {} \
                         elements — the SR counter block holds 2^{}",
                        e.idx,
                        numel,
                        crate::sr_bf16::SR_PARAM_SHIFT
                    );
                    std::process::abort();
                }
            }
            let elem_bytes = if t.dtype == 0 { 8 } else { 4 };
            let slice_bytes = if sr_param_idx.is_some() { 2 } else { elem_bytes };
            let shard = numel / ws;
            let slice = ShardBuf::alloc(shard * slice_bytes, t.device);
            let off = rank * shard * elem_bytes;
            unsafe {
                if t.device != 0 {
                    #[cfg(feature = "cuda")]
                    {
                        crate::cuda::inner::ensure_context();
                        if sr_param_idx.is_some() {
                            crate::cuda::gpu_cast_raw_f32_to_bf16(
                                (t.data as *const u8).add(off) as u64,
                                slice.ptr() as u64,
                                shard,
                            );
                        } else {
                            crate::cuda::inner::memcpy_dtod(
                                slice.ptr(),
                                (t.data as *const u8).add(off) as *const std::ffi::c_void,
                                shard * elem_bytes,
                            );
                        }
                        crate::cuda::inner::free_managed(t.data);
                    }
                } else {
                    std::ptr::copy_nonoverlapping(
                        (t.data as *const u8).add(off),
                        slice.ptr() as *mut u8,
                        shard * elem_bytes,
                    );
                    // CPU full replicas are heap allocations the runtime
                    // cannot reconstruct a Layout for here — null without
                    // freeing, exactly like the tensor-granular non-cuda
                    // eviction below.
                }
            }
            t.data = std::ptr::null_mut();
            e.elem = Some(ElemShard {
                slice,
                grad: None,
                grad_ready: false,
                shard,
                elem_bytes,
                sr_param_idx,
                sr_send: None,
            });
            e.state = ParameterResidency::Evicted;
            ZERO3_ELEM_PARAMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if sr_param_idx.is_some() {
                ZERO3_SR_ELEM_PARAMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        } else if e.state == ParameterResidency::GatheredTemporary && !t.data.is_null() {
            // Window-start re-register of a still-materialized full copy:
            // drop it (the slice holds the truth), mirroring the non-owner
            // re-evict below. Host transients pair with the gather's
            // checked_alloc_zeroed, exactly like zero3_release's arm.
            if t.device != 0 {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::ensure_context();
                    crate::cuda::inner::free_managed(t.data);
                }
            } else {
                unsafe {
                    crate::memory::checked_free(t.data as *mut u8, t.data_byte_size());
                }
            }
            t.data = std::ptr::null_mut();
            e.state = ParameterResidency::Evicted;
        }
        return;
    }
    if e.owner as usize == rank {
        // Window-start reset (mirrors the non-owner re-evict below): a
        // stale GatheredTemporary from the previous window drops back to
        // ShardedResident so this window's first gather broadcasts.
        e.state = ParameterResidency::ShardedResident;
        return;
    }
    match e.state {
        ParameterResidency::Replicated | ParameterResidency::GatheredTemporary => {
            if !t.data.is_null() {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::ensure_context();
                    crate::cuda::inner::free_managed(t.data);
                }
                t.data = std::ptr::null_mut();
            }
            e.state = ParameterResidency::Evicted;
        }
        ParameterResidency::Evicted => {}
        ParameterResidency::ShardedResident => unreachable!("owner handled above"),
    }
}

/// JIT gather (weight-stream `upload` redirect): symmetric on every rank —
/// the owner contributes its live data as the broadcast source; non-owners
/// allocate a fresh device buffer and receive into it. BOTH sides then
/// mark GatheredTemporary (for the owner it means "this window's gather
/// happened", data unchanged) so the idempotent early-return fires on the
/// SAME calls on every rank — an owner-only repeat broadcast would pair
/// with the other ranks' NEXT collective in the positional spin-barrier
/// (silent corruption / barrier timeout; review finding 1).
pub(crate) fn zero3_gather(tensor_ptr: i64) {
    let t = crate::tensor::NslTensor::from_ptr(tensor_ptr);
    // (owner, is_owner, Some((slice_ptr, shard, elem_bytes, sr)) for
    // elementwise — `sr` marks a bf16-authoritative slice that must be
    // widened to f32 before it can feed the (f32) local copy / collective)
    let (owner, is_owner, elem_info) = {
        let tbl = ZERO3_TABLE.lock().unwrap();
        let Some(e) = tbl.as_ref().and_then(|m| m.get(&tensor_ptr)) else {
            eprintln!("[zero3] FATAL: gather of untracked tensor {tensor_ptr}");
            std::process::abort();
        };
        if e.state == ParameterResidency::GatheredTemporary {
            return; // already materialized this window
        }
        if e.state == ParameterResidency::Replicated {
            // Never registered (resident / view-rooted / tied): the local
            // replica is always current — gathering would be a wasted
            // collective AND flip the state so a bracket-end release could
            // free a replica that must stay live.
            return;
        }
        let guard = ZERO_CTX.lock().unwrap();
        let rank = guard.as_ref().map(|c| c.rank).unwrap_or(0);
        (
            e.owner,
            e.owner as usize == rank,
            e.elem
                .as_ref()
                .map(|s| (s.slice.ptr() as i64, s.shard, s.elem_bytes, s.sr_param_idx.is_some())),
        )
    };
    // Materialize the full buffer. Tensor-granular: non-owners only (the
    // owner's full replica is live). Item 11 elementwise: EVERY rank — the
    // at-rest state holds nothing but the slice.
    let need_full = elem_info.is_some() || !is_owner;
    if need_full && t.data.is_null() {
        if t.device != 0 {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::inner::ensure_context();
                t.data = crate::cuda::inner::alloc_managed(t.data_byte_size());
            }
        } else if elem_info.is_some() {
            // Host transient for the elementwise unit paths; released (and
            // exactly-sized freed) by zero3_release. The tensor-granular host
            // arm keeps its pre-existing behavior (GPU-only in production).
            t.data =
                crate::memory::checked_alloc_zeroed(t.data_byte_size()) as *mut std::ffi::c_void;
        }
    }
    let rc = {
        let guard = ZERO_CTX.lock().unwrap();
        let Some(ctx) = guard.as_ref() else {
            // A gather after ZeRO teardown would hand back UNINITIALIZED
            // device bytes on non-owners with no error — refuse loudly
            // (deferral-must-refuse; review finding 5).
            eprintln!("[zero3] FATAL: gather with no ZeRO context (post-destroy?)");
            std::process::abort();
        };
        // The backend checks apply exactly where a collective will run
        // (ws>1) — a single-rank gather is a local copy (elementwise) or a
        // no-op (tensor-granular) and must not add new refusals.
        if t.device != 0 && ctx.world_size > 1 && !ctx.cuda_aware {
            eprintln!(
                "[zero3] FATAL: GPU-resident ZeRO-3 needs a CUDA-aware \
                 collective backend (sim-gpu or nccl)"
            );
            std::process::abort();
        }
        if t.device != 0 && t.dtype != 1 && ctx.world_size > 1 {
            eprintln!(
                "[zero3] FATAL: GPU param dtype {} unsupported (f32 only)",
                t.dtype
            );
            std::process::abort();
        }
        let dtype_id = if t.device != 0 {
            crate::tensor_parallel::collective::DTYPE_F32
        } else {
            t.dtype as crate::tensor_parallel::collective::DtypeId
        };
        if let Some((slice_ptr, shard, elem_bytes, sr)) = elem_info {
            // Elementwise: reconstruct the full tensor from every rank's
            // slice. all_gather's recv layout is rank-ordered concatenation
            // (pinned by the sim impl), which is exactly the slicing rule.
            //
            // Item 16×11: a bf16-authoritative slice widens to f32 FIRST —
            // the full tensor is always an f32 working transient and the
            // collective stays DTYPE_F32 (reduce_scatter and the backends
            // never see a 2-byte dtype). The widen is the same D2D cast as
            // the mirror backend's upload, so gather(slices) equals the
            // single-rank working view bit-for-bit.
            if ctx.world_size <= 1 {
                unsafe {
                    if t.device != 0 {
                        #[cfg(feature = "cuda")]
                        if sr {
                            crate::cuda::gpu_cast_raw_bf16_to_f32(
                                slice_ptr as u64,
                                t.data as u64,
                                shard,
                            );
                        } else {
                            crate::cuda::inner::memcpy_dtod(
                                t.data,
                                slice_ptr as *const std::ffi::c_void,
                                shard * elem_bytes,
                            );
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(
                            slice_ptr as *const u8,
                            t.data as *mut u8,
                            shard * elem_bytes,
                        );
                    }
                }
                0
            } else {
                let backend = ctx
                    .backend
                    .as_ref()
                    .expect("world_size > 1 implies a backend");
                // A bf16 slice widens into its CACHED f32 send buffer (the
                // collective never sees a 2-byte dtype); the buffer lives on
                // the entry and is reused every window, like `grad`. The
                // send pointer is read back out of the table because the
                // buffer may have just been created there.
                let send_ptr = if sr {
                    #[cfg(feature = "cuda")]
                    {
                        let mut tbl = ZERO3_TABLE.lock().unwrap();
                        let s = tbl
                            .as_mut()
                            .and_then(|m| m.get_mut(&tensor_ptr))
                            .and_then(|e| e.elem.as_mut())
                            .expect("elementwise entry verified above");
                        if s.sr_send.is_none() {
                            s.sr_send = Some(ShardBuf::alloc(shard * elem_bytes, t.device));
                        }
                        let buf = s.sr_send.as_ref().expect("just created").ptr();
                        crate::cuda::gpu_cast_raw_bf16_to_f32(
                            slice_ptr as u64,
                            buf as u64,
                            shard,
                        );
                        buf as *const std::ffi::c_void
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        eprintln!(
                            "[zero3] FATAL: bf16-sr elementwise gather in a \
                             non-cuda build"
                        );
                        std::process::abort();
                    }
                } else {
                    slice_ptr as *const std::ffi::c_void
                };
                // NOTE: the send buffer stays alive past this call by
                // construction (it is owned by the table, not this scope) —
                // which is also what a future stream-async all_gather will
                // need; a scope-local staging buffer freed on return would
                // be recycled while the collective still read it.
                let rc = backend.all_gather(
                    send_ptr,
                    t.data,
                    shard,
                    dtype_id,
                    std::ptr::null_mut(),
                );
                if rc == 0 {
                    ZERO_ALL_GATHER_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                rc
            }
        } else if ctx.world_size <= 1 {
            0
        } else {
            let backend = ctx
                .backend
                .as_ref()
                .expect("world_size > 1 implies a backend");
            backend.broadcast(
                t.data,
                t.len as usize,
                dtype_id,
                owner,
                std::ptr::null_mut(),
            )
        }
    };
    if rc != 0 {
        eprintln!("[zero3] FATAL: gather collective failed rc={rc}");
        std::process::abort();
    }
    let mut tbl = ZERO3_TABLE.lock().unwrap();
    if let Some(e) = tbl.as_mut().and_then(|m| m.get_mut(&tensor_ptr)) {
        e.state = ParameterResidency::GatheredTemporary;
    }
    ZERO3_GATHERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Release (weight-stream `evict` redirect): the owner keeps its
/// authoritative data (writeback is meaningless — there is no mirror);
/// non-owners free their gathered copy. Safe on an already-evicted param.
pub(crate) fn zero3_release(tensor_ptr: i64) {
    let t = crate::tensor::NslTensor::from_ptr(tensor_ptr);
    let (is_owner, elem_info) = {
        let tbl = ZERO3_TABLE.lock().unwrap();
        let Some(e) = tbl.as_ref().and_then(|m| m.get(&tensor_ptr)) else {
            eprintln!("[zero3] FATAL: release of untracked tensor {tensor_ptr}");
            std::process::abort();
        };
        if e.state == ParameterResidency::Replicated {
            // Resident replica — must never be freed by a bracket end.
            return;
        }
        let guard = ZERO_CTX.lock().unwrap();
        let rank = guard.as_ref().map(|c| c.rank).unwrap_or(0);
        (
            e.owner as usize == rank,
            e.elem
                .as_ref()
                .map(|s| (s.slice.ptr() as i64, s.shard, s.elem_bytes, rank, s.sr_param_idx.is_some())),
        )
    };
    let is_elem = elem_info.is_some();
    // Item 11 elementwise: the slice is the authoritative storage, so the
    // full transient is dropped on EVERY rank — the owner arm below is
    // tensor-granular-only.
    if is_owner && !is_elem {
        // Data stays (authoritative replica); the state transition mirrors
        // the non-owner's so the NEXT gather broadcasts symmetrically.
        let mut tbl = ZERO3_TABLE.lock().unwrap();
        if let Some(e) = tbl.as_mut().and_then(|m| m.get_mut(&tensor_ptr)) {
            if e.state == ParameterResidency::GatheredTemporary {
                e.state = ParameterResidency::ShardedResident;
            }
        }
        ZERO3_RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return;
    }
    if !t.data.is_null() {
        // Elementwise: persist this rank's region into the slice BEFORE the
        // free. On the normal schedule this is a no-op copy (the step
        // already persisted), but a θ-MUTATING callback bracket writes the
        // gathered full tensor after the step — tensor-granular keeps that
        // write on the owner's replica, and without this copy elementwise
        // would silently drop it on every rank (review finding, 2026-08-06).
        //
        // Item 16×11: NOT for a bf16-authoritative slice. Under bf16-sr the
        // fused SR step is the ONLY sanctioned θ writer — plain SR already
        // drops callback writes to the f32 working view (evict never writes
        // back), and the composed path must match plain SR, not plain
        // elementwise. A narrow here would also round the freshly-stepped
        // slice through f32 values the step never produced.
        if let Some((slice_ptr, shard, elem_bytes, rank, sr)) = elem_info {
            if !sr {
                let off = rank * shard * elem_bytes;
                unsafe {
                    if t.device != 0 {
                        #[cfg(feature = "cuda")]
                        {
                            crate::cuda::inner::ensure_context();
                            crate::cuda::inner::memcpy_dtod(
                                slice_ptr as *mut std::ffi::c_void,
                                (t.data as *const u8).add(off) as *const std::ffi::c_void,
                                shard * elem_bytes,
                            );
                        }
                    } else {
                        std::ptr::copy_nonoverlapping(
                            (t.data as *const u8).add(off),
                            slice_ptr as *mut u8,
                            shard * elem_bytes,
                        );
                    }
                }
            }
        }
        if t.device != 0 {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::inner::ensure_context();
                crate::cuda::inner::free_managed(t.data);
            }
        } else if is_elem {
            // Exact pair of the gather's host checked_alloc_zeroed.
            unsafe {
                crate::memory::checked_free(t.data as *mut u8, t.data_byte_size());
            }
        }
        t.data = std::ptr::null_mut();
    }
    let mut tbl = ZERO3_TABLE.lock().unwrap();
    if let Some(e) = tbl.as_mut().and_then(|m| m.get_mut(&tensor_ptr)) {
        e.state = ParameterResidency::Evicted;
    }
    ZERO3_RELEASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Item 14: all-reduce ONE layer-group gradient slot across ranks, at the
/// point where source AD has just completed that layer's backward. Every
/// rank calls this in the identical compile-time order (the spin-barrier
/// invariant). The slot is the layer's m_partial accumulator tensor.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_reduce_grad_slot(list_ptr: i64, idx: i64) -> i64 {
    let list = unsafe { &*(list_ptr as *const crate::list::NslList) };
    if idx < 0 || idx >= list.len {
        return -1;
    }
    let raw = unsafe { *list.data.add(idx as usize) };
    if raw == 0 {
        return 0; // null slot (already consumed) — nothing to reduce
    }
    let t = unsafe { &*(raw as *const crate::tensor::NslTensor) };

    // Item 11: an elementwise param's window gradient reduce_scatters into
    // its persistent grad slice — each rank receives only the averaged
    // elements it will step. The full slot is left with its LOCAL sum
    // (never read again; codegen frees it right after the group update).
    // Lock order: ZERO3_TABLE strictly before ZERO_CTX, matching gather.
    let elem_info = {
        let tbl = ZERO3_TABLE.lock().unwrap();
        tbl.as_ref().and_then(|m| {
            m.iter().find_map(|(k, e)| {
                if e.idx == idx as usize {
                    e.elem
                        .as_ref()
                        .map(|s| (*k, s.shard, s.elem_bytes, s.grad.is_some()))
                } else {
                    None
                }
            })
        })
    };
    if let Some((key, shard, elem_bytes, grad_allocated)) = elem_info {
        if shard * elem_bytes == 0 {
            return -1;
        }
        if !grad_allocated {
            let buf = ShardBuf::alloc(shard * elem_bytes, t.device);
            let mut tbl = ZERO3_TABLE.lock().unwrap();
            if let Some(s) = tbl
                .as_mut()
                .and_then(|m| m.get_mut(&key))
                .and_then(|e| e.elem.as_mut())
            {
                s.grad = Some(buf);
            } else {
                // Entry vanished between the lookup and the store (teardown
                // race the single-threaded schedule never produces) — free
                // rather than leak: ShardBuf::Dev has no Drop.
                buf.free();
                return -1;
            }
        }
        let grad_ptr = {
            let tbl = ZERO3_TABLE.lock().unwrap();
            tbl.as_ref()
                .and_then(|m| m.get(&key))
                .and_then(|e| e.elem.as_ref())
                .and_then(|s| s.grad.as_ref().map(|g| g.ptr() as i64))
                .unwrap_or(0)
        };
        if grad_ptr == 0 {
            return -1;
        }
        let (ws, rc) = {
            let guard = ZERO_CTX.lock().unwrap();
            let Some(ctx) = guard.as_ref() else {
                return -1;
            };
            if (t.len.max(0) as usize) != shard * ctx.world_size.max(1) {
                eprintln!(
                    "[zero3] FATAL: elementwise grad slot {idx} has {} elements \
                     but the shard layout says {}x{} — plan/runtime drift",
                    t.len,
                    shard,
                    ctx.world_size
                );
                std::process::abort();
            }
            if ctx.world_size <= 1 {
                // Degenerate single rank: the "slice" is the whole gradient,
                // unaveraged — copy it over (baseline semantics).
                unsafe {
                    if t.device != 0 {
                        #[cfg(feature = "cuda")]
                        crate::cuda::inner::memcpy_dtod(
                            grad_ptr as *mut std::ffi::c_void,
                            t.data as *const std::ffi::c_void,
                            shard * elem_bytes,
                        );
                    } else {
                        std::ptr::copy_nonoverlapping(
                            t.data as *const u8,
                            grad_ptr as *mut u8,
                            shard * elem_bytes,
                        );
                    }
                }
                (1usize, 0i32)
            } else {
                let backend = ctx
                    .backend
                    .as_ref()
                    .expect("world_size > 1 implies a backend");
                if t.device != 0 && !ctx.cuda_aware {
                    eprintln!(
                        "[zero3] FATAL: GPU-resident gradient reduce_scatter \
                         needs a CUDA-aware collective backend (sim-gpu or nccl)"
                    );
                    std::process::abort();
                }
                if t.device != 0 && t.dtype != 1 {
                    eprintln!("[zero3] FATAL: GPU grad dtype {} unsupported", t.dtype);
                    std::process::abort();
                }
                let rc = backend.reduce_scatter_sum(
                    t.data as *const std::ffi::c_void,
                    grad_ptr as *mut std::ffi::c_void,
                    t.len as usize,
                    if t.device != 0 {
                        crate::tensor_parallel::collective::DTYPE_F32
                    } else {
                        t.dtype as crate::tensor_parallel::collective::DtypeId
                    },
                    std::ptr::null_mut(),
                );
                if rc == 0 {
                    ZERO_REDUCE_SCATTER_COUNT
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                (ctx.world_size, rc)
            }
        };
        if rc != 0 {
            eprintln!("[zero3] gradient reduce_scatter failed rc={rc} for slot {idx}");
            return -1;
        }
        // The same sum-then-divide averaging as the all-reduce arm, applied
        // to the slice with the identical scalar conversion (f64 factor,
        // per-dtype rounding at the multiply — see
        // nsl_tensor_mul_scalar_inplace).
        if ws > 1 {
            let inv_ws = 1.0 / ws as f64;
            if t.device != 0 {
                #[cfg(feature = "cuda")]
                crate::cuda::gpu_scale_raw_f32(
                    grad_ptr as *mut std::ffi::c_void,
                    shard,
                    inv_ws as f32,
                );
            } else if t.dtype == 0 {
                let p = grad_ptr as *mut f64;
                for i in 0..shard {
                    unsafe { *p.add(i) *= inv_ws };
                }
            } else {
                let p = grad_ptr as *mut f32;
                let f = inv_ws as f32;
                for i in 0..shard {
                    unsafe { *p.add(i) *= f };
                }
            }
        }
        let mut tbl = ZERO3_TABLE.lock().unwrap();
        if let Some(s) = tbl
            .as_mut()
            .and_then(|m| m.get_mut(&key))
            .and_then(|e| e.elem.as_mut())
        {
            s.grad_ready = true;
        }
        return 0;
    }

    let guard = ZERO_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };
    if ctx.world_size <= 1 {
        return 0;
    }
    let backend = ctx
        .backend
        .as_ref()
        .expect("world_size > 1 implies a backend");
    if t.device != 0 && !ctx.cuda_aware {
        eprintln!(
            "[zero3] FATAL: GPU-resident gradient all-reduce needs a \
             CUDA-aware collective backend (sim-gpu or nccl)"
        );
        std::process::abort();
    }
    if t.device != 0 && t.dtype != 1 {
        eprintln!("[zero3] FATAL: GPU grad dtype {} unsupported", t.dtype);
        std::process::abort();
    }
    let rc = backend.all_reduce_sum(
        t.data as *const std::ffi::c_void,
        t.data,
        t.len as usize,
        if t.device != 0 {
            crate::tensor_parallel::collective::DTYPE_F32
        } else {
            t.dtype as crate::tensor_parallel::collective::DtypeId
        },
        std::ptr::null_mut(),
    );
    if rc != 0 {
        eprintln!("[zero3] gradient all-reduce failed rc={rc} for slot {idx}");
        return -1;
    }
    // Gradient AVERAGING — the same sum-then-divide convention as
    // nsl_zero_reduce_grads, so a rank-blind loader stays bit-exact with
    // the single-rank baseline ((g+g)/2 == g in IEEE for pow-2 world
    // sizes; identical semantics otherwise).
    let inv_ws = 1.0 / ctx.world_size as f64;
    drop(guard);
    crate::tensor::nsl_tensor_mul_scalar_inplace(raw, inv_ws);
    0
}

/// Item 11: the elementwise fused-AdamW step — EVERY rank updates its own
/// `[rank*shard, (rank+1)*shard)` region using the reduce_scattered gradient
/// slice, with the EXACT per-element math of `nsl_fase_fused_adamw_step`
/// (GPU: the same `FASE_FUSED_ADAMW_STEP_F32_PTX` kernel launched on the
/// region; CPU: the same loops). AdamW has no cross-element terms, so the
/// result is bit-identical to the baseline full-tensor step restricted to
/// this region. The updated theta region is then persisted into the slice
/// (release frees the full transient right after the group update).
///
/// Item C: `m`/`v` are THIS RANK's SLICE — exactly `shard` elements, from
/// `nsl_zero3_alloc_elem_moment` — so they are indexed at `i` while θ (the
/// gathered full transient) is indexed at `off + i`. The region result is
/// bit-identical to the old replicated form, which touched `m[off+i]` of a
/// full buffer whose other elements were dead weight on every rank.
///
/// The scalar argument order mirrors `nsl_fase_fused_adamw_step` exactly.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_zero3_elem_adamw_step(
    theta_ptr: i64,
    m_ptr: i64,
    v_ptr: i64,
    idx: i64,
    lr: f64,
    beta1: f64,
    one_minus_beta1: f64,
    beta2: f64,
    one_minus_beta2: f64,
    eps: f64,
    wd: f64,
    bc1_inv: f64,
    bc2_inv: f64,
) -> i64 {
    let th = unsafe { &*(theta_ptr as *const crate::tensor::NslTensor) };
    let m = unsafe { &*(m_ptr as *const crate::tensor::NslTensor) };
    let v = unsafe { &*(v_ptr as *const crate::tensor::NslTensor) };

    let (shard, elem_bytes, grad_ptr) = {
        let tbl = ZERO3_TABLE.lock().unwrap();
        let Some(e) = tbl.as_ref().and_then(|map| map.get(&theta_ptr)) else {
            eprintln!(
                "[zero3] FATAL: elementwise step of untracked tensor \
                 {theta_ptr} (param {idx})"
            );
            std::process::abort();
        };
        if e.idx != idx as usize {
            eprintln!(
                "[zero3] FATAL: elementwise step param-index mismatch \
                 (tensor says {}, codegen says {idx})",
                e.idx
            );
            std::process::abort();
        }
        let Some(s) = e.elem.as_ref() else {
            eprintln!(
                "[zero3] FATAL: elementwise step of param {idx}, which was \
                 never carved — the plan and the emitted step dispatch have \
                 drifted"
            );
            std::process::abort();
        };
        if !s.grad_ready {
            // A step without a fresh reduce would apply a stale or
            // uninitialized gradient slice — a silent-wrong-numerics class.
            eprintln!(
                "[zero3] FATAL: elementwise step of param {idx} before its \
                 window gradient was reduce_scattered"
            );
            std::process::abort();
        }
        if s.sr_param_idx.is_some() {
            // Item 16×11 belt: this entry's f32 kernel + persist would
            // update θ WITHOUT stochastic rounding and then overrun the
            // 2-byte slice with a 4-byte copy. The composed dispatch emits
            // `nsl_zero3_elem_sr_adamw_step` for SR-carved params.
            eprintln!(
                "[zero3] FATAL: bf16-sr-carved param {idx} routed to the \
                 plain elementwise step — the emitted dispatch and the carve \
                 have drifted"
            );
            std::process::abort();
        }
        (
            s.shard,
            s.elem_bytes,
            s.grad.as_ref().map(|g| g.ptr() as i64).unwrap_or(0),
        )
    };
    if th.data.is_null() || grad_ptr == 0 {
        eprintln!(
            "[zero3] FATAL: elementwise step of param {idx} without a \
             materialized full tensor (gather must precede the group update)"
        );
        std::process::abort();
    }
    // Item C: m/v are SLICE-sized. A full-sized buffer here means the
    // deferred fill fell back to `zeros_like` (or a caller kept the old
    // replicated allocation) — refuse rather than silently index the first
    // `shard` elements of a replicated buffer, which is arithmetically fine
    // on rank 0 and WRONG on every other rank.
    assert!(
        m.len.max(0) as usize == shard && v.len.max(0) as usize == shard,
        "zero3 elem step: m/v must be this rank's slice, not a full replica \
         (shard={}, theta={}, m={}, v={})",
        shard,
        th.len,
        m.len,
        v.len
    );
    assert!(
        m.device == th.device && v.device == th.device,
        "zero3 elem step: device mismatch"
    );
    let rank = {
        let guard = ZERO_CTX.lock().unwrap();
        guard.as_ref().map(|c| c.rank).unwrap_or(0)
    };
    // Bound proof for the θ region this rank writes: the gathered transient
    // must actually cover [rank*shard, (rank+1)*shard). Saturating on BOTH
    // the test and the message — a debug build would otherwise panic with
    // "attempt to multiply with overflow" while formatting, replacing the
    // drift diagnostic with a arithmetic one.
    let region_lo = rank.saturating_mul(shard);
    let region_hi = region_lo.saturating_add(shard);
    assert!(
        region_hi <= th.len.max(0) as usize,
        "zero3 elem step: region [{region_lo}, {region_hi}) escapes theta \
         (len={})",
        th.len
    );
    let off_bytes = rank * shard * elem_bytes;
    let has_wd = wd != 0.0;

    if th.device > 0 {
        #[cfg(feature = "cuda")]
        {
            assert!(
                th.dtype == 1 && m.dtype == 1 && v.dtype == 1,
                "zero3 elem step: GPU path requires f32 tensors"
            );
            // Identical f64→f32 conversions to nsl_fase_fused_adamw_step's
            // GPU arm — this is what makes the region bit-equal.
            crate::cuda::gpu_fase_fused_adamw_step_raw(
                (th.data as u64) + off_bytes as u64,
                // Item C: slice-sized moments start at their own base.
                m.data as u64,
                v.data as u64,
                grad_ptr as u64,
                shard,
                beta1 as f32,
                one_minus_beta1 as f32,
                beta2 as f32,
                one_minus_beta2 as f32,
                eps as f32,
                (-lr) as f32,
                ((-lr) * wd) as f32,
                bc1_inv as f32,
                bc2_inv as f32,
                has_wd,
            );
            crate::cuda::inner::ensure_context();
            crate::cuda::inner::memcpy_dtod(
                {
                    let tbl = ZERO3_TABLE.lock().unwrap();
                    tbl.as_ref()
                        .and_then(|map| map.get(&theta_ptr))
                        .and_then(|e| e.elem.as_ref())
                        .map(|s| s.slice.ptr())
                        .expect("entry verified above")
                },
                unsafe { (th.data as *const u8).add(off_bytes) } as *const std::ffi::c_void,
                shard * elem_bytes,
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("[zero3] FATAL: GPU elementwise step in a non-cuda build");
            std::process::abort();
        }
    } else if th.dtype == 0 && m.dtype == 0 && v.dtype == 0 {
        // CPU f64 — the same op order as nsl_fase_fused_adamw_step's f64 arm.
        let neg_lr = -lr;
        let neg_lr_wd = (-lr) * wd;
        let off = rank * shard;
        let (td, md, vd) = (
            th.data as *mut f64,
            m.data as *mut f64,
            v.data as *mut f64,
        );
        let gd = grad_ptr as *const f64;
        for i in 0..shard {
            unsafe {
                let g_i = *gd.add(i);
                // Item C: m/v are slice-local (`i`); θ is the gathered full
                // transient (`off + i`).
                let t1 = *md.add(i) * beta1;
                let t2 = g_i * one_minus_beta1;
                let m_new = t1 + t2;
                *md.add(i) = m_new;
                let sq = g_i * g_i;
                let ssq = sq * one_minus_beta2;
                let vdec = *vd.add(i) * beta2;
                let v_new = vdec + ssq;
                *vd.add(i) = v_new;
                let mh = m_new * bc1_inv;
                let vh = v_new * bc2_inv;
                let tden = vh.sqrt() + eps;
                let u = mh / tden;
                let mut adj = u * neg_lr;
                if has_wd {
                    let wdt = *td.add(off + i) * neg_lr_wd;
                    adj += wdt;
                }
                *td.add(off + i) += adj;
            }
        }
        persist_host_slice(theta_ptr, th, off_bytes, shard * elem_bytes);
    } else if th.dtype == 1 && m.dtype == 1 && v.dtype == 1 {
        // CPU f32 — mirrors the fused step's f32 arm (per-scalar `as f32`).
        let b1 = beta1 as f32;
        let omb1 = one_minus_beta1 as f32;
        let b2 = beta2 as f32;
        let omb2 = one_minus_beta2 as f32;
        let epsf = eps as f32;
        let bc1 = bc1_inv as f32;
        let bc2 = bc2_inv as f32;
        let neg_lr = (-lr) as f32;
        let neg_lr_wd = ((-lr) * wd) as f32;
        let off = rank * shard;
        let (td, md, vd) = (
            th.data as *mut f32,
            m.data as *mut f32,
            v.data as *mut f32,
        );
        let gd = grad_ptr as *const f32;
        for i in 0..shard {
            unsafe {
                let g_i = *gd.add(i);
                // Item C: m/v slice-local; θ at `off + i` (see the f64 arm).
                let t1 = *md.add(i) * b1;
                let t2 = g_i * omb1;
                let m_new = t1 + t2;
                *md.add(i) = m_new;
                let sq = g_i * g_i;
                let ssq = sq * omb2;
                let vdec = *vd.add(i) * b2;
                let v_new = vdec + ssq;
                *vd.add(i) = v_new;
                let mh = m_new * bc1;
                let vh = v_new * bc2;
                let tden = vh.sqrt() + epsf;
                let u = mh / tden;
                let mut adj = u * neg_lr;
                if has_wd {
                    let wdt = *td.add(off + i) * neg_lr_wd;
                    adj += wdt;
                }
                *td.add(off + i) += adj;
            }
        }
        persist_host_slice(theta_ptr, th, off_bytes, shard * elem_bytes);
    } else {
        eprintln!(
            "[zero3] FATAL: elementwise step dtype combination unsupported \
             (theta={}, m={}, v={})",
            th.dtype, m.dtype, v.dtype
        );
        std::process::abort();
    }

    let mut tbl = ZERO3_TABLE.lock().unwrap();
    if let Some(s) = tbl
        .as_mut()
        .and_then(|map| map.get_mut(&theta_ptr))
        .and_then(|e| e.elem.as_mut())
    {
        s.grad_ready = false;
    }
    ZERO3_ELEM_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}

/// Item 16×11: the composed elementwise step under `--param-dtype bf16-sr`
/// — EVERY rank updates its bf16-AUTHORITATIVE slice with the same fused SR
/// kernel the mirror backend launches, against its OWN slice-sized f32
/// moments and the reduce_scattered gradient slice. The SR
/// counter base carries the rank's flat element offset on top of the
/// param's counter block, so every element draws the dither the single-rank
/// mirror baseline draws at the same global index — which is what makes a
/// multi-rank composed run bit-identical to plain SR.
///
/// Scalar order mirrors `nsl_zero3_elem_adamw_step`, plus the trailing
/// `step` the SR stream is keyed on (the `nsl_sr_bf16_step_adamw` shape).
/// After the kernel, the rank's region of the still-live f32 working
/// transient is REFRESHED by widening (post-step readers — callbacks,
/// checkpoint save — see updated θ). Never the reverse direction: the
/// release-persist is skipped for SR slices, because the SR step is the
/// only sanctioned θ writer (plain SR's evict-never-writes-back contract).
///
/// Item C: m/v are SLICE-sized and slice-LOCAL, exactly as in the plain
/// twin — the deferred fill routes on `is_elementwise()` alone, so an
/// SR-carved param gets the same `nsl_zero3_alloc_elem_moment` buffer.
/// `off_bytes` therefore rides ONLY the two full-tensor traffics that
/// remain: the θ coherence widen below, and (via `rank * shard`) the SR
/// counter base. Re-adding it to the moment pointers would read a
/// rank-1 shard's worth of bytes past the end of a `shard`-sized
/// allocation.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_zero3_elem_sr_adamw_step(
    theta_ptr: i64,
    m_ptr: i64,
    v_ptr: i64,
    idx: i64,
    lr: f64,
    beta1: f64,
    one_minus_beta1: f64,
    beta2: f64,
    one_minus_beta2: f64,
    eps: f64,
    wd: f64,
    bc1_inv: f64,
    bc2_inv: f64,
    step: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let th = unsafe { &*(theta_ptr as *const crate::tensor::NslTensor) };
        let m = unsafe { &*(m_ptr as *const crate::tensor::NslTensor) };
        let v = unsafe { &*(v_ptr as *const crate::tensor::NslTensor) };

        let (shard, elem_bytes, grad_ptr, slice_ptr, sr_idx) = {
            let tbl = ZERO3_TABLE.lock().unwrap();
            let Some(e) = tbl.as_ref().and_then(|map| map.get(&theta_ptr)) else {
                eprintln!(
                    "[zero3] FATAL: bf16-sr elementwise step of untracked \
                     tensor {theta_ptr} (param {idx})"
                );
                std::process::abort();
            };
            if e.idx != idx as usize {
                eprintln!(
                    "[zero3] FATAL: bf16-sr elementwise step param-index \
                     mismatch (tensor says {}, codegen says {idx})",
                    e.idx
                );
                std::process::abort();
            }
            let Some(s) = e.elem.as_ref() else {
                eprintln!(
                    "[zero3] FATAL: bf16-sr elementwise step of param {idx}, \
                     which was never carved — the plan and the emitted step \
                     dispatch have drifted"
                );
                std::process::abort();
            };
            let Some(sr_idx) = s.sr_param_idx else {
                // The inverse of the plain step's belt: a plain-carved
                // param stepped here would draw a dither stream nothing
                // noted, and its 4-byte slice would be read as bf16.
                eprintln!(
                    "[zero3] FATAL: param {idx} was carved WITHOUT bf16-sr \
                     but routed to the SR elementwise step — the emitted \
                     dispatch and the carve have drifted"
                );
                std::process::abort();
            };
            if !s.grad_ready {
                eprintln!(
                    "[zero3] FATAL: bf16-sr elementwise step of param {idx} \
                     before its window gradient was reduce_scattered"
                );
                std::process::abort();
            }
            (
                s.shard,
                s.elem_bytes,
                s.grad.as_ref().map(|g| g.ptr() as i64).unwrap_or(0),
                s.slice.ptr() as u64,
                sr_idx,
            )
        };
        if th.data.is_null() || grad_ptr == 0 {
            eprintln!(
                "[zero3] FATAL: bf16-sr elementwise step of param {idx} \
                 without a materialized full tensor (gather must precede \
                 the group update)"
            );
            std::process::abort();
        }
        // Item C: m/v are SLICE-sized, same as the plain twin. A full-sized
        // buffer here means the deferred fill fell back to `zeros_like` (or
        // a caller kept the old replicated allocation) — refuse rather than
        // silently index the first `shard` elements of a replicated buffer,
        // which is arithmetically fine on rank 0 and WRONG on every other
        // rank.
        assert!(
            m.len.max(0) as usize == shard && v.len.max(0) as usize == shard,
            "zero3 sr elem step: m/v must be this rank's slice, not a full \
             replica (shard={}, theta={}, m={}, v={})",
            shard,
            th.len,
            m.len,
            v.len
        );
        assert!(
            th.device > 0 && m.device == th.device && v.device == th.device,
            "zero3 sr elem step: all tensors must be GPU-resident"
        );
        assert!(
            th.dtype == 1 && m.dtype == 1 && v.dtype == 1,
            "zero3 sr elem step: th/m/v must be f32 (got {}/{}/{})",
            th.dtype,
            m.dtype,
            v.dtype
        );
        // The mirror path asserts this too, and here it is load-bearing for
        // the POINTER ARITHMETIC below: `m.data + rank*shard*4` addresses
        // logical elements only for a contiguous tensor. A strided m/v would
        // pass every check above and silently update the wrong positions.
        assert!(
            th.is_contiguous() && m.is_contiguous() && v.is_contiguous(),
            "zero3 sr elem step: th/m/v must be contiguous (the region offset \
             is flat element arithmetic)"
        );
        let rank = {
            let guard = ZERO_CTX.lock().unwrap();
            guard.as_ref().map(|c| c.rank).unwrap_or(0)
        };
        // Same bound proof the plain twin carries, for the same reason: this
        // is now the ONLY consumer of `off_bytes`, and it feeds a raw
        // `th.data.add(off_bytes)` write in the coherence widen below.
        // Saturating on BOTH the test and the message — a debug build would
        // otherwise panic with "attempt to multiply with overflow" while
        // formatting, replacing the drift diagnostic with an arithmetic one.
        let region_lo = rank.saturating_mul(shard);
        let region_hi = region_lo.saturating_add(shard);
        assert!(
            region_hi <= th.len.max(0) as usize,
            "zero3 sr elem step: region [{region_lo}, {region_hi}) escapes \
             theta (len={})",
            th.len
        );
        let off_bytes = rank * shard * elem_bytes;
        let has_wd = wd != 0.0;

        let seed = crate::deterministic_ops::get_rng_seed();
        let key = crate::sr_bf16::sr_step_key(seed, step as u64);
        // The rank's flat element offset rides inside the param's counter
        // block, so element j of this slice draws the dither the unsharded
        // baseline draws at global element rank*shard + j.
        let ctr_base = crate::sr_bf16::sr_param_ctr_base(sr_idx, (rank * shard) as u64);

        // NSL_SR_HIST rides the composed path too — a new dispatch path
        // must not shrink the instrument's surface (the #468 batched-arm
        // lesson). Each rank samples the head of ITS slice, i.e. global
        // elements [rank*shard, rank*shard + hist_n) — NOT the same window
        // the mirror paths sample (their first min(len, 16384) of the FULL
        // param), and each rank reports its own accumulator. Stall RATES
        // stay comparable; absolute sample populations and per-rank
        // windows do not — read a composed histogram against the baseline
        // as per-rank evidence, not as one pooled distribution.
        let hist_n = crate::sr_bf16::sr_hist_sample_len(shard);
        let mut hist_before: Vec<u16> = vec![0; hist_n];
        if hist_n > 0 {
            crate::cuda::inner::ensure_context();
            crate::cuda::inner::memcpy_dtoh(
                hist_before.as_mut_ptr() as *mut std::ffi::c_void,
                slice_ptr as *const std::ffi::c_void,
                hist_n * 2,
            );
        }
        crate::cuda::gpu_fase_fused_adamw_step_bf16sr_raw(
            slice_ptr,
            // Item C: slice-sized moments start at their own base (the
            // plain twin's convention). `off_bytes` stays on θ only.
            m.data as u64,
            v.data as u64,
            grad_ptr as u64,
            shard,
            beta1 as f32,
            one_minus_beta1 as f32,
            beta2 as f32,
            one_minus_beta2 as f32,
            eps as f32,
            (-lr) as f32,
            ((-lr) * wd) as f32,
            bc1_inv as f32,
            bc2_inv as f32,
            has_wd,
            key,
            ctr_base,
        );
        if hist_n > 0 {
            let mut hist_after: Vec<u16> = vec![0; hist_n];
            crate::cuda::inner::memcpy_dtoh(
                hist_after.as_mut_ptr() as *mut std::ffi::c_void,
                slice_ptr as *const std::ffi::c_void,
                hist_n * 2,
            );
            crate::sr_bf16::sr_hist_record_step(&hist_before, &hist_after);
        }
        // Coherence refresh: the rank's region of the live f32 working
        // transient must equal widen(slice) after the step — the same
        // direction and cast as the mirror backend's refresh.
        unsafe {
            crate::cuda::gpu_cast_raw_bf16_to_f32(
                slice_ptr,
                (th.data as *const u8).add(off_bytes) as u64,
                shard,
            );
        }

        let mut tbl = ZERO3_TABLE.lock().unwrap();
        if let Some(s) = tbl
            .as_mut()
            .and_then(|map| map.get_mut(&theta_ptr))
            .and_then(|e| e.elem.as_mut())
        {
            s.grad_ready = false;
        }
        ZERO3_ELEM_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ZERO3_SR_ELEM_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Also count it on the SR backend's own surface: tooling that proves
        // "stochastic rounding ran" from nsl_sr_bf16_step_count() or the
        // [sr-bf16] teardown line must not read 0 on a composed run.
        crate::sr_bf16::note_external_sr_step();
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            theta_ptr, m_ptr, v_ptr, idx, lr, beta1, one_minus_beta1, beta2,
            one_minus_beta2, eps, wd, bc1_inv, bc2_inv, step,
        );
        eprintln!(
            "[zero3] FATAL: the bf16-sr elementwise step requires the cuda \
             feature (the SR backend is GPU-only)"
        );
        std::process::abort();
    }
}

/// Copy the just-updated host theta region into the persistent slice.
fn persist_host_slice(
    theta_ptr: i64,
    th: &crate::tensor::NslTensor,
    off_bytes: usize,
    bytes: usize,
) {
    let tbl = ZERO3_TABLE.lock().unwrap();
    if let Some(s) = tbl
        .as_ref()
        .and_then(|map| map.get(&theta_ptr))
        .and_then(|e| e.elem.as_ref())
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                (th.data as *const u8).add(off_bytes),
                s.slice.ptr() as *mut u8,
                bytes,
            );
        }
    }
}

/// Gather every REGISTERED (non-Replicated) param — the open bracket for
/// callbacks / model_save / teardown. Symmetric across ranks (identical
/// table iteration order via sorted keys).
pub(crate) fn zero3_gather_all() {
    // Order by PARAM INDEX, not tensor pointer: the iteration order is the
    // cross-rank collective schedule, and heap addresses are per-process
    // (review finding 2 — pointer order only matched by allocation-
    // determinism accident).
    let mut keys: Vec<(usize, i64)> = {
        let guard = ZERO3_TABLE.lock().unwrap();
        guard
            .as_ref()
            .map(|t| {
                t.iter()
                    .filter(|(_, e)| e.state != ParameterResidency::Replicated)
                    .map(|(k, e)| (e.idx, *k))
                    .collect()
            })
            .unwrap_or_default()
    };
    keys.sort_unstable();
    for (_, k) in keys {
        zero3_gather(k);
    }
}

/// Release every gathered temporary — the close bracket (Replicated and
/// owner replicas are untouched by construction).
pub(crate) fn zero3_release_all() {
    let mut keys: Vec<(usize, i64)> = {
        let guard = ZERO3_TABLE.lock().unwrap();
        guard
            .as_ref()
            .map(|t| {
                t.iter()
                    .filter(|(_, e)| e.state == ParameterResidency::GatheredTemporary)
                    .map(|(k, e)| (e.idx, *k))
                    .collect()
            })
            .unwrap_or_default()
    };
    keys.sort_unstable();
    for (_, k) in keys {
        zero3_release(k);
    }
}

/// Whether this pointer is a REGISTERED (sharded) zero3 param.
pub(crate) fn zero3_is_registered(tensor_ptr: i64) -> bool {
    let guard = ZERO3_TABLE.lock().unwrap();
    guard
        .as_ref()
        .and_then(|t| t.get(&tensor_ptr))
        .is_some_and(|e| e.state != ParameterResidency::Replicated)
}

/// Teardown: leave every replica FULL and CURRENT (the model_save /
/// eval-read end state, matching stages 1/2), then deactivate so
/// subsequent code paths see plain tensors.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_zero3_teardown() -> i64 {
    if zero3_active() {
        zero3_gather_all();
        // Anti-vacuity line for the gates: a zero3 run whose schedule never
        // gathered/released anything is a broken schedule, not a pass. The
        // elem_* fields let the elementwise gate prove its machinery ran —
        // and the tensor-granular gate that it did NOT (label-anchored
        // parsers; appended fields are compatible). Format first, write
        // once: both ranks exit through here simultaneously and `eprintln!`
        // issues one write(2) per fragment, TEARING the two ranks' lines
        // into each other (the same failure args.rs's [zero] counter line
        // already guards against — observed on this line the first time a
        // gate parsed a field from it).
        let line = format!(
            "[zero3] teardown: full residency restored (gathers={} releases={} \
             elem_params={} elem_steps={} sr_elem_params={} sr_elem_steps={} \
             elem_moments={})\n",
            ZERO3_GATHERS.load(std::sync::atomic::Ordering::Relaxed),
            ZERO3_RELEASES.load(std::sync::atomic::Ordering::Relaxed),
            ZERO3_ELEM_PARAMS.load(std::sync::atomic::Ordering::Relaxed),
            ZERO3_ELEM_STEPS.load(std::sync::atomic::Ordering::Relaxed),
            ZERO3_SR_ELEM_PARAMS.load(std::sync::atomic::Ordering::Relaxed),
            ZERO3_SR_ELEM_STEPS.load(std::sync::atomic::Ordering::Relaxed),
            ZERO3_ELEM_MOMENTS.load(std::sync::atomic::Ordering::Relaxed),
        );
        use std::io::Write;
        let _ = std::io::stderr().lock().write_all(line.as_bytes());
    }
    ZERO3_ACTIVE.store(false, std::sync::atomic::Ordering::SeqCst);
    // The NEW counters reset per train block so a second block's teardown
    // line reports its own run, not a cumulative sum (an exact-count gate
    // on a multi-block program would otherwise misfire). GATHERS/RELEASES
    // keep their pre-existing cumulative convention untouched.
    ZERO3_ELEM_PARAMS.store(0, std::sync::atomic::Ordering::Relaxed);
    ZERO3_ELEM_STEPS.store(0, std::sync::atomic::Ordering::Relaxed);
    ZERO3_SR_ELEM_PARAMS.store(0, std::sync::atomic::Ordering::Relaxed);
    ZERO3_SR_ELEM_STEPS.store(0, std::sync::atomic::Ordering::Relaxed);
    ZERO3_ELEM_MOMENTS.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut guard = ZERO3_TABLE.lock().unwrap();
    // Item 11: device shard buffers are freed explicitly — entries live in a
    // static, so ShardBuf cannot carry a cfg(cuda) Drop.
    if let Some(map) = guard.take() {
        for (_, e) in map {
            if let Some(s) = e.elem {
                s.slice.free();
                if let Some(g) = s.grad {
                    g.free();
                }
                if let Some(b) = s.sr_send {
                    b.free();
                }
            }
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

    // ── Item 11: elementwise sharding state machine (ws=1 degenerate) ────

    /// Owned-looking CPU tensor over a caller-owned Vec. The carve's CPU arm
    /// nulls `data` without freeing, so claiming `owns_data=1` over a
    /// Vec-backed buffer is safe here (and required — the register admission
    /// aborts on borrowed tensors).
    fn owned_cpu_tensor(data: *mut c_void, shape: *mut i64, strides: *mut i64, len: i64) -> NslTensor {
        NslTensor {
            magic: crate::tensor::TENSOR_MAGIC,
            data,
            shape,
            strides,
            ndim: 1,
            len,
            refcount: std::sync::atomic::AtomicI64::new(1),
            device: 0,
            dtype: 0, // f64 (CPU params)
            owns_data: 1,
            data_owner: 0,
            slab_managed: 0,
            tape_id: 0,
        }
    }

    /// The full lifecycle at ws=1: carve → gather (slice → full) → reduce
    /// (slot → grad slice) → elementwise step → release → teardown restore.
    /// The step's result is diffed BIT-FOR-BIT against
    /// `nsl_fase_fused_adamw_step` on an identical twin — at ws=1 the slice
    /// is the whole tensor, so any arithmetic divergence in the elementwise
    /// arm shows up here without a GPU or a second rank.
    #[test]
    fn elementwise_ws1_lifecycle_is_bit_identical_to_the_fused_step() {
        let _g = zero_serial_lock();
        // Fresh context, stage 3, single rank (no backend, no shm).
        nsl_zero_destroy();
        assert_eq!(nsl_zero_init(3, 1), 0);
        nsl_zero3_enable();

        let mut theta = vec![0.5f64, -1.25, 2.0, 3.75];
        let mut shape = vec![4i64];
        let mut strides = vec![1i64];
        let t = owned_cpu_tensor(
            theta.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
        );
        let tp = &t as *const NslTensor as i64;

        assert_eq!(nsl_zero3_note_param(tp, 0), 0);
        assert_eq!(nsl_zero3_mark_elementwise(tp, 0, 0), 0);
        assert_eq!(nsl_zero3_mark_elementwise(tp, 0, 0), 0, "mark must be idempotent");
        zero3_register(tp);
        assert!(t.data.is_null(), "carve must drop the full replica");
        assert_eq!(nsl_zero3_residency(tp), ParameterResidency::Evicted as i64);

        // Gather materializes the full tensor from the slice.
        zero3_gather(tp);
        assert!(!t.data.is_null());
        assert_eq!(
            nsl_zero3_residency(tp),
            ParameterResidency::GatheredTemporary as i64
        );
        let gathered: Vec<f64> =
            unsafe { std::slice::from_raw_parts(t.data as *const f64, 4) }.to_vec();
        assert_eq!(gathered, vec![0.5, -1.25, 2.0, 3.75], "carve+gather must roundtrip");

        // Window gradient in an accumulator slot; the elementwise arm copies
        // it into the grad slice (ws=1: no reduce, no averaging).
        let mut grad = vec![0.1f64, -0.2, 0.3, -0.4];
        let g = owned_cpu_tensor(
            grad.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            4,
        );
        let list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(list, &g as *const NslTensor as i64);
        assert_eq!(nsl_zero3_reduce_grad_slot(list, 0), 0);

        // Item C: m/v come from the SLICE allocator, not a hand-rolled full
        // buffer — this is the production shape (codegen's deferred fill
        // calls exactly this), and it pins the three properties the counter
        // gates rest on: slice-sized, θ's dtype/device, and `optim_elems`
        // credited by the shard's element count.
        let elems_before = ZERO_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed);
        let m_ptr = nsl_zero3_alloc_elem_moment(tp, 0);
        let v_ptr = nsl_zero3_alloc_elem_moment(tp, 0);
        assert!(m_ptr != 0 && v_ptr != 0, "slice moment allocation returned null");
        for (name, p) in [("m", m_ptr), ("v", v_ptr)] {
            let mt = crate::tensor::NslTensor::from_ptr(p);
            assert_eq!(mt.len, 4, "{name}: slice moment must be shard-sized");
            assert_eq!(mt.ndim, 1, "{name}: slice moment is 1-D");
            assert_eq!(mt.dtype, 0, "{name}: slice moment must carry theta's dtype");
            assert_eq!(mt.device, 0, "{name}: slice moment must carry theta's device");
            assert!(
                unsafe { std::slice::from_raw_parts(mt.data as *const f64, 4) }
                    .iter()
                    .all(|x| *x == 0.0),
                "{name}: slice moment must start zeroed"
            );
        }
        assert_eq!(
            ZERO_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed) - elems_before,
            8,
            "each slice moment must credit `optim_elems` with its shard \
             elements (2 moments x 4) — that counter is the ONLY instrument \
             that separates owner-only moments from replicated ones"
        );

        // The elementwise step vs the fused step on an identical twin.
        let (lr, b1, b2, eps, wd) = (0.002f64, 0.9f64, 0.95f64, 1e-8f64, 0.01f64);
        let (bc1, bc2) = (1.0 / (1.0 - b1), 1.0 / (1.0 - b2)); // t=1
        assert_eq!(
            nsl_zero3_elem_adamw_step(
                tp, m_ptr, v_ptr, 0,
                lr, b1, 1.0 - b1, b2, 1.0 - b2, eps, wd, bc1, bc2,
            ),
            0
        );
        let m: Vec<f64> = unsafe {
            std::slice::from_raw_parts(
                crate::tensor::NslTensor::from_ptr(m_ptr).data as *const f64,
                4,
            )
        }
        .to_vec();
        let v: Vec<f64> = unsafe {
            std::slice::from_raw_parts(
                crate::tensor::NslTensor::from_ptr(v_ptr).data as *const f64,
                4,
            )
        }
        .to_vec();
        crate::tensor::nsl_tensor_free(m_ptr);
        crate::tensor::nsl_tensor_free(v_ptr);

        // Twin: plain fused step over the same starting values.
        let mut theta2 = vec![0.5f64, -1.25, 2.0, 3.75];
        let mut m2 = vec![0.0f64; 4];
        let mut v2 = vec![0.0f64; 4];
        let t2 = owned_cpu_tensor(theta2.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), strides.as_mut_ptr(), 4);
        let m2t = owned_cpu_tensor(m2.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), strides.as_mut_ptr(), 4);
        let v2t = owned_cpu_tensor(v2.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), strides.as_mut_ptr(), 4);
        let g2 = owned_cpu_tensor(grad.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), strides.as_mut_ptr(), 4);
        crate::fase_step::nsl_fase_fused_adamw_step(
            &t2 as *const NslTensor as i64,
            &m2t as *const NslTensor as i64,
            &v2t as *const NslTensor as i64,
            &g2 as *const NslTensor as i64,
            lr, b1, 1.0 - b1, b2, 1.0 - b2, eps, wd, bc1, bc2,
        );
        let stepped: Vec<f64> =
            unsafe { std::slice::from_raw_parts(t.data as *const f64, 4) }.to_vec();
        assert_eq!(
            stepped.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            theta2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "elementwise step must be BIT-identical to the fused step"
        );
        assert_eq!(
            m.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            m2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "first moment must match"
        );
        assert_eq!(
            v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            v2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "second moment must match"
        );

        // A θ-mutating callback writes the GATHERED full tensor after the
        // step; release must persist this rank's region into the slice
        // before freeing (review finding: tensor-granular keeps such writes
        // on the owner's replica — dropping them only for elementwise
        // params would half-mutate a mixed-mode model).
        unsafe {
            *(t.data as *mut f64) = 42.0;
        }

        // Release frees the full transient on this (only) rank...
        zero3_release(tp);
        assert!(t.data.is_null());
        assert_eq!(nsl_zero3_residency(tp), ParameterResidency::Evicted as i64);
        // ...and teardown restores full residency with the UPDATED values
        // (the persist-into-slice happened at the step).
        nsl_zero3_teardown();
        assert!(!t.data.is_null(), "teardown must leave full replicas");
        let final_theta: Vec<f64> =
            unsafe { std::slice::from_raw_parts(t.data as *const f64, 4) }.to_vec();
        // Element 0 carries the callback-style mutation (release persisted
        // it); the rest must be the post-step values bit-for-bit.
        assert_eq!(
            final_theta[0].to_bits(),
            42.0f64.to_bits(),
            "release dropped a post-step write to the gathered full tensor"
        );
        assert_eq!(
            final_theta[1..].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            theta2[1..].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "teardown gather must serve the post-step slice"
        );
        unsafe {
            crate::memory::checked_free(t.data as *mut u8, 32);
        }
        crate::list::nsl_list_free(list);
        nsl_zero_destroy();
    }

    /// Item C at ws=2, rank 1 — the half of the elementwise contract the
    /// ws=1 lifecycle above CANNOT see. At ws=1 the region offset is
    /// `rank * shard == 0` and the shard IS the whole tensor, so
    /// slice-local `m[i]` and region-offset `m[off + i]` are the same
    /// address, and `m.len == shard` and `m.len == theta.len` are the same
    /// predicate. Restore either of main's replicated conventions and that
    /// test stays green — including its `optim_elems` delta, because at
    /// ws=1 a full replica IS shard-sized. This test runs the only rank
    /// where the two conventions disagree and pins the split directly:
    ///
    ///   * moments are indexed at their OWN base (`m[0..shard]` carries
    ///     this rank's update, not `m[shard..2*shard]`),
    ///   * θ is written ONLY inside `[rank*shard, (rank+1)*shard)`,
    ///   * `optim_elems` credits `shard`, not `theta.len`.
    ///
    /// The two collective-dependent stages are SUBSTITUTED, not faked. At
    /// ws=2 with both ranks holding identical values — which is exactly
    /// what the composed sim-gpu gates run, and why they are bit-exact —
    /// a correct all_gather reproduces the original full vector, and
    /// reduce_scatter with the 1/ws average reproduces this rank's region
    /// of the full gradient. The transient is allocated through the same
    /// `checked_alloc_zeroed` gather's host arm uses. Everything else —
    /// the carve, the moment allocator, the step, the persist — is the
    /// production path.
    #[test]
    fn elementwise_rank1_of_2_indexes_moments_slice_locally() {
        let _g = zero_serial_lock();
        nsl_zero_destroy();
        // `nsl_zero_init(3, 2)` returns -3 without NSL_TP_SHM_PATH (ws>1
        // opens the spawner's shared-memory file and blocks on a second
        // process). Init at ws=1 and retarget the context: the CPU
        // elementwise carve and step read nothing from it but
        // `(rank, world_size)`.
        assert_eq!(nsl_zero_init(3, 1), 0);
        {
            let mut g = ZERO_CTX.lock().unwrap();
            let c = g.as_mut().expect("context");
            c.rank = 1;
            c.world_size = 2;
        }
        nsl_zero3_enable();

        let full0 = [0.5f64, -1.25, 2.0, 3.75, -0.75, 1.5, -2.5, 0.25];
        let grad_full = [0.1f64, -0.2, 0.3, -0.4, 0.05, -0.15, 0.35, -0.45];
        let mut theta = full0.to_vec();
        let mut shape = vec![8i64];
        let mut strides = vec![1i64];
        let t = owned_cpu_tensor(
            theta.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            8,
        );
        let tp = &t as *const NslTensor as i64;

        assert_eq!(nsl_zero3_note_param(tp, 0), 0);
        // `sr = 0`: plain f32 elementwise storage. #493 widened this call to
        // carry the plan entry's storage decision; this test covers the
        // non-SR arm, which is where slice-local moment indexing lives.
        assert_eq!(nsl_zero3_mark_elementwise(tp, 0, 0), 0);
        zero3_register(tp); // REAL carve — rank 1 takes theta[4..8]
        assert!(t.data.is_null(), "carve must drop the full replica");

        // Substituted all_gather (see the note above).
        let full_bytes = 8 * std::mem::size_of::<f64>();
        let transient = crate::memory::checked_alloc_zeroed(full_bytes) as *mut f64;
        unsafe { std::ptr::copy_nonoverlapping(full0.as_ptr(), transient, 8) };
        crate::tensor::NslTensor::from_ptr(tp).data = transient as *mut c_void;

        // Substituted reduce_scatter, plus the shard/handshake state it sets.
        {
            let mut g = ZERO3_TABLE.lock().unwrap();
            let e = g.as_mut().unwrap().get_mut(&tp).unwrap();
            e.state = ParameterResidency::GatheredTemporary;
            let s = e.elem.as_mut().expect("carved");
            assert_eq!(s.shard, 4, "ws=2 carve must halve an 8-element param");
            assert_eq!(s.elem_bytes, 8, "f64 param");
            let buf = ShardBuf::alloc(s.shard * s.elem_bytes, 0);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    grad_full.as_ptr().add(4),
                    buf.ptr() as *mut f64,
                    4,
                );
            }
            s.grad = Some(buf);
            s.grad_ready = true;
        }

        // Real allocator: SHARD-sized, and `optim_elems` credited with the
        // shard — 8, not the 16 a replicated pair would credit. Unlike the
        // ws=1 test's identical-looking delta, this one discriminates.
        let elems_before = ZERO_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed);
        let m_ptr = nsl_zero3_alloc_elem_moment(tp, 0);
        let v_ptr = nsl_zero3_alloc_elem_moment(tp, 0);
        for (name, p) in [("m", m_ptr), ("v", v_ptr)] {
            assert_eq!(
                crate::tensor::NslTensor::from_ptr(p).len,
                4,
                "{name}: at ws=2 the moment must be the SHARD, not the parameter"
            );
        }
        assert_eq!(
            ZERO_OPTIM_ELEMS.load(std::sync::atomic::Ordering::Relaxed) - elems_before,
            8,
            "2 moments x 4 shard elements; 16 means the fill handed back \
             full replicas and ZeRO-3 removed nothing"
        );

        let (lr, b1, b2, eps, wd) = (0.002f64, 0.9f64, 0.95f64, 1e-8f64, 0.01f64);
        let (bc1, bc2) = (1.0 / (1.0 - b1), 1.0 / (1.0 - b2)); // t=1
        assert_eq!(
            nsl_zero3_elem_adamw_step(
                tp, m_ptr, v_ptr, 0, lr, b1, 1.0 - b1, b2, 1.0 - b2, eps, wd, bc1, bc2,
            ),
            0
        );

        let bits = |xs: &[f64]| xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let m: Vec<f64> = unsafe {
            std::slice::from_raw_parts(
                crate::tensor::NslTensor::from_ptr(m_ptr).data as *const f64,
                4,
            )
        }
        .to_vec();
        let v: Vec<f64> = unsafe {
            std::slice::from_raw_parts(
                crate::tensor::NslTensor::from_ptr(v_ptr).data as *const f64,
                4,
            )
        }
        .to_vec();
        let stepped: Vec<f64> =
            unsafe { std::slice::from_raw_parts(transient as *const f64, 8) }.to_vec();
        let persisted: Vec<f64> = {
            let g = ZERO3_TABLE.lock().unwrap();
            let s = g.as_ref().unwrap()[&tp].elem.as_ref().unwrap();
            unsafe { std::slice::from_raw_parts(s.slice.ptr() as *const f64, 4) }.to_vec()
        };
        crate::tensor::nsl_tensor_free(m_ptr);
        crate::tensor::nsl_tensor_free(v_ptr);

        // Twin: the fused step over THIS RANK'S region only.
        let mut theta2 = full0[4..].to_vec();
        let mut grad2 = grad_full[4..].to_vec();
        let mut m2 = vec![0.0f64; 4];
        let mut v2 = vec![0.0f64; 4];
        let mut shape4 = vec![4i64];
        let (sh4, st) = (shape4.as_mut_ptr(), strides.as_mut_ptr());
        let mk = |d: *mut f64| owned_cpu_tensor(d as *mut c_void, sh4, st, 4);
        let (t2, m2t, v2t, g2) = (
            mk(theta2.as_mut_ptr()),
            mk(m2.as_mut_ptr()),
            mk(v2.as_mut_ptr()),
            mk(grad2.as_mut_ptr()),
        );
        crate::fase_step::nsl_fase_fused_adamw_step(
            &t2 as *const NslTensor as i64,
            &m2t as *const NslTensor as i64,
            &v2t as *const NslTensor as i64,
            &g2 as *const NslTensor as i64,
            lr,
            b1,
            1.0 - b1,
            b2,
            1.0 - b2,
            eps,
            wd,
            bc1,
            bc2,
        );

        // The decisive pair. Moments at their OWN base: with `m[off + i]`
        // the shard-sized buffer would keep its zeros here (and scribble
        // past its end).
        assert_eq!(
            bits(&m),
            bits(&m2),
            "first moment must be written at the SLICE base — m[0..shard] \
             holds rank 1's update, not m[rank*shard..]"
        );
        assert_eq!(bits(&v), bits(&v2), "second moment must be slice-local");
        // θ at the region offset, and NOWHERE else.
        assert_eq!(
            bits(&stepped[..4]),
            bits(&full0[..4]),
            "rank 1 wrote into rank 0's region of the gathered tensor"
        );
        assert_eq!(
            bits(&stepped[4..]),
            bits(&theta2),
            "rank 1's region must be BIT-identical to the fused step"
        );
        assert_eq!(
            bits(&persisted),
            bits(&theta2),
            "the step must persist its region into the slice"
        );

        unsafe { crate::memory::checked_free(transient as *mut u8, full_bytes) };
        crate::tensor::NslTensor::from_ptr(tp).data = std::ptr::null_mut();
        // `nsl_zero3_teardown()` before destroy, per every sibling here, and
        // it is NOT optional bookkeeping: `nsl_zero_destroy` leaves
        // ZERO3_TABLE populated, so an entry keyed on this stack tensor
        // outlives the frame and the NEXT test's teardown walks it and
        // dereferences freed stack memory ("bad magic ... possible
        // use-after-free", a non-unwinding abort that kills the whole
        // binary and reports against the innocent test). Teardown restores
        // a full replica, so free what it hands back.
        nsl_zero3_teardown();
        if !t.data.is_null() {
            unsafe { crate::memory::checked_free(t.data as *mut u8, full_bytes) };
            crate::tensor::NslTensor::from_ptr(tp).data = std::ptr::null_mut();
        }
        nsl_zero_destroy();
    }

    /// The reduce/step handshake: a second step without a fresh reduce must
    /// abort (stale-gradient class). Verified via the grad_ready flag rather
    /// than a death test — the abort arm itself is a process::abort.
    #[test]
    fn elementwise_step_consumes_the_reduce_handshake() {
        let _g = zero_serial_lock();
        nsl_zero_destroy();
        assert_eq!(nsl_zero_init(3, 1), 0);
        nsl_zero3_enable();

        let mut theta = vec![1.0f64, 2.0];
        let mut shape = vec![2i64];
        let mut strides = vec![1i64];
        let t = owned_cpu_tensor(
            theta.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            2,
        );
        let tp = &t as *const NslTensor as i64;
        assert_eq!(nsl_zero3_note_param(tp, 0), 0);
        assert_eq!(nsl_zero3_mark_elementwise(tp, 0, 0), 0);
        zero3_register(tp);
        zero3_gather(tp);

        let mut grad = vec![0.5f64, 0.5];
        let g = owned_cpu_tensor(
            grad.as_mut_ptr() as *mut c_void,
            shape.as_mut_ptr(),
            strides.as_mut_ptr(),
            2,
        );
        let list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(list, &g as *const NslTensor as i64);
        assert_eq!(nsl_zero3_reduce_grad_slot(list, 0), 0);
        let ready = {
            let tbl = ZERO3_TABLE.lock().unwrap();
            tbl.as_ref()
                .and_then(|m| m.get(&tp))
                .and_then(|e| e.elem.as_ref())
                .map(|s| s.grad_ready)
        };
        assert_eq!(ready, Some(true), "reduce must arm the handshake");

        let mut m = vec![0.0f64; 2];
        let mut v = vec![0.0f64; 2];
        let ms = owned_cpu_tensor(m.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), strides.as_mut_ptr(), 2);
        let vs = owned_cpu_tensor(v.as_mut_ptr() as *mut c_void, shape.as_mut_ptr(), strides.as_mut_ptr(), 2);
        assert_eq!(
            nsl_zero3_elem_adamw_step(
                tp,
                &ms as *const NslTensor as i64,
                &vs as *const NslTensor as i64,
                0,
                0.001, 0.9, 0.1, 0.95, 0.05, 1e-8, 0.0, 1.0, 1.0,
            ),
            0
        );
        let ready = {
            let tbl = ZERO3_TABLE.lock().unwrap();
            tbl.as_ref()
                .and_then(|m| m.get(&tp))
                .and_then(|e| e.elem.as_ref())
                .map(|s| s.grad_ready)
        };
        assert_eq!(ready, Some(false), "the step must consume the handshake");

        zero3_release(tp);
        nsl_zero3_teardown();
        if !t.data.is_null() {
            unsafe { crate::memory::checked_free(t.data as *mut u8, 16) };
        }
        crate::list::nsl_list_free(list);
        nsl_zero_destroy();
    }

    /// Item 16×11: the composed bf16-sr elementwise lifecycle at ws=1,
    /// diffed BIT-FOR-BIT against the plain SR mirror path on an identical
    /// twin. At ws=1 the slice is the whole tensor and `ctr_base` carries a
    /// zero rank offset, so the composed carve/gather/step must reproduce
    /// register/upload/step exactly: same f32→bf16 carve cast as the
    /// mirror's register, same widen at gather as the mirror's upload, same
    /// SR kernel with the same (seed, step, param, element) counters at the
    /// step. Any divergence in the composed plumbing shows up here without
    /// a second rank. (The rank-offset half of the counter claim is the
    /// 2-rank CLI gate's job — rank is 0 here by construction.)
    #[cfg(feature = "cuda")]
    #[test]
    fn srbf16_elementwise_ws1_matches_the_mirror_path_bitwise() {
        let _g = zero_serial_lock();
        const N: usize = 8;
        let init: [f32; N] = [0.5, -1.25, 2.0, 3.75, -0.001, 7.5, -42.0, 0.375];
        let grad: [f32; N] = [0.1, -0.2, 0.3, -0.4, 0.05, -0.6, 0.7, -0.008];

        fn gpu_f32_tensor(
            vals: &[f32],
            shape: *mut i64,
            strides: *mut i64,
        ) -> NslTensor {
            crate::cuda::inner::ensure_context();
            let bytes = vals.len() * 4;
            let dev = crate::cuda::inner::alloc_managed(bytes);
            crate::cuda::inner::memcpy_htod(dev, vals.as_ptr() as *const c_void, bytes);
            let mut t = owned_cpu_tensor(dev, shape, strides, vals.len() as i64);
            t.device = 1;
            t.dtype = 1; // f32 (GPU params)
            t
        }
        fn read_gpu_f32(ptr: *const c_void, n: usize) -> Vec<u32> {
            let mut host = vec![0f32; n];
            crate::cuda::inner::memcpy_dtoh(
                host.as_mut_ptr() as *mut c_void,
                ptr,
                n * 4,
            );
            host.iter().map(|x| x.to_bits()).collect()
        }

        nsl_zero_destroy();
        assert_eq!(nsl_zero_init(3, 1), 0);
        nsl_zero3_enable();
        crate::sr_bf16::nsl_sr_bf16_enable();

        let mut shape = vec![N as i64];
        let mut strides = vec![1i64];

        // ── Composed side: SR note + elementwise mark + carve ──────────────
        let t = gpu_f32_tensor(&init, shape.as_mut_ptr(), strides.as_mut_ptr());
        let tp = &t as *const NslTensor as i64;
        crate::sr_bf16::nsl_sr_bf16_note_param(tp, 0);
        assert_eq!(nsl_zero3_note_param(tp, 0), 0);
        // sr=1: the plan's storage decision travels with the mark.
        assert_eq!(nsl_zero3_mark_elementwise(tp, 0, 1), 0);
        zero3_register(tp);
        assert!(t.data.is_null(), "carve must drop the full replica");
        {
            let tbl = ZERO3_TABLE.lock().unwrap();
            let sr = tbl
                .as_ref()
                .and_then(|m| m.get(&tp))
                .and_then(|e| e.elem.as_ref())
                .and_then(|s| s.sr_param_idx);
            assert_eq!(sr, Some(0), "carve must record the SR counter block");
        }

        // Gather widens the bf16 slice into the f32 working transient.
        zero3_gather(tp);
        assert!(!t.data.is_null());

        let gt = gpu_f32_tensor(&grad, shape.as_mut_ptr(), strides.as_mut_ptr());
        let list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(list, &gt as *const NslTensor as i64);
        assert_eq!(nsl_zero3_reduce_grad_slot(list, 0), 0);

        let zeros = [0f32; N];
        let ms = gpu_f32_tensor(&zeros, shape.as_mut_ptr(), strides.as_mut_ptr());
        let vs = gpu_f32_tensor(&zeros, shape.as_mut_ptr(), strides.as_mut_ptr());
        let (lr, b1, b2, eps, wd) = (0.002f64, 0.9f64, 0.95f64, 1e-8f64, 0.01f64);
        let (bc1, bc2) = (1.0 / (1.0 - b1), 1.0 / (1.0 - b2)); // t=1
        assert_eq!(
            nsl_zero3_elem_sr_adamw_step(
                tp,
                &ms as *const NslTensor as i64,
                &vs as *const NslTensor as i64,
                0,
                lr, b1, 1.0 - b1, b2, 1.0 - b2, eps, wd, bc1, bc2,
                1, // step
            ),
            0
        );
        // Post-step the working transient region is refreshed by widening.
        let composed_theta = read_gpu_f32(t.data, N);
        let composed_m = read_gpu_f32(ms.data, N);
        let composed_v = read_gpu_f32(vs.data, N);

        // ── Plain SR twin: note + register (mirror) + fused SR step ────────
        let t2 = gpu_f32_tensor(&init, shape.as_mut_ptr(), strides.as_mut_ptr());
        let t2p = &t2 as *const NslTensor as i64;
        crate::sr_bf16::nsl_sr_bf16_note_param(t2p, 0);
        crate::sr_bf16::srbf16_register(t2p);
        assert!(t2.data.is_null(), "register must free the f32 storage");
        let m2 = gpu_f32_tensor(&zeros, shape.as_mut_ptr(), strides.as_mut_ptr());
        let v2 = gpu_f32_tensor(&zeros, shape.as_mut_ptr(), strides.as_mut_ptr());
        let mp2 = gpu_f32_tensor(&grad, shape.as_mut_ptr(), strides.as_mut_ptr());
        crate::sr_bf16::nsl_sr_bf16_step_adamw(
            t2p,
            &m2 as *const NslTensor as i64,
            &v2 as *const NslTensor as i64,
            &mp2 as *const NslTensor as i64,
            lr, b1, 1.0 - b1, b2, 1.0 - b2, eps, wd, bc1, bc2,
            1, // step
        );
        // Materialize the twin's working view = widen(mirror).
        crate::sr_bf16::srbf16_upload(t2p);
        assert!(!t2.data.is_null());
        let plain_theta = read_gpu_f32(t2.data, N);
        let plain_m = read_gpu_f32(m2.data, N);
        let plain_v = read_gpu_f32(v2.data, N);

        assert_eq!(
            composed_theta, plain_theta,
            "composed slice (widened) must equal the mirror (widened) BIT-FOR-BIT"
        );
        assert_eq!(composed_m, plain_m, "first moment must match bitwise");
        assert_eq!(composed_v, plain_v, "second moment must match bitwise");

        // ── Teardown both backends; free what they left resident ──────────
        zero3_release(tp);
        nsl_zero3_teardown();
        assert!(!t.data.is_null(), "teardown must restore a full replica");
        let restored = read_gpu_f32(t.data, N);
        assert_eq!(
            restored, plain_theta,
            "teardown gather must serve the post-step slice"
        );
        crate::sr_bf16::nsl_sr_bf16_teardown();
        crate::cuda::inner::ensure_context();
        for tt in [&t, &t2, &gt, &ms, &vs, &m2, &v2, &mp2] {
            if !tt.data.is_null() {
                crate::cuda::inner::free_managed(tt.data);
            }
        }
        crate::list::nsl_list_free(list);
        nsl_zero_destroy();
    }
}
