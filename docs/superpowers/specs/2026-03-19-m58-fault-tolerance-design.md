# M58: Cluster-Scale Fault Tolerance & Elastic Execution — Design Spec

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M58
**Prerequisites:** M30 (Tensor Parallelism & NCCL), M43 (Pipeline Parallelism)
**Dependencies:** M61 (Cluster Debugging benefits from elastic infrastructure)

---

## Overview

When a GPU node dies during a 100,000-GPU training run, the system automatically detects the failure within seconds, re-shards weights and optimizer state to surviving nodes, loads the last micro-checkpoint, rebuilds NCCL communicators with the new world topology, and resumes training — without human intervention and with minimal lost compute.

**Key design:** Fault tolerance is a compile-time property, not a runtime library. The `@fault_tolerant` decorator on a `train` block tells the compiler to emit heartbeat infrastructure, checkpoint hooks, and re-sharding logic alongside the normal training loop. The `distribute: "dp=elastic"` configuration declares that the data-parallel dimension can grow or shrink dynamically while tensor-parallel and pipeline-parallel dimensions remain fixed (re-sharding TP/PP state is prohibitively expensive at scale).

**Why this is different from PyTorch Elastic (torchrun):** PyTorch Elastic restarts the entire process group when a node fails, requiring all ranks to reload from checkpoint. NSL's approach is incremental — only the failed rank's data-parallel replica is removed, surviving ranks continue with their in-memory state, and only the micro-checkpoint of the failed rank's optimizer state needs to be redistributed. This reduces recovery time from minutes to seconds.

**Why this requires a compiler:** The compiler statically knows the tensor shapes, communication patterns, and memory layout. It can pre-compute the re-sharding plan for every possible failure scenario (up to K simultaneous failures) at build time. Python frameworks must discover this at runtime via expensive coordination protocols.

---

## Section 1: Language Surface

### `@fault_tolerant` Decorator

```
@fault_tolerant
train FaultTolerantPretraining:
    model: LLaMA70B
    data: WebCorpus
    epochs: 1
    optimizer: AdamW(lr=3e-4)
    scheduler: CosineAnnealing(warmup=2000)

    distribute: "dp=elastic, tp=4, pp=4"

    # Fault tolerance configuration
    heartbeat_interval_ms: 5000     # UDP heartbeat every 5s
    heartbeat_timeout_ms: 30000     # node considered dead after 30s silence
    checkpoint_interval: 100        # async micro-checkpoint every 100 steps
    checkpoint_storage: "s3://checkpoints/run-001/"
    max_failures: 8                 # abort if more than 8 nodes fail total
    min_world_size: 64              # abort if world shrinks below 64 DP replicas

    callbacks:
        on_node_failure(rank, step):
            log(f"Node {rank} failed at step {step}, re-sharding...")
        on_recovery_complete(new_world_size, step):
            log(f"Recovered: {new_world_size} nodes, resuming at step {step}")
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `heartbeat_interval_ms` | `int` | 5000 | UDP heartbeat send interval per rank |
| `heartbeat_timeout_ms` | `int` | 30000 | Time before declaring a node dead |
| `checkpoint_interval` | `int` | 100 | Steps between async micro-checkpoints |
| `checkpoint_storage` | `str` | `"./checkpoints/"` | S3/NFS/NVMe-oF path for checkpoint data |
| `max_failures` | `int` | `world_size / 8` | Max cumulative failures before abort |
| `min_world_size` | `int` | 1 | Minimum DP replicas to continue training |
| `max_recovery_time_ms` | `int` | 60000 | Max time for recovery before abort |
| `checkpoint_compression` | `str` | `"lz4"` | Compression for checkpoint data (none/lz4/zstd) |

### Elastic Dimension Syntax

```
# Only DP dimension is elastic — TP and PP are fixed
distribute: "dp=elastic, tp=4, pp=4"

# DP dimension with explicit bounds
distribute: "dp=elastic(min=32, max=256), tp=8, pp=2"

# Error: TP cannot be elastic (would require weight re-sharding mid-step)
distribute: "dp=64, tp=elastic, pp=4"   # COMPILE ERROR
```

**Rationale for DP-only elasticity:** Data parallelism replicates the full model on each rank. Adding or removing a DP replica only requires adjusting the gradient all-reduce group — no weight redistribution. Tensor parallelism shards individual weight matrices across ranks; removing a TP rank requires redistributing weight slices across all surviving ranks in the TP group, which is O(model_size) communication during recovery. Pipeline parallelism assigns complete layers to stages; removing a PP rank requires re-partitioning and migrating layers. Both TP and PP re-sharding are too expensive for online recovery.

### Callbacks

```
@fault_tolerant
train WithCallbacks:
    model: MyModel
    # ...
    distribute: "dp=elastic, tp=4"

    callbacks:
        on_node_failure(rank: int, step: int):
            # Called on all surviving ranks when a failure is detected
            alert_slack(f"Rank {rank} died at step {step}")

        on_recovery_complete(new_world_size: int, step: int):
            # Called after communicators are rebuilt and training resumes
            log_metric("world_size", new_world_size, step)

        on_checkpoint_written(path: str, step: int):
            # Called after each successful micro-checkpoint
            log(f"Checkpoint saved: {path} at step {step}")

        on_abort(reason: str, step: int):
            # Called when max_failures or min_world_size is violated
            alert_pager(f"Training aborted: {reason} at step {step}")
```

---

## Section 2: Architecture

### System Topology

```
                    ┌──────────────────────┐
                    │   Coordinator        │ ← Rank 0 (or dedicated)
                    │   - Heartbeat RX     │
                    │   - Failure Detector  │
                    │   - Recovery Planner  │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────┴─────┐        ┌────┴────┐        ┌─────┴─────┐
    │  DP Group │        │DP Group │        │ DP Group  │
    │  Rank 0   │        │ Rank 1  │        │ Rank 2    │
    │ ┌───┬───┐ │        │┌──┬───┐ │        │ ┌───┬───┐ │
    │ │TP0│TP1│ │        ││TP│TP1│ │        │ │TP0│TP1│ │
    │ │PP0│PP0│ │        ││0 │PP0│ │        │ │PP0│PP0│ │
    │ ├───┼───┤ │        │├──┼───┤ │        │ ├───┼───┤ │
    │ │TP0│TP1│ │        ││TP│TP1│ │        │ │TP0│TP1│ │
    │ │PP1│PP1│ │        ││0 │PP1│ │        │ │PP1│PP1│ │
    │ └───┴───┘ │        │└──┴───┘ │        │ └───┴───┘ │
    └───────────┘        └─────────┘        └───────────┘
          │                    │                    │
          │         Heartbeat (UDP out-of-band)     │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────┴───────────┐
                    │  Heartbeat Multicast  │
                    │  (separate from NCCL) │
                    └──────────────────────┘
```

**Key insight:** The heartbeat channel is completely separate from NCCL. NCCL communicators block when a rank dies (hanging all-reduce). The out-of-band UDP heartbeat detects failure independently, allowing the coordinator to cancel the hung NCCL operation and initiate recovery.

### Failure Detection Flow

```
1. Rank N misses heartbeat_timeout_ms
       │
2. Coordinator marks Rank N as SUSPECT
       │
3. Coordinator sends PROBE to Rank N (3 retries, 1s apart)
       │
4a. Rank N responds → SUSPECT cleared, resume
4b. Rank N silent → CONFIRMED DEAD
       │
5. Coordinator broadcasts FAILURE_DETECTED(rank=N) to all surviving ranks
       │
6. All ranks:
   a. Cancel pending NCCL operations (cuStreamWaitValue + timeout)
   b. Flush local micro-checkpoint to storage
   c. ACK to coordinator
       │
7. Coordinator computes new rank mapping:
   - Remove Rank N from DP group
   - Compute new DP degree (old - 1)
   - TP/PP groups unchanged (no re-sharding needed)
       │
8. Coordinator broadcasts NEW_TOPOLOGY(rank_map, dp_degree)
       │
9. All surviving ranks:
   a. Destroy old NCCL communicator
   b. Create new NCCL communicator with updated rank list
   c. Adjust gradient all-reduce to new DP degree
   d. Adjust learning rate schedule if lr scales with world_size
       │
10. Coordinator broadcasts RESUME(step=last_checkpoint_step)
       │
11. All ranks load micro-checkpoint for step, continue training
```

### Recovery Time Budget

| Phase | Target | Notes |
|-------|--------|-------|
| Failure detection | < 30s | Configurable via `heartbeat_timeout_ms` |
| NCCL cancellation | < 5s | Via `cuStreamWaitValue` timeout |
| Checkpoint flush | < 10s | Async checkpoint already in flight |
| Communicator rebuild | < 5s | `ncclCommInitRank` with fewer ranks |
| Checkpoint reload | < 10s | Only reload micro-checkpoint diff |
| **Total recovery** | **< 60s** | vs. 5-15 minutes for full restart |

---

## Section 3: Heartbeat Monitor

### UDP Heartbeat Protocol

Each rank sends a fixed-size UDP datagram to the coordinator at `heartbeat_interval_ms`:

```rust
/// 32-byte heartbeat packet — fixed size for zero-alloc parsing.
#[repr(C, packed)]
pub struct HeartbeatPacket {
    pub magic: u32,          // 0x4E534C48 ("NSLH")
    pub version: u8,         // protocol version (1)
    pub rank: u16,           // sender's global rank
    pub status: u8,          // 0=healthy, 1=degraded, 2=shutting_down
    pub step: u64,           // current training step
    pub gpu_util: u8,        // GPU utilization % (0-100)
    pub gpu_temp: u8,        // GPU temperature celsius (0-255)
    pub mem_used_mb: u16,    // GPU memory used (MB, max 65535)
    pub loss_bits: u32,      // f32 bits of current loss value
    pub padding: [u8; 4],    // align to 32 bytes
}
```

**Why UDP, not TCP:** TCP connections between 100k ranks create O(N^2) state. UDP is stateless — the coordinator just listens on one port. Lost heartbeats are tolerated by the timeout mechanism (missing one heartbeat is fine; missing `timeout / interval` consecutive heartbeats triggers SUSPECT).

**Why out-of-band, not NCCL:** NCCL all-reduce operations block the entire communicator when one rank is unresponsive. If the heartbeat were sent via NCCL, a dead rank would block heartbeat delivery to all other ranks, making failure detection impossible.

### Coordinator State Machine

```rust
/// Per-rank state tracked by the coordinator.
pub struct RankState {
    pub rank: u16,
    pub status: RankStatus,
    pub last_heartbeat: Instant,
    pub last_step: u64,
    pub probe_count: u8,
    pub failure_time: Option<Instant>,
}

pub enum RankStatus {
    Healthy,
    Suspect,          // missed heartbeat timeout
    Probing(u8),      // probe sent, waiting for response (retry count)
    Dead,             // confirmed dead
    Recovering,       // in the process of being removed from topology
}
```

### Module: `crates/nsl-runtime/src/heartbeat.rs`

```rust
use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub struct HeartbeatSender {
    socket: UdpSocket,
    coordinator_addr: std::net::SocketAddr,
    rank: u16,
    interval: Duration,
    running: AtomicBool,
}

impl HeartbeatSender {
    pub fn new(
        coordinator_addr: &str,
        rank: u16,
        interval_ms: u64,
    ) -> std::io::Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.set_nonblocking(false)?;
        Ok(Self {
            socket,
            coordinator_addr: coordinator_addr.parse().unwrap(),
            rank,
            interval: Duration::from_millis(interval_ms),
            running: AtomicBool::new(false),
        })
    }

    /// Spawn the heartbeat sender thread. Non-blocking to the training loop.
    pub fn start(&self, step_ref: &'static std::sync::atomic::AtomicU64) {
        self.running.store(true, Ordering::SeqCst);
        // Thread sends HeartbeatPacket at self.interval
        // Reads current step from step_ref, GPU metrics from NVML
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

pub struct HeartbeatReceiver {
    socket: UdpSocket,
    rank_states: Vec<RankState>,
    timeout: Duration,
    probe_interval: Duration,
    max_probes: u8,
}

impl HeartbeatReceiver {
    /// Main coordinator loop. Runs on rank 0 in a dedicated thread.
    pub fn run(&mut self, failure_tx: crossbeam::channel::Sender<FailureEvent>) {
        loop {
            // 1. Try recv heartbeat (non-blocking, 100ms poll)
            // 2. Update rank_states with received heartbeats
            // 3. Check for timeouts → transition to Suspect
            // 4. Send probes to Suspect ranks
            // 5. Transition Probing(max) → Dead
            // 6. Send FailureEvent on failure_tx for Dead transitions
        }
    }
}

pub struct FailureEvent {
    pub rank: u16,
    pub step: u64,
    pub detected_at: Instant,
}
```

---

## Section 4: Micro-Checkpointing

### Design

Micro-checkpoints capture the minimum state needed to resume training: model weights, optimizer state (momentum buffers, variance buffers), learning rate scheduler state, RNG state, data loader position, and the current training step. They are written asynchronously — the checkpoint is serialized in a background thread while the next training step proceeds on the GPU.

**Key constraint:** The checkpoint must be consistent — all tensors must correspond to the same training step. This is achieved by snapshotting CPU copies of the tensors at checkpoint time (the GPU continues forward with the next batch while the CPU writes the snapshot to storage).

### Checkpoint Format

```
Header (JSON):
{
    "format": "nslm-micro",
    "version": 3,
    "step": 45000,
    "world_size": 128,
    "dp_degree": 32,
    "tp_degree": 4,
    "pp_degree": 1,
    "rank": 17,
    "timestamp": "2026-03-19T14:30:00Z",
    "model_hash": "a1b2c3...",
    "tensors": [
        {"name": "layer.0.attn.q.weight", "shape": [4096, 4096], "dtype": "f32", "offset": 0},
        {"name": "layer.0.attn.q.weight.m1", "shape": [4096, 4096], "dtype": "f32", "offset": 67108864},
        {"name": "layer.0.attn.q.weight.v", "shape": [4096, 4096], "dtype": "f32", "offset": 134217728}
    ],
    "rng_state": "base64...",
    "data_position": {"shard": 42, "offset": 100000},
    "lr_scheduler_state": {"current_lr": 2.8e-4, "step": 45000}
}

Body: 64-byte aligned raw tensor data (same as .nslm format from M14)
```

### Async Checkpoint Pipeline

```
Training Step N:
  GPU: forward → backward → all-reduce → optimizer step
                                              │
                                    ┌─────────┴─────────┐
                                    │ If step % interval │
                                    │ == 0: snapshot     │
                                    └─────────┬─────────┘
                                              │
                      ┌───────────────────────┴──────────────────┐
                      │ cuMemcpyDtoHAsync: GPU weights → pinned  │
                      │ host buffer (non-blocking, on copy stream)│
                      └───────────────────────┬──────────────────┘
                                              │
Training Step N+1:    │                       │
  GPU: forward → ...  │            ┌──────────┴───────────┐
                      │            │ Background thread:   │
                      │            │ 1. Wait for DtoH     │
                      │            │ 2. Compress (LZ4)    │
                      │            │ 3. Write to storage  │
                      │            │ 4. Ack coordinator   │
                      │            └──────────────────────┘
```

### Module: `crates/nsl-runtime/src/micro_checkpoint.rs`

```rust
use std::path::PathBuf;
use std::sync::mpsc;

pub struct MicroCheckpointer {
    storage_path: PathBuf,
    interval: u64,
    compression: CompressionKind,
    write_thread: Option<std::thread::JoinHandle<()>>,
    tx: mpsc::Sender<CheckpointJob>,
    last_completed_step: std::sync::atomic::AtomicU64,
}

pub enum CompressionKind {
    None,
    Lz4,
    Zstd { level: i32 },
}

pub struct CheckpointJob {
    pub step: u64,
    pub rank: u16,
    pub tensors: Vec<(String, Vec<u8>)>,  // (name, raw_bytes)
    pub metadata: CheckpointMetadata,
}

pub struct CheckpointMetadata {
    pub rng_state: Vec<u8>,
    pub data_position: DataPosition,
    pub lr_scheduler_state: LrSchedulerState,
    pub dp_degree: u32,
    pub tp_degree: u32,
    pub pp_degree: u32,
}

impl MicroCheckpointer {
    pub fn new(config: CheckpointConfig) -> Self {
        let (tx, rx) = mpsc::channel();
        let storage_path = config.storage_path.clone();
        let compression = config.compression;

        let write_thread = std::thread::spawn(move || {
            // Background writer loop
            while let Ok(job) = rx.recv() {
                let path = storage_path.join(format!(
                    "rank_{:04}_step_{:08}.nslm",
                    job.rank, job.step
                ));
                Self::write_checkpoint(&path, &job, compression);
            }
        });

        Self {
            storage_path: config.storage_path,
            interval: config.interval,
            compression,
            write_thread: Some(write_thread),
            tx,
            last_completed_step: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Called from the training loop. Non-blocking: snapshots tensors and
    /// sends them to the background writer.
    pub fn maybe_checkpoint(
        &self,
        step: u64,
        rank: u16,
        tensors: &[(String, *const f32, usize)],
        metadata: CheckpointMetadata,
    ) {
        if step % self.interval != 0 {
            return;
        }
        // Copy tensor data from GPU to pinned host memory (async)
        // Then send CheckpointJob to background thread
        let mut tensor_copies = Vec::with_capacity(tensors.len());
        for (name, ptr, len) in tensors {
            let mut buf = vec![0u8; *len * 4]; // f32 = 4 bytes
            unsafe {
                std::ptr::copy_nonoverlapping(
                    *ptr as *const u8,
                    buf.as_mut_ptr(),
                    buf.len(),
                );
            }
            tensor_copies.push((name.clone(), buf));
        }
        let _ = self.tx.send(CheckpointJob {
            step,
            rank,
            tensors: tensor_copies,
            metadata,
        });
    }

    fn write_checkpoint(path: &PathBuf, job: &CheckpointJob, compression: CompressionKind) {
        // 1. Serialize JSON header
        // 2. Compress tensor data with selected algorithm
        // 3. Write atomically (write to .tmp, then rename)
    }
}
```

### Checkpoint Storage Backends

```rust
pub trait CheckpointStorage: Send + Sync {
    fn write(&self, key: &str, data: &[u8]) -> Result<(), StorageError>;
    fn read(&self, key: &str) -> Result<Vec<u8>, StorageError>;
    fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError>;
    fn delete(&self, key: &str) -> Result<(), StorageError>;
}

pub struct LocalStorage { root: PathBuf }
pub struct S3Storage { bucket: String, prefix: String, client: S3Client }
pub struct NvmeOfStorage { target: String, namespace: u32 }

impl CheckpointStorage for LocalStorage { ... }
impl CheckpointStorage for S3Storage { ... }
impl CheckpointStorage for NvmeOfStorage { ... }
```

---

## Section 5: Elastic Re-Sharding

### Re-Sharding Protocol

When a DP rank dies, the elastic re-shard adjusts the data-parallel dimension:

```
Before failure (dp=4, tp=2, pp=1):
  DP0: [TP0, TP1]   ← replicas of full model
  DP1: [TP0, TP1]
  DP2: [TP0, TP1]   ← THIS NODE DIES
  DP3: [TP0, TP1]

After recovery (dp=3, tp=2, pp=1):
  DP0: [TP0, TP1]   ← unchanged
  DP1: [TP0, TP1]   ← unchanged
  DP2: [TP0, TP1]   ← was DP3, renumbered

Changes:
  - New NCCL communicator with 3 DP ranks (was 4)
  - Gradient all-reduce divisor: 3 (was 4)
  - Effective batch size: 3 * micro_batch (was 4 * micro_batch)
  - Learning rate adjusted: lr * (3/4) if using linear scaling rule
```

### Rank Remapping

```rust
/// Compute new rank assignments after a failure.
pub fn compute_rank_remap(
    old_world_size: u32,
    dead_ranks: &[u16],
    tp_degree: u32,
    pp_degree: u32,
) -> RankRemap {
    // TP groups are contiguous: ranks [0..tp_degree) form TP group 0, etc.
    // PP stages are within each TP group.
    // DP is the outer dimension.

    let old_dp_degree = old_world_size / (tp_degree * pp_degree);
    let mut surviving_dp_groups: Vec<u32> = Vec::new();

    for dp_idx in 0..old_dp_degree {
        let group_start = dp_idx * tp_degree * pp_degree;
        let group_end = group_start + tp_degree * pp_degree;
        let any_dead = dead_ranks.iter().any(|&r| {
            (r as u32) >= group_start && (r as u32) < group_end
        });
        if !any_dead {
            surviving_dp_groups.push(dp_idx);
        }
        // If any rank in a TP/PP group dies, the entire DP replica is lost
        // (TP/PP cannot function with missing members)
    }

    let new_dp_degree = surviving_dp_groups.len() as u32;

    RankRemap {
        new_world_size: new_dp_degree * tp_degree * pp_degree,
        new_dp_degree,
        tp_degree,  // unchanged
        pp_degree,  // unchanged
        old_to_new: build_rank_mapping(&surviving_dp_groups, tp_degree, pp_degree),
    }
}

pub struct RankRemap {
    pub new_world_size: u32,
    pub new_dp_degree: u32,
    pub tp_degree: u32,
    pub pp_degree: u32,
    pub old_to_new: HashMap<u16, u16>,
}
```

### NCCL Communicator Rebuild

```rust
/// Destroy old communicator and create a new one with surviving ranks.
pub fn rebuild_nccl_communicator(
    remap: &RankRemap,
    my_old_rank: u16,
) -> Result<NcclCommunicator, NcclError> {
    let my_new_rank = remap.old_to_new.get(&my_old_rank)
        .ok_or(NcclError::RankEliminated)?;

    // 1. Destroy old communicator
    // ncclCommDestroy(old_comm)

    // 2. Generate new unique ID (rank 0 broadcasts via out-of-band channel)
    // ncclGetUniqueId(&unique_id) on rank 0
    // broadcast unique_id via UDP to all surviving ranks

    // 3. Create new communicator
    // ncclCommInitRank(&new_comm, remap.new_world_size, unique_id, my_new_rank)

    // 4. Rebuild sub-communicators for TP groups
    // Each TP group creates its own ncclCommSplit

    Ok(NcclCommunicator {
        world_comm: new_comm,
        dp_comm: dp_sub_comm,
        tp_comm: tp_sub_comm,
        rank: *my_new_rank,
        world_size: remap.new_world_size,
    })
}
```

### Learning Rate Adjustment

When DP degree changes, the effective batch size changes. If the training uses linear LR scaling (common for large-batch training), the LR must be adjusted:

```rust
pub fn adjust_lr_for_elastic(
    base_lr: f64,
    original_dp_degree: u32,
    new_dp_degree: u32,
    scaling_rule: LrScalingRule,
) -> f64 {
    match scaling_rule {
        LrScalingRule::Linear => base_lr * (new_dp_degree as f64 / original_dp_degree as f64),
        LrScalingRule::Sqrt => base_lr * ((new_dp_degree as f64).sqrt() / (original_dp_degree as f64).sqrt()),
        LrScalingRule::None => base_lr,  // user handles LR adjustment in callback
    }
}

pub enum LrScalingRule {
    Linear,     // lr *= new_dp / original_dp
    Sqrt,       // lr *= sqrt(new_dp / original_dp)
    None,       // no automatic adjustment
}
```

---

## Section 6: Resume Protocol

### Full Recovery Sequence

```rust
pub fn execute_recovery(
    coordinator: &mut Coordinator,
    failure: FailureEvent,
    checkpoint_storage: &dyn CheckpointStorage,
) -> Result<ResumeCommand, RecoveryError> {
    // Phase 1: Halt training on all ranks
    coordinator.broadcast(Command::Halt)?;
    coordinator.wait_all_ack(Duration::from_secs(5))?;

    // Phase 2: Identify latest consistent checkpoint
    let latest_step = find_latest_consistent_checkpoint(
        checkpoint_storage,
        &coordinator.surviving_ranks(),
    )?;

    // Phase 3: Compute new topology
    let remap = compute_rank_remap(
        coordinator.world_size,
        &coordinator.dead_ranks(),
        coordinator.tp_degree,
        coordinator.pp_degree,
    );

    // Phase 4: Verify minimum world size
    if remap.new_dp_degree < coordinator.min_world_size {
        return Err(RecoveryError::BelowMinWorldSize {
            current: remap.new_dp_degree,
            minimum: coordinator.min_world_size,
        });
    }

    // Phase 5: Broadcast recovery plan
    coordinator.broadcast(Command::Recover {
        remap: remap.clone(),
        checkpoint_step: latest_step,
    })?;

    Ok(ResumeCommand {
        step: latest_step,
        remap,
    })
}

/// Find the latest step where ALL surviving ranks have a valid checkpoint.
fn find_latest_consistent_checkpoint(
    storage: &dyn CheckpointStorage,
    surviving_ranks: &[u16],
) -> Result<u64, RecoveryError> {
    // List all checkpoints, find the maximum step where every
    // surviving rank has a checkpoint file
    let mut step_counts: HashMap<u64, usize> = HashMap::new();
    for rank in surviving_ranks {
        let files = storage.list(&format!("rank_{:04}_step_", rank))?;
        for file in files {
            if let Some(step) = parse_step_from_filename(&file) {
                *step_counts.entry(step).or_insert(0) += 1;
            }
        }
    }
    step_counts.iter()
        .filter(|(_, count)| **count == surviving_ranks.len())
        .map(|(step, _)| *step)
        .max()
        .ok_or(RecoveryError::NoConsistentCheckpoint)
}
```

### Per-Rank Recovery

```rust
/// Executed by each surviving rank after receiving the recovery plan.
pub fn recover_rank(
    my_rank: u16,
    resume: &ResumeCommand,
    storage: &dyn CheckpointStorage,
    model: &mut Model,
    optimizer: &mut Optimizer,
) -> Result<(), RecoveryError> {
    // 1. Load checkpoint
    let ckpt_path = format!("rank_{:04}_step_{:08}.nslm", my_rank, resume.step);
    let ckpt_data = storage.read(&ckpt_path)?;
    load_checkpoint_into(model, optimizer, &ckpt_data)?;

    // 2. Rebuild NCCL communicator
    let new_comm = rebuild_nccl_communicator(&resume.remap, my_rank)?;

    // 3. Adjust learning rate
    let new_lr = adjust_lr_for_elastic(
        optimizer.base_lr(),
        resume.remap.tp_degree * resume.remap.pp_degree * original_dp,
        resume.remap.new_world_size,
        LrScalingRule::Linear,
    );
    optimizer.set_lr(new_lr);

    // 4. Reset data loader to checkpoint position
    // (data_position is stored in the checkpoint metadata)

    // 5. Signal ready to coordinator
    Ok(())
}
```

---

## Section 7: Codegen Changes

### Train Block Emission with Fault Tolerance

**Module:** `crates/nsl-codegen/src/stmt.rs`

When the `@fault_tolerant` decorator is present on a `train` block, the compiler emits additional infrastructure around the training loop:

```rust
fn compile_fault_tolerant_train_block(
    &mut self,
    train: &TrainBlock,
    ft_config: &FaultToleranceConfig,
) {
    // 1. Emit heartbeat initialization
    self.emit_call("nsl_heartbeat_start", &[
        self.const_str(&ft_config.coordinator_addr),
        self.const_u16(self.rank),
        self.const_u64(ft_config.heartbeat_interval_ms),
    ]);

    // 2. Emit micro-checkpointer initialization
    self.emit_call("nsl_checkpoint_init", &[
        self.const_str(&ft_config.checkpoint_storage),
        self.const_u64(ft_config.checkpoint_interval),
        self.const_u8(ft_config.compression as u8),
    ]);

    // 3. Emit recovery check at program start
    //    (check if we're resuming from a failure)
    let resume_step = self.emit_call("nsl_check_resume", &[
        self.const_str(&ft_config.checkpoint_storage),
        self.const_u16(self.rank),
    ]);

    // 4. Normal training loop (from M14/M43 codegen)
    //    but with injected checkpoint calls
    self.compile_train_loop_with_checkpoints(train, ft_config);

    // 5. Emit heartbeat stop at program end
    self.emit_call("nsl_heartbeat_stop", &[]);
}
```

### Injected Checkpoint Calls

Inside the training loop, after each optimizer step:

```rust
fn compile_train_step_with_checkpoint(
    &mut self,
    step_var: Value,
    ft_config: &FaultToleranceConfig,
) {
    // ... normal forward/backward/optimizer step ...

    // Inject: maybe_checkpoint(step, rank, &model_tensors, metadata)
    self.emit_call("nsl_maybe_checkpoint", &[
        step_var,
        self.const_u16(self.rank),
        self.model_tensors_ptr,
        self.optimizer_state_ptr,
    ]);

    // Inject: check for recovery signal from coordinator
    let should_recover = self.emit_call("nsl_check_recovery_signal", &[]);
    // If should_recover != 0, branch to recovery handler
    self.emit_branch_nonzero(should_recover, self.recovery_block);
}
```

### Semantic Validation

**Module:** `crates/nsl-semantic/src/checker/train.rs`

The semantic checker validates `@fault_tolerant` train blocks:

```rust
fn check_fault_tolerant_train(
    &mut self,
    train: &TrainBlock,
    decorators: &[Decorator],
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();

    // 1. Must have 'distribute' field
    if train.distribute.is_none() {
        diags.push(Diagnostic::error(
            "@fault_tolerant requires 'distribute' field",
            train.span,
        ));
    }

    // 2. If distribute has elastic DP, validate TP/PP are not elastic
    if let Some(dist) = &train.distribute {
        if dist.tp_elastic || dist.pp_elastic {
            diags.push(Diagnostic::error(
                "only DP dimension can be elastic (TP/PP re-sharding not supported)",
                dist.span,
            ));
        }
    }

    // 3. checkpoint_storage must be set
    if train.checkpoint_storage.is_none() {
        diags.push(Diagnostic::warning(
            "@fault_tolerant without checkpoint_storage defaults to './checkpoints/'",
            train.span,
        ));
    }

    // 4. Callbacks must have correct signatures
    for cb in &train.callbacks {
        match cb.name.as_str() {
            "on_node_failure" => self.check_callback_sig(cb, &["int", "int"], &diags),
            "on_recovery_complete" => self.check_callback_sig(cb, &["int", "int"], &diags),
            "on_checkpoint_written" => self.check_callback_sig(cb, &["str", "int"], &diags),
            "on_abort" => self.check_callback_sig(cb, &["str", "int"], &diags),
            _ => diags.push(Diagnostic::error(
                format!("unknown fault tolerance callback: {}", cb.name),
                cb.span,
            )),
        }
    }

    diags
}
```

### CompileOptions Extension

```rust
pub struct CompileOptions {
    pub no_autotune: bool,
    pub autotune_fresh: bool,
    pub world_size: usize,
    pub fusion_report: bool,
    pub trace: bool,
    pub deterministic: bool,
    pub fault_tolerant: bool,           // NEW
    pub checkpoint_storage: Option<String>, // NEW
}
```

### AST Extension

**Module:** `crates/nsl-ast/src/lib.rs`

```rust
pub struct FaultToleranceConfig {
    pub heartbeat_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub checkpoint_interval: u64,
    pub checkpoint_storage: String,
    pub max_failures: u32,
    pub min_world_size: u32,
    pub max_recovery_time_ms: u64,
    pub checkpoint_compression: CompressionKind,
    pub lr_scaling_rule: LrScalingRule,
    pub span: Span,
}

pub struct ElasticDistribute {
    pub dp: ElasticDim,
    pub tp: u32,
    pub pp: u32,
    pub span: Span,
}

pub enum ElasticDim {
    Fixed(u32),
    Elastic { min: Option<u32>, max: Option<u32> },
}
```

---

## Section 8: Runtime FFI Functions

### New FFI Surface

```rust
// crates/nsl-runtime/src/fault_tolerance.rs

/// Start the heartbeat sender thread.
#[no_mangle]
pub extern "C" fn nsl_heartbeat_start(
    coordinator_addr: *const c_char,
    rank: u16,
    interval_ms: u64,
) { ... }

/// Stop the heartbeat sender thread.
#[no_mangle]
pub extern "C" fn nsl_heartbeat_stop() { ... }

/// Initialize the micro-checkpointer.
#[no_mangle]
pub extern "C" fn nsl_checkpoint_init(
    storage_path: *const c_char,
    interval: u64,
    compression: u8,
) { ... }

/// Maybe write a checkpoint (checks step % interval internally).
#[no_mangle]
pub extern "C" fn nsl_maybe_checkpoint(
    step: u64,
    rank: u16,
    model_tensors: *const *const NslTensor,
    num_tensors: u32,
    optimizer_state: *const c_void,
) { ... }

/// Check if a recovery signal has been received from the coordinator.
/// Returns 0 for continue, non-zero for recovery needed.
#[no_mangle]
pub extern "C" fn nsl_check_recovery_signal() -> u32 { ... }

/// Check if we're resuming from a previous failure.
/// Returns the step to resume from, or 0 if fresh start.
#[no_mangle]
pub extern "C" fn nsl_check_resume(
    storage_path: *const c_char,
    rank: u16,
) -> u64 { ... }

/// Perform elastic resize: rebuild NCCL communicators with new world size.
#[no_mangle]
pub extern "C" fn nsl_elastic_resize(
    new_world_size: u32,
    new_rank: u16,
    new_dp_degree: u32,
) -> i32 { ... }

/// Trigger an async checkpoint immediately (used before recovery).
#[no_mangle]
pub extern "C" fn nsl_checkpoint_async(
    step: u64,
    rank: u16,
    model_tensors: *const *const NslTensor,
    num_tensors: u32,
) { ... }

/// Load a checkpoint from storage into model/optimizer tensors.
#[no_mangle]
pub extern "C" fn nsl_checkpoint_load(
    storage_path: *const c_char,
    rank: u16,
    step: u64,
    model_tensors: *mut *mut NslTensor,
    num_tensors: u32,
) -> i32 { ... }
```

---

## Section 9: File Changes

### New Files

| File | Description |
|------|-------------|
| `crates/nsl-runtime/src/heartbeat.rs` | UDP heartbeat sender/receiver |
| `crates/nsl-runtime/src/micro_checkpoint.rs` | Async micro-checkpoint writer/reader |
| `crates/nsl-runtime/src/fault_tolerance.rs` | FFI entry points, coordinator logic |
| `crates/nsl-runtime/src/elastic.rs` | Rank remapping, NCCL rebuild |
| `crates/nsl-runtime/src/checkpoint_storage.rs` | Storage backend trait + impls (local, S3) |
| `crates/nsl-semantic/src/fault_tolerance.rs` | Semantic validation for @fault_tolerant |
| `crates/nsl-ast/src/fault_tolerance.rs` | AST nodes for fault tolerance config |
| `crates/nsl-codegen/src/fault_tolerance.rs` | Codegen for heartbeat/checkpoint injection |

### Modified Files

| File | Change |
|------|--------|
| `crates/nsl-parser/src/lib.rs` | Parse `@fault_tolerant` decorator, `distribute: "dp=elastic"` |
| `crates/nsl-codegen/src/stmt.rs` | Call `compile_fault_tolerant_train_block` when decorator present |
| `crates/nsl-codegen/src/context.rs` | Add `fault_tolerant` to `CompileOptions` |
| `crates/nsl-cli/src/main.rs` | Wire `--fault-tolerant` CLI flag |
| `crates/nsl-runtime/src/lib.rs` | Export new FFI functions |
| `crates/nsl-codegen/src/linker.rs` | Link S3/networking libraries when fault tolerance enabled |

---

## Section 10: Testing Strategy

### Unit Tests

**`crates/nsl-runtime/src/heartbeat.rs`:**
- HeartbeatPacket serialization/deserialization round-trip
- HeartbeatReceiver detects timeout after `timeout_ms` of silence
- HeartbeatReceiver transitions: Healthy -> Suspect -> Probing -> Dead
- Probe response clears SUSPECT status
- Multiple simultaneous failures are tracked independently

**`crates/nsl-runtime/src/micro_checkpoint.rs`:**
- Checkpoint write/read round-trip preserves tensor data exactly
- LZ4 compression produces identical data on decompress
- Background writer thread handles concurrent writes without data races
- Atomic file write (tmp + rename) leaves no partial files on crash

**`crates/nsl-runtime/src/elastic.rs`:**
- `compute_rank_remap` with 1 dead rank in dp=4/tp=2/pp=1: new dp=3
- `compute_rank_remap` with 2 dead ranks from same DP group: new dp=3 (only 1 group lost)
- `compute_rank_remap` with dead rank in TP group: entire DP replica removed
- LR adjustment with linear scaling: `lr * (new_dp / old_dp)`

**`crates/nsl-semantic/src/fault_tolerance.rs`:**
- `@fault_tolerant` without `distribute`: compile error
- `distribute: "dp=64, tp=elastic"`: compile error (TP not elastic)
- `distribute: "dp=elastic, tp=4, pp=4"`: passes
- Missing `checkpoint_storage`: warning (default used)
- Invalid callback name: compile error
- Callback with wrong argument types: compile error

### Integration Tests

- Two-rank training with simulated failure: rank 1 stops sending heartbeats, rank 0 detects failure, training continues on rank 0 alone
- Checkpoint write at step 100, simulate crash at step 150, resume from step 100 and verify loss matches continued training
- Elastic resize: start with dp=4, remove dp=2, verify all-reduce still produces correct gradients with dp=2

### E2E Tests

- **`examples/m58_fault_tolerant_basic.nsl`** — `@fault_tolerant` train block with checkpoint_interval=5, verify checkpoints written
- **`examples/m58_elastic_resize.nsl`** — Start with dp=4, kill 1 rank at step 10, verify training continues with dp=3
- **`examples/m58_checkpoint_resume.nsl`** — Write checkpoint, restart fresh, verify resume from correct step
- **`examples/m58_max_failures.nsl`** — Verify training aborts when `max_failures` is exceeded

---

## Section 11: Deliverables

- `@fault_tolerant` decorator on `train` blocks enables all resilience features
- Out-of-band UDP heartbeat monitor with configurable timeout (default 30s)
- Three-phase failure detection: timeout -> suspect -> probe -> confirmed dead
- Async micro-checkpointing every N steps with LZ4/zstd compression
- Elastic DP dimension: `distribute: "dp=elastic, tp=4, pp=4"`
- Automatic NCCL communicator rebuild after node failure
- Learning rate adjustment for changed effective batch size
- Resume protocol: find latest consistent checkpoint, reload, continue
- Callbacks for failure/recovery/checkpoint/abort events
- Local and S3 checkpoint storage backends
- Recovery time target: < 60 seconds (vs. 5-15 minutes for full restart)

## Not in Scope

- Elastic TP/PP dimensions (requires full weight re-sharding — future work)
- Preemptive migration (moving work away from a node before it fails)
- Hot standby nodes (spare GPUs waiting to replace failed ones)
- Network partition handling (split-brain scenarios)
- Cross-region checkpoint replication
- Gradient compression during recovery (send full gradients)
- Automatic failure root-cause analysis (handled by M61 cluster debugging)
