# M61: Time-Travel Cluster Debugging — Design Spec

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M61
**Prerequisites:** M45 (Tensor Debugger — single-rank trace format), M58 (Elastic Fault Tolerance — cluster infrastructure)
**Dependencies:** M59 (Topology — rank-to-physical-location mapping for trace visualization)

---

## Overview

When loss spikes to NaN at iteration 450,000 on a 100,000-GPU training run, identify exactly which GPU, which layer, and which operation caused the overflow — across the entire cluster. This milestone extends M45's single-rank tensor debugger to cluster scale with distributed tracing, global clock synchronization, cross-rank anomaly detection, and a unified timeline viewer that merges traces from all ranks.

**Key design:** Each rank records a per-rank binary trace file in M45's format, extended with a global clock offset for cross-rank time alignment. A coordinator process periodically runs anomaly checks (NaN/Inf detection, gradient magnitude tracking) across all ranks via lightweight stat aggregation. On anomaly detection, all ranks simultaneously dump their full state. The `nsl debug --cluster` command merges per-rank traces into a unified timeline, pinpointing the first rank and operation where the anomaly originated.

**Why this is different from W&B / TensorBoard:** These tools log scalar metrics (loss, learning rate) at step granularity. They cannot tell you which operation within which rank produced a NaN — only that the loss became NaN at step N. NSL's cluster debugger operates at individual tensor operation granularity across all ranks simultaneously, with sub-microsecond time alignment.

**Why this requires M45 + M58:** M45 provides the per-rank binary trace format and the NaN sentinel detection. M58 provides the out-of-band heartbeat channel (reused for anomaly broadcast) and the coordinator infrastructure (reused for trace synchronization). Without these, cluster debugging would require building both from scratch.

---

## Section 1: Language Surface

### CLI Flags

```
# Record distributed trace during training
nsl run --cluster-trace model.nsl

# Record with custom output directory
nsl run --cluster-trace --trace-dir /shared/traces/run-001/ model.nsl

# Record with anomaly detection sensitivity
nsl run --cluster-trace --anomaly-threshold 1e3 model.nsl

# Analyze cluster traces
nsl debug --cluster /shared/traces/run-001/

# Find the first NaN across all ranks
nsl debug --cluster --find-nan /shared/traces/run-001/

# Compare gradient magnitudes across ranks
nsl debug --cluster --gradient-history /shared/traces/run-001/

# Verify all-reduce consistency across ranks
nsl debug --cluster --comm-audit /shared/traces/run-001/

# Export unified timeline to Chrome tracing JSON
nsl debug --cluster --export-chrome /shared/traces/run-001/ --output timeline.json

# Filter by rank range
nsl debug --cluster --ranks 0-7 /shared/traces/run-001/

# Filter by step range
nsl debug --cluster --steps 449990-450010 /shared/traces/run-001/
```

### Annotations

```
@cluster_trace_breakpoint
fn suspicious_attention(q: Tensor<[B, H, S, D], f32>,
                        k: Tensor<[B, H, S, D], f32>) -> Tensor<[B, H, S, D], f32>:
    # When ANY rank hits this function, ALL ranks dump their current state.
    let scores = matmul(q, k.transpose(-2, -1)) / sqrt(float(D))
    return softmax(scores)

@trace_gradient(layer="attention.q_proj")
fn attention_q(x: Tensor<[B, S, D], f32>,
               w: Tensor<[D, D], f32>) -> Tensor<[B, S, D], f32>:
    # Track gradient magnitudes for this function across all ranks.
    return matmul(x, w)
```

### Configuration in train Block

```
@fault_tolerant
train ClusterTraced:
    model: LLaMA70B
    data: WebCorpus
    optimizer: AdamW(lr=3e-4)
    distribute: "dp=32, tp=8, pp=4"

    # Cluster debugging configuration
    cluster_trace: true
    trace_dir: "/shared/nfs/traces/run-001/"
    anomaly_check_interval: 10         # check every 10 steps
    anomaly_threshold: 1e3             # gradient norm > 1e3 triggers alert
    nan_freeze: true                   # freeze all ranks on first NaN
    gradient_tracking: ["*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj"]
    max_trace_steps: 1000              # only keep last 1000 steps (ring buffer)
```

---

## Section 2: Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Coordinator (Rank 0)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Clock Sync   │  │  Anomaly     │  │  Freeze           │  │
│  │  Broadcaster  │  │  Aggregator  │  │  Coordinator      │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────────┘  │
│         │                 │                   │              │
└─────────┼─────────────────┼───────────────────┼──────────────┘
          │                 │                   │
    UDP broadcast     UDP collect          UDP broadcast
          │                 │                   │
┌─────────┼─────────────────┼───────────────────┼──────────────┐
│         │    Rank N       │                   │              │
│  ┌──────┴───────┐  ┌─────┴────────┐  ┌──────┴───────────┐  │
│  │ ClockOffset   │  │ StatsReporter│  │ FreezeHandler    │  │
│  │ Tracker       │  │ (periodic)   │  │ (dump on signal) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘  │
│         │                 │                  │               │
│  ┌──────┴─────────────────┴──────────────────┴───────────┐  │
│  │                   TraceRecorder (M45)                   │  │
│  │   extended with: global timestamps, grad tracking,     │  │
│  │                  ring buffer mode, comm audit entries   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Global Clock Synchronization

Each rank maintains a clock offset relative to rank 0. The coordinator broadcasts its timestamp periodically, and each rank computes the offset:

```rust
pub struct ClockSync {
    /// Offset from local clock to coordinator clock (nanoseconds).
    /// global_time = local_time + offset
    pub offset_ns: i64,
    /// Estimated precision of the offset (half round-trip time)
    pub precision_ns: u64,
    /// Last sync timestamp
    pub last_sync: Instant,
}

impl ClockSync {
    /// Compute clock offset via NTP-style exchange.
    ///
    /// Protocol:
    /// 1. Rank N sends REQUEST with local timestamp T1
    /// 2. Coordinator receives at its local time T2, sends RESPONSE(T2, T3)
    /// 3. Rank N receives at local time T4
    /// 4. offset = ((T2 - T1) + (T3 - T4)) / 2
    /// 5. precision = (T4 - T1) / 2
    pub fn sync(&mut self, socket: &UdpSocket, coordinator: &SocketAddr) {
        let t1 = self.now_ns();
        socket.send_to(&ClockSyncRequest { t1 }.to_bytes(), coordinator).unwrap();

        let mut buf = [0u8; 32];
        let (_, _) = socket.recv_from(&mut buf).unwrap();
        let t4 = self.now_ns();

        let resp = ClockSyncResponse::from_bytes(&buf);
        let t2 = resp.t2;
        let t3 = resp.t3;

        self.offset_ns = ((t2 as i64 - t1 as i64) + (t3 as i64 - t4 as i64)) / 2;
        self.precision_ns = ((t4 - t1) / 2) as u64;
        self.last_sync = Instant::now();
    }

    /// Convert a local timestamp to global (coordinator-aligned) time.
    pub fn to_global(&self, local_ns: u64) -> u64 {
        (local_ns as i64 + self.offset_ns) as u64
    }

    fn now_ns(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}
```

---

## Section 3: Extended Trace Format

### Cluster Trace Header

Per-rank trace files extend M45's `NSLT` format with cluster metadata:

```
Offset  Size  Field
0       4     Magic: "NSCL" (0x4E53434C) — cluster trace
4       4     Version: 1 (u32 LE)
8       8     Global timestamp: coordinator-aligned epoch nanos (u64 LE)
16      2     Rank: this rank's global rank (u16 LE)
18      2     World size: total ranks (u16 LE)
20      4     TP degree (u32 LE)
24      4     PP degree (u32 LE)
28      4     DP degree (u32 LE)
32      8     Clock offset_ns (i64 LE)
40      8     Clock precision_ns (u64 LE)
48      4     Num ops (u32 LE)
52      4     Num gradient entries (u32 LE)
56      4     Num comm audit entries (u32 LE)
60      4     Reserved
64      ..    Op entries (TraceEntry from M45, with global timestamps)
          ..    Gradient history entries
          ..    Communication audit entries
```

### Extended TraceEntry

M45's TraceEntry is extended with two fields:

```rust
#[repr(C, packed)]
pub struct ClusterTraceEntry {
    // --- M45 TraceEntry fields (124 bytes) ---
    pub op_id: u32,
    pub op_type: u16,
    pub flags: u16,        // bit 0: has_nan, bit 1: has_inf, bit 2: is_breakpoint
                           // bit 3: is_comm_op (NEW), bit 4: grad_tracked (NEW)
    pub timestamp_ns: u64, // NOW: global (coordinator-aligned) timestamp
    pub in0_ndim: u8,
    pub in0_dtype: u8,
    pub in0_device: u8,
    pub _pad0: u8,
    pub in0_shape: [u32; 4],
    pub in0_min: f32,
    pub in0_max: f32,
    pub in0_mean: f32,
    pub in0_std: f32,
    pub in1_ndim: u8,
    pub in1_dtype: u8,
    pub in1_device: u8,
    pub _pad1: u8,
    pub in1_shape: [u32; 4],
    pub in1_min: f32,
    pub in1_max: f32,
    pub in1_mean: f32,
    pub in1_std: f32,
    pub out_ndim: u8,
    pub out_dtype: u8,
    pub out_device: u8,
    pub _pad2: u8,
    pub out_shape: [u32; 4],
    pub out_min: f32,
    pub out_max: f32,
    pub out_mean: f32,
    pub out_std: f32,

    // --- Cluster extensions (16 bytes) ---
    pub training_step: u64,    // 8 bytes — which training step this op belongs to
    pub rank: u16,             // 2 bytes — redundant with header, useful for merged traces
    pub layer_id: u16,         // 2 bytes — semantic layer index (for gradient tracking)
    pub comm_op_id: u16,       // 2 bytes — links to CommAuditEntry (0xFFFF if not a comm op)
    pub _pad3: u16,            // 2 bytes — alignment
}
// Total: 124 + 16 = 140 bytes per entry
```

### Gradient History Entry

```rust
#[repr(C, packed)]
pub struct GradientHistoryEntry {
    pub step: u64,              // 8 bytes — training step
    pub rank: u16,              // 2 bytes
    pub layer_id: u16,          // 2 bytes
    pub grad_norm: f32,         // 4 bytes — L2 norm of gradient tensor
    pub grad_min: f32,          // 4 bytes
    pub grad_max: f32,          // 4 bytes
    pub grad_mean: f32,         // 4 bytes
    pub grad_std: f32,          // 4 bytes
    pub param_norm: f32,        // 4 bytes — L2 norm of the parameter tensor
    pub update_ratio: f32,      // 4 bytes — grad_norm / param_norm (gradient-to-weight ratio)
    pub _pad: u32,              // 4 bytes — alignment
}
// Total: 44 bytes per entry
```

### Communication Audit Entry

```rust
#[repr(C, packed)]
pub struct CommAuditEntry {
    pub comm_op_id: u16,        // 2 bytes — unique ID for this collective
    pub comm_type: u8,          // 1 byte  — AllReduce=0, AllGather=1, ReduceScatter=2,
                                //            Send=3, Recv=4, Broadcast=5
    pub comm_group: u8,         // 1 byte  — TP=0, PP=1, DP=2, Ring=3
    pub step: u64,              // 8 bytes — training step
    pub rank: u16,              // 2 bytes
    pub partner_rank: u16,      // 2 bytes — for send/recv (0xFFFF for collectives)
    pub input_hash: u64,        // 8 bytes — xxHash of input tensor data
    pub output_hash: u64,       // 8 bytes — xxHash of output tensor data
    pub byte_count: u64,        // 8 bytes — bytes transferred
    pub duration_ns: u64,       // 8 bytes — wall-clock time for the collective
    pub bandwidth_gbps: f32,    // 4 bytes — achieved bandwidth
    pub _pad: u32,              // 4 bytes — alignment
}
// Total: 56 bytes per entry
```

---

## Section 4: Cross-Rank NaN Detection

### Anomaly Aggregation Protocol

Every `anomaly_check_interval` steps, each rank sends a compact stats summary to the coordinator:

```rust
#[repr(C, packed)]
pub struct RankStepStats {
    pub rank: u16,
    pub step: u64,
    pub has_nan: bool,
    pub has_inf: bool,
    pub max_activation: f32,
    pub min_activation: f32,
    pub loss_value: f32,
    pub max_grad_norm: f32,       // max gradient norm across all tracked layers
    pub num_ops_since_last: u32,
}

impl RankStepStats {
    pub fn is_anomalous(&self, threshold: f32) -> bool {
        self.has_nan
            || self.has_inf
            || self.max_grad_norm > threshold
            || self.loss_value.is_nan()
            || self.loss_value.is_infinite()
    }
}
```

### Coordinator Anomaly Detection

```rust
pub struct AnomalyAggregator {
    pub stats_history: Vec<Vec<RankStepStats>>,  // [step][rank]
    pub threshold: f32,
    pub frozen: bool,
}

impl AnomalyAggregator {
    /// Process stats from all ranks for a given step.
    pub fn process_step(&mut self, stats: Vec<RankStepStats>) -> AnomalyResult {
        let step = stats[0].step;

        // Check for NaN on any rank
        let nan_ranks: Vec<u16> = stats.iter()
            .filter(|s| s.has_nan || s.loss_value.is_nan())
            .map(|s| s.rank)
            .collect();

        if !nan_ranks.is_empty() {
            return AnomalyResult::NanDetected {
                step,
                ranks: nan_ranks,
                first_nan_rank: self.find_first_nan_rank(&stats),
            };
        }

        // Check for gradient explosion
        let exploding_ranks: Vec<(u16, f32)> = stats.iter()
            .filter(|s| s.max_grad_norm > self.threshold)
            .map(|s| (s.rank, s.max_grad_norm))
            .collect();

        if !exploding_ranks.is_empty() {
            return AnomalyResult::GradientExplosion {
                step,
                ranks_and_norms: exploding_ranks,
            };
        }

        // Check for cross-rank divergence (some ranks much higher loss than others)
        let losses: Vec<f32> = stats.iter().map(|s| s.loss_value).collect();
        let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;
        let max_deviation = losses.iter()
            .map(|l| (l - mean_loss).abs())
            .fold(0.0f32, f32::max);

        if max_deviation > mean_loss * 0.5 {
            return AnomalyResult::CrossRankDivergence {
                step,
                mean_loss,
                max_deviation,
            };
        }

        self.stats_history.push(stats);
        AnomalyResult::Healthy
    }

    /// Find which rank had NaN first based on operation timestamps.
    fn find_first_nan_rank(&self, stats: &[RankStepStats]) -> u16 {
        // The rank with the earliest NaN-flagged operation
        // (requires reading the actual trace files for timestamp comparison)
        stats.iter()
            .filter(|s| s.has_nan)
            .min_by_key(|s| s.rank) // fallback: lowest rank
            .map(|s| s.rank)
            .unwrap_or(0)
    }
}

pub enum AnomalyResult {
    Healthy,
    NanDetected {
        step: u64,
        ranks: Vec<u16>,
        first_nan_rank: u16,
    },
    GradientExplosion {
        step: u64,
        ranks_and_norms: Vec<(u16, f32)>,
    },
    CrossRankDivergence {
        step: u64,
        mean_loss: f32,
        max_deviation: f32,
    },
}
```

### Cluster Freeze Protocol

When an anomaly is detected and `nan_freeze: true`, all ranks dump state simultaneously:

```
1. Coordinator detects anomaly at step N
       │
2. Coordinator broadcasts FREEZE(step=N, reason="NaN on rank 42")
       │
3. Each rank receives FREEZE:
   a. Complete current NCCL operation (don't hang)
   b. Flush trace buffer to disk
   c. Dump full tensor state for current step:
      - All model weights
      - All optimizer state (momentum, variance)
      - Current batch data
      - Gradient tensors (before all-reduce)
      - Gradient tensors (after all-reduce)
   d. ACK to coordinator
       │
4. Coordinator waits for all ACKs (timeout: 60s)
       │
5. Coordinator writes freeze manifest:
   {
     "freeze_step": N,
     "trigger_rank": 42,
     "reason": "NaN detected",
     "dump_files": ["rank_0000_freeze.bin", "rank_0001_freeze.bin", ...]
   }
```

```rust
/// Per-rank freeze dump structure.
pub struct FreezeDump {
    pub rank: u16,
    pub step: u64,
    pub reason: String,
    pub model_weights: Vec<NamedTensor>,
    pub optimizer_state: Vec<NamedTensor>,
    pub gradients_pre_allreduce: Vec<NamedTensor>,
    pub gradients_post_allreduce: Vec<NamedTensor>,
    pub current_batch: Vec<NamedTensor>,
    pub trace_tail: Vec<ClusterTraceEntry>,  // last 100 ops
    pub gradient_history: Vec<GradientHistoryEntry>,  // last 100 steps
}

pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: u8,
    pub data: Vec<u8>,  // raw bytes
}
```

---

## Section 5: Gradient Magnitude History

### Per-Rank, Per-Layer Tracking

The `gradient_tracking` config specifies which layers to track:

```
gradient_tracking: ["*.q_proj", "*.k_proj", "*.v_proj", "*.o_proj"]
```

The glob patterns are matched against fully-qualified parameter names. For each matched parameter, the runtime computes gradient statistics after the backward pass (before all-reduce):

```rust
pub struct GradientTracker {
    /// Layer name patterns to track
    patterns: Vec<GlobPattern>,
    /// History buffer (ring buffer of last N steps)
    history: RingBuffer<Vec<GradientHistoryEntry>>,
    /// Max steps to retain
    max_steps: usize,
}

impl GradientTracker {
    /// Called after backward pass, before all-reduce.
    pub fn record_gradients(
        &mut self,
        step: u64,
        rank: u16,
        params: &[(String, &NslTensor)],  // (name, gradient_tensor)
    ) {
        let mut step_entries = Vec::new();

        for (name, grad_tensor) in params {
            if !self.matches_any_pattern(name) {
                continue;
            }

            let stats = compute_tensor_stats(grad_tensor);
            let param_norm = compute_l2_norm(grad_tensor);

            step_entries.push(GradientHistoryEntry {
                step,
                rank,
                layer_id: self.layer_name_to_id(name),
                grad_norm: stats.l2_norm,
                grad_min: stats.min,
                grad_max: stats.max,
                grad_mean: stats.mean,
                grad_std: stats.std,
                param_norm,
                update_ratio: stats.l2_norm / param_norm.max(1e-12),
                _pad: 0,
            });
        }

        self.history.push(step_entries);
    }
}
```

---

## Section 6: Communication Audit

### All-Reduce Consistency Verification

After each all-reduce, audit that all ranks received the same result (detect bit-flip corruptions or NCCL bugs):

```rust
pub struct CommAuditor {
    /// Hash function for tensor data
    hasher: xxhash_rust::xxh3::Xxh3,
    /// Audit entries for current step
    current_step_entries: Vec<CommAuditEntry>,
}

impl CommAuditor {
    /// Record an all-reduce operation.
    pub fn audit_allreduce(
        &mut self,
        rank: u16,
        step: u64,
        input: &NslTensor,
        output: &NslTensor,
        group: CommGroup,
        duration_ns: u64,
    ) -> CommAuditEntry {
        let input_hash = self.hash_tensor(input);
        let output_hash = self.hash_tensor(output);
        let byte_count = tensor_byte_size(output);
        let bandwidth_gbps = (byte_count as f64 * 8.0) / (duration_ns as f64);

        CommAuditEntry {
            comm_op_id: self.next_op_id(),
            comm_type: 0, // AllReduce
            comm_group: group as u8,
            step,
            rank,
            partner_rank: 0xFFFF,
            input_hash,
            output_hash,
            byte_count,
            duration_ns,
            bandwidth_gbps: bandwidth_gbps as f32,
            _pad: 0,
        }
    }

    fn hash_tensor(&mut self, tensor: &NslTensor) -> u64 {
        let data = unsafe {
            std::slice::from_raw_parts(
                tensor.data as *const u8,
                tensor_byte_size(tensor) as usize,
            )
        };
        xxhash_rust::xxh3::xxh3_64(data)
    }
}

/// Verify all-reduce consistency across ranks (run by coordinator or offline analysis).
pub fn verify_allreduce_consistency(
    entries: &[CommAuditEntry],
    world_size: u32,
) -> Vec<ConsistencyViolation> {
    let mut violations = Vec::new();

    // Group entries by (step, comm_op_id, comm_group)
    let mut groups: HashMap<(u64, u16, u8), Vec<&CommAuditEntry>> = HashMap::new();
    for entry in entries {
        groups.entry((entry.step, entry.comm_op_id, entry.comm_group))
            .or_default()
            .push(entry);
    }

    for ((step, op_id, group), group_entries) in &groups {
        // All ranks in the same all-reduce should have the same output hash
        let output_hashes: HashSet<u64> = group_entries.iter()
            .map(|e| e.output_hash)
            .collect();

        if output_hashes.len() > 1 {
            violations.push(ConsistencyViolation {
                step: *step,
                comm_op_id: *op_id,
                comm_group: *group,
                distinct_hashes: output_hashes.len(),
                ranks_per_hash: output_hashes.iter().map(|h| {
                    let ranks: Vec<u16> = group_entries.iter()
                        .filter(|e| e.output_hash == *h)
                        .map(|e| e.rank)
                        .collect();
                    (*h, ranks)
                }).collect(),
            });
        }
    }

    violations
}

pub struct ConsistencyViolation {
    pub step: u64,
    pub comm_op_id: u16,
    pub comm_group: u8,
    pub distinct_hashes: usize,
    pub ranks_per_hash: Vec<(u64, Vec<u16>)>,
}
```

---

## Section 7: Cluster Debug CLI

### Module: `crates/nsl-cli/src/cluster_debug.rs` (NEW)

```rust
pub struct ClusterDebugger {
    trace_dir: PathBuf,
    traces: Vec<RankTrace>,
    merged_timeline: Vec<ClusterTraceEntry>,
    gradient_history: Vec<Vec<GradientHistoryEntry>>,
    comm_audit: Vec<CommAuditEntry>,
}

pub struct RankTrace {
    pub rank: u16,
    pub header: ClusterTraceHeader,
    pub entries: Vec<ClusterTraceEntry>,
    pub gradients: Vec<GradientHistoryEntry>,
    pub comm_entries: Vec<CommAuditEntry>,
}

impl ClusterDebugger {
    /// Load all rank traces from a directory.
    pub fn load(trace_dir: &Path) -> Result<Self, DebugError> {
        let mut traces = Vec::new();

        // Find all rank trace files: rank_NNNN.nscl
        for entry in std::fs::read_dir(trace_dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "nscl") {
                let trace = RankTrace::load(&path)?;
                traces.push(trace);
            }
        }

        traces.sort_by_key(|t| t.rank);

        // Merge all traces into a unified timeline sorted by global timestamp
        let mut merged: Vec<ClusterTraceEntry> = Vec::new();
        for trace in &traces {
            merged.extend_from_slice(&trace.entries);
        }
        merged.sort_by_key(|e| e.timestamp_ns);

        Ok(Self {
            trace_dir: trace_dir.to_path_buf(),
            traces,
            merged_timeline: merged,
            gradient_history: Vec::new(),
            comm_audit: Vec::new(),
        })
    }

    /// Find the first NaN across all ranks, returning rank and op.
    pub fn find_first_nan(&self) -> Option<NanFinding> {
        self.merged_timeline.iter()
            .find(|e| e.flags & 0x01 != 0)
            .map(|e| NanFinding {
                rank: e.rank,
                step: e.training_step,
                op_id: e.op_id,
                op_type: e.op_type,
                timestamp_ns: e.timestamp_ns,
                input_stats: InputStats {
                    in0_min: e.in0_min,
                    in0_max: e.in0_max,
                    in1_min: e.in1_min,
                    in1_max: e.in1_max,
                },
            })
    }

    /// Trace NaN backward: find the chain of operations that led to NaN.
    pub fn trace_nan_origin(&self, nan_entry: &ClusterTraceEntry) -> Vec<ClusterTraceEntry> {
        let mut chain = vec![nan_entry.clone()];
        let rank = nan_entry.rank;

        // Walk backward through the rank's trace
        let rank_trace = &self.traces[rank as usize];
        let mut current_op = nan_entry.op_id as usize;

        while current_op > 0 {
            current_op -= 1;
            let prev = &rank_trace.entries[current_op];

            // Check if this op's output could have fed into the NaN op
            if prev.out_min.is_nan() || prev.out_max.is_nan()
                || prev.out_min.is_infinite() || prev.out_max.is_infinite()
            {
                chain.push(prev.clone());
            } else {
                // Found a healthy op — the NaN originated in the next op
                break;
            }
        }

        chain.reverse();
        chain
    }

    /// Compare gradient magnitude histories across ranks.
    pub fn gradient_divergence_report(&self) -> Vec<GradientDivergence> {
        let mut divergences = Vec::new();

        // Group gradient entries by (step, layer_id)
        let mut grouped: HashMap<(u64, u16), Vec<&GradientHistoryEntry>> = HashMap::new();
        for trace in &self.traces {
            for entry in &trace.gradients {
                grouped.entry((entry.step, entry.layer_id))
                    .or_default()
                    .push(entry);
            }
        }

        for ((step, layer_id), entries) in &grouped {
            let norms: Vec<f32> = entries.iter().map(|e| e.grad_norm).collect();
            let mean_norm = norms.iter().sum::<f32>() / norms.len() as f32;
            let max_deviation = norms.iter()
                .map(|n| (n - mean_norm).abs())
                .fold(0.0f32, f32::max);

            // Flag if any rank's gradient norm deviates > 2x from mean
            if max_deviation > mean_norm {
                let outlier_ranks: Vec<(u16, f32)> = entries.iter()
                    .filter(|e| (e.grad_norm - mean_norm).abs() > mean_norm)
                    .map(|e| (e.rank, e.grad_norm))
                    .collect();

                divergences.push(GradientDivergence {
                    step: *step,
                    layer_id: *layer_id,
                    mean_norm,
                    max_deviation,
                    outlier_ranks,
                });
            }
        }

        divergences
    }
}

pub struct NanFinding {
    pub rank: u16,
    pub step: u64,
    pub op_id: u32,
    pub op_type: u16,
    pub timestamp_ns: u64,
    pub input_stats: InputStats,
}

pub struct GradientDivergence {
    pub step: u64,
    pub layer_id: u16,
    pub mean_norm: f32,
    pub max_deviation: f32,
    pub outlier_ranks: Vec<(u16, f32)>,
}
```

### TUI Layout for Cluster Debugging

```
┌─ Cluster Trace: /traces/run-001/ ─── 256 ranks, 450000 steps ──────────┐
│                                                                          │
│ ┌─ Timeline (global) ───────────────────────────────────────────────────┐│
│ │ Step 449999  │████████████████████████████████████████████│ 256 ranks ││
│ │ Step 450000  │████████████████████████████████NaN█████████│ 256 ranks ││
│ │ Step 450001  │ (frozen)                                  │           ││
│ └───────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│ ┌─ NaN Origin Chain (Rank 42) ──────────────────────────────────────────┐│
│ │ Op #8421: matmul    [4096,4096] f32  out_max=3.4e+38 (Inf)          ││
│ │ Op #8422: add       [4096,4096] f32  out_max=NaN                     ││
│ │ Op #8423: layernorm [4096,4096] f32  out_max=NaN                     ││
│ │ Op #8424: matmul    [4096,4096] f32  out_max=NaN    ← first visible ││
│ └───────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│ ┌─ Gradient History (Layer: attention.q_proj) ──────────────────────────┐│
│ │ Step 449990: mean_norm=1.23  rank42_norm=1.25  (normal)              ││
│ │ Step 449995: mean_norm=1.31  rank42_norm=4.87  (DIVERGING)           ││
│ │ Step 450000: mean_norm=1.28  rank42_norm=Inf   (EXPLODED)            ││
│ └───────────────────────────────────────────────────────────────────────┘│
│                                                                          │
│ [n] Find NaN  [g] Gradient history  [c] Comm audit  [r] Rank filter     │
│ [s] Step filter  [e] Export Chrome  [f] Freeze dump  [q] Quit           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Chrome Tracing Export

```rust
impl ClusterDebugger {
    pub fn export_chrome(&self, output: &Path, rank_filter: Option<&[u16]>) {
        let mut events = Vec::new();

        for entry in &self.merged_timeline {
            if let Some(filter) = rank_filter {
                if !filter.contains(&entry.rank) {
                    continue;
                }
            }

            events.push(serde_json::json!({
                "name": op_type_name(entry.op_type),
                "cat": if entry.flags & 0x08 != 0 { "comm" } else { "compute" },
                "ph": "X",
                "ts": entry.timestamp_ns / 1000,  // Chrome uses microseconds
                "dur": 1,  // placeholder — actual duration from next op
                "pid": entry.rank,
                "tid": 0,
                "args": {
                    "step": entry.training_step,
                    "op_id": entry.op_id,
                    "out_shape": format_shape(entry.out_shape, entry.out_ndim),
                    "out_min": entry.out_min,
                    "out_max": entry.out_max,
                    "out_mean": entry.out_mean,
                    "has_nan": entry.flags & 0x01 != 0,
                }
            }));
        }

        let trace = serde_json::json!({ "traceEvents": events });
        std::fs::write(output, serde_json::to_string_pretty(&trace).unwrap()).unwrap();
    }
}
```

---

## Section 8: Codegen Changes

### Module: `crates/nsl-codegen/src/cluster_trace.rs` (NEW)

```rust
pub struct ClusterTraceEmitter {
    enabled: bool,
    trace_dir: String,
    anomaly_interval: u32,
    anomaly_threshold: f32,
    gradient_patterns: Vec<String>,
    nan_freeze: bool,
}

impl ClusterTraceEmitter {
    /// Emit cluster trace initialization at program start.
    pub fn emit_init(&self, compiler: &mut Compiler) {
        // 1. Initialize clock sync
        compiler.emit_call("nsl_cluster_trace_init", &[
            compiler.const_str(&self.trace_dir),
            compiler.const_u16(compiler.rank),
            compiler.const_u32(compiler.world_size as u32),
        ]);

        // 2. Perform initial clock sync
        compiler.emit_call("nsl_cluster_trace_sync_clock", &[]);

        // 3. Configure gradient tracking
        for pattern in &self.gradient_patterns {
            compiler.emit_call("nsl_cluster_trace_add_gradient_pattern", &[
                compiler.const_str(pattern),
            ]);
        }
    }

    /// Emit per-step anomaly check (injected into training loop).
    pub fn emit_step_anomaly_check(&self, compiler: &mut Compiler, step_var: Value) {
        // Emit: if step % anomaly_interval == 0 { check_anomaly() }
        let interval = compiler.const_u32(self.anomaly_interval);
        let remainder = compiler.emit_urem(step_var, interval);
        let is_check_step = compiler.emit_icmp_eq(remainder, compiler.const_u32(0));

        compiler.emit_branch_true(is_check_step, |compiler| {
            let result = compiler.emit_call("nsl_cluster_trace_anomaly_check", &[step_var]);
            if self.nan_freeze {
                // If anomaly detected, trigger freeze
                compiler.emit_branch_nonzero(result, |compiler| {
                    compiler.emit_call("nsl_cluster_trace_freeze", &[step_var]);
                });
            }
        });
    }

    /// Emit gradient recording after backward pass.
    pub fn emit_gradient_recording(&self, compiler: &mut Compiler, step_var: Value) {
        compiler.emit_call("nsl_cluster_trace_record_gradients", &[step_var]);
    }

    /// Emit communication audit wrapper around NCCL calls.
    pub fn emit_comm_audit_wrapper(
        &self,
        compiler: &mut Compiler,
        comm_type: u8,
        comm_group: u8,
        input_ptr: Value,
        output_ptr: Value,
        byte_count: Value,
    ) {
        let start = compiler.emit_call("nsl_cluster_trace_comm_start", &[]);
        // ... original NCCL call ...
        compiler.emit_call("nsl_cluster_trace_comm_end", &[
            compiler.const_u8(comm_type),
            compiler.const_u8(comm_group),
            input_ptr,
            output_ptr,
            byte_count,
            start,
        ]);
    }
}
```

### Integration with Existing Codegen

**Module:** `crates/nsl-codegen/src/stmt.rs`

```rust
fn compile_train_block(&mut self, train: &TrainBlock) {
    // ... existing setup ...

    // NEW: Initialize cluster tracing if configured
    if train.cluster_trace {
        let emitter = ClusterTraceEmitter::new(train);
        emitter.emit_init(self);
        self.cluster_trace_emitter = Some(emitter);
    }

    // Inside the training loop:
    // After backward: record gradients
    // After optimizer step: check anomaly
    // Around NCCL calls: wrap with comm audit
}
```

### CompileOptions Extension

```rust
pub struct CompileOptions {
    // ... existing fields ...
    pub cluster_trace: bool,           // NEW: --cluster-trace flag
    pub trace_dir: Option<PathBuf>,    // NEW: --trace-dir
    pub anomaly_threshold: f32,        // NEW: --anomaly-threshold
}
```

---

## Section 9: Runtime FFI Functions

```rust
// crates/nsl-runtime/src/cluster_trace.rs

/// Initialize cluster trace recording for this rank.
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_init(
    trace_dir: *const c_char,
    rank: u16,
    world_size: u32,
) { ... }

/// Synchronize clock with coordinator (NTP-style exchange).
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_sync_clock() { ... }

/// Add a gradient tracking pattern (glob syntax).
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_add_gradient_pattern(
    pattern: *const c_char,
) { ... }

/// Record a tensor operation with cluster-extended metadata.
/// (Extends nsl_trace_record_op from M45)
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_record_op(
    op_type: u16,
    step: u64,
    layer_id: u16,
    num_inputs: u32,
    inputs: *const *const NslTensor,
    output: *const NslTensor,
) -> u32 { ... }

/// Record gradient statistics for tracked layers.
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_record_gradients(
    step: u64,
) { ... }

/// Check for anomalies across all ranks (coordinator aggregation).
/// Returns 0 = healthy, 1 = anomaly detected.
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_anomaly_check(
    step: u64,
) -> u32 { ... }

/// Freeze all ranks: dump full state and halt.
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_freeze(
    step: u64,
) { ... }

/// Record communication start time (returns timestamp).
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_comm_start() -> u64 { ... }

/// Record communication end with audit data.
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_comm_end(
    comm_type: u8,
    comm_group: u8,
    input: *const NslTensor,
    output: *const NslTensor,
    byte_count: u64,
    start_time: u64,
) { ... }

/// Flush all trace data to disk (called at program exit or on freeze).
#[no_mangle]
pub extern "C" fn nsl_cluster_trace_flush() { ... }
```

---

## Section 10: File Changes

### New Files

| File | Description |
|------|-------------|
| `crates/nsl-runtime/src/cluster_trace.rs` | FFI entry points, clock sync, anomaly aggregation |
| `crates/nsl-runtime/src/gradient_tracker.rs` | Per-layer gradient magnitude tracking |
| `crates/nsl-runtime/src/comm_auditor.rs` | Communication audit and consistency verification |
| `crates/nsl-runtime/src/freeze_dump.rs` | Full-state dump on anomaly freeze |
| `crates/nsl-codegen/src/cluster_trace.rs` | Cluster trace codegen (instrumentation emission) |
| `crates/nsl-cli/src/cluster_debug.rs` | `nsl debug --cluster` TUI and analysis |
| `crates/nsl-semantic/src/cluster_trace.rs` | Semantic validation for cluster trace config |

### Modified Files

| File | Change |
|------|--------|
| `crates/nsl-parser/src/lib.rs` | Parse `cluster_trace`, `gradient_tracking`, `nan_freeze` in train block |
| `crates/nsl-ast/src/lib.rs` | Add cluster trace config fields to TrainBlock |
| `crates/nsl-codegen/src/stmt.rs` | Integrate ClusterTraceEmitter into train block |
| `crates/nsl-codegen/src/context.rs` | Add `cluster_trace` to CompileOptions |
| `crates/nsl-codegen/src/tensor_parallel.rs` | Wrap NCCL calls with comm audit |
| `crates/nsl-codegen/src/pipeline.rs` | Wrap PP send/recv with comm audit |
| `crates/nsl-codegen/src/context_parallel.rs` | Wrap ring attention comms with comm audit |
| `crates/nsl-runtime/src/trace.rs` | Extend TraceRecorder for cluster mode (global timestamps) |
| `crates/nsl-cli/src/main.rs` | Add `--cluster-trace`, `--trace-dir`, `--anomaly-threshold` flags |
| `crates/nsl-cli/src/debug.rs` | Add `--cluster`, `--find-nan`, `--gradient-history`, `--comm-audit` |

---

## Section 11: Testing Strategy

### Unit Tests

**`crates/nsl-runtime/src/cluster_trace.rs`:**
- ClockSync computes correct offset from known T1/T2/T3/T4 values
- ClockSync precision is half the round-trip time
- `to_global` correctly applies positive and negative offsets
- ClusterTraceEntry serialization/deserialization round-trip (140 bytes)

**`crates/nsl-runtime/src/gradient_tracker.rs`:**
- Gradient tracking with glob pattern `*.q_proj` matches `layer.0.attention.q_proj`
- Gradient tracking ignores non-matching layers
- GradientHistoryEntry serialization round-trip
- Ring buffer correctly drops oldest entries when capacity exceeded

**`crates/nsl-runtime/src/comm_auditor.rs`:**
- `hash_tensor` produces different hashes for different tensors
- `hash_tensor` produces identical hashes for identical tensors
- `verify_allreduce_consistency` detects mismatched output hashes
- `verify_allreduce_consistency` passes when all output hashes match
- CommAuditEntry serialization round-trip

**`crates/nsl-cli/src/cluster_debug.rs`:**
- `find_first_nan` returns earliest NaN by global timestamp (not by rank)
- `trace_nan_origin` walks backward to find the first non-NaN op
- `gradient_divergence_report` detects outlier ranks with 3x mean gradient norm
- Chrome export produces valid JSON with correct `pid` (rank) assignments
- Rank/step filtering correctly excludes entries outside the filter range

**`crates/nsl-runtime/src/freeze_dump.rs`:**
- FreezeDump serialization includes all fields
- FreezeDump file size matches expected (headers + tensor data)

### Integration Tests

- 4-rank simulated cluster: inject NaN on rank 2, verify anomaly detection on coordinator
- Clock sync with artificial 10ms offset: verify to_global produces aligned timestamps
- All-reduce audit: 4 ranks with correct all-reduce, verify no violations
- All-reduce audit: 4 ranks with corrupted rank 3 output, verify violation detected

### E2E Tests

- **`examples/m61_cluster_trace_basic.nsl`** — 2-rank training with `cluster_trace: true`, verify per-rank trace files written
- **`examples/m61_cluster_nan_detect.nsl`** — Inject NaN via extreme learning rate, verify `nsl debug --cluster --find-nan` finds it
- **`examples/m61_cluster_freeze.nsl`** — `nan_freeze: true`, inject NaN, verify all ranks dump state files
- **`examples/m61_gradient_tracking.nsl`** — Track attention layer gradients, verify history file contains entries
- **`examples/m61_comm_audit.nsl`** — Run all-reduce, verify comm audit entries with matching output hashes

---

## Section 12: Deliverables

- `nsl run --cluster-trace` records per-rank binary traces with global clock sync
- Cluster trace format: M45 extended with training step, rank, layer ID, comm audit
- NTP-style global clock synchronization across all ranks (sub-millisecond alignment)
- Cross-rank NaN detection: find the first rank and operation where NaN/Inf appeared
- Gradient magnitude history: per-rank, per-layer gradient norm tracking
- Communication audit: verify all-reduce consistency via output hashing
- Cluster freeze: all ranks dump full state on anomaly (weights, gradients, batch)
- `nsl debug --cluster` merges per-rank traces into unified timeline
- `nsl debug --cluster --find-nan` pinpoints first NaN across all ranks
- `nsl debug --cluster --gradient-history` shows gradient divergence report
- `nsl debug --cluster --comm-audit` verifies all-reduce consistency
- `nsl debug --cluster --export-chrome` exports to Chrome tracing format
- TUI with rank filtering, step filtering, NaN chain tracing
- Configurable anomaly threshold, check interval, and gradient tracking patterns

## Not in Scope

- Live streaming trace viewer (traces are post-mortem only)
- Automated root-cause analysis (the debugger shows the data; human interprets)
- Trace compression for long-running jobs (use `max_trace_steps` ring buffer)
- Cross-job trace comparison (comparing different training runs)
- Integration with external observability systems (Prometheus, Grafana)
- Network packet capture (debugging at the NCCL/InfiniBand level)
- GPU kernel-level profiling (Nsight-level detail — use Nsight directly)
