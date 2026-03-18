# M43: Pipeline Parallelism — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split model layers across GPUs along the depth axis, implementing 1F1B and GPipe pipeline schedules, 3D parallelism rank mapping (dp×tp×pp), point-to-point inter-stage communication, gradient accumulation with loss scaling, and ZeRO optimizer state sharding. This completes NSL's 3D parallelism story (M30 tensor parallelism + M43 pipeline parallelism + data parallelism).

**Architecture:** Four new runtime modules under `crates/nsl-runtime/src/pipeline/` (schedule generation, point-to-point communication, stage partitioning) + `zero.rs` (ZeRO optimizer sharding). New codegen module `pipeline.rs` for 3D rank mapping and `@pipeline` decorator config extraction. Semantic validation for `@pipeline`, `distribute`, `zero_stage`, `gradient_accumulation_steps`.

**Tech Stack:** Rust (runtime FFI + codegen + semantic)

**Spec:** `docs/superpowers/specs/2026-03-15-m43-pipeline-parallelism-design.md`

**Prerequisites:** M30 (Tensor Parallelism — SPMD, collective communication, SimulatedBackend)

---

## Important: Scope of This Plan

**This plan builds the core pipeline scheduling, communication, and 3D rank infrastructure.** It delivers:
- `PipelineSchedule` with 1F1B and GPipe schedule generation
- `ScheduleStep` enum (Forward, Backward, Idle, Send/RecvActivation, Send/RecvGradient)
- `ParallelismConfig` — parse `"dp=2, tp=4, pp=4"` distribute strings
- 3D rank mapping: `rank_to_3d`, `rank_from_3d`, `get_tp_peers`, `get_pp_neighbors`, `get_dp_peers`
- Pipeline send/recv FFI stubs (SharedMem-backed, matching M30's SimulatedBackend pattern)
- `ZeROOptimizer` — parameter partitioning, gradient reduce-scatter, param all-gather FFI stubs
- Gradient accumulation FFI (accumulate_add, zero_grad, all_reduce)
- `@pipeline` and `distribute` semantic validation
- `--distribute` and `--zero-stage` CLI flags
- Codegen: `pipeline_config`, `parallelism_config`, `zero_stage` compiler fields
- Builtin registration for all pipeline/zero/gradient FFI functions
- 25+ unit tests

**Deferred to M43b:** `PipelinePartitioner` and `partition.rs` (layer-to-stage assignment, greedy balanced partitioning), actual Cranelift emission of pipelined train loops (`compile_train_block_pipelined`), activation checkpointing at stage boundaries (`nsl_checkpoint_save/load_input`), tied weight synchronization across stages (`TiedWeightSync`), ZeRO-3 per-layer AllGather/release, `CommGroups` struct for NCCL subgroup communicators, train block strategy selection, `CompileOptions` extension for distribute/zero-stage config flow, E2E convergence tests, performance benchmarks.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/pipeline/mod.rs` | Module declarations | 5 |
| `crates/nsl-runtime/src/pipeline/schedule.rs` | `PipelineSchedule`, 1F1B + GPipe generation, bubble ratio | 200 |
| `crates/nsl-runtime/src/pipeline/comm.rs` | Pipeline send/recv FFI (point-to-point, SharedMem) | 120 |
| `crates/nsl-runtime/src/zero.rs` | `ZeROOptimizer` — param partition, grad reduce-scatter FFI stubs | 150 |
| `crates/nsl-codegen/src/pipeline.rs` | `ParallelismConfig`, 3D rank mapping, `@pipeline` config | 200 |
| `crates/nsl-semantic/src/pipeline.rs` | `@pipeline`, `distribute`, `zero_stage` validation | 120 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod pipeline; pub mod zero;` |
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod pipeline;` |
| `crates/nsl-codegen/src/compiler.rs` | Add pipeline/parallelism/zero fields |
| `crates/nsl-codegen/src/builtins.rs` | Register ~15 FFI functions |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod pipeline;` |
| `crates/nsl-semantic/src/checker.rs` | Wire `@pipeline` validation |
| `crates/nsl-cli/src/main.rs` | Add `--distribute`, `--zero-stage` flags |

---

## Phase 1: Pipeline Schedule + 3D Rank Mapping

### Task 1: Pipeline Schedule Generation

**Files:**
- Create: `crates/nsl-runtime/src/pipeline/mod.rs`
- Create: `crates/nsl-runtime/src/pipeline/schedule.rs`

- [ ] **Step 1: Create pipeline module with 1F1B and GPipe schedule generation**

`pipeline/mod.rs`:
```rust
pub mod schedule;
pub mod comm;
```

`pipeline/schedule.rs` — the core scheduling logic from spec Section 3:

```rust
//! M43: Pipeline schedule generation — 1F1B and GPipe.

/// A single step in the pipeline schedule for one stage.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduleStep {
    Forward(usize),         // micro-batch index
    Backward(usize),        // micro-batch index
    Idle,                   // pipeline bubble
    SendActivation(usize),  // send to next stage
    RecvActivation(usize),  // recv from prev stage
    SendGradient(usize),    // send grad to prev stage
    RecvGradient(usize),    // recv grad from next stage
}

/// Complete pipeline schedule: steps[stage_id][time_slot].
#[derive(Debug, Clone)]
pub struct PipelineSchedule {
    pub steps: Vec<Vec<ScheduleStep>>,
    pub num_stages: usize,
    pub num_micro_batches: usize,
}

impl PipelineSchedule {
    /// Generate 1F1B (one-forward-one-backward) schedule.
    ///
    /// Per-stage algorithm (correct staggering):
    ///   warmup_forwards = num_stages - 1 - stage  (stage 0: P-1 warmups, last stage: 0)
    ///   steady_pairs = num_micro_batches - warmup_forwards
    ///   cooldown_backwards = warmup_forwards
    ///
    /// Warmup: Forward(0..warmup_forwards)
    /// Steady: for i in 0..steady_pairs: Backward(i), Forward(warmup_forwards + i)
    /// Cooldown: Backward(steady_pairs..num_micro_batches)
    pub fn generate_1f1b(num_stages: usize, num_micro_batches: usize) -> Self {
        let mut steps = vec![Vec::new(); num_stages];

        for stage in 0..num_stages {
            let warmup_fwds = (num_stages - 1 - stage).min(num_micro_batches);
            let steady_pairs = num_micro_batches.saturating_sub(warmup_fwds);
            let cooldown_bwds = warmup_fwds;

            let mut fwd_idx = 0usize; // next forward micro-batch to schedule
            let mut bwd_idx = 0usize; // next backward micro-batch to schedule

            // Phase 1: Warmup — pure forward fills
            for _ in 0..warmup_fwds {
                if stage > 0 {
                    steps[stage].push(ScheduleStep::RecvActivation(fwd_idx));
                }
                steps[stage].push(ScheduleStep::Forward(fwd_idx));
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::SendActivation(fwd_idx));
                }
                fwd_idx += 1;
            }

            // Phase 2: Steady state — alternating 1B 1F
            for _ in 0..steady_pairs {
                // Backward
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::RecvGradient(bwd_idx));
                }
                steps[stage].push(ScheduleStep::Backward(bwd_idx));
                if stage > 0 {
                    steps[stage].push(ScheduleStep::SendGradient(bwd_idx));
                }
                bwd_idx += 1;

                // Forward (if micro-batches remain)
                if fwd_idx < num_micro_batches {
                    if stage > 0 {
                        steps[stage].push(ScheduleStep::RecvActivation(fwd_idx));
                    }
                    steps[stage].push(ScheduleStep::Forward(fwd_idx));
                    if stage < num_stages - 1 {
                        steps[stage].push(ScheduleStep::SendActivation(fwd_idx));
                    }
                    fwd_idx += 1;
                }
            }

            // Phase 3: Cooldown — remaining backwards
            for _ in 0..cooldown_bwds {
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::RecvGradient(bwd_idx));
                }
                steps[stage].push(ScheduleStep::Backward(bwd_idx));
                if stage > 0 {
                    steps[stage].push(ScheduleStep::SendGradient(bwd_idx));
                }
                bwd_idx += 1;
            }
        }

        PipelineSchedule { steps, num_stages, num_micro_batches }
    }

    /// Generate GPipe schedule (all forwards then all backwards).
    pub fn generate_gpipe(num_stages: usize, num_micro_batches: usize) -> Self {
        let mut steps = vec![Vec::new(); num_stages];

        // All forwards first
        for mb in 0..num_micro_batches {
            for stage in 0..num_stages {
                if stage > 0 {
                    steps[stage].push(ScheduleStep::RecvActivation(mb));
                }
                steps[stage].push(ScheduleStep::Forward(mb));
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::SendActivation(mb));
                }
            }
        }

        // Then all backwards (reverse micro-batch order)
        for mb in (0..num_micro_batches).rev() {
            for stage in (0..num_stages).rev() {
                if stage < num_stages - 1 {
                    steps[stage].push(ScheduleStep::RecvGradient(mb));
                }
                steps[stage].push(ScheduleStep::Backward(mb));
                if stage > 0 {
                    steps[stage].push(ScheduleStep::SendGradient(mb));
                }
            }
        }

        PipelineSchedule { steps, num_stages, num_micro_batches }
    }

    /// Calculate the bubble ratio (fraction of idle time).
    pub fn bubble_ratio(&self) -> f64 {
        let total: usize = self.steps.iter().map(|s| s.len()).sum();
        let idle: usize = self.steps.iter()
            .map(|s| s.iter().filter(|step| matches!(step, ScheduleStep::Idle)).count())
            .sum();
        if total == 0 { return 0.0; }
        idle as f64 / total as f64
    }

    /// Count forward steps for a given stage.
    pub fn forward_count(&self, stage: usize) -> usize {
        self.steps[stage].iter()
            .filter(|s| matches!(s, ScheduleStep::Forward(_)))
            .count()
    }

    /// Count backward steps for a given stage.
    pub fn backward_count(&self, stage: usize) -> usize {
        self.steps[stage].iter()
            .filter(|s| matches!(s, ScheduleStep::Backward(_)))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1f1b_forward_backward_counts() {
        let sched = PipelineSchedule::generate_1f1b(4, 8);
        // Each stage should process all 8 micro-batches forward and backward
        for stage in 0..4 {
            assert_eq!(sched.forward_count(stage), 8,
                "stage {stage} forward count");
            assert_eq!(sched.backward_count(stage), 8,
                "stage {stage} backward count");
        }
    }

    #[test]
    fn test_1f1b_bubble_decreases_with_more_microbatches() {
        let sched_few = PipelineSchedule::generate_1f1b(4, 4);
        let sched_many = PipelineSchedule::generate_1f1b(4, 16);
        // More micro-batches → smaller bubble fraction
        assert!(sched_many.bubble_ratio() <= sched_few.bubble_ratio(),
            "bubble ratio should decrease: few={}, many={}",
            sched_few.bubble_ratio(), sched_many.bubble_ratio());
    }

    #[test]
    fn test_gpipe_all_forwards_before_backwards() {
        let sched = PipelineSchedule::generate_gpipe(2, 4);
        // Stage 0: all forwards should come before all backwards
        let mut seen_backward = false;
        for step in &sched.steps[0] {
            if matches!(step, ScheduleStep::Backward(_)) {
                seen_backward = true;
            }
            if matches!(step, ScheduleStep::Forward(_)) && seen_backward {
                panic!("GPipe: forward after backward in stage 0");
            }
        }
    }

    #[test]
    fn test_gpipe_forward_backward_counts() {
        let sched = PipelineSchedule::generate_gpipe(3, 6);
        for stage in 0..3 {
            assert_eq!(sched.forward_count(stage), 6);
            assert_eq!(sched.backward_count(stage), 6);
        }
    }

    #[test]
    fn test_single_stage_no_communication() {
        let sched = PipelineSchedule::generate_1f1b(1, 4);
        // Single stage: no send/recv steps
        for step in &sched.steps[0] {
            assert!(!matches!(step,
                ScheduleStep::SendActivation(_) | ScheduleStep::RecvActivation(_) |
                ScheduleStep::SendGradient(_) | ScheduleStep::RecvGradient(_)
            ), "single stage should have no communication");
        }
    }

    #[test]
    fn test_bubble_ratio_single_stage_zero() {
        let sched = PipelineSchedule::generate_1f1b(1, 4);
        assert_eq!(sched.bubble_ratio(), 0.0);
    }
}
```

### Task 2: 3D Rank Mapping + ParallelismConfig (Codegen)

**Files:**
- Create: `crates/nsl-codegen/src/pipeline.rs`

- [ ] **Step 2: Create pipeline.rs with ParallelismConfig, 3D rank mapping, and tests**

```rust
//! M43: Pipeline parallelism codegen — 3D rank mapping and @pipeline config.

use std::collections::HashMap;

/// 3D parallelism configuration parsed from `distribute: "dp=2, tp=4, pp=4"`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParallelismConfig {
    pub data_parallel: usize,
    pub tensor_parallel: usize,
    pub pipeline_parallel: usize,
}

impl ParallelismConfig {
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut dp = 1;
        let mut tp = 1;
        let mut pp = 1;

        for part in s.split(',') {
            let part = part.trim();
            if let Some(val) = part.strip_prefix("dp=") {
                dp = val.parse().map_err(|_| format!("invalid dp: '{val}'"))?;
            } else if let Some(val) = part.strip_prefix("tp=") {
                tp = val.parse().map_err(|_| format!("invalid tp: '{val}'"))?;
            } else if let Some(val) = part.strip_prefix("pp=") {
                pp = val.parse().map_err(|_| format!("invalid pp: '{val}'"))?;
            } else {
                return Err(format!("unknown key: '{part}'"));
            }
        }
        if dp < 1 || tp < 1 || pp < 1 {
            return Err("all dimensions must be >= 1".into());
        }
        Ok(ParallelismConfig { data_parallel: dp, tensor_parallel: tp, pipeline_parallel: pp })
    }

    pub fn total_gpus(&self) -> usize {
        self.data_parallel * self.tensor_parallel * self.pipeline_parallel
    }
}

/// Map flat rank to 3D (dp, pp, tp) coordinates.
/// Layout: dp outermost, pp middle, tp innermost (TP peers on adjacent GPUs).
pub fn rank_to_3d(rank: usize, config: &ParallelismConfig) -> (usize, usize, usize) {
    let tp_rank = rank % config.tensor_parallel;
    let pp_rank = (rank / config.tensor_parallel) % config.pipeline_parallel;
    let dp_rank = rank / (config.tensor_parallel * config.pipeline_parallel);
    (dp_rank, pp_rank, tp_rank)
}

/// Inverse: 3D coordinates to flat rank.
pub fn rank_from_3d(dp: usize, pp: usize, tp: usize, config: &ParallelismConfig) -> usize {
    dp * config.pipeline_parallel * config.tensor_parallel
        + pp * config.tensor_parallel
        + tp
}

/// Get tensor-parallel peers (same dp, same pp, all tp ranks).
pub fn get_tp_peers(rank: usize, config: &ParallelismConfig) -> Vec<usize> {
    let (dp, pp, _) = rank_to_3d(rank, config);
    (0..config.tensor_parallel).map(|tp| rank_from_3d(dp, pp, tp, config)).collect()
}

/// Get pipeline-parallel neighbors (prev stage, next stage).
pub fn get_pp_neighbors(rank: usize, config: &ParallelismConfig) -> (Option<usize>, Option<usize>) {
    let (dp, pp, tp) = rank_to_3d(rank, config);
    let prev = if pp > 0 { Some(rank_from_3d(dp, pp - 1, tp, config)) } else { None };
    let next = if pp < config.pipeline_parallel - 1 { Some(rank_from_3d(dp, pp + 1, tp, config)) } else { None };
    (prev, next)
}

/// Get data-parallel peers (same pp, same tp, all dp ranks).
pub fn get_dp_peers(rank: usize, config: &ParallelismConfig) -> Vec<usize> {
    let (_, pp, tp) = rank_to_3d(rank, config);
    (0..config.data_parallel).map(|dp| rank_from_3d(dp, pp, tp, config)).collect()
}

/// Pipeline configuration extracted from @pipeline decorator.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub num_stages: usize,
    pub schedule_type: ScheduleType,
    pub checkpoint_stages: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScheduleType { OneF1B, GPipe }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_distribute_basic() {
        let config = ParallelismConfig::parse("dp=2, tp=4, pp=4").unwrap();
        assert_eq!(config.data_parallel, 2);
        assert_eq!(config.tensor_parallel, 4);
        assert_eq!(config.pipeline_parallel, 4);
        assert_eq!(config.total_gpus(), 32);
    }

    #[test]
    fn parse_distribute_single() {
        let config = ParallelismConfig::parse("pp=4").unwrap();
        assert_eq!(config.data_parallel, 1);
        assert_eq!(config.pipeline_parallel, 4);
        assert_eq!(config.total_gpus(), 4);
    }

    #[test]
    fn parse_distribute_error() {
        assert!(ParallelismConfig::parse("dp=0").is_err());
        assert!(ParallelismConfig::parse("foo=2").is_err());
        assert!(ParallelismConfig::parse("dp=abc").is_err());
    }

    #[test]
    fn rank_3d_roundtrip() {
        let config = ParallelismConfig { data_parallel: 2, tensor_parallel: 4, pipeline_parallel: 4 };
        for rank in 0..32 {
            let (dp, pp, tp) = rank_to_3d(rank, &config);
            assert_eq!(rank_from_3d(dp, pp, tp, &config), rank, "roundtrip failed for rank {rank}");
        }
    }

    #[test]
    fn rank_3d_known_values() {
        let config = ParallelismConfig { data_parallel: 2, tensor_parallel: 4, pipeline_parallel: 4 };
        // Rank 0: dp=0, pp=0, tp=0
        assert_eq!(rank_to_3d(0, &config), (0, 0, 0));
        // Rank 3: dp=0, pp=0, tp=3
        assert_eq!(rank_to_3d(3, &config), (0, 0, 3));
        // Rank 4: dp=0, pp=1, tp=0
        assert_eq!(rank_to_3d(4, &config), (0, 1, 0));
        // Rank 16: dp=1, pp=0, tp=0
        assert_eq!(rank_to_3d(16, &config), (1, 0, 0));
    }

    #[test]
    fn tp_peers() {
        let config = ParallelismConfig { data_parallel: 2, tensor_parallel: 4, pipeline_parallel: 2 };
        let peers = get_tp_peers(0, &config);
        assert_eq!(peers, vec![0, 1, 2, 3]);
    }

    #[test]
    fn pp_neighbors() {
        let config = ParallelismConfig { data_parallel: 1, tensor_parallel: 1, pipeline_parallel: 4 };
        assert_eq!(get_pp_neighbors(0, &config), (None, Some(1)));
        assert_eq!(get_pp_neighbors(1, &config), (Some(0), Some(2)));
        assert_eq!(get_pp_neighbors(3, &config), (Some(2), None));
    }

    #[test]
    fn dp_peers() {
        let config = ParallelismConfig { data_parallel: 4, tensor_parallel: 2, pipeline_parallel: 2 };
        // Rank 0: dp=0, pp=0, tp=0. DP peers: ranks 0, 4, 8, 12
        let peers = get_dp_peers(0, &config);
        assert_eq!(peers, vec![0, 4, 8, 12]);
    }
}
```

---

## Phase 2: Communication + ZeRO + Semantic

### Task 3: Pipeline Communication FFI

**Files:**
- Create: `crates/nsl-runtime/src/pipeline/comm.rs`

- [ ] **Step 3: Create comm.rs with pipeline send/recv stubs**

Point-to-point send/recv for activations and gradients between adjacent pipeline stages. Uses the same SharedMem pattern as M30's SimulatedBackend for single-node testing.

FFI functions (all stubs returning 0 for M43a):
- `nsl_pipeline_init(num_stages, schedule_type, num_micro_batches) -> i64`
- `nsl_pipeline_send(tensor_ptr, dst_rank, tag, stream) -> i64`
- `nsl_pipeline_recv(shape_ptr, ndim, dtype, src_rank, tag, stream) -> i64`
- `nsl_pipeline_send_grad(grad_ptr, dst_rank, tag, stream) -> i64`
- `nsl_pipeline_recv_grad(shape_ptr, ndim, dtype, src_rank, tag, stream) -> i64`
- `nsl_pipeline_barrier() -> i64`
- `nsl_pipeline_destroy() -> i64`

2 tests: init/destroy lifecycle, barrier returns 0.

### Task 4: ZeRO Optimizer Stubs

**Files:**
- Create: `crates/nsl-runtime/src/zero.rs`

- [ ] **Step 4: Create zero.rs with ZeRO parameter partitioning and FFI stubs**

```rust
//! M43: ZeRO optimizer state sharding.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZeROStage { Stage1, Stage2, Stage3 }

impl ZeROStage {
    pub fn from_i64(v: i64) -> Option<Self> {
        match v { 1 => Some(Self::Stage1), 2 => Some(Self::Stage2), 3 => Some(Self::Stage3), _ => None }
    }
}

/// Partition parameters across data-parallel ranks using round-robin.
pub fn partition_params(num_params: usize, world_size: usize) -> Vec<Vec<usize>> {
    let mut partitions = vec![Vec::new(); world_size];
    for i in 0..num_params {
        partitions[i % world_size].push(i);
    }
    partitions
}
```

FFI stubs: `nsl_zero_init`, `nsl_zero_partition`, `nsl_zero_reduce_grads`, `nsl_zero_step`, `nsl_zero_destroy`.
Gradient accumulation FFI: `nsl_grad_accumulate_add`, `nsl_grad_zero`, `nsl_grad_all_reduce`.

3 tests: ZeROStage parsing, round-robin partition, FFI lifecycle.

### Task 5: Semantic Validation + CLI + Builtins

**Files:**
- Create: `crates/nsl-semantic/src/pipeline.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`, `checker.rs`
- Modify: `crates/nsl-cli/src/main.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`, `lib.rs`, `builtins.rs`

- [ ] **Step 5: Create semantic validation, add CLI flags, register builtins, add compiler fields**

Semantic `pipeline.rs`: validate `@pipeline(stages=N)`, `distribute` string, `zero_stage`, `gradient_accumulation_steps`.

CLI flags on `Run` and `Build`:
- `--distribute "dp=2, tp=4, pp=4"`
- `--zero-stage 2`

Compiler fields:
- `pipeline_config: Option<PipelineConfig>`
- `parallelism_config: Option<ParallelismConfig>`
- `zero_stage: Option<crate::pipeline::ZeROStageEnum>` (use the ZeROStage enum, not u8, for type safety)

Builtins (~15 functions): pipeline init/send/recv/barrier/destroy, zero init/partition/reduce/step/destroy, grad accumulate/zero/all_reduce.

---

## Phase 3: Wire + Verify

- [ ] **Step 6: Wire all modules into lib.rs files**

- [ ] **Step 7: `cargo build`**

- [ ] **Step 8: `cargo test` — expect 25+ new tests**

- [ ] **Step 9: `cargo clippy`**

---

## Verification Checklist

1. **1F1B schedule**: All stages process all micro-batches forward and backward
2. **GPipe schedule**: All forwards before any backward
3. **Bubble ratio**: Decreases with more micro-batches
4. **3D rank mapping**: Bijective for all ranks in config
5. **Distribute parsing**: Correct for "dp=2, tp=4, pp=4", errors for invalid
6. **TP/PP/DP peers**: Correct peer sets for all rank positions
7. **ZeRO partition**: Round-robin distributes evenly
8. **Pipeline FFI**: Init/destroy lifecycle works
9. **Semantic validation**: Invalid stages/distribute produce clear errors
10. **No regressions**: All 614+ existing tests pass
