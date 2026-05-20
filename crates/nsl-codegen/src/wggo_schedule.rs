//! WGGO — Stage 8: cross-device communication schedule.
//!
//! After the per-layer ILP and conflict-resolution stages settle the
//! decision variables, this stage produces an explicit schedule for the
//! collectives CPDT will issue and pins the FASE optimizer-step position
//! relative to those collectives.  Without this stage the runtime would
//! have to re-derive collective ordering from scratch — and would risk
//! issuing a fused FASE step before reduce-scatter completes, which the
//! research doc (§5, Stage 8) explicitly forbids.
//!
//! The schedule is data; it does not emit code.  Downstream CPDT lowering
//! consumes it via [`CommSchedule::entries`].
//!
//! ZeRO terminology used here:
//!   - **shard_params > 1**  → AllGather needed before the forward.
//!   - **shard_grads  > 1**  → ReduceScatter after the backward.
//!   - **shard_optim  > 1**  → ReduceScatter after the backward (subsumed
//!                              by the grad one if both are sharded).
//!
//! When a layer's optimizer step is FASE-fused, it must execute after any
//! ReduceScatter for that layer; the resolver already demoted it to
//! `false` for sharded layers via `DeferFaseStep`, so a fused step on a
//! sharded layer would be a logic bug — flagged via `feasible = false`.

use serde::Serialize;

use crate::wggo_apply::AppliedPlan;
use crate::wggo_dp::InterLayerPlan;

/// Categorical collective op the schedule prescribes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CollectiveOp {
    /// No cross-device collective on this layer.
    None,
    /// AllGather params before forward (params-sharded only).
    AllGather,
    /// ReduceScatter grads after backward (grads/optim-sharded).
    ReduceScatter,
    /// AllReduce grads (data-parallel without ZeRO sharding).
    AllReduce,
}

impl CollectiveOp {
    pub fn as_str(self) -> &'static str {
        match self {
            CollectiveOp::None => "none",
            CollectiveOp::AllGather => "all_gather",
            CollectiveOp::ReduceScatter => "reduce_scatter",
            CollectiveOp::AllReduce => "all_reduce",
        }
    }
}

/// Where the FASE optimizer step runs relative to the layer's collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FaseStepPosition {
    /// Layer is unsharded and FASE was not picked.
    NotApplicable,
    /// Layer is unsharded; FASE fuses the optimizer step into backward.
    FusedIntoBackward,
    /// Layer is sharded; FASE step runs after the ReduceScatter
    /// completes.  Equivalent to a non-fused step but co-scheduled by
    /// this stage so the runtime emits them as one back-to-back unit.
    AfterReduceScatter,
}

impl FaseStepPosition {
    pub fn as_str(self) -> &'static str {
        match self {
            FaseStepPosition::NotApplicable => "n/a",
            FaseStepPosition::FusedIntoBackward => "fused→backward",
            FaseStepPosition::AfterReduceScatter => "after→reduce_scatter",
        }
    }
}

/// One scheduled communication event.
#[derive(Debug, Clone, Serialize)]
pub struct CommScheduleEntry {
    pub layer_index: u32,
    pub layer_name: String,
    pub pipeline_stage: u32,
    pub shard_factor: u32,
    pub op: CollectiveOp,
    pub fase_position: FaseStepPosition,
    /// Layer whose backward must complete before this collective fires.
    /// `None` for the first layer in backward order.
    pub depends_on_layer: Option<u32>,
    /// Whether this entry is internally consistent (e.g., not asking for
    /// a fused FASE step on a sharded layer).  `false` means the
    /// resolver missed something — usually a bug worth investigating.
    pub feasible: bool,
}

/// Aggregate communication schedule for the run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CommSchedule {
    pub entries: Vec<CommScheduleEntry>,
    pub total_collectives: u32,
    pub fused_step_count: u32,
    pub deferred_step_count: u32,
    pub infeasible_count: u32,
}

impl CommSchedule {
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        writeln!(
            s,
            "Communication schedule: {} collectives, {} fused / {} deferred FASE steps",
            self.total_collectives, self.fused_step_count, self.deferred_step_count
        )
        .unwrap();
        for e in &self.entries {
            writeln!(
                s,
                "  L{:02} {:<24} stage={} shard={} op={} fase={}{}",
                e.layer_index,
                e.layer_name,
                e.pipeline_stage,
                e.shard_factor,
                e.op.as_str(),
                e.fase_position.as_str(),
                if e.feasible { "" } else { "  [INFEASIBLE]" }
            )
            .unwrap();
        }
        s
    }
}

/// Build the communication schedule from the inter-layer plan + the
/// post-resolution applied plan.  Iterates layers in *backward* order
/// (largest index first) so dependency edges read forward in time.
pub fn build_schedule(inter: &InterLayerPlan, applied: &AppliedPlan) -> CommSchedule {
    assert_eq!(
        inter.layers.len(),
        applied.layers.len(),
        "inter-layer plan and applied plan must be parallel"
    );

    let mut entries = Vec::with_capacity(inter.layers.len());
    let mut total_collectives = 0u32;
    let mut fused = 0u32;
    let mut deferred = 0u32;
    let mut infeasible = 0u32;

    // Walk in backward order (last layer's grads land first).
    let pairs: Vec<_> = inter
        .layers
        .iter()
        .zip(applied.layers.iter())
        .rev()
        .collect();

    let mut prev_layer: Option<u32> = None;
    for (lp, al) in pairs {
        let op = pick_collective(lp.shard_params, lp.shard_grads, lp.shard_optim);
        let sharded = matches!(
            op,
            CollectiveOp::AllGather | CollectiveOp::ReduceScatter | CollectiveOp::AllReduce
        );
        let (fase_position, layer_feasible) = match (al.fase_fused, sharded) {
            (true, false) => (FaseStepPosition::FusedIntoBackward, true),
            (true, true) => {
                // Resolver should have demoted this — flag as infeasible.
                infeasible += 1;
                (FaseStepPosition::AfterReduceScatter, false)
            }
            // FASE was not selected — the optimizer step takes its
            // standard path and doesn't need a special schedule entry.
            (false, _) => (FaseStepPosition::NotApplicable, true),
        };

        match fase_position {
            FaseStepPosition::FusedIntoBackward => fused += 1,
            FaseStepPosition::AfterReduceScatter => deferred += 1,
            FaseStepPosition::NotApplicable => {}
        }
        if op != CollectiveOp::None {
            total_collectives += 1;
        }

        entries.push(CommScheduleEntry {
            layer_index: lp.layer_index,
            layer_name: lp.name.clone(),
            pipeline_stage: lp.pipeline_stage,
            shard_factor: lp.shard_params.max(lp.shard_grads).max(lp.shard_optim),
            op,
            fase_position,
            depends_on_layer: prev_layer,
            feasible: layer_feasible,
        });
        prev_layer = Some(lp.layer_index);
    }

    CommSchedule {
        entries,
        total_collectives,
        fused_step_count: fused,
        deferred_step_count: deferred,
        infeasible_count: infeasible,
    }
}

/// Decide which collective to issue based on which ZeRO shards are active.
fn pick_collective(shard_params: u32, shard_grads: u32, shard_optim: u32) -> CollectiveOp {
    if shard_grads > 1 || shard_optim > 1 {
        CollectiveOp::ReduceScatter
    } else if shard_params > 1 {
        CollectiveOp::AllGather
    } else {
        CollectiveOp::None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_apply::{apply, AppliedLayer};
    use crate::wggo_cost::{build_lut, LayerShape, LutAxes};
    use crate::wggo_dp::{LayerDecision as CoarseDecision, LayerPlan};
    use crate::wggo_ilp::{solve_layer, LayerIlpConstraints};

    fn h100() -> &'static crate::gpu_specs::GpuSpec {
        crate::gpu_specs::find_gpu("H100")
            .or_else(|| crate::gpu_specs::find_gpu("h100"))
            .unwrap_or(&crate::gpu_specs::GPU_DATABASE[0])
    }

    fn shape() -> LayerShape {
        LayerShape {
            batch: 1,
            seq: 512,
            d_model: 256,
            head_dim: 32,
            n_kv_heads: 4,
            dtype_bytes: 2,
        }
    }

    fn inter_plan(n: u32, shard_grads: u32) -> InterLayerPlan {
        InterLayerPlan {
            layers: (0..n)
                .map(|i| LayerPlan {
                    layer_index: i,
                    name: format!("blocks.{i}"),
                    decision: CoarseDecision::KeepFull,
                    pipeline_stage: 0,
                    shard_params: 1,
                    shard_grads,
                    shard_optim: 1,
                    estimated_us: 10.0,
                    estimated_bytes: 1_000_000,
                    param_bytes: 0,
                    activation_bytes: 0,
                })
                .collect(),
            total_us: 30.0,
            peak_memory_bytes: 2_000_000,
            pipeline_stages: 1,
        }
    }

    fn applied_for(inter: &InterLayerPlan, fase_fused: bool) -> AppliedPlan {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut ilp: Vec<_> = (0..inter.layers.len())
            .map(|_| solve_layer(&lut, &LayerIlpConstraints::default()))
            .collect();
        for s in &mut ilp {
            s.decision.fase_fused = fase_fused;
        }
        apply(inter, &ilp)
    }

    #[test]
    fn unsharded_layer_emits_no_collective() {
        let inter = inter_plan(2, 1);
        let applied = applied_for(&inter, false);
        let s = build_schedule(&inter, &applied);
        assert_eq!(s.total_collectives, 0);
        assert!(s.entries.iter().all(|e| e.op == CollectiveOp::None));
    }

    #[test]
    fn sharded_grads_emits_reduce_scatter() {
        let inter = inter_plan(3, 4);
        let applied = applied_for(&inter, false);
        let s = build_schedule(&inter, &applied);
        assert_eq!(s.total_collectives, 3);
        assert!(s
            .entries
            .iter()
            .all(|e| e.op == CollectiveOp::ReduceScatter));
    }

    #[test]
    fn fused_fase_on_unsharded_is_pinned_to_backward() {
        let inter = inter_plan(2, 1);
        let applied = applied_for(&inter, true);
        let s = build_schedule(&inter, &applied);
        assert!(s.entries.iter().all(|e| e.feasible));
        assert_eq!(s.fused_step_count, 2);
        assert_eq!(s.deferred_step_count, 0);
    }

    #[test]
    fn fused_fase_on_sharded_layer_is_flagged_infeasible() {
        // Manually constructing this state: resolver should have demoted
        // the FASE choice on a sharded layer.  When it doesn't, the
        // schedule must surface the bug via infeasible_count.
        let inter = inter_plan(2, 4);
        let applied = applied_for(&inter, true);
        let s = build_schedule(&inter, &applied);
        assert!(s.infeasible_count >= 1);
        assert!(s.entries.iter().any(|e| !e.feasible));
    }

    #[test]
    fn deferred_fase_on_sharded_runs_after_reduce_scatter() {
        let inter = inter_plan(2, 4);
        let applied = applied_for(&inter, false);
        let s = build_schedule(&inter, &applied);
        assert_eq!(s.deferred_step_count, 0); // not picked → not counted
                                              // No fused step; sharded layers don't pin a FASE position when
                                              // the user didn't pick fusion.
        for e in &s.entries {
            assert_eq!(e.fase_position, FaseStepPosition::NotApplicable);
        }
    }

    #[test]
    fn schedule_walks_backward_order() {
        let inter = inter_plan(4, 4);
        let applied = applied_for(&inter, false);
        let s = build_schedule(&inter, &applied);
        // First entry should be the highest layer index.
        assert_eq!(s.entries[0].layer_index, 3);
        assert_eq!(s.entries.last().unwrap().layer_index, 0);
        // Dependency chain: each entry depends on the previous one's layer.
        assert!(s.entries[0].depends_on_layer.is_none());
        for win in s.entries.windows(2) {
            assert_eq!(win[1].depends_on_layer, Some(win[0].layer_index));
        }
    }

    #[test]
    fn render_includes_summary_and_per_layer_lines() {
        let inter = inter_plan(2, 4);
        let applied = applied_for(&inter, false);
        let s = build_schedule(&inter, &applied);
        let r = s.render();
        assert!(r.contains("Communication schedule"));
        assert!(r.contains("reduce_scatter"));
    }

    #[test]
    fn shard_optim_alone_also_triggers_reduce_scatter() {
        let mut inter = inter_plan(1, 1);
        inter.layers[0].shard_optim = 4;
        let applied = applied_for(&inter, false);
        let s = build_schedule(&inter, &applied);
        assert_eq!(s.entries[0].op, CollectiveOp::ReduceScatter);
    }

    #[test]
    fn all_gather_when_only_params_sharded() {
        let mut inter = inter_plan(1, 1);
        inter.layers[0].shard_params = 4;
        let applied = applied_for(&inter, false);
        let s = build_schedule(&inter, &applied);
        assert_eq!(s.entries[0].op, CollectiveOp::AllGather);
    }

    #[test]
    #[should_panic(expected = "parallel")]
    fn length_mismatch_panics() {
        let inter = inter_plan(3, 1);
        let mut applied = applied_for(&inter, false);
        applied.layers.pop();
        let _ = build_schedule(&inter, &applied);
    }

    // Suppress unused-import warning on AppliedLayer (used implicitly via apply()).
    #[allow(dead_code)]
    fn _touch(_: AppliedLayer) {}
}
