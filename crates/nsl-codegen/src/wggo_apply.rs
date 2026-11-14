//! WGGO — decision applicator.
//!
//! Takes the optimiser's structured output (InterLayerPlan + per-layer
//! ILP solutions, both after conflict resolution) and projects it into
//! the inputs each downstream technique consumes.  This is the bridge
//! that turns "WGGO chose CSHA level 1 for layer 4" into a modified
//! [`crate::wrga::WrgaInput`] / `CshaInput` / `FaseConfig` the rest of
//! the compiler uses.
//!
//! The module deliberately produces *data* — there's no mutation of
//! shared AST state here.  Downstream passes pick these up from the
//! compile context.

use serde::Serialize;

use crate::wggo_dp::{CoarseDecision, InterLayerPlan};
use crate::wggo_ilp::{AdapterPlacement, LayerIlpSolution};

/// Normalised per-layer decision ready for downstream consumption.
#[derive(Debug, Clone, Serialize)]
pub struct AppliedLayer {
    pub layer_index: u32,
    pub layer_name: String,
    pub coarse: CoarseDecision,
    pub pipeline_stage: u32,
    /// ZeRO parameter-shard factor.  Named `shard_factor` for back-compat;
    /// see `shard_grads`/`shard_optim` for the other two ZeRO dimensions.
    pub shard_factor: u32,
    /// ZeRO gradient-shard factor (preserved from the inter-layer plan, no
    /// longer collapsed into `shard_factor`).
    pub shard_grads: u32,
    /// ZeRO optimizer-state-shard factor (preserved from the inter-layer plan).
    pub shard_optim: u32,
    pub active_heads: u32,
    pub ffn_width: u64,
    pub csha_level: u8,
    pub adapter_rank: u64,
    /// Which projections the WRGA adapter attaches to (G5/G7 decision).
    /// `None` when `adapter_rank == 0` or the layer is pruned.  Carried to
    /// the consumer boundary alongside `adapter_rank` for the WRGA codegen
    /// wiring; see [`crate::wggo_overrides::PerLayerOverride`].
    pub adapter_placement: AdapterPlacement,
    pub optim_m_bits: u8,
    pub optim_v_bits: u8,
    /// Whether FASE fuses the optimizer step into the backward pass.
    /// Forced to `false` for pruned layers and for layers sharded by CPDT
    /// (the conflict resolver demotes the latter via `DeferFaseStep`).
    pub fase_fused: bool,
    /// PCA packing mode (0=none, 1=segment_id, 2=tile_skip, 3=multi_seq).
    /// Forced to 0 on pruned layers.
    pub packing_mode: u8,
    /// Estimated layer cost post-application (μs). 0 for a pruned layer and
    /// for a layer with no feasible configuration.
    pub estimated_us: f64,
    /// Parameter bytes of the configuration the plan chose for this layer
    /// (0 for a pruned layer).
    pub param_bytes: u64,
    /// Forward-activation bytes of that configuration.
    pub activation_bytes: u64,
}

/// Aggregate applied plan.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AppliedPlan {
    pub layers: Vec<AppliedLayer>,
    /// Sum of the layers' `estimated_us`.
    pub total_us: f64,
    /// Per-device resident training memory of the plan: for every kept
    /// layer, the parameters and optimizer moments of its chosen
    /// configuration divided by its shard factor, plus its activations.
    pub peak_memory_bytes: u64,
}

impl AppliedPlan {
    pub fn active_heads_total(&self) -> u32 {
        self.layers.iter().map(|l| l.active_heads).sum()
    }

    pub fn adapters_on_layers(&self) -> Vec<u32> {
        self.layers
            .iter()
            .filter(|l| l.adapter_rank > 0)
            .map(|l| l.layer_index)
            .collect()
    }
}

/// Combine a [`InterLayerPlan`] with per-layer ILP solutions into a
/// single [`AppliedPlan`].
///
/// Sizes come from the Level-2 solution: the configuration the solver chose
/// for the layer, at the moment precision it chose. The Level-1 plan's own
/// sizes describe one fixed configuration per layer and were what the plan
/// used to report as its memory; they are read here only for a layer the
/// Level-2 solve carries no size for (a hand-built solution).
pub fn apply(inter: &InterLayerPlan, ilp: &[LayerIlpSolution]) -> AppliedPlan {
    assert_eq!(
        inter.layers.len(),
        ilp.len(),
        "inter-layer plan and per-layer ILP solutions must be parallel"
    );
    let mut layers = Vec::with_capacity(inter.layers.len());
    let mut total = 0.0;
    let mut memory = 0u64;
    for (coarse, ilp_sol) in inter.layers.iter().zip(ilp.iter()) {
        let active_heads = ilp_sol.decision.active_heads() as u32;
        let pruned = coarse.decision == CoarseDecision::Prune;
        // Prune-decision propagation: if the inter-layer DP decided to
        // prune, force adapter rank and csha level to safe defaults.
        let (csha_level, adapter_rank, adapter_placement, fase_fused, packing_mode) = if pruned {
            (0, 0, AdapterPlacement::None, false, 0u8)
        } else {
            (
                ilp_sol.decision.csha_level,
                ilp_sol.decision.adapter_rank,
                ilp_sol.decision.adapter_placement,
                ilp_sol.decision.fase_fused,
                ilp_sol.decision.packing_mode,
            )
        };
        // A pruned layer costs nothing; a layer with no feasible
        // configuration has no cost to add (its `cost_us` is infinite, and
        // adding it made the plan's total infinite).
        let cost = if pruned || !ilp_sol.cost_us.is_finite() {
            0.0
        } else {
            ilp_sol.cost_us
        };
        total += cost;
        let (param_bytes, activation_bytes) = if pruned {
            (0, 0)
        } else if ilp_sol.param_bytes == 0 && ilp_sol.activation_bytes == 0 {
            (coarse.param_bytes, coarse.activation_bytes)
        } else {
            (ilp_sol.param_bytes, ilp_sol.activation_bytes)
        };
        if !pruned {
            // `memory_bytes` is the unsharded resident figure: parameters,
            // moments at the chosen precision, activations (and the FASE
            // surfaces under a budget). Everything but the activations is
            // sharded.
            let resident = ilp_sol
                .memory_bytes
                .max(param_bytes.saturating_add(activation_bytes));
            let sharded = resident.saturating_sub(activation_bytes)
                / u64::from(coarse.shard_params.max(1));
            memory = memory.saturating_add(sharded.saturating_add(activation_bytes));
        }
        layers.push(AppliedLayer {
            layer_index: coarse.layer_index,
            layer_name: coarse.name.clone(),
            coarse: coarse.decision,
            pipeline_stage: coarse.pipeline_stage,
            shard_factor: coarse.shard_params,
            shard_grads: coarse.shard_grads,
            shard_optim: coarse.shard_optim,
            active_heads,
            ffn_width: ilp_sol.decision.ffn_width,
            csha_level,
            adapter_rank,
            adapter_placement,
            optim_m_bits: ilp_sol.decision.optim_m_bits,
            optim_v_bits: ilp_sol.decision.optim_v_bits,
            fase_fused,
            packing_mode,
            estimated_us: cost,
            param_bytes,
            activation_bytes,
        });
    }
    AppliedPlan {
        layers,
        total_us: total,
        peak_memory_bytes: memory,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_cost::{build_lut, LayerShape, LutAxes};
    use crate::wggo_dp::{CoarseDecision, LayerPlan};
    use crate::wggo_ilp::{solve_layer, LayerIlpConstraints};

    fn toy_shape() -> LayerShape {
        LayerShape {
            batch: 1,
            seq: 512,
            d_model: 256,
            head_dim: 32,
            n_kv_heads: 4,
            dtype_bytes: 2,
        }
    }

    fn gpu() -> &'static crate::gpu_specs::GpuSpec {
        crate::gpu_specs::find_gpu("H100")
            .or_else(|| crate::gpu_specs::find_gpu("h100"))
            .unwrap_or(&crate::gpu_specs::GPU_DATABASE[0])
    }

    fn inter_plan(n: u32) -> InterLayerPlan {
        InterLayerPlan {
            layers: (0..n)
                .map(|i| LayerPlan {
                    layer_index: i,
                    name: format!("blocks.{i}"),
                    decision: if i == 2 {
                        CoarseDecision::Prune
                    } else {
                        CoarseDecision::KeepFull
                    },
                    pipeline_stage: 0,
                    shard_params: 4,
                    shard_grads: 4,
                    shard_optim: 4,
                    estimated_us: 10.0,
                    estimated_bytes: 1_000_000,
                    param_bytes: if i == 0 { 6_000_000 } else { 500_000 },
                    activation_bytes: if i == 0 { 2_000_000 } else { 300_000 },
                })
                .collect(),
            total_us: 30.0,
            peak_memory_bytes: 2_000_000,
            pipeline_stages: 1,
        }
    }

    #[test]
    fn apply_zeroes_pruned_layers() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp: Vec<_> = (0..3)
            .map(|_| solve_layer(&lut, &LayerIlpConstraints::default()))
            .collect();
        let applied = apply(&inter_plan(3), &ilp);
        // Layer 2 is pruned → rank 0, csha 0, cost 0.
        let pruned = &applied.layers[2];
        assert_eq!(pruned.coarse, CoarseDecision::Prune);
        assert_eq!(pruned.csha_level, 0);
        assert_eq!(pruned.adapter_rank, 0);
        assert_eq!(pruned.estimated_us, 0.0);
    }

    #[test]
    fn apply_preserves_ilp_decisions_on_kept_layers() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp: Vec<_> = (0..3)
            .map(|_| solve_layer(&lut, &LayerIlpConstraints::default()))
            .collect();
        let applied = apply(&inter_plan(3), &ilp);
        let kept = &applied.layers[0];
        assert_eq!(kept.csha_level, ilp[0].decision.csha_level);
        assert_eq!(kept.ffn_width, ilp[0].decision.ffn_width);
    }

    #[test]
    fn adapters_are_reported() {
        let inter = inter_plan(3);
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        // Solve all three layers normally, but directly inject a nonzero rank.
        let ilp = vec![
            {
                let mut s = solve_layer(&lut, &LayerIlpConstraints::default());
                s.decision.adapter_rank = 4;
                s
            },
            solve_layer(&lut, &LayerIlpConstraints::default()),
            solve_layer(&lut, &LayerIlpConstraints::default()),
        ];
        let applied = apply(&inter, &ilp);
        let adapters = applied.adapters_on_layers();
        // Layer 0 has rank=4; layer 2 is pruned → forced to 0.
        assert!(adapters.contains(&0));
        assert!(!adapters.contains(&2));
    }

    #[test]
    fn apply_carries_adapter_placement_and_overrides_propagate() {
        // G13: the placement decision flows ILP -> AppliedLayer ->
        // PerLayerOverride, and is forced to None on pruned layers (mirroring
        // adapter_rank).
        let inter = inter_plan(3); // layer 2 is Prune
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp = vec![
            {
                let mut s = solve_layer(&lut, &LayerIlpConstraints::default());
                s.decision.adapter_rank = 8;
                s.decision.adapter_placement = AdapterPlacement::AttnQKVO;
                s
            },
            solve_layer(&lut, &LayerIlpConstraints::default()),
            {
                let mut s = solve_layer(&lut, &LayerIlpConstraints::default());
                s.decision.adapter_rank = 8;
                s.decision.adapter_placement = AdapterPlacement::AttnAndFfn;
                s
            },
        ];
        let applied = apply(&inter, &ilp);
        assert_eq!(applied.layers[0].adapter_placement, AdapterPlacement::AttnQKVO);
        assert_eq!(
            applied.layers[2].adapter_placement,
            AdapterPlacement::None,
            "pruned layer must force placement to None"
        );

        let overrides = crate::wggo_overrides::WggoOverrides::from_applied(&applied);
        assert_eq!(
            overrides.per_layer[0].adapter_placement,
            AdapterPlacement::AttnQKVO,
            "override must carry the placement to the consumer boundary"
        );
        assert_eq!(overrides.per_layer[2].adapter_placement, AdapterPlacement::None);
    }

    #[test]
    #[should_panic(expected = "parallel")]
    fn length_mismatch_panics() {
        let inter = inter_plan(3);
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp = vec![solve_layer(&lut, &LayerIlpConstraints::default())];
        let _ = apply(&inter, &ilp);
    }

    #[test]
    fn pruned_layers_force_packing_off() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp: Vec<_> = (0..3)
            .map(|_| {
                let mut s = solve_layer(&lut, &LayerIlpConstraints::default());
                s.decision.packing_mode = 3;
                s
            })
            .collect();
        let applied = apply(&inter_plan(3), &ilp);
        assert_eq!(applied.layers[2].packing_mode, 0);
        assert_eq!(applied.layers[0].packing_mode, 3);
    }

    #[test]
    fn pruned_layers_force_fase_fused_off() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp: Vec<_> = (0..3)
            .map(|_| {
                let mut s = solve_layer(&lut, &LayerIlpConstraints::default());
                s.decision.fase_fused = true; // pretend ILP picked fused
                s
            })
            .collect();
        let applied = apply(&inter_plan(3), &ilp);
        assert!(!applied.layers[2].fase_fused, "pruned layer must not fuse");
        assert!(applied.layers[0].fase_fused, "kept layer preserves choice");
    }

    /// The applied plan carries the sizes of the configuration the Level-2
    /// solve chose, and its memory is their sum.
    ///
    /// Three layers at a shard factor of 4, the third pruned. Layer 0:
    /// 6,000,000 parameter bytes, 2,000,000 of activations, 20,000,000
    /// resident (so 18,000,000 of parameters and moments, 4,500,000 after
    /// sharding, 6,500,000 with the activations). Layer 1: 500,000 and
    /// 300,000 with 1,800,000 resident, so 375,000 + 300,000. The pruned
    /// layer holds nothing. The plan used to carry the Level-1 plan's sizes
    /// and report the larger of its total and one layer's resident bytes.
    #[test]
    fn applied_layer_carries_the_chosen_configurations_bytes() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let sized = |param: u64, act: u64, resident: u64| {
            let mut s = solve_layer(&lut, &LayerIlpConstraints::default());
            s.param_bytes = param;
            s.activation_bytes = act;
            s.memory_bytes = resident;
            s
        };
        let ilp = vec![
            sized(6_000_000, 2_000_000, 20_000_000),
            sized(500_000, 300_000, 1_800_000),
            sized(700_000, 100_000, 2_200_000),
        ];
        // The Level-1 plan's own sizes differ from every one of these.
        let mut inter = inter_plan(3);
        for l in &mut inter.layers {
            l.param_bytes = 1;
            l.activation_bytes = 1;
        }
        let applied = apply(&inter, &ilp);
        assert_eq!(applied.layers[0].param_bytes, 6_000_000);
        assert_eq!(applied.layers[0].activation_bytes, 2_000_000);
        assert_eq!(applied.layers[1].param_bytes, 500_000);
        assert_eq!(applied.layers[1].activation_bytes, 300_000);
        assert_eq!(applied.layers[2].param_bytes, 0, "a pruned layer holds nothing");
        assert_eq!(applied.layers[2].activation_bytes, 0);
        assert_eq!(applied.peak_memory_bytes, 6_500_000 + 675_000);
    }

    /// A layer with no feasible configuration has an infinite cost; it adds
    /// nothing to the plan's step time, which used to become infinite.
    #[test]
    fn an_infeasible_layer_does_not_make_the_step_time_infinite() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let mut ilp: Vec<_> = (0..3)
            .map(|_| solve_layer(&lut, &LayerIlpConstraints::default()))
            .collect();
        let finite = ilp[1].cost_us;
        ilp[0].cost_us = f64::INFINITY;
        ilp[0].feasible = false;
        let applied = apply(&inter_plan(3), &ilp);
        assert_eq!(applied.layers[0].estimated_us, 0.0);
        assert!((applied.total_us - finite).abs() < 1e-9, "{}", applied.total_us);
    }

    #[test]
    fn apply_preserves_three_shard_factors() {
        // The three ZeRO shard factors must survive the projection into
        // AppliedLayer rather than being collapsed into one (G6).
        let inter = InterLayerPlan {
            layers: vec![LayerPlan {
                layer_index: 0,
                name: "blocks.0".to_string(),
                decision: CoarseDecision::KeepFull,
                pipeline_stage: 0,
                shard_params: 8,
                shard_grads: 4,
                shard_optim: 2,
                estimated_us: 10.0,
                estimated_bytes: 1_000_000,
                param_bytes: 500_000,
                activation_bytes: 300_000,
            }],
            total_us: 10.0,
            peak_memory_bytes: 1_000_000,
            pipeline_stages: 1,
        };
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp = vec![solve_layer(&lut, &LayerIlpConstraints::default())];
        let applied = apply(&inter, &ilp);
        assert_eq!(applied.layers[0].shard_factor, 8);
        assert_eq!(applied.layers[0].shard_grads, 4);
        assert_eq!(applied.layers[0].shard_optim, 2);
    }

    #[test]
    fn active_heads_total_matches_per_layer_sum() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp: Vec<_> = (0..3)
            .map(|_| solve_layer(&lut, &LayerIlpConstraints::default()))
            .collect();
        let applied = apply(&inter_plan(3), &ilp);
        let sum: u32 = applied.layers.iter().map(|l| l.active_heads).sum();
        assert_eq!(applied.active_heads_total(), sum);
    }
}
