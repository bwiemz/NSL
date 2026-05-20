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

use crate::wggo_dp::{InterLayerPlan, LayerDecision as CoarseDecision};
use crate::wggo_ilp::LayerIlpSolution;

/// Normalised per-layer decision ready for downstream consumption.
#[derive(Debug, Clone, Serialize)]
pub struct AppliedLayer {
    pub layer_index: u32,
    pub layer_name: String,
    pub coarse: CoarseDecision,
    pub pipeline_stage: u32,
    pub shard_factor: u32,
    pub active_heads: u32,
    pub ffn_width: u64,
    pub csha_level: u8,
    pub adapter_rank: u64,
    pub optim_m_bits: u8,
    pub optim_v_bits: u8,
    /// Whether FASE fuses the optimizer step into the backward pass.
    /// Forced to `false` for pruned layers and for layers sharded by CPDT
    /// (the conflict resolver demotes the latter via `DeferFaseStep`).
    pub fase_fused: bool,
    /// PCA packing mode (0=none, 1=segment_id, 2=tile_skip, 3=multi_seq).
    /// Forced to 0 on pruned layers.
    pub packing_mode: u8,
    /// Estimated layer cost post-application (μs).
    pub estimated_us: f64,
    /// Parameter bytes for this layer (sum of all weights).
    pub param_bytes: u64,
    /// Peak activation bytes during forward pass for this layer.
    pub activation_bytes: u64,
}

/// Aggregate applied plan.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AppliedPlan {
    pub layers: Vec<AppliedLayer>,
    pub total_us: f64,
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
pub fn apply(inter: &InterLayerPlan, ilp: &[LayerIlpSolution]) -> AppliedPlan {
    assert_eq!(
        inter.layers.len(),
        ilp.len(),
        "inter-layer plan and per-layer ILP solutions must be parallel"
    );
    let mut layers = Vec::with_capacity(inter.layers.len());
    let mut total = 0.0;
    let mut peak = inter.peak_memory_bytes;
    for (coarse, ilp_sol) in inter.layers.iter().zip(ilp.iter()) {
        let active_heads = ilp_sol.decision.active_heads() as u32;
        // Prune-decision propagation: if the inter-layer DP decided to
        // prune, force adapter rank and csha level to safe defaults.
        let (csha_level, adapter_rank, fase_fused, packing_mode) = match coarse.decision {
            CoarseDecision::Prune => (0, 0, false, 0u8),
            _ => (
                ilp_sol.decision.csha_level,
                ilp_sol.decision.adapter_rank,
                ilp_sol.decision.fase_fused,
                ilp_sol.decision.packing_mode,
            ),
        };
        let cost = if coarse.decision == CoarseDecision::Prune {
            0.0
        } else {
            ilp_sol.cost_us
        };
        total += cost;
        peak = peak.max(ilp_sol.memory_bytes);
        layers.push(AppliedLayer {
            layer_index: coarse.layer_index,
            layer_name: coarse.name.clone(),
            coarse: coarse.decision,
            pipeline_stage: coarse.pipeline_stage,
            shard_factor: coarse.shard_params,
            active_heads,
            ffn_width: ilp_sol.decision.ffn_width,
            csha_level,
            adapter_rank,
            optim_m_bits: ilp_sol.decision.optim_m_bits,
            optim_v_bits: ilp_sol.decision.optim_v_bits,
            fase_fused,
            packing_mode,
            estimated_us: cost,
            param_bytes: coarse.param_bytes,
            activation_bytes: coarse.activation_bytes,
        });
    }
    AppliedPlan {
        layers,
        total_us: total,
        peak_memory_bytes: peak,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_cost::{build_lut, LayerShape, LutAxes};
    use crate::wggo_dp::{LayerDecision as CoarseDecision, LayerPlan};
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

    #[test]
    fn applied_layer_carries_param_and_activation_bytes() {
        let lut = build_lut(&toy_shape(), gpu(), &LutAxes::default());
        let ilp: Vec<_> = (0..3)
            .map(|_| solve_layer(&lut, &LayerIlpConstraints::default()))
            .collect();
        let applied = apply(&inter_plan(3), &ilp);
        assert_eq!(applied.layers[0].param_bytes, 6_000_000);
        assert_eq!(applied.layers[0].activation_bytes, 2_000_000);
        assert_eq!(applied.layers[1].param_bytes, 500_000);
        assert_eq!(applied.layers[1].activation_bytes, 300_000);
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
