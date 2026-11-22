//! Sections 6a–6b.5 of the train block's source-AD arm: the adjoint tape
//! optimizations that run between adjoint generation and the CCR splice —
//! the WRGA backward-live filter (6a), dead-gradient elimination and the
//! bit-exact backward folds (6b: SwiGLU gate pairs, RMSNorm dx + residual,
//! the elementwise-chain fuser and the scalar-immediate sweep, plus the
//! RoPE fold witness count) and the `NSL_CSLA_REPORT` schedule report
//! (6b.5).
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 120 lines,
//! 4 inputs ([`AdjointTapeOptInputs`]); a pure tape rewrite — no IR
//! is emitted, the adjoint is updated in place. Returns `adjoint_needed`,
//! the trainable parameter-gradient adjoint VarIds, which the driver's
//! P0.2 gradient-integrity guard reuses over the final (post-CCR) tape.
//! The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`) pin
//! the resulting backward tapes.

use std::collections::HashSet;

use crate::compiler::Compiler;
use crate::wengert::VarId;

/// Every binding of `compile_train_block_inner` the tape optimizations
/// read; names are the driver's.
pub(crate) struct AdjointTapeOptInputs<'a> {
    /// The freshly generated adjoint tape; rewritten in place.
    pub(crate) adjoint: &'a mut crate::wengert::WengertList,
    /// The forward extractor (its named parameter VarIds pick the trainable gradients).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// The adjoint generator (primal → adjoint VarId map).
    pub(crate) generator: &'a crate::source_ad::AdjointGenerator,
    /// The WRGA plan; `Some` carries the backward-live set of the prune pass.
    pub(crate) wrga_plan: &'a Option<crate::wrga::WrgaPlan>,
    /// The forward the adjoint was generated from (the SwiGLU recompute
    /// fold reads its fused ops).
    pub(crate) effective_primal: &'a crate::wengert::WengertList,
    /// The CCR plan, before its splice (`None` without checkpointing).
    pub(crate) ccr_plan: &'a Option<crate::ccr::CcrPlan>,
}

impl Compiler<'_> {
    /// Run the adjoint tape optimizations (see the module header) and
    /// return the trainable parameter-gradient adjoint VarIds.
    pub(crate) fn optimize_adjoint_tape(&self, inputs: AdjointTapeOptInputs<'_>) -> HashSet<VarId> {
        let AdjointTapeOptInputs {
            adjoint,
            extractor,
            generator,
            wrga_plan,
            effective_primal,
            ccr_plan,
        } = inputs;

        // 6a. (removed) The WRGA backward-live filter used to run here and
        // drop every adjoint op outside the WRGA prune pass's live set.
        // That set is built by matching `@wrga` patterns against parameter
        // names, and for `mode = manual` and `mode = hybrid` the patterns
        // (`Class.field`, a bare layer prefix) never match an instance path
        // (`m.b1.wq`): the set came out empty, every adjoint op was
        // dropped, and the train block ran to completion having trained
        // nothing, under source AD only -- tape AD ignored the mode and
        // trained everything. Where the set was right the filter was
        // redundant with step 6b, which prunes to the parameters `@freeze`
        // leaves trainable. So 6b alone decides, in both AD modes; the WRGA
        // plan's live set remains in its report.
        let _ = wrga_plan;

        // 6b. Dead gradient elimination: prune adjoint ops not needed
        // by any parameter gradient. This removes ghost VarId chains
        // from non-differentiable ops (shape, subscript, list) that
        // would cascade skip in the lowerer.
        //
        // `adjoint_needed` (the trainable parameter-gradient adjoint
        // VarIds) is hoisted here so the P0.2 gradient-integrity guard
        // below can reuse it to classify LIVE vs dead/ghost adjoint ops
        // over the FINAL (post-CCR) op list.
        let adjoint_needed: HashSet<VarId> = {
            let named_params = extractor.named_param_var_ids();
            named_params
                .iter()
                .filter(|(name, _)| self.is_trainable_param_name(name))
                .filter_map(|(_, vid)| generator.adjoint_of(*vid))
                .collect()
        };
        if !adjoint_needed.is_empty() {
            adjoint.ops =
                crate::source_ad::eliminate_dead_gradients(&adjoint.ops, &adjoint_needed);
        }
        // P5 item 20 slice B: fuse SwiGLU gate-gradient pairs
        // (bit-exact — see fuse_swiglu_gate_backward).
        let swiglu_fused = crate::source_ad::fuse_swiglu_gate_backward(&mut adjoint.ops, &adjoint_needed);
        if swiglu_fused > 0 {
            nsl_log::nsl_log!(INFO, "fuse", "[fuse] swiglu gate-backward pairs: {swiglu_fused}");
        }
        // The SwiGLU fusion, step 2: where CCR recomputes a fused
        // `swiglu` value only for the down projection's weight gradient,
        // the backward's adjoint launch produces it instead (see
        // `fuse_swiglu_recompute_into_backward`). Only where CCR will serve
        // the remaining shape read with a stub (segment-end frees), will not
        // flip the value back to a save (no byte budget), and does not
        // partition this tape by position (`--layerwise-accum`).
        if let Some(plan) = ccr_plan
            && self.ccr_frees_per_segment()
            && self.compile_options.checkpoint.budget_mib.is_none()
            && !self.compile_options.train.layerwise_accum
        {
            let mut fresh = effective_primal
                .ops
                .iter()
                .chain(adjoint.ops.iter())
                .flat_map(|o| std::iter::once(o.result).chain(o.inputs.iter().copied()))
                .max()
                .map_or(0, |m| m + 1);
            // Only a victim CCR may free: a shape read of any other is still
            // served by a clone (`ccr::backward_reads`), and the fold would then
            // compute `p` twice.
            let foldable: HashSet<VarId> = plan.recompute.intersection(&plan.free_eligible).copied().collect();
            let folded = crate::source_ad::fuse_swiglu_recompute_into_backward(
                adjoint,
                effective_primal,
                &foldable,
                &mut fresh,
            );
            if folded > 0 {
                nsl_log::nsl_log!(INFO, "fuse", "[fuse] swiglu recompute folded into the backward: {folded}");
            }
        }
        // P5 slice C: fold residual-gradient accumulates into fused
        // RMSNorm dx ops (bit-exact; no-op unless
        // --fuse-rmsnorm-backward emitted them).
        let norm_res_folds =
            crate::source_ad::fuse_rmsnorm_dx_residual(&mut adjoint.ops, &adjoint_needed);
        if norm_res_folds > 0 {
            nsl_log::nsl_log!(INFO, "fuse", "[fuse] rmsnorm dx+residual folds: {norm_res_folds}");
        }
        // MFU campaign C2: the RoPE backward fold is generation-time
        // (rotate_half_neg emitted instead of rotate_half + Neg);
        // count the ops here so gates have an anti-vacuity witness.
        let rope_folds = adjoint
            .ops
            .iter()
            .filter(|op| {
                matches!(&op.op,
                    crate::wengert::PrimalOp::Passthrough(n) if n == "rotate_half_neg")
            })
            .count();
        if rope_folds > 0 {
            nsl_log::nsl_log!(INFO, "fuse", "[fuse] rope backward folds: {rope_folds}");
        }
        // MFU campaign C3: generic elementwise-chain fusion + the
        // standalone scalar-immediate sweep. Chain fuser first so
        // chains absorb Constants as immediates; the sweep catches
        // standalone leftovers (reversed order would turn const
        // sites into Passthrough barriers and starve chains). Runs
        // after the specialized folds above so they claim their
        // better patterns first; skipped under --layerwise-accum
        // (the CSLA range partition is positional over this tape —
        // v1 defers, see ew_chain_fusion module docs).
        if !self.compile_options.train.layerwise_accum {
            let ew_stats = crate::ew_chain_fusion::run_backward_ew_fusion(
                &mut adjoint.ops,
                &adjoint_needed,
                &adjoint.var_types,
            );
            if ew_stats.chains > 0 {
                nsl_log::nsl_log!(INFO, "fuse", 
                    "[fuse] elementwise backward chains: {} ({} device ops elided, \
                     {} reduces absorbed, {} imms baked)",
                    ew_stats.chains,
                    ew_stats.device_ops_elided,
                    ew_stats.reduces_absorbed,
                    ew_stats.imms_baked
                );
            }
            let scalar_imms = crate::ew_chain_fusion::rewrite_scalar_immediates(
                &mut adjoint.ops,
                &adjoint_needed,
                &adjoint.var_types,
            );
            if scalar_imms > 0 {
                nsl_log::nsl_log!(INFO, "fuse", "[fuse] scalar immediates: {scalar_imms}");
            }
            if (ew_stats.chains > 0 || scalar_imms > 0)
                && std::env::var("NSL_PROFILE_ADJOINT").is_ok()
            {
                nsl_log::nsl_log!(INFO, "adjoint-profile", 
                    "[adjoint-profile] post-fusion: {} backward ops:",
                    adjoint.ops.len()
                );
                for (k, c) in crate::ew_chain_fusion::histogram(&adjoint.ops) {
                    nsl_log::nsl_log!(INFO, "adjoint-profile", "[adjoint-profile]   {c:>5}  {k}");
                }
            }
        } else {
            nsl_log::nsl_log!(WARN, "fuse", "[fuse] elementwise backward fusion skipped (--layerwise-accum)");
        }

        // 6b.5 CSLA (Milestone B): report the layerwise-accumulation
        // schedule when NSL_CSLA_REPORT=1. Pure analysis over the final
        // adjoint — no codegen change. Element counts are left
        // unquantified here (Stage-2 wires the memory planner's shapes);
        // the layer grouping + tied/cross-layer classification is the
        // correctness-relevant part.
        if std::env::var("NSL_CSLA_REPORT").ok().as_deref() == Some("1") {
            let params: Vec<(String, crate::wengert::VarId)> = extractor
                .named_param_var_ids()
                .iter()
                .filter(|(name, _)| self.is_trainable_param_name(name))
                .map(|(n, v)| (n.clone(), *v))
                .collect();
            let plan = crate::layerwise::analyze(adjoint, &params, &|_| None);
            nsl_log::nsl_log!(INFO, "csla", "[csla]\n{}", plan.render_report("  "));
        }

        adjoint_needed
    }
}
