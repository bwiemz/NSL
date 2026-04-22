//! Layer-level Wengert rewriting driven by WGGO `CoarseDecision::Prune`.
//!
//! Distinct from `wrga_prune.rs`, which handles parameter-level `backward_live`
//! filtering for frozen adapter weights. This module removes whole layer
//! computations from the forward; `wrga_prune` then computes `backward_live`
//! on the already-reduced forward.
//!
//! Pipeline position: runs before `wrga_prune::prune()` in `stmt.rs`, and
//! therefore before source-AD's adjoint generation. The rewrite produces
//! the final forward Wengert that both WRGA Prune and source-AD will consume.
//!
//! Design principle: this module refuses transformations when preconditions
//! aren't met; it does not fall back to weaker transformations with different
//! semantics. See memory/feedback_transformation_precondition_refusal.md for
//! the generalized rule.

use std::collections::BTreeSet;

use crate::wengert::{OpId, VarId, WengertList};
use crate::weight_aware::WeightMap;
use crate::wggo_apply::AppliedPlan;
use crate::wggo_graph::LayerRole;

/// Outcome of `run()`. Either `rewrites` is populated and `refusals` is empty
/// (all prune decisions applied), or `refusals` is populated and `rewrites`
/// is empty (any refusal → nothing applied; `wengert` is unchanged).
#[derive(Debug)]
pub struct PruneRewriteResult {
    pub rewrites: Vec<PruneRewrite>,
    pub refusals: Vec<PruneRefusal>,
    pub pruned_forward_var_ids: BTreeSet<VarId>,
    pub ops_deleted: usize,
}

/// Record of one layer successfully pruned.
#[derive(Debug)]
pub struct PruneRewrite {
    pub layer_name: String,
    pub layer_role: LayerRole,
    pub h_before_var: VarId,
    pub h_after_var: VarId,
    pub residual_add_op: OpId,
    pub closure_ops: Vec<OpId>,
}

/// A refusal. One variant per precondition failure enumerated in spec §3.
#[derive(Debug)]
pub enum PruneRefusal {
    CrossLayerParam {
        layer_name: String,
        layer_role: LayerRole,
        param_name: String,
        param_var: VarId,
        external_consumer: OpId,
        external_op_kind: String,
    },
    NoResidualAdd {
        layer_name: String,
        layer_role: LayerRole,
        closure_size: usize,
    },
    ParallelResidualBranches {
        layer_name: String,
        layer_role: LayerRole,
        add_ops: Vec<OpId>,
    },
    AmbiguousPatternMatch {
        layer_name: String,
        layer_role: LayerRole,
        h_before_var: VarId,
        candidate_adds: Vec<OpId>,
    },
    EmptyClosure {
        layer_name: String,
        layer_role: LayerRole,
        prefix: String,
    },
    WholeBlockUnsupported {
        layer_name: String,
    },
    ConflictingPruneDecisions {
        decision_a: String,
        decision_b: String,
        reason: String,
    },
}

/// Entry point. Dry-run-then-commit: validates all decisions first; applies
/// mutations only if all pass. On refusal, `wengert` is unchanged.
///
/// See spec §5.3 for the three-phase contract.
pub fn run(
    _wengert: &mut WengertList,
    _applied_plan: &AppliedPlan,
    _weight_map: &WeightMap,
) -> PruneRewriteResult {
    // Stub: Phase 1/2/3 land in Tasks 4–13.
    PruneRewriteResult {
        rewrites: Vec::new(),
        refusals: Vec::new(),
        pruned_forward_var_ids: BTreeSet::new(),
        ops_deleted: 0,
    }
}

// --- Internal Phase 1 validator types ---

/// Internal Phase 1 result: either a validated plan ready to commit, or
/// a refusal. Not `pub` — only `run()` uses it.
#[derive(Debug)]
pub(crate) enum PlanResult {
    Ok(PruneRewritePlan),
    Refused(PruneRefusal),
}

/// Internal Phase 1 output. Carries everything `apply_rewrite` needs to
/// commit the mutation without re-computing anything.
#[derive(Debug)]
pub(crate) struct PruneRewritePlan {
    pub(crate) layer_name: String,
    pub(crate) layer_role: LayerRole,
    pub(crate) closure_op_ids: Vec<OpId>,    // deleted in Phase 3 (sorted in wengert order)
    pub(crate) residual_add_op_id: OpId,     // rewritten then deleted
    pub(crate) h_before_var: VarId,
    pub(crate) h_after_var: VarId,
    pub(crate) parameter_var_ids: std::collections::BTreeSet<VarId>,
}

/// Intermediate refusal emitted by `find_residual_add` before context is
/// bound by the caller. `plan_rewrite` wraps into a `PruneRefusal`.
#[derive(Debug)]
enum PartialRefusal {
    NoResidualAdd,
    // Task 6 will add ParallelResidualBranches { add_ops: Vec<OpId> }
    // Task 7 will add AmbiguousPatternMatch { h_before: VarId, candidate_adds: Vec<OpId> }
}

// --- Phase 1 validator (positive case only — refusals land in Tasks 5-11) ---

/// Phase 1 validator for a single `CoarseDecision::Prune` decision. Does
/// NOT mutate `wengert`. Called once per Prune decision from `run()`.
///
/// Spec §2 (closure), §1.3 (pattern-match), §3 (refusals).
pub(crate) fn plan_rewrite(
    wengert: &WengertList,
    layer: &crate::wggo_apply::AppliedLayer,
    _weight_map: &WeightMap,
) -> PlanResult {
    use crate::wggo_graph::infer_role;
    use std::collections::BTreeSet;

    let layer_role = infer_role(&layer.layer_name);

    // Spec §3.6 — Task 10 will land this refusal path; for Task 4 the
    // positive case assumes non-Block roles. Leaving a TODO marker so Task 10
    // can slot in without re-shaping the function.

    // (b) Find parameter VarIds matching `{layer_name}.` prefix.
    //     Spec §2.3 precondition #1 (non-empty parameters) lands in Task 8.
    let prefix = format!("{}.", layer.layer_name);
    let parameter_var_ids: BTreeSet<VarId> = wengert
        .var_names
        .iter()
        .filter_map(|(v, name)| name.starts_with(&prefix).then_some(*v))
        .collect();

    // (c) Compute the data-flow closure.
    let closure_op_ids = compute_closure(wengert, &parameter_var_ids);

    // (d) Find the residual Add. Positive case only — Tasks 5-7 extend with
    //     NoResidualAdd, ParallelResidualBranches, AmbiguousPatternMatch.
    //     Task 9 (§3.1 CrossLayerParam) fires BEFORE this step; it doesn't
    //     exist yet — lands there.
    let (residual_add_op_id, h_before_var, h_after_var) =
        match find_residual_add(wengert, &closure_op_ids, &parameter_var_ids) {
            Ok(triple) => triple,
            Err(partial) => {
                return PlanResult::Refused(refusal_with_context(
                    partial,
                    layer,
                    layer_role,
                    closure_op_ids.len(),
                ));
            }
        };

    PlanResult::Ok(PruneRewritePlan {
        layer_name: layer.layer_name.clone(),
        layer_role,
        closure_op_ids,
        residual_add_op_id,
        h_before_var,
        h_after_var,
        parameter_var_ids,
    })
}

/// Compute the transitive forward-closure of ops owned by the layer.
/// Spec §2.2.
///
/// Returns `OpId`s in topological order (same order as `wengert.ops`).
pub(crate) fn compute_closure(
    wengert: &WengertList,
    param_var_ids: &std::collections::BTreeSet<VarId>,
) -> Vec<OpId> {
    use crate::wengert::PrimalOp;
    use std::collections::BTreeSet;

    // Tainted VarIds: layer-N params OR outputs of closure ops.
    let mut tainted_vars: BTreeSet<VarId> = param_var_ids.clone();
    let mut closure: Vec<OpId> = Vec::new();

    for op in &wengert.ops {
        let produces_param = param_var_ids.contains(&op.result);
        let reads_tainted = op.inputs.iter().any(|v| tainted_vars.contains(v));

        if !(produces_param || reads_tainted) {
            continue;
        }

        // Residual Add check: Add(tainted, untainted) means one input is
        // block_output (tainted) and the other is h_before (untainted).
        // This op is the BOUNDARY — EXCLUDED from the closure.
        if matches!(op.op, PrimalOp::Add) && op.inputs.len() == 2 {
            let a = op.inputs[0];
            let b = op.inputs[1];
            let a_tainted = tainted_vars.contains(&a);
            let b_tainted = tainted_vars.contains(&b);
            if a_tainted != b_tainted {
                // Boundary — don't include, don't taint result.
                continue;
            }
        }

        closure.push(op.id);
        tainted_vars.insert(op.result);
    }

    closure
}

/// Positive-case pattern-match: find THE residual Add.
///
/// Task 4 handles the single-match case. Tasks 5-7 extend:
/// - Task 5: zero matches → NoResidualAdd
/// - Task 6: multiple matches with distinct h_before → ParallelResidualBranches
/// - Task 7: multiple matches with shared h_before → AmbiguousPatternMatch
fn find_residual_add(
    wengert: &WengertList,
    closure: &[OpId],
    param_var_ids: &std::collections::BTreeSet<VarId>,
) -> Result<(OpId, VarId, VarId), PartialRefusal> {
    use crate::wengert::PrimalOp;
    use std::collections::BTreeSet;

    let closure_set: BTreeSet<OpId> = closure.iter().copied().collect();

    // Rebuild the tainted set from parameters + closure op outputs.
    let tainted: BTreeSet<VarId> = {
        let mut t: BTreeSet<VarId> = param_var_ids.clone();
        for op in &wengert.ops {
            if closure_set.contains(&op.id) {
                t.insert(op.result);
            }
        }
        t
    };

    // Find the first Add(tainted, untainted) outside the closure.
    for op in &wengert.ops {
        if closure_set.contains(&op.id) { continue; }
        if !matches!(op.op, PrimalOp::Add) { continue; }
        if op.inputs.len() != 2 { continue; }

        let a = op.inputs[0];
        let b = op.inputs[1];
        let a_tainted = tainted.contains(&a);
        let b_tainted = tainted.contains(&b);
        if a_tainted != b_tainted {
            let h_before = if a_tainted { b } else { a };
            return Ok((op.id, h_before, op.result));
        }
    }

    Err(PartialRefusal::NoResidualAdd)
}

/// Wrap a partial refusal in the caller's context. Task 5-7 extend with
/// more variants.
fn refusal_with_context(
    partial: PartialRefusal,
    layer: &crate::wggo_apply::AppliedLayer,
    layer_role: LayerRole,
    closure_size: usize,
) -> PruneRefusal {
    match partial {
        PartialRefusal::NoResidualAdd => PruneRefusal::NoResidualAdd {
            layer_name: layer.layer_name.clone(),
            layer_role,
            closure_size,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use crate::wggo_apply::{AppliedLayer, AppliedPlan};
    use crate::wggo_dp::LayerDecision;
    use crate::weight_aware::WeightMap;
    use std::collections::HashMap;

    /// Build a minimal synthetic WengertList for unit tests.
    fn mk_wengert(ops: Vec<WengertOp>, output: VarId, var_names: &[(VarId, &str)]) -> WengertList {
        WengertList {
            ops,
            output,
            var_names: var_names.iter().map(|(v, s)| (*v, s.to_string())).collect(),
            var_types: HashMap::new(),
        }
    }

    /// Shorthand: unary op with one input.
    fn op_unary(id: OpId, result: VarId, input: VarId, kind: PrimalOp) -> WengertOp {
        WengertOp {
            id, result, op: kind, inputs: vec![input],
            saved_for_backward: false, checkpointed: false,
        }
    }

    /// Shorthand: Add op.
    fn op_add(id: OpId, result: VarId, a: VarId, b: VarId) -> WengertOp {
        WengertOp {
            id, result, op: PrimalOp::Add, inputs: vec![a, b],
            saved_for_backward: false, checkpointed: false,
        }
    }

    /// Build a minimal AppliedLayer for Prune with a given name.
    /// (Role is inferred from layer_name via wggo_graph::infer_role inside
    /// plan_rewrite — no layer_role field exists on AppliedLayer.)
    fn mk_prune_layer(idx: u32, name: &str) -> AppliedLayer {
        AppliedLayer {
            layer_index: idx,
            layer_name: name.to_string(),
            coarse: LayerDecision::Prune,
            pipeline_stage: 0,
            shard_factor: 1,
            active_heads: 0,
            ffn_width: 0,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 0,
            optim_v_bits: 0,
            fase_fused: false,
            packing_mode: 0,
            estimated_us: 0.0,
            param_bytes: 0,
            activation_bytes: 0,
        }
    }

    #[test]
    fn closure_captures_transitive_compute_ops() {
        // Wengert list modeling: h_after = h_before + (h_before * blocks.7.attn.wq)
        //   op0: Mul v0 = v_hb * v_hb      → v0 is the "param producer" (v0 is named blocks.7.attn.wq)
        //   op1: Mul v1 = v0 * v0          → block_output (reads layer-7 param)
        //   op2: Add v_ha = v_hb + v1      → residual boundary
        //
        // Expected closure: {op0, op1}. The Add (op2) is the BOUNDARY, not the closure.

        let v_hb: VarId = 100;
        let v0:   VarId = 200;
        let v1:   VarId = 201;
        let v_ha: VarId = 202;

        let ops = vec![
            op_unary(0, v0, v_hb, PrimalOp::Mul),   // produces v0 (a layer-7 param VarId)
            op_unary(1, v1, v0, PrimalOp::Mul),     // reads layer-7 param → block_output
            op_add(2, v_ha, v_hb, v1),               // residual Add — the boundary
        ];

        let wengert = mk_wengert(
            ops,
            v_ha,
            &[(v_hb, "h_before"), (v0, "blocks.7.attn.wq"), (v_ha, "h_after")],
        );

        let layer = mk_prune_layer(7, "blocks.7.attn");
        let weight_map = WeightMap::default();

        let result = plan_rewrite(&wengert, &layer, &weight_map);

        match result {
            PlanResult::Ok(plan) => {
                assert_eq!(plan.closure_op_ids, vec![0, 1], "closure should include op0 (param producer) and op1 (compute), NOT op2 (residual Add)");
                assert_eq!(plan.residual_add_op_id, 2);
                assert_eq!(plan.h_before_var, v_hb);
                assert_eq!(plan.h_after_var, v_ha);
            }
            PlanResult::Refused(r) => panic!("expected Ok, got Refused({r:?})"),
        }
    }
}
