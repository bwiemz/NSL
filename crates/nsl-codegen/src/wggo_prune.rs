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
pub struct PruneRewriteResult {
    pub rewrites: Vec<PruneRewrite>,
    pub refusals: Vec<PruneRefusal>,
    pub pruned_forward_var_ids: BTreeSet<VarId>,
    pub ops_deleted: usize,
}

/// Record of one layer successfully pruned.
pub struct PruneRewrite {
    pub layer_name: String,
    pub layer_role: LayerRole,
    pub h_before_var: VarId,
    pub h_after_var: VarId,
    pub residual_add_op: OpId,
    pub closure_ops: Vec<OpId>,
}

/// A refusal. One variant per precondition failure enumerated in spec §3.
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
