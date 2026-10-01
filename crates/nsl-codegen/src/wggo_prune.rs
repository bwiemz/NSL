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
//! Two shapes of prune are executed (spec
//! `docs/superpowers/specs/2026-04-22-wggo-prune-ir-rewrite-design.md`):
//!
//! - **v1, sub-block** (`LayerRole::{Attention, Ffn}` and the other non-Block
//!   roles): exactly one residual `Add(h_before, block_output)` bounds the
//!   layer's closure; the closure and the Add are deleted and `h_after` is
//!   aliased to `h_before`.
//! - **v2, whole block — chain-collapse** (`LayerRole::Block`, `blocks.N` /
//!   `layers.N` / `h.N`): a pre-norm block threads the residual stream
//!   through k ≥ 1 Adds, `h0 → Add(h0, out1) = h1 → … → Add(h(k-1), outk) =
//!   hk`. The Adds must form ONE chain, every intermediate `h1..h(k-1)` must be
//!   read only by the block's own ops and the next Add, and every `out_i` only
//!   by its own Add; the closure and all k Adds are then deleted and `hk` is
//!   aliased to `h0`.
//!
//! A parameter belongs to layer `L` when its Wengert var name starts with
//! `"{L}."`, either directly or after the model variable (`m.blocks.0.wq`
//! belongs to `blocks.0`) — the source-AD extractor names every model field
//! by its full access path, while WGGO names layers bare.
//!
//! Design principle: this module refuses transformations when preconditions
//! aren't met; it does not fall back to weaker transformations with different
//! semantics. See memory/feedback_transformation_precondition_refusal.md for
//! the generalized rule.

use std::collections::{BTreeMap, BTreeSet};

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
    /// The residual stream value entering the layer (`h0`). Consumers of
    /// `h_after_var` now read it (or, when an adjacent earlier layer was
    /// pruned in the same plan, whatever that layer's input resolved to).
    pub h_before_var: VarId,
    /// The residual stream value leaving the layer (`hk`, the LAST chain
    /// Add's result).
    pub h_after_var: VarId,
    /// The last residual Add of the chain — the one producing `h_after_var`.
    /// Equal to `residual_add_ops.last()`; kept as its own field because the
    /// spec §6.1 success line reports exactly this op.
    pub residual_add_op: OpId,
    /// Every residual Add deleted, in stream order: one for a sub-block,
    /// k for a whole block.
    pub residual_add_ops: Vec<OpId>,
    pub closure_ops: Vec<OpId>,
    /// Ops this rewrite actually removed from wengert (measured as the
    /// list's shrink: `closure_ops.len()` plus one per residual Add when the
    /// commit is correct). Tracked per-rewrite to enable spec §6.1 stderr
    /// emission without ambiguity on multi-rewrite plans.
    pub ops_deleted: usize,
}

/// A refusal. One variant per precondition failure enumerated in spec §3,
/// plus the v2 chain-collapse precondition (`BrokenResidualChain`).
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
    /// v2 whole-block prune: the block's residual Adds do not form one
    /// collapsible stream chain (they do not chain, an intermediate stream
    /// value escapes the block, or a block output feeds something other
    /// than its own Add). `reason` names the first violation found.
    BrokenResidualChain {
        layer_name: String,
        layer_role: LayerRole,
        adds: Vec<OpId>,
        reason: String,
    },
    ConflictingPruneDecisions {
        decision_a: String,
        decision_b: String,
        reason: String,
    },
}

/// Whether the Wengert var `var_name` belongs to layer `layer_name`: it
/// starts with `"{layer_name}."`, either as written or after its first
/// dot-component (the model variable — the source-AD extractor names a
/// model field `m.blocks.0.wq`, while WGGO names the layer `blocks.0`).
///
/// Only ONE leading component is stripped: a field nested deeper
/// (`m.encoder.blocks.0.wq`) does not match `blocks.0`, so such a prune
/// refuses with `EmptyClosure` rather than guessing which `blocks` was meant.
pub fn var_in_layer(var_name: &str, layer_name: &str) -> bool {
    let has_layer_prefix = |s: &str| {
        s.len() > layer_name.len()
            && s.starts_with(layer_name)
            && s.as_bytes()[layer_name.len()] == b'.'
    };
    if has_layer_prefix(var_name) {
        return true;
    }
    var_name
        .split_once('.')
        .is_some_and(|(_, rest)| has_layer_prefix(rest))
}

/// Entry point. Dry-run-then-commit: validates all decisions first; applies
/// mutations only if all pass. On refusal, `wengert` is unchanged and
/// `rewrites` is empty.
///
/// See spec §5.3 for the three-phase contract.
pub fn run(
    wengert: &mut WengertList,
    applied_plan: &AppliedPlan,
    weight_map: &WeightMap,
) -> PruneRewriteResult {
    use crate::wggo_dp::CoarseDecision;

    // Phase 1: validate each Prune decision without mutating wengert.
    let mut plans: Vec<PruneRewritePlan> = Vec::new();
    let mut refusals: Vec<PruneRefusal> = Vec::new();
    for layer in &applied_plan.layers {
        if !matches!(layer.coarse, CoarseDecision::Prune) {
            continue;
        }
        match plan_rewrite(wengert, layer, weight_map) {
            PlanResult::Ok(plan) => plans.push(plan),
            PlanResult::Refused(refusal) => refusals.push(refusal),
        }
    }

    // Phase 1b: cross-plan conflict detection (spec §3.7). Two plans
    // conflict if they claim overlapping OpIds (same ops would be deleted
    // twice — closure ops OR residual Adds) OR target the same h_after_var
    // (VarId aliasing undefined).
    if refusals.is_empty() {
        'outer: for i in 0..plans.len() {
            for j in (i + 1)..plans.len() {
                let a = &plans[i];
                let b = &plans[j];
                let a_ops = a.deleted_op_ids();
                let b_ops = b.deleted_op_ids();
                let overlap: Vec<OpId> = a_ops.intersection(&b_ops).copied().collect();
                if !overlap.is_empty() {
                    refusals.push(PruneRefusal::ConflictingPruneDecisions {
                        decision_a: a.layer_name.clone(),
                        decision_b: b.layer_name.clone(),
                        reason: format!(
                            "closures overlap on ops: {:?} (same ops would be deleted by both rewrites)",
                            overlap
                        ),
                    });
                    break 'outer;
                }
                if a.h_after_var == b.h_after_var {
                    refusals.push(PruneRefusal::ConflictingPruneDecisions {
                        decision_a: a.layer_name.clone(),
                        decision_b: b.layer_name.clone(),
                        reason: format!(
                            "both rewrites target the same h_after VarId {:?}; aliasing is undefined",
                            a.h_after_var
                        ),
                    });
                    break 'outer;
                }
            }
        }
    }

    // Phase 2: early-return on any refusal. Wengert stays untouched.
    if !refusals.is_empty() {
        return PruneRewriteResult {
            rewrites: Vec::new(),
            refusals,
            pruned_forward_var_ids: BTreeSet::new(),
            ops_deleted: 0,
        };
    }

    // Phase 3: commit all plans. Captures pruned VarIds BEFORE the mutation
    // so we can return them to the caller for WRGA / source-AD handoff.
    //
    // `aliases` records every `h_after → h_before` collapse committed so
    // far. Two ADJACENT layers in one plan share a stream value — the first
    // layer's output is the second's input — and every plan was validated
    // against the unmutated list, so the second plan's `h_before_var` names
    // a value the first commit deleted. Resolving through the alias map
    // repoints the second layer's consumers at the first layer's input
    // instead of at a dangling VarId, in whichever order the plans commit.
    let mut aliases: BTreeMap<VarId, VarId> = BTreeMap::new();
    let mut rewrites: Vec<PruneRewrite> = Vec::with_capacity(plans.len());
    let mut pruned_forward_var_ids: BTreeSet<VarId> = BTreeSet::new();
    let mut ops_deleted: usize = 0;
    for plan in plans {
        // Capture pruned VarIds BEFORE the mutation (apply_rewrite will
        // delete ops and lose this info): every closure op's result and
        // every residual Add's result (h1..hk).
        let deleted = plan.deleted_op_ids();
        for o in wengert.ops.iter().filter(|o| deleted.contains(&o.id)) {
            pruned_forward_var_ids.insert(o.result);
        }

        let rewrite = apply_rewrite(wengert, plan, &mut aliases);
        ops_deleted += rewrite.ops_deleted;
        rewrites.push(rewrite);
    }
    // Deletion never renumbers here (that stability is what id-space
    // references rely on), so it cannot CREATE a duplicate — this assert is
    // the commit-point belt every in-place structural mutator wears, so a
    // future rewrite that mints ids cannot silently break the uniqueness
    // the claim tables assume.
    wengert.assert_unique_op_ids("wggo_prune::run (post-commit)");
    // The same belt for the rewrite's own contract: nothing that survived
    // may read a value the commit deleted. Phase 1 proved every reader of a
    // deleted value is itself deleted or repointed; a violation is a
    // compiler bug, and lowering would otherwise skip the ghost read
    // silently (a dead op over deleted inputs lowers to nothing).
    let dangling: Vec<(OpId, VarId)> = wengert
        .ops
        .iter()
        .flat_map(|o| {
            o.inputs
                .iter()
                .filter(|v| pruned_forward_var_ids.contains(v))
                .map(move |v| (o.id, *v))
        })
        .collect();
    assert!(
        dangling.is_empty() && !pruned_forward_var_ids.contains(&wengert.output),
        "wggo_prune::run (post-commit): surviving (op, VarId) reads of pruned values {dangling:?}; \
         output VarId {} pruned: {}",
        wengert.output,
        pruned_forward_var_ids.contains(&wengert.output)
    );

    PruneRewriteResult {
        rewrites,
        refusals: Vec::new(),
        pruned_forward_var_ids,
        ops_deleted,
    }
}

/// Follow `aliases` from `v` to the value it now stands for.
fn resolve_alias(aliases: &BTreeMap<VarId, VarId>, mut v: VarId) -> VarId {
    // Each committed collapse maps a deleted value to one that is still
    // live or itself aliased further upstream; the walk is bounded by the
    // number of commits (a cycle would need a layer whose output feeds its
    // own input, which a topologically ordered list cannot contain).
    let mut hops = 0usize;
    while let Some(&next) = aliases.get(&v) {
        v = next;
        hops += 1;
        assert!(
            hops <= aliases.len(),
            "wggo_prune: residual alias cycle through VarId {v}"
        );
    }
    v
}

/// Phase 3 mutation. Deletes the closure ops and every residual Add of the
/// chain, and repoints consumers of the chain's final `h_after` (and
/// `wengert.output`, if it was that value) to the layer's input stream
/// value. Also cleans up stale var_names / var_types entries.
///
/// Spec §1.1 / §2.2 three-category treatment:
///   - closure ops → DELETED
///   - residual Add(s) → REWRITTEN (consumers of the last one repointed)
///     then DELETED
///   - h_before → UNTOUCHED (belongs to the prior stream)
///
/// The intermediate stream values h1..h(k-1) of a whole-block chain need no
/// repointing: Phase 1 proved their only readers are closure ops and the
/// next chain Add, all of which are deleted here.
fn apply_rewrite(
    wengert: &mut WengertList,
    plan: PruneRewritePlan,
    aliases: &mut BTreeMap<VarId, VarId>,
) -> PruneRewrite {
    let to_delete = plan.deleted_op_ids();
    let target = resolve_alias(aliases, plan.h_before_var);

    // Repoint every surviving op's inputs from h_after_var → the layer's
    // (alias-resolved) input.
    for op in wengert.ops.iter_mut() {
        if to_delete.contains(&op.id) {
            continue;
        }
        for input in op.inputs.iter_mut() {
            if *input == plan.h_after_var {
                *input = target;
            }
        }
    }
    // Repoint wengert.output too, if it pointed at h_after.
    if wengert.output == plan.h_after_var {
        wengert.output = target;
    }
    aliases.insert(plan.h_after_var, target);

    // Delete closure ops + residual Adds from wengert.ops. `ops_deleted` is
    // what actually left the list (spec §6.1: "ops actually removed"), not
    // the plan's count — the two can only differ through a bug, and the
    // success line is where that would show.
    let before = wengert.ops.len();
    wengert.ops.retain(|op| !to_delete.contains(&op.id));
    let ops_deleted = before - wengert.ops.len();

    // Prune stale var_names / var_types for VarIds that no surviving op produces.
    // (h_before_var survives because it's produced by an upstream op outside the
    // closure OR is an initial input — either way, keep its entry.)
    let surviving_var_ids: BTreeSet<VarId> = wengert.ops.iter().map(|o| o.result).collect();
    wengert.var_names.retain(|v, _| surviving_var_ids.contains(v) || *v == wengert.output);
    wengert.var_types.retain(|v, _| surviving_var_ids.contains(v) || *v == wengert.output);

    let residual_add_op = *plan
        .residual_add_op_ids
        .last()
        .expect("Phase 1 only builds plans with at least one residual Add");
    PruneRewrite {
        layer_name: plan.layer_name,
        layer_role: plan.layer_role,
        h_before_var: plan.h_before_var,
        h_after_var: plan.h_after_var,
        residual_add_op,
        residual_add_ops: plan.residual_add_op_ids,
        closure_ops: plan.closure_op_ids,
        ops_deleted,
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
    /// The residual Adds, in stream order (exactly one for a sub-block;
    /// k ≥ 1 for a whole block). All are deleted; consumers of the last
    /// one's result (`h_after_var`) are repointed.
    pub(crate) residual_add_op_ids: Vec<OpId>,
    pub(crate) h_before_var: VarId,
    pub(crate) h_after_var: VarId,
    // Populated by Phase 1's residual-add resolver. Phase 3 deletes by op-id
    // and does not consume it, and the diagnostic/leak-detection readers the
    // original note anticipated do not exist yet — so this is a placeholder
    // with a producer, not a live channel.
    #[allow(dead_code)]
    pub(crate) parameter_var_ids: std::collections::BTreeSet<VarId>,
}

impl PruneRewritePlan {
    /// Every op this plan deletes: the closure plus the residual Adds.
    fn deleted_op_ids(&self) -> BTreeSet<OpId> {
        self.closure_op_ids
            .iter()
            .chain(self.residual_add_op_ids.iter())
            .copied()
            .collect()
    }
}

/// One residual-Add candidate: an op outside the closure computing
/// `Add(h_before, block_output)` with exactly one tainted operand.
#[derive(Debug, Clone, Copy)]
struct ResidualCandidate {
    add_op: OpId,
    /// The untainted (stream) operand.
    h_before: VarId,
    /// The tainted operand — the (sub-)block's contribution.
    block_output: VarId,
    /// The Add's result.
    h_after: VarId,
}

/// Intermediate refusal emitted by the residual resolvers before context
/// is bound by the caller. `plan_rewrite` wraps into a `PruneRefusal`.
#[derive(Debug)]
enum PartialRefusal {
    NoResidualAdd,
    ParallelResidualBranches { add_ops: Vec<OpId> },
    AmbiguousPatternMatch { h_before: VarId, candidate_adds: Vec<OpId> },
    BrokenChain { adds: Vec<OpId>, reason: String },
}

// --- Phase 1 validator ---

/// Phase 1 validator for a single `CoarseDecision::Prune` decision. Does
/// NOT mutate `wengert`. Called once per Prune decision from `run()`.
///
/// Spec §2 (closure), §1.3 (pattern-match), §3 (refusals); the
/// `LayerRole::Block` branch is the v2 chain-collapse (module docs).
pub(crate) fn plan_rewrite(
    wengert: &WengertList,
    layer: &crate::wggo_apply::AppliedLayer,
    _weight_map: &WeightMap,
) -> PlanResult {
    use crate::wggo_graph::infer_role;

    let layer_role = infer_role(&layer.layer_name);
    let whole_block = matches!(layer_role, LayerRole::Block);

    // (b) Find parameter VarIds belonging to the layer (`{layer_name}.`
    //     prefix, optionally behind the model variable — see `var_in_layer`).
    let prefix = format!("{}.", layer.layer_name);
    let parameter_var_ids: BTreeSet<VarId> = wengert
        .var_names
        .iter()
        .filter_map(|(v, name)| var_in_layer(name, &layer.layer_name).then_some(*v))
        .collect();

    // Spec §2.3 precondition #1 / §3.5: if no VarIds match the layer prefix,
    // the prune target doesn't exist (typo, off-by-one index, or layer not
    // instantiated in the compiled model).
    if parameter_var_ids.is_empty() {
        return PlanResult::Refused(PruneRefusal::EmptyClosure {
            layer_name: layer.layer_name.clone(),
            layer_role,
            prefix,
        });
    }

    // (c) Compute the data-flow closure.
    let closure_op_ids = compute_closure(wengert, &parameter_var_ids);

    // (d) Resolve the residual boundary: one Add for a sub-block, a chain
    //     of Adds for a whole block.
    let resolved = if whole_block {
        find_residual_chain(wengert, &closure_op_ids, &parameter_var_ids)
    } else {
        find_residual_add(wengert, &closure_op_ids, &parameter_var_ids).map(|c| vec![c])
    };
    let chain = match resolved {
        Ok(chain) => chain,
        Err(partial) => {
            return PlanResult::Refused(refusal_with_context(
                partial,
                layer,
                layer_role,
                closure_op_ids.len(),
            ));
        }
    };
    let residual_add_op_ids: Vec<OpId> = chain.iter().map(|c| c.add_op).collect();
    let chain_adds: BTreeSet<OpId> = residual_add_op_ids.iter().copied().collect();
    let h_before_var = chain.first().expect("resolver returns a non-empty chain").h_before;
    let h_after_var = chain.last().expect("resolver returns a non-empty chain").h_after;

    // Spec §2.3 precondition #2 / §3.1: detect leaks out of the closure.
    //
    // A "leak" is any closure op whose result escapes the closure without
    // going through a residual Add of the chain. There are two forms:
    //
    //   (a) A non-closure op (other than a chain Add) reads a closure op's
    //       result.
    //   (b) wengert.output is a closure op's result (the chain's final
    //       `h_after_var` is not a closure op result, so it never matches).
    //
    // Prefer to cite layer-N parameter VarIds when the leaked value is one
    // (matches the spec's "cross-layer parameter sharing" framing); fall
    // back to the generic closure-op result otherwise.
    {
        let closure_set: BTreeSet<OpId> = closure_op_ids.iter().copied().collect();

        // (a) Scan for external readers.
        for closure_op_id in &closure_op_ids {
            let closure_op = wengert.ops.iter().find(|o| o.id == *closure_op_id)
                .expect("closure op id missing from wengert.ops");
            let result_var = closure_op.result;

            for other_op in &wengert.ops {
                if closure_set.contains(&other_op.id) { continue; }
                if chain_adds.contains(&other_op.id) { continue; }
                if !other_op.inputs.contains(&result_var) { continue; }

                // Leak detected. Choose citation VarId: prefer the param itself
                // if the closure op's result is a layer-N param VarId.
                let (param_name, param_var) = if parameter_var_ids.contains(&result_var) {
                    (
                        wengert.var_names.get(&result_var).cloned().unwrap_or_default(),
                        result_var,
                    )
                } else {
                    // Find a source param (input of this closure op that is in params).
                    let sibling_param = closure_op.inputs.iter()
                        .find(|v| parameter_var_ids.contains(v))
                        .copied();
                    match sibling_param {
                        Some(p) => (
                            wengert.var_names.get(&p).cloned().unwrap_or_default(),
                            p,
                        ),
                        None => (
                            wengert.var_names.get(&result_var).cloned()
                                .unwrap_or_else(|| format!("v{result_var}")),
                            result_var,
                        ),
                    }
                };

                return PlanResult::Refused(PruneRefusal::CrossLayerParam {
                    layer_name: layer.layer_name.clone(),
                    layer_role,
                    param_name,
                    param_var,
                    external_consumer: other_op.id,
                    external_op_kind: format!("{:?}", other_op.op),
                });
            }
        }

        // (b) wengert.output is a closure op's result.
        if wengert.output != h_after_var {
            for closure_op_id in &closure_op_ids {
                let closure_op = wengert.ops.iter().find(|o| o.id == *closure_op_id)
                    .expect("closure op id missing from wengert.ops");
                if closure_op.result == wengert.output {
                    // Global escape leak.
                    let sibling_param = closure_op.inputs.iter()
                        .find(|v| parameter_var_ids.contains(v))
                        .copied();
                    let (param_name, param_var) = match sibling_param {
                        Some(p) => (
                            wengert.var_names.get(&p).cloned().unwrap_or_default(),
                            p,
                        ),
                        None => (
                            wengert.var_names.get(&wengert.output).cloned()
                                .unwrap_or_else(|| format!("v{}", wengert.output)),
                            wengert.output,
                        ),
                    };
                    return PlanResult::Refused(PruneRefusal::CrossLayerParam {
                        layer_name: layer.layer_name.clone(),
                        layer_role,
                        param_name,
                        param_var,
                        external_consumer: closure_op.id,
                        external_op_kind: format!("{:?} (produces wengert.output)", closure_op.op),
                    });
                }
            }
        }
    }

    PlanResult::Ok(PruneRewritePlan {
        layer_name: layer.layer_name.clone(),
        layer_role,
        closure_op_ids,
        residual_add_op_ids,
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

/// Collect every residual-Add candidate, in wengert (= topological) order:
/// ops outside the closure computing `Add(a, b)` with exactly one of `a`,
/// `b` tainted (a layer parameter or a closure op's result).
fn residual_candidates(
    wengert: &WengertList,
    closure: &[OpId],
    param_var_ids: &BTreeSet<VarId>,
) -> Vec<ResidualCandidate> {
    use crate::wengert::PrimalOp;

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

    let mut candidates: Vec<ResidualCandidate> = Vec::new();
    for op in &wengert.ops {
        if closure_set.contains(&op.id) { continue; }
        if !matches!(op.op, PrimalOp::Add) { continue; }
        if op.inputs.len() != 2 { continue; }

        let a = op.inputs[0];
        let b = op.inputs[1];
        let a_tainted = tainted.contains(&a);
        let b_tainted = tainted.contains(&b);

        if a_tainted != b_tainted {
            let (h_before, block_output) = if a_tainted { (b, a) } else { (a, b) };
            candidates.push(ResidualCandidate {
                add_op: op.id,
                h_before,
                block_output,
                h_after: op.result,
            });
        }
    }
    candidates
}

/// v1 (sub-block) pattern-match: exactly one residual Add candidate.
///
/// Spec §1.3 / §3.2 / §3.3 / §3.4. Returns:
/// - `Ok(candidate)` when exactly one candidate matches the residual
///   pattern Add(h_before, block_output).
/// - `Err(PartialRefusal::NoResidualAdd)` when zero candidates match.
/// - `Err(PartialRefusal::ParallelResidualBranches)` when ≥2 candidates
///   have DISTINCT h_before values (parallel residual paths).
/// - `Err(PartialRefusal::AmbiguousPatternMatch)` when ≥2 candidates share
///   the SAME h_before (architecturally ambiguous boundary).
fn find_residual_add(
    wengert: &WengertList,
    closure: &[OpId],
    param_var_ids: &BTreeSet<VarId>,
) -> Result<ResidualCandidate, PartialRefusal> {
    let candidates = residual_candidates(wengert, closure, param_var_ids);
    match candidates.len() {
        0 => Err(PartialRefusal::NoResidualAdd),
        1 => Ok(candidates[0]),
        _ => Err(multi_candidate_refusal(&candidates)),
    }
}

/// ≥2 candidates that a sub-block cannot accept: all sharing one `h_before`
/// is an ambiguous boundary, otherwise they are parallel branches.
fn multi_candidate_refusal(candidates: &[ResidualCandidate]) -> PartialRefusal {
    let first_h_before = candidates[0].h_before;
    if candidates.iter().all(|c| c.h_before == first_h_before) {
        PartialRefusal::AmbiguousPatternMatch {
            h_before: first_h_before,
            candidate_adds: candidates.iter().map(|c| c.add_op).collect(),
        }
    } else {
        PartialRefusal::ParallelResidualBranches {
            add_ops: candidates.iter().map(|c| c.add_op).collect(),
        }
    }
}

/// v2 (whole-block) pattern-match: the residual Add candidates must form ONE
/// stream chain `h0 → h1 → … → hk`. Returns the chain in stream order.
///
/// Preconditions, each refusing with `BrokenChain` naming the violation:
/// 1. **One chain** — ordered by position, candidate i's stream operand is
///    candidate i-1's result. (The first candidate's stream operand, h0, is
///    untainted by construction, so it is produced outside the closure.)
/// 2. **No escaping intermediate** — every h1..h(k-1) is read only by closure
///    ops and by the NEXT chain Add, and is not `wengert.output`. An outside
///    reader is a skip connection (or a parameter-free op on the stream that
///    the parameter-anchored closure does not own): deleting the Add that
///    produces it would leave that reader dangling, and repointing it to h0
///    would silently change what it computes.
/// 3. **Single-consumer block outputs** — every out_i is read only by its
///    own Add (spec §1.3's `consumers(block_output) == {this_Add}`).
///
/// Zero candidates is `NoResidualAdd`; ≥2 candidates that all share one
/// stream operand is `AmbiguousPatternMatch` (as for a sub-block).
fn find_residual_chain(
    wengert: &WengertList,
    closure: &[OpId],
    param_var_ids: &BTreeSet<VarId>,
) -> Result<Vec<ResidualCandidate>, PartialRefusal> {
    let chain = residual_candidates(wengert, closure, param_var_ids);
    if chain.is_empty() {
        return Err(PartialRefusal::NoResidualAdd);
    }
    if chain.len() >= 2 && chain.iter().all(|c| c.h_before == chain[0].h_before) {
        return Err(multi_candidate_refusal(&chain));
    }
    let adds: Vec<OpId> = chain.iter().map(|c| c.add_op).collect();
    let broken = |reason: String| PartialRefusal::BrokenChain { adds: adds.clone(), reason };
    let op_kind = |id: OpId| -> String {
        wengert
            .ops
            .iter()
            .find(|o| o.id == id)
            .map_or_else(|| "?".to_string(), |o| format!("{:?}", o.op))
    };

    // (1) One chain.
    for pair in chain.windows(2) {
        let (prev, next) = (pair[0], pair[1]);
        if next.h_before != prev.h_after {
            return Err(broken(format!(
                "Add op {} reads stream value VarId {}, not VarId {} (the result of the \
                 previous residual Add, op {}); the Adds are not one residual stream",
                next.add_op, next.h_before, prev.h_after, prev.add_op
            )));
        }
    }

    let closure_set: BTreeSet<OpId> = closure.iter().copied().collect();

    // (2) Intermediate stream values stay inside the block.
    for (i, link) in chain[..chain.len() - 1].iter().enumerate() {
        let h = link.h_after;
        let next_add = chain[i + 1].add_op;
        if wengert.output == h {
            return Err(broken(format!(
                "intermediate stream value VarId {h} (the result of Add op {}) is the \
                 program output",
                link.add_op
            )));
        }
        if let Some(reader) = wengert.ops.iter().find(|o| {
            o.inputs.contains(&h) && !closure_set.contains(&o.id) && o.id != next_add
        }) {
            return Err(broken(format!(
                "intermediate stream value VarId {h} (the result of Add op {}) is also read \
                 by op {} ({}), outside the block -- a skip connection, or a parameter-free \
                 op on the residual stream that the block's closure does not own",
                link.add_op,
                reader.id,
                op_kind(reader.id)
            )));
        }
    }

    // (3) Each block output feeds only its own Add.
    for link in &chain {
        if wengert.output == link.block_output {
            return Err(broken(format!(
                "block output VarId {} (added by Add op {}) is the program output",
                link.block_output, link.add_op
            )));
        }
        if let Some(reader) = wengert
            .ops
            .iter()
            .find(|o| o.inputs.contains(&link.block_output) && o.id != link.add_op)
        {
            return Err(broken(format!(
                "block output VarId {} (added by Add op {}) is also read by op {} ({})",
                link.block_output,
                link.add_op,
                reader.id,
                op_kind(reader.id)
            )));
        }
    }

    Ok(chain)
}

/// Wrap a partial refusal in the caller's context.
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
        PartialRefusal::ParallelResidualBranches { add_ops } => PruneRefusal::ParallelResidualBranches {
            layer_name: layer.layer_name.clone(),
            layer_role,
            add_ops,
        },
        PartialRefusal::AmbiguousPatternMatch { h_before, candidate_adds } => PruneRefusal::AmbiguousPatternMatch {
            layer_name: layer.layer_name.clone(),
            layer_role,
            h_before_var: h_before,
            candidate_adds,
        },
        PartialRefusal::BrokenChain { adds, reason } => PruneRefusal::BrokenResidualChain {
            layer_name: layer.layer_name.clone(),
            layer_role,
            adds,
            reason,
        },
    }
}

// ---------------------------------------------------------------------------
// Task 15: public diagnostic formatters
// ---------------------------------------------------------------------------

/// Map a refusal variant to its structured diagnostic code for Layer 4
/// structural assertions. Spec §6.3.
pub fn diagnostic_code(r: &PruneRefusal) -> crate::wggo_overrides::OverrideRejectReason {
    use crate::wggo_overrides::OverrideRejectReason;
    match r {
        PruneRefusal::CrossLayerParam { .. } => OverrideRejectReason::PruneCrossLayerParam,
        PruneRefusal::NoResidualAdd { .. } => OverrideRejectReason::PruneNoResidualAdd,
        PruneRefusal::ParallelResidualBranches { .. } => OverrideRejectReason::PruneParallelResidualBranches,
        PruneRefusal::AmbiguousPatternMatch { .. } => OverrideRejectReason::PruneAmbiguousPatternMatch,
        PruneRefusal::EmptyClosure { .. } => OverrideRejectReason::PruneEmptyClosure,
        PruneRefusal::BrokenResidualChain { .. } => OverrideRejectReason::PruneBrokenResidualChain,
        PruneRefusal::ConflictingPruneDecisions { .. } => OverrideRejectReason::PruneConflictingDecisions,
    }
}

/// Spec §6.1 success-path stderr line. Format:
///   [prune] layer=N name=... role=... applied=true closure_size=K ops_deleted=K residual_add_op=ID
/// Separator convention: key=value throughout (no colons). For a whole-block
/// chain-collapse `residual_add_op` is the LAST Add of the chain.
pub fn format_success_stderr(rewrite: &PruneRewrite, layer_index: u32, ops_deleted: usize) -> String {
    format!(
        "[prune] layer={} name={} role={:?} applied=true closure_size={} ops_deleted={} residual_add_op={}",
        layer_index,
        rewrite.layer_name,
        rewrite.layer_role,
        rewrite.closure_ops.len(),
        ops_deleted,
        rewrite.residual_add_op,
    )
}

/// Spec §3 three-part refusal message. One format per variant.
/// Format: three labeled sections (requested / expected / found) after a
/// one-line header. Trailing newline so multiple refusals separate cleanly.
pub fn format_refusal(r: &PruneRefusal) -> String {
    match r {
        PruneRefusal::CrossLayerParam {
            layer_name, layer_role, param_name, param_var, external_consumer, external_op_kind,
        } => format!(
"prune: layer has cross-layer parameter sharing (not supported in v1).
  requested:  prune {layer_name}  (role={layer_role:?})
  expected:   all parameters matching `{layer_name}.*` consumed only within
              the layer's computational closure
  found:      parameter `{param_name}` (VarId {param_var}) is consumed by
              op_id={external_consumer} ({external_op_kind}), which is
              outside the closure for {layer_name}
"
        ),
        PruneRefusal::NoResidualAdd { layer_name, layer_role, closure_size } => format!(
"prune: layer is not residual-structured (no boundary Add found).
  requested:  prune {layer_name}  (role={layer_role:?})
  expected:   exactly one op in the closure matching Add(h_before, block_output)
              with block_output in closure and block_output single-consumer
  found:      closure has {closure_size} ops but zero ops match the residual
              pattern; the layer appears to be non-residual (SSM / Mamba /
              non-standard architecture)
"
        ),
        PruneRefusal::ParallelResidualBranches { layer_name, layer_role, add_ops } => format!(
"prune: layer has parallel residual branches (not supported in v1).
  requested:  prune {layer_name}  (role={layer_role:?})
  expected:   exactly one residual boundary Add
  found:      {k} residual Adds detected at ops {add_ops:?}; each appears to
              be a separate residual branch (distinct h_before values). Parallel
              residual pruning requires branch-by-branch semantics not yet
              specified.
",
            k = add_ops.len(),
        ),
        PruneRefusal::AmbiguousPatternMatch { layer_name, layer_role, h_before_var, candidate_adds } => format!(
"prune: layer has multiple candidate residual boundaries (pattern-match ambiguous).
  requested:  prune {layer_name}  (role={layer_role:?})
  expected:   exactly one op matching the residual pattern
  found:      {k} candidate Adds match the residual pattern against the same
              h_before (VarId {h_before_var}): ops {candidate_adds:?}.
              Boundary disambiguation requires architecture-specific rules not
              yet specified.
",
            k = candidate_adds.len(),
        ),
        PruneRefusal::EmptyClosure { layer_name, layer_role, prefix } => format!(
"prune: no parameters match the requested layer prefix.
  requested:  prune {layer_name}  (role={layer_role:?})
  expected:   at least one parameter VarId with var_name starting
              with `{prefix}` (directly, or after the model variable as
              in `m.{prefix}`)
  found:      zero matching parameters in the WeightMap. Check layer name /
              index; the requested layer does not exist in the compiled model.
"
        ),
        PruneRefusal::BrokenResidualChain { layer_name, layer_role, adds, reason } => format!(
"prune: whole-block residual chain cannot be collapsed (v2 chain-collapse refused).
  requested:  prune {layer_name}  (role={layer_role:?})
  expected:   the block's residual Adds form ONE stream chain
              h0 -> Add(h0, out1)=h1 -> ... -> Add(h(k-1), outk)=hk, each
              intermediate h1..h(k-1) read only by the block's own ops and the
              next Add, each block output out_i read only by its own Add
  found:      {k} residual Add(s) at ops {adds:?}: {reason}
",
            k = adds.len(),
        ),
        PruneRefusal::ConflictingPruneDecisions { decision_a, decision_b, reason } => format!(
"prune: two prune decisions in the same plan conflict.
  requested:  prune {decision_a} AND prune {decision_b} in the same plan
  expected:   each rewrite's closure and VarId aliasing is disjoint from every
              other rewrite's
  found:      {reason}
"
        ),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use crate::wggo_apply::{AppliedLayer, AppliedPlan};
    use crate::wggo_dp::CoarseDecision;
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
            coarse: CoarseDecision::Prune,
            pipeline_stage: 0,
            shard_factor: 1,
            shard_grads: 1,
            shard_optim: 1,
            active_heads: 0,
            ffn_width: 0,
            csha_level: 0,
            adapter_rank: 0,
            adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
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
        // Wengert list modeling: h_after = h_before + relu(relu(v_hb))
        //   op0: Relu v0 = relu(v_hb)          — v0 is named blocks.7.attn.wq (param producer)
        //   op1: Relu v1 = relu(v0)            — block_output (reads layer-7 param)
        //   op2: Add  v_ha = v_hb + v1         — residual boundary
        //
        // Expected closure: {op0, op1}. The Add (op2) is the BOUNDARY, not the closure.

        let v_hb: VarId = 100;
        let v0:   VarId = 200;
        let v1:   VarId = 201;
        let v_ha: VarId = 202;

        let ops = vec![
            op_unary(0, v0, v_hb, PrimalOp::Relu),  // produces v0 (a layer-7 param VarId)
            op_unary(1, v1, v0, PrimalOp::Relu),    // reads layer-7 param → block_output
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
                assert_eq!(plan.residual_add_op_ids, vec![2]);
                assert_eq!(plan.h_before_var, v_hb);
                assert_eq!(plan.h_after_var, v_ha);
            }
            PlanResult::Refused(r) => panic!("expected Ok, got Refused({r:?})"),
        }
    }

    #[test]
    fn parallel_residuals_refusal() {
        // Two parallel residual paths sharing a common layer-N param:
        //   y1 = h_before_1 + (something reading param)
        //   y2 = h_before_2 + (something reading param)
        //
        // Both Adds match the residual pattern; their `h_before` inputs are
        // DISTINCT (different pre-block streams). Parallel-residual architectures
        // (e.g., Parallel Transformers) aren't supported in v1.

        let v_hb1: VarId = 100;
        let v_hb2: VarId = 110;
        let v_p:   VarId = 200;   // shared layer-N param
        let v_b1:  VarId = 201;
        let v_b2:  VarId = 211;
        let v_y1:  VarId = 300;
        let v_y2:  VarId = 310;

        let ops = vec![
            op_unary(0, v_p,  v_hb1, PrimalOp::Relu),   // param producer
            op_unary(1, v_b1, v_p,   PrimalOp::Relu),    // block_output branch 1
            op_add  (2, v_y1, v_hb1, v_b1),              // residual branch 1: Add(h_before_1, block_output_1)
            op_unary(3, v_b2, v_p,   PrimalOp::Relu),    // block_output branch 2 (reads same param)
            op_add  (4, v_y2, v_hb2, v_b2),              // residual branch 2: Add(h_before_2, block_output_2) — DISTINCT h_before
        ];

        let wengert = mk_wengert(
            ops, v_y1,
            &[(v_hb1, "h_before_1"), (v_hb2, "h_before_2"), (v_p, "blocks.7.attn.wq")],
        );
        let layer = mk_prune_layer(7, "blocks.7.attn");
        let weight_map = WeightMap::default();

        match plan_rewrite(&wengert, &layer, &weight_map) {
            PlanResult::Refused(PruneRefusal::ParallelResidualBranches { layer_name, add_ops, .. }) => {
                assert_eq!(layer_name, "blocks.7.attn");
                assert_eq!(add_ops.len(), 2, "expected 2 parallel-branch Adds, got {}", add_ops.len());
                assert!(add_ops.contains(&2) && add_ops.contains(&4),
                    "expected parallel Adds at ops 2 and 4; got {add_ops:?}");
            }
            PlanResult::Ok(plan) => panic!("expected ParallelResidualBranches, got Ok({plan:?})"),
            PlanResult::Refused(other) => panic!("expected ParallelResidualBranches, got: {other:?}"),
        }
    }

    #[test]
    fn ambiguous_pattern_match_refusal() {
        // Two Adds that both pattern-match against the SAME h_before (v_hb).
        // E.g., an architecture with both a pre-norm residual branch and a
        // post-norm residual branch visible at the layer boundary. Both Adds
        // match the residual pattern; we can't choose one without guessing.
        //
        //   op0: Relu v_p  = relu(v_hb)   — v_p is blocks.7.attn.wq (param producer)
        //   op1: Relu v_b1 = relu(v_p)    — candidate block_output 1
        //   op2: Relu v_b2 = relu(v_p)    — candidate block_output 2
        //   op3: Add  v_y1 = v_hb + v_b1  — candidate residual Add 1
        //   op4: Add  v_y2 = v_hb + v_b2  — candidate residual Add 2 (SAME h_before)
        //
        // Expected: AmbiguousPatternMatch with h_before_var = v_hb and
        // candidate_adds containing {3, 4}.

        let v_hb: VarId = 100;
        let v_p:  VarId = 200;
        let v_b1: VarId = 201;
        let v_b2: VarId = 202;
        let v_y1: VarId = 300;
        let v_y2: VarId = 310;

        let ops = vec![
            op_unary(0, v_p,  v_hb, PrimalOp::Relu),
            op_unary(1, v_b1, v_p,  PrimalOp::Relu),
            op_unary(2, v_b2, v_p,  PrimalOp::Relu),
            op_add  (3, v_y1, v_hb, v_b1),             // Add(h_before=v_hb, block_output=v_b1)
            op_add  (4, v_y2, v_hb, v_b2),             // Add(h_before=v_hb, block_output=v_b2)
        ];
        let wengert = mk_wengert(
            ops, v_y1,
            &[(v_hb, "h_before"), (v_p, "blocks.7.attn.wq")],
        );
        let layer = mk_prune_layer(7, "blocks.7.attn");
        let weight_map = WeightMap::default();

        match plan_rewrite(&wengert, &layer, &weight_map) {
            PlanResult::Refused(PruneRefusal::AmbiguousPatternMatch {
                layer_name, h_before_var, candidate_adds, ..
            }) => {
                assert_eq!(layer_name, "blocks.7.attn");
                assert_eq!(h_before_var, v_hb, "both Adds share the same h_before");
                assert_eq!(candidate_adds.len(), 2, "expected 2 candidate Adds");
                assert!(candidate_adds.contains(&3) && candidate_adds.contains(&4),
                    "expected candidates {{3, 4}}; got {candidate_adds:?}");
            }
            PlanResult::Ok(plan) => panic!("expected AmbiguousPatternMatch, got Ok({plan:?})"),
            PlanResult::Refused(other) => panic!("expected AmbiguousPatternMatch, got: {other:?}"),
        }
    }

    #[test]
    fn empty_closure_refusal() {
        // Wengert has parameters for blocks.0.* but user asks to prune blocks.99.attn.
        // Prefix match returns zero VarIds → EmptyClosure refusal.

        let v_hb: VarId = 100;
        let v_p:  VarId = 200;
        let v_y:  VarId = 300;

        let ops = vec![
            op_unary(0, v_p, v_hb, PrimalOp::Relu),
            op_add  (1, v_y, v_hb, v_p),
        ];
        let wengert = mk_wengert(
            ops, v_y,
            &[(v_hb, "h_before"), (v_p, "blocks.0.attn.wq")],
        );
        let layer = mk_prune_layer(99, "blocks.99.attn");  // nonexistent layer
        let weight_map = WeightMap::default();

        match plan_rewrite(&wengert, &layer, &weight_map) {
            PlanResult::Refused(PruneRefusal::EmptyClosure { layer_name, prefix, .. }) => {
                assert_eq!(layer_name, "blocks.99.attn");
                assert_eq!(prefix, "blocks.99.attn.");
            }
            PlanResult::Ok(plan) => panic!("expected EmptyClosure, got Ok({plan:?})"),
            PlanResult::Refused(other) => panic!("expected EmptyClosure, got: {other:?}"),
        }
    }

    #[test]
    fn cross_layer_param_refusal() {
        // Layer-7 parameter `v_p` (= blocks.7.attn.wq) is consumed inside the
        // layer AND by an external Relu whose output escapes to wengert.output.
        // Pruning the layer would delete the external Relu too, leaving
        // wengert.output dangling — the compiler must refuse.
        //
        //   op0: Relu v_p   = relu(v_hb)    — param producer
        //   op1: Relu v_b   = relu(v_p)     — in-layer consumer (block_output)
        //   op2: Add  v_y   = v_hb + v_b    — residual Add (h_after = v_y)
        //   op3: Relu v_ext = relu(v_p)     — external consumer; its output is wengert.output
        //
        // wengert.output = v_ext. Expected: CrossLayerParam refusal pointing at
        // v_p + op3 (the external consumer).

        let v_hb:  VarId = 100;
        let v_p:   VarId = 200;  // blocks.7.attn.wq
        let v_b:   VarId = 201;
        let v_y:   VarId = 300;
        let v_ext: VarId = 400;

        let ops = vec![
            op_unary(0, v_p,   v_hb, PrimalOp::Relu),
            op_unary(1, v_b,   v_p,  PrimalOp::Relu),     // in-layer
            op_add  (2, v_y,   v_hb, v_b),                // residual boundary
            op_unary(3, v_ext, v_p,  PrimalOp::Relu),     // external consumer — CROSS-LAYER LEAK
        ];
        let wengert = mk_wengert(
            ops, v_ext,
            &[(v_hb, "h_before"), (v_p, "blocks.7.attn.wq")],
        );
        let layer = mk_prune_layer(7, "blocks.7.attn");
        let weight_map = WeightMap::default();

        match plan_rewrite(&wengert, &layer, &weight_map) {
            PlanResult::Refused(PruneRefusal::CrossLayerParam {
                layer_name, param_var, external_consumer, ..
            }) => {
                assert_eq!(layer_name, "blocks.7.attn");
                assert_eq!(param_var, v_p, "expected blocks.7.attn.wq VarId");
                assert_eq!(external_consumer, 3, "expected op3 as external consumer");
            }
            PlanResult::Ok(plan) => panic!("expected CrossLayerParam, got Ok({plan:?})"),
            PlanResult::Refused(other) => panic!("expected CrossLayerParam, got: {other:?}"),
        }
    }

    #[test]
    fn conflicting_decisions_refusal() {
        // Two prune decisions whose closures would overlap on the same OpIds.
        //
        // Setup: one Wengert list with parameters for blocks.7.attn AND for
        // blocks.7.attn.wq (a more specific prefix that matches the same
        // param VarId). Both layers' closures will therefore overlap.
        //
        //   op0: Relu v_p  = relu(v_hb)    — v_p named "blocks.7.attn.wq"
        //   op1: Relu v_b  = relu(v_p)
        //   op2: Add  v_y  = v_hb + v_b    — residual
        //
        // Decision A: prune "blocks.7.attn"   → prefix "blocks.7.attn."    → matches v_p
        // Decision B: prune "blocks.7.attn.wq" → prefix "blocks.7.attn.wq." → matches no vars (empty closure)
        //
        // Actually, construction (b) hits EmptyClosure first. We need DISTINCT
        // prefixes that both successfully plan and share OpIds.
        //
        // Better setup: use two var names that share a common prefix:
        //   v_p1  named "blocks.7.attn.wq"  — matches prefix "blocks.7.attn."
        //   v_p2  named "blocks.7.attn.wk"  — matches prefix "blocks.7.attn."
        // Decision A: prune "blocks.7.attn"   → matches v_p1 AND v_p2
        // Decision B: prune "blocks.7.attn"   → same prefix (same layer twice in plan)
        //
        // Actually the SAME prefix twice is the clearest conflict — two plan
        // entries both trying to delete the same ops.

        let v_hb: VarId = 100;
        let v_p:  VarId = 200;   // blocks.7.attn.wq
        let v_b:  VarId = 201;
        let v_y:  VarId = 300;

        let ops = vec![
            op_unary(0, v_p, v_hb, PrimalOp::Relu),
            op_unary(1, v_b, v_p,  PrimalOp::Relu),
            op_add  (2, v_y, v_hb, v_b),
        ];
        let mut wengert = mk_wengert(
            ops,
            v_y,
            &[(v_hb, "h_before"), (v_p, "blocks.7.attn.wq")],
        );

        // Plan with two prune decisions whose closures overlap (same layer name
        // — they'd both claim the same closure ops).
        let plan = AppliedPlan {
            layers: vec![
                mk_prune_layer(7, "blocks.7.attn"),
                mk_prune_layer(70, "blocks.7.attn"),
            ],
            total_us: 0.0,
            peak_memory_bytes: 0,
        };

        let result = run(&mut wengert, &plan, &WeightMap::default());

        // Expect at least one ConflictingPruneDecisions refusal.
        let found_conflict = result.refusals.iter()
            .any(|r| matches!(r, PruneRefusal::ConflictingPruneDecisions { .. }));
        assert!(
            found_conflict,
            "expected a ConflictingPruneDecisions refusal; got refusals: {:?}",
            result.refusals,
        );

        // Dry-run invariant (spec §5.3 Phase 2): on refusal, wengert is unchanged.
        assert_eq!(
            wengert.ops.len(), 3,
            "wengert should be untouched on refusal (spec §5.3 Phase 2); still has {} ops",
            wengert.ops.len(),
        );

        // And result.rewrites must be empty per the dry-run contract.
        assert_eq!(
            result.rewrites.len(), 0,
            "expected empty rewrites on refusal; got {}",
            result.rewrites.len(),
        );
    }

    #[test]
    fn apply_rewrite_deletes_closure_and_aliases_h_after() {
        // Wengert:
        //   op0: Relu v0   = relu(v_hb)       (param producer for blocks.7.attn.wq)
        //   op1: Relu v1   = relu(v0)         (block_output)
        //   op2: Add  v_ha = v_hb + v1        (residual Add at the boundary)
        //   op3: Relu v_out = relu(v_ha)      (downstream consumer of h_after)
        //
        // wengert.output = v_out.
        //
        // After prune of blocks.7.attn:
        //   - op0, op1, op2 are deleted
        //   - op3's input v_ha is repointed to v_hb
        //   - wengert.output stays v_out (op3 survives)

        let v_hb:  VarId = 100;
        let v0:    VarId = 200;
        let v1:    VarId = 201;
        let v_ha:  VarId = 202;
        let v_out: VarId = 300;

        let ops = vec![
            op_unary(0, v0,    v_hb, PrimalOp::Relu),
            op_unary(1, v1,    v0,   PrimalOp::Relu),
            op_add  (2, v_ha,  v_hb, v1),
            op_unary(3, v_out, v_ha, PrimalOp::Relu),
        ];
        let mut wengert = mk_wengert(
            ops,
            v_out,
            &[(v_hb, "h_before"), (v0, "blocks.7.attn.wq"), (v_ha, "h_after")],
        );
        let plan = AppliedPlan {
            layers: vec![mk_prune_layer(7, "blocks.7.attn")],
            total_us: 0.0,
            peak_memory_bytes: 0,
        };

        let result = run(&mut wengert, &plan, &WeightMap::default());

        assert!(
            result.refusals.is_empty(),
            "expected no refusals; got: {:?}",
            result.refusals,
        );
        assert_eq!(result.rewrites.len(), 1);
        assert_eq!(result.ops_deleted, 3, "closure=2 (op0+op1) + residual Add (op2) = 3");

        // Exactly one op survives: op3 (the downstream consumer).
        assert_eq!(wengert.ops.len(), 1, "expected only op3 to survive; got {}", wengert.ops.len());
        let surviving = &wengert.ops[0];
        assert_eq!(surviving.id, 3);
        assert_eq!(
            surviving.inputs, vec![v_hb],
            "downstream consumer must be aliased from v_ha to v_hb"
        );

        // wengert.output still points at v_out (op3's result).
        assert_eq!(wengert.output, v_out);

        // pruned_forward_var_ids should include v0, v1, v_ha (everything removed).
        assert!(result.pruned_forward_var_ids.contains(&v0));
        assert!(result.pruned_forward_var_ids.contains(&v1));
        assert!(result.pruned_forward_var_ids.contains(&v_ha));
    }

    #[test]
    fn apply_rewrite_repoints_wengert_output_when_h_after_is_output() {
        // Edge case: wengert.output is the residual Add's result directly.
        // After prune, wengert.output must be repointed to v_hb.
        //
        //   op0: Relu v0   = relu(v_hb)
        //   op1: Add  v_ha = v_hb + v0    (residual)
        //
        // wengert.output = v_ha. After prune of blocks.7.attn:
        //   - op0, op1 deleted
        //   - wengert.output repointed to v_hb

        let v_hb: VarId = 100;
        let v0:   VarId = 200;
        let v_ha: VarId = 201;

        let ops = vec![
            op_unary(0, v0,   v_hb, PrimalOp::Relu),
            op_add  (1, v_ha, v_hb, v0),
        ];
        let mut wengert = mk_wengert(
            ops,
            v_ha,
            &[(v_hb, "h_before"), (v0, "blocks.7.attn.wq")],
        );
        let plan = AppliedPlan {
            layers: vec![mk_prune_layer(7, "blocks.7.attn")],
            total_us: 0.0,
            peak_memory_bytes: 0,
        };

        let result = run(&mut wengert, &plan, &WeightMap::default());

        assert!(result.refusals.is_empty(), "expected no refusals");
        assert_eq!(wengert.ops.len(), 0, "all ops deleted");
        assert_eq!(wengert.output, v_hb, "wengert.output repointed from v_ha to v_hb");
    }

    #[test]
    fn no_residual_add_refusal() {
        // Closure: 3 Relu ops, NO Add anywhere. Non-residual architecture
        // (e.g., SSM/Mamba-style layer at the sub-block level).
        //
        //   op0: Relu v0 = relu(v_hb)   — v0 is named blocks.7.attn.wq (param producer)
        //   op1: Relu v1 = relu(v0)
        //   op2: Relu v2 = relu(v1)
        //
        // Expected refusal: NoResidualAdd with closure_size = 3.

        let v_hb: VarId = 100;
        let v0:   VarId = 200;
        let v1:   VarId = 201;
        let v2:   VarId = 202;

        let ops = vec![
            op_unary(0, v0, v_hb, PrimalOp::Relu),
            op_unary(1, v1, v0,   PrimalOp::Relu),
            op_unary(2, v2, v1,   PrimalOp::Relu),
        ];
        let wengert = mk_wengert(
            ops,
            v2,
            &[(v_hb, "h_before"), (v0, "blocks.7.attn.wq")],
        );
        let layer = mk_prune_layer(7, "blocks.7.attn");
        let weight_map = WeightMap::default();

        match plan_rewrite(&wengert, &layer, &weight_map) {
            PlanResult::Refused(PruneRefusal::NoResidualAdd { layer_name, closure_size, .. }) => {
                assert_eq!(layer_name, "blocks.7.attn");
                assert_eq!(closure_size, 3,
                    "all 3 Relu ops are in the closure (no Add to terminate at); got {closure_size}");
            }
            PlanResult::Ok(plan) => panic!("expected NoResidualAdd refusal, got Ok({plan:?})"),
            PlanResult::Refused(other) => panic!("expected NoResidualAdd refusal, got other variant: {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // v2 whole-block chain-collapse (LayerRole::Block) + model-variable
    // prefix matching. The fixtures mirror what the source-AD extractor
    // actually emits for
    //
    //     model Blk:  fn forward(self, x): let h = x + (x @ self.wa)
    //                                      return h + (h @ self.wb)
    //     model Net:  blocks: [Blk; N];  for block in self.blocks: h = block.forward(h)
    //
    // (NSL_DEBUG_WENGERT dump, 2026-09-30): `Param("m.blocks.N.wa")` leaves
    // named with the model variable, `Matmul(h, w)`, `Add(h, mm)`.
    // -----------------------------------------------------------------------

    /// Shorthand: a zero-input leaf op (`Input` / `Param`).
    fn op_leaf(id: OpId, result: VarId, kind: PrimalOp) -> WengertOp {
        WengertOp {
            id, result, op: kind, inputs: vec![],
            saved_for_backward: false, checkpointed: false,
        }
    }

    /// Shorthand: Matmul op.
    fn op_matmul(id: OpId, result: VarId, a: VarId, b: VarId) -> WengertOp {
        WengertOp {
            id, result, op: PrimalOp::Matmul, inputs: vec![a, b],
            saved_for_backward: false, checkpointed: false,
        }
    }

    /// Ids of one two-residual block in [`mk_blocks_wengert`].
    #[derive(Debug, Clone, Copy)]
    struct BlkIds {
        h0: VarId,
        wa_op: OpId,
        mm_a_op: OpId,
        add1_op: OpId,
        h1: VarId,
        wb_op: OpId,
        mm_b_op: OpId,
        add2_op: OpId,
        h2: VarId,
    }

    /// `n_blocks` two-residual blocks in a chain over Input `x`, followed by
    /// a downstream `Relu(h_final)` consumer when `tail_consumer` (otherwise
    /// `wengert.output` is the last block's h2 directly). Op id == VarId for
    /// readability.
    fn mk_blocks_wengert(n_blocks: u32, tail_consumer: bool) -> (WengertList, Vec<BlkIds>, Option<OpId>) {
        let mut ops = vec![op_leaf(0, 0, PrimalOp::Input("x".into()))];
        let mut names: Vec<(VarId, String)> = vec![(0, "x".into())];
        let mut blocks = Vec::new();
        let mut h: VarId = 0;
        let mut next: u32 = 1;
        let mut alloc = || { let v = next; next += 1; v };
        for b in 0..n_blocks {
            let (wa, mm_a, h1, wb, mm_b, h2) = (alloc(), alloc(), alloc(), alloc(), alloc(), alloc());
            ops.push(op_leaf(wa, wa, PrimalOp::Param(format!("m.blocks.{b}.wa"))));
            ops.push(op_matmul(mm_a, mm_a, h, wa));
            ops.push(op_add(h1, h1, h, mm_a));
            ops.push(op_leaf(wb, wb, PrimalOp::Param(format!("m.blocks.{b}.wb"))));
            ops.push(op_matmul(mm_b, mm_b, h1, wb));
            ops.push(op_add(h2, h2, h1, mm_b));
            names.push((wa, format!("m.blocks.{b}.wa")));
            names.push((wb, format!("m.blocks.{b}.wb")));
            names.push((h1, "h".into()));
            names.push((h2, "x".into()));
            blocks.push(BlkIds {
                h0: h, wa_op: wa, mm_a_op: mm_a, add1_op: h1, h1,
                wb_op: wb, mm_b_op: mm_b, add2_op: h2, h2,
            });
            h = h2;
        }
        let (output, tail) = if tail_consumer {
            let t = alloc();
            ops.push(op_unary(t, t, h, PrimalOp::Relu));
            (t, Some(t))
        } else {
            (h, None)
        };
        let named: Vec<(VarId, &str)> = names.iter().map(|(v, s)| (*v, s.as_str())).collect();
        (mk_wengert(ops, output, &named), blocks, tail)
    }

    /// Every input of every surviving op is produced by a surviving op, and
    /// so is `wengert.output` — the rewrite left nothing dangling.
    fn assert_no_dangling(w: &WengertList) {
        let produced: BTreeSet<VarId> = w.ops.iter().map(|o| o.result).collect();
        for op in &w.ops {
            for v in &op.inputs {
                assert!(produced.contains(v), "op {} reads VarId {v}, which no surviving op produces", op.id);
            }
        }
        assert!(produced.contains(&w.output), "wengert.output VarId {} is not produced", w.output);
    }

    fn plan_of(layers: Vec<AppliedLayer>) -> AppliedPlan {
        AppliedPlan { layers, total_us: 0.0, peak_memory_bytes: 0 }
    }

    #[test]
    fn var_in_layer_matches_bare_and_model_variable_prefixed_names() {
        assert!(var_in_layer("blocks.1.wa", "blocks.1"));
        assert!(var_in_layer("m.blocks.1.wa", "blocks.1"));
        assert!(var_in_layer("m.blocks.1.attn.wq", "blocks.1.attn"));
        // Dot boundary: blocks.10 is not blocks.1.
        assert!(!var_in_layer("m.blocks.10.wa", "blocks.1"));
        assert!(!var_in_layer("blocks.10.wa", "blocks.1"));
        // The layer itself is not one of its params.
        assert!(!var_in_layer("blocks.1", "blocks.1"));
        assert!(!var_in_layer("m.blocks.1", "blocks.1"));
        // Only ONE leading component is stripped: a deeper nesting refuses
        // (EmptyClosure) rather than guessing.
        assert!(!var_in_layer("m.encoder.blocks.1.wa", "blocks.1"));
        assert!(!var_in_layer("x", "blocks.1"));
    }

    #[test]
    fn whole_block_two_add_chain_collapses_middle_block() {
        let (mut w, blocks, tail) = mk_blocks_wengert(3, true);
        let b1 = blocks[1];
        let b2 = blocks[2];
        let before = w.ops.len();

        let result = run(&mut w, &plan_of(vec![mk_prune_layer(2, "blocks.1")]), &WeightMap::default());

        assert!(result.refusals.is_empty(), "expected no refusals; got {:?}", result.refusals);
        assert_eq!(result.rewrites.len(), 1);
        let rw = &result.rewrites[0];
        assert_eq!(rw.layer_role, LayerRole::Block);
        assert_eq!(rw.h_before_var, b1.h0);
        assert_eq!(rw.h_after_var, b1.h2);
        assert_eq!(rw.residual_add_ops, vec![b1.add1_op, b1.add2_op], "both residual Adds, stream order");
        assert_eq!(rw.residual_add_op, b1.add2_op, "the success line reports the LAST Add");
        assert_eq!(rw.closure_ops, vec![b1.wa_op, b1.mm_a_op, b1.wb_op, b1.mm_b_op]);
        assert_eq!(rw.ops_deleted, 6, "4 closure ops + 2 residual Adds");
        assert_eq!(result.ops_deleted, 6);
        assert_eq!(w.ops.len(), before - 6);

        // Every op of blocks.1 is gone.
        let surviving: BTreeSet<OpId> = w.ops.iter().map(|o| o.id).collect();
        for id in [b1.wa_op, b1.mm_a_op, b1.add1_op, b1.wb_op, b1.mm_b_op, b1.add2_op] {
            assert!(!surviving.contains(&id), "op {id} of blocks.1 survived");
        }
        // blocks.2's stream consumers (its first Matmul and first Add) now
        // read blocks.1's input h0 — the block is an identity.
        let mm = w.ops.iter().find(|o| o.id == b2.mm_a_op).unwrap();
        assert_eq!(mm.inputs[0], b1.h0, "blocks.2 matmul must read h0 of blocks.1");
        let add = w.ops.iter().find(|o| o.id == b2.add1_op).unwrap();
        assert_eq!(add.inputs[0], b1.h0, "blocks.2 residual Add must read h0 of blocks.1");
        assert_eq!(Some(w.output), tail, "output (a downstream op) is untouched");
        assert_no_dangling(&w);

        // pruned_forward_var_ids: the closure results and BOTH stream values.
        for v in [b1.wa_op, b1.mm_a_op, b1.h1, b1.wb_op, b1.mm_b_op, b1.h2] {
            assert!(result.pruned_forward_var_ids.contains(&v), "VarId {v} missing from pruned set");
        }
        // The pruned params' names are gone; the survivors' are not.
        assert!(!w.var_names.values().any(|n| n.starts_with("m.blocks.1.")));
        assert!(w.var_names.values().any(|n| n == "m.blocks.2.wa"));
    }

    #[test]
    fn whole_block_last_block_repoints_wengert_output() {
        // wengert.output IS the last block's h2: it must become that block's h0.
        let (mut w, blocks, _) = mk_blocks_wengert(2, false);
        let b1 = blocks[1];
        assert_eq!(w.output, b1.h2);

        let result = run(&mut w, &plan_of(vec![mk_prune_layer(2, "blocks.1")]), &WeightMap::default());

        assert!(result.refusals.is_empty(), "expected no refusals; got {:?}", result.refusals);
        assert_eq!(w.output, b1.h0, "output repointed from the pruned block's h2 to its h0");
        assert_eq!(w.ops.len(), 1 + 6, "Input + blocks.0 survive");
        assert_no_dangling(&w);
    }

    #[test]
    fn whole_block_single_add_block_is_a_chain_of_one() {
        // Blk.forward(x) = x + (x @ wa): one residual Add.
        let ops = vec![
            op_leaf(0, 0, PrimalOp::Input("x".into())),
            op_leaf(1, 1, PrimalOp::Param("m.blocks.0.wa".into())),
            op_matmul(2, 2, 0, 1),
            op_add(3, 3, 0, 2),
            op_unary(4, 4, 3, PrimalOp::Relu),
        ];
        let mut w = mk_wengert(ops, 4, &[(0, "x"), (1, "m.blocks.0.wa")]);

        let result = run(&mut w, &plan_of(vec![mk_prune_layer(1, "blocks.0")]), &WeightMap::default());

        assert!(result.refusals.is_empty(), "expected no refusals; got {:?}", result.refusals);
        let rw = &result.rewrites[0];
        assert_eq!(rw.residual_add_ops, vec![3]);
        assert_eq!(rw.closure_ops, vec![1, 2]);
        assert_eq!(rw.ops_deleted, 3);
        assert_eq!(w.ops.iter().map(|o| o.id).collect::<Vec<_>>(), vec![0, 4]);
        assert_eq!(w.ops[1].inputs, vec![0], "consumer of h1 now reads h0");
        assert_no_dangling(&w);
    }

    #[test]
    fn whole_block_intermediate_with_outside_reader_is_refused() {
        // A skip connection: an op after blocks.0 reads blocks.0's
        // intermediate stream value h1. Deleting Add#1 would leave it
        // dangling; repointing it to h0 would silently change its value.
        let (mut w, blocks, _) = mk_blocks_wengert(2, false);
        let b0 = blocks[0];
        let skip: VarId = 100;
        w.ops.push(op_add(skip, skip, blocks[1].h2, b0.h1));
        w.output = skip;
        let snapshot = w.ops.len();

        let result = run(&mut w, &plan_of(vec![mk_prune_layer(1, "blocks.0")]), &WeightMap::default());

        assert!(result.rewrites.is_empty());
        match &result.refusals[..] {
            [PruneRefusal::BrokenResidualChain { layer_name, adds, reason, .. }] => {
                assert_eq!(layer_name, "blocks.0");
                assert_eq!(adds, &vec![b0.add1_op, b0.add2_op]);
                assert!(reason.contains(&format!("VarId {}", b0.h1)), "reason must name h1: {reason}");
                assert!(reason.contains(&format!("op {skip}")), "reason must name the outside reader: {reason}");
                assert!(reason.contains("skip connection"), "{reason}");
            }
            other => panic!("expected one BrokenResidualChain, got {other:?}"),
        }
        assert_eq!(w.ops.len(), snapshot, "wengert untouched on refusal");
    }

    #[test]
    fn whole_block_adds_that_do_not_chain_are_refused() {
        // Two residual Adds on DIFFERENT streams (x and z): neither one's
        // stream operand is the other's result, so there is no single
        // stream to collapse.
        let ops = vec![
            op_leaf(0, 0, PrimalOp::Input("x".into())),
            op_leaf(1, 1, PrimalOp::Input("z".into())),
            op_leaf(2, 2, PrimalOp::Param("m.blocks.0.wa".into())),
            op_matmul(3, 3, 0, 2),
            op_add(4, 4, 0, 3),             // x + x@wa
            op_leaf(5, 5, PrimalOp::Param("m.blocks.0.wb".into())),
            op_matmul(6, 6, 1, 5),
            op_add(7, 7, 1, 6),             // z + z@wb — a different stream
            op_add(8, 8, 4, 7),
        ];
        let mut w = mk_wengert(ops, 8, &[(0, "x"), (1, "z"), (2, "m.blocks.0.wa"), (5, "m.blocks.0.wb")]);

        let result = run(&mut w, &plan_of(vec![mk_prune_layer(1, "blocks.0")]), &WeightMap::default());

        match &result.refusals[..] {
            [r @ PruneRefusal::BrokenResidualChain { adds, reason, .. }] => {
                assert_eq!(adds, &vec![4, 7]);
                assert!(reason.contains("not one residual stream"), "{reason}");
                assert_eq!(
                    diagnostic_code(r),
                    crate::wggo_overrides::OverrideRejectReason::PruneBrokenResidualChain
                );
                let text = format_refusal(r);
                assert!(text.contains("requested:  prune blocks.0  (role=Block)"), "{text}");
                assert!(text.contains("expected:"), "{text}");
                assert!(text.contains("found:      2 residual Add(s) at ops [4, 7]"), "{text}");
            }
            other => panic!("expected one BrokenResidualChain, got {other:?}"),
        }
        assert_eq!(w.ops.len(), 9, "wengert untouched on refusal");
    }

    #[test]
    fn whole_block_output_with_a_second_reader_is_refused() {
        // blocks.0's first block output (x @ wa) also feeds an op outside the
        // block: the residual pattern requires it to feed only its Add.
        let (mut w, blocks, _) = mk_blocks_wengert(1, false);
        let b0 = blocks[0];
        let extra: VarId = 100;
        w.ops.push(op_add(extra, extra, b0.h2, b0.mm_a_op));
        w.output = extra;

        let result = run(&mut w, &plan_of(vec![mk_prune_layer(1, "blocks.0")]), &WeightMap::default());

        match &result.refusals[..] {
            [PruneRefusal::BrokenResidualChain { reason, .. }] => {
                assert!(reason.contains(&format!("block output VarId {}", b0.mm_a_op)), "{reason}");
                assert!(reason.contains(&format!("also read by op {extra}")), "{reason}");
            }
            other => panic!("expected one BrokenResidualChain, got {other:?}"),
        }
    }

    #[test]
    fn whole_block_unknown_layer_is_empty_closure() {
        let (mut w, _, _) = mk_blocks_wengert(2, true);
        let result = run(&mut w, &plan_of(vec![mk_prune_layer(9, "blocks.9")]), &WeightMap::default());
        match &result.refusals[..] {
            [PruneRefusal::EmptyClosure { layer_name, layer_role, prefix }] => {
                assert_eq!(layer_name, "blocks.9");
                assert_eq!(*layer_role, LayerRole::Block);
                assert_eq!(prefix, "blocks.9.");
            }
            other => panic!("expected one EmptyClosure, got {other:?}"),
        }
    }

    #[test]
    fn adjacent_whole_blocks_collapse_through_the_alias_map() {
        // blocks.1's output IS blocks.2's input. Both plans were validated
        // against the unmutated list, so blocks.2's h_before names a value
        // blocks.1's commit deletes. The consumer after blocks.2 must end up
        // reading blocks.1's h0 — not the deleted stream value.
        for order in [["blocks.1", "blocks.2"], ["blocks.2", "blocks.1"]] {
            let (mut w, blocks, tail) = mk_blocks_wengert(4, true);
            let (b1, b3) = (blocks[1], blocks[3]);
            let plan = plan_of(vec![mk_prune_layer(2, order[0]), mk_prune_layer(3, order[1])]);

            let result = run(&mut w, &plan, &WeightMap::default());

            assert!(result.refusals.is_empty(), "{order:?}: refusals {:?}", result.refusals);
            assert_eq!(result.rewrites.len(), 2);
            assert_eq!(result.ops_deleted, 12);
            let mm = w.ops.iter().find(|o| o.id == b3.mm_a_op).unwrap();
            assert_eq!(mm.inputs[0], b1.h0, "{order:?}: blocks.3 must read blocks.1's input");
            assert_eq!(Some(w.output), tail);
            assert_no_dangling(&w);
        }
    }

    #[test]
    fn whole_block_and_its_sub_block_conflict() {
        // `blocks.0` and `blocks.0.attn` claim the same ops — refuse both
        // rather than delete twice.
        let ops = vec![
            op_leaf(0, 0, PrimalOp::Input("x".into())),
            op_leaf(1, 1, PrimalOp::Param("m.blocks.0.attn.wq".into())),
            op_matmul(2, 2, 0, 1),
            op_add(3, 3, 0, 2),
        ];
        let mut w = mk_wengert(ops, 3, &[(0, "x"), (1, "m.blocks.0.attn.wq")]);
        let plan = plan_of(vec![mk_prune_layer(1, "blocks.0"), mk_prune_layer(2, "blocks.0.attn")]);

        let result = run(&mut w, &plan, &WeightMap::default());

        assert!(
            result.refusals.iter().any(|r| matches!(r, PruneRefusal::ConflictingPruneDecisions { .. })),
            "expected ConflictingPruneDecisions; got {:?}",
            result.refusals
        );
        assert_eq!(w.ops.len(), 4, "wengert untouched on refusal");
    }
}
