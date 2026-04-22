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
/// mutations only if all pass. On refusal, `wengert` is unchanged and
/// `rewrites` is empty.
///
/// See spec §5.3 for the three-phase contract.
pub fn run(
    wengert: &mut WengertList,
    applied_plan: &AppliedPlan,
    weight_map: &WeightMap,
) -> PruneRewriteResult {
    use crate::wggo_dp::LayerDecision;

    // Phase 1: validate each Prune decision without mutating wengert.
    let mut plans: Vec<PruneRewritePlan> = Vec::new();
    let mut refusals: Vec<PruneRefusal> = Vec::new();
    for layer in &applied_plan.layers {
        if !matches!(layer.coarse, LayerDecision::Prune) {
            continue;
        }
        match plan_rewrite(wengert, layer, weight_map) {
            PlanResult::Ok(plan) => plans.push(plan),
            PlanResult::Refused(refusal) => refusals.push(refusal),
        }
    }

    // Phase 1b: cross-plan conflict detection (spec §3.7). Two plans
    // conflict if they claim overlapping OpIds (same ops would be deleted
    // twice) OR target the same h_after_var (VarId aliasing undefined).
    if refusals.is_empty() {
        'outer: for i in 0..plans.len() {
            for j in (i + 1)..plans.len() {
                let a = &plans[i];
                let b = &plans[j];
                let a_ops: BTreeSet<OpId> = a.closure_op_ids.iter().copied().collect();
                let b_ops: BTreeSet<OpId> = b.closure_op_ids.iter().copied().collect();
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
    let mut rewrites: Vec<PruneRewrite> = Vec::with_capacity(plans.len());
    let mut pruned_forward_var_ids: BTreeSet<VarId> = BTreeSet::new();
    let mut ops_deleted: usize = 0;
    for plan in plans {
        // Capture pruned VarIds BEFORE the mutation (apply_rewrite will
        // delete ops and lose this info).
        let mut removed_vars: Vec<VarId> = wengert.ops.iter()
            .filter(|o| plan.closure_op_ids.contains(&o.id))
            .map(|o| o.result)
            .collect();
        removed_vars.push(plan.h_after_var);
        for v in removed_vars {
            pruned_forward_var_ids.insert(v);
        }

        let rewrite = apply_rewrite(wengert, plan);
        ops_deleted += rewrite.closure_ops.len() + 1; // +1 for the residual Add
        rewrites.push(rewrite);
    }

    PruneRewriteResult {
        rewrites,
        refusals: Vec::new(),
        pruned_forward_var_ids,
        ops_deleted,
    }
}

/// Phase 3 mutation. Deletes closure ops, repoints consumers of h_after to
/// h_before, then deletes the residual Add. Also cleans up stale var_names
/// / var_types entries and repoints wengert.output when h_after was it.
///
/// Spec §1.1 / §2.2 three-category treatment:
///   - closure ops → DELETED
///   - residual Add → REWRITTEN (consumers repointed) then DELETED
///   - h_before → UNTOUCHED (belongs to the prior stream)
fn apply_rewrite(
    wengert: &mut WengertList,
    plan: PruneRewritePlan,
) -> PruneRewrite {
    use std::collections::BTreeSet;

    // Collect the set of OpIds to delete: every closure op + the residual Add.
    let mut to_delete: BTreeSet<OpId> = plan.closure_op_ids.iter().copied().collect();
    to_delete.insert(plan.residual_add_op_id);

    // Repoint every surviving op's inputs from h_after_var → h_before_var.
    for op in wengert.ops.iter_mut() {
        if to_delete.contains(&op.id) { continue; }
        for input in op.inputs.iter_mut() {
            if *input == plan.h_after_var {
                *input = plan.h_before_var;
            }
        }
    }
    // Repoint wengert.output too, if it pointed at h_after.
    if wengert.output == plan.h_after_var {
        wengert.output = plan.h_before_var;
    }

    // Delete closure ops + residual Add from wengert.ops.
    wengert.ops.retain(|op| !to_delete.contains(&op.id));

    // Prune stale var_names / var_types for VarIds that no surviving op produces.
    // (h_before_var survives because it's produced by an upstream op outside the
    // closure OR is an initial input — either way, keep its entry.)
    let surviving_var_ids: BTreeSet<VarId> = wengert.ops.iter().map(|o| o.result).collect();
    wengert.var_names.retain(|v, _| surviving_var_ids.contains(v) || *v == wengert.output);
    wengert.var_types.retain(|v, _| surviving_var_ids.contains(v) || *v == wengert.output);

    PruneRewrite {
        layer_name: plan.layer_name,
        layer_role: plan.layer_role,
        h_before_var: plan.h_before_var,
        h_after_var: plan.h_after_var,
        residual_add_op: plan.residual_add_op_id,
        closure_ops: plan.closure_op_ids,
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
    ParallelResidualBranches { add_ops: Vec<OpId> },
    AmbiguousPatternMatch { h_before: VarId, candidate_adds: Vec<OpId> },
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

    // Spec §3.6: whole-block prune (LayerRole::Block) is not supported in v1.
    // Refuse immediately — don't compute a closure or pattern-match, since
    // the refusal semantic is role-based and the plan structure is
    // independent of the Wengert state.
    if matches!(layer_role, LayerRole::Block) {
        return PlanResult::Refused(PruneRefusal::WholeBlockUnsupported {
            layer_name: layer.layer_name.clone(),
        });
    }

    // (b) Find parameter VarIds matching `{layer_name}.` prefix.
    //     Spec §2.3 precondition #1 (non-empty parameters) lands in Task 8.
    let prefix = format!("{}.", layer.layer_name);
    let parameter_var_ids: BTreeSet<VarId> = wengert
        .var_names
        .iter()
        .filter_map(|(v, name)| name.starts_with(&prefix).then_some(*v))
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

    // Spec §2.3 precondition #2 / §3.1: detect leaks out of the closure.
    //
    // A "leak" is any closure op whose result escapes the closure without
    // going through the residual Add. There are two forms:
    //
    //   (a) A non-closure op (other than the residual Add) reads a
    //       closure op's result.
    //   (b) wengert.output is a closure op's result AND is not the
    //       residual Add's h_after_var.
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
                if other_op.id == residual_add_op_id { continue; }
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

        // (b) wengert.output is a closure op's result (and not the residual
        //     Add's result — which is `h_after_var`).
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

/// Pattern-match residual Add candidates in the non-closure region.
///
/// Spec §1.3 / §3.2 / §3.3 / §3.4. Returns:
/// - `Ok((add_op, h_before, h_after))` when exactly one candidate matches
///   the residual pattern Add(h_before, block_output).
/// - `Err(PartialRefusal::NoResidualAdd)` when zero candidates match.
/// - `Err(PartialRefusal::ParallelResidualBranches)` when ≥2 candidates
///   have DISTINCT h_before values (parallel residual paths).
/// - `Err(PartialRefusal::AmbiguousPatternMatch)` when ≥2 candidates share
///   the SAME h_before (architecturally ambiguous boundary; Task 7 covers).
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

    // Collect ALL candidate residual Adds (outside closure; inputs.len() == 2;
    // exactly one input tainted).
    let mut candidates: Vec<(OpId, VarId, VarId)> = Vec::new(); // (add_op_id, h_before_var, h_after_var)
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
            candidates.push((op.id, h_before, op.result));
        }
    }

    match candidates.len() {
        0 => Err(PartialRefusal::NoResidualAdd),
        1 => Ok(candidates[0]),
        _ => {
            // Multiple candidates: distinguish shared-h_before (ambiguous) from
            // distinct-h_before (parallel branches).
            let first_h_before = candidates[0].1;
            if candidates.iter().all(|(_, h, _)| *h == first_h_before) {
                Err(PartialRefusal::AmbiguousPatternMatch {
                    h_before: first_h_before,
                    candidate_adds: candidates.iter().map(|(op, _, _)| *op).collect(),
                })
            } else {
                Err(PartialRefusal::ParallelResidualBranches {
                    add_ops: candidates.iter().map(|(op, _, _)| *op).collect(),
                })
            }
        }
    }
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
        PruneRefusal::WholeBlockUnsupported { .. } => OverrideRejectReason::PruneWholeBlockUnsupported,
        PruneRefusal::ConflictingPruneDecisions { .. } => OverrideRejectReason::PruneConflictingDecisions,
    }
}

/// Spec §6.1 success-path stderr line. Format:
///   [prune] layer=N name=... role=... applied=true closure_size=K ops_deleted=K residual_add_op=ID
/// Separator convention: key=value throughout (no colons).
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
              with `{prefix}`
  found:      zero matching parameters in the WeightMap. Check layer name /
              index; the requested layer does not exist in the compiled model.
"
        ),
        PruneRefusal::WholeBlockUnsupported { layer_name } => format!(
"prune: whole-block pruning (LayerRole::Block) is not supported in v1.
  requested:  prune {layer_name}  (role=Block)
  supported:  prune {layer_name}.attn  (role=Attention)
              prune {layer_name}.ffn   (role=Ffn)
  workaround: emit two sub-block prune decisions for this layer; their combined
              effect is semantically equivalent to whole-block prune in standard
              pre-norm transformer architectures (NOT equivalent for post-norm,
              parallel, or scaled-residual architectures).
  planned:    whole-block prune tracked for v2 (chain-collapse transformation).
"
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
                assert_eq!(plan.residual_add_op_id, 2);
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
    fn whole_block_refusal_from_planner() {
        // Layer with role=Block — the planner shouldn't emit this in v1's
        // supported flow, but if something does, plan_rewrite refuses
        // immediately per spec §3.6.
        //
        // `infer_role("blocks.7")` returns LayerRole::Block (no .attn/.ffn
        // suffix; matches the `blocks.N` pattern).

        let v_hb: VarId = 100;
        let v_p:  VarId = 200;
        let v_y:  VarId = 300;
        let ops = vec![
            op_unary(0, v_p, v_hb, PrimalOp::Relu),
            op_add  (1, v_y, v_hb, v_p),
        ];
        let wengert = mk_wengert(ops, v_y, &[(v_hb, "h_before"), (v_p, "blocks.7.wq")]);
        let layer = mk_prune_layer(7, "blocks.7");   // Block role — no .attn/.ffn suffix
        let weight_map = WeightMap::default();

        match plan_rewrite(&wengert, &layer, &weight_map) {
            PlanResult::Refused(PruneRefusal::WholeBlockUnsupported { layer_name }) => {
                assert_eq!(layer_name, "blocks.7");
            }
            PlanResult::Ok(plan) => panic!("expected WholeBlockUnsupported, got Ok({plan:?})"),
            PlanResult::Refused(other) => panic!("expected WholeBlockUnsupported, got: {other:?}"),
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
}
