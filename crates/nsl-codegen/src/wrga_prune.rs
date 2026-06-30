//! WRGA Innovation 1: Wengert-Pruned Backward Generation (Dead Gradient Elimination).
//!
//! Given a [`WengertList`] (straight-line SSA of the forward pass) and a set of
//! `trainable` parameter [`VarId`]s, this pass computes the subset of forward
//! ops that actually participate in the backward pass for *those* trainable
//! parameters, then emits a pruned Wengert list plus:
//!
//! * a `backward_live` set — VarIds whose adjoint (`ā`) must be materialised
//! * an `activation_live` set — VarIds whose primal value must be saved for
//!   the backward pass
//! * a [`PruneStats`] summary describing compile-time elimination counts
//!
//! This is the physical Dead Gradient Elimination described in Section 2.1 of
//! the WRGA research proposal.  Unlike PyTorch's `requires_grad=False`
//! traversal, the pass eliminates adjoint ops *by not generating them in the
//! first place*.
//!
//! The pass is pure (no global state) and can be called independently of the
//! full WRGA driver — it is used both by `wrga::run` and by the `nsl check
//! --wrga-analyze` CLI subcommand for reporting.

use std::collections::{BTreeSet, HashSet};

use crate::wengert::{PrimalOp, VarId, WengertList};

/// Which primal operations require saving inputs/outputs for the backward pass.
///
/// This mirrors the `DataRequired` classification used by `source_ad.rs`: some
/// ops (e.g. `Matmul`) need at least one input live at backward time; others
/// (e.g. `Relu`) need the *output* live; and some (e.g. `Add`) need neither.
pub(crate) fn save_requirements(op: &PrimalOp) -> SaveRequirements {
    use PrimalOp::*;
    match op {
        // Linear ops — inputs needed
        Matmul | Conv2d { .. } | ConvTranspose2d { .. } | Embedding => SaveRequirements {
            needs_inputs: true,
            needs_output: false,
        },
        // Elementwise nonlinearities — output is typically enough (relu/sigmoid)
        // but some need inputs (gelu/silu require x for exact backward).
        Relu | Sigmoid | Tanh => SaveRequirements {
            needs_inputs: false,
            needs_output: true,
        },
        Gelu | Silu | Exp | Log | Sqrt | Abs | Neg | Clamp { .. } => SaveRequirements {
            needs_inputs: true,
            needs_output: false,
        },
        // Binary elementwise — only mul/div need inputs; add/sub are pure-scalar
        Mul | Div => SaveRequirements {
            needs_inputs: true,
            needs_output: false,
        },
        Add | Sub => SaveRequirements {
            needs_inputs: false,
            needs_output: false,
        },
        Softmax { .. } | LogSoftmax { .. } => SaveRequirements {
            needs_inputs: false,
            needs_output: true,
        },
        LayerNorm { .. } | RMSNorm { .. } | BatchNorm { .. } => SaveRequirements {
            needs_inputs: true,
            needs_output: false,
        },
        ScaledDotProductAttention { .. } | FlashAttentionBackwardExtract { .. } => {
            SaveRequirements { needs_inputs: true, needs_output: true }
        }
        CrossEntropyLoss | MSELoss | L1Loss => SaveRequirements {
            needs_inputs: true,
            needs_output: false,
        },
        // Everything else: conservative — no saves.
        _ => SaveRequirements {
            needs_inputs: false,
            needs_output: false,
        },
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SaveRequirements {
    pub(crate) needs_inputs: bool,
    pub(crate) needs_output: bool,
}

/// Summary of the pruning pass, suitable for `nsl build --wrga-report`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PruneStats {
    /// Total ops in the input (full forward) Wengert list.
    pub forward_ops_total: usize,
    /// Ops that survived the pruning (i.e. still needed for the backward pass
    /// of the specified trainable parameters).
    pub backward_ops_retained: usize,
    /// Number of activations that would have been saved in the *full* backward.
    pub full_backward_saved_activations: usize,
    /// Activations actually saved in the pruned backward.
    pub pruned_saved_activations: usize,
    /// Trainable parameter count (VarIds marked as gradient targets).
    pub gradient_targets: usize,
    /// Parameter VarIds present in the Wengert list but *not* targeted for
    /// gradients — these are the "frozen" params whose backward is eliminated.
    pub frozen_params: usize,
}

impl PruneStats {
    pub fn backward_elimination_ratio(&self) -> f64 {
        if self.forward_ops_total == 0 {
            return 0.0;
        }
        let eliminated = self
            .forward_ops_total
            .saturating_sub(self.backward_ops_retained);
        eliminated as f64 / self.forward_ops_total as f64
    }

    pub fn activation_elimination_ratio(&self) -> f64 {
        if self.full_backward_saved_activations == 0 {
            return 0.0;
        }
        let eliminated = self
            .full_backward_saved_activations
            .saturating_sub(self.pruned_saved_activations);
        eliminated as f64 / self.full_backward_saved_activations as f64
    }
}

/// Result of running the dead gradient elimination pass.
#[derive(Debug, Clone)]
pub struct PruneResult {
    /// Pruned forward list: same ops in original order, but with
    /// `saved_for_backward` flipped off on ops whose output is not needed by
    /// the pruned backward.
    pub pruned: WengertList,
    /// VarIds whose adjoint must be computed (gradient path reaches a
    /// trainable parameter).
    pub backward_live: BTreeSet<VarId>,
    /// VarIds whose primal value must be saved for the backward pass.
    pub activation_live: BTreeSet<VarId>,
    /// Statistics for reporting.
    pub stats: PruneStats,
}

/// Compute the forward cone *reaching* the output: every VarId such that there
/// is a data-dependency path from the VarId to the output.  An op's output is
/// "needed for the gradient" only if it lies on both cones.
fn reverse_reachable_from(list: &WengertList, sink: VarId) -> HashSet<VarId> {
    // Index ops by result VarId for O(1) producer lookup.
    let mut seen = HashSet::new();
    let mut stack = vec![sink];
    while let Some(v) = stack.pop() {
        if !seen.insert(v) {
            continue;
        }
        if let Some(op) = list.find_producer(v) {
            for &inp in &op.inputs {
                if !seen.contains(&inp) {
                    stack.push(inp);
                }
            }
        }
    }
    seen
}

/// Compute the forward cone *out of* the sources: every VarId whose producer
/// transitively depends on at least one of the `sources`.
fn forward_reachable_from(list: &WengertList, sources: &HashSet<VarId>) -> HashSet<VarId> {
    let mut reached: HashSet<VarId> = sources.clone();
    // Straight-line SSA: a single left-to-right sweep is a fixed point because
    // inputs always precede their users in a Wengert list.
    for op in &list.ops {
        if op.inputs.iter().any(|v| reached.contains(v)) {
            reached.insert(op.result);
        }
    }
    reached
}

/// Run the Wengert-pruned backward generation pass.
///
/// * `list`          — the full forward Wengert list (typically produced by
///                     `crate::source_ad::extract_wengert`)
/// * `trainable`     — VarIds of parameters that will receive gradients
///                     (everything else is effectively `@freeze`d)
/// * `loss_output`   — the VarId representing the scalar loss (the adjoint
///                     seed)
///
/// Returns a [`PruneResult`] with the pruned list and live sets.
pub fn prune(
    list: &WengertList,
    trainable: &BTreeSet<VarId>,
    loss_output: VarId,
) -> PruneResult {
    // 1. Backward cone: which VarIds contribute to the loss?
    //    Anything not on this set has no adjoint.  (This is the "output-cone".)
    let to_loss = reverse_reachable_from(list, loss_output);

    // 2. Forward cone from trainable params: which VarIds depend on a
    //    trainable?  Only ops on *this* cone can have a nonzero gradient with
    //    respect to any trainable.
    let mut trainable_hs: HashSet<VarId> = HashSet::new();
    for &v in trainable {
        trainable_hs.insert(v);
    }
    let from_trainable = forward_reachable_from(list, &trainable_hs);

    // 3. Backward-live: intersection of the two cones plus the trainable
    //    params themselves.  The adjoint of a VarId `v` is non-trivially
    //    needed iff v is downstream of a trainable AND upstream of the loss.
    //    If there are no trainables, there is nothing to differentiate with
    //    respect to — the adjoint graph is empty and nothing is live.
    let mut backward_live: BTreeSet<VarId> = BTreeSet::new();
    if !trainable.is_empty() {
        for v in from_trainable.iter().copied() {
            if to_loss.contains(&v) {
                backward_live.insert(v);
            }
        }
        for &t in trainable {
            backward_live.insert(t);
        }
        backward_live.insert(loss_output);
    }

    // 4. Activation-live: a forward op's primal result needs to be saved only
    //    if either (a) it's on the backward_live set and the op's backward
    //    rule requires its output, or (b) it's the input to a live op whose
    //    rule requires its inputs.
    let mut activation_live: BTreeSet<VarId> = BTreeSet::new();
    let mut total_full_saves: usize = 0;

    for op in &list.ops {
        let req = save_requirements(&op.op);
        // Full-backward save budget — a reasonable proxy for the "what
        // PyTorch would save" baseline.
        if req.needs_output || req.needs_inputs {
            total_full_saves += 1;
        }

        if !backward_live.contains(&op.result) {
            continue; // adjoint of this node is never computed, so no saves
        }
        if req.needs_output {
            activation_live.insert(op.result);
        }
        if req.needs_inputs {
            for &inp in &op.inputs {
                activation_live.insert(inp);
            }
        }
    }

    // 5. Build the pruned list.  Ops whose result is not in backward_live *and*
    //    whose result is not on the output path are unreachable in both the
    //    pruned forward (for the loss) and the pruned backward — but we keep
    //    the forward pass intact because the user still needs the loss value.
    //    What we *do* update is the `saved_for_backward` bit: only activations
    //    that the pruned backward will read should be saved.
    let mut pruned = list.clone();
    for op in pruned.ops.iter_mut() {
        op.saved_for_backward = activation_live.contains(&op.result);
    }

    // 6. Count frozen params: anything tagged `Param(...)` but not in trainable.
    let mut frozen_params = 0usize;
    for op in &list.ops {
        if matches!(op.op, PrimalOp::Param(_)) && !trainable.contains(&op.result) {
            frozen_params += 1;
        }
    }

    let stats = PruneStats {
        forward_ops_total: list.len(),
        backward_ops_retained: backward_live.len(),
        full_backward_saved_activations: total_full_saves,
        pruned_saved_activations: activation_live.len(),
        gradient_targets: trainable.len(),
        frozen_params,
    };

    PruneResult {
        pruned,
        backward_live,
        activation_live,
        stats,
    }
}

/// Convenience: walk a Wengert list and return the set of `Param(name)` VarIds
/// whose name matches any pattern in `patterns`.
///
/// Patterns follow a minimal glob syntax: a trailing `*` is a prefix match,
/// otherwise exact match.  This is the same shape used by `@freeze(exclude=[...])`
/// and `@adapter(target=[...])`.
pub fn param_ids_matching(list: &WengertList, patterns: &[&str]) -> BTreeSet<VarId> {
    let mut out = BTreeSet::new();
    for op in &list.ops {
        if let PrimalOp::Param(name) = &op.op {
            if patterns.iter().any(|p| glob_match(p, name)) {
                out.insert(op.result);
            }
        }
    }
    out
}

/// Tiny glob: supports `*` (match any suffix) and exact equality.
pub fn glob_match(pattern: &str, candidate: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        candidate.starts_with(prefix)
    } else {
        pattern == candidate
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertList, WengertOp};
    use std::collections::HashMap;

    fn op(id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: o,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    /// Build a tiny two-layer graph:
    ///   x      (input)      #0
    ///   w1     (param)      #1    (FROZEN)
    ///   w2     (param)      #2    (TRAINABLE)
    ///   h = w1 @ x          #3
    ///   y = w2 @ h          #4
    ///   loss = y            #5  (passthrough)
    fn two_layer() -> (WengertList, u32) {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("w1".into()), vec![]),
            op(2, 2, PrimalOp::Param("w2".into()), vec![]),
            op(3, 3, PrimalOp::Matmul, vec![1, 0]),
            op(4, 4, PrimalOp::Matmul, vec![2, 3]),
        ];
        let list = WengertList {
            ops,
            output: 4,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        (list, 4)
    }

    #[test]
    fn prune_freezes_w1_gradient() {
        let (list, loss) = two_layer();
        // Only w2 is trainable.
        let mut trainable = BTreeSet::new();
        trainable.insert(2);
        let res = prune(&list, &trainable, loss);

        // w2 (2) and y (4) carry an adjoint: y's adjoint is the loss seed,
        // w2's is ∂L/∂w2.  h (3) does *not* need an adjoint — it's frozen-
        // upstream of w2 and we don't propagate gradients to w1.  That's the
        // whole point of dead gradient elimination.
        assert!(res.backward_live.contains(&2));
        assert!(res.backward_live.contains(&4));
        assert!(!res.backward_live.contains(&3));
        // w1 (1) does *not* depend on the trainable, so no adjoint for it.
        assert!(!res.backward_live.contains(&1));
        // x (0) is neither an input to anything downstream of w2 uniquely, but
        // with this graph h = w1 @ x, so x is upstream of w2's matmul via h.
        // The forward cone from w2 includes h and y only; x is *not* in it.
        assert!(!res.backward_live.contains(&0));

        // Matmul needs inputs — for op #4 (w2 @ h) that means w2 (2) and h (3)
        // must be saved.  For op #3 (w1 @ x) we don't enter the backward so
        // its inputs are not saved.
        assert!(res.activation_live.contains(&2));
        assert!(res.activation_live.contains(&3));
        assert!(!res.activation_live.contains(&0));
        assert!(!res.activation_live.contains(&1));

        assert_eq!(res.stats.gradient_targets, 1);
        assert_eq!(res.stats.frozen_params, 1);
    }

    #[test]
    fn prune_everything_trainable_matches_full() {
        let (list, loss) = two_layer();
        let mut trainable = BTreeSet::new();
        trainable.insert(1);
        trainable.insert(2);
        let res = prune(&list, &trainable, loss);
        // Both params upstream of the loss — full backward lives.
        assert!(res.backward_live.contains(&1));
        assert!(res.backward_live.contains(&2));
        assert_eq!(res.stats.frozen_params, 0);
    }

    #[test]
    fn prune_no_trainable_produces_empty_backward() {
        let (list, loss) = two_layer();
        let trainable = BTreeSet::new();
        let res = prune(&list, &trainable, loss);
        // Nothing is trainable — backward_live should contain only the loss
        // seed and nothing else (no real adjoints).
        for v in &res.backward_live {
            assert!(v == &loss, "unexpectedly live VarId {v} with no trainables");
        }
        // No activations saved either.
        assert!(res.activation_live.is_empty());
    }

    #[test]
    fn glob_match_semantics() {
        assert!(glob_match("blocks.6.*", "blocks.6.wq"));
        assert!(glob_match("blocks.6.*", "blocks.6."));
        assert!(!glob_match("blocks.6.*", "blocks.7.wq"));
        assert!(glob_match("norm.weight", "norm.weight"));
        assert!(!glob_match("norm.weight", "norm.bias"));
    }

    #[test]
    fn param_ids_matching_collects_patterns() {
        let (list, _) = two_layer();
        let ids = param_ids_matching(&list, &["w2"]);
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&2));

        let ids = param_ids_matching(&list, &["w*"]);
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn stats_ratios_are_sane() {
        let s = PruneStats {
            forward_ops_total: 100,
            backward_ops_retained: 15,
            full_backward_saved_activations: 60,
            pruned_saved_activations: 10,
            gradient_targets: 2,
            frozen_params: 30,
        };
        assert!((s.backward_elimination_ratio() - 0.85).abs() < 1e-9);
        assert!((s.activation_elimination_ratio() - (50.0 / 60.0)).abs() < 1e-9);
    }
}
