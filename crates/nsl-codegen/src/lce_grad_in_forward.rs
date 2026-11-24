//! The fused LM head's gradient computed in its forward.
//!
//! Source AD gives the fused linear cross-entropy (`FusedLinearCe`) an adjoint
//! of two or three `FusedLinearCeBackwardExtract` ops, and their backward
//! launch recomputes every logits chunk the forward already computed. When
//! the upstream gradient the backward will hand the head is known while the
//! forward runs, the forward can finish the head's whole backward with its
//! logits at hand (`nsl_fused_linear_ce_forward_gemm_grad`): the per-row
//! losses, the x gradient, and the weight gradient added straight into the
//! parameter's FASE accumulation buffer. The backward then only hands the
//! saved x gradient on.
//!
//! That is only correct if the scale the forward used is the one the backward
//! receives. This module plans a head only where that is a fact of the tape:
//!
//! - the extracts' upstream gradient (`inputs[0]`) IS the loss seed, the
//!   adjoint's `Constant` (1) for the step's loss, so the head's output is the
//!   loss or a term that reaches it through gradient-preserving ops only (an
//!   `Add` passes the seed on unchanged; a scale such as `0.37 * loss` makes a
//!   different value and keeps the head in its backward);
//! - the weight gradient (component 1) is a parameter's whole adjoint, or one
//!   half of the tied-embedding sum `crate::tied_embed_fusion` accumulates in
//!   place, so it has a buffer to go to and no other reader;
//! - the head is biasless (the GEMM route) and nothing recomputes it.
//!
//! The FASE hook's `1 / N` is the accumulation's alpha, as for the tied
//! embedding. The backward checks at run time that the upstream gradient is
//! the value the forward assumed (`nsl_fused_lce_check_upstream_grad`) and
//! stops the program otherwise.
//!
//! The planned extracts stop reading `x`, `W` and the targets
//! ([`rewrite_adjoint`]): every later pass (CCR's splice, the last-use frees,
//! the storage plan, the arena) then sees that the backward no longer needs
//! the head's input, and the forward frees it right after the head when
//! nothing else reads it ([`LceGifHead::free_x`]).

use crate::tied_embed_fusion::TiedEmbedPlan;
use crate::wengert::{PrimalOp, VarId, WengertList, WengertOp};
use std::collections::{HashMap, HashSet};

/// One LM head whose backward runs inside its forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LceGifHead {
    /// The `FusedLinearCe` result (the extracts' `inputs[5]`).
    pub key: VarId,
    /// The head's input `x` (the op's `inputs[0]`).
    pub x: VarId,
    /// The x-gradient extract (component 0), when the adjoint keeps one.
    pub dx: Option<VarId>,
    /// The weight-gradient extract (component 1).
    pub dw: VarId,
    /// The parameter adjoint whose accumulation buffer receives dW: the dW
    /// extract itself, or the tied-embedding sum it is half of.
    pub param_adjoint: VarId,
    /// dW reaches the parameter through the tied-embedding sum.
    pub tied: bool,
    /// The upstream gradient the forward assumes: the loss seed's literal.
    pub seed: f64,
    /// No op but the head reads `x` after the forward: the forward frees it.
    pub free_x: bool,
}

/// The heads of one train step whose gradient is computed in the forward.
#[derive(Debug, Default, Clone)]
pub struct LceGifPlan {
    pub by_key: HashMap<VarId, LceGifHead>,
    /// dW extract result -> key.
    pub by_dw: HashMap<VarId, VarId>,
}

impl LceGifPlan {
    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    /// The head whose dW extract is `dw` and IS the parameter's adjoint (not
    /// tied): the FASE hook is told its gradient is already in place.
    pub fn direct_dw(&self, dw: VarId) -> Option<&LceGifHead> {
        self.by_dw
            .get(&dw)
            .and_then(|k| self.by_key.get(k))
            .filter(|h| !h.tied && h.param_adjoint == dw)
    }
}

/// What [`plan`] reads.
pub struct PlanInputs<'a> {
    pub primal: &'a WengertList,
    /// The adjoint after the tape optimizations, before CCR's splice.
    pub adjoint: &'a WengertList,
    /// The loss seed: the adjoint of `primal.output`.
    pub seed: VarId,
    /// Parameter adjoints with an accumulation buffer the forward can write.
    pub param_adj_set: &'a HashSet<VarId>,
    /// The tied-embedding plan the adjoint lowering will make (empty when it
    /// will make none).
    pub tied: &'a TiedEmbedPlan,
    /// Primal values CCR recomputes, compresses or frees on its own.
    pub ccr_handled: &'a HashSet<VarId>,
}

fn is_extract(op: &PrimalOp) -> Option<u8> {
    match op {
        PrimalOp::FusedLinearCeBackwardExtract { component, .. } => Some(*component),
        _ => None,
    }
}

/// Plan every head the module header admits. The second value says, per
/// head left in its backward, why (for the build log).
pub fn plan(inp: &PlanInputs<'_>) -> (LceGifPlan, Vec<String>) {
    let mut out = LceGifPlan::default();
    let mut declined = Vec::new();
    let adjoint_producer: HashMap<VarId, &WengertOp> =
        inp.adjoint.ops.iter().map(|o| (o.result, o)).collect();
    let mut adjoint_readers: HashMap<VarId, usize> = HashMap::new();
    for op in &inp.adjoint.ops {
        if matches!(op.op, PrimalOp::FreeTensor) {
            continue;
        }
        for &v in &op.inputs {
            *adjoint_readers.entry(v).or_insert(0) += 1;
        }
    }
    // The loss seed: `generate`'s `Constant`, the value every planned head's
    // forward scales by.
    let seed = adjoint_producer.get(&inp.seed).and_then(|o| match o.op {
        PrimalOp::Constant(c) if o.inputs.is_empty() => Some(c),
        _ => None,
    });
    for (idx, op) in inp.primal.ops.iter().enumerate() {
        let PrimalOp::FusedLinearCe { has_bias, .. } = op.op else {
            continue;
        };
        let key = op.result;
        let mut decline = |why: &str| declined.push(format!("head {key}: {why}"));
        if has_bias {
            decline("it has a bias");
            continue;
        }
        if op.inputs.len() != 4 {
            decline("its operands are not [x, W, bias, targets]");
            continue;
        }
        if inp.ccr_handled.contains(&key) {
            decline("checkpointing recomputes it");
            continue;
        }
        let Some(seed) = seed.filter(|c| *c == 1.0) else {
            decline("the loss seed is not the constant 1");
            continue;
        };
        let extracts: Vec<&WengertOp> = inp
            .adjoint
            .ops
            .iter()
            .filter(|o| is_extract(&o.op).is_some() && o.inputs.get(5) == Some(&key))
            .collect();
        if extracts
            .iter()
            .any(|o| o.inputs.len() != 6 || o.inputs[0] != inp.seed)
        {
            decline("its upstream gradient is not the loss seed");
            continue;
        }
        let of = |c: u8| -> Vec<VarId> {
            extracts
                .iter()
                .filter(|o| is_extract(&o.op) == Some(c))
                .map(|o| o.result)
                .collect()
        };
        let (dxs, dws) = (of(0), of(1));
        if dws.len() != 1 || dxs.len() > 1 || extracts.len() != dws.len() + dxs.len() {
            decline(
                "its adjoint is not one weight-gradient extract and at most one x-gradient extract",
            );
            continue;
        }
        let dw = dws[0];
        let (param_adjoint, tied) = if let Some(&pa) = inp.tied.by_dw.get(&dw) {
            if inp.tied.by_lce_key.get(&key) != Some(&pa) {
                decline("the tied-embedding plan does not pair its weight gradient with it");
                continue;
            }
            (pa, true)
        } else if inp.param_adj_set.contains(&dw)
            && adjoint_readers.get(&dw).copied().unwrap_or(0) == 0
        {
            (dw, false)
        } else {
            decline(
                "its weight gradient is neither a parameter's accumulated gradient nor a tied embedding's half",
            );
            continue;
        };
        let x = op.inputs[0];
        let x_is_computed = inp
            .primal
            .ops
            .iter()
            .take(idx)
            .any(|o| o.result == x && !matches!(o.op, PrimalOp::Input(_) | PrimalOp::Param(_)));
        let read_later = inp
            .primal
            .ops
            .iter()
            .skip(idx + 1)
            .any(|o| o.inputs.contains(&x));
        let extract_ids: HashSet<VarId> = extracts.iter().map(|o| o.result).collect();
        let read_by_backward = inp
            .adjoint
            .ops
            .iter()
            .any(|o| !extract_ids.contains(&o.result) && o.inputs.contains(&x));
        let free_x = x_is_computed
            && !read_later
            && !read_by_backward
            && x != inp.primal.output
            && x != key
            && !inp.ccr_handled.contains(&x);
        let head = LceGifHead {
            key,
            x,
            dx: dxs.first().copied(),
            dw,
            param_adjoint,
            tied,
            seed,
            free_x,
        };
        out.by_dw.insert(dw, key);
        out.by_key.insert(key, head);
    }
    (out, declined)
}

/// Point every planned extract's `x`, `W`, bias and targets operands at the
/// head's key: the lowering reads only the upstream gradient and the key, and
/// no later pass may think the backward still needs the head's input.
pub fn rewrite_adjoint(adjoint: &mut WengertList, plan: &LceGifPlan) {
    for op in &mut adjoint.ops {
        if is_extract(&op.op).is_none() {
            continue;
        }
        let Some(&key) = op.inputs.get(5) else {
            continue;
        };
        if plan.by_key.contains_key(&key) {
            for i in 1..5 {
                op.inputs[i] = key;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tied_embed_fusion::TiedEmbedFusion;

    fn op(id: u32, result: VarId, o: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: o,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn list(ops: Vec<WengertOp>, output: VarId) -> WengertList {
        WengertList {
            ops,
            output,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    fn lce(has_bias: bool) -> PrimalOp {
        PrimalOp::FusedLinearCe {
            vocab_size: 128,
            hidden_size: 64,
            batch_size: 2,
            seq_len: 8,
            vocab_tile: 128,
            ignore_index: -100,
            is_large: false,
            has_bias,
            x_rank3: true,
        }
    }

    fn extract(component: u8) -> PrimalOp {
        PrimalOp::FusedLinearCeBackwardExtract {
            component,
            vocab_size: 128,
            hidden_size: 64,
            batch_size: 2,
            seq_len: 8,
            vocab_tile: 128,
            ignore_index: -100,
            has_bias: false,
            x_rank3: true,
        }
    }

    /// Primal: h=1 (input), g=2 (param), W=3 (param), t=4 (input),
    /// x = rmsnorm(h, g) = 5, loss = lce(x, W, W, t) = 6.
    fn primal(bias: bool) -> WengertList {
        list(
            vec![
                op(0, 1, PrimalOp::Input("h".into()), vec![]),
                op(1, 2, PrimalOp::Param("g".into()), vec![]),
                op(2, 3, PrimalOp::Param("W".into()), vec![]),
                op(3, 4, PrimalOp::Input("t".into()), vec![]),
                op(
                    4,
                    5,
                    PrimalOp::RMSNorm {
                        eps: crate::wengert::NormEps::Const(1e-5),
                    },
                    vec![1, 2],
                ),
                op(5, 6, lce(bias), vec![5, 3, 3, 4]),
            ],
            6,
        )
    }

    /// Adjoint: seed=100, dx=101, dW=102 (W's whole adjoint), dh=103.
    fn adjoint(seed: f64) -> WengertList {
        list(
            vec![
                op(1000, 100, PrimalOp::Constant(seed), vec![]),
                op(1001, 101, extract(0), vec![100, 5, 3, 3, 4, 6]),
                op(1002, 102, extract(1), vec![100, 5, 3, 3, 4, 6]),
                op(
                    1003,
                    103,
                    PrimalOp::Passthrough("rmsnorm_backward".into()),
                    vec![101, 1, 2],
                ),
            ],
            103,
        )
    }

    fn run(
        p: &WengertList,
        a: &WengertList,
        params: &[VarId],
        tied: &TiedEmbedPlan,
    ) -> (LceGifPlan, Vec<String>) {
        let set: HashSet<VarId> = params.iter().copied().collect();
        plan(&PlanInputs {
            primal: p,
            adjoint: a,
            seed: 100,
            param_adj_set: &set,
            tied,
            ccr_handled: &HashSet::new(),
        })
    }

    #[test]
    fn a_head_whose_loss_is_the_seed_is_planned_and_frees_its_input() {
        let (p, why) = run(
            &primal(false),
            &adjoint(1.0),
            &[102],
            &TiedEmbedPlan::default(),
        );
        let h = p
            .by_key
            .get(&6)
            .unwrap_or_else(|| panic!("not planned: {why:?}"));
        assert_eq!(
            (h.x, h.dx, h.dw, h.param_adjoint, h.tied, h.seed, h.free_x),
            (5, Some(101), 102, 102, false, 1.0, true)
        );
        assert!(p.direct_dw(102).is_some());
    }

    #[test]
    fn nothing_is_planned_outside_the_proof() {
        let none = TiedEmbedPlan::default();
        // A seed that is not 1 (or not a constant).
        assert!(
            run(&primal(false), &adjoint(0.5), &[102], &none)
                .0
                .is_empty()
        );
        // A biased head.
        assert!(
            run(&primal(true), &adjoint(1.0), &[102], &none)
                .0
                .is_empty()
        );
        // dW is not a parameter's adjoint.
        assert!(run(&primal(false), &adjoint(1.0), &[], &none).0.is_empty());
        // The upstream gradient is a scaled seed, not the seed.
        let mut a = adjoint(1.0);
        a.ops
            .insert(1, op(1004, 104, PrimalOp::Mul, vec![100, 100]));
        a.ops[2].inputs[0] = 104;
        a.ops[3].inputs[0] = 104;
        assert!(run(&primal(false), &a, &[102], &none).0.is_empty());
        // dW read by another op besides the hook.
        let mut a = adjoint(1.0);
        a.ops.push(op(1005, 105, PrimalOp::Neg, vec![102]));
        assert!(run(&primal(false), &a, &[102], &none).0.is_empty());
        // Checkpointing recomputes the head.
        let set = HashSet::from([102]);
        let (p, _) = plan(&PlanInputs {
            primal: &primal(false),
            adjoint: &adjoint(1.0),
            seed: 100,
            param_adj_set: &set,
            tied: &none,
            ccr_handled: &HashSet::from([6]),
        });
        assert!(p.is_empty());
    }

    #[test]
    fn a_free_marker_does_not_count_as_a_reader_of_dw() {
        let mut a = adjoint(1.0);
        a.ops.push(op(1006, 106, PrimalOp::FreeTensor, vec![102]));
        assert!(
            !run(&primal(false), &a, &[102], &TiedEmbedPlan::default())
                .0
                .is_empty()
        );
    }

    #[test]
    fn x_is_kept_when_anything_else_reads_it() {
        let none = TiedEmbedPlan::default();
        // A later primal op.
        let mut p = primal(false);
        p.ops.push(op(6, 7, PrimalOp::Neg, vec![5]));
        assert!(!run(&p, &adjoint(1.0), &[102], &none).0.by_key[&6].free_x);
        // A backward op other than the extracts.
        let mut a = adjoint(1.0);
        a.ops[3].inputs.push(5);
        assert!(!run(&primal(false), &a, &[102], &none).0.by_key[&6].free_x);
        // An input of the step (not a value the step computed).
        let mut p = primal(false);
        p.ops[5].inputs[0] = 1;
        let mut a = adjoint(1.0);
        a.ops[1].inputs[1] = 1;
        a.ops[2].inputs[1] = 1;
        a.ops[3].inputs = vec![101, 2];
        assert!(!run(&p, &a, &[102], &none).0.by_key[&6].free_x);
    }

    #[test]
    fn a_tied_head_goes_to_the_sum_the_embedding_plan_made() {
        // dW=102 + scatter=107 -> g=108 (the embedding's adjoint).
        let mut a = adjoint(1.0);
        a.ops.push(op(
            1007,
            107,
            PrimalOp::Passthrough("embedding_backward".into()),
            vec![103, 1, 3],
        ));
        a.ops.push(op(1008, 108, PrimalOp::Add, vec![102, 107]));
        let mut tied = TiedEmbedPlan::default();
        let f = TiedEmbedFusion {
            param_adjoint: 108,
            dw: 102,
            lce_key: 6,
            emb: 107,
        };
        tied.by_param_adjoint.insert(108, f);
        tied.by_lce_key.insert(6, 108);
        tied.by_dw.insert(102, 108);
        tied.by_emb.insert(107, 108);
        let (p, why) = run(&primal(false), &a, &[108], &tied);
        let h = p
            .by_key
            .get(&6)
            .unwrap_or_else(|| panic!("not planned: {why:?}"));
        assert_eq!((h.param_adjoint, h.tied), (108, true));
        assert!(p.direct_dw(102).is_none());
        // Without the embedding plan the sum reads dW: not planned.
        assert!(
            run(&primal(false), &a, &[108], &TiedEmbedPlan::default())
                .0
                .is_empty()
        );
    }

    #[test]
    fn the_rewrite_drops_the_heads_operands_from_its_extracts() {
        let (p, _) = run(
            &primal(false),
            &adjoint(1.0),
            &[102],
            &TiedEmbedPlan::default(),
        );
        let mut a = adjoint(1.0);
        rewrite_adjoint(&mut a, &p);
        for o in &a.ops[1..3] {
            assert_eq!(o.inputs, vec![100, 6, 6, 6, 6, 6]);
        }
        assert!(!a.ops.iter().any(|o| o.inputs.contains(&5)));
    }
}
