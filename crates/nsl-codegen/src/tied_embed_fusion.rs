//! A tied embedding's gradient accumulated in place.
//!
//! With the token embedding tied to the LM head (`x @ embed.transpose(0, 1)`
//! under the fused linear cross-entropy), source AD gives the embedding two
//! gradient contributions and sums them:
//!
//! ```text
//!   dW  = FusedLinearCeBackwardExtract { component: 1 }(..)   // LM head, [V, H]
//!   dE  = Passthrough("embedding_backward")(y_bar, ids, embed) // scatter, [V, H]
//!   g   = Add(dW, dE)                                          // the param adjoint
//! ```
//!
//! and the FASE Deferred hook then folds `g` in with `m_partial +=
//! accum_scale · g`. Per micro-batch that is two zero-filled `[V, H]` f32
//! buffers (402 MiB each at 49152 x 2048), a `[V, H]` add and the
//! scale-accumulate — ~4 ms at Coder-1B, more than the embedding's own
//! forward and backward kernels.
//!
//! This pass sends both contributions straight into `m_partial`: the LM
//! head's dW GEMMs run with `alpha = accum_scale, beta = 1` on `m_partial`
//! (`nsl_fused_linear_ce_backward_gemm_acc`) and the embedding backward
//! scatter-adds `accum_scale · dy` rows into it
//! (`nsl_embedding_backward_accum`). The two extract results become
//! placeholders and the `Add` is suppressed; the hook is told the gradient
//! is already accumulated (`ParamGradSource::AccumulatedInPlace`).
//!
//! NOT BIT-EXACT: `m + s·(dW + dE)` becomes `(m + s·dW) + s·dE`, each
//! embedding row's tokens added into `m` one at a time. `NSL_TIED_EMBED_INPLACE=0`
//! (compile time) keeps the materialised sum.
//!
//! What this module checks is aliasing and lifetime (the shapes are the
//! runtime's to check): the two contributions and the param adjoint have no
//! reader but the `Add` (and the CCR `FreeTensor` markers, which free a
//! placeholder), and the LM head's backward takes the GEMM route the
//! in-place accumulation needs. The lowering decides the rest: a FASE hook
//! with a device-resident accumulator (`Compiler::fase_inplace`).

use crate::wengert::{PrimalOp, VarId, WengertList};
use std::collections::{HashMap, HashSet};

/// One tied embedding whose gradient is accumulated in place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiedEmbedFusion {
    /// The `Add` result: the embedding parameter's adjoint.
    pub param_adjoint: VarId,
    /// The LM head's dW extract (`FusedLinearCeBackwardExtract`, component 1).
    pub dw: VarId,
    /// That extract's forward key (its `inputs[5]`): the `FusedLinearCe`
    /// result the backward launch is cached under.
    pub lce_key: VarId,
    /// The embedding scatter (`Passthrough("embedding_backward")`).
    pub emb: VarId,
}

/// The tied embeddings of one adjoint tape whose gradient goes in place.
#[derive(Debug, Default, Clone)]
pub struct TiedEmbedPlan {
    pub by_param_adjoint: HashMap<VarId, TiedEmbedFusion>,
    /// LM-head forward key → param adjoint.
    pub by_lce_key: HashMap<VarId, VarId>,
    /// dW extract result → param adjoint.
    pub by_dw: HashMap<VarId, VarId>,
    /// Embedding scatter result → param adjoint.
    pub by_emb: HashMap<VarId, VarId>,
}

impl TiedEmbedPlan {
    pub fn is_empty(&self) -> bool {
        self.by_param_adjoint.is_empty()
    }
}

fn is_dw_extract(op: &PrimalOp) -> bool {
    matches!(
        op,
        PrimalOp::FusedLinearCeBackwardExtract {
            component: 1,
            has_bias: false,
            ..
        }
    )
}

fn is_embedding_backward(op: &PrimalOp) -> bool {
    matches!(op, PrimalOp::Passthrough(name) if name == "embedding_backward")
}

/// Plan the in-place accumulation for the param adjoints in `param_adj_set`.
/// `gemm_route` says whether the LM head's backward takes the GEMM-chunked
/// route (`nsl_fused_linear_ce_backward_gemm`), the only one whose dW can be
/// accumulated into an existing buffer; without it nothing is planned.
pub fn plan(
    adjoint: &WengertList,
    param_adj_set: &HashSet<VarId>,
    gemm_route: bool,
) -> TiedEmbedPlan {
    let mut out = TiedEmbedPlan::default();
    if param_adj_set.is_empty() || !gemm_route {
        return out;
    }
    // Readers of each VarId, the CCR free markers aside (they free whatever
    // the value lowers to, here a placeholder).
    let mut readers: HashMap<VarId, usize> = HashMap::new();
    for op in &adjoint.ops {
        if matches!(op.op, PrimalOp::FreeTensor) {
            continue;
        }
        for &v in &op.inputs {
            *readers.entry(v).or_insert(0) += 1;
        }
    }
    let producer: HashMap<VarId, &crate::wengert::WengertOp> =
        adjoint.ops.iter().map(|o| (o.result, o)).collect();
    // dW extracts per LM-head launch. The launch is cached under its forward
    // key and, once planned in place, its dW slot is a placeholder: every
    // dW extract of that key would read it. Only a key with exactly one dW
    // extract (the one feeding the Add) may be planned.
    let mut dw_extracts: HashMap<VarId, usize> = HashMap::new();
    for o in &adjoint.ops {
        if matches!(o.op, PrimalOp::FusedLinearCeBackwardExtract { component: 1, .. })
            && let Some(&k) = o.inputs.get(5)
        {
            *dw_extracts.entry(k).or_insert(0) += 1;
        }
    }
    let mut keys_seen: HashSet<VarId> = HashSet::new();
    for op in &adjoint.ops {
        if op.op != PrimalOp::Add || op.inputs.len() != 2 || !param_adj_set.contains(&op.result) {
            continue;
        }
        let (a, b) = (op.inputs[0], op.inputs[1]);
        let (Some(pa), Some(pb)) = (producer.get(&a), producer.get(&b)) else {
            continue;
        };
        let (dw_op, emb_op) = if is_dw_extract(&pa.op) && is_embedding_backward(&pb.op) {
            (*pa, *pb)
        } else if is_dw_extract(&pb.op) && is_embedding_backward(&pa.op) {
            (*pb, *pa)
        } else {
            continue;
        };
        let Some(&lce_key) = dw_op.inputs.get(5) else {
            continue;
        };
        // Each contribution is read by the Add alone, and the adjoint is
        // consumed by the hook alone: nothing reads a value that is never
        // materialised.
        if a == b
            || readers.get(&dw_op.result).copied().unwrap_or(0) != 1
            || readers.get(&emb_op.result).copied().unwrap_or(0) != 1
            || readers.get(&op.result).copied().unwrap_or(0) != 0
        {
            continue;
        }
        // One LM head, one accumulator: a head whose dW fed two parameters
        // cannot write both, and a second dW extract of the same launch
        // would read the placeholder.
        if dw_extracts.get(&lce_key).copied().unwrap_or(0) != 1 || !keys_seen.insert(lce_key) {
            continue;
        }
        let f = TiedEmbedFusion {
            param_adjoint: op.result,
            dw: dw_op.result,
            lce_key,
            emb: emb_op.result,
        };
        out.by_lce_key.insert(lce_key, op.result);
        out.by_dw.insert(f.dw, op.result);
        out.by_emb.insert(f.emb, op.result);
        out.by_param_adjoint.insert(op.result, f);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::WengertOp;

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

    fn extract(component: u8, has_bias: bool) -> PrimalOp {
        PrimalOp::FusedLinearCeBackwardExtract {
            component,
            vocab_size: 128,
            hidden_size: 64,
            batch_size: 2,
            seq_len: 8,
            vocab_tile: 128,
            ignore_index: -100,
            has_bias,
            x_rank3: true,
        }
    }

    fn list(ops: Vec<WengertOp>) -> WengertList {
        let output = ops.last().map(|o| o.result).unwrap_or(0);
        WengertList {
            ops,
            output,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    /// seed=1, x=2, W=3, b=4(placeholder), t=5, key=6, ids=7, y_bar=8;
    /// dx=20, dW=21, dE=22, g=23.
    fn tied(has_bias: bool) -> Vec<WengertOp> {
        vec![
            op(0, 20, extract(0, has_bias), vec![1, 2, 3, 4, 5, 6]),
            op(1, 21, extract(1, has_bias), vec![1, 2, 3, 4, 5, 6]),
            op(
                2,
                22,
                PrimalOp::Passthrough("embedding_backward".into()),
                vec![8, 7, 3],
            ),
            op(3, 23, PrimalOp::Add, vec![21, 22]),
        ]
    }

    #[test]
    fn the_tied_sum_is_planned() {
        let p = plan(&list(tied(false)), &HashSet::from([23]), true);
        let f = p.by_param_adjoint.get(&23).expect("planned");
        assert_eq!((f.dw, f.emb, f.lce_key), (21, 22, 6));
        assert_eq!(p.by_lce_key.get(&6), Some(&23));
        assert_eq!(p.by_dw.get(&21), Some(&23));
        assert_eq!(p.by_emb.get(&22), Some(&23));
    }

    #[test]
    fn either_operand_order_is_planned() {
        let mut ops = tied(false);
        ops[3].inputs = vec![22, 21];
        assert!(!plan(&list(ops), &HashSet::from([23]), true).is_empty());
    }

    #[test]
    fn a_free_marker_does_not_count_as_a_reader() {
        let mut ops = tied(false);
        ops.push(op(4, 30, PrimalOp::FreeTensor, vec![21]));
        ops.push(op(5, 31, PrimalOp::FreeTensor, vec![23]));
        assert!(!plan(&list(ops), &HashSet::from([23]), true).is_empty());
    }

    #[test]
    fn nothing_is_planned_outside_the_proven_pattern() {
        let adj = HashSet::from([23]);
        // Not a parameter adjoint (no accumulator to write).
        assert!(plan(&list(tied(false)), &HashSet::from([99]), true).is_empty());
        // The LM head's backward off the GEMM route.
        assert!(plan(&list(tied(false)), &adj, false).is_empty());
        // A biased head (its v1 kernels and dbias need zeroed buffers).
        assert!(plan(&list(tied(true)), &adj, true).is_empty());
        // dW read by another op: it must be materialised.
        let mut ops = tied(false);
        ops.push(op(4, 40, PrimalOp::Mul, vec![21, 21]));
        assert!(plan(&list(ops), &adj, true).is_empty());
        // The scatter read by another op.
        let mut ops = tied(false);
        ops.push(op(4, 40, PrimalOp::Neg, vec![22]));
        assert!(plan(&list(ops), &adj, true).is_empty());
        // The adjoint read after the hook (a shared intermediate).
        let mut ops = tied(false);
        ops.push(op(4, 40, PrimalOp::Neg, vec![23]));
        assert!(plan(&list(ops), &adj, true).is_empty());
        // The x-gradient component, not dW.
        let mut ops = tied(false);
        ops[3].inputs = vec![20, 22];
        assert!(plan(&list(ops), &adj, true).is_empty());
        // Two embeddings: the sum is not dW + scatter.
        let mut ops = tied(false);
        ops[1] = op(
            1,
            21,
            PrimalOp::Passthrough("embedding_backward".into()),
            vec![8, 7, 3],
        );
        assert!(plan(&list(ops), &adj, true).is_empty());
    }

    #[test]
    fn one_head_feeds_one_accumulator() {
        // A second parameter adjoint summing the same head's dW with another
        // scatter: neither is planned. Planning the first would make the
        // shared launch write dW in place and hand the second extract (24)
        // the placeholder instead of a materialised dW.
        let mut ops = tied(false);
        ops.push(op(4, 24, extract(1, false), vec![1, 2, 3, 4, 5, 6]));
        ops.push(op(
            5,
            25,
            PrimalOp::Passthrough("embedding_backward".into()),
            vec![8, 7, 3],
        ));
        ops.push(op(6, 26, PrimalOp::Add, vec![24, 25]));
        let p = plan(&list(ops), &HashSet::from([23, 26]), true);
        assert!(p.is_empty(), "{p:?}");
    }

    #[test]
    fn a_second_dw_extract_of_the_launch_blocks_the_plan() {
        // Another reader of the same launch's dW (here a plain gradient sum
        // that is not a parameter adjoint) would get the placeholder.
        let mut ops = tied(false);
        ops.push(op(4, 24, extract(1, false), vec![1, 2, 3, 4, 5, 6]));
        ops.push(op(5, 40, PrimalOp::Neg, vec![24]));
        assert!(plan(&list(ops), &HashSet::from([23]), true).is_empty());
        // A dW extract of a different launch does not.
        let mut ops = tied(false);
        ops.push(op(4, 24, extract(1, false), vec![1, 2, 3, 4, 5, 9]));
        ops.push(op(5, 40, PrimalOp::Neg, vec![24]));
        assert!(!plan(&list(ops), &HashSet::from([23]), true).is_empty());
    }
}
