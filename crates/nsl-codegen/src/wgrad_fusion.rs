//! Item 7 (`--fuse-wgrad-accum`): collapse the source-AD weight-gradient
//! chain into a single accumulating GEMM.
//!
//! Source AD lowers `d_loss/d_W` for `Y = X @ W` as three tape ops
//! ([`crate::source_ad`], `AdjointExpr::MatmulTransposeRight`):
//!
//! ```text
//!   a_t      = Transpose(X, -2, -1)          // view
//!   raw_grad = Matmul(a_t, G)                // [B, d, o]  <-- B x the weight
//!   dW       = reduce_to_shape(raw_grad, W)  // [d, o]     sums the batch dims
//! ```
//!
//! and the FASE Deferred hook then folds `dW` in with
//! `m_partial += accum_scale * dW`.
//!
//! The batched matmul materializes a tensor `B` times the size of the
//! parameter and the reduce reads all of it back. Because `X^T @ G` summed
//! over the leading dims is exactly the flattened contraction
//! `[d, B*T] x [B*T, o]`, the whole chain — including the accumulate — is
//! one cuBLAS call with `beta = 1.0` writing straight into `m_partial`.
//! See `nsl_tensor_wgrad_accum` in nsl-runtime.
//!
//! NOT BIT-EXACT, by construction: the fused form sums the `B*T` products in
//! cuBLAS's own order rather than rounding each per-batch partial before the
//! reduce. Measured on an RTX 5070 Ti against an f64 CPU reference, it is
//! *closer* to the true value than the chain it replaces in 2 of 3 shapes
//! (see `nsl-runtime/tests/wgrad_accum_fused.rs`) — but "different, usually
//! better" is still different, which is why the flag defaults off and is
//! documented under the same tolerance contract as `--fuse-rmsnorm-backward`.
//!
//! # What this module does NOT check
//!
//! Shapes. The tape does not carry reliable shape information at this point,
//! so the *runtime* validates the preconditions (matching leading dims,
//! contiguous, GPU-resident f32) and falls back to the exact decomposed chain
//! when they do not hold. A misjudgement here therefore costs performance,
//! never correctness. Everything this module checks is about *aliasing and
//! lifetime* — the things the runtime cannot see.

use crate::wengert::{PrimalOp, VarId, WengertList};
use std::collections::{HashMap, HashSet};

/// One fusable weight-gradient chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgradFusion {
    /// The PRE-transpose activation (`X`). The fused GEMM transposes it
    /// internally via cuBLAS `op`, so the tape's transpose never runs.
    pub x: VarId,
    /// The upstream gradient (`G`).
    pub g: VarId,
    /// Tape index of the `Transpose` op.
    pub transpose_idx: usize,
    /// Tape index of the `Matmul` op.
    pub matmul_idx: usize,
    /// Tape index of the `reduce_to_shape` op (== the param-adjoint result).
    pub reduce_idx: usize,
}

/// The set of chains this tape can fuse, plus the op results whose lowering
/// must be suppressed.
#[derive(Debug, Default, Clone)]
pub struct WgradFusionPlan {
    /// `reduce_to_shape` result VarId → the chain that produced it.
    pub by_reduce_result: HashMap<VarId, WgradFusion>,
    /// VarIds produced by the transpose and matmul of a fused chain. These
    /// ops emit NOTHING and their results stay unmapped — [`plan`] has
    /// already proven no other op reads them.
    pub suppressed: HashSet<VarId>,
}

impl WgradFusionPlan {
    pub fn is_empty(&self) -> bool {
        self.by_reduce_result.is_empty()
    }
}

/// Is this the `[-2, -1]` transpose that `MatmulTransposeRight` emits?
///
/// Source AD encodes "last two dims" as the sentinels `usize::MAX - 1` /
/// `usize::MAX` so the same tape op works for any rank. A transpose of two
/// *concrete* dims is a different operation and must not be fused away.
fn is_last_two_dims_transpose(op: &PrimalOp) -> bool {
    matches!(
        op,
        PrimalOp::Transpose { dim0, dim1 }
            if *dim0 == usize::MAX - 1 && *dim1 == usize::MAX
    )
}

fn is_reduce_to_shape(op: &PrimalOp) -> bool {
    matches!(op, PrimalOp::Passthrough(name) if name == "reduce_to_shape")
}

/// Build the fusion plan for `wengert`, considering only chains that
/// terminate in a parameter adjoint listed in `param_adj_set`.
///
/// A chain is fusable only when ALL of the following hold. Each is a real
/// hazard, not a stylistic preference:
///
/// 1. **The three ops are contiguous** (`i`, `i+1`, `i+2`). This is exactly
///    what source AD emits. Requiring adjacency is what makes it safe to
///    move the *consumption* of `X` and `G` forward from the matmul to the
///    reduce: nothing — no CCR `FreeTensor`, no in-place op, no other
///    reader — can run in between. A tape transform that separates them
///    simply disables the fusion instead of silently changing meaning.
/// 2. **`a_t` and `raw_grad` each have exactly one reader.** Their producing
///    ops are deleted, so a second reader would ghost-skip on an unmapped
///    VarId and lose a gradient silently.
/// 3. **The param adjoint itself has no later reader.** This is the
///    shared-intermediate case the FASE hook calls `still_needed` — the
///    materialized `dW` is genuinely required downstream, so there is
///    nothing to elide.
/// 4. **`X` is not the fused output.** A self-referential chain would read
///    `m_partial` while writing it.
pub fn plan(wengert: &WengertList, param_adj_set: &HashSet<VarId>) -> WgradFusionPlan {
    let mut out = WgradFusionPlan::default();
    if param_adj_set.is_empty() {
        return out;
    }

    // How many ops read each VarId as an input. Multiplicity matters: an op
    // that reads the same VarId twice (`x * x`) counts as two readers, which
    // correctly disqualifies it.
    let mut read_count: HashMap<VarId, usize> = HashMap::new();
    for op in &wengert.ops {
        for &inp in &op.inputs {
            *read_count.entry(inp).or_insert(0) += 1;
        }
    }

    for (i, t_op) in wengert.ops.iter().enumerate() {
        // Walk forward from a candidate transpose so the contiguity check is
        // structural rather than a lookup that could stride over other ops.
        if !is_last_two_dims_transpose(&t_op.op) || t_op.inputs.len() != 1 {
            continue;
        }
        let (Some(m_op), Some(r_op)) = (wengert.ops.get(i + 1), wengert.ops.get(i + 2)) else {
            continue;
        };

        // Shape of the chain.
        if m_op.op != PrimalOp::Matmul || m_op.inputs.len() != 2 {
            continue;
        }
        if !is_reduce_to_shape(&r_op.op) || r_op.inputs.len() != 2 {
            continue;
        }

        // Wiring: transpose feeds the matmul's LEFT operand, matmul feeds the
        // reduce's value operand. `MatmulTransposeLeft` produces a transpose
        // on the RIGHT and must not match here.
        let a_t = t_op.result;
        let raw_grad = m_op.result;
        if m_op.inputs[0] != a_t || r_op.inputs[0] != raw_grad {
            continue;
        }

        // (3) Only fuse chains that terminate in a parameter gradient the
        // FASE hook is going to consume — otherwise there is no accumulator
        // to fold into and the materialized dW is the actual result.
        let dw = r_op.result;
        if !param_adj_set.contains(&dw) {
            continue;
        }

        let x = t_op.inputs[0];
        let g = m_op.inputs[1];

        // (2) Single reader for each deleted intermediate.
        if read_count.get(&a_t).copied().unwrap_or(0) != 1
            || read_count.get(&raw_grad).copied().unwrap_or(0) != 1
        {
            continue;
        }

        // (3) The param adjoint must be dead after the hook consumes it.
        if read_count.get(&dw).copied().unwrap_or(0) != 0 {
            continue;
        }

        // (4) No self-reference.
        if x == dw || g == dw {
            continue;
        }

        out.suppressed.insert(a_t);
        out.suppressed.insert(raw_grad);
        out.by_reduce_result.insert(
            dw,
            WgradFusion {
                x,
                g,
                transpose_idx: i,
                matmul_idx: i + 1,
                reduce_idx: i + 2,
            },
        );
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

    fn t_op(id: u32, result: VarId, input: VarId) -> WengertOp {
        op(
            id,
            result,
            PrimalOp::Transpose {
                dim0: usize::MAX - 1,
                dim1: usize::MAX,
            },
            vec![input],
        )
    }

    fn reduce(id: u32, result: VarId, raw: VarId, target: VarId) -> WengertOp {
        op(
            id,
            result,
            PrimalOp::Passthrough("reduce_to_shape".into()),
            vec![raw, target],
        )
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

    /// x=0, g=1, w=2 ; a_t=10, raw=11, dw=12
    fn canonical_chain() -> Vec<WengertOp> {
        vec![
            t_op(0, 10, 0),
            op(1, 11, PrimalOp::Matmul, vec![10, 1]),
            reduce(2, 12, 11, 2),
        ]
    }

    fn adj(v: VarId) -> HashSet<VarId> {
        HashSet::from([v])
    }

    #[test]
    fn canonical_wgrad_chain_is_fused() {
        let p = plan(&list(canonical_chain()), &adj(12));
        let f = p.by_reduce_result.get(&12).expect("chain must fuse");
        assert_eq!((f.x, f.g), (0, 1));
        assert_eq!(p.suppressed, HashSet::from([10, 11]));
    }

    #[test]
    fn chain_not_terminating_in_a_param_adjoint_is_left_alone() {
        // ANTI-VACUITY for the whole module: the pattern alone is not
        // enough. `x^T @ g` reduced to a shape appears in the adjoint of
        // ordinary (non-parameter) matmuls too, and those results are read
        // by later adjoint ops rather than folded into an accumulator.
        let p = plan(&list(canonical_chain()), &HashSet::new());
        assert!(p.is_empty(), "no param adjoint → nothing to fuse");
        assert!(p.suppressed.is_empty());
    }

    #[test]
    fn second_reader_of_the_raw_grad_blocks_fusion() {
        // The matmul's result feeds the reduce AND something else. Deleting
        // the matmul would leave that other op reading an unmapped VarId,
        // which ghost-skips — a SILENTLY dropped gradient.
        let mut ops = canonical_chain();
        ops.push(op(3, 13, PrimalOp::Relu, vec![11]));
        let p = plan(&list(ops), &adj(12));
        assert!(p.is_empty(), "raw_grad has a second reader → must not fuse");
    }

    #[test]
    fn second_reader_of_the_transpose_blocks_fusion() {
        let mut ops = canonical_chain();
        ops.push(op(3, 13, PrimalOp::Relu, vec![10]));
        let p = plan(&list(ops), &adj(12));
        assert!(p.is_empty(), "a_t has a second reader → must not fuse");
    }

    #[test]
    fn later_reader_of_the_param_gradient_blocks_fusion() {
        // `still_needed` in the FASE hook: this dW is a shared intermediate
        // that a later adjoint op consumes, so it has to be materialized.
        let mut ops = canonical_chain();
        ops.push(op(3, 13, PrimalOp::Relu, vec![12]));
        let p = plan(&list(ops), &adj(12));
        assert!(p.is_empty(), "dW is read later → must not fuse");
    }

    #[test]
    fn non_contiguous_chain_is_not_fused() {
        // A CCR FreeTensor (or any other op) landing between the matmul and
        // the reduce breaks the guarantee that nothing mutates or frees X/G
        // between where the tape consumes them and where the fused GEMM
        // would. Fail SAFE: no fusion, exact chain, no speedup.
        let mut ops = canonical_chain();
        ops.insert(2, op(9, 99, PrimalOp::FreeTensor, vec![0]));
        let p = plan(&list(ops), &adj(12));
        assert!(p.is_empty(), "non-contiguous chain must not fuse");
    }

    #[test]
    fn concrete_dim_transpose_is_not_the_wgrad_pattern() {
        // Transpose(1, 2) on a 4-D tensor is a real data movement with
        // different semantics from the rank-agnostic [-2,-1] sentinel pair.
        let mut ops = canonical_chain();
        ops[0] = op(
            0,
            10,
            PrimalOp::Transpose { dim0: 1, dim1: 2 },
            vec![0],
        );
        let p = plan(&list(ops), &adj(12));
        assert!(p.is_empty(), "concrete-dim transpose must not fuse");
    }

    #[test]
    fn transpose_on_the_matmul_right_operand_is_not_fused() {
        // MatmulTransposeLeft: dA = grad @ B^T. Same three op KINDS, but the
        // transpose feeds the RIGHT operand — fusing it as a wgrad accumulate
        // would compute a completely different contraction.
        let ops = vec![
            t_op(0, 10, 0),
            op(1, 11, PrimalOp::Matmul, vec![1, 10]),
            reduce(2, 12, 11, 2),
        ];
        let p = plan(&list(ops), &adj(12));
        assert!(p.is_empty(), "transposed RIGHT operand must not fuse");
    }

    #[test]
    fn reduce_reading_the_matmul_as_its_target_is_not_fused() {
        // reduce_to_shape(value, target): only inputs[0] is the value being
        // reduced. A tape where the matmul lands in the target slot is not
        // this pattern.
        let ops = vec![
            t_op(0, 10, 0),
            op(1, 11, PrimalOp::Matmul, vec![10, 1]),
            reduce(2, 12, 2, 11),
        ];
        let p = plan(&list(ops), &adj(12));
        assert!(p.is_empty(), "matmul in the target slot must not fuse");
    }

    #[test]
    fn two_independent_chains_both_fuse() {
        // The common case: every Linear in the model contributes one chain.
        let mut ops = canonical_chain();
        ops.extend([
            t_op(3, 20, 5),
            op(4, 21, PrimalOp::Matmul, vec![20, 6]),
            reduce(5, 22, 21, 7),
        ]);
        let p = plan(&list(ops), &HashSet::from([12, 22]));
        assert_eq!(p.by_reduce_result.len(), 2);
        assert_eq!(p.suppressed, HashSet::from([10, 11, 20, 21]));
    }
}
