//! M40: Source-to-source AD — adjoint generation, dead gradient elimination,
//! and saved tensor analysis.

use std::collections::{HashMap, HashSet};
use crate::ad_rules::{apply_ad_rule, AdjointExpr, InputAdjoint};
use crate::wengert::{OpId, PrimalOp, VarId, WengertList, WengertOp};

// ---------------------------------------------------------------------------
// Adjoint generation
// ---------------------------------------------------------------------------

/// Generates the backward (adjoint) Wengert list from a forward (primal) list.
pub struct AdjointGenerator {
    adjoint_ops: Vec<WengertOp>,
    adjoint_vars: HashMap<VarId, VarId>,
    var_counter: VarId,
    op_counter: OpId,
}

impl AdjointGenerator {
    pub fn new(start_var: VarId) -> Self {
        Self {
            adjoint_ops: Vec::new(),
            adjoint_vars: HashMap::new(),
            var_counter: start_var,
            op_counter: 0,
        }
    }

    fn next_var(&mut self) -> VarId {
        let v = self.var_counter;
        self.var_counter += 1;
        v
    }

    fn next_op(&mut self) -> OpId {
        let o = self.op_counter;
        self.op_counter += 1;
        o
    }

    pub fn get_or_create_adjoint(&mut self, primal_var: VarId) -> VarId {
        if let Some(&adj) = self.adjoint_vars.get(&primal_var) {
            adj
        } else {
            let adj = self.next_var();
            self.adjoint_vars.insert(primal_var, adj);
            adj
        }
    }

    /// Generate the backward Wengert list by walking primal ops in reverse.
    pub fn generate(&mut self, primal: &WengertList) -> WengertList {
        let loss_bar = self.next_var();
        let seed_op_id = self.next_op();
        self.adjoint_vars.insert(primal.output, loss_bar);
        self.adjoint_ops.push(WengertOp {
            id: seed_op_id, result: loss_bar, op: PrimalOp::Constant(1.0),
            inputs: vec![], saved_for_backward: false, checkpointed: false,
        });

        for op in primal.ops.iter().rev() {
            let output_bar = self.get_or_create_adjoint(op.result);
            let input_adjoints = apply_ad_rule(op, output_bar);

            for InputAdjoint { input_var, expr } in input_adjoints {
                let adj_val = self.lower_adjoint_expr(expr);
                self.accumulate_adjoint(input_var, adj_val);
            }
        }

        WengertList {
            ops: self.adjoint_ops.clone(),
            output: loss_bar,
            var_names: HashMap::new(),
        }
    }

    fn lower_adjoint_expr(&mut self, expr: AdjointExpr) -> VarId {
        // Identity: pass-through, no op needed
        if let AdjointExpr::Identity(v) = expr {
            return v;
        }

        let result = self.next_var();
        let op_id = self.next_op();
        let (op, inputs) = match expr {
            AdjointExpr::Identity(_) => unreachable!(),
            AdjointExpr::Negate(v) => (PrimalOp::Neg, vec![v]),
            AdjointExpr::MulElementwise(a, b) => (PrimalOp::Mul, vec![a, b]),
            AdjointExpr::MatmulTransposeLeft(a, b) => (PrimalOp::Matmul, vec![a, b]),
            AdjointExpr::MatmulTransposeRight(a, b) => (PrimalOp::Matmul, vec![a, b]),
            AdjointExpr::Scale(v, _s) => (PrimalOp::Mul, vec![v]),
            AdjointExpr::Broadcast(v) => (PrimalOp::Broadcast, vec![v]),
            AdjointExpr::ScaleBroadcast(v, _n) => (PrimalOp::Broadcast, vec![v]),
            AdjointExpr::Transpose(v, d0, d1) => (PrimalOp::Transpose { dim0: d0, dim1: d1 }, vec![v]),
            AdjointExpr::Reshape(v) => (PrimalOp::Reshape { target_ndim: 0 }, vec![v]),
            AdjointExpr::ExpBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::ReluBackward(y_bar, x) => (PrimalOp::Mul, vec![y_bar, x]),
            AdjointExpr::SigmoidBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::TanhBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::LogBackward(y_bar, x) => (PrimalOp::Div, vec![y_bar, x]),
            AdjointExpr::SqrtBackward(y_bar, y) => (PrimalOp::Div, vec![y_bar, y]),
            AdjointExpr::DivNumeratorBackward(y_bar, b) => (PrimalOp::Div, vec![y_bar, b]),
            AdjointExpr::DivDenominatorBackward(y_bar, a, b) => (PrimalOp::Mul, vec![y_bar, a, b]),
        };
        self.adjoint_ops.push(WengertOp {
            id: op_id, result, op, inputs,
            saved_for_backward: false, checkpointed: false,
        });
        result
    }

    fn accumulate_adjoint(&mut self, var: VarId, value: VarId) {
        if let Some(&existing) = self.adjoint_vars.get(&var) {
            let sum = self.next_var();
            let op_id = self.next_op();
            self.adjoint_ops.push(WengertOp {
                id: op_id, result: sum, op: PrimalOp::Add,
                inputs: vec![existing, value],
                saved_for_backward: false, checkpointed: false,
            });
            self.adjoint_vars.insert(var, sum);
        } else {
            self.adjoint_vars.insert(var, value);
        }
    }

    /// Get the adjoint variable for a primal variable (after generation).
    pub fn adjoint_of(&self, primal_var: VarId) -> Option<VarId> {
        self.adjoint_vars.get(&primal_var).copied()
    }
}

// ---------------------------------------------------------------------------
// Dead gradient elimination
// ---------------------------------------------------------------------------

/// Prune backward ops whose results are never needed by any parameter gradient.
pub fn eliminate_dead_gradients(
    adjoint_ops: &[WengertOp],
    needed_vars: &HashSet<VarId>,
) -> Vec<WengertOp> {
    let mut live_ops = HashSet::new();
    let mut worklist: Vec<VarId> = needed_vars.iter().copied().collect();

    while let Some(var) = worklist.pop() {
        for (i, op) in adjoint_ops.iter().enumerate() {
            if op.result == var && !live_ops.contains(&i) {
                live_ops.insert(i);
                worklist.extend(op.inputs.iter().copied());
            }
        }
    }

    adjoint_ops
        .iter()
        .enumerate()
        .filter(|(i, _)| live_ops.contains(i))
        .map(|(_, op)| op.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Saved tensor analysis
// ---------------------------------------------------------------------------

/// Information about a tensor that must be saved from forward for backward.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedTensorInfo {
    pub var: VarId,
    pub checkpointed: bool,
}

/// Identify which forward-pass intermediates must be saved for the backward pass.
pub fn analyze_saved_tensors(
    primal: &WengertList,
    adjoint: &WengertList,
) -> Vec<SavedTensorInfo> {
    let primal_vars: HashSet<VarId> = primal.ops.iter().map(|op| op.result).collect();
    let adjoint_vars: HashSet<VarId> = adjoint.ops.iter().map(|op| op.result).collect();

    let mut saved = Vec::new();
    for adj_op in &adjoint.ops {
        for &input in &adj_op.inputs {
            if primal_vars.contains(&input) && !adjoint_vars.contains(&input) {
                saved.push(SavedTensorInfo {
                    var: input,
                    checkpointed: primal.is_checkpointed(input),
                });
            }
        }
    }

    saved.sort_by_key(|s| s.var);
    saved.dedup_by_key(|s| s.var);
    saved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertList, WengertOp};

    fn make_op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp { id, result, op, inputs, saved_for_backward: false, checkpointed: false }
    }

    #[test]
    fn test_adjoint_generator_simple_add() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("a".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("b".into()), vec![]),
                make_op(2, 2, PrimalOp::Add, vec![0, 1]),
            ],
            output: 2, var_names: HashMap::new(),
        };
        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);
        assert!(!adjoint.ops.is_empty());
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    #[test]
    fn test_adjoint_generator_matmul() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("A".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("B".into()), vec![]),
                make_op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            output: 2, var_names: HashMap::new(),
        };
        let mut gen = AdjointGenerator::new(10);
        let _adjoint = gen.generate(&primal);
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    #[test]
    fn test_dead_gradient_elimination() {
        let ops = vec![
            make_op(0, 10, PrimalOp::Mul, vec![3, 4]),
            make_op(1, 11, PrimalOp::Add, vec![5, 6]),
            make_op(2, 12, PrimalOp::Neg, vec![7]),
            make_op(3, 3, PrimalOp::Add, vec![10, 8]),
            make_op(4, 13, PrimalOp::Mul, vec![9, 10]),
        ];
        let needed = HashSet::from([3_u32]);
        let pruned = eliminate_dead_gradients(&ops, &needed);
        assert!(pruned.len() < ops.len());
        assert!(pruned.iter().any(|op| op.result == 3));
    }

    #[test]
    fn test_dead_gradient_empty_needed() {
        let ops = vec![make_op(0, 10, PrimalOp::Add, vec![0, 1])];
        let pruned = eliminate_dead_gradients(&ops, &HashSet::new());
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_saved_tensor_analysis() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1, var_names: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![
                make_op(0, 10, PrimalOp::Constant(1.0), vec![]),
                make_op(1, 11, PrimalOp::Mul, vec![10, 0]),
            ],
            output: 10, var_names: HashMap::new(),
        };
        let saved = analyze_saved_tensors(&primal, &adjoint);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].var, 0);
    }

    #[test]
    fn test_saved_tensor_no_cross_reference() {
        let primal = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Input("x".into()), vec![])],
            output: 0, var_names: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![make_op(0, 10, PrimalOp::Constant(1.0), vec![])],
            output: 10, var_names: HashMap::new(),
        };
        assert!(analyze_saved_tensors(&primal, &adjoint).is_empty());
    }
}
