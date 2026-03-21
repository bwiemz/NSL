//! M40: Wengert list (straight-line SSA) for source-to-source AD.

use std::collections::HashMap;

pub type OpId = u32;
pub type VarId = u32;

/// A single operation in the primal (forward) computation trace.
#[derive(Debug, Clone)]
pub struct WengertOp {
    pub id: OpId,
    pub result: VarId,
    pub op: PrimalOp,
    pub inputs: Vec<VarId>,
    pub saved_for_backward: bool,
    pub checkpointed: bool,
}

/// Primitive operations in the computation graph.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimalOp {
    // Elementwise unary
    Relu, Sigmoid, Tanh, Gelu, Silu,
    Exp, Log, Sqrt, Abs, Neg,
    // Elementwise binary
    Add, Sub, Mul, Div,
    // Linear algebra
    Matmul,
    Transpose { dim0: usize, dim1: usize },
    // Reductions
    Sum { dim: Option<i64> },
    Mean { dim: Option<i64> },
    Softmax { dim: i64 },
    // Shape ops
    Reshape { target_ndim: usize },
    Broadcast,
    // Control flow
    /// Conditional select: result = inputs[0] (cond) ? inputs[1] (true_val) : inputs[2] (false_val)
    Select,
    /// Boolean comparison (non-differentiable). Used to save branch conditions.
    /// inputs = [lhs, rhs], stores the comparison kind.
    Condition(CompareKind),
    // Markers
    Input(String),
    Param(String),
    Constant(f64),
}

/// Comparison operators for branch conditions.
#[derive(Debug, Clone, PartialEq)]
pub enum CompareKind {
    Gt,
    Lt,
    GtEq,
    LtEq,
    Eq,
    NotEq,
}

/// Linearized forward computation graph.
#[derive(Debug, Clone)]
pub struct WengertList {
    pub ops: Vec<WengertOp>,
    pub output: VarId,
    pub var_names: HashMap<VarId, String>,
}

impl WengertList {
    pub fn defines(&self, var: VarId) -> bool {
        self.ops.iter().any(|op| op.result == var)
    }

    pub fn find_producer(&self, var: VarId) -> Option<&WengertOp> {
        self.ops.iter().find(|op| op.result == var)
    }

    pub fn is_checkpointed(&self, var: VarId) -> bool {
        self.find_producer(var).map(|op| op.checkpointed).unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp { id, result, op, inputs, saved_for_backward: false, checkpointed: false }
    }

    #[test]
    fn test_wengert_list_defines() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1, var_names: HashMap::new(),
        };
        assert!(list.defines(0));
        assert!(list.defines(1));
        assert!(!list.defines(99));
    }

    #[test]
    fn test_find_producer() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1, var_names: HashMap::new(),
        };
        assert_eq!(list.find_producer(1).unwrap().op, PrimalOp::Relu);
        assert!(list.find_producer(99).is_none());
    }

    #[test]
    fn test_checkpoint_detection() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                WengertOp { id: 1, result: 1, op: PrimalOp::Relu, inputs: vec![0], saved_for_backward: true, checkpointed: true },
            ],
            output: 1, var_names: HashMap::new(),
        };
        assert!(!list.is_checkpointed(0));
        assert!(list.is_checkpointed(1));
    }

    #[test]
    fn test_len_and_empty() {
        let list = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Constant(1.0), vec![])],
            output: 0, var_names: HashMap::new(),
        };
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
    }
}
