//! M40: Wengert list (straight-line SSA) for source-to-source AD.

use std::collections::HashMap;

pub type OpId = u32;
pub type VarId = u32;

/// Cranelift-level type tag for a VarId in the Wengert graph.
///
/// The lowerer needs to know whether a VarId is a tensor pointer (i64),
/// a raw f64 scalar, a raw i64 integer, or an NslList pointer so it can
/// emit the correct IR (e.g., skip `nsl_tensor_item` on non-tensor values).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WengertType {
    /// i64 — NslTensor pointer (from param_list, forward ops like matmul, relu, etc.)
    Tensor,
    /// f64 — raw scalar float (from nsl_tensor_item)
    Scalar,
    /// i64 — raw integer (dimension size, index, from shape subscript or int())
    Integer,
    /// i64 — NslList pointer (from nsl_tensor_shape, list literals)
    List,
}

/// Determine the `WengertType` produced by a `PrimalOp`.
///
/// `Constant` defaults to `Tensor` because most constants in the graph
/// (especially adjoint seeds) are used in tensor arithmetic.  The
/// `WengertExtractor` overrides specific constants to `Scalar`/`Integer`
/// when it knows from context (e.g., subscript index, list element).
pub fn type_for_op(op: &PrimalOp) -> WengertType {
    match op {
        PrimalOp::Constant(_) => WengertType::Tensor, // overridden by extractor when needed
        PrimalOp::Passthrough(name) => match name.as_str() {
            "shape" | "list" => WengertType::List,
            "ndim" | "subscript" | "int" => WengertType::Integer,
            "float" => WengertType::Scalar,
            // "item" stays Tensor: the lowerer wraps the f64 back into a 0-dim
            // scalar tensor so the result can be used in tensor arithmetic (Mul, Add, etc.).
            // If the consumer is "int()", it handles Tensor input via nsl_tensor_item + fcvt.
            _ => WengertType::Tensor, // reshape, contiguous, cos, sin, item, etc.
        },
        _ => WengertType::Tensor, // All tensor ops (Input, Param, Relu, Matmul, ...)
    }
}

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
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Exp,
    Log,
    Sqrt,
    Abs,
    Neg,
    Clamp {
        min: f64,
        max: f64,
    },
    // Elementwise binary
    Add,
    Sub,
    Mul,
    Div,
    // Linear algebra
    Matmul,
    Transpose {
        dim0: usize,
        dim1: usize,
    },
    // Reductions
    Sum {
        dim: Option<i64>,
    },
    Mean {
        dim: Option<i64>,
    },
    Softmax {
        dim: i64,
    },
    LogSoftmax {
        dim: i64,
    },
    // Shape ops
    Reshape {
        target_ndim: usize,
    },
    Broadcast,
    Concat {
        dim: i64,
    },
    Split {
        dim: i64,
        chunks: usize,
    },
    Slice {
        dim: i64,
        start: i64,
        end: i64,
        orig_dim_size: i64,
    },
    /// Zero-pad a sliced gradient back to the original shape along a dimension.
    PadZero {
        dim: i64,
        pad_before: i64,
        pad_after: i64,
    },
    // Indexing
    Gather {
        dim: i64,
    },
    ScatterAdd {
        dim: i64,
    },
    Embedding,
    // Normalization
    LayerNorm {
        eps: f64,
    },
    RMSNorm {
        eps: f64,
    },
    BatchNorm {
        eps: f64,
        training: bool,
    },
    // Pooling
    MaxPool2d {
        kernel: usize,
        stride: usize,
    },
    AvgPool2d {
        kernel: usize,
        stride: usize,
    },
    // Convolution
    Conv2d {
        stride: usize,
        padding: usize,
    },
    /// Transposed convolution (deconvolution) — used in Conv2d backward for input gradient.
    ConvTranspose2d {
        stride: usize,
        padding: usize,
    },
    /// Repeat (upsample) tensor elements by kernel factor — used in AvgPool backward.
    Repeat {
        kernel: usize,
    },
    // Loss functions
    CrossEntropyLoss,
    MSELoss,
    L1Loss,
    // Attention
    ScaledDotProductAttention {
        causal: bool,
    },
    /// Extract one component (0=dQ, 1=dK, 2=dV) from FlashAttention backward.
    FlashAttentionBackwardExtract {
        causal: bool,
        component: u8,
    },
    RoPE {
        dim: usize,
    },
    /// Inverse RoPE rotation (rotate by -θ) — used in backward pass.
    RoPEInverse {
        dim: usize,
    },
    // Regularization
    Dropout {
        p: f64,
    },
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
    /// Non-differentiable passthrough: compiled via the regular codegen path.
    /// The string identifies the operation for the lowering to dispatch.
    /// Zero gradient — these are shape/metadata ops that don't affect gradients.
    Passthrough(String),
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
    /// Cranelift-level type for each VarId (Tensor / Scalar / Integer / List).
    pub var_types: HashMap<VarId, WengertType>,
}

impl WengertList {
    pub fn defines(&self, var: VarId) -> bool {
        self.ops.iter().any(|op| op.result == var)
    }

    pub fn find_producer(&self, var: VarId) -> Option<&WengertOp> {
        self.ops.iter().find(|op| op.result == var)
    }

    pub fn is_checkpointed(&self, var: VarId) -> bool {
        self.find_producer(var)
            .map(|op| op.checkpointed)
            .unwrap_or(false)
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
        WengertOp {
            id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    #[test]
    fn test_wengert_list_defines() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
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
            output: 1,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        assert_eq!(list.find_producer(1).unwrap().op, PrimalOp::Relu);
        assert!(list.find_producer(99).is_none());
    }

    #[test]
    fn test_checkpoint_detection() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                WengertOp {
                    id: 1,
                    result: 1,
                    op: PrimalOp::Relu,
                    inputs: vec![0],
                    saved_for_backward: true,
                    checkpointed: true,
                },
            ],
            output: 1,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        assert!(!list.is_checkpointed(0));
        assert!(list.is_checkpointed(1));
    }

    #[test]
    fn test_len_and_empty() {
        let list = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Constant(1.0), vec![])],
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
    }
}
