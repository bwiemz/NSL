use serde::Serialize;
use crate::expr::Expr;
use crate::{NodeId, Span, Symbol};

#[derive(Debug, Clone, Serialize)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
    pub id: NodeId,
}

#[derive(Debug, Clone, Serialize)]
pub enum TypeExprKind {
    /// Simple named type: int, float, bool, str, fp32, etc.
    Named(Symbol),

    /// Generic type: list<int>, dict<str, float>
    Generic {
        name: Symbol,
        args: Vec<TypeExpr>,
    },

    /// Tensor<[shape], dtype, device>
    Tensor {
        shape: Vec<DimExpr>,
        dtype: Symbol,
        device: Option<DeviceExpr>,
    },

    /// Param<[shape], dtype>
    Param {
        shape: Vec<DimExpr>,
        dtype: Symbol,
    },

    /// Buffer<[shape], dtype>
    Buffer {
        shape: Vec<DimExpr>,
        dtype: Symbol,
    },

    /// Sparse<[shape], dtype, format>
    Sparse {
        shape: Vec<DimExpr>,
        dtype: Symbol,
        format: Symbol,
    },

    /// (A, B) -> C
    Function {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
    },

    /// A | B
    Union(Vec<TypeExpr>),

    /// (A, B, C)
    Tuple(Vec<TypeExpr>),

    /// _
    Wildcard,

    /// Fixed-size array type: [TransformerBlock; 12]
    FixedArray {
        element_type: Box<TypeExpr>,
        size: i64,
    },

    /// Immutable borrow: &T
    Borrow(Box<TypeExpr>),
}

#[derive(Debug, Clone, Serialize)]
pub enum DimExpr {
    /// Concrete dimension: 768
    Concrete(i64),
    /// Symbolic dimension: batch, seq
    Symbolic(Symbol),
    /// Named dimension: batch="B", heads=12
    Named { name: Symbol, value: DimValue },
    /// Bounded symbolic dimension: SeqLen < 4096
    Bounded { name: Symbol, upper_bound: i64 },
    /// Wildcard: _
    Wildcard,
}

#[derive(Debug, Clone, Serialize)]
pub enum DimValue {
    String(String),
    Int(i64),
}

#[derive(Debug, Clone, Serialize)]
pub enum DeviceExpr {
    Cpu,
    Cuda(Option<Box<Expr>>),
    Metal,
    Rocm(Option<Box<Expr>>),
    Npu(Symbol),
}
