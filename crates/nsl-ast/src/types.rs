use serde::Serialize;
use crate::expr::Expr;
use crate::{NodeId, Span, Symbol};

#[derive(Debug, Clone, Serialize)]
pub struct TypeExpr {
    pub kind: TypeExprKind,
    pub span: Span,
    pub id: NodeId,
}

/// The error for a `Sparse<...>` type, shared by the parser (which refuses
/// the `Sparse<...>` spelling) and the resolver (which refuses a bare
/// `Sparse` that names no user type). General sparse tensors (M50) had no
/// constructor, so no program could hold one; the type, its code generation
/// and the runtime sparse library were removed in 2026-11.
pub const SPARSE_TYPE_REMOVED: &str = "`Sparse<...>` types were removed: general sparse tensors (M50) had no \
     constructor, so no program could hold one; the code is preserved at tag `attic/sparse-tensors-2026-11`. \
     Use a dense `Tensor`";

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

    /// (A, B) -> C | E
    Function {
        params: Vec<TypeExpr>,
        ret: Box<TypeExpr>,
        effect: Option<crate::decl::EffectExpr>,
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
