use serde::Serialize;
use crate::expr::Expr;
use crate::types::TypeExpr;
use crate::{NodeId, Span, Symbol};

#[derive(Debug, Clone, Serialize)]
pub struct Pattern {
    pub kind: PatternKind,
    pub span: Span,
    pub id: NodeId,
}

#[derive(Debug, Clone, Serialize)]
pub enum PatternKind {
    /// Identifier binding: x
    Ident(Symbol),

    /// Wildcard: _
    Wildcard,

    /// Literal: 42, "hello", true
    Literal(Box<Expr>),

    /// Tuple: (a, b, c)
    Tuple(Vec<Pattern>),

    /// List: [a, b, c]
    List(Vec<Pattern>),

    /// Struct: {field1, field2, ..rest}
    Struct {
        fields: Vec<FieldPattern>,
        rest: Option<Symbol>,
    },

    /// Constructor: Variant(a, b) or Path.Variant(a)
    Constructor {
        path: Vec<Symbol>,
        args: Vec<Pattern>,
    },

    /// Or pattern: a | b
    Or(Vec<Pattern>),

    /// Guarded: pattern : if condition
    Guarded {
        pattern: Box<Pattern>,
        guard: Box<Expr>,
    },

    /// Rest/spread: *args or ..rest
    Rest(Option<Symbol>),

    /// Type-annotated: pattern: Type
    Typed {
        pattern: Box<Pattern>,
        type_ann: TypeExpr,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct FieldPattern {
    pub name: Symbol,
    pub pattern: Option<Pattern>,
    pub span: Span,
}
