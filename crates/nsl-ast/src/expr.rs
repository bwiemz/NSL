use serde::Serialize;
use crate::operator::{BinOp, UnaryOp};
use crate::pattern::Pattern;
use crate::types::TypeExpr;
use crate::{NodeId, Span, Symbol};

#[derive(Debug, Clone, Serialize)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
    pub id: NodeId,
}

#[derive(Debug, Clone, Serialize)]
pub enum ExprKind {
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    FString(Vec<FStringPart>),
    BoolLiteral(bool),
    NoneLiteral,

    ListLiteral(Vec<Expr>),
    TupleLiteral(Vec<Expr>),
    DictLiteral(Vec<(Expr, Expr)>),

    Ident(Symbol),
    SelfRef,

    BinaryOp {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
    },

    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
    },

    Pipe {
        left: Box<Expr>,
        right: Box<Expr>,
    },

    MemberAccess {
        object: Box<Expr>,
        member: Symbol,
    },

    Subscript {
        object: Box<Expr>,
        index: Box<SubscriptKind>,
    },

    Call {
        callee: Box<Expr>,
        args: Vec<Arg>,
    },

    Lambda {
        params: Vec<LambdaParam>,
        body: Box<Expr>,
    },

    ListComp {
        element: Box<Expr>,
        generators: Vec<CompGenerator>,
    },

    IfExpr {
        condition: Box<Expr>,
        then_expr: Box<Expr>,
        else_expr: Box<Expr>,
    },

    MatchExpr {
        subject: Box<Expr>,
        arms: Vec<MatchArm>,
    },

    Range {
        start: Option<Box<Expr>>,
        end: Option<Box<Expr>>,
        inclusive: bool,
    },

    Paren(Box<Expr>),

    Await(Box<Expr>),

    Error,
}

#[derive(Debug, Clone, Serialize)]
pub enum FStringPart {
    Text(String),
    Expr(Expr),
}

#[derive(Debug, Clone, Serialize)]
pub enum SubscriptKind {
    Index(Expr),
    Slice {
        lower: Option<Expr>,
        upper: Option<Expr>,
        step: Option<Expr>,
    },
    MultiDim(Vec<SubscriptKind>),
}

#[derive(Debug, Clone, Serialize)]
pub struct Arg {
    pub name: Option<Symbol>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct LambdaParam {
    pub name: Symbol,
    pub type_ann: Option<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompGenerator {
    pub pattern: Pattern,
    pub iterable: Expr,
    pub conditions: Vec<Expr>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: crate::stmt::Block,
    pub span: Span,
}
