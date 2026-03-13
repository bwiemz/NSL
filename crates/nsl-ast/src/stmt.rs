use serde::Serialize;
use crate::block::*;
use crate::decl::*;
use crate::expr::{Expr, MatchArm};
use crate::operator::AssignOp;
use crate::pattern::Pattern;
use crate::types::TypeExpr;
use crate::{NodeId, Span};

#[derive(Debug, Clone, Serialize)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: Span,
    pub id: NodeId,
}

#[derive(Debug, Clone, Serialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub enum StmtKind {
    /// let x: Type = expr  /  const x: Type = expr
    VarDecl {
        is_const: bool,
        pattern: Pattern,
        type_ann: Option<TypeExpr>,
        value: Option<Expr>,
    },

    /// fn definitions
    FnDef(FnDef),

    /// model definitions
    ModelDef(ModelDef),

    /// struct definitions
    StructDef(StructDef),

    /// enum definitions
    EnumDef(EnumDef),

    /// trait definitions
    TraitDef(TraitDef),

    /// if cond: ... elif cond: ... else: ...
    If {
        condition: Expr,
        then_block: Block,
        elif_clauses: Vec<(Expr, Block)>,
        else_block: Option<Block>,
    },

    /// for pattern in iterable: ...
    For {
        pattern: Pattern,
        iterable: Expr,
        body: Block,
    },

    /// while condition: ...
    While {
        condition: Expr,
        body: Block,
    },

    /// while let pattern = expr: ...
    WhileLet {
        pattern: Pattern,
        expr: Expr,
        body: Block,
    },

    /// match expr: case pat => body ...
    Match {
        subject: Expr,
        arms: Vec<MatchArm>,
    },

    Break,
    Continue,
    Return(Option<Expr>),
    Yield(Option<Expr>),

    /// lvalue op= expr
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
    },

    /// import path.{items}
    Import(ImportStmt),

    /// from path import {items}
    FromImport(FromImportStmt),

    /// train(config): ...
    TrainBlock(TrainBlock),

    /// grad(targets): ...
    GradBlock(GradBlock),

    /// quant(config): ...
    QuantBlock(QuantBlock),

    /// kernel Name(params): ...
    KernelDef(KernelDef),

    /// tokenizer Name(config): ...
    TokenizerDef(TokenizerDef),

    /// dataset Name("id"): ...
    DatasetDef(DatasetDef),

    /// datatype Name: ...
    DatatypeDef(DatatypeDef),

    /// @decorator(args) \n stmt
    Decorated {
        decorators: Vec<Decorator>,
        stmt: Box<Stmt>,
    },

    /// expression statement
    Expr(Expr),
}
