use serde::Serialize;
use crate::decl::{Decorator, Param};
use crate::expr::{Arg, Expr};
use crate::stmt::{Block, Stmt};
use crate::types::TypeExpr;
use crate::{Span, Symbol};

#[derive(Debug, Clone, Serialize)]
pub struct TrainBlock {
    pub config: Vec<Arg>,
    pub sections: Vec<TrainSection>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub enum TrainSection {
    Data(Vec<Stmt>),
    Optimizer(Expr),
    Scheduler(Expr),
    Step {
        param: Symbol,
        body: Block,
    },
    Eval {
        param: Symbol,
        body: Block,
    },
    Callbacks(Vec<CallbackDef>),
    Distribute(Expr),
    Stmt(Stmt),
}

#[derive(Debug, Clone, Serialize)]
pub struct CallbackDef {
    pub name: Symbol,
    pub params: Vec<Param>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct GradBlock {
    pub outputs: Option<crate::pattern::Pattern>,
    pub targets: Expr,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantBlock {
    pub config: Vec<Arg>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelDef {
    pub name: Symbol,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Block,
    pub decorators: Vec<Decorator>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizerDef {
    pub name: Symbol,
    pub config: Vec<Arg>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetDef {
    pub name: Symbol,
    pub source: Expr,
    pub body: Vec<Stmt>,
    pub span: Span,
}
