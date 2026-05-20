//! M56: Agent declaration AST nodes. Mirrors `ModelDef`/`ModelMember` (see
//! `decl.rs:48-66`) with agent-specific ownership markers and port derivation.

use crate::decl::{Decorator, FnDef, Param, TypeParam};
use crate::expr::Expr;
use crate::types::TypeExpr;
use crate::{Span, Symbol};
use serde::Serialize;

/// An `agent Name(params): <members>` declaration.
#[derive(Debug, Clone, Serialize)]
pub struct AgentDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<Param>,
    pub members: Vec<AgentMember>,
    pub span: Span,
}

/// A field or method inside an agent body.
#[derive(Debug, Clone, Serialize)]
pub enum AgentMember {
    /// `name: Type = init_expr` field declaration. Decorators may include
    /// `@shared` (M38 semantics — spec §1.5).
    FieldDecl {
        name: Symbol,
        type_ann: TypeExpr,
        init: Option<Expr>,
        decorators: Vec<Decorator>,
        span: Span,
    },
    /// Agent method (init/reset/shutdown or user-defined). Decorators may
    /// include `@auto_device_transfer` (spec §1.6). The method signature is
    /// also the port declaration per spec §1.2 — port names are derived from
    /// parameter names during semantic analysis.
    Method(FnDef, Vec<Decorator>),
}
