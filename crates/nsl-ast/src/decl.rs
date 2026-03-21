use serde::Serialize;
use crate::expr::{Arg, Expr};
use crate::stmt::Block;
use crate::types::TypeExpr;
use crate::{Span, Symbol};

#[derive(Debug, Clone, Serialize)]
pub struct FnDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub effect_params: Vec<Symbol>,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub return_effect: Option<EffectExpr>,
    pub body: Block,
    pub is_async: bool,
    pub span: Span,
}

/// Syntactic effect annotation (before resolution).
#[derive(Debug, Clone, Serialize)]
pub enum EffectExpr {
    /// Effect variable: `E`
    Var(Symbol),
    /// Concrete effect name: `IO`, `RANDOM`, etc.
    Named(Symbol),
    /// Union: `E1 | E2` or `E | {IO}`
    Union(Vec<EffectExpr>),
}

#[derive(Debug, Clone, Serialize)]
pub struct Param {
    pub name: Symbol,
    pub type_ann: Option<TypeExpr>,
    pub default: Option<Expr>,
    pub is_variadic: bool,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypeParam {
    pub name: Symbol,
    pub bounds: Vec<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<Param>,
    pub members: Vec<ModelMember>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub enum ModelMember {
    LayerDecl {
        name: Symbol,
        type_ann: TypeExpr,
        init: Option<Expr>,
        decorators: Vec<Decorator>,
        span: Span,
    },
    Method(FnDef, Vec<Decorator>),
}

#[derive(Debug, Clone, Serialize)]
pub struct StructDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub fields: Vec<StructField>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct StructField {
    pub name: Symbol,
    pub type_ann: TypeExpr,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnumDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub variants: Vec<EnumVariant>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnumVariant {
    pub name: Symbol,
    pub fields: Vec<TypeExpr>,
    pub value: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraitDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub methods: Vec<FnDef>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct Decorator {
    pub name: Vec<Symbol>,
    pub args: Option<Vec<Arg>>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportStmt {
    pub path: Vec<Symbol>,
    pub items: ImportItems,
    pub alias: Option<Symbol>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub enum ImportItems {
    Module,
    Named(Vec<ImportItem>),
    Glob,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportItem {
    pub name: Symbol,
    pub alias: Option<Symbol>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct FromImportStmt {
    pub module_path: Vec<Symbol>,
    pub items: ImportItems,
    pub span: Span,
}
