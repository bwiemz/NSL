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
    Stmt(Box<Stmt>),
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
    pub kind: QuantKind,
    pub name: Symbol,
    pub source: Symbol,
    pub default_dtype: Option<QuantDtype>,
    pub default_granularity: Option<QuantGranularity>,
    pub exclude: Vec<String>,
    pub calibration: Option<CalibrationConfig>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub enum QuantKind {
    Static,
}

#[derive(Debug, Clone, Serialize)]
pub enum QuantDtype {
    Int8,
    Int4,
    Awq4,
    Gptq4,
    Gptq8,
}

#[derive(Debug, Clone, Serialize)]
pub enum QuantGranularity {
    PerTensor,
    PerChannel(i64),
    PerGroup(i64, i64),
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationConfig {
    pub data: Symbol,
    pub samples: i64,
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

/// Custom datatype definition: `datatype Ternary158:`
#[derive(Debug, Clone, Serialize)]
pub struct DatatypeDef {
    pub name: Symbol,
    pub bits: Option<u8>,
    pub block_size: Option<u32>,
    pub methods: Vec<DatatypeMethod>,
    pub ptx_blocks: Vec<DatatypePtxBlock>,
    pub span: Span,
}

/// A method inside a datatype block (e.g., @pack, @unpack)
#[derive(Debug, Clone, Serialize)]
pub struct DatatypeMethod {
    pub kind: DatatypeMethodKind,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DatatypeMethodKind {
    Pack,
    Unpack,
    BackwardPack,
    Arithmetic,
}

/// PTX escape hatch: @pack_ptx(ptx="...")
#[derive(Debug, Clone, Serialize)]
pub struct DatatypePtxBlock {
    pub kind: DatatypePtxKind,
    pub ptx_source: String,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DatatypePtxKind {
    PackPtx,
    UnpackPtx,
    ArithmeticPtx,
}

/// serve Name:
///     config entries + @endpoint functions
#[derive(Debug, Clone, Serialize)]
pub struct ServeBlock {
    pub name: Symbol,
    pub config: Vec<ServeConfigEntry>,
    pub endpoints: Vec<EndpointDef>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct ServeConfigEntry {
    pub key: Symbol,
    pub type_ann: Option<TypeExpr>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct EndpointDef {
    pub name: Symbol,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Block,
    pub span: Span,
}
