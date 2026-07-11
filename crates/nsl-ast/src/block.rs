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

/// CPKD distillation block: `distill(teacher=t, student=s, epochs=N):`.
///
/// Sections reuse [`TrainSection`] (optimizer/data/step behave identically to
/// `train`); the distill-specific `loss:` section is a flat key=value list
/// (alpha, temperature, feature_layers, feature_weight, attn_transfer) kept
/// outside `sections` because `train` has no loss section.
#[derive(Debug, Clone, Serialize)]
pub struct DistillBlock {
    pub config: Vec<Arg>,
    pub sections: Vec<TrainSection>,
    pub loss: Vec<Arg>,
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
pub struct KeyValueEntry {
    pub key: Symbol,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub enum TokenizerStmt {
    SpecialTokens {
        entries: Vec<KeyValueEntry>,
        span: Span,
    },
    Normalize {
        rules: Vec<Symbol>,
        span: Span,
    },
    PreTokenize {
        rules: Vec<Symbol>,
        span: Span,
    },
    Padding {
        entries: Vec<KeyValueEntry>,
        span: Span,
    },
    Truncation {
        entries: Vec<KeyValueEntry>,
        span: Span,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizerDef {
    pub name: Symbol,
    pub config: Vec<Arg>,
    pub body: Vec<TokenizerStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetDef {
    pub name: Symbol,
    pub source: Expr,
    pub body: Vec<KeyValueEntry>,
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

/// WRGA (Wengert-Pruned Roofline-Guided Adaptation) configuration.
///
/// Parsed from a `@wrga(mode=auto|manual|hybrid, budget=..., target=..., layers=[...])`
/// decorator.  Decorators themselves are already stored as `Decorator` nodes;
/// this struct is the validated, resolved form used by the semantic and codegen
/// passes.  Keeping it in the AST crate lets both `nsl-semantic` and
/// `nsl-codegen` share one vocabulary.
#[derive(Debug, Clone, Serialize)]
pub struct WrgaBlock {
    pub mode: WrgaMode,
    /// Total adapter parameter budget, if specified (e.g. `budget=100K`).
    pub budget: Option<i64>,
    /// Target hardware symbol, e.g. `h100`, `rtx5070ti`.
    pub target: Option<Symbol>,
    /// Explicit layer selection for `mode=hybrid`.
    pub layers: Vec<String>,
    /// WRGA paper §8.2: user-defined custom adapter model.  When set,
    /// the name (a Symbol resolving to a `model` declaration in scope)
    /// tells WRGA to instantiate this user-defined adapter at each
    /// selected site instead of one of the built-in `lora`/`ia3`/
    /// `gatedlora` kinds.  Defaults to `None` (compiler-chosen kind).
    pub adapter: Option<Symbol>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum WrgaMode {
    /// Compiler makes all adapter placement + rank decisions.
    Auto,
    /// User explicitly specifies `@adapter(...)` sites; compiler only optimizes
    /// the backward pass, fusion, and memory.
    Manual,
    /// User selects layer scope; compiler chooses adapter type, rank, fusion.
    Hybrid,
}

impl WrgaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            WrgaMode::Auto => "auto",
            WrgaMode::Manual => "manual",
            WrgaMode::Hybrid => "hybrid",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(WrgaMode::Auto),
            "manual" => Some(WrgaMode::Manual),
            "hybrid" => Some(WrgaMode::Hybrid),
            _ => None,
        }
    }
}

/// serve Name:
///     config entries + nested sub-blocks + @endpoint functions
#[derive(Debug, Clone, Serialize)]
pub struct ServeBlock {
    pub name: Symbol,
    pub config: Vec<ServeConfigEntry>,
    /// CFIE: nested config sections (`speculative:` / `sampling:` /
    /// `grammar:`) — a bare `key:` followed by an indented run of
    /// ordinary config entries.
    pub sub_blocks: Vec<ServeSubBlock>,
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

/// A nested serve config section: `sampling:` + indented `key: value`
/// entries.  Which section names are meaningful is a semantic concern
/// (see nsl-semantic's CFIE validation); the parser accepts any ident.
#[derive(Debug, Clone, Serialize)]
pub struct ServeSubBlock {
    pub key: Symbol,
    pub entries: Vec<ServeConfigEntry>,
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
