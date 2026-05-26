//! CEP frontend — AST recognizer: canonical NSL transformer block -> ModelSpec / SearchAxes.

use std::collections::HashMap;

use nsl_ast::decl::{Decorator, ModelDef, ModelMember};
use nsl_ast::expr::{Arg, Expr, ExprKind};
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::StmtKind;
use nsl_ast::types::TypeExprKind;
use nsl_ast::{Module, Symbol};

use crate::cep_oracle::{Activation, ModelSpec, NormType};

/// Recognizer refusals — each names what was expected vs. what was found.
#[derive(Debug, Clone, PartialEq)]
pub enum CepExtractError {
    MissingStructure(String),
    UnrecognizedAttention { expected: String, found: String },
    UnresolvableBinding { dim: String, detail: String },
    UnknownFfnActivation { ffn_type: String },
    UnknownSearchAxis { axis: String },
    InvalidSpec(String),
}

impl std::fmt::Display for CepExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CepExtractError::MissingStructure(m) => write!(f, "CEP: cannot recognize model: {m}"),
            CepExtractError::UnrecognizedAttention { expected, found } => write!(
                f,
                "CEP: unrecognized attention\n  expected: {expected}\n  found:    {found}\n  (GPT-2 fused-QKV and stdlib Attention are a v2 recognizer broadening)"
            ),
            CepExtractError::UnresolvableBinding { dim, detail } => {
                write!(f, "CEP: cannot resolve {dim} to a concrete value: {detail}")
            }
            CepExtractError::UnknownFfnActivation { ffn_type } => {
                write!(
                    f,
                    "CEP: cannot determine activation for FFN type '{ffn_type}' (known: SwiGLUFFN)"
                )
            }
            CepExtractError::UnknownSearchAxis { axis } => {
                write!(
                    f,
                    "CEP: unknown @search axis '{axis}' (known: d_model, n_layers, n_heads, n_kv_heads, d_ff)"
                )
            }
            CepExtractError::InvalidSpec(m) => write!(f, "CEP: extracted spec is invalid: {m}"),
        }
    }
}

pub struct BindingCtx {
    pub params: HashMap<String, i64>,
    pub consts: HashMap<String, i64>,
}

type Resolve<'a> = &'a dyn Fn(Symbol) -> String;

pub fn resolve_binding(
    arg: &Expr,
    ctx: &BindingCtx,
    resolve: Resolve,
    dim: &str,
) -> Result<i64, CepExtractError> {
    match &arg.kind {
        ExprKind::IntLiteral(n) => Ok(*n),
        ExprKind::Ident(sym) => {
            let name = resolve(*sym);
            if let Some(v) = ctx.params.get(&name) {
                Ok(*v)
            } else if let Some(v) = ctx.consts.get(&name) {
                Ok(*v)
            } else {
                Err(CepExtractError::UnresolvableBinding {
                    dim: dim.to_string(),
                    detail: format!(
                        "identifier '{name}' is not a constructor parameter or const"
                    ),
                })
            }
        }
        _ => Err(CepExtractError::UnresolvableBinding {
            dim: dim.to_string(),
            detail: "expected an integer literal, const, or constructor parameter".to_string(),
        }),
    }
}

fn collect_models<'a>(
    module: &'a Module,
    resolve: Resolve,
) -> HashMap<String, (&'a ModelDef, &'a [Decorator])> {
    let mut out = HashMap::new();
    for stmt in &module.stmts {
        let (md, decos): (&ModelDef, &[Decorator]) = match &stmt.kind {
            StmtKind::ModelDef(md) => (md, &[]),
            StmtKind::Decorated { decorators, stmt } => match &stmt.kind {
                StmtKind::ModelDef(md) => (md, decorators.as_slice()),
                _ => continue,
            },
            _ => continue,
        };
        out.insert(resolve(md.name), (md, decos));
    }
    out
}

/// Extract `(name, &init.kind)` from a top-level const/let `VarDecl` whose
/// binding is a single identifier. Returns `None` for destructuring binds or
/// declarations with no value.
fn var_decl_name_and_init<'a>(kind: &'a StmtKind, resolve: Resolve) -> Option<(String, &'a ExprKind)> {
    if let StmtKind::VarDecl { pattern, value, .. } = kind {
        if let PatternKind::Ident(sym) = &pattern.kind {
            if let Some(init_expr) = value {
                return Some((resolve(*sym), &init_expr.kind));
            }
        }
    }
    None
}

/// Collect top-level `const NAME = <int>` declarations.
fn collect_int_consts(module: &Module, resolve: Resolve) -> HashMap<String, i64> {
    let mut consts = HashMap::new();
    for stmt in &module.stmts {
        if let Some((name, ExprKind::IntLiteral(n))) = var_decl_name_and_init(&stmt.kind, resolve) {
            consts.insert(name, *n);
        }
    }
    consts
}

/// Read the simple named-type name from a TypeExpr.
/// `TypeExprKind::Named(Symbol)` is confirmed in crates/nsl-ast/src/types.rs:15.
fn named_type(ty: &nsl_ast::types::TypeExpr, resolve: Resolve) -> Option<String> {
    if let TypeExprKind::Named(sym) = &ty.kind {
        Some(resolve(*sym))
    } else {
        None
    }
}

fn as_call<'a>(expr: &'a Expr, resolve: Resolve) -> Option<(String, &'a [Arg])> {
    if let ExprKind::Call { callee, args } = &expr.kind {
        if let ExprKind::Ident(sym) = &callee.kind {
            return Some((resolve(*sym), args.as_slice()));
        }
    }
    None
}

/// Recursively find the first ListLiteral of exactly `rank` integer literals.
fn first_int_list(expr: &Expr, rank: usize) -> Option<Vec<i64>> {
    match &expr.kind {
        ExprKind::ListLiteral(items) if items.len() == rank => {
            let mut out = Vec::with_capacity(rank);
            for it in items {
                if let ExprKind::IntLiteral(n) = &it.kind {
                    out.push(*n);
                } else {
                    return None;
                }
            }
            Some(out)
        }
        ExprKind::BinaryOp { left, right, .. } => {
            first_int_list(left, rank).or_else(|| first_int_list(right, rank))
        }
        ExprKind::Call { args, .. } => args.iter().find_map(|a| first_int_list(&a.value, rank)),
        _ => None,
    }
}

fn ffn_activation(ffn_type: &str) -> Result<Activation, CepExtractError> {
    match ffn_type {
        "SwiGLUFFN" => Ok(Activation::SwiGlu),
        other => Err(CepExtractError::UnknownFfnActivation {
            ffn_type: other.to_string(),
        }),
    }
}

fn norm_type(name: &str) -> Option<NormType> {
    match name {
        "RMSNorm" => Some(NormType::RmsNorm),
        "LayerNorm" => Some(NormType::LayerNorm),
        _ => None,
    }
}

fn arg_missing(what: &'static str) -> impl Fn() -> CepExtractError {
    move || CepExtractError::MissingStructure(format!("missing constructor argument: {what}"))
}

/// Find the top-level model in source order: the first model that has a
/// `FixedArray`-typed `LayerDecl` field with a concrete initializer.
/// Within a model, a field named `"blocks"` is preferred; otherwise the
/// first such field is taken.  Returns `(model_def, element_type_name,
/// array_size, init_expr)`.
fn find_top_model<'a>(
    module: &'a Module,
    resolve: Resolve,
) -> Option<(&'a ModelDef, String, i64, &'a Expr)> {
    for stmt in &module.stmts {
        let md: &ModelDef = match &stmt.kind {
            StmtKind::ModelDef(md) => md,
            StmtKind::Decorated { stmt, .. } => match &stmt.kind {
                StmtKind::ModelDef(md) => md,
                _ => continue,
            },
            _ => continue,
        };

        // Scan fields: collect FixedArray candidates, preferring "blocks".
        let mut preferred: Option<(String, i64, &Expr)> = None; // field named "blocks"
        let mut fallback: Option<(String, i64, &Expr)> = None; // first FixedArray field

        for member in &md.members {
            let ModelMember::LayerDecl { name, type_ann, init, .. } = member else {
                continue;
            };
            let TypeExprKind::FixedArray { element_type, size } = &type_ann.kind else {
                continue;
            };
            let (Some(elem_name), Some(init_expr)) =
                (named_type(element_type, resolve), init.as_ref())
            else {
                continue;
            };
            if resolve(*name) == "blocks" {
                // Preferred field — take it immediately and stop scanning this model.
                preferred = Some((elem_name, *size, init_expr));
                break;
            } else if fallback.is_none() {
                fallback = Some((elem_name, *size, init_expr));
            }
        }

        if let Some((elem, size, init_expr)) = preferred.or(fallback) {
            return Some((md, elem, size, init_expr));
        }
    }
    None
}

pub fn extract_model_spec(module: &Module, resolve: Resolve) -> Result<ModelSpec, CepExtractError> {
    let models = collect_models(module, resolve);
    let consts = collect_int_consts(module, resolve);

    // 1. Find the top-level model: the first (in source order) model with a
    //    `blocks: [Block; N]`-style field, preferring a field named "blocks".
    let (top_md, block_type, n_layers_i, block_init) =
        find_top_model(module, resolve).ok_or_else(|| {
            CepExtractError::MissingStructure(
                "no top-level model with a `blocks: [Block; N]` field".to_string(),
            )
        })?;

    // Fix 2: guard non-positive n_layers before casting to u32.
    if n_layers_i <= 0 {
        return Err(CepExtractError::InvalidSpec(
            "n_layers must be positive".to_string(),
        ));
    }
    let n_layers = n_layers_i as u32;

    // 2. Resolve the block constructor's positional args into a BindingCtx.
    let (block_md, _) = models.get(&block_type).ok_or_else(|| {
        CepExtractError::MissingStructure(format!(
            "block type '{block_type}' is not defined in the module"
        ))
    })?;
    let (_call_name, block_args) = as_call(block_init, resolve).ok_or_else(|| {
        CepExtractError::MissingStructure(format!(
            "`blocks` initializer is not a {block_type}(...) call"
        ))
    })?;
    let mut params = HashMap::new();
    for (i, param) in block_md.params.iter().enumerate() {
        if let Some(arg) = block_args.get(i) {
            if let Ok(v) = resolve_binding(
                &arg.value,
                &BindingCtx {
                    params: HashMap::new(),
                    consts: consts.clone(),
                },
                resolve,
                "block-param",
            ) {
                params.insert(resolve(param.name), v);
            }
        }
    }
    let ctx = BindingCtx { params, consts };

    // 3. Read attention (GroupedQueryAttention) + FFN + norm from the block.
    let mut attn: Option<Vec<Arg>> = None;
    let mut ffn: Option<(String, Vec<Arg>)> = None;
    let mut norm: Option<NormType> = None;
    for member in &block_md.members {
        let ModelMember::LayerDecl { type_ann, init, .. } = member else {
            continue;
        };
        let tname = named_type(type_ann, resolve);
        if let Some(init_expr) = init {
            if let Some((call_name, args)) = as_call(init_expr, resolve) {
                if call_name == "GroupedQueryAttention" {
                    attn = Some(args.to_vec());
                } else if ffn_activation(&call_name).is_ok() {
                    ffn = Some((call_name, args.to_vec()));
                }
            }
        }
        if let Some(t) = tname.as_deref().and_then(norm_type) {
            norm.get_or_insert(t);
        }
    }

    let attn_args = attn.ok_or_else(|| {
        let found = block_md
            .members
            .iter()
            .find_map(|m| {
                let ModelMember::LayerDecl {
                    init: Some(init), ..
                } = m
                else {
                    return None;
                };
                as_call(init, resolve).map(|(n, _)| n)
            })
            .unwrap_or_else(|| "no constructor call".to_string());
        CepExtractError::UnrecognizedAttention {
            expected: "GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)".to_string(),
            found: format!("attention field is `{found}(...)`, not GroupedQueryAttention"),
        }
    })?;

    let d_model = resolve_binding(
        &attn_args
            .first()
            .ok_or_else(arg_missing("GQA d_model"))?
            .value,
        &ctx,
        resolve,
        "d_model",
    )? as u32;
    let n_heads = resolve_binding(
        &attn_args
            .get(1)
            .ok_or_else(arg_missing("GQA n_heads"))?
            .value,
        &ctx,
        resolve,
        "n_heads",
    )? as u32;
    let n_kv_heads = resolve_binding(
        &attn_args
            .get(2)
            .ok_or_else(arg_missing("GQA n_kv_heads"))?
            .value,
        &ctx,
        resolve,
        "n_kv_heads",
    )? as u32;
    if n_heads == 0 {
        return Err(CepExtractError::InvalidSpec("n_heads is 0".to_string()));
    }
    let head_dim = d_model / n_heads;

    let (ffn_type, ffn_args) = ffn.ok_or_else(|| {
        CepExtractError::MissingStructure(
            "no recognized FFN field in the block (expected SwiGLUFFN(...))".to_string(),
        )
    })?;
    let activation = ffn_activation(&ffn_type)?;
    let d_ff = resolve_binding(
        &ffn_args
            .get(1)
            .ok_or_else(arg_missing("FFN d_ff"))?
            .value,
        &ctx,
        resolve,
        "d_ff",
    )? as u32;

    let norm = norm.unwrap_or(NormType::RmsNorm);

    // 4. Embedding: first 2-int ListLiteral on an Embedding-role field.
    let mut vocab = None;
    for member in &top_md.members {
        let ModelMember::LayerDecl {
            name,
            init: Some(init),
            ..
        } = member
        else {
            continue;
        };
        if crate::wggo_graph::infer_role(&resolve(*name))
            == crate::wggo_graph::LayerRole::Embedding
        {
            if let Some(shape) = first_int_list(init, 2) {
                vocab = Some(shape[0] as u32);
                break;
            }
        }
    }
    let vocab = vocab.ok_or_else(|| {
        CepExtractError::MissingStructure(
            "no embedding tensor (a 2-D `embed`/`embedding` field) found".to_string(),
        )
    })?;

    let max_seq = ctx
        .consts
        .get("MAX_SEQ_LEN")
        .copied()
        .unwrap_or(2048) as u32;

    let spec = ModelSpec {
        d_model,
        n_layers,
        n_heads: vec![n_heads; n_layers as usize],
        n_kv_heads: vec![n_kv_heads; n_layers as usize],
        head_dim: vec![head_dim; n_layers as usize],
        d_ff: vec![d_ff; n_layers as usize],
        vocab,
        max_seq,
        batch: 1,
        activation,
        norm,
        dtype_bytes: 4,
    };
    spec.validate().map_err(CepExtractError::InvalidSpec)?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_oracle::{Activation, NormType};

    fn parse(src: &str) -> (nsl_ast::Module, nsl_lexer::Interner) {
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _d) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        (parsed.module, interner)
    }

    const CANONICAL: &str = r#"
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
    fn forward(self, x: Tensor) -> Tensor:
        return silu(x @ self.w_gate)
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn_norm: RMSNorm = RMSNorm(d_model)
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
model TinyCoder:
    embed: Tensor = randn([4096, 384]) * full([1], 0.02)
    blocks: [TransformerBlock; 6] = TransformerBlock(384, 6, 3, 1024, 0.1)
    norm: RMSNorm = RMSNorm(384)
"#;

    #[test]
    fn extracts_canonical_modelspec() {
        let (module, interner) = parse(CANONICAL);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let spec = extract_model_spec(&module, &resolve).expect("recognized");
        assert_eq!(spec.d_model, 384);
        assert_eq!(spec.n_layers, 6);
        assert_eq!(spec.n_heads, vec![6; 6]);
        assert_eq!(spec.n_kv_heads, vec![3; 6]);
        assert_eq!(spec.head_dim, vec![64; 6]); // 384 / 6
        assert_eq!(spec.d_ff, vec![1024; 6]);
        assert_eq!(spec.vocab, 4096);
        assert_eq!(spec.activation, Activation::SwiGlu);
        assert_eq!(spec.norm, NormType::RmsNorm);
    }

    const HAND_ROLLED: &str = r#"
model ToyAttention:
    wq: Tensor = ones([32, 32])
    fn forward(self, x: Tensor) -> Tensor:
        return softmax(x @ self.wq)
model ToyNet:
    blocks: [ToyAttention; 4] = ToyAttention()
    embed: Tensor = randn([100, 32])
"#;

    const FUSED_QKV: &str = r#"
model GPT2Block(d_model: int, n_heads: int, d_ff: int):
    c_attn: Linear = Linear(d_model, 3 * d_model)
    ln1: LayerNorm = LayerNorm(d_model)
model GPT2:
    blocks: [GPT2Block; 4] = GPT2Block(768, 12, 3072)
    embed: Tensor = randn([50257, 768])
"#;

    const CONST_BOUND: &str = r#"
const D_MODEL = 256
const N_HEADS = 8
const N_KV_HEADS = 4
const D_FF = 512
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model ConstNet:
    embed: Tensor = randn([1000, 256])
    blocks: [TransformerBlock; 2] = TransformerBlock(D_MODEL, N_HEADS, N_KV_HEADS, D_FF, 0.1)
"#;

    const UNRESOLVABLE: &str = r#"
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, mystery, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model BadNet:
    embed: Tensor = randn([1000, 256])
    blocks: [TransformerBlock; 2] = TransformerBlock(256, 8, 4, 512, 0.1)
"#;

    #[test]
    fn refuses_hand_rolled_attention() {
        let (module, interner) = parse(HAND_ROLLED);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let err = extract_model_spec(&module, &resolve).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("GroupedQueryAttention") || msg.contains("FFN"), "got: {msg}");
    }

    #[test]
    fn refuses_fused_qkv() {
        let (module, interner) = parse(FUSED_QKV);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let err = extract_model_spec(&module, &resolve).unwrap_err();
        assert!(matches!(err, CepExtractError::UnrecognizedAttention { .. } | CepExtractError::MissingStructure(_)));
        assert!(err.to_string().contains("GroupedQueryAttention"));
    }

    #[test]
    fn resolves_const_bound_dims() {
        let (module, interner) = parse(CONST_BOUND);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let spec = extract_model_spec(&module, &resolve).expect("recognized");
        assert_eq!(spec.d_model, 256);
        assert_eq!(spec.n_heads, vec![8; 2]);
        assert_eq!(spec.n_kv_heads, vec![4; 2]);
        assert_eq!(spec.d_ff, vec![512; 2]);
        assert_eq!(spec.n_layers, 2);
    }

    #[test]
    fn refuses_unresolvable_binding() {
        let (module, interner) = parse(UNRESOLVABLE);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let err = extract_model_spec(&module, &resolve).unwrap_err();
        assert!(matches!(err, CepExtractError::UnresolvableBinding { .. }), "got: {err}");
    }
}
