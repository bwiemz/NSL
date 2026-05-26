//! CEP frontend — AST recognizer: canonical NSL transformer block -> ModelSpec / SearchAxes.

use std::collections::HashMap;

use nsl_ast::decl::{Decorator, ModelDef, ModelMember};
use nsl_ast::expr::{Arg, Expr, ExprKind};
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

/// Collect top-level `const NAME = <int>` declarations.
fn collect_int_consts(_module: &Module, _resolve: Resolve) -> HashMap<String, i64> {
    // TODO(Task 3): parse const NAME = <int>
    // Task 2 happy path uses integer literals; const parsing lands in Task 3.
    // Stub returns empty; do NOT block the happy path on it.
    HashMap::new()
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

pub fn extract_model_spec(module: &Module, resolve: Resolve) -> Result<ModelSpec, CepExtractError> {
    let models = collect_models(module, resolve);
    let consts = collect_int_consts(module, resolve);

    // 1. Find the top-level model: the one with a `blocks: [Block; N]` field.
    let mut top = None;
    for (md, _decos) in models.values() {
        for member in &md.members {
            if let ModelMember::LayerDecl { type_ann, init, .. } = member {
                if let TypeExprKind::FixedArray { element_type, size } = &type_ann.kind {
                    if let (Some(block_name), Some(init_expr)) =
                        (named_type(element_type, resolve), init.as_ref())
                    {
                        top = Some((*md, block_name, *size, init_expr));
                    }
                }
            }
        }
    }
    let (top_md, block_type, n_layers_i, block_init) = top.ok_or_else(|| {
        CepExtractError::MissingStructure(
            "no top-level model with a `blocks: [Block; N]` field".to_string(),
        )
    })?;
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
    let empty_consts: HashMap<String, i64> = HashMap::new();
    for (i, param) in block_md.params.iter().enumerate() {
        if let Some(arg) = block_args.get(i) {
            if let Ok(v) = resolve_binding(
                &arg.value,
                &BindingCtx {
                    params: HashMap::new(),
                    consts: empty_consts.clone(),
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
}
