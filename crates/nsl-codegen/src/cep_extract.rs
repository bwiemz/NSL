//! CEP frontend — AST recognizer: canonical NSL transformer block -> ModelSpec / SearchAxes.

use std::collections::HashMap;

use nsl_ast::decl::{Decorator, ModelDef, ModelMember};
use nsl_ast::expr::{Arg, Expr, ExprKind};
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::StmtKind;
use nsl_ast::types::TypeExprKind;
use nsl_ast::{Module, Symbol};

use crate::cep_oracle::{Activation, ModelSpec, NormType};
use crate::cep_rewrite::SearchAxes;
use crate::weight_aware::WeightMap;

/// Recognizer refusals — each names what was expected vs. what was found.
#[derive(Debug, Clone, PartialEq)]
pub enum CepExtractError {
    MissingStructure(String),
    UnrecognizedAttention { expected: String, found: String },
    UnresolvableBinding { dim: String, detail: String },
    UnknownFfnActivation { ffn_type: String },
    UnknownSearchAxis { axis: String },
    UnrecognizedActivation { name: String },
    UnrecognizedNorm { name: String },
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
                    "CEP: unknown @search axis '{axis}' (known: d_model, n_layers, n_heads, n_kv_heads, d_ff, activation, norm)"
                )
            }
            CepExtractError::UnrecognizedActivation { name } => {
                write!(
                    f,
                    "CEP: unrecognized activation '{name}' (known: relu, gelu, silu, swiglu)"
                )
            }
            CepExtractError::UnrecognizedNorm { name } => {
                write!(
                    f,
                    "CEP: unrecognized norm '{name}' (known: layernorm, layer_norm, ln, rmsnorm, rms_norm, rms)"
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
    if let StmtKind::VarDecl { is_const, pattern, value, .. } = kind {
        if !is_const {
            return None;
        }
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
    let block_arg_ctx = BindingCtx { params: HashMap::new(), consts: consts.clone() };
    let mut params = HashMap::new();
    for (i, param) in block_md.params.iter().enumerate() {
        if let Some(arg) = block_args.get(i) {
            if let Ok(v) = resolve_binding(
                &arg.value,
                &block_arg_ctx,
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

/// Collect every `@search(axis, [values])` decorator whose values are integer
/// literals.
fn collect_search_axes(module: &Module, resolve: Resolve) -> Vec<(String, Vec<u32>)> {
    let mut out = Vec::new();
    for stmt in &module.stmts {
        let StmtKind::Decorated { decorators, .. } = &stmt.kind else { continue };
        for deco in decorators {
            if deco.name.len() != 1 || resolve(deco.name[0]) != "search" {
                continue;
            }
            let Some(args) = &deco.args else { continue };
            let (Some(a0), Some(a1)) = (args.first(), args.get(1)) else { continue };
            let ExprKind::Ident(axis_sym) = &a0.value.kind else { continue };
            let ExprKind::ListLiteral(items) = &a1.value.kind else { continue };
            let mut values = Vec::new();
            for it in items {
                if let ExprKind::IntLiteral(n) = &it.kind {
                    values.push(*n as u32);
                }
            }
            if !values.is_empty() {
                out.push((resolve(*axis_sym), values));
            }
        }
    }
    out
}

/// Collect every `@search(axis, [values])` decorator whose values are string
/// literals or identifiers (used for categorical axes like `activation` / `norm`).
fn collect_search_string_axes(module: &Module, resolve: Resolve) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for stmt in &module.stmts {
        let StmtKind::Decorated { decorators, .. } = &stmt.kind else { continue };
        for deco in decorators {
            if deco.name.len() != 1 || resolve(deco.name[0]) != "search" {
                continue;
            }
            let Some(args) = &deco.args else { continue };
            let (Some(a0), Some(a1)) = (args.first(), args.get(1)) else { continue };
            let ExprKind::Ident(axis_sym) = &a0.value.kind else { continue };
            let ExprKind::ListLiteral(items) = &a1.value.kind else { continue };
            let mut values = Vec::new();
            for it in items {
                match &it.kind {
                    ExprKind::StringLiteral(s) => values.push(s.clone()),
                    ExprKind::Ident(sym) => values.push(resolve(*sym)),
                    _ => {}
                }
            }
            if !values.is_empty() {
                out.push((resolve(*axis_sym), values));
            }
        }
    }
    out
}

/// Parse an activation name string (from a `@search(activation, [...])` axis).
fn parse_activation(name: &str) -> Result<Activation, CepExtractError> {
    match name {
        "relu" => Ok(Activation::Relu),
        "gelu" => Ok(Activation::Gelu),
        "silu" => Ok(Activation::SiLU),
        "swiglu" => Ok(Activation::SwiGlu),
        other => Err(CepExtractError::UnrecognizedActivation { name: other.to_string() }),
    }
}

/// Parse a norm-type name string (from a `@search(norm, [...])` axis).
fn parse_norm(name: &str) -> Result<NormType, CepExtractError> {
    match name {
        "layernorm" | "layer_norm" | "ln" => Ok(NormType::LayerNorm),
        "rmsnorm" | "rms_norm" | "rms" => Ok(NormType::RmsNorm),
        other => Err(CepExtractError::UnrecognizedNorm { name: other.to_string() }),
    }
}

pub fn extract_search_axes(module: &Module, resolve: Resolve) -> Result<SearchAxes, CepExtractError> {
    let base = extract_model_spec(module, resolve)?;
    let mut axes = SearchAxes {
        d_model: vec![base.d_model],
        n_layers: vec![base.n_layers],
        n_heads: vec![base.n_heads[0]],
        n_kv_heads: vec![base.n_kv_heads[0]],
        d_ff: vec![base.d_ff[0]],
        activation: vec![base.activation],
        norm: vec![base.norm],
        vocab: base.vocab,
        head_dim: base.head_dim[0],
        max_seq: base.max_seq,
        batch: base.batch,
        dtype_bytes: base.dtype_bytes,
    };
    // Integer-valued axes (d_model, n_layers, n_heads, n_kv_heads, d_ff).
    for (axis, values) in collect_search_axes(module, resolve) {
        match axis.as_str() {
            "d_model" => axes.d_model = values,
            "n_layers" => axes.n_layers = values,
            "n_heads" => axes.n_heads = values,
            "n_kv_heads" => axes.n_kv_heads = values,
            "d_ff" => axes.d_ff = values,
            // Categorical axes are handled below via collect_search_string_axes;
            // ignore them here so we don't double-error.
            "activation" | "norm" | "norm_type" => {}
            other => return Err(CepExtractError::UnknownSearchAxis { axis: other.to_string() }),
        }
    }
    // String-valued categorical axes (activation, norm / norm_type).
    for (axis, values) in collect_search_string_axes(module, resolve) {
        match axis.as_str() {
            "activation" => {
                let parsed: Result<Vec<Activation>, _> =
                    values.iter().map(|s| parse_activation(s)).collect();
                axes.activation = parsed?;
            }
            "norm" | "norm_type" => {
                let parsed: Result<Vec<NormType>, _> =
                    values.iter().map(|s| parse_norm(s)).collect();
                axes.norm = parsed?;
            }
            // Numeric axes may show up here if someone writes string-like values;
            // skip them — they were already handled (or will error) above.
            _ => {}
        }
    }
    Ok(axes)
}

/// Mutual-consistency disagreement between AST-extracted dims and weight shapes.
/// Names the disagreement, not the culprit.
#[derive(Debug, Clone, PartialEq)]
pub struct CepCrossCheckError {
    pub tensor: String,
    pub expected: String,
    pub found: String,
}

impl std::fmt::Display for CepCrossCheckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CEP: model source and weights disagree\n  tensor:   {}\n  expected: {}\n  found:    {}\n  (cannot tell which side is wrong — check the model source AND the weights file)",
            self.tensor, self.expected, self.found
        )
    }
}

/// True if `shape` matches `[a, b]` in either orientation.
fn dims_match(shape: &[usize], a: u32, b: u32) -> bool {
    shape.len() == 2
        && ((shape[0] as u32 == a && shape[1] as u32 == b)
            || (shape[0] as u32 == b && shape[1] as u32 == a))
}

/// Cross-check extracted dims against actual tensor shapes for the attention +
/// FFN tensors only. The embedding is excluded (vocab is fixed/legitimately
/// padded — DR-8). Returns the first disagreement.
pub fn cross_check_dims(
    spec: &ModelSpec,
    wm: &WeightMap,
    _resolve: Resolve,
) -> Result<(), CepCrossCheckError> {
    let mut checked = 0usize;
    for l in 0..spec.n_layers as usize {
        let q_proj = spec.n_heads[l] * spec.head_dim[l];
        let kv_proj = spec.n_kv_heads[l] * spec.head_dim[l];
        let d_ff = spec.d_ff[l];
        let dm = spec.d_model;

        let checks: [(String, u32, String); 5] = [
            (format!("blocks.{l}.attn.wq"), q_proj, format!("n_heads·head_dim = {q_proj}")),
            (format!("blocks.{l}.attn.wk"), kv_proj, format!("n_kv_heads·head_dim = {kv_proj}")),
            (format!("blocks.{l}.attn.wv"), kv_proj, format!("n_kv_heads·head_dim = {kv_proj}")),
            (format!("blocks.{l}.ffn.w_gate"), d_ff, format!("d_ff = {d_ff}")),
            (format!("blocks.{l}.ffn.w_up"), d_ff, format!("d_ff = {d_ff}")),
        ];
        for (name, other, label) in checks {
            let Some(entry) = wm.get(&name) else { continue };
            checked += 1;
            if !dims_match(&entry.shape, dm, other) {
                return Err(CepCrossCheckError {
                    tensor: name,
                    expected: format!(
                        "{label} (with d_model = {dm}) from GroupedQueryAttention/SwiGLUFFN args"
                    ),
                    found: format!("weight shape {:?}", entry.shape),
                });
            }
        }
    }
    if checked == 0 {
        return Err(CepCrossCheckError {
            tensor: "blocks.0.attn.wq".to_string(),
            expected: "tensors named blocks.N.attn.* / blocks.N.ffn.*".to_string(),
            found: "no tensors matching the model's attention/FFN naming convention".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_oracle::{Activation, NormType};
    use crate::weight_aware::WeightMap;

    // Write a minimal safetensors file with the given named f32 tensors (zero data).
    fn write_safetensors(path: &std::path::Path, tensors: &[(String, Vec<usize>)]) {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        use std::collections::HashMap as StdHashMap;
        let owned: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
            .iter()
            .map(|(n, s)| {
                let len: usize = s.iter().product();
                (n.clone(), s.clone(), vec![0u8; len * 4])
            })
            .collect();
        let views: StdHashMap<String, TensorView<'_>> = owned
            .iter()
            .map(|(n, s, d)| {
                (n.clone(), TensorView::new(Dtype::F32, s.clone(), d).unwrap())
            })
            .collect();
        let bytes = safetensors::tensor::serialize(&views, &None).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    fn canonical_spec() -> ModelSpec {
        ModelSpec {
            d_model: 384,
            n_layers: 6,
            n_heads: vec![6; 6],
            n_kv_heads: vec![3; 6],
            head_dim: vec![64; 6],
            d_ff: vec![1024; 6],
            vocab: 4096,
            max_seq: 2048,
            batch: 1,
            activation: Activation::SwiGlu,
            norm: NormType::RmsNorm,
            dtype_bytes: 4,
        }
    }

    fn matching_tensors() -> Vec<(String, Vec<usize>)> {
        let mut t = Vec::new();
        for l in 0..6 {
            t.push((format!("blocks.{l}.attn.wq"), vec![384, 384])); // n_heads*head_dim = 384
            t.push((format!("blocks.{l}.attn.wk"), vec![384, 192])); // n_kv_heads*head_dim = 192
            t.push((format!("blocks.{l}.attn.wv"), vec![384, 192]));
            t.push((format!("blocks.{l}.ffn.w_gate"), vec![384, 1024]));
            t.push((format!("blocks.{l}.ffn.w_up"), vec![384, 1024]));
        }
        t.push(("embed".to_string(), vec![4224, 384])); // padded vocab — must NOT be flagged
        t
    }

    #[test]
    fn cross_check_matching_weights_pass() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.safetensors");
        write_safetensors(&path, &matching_tensors());
        let wm = WeightMap::load(&path).unwrap();
        let resolve = |_s: nsl_ast::Symbol| String::new();
        assert!(cross_check_dims(&canonical_spec(), &wm, &resolve).is_ok());
    }

    #[test]
    fn cross_check_wrong_wq_refuses() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w.safetensors");
        let mut t = matching_tensors();
        t[0].1 = vec![384, 768]; // blocks.0.attn.wq disagrees (768 != 384)
        write_safetensors(&path, &t);
        let wm = WeightMap::load(&path).unwrap();
        let resolve = |_s: nsl_ast::Symbol| String::new();
        let err = cross_check_dims(&canonical_spec(), &wm, &resolve).unwrap_err();
        assert_eq!(err.tensor, "blocks.0.attn.wq");
        assert!(err.to_string().contains("disagree"));
    }

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

    const SEARCHABLE: &str = r#"
@cep_search(target = h100, objective = param_efficiency)
@search(d_model, [256, 384, 512])
@search(n_heads, [4, 6, 8])
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model SearchNet:
    embed: Tensor = randn([4096, 384]) * full([1], 0.02)
    blocks: [TransformerBlock; 6] = TransformerBlock(384, 6, 3, 1024, 0.1)
"#;

    #[test]
    fn extracts_search_axes() {
        let (module, interner) = parse(SEARCHABLE);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let axes = extract_search_axes(&module, &resolve).expect("axes");
        assert_eq!(axes.d_model, vec![256, 384, 512]);
        assert_eq!(axes.n_heads, vec![4, 6, 8]);
        // Un-searched axes default to the base spec's single value.
        assert_eq!(axes.n_kv_heads, vec![3]);
        assert_eq!(axes.d_ff, vec![1024]);
        assert_eq!(axes.n_layers, vec![6]);
        assert_eq!(axes.vocab, 4096);
    }

    #[test]
    fn refuses_unknown_search_axis() {
        let src = r#"
@search(bogus_axis, [1, 2, 3])
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model AxNet:
    embed: Tensor = randn([1000, 256])
    blocks: [TransformerBlock; 2] = TransformerBlock(256, 8, 4, 512, 0.1)
"#;
        let (module, interner) = parse(src);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let err = extract_search_axes(&module, &resolve).unwrap_err();
        assert!(matches!(err, CepExtractError::UnknownSearchAxis { .. }), "got: {err}");
    }

    // G17 — @search(activation, [...]) and @search(norm, [...]) axes.
    const ACTIVATION_NORM_SEARCHABLE: &str = r#"
@search(activation, ["silu", "swiglu"])
@search(norm, ["rmsnorm"])
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model ActNormNet:
    embed: Tensor = randn([4096, 384]) * full([1], 0.02)
    blocks: [TransformerBlock; 6] = TransformerBlock(384, 6, 3, 1024, 0.1)
"#;

    #[test]
    fn extracts_activation_and_norm_axes() {
        let (module, interner) = parse(ACTIVATION_NORM_SEARCHABLE);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let axes = extract_search_axes(&module, &resolve).expect("axes");
        // Both activation variants must be present.
        assert!(
            axes.activation.contains(&Activation::SiLU),
            "SiLU should be in activation axes: {:?}", axes.activation
        );
        assert!(
            axes.activation.contains(&Activation::SwiGlu),
            "SwiGlu should be in activation axes: {:?}", axes.activation
        );
        assert_eq!(axes.activation.len(), 2);
        // Norm axis.
        assert_eq!(axes.norm, vec![NormType::RmsNorm]);
    }

    #[test]
    fn unrecognized_activation_returns_error() {
        let src = r#"
@search(activation, ["bogus_act"])
model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int, dropout_p: float):
    wq: Tensor = randn([d_model, d_model])
model SwiGLUFFN(d_model: int, d_ff: int, dropout_p: float):
    w_gate: Tensor = randn([d_model, d_ff])
model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout_p: float):
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout_p)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff, dropout_p)
    norm: RMSNorm = RMSNorm(d_model)
model BadActNet:
    embed: Tensor = randn([4096, 384]) * full([1], 0.02)
    blocks: [TransformerBlock; 6] = TransformerBlock(384, 6, 3, 1024, 0.1)
"#;
        let (module, interner) = parse(src);
        let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let err = extract_search_axes(&module, &resolve).unwrap_err();
        assert!(
            matches!(err, CepExtractError::UnrecognizedActivation { .. }),
            "got: {err}"
        );
    }
}
