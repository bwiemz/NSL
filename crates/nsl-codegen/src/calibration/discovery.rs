//! AWQ projection discovery — Task 3 of the calibration forward-pass plan.
//!
//! Walks a model's `forward` method body, collects every site where a model
//! layer field is piped or called (i.e. a linear projection), filters by a
//! `*`-per-segment glob derived from the quantisation block, and returns a
//! sorted `Vec<DiscoveredProjection>`.

use crate::calibration::observation::ProjectionRef;
use nsl_ast::decl::ModelMember;
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::Block;
use nsl_lexer::Interner;

// ── error type ────────────────────────────────────────────────────────────────

/// Error variants surfaced by [`discover_awq_projections`].
#[derive(Debug)]
pub enum DiscoveryError {
    /// The glob matched zero projections in the forward body.
    NoMatch { glob: String },
    /// A pipe target or call callee that looks like a layer reference is not
    /// a static model parameter (not in `model_field_types`).
    NonStaticWeight { path: String },
}

impl std::fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMatch { glob } => {
                write!(f, "AWQ glob '{glob}' matched zero projections")
            }
            Self::NonStaticWeight { path } => write!(
                f,
                "Projection '{path}' uses a non-parameter weight; AWQ requires static weights"
            ),
        }
    }
}

impl std::error::Error for DiscoveryError {}

// ── public result type ────────────────────────────────────────────────────────

/// A single discovered linear-projection site inside a model's forward pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredProjection {
    /// The qualified path of this projection, e.g. `"TinyMLP.up_proj"`.
    pub projection: ProjectionRef,
    /// Weight shape `[out_features, in_features]`.
    pub weight_shape: [u32; 2],
}

// ── glob matching ─────────────────────────────────────────────────────────────

/// Returns `true` when `path` matches `glob`.
///
/// Rules (spec §3):
/// * Segments are split on `'.'`.
/// * `*` matches exactly one non-empty segment (any content).
/// * Segment count must match exactly.
pub fn glob_matches(glob: &str, path: &str) -> bool {
    let gs: Vec<&str> = glob.split('.').collect();
    let ps: Vec<&str> = path.split('.').collect();
    gs.len() == ps.len()
        && gs
            .iter()
            .zip(ps.iter())
            .all(|(g, p)| *g == "*" || g == p)
}

// ── shape string parser ───────────────────────────────────────────────────────

/// Parse `"Tensor<[out, in], dtype>"` → `[out, in]` as u32.
/// Returns `[0, 0]` on any format deviation.
fn parse_tensor_shape_string(s: &str) -> [u32; 2] {
    // Expected format: "Tensor<[A, B], ...>"
    let inner = s.strip_prefix("Tensor<[").unwrap_or("");
    let end = match inner.find(']') {
        Some(i) => &inner[..i],
        None => return [0, 0],
    };
    let parts: Vec<&str> = end.split(',').collect();
    if parts.len() < 2 {
        return [0, 0];
    }
    let out_f = parts[0].trim().parse::<u32>().unwrap_or(0);
    let in_f = parts[1].trim().parse::<u32>().unwrap_or(0);
    [out_f, in_f]
}

// ── built-in filter ───────────────────────────────────────────────────────────

/// Returns `true` for well-known non-layer built-in names that commonly appear
/// as pipe targets but are not model field references.
fn is_builtin_fn(name: &str) -> bool {
    matches!(
        name,
        "relu"
            | "gelu"
            | "silu"
            | "sigmoid"
            | "tanh"
            | "softmax"
            | "log_softmax"
            | "dropout"
            | "layer_norm"
            | "rms_norm"
            | "flatten"
            | "reshape"
            | "transpose"
            | "squeeze"
            | "unsqueeze"
            | "mean"
            | "sum"
            | "max"
            | "min"
            | "abs"
            | "sqrt"
            | "log"
            | "exp"
            | "norm"
            | "clip"
            | "clamp"
    )
}

// ── forward body finder ───────────────────────────────────────────────────────

/// Find the `forward` method body inside a model definition.
fn find_forward_body<'a>(
    model_def: &'a nsl_ast::decl::ModelDef,
    interner: &Interner,
) -> Option<&'a Block> {
    for member in &model_def.members {
        if let ModelMember::Method(fn_def, _) = member {
            let name = interner.resolve(fn_def.name.0).unwrap_or("");
            if name == "forward" {
                return Some(&fn_def.body);
            }
        }
    }
    None
}

// ── pipe-target collector ─────────────────────────────────────────────────────

/// Collect the resolved names of every identifier that appears as the
/// right-hand side of a `|>` operator in `block`, recursively.
///
/// Returns a `Vec<String>` of field names (already resolved via the interner).
fn collect_pipe_rhs_names(block: &Block, interner: &Interner) -> Vec<String> {
    let mut out = Vec::new();
    for stmt in &block.stmts {
        collect_rhs_stmt(stmt, interner, &mut out);
    }
    out
}

fn collect_rhs_stmt(stmt: &nsl_ast::stmt::Stmt, interner: &Interner, out: &mut Vec<String>) {
    use nsl_ast::stmt::StmtKind;
    match &stmt.kind {
        StmtKind::Expr(e) => collect_rhs_expr(e, interner, out),
        StmtKind::VarDecl { value: Some(e), .. } => collect_rhs_expr(e, interner, out),
        StmtKind::Assign { value, .. } => collect_rhs_expr(value, interner, out),
        StmtKind::Return(Some(e)) => collect_rhs_expr(e, interner, out),
        _ => {}
    }
}

fn collect_rhs_expr(expr: &nsl_ast::expr::Expr, interner: &Interner, out: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::Pipe { left, right } => {
            // Left side is the data; recurse to find nested pipe RHS names.
            collect_rhs_expr(left, interner, out);
            // Right side is the layer reference.
            match &right.kind {
                ExprKind::Ident(sym) => {
                    if let Some(name) = interner.resolve(sym.0) {
                        out.push(name.to_string());
                    }
                }
                ExprKind::Call { callee, .. } => {
                    if let ExprKind::Ident(sym) = &callee.kind {
                        if let Some(name) = interner.resolve(sym.0) {
                            out.push(name.to_string());
                        }
                    }
                }
                _ => {
                    collect_rhs_expr(right, interner, out);
                }
            }
        }
        ExprKind::Paren(inner) => collect_rhs_expr(inner, interner, out),
        ExprKind::BlockExpr(b) => {
            for s in &b.stmts {
                collect_rhs_stmt(s, interner, out);
            }
        }
        _ => {}
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Walk `model_def`'s `forward` method body, collecting linear-projection sites,
/// and return a sorted `Vec<DiscoveredProjection>`.
///
/// # Parameters
///
/// * `model_def`          — model AST node.
/// * `quant_block`        — the quant AST node; used for the exclusion list.
/// * `interner`           — symbol ↔ string resolver.
/// * `model_field_types`  — `field_name → type_name` for this model
///                           (e.g. `"up_proj"` → `"Linear"`).
/// * `tensor_shapes`      — `field_name → "Tensor<[out, in], dtype>"` for
///                           fields that carry a static shape annotation.
///
/// # Contract
///
/// The glob applied for filtering is `"<ModelName>.*"` (match all layers in
/// the model).  The `QuantBlock`'s `exclude` list overrides individual paths.
///
/// # Errors
///
/// * [`DiscoveryError::NoMatch`]         — glob + exclusions left nothing.
/// * [`DiscoveryError::NonStaticWeight`] — a pipe-RHS identifier is not a
///                                          registered model field.
pub fn discover_awq_projections(
    model_def: &nsl_ast::decl::ModelDef,
    quant_block: &nsl_ast::block::QuantBlock,
    interner: &Interner,
    model_field_types: &std::collections::HashMap<String, String>,
    tensor_shapes: &std::collections::HashMap<String, String>,
) -> Result<Vec<DiscoveredProjection>, DiscoveryError> {
    // 1. Resolve the model name (first glob segment).
    let model_name = interner
        .resolve(model_def.name.0)
        .unwrap_or("<unknown>")
        .to_string();

    // 2. Synthesise the target glob: `"<ModelName>.*"`.
    //    The current QuantBlock AST carries no explicit `awq_target` field;
    //    we default to matching every direct field of the model.
    let synthetic_glob = format!("{}.*", model_name);

    // 3. Find the forward body.
    let forward_body = find_forward_body(model_def, interner);

    // 4. Collect resolved field names from the pipe chain.
    let raw_names: Vec<String> = match forward_body {
        Some(body) => collect_pipe_rhs_names(body, interner),
        None => Vec::new(),
    };

    // 5. Deduplicate while preserving first-occurrence order.
    let mut seen = std::collections::HashSet::new();
    let mut ordered_fields: Vec<String> = Vec::new();
    for name in raw_names {
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        ordered_fields.push(name);
    }

    // 6. Validate each field and build LinearSite records.
    let mut sites: Vec<(String, [u32; 2])> = Vec::new();
    for field_name in ordered_fields {
        if is_builtin_fn(&field_name) {
            continue; // skip activation functions
        }
        if !model_field_types.contains_key(&field_name) {
            let qualified = format!("{}.{}", model_name, field_name);
            return Err(DiscoveryError::NonStaticWeight { path: qualified });
        }
        let weight_shape = tensor_shapes
            .get(&field_name)
            .map(|s| parse_tensor_shape_string(s))
            .unwrap_or([0, 0]);
        sites.push((field_name, weight_shape));
    }

    // 7. Apply glob + exclusion list.
    let exclude_set: std::collections::HashSet<&str> =
        quant_block.exclude.iter().map(String::as_str).collect();

    let mut matched: Vec<DiscoveredProjection> = sites
        .into_iter()
        .filter_map(|(field_name, weight_shape)| {
            let qualified = format!("{}.{}", model_name, field_name);
            if !glob_matches(&synthetic_glob, &qualified) {
                return None;
            }
            if exclude_set.contains(qualified.as_str())
                || exclude_set.contains(field_name.as_str())
            {
                return None;
            }
            Some(DiscoveredProjection {
                projection: ProjectionRef::new(qualified),
                weight_shape,
            })
        })
        .collect();

    // 8. Hard-error when nothing survived.
    if matched.is_empty() {
        return Err(DiscoveryError::NoMatch {
            glob: synthetic_glob,
        });
    }

    // 9. Sort for determinism.
    matched.sort_by(|a, b| a.projection.0.cmp(&b.projection.0));

    Ok(matched)
}

// ── codegen-state entry point ─────────────────────────────────────────────────

/// Discover AWQ projections from already-resolved codegen metadata — no raw
/// AST required.
///
/// This is the entry point used by the `Compiler::discover_awq_projections`
/// stub replacement.  Unlike the AST-based function above, it takes:
///
/// * `model_name`        — the resolved model type name (e.g. `"TinyMLP"`).
/// * `forward_body`      — the `forward` method's `Block`, if available.
/// * `model_field_types` — `field_name → type_name` for the model.
/// * `tensor_shapes`     — `field_name → "Tensor<[out, in], dtype>"`.
/// * `exclude`           — qualified paths or bare field names to skip.
///
/// Semantics and return type are identical to [`discover_awq_projections`].
pub fn discover_awq_projections_from_state(
    model_name: &str,
    forward_body: Option<&Block>,
    model_field_types: &std::collections::HashMap<String, String>,
    tensor_shapes: &std::collections::HashMap<String, String>,
    exclude: &[String],
    interner: &Interner,
) -> Result<Vec<DiscoveredProjection>, DiscoveryError> {
    let synthetic_glob = format!("{}.*", model_name);

    // Collect resolved field names from the pipe chain.
    let raw_names: Vec<String> = match forward_body {
        Some(body) => collect_pipe_rhs_names(body, interner),
        None => Vec::new(),
    };

    // Deduplicate while preserving first-occurrence order.
    let mut seen = std::collections::HashSet::new();
    let mut ordered_fields: Vec<String> = Vec::new();
    for name in raw_names {
        if name.is_empty() || !seen.insert(name.clone()) {
            continue;
        }
        ordered_fields.push(name);
    }

    // Validate and build sites.
    let mut sites: Vec<(String, [u32; 2])> = Vec::new();
    for field_name in ordered_fields {
        if is_builtin_fn(&field_name) {
            continue;
        }
        if !model_field_types.contains_key(&field_name) {
            let qualified = format!("{}.{}", model_name, field_name);
            return Err(DiscoveryError::NonStaticWeight { path: qualified });
        }
        let weight_shape = tensor_shapes
            .get(&field_name)
            .map(|s| parse_tensor_shape_string(s))
            .unwrap_or([0, 0]);
        sites.push((field_name, weight_shape));
    }

    // Apply glob + exclusion list.
    let exclude_set: std::collections::HashSet<&str> =
        exclude.iter().map(String::as_str).collect();

    let mut matched: Vec<DiscoveredProjection> = sites
        .into_iter()
        .filter_map(|(field_name, weight_shape)| {
            let qualified = format!("{}.{}", model_name, field_name);
            if !glob_matches(&synthetic_glob, &qualified) {
                return None;
            }
            if exclude_set.contains(qualified.as_str())
                || exclude_set.contains(field_name.as_str())
            {
                return None;
            }
            Some(DiscoveredProjection {
                projection: ProjectionRef::new(qualified),
                weight_shape,
            })
        })
        .collect();

    if matched.is_empty() {
        return Err(DiscoveryError::NoMatch {
            glob: synthetic_glob,
        });
    }

    matched.sort_by(|a, b| a.projection.0.cmp(&b.projection.0));

    Ok(matched)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::block::{QuantBlock, QuantDtype, QuantKind};
    use nsl_ast::decl::{FnDef, ModelDef, ModelMember, Param};
    use nsl_ast::expr::{Expr, ExprKind};
    use nsl_ast::stmt::{Block, Stmt, StmtKind};
    use nsl_ast::{NodeId, Span};
    use nsl_lexer::Interner;

    // ── fixture helpers ───────────────────────────────────────────────────

    fn dummy_span() -> Span {
        Span::dummy()
    }

    fn make_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    fn make_stmt(kind: StmtKind) -> Stmt {
        Stmt {
            kind,
            span: dummy_span(),
            id: NodeId::next(),
        }
    }

    /// Build a `TinyMLP` fixture:
    /// * `up_proj: Linear`   (weight shape [64, 128])
    /// * `down_proj: Linear` (weight shape [128, 64])
    /// * `forward(x): x |> up_proj |> relu |> down_proj`
    ///
    /// Returns `(model_def, quant_block, interner, field_types, tensor_shapes)`.
    fn mlp_fixture_two_linears() -> (
        ModelDef,
        QuantBlock,
        Interner,
        std::collections::HashMap<String, String>,
        std::collections::HashMap<String, String>,
    ) {
        let mut interner = Interner::new();

        let sym_tiny_mlp = nsl_ast::Symbol(interner.get_or_intern("TinyMLP"));
        let sym_up_proj = nsl_ast::Symbol(interner.get_or_intern("up_proj"));
        let sym_down_proj = nsl_ast::Symbol(interner.get_or_intern("down_proj"));
        let sym_relu = nsl_ast::Symbol(interner.get_or_intern("relu"));
        let sym_x = nsl_ast::Symbol(interner.get_or_intern("x"));
        let sym_forward = nsl_ast::Symbol(interner.get_or_intern("forward"));
        let sym_linear = nsl_ast::Symbol(interner.get_or_intern("Linear"));
        let sym_quant_name = nsl_ast::Symbol(interner.get_or_intern("q0"));

        // forward body: x |> up_proj |> relu |> down_proj
        let pipe_chain = make_expr(ExprKind::Pipe {
            left: Box::new(make_expr(ExprKind::Pipe {
                left: Box::new(make_expr(ExprKind::Pipe {
                    left: Box::new(make_expr(ExprKind::Ident(sym_x))),
                    right: Box::new(make_expr(ExprKind::Ident(sym_up_proj))),
                })),
                right: Box::new(make_expr(ExprKind::Ident(sym_relu))),
            })),
            right: Box::new(make_expr(ExprKind::Ident(sym_down_proj))),
        });

        let forward_body = Block {
            stmts: vec![make_stmt(StmtKind::Expr(pipe_chain))],
            span: dummy_span(),
        };

        let forward_method = FnDef {
            name: sym_forward,
            type_params: vec![],
            effect_params: vec![],
            params: vec![Param {
                name: sym_x,
                type_ann: None,
                default: None,
                is_variadic: false,
                span: dummy_span(),
            }],
            return_type: None,
            return_effect: None,
            body: forward_body,
            is_async: false,
            span: dummy_span(),
        };

        let type_ann_linear = nsl_ast::types::TypeExpr {
            kind: nsl_ast::types::TypeExprKind::Named(sym_linear),
            span: dummy_span(),
            id: NodeId::next(),
        };

        let model_def = ModelDef {
            name: sym_tiny_mlp,
            type_params: vec![],
            params: vec![],
            members: vec![
                ModelMember::LayerDecl {
                    name: sym_up_proj,
                    type_ann: type_ann_linear.clone(),
                    init: None,
                    decorators: vec![],
                    span: dummy_span(),
                },
                ModelMember::LayerDecl {
                    name: sym_down_proj,
                    type_ann: type_ann_linear,
                    init: None,
                    decorators: vec![],
                    span: dummy_span(),
                },
                ModelMember::Method(forward_method, vec![]),
            ],
            span: dummy_span(),
        };

        let quant_block = QuantBlock {
            kind: QuantKind::Static,
            name: sym_quant_name,
            source: sym_tiny_mlp,
            default_dtype: Some(QuantDtype::Awq4),
            default_granularity: None,
            exclude: vec![],
            calibration: None,
            span: dummy_span(),
        };

        let mut field_types = std::collections::HashMap::new();
        field_types.insert("up_proj".to_string(), "Linear".to_string());
        field_types.insert("down_proj".to_string(), "Linear".to_string());

        let mut tensor_shapes = std::collections::HashMap::new();
        tensor_shapes.insert("up_proj".to_string(), "Tensor<[64, 128], f32>".to_string());
        tensor_shapes.insert("down_proj".to_string(), "Tensor<[128, 64], f32>".to_string());

        (model_def, quant_block, interner, field_types, tensor_shapes)
    }

    // ── test cases ─────────────────────────────────────────────────────────

    #[test]
    fn single_glob_resolves_to_both_projections_sorted() {
        let (model_def, quant_block, interner, field_types, tensor_shapes) =
            mlp_fixture_two_linears();
        let result = discover_awq_projections(
            &model_def,
            &quant_block,
            &interner,
            &field_types,
            &tensor_shapes,
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        // Alphabetical: down_proj < up_proj
        assert_eq!(result[0].projection.0, "TinyMLP.down_proj");
        assert_eq!(result[1].projection.0, "TinyMLP.up_proj");
        assert_eq!(result[0].weight_shape, [128, 64]);
        assert_eq!(result[1].weight_shape, [64, 128]);
    }

    #[test]
    fn excluded_projection_is_omitted() {
        let (model_def, mut quant_block, interner, field_types, tensor_shapes) =
            mlp_fixture_two_linears();
        quant_block.exclude.push("TinyMLP.up_proj".to_string());
        let result = discover_awq_projections(
            &model_def,
            &quant_block,
            &interner,
            &field_types,
            &tensor_shapes,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].projection.0, "TinyMLP.down_proj");
    }

    #[test]
    fn all_excluded_gives_no_match_error() {
        let (model_def, mut quant_block, interner, field_types, tensor_shapes) =
            mlp_fixture_two_linears();
        quant_block.exclude.push("TinyMLP.up_proj".to_string());
        quant_block.exclude.push("TinyMLP.down_proj".to_string());
        let err = discover_awq_projections(
            &model_def,
            &quant_block,
            &interner,
            &field_types,
            &tensor_shapes,
        )
        .unwrap_err();
        assert!(matches!(err, DiscoveryError::NoMatch { .. }));
    }

    #[test]
    fn non_static_weight_errors_for_unknown_field() {
        let (model_def, quant_block, interner, mut field_types, tensor_shapes) =
            mlp_fixture_two_linears();
        // Remove down_proj from the field map; it still appears in the pipe chain.
        field_types.remove("down_proj");
        let err = discover_awq_projections(
            &model_def,
            &quant_block,
            &interner,
            &field_types,
            &tensor_shapes,
        )
        .unwrap_err();
        assert!(
            matches!(err, DiscoveryError::NonStaticWeight { ref path } if path.contains("down_proj")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn model_with_no_forward_gives_no_match() {
        let (mut model_def, quant_block, interner, field_types, tensor_shapes) =
            mlp_fixture_two_linears();
        // Remove the Method member.
        model_def
            .members
            .retain(|m| !matches!(m, ModelMember::Method(..)));
        let err = discover_awq_projections(
            &model_def,
            &quant_block,
            &interner,
            &field_types,
            &tensor_shapes,
        )
        .unwrap_err();
        assert!(matches!(err, DiscoveryError::NoMatch { .. }));
    }

    #[test]
    fn glob_matches_semantics() {
        assert!(glob_matches("TinyMLP.*", "TinyMLP.up_proj"));
        assert!(glob_matches("TinyMLP.*", "TinyMLP.down_proj"));
        assert!(!glob_matches("TinyMLP.*", "Other.up_proj"));
        // extra segment → mismatch
        assert!(!glob_matches("TinyMLP.*", "TinyMLP.sub.proj"));
        assert!(glob_matches("*.*", "TinyMLP.up_proj"));
        assert!(!glob_matches("*", "TinyMLP.up_proj"));
    }

    #[test]
    fn parse_tensor_shape_string_cases() {
        assert_eq!(parse_tensor_shape_string("Tensor<[64, 128], f32>"), [64, 128]);
        assert_eq!(parse_tensor_shape_string("Tensor<[128, 64], f32>"), [128, 64]);
        assert_eq!(parse_tensor_shape_string(""), [0, 0]);
        assert_eq!(parse_tensor_shape_string("Tensor<[99], f32>"), [0, 0]);
    }
}
