//! AWQ projection discovery — Task 3 of the calibration forward-pass plan.
//!
//! Walks a model's `forward` method body, collects every site where a model
//! layer field is piped or called (i.e. a linear projection), filters by a
//! `*`-per-segment glob derived from the quantisation block, and returns a
//! `Vec<DiscoveredProjection>` in **forward-pipe order** (first occurrence
//! per field name). Pipe order is canonical: the AST pre-scan and the
//! in-compile path both produce it, and `check_discovery_agreement` requires
//! the two to match.

use crate::calibration::observation::ProjectionRef;
use nsl_ast::decl::{Decorator, ModelDef, ModelMember};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::{Block, StmtKind};
use nsl_ast::types::{TypeExpr, TypeExprKind};
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
    /// The AST pre-scan and in-compiler discovery paths disagreed.
    Divergence {
        pre_scan_names: Vec<String>,
        in_compile_names: Vec<String>,
    },
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
            Self::Divergence {
                pre_scan_names,
                in_compile_names,
            } => write!(
                f,
                "calibration: discovery divergence.\n pre-scan:    {}\n in-compile:  {}",
                pre_scan_names.join(", "),
                in_compile_names.join(", ")
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

/// A single WGGO Phase 2 target — one per instance of a
/// `@wggo_target`-decorated model.
///
/// Produced by [`pre_scan_wggo_targets_from_ast`] (Task 6 of the WGGO
/// Phase 2 backward-pass calibration plan). Each record carries the
/// qualified instance path, the resolved `head_dim`, and the four
/// projection refs + shapes drawn from the model's
/// `@wggo_target(w_q=self.<…>, w_k=self.<…>, w_v=self.<…>,
/// w_o=self.<…>, head_dim=self.<…>)` decorator on `forward`.
#[derive(Debug, Clone, PartialEq)]
pub struct WggoGradTarget {
    /// Qualified path of the instance, e.g. `"gpt.blocks.0.attn"`.
    /// For Phase 2 Task 6 this is the variable name from the `let`
    /// binding at the instantiation site (e.g.
    /// `let attn = Attention(...)` → `"attn"`).
    pub layer_key: String,
    /// Model class name, e.g. `"Attention"`.
    pub class_name: String,
    /// Resolved at pre-scan from the model's `head_dim` field
    /// initializer via the AST evaluator (e.g.
    /// `head_dim: int = dim // num_heads` with `dim=4096, num_heads=32`
    /// → `128`).
    pub head_dim: u32,
    pub w_q: ProjectionRef,
    pub w_k: ProjectionRef,
    pub w_v: ProjectionRef,
    pub w_o: ProjectionRef,
    /// Resolved from `W_*`'s tensor field initializer via the AST
    /// evaluator. `[out_features, in_features]`.
    pub w_q_shape: [u32; 2],
    pub w_k_shape: [u32; 2],
    pub w_v_shape: [u32; 2],
    pub w_o_shape: [u32; 2],
}

fn dedup_preserving_first_seen(projections: &mut Vec<DiscoveredProjection>) {
    let mut seen = std::collections::HashSet::new();
    projections.retain(|projection| seen.insert(projection.projection.0.clone()));
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

fn parse_tensor_shape_from_type_expr(ty: &TypeExpr) -> Option<[u32; 2]> {
    let TypeExprKind::Tensor { shape, .. } = &ty.kind else {
        return None;
    };
    if shape.len() < 2 {
        return None;
    }
    let out_features = match shape.first() {
        Some(nsl_ast::types::DimExpr::Concrete(n)) if *n >= 0 => *n as u32,
        _ => return None,
    };
    let in_features = match shape.get(1) {
        Some(nsl_ast::types::DimExpr::Concrete(n)) if *n >= 0 => *n as u32,
        _ => return None,
    };
    Some([out_features, in_features])
}

fn extract_shape_from_tensor_init(expr: &nsl_ast::expr::Expr, interner: &Interner) -> Option<[u32; 2]> {
    let ExprKind::Call { callee, args } = &expr.kind else {
        return None;
    };
    let ExprKind::Ident(sym) = &callee.kind else {
        return None;
    };
    let fname = interner.resolve(sym.0).unwrap_or("");
    if !matches!(fname, "zeros" | "ones" | "randn" | "rand" | "full" | "arange") {
        return None;
    }
    let first = args.first()?;
    let ExprKind::ListLiteral(items) = &first.value.kind else {
        return None;
    };
    if items.len() < 2 {
        return None;
    }
    let out_features = match &items[0].kind {
        ExprKind::IntLiteral(n) if *n >= 0 => *n as u32,
        _ => return None,
    };
    let in_features = match &items[1].kind {
        ExprKind::IntLiteral(n) if *n >= 0 => *n as u32,
        _ => return None,
    };
    Some([out_features, in_features])
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

fn model_has_awq_quantize_decorator(decorators: &[Decorator], interner: &Interner) -> bool {
    decorators.iter().any(|decorator| {
        if decorator.name.len() != 1 || interner.resolve(decorator.name[0].0).unwrap_or("") != "quantize" {
            return false;
        }
        let mut dtype = "awq4";
        if let Some(args) = &decorator.args {
            for arg in args {
                let Some(name_sym) = arg.name else {
                    continue;
                };
                if interner.resolve(name_sym.0).unwrap_or("") != "dtype" {
                    continue;
                }
                if let ExprKind::StringLiteral(s) = &arg.value.kind {
                    dtype = s.as_str();
                }
            }
        }
        dtype == "awq4"
    })
}

fn collect_model_layer_metadata(
    model_def: &ModelDef,
    interner: &Interner,
) -> (
    std::collections::HashMap<String, String>,
    std::collections::HashMap<String, String>,
) {
    let mut field_types = std::collections::HashMap::new();
    let mut tensor_shapes = std::collections::HashMap::new();

    for member in &model_def.members {
        let ModelMember::LayerDecl {
            name,
            type_ann,
            init,
            ..
        } = member else {
            continue;
        };

        let field_name = interner.resolve(name.0).unwrap_or("").to_string();
        if field_name.is_empty() {
            continue;
        }

        let type_name = match &type_ann.kind {
            TypeExprKind::Named(sym) => interner.resolve(sym.0).unwrap_or("").to_string(),
            TypeExprKind::Tensor { .. } => "Tensor".to_string(),
            _ => String::new(),
        };
        if !type_name.is_empty() {
            field_types.insert(field_name.clone(), type_name);
        }

        let shape = parse_tensor_shape_from_type_expr(type_ann)
            .or_else(|| init.as_ref().and_then(|expr| extract_shape_from_tensor_init(expr, interner)));
        if let Some([out_features, in_features]) = shape {
            tensor_shapes.insert(
                field_name,
                format!("Tensor<[{}, {}], f32>", out_features, in_features),
            );
        }
    }

    (field_types, tensor_shapes)
}

/// Discover AWQ projections directly from the parsed AST before any compiler
/// state exists.
pub fn pre_scan_awq_projections_from_ast(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> Vec<DiscoveredProjection> {
    let mut all_projections = Vec::new();

    for stmt in &ast.stmts {
        let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind else {
            continue;
        };
        let StmtKind::ModelDef(model_def) = &inner.kind else {
            continue;
        };
        if !model_has_awq_quantize_decorator(decorators, interner) {
            continue;
        }

        let model_name = interner.resolve(model_def.name.0).unwrap_or("").to_string();
        if model_name.is_empty() {
            continue;
        }

        let forward_body = find_forward_body(model_def, interner);
        let (field_types, tensor_shapes) = collect_model_layer_metadata(model_def, interner);

        if let Ok(mut discovered) = discover_awq_projections_from_state(
            &model_name,
            forward_body,
            &field_types,
            &tensor_shapes,
            &[],
            interner,
        ) {
            all_projections.append(&mut discovered);
        }
    }

    dedup_preserving_first_seen(&mut all_projections);
    all_projections
}

pub fn first_awq_quantized_model_name(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> Option<String> {
    ast.stmts.iter().find_map(|stmt| {
        let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind else {
            return None;
        };
        let StmtKind::ModelDef(model_def) = &inner.kind else {
            return None;
        };
        if !model_has_awq_quantize_decorator(decorators, interner) {
            return None;
        }
        let model_name = interner.resolve(model_def.name.0).unwrap_or("").to_string();
        if model_name.is_empty() {
            None
        } else {
            Some(model_name)
        }
    })
}

pub fn ast_has_awq_quantize_decorator(ast: &nsl_ast::Module, interner: &Interner) -> bool {
    first_awq_quantized_model_name(ast, interner).is_some()
}

// ── WGGO Phase 2 pre-scan ─────────────────────────────────────────────────────

/// Returns `true` when `model_def` carries an `@wggo_target` decorator on
/// its `forward` method.
fn model_has_wggo_target_decorator(model_def: &ModelDef, interner: &Interner) -> bool {
    for member in &model_def.members {
        let ModelMember::Method(fn_def, decorators) = member else {
            continue;
        };
        if interner.resolve(fn_def.name.0).unwrap_or("") != "forward" {
            continue;
        }
        for deco in decorators {
            if deco.name.len() == 1
                && interner.resolve(deco.name[0].0).unwrap_or("") == "wggo_target"
            {
                return true;
            }
        }
    }
    false
}

/// Holds the field names referenced by a `@wggo_target` decorator (each
/// arg's value is a `self.<field>` reference; we record only the field
/// name string).
struct WggoTargetFieldNames {
    w_q: String,
    w_k: String,
    w_v: String,
    w_o: String,
    head_dim: String,
}

/// Extract the five `self.<field>` references from a model's
/// `@wggo_target` decorator on `forward`. Returns `None` if any of the
/// five required fields is missing or shaped unexpectedly. Phase 1
/// (Tasks 2-3) is the canonical place that *errors* on these problems —
/// pre-scan is best-effort and silently skips.
fn extract_wggo_target_field_names(
    model_def: &ModelDef,
    interner: &Interner,
) -> Option<WggoTargetFieldNames> {
    for member in &model_def.members {
        let ModelMember::Method(fn_def, decorators) = member else {
            continue;
        };
        if interner.resolve(fn_def.name.0).unwrap_or("") != "forward" {
            continue;
        }
        for deco in decorators {
            if deco.name.len() != 1
                || interner.resolve(deco.name[0].0).unwrap_or("") != "wggo_target"
            {
                continue;
            }
            let args = deco.args.as_ref()?;
            let mut w_q = None;
            let mut w_k = None;
            let mut w_v = None;
            let mut w_o = None;
            let mut head_dim = None;
            for arg in args {
                let Some(arg_name_sym) = arg.name else {
                    continue;
                };
                let arg_name = interner.resolve(arg_name_sym.0).unwrap_or("");
                let ExprKind::MemberAccess { object, member } = &arg.value.kind else {
                    continue;
                };
                if !matches!(object.kind, ExprKind::SelfRef) {
                    continue;
                }
                let field = interner.resolve(member.0).unwrap_or("").to_string();
                if field.is_empty() {
                    continue;
                }
                match arg_name {
                    "w_q" => w_q = Some(field),
                    "w_k" => w_k = Some(field),
                    "w_v" => w_v = Some(field),
                    "w_o" => w_o = Some(field),
                    "head_dim" => head_dim = Some(field),
                    _ => {}
                }
            }
            return Some(WggoTargetFieldNames {
                w_q: w_q?,
                w_k: w_k?,
                w_v: w_v?,
                w_o: w_o?,
                head_dim: head_dim?,
            });
        }
    }
    None
}

/// Build `field_name → init_expr` for every `LayerDecl` member that has
/// an initializer. Used by the WGGO pre-scan to look up `W_*`/`head_dim`
/// initializers by the field names extracted from `@wggo_target`.
fn collect_layer_init_exprs<'a>(
    model_def: &'a ModelDef,
    interner: &Interner,
) -> std::collections::HashMap<String, &'a nsl_ast::expr::Expr> {
    let mut out = std::collections::HashMap::new();
    for member in &model_def.members {
        let ModelMember::LayerDecl {
            name, init: Some(init), ..
        } = member
        else {
            continue;
        };
        let fname = interner.resolve(name.0).unwrap_or("");
        if fname.is_empty() {
            continue;
        }
        out.insert(fname.to_string(), init);
    }
    out
}

/// Try to build a `WggoGradTarget` from a `let <var_name> = <init_expr>`
/// statement. Returns `None` (silent skip) on any of:
///
/// * `init_expr` isn't a direct call `ClassName(args...)`.
/// * `ClassName` isn't in `decorated`.
/// * Constructor-arg evaluation (`bind_constructor_args`) fails.
/// * The decorator's required field references are incomplete.
/// * Any field's initializer fails to evaluate to the expected shape
///   (`Int` for `head_dim`; 2-element `IntList` for each `W_*`).
///
/// This best-effort behaviour mirrors the AWQ pre-scan: errors surface
/// later in semantic check / Task 11 refusal, not here.
fn try_build_wggo_target(
    var_name: &str,
    init_expr: &nsl_ast::expr::Expr,
    decorated: &std::collections::HashMap<nsl_ast::Symbol, &ModelDef>,
    interner: &Interner,
) -> Option<WggoGradTarget> {
    use crate::calibration::ast_evaluator::{bind_constructor_args, evaluate_expr, EvalValue};

    let ExprKind::Call { callee, args } = &init_expr.kind else {
        return None;
    };
    let ExprKind::Ident(class_sym) = &callee.kind else {
        return None;
    };
    let model_def = decorated.get(class_sym)?;

    // Bind constructor args (best-effort).
    let scope = bind_constructor_args(&model_def.params, args, interner).ok()?;

    let fields = extract_wggo_target_field_names(model_def, interner)?;
    let inits = collect_layer_init_exprs(model_def, interner);

    let head_dim_init = inits.get(fields.head_dim.as_str())?;
    let head_dim_val = match evaluate_expr(head_dim_init, &scope, interner).ok()? {
        EvalValue::Int(n) if n >= 0 && n <= u32::MAX as i64 => n as u32,
        _ => return None,
    };

    let resolve_shape = |field: &str| -> Option<[u32; 2]> {
        let init = inits.get(field)?;
        let EvalValue::IntList(dims) = evaluate_expr(init, &scope, interner).ok()? else {
            return None;
        };
        if dims.len() != 2 {
            return None;
        }
        if dims[0] < 0 || dims[1] < 0 {
            return None;
        }
        if dims[0] > u32::MAX as i64 || dims[1] > u32::MAX as i64 {
            return None;
        }
        Some([dims[0] as u32, dims[1] as u32])
    };

    let w_q_shape = resolve_shape(&fields.w_q)?;
    let w_k_shape = resolve_shape(&fields.w_k)?;
    let w_v_shape = resolve_shape(&fields.w_v)?;
    let w_o_shape = resolve_shape(&fields.w_o)?;

    let class_name = interner.resolve(model_def.name.0).unwrap_or("").to_string();

    Some(WggoGradTarget {
        layer_key: var_name.to_string(),
        class_name,
        head_dim: head_dim_val,
        w_q: ProjectionRef(format!("{var_name}.{}", fields.w_q)),
        w_k: ProjectionRef(format!("{var_name}.{}", fields.w_k)),
        w_v: ProjectionRef(format!("{var_name}.{}", fields.w_v)),
        w_o: ProjectionRef(format!("{var_name}.{}", fields.w_o)),
        w_q_shape,
        w_k_shape,
        w_v_shape,
        w_o_shape,
    })
}

/// Recursively walk `stmts` for `let <name> = <ClassName>(...)` bindings
/// where `<ClassName>` is a `@wggo_target`-decorated model. Descends
/// into block-shaped statements (`FnDef.body`, top-level `If`/`For`/
/// `While`/`WhileLet`/`Match` arms, plain `Block`s, `Decorated` inner
/// stmts, and the training DSL: `TrainBlock` section bodies +
/// `GradBlock` body — the latter two matter because real user code
/// puts model instantiations inside `train(...)` and `grad(...)`).
fn walk_for_wggo_instantiations(
    stmts: &[nsl_ast::stmt::Stmt],
    decorated: &std::collections::HashMap<nsl_ast::Symbol, &ModelDef>,
    interner: &Interner,
    targets: &mut Vec<WggoGradTarget>,
) {
    use nsl_ast::block::TrainSection;
    use nsl_ast::pattern::PatternKind;
    use nsl_ast::stmt::StmtKind as SK;

    for stmt in stmts {
        match &stmt.kind {
            SK::VarDecl {
                pattern,
                value: Some(init),
                ..
            } => {
                if let PatternKind::Ident(name_sym) = &pattern.kind {
                    let var_name = interner.resolve(name_sym.0).unwrap_or("").to_string();
                    if !var_name.is_empty() {
                        if let Some(t) =
                            try_build_wggo_target(&var_name, init, decorated, interner)
                        {
                            targets.push(t);
                        }
                    }
                }
            }
            SK::FnDef(fn_def) => {
                walk_for_wggo_instantiations(&fn_def.body.stmts, decorated, interner, targets);
            }
            SK::If {
                then_block,
                elif_clauses,
                else_block,
                ..
            } => {
                walk_for_wggo_instantiations(&then_block.stmts, decorated, interner, targets);
                for (_, blk) in elif_clauses {
                    walk_for_wggo_instantiations(&blk.stmts, decorated, interner, targets);
                }
                if let Some(blk) = else_block {
                    walk_for_wggo_instantiations(&blk.stmts, decorated, interner, targets);
                }
            }
            SK::For { body, .. } | SK::While { body, .. } | SK::WhileLet { body, .. } => {
                walk_for_wggo_instantiations(&body.stmts, decorated, interner, targets);
            }
            SK::Match { arms, .. } => {
                for arm in arms {
                    walk_for_wggo_instantiations(&arm.body.stmts, decorated, interner, targets);
                }
            }
            SK::Decorated { stmt: inner, .. } => {
                walk_for_wggo_instantiations(
                    std::slice::from_ref(inner.as_ref()),
                    decorated,
                    interner,
                    targets,
                );
            }
            // Training DSL: `train(model = m, epochs = N): ...`. Each
            // section may carry user statements (Step/Eval bodies, raw
            // `Stmt`, `Data` stmt list, `Callbacks` with method bodies).
            // Variants that hold a single `Expr` (Optimizer, Scheduler,
            // Distribute) cannot bind a model instantiation `let` and
            // are skipped.
            SK::TrainBlock(train) => {
                for section in &train.sections {
                    match section {
                        TrainSection::Step { body, .. }
                        | TrainSection::Eval { body, .. } => {
                            walk_for_wggo_instantiations(
                                &body.stmts,
                                decorated,
                                interner,
                                targets,
                            );
                        }
                        TrainSection::Data(stmts) => {
                            walk_for_wggo_instantiations(stmts, decorated, interner, targets);
                        }
                        TrainSection::Stmt(inner) => {
                            walk_for_wggo_instantiations(
                                std::slice::from_ref(inner.as_ref()),
                                decorated,
                                interner,
                                targets,
                            );
                        }
                        TrainSection::Callbacks(callbacks) => {
                            for cb in callbacks {
                                walk_for_wggo_instantiations(
                                    &cb.body.stmts,
                                    decorated,
                                    interner,
                                    targets,
                                );
                            }
                        }
                        TrainSection::Optimizer(_)
                        | TrainSection::Scheduler(_)
                        | TrainSection::Distribute(_) => {
                            // Pure expressions; cannot contain `let` instantiations.
                        }
                    }
                }
            }
            // `grad(targets): body` — the body is a `Block` of user
            // statements, which is exactly where instantiations live in
            // gradient-eval contexts.
            SK::GradBlock(grad) => {
                walk_for_wggo_instantiations(&grad.body.stmts, decorated, interner, targets);
            }
            // Remaining variants are leaves for instantiation-discovery
            // purposes (Break/Continue/Return/Yield/Assign/Expr/Import/
            // FromImport, type/struct/enum/trait/agent defs, KernelDef/
            // TokenizerDef/DatasetDef/DatatypeDef/QuantBlock/ServeBlock).
            // None can contain a `let x = Class(...)` that binds in the
            // enclosing scope.
            _ => {}
        }
    }
}

/// Pre-scan the AST for WGGO Phase 2 backward-pass targets — one
/// `WggoGradTarget` per *instantiation site* of any `@wggo_target`-
/// decorated model.
///
/// Behaviour:
///
/// * No `@wggo_target` decorators in the AST → empty `Vec`.
/// * Decorator present but no instantiation reachable → empty `Vec`
///   (the §5.5 refusal in Task 11 is the place that errors here, not
///   pre-scan).
/// * Constructor-arg or field-shape evaluation failure for a single
///   instantiation → silent skip of that instantiation. Other
///   instantiations may still succeed.
/// * Initializer that isn't a direct call to a decorated class → skip
///   (e.g. `let attn = SomeOther()`).
///
/// This best-effort approach mirrors
/// [`pre_scan_awq_projections_from_ast`]: pre-scan runs early, and the
/// authoritative validation lives in semantic check (Phase 1 Tasks 1-4)
/// and Task 11 refusal.
pub fn pre_scan_wggo_targets_from_ast(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> Vec<WggoGradTarget> {
    use std::collections::HashMap;

    // 1. Collect every `@wggo_target`-decorated model (top-level, with
    //    or without an outer `@decorator` envelope around the model).
    let mut decorated: HashMap<nsl_ast::Symbol, &ModelDef> = HashMap::new();
    for stmt in &ast.stmts {
        let model_def_opt = match &stmt.kind {
            StmtKind::ModelDef(m) => Some(m),
            StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
                StmtKind::ModelDef(m) => Some(m),
                _ => None,
            },
            _ => None,
        };
        if let Some(model_def) = model_def_opt {
            if model_has_wggo_target_decorator(model_def, interner) {
                decorated.insert(model_def.name, model_def);
            }
        }
    }

    if decorated.is_empty() {
        return Vec::new();
    }

    // 2. Walk for instantiations and build targets.
    let mut targets = Vec::new();
    walk_for_wggo_instantiations(&ast.stmts, &decorated, interner, &mut targets);
    targets
}

pub fn check_discovery_agreement(
    pre_scan: &[DiscoveredProjection],
    in_compile: &[DiscoveredProjection],
) -> Result<(), DiscoveryError> {
    if pre_scan == in_compile {
        return Ok(());
    }

    Err(DiscoveryError::Divergence {
        pre_scan_names: pre_scan.iter().map(|p| p.projection.0.clone()).collect(),
        in_compile_names: in_compile.iter().map(|p| p.projection.0.clone()).collect(),
    })
}

// ── public API ────────────────────────────────────────────────────────────────

/// Walk `model_def`'s `forward` method body, collecting linear-projection sites,
/// and return a `Vec<DiscoveredProjection>` in forward-pipe order
/// (first occurrence per field name).
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
        if !model_field_types.contains_key(&field_name)
            && !tensor_shapes.contains_key(&field_name)
        {
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

    dedup_preserving_first_seen(&mut matched);

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
        if !model_field_types.contains_key(&field_name)
            && !tensor_shapes.contains_key(&field_name)
        {
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

    dedup_preserving_first_seen(&mut matched);

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

    fn repo_fixture(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    fn parse_module(source: &str) -> (nsl_ast::Module, Interner) {
        let mut interner = Interner::new();
        let (tokens, lex_diags) =
            nsl_lexer::tokenize(source, nsl_errors::FileId(0), &mut interner);
        assert!(
            lex_diags
                .iter()
                .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
            "fixture must lex cleanly: {lex_diags:?}"
        );
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parsed
                .diagnostics
                .iter()
                .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
            "fixture must parse cleanly: {:?}",
            parsed.diagnostics
        );
        (parsed.module, interner)
    }

    #[test]
    fn pre_scan_finds_both_projections_in_tinymlp_fixture() {
        let source = std::fs::read_to_string(repo_fixture("awq_calibration_mlp.nsl"))
            .expect("fixture readable");
        let (ast, interner) = parse_module(&source);
        let discovered = pre_scan_awq_projections_from_ast(&ast, &interner);

        assert_eq!(discovered.len(), 2, "TinyMLP has up_proj + down_proj");
        assert!(discovered.iter().any(|d| d.projection.0 == "TinyMLP.up_proj"));
        assert!(discovered.iter().any(|d| d.projection.0 == "TinyMLP.down_proj"));

        let up = discovered
            .iter()
            .find(|d| d.projection.0 == "TinyMLP.up_proj")
            .expect("up projection present");
        assert_eq!(up.weight_shape, [128, 64]);

        let down = discovered
            .iter()
            .find(|d| d.projection.0 == "TinyMLP.down_proj")
            .expect("down projection present");
        assert_eq!(down.weight_shape, [64, 128]);
    }

    #[test]
    fn pre_scan_returns_empty_when_no_quantize_decorator() {
        let source = "fn main():\n    return 0\n";
        let (ast, interner) = parse_module(source);
        let discovered = pre_scan_awq_projections_from_ast(&ast, &interner);
        assert!(discovered.is_empty());
    }

    #[test]
    fn pre_scan_preserves_forward_projection_order() {
        let source = std::fs::read_to_string(repo_fixture("awq_calibration_mlp.nsl"))
            .expect("fixture readable");
        let (ast, interner) = parse_module(&source);
        let discovered = pre_scan_awq_projections_from_ast(&ast, &interner);
        let names: Vec<_> = discovered.iter().map(|d| d.projection.0.as_str()).collect();

        assert_eq!(
            names,
            vec!["TinyMLP.up_proj", "TinyMLP.down_proj"],
            "discovery order should follow the forward pipe, not alphabetical sorting"
        );
    }

    #[test]
    fn check_discovery_agreement_reports_divergence() {
        let pre_scan = vec![
            DiscoveredProjection {
                projection: ProjectionRef("TinyMLP.up_proj".into()),
                weight_shape: [128, 64],
            },
            DiscoveredProjection {
                projection: ProjectionRef("TinyMLP.down_proj".into()),
                weight_shape: [64, 128],
            },
        ];
        let in_compile = vec![DiscoveredProjection {
            projection: ProjectionRef("TinyMLP.up_proj".into()),
            weight_shape: [128, 64],
        }];

        let err = check_discovery_agreement(&pre_scan, &in_compile)
            .expect_err("mismatched discovery sets must refuse");
        let msg = err.to_string();

        assert!(msg.contains("calibration: discovery divergence"));
        assert!(msg.contains("pre-scan:"));
        assert!(msg.contains("in-compile:"));
        assert!(msg.contains("TinyMLP.down_proj"));
    }

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
    fn single_glob_resolves_to_both_projections_in_pipe_order() {
        // Canonical order is forward-pipe order, NOT alphabetical. The pre-scan
        // path walks the AST in the same order; check_discovery_agreement
        // requires the in-compile path to produce an identical sequence. See
        // pre_scan_preserves_forward_projection_order for the matching anchor.
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
        // Pipe order in the fixture: up_proj first, then down_proj.
        assert_eq!(result[0].projection.0, "TinyMLP.up_proj");
        assert_eq!(result[1].projection.0, "TinyMLP.down_proj");
        assert_eq!(result[0].weight_shape, [64, 128]);
        assert_eq!(result[1].weight_shape, [128, 64]);
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
        let (model_def, quant_block, interner, mut field_types, mut tensor_shapes) =
            mlp_fixture_two_linears();
        // Remove down_proj from BOTH maps so it appears in the pipe chain but
        // resolves to no static weight. The discovery validator accepts a
        // field present in *either* `model_field_types` (named layer types like
        // `Linear`) or `tensor_shapes` (raw `Tensor<…>` parameters); both
        // qualify as static weights for AWQ. The error fires only when both
        // are absent.
        field_types.remove("down_proj");
        tensor_shapes.remove("down_proj");
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

    // ── WGGO Phase 2 Task 6: pre_scan_wggo_targets_from_ast ───────────────

    #[test]
    fn pre_scan_wggo_finds_attention_target_with_resolved_shapes() {
        let source = r#"
model Attention(dim: int, num_heads: int):
    head_dim: int = dim // num_heads
    q_proj: Tensor = randn([dim, dim])
    k_proj: Tensor = randn([dim, dim])
    v_proj: Tensor = randn([dim, dim])
    o_proj: Tensor = randn([dim, dim])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let m = Attention(dim=4096, num_heads=32)
    let x = randn([1, 4096])
    let y = m.forward(x)
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert_eq!(targets.len(), 1, "one Attention instantiation → one target");
        let t = &targets[0];
        assert_eq!(t.layer_key, "m");
        assert_eq!(t.class_name, "Attention");
        assert_eq!(t.head_dim, 128);
        assert_eq!(t.w_q_shape, [4096, 4096]);
        assert_eq!(t.w_k_shape, [4096, 4096]);
        assert_eq!(t.w_v_shape, [4096, 4096]);
        assert_eq!(t.w_o_shape, [4096, 4096]);
        assert_eq!(t.w_q.0, "m.q_proj");
        assert_eq!(t.w_k.0, "m.k_proj");
        assert_eq!(t.w_v.0, "m.v_proj");
        assert_eq!(t.w_o.0, "m.o_proj");
    }

    #[test]
    fn pre_scan_wggo_returns_empty_without_decorator() {
        let source = "fn main():\n    let x = 0\n";
        let (ast, interner) = parse_module(source);
        assert!(pre_scan_wggo_targets_from_ast(&ast, &interner).is_empty());
    }

    #[test]
    fn pre_scan_wggo_returns_empty_when_decorated_class_not_instantiated() {
        let source = r#"
model Attention(dim: int):
    head_dim: int = dim
    q_proj: Tensor = zeros([4, 4])
    k_proj: Tensor = zeros([4, 4])
    v_proj: Tensor = zeros([4, 4])
    o_proj: Tensor = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor) -> Tensor:
        return x

model OtherModel:
    weight: Tensor = zeros([4, 4])
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let m = OtherModel()
    let x = zeros([4])
    let y = m.forward(x)
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert!(
            targets.is_empty(),
            "decorated class not instantiated → empty (Task 11 refusal handles this)"
        );
    }

    #[test]
    fn pre_scan_wggo_handles_multiple_instantiations_in_source_order() {
        let source = r#"
model Attention(dim: int, num_heads: int):
    head_dim: int = dim // num_heads
    q_proj: Tensor = zeros([dim, dim])
    k_proj: Tensor = zeros([dim, dim])
    v_proj: Tensor = zeros([dim, dim])
    o_proj: Tensor = zeros([dim, dim])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let small = Attention(dim=64, num_heads=4)
    let large = Attention(dim=4096, num_heads=32)
    let x = zeros([1, 64])
    let y = small.forward(x)
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].layer_key, "small");
        assert_eq!(targets[0].head_dim, 16);
        assert_eq!(targets[0].w_q_shape, [64, 64]);
        assert_eq!(targets[1].layer_key, "large");
        assert_eq!(targets[1].head_dim, 128);
        assert_eq!(targets[1].w_q_shape, [4096, 4096]);
    }

    /// Real user code instantiates models inside `train` blocks because that's
    /// where the training DSL lives. The walker must descend into TrainBlock
    /// section bodies (regression for code-review I1 of commit 795cbabc).
    #[test]
    fn pre_scan_wggo_finds_target_inside_train_block() {
        let source = r#"
model Attention(dim: int, num_heads: int):
    head_dim: int = dim // num_heads
    q_proj: Tensor = zeros([dim, dim])
    k_proj: Tensor = zeros([dim, dim])
    v_proj: Tensor = zeros([dim, dim])
    o_proj: Tensor = zeros([dim, dim])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let m = Attention(dim=64, num_heads=4)
    train(model = m, epochs = 1):
        optimizer: SGD(lr = 0.01)
        step(batch):
            let inner_attn = Attention(dim=128, num_heads=8)
            let pred = m.forward(zeros([1, 64]))
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        // Both `m` (in fn main scope) AND `inner_attn` (inside train.step.body) must be found.
        assert_eq!(
            targets.len(),
            2,
            "walker must descend into TrainBlock.sections[].body — got {targets:?}",
        );
        let layer_keys: Vec<&str> = targets.iter().map(|t| t.layer_key.as_str()).collect();
        assert!(
            layer_keys.contains(&"m"),
            "missing top-level instantiation: {layer_keys:?}"
        );
        assert!(
            layer_keys.contains(&"inner_attn"),
            "missing train.step body instantiation: {layer_keys:?}"
        );
    }

    /// `grad(targets): ...` blocks have a `body: Block` that may contain user
    /// instantiations. The walker must descend into it (regression for
    /// code-review I1 of commit 795cbabc).
    #[test]
    fn pre_scan_wggo_finds_target_inside_grad_block() {
        let source = r#"
model Attention(dim: int, num_heads: int):
    head_dim: int = dim // num_heads
    q_proj: Tensor = zeros([dim, dim])
    k_proj: Tensor = zeros([dim, dim])
    v_proj: Tensor = zeros([dim, dim])
    o_proj: Tensor = zeros([dim, dim])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let w = ones([4])
    let (loss, grads) = grad(w):
        let m = Attention(dim=64, num_heads=4)
        let y = m.forward(zeros([1, 64]))
        y.sum()
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert_eq!(
            targets.len(),
            1,
            "walker must descend into GradBlock.body — got {targets:?}",
        );
        assert_eq!(targets[0].layer_key, "m");
    }
}
