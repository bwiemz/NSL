//! Calibration target discovery.
//!
//! [`pre_scan_wggo_targets_from_ast`] finds the instances of models whose
//! `forward` carries `@wggo_target` and resolves the four projection weights
//! and `head_dim` each one names. [`DiscoveredProjection`] is the record a
//! calibration caller uses to describe a linear projection whose input
//! activations the harness should retain.
//!
//! This module used to open with AWQ projection discovery: a walk of the
//! forward body of every `@quantize(dtype="awq4")` model. That decorator is
//! refused and the walk is deleted with the AWQ calibration hook.

use crate::calibration::observation::ProjectionRef;
use nsl_ast::decl::{ModelDef, ModelMember};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;

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
    /// Declaration-order indices into the model's tensor-weight list.
    /// Computed by pre-scan via the same walk as
    /// `nsl-semantic::export::validate_model_method_exports`:
    /// iterate `ModelDef.members`, count tensor-typed `LayerDecl`s,
    /// match by field name to each projection.
    ///
    /// Indices feed `weights_base + index * PTR_SIZE` in the
    /// scaffolding emitter to load the runtime weight pointer for
    /// each projection. See spec §5.7.
    pub w_q_index: u32,
    pub w_k_index: u32,
    pub w_v_index: u32,
    pub w_o_index: u32,
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

/// Returns `true` if `type_ann` is a tensor-like type (`Tensor`, `Param`,
/// or `Buffer`). Mirrors the predicate in
/// `nsl-semantic::export::is_tensor_type` so that both the pre-scan and
/// the semantic validator agree on which `LayerDecl` fields are counted as
/// tensor weights when assigning declaration-order indices.
///
/// Pre-scan operates on the raw AST before the type-checker runs. A bare
/// `Tensor` annotation (without angle brackets, e.g. `q_proj: Tensor = ...`)
/// is parsed as `TypeExprKind::Named("Tensor")`. A parameterized form
/// (`Tensor<[dim, dim], f32>`) is parsed as `TypeExprKind::Tensor { .. }`.
/// Both forms must be treated as tensor-typed; the `Named` case is resolved
/// by `resolve_tensor_field_index` which passes the resolved name string.
fn is_tensor_type_ann(type_ann: &nsl_ast::types::TypeExpr, interner: &Interner) -> bool {
    match &type_ann.kind {
        nsl_ast::types::TypeExprKind::Tensor { .. }
        | nsl_ast::types::TypeExprKind::Param { .. }
        | nsl_ast::types::TypeExprKind::Buffer { .. } => true,
        // Bare names: `Tensor`, `Param`, `Buffer` parse as Named(sym) before
        // the type-checker upgrades them to the parameterized forms.
        nsl_ast::types::TypeExprKind::Named(sym) => matches!(
            interner.resolve(sym.0),
            Some("Tensor") | Some("Param") | Some("Buffer")
        ),
        _ => false,
    }
}

/// Compute the declaration-order index of `field_name` in the model's
/// tensor-weight list. Returns `None` if the field doesn't exist or
/// isn't tensor-typed.
///
/// Mirrors `nsl-semantic::export::validate_model_method_exports`'s
/// tensor-weight enumeration (declaration-order, tensor-typed
/// `LayerDecl`s only). The two must agree because both produce
/// indices into the same runtime weight-pointer array.
fn resolve_tensor_field_index(
    model_def: &ModelDef,
    field_name: &str,
    interner: &Interner,
) -> Option<u32> {
    let mut idx: u32 = 0;
    for member in &model_def.members {
        if let nsl_ast::decl::ModelMember::LayerDecl { name, type_ann, .. } = member {
            if !is_tensor_type_ann(type_ann, interner) {
                continue;
            }
            if interner.resolve(name.0) == Some(field_name) {
                return Some(idx);
            }
            idx = idx.checked_add(1)?;
        }
    }
    None
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
/// Best-effort by design: errors surface later, in the semantic check and
/// the Task 11 refusal, not here.
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
        // head_dim == 0 is degenerate; skip silently — emit-time refusals would be worse than a no-op.
        EvalValue::Int(n) if n >= 1 && n <= u32::MAX as i64 => n as u32,
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

    // Resolve declaration-order indices for each projection's field name.
    // If any projection's field can't be resolved (e.g., typo, non-tensor
    // field), silently skip the target — the existing best-effort contract.
    // Semantic check Task 4 from PR #132 catches the typo case at semantic
    // time with a "field not found" diagnostic, so pre-scan never sees that.
    let w_q_index = resolve_tensor_field_index(model_def, &fields.w_q, interner)?;
    let w_k_index = resolve_tensor_field_index(model_def, &fields.w_k, interner)?;
    let w_v_index = resolve_tensor_field_index(model_def, &fields.w_v, interner)?;
    let w_o_index = resolve_tensor_field_index(model_def, &fields.w_o, interner)?;

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
        w_q_index,
        w_k_index,
        w_v_index,
        w_o_index,
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
                    if !var_name.is_empty()
                        && let Some(t) =
                            try_build_wggo_target(&var_name, init, decorated, interner)
                    {
                        targets.push(t);
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
            // FromImport, type/struct/enum/trait defs, KernelDef/
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
/// Best-effort by design: pre-scan runs early, and the authoritative
/// validation lives in semantic check (Phase 1 Tasks 1-4) and Task 11
/// refusal.
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
        if let Some(model_def) = model_def_opt
            && model_has_wggo_target_decorator(model_def, interner)
        {
            decorated.insert(model_def.name, model_def);
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

// ── Task 11 refusal helpers ───────────────────────────────────────────────────

/// Returns `true` when the AST contains at least one model with a
/// `@wggo_target` decorator on its `forward` method. Used by the §5.4
/// refusal in `entry_points.rs` to distinguish "no decorators at all"
/// from "decorators present but no reachable instantiation" (§5.5).
pub fn ast_has_wggo_target_decorators(ast: &nsl_ast::Module, interner: &Interner) -> bool {
    for stmt in &ast.stmts {
        let model_def_opt = match &stmt.kind {
            StmtKind::ModelDef(m) => Some(m),
            StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
                StmtKind::ModelDef(m) => Some(m),
                _ => None,
            },
            _ => None,
        };
        if let Some(model_def) = model_def_opt
            && model_has_wggo_target_decorator(model_def, interner)
        {
            return true;
        }
    }
    false
}

/// Returns the names of all model classes decorated with `@wggo_target`.
/// Diagnostics-only: used by the §5.5 refusal to report which classes
/// were decorated but none was reachable from the entry point.
pub fn list_decorated_class_names(ast: &nsl_ast::Module, interner: &Interner) -> Vec<String> {
    let mut out = Vec::new();
    for stmt in &ast.stmts {
        let model_def_opt = match &stmt.kind {
            StmtKind::ModelDef(m) => Some(m),
            StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
                StmtKind::ModelDef(m) => Some(m),
                _ => None,
            },
            _ => None,
        };
        if let Some(model_def) = model_def_opt
            && model_has_wggo_target_decorator(model_def, interner)
        {
            let name = interner.resolve(model_def.name.0).unwrap_or("").to_string();
            if !name.is_empty() {
                out.push(name);
            }
        }
    }
    out
}

/// Returns the name of the first top-level function that looks like an
/// entry point (named `"main"`) as a hint for the §5.5 diagnostic.
/// Returns `None` when the AST has no `fn main()`.
pub fn entry_point_fn_name(ast: &nsl_ast::Module, interner: &Interner) -> Option<String> {
    for stmt in &ast.stmts {
        let fn_def = match &stmt.kind {
            StmtKind::FnDef(f) => f,
            StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
                StmtKind::FnDef(f) => f,
                _ => continue,
            },
            _ => continue,
        };
        let name = interner.resolve(fn_def.name.0).unwrap_or("");
        if name == "main" {
            return Some(name.to_string());
        }
    }
    // Fall back to the first non-model, non-import top-level function.
    for stmt in &ast.stmts {
        let fn_def = match &stmt.kind {
            StmtKind::FnDef(f) => f,
            _ => continue,
        };
        let name = interner.resolve(fn_def.name.0).unwrap_or("").to_string();
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_lexer::Interner;

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
    /// Regression: head_dim == 0 is degenerate and must be silently skipped by
    /// pre-scan.  If it were accepted, the three integer-division sites in
    /// `WggoGradientHook` would panic on `x / 0` before the `.max(1)` guard
    /// could run.
    #[test]
    fn pre_scan_wggo_rejects_head_dim_zero() {
        // head_dim field is literally 0; this must produce zero targets, not panic.
        let source = r#"
model Attention(dim: int):
    head_dim: int = 0
    q_proj: Tensor = zeros([dim, dim])
    k_proj: Tensor = zeros([dim, dim])
    v_proj: Tensor = zeros([dim, dim])
    o_proj: Tensor = zeros([dim, dim])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let m = Attention(dim=64)
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert!(
            targets.is_empty(),
            "head_dim=0 must be silently rejected by pre-scan, got {targets:?}"
        );
    }

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

    // ── Task 6 fast-follows: walker edge-case contracts ────────────────

    /// Default constructor args propagate end-to-end through pre-scan: a
    /// `model Attention(dim: int = 4096, num_heads: int = 32)` instantiated
    /// with no args should resolve via `bind_constructor_args`'s default
    /// path (already unit-tested at the function level), and pre-scan
    /// should produce a target whose shapes reflect the defaults.
    ///
    /// This locks in the end-to-end contract — without it, a regression
    /// in the walker → bind_constructor_args path could silently swallow
    /// default-arg models.
    #[test]
    fn pre_scan_wggo_resolves_default_constructor_args() {
        let source = r#"
model Attention(dim: int = 4096, num_heads: int = 32):
    head_dim: int = dim // num_heads
    q_proj: Tensor = randn([dim, dim])
    k_proj: Tensor = randn([dim, dim])
    v_proj: Tensor = randn([dim, dim])
    o_proj: Tensor = randn([dim, dim])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let m = Attention()
    let x = randn([1, 4096])
    let y = m.forward(x)
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert_eq!(
            targets.len(),
            1,
            "Attention() with default args must produce one target; got {targets:?}",
        );
        let t = &targets[0];
        assert_eq!(t.layer_key, "m");
        assert_eq!(t.head_dim, 128, "default 4096 // 32 = 128");
        assert_eq!(t.w_q_shape, [4096, 4096], "default dim=4096 propagates to projections");
    }

    /// Walker contract: a non-`Call` initializer (here a parenthesized
    /// `Call`, parsed as `ExprKind::Paren(Call(...))`) is silently
    /// skipped because `try_build_wggo_target` matches only direct
    /// `ExprKind::Call`. Documents the intentional silent-skip; if the
    /// walker is ever generalized to unwrap `Paren`, update the
    /// expectation here.
    #[test]
    fn pre_scan_wggo_silently_skips_parenthesized_init() {
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
    let m = (Attention(dim=64, num_heads=4))
    let y = m.forward(zeros([1, 64]))
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert!(
            targets.is_empty(),
            "parenthesized init is not a direct Call → silently skipped (current contract); got {targets:?}",
        );
    }

    /// Walker contract: a `Call` whose callee is itself a member-access
    /// (e.g. `pkg.Attention(...)`) is silently skipped because
    /// `try_build_wggo_target` requires `ExprKind::Ident` for the
    /// callee. Documents the intentional silent-skip — qualified paths
    /// would need explicit symbol resolution before they can be
    /// matched against the `decorated` map.
    #[test]
    fn pre_scan_wggo_silently_skips_qualified_path_callee() {
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
    let pkg = some_factory()
    let m = pkg.Attention(dim=64, num_heads=4)
    let y = m.forward(zeros([1, 64]))
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert!(
            targets.is_empty(),
            "member-access callee `pkg.Attention(...)` is not an Ident callee → silently skipped (current contract); got {targets:?}",
        );
    }

    #[test]
    fn pre_scan_wggo_resolves_weight_indices_in_declaration_order() {
        // Attention model with q_proj, k_proj, v_proj, o_proj as the
        // ONLY tensor fields, in that declaration order. Indices must
        // be 0, 1, 2, 3 respectively. head_dim is int and not counted.
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
    let y = m.forward(zeros([1, 64]))
"#;
        let (ast, interner) = parse_module(source);
        let targets = pre_scan_wggo_targets_from_ast(&ast, &interner);
        assert_eq!(targets.len(), 1);
        let t = &targets[0];
        assert_eq!(t.w_q_index, 0, "q_proj is the first tensor field → index 0");
        assert_eq!(t.w_k_index, 1, "k_proj is the second tensor field → index 1");
        assert_eq!(t.w_v_index, 2, "v_proj is the third tensor field → index 2");
        assert_eq!(t.w_o_index, 3, "o_proj is the fourth tensor field → index 3");
    }
}
