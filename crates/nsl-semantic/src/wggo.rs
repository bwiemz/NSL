//! Semantic validation for the `@wggo` decorator.
//!
//! Accepted arguments:
//!   * `mode = full | greedy | off`
//!   * `target = <ident|string>`
//!
//! `@wggo` is valid on the `train` block.  When absent, WGGO decides
//! whether to activate based on the presence of other training
//! decorators (WRGA + CPDT + CEP).

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WggoMode {
    Full,
    Greedy,
    Off,
}

impl WggoMode {
    pub fn as_str(self) -> &'static str {
        match self {
            WggoMode::Full => "full",
            WggoMode::Greedy => "greedy",
            WggoMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" | "auto" => Some(WggoMode::Full),
            "greedy" => Some(WggoMode::Greedy),
            "off" | "disable" | "disabled" => Some(WggoMode::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WggoConfig {
    pub mode: WggoMode,
    pub target: Option<Symbol>,
    pub span: Span,
}

pub fn validate_wggo_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<WggoConfig> {
    let mut mode = WggoMode::Full;
    let mut target: Option<Symbol> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error(
                        "@wggo: positional arguments are not allowed".to_string(),
                    )
                    .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => WggoMode::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => WggoMode::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(m) => mode = m,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@wggo: mode must be full, greedy, or off".to_string(),
                            )
                            .with_label(arg.span, "invalid mode"),
                        ),
                    }
                }
                "target" => match &arg.value.kind {
                    ExprKind::Ident(sym) => target = Some(*sym),
                    ExprKind::StringLiteral(_) => target = Some(*name_sym),
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@wggo: target must be an identifier (e.g. h100)".to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@wggo: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(WggoConfig {
        mode,
        target,
        span: deco.span,
    })
}

// ---------------------------------------------------------------------------
// WGGO Phase 2 Task 2: `@wggo_target` required-arguments validation
// ---------------------------------------------------------------------------

/// The five named arguments `@wggo_target` requires.
///
/// Order matters for the diagnostic message: it's the canonical ordering
/// the user-facing error lists, and the order of `WGGO_TARGET_REQUIRED_ARGS`
/// is the source of truth for that header.
pub const WGGO_TARGET_REQUIRED_ARGS: &[&str] = &["w_q", "w_k", "w_v", "w_o", "head_dim"];

/// Validate that a `@wggo_target` decorator provides all five required
/// named arguments (`w_q`, `w_k`, `w_v`, `w_o`, `head_dim`).
///
/// This task ONLY checks that the names appear as arguments. It does NOT
/// validate that the values are `self.<field>` references (Task 3) or that
/// those fields exist on the model with appropriate types (Task 4).
///
/// Called from both:
///   * `crates/nsl-semantic/src/checker/model.rs` — `ModelMember::Method`
///   * `crates/nsl-semantic/src/checker/stmt.rs` — top-level `StmtKind::FnDef`
///
/// Both call sites guard on the placement check from Task 1 first; this
/// function assumes placement has already been validated.
pub fn validate_wggo_target_required_args(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let provided: std::collections::HashSet<String> = match deco.args {
        Some(ref args) => args
            .iter()
            .filter_map(|a| a.name.map(resolve_sym))
            .collect(),
        None => std::collections::HashSet::new(),
    };
    let missing: Vec<&str> = WGGO_TARGET_REQUIRED_ARGS
        .iter()
        .filter(|a| !provided.contains(**a))
        .copied()
        .collect();
    if !missing.is_empty() {
        diagnostics.push(
            Diagnostic::error(format!(
                "@wggo_target requires arguments: w_q, w_k, w_v, w_o, head_dim; missing: {missing:?}"
            ))
            .with_label(deco.span, "missing required arguments"),
        );
    }
}

// ---------------------------------------------------------------------------
// WGGO Phase 2 Task 3: `@wggo_target` argument-expression validation
// ---------------------------------------------------------------------------

/// Returns the field-name `Symbol` if `expr` is a `self.<field>`
/// member-access; otherwise returns `None`.
///
/// The returned `Symbol` is the interned identifier of the `<field>`
/// portion (e.g. for `self.q_proj`, returns `Some(<symbol of "q_proj">)`).
/// Resolve it back to a string with `Interner::resolve(sym.0)` if you need
/// the field name for diagnostics or field-type lookup.
///
/// In the NSL AST, `self` is parsed as `ExprKind::SelfRef` (a distinct
/// variant), and `self.foo` is `MemberAccess { object: SelfRef, member: foo }`.
///
/// Visibility: `pub(crate)` so Task 4's field-type validator can re-use the
/// same parsing logic to extract the referenced field symbol.
pub(crate) fn is_self_field_expr(expr: &nsl_ast::expr::Expr) -> Option<Symbol> {
    let ExprKind::MemberAccess { object, member } = &expr.kind else {
        return None;
    };
    if matches!(object.kind, ExprKind::SelfRef) {
        Some(*member)
    } else {
        None
    }
}

/// Returns a short, human-readable summary of an expression's kind for use
/// in the `must be a self.<field> reference; got <KIND>` diagnostic.
///
/// The returned strings are stable, lowercase, no trailing punctuation, and
/// chosen to read naturally after the word "got" in the diagnostic.
fn describe_expr_kind(expr: &nsl_ast::expr::Expr) -> &'static str {
    match &expr.kind {
        ExprKind::IntLiteral(_) => "int literal",
        ExprKind::FloatLiteral(_) => "float literal",
        ExprKind::StringLiteral(_) => "string literal",
        ExprKind::FString(_) => "f-string literal",
        ExprKind::BoolLiteral(_) => "bool literal",
        ExprKind::NoneLiteral => "None literal",
        ExprKind::ListLiteral(_) => "list literal",
        ExprKind::TupleLiteral(_) => "tuple literal",
        ExprKind::DictLiteral(_) => "dict literal",
        ExprKind::Ident(_) => "bare identifier",
        // `SelfRef` and `MemberAccess` covering non-self cases — the
        // self-field case is filtered out before we call this helper.
        ExprKind::SelfRef => "bare self",
        ExprKind::MemberAccess { .. } => "non-self member access",
        ExprKind::BinaryOp { .. } => "binary op",
        ExprKind::UnaryOp { .. } => "unary op",
        ExprKind::Pipe { .. } => "pipe expression",
        ExprKind::Subscript { .. } => "subscript expression",
        ExprKind::Call { .. } => "function call",
        ExprKind::Lambda { .. } => "lambda",
        ExprKind::BlockExpr(_) => "block expression",
        ExprKind::ListComp { .. } => "list comprehension",
        ExprKind::IfExpr { .. } => "if expression",
        ExprKind::MatchExpr { .. } => "match expression",
        ExprKind::Range { .. } => "range expression",
        ExprKind::Paren(_) => "parenthesized expression",
        ExprKind::Await(_) => "await expression",
        ExprKind::Error => "invalid expression",
    }
}

/// Validate that each `@wggo_target` argument whose name is in
/// `WGGO_TARGET_REQUIRED_ARGS` (`w_q`, `w_k`, `w_v`, `w_o`, `head_dim`) is
/// a `self.<field>` member-access expression.
///
/// This task does NOT verify that the field actually exists on the
/// enclosing model or that the field's type matches expectations — those
/// checks live in Task 4.
///
/// Args with names NOT in the required-args set are skipped here and
/// flagged elsewhere (Task 2 handles missing names; unknown extra names
/// are out of scope for both Tasks 2 and 3).
///
/// Args with no `name` (positional) are silently skipped — Task 2 will
/// have already reported them as missing required arguments since their
/// canonical name slot is empty.
///
/// Called from both:
///   * `crates/nsl-semantic/src/checker/model.rs` — `ModelMember::Method`
///   * `crates/nsl-semantic/src/checker/stmt.rs` — top-level `StmtKind::FnDef`
pub fn validate_wggo_target_self_field_args(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(ref args) = deco.args else { return };
    for arg in args {
        let Some(arg_name_sym) = arg.name else { continue };
        let arg_name = resolve_sym(arg_name_sym);
        if !WGGO_TARGET_REQUIRED_ARGS.contains(&arg_name.as_str()) {
            continue;
        }
        if is_self_field_expr(&arg.value).is_none() {
            let summary = describe_expr_kind(&arg.value);
            diagnostics.push(
                Diagnostic::error(format!(
                    "@wggo_target argument '{arg_name}' must be a self.<field> reference; got {summary}"
                ))
                .with_label(arg.value.span, "expected self.<field>"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// WGGO Phase 2 Task 4: `@wggo_target` field-existence + field-type validation
// ---------------------------------------------------------------------------

/// Returns a short, human-readable summary of a field's type annotation
/// for use in the `must reference a Tensor field; got <SUMMARY>` diagnostic.
///
/// Stable shape:
///   - Builtins (`int`, `float`, `bool`, `str`) collapse to that bare name.
///   - `Tensor<...>` collapses to `"Tensor"` (shape/dtype not part of Task 4).
///   - `Param<...>` / `Buffer<...>` / `Sparse<...>` are tensor-shaped; we
///     report them as their kind name so users see a precise hint about
///     why their field isn't accepted as a `Tensor` arg (a `Param`/`Buffer`
///     is not a `Tensor` for the purpose of `@wggo_target`).
///   - Other named types render as `named:<Name>`.
///   - All remaining variants collapse to a coarse kind label.
///
/// The strings here are reused as the literal `got <SUMMARY>` tail of the
/// diagnostic, so they should read naturally after `got `.
fn type_expr_summary(
    type_ann: &nsl_ast::types::TypeExpr,
    resolve_sym: &dyn Fn(Symbol) -> String,
) -> String {
    use nsl_ast::types::TypeExprKind;
    match &type_ann.kind {
        TypeExprKind::Named(sym) => {
            let name = resolve_sym(*sym);
            if matches!(name.as_str(), "int" | "float" | "bool" | "str") {
                name
            } else {
                format!("named:{name}")
            }
        }
        TypeExprKind::Tensor { .. } => "Tensor".to_string(),
        TypeExprKind::Param { .. } => "Param".to_string(),
        TypeExprKind::Buffer { .. } => "Buffer".to_string(),
        TypeExprKind::Sparse { .. } => "Sparse".to_string(),
        TypeExprKind::Generic { name, .. } => format!("generic:{}", resolve_sym(*name)),
        TypeExprKind::Function { .. } => "function type".to_string(),
        TypeExprKind::Union(_) => "union type".to_string(),
        TypeExprKind::Tuple(_) => "tuple type".to_string(),
        TypeExprKind::Wildcard => "wildcard".to_string(),
        TypeExprKind::FixedArray { .. } => "fixed array".to_string(),
        TypeExprKind::Borrow(_) => "borrow".to_string(),
    }
}

/// Validate that each `@wggo_target` argument named in
/// `WGGO_TARGET_REQUIRED_ARGS` references an existing field of the
/// enclosing model with an appropriate type:
///
///   * `w_q` / `w_k` / `w_v` / `w_o`  →  must reference a `Tensor` field
///   * `head_dim`                      →  must reference an `int` field
///
/// Run AFTER `validate_wggo_target_self_field_args` (Task 3): args whose
/// values are not `self.<field>` are silently skipped here, since Task 3
/// will already have reported the higher-priority "must be a self.<field>
/// reference" error for them. Likewise this helper does NOT re-validate
/// shape/dimension of Tensor fields — that's pre-scan's job in Task 6.
///
/// Called only from `crates/nsl-semantic/src/checker/model.rs` because we
/// need the enclosing `ModelDef` to resolve field types. The standalone-fn
/// site in `checker/stmt.rs` deliberately skips this check (no enclosing
/// model context is available there).
pub fn validate_wggo_target_field_types(
    deco: &Decorator,
    model_def: &nsl_ast::decl::ModelDef,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(ref args) = deco.args else { return };
    let model_name = resolve_sym(model_def.name);

    // Build field-name → type map from the model's `LayerDecl` members.
    // Method members (`ModelMember::Method`) are skipped — `@wggo_target`
    // arguments only ever reference data fields, never methods.
    let mut field_types: std::collections::HashMap<String, &nsl_ast::types::TypeExpr> =
        std::collections::HashMap::new();
    for member in &model_def.members {
        if let nsl_ast::decl::ModelMember::LayerDecl { name, type_ann, .. } = member {
            field_types.insert(resolve_sym(*name), type_ann);
        }
    }

    for arg in args {
        let Some(arg_name_sym) = arg.name else { continue };
        let arg_name = resolve_sym(arg_name_sym);
        if !WGGO_TARGET_REQUIRED_ARGS.contains(&arg_name.as_str()) {
            continue;
        }
        // Only consider args whose value IS a `self.<field>` ref. If it
        // isn't, Task 3 has already reported the higher-priority error
        // and we'd just be piling on with a misleading "not found"
        // message.
        let Some(field_sym) = is_self_field_expr(&arg.value) else {
            continue;
        };
        let field_name = resolve_sym(field_sym);
        let Some(field_type) = field_types.get(&field_name) else {
            diagnostics.push(
                Diagnostic::error(format!(
                    "@wggo_target field reference 'self.{field_name}' not found in model '{model_name}'"
                ))
                .with_label(arg.value.span, "unknown field"),
            );
            continue;
        };

        let summary = type_expr_summary(field_type, resolve_sym);
        // Per spec: head_dim must be an int; the four projections must be
        // Tensor. The article ("a"/"an") is chosen per the expected type.
        //
        // `expected` matches the literal output of `type_expr_summary` —
        // for `head_dim` that is the string "int" (the NSL builtin). NSL
        // does not currently expose other integer scalar types like
        // `i32`/`i64`/etc. as field annotations, so this exact-string
        // comparison is sufficient. If NSL ever adds them, extend
        // `type_expr_summary` and broaden the match here.
        let (expected, article) = if arg_name == "head_dim" {
            ("int", "an")
        } else {
            ("Tensor", "a")
        };
        let summary_matches = summary == expected;
        if !summary_matches {
            diagnostics.push(
                Diagnostic::error(format!(
                    "@wggo_target argument '{arg_name}' must reference {article} {expected} field; got {summary}"
                ))
                .with_label(arg.value.span, "wrong field type"),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse_roundtrip() {
        for m in [WggoMode::Full, WggoMode::Greedy, WggoMode::Off] {
            assert_eq!(WggoMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(WggoMode::parse("auto"), Some(WggoMode::Full));
        assert!(WggoMode::parse("nonsense").is_none());
    }

    #[test]
    fn wggo_target_required_args_is_canonical_five() {
        // Guard against accidental reordering or additions: the diagnostic
        // message header lists these in a fixed order.
        assert_eq!(
            WGGO_TARGET_REQUIRED_ARGS,
            &["w_q", "w_k", "w_v", "w_o", "head_dim"]
        );
    }
}
