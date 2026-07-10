//! Semantic validation for the WRGA decorators: `@wrga`, `@freeze`, `@adapter`.
//!
//! These decorators resolve at compile time to a [`WrgaBlock`] plus a set of
//! freeze/adapter glob patterns that the `nsl-codegen::wrga` driver consumes.
//!
//! The validation here mirrors the style of `vmap.rs` and `moe.rs`: we
//! check argument names, basic types, and enum values; the heavy lifting
//! (shape inference, rank allocation, roofline analysis) happens in codegen
//! once all inputs are resolved.

use nsl_ast::block::{WrgaBlock, WrgaMode};
use nsl_ast::decl::{Decorator, ModelMember};
use nsl_ast::expr::{Arg, ExprKind};
use nsl_ast::stmt::StmtKind;
use nsl_ast::types::TypeExprKind;
use nsl_ast::{Module, Span, Symbol};
use nsl_errors::Diagnostic;

/// Validated `@wrga(...)` configuration.
#[derive(Debug, Clone)]
pub struct WrgaConfig {
    pub block: WrgaBlock,
    /// WRGA paper §8.2: resolved string form of `block.adapter` (the
    /// `adapter=<Ident>` argument).  Populated at decorator-validation
    /// time so the codegen bridge does not need access to the interner
    /// to pass the adapter name through.  `None` when `block.adapter`
    /// is `None`.
    pub adapter_name: Option<String>,
}

/// Validate a single `@wrga(...)` decorator.
///
/// Accepted arguments:
/// * `mode = auto | manual | hybrid`   (default: `auto`)
/// * `budget = <int>`                  (optional; total adapter parameter cap)
/// * `target = <identifier>`           (optional; GPU name)
/// * `layers = ["blocks.6", ...]`      (optional; hybrid-mode layer scope)
/// * `adapter = <identifier>`          (optional; WRGA paper §8.2 — name of a
///   user-defined adapter model, e.g. `adapter=GatedLoRA`)
pub fn validate_wrga_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<WrgaConfig> {
    let mut mode = WrgaMode::Auto;
    let mut budget: Option<i64> = None;
    let mut target: Option<Symbol> = None;
    let mut layers: Vec<String> = Vec::new();
    let mut adapter: Option<Symbol> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error("@wrga: positional arguments are not allowed".to_string())
                        .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "mode" => {
                    let mode_str = extract_mode_symbol(arg, resolve_sym, diagnostics);
                    if let Some(s) = mode_str {
                        match WrgaMode::parse(&s) {
                            Some(m) => mode = m,
                            None => diagnostics.push(
                                Diagnostic::error(format!(
                                    "@wrga: invalid mode '{s}' (expected auto, manual, or hybrid)"
                                ))
                                .with_label(arg.span, "invalid mode"),
                            ),
                        }
                    }
                }
                "budget" => match &arg.value.kind {
                    ExprKind::IntLiteral(n) => {
                        if *n < 0 {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@wrga: budget must be non-negative".to_string(),
                                )
                                .with_label(arg.span, "negative budget"),
                            );
                        } else {
                            budget = Some(*n);
                        }
                    }
                    _ => diagnostics.push(
                        Diagnostic::error("@wrga: budget must be an integer literal".to_string())
                            .with_label(arg.span, "expected integer"),
                    ),
                },
                "target" => match &arg.value.kind {
                    ExprKind::Ident(sym) => target = Some(*sym),
                    ExprKind::StringLiteral(_) => {
                        // Allowed but we prefer an ident.  Don't error.
                        target = Some(*name_sym);
                    }
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@wrga: target must be an identifier (e.g. h100, rtx5070ti)"
                                .to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    ),
                },
                "adapter" => match &arg.value.kind {
                    ExprKind::Ident(sym) => adapter = Some(*sym),
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@wrga: adapter must be an identifier referring to a `model` \
                             declaration (WRGA paper §8.2; e.g. `adapter=GatedLoRA`)"
                                .to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    ),
                },
                "layers" => match &arg.value.kind {
                    ExprKind::ListLiteral(items) => {
                        for item in items {
                            match &item.kind {
                                ExprKind::StringLiteral(s) => layers.push(s.clone()),
                                _ => diagnostics.push(
                                    Diagnostic::error(
                                        "@wrga: layer entries must be string literals"
                                            .to_string(),
                                    )
                                    .with_label(item.span, "expected string"),
                                ),
                            }
                        }
                    }
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@wrga: layers must be a list of strings (e.g. [\"blocks.6\", \"blocks.7\"])"
                                .to_string(),
                        )
                        .with_label(arg.span, "expected list"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@wrga: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    // Cross-field validation:
    //   - hybrid mode needs at least one layer
    //   - manual mode should not be paired with `layers`
    match mode {
        WrgaMode::Hybrid if layers.is_empty() => {
            diagnostics.push(
                Diagnostic::error(
                    "@wrga(mode=hybrid) requires a `layers=[...]` argument".to_string(),
                )
                .with_label(deco.span, "missing layer scope"),
            );
        }
        WrgaMode::Manual if !layers.is_empty() => {
            diagnostics.push(
                Diagnostic::error(
                    "@wrga(mode=manual) ignores `layers=...`; use @adapter(target=...) instead"
                        .to_string(),
                )
                .with_label(deco.span, "layers ignored in manual mode"),
            );
        }
        _ => {}
    }

    let adapter_name = adapter.map(resolve_sym);
    Some(WrgaConfig {
        block: WrgaBlock {
            mode,
            budget,
            target,
            layers,
            adapter,
            span: deco.span,
        },
        adapter_name,
    })
}

/// Validated `@freeze(exclude=[...], include=[...])` configuration.
#[derive(Debug, Clone, Default)]
pub struct FreezeConfig {
    /// Patterns to *exclude* from freezing (i.e. leave trainable).
    pub exclude: Vec<String>,
    /// Patterns to *include* (freeze explicitly); cannot be combined with exclude.
    pub include: Vec<String>,
    pub span: Option<Span>,
}

pub fn validate_freeze_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> FreezeConfig {
    let mut out = FreezeConfig {
        span: Some(deco.span),
        ..Default::default()
    };
    let Some(ref args) = deco.args else {
        return out;
    };
    for arg in args {
        let Some(ref name_sym) = arg.name else {
            diagnostics.push(
                Diagnostic::error("@freeze: positional arguments are not allowed".to_string())
                    .with_label(arg.span, "expected `key = value`"),
            );
            continue;
        };
        let aname = resolve_sym(*name_sym);
        let target_vec = match aname.as_str() {
            "exclude" => &mut out.exclude,
            "include" => &mut out.include,
            other => {
                diagnostics.push(
                    Diagnostic::error(format!("@freeze: unknown argument '{other}'"))
                        .with_label(arg.span, "unknown argument"),
                );
                continue;
            }
        };
        match &arg.value.kind {
            ExprKind::ListLiteral(items) => {
                for item in items {
                    match &item.kind {
                        ExprKind::StringLiteral(s) => target_vec.push(s.clone()),
                        _ => diagnostics.push(
                            Diagnostic::error(
                                "@freeze: pattern entries must be string literals".to_string(),
                            )
                            .with_label(item.span, "expected string"),
                        ),
                    }
                }
            }
            _ => diagnostics.push(
                Diagnostic::error(
                    "@freeze arguments must be lists of glob strings (e.g. [\"blocks.6.*\"])"
                        .to_string(),
                )
                .with_label(arg.span, "expected list"),
            ),
        }
    }
    if !out.include.is_empty() && !out.exclude.is_empty() {
        diagnostics.push(
            Diagnostic::error(
                "@freeze: `include` and `exclude` cannot be combined — pick one".to_string(),
            )
            .with_label(deco.span, "conflicting freeze mode"),
        );
    }
    out
}

/// Validated `@adapter(type=lora|ia3|gatedlora, target=[...], rank=N)` config.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    pub kind: AdapterKind,
    pub targets: Vec<String>,
    /// Explicit rank override; if `None`, WRGA picks the rank via roofline
    /// + spectral analysis.
    pub rank: Option<i64>,
    /// LoRA scaling: `scale = alpha / rank`.  `None` → codegen uses
    /// `alpha = rank` (scale = 1.0).  Ignored for non-LoRA adapters.
    pub alpha: Option<i64>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    Lora,
    Ia3,
    GatedLora,
}

impl AdapterKind {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "lora" | "LoRA" => Some(AdapterKind::Lora),
            "ia3" | "IA3" | "IA\u{00B3}" => Some(AdapterKind::Ia3),
            "gatedlora" | "GatedLoRA" => Some(AdapterKind::GatedLora),
            _ => None,
        }
    }
}

pub fn validate_adapter_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<AdapterConfig> {
    let mut kind: Option<AdapterKind> = None;
    let mut targets: Vec<String> = Vec::new();
    let mut rank: Option<i64> = None;
    let mut alpha: Option<i64> = None;

    let Some(ref args) = deco.args else {
        diagnostics.push(
            Diagnostic::error(
                "@adapter requires at least a `type=...` argument".to_string(),
            )
            .with_label(deco.span, "missing arguments"),
        );
        return None;
    };
    for arg in args {
        let Some(ref name_sym) = arg.name else {
            diagnostics.push(
                Diagnostic::error("@adapter: positional arguments are not allowed".to_string())
                    .with_label(arg.span, "expected `key = value`"),
            );
            continue;
        };
        let aname = resolve_sym(*name_sym);
        match aname.as_str() {
            "type" => {
                let Some(sym) = extract_ident(arg) else {
                    diagnostics.push(
                        Diagnostic::error(
                            "@adapter: type must be an identifier (lora, ia3, gatedlora)"
                                .to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    );
                    continue;
                };
                let s = resolve_sym(sym);
                match AdapterKind::parse(&s) {
                    Some(k) => kind = Some(k),
                    None => diagnostics.push(
                        Diagnostic::error(format!(
                            "@adapter: unknown adapter type '{s}' (expected lora, ia3, gatedlora)"
                        ))
                        .with_label(arg.span, "unknown type"),
                    ),
                }
            }
            "target" => match &arg.value.kind {
                ExprKind::ListLiteral(items) => {
                    for item in items {
                        match &item.kind {
                            ExprKind::StringLiteral(s) => targets.push(s.clone()),
                            _ => diagnostics.push(
                                Diagnostic::error(
                                    "@adapter: target entries must be string literals"
                                        .to_string(),
                                )
                                .with_label(item.span, "expected string"),
                            ),
                        }
                    }
                }
                ExprKind::StringLiteral(s) => targets.push(s.clone()),
                _ => diagnostics.push(
                    Diagnostic::error(
                        "@adapter: target must be a string or list of strings".to_string(),
                    )
                    .with_label(arg.span, "expected list/string"),
                ),
            },
            "rank" => match &arg.value.kind {
                ExprKind::IntLiteral(n) => {
                    if *n <= 0 {
                        diagnostics.push(
                            Diagnostic::error("@adapter: rank must be positive".to_string())
                                .with_label(arg.span, "rank <= 0"),
                        );
                    } else {
                        rank = Some(*n);
                    }
                }
                _ => diagnostics.push(
                    Diagnostic::error("@adapter: rank must be an integer literal".to_string())
                        .with_label(arg.span, "expected integer"),
                ),
            },
            "alpha" => match &arg.value.kind {
                ExprKind::IntLiteral(n) => {
                    if *n <= 0 {
                        diagnostics.push(
                            Diagnostic::error("@adapter: alpha must be positive".to_string())
                                .with_label(arg.span, "alpha <= 0"),
                        );
                    } else {
                        alpha = Some(*n);
                    }
                }
                _ => diagnostics.push(
                    Diagnostic::error("@adapter: alpha must be an integer literal".to_string())
                        .with_label(arg.span, "expected integer"),
                ),
            },
            _ => diagnostics.push(
                Diagnostic::error(format!("@adapter: unknown argument '{aname}'"))
                    .with_label(arg.span, "unknown argument"),
            ),
        }
    }
    let kind = match kind {
        Some(k) => k,
        None => {
            diagnostics.push(
                Diagnostic::error(
                    "@adapter requires a `type=...` argument (lora, ia3, gatedlora)".to_string(),
                )
                .with_label(deco.span, "missing type"),
            );
            return None;
        }
    };
    if targets.is_empty() {
        diagnostics.push(
            Diagnostic::error(
                "@adapter requires a `target=[...]` list of parameter globs".to_string(),
            )
            .with_label(deco.span, "missing target"),
        );
    }
    if alpha.is_some() && !matches!(kind, AdapterKind::Lora) {
        diagnostics.push(
            Diagnostic::warning(
                "@adapter: alpha is only meaningful for LoRA; ignored for ia3/gatedlora"
                    .to_string(),
            )
            .with_label(deco.span, "alpha ignored"),
        );
    }
    Some(AdapterConfig {
        kind,
        targets,
        rank,
        alpha,
        span: deco.span,
    })
}

/// WRGA paper §8.2 — post-pass that validates the `adapter=<Ident>` argument
/// on every `@wrga(...)` decorator against the module's declared models.
///
/// Runs after `check_module` so all top-level `model` declarations have been
/// pre-declared (sub-pass 2 in `checker::collect_top_level_decls`).  Three
/// failure modes produce diagnostics:
///
/// 1. The named symbol resolves to neither a same-file `ModelDef` nor any
///    other in-scope name — likely a typo.
/// 2. The same-file model exists but has no `Tensor` field — WRGA needs at
///    least one trainable parameter to place.
/// 3. The same-file model exists but has no `forward` method — the @wrga
///    rewrite has no entry point to call into.
///
/// Cross-module imports are accepted on trust: if the adapter symbol is bound
/// somewhere in scope (via `from foo.peft import GatedLoRA`) but no matching
/// declaration of *any* kind is in this file's `stmts`, the post-pass skips
/// the model-shape checks rather than emitting a false "undeclared adapter"
/// error.  The codegen integration cycle (which has the full multi-file
/// analysis) will produce the deeper contract check.
///
/// That trust only extends to symbols this file genuinely can't see the
/// definition of.  A same-file declaration of the wrong kind (`struct`,
/// `fn`, `enum`, `trait`, `agent`) is not a cross-module import — this pass
/// has full access to its shape and must reject it outright, since only a
/// `model` can satisfy the adapter contract (`struct` in particular can
/// never declare a `forward` method at all).
///
/// Sub-passes only inspect surface signature shape — the heavy contract
/// check (forward takes a single `Tensor` arg and returns a `Tensor`) is
/// deferred to codegen integration, where the existing type-map can be
/// queried.  This post-pass catches the common-case mistakes (typo,
/// nonexistent model, empty model body) loudly and early.
pub fn validate_wrga_custom_adapters(
    configs: &[WrgaConfig],
    module: &Module,
    scopes: &crate::scope::ScopeMap,
    resolve_sym: &dyn Fn(Symbol) -> String,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    for cfg in configs {
        let Some(adapter_sym) = cfg.block.adapter else {
            continue;
        };
        let adapter_name = resolve_sym(adapter_sym);
        let model = module.stmts.iter().find_map(|s| match &s.kind {
            StmtKind::ModelDef(m) if m.name == adapter_sym => Some(m),
            _ => None,
        });
        let Some(model) = model else {
            // Same-file declaration of the wrong kind: not a cross-module
            // import, so we can see its shape directly and it definitively
            // fails the adapter contract (only `model` can have a `forward`
            // method or WRGA-placeable `Tensor` field).
            let wrong_kind = module.stmts.iter().find_map(|s| match &s.kind {
                StmtKind::StructDef(d) if d.name == adapter_sym => Some("struct"),
                StmtKind::FnDef(d) if d.name == adapter_sym => Some("fn"),
                StmtKind::EnumDef(d) if d.name == adapter_sym => Some("enum"),
                StmtKind::TraitDef(d) if d.name == adapter_sym => Some("trait"),
                StmtKind::AgentDef(d) if d.name == adapter_sym => Some("agent"),
                _ => None,
            });
            if let Some(kind) = wrong_kind {
                diags.push(
                    Diagnostic::error(format!(
                        "@wrga: custom adapter '{adapter_name}' is a `{kind}`, not a `model` \
                         (WRGA paper §8.2 requires `adapter={adapter_name}` to name a `model \
                         {adapter_name}(...)` with a `Tensor` field and a `forward` method — a \
                         `{kind}` cannot satisfy the adapter contract)"
                    ))
                    .with_label(cfg.block.span, "wrong declaration kind"),
                );
                continue;
            }
            // Fallback: an imported adapter (`from foo.peft import GatedLoRA`)
            // is bound in the module's root scope but never appears in
            // `module.stmts` at all.  Trust the import — the deeper
            // contract check happens at codegen.
            if scopes.lookup(crate::scope::ScopeId::ROOT, adapter_sym).is_some() {
                continue;
            }
            diags.push(
                Diagnostic::error(format!(
                    "@wrga: custom adapter '{adapter_name}' is not a declared model in this \
                     module and is not in scope (WRGA paper §8.2 requires either `model \
                     {adapter_name}(...)` to be defined here OR `from <pkg> import \
                     {adapter_name}` to be in scope before the `@wrga(adapter={adapter_name}, \
                     ...)` annotation)"
                ))
                .with_label(cfg.block.span, "undeclared adapter"),
            );
            continue;
        };
        let has_tensor_field = model.members.iter().any(|m| match m {
            ModelMember::LayerDecl { type_ann, .. } => is_tensor_type(&type_ann.kind, resolve_sym),
            _ => false,
        });
        if !has_tensor_field {
            diags.push(
                Diagnostic::error(format!(
                    "@wrga: custom adapter '{adapter_name}' has no `Tensor` field — WRGA \
                     needs at least one trainable adapter parameter to place at each site \
                     (paper §8.2)"
                ))
                .with_label(model.span, "no Tensor field"),
            );
        }
        let has_forward = model.members.iter().any(|m| {
            matches!(m, ModelMember::Method(fd, _) if resolve_sym(fd.name) == "forward")
        });
        if !has_forward {
            diags.push(
                Diagnostic::error(format!(
                    "@wrga: custom adapter '{adapter_name}' has no `forward` method — the \
                     WRGA rewrite needs `fn forward(self, x: Tensor) -> Tensor` to call into \
                     the adapter at each placement site (paper §8.2)"
                ))
                .with_label(model.span, "missing forward"),
            );
        }
    }
    diags
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// `true` if `kind` is a Tensor type annotation in any of NSL's surface
/// spellings: the dedicated `TypeExprKind::Tensor { ... }` (full
/// `Tensor<[shape], dtype, device>`), or the bare `Tensor` ident which
/// parses as `TypeExprKind::Named` with the symbol resolving to "Tensor".
fn is_tensor_type(kind: &TypeExprKind, resolve_sym: &dyn Fn(Symbol) -> String) -> bool {
    match kind {
        TypeExprKind::Tensor { .. } => true,
        TypeExprKind::Named(sym) => resolve_sym(*sym) == "Tensor",
        _ => false,
    }
}

fn extract_ident(arg: &Arg) -> Option<Symbol> {
    match &arg.value.kind {
        ExprKind::Ident(sym) => Some(*sym),
        _ => None,
    }
}

fn extract_mode_symbol(
    arg: &Arg,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<String> {
    match &arg.value.kind {
        ExprKind::Ident(sym) => Some(resolve_sym(*sym)),
        ExprKind::StringLiteral(s) => Some(s.clone()),
        _ => {
            diagnostics.push(
                Diagnostic::error(
                    "@wrga: mode must be an identifier or string (auto|manual|hybrid)"
                        .to_string(),
                )
                .with_label(arg.span, "expected ident or string"),
            );
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    // Validation here requires a fully-constructed `Decorator` AST node,
    // which in turn depends on the lexer's `Symbol` + `Span` machinery.  The
    // full test suite lives alongside the stmt checker; these unit tests just
    // exercise the enum/string plumbing.
    use super::*;

    #[test]
    fn adapter_kind_parses() {
        assert_eq!(AdapterKind::parse("lora"), Some(AdapterKind::Lora));
        assert_eq!(AdapterKind::parse("ia3"), Some(AdapterKind::Ia3));
        assert_eq!(AdapterKind::parse("gatedlora"), Some(AdapterKind::GatedLora));
        assert_eq!(AdapterKind::parse("unknown"), None);
    }

    #[test]
    fn wrga_mode_roundtrip() {
        for m in [WrgaMode::Auto, WrgaMode::Manual, WrgaMode::Hybrid] {
            assert_eq!(WrgaMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(WrgaMode::parse("nonsense"), None);
    }
}
