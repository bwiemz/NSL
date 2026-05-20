//! Semantic validation for the `@cep_prune` and `@cep_search` decorators.
//!
//! `@cep_prune(target, sparsity, constraints={...}, granularity, preserve=[...])`
//!     — attach to a `let model = ...` binding to request compilation-
//!     verified pruning.
//!
//! `@cep_search(target, constraints={...}, objective)` — attach to a
//!     model block with `@search(name, [values])` annotations inside.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    Head,
    Ffn,
    HeadAndFfn,
}

impl Granularity {
    pub fn as_str(self) -> &'static str {
        match self {
            Granularity::Head => "head",
            Granularity::Ffn => "ffn",
            Granularity::HeadAndFfn => "head+ffn",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "head" | "heads" => Some(Granularity::Head),
            "ffn" => Some(Granularity::Ffn),
            "head+ffn" | "both" => Some(Granularity::HeadAndFfn),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CepPruneConfig {
    pub target: Option<Symbol>,
    pub sparsity: f64,
    pub granularity: Granularity,
    pub preserve: Vec<String>,
    pub span: Span,
}

pub fn validate_cep_prune_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CepPruneConfig> {
    let mut target: Option<Symbol> = None;
    let mut sparsity: f64 = 0.3;
    let mut granularity = Granularity::HeadAndFfn;
    let mut preserve: Vec<String> = Vec::new();

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error(
                        "@cep_prune: positional arguments are not allowed".to_string(),
                    )
                    .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "target" => match &arg.value.kind {
                    ExprKind::Ident(sym) => target = Some(*sym),
                    ExprKind::StringLiteral(_) => target = Some(*name_sym),
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@cep_prune: target must be an identifier (e.g. h100)".to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    ),
                },
                "sparsity" => match &arg.value.kind {
                    ExprKind::FloatLiteral(f) => {
                        if *f < 0.0 || *f > 1.0 {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@cep_prune: sparsity must be in [0, 1]".to_string(),
                                )
                                .with_label(arg.span, "out of range"),
                            );
                        } else {
                            sparsity = *f;
                        }
                    }
                    ExprKind::IntLiteral(n) => sparsity = *n as f64,
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@cep_prune: sparsity must be a numeric literal".to_string(),
                        )
                        .with_label(arg.span, "expected number"),
                    ),
                },
                "granularity" => {
                    let s = match &arg.value.kind {
                        ExprKind::Ident(sym) => Some(resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => Some(s.clone()),
                        _ => None,
                    };
                    match s.as_deref().and_then(Granularity::parse) {
                        Some(g) => granularity = g,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@cep_prune: granularity must be head, ffn, or head+ffn"
                                    .to_string(),
                            )
                            .with_label(arg.span, "invalid granularity"),
                        ),
                    }
                }
                "preserve" => match &arg.value.kind {
                    ExprKind::ListLiteral(items) => {
                        for item in items {
                            match &item.kind {
                                ExprKind::StringLiteral(s) => preserve.push(s.clone()),
                                _ => diagnostics.push(
                                    Diagnostic::error(
                                        "@cep_prune: preserve entries must be strings".to_string(),
                                    )
                                    .with_label(item.span, "expected string"),
                                ),
                            }
                        }
                    }
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@cep_prune: preserve must be a list of glob strings".to_string(),
                        )
                        .with_label(arg.span, "expected list"),
                    ),
                },
                "constraints" => {
                    // Accept but don't validate deeply here — constraints
                    // (peak_memory / latency_per_token) are parsed at
                    // codegen time where they have meaning.
                }
                _ => diagnostics.push(
                    Diagnostic::error(format!("@cep_prune: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(CepPruneConfig {
        target,
        sparsity,
        granularity,
        preserve,
        span: deco.span,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NasObjective {
    ParamEfficiency,
    MinLatency,
    MinMemory,
    MinParams,
}

impl NasObjective {
    pub fn as_str(self) -> &'static str {
        match self {
            NasObjective::ParamEfficiency => "param_efficiency",
            NasObjective::MinLatency => "min_latency",
            NasObjective::MinMemory => "min_memory",
            NasObjective::MinParams => "min_params",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "param_efficiency" | "maximize param_efficiency" | "params_per_us" => {
                Some(NasObjective::ParamEfficiency)
            }
            "min_latency" | "minimize_latency" => Some(NasObjective::MinLatency),
            "min_memory" | "minimize_memory" => Some(NasObjective::MinMemory),
            "min_params" | "minimize_params" => Some(NasObjective::MinParams),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CepSearchConfig {
    pub target: Option<Symbol>,
    pub objective: NasObjective,
    pub span: Span,
}

pub fn validate_cep_search_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CepSearchConfig> {
    let mut target: Option<Symbol> = None;
    let mut objective = NasObjective::ParamEfficiency;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error(
                        "@cep_search: positional arguments are not allowed".to_string(),
                    )
                    .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "target" => match &arg.value.kind {
                    ExprKind::Ident(sym) => target = Some(*sym),
                    ExprKind::StringLiteral(_) => target = Some(*name_sym),
                    _ => diagnostics.push(
                        Diagnostic::error("@cep_search: target must be an identifier".to_string())
                            .with_label(arg.span, "expected ident"),
                    ),
                },
                "objective" => {
                    let s = match &arg.value.kind {
                        ExprKind::Ident(sym) => Some(resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => Some(s.clone()),
                        _ => None,
                    };
                    match s.as_deref().and_then(NasObjective::parse) {
                        Some(o) => objective = o,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@cep_search: objective must be param_efficiency, \
                                 min_latency, min_memory, or min_params"
                                    .to_string(),
                            )
                            .with_label(arg.span, "invalid objective"),
                        ),
                    }
                }
                "constraints" => {
                    // As with @cep_prune, parsed later.
                }
                _ => diagnostics.push(
                    Diagnostic::error(format!("@cep_search: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(CepSearchConfig {
        target,
        objective,
        span: deco.span,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn granularity_parse_roundtrip() {
        for g in [Granularity::Head, Granularity::Ffn, Granularity::HeadAndFfn] {
            assert_eq!(Granularity::parse(g.as_str()), Some(g));
        }
        assert_eq!(Granularity::parse("heads"), Some(Granularity::Head));
        assert!(Granularity::parse("bogus").is_none());
    }

    #[test]
    fn nas_objective_parse_roundtrip() {
        for o in [
            NasObjective::ParamEfficiency,
            NasObjective::MinLatency,
            NasObjective::MinMemory,
            NasObjective::MinParams,
        ] {
            assert_eq!(NasObjective::parse(o.as_str()), Some(o));
        }
        assert!(NasObjective::parse("bogus").is_none());
    }
}
