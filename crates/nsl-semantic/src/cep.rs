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

/// Structured optimization constraints parsed from `constraints={...}`.
/// A `None` field means "unconstrained"; the codegen bridge (added in a later
/// task) will map it to the permissive default (`u64::MAX` / `f64::INFINITY`).
#[derive(Debug, Clone, Default)]
pub struct CepConstraints {
    pub peak_memory_bytes: Option<u64>,
    pub latency_us: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CepPruneConfig {
    pub target: Option<Symbol>,
    pub sparsity: f64,
    pub granularity: Granularity,
    pub preserve: Vec<String>,
    pub constraints: CepConstraints,
    pub span: Span,
}

fn parse_cep_constraints(
    arg: &nsl_ast::expr::Arg,
    deco_label: &str,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> CepConstraints {
    let mut constraints = CepConstraints::default();
    match &arg.value.kind {
        ExprKind::DictLiteral(pairs) => {
            for (k, v) in pairs {
                let key = match &k.kind {
                    ExprKind::Ident(sym) => resolve_sym(*sym),
                    ExprKind::StringLiteral(s) => s.clone(),
                    _ => continue,
                };
                match key.as_str() {
                    "peak_memory_bytes" => match &v.kind {
                        ExprKind::IntLiteral(n) if *n >= 0 => {
                            constraints.peak_memory_bytes = Some(*n as u64)
                        }
                        _ => diagnostics.push(
                            Diagnostic::error(format!(
                                "{deco_label}: peak_memory_bytes must be a non-negative integer (bytes)"
                            ))
                            .with_label(v.span, "expected integer bytes"),
                        ),
                    },
                    "latency_us" => match &v.kind {
                        ExprKind::FloatLiteral(f) => constraints.latency_us = Some(*f),
                        ExprKind::IntLiteral(n) => constraints.latency_us = Some(*n as f64),
                        _ => diagnostics.push(
                            Diagnostic::error(format!(
                                "{deco_label}: latency_us must be a number (microseconds)"
                            ))
                            .with_label(v.span, "expected number"),
                        ),
                    },
                    other => diagnostics.push(
                        Diagnostic::error(format!(
                            "{deco_label}: unknown constraint '{other}' (expected peak_memory_bytes or latency_us)"
                        ))
                        .with_label(k.span, "unknown constraint"),
                    ),
                }
            }
        }
        _ => diagnostics.push(
            Diagnostic::error(format!(
                "{deco_label}: constraints must be a dict, e.g. {{peak_memory_bytes: 6000000000, latency_us: 3000.0}}"
            ))
            .with_label(arg.span, "expected dict"),
        ),
    }
    constraints
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
    let mut constraints = CepConstraints::default();

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
                                        "@cep_prune: preserve entries must be strings"
                                            .to_string(),
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
                    constraints = parse_cep_constraints(arg, "@cep_prune", resolve_sym, diagnostics);
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
        constraints,
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
    pub constraints: CepConstraints,
    pub span: Span,
}

pub fn validate_cep_search_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CepSearchConfig> {
    let mut target: Option<Symbol> = None;
    let mut objective = NasObjective::ParamEfficiency;
    let mut constraints = CepConstraints::default();

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
                        Diagnostic::error(
                            "@cep_search: target must be an identifier".to_string(),
                        )
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
                    constraints = parse_cep_constraints(arg, "@cep_search", resolve_sym, diagnostics);
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
        constraints,
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

    /// Verify that `@cep_prune(constraints={peak_memory_bytes: 6000000000, latency_us: 3000.0})`
    /// parses into the correct `CepConstraints` fields with zero diagnostics.
    #[test]
    fn cep_prune_constraints_dict_parsed() {
        // Parse a minimal NSL snippet carrying the decorator we want to test.
        // The model body only needs to be syntactically valid; we only care
        // about the decorator that wraps it.
        let src = "\
@cep_prune(constraints={peak_memory_bytes: 6000000000, latency_us: 3000.0})\n\
model M:\n    fn forward(self) -> f32:\n        return 0.0\n";

        let mut interner = nsl_lexer::Interner::new();
        let (tokens, lex_diags) =
            nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        assert!(lex_diags.is_empty(), "lex errors: {lex_diags:?}");

        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parse_result.diagnostics.is_empty(),
            "parse errors: {:?}",
            parse_result.diagnostics
        );

        // Pull the single decorator out of the Decorated statement.
        let stmt = parse_result.module.stmts.first().expect("expected one stmt");
        let nsl_ast::stmt::StmtKind::Decorated { ref decorators, .. } = stmt.kind else {
            panic!("expected Decorated stmt, got {:?}", stmt.kind);
        };
        let deco = decorators.first().expect("expected one decorator");

        let resolve_sym = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let mut diagnostics = Vec::new();
        let cfg = validate_cep_prune_decorator(deco, &resolve_sym, &mut diagnostics)
            .expect("validate_cep_prune_decorator returned None");

        assert!(
            diagnostics.is_empty(),
            "unexpected diagnostics: {diagnostics:?}"
        );
        assert_eq!(
            cfg.constraints.peak_memory_bytes,
            Some(6_000_000_000u64),
            "peak_memory_bytes mismatch"
        );
        assert_eq!(
            cfg.constraints.latency_us,
            Some(3000.0f64),
            "latency_us mismatch"
        );
    }

    /// Verify that `@cep_prune(constraints=5)` (a non-dict value) produces a
    /// diagnostic whose message mentions "must be a dict".
    #[test]
    fn cep_prune_constraints_non_dict_is_error() {
        let src = "\
@cep_prune(constraints=5)\n\
model M:\n    fn forward(self) -> f32:\n        return 0.0\n";

        let mut interner = nsl_lexer::Interner::new();
        let (tokens, lex_diags) =
            nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
        assert!(lex_diags.is_empty(), "lex errors: {lex_diags:?}");

        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parse_result.diagnostics.is_empty(),
            "parse errors: {:?}",
            parse_result.diagnostics
        );

        let stmt = parse_result.module.stmts.first().expect("expected one stmt");
        let nsl_ast::stmt::StmtKind::Decorated { ref decorators, .. } = stmt.kind else {
            panic!("expected Decorated stmt, got {:?}", stmt.kind);
        };
        let deco = decorators.first().expect("expected one decorator");

        let resolve_sym = |s: Symbol| interner.resolve(s.0).unwrap_or("").to_string();
        let mut diagnostics = Vec::new();
        let _cfg = validate_cep_prune_decorator(deco, &resolve_sym, &mut diagnostics);

        assert!(
            !diagnostics.is_empty(),
            "expected a diagnostic for non-dict constraints, but got none"
        );
        let any_mentions_dict = diagnostics
            .iter()
            .any(|d| d.message.contains("must be a dict"));
        assert!(
            any_mentions_dict,
            "expected a diagnostic mentioning 'must be a dict', got: {diagnostics:?}"
        );
    }
}
