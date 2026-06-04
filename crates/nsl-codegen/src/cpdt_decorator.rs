//! CPDT §6.1 — `@cpdt(...)` train-block decorator wiring.
//!
//! Paper §6.1 example:
//! ```text
//! @cpdt(
//!     cluster = { gpus = 8, intra_bw = "900GB/s", inter_bw = "100GB/s" },
//!     target_memory = "80%",
//!     precision = auto
//! )
//! train(model=m, epochs=1, grad_clip=1.0):
//!     optimizer: AdamW(lr=0.0003, weight_decay=0.1)
//!     step(batch): ...
//! ```
//!
//! `nsl-semantic` already validates the decorator's syntax (`validate_cpdt_decorator`).
//! This module is the wiring layer: walk the module-level statements looking for
//! `@cpdt(...)` and write the resolved config into `compiler.cpdt_mode` /
//! `compiler.cpdt_cluster`. The decorator OVERRIDES any CLI flag values (paper's
//! semantics: the source-level decorator is authoritative when present).
//!
//! This is a strict-additive pass: when the decorator is absent, the CLI-derived
//! configuration is left intact. The pass is independent of WGGO / source-AD
//! (mirrors the prune + capacity passes' blocker-sidestep posture).

use nsl_ast::{Module, Span};
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;
use nsl_semantic::cpdt::{
    validate_cpdt_decorator, CpdtConfig, CpdtMode as SemCpdtMode,
};

use crate::cpdt::CpdtMode as CgCpdtMode;
use crate::cpdt_zero::ClusterSpec;

/// Outcome of the decorator pass — used for testing + diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub enum CpdtDecoratorOutcome {
    /// No `@cpdt` decorator found in the module.
    Absent,
    /// Found and applied. Reports the mode and whether the cluster was set.
    Applied { mode: CgCpdtMode, cluster_set: bool, span: Span },
    /// Found but the parser produced diagnostics — leave compiler state untouched.
    Invalid { span: Span },
}

/// Translate the semantic-side mode to the codegen-side mode.
fn map_mode(m: SemCpdtMode) -> CgCpdtMode {
    match m {
        SemCpdtMode::Full => CgCpdtMode::Full,
        SemCpdtMode::ZeroOnly => CgCpdtMode::ZeroOnly,
        SemCpdtMode::Off => CgCpdtMode::Off,
    }
}

/// Build a codegen `ClusterSpec` from the semantic-side `CpdtClusterSpec`,
/// filling in defaults for any field the decorator omitted.
fn build_cluster(cfg: &CpdtConfig, base: &Option<ClusterSpec>) -> Option<ClusterSpec> {
    let any_field_set = cfg.cluster.gpus.is_some()
        || cfg.cluster.intra_bw_bps.is_some()
        || cfg.cluster.inter_bw_bps.is_some()
        || cfg.cluster.gpus_per_node.is_some();
    if !any_field_set {
        return base.clone();
    }
    let defaults = base.clone().unwrap_or_default();
    Some(ClusterSpec {
        num_gpus: cfg.cluster.gpus.map(|n| n as u32).unwrap_or(defaults.num_gpus),
        memory_budget_bytes: defaults.memory_budget_bytes,
        intra_bw_bps: cfg.cluster.intra_bw_bps.unwrap_or(defaults.intra_bw_bps),
        inter_bw_bps: cfg.cluster.inter_bw_bps.unwrap_or(defaults.inter_bw_bps),
        gpus_per_node: cfg.cluster.gpus_per_node.map(|n| n as u32).unwrap_or(defaults.gpus_per_node),
    })
}

/// Walk top-level statements looking for a `@cpdt(...)` decorator on a `train`
/// block. Returns the first one found (Phase 1 already enforces at-most-one).
fn find_cpdt_on_train(ast: &Module, interner: &Interner) -> Option<(nsl_ast::decl::Decorator, Span)> {
    for stmt in &ast.stmts {
        let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind else {
            continue;
        };
        // The decorator only applies if it wraps a TrainBlock (paper §6.1).
        let StmtKind::TrainBlock(_) = &inner.kind else {
            continue;
        };
        for deco in decorators {
            if deco.name.is_empty() {
                continue;
            }
            let name = interner.resolve(deco.name[0].0).unwrap_or("");
            if name == "cpdt" {
                return Some((deco.clone(), deco.span));
            }
        }
    }
    None
}

/// Decorator pass entry point.
pub fn apply_cpdt_decorator_from_ast(
    ast: &Module,
    interner: &Interner,
    compiler: &mut crate::compiler::Compiler,
) -> CpdtDecoratorOutcome {
    let Some((deco, span)) = find_cpdt_on_train(ast, interner) else {
        return CpdtDecoratorOutcome::Absent;
    };

    let resolve = |s: nsl_ast::Symbol| -> String {
        interner.resolve(s.0).unwrap_or("").to_string()
    };
    let mut diagnostics = Vec::new();
    let cfg = validate_cpdt_decorator(&deco, &resolve, &mut diagnostics);
    if !diagnostics.is_empty() {
        // nsl-semantic already surfaced these diagnostics during type-check;
        // leave the compiler state untouched rather than apply a half-parsed config.
        return CpdtDecoratorOutcome::Invalid { span };
    }
    let Some(cfg) = cfg else {
        return CpdtDecoratorOutcome::Invalid { span };
    };

    // Override the mode (decorator wins over CLI).
    let new_mode = map_mode(cfg.mode);
    compiler.cpdt_mode = new_mode;

    // Build / merge cluster spec.
    let new_cluster = build_cluster(&cfg, &compiler.cpdt_cluster);
    let cluster_set = new_cluster.is_some();
    compiler.cpdt_cluster = new_cluster;

    CpdtDecoratorOutcome::Applied { mode: new_mode, cluster_set, span }
}

/// Render the outcome to stderr (matches the other CPDT passes' reporting).
pub fn report_outcome(outcome: &CpdtDecoratorOutcome) {
    match outcome {
        CpdtDecoratorOutcome::Absent => {}
        CpdtDecoratorOutcome::Applied { mode, cluster_set, .. } => {
            eprintln!(
                "[cpdt] @cpdt decorator applied: mode={} cluster={} (§6.1)",
                mode.as_str(),
                if *cluster_set { "set" } else { "inherited" }
            );
        }
        CpdtDecoratorOutcome::Invalid { .. } => {
            eprintln!("[cpdt] @cpdt decorator invalid; compiler state untouched (§6.1)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_semantic::cpdt::{CpdtClusterSpec, PrecisionMode};

    fn span() -> Span {
        Span::dummy()
    }

    fn make_cfg(mode: SemCpdtMode, cluster: CpdtClusterSpec) -> CpdtConfig {
        CpdtConfig {
            mode,
            cluster,
            target_memory_fraction: 0.8,
            precision: PrecisionMode::Auto,
            weight_aware: true,
            span: span(),
        }
    }

    #[test]
    fn map_mode_preserves_value() {
        assert_eq!(map_mode(SemCpdtMode::Full), CgCpdtMode::Full);
        assert_eq!(map_mode(SemCpdtMode::ZeroOnly), CgCpdtMode::ZeroOnly);
        assert_eq!(map_mode(SemCpdtMode::Off), CgCpdtMode::Off);
    }

    #[test]
    fn build_cluster_inherits_base_when_decorator_silent() {
        let cfg = make_cfg(SemCpdtMode::Full, CpdtClusterSpec::default());
        let base = Some(ClusterSpec {
            num_gpus: 4,
            memory_budget_bytes: 99,
            intra_bw_bps: 1e10,
            inter_bw_bps: 2e10,
            gpus_per_node: 4,
        });
        let result = build_cluster(&cfg, &base).expect("inherits base");
        assert_eq!(result.num_gpus, 4);
        assert_eq!(result.intra_bw_bps, 1e10);
    }

    #[test]
    fn build_cluster_returns_none_when_decorator_silent_and_no_base() {
        let cfg = make_cfg(SemCpdtMode::Full, CpdtClusterSpec::default());
        assert!(build_cluster(&cfg, &None).is_none());
    }

    #[test]
    fn build_cluster_overrides_explicit_fields() {
        let mut cluster = CpdtClusterSpec::default();
        cluster.gpus = Some(8);
        cluster.intra_bw_bps = Some(9e11);
        let cfg = make_cfg(SemCpdtMode::Full, cluster);
        // No base -> uses defaults for unspecified fields.
        let result = build_cluster(&cfg, &None).expect("decorator-only cluster");
        assert_eq!(result.num_gpus, 8);
        assert_eq!(result.intra_bw_bps, 9e11);
        // Unspecified field inter_bw_bps falls back to ClusterSpec::default().
        assert_eq!(result.inter_bw_bps, ClusterSpec::default().inter_bw_bps);
    }

    #[test]
    fn build_cluster_decorator_wins_over_base() {
        let mut cluster = CpdtClusterSpec::default();
        cluster.gpus = Some(16);
        let cfg = make_cfg(SemCpdtMode::Full, cluster);
        let base = Some(ClusterSpec {
            num_gpus: 4,
            memory_budget_bytes: 99,
            intra_bw_bps: 1e10,
            inter_bw_bps: 2e10,
            gpus_per_node: 4,
        });
        let result = build_cluster(&cfg, &base).expect("override");
        assert_eq!(result.num_gpus, 16); // decorator-overridden
        assert_eq!(result.intra_bw_bps, 1e10); // base-inherited
    }
}
