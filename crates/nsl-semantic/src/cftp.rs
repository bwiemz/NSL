//! Semantic validation for CFTP training-pipeline optimisations.
//!
//! Two feature surfaces:
//!
//!   * `@fase(...)` — optional decorator on the `train` block that lets the
//!     user pin a specific FASE mode (`auto` / `deferred` / `full_buffer`
//!     / `off`) and explicitly toggle the AdamW v_t approximation.
//!   * `@pca(...)` — optional decorator on the `dataset` or `train` block
//!     that forces a specific PCA strategy (`auto` / `segment_id` /
//!     `per_document` / `off`).
//!
//! When no decorator is present, the compiler applies FASE/PCA
//! automatically based on the other configuration (grad_accumulation,
//! packing, etc.) — this mirrors the design intent in Section 7 of the
//! CFTP paper.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaseMode {
    Auto,
    Deferred,
    FullBuffer,
    Off,
}

impl FaseMode {
    pub fn as_str(self) -> &'static str {
        match self {
            FaseMode::Auto => "auto",
            FaseMode::Deferred => "deferred",
            FaseMode::FullBuffer => "full_buffer",
            FaseMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(FaseMode::Auto),
            "deferred" => Some(FaseMode::Deferred),
            "full_buffer" | "full" => Some(FaseMode::FullBuffer),
            "off" | "disable" | "disabled" => Some(FaseMode::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaseConfig {
    pub mode: FaseMode,
    pub allow_v_approx: bool,
    pub span: Span,
}

pub fn validate_fase_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<FaseConfig> {
    let mut mode = FaseMode::Auto;
    let mut allow_v_approx = true;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error("@fase: positional arguments are not allowed".to_string())
                        .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => FaseMode::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => FaseMode::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(m) => mode = m,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@fase: mode must be auto, deferred, full_buffer, or off"
                                    .to_string(),
                            )
                            .with_label(arg.span, "invalid mode"),
                        ),
                    }
                }
                "allow_v_approx" | "v_approx" => match &arg.value.kind {
                    ExprKind::BoolLiteral(b) => allow_v_approx = *b,
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@fase: allow_v_approx must be a bool literal".to_string(),
                        )
                        .with_label(arg.span, "expected bool"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@fase: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(FaseConfig {
        mode,
        allow_v_approx,
        span: deco.span,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaStrategy {
    Auto,
    SegmentId,
    PerDocument,
    Off,
}

impl PcaStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            PcaStrategy::Auto => "auto",
            PcaStrategy::SegmentId => "segment_id",
            PcaStrategy::PerDocument => "per_document",
            PcaStrategy::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(PcaStrategy::Auto),
            "segment_id" | "segment" => Some(PcaStrategy::SegmentId),
            "per_document" | "per_doc" | "perdoc" => Some(PcaStrategy::PerDocument),
            "off" | "disable" | "disabled" => Some(PcaStrategy::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PcaConfig {
    pub strategy: PcaStrategy,
    pub span: Span,
}

/// Parse + validate the `@pca(strategy=...)` decorator on a train block.
///
/// # Sprint 1 status (CFTP v2 follow-on)
///
/// As of Sprint 1 (commit `c80312dc`), the returned `PcaConfig` is
/// **dropped by the caller** in `checker/stmt.rs::stmt_decorator` — see
/// the matching block for `@wrga`, which pushes onto `self.wrga_configs`.
/// `@pca` has no analogous collection list yet.
///
/// Plumbing `PcaConfig::strategy = PerDocument` through to
/// `nsl_codegen::pca_per_doc::PerDocAdmitConfig::enable_per_doc_cta` is
/// Sprint 2 follow-on work. The per-doc CTA emitter
/// (`nsl_codegen::flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_forward`)
/// and the FFI dispatch (`nsl_runtime::flash_attention::nsl_flash_attention_csha`)
/// are already in place to receive that decision.
///
/// Until decorator collection lands, the per-doc CTA path is reachable
/// only from tests that hardcode `enable_per_doc_cta=true` on the
/// admission config.
pub fn validate_pca_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<PcaConfig> {
    let mut strategy = PcaStrategy::Auto;
    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error("@pca: positional arguments are not allowed".to_string())
                        .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "strategy" | "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => PcaStrategy::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => PcaStrategy::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(s) => strategy = s,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@pca: strategy must be auto, segment_id, per_document, or off"
                                    .to_string(),
                            )
                            .with_label(arg.span, "invalid strategy"),
                        ),
                    }
                }
                _ => diagnostics.push(
                    Diagnostic::error(format!("@pca: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }
    Some(PcaConfig {
        strategy,
        span: deco.span,
    })
}

// ---------------------------------------------------------------------------
// G3: @fused_lm_ce decorator
// ---------------------------------------------------------------------------

/// Configuration extracted from a `@fused_lm_ce(...)` decorator on a `train`
/// block.  Parsed by [`validate_fused_ce_decorator`].
///
/// v1 is OPT-IN ONLY.  No automatic substitution of `cross_entropy` happens
/// unless this config is present with `enabled = true`.
#[derive(Debug, Clone)]
pub struct FusedCeConfig {
    /// Whether the fused kernel is enabled for this train block.
    pub enabled: bool,
    /// Vocabulary tile size passed through to `FusedLinearCEConfig::vocab_tile`.
    /// `None` → use the codegen default (1024).
    pub vocab_tile: Option<u32>,
    /// Source span of the decorator (for diagnostics).
    pub span: Span,
}

/// Parse and validate a `@fused_lm_ce(enabled = true, vocab_tile = 1024)`
/// decorator.
///
/// Mirrors the shape of [`validate_fase_decorator`] / [`validate_pca_decorator`].
pub fn validate_fused_ce_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<FusedCeConfig> {
    let mut enabled = false;
    let mut vocab_tile: Option<u32> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error(
                        "@fused_lm_ce: positional arguments are not allowed".to_string(),
                    )
                    .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "enabled" => match &arg.value.kind {
                    ExprKind::BoolLiteral(b) => enabled = *b,
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@fused_lm_ce: `enabled` must be a bool literal".to_string(),
                        )
                        .with_label(arg.span, "expected bool"),
                    ),
                },
                "vocab_tile" => match &arg.value.kind {
                    ExprKind::IntLiteral(n) => {
                        if *n > 0 && *n <= 8192 && (n % 128 == 0) {
                            vocab_tile = Some(*n as u32);
                        } else if *n > 0 && *n <= 8192 {
                            // Same invariant as FusedLinearCEConfig::validate:
                            // the v1 kernel's inner fill is 128-thread-wide;
                            // a non-128-aligned tile leaves the tail
                            // uninitialised in smem and silently corrupts the
                            // online-softmax reduction (see fused_linear_ce.rs).
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_lm_ce: vocab_tile {n} must be a multiple of 128 \
                                     (the v1 inner fill is 128-thread-wide; non-128-aligned \
                                     tiles silently corrupt the online-softmax reduction)"
                                ))
                                .with_label(arg.span, "must be a multiple of 128"),
                            );
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_lm_ce: vocab_tile {n} out of range [1, 8192]"
                                ))
                                .with_label(arg.span, "invalid vocab_tile"),
                            );
                        }
                    }
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@fused_lm_ce: `vocab_tile` must be an integer literal".to_string(),
                        )
                        .with_label(arg.span, "expected integer"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@fused_lm_ce: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(FusedCeConfig {
        enabled,
        vocab_tile,
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
    fn fase_mode_roundtrip() {
        for m in [
            FaseMode::Auto,
            FaseMode::Deferred,
            FaseMode::FullBuffer,
            FaseMode::Off,
        ] {
            assert_eq!(FaseMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(FaseMode::parse("nonsense"), None);
    }

    #[test]
    fn pca_strategy_roundtrip() {
        for s in [
            PcaStrategy::Auto,
            PcaStrategy::SegmentId,
            PcaStrategy::PerDocument,
            PcaStrategy::Off,
        ] {
            assert_eq!(PcaStrategy::parse(s.as_str()), Some(s));
        }
        assert_eq!(PcaStrategy::parse("nonsense"), None);
    }
}
