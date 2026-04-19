//! Semantic validation for the `@cpdt` decorator.
//!
//! Accepted arguments:
//!   * `mode = full | zero_only | off`
//!   * `cluster = { gpus, intra_bw, inter_bw, gpus_per_node }`
//!   * `target_memory = <float>`  — fraction of per-GPU memory to use
//!   * `precision = auto | fp32 | mixed`
//!
//! When absent, CPDT activates automatically if `grad_accumulation > 1`
//! and the cluster spec is non-trivial.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpdtMode {
    Full,
    ZeroOnly,
    Off,
}

impl CpdtMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CpdtMode::Full => "full",
            CpdtMode::ZeroOnly => "zero_only",
            CpdtMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" | "auto" => Some(CpdtMode::Full),
            "zero_only" | "zero" => Some(CpdtMode::ZeroOnly),
            "off" | "disable" | "disabled" => Some(CpdtMode::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionMode {
    Auto,
    Fp32,
    Mixed,
}

impl PrecisionMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(PrecisionMode::Auto),
            "fp32" => Some(PrecisionMode::Fp32),
            "mixed" => Some(PrecisionMode::Mixed),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            PrecisionMode::Auto => "auto",
            PrecisionMode::Fp32 => "fp32",
            PrecisionMode::Mixed => "mixed",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CpdtClusterSpec {
    pub gpus: Option<i64>,
    pub intra_bw_bps: Option<f64>,
    pub inter_bw_bps: Option<f64>,
    pub gpus_per_node: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct CpdtConfig {
    pub mode: CpdtMode,
    pub cluster: CpdtClusterSpec,
    pub target_memory_fraction: f64,
    pub precision: PrecisionMode,
    /// When `false`, CPDT skips the sensitivity scorer entirely and falls
    /// back to heuristic-only tier assignment (equivalent to pre-weight-
    /// aware behavior). Default: `true`. Exposed via `@cpdt(weight_aware=false)`.
    pub weight_aware: bool,
    pub span: Span,
}

/// Parse a bandwidth string like "900GB/s" or "100GB/s" into bytes-per-second.
fn parse_bandwidth(s: &str) -> Option<f64> {
    let trimmed = s.trim();
    let (num_part, suffix) = split_bandwidth_suffix(trimmed)?;
    let value: f64 = num_part.trim().parse().ok()?;
    let multiplier = match suffix.to_ascii_lowercase().as_str() {
        "gb/s" | "gbs" => 1e9,
        "tb/s" | "tbs" => 1e12,
        "mb/s" | "mbs" => 1e6,
        "b/s" | "bs" => 1.0,
        _ => return None,
    };
    Some(value * multiplier)
}

fn split_bandwidth_suffix(s: &str) -> Option<(String, String)> {
    // Split at the first non-digit / non-dot character.
    let split_at = s.chars().position(|c| !(c.is_ascii_digit() || c == '.'))?;
    Some((s[..split_at].to_string(), s[split_at..].to_string()))
}

pub fn validate_cpdt_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CpdtConfig> {
    let mut mode = CpdtMode::Full;
    let mut cluster = CpdtClusterSpec::default();
    let mut target_memory = 0.8;
    let mut precision = PrecisionMode::Auto;
    let mut weight_aware = true;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error("@cpdt: positional arguments are not allowed".to_string())
                        .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => CpdtMode::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => CpdtMode::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(m) => mode = m,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@cpdt: mode must be full, zero_only, or off".to_string(),
                            )
                            .with_label(arg.span, "invalid mode"),
                        ),
                    }
                }
                "precision" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => PrecisionMode::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => PrecisionMode::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(p) => precision = p,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@cpdt: precision must be auto, fp32, or mixed".to_string(),
                            )
                            .with_label(arg.span, "invalid precision"),
                        ),
                    }
                }
                "target_memory" => match &arg.value.kind {
                    ExprKind::FloatLiteral(f) => {
                        if *f <= 0.0 || *f > 1.0 {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@cpdt: target_memory must be in (0, 1]".to_string(),
                                )
                                .with_label(arg.span, "out of range"),
                            );
                        } else {
                            target_memory = *f;
                        }
                    }
                    ExprKind::StringLiteral(s) => {
                        // Accept "80%" shorthand.
                        let trimmed = s.trim_end_matches('%');
                        match trimmed.parse::<f64>() {
                            Ok(v) if (0.0..=100.0).contains(&v) => {
                                target_memory = v / 100.0;
                            }
                            _ => diagnostics.push(
                                Diagnostic::error(
                                    "@cpdt: target_memory must be a percentage or fraction"
                                        .to_string(),
                                )
                                .with_label(arg.span, "invalid value"),
                            ),
                        }
                    }
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@cpdt: target_memory must be a float or percentage string"
                                .to_string(),
                        )
                        .with_label(arg.span, "expected number"),
                    ),
                },
                "cluster" => {
                    // Cluster accepts a nested struct literal.  We try to
                    // decode whatever the user provided into the
                    // optional cluster fields; unknown sub-keys are
                    // warned about but don't abort.
                    decode_cluster_arg(&arg.value.kind, resolve_sym, &mut cluster, diagnostics, arg.span);
                }
                "weight_aware" => match &arg.value.kind {
                    ExprKind::BoolLiteral(b) => weight_aware = *b,
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@cpdt: weight_aware must be a bool literal".to_string(),
                        )
                        .with_label(arg.span, "expected true or false"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@cpdt: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(CpdtConfig {
        mode,
        cluster,
        target_memory_fraction: target_memory,
        precision,
        weight_aware,
        span: deco.span,
    })
}

fn decode_cluster_arg(
    kind: &ExprKind,
    resolve_sym: &dyn Fn(Symbol) -> String,
    cluster: &mut CpdtClusterSpec,
    diagnostics: &mut Vec<Diagnostic>,
    span: Span,
) {
    // Two accepted shapes:
    //   * `{ gpus = 8, intra_bw = "900GB/s", ... }`  (dict-style)
    //   * `cluster()` call with kwargs (handled upstream by parser)
    //
    // The AST currently represents dict-style literals as DictLiteral
    // entries with string keys.  We walk those.
    if let ExprKind::DictLiteral(entries) = kind {
        for (key, value) in entries {
            let key_name = match &key.kind {
                ExprKind::Ident(sym) => resolve_sym(*sym),
                ExprKind::StringLiteral(s) => s.clone(),
                _ => continue,
            };
            match key_name.as_str() {
                "gpus" => {
                    if let ExprKind::IntLiteral(n) = &value.kind {
                        cluster.gpus = Some(*n);
                    }
                }
                "gpus_per_node" => {
                    if let ExprKind::IntLiteral(n) = &value.kind {
                        cluster.gpus_per_node = Some(*n);
                    }
                }
                "intra_bw" => {
                    if let ExprKind::StringLiteral(s) = &value.kind {
                        if let Some(bw) = parse_bandwidth(s) {
                            cluster.intra_bw_bps = Some(bw);
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@cpdt: cannot parse intra_bw '{s}' (expected e.g. '900GB/s')"
                                ))
                                .with_label(value.span, "invalid bandwidth"),
                            );
                        }
                    }
                }
                "inter_bw" => {
                    if let ExprKind::StringLiteral(s) = &value.kind {
                        if let Some(bw) = parse_bandwidth(s) {
                            cluster.inter_bw_bps = Some(bw);
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@cpdt: cannot parse inter_bw '{s}' (expected e.g. '100GB/s')"
                                ))
                                .with_label(value.span, "invalid bandwidth"),
                            );
                        }
                    }
                }
                _ => {
                    // Unknown sub-key — silent ignore for forward compat.
                }
            }
        }
    } else {
        diagnostics.push(
            Diagnostic::error(
                "@cpdt: cluster must be a dict literal (e.g. { gpus = 8, intra_bw = \"900GB/s\" })"
                    .to_string(),
            )
            .with_label(span, "expected dict"),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_roundtrip() {
        for m in [CpdtMode::Full, CpdtMode::ZeroOnly, CpdtMode::Off] {
            assert_eq!(CpdtMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(CpdtMode::parse("auto"), Some(CpdtMode::Full));
        assert!(CpdtMode::parse("nonsense").is_none());
    }

    #[test]
    fn precision_roundtrip() {
        for p in [PrecisionMode::Auto, PrecisionMode::Fp32, PrecisionMode::Mixed] {
            assert_eq!(PrecisionMode::parse(p.as_str()), Some(p));
        }
        assert!(PrecisionMode::parse("nonsense").is_none());
    }

    #[test]
    fn bandwidth_parses_units() {
        assert!((parse_bandwidth("900GB/s").unwrap() - 9e11).abs() < 1e-3);
        assert!((parse_bandwidth("100GB/s").unwrap() - 1e11).abs() < 1e-3);
        assert!((parse_bandwidth("1TB/s").unwrap() - 1e12).abs() < 1e-3);
        assert!((parse_bandwidth("500MB/s").unwrap() - 5e8).abs() < 1e-3);
    }

    #[test]
    fn bandwidth_rejects_garbage() {
        assert!(parse_bandwidth("fast").is_none());
        assert!(parse_bandwidth("").is_none());
        assert!(parse_bandwidth("GB/s").is_none());
    }
}
