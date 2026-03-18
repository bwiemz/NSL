//! M42: @kv_compress decorator validation.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use nsl_errors::Diagnostic;

/// Validated KV compression configuration from a single @kv_compress decorator.
#[derive(Debug, Clone)]
pub struct KvCompressConfig {
    pub method: String,
    pub bits: usize,
    pub dtype: String,
    pub granularity: String,
    pub group_size: usize,
    pub window_size: usize,
    pub num_sinks: usize,
    pub budget: usize,
}

/// Validate a @kv_compress decorator and return the parsed config.
pub fn validate_kv_compress_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<KvCompressConfig> {
    let mut method = String::new();
    let mut bits: usize = 8;
    let mut dtype = String::new();
    let mut granularity = "per_head".to_string();
    let mut group_size: usize = 64;
    let mut window_size: usize = 4096;
    let mut num_sinks: usize = 32;
    let mut budget: usize = 2048;

    // Parse keyword arguments (deco.args is Option<Vec<Arg>>, matches moe.rs pattern)
    if let Some(ref args) = deco.args {
        for arg in args {
            let key = if let Some(ref name_sym) = arg.name {
                resolve_sym(*name_sym)
            } else {
                continue;
            };
            match key.as_str() {
                "method" => {
                    if let ExprKind::StringLiteral(s) = &arg.value.kind {
                        method = s.clone();
                    }
                }
                "bits" => {
                    if let ExprKind::IntLiteral(v) = &arg.value.kind {
                        bits = *v as usize;
                    }
                }
                "dtype" => {
                    if let ExprKind::StringLiteral(s) = &arg.value.kind {
                        dtype = s.clone();
                    }
                }
                "granularity" => {
                    if let ExprKind::StringLiteral(s) = &arg.value.kind {
                        granularity = s.clone();
                    }
                }
                "group_size" => {
                    if let ExprKind::IntLiteral(v) = &arg.value.kind {
                        group_size = *v as usize;
                    }
                }
                "window" => {
                    if let ExprKind::IntLiteral(v) = &arg.value.kind {
                        window_size = *v as usize;
                    }
                }
                "sinks" => {
                    if let ExprKind::IntLiteral(v) = &arg.value.kind {
                        num_sinks = *v as usize;
                    }
                }
                "budget" => {
                    if let ExprKind::IntLiteral(v) = &arg.value.kind {
                        budget = *v as usize;
                    }
                }
                other => {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "unknown @kv_compress parameter '{other}'"
                        ))
                        .with_label(arg.value.span, "here"),
                    );
                    return None;
                }
            }
        }
    }

    // Validate method
    if method.is_empty() {
        diagnostics.push(
            Diagnostic::error("@kv_compress requires a 'method' parameter")
                .with_label(deco.span, "missing method"),
        );
        return None;
    }

    match method.as_str() {
        "quantize" => {
            if !dtype.is_empty() {
                if !["int8", "int4", "fp8"].contains(&dtype.as_str()) {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "unsupported kv_compress dtype '{dtype}', expected: int8, int4, fp8"
                        ))
                        .with_label(deco.span, "here"),
                    );
                    return None;
                }
            } else if ![4, 8].contains(&bits) {
                diagnostics.push(
                    Diagnostic::error("kv_compress quantize bits must be 4 or 8")
                        .with_label(deco.span, "here"),
                );
                return None;
            }
            if !["per_head", "per_token", "per_group"].contains(&granularity.as_str()) {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "unknown granularity '{granularity}', expected: per_head, per_token, per_group"
                    ))
                    .with_label(deco.span, "here"),
                );
                return None;
            }
        }
        "sliding_window" => {
            if num_sinks >= window_size {
                diagnostics.push(
                    Diagnostic::error("sinks must be less than window size")
                        .with_label(deco.span, "here"),
                );
                return None;
            }
        }
        "h2o" => {
            if num_sinks >= budget {
                diagnostics.push(
                    Diagnostic::error("sinks must be less than budget")
                        .with_label(deco.span, "here"),
                );
                return None;
            }
        }
        other => {
            diagnostics.push(
                Diagnostic::error(format!(
                    "unknown kv_compress method '{other}', expected: quantize, sliding_window, h2o"
                ))
                .with_label(deco.span, "here"),
            );
            return None;
        }
    }

    Some(KvCompressConfig {
        method,
        bits,
        dtype,
        granularity,
        group_size,
        window_size,
        num_sinks,
        budget,
    })
}
