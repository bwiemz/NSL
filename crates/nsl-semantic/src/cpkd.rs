//! CPKD semantic layer — the `@fused_kl_ce(...)` decorator on `distill`
//! blocks.
//!
//! Mirrors the CFTP `@fused_lm_ce` contract: OPT-IN ONLY (no fused KL-CE
//! op fires without `enabled = true` plus all five shape hints), keyword
//! args only, literal-only values, per-distill-block dispatch via
//! `distill_block_stmt_id` (the checker stamps the enclosing Stmt id).
//!
//! v1 deferrals refuse loudly here rather than degrade at codegen:
//! - `vocab_size > 8192` (single-CTA kernel ceiling; large-vocab variant
//!   deferred),
//! - any `dtype` other than f32 (fp16/bf16 emitters deferred).

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

/// Configuration extracted from a `@fused_kl_ce(...)` decorator on a
/// `distill` block. Parsed by [`validate_fused_kl_ce_decorator`].
#[derive(Debug, Clone)]
pub struct FusedKlCeConfig {
    pub enabled: bool,
    /// Vocabulary size (student AND teacher — they must share a tokenizer,
    /// which is a CPKD precondition). Range (0, 8192] in v1.
    pub vocab_size: Option<u32>,
    /// Student hidden dim (`hidden_size =`, matching @fused_lm_ce naming
    /// for the trainable side). Must be divisible by 32.
    pub hidden_size: Option<u32>,
    /// Teacher hidden dim (`teacher_hidden =`). Must be divisible by 32.
    pub teacher_hidden: Option<u32>,
    pub batch_size: Option<u32>,
    pub seq_len: Option<u32>,
    /// SMEM vocab tile; must be a multiple of 128 (same silent-corruption
    /// invariant as @fused_lm_ce — the inner fill is 128-thread-wide).
    pub vocab_tile: Option<u32>,
    pub span: Span,
    /// AST NodeId of the enclosing `distill` block Stmt; the validator
    /// writes `NodeId::dummy()`, the checker stamps the real id at the
    /// push site (per-block dispatch, CFTP v10 pattern).
    pub distill_block_stmt_id: nsl_ast::NodeId,
}

/// Parse and validate a `@fused_kl_ce(enabled = true, vocab_size = 4096,
/// hidden_size = 128, teacher_hidden = 256, batch_size = 2, seq_len = 32)`
/// decorator. Mirrors [`crate::cftp::validate_fused_ce_decorator`].
pub fn validate_fused_kl_ce_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<FusedKlCeConfig> {
    let mut enabled = false;
    let mut vocab_size: Option<u32> = None;
    let mut hidden_size: Option<u32> = None;
    let mut teacher_hidden: Option<u32> = None;
    let mut batch_size: Option<u32> = None;
    let mut seq_len: Option<u32> = None;
    let mut vocab_tile: Option<u32> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error(
                        "@fused_kl_ce: positional arguments are not allowed".to_string(),
                    )
                    .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            let int_lit = |diagnostics: &mut Vec<Diagnostic>| -> Option<i64> {
                match &arg.value.kind {
                    ExprKind::IntLiteral(n) => Some(*n),
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@fused_kl_ce: `{aname}` must be an integer literal"
                            ))
                            .with_label(arg.span, "expected integer"),
                        );
                        None
                    }
                }
            };
            match aname.as_str() {
                "enabled" => match &arg.value.kind {
                    ExprKind::BoolLiteral(b) => enabled = *b,
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@fused_kl_ce: `enabled` must be a bool literal".to_string(),
                        )
                        .with_label(arg.span, "expected bool"),
                    ),
                },
                "vocab_size" => {
                    if let Some(n) = int_lit(diagnostics) {
                        if n > 0 && n <= 8192 {
                            vocab_size = Some(n as u32);
                        } else if n > 8192 {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_kl_ce: vocab_size {n} exceeds the v1 ceiling of \
                                     8192 (single-CTA kernel; the large-vocab two-kernel \
                                     variant is a deferred extension)"
                                ))
                                .with_label(arg.span, "vocab_size > 8192 deferred"),
                            );
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_kl_ce: vocab_size {n} out of range (0, 8192]"
                                ))
                                .with_label(arg.span, "invalid vocab_size"),
                            );
                        }
                    }
                }
                "hidden_size" | "teacher_hidden" => {
                    if let Some(n) = int_lit(diagnostics) {
                        if n > 0 && n <= 65_536 && n % 32 == 0 {
                            if aname == "hidden_size" {
                                hidden_size = Some(n as u32);
                            } else {
                                teacher_hidden = Some(n as u32);
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_kl_ce: {aname} {n} must be in (0, 65536] and \
                                     divisible by 32"
                                ))
                                .with_label(arg.span, "invalid hidden dim"),
                            );
                        }
                    }
                }
                "batch_size" => {
                    if let Some(n) = int_lit(diagnostics) {
                        if n > 0 && n <= 65_536 {
                            batch_size = Some(n as u32);
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_kl_ce: batch_size {n} out of range (0, 65536]"
                                ))
                                .with_label(arg.span, "invalid batch_size"),
                            );
                        }
                    }
                }
                "seq_len" => {
                    if let Some(n) = int_lit(diagnostics) {
                        if n > 0 && n <= 65_536 {
                            seq_len = Some(n as u32);
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_kl_ce: seq_len {n} out of range (0, 65536]"
                                ))
                                .with_label(arg.span, "invalid seq_len"),
                            );
                        }
                    }
                }
                "vocab_tile" => {
                    if let Some(n) = int_lit(diagnostics) {
                        if n > 0 && n <= 8192 && n % 128 == 0 {
                            vocab_tile = Some(n as u32);
                        } else {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@fused_kl_ce: vocab_tile {n} must be in [128, 8192] and a \
                                     multiple of 128 (the inner fill is 128-thread-wide; \
                                     non-aligned tiles silently corrupt the online-softmax \
                                     reduction)"
                                ))
                                .with_label(arg.span, "invalid vocab_tile"),
                            );
                        }
                    }
                }
                "dtype" => match &arg.value.kind {
                    ExprKind::StringLiteral(s) if s == "f32" || s == "fp32" => {}
                    ExprKind::StringLiteral(s) => diagnostics.push(
                        Diagnostic::error(format!(
                            "@fused_kl_ce: dtype '{s}' is not supported in v1 — the fused \
                             KL-CE kernel is f32-only (fp16/bf16 emitters are a deferred \
                             extension)"
                        ))
                        .with_label(arg.span, "f32 only in v1"),
                    ),
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@fused_kl_ce: `dtype` must be a string literal".to_string(),
                        )
                        .with_label(arg.span, "expected string"),
                    ),
                },
                other => diagnostics.push(
                    Diagnostic::error(format!("@fused_kl_ce: unknown argument '{other}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(FusedKlCeConfig {
        enabled,
        vocab_size,
        hidden_size,
        teacher_hidden,
        batch_size,
        seq_len,
        vocab_tile,
        span: deco.span,
        distill_block_stmt_id: nsl_ast::NodeId::dummy(),
    })
}
