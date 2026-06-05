//! Sprint 2 cycle-4 — @attention_sink decorator semantic validation.
//!
//! Paper §4.3 attention sinks: `@attention_sink(tokens=N)` is a companion
//! decorator on an `@flash_attention` function. The semantic checker
//! enforces:
//!
//!   1. `@attention_sink` requires `@flash_attention` on the same function
//!      (mirrors @tree_mask / @gqa / @rope / @paged_kv companion-required
//!      rule).
//!   2. Unknown args at function scope error (only `tokens` is recognised
//!      by `compiler/kernel.rs::attention_sink` arm; others would be
//!      silently ignored, so we surface them).
//!   3. `tokens` must be a positive integer literal:
//!      3a. zero / negative integers error,
//!      3b. non-literal arg values error.
//!
//! These tests pin all three rules. The v0 codegen path threads
//! `tokens=N` into `FlashAttentionConfig::num_sink_tokens` but the
//! SMEM-layout codegen that actually materializes the sink cache is
//! DEFERRED to a future sprint. The integration test in
//! `nsl-codegen/tests/attention_sink_decorator_integration.rs` pins the
//! decorator-to-config wiring.

use nsl_errors::FileId;
use nsl_lexer::Interner;

fn analyze_src(src: &str) -> Vec<String> {
    let mut interner = Interner::new();
    let file_id = FileId(0);
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, file_id, &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    let mut msgs: Vec<String> = lex_diags.iter().map(|d| d.message.clone()).collect();
    msgs.extend(parse_result.diagnostics.iter().map(|d| d.message.clone()));
    msgs.extend(
        analysis
            .diagnostics
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .map(|d| d.message.clone()),
    );
    msgs
}

fn attention_sink_errs(src: &str) -> Vec<String> {
    analyze_src(src)
        .into_iter()
        .filter(|m| m.to_lowercase().contains("attention_sink"))
        .collect()
}

/// Positive: `@flash_attention` + `@attention_sink(tokens=4)` on the
/// same fn is clean — the decorator extraction site threads `tokens=4`
/// into `config.num_sink_tokens`. tokens=4 is the canonical value (paper
/// §4.3: 4 sink tokens stabilize streaming attention).
#[test]
fn flash_attention_with_attention_sink_is_clean() {
    let src = "\
@flash_attention
@attention_sink(tokens=4)
fn forward():
    pass
";
    let errs = attention_sink_errs(src);
    assert!(
        errs.is_empty(),
        "expected no @attention_sink errors, got {:?}",
        errs
    );
}

/// Negative (Rule 1): `@attention_sink` without `@flash_attention`
/// companion errors. Mirrors the @paged_kv / @tree_mask / @gqa / @rope
/// validation pattern — attention_sink is a flash-attention-only
/// modifier at function scope and has no meaning standalone.
#[test]
fn attention_sink_without_flash_attention_errors() {
    let src = "\
@attention_sink(tokens=4)
fn forward():
    pass
";
    let errs = attention_sink_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("requires @flash_attention")),
        "expected '@attention_sink requires @flash_attention' error, got {:?}",
        errs
    );
}

/// Negative (Rule 2): unknown args at function scope error. Only
/// `tokens` is recognised by the kernel.rs `attention_sink` arm; any
/// other arg would be silently ignored by the extraction site, so the
/// semantic checker surfaces the typo as an error.
#[test]
fn attention_sink_with_unknown_arg_errors() {
    let src = "\
@flash_attention
@attention_sink(count=4)
fn forward():
    pass
";
    let errs = attention_sink_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("unknown argument")),
        "expected '@attention_sink unknown argument' error, got {:?}",
        errs
    );
}

/// Negative (Rule 3a): `tokens=0` errors. A sink count of zero is the
/// disabled sentinel and should be expressed by omitting the decorator
/// entirely — not by writing `tokens=0`, which is almost certainly a bug.
/// The semantic checker rejects non-positive values up front.
#[test]
fn attention_sink_with_zero_tokens_errors() {
    let src = "\
@flash_attention
@attention_sink(tokens=0)
fn forward():
    pass
";
    let errs = attention_sink_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("positive integer")),
        "expected '@attention_sink tokens must be a positive integer' error, got {:?}",
        errs
    );
}

/// Negative (Rule 3b): `tokens` with a non-literal arg errors. The
/// checker only accepts integer literals; identifiers or expressions
/// would silently produce wrong codegen behavior (the kernel.rs
/// extraction arm pattern-matches `ExprKind::IntLiteral`).
#[test]
fn attention_sink_with_non_literal_tokens_errors() {
    let src = "\
@flash_attention
@attention_sink(tokens=x)
fn forward():
    pass
";
    let errs = attention_sink_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("integer literal")),
        "expected '@attention_sink tokens must be an integer literal' error, got {:?}",
        errs
    );
}
