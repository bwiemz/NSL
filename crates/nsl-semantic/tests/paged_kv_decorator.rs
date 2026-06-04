//! Sprint 2 cycle-3 — @paged_kv decorator semantic validation.
//!
//! Paper §3.2 paged KV: `@paged_kv(block_size=N)` is a companion
//! decorator on an `@flash_attention` function. The semantic checker
//! enforces (function-scope only — model-block @paged_kv is handled
//! by `checker/model.rs` which validates a wider set of args):
//!
//!   1. `@paged_kv` requires `@flash_attention` on the same function
//!      (mirrors @tree_mask / @gqa / @rope companion-required rule).
//!   2. Unknown args at function scope error (only `block_size` is
//!      recognised by `compiler/kernel.rs:1005-1022`; others would be
//!      silently ignored, so we surface them).
//!   3. `block_size` must be a positive integer literal.
//!
//! These tests pin all three rules. The actual paged-KV PTX-side
//! block-table indirection was shipped pre-Sprint-2 and is gated on
//! `FlashAttentionConfig::paged`; the integration test in
//! `nsl-codegen/tests/paged_kv_decorator_integration.rs` pins the
//! decorator-to-codegen wiring.

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

fn paged_kv_errs(src: &str) -> Vec<String> {
    analyze_src(src)
        .into_iter()
        .filter(|m| m.to_lowercase().contains("paged_kv"))
        .collect()
}

/// Positive: `@flash_attention` + `@paged_kv(block_size=32)` on the
/// same fn is clean — the decorator extraction site flips
/// `config.paged=true` and the kernel-name dispatcher picks the paged
/// variant. block_size=32 is the canonical fixture value (see
/// `nsl-codegen/tests/fixtures/paged_kv_decorator.nsl`).
#[test]
fn flash_attention_with_paged_kv_is_clean() {
    let src = "\
@flash_attention
@paged_kv(block_size=32)
fn forward():
    pass
";
    let errs = paged_kv_errs(src);
    assert!(
        errs.is_empty(),
        "expected no @paged_kv errors, got {:?}",
        errs
    );
}

/// Negative: `@paged_kv` without `@flash_attention` companion errors.
/// Mirrors the @tree_mask / @gqa / @rope validation pattern — paged_kv
/// is a flash-attention-only modifier at function scope and has no
/// meaning standalone (the function-level extraction site at
/// `compiler/kernel.rs:1005-1022` would silently no-op).
#[test]
fn paged_kv_without_flash_attention_errors() {
    let src = "\
@paged_kv(block_size=32)
fn forward():
    pass
";
    let errs = paged_kv_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("requires @flash_attention")),
        "expected '@paged_kv requires @flash_attention' error, got {:?}",
        errs
    );
}
