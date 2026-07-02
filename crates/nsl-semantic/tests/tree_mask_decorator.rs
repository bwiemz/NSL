//! Sprint 1 cycle-3 — @tree_mask decorator semantic validation.
//!
//! Paper §4 tree-mask v1: bare `@tree_mask` is a companion decorator on
//! an `@flash_attention` function. The semantic checker enforces:
//!
//!   1. `@tree_mask` requires `@flash_attention` on the same function.
//!   2. `@tree_mask` takes no arguments in v1 (mirrors the kernel.rs
//!      extraction-site validation so users see the same error from
//!      both the semantic analyzer and the codegen path).
//!
//! These tests pin both rules. The actual PTX-side DFS-enter/DFS-exit
//! ancestor check was shipped in M33 and is gated on
//! `FlashAttentionConfig::tree_mask`; the integration test in
//! `nsl-codegen/tests/tree_mask_decorator_integration.rs` pins the
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

fn tree_mask_errs(src: &str) -> Vec<String> {
    analyze_src(src)
        .into_iter()
        .filter(|m| m.to_lowercase().contains("tree_mask"))
        .collect()
}

/// Positive: `@flash_attention` + bare `@tree_mask` on the same fn is
/// clean — the decorator extraction site flips `config.tree_mask=true`
/// and the kernel-name dispatcher picks the tree_mask variant.
#[test]
fn flash_attention_with_tree_mask_is_clean() {
    let src = "\
@flash_attention
@tree_mask
fn forward():
    pass
";
    let errs = tree_mask_errs(src);
    assert!(
        errs.is_empty(),
        "expected no @tree_mask errors, got {:?}",
        errs
    );
}

/// Negative: `@tree_mask` without `@flash_attention` companion errors.
/// Mirrors the @rope and @gqa validation patterns — tree_mask is a
/// flash-attention-only modifier and has no meaning standalone.
#[test]
fn tree_mask_without_flash_attention_errors() {
    let src = "\
@tree_mask
fn forward():
    pass
";
    let errs = tree_mask_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("requires @flash_attention")),
        "expected '@tree_mask requires @flash_attention' error, got {:?}",
        errs
    );
}

/// Negative: `@tree_mask(...)` with any args is a v1-spec violation.
/// Future tree-mask args (e.g., a max-depth budget) require an explicit
/// spec update + extraction-site change rather than silently being
/// ignored. The extractor in `kernel.rs` enforces the same rule so the
/// two layers agree on what "v1" means.
#[test]
fn tree_mask_with_args_errors() {
    let src = "\
@flash_attention
@tree_mask(depth=4)
fn forward():
    pass
";
    let errs = tree_mask_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("no arguments")),
        "expected 'no arguments in v1' error, got {:?}",
        errs
    );
}
