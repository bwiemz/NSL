//! CFIE Tier-A wiring (audit gap G3): serve-block CFIE config
//! validation — flat `kv_layout` / `kv_quant` / `max_seq` keys and the
//! nested `sampling:` / `speculative:` / `grammar:` sections.

use nsl_errors::FileId;
use nsl_lexer::Interner;

fn analyze_errs(src: &str) -> Vec<String> {
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

const VALID_CFIE_SERVE: &str = r#"
serve Inference:
    max_batch: 32
    max_seq: 2048
    kv_layout: "static"
    kv_quant: "auto"
    target_gpu: "h100"

    sampling:
        temperature: 0.7
        top_k: 50
        top_p: 0.9
        fused: true

    speculative:
        draft: "draft.nslm"
        tokens: 5
        method: "tree"
        tree_width: 2

    grammar:
        schema: "output_schema.json"

    @endpoint
    fn generate(prompt: str) -> str:
        let x = 0
"#;

#[test]
fn full_paper_config_produces_no_errors() {
    let errs = analyze_errs(VALID_CFIE_SERVE);
    assert!(errs.is_empty(), "expected clean analysis, got: {errs:?}");
}

#[test]
fn invalid_kv_layout_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("\"static\"", "\"blockwise\"");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("kv_layout")),
        "expected kv_layout error, got: {errs:?}"
    );
}

#[test]
fn invalid_kv_quant_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("\"auto\"", "\"int3\"");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("kv_quant")),
        "expected kv_quant error, got: {errs:?}"
    );
}

#[test]
fn top_p_out_of_range_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("top_p: 0.9", "top_p: 1.5");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("top_p")),
        "expected top_p error, got: {errs:?}"
    );
}

#[test]
fn non_bool_fused_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("fused: true", "fused: 1");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("fused")),
        "expected fused error, got: {errs:?}"
    );
}

#[test]
fn unknown_speculative_method_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("\"tree\"", "\"eagle3\"");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("method")),
        "expected method error, got: {errs:?}"
    );
}

#[test]
fn speculative_tokens_out_of_range_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("tokens: 5", "tokens: 64");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("tokens")),
        "expected tokens error, got: {errs:?}"
    );
}

#[test]
fn unknown_section_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("sampling:", "samplign:");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("unknown serve section")),
        "expected unknown-section error, got: {errs:?}"
    );
}

#[test]
fn unknown_sampling_key_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("top_k: 50", "topk: 50");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("unknown sampling key")),
        "expected unknown-key error, got: {errs:?}"
    );
}

#[test]
fn grammar_without_schema_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("schema: \"output_schema.json\"", "schema: \"x.json\"");
    // sanity: still valid
    assert!(analyze_errs(&src).is_empty());
    let src2 = VALID_CFIE_SERVE.replace(
        "    grammar:\n        schema: \"output_schema.json\"\n",
        "    grammar:\n        strict: true\n",
    );
    let errs = analyze_errs(&src2);
    assert!(
        errs.iter().any(|m| m.contains("schema")),
        "expected missing-schema error, got: {errs:?}"
    );
}

#[test]
fn cfie_decorator_on_non_serve_stmt_is_rejected() {
    let src = r#"
@cfie(mode=full)
fn helper() -> int:
    return 1
"#;
    let errs = analyze_errs(src);
    assert!(
        errs.iter()
            .any(|m| m.contains("@cfie can only be applied to serve blocks")),
        "expected invalid-target error, got: {errs:?}"
    );
}

#[test]
fn serve_without_cfie_keys_is_untouched() {
    // Pre-CFIE serve blocks must keep passing unchanged (M29/M41 path).
    let src = r#"
serve Legacy:
    max_batch: 32
    kv_blocks: 2048

    @endpoint
    fn handle(prompt: str) -> str:
        let x = 0
"#;
    let errs = analyze_errs(src);
    assert!(errs.is_empty(), "legacy serve must stay clean, got: {errs:?}");
}
