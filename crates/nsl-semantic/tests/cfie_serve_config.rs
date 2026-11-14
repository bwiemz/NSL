//! Serve-block CFIE config validation — flat `kv_layout` / `kv_quant` /
//! `max_seq` keys and the nested `sampling:` / `speculative:` sections —
//! and the refusal of every request nothing would honour.

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
        seed: 7
        fused: true

    speculative:
        tokens: 5
        method: "standard"
        draft_weights: "draft.safetensors"

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
    let src = VALID_CFIE_SERVE.replace("\"standard\"", "\"eagle3\"");
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

const VALID_CFIE_SERVE_WITH_DRAFT_IN_BINARY: &str = r#"
serve Inference:
    max_batch: 32
    max_seq: 2048
    kv_layout: "static"
    kv_quant: "uniform_fp16"
    target_gpu: "h100"

    sampling:
        temperature: 0.7
        top_k: 50
        top_p: 0.9
        fused: true

    speculative:
        tokens: 5
        method: "standard"
        temperature: 0.0
        draft_weights: "draft.safetensors"
        draft_n_layers: 2
        draft_d_model: 256
        draft_n_heads: 4
        draft_n_kv_heads: 4
        draft_head_dim: 64
        draft_d_ff: 1024

    @endpoint
    fn generate(prompt: str) -> str:
        let x = 0
"#;

#[test]
fn draft_in_binary_config_produces_no_errors() {
    let errs = analyze_errs(VALID_CFIE_SERVE_WITH_DRAFT_IN_BINARY);
    assert!(errs.is_empty(), "expected clean analysis, got: {errs:?}");
}

#[test]
fn draft_weights_non_string_is_rejected() {
    let src = VALID_CFIE_SERVE_WITH_DRAFT_IN_BINARY
        .replace("draft_weights: \"draft.safetensors\"", "draft_weights: 5");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter().any(|m| m.contains("draft_weights")),
        "expected draft_weights error, got: {errs:?}"
    );
}

#[test]
fn draft_shape_key_zero_is_rejected() {
    let src = VALID_CFIE_SERVE_WITH_DRAFT_IN_BINARY.replace("draft_n_layers: 2", "draft_n_layers: 0");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter()
            .any(|m| m.contains("draft_n_layers") && m.contains("positive integer")),
        "expected draft_n_layers error, got: {errs:?}"
    );
}

#[test]
fn draft_shape_key_negative_is_rejected() {
    let src = VALID_CFIE_SERVE_WITH_DRAFT_IN_BINARY
        .replace("draft_head_dim: 64", "draft_head_dim: -1");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter()
            .any(|m| m.contains("draft_head_dim") && m.contains("positive integer")),
        "expected draft_head_dim error, got: {errs:?}"
    );
}

#[test]
fn unknown_speculative_key_names_draft_shape_keys() {
    let src = VALID_CFIE_SERVE_WITH_DRAFT_IN_BINARY
        .replace("draft_d_ff: 1024", "draft_dff_typo: 1024");
    let errs = analyze_errs(&src);
    assert!(
        errs.iter()
            .any(|m| m.contains("unknown speculative key") && m.contains("draft_d_ff")),
        "expected unknown-key error naming draft_* shape keys, got: {errs:?}"
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

/// The one error in `errs` that contains `needle`, or a panic that shows
/// what was reported instead.
fn expect_error(errs: &[String], needle: &str) {
    assert!(
        errs.iter().any(|m| m.contains(needle)),
        "expected an error containing `{needle}`, got: {errs:?}"
    );
}

/// A `grammar:` section used to be accepted and to build a sampler that
/// held every token to the grammar's start state.
#[test]
fn a_grammar_section_is_refused() {
    let src = VALID_CFIE_SERVE.replace(
        "    @endpoint",
        "    grammar:\n        schema: \"output_schema.json\"\n        tokenizer: \"vocab.txt\"\n\n    @endpoint",
    );
    let errs = analyze_errs(&src);
    expect_error(&errs, "serve `grammar:` is not supported");
    assert_eq!(errs.len(), 1, "{errs:?}");
}

#[test]
fn int8_kv_is_refused() {
    for spelling in ["uniform_int8", "int8"] {
        let src = VALID_CFIE_SERVE.replace("kv_quant: \"auto\"", &format!("kv_quant: \"{spelling}\""));
        expect_error(&analyze_errs(&src), "INT8 KV is not supported");
    }
    for spelling in ["uniform_fp16", "fp16", "off", "auto"] {
        let src = VALID_CFIE_SERVE.replace("kv_quant: \"auto\"", &format!("kv_quant: \"{spelling}\""));
        assert!(analyze_errs(&src).is_empty(), "{spelling}");
    }
}

/// A section with only `temperature` used to get top_k = 50 and
/// top_p = 0.9 without notice.
#[test]
fn a_temperature_without_top_k_is_refused() {
    let src = VALID_CFIE_SERVE.replace("        top_k: 50\n", "");
    expect_error(&analyze_errs(&src), "a non-zero temperature needs top_k");
    // top_p is optional, and greedy needs neither.
    let src = VALID_CFIE_SERVE.replace("        top_p: 0.9\n", "");
    assert!(analyze_errs(&src).is_empty());
    let src = VALID_CFIE_SERVE
        .replace("temperature: 0.7", "temperature: 0.0")
        .replace("        top_k: 50\n", "")
        .replace("        top_p: 0.9\n", "");
    assert!(analyze_errs(&src).is_empty(), "{:?}", analyze_errs(&src));
}

/// `top_k: 100` used to pass the checker and then skip the sampler kernel
/// with only a note, leaving a binary whose generate() always refused.
#[test]
fn a_top_k_the_sampler_cannot_hold_is_refused() {
    let src = VALID_CFIE_SERVE.replace("top_k: 50", "top_k: 100");
    expect_error(&analyze_errs(&src), "top_k must be an integer in [1, 64]");
    let src = VALID_CFIE_SERVE.replace("top_k: 50", "top_k: 64");
    assert!(analyze_errs(&src).is_empty());
}

#[test]
fn a_negative_seed_is_rejected() {
    let src = VALID_CFIE_SERVE.replace("seed: 7", "seed: -1");
    expect_error(&analyze_errs(&src), "seed must be a non-negative integer");
}

/// A shape key that is not a positive integer literal used to fall back to
/// a default without a word, and `rope_theta: 0` baked NaN into the
/// decode kernel's rotation.
#[test]
fn shape_keys_and_baked_constants_must_be_positive() {
    let with = |line: &str| VALID_CFIE_SERVE.replace("    target_gpu: \"h100\"\n", &format!("    target_gpu: \"h100\"\n    {line}\n"));
    assert!(analyze_errs(&with("n_layers: 8")).is_empty());
    assert!(analyze_errs(&with("rope_theta: 500000.0")).is_empty());
    assert!(analyze_errs(&with("norm_eps: 0.000001")).is_empty());
    for (line, needle) in [
        ("n_layers: 0", "n_layers must be an integer in [1, 16777216]"),
        ("n_layers: -2", "n_layers must be an integer"),
        ("d_model: 4294967297", "d_model must be an integer"),
        ("head_dim: \"wide\"", "head_dim must be an integer"),
        ("vocab_size: 0", "vocab_size must be an integer"),
        ("max_new_tokens: 0", "max_new_tokens must be an integer"),
        ("rope_theta: 0", "rope_theta must be a positive number"),
        ("rope_theta: -10000.0", "rope_theta must be a positive number"),
        ("norm_eps: 0.0", "norm_eps must be a positive number"),
    ] {
        expect_error(&analyze_errs(&with(line)), needle);
    }
}

/// Each of these was accepted, stored and never read.
#[test]
fn speculative_settings_with_no_driver_are_refused() {
    for (from, to, needle) in [
        ("method: \"standard\"", "method: \"tree\"", "method \"tree\" is not supported"),
        ("method: \"standard\"", "method: \"medusa\"", "method \"medusa\" is not supported"),
        ("method: \"standard\"", "method: \"lookahead\"", "method \"lookahead\" is not supported"),
        ("tokens: 5", "tokens: 5\n        tree_width: 2", "tree_width is not supported"),
        ("tokens: 5", "tokens: 5\n        temperature: 0.8", "speculative temperature is not supported"),
        ("tokens: 5", "tokens: 5\n        draft: \"draft.nslm\"", "speculative `draft` is not supported"),
        ("        draft_weights: \"draft.safetensors\"\n", "", "the section needs draft_weights"),
    ] {
        let src = VALID_CFIE_SERVE.replace(from, to);
        assert_ne!(src, VALID_CFIE_SERVE, "`{from}` not found in the fixture");
        expect_error(&analyze_errs(&src), needle);
    }
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
