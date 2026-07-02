//! Cycle-10 §5.3 Task 4: `EffectChecker::paged_kv_models` tracker.
//!
//! The dispatch fork at `flash_attention_v2/mod.rs::synthesize_backward_with_tier`
//! (Task 9) consults this set to refuse the `@checkpoint(policy="full")` +
//! `@paged_kv` composition. v1 records model-block (layer-decl-scoped)
//! `@paged_kv` membership; call-graph resolution deferred to v4.

use nsl_errors::FileId;
use nsl_lexer::Interner;

fn analyze_source(src: &str) -> nsl_semantic::AnalysisResult {
    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    nsl_semantic::analyze(&parse_result.module, &mut interner)
}

#[test]
fn paged_kv_model_layer_is_recorded() {
    let src = r#"
model Decoder:
    @paged_kv(block_size=32)
    layer kv: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.kv(x)
"#;
    let res = analyze_source(src);
    assert!(
        res.paged_kv_models.contains("Decoder"),
        "expected 'Decoder' in paged_kv_models, got {:?}",
        res.paged_kv_models
    );
}

#[test]
fn model_without_paged_kv_is_absent() {
    let src = r#"
model Plain:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    assert!(
        !res.paged_kv_models.contains("Plain"),
        "expected 'Plain' NOT in paged_kv_models, got {:?}",
        res.paged_kv_models
    );
}
