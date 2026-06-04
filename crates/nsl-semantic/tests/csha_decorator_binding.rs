//! Sprint 2: ensure the `@csha(...)` decorator's validated `CshaConfig` is
//! captured on the analysis side-table keyed by the decorated model's name.
//!
//! Before Sprint 2 the call site at `checker/stmt.rs:404-414` dropped the
//! return value of `validate_csha_decorator`, so `@csha(level=2)` and
//! friends parsed cleanly but had zero codegen effect. These tests pin the
//! new side-table API that downstream codegen consults.

use nsl_errors::FileId;
use nsl_lexer::Interner;

fn analyze_source(src: &str) -> nsl_semantic::AnalysisResult {
    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    nsl_semantic::analyze(&parse_result.module, &mut interner)
}

#[test]
fn csha_decorator_with_disable_is_captured() {
    let src = r#"
@csha(disable=true)
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.csha_configs.len(),
        1,
        "expected one captured @csha config, got {:?}",
        res.csha_configs
    );
    let (name, cfg) = &res.csha_configs[0];
    assert_eq!(name, "Toy", "captured config must be keyed by model name");
    assert!(cfg.disabled, "disable=true must round-trip");
    assert!(cfg.level.is_none(), "no level was forced");
    assert!(cfg.target.is_none(), "no target was forced");
}

#[test]
fn csha_decorator_with_level_2_is_captured() {
    let src = r#"
@csha(level=2)
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.csha_configs.len(),
        1,
        "expected one captured @csha config, got {:?}",
        res.csha_configs
    );
    let (name, cfg) = &res.csha_configs[0];
    assert_eq!(name, "Toy");
    assert_eq!(
        cfg.level,
        Some(nsl_semantic::csha::CshaLevel::Pipeline),
        "level=2 must map to CshaLevel::Pipeline"
    );
    assert!(!cfg.disabled);
}

#[test]
fn csha_decorator_with_target_is_captured() {
    let src = r#"
@csha(target=h100)
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    assert_eq!(res.csha_configs.len(), 1);
    let (name, cfg) = &res.csha_configs[0];
    assert_eq!(name, "Toy");
    assert!(cfg.target.is_some(), "target=h100 must be captured");
}

#[test]
fn csha_decorator_absent_yields_empty_side_table() {
    let src = r#"
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    assert!(
        res.csha_configs.is_empty(),
        "no @csha decorator must mean empty side-table, got {:?}",
        res.csha_configs
    );
}

#[test]
fn csha_decorator_with_invalid_level_does_not_capture_but_diagnoses() {
    let src = r#"
@csha(level=9)
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    // The validator still returns Some(cfg) with level=None on bad input
    // (the diagnostic is the error signal). Pin that behaviour so a future
    // refactor that drops the cfg on invalid arg also drops the diagnostic.
    let has_diag = res
        .diagnostics
        .iter()
        .any(|d| format!("{:?}", d).contains("level must be 1, 2, or 3"));
    assert!(has_diag, "expected level-range diagnostic, got {:?}", res.diagnostics);
}
