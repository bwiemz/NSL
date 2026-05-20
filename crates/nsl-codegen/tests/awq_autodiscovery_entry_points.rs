use nsl_codegen::{
    compile_returning_plan, compile_returning_splice_count_for_tests, CompileOptions,
};

const AUTO_DISCOVERY_FIXTURE: &str = r#"
fn up_proj(x: Tensor) -> Tensor:
    return x

fn down_proj(x: Tensor) -> Tensor:
    return x

fn relu(x: Tensor) -> Tensor:
    return x

@quantize(dtype="awq4")
model TinyMLP:
    up_proj: Tensor = zeros([128, 64])
    down_proj: Tensor = zeros([64, 128])

    fn forward(self, x: Tensor) -> Tensor:
        return x |> up_proj |> relu |> down_proj
"#;

const MODEL_TENSOR_PIPE_FIXTURE: &str = r#"
@quantize(dtype="awq4")
model TinyMLP:
    up_proj: Tensor = zeros([128, 64])
    down_proj: Tensor = zeros([64, 128])

    fn forward(self, x: Tensor) -> Tensor:
        return x |> up_proj |> relu |> down_proj
"#;

const EMPTY_QUANTIZE_FIXTURE: &str = r#"
@quantize(dtype="awq4")
model EmptyModel:
    fn forward(self, x: Tensor) -> Tensor:
        return x

fn main():
    let _x = 0
"#;

fn splice_count_for_source(source: &str) -> u32 {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, nsl_errors::FileId(0), &mut interner);
    assert!(
        lex_diags
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must pass semantic analysis: {:?}",
        analysis.diagnostics
    );

    let opts = CompileOptions::default();
    assert!(opts.calibration_retention.is_none());

    compile_returning_splice_count_for_tests(&parsed.module, &interner, &analysis.type_map, &opts)
        .expect("compile must succeed with auto-discovery")
}

fn compile_plan_error_for_source(source: &str) -> String {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, nsl_errors::FileId(0), &mut interner);
    assert!(
        lex_diags
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must pass semantic analysis: {:?}",
        analysis.diagnostics
    );

    let opts = CompileOptions::default();
    let err = compile_returning_plan(&parsed.module, &interner, &analysis.type_map, false, &opts)
        .expect_err("compile must refuse empty AWQ discovery");
    format!("{err}")
}

#[test]
fn auto_discovery_populates_retention_before_splice_codegen() {
    let splice_count = splice_count_for_source(AUTO_DISCOVERY_FIXTURE);

    assert_eq!(
        splice_count, 2,
        "AWQ AST pre-scan must populate calibration_retention before method-body codegen"
    );
}

#[test]
fn auto_discovery_handles_real_awq_fixture_tensor_field_pipes() {
    assert_eq!(
        splice_count_for_source(MODEL_TENSOR_PIPE_FIXTURE),
        2,
        "model tensor pipe targets should emit both retention splices without free-function shims"
    );
}

#[test]
fn quantize_decorator_with_zero_linear_layers_refuses() {
    let err = compile_plan_error_for_source(EMPTY_QUANTIZE_FIXTURE);

    assert!(err.contains("calibration: @quantize model declared but no AWQ projections discovered"));
    assert!(err.contains("requested:"));
    assert!(err.contains("expected:"));
    assert!(err.contains("found:"));
    assert!(err.contains("Action:"));
}
