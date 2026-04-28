//! M56 Task 12 integration test — verifies the agent analysis pipeline
//! runs end-to-end via `analyze_with_imports` (the public semantic entry point).

use std::collections::HashMap;

#[test]
fn linear_pipeline_passes_full_semantic_check() {
    let src = "\
agent Drafter:\n    fn draft(self, p: Tensor) -> Tensor:\n        return p\n\
agent Reviewer:\n    fn review(self, draft: Tensor) -> Tensor:\n        return draft\n\
@pipeline_agent(agents=[Drafter, Reviewer])\n\
fn pipeline(prompt: Tensor) -> Tensor:\n    let draft = drafter.draft(prompt)\n    return reviewer.review(draft)\n";

    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parse_result.diagnostics.is_empty(),
        "parse: {:?}",
        parse_result.diagnostics
    );

    let result = nsl_semantic::analyze_with_imports(
        &parse_result.module,
        &mut interner,
        &HashMap::new(),
        true, // linear_types_enabled
    );

    // No M56 diagnostics expected for the linear pipeline.
    let m56_diags: Vec<&str> = result
        .diagnostics
        .iter()
        .filter(|d| d.message.contains("E060") || d.message.contains("E0610"))
        .map(|d| d.message.as_str())
        .collect();
    assert!(
        m56_diags.is_empty(),
        "expected no M56 diagnostics for valid linear pipeline; got: {:?}",
        m56_diags
    );
}

#[test]
fn cyclic_pipeline_produces_e0603_via_full_pipeline() {
    let src = "\
agent A:\n    fn a_fn(self, x: Tensor) -> Tensor:\n        return x\n\
agent B:\n    fn b_fn(self, y: Tensor) -> Tensor:\n        return y\n\
@pipeline_agent(agents=[A, B])\n\
fn loop_pipe(x: Tensor) -> Tensor:\n    let y = a.a_fn(x)\n    let z = b.b_fn(y)\n    return a.a_fn(z)\n";

    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);

    let result = nsl_semantic::analyze_with_imports(
        &parse_result.module,
        &mut interner,
        &HashMap::new(),
        true, // linear_types_enabled
    );

    assert!(
        result.diagnostics.iter().any(|d| d.message.contains("E0603")),
        "expected E0603 from full pipeline; got: {:?}",
        result
            .diagnostics
            .iter()
            .map(|d| &d.message)
            .collect::<Vec<_>>()
    );
}

#[test]
fn agent_without_linear_types_flag_produces_e0610() {
    let src = "agent X:\n    fn noop(self) -> i32:\n        return 0\n";
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);

    let result = nsl_semantic::analyze_with_imports(
        &parse_result.module,
        &mut interner,
        &HashMap::new(),
        false, // flag OFF
    );

    assert!(
        result.diagnostics.iter().any(|d| d.message.contains("E0610")),
        "expected E0610 when --linear-types is off"
    );

    // E0610 must fire EXACTLY ONCE — not duplicated by the integration.
    let e0610_count = result
        .diagnostics
        .iter()
        .filter(|d| d.message.contains("E0610"))
        .count();
    assert_eq!(
        e0610_count, 1,
        "E0610 should fire exactly once; got {}",
        e0610_count
    );
}
