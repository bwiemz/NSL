//! Phase 5 Task 6: @inspect decorator argument validation.

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

fn inspect_errs(src: &str) -> Vec<String> {
    analyze_src(src)
        .into_iter()
        .filter(|m| m.to_lowercase().contains("inspect"))
        .collect()
}

#[test]
fn at_inspect_with_no_args_errors() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    @inspect()
    return x
"#;
    let errs = inspect_errs(src);
    assert!(
        !errs.is_empty(),
        "expected error about inspect, got {:?}",
        errs
    );
}

#[test]
fn at_inspect_with_only_tensor_errors_missing_kw() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h)
    return h
"#;
    let errs = inspect_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("every") || e.contains("condition")),
        "expected error mentioning every=/condition=, got {:?}",
        errs
    );
}

#[test]
fn at_inspect_with_every_is_clean() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h, every=10)
    return h
"#;
    let errs = inspect_errs(src);
    assert!(errs.is_empty(), "expected no @inspect errors, got {:?}", errs);
}

#[test]
fn at_inspect_with_condition_is_clean() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h, condition="step > 100")
    return h
"#;
    let errs = inspect_errs(src);
    assert!(errs.is_empty(), "expected no @inspect errors, got {:?}", errs);
}

#[test]
fn at_inspect_with_both_is_clean() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h, every=10, condition="loss > 5.0")
    return h
"#;
    let errs = inspect_errs(src);
    assert!(errs.is_empty(), "expected no @inspect errors, got {:?}", errs);
}

#[test]
fn at_inspect_with_unknown_kw_errors() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h, every=10, flavor="banana")
    return h
"#;
    let errs = inspect_errs(src);
    assert!(
        errs.iter().any(|e| e.contains("unknown keyword")),
        "expected 'unknown keyword' error, got {:?}",
        errs
    );
}
