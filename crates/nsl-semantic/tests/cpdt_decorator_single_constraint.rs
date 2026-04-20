//! Phase 1 Item 2 (post-retune Tier A): `@cpdt` decorator single-instance
//! constraint. Multiple `@cpdt` decorators in the same program emit a
//! diagnostic error naming both spans. See
//! docs/superpowers/specs/2026-04-20-cpdt-weight-aware-opt-out-design.md.

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

fn cpdt_errs(src: &str) -> Vec<String> {
    analyze_src(src)
        .into_iter()
        .filter(|m| m.to_lowercase().contains("cpdt"))
        .collect()
}

#[test]
fn single_cpdt_decorator_is_clean() {
    // A single @cpdt decorator is the expected case — no error on the
    // single-instance constraint. (Other @cpdt validation errors may still
    // fire, but not the "at most once" diagnostic.)
    let src = r#"
@cpdt(num_gpus=4)
train Toy:
    data: []
    optimizer: adam
"#;
    let errs = cpdt_errs(src);
    assert!(
        !errs.iter().any(|m| m.contains("at most once")),
        "single @cpdt should not emit the single-instance diagnostic; got {:?}",
        errs
    );
}

#[test]
fn duplicate_cpdt_decorator_emits_diagnostic() {
    // Two @cpdt decorators — on different train blocks — trigger the
    // Phase 1 single-instance constraint diagnostic.
    let src = r#"
@cpdt(num_gpus=4)
train A:
    data: []
    optimizer: adam

@cpdt(num_gpus=2)
train B:
    data: []
    optimizer: adam
"#;
    let errs = cpdt_errs(src);
    assert!(
        errs.iter().any(|m| m.contains("at most once")),
        "duplicate @cpdt should emit single-instance diagnostic; got {:?}",
        errs
    );
}

#[test]
fn cpdt_weight_aware_false_parses_clean() {
    // The weight_aware=false kwarg is the scope of this PR. A single
    // @cpdt(weight_aware=false) decorator must parse cleanly without any
    // single-instance diagnostic.
    let src = r#"
@cpdt(num_gpus=4, weight_aware=false)
train Toy:
    data: []
    optimizer: adam
"#;
    let errs = cpdt_errs(src);
    assert!(
        !errs.iter().any(|m| m.contains("at most once")),
        "single @cpdt(weight_aware=false) should not emit single-instance diagnostic; got {:?}",
        errs
    );
}
