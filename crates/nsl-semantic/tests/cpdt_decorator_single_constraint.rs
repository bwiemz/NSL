//! Phase 1 Item 2 (post-retune Tier A): `@cpdt` decorator single-instance
//! constraint. Multiple `@cpdt` decorators in the same program emit a
//! diagnostic error naming both spans. See
//! docs/superpowers/specs/2026-04-20-cpdt-weight-aware-opt-out-design.md.

use nsl_errors::{Diagnostic, FileId};
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

/// Returns the full semantic-analysis error diagnostics so tests can inspect
/// labels, not just message strings.
fn analyze_full(src: &str) -> Vec<Diagnostic> {
    let mut interner = Interner::new();
    let file_id = FileId(0);
    let (_tokens, _) = nsl_lexer::tokenize(src, file_id, &mut interner);
    let parse_result = nsl_parser::parse(
        &nsl_lexer::tokenize(src, file_id, &mut interner).0,
        &mut interner,
    );
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    analysis
        .diagnostics
        .into_iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect()
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
fn duplicate_cpdt_decorator_emits_diagnostic_with_two_labels() {
    // Two @cpdt decorators — on different train blocks — trigger the
    // Phase 1 single-instance constraint diagnostic. The diagnostic
    // must have two labels (one for each decorator site) so users can
    // immediately locate both decorators in their source.
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
    let diags = analyze_full(src);
    let single_instance_diag = diags
        .iter()
        .find(|d| d.message.contains("at most once"))
        .expect("duplicate @cpdt should emit single-instance diagnostic");
    assert_eq!(
        single_instance_diag.labels.len(),
        2,
        "single-instance diagnostic must carry two labels (one per @cpdt site); got {}: {:?}",
        single_instance_diag.labels.len(),
        single_instance_diag.labels
    );
    // Verify the label messages match the spec's contract: one "duplicate",
    // one "previous".
    let label_msgs: Vec<&str> = single_instance_diag
        .labels
        .iter()
        .map(|l| l.message.as_str())
        .collect();
    assert!(
        label_msgs.iter().any(|m| m.contains("duplicate")),
        "expected a `duplicate` label; got {label_msgs:?}"
    );
    assert!(
        label_msgs.iter().any(|m| m.contains("previous")),
        "expected a `previous` label; got {label_msgs:?}"
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
