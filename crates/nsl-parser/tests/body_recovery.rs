//! Error recovery inside the bodies that do not parse statements
//! (`tokenizer`, its `key = value` sections, `dataset`) must make progress
//! on every line.
//!
//! Found by the parse fuzz target (`fuzz/`): a statement keyword on such a
//! line — `if b` inside a tokenizer body — left `synchronize` parked in
//! front of the keyword, and the body loop pushed the same diagnostic until
//! the process ran out of memory. These tests pin the shape of the
//! recovery — one diagnostic per bad line, and the good lines around it
//! still parsed — and run every parse under a deadline, so a regression
//! fails the test instead of taking the test process down with it.

use std::io::Write;
use std::sync::mpsc;
use std::time::Duration;

use nsl_ast::block::TokenizerStmt;
use nsl_ast::stmt::StmtKind;
use nsl_errors::FileId;
use nsl_lexer::Interner;

/// Parse `source`, or exit the test process if that takes longer than
/// the deadline: a body loop that is not making progress allocates a
/// diagnostic per iteration, so leaving it running while the other tests
/// finish would replace the failure with an out-of-memory kill.
fn parse(source: &'static str) -> (nsl_ast::Module, Vec<nsl_errors::Diagnostic>) {
    const DEADLINE: Duration = Duration::from_secs(10);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut interner = Interner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(source, FileId(1), &mut interner);
        assert!(lex_diags.is_empty(), "{lex_diags:?}");
        let result = nsl_parser::parse(&tokens, &mut interner);
        tx.send((result.module, result.diagnostics)).unwrap();
    });
    match rx.recv_timeout(DEADLINE) {
        Ok(parsed) => parsed,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            // Straight to the process's stderr: the test harness captures
            // `eprintln!` and would lose it when the process exits.
            let _ = std::io::stderr().write_all(
                format!(
                    "\nFAILED: the parser did not finish {source:?} within {DEADLINE:?}; \
                     a body loop is not making progress\n"
                )
                .as_bytes(),
            );
            std::process::exit(101);
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("the frontend panicked on {source:?} (the panic itself is captured above)")
        }
    }
}

fn messages(diags: &[nsl_errors::Diagnostic]) -> Vec<&str> {
    diags.iter().map(|d| d.message.as_str()).collect()
}

fn tokenizer_body(module: &nsl_ast::Module) -> &[TokenizerStmt] {
    let [stmt] = module.stmts.as_slice() else {
        panic!("expected one top-level statement, got {:?}", module.stmts);
    };
    let StmtKind::TokenizerDef(def) = &stmt.kind else {
        panic!("expected a tokenizer def, got {:?}", stmt.kind);
    };
    &def.body
}

fn is_one_normalize_rule(body: &[TokenizerStmt]) -> bool {
    matches!(body, [TokenizerStmt::Normalize { rules, .. }] if rules.len() == 1)
}

#[test]
fn keyword_line_in_a_tokenizer_body_is_skipped_once() {
    let (module, diags) = parse("tokenizer t:\n    if b\n    normalize: lowercase\n");
    assert_eq!(messages(&diags), ["expected tokenizer section"]);
    let body = tokenizer_body(&module);
    assert!(
        is_one_normalize_rule(body),
        "the section after the bad line was not parsed: {body:?}"
    );
}

#[test]
fn keyword_line_in_a_key_value_section_is_skipped_once() {
    let (module, diags) = parse(
        "tokenizer t:\n    special_tokens:\n        pad = 0\n        for x\n        eos = 1\n",
    );
    assert_eq!(messages(&diags), ["expected key = value entry"]);
    let body = tokenizer_body(&module);
    assert!(
        matches!(body, [TokenizerStmt::SpecialTokens { entries, .. }] if entries.len() == 2),
        "the entries around the bad line were not both parsed: {body:?}"
    );
}

#[test]
fn keyword_line_in_a_dataset_body_is_skipped_once() {
    let (module, diags) =
        parse("dataset d(source):\n    path = \"x\"\n    return\n    shards = 4\n");
    assert_eq!(messages(&diags), ["expected dataset field assignment"]);

    let StmtKind::DatasetDef(def) = &module.stmts[0].kind else {
        panic!("expected a dataset def, got {:?}", module.stmts);
    };
    assert_eq!(def.body.len(), 2, "{:?}", def.body);
}

#[test]
fn keyword_line_with_a_suite_in_a_tokenizer_body_is_skipped_with_its_suite() {
    // `if b:` opens an indented block the body loop has no arm for: the
    // block is part of the bad line, so it is dropped with it — one
    // diagnostic, and the section after it is still parsed.
    let (module, diags) = parse(
        "tokenizer t:\n    if b:\n        x = 1\n        y = 2\n    normalize: lowercase\n",
    );
    assert_eq!(messages(&diags), ["expected tokenizer section"]);
    let body = tokenizer_body(&module);
    assert!(is_one_normalize_rule(body), "{body:?}");
}

#[test]
fn keyword_line_with_a_suite_in_a_dataset_body_is_skipped_with_its_suite() {
    let (module, diags) = parse(
        "dataset d(source):\n    path = \"x\"\n    for i in xs:\n        y = 2\n    shards = 4\n",
    );
    assert_eq!(messages(&diags), ["expected dataset field assignment"]);
    let StmtKind::DatasetDef(def) = &module.stmts[0].kind else {
        panic!("expected a dataset def, got {:?}", module.stmts);
    };
    assert_eq!(def.body.len(), 2, "the field after the suite was lost: {:?}", def.body);
}

#[test]
fn keyword_line_with_a_nested_suite_is_skipped_whole() {
    // The suite has a block of its own; the skip must balance the inner
    // Indent/Dedent pair and stop at the end of the outer suite, not at
    // the first Dedent.
    let (module, diags) = parse(
        "tokenizer t:\n    if b:\n        if c:\n            x = 1\n        y = 2\n    normalize: lowercase\n",
    );
    assert_eq!(messages(&diags), ["expected tokenizer section"]);
    let body = tokenizer_body(&module);
    assert!(is_one_normalize_rule(body), "{body:?}");
}

#[test]
fn bad_line_at_the_end_of_a_body_ends_the_body() {
    // The bad line is the last one: recovery must stop at the Dedent so the
    // statement after the body is parsed at the top level, not swallowed.
    let (module, diags) = parse("tokenizer t:\n    if b\nlet x = 1\n");
    assert_eq!(messages(&diags), ["expected tokenizer section"]);
    assert_eq!(module.stmts.len(), 2, "{:?}", module.stmts);
    assert!(
        matches!(module.stmts[1].kind, StmtKind::VarDecl { .. }),
        "{:?}",
        module.stmts[1].kind
    );
}

#[test]
fn unknown_tokenizer_section_with_a_keyword_on_its_line_reports_once() {
    // An unknown section used to recover with `synchronize`, which stops
    // in front of `if`; the body loop then reported the same line a second
    // time as "expected tokenizer section".
    let (module, diags) = parse("tokenizer t:\n    foo: if b\n    normalize: lowercase\n");
    assert_eq!(messages(&diags), ["unexpected tokenizer section 'foo'"]);
    let body = tokenizer_body(&module);
    assert!(is_one_normalize_rule(body), "{body:?}");
}

#[test]
fn unknown_tokenizer_section_with_an_indented_body_skips_the_body() {
    let (module, diags) = parse(
        "tokenizer t:\n    foo:\n        x = 1\n        y = 2\n    normalize: lowercase\n",
    );
    assert_eq!(messages(&diags), ["unexpected tokenizer section 'foo'"]);
    let body = tokenizer_body(&module);
    assert!(is_one_normalize_rule(body), "{body:?}");
}

#[test]
fn unknown_tokenizer_section_without_a_body_keeps_the_next_section() {
    // `foo:` alone on its line: the recovery used to look for a body,
    // find the next section instead, and skip that line as if it were
    // the body.
    let (module, diags) = parse("tokenizer t:\n    foo:\n    normalize: lowercase\n");
    assert_eq!(messages(&diags), ["unexpected tokenizer section 'foo'"]);
    let body = tokenizer_body(&module);
    assert!(is_one_normalize_rule(body), "{body:?}");
}
