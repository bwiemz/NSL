//! Golden-file parser tests.
//!
//! Every `tests/parse/*.nsl` is lexed and parsed, and the outcome — the
//! lexer's and parser's diagnostics first, then the AST — is rendered to
//! text and compared against `tests/parse/<stem>.ast.snap` with insta.
//! A fixture whose stem starts with `err_` is expected to produce at least
//! one error and the harness checks that it does, so a golden of "no
//! diagnostics" can never be accepted for one by mistake; every other
//! fixture must parse clean.
//!
//! The AST text is the `Debug` form of the module with the three things
//! that would make it unreadable or non-deterministic rewritten:
//! `NodeId`s (a process-global counter, so they depend on which tests ran
//! first) are dropped, interned `Symbol`s are resolved to their names,
//! and spans are printed as `line:col-line:col`. Nothing else is filtered,
//! so a golden pins every field the parser fills in.
//!
//! To add a fixture, drop the `.nsl` file in `tests/parse/` and run the
//! test once: insta writes `<stem>.ast.snap.new` beside it. Review the
//! rendering, then rename it to `.ast.snap` (or use `cargo insta review`).
//! A changed golden shows up as a diff in the PR, which is the point.

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use nsl_errors::{Diagnostic, FileId, Level, Span};
use nsl_lexer::Interner;
use regex::Regex;

/// `tests/parse/` in this crate — fixtures and their goldens live together.
fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("parse")
}

/// Every `*.nsl` under [`fixture_dir`], sorted so failures list stably.
fn fixtures() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(fixture_dir())
        .expect("tests/parse/ exists")
        .map(|e| e.expect("readable entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "nsl"))
        .collect();
    out.sort();
    out
}

/// Lex + parse one fixture and render diagnostics followed by the AST.
fn render(source: &str) -> Rendered {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);

    let lines = LineIndex::new(source);
    let mut out = String::new();

    let total = lex_diags.len() + parsed.diagnostics.len();
    let errors = lex_diags
        .iter()
        .chain(&parsed.diagnostics)
        .filter(|d| d.level == Level::Error)
        .count();
    if total == 0 {
        out.push_str("diagnostics: none\n");
    } else {
        out.push_str(&format!("diagnostics: {total}\n"));
        for d in &lex_diags {
            out.push_str(&render_diagnostic("lex", d, &lines));
        }
        for d in &parsed.diagnostics {
            out.push_str(&render_diagnostic("parse", d, &lines));
        }
    }

    out.push_str("\nast:\n");
    out.push_str(&readable_debug(
        &format!("{:#?}", parsed.module),
        &interner,
        &lines,
    ));
    Rendered { text: out, errors }
}

struct Rendered {
    text: String,
    errors: usize,
}

fn render_diagnostic(stage: &str, d: &Diagnostic, lines: &LineIndex) -> String {
    let level = match d.level {
        Level::Error => "error",
        Level::Warning => "warning",
        Level::Info => "info",
    };
    let mut s = format!("  {stage} {level}: {}\n", d.message);
    for label in &d.labels {
        let style = match label.style {
            nsl_errors::LabelStyle::Primary => "-->",
            nsl_errors::LabelStyle::Secondary => "...",
        };
        s.push_str(&format!(
            "    {style} {} {}\n",
            lines.span(label.span),
            label.message
        ));
    }
    for note in &d.notes {
        s.push_str(&format!("    note: {note}\n"));
    }
    s
}

/// Byte offset → `line:col` (both 1-based, columns in chars).
struct LineIndex {
    /// Byte offset at which each line starts.
    starts: Vec<usize>,
    source: String,
}

impl LineIndex {
    fn new(source: &str) -> Self {
        let mut starts = vec![0];
        for (i, b) in source.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        Self {
            starts,
            source: source.to_string(),
        }
    }

    fn line_col(&self, offset: usize) -> (usize, usize) {
        let line = match self.starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.starts[line];
        let end = offset.min(self.source.len());
        let col = self.source[start..end].chars().count();
        (line + 1, col + 1)
    }

    fn span(&self, span: Span) -> String {
        if span == Span::DUMMY {
            return "<dummy>".to_string();
        }
        if span.file_id != FileId(0) {
            return format!("<file {}>", span.file_id.0);
        }
        let (l1, c1) = self.line_col(span.start.0 as usize);
        let (l2, c2) = self.line_col(span.end.0 as usize);
        format!("{l1}:{c1}-{l2}:{c2}")
    }
}

/// Rewrite the pretty `Debug` of an AST into the golden form described in
/// the module docs. Each regex matches both the single-line and the
/// multi-line (`{:#?}`) rendering of the type it targets.
fn readable_debug(debug: &str, interner: &Interner, lines: &LineIndex) -> String {
    // `Symbol(SymbolU32 { value: 7 })` → `` `name` ``. The map is built by
    // formatting each interned symbol the same way the AST does, so it
    // holds whatever the interner's Debug happens to print — the test
    // never assumes how a symbol index relates to the printed number.
    let sym_re = Regex::new(r"Symbol\(\s*SymbolU32 \{\s*value: (\d+),?\s*\},?\s*\)").unwrap();
    let mut names = std::collections::HashMap::new();
    for (sym, name) in interner.iter() {
        let printed = format!("{:?}", nsl_ast::Symbol(sym));
        let key = sym_re
            .captures(&printed)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| panic!("symbol Debug form changed: {printed}"));
        names.insert(key, name.to_string());
    }
    let text = sym_re.replace_all(debug, |caps: &regex::Captures| {
        let key = &caps[1];
        match names.get(key) {
            Some(name) => format!("`{name}`"),
            None => format!("`<symbol {key} not in interner>`"),
        }
    });

    // `Span { file_id: FileId(0), start: BytePos(4), end: BytePos(9) }`
    // → `1:5-1:10`.
    let span_re = Regex::new(
        r"Span \{\s*file_id: FileId\(\s*(\d+),?\s*\),\s*start: BytePos\(\s*(\d+),?\s*\),\s*end: BytePos\(\s*(\d+),?\s*\),?\s*\}",
    )
    .unwrap();
    let text = span_re.replace_all(&text, |caps: &regex::Captures| {
        let n = |i: usize| caps[i].parse::<u32>().expect("digits");
        let span = Span::new(
            FileId(n(1) as usize),
            nsl_errors::BytePos(n(2)),
            nsl_errors::BytePos(n(3)),
        );
        lines.span(span)
    });

    // Drop `id: NodeId(n),` lines entirely.
    let id_re = Regex::new(r"\n\s*id: NodeId\(\s*\d+,?\s*\),").unwrap();
    let text = id_re.replace_all(&text, "");

    // Pretty Debug puts even a lone scalar on its own line:
    // `Int(\n    1,\n)` → `Int(1)`. Collapsing any parenthesised group whose
    // whole content is one line, until nothing changes, folds
    // `Some(\n    Int(\n        1,\n    ),\n)` into `Some(Int(1))` while
    // leaving multi-field structs and lists laid out as before.
    let one_line_re = Regex::new(r"\(\n\s*([^\n]+?),?\n\s*\)").unwrap();
    let mut text = text.into_owned();
    loop {
        let next = one_line_re.replace_all(&text, "($1)").into_owned();
        if next == text {
            break;
        }
        text = next;
    }
    text
}

#[test]
fn goldens() {
    let fixtures = fixtures();
    assert!(
        fixtures.len() >= 10,
        "tests/parse/ holds {} fixtures; the corpus was larger — was it moved?",
        fixtures.len()
    );

    // A golden with no fixture is a stale file nobody will ever re-check.
    let stems: std::collections::HashSet<String> = fixtures
        .iter()
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    let orphans: Vec<String> = fs::read_dir(fixture_dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".ast.snap"))
        .filter(|n| !stems.contains(n.trim_end_matches(".ast.snap")))
        .collect();
    assert!(orphans.is_empty(), "goldens without a fixture: {orphans:?}");

    let mut failed = Vec::new();
    for path in &fixtures {
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let source = fs::read_to_string(path).unwrap();
        let rendered = render(&source);

        // The prefix is the contract: an `err_` fixture that parses clean
        // has stopped testing recovery, and a clean fixture that errors
        // is a regression even if its golden were updated to match.
        let expects_error = stem.starts_with("err_");
        if expects_error != (rendered.errors > 0) {
            let diagnostics = rendered.text.split("\nast:\n").next().unwrap_or("");
            eprintln!(
                "{stem}: {} error diagnostics, but the `err_` prefix says {}\n{diagnostics}",
                rendered.errors,
                if expects_error { "at least one" } else { "none" }
            );
            failed.push(stem);
            continue;
        }

        let mut settings = insta::Settings::clone_current();
        settings.set_snapshot_path(fixture_dir());
        settings.set_prepend_module_to_snapshot(false);
        settings.set_omit_expression(true);
        settings.set_input_file(path);
        let ok = settings.bind(|| {
            catch_unwind(AssertUnwindSafe(|| {
                insta::assert_snapshot!(format!("{stem}.ast"), rendered.text);
            }))
            .is_ok()
        });
        if !ok {
            failed.push(stem);
        }
    }
    assert!(
        failed.is_empty(),
        "{} of {} goldens mismatched: {}",
        failed.len(),
        fixtures.len(),
        failed.join(", ")
    );
}
