//! Golden-file lexer tests.
//!
//! Every `tests/lex/*.nsl` is tokenized and the result — diagnostics
//! first, then one line per token as `line:col-line:col  Kind` — is
//! compared against `tests/lex/<stem>.tokens.snap` with insta. A fixture
//! whose stem starts with `err_` must produce at least one error-level
//! diagnostic; every other fixture must produce none (a warning-only
//! fixture would be a plain golden, as in the parser harness). Identifiers print their text
//! rather than their interner index, so the goldens do not depend on
//! interning order.
//!
//! The token stream is what the parser sees, so these goldens pin the
//! things the parser goldens (`crates/nsl-parser/tests/parse/`) only
//! show indirectly: where INDENT/DEDENT/NEWLINE are synthesized, how
//! brackets and `\` suppress newlines, how f-strings are split, and the
//! exact value each literal lexes to.
//!
//! To add a fixture, drop the `.nsl` in `tests/lex/` and run the test
//! once: insta writes `<stem>.tokens.snap.new` beside it. Review it, then
//! rename it to `.tokens.snap` (or use `cargo insta review`).

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use nsl_errors::{Diagnostic, FileId, Level, Span};
use nsl_lexer::{Interner, TokenKind};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("lex")
}

fn fixtures() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(fixture_dir())
        .expect("tests/lex/ exists")
        .map(|e| e.expect("readable entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "nsl"))
        .collect();
    out.sort();
    out
}

struct Rendered {
    text: String,
    errors: usize,
}

fn render(source: &str) -> Rendered {
    let mut interner = Interner::new();
    let (tokens, diags) = nsl_lexer::tokenize(source, FileId(0), &mut interner);
    let lines = LineIndex::new(source);

    let mut out = String::new();
    if diags.is_empty() {
        out.push_str("diagnostics: none\n");
    } else {
        out.push_str(&format!("diagnostics: {}\n", diags.len()));
        for d in &diags {
            out.push_str(&render_diagnostic(d, &lines));
        }
    }

    out.push_str("\ntokens:\n");
    for t in &tokens {
        let kind = match &t.kind {
            TokenKind::Ident(sym) => format!("Ident({})", interner.resolve(*sym).unwrap()),
            other => format!("{other:?}"),
        };
        out.push_str(&format!("{:<12} {kind}\n", lines.span(t.span)));
    }
    Rendered {
        text: out,
        errors: diags.iter().filter(|d| d.level == Level::Error).count(),
    }
}

fn render_diagnostic(d: &Diagnostic, lines: &LineIndex) -> String {
    let level = match d.level {
        Level::Error => "error",
        Level::Warning => "warning",
        Level::Info => "info",
    };
    let mut s = format!("  {level}: {}\n", d.message);
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
        let (l1, c1) = self.line_col(span.start.0 as usize);
        let (l2, c2) = self.line_col(span.end.0 as usize);
        format!("{l1}:{c1}-{l2}:{c2}")
    }
}

#[test]
fn goldens() {
    let fixtures = fixtures();
    assert!(
        fixtures.len() >= 5,
        "tests/lex/ holds {} fixtures; the corpus was larger — was it moved?",
        fixtures.len()
    );

    let stems: std::collections::HashSet<String> = fixtures
        .iter()
        .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
        .collect();
    let orphans: Vec<String> = fs::read_dir(fixture_dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tokens.snap"))
        .filter(|n| !stems.contains(n.trim_end_matches(".tokens.snap")))
        .collect();
    assert!(orphans.is_empty(), "goldens without a fixture: {orphans:?}");

    let mut failed = Vec::new();
    for path in &fixtures {
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let source = fs::read_to_string(path).unwrap();
        let rendered = render(&source);

        let expects_error = stem.starts_with("err_");
        if expects_error != (rendered.errors > 0) {
            let diagnostics = rendered.text.split("\ntokens:\n").next().unwrap_or("");
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
                insta::assert_snapshot!(format!("{stem}.tokens"), rendered.text);
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
