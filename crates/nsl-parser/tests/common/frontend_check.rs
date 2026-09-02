//! The invariants the frontend must hold on *any* input — what the fuzz
//! targets in `fuzz/` check on every generated input, and what
//! `tests/fuzz_seeds.rs` checks over the committed seed corpus on the
//! stable toolchain so the invariants stay exercised in CI.
//!
//! This file is compiled in both places by `#[path]` so there is one
//! definition. It must stay free of test-only and nightly-only
//! dependencies.

use nsl_errors::{Diagnostic, FileId, Span};
use nsl_lexer::{Interner, Token, TokenKind};

/// The file id the harness lexes with.
///
/// Not `FileId(0)`: that is the file id of `Span::DUMMY`, the span of the
/// parser's end-of-input sentinel, and `Span::merge` panics on spans from
/// different files. Lexing as file 0 would let a sentinel span merge
/// silently with the input's; any other id — which is what every imported
/// module gets — makes the parser panic instead, so the harness sees it.
const FILE: FileId = FileId(1);

/// Everything the lexer must hold for `source`: no panic, every span in
/// bounds and on char boundaries, token starts non-decreasing, exactly
/// one `Eof` and it comes last, every diagnostic label in bounds.
pub fn lex(source: &str) -> Vec<Token> {
    lex_into(source, &mut Interner::new())
}

fn lex_into(source: &str, interner: &mut Interner) -> Vec<Token> {
    let (tokens, diagnostics) = nsl_lexer::tokenize(source, FILE, interner);

    assert!(
        matches!(tokens.last().map(|t| &t.kind), Some(TokenKind::Eof)),
        "last token is not Eof: {:?}",
        tokens.last()
    );
    let eofs = tokens
        .iter()
        .filter(|t| matches!(t.kind, TokenKind::Eof))
        .count();
    assert_eq!(eofs, 1, "{eofs} Eof tokens");

    let mut last_start = 0usize;
    for t in &tokens {
        check_span(source, t.span, &format!("token {:?}", t.kind));
        let start = t.span.start.0 as usize;
        assert!(
            start >= last_start,
            "token {:?} starts at {start}, before the previous token's start {last_start}",
            t.kind
        );
        last_start = start;
    }
    check_diagnostics(source, &diagnostics);
    tokens
}

/// Everything the parser must hold for `source`: the lexer invariants,
/// then no panic and every diagnostic label in the input file and in
/// bounds.
///
/// The tree itself is not walked: a walk recurses as deep as the tree,
/// and a flat chain (`a + b + c + …`) builds one as deep as it is long,
/// which the parser's nesting limit does not bound. Dropping the tree
/// recurses the same way, so the fuzz targets keep `-max_len` at 4096:
/// a crash on a longer input is that known follow-up, not a parser bug.
pub fn parse(source: &str) {
    let mut interner = Interner::new();
    let tokens = lex_into(source, &mut interner);
    let result = nsl_parser::parse(&tokens, &mut interner);
    check_diagnostics(source, &result.diagnostics);
}

fn check_diagnostics(source: &str, diagnostics: &[Diagnostic]) {
    for d in diagnostics {
        for label in &d.labels {
            let what = format!("label of {:?}", d.message);
            assert_eq!(
                label.span.file_id, FILE,
                "{what}: span is in file {:?}, not the input",
                label.span.file_id
            );
            check_span(source, label.span, &what);
        }
    }
}

fn check_span(source: &str, span: Span, what: &str) {
    let (start, end) = (span.start.0 as usize, span.end.0 as usize);
    assert!(start <= end, "{what}: span {start}..{end} runs backwards");
    assert!(
        end <= source.len(),
        "{what}: span {start}..{end} past the end of {} bytes",
        source.len()
    );
    assert!(
        source.is_char_boundary(start) && source.is_char_boundary(end),
        "{what}: span {start}..{end} splits a character"
    );
}
