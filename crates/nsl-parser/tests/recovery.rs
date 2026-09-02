//! Error recovery: a file with several independent errors must report
//! every one of them, and the statements around them must still reach
//! the AST. The fixtures are shared with the goldens in `tests/parse/`
//! (their `.ast.snap` shows the full picture); these tests state the
//! contract explicitly so a regenerated golden cannot quietly weaken it.

use std::path::Path;

use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_errors::{Diagnostic, FileId, Level};
use nsl_lexer::Interner;

struct Parsed {
    lex_diags: Vec<Diagnostic>,
    parse_diags: Vec<Diagnostic>,
    stmts: Vec<Stmt>,
    interner: Interner,
    source: String,
}

fn parse_fixture(name: &str) -> Parsed {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parse")
        .join(name);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(&source, FileId(0), &mut interner);
    let result = nsl_parser::parse(&tokens, &mut interner);
    Parsed {
        lex_diags,
        parse_diags: result.diagnostics,
        stmts: result.module.stmts,
        interner,
        source,
    }
}

impl Parsed {
    /// 1-based line of a diagnostic's first label.
    fn line_of(&self, d: &Diagnostic) -> usize {
        let start = d.labels.first().expect("labelled diagnostic").span.start.0 as usize;
        self.source[..start].matches('\n').count() + 1
    }

    /// `(message, line)` for every diagnostic, in emission order.
    fn parse_summary(&self) -> Vec<(String, usize)> {
        self.parse_diags
            .iter()
            .map(|d| (d.message.clone(), self.line_of(d)))
            .collect()
    }

    /// The bound name of a `let` statement, or a description of what the
    /// statement is instead.
    fn let_name(&self, stmt: &Stmt) -> String {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, .. } => match &pattern.kind {
                PatternKind::Ident(sym) => self.interner.resolve(sym.0).unwrap().to_string(),
                other => format!("<let with {other:?} pattern>"),
            },
            other => format!("<{}>", stmt_kind_name(other)),
        }
    }
}

fn stmt_kind_name(kind: &StmtKind) -> String {
    let debug = format!("{kind:?}");
    debug
        .split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap()
        .to_string()
}

#[test]
fn three_independent_errors_are_all_reported_and_neighbours_survive() {
    let p = parse_fixture("err_three_independent.nsl");
    assert!(p.lex_diags.is_empty(), "lexer: {:?}", p.lex_diags);

    // One diagnostic per error, plus the one follow-up the `let mut` line
    // earns (the parser reports the stray identifier after the failed
    // pattern before it resynchronises). Every line with an error is
    // named; no error is dropped and no clean line is blamed.
    assert_eq!(
        p.parse_summary(),
        vec![
            ("expected newline or end of statement, found identifier".to_string(), 6),
            ("expected :, found newline".to_string(), 8),
            ("expected pattern, found Mut".to_string(), 11),
            ("expected newline or end of statement, found identifier".to_string(), 11),
        ]
    );
    assert!(p.parse_diags.iter().all(|d| d.level == Level::Error));

    // Every top-level statement is still there, in order: the erroring
    // lines become a truncated `let b`, an `if` whose body is attached,
    // and a placeholder `let` — none of them swallow a neighbour.
    let names: Vec<String> = p.stmts.iter().map(|s| p.let_name(s)).collect();
    assert_eq!(
        names,
        vec![
            "a",
            "b",
            "c",
            "<If>",
            "d",
            "<let with Wildcard pattern>",
            "f",
        ]
    );

    let StmtKind::If { then_block, .. } = &p.stmts[3].kind else {
        unreachable!()
    };
    assert_eq!(
        then_block.stmts.len(),
        1,
        "the `if` header lost its colon but its body must still be its own"
    );
}

#[test]
fn lexer_errors_become_error_tokens_and_the_parser_carries_on() {
    let p = parse_fixture("err_lexer.nsl");
    let lex: Vec<(String, usize)> = p
        .lex_diags
        .iter()
        .map(|d| (d.message.clone(), p.line_of(d)))
        .collect();
    assert_eq!(
        lex,
        vec![
            ("unexpected character: '$'".to_string(), 4),
            ("unterminated string literal".to_string(), 5),
            ("expected hex digits after '0x'".to_string(), 6),
        ]
    );
    // Each error token is reported once more by the parser, at the same
    // line, as the thing it found where an expression should be.
    let parse_lines: Vec<usize> = p.parse_diags.iter().map(|d| p.line_of(d)).collect();
    assert_eq!(parse_lines, vec![4, 5, 6]);

    let names: Vec<String> = p.stmts.iter().map(|s| p.let_name(s)).collect();
    assert_eq!(names.first().map(String::as_str), Some("a"));
    assert_eq!(
        names.last().map(String::as_str),
        Some("e"),
        "the statement after the last bad token must parse: {names:?}"
    );
}

/// Documents today's behaviour rather than the ideal: an unclosed paren
/// suppresses newlines in the lexer, so the next `let` is consumed as
/// part of the expression, and a trailing binary operator continues onto
/// the following line. If recovery improves, this test — and the
/// `err_cascades` golden and its header comment — are what to update.
#[test]
fn unclosed_paren_and_trailing_operator_swallow_the_next_statement_today() {
    let p = parse_fixture("err_cascades.nsl");
    let names: Vec<String> = p.stmts.iter().map(|s| p.let_name(s)).collect();
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"b".to_string()));
    assert!(
        !names.contains(&"c".to_string()),
        "`let c` after an unclosed paren reached the AST — recovery improved; update err_cascades"
    );
    assert!(
        !names.contains(&"f".to_string()),
        "`let f` after a trailing `+` reached the AST — recovery improved; update err_cascades"
    );
    // The fn-header cascade nests everything after it, so `g` is a
    // statement of `broken`, not of the module.
    assert!(
        !names.contains(&"g".to_string()),
        "`let g` after a colon-less fn header reached the module — recovery improved; update err_cascades"
    );
    assert_eq!(names.last().map(String::as_str), Some("<FnDef>"), "{names:?}");
    // The three errors are still all reported, at their own lines.
    let lines: Vec<usize> = p.parse_diags.iter().map(|d| p.line_of(d)).collect();
    assert!(lines.contains(&10), "unclosed paren blamed at the line it swallowed: {lines:?}");
    assert!(lines.contains(&13), "trailing operator blamed at the line it swallowed: {lines:?}");
    assert!(lines.contains(&15), "missing colon on the fn header: {lines:?}");
}
