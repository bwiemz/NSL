//! The parser refuses nesting deeper than `MAX_NESTING` instead of
//! overflowing its stack.
//!
//! The parse fuzz target found that `let x = ((((…))))` two thousand deep
//! crashes the parser (a plain recursive descent, one stack frame per
//! level): a SIGSEGV from `nsl check`, not a diagnostic. These tests run
//! on `cargo test`'s 2 MB threads with the debug build, the tightest
//! stack the parser sees, so a passing "at the limit" case is evidence the
//! limit fits and a passing "over the limit" case is evidence the guard
//! fires before the stack runs out — without the guard the whole test
//! process aborts.

use nsl_ast::stmt::StmtKind;
use nsl_ast::Module;
use nsl_errors::{Diagnostic, FileId};
use nsl_lexer::Interner;
use nsl_parser::parser::MAX_NESTING;

fn parse(source: &str) -> (Module, Vec<Diagnostic>) {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, FileId(1), &mut interner);
    assert!(lex_diags.is_empty(), "lexer: {lex_diags:?}");
    let result = nsl_parser::parse(&tokens, &mut interner);
    (result.module, result.diagnostics)
}

fn messages(diags: &[Diagnostic]) -> Vec<&str> {
    diags.iter().map(|d| d.message.as_str()).collect()
}

/// One way of nesting: `units` repetitions of a construct around an
/// innermost `1` (or `pass`), and the nesting levels — statement,
/// expression, block, type or pattern, outermost first — the parser
/// enters to reach that innermost point, so a test knows both how deep a
/// source nests and which construct sits at any given level.
struct Shape {
    name: &'static str,
    source: fn(usize) -> String,
    /// The levels the parser enters for `units` repetitions.
    levels: fn(usize) -> Vec<&'static str>,
}

/// `units` repetitions of `line` at increasing indentation, then `pass`.
fn indented(units: usize, line: &str) -> String {
    let mut out = String::new();
    for i in 0..units {
        out.push_str(&"    ".repeat(i));
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&"    ".repeat(units));
    out.push_str("pass\n");
    out
}

/// `units` nested `match` statements, each with one `case _:` arm.
fn matches(units: usize) -> String {
    let mut out = String::new();
    for i in 0..units {
        out.push_str(&"    ".repeat(2 * i));
        out.push_str("match x:\n");
        out.push_str(&"    ".repeat(2 * i + 1));
        out.push_str("case _:\n");
    }
    out.push_str(&"    ".repeat(2 * units));
    out.push_str("pass\n");
    out
}

/// The statement, then `units + 1` levels of `what`: the `let` value (or
/// annotation, or pattern) and then one per repetition, the innermost
/// being the `1`/`int`/`a`.
fn statement_then(what: &'static str, units: usize) -> Vec<&'static str> {
    let mut levels = vec!["statement"];
    levels.extend(std::iter::repeat_n(what, units + 1));
    levels
}

/// `cycle` per repetition, then the innermost `pass` — a statement holding
/// an expression.
fn repeated_then_pass(cycle: &[&'static str], units: usize) -> Vec<&'static str> {
    let mut levels: Vec<&'static str> = cycle.iter().copied().cycle().take(cycle.len() * units).collect();
    levels.extend(["statement", "expression"]);
    levels
}

fn shapes() -> Vec<Shape> {
    vec![
        Shape {
            name: "parens",
            source: |n| format!("let x = {}1{}\n", "(".repeat(n), ")".repeat(n)),
            levels: |n| statement_then("expression", n),
        },
        Shape {
            name: "lists",
            source: |n| format!("let x = {}1{}\n", "[".repeat(n), "]".repeat(n)),
            levels: |n| statement_then("expression", n),
        },
        Shape {
            name: "unary minus",
            source: |n| format!("let x = {}1\n", "-".repeat(n)),
            levels: |n| statement_then("expression", n),
        },
        Shape {
            name: "lambdas",
            source: |n| format!("let x = {}1\n", "|a| ".repeat(n)),
            levels: |n| statement_then("expression", n),
        },
        Shape {
            name: "power (right-associative)",
            source: |n| format!("let x = {}1\n", "2 ** ".repeat(n)),
            levels: |n| statement_then("expression", n),
        },
        Shape {
            name: "f-strings",
            source: |n| format!("let x = {}1{}\n", "f\"{".repeat(n), "}\"".repeat(n)),
            levels: |n| statement_then("expression", n),
        },
        Shape {
            name: "array types",
            source: |n| format!("let x: {}int{} = 1\n", "[".repeat(n), "; 1]".repeat(n)),
            levels: |n| statement_then("type", n),
        },
        Shape {
            name: "tuple types",
            source: |n| format!("let x: {}int{} = 1\n", "(".repeat(n), ")".repeat(n)),
            levels: |n| statement_then("type", n),
        },
        Shape {
            name: "tuple patterns",
            source: |n| format!("let {}a{} = 1\n", "(".repeat(n), ")".repeat(n)),
            levels: |n| statement_then("pattern", n),
        },
        // Statement-level recursion. `pub` and `train:` bodies re-enter
        // the statement parser without a block in between; a nested
        // `if`/`match` costs a statement and a block per level; a nested
        // `pub fn` costs the `pub` statement, the `fn` statement and its
        // block.
        Shape {
            name: "pub prefixes",
            source: |n| format!("{}pass\n", "pub ".repeat(n)),
            levels: |n| repeated_then_pass(&["statement"], n),
        },
        Shape {
            name: "train bodies",
            source: |n| indented(n, "train:"),
            levels: |n| repeated_then_pass(&["statement"], n),
        },
        Shape {
            name: "if blocks",
            source: |n| indented(n, "if b:"),
            levels: |n| repeated_then_pass(&["statement", "block"], n),
        },
        Shape {
            name: "match arms",
            source: matches,
            levels: |n| repeated_then_pass(&["statement", "block"], n),
        },
        Shape {
            name: "pub fn bodies",
            source: |n| indented(n, "pub fn f():"),
            levels: |n| repeated_then_pass(&["statement", "statement", "block"], n),
        },
    ]
}

/// The most repetitions of `shape` that nest no deeper than the limit.
///
/// A shape whose repetition costs more than one level (a nested `if` is
/// two, a nested `pub fn` three) may not land exactly on the limit; it
/// lands within one repetition of it, which is what the tests need.
fn units_at_the_limit(shape: &Shape) -> usize {
    let mut units = 0;
    while (shape.levels)(units + 1).len() <= MAX_NESTING as usize {
        units += 1;
    }
    units
}

#[test]
fn every_shape_parses_clean_at_the_limit() {
    for shape in shapes() {
        let units = units_at_the_limit(&shape);
        let depth = (shape.levels)(units).len();
        let (_, diags) = parse(&(shape.source)(units));
        assert!(
            diags.is_empty(),
            "{} at {depth} levels ({units} units): {:?}",
            shape.name,
            messages(&diags)
        );
    }
}

#[test]
fn one_unit_past_the_limit_is_refused_with_one_diagnostic_naming_the_construct() {
    for shape in shapes() {
        let units = units_at_the_limit(&shape) + 1;
        let levels = (shape.levels)(units);
        // The construct the parser is entering when it reaches one level
        // past the limit is the one the diagnostic names.
        let construct = levels[MAX_NESTING as usize];
        let (_, diags) = parse(&(shape.source)(units));
        let msgs = messages(&diags);
        assert_eq!(
            msgs,
            vec![format!("{construct} nested more than {MAX_NESTING} levels deep")],
            "{} at {} levels ({units} units)",
            shape.name,
            levels.len()
        );
    }
}

#[test]
fn the_fuzzer_found_depth_is_refused_not_crashed() {
    // The depths the fuzz target crashed at; a fixed 2000 rather than a
    // multiple of the limit so a change to `MAX_NESTING` cannot hide it.
    for shape in shapes() {
        let (_, diags) = parse(&(shape.source)(2000));
        let msgs = messages(&diags);
        assert_eq!(msgs.len(), 1, "{}: {} diagnostics", shape.name, msgs.len());
        assert!(msgs[0].contains("nested more than"), "{}: {msgs:?}", shape.name);
    }
}

fn deep(name: &str) -> String {
    let shape = shapes().into_iter().find(|s| s.name == name).unwrap();
    (shape.source)(2000)
}

#[test]
fn errors_before_the_limit_are_kept_and_the_unwinding_cascade_is_dropped() {
    // Whatever line 1 reports on its own, then a too-deep expression on
    // line 2: the report is line 1's errors plus the limit, nothing else.
    let bad_line = "let = 1\n";
    let (_, line1_diags) = parse(bad_line);
    let mut expected = messages(&line1_diags);
    assert!(!expected.is_empty(), "the first line must be an error on its own");
    let limit = format!("expression nested more than {MAX_NESTING} levels deep");
    expected.push(&limit);

    let source = format!("{bad_line}{}", deep("parens"));
    let (_, diags) = parse(&source);
    assert_eq!(messages(&diags), expected);
    let label = &diags.last().unwrap().labels[0];
    let line = source[..label.span.start.0 as usize].matches('\n').count() + 1;
    assert_eq!(line, 2, "the limit is reported where it was hit");
}

#[test]
fn nothing_after_the_limit_is_parsed() {
    // The module holds the good statement and the cut-off one, and
    // nothing for the bad statement after the deep one: the parser
    // stopped before reaching it (that statement would otherwise report,
    // and the diagnostic count checks that too).
    let source = format!("let a = 1\n{}let = 2\n", deep("lists"));
    let (module, diags) = parse(&source);
    assert_eq!(messages(&diags).len(), 1, "{:?}", messages(&diags));
    assert_eq!(module.stmts.len(), 2, "the good statement and the cut-off one");
    assert!(matches!(module.stmts[0].kind, StmtKind::VarDecl { .. }));
}

#[test]
fn the_limit_stops_the_parser_on_the_input_not_past_it() {
    // Past the last token is the parser's end-of-input sentinel, whose
    // span is in file 0; merging it with a span from another file panics.
    // Every imported module is a file other than 0, so the limit (and the
    // unwinding after it) must leave every span in the input's file.
    for shape in shapes() {
        let source = (shape.source)(2000);
        let (module, diags) = parse(&source);
        for d in &diags {
            for label in &d.labels {
                assert_eq!(label.span.file_id, FileId(1), "{}: {d:?}", shape.name);
            }
        }
        for stmt in &module.stmts {
            assert_eq!(stmt.span.file_id, FileId(1), "{}: {:?}", shape.name, stmt.span);
        }
    }
}
