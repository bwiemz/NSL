//! `nsl fmt` must not change what a program means.
//!
//! For every `.nsl` the repository tracks (`CORPUS_DIRS`) that parses
//! clean, formatting it must (1) succeed, (2) still parse clean, (3)
//! produce the same AST — compared modulo spans and node ids, since moving
//! text around is the formatter's job — and (4) be a fixed point:
//! formatting the output again changes nothing.
//!
//! The corpus is already formatted, so on the files as committed the
//! formatter is a no-op and (3) would hold trivially. Each file is therefore
//! also checked in three de-formatted spellings that keep its token stream
//! — every one-space gap between two tokens on a line widened to three
//! spaces, every such gap the lexer allows squeezed out, and every plain
//! double-quoted literal flipped to single quotes — which the formatter has
//! real work to do on. The number of inputs it actually changes under those
//! spellings has a floor, so the property cannot go vacuous again
//! unnoticed.
//!
//! A file that does not parse clean to begin with is skipped (the property
//! says nothing about broken input), but the set of skipped files is pinned
//! exactly, so a file that stops parsing fails this test rather than
//! dropping out of it.
//!
//! The AST comparison is on the pretty `Debug` of the module with `Span`s
//! and `NodeId`s erased and interned `Symbol`s resolved to their names, so
//! both sides can use their own interner.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use nsl_cli::formatter::format_source;
use nsl_errors::FileId;
use nsl_lexer::{Interner, Token, TokenKind};
use regex::Regex;

/// Files in the corpus the frontend rejects today, with the reason. A new
/// entry needs a reason; a stale entry (the file parses again) fails too.
const EXPECTED_SKIPS: &[(&str, &str)] = &[
    (
        "examples/bitnet_b158_inference.nsl",
        "a `= nsl.quant...` continuation line the parser does not accept",
    ),
    (
        "examples/gpt2.nsl",
        "`quant(scheme = int8, ...)` where the grammar expects `static`",
    ),
    (
        "examples/m56_device_transfer_opt_in.nsl",
        "uses `tokenizer` (a keyword) as a name",
    ),
    (
        "models/coder1b/pretrain_1b2048.nsl",
        "a template with a placeholder line",
    ),
    (
        "crates/nsl-cli/tests/fixtures/sdpa_causal_contract.nsl",
        "a template with a `CAUSAL_SPELLING` placeholder",
    ),
    // Lexer goldens: token soup and a deliberate lexer error.
    ("crates/nsl-lexer/tests/lex/err_lexer.nsl", "lexer golden: a tab indent"),
    ("crates/nsl-lexer/tests/lex/keywords.nsl", "lexer golden: bare keywords"),
    ("crates/nsl-lexer/tests/lex/operators.nsl", "lexer golden: bare operators"),
    // Parser error-recovery goldens: each is a deliberate error.
    ("crates/nsl-parser/tests/parse/err_cascades.nsl", "error-recovery golden"),
    ("crates/nsl-parser/tests/parse/err_indentation.nsl", "error-recovery golden"),
    ("crates/nsl-parser/tests/parse/err_let_mut.nsl", "error-recovery golden"),
    ("crates/nsl-parser/tests/parse/err_lexer.nsl", "error-recovery golden"),
    ("crates/nsl-parser/tests/parse/err_three_independent.nsl", "error-recovery golden"),
    ("crates/nsl-parser/tests/parse/err_unexpected_tokens.nsl", "error-recovery golden"),
];

/// Every directory holding a tracked `.nsl`, relative to the workspace
/// root. A directory that goes missing fails the test rather than
/// silently shrinking the corpus.
const CORPUS_DIRS: &[&str] = &[
    "benchmarks",
    "examples",
    "models",
    "python",
    "stdlib",
    "tests",
    "crates/nsl-cli/tests/differential_scripts",
    "crates/nsl-cli/tests/fixtures",
    "crates/nsl-codegen/tests/fixtures",
    "crates/nsl-lexer/tests/lex",
    "crates/nsl-parser/tests/parse",
    "crates/nsl-test/fixtures",
];

/// Below this many inputs of a spelling changed by `fmt`, the AST check is
/// not testing that spelling's formatter phases (the test prints the
/// counts; each floor is well under its count).
const CHANGED_FLOOR: &[(&str, usize)] = &[("widened", 300), ("squeezed", 300), ("requoted", 100)];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/nsl-cli sits two levels below the workspace root")
        .to_path_buf()
}

/// Every `.nsl` under `dir`, recursively, sorted.
fn nsl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("corpus directory {}: {e}", dir.display()));
    let mut entries: Vec<PathBuf> = entries.map(|e| e.unwrap().path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            nsl_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "nsl") {
            out.push(path);
        }
    }
}

fn corpus() -> Vec<PathBuf> {
    let root = workspace_root();
    let mut files = Vec::new();
    for dir in CORPUS_DIRS {
        nsl_files(&root.join(dir), &mut files);
    }
    files
}

/// The AST of `source`, or the first diagnostic (with its line) if lexing
/// or parsing reported anything.
fn clean_ast(source: &str) -> Result<String, String> {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, FileId(0), &mut interner);
    if let Some(d) = lex_diags.first() {
        return Err(describe(d, source));
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if let Some(d) = parsed.diagnostics.first() {
        return Err(describe(d, source));
    }
    Ok(scrub(&format!("{:#?}", parsed.module), &interner))
}

/// `line N: message — <that line of source>` for a failure report.
fn describe(d: &nsl_errors::Diagnostic, source: &str) -> String {
    let Some(label) = d.labels.first() else {
        return d.message.clone();
    };
    let start = (label.span.start.0 as usize).min(source.len());
    let line_no = source[..start].matches('\n').count();
    let line = source.lines().nth(line_no).unwrap_or("");
    format!("line {}: {} — {line:?}", line_no + 1, d.message)
}

/// Erase spans and node ids; resolve symbols to names.
fn scrub(debug: &str, interner: &Interner) -> String {
    let sym_re = Regex::new(r"Symbol\(\s*SymbolU32 \{\s*value: (\d+),?\s*\},?\s*\)").unwrap();
    let mut names = std::collections::HashMap::new();
    for (sym, name) in interner.iter() {
        let printed = format!("{:?}", nsl_ast::Symbol(sym));
        let key = sym_re.captures(&printed).expect("symbol Debug form")[1].to_string();
        names.insert(key, name.to_string());
    }
    let text = sym_re.replace_all(debug, |caps: &regex::Captures| {
        format!("`{}`", names.get(&caps[1]).map_or("?", String::as_str))
    });
    let span_re = Regex::new(
        r"Span \{\s*file_id: FileId\(\s*\d+,?\s*\),\s*start: BytePos\(\s*\d+,?\s*\),\s*end: BytePos\(\s*\d+,?\s*\),?\s*\}",
    )
    .unwrap();
    let text = span_re.replace_all(&text, "<span>");
    let id_re = Regex::new(r"NodeId\(\s*\d+,?\s*\)").unwrap();
    let text = id_re.replace_all(&text, "<id>").into_owned();
    // If the Debug forms drift, the regexes stop matching and every
    // comparison would silently pass on raw positions.
    for raw in ["BytePos(", "NodeId(", "SymbolU32"] {
        assert!(!text.contains(raw), "scrub left a {raw} in the AST text");
    }
    text
}

/// First line that differs between two texts, for the failure report.
fn first_difference(a: &str, b: &str) -> String {
    for (i, (x, y)) in a.lines().zip(b.lines()).enumerate() {
        if x != y {
            return format!("line {}: {x:?} vs {y:?}", i + 1);
        }
    }
    format!("lengths differ ({} vs {} lines)", a.lines().count(), b.lines().count())
}

/// The byte ranges of every gap between two tokens on one line that is
/// exactly one space, in source order.
fn one_space_gaps(source: &str, tokens: &[Token]) -> Vec<(usize, usize)> {
    let mut gaps = Vec::new();
    let mut prev_end: Option<usize> = None;
    for tok in tokens {
        let start = tok.span.start.0 as usize;
        let end = tok.span.end.0 as usize;
        match tok.kind {
            TokenKind::Newline
            | TokenKind::Indent
            | TokenKind::Dedent
            | TokenKind::Eof
            | TokenKind::DocComment(_) => {
                prev_end = None;
                continue;
            }
            _ => {}
        }
        // An `FStringEnd` after trailing text shares the text's span, so
        // its start can precede the previous end; `get` refuses that.
        if let Some(p) = prev_end
            && source.get(p..start) == Some(" ")
        {
            gaps.push((p, start));
        }
        prev_end = Some(end.max(start));
    }
    gaps
}

/// `source` with each gap in `gaps` replaced by `filler`.
fn refill(source: &str, gaps: &[(usize, usize)], filler: &str) -> String {
    let mut out = String::with_capacity(source.len() + gaps.len() * filler.len());
    let mut copied = 0;
    for &(start, end) in gaps {
        out.push_str(&source[copied..start]);
        out.push_str(filler);
        copied = end;
    }
    out.push_str(&source[copied..]);
    out
}

fn kinds(tokens: &[Token]) -> Vec<&TokenKind> {
    tokens.iter().map(|t| &t.kind).collect()
}

/// `source` with every plain double-quoted string literal — outside any
/// f-string, single-quote-character quotes, body free of quotes,
/// backslashes and newlines — written with single quotes instead, so the
/// quote normalisation has something to do (the committed corpus uses
/// double quotes throughout).
fn requote_plain_literals(source: &str, tokens: &[Token]) -> String {
    let mut out = source.to_string();
    let mut fstring_depth = 0usize;
    for tok in tokens {
        match tok.kind {
            TokenKind::FStringStart => fstring_depth += 1,
            TokenKind::FStringEnd => fstring_depth = fstring_depth.saturating_sub(1),
            TokenKind::StringLiteral(_) if fstring_depth == 0 => {
                let (start, end) = (tok.span.start.0 as usize, tok.span.end.0 as usize);
                let raw = &source[start..end];
                let plain = raw.len() >= 2
                    && raw.starts_with('"')
                    && !raw.starts_with("\"\"\"")
                    && !raw[1..raw.len() - 1].contains(['"', '\'', '\\', '\n']);
                if plain {
                    out.replace_range(start..start + 1, "'");
                    out.replace_range(end - 1..end, "'");
                }
            }
            _ => {}
        }
    }
    out
}

/// De-formatted spellings of `source` that lex to the same token stream:
/// every one-space gap widened to three spaces, every one the lexer allows
/// squeezed out, and plain literals requoted. A spelling the lexer reads
/// differently is dropped; `CHANGED_FLOOR` catches that happening
/// wholesale.
fn perturbations(source: &str) -> Vec<(&'static str, String)> {
    let mut interner = Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(source, FileId(0), &mut interner);
    let gaps = one_space_gaps(source, &tokens);

    // A gap whose neighbours would fuse into one token (`let x`, `< =`,
    // `- -`) has to stay; drop those first, then let the lexer be the judge.
    let sticky = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let operator = |b: u8| b"+-*/<>=!|.&%@:".contains(&b);
    let bytes = source.as_bytes();
    let removable: Vec<(usize, usize)> = gaps
        .iter()
        .copied()
        .filter(|&(start, end)| {
            let (before, after) = (bytes[start - 1], bytes[end]);
            let fuses = (sticky(before) && sticky(after)) || (operator(before) && operator(after));
            !fuses
        })
        .collect();

    let candidates = [
        ("widened", refill(source, &gaps, "   ")),
        ("squeezed", refill(source, &removable, "")),
        ("requoted", requote_plain_literals(source, &tokens)),
    ];
    candidates
        .into_iter()
        .filter(|(_, text)| {
            let (retokens, diags) = nsl_lexer::tokenize(text, FileId(0), &mut interner);
            diags.is_empty() && kinds(&retokens) == kinds(&tokens)
        })
        .collect()
}

#[test]
fn fmt_preserves_the_ast_and_is_idempotent() {
    let root = workspace_root();
    let mut checked = 0usize;
    let mut changed: BTreeMap<&str, usize> = BTreeMap::new();
    let mut skipped = BTreeSet::new();
    let mut failures = Vec::new();

    for path in corpus() {
        let rel = path
            .strip_prefix(&root)
            .unwrap()
            .display()
            .to_string()
            .replace('\\', "/");
        let source = fs::read_to_string(&path).unwrap();
        let before = match clean_ast(&source) {
            Ok(ast) => ast,
            Err(why) => {
                skipped.insert(rel.clone());
                eprintln!("skipped {rel}: {why}");
                continue;
            }
        };
        checked += 1;

        let mut inputs = vec![("as committed", source.clone())];
        inputs.extend(perturbations(&source));
        for (spelling, input) in &inputs {
            let formatted = match format_source(input) {
                Ok(r) => {
                    if r.changed {
                        *changed.entry(spelling).or_default() += 1;
                    }
                    r.output
                }
                Err(e) => {
                    failures.push(format!("{rel} ({spelling}): fmt refused: {e}"));
                    continue;
                }
            };

            match clean_ast(&formatted) {
                Err(why) => failures.push(format!(
                    "{rel} ({spelling}): formatted output no longer parses: {why}"
                )),
                Ok(after) if after != before => failures.push(format!(
                    "{rel} ({spelling}): AST changed — {}",
                    first_difference(&before, &after)
                )),
                Ok(_) => {}
            }

            match format_source(&formatted) {
                Ok(r) if r.changed => failures.push(format!(
                    "{rel} ({spelling}): not idempotent — {}",
                    first_difference(&formatted, &r.output)
                )),
                Ok(_) => {}
                Err(e) => failures.push(format!("{rel} ({spelling}): second fmt refused: {e}")),
            }
        }
    }

    eprintln!("checked {checked} files; fmt changed {changed:?}");
    let expected: BTreeSet<String> = EXPECTED_SKIPS.iter().map(|(p, _)| p.to_string()).collect();
    assert_eq!(
        skipped, expected,
        "the set of corpus files the frontend rejects changed; update EXPECTED_SKIPS \
         with a reason for each new entry (checked {checked})"
    );
    for (spelling, floor) in CHANGED_FLOOR {
        let n = changed.get(spelling).copied().unwrap_or(0);
        assert!(
            n >= *floor,
            "fmt changed only {n} {spelling} inputs of {checked} files; the property is no \
             longer exercising the formatter on that spelling"
        );
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} files: fmt changed the program\n{}",
        failures.len(),
        failures.join("\n")
    );
}
