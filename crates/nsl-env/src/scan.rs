//! Source scanner behind the agreement gate: finds every `std::env::var` /
//! `var_os` read of an `NSL_*` name in the workspace's Rust sources.
//!
//! It is a text scanner, not a parser, on purpose — the gate must stay
//! dependency-free and fast. It handles the forms the codebase uses:
//!
//! - `env::var("NSL_X")` and `env::var_os("NSL_X")`, with any path prefix
//!   and whitespace/newlines between the `(` and the literal;
//! - `env::var(IDENT)` where the same file declares
//!   `const IDENT: &str = "NSL_X"` (or `static`); two declarations of one
//!   identifier in different modules of a file both count;
//!
//! and reports anything else passed to `env::var` as a [`Read::Dynamic`] so
//! the gate can pin the (short) list of reads it cannot see through, rather
//! than silently missing a new one. A call quoted inside a `//` comment, a
//! `/* */` block, or a string literal is prose, not a read.
//!
//! Scope: `crates/*/{src,tests,benches,examples}/**/*.rs` and each crate's
//! `build.rs`. Shell scripts, `tools/`, and CI workflows are not Rust and
//! are not scanned; a variable only they read does not belong in the
//! registry.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// One `env::var` / `env::var_os` call site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Read {
    /// The name is a literal, or an identifier resolved to a literal in the
    /// same file.
    Named { name: String, line: usize },
    /// The argument is an expression the scanner cannot resolve — a loop
    /// variable, a function call, a `format!`. `expr` is the argument text.
    Dynamic { expr: String, line: usize },
}

impl Read {
    pub fn line(&self) -> usize {
        match self {
            Read::Named { line, .. } | Read::Dynamic { line, .. } => *line,
        }
    }
}

/// Scan one Rust source text for env reads.
pub fn scan_source(text: &str) -> Vec<Read> {
    let consts = string_consts(text);
    let prose = prose_mask(text);
    let mut out = Vec::new();
    for needle in ["env::var_os(", "env::var("] {
        let mut from = 0;
        while let Some(pos) = text[from..].find(needle) {
            let call = from + pos;
            let arg_start = call + needle.len();
            from = arg_start;
            // The two needles are disjoint (`var(` vs `var_os(`), and neither
            // matches `set_var(` / `remove_var(` (the `env::` must directly
            // precede `var`).
            if prose[call] {
                continue;
            }
            let rest = &text[arg_start..];
            let trimmed = rest.trim_start();
            let line = line_of(text, call);
            if let Some(lit) = trimmed.strip_prefix('"') {
                if let Some(end) = lit.find('"') {
                    let name = &lit[..end];
                    if name.starts_with("NSL_") {
                        out.push(Read::Named { name: name.to_string(), line });
                    }
                    continue;
                }
            }
            // An expression: everything up to the `)` that closes the call.
            let expr = trimmed[..balanced_len(trimmed)].trim();
            if let Some(names) = consts.get(expr) {
                for name in names.iter().filter(|n| n.starts_with("NSL_")) {
                    out.push(Read::Named { name: name.clone(), line });
                }
            } else if !expr.is_empty() {
                out.push(Read::Dynamic { expr: expr.to_string(), line });
            }
        }
    }
    out.sort_by_key(|r| r.line());
    out
}

/// `const NAME: &str = "…"` / `static NAME: &str = "…"` declarations in a
/// file, by identifier. One identifier can be declared more than once (a
/// `PROBE` in each of two test modules); every distinct value is kept, so a
/// read through the identifier counts for all of them rather than for
/// whichever declaration came last.
fn string_consts(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for kw in ["const ", "static "] {
        let mut from = 0;
        while let Some(pos) = text[from..].find(kw) {
            let at = from + pos;
            from = at + kw.len();
            // Must start a token: preceded by whitespace/newline/`pub `.
            if at > 0 && !text.as_bytes()[at - 1].is_ascii_whitespace() {
                continue;
            }
            let rest = &text[from..];
            let ident_len = rest
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
                .count();
            if ident_len == 0 {
                continue;
            }
            let ident = &rest[..ident_len];
            let after = rest[ident_len..].trim_start();
            let Some(after) = after.strip_prefix(':') else { continue };
            let after = after.trim_start();
            let Some(after) = after.strip_prefix("&str").or_else(|| after.strip_prefix("&'static str")) else {
                continue;
            };
            let after = after.trim_start();
            let Some(after) = after.strip_prefix('=') else { continue };
            let after = after.trim_start();
            let Some(lit) = after.strip_prefix('"') else { continue };
            let Some(end) = lit.find('"') else { continue };
            let value = lit[..end].to_string();
            let values = out.entry(ident.to_string()).or_default();
            if !values.contains(&value) {
                values.push(value);
            }
        }
    }
    out
}

fn line_of(text: &str, byte: usize) -> usize {
    text[..byte].bytes().filter(|b| *b == b'\n').count() + 1
}

/// For every byte of `text`, whether it is prose rather than code: inside a
/// `//` line comment, a (nestable) `/* */` block, or a string literal
/// (plain or raw). One pass with real lexer state, so `"http://x"` does not
/// open a comment, `'"'` does not open a string, and a `tests/*.rs` glob
/// inside a `//` comment does not open a block that swallows the rest of
/// the file. Char literals are skipped whole and never count as prose.
fn prose_mask(text: &str) -> Vec<bool> {
    #[derive(PartialEq)]
    enum State {
        Code,
        Str,
        RawStr(usize),
        Line,
        Block(usize),
    }
    let b = text.as_bytes();
    let mut mask = vec![false; b.len()];
    let mut state = State::Code;
    let mut i = 0;
    while i < b.len() {
        match state {
            State::Code => {
                if b[i] == b'/' && b.get(i + 1) == Some(&b'/') {
                    state = State::Line;
                    mask[i] = true;
                } else if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    state = State::Block(1);
                    mask[i] = true;
                    mask[i + 1] = true;
                    i += 1;
                } else if b[i] == b'"' {
                    state = State::Str;
                } else if b[i] == b'r' && (b.get(i + 1) == Some(&b'"') || b.get(i + 1) == Some(&b'#')) {
                    // r"…", r#"…"#, br"…" (the `b` is just a preceding byte).
                    let hashes = b[i + 1..].iter().take_while(|c| **c == b'#').count();
                    if b.get(i + 1 + hashes) == Some(&b'"') {
                        state = State::RawStr(hashes);
                        i += 1 + hashes;
                    }
                } else if b[i] == b'\'' {
                    // A char literal (`'"'`, `'\n'`) is skipped whole; a
                    // lifetime (`'a`) is a single quote followed by code.
                    if b.get(i + 1) == Some(&b'\\') {
                        // `'\n'`, `'\''`, `'\u{…}'`: the closing quote is the
                        // first one after the escaped character.
                        if let Some(close) = b.get(i + 3..).and_then(|t| t.iter().position(|c| *c == b'\'')) {
                            i += 3 + close;
                        }
                    } else if b.get(i + 2) == Some(&b'\'') {
                        i += 2;
                    }
                }
            }
            State::Str => {
                mask[i] = true;
                if b[i] == b'\\' {
                    if let Some(escaped) = mask.get_mut(i + 1) {
                        *escaped = true;
                    }
                    i += 1;
                } else if b[i] == b'"' {
                    state = State::Code;
                }
            }
            State::RawStr(hashes) => {
                mask[i] = true;
                if b[i] == b'"' && b[i + 1..].iter().take_while(|c| **c == b'#').count() >= hashes {
                    mask[i + 1..i + 1 + hashes].iter_mut().for_each(|m| *m = true);
                    i += hashes;
                    state = State::Code;
                }
            }
            State::Line => {
                mask[i] = true;
                if b[i] == b'\n' {
                    state = State::Code;
                }
            }
            State::Block(depth) => {
                mask[i] = true;
                if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                    mask[i + 1] = true;
                    i += 1;
                    state = State::Block(depth + 1);
                } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                    mask[i + 1] = true;
                    i += 1;
                    state = if depth == 1 { State::Code } else { State::Block(depth - 1) };
                }
            }
        }
        i += 1;
    }
    mask
}

/// Length of the prefix of `s` up to (not including) the `)` that closes an
/// already-open call — i.e. the call's argument text.
fn balanced_len(s: &str) -> usize {
    let mut depth = 0usize;
    for (i, b) in s.bytes().enumerate() {
        match b {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                if depth == 0 {
                    return i;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    s.len()
}

/// A read site located in the workspace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Site {
    /// Path relative to the workspace root, `/`-separated.
    pub path: String,
    pub read: Read,
}

/// Every env read under `<root>/crates/*/{src,tests,benches,examples}` and
/// in each crate's `build.rs`, except this crate's own sources (its scanner
/// tests quote reads of made-up names).
pub fn scan_workspace(root: &Path) -> io::Result<Vec<Site>> {
    let mut files = Vec::new();
    let crates = root.join("crates");
    for entry in fs::read_dir(&crates)? {
        let krate = entry?.path();
        if !krate.is_dir() || krate.file_name().is_some_and(|n| n == "nsl-env") {
            continue;
        }
        let build_script = krate.join("build.rs");
        if build_script.is_file() {
            files.push(build_script);
        }
        for sub in ["src", "tests", "benches", "examples"] {
            let dir = krate.join(sub);
            if dir.is_dir() {
                collect_rs(&dir, &mut files)?;
            }
        }
    }
    files.sort();
    let mut out = Vec::new();
    for file in files {
        let text = fs::read_to_string(&file)?;
        let rel = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");
        for read in scan_source(&text) {
            out.push(Site { path: rel.clone(), read });
        }
    }
    Ok(out)
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_rs(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_reads_with_any_prefix_and_whitespace() {
        let src = r#"
            let a = std::env::var("NSL_A").ok();
            let b = env::var_os("NSL_B");
            let c = std::env::var(
                "NSL_C",
            );
            let not_nsl = std::env::var("CUDA_PATH");
        "#;
        let reads = scan_source(src);
        assert_eq!(
            reads,
            vec![
                Read::Named { name: "NSL_A".into(), line: 2 },
                Read::Named { name: "NSL_B".into(), line: 3 },
                Read::Named { name: "NSL_C".into(), line: 4 },
            ]
        );
    }

    #[test]
    fn identifier_reads_resolve_through_same_file_consts() {
        let src = r#"
            const KNOB: &str = "NSL_FORCE_THING";
            pub static OTHER: &'static str = "NSL_OTHER";
            fn f() {
                let forced = std::env::var(KNOB).map(|v| v == "1").unwrap_or(false);
                let o = std::env::var_os(OTHER);
            }
        "#;
        let reads = scan_source(src);
        assert_eq!(
            reads,
            vec![
                Read::Named { name: "NSL_FORCE_THING".into(), line: 5 },
                Read::Named { name: "NSL_OTHER".into(), line: 6 },
            ]
        );
    }

    #[test]
    fn unresolvable_arguments_are_reported_not_dropped() {
        let src = r#"
            for root in ["CUDA_PATH", "CUDA_HOME"] {
                if let Ok(p) = std::env::var(root) {}
            }
            let v = std::env::var(format!("NSL_{suffix}"));
        "#;
        let reads = scan_source(src);
        assert_eq!(
            reads,
            vec![
                Read::Dynamic { expr: "root".into(), line: 3 },
                Read::Dynamic { expr: "format!(\"NSL_{suffix}\")".into(), line: 5 },
            ]
        );
    }

    #[test]
    fn a_read_quoted_in_a_comment_is_prose() {
        let src = r#"
            // Set via `std::env::var("NSL_DOCUMENTED")` by the harness.
            /// Reads `env::var("NSL_DOCUMENTED_TOO")`.
            let real = std::env::var("NSL_REAL");
            let x = 1; // was std::env::var("NSL_TRAILING") before the flag died
            let url = "http://host"; let y = std::env::var("NSL_AFTER_A_STRING_SLASH");
            /* a block comment:
               std::env::var("NSL_BLOCKED") */
            let z = std::env::var("NSL_AFTER_BLOCK");
        "#;
        assert_eq!(
            scan_source(src),
            vec![
                Read::Named { name: "NSL_REAL".into(), line: 4 },
                Read::Named { name: "NSL_AFTER_A_STRING_SLASH".into(), line: 6 },
                Read::Named { name: "NSL_AFTER_BLOCK".into(), line: 9 },
            ]
        );
    }

    /// The traps a text scanner falls into: a `/*` inside a `//` comment
    /// (the `tests/*.rs` glob in `tensor/mod.rs` once hid every read after
    /// it), a `//` inside a string, a `'"'` char literal, a raw string
    /// holding both, and a nested block comment.
    #[test]
    fn comment_state_survives_globs_chars_and_raw_strings() {
        let src = r##"
            // integration tests in `tests/*.rs` are gated on the feature
            let a = std::env::var("NSL_AFTER_GLOB");
            let sep = '"'; let quote = '\''; let b = std::env::var("NSL_AFTER_CHARS");
            let raw = r#"// not a comment: std::env::var("NSL_IN_RAW") "#; let c = std::env::var("NSL_AFTER_RAW");
            /* outer /* inner std::env::var("NSL_NESTED") */ still comment std::env::var("NSL_STILL") */
            let d = std::env::var("NSL_AFTER_NESTED");
        "##;
        let names: Vec<String> = scan_source(src)
            .into_iter()
            .filter_map(|r| match r {
                Read::Named { name, .. } => Some(name),
                Read::Dynamic { .. } => None,
            })
            .collect();
        assert_eq!(names, ["NSL_AFTER_GLOB", "NSL_AFTER_CHARS", "NSL_AFTER_RAW", "NSL_AFTER_NESTED"]);
    }

    #[test]
    fn an_identifier_declared_twice_counts_for_both_names() {
        let src = r#"
            mod a {
                const PROBE: &str = "NSL_PROBE_A";
                fn f() { let _ = std::env::var(PROBE); }
            }
            mod b {
                const PROBE: &str = "NSL_PROBE_B";
                fn f() { let _ = std::env::var(PROBE); }
            }
        "#;
        let names: Vec<String> = scan_source(src)
            .into_iter()
            .filter_map(|r| match r {
                Read::Named { name, .. } => Some(name),
                Read::Dynamic { .. } => None,
            })
            .collect();
        assert_eq!(names, ["NSL_PROBE_A", "NSL_PROBE_B", "NSL_PROBE_A", "NSL_PROBE_B"]);
    }

    #[test]
    fn writes_are_not_reads() {
        let src = r#"
            std::env::set_var("NSL_X", "1");
            std::env::remove_var("NSL_Y");
            cmd.env("NSL_Z", "1");
        "#;
        assert!(scan_source(src).is_empty());
    }
}
