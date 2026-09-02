//! Parser throughput over the in-tree NSL corpus, pre-tokenized so the lexer
//! is outside the measured region. `cargo bench -p nsl-parser`.
//!
//! Files that do not parse cleanly (there is one deliberately broken example
//! at the time of writing) are dropped at setup and reported on stderr; the
//! tokens/s figure is what to compare across baselines.

use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner, Token};
use nsl_parser::parse;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

/// Gather `.nsl` files under `dir`, keyed by their path relative to `root`.
fn collect(root: &Path, dir: &Path, recurse: bool, out: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if recurse {
                collect(root, &path, true, out);
            }
        } else if path.extension().is_some_and(|e| e == "nsl") {
            let src = std::fs::read_to_string(&path).expect("read corpus file");
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            out.push((rel, src));
        }
    }
}

fn corpus() -> Vec<(String, String)> {
    let root = repo_root();
    let mut files = Vec::new();
    collect(&root, &root.join("stdlib"), true, &mut files);
    collect(&root, &root.join("examples"), false, &mut files);
    collect(&root, &root.join("models"), true, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(
        files.len() > 100,
        "corpus discovery found only {} files",
        files.len()
    );
    files
}

/// Tokenize every corpus file once on a shared interner and keep those that
/// lex and parse with no Error-level diagnostic: the parser's happy path.
fn tokenized(interner: &mut Interner) -> Vec<(String, Vec<Token>)> {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for (i, (name, src)) in corpus().into_iter().enumerate() {
        let (tokens, lex_diags) = tokenize(&src, FileId(i), interner);
        let lex_ok = !lex_diags.iter().any(|d| d.level == Level::Error);
        let parse_ok = lex_ok
            && !parse(&tokens, interner)
                .diagnostics
                .iter()
                .any(|d| d.level == Level::Error);
        if parse_ok {
            kept.push((name, tokens));
        } else {
            dropped.push(name);
        }
    }
    for name in &dropped {
        eprintln!("parse corpus: dropped {name} (does not parse cleanly)");
    }
    kept
}

fn bench_parse(c: &mut Criterion) {
    let mut interner = Interner::new();
    let files = tokenized(&mut interner);
    let tokens: usize = files.iter().map(|(_, t)| t.len()).sum();
    eprintln!("parse corpus: {} files, {} tokens", files.len(), tokens);

    let mut group = c.benchmark_group("parse");
    // The whole-corpus pass is ~1 ms; criterion's linear sampling needs
    // ~5000 passes for its default 100 samples, which does not fit the
    // default 5 s window (it warns and runs long anyway).
    group.measurement_time(Duration::from_secs(10));

    group.throughput(Throughput::Elements(tokens as u64));
    group.bench_function("corpus", |b| {
        b.iter(|| {
            let mut stmts = 0usize;
            for (_, toks) in &files {
                stmts += parse(toks, &mut interner).module.stmts.len();
            }
            stmts
        })
    });

    let (largest_name, largest) = files
        .iter()
        .max_by_key(|(_, t)| t.len())
        .expect("non-empty corpus");
    eprintln!("parse largest: {largest_name} ({} tokens)", largest.len());
    group.throughput(Throughput::Elements(largest.len() as u64));
    group.bench_function("largest_file", |b| {
        b.iter(|| parse(largest, &mut interner).module.stmts.len())
    });

    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
