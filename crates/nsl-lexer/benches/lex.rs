//! Lexer throughput over the in-tree NSL corpus (stdlib + examples + the
//! coder recipes). `cargo bench -p nsl-lexer`; see scripts/bench.sh for the
//! save/compare-baseline workflow.
//!
//! The corpus is discovered, not listed, so it tracks the tree; the bytes/s
//! figure is what to compare across baselines, since the absolute time moves
//! with the corpus size.

use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nsl_errors::FileId;
use nsl_lexer::{tokenize, Interner};

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

/// Every `.nsl` under stdlib/ (recursive), examples/ and models/ (recursive),
/// sorted by path so the workload order is stable between runs.
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

fn bench_lex(c: &mut Criterion) {
    let files = corpus();
    let bytes: usize = files.iter().map(|(_, s)| s.len()).sum();
    let mut interner = Interner::new();
    let tokens: usize = files
        .iter()
        .enumerate()
        .map(|(i, (_, src))| tokenize(src, FileId(i), &mut interner).0.len())
        .sum();
    eprintln!(
        "lex corpus: {} files, {} bytes, {} tokens",
        files.len(),
        bytes,
        tokens
    );

    let mut group = c.benchmark_group("lex");

    // Fresh interner per pass, as every `nsl build` starts with one: symbol
    // interning is part of what the lexer costs.
    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("corpus", |b| {
        b.iter(|| {
            let mut interner = Interner::new();
            let mut n = 0usize;
            for (i, (_, src)) in files.iter().enumerate() {
                n += tokenize(src, FileId(i), &mut interner).0.len();
            }
            n
        })
    });

    // The largest single file, on a warm interner: the per-file cost the
    // multi-module frontend pays for each stdlib module after the first.
    let (largest_name, largest) = files
        .iter()
        .max_by_key(|(_, s)| s.len())
        .expect("non-empty corpus");
    eprintln!("lex largest: {largest_name} ({} bytes)", largest.len());
    group.throughput(Throughput::Bytes(largest.len() as u64));
    group.bench_function("largest_file", |b| {
        let mut interner = Interner::new();
        b.iter(|| tokenize(largest, FileId(0), &mut interner).0.len())
    });

    group.finish();
}

criterion_group!(benches, bench_lex);
criterion_main!(benches);
