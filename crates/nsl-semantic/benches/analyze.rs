//! Single-module semantic analysis over the import-free examples, parsed
//! once so only `analyze` is inside the measured region.
//! `cargo bench -p nsl-semantic`.
//!
//! `analyze` is the no-imports entry, so the corpus is every example that
//! analyzes standalone with no Error-level diagnostic (the `*_error.nsl`
//! negative tests and anything with an `import` are dropped at setup and
//! reported on stderr). The multi-module frontend — stdlib modules analyzed
//! with real imported types — is nsl-cli's `frontend` bench.

use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nsl_ast::Module;
use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner};
use nsl_semantic::analyze;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn examples() -> Vec<(String, String)> {
    let mut files: Vec<(String, String)> = std::fs::read_dir(repo_root().join("examples"))
        .expect("examples/ directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "nsl"))
        .map(|p| {
            let src = std::fs::read_to_string(&p).expect("read example");
            let name = p
                .file_name()
                .expect("file name")
                .to_string_lossy()
                .into_owned();
            (name, src)
        })
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(files.len() > 100, "found only {} examples", files.len());
    files
}

/// Parse every example and keep those whose standalone analysis is clean.
fn analyzable(interner: &mut Interner) -> Vec<(String, usize, Module)> {
    let mut kept = Vec::new();
    let mut dropped = 0usize;
    for (i, (name, src)) in examples().into_iter().enumerate() {
        let (tokens, lex_diags) = tokenize(&src, FileId(i), interner);
        if lex_diags.iter().any(|d| d.level == Level::Error) {
            dropped += 1;
            continue;
        }
        let parsed = parse_clean(&tokens, interner);
        let Some(module) = parsed else {
            dropped += 1;
            continue;
        };
        let clean = !analyze(&module, interner)
            .diagnostics
            .iter()
            .any(|d| d.level == Level::Error);
        if clean {
            kept.push((name, src.len(), module));
        } else {
            dropped += 1;
        }
    }
    eprintln!(
        "analyze corpus: {} examples analyze standalone, {} dropped (imports or diagnostics)",
        kept.len(),
        dropped
    );
    kept
}

fn parse_clean(tokens: &[nsl_lexer::Token], interner: &mut Interner) -> Option<Module> {
    let result = nsl_parser::parse(tokens, interner);
    if result.diagnostics.iter().any(|d| d.level == Level::Error) {
        None
    } else {
        Some(result.module)
    }
}

fn bench_analyze(c: &mut Criterion) {
    let mut interner = Interner::new();
    let modules = analyzable(&mut interner);
    let bytes: usize = modules.iter().map(|(_, n, _)| *n).sum();

    let mut group = c.benchmark_group("analyze");
    // The whole-corpus pass is ~2.6 ms; criterion's linear sampling needs
    // ~5000 passes for its default 100 samples, which does not fit the
    // default 5 s window (it warns and runs long anyway).
    group.measurement_time(Duration::from_secs(15));

    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("examples", |b| {
        b.iter(|| {
            let mut entries = 0usize;
            for (_, _, module) in &modules {
                entries += analyze(module, &mut interner).type_map.len();
            }
            entries
        })
    });

    let (largest_name, largest_bytes, largest) = modules
        .iter()
        .max_by_key(|(_, n, _)| *n)
        .expect("non-empty corpus");
    eprintln!("analyze largest: {largest_name} ({largest_bytes} bytes)");
    group.throughput(Throughput::Bytes(*largest_bytes as u64));
    group.bench_function("largest_example", |b| {
        b.iter(|| analyze(largest, &mut interner).type_map.len())
    });

    group.finish();
}

criterion_group!(benches, bench_analyze);
criterion_main!(benches);
