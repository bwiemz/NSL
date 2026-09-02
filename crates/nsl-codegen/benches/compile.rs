//! Codegen latency: `compile_module` (the full pass pipeline + Cranelift
//! emission to an object file) over the import-free examples, with the
//! frontend run once at setup so only codegen is inside the measured region.
//! `cargo bench -p nsl-codegen --bench compile`. CPU-only; not the `bench`
//! BINARY, which is the PCA Tier B GPU harness.
//!
//! Corpus: every example with no `import` (and no `@autotune`, see
//! `skip_reason`) that lexes, parses and analyzes with no Error-level
//! diagnostic and whose codegen succeeds under the default options. The set
//! is discovered, not listed, so it tracks the tree; the counts and the
//! skipped/dropped names go to stderr. The multi-module compile (`nsl build`
//! on a recipe graph) lives in the CLI's `run_build_multi` and is not
//! library-callable yet — roadmap A1.

use std::path::{Path, PathBuf};
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nsl_ast::Module;
use nsl_codegen::{compile_module, CompileOptions, WggoImportance};
use nsl_errors::{FileId, Level};
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

/// The default options, except WGGO importance pinned to `Magnitude`: with
/// the default `Auto`, every `compile_module` of a source with no
/// `@wggo_target` prints a multi-line "fell back to magnitude scoring" note
/// (entry_points.rs, §5.7) — one per example per iteration, which would be
/// thousands of lines of stderr. The pass runs the same either way.
fn options() -> CompileOptions {
    let mut opts = CompileOptions::default();
    opts.wggo.importance = WggoImportance::Magnitude;
    opts
}

struct Case {
    name: String,
    bytes: usize,
    module: Module,
    interner: Interner,
    type_map: TypeMap,
}

/// Examples the bench does not measure, by construction rather than by
/// outcome: multi-module sources (this is the single-module entry), and
/// `@autotune` kernels, whose compile goes through the on-disk
/// `.nsl-cache/autotune/` — a cache miss on the first iteration and a hit on
/// every later one, plus a selection line on stderr per compile.
fn skip_reason(src: &str) -> Option<&'static str> {
    for l in src.lines().map(str::trim_start) {
        if l.starts_with("import ") || l.starts_with("from ") {
            return Some("imports");
        }
        if l.starts_with("@autotune") {
            return Some("@autotune");
        }
    }
    None
}

fn frontend(src: &str) -> Option<(Module, Interner, TypeMap)> {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    if lex_diags.iter().any(|d| d.level == Level::Error) {
        return None;
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed.diagnostics.iter().any(|d| d.level == Level::Error) {
        return None;
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if analysis.diagnostics.iter().any(|d| d.level == Level::Error) {
        return None;
    }
    Some((parsed.module, interner, analysis.type_map))
}

fn cases() -> Vec<Case> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(repo_root().join("examples"))
        .expect("examples/ directory")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "nsl"))
        .collect();
    paths.sort();

    let opts = options();
    let mut kept = Vec::new();
    let mut skipped = Vec::new();
    let mut dropped = Vec::new();
    for path in paths {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let src = std::fs::read_to_string(&path).expect("read example");
        if let Some(why) = skip_reason(&src) {
            skipped.push(format!("{name} ({why})"));
            continue;
        }
        let Some((module, interner, type_map)) = frontend(&src) else {
            dropped.push(name);
            continue;
        };
        if compile_module(&module, &interner, &type_map, "", false, &opts).is_err() {
            dropped.push(name);
            continue;
        }
        kept.push(Case {
            name,
            bytes: src.len(),
            module,
            interner,
            type_map,
        });
    }
    eprintln!(
        "compile corpus: {} examples compile clean; skipped {} (by construction): {}",
        kept.len(),
        skipped.len(),
        skipped.join(", ")
    );
    eprintln!(
        "compile corpus: dropped {} (frontend diagnostics or codegen error): {}",
        dropped.len(),
        dropped.join(", ")
    );
    assert!(
        kept.len() > 50,
        "compile corpus collapsed to {} cases",
        kept.len()
    );
    kept
}

fn bench_compile(c: &mut Criterion) {
    let cases = cases();
    let opts = options();
    let bytes: usize = cases.iter().map(|k| k.bytes).sum();

    let mut group = c.benchmark_group("compile");
    // One pass over the corpus is ~40 ms (≈0.4 ms per example), so the
    // default 100 samples would not fit criterion's default 5 s window.
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    group.throughput(Throughput::Bytes(bytes as u64));
    group.bench_function("examples", |b| {
        b.iter(|| {
            let mut object_bytes = 0usize;
            for k in &cases {
                let obj = compile_module(&k.module, &k.interner, &k.type_map, "", false, &opts)
                    .expect("compiled at setup");
                object_bytes += obj.len();
            }
            object_bytes
        })
    });

    let largest = cases
        .iter()
        .max_by_key(|k| k.bytes)
        .expect("non-empty corpus");
    eprintln!(
        "compile largest: {} ({} bytes)",
        largest.name, largest.bytes
    );
    group.throughput(Throughput::Bytes(largest.bytes as u64));
    group.bench_function("largest_example", |b| {
        b.iter(|| {
            compile_module(
                &largest.module,
                &largest.interner,
                &largest.type_map,
                "",
                false,
                &opts,
            )
            .expect("compiled at setup")
            .len()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_compile);
criterion_main!(benches);
