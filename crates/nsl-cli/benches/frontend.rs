//! The multi-module frontend `nsl build` runs before codegen —
//! `loader::load_all_modules`: read, lex and parse the entry file, resolve
//! its imports transitively (stdlib included), topologically sort, and
//! analyze every module in dependency order with real imported types. This
//! is the semantic checker's real workload; the single-module `analyze`
//! bench in nsl-semantic only sees import-free examples.
//! `cargo bench -p nsl-cli`.
//!
//! Inputs are the three production coder recipes. Each pulls `model.nsl`
//! beside it plus the `nsl.nn.*` / optimizer stdlib modules, so one
//! iteration is ~a dozen files. The measured region includes file reads
//! (page-cache hits after the first pass) because the loader does them
//! inline; the codegen half of `nsl build` is not library-callable yet
//! (roadmap A1) and is not measured here.

use std::path::{Path, PathBuf};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use nsl_cli::loader::load_all_modules;
use nsl_errors::SourceMap;
use nsl_lexer::Interner;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

const RECIPES: &[&str] = &[
    "models/coder50m/pretrain_prod.nsl",
    "models/coder500m/pretrain_prod.nsl",
    "models/coder1b/pretrain_prod.nsl",
];

/// One frontend pass. The loader renders every warning it sees through the
/// source map (the stdlib modules carry a few), which per iteration would be
/// thousands of lines, so the map is silent; a failure still surfaces as the
/// `Err` the loader returns, and `nsl check <recipe>` shows the rendering.
fn load(entry: &Path) -> usize {
    let mut source_map = SourceMap::silent();
    let mut interner = Interner::new();
    let graph = load_all_modules(entry, &mut source_map, &mut interner)
        .unwrap_or_else(|e| panic!("{}: {e}", entry.display()));
    graph.modules.len()
}

fn bench_frontend(c: &mut Criterion) {
    let root = repo_root();
    // The resolver's first stdlib root. `cargo bench` runs with the crate
    // directory as cwd, so the cwd-relative fallback would miss.
    std::env::set_var("NSL_STDLIB_PATH", root.join("stdlib"));

    let mut group = c.benchmark_group("frontend");

    for recipe in RECIPES {
        let entry = root.join(recipe);
        let modules = load(&entry);
        eprintln!("frontend: {recipe} -> {modules} modules");
        let label = entry
            .parent()
            .and_then(Path::file_name)
            .map(|d| d.to_string_lossy().into_owned())
            .expect("recipe directory name");
        group.bench_with_input(
            BenchmarkId::new("load_all_modules", label),
            &entry,
            |b, entry| b.iter(|| load(entry)),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_frontend);
criterion_main!(benches);
