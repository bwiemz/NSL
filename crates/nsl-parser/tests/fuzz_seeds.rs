//! The fuzz targets' invariants, run over the committed seed corpus on the
//! stable toolchain.
//!
//! `fuzz/` needs nightly and cargo-fuzz, so CI never builds it. This test
//! compiles the same check module (`common/frontend_check.rs`, included
//! by `#[path]` from both sides) and runs it over every `.nsl` file in
//! `fuzz/corpus/lex/` and `fuzz/corpus/parse/`. Only `.nsl` files: the
//! hash-named inputs libFuzzer grows the corpus with are untracked and a
//! running fuzzer adds and removes them at will. A seed that starts
//! failing here is a frontend regression; a crash found by fuzzing
//! becomes a permanent regression test by copying its reproducer into the
//! corpus directory under a `.nsl` name.

mod common {
    pub mod frontend_check;
}

use std::fs;
use std::path::{Path, PathBuf};

fn fuzz_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/nsl-parser sits two levels below the workspace root")
        .join("fuzz")
}

/// Every `.nsl` file directly under `dir`, sorted.
fn seeds(dir: &Path) -> Vec<PathBuf> {
    let entries = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read the seed corpus {}: {e}", dir.display()));
    let mut files: Vec<PathBuf> = entries
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|ext| ext == "nsl"))
        .collect();
    files.sort();
    files
}

fn run_over(target: &str, check: fn(&str)) {
    let files = seeds(&fuzz_dir().join("corpus").join(target));
    assert!(
        files.len() >= 5,
        "fuzz/corpus/{target} holds {} seeds; the committed corpus was larger",
        files.len()
    );

    for path in files {
        let bytes = fs::read(&path).unwrap();
        // Same decoding as the fuzz targets (fuzz/src/lib.rs).
        let source = String::from_utf8_lossy(&bytes);
        let outcome = std::panic::catch_unwind(|| check(&source));
        assert!(outcome.is_ok(), "frontend invariant violated on {}", path.display());
    }
}

#[test]
fn lexer_invariants_hold_on_the_seed_corpus() {
    run_over("lex", |source| {
        common::frontend_check::lex(source);
    });
}

#[test]
fn parser_invariants_hold_on_the_seed_corpus() {
    run_over("parse", common::frontend_check::parse);
}
