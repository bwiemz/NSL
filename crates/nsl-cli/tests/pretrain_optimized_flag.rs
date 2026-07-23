//! CLI-level gates for `--pretrain-optimized` (roadmap 3.3).
//!
//! Expansion semantics (fill-None, explicit-wins, source-AD forced) are pinned
//! by the unit tests in `src/meta_flags.rs`; these tests pin the CLI surface:
//! the clap-level `--tape-ad` conflict, and that a bundled invocation drives a
//! real program end-to-end (the bundle must never wedge a working program).

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_source() -> String {
    let root = repo_root();
    std::fs::read_to_string(
        root.join("crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl"),
    )
    .expect("fixture")
    .replace("model_save(m, \"adamw_out.nslm\")", "print(\"BUNDLE_DONE\")")
}

fn run_nsl(args: &[&str], program: &str, tag: &str) -> (bool, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_pretrain_flag_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, program).unwrap();

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"])
        .args(args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl");
    let res = (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    );
    std::fs::remove_dir_all(&tmp).ok();
    res
}

/// The bundle conflicts with --tape-ad at the clap layer (the planning stack
/// lives in the source-AD branch; tape AD would silently bypass all of it).
#[test]
fn pretrain_optimized_conflicts_with_tape_ad() {
    let (ok, _out, err) = run_nsl(&["--pretrain-optimized", "--tape-ad"], &fixture_source(), "conflict");
    assert!(!ok, "conflicting flags must fail");
    assert!(
        err.contains("cannot be used with"),
        "expected the clap conflict diagnostic, got:\n{err}"
    );
}

/// A bundled invocation must run a real training program to completion —
/// the expanded `--source-ad --wggo greedy --csha auto` path may not wedge
/// a program that runs fine unbundled.
#[test]
fn pretrain_optimized_runs_end_to_end() {
    let (ok, out, err) = run_nsl(&["--pretrain-optimized"], &fixture_source(), "e2e");
    assert!(
        ok && out.contains("BUNDLE_DONE"),
        "bundled run failed.\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
    );
}
