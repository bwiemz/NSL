//! Sprint 3 (paper §6.3): `nsl check --csha-report` exposes the CSHA
//! attention-fusion planner as a user-facing diagnostic that runs
//! without emitting any kernels or linking a binary.
//!
//! Before Sprint 3, `--csha` / `--csha-report` were only accepted by
//! `nsl run` and `nsl build` — clap rejected them on the `Check` arm.
//! The paper text specifies them on `nsl check`, so this test pins:
//!   1. clap accepts `--csha-report` on the Check subcommand.
//!   2. clap accepts `--csha auto` on the Check subcommand.
//!   3. A file without a `train(...)` block produces a friendly
//!      `note:` instead of a silent no-op.
//!   4. Invalid `--csha` values are rejected with a clear error.
//!
//! The full report-content assertion (per-layer Boundary/Pipeline/Block
//! lines + the new `Backward:` line) is covered by the in-process
//! `nsl-codegen` unit tests on `CshaPlan::render_report` so this CLI
//! integration test stays fast and doesn't depend on a train block
//! fixture that resolves stdlib imports.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Trivial NSL file with no `train(...)` block. The CSHA planner has
/// nothing to do, so `--csha-report` should emit a `note:` and exit
/// successfully without producing a report.
const NO_TRAIN_SRC: &str = r#"let x = 1
"#;

#[test]
fn check_accepts_csha_report_flag() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("trivial.nsl");
    fs::write(&src_path, NO_TRAIN_SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&src_path).arg("--csha-report");
    // clap MUST accept the flag — pre-Sprint-3 this returned
    // `error: unexpected argument '--csha-report' found`.  The flag
    // ITSELF being mentioned in our own `note:` message is fine; we
    // assert clap didn't reject it (no "unexpected argument" line).
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("unexpected argument").not());
}

#[test]
fn check_accepts_csha_mode_flag() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("trivial.nsl");
    fs::write(&src_path, NO_TRAIN_SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check")
        .arg(&src_path)
        .arg("--csha")
        .arg("auto");
    cmd.assert().success();
}

#[test]
fn check_csha_report_emits_note_for_file_without_train_block() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("no_train.nsl");
    fs::write(&src_path, NO_TRAIN_SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&src_path).arg("--csha-report");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("--csha-report"))
        .stderr(predicate::str::contains("no `train(...)` block"));
}

#[test]
fn check_csha_rejects_invalid_mode_value() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("trivial.nsl");
    fs::write(&src_path, NO_TRAIN_SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check")
        .arg(&src_path)
        .arg("--csha")
        .arg("bogus-mode");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "--csha value 'bogus-mode' is not one of auto|boundary|pipeline|block|off",
        ));
}
