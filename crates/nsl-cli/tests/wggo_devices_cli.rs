//! G10: the `build` command accepts `--devices N`, threading the cluster
//! `world_size` into WGGO's ZeRO-sharding budget.
//!
//! Before this flag existed, `build` hardcoded `world_size = 1`, so multi-GPU
//! ZeRO sharding could never be triggered from the CLI even though the planner
//! supports it (see `wggo::run_on_wengert_propagates_world_size`). These tests
//! pin (1) the flag is documented in `build --help`, and (2) a real build with
//! `--devices N --wggo full` parses and runs the WGGO pass to completion.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Minimal `@train`-bearing fixture WGGO can plan over.  Uses an INLINE
/// `sum(out)` loss (like `cep_canonical_with_train.nsl`) rather than importing
/// `nsl.nn.losses`, so the test is independent of the heavy stdlib loss path.
const SRC: &str = r#"model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

#[test]
fn build_help_documents_devices_flag() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("build").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--devices"));
}

#[test]
fn build_help_documents_memory_budget_flag() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("build").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("--wggo-memory-budget"));
}

#[test]
fn build_rejects_zero_memory_budget() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg("--wggo-memory-budget")
        .arg("0");
    cmd.assert().failure().stderr(predicate::str::contains(
        "--wggo-memory-budget must be > 0 MiB",
    ));
}

#[test]
fn build_accepts_generous_memory_budget_and_runs_wggo() {
    // A generous budget (64 GiB) fits the tiny fixture: the WGGO pass runs to
    // completion (report renders) and the implied --wggo-moment-precision does
    // not derail the build. A too-small budget would hard-fail instead.
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let out_path = tmp.path().join("t_out");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wggo")
        .arg("full")
        .arg("--wggo-report")
        .arg("--wggo-memory-budget")
        .arg("65536");
    cmd.assert().success().stderr(predicate::str::contains(
        "=== WGGO Global Optimization Report ===",
    ));
}

#[test]
fn build_accepts_devices_flag_and_runs_wggo() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let out_path = tmp.path().join("t_out");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--devices")
        .arg("4")
        .arg("--wggo")
        .arg("full")
        .arg("--wggo-report");
    // The WGGO report renders to stderr; its presence proves the `--devices`
    // build parsed and the global-optimization pass executed.
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "=== WGGO Global Optimization Report ===",
        ));
}
