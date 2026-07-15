//! CPDT Part II activation — S3: `nsl run --wggo` CLI surface tests.
//!
//! `nsl build` already exposes `--wggo`, `--wggo-report`, `--wggo-weights`,
//! `--wggo-importance`, and `--wggo-prune-fraction`. Until S3, `nsl run`
//! hardcoded all five to defaults, blocking end-to-end activation of the
//! Part II FP16 optimizer wrap (which requires the WGGO mode-table dispatch
//! to reach `emit_unified_optim_step_dispatch`).
//!
//! These tests pin the four validation gates that ride alongside the new
//! surface plus a happy-path smoke that confirms `--wggo full` survives the
//! whole compile + execute pipeline on the cpdt_precision_fp16 fixture.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;
use std::process::Command;

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

fn fp16_fixture() -> PathBuf {
    workspace_root()
        .join("crates")
        .join("nsl-codegen")
        .join("tests")
        .join("fixtures")
        .join("cpdt_precision_fp16.nsl")
}

#[test]
fn run_wggo_invalid_mode_rejected() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run")
        .arg(fp16_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("BOGUS");
    cmd.assert().failure().stderr(predicate::str::contains(
        "--wggo value 'BOGUS' is not one of full|greedy|off|auto",
    ));
}

#[test]
fn run_wggo_prune_fraction_out_of_range_rejected() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run")
        .arg(fp16_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg("--wggo-prune-fraction")
        .arg("1.5");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "--wggo-prune-fraction must be in [0.0, 0.9]",
        ));
}

#[test]
fn run_wggo_memory_budget_zero_rejected() {
    // 0 MiB is meaningless — reject loudly rather than silently disabling
    // the budget driver.
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run")
        .arg(fp16_fixture())
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
fn run_wggo_memory_budget_unparseable_rejected() {
    // A non-numeric value is rejected by clap's u64 parse before our
    // validation runs.
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run")
        .arg(fp16_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg("--wggo-memory-budget")
        .arg("not-a-number");
    cmd.assert().failure();
}

#[test]
fn run_wggo_weights_missing_path_rejected() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run")
        .arg(fp16_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg("--wggo-weights")
        .arg("/no/such/path/__definitely_missing__.nslweights");
    cmd.assert().failure().stderr(predicate::str::contains(
        "--wggo-weights path does not exist",
    ));
}

#[test]
fn run_wggo_full_parses_and_threads() {
    // Happy path: clap accepts the flag, the validation block passes, and the
    // pipeline executes the train block to completion. Pre-S3, clap would
    // reject `--wggo` as an unknown flag on the Run subcommand.
    //
    // Pre-S1 this fixture aborted at runtime with
    // `nsl_tensor_add_inplace: dst len 64 != src len 128` (broadcast Mul
    // missing reduce_to_shape on its backward); S1 fixed that. This test
    // confirms S3's --wggo plumbing doesn't reintroduce a crash via the
    // unified-dispatch arm.
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    // Run from a temp cwd so the fixture's `model_save("cpdt_precision_out.nslm")`
    // doesn't pollute the repo root.
    let tmp = tempfile::TempDir::new().unwrap();
    cmd.current_dir(tmp.path());
    cmd.arg("run")
        .arg(fp16_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full");
    cmd.assert().success();
}
