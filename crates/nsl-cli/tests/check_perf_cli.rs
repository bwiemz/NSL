//! M37 deferred-closure: `nsl check --perf/--gpu/--trace` must not be
//! silently dormant.
//!
//! Before this fix, all three flags were parsed by clap and then dropped
//! on the floor (`perf: _perf, gpu: _gpu, trace: _trace` in the Check
//! dispatch) — `nsl check file.nsl --perf` printed only the plain "OK"
//! line, exit 0, no roofline report, and `--trace out.json` wrote no
//! file. Docs (docs/summaries/01/02/05) actively advertise
//! `nsl check --perf` as "Roofline analysis", so users were directed
//! straight into the silent path. This test pins the honest behavior:
//!   1. `--perf` renders the predictive-profile report (same engine as
//!      `nsl profile`) after the OK line.
//!   2. `--gpu <target>` selects the roofline target, and is refused
//!      without `--perf` (alone it would do nothing).
//!   3. `--trace` is refused loudly ("not implemented") and creates no
//!      output file — deferral must refuse.
//!   4. Plain `nsl check` is unchanged (regression guard).

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::process::Command;
use tempfile::TempDir;

/// Small fn-only fixture with concrete matmul shapes so the profile
/// walker emits non-empty rows under `--entry auto` defaults
/// (batch=1, seq=2048 are irrelevant here — shapes are literal).
const MATMUL_FN_SRC: &str = r#"fn forward(
    a: Tensor<[64, 128], bf16>,
    b: Tensor<[128, 256], bf16>,
) -> Tensor<[64, 256], bf16>:
    let c = a @ b
    return c
"#;

fn write_fixture(tmp: &TempDir) -> std::path::PathBuf {
    let src_path = tmp.path().join("matmul_fn.nsl");
    fs::write(&src_path, MATMUL_FN_SRC).unwrap();
    src_path
}

#[test]
fn check_perf_renders_predictive_profile() {
    let tmp = TempDir::new().unwrap();
    let src_path = write_fixture(&tmp);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("check").arg(&src_path).arg("--perf");
    cmd.assert()
        .success()
        // Plain check output still leads.
        .stdout(predicate::str::contains("OK:"))
        // The roofline report follows — same renderer as `nsl profile`.
        .stdout(predicate::str::contains("NSL Predictive Profile"))
        .stdout(predicate::str::contains("matmul"))
        .stdout(predicate::str::contains("Totals:"));
}

#[test]
fn check_perf_with_gpu_targets_named_gpu() {
    let tmp = TempDir::new().unwrap();
    let src_path = write_fixture(&tmp);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("check")
        .arg(&src_path)
        .arg("--perf")
        .arg("--gpu")
        .arg("A100-PCIe");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("NSL Predictive Profile"))
        .stdout(predicate::str::contains("A100-PCIe"));
}

#[test]
fn check_gpu_without_perf_is_refused() {
    let tmp = TempDir::new().unwrap();
    let src_path = write_fixture(&tmp);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("check").arg(&src_path).arg("--gpu").arg("H100");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--gpu requires --perf"));
}

#[test]
fn check_trace_is_refused_and_writes_no_file() {
    let tmp = TempDir::new().unwrap();
    let src_path = write_fixture(&tmp);
    let trace_path = tmp.path().join("out.json");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("check")
        .arg(&src_path)
        .arg("--trace")
        .arg(trace_path.to_str().unwrap());
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--trace is not implemented"))
        .stderr(predicate::str::contains("nsl debug"));
    assert!(
        !trace_path.exists(),
        "refused --trace must not create the output file"
    );
}

#[test]
fn plain_check_still_succeeds_without_report() {
    let tmp = TempDir::new().unwrap();
    let src_path = write_fixture(&tmp);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("check").arg(&src_path);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"))
        .stdout(predicate::str::contains("NSL Predictive Profile").not());
}
