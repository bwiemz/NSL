//! WRGA paper §8.3 — `nsl check --wrga-compare` integration tests.
//!
//! Closes audit Gap #3: the paper specifies a PEFT comparison CLI that
//! contrasts WRGA's measured numbers against LoRA / AdaLoRA / GaLore / ReFT
//! baselines. Before this PR there was no comparison surface — only the
//! existing `nsl build --wrga-report` and (since Gap #2 / PR #262)
//! `nsl check --wrga-analyze`.

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

const SRC_WITH_DECORATORS: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@freeze(include=["m.w"])
let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

const SRC_NO_DECORATORS: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

/// Paper §8.3 form: `nsl check --wrga-compare model.nsl` prints the
/// comparison report to stdout (no path argument → default `-`). The output
/// must contain the comparison header and every baseline row.
#[test]
fn wrga_compare_no_path_prints_full_comparison_table() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-compare");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WRGA PEFT Comparison Report ==="))
        .stdout(predicate::str::contains("WRGA (this run)"))
        .stdout(predicate::str::contains("LoRA (r=16)"))
        .stdout(predicate::str::contains("AdaLoRA"))
        .stdout(predicate::str::contains("GaLore"))
        .stdout(predicate::str::contains("ReFT (r=4, top-25%)"));
}

/// `--wrga-compare=path.txt` writes to file and emits nothing of substance
/// on stdout (no comparison header).
#[test]
fn wrga_compare_with_path_writes_to_file() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let report_path = tmp.path().join("cmp.txt");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg(format!("--wrga-compare={}", report_path.display()));
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WRGA PEFT Comparison Report ===").not());

    let contents = fs::read_to_string(&report_path).unwrap();
    assert!(contents.contains("=== WRGA PEFT Comparison Report ==="));
    assert!(contents.contains("LoRA (r=16)"));
    assert!(contents.contains("§9 benchmark suite"));
}

/// `--wrga-compare` must suppress the analyze report — only the comparison
/// table should appear on stdout. Catches a future regression where the
/// capture-and-suppress wiring is accidentally broken so both reports print.
#[test]
fn wrga_compare_suppresses_analyze_report_on_stdout() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-compare");
    cmd.assert()
        .success()
        // The analyze header should NOT appear when --wrga-compare is used.
        .stdout(predicate::str::contains("=== WRGA Compilation Report ===").not())
        // But the compare header must.
        .stdout(predicate::str::contains("=== WRGA PEFT Comparison Report ==="));
}

/// `--wrga-target` overrides the WRGA target GPU; the compare report's
/// "Target hardware:" line must reflect the override.
#[test]
fn wrga_compare_with_target_overrides_gpu_in_report() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-compare")
        .arg("--wrga-target")
        .arg("H100-SXM");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Target hardware: H100-SXM"));
}

/// No `@wrga` / `@freeze` / `@adapter` decorators → exit 2 with distinct
/// stderr (so CI can distinguish absence from compile failure). Mirrors the
/// analyze path's contract.
#[test]
fn wrga_compare_no_decorators_exits_with_distinct_code() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_NO_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-compare");
    cmd.assert()
        .code(2)
        .stderr(predicate::str::contains("no @wrga / @freeze / @adapter decorators"));
}

/// `--wrga-analyze` and `--wrga-compare` are mutually exclusive — passing
/// both is a user error and must fail fast with a clear message.
#[test]
fn wrga_analyze_and_wrga_compare_are_mutually_exclusive() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-compare");
    cmd.assert()
        .code(1)
        .stderr(predicate::str::contains(
            "--wrga-analyze and --wrga-compare are mutually exclusive",
        ));
}

/// Side-effect-free contract: source-directory contents are unchanged after
/// the call. Mirrors the analyze path's test — lists the whole directory so
/// the assertion cannot pass vacuously by checking a specific filename.
#[test]
fn wrga_compare_does_not_emit_object_file() {
    let tmp = TempDir::new().unwrap();
    let src_dir = tmp.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    let src_path = src_dir.join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-compare");
    cmd.assert().success();

    let after: Vec<String> = fs::read_dir(&src_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().into_string().unwrap_or_default())
        .collect();
    assert_eq!(
        after,
        vec!["t.nsl".to_string()],
        "--wrga-compare leaked artifacts into source dir: {after:?}"
    );
}
