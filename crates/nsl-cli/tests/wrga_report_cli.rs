//! WRGA Milestone B.1 Task 2: `nsl build --wrga-report` emits
//! `WrgaPlan::render_report()` output to stdout or to a file.

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

const SRC: &str = r#"from nsl.nn.losses import mse_loss

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

#[test]
fn wrga_report_flag_prints_to_stdout() {
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
        .arg("--wrga-report");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="))
        .stdout(predicate::str::contains("Frozen parameters"));
}

#[test]
fn wrga_report_flag_writes_to_file_when_path_given() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let out_path = tmp.path().join("t_out");
    let report_path = tmp.path().join("report.txt");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src_path)
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out_path)
        .arg("--wrga-report")
        .arg(&report_path);
    cmd.assert().success();

    let contents = fs::read_to_string(&report_path).unwrap();
    assert!(
        contents.contains("=== WRGA Compilation Report ==="),
        "report file missing header; got: {contents}",
    );
}

/// Task 3 (B.1): WRGA decorators must take effect on the shared-lib build path.
///
/// The test uses `--source-ad` (required for WRGA to fire at all in the current
/// wiring) and `--wrga-report` to observe that the bridge is live on that path.
#[test]
fn wrga_report_works_on_shared_library_build_path() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src_path)
        .arg("--shared-lib")
        .arg("--source-ad")
        .arg("--wrga-report");
    cmd.assert()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="));
}

/// Task 3 (B.1): `--wrga-report` without `--source-ad` must fail with a clear error
/// when the source has WRGA decorators, rather than silently producing no plan.
#[test]
fn wrga_report_without_source_ad_errors_when_decorators_present() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src_path)
        .arg("--wrga-report"); // no --source-ad
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--wrga-report requires --source-ad"));
}
