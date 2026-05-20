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
    cmd.assert().failure().stderr(predicate::str::contains(
        "--wrga-report requires --source-ad",
    ));
}

/// Task 4 (B.2): `--wrga-report` must emit the report header on the `--zk-circuit`
/// build path (or at least produce a plan via the `_returning_plan` variant).
#[test]
fn wrga_report_works_on_zk_build_path() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src_path)
        .arg("--zk-circuit")
        .arg("--source-ad")
        .arg("--wrga-report");
    cmd.assert()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="));
}

/// Write a minimal .safetensors file containing a single f32 tensor "w" of
/// shape [2, 1] with zero values — enough to satisfy
/// `--standalone`'s `-w/--weights` requirement for the WRGA-report smoke test.
fn write_dummy_safetensors(path: &std::path::Path) {
    // Header: {"w": {"dtype":"F32","shape":[2,1],"data_offsets":[0,8]}}
    let header = r#"{"w":{"dtype":"F32","shape":[2,1],"data_offsets":[0,8]}}"#;
    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u64;
    let mut out = Vec::with_capacity(8 + header_bytes.len() + 8);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    // 2 f32 zeros = 8 bytes
    out.extend_from_slice(&[0u8; 8]);
    fs::write(path, &out).unwrap();
}

/// Task 4 (B.2): `--wrga-report` must emit the report header on the `--standalone`
/// build path.
#[test]
fn wrga_report_works_on_standalone_build_path() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    let w_path = tmp.path().join("w.safetensors");
    fs::write(&src_path, SRC).unwrap();
    write_dummy_safetensors(&w_path);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("build")
        .arg(&src_path)
        .arg("--standalone")
        .arg("-w")
        .arg(&w_path)
        .arg("--source-ad")
        .arg("--wrga-report");
    cmd.assert()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="));
}
