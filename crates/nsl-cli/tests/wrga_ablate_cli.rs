//! WRGA paper §9.3 — `nsl check --wrga-ablate=<flags>` integration tests.
//!
//! Closes audit Gap #4: the paper specifies five independent ablations
//! (without_wengert, without_roofline, without_spectral, without_fusion,
//! without_memory). The CLI flag combines with `--wrga-analyze` or
//! `--wrga-compare` so users can fill in the §9.3 table by running each
//! ablation in sequence.

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

/// `nsl check --wrga-analyze --wrga-ablate=fusion` surfaces the Ablation:
/// header line so users can tell a baseline run apart from an ablation run.
#[test]
fn wrga_ablate_fusion_surfaces_ablation_header() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-ablate=fusion");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="))
        .stdout(predicate::str::contains("Ablation: skip=fusion"));
}

/// Multi-flag form: `--wrga-ablate=wengert,fusion` skips Innovations 1 and 4.
/// Header lists the active flags in canonical order.
#[test]
fn wrga_ablate_multi_flag_lists_all_active_in_header() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-ablate=wengert,fusion");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Ablation: skip=wengert,fusion"));
}

/// `--wrga-ablate=all` skips every Innovation — the canonical §9.3
/// "absolute baseline" measurement.
#[test]
fn wrga_ablate_all_skips_every_innovation() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-ablate=all");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "Ablation: skip=wengert,roofline,spectral,fusion,memory",
        ));
}

/// `--wrga-ablate=none` is a no-op — no Ablation: header line, regular
/// WRGA Compilation Report output.
#[test]
fn wrga_ablate_none_suppresses_ablation_header() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-ablate=none");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="))
        .stdout(predicate::str::contains("Ablation:").not());
}

/// Unknown ablation flag → exit 1 with a clear error listing the valid set.
#[test]
fn wrga_ablate_unknown_flag_fails_fast() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-ablate=spectral,bogus,fusion");
    cmd.assert()
        .code(1)
        .stderr(predicate::str::contains("unknown ablation flag 'bogus'"))
        .stderr(predicate::str::contains(
            "valid: wengert, roofline, spectral, fusion, memory, all, none",
        ));
}

/// `--wrga-ablate` without `--wrga-analyze` or `--wrga-compare` is a user
/// error — fails fast with a clear message.
#[test]
fn wrga_ablate_without_analyze_or_compare_fails_fast() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-ablate=fusion");
    cmd.assert()
        .code(1)
        .stderr(predicate::str::contains(
            "--wrga-ablate requires --wrga-analyze or --wrga-compare",
        ));
}

/// `--wrga-ablate=fusion` paired with `--wrga-compare` flows through the
/// compare path and renders the PEFT comparison table with the ablation
/// header. Pins the cross-command integration.
#[test]
fn wrga_ablate_combines_with_wrga_compare() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-compare")
        .arg("--wrga-ablate=fusion");
    cmd.assert()
        .success()
        // The compare report is what shows; it doesn't carry the Ablation:
        // header (that's on the analyze report). The fusion column on the
        // WRGA row drops to 0% because Innovation 4 is skipped.
        .stdout(predicate::str::contains("=== WRGA PEFT Comparison Report ==="))
        .stdout(predicate::str::contains("WRGA (this run)"));
}

/// `--wrga-compare --wrga-ablate=spectral` emits a warning explaining that
/// the SVD-disabled run cannot recover per-site weight shapes, so the PEFT
/// table's LoRA/AdaLoRA/ReFT rows will be zero. Pins the warning so a
/// future refactor that loses it produces a silently-wrong table.
#[test]
fn wrga_ablate_spectral_with_compare_emits_warning() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-compare")
        .arg("--wrga-ablate=spectral");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "--wrga-ablate=spectral disables the SVD",
        ));
}

/// `--wrga-analyze --wrga-ablate=spectral` does NOT emit the comparison
/// warning — analyze just shows the bare ablated plan and is fine without
/// SVD shape data.
#[test]
fn wrga_ablate_spectral_with_analyze_does_not_emit_compare_warning() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-ablate=spectral");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("--wrga-ablate=spectral disables the SVD").not());
}

/// Side-effect-free contract is preserved under ablation — same as for
/// analyze/compare. Pins that the temp-dir cleanup applies regardless of
/// whether ablation is active.
#[test]
fn wrga_ablate_does_not_emit_object_file() {
    let tmp = TempDir::new().unwrap();
    let src_dir = tmp.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    let src_path = src_dir.join("t.nsl");
    fs::write(&src_path, SRC_WITH_DECORATORS).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path())
        .arg("check")
        .arg(&src_path)
        .arg("--wrga-analyze")
        .arg("--wrga-ablate=all");
    cmd.assert().success();

    let after: Vec<String> = fs::read_dir(&src_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.file_name().into_string().unwrap_or_default())
        .collect();
    assert_eq!(
        after,
        vec!["t.nsl".to_string()],
        "--wrga-ablate must not leak artifacts: {after:?}"
    );
}
