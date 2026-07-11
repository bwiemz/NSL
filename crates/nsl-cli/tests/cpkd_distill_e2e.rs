//! CPKD distill-block end-to-end tests via the `nsl` CLI.
//!
//! Covers the v1 acceptance surface:
//! - `nsl check` accepts the basic and fused fixtures;
//! - loud refusals: attn_transfer=true, @fused_kl_ce on a train block,
//!   distill without teacher=, tape-fallback refusal is compile-side;
//! - `nsl run` executes a full distillation loop on CPU (composite CE+MSE
//!   proxy loss) and the loss DECREASES — proof the student learns while
//!   the frozen teacher provides a stable target (I-11);
//! - the Distillation Build Report renders on stderr with the fused
//!   KL-CE line when @fused_kl_ce fires.

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

fn fixture_path(name: &str) -> PathBuf {
    workspace_root()
        .join("crates")
        .join("nsl-codegen")
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn write_temp_fixture(tag: &str, source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nsl_cpkd_e2e_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, source).expect("write fixture");
    file
}

#[test]
fn nsl_check_accepts_basic_distill_block() {
    let fixture = fixture_path("cpkd_distill_basic.nsl");
    assert!(fixture.exists(), "missing: {}", fixture.display());
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

#[test]
fn nsl_check_accepts_fused_kl_ce_distill_block() {
    let fixture = fixture_path("cpkd_distill_fused_kl_ce.nsl");
    assert!(fixture.exists(), "missing: {}", fixture.display());
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

/// attn_transfer = true is a DEFERRED feature and must refuse loudly at
/// the semantic layer (attention probabilities are never materialized in
/// HBM by the fused attention kernels).
#[test]
fn nsl_check_refuses_attn_transfer_true() {
    let src = r#"model M:
    w: Tensor = randn([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let teacher = M()
let student = M()

distill(teacher = teacher, student = student, epochs = 1):
    optimizer: SGD(lr = 0.01)
    loss:
        attn_transfer = true
    step(batch):
        let loss = student.forward(randn([4, 8]))
"#;
    let file = write_temp_fixture("attn", src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert().failure().stderr(predicate::str::contains(
        "attention transfer is not implemented in CPKD v1",
    ));
}

/// @fused_kl_ce is distill-block-only (mirrors @fused_lm_ce/train gating).
#[test]
fn nsl_check_refuses_fused_kl_ce_on_train_block() {
    let src = r#"model M:
    w: Tensor = randn([8, 8])

let m = M()

@fused_kl_ce(enabled = true)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let loss = randn([1])
"#;
    let file = write_temp_fixture("wrongtarget", src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert().failure().stderr(predicate::str::contains(
        "@fused_kl_ce may only be applied to a `distill` block",
    ));
}

#[test]
fn nsl_check_refuses_distill_without_teacher() {
    let src = r#"model M:
    w: Tensor = randn([8, 8])

let student = M()

distill(student = student, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let loss = student.forward(randn([4, 8]))
"#;
    let file = write_temp_fixture("noteacher", src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert().failure().stderr(predicate::str::contains(
        "distill block requires both teacher= and student=",
    ));
}

/// Full CPU distillation run: 8 epochs of the composite CE + logit-MSE
/// proxy loss. The loss stream must be finite and strictly lower at the
/// end than at the start (the student learns; the frozen teacher's
/// contribution is stable across epochs — its weights never move because
/// no teacher gradient exists, I-11).
#[test]
fn nsl_run_distill_loss_decreases() {
    let base = std::fs::read_to_string(fixture_path("cpkd_distill_basic.nsl")).unwrap();
    let scaled = base
        .replace("epochs = 1", "epochs = 8")
        .replace("lr = 0.001", "lr = 0.05");
    assert_ne!(base, scaled, "fixture knobs not found — resync test");
    let file = write_temp_fixture("run", &scaled);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run").arg("--source-ad").arg(&file);
    let output = cmd.output().expect("nsl run failed to spawn");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "nsl run failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let losses: Vec<f64> = stdout
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(
        losses.len() >= 8,
        "expected >= 8 per-epoch loss prints, got {} .\nstdout:\n{stdout}",
        losses.len()
    );
    assert!(
        losses.iter().all(|v| v.is_finite()),
        "non-finite loss: {losses:?}"
    );
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss did not decrease: {losses:?}"
    );
}

/// The Distillation Build Report must render on stderr with the fused
/// KL-CE line and the I-11 freeze evidence when @fused_kl_ce fires.
#[test]
fn distillation_build_report_renders_fused_line() {
    let fixture = fixture_path("cpkd_distill_fused_kl_ce.nsl");
    let out_dir = std::env::temp_dir().join(format!("nsl_cpkd_report_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&out_dir);
    let out = out_dir.join("cpkd_fused_bin");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg("--source-ad")
        .arg(&fixture)
        .arg("-o")
        .arg(&out);
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("=== CPKD Distillation Build Report ==="))
        .stderr(predicate::str::contains(
            "Fused KL-CE: teacher+student logits never materialized",
        ))
        .stderr(predicate::str::contains(
            "Teacher freeze (I-11): teacher backward structurally absent",
        ));
    let _ = std::fs::remove_dir_all(&out_dir);
}
