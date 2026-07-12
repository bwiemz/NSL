//! Dev-tools paper completion: `nsl profile` real-path e2e.
//!
//! With a train-block program, `--explain-wggo` must explain the USER'S
//! model (no synthetic-model disclaimer) and `--memory` must render the
//! real-liveness timeline with the paper's phase markers; without a train
//! block, both fall back to the labeled synthetic/approximate paths.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;
use std::process::Command;

fn write_temp_fixture(tag: &str, source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "nsl_profile_real_{tag}_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, source).expect("write fixture");
    file
}

const TRAIN_SRC: &str = r#"model Tiny:
    w1: Tensor = randn([32, 16])
    w2: Tensor = randn([16, 4])

    fn forward(self, x: Tensor) -> Tensor:
        let h = x @ self.w1
        let y = h @ self.w2
        return y

let m = Tiny()

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let x = randn([8, 32])
        let out = m.forward(x)
        let loss = sum(out)
"#;

const PLAIN_SRC: &str = "fn main() -> int:\n    return 0\n";

#[test]
fn explain_wggo_uses_real_train_block() {
    let file = write_temp_fixture("explain", TRAIN_SRC);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("profile").arg("--explain-wggo").arg(&file);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== WGGO Decision Explanation ==="))
        .stdout(predicate::str::contains("SYNTHETIC two-block").not());
}

#[test]
fn explain_wggo_falls_back_to_synthetic_without_train_block() {
    let file = write_temp_fixture("explain_fb", PLAIN_SRC);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("profile").arg("--explain-wggo").arg(&file);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("SYNTHETIC two-block"))
        .stderr(predicate::str::contains("no source-AD train block"));
}

#[test]
fn memory_renders_real_liveness_timeline_for_train_block() {
    let file = write_temp_fixture("memory", TRAIN_SRC);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("profile").arg("--memory").arg(&file);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Memory Timeline (real liveness)"))
        .stdout(predicate::str::contains("backward begins"))
        .stdout(predicate::str::contains("Peak:"));
}

#[test]
fn memory_keeps_labeled_approximation_without_train_block() {
    let file = write_temp_fixture("memory_fb", PLAIN_SRC);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("profile").arg("--memory").arg(&file);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("APPROXIMATE timeline"));
}

#[test]
fn html_flag_writes_selfcontained_report() {
    let file = write_temp_fixture("html", TRAIN_SRC);
    let html_path = file.with_file_name("report.html");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("profile")
        .arg("--memory")
        .arg("--html")
        .arg(&html_path)
        .arg(&file);
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("HTML profile report written"));
    let html = std::fs::read_to_string(&html_path).expect("HTML report exists");
    assert!(html.to_lowercase().starts_with("<!doctype html"));
    assert!(html.contains("<svg"));
    assert!(!html.contains("NaN"));
}

#[test]
fn json_report_carries_what_if_and_peak_fields() {
    let file = write_temp_fixture("json", TRAIN_SRC);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.arg("profile").arg("--memory").arg("--json").arg(&file);
    let out = cmd.output().expect("spawn nsl profile");
    assert!(out.status.success());
    let json: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("valid JSON report");
    assert_eq!(json["memory_timeline_approximate"], false);
    assert!(json["memory_peak_bytes"].as_u64().is_some());
    // what-if array present (possibly empty, but the field must exist and
    // for this fixture FASE should fire: param grads live to the end).
    assert!(json["memory_what_if"].is_array());
}
