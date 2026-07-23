//! Integration tests for `nsl check --training-report`.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn run_check_with_report(
    fixture_name: &str,
    format: Option<&str>,
) -> (i32, String, String) {
    let root = workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let stdlib_path = root.join("stdlib");

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(&cargo_toml)
        .args(["-p", "nsl-cli"]);
    if cfg!(feature = "cuda") {
        // Feature-compatibility guard: an unfeatured `cargo run -p nsl-cli`
        // REBUILDS AND REPLACES target/debug/nsl as a non-CUDA binary while
        // the workspace suite runs, breaking every concurrently-running test
        // that spawns that path (phantom "CUDA support not compiled"
        // failures inside their trained programs). Forward cuda so the
        // spawned build matches the suite's binary (and is usually a no-op).
        cmd.args(["--features", "cuda"]);
    }
    cmd.args(["--", "check"]);

    match format {
        Some(f) => {
            cmd.arg(format!("--training-report={}", f));
        }
        None => {
            cmd.arg("--training-report");
        }
    }
    cmd.arg(fixture(fixture_name));
    cmd.env("NSL_STDLIB_PATH", &stdlib_path);

    let out = cmd.output().expect("spawn nsl check");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn fase_deferred_text_report_has_expected_fields() {
    let (code, stdout, stderr) = run_check_with_report(
        "training_report_fase_deferred.nsl",
        None,
    );
    assert_eq!(code, 0, "exit code non-zero; stderr:\n{}", stderr);
    assert!(
        stdout.contains("Training Pipeline Report"),
        "missing header; stdout:\n{}",
        stdout
    );
    assert!(
        stdout.contains("grad_accumulation: 4"),
        "missing grad_accumulation line; stdout:\n{}",
        stdout
    );
    assert!(
        stdout.contains("Deferred"),
        "missing 'Deferred' mode; stdout:\n{}",
        stdout
    );
    assert!(
        stdout.contains("AdamW"),
        "missing 'AdamW' optimizer; stdout:\n{}",
        stdout
    );
}

#[test]
fn fase_deferred_json_report_round_trips() {
    let (code, stdout, stderr) = run_check_with_report(
        "training_report_fase_deferred.nsl",
        Some("json"),
    );
    assert_eq!(code, 0, "exit code non-zero; stderr:\n{}", stderr);
    // Strip any non-JSON prefix (e.g. "OK: ..." line before the JSON blob).
    let json_start = stdout.find('{').unwrap_or_else(|| {
        panic!("no JSON object in stdout:\n{}", stdout)
    });
    let json_str = &stdout[json_start..];
    let json: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|e| {
        panic!("invalid JSON: {}\nstdout:\n{}", e, stdout)
    });
    // The JSON should contain train_blocks[0].fase.plan.mode = "Deferred"
    let mode = json
        .pointer("/train_blocks/0/fase/plan/mode")
        .expect("mode path missing");
    assert_eq!(mode, "Deferred", "mode value mismatch; json:\n{}", stdout);

    let accum = json
        .pointer("/train_blocks/0/fase/plan/accumulation")
        .expect("accumulation path missing");
    assert_eq!(accum, 4, "accumulation mismatch; json:\n{}", stdout);
}

#[test]
fn lion_optimizer_triggers_full_buffer_mode() {
    let (code, stdout, stderr) = run_check_with_report(
        "training_report_lion_fallback.nsl",
        None,
    );
    assert_eq!(code, 0, "exit code non-zero; stderr:\n{}", stderr);
    // Lion is in FaseOptimizer::Lion; planner returns FaseMode::FullBuffer.
    assert!(
        stdout.contains("FullBuffer"),
        "Lion should produce FullBuffer mode; stdout:\n{}",
        stdout
    );
    assert!(
        stdout.contains("Lion"),
        "report should mention Lion optimizer; stdout:\n{}",
        stdout
    );
}

#[test]
fn no_train_blocks_reports_empty_gracefully() {
    let (code, stdout, stderr) = run_check_with_report(
        "training_report_no_train.nsl",
        None,
    );
    assert_eq!(code, 0, "exit code non-zero; stderr:\n{}", stderr);
    assert!(
        stdout.contains("No train blocks found"),
        "missing empty-report message; stdout:\n{}",
        stdout
    );
}

#[test]
fn no_flag_produces_no_report_stdout() {
    // Without --training-report, nsl check should produce its normal output
    // (no training-report header).
    let root = workspace_root();
    let cargo_toml = root.join("Cargo.toml");
    let stdlib_path = root.join("stdlib");

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(&cargo_toml)
        .args(["-p", "nsl-cli"]);
    if cfg!(feature = "cuda") {
        // Feature-compatibility guard: an unfeatured `cargo run -p nsl-cli`
        // REBUILDS AND REPLACES target/debug/nsl as a non-CUDA binary while
        // the workspace suite runs, breaking every concurrently-running test
        // that spawns that path (phantom "CUDA support not compiled"
        // failures inside their trained programs). Forward cuda so the
        // spawned build matches the suite's binary (and is usually a no-op).
        cmd.args(["--features", "cuda"]);
    }
    let out = cmd
        .args(["--", "check"])
        .arg(fixture("training_report_fase_deferred.nsl"))
        .env("NSL_STDLIB_PATH", &stdlib_path)
        .output()
        .expect("spawn nsl check");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code().unwrap_or(-1),
        0,
        "exit code non-zero; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !stdout.contains("Training Pipeline Report"),
        "report should only emit when flag is present; stdout:\n{}",
        stdout
    );
}
