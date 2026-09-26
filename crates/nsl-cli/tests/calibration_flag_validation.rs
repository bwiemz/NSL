//! `nsl build` refuses every calibration flag before it reads the source
//! (roadmap item 8: an ignored request is either implemented or refused).
//!
//! The calibration harness lives in `nsl_codegen::compile_and_calibrate`,
//! which no build path calls, so `--calibration-data` used to be validated
//! and then dropped (with a warning), and `--calibrate`,
//! `--calibration-samples`, `--calibration-batch-size` and
//! `--calibration-timeout` configured a run that never happened. Each is now
//! a refusal that names the flag, so the model path only has to be a real
//! file; nothing is compiled.

use std::path::PathBuf;
use std::process::Command;

/// A source file that exists. The refusal fires before the source is
/// opened, but a path that resolves keeps the failure pinned to the flag
/// under test rather than to a missing file.
fn model() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/nsl-cli
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("models/coder-rl/train_sft.nsl")
}

fn run(args: &[&str]) -> (String, String, i32) {
    let model = model();
    assert!(model.is_file(), "fixture missing: {}", model.display());
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("build")
        .arg(&model)
        .args(args)
        .output()
        .expect("run nsl");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code().unwrap_or(-1),
    )
}

/// The refusal names the flag, says calibration is not implemented, and
/// points WGGO users at a scoring mode that needs no data.
fn assert_refused(args: &[&str], flags: &str) {
    let (_o, e, code) = run(args);
    assert_ne!(code, 0, "{args:?} must refuse:\n{e}");
    assert!(e.contains(&format!("error: {flags}")), "{args:?}: the refusal must name {flags}:\n{e}");
    assert!(e.contains("calibration is not implemented by `nsl build`"), "{args:?}:\n{e}");
    assert!(e.contains("--wggo-importance=magnitude"), "{args:?}: the fix must be named:\n{e}");
}

#[test]
fn calibration_data_is_refused_not_ignored() {
    let dir = tempfile::tempdir().unwrap();
    let corpus = dir.path().join("c.bin");
    std::fs::write(&corpus, b"NSLB").unwrap();
    assert_refused(&["--calibration-data", corpus.to_str().unwrap()], "--calibration-data is refused");
    // The old validated-then-ignored warning is gone.
    let (_o, e, _) = run(&["--calibration-data", corpus.to_str().unwrap()]);
    assert!(!e.contains("NOT consumed"), "{e}");
}

/// Refused before any path check: a missing or oddly named corpus gets the
/// same answer, because no corpus can be honoured.
#[test]
fn calibration_data_refuses_before_validating_the_path() {
    assert_refused(&["--calibration-data", "/nonexistent/path/to/data.bin"], "--calibration-data is refused");
    assert_refused(&["--calibration-data", "/tmp/any.jsonl"], "--calibration-data is refused");
}

#[test]
fn each_calibration_knob_is_refused_on_its_own() {
    for (args, flag) in [
        (&["--calibrate", "best-effort"][..], "--calibrate"),
        (&["--calibrate", "required"][..], "--calibrate"),
        (&["--calibration-samples", "64"][..], "--calibration-samples"),
        (&["--calibration-batch-size", "4"][..], "--calibration-batch-size"),
        (&["--calibration-timeout", "30"][..], "--calibration-timeout"),
    ] {
        assert_refused(args, &format!("{flag} is refused"));
    }
}

#[test]
fn several_flags_are_named_together() {
    assert_refused(
        &["--calibration-data", "/tmp/c.bin", "--calibration-samples", "8"],
        "--calibration-data, --calibration-samples are refused",
    );
}
