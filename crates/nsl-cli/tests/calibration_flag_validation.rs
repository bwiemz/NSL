//! `nsl build` refuses malformed calibration flags before it reads the
//! source (spec §8). Every case here fails validation, so the model path
//! only has to be a real file; nothing is compiled.

use std::path::PathBuf;
use std::process::Command;

/// A source file that exists. Validation runs before the source is opened,
/// but a path that resolves keeps the failure pinned to the flag under
/// test rather than to a missing file.
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

#[test]
fn calibrate_without_data_errors() {
    let (_o, e, code) = run(&["--calibrate", "best-effort"]);
    assert_ne!(code, 0);
    assert!(e.contains("--calibrate") && e.contains("--calibration-data"), "stderr: {e}");
}

#[test]
fn bad_calibrate_mode_errors() {
    let (_o, e, code) = run(&["--calibration-data", "/tmp/any.bin", "--calibrate", "maybe"]);
    assert_ne!(code, 0);
    assert!(e.contains("required") || e.contains("best-effort"), "stderr: {e}");
}

#[test]
fn nonexistent_calibration_data_errors() {
    let (_o, e, code) = run(&["--calibration-data", "/nonexistent/path/to/data.bin"]);
    assert_ne!(code, 0);
    assert!(e.contains("calibration-data") && e.contains("does not exist"), "stderr: {e}");
}

#[test]
fn bad_extension_errors() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("nsl-calib-bad-ext.jsonl");
    std::fs::write(&p, b"x").unwrap();
    let (_o, e, code) = run(&["--calibration-data", p.to_str().unwrap()]);
    assert_ne!(code, 0);
    assert!(e.contains(".bin") && e.contains(".safetensors"), "stderr: {e}");
}

#[test]
fn samples_zero_errors() {
    let (_o, e, code) = run(&["--calibration-samples", "0"]);
    assert_ne!(code, 0);
    assert!(e.contains("calibration-samples"), "stderr: {e}");
}

/// `--calibration-data` is accepted and then IGNORED: the harness lives in
/// `nsl_codegen::compile_and_calibrate`, which no CLI path calls, so the
/// corpus is validated and dropped. Until the wiring lands that has to be
/// LOUD -- accepting a corpus and silently discarding it is the one behaviour
/// with no defence, and it is what let the gap survive from 2026-05-10 to
/// 2026-09-01.
///
/// Paired with `--calibration-samples 0`, which is rejected immediately after
/// the calibration-data block, so this pins the warning without compiling.
#[test]
fn calibration_data_warns_that_it_is_not_consumed() {
    let dir = std::env::temp_dir().join(format!("nsl_calibwarn_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let corpus = dir.join("c.bin");
    std::fs::write(&corpus, b"NSLB").expect("write corpus");

    let (_o, e, code) = run(&[
        "--calibration-data",
        corpus.to_str().expect("utf-8 path"),
        "--calibration-samples",
        "0",
    ]);

    assert_ne!(code, 0, "the samples=0 refusal still has to fire: {e}");
    assert!(
        e.contains("NOT consumed by `nsl build`"),
        "a supplied calibration corpus must say it is ignored:\n{e}"
    );
    assert!(
        e.contains(corpus.to_str().expect("utf-8 path")),
        "the warning must name the corpus it is discarding:\n{e}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
