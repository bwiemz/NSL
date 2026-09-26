//! `nsl build` with and without `--calibration-data` on a program that has
//! no calibration consumer (`tests/fixtures/minimal_model.nsl`).
//!
//! History: this file was added on 2026-04-13 with a skip-if-missing branch
//! for a fixture that was never committed, so neither test ran until the
//! fixture landed. The build is done on a copy of the fixture in a temp
//! directory because `nsl build` writes `<stem>.o` and the linked binary
//! beside the source.

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/nsl-cli
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Copy the fixture into `dir` and return the copy's path.
fn fixture_in(dir: &Path) -> PathBuf {
    let src = workspace_root().join("crates/nsl-cli/tests/fixtures/minimal_model.nsl");
    let dst = dir.join("minimal_model.nsl");
    std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("copy {}: {e}", src.display()));
    dst
}

/// The smallest `.bin` corpus `nsl_runtime::calibration_data` loads: magic
/// `NSLB`, rank 3, dims `[count=1, seq=4, dim=1]`, then `1*4*1` f32 zeros.
/// (The original tests wrote `1u32 4u32` + 16 bytes with no magic, which
/// `load_bin` rejects. Nothing on the CLI path reads the corpus today, so
/// the shape only matters once the harness fires from `nsl build`.)
fn calibration_bin(dir: &Path) -> PathBuf {
    let path = dir.join("calib.bin");
    let mut blob = b"NSLB".to_vec();
    blob.extend_from_slice(&3u32.to_le_bytes());
    for dim in [1u32, 4, 1] {
        blob.extend_from_slice(&dim.to_le_bytes());
    }
    blob.extend_from_slice(&[0u8; 16]);
    std::fs::write(&path, blob).unwrap();
    path
}

fn run(model: &Path, args: &[&str]) -> (String, String, i32) {
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("build")
        .arg(model)
        .args(args)
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .output()
        .expect("run nsl");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code().unwrap_or(-1),
    )
}

/// Where `nsl build` puts the linked executable: `<stem>` beside the source,
/// `<stem>.exe` on Windows (`linker::default_output_path`).
fn linked_binary(dir: &Path) -> PathBuf {
    dir.join(format!("minimal_model{}", std::env::consts::EXE_SUFFIX))
}

#[test]
fn build_without_calibration_data_is_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let model = fixture_in(dir.path());
    let (_, err, code) = run(&model, &[]);
    assert_eq!(code, 0, "stderr:\n{err}");
    assert!(linked_binary(dir.path()).is_file(), "no linked binary beside the source");
}

/// A `--calibration-data` the build cannot honour is refused, and nothing is
/// built. This replaces two older contracts:
///
/// - `build_with_calibration_data_emits_no_consumer_warning` (ignored since
///   e3ab23ad, 2026-05-10): it wanted the harness to run from `nsl build` and
///   warn that no hook consumed the corpus. The harness lives in
///   `nsl_codegen::compile_and_calibrate`, which nothing calls, so the
///   warning was unreachable.
/// - `build_with_calibration_data_and_no_consumer_still_builds`: it pinned
///   that such a build succeeded, i.e. that the corpus was accepted and
///   dropped.
///
/// Roadmap item 8 settles it the other way: an ignored request is refused.
/// Wiring the harness into the build is future work; when it lands, this
/// test flips to asserting the harness ran.
#[test]
fn build_with_calibration_data_is_refused_and_builds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let model = fixture_in(dir.path());
    let data = calibration_bin(dir.path());
    let (_, err, code) = run(
        &model,
        &["--calibration-data", data.to_str().unwrap(), "--calibrate", "best-effort"],
    );
    assert_ne!(code, 0, "a calibration corpus nothing consumes must be refused:\n{err}");
    assert!(
        err.contains("--calibration-data, --calibrate are refused")
            && err.contains("calibration is not implemented by `nsl build`"),
        "the refusal must name the flags and the reason:\n{err}"
    );
    assert!(!linked_binary(dir.path()).exists(), "a refused build must not link a binary");
}
