//! `nsl build --calibration-data …` must work where cargo/rustc are not on
//! PATH: a shipped `nsl` links with the system C compiler and must never
//! reach for the Rust toolchain, whatever flags it was given.
//!
//! History: added 2026-04-13 with a skip-if-missing branch for a fixture
//! that was never committed, so it did not run until the fixture landed.
//! Note that the calibration harness itself does not fire on the CLI build
//! path (see `calibration_pipeline_integration.rs`); what this proves is
//! that the compile + link with the flag set needs nothing from `~/.cargo`
//! or a rustup toolchain.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/nsl-cli
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn nsl_build_with_calibration_data_succeeds_without_cargo_on_path() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("minimal_model.nsl");
    std::fs::copy(
        workspace_root().join("crates/nsl-cli/tests/fixtures/minimal_model.nsl"),
        &model,
    )
    .expect("copy fixture");

    // A valid NSLB corpus: rank 3, dims [count=1, seq=4, dim=1], 4 f32 zeros.
    let data = dir.path().join("calib.bin");
    let mut blob = b"NSLB".to_vec();
    blob.extend_from_slice(&3u32.to_le_bytes());
    for d in [1u32, 4, 1] {
        blob.extend_from_slice(&d.to_le_bytes());
    }
    blob.extend_from_slice(&[0u8; 16]);
    std::fs::write(&data, blob).unwrap();

    // Strip cargo/rustup/toolchain entries from PATH.
    let orig_path = std::env::var("PATH").unwrap_or_default();
    let separator = if cfg!(windows) { ';' } else { ':' };
    let stripped_path: String = orig_path
        .split(separator)
        .filter(|entry| {
            let low = entry.to_ascii_lowercase();
            !low.contains(".cargo") && !low.contains("rustup") && !low.contains("toolchain")
        })
        .collect::<Vec<_>>()
        .join(&separator.to_string());
    assert!(
        !stripped_path.is_empty(),
        "PATH was nothing but toolchain entries; the linker needs a C compiler somewhere"
    );

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("build")
        .arg(&model)
        .args(["--calibration-data"])
        .arg(&data)
        .args(["--calibrate", "best-effort"])
        .env("PATH", &stripped_path)
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .output()
        .expect("run nsl");

    assert_eq!(
        out.status.code(),
        Some(0),
        "build should succeed without cargo/rustc on PATH.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    // `<stem>` beside the source, `<stem>.exe` on Windows.
    let binary = dir.path().join(format!("minimal_model{}", std::env::consts::EXE_SUFFIX));
    assert!(binary.is_file(), "no linked binary beside the source");
}
