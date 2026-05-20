use std::process::Command;

fn nsl_binary() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("target/debug/nsl");
    #[cfg(windows)]
    p.set_extension("exe");
    p
}

fn run(args: &[&str]) -> (String, String, i32) {
    let out = Command::new(nsl_binary())
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
fn build_without_calibration_data_is_unchanged() {
    let model = "crates/nsl-cli/tests/fixtures/minimal_model.nsl";
    if !std::path::Path::new(model).exists() {
        return;
    }
    let (_, _, code) = run(&["build", model]);
    assert_eq!(code, 0);
}

#[test]
fn build_with_calibration_data_emits_no_consumer_warning() {
    let model = "crates/nsl-cli/tests/fixtures/minimal_model.nsl";
    if !std::path::Path::new(model).exists() {
        return;
    }
    let mut data = std::env::temp_dir();
    data.push("nsl-calib-integration.bin");
    let mut blob = Vec::new();
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&4u32.to_le_bytes());
    blob.extend_from_slice(&[0u8; 16]);
    std::fs::write(&data, blob).unwrap();

    let (_, err, code) = run(&[
        "build",
        model,
        "--calibration-data",
        data.to_str().unwrap(),
        "--calibrate",
        "best-effort",
    ]);
    let _ = std::fs::remove_file(&data);
    assert_eq!(code, 0, "best-effort should not fail the build");
    assert!(
        err.contains("no calibration hooks registered") || err.contains("no consumers"),
        "expected a no-consumer warning in stderr, got: {err}"
    );
}
