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
    let out = Command::new(nsl_binary()).args(args).output().expect("run nsl");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code().unwrap_or(-1),
    )
}

#[test]
fn calibrate_without_data_errors() {
    let (_o, e, code) = run(&["build", "models/coder-rl/train_sft.nsl", "--calibrate", "best-effort"]);
    assert_ne!(code, 0);
    assert!(e.contains("--calibrate") && e.contains("--calibration-data"));
}

#[test]
fn bad_calibrate_mode_errors() {
    let (_o, e, code) = run(&[
        "build", "models/coder-rl/train_sft.nsl",
        "--calibration-data", "/tmp/any.bin",
        "--calibrate", "maybe",
    ]);
    assert_ne!(code, 0);
    assert!(e.contains("required") || e.contains("best-effort"));
}

#[test]
fn nonexistent_calibration_data_errors() {
    let (_o, e, code) = run(&[
        "build", "models/coder-rl/train_sft.nsl",
        "--calibration-data", "/nonexistent/path/to/data.bin",
    ]);
    assert_ne!(code, 0);
    assert!(e.contains("calibration-data") && e.contains("does not exist"));
}

#[test]
fn bad_extension_errors() {
    let mut p = std::env::temp_dir();
    p.push("nsl-calib-bad-ext.jsonl");
    std::fs::write(&p, b"x").unwrap();
    let (_o, e, code) = run(&[
        "build", "models/coder-rl/train_sft.nsl",
        "--calibration-data", p.to_str().unwrap(),
    ]);
    let _ = std::fs::remove_file(&p);
    assert_ne!(code, 0);
    assert!(e.contains(".bin") && e.contains(".safetensors"));
}

#[test]
fn samples_zero_errors() {
    let (_o, e, code) = run(&[
        "build", "models/coder-rl/train_sft.nsl",
        "--calibration-samples", "0",
    ]);
    assert_ne!(code, 0);
    assert!(e.contains("calibration-samples"));
}
