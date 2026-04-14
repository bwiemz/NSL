//! Verifies that `nsl build --calibration-data ...` works in an
//! environment where cargo/rustc are not on PATH.
//!
//! Current scope: stmt.rs's discover_awq_projections returns None,
//! so the calibration path takes the "no hooks registered" branch.
//! The test therefore only proves the CLI doesn't accidentally invoke
//! cargo/rustc when --calibration-data is passed.  A follow-up plan
//! that wires discover_awq_projections will upgrade this test
//! automatically — the same invocation will exercise the real
//! emit+link+spawn path.

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

#[test]
fn nsl_build_with_calibration_data_succeeds_without_cargo_on_path() {
    // The test needs a minimal fixture model; skip if absent.
    let model = "crates/nsl-cli/tests/fixtures/minimal_model.nsl";
    if !std::path::Path::new(model).exists() {
        eprintln!("skipping: fixture {model} not present in this checkout");
        return;
    }
    // Also skip if the nsl binary hasn't been built.
    let bin = nsl_binary();
    if !bin.exists() {
        eprintln!("skipping: nsl binary not built at {}", bin.display());
        return;
    }

    // Create a minimal .bin calibration file.
    let mut data = std::env::temp_dir();
    data.push(format!(
        "nsl-no-cargo-{}-{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut blob = Vec::new();
    blob.extend_from_slice(&1u32.to_le_bytes());
    blob.extend_from_slice(&4u32.to_le_bytes());
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

    let out = Command::new(&bin)
        .args(&[
            "build",
            model,
            "--calibration-data",
            data.to_str().unwrap(),
            "--calibrate",
            "best-effort",
        ])
        .env("PATH", &stripped_path)
        .output()
        .expect("run nsl");

    let _ = std::fs::remove_file(&data);

    assert_eq!(
        out.status.code(),
        Some(0),
        "calibration should succeed without cargo/rustc on PATH.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
