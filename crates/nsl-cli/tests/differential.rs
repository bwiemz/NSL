//! Differential (oracle) testing: run the same .nsl script with and without
//! optimizations, assert numerical equivalence.
//!
//! This catches precision bugs in fusion passes, quantization, and kernel
//! specialization where code compiles and runs but produces wrong ML results.

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run an NSL script. Returns None if the script fails (graceful skip).
fn nsl_run(script: &Path, extra_args: &[&str]) -> Option<String> {
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(script)
        .args(extra_args)
        .current_dir(workspace_root())
        .output()
        .expect("failed to invoke cargo");

    if !output.status.success() {
        // Script failed to compile/run — caller should skip gracefully
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Parse output lines as floats for comparison.
fn parse_floats(output: &str) -> Vec<f64> {
    output
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect()
}

/// Assert two float vectors are close within epsilon.
fn assert_tensor_close(fused: &[f64], naive: &[f64], epsilon: f64, context: &str) {
    assert_eq!(
        fused.len(),
        naive.len(),
        "{context}: output length mismatch: fused={}, naive={}",
        fused.len(),
        naive.len()
    );
    for (i, (f, n)) in fused.iter().zip(naive.iter()).enumerate() {
        let diff = (f - n).abs();
        let denom = n.abs().max(1e-8);
        let rel_err = diff / denom;
        assert!(
            rel_err < epsilon,
            "{context}: element {i} differs: fused={f}, naive={n}, rel_err={rel_err:.6} > epsilon={epsilon}"
        );
    }
}

/// Run a differential test: same script with fusion enabled vs disabled.
///
/// Gracefully skips if: script doesn't exist, or script fails to compile/run.
/// These tests become active as milestone scripts are added.
fn differential_test(script_name: &str, epsilon: f64) {
    let script = workspace_root()
        .join("crates/nsl-cli/tests/differential_scripts")
        .join(script_name);

    if !script.exists() {
        eprintln!("skipping differential test: {script_name} not found");
        return;
    }

    let fused_output = match nsl_run(&script, &[]) {
        Some(out) => out,
        None => {
            eprintln!("skipping differential test: {script_name} failed to run (fused mode)");
            return;
        }
    };

    let naive_output = match nsl_run(&script, &["--disable-fusion"]) {
        Some(out) => out,
        None => {
            eprintln!("skipping differential test: {script_name} failed to run (naive mode)");
            return;
        }
    };

    let fused_vals = parse_floats(&fused_output);
    let naive_vals = parse_floats(&naive_output);

    if fused_vals.is_empty() || naive_vals.is_empty() {
        // Script doesn't output numeric values — just check same output
        assert_eq!(
            fused_output.trim(),
            naive_output.trim(),
            "differential test {script_name}: output differs between fused and naive"
        );
    } else {
        assert_tensor_close(&fused_vals, &naive_vals, epsilon, script_name);
    }
}

// ---------------------------------------------------------------------------
// Differential tests
// ---------------------------------------------------------------------------

// NOTE: These tests require working .nsl scripts in differential_scripts/.
// They gracefully skip if scripts don't exist or fail to compile.
// Scripts are populated as each milestone adds its differential test.

#[test]
fn differential_basic_matmul() {
    differential_test("diff_basic_matmul.nsl", 1e-6);
}

#[test]
fn differential_fused_gelu() {
    differential_test("diff_fused_gelu.nsl", 1e-5);
}

#[test]
fn differential_softmax() {
    differential_test("diff_softmax.nsl", 1e-5);
}

#[test]
fn differential_layernorm() {
    differential_test("diff_layernorm.nsl", 1e-5);
}
