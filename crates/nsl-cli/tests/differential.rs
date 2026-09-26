//! Differential (oracle) testing: run the same .nsl script with and without
//! optimizations, assert numerical equivalence.
//!
//! This catches precision bugs in fusion passes, quantization, and kernel
//! specialization where code compiles and runs but produces wrong ML results.
//! Both runs are also held to an f64 reference, so the test fails on a wrong
//! answer even when no fusion fires and the two runs are the same program.

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

/// Run an NSL script from a scratch copy (`nsl run` leaves build products
/// beside the script) and return its stdout. A script that fails to compile
/// or run fails the test: a differential test that skips is a test that
/// passes without comparing anything, which is how every test in this file
/// used to "pass" (three named scripts that did not exist, and one that did
/// not compile).
fn nsl_run(script: &Path, extra_args: &[&str], tag: &str) -> String {
    let tmp = tempfile::tempdir().expect("tempdir");
    let copy = tmp.path().join(script.file_name().expect("script name"));
    std::fs::copy(script, &copy).unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    let output = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .arg(&copy)
        .args(extra_args)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .output()
        .expect("failed to invoke nsl");
    assert!(
        output.status.success(),
        "{} ({tag}) failed to run:\n{}",
        script.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).to_string()
}

/// Every number the script printed, in order (a tensor prints as one
/// `tensor([[...]])` line, so this does not go line by line).
fn parse_floats(output: &str) -> Vec<f64> {
    output
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter(|t| t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

/// Assert two float vectors agree to `epsilon`, relative to the larger of
/// the reference magnitude and 1 (so values near zero are held to an
/// absolute `epsilon`, not an unbounded relative one).
fn assert_close(got: &[f64], want: &[f64], epsilon: f64, context: &str) {
    assert_eq!(got.len(), want.len(), "{context}: length mismatch: got {}, want {}", got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let err = (g - w).abs() / w.abs().max(1.0);
        assert!(err < epsilon, "{context}: element {i}: got {g}, want {w} (error {err:.3e} >= {epsilon:.0e})");
    }
}

/// Run `script_name` with fusion on and with `--disable-fusion`; both must
/// print `reference` (an f64 evaluation of the same expression) to
/// `epsilon`, and so each other. The reference is what makes the test
/// mean something when no fusion fires: two identical runs would agree on
/// any wrong answer.
fn differential_test(script_name: &str, epsilon: f64, reference: &[f64]) {
    let script = workspace_root()
        .join("crates/nsl-cli/tests/differential_scripts")
        .join(script_name);
    assert!(script.exists(), "differential script {} is missing", script.display());

    let fused = parse_floats(&nsl_run(&script, &[], "fused"));
    let naive = parse_floats(&nsl_run(&script, &["--disable-fusion"], "naive"));
    assert!(!fused.is_empty(), "{script_name} printed no numbers");
    assert_close(&fused, reference, epsilon, &format!("{script_name} (fused) vs reference"));
    assert_close(&naive, reference, epsilon, &format!("{script_name} (naive) vs reference"));
    assert_close(&fused, &naive, epsilon, &format!("{script_name}: fused vs naive"));
}

/// `arange(n) * scale + offset`, reshaped row-major into `rows` rows.
fn grid(rows: usize, cols: usize, scale: f64, offset: f64) -> Vec<Vec<f64>> {
    (0..rows).map(|r| (0..cols).map(|c| (r * cols + c) as f64 * scale + offset).collect()).collect()
}

/// The runtime's CPU gelu: the tanh approximation.
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

// ---------------------------------------------------------------------------
// Differential tests
// ---------------------------------------------------------------------------

#[test]
fn differential_basic_matmul() {
    let (a, b) = (grid(2, 3, 0.5, -1.0), grid(3, 4, 0.25, 0.1));
    let want: Vec<f64> = (0..2)
        .flat_map(|i| {
            let (a, b) = (&a, &b);
            (0..4).map(move |j| (0..3).map(|k| a[i][k] * b[k][j]).sum::<f64>() * 2.0 + 0.5)
        })
        .collect();
    differential_test("diff_basic_matmul.nsl", 1e-6, &want);
}

#[test]
fn differential_fused_gelu() {
    let want: Vec<f64> =
        grid(4, 4, 0.35, -2.5).concat().into_iter().map(|x| gelu(x * 1.5 + 0.25) * 2.0 - x).collect();
    differential_test("diff_fused_gelu.nsl", 1e-5, &want);
}

#[test]
fn differential_softmax() {
    let want: Vec<f64> = grid(3, 4, 0.3, -1.5)
        .into_iter()
        .flat_map(|row| {
            let z: Vec<f64> = row.iter().map(|x| x * x + x * 0.5).collect();
            let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = z.iter().map(|v| (v - m).exp()).sum();
            z.into_iter().map(move |v| (v - m).exp() / sum)
        })
        .collect();
    differential_test("diff_softmax.nsl", 1e-5, &want);
}

#[test]
fn differential_layernorm() {
    let (w, b): (Vec<f64>, Vec<f64>) = ((0..4).map(|i| i as f64 * 0.5 + 0.75).collect(), (0..4).map(|i| i as f64 * -0.2).collect());
    let want: Vec<f64> = grid(3, 4, 0.4, -1.7)
        .into_iter()
        .flat_map(|row| {
            let z: Vec<f64> = row.iter().map(|x| x * x - x).collect();
            let mean = z.iter().sum::<f64>() / 4.0;
            let var = z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 4.0;
            let inv = 1.0 / (var + 1e-5).sqrt();
            let (w, b) = (w.clone(), b.clone());
            z.into_iter().enumerate().map(move |(j, v)| ((v - mean) * inv * w[j] + b[j]) * 3.0 + 0.1)
        })
        .collect();
    differential_test("diff_layernorm.nsl", 1e-5, &want);
}
