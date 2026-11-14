//! `gelu` is one function: the tanh approximation, on every device, in both
//! AD modes.
//!
//! `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
//!
//! The CPU computed this; the GPU computed the sigmoid approximation
//! `x * sigmoid(1.702 * x)` instead, so one model produced different values
//! depending on the device it ran on (up to about 2e-2 apart). The expected
//! values below are literals from PyTorch 2.15,
//! `F.gelu(x, approximate="tanh")` at `x = arange(9) * 0.75 - 3`, and the
//! gradient of `sum(gelu(x) * arange(1, 10))`.
//!
//! GPU: cargo test -p nsl-cli --features cuda --test gelu_definition_gate
//!      -- --ignored --test-threads=1

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const VALUES: [f64; 9] =
    [-0.0036374, -0.0272014, -0.1004284, -0.1700394, 0.0, 0.5799606, 1.3995715, 2.2227986, 2.9963627];
const GRADIENT: [f64; 9] =
    [-0.0115843, -0.1198292, -0.3831325, 0.0042471, 2.5, 5.9936295, 7.8939757, 8.4793167, 9.1042585];

/// The forward values and the gradient, flattened, as `device` and the AD
/// mode in `ad_flag` compute them.
fn gelu_on(device: &str, ad_flag: &str) -> Vec<f64> {
    let source = format!(
        "let x = (arange(0.0, 9.0) * 0.75 - 3.0).to({device})\n\
         let w = arange(1.0, 10.0).to({device})\n\
         let y: Tensor = gelu(x)\n\
         print(y.to(cpu))\n\
         let (loss, g) = grad(x):\n\
         \x20   sum(gelu(x) * w)\n\
         print(g.to(cpu))\n"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib")).current_dir(dir.path()).arg("run").arg(&path).arg(ad_flag);
    if device == "cpu" {
        cmd.args(["--target", "cpu"]);
    }
    let out = cmd.output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "[{device} {ad_flag}] run failed:\n{stderr}");
    if ad_flag == "--source-ad" {
        assert!(!stderr.contains("falling back to tape"), "[{device}] source AD fell back to the tape:\n{stderr}");
    }
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn assert_matches_pytorch(got: &[f64], what: &str) {
    assert_eq!(got.len(), 18, "{what}: nine values and nine gradient elements expected, got {got:?}");
    for (i, (g, w)) in got.iter().zip(VALUES.iter().chain(&GRADIENT)).enumerate() {
        assert!((g - w).abs() <= 2e-5 * (1.0 + w.abs()), "{what}: element {i} is {g}, expected {w}\n{got:?}");
    }
}

#[test]
fn cpu_gelu_is_the_tanh_approximation_in_both_ad_modes() {
    assert_matches_pytorch(&gelu_on("cpu", "--tape-ad"), "cpu, tape AD");
    assert_matches_pytorch(&gelu_on("cpu", "--source-ad"), "cpu, source AD");
}

/// The sigmoid form's value at x = -2.25 is -0.0478 against the tanh
/// form's -0.0272, hundreds of times the tolerance: this is the check that fails
/// on the old GPU kernel.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_gelu_is_the_same_function_in_both_ad_modes() {
    assert_matches_pytorch(&gelu_on("cuda", "--tape-ad"), "gpu, tape AD");
    assert_matches_pytorch(&gelu_on("cuda", "--source-ad"), "gpu, source AD");
}
