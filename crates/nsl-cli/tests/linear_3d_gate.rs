//! `bias_add` (the stdlib `Linear`'s and `MLP`'s bias) on a
//! `[batch, seq, features]` activation, forward and both AD modes.
//!
//! `nsl_tensor_bias_add` accepted only a 2-D tensor, so `Linear.forward` on
//! a 3-D input aborted with "bias_add requires 2D tensor (got 3D)", and the
//! tape's backward read dims 0 and 1 as rows and columns. Source AD did not
//! extract `bias_add` and fell back to the tape. The bias now runs along the
//! last dimension and its gradient sums over every leading one; source AD
//! extracts it as the broadcasting Add.
//!
//! The expected values are PyTorch 2.15 (float64): `F.linear(x, w, b)` and
//! `(y * r).sum().backward()` with the same x, w, b, r.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const PROGRAM: &str = r#"from nsl.nn.layers import Linear

let x = arange(0.0, 24.0).reshape([2, 3, 4]) * 0.1 - 1.0
let w = arange(0.0, 8.0).reshape([2, 4]) * 0.05 + 0.1
let b = zeros([2])
b[0] = 0.5
b[1] = -0.25
let r = arange(0.0, 12.0).reshape([2, 3, 2]) * 0.1 - 0.5
let lin = Linear(4, 2)
lin.w = w
lin.b = b
print(lin.forward(x))
let (l_w, d_w) = grad(w):
    sum(bias_add(x @ w.transpose(0, 1), b) * r)
let (l_b, d_b) = grad(b):
    sum(bias_add(x @ w.transpose(0, 1), b) * r)
let (l_x, d_x) = grad(x):
    sum(bias_add(x @ w.transpose(0, 1), b) * r)
print(l_w)
print(d_w)
print(d_b)
print(d_x)
"#;

#[rustfmt::skip]
const WANT: [f64; 1 + 12 + 8 + 2 + 24] = [
    // F.linear(x, w, b)
    -0.07, -1.5, 0.21, -0.9, 0.49, -0.3, 0.77, 0.3, 1.05, 0.9, 1.33, 1.5,
    // loss
    3.08,
    // d/dw
    1.4, 1.4, 1.4, 1.4, 1.4, 1.46, 1.52, 1.58,
    // d/db
    0.0, 0.6,
    // d/dx
    -0.17, -0.215, -0.26, -0.305, -0.09, -0.115, -0.14, -0.165, -0.01, -0.015, -0.02, -0.025,
    0.07, 0.085, 0.1, 0.115, 0.15, 0.185, 0.22, 0.255, 0.23, 0.285, 0.34, 0.395,
];

fn run(extra: &[&str]) -> (Vec<f64>, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, PROGRAM).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu"])
        .args(extra)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed:\n{stderr}");
    let got = String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    (got, stderr)
}

fn assert_matches_pytorch(got: &[f64], what: &str) {
    assert_eq!(got.len(), WANT.len(), "{what}: printed {got:?}");
    // The program runs in f32: a relative 1e-6 is a few ulps.
    for (i, (g, w)) in got.iter().zip(WANT).enumerate() {
        assert!((g - w).abs() <= 1e-6 * w.abs().max(1.0), "{what}: value {i} is {g}, PyTorch {w}");
    }
}

#[test]
fn a_3d_linear_matches_pytorch_on_the_tape() {
    let (got, _) = run(&[]);
    assert_matches_pytorch(&got, "tape AD");
}

#[test]
fn a_3d_linear_matches_pytorch_under_source_ad() {
    let (got, stderr) = run(&["--source-ad"]);
    assert_eq!(stderr.matches("Using source-to-source AD for grad block").count(), 3, "{stderr}");
    assert!(!stderr.contains("falling back to tape"), "{stderr}");
    assert_matches_pytorch(&got, "source AD");
}
