//! The GPU half of `linear_3d_gate.rs`: `bias_add` on a 3-D activation
//! through the CUDA launcher (`gpu_bias_add`, which read dims 0 and 1 as
//! rows and columns) and the tape's BiasAdd backward on device gradients,
//! against the same PyTorch 2.15 literals. Strict f32 matmuls
//! (`NSL_MATMUL_TF32=0`), so the tolerance is f32's.
//!
//! Run (GPU required):
//!   cargo test -p nsl-cli --features cuda --test linear_3d_gpu_gate \
//!     -- --ignored --test-threads=1
#![cfg(feature = "cuda")]

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

const PROGRAM: &str = r#"from nsl.nn.layers import Linear

let b0 = zeros([2])
b0[0] = 0.5
b0[1] = -0.25
let x = (arange(0.0, 24.0).reshape([2, 3, 4]) * 0.1 - 1.0).to(cuda)
let w = (arange(0.0, 8.0).reshape([2, 4]) * 0.05 + 0.1).to(cuda)
let b = b0.to(cuda)
let r = (arange(0.0, 12.0).reshape([2, 3, 2]) * 0.1 - 0.5).to(cuda)
let lin = Linear(4, 2)
lin.w = w
lin.b = b
print(lin.forward(x).to(cpu))
let (l_w, d_w) = grad(w):
    sum(bias_add(x @ w.transpose(0, 1), b) * r)
let (l_b, d_b) = grad(b):
    sum(bias_add(x @ w.transpose(0, 1), b) * r)
let (l_x, d_x) = grad(x):
    sum(bias_add(x @ w.transpose(0, 1), b) * r)
print(l_w.to(cpu))
print(d_w.to(cpu))
print(d_b.to(cpu))
print(d_x.to(cpu))
"#;

#[rustfmt::skip]
const WANT: [f64; 1 + 12 + 8 + 2 + 24] = [
    -0.07, -1.5, 0.21, -0.9, 0.49, -0.3, 0.77, 0.3, 1.05, 0.9, 1.33, 1.5,
    3.08,
    1.4, 1.4, 1.4, 1.4, 1.4, 1.46, 1.52, 1.58,
    0.0, 0.6,
    -0.17, -0.215, -0.26, -0.305, -0.09, -0.115, -0.14, -0.165, -0.01, -0.015, -0.02, -0.025,
    0.07, 0.085, 0.1, 0.115, 0.15, 0.185, 0.22, 0.255, 0.23, 0.285, 0.34, 0.395,
];

#[test]
#[ignore = "requires CUDA GPU"]
fn a_3d_linear_on_the_gpu_matches_pytorch() {
    let dir = tempfile::tempdir().unwrap();
    let prog = dir.path().join("p.nsl");
    std::fs::write(&prog, PROGRAM).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--tape-ad"])
        .arg(&prog)
        .current_dir(dir.path())
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .env("NSL_MATMUL_TF32", "0")
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stdout}\n{stderr}");
    let got: Vec<f64> = stdout
        .lines()
        .filter(|l| l.starts_with("tensor(") || l.trim().parse::<f64>().is_ok())
        .flat_map(|l| {
            l.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
                .filter_map(|t| t.parse::<f64>().ok())
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(got.len(), WANT.len(), "printed:\n{stdout}");
    for (i, (g, w)) in got.iter().zip(WANT).enumerate() {
        assert!((g - w).abs() <= 2e-6 * w.abs().max(1.0), "value {i} is {g}, PyTorch {w}\n{stdout}");
    }
}
