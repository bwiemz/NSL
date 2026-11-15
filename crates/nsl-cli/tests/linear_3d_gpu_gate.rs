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

fn run(program: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let prog = dir.path().join("p.nsl");
    std::fs::write(&prog, program).unwrap();
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--tape-ad"])
        .arg(&prog)
        .current_dir(dir.path())
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .env("NSL_MATMUL_TF32", "0")
        .output()
        .expect("spawn nsl run")
}

fn printed_numbers(stdout: &str) -> Vec<f64> {
    stdout
        .lines()
        .filter(|l| l.starts_with("tensor(") || l.trim().parse::<f64>().is_ok())
        .flat_map(|l| {
            l.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
                .filter_map(|t| t.parse::<f64>().ok())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_3d_linear_on_the_gpu_matches_pytorch() {
    let out = run(PROGRAM);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stdout}\n{stderr}");
    let got = printed_numbers(&stdout);
    assert_eq!(got.len(), WANT.len(), "printed:\n{stdout}");
    for (i, (g, w)) in got.iter().zip(WANT).enumerate() {
        assert!((g - w).abs() <= 2e-6 * w.abs().max(1.0), "value {i} is {g}, PyTorch {w}\n{stdout}");
    }
}

/// A transposed (strided) input is read in its logical order. The launcher
/// handed the kernel the view's storage, which it indexes linearly, so the
/// rows came out permuted. `base` is `[2, 4, 3]` and the input its
/// `[2, 3, 4]` transpose, so element `[b, i, j]` is `base[b, j, i]` plus
/// `bias[j]`, computed here.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_transposed_input_on_the_gpu_is_read_in_logical_order() {
    let out = run(r#"let b0 = zeros([4])
b0[0] = 0.5
b0[1] = -0.25
b0[2] = 1.0
b0[3] = 2.0
let base = (arange(0.0, 24.0).reshape([2, 4, 3]) * 0.1 - 1.0).to(cuda)
let xt = base.transpose(1, 2)
print(bias_add(xt, b0.to(cuda)).to(cpu))
"#);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "run failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    let bias = [0.5, -0.25, 1.0, 2.0];
    let mut want = Vec::new();
    for b in 0..2 {
        for i in 0..3 {
            for (j, bj) in bias.iter().enumerate() {
                want.push((b * 12 + j * 3 + i) as f64 * 0.1 - 1.0 + bj);
            }
        }
    }
    let got = printed_numbers(&stdout);
    assert_eq!(got.len(), want.len(), "printed:\n{stdout}");
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!((g - w).abs() <= 1e-6, "element {k} is {g}, expected {w}\n{stdout}");
    }
}

/// A bias whose length is not the last dimension stops the program on the
/// GPU as on the CPU. The GPU arm was taken before the length check and the
/// kernel read `bias[i % cols]` past the short buffer.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_wrong_length_bias_on_the_gpu_is_refused() {
    let out = run("let x = ones([2, 3]).to(cuda)\nlet b = ones([4]).to(cuda)\nprint(bias_add(x, b).to(cpu))\n");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "ran:\n{}", String::from_utf8_lossy(&out.stdout));
    assert!(stderr.contains("bias_add shape mismatch"), "{stderr}");
}

/// A zero-element input gives a zero-element output. The launcher asked
/// for a zero-size grid, which fails the launch.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_zero_element_input_on_the_gpu_gives_an_empty_output() {
    let out = run("let x = zeros([0, 4]).to(cuda)\nlet b = ones([4]).to(cuda)\nlet y = bias_add(x, b)\nprint(y.to(cpu))\nprint(\"DONE\")\n");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "run failed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("tensor([])") && stdout.contains("DONE"), "{stdout}");
}
