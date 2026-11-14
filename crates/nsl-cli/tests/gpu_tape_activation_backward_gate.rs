//! Tape-AD activation backward on GPU: strided operands, and GELU's pairing.
//!
//! Two defects, both silent wrong gradients on the default (tape) AD path:
//!
//! * `gpu_backward_binary` handed `grad.data` and `saved.data` to a flat
//!   kernel without checking contiguity. The Transpose adjoint returns a
//!   strided view, so an activation whose output is transposed received its
//!   gradient in storage order: `relu(x)` followed by `.transpose(0, 1)`
//!   multiplied `g[i]` by `f'(x[j])` for the wrong `(i, j)`. The CPU arms
//!   wrap the same loop in `with_row_major`.
//! * The GPU GELU forward was `x * sigmoid(1.702 x)` while the tape backward
//!   launched the adjoint of the TANH approximation the CPU runs — the
//!   derivative of a different function, 22% off at x = -1. The forward is
//!   now the tanh approximation on both devices, so GELU is one function
//!   with one derivative; this gate holds the GPU gradient to PyTorch's
//!   `gelu(approximate="tanh")`.
//!
//! Each arm runs `d/dx sum(act(x).transpose(0, 1) * w)` on both devices.
//!
//! Run: cargo test -p nsl-cli --features cuda --test gpu_tape_activation_backward_gate
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// `d/dx sum(act(x).transpose(0, 1) * w)` with
/// x = [[-0.6, -0.3, 0.0], [0.3, 0.6, 0.9]] and w = [[1, 2], [3, 4], [5, 6]],
/// so dL/dy for y = act(x) is `w` transposed back: [[1, 3, 5], [2, 4, 6]].
fn gradient(act: &str, device: &str) -> Vec<f32> {
    let dir = std::env::temp_dir().join(format!("nsl_actbwd_{}_{act}_{device}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(
        &prog,
        format!(
            "let x = (arange(1.0, 7.0).reshape([2, 3]) * 0.3 - 0.9).to({device})\n\
             let w = arange(1.0, 7.0).reshape([3, 2]).to({device})\n\
             let (loss, grads) = grad(x):\n\
             \x20   let y = {act}(x)\n\
             \x20   let z = sum(y.transpose(0, 1) * w)\n\
             \x20   z\n\
             print(grads.to(cpu))\n"
        ),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--tape-ad"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("tensor("))
        .unwrap_or_else(|| panic!("[{act} {device}] no gradient printed:\n{stdout}\n{}", String::from_utf8_lossy(&out.stderr)));
    let values: Vec<f32> = line
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter_map(|t| t.parse().ok())
        .collect();
    assert_eq!(values.len(), 6, "[{act} {device}] six gradient elements: {line}");
    values
}

fn assert_close(what: &str, got: &[f32], want: &[f32]) {
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= 2e-5 * w.abs().max(1.0), "{what}: element {i} is {g}, expected {w}\n got {got:?}\nwant {want:?}");
    }
}

/// These four run the same formula on both devices, so the GPU gradient is
/// the CPU's. Read in storage order the relu arm came back as
/// `[[0, 0, 3], [4, 5, 6]]` where the gradient is `[[0, 0, 5], [2, 4, 6]]`.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_gradient_through_a_transposed_activation_matches_cpu() {
    for act in ["relu", "silu", "sigmoid", "tanh"] {
        let (cpu, gpu) = (gradient(act, "cpu"), gradient(act, "cuda"));
        assert_close(act, &gpu, &cpu);
    }
}

/// GELU is the tanh approximation on both devices. The expected gradient is
/// a literal from PyTorch 2.15: `F.gelu(x, approximate="tanh")` through the
/// same transpose and weights, `.backward()`.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_gelu_gradient_is_the_tanh_approximations() {
    let want = [0.074_508_5f32, 0.803_113_7, 2.5, 1.464_591, 3.701_966_3, 6.330_403_8];
    let (cpu, gpu) = (gradient("gelu", "cpu"), gradient("gelu", "cuda"));
    assert_close("gelu (cpu)", &cpu, &want);
    assert_close("gelu (gpu)", &gpu, &want);
    // The sigmoid form's derivative at x = -0.6 is 0.0660 against the tanh
    // form's 0.0745: the two are told apart.
    assert!((gpu[0] - 0.066_0).abs() > 5e-3, "gelu'(-0.6) = {} is the sigmoid form's", gpu[0]);
}
