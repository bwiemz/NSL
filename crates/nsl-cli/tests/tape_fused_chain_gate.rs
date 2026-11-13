//! Tape AD must differentiate through a fused elementwise chain.
//!
//! The compiler fuses a chain of elementwise ops (`tanh(sigmoid(u))`,
//! `neg(log(u))`, `relu(a + b)`) into one runtime call that makes a single
//! pass over the data. That call records nothing on the autodiff tape. The
//! compiler knows this and does not fuse inside the body of a `grad` or
//! `train` block -- but the tape is global runtime state, live in every
//! function and model method the block calls, and those are compiled on
//! their own. So a chain inside a model's `forward` was fused, and under
//! tape AD (the default for `nsl run`) the gradient of everything behind it
//! was exactly zero while the loss was correct. Nothing was printed unless
//! EVERY parameter was cut off.
//!
//! Each case below is checked two ways: against the value worked out by
//! hand, and against the same program run with `--disable-fusion`.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Runs `source` on the CPU with tape AD and returns every number it
/// prints, in order (tensor literals are flattened).
fn printed_numbers(source: &str, extra_args: &[&str]) -> Vec<f64> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .arg("--target")
        .arg("cpu")
        .args(extra_args)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn assert_all_close(got: &[f64], want: f64, what: &str) {
    assert!(!got.is_empty(), "{what}: nothing printed");
    for (i, &g) in got.iter().enumerate() {
        assert!(
            (g - want).abs() < 1e-5,
            "{what}: element {i} is {g}, expected {want} (all: {got:?})"
        );
    }
}

/// A chain in a free function: `-log(0.1 * sum(z))`, with `z` four elements
/// of 1.5, so `u = 0.6` and `d/dz = -0.1 / 0.6`.
#[test]
fn a_chain_in_a_function_called_from_a_grad_block() {
    let source = r#"
let zw = full([2, 2], 1.5)
fn c(z: Tensor) -> Tensor:
    let u = 0.1 * sum(z)
    return neg(log(u))
let (l, g) = grad(zw):
    c(zw)
print(g)
"#;
    let want = -0.1 / 0.6;
    assert_all_close(&printed_numbers(source, &[]), want, "fused");
    assert_all_close(&printed_numbers(source, &["--disable-fusion"]), want, "unfused");
}

/// A chain in a model method. With `u = x * w = 1.5`,
/// `d/dw sum(tanh(sigmoid(u))) = (1 - tanh(s)^2) * s * (1 - s)` at
/// `s = sigmoid(1.5)`.
#[test]
fn a_chain_in_a_model_method_called_from_a_grad_block() {
    let source = r#"
model M:
    w: Tensor = full([2, 2], 1.5)

    fn forward(self, x: Tensor) -> Tensor:
        let u = x * self.w
        return sum(tanh(sigmoid(u)))

let m = M()
let x = ones([2, 2])
let (l, g) = grad(m.w):
    m.forward(x)
print(g)
"#;
    let s = 1.0 / (1.0 + (-1.5_f64).exp());
    let want = (1.0 - s.tanh().powi(2)) * s * (1.0 - s);
    assert_all_close(&printed_numbers(source, &[]), want, "fused");
    assert_all_close(&printed_numbers(source, &["--disable-fusion"]), want, "unfused");
}

/// A two-input chain, `relu(a + b)`, the shape of every bias-then-activate
/// layer. `d/da sum(relu(a + b))` is 1 where `a + b > 0`; here everywhere.
#[test]
fn a_binary_then_unary_chain_in_a_function() {
    let source = r#"
let a = full([2, 2], 0.5)
let b = full([2, 2], 0.25)
fn layer(p: Tensor, q: Tensor) -> Tensor:
    return sum(relu(p + q))
let (l, g) = grad(a):
    layer(a, b)
print(g)
"#;
    assert_all_close(&printed_numbers(source, &[]), 1.0, "fused");
    assert_all_close(&printed_numbers(source, &["--disable-fusion"]), 1.0, "unfused");
}

/// The same hole in a `train` block: a model whose forward holds a chain,
/// trained with SGD under tape AD, must end where the unfused build ends.
/// Both parameters sit behind the chain, so before the fix neither moved
/// (and the run stopped with the "gradient for NONE of the parameters"
/// abort); a model with one parameter in front of the chain and one behind
/// would have trained half of itself silently.
#[test]
fn a_model_with_a_chain_in_its_forward_trains_the_same_fused_and_unfused() {
    let source = r#"
from nsl.nn.losses import mse_loss

model Net:
    w: Tensor = full([2, 2], 0.5)
    v: Tensor = full([2, 2], 0.25)

    fn forward(self, x: Tensor) -> Tensor:
        let h = x @ self.w
        let a = tanh(sigmoid(h))
        return a @ self.v

let m = Net()
let x = ones([3, 2])
let y = zeros([3, 2])

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.5)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print(m.w)
print(m.v)
"#;
    let fused = printed_numbers(source, &[]);
    let unfused = printed_numbers(source, &["--disable-fusion"]);
    assert_eq!(fused.len(), 8, "expected two 2x2 tensors: {fused:?}");
    assert_eq!(fused.len(), unfused.len());
    for (i, (f, u)) in fused.iter().zip(&unfused).enumerate() {
        assert!((f - u).abs() < 1e-6, "element {i}: fused {f}, unfused {u}");
    }
    assert!(
        (fused[0] - 0.5).abs() > 1e-4,
        "w, which sits behind the chain, did not move: {fused:?}"
    );
}
