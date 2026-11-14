//! Tape AD through an in-place element write: `y[i, j] = v` and its
//! compound forms inside a `grad` or `train` block.
//!
//! The tape did not record `nsl_tensor_set`, so the gradient through a write
//! was that of the tensor BEFORE the write: after `y = w * 2; y[0, 0] = 0`
//! the overwritten element still sent 2 back to `w[0, 0]`. A write is now a
//! tape op (`TapeOp::IndexWrite`): the written tensor takes a new tape id,
//! and the write's backward passes its gradient on with the overwritten
//! elements multiplied by the derivative of the stored value with respect to
//! the old one (0 for `=`, 1 for `+=`, `v` for `*= v`, `1/v` for `/= v`).
//!
//! Every expected number is PyTorch 2.15 (CPU, float64) autograd on the same
//! program, written out next to each test. Where PyTorch refuses the program
//! (a write over a value saved for the backward) NSL refuses it too; the one
//! program PyTorch differentiates and NSL refuses (a view of the written
//! tensor used after the write) is pinned as a refusal below.

use std::path::PathBuf;
use std::process::{Command, Output};

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run_with(source: &str, extra: &[&str]) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu"])
        .args(extra)
        .output()
        .unwrap()
}

/// Runs `source` and returns every number it printed.
fn numbers(source: &str, extra: &[&str]) -> Vec<f64> {
    let out = run_with(source, extra);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: printed {got:?}, expected {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-9, "{what}: element {i} is {g}, expected {w} (all: {got:?})");
    }
}

/// Runs `source`, which must be refused at run time with `message`.
fn assert_refused(source: &str, message: &str) {
    let out = run_with(source, &[]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the program ran:\n{}", String::from_utf8_lossy(&out.stdout));
    assert!(stderr.contains(message), "expected {message:?} in:\n{stderr}");
}

/// PyTorch: `y = w * 2.0; y[0, 0] = 0.0; y.sum().backward()` with
/// `w = arange(1, 5).reshape(2, 2)`: loss 18, grad [[0, 2], [2, 2]].
/// Unrecorded, the grad was [[2, 2], [2, 2]].
const ASSIGN: &str = r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 2.0
    y[0, 0] = 0.0
    sum(y)
print(l)
print(g)
"#;

#[test]
fn a_written_element_passes_no_gradient_back() {
    assert_close(&numbers(ASSIGN, &[]), &[18.0, 0.0, 2.0, 2.0, 2.0], "y[0, 0] = 0");
}

/// Under `--source-ad` the block falls back to the tape (the extraction
/// does not differentiate writes), so the same numbers come out.
#[test]
fn the_source_ad_fallback_gets_the_same_gradient() {
    assert_close(&numbers(ASSIGN, &["--source-ad"]), &[18.0, 0.0, 2.0, 2.0, 2.0], "y[0, 0] = 0, --source-ad");
}

/// One index into a vector. PyTorch: `y = w * 2.0; y[0] = 0.0` with
/// `w = arange(1, 5)`: loss 18, grad [0, 2, 2, 2]. This form did not
/// compile at all (it was lowered as a list store: a Cranelift verifier
/// error for a float value).
#[test]
fn a_one_index_write_into_a_vector() {
    let got = numbers(
        r#"
let w = arange(1.0, 5.0)
let (l, g) = grad(w):
    let y = w * 2.0
    y[0] = 0.0
    sum(y)
print(l)
print(g)
"#,
        &[],
    );
    assert_close(&got, &[18.0, 0.0, 2.0, 2.0, 2.0], "y[0] = 0");
}

/// `+=` keeps the old element's gradient and `=` drops it. PyTorch:
/// `y = w * 2.0; y[0, 1] = 5.0; y[0, 0] += 3.0; (y * y).sum()`: loss 150,
/// grad [[20, 0], [24, 32]] (unrecorded: [[20, 20], [24, 32]]; recorded as a
/// plain store, `+=` would have given 0 at [0, 0]).
#[test]
fn plus_assign_keeps_the_gradient_and_assign_drops_it() {
    let got = numbers(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 2.0
    y[0, 1] = 5.0
    y[0, 0] += 3.0
    sum(y * y)
print(l)
print(g)
"#,
        &[],
    );
    assert_close(&got, &[150.0, 20.0, 0.0, 24.0, 32.0], "y[0, 1] = 5; y[0, 0] += 3");
}

/// `*=` and `/=` scale the old element's gradient. PyTorch:
/// `y = w * 2.0; y[1, 0] *= 3.0; y[0, 1] /= 4.0; (y * y).sum()`: loss 393,
/// grad [[8, 1], [216, 32]].
#[test]
fn times_and_divide_assign_scale_the_gradient() {
    let got = numbers(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 2.0
    y[1, 0] *= 3.0
    y[0, 1] /= 4.0
    sum(y * y)
print(l)
print(g)
"#,
        &[],
    );
    assert_close(&got, &[393.0, 8.0, 1.0, 216.0, 32.0], "y[1, 0] *= 3; y[0, 1] /= 4");
}

/// Two writes, the second into a product that read the first. PyTorch:
/// `y = w * 3.0; y[1, 1] = 1.0; z = y * w; z[0, 1] = 2.0; (z * z).sum()`:
/// loss 758, grad [[36, 0], [972, 8]].
#[test]
fn writes_into_two_tensors_of_one_chain() {
    let got = numbers(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 3.0
    y[1, 1] = 1.0
    let z = y * w
    z[0, 1] = 2.0
    sum(z * z)
print(l)
print(g)
"#,
        &[],
    );
    assert_close(&got, &[758.0, 36.0, 0.0, 972.0, 8.0], "two writes");
}

/// A write into the differentiated tensor itself, through a function. Its
/// gradient is taken at the value it entered the block with. PyTorch
/// refuses an in-place write into a leaf that requires grad; on a copy
/// (`w2 = w.clone(); w2[0, 0] = 10.0; (w2 * w2).sum()`) it gives loss 129,
/// grad [[0, 4], [6, 8]]. The write itself reaches the caller's `w`.
#[test]
fn a_written_parameter_is_differentiated_at_its_entry_value() {
    let got = numbers(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
fn f(x: Tensor) -> Tensor:
    x[0, 0] = 10.0
    return sum(x * x)
let (l, g) = grad(w):
    f(w)
print(l)
print(g)
print(w)
"#,
        &[],
    );
    assert_close(
        &got,
        &[129.0, 0.0, 4.0, 6.0, 8.0, 10.0, 2.0, 3.0, 4.0],
        "loss, grad, then the caller's w",
    );
}

/// Writes into a tensor the tape has not seen (a mask built in a loop) are
/// not versioned and change nothing. PyTorch: `m = zeros(2, 2)`,
/// `m[i, i] = 1`, `(w * m).sum()`: loss 5, grad [[1, 0], [0, 1]].
#[test]
fn writes_into_a_fresh_tensor() {
    let got = numbers(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let m = zeros([2, 2])
    for i in range(2):
        m[i, i] = 1.0
    sum(w * m)
print(l)
print(g)
"#,
        &[],
    );
    assert_close(&got, &[5.0, 1.0, 0.0, 0.0, 1.0], "mask");
}

/// A train step through a model method that writes into its output. PyTorch:
/// `y = w * 2.0; y[0, 0] = 0.0; y.sum().backward(); SGD(lr=0.5).step()`
/// leaves w = [[1, 1], [2, 3]] (unrecorded: [[0, 1], [2, 3]]).
#[test]
fn a_train_step_through_a_write_in_a_model_method() {
    let got = numbers(
        r#"
model M:
    w: Tensor = arange(1.0, 5.0).reshape([2, 2])
    fn forward(self, x: Tensor) -> Tensor:
        let y = self.w * x
        y[0, 0] = 0.0
        return y

let m = M()
let x = full([2, 2], 2.0)
train(model=m, epochs=1):
    optimizer: SGD(lr=0.5)
    step(batch):
        let loss = sum(m.forward(x))
print(m.w)
"#,
        &[],
    );
    assert_close(&got, &[1.0, 1.0, 2.0, 3.0], "w after one SGD step");
}

const SAVED_REFUSAL: &str = "saved for the backward pass";

/// PyTorch refuses both: "one of the variables needed for gradient
/// computation has been modified by an inplace operation". `z = y * y`
/// saved `y`, and `exp` saved its own output.
#[test]
fn a_write_over_a_value_saved_for_the_backward_is_refused() {
    assert_refused(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 2.0
    let z = y * y
    y[0, 0] = 0.0
    sum(z)
print(g)
"#,
        SAVED_REFUSAL,
    );
    assert_refused(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = exp(w)
    y[0, 0] = 0.0
    sum(y)
print(g)
"#,
        SAVED_REFUSAL,
    );
    // `+=` changes the stored value too.
    assert_refused(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 2.0
    let z = y * y
    y[0, 0] += 1.0
    sum(z)
print(g)
"#,
        SAVED_REFUSAL,
    );
}

/// A view of the written tensor, taken before the write and used after it,
/// still names the values before the write in the tape. PyTorch rebases the
/// view (grad [[0, 2], [2, 2]]); the tape has no view relation to rebase
/// along, so the program is refused rather than given [[2, 2], [2, 2]].
#[test]
fn a_view_used_after_a_write_into_its_base_is_refused() {
    assert_refused(
        r#"
let w = arange(1.0, 5.0).reshape([2, 2])
let (l, g) = grad(w):
    let y = w * 2.0
    let v = y.reshape([4])
    y[0, 0] = 0.0
    sum(v)
print(g)
"#,
        "shares its storage with one written in place",
    );
}
