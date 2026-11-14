//! Source AD: `mse_loss` and `l1_loss` when prediction and target broadcast.
//!
//! Both losses are a mean over the elements of the BROADCAST difference
//! `pred - target`. The compile-time adjoint took the element count from
//! `pred` alone and never summed the gradient back down to `pred`'s shape,
//! so with `pred: [3]` against `target: [2, 3]` the gradient came back with
//! the wrong shape and scaled by 1/3 instead of 1/6; `l1_loss` also walked
//! both operands with one index, reading past the end of the smaller one.
//!
//! Every expected value below is a literal produced by PyTorch 2.15 (CPU,
//! f32): `F.mse_loss` / `F.l1_loss` on the operands expanded to the common
//! shape, then `.backward()`. The operands are `sin` / `cos` of `arange`, so
//! the NSL program builds the same numbers.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// Runs `source` on the CPU with `--source-ad` and returns every number it
/// prints, in order (tensor literals are flattened), plus stderr.
fn printed_numbers(source: &str) -> (Vec<f64>, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu", "--source-ad"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed:\n{stderr}");
    let numbers = String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect();
    (numbers, stderr)
}

/// The program must have been differentiated at compile time: a fallback to
/// the tape would pass these checks without exercising the adjoint at all.
fn assert_source_ad_took_it(stderr: &str) {
    assert!(
        !stderr.contains("falling back to tape"),
        "the block fell back to tape AD, so the source-AD adjoint was not tested:\n{stderr}"
    );
}

fn assert_close(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: printed {got:?}, expected {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 2e-5, "{what}: element {i} is {g}, expected {w} (all: {got:?})");
    }
}

/// One `grad` block with respect to the tensor `w`; prints the loss, then
/// the gradient.
fn grad_program(w_init: &str, others: &str, loss: &str) -> String {
    format!(
        "from nsl.nn.losses import mse_loss, l1_loss\n\
         let w = {w_init}\n\
         {others}\n\
         let (l, g) = grad(w):\n\
         \x20   {loss}\n\
         print(l)\n\
         print(g)\n"
    )
}

/// The target is the smaller operand: `pred: [2, 3]`, `target: [3]`.
#[test]
fn prediction_larger_than_target() {
    let w = "tensor_sin(arange(0.0, 6.0)).reshape([2, 3])";
    let t = "let t = tensor_cos(arange(0.0, 3.0))";

    let (got, stderr) = printed_numbers(&grad_program(w, t, "mse_loss(w * 2.0, t)"));
    assert_source_ad_took_it(&stderr);
    assert_close(&got[..1], &[2.38141966], "mse loss");
    assert_close(
        &got[1..],
        &[-0.66666669, 0.76175976, 1.48982787, -0.47850665, -1.36927164, -1.00113451],
        "mse gradient",
    );

    let (got, stderr) = printed_numbers(&grad_program(w, t, "l1_loss(w * 2.0, t)"));
    assert_source_ad_took_it(&stderr);
    assert_close(&got[..1], &[1.44179165], "l1 loss");
    assert_close(
        &got[1..],
        &[-0.33333334, 0.33333334, 0.33333334, -0.33333334, -0.33333334, -0.33333334],
        "l1 gradient",
    );
}

/// The prediction is the smaller operand: `pred: [3]`, `target: [2, 3]`.
/// The gradient has `pred`'s three elements, each the sum over the two rows
/// it was broadcast to, and the mean is over six elements.
#[test]
fn prediction_smaller_than_target() {
    let w = "tensor_sin(arange(0.0, 3.0))";
    let t = "let t = tensor_cos(arange(0.0, 6.0)).reshape([2, 3])";

    let (got, stderr) = printed_numbers(&grad_program(w, t, "mse_loss(w * 2.0, t)"));
    assert_source_ad_took_it(&stderr);
    assert_close(&got[..1], &[2.68257165], "mse loss");
    assert_close(&got[1..], &[-0.00667167, 2.31948352, 2.51311636], "mse gradient");

    let (got, stderr) = printed_numbers(&grad_program(w, t, "l1_loss(w * 2.0, t)"));
    assert_source_ad_took_it(&stderr);
    assert_close(&got[..1], &[1.53981531], "l1 loss");
    assert_close(&got[1..], &[0.0, 0.66666669, 0.66666669], "l1 gradient");
}

/// Neither operand has the common shape: `pred: [2, 1]`, `target: [1, 3]`.
#[test]
fn both_operands_broadcast() {
    let w = "tensor_sin(arange(0.0, 2.0)).reshape([2, 1])";
    let t = "let t = tensor_cos(arange(0.0, 3.0)).reshape([1, 3])";

    let (got, stderr) = printed_numbers(&grad_program(w, t, "mse_loss(w * 2.0, t)"));
    assert_source_ad_took_it(&stderr);
    assert_close(&got[..1], &[1.27388549], "mse loss");
    assert_close(&got[1..], &[-0.74943697, 2.61644697], "mse gradient");

    let (got, stderr) = printed_numbers(&grad_program(w, t, "l1_loss(w * 2.0, t)"));
    assert_source_ad_took_it(&stderr);
    assert_close(&got[..1], &[0.98018652], "l1 loss");
    assert_close(&got[1..], &[-0.33333334, 1.0], "l1 gradient");
}

/// The differentiated operand is the TARGET, and it is the smaller one:
/// `pred: [2, 3]` (data), `target: [3]` (the differentiated tensor).
#[test]
fn gradient_with_respect_to_a_smaller_target() {
    let w = "tensor_cos(arange(0.0, 3.0))";
    let p = "let p = tensor_sin(arange(0.0, 6.0)).reshape([2, 3]) * 2.0";
    let (got, stderr) = printed_numbers(&grad_program(w, p, "mse_loss(p, w * 1.0)"));
    assert_source_ad_took_it(&stderr);
    assert_close(&got[..1], &[2.38141966], "mse loss");
    assert_close(&got[1..], &[0.57258666, 0.30375594, -0.24434668], "mse gradient");
}
