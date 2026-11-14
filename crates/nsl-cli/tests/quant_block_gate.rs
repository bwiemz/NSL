//! `quant static q from m:` on models the block used to get wrong.
//!
//! Every expected number here is worked out by hand from the model's
//! weights and the size of one quantization step; none comes from running
//! NSL.
//!
//! - **A strided view.** A transposed field shares its parent's buffer; the
//!   quantizer read the buffer in storage order under the view's shape, so
//!   the quantized model computed with a different matrix.
//! - **A sub-model.** The block took the model type from a scan of a hash
//!   map, so with two model types it used one or the other from run to run,
//!   and it passed every field, sub-model pointers included, to the tensor
//!   quantizer: "invalid tensor handle" on some runs, the inner model's
//!   layout over the outer model's memory on the others.
//! - **`exclude:` in a sub-model**, and fields that are not tensors.

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

fn nsl(subcommand: &str, source: &str) -> std::process::Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg(subcommand)
        .arg(&path);
    if subcommand == "run" {
        cmd.arg("--target").arg("cpu");
    }
    cmd.output().unwrap()
}

/// The numbers a successful run printed, in order.
fn numbers(out: &std::process::Output) -> Vec<f64> {
    assert!(
        out.status.success(),
        "run failed: {}\n--- stdout\n{}\n--- stderr\n{}",
        out.status,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|tok| {
            let tok = tok.trim_matches(|c: char| matches!(c, 'e' | 'E' | '+'));
            if tok.chars().any(|c| c.is_ascii_digit()) { tok.parse::<f64>().ok() } else { None }
        })
        .collect()
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: printed {got:?}, expected {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}: element {i} is {g}, expected {w} (within {tol}); all: {got:?}");
    }
}

/// `w` is the transpose of [[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]]: a 3x2
/// view whose rows are (-2.5, 0.5), (-1.5, 1.5), (-0.5, 2.5). With x = ones,
/// x @ w is the column sums: [-4.5, 4.5].
///
/// At int8 over [-2.5, 2.5] one step is 5/255, so each of three summed
/// weights is off by at most half of that: |error| <= 3 * 2.5/255 < 0.03.
/// Reading the buffer in storage order gave [-1.49, 1.49].
#[test]
fn a_transposed_field_is_quantized_as_the_matrix_it_shows() {
    let source = r#"model V(dim: int):
    w: Tensor = (arange(0.0, 6.0) - 2.5).reshape([2, 3]).transpose(0, 1)

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = V(3)
let x = ones([1, 3])
let full: Tensor = m.forward(x)
print(full)
quant static q8 from m:
    default: int8, per_tensor
let quantized: Tensor = q8.forward(x)
print(quantized)
"#;
    let got = numbers(&nsl("run", source));
    assert_close(&got[..2], &[-4.5, 4.5], 1e-5, "full precision");
    assert_close(&got[2..], &[-4.5, 4.5], 0.03, "int8 quant block");
}

const NESTED: &str = r#"model Inner(dim: int):
    w: Tensor = arange(0.0, 16.0).reshape([4, 4]) * 0.1 - 0.75

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Outer(dim: int):
    emb: Tensor = arange(0.0, 32.0).reshape([8, 4]) * 0.01 - 0.155
    scale: int = 2
    inner: Inner = Inner(dim)

    fn forward(self, ids: Tensor) -> Tensor:
        let h = embedding_lookup(self.emb, ids)
        return self.inner.forward(h) * float(self.scale)

let m = Outer(4)
let ids = arange(1.0, 3.0)
"#;

/// The full-precision output of `NESTED`, by hand.
///
/// emb row r is 0.01 * (4r + c) - 0.155 for c = 0..3, so rows 1 and 2 are
/// (-0.115, -0.105, -0.095, -0.085) and (-0.075, -0.065, -0.055, -0.045).
/// w[k][j] = 0.1 * (4k + j) - 0.75. Output[i][j] = 2 * sum_k h[i][k] w[k][j].
fn nested_reference() -> Vec<f64> {
    let emb = |r: usize, c: usize| 0.01 * (4 * r + c) as f64 - 0.155;
    let w = |k: usize, j: usize| 0.1 * (4 * k + j) as f64 - 0.75;
    let mut out = Vec::new();
    for r in [1usize, 2] {
        for j in 0..4 {
            out.push(2.0 * (0..4).map(|k| emb(r, k) * w(k, j)).sum::<f64>());
        }
    }
    out
}

/// A model with a sub-model, an `int` field and an embedding goes through
/// an int8 block and comes out within the rounding error of the reference.
///
/// The bound: emb spans [-0.155, 0.155] (step 0.31/255) and w spans
/// [-0.75, 0.75] (step 1.5/255). With |h| <= 0.115 and |w| <= 0.75, each
/// output is 2 * a sum of four products, so its error is at most
/// 2 * 4 * (0.115 * 0.75/255 + 0.75 * 0.155/255 + small) < 0.007.
///
/// Run several times: the old lowering picked the model type from a hash
/// map's iteration order and failed on about half of the runs.
#[test]
fn a_model_with_a_sub_model_goes_through_a_quant_block() {
    let source = format!(
        "{NESTED}let full: Tensor = m.forward(ids)\nprint(full)\nquant static q from m:\n    default: int8, per_tensor\nlet quantized: Tensor = q.forward(ids)\nprint(quantized)\n"
    );
    let want = nested_reference();
    for attempt in 0..4 {
        let got = numbers(&nsl("run", &source));
        assert_eq!(got.len(), 16, "attempt {attempt}: {got:?}");
        assert_close(&got[..8], &want, 2e-5, "full precision");
        assert_close(&got[8..], &want, 0.007, "int8 quant block");
        // And the block did quantize: at int8 the sub-model's weight cannot
        // reproduce the reference to 1e-6 everywhere.
        assert!(
            got[8..].iter().zip(&want).any(|(g, w)| (g - w).abs() > 1e-6),
            "attempt {attempt}: the quantized output equals the full-precision one: {got:?}"
        );
    }
}

/// `exclude:` reaches a field of a sub-model by its dotted path, and a
/// pattern that names a top-level field still means that field. With both
/// tensors excluded the copy computes exactly what the source does; with
/// only the sub-model's weight quantized at int4 (step 1.5/15 = 0.1) it
/// cannot.
#[test]
fn exclude_patterns_reach_into_sub_models() {
    let both = format!(
        "{NESTED}quant static q from m:\n    default: int4, per_tensor\n    exclude: [\"emb\", \"inner.w\"]\nlet quantized: Tensor = q.forward(ids)\nprint(quantized)\n"
    );
    let want = nested_reference();
    assert_close(&numbers(&nsl("run", &both)), &want, 2e-5, "both tensors excluded");

    let only_emb = format!(
        "{NESTED}quant static q from m:\n    default: int4, per_tensor\n    exclude: [\"emb\"]\nlet quantized: Tensor = q.forward(ids)\nprint(quantized)\n"
    );
    let got = numbers(&nsl("run", &only_emb));
    // int4 on w: half a step is 0.05 on each of four weights, |h| <= 0.115,
    // doubled: |error| <= 2 * 4 * 0.115 * 0.05 < 0.05.
    assert_close(&got, &want, 0.05, "inner.w quantized at int4");
    assert!(
        got.iter().zip(&want).any(|(g, w)| (g - w).abs() > 1e-4),
        "inner.w was not quantized: {got:?}"
    );
}

/// The source model is left as it was: the block builds a copy.
#[test]
fn the_source_model_is_unchanged_by_the_block() {
    let source = format!(
        "{NESTED}quant static q from m:\n    default: int4, per_tensor\nlet after: Tensor = m.forward(ids)\nprint(after)\n"
    );
    assert_close(&numbers(&nsl("run", &source)), &nested_reference(), 2e-5, "source model after the block");
}

/// An array of sub-models: each element is copied and quantized.
///
/// Each block's w is 0.5 i^2 - 0.5 i + 0.5 for i = 0..3, i.e.
/// [[0.5, 0.5], [1.5, 3.5]]. With x = [1, 1]: x @ w = [2, 4], and
/// [2, 4] @ w = [1 + 6, 1 + 14] = [7, 15].
///
/// At int8 over [0, 3.5] half a step is 3.5/510 = 0.0069. The first
/// product sums two weights, so it is off by at most 0.014; the second is
/// off by at most (2 + 4) * 0.0069 + (0.5 + 3.5) * 0.014 = 0.1.
#[test]
fn a_fixed_array_of_sub_models_goes_through_a_quant_block() {
    let source = r#"model Block:
    w: Tensor = (arange(0.0, 4.0) * arange(0.0, 4.0) * 0.5 - arange(0.0, 4.0) * 0.5 - 0.5 + 1.0).reshape([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Net:
    blocks: [Block; 2] = Block()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
let x = ones([1, 2])
let full: Tensor = m.forward(x)
print(full)
quant static q from m:
    default: int8, per_tensor
let quantized: Tensor = q.forward(x)
print(quantized)
"#;
    let got = numbers(&nsl("run", source));
    assert_close(&got[..2], &[7.0, 15.0], 1e-4, "full precision");
    assert_close(&got[2..], &[7.0, 15.0], 0.1, "int8 quant block");
}

/// A per-channel axis that a field does not have is reported. It used to
/// index past the field's shape.
#[test]
fn an_axis_a_field_does_not_have_is_an_error_not_a_wild_read() {
    let source = r#"model M:
    w: Tensor = ones([2, 2])
    b: Tensor = ones([2])

let m = M()
quant static q from m:
    default: int8, per_channel(1)
print(1)
"#;
    let out = nsl("run", source);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the run should fail; stdout: {}", String::from_utf8_lossy(&out.stdout));
    assert!(stderr.contains("axis 1 is out of range for a 1-D tensor"), "{stderr}");
}
