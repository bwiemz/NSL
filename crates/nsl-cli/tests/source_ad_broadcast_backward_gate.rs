//! Source-AD backward through broadcasts: each operand's gradient has the
//! operand's shape.
//!
//! The Add and Sub adjoints were `Identity` and `Negate`, and the Div adjoints
//! reduced nothing, so a broadcast operand received the gradient at the
//! OUTPUT's shape. A `[4]` bias added to `[3, 4]` rows got a `[3, 4]`
//! gradient (SGD's update then failed on the shape mismatch); a one-element
//! `sum(x @ v)` added to `x @ w` got the whole `[3, 4]` gradient, element for
//! element, so `v` trained as if it were `w`. That leak also hid a second
//! defect: a full `sum`'s backward never expanded its gradient, and only ever
//! received a full-shape one because the Add above it had not reduced.
//!
//! Each fixture trains with SGD (AdamW's scale invariance hides gradient
//! errors) in both AD modes, and each mode's parameters are held to an f64
//! reference trajectory computed here by central differences.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run(fixture: &str, source_ad: bool) -> String {
    let root = repo_root();
    let tmp = tempfile::TempDir::new().unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if source_ad {
        cmd.arg("--source-ad");
    }
    let out = cmd
        .arg(root.join("crates/nsl-cli/tests/fixtures").join(fixture))
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        out.status.success(),
        "{fixture} (source_ad={source_ad}) failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn parse_between(stdout: &str, begin: &str, end: &str) -> Vec<f64> {
    let after = stdout.split_once(begin).map(|(_, r)| r).unwrap_or("");
    let inner = after.split_once(end).map(|(l, _)| l).unwrap_or("");
    inner
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

// Every fixture's data: x is 3x4 and positive, y is 3x4, w is 4x4 (row-major).
fn x(r: usize, c: usize) -> f64 {
    (4 * r + c) as f64 * 0.1 + 0.1
}
fn y(r: usize, c: usize) -> f64 {
    (4 * r + c) as f64 * 0.2 - 1.0
}
fn w0() -> Vec<f64> {
    (0..16).map(|i| i as f64 * 0.05 - 0.3).collect()
}

/// `(x @ m)[r][j]` for the 3x4 input and a row-major 4x4 `m`.
fn xm(m: &[f64], r: usize, j: usize) -> f64 {
    (0..4).map(|k| x(r, k) * m[4 * k + j]).sum()
}

fn cells() -> impl Iterator<Item = (usize, usize)> {
    (0..3).flat_map(|r| (0..4).map(move |j| (r, j)))
}

/// `mse_loss(pred, y)` = mean of the squared error over the 12 cells.
fn mse(pred: impl Fn(usize, usize) -> f64) -> f64 {
    cells().map(|(r, j)| (pred(r, j) - y(r, j)).powi(2)).sum::<f64>() / 12.0
}

/// SGD on `loss` from `p`, with each gradient taken by central differences.
fn sgd_reference(loss: impl Fn(&[f64]) -> f64, mut p: Vec<f64>, lr: f64, steps: usize) -> Vec<f64> {
    for _ in 0..steps {
        let grad: Vec<f64> = (0..p.len())
            .map(|i| {
                let (mut up, mut down) = (p.clone(), p.clone());
                up[i] += 1e-6;
                down[i] -= 1e-6;
                (loss(&up) - loss(&down)) / 2e-6
            })
            .collect();
        for (pi, gi) in p.iter_mut().zip(&grad) {
            *pi -= lr * gi;
        }
    }
    p
}

fn assert_close(label: &str, got: &[f64], want: &[f64], moved_from: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: expected {} values, got {}", want.len(), got.len());
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let travel = want.iter().zip(moved_from).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max |Δ| from the f64 reference {worst:.2e}; the reference moves {travel:.3}");
    // 500x the per-element tolerance below.
    assert!(travel > 5e-3, "{label}: the reference barely moves ({travel}); the check would be vacuous");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 1e-5, "{label}: [{i}] = {a}, f64 reference {b} (|Δ|={:.2e})", (a - b).abs());
    }
}

/// Runs `fixture` in both AD modes and holds each named parameter (its
/// markers, its slice of the flat reference, its initial value) to `want`.
fn check_both_modes(fixture: &str, want: &[f64], params: &[(&str, std::ops::Range<usize>)], init: &[f64]) {
    for source_ad in [false, true] {
        let out = run(fixture, source_ad);
        let mode = if source_ad { "source AD" } else { "tape AD" };
        for (name, range) in params {
            let got = parse_between(&out, &format!("{name}_BEGIN"), &format!("{name}_END"));
            assert_close(&format!("{mode} {name}"), &got, &want[range.clone()], &init[range.clone()]);
        }
    }
}

/// `x @ w + b - c`, b and c `[4]`: the bias's gradient is the column sum of
/// the prediction's gradient, the subtrahend's its negation.
#[test]
fn a_broadcast_bias_and_subtrahend_train_like_their_gradients() {
    let init: Vec<f64> = w0()
        .into_iter()
        .chain((0..4).map(|i| i as f64 * 0.1))
        .chain(std::iter::repeat_n(0.2, 4))
        .collect();
    let loss = |p: &[f64]| mse(|r, j| xm(&p[..16], r, j) + p[16 + j] - p[20 + j]);
    let want = sgd_reference(loss, init.clone(), 0.1, 4);
    check_both_modes(
        "source_ad_broadcast_bias_sub.nsl",
        &want,
        &[("W", 0..16), ("B", 16..20), ("C", 20..24)],
        &init,
    );
}

/// `x @ w + sum(x @ v)`: the one-element sum takes the SUM of the
/// prediction's gradient, and its backward expands that to `[3, 4]` for the
/// matmul. Unreduced, v's rows varied by column like w's; the true gradient is
/// constant along each row.
#[test]
fn a_broadcast_sum_intermediate_trains_like_its_gradient() {
    let init: Vec<f64> = w0().into_iter().chain(std::iter::repeat_n(0.1, 16)).collect();
    let loss = |p: &[f64]| {
        let s: f64 = cells().map(|(r, j)| xm(&p[16..], r, j)).sum();
        mse(|r, j| xm(&p[..16], r, j) + s)
    };
    let want = sgd_reference(loss, init.clone(), 0.01, 3);
    check_both_modes("source_ad_broadcast_sum.nsl", &want, &[("W", 0..16), ("V", 16..32)], &init);
}

/// `(x @ w) / s + (mean(x @ v) + 2.0)`, s `[4]`: the denominator's gradient
/// is summed over the rows, and the `+ 2.0` adds a float literal that has no
/// shape to reduce to.
#[test]
fn a_broadcast_denominator_trains_like_its_gradient() {
    let init: Vec<f64> = w0()
        .into_iter()
        .chain((0..4).map(|i| i as f64 * 0.5 + 2.0))
        .chain(std::iter::repeat_n(0.1, 16))
        .collect();
    let loss = |p: &[f64]| {
        let m = cells().map(|(r, j)| xm(&p[20..], r, j)).sum::<f64>() / 12.0;
        mse(|r, j| xm(&p[..16], r, j) / p[16 + j] + m + 2.0)
    };
    let want = sgd_reference(loss, init.clone(), 0.1, 4);
    check_both_modes(
        "source_ad_broadcast_div.nsl",
        &want,
        &[("W", 0..16), ("S", 16..20), ("V", 20..36)],
        &init,
    );
}
