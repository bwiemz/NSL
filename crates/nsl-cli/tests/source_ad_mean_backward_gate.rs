//! Source-AD `mean` backward: the gradient of a mean carries its 1/N.
//!
//! The Mean adjoint rule in `ad_rules.rs` emitted a bare `Broadcast` of the
//! output gradient, commented as "for source AD analysis" with the 1/N left to
//! "the tape-based runtime backward". But source AD lowers these rules and
//! never runs the tape, so every `mean(...)` in a source-AD step produced a
//! gradient N times too large. Under AdamW, which is invariant to a uniform
//! scale, a mean-reduced loss trained almost the same; every gate that trains
//! with AdamW was blind to it. Under SGD the same program went uphill, and a
//! mean inside a forward scales only the paths through it, which changes the
//! gradient's direction.
//!
//! Each fixture trains with SGD in both AD modes, and each mode's weights are
//! held to an f64 reference trajectory computed here.

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

// The fixtures' data: x is 3x4, y is 3x4, w is 4x4 (row-major). Fixture 1's
// x is centred; fixture 2's is positive (`x_pos`), or v's gradient cancels.
fn x(r: usize, c: usize) -> f64 {
    (4 * r + c) as f64 * 0.1 - 0.5
}
fn x_pos(r: usize, c: usize) -> f64 {
    (4 * r + c) as f64 * 0.1 + 0.1
}
fn y(r: usize, c: usize) -> f64 {
    (4 * r + c) as f64 * 0.2 - 1.0
}
fn w0() -> Vec<f64> {
    (0..16).map(|i| i as f64 * 0.05 - 0.3).collect()
}

/// `(x @ w)[r][j]` for a 3x4 input and a row-major 4x4 `w`.
fn xw(x: fn(usize, usize) -> f64, w: &[f64], r: usize, j: usize) -> f64 {
    (0..4).map(|k| x(r, k) * w[4 * k + j]).sum()
}

const LR: f64 = 0.1;
const STEPS: usize = 4;
const N: f64 = 12.0; // elements in the 3x4 prediction

/// Fixture 1 in f64: `loss = mean((x @ w - y)^2)`, SGD. The gradient is the
/// closed form `2/N * x^T (x @ w - y)`.
fn reference_mean_loss() -> Vec<f64> {
    let mut w = w0();
    for _ in 0..STEPS {
        let mut g = vec![0.0; 16];
        for k in 0..4 {
            for j in 0..4 {
                g[4 * k + j] = (0..3).map(|r| x(r, k) * 2.0 * (xw(x, &w, r, j) - y(r, j)) / N).sum();
            }
        }
        for (wi, gi) in w.iter_mut().zip(&g) {
            *wi -= LR * gi;
        }
    }
    w
}

/// Fixture 2 in f64: `pred = x @ w + mean(x @ v)`, `loss = mean((pred - y)^2)`,
/// SGD, with gradients by central differences of that loss. Returns (w, v).
fn reference_mean_in_forward() -> (Vec<f64>, Vec<f64>) {
    let loss = |p: &[f64]| -> f64 {
        let (w, v) = (&p[..16], &p[16..]);
        let mean_xv: f64 =
            (0..3).flat_map(|r| (0..4).map(move |j| (r, j))).map(|(r, j)| xw(x_pos, v, r, j)).sum::<f64>() / N;
        (0..3)
            .flat_map(|r| (0..4).map(move |j| (r, j)))
            .map(|(r, j)| {
                let d = xw(x_pos, w, r, j) + mean_xv - y(r, j);
                d * d
            })
            .sum::<f64>()
            / N
    };
    let mut p: Vec<f64> = w0().into_iter().chain(std::iter::repeat_n(0.1, 16)).collect();
    for _ in 0..STEPS {
        let grad: Vec<f64> = (0..p.len())
            .map(|i| {
                let (mut up, mut down) = (p.clone(), p.clone());
                up[i] += 1e-6;
                down[i] -= 1e-6;
                (loss(&up) - loss(&down)) / 2e-6
            })
            .collect();
        for (pi, gi) in p.iter_mut().zip(&grad) {
            *pi -= LR * gi;
        }
    }
    (p[..16].to_vec(), p[16..].to_vec())
}

fn assert_close(label: &str, got: &[f64], want: &[f64], moved_from: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: expected {} values, got {}", want.len(), got.len());
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    let travel = want.iter().zip(moved_from).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max |Δ| from the f64 reference {worst:.2e}; the reference moves {travel:.3}");
    assert!(travel > 0.02, "{label}: the reference barely moves ({travel}); the check would be vacuous");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 1e-5, "{label}: [{i}] = {a}, f64 reference {b} (|Δ|={:.2e})", (a - b).abs());
    }
}

#[test]
fn a_mean_loss_trains_like_its_gradient_in_both_ad_modes() {
    let want = reference_mean_loss();
    for source_ad in [false, true] {
        let out = run("source_ad_mean_loss.nsl", source_ad);
        let label = if source_ad { "source AD" } else { "tape AD" };
        assert_close(label, &parse_between(&out, "W_BEGIN", "W_END"), &want, &w0());
    }
}

fn check_mean_inside_a_forward(source_ad: bool) {
    let (want_w, want_v) = reference_mean_in_forward();
    let out = run("source_ad_mean_in_forward.nsl", source_ad);
    let label = if source_ad { "source AD" } else { "tape AD" };
    assert_close(&format!("{label} w"), &parse_between(&out, "W_BEGIN", "W_END"), &want_w, &w0());
    assert_close(&format!("{label} v"), &parse_between(&out, "V_BEGIN", "V_END"), &want_v, &[0.1; 16]);
}

#[test]
fn a_mean_inside_a_forward_trains_like_its_gradient_under_tape_ad() {
    check_mean_inside_a_forward(false);
}

/// Source AD gets this program wrong for a second, separate reason: the Add
/// (and Sub) adjoint is `Identity`, with no reduction to the operand's shape,
/// so the [3, 4] gradient of `x @ w + mean(x @ v)` reaches the scalar
/// `mean(x @ v)` unreduced, and v trains like w (its rows vary by column; the
/// true gradient is constant along each row). Parameter gradients are reduced
/// to their parameter's shape at the end, which is why a bias add works; a
/// broadcast INTERMEDIATE is not. Fixing it touches every Add/Sub adjoint, so
/// it is its own change; this test is its reproduction.
#[test]
#[ignore = "known bug: the source-AD Add/Sub adjoint does not reduce a broadcast operand (see the doc comment)"]
fn a_mean_inside_a_forward_trains_like_its_gradient_under_source_ad() {
    check_mean_inside_a_forward(true);
}
