//! LoRA adapters train, in both AD modes, on a flat model and on the layers
//! of a `[Blk; N]` array.
//!
//! `@adapter(type=lora, target=["Toy.w"])` rewrites `x @ self.w` into
//! `x @ w + ((x @ A) @ B) * (alpha / rank)`, with A and B in a side-table
//! hanging off the model instance. Three defects stood between that and a
//! LoRA run:
//! - the side-table was built only at source-AD train-block entry, so a
//!   tape-AD run read a null A and aborted on the first matmul;
//! - only the top-level instance got a table, and the source-AD load walk
//!   could not step through an array field, so adapters on `[Blk; 2]`
//!   layers resolved to a null placeholder and aborted;
//! - A and B were in neither AD mode's optimizer parameter list, so even a
//!   source-AD run that worked trained `w` alone and left B at zero.
//!
//! Each fixture trains with SGD and prints the adapters' random initial A;
//! each mode's w, A and B are held to an f64 reference trajectory computed
//! here from that A by central differences.

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

// The fixtures' data: x is 3x4 and positive, y is 3x4, each w is a 4x4 draw.
fn x(r: usize, c: usize) -> f64 {
    (4 * r + c) as f64 * 0.1 + 0.1
}
fn y(r: usize, c: usize) -> f64 {
    (4 * r + c) as f64 * 0.2 - 1.0
}

const RANK: usize = 2;
const LR: f64 = 0.1;
const STEPS: usize = 4;

/// One adapted layer: `h @ w + ((h @ a) @ b) * scale` for a 3x4 `h`, with
/// `w` 4x4, `a` 4x2 and `b` 2x4, all row-major.
fn adapted(h: &[f64], w: &[f64], a: &[f64], b: &[f64], scale: f64) -> Vec<f64> {
    let mut out = vec![0.0; 12];
    for r in 0..3 {
        let ha: Vec<f64> = (0..RANK).map(|q| (0..4).map(|k| h[4 * r + k] * a[RANK * k + q]).sum()).collect();
        for j in 0..4 {
            let base: f64 = (0..4).map(|k| h[4 * r + k] * w[4 * k + j]).sum();
            let delta: f64 = (0..RANK).map(|q| ha[q] * b[4 * q + j]).sum();
            out[4 * r + j] = base + delta * scale;
        }
    }
    out
}

/// `mse_loss(pred, y)`: the mean squared error over the 12 cells.
fn mse(pred: &[f64]) -> f64 {
    (0..12).map(|i| (pred[i] - y(i / 4, i % 4)).powi(2)).sum::<f64>() / 12.0
}

/// SGD on `loss` from `p`, each gradient by central differences.
fn sgd_reference(loss: impl Fn(&[f64]) -> f64, mut p: Vec<f64>) -> Vec<f64> {
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

/// The flat model's input, as a row-major 3x4.
fn x_rows() -> Vec<f64> {
    (0..12).map(|i| x(i / 4, i % 4)).collect()
}

#[test]
fn a_lora_adapter_trains_like_its_gradient_in_both_ad_modes() {
    // rank 2, alpha 4: the adapter's output is scaled by 2.
    let scale = 4.0 / RANK as f64;
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run("lora_adapter_flat.nsl", source_ad);
        let w0 = parse_between(&out, "W0_BEGIN", "W0_END");
        let a0 = parse_between(&out, "A0_BEGIN", "A0_END");
        assert_eq!(w0.len(), 16, "{mode}: the initial w should print as 4x4");
        assert_eq!(a0.len(), 4 * RANK, "{mode}: the initial A should print as 4x{RANK}");
        // w, then A (4x2), then B (2x4, starting at zero).
        let init: Vec<f64> = w0.into_iter().chain(a0).chain(std::iter::repeat_n(0.0, 8)).collect();
        let loss = |p: &[f64]| mse(&adapted(&x_rows(), &p[..16], &p[16..24], &p[24..], scale));
        let want = sgd_reference(loss, init.clone());
        let parts = [("W", 0..16), ("A", 16..24), ("B", 24..32)];
        for (name, range) in parts {
            let got = parse_between(&out, &format!("{name}_BEGIN"), &format!("{name}_END"));
            assert_close(&format!("{mode} {name}"), &got, &want[range.clone()], &init[range]);
        }
    }
}

#[test]
fn lora_adapters_on_an_array_of_layers_train_like_their_gradient_in_both_ad_modes() {
    let scale = 2.0 / RANK as f64;
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run("lora_adapter_blocks.nsl", source_ad);
        let w0 = parse_between(&out, "W0_BEGIN", "W0_END");
        let a0 = parse_between(&out, "A0_BEGIN", "A0_END");
        assert_eq!(w0.len(), 2 * 16, "{mode}: two initial w's should print, each 4x4");
        assert_eq!(a0.len(), 2 * 4 * RANK, "{mode}: two initial A's should print, each 4x{RANK}");
        assert_ne!(a0[..8], a0[8..], "{mode}: each block should draw its own A");
        // Per block i: w_i (16), A_i (8), B_i (8), at 32 * i.
        let mut init = Vec::new();
        for i in 0..2 {
            init.extend_from_slice(&w0[16 * i..16 * (i + 1)]);
            init.extend_from_slice(&a0[8 * i..8 * (i + 1)]);
            init.extend(std::iter::repeat_n(0.0, 8));
        }
        let loss = |p: &[f64]| {
            let h = adapted(&x_rows(), &p[..16], &p[16..24], &p[24..32], scale);
            mse(&adapted(&h, &p[32..48], &p[48..56], &p[56..64], scale))
        };
        let want = sgd_reference(loss, init.clone());
        // The fixture prints each name's two blocks back to back.
        for (name, offset, len) in [("W", 0, 16), ("A", 16, 8), ("B", 24, 8)] {
            let got = parse_between(&out, &format!("{name}_BEGIN"), &format!("{name}_END"));
            let pick = |v: &[f64]| -> Vec<f64> {
                (0..2).flat_map(|i| v[32 * i + offset..32 * i + offset + len].to_vec()).collect()
            };
            assert_close(&format!("{mode} {name}"), &got, &pick(&want), &pick(&init));
        }
    }
}
