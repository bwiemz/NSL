//! `@freeze` freezes what it names, in both AD modes.
//!
//! The decorator was validated and handed to WRGA's analysis, and nothing in
//! the train block read it: a frozen weight stayed in the optimizer's
//! parameter list and trained exactly as if it were not frozen. So a LoRA
//! fine-tune that froze its base trained the base anyway.
//!
//! Each fixture trains with SGD and prints its random initial weights; the
//! frozen weights must come out of training unchanged, and every trained
//! tensor is held to an f64 reference trajectory computed here with the
//! frozen ones held fixed. A freeze reaches only the binding or model it
//! decorates. A pattern that matches no parameter is refused; a freeze that
//! leaves nothing to train runs, updates nothing, and warns.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn nsl_run(fixture: &Path, source_ad: bool) -> Output {
    let root = repo_root();
    let tmp = tempfile::TempDir::new().unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if source_ad {
        cmd.arg("--source-ad");
    }
    cmd.arg(fixture)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run")
}

fn run(fixture: &str, source_ad: bool) -> String {
    let path = repo_root().join("crates/nsl-cli/tests/fixtures").join(fixture);
    let out = nsl_run(&path, source_ad);
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

// The fixtures' data: x is 3x4 and positive, y is 3x4.
fn x_rows() -> Vec<f64> {
    (0..12).map(|i| i as f64 * 0.1 + 0.1).collect()
}
fn y(i: usize) -> f64 {
    i as f64 * 0.2 - 1.0
}

/// `h @ m` for a row-major 3xk `h` and kxn `m`.
fn matmul(h: &[f64], m: &[f64], k: usize, n: usize) -> Vec<f64> {
    (0..3 * n).map(|i| (0..k).map(|j| h[k * (i / n) + j] * m[n * j + i % n]).sum()).collect()
}

/// `mse_loss(pred, y)`: the mean squared error over the 12 cells.
fn mse(pred: &[f64]) -> f64 {
    (0..12).map(|i| (pred[i] - y(i)).powi(2)).sum::<f64>() / 12.0
}

/// SGD on `loss` from `p` for the entries `trainable` selects, each gradient
/// by central differences; the other entries stay fixed.
fn sgd_reference(loss: impl Fn(&[f64]) -> f64, mut p: Vec<f64>, trainable: impl Fn(usize) -> bool) -> Vec<f64> {
    for _ in 0..4 {
        let grad: Vec<f64> = (0..p.len())
            .map(|i| {
                if !trainable(i) {
                    return 0.0;
                }
                let (mut up, mut down) = (p.clone(), p.clone());
                up[i] += 1e-6;
                down[i] -= 1e-6;
                (loss(&up) - loss(&down)) / 2e-6
            })
            .collect();
        for (pi, gi) in p.iter_mut().zip(&grad) {
            *pi -= 0.1 * gi;
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

/// `(x @ w1) @ w2` with w1 frozen, by `include` or by `exclude`.
fn check_w1_frozen(fixture: &str) {
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run(fixture, source_ad);
        let w1_0 = parse_between(&out, "W1_0_BEGIN", "W1_0_END");
        let w2_0 = parse_between(&out, "W2_0_BEGIN", "W2_0_END");
        assert_eq!((w1_0.len(), w2_0.len()), (16, 16), "{mode}: the initial weights should print as 4x4");
        let init: Vec<f64> = w1_0.iter().chain(&w2_0).copied().collect();
        let loss = |p: &[f64]| mse(&matmul(&matmul(&x_rows(), &p[..16], 4, 4), &p[16..], 4, 4));
        let want = sgd_reference(loss, init.clone(), |i| i >= 16);

        let w1 = parse_between(&out, "W1_BEGIN", "W1_END");
        assert_eq!(w1, w1_0, "{mode}: the frozen w1 must come out of training unchanged");
        let w2 = parse_between(&out, "W2_BEGIN", "W2_END");
        assert_close(&format!("{mode} w2"), &w2, &want[16..], &w2_0);
    }
}

#[test]
fn freeze_include_freezes_the_weights_it_names() {
    check_w1_frozen("freeze_include.nsl");
}

#[test]
fn freeze_exclude_freezes_every_weight_it_does_not_name() {
    check_w1_frozen("freeze_exclude.nsl");
}

/// `w2` frozen DOWNSTREAM of the trained `w1`: dropping w2's own gradient
/// must not drop the gradient that flows through it (`dy @ w2^T`) to w1.
#[test]
fn a_frozen_weight_downstream_still_passes_the_gradient_through() {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/freeze_include.nsl")).unwrap();
    assert!(src.contains("@freeze(include=[\"w1\"])"));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("freeze_downstream.nsl");
    std::fs::write(&path, src.replace("@freeze(include=[\"w1\"])", "@freeze(include=[\"m.w2\"])")).unwrap();
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = nsl_run(&path, source_ad);
        assert!(out.status.success(), "{mode} failed:\n{}", String::from_utf8_lossy(&out.stderr));
        let out = String::from_utf8_lossy(&out.stdout).into_owned();
        let w1_0 = parse_between(&out, "W1_0_BEGIN", "W1_0_END");
        let w2_0 = parse_between(&out, "W2_0_BEGIN", "W2_0_END");
        let init: Vec<f64> = w1_0.iter().chain(&w2_0).copied().collect();
        let loss = |p: &[f64]| mse(&matmul(&matmul(&x_rows(), &p[..16], 4, 4), &p[16..], 4, 4));
        let want = sgd_reference(loss, init.clone(), |i| i < 16);

        let w2 = parse_between(&out, "W2_BEGIN", "W2_END");
        assert_eq!(w2, w2_0, "{mode}: the frozen w2 must come out of training unchanged");
        let w1 = parse_between(&out, "W1_BEGIN", "W1_END");
        assert_close(&format!("{mode} w1"), &w1, &want[..16], &w1_0);
    }
}

/// A freeze on another binding -- even one of the trained model's own type,
/// with the same field names -- leaves the trained `m` alone: both its
/// weights train, and a pattern matching nothing in `m` is not refused.
#[test]
fn a_freeze_on_another_binding_does_not_reach_the_trained_model() {
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run("freeze_other_bindings.nsl", source_ad);
        let w1_0 = parse_between(&out, "W1_0_BEGIN", "W1_0_END");
        let w2_0 = parse_between(&out, "W2_0_BEGIN", "W2_0_END");
        let init: Vec<f64> = w1_0.iter().chain(&w2_0).copied().collect();
        let loss = |p: &[f64]| mse(&matmul(&matmul(&x_rows(), &p[..16], 4, 4), &p[16..], 4, 4));
        let want = sgd_reference(loss, init.clone(), |_| true);
        let w1 = parse_between(&out, "W1_BEGIN", "W1_END");
        let w2 = parse_between(&out, "W2_BEGIN", "W2_END");
        assert_close(&format!("{mode} w1"), &w1, &want[..16], &w1_0);
        assert_close(&format!("{mode} w2"), &w2, &want[16..], &w2_0);
    }
}

/// A binding-scoped freeze reaches the model its own `let` bound: a second
/// `let net` of the same type rebinds the name, and both its weights train.
#[test]
fn a_freeze_does_not_reach_a_later_binding_of_the_same_name() {
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run("freeze_same_name_elsewhere.nsl", source_ad);
        let w1_0 = parse_between(&out, "W1_0_BEGIN", "W1_0_END");
        let w2_0 = parse_between(&out, "W2_0_BEGIN", "W2_0_END");
        let init: Vec<f64> = w1_0.iter().chain(&w2_0).copied().collect();
        let loss = |p: &[f64]| mse(&matmul(&matmul(&x_rows(), &p[..16], 4, 4), &p[16..], 4, 4));
        let want = sgd_reference(loss, init.clone(), |_| true);
        let w1 = parse_between(&out, "W1_BEGIN", "W1_END");
        let w2 = parse_between(&out, "W2_BEGIN", "W2_END");
        assert_close(&format!("{mode} w1"), &w1, &want[..16], &w1_0);
        assert_close(&format!("{mode} w2"), &w2, &want[16..], &w2_0);
    }
}

/// `@freeze` on `model Blk:` freezes both `Blk`s of `Net.blocks`; `Net`'s
/// own `head` trains against `((x @ w0) @ w1) @ head`.
#[test]
fn a_freeze_on_a_model_definition_freezes_every_instance_of_it() {
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run("freeze_model_definition.nsl", source_ad);
        let w0 = parse_between(&out, "W0_BEGIN", "W0_END");
        let head0 = parse_between(&out, "HEAD0_BEGIN", "HEAD0_END");
        assert_eq!((w0.len(), head0.len()), (32, 16), "{mode}: two block weights and the head should print");
        let init: Vec<f64> = w0.iter().chain(&head0).copied().collect();
        let loss = |p: &[f64]| {
            let h = matmul(&matmul(&x_rows(), &p[..16], 4, 4), &p[16..32], 4, 4);
            mse(&matmul(&h, &p[32..], 4, 4))
        };
        let want = sgd_reference(loss, init.clone(), |i| i >= 32);
        let w = parse_between(&out, "W_BEGIN", "W_END");
        assert_eq!(w, w0, "{mode}: both frozen Blk weights must come out of training unchanged");
        let head = parse_between(&out, "HEAD_BEGIN", "HEAD_END");
        assert_close(&format!("{mode} head"), &head, &want[32..], &head0);
    }
}

/// A bare `@freeze` with a LoRA adapter: the base `w` is frozen and only A
/// and B train, against `x @ w + ((x @ A) @ B) * 2` with `w` fixed.
#[test]
fn a_bare_freeze_trains_only_the_lora_adapter() {
    const RANK: usize = 2;
    let scale = 4.0 / RANK as f64;
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = run("freeze_lora_base.nsl", source_ad);
        let w0 = parse_between(&out, "W0_BEGIN", "W0_END");
        let a0 = parse_between(&out, "A0_BEGIN", "A0_END");
        assert_eq!((w0.len(), a0.len()), (16, 4 * RANK), "{mode}: the initial w and A should print");
        // w, A (4x2), B (2x4, starting at zero).
        let init: Vec<f64> = w0.iter().chain(&a0).copied().chain(std::iter::repeat_n(0.0, 8)).collect();
        let loss = |p: &[f64]| {
            let base = matmul(&x_rows(), &p[..16], 4, 4);
            let delta = matmul(&matmul(&x_rows(), &p[16..24], 4, RANK), &p[24..], RANK, 4);
            mse(&base.iter().zip(&delta).map(|(b, d)| b + d * scale).collect::<Vec<_>>())
        };
        let want = sgd_reference(loss, init.clone(), |i| i >= 16);

        let w = parse_between(&out, "W_BEGIN", "W_END");
        assert_eq!(w, w0, "{mode}: the frozen base w must come out of training unchanged");
        for (name, range) in [("A", 16..24), ("B", 24..32)] {
            let got = parse_between(&out, &format!("{name}_BEGIN"), &format!("{name}_END"));
            assert_close(&format!("{mode} {name}"), &got, &want[range.clone()], &init[range]);
        }
    }
}

/// Write `freeze_include.nsl` with its decorator replaced, run it, and
/// return the combined output of a run that must fail.
fn refused(decorator: &str) -> String {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/freeze_include.nsl")).unwrap();
    assert!(src.contains("@freeze(include=[\"w1\"])"));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("refused.nsl");
    std::fs::write(&path, src.replace("@freeze(include=[\"w1\"])", decorator)).unwrap();
    let out = nsl_run(&path, false);
    assert!(!out.status.success(), "`{decorator}` should be refused");
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// A pattern that matches nothing would freeze nothing; it is refused, and
/// the error lists what the patterns can name.
#[test]
fn a_freeze_pattern_that_matches_no_parameter_is_refused() {
    let err = refused("@freeze(include=[\"w3\"])");
    assert!(
        err.contains("@freeze pattern 'w3' matches no parameter") && err.contains("m.w1") && err.contains("m.w2"),
        "unexpected error:\n{err}"
    );
}

/// Freezing every parameter (with no adapter) is an explicit request: the
/// train block runs, updates nothing, and says so.
#[test]
fn a_freeze_that_leaves_nothing_to_train_runs_and_warns() {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures/freeze_include.nsl")).unwrap();
    assert!(src.contains("@freeze(include=[\"w1\"])"));
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("freeze_all.nsl");
    std::fs::write(&path, src.replace("@freeze(include=[\"w1\"])", "@freeze")).unwrap();
    for source_ad in [false, true] {
        let mode = if source_ad { "source AD" } else { "tape AD" };
        let out = nsl_run(&path, source_ad);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(out.status.success(), "{mode} failed:\n{stderr}");
        assert!(stderr.contains("no trainable parameter"), "{mode}: expected the warning:\n{stderr}");
        let stdout = String::from_utf8_lossy(&out.stdout);
        for name in ["W1", "W2"] {
            let before = parse_between(&stdout, &format!("{name}_0_BEGIN"), &format!("{name}_0_END"));
            let after = parse_between(&stdout, &format!("{name}_BEGIN"), &format!("{name}_END"));
            assert_eq!(before.len(), 16, "{mode}: {name} should print as 4x4");
            assert_eq!(after, before, "{mode}: a fully frozen {name} must come out of training unchanged");
        }
    }
}
