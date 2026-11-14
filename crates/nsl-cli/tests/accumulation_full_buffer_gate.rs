//! Gradient accumulation must mean the same thing on every optimizer path.
//!
//! `grad_accumulation = N` has two implementations. AdamW, Adam and plain
//! SGD take the "deferred" one, which accumulates the window-mean gradient
//! and steps once on it. Everything else -- Lion, SOAP, Muon, and any train
//! block carrying `@fase(mode = off | full_buffer)` or
//! `@fase(allow_v_approx = false)` -- takes the "full buffer" one, which
//! until 2026-11 computed a different optimizer in three ways:
//!
//! 1. it accumulated the SUM of the micro-batch gradients (its comment
//!    promised a division that did not exist);
//! 2. it counted micro-batches, not optimizer steps, in Adam's bias
//!    correction, so at the default `eps` the first updates were about a
//!    third too small;
//! 3. it clipped each micro-batch to `grad_clip` and then summed them, so
//!    the gradient that reached the optimizer could have N times the norm
//!    the clip is meant to bound.
//!
//! `@fase(allow_v_approx = false)` reads as "be exact" and selected that
//! path, while the default was the exact one.
//!
//! The reference here is an AdamW written out in this file in f64, on a
//! model small enough to differentiate by hand: `pred = x @ w` with
//! `x = s * ones([4, 2])`, `loss = mean(pred^2)`, so
//! `dL/dw_j = 2 s^2 (w_0 + w_1)`. `s` grows by 1.25 per micro-batch, which
//! makes sum, mean and per-micro-batch clipping give different answers, and
//! `eps = 1.0` makes the gradient's magnitude matter. The same numbers come
//! out of `torch.optim.AdamW` with `clip_grad_norm_`.

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

const LR: f64 = 0.001;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1.0;
const WEIGHT_DECAY: f64 = 0.1;
const WINDOW: usize = 4;
const MICRO_BATCHES: usize = 8;

/// Standard accumulation: mean gradient over the window, one global-norm
/// clip of that mean, AdamW with `t` counting optimizer steps.
fn reference(clip: Option<f64>) -> f64 {
    reference_run(MICRO_BATCHES, clip)
}

/// The same, for a run of `micro_batches`. When that is not a multiple of
/// the window, the run ends with one more step on the mean of the
/// micro-batches the last window holds: the gradients are summed and
/// divided by their own count, not by `WINDOW`.
fn reference_run(micro_batches: usize, clip: Option<f64>) -> f64 {
    let mut w = [1.0_f64; 2];
    let (mut m, mut v) = ([0.0_f64; 2], [0.0_f64; 2]);
    let mut acc = [0.0_f64; 2];
    let mut held = 0_usize;
    let mut scale = 1.0_f64;
    let mut t = 0_i32;
    for micro in 0..micro_batches {
        let g = 2.0 * scale * scale * (w[0] + w[1]);
        for a in &mut acc {
            *a += g;
        }
        held += 1;
        if held == WINDOW || micro + 1 == micro_batches {
            for a in &mut acc {
                *a /= held as f64;
            }
            held = 0;
            t += 1;
            if let Some(max_norm) = clip {
                let norm = (acc[0] * acc[0] + acc[1] * acc[1]).sqrt();
                let factor = (max_norm / (norm + 1e-6)).min(1.0);
                for a in &mut acc {
                    *a *= factor;
                }
            }
            for j in 0..2 {
                m[j] = BETA1 * m[j] + (1.0 - BETA1) * acc[j];
                v[j] = BETA2 * v[j] + (1.0 - BETA2) * acc[j] * acc[j];
                let m_hat = m[j] / (1.0 - BETA1.powi(t));
                let v_hat = v[j] / (1.0 - BETA2.powi(t));
                w[j] -= LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * w[j]);
                acc[j] = 0.0;
            }
        }
        scale *= 1.25;
    }
    w[0]
}

fn program(decorator: &str, train_args: &str) -> String {
    program_for(MICRO_BATCHES, decorator, train_args)
}

fn program_for(micro_batches: usize, decorator: &str, train_args: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()

let x = ones([4, 2])
let y = zeros([4, 1])

{decorator}
train(model = m, epochs = {micro_batches}, grad_accumulation = {WINDOW}{train_args}):
    optimizer: AdamW(lr = {LR}, beta1 = {BETA1}, beta2 = {BETA2}, eps = {EPS}, weight_decay = {WEIGHT_DECAY})
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            x = x * 1.25

print(m.w.sum())
"#
    )
}

/// Runs `source` on the CPU and returns the last number it prints (the sum
/// of the two weights, which stay equal by symmetry), halved.
fn trained_weight(source: &str, extra_args: &[&str]) -> f64 {
    trained_weight_with_env(source, extra_args, &[]).0
}

/// The same, with environment variables for the compile, and the run's
/// stderr.
fn trained_weight_with_env(source: &str, extra_args: &[&str], env: &[(&str, &str)]) -> (f64, String) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("accum.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        // The suite may itself be run with the switch set (that is how the
        // tests of the final step were shown able to fail).
        .env_remove("NSL_FASE_FLUSH_TAIL")
        .envs(env.iter().copied())
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
    let stdout = String::from_utf8_lossy(&out.stdout);
    let sum: f64 = stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().parse().ok())
        .unwrap_or_else(|| panic!("no number printed:\n{stdout}\n{stderr}"));
    (sum / 2.0, stderr.into_owned())
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() < 2e-7,
        "{what}: trained weight {got}, reference {want} (difference {:e})",
        got - want
    );
}

/// The three wrong variants differ from the reference by far more than the
/// tolerance, so the assertions below can tell them apart. Without this a
/// fixture change could quietly make every variant agree.
#[test]
fn the_fixture_separates_the_defects_it_guards() {
    let standard = reference(None);
    let clipped = reference(Some(0.5));
    // Values the old full-buffer path produced on this fixture.
    let old_sum_and_micro_t = 0.998_694_598_7;
    let old_per_micro_clip = 0.999_283_015_7;
    assert!((standard - old_sum_and_micro_t).abs() > 5e-4);
    assert!((clipped - old_per_micro_clip).abs() > 5e-6);
    assert!((standard - clipped).abs() > 1e-3);
}

#[test]
fn every_accumulation_path_matches_standard_adamw() {
    let want = reference(None);
    for (decorator, what) in [
        ("", "default (deferred)"),
        ("@fase(mode = deferred)", "@fase(mode = deferred)"),
        ("@fase(mode = off)", "@fase(mode = off)"),
        ("@fase(mode = full_buffer)", "@fase(mode = full_buffer)"),
        ("@fase(allow_v_approx = false)", "@fase(allow_v_approx = false)"),
    ] {
        let source = program(decorator, "");
        assert_close(trained_weight(&source, &[]), want, &format!("{what}, tape AD"));
        assert_close(
            trained_weight(&source, &["--source-ad"]),
            want,
            &format!("{what}, source AD"),
        );
    }
}

#[test]
fn the_clip_bounds_the_window_mean_on_every_path() {
    let want = reference(Some(0.5));
    for (decorator, what) in [
        ("", "default (deferred)"),
        ("@fase(mode = full_buffer)", "@fase(mode = full_buffer)"),
        ("@fase(mode = off)", "@fase(mode = off)"),
    ] {
        let source = program(decorator, ", grad_clip = 0.5");
        assert_close(trained_weight(&source, &[]), want, &format!("{what}, tape AD"));
        assert_close(
            trained_weight(&source, &["--source-ad"]),
            want,
            &format!("{what}, source AD"),
        );
    }
}

/// The f64 reference above against `torch.optim.AdamW` (torch 2.x, float64)
/// run on the same fixture: `backward()` per micro-batch, `w.grad /= held`
/// at each step, `clip_grad_norm_`, `step()`, with a final step on a
/// trailing partial window.
#[test]
fn the_reference_is_pytorch_adamw() {
    for (micro_batches, clip, torch) in [
        (8, None, 0.998_079_008_032_501_3),
        (8, Some(0.5), 0.999_277_628_394_509_6),
        (10, None, 0.997_189_820_979_887_6),
        (10, Some(0.5), 0.998_916_496_762_292_2),
        (3, None, 0.999_030_346_232_179_2),
        (11, None, 0.997_218_076_737_901_9),
    ] {
        let ours = reference_run(micro_batches, clip);
        assert!(
            (ours - torch).abs() < 1e-12,
            "{micro_batches} micro-batches, clip {clip:?}: reference {ours}, PyTorch {torch}"
        );
    }
}

/// A run of 10 micro-batches at a window of 4 used to end with the weights
/// of a run of 8: the last two gradients were computed, accumulated and
/// freed with the buffers. The fixture's gradient grows by 1.5625 per
/// micro-batch, so the third step is the largest of the three and its
/// absence is far outside the tolerance.
#[test]
fn the_fixture_separates_a_dropped_or_unscaled_trailing_window() {
    for clip in [None, Some(0.5)] {
        let flushed = reference_run(10, clip);
        let dropped = reference_run(8, clip);
        assert!(
            (flushed - dropped).abs() > 2e-4,
            "clip {clip:?}: flushed {flushed}, dropped {dropped}"
        );
    }
    // Dividing the two pending gradients by the window (4) instead of by
    // their count (2) halves the gradient of the last step. With eps = 1.0
    // that is a different update.
    let mut w = [1.0_f64; 2];
    let (mut m, mut v, mut acc) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
    let (mut scale, mut t) = (1.0_f64, 0_i32);
    for micro in 0..10 {
        let g = 2.0 * scale * scale * (w[0] + w[1]);
        for a in &mut acc {
            *a += g / WINDOW as f64;
        }
        if (micro + 1) % WINDOW == 0 || micro == 9 {
            t += 1;
            for j in 0..2 {
                m[j] = BETA1 * m[j] + (1.0 - BETA1) * acc[j];
                v[j] = BETA2 * v[j] + (1.0 - BETA2) * acc[j] * acc[j];
                let m_hat = m[j] / (1.0 - BETA1.powi(t));
                let v_hat = v[j] / (1.0 - BETA2.powi(t));
                w[j] -= LR * (m_hat / (v_hat.sqrt() + EPS) + WEIGHT_DECAY * w[j]);
                acc[j] = 0.0;
            }
        }
        scale *= 1.25;
    }
    let unscaled = w[0];
    assert!(
        (reference_run(10, None) - unscaled).abs() > 2e-5,
        "mean over the window size would not be told apart: {unscaled}"
    );
}

#[test]
fn a_trailing_partial_window_is_stepped_on_its_own_mean() {
    for (clip, train_args) in [(None, ""), (Some(0.5), ", grad_clip = 0.5")] {
        let want = reference_run(10, clip);
        for (decorator, what) in [
            ("", "default (deferred)"),
            ("@fase(mode = full_buffer)", "@fase(mode = full_buffer)"),
            ("@fase(mode = off)", "@fase(mode = off)"),
        ] {
            let source = program_for(10, decorator, train_args);
            for args in [&[][..], &["--source-ad"][..], &["--training-reference"][..]] {
                assert_close(
                    trained_weight(&source, args),
                    want,
                    &format!("10 micro-batches, {what}, clip {clip:?}, flags {args:?}"),
                );
            }
        }
    }
}

/// `NSL_FASE_FLUSH_TAIL=0` at compile time turns the final step off (the
/// layer-major schedule cannot take it, and a comparison of the two
/// schedules needs both to stop at the same place). The run then ends with
/// the weights of the two full windows, and says what it dropped and why.
/// Without the switch nothing is dropped and nothing is said.
#[test]
fn the_final_step_can_be_compiled_out_and_then_says_so() {
    let source = program_for(10, "", "");
    let (dropped, stderr) = trained_weight_with_env(&source, &[], &[("NSL_FASE_FLUSH_TAIL", "0")]);
    assert_close(dropped, reference_run(8, None), "10 micro-batches, final step off");
    assert!(
        stderr.contains("training ended 2 micro-batch(es) into a gradient-accumulation window of 4")
            && stderr.contains("NSL_FASE_FLUSH_TAIL=0")
            && stderr.contains("were not applied"),
        "the dropped micro-batches were not reported:\n{stderr}"
    );

    let (flushed, stderr) = trained_weight_with_env(&source, &[], &[]);
    assert_close(flushed, reference_run(10, None), "10 micro-batches");
    assert!(!stderr.contains("training ended"), "{stderr}");

    // A whole number of windows: nothing pending, so nothing to say even
    // with the switch set.
    let (whole, stderr) =
        trained_weight_with_env(&program_for(8, "", ""), &[], &[("NSL_FASE_FLUSH_TAIL", "0")]);
    assert_close(whole, reference_run(8, None), "8 micro-batches, final step off");
    assert!(!stderr.contains("training ended"), "{stderr}");
}

/// The same when the micro-batches come from a `DataLoader`: ten batches in
/// one epoch, so the loop that ends inside a window is the loader's. The
/// step ignores the batch's contents, so the reference is unchanged.
#[test]
fn a_trailing_partial_window_is_flushed_after_a_dataloader_epoch() {
    let source = format!(
        r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()

let x = ones([4, 2])
let y = zeros([4, 1])
let corpus = arange(80)
let loader = DataLoader(corpus, batch_size = 2, seq_len = 4, shuffle = false, drop_last = true)

train(model = m, epochs = 1, grad_accumulation = {WINDOW}):
    optimizer: AdamW(lr = {LR}, beta1 = {BETA1}, beta2 = {BETA2}, eps = {EPS}, weight_decay = {WEIGHT_DECAY})
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            x = x * 1.25

print(m.w.sum())
"#
    );
    let want = reference_run(10, None);
    assert_close(trained_weight(&source, &[]), want, "10 loader batches, tape AD");
    assert_close(trained_weight(&source, &["--source-ad"]), want, "10 loader batches, source AD");
}

/// One and three pending micro-batches, and a run shorter than one window
/// (whose only step is the flush, at `t = 1`).
#[test]
fn every_remainder_of_the_window_is_flushed() {
    for micro_batches in [3, 5, 7, 9, 11] {
        let want = reference_run(micro_batches, None);
        let source = program_for(micro_batches, "", "");
        assert_close(
            trained_weight(&source, &[]),
            want,
            &format!("{micro_batches} micro-batches, tape AD"),
        );
        assert_close(
            trained_weight(&source, &["--source-ad"]),
            want,
            &format!("{micro_batches} micro-batches, source AD"),
        );
    }
}

/// `no_decay` under `grad_clip`: the fused multi-tensor launch honoured it
/// and the per-parameter fallback did not, so `--training-reference` (which
/// forces the fallback) decayed a parameter the default run exempted. The
/// "reference" mode was the wrong one.
#[test]
fn no_decay_is_honoured_by_the_per_parameter_step_under_a_clip() {
    let source = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])
    g: Tensor = ones([1])

    fn forward(self, x: Tensor) -> Tensor:
        return (x @ self.w) * self.g

let m = Linear()

let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 8, grad_accumulation = 4, grad_clip = 0.5):
    optimizer: AdamW(lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1.0, weight_decay = 0.5, no_decay = ["vector"])
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print(m.g.sum())
"#;
    // `trained_weight` halves what is printed; undo that for the scalar.
    let exempt = |args: &[&str]| 2.0 * trained_weight(source, args);
    let default_run = exempt(&[]);
    let reference_run = exempt(&["--training-reference"]);
    // Two steps at lr 0.01 with decay 0.5 would take a further 1% off `g`;
    // the gradient part alone moves it by well under that.
    assert!(
        (default_run - reference_run).abs() < 1e-6,
        "no_decay = [\"vector\"]: the default run leaves g at {default_run}, \
         --training-reference at {reference_run}"
    );
    assert!(
        default_run > 0.99,
        "g = {default_run}: the rank-1 parameter was decayed despite no_decay"
    );
}
