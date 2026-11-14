//! What one step of the learning-rate schedule is, under accumulation.
//!
//! With `grad_accumulation = N` a train block advances its `scheduler:`
//! once per micro-batch, so `warmup_steps`, `total_steps`, `step_size` and
//! `t_max` are reached N times sooner than the same number of optimizer
//! steps. Nothing in the source says so, and a recipe copied from a
//! framework that counts optimizer steps warms up and decays N times too
//! fast without any message.
//!
//! The unit is now something a program can state:
//! `schedule_unit = "optimizer_step"` counts optimizer steps, and
//! `schedule_unit = "micro_batch"` is the unit a block has always had. A
//! block that combines a schedule with accumulation and states neither
//! keeps counting micro-batches, so no run changes, and gets a compile-time
//! warning.
//!
//! The reference is worked out by hand. `pred = x @ w` with
//! `x = ones([4, 2])`, `w = ones([2, 1])` and `loss = mean(pred^2)` gives
//! `dL/dw_j = 2 (w_0 + w_1) = 4 w` while the two weights are equal, which
//! plain SGD keeps them. Every micro-batch of a window sees the same
//! weights, so the window mean is `4 w` and one optimizer step is
//! `w <- w (1 - 4 lr)`. With `SGD(lr = 0.1)` and
//! `step_lr(step_size = 2, gamma = 0.5)` the rate at schedule step `s` is
//! `0.1 * 0.5^floor(s / 2)`.
//!
//! Eight micro-batches at a window of two are four optimizer steps, taken
//! on micro-batches 1, 3, 5 and 7:
//!
//! * micro-batch units: `s = 1, 3, 5, 7`, rates `0.1, 0.05, 0.025, 0.0125`,
//!   so `w = 0.6 * 0.8 * 0.9 * 0.95 = 0.4104`;
//! * optimizer-step units: `s = 0, 1, 2, 3`, rates `0.1, 0.1, 0.05, 0.05`,
//!   so `w = 0.6 * 0.6 * 0.8 * 0.8 = 0.2304`.

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

fn program(micro_batches: usize, train_args: &str, scheduler: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()

let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = {micro_batches}{train_args}):
    optimizer: SGD(lr = 0.1)
{scheduler}    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print(m.w.sum())
"#
    )
}

const STEP_LR: &str = "    scheduler: step_lr(step_size = 2, gamma = 0.5)\n";

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn nsl(subcommand: &[&str], source: &str) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sched.nsl");
    std::fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .args(subcommand)
        .arg(&path)
        .output()
        .unwrap();
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The trained weight: the program prints the sum of the two weights.
fn trained_weight(source: &str, extra: &[&str]) -> (f64, String) {
    let mut args = vec!["run", "--target", "cpu"];
    args.extend_from_slice(extra);
    let r = nsl(&args, source);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    let sum: f64 = r
        .stdout
        .lines()
        .rev()
        .find_map(|l| l.trim().parse().ok())
        .unwrap_or_else(|| panic!("no number printed:\n{}\n{}", r.stdout, r.stderr));
    (sum / 2.0, r.stderr)
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() < 2e-6,
        "{what}: trained weight {got}, worked out by hand {want}"
    );
}

const WARNING: &str = "the learning-rate schedule counts micro-batches, not optimizer steps";

#[test]
fn the_default_unit_is_micro_batches_and_says_so() {
    let source = program(8, ", grad_accumulation = 2", STEP_LR);
    for ad in [&[][..], &["--source-ad"][..]] {
        let (w, stderr) = trained_weight(&source, ad);
        assert_close(w, 0.6 * 0.8 * 0.9 * 0.95, &format!("default unit, {ad:?}"));
        assert!(
            stderr.contains(WARNING),
            "{ad:?}: no warning about the unit:\n{stderr}"
        );
    }
}

#[test]
fn micro_batch_units_can_be_stated_and_are_then_silent() {
    let source = program(
        8,
        ", grad_accumulation = 2, schedule_unit = \"micro_batch\"",
        STEP_LR,
    );
    let (w, stderr) = trained_weight(&source, &[]);
    assert_close(w, 0.6 * 0.8 * 0.9 * 0.95, "schedule_unit = \"micro_batch\"");
    assert!(!stderr.contains(WARNING), "{stderr}");
}

#[test]
fn optimizer_step_units_count_optimizer_steps() {
    let source = program(
        8,
        ", grad_accumulation = 2, schedule_unit = \"optimizer_step\"",
        STEP_LR,
    );
    for ad in [&[][..], &["--source-ad"][..], &["--training-reference"][..]] {
        let (w, stderr) = trained_weight(&source, ad);
        assert_close(
            w,
            0.6 * 0.6 * 0.8 * 0.8,
            &format!("optimizer-step unit, {ad:?}"),
        );
        assert!(!stderr.contains(WARNING), "{ad:?}: {stderr}");
    }
}

/// The two answers are far apart on this fixture, so the assertions above
/// cannot pass for the wrong unit.
#[test]
fn the_fixture_separates_the_two_units() {
    let micro: f64 = 0.6 * 0.8 * 0.9 * 0.95;
    let optimizer: f64 = 0.6 * 0.6 * 0.8 * 0.8;
    assert!((micro - 0.4104).abs() < 1e-12 && (optimizer - 0.2304).abs() < 1e-12);
    assert!((micro - optimizer).abs() > 0.1);
}

/// Without accumulation the two units are the same thing: both spellings
/// give the unaccumulated answer, eight steps at
/// `0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125`, and nothing warns.
#[test]
fn without_accumulation_the_unit_changes_nothing() {
    let want = 0.6 * 0.6 * 0.8 * 0.8 * 0.9 * 0.9 * 0.95 * 0.95;
    for unit in [
        "",
        ", schedule_unit = \"micro_batch\"",
        ", schedule_unit = \"optimizer_step\"",
    ] {
        let (w, stderr) = trained_weight(&program(8, unit, STEP_LR), &[]);
        assert_close(w, want, &format!("no accumulation{unit}"));
        assert!(!stderr.contains(WARNING), "{unit}: {stderr}");
    }
}

/// No schedule, or one that ignores the step: nothing to be ambiguous about.
#[test]
fn accumulation_without_a_stepped_schedule_does_not_warn() {
    for scheduler in ["", "    scheduler: constant_lr()\n"] {
        let r = nsl(
            &["check"],
            &program(8, ", grad_accumulation = 2", scheduler),
        );
        assert!(r.ok, "{}", r.stderr);
        assert!(!r.stderr.contains(WARNING), "{scheduler:?}: {}", r.stderr);
    }
}

#[test]
fn check_warns_and_still_succeeds() {
    let r = nsl(&["check"], &program(8, ", grad_accumulation = 2", STEP_LR));
    assert!(r.ok, "a warning must not fail the check:\n{}", r.stderr);
    assert!(r.stderr.contains(WARNING), "{}", r.stderr);
    assert!(
        r.stderr.contains("schedule_unit = \"optimizer_step\""),
        "{}",
        r.stderr
    );
}

#[test]
fn an_unknown_unit_is_refused() {
    for bad in ["\"steps\"", "optimizer_step", "1"] {
        let r = nsl(
            &["check"],
            &program(
                8,
                &format!(", grad_accumulation = 2, schedule_unit = {bad}"),
                STEP_LR,
            ),
        );
        assert!(!r.ok, "schedule_unit = {bad} was accepted");
        assert!(
            r.stderr
                .contains("train 'schedule_unit' must be \"micro_batch\" or \"optimizer_step\""),
            "{bad}: {}",
            r.stderr
        );
    }
}
