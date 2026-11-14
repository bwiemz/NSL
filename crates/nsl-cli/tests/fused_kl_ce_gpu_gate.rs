//! The fused KL-CE kernel, end to end on the GPU, against PyTorch; and what
//! it does with targets it cannot use.
//!
//! * The loss and the student's updates over four SGD steps match the
//!   PyTorch reference in `support/kd_reference.rs`, the same numbers the
//!   CPU composite is held to in `cpkd_distill_loss_gate.rs`.
//! * A negative target other than -100 is an ignored row, as it is for the
//!   composite. The kernel skips only -100 while the lowering's mask drops
//!   every negative target, so target -1 was a row the kernel computed
//!   (indexing class -1) and the mask then multiplied by zero: the loss
//!   printed `NaN` and the step still updated the student.
//! * A target at or above the vocabulary aborts, naming the row. It used to
//!   print `inf` and update.
//! * A wrong `batch_size` is reported against `@fused_kl_ce`, the decorator
//!   the program contains; the message used to name `@fused_lm_ce`.
//!
//! Run: cargo test -p nsl-cli --features cuda --test fused_kl_ce_gpu_gate
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::{Command, Output};

#[allow(dead_code)]
#[path = "support/kd_reference.rs"]
mod kd;

const GPU: &str = "teacher.to(cuda)\nstudent.to(cuda)\nlet targets = targets.to(cuda)";

fn run(source: &str) -> Output {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad"])
        .arg(&path)
        .current_dir(dir.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run")
}

fn text(out: &Output) -> (String, String) {
    (String::from_utf8_lossy(&out.stdout).into_owned(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// The run must have used the kernel: the build report says so.
fn assert_kernel_fired(stderr: &str) {
    assert!(
        stderr.contains("Fused KL-CE: teacher+student logits never materialized")
            && stderr.contains("computed by the fused kernel"),
        "the fused kernel did not fire:\n{stderr}"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_fused_kernel_matches_pytorch_over_four_steps() {
    let out = run(&kd::program(GPU, kd::DECORATOR, kd::LOSS_SECTION, kd::TARGETS, "0.6", "2.0"));
    let (stdout, stderr) = text(&out);
    assert!(out.status.success(), "run failed:\n{stderr}");
    assert_kernel_fired(&stderr);
    kd::assert_matches_pytorch(&kd::numbers(&stdout), 2e-5, "fused kernel");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_target_of_minus_one_is_an_ignored_row() {
    let out = run(&kd::program(GPU, kd::DECORATOR, kd::LOSS_SECTION, kd::TARGETS_MINUS_ONE, "0.6", "2.0"));
    let (stdout, stderr) = text(&out);
    assert!(out.status.success(), "run failed:\n{stderr}");
    assert_kernel_fired(&stderr);
    assert!(!stdout.to_lowercase().contains("nan"), "a NaN loss was printed:\n{stdout}");
    kd::assert_matches_pytorch(&kd::numbers(&stdout), 2e-5, "targets with -1");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_target_at_the_vocabulary_size_aborts_before_any_step() {
    let out = run(&kd::program(GPU, kd::DECORATOR, kd::LOSS_SECTION, kd::TARGETS_OUT_OF_RANGE, "0.6", "2.0"));
    let (stdout, stderr) = text(&out);
    assert!(!out.status.success(), "an out-of-range target must stop the run:\n{stdout}\n{stderr}");
    assert!(
        stderr.contains("target 128 at row 7 is out of range for the @fused_kl_ce vocab_size of 128"),
        "the abort must name the target, the row and the decorator:\n{stderr}"
    );
    assert!(kd::numbers(&stdout).is_empty(), "a loss was printed before the abort:\n{stdout}");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_wrong_row_count_is_reported_against_the_decorator_in_the_program() {
    let decorator = kd::DECORATOR.replace("batch_size = 2", "batch_size = 3");
    assert_ne!(decorator, kd::DECORATOR);
    let out = run(&kd::program(GPU, &decorator, kd::LOSS_SECTION, kd::TARGETS, "0.6", "2.0"));
    let (stdout, stderr) = text(&out);
    assert!(!out.status.success(), "a wrong batch_size must stop the run:\n{stdout}\n{stderr}");
    assert!(stderr.contains("the @fused_kl_ce decorator pins batch_size * seq_len = 12"), "{stderr}");
    assert!(!stderr.contains("@fused_lm_ce"), "the message names a decorator the program does not contain:\n{stderr}");
}
