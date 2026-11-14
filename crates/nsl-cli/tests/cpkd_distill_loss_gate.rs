//! Distillation on the CPU: what the loss is, where it runs, and how the
//! teacher is evaluated.
//!
//! Each test names one thing a `distill` block used to get wrong:
//!
//! * `fused_kl_ce(...)` could only be used where the fused kernel could
//!   fire. Without an enabled `@fused_kl_ce` decorator the build stopped
//!   ("distill step body could not be extracted"), under `--target cpu` the
//!   fused path was lowered anyway and allocated on the device, and
//!   `--training-reference` did not run the composite it said it ran. All
//!   three now compute the standard library's composite `fused_kl_ce`.
//! * The teacher had no evaluation mode: `dropout` in its forward fired on
//!   every step, so the target the student is fitted to changed from step
//!   to step, and a teacher built on the stdlib `Dropout` module did not
//!   compile.
//! * The `loss:` section was parsed, range-checked, echoed in the build
//!   report and otherwise ignored.
//!
//! The numbers are literals from PyTorch 2.15: for the `fused_kl_ce` tests
//! see `support/kd_reference.rs`; the teacher tests use `F.mse_loss` on the
//! same `sin` / `cos` tensors under `torch.optim.SGD(lr=0.5)`. A loss
//! sequence that matches over four steps means the gradients match too.

use std::path::PathBuf;
use std::process::{Command, Output};

use assert_cmd::prelude::*;

#[allow(dead_code)]
#[path = "support/kd_reference.rs"]
mod kd;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn nsl(subcommand: &str, source: &str, extra_args: &[&str]) -> Output {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(&path, source).unwrap();
    Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg(subcommand)
        .arg(&path)
        .args(extra_args)
        .output()
        .unwrap()
}

use kd::numbers;

/// Runs `source` with `--source-ad` and returns (printed numbers, stderr).
fn run(source: &str, extra_args: &[&str]) -> (Vec<f64>, String) {
    let mut args = vec!["--source-ad"];
    args.extend_from_slice(extra_args);
    let out = nsl("run", source, &args);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "run failed:\n{stderr}");
    (numbers(&String::from_utf8_lossy(&out.stdout)), stderr)
}

/// The compile must fail, and say `needle`.
fn assert_refused(subcommand: &str, source: &str, extra_args: &[&str], needle: &str) {
    let out = nsl(subcommand, source, extra_args);
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert!(!out.status.success(), "expected a refusal containing {needle:?}, but it succeeded:\n{text}");
    assert!(text.contains(needle), "expected {needle:?} in:\n{text}");
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: printed {got:?}, expected {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * (1.0 + w.abs()), "{what}: element {i} is {g}, expected {w} (all: {got:?})");
    }
}

fn kd_program(decorator: &str, loss_section: &str, alpha: &str, temperature: &str) -> String {
    kd::program("", decorator, loss_section, kd::TARGETS, alpha, temperature)
}

const DECORATOR: &str = kd::DECORATOR;
const LOSS_SECTION: &str = kd::LOSS_SECTION;

fn assert_kd_numbers(got: &[f64], what: &str) {
    kd::assert_matches_pytorch(got, 2e-5, what);
}

/// No decorator at all: `fused_kl_ce` is the composite, differentiated at
/// compile time, and the build report says so.
#[test]
fn fused_kl_ce_without_a_decorator_runs_the_composite() {
    let (got, stderr) = run(&kd_program("", LOSS_SECTION, "0.6", "2.0"), &["--target", "cpu"]);
    assert_kd_numbers(&got, "no decorator");
    assert!(
        stderr.contains("Loss: fused_kl_ce(alpha=0.6, temperature=2), computed by the standard-library composite"),
        "the build report must name the loss that runs:\n{stderr}"
    );
    assert!(stderr.contains("Fused KL-CE: NOT active"), "{stderr}");
    assert!(!stderr.contains("Spectral"), "the report still points at the removed spectral analysis:\n{stderr}");
}

/// The composite is the standard library's code: a function the user's
/// module declares under a name it calls (`log_softmax` here) is not
/// substituted into it. It used to be, and the user's `x * 0.0` replaced
/// both log-softmaxes of the KL term.
#[test]
fn the_composite_does_not_call_the_users_functions() {
    let user = "fn log_softmax(x: Tensor, dim: int) -> Tensor:\n    return x * 0.0\n";
    let source = kd::program(user, "", LOSS_SECTION, kd::TARGETS, "0.6", "2.0");
    let (got, _) = run(&source, &["--target", "cpu"]);
    assert_kd_numbers(&got, "a user log_softmax in the program");
}

/// `--target cpu` with the decorator enabled: the fused kernel is CUDA-only,
/// so the composite runs on the host and the compile says it did. Nothing
/// is allocated on a device (this test runs on a build without CUDA, where
/// a device allocation is fatal).
#[test]
fn target_cpu_keeps_an_enabled_decorator_off_the_gpu() {
    let (got, stderr) = run(&kd_program(DECORATOR, LOSS_SECTION, "0.6", "2.0"), &["--target", "cpu"]);
    assert_kd_numbers(&got, "--target cpu");
    assert!(stderr.contains("the fused kernel is CUDA-only"), "the downgrade must be announced:\n{stderr}");
    assert!(stderr.contains("computed by the standard-library composite"), "{stderr}");
}

/// `--training-reference` selects the unfused baseline. It used to end in
/// "distill step body could not be extracted for source AD".
#[test]
fn training_reference_runs_the_composite_baseline() {
    let (got, stderr) = run(&kd_program(DECORATOR, LOSS_SECTION, "0.6", "2.0"), &["--training-reference"]);
    assert_kd_numbers(&got, "--training-reference");
    assert!(stderr.contains("computed by the standard-library composite"), "{stderr}");
}

fn teacher_program(teacher_body: &str, imports: &str, loss_fn: &str, loss_call: &str) -> String {
    format!(
        r#"{imports}
{loss_fn}
model Teacher:
    w: Tensor = tensor_cos(arange(0.0, 24.0)).reshape([6, 4]) * 0.5
{teacher_body}

model Student:
    w: Tensor = tensor_sin(arange(0.0, 24.0) * 0.3).reshape([6, 4]) * 0.5

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w.transpose(0, 1)

let teacher = Teacher()
let student = Student()
let x = tensor_sin(arange(0.0, 20.0) * 0.9).reshape([5, 4])

distill(teacher = teacher, student = student, epochs = 4):
    optimizer: SGD(lr = 0.5)
    step(batch):
        let t_logits = teacher.forward(x)
        let s_logits = student.forward(x)
        let loss = {loss_call}
    callbacks:
        on_step(step, loss):
            print(loss)
"#
    )
}

/// PyTorch: `F.mse_loss(x @ sw.T, x @ tw.T)` under SGD(lr=0.5), the teacher
/// in evaluation mode (no dropout).
const TEACHER_EVAL_LOSSES: [f64; 4] = [0.91949958, 0.62465149, 0.43220538, 0.3046121];

/// `dropout(..., 0.5, true)` in the teacher's forward. With the dropout
/// live the first loss alone would be off by a factor of order one (half
/// the teacher's logits zeroed, the rest doubled), and no two runs agree.
#[test]
fn dropout_in_the_teachers_forward_is_the_identity() {
    let teacher = "\n    fn forward(self, x: Tensor) -> Tensor:\n        return dropout(x @ self.w.transpose(0, 1), 0.5, true)";
    let source = teacher_program(teacher, "from nsl.nn.losses import mse_loss", "", "mse_loss(s_logits, t_logits)");
    let (got, _) = run(&source, &["--target", "cpu"]);
    assert_close(&got, &TEACHER_EVAL_LOSSES, 2e-5, "teacher with dropout(p=0.5, training=true)");
}

/// The same teacher built on the stdlib `Dropout` module, whose forward
/// asks `is_training()`. It did not compile inside a distill block.
#[test]
fn a_teacher_built_on_the_stdlib_dropout_module_is_evaluated() {
    let teacher = "    drop: Dropout = Dropout(0.5)\n\n    fn forward(self, x: Tensor) -> Tensor:\n        return self.drop.forward(x @ self.w.transpose(0, 1))";
    let source = teacher_program(
        teacher,
        "from nsl.nn.losses import mse_loss\nfrom nsl.nn.dropout import Dropout",
        "",
        "mse_loss(s_logits, t_logits)",
    );
    let (got, _) = run(&source, &["--target", "cpu"]);
    assert_close(&got, &TEACHER_EVAL_LOSSES, 2e-5, "teacher with the stdlib Dropout module");
}

/// A method called on a sub-model of the teacher straight from the step
/// body (`teacher.enc.forward(x)`) is the teacher's forward as well.
#[test]
fn a_method_called_on_a_teacher_sub_model_is_evaluated() {
    let source = r#"from nsl.nn.losses import mse_loss

model Enc:
    w: Tensor = tensor_cos(arange(0.0, 24.0)).reshape([6, 4]) * 0.5

    fn forward(self, x: Tensor) -> Tensor:
        return dropout(x @ self.w.transpose(0, 1), 0.5, true)

model Teacher:
    enc: Enc = Enc()

model Student:
    w: Tensor = tensor_sin(arange(0.0, 24.0) * 0.3).reshape([6, 4]) * 0.5

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w.transpose(0, 1)

let teacher = Teacher()
let student = Student()
let x = tensor_sin(arange(0.0, 20.0) * 0.9).reshape([5, 4])

distill(teacher = teacher, student = student, epochs = 4):
    optimizer: SGD(lr = 0.5)
    step(batch):
        let t_logits = teacher.enc.forward(x)
        let s_logits = student.forward(x)
        let loss = mse_loss(s_logits, t_logits)
    callbacks:
        on_step(step, loss):
            print(loss)
"#;
    let (got, _) = run(source, &["--target", "cpu"]);
    assert_close(&got, &TEACHER_EVAL_LOSSES, 2e-5, "dropout in a teacher sub-model");
}

/// A loss written as the user's own function. A distill block refuses the
/// tape, so before functions were inlined this was a hard error.
#[test]
fn a_user_defined_loss_function_works_in_a_distill_block() {
    let teacher = "\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w.transpose(0, 1)";
    let loss_fn = "fn my_mse(a: Tensor, b: Tensor) -> Tensor:\n    let d = a - b\n    return mean(d * d)\n";
    let source = teacher_program(teacher, "", loss_fn, "my_mse(s_logits, t_logits)");
    let (got, stderr) = run(&source, &["--target", "cpu"]);
    assert_close(&got, &TEACHER_EVAL_LOSSES, 2e-5, "user-defined loss");
    assert!(stderr.contains("Loss: the step body's own expression (no fused_kl_ce call)"), "{stderr}");
}

/// `feature_weight` and `feature_layers` configure a feature-matching loss
/// that does not exist.
#[test]
fn the_feature_matching_keys_are_refused() {
    for (key, value) in [("feature_weight", "0.7"), ("feature_layers", "[1, 3]"), ("feature_layers", "\"auto\"")] {
        let section = format!("    loss:\n        {key} = {value}\n");
        assert_refused(
            "check",
            &kd_program("", &section, "0.6", "2.0"),
            &[],
            &format!("distill loss key `{key}` is not supported"),
        );
    }
}

/// `alpha` / `temperature` in the `loss:` section with a step body that
/// never calls `fused_kl_ce`: nothing reads them.
#[test]
fn a_loss_section_nothing_consumes_is_refused() {
    let teacher = "\n    fn forward(self, x: Tensor) -> Tensor:\n        return x @ self.w.transpose(0, 1)";
    let source = teacher_program(teacher, "from nsl.nn.losses import mse_loss", "", "mse_loss(s_logits, t_logits)")
        .replace("    step(batch):", "    loss:\n        alpha = 0.5\n        temperature = 2.0\n    step(batch):");
    assert_refused(
        "run",
        &source,
        &["--source-ad", "--target", "cpu"],
        "`loss:` section declares alpha and temperature, but the step body does not call fused_kl_ce",
    );
}

/// The section and the call must state the same numbers.
#[test]
fn a_call_that_disagrees_with_the_loss_section_is_refused() {
    assert_refused(
        "run",
        &kd_program("", LOSS_SECTION, "0.3", "2.0"),
        &["--source-ad", "--target", "cpu"],
        "fused_kl_ce is called with alpha = 0.3, but the distill block's `loss:` section declares alpha = 0.6",
    );
    assert_refused(
        "run",
        &kd_program("", LOSS_SECTION, "0.6", "4.0"),
        &["--source-ad", "--target", "cpu"],
        "fused_kl_ce is called with temperature = 4, but the distill block's `loss:` section declares temperature = 2",
    );
}

/// Call-site literals are range-checked whether or not a `loss:` section
/// restates them: `nsl check` used to pass both of these.
#[test]
fn out_of_range_call_site_literals_fail_nsl_check() {
    assert_refused("check", &kd_program("", "", "1.7", "2.0"), &[], "fused_kl_ce: alpha = 1.7 is outside [0, 1]");
    assert_refused("check", &kd_program("", "", "0.6", "0.0"), &[], "fused_kl_ce: temperature = 0 is not positive");
    assert_refused("check", &kd_program("", "", "-0.1", "2.0"), &[], "fused_kl_ce: alpha = -0.1 is outside [0, 1]");
}
