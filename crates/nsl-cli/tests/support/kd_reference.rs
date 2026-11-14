//! A small, fully determined distillation problem and its PyTorch answers,
//! shared by the CPU gate (`cpkd_distill_loss_gate.rs`) and the GPU gate
//! (`fused_kl_ce_gpu_gate.rs`) so the composite and the fused kernel are
//! held to the same numbers.
//!
//! Eight rows, vocabulary 128, student hidden 32, teacher hidden 64 (sizes
//! the `@fused_kl_ce` kernel accepts). Rows 6 and 7 are ignored. Every
//! tensor is `sin` / `cos` of a short ramp, so the program builds exactly
//! what the reference did.
//!
//! The reference is PyTorch 2.15 in float64:
//! `alpha * F.cross_entropy(s, targets, ignore_index=-100)
//!  + (1 - alpha) * T^2 * mean_valid_rows(sum(p_t * (log p_t - log p_s)))`
//! with `alpha = 0.6`, `T = 2`, under `torch.optim.SGD(lr=0.5)` on the
//! student's `xs`, `ws`, `bs` for four steps. A loss sequence that matches
//! over four steps means the three student gradients match too.

/// The NSL program. `placement` is inserted after the model instances are
/// created (GPU moves), `decorator` on the distill block, `loss_section`
/// inside it; `targets` is the expression for the eight targets.
pub fn program(placement: &str, decorator: &str, loss_section: &str, targets: &str, alpha: &str, temperature: &str) -> String {
    format!(
        r#"from nsl.nn.losses import fused_kl_ce

model Teacher:
    xt: Tensor = tensor_sin(arange(0.0, 512.0) * 0.11).reshape([8, 64]) * 0.5
    wt: Tensor = tensor_cos(arange(0.0, 8192.0) * 0.007).reshape([128, 64]) * 0.3
    bt: Tensor = tensor_sin(arange(0.0, 128.0) * 0.3) * 0.1

model Student:
    xs: Tensor = tensor_cos(arange(0.0, 256.0) * 0.23).reshape([8, 32]) * 0.5
    ws: Tensor = tensor_sin(arange(0.0, 4096.0) * 0.013).reshape([128, 32]) * 0.3
    bs: Tensor = tensor_cos(arange(0.0, 128.0) * 0.3) * 0.1

let teacher = Teacher()
let student = Student()
let targets = {targets}
{placement}
{decorator}
distill(teacher = teacher, student = student, epochs = 4):
    optimizer: SGD(lr = 0.5)
{loss_section}    step(batch):
        let loss = fused_kl_ce(student.xs, student.ws, student.bs, teacher.xt, teacher.wt, teacher.bt, targets, {alpha}, {temperature})
    callbacks:
        on_step(step, loss):
            print(loss)
let ws_sum: Tensor = student.ws.sum()
let xs_sum: Tensor = student.xs.sum()
let bs_sum: Tensor = student.bs.sum()
print(ws_sum)
print(xs_sum)
print(bs_sum)
let wt_sum: Tensor = teacher.wt.sum()
print(wt_sum)
"#
    )
}

/// Targets 0, 17, 34, 51, 68, 85 and two rows at the ignore index.
pub const TARGETS: &str = "tensor_cat([arange(0.0, 6.0) * 17.0, full([2], -100.0)], 0)";
/// The same, with the two ignored rows marked -1 instead of -100. PyTorch
/// has one ignore index; NSL's `cross_entropy` ignores every negative
/// target, so this is the same problem.
pub const TARGETS_MINUS_ONE: &str = "tensor_cat([arange(0.0, 6.0) * 17.0, full([2], -1.0)], 0)";
/// Row 7 names class 128 of a 128-class vocabulary.
pub const TARGETS_OUT_OF_RANGE: &str = "tensor_cat([arange(0.0, 7.0) * 17.0, full([1], 128.0)], 0)";

pub const DECORATOR: &str = "@fused_kl_ce(enabled = true, vocab_size = 128, hidden_size = 32, teacher_hidden = 64, batch_size = 2, seq_len = 4, vocab_tile = 128)";
pub const LOSS_SECTION: &str = "    loss:\n        alpha = 0.6\n        temperature = 2.0\n";

/// PyTorch: the loss at each of the four steps.
pub const LOSSES: [f64; 4] = [3.07935422, 2.86753303, 2.67359455, 2.4927872];
/// PyTorch: `sum` of the student's `ws`, `xs`, `bs` after the four steps,
/// then of the teacher's `wt`, which must not have moved.
pub const SUMS: [f64; 4] = [45.838075, 3.577159, 0.225137, 30.648992];

pub fn numbers(text: &str) -> Vec<f64> {
    text.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+')))
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

/// `got` is the four losses followed by the four sums.
pub fn assert_matches_pytorch(got: &[f64], tolerance: f64, what: &str) {
    assert_eq!(got.len(), 8, "{what}: four losses and four sums expected, got {got:?}");
    for (i, (g, w)) in got.iter().zip(LOSSES.iter().chain(&SUMS)).enumerate() {
        assert!((g - w).abs() <= tolerance * (1.0 + w.abs()), "{what}: value {i} is {g}, expected {w} (all: {got:?})");
    }
}
