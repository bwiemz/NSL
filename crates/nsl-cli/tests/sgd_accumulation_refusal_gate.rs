//! `SGD(momentum | weight_decay | dampening | nesterov)` under
//! `grad_accumulation > 1` is refused.
//!
//! The accumulated SGD step is the bare `theta -= lr * mean(g)`: the plan
//! carries no momentum, the velocity buffer is allocated and never read, and
//! weight decay, dampening and Nesterov are not applied. With
//! `grad_accumulation = 1` the same optimizer line runs the stdlib
//! `sgd_step`, which honours all four, so one source trained two different
//! models depending on a knob that is supposed to change only the batch
//! size. `SGD(lr=0.1, momentum=0.9)` on a constant gradient moves 0.561 in
//! three steps; under accumulation it moved 0.300.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run(tag: &str, train_cfg: &str, optimizer: &str) -> (bool, String, String) {
    let dir = std::env::temp_dir().join(format!("nsl_sgdaccum_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(
        &prog,
        format!(
            r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = randn([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let corpus = arange(64)
let loader = DataLoader(corpus, batch_size = 2, seq_len = 4, shuffle = false, drop_last = true)
let x = full([4, 8], 0.5)
let target = zeros([4, 8])

train(model = m, {train_cfg}):
    optimizer: {optimizer}
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, target)
print("SGD_ACCUM_DONE")
"#
        ),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cpu", "--source-ad", "--seed", "7"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn the_options_the_accumulated_step_ignores_are_refused() {
    for (tag, optimizer) in [
        ("momentum", "SGD(lr = 0.1, momentum = 0.9)"),
        ("decay", "SGD(lr = 0.1, weight_decay = 0.01)"),
        ("nesterov", "SGD(lr = 0.1, momentum = 0.9, nesterov = true)"),
    ] {
        let (ok, out, err) = run(tag, "epochs = 1, grad_accumulation = 2", optimizer);
        assert!(!ok && !out.contains("SGD_ACCUM_DONE"), "[{tag}] `{optimizer}` trained under accumulation:\n{out}");
        assert!(
            err.contains("with grad_accumulation=2 is not supported"),
            "[{tag}] the refusal must say what is unsupported:\n{err}"
        );
    }
}

/// What is implemented keeps working: plain SGD under accumulation, and the
/// full optimizer without it.
#[test]
fn plain_sgd_accumulates_and_momentum_runs_without_accumulation() {
    for (tag, cfg, optimizer) in [
        ("plain_accum", "epochs = 1, grad_accumulation = 2", "SGD(lr = 0.1)"),
        ("momentum_noaccum", "epochs = 1", "SGD(lr = 0.1, momentum = 0.9, weight_decay = 0.01)"),
    ] {
        let (ok, out, err) = run(tag, cfg, optimizer);
        assert!(ok && out.contains("SGD_ACCUM_DONE"), "[{tag}] `{optimizer}` ({cfg}) must run:\n{out}\n{err}");
    }
}
