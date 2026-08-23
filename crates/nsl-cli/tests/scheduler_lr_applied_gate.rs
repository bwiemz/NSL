//! The `scheduler:` clause must actually modulate the learning rate.
//!
//! It did not, for any run with `grad_accumulation > 1` — which is every
//! production recipe in this repo. The scheduler was parsed, name/kwarg
//! validated, CALLED once per micro-step, and its result thrown away: the
//! FASE/deferred update lowering folded the optimizer's BASE lr into an
//! `f64const` and emitted that. Training ran at constant lr while
//! `warmup_cosine(...)` sat in the recipe looking load-bearing.
//!
//! WHY THE OLD GATE MISSED IT. `optim_config_contract_gate`'s
//! `a_valid_program_with_scheduler_and_callbacks_still_runs` asserts that the
//! program runs, the callback fires, and no diagnostics appear — while that
//! module's doc claims it proves the scheduler "actually applies". None of
//! those three can fail when the lr is silently constant. **A gate for
//! "feature X takes effect" must assert on a value X changes.**
//!
//! So these arms are built so that "applied" and "inert" are unmistakable
//! rather than subtle:
//!
//!   * FREEZE — `warmup_steps=0, total_steps=1, min_lr=0.0` makes
//!     `warmup_cosine` return 0.0 for every step >= 1. Applied => the loss
//!     stops moving, byte-identically. Inert => it keeps descending.
//!   * EXPLODE — the same shape with `min_lr=100.0`. Applied => the loss
//!     blows up by orders of magnitude. Inert => it trains normally.
//!     Asserted as DIVERGENCE, not as NaN: this fixture's linear model under
//!     MSE reaches ~1.6e5 rather than overflowing, while the same schedule on
//!     a 50M model under cross-entropy NaNs at once. NaN is one way to
//!     diverge, not the property being tested.
//!
//! A single arm would not be enough: FREEZE alone passes if training is
//! broken outright (a model that never learns also "stops moving"), which is
//! why the unscheduled CONTROL is asserted to descend in the same test.
//!
//! WHAT THIS FIXTURE COVERS, stated because it is less than the fix touches.
//! The fold appeared at four emitter sites; this program routes through
//! `emit_fused_multi_launch` (verified: re-folding lr THERE fails both tests
//! below, while re-folding it in the deferred `Update` op or the FASE fused
//! arm leaves them green). The CSLA multi-param arm and the ZeRO-3
//! elementwise step need `--layerwise-accum` / `--zero-stage 3` and are NOT
//! exercised here — they were fixed by inspection and checked by hand on a
//! Coder-50M GPU probe. A regression in those two would not fail this gate.
//!
//! `grad_accumulation = 2` IS THE TRIGGER and must not be dropped. With
//! accumulation 1 the update takes a different lowering that was always
//! correct, so an otherwise-identical fixture at accum 1 passes with the bug
//! fully present — the same shape of vacuity as the gate this one replaces.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// CPU + a fixed input: the loss stream is deterministic, so the freeze arm
/// can be compared BYTE-wise rather than with a tolerance.
fn fixture(scheduler_line: &str) -> String {
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

print("LOSS_STREAM_BEGIN")
train(model = m, epochs = 1, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.01)
{scheduler_line}    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, target)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
print("SCHED_GATE_DONE")
"#
    )
}

fn run(tag: &str, src: &str) -> (bool, String) {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_schedgate_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--seed", "7"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
    )
}

/// The printed loss values, as RAW strings — byte comparison, not float
/// closeness, so "stopped moving" means exactly that.
fn losses(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .skip_while(|l| !l.contains("LOSS_STREAM_BEGIN"))
        .take_while(|l| !l.contains("LOSS_STREAM_END"))
        .map(|l| l.trim())
        .filter(|l| l.parse::<f64>().is_ok() || l.eq_ignore_ascii_case("nan"))
        .map(|l| l.to_string())
        .collect()
}

const FREEZE: &str = "    scheduler: warmup_cosine(warmup_steps = 0, total_steps = 1, min_lr = 0.0)\n";
const EXPLODE: &str =
    "    scheduler: warmup_cosine(warmup_steps = 0, total_steps = 1, min_lr = 100.0)\n";

#[test]
fn a_schedule_that_drives_lr_to_zero_freezes_training() {
    // Control FIRST: the fixture must be able to learn, or "it stopped
    // moving" proves nothing at all.
    let (ok, ctl) = run("control", &fixture(""));
    assert!(ok, "unscheduled control failed to run:\n{ctl}");
    let ctl_losses = losses(&ctl);
    assert!(
        ctl_losses.len() >= 6,
        "control produced too few losses to judge:\n{ctl}"
    );
    assert!(
        ctl_losses.last() != ctl_losses.first(),
        "the unscheduled control did not train at all, so the freeze arm \
         below would pass for the wrong reason:\n{ctl_losses:?}"
    );

    let (ok, out) = run("freeze", &fixture(FREEZE));
    assert!(ok, "freeze arm failed to run:\n{out}");
    let l = losses(&out);
    assert_eq!(l.len(), ctl_losses.len(), "arms must be the same length");

    // lr hits 0 at the first scheduled step, so from that point the weights
    // cannot move and every later loss is the SAME BYTES.
    let tail = &l[2..];
    assert!(
        tail.iter().all(|v| v == &tail[0]),
        "lr was driven to 0.0 but training kept moving — the schedule is not \
         reaching the optimizer:\n{l:?}"
    );
    assert_ne!(
        l, ctl_losses,
        "the freeze arm is byte-identical to the UNSCHEDULED control, which \
         is exactly what an ignored scheduler looks like:\n{l:?}"
    );
}

#[test]
fn a_schedule_that_drives_lr_up_diverges() {
    let (ok, ctl) = run("explode_control", &fixture(""));
    assert!(ok, "unscheduled control failed to run:\n{ctl}");
    let ctl_max = losses(&ctl)
        .iter()
        .filter_map(|v| v.parse::<f64>().ok())
        .fold(0.0_f64, f64::max);
    assert!(ctl_max > 0.0, "control produced no finite loss:\n{ctl}");

    let (ok, out) = run("explode", &fixture(EXPLODE));
    assert!(ok, "explode arm failed to run:\n{out}");
    let l = losses(&out);
    let blew_up = l.iter().any(|v| {
        v.eq_ignore_ascii_case("nan")
            || v.parse::<f64>().map(|f| f > ctl_max * 100.0).unwrap_or(false)
    });
    assert!(
        blew_up,
        "lr was driven to 100.0 and training stayed well-behaved (control max \
         {ctl_max}) — the schedule is not reaching the optimizer:\n{l:?}"
    );
}
