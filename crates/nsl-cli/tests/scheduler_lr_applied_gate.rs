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
//! ROUTE COVERAGE (next-roadmap item 3, 2026-08-25). The original two tests
//! cover exactly ONE update route — `emit_fused_multi_launch`, non-clip —
//! and their doc said so. The per-route tests below extend the same
//! freeze/control differential to every CPU-reachable single-GPU route:
//! standard per-param stdlib (accum=1 and Lion/FullBuffer), the fused
//! multi-tensor TWO-PHASE-CLIP call site, the three FASE-Deferred
//! per-param sub-arms (fused single kernel, interpreted Update+wd via
//! --training-reference, SgdUpdate), the --optim-state-offload selection,
//! and CSLA layerwise (batched multi_idx arm and, under
//! NSL_FASE_MULTI_STEP=0, the per-param fallback arm).
//!
//! DELIBERATELY NOT COVERED HERE, so the omission is a statement: the
//! ZeRO-3 elementwise step and the bf16-sr twins (GPU/multi-rank only —
//! the roadmap item scopes to single-GPU), the WGGO unified dispatcher
//! (needs per-layer --wggo overrides; its two arms re-enter the covered
//! emitters), and the Muon-CSLA lr read (the `MuonCslaCtx { lr: … }`
//! construction in stmt.rs; the struct lives in stmt_csla.rs) — recorded
//! as the one #520 lr site with no dedicated gate.
//!
//! WHY NO "applied lr" observation surface was added instead: a reported
//! number can lie in exactly the #520 shape — the scheduler computes the
//! right value and the update ignores it. Only a behavioral differential
//! proves the UPDATE changed. The freeze arm is the sharp edge: lr driven
//! to 0.0 must stop the weights BYTE-identically, and a route that ignores
//! the scheduler keeps training at base lr.
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


// ---------------------------------------------------------------------------
// Per-route coverage (item 3)
// ---------------------------------------------------------------------------

/// One CPU-reachable update route: how to build the fixture and how to
/// prove, from the run itself, that THIS route fired (anti-vacuity — a
/// config typo that silently lands on the already-covered fused route must
/// not pass as new coverage).
struct Route {
    tag: &'static str,
    /// train(...) config after `model = m`.
    train_cfg: &'static str,
    optimizer: &'static str,
    extra_flags: &'static [&'static str],
    env: &'static [(&'static str, &'static str)],
    /// Substring that must appear in stderr (route witness), if any.
    stderr_witness: Option<&'static str>,
    /// Substring that must NOT appear in stderr, if any.
    stderr_absent: Option<&'static str>,
}

fn route_fixture(r: &Route, scheduler_line: &str) -> String {
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
train(model = m, {cfg}):
    optimizer: {opt}
{sched}    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, target)
    callbacks:
        on_step(step, loss):
            print(loss)
print("LOSS_STREAM_END")
print("SCHED_GATE_DONE")
"#,
        cfg = r.train_cfg,
        opt = r.optimizer,
        sched = scheduler_line,
    )
}

fn run_route(r: &Route, tag2: &str, src: &str) -> (bool, String, String) {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!(
        "nsl_schedroute_{}_{}_{tag2}",
        std::process::id(),
        r.tag
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--seed", "7"]);
    for f in r.extra_flags {
        cmd.arg(f);
    }
    cmd.arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for (k, v) in r.env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// The freeze/control differential, per route. Same proof as the original
/// tests: the control must LEARN (anti-vacuity), the freeze arm must stop
/// byte-identically from `FREEZE_FROM` on and differ from the control.
/// Index 2 is the one-micro-step lag: the optimizer step at micro k applies
/// the schedule value computed after micro k-1, so the first update still
/// runs at base lr in every route.
fn assert_route_freeze_differential(r: &Route) {
    let (ok, ctl, ctl_err) = run_route(r, "control", &route_fixture(r, ""));
    assert!(ok, "[{}] unscheduled control failed:\n{ctl}\n{ctl_err}", r.tag);
    let ctl_losses = losses(&ctl);
    assert!(
        ctl_losses.len() >= 6,
        "[{}] control produced too few losses:\n{ctl}",
        r.tag
    );
    assert!(
        ctl_losses.last() != ctl_losses.first(),
        "[{}] the unscheduled control did not train — the freeze arm would \
         pass for the wrong reason:\n{ctl_losses:?}",
        r.tag
    );

    let (ok, out, err) = run_route(r, "freeze", &route_fixture(r, FREEZE));
    assert!(ok, "[{}] freeze arm failed:\n{out}\n{err}", r.tag);
    if let Some(w) = r.stderr_witness {
        assert!(
            err.contains(w),
            "[{}] route witness '{w}' missing — the fixture did not take \
             the route this test exists to cover:\n{err}",
            r.tag
        );
    }
    if let Some(a) = r.stderr_absent {
        assert!(
            !err.contains(a),
            "[{}] '{a}' present — the fixture landed on a different route \
             than this test claims to cover:\n{err}",
            r.tag
        );
    }
    let l = losses(&out);
    assert_eq!(
        l.len(),
        ctl_losses.len(),
        "[{}] arms must be the same length",
        r.tag
    );
    let tail = &l[2..];
    assert!(
        tail.iter().all(|v| v == &tail[0]),
        "[{}] lr was driven to 0.0 but training kept moving — the schedule \
         is not reaching THIS route's update:\n{l:?}",
        r.tag
    );
    assert_ne!(
        l, ctl_losses,
        "[{}] freeze arm is byte-identical to the unscheduled control — \
         exactly what an ignored scheduler looks like",
        r.tag
    );
}

/// Route (a): standard per-param stdlib update — `grad_accumulation` absent,
/// so FASE plans Passthrough and the final `else` per-param loop runs. This
/// lowering "was always correct" in #520's post-mortem; that is a reason to
/// pin it, not to skip it.
#[test]
fn standard_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "standard",
        train_cfg: "epochs = 1",
        optimizer: "AdamW(lr = 0.01)",
        extra_flags: &[],
        env: &[("NSL_FASE_FUSED_COUNTER", "1")],
        // The armed counter prints its line unconditionally; ZERO launches
        // is the positive witness that the stdlib per-param route ran.
        stderr_witness: Some("optimizer fused-step launches: 0"),
        stderr_absent: None,
    });
}

/// Route (a), Lion arm: Lion plans FullBuffer regardless of accumulation,
/// so even with accum=2 the stdlib per-param loop runs — a second, distinct
/// entry into route (a) whose update (sign * lr) is maximally lr-sensitive.
#[test]
fn lion_fullbuffer_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "lion",
        train_cfg: "epochs = 1, grad_accumulation = 2",
        optimizer: "Lion(lr = 0.01)",
        extra_flags: &[],
        env: &[("NSL_FASE_FUSED_COUNTER", "1")],
        stderr_witness: Some("optimizer fused-step launches: 0"),
        stderr_absent: None,
    });
}

/// Route (b), TWO-PHASE-CLIP call site of `emit_fused_multi_launch` — the
/// original tests cover the non-clip site only; `grad_clip` selects a
/// genuinely separate call site with its own lr threading.
#[test]
fn fused_multi_clip_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "fused_clip",
        train_cfg: "epochs = 1, grad_accumulation = 2, grad_clip = 1.0",
        optimizer: "AdamW(lr = 0.01)",
        extra_flags: &[],
        env: &[],
        stderr_witness: None,
        stderr_absent: None,
    });
}

/// Route (c), SgdUpdate arm: SGD has one state buffer, which fails the
/// multi-tensor admission, and `match_adamw_program` cannot match it — the
/// interpreted SgdUpdate in `fase_emit_final_step` is the only path left.
#[test]
fn fase_perparam_sgd_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "fase_sgd",
        train_cfg: "epochs = 1, grad_accumulation = 2",
        optimizer: "SGD(lr = 0.05)",
        extra_flags: &[],
        env: &[],
        stderr_witness: None,
        stderr_absent: None,
    });
}

/// Route (c), fused single-kernel arm: AdamW under accumulation with the
/// MULTI batching kill-switched — per-param `fase_emit_final_step`, fused
/// scalar arm. The fused-step counter is the route witness (it counts on
/// CPU too).
#[test]
fn fase_perparam_fused_single_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "fase_single",
        train_cfg: "epochs = 1, grad_accumulation = 2",
        optimizer: "AdamW(lr = 0.01)",
        extra_flags: &[],
        env: &[("NSL_FASE_MULTI_STEP", "0"), ("NSL_FASE_FUSED_COUNTER", "1")],
        stderr_witness: Some("[fase-fused] optimizer fused-step launches"),
        stderr_absent: Some("optimizer fused-step launches: 0"),
    });
}

/// Route (c), interpreted Update + weight-decay arm: `--training-reference`
/// kills both fused arms, leaving the interpreted UpdateOp path — the site
/// where #520 found lr folded AND the wd term multiplied into it.
#[test]
fn fase_perparam_interpreted_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "fase_interp",
        train_cfg: "epochs = 1, grad_accumulation = 2",
        optimizer: "AdamW(lr = 0.01, weight_decay = 0.01)",
        extra_flags: &["--training-reference"],
        env: &[],
        stderr_witness: None,
        stderr_absent: None,
    });
}

/// Route (b/c seam), TWO-PHASE-CLIP per-param Phase B: `grad_clip` with an
/// optimizer that FAILS the multi admission (SGD's single state buffer)
/// lands on the clip loop's per-param `fase_emit_final_step` call — a
/// distinct lr forwarding the review's mutation M8 proved was covered by
/// NO test (a #520-shaped fold there passed the entire suite), reachable
/// with no env vars at all.
#[test]
fn clip_perparam_phase_b_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "clip_perparam",
        train_cfg: "epochs = 1, grad_accumulation = 2, grad_clip = 1.0",
        optimizer: "SGD(lr = 0.05)",
        extra_flags: &[],
        env: &[],
        stderr_witness: None,
        stderr_absent: None,
    });
}

/// `--optim-state-offload` is not its own emitter but it CHANGES ROUTE
/// SELECTION (excluded from the multi admission), and its staging wraps the
/// per-param step — a scheduler regression specific to the offload
/// composition would hit every pre-item-11 production 1B run.
#[test]
fn offload_perparam_route_applies_the_schedule() {
    assert_route_freeze_differential(&Route {
        tag: "offload",
        train_cfg: "epochs = 1, grad_accumulation = 2",
        optimizer: "AdamW(lr = 0.01)",
        extra_flags: &["--optim-state-offload"],
        env: &[],
        stderr_witness: None,
        stderr_absent: None,
    });
}


// ---------------------------------------------------------------------------
// Route (d): CSLA layerwise — the committed FFN fixture with a scheduler
// line injected after its optimizer. CSLA's lr reads live at the WINDOW
// boundaries (before the per-micro scheduler eval), so the same
// one-step-lag freeze offset applies. `NSL_CSLA_COUNTER=1` arms the
// window-phase counter as the route witness.
// ---------------------------------------------------------------------------

fn run_csla(tag: &str, scheduler_line: &str, envs: &[(&str, &str)]) -> (String, String) {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_schedcsla_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .unwrap();
    src = src.replace(
        "CSLA_SAVE_PATH",
        &dir.join("out.nslm").display().to_string().replace('\\', "/"),
    );
    let opt_line = "    optimizer: AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)";
    assert!(src.contains(opt_line), "fixture optimizer line moved");
    src = src.replace(opt_line, &format!("{opt_line}\n{scheduler_line}"));
    // Two epochs over the UNSHUFFLED loader: with frozen params, epoch 2
    // must echo epoch 1's per-batch losses byte-for-byte (the fixture's
    // data VARIES per micro-batch, so within-epoch byte-identity — the
    // Tiny-fixture proof — does not apply here).
    assert!(src.contains("epochs=1"), "fixture epochs literal moved");
    src = src.replace("epochs=1", "epochs=2");
    let prog = dir.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args([
        "run",
        "--deterministic",
        "--source-ad",
        "--checkpoint-blocks",
        "--layerwise-accum",
    ])
    .arg(&prog)
    .current_dir(&dir)
    .env("NSL_STDLIB_PATH", root.join("stdlib"))
    .env("NSL_CSLA_COUNTER", "1");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    assert!(
        out.status.success(),
        "[csla {tag}] run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn assert_csla_freeze(tag: &str, envs: &[(&str, &str)]) {
    let (ctl, ctl_err) = run_csla(&format!("{tag}_control"), "", envs);
    assert!(
        ctl_err.contains("[csla] window backward phases"),
        "[csla {tag}] route witness missing — the fixture did not take the \
         CSLA window path:\n{ctl_err}"
    );
    let ctl_losses = losses(&ctl);
    assert!(ctl_losses.len() >= 6, "[csla {tag}] too few losses:\n{ctl}");
    assert!(
        ctl_losses.last() != ctl_losses.first(),
        "[csla {tag}] control did not train:\n{ctl_losses:?}"
    );

    let sched =
        "    scheduler: warmup_cosine(warmup_steps=0, total_steps=1, min_lr=0.0)";
    let (out, err) = run_csla(&format!("{tag}_freeze"), sched, envs);
    assert!(
        err.contains("[csla] window backward phases"),
        "[csla {tag}] freeze arm lost the route witness:\n{err}"
    );
    let l = losses(&out);
    assert_eq!(l.len(), ctl_losses.len(), "[csla {tag}] arms must match in length");
    // EPOCH ECHO. The schedule zeroes lr from micro-step 1 on, so after the
    // first window's base-lr update the params never move again — epoch 2
    // must replay epoch 1's per-batch losses BYTE-identically from index 2
    // (the one-step lag) onward. The unscheduled control must NOT echo.
    let per_epoch = l.len() / 2;
    assert!(per_epoch >= 6, "[csla {tag}] too few losses per epoch: {l:?}");
    for i in 2..per_epoch {
        assert_eq!(
            l[per_epoch + i], l[i],
            "[csla {tag}] epoch 2 diverged from epoch 1 at batch {i} with lr \
             frozen to 0.0 — the schedule is not reaching the CSLA group \
             update:\n{l:?}"
        );
    }
    let ctl_echoes = (2..per_epoch).all(|i| ctl_losses[per_epoch + i] == ctl_losses[i]);
    assert!(
        !ctl_echoes,
        "[csla {tag}] the UNSCHEDULED control also echoes epoch 1 — training \
         is not moving and this differential proves nothing:\n{ctl_losses:?}"
    );
}

/// CSLA batched multi_idx arm (the codegen default; the runtime falls back
/// per-param for CPU members but consumes the SAME threaded lr argument).
#[test]
fn csla_multi_idx_route_applies_the_schedule() {
    assert_csla_freeze("multi", &[]);
}

/// CSLA per-param fallback arm at the CODEGEN level: NSL_FASE_MULTI_STEP=0
/// makes `emit_csla_group_update` emit `fase_emit_final_step` per param —
/// a different lr threading site than the multi_idx call.
#[test]
fn csla_perparam_fallback_route_applies_the_schedule() {
    assert_csla_freeze("fallback", &[("NSL_FASE_MULTI_STEP", "0")]);
}
