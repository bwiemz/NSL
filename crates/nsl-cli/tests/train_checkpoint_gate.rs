//! Milestone B gates for full-train-state checkpoint/restart.
//!
//! `train(checkpoint_save=…, checkpoint_every=N)` writes θ (`.nslm`) plus an
//! `.optim` sidecar (AdamW m/v + the micro-batch step counter) at optimizer-
//! step boundaries, atomically (tmp + rename). `checkpoint_load=…` restores
//! all of it and seeds the step counter, so bias correction, the scheduler,
//! and the checkpoint cadence continue instead of re-warming.
//!
//! The headline gate is BIT-EXACTNESS on CPU: an interrupted run (save at
//! step 6, new process, resume, 3 more steps) must print byte-identical loss
//! lines to an uninterrupted 9-step control. CPU f64 with a fixed input is
//! deterministic, so any drift — a moment not restored, a step counter
//! re-zeroed (bias correction re-warms: m̂ = m/(1-β₁ᵗ) is 10x at t=1), an
//! LR-schedule reset — turns the gate red. A loss-level "close enough"
//! check would pass all of those (the srbf16_campaign RESUMED check
//! tolerates half the parent's loss drop; this gate exists to be strict
//! where strictness is cheap).
//!
//! GPU-scale resume continuity for the 1B workload is validated by the
//! endurance harness (models/benchmarks/endurance_1b.py), not here — at 1B
//! bit-identity is not an environment guarantee (TF32 GEMMs, fused-CE
//! backward atomics; see long_run_drift_gpu_gate's control-vs-control
//! doctrine).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// A tiny AdamW training program. `train_cfg` is spliced into the train(...)
/// arg list verbatim; the loss stream prints one (step, loss) line pair per
/// micro-batch.
fn fixture(train_cfg: &str, optimizer: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()

let x = full([2, 2], 2.0)
let y = zeros([2, 2])
train(model = m{train_cfg}):
    optimizer: {optimizer}
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(step)
            print(loss)

print("FIXTURE_DONE")
"#
    )
}

struct RunOut {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run_in(dir: &std::path::Path, name: &str, src: &str) -> RunOut {
    let root = repo_root();
    let prog = dir.join(name);
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad"])
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    RunOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

fn fresh_dir(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("nsl_ckptgate_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    tmp
}

/// The loss stream: (step, loss) line pairs from stdout, as raw strings —
/// raw because the comparison below is BYTE equality, not parsed-float
/// closeness.
fn loss_lines(stdout: &str) -> Vec<(String, String)> {
    let lines: Vec<&str> = stdout.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < lines.len() {
        if lines[i].trim().parse::<i64>().is_ok() {
            out.push((lines[i].trim().to_string(), lines[i + 1].trim().to_string()));
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

#[test]
fn resume_is_bit_exact_with_uninterrupted_control() {
    let tmp = fresh_dir("bitexact");

    // Phase A: 6 micro-batches, save every 2 optimizer steps (accum=1).
    let a = run_in(
        &tmp,
        "phase_a.nsl",
        &fixture(
            r#", epochs = 6, checkpoint_save = "ck.nslm", checkpoint_every = 2"#,
            "AdamW(lr = 0.01)",
        ),
    );
    assert!(a.ok, "phase A failed:\n{}", a.stderr);
    let saves = a.stderr.matches("[checkpoint] saved:").count();
    assert_eq!(saves, 3, "expected saves at steps 2/4/6:\n{}", a.stderr);
    assert!(
        a.stderr.contains("at micro-batch step 6"),
        "last save must be the step-6 boundary:\n{}",
        a.stderr
    );
    assert!(tmp.join("ck.nslm").exists(), "θ checkpoint missing");
    assert!(tmp.join("ck.nslm.optim").exists(), "optim sidecar missing");

    // Phase B: NEW process, resume, run out to the TOTAL of 9.
    //
    // Item 8 fixed the meaning of `epochs` under resume: it is the run total,
    // not "how many more". Before, a resumed run restarted the epoch counter
    // at 0, so an unedited re-run of a recipe trained its full epoch budget a
    // SECOND time, and an author who wanted the real remainder had to compute
    // it by hand from a step counter. Now the checkpoint carries the epoch
    // and the loop continues into it — which is also what makes the loader
    // position meaningful (epoch 3 slot 412 is a position, 0 is not).
    let b = run_in(
        &tmp,
        "phase_b.nsl",
        &fixture(r#", epochs = 9, checkpoint_load = "ck.nslm""#, "AdamW(lr = 0.01)"),
    );
    assert!(b.ok, "phase B failed:\n{}", b.stderr);
    assert!(
        b.stderr.contains("[checkpoint] resumed:") && b.stderr.contains("step 6"),
        "resume witness missing:\n{}",
        b.stderr
    );

    // Control: 9 steps uninterrupted, in its own dir so it cannot touch the
    // checkpoint files.
    let ctl_dir = fresh_dir("bitexact_ctl");
    let c = run_in(
        &ctl_dir,
        "control.nsl",
        &fixture(", epochs = 9", "AdamW(lr = 0.01)"),
    );
    assert!(c.ok, "control failed:\n{}", c.stderr);

    let b_stream = loss_lines(&b.stdout);
    let c_stream = loss_lines(&c.stdout);
    assert_eq!(b_stream.len(), 3, "resumed run must print steps 7..9:\n{}", b.stdout);
    assert_eq!(c_stream.len(), 9, "control must print steps 1..9:\n{}", c.stdout);
    // Steps continue from the restored counter — 7, 8, 9 — and every loss
    // byte matches the uninterrupted control. This is the whole contract:
    // a resumed run is indistinguishable from one that never stopped.
    assert_eq!(
        b_stream,
        c_stream[6..].to_vec(),
        "resumed trajectory diverged from the uninterrupted control"
    );

    let _ = std::fs::remove_dir_all(&tmp);
    let _ = std::fs::remove_dir_all(&ctl_dir);
}

#[test]
fn checkpoint_cadence_respects_grad_accumulation() {
    let tmp = fresh_dir("accum");
    // accum=2: 6 micro-batches = 3 optimizer steps; every=1 must save at
    // micro-batch steps 2, 4, 6 — never mid-accumulation.
    let a = run_in(
        &tmp,
        "accum.nsl",
        &fixture(
            r#", epochs = 6, grad_accumulation = 2, checkpoint_save = "ck.nslm", checkpoint_every = 1"#,
            "AdamW(lr = 0.01)",
        ),
    );
    assert!(a.ok, "accum run failed:\n{}", a.stderr);
    for step in ["step 2", "step 4", "step 6"] {
        assert!(
            a.stderr.contains(&format!("at micro-batch {step}")),
            "missing boundary save at {step}:\n{}",
            a.stderr
        );
    }
    assert_eq!(
        a.stderr.matches("[checkpoint] saved:").count(),
        3,
        "a mid-accumulation save would corrupt the resume contract:\n{}",
        a.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn checkpoint_save_without_every_is_a_compile_error() {
    let tmp = fresh_dir("noevery");
    let r = run_in(
        &tmp,
        "noevery.nsl",
        &fixture(r#", epochs = 2, checkpoint_save = "ck.nslm""#, "AdamW(lr = 0.01)"),
    );
    assert!(!r.ok, "checkpoint_save without checkpoint_every must refuse");
    assert!(
        r.stderr.contains("checkpoint_every"),
        "refusal must name the missing arg:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn checkpoint_refuses_non_adamw_optimizers() {
    let tmp = fresh_dir("sgd");
    let r = run_in(
        &tmp,
        "sgd.nsl",
        &fixture(
            r#", epochs = 2, checkpoint_save = "ck.nslm", checkpoint_every = 1"#,
            "SGD(lr = 0.01)",
        ),
    );
    assert!(!r.ok, "checkpoint under SGD must refuse (sidecar is AdamW m/v)");
    assert!(
        r.stderr.contains("AdamW"),
        "refusal must explain the optimizer contract:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn missing_checkpoint_aborts_loudly() {
    let tmp = fresh_dir("missing");
    let r = run_in(
        &tmp,
        "missing.nsl",
        &fixture(r#", epochs = 2, checkpoint_load = "no_such.nslm""#, "AdamW(lr = 0.01)"),
    );
    assert!(
        !r.ok,
        "a resume from a missing checkpoint must abort, not silently start fresh"
    );
    assert!(
        r.stderr.contains("cannot read"),
        "abort must name the missing file:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn non_literal_checkpoint_args_are_compile_errors() {
    let tmp = fresh_dir("nonlit");
    // A variable path defeats the literal-only contract the runtime length
    // read depends on (same contract as model_save).
    let src = fixture(", epochs = 2, checkpoint_save = p, checkpoint_every = 1", "AdamW(lr = 0.01)")
        .replace("let m = Tiny()", "let p = \"ck.nslm\"\nlet m = Tiny()");
    let r = run_in(&tmp, "nonlit.nsl", &src);
    assert!(!r.ok, "non-literal checkpoint_save must refuse");
    assert!(
        r.stderr.contains("string literal"),
        "refusal must state the literal contract:\n{}",
        r.stderr
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
