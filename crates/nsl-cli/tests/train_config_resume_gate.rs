//! The resolved train/optimizer/scheduler configuration is checkpoint
//! identity (next-roadmap item 4).
//!
//! Item 8 refuses corpus/geometry/seed drift and #519 refuses compile-flag
//! arithmetic drift — but the CONFIG (lr, betas, weight decay, the
//! schedule, grad_accumulation, the clip) was recorded nowhere: editing any
//! of them between save and resume silently produced a run that is not a
//! continuation of the checkpoint, while the resume contract says "re-run
//! the file UNCHANGED with checkpoint_load= added".
//!
//! Two classes, split by what the drift corrupts:
//!
//!  * moment-meaning (optimizer kind, betas, eps, wd, grad_accumulation…):
//!    changes what the restored m/v and step counter MEAN. ABORT, no
//!    escape — `model_load(...)` is the weights-only warm start.
//!  * trajectory (lr, schedule shape, clip): changes the future, not the
//!    restored state's meaning — and mid-resume lr changes are a real
//!    workflow. ABORT naming `NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1`, which
//!    converts to a loud acknowledgment.
//!
//! Anti-vacuity is the first test: the sidecar must literally carry the
//! record and a matching resume must NOT print the skip notice — without
//! that, every drift test below could pass with the field never written
//! (the #519 lesson: a comparison that silently skips gates nothing).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Loader-free and dropout-free (the exec-gate precedent): this gate is
/// about the config record, and a data stream would only add unrelated ways
/// for the runs to differ. `opt_line` and `sched_line` are the drift axes.
fn fixture(train_cfg: &str, opt_line: &str, sched_line: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    emb: Tensor = randn([64, 4])

    fn forward(self, ids: Tensor) -> Tensor:
        return embedding_lookup(self.emb, ids.reshape([8]))

let m = Tiny()
let ids = full([2, 4], 3.0)
let target = zeros([8, 4])

train(model = m{train_cfg}):
    optimizer: {opt_line}
{sched_line}    step(batch):
        let pred = m.forward(ids)
        let loss = mse_loss(pred, target)

print("FIXTURE_DONE")
"#
    )
}

struct RunOut {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run_in(
    dir: &std::path::Path,
    name: &str,
    envs: &[(&str, &str)],
    src: &str,
) -> RunOut {
    let root = repo_root();
    let prog = dir.join(name);
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad"])
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // A CI environment exporting the escape globally must not leak in.
        .env_remove("NSL_RESUME_ALLOW_TRAJECTORY_DRIFT");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    RunOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

fn fresh_dir(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("nsl_cfggate_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    tmp
}

// Loaderless: ONE micro-step per epoch, and the save cadence is
// checkpoint_every * accum = 2 micro-steps — so epochs = 2 makes the save
// fire exactly once, at the end of the single optimizer step.
const SAVE_CFG: &str = r#", epochs = 2, grad_accumulation = 2, grad_clip = 1.0, checkpoint_save = "ck.nslm", checkpoint_every = 1"#;
const BASE_OPT: &str = "AdamW(lr = 0.01, weight_decay = 0.01, beta1 = 0.9, beta2 = 0.95, eps = 1e-8)";
const BASE_SCHED: &str =
    "    scheduler: warmup_cosine(warmup_steps = 2, total_steps = 8, min_lr = 0.001)\n";

fn load_cfg(extra: &str) -> String {
    format!(r#", epochs = 4, grad_accumulation = 2{extra}, checkpoint_load = "ck.nslm""#)
}

fn save_phase(dir: &std::path::Path) {
    let a = run_in(dir, "a.nsl", &[], &fixture(SAVE_CFG, BASE_OPT, BASE_SCHED));
    assert!(a.ok, "save phase failed:\n{}", a.stderr);
}

fn sidecar_text(dir: &std::path::Path) -> String {
    let raw = std::fs::read(dir.join("ck.nslm.optim")).expect("sidecar missing");
    String::from_utf8_lossy(&raw[..raw.len().min(4096)]).to_string()
}

#[test]
fn record_is_written_and_a_matching_resume_passes_without_skipping() {
    let dir = fresh_dir("match");
    save_phase(&dir);

    let header = sidecar_text(&dir);
    // EVERY key the render emits, not a sample: the review proved a deleted
    // render arm for an unpinned key is invisible to every behavioral test
    // (absent-on-both-sides is no-diff by design). Values where the fixture
    // fixes them, bare `key=` where only presence is checkable.
    for needle in [
        "\"train_cfg\":\"",
        "opt=adamw",
        ",lr=0.01",
        "accum=2",
        "clip=1",
        "sched=warmup_cosine",
        "sp1=2",
        "sp2=8",
        "sp3=0.001",
        "wd=0.01",
        "beta1=0.9",
        "beta2=0.95",
        "eps=0.00000001",
        "momentum=",
        "dampening=",
        "nesterov=0",
        "ns_steps=",
        "adamw_lr=none",
        "no_decay=none",
    ] {
        assert!(
            header.contains(needle),
            "sidecar lacks {needle:?} — the record is not being written and \
             every drift test in this file is vacuous:\n{header}"
        );
    }

    let b = run_in(
        &dir,
        "b.nsl",
        &[],
        &fixture(&load_cfg(", grad_clip = 1.0"), BASE_OPT, BASE_SCHED),
    );
    assert!(b.ok, "matching resume failed:\n{}", b.stderr);
    assert!(
        !b.stderr.contains("check is SKIPPED"),
        "matching resume skipped the config check — the live side is not \
         installed:\n{}",
        b.stderr
    );
    assert!(b.stdout.contains("FIXTURE_DONE"));
}

#[test]
fn changing_lr_mid_resume_is_refused_and_names_the_escape() {
    let dir = fresh_dir("lr");
    save_phase(&dir);
    let opt2 = BASE_OPT.replace("lr = 0.01", "lr = 0.02");
    let b = run_in(
        &dir,
        "b.nsl",
        &[],
        &fixture(&load_cfg(", grad_clip = 1.0"), &opt2, BASE_SCHED),
    );
    assert!(!b.ok, "an lr change mid-resume must refuse:\n{}", b.stdout);
    assert!(
        b.stderr.contains("LR/SCHEDULE/CLIP differs")
            && b.stderr.contains("lr: checkpoint 0.01 -> this run 0.02")
            && b.stderr.contains("NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1"),
        "refusal must name the drift and the escape:\n{}",
        b.stderr
    );
    assert!(!b.stdout.contains("FIXTURE_DONE"));
}

#[test]
fn the_escape_env_converts_trajectory_drift_into_an_acknowledged_resume() {
    let dir = fresh_dir("lr_ack");
    save_phase(&dir);
    let opt2 = BASE_OPT.replace("lr = 0.01", "lr = 0.02");
    let b = run_in(
        &dir,
        "b.nsl",
        &[("NSL_RESUME_ALLOW_TRAJECTORY_DRIFT", "1")],
        &fixture(&load_cfg(", grad_clip = 1.0"), &opt2, BASE_SCHED),
    );
    assert!(b.ok, "acknowledged trajectory drift must resume:\n{}", b.stderr);
    assert!(
        b.stderr.contains("TRAJECTORY drift acknowledged")
            && b.stderr.contains("lr: checkpoint 0.01 -> this run 0.02"),
        "the acknowledgment must be loud and name the change:\n{}",
        b.stderr
    );
    assert!(b.stdout.contains("FIXTURE_DONE"));
}

#[test]
fn changing_the_schedule_mid_resume_is_trajectory_drift() {
    let dir = fresh_dir("sched");
    save_phase(&dir);
    let sched2 =
        "    scheduler: warmup_cosine(warmup_steps = 4, total_steps = 8, min_lr = 0.001)\n";
    let b = run_in(
        &dir,
        "b.nsl",
        &[],
        &fixture(&load_cfg(", grad_clip = 1.0"), BASE_OPT, sched2),
    );
    assert!(!b.ok, "a warmup change mid-resume must refuse:\n{}", b.stdout);
    assert!(
        b.stderr.contains("sp1: checkpoint 2 -> this run 4"),
        "refusal must name the changed schedule parameter:\n{}",
        b.stderr
    );
}

#[test]
fn moment_class_drift_is_refused_even_with_the_escape_env() {
    let dir = fresh_dir("beta");
    save_phase(&dir);
    let opt2 = BASE_OPT.replace("beta2 = 0.95", "beta2 = 0.999");
    let b = run_in(
        &dir,
        "b.nsl",
        &[("NSL_RESUME_ALLOW_TRAJECTORY_DRIFT", "1")],
        &fixture(&load_cfg(", grad_clip = 1.0"), &opt2, BASE_SCHED),
    );
    assert!(
        !b.ok,
        "a beta2 change must refuse even with the trajectory escape — it \
         changes what the restored v moments MEAN:\n{}",
        b.stdout
    );
    assert!(
        b.stderr.contains("OPTIMIZER CONFIGURATION")
            && b.stderr.contains("beta2: checkpoint 0.95 -> this run 0.999")
            && b.stderr.contains("model_load"),
        "refusal must name the drift and the weights-only escape hatch:\n{}",
        b.stderr
    );
}

#[test]
fn changing_grad_accumulation_mid_resume_is_moment_class() {
    let dir = fresh_dir("accum");
    save_phase(&dir);
    // accum is the optimizer-step divisor and the bias-correction clock for
    // the RESTORED step counter — item 4's map found it covered by NO
    // existing guard (not in the loader identity, not in the exec record).
    let b = run_in(
        &dir,
        "b.nsl",
        &[("NSL_RESUME_ALLOW_TRAJECTORY_DRIFT", "1")],
        &fixture(
            &format!(r#", epochs = 4, grad_accumulation = 4, grad_clip = 1.0, checkpoint_load = "ck.nslm""#),
            BASE_OPT,
            BASE_SCHED,
        ),
    );
    assert!(!b.ok, "an accum change mid-resume must refuse:\n{}", b.stdout);
    assert!(
        b.stderr.contains("accum: checkpoint 2 -> this run 4"),
        "refusal must name the accumulation change:\n{}",
        b.stderr
    );
}

/// An old sidecar without the field must SKIP loudly, not refuse and not
/// silently pass. Constructed by blanking the record in place (same header
/// length, so the size word stays honest — the v1-construction precedent).
#[test]
fn a_sidecar_without_the_record_skips_loudly_and_resumes() {
    let dir = fresh_dir("old");
    save_phase(&dir);
    let path = dir.join("ck.nslm.optim");
    let mut raw = std::fs::read(&path).unwrap();
    let needle = b"\"train_cfg\":\"";
    let start = raw
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("record not in sidecar");
    let vstart = start + needle.len();
    let vend = vstart
        + raw[vstart..]
            .iter()
            .position(|&b| b == b'"')
            .expect("unterminated record");
    // Blank the key and the value; spaces are invisible to needle scanners.
    for b in &mut raw[start..=vend] {
        *b = b' ';
    }
    std::fs::write(&path, raw).unwrap();

    let b = run_in(
        &dir,
        "b.nsl",
        &[],
        &fixture(&load_cfg(", grad_clip = 1.0"), BASE_OPT, BASE_SCHED),
    );
    assert!(b.ok, "a record-less sidecar must still resume:\n{}", b.stderr);
    assert!(
        b.stderr.contains("no train-config") && b.stderr.contains("SKIPPED"),
        "the skip must be loud:\n{}",
        b.stderr
    );
    assert!(b.stdout.contains("FIXTURE_DONE"));
}
