//! Resume must refuse a build whose ARITHMETIC differs from the checkpoint's.
//!
//! Item 8 made training state resumable and guarded the inputs that live on
//! the command line rather than in the recipe: `--seed`, the corpus, the
//! batch geometry. It did not guard the COMPILE FLAGS. `--source-ad` vs the
//! tape, `--deterministic`, `--dtype`, the fused backward kernels — each
//! changes what a step computes, each is a command-line word, and the resume
//! workflow is documented as "re-run the recipe unchanged". Dropping one
//! resumed θ and the AdamW moments under different arithmetic and said
//! nothing.
//!
//! THREE GUARDS, because each alone passes with the feature broken:
//!
//! 1. `fingerprint_is_recorded_and_matching_resume_succeeds` — the record is
//!    actually WRITTEN. Without this, a fingerprint that is never emitted
//!    makes guard 3 pass for the wrong reason: both sides empty, comparison
//!    skipped, resume allowed.
//! 2. `dropping_source_ad_on_resume_is_refused` — the refusal FIRES, and the
//!    message names the offending key. Asserting only "the run failed" would
//!    pass on any unrelated abort.
//! 3. `toggling_placement_flags_warns_but_resumes` — the placement class is
//!    NOT refused. This is the guard against a future "simplification" that
//!    collapses the two classes into one abort: doing so would break the
//!    shipped production-1B workflow, which resumes with
//!    `--optim-state-offload` toggled, and would be invisible to guards 1-2.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Deliberately loader-free and dropout-free: this gate is about compile
/// flags, and a shuffled loader or a mask stream would only add ways for the
/// run to differ that have nothing to do with the fingerprint.
fn fixture(train_cfg: &str) -> String {
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
    optimizer: AdamW(lr = 0.01)
    step(batch):
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

fn run_in(dir: &std::path::Path, name: &str, flags: &[&str], src: &str) -> RunOut {
    let root = repo_root();
    let prog = dir.join(name);
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
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
    let tmp = std::env::temp_dir().join(format!("nsl_fpgate_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    tmp
}

fn save_cfg() -> String {
    fixture(r#", epochs = 1, checkpoint_save = "ck.nslm", checkpoint_every = 1"#)
}

fn load_cfg() -> String {
    fixture(r#", epochs = 2, checkpoint_load = "ck.nslm""#)
}

/// The sidecar item 8 writes alongside the `.nslm`.
fn sidecar_text(dir: &std::path::Path) -> String {
    let mut found = String::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) == Some("optim") {
            let raw = std::fs::read(&p).unwrap();
            found = String::from_utf8_lossy(&raw[..raw.len().min(4096)]).to_string();
        }
    }
    found
}

#[test]
fn fingerprint_is_recorded_and_matching_resume_succeeds() {
    let tmp = fresh_dir("match");

    let a = run_in(&tmp, "a.nsl", &["--source-ad"], &save_cfg());
    assert!(a.ok, "save run failed:\n{}", a.stderr);

    // The record must EXIST and carry the mode this run compiled with. An
    // absent record would make the mismatch guard vacuous.
    let side = sidecar_text(&tmp);
    assert!(
        side.contains(r#""exec":"#),
        "the sidecar carries no 'exec' record — the mismatch guard below \
         would then pass by skipping, not by agreeing:\n{side}"
    );
    assert!(
        side.contains("ad=source"),
        "the record must name the AD mode this run used:\n{side}"
    );

    let b = run_in(&tmp, "b.nsl", &["--source-ad"], &load_cfg());
    assert!(b.ok, "matching resume must succeed:\n{}", b.stderr);
    assert!(
        !b.stderr.contains("ARITHMETIC differs"),
        "an identical flag set must not report a difference:\n{}",
        b.stderr
    );
    assert!(
        !b.stderr.contains("compile-flag check is SKIPPED"),
        "both sides carry a record, so the check must actually RUN:\n{}",
        b.stderr
    );
}

#[test]
fn dropping_source_ad_on_resume_is_refused() {
    let tmp = fresh_dir("mismatch");

    let a = run_in(&tmp, "a.nsl", &["--source-ad"], &save_cfg());
    assert!(a.ok, "save run failed:\n{}", a.stderr);

    // Same recipe, same seed, same (absent) loader — the ONLY thing that
    // changed is the AD mode, which is the point.
    let b = run_in(&tmp, "b.nsl", &[], &load_cfg());
    assert!(
        !b.ok,
        "resuming a source-AD checkpoint under the tape must be refused:\n\
         stdout:\n{}\nstderr:\n{}",
        b.stdout, b.stderr
    );
    assert!(
        b.stderr.contains("ARITHMETIC differs"),
        "the refusal must be the FINGERPRINT refusal, not some other abort \
         that happens to fail the run:\n{}",
        b.stderr
    );
    assert!(
        b.stderr.contains("ad: checkpoint source -> this run tape"),
        "the message must name the offending key and both values, or an \
         operator cannot act on it:\n{}",
        b.stderr
    );
    assert!(
        !b.stdout.contains("FIXTURE_DONE"),
        "the refusal must land BEFORE training resumes:\n{}",
        b.stdout
    );
}

#[test]
fn toggling_placement_flags_warns_but_resumes() {
    let tmp = fresh_dir("placement");

    let a = run_in(&tmp, "a.nsl", &["--source-ad"], &save_cfg());
    assert!(a.ok, "save run failed:\n{}", a.stderr);

    let b = run_in(
        &tmp,
        "b.nsl",
        &["--source-ad", "--optim-state-offload"],
        &load_cfg(),
    );
    assert!(
        b.ok,
        "--optim-state-offload is value-neutral and the production 1B recipe \
         resumes with it toggled; refusing it would break a shipped \
         workflow:\n{}",
        b.stderr
    );
    assert!(
        b.stderr.contains("different memory placement"),
        "a legitimate placement change must still be REPORTED — silently \
         resuming under a different memory shape is what this record is for:\n{}",
        b.stderr
    );
    assert!(
        b.stderr.contains("offload: checkpoint 0 -> this run 1"),
        "the placement warning must name the key that changed:\n{}",
        b.stderr
    );
}
