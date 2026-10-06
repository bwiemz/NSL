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

/// A residual-block model WGGO can prune (`blocks.N`, two residual Adds per
/// block), for the layer-prune resume gate below. AdamW because checkpoints
/// carry only Adam-family moments; this gate is about the record, not numerics.
fn prunable_fixture(train_cfg: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Blk:
    wa: Tensor = ones([4, 4]) * 0.05
    wb: Tensor = ones([4, 4]) * 0.05

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + (x @ self.wa)
        return h + (h @ self.wb)

model Net:
    blocks: [Blk; 3] = Blk()

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
let x = ones([2, 4]) * 0.1
let y = zeros([2, 4])

train(model = m{train_cfg}):
    optimizer: AdamW(lr = 0.01)
    step(batch):
        let loss = mse_loss(m.forward(x), y)

print("FIXTURE_DONE")
"#
    )
}

/// A WGGO layer prune deletes blocks from the model a step computes, so a
/// resume across a prune change is refused, naming `prune_layers`; the same
/// prune resumes. (#807 left the prune out of the record.)
#[test]
fn changing_the_layer_prune_on_resume_is_refused() {
    let tmp = fresh_dir("prune");
    let pruned: &[&str] = &["--source-ad", "--wggo", "greedy", "--wggo-prune-layers", "blocks.1"];
    let save = prunable_fixture(r#", epochs = 1, checkpoint_save = "ck.nslm", checkpoint_every = 1"#);
    let load = prunable_fixture(r#", epochs = 2, checkpoint_load = "ck.nslm""#);

    let a = run_in(&tmp, "a.nsl", pruned, &save);
    assert!(a.ok, "pruned save run failed:\n{}", a.stderr);
    assert!(
        sidecar_text(&tmp).contains("prune_layers=blocks.1"),
        "the record must carry the prune:\n{}",
        sidecar_text(&tmp)
    );

    let same = run_in(&tmp, "b.nsl", pruned, &load);
    assert!(same.ok, "resuming with the same prune must succeed:\n{}", same.stderr);
    assert!(!same.stderr.contains("ARITHMETIC differs"), "{}", same.stderr);

    let dropped = run_in(&tmp, "c.nsl", &["--source-ad", "--wggo", "greedy"], &load);
    assert!(
        !dropped.ok,
        "resuming a pruned checkpoint unpruned must be refused:\n{}",
        dropped.stderr
    );
    assert!(
        dropped.stderr.contains("prune_layers: checkpoint blocks.1 -> this run <absent>"),
        "the refusal must name the prune:\n{}",
        dropped.stderr
    );
    assert!(!dropped.stdout.contains("FIXTURE_DONE"), "the refusal must land before training resumes");
}


fn run_in_env(
    dir: &std::path::Path,
    name: &str,
    flags: &[&str],
    src: &str,
    env: &[(&str, &str)],
) -> RunOut {
    let root = repo_root();
    let prog = dir.join(name);
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("run")
        .args(flags)
        .arg(&prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .envs(env.iter().copied())
        .output()
        .expect("spawn nsl run");
    RunOut {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).to_string(),
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

/// External review 2026-10-06: `--matmul-mode tf32` EQUALS the default, and
/// the old "still equals the default" rule could not tell it from an omitted
/// flag -- an inherited `NSL_MATMUL_BF16=1` replaced it, and the checkpoint
/// recorded (as the run used) bf16. The anti-vacuity half: without the flag
/// the variable still applies and is recorded.
#[test]
fn an_explicit_default_matmul_mode_beats_an_inherited_variable() {
    let bf16_env = [("NSL_MATMUL_BF16", "1")];
    let dir = fresh_dir("explicit_tf32");
    let r = run_in_env(&dir, "s.nsl", &["--matmul-mode", "tf32"], &save_cfg(), &bf16_env);
    assert!(r.ok && r.stdout.contains("FIXTURE_DONE"), "save run failed:\n{}", r.stderr);
    let sidecar = sidecar_text(&dir);
    assert!(sidecar.contains("mm=tf32"), "the explicit tf32 must be what ran and what is recorded: {sidecar}");

    let dir2 = fresh_dir("env_bf16");
    let r = run_in_env(&dir2, "s.nsl", &[], &save_cfg(), &bf16_env);
    assert!(r.ok, "save run failed:\n{}", r.stderr);
    let sidecar = sidecar_text(&dir2);
    assert!(sidecar.contains("mm=bf16"), "without the flag the variable still applies: {sidecar}");

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

/// The runtime half, in a CUDA build: an explicit `--matmul-mode tf32` also
/// beats an inherited `NSL_MATMUL_TF32=0` at run time (it used to run f32
/// cores and fingerprint tf32), and says so. Without the flag the variable
/// still applies -- and the checkpoint now records the f32 the runtime ran,
/// not the compiled tf32.
#[test]
#[ignore = "requires CUDA GPU (a cuda-feature build of nsl; resolves the cuBLAS math mode)"]
fn an_explicit_mode_beats_the_runtime_tf32_variable() {
    let off = [("NSL_MATMUL_TF32", "0")];
    let dir = fresh_dir("rt_explicit_tf32");
    let r = run_in_env(&dir, "s.nsl", &["--matmul-mode", "tf32"], &save_cfg(), &off);
    assert!(r.ok, "save run failed:\n{}", r.stderr);
    assert!(sidecar_text(&dir).contains("mm=tf32"), "{}", sidecar_text(&dir));
    assert!(r.stderr.contains("NSL_MATMUL_TF32=0 is set but ignored"), "{}", r.stderr);

    let dir2 = fresh_dir("rt_env_f32");
    let r = run_in_env(&dir2, "s.nsl", &[], &save_cfg(), &off);
    assert!(r.ok, "save run failed:\n{}", r.stderr);
    assert!(sidecar_text(&dir2).contains("mm=f32"), "the record names what ran: {}", sidecar_text(&dir2));

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}
