//! Resume must refuse a RUNTIME ENVIRONMENT whose behavior-tier `NSL_*`
//! variables differ from the checkpoint's (roadmap A5, increment 3).
//!
//! `exec_fingerprint_resume_gate.rs` guards the compile flags; this guards
//! the words in the LAUNCHING shell. `NSL_SUM_SQ_CPU=1`, `NSL_FLASH_BWD_CPU=1`
//! and their siblings route a kernel to a different implementation with a
//! different reduction order, and a resume under a different set of them
//! continued theta and the moments under different arithmetic and said
//! nothing.
//!
//! Same three-guard shape as the fingerprint gate, because each alone passes
//! with the feature broken:
//!
//! 1. the record is WRITTEN — an `env` key exists in the sidecar even when
//!    nothing is set (that is how the loader tells "nothing set" from
//!    "predates the record"), and a set variable appears in it;
//! 2. the refusal FIRES and names the variable and both values;
//! 3. `NSL_RESUME_ALLOW_ENV_DRIFT=1` converts the refusal into an
//!    acknowledgment and the run continues.
//!
//! The variable used is `NSL_SUM_SQ_CPU`: behavior tier, runtime-read, and a
//! no-op on the CPU path this test runs on — so the ONLY thing that differs
//! between the two runs is the record, which is the point.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

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

/// `nsl run` with every inherited `NSL_*` stripped, then `env` applied, so
/// the calling shell's environment cannot leak into the record under test.
fn run_in(dir: &std::path::Path, name: &str, env: &[(&str, &str)], src: &str) -> RunOut {
    let root = repo_root();
    let prog = dir.join(name);
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    for (k, _) in std::env::vars_os().map(|(k, v)| (k, v)) {
        if k.to_string_lossy().starts_with("NSL_") {
            cmd.env_remove(k);
        }
    }
    let out = cmd
        .arg("run")
        .arg("--source-ad")
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

fn fresh_dir(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("nsl_envgate_{}_{tag}", std::process::id()));
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
fn the_record_is_written_and_a_matching_environment_resumes() {
    let tmp = fresh_dir("match");

    let a = run_in(&tmp, "a.nsl", &[], &save_cfg());
    assert!(a.ok, "save run failed:\n{}", a.stderr);
    let side = sidecar_text(&tmp);
    // The KEY must exist whether or not anything is set: its absence is what
    // the loader reads as "predates the record", which SKIPS the check — an
    // omitted-when-empty key would make guard 2 pass by skipping. (`nsl run`
    // itself exports `NSL_COLLECTIVES` to the program — `commands/run.rs` —
    // so the record is never literally empty under the CLI; what this run
    // must NOT carry is the variable the next run sets.)
    assert!(
        side.contains(r#""env":""#) && !side.contains("NSL_SUM_SQ_CPU"),
        "the sidecar must carry an 'env' record without NSL_SUM_SQ_CPU when \
         that variable is not set:\n{side}"
    );

    let b = run_in(&tmp, "b.nsl", &[], &load_cfg());
    assert!(b.ok, "matching resume must succeed:\n{}", b.stderr);
    assert!(
        !b.stderr.contains("ENVIRONMENT differs") && !b.stderr.contains("behavior check is SKIPPED"),
        "an identical environment must neither refuse nor skip:\n{}",
        b.stderr
    );

    // And a SET variable reaches the record by name and value.
    let tmp2 = fresh_dir("recorded");
    let c = run_in(&tmp2, "a.nsl", &[("NSL_SUM_SQ_CPU", "1")], &save_cfg());
    assert!(c.ok, "save run failed:\n{}", c.stderr);
    let side2 = sidecar_text(&tmp2);
    let env_field = side2
        .split(r#""env":""#)
        .nth(1)
        .and_then(|rest| rest.split('"').next())
        .unwrap_or_default();
    assert!(
        env_field.split(',').any(|kv| kv == "NSL_SUM_SQ_CPU=1"),
        "a set behavior-tier variable must be recorded as its own `NAME=value` \
         field:\n{side2}"
    );
}

#[test]
fn a_behavior_variable_set_on_resume_only_is_refused() {
    let tmp = fresh_dir("mismatch");

    let a = run_in(&tmp, "a.nsl", &[], &save_cfg());
    assert!(a.ok, "save run failed:\n{}", a.stderr);

    let b = run_in(&tmp, "b.nsl", &[("NSL_SUM_SQ_CPU", "1")], &load_cfg());
    assert!(
        !b.ok,
        "resuming under a behavior-tier variable the save did not have must \
         be refused:\nstdout:\n{}\nstderr:\n{}",
        b.stdout, b.stderr
    );
    assert!(
        b.stderr.contains("ENVIRONMENT differs"),
        "the refusal must be the ENVIRONMENT refusal, not some other abort:\n{}",
        b.stderr
    );
    assert!(
        b.stderr.contains("NSL_SUM_SQ_CPU: checkpoint <absent> -> this run 1"),
        "the message must name the variable and both sides:\n{}",
        b.stderr
    );
    assert!(
        !b.stdout.contains("FIXTURE_DONE"),
        "the refusal must land BEFORE training resumes:\n{}",
        b.stdout
    );

    // The other direction — set at save, dropped at resume — is the same
    // difference, not a silently-accepted "back to defaults".
    let tmp2 = fresh_dir("dropped");
    let c = run_in(&tmp2, "a.nsl", &[("NSL_SUM_SQ_CPU", "1")], &save_cfg());
    assert!(c.ok, "save run failed:\n{}", c.stderr);
    let d = run_in(&tmp2, "b.nsl", &[], &load_cfg());
    assert!(!d.ok && d.stderr.contains("NSL_SUM_SQ_CPU: checkpoint 1 -> this run <absent>"), "{}", d.stderr);
}

#[test]
fn the_drift_can_be_acknowledged_and_a_diagnostic_variable_is_not_drift() {
    let tmp = fresh_dir("ack");

    let a = run_in(&tmp, "a.nsl", &[], &save_cfg());
    assert!(a.ok, "save run failed:\n{}", a.stderr);

    let b = run_in(
        &tmp,
        "b.nsl",
        &[("NSL_SUM_SQ_CPU", "1"), ("NSL_RESUME_ALLOW_ENV_DRIFT", "1")],
        &load_cfg(),
    );
    assert!(b.ok, "an acknowledged drift must resume:\n{}", b.stderr);
    assert!(
        b.stderr.contains("ENVIRONMENT drift acknowledged") && b.stderr.contains("NSL_SUM_SQ_CPU"),
        "the acknowledgment must be printed and name the variable:\n{}",
        b.stderr
    );
    assert!(b.stdout.contains("FIXTURE_DONE"), "{}", b.stdout);

    // A diagnostic-tier variable is not part of the record: setting one on
    // resume neither refuses nor acknowledges. This pins the tier boundary
    // — the guard must not grow into "any NSL_* difference refuses".
    let tmp2 = fresh_dir("diag");
    let c = run_in(&tmp2, "a.nsl", &[], &save_cfg());
    assert!(c.ok, "save run failed:\n{}", c.stderr);
    let d = run_in(&tmp2, "b.nsl", &[("NSL_MEMSTATS", "1")], &load_cfg());
    assert!(d.ok, "a diagnostic-tier variable must not refuse a resume:\n{}", d.stderr);
    assert!(!d.stderr.contains("ENVIRONMENT"), "{}", d.stderr);
}
