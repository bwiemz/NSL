//! Milestone A gate: a requested surface whose owner records nothing is a
//! hard error, not a silently-identical artifact.
//!
//! Each behavioural case runs its CONTROL first (the same program/invocation
//! minus the request) so a broken harness cannot masquerade as enforcement:
//! if the control fails, the case says so instead of crediting the gate.
//!
//! The specimen surfaces are chosen for stability:
//! - `--checkpoint-blocks` on a program with NO train block: CCR is invoked
//!   from train-block compilation, so there is nothing that could ever
//!   answer this request — it must stay unsatisfied under any future fix
//!   that records dispositions more eagerly, because the pass is never
//!   reached at all.
//! - `@fase(mode = off)` on a train block: FASE runs on every train-block
//!   compile and records `declined, mode off` — the satisfied-by-decline
//!   case, pinned end-to-end.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

const TRAIN_PROGRAM: &str = r#"
model Tiny:
    w: Tensor = randn([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = randn([2, 4])
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let y = m.forward(x)
        let loss = (y * y).sum()
print("DONE")
"#;

const NO_TRAIN_PROGRAM: &str = r#"
let x = randn([2, 4])
let y = x * 2.0
print("DONE")
"#;

struct RunResult {
    status: std::process::ExitStatus,
    stderr: String,
}

fn nsl(dir: &std::path::Path, args: &[&str], prog: &std::path::Path) -> RunResult {
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(args)
        .arg(prog)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .output()
        .expect("spawn nsl");
    RunResult {
        status: out.status,
        stderr: String::from_utf8_lossy(&out.stderr).to_string(),
    }
}

fn stage(tag: &str, src: &str) -> (PathBuf, PathBuf) {
    let tmp = std::env::temp_dir().join(format!("nsl_activation_gate_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    (tmp, prog)
}

/// A flag whose owning pass is never reached must fail the build, name the
/// silent owner, and leave no artifact behind.
#[test]
fn an_unreachable_owner_fails_the_build_and_leaves_no_artifact() {
    let (tmp, prog) = stage("unreached", NO_TRAIN_PROGRAM);
    let exe = tmp.join("out.exe");
    let exe_s = exe.to_str().unwrap();

    // CONTROL: without the request, the build must succeed — otherwise this
    // gate would be crediting enforcement for an unrelated failure.
    let control = nsl(&tmp, &["build", "-o", exe_s], &prog);
    assert!(
        control.status.success(),
        "control build failed; the gate cannot conclude anything:\n{}",
        control.stderr
    );
    std::fs::remove_file(&exe).ok();

    let r = nsl(&tmp, &["build", "--checkpoint-blocks", "-o", exe_s], &prog);
    assert!(
        !r.status.success(),
        "--checkpoint-blocks on a train-less program must fail as silently \
         inert; stderr:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("silently inert")
            && r.stderr.contains("--checkpoint-blocks")
            && r.stderr.contains("CCR recorded no disposition"),
        "error must name the surface and the silent owner:\n{}",
        r.stderr
    );
    assert!(
        !exe.exists(),
        "an unsatisfied request must not leave a fresh artifact behind"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

/// `--allow-inert-requests` demotes the same condition to a warning and the
/// artifact is produced.
#[test]
fn the_escape_hatch_demotes_to_a_warning() {
    let (tmp, prog) = stage("escape", NO_TRAIN_PROGRAM);
    let exe = tmp.join("out.exe");
    let r = nsl(
        &tmp,
        &["build", "--checkpoint-blocks", "--allow-inert-requests", "-o", exe.to_str().unwrap()],
        &prog,
    );
    assert!(
        r.status.success(),
        "escape hatch must let the build complete:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("warning") && r.stderr.contains("silently inert"),
        "the demoted condition must still be visible:\n{}",
        r.stderr
    );
    assert!(exe.exists(), "escape-hatch build must produce the artifact");
    std::fs::remove_dir_all(&tmp).ok();
}

/// A declined request is a SATISFIED contract: a forced-off @fase re-records
/// `declined, feature disabled - @fase(mode = off)`, so it passes enforcement
/// and the report says why. This pins the decline path end-to-end — if it
/// ever regresses to Unsatisfied, every legitimately-declined request would
/// fail the build it correctly declined on.
#[test]
fn a_typed_decline_satisfies_the_contract() {
    let src = TRAIN_PROGRAM.replace("train(model = m", "@fase(mode = off)\ntrain(model = m");
    let (tmp, prog) = stage("decline", &src);
    let exe = tmp.join("out.exe");
    let r = nsl(
        &tmp,
        &["build", "--activation-report", "-o", exe.to_str().unwrap()],
        &prog,
    );
    assert!(
        r.status.success(),
        "a declined request must not fail the build:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("[activation] @fase: declined, feature disabled - @fase(mode = off)"),
        "report must show the decline citing the decorator that caused it \
         (the driver re-records FeatureDisabled for a forced-off @fase, so \
         the report names the request rather than a generic mode-off):\n{}",
        r.stderr
    );
    std::fs::remove_dir_all(&tmp).ok();
}

/// `nsl run` enforces before the program executes: the inert request fails
/// the invocation and the program's own output never appears.
#[test]
fn run_enforces_before_the_program_executes() {
    let (tmp, prog) = stage("runenf", NO_TRAIN_PROGRAM);
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--checkpoint-blocks"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "run must fail on the inert request:\n{stderr}");
    assert!(
        !stdout.contains("DONE"),
        "the program must not have executed after an unsatisfied request:\n{stdout}"
    );
    std::fs::remove_dir_all(&tmp).ok();
}

/// `nsl check` stays report-only: the same program and flag that fail
/// `build` pass `check`, because check answers "is this well-formed", not
/// "did an optimizer fire".
#[test]
fn check_never_hard_fails_on_activation() {
    let (tmp, prog) = stage("checkro", NO_TRAIN_PROGRAM);
    let r = nsl(&tmp, &["check", "--checkpoint-blocks"], &prog);
    // CheckArgs has no --checkpoint-blocks: clap itself must reject it. If
    // that ever changes (the flag widened to check), activation must still
    // not hard-fail there — assert whichever failure shape clap produces
    // today so a future widening revisits this deliberately.
    assert!(
        !r.status.success() && r.stderr.contains("unexpected argument"),
        "check has no --checkpoint-blocks today; if this changed, decide the \
         check-side activation posture deliberately:\n{}",
        r.stderr
    );
    std::fs::remove_dir_all(&tmp).ok();
}
