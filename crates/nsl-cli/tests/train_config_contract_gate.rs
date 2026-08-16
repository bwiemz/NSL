//! The Training Configuration Contract, end to end: the train(...) key
//! namespace is CLOSED, and the refusal fires at `nsl check` time AND on
//! the run/build path (loader emits the same semantic diagnostics and
//! refuses before codegen).
//!
//! The per-rule matrix (duplicates, positional args, epochs/grad_accumulation/
//! grad_clip ranges and literal-ness, checkpoint pairing, missing model)
//! lives in nsl-semantic's checker tests — this gate pins the CLI surface:
//! a typo'd key that used to silently train with defaults now names itself
//! on stderr, and a fully valid header still compiles and runs.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture(header: &str) -> String {
    format!(
        r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = full([2, 2], 2.0)
let y = zeros([2, 2])
{header}
    optimizer: AdamW(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("TRAIN_CONTRACT_DONE")
"#
    )
}

fn run_cmd(tag: &str, src: &str, args: &[&str]) -> (bool, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!(
        "nsl_traincfg_{}_{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl");
    let res = (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    );
    let _ = std::fs::remove_dir_all(&tmp);
    res
}

#[test]
fn a_typo_key_refuses_at_check_time_with_the_key_named() {
    // The motivating case: `epochz=8` used to pass `nsl check` clean and
    // lower with the epochs DEFAULT — a silently different training run.
    let src = fixture("train(model = m, epochz = 8):");
    let (ok, _stdout, stderr) = run_cmd("typo_check", &src, &["check"]);
    assert!(!ok, "epochz=8 must fail nsl check:\n{stderr}");
    assert!(
        stderr.contains("unknown train config key 'epochz'"),
        "the refusal must name the key:\n{stderr}"
    );
    assert!(
        stderr.contains("expected model=, epochs="),
        "the refusal must list the closed set:\n{stderr}"
    );
}

#[test]
fn a_typo_key_refuses_on_the_run_path_too() {
    // The loader emits the same semantic diagnostics and refuses before
    // codegen — no path runs a typo'd header.
    let src = fixture("train(model = m, epochz = 8):");
    let (ok, _stdout, stderr) = run_cmd("typo_run", &src, &["run", "--source-ad"]);
    assert!(!ok, "epochz=8 must fail nsl run:\n{stderr}");
    assert!(
        stderr.contains("unknown train config key 'epochz'"),
        "the run path must carry the same refusal:\n{stderr}"
    );
}

#[test]
fn a_duplicate_key_refuses() {
    // Last-wins was the old silent behavior.
    let src = fixture("train(model = m, epochs = 2, epochs = 3):");
    let (ok, _stdout, stderr) = run_cmd("dup", &src, &["check"]);
    assert!(!ok, "duplicate epochs must fail:\n{stderr}");
    assert!(
        stderr.contains("duplicate train config key 'epochs'"),
        "expected the duplicate refusal:\n{stderr}"
    );
}

#[test]
fn a_valid_header_still_compiles_and_runs() {
    // Anti-overbreadth control: the contract must not refuse the language
    // it exists to protect.
    let src = fixture("train(model = m, epochs = 2, grad_accumulation = 2, grad_clip = 1.0):");
    let (ok, stdout, stderr) = run_cmd("valid", &src, &["run", "--source-ad"]);
    assert!(ok, "valid header failed:\n{stderr}");
    assert!(
        stdout.contains("TRAIN_CONTRACT_DONE"),
        "fixture did not finish:\n{stdout}"
    );
    assert!(
        !stderr.contains("train config"),
        "no contract diagnostics may fire on a valid header:\n{stderr}"
    );
}
