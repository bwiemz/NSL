//! e2e regression for train-block section handling.
//!
//! The section-processing loops (standard + pipelined) ended in a bare
//! `_ => {}` that silently dropped three parsed-and-type-checked variants:
//! bare `Stmt`s never executed, and `eval:` / `distribute:` sections —
//! including the spec's documented best-checkpoint-saving eval examples —
//! silently never ran. Now: bare statements compile (once, pre-training, like
//! Data-section statements) and `eval:`/`distribute:` refuse loudly
//! (deferral-must-refuse) until eval-loader support exists.

use std::process::Command;

fn workspace_root() -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const COMMON: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])
"#;

fn run_fixture(body: &str, tag: &str) -> (Option<i32>, String, String) {
    let root = workspace_root();
    let dir = std::env::temp_dir().join(format!("nsl_train_sections_{}_{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, format!("{COMMON}{body}")).expect("write fixture");

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"])
        .arg(&file)
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("failed to execute nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

#[test]
fn e2e_train_bare_statement_executes() {
    let (code, stdout, stderr) = run_fixture(
        r#"
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0001)
    print("train-bare-stmt-ran")
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("train-done")
"#,
        "stmt",
    );
    assert_eq!(code, Some(0), "run failed.\nstdout:{stdout}\nstderr:{stderr}");
    assert!(
        stdout.contains("train-bare-stmt-ran"),
        "bare train-block statement was dropped (must execute pre-training).\nstdout:{stdout}"
    );
}

#[test]
fn e2e_train_eval_section_refuses_loudly() {
    let (code, stdout, stderr) = run_fixture(
        r#"
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    eval(batch):
        let ev = m.forward(x)

print("should_not_reach")
"#,
        "eval",
    );
    assert_ne!(code, Some(0), "eval: section must be refused, not silently skipped");
    assert!(
        !stdout.contains("should_not_reach"),
        "program ran despite eval refusal.\nstdout:{stdout}"
    );
    assert!(
        stderr.contains("`eval:` sections are not yet executed"),
        "refusal must name the eval section and suggest on_epoch.\nstderr:{stderr}"
    );
}

// Regression for the on_epoch/on_epoch_end `loss` binding: that param
// aliases the same tensor pointer as the training loop's internal
// epoch_loss_var, which the loop frees itself right after the callback
// runs (see nsl-codegen/src/stmt.rs, the on_epoch/on_epoch_end callback
// block). Nothing under this repo's existing tests previously compiled
// an on_epoch callback that binds `loss` at all, so a codegen mistake in
// how that alias is registered (double free, or freeing the model's live
// loss out from under a later epoch) would compile clean and only show up
// as GPU-memory corruption in a real training run. Running across several
// epochs and asserting each one printed exactly once is enough to catch a
// crash or a corrupted/skipped read of that aliased value.
#[test]
fn e2e_train_on_epoch_binds_loss() {
    let (code, stdout, stderr) = run_fixture(
        r#"
train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.0001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_epoch(epoch, loss):
            print("epoch-done")

print("train-done")
"#,
        "on_epoch_loss",
    );
    assert_eq!(code, Some(0), "run failed.\nstdout:{stdout}\nstderr:{stderr}");
    assert_eq!(
        stdout.matches("epoch-done").count(),
        3,
        "on_epoch(epoch, loss) must fire exactly once per epoch with no crash \
         or corruption from the loss alias.\nstdout:{stdout}\nstderr:{stderr}"
    );
    assert!(
        stdout.contains("train-done"),
        "training loop did not complete cleanly.\nstdout:{stdout}\nstderr:{stderr}"
    );
}
