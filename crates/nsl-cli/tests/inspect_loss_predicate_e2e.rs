//! e2e regression for the `loss` identifier in `@inspect(condition="...")`
//! predicates.
//!
//! `loss` used to lower to a compile-time `f64const(0.0)` — so
//! `condition="loss > x"` silently never fired and `loss < x` always fired.
//! It now reads the most recent recorded loss at runtime via
//! `nsl_health_get_last_loss` (recorded per step whenever `--inspect` or the
//! health monitor is active). Because `@inspect` fires at the let-binding site
//! (before the current step's loss compute), the predicate sees the previous
//! completed step's loss: with 3 epochs and a constant loss of 4.0, a
//! `loss > 0.5` condition fires on steps 1 and 2 → exactly 2 dump files.

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

const FIXTURE_TEMPLATE: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.0001)
    step(batch):
        @inspect(pred, condition="__COND__")
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("inspect-loss-run-done")
"#;

/// Run the fixture with the given predicate in a fresh temp dir; return the
/// number of dump files written to `.nsl-inspect/`.
fn run_with_condition(cond: &str) -> usize {
    let root = workspace_root();
    let dir = std::env::temp_dir().join(format!(
        "nsl_inspect_loss_e2e_{}_{}",
        std::process::id(),
        cond.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let src = FIXTURE_TEMPLATE.replace("__COND__", cond);
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, src).expect("write fixture");

    let manifest = root.join("Cargo.toml");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(&manifest)
        .args(["-p", "nsl-cli", "--", "run", "--inspect"])
        .arg(&file)
        // .nsl-inspect/ is created relative to the process CWD, so run from
        // the temp dir while cargo resolves the workspace via manifest-path
        // and the stdlib resolves via NSL_STDLIB_PATH (CWD-relative otherwise).
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("failed to execute nsl run --inspect");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("inspect-loss-run-done"),
        "run failed for condition {:?} (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
        cond,
        output.status.code(),
        stdout,
        stderr
    );

    let count = std::fs::read_dir(dir.join(".nsl-inspect"))
        .map(|rd| rd.count())
        .unwrap_or(0);
    let _ = std::fs::remove_dir_all(&dir);
    count
}

#[test]
fn e2e_inspect_loss_predicate_reads_real_loss() {
    // loss is constantly 4.0; predicate sees the previous step's loss, so it
    // fires on steps 1 and 2 of 3. Zero dumps here is the old placeholder bug.
    let fired = run_with_condition("loss > 0.5");
    assert_eq!(
        fired, 2,
        "loss > 0.5 must fire on steps 1 and 2 (loss==4.0); \
         0 means `loss` regressed to a compile-time constant"
    );

    // Negative control: an unsatisfiable threshold must never fire — always
    // firing would mean `loss` degenerated some other way.
    let fired = run_with_condition("loss > 99999.0");
    assert_eq!(fired, 0, "loss > 99999 must never fire (loss==4.0)");
}
