//! e2e regression: `nsl run --monitor` on a training program must compile.
//!
//! The health-monitor weight-norm hook computed its `is_init` flag as
//! `uextend(I8, icmp(..))` — written for the old Cranelift `b1` comparison
//! type. Modern Cranelift's `icmp` already yields `I8`, so the same-width
//! `uextend` failed function verification and every `nsl run --monitor` on a
//! train program died with "failed to define main: ... Verifier errors"
//! (with the detail swallowed by Display formatting, also fixed).

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

const FIXTURE: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 2):
    optimizer: SGD(lr = 0.0001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("monitor-train-ok")
"#;

#[test]
fn e2e_monitor_on_train_program_compiles_and_runs() {
    let root = workspace_root();
    let dir = std::env::temp_dir().join(format!("nsl_monitor_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, FIXTURE).expect("write fixture");

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run", "--monitor"])
        .arg(&file)
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("failed to execute nsl run --monitor");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        output.status.success() && stdout.contains("monitor-train-ok"),
        "--monitor on a train program failed (exit {:?}); a 'Verifier errors' \
         message here means the health-hook IR regressed.\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(
        !stderr.contains("Verifier"),
        "Cranelift verification failed under --monitor:\n{}",
        stderr
    );
}
