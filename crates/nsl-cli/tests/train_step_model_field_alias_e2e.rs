//! e2e regression: binding a model field to a train-step local
//! (`let alias = m.w`) must not free the weight.
//!
//! Model-field member access hands out the model's own tensor handle (a
//! borrow, no retain). The train-step end-of-step cleanup used to
//! `nsl_tensor_free` every step-local tensor variable — including such
//! aliases — which freed the weight itself; the next step's forward then
//! died with "NslTensor::from_ptr: bad magic 0x0000DEAD" (use-after-free).
//! The alias is now classified into `non_owning_symbols` (same discipline as
//! parameter aliases) and skipped by the cleanup.
//!
//! This matters in particular for `@inspect(w, ...)`-style weight probing,
//! which is documented usage and requires exactly this binding shape.

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

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.0001)
    step(batch):
        let alias = m.w
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("alias-ok")
"#;

#[test]
fn e2e_model_field_alias_survives_step_cleanup() {
    let root = workspace_root();
    let dir = std::env::temp_dir().join(format!("nsl_alias_e2e_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, FIXTURE).expect("write fixture");

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run"])
        .arg(&file)
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("failed to execute nsl run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(
        output.status.success() && stdout.contains("alias-ok"),
        "model-field alias in train step crashed or failed (exit {:?}) — \
         a 'bad magic' panic here means step cleanup freed the weight.\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(
        !stderr.contains("bad magic"),
        "use-after-free detected:\n{}",
        stderr
    );
}
