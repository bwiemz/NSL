//! e2e regression: a tuple that passes a BORROWED tensor through a function,
//! destructured inside a train step.
//!
//! Tuple construction must RETAIN borrowed elements (bare variable/param
//! reads). With a plain ownership-consume, `return (x, x * 2.0)` left the
//! caller holding two owned aliases of a refcount-1 tensor; the train-step
//! end-of-step sweep frees step-locals with the unguarded nsl_tensor_free, so
//! the second free was a double-free (found by adversarial review of the
//! tuple-ownership fix — the plain-function sweep uses the magic-guarded
//! free_if_valid and only leaked, which is why the non-train fixture alone
//! missed it).

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

fn pass_through(x: Tensor) -> (Tensor, Tensor):
    return (x, x * 2.0)

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.0001)
    step(batch):
        let probe = ones([2, 1])
        let (orig, doubled) = pass_through(probe)
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

print("tuple-train-ok")
"#;

#[test]
fn e2e_tuple_passthrough_in_train_step_no_double_free() {
    let root = workspace_root();
    let dir = std::env::temp_dir().join(format!("nsl_tuple_train_{}", std::process::id()));
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
        output.status.success() && stdout.contains("tuple-train-ok"),
        "tuple pass-through in a train step failed (exit {:?}) — a 'bad magic' \
         or access-violation here means the borrowed tuple element was not \
         retained (double-free).\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(
        !stderr.contains("bad magic"),
        "double-free detected:\n{}",
        stderr
    );
}
