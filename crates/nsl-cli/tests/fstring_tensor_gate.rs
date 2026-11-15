//! A tensor interpolated into an f-string (or converted with `str`, or
//! concatenated to a string) reads as `print` shows it.
//!
//! The conversion treated every type it did not name as an integer, so a
//! tensor was interpolated as its handle: `f"loss {loss}"` in an `on_step`
//! callback printed `loss 94297074379216` where `print(loss)` printed `3.0`.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

/// `x` is three rows of ones and `w` two halves, so the step-1 loss
/// `sum(x @ w)` is 3 x (0.5 + 0.5) = 3.
#[test]
fn a_tensor_in_a_string_reads_as_print_shows_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prog.nsl");
    std::fs::write(
        &path,
        r#"let t = arange(1.0, 5.0).reshape([2, 2])
print(f"t = {t}")
print(f"sum = {sum(t)}")
print("as str: " + str(t))
let a = 3
let b = 1.5
print(f"{a} {b} {true}")
let x = ones([3, 2])
model M:
    w: Tensor = full([2, 1], 0.5)
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
let m = M()
train(model=m, epochs=1):
    optimizer: SGD(lr=0.1)
    step(batch):
        let loss = sum(m.forward(x))
    callbacks:
        on_step(step, loss):
            print(f"step {step} | loss {loss}")
"#,
    )
    .unwrap();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", workspace_root().join("stdlib"))
        .current_dir(dir.path())
        .arg("run")
        .arg(&path)
        .args(["--target", "cpu"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines,
        [
            "t = tensor([[1.0, 2.0], [3.0, 4.0]])",
            "sum = 10.0",
            "as str: tensor([[1.0, 2.0], [3.0, 4.0]])",
            "3 1.5 true",
            "step 1 | loss 3.0",
        ],
        "stderr:\n{stderr}"
    );
}
