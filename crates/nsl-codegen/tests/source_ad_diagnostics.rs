//! Diagnostic coverage tests for source-AD fallback paths.

use std::process::Command;
use tempfile::TempDir;

#[cfg(feature = "cuda")]
#[test]
fn source_ad_warns_on_unrecognized_ffi_callee() {
    // copy_data is a runtime intrinsic that source-AD's Wengert extractor
    // does NOT have a match arm for (see B.2.1 close-out comment in
    // crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs:28 which
    // cites this exact pattern as hitting the "unsupported function" path).
    // Invoking it inside a train step triggers the fallback and, post-fix,
    // emits the new warning naming the unrecognized callee.
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("probe.nsl");
    std::fs::write(
        &src_path,
        r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Toy()
m.to(cuda)
let x = zeros([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
let seed = ones([8, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        copy_data(m.w, seed)
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#,
    )
    .unwrap();

    let root: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    let workspace_root = root.parent().unwrap().parent().unwrap();
    let stdlib = workspace_root.join("stdlib");

    let out = Command::new(nsl_bin())
        .env("NSL_STDLIB_PATH", &stdlib)
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[source-ad] warning: unrecognized FFI callee"),
        "expected source-AD warning about unrecognized FFI callee 'copy_data'.\nstderr:\n{stderr}",
    );
    assert!(
        stderr.contains("copy_data"),
        "warning should name the specific unrecognized FFI.\nstderr:\n{stderr}",
    );
}
/// Path to the `nsl` binary built by `cargo test --workspace`.
///
/// `nsl` lives in the sibling `nsl-cli` crate, so Cargo does not set
/// `CARGO_BIN_EXE_nsl` for this crate's integration tests, and assert_cmd 2.2+
/// no longer falls back to the target directory. Resolve it next to the running
/// test executable instead: `target/<profile>/deps/<test>` -> `target/<profile>/nsl`.
fn nsl_bin() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("locate test executable");
    dir.pop(); // drop the test-binary file name
    if dir.ends_with("deps") {
        dir.pop();
    }
    dir.join(format!("nsl{}", std::env::consts::EXE_SUFFIX))
}
