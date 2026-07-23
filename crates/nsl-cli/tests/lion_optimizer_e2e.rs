//! End-to-end regression for the stdlib Lion optimizer.
//!
//! Lion's update is `sign(β₁m + (1-β₁)g)`, which requires the element-wise
//! tensor `sign` builtin. Before this was wired up, a Lion `train` block failed
//! two different ways in sequence:
//!   1. semantic: `undefined variable sign` (no builtin registration), then
//!   2. codegen: `sign` typed `Unknown` made `lr * update` mis-dispatch to a
//!      scalar multiply, failing Cranelift verification ("arg has type f64,
//!      expected i64").
//!
//! This runs `nsl run` on a Lion training fixture whose `assert_close` guard
//! encodes Lion's signature property (each weight moves by exactly `lr`, so
//! `w = 1.0 - 0.01 = 0.99`). A pass proves the whole pipeline — semantic →
//! codegen → runtime → stdlib `lion.nsl` — is correct end to end. CPU-only; no
//! GPU required.

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

#[test]
fn e2e_lion_optimizer_trains_and_verifies() {
    let root = workspace_root();
    let example_path = root.join("examples/lion_optimizer_sign_e2e.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Nonzero exit means either a compile error (sign not wired up) or the
    // in-program `assert_close` aborted (wrong Lion update).
    assert!(
        output.status.success(),
        "Lion training e2e failed (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(
        stdout.contains("lion-verified"),
        "Expected 'lion-verified' (post-assert_close) in stdout, got:\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
}
