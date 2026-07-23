//! End-to-end regression for source-AD conv2d gradients.
//!
//! Conv2d gradients used to be refused in source AD (the extractor never emitted
//! `PrimalOp::Conv2d`, so conv fell back to tape AD, and the lowering panicked if
//! ever reached). They are now implemented: the extractor emits `Conv2d`, and the
//! input/weight/bias gradients lower to `nsl_conv2d_{input,weight,bias}_backward`,
//! which wrap the same verified nested-loop `conv2d_backward` the tape path uses.
//!
//! This runs the fixture with `--source-ad` so the new path (not the tape
//! fallback) is exercised. The fixture's `assert_close` guards encode the
//! closed-form gradients of a sum-loss over an all-ones conv (dW=4, dBias=4,
//! dInput=[[1,2,1],[2,4,2],[1,2,1]]), so a pass proves the gradients are
//! numerically correct end to end.

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
fn e2e_conv2d_source_ad_gradients() {
    let root = workspace_root();
    let example = root.join("examples/conv2d_source_ad_grad.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run", "--source-ad"])
        .arg(&example)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run --source-ad");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Nonzero exit means either a compile error or an in-program `assert_close`
    // abort (wrong conv gradient).
    assert!(
        output.status.success(),
        "conv2d source-AD gradient e2e failed (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
    assert!(
        stdout.contains("conv-source-ad-grad-ok"),
        "expected 'conv-source-ad-grad-ok' in stdout, got:\nstdout:\n{}\nstderr:\n{}",
        stdout,
        stderr
    );
    // The source-AD path must actually be used — not silently fall back to tape.
    assert!(
        !stderr.contains("source AD extraction failed"),
        "source AD fell back to tape AD; conv path was not exercised.\nstderr:\n{}",
        stderr
    );
}
