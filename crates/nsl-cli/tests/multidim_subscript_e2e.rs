//! e2e regression for multi-dim tensor subscript `t[i, j]`.
//!
//! Previously: reads hit "multi-dim subscript not supported yet" and
//! assignments hit "only simple index assignment supported" — both loud
//! refusals despite the parser producing SubscriptKind::MultiDim and the
//! runtime already shipping stride-aware, bounds-checked
//! nsl_tensor_get/nsl_tensor_set. Now: reads type as Float (f64 element),
//! writes and compound assigns (+=, -=, *=, /=) work, verified by the
//! self-checking fixture (assert_eq aborts on mismatch).

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
fn e2e_multidim_tensor_subscript() {
    let root = workspace_root();
    let example = root.join("examples/multidim_tensor_subscript.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"])
        .arg(&example)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("multidim-ok"),
        "multi-dim subscript e2e failed (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
}
