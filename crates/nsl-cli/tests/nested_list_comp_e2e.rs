//! e2e regression for nested (multi-generator) list comprehensions.
//!
//! Codegen previously lowered only `generators[0]` and silently ignored the
//! rest: `[x*100 for x in [1,2] for y in [10,20,30]]` produced 2 elements
//! instead of 6 (silent wrong results), inner `if` filters were dropped, and
//! elements referencing the inner variable failed with "undefined variable".
//! The fixture is self-checking (assert_eq aborts on mismatch) and covers
//! cross products, the silent-wrong shape, filters on both generators,
//! dependent generators (flatten), the single-generator regression, and
//! triple nesting.

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
fn e2e_nested_list_comprehensions() {
    let root = workspace_root();
    let example = root.join("examples/nested_list_comprehensions.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("nested-comp-ok"),
        "nested list-comprehension e2e failed (exit {:?}).\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
}
