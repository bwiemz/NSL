//! CLI-level negative e2e tests: user `kernel` blocks with unsupported
//! constructs must FAIL compilation with an error naming the construct,
//! instead of silently compiling to PTX that produces wrong numbers
//! (deferral-must-refuse invariant).
//!
//! These run `nsl run` on fixtures in `examples/` and only exercise the
//! compile phase — no GPU required (the compile error fires before any
//! CUDA launch).

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

/// Run `nsl run` on an examples fixture, expect a nonzero exit, and return
/// the combined stderr for message assertions.
fn run_expect_compile_error(example: &str) -> String {
    let root = workspace_root();
    let example_path = root.join(format!("examples/{}.nsl", example));
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        !output.status.success(),
        "Expected compile error for '{}', but it succeeded.\nstdout: {}\nstderr: {}",
        example,
        stdout,
        stderr
    );
    assert!(
        !stdout.contains("should_not_compile"),
        "Fixture '{}' must not reach runtime, but its print executed.\nstdout: {}",
        example,
        stdout
    );
    stderr
}

#[test]
fn e2e_kernel_for_loop_refused() {
    let stderr = run_expect_compile_error("m17_kernel_for_loop_error");
    assert!(
        stderr.contains("for loop"),
        "Expected refusal naming the `for loop` construct, got:\n{}",
        stderr
    );
    assert!(
        stderr.contains("bad_loop"),
        "Expected refusal naming the kernel 'bad_loop', got:\n{}",
        stderr
    );
}

#[test]
fn e2e_kernel_eq_condition_refused() {
    let stderr = run_expect_compile_error("m17_kernel_eq_condition_error");
    assert!(
        stderr.contains("'=='"),
        "Expected refusal naming the `==` operator, got:\n{}",
        stderr
    );
    assert!(
        stderr.contains("bad_eq"),
        "Expected refusal naming the kernel 'bad_eq', got:\n{}",
        stderr
    );
}
