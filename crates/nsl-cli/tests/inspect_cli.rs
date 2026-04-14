//! Phase 5 Task 8: --inspect flag presence + CLI help shows the flag.

use std::process::Command;

#[test]
fn inspect_flag_is_accepted_by_run_help() {
    let bin = env!("CARGO_BIN_EXE_nsl");
    let out = Command::new(bin)
        .args(["run", "--help"])
        .output()
        .expect("failed to spawn nsl binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let combined = format!("{}{}", stdout, stderr);
    assert!(
        combined.contains("--inspect"),
        "--inspect not in `nsl run --help`:\nstdout: {}\nstderr: {}",
        stdout,
        stderr
    );
}
