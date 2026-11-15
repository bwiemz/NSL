//! A compiled program refuses a run-time `NSL_MATMUL_BF16` that disagrees with
//! the matmul mode it was compiled with.
//!
//! The mode is fixed when the program is compiled and recorded in its
//! checkpoints' execution fingerprint, so the runtime never switches on its own
//! environment. Until November 2026 it also did not say so: a binary built
//! without the variable and run with `NSL_MATMUL_BF16=1` ran TF32 under a TF32
//! banner, and `models/benchmarks/posture_bench_1b.py`'s bf16 arm measured TF32
//! that way. CPU-only: the check runs at program start, before any GEMM.
//!
//! Run: cargo test -p nsl-cli --test matmul_mode_runtime_env_gate

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

struct Out {
    ok: bool,
    stdout: String,
    stderr: String,
}

/// `nsl build` a one-line program with `flags` in `dir`; returns the binary.
fn build(dir: &std::path::Path, flags: &[&str]) -> std::path::PathBuf {
    let src = dir.join("p.nsl");
    std::fs::write(&src, "print(\"RAN\")\n").unwrap();
    let bin = dir.join("p");
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .arg("build")
        .args(flags)
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .current_dir(dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        // The build's own environment must not pick the mode.
        .env_remove("NSL_MATMUL_BF16")
        .output()
        .expect("spawn nsl build");
    assert!(out.status.success(), "build failed:\n{}", String::from_utf8_lossy(&out.stderr));
    bin
}

fn run(bin: &std::path::Path, bf16: Option<&str>) -> Out {
    let mut cmd = Command::new(bin);
    cmd.env_remove("NSL_MATMUL_BF16");
    if let Some(v) = bf16 {
        cmd.env("NSL_MATMUL_BF16", v);
    }
    let out = cmd.output().expect("spawn the built program");
    Out {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn fresh_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("nsl_mmenv_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// THE REGRESSION: built with the default mode (tf32), run with
/// NSL_MATMUL_BF16=1. Refused, before the program does anything, with a
/// message that names the fix.
#[test]
fn a_disagreeing_run_time_variable_is_refused() {
    let dir = fresh_dir("refuse");
    let bin = build(&dir, &[]);
    let r = run(&bin, Some("1"));
    assert!(!r.ok, "a tf32 build ran under NSL_MATMUL_BF16=1:\n{}{}", r.stdout, r.stderr);
    assert!(!r.stdout.contains("RAN"), "the refusal must come before the program runs");
    assert!(
        r.stderr.contains("compiled with matmul mode tf32") && r.stderr.contains("--matmul-mode bf16"),
        "the message must say what was compiled and how to get bf16:\n{}",
        r.stderr
    );
    // Without the variable, the same binary runs.
    let r = run(&bin, None);
    assert!(r.ok && r.stdout.contains("RAN"), "{}", r.stderr);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A bf16 build run with the variable that names its mode runs; one pinned
/// off with NSL_MATMUL_BF16=0 is refused too.
#[test]
fn an_agreeing_variable_runs_and_a_pin_off_is_refused() {
    let dir = fresh_dir("bf16");
    let bin = build(&dir, &["--matmul-mode", "bf16"]);
    let r = run(&bin, Some("1"));
    assert!(r.ok && r.stdout.contains("RAN"), "{}", r.stderr);
    let dir2 = fresh_dir("bf16_env");
    // Not explicit: the mode came from the variable at build time.
    let src = dir2.join("p.nsl");
    std::fs::write(&src, "print(\"RAN\")\n").unwrap();
    let bin2 = dir2.join("p");
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["build"])
        .arg(&src)
        .arg("-o")
        .arg(&bin2)
        .current_dir(&dir2)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .env("NSL_MATMUL_BF16", "1")
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let r = run(&bin2, Some("0"));
    assert!(!r.ok, "a bf16 build ran under NSL_MATMUL_BF16=0:\n{}", r.stderr);
    assert!(r.stderr.contains("compiled with matmul mode bf16"), "{}", r.stderr);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

/// An explicit --matmul-mode keeps its precedence over an inherited variable
/// (the 2026-10-06 review's rule): warned, not refused.
#[test]
fn an_explicit_mode_is_warned_not_refused() {
    let dir = fresh_dir("explicit");
    let bin = build(&dir, &["--matmul-mode", "tf32"]);
    let r = run(&bin, Some("1"));
    assert!(r.ok && r.stdout.contains("RAN"), "{}", r.stderr);
    assert!(r.stderr.contains("NSL_MATMUL_BF16=1 is set but ignored"), "{}", r.stderr);
    let _ = std::fs::remove_dir_all(&dir);
}
