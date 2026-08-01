//! `nsl run` stdout must contain the PROGRAM's output and nothing else.
//!
//! Build-toolchain chatter belongs on stderr. This is not a style preference:
//! `nsl run prog.nsl > out.txt` is a normal thing to do, and anything the
//! compile step prints to stdout lands in the user's data file.
//!
//! ## The regression this pins
//!
//! MSVC's `link.exe` writes informational text to its STDOUT — notably
//! `   Creating library X.lib and object X.exp` whenever it emits an import
//! library. `linker.rs::link_msvc_multi` captured that output and re-emitted it
//! with `print!`, so it was injected straight into program output on Windows.
//! `link_shared_msvc` was worse: it used `.status()`, letting link.exe stream
//! directly to the inherited stdout.
//!
//! The symptom was worked around test-by-test for a long time —
//! `e2e.rs` carries a `strip()` helper plus a second inline filter,
//! `ccr_checkpoint_activation_parity.rs` and `fused_lm_ce_e2e_nsl_source.rs`
//! each document and filter it locally. `builtin_shadow_dispatch.rs` did not
//! filter, which is why `unshadowed_sum_abs_reduce_max_still_hit_builtins` was
//! the standing `build-and-test (windows-latest)` CI failure rather than a
//! known-noisy-output annoyance.
//!
//! Both linker paths now route captured output to stderr, so the contract holds
//! on every platform. This test states the contract in one obvious place
//! instead of leaving it implicit in a dispatch test.
//!
//! On Linux/macOS `cc` is silent on success, so this passes there both before
//! and after the fix — it is a Windows regression gate that costs a CPU-only
//! run elsewhere, and it also catches any future `println!` added to the
//! compile path on any platform.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_nsl(src: &str, tag: &str) -> std::process::Output {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_stdout_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    std::fs::remove_dir_all(&tmp).ok();
    out
}

/// Exact-match the whole stdout stream against what the program prints.
#[test]
fn run_stdout_contains_only_program_output() {
    let src = r#"
print("ALPHA")
print(21 + 21)
print("OMEGA")
"#;
    let out = run_nsl(src, "onlyprog");
    assert!(
        out.status.success(),
        "nsl run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();

    assert_eq!(
        lines,
        ["ALPHA", "42", "OMEGA"],
        "nsl run stdout carried non-program output. Everything the build emits \
         (notably MSVC link.exe's `Creating library ...`) must go to stderr, or \
         `nsl run prog.nsl > out.txt` corrupts the user's data.\n\
         --- stdout ---\n{stdout}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The complement: a program that prints NOTHING must produce EMPTY stdout.
///
/// Sharper than the test above — an exact-line assertion can be satisfied by
/// noise that happens to sort outside the compared slice, but "empty means
/// empty" cannot. This is the assertion that actually fails on a stock Windows
/// runner pre-fix, since `Creating library ...` is emitted regardless of what
/// the program prints.
#[test]
fn silent_program_produces_empty_stdout() {
    let src = r#"
let x = 1 + 1
"#;
    let out = run_nsl(src, "silent");
    assert!(
        out.status.success(),
        "nsl run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.trim().is_empty(),
        "a program that prints nothing must leave stdout empty, but it \
         contained build output:\n--- stdout ---\n{stdout}"
    );
}
