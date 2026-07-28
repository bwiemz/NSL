//! `nsl run` / `nsl build` must not leave their compile scratch behind.
//!
//! Every temp-dir site used to finish with `std::fs::remove_dir(&temp_dir)`,
//! which only succeeds on an EMPTY directory — and a build leaves object files,
//! PTX and other intermediates in there. The call was `let _ = ...`, so the
//! failure was silent and the directory survived intact.
//!
//! Measured on 2026-07-24: **120 MB retained per `nsl run` invocation**, and
//! 8781 leftover `/tmp/nsl_*` directories totalling **29 GB** had accumulated.
//! Because `/tmp` is tmpfs on this machine, a full `cargo test --workspace`
//! eventually exhausted it and `ld` began failing with
//! "final link failed: No space left on device" — which the harness surfaced as
//! 15 test failures across five unrelated targets, reading exactly like a
//! codegen regression. That is the real cost of this bug: not the disk, but the
//! hours spent bisecting a compiler change that was never at fault.
//!
//! Race-free by construction: each case points the child at its OWN `TMPDIR`,
//! so a concurrently running test cannot perturb the count.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const PROG: &str = r#"
let a = zeros([4, 4])
let b = a + 1.0
print(b.ndim)
print("DONE")
"#;

/// Run `nsl run` with a private TMPDIR and return (stdout, entries left in it).
fn run_with_private_tmp(tag: &str, keep_temp: bool) -> (String, Vec<String>) {
    let root = repo_root();
    let base = std::env::temp_dir().join(format!(
        "nsl_scratchgate_{}_{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let work = base.join("work");
    let priv_tmp = base.join("tmp");
    std::fs::create_dir_all(&work).unwrap();
    std::fs::create_dir_all(&priv_tmp).unwrap();

    let prog = work.join("prog.nsl");
    std::fs::write(&prog, PROG).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run"])
        .arg(&prog)
        .current_dir(&work)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // std::env::temp_dir() honours TMPDIR on unix, TMP/TEMP on Windows.
        .env("TMPDIR", &priv_tmp)
        .env("TMP", &priv_tmp)
        .env("TEMP", &priv_tmp);
    if keep_temp {
        cmd.env("NSL_KEEP_TEMP", "1");
    }
    let out = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "[{tag}] nsl run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("DONE"), "[{tag}] program did not run:\n{stdout}");

    let left: Vec<String> = std::fs::read_dir(&priv_tmp)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.starts_with("nsl_"))
                .collect()
        })
        .unwrap_or_default();
    let _ = std::fs::remove_dir_all(&base);
    (stdout, left)
}

#[test]
fn nsl_run_removes_its_compile_scratch() {
    let (_, left) = run_with_private_tmp("clean", false);
    assert!(
        left.is_empty(),
        "nsl run left compile scratch behind: {left:?} — the cleanup is \
         non-recursive again (remove_dir cannot delete a directory holding \
         object files, and the failure is silent)"
    );
}

/// Anti-vacuity: the gate above proves nothing unless the scratch directory
/// genuinely exists during the run. `NSL_KEEP_TEMP=1` opts out of the cleanup,
/// so the directory must be there.
#[test]
fn keep_temp_env_retains_the_scratch() {
    let (_, left) = run_with_private_tmp("keep", true);
    assert!(
        !left.is_empty(),
        "NSL_KEEP_TEMP=1 should have retained the compile scratch, but the \
         private TMPDIR is empty — either the opt-out broke, or the child is \
         not using TMPDIR and the companion gate is vacuous"
    );
}
