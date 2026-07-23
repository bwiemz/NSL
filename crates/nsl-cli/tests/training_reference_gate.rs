//! P1.7: `--training-reference` mode gate.
//!
//! The reference path forces the simplest correct training: no CCR, no CSLA, no
//! weight streaming, no offload, no WGGO, no CPDT/reduced precision, no CSHA, no
//! kernel `@fuse`, no FBIP in-place, no fused FASE optimizer step, and no
//! fused-CE substitution. It exists so an optimized stack can be compared
//! against an INDEPENDENT baseline (stronger than comparing two optimized paths
//! that may share the same bug).
//!
//! This gate pins two properties on a small FASE-Deferred fixture:
//!   1. the optimization is actually OFF — the fused optimizer step launches N
//!      times on the default path and ZERO times under --training-reference;
//!   2. the reference path is numerically FAITHFUL — its loss stream matches the
//!      optimized path (here bit-exact, since FASE≡AdamW), so it is a valid
//!      comparison baseline rather than a different computation.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run(fixture: &str, reference: bool) -> (bool, String, String) {
    let root = repo_root();
    let path = root.join("crates/nsl-cli/tests/fixtures").join(fixture);
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run", "--source-ad"]);
    if reference {
        cmd.arg("--training-reference");
    }
    let out = cmd
        .arg(&path)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_FASE_FUSED_COUNTER", "1")
        .output()
        .expect("spawn nsl run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn loss_stream(stdout: &str) -> String {
    let mut out = String::new();
    let mut in_s = false;
    for l in stdout.lines() {
        match l.trim() {
            "LOSS_STREAM_BEGIN" => in_s = true,
            "LOSS_STREAM_END" => in_s = false,
            x if in_s => {
                out.push_str(x);
                out.push('\n');
            }
            _ => {}
        }
    }
    out
}

fn fused_step_launches(stderr: &str) -> i64 {
    stderr
        .lines()
        .find_map(|l| l.strip_prefix("[fase-fused] optimizer fused-step launches: "))
        .and_then(|n| n.trim().parse().ok())
        .expect("fused-step counter missing (NSL_FASE_FUSED_COUNTER)")
}

#[test]
fn training_reference_disables_fused_step_and_stays_faithful() {
    // The fixture prints its loss stream via on_step.
    let (ok_opt, out_opt, err_opt) = run("grad_integrity_fase.nsl", false);
    assert!(ok_opt, "optimized run failed:\n{err_opt}");
    let (ok_ref, out_ref, err_ref) = run("grad_integrity_fase.nsl", true);
    assert!(ok_ref, "reference run failed:\n{err_ref}");

    // The override note is loud.
    assert!(
        err_ref.contains("--training-reference forces the simplest correct training path"),
        "reference-mode override note missing:\n{err_ref}"
    );

    // The optimization is actually off.
    assert!(
        fused_step_launches(&err_opt) > 0,
        "the optimized path must launch the fused optimizer step"
    );
    assert_eq!(
        fused_step_launches(&err_ref),
        0,
        "--training-reference must NOT launch the fused optimizer step"
    );

    // The reference path is a faithful baseline.
    let l_opt = loss_stream(&out_opt);
    let l_ref = loss_stream(&out_ref);
    assert!(!l_opt.trim().is_empty(), "no loss stream captured");
    assert_eq!(
        l_opt, l_ref,
        "the reference loss stream must match the optimized path (FASE≡AdamW) — \
         if it diverges, one of the two paths is wrong"
    );
}
