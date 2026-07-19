//! P0.3: gradient-integrity gate (`--grad-integrity`) + P0.2 guard integration.
//!
//! The gate proves, at runtime, that every trainable parameter receives a
//! finite, mostly-nonzero gradient on EVERY optimizer step — converting the
//! #396 silent-drop failure mode into an assertable signal. Two feed paths:
//!   - FullBuffer / composite (no grad accumulation): the grads list is
//!     materialized and scanned once per step.
//!   - FASE-Deferred (grad_accumulation >= 2, AdamW, source-AD): gradients are
//!     consumed during backward lowering, so the gate is fed per-parameter.
//!
//! The last test pins the P0.2 guard: a linear layer WITH a bias under
//! FASE-Deferred silently dropped its weight gradient on main (the bias grad is
//! a shared intermediate the optimizer hook frees before the weight matmul
//! reads it). The guard must now make that a hard compile error, and the
//! documented escape hatch must downgrade it to a warning.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(fixture: &str, extra_args: &[&str], envs: &[(&str, &str)]) -> Run {
    let root = repo_root();
    let path = root.join("crates/nsl-cli/tests/fixtures").join(fixture);
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli", "--", "run", "--source-ad"]);
    cmd.args(extra_args);
    cmd.arg(&path)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn nsl run");
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Parse the `[grad-integrity]` block from stderr into
/// (checks, expected, gradient, finite, nonzero, missing_count).
fn parse_integrity(stderr: &str) -> Option<(u64, u64, u64, u64, u64, usize)> {
    let mut lines = stderr.lines();
    // find the header
    lines.by_ref().find(|l| l.trim() == "[grad-integrity]")?;
    let mut map = std::collections::HashMap::new();
    let mut missing = 0usize;
    for l in lines.by_ref().take(6) {
        let l = l.trim();
        if let Some(rest) = l.strip_prefix("missing=[") {
            let inner = rest.trim_end_matches(']');
            missing = if inner.trim().is_empty() {
                0
            } else {
                inner.split(',').count()
            };
        } else if let Some((k, v)) = l.split_once('=') {
            if let Ok(n) = v.trim().parse::<u64>() {
                map.insert(k.trim().to_string(), n);
            }
        }
    }
    Some((
        *map.get("checks")?,
        *map.get("expected_params")?,
        *map.get("gradient_params")?,
        *map.get("finite")?,
        *map.get("nonzero")?,
        missing,
    ))
}

#[test]
fn grad_integrity_fullbuffer_all_params_finite_and_nonzero() {
    let r = run("grad_integrity_fullbuffer.nsl", &["--grad-integrity"], &[]);
    assert!(r.ok, "run failed:\nstdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    let (checks, expected, gradient, finite, nonzero, missing) =
        parse_integrity(&r.stderr).unwrap_or_else(|| panic!("no [grad-integrity] block:\n{}", r.stderr));
    // 3 epochs × 1 step = 3 checks (anti-vacuity).
    assert_eq!(checks, 3, "expected 3 steps checked");
    // 2 blocks × (w_up, w_down) = 4 trainable params.
    assert_eq!(expected, 4, "expected 4 trainable params");
    assert_eq!(gradient, 4, "every param must receive a gradient (no missing)");
    assert_eq!(finite, 4, "every gradient must be finite (no NaN/Inf)");
    assert_eq!(nonzero, 4, "every gradient must be nonzero on this fixture");
    assert_eq!(missing, 0, "no param may be missing a gradient");
}

#[test]
fn grad_integrity_fase_deferred_feeds_per_param() {
    let r = run("grad_integrity_fase.nsl", &["--grad-integrity"], &[]);
    assert!(r.ok, "run failed:\nstdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    let (checks, expected, gradient, finite, nonzero, missing) =
        parse_integrity(&r.stderr).unwrap_or_else(|| panic!("no [grad-integrity] block:\n{}", r.stderr));
    // 12 epochs, grad_accumulation=4 → the optimizer (and a step_end) fires 3×.
    assert!(checks >= 1, "the FASE feed path must finalize at least one step");
    assert_eq!(expected, 2, "expected 2 trainable params (w1, w2)");
    assert_eq!(gradient, 2, "both params must be noted every step");
    assert_eq!(finite, 2, "both gradients must be finite");
    assert_eq!(nonzero, 2, "both gradients must be nonzero");
    assert_eq!(missing, 0, "no param may be missing a gradient");
}

#[test]
fn p02_guard_fires_on_bias_fase_gradient_drop() {
    // Default: the P0.2 guard must reject this program (a silently-dropped
    // weight gradient becomes a hard compile error).
    let r = run("grad_integrity_bias_fase_drop.nsl", &[], &[]);
    assert!(!r.ok, "the bias/FASE gradient-drop program must NOT compile");
    assert!(
        r.stderr.contains("live gradient op has an unresolved input"),
        "must fail with the P0.2 diagnostic, got:\n{}",
        r.stderr
    );
    assert!(
        r.stderr.contains("producer chain"),
        "diagnostic must include the producer chain:\n{}",
        r.stderr
    );

    // Escape hatch: downgrade to a loud warning and let codegen proceed
    // (legacy skip). We assert codegen got PAST the guard — the warning is
    // present and the guard's hard `codegen error` is gone — rather than the
    // whole run, since the legacy skip still trains with the weight gradient
    // dropped (that is the pre-existing bug the guard exists to surface).
    let r2 = run(
        "grad_integrity_bias_fase_drop.nsl",
        &[],
        &[("NSL_ALLOW_UNRESOLVED_LIVE_ADJOINT", "1")],
    );
    assert!(
        r2.stderr.contains("NSL_ALLOW_UNRESOLVED_LIVE_ADJOINT=1"),
        "the escape hatch must emit a loud warning:\n{}",
        r2.stderr
    );
    assert!(
        !r2.stderr.contains("codegen error: [source-ad] live gradient op"),
        "the escape hatch must downgrade the hard error (codegen proceeds):\n{}",
        r2.stderr
    );
}
