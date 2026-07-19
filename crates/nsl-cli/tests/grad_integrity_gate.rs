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
//! The last test pins the ROOT FIX of the shared-intermediate drop: a linear
//! layer WITH a bias under FASE-Deferred silently dropped its weight gradient on
//! main (the bias grad IS `d_out`, which the weight matmul also reads, but the
//! optimizer hook freed it early). The fix defers that free when a later adjoint
//! op still reads the tensor, so the program now compiles and BOTH parameters
//! receive a correct, finite, nonzero gradient. (The P0.2 guard's own coverage
//! — that a genuinely-unresolvable live adjoint is still a hard compile error —
//! lives in tests/grad_integrity_unresolved_input.rs.)

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
    run_ad(fixture, true, extra_args, envs)
}

/// `source_ad = false` runs the tape-AD reference path (no `--source-ad`), which
/// never engages the FASE consume-hook — the independent oracle for the fix.
fn run_ad(fixture: &str, source_ad: bool, extra_args: &[&str], envs: &[(&str, &str)]) -> Run {
    let root = repo_root();
    let path = root.join("crates/nsl-cli/tests/fixtures").join(fixture);
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli", "--", "run"]);
    if source_ad {
        cmd.arg("--source-ad");
    }
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

/// Extract the text between `<marker>_BEGIN` and `<marker>_END` lines.
fn between<'a>(stdout: &'a str, marker: &str) -> &'a str {
    let begin = format!("{marker}_BEGIN");
    let end = format!("{marker}_END");
    let s = stdout.split_once(&begin).map(|(_, r)| r).unwrap_or("");
    s.split_once(&end).map(|(l, _)| l.trim()).unwrap_or("")
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
fn bias_fase_shared_grad_now_resolves_correctly() {
    // ROOT FIX: the bias/FASE shared-intermediate program (the bias grad IS
    // `d_out`, also read by the weight matmul) used to drop the weight gradient
    // silently, then — after P0.2 — became a hard compile error. The deferred-
    // free fix makes it compile AND both params receive correct gradients.
    let r = run("grad_integrity_bias_fase_drop.nsl", &["--grad-integrity"], &[]);
    assert!(
        r.ok,
        "the shared-intermediate bias/FASE program must now compile and run:\n\
         stdout:\n{}\nstderr:\n{}",
        r.stdout, r.stderr
    );
    // The P0.2 guard must NOT fire — the tensor is kept live, not dropped.
    assert!(
        !r.stderr.contains("live gradient op has an unresolved input"),
        "the fix must resolve the shared intermediate, not trip the guard:\n{}",
        r.stderr
    );
    let (checks, expected, gradient, finite, nonzero, missing) = parse_integrity(&r.stderr)
        .unwrap_or_else(|| panic!("no [grad-integrity] block:\n{}", r.stderr));
    assert!(checks >= 1, "the FASE feed path must finalize at least one step");
    // Linear: w (weight) + b (bias) = 2 trainable params, BOTH must get grads.
    assert_eq!(expected, 2, "expected 2 trainable params (w, b)");
    assert_eq!(gradient, 2, "both params must receive a gradient (weight NOT dropped)");
    assert_eq!(finite, 2, "both gradients must be finite");
    assert_eq!(nonzero, 2, "both gradients must be nonzero (the weight actually moves)");
    assert_eq!(missing, 0, "no param may be missing a gradient");
}

#[test]
fn fase_bias_shared_grad_is_bit_exact_with_tape_ad() {
    // The strongest correctness check for the fix: the deferred gradient must be
    // the RIGHT value, not just nonzero. source-AD+FASE (the fixed consume-hook
    // path) must produce bit-identical final weights to tape-AD (which never
    // uses the hook). Any residual drop/corruption of the shared intermediate
    // would diverge the two.
    let fase = run_ad("fase_bias_shared_grad_parity.nsl", true, &[], &[]);
    let tape = run_ad("fase_bias_shared_grad_parity.nsl", false, &[], &[]);
    assert!(fase.ok, "source-AD+FASE run failed:\n{}", fase.stderr);
    assert!(tape.ok, "tape-AD run failed:\n{}", tape.stderr);

    let (fw, fb) = (between(&fase.stdout, "FINAL_W"), between(&fase.stdout, "FINAL_B"));
    let (tw, tb) = (between(&tape.stdout, "FINAL_W"), between(&tape.stdout, "FINAL_B"));
    assert!(!fw.is_empty() && !fb.is_empty(), "FASE run printed no weights:\n{}", fase.stdout);
    // The weight moved off its init (1.0) — the fix restored a real update.
    assert!(
        fw.contains("0.99"),
        "the weight must actually move under FASE (was frozen with the bug): {fw}"
    );
    assert_eq!(fw, tw, "final weight must be bit-exact: FASE={fw} tape={tw}");
    assert_eq!(fb, tb, "final bias must be bit-exact: FASE={fb} tape={tb}");
}
