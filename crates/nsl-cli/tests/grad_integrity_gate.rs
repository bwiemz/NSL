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
    cmd.args(["run", "-q", "-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"]);
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

/// The contribution-count fields, APPENDED after the frozen six-line prefix:
/// `(notes_expected, notes_observed, under_noted, over_noted)`.
///
/// Every field is anchored on its LABEL, inside a deliberately generous
/// bounded window (the report is 11 lines today; 32 leaves slack). The
/// six-line window above is what pins the legacy fields' positions —
/// duplicating that exactness here would make a future append break two
/// parsers instead of none, which is the whole reason the new fields were
/// appended rather than inserted.
fn parse_notes(stderr: &str) -> Option<(u64, String, Vec<u64>, Vec<u64>)> {
    let block: Vec<&str> = stderr
        .lines()
        .skip_while(|l| l.trim() != "[grad-integrity]")
        .skip(1)
        .take(32)
        .collect();
    let idx_list = |key: &str| -> Option<Vec<u64>> {
        let rest = block
            .iter()
            .find_map(|l| l.trim().strip_prefix(key))?
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']');
        if rest.trim().is_empty() {
            return Some(Vec::new());
        }
        rest.split(',').map(|s| s.trim().parse().ok()).collect()
    };
    Some((
        block
            .iter()
            .find_map(|l| l.trim().strip_prefix("notes_expected="))?
            .trim()
            .parse()
            .ok()?,
        block
            .iter()
            .find_map(|l| l.trim().strip_prefix("notes_observed="))?
            .trim()
            .to_string(),
        idx_list("under_noted=")?,
        idx_list("over_noted=")?,
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
    // The composite scan sees each param's gradient exactly once per step.
    let (n_exp, n_obs, under, over) = parse_notes(&r.stderr)
        .unwrap_or_else(|| panic!("no contribution-count fields:\n{}", r.stderr));
    assert_eq!(n_exp, 1, "the composite scan declares one contribution/param");
    assert_eq!(n_obs, "1..1", "every param was scanned exactly once");
    // TAUTOLOGICAL ON THIS PATH, and kept deliberately as a shape check
    // rather than a detection claim: `nsl_grad_integrity_check` visits each
    // index exactly once, so a FullBuffer count can only be 0 (which
    // `missing` owns) or 1 (which equals `expected`). Neither verdict list
    // is reachable here. The detection this feature exists for is proven on
    // the paths where a param CAN receive several contributions — the
    // fault-injected arms below and the CSLA window arms.
    assert!(under.is_empty() && over.is_empty(), "under={under:?} over={over:?}");
}

#[test]
fn grad_integrity_fase_deferred_feeds_per_param() {
    let r = run("grad_integrity_fase.nsl", &["--grad-integrity"], &[]);
    assert!(r.ok, "run failed:\nstdout:\n{}\nstderr:\n{}", r.stdout, r.stderr);
    let (checks, expected, gradient, finite, nonzero, missing) =
        parse_integrity(&r.stderr).unwrap_or_else(|| panic!("no [grad-integrity] block:\n{}", r.stderr));
    // 12 epochs, grad_accumulation=4 → the optimizer fires 3×, but a step_end
    // fires per MICRO-batch (the bracket wraps one adjoint lowering, not the
    // window), so `checks` is 12 — pinned below.
    assert!(checks >= 1, "the FASE feed path must finalize at least one step");
    assert_eq!(expected, 2, "expected 2 trainable params (w1, w2)");
    assert_eq!(gradient, 2, "both params must be noted every step");
    assert_eq!(finite, 2, "both gradients must be finite");
    assert_eq!(nonzero, 2, "both gradients must be nonzero");
    assert_eq!(missing, 0, "no param may be missing a gradient");
    // The FASE-interleaved bracket wraps ONE micro-batch, so the emitter
    // declares one contribution per param and the hook must deliver exactly
    // that. `checks` counting micro-batches (12 = epochs) rather than
    // optimizer steps (3) is what makes that declaration correct.
    assert_eq!(checks, 12, "the baseline bracket is per MICRO-batch");
    let (n_exp, n_obs, under, over) = parse_notes(&r.stderr)
        .unwrap_or_else(|| panic!("no contribution-count fields:\n{}", r.stderr));
    assert_eq!(n_exp, 1, "per-micro-batch bracket declares 1 note/param");
    assert_eq!(n_obs, "1..1", "each param's gradient is consumed once/micro-batch");
    assert!(under.is_empty() && over.is_empty(), "under={under:?} over={over:?}");
}

/// Anti-vacuity for the contribution counter: the one-note-per-(param,
/// micro-batch) invariant is structurally true on every fixture in the tree,
/// so `under_noted`/`over_noted` read `[]` everywhere and a stub that always
/// printed `[]` would be indistinguishable from the real check. Drive the
/// announced fault hook to make the failing arm reachable.
///
/// The load-bearing part is the SECOND assertion block: with the fault on,
/// all six pre-existing fields still read fully green. A gradient consumed
/// twice — double-counted into `m_partial` at the same 1/N scale — is exactly
/// the class of silent corruption this gate exists to catch, and before the
/// count it was invisible.
#[test]
fn double_consumed_gradient_is_reported_and_the_control_run_is_clean() {
    let faulted = run(
        "grad_integrity_fase.nsl",
        &["--grad-integrity"],
        &[("NSL_GRAD_INTEGRITY_FAULT", "double:1")],
    );
    assert!(faulted.ok, "fault run failed:\n{}", faulted.stderr);
    assert!(
        faulted.stderr.contains("FAULT INJECTION ACTIVE"),
        "the fault hook must announce itself (an unannounced one would let a \
         faulted run be mistaken for a real one):\n{}",
        faulted.stderr
    );
    let (_, n_obs, under, over) = parse_notes(&faulted.stderr)
        .unwrap_or_else(|| panic!("no contribution-count fields:\n{}", faulted.stderr));
    assert_eq!(over, vec![1], "param 1 was consumed twice per bracket");
    assert!(under.is_empty(), "a doubled note is not an under-note: {under:?}");
    assert_eq!(n_obs, "1..2", "param 0 kept 1 contribution, param 1 got 2");

    let (checks, expected, gradient, finite, nonzero, missing) = parse_integrity(&faulted.stderr)
        .unwrap_or_else(|| panic!("no [grad-integrity] block:\n{}", faulted.stderr));
    assert_eq!(
        (checks, expected, gradient, finite, nonzero, missing),
        (12, 2, 2, 2, 2, 0),
        "the six legacy fields must be BLIND to a double-consumed gradient — \
         if this ever fails they grew a signal and the claim above is stale:\n{}",
        faulted.stderr
    );

    // Control: same program, no fault. Proves the verdict is not stuck on.
    let clean = run("grad_integrity_fase.nsl", &["--grad-integrity"], &[]);
    assert!(clean.ok, "control run failed:\n{}", clean.stderr);
    assert!(
        !clean.stderr.contains("FAULT INJECTION ACTIVE"),
        "the control run must not have the fault hook armed:\n{}",
        clean.stderr
    );
    let (_, c_obs, c_under, c_over) = parse_notes(&clean.stderr)
        .unwrap_or_else(|| panic!("no contribution-count fields:\n{}", clean.stderr));
    assert_eq!(c_obs, "1..1");
    assert!(
        c_under.is_empty() && c_over.is_empty(),
        "control run must be clean: under={c_under:?} over={c_over:?}"
    );
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
