//! Roadmap item 2 step 1 — gates for the pass execution trace.
//!
//! The trace answers a question the [`nsl_codegen::pass_registry`] cannot:
//! *did this pass actually run?* The registry proves a pass exists and that
//! its flags are wired; nothing proved the flag reached the pass. That gap has
//! produced real defects in this tree (an autotuner chooser with zero
//! production callers, GPU kernels declining silently to CPU, a certification
//! gate silently off), all of which look like a clean build.
//!
//! Every test here asserts BOTH halves — flag on ⇒ the pass ran, flag off ⇒ it
//! did not. The negative half is what stops the gate degenerating into "the
//! marker appeared", which a hard-coded trace would satisfy just as well.
//!
//! CPU-only and fast: `nsl run` on a tiny FFN fixture, no GPU required, so
//! these run in ordinary CI rather than the hardware lane.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct Run {
    stdout: String,
    stderr: String,
    ok: bool,
}

/// Compile+run the CSLA FFN fixture on the CPU with `extra` flags.
fn run(tag: &str, extra: &[&str], trace: bool) -> Run {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_passtrace_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl"),
    )
    .expect("ffn fixture missing")
    .replace(
        "CSLA_SAVE_PATH",
        &tmp.join("out.nslm").display().to_string().replace('\\', "/"),
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if trace {
        cmd.env("NSL_PASS_TRACE", "1");
    }
    let out = cmd.output().expect("spawn nsl run");
    let r = Run {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        ok: out.status.success(),
    };
    let _ = std::fs::remove_dir_all(&tmp);
    r
}

/// The `N pass(es) ran: A(Stage) -> B(Stage)` line, as a list of pass names.
fn ran(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .find(|l| l.contains("[pass-trace]") && l.contains("pass(es) ran:"))
        .map(|l| {
            l.split("ran:")
                .nth(1)
                .unwrap_or("")
                .split("->")
                .filter_map(|p| p.trim().split('(').next())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// The passes the report says did NOT run.
fn idle(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .find(|l| l.contains("[pass-trace] did not run:"))
        .map(|l| {
            l.split("did not run:")
                .nth(1)
                .unwrap_or("")
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// `NAME(Stage)@Phase` -> (name, phase-or-None-if-unattributed), in order.
fn ran_with_phases(stderr: &str) -> Vec<(String, Option<String>)> {
    stderr
        .lines()
        .find(|l| l.contains("[pass-trace]") && l.contains("pass(es) ran:"))
        .map(|l| {
            l.split("ran:")
                .nth(1)
                .unwrap_or("")
                .split("->")
                .filter_map(|tok| {
                    let tok = tok.trim();
                    let name = tok.split('(').next()?.trim().to_string();
                    if name.is_empty() {
                        return None;
                    }
                    let phase = tok.split('@').nth(1).map(|p| p.trim().to_string());
                    Some((name, phase))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn loss_stream(stdout: &str) -> String {
    stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default()
}

/// The core gate: a pass runs when its flag is set, and does NOT run when it
/// is not. Both directions, for every pass whose activating flag works on
/// this CPU fixture.
///
/// Passes deliberately not covered here, with the reason each is out of reach
/// rather than merely omitted:
///   * CPDT   — `--cpdt` refuses without a weights file;
///   * CPKD   — `--cpkd-target` needs a teacher checkpoint;
///   * CEP    — offline weight surgery, needs real weights;
///   * CFIE / MemoryPlanner — their flags live on `build`/`check`, not `run`
///     (which is exactly the non-uniform subcommand placement the registry's
///     `Subcommand` field records);
///   * PCA    — decorator-only, needs a packing-shaped model.
/// FASE is covered by `fase_runs_unconditionally` below.
#[test]
fn a_pass_runs_only_when_its_activating_flag_is_set() {
    // (tag, activating flags, pass name)
    let cases: &[(&str, &[&str], &str)] = &[
        ("ccr", &["--checkpoint-blocks", "--layerwise-accum"], "CCR"),
        ("csha", &["--csha", "auto"], "CSHA"),
        ("wggo", &["--wggo", "full"], "WGGO"),
    ];
    let base = run("base", &[], true);
    assert!(base.ok, "baseline run failed:\n{}", base.stderr);
    let base_ran = ran(&base.stderr);
    assert!(
        !base_ran.is_empty(),
        "baseline recorded no passes at all — the trace is not wired:\n{}",
        base.stderr
    );

    for (tag, flags, pass) in cases {
        // OFF: the pass must be reported as idle, and absent from `ran`.
        assert!(
            !base_ran.iter().any(|p| p == pass),
            "{pass} ran with no flag set — the positive half below would be \
             vacuous:\n{}",
            base.stderr
        );
        assert!(
            idle(&base.stderr).iter().any(|p| p == pass),
            "{pass} is missing from the baseline's 'did not run' list, so the \
             report is not enumerating idle passes:\n{}",
            base.stderr
        );

        // ON: the flag must actually reach the pass.
        let on = run(tag, flags, true);
        assert!(on.ok, "{tag} run failed:\n{}", on.stderr);
        assert!(
            ran(&on.stderr).iter().any(|p| p == pass),
            "{:?} did not cause {pass} to run — a flag that never reaches its \
             pass produces a clean, plausible, wrong build:\n{}",
            flags,
            on.stderr
        );
    }
}

/// FASE is planned by the train-block driver on every source-AD compile, with
/// no flag. Pinned separately because it is the one pass whose *absence*
/// would mean the trace broke rather than that a feature was off.
#[test]
fn fase_runs_unconditionally_on_a_source_ad_compile() {
    let r = run("fase", &[], true);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    assert!(
        ran(&r.stderr).iter().any(|p| p == "FASE"),
        "FASE did not run on a plain --source-ad compile:\n{}",
        r.stderr
    );
}

/// The trace is opt-in and pure: without the env var it is silent, and
/// enabling it must not move a single loss value. It is compiler-side data —
/// nothing is emitted into the program — so any divergence here means the
/// instrumentation is doing more than observing.
#[test]
fn the_trace_is_opt_in_and_cannot_change_the_program() {
    let off = run("off", &["--checkpoint-blocks", "--layerwise-accum"], false);
    assert!(off.ok, "run failed:\n{}", off.stderr);
    assert!(
        !off.stderr.contains("[pass-trace]"),
        "the trace printed without NSL_PASS_TRACE=1:\n{}",
        off.stderr
    );

    let on = run("on", &["--checkpoint-blocks", "--layerwise-accum"], true);
    assert!(on.ok, "run failed:\n{}", on.stderr);
    assert!(on.stderr.contains("[pass-trace]"), "no trace with the env set");

    assert!(
        !loss_stream(&off.stdout).is_empty(),
        "no losses captured — the purity check would be vacuous"
    );
    assert_eq!(
        loss_stream(&off.stdout),
        loss_stream(&on.stdout),
        "NSL_PASS_TRACE changed the loss stream — the trace is not pure"
    );
}

/// The report must name what did NOT run. A report showing only what happened
/// makes "no passes ran" indistinguishable from "the trace is broken", which
/// is the failure mode this whole mechanism exists to prevent.
#[test]
fn the_report_enumerates_idle_passes_too() {
    let r = run("idle", &[], true);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    let idle = idle(&r.stderr);
    for expected in ["CCR", "WGGO", "CSHA", "CPDT", "CEP", "CFIE"] {
        assert!(
            idle.iter().any(|p| p == expected),
            "{expected} should be listed as not-run on a bare compile:\n{}",
            r.stderr
        );
    }
}

/// **Pins a real finding, so it cannot be lost or silently "fixed".**
///
/// Under `--wggo full` the observed order is `WGGO -> FASE`. By the
/// registry's declared stages that is an inversion: WGGO is `OnWengert` and
/// FASE is `PreExtraction`, so FASE should come first.
///
/// It is not a bug. Traced to its cause: WGGO's first invocation comes from
/// `wggo_prepass::plan_train_block`, a driver that runs BEFORE
/// `compile_train_block_inner`, where FASE is planned at `stmt.rs` ~5599.
/// Two different drivers invoke passes, and `PipelineStage` describes where a
/// pass conceptually *acts*, not when it is *invoked*.
///
/// The consequence matters for the rest of item 2: `stage` cannot be used as
/// a scheduling key by a future pass manager — an explicit ordering or
/// dependency declaration is required. This test exists so that conclusion
/// stays attached to evidence; if the order ever changes, whoever changes it
/// reads this and updates the plan rather than discovering it later.
#[test]
fn wggo_is_invoked_before_the_train_block_driver_despite_its_declared_stage() {
    let r = run("order", &["--wggo", "full"], true);
    assert!(r.ok, "run failed:\n{}", r.stderr);
    let seq = ran(&r.stderr);
    let wggo = seq.iter().position(|p| p == "WGGO");
    let fase = seq.iter().position(|p| p == "FASE");
    assert!(
        wggo.is_some() && fase.is_some(),
        "expected both WGGO and FASE in the trace, got {seq:?}:\n{}",
        r.stderr
    );
    assert!(
        wggo < fase,
        "WGGO no longer precedes FASE (got {seq:?}). If this changed \
         deliberately, update the item-2 notes: the conclusion that \
         PipelineStage is not a scheduling key rests on this observation."
    );
}

/// Guards the fixture the tests above depend on, so a rename turns into a
/// clear failure rather than six confusing ones.
#[test]
fn the_fixture_these_gates_use_exists() {
    let p = repo_root().join("crates/nsl-cli/tests/fixtures/csla_layerwise_ffn.nsl");
    assert!(p.exists(), "missing fixture: {}", p.display());
    assert!(
        std::fs::read_to_string(&p).unwrap().contains("CSLA_SAVE_PATH"),
        "fixture lost its save marker"
    );
}

/// **Item 2 step 2.** Every pass must run in the driver phase the registry
/// declares, checked against `pass_registry` itself rather than a literal
/// copied into the test.
///
/// This is what turns the step-1 finding into an enforced contract. The
/// apparent stage inversion (`WGGO(OnWengert)` before `FASE(PreExtraction)`)
/// is not an inversion at all — WGGO runs in `KernelPrepass`, which precedes
/// every train block — and now the report says so on its face.
///
/// The `unattributed` case is the load-bearing one: a pass reached from a
/// driver nobody wrapped in `enter_phase` has no phase, and that is exactly
/// how a THIRD driver would announce itself. Discovering the second one cost
/// a backtrace hunt; this makes the next one fail a test instead.
#[test]
fn every_pass_runs_in_the_compile_phase_the_registry_declares() {
    use nsl_codegen::pass_registry::CompilePhase;
    let cases: &[(&str, &[&str])] = &[
        ("p_wggo", &["--wggo", "full", "--csha", "auto"]),
        ("p_ccr", &["--checkpoint-blocks", "--layerwise-accum", "--csha", "auto"]),
        ("p_bare", &[]),
    ];
    let mut checked = 0usize;
    let mut phases_seen = std::collections::HashSet::new();
    for (tag, flags) in cases {
        let r = run(tag, flags, true);
        assert!(r.ok, "{tag} run failed:\n{}", r.stderr);
        let observed = ran_with_phases(&r.stderr);
        assert!(
            !observed.is_empty(),
            "{tag}: no passes parsed from the trace line:\n{}",
            r.stderr
        );
        for (name, phase) in observed {
            let d = nsl_codegen::pass_registry::pass(&name)
                .unwrap_or_else(|| panic!("{name} ran but is not registered"));
            // `report()` always emits an `@token`, so an unattributed pass
            // reads as the literal "unattributed" — NOT as a missing field.
            // Mapping OutOfBand to None here made the arm fail on correct
            // behaviour; it is only latent because no gate config runs CEP.
            let observed = phase.as_deref().unwrap_or("unattributed");
            assert_ne!(
                observed, "unattributed",
                "{tag}: {name} ran outside every declared phase — some driver \
                 is missing an enter_phase scope. Finding the SECOND driver \
                 cost a backtrace hunt; this is how the next one surfaces.\n{}",
                r.stderr
            );
            let allowed: Vec<String> =
                d.phases.iter().map(|p| format!("{p:?}")).collect();
            assert!(
                allowed.iter().any(|a| a == observed),
                "{tag}: {name} ran in phase {observed} but the registry declares \
                 {allowed:?}. Either the pass moved, or a driver needs an \
                 enter_phase scope.\n{}",
                r.stderr
            );
            // Nothing in the pipeline may declare an empty (OutOfBand) set and
            // still run here.
            assert!(
                !d.phases.is_empty(),
                "{tag}: {name} declares no phases (OutOfBand) but ran inside \
                 the compile pipeline:\n{}",
                r.stderr
            );
            phases_seen.insert(observed.to_string());
            checked += 1;
        }
    }
    assert!(checked >= 5, "only {checked} pass/phase pairs checked — too thin");
    // Anti-vacuity: if every pass reported the same phase, the check could not
    // distinguish a correct attribution from a constant.
    assert!(
        phases_seen.len() >= 2,
        "all observed passes shared one phase ({phases_seen:?}); the check \
         cannot tell attribution from a constant"
    );
}
