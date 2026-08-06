//! The CPDT-sourced optimizer-moment precision path, end to end through the
//! train-block driver.
//!
//! # Why this gate exists
//!
//! Until 2026-08-06 this path was STRUCTURALLY DEAD, twice over, and no test
//! could tell:
//!
//! 1. The moment-precision consult read `bus.cpdt_plan` ~2.2k driver lines
//!    before `invoke_cpdt_if_enabled` published it, so on every fresh compile
//!    the consult saw an empty channel and CPDT-sourced precision stayed
//!    inactive. The pass bus proved it on a fully-flag-enabled compile:
//!    `cpdt_plan: published 1x, read 0x full, 1x empty` plus a
//!    `DEAD OUTPUT` finding. (Roadmap item 2's instrumentation — #462/#466 —
//!    is what made the defect visible; the fix offers the WGGO pre-plan to
//!    CPDT in `compile_train_block`, before the body compiles.)
//! 2. Even with a plan present, the param join matched NOTHING on any real
//!    model: `param_paths` are `<model var>.<field path>` (`m.blocks.0.w`)
//!    while the plan's names are weight-file keys (`blocks.0.w`). The active
//!    arm then reported "0 moment buffer(s) in FP16" — activation with no
//!    effect. `precision_for_path` now strips the leading model-variable
//!    segment as a fallback, the same accommodation the WGGO-side join makes
//!    via `wggo_graph::layer_prefix`.
//!
//! The existing coverage could not catch either: `run_wggo_cli.rs` pins flag
//! plumbing, and `cpdt_precision_optim_numerical.rs` drives the runtime cast
//! FFIs directly (its own header says the compiled path cannot be run there).
//! This gate is the missing end-to-end half: the real driver, the real
//! fixture, asserting the ACTIVATION and its exact effect.
//!
//! The fixture's tier split is designed (see `cpdt_precision_fp16.nsl`):
//! High tier = embed, final_norm, blocks.0.w, blocks.7.w (FP32/FP32 moments);
//! Medium tier = blocks.1..6 (FP16 m, FP32 v). So exactly SIX buffers go to
//! FP16 storage. Pinning the count is deliberate: a join regression reports
//! "0 buffer(s)" while still printing "active", which is precisely the shape
//! defect 2 had.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture() -> PathBuf {
    workspace_root()
        .join("crates/nsl-codegen/tests/fixtures/cpdt_precision_fp16.nsl")
}

fn weights() -> PathBuf {
    workspace_root()
        .join("crates/nsl-codegen/tests/fixtures/cpdt_precision_fp16_weights.safetensors")
}

fn cmd_with_full_stack() -> Command {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.arg("run")
        .arg(fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg("--weights")
        .arg(weights())
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    cmd
}

/// The full stack activates CPDT-sourced moment precision, with the designed
/// effect: six FP16 moment buffers, a healthy bus channel, and a run that
/// trains to completion.
#[test]
fn cpdt_moment_precision_activates_through_the_driver() {
    let mut cmd = cmd_with_full_stack();
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "[cpdt] optimizer-moment precision active (CPDT per-param plan): \
             6 moment buffer(s) in FP16 storage",
        ))
        // The channel the defect lived on: published once (the pre-plan
        // offer; the in-place site skips when the offer was accepted), read
        // full by the moment consult AND the stale-plan check, empty never.
        .stderr(predicate::str::contains(
            "[pass-bus] cpdt_plan: published 1x by CPDT, read 2x full, 0x empty",
        ))
        // The finding that exposed the defect must stay gone. (Plain
        // substring, not a `[marker]` token — the negative-needle registry
        // pins bracketed tokens only.)
        .stderr(predicate::str::contains("DEAD OUTPUT: cpdt_plan").not());
}

/// A pre-plan the fingerprint rejected, whose precision plan the moment
/// consult already consumed, must REFUSE — the moments are allocated and
/// their dtypes cannot be re-derived. Driven by the same test-knob
/// convention as `NSL_WGGO_FORCE_STALE_TABLE`, because a real fingerprint
/// drift cannot be engineered from a fixture.
#[test]
fn a_stale_precision_plan_refuses_rather_than_training_on_it() {
    let mut cmd = cmd_with_full_stack();
    cmd.env("NSL_CPDT_FORCE_STALE_PLAN", "1");
    cmd.assert().failure().stderr(predicate::str::contains(
        "refusing to execute a stale precision plan",
    ));
}

/// `--cpdt` without a WGGO plan used to skip planning SILENTLY — the code
/// carried a comment asking a CLI layer to warn, and nothing did. The
/// driver itself now says so.
#[test]
fn cpdt_without_wggo_warns_instead_of_silently_skipping() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.arg("run")
        .arg(fixture())
        .arg("--source-ad")
        .arg("--weights")
        .arg(weights())
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    cmd.assert().success().stderr(predicate::str::contains(
        "[cpdt] skipped: CPDT planning requires a WGGO plan",
    ));
}

/// The FP16 moments must CHANGE training — activation without effect is
/// theater, and the "active: N buffer(s)" line alone cannot rule it out
/// (the count comes from the same lists whose consumption it claims).
///
/// Instrument calibration, learned the hard way while building this gate: at
/// the shipped fixture's 2 epochs x lr=0.001 the FP16-m perturbation sits
/// BELOW the f32 ulp of theta~1.0 and the saved models come out
/// byte-identical with the mechanism fully working — an identity check there
/// "confirms theater" on correct code. 60 epochs at lr=0.05 puts the
/// perturbation orders of magnitude above ulp, so: models must DIFFER with
/// the plan active, and a divergent/NaN blowup would fail the success
/// asserts — differing-but-finite is exactly the dequant->step->quant
/// signature.
#[test]
fn fp16_moments_change_training_rather_than_being_theater() {
    let tmp = tempfile::tempdir().unwrap();
    let src = std::fs::read_to_string(fixture()).unwrap();
    let hot = src
        .replace("epochs = 2", "epochs = 60")
        .replace("lr = 0.001", "lr = 0.05");
    assert_ne!(src, hot, "the rewrite matched nothing — fixture drifted");
    let hot_path = tmp.path().join("hot.nsl");
    std::fs::write(&hot_path, hot).unwrap();

    let run = |dir: &std::path::Path, cpdt: bool| {
        std::fs::create_dir_all(dir).unwrap();
        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
        cmd.current_dir(dir);
        cmd.arg("run")
            .arg(&hot_path)
            .arg("--source-ad")
            .arg("--wggo")
            .arg("full")
            .arg("--weights")
            .arg(weights());
        if cpdt {
            cmd.arg("--cpdt").arg("full").arg("--cpdt-num-gpus").arg("2");
        }
        cmd.assert().success();
        std::fs::read(dir.join("cpdt_precision_out.nslm")).unwrap()
    };
    let with_plan = run(&tmp.path().join("on"), true);
    let without_plan = run(&tmp.path().join("off"), false);
    assert_eq!(
        with_plan.len(),
        without_plan.len(),
        "same model, same format — a size change means something else moved"
    );
    assert_ne!(
        with_plan, without_plan,
        "the CPDT FP16-moment run saved a model byte-identical to the F32 \
         run at a step count where FP16-m noise must exceed f32 ulp — the \
         precision plan is being reported active without reaching the \
         emitted arithmetic"
    );
}

/// The gate's own premise: the fixture and its weights exist, and the tier
/// split documented in the fixture header still yields a MEDIUM tier — six
/// FP16 buffers is a designed number, not an incidental one.
#[test]
fn the_fixture_these_gates_use_exists() {
    assert!(fixture().exists(), "missing {:?}", fixture());
    assert!(weights().exists(), "missing {:?}", weights());
    let src = std::fs::read_to_string(fixture()).unwrap();
    assert!(
        src.contains("Medium tier: blocks.1.w .. blocks.6.w"),
        "the fixture's documented tier split changed — re-derive the pinned \
         buffer count in cpdt_moment_precision_activates_through_the_driver"
    );
}
