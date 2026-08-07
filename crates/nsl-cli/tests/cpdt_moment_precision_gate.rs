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

fn cmd_without_wggo(fixture_path: PathBuf) -> Command {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.arg("run")
        .arg(fixture_path)
        .arg("--source-ad")
        .arg("--weights")
        .arg(weights())
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    cmd
}

/// `--cpdt` without a WGGO plan used to skip planning entirely (first
/// silently, then — after #470 — with a "[cpdt] skipped" notice). The
/// precision plan is a pure function of the weight map, so the skip was
/// never necessary: the wrapper now plans WEIGHTS-ONLY on the no-pre-plan
/// path and the moments get their designed dtypes with no WGGO plan at all.
#[test]
fn cpdt_without_wggo_activates_weights_only() {
    let mut cmd = cmd_without_wggo(fixture());
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "[cpdt] planned without a WGGO plan for this block (weights-only)",
        ))
        .stderr(predicate::str::contains(
            "[cpdt] optimizer-moment precision active (CPDT per-param plan): \
             6 moment buffer(s) in FP16 storage",
        ))
        // One publish (the wrapper's weights-only offer), read full by the
        // consult AND the weights-only staleness re-check; never empty.
        .stderr(predicate::str::contains(
            "[pass-bus] cpdt_plan: published 1x by CPDT, read 2x full, 0x empty",
        ))
        // The pre-weights-only skip notice must be gone, and the channel
        // must be healthy — the read_before_publish invariant on cpdt_plan
        // is Enforced on the strength of exactly this path.
        .stderr(predicate::str::contains("[cpdt] skipped").not())
        .stderr(predicate::str::contains("DEAD OUTPUT: cpdt_plan").not())
        .stderr(predicate::str::contains("READ BEFORE PUBLISH: cpdt_plan").not());
}

/// The weights-only path has the same staleness discipline as the pre-plan
/// path: the moments are allocated, so a diverging re-derivation refuses.
/// The forced-stale knob drives the arm; the needle pins WHICH refusal
/// fired — the weights-only one, not the pre-plan-fingerprint one.
#[test]
fn a_stale_weights_only_plan_refuses_rather_than_training_on_it() {
    let mut cmd = cmd_without_wggo(fixture());
    cmd.env("NSL_CPDT_FORCE_STALE_PLAN", "1");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "refusing to execute a stale precision plan",
        ))
        .stderr(predicate::str::contains("weights-only CPDT offer"));
}

fn loop_fixture() -> PathBuf {
    workspace_root()
        .join("crates/nsl-codegen/tests/fixtures/cpdt_precision_fp16_loop.nsl")
}

/// The weights-only path cannot run `cpdt_sensitivity::validate` (there is
/// no AppliedPlan to check the WeightMap against), so a checkpoint naming a
/// DIFFERENT model used to sail through as "active: 0 moment buffer(s)" —
/// activation with no effect, the exact shape the join defect had (this
/// file's module doc calls it defect 2). The consult now refuses a
/// non-empty precision plan that joins zero of the block's params.
#[test]
fn a_wrong_checkpoint_refuses_instead_of_activating_nothing() {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::collections::HashMap;

    let tmp = tempfile::tempdir().unwrap();
    let wrong = tmp.path().join("wrong_model.safetensors");
    // Names that join NOTHING in the fixture's model (embed/blocks.N.w/
    // final_norm), with values in the Medium-tier range so the plan is
    // non-empty and carries sub-32 decisions — the maximally-misleading
    // wrong checkpoint.
    let mut raw: HashMap<String, Vec<u8>> = HashMap::new();
    for name in ["foo.0.w", "foo.1.w"] {
        raw.insert(
            name.to_string(),
            (0..64 * 64).flat_map(|_| 1e-4_f32.to_le_bytes()).collect(),
        );
    }
    let views: HashMap<String, TensorView<'_>> = raw
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                TensorView::new(Dtype::F32, vec![64, 64], v.as_slice()).unwrap(),
            )
        })
        .collect();
    std::fs::write(&wrong, serialize(&views, &None).unwrap()).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.arg("run")
        .arg(fixture())
        .arg("--source-ad")
        .arg("--weights")
        .arg(&wrong)
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "none of its parameter names join this train block's parameters",
        ))
        .stderr(predicate::str::contains("0 moment buffer(s)").not());
}

/// The structural case the weights-only offer exists for: a train block whose
/// model variable is LOOP-BOUND (a `for` over a fixed model array), which the
/// WGGO prepass cannot type — so no pre-plan can ever exist. Before the
/// weights-only offer this printed the not-lowered notice and trained with
/// FP32 moments; now the moments get their designed dtypes, the in-place
/// WGGO site re-plans with the full model afterwards (the second publish),
/// and the final-vs-final re-arbitration AGREES — empirical proof that the
/// weights-only offer's precision equals the final plan's, which is exactly
/// the property that makes the pre-body offer sound.
#[test]
fn a_loop_bound_train_block_activates_weights_only() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.arg("run")
        .arg(loop_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg("--weights")
        .arg(weights())
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "[cpdt] planned without a WGGO plan for this block (weights-only)",
        ))
        .stderr(predicate::str::contains(
            "[cpdt] optimizer-moment precision active (CPDT per-param plan): \
             6 moment buffer(s) in FP16 storage",
        ))
        // Wrapper weights-only publish + post-body full-model re-publish;
        // consult read + re-arbitration read. A refusal here would mean the
        // weights-only precision diverged from the final plan's — the
        // property this arm exists to pin.
        .stderr(predicate::str::contains(
            "[pass-bus] cpdt_plan: published 2x by CPDT, read 2x full, 0x empty",
        ))
        .stderr(predicate::str::contains("DEAD OUTPUT: cpdt_plan").not())
        .stderr(predicate::str::contains("READ BEFORE PUBLISH: cpdt_plan").not());
}

/// On the loop-bound path the forced-stale knob must produce the refusal
/// that names the REAL cause — the weights-only offer preceding the
/// in-place plan — not the pre-plan-fingerprint message (no pre-plan was
/// ever offered; naming one sends the user at the wrong artifact).
#[test]
fn a_stale_loop_bound_plan_refuses_with_the_in_place_cause() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.env("NSL_CPDT_FORCE_STALE_PLAN", "1");
    cmd.arg("run")
        .arg(loop_fixture())
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg("--weights")
        .arg(weights())
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "refusing to execute a stale precision plan",
        ))
        .stderr(predicate::str::contains(
            "before this block's in-place WGGO plan existed",
        ))
        .stderr(predicate::str::contains("graph fingerprint no longer matches").not());
}

fn distill_fixture() -> PathBuf {
    workspace_root()
        .join("crates/nsl-codegen/tests/fixtures/cpdt_precision_fp16_distill.nsl")
}

/// Distill's synthetic train block reaches the same wrapper and gets the
/// same weights-only offer. Until the distill header could carry
/// `grad_accumulation` the synthetic config was EMPTY, so the block planned
/// FASE Passthrough, the Deferred envelope never existed, and arbitration
/// lowered nothing — planning with no effect, the same shape as this file's
/// defect 2. With the window forwarded the student's moments get their
/// designed dtypes.
///
/// Six buffers, not twelve: the teacher's params are frozen Input leaves
/// (I-11) and are not in the block's parameter list at all, so the join sees
/// the student only. A regression that started training the teacher would
/// show up here as a changed count.
#[test]
fn a_distill_block_activates_weights_only_under_the_accumulation_window() {
    let mut cmd = cmd_without_wggo(distill_fixture());
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "[cpdt] planned without a WGGO plan for this block (weights-only)",
        ))
        .stderr(predicate::str::contains(
            "[cpdt] optimizer-moment precision active (CPDT per-param plan): \
             6 moment buffer(s) in FP16 storage",
        ))
        // Same channel health as the train-block weights-only arm: one
        // publish (the wrapper's offer), read full by the consult and the
        // staleness re-check, never empty.
        .stderr(predicate::str::contains(
            "[pass-bus] cpdt_plan: published 1x by CPDT, read 2x full, 0x empty",
        ))
        .stderr(predicate::str::contains("arbitration lowered nothing").not())
        .stderr(predicate::str::contains("DEAD OUTPUT: cpdt_plan").not());
}

/// The control for the arm above, and the reason it means anything: the SAME
/// fixture with the window token deleted must plan identically and activate
/// NOTHING. One token is the only difference, so the parameter join, the
/// weight map and the tier split are all held fixed — the window is the
/// cause. (Comparing two different distill fixtures could not say that.)
#[test]
fn the_same_distill_block_stays_fp32_when_the_window_is_removed() {
    let tmp = tempfile::tempdir().unwrap();
    let src = std::fs::read_to_string(distill_fixture()).unwrap();
    let no_window = src.replace(", grad_accumulation = 4)", ")");
    assert_ne!(src, no_window, "the rewrite matched nothing — fixture drifted");
    let path = tmp.path().join("no_window.nsl");
    std::fs::write(&path, no_window).unwrap();

    let mut cmd = cmd_without_wggo(path);
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "[cpdt] planned without a WGGO plan for this block (weights-only)",
        ))
        .stderr(predicate::str::contains("arbitration lowered nothing"))
        .stderr(predicate::str::contains("optimizer-moment precision active").not());
}

/// `--cpdt-report` on a weights-only build must not present zero-model
/// shard math as a recommendation: the ZeRO section's numbers describe an
/// empty cost model (0.00 GB per GPU), and pre-caveat they rendered
/// exactly like a real plan. The NOTE sits between the Mode line and the
/// numbers it disclaims.
#[test]
fn the_report_caveats_the_zero_model_zero_halves() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
    cmd.current_dir(tmp.path());
    cmd.arg("build")
        .arg(fixture())
        // -o into the temp dir. WITHOUT it `nsl build` derives the output
        // path from the SOURCE stem, so this arm dropped a ~128 MB unstripped
        // ELF into crates/nsl-codegen/tests/fixtures/ — extensionless, and
        // therefore sitting among the .nsl files where `git add -A` stages it
        // (it reached a commit once). `current_dir(tmp)` does not help,
        // because the default is relative to the source, not the cwd.
        .arg("-o")
        .arg(tmp.path().join("cpdt_report_probe"))
        .arg("--source-ad")
        .arg("--weights")
        .arg(weights())
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("2")
        .arg("--cpdt-report");
    cmd.assert().success().stdout(
        predicate::str::contains(
            "NOTE: planned without a WGGO plan (weights-only).",
        )
        .and(predicate::str::contains("=== CPDT Training Plan ===")),
    );
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
    assert!(loop_fixture().exists(), "missing {:?}", loop_fixture());
    let src = std::fs::read_to_string(fixture()).unwrap();
    assert!(
        src.contains("Medium tier: blocks.1.w .. blocks.6.w"),
        "the fixture's documented tier split changed — re-derive the pinned \
         buffer count in cpdt_moment_precision_activates_through_the_driver"
    );
    // The loop twin must keep joining the SAME weights file: its param paths
    // are `member.<key>` and the one-segment strip is what maps them onto
    // the safetensors keys. A renamed loop variable is fine; a deeper
    // nesting (two prefixes) would silently re-open defect 2's shape.
    let loop_src = std::fs::read_to_string(loop_fixture()).unwrap();
    assert!(
        loop_src.contains("for member in ens.members:")
            && loop_src.contains("train(model = member"),
        "the loop fixture's binding shape changed — verify the param-path \
         prefix still strips to the safetensors keys"
    );
}
