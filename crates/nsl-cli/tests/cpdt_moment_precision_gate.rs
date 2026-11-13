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
//! Medium tier = blocks.1..6 (reduced m, FP32 v). So exactly SIX buffers go
//! to reduced storage, which for a first moment is BF16 (see
//! `nsl_codegen::cpdt_precision_exec` for why not FP16, and why the second
//! moment is never narrowed). Pinning the count is deliberate: a join regression reports
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
/// effect: six reduced first-moment buffers, a healthy bus channel, and a run that
/// trains to completion.
#[test]
fn cpdt_moment_precision_activates_through_the_driver() {
    let mut cmd = cmd_with_full_stack();
    cmd.env("NSL_PASS_TRACE", "1");
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "[cpdt] optimizer-moment precision active (CPDT per-param plan): \
             6 first-moment buffer(s) in BF16 storage",
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
             6 first-moment buffer(s) in BF16 storage",
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
    std::fs::write(&wrong, serialize(&views, None).unwrap()).unwrap();

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
             6 first-moment buffer(s) in BF16 storage",
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
             6 first-moment buffer(s) in BF16 storage",
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

/// The gate's own premise: the fixture and its weights exist, and the tier
/// split documented in the fixture header still yields a MEDIUM tier — six
/// reduced buffers is a designed number, not an incidental one.
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

/// Writes a safetensors file of f32 tensors, each filled with one value.
fn write_filled_safetensors(path: &std::path::Path, tensors: &[(&str, &[usize], f32)]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, value) in tensors {
        let count: usize = shape.iter().product();
        let start = data.len();
        for _ in 0..count {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut json = serde_json::Value::Object(header).to_string().into_bytes();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&json);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// Reduced-precision moments must leave training where full-precision
/// moments leave it, for gradients of realistic size.
///
/// Every block weight here receives a gradient of exactly 1e-3 per element
/// (`x = ones[2, 8]`, `embed = ones`, loss = `sum(pred) * 6.25e-5`), so the
/// AdamW trajectory is known: the weight falls from 1.0 at close to `lr`
/// per step. The weights file puts blocks 1..6 in the plan's LOW tier,
/// which asks for an 8-bit `m` and a 16-bit `v`.
///
/// Stored as f16, `v` could not hold `(1 - beta2) * g^2 = 1e-9` at all: it
/// stayed zero, the step became `m_hat / eps`, and after 200 steps those
/// blocks sat at -0.85 where the f32 run reaches +0.82 (ten times the
/// displacement). The existing numerical test drives gradients of 0.1 to
/// 1, where that cannot show.
#[test]
fn reduced_moments_track_full_precision_at_a_realistic_gradient_size() {
    let tmp = tempfile::tempdir().unwrap();
    let src = r#"
model Block:
    w: Tensor = ones([64, 64])

    fn forward(self, h: Tensor) -> Tensor:
        return h @ self.w

    fn mean_w(self) -> Tensor:
        return mean(self.w)

model TinyNet:
    embed: Tensor = ones([8, 64])
    blocks: [Block; 8] = Block()
    final_norm: Tensor = ones([64])

    fn forward(self, x: Tensor) -> Tensor:
        let e = x @ self.embed
        let h = e * 0.0
        for block in self.blocks:
            h = h + block.forward(e)
        return h * self.final_norm

    fn probe(self) -> Tensor:
        for block in self.blocks:
            let t: Tensor = block.mean_w()
            print(t)
        return zeros([1])

let m = TinyNet()
let x = ones([2, 8])

train(model = m, epochs = 400, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = sum(pred) * 0.0000625

m.probe()
"#;
    let program = tmp.path().join("const_grad.nsl");
    std::fs::write(&program, src).unwrap();

    let block_names: Vec<String> = (0..8).map(|i| format!("blocks.{i}.w")).collect();
    let mut tensors: Vec<(&str, &[usize], f32)> =
        vec![("embed", &[8, 64], 1.0), ("final_norm", &[64], 1.0)];
    for (i, name) in block_names.iter().enumerate() {
        // Small-magnitude interior blocks score into the LOW tier; the
        // first and last blocks are pinned HIGH by position.
        let fill = if i == 0 || i == 7 { 1.0 } else { 2e-5 };
        tensors.push((name.as_str(), &[64, 64], fill));
    }
    let weights_path = tmp.path().join("low_tier.safetensors");
    write_filled_safetensors(&weights_path, &tensors);

    let run = |reduced: bool| -> (Vec<f64>, String) {
        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", workspace_root().join("stdlib"));
        cmd.current_dir(tmp.path());
        cmd.arg("run").arg(&program).arg("--target").arg("cpu").arg("--source-ad");
        if reduced {
            cmd.arg("--cpdt")
                .arg("full")
                .arg("--cpdt-num-gpus")
                .arg("1")
                .arg("--weights")
                .arg(&weights_path);
        }
        let out = cmd.output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "run failed:\n{stderr}");
        let means: Vec<f64> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .collect();
        assert_eq!(means.len(), 8, "expected one mean per block");
        (means, stderr)
    };

    let (full, _) = run(false);
    let (reduced, stderr) = run(true);
    assert!(
        stderr.contains("6 first-moment buffer(s) in BF16 storage"),
        "the reduced run did not put the six LOW-tier blocks in reduced \
         storage, so it proves nothing:\n{stderr}"
    );
    // 200 optimizer steps at lr 1e-3 on a constant gradient: about 0.18.
    for (i, &w) in full.iter().enumerate() {
        assert!((0.80..0.83).contains(&w), "f32 reference drifted: block {i} = {w}");
    }
    for (i, (&r, &f)) in reduced.iter().zip(&full).enumerate() {
        let (moved_reduced, moved_full) = (1.0 - r, 1.0 - f);
        assert!(
            (moved_reduced / moved_full - 1.0).abs() < 0.02,
            "block {i}: reduced-precision moments moved the weight by \
             {moved_reduced}, full precision by {moved_full}"
        );
    }
    // And the plan must reach the arithmetic: a BF16 first moment rounds,
    // so the six reduced blocks cannot land on the f32 run's exact value.
    // (An earlier form of this check trained the shared fixture at a loss
    // of 5e30, where `v` overflows to infinity and the Adam step is zero in
    // both runs; it told the two apart only because an f16 `m` overflowed
    // too.)
    for i in 1..7 {
        assert_ne!(
            reduced[i], full[i],
            "block {i} is reported in reduced storage but trained to the \
             bit-identical value of the f32 run"
        );
    }
}
