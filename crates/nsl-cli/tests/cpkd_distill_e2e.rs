//! CPKD distill-block end-to-end tests via the `nsl` CLI.
//!
//! Covers the v1 acceptance surface:
//! - `nsl check` accepts the basic and fused fixtures;
//! - loud refusals: attn_transfer=true, @fused_kl_ce on a train block,
//!   distill without teacher=, tape-fallback refusal is compile-side;
//! - `nsl run` executes a full distillation loop on CPU (composite CE+MSE
//!   proxy loss) and the loss DECREASES — proof the student learns while
//!   the frozen teacher provides a stable target (I-11);
//! - the Distillation Build Report renders on stderr with the fused
//!   KL-CE line when @fused_kl_ce fires.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

fn fixture_path(name: &str) -> PathBuf {
    workspace_root()
        .join("crates")
        .join("nsl-codegen")
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn write_temp_fixture(tag: &str, source: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nsl_cpkd_e2e_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let file = dir.join("fixture.nsl");
    std::fs::write(&file, source).expect("write fixture");
    file
}

#[test]
fn nsl_check_accepts_basic_distill_block() {
    let fixture = fixture_path("cpkd_distill_basic.nsl");
    assert!(fixture.exists(), "missing: {}", fixture.display());
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

#[test]
fn nsl_check_accepts_fused_kl_ce_distill_block() {
    let fixture = fixture_path("cpkd_distill_fused_kl_ce.nsl");
    assert!(fixture.exists(), "missing: {}", fixture.display());
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&fixture);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

/// attn_transfer = true is a DEFERRED feature and must refuse loudly at
/// the semantic layer (attention probabilities are never materialized in
/// HBM by the fused attention kernels).
#[test]
fn nsl_check_refuses_attn_transfer_true() {
    let src = r#"model M:
    w: Tensor = randn([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let teacher = M()
let student = M()

distill(teacher = teacher, student = student, epochs = 1):
    optimizer: SGD(lr = 0.01)
    loss:
        attn_transfer = true
    step(batch):
        let loss = student.forward(randn([4, 8]))
"#;
    let file = write_temp_fixture("attn", src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert().failure().stderr(predicate::str::contains(
        "attention transfer is not implemented in CPKD v1",
    ));
}

/// @fused_kl_ce is distill-block-only (mirrors @fused_lm_ce/train gating).
#[test]
fn nsl_check_refuses_fused_kl_ce_on_train_block() {
    let src = r#"model M:
    w: Tensor = randn([8, 8])

let m = M()

@fused_kl_ce(enabled = true)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let loss = randn([1])
"#;
    let file = write_temp_fixture("wrongtarget", src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert().failure().stderr(predicate::str::contains(
        "@fused_kl_ce may only be applied to a `distill` block",
    ));
}

#[test]
fn nsl_check_refuses_distill_without_teacher() {
    let src = r#"model M:
    w: Tensor = randn([8, 8])

let student = M()

distill(student = student, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let loss = student.forward(randn([4, 8]))
"#;
    let file = write_temp_fixture("noteacher", src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert().failure().stderr(predicate::str::contains(
        "distill block requires both teacher= and student=",
    ));
}

/// `grad_accumulation` is refused by VALUE, not clamped. The train path
/// clamps a non-literal window to 1 silently; on distill that clamp is
/// indistinguishable from the pre-fix behaviour (no FASE-Deferred envelope,
/// so no CPDT optimizer-moment precision), so a bad window has to say so.
///
/// The refusal is a single match arm — `IntLiteral(n) if n >= 1` — with two
/// ways to miss it, and the four spellings below do NOT split evenly across
/// them. In EXPRESSION position the parser never folds a sign into the
/// literal (only pattern position does), so `-2` is a `UnaryOp::Neg` wrapping
/// `2` and misses on SHAPE, exactly as `2.5` and `"two"` do. That leaves `0`
/// as the only witness of the VALUE half — it is the only integer literal
/// below the bound that can be spelled at all. It is still worth feeding all
/// four: `-2` is the likeliest thing a user actually types, and a fixture
/// whose window were simply ignored would pass none of them.
///
/// `0` alone cannot distinguish `n >= 1` from an off-by-one `n > 1`, which
/// would refuse every case here and still be wrong; the accepting side of the
/// boundary is pinned by the companion test below.
#[test]
fn nsl_check_refuses_a_non_positive_or_non_literal_grad_accumulation() {
    let base = std::fs::read_to_string(fixture_path("cpkd_distill_basic.nsl")).unwrap();
    for (tag, value) in [("zero", "0"), ("neg", "-2"), ("float", "2.5"), ("str", "\"two\"")] {
        let src = base.replace(
            "epochs = 1)",
            &format!("epochs = 1, grad_accumulation = {value})"),
        );
        assert_ne!(base, src, "fixture knobs not found — resync test");
        let file = write_temp_fixture(&format!("badaccum_{tag}"), &src);
        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", stdlib_path());
        cmd.arg("check").arg(&file);
        cmd.assert().failure().stderr(
            predicate::str::contains(
                "distill 'grad_accumulation' must be a positive integer literal",
            )
            // The key must not ALSO be reported as unknown: that would mean
            // the allowlist and the value check disagree about who owns it.
            .and(predicate::str::contains("unknown distill config key").not()),
        );
    }
}

/// The accepting side of the `n >= 1` boundary, and the control that gives
/// the refusal arm above its meaning: `grad_accumulation = 1` — the window
/// that means "no accumulation" — must CHECK. Written `n > 1`, the guard
/// would refuse every input the refusal test feeds it and pass that test
/// while silently rejecting the one window a user is most likely to write
/// explicitly.
#[test]
fn nsl_check_accepts_a_grad_accumulation_of_one() {
    let base = std::fs::read_to_string(fixture_path("cpkd_distill_basic.nsl")).unwrap();
    let src = base.replace("epochs = 1)", "epochs = 1, grad_accumulation = 1)");
    assert_ne!(base, src, "fixture knobs not found — resync test");
    let file = write_temp_fixture("accum_boundary_one", &src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

/// The accumulation window on a distill block must do what it does on a
/// train block: run the step body per MICRO-batch and fire the optimizer
/// only on window boundaries.
///
/// Anti-vacuity, which is the whole point of this arm: a test that merely
/// compiles the new key proves the parser changed and nothing else. This
/// counts the NSL_PHASE_TIMING instrumentation instead — `[phase] fwd=` is
/// emitted once per micro-batch and `[phase] opt=` once per optimizer step
/// (the accumulation gate skips that block on non-boundary micro-batches).
/// The fixture has no `data:` section, so one micro-batch per epoch: 8
/// micro-batches in every arm, and 8 / 4 / 2 optimizer steps at windows of
/// 1 / 2 / 4. Nothing but a live accumulation gate produces that spread.
///
/// The loss stream carries a second, independent witness. Every micro-batch
/// in a window sees the SAME parameters (the optimizer has not run), so its
/// losses are bit-identical within the window; and because each micro-batch
/// replays the same `x`, the 1/N-scaled accumulated gradient equals the
/// un-accumulated one — the window leaders must reproduce the window-of-1
/// trajectory EXACTLY. A wrong scale factor, or an optimizer that stepped
/// mid-window, breaks that equality while leaving the counts intact.
#[test]
fn distill_grad_accumulation_windows_the_optimizer_over_micro_batches() {
    let base = std::fs::read_to_string(fixture_path("cpkd_distill_basic.nsl")).unwrap();

    let run_at = |accum: usize| -> (usize, usize, Vec<f64>) {
        let src = base
            .replace(
                "epochs = 1)",
                &format!("epochs = 8, grad_accumulation = {accum})"),
            )
            // The shipped lr is deliberately tiny; at 0.05 the per-step move
            // is far above the f32 ulp of the loss, so "unchanged within a
            // window" is a real observation rather than a rounding artifact.
            .replace("lr = 0.001", "lr = 0.05");
        assert_ne!(base, src, "fixture knobs not found — resync test");
        let file = write_temp_fixture(&format!("accum{accum}"), &src);

        let mut cmd = Command::cargo_bin("nsl").unwrap();
        cmd.env("NSL_STDLIB_PATH", stdlib_path());
        // Read at COMPILE time by the train-block lowering, and `nsl run`
        // compiles and runs in one process — so setting it here arms the
        // instrumentation for this run.
        cmd.env("NSL_PHASE_TIMING", "1");
        cmd.arg("run").arg("--source-ad").arg(&file);
        let output = cmd.output().expect("nsl run failed to spawn");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "nsl run failed at accum={accum}.\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
        let micro = stdout
            .lines()
            .filter(|l| l.trim_start().starts_with("[phase] fwd="))
            .count();
        let opt = stdout
            .lines()
            .filter(|l| l.trim_start().starts_with("[phase] opt="))
            .count();
        let losses: Vec<f64> = stdout
            .lines()
            .filter_map(|l| l.trim().parse::<f64>().ok())
            .collect();
        (micro, opt, losses)
    };

    let (m1, o1, l1) = run_at(1);
    let (m2, o2, l2) = run_at(2);
    let (m4, o4, l4) = run_at(4);

    // The instrument itself must be alive — without this the three count
    // assertions below are all satisfied by 0 == 0.
    assert_eq!(m1, 8, "NSL_PHASE_TIMING emitted no per-micro-batch lines");
    assert_eq!((m2, m4), (8, 8), "the step body must run once per micro-batch \
         regardless of the window: got {m2} at accum=2, {m4} at accum=4");
    assert_eq!(
        (o1, o2, o4),
        (8, 4, 2),
        "optimizer steps must be micro-batches / window; got {o1}/{o2}/{o4} \
         at windows 1/2/4"
    );

    for (accum, losses) in [(1usize, &l1), (2, &l2), (4, &l4)] {
        assert_eq!(
            losses.len(),
            8,
            "expected 8 per-micro-batch loss prints at accum={accum}: {losses:?}"
        );
        assert!(
            losses.iter().all(|v| v.is_finite()),
            "non-finite loss at accum={accum}: {losses:?}"
        );
        // Within a window the parameters have not moved, so the loss cannot.
        for window in losses.chunks(accum) {
            assert!(
                window.iter().all(|v| *v == window[0]),
                "loss moved inside an accumulation window at accum={accum} \
                 (the optimizer stepped mid-window): {losses:?}"
            );
        }
        // Across windows it must, and downward.
        let leaders: Vec<f64> = losses.chunks(accum).map(|w| w[0]).collect();
        assert!(
            leaders.windows(2).all(|p| p[1] < p[0]),
            "the student stopped learning at accum={accum}: {leaders:?}"
        );
        // N identical micro-batches averaged at 1/N reproduce the
        // un-accumulated trajectory, sampled every N steps. The accum=1 arm
        // is EXCLUDED: there `leaders` IS `l1`, so the comparison is l1
        // against itself and holds however wrong the scale factor is. Two
        // arms witness this, not three.
        if accum > 1 {
            let expected: Vec<f64> = l1.iter().copied().take(leaders.len()).collect();
            assert_eq!(
                leaders, expected,
                "accum={accum} window leaders diverged from the window-of-1 \
                 trajectory — the 1/N accumulation scale is wrong"
            );
        }
    }
}

/// Full CPU distillation run: 8 epochs of the composite CE + logit-MSE
/// proxy loss. The loss stream must be finite and strictly lower at the
/// end than at the start (the student learns; the frozen teacher's
/// contribution is stable across epochs — its weights never move because
/// no teacher gradient exists, I-11).
#[test]
fn nsl_run_distill_loss_decreases() {
    let base = std::fs::read_to_string(fixture_path("cpkd_distill_basic.nsl")).unwrap();
    let scaled = base
        .replace("epochs = 1", "epochs = 8")
        .replace("lr = 0.001", "lr = 0.05");
    assert_ne!(base, scaled, "fixture knobs not found — resync test");
    let file = write_temp_fixture("run", &scaled);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("run").arg("--source-ad").arg(&file);
    let output = cmd.output().expect("nsl run failed to spawn");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "nsl run failed.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    let losses: Vec<f64> = stdout
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(
        losses.len() >= 8,
        "expected >= 8 per-epoch loss prints, got {} .\nstdout:\n{stdout}",
        losses.len()
    );
    assert!(
        losses.iter().all(|v| v.is_finite()),
        "non-finite loss: {losses:?}"
    );
    assert!(
        losses.last().unwrap() < losses.first().unwrap(),
        "loss did not decrease: {losses:?}"
    );
}

/// `epochs` gets the same by-value contract as `grad_accumulation`, for the
/// same reason and one step worse: a non-literal used to leave the count at
/// `1` in `compile_distill_block`, so `let n = 8` + `distill(..., epochs = n)`
/// COMPILED CLEAN, reported "Epochs: 1", and trained the student on an eighth
/// of the requested work.
///
/// The control is the `train` twin, which has always hard-errored on the
/// identical input — the asymmetry was the bug, not the strictness. Both arms
/// are the same program differing only in the block spelling, so the block
/// kind is the only thing that can explain a difference between them.
#[test]
fn a_non_literal_epochs_refuses_on_distill_exactly_as_it_does_on_train() {
    let base = std::fs::read_to_string(fixture_path("cpkd_distill_basic.nsl")).unwrap();
    let src = base
        .replace("distill(", "let n_ep = 8\ndistill(")
        .replace("epochs = 1)", "epochs = n_ep)");
    assert_ne!(base, src, "fixture knobs not found — resync test");
    assert!(src.contains("epochs = n_ep)"), "the epochs rewrite matched nothing");
    let file = write_temp_fixture("nonlit_epochs", &src);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check").arg(&file);
    cmd.assert().failure().stderr(
        predicate::str::contains("distill 'epochs' must be an integer literal")
            .and(predicate::str::contains("unknown distill config key").not()),
    );

    // Control: an integer literal still checks, so the guard is not simply
    // rejecting the key.
    let ok = base.replace("epochs = 1)", "epochs = 8)");
    let ok_file = write_temp_fixture("lit_epochs", &ok);
    let mut ok_cmd = Command::cargo_bin("nsl").unwrap();
    ok_cmd.env("NSL_STDLIB_PATH", stdlib_path());
    ok_cmd.arg("check").arg(&ok_file);
    ok_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("checked successfully"));
}

/// The Distillation Build Report states the accumulation window and the FASE
/// mode it selected.
///
/// It reported `Epochs: N` and stopped there, which is not the optimizer-step
/// count once a window exists, and said nothing about the decision the window
/// actually drives. `Passthrough` is the load-bearing word: it means there is
/// no Deferred envelope, and therefore that a CPDT optimizer-moment precision
/// plan will arbitrate to nothing however good the plan is — the exact failure
/// PR #484 fixed, which this report could not have explained.
///
/// Two arms, one token apart, so the window is demonstrably the cause.
#[test]
fn the_build_report_states_the_accumulation_window_and_its_fase_mode() {
    let fixture = fixture_path("cpdt_precision_fp16_distill.nsl");
    let dir = std::env::temp_dir().join(format!("nsl_cpkd_window_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg("--source-ad")
        .arg(&fixture)
        .arg("-o")
        .arg(dir.join("with_window"));
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "Accumulation: grad_accumulation=4 (FASE Deferred)",
        ))
        .stderr(predicate::str::contains("no window, so no Deferred envelope").not());

    let src = std::fs::read_to_string(&fixture).unwrap();
    let no_window = src.replace(", grad_accumulation = 4)", ")");
    assert_ne!(src, no_window, "the rewrite matched nothing — fixture drifted");
    let no_window_file = dir.join("no_window.nsl");
    std::fs::write(&no_window_file, no_window).unwrap();

    let mut off = Command::cargo_bin("nsl").unwrap();
    off.env("NSL_STDLIB_PATH", stdlib_path());
    off.arg("build")
        .arg("--source-ad")
        .arg(&no_window_file)
        .arg("-o")
        .arg(dir.join("without_window"));
    off.assert().success().stderr(
        predicate::str::contains("Accumulation: grad_accumulation=1 (FASE Passthrough)")
            .and(predicate::str::contains(
                "no window, so no Deferred envelope for a CPDT optimizer-moment \
                 precision plan to lower into",
            )),
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Diagnostics emitted from the SHARED `compile_train_block_inner` path must
/// name the block the user actually wrote. `--fuse-wgrad-accum` and
/// `--layerwise-accum` both refuse on a window of 1, and both told a distill
/// author to go edit a "train block" — a construct their program does not
/// contain — while never mentioning `distill`'s own `grad_accumulation` key,
/// which is the thing they need to set.
///
/// The train twin is the control: it must keep saying "train block", so the
/// selector is demonstrably reading the block kind rather than having been
/// globally reworded.
#[test]
fn window_diagnostics_name_the_distill_block_not_a_train_block() {
    let dir = std::env::temp_dir().join(format!("nsl_cpkd_noun_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let distill_src =
        std::fs::read_to_string(fixture_path("cpdt_precision_fp16_distill.nsl")).unwrap();
    let no_window = distill_src.replace(", grad_accumulation = 4)", ")");
    assert_ne!(distill_src, no_window, "the rewrite matched nothing");
    let distill_file = dir.join("distill_no_window.nsl");
    std::fs::write(&distill_file, no_window).unwrap();

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg("--source-ad")
        .arg("--fuse-wgrad-accum")
        .arg(&distill_file)
        .arg("-o")
        .arg(dir.join("distill_bin"));
    // Failure, not success: with no block fusing, the compile-scoped
    // admission refuses. What this arm pins is the WORDING on the way there.
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains(
            "[wgrad-fusion] declined: distill block #1",
        ))
        .stderr(predicate::str::contains(
            "the distill block omits grad_accumulation",
        ));

    // Control: the train twin keeps the train wording.
    let train_src =
        std::fs::read_to_string(fixture_path("cpdt_precision_fp16.nsl")).unwrap();
    let train_no_window = train_src.replace(", grad_accumulation = 4)", ")");
    assert_ne!(train_src, train_no_window, "the train rewrite matched nothing");
    let train_file = dir.join("train_no_window.nsl");
    std::fs::write(&train_file, train_no_window).unwrap();

    let mut train_cmd = Command::cargo_bin("nsl").unwrap();
    train_cmd.env("NSL_STDLIB_PATH", stdlib_path());
    train_cmd
        .arg("build")
        .arg("--source-ad")
        .arg("--fuse-wgrad-accum")
        .arg(&train_file)
        .arg("-o")
        .arg(dir.join("train_bin"));
    train_cmd
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "[wgrad-fusion] declined: train block #1",
        ))
        .stderr(predicate::str::contains("the train block omits grad_accumulation"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The WGGO pre-pass must count a distill block in its document order.
///
/// `walk_stmts` stamps `is_first_train_block` from a running counter, and its
/// `For`/`While` arms exist precisely because a block that yields no pre-plan
/// must still ADVANCE that counter — "otherwise a LATER top-level train block
/// wrongly inherits `is_first_train_block`, corrupting the 'first train block
/// governs admission' invariant every consumer in kernel.rs relies on". A
/// distill block yields no pre-plan either, and it was falling through to
/// `_ => {}`.
///
/// Two distill blocks make the ordinal the whole assertion: the notes must
/// read #1 then #2. If the counter did not advance, both would read #1 — which
/// is exactly the mutation this arm exists to catch, and no `contains` check
/// on a single-block program could see it.
#[test]
fn the_wggo_prepass_counts_distill_blocks_in_document_order() {
    let base = std::fs::read_to_string(fixture_path("cpdt_precision_fp16_distill.nsl")).unwrap();
    let (head, block) = base
        .split_once("distill(teacher = teacher")
        .expect("fixture header not found — resync test");
    let block = format!("distill(teacher = teacher{block}");
    let second = block.replace("epochs = 8, grad_accumulation = 4", "epochs = 2, grad_accumulation = 2");
    assert_ne!(block, second, "the second-block rewrite matched nothing");
    let file = write_temp_fixture("two_distill", &format!("{head}{block}\n{second}"));

    let dir = file.parent().unwrap().to_path_buf();
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg("--source-ad")
        .arg("--wggo")
        .arg("full")
        .arg(&file)
        .arg("-o")
        .arg(dir.join("two_distill_bin"));
    cmd.assert()
        .success()
        .stderr(predicate::str::contains(
            "[wggo] pre-pass: no pre-plan for training block #1 (distill block",
        ))
        .stderr(predicate::str::contains(
            "[wggo] pre-pass: no pre-plan for training block #2 (distill block",
        ));
}

/// The Distillation Build Report must render on stderr with the fused
/// KL-CE line and the I-11 freeze evidence when @fused_kl_ce fires.
#[test]
fn distillation_build_report_renders_fused_line() {
    let fixture = fixture_path("cpkd_distill_fused_kl_ce.nsl");
    let out_dir = std::env::temp_dir().join(format!("nsl_cpkd_report_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&out_dir);
    let out = out_dir.join("cpkd_fused_bin");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg("--source-ad")
        .arg(&fixture)
        .arg("-o")
        .arg(&out);
    cmd.assert()
        .success()
        .stderr(predicate::str::contains("=== CPKD Distillation Build Report ==="))
        .stderr(predicate::str::contains(
            "Fused KL-CE: teacher+student logits never materialized",
        ))
        .stderr(predicate::str::contains(
            "Teacher freeze (I-11): teacher backward structurally absent",
        ));
    let _ = std::fs::remove_dir_all(&out_dir);
}
