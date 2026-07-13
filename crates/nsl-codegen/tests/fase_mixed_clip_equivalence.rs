//! Differential tests for mixed per-layer FASE modes under two-phase clip.
//!
//! WGGO's per-layer `fase_fused` used to be clamped to uniform Deferred
//! whenever `grad_clip` was set. These tests pin the machinery that made
//! honoring mixed modes possible:
//!
//! 1. The unified per-param dispatch's Phase A fuses the final
//!    micro-batch's accumulation on the non-hook (tape-AD) path — the
//!    standard accumulation loop is skipped on the stepping batch, and the
//!    monolithic Deferred Phase A was previously the only place that
//!    recovered it.
//! 2. Under `two_phase_clip` the FullBuffer accumulation arm uses the same
//!    scaled window-mean convention as Deferred, so the global norm is
//!    uniform, and both dispatch arms apply the shared clip factor.
//! 3. The FullBuffer arm zeroes its accumulation buffer after the stdlib
//!    step (the Deferred arm's recipe epilogue already zeroed m_partial).
//! 4. The FullBuffer arm's Adam/AdamW bias correction uses the window
//!    counter (same as the Deferred arm), not the micro-batch counter.
//!
//! Strategy: run the same fixture with and without a WGGO mode table (the
//! table is pinned via the NSL_FASE_FUSED_OVERRIDE diagnostic knob) and
//! compare saved checkpoints. The no-WGGO run takes the monolithic
//! emission paths whose numerics are already pinned by
//! fase_numerical_validation.rs — they are the oracle.
//!
//! Path notes: every mode-table run uses --source-ad. WGGO's plan is only
//! consumed by the train block when a Wengert list exists (source AD), so
//! `--wggo` without `--source-ad` produces NO overrides and NO mode table
//! (verified empirically) — the tape path cannot reach the unified
//! dispatch today. (Tape training through array-block fields also aborts
//! in the runtime — pre-existing, unrelated.) The dispatch helper's
//! non-hook Phase A branch is therefore defensive: reachable only if a
//! future path hands it a mode table with fase_hook_active=false.

mod common {
    include!("common/mod.rs");
}

use common::nslm_reader::read_nslm;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    p
}

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

/// Run `nsl run <fixture> [extra...]` in `workdir`, optionally with
/// NSL_FASE_FUSED_OVERRIDE set on the child process. Returns captured
/// stderr. Panics on non-zero exit.
fn nsl_run(
    fixture: &Path,
    workdir: &Path,
    extra: &[&str],
    fused_override: Option<&str>,
) -> String {
    let root = workspace_root();
    let mut cmd = std::process::Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run"])
        .arg(fixture)
        .current_dir(workdir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    for a in extra {
        cmd.arg(a);
    }
    if let Some(spec) = fused_override {
        cmd.env("NSL_FASE_FUSED_OVERRIDE", spec);
    } else {
        cmd.env_remove("NSL_FASE_FUSED_OVERRIDE");
    }
    let out = cmd.output().expect("failed to spawn cargo run nsl-cli");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "nsl run failed on {fixture:?} extra={extra:?} override={fused_override:?}:\n{stderr}"
    );
    stderr
}

/// Fetch a tensor from a checkpoint map, tolerating the model-variable
/// prefix.
fn tensor<'t>(tensors: &'t HashMap<String, Vec<f32>>, bare: &str) -> &'t Vec<f32> {
    let prefixed = format!("m.{bare}");
    tensors.get(bare).or_else(|| tensors.get(&prefixed)).unwrap_or_else(|| {
        panic!(
            "{bare} not in checkpoint; available keys: {:?}",
            tensors.keys().collect::<Vec<_>>()
        )
    })
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "tensor length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Run a fixture, return (checkpoint tensors, stderr).
fn run_fixture(
    fixture: &str,
    checkpoint: &str,
    extra: &[&str],
    fused_override: Option<&str>,
) -> (HashMap<String, Vec<f32>>, String) {
    let tmp = TempDir::new().expect("tempdir");
    let stderr = nsl_run(&fixture_path(fixture), tmp.path(), extra, fused_override);
    let ckpt = tmp.path().join(checkpoint);
    assert!(ckpt.exists(), "missing checkpoint after run:\n{stderr}");
    (read_nslm(&ckpt).expect("read nslm"), stderr)
}

/// When a run pins the mode table via the override knob, the knob must
/// report it was applied — otherwise the run silently tested the plan's
/// own pattern and the differential comparison is meaningless.
fn assert_override_applied(stderr: &str) {
    assert!(
        stderr.contains("NSL_FASE_FUSED_OVERRIDE applied"),
        "expected the fused-override knob to apply (layer-count mismatch?):\n{stderr}"
    );
}

/// Pin the path premise this suite relies on: `--wggo` WITHOUT
/// `--source-ad` consumes no plan (no Wengert list on the tape path), so
/// no overrides, no mode table — and training still matches the plain
/// oracle bit-for-bit. If WGGO ever gains tape-path consumption, this
/// starts failing and the suite needs real tape-path differentials.
#[test]
fn tape_wggo_produces_no_mode_table_and_is_behavior_neutral() {
    let (oracle, _) =
        run_fixture("fase_tape_clip_simple.nsl", "tape_clip_out.nslm", &[], None);
    let (mtb, stderr) = run_fixture(
        "fase_tape_clip_simple.nsl",
        "tape_clip_out.nslm",
        &["--wggo", "greedy"],
        None,
    );
    assert!(
        !stderr.contains("consumed pre-solved plan"),
        "tape path unexpectedly consumed a WGGO plan — mode tables are now \
         reachable without source AD; extend this suite with tape-path \
         differentials:\n{stderr}"
    );
    assert_eq!(
        tensor(&oracle, "w"),
        tensor(&mtb, "w"),
        "w diverged between plain and --wggo tape runs (both should take \
         the identical monolithic path)"
    );
}

/// SOURCE-AD (hook) path: uniform-Deferred mode table vs monolithic — the
/// real pretraining configuration's dispatch shape.
#[test]
fn source_ad_uniform_deferred_matches_monolithic_under_clip() {
    let (oracle, _) = run_fixture(
        "fase_mixed_clip_blocks.nsl",
        "mixed_clip_out.nslm",
        &["--source-ad"],
        None,
    );
    let (mtb, stderr) = run_fixture(
        "fase_mixed_clip_blocks.nsl",
        "mixed_clip_out.nslm",
        &["--source-ad", "--wggo", "greedy"],
        Some("1,1,1"),
    );
    assert_override_applied(&stderr);
    for name in ["blocks[0].w", "blocks[1].w"] {
        assert_eq!(
            tensor(&oracle, name),
            tensor(&mtb, name),
            "{name} diverged between monolithic and mode-table Deferred"
        );
    }
}

/// SOURCE-AD mixed table under clip: the Deferred-mode block must stay
/// bit-identical to the oracle; the FullBuffer-mode block steps through
/// the stdlib AdamW with the same clipped window-mean gradient and
/// window-t bias correction — the same mathematical update as the
/// Deferred recipe, so agreement is tight but not bit-exact (op order
/// differs).
///
/// Failure signatures this discriminates:
///  - missing clip factor in the FullBuffer arm → blocks.1 drifts by the
///    unclipped/clipped ratio (orders of magnitude beyond 1e-6 here);
///  - raw-sum (unscaled) accumulation for the FullBuffer param → 2x
///    effective gradient;
///  - micro-batch-t bias correction → visibly different warmup steps;
///  - missing buffer zeroing → cross-window accumulation blowup.
#[test]
fn source_ad_mixed_table_under_clip_matches_deferred_oracle() {
    let (oracle, _) = run_fixture(
        "fase_mixed_clip_blocks.nsl",
        "mixed_clip_out.nslm",
        &["--source-ad"],
        None,
    );
    let (mix, stderr) = run_fixture(
        "fase_mixed_clip_blocks.nsl",
        "mixed_clip_out.nslm",
        &["--source-ad", "--wggo", "greedy"],
        Some("1,1,0"),
    );
    assert_override_applied(&stderr);

    assert_eq!(
        tensor(&oracle, "blocks[0].w"),
        tensor(&mix, "blocks[0].w"),
        "blocks.0.w (Deferred mode) must be bit-identical to the oracle"
    );
    let diff = max_abs_diff(tensor(&oracle, "blocks[1].w"), tensor(&mix, "blocks[1].w"));
    assert!(
        diff <= 1e-9,
        "blocks.1.w (FullBuffer mode) drifted {diff} from the Deferred \
         oracle — clip factor / scaling / window-t mismatch in the \
         FullBuffer dispatch arm"
    );
    // The FullBuffer arm must have actually trained the block (a dropped
    // step or a zeroed gradient would leave w at its init value of 1.0).
    assert!(
        tensor(&mix, "blocks[1].w").iter().any(|v| (v - 1.0).abs() > 1e-6),
        "blocks.1.w never moved from init — FullBuffer arm did not step"
    );
}

/// No clamp diagnostics may appear: mixed modes under clip are honored,
/// not clamped. (The old behavior printed `[fase] ... wggo-override-
/// rejected ... two_phase_clip_threshold_...` for every fase_fused=false
/// layer.)
#[test]
fn mixed_table_under_clip_emits_no_clamp_diagnostics() {
    let (_, stderr) = run_fixture(
        "fase_mixed_clip_blocks.nsl",
        "mixed_clip_out.nslm",
        &["--source-ad", "--wggo", "greedy"],
        Some("1,1,0"),
    );
    assert_override_applied(&stderr);
    assert!(
        !stderr.contains("two_phase_clip_threshold"),
        "unexpected two-phase-clip clamp diagnostic:\n{stderr}"
    );
}

/// Lion (global-FullBuffer plan) with a mode table must reproduce the
/// monolithic FullBuffer path exactly across multiple windows. This pins
/// fix (3): without the FullBuffer arm zeroing its accumulation buffer,
/// windows 2-4 step with gradients accumulated across windows.
#[test]
fn mode_table_fullbuffer_lion_matches_monolithic() {
    let (oracle, _) = run_fixture(
        "fase_fullbuffer_lion_blocks.nsl",
        "lion_fullbuffer_out.nslm",
        &["--source-ad"],
        None,
    );
    let (mtb, stderr) = run_fixture(
        "fase_fullbuffer_lion_blocks.nsl",
        "lion_fullbuffer_out.nslm",
        &["--source-ad", "--wggo", "greedy"],
        Some("0,0,0"),
    );
    assert_override_applied(&stderr);
    for name in ["blocks[0].w", "blocks[1].w"] {
        assert_eq!(
            tensor(&oracle, name),
            tensor(&mtb, name),
            "{name} diverged between monolithic and mode-table FullBuffer \
             (accumulation buffer not zeroed between windows?)"
        );
    }
}
