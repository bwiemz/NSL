//! Differential tests for mixed per-layer FASE modes under two-phase clip.
//!
//! WGGO's per-layer `fase_fused` used to be clamped to uniform Deferred
//! whenever `grad_clip` was set. These tests pin the machinery that made
//! honoring mixed modes possible:
//!
//! 1. On the source-AD hook path (the only one where a mode table exists),
//!    `fase_cb` accumulates EVERY param with the scaled window-mean
//!    convention, so Phase A's global norm is uniform across a mixed
//!    table; the dispatch REFUSES two-phase clip without the hook.
//! 2. Both dispatch arms apply the shared clip factor to their param's
//!    accumulated gradient before stepping.
//! 3. The FullBuffer arm zeroes its accumulation buffer after the stdlib
//!    step (the Deferred arm's recipe epilogue already zeroed m_partial).
//! 4. Bias-correction t: mixed Deferred-global tables feed the FullBuffer
//!    arm the window counter (lockstep with the Deferred arm);
//!    FullBuffer-GLOBAL tables keep the historical micro-batch counter
//!    (bit-compatible with the monolithic path).
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
//! (pinned by the tape premise test below). Tape training through
//! array-block fields also aborts in the runtime — pre-existing,
//! unrelated — so the blocks fixtures iterate with
//! `for block in self.blocks:` and run under --source-ad.

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
        .args(["-p", "nsl-cli"]);
    if cfg!(feature = "cuda") {
        // Feature-compatibility guard: an unfeatured `cargo run -p nsl-cli`
        // REBUILDS AND REPLACES target/debug/nsl as a non-CUDA binary while
        // the workspace suite runs, breaking every concurrently-running test
        // that spawns that path (phantom "CUDA support not compiled"
        // failures inside their trained programs). Forward cuda so the
        // spawned build matches the suite's binary (and is usually a no-op).
        cmd.args(["--features", "cuda"]);
    }
    cmd.args(["--", "run"])
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
    // Set the override knob on the tape run as a POSITIVE tripwire: the
    // knob only prints (applied OR ignored) when WGGO overrides actually
    // materialized for the train block. Its total silence proves no mode
    // table existed, independent of the consumption log message's wording.
    let (mtb, stderr) = run_fixture(
        "fase_tape_clip_simple.nsl",
        "tape_clip_out.nslm",
        &["--wggo", "greedy"],
        Some("1"),
    );
    assert!(
        !stderr.contains("NSL_FASE_FUSED_OVERRIDE"),
        "the fused-override knob fired on the tape path — WGGO overrides \
         (and thus mode tables) are now reachable without source AD; the \
         dispatch's two-phase non-hook refusal will reject clipped configs \
         until a tape-path fused accumulate exists, and this suite needs \
         real tape-path differentials:\n{stderr}"
    );
    assert!(
        !stderr.contains("consumed pre-solved plan"),
        "tape path unexpectedly consumed a WGGO plan:\n{stderr}"
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
/// monolithic FullBuffer path exactly. NOTE: Lion's sign-only update is
/// magnitude-invariant, so with this fixture's constant-sign gradients it
/// canNOT detect cross-window buffer contamination — it covers the lion
/// dispatch-arm shape and t-insensitivity only. The zeroing fix is pinned
/// by the SOAP test below (magnitude-sensitive).
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
            "{name} diverged between monolithic and mode-table FullBuffer"
        );
    }
}

/// SOAP (global-FullBuffer plan: soap → FaseOptimizer::Unknown) with a
/// mode table must reproduce the monolithic FullBuffer path bit-for-bit
/// across 4 windows. SOAP's update is magnitude- AND t-sensitive, so this
/// pins BOTH:
///  - fix (3): the dispatch FullBuffer arm zeroing its accumulation buffer
///    after the stdlib step (without it, windows 2-4 step with cross-window
///    gradient sums → magnitude divergence), and
///  - the t semantics: FullBuffer-GLOBAL tables keep the historical
///    micro-batch-t bias correction (window-t is reserved for mixed
///    Deferred-global tables; forcing it here diverged from monolithic by
///    ~8e-4 over 4 windows).
#[test]
fn mode_table_fullbuffer_soap_matches_monolithic() {
    let (oracle, _) = run_fixture(
        "fase_fullbuffer_soap_blocks.nsl",
        "soap_fullbuffer_out.nslm",
        &["--source-ad"],
        None,
    );
    let (mtb, stderr) = run_fixture(
        "fase_fullbuffer_soap_blocks.nsl",
        "soap_fullbuffer_out.nslm",
        &["--source-ad", "--wggo", "greedy"],
        Some("0,0,0"),
    );
    assert_override_applied(&stderr);
    for name in ["blocks[0].w", "blocks[1].w"] {
        assert_eq!(
            tensor(&oracle, name),
            tensor(&mtb, name),
            "{name} diverged between monolithic and mode-table SOAP \
             (buffer zeroing or t-override regression in the FullBuffer arm)"
        );
    }
    // Multi-window training must actually move the weights (guards against
    // a vacuous pass where the optimizer never stepped).
    assert!(
        tensor(&mtb, "blocks[1].w").iter().any(|v| (v - 1.0).abs() > 1e-6),
        "SOAP mode-table run never moved blocks[1].w from init"
    );
}
