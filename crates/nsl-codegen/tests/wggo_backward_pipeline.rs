//! Merge gate for WGGO Phase 2.  Spec ref: §6.4.
//!
//! **BLOCKED — do not remove `#[ignore]` until all three blockers are resolved.**
//!
//! # Purpose
//!
//! This is the single end-to-end test that gates merging the entire WGGO Phase 2
//! stack (Tasks 14-28).  It drives `compile_and_calibrate` on the real
//! `wggo_attention_mlp_real.nsl` fixture, runs the calibration subprocess,
//! reads back `sidecar.wggo_head_gradients`, and checks each per-head score
//! against the hand-coded analytical reference in `wggo_reference.rs` within
//! 1 × 10⁻⁴ tolerance.
//!
//! # Blockers (all three must be resolved before un-ignoring)
//!
//! ## Blocker 1 — NSL fixture missing a `train` block (Task 26 scope)
//!
//! `compile_and_calibrate` fires the calibration harness only inside
//! `compile_train_block` (see `crates/nsl-codegen/src/stmt.rs:~3966`).
//! `wggo_attention_mlp_real.nsl` exposes `fn main()` only; it has no `train`
//! block.  The function will return `Err("calibration harness ran but produced
//! no sidecar…")`.
//!
//! **Fix**: add a minimal `train` block to `wggo_attention_mlp_real.nsl` that
//! uses `AttentionMLP` so the harness fires, OR extend `compile_and_calibrate`
//! to call the harness for non-train calibration runs.
//!
//! ## Blocker 2 — `WggoGradientHook` not registered in production path (Task 22 scope)
//!
//! In `stmt.rs:compile_train_block` the calibration registry is assembled from
//! AWQ projections only (`AwqCalibrationHook`).  `WggoGradientHook` is never
//! added.  Without it, `sidecar.wggo_head_gradients` will always be `None`
//! even after a successful subprocess run.
//!
//! **Fix**: after the AWQ-hook registration block in `stmt.rs:~3984`, check
//! `self.compile_options.calibration_grad_retention` (populated by
//! `run_pre_scan_phase`).  If it is non-empty, push a
//! `WggoGradientHook::new(targets)` into the registry alongside the AWQ hook.
//!
//! ## Blocker 3 — BSS readback is a Task-22 placeholder (Task 22 scope)
//!
//! `emit_calibration_scaffolding_object` (binary_codegen.rs:~2603) emits a
//! placeholder `stack_store(run_ptr, scratch_slot, 0)` for each
//! `__nsl_wggo_grad.*` symbol instead of calling the real
//! `emit_per_head_dot_abs_accum` reduction.  The running buffers stay at
//! all-zeros throughout the subprocess run.
//!
//! As a consequence, `CalibCtx::read_running_buffer_f64_as_f32` is called with
//! an empty `running_buffers_f64` map (it is only populated by the test-only
//! `set_running_buffer_f64` setter), triggering the
//! `debug_assert!(cfg!(test), ...)` guard and returning all-zero scores.
//! The actual vs reference comparison would then yield differences of O(1)
//! on every head, far outside the 1 × 10⁻⁴ tolerance.
//!
//! **Fix**: replace the placeholder in the scaffolding loop with real
//! Cranelift IR that calls `emit_per_head_dot_abs_accum` (or equivalent) for
//! each `__nsl_wggo_grad.*` symbol, accumulating `|dW · W|` per head into
//! the f64 running buffers.  Then populate `CalibCtx::running_buffers_f64`
//! from the subprocess output before `emit_finalize` is called (matching the
//! AWQ path where `nsl_awq_write_sidecar` reads back running buffers directly
//! from BSS in the subprocess binary).
//!
//! # How to un-ignore
//!
//! 1. Resolve Blocker 1 (add `train` block to fixture).
//! 2. Resolve Blocker 2 (register `WggoGradientHook` in `stmt.rs`).
//! 3. Resolve Blocker 3 (wire real BSS readback in scaffolding emitter).
//! 4. Remove the `#[ignore = "…"]` attribute from the test below.
//! 5. Run `cargo test -p nsl-codegen --test wggo_backward_pipeline -- --nocapture`.
//! 6. Verify all four per-head scores pass within 1e-4 tolerance.

use std::collections::HashMap;
use std::path::PathBuf;

#[path = "wggo_reference.rs"]
mod wggo_reference;
use wggo_reference::reference_wggo_head_scores;

// ---------------------------------------------------------------------------
// Path helpers (mirrors awq_full_pipeline.rs convention)
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/nsl-codegen
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..") // crates
        .join("..") // repo root
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests").join("fixtures").join(name)
}

// ---------------------------------------------------------------------------
// Safetensors helpers
// ---------------------------------------------------------------------------

fn read_safetensors_flat(path: &std::path::Path, tensor_name: &str) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("deserializing {}: {e}", path.display()));
    let tensor = tensors
        .tensor(tensor_name)
        .unwrap_or_else(|e| panic!("missing tensor {tensor_name} in {}: {e}", path.display()));
    assert_eq!(
        tensor.dtype(),
        safetensors::Dtype::F32,
        "{tensor_name} in {} must be f32",
        path.display()
    );
    tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

// ---------------------------------------------------------------------------
// Assertion helper
// ---------------------------------------------------------------------------

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch for {name}: actual {} vs expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol,
            "{name}[{i}]: actual={a} expected={e} diff={diff} tol={tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Merge gate test
// ---------------------------------------------------------------------------

/// End-to-end WGGO Phase 2 merge gate.  Spec §6.4.
///
/// Drives the full compilation + calibration subprocess pipeline and checks
/// per-head gradient importance scores against the analytical reference within
/// a 1 × 10⁻⁴ tolerance.
///
/// # Tolerance rationale
///
/// The spec (§4.6) mandates f64 accumulators for the running gradient buffers.
/// With f64 accumulators and f32 input data the rounding error is O(N × ε_f64)
/// where N ≤ dim² = 1024 and ε_f64 ≈ 1.1 × 10⁻¹⁶, giving an expected
/// deviation well below 10⁻¹².  The 10⁻⁴ guard is intentionally generous to
/// absorb any f32/f64 promotion boundary at the BSS read-back step.
///
/// # Current status
///
/// **BLOCKED** — see module-level comment for the three required fixes.
/// All infrastructure is in place; remove `#[ignore]` once the blockers land.
#[test]
#[ignore = "BLOCKED on three deferred items: (1) wggo_attention_mlp_real.nsl needs a train \
            block so compile_and_calibrate fires the harness; (2) WggoGradientHook must be \
            registered alongside AwqCalibrationHook in stmt.rs compile_train_block; \
            (3) Task 22 BSS readback placeholder must be replaced with real \
            emit_per_head_dot_abs_accum Cranelift IR so running buffers are non-zero. \
            See module-level comment in this file for full details."]
fn end_to_end_backward_subprocess_matches_analytical_reference() {
    let nsl_path = fixture("wggo_attention_mlp_real.nsl");
    let data_path = fixture("wggo_calib_data.safetensors");
    let weights_path = fixture("wggo_calib_weights.safetensors");

    // Verify fixtures exist so the ignore message is not the only feedback
    // when this test is re-enabled.
    assert!(
        nsl_path.exists(),
        "fixture missing: {}",
        nsl_path.display()
    );
    assert!(
        data_path.exists(),
        "fixture missing: {}",
        data_path.display()
    );
    assert!(
        weights_path.exists(),
        "fixture missing: {}",
        weights_path.display()
    );

    // ── Step 1: run the full compilation + calibration subprocess pipeline ──

    let sidecar = nsl_codegen::compile_and_calibrate(&nsl_path, &data_path, &weights_path)
        .expect("backward subprocess pipeline should run end-to-end without error");

    // ── Step 2: extract per-head scores from sidecar ─────────────────────────

    let actual_grads = sidecar.wggo_head_gradients.expect(
        "sidecar.wggo_head_gradients must be Some after WGGO hook ran; \
         if this panics, Blocker 2 (WggoGradientHook registration) is not resolved",
    );

    // ── Step 3: build analytical reference ───────────────────────────────────

    // Calibration data: key "calibration", shape [8, 4, 32] → flatten to [1024] f32.
    let calib = read_safetensors_flat(&data_path, "calibration");

    // Weights: four [32, 32] projections.
    let mut weights: HashMap<&str, Vec<f32>> = HashMap::new();
    for name in &[
        "AttentionMLP.q_proj",
        "AttentionMLP.k_proj",
        "AttentionMLP.v_proj",
        "AttentionMLP.o_proj",
    ] {
        weights.insert(*name, read_safetensors_flat(&weights_path, name));
    }

    // dim=32, num_heads=4 (from fixture AttentionMLP(dim=32, num_heads=4))
    let reference = reference_wggo_head_scores(&calib, &weights, 32, 4);

    // ── Step 4: compare per-head scores within tolerance ─────────────────────

    // The fixture has one layer key "AttentionMLP" — the model class name used
    // as the layer_key in the @wggo_target decorator (see fixture: @wggo_target
    // on AttentionMLP.forward with no explicit layer_key → defaults to class name).
    let layer_key = "AttentionMLP";

    let ref_scores = reference
        .get(layer_key)
        .unwrap_or_else(|| panic!("reference missing layer '{layer_key}'"));

    let actual_layer = actual_grads.by_layer.get(layer_key).unwrap_or_else(|| {
        panic!(
            "sidecar.wggo_head_gradients.by_layer missing layer '{layer_key}'; \
             available layers: {:?}",
            actual_grads.by_layer.keys().collect::<Vec<_>>()
        )
    });

    assert_close(
        &actual_layer.per_head_score,
        ref_scores,
        1e-4,
        layer_key,
    );

    // Sanity: exactly 4 heads (dim=32, head_dim=8).
    assert_eq!(
        actual_layer.per_head_score.len(),
        4,
        "expected 4 per-head scores (dim=32, num_heads=4)"
    );

    // Sanity: at least one batch observed.
    assert!(
        actual_layer.batches_observed > 0,
        "batches_observed must be > 0 after calibration run"
    );
}

// ---------------------------------------------------------------------------
// Smoke tests that DO run (infrastructure verification, no subprocess needed)
// ---------------------------------------------------------------------------

/// Confirm the fixture files exist at the expected paths so fixture regressions
/// are caught immediately rather than surfacing only when the ignore is removed.
#[test]
fn merge_gate_fixtures_exist_on_disk() {
    let nsl_path = fixture("wggo_attention_mlp_real.nsl");
    let data_path = fixture("wggo_calib_data.safetensors");
    let weights_path = fixture("wggo_calib_weights.safetensors");

    assert!(
        nsl_path.exists(),
        "merge-gate NSL fixture missing: {}\n\
         Re-run: cargo run --bin build_wggo_fixtures",
        nsl_path.display()
    );
    assert!(
        data_path.exists(),
        "calibration data fixture missing: {}",
        data_path.display()
    );
    assert!(
        weights_path.exists(),
        "weight fixture missing: {}",
        weights_path.display()
    );
}

/// Confirm the analytical reference returns the right shape for the merge-gate
/// dimensions (dim=32, num_heads=4).
#[test]
fn analytical_reference_produces_four_head_scores() {
    let data_path = fixture("wggo_calib_data.safetensors");
    let weights_path = fixture("wggo_calib_weights.safetensors");

    if !data_path.exists() || !weights_path.exists() {
        // If fixtures are absent, skip rather than panic — the fixture test above
        // already catches the missing-file case.
        return;
    }

    let calib = read_safetensors_flat(&data_path, "calibration");
    let mut weights: HashMap<&str, Vec<f32>> = HashMap::new();
    for name in &[
        "AttentionMLP.q_proj",
        "AttentionMLP.k_proj",
        "AttentionMLP.v_proj",
        "AttentionMLP.o_proj",
    ] {
        weights.insert(*name, read_safetensors_flat(&weights_path, name));
    }

    let reference = reference_wggo_head_scores(&calib, &weights, 32, 4);

    let scores = reference
        .get("AttentionMLP")
        .expect("reference must contain 'AttentionMLP' layer");
    assert_eq!(scores.len(), 4, "dim=32, num_heads=4 → 4 per-head scores");

    // Scores should be non-trivial (calib + weights are non-zero by construction).
    let nonzero = scores.iter().filter(|&&s| s.abs() > 1e-6).count();
    assert!(
        nonzero > 0,
        "at least one per-head score should be non-zero for non-trivial input"
    );
}
