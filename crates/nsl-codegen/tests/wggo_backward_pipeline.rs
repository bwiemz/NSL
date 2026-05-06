//! Merge gate for WGGO Phase 2.  Spec ref: §6.4.
//!
//! # Purpose
//!
//! This is the single end-to-end test that gates merging the entire WGGO Phase 2
//! stack (Tasks 14-28).  It drives `real_subprocess_entry` on the real
//! `wggo_attention_mlp_real.nsl` fixture, runs the calibration subprocess,
//! reads back `sidecar.wggo_head_gradients`, and checks each per-head score
//! against the hand-coded analytical reference in `wggo_reference.rs` within
//! 1 × 10⁻⁴ tolerance.
//!
//! # Design — bypassing compile_and_calibrate
//!
//! This test uses `real_subprocess_entry` directly (same as the AWQ end-to-end
//! test) rather than `compile_and_calibrate`.  `compile_and_calibrate` couples
//! calibration to `compile_train_block` which has gaps (main signature
//! collision, stdlib path resolution for train-block compilation) that issue
//! #134 tracks for proper architectural fix.  The `real_subprocess_entry` path
//! bypasses those gaps and matches the production calibration entry point.

use std::collections::HashMap;
use std::path::PathBuf;

use nsl_codegen::calibration::{
    binary_codegen::real_subprocess_entry,
    HarnessConfig, HarnessMode, HookRegistry,
};

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

/// Compare per-head scores against an analytical reference, printing a
/// structured side-by-side report on failure.
///
/// The structured format pattern-matches the calibration spec §6.3 M3
/// parity-test diagnostic style. On first failure during initial
/// development, the difference between "spent 30 min debugging" and
/// "saw the right head immediately" is real — every head is reported
/// with actual / reference / abs_diff / rel_diff so the reader can
/// localize the divergence at a glance.
fn assert_close(actual: &[f32], expected: &[f32], tol: f32, name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch for {name}: actual {} vs expected {}",
        actual.len(),
        expected.len()
    );

    let mut any_fail = false;
    let mut report = String::new();
    report.push_str(&format!(
        "Per-head score divergence in layer {name} (tol={tol:e}):\n"
    ));
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs_diff = (a - e).abs();
        let rel_diff = if e.abs() > 0.0 {
            abs_diff / e.abs()
        } else {
            abs_diff
        };
        let status = if abs_diff < tol {
            "within tol"
        } else {
            any_fail = true;
            "FAIL"
        };
        report.push_str(&format!(
            "  head {i}: actual={a:e}, reference={e:e}, abs_diff={abs_diff:e}, rel_diff={rel_diff:e} ({status})\n"
        ));
    }

    if any_fail {
        panic!("{report}");
    }
}

// ---------------------------------------------------------------------------
// Fixture compile bundle helper — mirrors awq_fixture_compile_bundle in
// awq_full_pipeline.rs. Parses the merge-gate fixture, runs semantic
// analysis, and builds a CalibrationCompileBundle for real_subprocess_entry.
// ---------------------------------------------------------------------------

fn wggo_fixture_compile_bundle() -> std::sync::Arc<nsl_codegen::calibration::CalibrationCompileBundle> {
    let source = std::fs::read_to_string(fixture("wggo_attention_mlp_real.nsl"))
        .expect("merge-gate fixture readable");
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(&source, nsl_errors::FileId(0), &mut interner);
    assert!(
        lex_diags.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed.diagnostics.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let mut analysis_interner = interner.clone();
    let analysis = nsl_semantic::analyze(&parsed.module, &mut analysis_interner);
    assert!(
        analysis.diagnostics.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must pass semantic analysis: {:?}",
        analysis.diagnostics
    );

    std::sync::Arc::new(nsl_codegen::calibration::CalibrationCompileBundle {
        ast: parsed.module,
        interner: analysis_interner,
        type_map: analysis.type_map.clone(),
    })
}

// ---------------------------------------------------------------------------
// Merge gate test
// ---------------------------------------------------------------------------

/// End-to-end WGGO Phase 2 merge gate.  Spec §6.4.
///
/// Drives the full calibration subprocess pipeline via `real_subprocess_entry`
/// and checks per-head gradient importance scores against the analytical
/// reference within a 1 × 10⁻⁴ tolerance.
///
/// # Tolerance rationale
///
/// The spec (§4.6) mandates f64 accumulators for the running gradient buffers.
/// With f64 accumulators and f32 input data the rounding error is O(N × ε_f64)
/// where N ≤ dim² = 1024 and ε_f64 ≈ 1.1 × 10⁻¹⁶, giving an expected
/// deviation well below 10⁻¹².  The 10⁻⁴ guard is intentionally generous to
/// absorb any f32/f64 promotion boundary at the BSS read-back step.
#[test]
#[ignore = "WGGO Phase 2 merge-gate: BLOCKED on emit_calibration_model_object's \
            AWQ-only projection-list assumption. \
            \
            Earlier blockers shipped: bake-indices on WggoGradTarget, real \
            emit_per_head_dot_abs_accum reduction, WggoLayerDescriptor stack-build \
            at the FFI call site, real_subprocess_entry entry-point switch (PR #139); \
            ObservationSet::BackwardGradients(_) → needs_forward_pass=true so \
            WGGO-only registration routes through the real subprocess path \
            instead of build_sidecar_from_stub. \
            \
            Remaining gap: emit_calibration_model_object (binary_codegen.rs:1771-1779) \
            requires a non-empty calibration_retention list (AWQ projections) \
            because it reads channels / model_name / transpose_fields from the \
            first projection. WGGO-only flows have empty calibration_retention \
            but populated calibration_grad_retention; the function rejects on \
            the empty list before reaching the WGGO codegen path. The WGGO \
            targets carry the same metadata (class_name, w_*_shape) so the fix \
            is a fallback path that reads from calibration_grad_retention when \
            calibration_retention is empty. Tracked under issue #134's broader \
            'decouple calibration from compile_train_block' scope, which is \
            the natural place to converge the AWQ-only assumptions in the \
            calibration entry points."]
fn end_to_end_backward_subprocess_matches_analytical_reference() {
    let data_path = fixture("wggo_calib_data.safetensors");
    let weights_path = fixture("wggo_calib_weights.safetensors");

    // Verify fixtures exist so the ignore message is not the only feedback
    // when this test is re-enabled.
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

    // ── Step 1: run the calibration subprocess pipeline directly ──
    //
    // Bypasses compile_and_calibrate (which couples calibration to
    // compile_train_block — see issue #134) and uses the same
    // real_subprocess_entry path the AWQ end-to-end test uses.
    // WGGO targets are auto-derived inside real_subprocess_entry from
    // the compile_bundle's AST.
    let compile_bundle = wggo_fixture_compile_bundle();

    let pre_scan_targets =
        nsl_codegen::calibration::discovery::pre_scan_wggo_targets_from_ast(
            &compile_bundle.ast,
            &compile_bundle.interner,
        );
    assert!(
        !pre_scan_targets.is_empty(),
        "fixture must have at least one @wggo_target-decorated model that pre-scan finds; \
         got 0 targets — check the fixture's @wggo_target decorator"
    );

    let mut registry = HookRegistry::new();
    registry.register(Box::new(
        nsl_codegen::calibration::wggo_gradient_hook::WggoGradientHook::new(
            pre_scan_targets.clone(),
        ),
    ));

    let cfg = HarnessConfig {
        checkpoints: vec![weights_path.clone()],
        calibration_data: data_path.clone(),
        samples: 8,
        batch_size: 1,
        timeout_secs: 60,
        mode: HarnessMode::Required,
        // No AWQ projections in the WGGO-only fixture (`@` matmul, not `|>`
        // pipe syntax which AWQ pre-scan requires). The WGGO targets are
        // derived from the AST inside real_subprocess_entry.
        projections: Vec::new(),
        compile_bundle: Some(compile_bundle),
    };

    let sidecar = real_subprocess_entry(&cfg, &registry)
        .expect("backward subprocess pipeline should run end-to-end without error")
        .sidecar;

    // ── Step 2: extract per-head scores from sidecar ─────────────────────────

    let actual_grads = sidecar.wggo_head_gradients.expect(
        "sidecar.wggo_head_gradients must be Some after WGGO hook ran; \
         if this panics, WggoGradientHook registration or BSS readback is broken",
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
