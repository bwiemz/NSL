//! FASE optim-step per-param dispatch: confirm the contract surface
//! that the unified dispatch helper relies on. Full end-to-end runtime
//! behavior is covered by the existing FASE test suite (fallback path)
//! plus the plan-level unit tests in fase.rs (two-phase-clip clamp).

#[test]
fn two_phase_clip_mixed_produces_conflict_diagnostics() {
    use nsl_codegen::fase::{plan_with_overrides, FaseConfig, FaseMode, FaseOptimizer};
    use nsl_codegen::wggo_overrides::OverrideRejectReason;

    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: Some(1.5),
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, false, true]);

    // All clamped to Deferred so Phase A's global norm stays valid.
    assert_eq!(p.per_layer_mode, Some(vec![FaseMode::Deferred; 4]));

    // Two diagnostics (layers 1 and 2 — the false inputs).
    assert_eq!(p.override_diagnostics.len(), 2);
    let layer_indices: Vec<u32> = p
        .override_diagnostics
        .iter()
        .map(|d| d.layer_index)
        .collect();
    assert_eq!(layer_indices, vec![1, 2]);
    for d in &p.override_diagnostics {
        assert!(matches!(
            d.reason,
            OverrideRejectReason::TwoPhaseClipConflict { grad_clip_threshold: t }
                if (t - 1.5).abs() < 1e-12
        ));
    }
}

#[test]
fn no_clip_preserves_mixed_modes_for_unified_dispatch() {
    use nsl_codegen::fase::{plan_with_overrides, FaseConfig, FaseMode, FaseOptimizer};

    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: None,
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, true, false]);

    // Mixed modes preserved — the unified dispatch helper branches on
    // these at runtime via the .rodata mode table.
    assert_eq!(
        p.per_layer_mode,
        Some(vec![
            FaseMode::Deferred,
            FaseMode::FullBuffer,
            FaseMode::Deferred,
            FaseMode::FullBuffer
        ])
    );
    assert!(p.override_diagnostics.is_empty());
}
