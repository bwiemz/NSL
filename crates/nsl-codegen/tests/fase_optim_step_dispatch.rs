//! FASE optim-step per-param dispatch: confirm the contract surface
//! that the unified dispatch helper relies on. Full end-to-end runtime
//! behavior is covered by the existing FASE test suite (fallback path)
//! plus the plan-level unit tests in fase.rs and the mixed-mode
//! differential CLI tests in nsl-cli/tests/fase_mixed_clip_equivalence.rs.

#[test]
fn two_phase_clip_mixed_modes_are_honored() {
    use nsl_codegen::fase::{plan_with_overrides, FaseConfig, FaseMode, FaseOptimizer};

    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        grad_clip: Some(1.5),
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, false, true]);

    // WGGO's per-layer fase_fused survives two-phase clip verbatim: the
    // accumulation loop keeps a uniform window-mean convention for both
    // modes when clipping is active, and both unified-dispatch arms apply
    // the shared clip factor — there is nothing left to clamp.
    assert!(p.two_phase_clip);
    assert_eq!(
        p.per_layer_mode,
        Some(vec![
            FaseMode::Deferred,
            FaseMode::FullBuffer,
            FaseMode::FullBuffer,
            FaseMode::Deferred,
        ])
    );
    assert!(p.override_diagnostics.is_empty());
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
        Some(vec![FaseMode::Deferred, FaseMode::FullBuffer, FaseMode::Deferred, FaseMode::FullBuffer])
    );
    assert!(p.override_diagnostics.is_empty());
}
