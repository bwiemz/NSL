//! FASE Codegen Phase 2: confirm the fallback path emits zero new IR
//! when no WGGO overrides are active.
//!
//! These two tests pin the fallback contract structurally without
//! requiring full-compile infrastructure. The end-to-end "no symbol
//! in object file" property follows: if `build_param_mode_table`
//! returns `None`, the allocation site in `compile_train_block` never
//! declares the rodata symbol, so it cannot appear in the emitted
//! object.

#[test]
fn fallback_path_when_overrides_absent() {
    use nsl_codegen::fase::{plan, FaseConfig, FaseMode, FaseOptimizer};
    use nsl_codegen::fase_codegen_table::build_param_mode_table;

    let mut p = plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    p.per_layer_mode = Some(vec![FaseMode::Deferred, FaseMode::FullBuffer]);

    let paths: Vec<String> = vec!["m.blocks.0.wq".into(), "m.blocks.1.wq".into()];

    // No overrides → None → caller skips rodata emission entirely.
    assert!(build_param_mode_table(&paths, "m", &p, None).is_none());
}

#[test]
fn fallback_path_when_per_layer_mode_none() {
    use nsl_codegen::fase::{plan, FaseConfig, FaseOptimizer};
    use nsl_codegen::fase_codegen_table::build_param_mode_table;
    use nsl_codegen::wggo_apply::AppliedPlan;
    use nsl_codegen::wggo_overrides::WggoOverrides;

    let p = plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    assert!(p.per_layer_mode.is_none());

    let empty_overrides = WggoOverrides::from_applied(&AppliedPlan {
        layers: vec![],
        total_us: 0.0,
        peak_memory_bytes: 0,
    });
    let paths: Vec<String> = vec!["m.blocks.0.wq".into()];

    // per_layer_mode is None → None → caller skips rodata emission.
    assert!(build_param_mode_table(&paths, "m", &p, Some(&empty_overrides)).is_none());
}
