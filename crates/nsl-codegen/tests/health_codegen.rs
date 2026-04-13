//! Dev Tools Phase 4 Task 4: verify `compile_train_block` exposes a
//! `health_monitor` option that opts into per-step health hook emission.
//!
//! Runtime verification that the emitted calls fire is Task 7's manual
//! smoke — this file only proves:
//!   1. The option exists and defaults cleanly to `false` / `None`.
//!   2. Turning it on through `CompileOptions` does not break the
//!      Phase 2 pre-pass codegen entry (smoke-level; full train IR is
//!      only exercised by the `nsl build` path in Task 7).
#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;

#[test]
fn health_monitor_default_is_false() {
    let opts = CompileOptions::default();
    assert!(
        !opts.health_monitor,
        "health_monitor must default to false to keep codegen byte-identical"
    );
    assert_eq!(
        opts.health_flush_interval, None,
        "health_flush_interval must default to None"
    );
}

#[test]
fn health_monitor_flag_round_trips_through_compile_options() {
    // Pure struct-level smoke: the flag survives clone and respects
    // explicit interval overrides.
    let mut opts = CompileOptions::default();
    opts.health_monitor = true;
    opts.health_flush_interval = Some(250);

    let cloned = opts.clone();
    assert!(cloned.health_monitor);
    assert_eq!(cloned.health_flush_interval, Some(250));
}

#[test]
fn pre_pass_with_health_monitor_does_not_regress() {
    // The pre-pass codegen entry (run_pre_pass_only) is what the Phase 2
    // tests exercise — verify enabling health_monitor alongside it doesn't
    // poison the entry.  Real train-block IR emission is exercised by
    // Task 7's runtime smoke.
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    let y = matmul(x, W)
    return y
"#;
    let mut opts = CompileOptions::default();
    opts.health_monitor = true;
    opts.health_flush_interval = Some(100);
    opts.profile_kernels = true;

    let result = nsl_codegen::test_helpers::run_pre_pass_only(src, &opts)
        .expect("pre-pass should succeed with health_monitor enabled");
    assert!(
        result.manifest_builder_set,
        "profile_kernels pre-pass must still run cleanly when health_monitor is on"
    );
}
