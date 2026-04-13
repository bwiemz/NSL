//! Dev Tools Phase 2, Task 4: verify the kernel-profile pre-pass populates
//! `Compiler::prediction_map` and `Compiler::manifest_builder` when
//! `CompileOptions.profile_kernels` is true, and is a no-op when it is false.

#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;

#[test]
fn pre_pass_populates_prediction_map_when_profile_kernels_enabled() {
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    let y = matmul(x, W)
    return y
"#;
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    // target_gpu/dtype defaults ("h100"/"bf16") are fine.

    let result = nsl_codegen::test_helpers::run_pre_pass_only(src, &opts)
        .expect("pre-pass should succeed");
    assert!(
        result.prediction_map_len > 0,
        "prediction_map should be non-empty for a module with at least one matmul \
         (got len={})",
        result.prediction_map_len
    );
    assert!(
        result.manifest_builder_set,
        "manifest_builder must be set when profile_kernels is true"
    );
}

#[test]
fn pre_pass_api_surface_with_profile_kernels_enabled() {
    // Task 6: confirm the pre-pass helper doesn't panic when profile_kernels
    // is set.  Full source_text population is a side-effect of the entry
    // function, not the pre-pass alone; end-to-end manifest writes are
    // covered by Task 9's E2E test.
    let src = r#"
fn forward(x: Tensor<[1, 2048, 512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    return matmul(x, W)
"#;
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    let _result = nsl_codegen::test_helpers::run_pre_pass_only(src, &opts)
        .expect("pre-pass should succeed");
}

#[test]
fn pre_pass_skipped_when_profile_kernels_disabled() {
    let src = r#"
fn forward(x: Tensor<[1, 2048, 512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    return matmul(x, W)
"#;
    let mut opts = CompileOptions::default();
    opts.profile_kernels = false;

    let result = nsl_codegen::test_helpers::run_pre_pass_only(src, &opts)
        .expect("pre-pass skipped path should succeed");
    assert_eq!(result.prediction_map_len, 0);
    assert!(!result.manifest_builder_set);
    assert!(!result.fusion_plan_set);
}
