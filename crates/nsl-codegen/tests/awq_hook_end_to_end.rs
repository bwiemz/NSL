//! End-to-end data-flow test for AWQ calibration.
//!
//! Verifies: AwqCalibrationHook's sidecar output →
//! nsl-runtime::awq::AwqScales::from_blob → awq_quantize_with_scales
//! produces output that differs from the no-calibration baseline.
//!
//! Routes through the in-process harness stub (matching Task 12's
//! run_harness_production dispatch for forward-activation hooks); a
//! follow-up plan that emits the model forward pass into
//! calibration_main will transparently flip this over to the real
//! subprocess path.

use nsl_codegen::calibration::{
    CalibCtx, CalibrationHook, CalibrationResult, ProjectionRef,
    awq_hook::AwqCalibrationHook,
    retention::{RetentionTable, TensorShape},
};
use nsl_runtime::awq::{AwqScales, awq_quantize_with_scales};

fn setup_retention(projections: &[ProjectionRef], in_channels: u64) -> RetentionTable {
    let mut table = RetentionTable::new();
    for p in projections {
        // Shape: [batch=1, seq=8, in_channels]
        table.register(p.clone(), TensorShape::new(vec![1, 8, in_channels]), 4);
    }
    table
}

#[test]
fn awq_calibration_produces_nonuniform_scales_that_alter_quantization() {
    // 1. Synthetic scenario: one projection with 8 input channels.
    //    Simulated activations vary per channel (channel 4 is 10x the
    //    others), so the calibration should produce a non-uniform
    //    scale vector that's distinguishable from the all-ones
    //    baseline.
    let proj = ProjectionRef::new("blocks.0.attn.wq");
    let in_channels: usize = 8;
    let table = setup_retention(&[proj.clone()], in_channels as u64);

    let hook = AwqCalibrationHook::new(vec![proj.clone()]);
    let mut ctx = CalibCtx::for_tests(&table);

    hook.emit_init(&mut ctx);

    // Simulate 4 calibration samples.  Each sample observes per-channel
    // activations where channel 4's magnitude dominates.
    for sample_idx in 0..4 {
        let handle = *hook.handles_for_test().get(&proj).unwrap();
        let per_sample: Vec<f32> = (0..in_channels)
            .map(|c| if c == 4 { 10.0 + sample_idx as f32 } else { 1.0 })
            .collect();
        ctx.stub_running_max_abs(handle, &per_sample);
    }

    // Extract the sidecar blob.
    let blob = match hook.emit_finalize(&mut ctx) {
        CalibrationResult::Ok(b) => b,
        other => panic!("expected Ok, got {other:?}"),
    };

    // 2. Parse the blob via nsl-runtime's reader.
    let scales = AwqScales::from_blob(&blob).expect("sidecar should parse");
    let observed = scales
        .by_projection
        .get("blocks.0.attn.wq")
        .expect("projection scale");
    assert_eq!(observed.len(), in_channels);
    // Channel 4 should be the maximum (we injected a dominant value).
    let (max_idx, _) = observed
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    assert_eq!(
        max_idx, 4,
        "channel 4 should have the largest scale: {observed:?}"
    );

    // 3. Feed the calibrated scales into the quantizer and verify its
    //    output differs from the uncalibrated (None / all-ones) baseline.
    let out_channels = 4;
    let weight: Vec<f32> = (0..out_channels * in_channels)
        .map(|i| (i as f32 - 16.0) * 0.1)
        .collect();
    let alpha = 0.5;

    let baseline = awq_quantize_with_scales(&weight, in_channels, out_channels, None, alpha);
    let calibrated =
        awq_quantize_with_scales(&weight, in_channels, out_channels, Some(observed), alpha);

    assert_ne!(
        baseline.dequantized_check_bytes, calibrated.dequantized_check_bytes,
        "calibrated quantization must differ from the uncalibrated baseline"
    );
}

#[test]
fn awq_calibration_with_uniform_activations_matches_baseline() {
    // Sanity: when all channels see the same activation magnitude, the
    // computed scales should be uniform (all equal to the injected
    // max-abs value).
    let proj = ProjectionRef::new("blocks.0.attn.wq");
    let in_channels = 4;
    let table = setup_retention(&[proj.clone()], in_channels as u64);

    let hook = AwqCalibrationHook::new(vec![proj.clone()]);
    let mut ctx = CalibCtx::for_tests(&table);
    hook.emit_init(&mut ctx);

    for _ in 0..2 {
        let handle = *hook.handles_for_test().get(&proj).unwrap();
        ctx.stub_running_max_abs(handle, &vec![2.0; in_channels]);
    }

    let blob = match hook.emit_finalize(&mut ctx) {
        CalibrationResult::Ok(b) => b,
        other => panic!("expected Ok, got {other:?}"),
    };
    let scales = AwqScales::from_blob(&blob).unwrap();
    let observed = scales.by_projection.get("blocks.0.attn.wq").unwrap();

    // All channels see max|x| = 2.0, so scales should be uniform.
    assert!(
        observed.iter().all(|v| (*v - 2.0).abs() < 1e-6),
        "scales should all be 2.0: {observed:?}"
    );
}
