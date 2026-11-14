//! Data-flow test for the AWQ calibration hook.
//!
//! Verifies: AwqCalibrationHook's sidecar output →
//! nsl-runtime::awq::AwqScales::from_blob, i.e. that the per-channel
//! statistic the hook records is the one the blob decoder reads back.
//!
//! Nothing quantizes with these scales. The tests that fed them to a
//! quantizer checked only that an int8 stand-in's bytes changed, and the
//! tests of the `default: awq4` lowering compiled a program the checker
//! refuses; both are gone, and the last test here pins the refusal.

use nsl_codegen::calibration::{
    CalibCtx, CalibrationHook, CalibrationResult, ProjectionRef,
    awq_hook::AwqCalibrationHook,
    retention::{RetentionTable, TensorShape},
};
use nsl_runtime::awq::AwqScales;

fn setup_retention(projections: &[ProjectionRef], in_channels: u64) -> RetentionTable {
    let mut table = RetentionTable::new();
    for p in projections {
        // Shape: [batch=1, seq=8, in_channels]
        table.register(p.clone(), TensorShape::new(vec![1, 8, in_channels]), 4);
    }
    table
}

#[test]
fn awq_calibration_records_the_largest_activation_per_channel() {
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

    // The statistic is the running maximum of |x| per channel: 13.0 for
    // channel 4 (the largest of 10, 11, 12, 13) and 1.0 elsewhere.
    let want: Vec<f32> = (0..in_channels).map(|c| if c == 4 { 13.0 } else { 1.0 }).collect();
    assert_eq!(observed, &want);
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

// ---------------------------------------------------------------------------
// `default: awq4` is not lowered
// ---------------------------------------------------------------------------

const TINY_AWQ_SRC: &str = r#"model Tiny:
    w: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = Tiny()
    quant static qm from m:
        default: awq4
    let x = ones([1, 4])
    let y = qm.forward(x)
"#;

/// The checker refuses `default: awq4`. Code generation used to accept it
/// from a caller that skipped the checker, pre-scale the weight by the
/// sidecar's activation scales with no inverse applied anywhere, and pass a
/// dtype code the runtime quantizer aborts on. It refuses too.
#[test]
fn codegen_refuses_an_awq4_quant_block_the_checker_was_not_asked_about() {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(TINY_AWQ_SRC, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis.diagnostics.iter().any(|d| d.message.contains("`default: awq4` is not implemented")),
        "the checker refuses it first: {:?}",
        analysis.diagnostics.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
    let err = nsl_codegen::compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        &nsl_codegen::CompileOptions::default(),
    )
    .map(|_| ())
    .expect_err("codegen must not lower an awq4 quant block");
    let text = format!("{err:?}");
    assert!(text.contains("awq4") && text.contains("not implemented"), "{text}");
}
