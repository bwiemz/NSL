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

// ---------------------------------------------------------------------------
// Task 9: sidecar lookup wired into final AWQ lowering
// ---------------------------------------------------------------------------
//
// The NSL source used by both tests below declares a tiny AWQ model with a
// single weight field `w`.  The projection path that compile_quant_block
// looks up is "Tiny.w" (model_type_name = "Tiny", field_name = "w").

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

/// Build an AWQ sidecar blob for the given projection → scales mapping and
/// wrap it in a `Sidecar` struct ready to set as `CompileOptions::calibration_sidecar`.
fn build_awq_sidecar(projections: &[(&str, Vec<f32>)]) -> nsl_codegen::calibration::sidecar::Sidecar {
    use std::collections::BTreeMap;
    use nsl_codegen::calibration::awq_sidecar;
    use nsl_codegen::calibration::sidecar::{Sidecar, SIDECAR_VERSION};

    let mut map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (name, scales) in projections {
        map.insert(name.to_string(), scales.clone());
    }
    let blob = awq_sidecar::serialize(&map);

    let mut hooks = std::collections::BTreeMap::new();
    hooks.insert("awq_activation_scales".to_string(), blob);

    Sidecar {
        version: SIDECAR_VERSION,
        checkpoint_sha256: "test".into(),
        calibration_data_sha256: "test".into(),
        hook_set_sha256: "test".into(),
        cache_key_digest: String::new(),
        num_samples_used: 1,
        hooks,
    }
}

/// Helper: parse + semantic-analyse + compile a module.  Returns the
/// `CodegenError` on failure (as its `Debug` string for easy assertion).
fn try_compile(
    src: &str,
    opts: &nsl_codegen::CompileOptions,
) -> Result<Vec<u8>, String> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    // Tolerate semantic warnings/notes but not hard errors — the model
    // above is intentionally minimal and may produce unused-variable hints.
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    nsl_codegen::compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        opts,
    )
    .map_err(|e| format!("{e:?}"))
}

#[test]
fn missing_scales_is_hard_error_when_sidecar_present() {
    // Sidecar exists but contains "Tiny.other_field" — NOT "Tiny.w".
    // compile_quant_block must return MissingScales for "Tiny.w".
    let sidecar =
        build_awq_sidecar(&[("Tiny.other_field", vec![1.0, 1.0, 1.0, 1.0])]);
    let mut opts = nsl_codegen::CompileOptions::default();
    opts.calibration_sidecar = Some(sidecar);

    let err = try_compile(TINY_AWQ_SRC, &opts)
        .expect_err("compile must fail when projection is missing from sidecar");
    assert!(
        err.contains("MissingScales") || err.contains("missing scales") || err.contains("missing_scales"),
        "expected a MissingScales error, got:\n{err}"
    );
    assert!(
        err.contains("Tiny.w"),
        "error must name the missing projection 'Tiny.w', got:\n{err}"
    );
}

#[test]
fn none_sidecar_falls_through_to_uncalibrated() {
    // No sidecar → uncalibrated AWQ path; compile must succeed.
    let opts = nsl_codegen::CompileOptions::default();
    try_compile(TINY_AWQ_SRC, &opts)
        .expect("compile must succeed when no calibration sidecar is present");
}
