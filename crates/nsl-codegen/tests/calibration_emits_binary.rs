//! Smoke test: real_subprocess_entry produces a standalone executable
//! that runs and exits cleanly with the IdentityHook swapped in.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use nsl_codegen::calibration::{
    binary_codegen::real_subprocess_entry, identity_hook::IdentityHook, HarnessConfig, HarnessMode,
    HookRegistry,
};
use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests").join("fixtures").join(name)
}

fn awq_fixture_compile_bundle() -> (
    Vec<nsl_codegen::calibration::discovery::DiscoveredProjection>,
    Arc<nsl_codegen::calibration::CalibrationCompileBundle>,
) {
    let source =
        std::fs::read_to_string(fixture("awq_calibration_mlp.nsl")).expect("awq fixture readable");
    let mut interner = Interner::new();
    let (tokens, lex_diags) = tokenize(&source, FileId(0), &mut interner);
    assert!(
        lex_diags
            .iter()
            .all(|diag| !matches!(diag.level, Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.level, Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let projections =
        nsl_codegen::calibration::pre_scan_awq_projections_from_ast(&parsed.module, &interner);
    let mut analysis_interner = interner.clone();
    let analysis = nsl_semantic::analyze(&parsed.module, &mut analysis_interner);
    let bundle = Arc::new(nsl_codegen::calibration::CalibrationCompileBundle {
        ast: parsed.module,
        interner: analysis_interner,
        type_map: analysis.type_map.clone(),
    });

    (projections, bundle)
}

#[test]
fn real_subprocess_entry_produces_runnable_binary_with_identity_hook() {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp_dir = std::env::temp_dir().join(format!("nsl-calib-binary-test-{nanos}"));
    fs::create_dir_all(&tmp_dir).unwrap();
    let ckpt = tmp_dir.join("ckpt.safetensors");
    let data = tmp_dir.join("data.bin");
    fs::write(&ckpt, b"stub checkpoint").unwrap();
    let mut d = Vec::new();
    d.extend_from_slice(&1u32.to_le_bytes());
    d.extend_from_slice(&4u32.to_le_bytes());
    d.extend_from_slice(&[0u8; 16]);
    fs::write(&data, d).unwrap();

    let mut registry = HookRegistry::new();
    registry.register(Box::new(IdentityHook::new(b"smoke-test-payload".to_vec())));

    let cfg = HarnessConfig {
        checkpoints: vec![ckpt.clone()],
        calibration_data: data.clone(),
        samples: 1,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
        projections: vec![],
        compile_bundle: None,
    };

    let result = real_subprocess_entry(&cfg, &registry);

    let _ = fs::remove_dir_all(&tmp_dir);

    let out = result.expect("real_subprocess_entry should succeed for IdentityHook smoke test");
    assert_eq!(
        out.sidecar.hooks.get("identity"),
        Some(&b"smoke-test-payload".to_vec()),
        "IdentityHook output should round-trip through the calibration binary"
    );
}

/// Task 6: hooks that require forward activations (AWQ) now go through the
/// real subprocess entry instead of being rejected. This test supplies valid
/// calibration data so `peek_batch_seq` and `nsl_calibration_load` succeed,
/// then forces the emitted subprocess down the runtime model-load failure path
/// with an invalid checkpoint file, which returns status 4 → Infrastructure.
#[test]
fn real_subprocess_entry_accepts_linear_input_activations_hooks() {
    use nsl_codegen::calibration::awq_hook::AwqCalibrationHook;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp_dir = std::env::temp_dir().join(format!("nsl-calib-accept-test-{nanos}"));
    fs::create_dir_all(&tmp_dir).unwrap();
    let ckpt = tmp_dir.join("ckpt.safetensors");
    fs::write(&ckpt, b"x").unwrap();
    let data = fixture("awq_calib_data.safetensors");

    let (mut projections, compile_bundle) = awq_fixture_compile_bundle();
    projections.retain(|projection| projection.weight_shape[1] == 64);
    assert_eq!(
        projections.len(),
        1,
        "fixture should retain exactly one 64-channel projection"
    );

    let mut registry = HookRegistry::new();
    registry.register(Box::new(AwqCalibrationHook::from_discovered(&projections)));

    let cfg = HarnessConfig {
        checkpoints: vec![ckpt],
        calibration_data: data,
        samples: 1,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
        projections,
        compile_bundle: Some(compile_bundle),
    };

    let result = real_subprocess_entry(&cfg, &registry);
    let _ = fs::remove_dir_all(&tmp_dir);

    // The binary was emitted and run (no early rejection), but the data load
    // and batch-shape preflight succeeded; the runtime model load failed
    // → Infrastructure (subprocess status 4), NOT a rejection at the
    // gate with a "LinearInputActivations" message.
    match result {
        Err(nsl_codegen::calibration::HarnessError::Infrastructure { reason }) => {
            assert!(
                reason.contains("status 4"),
                "expected subprocess data-load failure status, got: {reason}"
            );
            // Must NOT be the old "LinearInputActivations" rejection — that gate is gone.
            assert!(
                !reason.contains("LinearInputActivations"),
                "should not see old rejection message; got: {reason}"
            );
            assert!(
                !reason.contains("compile_bundle"),
                "forward-path smoke test should reach the subprocess, not fail local bundle validation: {reason}"
            );
        }
        other => panic!("expected Infrastructure error (data load failed), got {other:?}"),
    }
}

#[test]
fn real_subprocess_entry_reports_batch_shape_mismatch_for_forward_hooks() {
    use nsl_codegen::calibration::awq_hook::AwqCalibrationHook;

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp_dir = std::env::temp_dir().join(format!("nsl-calib-batch-shape-test-{nanos}"));
    fs::create_dir_all(&tmp_dir).unwrap();
    let ckpt = tmp_dir.join("ckpt.safetensors");
    fs::write(&ckpt, b"x").unwrap();

    let (mut projections, compile_bundle) = awq_fixture_compile_bundle();
    projections[0].weight_shape[1] += 1;

    let mut registry = HookRegistry::new();
    registry.register(Box::new(AwqCalibrationHook::from_discovered(&projections)));

    let data = fixture("awq_calib_data.safetensors");
    let cfg = HarnessConfig {
        checkpoints: vec![ckpt],
        calibration_data: data,
        samples: 1,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
        projections,
        compile_bundle: Some(compile_bundle),
    };

    let result = real_subprocess_entry(&cfg, &registry);
    let _ = fs::remove_dir_all(&tmp_dir);

    match result {
        Err(nsl_codegen::calibration::HarnessError::Infrastructure { reason }) => {
            assert!(
                reason.contains("status 3"),
                "expected subprocess batch-shape refusal status, got: {reason}"
            );
            assert!(
                reason.contains("calibration: batch shape mismatch"),
                "expected structured mismatch refusal, got: {reason}"
            );
            assert!(reason.contains("requested:"), "missing requested: {reason}");
            assert!(reason.contains("expected:"), "missing expected: {reason}");
            assert!(reason.contains("found:"), "missing found: {reason}");
            assert!(reason.contains("Action:"), "missing Action: {reason}");
        }
        other => panic!("expected Infrastructure error (batch shape mismatch), got {other:?}"),
    }
}

#[test]
fn real_subprocess_entry_with_valid_awq_weights_produces_sidecar() {
    use nsl_codegen::calibration::awq_hook::AwqCalibrationHook;

    let (mut projections, compile_bundle) = awq_fixture_compile_bundle();
    projections.retain(|projection| projection.weight_shape[1] == 64);
    assert_eq!(
        projections.len(),
        1,
        "fixture should retain exactly one 64-channel projection"
    );

    let mut registry = HookRegistry::new();
    registry.register(Box::new(AwqCalibrationHook::from_discovered(&projections)));

    let cfg = HarnessConfig {
        checkpoints: vec![fixture("awq_calib_weights.safetensors")],
        calibration_data: fixture("awq_calib_data.safetensors"),
        samples: 8,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
        projections,
        compile_bundle: Some(compile_bundle),
    };

    let out = real_subprocess_entry(&cfg, &registry)
        .expect("valid AWQ weights should produce a subprocess sidecar for a matching projection");
    assert!(
        out.sidecar.hooks.contains_key("awq_activation_scales"),
        "AWQ sidecar should contain serialized activation scales"
    );
}
