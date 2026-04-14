//! Smoke test: real_subprocess_entry produces a standalone executable
//! that runs and exits cleanly with the IdentityHook swapped in.

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use nsl_codegen::calibration::{
    binary_codegen::real_subprocess_entry, identity_hook::IdentityHook, HarnessConfig,
    HarnessMode, HookRegistry,
};

#[test]
fn real_subprocess_entry_produces_runnable_binary_with_identity_hook() {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
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

#[test]
fn real_subprocess_entry_rejects_linear_input_activations_hooks() {
    use nsl_codegen::calibration::hooks::{CalibrationHook, CalibrationResult};
    use nsl_codegen::calibration::observation::{ObservationSet, ProjectionRef};
    use nsl_codegen::calibration::CalibCtx;

    struct NeedsLinearInputs;
    impl CalibrationHook for NeedsLinearInputs {
        fn id(&self) -> &'static str {
            "needs_linear_inputs"
        }
        fn requires(&self) -> ObservationSet {
            ObservationSet::LinearInputActivations(vec![ProjectionRef::new("blocks.0.attn.wq")])
        }
        fn emit_init(&self, _: &mut CalibCtx) {}
        fn emit_per_step(&self, _: &mut CalibCtx) {}
        fn emit_finalize(&self, _: &mut CalibCtx) -> CalibrationResult {
            CalibrationResult::Ok(vec![0u8; 1])
        }
    }

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let tmp_dir = std::env::temp_dir().join(format!("nsl-calib-reject-test-{nanos}"));
    fs::create_dir_all(&tmp_dir).unwrap();
    let ckpt = tmp_dir.join("ckpt.safetensors");
    let data = tmp_dir.join("data.bin");
    fs::write(&ckpt, b"x").unwrap();
    fs::write(&data, [0u8; 8]).unwrap();

    let mut registry = HookRegistry::new();
    registry.register(Box::new(NeedsLinearInputs));

    let cfg = HarnessConfig {
        checkpoints: vec![ckpt],
        calibration_data: data,
        samples: 1,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
    };

    let result = real_subprocess_entry(&cfg, &registry);
    let _ = fs::remove_dir_all(&tmp_dir);

    match result {
        Err(nsl_codegen::calibration::HarnessError::Infrastructure { reason }) => {
            assert!(
                reason.contains("LinearInputActivations"),
                "expected LinearInputActivations rejection, got: {reason}"
            );
        }
        other => panic!("expected Infrastructure error, got {other:?}"),
    }
}
