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
        projections: vec![],
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
/// real subprocess entry instead of being rejected.  The subprocess emits a
/// `calibration_main` that calls `nsl_calibration_load`; invalid calibration
/// data (8 zero bytes, no NSLB magic) causes the load to return null and the
/// binary to exit with status 4 → Infrastructure.
#[test]
fn real_subprocess_entry_accepts_linear_input_activations_hooks() {
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
    let tmp_dir = std::env::temp_dir().join(format!("nsl-calib-accept-test-{nanos}"));
    fs::create_dir_all(&tmp_dir).unwrap();
    let ckpt = tmp_dir.join("ckpt.safetensors");
    let data = tmp_dir.join("data.bin");
    fs::write(&ckpt, b"x").unwrap();
    // Invalid calibration data (bad magic) → nsl_calibration_load returns null
    // → subprocess exits 4 → Infrastructure.  This tests that the hook is
    // ACCEPTED (no rejection at the entry gate) and goes through the forward path.
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
        projections: vec![],
    };

    let result = real_subprocess_entry(&cfg, &registry);
    let _ = fs::remove_dir_all(&tmp_dir);

    // The binary was emitted and run (no early rejection), but the data load
    // failed → Infrastructure (subprocess status 4), NOT a rejection at the
    // gate with a "LinearInputActivations" message.
    match result {
        Err(nsl_codegen::calibration::HarnessError::Infrastructure { reason }) => {
            // Must NOT be the old "LinearInputActivations" rejection — that gate is gone.
            assert!(
                !reason.contains("LinearInputActivations"),
                "should not see old rejection message; got: {reason}"
            );
        }
        other => panic!("expected Infrastructure error (data load failed), got {other:?}"),
    }
}
