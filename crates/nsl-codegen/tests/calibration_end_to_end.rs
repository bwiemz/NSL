//! End-to-end integration test for the calibration harness.  Exercises
//! the full public API surface that real consumers (AWQ, WGGO Phase 2,
//! FP8, GPTQ) will use when they land, without depending on the
//! in-progress subprocess codegen.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use nsl_codegen::calibration::{
    HarnessConfig, HarnessError, HarnessMode, HarnessOutput, HookRegistry,
    identity_hook::IdentityHook, run_harness_simulated,
};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
fn tmp_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = TMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut p = std::env::temp_dir();
    p.push(format!("nsl-e2e-{tag}-{nanos}-{n}"));
    p
}

fn minimal_bin(path: &std::path::Path) {
    let mut blob = Vec::new();
    blob.extend_from_slice(&2u32.to_le_bytes());
    blob.extend_from_slice(&4u32.to_le_bytes());
    blob.extend_from_slice(&[0u8; 32]);
    fs::write(path, blob).unwrap();
}

fn identity_seam(
    _cfg: &HarnessConfig,
    registry: &HookRegistry,
) -> Result<HarnessOutput, HarnessError> {
    nsl_codegen::calibration::run_harness_stub(registry, b"c", b"d", 2)
}

#[test]
fn e2e_single_hook_round_trip() {
    let ckpt = tmp_path("ckpt");
    fs::write(&ckpt, b"ckpt-v1").unwrap();
    let data = tmp_path("data.bin");
    minimal_bin(&data);

    let mut registry = HookRegistry::new();
    registry.register(Box::new(IdentityHook::new(b"e2e-payload".to_vec())));

    let cfg = HarnessConfig {
        checkpoints: vec![ckpt.clone()],
        calibration_data: data.clone(),
        samples: 2,
        batch_size: 1,
        timeout_secs: 5,
        mode: HarnessMode::Required,
        projections: vec![],
        compile_bundle: None,
    };

    let first = run_harness_simulated(&registry, &cfg, identity_seam).expect("first run");
    assert_eq!(first.outcome_repr, "clean");
    assert_eq!(
        first.sidecar.hooks.get("identity"),
        Some(&b"e2e-payload".to_vec())
    );

    let second = run_harness_simulated(&registry, &cfg, identity_seam).expect("second run");
    assert_eq!(second.outcome_repr, "cached");
    assert_eq!(
        second.sidecar.hooks.get("identity"),
        Some(&b"e2e-payload".to_vec())
    );

    let _ = fs::remove_file(&ckpt);
    let _ = fs::remove_file(&data);
    let _ = fs::remove_file(
        nsl_codegen::calibration::cache::sidecar_path_for(&ckpt),
    );
}

#[test]
fn e2e_multi_hook_sidecar_isolation() {
    struct A;
    impl nsl_codegen::calibration::CalibrationHook for A {
        fn id(&self) -> &'static str { "hook_a" }
        fn requires(&self) -> nsl_codegen::calibration::ObservationSet {
            nsl_codegen::calibration::ObservationSet::Empty
        }
        fn emit_init(&self, _: &mut nsl_codegen::calibration::CalibCtx) {}
        fn emit_per_step(&self, _: &mut nsl_codegen::calibration::CalibCtx) {}
        fn emit_finalize(
            &self,
            _: &mut nsl_codegen::calibration::CalibCtx,
        ) -> nsl_codegen::calibration::CalibrationResult {
            nsl_codegen::calibration::CalibrationResult::Ok(b"alpha".to_vec())
        }
    }
    struct B;
    impl nsl_codegen::calibration::CalibrationHook for B {
        fn id(&self) -> &'static str { "hook_b" }
        fn requires(&self) -> nsl_codegen::calibration::ObservationSet {
            nsl_codegen::calibration::ObservationSet::Empty
        }
        fn emit_init(&self, _: &mut nsl_codegen::calibration::CalibCtx) {}
        fn emit_per_step(&self, _: &mut nsl_codegen::calibration::CalibCtx) {}
        fn emit_finalize(
            &self,
            _: &mut nsl_codegen::calibration::CalibCtx,
        ) -> nsl_codegen::calibration::CalibrationResult {
            nsl_codegen::calibration::CalibrationResult::Ok(b"beta".to_vec())
        }
    }

    let ckpt = tmp_path("ckpt-multi");
    fs::write(&ckpt, b"multi").unwrap();
    let data = tmp_path("data-multi.bin");
    minimal_bin(&data);

    let mut registry = HookRegistry::new();
    registry.register(Box::new(A));
    registry.register(Box::new(B));

    let cfg = HarnessConfig {
        checkpoints: vec![ckpt.clone()],
        calibration_data: data.clone(),
        samples: 1,
        batch_size: 1,
        timeout_secs: 5,
        mode: HarnessMode::Required,
        projections: vec![],
        compile_bundle: None,
    };

    let out = run_harness_simulated(&registry, &cfg, identity_seam).unwrap();
    assert_eq!(out.sidecar.hooks.get("hook_a"), Some(&b"alpha".to_vec()));
    assert_eq!(out.sidecar.hooks.get("hook_b"), Some(&b"beta".to_vec()));
    assert_eq!(out.sidecar.hooks.len(), 2);

    let _ = fs::remove_file(&ckpt);
    let _ = fs::remove_file(&data);
    let _ = fs::remove_file(
        nsl_codegen::calibration::cache::sidecar_path_for(&ckpt),
    );
}
