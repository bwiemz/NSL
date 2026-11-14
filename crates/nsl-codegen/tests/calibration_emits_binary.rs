//! Smoke test: real_subprocess_entry produces a standalone executable
//! that runs and exits cleanly with the IdentityHook swapped in.

use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

use nsl_codegen::calibration::{
    binary_codegen::real_subprocess_entry, identity_hook::IdentityHook, HarnessConfig,
    HarnessMode, HookRegistry,
};

// Three more tests lived here. They drove the AWQ calibration hook through
// the subprocess (a bad checkpoint, a batch-shape mismatch, and a run that
// produced `awq_activation_scales`). The hook is deleted: it recorded
// statistics no quantizer read.

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
        compile_bundle: None,
        runtime_data_override: None,
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
