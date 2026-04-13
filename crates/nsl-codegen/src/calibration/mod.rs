//! Compile-time calibration harness — see
//! `docs/superpowers/specs/2026-04-13-calibration-harness-design.md`.

pub mod cache;
pub mod ctx;
pub mod data_loader;
pub mod hooks;
pub mod identity_hook;
pub mod observation;
pub mod registry;
pub mod sidecar;
pub mod subprocess;

pub use ctx::CalibCtx;
pub use hooks::{CalibrationHook, CalibrationResult};
pub use observation::{LayerRef, ObservationPlan, ObservationSet, ParamRef};

use std::collections::BTreeMap;

use crate::calibration::registry::HookRegistry;
use crate::calibration::sidecar::{Sidecar, SIDECAR_VERSION};
use sha2::{Digest, Sha256};

/// Error surfaced by the harness driver.
#[derive(Debug)]
pub enum HarnessError {
    /// Infrastructure failure — behaviour depends on `--calibrate` mode.
    Infrastructure { reason: String },
    /// A hook returned `CalibrationResult::Degenerate` — always fatal,
    /// even in `best-effort` mode (spec §6.1).
    Degenerate { hook_id: String, reason: String },
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Infrastructure { reason } => {
                write!(f, "calibration infrastructure failure: {reason}")
            }
            Self::Degenerate { hook_id, reason } => {
                write!(f, "calibration hook '{hook_id}' returned degenerate data: {reason}")
            }
        }
    }
}
impl std::error::Error for HarnessError {}

/// Output of a successful calibration run.
#[derive(Debug)]
pub struct HarnessOutput {
    pub sidecar: Sidecar,
    /// Tag describing how the subprocess exited — "clean", "cached",
    /// "fallback".  Used by the driver for user-facing messaging.
    pub outcome_repr: &'static str,
}

/// **Stub harness** (Task 12): runs each hook's `emit_init` →
/// `emit_per_step` (per sample) → `emit_finalize` in-process, without
/// spawning a subprocess or generating a calibration binary.  Task 13
/// adds the real subprocess-based driver alongside this one.
pub fn run_harness_stub(
    registry: &HookRegistry,
    checkpoint_bytes: &[u8],
    calibration_data_bytes: &[u8],
    num_samples: u32,
) -> Result<HarnessOutput, HarnessError> {
    let mut ctx = CalibCtx {
        sample_idx: 0,
        total_samples: num_samples,
    };
    for hook in registry.iter() {
        hook.emit_init(&mut ctx);
    }
    for i in 0..num_samples {
        ctx.sample_idx = i;
        for hook in registry.iter() {
            hook.emit_per_step(&mut ctx);
        }
    }

    let mut hooks_out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for hook in registry.iter() {
        match hook.emit_finalize(&mut ctx) {
            CalibrationResult::Ok(bytes) => {
                hooks_out.insert(hook.id().to_string(), bytes);
            }
            CalibrationResult::Degenerate { reason } => {
                return Err(HarnessError::Degenerate {
                    hook_id: hook.id().to_string(),
                    reason,
                });
            }
        }
    }

    let sidecar = Sidecar {
        version: SIDECAR_VERSION,
        checkpoint_sha256: hex_digest(checkpoint_bytes),
        calibration_data_sha256: hex_digest(calibration_data_bytes),
        hook_set_sha256: hex_digest_ids(&registry.enabled_ids_sorted()),
        num_samples_used: num_samples,
        hooks: hooks_out,
    };
    Ok(HarnessOutput {
        sidecar,
        outcome_repr: "clean",
    })
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn hex_digest_ids(ids: &[String]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.as_bytes());
        h.update(b"\0");
    }
    let digest = h.finalize();
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod driver_tests {
    use super::*;
    use crate::calibration::identity_hook::IdentityHook;
    use crate::calibration::registry::HookRegistry;

    #[test]
    fn empty_registry_is_a_noop() {
        let registry = HookRegistry::new();
        let out = run_harness_stub(&registry, b"ckpt-bytes", b"calib-bytes", 10)
            .expect("stub harness should succeed");
        assert!(out.sidecar.hooks.is_empty());
        assert_eq!(out.outcome_repr, "clean");
    }

    #[test]
    fn identity_hook_produces_expected_sidecar_key() {
        let mut registry = HookRegistry::new();
        registry.register(Box::new(IdentityHook::new(b"fixed-bytes".to_vec())));
        let out = run_harness_stub(&registry, b"c", b"d", 4).unwrap();
        assert_eq!(out.sidecar.hooks.get("identity"), Some(&b"fixed-bytes".to_vec()));
        assert_eq!(out.sidecar.num_samples_used, 4);
    }

    #[test]
    fn degenerate_hook_fails_the_run() {
        let mut registry = HookRegistry::new();
        registry.register(Box::new(IdentityHook::new(Vec::new())));
        match run_harness_stub(&registry, b"c", b"d", 4) {
            Err(HarnessError::Degenerate { hook_id, reason }) => {
                assert_eq!(hook_id, "identity");
                assert!(reason.contains("empty"));
            }
            other => panic!("expected Degenerate error, got {other:?}"),
        }
    }
}
