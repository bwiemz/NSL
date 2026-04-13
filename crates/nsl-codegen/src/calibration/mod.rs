//! Compile-time calibration harness — see
//! `docs/superpowers/specs/2026-04-13-calibration-harness-design.md`.

pub mod binary_codegen;
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
pub use registry::HookRegistry;

use std::collections::BTreeMap;

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
        cache_key_digest: String::new(),
        num_samples_used: num_samples,
        hooks: hooks_out,
    };
    Ok(HarnessOutput {
        sidecar,
        outcome_repr: "clean",
    })
}

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct HarnessConfig {
    pub checkpoints: Vec<PathBuf>,
    pub calibration_data: PathBuf,
    pub samples: u32,
    pub batch_size: u32,
    pub timeout_secs: u64,
    pub mode: HarnessMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessMode {
    Required,
    BestEffort,
}

pub type SubprocessEntry =
    fn(&HarnessConfig, &HookRegistry) -> Result<HarnessOutput, HarnessError>;

pub fn run_harness_simulated(
    registry: &HookRegistry,
    cfg: &HarnessConfig,
    entry: SubprocessEntry,
) -> Result<HarnessOutput, HarnessError> {
    let ckpt_hashes: Vec<String> = cfg
        .checkpoints
        .iter()
        .map(|p| {
            crate::calibration::cache::hash_file(p).map_err(|e| HarnessError::Infrastructure {
                reason: format!("hashing checkpoint {}: {e}", p.display()),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let primary_ckpt = cfg.checkpoints.first().ok_or(HarnessError::Infrastructure {
        reason: "no checkpoint paths supplied".into(),
    })?;

    // Compute the full cache-key digest covering all invalidation
    // dimensions — checkpoints, calibration data, hook set, knobs.
    let calibration_data_hash = crate::calibration::cache::hash_file(&cfg.calibration_data)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!(
                "hashing calibration data {}: {e}",
                cfg.calibration_data.display()
            ),
        })?;
    let cache_key = crate::calibration::cache::CacheKey {
        checkpoint_hashes: ckpt_hashes.clone(),
        calibration_data_hash: calibration_data_hash.clone(),
        hook_ids_sorted: registry.enabled_ids_sorted(),
        samples: cfg.samples,
        batch_size: cfg.batch_size,
    };
    let cache_key_digest = cache_key.digest();

    if let Some(cached) = crate::calibration::cache::try_load(primary_ckpt, &cache_key_digest) {
        return Ok(HarnessOutput {
            sidecar: cached,
            outcome_repr: "cached",
        });
    }

    let result = entry(cfg, registry);

    match result {
        Ok(mut out) => {
            // Driver owns canonical identity — overwrite all fields the
            // subprocess may have stubbed.
            let primary_hash = ckpt_hashes.first().cloned().unwrap_or_default();
            out.sidecar.checkpoint_sha256 = primary_hash;
            out.sidecar.calibration_data_sha256 = calibration_data_hash;
            out.sidecar.hook_set_sha256 = hex_digest_ids(&registry.enabled_ids_sorted());
            out.sidecar.cache_key_digest = cache_key_digest;
            let _ = crate::calibration::cache::store(primary_ckpt, &out.sidecar);
            Ok(out)
        }
        Err(HarnessError::Degenerate { .. }) => result,
        Err(HarnessError::Infrastructure { reason }) => match cfg.mode {
            HarnessMode::Required => Err(HarnessError::Infrastructure { reason }),
            HarnessMode::BestEffort => {
                eprintln!(
                    "warning: calibration infrastructure failure in best-effort mode: {reason}"
                );
                Ok(HarnessOutput {
                    sidecar: empty_sidecar_for_fallback(&ckpt_hashes, registry),
                    outcome_repr: "fallback",
                })
            }
        },
    }
}

fn empty_sidecar_for_fallback(
    ckpt_hashes: &[String],
    registry: &HookRegistry,
) -> Sidecar {
    Sidecar {
        version: SIDECAR_VERSION,
        checkpoint_sha256: ckpt_hashes.first().cloned().unwrap_or_default(),
        calibration_data_sha256: String::new(),
        hook_set_sha256: hex_digest_ids(&registry.enabled_ids_sorted()),
        cache_key_digest: String::new(),
        num_samples_used: 0,
        hooks: BTreeMap::new(),
    }
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

    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static T13_COUNTER: AtomicU64 = AtomicU64::new(0);
    fn t13_tmp(tag: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let n = T13_COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!("nsl-t13-{tag}-{nanos}-{n}"));
        p
    }

    #[test]
    fn harness_writes_and_reads_cache() {
        let ckpt = t13_tmp("ckpt");
        fs::write(&ckpt, b"checkpoint-bytes-v1").unwrap();
        let data = t13_tmp("data.bin");
        let mut header = Vec::new();
        header.extend_from_slice(&0u32.to_le_bytes());
        header.extend_from_slice(&0u32.to_le_bytes());
        fs::write(&data, header).unwrap();

        let mut registry = HookRegistry::new();
        registry.register(Box::new(
            crate::calibration::identity_hook::IdentityHook::new(b"payload".to_vec()),
        ));

        let cfg = HarnessConfig {
            checkpoints: vec![ckpt.clone()],
            calibration_data: data.clone(),
            samples: 4,
            batch_size: 2,
            timeout_secs: 5,
            mode: HarnessMode::Required,
        };

        let r1 = run_harness_simulated(&registry, &cfg, simulated_ok_subprocess)
            .expect("first run ok");
        assert_eq!(r1.outcome_repr, "clean");
        assert_eq!(
            r1.sidecar.hooks.get("identity"),
            Some(&b"payload".to_vec())
        );

        let r2 = run_harness_simulated(&registry, &cfg, simulated_would_panic)
            .expect("second run ok");
        assert_eq!(r2.outcome_repr, "cached");
        assert_eq!(
            r2.sidecar.hooks.get("identity"),
            Some(&b"payload".to_vec())
        );

        let _ = fs::remove_file(&ckpt);
        let _ = fs::remove_file(&data);
        let _ = fs::remove_file(crate::calibration::cache::sidecar_path_for(&ckpt));
    }

    #[test]
    fn required_mode_fails_on_infra_error() {
        let ckpt = t13_tmp("ckpt2");
        fs::write(&ckpt, b"x").unwrap();
        let data = t13_tmp("data2.bin");
        fs::write(&data, [0u8; 8]).unwrap();

        let registry = HookRegistry::new();
        let cfg = HarnessConfig {
            checkpoints: vec![ckpt.clone()],
            calibration_data: data.clone(),
            samples: 1,
            batch_size: 1,
            timeout_secs: 1,
            mode: HarnessMode::Required,
        };
        match run_harness_simulated(&registry, &cfg, simulated_infra_error) {
            Err(HarnessError::Infrastructure { reason }) => {
                assert!(reason.contains("simulated infra"));
            }
            other => panic!("expected Infrastructure, got {other:?}"),
        }
        let _ = fs::remove_file(&ckpt);
        let _ = fs::remove_file(&data);
    }

    #[test]
    fn best_effort_mode_returns_fallback_on_infra_error() {
        let ckpt = t13_tmp("ckpt3");
        fs::write(&ckpt, b"x").unwrap();
        let data = t13_tmp("data3.bin");
        fs::write(&data, [0u8; 8]).unwrap();

        let registry = HookRegistry::new();
        let cfg = HarnessConfig {
            checkpoints: vec![ckpt.clone()],
            calibration_data: data.clone(),
            samples: 1,
            batch_size: 1,
            timeout_secs: 1,
            mode: HarnessMode::BestEffort,
        };
        let r = run_harness_simulated(&registry, &cfg, simulated_infra_error)
            .expect("best-effort should not error");
        assert_eq!(r.outcome_repr, "fallback");
        assert!(r.sidecar.hooks.is_empty());
        let _ = fs::remove_file(&ckpt);
        let _ = fs::remove_file(&data);
    }

    #[test]
    fn degenerate_always_fatal_even_in_best_effort() {
        let ckpt = t13_tmp("ckpt4");
        fs::write(&ckpt, b"x").unwrap();
        let data = t13_tmp("data4.bin");
        fs::write(&data, [0u8; 8]).unwrap();

        let mut registry = HookRegistry::new();
        registry.register(Box::new(
            crate::calibration::identity_hook::IdentityHook::new(Vec::new()),
        ));
        let cfg = HarnessConfig {
            checkpoints: vec![ckpt.clone()],
            calibration_data: data.clone(),
            samples: 1,
            batch_size: 1,
            timeout_secs: 1,
            mode: HarnessMode::BestEffort,
        };
        let err = run_harness_simulated(&registry, &cfg, simulated_ok_subprocess).unwrap_err();
        match err {
            HarnessError::Degenerate { hook_id, .. } => assert_eq!(hook_id, "identity"),
            other => panic!("expected Degenerate, got {other:?}"),
        }
        let _ = fs::remove_file(&ckpt);
        let _ = fs::remove_file(&data);
    }

    #[test]
    fn cache_invalidates_on_sample_count_change() {
        let ckpt = t13_tmp("ckpt-samples");
        fs::write(&ckpt, b"x").unwrap();
        let data = t13_tmp("data-samples.bin");
        fs::write(&data, [0u8; 8]).unwrap();

        let mut registry = HookRegistry::new();
        registry.register(Box::new(
            crate::calibration::identity_hook::IdentityHook::new(b"v1".to_vec()),
        ));

        let mut cfg = HarnessConfig {
            checkpoints: vec![ckpt.clone()],
            calibration_data: data.clone(),
            samples: 4,
            batch_size: 2,
            timeout_secs: 5,
            mode: HarnessMode::Required,
        };
        let r1 = run_harness_simulated(&registry, &cfg, simulated_ok_subprocess).unwrap();
        assert_eq!(r1.outcome_repr, "clean");

        // Change sample count → cache must miss.
        cfg.samples = 8;
        let r2 = run_harness_simulated(&registry, &cfg, simulated_ok_subprocess).unwrap();
        assert_eq!(r2.outcome_repr, "clean", "expected cache miss on sample-count change");

        let _ = fs::remove_file(&ckpt);
        let _ = fs::remove_file(&data);
        let _ = fs::remove_file(crate::calibration::cache::sidecar_path_for(&ckpt));
    }

    fn simulated_ok_subprocess(
        _cfg: &HarnessConfig,
        registry: &HookRegistry,
    ) -> Result<HarnessOutput, HarnessError> {
        super::run_harness_stub(registry, b"ck", b"cd", 1)
    }

    fn simulated_infra_error(
        _cfg: &HarnessConfig,
        _registry: &HookRegistry,
    ) -> Result<HarnessOutput, HarnessError> {
        Err(HarnessError::Infrastructure {
            reason: "simulated infra error".into(),
        })
    }

    fn simulated_would_panic(
        _cfg: &HarnessConfig,
        _registry: &HookRegistry,
    ) -> Result<HarnessOutput, HarnessError> {
        panic!("subprocess should not have been invoked — cache hit expected");
    }
}
