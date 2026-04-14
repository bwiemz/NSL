//! Calibration sidecar cache.  Mirrors the pattern in
//! `wggo_weight_analysis_cache`: SHA-256 of (all checkpoint hashes ⊕
//! calibration data hash ⊕ enabled hook ids ⊕ sample/batch knobs)
//! drives cache invalidation.  See spec §7.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::calibration::discovery::DiscoveredProjection;
use crate::calibration::sidecar::{Sidecar, SIDECAR_VERSION};

const CACHE_SUFFIX: &str = ".calibration.json";

/// Components that influence calibration output.  Any change here
/// invalidates the cache.
#[derive(Debug, Clone)]
pub struct CacheKey {
    /// SHA-256 of every checkpoint involved (base model + LoRA
    /// adapters + CPKD teacher/student).  Order-insensitive: `digest`
    /// sorts internally.
    pub checkpoint_hashes: Vec<String>,
    pub calibration_data_hash: String,
    /// Enabled hook ids.  Order-insensitive: `digest` sorts internally.
    pub hook_ids_sorted: Vec<String>,
    pub samples: u32,
    pub batch_size: u32,
    /// Per-projection `(path, weight_shape)` included in the digest so
    /// that renaming a projection path or changing `in_features` /
    /// `out_features` under the same name both invalidate the cache.
    /// Order-insensitive: `digest` sorts by path internally.
    pub projections: Vec<DiscoveredProjection>,
}

impl CacheKey {
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        let mut ckpts = self.checkpoint_hashes.clone();
        ckpts.sort();
        for c in &ckpts {
            h.update((c.len() as u64).to_le_bytes());
            h.update(c.as_bytes());
        }
        h.update((self.calibration_data_hash.len() as u64).to_le_bytes());
        h.update(self.calibration_data_hash.as_bytes());
        let mut hooks = self.hook_ids_sorted.clone();
        hooks.sort();
        for hk in &hooks {
            h.update((hk.len() as u64).to_le_bytes());
            h.update(hk.as_bytes());
        }
        h.update(self.samples.to_le_bytes());
        h.update(self.batch_size.to_le_bytes());
        // Sort projections by path for determinism (order-insensitive).
        let mut ps = self.projections.clone();
        ps.sort_by(|a, b| a.projection.0.cmp(&b.projection.0));
        h.update((ps.len() as u64).to_le_bytes());
        for p in &ps {
            // length-prefixed path string — same framing as other fields.
            h.update((p.projection.0.len() as u64).to_le_bytes());
            h.update(p.projection.0.as_bytes());
            // concrete shape dimensions (fixed 8 bytes total, no prefix needed).
            h.update(p.weight_shape[0].to_le_bytes());
            h.update(p.weight_shape[1].to_le_bytes());
        }
        hex(&h.finalize())
    }
}

/// Sidecar path for a given checkpoint (next to it, with
/// `.calibration.json` suffix).
pub fn sidecar_path_for(ckpt_path: &Path) -> PathBuf {
    let mut s = ckpt_path.as_os_str().to_os_string();
    s.push(CACHE_SUFFIX);
    PathBuf::from(s)
}

/// Stream a file's SHA-256 in 64 KiB chunks.
pub fn hash_file(path: &Path) -> std::io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Best-effort cache read.  Returns `None` on any failure.  Caller
/// passes the *full* `CacheKey::digest()` so invalidation covers all
/// checkpoints, calibration data, enabled hook set, and knobs — not
/// just the primary checkpoint hash.
pub fn try_load(ckpt_path: &Path, expected_cache_key_digest: &str) -> Option<Sidecar> {
    let sp = sidecar_path_for(ckpt_path);
    if !sp.exists() {
        return None;
    }
    let json = fs::read_to_string(&sp).ok()?;
    let s: Sidecar = serde_json::from_str(&json).ok()?;
    if s.version != SIDECAR_VERSION {
        return None;
    }
    if s.cache_key_digest != expected_cache_key_digest {
        return None;
    }
    Some(s)
}

/// Write the sidecar next to the checkpoint.  Errors are the caller's
/// to log; the cache is advisory.
pub fn store(ckpt_path: &Path, sidecar: &Sidecar) -> std::io::Result<()> {
    let sp = sidecar_path_for(ckpt_path);
    let json = serde_json::to_string_pretty(sidecar).map_err(std::io::Error::other)?;
    let mut f = fs::File::create(&sp)?;
    f.write_all(json.as_bytes())?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod projection_key_tests {
    use super::*;
    use crate::calibration::discovery::DiscoveredProjection;
    use crate::calibration::observation::ProjectionRef;

    fn proj(path: &str, out_f: u32, in_f: u32) -> DiscoveredProjection {
        DiscoveredProjection {
            projection: ProjectionRef(path.into()),
            weight_shape: [out_f, in_f],
        }
    }

    fn base_key() -> CacheKey {
        CacheKey {
            checkpoint_hashes: vec!["ckpt-abc".into()],
            calibration_data_hash: "data-xyz".into(),
            hook_ids_sorted: vec!["awq".into()],
            samples: 128,
            batch_size: 4,
            projections: vec![proj("model.up_proj", 128, 64)],
        }
    }

    #[test]
    fn shape_change_changes_digest() {
        let a = base_key();
        let mut b = a.clone();
        b.projections = vec![proj("model.up_proj", 128, 96)]; // in_f changed
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn path_rename_changes_digest() {
        let a = base_key();
        let mut b = a.clone();
        b.projections = vec![proj("model.u_proj", 128, 64)];
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn projection_order_does_not_affect_digest() {
        let mut a = base_key();
        a.projections = vec![
            proj("model.down_proj", 64, 128),
            proj("model.up_proj", 128, 64),
        ];
        let mut b = a.clone();
        b.projections.reverse();
        assert_eq!(a.digest(), b.digest(), "digest must be order-insensitive");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn tmp(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!("nsl-calib-cache-{tag}-{nanos}-{n}.bin"));
        p
    }

    fn write(path: &std::path::Path, bytes: &[u8]) {
        let mut f = fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    fn cleanup(p: &std::path::Path) {
        let _ = fs::remove_file(p);
        let _ = fs::remove_file(sidecar_path_for(p));
    }

    fn sample_sidecar(digest: &str) -> Sidecar {
        let mut hooks = BTreeMap::new();
        hooks.insert("identity".to_string(), vec![1, 2, 3]);
        Sidecar {
            version: crate::calibration::sidecar::SIDECAR_VERSION,
            checkpoint_sha256: "ckpt-hash".into(),
            calibration_data_sha256: "data".into(),
            hook_set_sha256: "hooks".into(),
            cache_key_digest: digest.into(),
            num_samples_used: 10,
            hooks,
            wggo_head_gradients: None,
        }
    }

    #[test]
    fn key_is_sha256_of_sorted_components() {
        let k1 = CacheKey {
            checkpoint_hashes: vec!["a".into(), "b".into()],
            calibration_data_hash: "d".into(),
            hook_ids_sorted: vec!["h1".into(), "h2".into()],
            samples: 128,
            batch_size: 4,
            projections: vec![],
        };
        let k2 = CacheKey {
            checkpoint_hashes: vec!["b".into(), "a".into()],
            calibration_data_hash: "d".into(),
            hook_ids_sorted: vec!["h2".into(), "h1".into()],
            samples: 128,
            batch_size: 4,
            projections: vec![],
        };
        assert_eq!(k1.digest(), k2.digest());
    }

    #[test]
    fn key_differs_on_any_component_change() {
        let base = CacheKey {
            checkpoint_hashes: vec!["a".into()],
            calibration_data_hash: "d".into(),
            hook_ids_sorted: vec!["h1".into()],
            samples: 128,
            batch_size: 4,
            projections: vec![],
        };
        let changed = CacheKey {
            samples: 129,
            ..base.clone()
        };
        assert_ne!(base.digest(), changed.digest());
    }

    #[test]
    fn store_then_load_roundtrips() {
        let p = tmp("roundtrip");
        write(&p, b"checkpoint-bytes");
        let s = sample_sidecar("test-digest");
        store(&p, &s).unwrap();
        let loaded = try_load(&p, "test-digest").expect("should load");
        assert_eq!(loaded.hooks.get("identity"), Some(&vec![1, 2, 3]));
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_hash_mismatch() {
        let p = tmp("mismatch");
        write(&p, b"x");
        let s = sample_sidecar("A");
        store(&p, &s).unwrap();
        assert!(try_load(&p, "B").is_none());
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_missing_sidecar() {
        let p = tmp("nosidecar");
        write(&p, b"x");
        assert!(try_load(&p, "anything").is_none());
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_stale_version() {
        let p = tmp("badver");
        write(&p, b"x");
        let sp = sidecar_path_for(&p);
        fs::write(&sp, r#"{"version":0,"checkpoint_sha256":"x","calibration_data_sha256":"","hook_set_sha256":"","cache_key_digest":"","num_samples_used":0,"hooks":{}}"#).unwrap();
        assert!(try_load(&p, "x").is_none());
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_unparseable_sidecar() {
        let p = tmp("garbage");
        write(&p, b"x");
        fs::write(sidecar_path_for(&p), b"{not json").unwrap();
        assert!(try_load(&p, "anything").is_none());
        cleanup(&p);
    }

    #[test]
    fn null_byte_in_hook_id_does_not_collide_with_split_ids() {
        // The pre-fix code could collide ["h1\0h2"] with ["h1", "h2"]
        // because the domain separator was a single \0.  Length-prefix
        // framing prevents this.
        let combined = CacheKey {
            checkpoint_hashes: vec!["c".into()],
            calibration_data_hash: "d".into(),
            hook_ids_sorted: vec!["h1\0h2".into()],
            samples: 1,
            batch_size: 1,
            projections: vec![],
        };
        let split = CacheKey {
            checkpoint_hashes: vec!["c".into()],
            calibration_data_hash: "d".into(),
            hook_ids_sorted: vec!["h1".into(), "h2".into()],
            samples: 1,
            batch_size: 1,
            projections: vec![],
        };
        assert_ne!(combined.digest(), split.digest());
    }

    #[test]
    fn null_byte_at_checkpoint_data_boundary_does_not_collide() {
        // Under the pre-fix \0-separator framing, both inputs produced
        // the byte stream `a\0b\0d\0` and collided.  The length-prefix
        // framing distinguishes them.
        let a = CacheKey {
            checkpoint_hashes: vec!["a\0b".into()],
            calibration_data_hash: "d".into(),
            hook_ids_sorted: vec![],
            samples: 1,
            batch_size: 1,
            projections: vec![],
        };
        let b = CacheKey {
            checkpoint_hashes: vec!["a".into()],
            calibration_data_hash: "b\0d".into(),
            hook_ids_sorted: vec![],
            samples: 1,
            batch_size: 1,
            projections: vec![],
        };
        assert_ne!(a.digest(), b.digest());
    }
}
