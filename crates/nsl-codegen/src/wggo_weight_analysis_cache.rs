//! WGGO — Stage-3 sidecar cache.
//!
//! Weight analysis on a multi-GB checkpoint is deterministic given the
//! checkpoint bytes, so re-running the pass on every compile is waste.
//! This module stores the [`WeightAnalysisReport`] alongside the
//! checkpoint as `<ckpt>.wggo-importance.json`, keyed by the
//! checkpoint's SHA-256.
//!
//! The cache is intentionally best-effort: any I/O or deserialisation
//! failure silently falls back to a fresh analysis.  The only way this
//! module can cause a compile to fail is by surfacing an error from a
//! caller that opts into strict mode (not exposed yet).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::wggo_weight_analysis::{LayerImportance, WeightAnalysisReport};

const CACHE_VERSION: u32 = 1;
const CACHE_SUFFIX: &str = ".wggo-importance.json";

/// Serialised payload written to disk.  Version tag lets us invalidate
/// older caches without migration when the score format changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachePayload {
    version: u32,
    checkpoint_sha256: String,
    per_layer: Vec<LayerImportance>,
    layers_without_weights: u32,
}

/// Derive the sidecar cache path for a checkpoint at `ckpt_path`.
pub fn sidecar_path(ckpt_path: &Path) -> PathBuf {
    let mut s = ckpt_path.as_os_str().to_os_string();
    s.push(CACHE_SUFFIX);
    PathBuf::from(s)
}

/// Compute the SHA-256 of a file in 64 KiB chunks.
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

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Try to load a cached report for `ckpt_path`.  Returns `None` when
/// the cache is missing, stale, or unparseable — the caller should run
/// a fresh analysis in any of those cases.
pub fn try_load(ckpt_path: &Path) -> Option<WeightAnalysisReport> {
    let sidecar = sidecar_path(ckpt_path);
    if !sidecar.exists() {
        return None;
    }
    let current_hash = hash_file(ckpt_path).ok()?;
    let json = fs::read_to_string(&sidecar).ok()?;
    let payload: CachePayload = serde_json::from_str(&json).ok()?;
    if payload.version != CACHE_VERSION {
        return None;
    }
    if payload.checkpoint_sha256 != current_hash {
        return None;
    }
    Some(WeightAnalysisReport {
        per_layer: payload.per_layer,
        layers_without_weights: payload.layers_without_weights,
    })
}

/// Write `report` to the sidecar next to `ckpt_path`.  Errors are
/// returned to the caller but can be safely ignored — failure to
/// persist never breaks the compile.
pub fn store(ckpt_path: &Path, report: &WeightAnalysisReport) -> std::io::Result<()> {
    let current_hash = hash_file(ckpt_path)?;
    let payload = CachePayload {
        version: CACHE_VERSION,
        checkpoint_sha256: current_hash,
        per_layer: report.per_layer.clone(),
        layers_without_weights: report.layers_without_weights,
    };
    let json = serde_json::to_string_pretty(&payload).map_err(std::io::Error::other)?;
    let sidecar = sidecar_path(ckpt_path);
    let mut f = fs::File::create(&sidecar)?;
    f.write_all(json.as_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Build a unique temp path under the OS temp dir.
    fn temp_path(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let mut p = std::env::temp_dir();
        p.push(format!("wggo-cache-{tag}-{nanos}-{n}.bin"));
        p
    }

    fn cleanup(p: &Path) {
        let _ = fs::remove_file(p);
        let _ = fs::remove_file(sidecar_path(p));
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        let mut f = fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    fn sample_report() -> WeightAnalysisReport {
        WeightAnalysisReport {
            per_layer: vec![LayerImportance {
                head_scores: vec![1.0, 2.0, 3.0, 4.0],
                default_min_retained: 9.0,
            }],
            layers_without_weights: 0,
        }
    }

    #[test]
    fn sidecar_path_appends_suffix() {
        let p = Path::new("/tmp/model.nslweights");
        let sp = sidecar_path(p);
        assert!(sp.to_string_lossy().ends_with(".wggo-importance.json"));
    }

    #[test]
    fn hash_is_deterministic() {
        let p = temp_path("hash");
        write_bytes(&p, b"hello world");
        let h1 = hash_file(&p).unwrap();
        let h2 = hash_file(&p).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // 32 bytes → 64 hex chars
        cleanup(&p);
    }

    #[test]
    fn hash_differs_for_different_content() {
        let p1 = temp_path("diff1");
        let p2 = temp_path("diff2");
        write_bytes(&p1, b"AAA");
        write_bytes(&p2, b"BBB");
        assert_ne!(hash_file(&p1).unwrap(), hash_file(&p2).unwrap());
        cleanup(&p1);
        cleanup(&p2);
    }

    #[test]
    fn store_then_load_roundtrips() {
        let p = temp_path("roundtrip");
        write_bytes(&p, b"checkpoint-bytes");
        let rep = sample_report();
        store(&p, &rep).unwrap();
        let loaded = try_load(&p).expect("cache should load");
        assert_eq!(loaded.per_layer.len(), rep.per_layer.len());
        assert_eq!(
            loaded.per_layer[0].head_scores,
            rep.per_layer[0].head_scores
        );
        assert_eq!(
            loaded.per_layer[0].default_min_retained,
            rep.per_layer[0].default_min_retained
        );
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_changed_checkpoint() {
        let p = temp_path("stale");
        write_bytes(&p, b"original");
        store(&p, &sample_report()).unwrap();
        // Modify the checkpoint — hash changes → cache miss.
        write_bytes(&p, b"modified");
        assert!(try_load(&p).is_none());
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_missing_sidecar() {
        let p = temp_path("no-sidecar");
        write_bytes(&p, b"x");
        assert!(try_load(&p).is_none());
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_stale_version() {
        let p = temp_path("badver");
        write_bytes(&p, b"v");
        let sidecar = sidecar_path(&p);
        let json =
            r#"{"version":0,"checkpoint_sha256":"x","per_layer":[],"layers_without_weights":0}"#;
        fs::write(&sidecar, json).unwrap();
        assert!(try_load(&p).is_none());
        cleanup(&p);
    }

    #[test]
    fn cache_miss_on_unparseable_sidecar() {
        let p = temp_path("garbage");
        write_bytes(&p, b"v");
        let sidecar = sidecar_path(&p);
        fs::write(&sidecar, b"{not valid json").unwrap();
        assert!(try_load(&p).is_none());
        cleanup(&p);
    }
}
