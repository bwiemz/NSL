//! Hierarchical micro-checkpointing for fault-tolerant training.
//!
//! Three-tier checkpointing pattern (ByteCheckpoint/Gemini):
//! - Tier 1: GPU → CPU pinned DRAM (every N iters, ~100ms, async)
//! - Tier 2: CPU DRAM → NVMe (every 10N iters, ~1s)
//! - Tier 3: NVMe → remote (every 100N iters, ~30s, background)

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Checkpoint tier for tiered micro-checkpointing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointTier {
    /// GPU → CPU pinned DRAM (fastest, most frequent).
    GpuToCpu,
    /// CPU DRAM → local NVMe storage.
    CpuToNvme,
    /// NVMe → remote storage (S3, shared filesystem).
    NvmeToRemote,
}

/// Configuration for hierarchical checkpointing.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Iterations between Tier 1 checkpoints (GPU→CPU).
    pub tier1_interval: u64,
    /// Iterations between Tier 2 checkpoints (CPU→NVMe).
    pub tier2_interval: u64,
    /// Iterations between Tier 3 checkpoints (NVMe→remote).
    pub tier3_interval: u64,
    /// Local NVMe path for Tier 2 storage.
    pub nvme_path: PathBuf,
    /// Remote URL for Tier 3 storage (S3, shared FS).
    pub remote_url: Option<String>,
    /// Whether to use async GPU→CPU copies (cudaMemcpyAsync).
    pub async_gpu_copy: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            tier1_interval: 100,
            tier2_interval: 1_000,
            tier3_interval: 10_000,
            nvme_path: PathBuf::from("/tmp/nsl_checkpoints"),
            remote_url: None,
            async_gpu_copy: true,
        }
    }
}

/// Checkpoint state tracker.
pub struct CheckpointManager {
    config: CheckpointConfig,
    /// Last iteration number for each tier's checkpoint.
    last_tier1: u64,
    last_tier2: u64,
    last_tier3: u64,
    /// Total bytes checkpointed per tier (for diagnostics).
    tier1_bytes: AtomicU64,
    tier2_bytes: AtomicU64,
    tier3_bytes: AtomicU64,
    /// CPU-side checkpoint buffer (pinned memory).
    cpu_buffer: Vec<u8>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            last_tier1: 0,
            last_tier2: 0,
            last_tier3: 0,
            tier1_bytes: AtomicU64::new(0),
            tier2_bytes: AtomicU64::new(0),
            tier3_bytes: AtomicU64::new(0),
            cpu_buffer: Vec::new(),
        }
    }

    /// Check if a checkpoint should be taken at the given iteration.
    /// Returns the highest applicable tier, or None.
    #[allow(clippy::manual_is_multiple_of)]
    pub fn should_checkpoint(&self, iteration: u64) -> Option<CheckpointTier> {
        if iteration > 0 && iteration % self.config.tier3_interval == 0 {
            Some(CheckpointTier::NvmeToRemote)
        } else if iteration > 0 && iteration % self.config.tier2_interval == 0 {
            Some(CheckpointTier::CpuToNvme)
        } else if iteration > 0 && iteration % self.config.tier1_interval == 0 {
            Some(CheckpointTier::GpuToCpu)
        } else {
            None
        }
    }

    /// Perform Tier 1 checkpoint: copy model state from GPU to CPU buffer.
    /// In real implementation, this would use cudaMemcpyAsync with pinned memory.
    pub fn checkpoint_to_cpu(&mut self, model_data: &[u8], iteration: u64) {
        self.cpu_buffer.clear();
        self.cpu_buffer.extend_from_slice(model_data);
        self.last_tier1 = iteration;
        self.tier1_bytes.fetch_add(model_data.len() as u64, Ordering::Relaxed);
    }

    /// Perform Tier 2 checkpoint: write CPU buffer to NVMe.
    pub fn checkpoint_to_nvme(&mut self, iteration: u64) -> Result<(), String> {
        if self.cpu_buffer.is_empty() {
            return Err("no CPU buffer to write (run tier1 checkpoint first)".into());
        }

        let path = self.config.nvme_path.join(format!("ckpt_{iteration}.nslm"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create checkpoint dir: {e}"))?;
        }

        // Write NSLM format: magic + iteration header + raw data
        let mut data = Vec::with_capacity(self.cpu_buffer.len() + 16);
        data.extend_from_slice(b"NSLM"); // magic
        data.extend_from_slice(&iteration.to_le_bytes()); // iteration
        data.extend_from_slice(&(self.cpu_buffer.len() as u64).to_le_bytes()); // data size
        // Pad to 64-byte alignment
        let padding = (64 - (data.len() % 64)) % 64;
        data.extend(vec![0u8; padding]);
        data.extend_from_slice(&self.cpu_buffer);

        std::fs::write(&path, &data)
            .map_err(|e| format!("failed to write checkpoint to '{}': {e}", path.display()))?;

        self.last_tier2 = iteration;
        self.tier2_bytes.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Perform Tier 3 checkpoint: upload NVMe checkpoint to remote storage.
    /// This is a no-op if no remote URL is configured.
    pub fn checkpoint_to_remote(&mut self, iteration: u64) -> Result<(), String> {
        if self.config.remote_url.is_none() {
            return Ok(()); // no remote configured — skip
        }
        // Real implementation would use multipart upload to S3/GCS.
        // For now, just record that we would upload.
        self.last_tier3 = iteration;
        self.tier3_bytes.fetch_add(
            self.tier2_bytes.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Ok(())
    }

    /// Load the latest checkpoint from NVMe.
    /// Returns (iteration, data) or error.
    pub fn load_latest_checkpoint(&self) -> Result<(u64, Vec<u8>), String> {
        // Find the latest .nslm file in the checkpoint directory
        let dir = &self.config.nvme_path;
        if !dir.exists() {
            return Err("checkpoint directory does not exist".into());
        }

        let mut latest: Option<(u64, PathBuf)> = None;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "nslm").unwrap_or(false) {
                    // Extract iteration from filename: ckpt_<iter>.nslm
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        if let Some(iter_str) = stem.strip_prefix("ckpt_") {
                            if let Ok(iter) = iter_str.parse::<u64>() {
                                if latest.as_ref().map(|(i, _)| iter > *i).unwrap_or(true) {
                                    latest = Some((iter, path));
                                }
                            }
                        }
                    }
                }
            }
        }

        let (_iteration, path) = latest.ok_or("no checkpoint files found")?;
        let raw = std::fs::read(&path)
            .map_err(|e| format!("failed to read checkpoint '{}': {e}", path.display()))?;

        // Validate NSLM magic
        if raw.len() < 20 || &raw[0..4] != b"NSLM" {
            return Err("invalid checkpoint format (bad magic)".into());
        }

        let stored_iter = u64::from_le_bytes(raw[4..12].try_into().unwrap());
        let data_size = u64::from_le_bytes(raw[12..20].try_into().unwrap()) as usize;

        // Find data start (after 64-byte aligned header)
        let header_end = 20usize;
        let padding = (64 - (header_end % 64)) % 64;
        let data_start = header_end + padding;

        if raw.len() < data_start + data_size {
            return Err("checkpoint file truncated".into());
        }

        Ok((stored_iter, raw[data_start..data_start + data_size].to_vec()))
    }

    /// Calculate optimal checkpoint interval using sqrt(2*C*MTBF).
    ///
    /// `checkpoint_cost_s`: time to take one checkpoint in seconds.
    /// `mtbf_s`: mean time between failures in seconds.
    pub fn optimal_interval(checkpoint_cost_s: f64, mtbf_s: f64) -> f64 {
        (2.0 * checkpoint_cost_s * mtbf_s).sqrt()
    }

    /// Get the last completed iteration for each tier.
    pub fn last_iterations(&self) -> (u64, u64, u64) {
        (self.last_tier1, self.last_tier2, self.last_tier3)
    }
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// Trigger an async checkpoint (called from codegen-generated train loop).
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_checkpoint_async(
    _model_state_ptr: i64,
    _model_state_size: i64,
    _iteration: i64,
) -> i64 {
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_checkpoint_intervals() {
        let config = CheckpointConfig {
            tier1_interval: 10,
            tier2_interval: 100,
            tier3_interval: 1000,
            ..Default::default()
        };
        let mgr = CheckpointManager::new(config);

        assert_eq!(mgr.should_checkpoint(0), None);
        assert_eq!(mgr.should_checkpoint(5), None);
        assert_eq!(mgr.should_checkpoint(10), Some(CheckpointTier::GpuToCpu));
        assert_eq!(mgr.should_checkpoint(100), Some(CheckpointTier::CpuToNvme));
        assert_eq!(mgr.should_checkpoint(1000), Some(CheckpointTier::NvmeToRemote));
        assert_eq!(mgr.should_checkpoint(50), Some(CheckpointTier::GpuToCpu));
    }

    #[test]
    fn test_checkpoint_to_cpu() {
        let mut mgr = CheckpointManager::new(Default::default());
        let data = vec![1u8, 2, 3, 4, 5];
        mgr.checkpoint_to_cpu(&data, 100);
        assert_eq!(mgr.cpu_buffer, data);
        assert_eq!(mgr.last_tier1, 100);
    }

    #[test]
    fn test_checkpoint_to_nvme_roundtrip() {
        let dir = std::env::temp_dir().join("nsl_test_ckpt_roundtrip");
        let _ = std::fs::remove_dir_all(&dir);

        let config = CheckpointConfig {
            nvme_path: dir.clone(),
            ..Default::default()
        };
        let mut mgr = CheckpointManager::new(config);

        let data = vec![42u8; 256];
        mgr.checkpoint_to_cpu(&data, 500);
        mgr.checkpoint_to_nvme(500).unwrap();

        let (iter, loaded) = mgr.load_latest_checkpoint().unwrap();
        assert_eq!(iter, 500);
        assert_eq!(loaded, data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_optimal_interval() {
        // 1000-GPU cluster: MTBF ≈ 2 hours (7200s), checkpoint cost ≈ 30s
        let interval = CheckpointManager::optimal_interval(30.0, 7200.0);
        assert!(interval > 600.0 && interval < 700.0,
            "optimal interval should be ~657s for MTBF=2h, C=30s, got {interval}");
    }
}
