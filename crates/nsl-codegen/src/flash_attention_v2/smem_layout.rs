//! SMEM layout + config validation for the v2 scalar emitter.
//!
//! Regions (all f16 for Q/K/V, f32 for S/P):
//!   Q   tile: offset 0,                 bytes = block_q  × head_dim × 2
//!   K/V tile: offset Q_bytes,            bytes = block_kv × head_dim × 2  (V reuses)
//!   S/P rows: offset Q_bytes + KV_bytes, bytes = 4 warps × block_kv × 4

use crate::flash_attention::FlashAttentionConfig;

#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Runtime validation called by `synthesize_flash_attention_ptx_v2`.
pub fn validate_scalar_v2_config(config: &FlashAttentionConfig) -> Result<(), ConfigError> {
    const ALLOWED_BLOCK_Q:  &[i64] = &[4, 8, 16, 32, 64, 128];
    const ALLOWED_BLOCK_KV: &[i64] = &[16, 32, 64, 128];
    const ALLOWED_HEAD_DIM: &[i64] = &[32, 64, 128, 256];
    const ALLOWED_GQA:      &[u32] = &[1, 2, 4, 8];
    const SMEM_BUDGET_BYTES: u32 = 48 * 1024;

    if !ALLOWED_BLOCK_Q.contains(&config.block_q) {
        return Err(ConfigError(format!(
            "block_q = {}: must be one of {:?}", config.block_q, ALLOWED_BLOCK_Q
        )));
    }
    if config.block_q % 4 != 0 {
        return Err(ConfigError(format!(
            "block_q = {}: must be a multiple of 4 (warp-per-row contract)",
            config.block_q
        )));
    }
    if !ALLOWED_BLOCK_KV.contains(&config.block_kv) {
        return Err(ConfigError(format!(
            "block_kv = {}: must be one of {:?}", config.block_kv, ALLOWED_BLOCK_KV
        )));
    }
    if !ALLOWED_HEAD_DIM.contains(&config.head_dim) {
        return Err(ConfigError(format!(
            "head_dim = {}: must be one of {:?}", config.head_dim, ALLOWED_HEAD_DIM
        )));
    }
    if !ALLOWED_GQA.contains(&config.gqa_group_size) {
        return Err(ConfigError(format!(
            "gqa_group_size = {}: must be one of {:?}",
            config.gqa_group_size, ALLOWED_GQA
        )));
    }

    let q_bytes  = (config.block_q  * config.head_dim * 2) as u32;
    let kv_bytes = (config.block_kv * config.head_dim * 2) as u32;
    let sp_bytes = 4 * (config.block_kv as u32) * 4;
    let total    = q_bytes + kv_bytes + sp_bytes;
    if total > SMEM_BUDGET_BYTES {
        return Err(ConfigError(format!(
            "SMEM total {} bytes exceeds 48 KB budget (Q={} KV={} SP={})",
            total, q_bytes, kv_bytes, sp_bytes
        )));
    }
    Ok(())
}

pub fn total_bytes(_config: &FlashAttentionConfig) -> u32 {
    // Populated in Task 3.
    0
}

pub fn q_offset(_config: &FlashAttentionConfig) -> u32 { 0 }
pub fn kv_offset(_config: &FlashAttentionConfig) -> u32 { 0 }
pub fn sp_offset(_config: &FlashAttentionConfig) -> u32 { 0 }
