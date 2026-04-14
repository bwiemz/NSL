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
pub fn validate_scalar_v2_config(_config: &FlashAttentionConfig) -> Result<(), ConfigError> {
    // Populated in Task 2.
    Ok(())
}

pub fn total_bytes(_config: &FlashAttentionConfig) -> u32 {
    // Populated in Task 3.
    0
}

pub fn q_offset(_config: &FlashAttentionConfig) -> u32 { 0 }
pub fn kv_offset(_config: &FlashAttentionConfig) -> u32 { 0 }
pub fn sp_offset(_config: &FlashAttentionConfig) -> u32 { 0 }
