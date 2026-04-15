//! SMEM layout + config validation for the v2 scalar emitter.
//!
//! Regions (all f16 for Q/K/V, f32 for S/P):
//!   Q   tile: offset 0,                 bytes = block_q  × head_dim × 2
//!   K/V tile: offset Q_bytes,            bytes = block_kv × head_dim × 2  (V reuses)
//!   S/P rows: offset Q_bytes + KV_bytes, bytes = 4 warps × block_kv × 4

use crate::flash_attention::FlashAttentionConfig;

// Supported-config matrix. Published so Task 3's per-config iteration
// tests and downstream phase emitters can consume the same lists the
// validator uses (single source of truth — no duplication).
pub const ALLOWED_BLOCK_Q:   &[i64] = &[4, 8, 16, 32, 64, 128];
pub const ALLOWED_BLOCK_KV:  &[i64] = &[16, 32, 64, 128];
pub const ALLOWED_HEAD_DIM:  &[i64] = &[32, 64, 128, 256];
pub const ALLOWED_GQA:       &[u32] = &[1, 2, 4, 8];
pub const SMEM_BUDGET_BYTES: u32    = 48 * 1024;

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

    // Derive region sizes from the layout helpers so the validator and
    // emitter stay in sync automatically if f16/f32 storage decisions
    // ever shift.
    let kv_start = kv_offset(config);
    let sp_start = sp_offset(config);
    let total    = total_bytes(config);
    let q_bytes  = kv_start;              // Q region: [0, kv_start)
    let kv_bytes = sp_start - kv_start;   // KV region: [kv_start, sp_start)
    let sp_bytes = total - sp_start;      // SP region: [sp_start, total)
    if total > SMEM_BUDGET_BYTES {
        return Err(ConfigError(format!(
            "SMEM total {} bytes exceeds 48 KB budget (Q={} KV={} SP={})",
            total, q_bytes, kv_bytes, sp_bytes
        )));
    }
    Ok(())
}

/// Q tile always starts at byte 0 (parameter intentionally unused — offset is a constant).
pub fn q_offset(_config: &FlashAttentionConfig) -> u32 { 0 }

/// KV tile starts immediately after the Q tile (block_q × head_dim × 2 bytes, f16).
pub fn kv_offset(config: &FlashAttentionConfig) -> u32 {
    (config.block_q * config.head_dim * 2) as u32
}

/// S/P scratch region starts immediately after the KV tile (block_kv × head_dim × 2 bytes, f16).
pub fn sp_offset(config: &FlashAttentionConfig) -> u32 {
    kv_offset(config) + (config.block_kv * config.head_dim * 2) as u32
}

/// Total SMEM bytes: Q tile + KV tile + S/P rows (4 warps × block_kv × 4 bytes, f32).
/// When `csha.fused_projections`, also includes Wq/Wk/Wv weight tile slots.
/// When `csha.fused_output_proj`, also includes Wo tile + x_residual input slot.
pub fn total_bytes(config: &FlashAttentionConfig) -> u32 {
    let base = sp_offset(config) + 4 * (config.block_kv as u32) * 4;
    base + wq_tile_bytes(config) + wk_tile_bytes(config) + wv_tile_bytes(config)
        + wo_tile_bytes(config) + x_residual_bytes(config)
}

/// Wq weight tile bytes when `csha.fused_projections` is set: `d_model × head_dim × 2` (f16).
/// Returns 0 when `fused_projections` is false or `d_model == 0`.
pub fn wq_tile_bytes(config: &FlashAttentionConfig) -> u32 {
    if config.csha.as_ref().map_or(false, |c| c.fused_projections) {
        let d_model = config.csha.as_ref().map_or(0, |c| c.d_model);
        2 * d_model * (config.head_dim as u32)
    } else {
        0
    }
}

/// Wk weight tile bytes — same as Wq.
pub fn wk_tile_bytes(config: &FlashAttentionConfig) -> u32 {
    wq_tile_bytes(config)
}

/// Wv weight tile bytes — same as Wq.
pub fn wv_tile_bytes(config: &FlashAttentionConfig) -> u32 {
    wq_tile_bytes(config)
}

/// Wo output projection tile bytes when `csha.fused_output_proj` is set:
/// `head_dim * d_model * 2` (f16). Returns 0 when not enabled or `d_model == 0`.
pub fn wo_tile_bytes(config: &FlashAttentionConfig) -> u32 {
    if config.csha.as_ref().map_or(false, |c| c.fused_output_proj) {
        let d_model = config.csha.as_ref().map_or(0, |c| c.d_model);
        2 * (config.head_dim as u32) * d_model
    } else {
        0
    }
}

/// Residual input tile bytes when `csha.fused_output_proj` is set:
/// `block_q * d_model * 2` (f16). Returns 0 when not enabled or `d_model == 0`.
pub fn x_residual_bytes(config: &FlashAttentionConfig) -> u32 {
    if config.csha.as_ref().map_or(false, |c| c.fused_output_proj) {
        let d_model = config.csha.as_ref().map_or(0, |c| c.d_model);
        2 * (config.block_q as u32) * d_model
    } else {
        0
    }
}

/// SmemLayout captures all per-config SMEM region sizes.
#[derive(Debug)]
pub struct SmemLayout {
    pub q_tile_bytes: usize,
    pub kv_tile_bytes: usize,
    pub sp_tile_bytes: usize,
    pub wq_tile_bytes: usize,
    pub wk_tile_bytes: usize,
    pub wv_tile_bytes: usize,
    pub wo_tile_bytes: usize,
    pub x_residual_bytes: usize,
    pub total_bytes: usize,
}

/// Compute the full SMEM layout for a given config.
pub fn compute_layout(config: &FlashAttentionConfig) -> SmemLayout {
    let q_bytes      = (config.block_q  * config.head_dim * 2) as usize;
    let kv_bytes     = (config.block_kv * config.head_dim * 2) as usize;
    let sp_bytes     = (4 * config.block_kv as u32 * 4) as usize;
    let wq_bytes     = wq_tile_bytes(config) as usize;
    let wk_bytes     = wk_tile_bytes(config) as usize;
    let wv_bytes     = wv_tile_bytes(config) as usize;
    let wo_bytes     = wo_tile_bytes(config) as usize;
    let xres_bytes   = x_residual_bytes(config) as usize;
    SmemLayout {
        q_tile_bytes: q_bytes,
        kv_tile_bytes: kv_bytes,
        sp_tile_bytes: sp_bytes,
        wq_tile_bytes: wq_bytes,
        wk_tile_bytes: wk_bytes,
        wv_tile_bytes: wv_bytes,
        wo_tile_bytes: wo_bytes,
        x_residual_bytes: xres_bytes,
        total_bytes: q_bytes + kv_bytes + sp_bytes + wq_bytes + wk_bytes + wv_bytes
            + wo_bytes + xres_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 75,
            csha: None,
        }
    }

    #[test]
    fn a5_wo_fits_in_smem_for_worst_case_matrix_config() {
        let mut cfg = base_cfg();
        cfg.block_q = 64; cfg.block_kv = 64; cfg.head_dim = 64;
        cfg.rope_q = true; cfg.causal = true;
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            fused_output_proj: true,
            d_model: 128,
            ..CshaExtras::default()
        });
        let layout = compute_layout(&cfg);
        assert!(
            layout.total_bytes <= 48 * 1024,
            "worst-case matrix config exceeds 48 KB SMEM: {} bytes", layout.total_bytes
        );
    }

    #[test]
    fn a3_smem_includes_wq_wk_wv_tiles() {
        let mut cfg = base_cfg();
        cfg.csha = Some(CshaExtras { fused_projections: true, d_model: 128, ..CshaExtras::default() });
        let layout = compute_layout(&cfg);
        assert!(layout.wq_tile_bytes > 0, "wq tile missing");
        assert!(layout.wk_tile_bytes > 0, "wk tile missing");
        assert!(layout.wv_tile_bytes > 0, "wv tile missing");
        // f16: 2 bytes × d_model × head_dim
        let d_model = cfg.csha.as_ref().unwrap().d_model as usize;
        assert_eq!(layout.wq_tile_bytes, 2 * d_model * cfg.head_dim as usize);
    }
}
