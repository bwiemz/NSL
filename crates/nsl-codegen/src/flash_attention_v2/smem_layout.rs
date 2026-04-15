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
/// 48 KB: CUDA static `.shared` cap (all SM generations).
/// Configs within this budget use a statically-sized shmem array in PTX;
/// configs above it use an `extern .shared` declaration and require dynamic
/// SMEM opt-in via `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`.
pub const SMEM_BUDGET_BYTES: u32         = 48 * 1024;
/// 99 KB: dynamic shared memory opt-in cap.
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` returns 101376 bytes
/// (99 KB) on RTX 5070 Ti (sm_120 / Blackwell) with CUDA 13.2.  sm_90/sm_89
/// (Ada/Hopper) also report 99 KB.  sm_86 supports ~100 KB.
/// We use 99 KB (101376 bytes) as the conservative cross-generation limit.
/// Configs in (SMEM_BUDGET_BYTES, SMEM_DYNAMIC_BUDGET_BYTES] use extern
/// .shared PTX + runtime cuFuncSetAttribute opt-in.
pub const SMEM_DYNAMIC_BUDGET_BYTES: u32 = 99 * 1024;

#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Which kernel direction the caller intends to emit. Used by
/// `validate_scalar_v2_config` to pick the right SMEM budget: the
/// backward pass needs additional tiles (`dQ`, `dK`, `dV`, recomputed
/// `P`) on top of the forward budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// Additional SMEM bytes the Tier C fused backward pass needs on top of
/// the forward total. Covers the recomputed `P` tile plus the `dS`
/// tile plus three f32 gradient accumulators (`dQ`, `dK`, `dV`).
/// The gradient accumulators are ultimately kept in registers
/// (`%f_dq_*/%f_dk_*/%f_dv_*`) but the SMEM is budgeted conservatively
/// so future refactors to SMEM-based tiles don't trip the validator.
///
/// Layout (all f32 — backward uses higher-precision accumulators):
///   P     tile: block_q × block_kv × 4 (recomputed)
///   dS    tile: block_q × block_kv × 4
///   dQ    tile: block_q × head_dim × 4 (reserved, register-held today)
///   dK    tile: block_kv × head_dim × 4 (reserved)
///   dV    tile: block_kv × head_dim × 4 (reserved)
pub fn backward_extra_bytes(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    let p = bq * bkv * 4;
    let ds = bq * bkv * 4;
    let dq = bq * hd * 4;
    let dk = bkv * hd * 4;
    let dv = bkv * hd * 4;
    p + ds + dq + dk + dv
}

/// SMEM byte offset of the recomputed P tile (Tier C backward).
/// P is stored f32 row-major `[block_q, block_kv]` at the start of
/// the backward extras region.
pub fn backward_p_offset(config: &FlashAttentionConfig) -> u32 {
    total_bytes(config)
}

/// SMEM byte offset of the dS tile (Tier C backward), immediately
/// after the P tile.
pub fn backward_ds_offset(config: &FlashAttentionConfig) -> u32 {
    backward_p_offset(config) + (config.block_q * config.block_kv * 4) as u32
}

/// Runtime validation called by `synthesize_flash_attention_ptx_v2`.
///
/// `direction` controls whether the backward-pass extra SMEM tiles are
/// added to the budget before the 99 KB cap check. Forward: unchanged
/// from Tier A. Backward (Tier C): adds `backward_extra_bytes` to the
/// forward total.
pub fn validate_scalar_v2_config(
    config: &FlashAttentionConfig,
    direction: Direction,
) -> Result<(), ConfigError> {
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

    // Fused-projection constraint: the K pre-pass writes exactly `block_q`
    // K rows (4 warps × ceil(block_q/4) iters = block_q rows).  The K tile
    // in SMEM holds `block_kv` rows.  block_q ≠ block_kv causes either an
    // under-fill (garbage S-compute) or a SMEM out-of-bounds write.
    if config.csha.as_ref().map_or(false, |c| c.fused_projections)
        && config.block_q != config.block_kv
    {
        return Err(ConfigError(format!(
            "fused_projections requires block_q == block_kv (got block_q={}, block_kv={}); \
             the K pre-pass writes exactly block_q rows into the block_kv-row K tile",
            config.block_q, config.block_kv
        )));
    }

    // Derive region sizes from the layout helpers so the validator and
    // emitter stay in sync automatically if f16/f32 storage decisions
    // ever shift.
    let kv_start = kv_offset(config);
    let sp_start = sp_offset(config);
    let fwd_total = total_bytes(config);
    let extra = match direction {
        Direction::Forward => 0,
        Direction::Backward => backward_extra_bytes(config),
    };
    let total = fwd_total + extra;
    let q_region  = kv_start;              // Q region: [0, kv_start)
    let kv_region = sp_start - kv_start;   // KV region: [kv_start, sp_start)
    let sp_region = fwd_total - sp_start;  // SP + weight + save region: [sp_start, fwd_total)
    if total > SMEM_DYNAMIC_BUDGET_BYTES {
        return Err(ConfigError(match direction {
            Direction::Forward => format!(
                "SMEM total {} bytes ({:.1} KB) exceeds 99 KB dynamic SMEM budget \
                 (Q={} KV={} SP+rest={}); reduce head_dim, block_q/block_kv, or d_model",
                total, total as f32 / 1024.0, q_region, kv_region, sp_region
            ),
            Direction::Backward => format!(
                "CSHA fused Backward rejected: {} bytes > {} byte cap at \
                 (block_q={}, head_dim={}); forward={} backward_extra={} \
                 (P+dQ+dK+dV). Reduce head_dim, block_q/block_kv, or d_model.",
                total, SMEM_DYNAMIC_BUDGET_BYTES,
                config.block_q, config.head_dim,
                fwd_total, extra
            ),
        }));
    }
    // Configs in (48 KB, 99 KB] use dynamic SMEM (extern .shared in PTX +
    // cuFuncSetAttribute opt-in at launch).  These are valid configurations.
    Ok(())
}

/// Returns true when `total_bytes(config)` exceeds the 48 KB static SMEM cap.
///
/// When true, the PTX emitter declares `extern .shared` (dynamic SMEM) instead
/// of the static `.shared .align 16 .b8 shmem[N]` form.  The runtime must then
/// call `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
/// total_bytes)` before launch and pass `total_bytes` as `sharedMemBytes` to
/// `cuLaunchKernel`.
pub fn needs_dynamic_smem(config: &FlashAttentionConfig) -> bool {
    total_bytes(config) > SMEM_BUDGET_BYTES
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

/// S/P scratch region size (bytes).
///
/// Standard path: 4 warps × block_kv × 4 bytes (f32) — one row per warp.
/// Fused-projection path: iters × 4 warps × block_kv × 4 bytes — one row per
/// (q_tile_iter, warp_id) pair.  The split-loop design (all S-passes before all
/// PV-accums) requires keeping P values for EVERY q_tile_iter live at the same
/// time; without the expanded region each S-pass overwrites the previous iter's P.
pub fn sp_bytes(config: &FlashAttentionConfig) -> u32 {
    let warps     = 4u32;
    let block_kv  = config.block_kv as u32;
    let base      = warps * block_kv * 4;
    if config.csha.as_ref().map_or(false, |c| c.fused_projections) {
        let iters = (config.block_q as u32).div_ceil(4);
        iters * base
    } else {
        base
    }
}

/// Total SMEM bytes: Q tile + KV tile + S/P rows (4 warps × block_kv × 4 bytes, f32).
/// When `csha.fused_projections`, also includes Wq/Wk/Wv weight tile slots and
/// a softmax-state save area for the K/V pre-pass split design.
/// When `csha.fused_output_proj`, also includes Wo tile + x_residual input slot.
pub fn total_bytes(config: &FlashAttentionConfig) -> u32 {
    let base = sp_offset(config) + sp_bytes(config);
    base + wq_tile_bytes(config) + wk_tile_bytes(config) + wv_tile_bytes(config)
        + wo_tile_bytes(config) + x_residual_bytes(config)
        + softmax_save_bytes(config)
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

/// Softmax-state save area bytes when `fused_projections` is set.
///
/// The K/V pre-pass design splits the attention loop into S-compute and
/// PV-accum passes with a V pre-pass between them.  Between the two passes,
/// per-(q_tile_iter, warp_id) `row_max` and `row_sum` must be saved.
///
/// Layout: 4 warps × iters × 2 f32 × 4 bytes
/// Slot for (warp, iter): byte offset = warp*iters*8 + iter*8
pub fn softmax_save_bytes(config: &FlashAttentionConfig) -> u32 {
    if config.csha.as_ref().map_or(false, |c| c.fused_projections) {
        let iters = (config.block_q as u32).div_ceil(4);
        4 * iters * 2 * 4  // 4 warps × iters × (row_max, row_sum) × sizeof(f32)
    } else {
        0
    }
}

/// Byte offset of the softmax-state save area within SMEM.
/// Valid only when `fused_projections` is true; returns 0 otherwise.
pub fn softmax_save_offset(config: &FlashAttentionConfig) -> u32 {
    let base = sp_offset(config) + sp_bytes(config);
    base + wq_tile_bytes(config) + wk_tile_bytes(config) + wv_tile_bytes(config)
        + wo_tile_bytes(config) + x_residual_bytes(config)
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
    let sp_bytes     = sp_bytes(config) as usize;
    let wq_bytes     = wq_tile_bytes(config) as usize;
    let wk_bytes     = wk_tile_bytes(config) as usize;
    let wv_bytes     = wv_tile_bytes(config) as usize;
    let wo_bytes     = wo_tile_bytes(config) as usize;
    let xres_bytes   = x_residual_bytes(config) as usize;
    let save_bytes   = softmax_save_bytes(config) as usize;
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
            + wo_bytes + xres_bytes + save_bytes,
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
    fn a5_wo_fits_in_smem_for_fused_proj_output_proj() {
        // Verify that a fused_projections + fused_output_proj config fits within
        // the 48 KB SMEM budget.  The SP region is expanded to iters×4_warps×block_kv
        // when fused_projections=true (required to avoid P overwrite in the split-loop
        // design), so only smaller configs fit the budget.
        // Config: block_q=32, block_kv=32, head_dim=32, d_model=32
        // SP(fused) = 8*4*32*4 = 4096 B; Wq=Wk=Wv=Wo=2048 B each; total ~ 16640 B.
        let mut cfg = base_cfg();
        cfg.rope_q = true; cfg.causal = true;
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            fused_output_proj: true,
            d_model: 32,
            ..CshaExtras::default()
        });
        let layout = compute_layout(&cfg);
        assert!(
            layout.total_bytes <= 48 * 1024,
            "fused proj+output config exceeds 48 KB SMEM: {} bytes", layout.total_bytes
        );
    }

    // ── T2.1 Direction-parameterised validator tests ──────────────────────

    fn base_cfg_fused_forward(
        block_q: i64, block_kv: i64, head_dim: i64, heads: u32, d_model: u32,
    ) -> FlashAttentionConfig {
        let _ = heads;
        FlashAttentionConfig {
            block_q, block_kv, head_dim,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 75,
            csha: Some(CshaExtras {
                fused_projections: true,
                d_model,
                ..CshaExtras::default()
            }),
        }
    }

    fn base_cfg_fused_backward(
        block_q: i64, block_kv: i64, head_dim: i64, heads: u32, d_model: u32,
    ) -> FlashAttentionConfig {
        let mut cfg = base_cfg_fused_forward(block_q, block_kv, head_dim, heads, d_model);
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model,
            ..CshaExtras::default()
        });
        cfg
    }

    #[test]
    fn direction_backward_accepts_smallest_config() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        assert!(
            validate_scalar_v2_config(&cfg, Direction::Backward).is_ok(),
            "smallest backward config must pass (total = forward + P+dQ+dK+dV)",
        );
    }

    #[test]
    fn direction_backward_rejects_over_budget_with_detailed_diagnostic() {
        let cfg = base_cfg_fused_backward(64, 64, 64, 8, 64);
        let err = validate_scalar_v2_config(&cfg, Direction::Backward)
            .expect_err("expected backward over-budget rejection");
        let msg = format!("{err}");
        assert!(msg.contains("bytes >"), "err must include byte comparison: {msg}");
        assert!(msg.contains("block_q=64"), "err must include block_q: {msg}");
        assert!(msg.contains("head_dim=64"), "err must include head_dim: {msg}");
        assert!(msg.contains("Backward"), "err must name direction: {msg}");
    }

    #[test]
    fn direction_forward_budget_unchanged_by_phase_2() {
        let cfg = base_cfg_fused_forward(32, 32, 32, 4, 32);
        assert!(validate_scalar_v2_config(&cfg, Direction::Forward).is_ok());
    }

    #[test]
    fn direction_backward_adds_extra_bytes() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let extra = backward_extra_bytes(&cfg);
        // P = 32*32*4 = 4096; dQ = dK = dV = 32*32*4 = 4096 each → 16384 total.
        // P + dS + dQ + dK + dV, each 32*32*4 = 4096 at this config.
        assert_eq!(extra, 20480, "backward_extra_bytes = P+dS+dQ+dK+dV");
    }

    #[test]
    fn direction_forward_still_rejects_big_config() {
        // Config that was already over-budget forward-side stays rejected.
        let cfg = base_cfg_fused_forward(128, 128, 256, 4, 128);
        assert!(validate_scalar_v2_config(&cfg, Direction::Forward).is_err());
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
