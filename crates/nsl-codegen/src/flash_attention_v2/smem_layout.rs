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
///   V_in  tile: block_kv × head_dim × 2 (f16 V input for backward; can NOT
///              alias the forward kv_offset slot because backward's
///              emit_v otherwise clobbers K mid-kernel before ds_compute
///              reads it — see commit fixing bug #2).
///   x_norm    tile: block_q × d_model × 4 (recomputed x_norm for dproj +
///                   dx_norm chain; only present when csha.d_model > 0).
///   dx_norm   tile: block_q × d_model × 4 (dx_norm staging for dRMSNorm).
///   rms_strip:      block_q × 4         (per-row rms cache).
pub fn backward_extra_bytes(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    let p = bq * bkv * 4;
    let ds = bq * bkv * 4;
    let dq = bq * hd * 4;
    let dk = bkv * hd * 4;
    let dv = bkv * hd * 4;
    let v_in = bkv * hd * 2;
    let dm = config.csha.as_ref().map_or(0, |c| c.d_model);
    let x_norm = bq * dm * 4;
    let dx_norm = bq * dm * 4;
    let rms_strip = bq * 4;
    p + ds + dq + dk + dv + v_in + x_norm + dx_norm + rms_strip
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

/// SMEM byte offset of the dQ accumulator tile. f32 row-major
/// `[block_q, head_dim]`. Lives immediately after the dS tile.
///
/// Per-iter %f_dq_{slice} registers are flushed into this tile after
/// each q_tile_iter so finalize can read the full dQ rather than only
/// the last iter's register state.
pub fn backward_dq_offset(config: &FlashAttentionConfig) -> u32 {
    backward_ds_offset(config) + (config.block_q * config.block_kv * 4) as u32
}

/// SMEM byte offset of the dV accumulator tile. f32 row-major
/// `[block_kv, head_dim]`. Lives immediately after the dQ tile.
pub fn backward_dv_offset(config: &FlashAttentionConfig) -> u32 {
    backward_dq_offset(config) + (config.block_q * config.head_dim * 4) as u32
}

/// SMEM byte offset of the dK accumulator tile. f32 row-major
/// `[block_kv, head_dim]`. Lives immediately after the dV tile.
pub fn backward_dk_offset(config: &FlashAttentionConfig) -> u32 {
    backward_dv_offset(config) + (config.block_kv * config.head_dim * 4) as u32
}

/// SMEM byte offset of the backward-pass V input tile (f16 row-major
/// `[block_kv, head_dim]`). MUST be disjoint from the forward `kv_offset`
/// slot that holds K, because the backward's `kv_load::emit_v` would
/// otherwise overwrite K mid-kernel before `ds_compute` reads it.
pub fn backward_v_input_offset(config: &FlashAttentionConfig) -> u32 {
    backward_dk_offset(config) + (config.block_kv * config.head_dim * 4) as u32
}

/// SMEM byte offset of the recomputed x_norm tile. f32 row-major
/// `[block_q, d_model]`. Populated once by `emit_xnorm_recompute`
/// before the dproj/dRMSNorm phases. Lives after the V input tile.
pub fn backward_x_norm_offset(config: &FlashAttentionConfig) -> u32 {
    backward_v_input_offset(config) + (config.block_kv * config.head_dim * 2) as u32
}

/// SMEM byte offset of the dx_norm staging tile. f32 row-major
/// `[block_q, d_model]`. Populated by `emit_drmsnorm` phase 1 (chain-rule
/// contraction across dQ/dK/dV) and consumed by phase 2 (dx computation).
pub fn backward_dx_norm_offset(config: &FlashAttentionConfig) -> u32 {
    let dm = config.csha.as_ref().map_or(0, |c| c.d_model);
    backward_x_norm_offset(config) + (config.block_q as u32) * dm * 4
}

/// SMEM byte offset of the per-row rms cache. f32 `[block_q]`.
pub fn backward_rms_strip_offset(config: &FlashAttentionConfig) -> u32 {
    let dm = config.csha.as_ref().map_or(0, |c| c.d_model);
    backward_dx_norm_offset(config) + (config.block_q as u32) * dm * 4
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

    // Fused-projection asymmetric-tile constraint: the K and V pre-passes
    // are driven by `kv_iters = ceil(block_kv / 4)` separately from the Q
    // sweep / S-compute / PV-accum loop (which uses `iters =
    // ceil(block_q / 4)`).  Both block_q and block_kv are already
    // constrained to multiples of 4 via `ALLOWED_BLOCK_Q`/`ALLOWED_BLOCK_KV`,
    // which guarantees the warp-per-row contract (4 warps × N iters = 4N
    // rows, exactly matches the tile size for both dimensions).  Asymmetric
    // configs (`block_q != block_kv`) are therefore valid and no additional
    // predicate is needed here.
    if config.csha.as_ref().is_some_and(|c| c.fused_projections)
        && (config.block_kv % 4 != 0)
    {
        return Err(ConfigError(format!(
            "fused_projections requires block_kv to be a multiple of 4 \
             (warp-per-row contract); got block_kv={}",
            config.block_kv
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
    // PCA Tier A: when `segment_masked`, the FA prelude reserves an
    // additional `DEFAULT_SMEM_SEGMENT_BUDGET`-byte `seg_smem` region —
    // separately-declared static SMEM on the forward side, and embedded
    // at the tail of the extern shmem allocation on the backward side
    // (see `phases/{forward,backward}/prelude.rs`). Either way the device
    // sees `total + seg_overhead` bytes per CTA, so the validator must
    // account for it or `segment_masked = true` configs that fit the
    // 99 KB cap on paper will fail at launch with insufficient SMEM.
    let seg_overhead = if config.segment_masked {
        crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32
    } else {
        0
    };
    let total = fwd_total + extra + seg_overhead;
    let q_region  = kv_start;              // Q region: [0, kv_start)
    let kv_region = sp_start - kv_start;   // KV region: [kv_start, sp_start)
    let sp_region = fwd_total - sp_start;  // SP + weight + save region: [sp_start, fwd_total)
    if total > SMEM_DYNAMIC_BUDGET_BYTES {
        return Err(ConfigError(match direction {
            Direction::Forward => format!(
                "SMEM total {} bytes ({:.1} KB) exceeds 99 KB dynamic SMEM budget \
                 (Q={} KV={} SP+rest={} seg={}); reduce head_dim, block_q/block_kv, or d_model",
                total, total as f32 / 1024.0, q_region, kv_region, sp_region, seg_overhead
            ),
            Direction::Backward => format!(
                "CSHA fused Backward rejected: {} bytes > {} byte cap at \
                 (block_q={}, head_dim={}); forward={} backward_extra={} seg={} \
                 (P+dQ+dK+dV). Reduce head_dim, block_q/block_kv, or d_model.",
                total, SMEM_DYNAMIC_BUDGET_BYTES,
                config.block_q, config.head_dim,
                fwd_total, extra, seg_overhead
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
    if config.csha.as_ref().is_some_and(|c| c.fused_projections) {
        let iters = (config.block_q as u32).div_ceil(4);
        iters * base
    } else {
        base
    }
}

/// PCA Tier B range table — direction-aware tail offset into extern shmem.
///
/// Returns `align_up(direction_total + seg_overhead, 2)` so the returned
/// offset is 2-byte aligned for u16 slot loads. Alignment is owned here
/// per IR-004; `pca_tilerange::range_table_addrs` assumes the base is
/// ready-to-use.
///
/// `direction` selects which SMEM total the range table sits at the tail
/// of:
///   - `Direction::Forward`  → `total_bytes(config) + seg_overhead`
///     (used by forward-only kernels; forward total is ~17 KB at the
///     spec-pinned gate fixture, leaving ample headroom under the 99 KB
///     dynamic SMEM cap).
///   - `Direction::Backward` → `backward_total_bytes(config) +
///     seg_overhead` (used when backward Tier B ships in B.2; the table
///     rides past the backward extras so forward+backward kernels can
///     share an arena layout when emitted under a single SMEM budget).
///
/// Forward-only kernels at the gate fixture's pinned dims (block 64×64,
/// head_dim=64) would otherwise inherit the backward-sized offset
/// (~140 KB), pushing total SMEM past the 99 KB cap on Blackwell. See
/// design spec §11.4 for the discovery and rationale.
pub fn tier_b_range_table_offset(
    config: &crate::flash_attention::FlashAttentionConfig,
    direction: Direction,
) -> u32 {
    let seg_overhead = if config.segment_masked {
        crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32
    } else {
        0
    };
    let base = match direction {
        Direction::Forward => total_bytes(config),
        Direction::Backward =>
            crate::flash_attention_v2::phases::backward::prelude::backward_total_bytes(config),
    } + seg_overhead;
    align_up_u32(base, 2)
}

#[inline]
fn align_up_u32(x: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
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
    if config.csha.as_ref().is_some_and(|c| c.fused_projections) {
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
    if config.csha.as_ref().is_some_and(|c| c.fused_output_proj) {
        let d_model = config.csha.as_ref().map_or(0, |c| c.d_model);
        2 * (config.head_dim as u32) * d_model
    } else {
        0
    }
}

/// Residual input tile bytes when `csha.fused_output_proj` is set:
/// `block_q * d_model * 2` (f16). Returns 0 when not enabled or `d_model == 0`.
pub fn x_residual_bytes(config: &FlashAttentionConfig) -> u32 {
    if config.csha.as_ref().is_some_and(|c| c.fused_output_proj) {
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
    if config.csha.as_ref().is_some_and(|c| c.fused_projections) {
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

// ---------------------------------------------------------------------------
// Tier B.1 SMEM offsets
// ---------------------------------------------------------------------------
//
// Layout (per spec section 3.3) for the Level-2 pipelined attention forward
// kernel.  ALL offsets live within the single extern `.shared` region — per
// the Blackwell sm_120 illegal-address invariant we MUST NOT mix static
// `.shared` declarations with extern.  Tier B.1's emission uses one extern
// allocation per CTA; these accessors give the byte offsets of each
// sub-region within that single allocation.
//
// Region order:
//   q_offset       = 0                                size = bq * hd * 2 (f16)
//   k_offset_ping  = q_offset + (bq * hd * 2)         size = bkv * hd * 2
//   k_offset_pong  = k_offset_ping + (bkv * hd * 2)   size = bkv * hd * 2
//   v_offset_ping  = k_offset_pong + (bkv * hd * 2)   size = bkv * hd * 2
//   v_offset_pong  = v_offset_ping + (bkv * hd * 2)   size = bkv * hd * 2
//   w_chunk_offset = v_offset_pong + (bkv * hd * 2)   size = chunk-dependent
//                                                     (W chunks + x chunks,
//                                                      sized at emission time)

/// Q tile offset (always 0 — Q lives at the start of the extern shared
/// allocation). Parameter intentionally unused — offset is a constant.
pub fn tier_b1_q_offset(_config: &FlashAttentionConfig) -> u32 { 0 }

/// K ping slot offset. f16 row-major `[block_kv, head_dim]`. Lives
/// immediately after the Q tile.
pub fn tier_b1_k_offset_ping(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let hd = config.head_dim as u32;
    bq * hd * 2
}

/// K pong slot offset. f16 row-major `[block_kv, head_dim]`. Lives
/// immediately after K ping.
pub fn tier_b1_k_offset_pong(config: &FlashAttentionConfig) -> u32 {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    tier_b1_k_offset_ping(config) + bkv * hd * 2
}

/// V ping slot offset. f16 row-major `[block_kv, head_dim]`. Lives
/// immediately after K pong.
pub fn tier_b1_v_offset_ping(config: &FlashAttentionConfig) -> u32 {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    tier_b1_k_offset_pong(config) + bkv * hd * 2
}

/// V pong slot offset. f16 row-major `[block_kv, head_dim]`. Lives
/// immediately after V ping.
pub fn tier_b1_v_offset_pong(config: &FlashAttentionConfig) -> u32 {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    tier_b1_v_offset_ping(config) + bkv * hd * 2
}

/// P scratch region offset. f16 row-major `[block_q, block_kv]`. Lives
/// immediately after V pong. Used for the SMEM round-trip that bridges
/// softmax D-fragment-shaped output into the PV MMA's A-fragment input
/// (B1.6 deferral resolution — required for numerical correctness since
/// the PV A-fragment k=16 span crosses two consecutive QK^T n-tiles
/// which can't be sourced from a single D-fragment in-register).
pub fn tier_b1_p_offset(config: &FlashAttentionConfig) -> u32 {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    tier_b1_v_offset_pong(config) + bkv * hd * 2
}

/// Size of the P scratch region in bytes (block_q × block_kv f16).
pub fn tier_b1_p_scratch_bytes(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    bq * bkv * 2
}

/// W chunk staging region offset. Lives immediately after P scratch. Size
/// is chunk-dependent and sized at emission time (B1.3+) from the
/// chunk selected by `tier_b1::chunk_config::select`.
pub fn tier_b1_w_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b1_p_offset(config) + tier_b1_p_scratch_bytes(config)
}

/// Total SMEM bytes for a Tier B.1 kernel at the given chunk.
///
/// Layout:
///   `q_tile`        — block_q × head_dim × 2 (f16)
///   `4 × kv_slab`   — K_ping + K_pong + V_ping + V_pong, each block_kv × head_dim × 2
///   `p_scratch`     — block_q × block_kv × 2 (f16 softmax-output bridge)
///   `chunk_staging` — Wk + Wv slots (chunk × head_dim × 2 each)
///                   + x_q + x_kv slots ((block_q + block_kv) × chunk × 2)
///
/// Mirrors `validate_tier_b1_config`'s formula exactly so the validator
/// and the prelude's `.shared` allocation always agree byte-for-byte.
/// Without this helper the prelude declared the Tier-A baseline (Q tile +
/// KV tile + SP rows, ~5 KB), but Tier B.1 writes to offsets past 36 KB
/// (the `x_kv` chunk slot lives at `tier_b1_w_chunk_offset + 3 * chunk*head_dim*2`).
/// Static SMEM under-declaration silently stomps neighboring GPU state;
/// ptxas can't detect it because SMEM offsets are computed dynamically.
pub fn tier_b1_total_smem_bytes(config: &FlashAttentionConfig, chunk: u32) -> u32 {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    let q_tile = bq * hd * 2;
    let kv_ping_pong = 2 * (bkv * hd * 2) + 2 * (bkv * hd * 2);
    let p_scratch = tier_b1_p_scratch_bytes(config);
    let chunk_staging = chunk * hd * 2 * 2          // Wk + Wv chunk slots
                      + (bq + bkv) * chunk * 2;     // x_q + x_kv chunk slots
    q_tile + kv_ping_pong + p_scratch + chunk_staging
}

/// Validate config against Tier B.1's SMEM budget (per spec section 3.4).
/// Called by `tier_b1::chunk_config::select`. Returns the selected chunk
/// on success; returns `ConfigError` if the budget is exceeded for this
/// chunk.
pub fn validate_tier_b1_config(
    config: &FlashAttentionConfig,
    chunk: u32,
) -> Result<u32, ConfigError> {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;

    let q_tile = bq * hd * 2;
    // 4 KV slabs total: K_ping + K_pong + V_ping + V_pong, each
    // `bkv * hd * 2` bytes (f16).
    let kv_ping_pong = 2 * (bkv * hd * 2) + 2 * (bkv * hd * 2);
    // P scratch: SMEM round-trip for softmax(QK^T) D-fragment → PV
    // A-fragment bridge (B1.6 deferral resolution). Size = block_q
    // × block_kv f16.
    let p_scratch = tier_b1_p_scratch_bytes(config);
    let fixed_bytes = q_tile + kv_ping_pong + p_scratch;

    let chunk_staging = chunk * hd * 2 * 2          // Wk + Wv chunk slots
                      + (bq + bkv) * chunk * 2;     // x_q + x_kv chunk slots
    let total = fixed_bytes + chunk_staging;

    if total > SMEM_DYNAMIC_BUDGET_BYTES {
        return Err(ConfigError(format!(
            "Tier B.1 SMEM total {} exceeds dynamic budget {} (fixed={}, chunk_staging={})",
            total, SMEM_DYNAMIC_BUDGET_BYTES, fixed_bytes, chunk_staging
        )));
    }
    Ok(chunk)
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
            segment_masked: false,
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
            segment_masked: false,
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
        // P = dS = dQ = dK = dV = 32*32*4 = 4096 each (20480); V_in = 32*32*2 = 2048.
        // x_norm = dx_norm = 32*32*4 = 4096 each (8192); rms_strip = 32*4 = 128.
        // Total = 22528 + 8192 + 128 = 30848.
        assert_eq!(extra, 30848,
            "backward_extra_bytes = P+dS+dQ+dK+dV+V_in+x_norm+dx_norm+rms_strip");
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

    /// `segment_masked` adds DEFAULT_SMEM_SEGMENT_BUDGET to the validator's
    /// budget check on both directions. A config that fits the 99 KB cap on
    /// paper but trips the cap once `seg_overhead` is folded in must be
    /// rejected here rather than at launch time (where the failure mode is
    /// CUDA_ERROR_OUT_OF_MEMORY or silent SMEM corruption).
    #[test]
    fn segment_masked_inflates_budget_check_by_default_smem_segment_budget() {
        // Backward fixture sized so backward_total_bytes lands in the
        // (99 KB - 32 KB, 99 KB] window — passes without seg, fails with.
        let mut cfg = base_cfg_fused_backward(64, 64, 64, 8, 64);
        cfg.segment_masked = false;
        let unmasked_total =
            total_bytes(&cfg) + backward_extra_bytes(&cfg);
        // Only run the assertion when the fixture actually lands in the
        // sensitive window — otherwise the property we're testing isn't
        // exercised (the test still passes either way, but the assertion
        // would be vacuous outside the window).
        if unmasked_total <= SMEM_DYNAMIC_BUDGET_BYTES
            && unmasked_total + (crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32)
                > SMEM_DYNAMIC_BUDGET_BYTES
        {
            assert!(
                validate_scalar_v2_config(&cfg, Direction::Backward).is_ok(),
                "unmasked config must pass at total={unmasked_total}"
            );
            cfg.segment_masked = true;
            let err = validate_scalar_v2_config(&cfg, Direction::Backward)
                .expect_err("segment_masked must inflate budget past 99 KB cap");
            let msg = format!("{err}");
            assert!(
                msg.contains("seg=")
                    && msg.contains(
                        &(crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET).to_string(),
                    ),
                "rejection diagnostic must surface seg_overhead value: {msg}"
            );
        }
    }

    /// Smallest segment_masked fixture must stay below the 99 KB cap even
    /// after `seg_overhead` is folded in — otherwise the planner can never
    /// emit a packed backward at all.
    #[test]
    fn smallest_segment_masked_backward_still_fits_budget() {
        let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        cfg.segment_masked = true;
        assert!(
            validate_scalar_v2_config(&cfg, Direction::Backward).is_ok(),
            "smallest segment_masked backward (32x32x32 d_model=32) must fit \
             the 99 KB cap with seg_overhead included",
        );
    }

    // ---- Tier B.1 offset + validation tests ------------------------------

    /// Build a canonical Tier B.1 forward config with caller-supplied tile
    /// dimensions.  `csha.level = 2` (Pipeline) so eligibility checks pass.
    fn tier_b1_cfg(block_q: i64, block_kv: i64, head_dim: i64) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q, block_kv, head_dim,
            causal: true, paged: false, rope_q: true,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn tier_b1_offsets_are_monotone() {
        // Canonical Tier B.1 config (bq=64, bkv=64, hd=128) per spec §3.3.
        let config = tier_b1_cfg(64, 64, 128);
        assert!(tier_b1_q_offset(&config) < tier_b1_k_offset_ping(&config),
            "q_offset must precede k_offset_ping");
        assert!(tier_b1_k_offset_ping(&config) < tier_b1_k_offset_pong(&config),
            "k_offset_ping must precede k_offset_pong");
        assert!(tier_b1_k_offset_pong(&config) < tier_b1_v_offset_ping(&config),
            "k_offset_pong must precede v_offset_ping");
        assert!(tier_b1_v_offset_ping(&config) < tier_b1_v_offset_pong(&config),
            "v_offset_ping must precede v_offset_pong");
        assert!(tier_b1_v_offset_pong(&config) < tier_b1_p_offset(&config),
            "v_offset_pong must precede p_offset");
        assert!(tier_b1_p_offset(&config) < tier_b1_w_chunk_offset(&config),
            "p_offset must precede w_chunk_offset");
        // P scratch is block_q × block_kv f16.
        assert_eq!(
            tier_b1_w_chunk_offset(&config) - tier_b1_p_offset(&config),
            tier_b1_p_scratch_bytes(&config),
            "p_scratch region between p_offset and w_chunk_offset must equal block_q × block_kv × 2"
        );
    }

    #[test]
    fn tier_b1_validate_canonical_64_64_128_fails_at_chunk_64() {
        // Per spec section 3.5: canonical config (64, 64, 128) at chunk=64
        // exceeds 99 KB even with S/P-scratch reduction.  The V3 supported-
        // matrix CSV at
        // `docs/superpowers/specs/2026-05-11-tier-b1-v3-supported-matrix.csv`
        // shows this tuple as rejected.
        let config = tier_b1_cfg(64, 64, 128);
        let err = validate_tier_b1_config(&config, 64);
        assert!(err.is_err(),
            "canonical (64,64,128) at chunk=64 must exceed 99 KB; got {:?}", err);
    }

    #[test]
    fn tier_b1_validate_small_config_passes() {
        // V3 CSV admits bq=bkv=32, hd=64 at chunk=64.
        let config = tier_b1_cfg(32, 32, 64);
        let result = validate_tier_b1_config(&config, 64);
        assert!(result.is_ok(),
            "small (32,32,64) at chunk=64 must fit 99 KB; got {:?}", result);
        assert_eq!(result.unwrap(), 64, "validator must return selected chunk on success");
    }
}
