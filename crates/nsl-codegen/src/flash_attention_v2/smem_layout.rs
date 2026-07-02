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

/// SMEM rebasing base for the standalone Tier B.2 `proj_backward` kernel.
///
/// The proj-backward kernel (Phase 3 T4) reuses the scalar fused-backward
/// prelude + emitters, which place the dQ/dK/dV/x_norm/dx_norm/rms tiles AFTER
/// the full forward layout (`total_bytes` + P + dS). As a *standalone* kernel
/// proj_backward never touches the forward Q/KV/SP/weight tiles nor the P/dS
/// tiles, so allocating SMEM up through the absolute `backward_rms_strip_offset`
/// (~137 KB at hd=64, ~113 KB at hd=128) blows past the 99 KB dynamic-SMEM
/// device cap and the launch fails with `CUDA_ERROR_INVALID_VALUE`.
///
/// `synthesize_proj_backward` therefore shifts `%shmem_base` DOWN by this base
/// (`backward_dq_offset`, the lowest offset the proj kernel references) so the
/// SAME emitter offsets land in a compacted region starting at allocation byte
/// 0. The launch then only needs `tier_b2_proj_backward_smem_bytes` (~88 KB).
/// All proj references go through `%shmem_base + backward_{dq,dv,dk,x_norm,
/// dx_norm,rms}_offset`, and all rebase consistently because the shift is a
/// single subtract applied once after the prelude.
pub fn tier_b2_proj_backward_smem_base(config: &FlashAttentionConfig) -> u32 {
    backward_dq_offset(config)
}

/// Compacted dynamic-SMEM byte count for the standalone proj-backward kernel
/// after the `%shmem_base` rebase (see `tier_b2_proj_backward_smem_base`).
/// Equals the span from the lowest referenced tile (dQ) through the end of the
/// highest (the rms strip).
pub fn tier_b2_proj_backward_smem_bytes(config: &FlashAttentionConfig) -> u32 {
    let rms_end = backward_rms_strip_offset(config) + (config.block_q as u32) * 4;
    rms_end - tier_b2_proj_backward_smem_base(config)
}

/// Cycle-11 §4 update: extra SMEM scratch the `policy="full"` recompute
/// path needs on top of the existing forward + backward tiles to hold the
/// re-derived prologue values during backward.
///
/// **Cycle-11 3-tile strategy (NOT 6).** Cycle 10 reserved 6 tiles
/// (x_norm, Q_proj, K_proj, V_proj, Q_rope, K_rope) conservatively. The
/// cycle-11 functional-substitution wiring observes that K_proj and V_proj
/// can be written back into the existing `%k_smem_base` / `%v_smem_base`
/// slots (overwriting the now-unused kv_load tiles — those are dispatched
/// off when `checkpoint.is_some()`). Q_proj is the only producer whose
/// downstream consumer survives, but in v1.1 the only required NEW SMEM
/// surface is the **xnorm scratch** read by both projection-recompute
/// matmuls. K_rope rotation is applied in place on `%k_smem_base` via
/// `emit_rope_k_epilogue` (same as forward), needing no extra scratch.
///
/// Byte math: **1 tile × f16** =
///
///   recompute_extra_bytes = 1 * block_q * head_dim * 2
///                         = 2 * bq * hd  (bytes)
///
/// At hd=64, bq=64 this is 2*64*64 = 8192 = 8 KB.
/// At hd=256, bq=128 it is 2*256*128 = 65536 = 64 KB (the R5 large-config
/// refusal still trips because the forward+backward base layout at
/// hd=256/bq=128 exceeds 99 KB on its own; the recompute term is no longer
/// the dominant overflow source but the cap-check still rejects).
pub fn recompute_extra_bytes(config: &FlashAttentionConfig) -> usize {
    let bq = config.block_q as usize;
    let hd = config.head_dim as usize;
    // Cycle-11: single xnorm scratch tile (f16). K_proj/V_proj reuse
    // `%k_smem_base`/`%v_smem_base`; K_rope rotates in place.
    bq * hd * 2
}

/// Cycle-11 §4 helper: byte offset of the new xnorm scratch tile within
/// the dynamic-SMEM region, used by `emit_prologue_recompute_from_raw`
/// to write the recomputed `x_norm[block_q, head_dim]` (f16, row-major,
/// stride `head_dim*2`).
///
/// The xnorm scratch slot lives at the very top of the recompute extra
/// region (immediately past the forward + backward layouts). Total budget
/// allocated for this slot is `recompute_extra_bytes(config)` =
/// `block_q * head_dim * 2` bytes — exactly one tile.
pub fn recompute_xnorm_offset(config: &FlashAttentionConfig) -> u32 {
    // Anchor at the end of the backward extra region — the forward tiles
    // occupy `total_bytes(config)`, the backward extra (`dQ`/`dK`/`dV`/
    // `x_norm`/`dx_norm`/`rms_strip`/`P`) sits after, and the recompute
    // xnorm scratch comes last.
    total_bytes(config) + backward_extra_bytes(config)
}

/// Runtime validation called by `synthesize_flash_attention_ptx_v2`.
///
/// `direction` controls whether the backward-pass extra SMEM tiles are
/// added to the budget before the 99 KB cap check. Forward: unchanged
/// from Tier A. Backward (Tier C): adds `backward_extra_bytes` to the
/// forward total.
///
/// Cycle-10 §5.3 Task 7 (R5): when `config.checkpoint.is_some()` with
/// `CheckpointPolicy::Full`, the prologue-recompute scratch tiles
/// (`recompute_extra_bytes`) are added on top of the forward + backward
/// extra. If the total exceeds `SMEM_DYNAMIC_BUDGET_BYTES` the validator
/// returns a refusal whose message contains the substring "exceeds device"
/// (R5 testable contract).
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
    // Cycle-10 §5.3 Task 7 (R5): policy=Full needs prologue-recompute
    // scratch on top of the forward + backward layouts. Add only when
    // a CheckpointExtras carrier is present; absence preserves
    // byte-identity.
    let recompute_extra = if let Some(ref ckpt) = config.checkpoint {
        match ckpt.policy {
            crate::flash_attention::CheckpointPolicy::Full => {
                recompute_extra_bytes(config) as u32
            }
        }
    } else {
        0
    };
    // PCA Tier A: when `segment_masked`, the FA prelude reserves a
    // `DEFAULT_SMEM_SEGMENT_BUDGET`-byte `seg_smem` region (+ a 1028-byte
    // `smem_doc_starts` region when `rope_q`) — separately-declared static SMEM
    // on the forward side, `seg_smem` embedded in the extern shmem tail on the
    // backward side (see `phases/{forward,backward}/prelude.rs`). Route the total
    // through the single-source `pca_smem_layout` so this validator agrees with
    // `fwd_needs_dynamic_smem` / `backward_needs_dynamic_smem` on what a PCA
    // kernel needs (else a `segment_masked` config that fits the 99 KB cap on
    // paper fails at launch with insufficient SMEM). `seg_overhead` is kept for
    // the diagnostic message below.
    let seg_overhead = if config.segment_masked {
        crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32
    } else {
        0
    };
    // pca_smem_layout handles segment_masked + rope_q segment regions;
    // recompute_extra is layered on top for checkpoint policy=Full (cycle-10 R5).
    let total = pca_smem_layout(fwd_total + extra, config.segment_masked, config.rope_q).total
        + recompute_extra;
    let q_region  = kv_start;              // Q region: [0, kv_start)
    let kv_region = sp_start - kv_start;   // KV region: [kv_start, sp_start)
    let sp_region = fwd_total - sp_start;  // SP + weight + save region: [sp_start, fwd_total)
    if total > SMEM_DYNAMIC_BUDGET_BYTES {
        // Cycle-10 §5.3 Task 7 (R5): when the overflow is driven by the
        // policy="full" recompute term, emit a refusal whose message
        // contains the substring "exceeds device" so callers / tests can
        // discriminate this from forward / backward exhaustion.
        if recompute_extra > 0 {
            return Err(ConfigError(format!(
                "checkpoint policy=\"full\" SMEM {} KB exceeds device {} KB \
                 at hd={}, block_q={} (forward={} backward_extra={} recompute_extra={} seg={})",
                total / 1024,
                SMEM_DYNAMIC_BUDGET_BYTES / 1024,
                config.head_dim,
                config.block_q,
                fwd_total,
                extra,
                recompute_extra,
                seg_overhead,
            )));
        }
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

/// S/P scratch region starts immediately after the KV tile.
///
/// KV slab spans `effective_block_kv × head_dim × 2` bytes — at
/// `num_sink_tokens == 0` this is identical to
/// `block_kv × head_dim × 2` (byte-identity invariant); when sinks are
/// enabled (Sprint 1b cycle-7) the sink rows are pinned at the front of
/// the slab and the SP region shifts up by `sink_slab_bytes(config)` so
/// the rolling K/V rows don't stomp it.
pub fn sp_offset(config: &FlashAttentionConfig) -> u32 {
    let effective_bkv = crate::flash_attention_v2::sinks::effective_block_kv(config) as u32;
    kv_offset(config) + effective_bkv * (config.head_dim as u32) * 2
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
    // §4.3 sinks (Sprint 1a precursor): SP scratch holds one S/P slot per
    // (warp, kv-row). The kv-row count is `effective_block_kv` — what
    // s_compute writes to and softmax / pv_accum read. If sp_bytes used
    // raw `block_kv` while s_compute wrote to `effective_block_kv`-many
    // rows the S-store would overflow into adjacent SMEM regions. At
    // num_sink_tokens==0, this is identical to `config.block_kv` (no-op).
    let block_kv  = crate::flash_attention_v2::sinks::effective_block_kv(config) as u32;
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

/// Single source of truth for PCA static-SMEM **sizing**. `base_total` is the
/// core kernel SMEM (`total_bytes` forward / `backward_total_bytes` backward);
/// `.total` adds the `seg_smem` (32 KB) region and — when `rope_q` — the
/// `smem_doc_starts` (1028 B) region, 16-aligned. The dynamic-SMEM budget
/// checks (`fwd_needs_dynamic_smem` / `backward_needs_dynamic_smem`) and the
/// config validator all derive their total from this one function, so they
/// cannot diverge on how much SMEM a PCA kernel needs.
///
/// NOTE — what is actually emitted today: the PCA regions are SEPARATE static
/// `.shared` declarations, NOT embedded in `shmem[]` (forward: `seg_smem` +
/// `smem_doc_starts` both separate static; backward: `seg_smem` embedded in the
/// `shmem[]` tail, `smem_doc_starts` separate static). The `seg_off` / `doc_off`
/// fields are therefore **dormant** — only `.total` is consumed. They are a
/// ready primitive for embedding all regions in the `shmem[]` tail IF a future
/// driver ever reproduces the Blackwell static+extern mixed-layout crash, which
/// the 2026-05-12 SMEM probe + a 2026-05-27 reproduce found does NOT occur on
/// CUDA 13.2 / driver 591.86 / RTX 5070 Ti (so the embed was intentionally not
/// built). See `project_pca_smem_mixed_layout_crash_disproven` (memory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcaSmemLayout {
    /// Total SMEM bytes a PCA kernel needs (base + seg [+ doc when rope_q]), 16-aligned.
    /// This is the consumed field.
    pub total: u32,
    /// DORMANT (not consumed today): byte offset `seg_smem` WOULD have in a
    /// shmem[]-tail embed. Valid only when `segment_masked`.
    pub seg_off: u32,
    /// DORMANT (not consumed today): byte offset `smem_doc_starts` WOULD have in
    /// a shmem[]-tail embed. `Some` only when `segment_masked && rope_q`.
    pub doc_off: Option<u32>,
}

/// `seg_smem` size = the cooperative-load ceiling (`DEFAULT_SMEM_SEGMENT_BUDGET`, 32 KB).
const PCA_SEG_BYTES: u32 = crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32;
/// `smem_doc_starts` size = `(MAX_NUM_DOCS + 1) * 4` = `257 * 4` = 1028 bytes.
const PCA_DOC_BYTES: u32 = 1028;

/// Compute the PCA SMEM layout for a given base total. `base_total` is
/// `total_bytes(config)` (forward) or `backward_total_bytes(config)` (backward).
pub fn pca_smem_layout(base_total: u32, segment_masked: bool, rope_q: bool) -> PcaSmemLayout {
    if !segment_masked {
        return PcaSmemLayout { total: base_total, seg_off: 0, doc_off: None };
    }
    let seg_off = align_up_u32(base_total, 16);
    let after_seg = seg_off + PCA_SEG_BYTES; // PCA_SEG_BYTES (32768) is 16-aligned → after_seg 16-aligned
    if rope_q {
        let doc_off = after_seg;
        let total = align_up_u32(doc_off + PCA_DOC_BYTES, 16);
        PcaSmemLayout { total, seg_off, doc_off: Some(doc_off) }
    } else {
        let total = align_up_u32(after_seg, 16);
        PcaSmemLayout { total, seg_off, doc_off: None }
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

/// Softmax cross-warp scratch region offset. Lives immediately after the
/// P scratch. Holds per-(global_q_row × n_tile_kv) partial row-max and
/// row-sum f32 values used by the cross-warp softmax combine. Each warp
/// contributes one (max, sum) pair per row in its (m_tile × n_tile) slice;
/// all 32 lanes within a row's quad share the same partial (per-warp bfly
/// reduction). The cross-warp combine then reads ALL n_tile slots for the
/// row to compute the global max and sum.
///
/// Without this scratch, each warp normalized its P slice by its own
/// 8-column partial sum rather than the full bkv-row sum, producing
/// roughly `n_tiles_kv`-times-too-large attention contributions.
pub fn tier_b1_softmax_scratch_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b1_p_offset(config) + tier_b1_p_scratch_bytes(config)
}

/// Number of (m=16, n=8) tiles in the bkv direction. Powers of 2 only
/// (asserted at codegen via the existing `tiles_per_warp_qkt` / N3 checks).
pub fn tier_b1_n_tiles_kv(config: &FlashAttentionConfig) -> u32 {
    (config.block_kv as u32 / 8).max(1)
}

/// Size of the softmax cross-warp scratch in bytes. Per global Q-row, we
/// reserve one f32 partial-max + one f32 partial-sum per n_tile. That's
/// `block_q * n_tiles_kv * 2 * 4` bytes total. For canonical 32×32 this
/// is 32 × 4 × 2 × 4 = 1024 bytes; for 64×64×64 it's 64 × 8 × 2 × 4 = 4096.
pub fn tier_b1_softmax_scratch_bytes(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let n_tiles_kv = tier_b1_n_tiles_kv(config);
    bq * n_tiles_kv * 2 * 4
}

/// Reduced softmax-stats region offset. Lives immediately after the softmax
/// cross-warp scratch. Holds the FULLY-REDUCED per-query-row max/sum keyed by
/// ABSOLUTE global query row (`bq` f32 maxes followed by `bq` f32 sums).
///
/// ## Why this region exists (hd > block_kv fix)
///
/// The softmax row-max/row-sum stats are per-query-ROW values, but the
/// producer (`attention_mma.rs`) historically kept them in registers
/// `%s_max_<t>_<half>` / `%s_sum_<t>_<half>` whose index `<t>` is a per-warp
/// QK^T output-tile SLOT (`m_tile = global_t / (block_kv/8)`). The consumer
/// (`finalize.rs`) reads them under the PV slot decomposition
/// (`m_tile = global_t / (head_dim/8)`). The two slot->m_tile maps agree ONLY
/// when `head_dim == block_kv`; at `head_dim > block_kv` finalize references
/// slots the producer never declared (ptxas "Unknown symbol %s_sum_1_lo") AND
/// maps slot 0 to the wrong m-tile in half the warps.
///
/// The fix re-keys the stats by ABSOLUTE global query row via this dedicated
/// SMEM region. The producer persists each reduced row-max/row-sum here keyed
/// by `global_row`; finalize reads back by its PV m-tile's `global_row`,
/// making finalize independent of the QK^T-vs-PV slot basis.
///
/// Survival argument: this region is written at the END of the online-softmax
/// phase (after the cross-warp max+sum combine) and read in finalize. The only
/// `bar.sync`s between the two points (P-scatter sync, Phase-C swap sync, and
/// the projection-save fence) are all visibility fences — none reuse this byte
/// range. It is disjoint from the P scratch, the V slabs, the softmax
/// cross-warp scratch, and the projection-save SMEM (which all live at LOWER
/// offsets), and from the W/x chunk staging (which lives at a HIGHER offset and
/// is only live during the projection pre-pass, before softmax runs).
pub fn tier_b1_reduced_stats_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b1_softmax_scratch_offset(config) + tier_b1_softmax_scratch_bytes(config)
}

/// Size of the reduced softmax-stats region in bytes: `block_q` f32 maxes +
/// `block_q` f32 sums = `block_q * 2 * 4` bytes (256 B for bq=32).
pub fn tier_b1_reduced_stats_bytes(config: &FlashAttentionConfig) -> u32 {
    (config.block_q as u32) * 2 * 4
}

/// Byte offset of the reduced row-SUM sub-region within the reduced-stats
/// region (the row-MAX sub-region is at `tier_b1_reduced_stats_offset`).
/// Sum-base = max-base + `block_q * 4`.
pub fn tier_b1_reduced_stats_sum_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b1_reduced_stats_offset(config) + (config.block_q as u32) * 4
}

/// W chunk staging region offset. Lives immediately after the reduced
/// softmax-stats region. Size is chunk-dependent and sized at emission
/// time (B1.3+) from the chunk selected by `tier_b1::chunk_config::select`.
pub fn tier_b1_w_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b1_reduced_stats_offset(config) + tier_b1_reduced_stats_bytes(config)
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
    let softmax_scratch = tier_b1_softmax_scratch_bytes(config);
    let reduced_stats = tier_b1_reduced_stats_bytes(config);
    let chunk_staging = chunk * hd * 2 * 2          // Wk + Wv chunk slots
                      + (bq + bkv) * chunk * 2;     // x_q + x_kv chunk slots
    q_tile + kv_ping_pong + p_scratch + softmax_scratch + reduced_stats + chunk_staging
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
    // Softmax cross-warp scratch: per-(global_q_row, n_tile_kv) f32 (max, sum).
    let softmax_scratch = tier_b1_softmax_scratch_bytes(config);
    // Reduced softmax-stats: per-absolute-row f32 (max, sum) — re-keys the
    // softmax stats so finalize can read them by absolute query row,
    // independent of the QK^T-vs-PV slot basis (hd>bkv fix).
    let reduced_stats = tier_b1_reduced_stats_bytes(config);
    let fixed_bytes = q_tile + kv_ping_pong + p_scratch + softmax_scratch + reduced_stats;

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
            num_sink_tokens: 0,
            gpu_sm: 75,
            segment_masked: false,
            csha: None,
            checkpoint: None,
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
            num_sink_tokens: 0,
            gpu_sm: 75,
            segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: true,
                d_model,
                ..CshaExtras::default()
            }),
            checkpoint: None,
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
            num_sink_tokens: 0,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                ..CshaExtras::default()
            }),
            checkpoint: None,
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
        assert!(tier_b1_p_offset(&config) < tier_b1_softmax_scratch_offset(&config),
            "p_offset must precede softmax_scratch_offset");
        assert!(tier_b1_softmax_scratch_offset(&config) < tier_b1_reduced_stats_offset(&config),
            "softmax_scratch_offset must precede reduced_stats_offset");
        assert!(tier_b1_reduced_stats_offset(&config) < tier_b1_w_chunk_offset(&config),
            "reduced_stats_offset must precede w_chunk_offset");
        // P scratch is block_q × block_kv f16.
        assert_eq!(
            tier_b1_softmax_scratch_offset(&config) - tier_b1_p_offset(&config),
            tier_b1_p_scratch_bytes(&config),
            "p_scratch region between p_offset and softmax_scratch_offset must equal block_q × block_kv × 2"
        );
        // Softmax cross-warp scratch sits between p_scratch and reduced_stats.
        assert_eq!(
            tier_b1_reduced_stats_offset(&config) - tier_b1_softmax_scratch_offset(&config),
            tier_b1_softmax_scratch_bytes(&config),
            "softmax scratch region size mismatch"
        );
        // Reduced softmax-stats region sits between softmax_scratch and w_chunk.
        assert_eq!(
            tier_b1_w_chunk_offset(&config) - tier_b1_reduced_stats_offset(&config),
            tier_b1_reduced_stats_bytes(&config),
            "reduced softmax-stats region size mismatch"
        );
        // Sum sub-region base = max base + block_q * 4.
        assert_eq!(
            tier_b1_reduced_stats_sum_offset(&config) - tier_b1_reduced_stats_offset(&config),
            (config.block_q as u32) * 4,
            "reduced-stats sum sub-region must follow the bq f32 maxes"
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

    // ── Cycle-10 §5.3 Task 7 (R5): policy="full" SMEM budget tests ────────
    //
    // The validator must add `recompute_extra_bytes(config)` to the SMEM
    // total whenever `config.checkpoint = Some(Full)`. Small configs still
    // fit the 99 KB dynamic budget; large configs must refuse with the
    // testable substring "exceeds device".

    fn checkpoint_full_cfg(block_q: i64, head_dim: i64) -> FlashAttentionConfig {
        use crate::flash_attention::{CheckpointExtras, CheckpointPolicy};
        FlashAttentionConfig {
            block_q,
            block_kv: block_q,
            head_dim,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75,
            segment_masked: false,
            csha: None,
            checkpoint: Some(CheckpointExtras {
                policy: CheckpointPolicy::Full,
                paged_kv_collision: false,
                #[cfg(any(test, feature = "test-helpers"))]
                r0_bypass: false,
            }),
        }
    }

    #[test]
    fn cycle11_recompute_extra_bytes_byte_math() {
        // Cycle-11 §4: shrunk to 1 xnorm scratch tile (was 6 in cycle-10).
        // K_proj/V_proj reuse %k_smem_base / %v_smem_base; K_rope rotates
        // in place. Only the f16 xnorm scratch needs new SMEM.
        //
        // At hd=64, bq=64: 1 * 64 * 64 * 2 = 8192 bytes.
        let mut cfg = base_cfg();
        cfg.block_q = 64;
        cfg.block_kv = 64;
        cfg.head_dim = 64;
        assert_eq!(recompute_extra_bytes(&cfg), 8192);

        // At hd=256, bq=128: 1 * 128 * 256 * 2 = 65536 bytes.
        cfg.block_q = 128;
        cfg.block_kv = 128;
        cfg.head_dim = 256;
        assert_eq!(recompute_extra_bytes(&cfg), 65536);
    }

    #[test]
    fn cycle11_recompute_xnorm_offset_above_backward_region() {
        // The xnorm scratch slot must sit ABOVE the backward extra region
        // so it can't collide with dQ/dK/dV/x_norm/rms_strip/P. This pins
        // the offset to total_bytes + backward_extra_bytes — any future
        // refactor that shifts either tile must re-validate this offset.
        let mut cfg = base_cfg();
        cfg.block_q = 64;
        cfg.block_kv = 64;
        cfg.head_dim = 64;
        let off = recompute_xnorm_offset(&cfg);
        assert!(off >= total_bytes(&cfg) + backward_extra_bytes(&cfg));
    }

    #[test]
    fn cycle10_task7_small_full_policy_fits_budget() {
        // hd=64, block_q=64 + policy=Full: forward (~9 KB) + 0 backward extra
        // + ~48 KB recompute < 99 KB. Validator accepts on the Forward path.
        let cfg = checkpoint_full_cfg(64, 64);
        let result = validate_scalar_v2_config(&cfg, Direction::Forward);
        assert!(
            result.is_ok(),
            "hd=64 bq=64 + policy=Full must fit the 99 KB budget; got {:?}",
            result
        );
    }

    #[test]
    fn cycle10_task7_large_full_policy_exceeds_device() {
        // hd=256, block_q=128 + policy=Full: ~384 KB recompute term alone
        // dwarfs the 99 KB budget. Validator must refuse with the R5
        // substring "exceeds device" and identify the policy="full" branch.
        let cfg = checkpoint_full_cfg(128, 256);
        let err = validate_scalar_v2_config(&cfg, Direction::Forward)
            .expect_err("hd=256 bq=128 + policy=Full must overflow the budget");
        let msg = err.0;
        assert!(
            msg.contains("exceeds device"),
            "R5 error must contain substring 'exceeds device'; got: {}",
            msg
        );
        assert!(
            msg.contains("policy=\"full\""),
            "R5 error must identify policy=\"full\" branch; got: {}",
            msg
        );
    }

    #[test]
    fn cycle10_task7_no_policy_preserves_byte_identity() {
        // checkpoint=None must add zero bytes to the budget at any size.
        // Verifies the byte-identity invariant for the default path.
        let mut cfg = base_cfg();
        cfg.block_q = 64;
        cfg.block_kv = 64;
        cfg.head_dim = 64;
        cfg.checkpoint = None;

        // Same config WITHOUT the checkpoint must accept on the Forward path.
        let result = validate_scalar_v2_config(&cfg, Direction::Forward);
        assert!(
            result.is_ok(),
            "hd=64 bq=64 without checkpoint must fit; got {:?}",
            result
        );
    }

    #[test]
    fn pca_smem_layout_offsets_correct_and_nonoverlapping() {
        // Not masked → passthrough (no PCA regions).
        assert_eq!(
            pca_smem_layout(10_000, false, false),
            PcaSmemLayout { total: 10_000, seg_off: 0, doc_off: None }
        );

        // Masked, no rope → seg region in the tail after base; doc absent.
        let l = pca_smem_layout(10_000, true, false);
        assert_eq!(l.seg_off, 10_000);              // 10_000 already 16-aligned
        assert!(l.seg_off >= 10_000);               // non-overlap: seg starts at/after base end
        assert_eq!(l.seg_off % 16, 0);              // aligned
        assert_eq!(l.doc_off, None);
        assert!(l.total >= l.seg_off + 32_768);     // total covers seg (32 KB)
        assert_eq!(l.total % 16, 0);

        // Masked + rope, unaligned base → seg then doc, non-overlapping, aligned.
        let l = pca_smem_layout(10_001, true, true);
        assert_eq!(l.seg_off, 10_016);              // align_up(10_001, 16)
        assert_eq!(l.seg_off % 16, 0);
        let doc = l.doc_off.expect("doc_off present when rope_q");
        assert_eq!(doc, 10_016 + 32_768);           // seg_off + SEG_BYTES
        assert!(doc >= l.seg_off + 32_768);         // non-overlap: doc starts after seg
        assert_eq!(doc % 16, 0);
        assert!(l.total >= doc + 1028);             // total covers doc (1028 B)
        assert_eq!(l.total % 16, 0);
    }
}

// =====================================================================
// Tier B.2 backward sub-kernel SMEM accessors (Phase 2 — dQ-kernel only;
// dK/dV-kernel accessors stubbed for Phase 3).
//
// Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §3.1 + §5.2
//
// Per-sub-kernel layout (canonical hd=128, EFFECTIVE bq=32 after fallback, chunk=4):
//   q_offset:                0       (effective_bq * hd * 2 bytes, f16 Q tile)
//   k_offset:                        (effective_bkv * hd * 2 bytes, f16 K tile row-major)
//   v_offset:                        (effective_bkv * hd * 2 bytes, f16 V tile row-major)
//   dO_offset:                       (effective_bq * hd * 2 bytes, f16 dO tile, resident across kv_iter)
//   ds_offset:                       (effective_bq * effective_bkv * 4 bytes, f32 dS scratch)
//   wk_chunk_offset:                 (chunk * hd * 2 bytes, RMSNorm chunk staging)
//   wv_chunk_offset:                 (chunk * hd * 2 bytes)
//   x_q_chunk_offset:                (effective_bq * chunk * 2 bytes)
//   x_kv_chunk_offset:               (effective_bkv * chunk * 2 bytes)
//   k_colmajor_offset:               (effective_bkv * hd * 2 bytes, Path A re-stage)
//
// "chunk=4" is B.1's RMSNorm/projection chunked-sweep count (NOT accumulator
// staging). With dO-in-SMEM there is no separate x_norm accumulator staging.
//
// effective_bq differs from config.block_q at hd in {128, 256} per the Approach A"
// fallback schedule extended by Phase 2 spec §5.2 to also cover hd=128 SMEM pressure.

const TIER_B2_RMSNORM_CHUNK: u32 = 4;

/// Per-hd bq fallback schedule (extends Approach A" from register-pressure to
/// SMEM-pressure triggers per Phase 2 spec §5.2).
///
/// - hd in {32, 64}: no fallback (returns config.block_q).
/// - hd = 128: SMEM-pressure fallback to bq=32 (Path A col-major K re-stage
///   at bq=64 would push total SMEM to exactly the 99 KB cap with no headroom).
/// - hd = 256: register-pressure fallback to bq=32 (existing Approach A").
pub fn tier_b2_effective_bq(config: &FlashAttentionConfig) -> u32 {
    let raw_bq = config.block_q as u32;
    match config.head_dim {
        128 | 256 => raw_bq.min(32),
        _ => raw_bq,
    }
}

/// Symmetric helper to `tier_b2_effective_bq`, applied to bkv (Approach A"'s
/// bq=bkv invariant means the fallback applies to both).
pub fn tier_b2_effective_bkv(config: &FlashAttentionConfig) -> u32 {
    let raw_bkv = config.block_kv as u32;
    match config.head_dim {
        128 | 256 => raw_bkv.min(32),
        _ => raw_bkv,
    }
}

/// Q tile offset (always 0 — Q lives at the start of the dQ-kernel SMEM region).
pub fn tier_b2_dq_q_offset(_config: &FlashAttentionConfig) -> u32 {
    0
}

/// K tile offset. f16 row-major `[effective_bkv, head_dim]`.
/// Lives immediately after the Q tile.
pub fn tier_b2_dq_k_offset(config: &FlashAttentionConfig) -> u32 {
    let bq = tier_b2_effective_bq(config);
    let hd = config.head_dim as u32;
    bq * hd * 2
}

/// V tile offset. f16 row-major `[effective_bkv, head_dim]`.
/// Lives immediately after the K tile.
pub fn tier_b2_dq_v_offset(config: &FlashAttentionConfig) -> u32 {
    let bkv = tier_b2_effective_bkv(config);
    let hd = config.head_dim as u32;
    tier_b2_dq_k_offset(config) + bkv * hd * 2
}

/// dO tile offset. f16 row-major `[effective_bq, head_dim]`.
/// Resident across kv_iter — loaded once per dQ-kernel CTA, not per kv-block.
/// Lives immediately after the V tile.
#[allow(non_snake_case)]
pub fn tier_b2_dq_dO_offset(config: &FlashAttentionConfig) -> u32 {
    let bkv = tier_b2_effective_bkv(config);
    let hd = config.head_dim as u32;
    tier_b2_dq_v_offset(config) + bkv * hd * 2
}

/// dS scratch tile offset. f16 `[effective_bq, effective_bkv]`.
/// Written by the `dP = dO @ V^T` + softmax-backward combine.
/// Lives immediately after the dO tile.
pub fn tier_b2_dq_ds_offset(config: &FlashAttentionConfig) -> u32 {
    let bq = tier_b2_effective_bq(config);
    let hd = config.head_dim as u32;
    tier_b2_dq_dO_offset(config) + bq * hd * 2
}

/// Wk chunk staging region offset. f16 `[TIER_B2_RMSNORM_CHUNK, head_dim]`.
/// Used for the RMSNorm projection chunked sweep (mirrors B.1 forward's Wk slot).
/// Lives immediately after the dS scratch tile.
pub fn tier_b2_dq_wk_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    let bq = tier_b2_effective_bq(config);
    let bkv = tier_b2_effective_bkv(config);
    tier_b2_dq_ds_offset(config) + bq * bkv * 2
}

/// Wv chunk staging region offset. f16 `[TIER_B2_RMSNORM_CHUNK, head_dim]`.
/// Lives immediately after the Wk chunk slot.
pub fn tier_b2_dq_wv_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    let hd = config.head_dim as u32;
    tier_b2_dq_wk_chunk_offset(config) + TIER_B2_RMSNORM_CHUNK * hd * 2
}

/// x_q chunk staging region offset. f16 `[effective_bq, TIER_B2_RMSNORM_CHUNK]`.
/// Lives immediately after the Wv chunk slot.
pub fn tier_b2_dq_x_q_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    let hd = config.head_dim as u32;
    tier_b2_dq_wv_chunk_offset(config) + TIER_B2_RMSNORM_CHUNK * hd * 2
}

/// x_kv chunk staging region offset. f16 `[effective_bkv, TIER_B2_RMSNORM_CHUNK]`.
/// Lives immediately after the x_q chunk slot.
pub fn tier_b2_dq_x_kv_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    let bq = tier_b2_effective_bq(config);
    tier_b2_dq_x_q_chunk_offset(config) + bq * TIER_B2_RMSNORM_CHUNK * 2
}

/// Col-major K re-stage band offset (Path A per Phase 2 spec §5.2-5.3).
/// f16 `[head_dim, effective_bkv]` — stored col-major so MMA B-fragment
/// loads can use `emit_load_b_fragment_smem` without a transpose.
/// LAST SMEM band; placed after all chunk-staging bands.
pub fn tier_b2_dq_k_colmajor_offset(config: &FlashAttentionConfig) -> u32 {
    let bkv = tier_b2_effective_bkv(config);
    tier_b2_dq_x_kv_chunk_offset(config) + bkv * TIER_B2_RMSNORM_CHUNK * 2
}

/// Col-major K re-stage band size in bytes (f16 storage).
/// Size = effective_bkv * head_dim * 2 (same capacity as the row-major K
/// tile, different layout for MMA B-fragment compatibility).
pub fn tier_b2_dq_k_colmajor_bytes(config: &FlashAttentionConfig) -> u32 {
    let bkv = tier_b2_effective_bkv(config);
    let hd = config.head_dim as u32;
    bkv * hd * 2
}

/// Total dQ-kernel SMEM bytes (sums all bands including K-colmajor).
/// Must be <= SMEM_DYNAMIC_BUDGET_BYTES (99 KB) at all canonical configs.
///
/// Per-hd totals at canonical block_q=block_kv=64 requested (after fallback):
///   hd=32  bq=64  (no fallback):     38400 bytes = 37.5 KB (61.5 KB headroom)
///   hd=64  bq=64  (no fallback):     59392 bytes = 58.0 KB (41.0 KB headroom)
///   hd=128 bq=32  (SMEM fallback):   47616 bytes = 46.5 KB (52.5 KB headroom)
///   hd=256 bq=32  (reg-pressure fallback): 90624 bytes = 88.5 KB (10.5 KB headroom)
pub fn tier_b2_dq_total_smem_bytes(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dq_k_colmajor_offset(config) + tier_b2_dq_k_colmajor_bytes(config)
}

// Phase 3 dK/dV-kernel SMEM layout accessors.
//
// Layout (low → high address, all f16 = 2 bytes/element):
//   [Q row-major]    eb * hd * 2
//   [K row-major]    eb * hd * 2
//   [V row-major]    eb * hd * 2
//   [dO row-major]   eb * hd * 2
//   [Wk chunk]       CHUNK * hd * 2
//   [Wv chunk]       CHUNK * hd * 2
//   [x_q chunk]      eb * CHUNK * 2
//   [x_kv chunk]     eb * CHUNK * 2
//   [Q col-major]    eb * hd * 2
//   [dO col-major]   eb * hd * 2
//   [P col-major]    eb * eb * 2   (attention weight tile, A-operand scatter)
//   [dS col-major]   eb * eb * 2   (softmax grad tile)
//
// Total = 4*eb*hd*2 + 2*CHUNK*hd*2 + 2*eb*CHUNK*2 + 2*eb*hd*2 + 2*eb*eb*2
//       = 12*eb*hd + 4*eb^2 + 16*hd + 16*eb   (bytes, with CHUNK=4)

// Row-major tiles (mirror tier_b2_dq_* chain).
pub fn tier_b2_dkdv_q_offset(_config: &FlashAttentionConfig) -> u32 { 0 }
pub fn tier_b2_dkdv_k_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_effective_bq(config) * config.head_dim as u32 * 2
}
pub fn tier_b2_dkdv_v_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_k_offset(config) + tier_b2_effective_bkv(config) * config.head_dim as u32 * 2
}
#[allow(non_snake_case)]
pub fn tier_b2_dkdv_dO_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_v_offset(config) + tier_b2_effective_bkv(config) * config.head_dim as u32 * 2
}
// Chunk staging (mirror dQ's Wk/Wv/x_q/x_kv, base = after dO).
pub fn tier_b2_dkdv_wk_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_dO_offset(config) + tier_b2_effective_bq(config) * config.head_dim as u32 * 2
}
pub fn tier_b2_dkdv_wv_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_wk_chunk_offset(config) + TIER_B2_RMSNORM_CHUNK * config.head_dim as u32 * 2
}
pub fn tier_b2_dkdv_x_q_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_wv_chunk_offset(config) + TIER_B2_RMSNORM_CHUNK * config.head_dim as u32 * 2
}
pub fn tier_b2_dkdv_x_kv_chunk_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_x_q_chunk_offset(config) + tier_b2_effective_bq(config) * TIER_B2_RMSNORM_CHUNK * 2
}
// Col-major B-operand re-stage bands ([hd, eb] f16, col_stride eb*2).
pub fn tier_b2_dkdv_q_colmajor_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_x_kv_chunk_offset(config) + tier_b2_effective_bkv(config) * TIER_B2_RMSNORM_CHUNK * 2
}
#[allow(non_snake_case)]
pub fn tier_b2_dkdv_dO_colmajor_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_q_colmajor_offset(config) + tier_b2_effective_bq(config) * config.head_dim as u32 * 2
}
// Col-major A-operand scatter bands ([ek, eb] viewed row-major = P^T/dS^T, f16).
pub fn tier_b2_dkdv_p_colmajor_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_dO_colmajor_offset(config) + tier_b2_effective_bq(config) * config.head_dim as u32 * 2
}
pub fn tier_b2_dkdv_ds_colmajor_offset(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_p_colmajor_offset(config)
        + tier_b2_effective_bq(config) * tier_b2_effective_bkv(config) * 2
}
pub fn tier_b2_dkdv_total_smem_bytes(config: &FlashAttentionConfig) -> u32 {
    tier_b2_dkdv_ds_colmajor_offset(config)
        + tier_b2_effective_bq(config) * tier_b2_effective_bkv(config) * 2
}

#[cfg(test)]
mod tier_b2_dq_offset_tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_hd128_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 128,
            causal: true, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
            checkpoint: None,
        }
    }

    // === Per-hd bq fallback schedule (spec §5.2) ===

    #[test]
    fn tier_b2_effective_bq_no_fallback_at_hd_32() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 32;
        cfg.block_q = 64;
        assert_eq!(tier_b2_effective_bq(&cfg), 64);
    }

    #[test]
    fn tier_b2_effective_bq_no_fallback_at_hd_64() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 64;
        cfg.block_q = 64;
        assert_eq!(tier_b2_effective_bq(&cfg), 64);
    }

    #[test]
    fn tier_b2_effective_bq_smem_pressure_fallback_at_hd_128() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 128;
        cfg.block_q = 64;
        assert_eq!(tier_b2_effective_bq(&cfg), 32, "hd=128 SMEM-pressure fallback");
    }

    #[test]
    fn tier_b2_effective_bq_register_pressure_fallback_at_hd_256() {
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 256;
        cfg.block_q = 64;
        assert_eq!(tier_b2_effective_bq(&cfg), 32, "hd=256 register-pressure fallback");
    }

    #[test]
    fn tier_b2_effective_bkv_mirrors_effective_bq() {
        // Approach A"'s bq=bkv invariant: effective_bkv must mirror effective_bq.
        for &hd in &[32i64, 64, 128, 256] {
            let mut cfg = canonical_hd128_cfg();
            cfg.head_dim = hd;
            cfg.block_q = 64;
            cfg.block_kv = 64;
            assert_eq!(
                tier_b2_effective_bkv(&cfg),
                tier_b2_effective_bq(&cfg),
                "hd={} effective_bkv must mirror effective_bq", hd,
            );
        }
    }

    // === SMEM offset accessors (9 from original Task 3 + 1 new for K-colmajor) ===

    #[test]
    fn tier_b2_dq_offsets_match_spec_at_canonical_hd128() {
        let cfg = canonical_hd128_cfg();
        // At hd=128 with bq=64 requested, effective bq is 32 per spec §5.2 fallback.
        // SMEM layout uses effective_bq, NOT raw config.block_q.
        let eff_bq = tier_b2_effective_bq(&cfg);
        let eff_bkv = tier_b2_effective_bkv(&cfg);
        let hd = cfg.head_dim as u32;

        assert_eq!(tier_b2_dq_q_offset(&cfg), 0);
        assert_eq!(tier_b2_dq_k_offset(&cfg), eff_bq * hd * 2);
        assert_eq!(tier_b2_dq_v_offset(&cfg), eff_bq * hd * 2 + eff_bkv * hd * 2);
        // K-colmajor must be the last band
        assert!(tier_b2_dq_k_colmajor_offset(&cfg) > tier_b2_dq_x_kv_chunk_offset(&cfg),
            "K-colmajor band must be the LAST SMEM band");
        assert_eq!(tier_b2_dq_k_colmajor_bytes(&cfg), eff_bkv * hd * 2);
    }

    #[test]
    fn tier_b2_dq_k_colmajor_bytes_scales_with_hd_and_effective_bkv() {
        // hd=32 bq=64: K-colmajor = 64 * 32 * 2 = 4 KB
        let mut cfg = canonical_hd128_cfg();
        cfg.head_dim = 32;
        cfg.block_q = 64; cfg.block_kv = 64;
        assert_eq!(tier_b2_dq_k_colmajor_bytes(&cfg), 4 * 1024);

        // hd=64 bq=64: K-colmajor = 64 * 64 * 2 = 8 KB
        cfg.head_dim = 64;
        assert_eq!(tier_b2_dq_k_colmajor_bytes(&cfg), 8 * 1024);

        // hd=128 bq=64 (with fallback to effective bq=32): K-colmajor = 32 * 128 * 2 = 8 KB
        cfg.head_dim = 128;
        assert_eq!(tier_b2_dq_k_colmajor_bytes(&cfg), 8 * 1024);
    }

    #[test]
    fn tier_b2_dq_total_smem_under_dynamic_budget_at_all_hd() {
        for &hd in &[32i64, 64, 128, 256] {
            let mut cfg = canonical_hd128_cfg();
            cfg.head_dim = hd;
            cfg.block_q = 64;  // requested; effective may differ via fallback
            cfg.block_kv = 64;
            let total = tier_b2_dq_total_smem_bytes(&cfg);
            assert!(total <= SMEM_DYNAMIC_BUDGET_BYTES,
                "hd={} total SMEM {} > 99 KB cap", hd, total);
        }
    }

    #[test]
    fn tier_b2_dq_total_smem_includes_k_colmajor_band() {
        let cfg = canonical_hd128_cfg();
        // The K-colmajor band size must be reflected in the total.
        let total_with = tier_b2_dq_total_smem_bytes(&cfg);
        let last_band_start = tier_b2_dq_k_colmajor_offset(&cfg);
        let last_band_size = tier_b2_dq_k_colmajor_bytes(&cfg);
        assert_eq!(total_with, last_band_start + last_band_size,
            "total must equal K-colmajor offset + K-colmajor bytes");
    }

    #[test]
    fn tier_b2_dq_ds_band_is_f16_sized() {
        // Spec bug #2: dS staged f16 (bq*bkv*2), not f32. The dS band size is
        // the gap between ds_offset and the next band (wk_chunk).
        for &hd in &[32u32, 64, 128] {
            let cfg = FlashAttentionConfig { head_dim: hd as i64, ..canonical_hd128_cfg() };
            let bq = tier_b2_effective_bq(&cfg);
            let bkv = tier_b2_effective_bkv(&cfg);
            let ds_band = tier_b2_dq_wk_chunk_offset(&cfg) - tier_b2_dq_ds_offset(&cfg);
            assert_eq!(ds_band, bq * bkv * 2,
                "hd={hd}: dS band must be f16-sized (bq*bkv*2={}), got {ds_band}", bq * bkv * 2);
            assert!(tier_b2_dq_total_smem_bytes(&cfg) <= SMEM_DYNAMIC_BUDGET_BYTES,
                "hd={hd}: total SMEM exceeds 99 KB budget");
        }
    }

    // === dK/dV-kernel SMEM layout (Phase 3a Task 1) ===

    #[test]
    fn tier_b2_dkdv_total_matches_spec_schedule() {
        // (head_dim, requested bq=bkv, expected total bytes) per spec section 3.2.
        // Totals = 12*b*hd + 4*b^2 + 16*hd + 16*b at effective_bq.
        let cases = [
            (32i64, 32i64, 17408u32),  // smoke
            (32, 64, 42496),           // sweep
            (64, 64, 67584),           // no fallback
            (128, 64, 55808),          // effective bq=32 fallback
        ];
        for (hd, bq, expected) in cases {
            let mut cfg = canonical_hd128_cfg();
            cfg.head_dim = hd;
            cfg.block_q = bq;
            cfg.block_kv = bq;
            assert_eq!(tier_b2_dkdv_total_smem_bytes(&cfg), expected,
                "dK/dV total SMEM at hd={} bq={}", hd, bq);
        }
    }

    #[test]
    fn tier_b2_dkdv_total_under_budget_at_all_in_scope_hd() {
        for &hd in &[32i64, 64, 128] {
            let mut cfg = canonical_hd128_cfg();
            cfg.head_dim = hd;
            cfg.block_q = 64;
            cfg.block_kv = 64;
            assert!(tier_b2_dkdv_total_smem_bytes(&cfg) <= SMEM_DYNAMIC_BUDGET_BYTES,
                "hd={} dK/dV SMEM over 99 KB cap", hd);
        }
    }

    #[test]
    fn tier_b2_dkdv_bands_are_non_overlapping_and_ordered() {
        let cfg = canonical_hd128_cfg(); // hd=128 -> effective bq=bkv=32
        let eb = tier_b2_effective_bq(&cfg);
        let hd = cfg.head_dim as u32;
        assert_eq!(tier_b2_dkdv_q_offset(&cfg), 0);
        assert_eq!(tier_b2_dkdv_k_offset(&cfg), eb * hd * 2);
        assert_eq!(tier_b2_dkdv_v_offset(&cfg), 2 * eb * hd * 2);
        assert_eq!(tier_b2_dkdv_dO_offset(&cfg), 3 * eb * hd * 2);
        let q_col = tier_b2_dkdv_q_colmajor_offset(&cfg);
        let do_col = tier_b2_dkdv_dO_colmajor_offset(&cfg);
        let p_col = tier_b2_dkdv_p_colmajor_offset(&cfg);
        let ds_col = tier_b2_dkdv_ds_colmajor_offset(&cfg);
        assert!(q_col < do_col && do_col < p_col && p_col < ds_col,
            "col bands must be strictly ordered: q={q_col} dO={do_col} p={p_col} ds={ds_col}");
        assert!(ds_col + eb * eb * 2 <= tier_b2_dkdv_total_smem_bytes(&cfg),
            "last col band must fit within total");
    }
}
