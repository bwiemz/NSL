//! CSHA Level 2 — Projection-Attention Pipelining (paper §2.2) and
//! Level 3 — Full Block Fusion (paper §2.3) feasibility + cost model.
//!
//! The decision this module answers is:
//!
//! > Given GPU `G`, layer shape `L`, and fusion level `Level`, does the
//! > kernel fit in shared memory, and what HBM traffic / latency does it
//! > incur?
//!
//! Level 2 pipelines Q/K/V projections into FlashAttention via SMEM
//! ping-buffers; Level 3 extends that to fold the output projection,
//! residual add, and FFN into the same persistent kernel.  Both hinge on
//! the target GPU's per-SM SMEM capacity.
//!
//! This module is numeric-only — it operates on GPU specs and layer
//! shapes.  No Wengert list, no PTX.  The actual kernel emission happens
//! in `flash_attention` / `epilogue_fusion`, guided by the plan this
//! module produces.

use serde::Serialize;

use crate::csha_boundary::BoundaryScan;
use crate::gpu_specs::GpuSpec;
use crate::wggo_cost::LayerShape;

/// Smallest chunk value the Tier B.1 emitter's `chunk_config::select` may
/// return — load-bearing for the planner / emitter contract.
///
/// `chunk_config::select` performs a descending search over `{128, 64, 32,
/// 16}` and is REQUIRED to treat this constant as its floor: returning a
/// smaller value would invalidate the planner's admission decision in
/// `pipeline_smem_bytes`, which sizes chunk staging against this floor so
/// any admitted config has SOME valid chunk choice. Conversely, the
/// emitter MUST NOT silently downgrade below this floor or the planner
/// over-admits.
///
/// If you change this value you MUST also update `chunk_config::select`'s
/// descending search to match, and re-run the cost-model snapshot tests
/// (`csha_pipeline_cost_model.rs::snapshot_*`) which freeze admission
/// behavior across the V3 supported matrix.
pub const CHUNK_PLANNER_FLOOR: u64 = 16;

/// Selected CSHA fusion level for one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FusionLevel {
    /// Standard kernels (no CSHA).
    None,
    /// Level 1 — boundary fusion (epilogue/prologue absorption).
    Boundary,
    /// Level 2 — projection-attention pipelining.
    Pipeline,
    /// Level 3 — full block fusion.
    Block,
}

impl FusionLevel {
    pub fn as_u8(self) -> u8 {
        match self {
            FusionLevel::None => 0,
            FusionLevel::Boundary => 1,
            FusionLevel::Pipeline => 2,
            FusionLevel::Block => 3,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            FusionLevel::None => "none",
            FusionLevel::Boundary => "L1-boundary",
            FusionLevel::Pipeline => "L2-pipeline",
            FusionLevel::Block => "L3-block",
        }
    }
}

/// Tile sizes selected for a CSHA kernel at a given layer.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TileConfig {
    pub block_q: u32,
    pub block_kv: u32,
    pub head_dim: u32,
}

impl Default for TileConfig {
    fn default() -> Self {
        Self {
            block_q: 64,
            block_kv: 64,
            head_dim: 64,
        }
    }
}

/// Which backward kernel family this layer's planner picked.
///
/// Sprint 3 (paper §6.3 visibility): rendered alongside the per-layer
/// forward tier in [`crate::csha::CshaPlan::render_report`] so users can
/// audit `nsl check --csha-report` and see whether each attention layer
/// gets the hybrid Tier B.2 MMA backward or the scalar Tier C fallback.
///
/// Decoupled from the `flash_attention_v2::tier_b2::BackwardTier` runtime
/// enum so this struct doesn't depend on the `tier_b2` module's exact
/// layout: the planner only needs to surface "scalar vs tier_b2" with the
/// tier_b2 tile dims when applicable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BackwardTierReport {
    /// Scalar v2 backward (current default; lower throughput).
    Scalar,
    /// Tier B.2 MMA backward — planner-pinned `bq`, `bkv`, `chunk`.
    TierB2 { bq: u32, bkv: u32, chunk: u32 },
}

impl BackwardTierReport {
    /// Human-readable label used by [`crate::csha::CshaPlan::render_report`].
    pub fn as_label(&self) -> String {
        match self {
            BackwardTierReport::Scalar => "Tier C (scalar)".to_string(),
            BackwardTierReport::TierB2 { bq, bkv, chunk } => {
                format!("Tier B.2 (hybrid, bq={bq}, bkv={bkv}, chunk={chunk})")
            }
        }
    }
}

/// Per-layer feasibility + cost result.
#[derive(Debug, Clone, Serialize)]
pub struct LayerPlan {
    /// Human-readable identifier (`blocks.0`, `layers.7`, ...).
    pub layer: String,
    /// Winning fusion level after feasibility checks.
    pub level: FusionLevel,
    /// Tile sizes selected by [`roofline_tile_config`].
    pub tiles: TileConfig,
    /// Shared-memory bytes required by the selected kernel.
    pub smem_bytes: u64,
    /// SMEM budget on this GPU (bytes).
    pub smem_budget_bytes: u64,
    /// HBM traffic in bytes for this layer's attention sublayer.
    pub hbm_traffic_bytes: u64,
    /// HBM traffic under the "no CSHA" baseline, same layer — used for
    /// the speed-up ratio in reports.
    pub baseline_hbm_bytes: u64,
    /// Estimated latency in microseconds (memory-bound approximation).
    pub est_time_us: f64,
    /// Latency under the baseline (for the speed-up ratio).
    pub baseline_time_us: f64,
    /// Human-readable reason when `level` had to be downgraded from the
    /// requested level.
    pub downgrade_reason: Option<String>,
    /// Sprint 3 (paper §6.3): which backward kernel family the planner
    /// will dispatch for this layer at training time. `Scalar` when the
    /// Tier B.2 preconditions don't hold (level<2, sm<80, hd not in
    /// {64,128,256}, or SMEM over budget); `TierB2 { bq, bkv, chunk }`
    /// when the hybrid backward is eligible.
    pub backward_tier: BackwardTierReport,
}

impl LayerPlan {
    pub fn speedup(&self) -> f64 {
        if self.est_time_us > 0.0 {
            self.baseline_time_us / self.est_time_us
        } else {
            1.0
        }
    }

    pub fn hbm_reduction(&self) -> f64 {
        if self.hbm_traffic_bytes > 0 {
            self.baseline_hbm_bytes as f64 / self.hbm_traffic_bytes as f64
        } else {
            1.0
        }
    }
}

/// Per-SM shared-memory budget for a GPU.  Conservative defaults are
/// chosen so CSHA doesn't try to saturate all of SMEM with its own data.
pub fn smem_budget_bytes(gpu: &GpuSpec) -> u64 {
    // Reserve ~12KB for registers spills, runtime metadata, etc.
    let per_sm_bytes = gpu.l1_cache_kb.saturating_sub(12) as u64 * 1024;
    // Clamp to the documented H100 maximum even if the database reports
    // more — that's all CUDA lets a single CTA request.
    per_sm_bytes.min(228 * 1024)
}

/// SMEM bytes required by a Level-2 pipelined attention kernel at the
/// given shape / tile config.
///
/// Layout (Tier B.1 design spec §3.4 + V3 cost-model audit findings,
/// `docs/superpowers/specs/2026-05-11-tier-b1-v3-cost-model-audit.md`):
///   * Q tile          : `block_q  * head_dim * dtype`
///   * K/V ping-pong   : `2 * (block_kv * head_dim * dtype)` per stream
///                       (4 slabs total — Tier B.1 depth=2 over K and V)
///   * O accumulator   : 0  (register-resident; was a phantom over-count)
///   * Stats           : `block_q * 2 * 4`        (logsumexp + max, f32)
///   * Chunk staging   : two W chunk slots (Wk + Wv) + two x chunk slots
///                       (x_q + x_kv) — sized at the **minimum** chunk
///                       (16) so the planner admits any config that any
///                       valid chunk selection would accept.
///
/// **V3 audit corrections (2026-05-11) over the original formula:**
///
/// | Term            | Pre-correction         | Post-correction              | Why                                                                                  |
/// |-----------------|------------------------|------------------------------|--------------------------------------------------------------------------------------|
/// | `o_acc`         | `block_q*head_dim*4`   | **0**                        | Phantom term — Tier A already keeps O_acc in registers (V3 case alpha, retroactive). |
/// | K tile          | `block_kv*head_dim*dt` | **2 × that**                 | Tier B.1 ping-pongs K (depth=2).                                                     |
/// | V tile          | `block_kv*head_dim*dt` | **2 × that**                 | Tier B.1 ping-pongs V (depth=2).                                                     |
/// | `w_tile` (one)  | `head_dim*min(dm,256)*2` | **chunk staging w/ chunk=16** | Single weight tile replaced by Wk/Wv + x_q/x_kv staging; min chunk=16 = most permissive admission. |
///
/// The retained `o_acc: u64 = 0` location is intentional per spec §7.2's
/// "set to zero, not delete" discipline so a future Tier B.2 reintroducing
/// SMEM O_acc has the location preserved.
///
/// Canonical example (small_shape, bq=bkv=64, hd=64, dtype=2, dm=512):
///   q_tile        = 64*64*2  = 8192
///   kv_tiles      = 2*(64*64*2 + 64*64*2) = 32768
///   o_acc         = 0  (was 16384 phantom)
///   stats         = 64*2*4   = 512
///   chunk_staging = 2*(16*64*2) + (64+64)*16*2 = 4096 + 4096 = 8192
///   Total         = 8192 + 32768 + 0 + 512 + 8192 = **49664**
///
/// (Pre-correction baseline was 74240; delta = -24576.)
pub fn pipeline_smem_bytes(shape: LayerShape, tiles: TileConfig) -> u64 {
    let dtype = shape.dtype_bytes;
    let block_q = tiles.block_q as u64;
    let block_kv = tiles.block_kv as u64;
    let head_dim = tiles.head_dim as u64;
    let _ = shape.d_model; // dm is unused here; chunk selector lives in tier_b1::chunk_config.

    let q_tile = block_q * head_dim * dtype;
    let k_tile = block_kv * head_dim * dtype;
    let v_tile = block_kv * head_dim * dtype;
    // Tier B.1 ping-pong (depth=2) over both K and V — 4 slabs total.
    let kv_tiles = 2 * (k_tile + v_tile);
    // V3 case alpha: O_acc is register-resident (see `pv_accum::*`,
    // `finalize::*`, `smem_layout::total_bytes` — no O region). Zeroed,
    // not deleted, so B.2 has the location preserved.
    let o_acc: u64 = 0;
    let stats = block_q * 2 * 4;
    // Chunk staging budget per spec §3.4: at any instant the kernel holds
    // two W chunk slots (Wk_chunk + Wv_chunk) plus two x chunk slots
    // (x_q_chunk + x_kv_chunk). The planner uses the MINIMUM chunk
    // (`CHUNK_PLANNER_FLOOR`) so it admits configs that any valid chunk
    // selection would accept; the actual chunk is selected at emission
    // time by `tier_b1::chunk_config::select`, which descends
    // {128, 64, 32, 16} and must treat `CHUNK_PLANNER_FLOOR` as its floor.
    let chunk_staging =
        2 * (CHUNK_PLANNER_FLOOR * head_dim * dtype)       // Wk + Wv chunk slots
        + (block_q + block_kv) * CHUNK_PLANNER_FLOOR * dtype; // x_q + x_kv chunk slots
    q_tile + kv_tiles + o_acc + stats + chunk_staging
}

/// SMEM for Level-3 block fusion — pipeline cost plus the FFN staging
/// buffers and output-projection weight tile.
pub fn block_smem_bytes(shape: LayerShape, tiles: TileConfig) -> u64 {
    let base = pipeline_smem_bytes(shape, tiles);
    let dtype = shape.dtype_bytes;
    // Output projection + FFN gate/up projection weight tiles (one at a
    // time, streamed).  FFN expansion factor = 4 (SwiGLU has 2.67, but
    // we're conservative).
    let wo_tile = tiles.head_dim as u64 * shape.d_model.min(256) * dtype;
    let ffn_tile = shape.d_model.min(256) * shape.d_model.min(256) * dtype;
    base + wo_tile + ffn_tile
}

/// Pick tile sizes from roofline balance (paper §3.2).
///
/// * Memory-bound layers (low arithmetic intensity) → larger tiles to
///   amortise HBM loads.
/// * Compute-bound → smaller tiles to fit in registers.
pub fn roofline_tile_config(shape: LayerShape, gpu: &GpuSpec) -> TileConfig {
    // Very rough roofline estimate: projection arithmetic intensity is
    // `D / dtype_bytes` (one multiply-add per matrix element, dtype_bytes
    // per element loaded).
    let ai = shape.d_model as f64 / shape.dtype_bytes as f64;
    let crossover = gpu.crossover(shape.dtype_bytes as usize);

    let (bq, bkv) = if ai < crossover * 0.5 {
        (128, 128) // memory-bound — large tiles
    } else if ai < crossover * 2.0 {
        (64, 64) // balanced
    } else {
        (32, 64) // compute-bound — smaller Q tiles, smaller reg pressure
    };

    TileConfig {
        block_q: bq,
        block_kv: bkv,
        head_dim: shape.head_dim as u32,
    }
}

// ---------------------------------------------------------------------------
// Cost modelling
// ---------------------------------------------------------------------------

/// Baseline (unfused FA2) HBM traffic for the attention sublayer.
///
/// Count (paper §1 table, scaled to the layer shape):
///   RMSNorm    : 2·B·S·D
///   Q proj     : 2·B·S·D
///   K proj     : 2·B·S·D_kv
///   V proj     : 2·B·S·D_kv
///   RoPE Q     : 2·B·S·D
///   RoPE K     : 2·B·S·D_kv
///   GQA reshape: 2·B·S·D
///   FlashAttn  : 1·B·S·D + 1·B·S·D_kv·2  (reads QKV, writes O)
///   O proj     : 2·B·S·D
///   Residual   : 2·B·S·D
pub fn baseline_hbm_bytes(shape: LayerShape) -> u64 {
    let d = shape.d_model;
    let d_kv = shape.n_kv_heads * shape.head_dim;
    let bs = shape.batch * shape.seq;
    let dtype = shape.dtype_bytes;
    let bsd = bs * d * dtype;
    let bsdkv = bs * d_kv * dtype;
    2 * bsd          // norm
        + 2 * bsd    // q proj
        + 2 * bsdkv  // k proj
        + 2 * bsdkv  // v proj
        + 2 * bsd    // rope q
        + 2 * bsdkv  // rope k
        + 2 * bsd    // gqa
        + bsd + 2 * bsdkv // fa read
        + bsd        // fa write
        + 2 * bsd    // o proj
        + 2 * bsd    // residual
}

/// HBM traffic under a given fusion level.
pub fn fused_hbm_bytes(shape: LayerShape, level: FusionLevel) -> u64 {
    let d = shape.d_model;
    let bs = shape.batch * shape.seq;
    let dtype = shape.dtype_bytes;
    let bsd = bs * d * dtype;
    match level {
        FusionLevel::None => baseline_hbm_bytes(shape),
        FusionLevel::Boundary => {
            // Eliminates RMSNorm output + RoPE scratch across Q/K.  The
            // paper reports ~16MB savings out of ~40MB (roughly 40%).
            (baseline_hbm_bytes(shape) as f64 * 0.58) as u64
        }
        FusionLevel::Pipeline => {
            // V3 audit (spec §7.1) correction: L2 reads x + writes attn_out,
            // but the downstream Wo kernel must re-read attn_out from HBM
            // (L2 does NOT fuse the output projection — that's L3 only).
            // The original `2 * bsd` over-counted L2's savings by omitting
            // the Wo re-read.
            let x_in = bsd;
            let attn_out_write = bsd;
            let downstream_wo_in = bsd; // downstream Wo kernel re-reads attn_out
            x_in + attn_out_write + downstream_wo_in // = 3 * bsd
        }
        FusionLevel::Block => {
            // L3 fuses Wo into the persistent kernel, eliminating the
            // attn_out → Wo HBM round-trip. So 2*bsd is correct here: read
            // x + write final output. (No Wo re-read — Wo is in-kernel.)
            2 * bsd
        }
    }
}

/// Bandwidth-bound latency estimate in microseconds.  Used as a
/// conservative lower bound — the real kernel may be compute-bound,
/// but the HBM savings translate directly to time savings on memory-
/// bound layers.
pub fn est_latency_us(hbm_bytes: u64, gpu: &GpuSpec) -> f64 {
    let bw_bps = gpu.peak_bandwidth_gbs * 1e9;
    (hbm_bytes as f64 / bw_bps) * 1e6
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Plan a single layer at the requested level, downgrading if SMEM is
/// insufficient.  Returns the actual level picked plus its cost.
pub fn plan_layer(
    layer: impl Into<String>,
    shape: LayerShape,
    gpu: &GpuSpec,
    requested: FusionLevel,
) -> LayerPlan {
    let layer = layer.into();
    let tiles = roofline_tile_config(shape, gpu);
    let budget = smem_budget_bytes(gpu);

    // Walk down levels until one fits.
    let (level, smem, downgrade_reason) = pick_level(shape, tiles, budget, requested);

    let baseline_hbm = baseline_hbm_bytes(shape);
    let fused_hbm = fused_hbm_bytes(shape, level);
    let baseline_us = est_latency_us(baseline_hbm, gpu);
    let fused_us = est_latency_us(fused_hbm, gpu);

    // Sprint 3 (paper §6.3 visibility): derive backward tier per layer so the
    // CSHA report can surface "Tier B.2 (hybrid)" vs "Tier C (scalar)" alongside
    // the forward tier. Uses the same dispatch predicate the training pipeline
    // calls at codegen time so the report matches what would actually fire.
    let backward_tier = layer_backward_tier_report(shape, tiles, gpu, level);

    LayerPlan {
        layer,
        level,
        tiles,
        smem_bytes: smem,
        smem_budget_bytes: budget,
        hbm_traffic_bytes: fused_hbm,
        baseline_hbm_bytes: baseline_hbm,
        est_time_us: fused_us,
        baseline_time_us: baseline_us,
        downgrade_reason,
        backward_tier,
    }
}

/// Construct a representative `FlashAttentionConfig` from a planner layer
/// + shape + GPU and ask `backward_dispatch_tier` what backward kernel
/// would fire for it. Maps the planner's `FusionLevel` to the kernel's
/// `csha.level: u8` (Boundary=1, Pipeline=2, Block=3); `FusionLevel::None`
/// yields `BackwardTierReport::Scalar` without consulting the dispatcher
/// (no CSHA -> no Tier B.2 by definition).
///
/// This is a planner-side bridge: it does NOT emit code or affect the
/// dispatcher's runtime decision. It only mirrors what the dispatcher
/// would pick given the same inputs, so `--csha-report` can show it.
fn layer_backward_tier_report(
    shape: LayerShape,
    tiles: TileConfig,
    gpu: &GpuSpec,
    level: FusionLevel,
) -> BackwardTierReport {
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use crate::flash_attention_v2::tier_b2::BackwardTier;

    let csha_level: u8 = match level {
        FusionLevel::None => return BackwardTierReport::Scalar,
        FusionLevel::Boundary => 1,
        FusionLevel::Pipeline => 2,
        FusionLevel::Block => 3,
    };

    let cfg = FlashAttentionConfig {
        block_q: tiles.block_q as i64,
        block_kv: tiles.block_kv as i64,
        head_dim: tiles.head_dim as i64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: gpu.sm_version,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: csha_level,
            d_model: shape.d_model as u32,
            ..Default::default()
        }),
    };

    match backward_dispatch_tier(&cfg) {
        BackwardTier::Scalar => BackwardTierReport::Scalar,
        BackwardTier::TierB2 { bq, bkv, chunk } => {
            BackwardTierReport::TierB2 { bq, bkv, chunk }
        }
    }
}

fn pick_level(
    shape: LayerShape,
    tiles: TileConfig,
    budget: u64,
    requested: FusionLevel,
) -> (FusionLevel, u64, Option<String>) {
    // Fast exit for None / Boundary which don't allocate kernel SMEM.
    match requested {
        FusionLevel::None => return (FusionLevel::None, 0, None),
        FusionLevel::Boundary => return (FusionLevel::Boundary, 0, None),
        _ => {}
    }

    if requested == FusionLevel::Block {
        let need = block_smem_bytes(shape, tiles);
        if need <= budget {
            return (FusionLevel::Block, need, None);
        }
        // Downgrade to Pipeline.
        let reason = format!(
            "Level 3 needs {} KB SMEM, budget is {} KB",
            need / 1024,
            budget / 1024
        );
        let need2 = pipeline_smem_bytes(shape, tiles);
        if need2 <= budget {
            return (FusionLevel::Pipeline, need2, Some(reason));
        }
        // Downgrade further to Boundary.
        return (
            FusionLevel::Boundary,
            0,
            Some(format!(
                "{}; Level 2 also over budget ({} KB)",
                reason,
                need2 / 1024
            )),
        );
    }

    // requested == Pipeline
    let need = pipeline_smem_bytes(shape, tiles);
    if need <= budget {
        (FusionLevel::Pipeline, need, None)
    } else {
        (
            FusionLevel::Boundary,
            0,
            Some(format!(
                "Level 2 needs {} KB SMEM, budget is {} KB",
                need / 1024,
                budget / 1024
            )),
        )
    }
}

/// Plan every layer in a boundary scan at the requested level.  Each
/// layer gets its own [`LayerPlan`]; layers absent from the scan are
/// skipped (they have no attention sublayer to optimise).
pub fn plan_all(
    scan: &BoundaryScan,
    shape: LayerShape,
    gpu: &GpuSpec,
    requested: FusionLevel,
) -> Vec<LayerPlan> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for c in &scan.chains {
        let key = c.layer.clone().unwrap_or_else(|| "other".to_string());
        if seen.insert(key.clone()) {
            out.push(plan_layer(key, shape, gpu, requested));
        }
    }
    out
}

/// Returns which backward kernel family the planner should dispatch for this config.
/// Tier B.2 takes precedence when its preconditions hold; falls back to scalar v2 otherwise.
///
/// **Phase 1 contract:** Tier B.2 emitters are stubs that always return
/// `Err(NotImplemented)`. Callers that receive `BackwardTier::TierB2` MUST
/// transparently fall back to scalar v2 backward (see
/// `flash_attention_v2::synthesize_backward_with_tier`). This reserves the
/// dispatch path so Phase 2 can replace the stubs without changing the
/// cost-model or planner interface.
pub fn backward_dispatch_tier(
    config: &crate::flash_attention::FlashAttentionConfig,
) -> crate::flash_attention_v2::tier_b2::BackwardTier {
    use crate::flash_attention_v2::tier_b2::BackwardTier;
    use crate::flash_attention_v2::tier_b2::dispatch::tier_b2_can_dispatch;
    match tier_b2_can_dispatch(config) {
        Ok(tier_b2) => tier_b2,
        Err(_) => BackwardTier::Scalar,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csha_boundary::BoundaryChain;
    use crate::gpu_specs::{default_gpu, find_gpu};

    fn small_shape() -> LayerShape {
        LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn baseline_exceeds_pipeline_traffic() {
        let s = small_shape();
        assert!(baseline_hbm_bytes(s) > fused_hbm_bytes(s, FusionLevel::Pipeline));
    }

    #[test]
    fn boundary_traffic_is_between_baseline_and_pipeline() {
        let s = small_shape();
        let b = fused_hbm_bytes(s, FusionLevel::Boundary);
        let p = fused_hbm_bytes(s, FusionLevel::Pipeline);
        let base = baseline_hbm_bytes(s);
        assert!(p < b);
        assert!(b < base);
    }

    #[test]
    fn plan_layer_fits_on_h100() {
        let gpu = find_gpu("H100").unwrap_or_else(default_gpu);
        let plan = plan_layer("blocks.0", small_shape(), gpu, FusionLevel::Pipeline);
        assert_eq!(plan.level, FusionLevel::Pipeline);
        assert!(plan.smem_bytes > 0);
        assert!(plan.smem_bytes <= plan.smem_budget_bytes);
        assert!(plan.downgrade_reason.is_none());
    }

    #[test]
    fn huge_layer_downgrades_to_boundary() {
        let gpu = find_gpu("H100").unwrap_or_else(default_gpu);
        let huge = LayerShape {
            batch: 1,
            seq: 4096,
            d_model: 16384,
            head_dim: 256,
            n_kv_heads: 8,
            dtype_bytes: 2,
        };
        let plan = plan_layer("blocks.0", huge, gpu, FusionLevel::Block);
        // Block fusion will not fit; planner must downgrade.
        assert_ne!(plan.level, FusionLevel::Block);
        assert!(plan.downgrade_reason.is_some());
    }

    #[test]
    fn none_level_skips_smem_calculation() {
        let gpu = default_gpu();
        let plan = plan_layer("x", small_shape(), gpu, FusionLevel::None);
        assert_eq!(plan.level, FusionLevel::None);
        assert_eq!(plan.smem_bytes, 0);
    }

    #[test]
    fn speedup_is_positive_for_memory_bound() {
        let gpu = find_gpu("H100").unwrap_or_else(default_gpu);
        let plan = plan_layer("blocks.0", small_shape(), gpu, FusionLevel::Pipeline);
        assert!(plan.speedup() > 1.0);
        assert!(plan.hbm_reduction() > 1.0);
    }

    #[test]
    fn plan_all_deduplicates_layers() {
        let gpu = default_gpu();
        let scan = BoundaryScan {
            chains: vec![
                BoundaryChain {
                    layer: Some("blocks.0".into()),
                    kind: crate::csha_boundary::ProjKind::Q,
                    norm_op: 0,
                    matmul_op: 1,
                    rope_op: Some(2),
                    weight_param: "blocks.0.attn.wq".into(),
                    sdpa_op: None,
                },
                BoundaryChain {
                    layer: Some("blocks.0".into()),
                    kind: crate::csha_boundary::ProjKind::K,
                    norm_op: 0,
                    matmul_op: 3,
                    rope_op: Some(4),
                    weight_param: "blocks.0.attn.wk".into(),
                    sdpa_op: None,
                },
                BoundaryChain {
                    layer: Some("blocks.1".into()),
                    kind: crate::csha_boundary::ProjKind::Q,
                    norm_op: 5,
                    matmul_op: 6,
                    rope_op: Some(7),
                    weight_param: "blocks.1.attn.wq".into(),
                    sdpa_op: None,
                },
            ],
        };
        let plans = plan_all(&scan, small_shape(), gpu, FusionLevel::Pipeline);
        // Two unique layers in the scan → two plans.
        assert_eq!(plans.len(), 2);
    }

    #[test]
    fn roofline_picks_valid_tiles() {
        let gpu = default_gpu();
        let t = roofline_tile_config(small_shape(), gpu);
        assert!(t.block_q >= 32 && t.block_q <= 128);
        assert!(t.block_kv >= 32 && t.block_kv <= 128);
        assert_eq!(t.head_dim, small_shape().head_dim as u32);
    }

    #[test]
    fn smem_budget_is_capped() {
        let gpu = default_gpu();
        let b = smem_budget_bytes(gpu);
        assert!(b <= 228 * 1024);
    }

    // ----------------------------------------------------------------------
    // B1.1 coverage: configurations newly admitted by the corrected formula
    // ----------------------------------------------------------------------
    //
    // The B1.1 corrections (see V3 findings doc) shrink
    // `pipeline_smem_bytes` for typical shapes via two competing effects:
    //   * `+24576` from doubled K/V tiles + chunk staging (worst-case
    //     inflation)
    //   * `-bq*hd*4 - 2*hd*min(dm,256)` from O_acc zeroed + w_tile replaced
    //     (worst-case savings)
    //
    // The net effect is a NEGATIVE delta for the common "hd=256" regime
    // (e.g. -110_592 bytes at balanced tiles, -78_848 at compute-bound),
    // which moves configurations that were over the 228 KB H100 cap into
    // the admitted set.  Each test below picks a representative such
    // config from the §3.4 supported matrix and asserts the planner now
    // admits it at Pipeline level.

    fn h100() -> &'static GpuSpec {
        find_gpu("H100").unwrap_or_else(default_gpu)
    }

    fn shape_with(d_model: u64, head_dim: u64) -> LayerShape {
        LayerShape {
            batch: 1,
            seq: 1024,
            d_model,
            head_dim,
            n_kv_heads: 8,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn newly_unblocked_balanced_dm512_hd256_admits_pipeline() {
        // roofline: ai = 512/2 = 256, balanced regime → (bq=64, bkv=64).
        // Pre-B1.1:  74240 ... no wait — at hd=256 not hd=64:
        //   pre  = 32768 + 32768 + 32768 + 65536 + 512 + 131072 = 295424  (over 228 KB)
        //   post = 32768 + 131072 + 0 + 512 + 20480              = 184832  (fits)
        let gpu = h100();
        let plan = plan_layer(
            "newly_unblocked_balanced_dm512_hd256",
            shape_with(512, 256),
            gpu,
            FusionLevel::Pipeline,
        );
        assert_eq!(
            plan.level,
            FusionLevel::Pipeline,
            "balanced (64,64) at hd=256 must admit Pipeline post-B1.1; got level={:?} reason={:?}",
            plan.level,
            plan.downgrade_reason
        );
        assert!(plan.downgrade_reason.is_none());
    }

    #[test]
    fn newly_unblocked_balanced_dm1024_hd256_admits_pipeline() {
        // roofline: ai = 1024/2 = 512, still balanced → (bq=64, bkv=64).
        // Pre-B1.1: 295424 (same as dm=512 case — min(dm,256) caps the
        //                   pre-correction w_tile term)
        // Post-B1.1: 184832 (chunk staging does not depend on dm)
        let gpu = h100();
        let plan = plan_layer(
            "newly_unblocked_balanced_dm1024_hd256",
            shape_with(1024, 256),
            gpu,
            FusionLevel::Pipeline,
        );
        assert_eq!(
            plan.level,
            FusionLevel::Pipeline,
            "balanced (64,64) at hd=256 dm=1024 must admit Pipeline post-B1.1; got level={:?} reason={:?}",
            plan.level,
            plan.downgrade_reason
        );
        assert!(plan.downgrade_reason.is_none());
    }

    // ----------------------------------------------------------------------
    // Sprint 3 (paper §6.3 visibility): backward_tier field
    // ----------------------------------------------------------------------

    #[test]
    fn backward_tier_is_tier_b2_for_hd128_sm80_pipeline() {
        // Canonical Tier B.2 admission: hd=128, Ampere+ SM, level=2 (Pipeline).
        // Picks up the per-hd ladder bq=64, chunk=4 (see tier_b2/dispatch.rs).
        let gpu = h100();
        let shape = LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 1024,
            head_dim: 128,
            n_kv_heads: 8,
            dtype_bytes: 2,
        };
        let plan = plan_layer("blocks.0", shape, gpu, FusionLevel::Pipeline);
        match plan.backward_tier {
            BackwardTierReport::TierB2 { bq, bkv, chunk } => {
                assert_eq!(bq, 64, "hd=128 ladder pins bq=64");
                assert_eq!(bkv, 64, "hd=128 ladder pins bkv=64");
                assert_eq!(chunk, 4, "hd=128 ladder pins chunk=4");
            }
            BackwardTierReport::Scalar => {
                panic!(
                    "expected Tier B.2 for hd=128 sm=90 level=Pipeline, got Scalar (plan={:?})",
                    plan
                );
            }
        }
    }

    #[test]
    fn backward_tier_is_scalar_for_level_none() {
        // FusionLevel::None has no CSHA -> Tier B.2 can't fire.
        let gpu = h100();
        let plan = plan_layer("blocks.0", small_shape(), gpu, FusionLevel::None);
        assert_eq!(plan.backward_tier, BackwardTierReport::Scalar);
    }

    #[test]
    fn backward_tier_is_scalar_for_boundary_level() {
        // FusionLevel::Boundary maps to csha.level=1 -> dispatch rejects with LevelTooLow.
        let gpu = h100();
        let plan = plan_layer("blocks.0", small_shape(), gpu, FusionLevel::Boundary);
        assert_eq!(plan.backward_tier, BackwardTierReport::Scalar);
    }

    #[test]
    fn backward_tier_label_distinguishes_scalar_from_tier_b2() {
        assert!(BackwardTierReport::Scalar.as_label().contains("Tier C"));
        assert!(BackwardTierReport::Scalar.as_label().contains("scalar"));
        let label = BackwardTierReport::TierB2 { bq: 64, bkv: 64, chunk: 4 }.as_label();
        assert!(label.contains("Tier B.2"), "label='{label}'");
        assert!(label.contains("hybrid"), "label='{label}'");
        assert!(label.contains("bq=64"), "label='{label}'");
    }

    #[test]
    fn newly_unblocked_compute_bound_dm2048_hd256_admits_pipeline() {
        // roofline: ai = 2048/2 = 1024, compute-bound → (bq=32, bkv=64).
        //   pre  = 16384 + 32768 + 32768 + 32768 + 256 + 131072 = 246016  (over 228 KB)
        //   post = 16384 + 131072 + 0 + 256 + 19456              = 167168  (fits)
        let gpu = h100();
        let plan = plan_layer(
            "newly_unblocked_compute_dm2048_hd256",
            shape_with(2048, 256),
            gpu,
            FusionLevel::Pipeline,
        );
        assert_eq!(
            plan.level,
            FusionLevel::Pipeline,
            "compute-bound (32,64) at hd=256 dm=2048 must admit Pipeline post-B1.1; got level={:?} reason={:?}",
            plan.level,
            plan.downgrade_reason
        );
        assert!(plan.downgrade_reason.is_none());
    }
}
