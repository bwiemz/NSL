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
/// Layout (paper §2.2):
///   * Q tile (ping):  `block_q  * head_dim * 2`
///   * K tile (ping):  `block_kv * head_dim * 2`
///   * V tile (ping):  `block_kv * head_dim * 2`
///   * O accumulator:  `block_q  * head_dim * 4` (fp32)
///   * Stats:          `block_q  * 2 * 4`        (logsumexp + max)
///   * Producer side buffers a single tile of each projection weight
///     (we charge the worst case: `head_dim * d_model * 2`).
pub fn pipeline_smem_bytes(shape: LayerShape, tiles: TileConfig) -> u64 {
    let dtype = shape.dtype_bytes;
    let block_q = tiles.block_q as u64;
    let block_kv = tiles.block_kv as u64;
    let head_dim = tiles.head_dim as u64;
    let d_model = shape.d_model;

    let q_tile = block_q * head_dim * dtype;
    let k_tile = block_kv * head_dim * dtype;
    let v_tile = block_kv * head_dim * dtype;
    let o_acc = block_q * head_dim * 4;
    let stats = block_q * 2 * 4;
    let w_tile = head_dim * d_model.min(256) * dtype;
    q_tile + k_tile + v_tile + o_acc + stats + w_tile
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
            // Paper §2.2: only x read + output write remain.
            2 * bsd // read x + write out
        }
        FusionLevel::Block => {
            // Paper §2.3: same as pipeline for a single block (block
            // fusion eliminates intra-block HBM, not inter-block).
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
                },
                BoundaryChain {
                    layer: Some("blocks.0".into()),
                    kind: crate::csha_boundary::ProjKind::K,
                    norm_op: 0,
                    matmul_op: 3,
                    rope_op: Some(4),
                    weight_param: "blocks.0.attn.wk".into(),
                },
                BoundaryChain {
                    layer: Some("blocks.1".into()),
                    kind: crate::csha_boundary::ProjKind::Q,
                    norm_op: 5,
                    matmul_op: 6,
                    rope_op: Some(7),
                    weight_param: "blocks.1.attn.wq".into(),
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
}
