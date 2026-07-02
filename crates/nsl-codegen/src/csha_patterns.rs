//! CSHA Compile-Time Attention Patterns (paper §4).
//!
//! Three per-layer static decisions, orthogonal to boundary fusion and
//! pipelining:
//!
//!   * **Causal mask** (§4.1): short sequences need no per-query mask at
//!     all; tile-aligned sequences use a bit-level mask; general case
//!     needs a full dynamic mask.
//!   * **GQA expansion** (§4.2): lower KV-head replication to a SMEM
//!     stride pattern, eliminating the runtime `contiguous()` copy.
//!   * **Attention sinks** (§4.3): detect whether the first N tokens
//!     consistently dominate attention and should be kept resident.
//!
//! All outputs are static functions of (shape, tile size, pre-trained
//! weights) — pure analysis, no PTX emission. `csha.rs` composes this
//! with the boundary/pipeline/specialize plans into the final `CshaPlan`.

use serde::Serialize;

use crate::csha_pipeline::LayerPlan;
use crate::weight_aware::{WeightEntry, WeightMap};
use crate::wggo_cost::LayerShape;

/// Causal mask strategy (§4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CausalMaskStrategy {
    /// Sequence fits in a single KV tile; upper-triangular mask baked in
    /// as a compile-time constant — no per-query runtime check.
    SingleTileStatic,
    /// Sequence length is a compile-time multiple of `block_kv`; use a
    /// bitmask that selects whole tiles instead of per-element compares.
    TileAlignedBitmask,
    /// General case: per-element mask in the kernel.
    Dynamic,
    /// Causal masking disabled (bidirectional attention).
    None,
}

impl CausalMaskStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            CausalMaskStrategy::SingleTileStatic => "single-tile-static",
            CausalMaskStrategy::TileAlignedBitmask => "tile-aligned-bitmask",
            CausalMaskStrategy::Dynamic => "dynamic",
            CausalMaskStrategy::None => "none",
        }
    }
}

/// GQA expansion strategy (§4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GqaStrategy {
    /// No GQA — every head has its own KV.
    None,
    /// Zero-copy expansion: replicate KV into each head group via a
    /// shared-memory stride pattern. Eliminates runtime `contiguous()`.
    ZeroCopyStride { group_size: u32 },
}

impl GqaStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            GqaStrategy::None => "none",
            GqaStrategy::ZeroCopyStride { .. } => "zero-copy-stride",
        }
    }
}

/// Attention-sink caching decision (§4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SinkStrategy {
    Off,
    PersistentCache { tokens: u32 },
}

impl SinkStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            SinkStrategy::Off => "off",
            SinkStrategy::PersistentCache { .. } => "persistent-cache",
        }
    }
}

/// Aggregate result of the per-layer pattern analysis.
#[derive(Debug, Clone, Serialize)]
pub struct PatternDecision {
    pub layer: String,
    pub causal_mask: CausalMaskStrategy,
    pub gqa: GqaStrategy,
    pub sink: SinkStrategy,
}

/// Full pattern plan: one entry per layer in the pipeline scan.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PatternPlan {
    pub layers: Vec<PatternDecision>,
}

impl PatternPlan {
    pub fn get(&self, layer: &str) -> Option<&PatternDecision> {
        self.layers.iter().find(|d| d.layer == layer)
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

/// Config for pattern analysis (thresholds / flags the driver cannot
/// derive itself).
#[derive(Debug, Clone, Copy)]
pub struct PatternConfig {
    /// Whether the model uses causal masking. GQA LLMs default to `true`;
    /// encoder models to `false`.
    pub causal: bool,
    /// Upper bound on sink tokens we are willing to pin in SMEM.
    pub max_sink_tokens: u32,
    /// Sink detection: a token is flagged as a sink when its embedding
    /// Frobenius norm exceeds `sink_threshold_rel × median(embedding)`.
    pub sink_threshold_rel: f64,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            causal: true,
            max_sink_tokens: 4,
            sink_threshold_rel: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Decision helpers
// ---------------------------------------------------------------------------

/// Pick the causal-mask strategy.
pub fn decide_causal_mask(seq_len: u64, block_kv: u64, enabled: bool) -> CausalMaskStrategy {
    if !enabled {
        return CausalMaskStrategy::None;
    }
    if block_kv == 0 {
        return CausalMaskStrategy::Dynamic;
    }
    if seq_len <= block_kv {
        CausalMaskStrategy::SingleTileStatic
    } else if seq_len.is_multiple_of(block_kv) {
        CausalMaskStrategy::TileAlignedBitmask
    } else {
        CausalMaskStrategy::Dynamic
    }
}

/// Decide whether GQA should be expressed as a compile-time stride.
///
/// If `n_heads == n_kv_heads`, standard MHA — no expansion needed. When
/// `n_heads > n_kv_heads`, CSHA always emits the zero-copy stride pattern
/// (there is no reason to prefer a runtime `contiguous()` copy once the
/// dims are known).
pub fn decide_gqa(n_heads: u32, n_kv_heads: u32) -> GqaStrategy {
    if n_heads == n_kv_heads || n_kv_heads == 0 {
        GqaStrategy::None
    } else if n_heads.is_multiple_of(n_kv_heads) {
        GqaStrategy::ZeroCopyStride {
            group_size: n_heads / n_kv_heads,
        }
    } else {
        // Non-integer group size is not a legal GQA config; keep unfused
        // so the caller falls back to standard runtime expansion.
        GqaStrategy::None
    }
}

/// Detect attention sinks from the pre-trained embedding.
///
/// Heuristic (paper §4.3): compute per-token Frobenius norm of the
/// embedding for the first few token indices, and compare against the
/// median norm across a stride-sampled subset of the vocabulary. Tokens
/// whose norm exceeds `threshold_rel × median` are flagged as sinks.
/// Without weights, returns `Off`.
pub fn decide_sink(embedding: Option<&WeightEntry>, cfg: &PatternConfig) -> SinkStrategy {
    let Some(entry) = embedding else {
        return SinkStrategy::Off;
    };
    if entry.shape.len() != 2 {
        return SinkStrategy::Off;
    }
    let vocab = entry.shape[0];
    let d = entry.shape[1];
    let bw = entry.dtype.byte_width();
    if vocab < 4 || d == 0 || bw == 0 {
        return SinkStrategy::Off;
    }

    let sniff_count = vocab.min(8);
    let mut sample_norms = Vec::with_capacity(sniff_count);
    for t in 0..sniff_count {
        let mut sum_sq = 0.0_f64;
        for c in 0..d {
            let idx = t * d + c;
            let off = idx * bw;
            if off + bw > entry.data.len() {
                return SinkStrategy::Off;
            }
            let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
            sum_sq += v * v;
        }
        sample_norms.push(sum_sq.sqrt());
    }

    let stride = (vocab / 64).max(1);
    let mut stride_norms = Vec::new();
    for t in (0..vocab).step_by(stride) {
        let mut sum_sq = 0.0_f64;
        for c in 0..d {
            let idx = t * d + c;
            let off = idx * bw;
            if off + bw > entry.data.len() {
                return SinkStrategy::Off;
            }
            let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
            sum_sq += v * v;
        }
        stride_norms.push(sum_sq.sqrt());
    }
    if stride_norms.is_empty() {
        return SinkStrategy::Off;
    }
    stride_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = stride_norms[stride_norms.len() / 2];
    let threshold = median * cfg.sink_threshold_rel;

    // Sinks are a *contiguous* prefix of the vocabulary (paper §4.3 —
    // BOS tokens). Stop counting at the first non-sink so that isolated
    // high-norm tokens later in the sniff window do not pin non-sink
    // tokens into the persistent cache.
    let mut sink_tokens = 0u32;
    for (i, n) in sample_norms.iter().enumerate() {
        if i as u32 >= cfg.max_sink_tokens {
            break;
        }
        if *n > threshold {
            sink_tokens = (i as u32) + 1;
        } else {
            break;
        }
    }
    if sink_tokens == 0 {
        SinkStrategy::Off
    } else {
        SinkStrategy::PersistentCache {
            tokens: sink_tokens,
        }
    }
}

/// Locate the token-embedding tensor in a `WeightMap` using common
/// naming conventions. Falls back to any 2-D entry whose name contains
/// `embed`. Returns `None` if nothing matches.
pub fn find_embedding(wm: &WeightMap) -> Option<&WeightEntry> {
    for cand in [
        "embedding.weight",
        "tok_embeddings.weight",
        "wte.weight",
        "embeddings.token.weight",
        "model.embed_tokens.weight",
    ] {
        if let Some(e) = wm.get(cand) {
            return Some(e);
        }
    }
    for (n, e) in wm.entries() {
        if n.contains("embed") && e.shape.len() == 2 {
            return Some(e);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// Run the pattern analysis for every layer in `per_layer`, using the
/// global `shape` for sequence / head counts and the per-layer
/// `block_kv` from each [`LayerPlan`].
pub fn analyze(
    per_layer: &[LayerPlan],
    shape: &LayerShape,
    n_heads: u32,
    weights: Option<&WeightMap>,
    cfg: &PatternConfig,
) -> PatternPlan {
    let embedding = weights.and_then(find_embedding);
    let sink = decide_sink(embedding, cfg);
    let gqa = decide_gqa(n_heads, shape.n_kv_heads as u32);

    let layers = per_layer
        .iter()
        .map(|plan| {
            let causal_mask =
                decide_causal_mask(shape.seq, plan.tiles.block_kv as u64, cfg.causal);
            PatternDecision {
                layer: plan.layer.clone(),
                causal_mask,
                gqa,
                sink,
            }
        })
        .collect();
    PatternPlan { layers }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csha_pipeline::{FusionLevel, TileConfig};
    use crate::weight_aware::WeightDType;

    fn layer_plan(name: &str, block_kv: u32) -> LayerPlan {
        LayerPlan {
            layer: name.to_string(),
            level: FusionLevel::Boundary,
            tiles: TileConfig {
                block_q: 64,
                block_kv,
                head_dim: 64,
            },
            smem_bytes: 0,
            smem_budget_bytes: 228 * 1024,
            hbm_traffic_bytes: 0,
            baseline_hbm_bytes: 0,
            est_time_us: 0.0,
            baseline_time_us: 0.0,
            downgrade_reason: None,
            backward_tier: crate::csha_pipeline::BackwardTierReport::Scalar,
        }
    }

    fn shape(seq: u64) -> LayerShape {
        LayerShape {
            batch: 1,
            seq,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn short_seq_uses_single_tile_static() {
        assert_eq!(
            decide_causal_mask(32, 128, true),
            CausalMaskStrategy::SingleTileStatic
        );
    }

    #[test]
    fn aligned_seq_uses_bitmask() {
        assert_eq!(
            decide_causal_mask(1024, 128, true),
            CausalMaskStrategy::TileAlignedBitmask
        );
    }

    #[test]
    fn unaligned_seq_is_dynamic() {
        assert_eq!(
            decide_causal_mask(1025, 128, true),
            CausalMaskStrategy::Dynamic
        );
    }

    #[test]
    fn causal_disabled_returns_none() {
        assert_eq!(
            decide_causal_mask(1024, 128, false),
            CausalMaskStrategy::None
        );
    }

    #[test]
    fn block_kv_zero_is_dynamic() {
        assert_eq!(
            decide_causal_mask(1024, 0, true),
            CausalMaskStrategy::Dynamic
        );
    }

    #[test]
    fn gqa_stride_picked_when_heads_gt_kv() {
        assert_eq!(
            decide_gqa(8, 4),
            GqaStrategy::ZeroCopyStride { group_size: 2 }
        );
    }

    #[test]
    fn gqa_none_when_balanced() {
        assert_eq!(decide_gqa(8, 8), GqaStrategy::None);
    }

    #[test]
    fn gqa_none_on_non_integer_ratio() {
        // 7 heads / 3 kv heads is not a legal GQA config.
        assert_eq!(decide_gqa(7, 3), GqaStrategy::None);
    }

    #[test]
    fn sink_off_without_weights() {
        assert_eq!(
            decide_sink(None, &PatternConfig::default()),
            SinkStrategy::Off
        );
    }

    fn make_embedding(per_token_norms: &[f64]) -> WeightEntry {
        let vocab = per_token_norms.len();
        let d = 16;
        let mut data = vec![0u8; vocab * d * 4];
        for (t, target) in per_token_norms.iter().enumerate() {
            // Uniform components so that the L2 norm equals `target`.
            let v = (target / (d as f64).sqrt()) as f32;
            for c in 0..d {
                let idx = t * d + c;
                let off = idx * 4;
                data[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        WeightEntry {
            name: "embedding.weight".to_string(),
            data,
            shape: vec![vocab, d],
            dtype: WeightDType::F32,
            num_elements: vocab * d,
            sparsity: None,
            eliminated: false,
        }
    }

    #[test]
    fn sink_detected_when_first_token_dominates() {
        let mut norms = vec![1.0; 128];
        norms[0] = 100.0;
        let e = make_embedding(&norms);
        match decide_sink(Some(&e), &PatternConfig::default()) {
            SinkStrategy::PersistentCache { tokens } => assert!(tokens >= 1),
            SinkStrategy::Off => panic!("expected sink detected"),
        }
    }

    #[test]
    fn sink_requires_contiguous_prefix() {
        // Token 0 uniform, token 2 high: first non-sink at index 0
        // breaks the scan before the high-norm token is considered.
        let mut norms = vec![1.0; 128];
        norms[2] = 100.0;
        let e = make_embedding(&norms);
        assert_eq!(
            decide_sink(Some(&e), &PatternConfig::default()),
            SinkStrategy::Off,
            "non-contiguous high-norm tokens must not be reported as sinks"
        );
    }

    #[test]
    fn sink_off_for_uniform_embedding() {
        let e = make_embedding(&vec![1.0; 128]);
        assert_eq!(
            decide_sink(Some(&e), &PatternConfig::default()),
            SinkStrategy::Off
        );
    }

    #[test]
    fn analyze_composes_per_layer_causal_and_shared_gqa_sink() {
        let per_layer = vec![
            layer_plan("blocks.0", 128), // seq=1024, block_kv=128 → bitmask
            layer_plan("blocks.1", 256), // seq=1024, block_kv=256 → bitmask
            layer_plan("blocks.2", 2048), // seq=1024 < block_kv=2048 → single-tile
        ];
        let plan = analyze(&per_layer, &shape(1024), 8, None, &PatternConfig::default());
        assert_eq!(plan.layers.len(), 3);
        assert_eq!(plan.layers[0].causal_mask, CausalMaskStrategy::TileAlignedBitmask);
        assert_eq!(plan.layers[1].causal_mask, CausalMaskStrategy::TileAlignedBitmask);
        assert_eq!(plan.layers[2].causal_mask, CausalMaskStrategy::SingleTileStatic);
        // GQA and sink are model-wide.
        for d in &plan.layers {
            assert_eq!(d.gqa, GqaStrategy::ZeroCopyStride { group_size: 2 });
            assert_eq!(d.sink, SinkStrategy::Off);
        }
    }

    #[test]
    fn analyze_empty_when_no_layers() {
        let plan = analyze(&[], &shape(1024), 8, None, &PatternConfig::default());
        assert!(plan.is_empty());
    }
}
