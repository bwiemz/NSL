//! WGGO — cost-model interface with per-layer LUT (linearization).
//!
//! The ILP solver needs an objective function that evaluates a candidate
//! configuration `(head_count, ffn_width, csha_level, adapter_rank,
//! precision, shard_factor)` in constant time.  That's what this module
//! does: it pre-computes the M37 roofline cost model over a discrete
//! product-of-domains (paper §4.3.4) and exposes a lookup API.
//!
//! The LUT has at most ~160 entries per layer (8 heads × 5 FFN widths × 4
//! CSHA levels) and is trivially small.  All four dimensions are
//! compile-time-bounded.

use crate::cost_model::{
    arithmetic_intensity, classify_op, matmul_cost, rmsnorm_cost, softmax_cost,
    BoundClassification,
};
use crate::gpu_specs::GpuSpec;

/// Per-entry cost vector.
#[derive(Debug, Clone, Copy)]
pub struct LayerCostEntry {
    /// Forward-pass latency estimate (microseconds).
    pub forward_us: f64,
    /// Backward-pass latency estimate (microseconds).
    pub backward_us: f64,
    /// Parameter bytes (shard-agnostic).
    pub param_bytes: u64,
    /// Activation bytes (per micro-batch).
    pub activation_bytes: u64,
    /// Shared-memory usage for the chosen fusion level (bytes).
    pub smem_bytes: u64,
    /// Whether this configuration is feasible on the target.
    pub feasible: bool,
    /// Classification of the dominant matmul — informs sharding decisions.
    pub classification: BoundClassification,
}

impl LayerCostEntry {
    /// Sum of per-layer latencies used as an objective contribution.
    pub fn total_us(self) -> f64 {
        self.forward_us + self.backward_us
    }
}

/// Shape parameters for one transformer layer.
#[derive(Debug, Clone, Copy)]
pub struct LayerShape {
    pub batch: u64,
    pub seq: u64,
    pub d_model: u64,
    pub head_dim: u64,
    pub n_kv_heads: u64,
    pub dtype_bytes: u64,
}

impl LayerShape {
    /// Minimal 4-head fixture for unit tests.
    ///
    /// `d_model=256`, `head_dim=64` → `num_heads = 4`.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn default_for_test_4heads() -> Self {
        Self {
            batch: 1,
            seq: 64,
            d_model: 256,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        }
    }
}

/// Decision dimensions the LUT spans.  Each dimension has a small discrete
/// domain — the LUT stores the full cross-product.
#[derive(Debug, Clone)]
pub struct LutAxes {
    pub head_counts: Vec<u64>,
    pub ffn_widths: Vec<u64>,
    pub csha_levels: Vec<u8>,
    pub adapter_ranks: Vec<u64>,
}

impl Default for LutAxes {
    fn default() -> Self {
        Self {
            head_counts: (1..=8).collect(),
            ffn_widths: vec![384, 768, 1024, 1408, 2048],
            csha_levels: vec![0, 1, 2, 3],
            adapter_ranks: vec![0, 2, 4, 8, 16],
        }
    }
}

/// Per-layer LUT covering every (heads × ffn × csha × adapter) combination.
#[derive(Debug, Clone)]
pub struct LayerCostLut {
    pub axes_head_counts: Vec<u64>,
    pub axes_ffn_widths: Vec<u64>,
    pub axes_csha_levels: Vec<u8>,
    pub axes_adapter_ranks: Vec<u64>,
    /// Flattened row-major `[heads][ffn][csha][rank]` entries.
    pub entries: Vec<LayerCostEntry>,
}

impl LayerCostLut {
    fn index(&self, h_idx: usize, f_idx: usize, c_idx: usize, r_idx: usize) -> usize {
        ((h_idx * self.axes_ffn_widths.len() + f_idx) * self.axes_csha_levels.len() + c_idx)
            * self.axes_adapter_ranks.len()
            + r_idx
    }

    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Look up an entry by (head_count, ffn_width, csha_level, adapter_rank).
    pub fn get(&self, heads: u64, ffn: u64, csha: u8, rank: u64) -> Option<LayerCostEntry> {
        let hi = self.axes_head_counts.iter().position(|&x| x == heads)?;
        let fi = self.axes_ffn_widths.iter().position(|&x| x == ffn)?;
        let ci = self.axes_csha_levels.iter().position(|&x| x == csha)?;
        let ri = self.axes_adapter_ranks.iter().position(|&x| x == rank)?;
        Some(self.entries[self.index(hi, fi, ci, ri)])
    }

    /// Cheapest feasible entry, if any.  Returned tuple is `(heads, ffn,
    /// csha, rank, entry)`.
    pub fn argmin_feasible(&self) -> Option<(u64, u64, u8, u64, LayerCostEntry)> {
        let mut best: Option<(u64, u64, u8, u64, LayerCostEntry)> = None;
        for (hi, &h) in self.axes_head_counts.iter().enumerate() {
            for (fi, &f) in self.axes_ffn_widths.iter().enumerate() {
                for (ci, &c) in self.axes_csha_levels.iter().enumerate() {
                    for (ri, &r) in self.axes_adapter_ranks.iter().enumerate() {
                        let e = self.entries[self.index(hi, fi, ci, ri)];
                        if !e.feasible {
                            continue;
                        }
                        if best.is_none_or(|(_, _, _, _, b)| e.total_us() < b.total_us()) {
                            best = Some((h, f, c, r, e));
                        }
                    }
                }
            }
        }
        best
    }
}

/// Build the per-layer LUT.
///
/// The model is deliberately coarse: for each decision tuple we evaluate
/// the dominant matmul (attention projection) + FFN matmul + norm +
/// softmax and sum the roofline latencies, then derive memory / SMEM.
///
/// This is *the* linearisation Gemini's review references — it's what makes
/// the ILP objective a simple table lookup.
pub fn build_lut(shape: &LayerShape, gpu: &GpuSpec, axes: &LutAxes) -> LayerCostLut {
    let mut entries = Vec::with_capacity(
        axes.head_counts.len() * axes.ffn_widths.len() * axes.csha_levels.len() * axes.adapter_ranks.len(),
    );
    let bs = shape.batch * shape.seq;
    for &heads in &axes.head_counts {
        let d_out_attn = heads * shape.head_dim;
        for &ffn in &axes.ffn_widths {
            for &csha in &axes.csha_levels {
                for &rank in &axes.adapter_ranks {
                    entries.push(evaluate_single(
                        shape, gpu, heads, d_out_attn, ffn, csha, rank, bs,
                    ));
                }
            }
        }
    }
    LayerCostLut {
        axes_head_counts: axes.head_counts.clone(),
        axes_ffn_widths: axes.ffn_widths.clone(),
        axes_csha_levels: axes.csha_levels.clone(),
        axes_adapter_ranks: axes.adapter_ranks.clone(),
        entries,
    }
}

/// Latency (μs) of a pretend matmul with the given flop/byte footprint.
fn latency_us(flops: u64, bytes: u64, gpu: &GpuSpec, dtype_bytes: usize) -> f64 {
    let peak_flops = gpu.peak_tflops(dtype_bytes) * 1e12;
    let peak_bw = gpu.peak_bandwidth_gbs * 1e9;
    let compute_s = flops as f64 / peak_flops.max(1.0);
    let memory_s = bytes as f64 / peak_bw.max(1.0);
    compute_s.max(memory_s) * 1e6 // seconds → μs
}

/// SMEM required at the given CSHA level.  Level 1 = small norm-prologue
/// scratch; Level 2 = producer/consumer tile pair; Level 3 = full block
/// co-scheduling.
fn smem_required(shape: &LayerShape, heads: u64, csha: u8) -> u64 {
    let bytes = shape.dtype_bytes;
    let tile = 64 * shape.head_dim * bytes;
    match csha {
        0 => 0,
        1 => 4 * 1024 + tile,               // norm scratch + one tile
        2 => 2 * tile + heads * tile,        // producer + consumer + weight tiles
        3 => 4 * tile + heads * tile + 32768, // plus FFN scratch
        _ => 0,
    }
}

fn evaluate_single(
    shape: &LayerShape,
    gpu: &GpuSpec,
    heads: u64,
    d_out_attn: u64,
    ffn: u64,
    csha: u8,
    rank: u64,
    bs: u64,
) -> LayerCostEntry {
    let dtype = shape.dtype_bytes as usize;

    // Attention projections: 3× matmul [BS, d_model] × [d_model, d_out_attn].
    let (proj_flops, proj_br, proj_bw) = matmul_cost(bs, shape.d_model, d_out_attn, shape.dtype_bytes);
    let proj_us = 3.0 * latency_us(proj_flops, proj_br + proj_bw, gpu, dtype);

    // FFN: two matmuls [BS, d_model] × [d_model, ffn] + [BS, ffn] × [ffn, d_model].
    let (ffn1_flops, ffn1_br, ffn1_bw) = matmul_cost(bs, shape.d_model, ffn, shape.dtype_bytes);
    let (ffn2_flops, ffn2_br, ffn2_bw) = matmul_cost(bs, ffn, shape.d_model, shape.dtype_bytes);
    let ffn_us = latency_us(ffn1_flops, ffn1_br + ffn1_bw, gpu, dtype)
        + latency_us(ffn2_flops, ffn2_br + ffn2_bw, gpu, dtype);

    // Norm + softmax.
    let (nrm_f, nrm_r, nrm_w) = rmsnorm_cost(shape.batch, shape.seq, shape.d_model, shape.dtype_bytes);
    let (sm_f, sm_r, sm_w) = softmax_cost(shape.batch * heads, shape.seq * shape.seq, shape.dtype_bytes);
    let norm_us = latency_us(nrm_f, nrm_r + nrm_w, gpu, dtype);
    let softmax_us = latency_us(sm_f, sm_r + sm_w, gpu, dtype);

    // CSHA saves 20 % / 45 % / 65 % of boundary traffic at levels 1/2/3.
    let csha_bonus = match csha {
        0 => 0.0,
        1 => 0.20,
        2 => 0.45,
        3 => 0.65,
        _ => 0.0,
    };
    let forward_us = (1.0 - csha_bonus) * (proj_us + ffn_us + norm_us + softmax_us);

    // Adapters add ~rank · d_model · 2 flops per token; tiny compared to main matmuls
    // unless rank is large.
    let adapter_us = (rank as f64) * (bs as f64) * (shape.d_model as f64) * 2.0
        / gpu.peak_tflops(dtype).max(1.0)
        / 1e12
        * 1e6;

    // Backward is ~2× forward for a standard dense layer, minus a smaller
    // CSHA bonus because backward fusion is less aggressive.
    let backward_us = 2.0 * forward_us * (1.0 - csha_bonus * 0.7) + adapter_us;

    // Parameters: attention 4·d_model·d_out_attn (Q/K/V/O) + FFN 3·d_model·ffn
    // (SwiGLU gate+up+down).
    let param_bytes = (4 * shape.d_model * d_out_attn + 3 * shape.d_model * ffn) * shape.dtype_bytes
        + (rank * shape.d_model + rank * d_out_attn) * shape.dtype_bytes;

    let activation_bytes = bs * shape.d_model * shape.dtype_bytes * 4;

    let smem_bytes = smem_required(shape, heads, csha);
    let smem_budget = (gpu.l1_cache_kb as u64) * 1024;
    let feasible = smem_bytes <= smem_budget && heads > 0 && ffn > 0;

    let ai = arithmetic_intensity(proj_flops, proj_br, proj_bw);
    let classification = classify_op(ai, gpu.crossover(dtype));

    LayerCostEntry {
        forward_us: forward_us + adapter_us,
        backward_us,
        param_bytes,
        activation_bytes,
        smem_bytes,
        feasible,
        classification,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs::{find_gpu, GPU_DATABASE};

    fn h100() -> &'static GpuSpec {
        find_gpu("H100")
            .or_else(|| find_gpu("h100"))
            .unwrap_or(&GPU_DATABASE[0])
    }

    fn nslcoder_shape() -> LayerShape {
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
    fn lut_has_correct_size() {
        let axes = LutAxes::default();
        let lut = build_lut(&nslcoder_shape(), h100(), &axes);
        let expected = axes.head_counts.len()
            * axes.ffn_widths.len()
            * axes.csha_levels.len()
            * axes.adapter_ranks.len();
        assert_eq!(lut.num_entries(), expected);
    }

    #[test]
    fn lut_lookup_returns_entries() {
        let lut = build_lut(&nslcoder_shape(), h100(), &LutAxes::default());
        let e = lut.get(8, 1408, 2, 4).expect("entry exists");
        assert!(e.forward_us > 0.0);
        assert!(e.param_bytes > 0);
    }

    #[test]
    fn csha_level_reduces_forward_time() {
        let lut = build_lut(&nslcoder_shape(), h100(), &LutAxes::default());
        let l0 = lut.get(8, 1408, 0, 0).unwrap();
        let l2 = lut.get(8, 1408, 2, 0).unwrap();
        assert!(l2.forward_us < l0.forward_us);
    }

    #[test]
    fn higher_adapter_rank_adds_cost() {
        let lut = build_lut(&nslcoder_shape(), h100(), &LutAxes::default());
        let r0 = lut.get(8, 1408, 0, 0).unwrap();
        let r16 = lut.get(8, 1408, 0, 16).unwrap();
        assert!(r16.forward_us >= r0.forward_us);
    }

    #[test]
    fn argmin_prefers_feasible_low_cost() {
        let lut = build_lut(&nslcoder_shape(), h100(), &LutAxes::default());
        let (_, _, _, _, best) = lut.argmin_feasible().expect("at least one feasible");
        for e in &lut.entries {
            if e.feasible {
                assert!(best.total_us() <= e.total_us() + 1e-9);
            }
        }
    }

    #[test]
    fn infeasible_csha_marked() {
        // Force a huge head count to exceed SMEM; should report infeasible
        // at high CSHA levels.
        let shape = LayerShape {
            batch: 1,
            seq: 4096,
            d_model: 16384,
            head_dim: 128,
            n_kv_heads: 16,
            dtype_bytes: 2,
        };
        let lut = build_lut(
            &shape,
            h100(),
            &LutAxes {
                head_counts: vec![128],
                ffn_widths: vec![65536],
                csha_levels: vec![3],
                adapter_ranks: vec![0],
            },
        );
        // CSHA-L3 with a 16 k d_model must blow the SMEM budget.
        assert!(!lut.entries[0].feasible);
    }

    #[test]
    fn lut_index_roundtrips() {
        let lut = build_lut(&nslcoder_shape(), h100(), &LutAxes::default());
        for &h in &lut.axes_head_counts {
            for &f in &lut.axes_ffn_widths {
                for &c in &lut.axes_csha_levels {
                    for &r in &lut.axes_adapter_ranks {
                        assert!(lut.get(h, f, c, r).is_some());
                    }
                }
            }
        }
        assert!(lut.get(99, 1408, 0, 0).is_none()); // outside axis
    }
}
