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
    /// Model dtype size (bytes).  Needed to convert an entry's `param_bytes`
    /// into a parameter *element* count for the precision-aware optimizer term
    /// ([`optimizer_us`]).
    pub dtype_bytes: u64,
    /// Target HBM bandwidth (GB/s).  The optimizer term divides bytes-moved by
    /// this (same divisor as [`comm_optim_us`]) so it is commensurate, in μs,
    /// with the forward/backward latencies the ILP already sums.
    pub peak_bandwidth_gbs: f64,
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
        dtype_bytes: shape.dtype_bytes,
        peak_bandwidth_gbs: gpu.peak_bandwidth_gbs,
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

/// Communication + optimizer-step latency (μs) for one layer under ZeRO
/// sharding.
///
/// These two cost components are *shard-dependent*, so they are deliberately
/// kept out of the shard-agnostic [`LayerCostEntry`] LUT and computed here
/// once the inter-layer DP (or greedy re-cost) has fixed a layer's shard
/// factors.
///
/// **Deterministic fallback heuristic** — the repo has no measured
/// collective-latency oracle, so this models well-known first-order costs:
///  * ZeRO comm ≈ `2·param_bytes·(S-1)/S` bytes per step (all-gather params +
///    reduce-scatter grads) over the inter-GPU interconnect; `shard_comm <= 1`
///    means no collective, so comm is exactly `0`.
///  * Adam's optimizer step is memory-bound: it reads+writes the parameter
///    plus two moment buffers ≈ `6·param_bytes / shard_optim` bytes over HBM
///    bandwidth (`gpu.peak_bandwidth_gbs`).
///
/// Returns `(comm_us, optim_us)`.
pub fn comm_optim_us(
    param_bytes: u64,
    shard_comm: u32,
    shard_optim: u32,
    interconnect_gbs: f64,
    gpu: &GpuSpec,
) -> (f64, f64) {
    let comm_us = if shard_comm <= 1 {
        0.0
    } else {
        let s = shard_comm as f64;
        let bytes = 2.0 * param_bytes as f64 * (s - 1.0) / s;
        bytes / (interconnect_gbs.max(1.0) * 1e9) * 1e6
    };
    let optim_bytes = 6.0 * param_bytes as f64 / (shard_optim.max(1) as f64);
    let optim_us = optim_bytes / (gpu.peak_bandwidth_gbs.max(1.0) * 1e9) * 1e6;
    (comm_us, optim_us)
}

/// Communication + optimizer-step latency (μs) for a layer under an explicit
/// ZeRO **stage** (audit gap #6), where parameters/gradients/optimizer state
/// carry independent shard factors.
///
/// Higher stages move more data over the interconnect but need less HBM for the
/// optimizer step — the memory/comm trade-off the Level-1 DP weighs. Each tensor
/// sharded beyond DDP needs one collective proportional to `2·param·(N-1)/N`:
/// gradient reduce-scatter at stage ≥ 2 and parameter all-gather at stage 3
/// (full FSDP does both). Optimizer-state sharding adds no collective (the Adam
/// update is local once the gradient is reduced) but scales its HBM traffic by
/// `1/shard_optim`. Coarse heuristic, consistent with [`comm_optim_us`] (the
/// repo has no measured collective-latency oracle); at stage 0 it returns the
/// same *cost* `(0, 6·param/1)` as the legacy unsharded path. (The *memory*
/// model differs: [`resident_training_bytes_sharded`] additionally counts
/// gradient memory, so a staged stage-0 layer is `4·param` where the legacy
/// [`resident_training_bytes`] is `3·param` — the staged path is the more
/// physically faithful accounting.)
///
/// The heuristic omits DDP's own baseline gradient all-reduce, so stage 0 shows
/// zero comm; consequently stage 1 (optimizer-state sharding) strictly dominates
/// stage 0 whenever `num_gpus > 1` — the expected "ZeRO-1 is near-free" property,
/// not a modeling error.
pub fn zero_stage_costs(
    param_bytes: u64,
    shard_param: u32,
    shard_grad: u32,
    shard_optim: u32,
    interconnect_gbs: f64,
    gpu: &GpuSpec,
) -> (f64, f64) {
    let cluster = shard_param.max(shard_grad).max(shard_optim).max(1);
    let sharded_collectives = (shard_grad > 1) as u32 + (shard_param > 1) as u32;
    let comm_us = if cluster <= 1 || sharded_collectives == 0 {
        0.0
    } else {
        let s = cluster as f64;
        let bytes = 2.0 * param_bytes as f64 * (s - 1.0) / s * sharded_collectives as f64;
        bytes / (interconnect_gbs.max(1.0) * 1e9) * 1e6
    };
    let optim_bytes = 6.0 * param_bytes as f64 / (shard_optim.max(1) as f64);
    let optim_us = optim_bytes / (gpu.peak_bandwidth_gbs.max(1.0) * 1e9) * 1e6;
    (comm_us, optim_us)
}

/// Optimizer-step latency (μs) for one layer — **precision- and FASE-aware**.
///
/// Adam's update is HBM-bandwidth-bound: each step streams the master
/// parameter (read + write) and the gradient (read) at the model dtype, plus
/// the two moment buffers `m`, `v` (each read + write) at their *chosen*
/// precisions `p_m`, `p_v`.  Lower moment precision ⇒ fewer bytes moved ⇒ a
/// cheaper step.
///
/// This is the cost signal the Level-2 ILP needs to trade optimizer precision
/// against numerical safety: without it the ILP's objective is independent of
/// `p_m`/`p_v`, so it has no reason to ever pick sub-fp32 moments (it just
/// keeps whichever precision it enumerates first — 32-bit).  The DP's coarser
/// [`comm_optim_us`] stays precision-blind on purpose: precision is a Level-2
/// decision, unknown when the Level-1 DP runs.
///
/// When FASE fuses the optimizer step into the backward pass, the standalone
/// gradient HBM round-trip is eliminated (the gradient is already resident from
/// backward), so that read is dropped.
///
/// `param_bytes` / `dtype_bytes` describe the layer's parameters at the model
/// dtype; `peak_bandwidth_gbs` is the same HBM-bandwidth divisor
/// [`comm_optim_us`] uses, so the result is commensurate, in μs, with the
/// forward/backward latencies the ILP already sums.
///
/// Simplification: the master parameter and gradient are modeled at the *model*
/// dtype (`param_bytes`), not a separate fp32 master copy.  For sub-fp32 models
/// this under-counts master-copy traffic, but it matches the existing
/// [`comm_optim_us`] convention and preserves the term's purpose — the relative
/// ordering across `m_bits`/`v_bits` (lower precision is cheaper).  A separate
/// `master_dtype_bytes` is a future refinement.
pub fn optimizer_us(
    param_bytes: u64,
    dtype_bytes: u64,
    m_bits: u8,
    v_bits: u8,
    fase_fused: bool,
    peak_bandwidth_gbs: f64,
) -> f64 {
    let param_rw = 2.0 * param_bytes as f64; // master read + write
    let grad_read = if fase_fused { 0.0 } else { param_bytes as f64 };
    // m and v are each read + written once.  Their combined resident size comes
    // from the *same* helper the memory model uses (`moment_state_bytes`), so the
    // latency and memory views of the optimizer state can never disagree on how
    // large the moments are (audit gap #3).
    let moment_rw = 2.0 * moment_state_bytes(param_bytes, dtype_bytes, m_bits, v_bits) as f64;
    let total_bytes = param_rw + grad_read + moment_rw;
    total_bytes / (peak_bandwidth_gbs.max(1.0) * 1e9) * 1e6
}

/// Combined resident size (bytes) of the two Adam moment buffers `m`, `v` for a
/// parameter tensor, at their chosen per-moment precisions.
///
/// `param_bytes` / `dtype_bytes` gives the parameter *element* count; each
/// moment stores one value per element, sized by its bit-width.  This is the
/// single definition of "how big are the optimizer moments" shared by the
/// latency model ([`optimizer_us`]) and the resident-memory model
/// ([`resident_training_bytes`]).  `param_bytes` is an element count times
/// `dtype_bytes`, so the integer division is exact.
pub fn moment_state_bytes(param_bytes: u64, dtype_bytes: u64, m_bits: u8, v_bits: u8) -> u64 {
    let elements = param_bytes / dtype_bytes.max(1);
    let m_bytes = elements.saturating_mul(m_bits as u64) / 8;
    let v_bytes = elements.saturating_mul(v_bits as u64) / 8;
    m_bytes.saturating_add(v_bytes)
}

/// Extra resident bytes from the FASE **Deferred** fused optimizer step that
/// [`resident_training_bytes`] does not count. `crate::fase_memory` models
/// these correctly (`params + accumulator + one-layer-grad + opt_state + peak
/// activation`) but is disconnected from the cost model, so a budget compared
/// against the bare `param + m + v + activation` charge under-counts.
///
/// Two surfaces are added:
///   * **`m_partial`** — the windowed gradient-mean accumulator materialised
///     only when `grad_accumulation > 1`. It is param-sized and **always f32**,
///     regardless of the chosen moment precision: it accumulates raw gradients,
///     so it does NOT quantize with `m_bits`/`v_bits`.
///   * **one live parameter gradient** — the single gradient held during the
///     deferred per-param epilogue, at the model/compute dtype (`param_bytes`).
///
/// There is deliberately **no `v_partial`** term: FASE accumulates only the
/// first-moment (gradient) window, so the Deferred optimizer adds exactly these
/// **two** surfaces (making 4 live surfaces total with the base
/// `param + m + v + activation` — NOT 5). Sized per-layer against the layer's
/// own `param_bytes`, which is a conservative over-estimate of the true
/// single-tensor `one-live-param` grad (`fase_memory` uses `max_param_bytes`);
/// over-estimating is the safe direction for a hard budget gate.
pub fn fase_deferred_extra_bytes(param_bytes: u64, dtype_bytes: u64) -> u64 {
    let elements = param_bytes / dtype_bytes.max(1);
    let m_partial_bytes = elements.saturating_mul(4); // f32 window accumulator
    let one_live_grad = param_bytes; // model-dtype single live gradient
    m_partial_bytes.saturating_add(one_live_grad)
}

/// Resident training memory for one layer under ZeRO sharding — the single
/// formula shared by the Level-1 DP ([`crate::wggo_dp`]) and the Level-2 ILP
/// ([`crate::wggo_ilp`]) so the two levels can never diverge on what "resident
/// memory" means (audit gap #3):
///
/// ```text
/// (param + optimizer_state) / shard + activation
/// ```
///
/// ZeRO shards the parameters and optimizer state (the Adam moments) `shard`
/// ways; activations are not sharded.  `shard == 0` is treated as `1`.
///
/// The two levels supply `optimizer_state_bytes` differently *by design* — the
/// coarse DP has no precision variable and sizes moments at parameter precision
/// (`2·param`), while the ILP, which decides moment precision (gap #2), sizes
/// them exactly via [`moment_state_bytes`].  The **formula** is identical; only
/// the fidelity of its optimizer-state input differs.
pub fn resident_training_bytes(
    param_bytes: u64,
    optimizer_state_bytes: u64,
    activation_bytes: u64,
    shard: u32,
) -> u64 {
    let sharded = param_bytes.saturating_add(optimizer_state_bytes) / (shard.max(1) as u64);
    sharded.saturating_add(activation_bytes)
}

/// Per-GPU resident training memory with each ZeRO component sharded
/// *independently* (audit gap #6): parameters, gradients, and optimizer state
/// carry their own shard factors, while activations are never sharded.
///
/// ```text
/// param/shard_param + grad/shard_grad + optimizer_state/shard_optim + activation
/// ```
///
/// This is the faithful counterpart to [`resident_training_bytes`] (which shards
/// `param + optimizer_state` by a single factor and omits gradient memory).
/// The Level-1 DP drives it from a [`crate::wggo_dp::ZeroStage`] whose nested
/// chain supplies the three factors, so each stage is a genuinely distinct
/// memory point.  `shard == 0` is treated as `1`.
pub fn resident_training_bytes_sharded(
    param_bytes: u64,
    grad_bytes: u64,
    optimizer_state_bytes: u64,
    activation_bytes: u64,
    shard_param: u32,
    shard_grad: u32,
    shard_optim: u32,
) -> u64 {
    let p = param_bytes / (shard_param.max(1) as u64);
    let g = grad_bytes / (shard_grad.max(1) as u64);
    let o = optimizer_state_bytes / (shard_optim.max(1) as u64);
    p.saturating_add(g).saturating_add(o).saturating_add(activation_bytes)
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
    fn comm_is_zero_when_unsharded() {
        let (comm, _optim) = comm_optim_us(1_000_000, 1, 1, 300.0, h100());
        assert_eq!(comm, 0.0);
    }

    #[test]
    fn comm_grows_with_shard_and_bytes() {
        let (c2, _) = comm_optim_us(1_000_000, 2, 2, 300.0, h100());
        let (c8, _) = comm_optim_us(1_000_000, 8, 8, 300.0, h100());
        let (c8_big, _) = comm_optim_us(4_000_000, 8, 8, 300.0, h100());
        assert!(c2 > 0.0);
        assert!(c8 > c2);
        assert!(c8_big > c8);
    }

    #[test]
    fn optim_time_decreases_with_optim_shard() {
        let (_, o1) = comm_optim_us(1_000_000, 1, 1, 300.0, h100());
        let (_, o8) = comm_optim_us(1_000_000, 1, 8, 300.0, h100());
        assert!(o1 > 0.0);
        assert!(o8 < o1);
    }

    #[test]
    fn optimizer_us_decreases_with_lower_moment_precision() {
        // The core gap-#2 signal: cheaper moments ⇒ cheaper step, so the ILP
        // gets a reason to lower precision when numerically safe.
        let o32 = optimizer_us(1_000_000, 2, 32, 32, false, 3350.0);
        let o16 = optimizer_us(1_000_000, 2, 16, 16, false, 3350.0);
        let o8 = optimizer_us(1_000_000, 2, 8, 8, false, 3350.0);
        assert!(o32 > o16, "32-bit moments must cost more than 16-bit");
        assert!(o16 > o8, "16-bit moments must cost more than 8-bit");
        assert!(o8 > 0.0);
    }

    #[test]
    fn optimizer_us_fase_fusion_is_cheaper() {
        // Fusing the step into backward drops the standalone gradient read.
        let fused = optimizer_us(1_000_000, 2, 16, 16, true, 3350.0);
        let separate = optimizer_us(1_000_000, 2, 16, 16, false, 3350.0);
        assert!(fused < separate);
    }

    #[test]
    fn optimizer_us_scales_with_param_bytes_and_is_deterministic() {
        let small = optimizer_us(1_000_000, 2, 16, 16, false, 3350.0);
        let big = optimizer_us(4_000_000, 2, 16, 16, false, 3350.0);
        assert!(big > small);
        assert_eq!(
            optimizer_us(1_234_567, 2, 8, 16, false, 3350.0),
            optimizer_us(1_234_567, 2, 8, 16, false, 3350.0)
        );
    }

    #[test]
    fn comm_optim_is_deterministic() {
        let a = comm_optim_us(1_234_567, 4, 4, 300.0, h100());
        let b = comm_optim_us(1_234_567, 4, 4, 300.0, h100());
        assert_eq!(a, b);
    }

    #[test]
    fn moment_state_bytes_is_precision_and_dtype_aware() {
        // 1024 param bytes at 2 bytes/element ⇒ 512 parameter elements; each of
        // the two Adam moments holds one value per element, sized by its bits.
        assert_eq!(moment_state_bytes(1024, 2, 32, 32), 512 * 4 + 512 * 4); // 4096
        assert_eq!(moment_state_bytes(1024, 2, 16, 16), 512 * 2 + 512 * 2); // 2048
        assert_eq!(moment_state_bytes(1024, 2, 8, 8), 512 * 1 + 512 * 1); //   1024
        // Monotone in precision.
        assert!(moment_state_bytes(1024, 2, 32, 32) > moment_state_bytes(1024, 2, 16, 16));
        assert!(moment_state_bytes(1024, 2, 16, 16) > moment_state_bytes(1024, 2, 8, 8));
        // The two moments are sized independently (m vs v may differ).
        assert_eq!(moment_state_bytes(1024, 2, 8, 32), 512 * 1 + 512 * 4); // 2560
    }

    #[test]
    fn resident_training_bytes_shards_optimizer_not_activation() {
        // (param + optimizer_state) / shard + activation.
        assert_eq!(resident_training_bytes(1000, 2000, 500, 1), 3500);
        // Sharding divides the param+optimizer block only.
        assert_eq!(resident_training_bytes(1000, 2000, 500, 3), 3000 / 3 + 500);
        // shard == 0 is treated as 1 (never divide by zero).
        assert_eq!(resident_training_bytes(1000, 2000, 500, 0), 3500);
        // Activation is not sharded: it adds one-for-one at any shard factor.
        assert_eq!(
            resident_training_bytes(1000, 2000, 900, 4) - resident_training_bytes(1000, 2000, 500, 4),
            400
        );
    }

    #[test]
    fn resident_training_bytes_matches_legacy_dp_formula() {
        // The Level-1 DP's historical formula was `3·param/shard + activation`
        // (moments sized at parameter precision ⇒ optimizer_state = 2·param).
        // Routing it through the shared formula must be behaviour-identical,
        // including integer-division flooring.
        for &(p, a, s) in &[(1000u64, 500u64, 1u32), (1200, 100, 4), (777, 33, 8)] {
            let legacy = 3u64 * p / (s as u64) + a;
            assert_eq!(resident_training_bytes(p, 2 * p, a, s), legacy);
        }
    }

    #[test]
    fn resident_training_bytes_sharded_shards_each_component() {
        // param=1000, grad=1000, optim=2000, activation=500, N=4. The ZeRO
        // stages shard progressively more, so per-GPU memory strictly decreases.
        let r0 = resident_training_bytes_sharded(1000, 1000, 2000, 500, 1, 1, 1); // stage 0
        let r1 = resident_training_bytes_sharded(1000, 1000, 2000, 500, 1, 1, 4); // stage 1: optim
        let r2 = resident_training_bytes_sharded(1000, 1000, 2000, 500, 1, 4, 4); // stage 2: + grad
        let r3 = resident_training_bytes_sharded(1000, 1000, 2000, 500, 4, 4, 4); // stage 3: + param
        assert_eq!(r0, 1000 + 1000 + 2000 + 500); // 4500
        assert_eq!(r1, 1000 + 1000 + 500 + 500); //  3000
        assert_eq!(r2, 1000 + 250 + 500 + 500); //   2250
        assert_eq!(r3, 250 + 250 + 500 + 500); //    1500
        assert!(r0 > r1 && r1 > r2 && r2 > r3, "nested chain ⇒ strictly less memory");
        // Activations are never sharded.
        assert_eq!(
            resident_training_bytes_sharded(1000, 1000, 2000, 900, 4, 4, 4)
                - resident_training_bytes_sharded(1000, 1000, 2000, 500, 4, 4, 4),
            400
        );
        // shard == 0 is treated as 1 (never divide by zero).
        assert_eq!(resident_training_bytes_sharded(1000, 1000, 2000, 500, 0, 0, 0), 4500);
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
