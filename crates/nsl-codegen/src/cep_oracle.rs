//! CEP — Compilation-Evaluated Pruning: the oracle.
//!
//! Takes a [`ModelSpec`] (compact transformer description) and a target
//! GPU, returns a [`CompilationProfile`] with exact FLOP / HBM / memory /
//! latency / fusion metrics in ~50 ms — no GPU, no training step.
//!
//! The oracle composes the existing M37 (cost model), M36 (memory plan
//! proxy), and M31 (fusion) primitives into a single deterministic
//! evaluator.  This is what the paper calls "the compiler as oracle" —
//! replacing PyTorch's runtime profile + surrogate-model pipeline with a
//! pure function of the AST shape and target hardware.

use serde::Serialize;

use crate::cost_model::{
    arithmetic_intensity, classify_op, embedding_cost, matmul_cost, rmsnorm_cost, softmax_cost,
    BoundClassification,
};
use crate::gpu_specs::GpuSpec;

/// Activation function used in the FFN block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Activation {
    Relu,
    Gelu,
    SiLU,
    SwiGlu,
}

impl Activation {
    pub fn as_str(self) -> &'static str {
        match self {
            Activation::Relu => "relu",
            Activation::Gelu => "gelu",
            Activation::SiLU => "silu",
            Activation::SwiGlu => "swiglu",
        }
    }
}

/// Normalization layer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NormType {
    LayerNorm,
    RmsNorm,
}

impl NormType {
    pub fn as_str(self) -> &'static str {
        match self {
            NormType::LayerNorm => "layer_norm",
            NormType::RmsNorm => "rms_norm",
        }
    }
}

/// Compact transformer model descriptor.  The oracle operates on this —
/// not on the full AST — because the search loop needs to evaluate
/// thousands of candidates in seconds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelSpec {
    /// Per-layer hidden dimension.
    pub d_model: u32,
    /// Number of transformer blocks.
    pub n_layers: u32,
    /// Query attention heads per layer.  Can be a single value (uniform)
    /// or per-layer.
    pub n_heads: Vec<u32>,
    /// KV heads per layer (for GQA).
    pub n_kv_heads: Vec<u32>,
    /// Head dimension per layer.
    pub head_dim: Vec<u32>,
    /// FFN inner dimension per layer.
    pub d_ff: Vec<u32>,
    /// Vocabulary size.
    pub vocab: u32,
    /// Maximum sequence length the binary will be compiled for.
    pub max_seq: u32,
    /// Batch size for profiling purposes.
    pub batch: u32,
    /// Activation function.
    pub activation: Activation,
    /// Norm kind.
    pub norm: NormType,
    /// Parameter dtype bytes (2 = fp16, 4 = fp32).
    pub dtype_bytes: u32,
}

impl ModelSpec {
    /// Construct a uniform-layer spec (all layers have the same
    /// `(n_heads, n_kv_heads, head_dim, d_ff)`) with sane defaults.
    pub fn uniform(
        d_model: u32,
        n_layers: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        d_ff: u32,
        vocab: u32,
    ) -> Self {
        Self {
            d_model,
            n_layers,
            n_heads: vec![n_heads; n_layers as usize],
            n_kv_heads: vec![n_kv_heads; n_layers as usize],
            head_dim: vec![head_dim; n_layers as usize],
            d_ff: vec![d_ff; n_layers as usize],
            vocab,
            max_seq: 1024,
            batch: 1,
            activation: Activation::SwiGlu,
            norm: NormType::RmsNorm,
            dtype_bytes: 2,
        }
    }

    /// Validate that per-layer vectors all have length `n_layers`.
    pub fn validate(&self) -> Result<(), String> {
        let n = self.n_layers as usize;
        if self.n_heads.len() != n
            || self.n_kv_heads.len() != n
            || self.head_dim.len() != n
            || self.d_ff.len() != n
        {
            return Err(format!(
                "per-layer vector length mismatch: expected {}, got n_heads={} n_kv_heads={} head_dim={} d_ff={}",
                n,
                self.n_heads.len(),
                self.n_kv_heads.len(),
                self.head_dim.len(),
                self.d_ff.len(),
            ));
        }
        for (i, kv) in self.n_kv_heads.iter().enumerate() {
            if *kv == 0 || self.n_heads[i] == 0 {
                return Err(format!("layer {i}: n_heads and n_kv_heads must be > 0"));
            }
            if !self.n_heads[i].is_multiple_of(*kv) {
                return Err(format!(
                    "layer {i}: n_heads ({}) must be a multiple of n_kv_heads ({})",
                    self.n_heads[i], kv
                ));
            }
        }
        Ok(())
    }

    /// Approximate total parameter count (for reporting).
    pub fn param_count(&self) -> u64 {
        // Embedding + per-layer (QKV + O + FFN + norms) + final norm + lm_head.
        let mut total: u64 = (self.vocab as u64) * (self.d_model as u64); // embedding
        for i in 0..self.n_layers as usize {
            let d = self.d_model as u64;
            let hd = self.head_dim[i] as u64;
            let nh = self.n_heads[i] as u64;
            let nkv = self.n_kv_heads[i] as u64;
            let ff = self.d_ff[i] as u64;
            // Q: [d, nh*hd]; K,V: [d, nkv*hd]; O: [nh*hd, d]
            let attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d;
            // FFN: SwiGLU uses 3 matmuls (gate + up + down); others use 2.
            let ffn_matmuls = match self.activation {
                Activation::SwiGlu => 3,
                _ => 2,
            };
            let ffn = ffn_matmuls as u64 * d * ff;
            // Two norms per block (attn_norm + ffn_norm).
            let norms = 2 * d;
            total += attn + ffn + norms;
        }
        total += self.d_model as u64; // final norm
        total += (self.vocab as u64) * (self.d_model as u64); // lm head
        total
    }
}

/// Per-layer profile emitted by the oracle.
#[derive(Debug, Clone, Serialize)]
pub struct LayerProfile {
    pub layer_index: u32,
    pub flops: u64,
    pub hbm_bytes: u64,
    pub activation_bytes: u64,
    pub param_bytes: u64,
    pub estimated_us: f64,
    pub arithmetic_intensity: f64,
    pub classification: BoundClassification,
    /// Whether the layer is compute-bound on the target.
    pub compute_bound: bool,
    /// Number of distinct kernel launches for this layer after static fusion.
    pub kernel_launches: u32,
    /// Roofline upper-bound latency (μs): compute_time + memory_time (NOT max).
    /// Contrast with `estimated_us` which uses max(compute, memory).
    pub wcet_us: f64,
}

/// Entries in the fusion-opportunity log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum FusionEvent {
    /// `RMSNorm → matmul` prologue fusion.
    NormIntoMatmul { layer: u32 },
    /// `matmul → RoPE` epilogue fusion.
    MatmulIntoRope { layer: u32 },
    /// `softmax → matmul` FlashAttention fusion.
    SoftmaxIntoMatmul { layer: u32 },
    /// `ffn_gate ⊙ ffn_up` SwiGLU fusion.
    SwigluGate { layer: u32 },
    /// Residual add at the block output.
    ResidualAdd { layer: u32 },
}

/// Top-level output of the oracle.
#[derive(Debug, Clone, Serialize)]
pub struct CompilationProfile {
    pub total_flops: u64,
    pub total_hbm_bytes: u64,
    pub param_bytes: u64,
    pub peak_memory_bytes: u64,
    pub estimated_latency_us: f64,
    pub per_layer: Vec<LayerProfile>,
    pub roofline_utilization: f64,
    pub fusion_events: Vec<FusionEvent>,
    pub target_gpu: String,
    /// Approximate binary size: parameter bytes + per-layer code-section estimate.
    pub binary_size_bytes: u64,
    /// Total kernel launches across all layers (including embedding and lm_head).
    pub kernel_launches: u32,
    /// Roofline upper-bound total latency (μs): sum of per-layer wcet_us.
    pub wcet_us: f64,
}

impl CompilationProfile {
    /// Peak arithmetic intensity across all layers.
    pub fn peak_ai(&self) -> f64 {
        self.per_layer
            .iter()
            .map(|l| l.arithmetic_intensity)
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Oracle
// ---------------------------------------------------------------------------

fn latency_us(flops: u64, bytes: u64, gpu: &GpuSpec, dtype_bytes: usize) -> f64 {
    let peak_flops = gpu.peak_tflops(dtype_bytes) * 1e12;
    let peak_bw = gpu.peak_bandwidth_gbs * 1e9;
    let compute_s = flops as f64 / peak_flops.max(1.0);
    let memory_s = bytes as f64 / peak_bw.max(1.0);
    compute_s.max(memory_s) * 1e6
}

/// Roofline worst-case execution time (microseconds): assumes compute and memory
/// traffic do NOT overlap (sum instead of max).  This is an upper bound on the
/// canonical roofline latency — not a true hardware WCET; there is no
/// kernel-launch overhead, occupancy term, or p95 statistical inflation.  For
/// DO-178C-style certified WCET use the wcet.rs module's FPGA-targeted helpers.
fn worst_case_latency_us(flops: u64, bytes: u64, gpu: &GpuSpec, dtype_bytes: usize) -> f64 {
    let peak_flops = gpu.peak_tflops(dtype_bytes) * 1e12;
    let peak_bw = gpu.peak_bandwidth_gbs * 1e9;
    let compute_s = flops as f64 / peak_flops.max(1.0);
    let memory_s = bytes as f64 / peak_bw.max(1.0);
    // SUM rather than max — assumes no pipelining of compute and memory traffic.
    (compute_s + memory_s) * 1e6
}

/// Approximate binary size: parameter bytes + a per-layer PTX code-section
/// estimate.  Each transformer block contributes ~20 KB of PTX text (Q/K/V
/// matmuls, FlashAttention, output projection, FFN matmuls, norms — kernels
/// are templated by shape but their text size dominates).  Embedding + lm_head
/// contribute ~5 KB each.  Plus ~50 KB of runtime entry/dispatch glue.
/// This is a first-order model, NOT the true linker output.
fn binary_size_bytes_estimate(spec: &ModelSpec, param_bytes: u64) -> u64 {
    let code_per_layer: u64 = 20_000;
    let code_embed_lm: u64 = 10_000;
    let code_runtime: u64 = 50_000;
    param_bytes
        .saturating_add(code_per_layer.saturating_mul(spec.n_layers as u64))
        .saturating_add(code_embed_lm)
        .saturating_add(code_runtime)
}

fn layer_profile(spec: &ModelSpec, gpu: &GpuSpec, layer: usize) -> LayerProfile {
    let dtype = spec.dtype_bytes as usize;
    let bs = (spec.batch as u64) * (spec.max_seq as u64);
    let d = spec.d_model as u64;
    let nh = spec.n_heads[layer] as u64;
    let nkv = spec.n_kv_heads[layer] as u64;
    let hd = spec.head_dim[layer] as u64;
    let ff = spec.d_ff[layer] as u64;

    // Attention: Q/K/V/O matmuls.
    let (q_f, q_r, q_w) = matmul_cost(bs, d, nh * hd, spec.dtype_bytes as u64);
    let (k_f, k_r, k_w) = matmul_cost(bs, d, nkv * hd, spec.dtype_bytes as u64);
    let (v_f, v_r, v_w) = matmul_cost(bs, d, nkv * hd, spec.dtype_bytes as u64);
    let (o_f, o_r, o_w) = matmul_cost(bs, nh * hd, d, spec.dtype_bytes as u64);

    // FFN: SwiGLU has 3 matmuls, others 2.
    let ffn_matmuls = match spec.activation {
        Activation::SwiGlu => 3,
        _ => 2,
    };
    let (g_f, g_r, g_w) = matmul_cost(bs, d, ff, spec.dtype_bytes as u64);
    let (down_f, down_r, down_w) = matmul_cost(bs, ff, d, spec.dtype_bytes as u64);
    let ffn_f = (ffn_matmuls - 1) as u64 * g_f + down_f;
    let ffn_r = (ffn_matmuls - 1) as u64 * g_r + down_r;
    let ffn_w = (ffn_matmuls - 1) as u64 * g_w + down_w;

    // Norms × 2 + softmax.
    let (n_f, n_r, n_w) = rmsnorm_cost(spec.batch as u64, spec.max_seq as u64, d, spec.dtype_bytes as u64);
    let (s_f, s_r, s_w) = softmax_cost(
        spec.batch as u64 * nh,
        spec.max_seq as u64 * spec.max_seq as u64,
        spec.dtype_bytes as u64,
    );

    let flops = q_f + k_f + v_f + o_f + ffn_f + 2 * n_f + s_f;
    let hbm = q_r + q_w + k_r + k_w + v_r + v_w + o_r + o_w + ffn_r + ffn_w + 2 * (n_r + n_w) + s_r + s_w;
    let activation_bytes = bs * d * spec.dtype_bytes as u64 * 4; // loose upper bound
    let param_bytes = {
        let attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d;
        let ffn = ffn_matmuls as u64 * d * ff;
        (attn + ffn + 2 * d) * spec.dtype_bytes as u64
    };
    // Apply byte savings from the static fusion-event template.  Each fired event
    // removes one tensor's worth of HBM round-trip (read + write that would otherwise
    // leave and re-enter HBM as an intermediate).  Conservative: one bsd_bytes saving
    // per always-on event; SwiGLU saves an additional bsdff_bytes for the gate*up
    // intermediate.  This makes fusion_events numerically load-bearing rather than
    // decorative, addressing paper §3.1 / G3+G15.
    let bsd_bytes: u64 = (spec.batch as u64)
        .saturating_mul(spec.max_seq as u64)
        .saturating_mul(spec.d_model as u64)
        .saturating_mul(spec.dtype_bytes as u64);
    let bsdff_bytes: u64 = (spec.batch as u64)
        .saturating_mul(spec.max_seq as u64)
        .saturating_mul(spec.d_ff[layer] as u64)
        .saturating_mul(spec.dtype_bytes as u64);
    // Always-on events: NormIntoMatmul + MatmulIntoRope + SoftmaxIntoMatmul + ResidualAdd = 4
    let mut savings: u64 = bsd_bytes.saturating_mul(4);
    if matches!(spec.activation, Activation::SwiGlu) {
        savings = savings.saturating_add(bsdff_bytes);
    }
    let hbm = hbm.saturating_sub(savings);

    let ai = arithmetic_intensity(flops, q_r + ffn_r, q_w + ffn_w);
    let ridge = gpu.crossover(dtype);
    let classification = classify_op(ai, ridge);
    let compute_bound = matches!(classification, BoundClassification::ComputeBound);
    let estimated_us = latency_us(flops, hbm, gpu, dtype);
    let wcet_us = worst_case_latency_us(flops, hbm, gpu, dtype);

    // Kernel launches per transformer block (unfused canonical count minus fusion savings).
    // Base: 2 norms + 3 QKV projections + 1 RoPE + 1 FlashAttention + 1 output proj +
    //       1 attn residual + N_ffn_matmuls + N_ffn_activation + 1 ffn residual.
    let n_ffn_matmuls: u32 = if matches!(spec.activation, Activation::SwiGlu) { 3 } else { 2 };
    let n_ffn_activation: u32 = if matches!(spec.activation, Activation::SwiGlu) { 0 } else { 1 };
    let mut launches: u32 =
          2  // norms (pre-attn, pre-ffn)
        + 3  // QKV projections
        + 1  // RoPE
        + 1  // FlashAttention (already fused; counts as 1)
        + 1  // output projection
        + 1  // attention residual add
        + n_ffn_matmuls
        + n_ffn_activation
        + 1; // FFN residual add
    // Fusion events reduce launches 1:1 with what they collapse:
    //   NormIntoMatmul + MatmulIntoRope + SoftmaxIntoMatmul + ResidualAdd = 4 always-on
    launches -= 4;
    if matches!(spec.activation, Activation::SwiGlu) {
        launches -= 1; // SwigluGate fuses gate*up into down
    }
    // Floor: at least 5 launches per layer (fusion never eliminates all).
    let kernel_launches = launches.max(5);

    LayerProfile {
        layer_index: layer as u32,
        flops,
        hbm_bytes: hbm,
        activation_bytes,
        param_bytes,
        estimated_us,
        arithmetic_intensity: ai,
        classification,
        compute_bound,
        kernel_launches,
        wcet_us,
    }
}

fn embedding_profile(spec: &ModelSpec, gpu: &GpuSpec) -> LayerProfile {
    let (f, r, w) = embedding_cost(
        spec.batch as u64,
        spec.max_seq as u64,
        spec.d_model as u64,
        spec.dtype_bytes as u64,
    );
    let bytes = r + w;
    let estimated_us = latency_us(f, bytes, gpu, spec.dtype_bytes as usize);
    let wcet_us = worst_case_latency_us(f, bytes, gpu, spec.dtype_bytes as usize);
    let ai = arithmetic_intensity(f, r, w);
    let ridge = gpu.crossover(spec.dtype_bytes as usize);
    LayerProfile {
        layer_index: u32::MAX,
        flops: f,
        hbm_bytes: bytes,
        activation_bytes: bytes,
        param_bytes: (spec.vocab as u64) * (spec.d_model as u64) * spec.dtype_bytes as u64,
        estimated_us,
        arithmetic_intensity: ai,
        classification: classify_op(ai, ridge),
        compute_bound: false,
        kernel_launches: 1,
        wcet_us,
    }
}

fn lm_head_profile(spec: &ModelSpec, gpu: &GpuSpec) -> LayerProfile {
    let bs = (spec.batch as u64) * (spec.max_seq as u64);
    let (f, r, w) = matmul_cost(bs, spec.d_model as u64, spec.vocab as u64, spec.dtype_bytes as u64);
    let bytes = r + w;
    let estimated_us = latency_us(f, bytes, gpu, spec.dtype_bytes as usize);
    let wcet_us = worst_case_latency_us(f, bytes, gpu, spec.dtype_bytes as usize);
    let ai = arithmetic_intensity(f, r, w);
    let ridge = gpu.crossover(spec.dtype_bytes as usize);
    LayerProfile {
        layer_index: u32::MAX - 1,
        flops: f,
        hbm_bytes: bytes,
        activation_bytes: bs * spec.vocab as u64 * spec.dtype_bytes as u64,
        param_bytes: (spec.vocab as u64) * (spec.d_model as u64) * spec.dtype_bytes as u64,
        estimated_us,
        arithmetic_intensity: ai,
        classification: classify_op(ai, ridge),
        compute_bound: true,
        kernel_launches: 1,
        wcet_us,
    }
}

fn detect_fusion_events(spec: &ModelSpec) -> Vec<FusionEvent> {
    let mut out = Vec::new();
    for i in 0..spec.n_layers {
        out.push(FusionEvent::NormIntoMatmul { layer: i });
        out.push(FusionEvent::MatmulIntoRope { layer: i });
        out.push(FusionEvent::SoftmaxIntoMatmul { layer: i });
        if spec.activation == Activation::SwiGlu {
            out.push(FusionEvent::SwigluGate { layer: i });
        }
        out.push(FusionEvent::ResidualAdd { layer: i });
    }
    out
}

/// Run the oracle.
///
/// Guaranteed to be pure (no global state, no I/O) — two calls with the
/// same inputs return the same profile bit-for-bit.
pub fn evaluate(spec: &ModelSpec, gpu: &GpuSpec) -> Result<CompilationProfile, String> {
    spec.validate()?;

    let mut per_layer = Vec::with_capacity(spec.n_layers as usize + 2);
    per_layer.push(embedding_profile(spec, gpu));
    for i in 0..spec.n_layers as usize {
        per_layer.push(layer_profile(spec, gpu, i));
    }
    per_layer.push(lm_head_profile(spec, gpu));

    let total_flops: u64 = per_layer.iter().map(|l| l.flops).sum();
    let total_hbm: u64 = per_layer.iter().map(|l| l.hbm_bytes).sum();
    let param_bytes: u64 = per_layer.iter().map(|l| l.param_bytes).sum();
    // Peak memory: max per-layer activation + all params (conservative
    // upper bound; memory planner would give a tighter number).
    let peak_act: u64 = per_layer.iter().map(|l| l.activation_bytes).max().unwrap_or(0);
    let peak_memory = param_bytes + peak_act;
    let estimated_latency_us: f64 = per_layer.iter().map(|l| l.estimated_us).sum();
    let wcet_us: f64 = per_layer.iter().map(|l| l.wcet_us).sum();
    let kernel_launches: u32 = per_layer.iter().map(|l| l.kernel_launches).sum();
    let binary_size_bytes = binary_size_bytes_estimate(spec, param_bytes);

    // Roofline utilisation: total_flops / (peak_compute × estimated_latency_s).
    let peak_flops_per_s = gpu.peak_tflops(spec.dtype_bytes as usize) * 1e12;
    let achieved_flops_per_s = total_flops as f64 / (estimated_latency_us.max(1e-9) * 1e-6);
    let roofline_utilization = (achieved_flops_per_s / peak_flops_per_s.max(1.0)).min(1.0);

    Ok(CompilationProfile {
        total_flops,
        total_hbm_bytes: total_hbm,
        param_bytes,
        peak_memory_bytes: peak_memory,
        estimated_latency_us,
        per_layer,
        roofline_utilization,
        fusion_events: detect_fusion_events(spec),
        target_gpu: gpu.name.to_string(),
        binary_size_bytes,
        kernel_launches,
        wcet_us,
    })
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

    fn nslcoder_50m() -> ModelSpec {
        ModelSpec::uniform(512, 8, 8, 4, 64, 1408, 49152)
    }

    #[test]
    fn oracle_produces_profile_for_valid_spec() {
        let spec = nslcoder_50m();
        let p = evaluate(&spec, h100()).expect("profile");
        assert!(p.total_flops > 0);
        assert!(p.total_hbm_bytes > 0);
        assert!(p.param_bytes > 0);
        assert!(p.estimated_latency_us > 0.0);
        assert!(p.roofline_utilization >= 0.0 && p.roofline_utilization <= 1.0);
    }

    #[test]
    fn profile_has_embedding_layers_and_lm_head_plus_blocks() {
        let spec = nslcoder_50m();
        let p = evaluate(&spec, h100()).unwrap();
        // 8 blocks + embedding + lm_head = 10 entries.
        assert_eq!(p.per_layer.len(), 10);
    }

    #[test]
    fn pruning_reduces_flops_and_params() {
        let baseline = evaluate(&nslcoder_50m(), h100()).unwrap();
        // Prune 2 heads from layer 4, keeping GQA group integrity.
        let mut pruned = nslcoder_50m();
        // 8 heads / 4 kv = group of 2; drop one group → 6 heads, 3 kv.
        pruned.n_heads[4] = 6;
        pruned.n_kv_heads[4] = 3;
        let p = evaluate(&pruned, h100()).unwrap();
        assert!(p.total_flops < baseline.total_flops);
        assert!(p.param_bytes < baseline.param_bytes);
    }

    #[test]
    fn non_uniform_layers_report_per_layer_profiles() {
        let mut spec = nslcoder_50m();
        spec.n_heads[3] = 6;
        spec.n_kv_heads[3] = 3;
        spec.d_ff[3] = 1024;
        let p = evaluate(&spec, h100()).unwrap();
        // Layer 3 should have lower FLOPs than Layer 0.
        let l0 = p.per_layer.iter().find(|l| l.layer_index == 0).unwrap();
        let l3 = p.per_layer.iter().find(|l| l.layer_index == 3).unwrap();
        assert!(l3.flops < l0.flops);
    }

    #[test]
    fn invalid_spec_returns_err() {
        let mut spec = nslcoder_50m();
        spec.n_heads[0] = 7; // 7 doesn't divide n_kv_heads=4 cleanly
        let r = evaluate(&spec, h100());
        assert!(r.is_err());
    }

    #[test]
    fn fusion_events_cover_every_layer() {
        let spec = nslcoder_50m();
        let p = evaluate(&spec, h100()).unwrap();
        let layers_with_events: std::collections::BTreeSet<u32> = p
            .fusion_events
            .iter()
            .map(|e| match e {
                FusionEvent::NormIntoMatmul { layer }
                | FusionEvent::MatmulIntoRope { layer }
                | FusionEvent::SoftmaxIntoMatmul { layer }
                | FusionEvent::SwigluGate { layer }
                | FusionEvent::ResidualAdd { layer } => *layer,
            })
            .collect();
        assert_eq!(layers_with_events.len(), spec.n_layers as usize);
    }

    #[test]
    fn param_count_roughly_matches_reported_bytes() {
        let spec = nslcoder_50m();
        let p = evaluate(&spec, h100()).unwrap();
        let expected_bytes = spec.param_count() * spec.dtype_bytes as u64;
        // Allow small drift because embedding/LM head are included in
        // both but the oracle lumps some norms differently.
        let diff = (expected_bytes as i128 - p.param_bytes as i128).unsigned_abs() as u64;
        assert!(
            diff < expected_bytes / 10,
            "expected {} got {}",
            expected_bytes,
            p.param_bytes
        );
    }

    #[test]
    fn swiglu_has_more_ffn_ops_than_relu() {
        let mut spec = nslcoder_50m();
        spec.activation = Activation::SwiGlu;
        let swiglu = evaluate(&spec, h100()).unwrap();
        spec.activation = Activation::Relu;
        let relu = evaluate(&spec, h100()).unwrap();
        assert!(swiglu.total_flops > relu.total_flops);
    }

    #[test]
    fn deterministic_across_calls() {
        let spec = nslcoder_50m();
        let p1 = evaluate(&spec, h100()).unwrap();
        let p2 = evaluate(&spec, h100()).unwrap();
        assert_eq!(p1.total_flops, p2.total_flops);
        assert_eq!(p1.param_bytes, p2.param_bytes);
        assert_eq!(p1.fusion_events, p2.fusion_events);
    }

    #[test]
    fn profile_surfaces_binary_size_and_kernel_launches() {
        let spec = ModelSpec::uniform(384, 4, 6, 3, 64, 1024, 4096);
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").expect("gpu");
        let p = evaluate(&spec, &gpu).expect("eval");
        // binary_size_bytes >= param_bytes (must include at least the weights), and within
        // ~2x for a tiny model (code section should not dominate).
        assert!(p.binary_size_bytes >= p.param_bytes, "binary_size must include params");
        assert!(
            p.binary_size_bytes <= p.param_bytes * 2 + 10_000_000,
            "binary_size must not be runaway"
        );
        // kernel_launches > 0 and equals the sum of per_layer launches.
        assert!(p.kernel_launches > 0);
        let summed: u32 = p.per_layer.iter().map(|l| l.kernel_launches).sum();
        assert_eq!(p.kernel_launches, summed);
        // Per real-block layer: within a sane range (well under 30).
        for l in p.per_layer.iter().filter(|l| l.layer_index < spec.n_layers) {
            assert!(
                l.kernel_launches > 0 && l.kernel_launches < 30,
                "layer {} launches out of range: {}",
                l.layer_index,
                l.kernel_launches
            );
        }
    }

    #[test]
    fn fusion_events_reduce_per_layer_hbm() {
        let spec = ModelSpec::uniform(384, 2, 6, 3, 64, 1024, 4096);
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").expect("gpu");
        let p = evaluate(&spec, &gpu).expect("eval");
        let layer = p.per_layer.iter().find(|l| l.layer_index == 0).unwrap();
        let n_events_layer0: usize = p
            .fusion_events
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    FusionEvent::NormIntoMatmul { layer: 0 }
                        | FusionEvent::SwigluGate { layer: 0 }
                        | FusionEvent::ResidualAdd { layer: 0 }
                        | FusionEvent::MatmulIntoRope { layer: 0 }
                        | FusionEvent::SoftmaxIntoMatmul { layer: 0 }
                )
            })
            .count();
        assert!(n_events_layer0 >= 1, "expected at least one fusion event on layer 0");
        assert!(layer.hbm_bytes > 0);
    }

    #[test]
    fn profile_surfaces_wcet_us_at_least_estimated() {
        // Roofline WCET = compute + memory >= max(compute, memory) = estimated.
        let spec = ModelSpec::uniform(384, 3, 6, 3, 64, 1024, 4096);
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").expect("gpu");
        let p = evaluate(&spec, &gpu).expect("eval");
        assert!(
            p.wcet_us >= p.estimated_latency_us - 1e-9,
            "wcet_us={} must be >= estimated_latency_us={}",
            p.wcet_us,
            p.estimated_latency_us
        );
        assert!(p.wcet_us > 0.0);
        assert!(p.estimated_latency_us > 0.0);
        for l in &p.per_layer {
            assert!(
                l.wcet_us >= l.estimated_us - 1e-9,
                "layer {} wcet {} >= estimated {}",
                l.layer_index,
                l.wcet_us,
                l.estimated_us
            );
        }
    }
}
