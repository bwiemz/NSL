//! CEP — the cost estimate ("oracle").
//!
//! Takes a [`ModelSpec`] (compact transformer description) and a target GPU
//! and returns a [`CompilationProfile`]: parameter count, FLOPs, memory
//! traffic and a roofline latency for ONE forward pass of `batch` sequences
//! of `max_seq` tokens. It is closed-form arithmetic on the spec: nothing is
//! compiled, and the fusion events and kernel-launch counts come from a fixed
//! template, not from the fusion pass. Every figure is an estimate.
//!
//! One model is used throughout. Attention is costed as the fused
//! (FlashAttention-style) kernel the template credits: the `Q·Kᵀ` and
//! `A·V` products and the softmax are counted as FLOPs, and the kernel reads
//! Q, K and V and writes its output without storing the `S×S` score matrix.
//! Latency is `max(compute, memory)` per layer plus the target's launch
//! overhead for each kernel launch.

use serde::Serialize;

use crate::cost_model::{
    BoundClassification, arithmetic_intensity, classify_op, embedding_cost, layernorm_cost,
    matmul_cost, rmsnorm_cost, softmax_cost,
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
    /// Whether the output head reuses the embedding matrix
    /// (`x @ embed.transpose(0, 1)`), as the shipped NSL models do. A tied
    /// head adds no parameters; an untied one adds `vocab × d_model`.
    pub tied_lm_head: bool,
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
            tied_lm_head: true,
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

    /// FFN weight matrices per block: SwiGLU has three (gate, up, down).
    pub fn ffn_matmuls(&self) -> u64 {
        match self.activation {
            Activation::SwiGlu => 3,
            _ => 2,
        }
    }

    /// Parameters of one norm layer: RMSNorm has a scale, LayerNorm a scale
    /// and a bias.
    pub fn norm_params(&self) -> u64 {
        match self.norm {
            NormType::RmsNorm => self.d_model as u64,
            NormType::LayerNorm => 2 * self.d_model as u64,
        }
    }

    /// Parameters of block `i`: Q `[d, H·hd]`, K and V `[d, KV·hd]`,
    /// O `[H·hd, d]`, the FFN matrices and the two norms.
    pub fn layer_param_count(&self, i: usize) -> u64 {
        let d = self.d_model as u64;
        let hd = self.head_dim[i] as u64;
        let nh = self.n_heads[i] as u64;
        let nkv = self.n_kv_heads[i] as u64;
        let ff = self.d_ff[i] as u64;
        let attn = d * nh * hd + 2 * d * nkv * hd + nh * hd * d;
        let ffn = self.ffn_matmuls() * d * ff;
        attn + ffn + 2 * self.norm_params()
    }

    /// Parameters outside the blocks that belong to the output side: the
    /// final norm, and the head matrix unless it is tied to the embedding.
    pub fn head_param_count(&self) -> u64 {
        let untied = if self.tied_lm_head {
            0
        } else {
            self.vocab as u64 * self.d_model as u64
        };
        self.norm_params() + untied
    }

    /// Total parameter count: embedding, blocks, final norm, and the output
    /// head when it is not tied to the embedding.
    pub fn param_count(&self) -> u64 {
        let embed = self.vocab as u64 * self.d_model as u64;
        let blocks: u64 = (0..self.n_layers as usize)
            .map(|i| self.layer_param_count(i))
            .sum();
        embed + blocks + self.head_param_count()
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
    /// Parameter bytes plus a nominal code allowance (20 KB per block, 60 KB
    /// fixed). Not a linker output.
    pub binary_size_bytes: u64,
    /// Kernel launches of one forward under the fixed fusion template
    /// (including embedding and output head).
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

/// Seconds lost to launching `launches` kernels on `gpu`, in microseconds.
fn launch_overhead_us(launches: u32, gpu: &GpuSpec) -> f64 {
    launches as f64 * gpu.kernel_launch_overhead_ns as f64 / 1e3
}

/// Roofline latency (μs): the slower of compute and memory traffic, plus the
/// launch overhead of the layer's kernels.
fn latency_us(flops: u64, bytes: u64, launches: u32, gpu: &GpuSpec, dtype_bytes: usize) -> f64 {
    let peak_flops = gpu.peak_tflops(dtype_bytes) * 1e12;
    let peak_bw = gpu.peak_bandwidth_gbs * 1e9;
    let compute_s = flops as f64 / peak_flops.max(1.0);
    let memory_s = bytes as f64 / peak_bw.max(1.0);
    compute_s.max(memory_s) * 1e6 + launch_overhead_us(launches, gpu)
}

/// Upper bound on the roofline latency (μs): compute and memory traffic are
/// assumed not to overlap (sum instead of max), plus launch overhead. This is
/// not a hardware worst-case execution time: there is no occupancy term and
/// no statistical inflation, and the tree has no certified WCET analysis.
fn worst_case_latency_us(
    flops: u64,
    bytes: u64,
    launches: u32,
    gpu: &GpuSpec,
    dtype_bytes: usize,
) -> f64 {
    let peak_flops = gpu.peak_tflops(dtype_bytes) * 1e12;
    let peak_bw = gpu.peak_bandwidth_gbs * 1e9;
    let compute_s = flops as f64 / peak_flops.max(1.0);
    let memory_s = bytes as f64 / peak_bw.max(1.0);
    (compute_s + memory_s) * 1e6 + launch_overhead_us(launches, gpu)
}

/// Parameter bytes plus a nominal code allowance: 20 KB per block, 10 KB for
/// embedding and head, 50 KB of runtime glue. The allowances are round
/// numbers, not measurements; the parameters dominate for any real model.
fn binary_size_bytes_estimate(spec: &ModelSpec, param_bytes: u64) -> u64 {
    let code_per_layer: u64 = 20_000;
    let code_embed_lm: u64 = 10_000;
    let code_runtime: u64 = 50_000;
    param_bytes
        .saturating_add(code_per_layer.saturating_mul(spec.n_layers as u64))
        .saturating_add(code_embed_lm)
        .saturating_add(code_runtime)
}

/// FLOPs, bytes read and bytes written by one norm layer over `[B, S, d]`.
fn norm_cost(spec: &ModelSpec) -> (u64, u64, u64) {
    let (b, s, d, w) = (
        spec.batch as u64,
        spec.max_seq as u64,
        spec.d_model as u64,
        spec.dtype_bytes as u64,
    );
    match spec.norm {
        NormType::RmsNorm => rmsnorm_cost(b, s, d, w),
        NormType::LayerNorm => layernorm_cost(b, s, d, w),
    }
}

/// Kernel launches of one block under the fixed fusion template: each norm
/// is fused into the projections it feeds, RoPE into the Q/K projections,
/// softmax into the attention kernel, and one residual add into its producer.
fn block_kernel_launches(spec: &ModelSpec) -> u32 {
    let swiglu = matches!(spec.activation, Activation::SwiGlu);
    let n_ffn_matmuls: u32 = if swiglu { 3 } else { 2 };
    let n_ffn_activation: u32 = if swiglu { 0 } else { 1 };
    let mut launches: u32 = 2  // norms (pre-attn, pre-ffn)
        + 3  // QKV projections
        + 1  // RoPE
        + 1  // attention kernel
        + 1  // output projection
        + 1  // attention residual add
        + n_ffn_matmuls
        + n_ffn_activation
        + 1; // FFN residual add
    launches -= 4; // NormIntoMatmul, MatmulIntoRope, SoftmaxIntoMatmul, ResidualAdd
    if swiglu {
        launches -= 1; // SwigluGate
    }
    launches.max(5)
}

fn layer_profile(spec: &ModelSpec, gpu: &GpuSpec, layer: usize) -> LayerProfile {
    let dtype = spec.dtype_bytes as usize;
    let w = spec.dtype_bytes as u64;
    let b = spec.batch as u64;
    let s = spec.max_seq as u64;
    let bs = b * s;
    let d = spec.d_model as u64;
    let nh = spec.n_heads[layer] as u64;
    let nkv = spec.n_kv_heads[layer] as u64;
    let hd = spec.head_dim[layer] as u64;
    let ff = spec.d_ff[layer] as u64;

    // Projections.
    let (q_f, q_r, q_w) = matmul_cost(bs, d, nh * hd, w);
    let (k_f, k_r, k_w) = matmul_cost(bs, d, nkv * hd, w);
    let (v_f, v_r, v_w) = matmul_cost(bs, d, nkv * hd, w);
    let (o_f, o_r, o_w) = matmul_cost(bs, nh * hd, d, w);

    // Attention core, as one fused kernel: per head, `Q·Kᵀ` (S×S scores from
    // head_dim-wide rows), a softmax over the S×S scores, and `A·V`. It reads
    // Q, K and V and writes the per-head output; the scores stay on chip.
    let scores_f = 2 * b * nh * s * s * hd;
    let (softmax_f, _, _) = softmax_cost(b * nh, s * s, w);
    let av_f = 2 * b * nh * s * s * hd;
    let core_f = scores_f + softmax_f + av_f;
    let core_r = bs * (nh + 2 * nkv) * hd * w;
    let core_w = bs * nh * hd * w;

    // FFN.
    let ffn_matmuls = spec.ffn_matmuls();
    let (g_f, g_r, g_w) = matmul_cost(bs, d, ff, w);
    let (down_f, down_r, down_w) = matmul_cost(bs, ff, d, w);
    let ffn_f = (ffn_matmuls - 1) * g_f + down_f;
    let ffn_r = (ffn_matmuls - 1) * g_r + down_r;
    let ffn_w = (ffn_matmuls - 1) * g_w + down_w;

    let (n_f, n_r, n_w) = norm_cost(spec);

    let flops = q_f + k_f + v_f + o_f + core_f + ffn_f + 2 * n_f;
    let unfused_bytes = q_r
        + q_w
        + k_r
        + k_w
        + v_r
        + v_w
        + o_r
        + o_w
        + core_r
        + core_w
        + ffn_r
        + ffn_w
        + 2 * (n_r + n_w);
    // Fused intermediates never make the round trip to memory (one write and
    // one read each): the norm output, the RoPE input and one residual sum,
    // each `[B, S, d]`, and for SwiGLU the gate product, `[B, S, d_ff]`. The
    // softmax fusion needs no entry here: the attention core above was never
    // charged for a stored score matrix.
    let bsd_bytes = bs * d * w;
    let mut savings = 6 * bsd_bytes;
    if matches!(spec.activation, Activation::SwiGlu) {
        savings += 2 * bs * ff * w;
    }
    let hbm = unfused_bytes.saturating_sub(savings);

    // Widest set of tensors alive at once inside the block: the residual and
    // its normed copy, with either Q, K, V and the attention output, or the
    // FFN's hidden activations.
    let attn_live = bs * (2 * d + (2 * nh + 2 * nkv) * hd) * w;
    let ffn_live = bs * (2 * d + (ffn_matmuls - 1) * ff) * w;
    let activation_bytes = attn_live.max(ffn_live);

    let param_bytes = spec.layer_param_count(layer) * w;

    let ai = arithmetic_intensity(flops, hbm, 0);
    let ridge = gpu.crossover(dtype);
    let classification = classify_op(ai, ridge);
    let compute_bound = matches!(classification, BoundClassification::ComputeBound);
    let kernel_launches = block_kernel_launches(spec);
    let estimated_us = latency_us(flops, hbm, kernel_launches, gpu, dtype);
    let wcet_us = worst_case_latency_us(flops, hbm, kernel_launches, gpu, dtype);

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
    let estimated_us = latency_us(f, bytes, 1, gpu, spec.dtype_bytes as usize);
    let wcet_us = worst_case_latency_us(f, bytes, 1, gpu, spec.dtype_bytes as usize);
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

/// The output side: the final norm and the `[B·S, d] × [d, vocab]` head
/// matmul. The head's matrix is counted as a parameter only when it is not
/// the embedding.
fn lm_head_profile(spec: &ModelSpec, gpu: &GpuSpec) -> LayerProfile {
    let bs = (spec.batch as u64) * (spec.max_seq as u64);
    let (mf, mr, mw) = matmul_cost(
        bs,
        spec.d_model as u64,
        spec.vocab as u64,
        spec.dtype_bytes as u64,
    );
    let (nf, nr, nw) = norm_cost(spec);
    // The final norm is fused into the head matmul like every other norm in
    // the template, so its output makes no round trip and it adds no launch.
    let fused = 2 * bs * spec.d_model as u64 * spec.dtype_bytes as u64;
    let (f, r, w) = (
        mf + nf,
        (mr + nr).saturating_sub(fused / 2),
        (mw + nw).saturating_sub(fused / 2),
    );
    let bytes = r + w;
    let launches = 1;
    let estimated_us = latency_us(f, bytes, launches, gpu, spec.dtype_bytes as usize);
    let wcet_us = worst_case_latency_us(f, bytes, launches, gpu, spec.dtype_bytes as usize);
    let ai = arithmetic_intensity(f, r, w);
    let ridge = gpu.crossover(spec.dtype_bytes as usize);
    let classification = classify_op(ai, ridge);
    LayerProfile {
        layer_index: u32::MAX - 1,
        flops: f,
        hbm_bytes: bytes,
        activation_bytes: bs * (spec.vocab as u64 + spec.d_model as u64) * spec.dtype_bytes as u64,
        param_bytes: spec.head_param_count() * spec.dtype_bytes as u64,
        estimated_us,
        arithmetic_intensity: ai,
        compute_bound: matches!(classification, BoundClassification::ComputeBound),
        classification,
        kernel_launches: launches,
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

/// Estimate one forward pass of `spec` on `gpu`.
///
/// Pure (no global state, no I/O): two calls with the same inputs return the
/// same profile bit-for-bit.
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
    // Peak memory of one inference forward: all parameters plus the widest
    // set of live activations. No KV cache and no training state.
    let peak_act: u64 = per_layer
        .iter()
        .map(|l| l.activation_bytes)
        .max()
        .unwrap_or(0);
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
    use crate::gpu_specs::find_gpu;

    fn h100() -> &'static GpuSpec {
        let gpu = find_gpu("H100-SXM").expect("H100-SXM is in the GPU database");
        // The latency literals below are worked out for these three figures.
        assert_eq!(gpu.peak_fp32_tflops, 67.0);
        assert_eq!(gpu.peak_bandwidth_gbs, 3350.0);
        assert_eq!(gpu.kernel_launch_overhead_ns, 5000);
        gpu
    }

    /// The shipped 50M-parameter model's shape (`models/coder50m/model.nsl`).
    fn nslcoder_50m() -> ModelSpec {
        ModelSpec::uniform(512, 8, 8, 4, 64, 1408, 49152)
    }

    /// d=8, one layer, 4 query heads in 2 key/value groups, head_dim 2,
    /// d_ff 16, vocab 10, 4 tokens, batch 1, 4-byte values, SwiGLU, RMSNorm,
    /// tied head. Small enough to cost by hand; see the test below.
    fn tiny() -> ModelSpec {
        let mut s = ModelSpec::uniform(8, 1, 4, 2, 2, 16, 10);
        s.max_seq = 4;
        s.dtype_bytes = 4;
        s
    }

    /// Every figure of the estimate for `tiny()`, worked out by hand.
    ///
    /// Block FLOPs (a matmul `[m,k]x[k,n]` costs `2mkn`):
    ///   Q 2·4·8·8 = 512; K and V 2·4·8·4 = 256 each; O 512;
    ///   attention core, per head `S×S` scores from 2-wide rows, a softmax
    ///   (7 per score) and `A·V`: 2·4·16·2 + 7·4·16 + 2·4·16·2 = 960;
    ///   FFN gate and up 2·4·8·16 = 1024 each, down 1024;
    ///   two RMSNorms at 5 per element: 2·160.               total 5888
    /// Block bytes (inputs and weights read, output written, 4 bytes each):
    ///   Q 384+128; K and V 256+64 each; O 384+128; core reads Q, K, V
    ///   (4·(4+2·2)·2·4 = 256) and writes 128; FFN 2·(640+256) + 768+128;
    ///   norms 2·(160+128)                                    = 5312
    ///   less the fused intermediates, 6·128 + 2·256 = 1280   → 4032
    /// Widest live set: residual, normed copy, gate and up: 4·(16+32)·4 = 768.
    /// Block parameters: 64 + 2·32 + 64 + 3·128 + 2·8 = 592.
    /// Embedding: 256 bytes moved, 80 parameters. Head: 2·4·8·10 + 160 = 800
    /// FLOPs, 448+160+160+128−256 = 640 bytes, 8 parameters (final norm; the
    /// matrix is the embedding).
    /// Latency on an H100 at 67 TFLOPS / 3350 GB/s / 5 µs per launch: every
    /// part is memory-bound, so bytes/3.35e12 s plus launches:
    ///   block 4032/3.35e6 µs + 8·5; embedding 256/3.35e6 + 5; head
    ///   640/3.35e6 + 5.
    #[test]
    fn tiny_model_matches_the_hand_costing() {
        let p = evaluate(&tiny(), h100()).unwrap();
        let block = p.per_layer.iter().find(|l| l.layer_index == 0).unwrap();
        assert_eq!(block.flops, 5888);
        assert_eq!(block.hbm_bytes, 4032);
        assert_eq!(block.activation_bytes, 768);
        assert_eq!(block.param_bytes, 592 * 4);
        assert_eq!(block.kernel_launches, 8);
        assert!(
            (block.estimated_us - 40.001_203_582_089_55).abs() < 1e-9,
            "{}",
            block.estimated_us
        );

        assert_eq!(p.total_flops, 6688);
        assert_eq!(p.total_hbm_bytes, 4928);
        assert_eq!(p.param_bytes, 680 * 4);
        assert_eq!(p.peak_memory_bytes, 680 * 4 + 768);
        assert_eq!(p.kernel_launches, 10);
        assert!(
            (p.estimated_latency_us - 50.001_471_044_776_125).abs() < 1e-9,
            "{}",
            p.estimated_latency_us
        );
        // Compute and memory summed instead of the larger taken:
        // 6688/67e6 µs more than the roofline figure.
        let extra = 6688.0 / 67e6;
        assert!(
            (p.wcet_us - p.estimated_latency_us - extra).abs() < 1e-9,
            "{}",
            p.wcet_us
        );
    }

    /// The attention core's `Q·Kᵀ` and `A·V` products scale with the square
    /// of the sequence length. At 4 tokens the block costs 5888 FLOPs (above);
    /// at 8 tokens every linear term doubles (2·(5888 − 960) = 9856) and the
    /// core quadruples (4·960 = 3840): 13696. A model that leaves the two
    /// products out grows by less.
    #[test]
    fn attention_core_grows_with_the_square_of_the_sequence() {
        let mut s = tiny();
        s.max_seq = 8;
        let p = evaluate(&s, h100()).unwrap();
        let block = p.per_layer.iter().find(|l| l.layer_index == 0).unwrap();
        assert_eq!(block.flops, 13696);
    }

    /// No `S×S` score matrix is charged to memory traffic: the bytes of a
    /// block are linear in the sequence length. 4032 at 4 tokens; at 8 tokens
    /// every activation term doubles and the weights read (4·(64+32+32+64 +
    /// 3·128) + 2·8·4 = 2368 bytes) do not: 2·(4032 − 2368) + 2368 = 5696.
    #[test]
    fn block_memory_traffic_is_linear_in_the_sequence() {
        let mut s = tiny();
        s.max_seq = 8;
        let p = evaluate(&s, h100()).unwrap();
        let block = p.per_layer.iter().find(|l| l.layer_index == 0).unwrap();
        assert_eq!(block.hbm_bytes, 5696);
    }

    /// 48,767,488: the embedding 49152·512 = 25,165,824 (the output head
    /// reuses it), eight blocks of 786,432 (attention) + 2,162,688 (FFN) +
    /// 1,024 (two norms) = 2,950,144, and the final norm's 512.
    #[test]
    fn tied_head_is_counted_once() {
        let spec = nslcoder_50m();
        assert_eq!(spec.layer_param_count(0), 2_950_144);
        assert_eq!(spec.param_count(), 48_767_488);
        let p = evaluate(&spec, h100()).unwrap();
        assert_eq!(p.param_bytes, 48_767_488 * spec.dtype_bytes as u64);
    }

    #[test]
    fn untied_head_adds_one_vocab_by_d_model_matrix() {
        let mut spec = nslcoder_50m();
        spec.tied_lm_head = false;
        assert_eq!(spec.param_count(), 48_767_488 + 25_165_824);
    }

    /// LayerNorm has a scale and a bias: 512 more parameters per norm than
    /// RMSNorm, for 17 norms.
    #[test]
    fn layernorm_has_twice_the_norm_parameters() {
        let mut spec = nslcoder_50m();
        spec.norm = NormType::LayerNorm;
        assert_eq!(spec.param_count(), 48_767_488 + 17 * 512);
        // And costs 8 FLOPs per element where RMSNorm costs 5.
        let rms = evaluate(&nslcoder_50m(), h100()).unwrap();
        let ln = evaluate(&spec, h100()).unwrap();
        assert_eq!(ln.total_flops - rms.total_flops, 17 * 3 * 1024 * 512);
    }

    #[test]
    fn profile_has_embedding_layers_and_lm_head_plus_blocks() {
        let spec = nslcoder_50m();
        let p = evaluate(&spec, h100()).unwrap();
        // 8 blocks + embedding + lm_head = 10 entries.
        assert_eq!(p.per_layer.len(), 10);
        // 8 launches per block under the template, plus embedding and head.
        assert_eq!(p.kernel_launches, 66);
    }

    /// Dropping one key/value group from one layer of the 50M shape (8 -> 6
    /// query heads, 4 -> 3 key/value heads) removes 2·(512·128) from Q and O
    /// and 2·(512·64) from K and V: 196,608 parameters.
    #[test]
    fn pruning_one_group_removes_its_parameters() {
        let baseline = evaluate(&nslcoder_50m(), h100()).unwrap();
        let mut pruned = nslcoder_50m();
        pruned.n_heads[4] = 6;
        pruned.n_kv_heads[4] = 3;
        let p = evaluate(&pruned, h100()).unwrap();
        assert_eq!(baseline.param_bytes - p.param_bytes, 196_608 * 2);
        assert!(p.total_flops < baseline.total_flops);
        assert!(p.estimated_latency_us < baseline.estimated_latency_us);
    }

    #[test]
    fn invalid_spec_returns_err() {
        let mut spec = nslcoder_50m();
        spec.n_heads[0] = 7; // 7 is not a multiple of n_kv_heads=4
        let r = evaluate(&spec, h100());
        assert!(r.is_err());
    }

    #[test]
    fn fusion_events_cover_every_layer() {
        let spec = nslcoder_50m();
        let p = evaluate(&spec, h100()).unwrap();
        // Five template events per SwiGLU block.
        assert_eq!(p.fusion_events.len(), 5 * spec.n_layers as usize);
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

    /// The SwiGLU gate fusion is the one event that depends on the spec. For
    /// `tiny()` it spares the `[B, S, d_ff]` product one write and one read,
    /// 2·4·16·4 = 512 bytes; a two-matrix FFN (ReLU) has no such event, and
    /// one fewer `[4,8]x[8,16]` matmul (640 read + 256 written). So the ReLU
    /// block moves 4032 + 512 − 896 = 3648 bytes.
    #[test]
    fn swiglu_gate_fusion_saves_its_intermediate() {
        let mut spec = tiny();
        spec.activation = Activation::Relu;
        let p = evaluate(&spec, h100()).unwrap();
        let block = p.per_layer.iter().find(|l| l.layer_index == 0).unwrap();
        assert_eq!(block.hbm_bytes, 3648);
        assert_eq!(block.flops, 5888 - 1024);
        assert!(
            !p.fusion_events
                .contains(&FusionEvent::SwigluGate { layer: 0 })
        );
    }

    #[test]
    fn deterministic_across_calls() {
        let spec = nslcoder_50m();
        let p1 = evaluate(&spec, h100()).unwrap();
        let p2 = evaluate(&spec, h100()).unwrap();
        assert_eq!(p1.total_flops, p2.total_flops);
        assert_eq!(p1.param_bytes, p2.param_bytes);
        assert_eq!(p1.fusion_events, p2.fusion_events);
        assert_eq!(
            p1.estimated_latency_us.to_bits(),
            p2.estimated_latency_us.to_bits()
        );
    }

    /// The size figure is the parameters plus 20 KB per block and 60 KB.
    #[test]
    fn size_estimate_is_parameters_plus_the_nominal_code_allowance() {
        let spec = ModelSpec::uniform(384, 4, 6, 3, 64, 1024, 4096);
        let p = evaluate(&spec, h100()).expect("eval");
        assert_eq!(p.binary_size_bytes, p.param_bytes + 4 * 20_000 + 60_000);
        let summed: u32 = p.per_layer.iter().map(|l| l.kernel_launches).sum();
        assert_eq!(p.kernel_launches, summed);
        assert_eq!(p.kernel_launches, 4 * 8 + 2);
    }
}
