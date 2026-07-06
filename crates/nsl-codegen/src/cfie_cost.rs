//! CFIE — decode latency / throughput cost model (audit gap G22).
//!
//! The paper's build report (§8) prints two estimate lines under the
//! kernel-launch count:
//!
//! ```text
//! Estimated decode latency: 82us/token (vs 210us baseline)
//! Estimated throughput (batch=32): 18,400 tok/s
//! ```
//!
//! These are **estimates from an explicit, inspectable roofline model**,
//! not measurements — and the report labels them as such.  This module
//! is that model.  It is deliberately simple: batch-1 autoregressive
//! decode is well known to be memory-bandwidth-bound plus CPU launch
//! overhead, so we model exactly those two terms and nothing else.
//!
//! # The two terms
//!
//! For one decode step producing one token, the GPU must move a fixed
//! amount of data through HBM (paper §6 lists it: read `x`, read the
//! weights, read/write the KV cache, write `x'`), and the CPU must pay
//! a fixed overhead per kernel launch:
//!
//! ```text
//! t_mem    = bytes_per_token / memory_bandwidth
//! t_launch = launches_per_token * PER_LAUNCH_US
//! latency  = t_mem + t_launch
//! ```
//!
//! `bytes_per_token` has three parts:
//!
//!   * **weight bytes** — every attention (Q/K/V/O) and FFN
//!     (gate/up/down) matrix, plus the LM head, read once per token at
//!     the weights precision.  Batch-1 decode is a sequence of
//!     matrix-*vector* products, so each weight element is touched
//!     exactly once and this term dominates.
//!   * **KV bytes** — the K and V cache read by attention.  This is
//!     sequence-length-dependent; we evaluate it at a *representative
//!     operating point* `seq = max_seq / 2` (a half-full cache) and the
//!     report SAYS so, both in a `[... @ seq=N]` tag and the assumptions
//!     sub-line.  For the CFIE path the per-token KV footprint comes
//!     from the per-layer KV-quant plan (INT8 layers move half the bytes
//!     of FP16); the baseline uses uniform FP16.
//!   * **activation bytes** — the residual-stream `x`/`x'` round trips
//!     the persistent block still pays (read + write, per layer).  A
//!     minor term, included for completeness.
//!
//! # Throughput (batched decode)
//!
//! Continuous batching amortizes the weight read across the whole batch:
//! the weights are read ONCE per decode step regardless of batch size,
//! while each sequence in the batch reads its own KV cache and moves its
//! own activations.  So for batch `B`:
//!
//! ```text
//! step_bytes = weight_bytes + B * (kv_bytes + activation_bytes)
//! t_mem_step = step_bytes / memory_bandwidth
//! step_time  = t_mem_step + t_launch     (launches are shared per step)
//! throughput = B / step_time             (B tokens produced per step)
//! ```
//!
//! Because `weight_bytes` is fixed per step, throughput grows
//! *sublinearly* with `B` — doubling the batch less-than-doubles
//! tokens/s once the per-sequence terms stop being negligible against
//! the fixed weight read.  This is the standard batched-decode roofline.
//!
//! # Honesty
//!
//! Every constant here is documented; the numbers are labeled estimates
//! in the report; and both the CFIE and baseline latencies are computed
//! by the *same* formula (they differ only in launch count and KV
//! precision), so the "vs baseline" comparison is apples-to-apples.

use crate::cfie_kv_quant::KvQuantPlan;
use crate::gpu_specs::GpuSpec;

/// CPU-side cost of a single kernel launch, in microseconds.  Paper §2:
/// "At ~5us per launch, [~500-1000 launches/token] is 2.5-5ms of pure
/// launch overhead."  We use the same 5us for BOTH paths so the CFIE-vs-
/// baseline comparison isolates the launch-*count* reduction.
pub const PER_LAUNCH_US: f64 = 5.0;

/// Inputs the cost model needs, all resolved by the time
/// `run_cfie_for_serve` has a plan (model shape, weights precision, the
/// KV-quant plan, launch counts, the GPU spec, and the batch size).
#[derive(Debug, Clone, Copy)]
pub struct CostModelInputs<'a> {
    // --- model shape (per resolved serve config) ---
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub vocab_size: u32,
    /// Bytes per weight element (2 = FP16/BF16 serving default).
    pub weight_dtype_bytes: u32,
    // --- launch counts (from the CfiePlan) ---
    pub cfie_launches_per_token: u32,
    pub baseline_launches_per_token: u32,
    // --- KV-cache precision mix ---
    /// Per-layer KV-quant plan (drives the CFIE KV byte accounting).
    pub kv_quant: &'a KvQuantPlan,
    /// Uniform-FP16 KV bytes per stored token across all layers (the
    /// baseline's KV footprint).  Falls back to a shape-derived value
    /// when the plan didn't populate it (e.g. Sampling mode).
    pub baseline_kv_bytes_per_stored_token: u64,
    // --- operating point ---
    /// Max sequence length; the KV cache is evaluated at `max_seq / 2`.
    pub max_seq: u32,
    /// Batch size for the throughput line (the scheduler's max_active).
    pub batch: u32,
    // --- hardware ---
    pub gpu: &'a GpuSpec,
}

/// The finished estimate the report renders.
#[derive(Debug, Clone)]
pub struct CfieCostEstimate {
    pub cfie_us_per_token: f64,
    pub baseline_us_per_token: f64,
    pub throughput_tok_s: f64,
    pub batch: u32,
    /// Representative KV operating point (`max_seq / 2`) baked into the
    /// numbers, surfaced in the report tag and assumptions line.
    pub kv_seq: u32,
    /// One-line human-readable summary of the model + assumptions.
    pub assumptions: String,
}

/// Per-layer weight bytes read once per decode token: attention Q/K/V/O
/// projections + FFN gate/up/down.  GQA is honoured — K and V project to
/// `n_kv_heads` not `n_heads`.
fn weight_bytes_per_token(inp: &CostModelInputs) -> u64 {
    let d_model = inp.d_model as u64;
    let q_dim = (inp.n_heads as u64) * (inp.head_dim as u64);
    let kv_dim = (inp.n_kv_heads as u64) * (inp.head_dim as u64);
    let d_ff = inp.d_ff as u64;
    let bytes = inp.weight_dtype_bytes as u64;

    // Attention: W_q [d_model x q_dim], W_k/W_v [d_model x kv_dim] each,
    // W_o [q_dim x d_model].
    let attn_elems = d_model * q_dim      // Q
        + 2 * d_model * kv_dim            // K + V
        + q_dim * d_model; // O
    // FFN (SwiGLU family): gate + up [d_model x d_ff] each, down
    // [d_ff x d_model].
    let ffn_elems = 3 * d_model * d_ff;
    let per_layer = (attn_elems + ffn_elems) * bytes;

    // LM head [vocab x d_model], read once per token (not per layer).
    let lm_head = (inp.vocab_size as u64) * d_model * bytes;

    per_layer * (inp.n_layers as u64) + lm_head
}

/// Residual-stream activation traffic per token: the persistent block
/// reads `x` and writes `x'` per layer (2 * d_model elements/layer).  A
/// minor term relative to the weight read; included for completeness.
fn activation_bytes_per_token(inp: &CostModelInputs) -> u64 {
    2 * (inp.d_model as u64) * (inp.n_layers as u64) * (inp.weight_dtype_bytes as u64)
}

/// KV bytes read per token by attention at the representative operating
/// point: `kv_seq` cached tokens times the per-stored-token KV footprint.
fn kv_bytes_per_token(bytes_per_stored_token: u64, kv_seq: u32) -> u64 {
    bytes_per_stored_token * (kv_seq as u64)
}

/// Round the bandwidth (GB/s) into bytes/us: `GB/s = 1e9 bytes / 1e6 us`
/// so bytes/us = GB/s * 1e3.
fn bandwidth_bytes_per_us(gpu: &GpuSpec) -> f64 {
    gpu.peak_bandwidth_gbs * 1_000.0
}

/// Latency for one path: memory time + launch time (microseconds).
fn latency_us(bytes_per_token: u64, launches: u32, bw_bytes_per_us: f64) -> f64 {
    let t_mem = bytes_per_token as f64 / bw_bytes_per_us;
    let t_launch = launches as f64 * PER_LAUNCH_US;
    t_mem + t_launch
}

/// Run the model and produce the report-ready estimate.
pub fn estimate(inp: &CostModelInputs) -> CfieCostEstimate {
    let kv_seq = (inp.max_seq / 2).max(1);
    let bw = bandwidth_bytes_per_us(inp.gpu);

    let weight_bytes = weight_bytes_per_token(inp);
    let activation_bytes = activation_bytes_per_token(inp);

    // CFIE path: KV footprint from the per-layer quant plan when it
    // populated a selected byte count; else fall back to uniform FP16.
    let cfie_kv_stored = if inp.kv_quant.bytes_per_token_selected > 0 {
        inp.kv_quant.bytes_per_token_selected
    } else {
        inp.baseline_kv_bytes_per_stored_token
    };
    let baseline_kv_stored = if inp.kv_quant.bytes_per_token_uniform_fp16 > 0 {
        inp.kv_quant.bytes_per_token_uniform_fp16
    } else {
        inp.baseline_kv_bytes_per_stored_token
    };

    let cfie_kv = kv_bytes_per_token(cfie_kv_stored, kv_seq);
    let baseline_kv = kv_bytes_per_token(baseline_kv_stored, kv_seq);

    let cfie_bytes = weight_bytes + cfie_kv + activation_bytes;
    let baseline_bytes = weight_bytes + baseline_kv + activation_bytes;

    let cfie_us = latency_us(cfie_bytes, inp.cfie_launches_per_token, bw);
    let baseline_us = latency_us(baseline_bytes, inp.baseline_launches_per_token, bw);

    // Throughput: weights read once per step, KV + activations per
    // sequence in the batch; launches shared per step.
    let batch = inp.batch.max(1);
    let step_bytes = weight_bytes + (batch as u64) * (cfie_kv + activation_bytes);
    let t_mem_step = step_bytes as f64 / bw;
    let t_launch = inp.cfie_launches_per_token as f64 * PER_LAUNCH_US;
    let step_time_us = t_mem_step + t_launch;
    // tokens/s = batch tokens per step / step_time (us -> s).
    let throughput_tok_s = if step_time_us > 0.0 {
        (batch as f64) / (step_time_us / 1_000_000.0)
    } else {
        0.0
    };

    let assumptions = format!(
        "bandwidth-roofline model: t_mem = bytes/token / {:.0} GB/s + t_launch = \
         launches x {:.0}us; KV cache @ seq={} (max_seq/2); weights @ {}B/elem; \
         throughput amortizes weights across batch={}",
        inp.gpu.peak_bandwidth_gbs, PER_LAUNCH_US, kv_seq, inp.weight_dtype_bytes, batch
    );

    CfieCostEstimate {
        cfie_us_per_token: cfie_us,
        baseline_us_per_token: baseline_us,
        throughput_tok_s,
        batch,
        kv_seq,
        assumptions,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_kv_quant::KvQuantPlan;
    use crate::gpu_specs::find_gpu;

    /// A KV-quant plan with explicit byte counts, bypassing the planner
    /// so the arithmetic is hand-verifiable.
    fn quant_plan(selected: u64, uniform: u64) -> KvQuantPlan {
        KvQuantPlan {
            layers: Vec::new(),
            bytes_per_token_uniform_fp16: uniform,
            bytes_per_token_selected: selected,
        }
    }

    /// Minimal hand-computable config: 1 layer, no FFN, no LM head, no
    /// activations-of-consequence, so the arithmetic is checkable by hand.
    fn tiny_inputs<'a>(
        gpu: &'a GpuSpec,
        kv: &'a KvQuantPlan,
    ) -> CostModelInputs<'a> {
        CostModelInputs {
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 64,
            d_model: 64,
            d_ff: 0,
            vocab_size: 0,
            weight_dtype_bytes: 2,
            cfie_launches_per_token: 34,
            baseline_launches_per_token: 500,
            kv_quant: kv,
            baseline_kv_bytes_per_stored_token: 0,
            max_seq: 2,
            batch: 1,
            gpu,
        }
    }

    #[test]
    fn weight_bytes_hand_computed() {
        // 1 layer, d_model=64, head_dim=64, 1 head, 1 kv head, d_ff=0,
        // vocab=0, 2 bytes/elem.
        //   attn = d_model*q_dim + 2*d_model*kv_dim + q_dim*d_model
        //        = 64*64 + 2*64*64 + 64*64 = 4*4096 = 16384 elems
        //   ffn  = 0, lm_head = 0
        //   bytes = 16384 * 2 = 32768
        let gpu = find_gpu("H100-SXM").unwrap();
        let kv = quant_plan(0, 0);
        let inp = tiny_inputs(gpu, &kv);
        assert_eq!(weight_bytes_per_token(&inp), 32_768);
    }

    #[test]
    fn activation_bytes_hand_computed() {
        // 2 * d_model * n_layers * bytes = 2 * 64 * 1 * 2 = 256.
        let gpu = find_gpu("H100-SXM").unwrap();
        let kv = quant_plan(0, 0);
        let inp = tiny_inputs(gpu, &kv);
        assert_eq!(activation_bytes_per_token(&inp), 256);
    }

    #[test]
    fn latency_is_mem_plus_launch_hand_computed() {
        // H100-SXM bandwidth = 3350 GB/s = 3_350_000 bytes/us.
        // CFIE bytes = weight(32768) + kv(selected*1) + activation(256).
        // With kv_seq = max_seq/2 = 1 and selected KV = 100 bytes:
        //   cfie_bytes = 32768 + 100 + 256 = 33124
        //   t_mem = 33124 / 3_350_000 ≈ 0.009888 us
        //   t_launch = 34 * 5 = 170 us
        //   latency ≈ 170.0099 us
        let gpu = find_gpu("H100-SXM").unwrap();
        let kv = quant_plan(100, 200);
        let inp = tiny_inputs(gpu, &kv);
        let est = estimate(&inp);
        let expected_cfie = (32_768.0 + 100.0 + 256.0) / 3_350_000.0 + 34.0 * 5.0;
        assert!(
            (est.cfie_us_per_token - expected_cfie).abs() < 1e-6,
            "cfie {} vs expected {}",
            est.cfie_us_per_token,
            expected_cfie
        );
        // Baseline: KV uniform = 200 bytes, 500 launches.
        let expected_base = (32_768.0 + 200.0 + 256.0) / 3_350_000.0 + 500.0 * 5.0;
        assert!((est.baseline_us_per_token - expected_base).abs() < 1e-6);
        assert_eq!(est.kv_seq, 1);
    }

    #[test]
    fn cfie_latency_below_baseline_when_launches_drop() {
        // The report's headline property: fewer launches + lighter KV =>
        // lower latency.
        let gpu = find_gpu("H100-SXM").unwrap();
        let kv = quant_plan(8_000, 16_000);
        let inp = tiny_inputs(gpu, &kv);
        let est = estimate(&inp);
        assert!(
            est.cfie_us_per_token < est.baseline_us_per_token,
            "cfie {} must be < baseline {}",
            est.cfie_us_per_token,
            est.baseline_us_per_token
        );
    }

    #[test]
    fn int8_kv_reduces_bytes_vs_fp16() {
        // Same shape, only KV precision differs: a plan whose selected KV
        // is half the uniform FP16 must yield a strictly lower latency
        // than one whose selected KV equals uniform FP16.
        let gpu = find_gpu("H100-SXM").unwrap();
        let heavy_seq = 1024u32;

        let int8 = quant_plan(8_000, 16_000); // half the FP16 footprint
        let fp16 = quant_plan(16_000, 16_000); // no savings

        let mut base_inp = tiny_inputs(gpu, &int8);
        base_inp.max_seq = heavy_seq * 2; // kv_seq = heavy_seq

        let est_int8 = estimate(&base_inp);
        base_inp.kv_quant = &fp16;
        let est_fp16 = estimate(&base_inp);

        assert!(
            est_int8.cfie_us_per_token < est_fp16.cfie_us_per_token,
            "INT8 KV ({}) must move fewer bytes than FP16 KV ({})",
            est_int8.cfie_us_per_token,
            est_fp16.cfie_us_per_token
        );
    }

    #[test]
    fn throughput_grows_sublinearly_with_batch() {
        // Doubling the batch must raise tokens/s but by LESS than 2x,
        // because the fixed per-step weight read does not scale with the
        // batch.  Use a realistic shape so the weight read is meaningful
        // against the per-sequence KV term.
        let gpu = find_gpu("H100-SXM").unwrap();
        let kv = quant_plan(8_000, 16_000);
        let mut inp = tiny_inputs(gpu, &kv);
        // Give the model real FFN + LM-head weight so the fixed term
        // dominates and the sublinearity is visible.
        inp.d_ff = 1408;
        inp.vocab_size = 32_000;
        inp.d_model = 512;
        inp.head_dim = 128;
        inp.n_heads = 4;
        inp.n_kv_heads = 4;
        inp.n_layers = 8;
        inp.max_seq = 2048;

        inp.batch = 16;
        let t16 = estimate(&inp).throughput_tok_s;
        inp.batch = 32;
        let t32 = estimate(&inp).throughput_tok_s;

        assert!(t32 > t16, "throughput must grow with batch: {t16} -> {t32}");
        assert!(
            t32 < 2.0 * t16,
            "throughput must grow SUBLINEARLY: {t16} -> {t32} (2x = {})",
            2.0 * t16
        );
    }

    #[test]
    fn all_outputs_positive_for_realistic_config() {
        let gpu = find_gpu("H100-SXM").unwrap();
        let kv = quant_plan(12_288, 16_384);
        let mut inp = tiny_inputs(gpu, &kv);
        inp.d_model = 512;
        inp.d_ff = 1408;
        inp.vocab_size = 49_152;
        inp.head_dim = 128;
        inp.n_heads = 4;
        inp.n_kv_heads = 4;
        inp.n_layers = 8;
        inp.max_seq = 2048;
        inp.batch = 32;
        let est = estimate(&inp);
        assert!(est.cfie_us_per_token > 0.0);
        assert!(est.baseline_us_per_token > 0.0);
        assert!(est.throughput_tok_s > 0.0);
        assert_eq!(est.batch, 32);
        assert_eq!(est.kv_seq, 1024);
        assert!(est.assumptions.contains("seq=1024"));
    }
}
