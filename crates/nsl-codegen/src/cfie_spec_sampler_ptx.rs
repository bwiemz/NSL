//! CFIE Cycle 13 (G15 draft-model-in-binary): the two draft-side
//! sampler-family kernels.
//!
//!   * `nsl_cfie_draft_sample` (registration kind 7) - greedy argmax +
//!     p(chosen) over the DRAFT model's LM head.  The paper drafts K
//!     tokens at temperature 0.0, recording `draft_probs[k] =
//!     softmax(draft_logits)[chosen]`; greedy means chosen == argmax,
//!     so p(argmax) falls out of ONE streaming flash-softmax pass for
//!     free: `exp(x_argmax - max) == 1`, hence `p = 1 / sum_final`.
//!     The pass keeps a running max + argmax + online sum of
//!     `exp(x - max)` with rescale - the same trick the fused sampler
//!     (`cfie_sample_ptx`) already uses.  `rng_seed` is ACCEPTED for
//!     ABI symmetry with the fused sampler but UNUSED - v1 drafting is
//!     greedy per the paper.
//!
//!   * `nsl_cfie_verify_probs` (registration kind 8) - full softmaxed
//!     prob-ROW writer for the TARGET model.  The fused sampler never
//!     materializes probs by design (its whole point is that the
//!     `[1, vocab]` logits row never touches HBM), but the rejection
//!     kernel (`cfie_speculative_ptx::REJECT_KERNEL_NAME`, kind 4)
//!     consumes softmaxed f32 rows `target_probs[k][vocab]`.  This
//!     kernel EXISTS to materialize that row: pass 1 computes max +
//!     sum online (recomputing the matvec per tile), pass 2 recomputes
//!     the matvec per tile and stores `p_i = exp(x_i - max) / sum`.
//!     The 2x matvec is the price of never staging the logits row and
//!     is bounded by K <= 32 verify positions per round.
//!
//! Both kernels are sampler-family: NO KV access, grid = 1 CTA,
//! block = 128, static `.shared`.  RMSNorm(hidden) with the final-norm
//! gamma runs once into SMEM - the section is identical to
//! `cfie_sample_ptx` (strided partial sums, SMEM tree reduction,
//! rsqrt.approx, gamma scale in place).
//!
//! Lossless self-speculation anchor (the engine's determinism proof):
//! with draft == target weights both kernels compute bit-identical
//! (max, sum) - same fma dot order, same online-merge order - so
//! `p_draft = div(1, sum)` from kind 7 equals
//! `p_target[tok] = div(ex2(0), sum)` from kind 8 bit-for-bit
//! (`ex2(+-0) == 1.0` exactly per the PTX ISA).  The reject kernel's
//! ratio is then exactly 1.0 and every drafted token accepts
//! regardless of seed.
//!
//! `cpu_reference_*` mirror the kernels' arithmetic order exactly
//! (same strided partial sums + tree reduction, same fma dot order,
//! same online max/sum merge, same division).  The kernels use
//! `rsqrt.approx` / `ex2.approx` where the CPU uses exact libm - token
//! parity is expected-exact modulo rsqrt.approx rounding on knife-edge
//! argmax ties (the rstd delta perturbs the normalized hidden by <1
//! ulp-scale, so only dots equal to within that delta could flip), and
//! prob parity is tight-float.  The determinism contract does NOT rest
//! on CPU-vs-GPU parity - it rests on GPU-side kind-7 vs kind-8 BIT
//! identity (above), proven by the engine's self-speculation contract
//! (spec generate == plain generate).

use std::fmt::Write;

/// Threads per CTA == vocab tile width (thread t owns row tile_base+t).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

/// Baked RMSNorm epsilon - same value as `cfie_sample_ptx` (paper
/// stage 1); the draft and target final norms share it.
const RMS_EPS: f32 = 1e-5;

pub const DRAFT_SAMPLE_KERNEL_NAME: &str = "nsl_cfie_draft_sample";
pub const VERIFY_PROBS_KERNEL_NAME: &str = "nsl_cfie_verify_probs";

/// Registration kind the serve wiring passes to
/// `nsl_cfie_register_kernel` for the draft sampler (Cycle-13 frozen
/// ABI; kinds 0-5 are the Cycle-6 ABI in `cfie.rs`, kind 6 is the
/// draft decode_block emitted by `cfie_persistent_ptx`).
pub const DRAFT_SAMPLE_KERNEL_KIND: u8 = 7;
/// Registration kind for the verify prob-row writer (Cycle-13 ABI).
pub const VERIFY_PROBS_KERNEL_KIND: u8 = 8;

/// Compile-time configuration shared by both kernels - mirrors
/// `cfie_sample_ptx::FusedSampleKernelConfig` minus the sampling
/// params (the draft is fixed-function greedy; the verify writer has
/// no sampling at all).
#[derive(Debug, Clone)]
pub struct SpecSamplerConfig {
    pub d_model: u32,
    pub vocab_size: u32,
    pub vocab_tile: u32,
    pub sm_version: u32,
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct SpecSamplerMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
}

/// Mirrors `gpu_specs::GpuSpec::ptx_version` (same convention as the
/// sibling CFIE emitters): sm_100+ -> 8.6, sm_90+ -> 8.4, else 7.0.
fn ptx_version_for_sm(sm: u32) -> &'static str {
    if sm >= 100 {
        "8.6"
    } else if sm >= 90 {
        "8.4"
    } else {
        "7.0"
    }
}

fn f32_imm(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

fn validate_config(cfg: &SpecSamplerConfig) {
    assert_eq!(
        cfg.vocab_tile, TILE,
        "vocab_tile must be {} so thread t owns row tile_base+t",
        TILE
    );
    assert!(
        cfg.d_model >= 1 && cfg.d_model <= 8192,
        "d_model must be in 1..=8192 (hidden state staged in static SMEM)"
    );
    assert!(cfg.vocab_size >= 1, "vocab_size must be >= 1");
}

/// Cooperative hidden load + unconditional RMSNorm in SMEM - the
/// identical section to `cfie_sample_ptx` (both kernels always
/// normalize: they consume the raw last-layer hidden state and the
/// bound final-norm gamma).  The scores region doubles as the
/// reduction scratch pre-tile-loop, same reuse as the fused sampler.
fn emit_hidden_load_and_rmsnorm(w: &mut String, dm: u32, scores_off: u32, rms_off: u32) {
    let zero = f32_imm(0.0);
    let inv_dm = f32_imm(1.0 / dm as f32);
    let eps = f32_imm(RMS_EPS);

    writeln!(w, "    // 1. cooperative strided load: hidden [1, d_model] f32 -> SMEM").unwrap();
    writeln!(w, "    mov.u32 %r_i, %r_tid;").unwrap();
    writeln!(w, "HLOAD_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_i, {};", dm).unwrap();
    writeln!(w, "    @%p_a bra HLOAD_DONE;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_i, 4;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_t0, %r_t0;").unwrap();
    writeln!(w, "    add.u64 %rd_a, %rd_hidden, %rd_t0;").unwrap();
    writeln!(w, "    ld.global.f32 %f_h, [%rd_a];").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t0], %f_h;").unwrap();
    writeln!(w, "    add.u32 %r_i, %r_i, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra HLOAD_LOOP;").unwrap();
    writeln!(w, "HLOAD_DONE:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // 2. RMSNorm in SMEM: per-thread strided partial sum of squares").unwrap();
    writeln!(w, "    mov.f32 %f_ss, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_i, %r_tid;").unwrap();
    writeln!(w, "SS_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_i, {};", dm).unwrap();
    writeln!(w, "    @%p_a bra SS_DONE;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_i, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_h, [%r_t0];").unwrap();
    writeln!(w, "    fma.rn.f32 %f_ss, %f_h, %f_h, %f_ss;").unwrap();
    writeln!(w, "    add.u32 %r_i, %r_i, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra SS_LOOP;").unwrap();
    writeln!(w, "SS_DONE:").unwrap();
    writeln!(w, "    // scores region doubles as reduction scratch pre-tile-loop").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t0+{}], %f_ss;", scores_off).unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    for off in [64u32, 32, 16, 8, 4, 2, 1] {
        writeln!(w, "    setp.ge.u32 %p_a, %r_tid, {};", off).unwrap();
        writeln!(w, "    @%p_a bra RED_{};", off).unwrap();
        writeln!(w, "    mul.lo.u32 %r_t0, %r_tid, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_t0, [%r_t0+{}];", scores_off).unwrap();
        writeln!(w, "    ld.shared.f32 %f_t1, [%r_t0+{}];", scores_off + off * 4).unwrap();
        writeln!(w, "    add.f32 %f_t0, %f_t0, %f_t1;").unwrap();
        writeln!(w, "    st.shared.f32 [%r_t0+{}], %f_t0;", scores_off).unwrap();
        writeln!(w, "RED_{}:", off).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
    }
    writeln!(w, "    // thread 0: rstd = rsqrt(sum_sq / d_model + eps)").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra RSTD_DONE;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_ss, [%r_sbase+{}];", scores_off).unwrap();
    writeln!(w, "    mul.f32 %f_ss, %f_ss, {};", inv_dm).unwrap();
    writeln!(w, "    add.f32 %f_ss, %f_ss, {};", eps).unwrap();
    writeln!(w, "    rsqrt.approx.f32 %f_rstd, %f_ss;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sbase+{}], %f_rstd;", rms_off).unwrap();
    writeln!(w, "RSTD_DONE:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_rstd, [%r_sbase+{}];", rms_off).unwrap();
    writeln!(w, "    // scale hidden in place: h = h * rstd * gamma").unwrap();
    writeln!(w, "    mov.u32 %r_i, %r_tid;").unwrap();
    writeln!(w, "NRM_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_i, {};", dm).unwrap();
    writeln!(w, "    @%p_a bra NRM_DONE;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_i, 4;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_t0, %r_t0;").unwrap();
    writeln!(w, "    add.u64 %rd_a, %rd_norm, %rd_t0;").unwrap();
    writeln!(w, "    ld.global.f32 %f_g, [%rd_a];").unwrap();
    writeln!(w, "    add.u32 %r_t1, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_h, [%r_t1];").unwrap();
    writeln!(w, "    mul.f32 %f_h, %f_h, %f_rstd;").unwrap();
    writeln!(w, "    mul.f32 %f_h, %f_h, %f_g;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t1], %f_h;").unwrap();
    writeln!(w, "    add.u32 %r_i, %r_i, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra NRM_LOOP;").unwrap();
    writeln!(w, "NRM_DONE:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
}

/// Streaming f16-row matvec for token `%r_tok`: leaves
/// `dot(x_smem, W[tok])` in `%f_dot` - same fma order + f16 loads as
/// the fused sampler's DOT_LOOP.  Clobbers `%rd_t0`, `%rd_a`, `%r_d`,
/// `%r_t1`, `%h_w`, `%f_w`, `%f_h`, `%p_b`.
fn emit_row_dot(w: &mut String, prefix: &str, dm: u32, w_row_bytes: u64) {
    let zero = f32_imm(0.0);
    writeln!(w, "    // f16 row: lm_head_ptr + tok * d_model * 2").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t0, %r_tok, {};", w_row_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_a, %rd_w, %rd_t0;").unwrap();
    writeln!(w, "    mov.f32 %f_dot, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_d, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_t1, %r_sbase;").unwrap();
    writeln!(w, "{}_DOT:", prefix).unwrap();
    writeln!(w, "    setp.ge.u32 %p_b, %r_d, {};", dm).unwrap();
    writeln!(w, "    @%p_b bra {}_DOTD;", prefix).unwrap();
    writeln!(w, "    ld.global.b16 %h_w, [%rd_a];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f_w, %h_w;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_h, [%r_t1];").unwrap();
    writeln!(w, "    fma.rn.f32 %f_dot, %f_w, %f_h, %f_dot;").unwrap();
    writeln!(w, "    add.u64 %rd_a, %rd_a, 2;").unwrap();
    writeln!(w, "    add.u32 %r_t1, %r_t1, 4;").unwrap();
    writeln!(w, "    add.u32 %r_d, %r_d, 1;").unwrap();
    writeln!(w, "    bra {}_DOT;", prefix).unwrap();
    writeln!(w, "{}_DOTD:", prefix).unwrap();
}

/// The streaming online-softmax pass shared by the draft kernel and
/// verify pass 1: per tile all 128 threads score their row into SMEM,
/// then thread 0 serially merges the tile into the running
/// (max, sum[, argmax]) state - per element:
///   if s > m: sum *= ex2((m - s) * log2e); m = s; [sel = token;]
///   sum += ex2((s - m) * log2e)   (== 1.0 exactly on the max path).
/// Strict `>` keeps first-max-wins tie-breaks; the CPU reference
/// mirrors the order exactly.  Thread 0's `%f_m` / `%f_sum`
/// [/ `%r_sel`] carry the state across tiles.
fn emit_streaming_pass(
    w: &mut String,
    prefix: &str,
    dm: u32,
    vocab: u32,
    w_row_bytes: u64,
    scores_off: u32,
    track_argmax: bool,
) {
    let neg_inf = f32_imm(f32::NEG_INFINITY);
    let zero = f32_imm(0.0);
    let log2e = f32_imm(std::f32::consts::LOG2_E);

    writeln!(w, "    mov.f32 %f_m, {};", neg_inf).unwrap();
    writeln!(w, "    mov.f32 %f_sum, {};", zero).unwrap();
    if track_argmax {
        writeln!(w, "    mov.u32 %r_sel, 0;").unwrap();
    }
    writeln!(w, "    mov.u32 %r_tile, 0;").unwrap();
    writeln!(w, "{}_TILE:", prefix).unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_tile, {};", vocab).unwrap();
    writeln!(w, "    @%p_a bra {}_TILES_DONE;", prefix).unwrap();
    writeln!(w, "    add.u32 %r_tok, %r_tile, %r_tid;").unwrap();
    writeln!(w, "    mov.f32 %f_s, {};", neg_inf).unwrap();
    writeln!(w, "    // tail-tile guard: lanes past vocab keep -inf").unwrap();
    writeln!(w, "    setp.ge.u32 %p_b, %r_tok, {};", vocab).unwrap();
    writeln!(w, "    @%p_b bra {}_SSTORE;", prefix).unwrap();
    emit_row_dot(w, prefix, dm, w_row_bytes);
    writeln!(w, "    mov.f32 %f_s, %f_dot;").unwrap();
    writeln!(w, "{}_SSTORE:", prefix).unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t0+{}], %f_s;", scores_off).unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // thread 0: online flash-softmax merge of the tile's cnt scores").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra {}_MERGED;", prefix).unwrap();
    writeln!(w, "    sub.u32 %r_cnt, {}, %r_tile;", vocab).unwrap();
    writeln!(w, "    min.u32 %r_cnt, %r_cnt, {};", TILE).unwrap();
    writeln!(w, "    mov.u32 %r_i, 0;").unwrap();
    writeln!(w, "{}_MERGE:", prefix).unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_i, %r_cnt;").unwrap();
    writeln!(w, "    @%p_a bra {}_MERGED;", prefix).unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_i, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_s, [%r_t0+{}];", scores_off).unwrap();
    writeln!(w, "    setp.gt.f32 %p_b, %f_s, %f_m;").unwrap();
    writeln!(w, "    @!%p_b bra {}_ACC;", prefix).unwrap();
    if track_argmax {
        writeln!(w, "    // new running max: rescale the online sum, adopt the argmax").unwrap();
    } else {
        writeln!(w, "    // new running max: rescale the online sum").unwrap();
    }
    writeln!(w, "    sub.f32 %f_t0, %f_m, %f_s;").unwrap();
    writeln!(w, "    mul.f32 %f_t0, %f_t0, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f_t0, %f_t0;").unwrap();
    writeln!(w, "    mul.f32 %f_sum, %f_sum, %f_t0;").unwrap();
    writeln!(w, "    mov.f32 %f_m, %f_s;").unwrap();
    if track_argmax {
        writeln!(w, "    add.u32 %r_sel, %r_tile, %r_i;").unwrap();
    }
    writeln!(w, "{}_ACC:", prefix).unwrap();
    writeln!(w, "    // ex2(0) == 1.0 exactly on the max path (PTX ISA)").unwrap();
    writeln!(w, "    sub.f32 %f_t0, %f_s, %f_m;").unwrap();
    writeln!(w, "    mul.f32 %f_t0, %f_t0, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f_t0, %f_t0;").unwrap();
    writeln!(w, "    add.f32 %f_sum, %f_sum, %f_t0;").unwrap();
    writeln!(w, "    add.u32 %r_i, %r_i, 1;").unwrap();
    writeln!(w, "    bra {}_MERGE;", prefix).unwrap();
    writeln!(w, "{}_MERGED:", prefix).unwrap();
    writeln!(w, "    // scores SMEM rewritten next tile; sync before loop back").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    add.u32 %r_tile, %r_tile, {};", TILE).unwrap();
    writeln!(w, "    bra {}_TILE;", prefix).unwrap();
    writeln!(w, "{}_TILES_DONE:", prefix).unwrap();
}

// ---------------------------------------------------------------------------
// Kind 7: nsl_cfie_draft_sample
// ---------------------------------------------------------------------------

/// Emit the draft greedy sampler kernel for `cfg`.
pub fn emit_draft_sample(cfg: &SpecSamplerConfig) -> (String, SpecSamplerMeta) {
    validate_config(cfg);
    let dm = cfg.d_model;
    let vocab = cfg.vocab_size;
    let w_row_bytes = dm as u64 * 2;

    // SMEM layout (f32): [hidden: d_model][scores: TILE][rstd: 1].
    let scores_off = dm * 4;
    let rms_off = scores_off + TILE * 4;
    let smem_bytes = rms_off + 4;

    let one = f32_imm(1.0);
    let eps = f32_imm(RMS_EPS);

    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(w, "// {} - CFIE draft-model greedy sampler (Cycle 13, G15).", DRAFT_SAMPLE_KERNEL_NAME).unwrap();
    writeln!(w, "// One CTA, {} threads; ONE streaming pass over the vocab tiles keeps", BLOCK_DIM).unwrap();
    writeln!(w, "// a running max + argmax + online sum of exp(x - max) with rescale").unwrap();
    writeln!(w, "// (flash softmax).  p(argmax) = 1/sum_final because").unwrap();
    writeln!(w, "// exp(x_argmax - max) == 1 when the argmax attains the max.").unwrap();
    writeln!(w, "// Outputs: token id (u32) + p(argmax) (f32) - 8 bytes to HBM.").unwrap();
    writeln!(w, "// LM-head layout: f16 [vocab, d_model], ROW-major per vocab row.").unwrap();
    writeln!(w, "// rng_seed is ACCEPTED for ABI symmetry with the fused sampler but").unwrap();
    writeln!(w, "// UNUSED: v1 drafting is greedy (the paper's temperature 0.0).").unwrap();
    writeln!(w, "// Baked constants:").unwrap();
    writeln!(w, "//   d_model    = {}", dm).unwrap();
    writeln!(w, "//   vocab_size = {}", vocab).unwrap();
    writeln!(w, "//   vocab_tile = {}", TILE).unwrap();
    writeln!(w, "//   rms_eps    = {} ({})", RMS_EPS, eps).unwrap();
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", ptx_version_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".shared .align 4 .b8 cfie_draft_sample_smem[{}];", smem_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", DRAFT_SAMPLE_KERNEL_NAME).unwrap();
    writeln!(w, "    .param .u64 hidden_ptr,").unwrap();
    writeln!(w, "    .param .u64 norm_w_ptr,").unwrap();
    writeln!(w, "    .param .u64 lm_head_ptr,").unwrap();
    writeln!(w, "    .param .u64 out_token_ptr,").unwrap();
    writeln!(w, "    .param .u64 out_prob_ptr,").unwrap();
    writeln!(w, "    .param .u64 rng_seed").unwrap();
    writeln!(w, ")").unwrap();
    writeln!(w, "{{").unwrap();
    writeln!(w, "    .reg .pred %p_a, %p_b, %p_t0;").unwrap();
    writeln!(w, "    .reg .b16 %h_w;").unwrap();
    writeln!(
        w,
        "    .reg .f32 %f_h, %f_w, %f_dot, %f_s, %f_ss, %f_g, %f_rstd, %f_m, %f_sum, %f_p, %f_t0, %f_t1;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_tid, %r_sbase, %r_i, %r_d, %r_tile, %r_tok, %r_cnt, %r_sel, %r_t0, %r_t1;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_hidden, %rd_norm, %rd_w, %rd_outtok, %rd_outprob, %rd_seed, %rd_a, %rd_t0;"
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_hidden, [hidden_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_norm, [norm_w_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_w, [lm_head_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_outtok, [out_token_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_outprob, [out_prob_ptr];").unwrap();
    writeln!(w, "    // ACCEPTED for ABI symmetry; UNUSED - v1 draft is greedy").unwrap();
    writeln!(w, "    ld.param.u64 %rd_seed, [rng_seed];").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_sbase, cfie_draft_sample_smem;").unwrap();
    writeln!(w).unwrap();

    emit_hidden_load_and_rmsnorm(w, dm, scores_off, rms_off);

    writeln!(w, "    // 3. streaming flash-softmax argmax pass over the vocab tiles").unwrap();
    emit_streaming_pass(w, "DS", dm, vocab, w_row_bytes, scores_off, true);
    writeln!(w).unwrap();
    writeln!(w, "    // 4. thread 0 publishes; the kernel's only global stores").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra EXIT;").unwrap();
    writeln!(w, "    // p(argmax) = 1 / sum_final (exp(x_argmax - max) == 1)").unwrap();
    writeln!(w, "    mov.f32 %f_t0, {};", one).unwrap();
    writeln!(w, "    div.rn.f32 %f_p, %f_t0, %f_sum;").unwrap();
    writeln!(w, "    st.global.u32 [%rd_outtok], %r_sel;").unwrap();
    writeln!(w, "    st.global.f32 [%rd_outprob], %f_p;").unwrap();
    writeln!(w, "EXIT:").unwrap();
    writeln!(w, "    ret;").unwrap();
    writeln!(w, "}}").unwrap();

    let meta = SpecSamplerMeta {
        kernel_name: DRAFT_SAMPLE_KERNEL_NAME.to_string(),
        smem_bytes,
        block_dim: BLOCK_DIM,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit_draft_sample`].
pub fn emit_draft_sample_ptx(cfg: &SpecSamplerConfig) -> String {
    emit_draft_sample(cfg).0
}

// ---------------------------------------------------------------------------
// Kind 8: nsl_cfie_verify_probs
// ---------------------------------------------------------------------------

/// Emit the target prob-row writer kernel for `cfg`.
pub fn emit_verify_probs(cfg: &SpecSamplerConfig) -> (String, SpecSamplerMeta) {
    validate_config(cfg);
    let dm = cfg.d_model;
    let vocab = cfg.vocab_size;
    let w_row_bytes = dm as u64 * 2;

    // SMEM layout (f32): [hidden: d_model][scores: TILE][rstd: 1]
    //                    [max: 1][sum: 1].
    let scores_off = dm * 4;
    let rms_off = scores_off + TILE * 4;
    let max_off = rms_off + 4;
    let sum_off = max_off + 4;
    let smem_bytes = sum_off + 4;

    let log2e = f32_imm(std::f32::consts::LOG2_E);
    let eps = f32_imm(RMS_EPS);

    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(w, "// {} - CFIE target prob-row writer (Cycle 13, G15).", VERIFY_PROBS_KERNEL_NAME).unwrap();
    writeln!(w, "// One CTA, {} threads; TWO passes over the vocab tiles:", BLOCK_DIM).unwrap();
    writeln!(w, "//   pass 1: online max + exp-sum (recomputing the matvec per tile),").unwrap();
    writeln!(w, "//   pass 2: recompute the matvec, store p_i = exp(x_i - max)/sum.").unwrap();
    writeln!(w, "// This kernel EXISTS to materialize softmaxed f32 rows for the").unwrap();
    writeln!(w, "// rejection kernel (nsl_cfie_spec_reject) - the fused sampler never").unwrap();
    writeln!(w, "// writes probs by design.  The 2x matvec is the price and is").unwrap();
    writeln!(w, "// bounded by K <= 32 verify positions per round.").unwrap();
    writeln!(w, "// LM-head layout: f16 [vocab, d_model], ROW-major per vocab row.").unwrap();
    writeln!(w, "// Baked constants:").unwrap();
    writeln!(w, "//   d_model    = {}", dm).unwrap();
    writeln!(w, "//   vocab_size = {}", vocab).unwrap();
    writeln!(w, "//   vocab_tile = {}", TILE).unwrap();
    writeln!(w, "//   rms_eps    = {} ({})", RMS_EPS, eps).unwrap();
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", ptx_version_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".shared .align 4 .b8 cfie_verify_probs_smem[{}];", smem_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", VERIFY_PROBS_KERNEL_NAME).unwrap();
    writeln!(w, "    .param .u64 hidden_ptr,").unwrap();
    writeln!(w, "    .param .u64 norm_w_ptr,").unwrap();
    writeln!(w, "    .param .u64 lm_head_ptr,").unwrap();
    writeln!(w, "    .param .u64 out_probs_ptr").unwrap();
    writeln!(w, ")").unwrap();
    writeln!(w, "{{").unwrap();
    writeln!(w, "    .reg .pred %p_a, %p_b, %p_t0;").unwrap();
    writeln!(w, "    .reg .b16 %h_w;").unwrap();
    writeln!(
        w,
        "    .reg .f32 %f_h, %f_w, %f_dot, %f_s, %f_ss, %f_g, %f_rstd, %f_m, %f_sum, %f_t0, %f_t1;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_tid, %r_sbase, %r_i, %r_d, %r_tile, %r_tok, %r_cnt, %r_t0, %r_t1;"
    )
    .unwrap();
    writeln!(w, "    .reg .u64 %rd_hidden, %rd_norm, %rd_w, %rd_out, %rd_a, %rd_t0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_hidden, [hidden_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_norm, [norm_w_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_w, [lm_head_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_out, [out_probs_ptr];").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_sbase, cfie_verify_probs_smem;").unwrap();
    writeln!(w).unwrap();

    emit_hidden_load_and_rmsnorm(w, dm, scores_off, rms_off);

    writeln!(w, "    // 3. pass 1: online max + exp-sum over the vocab tiles").unwrap();
    writeln!(w, "    //    (identical merge order to nsl_cfie_draft_sample - the").unwrap();
    writeln!(w, "    //    self-speculation anchor relies on bit-identical (max, sum))").unwrap();
    emit_streaming_pass(w, "P1", dm, vocab, w_row_bytes, scores_off, false);
    writeln!(w).unwrap();
    writeln!(w, "    // 4. thread 0 publishes (max, sum); all threads reload them").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra PUB_DONE;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sbase+{}], %f_m;", max_off).unwrap();
    writeln!(w, "    st.shared.f32 [%r_sbase+{}], %f_sum;", sum_off).unwrap();
    writeln!(w, "PUB_DONE:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_m, [%r_sbase+{}];", max_off).unwrap();
    writeln!(w, "    ld.shared.f32 %f_sum, [%r_sbase+{}];", sum_off).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // 5. pass 2: recompute the matvec per tile, store the row;").unwrap();
    writeln!(w, "    //    no SMEM writes -> no barriers needed inside the loop").unwrap();
    writeln!(w, "    mov.u32 %r_tile, 0;").unwrap();
    writeln!(w, "P2_TILE:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_tile, {};", vocab).unwrap();
    writeln!(w, "    @%p_a bra P2_DONE;").unwrap();
    writeln!(w, "    add.u32 %r_tok, %r_tile, %r_tid;").unwrap();
    writeln!(w, "    // tail-tile guard: lanes past vocab store nothing").unwrap();
    writeln!(w, "    setp.ge.u32 %p_b, %r_tok, {};", vocab).unwrap();
    writeln!(w, "    @%p_b bra P2_NEXT;").unwrap();
    emit_row_dot(w, "P2", dm, w_row_bytes);
    writeln!(w, "    // p = ex2((x - max) * log2e) / sum; div.rn matches the CPU '/'").unwrap();
    writeln!(w, "    sub.f32 %f_t0, %f_dot, %f_m;").unwrap();
    writeln!(w, "    mul.f32 %f_t0, %f_t0, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f_t0, %f_t0;").unwrap();
    writeln!(w, "    div.rn.f32 %f_t0, %f_t0, %f_sum;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t0, %r_tok, 4;").unwrap();
    writeln!(w, "    add.u64 %rd_a, %rd_out, %rd_t0;").unwrap();
    writeln!(w, "    st.global.f32 [%rd_a], %f_t0;").unwrap();
    writeln!(w, "P2_NEXT:").unwrap();
    writeln!(w, "    add.u32 %r_tile, %r_tile, {};", TILE).unwrap();
    writeln!(w, "    bra P2_TILE;").unwrap();
    writeln!(w, "P2_DONE:").unwrap();
    writeln!(w, "    ret;").unwrap();
    writeln!(w, "}}").unwrap();

    let meta = SpecSamplerMeta {
        kernel_name: VERIFY_PROBS_KERNEL_NAME.to_string(),
        smem_bytes,
        block_dim: BLOCK_DIM,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit_verify_probs`].
pub fn emit_verify_probs_ptx(cfg: &SpecSamplerConfig) -> String {
    emit_verify_probs(cfg).0
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

/// Kernel-order RMSNorm: per-thread strided partial sums, the SMEM
/// tree reduction, gamma scale - identical staging to the kernels
/// (and to `cfie_sample_ptx::cpu_reference`).  The kernel's
/// `rsqrt.approx` is exact libm here; token parity is unaffected (the
/// argmax compares dots computed from the same scaled hidden).
fn rms_norm_kernel_order(hidden: &[f32], norm_w: &[f32]) -> Vec<f32> {
    let dm = hidden.len();
    let mut partials = [0f32; TILE as usize];
    for (t, part) in partials.iter_mut().enumerate() {
        let mut s = 0f32;
        let mut i = t;
        while i < dm {
            s = hidden[i].mul_add(hidden[i], s);
            i += TILE as usize;
        }
        *part = s;
    }
    for off in [64usize, 32, 16, 8, 4, 2, 1] {
        for t in 0..off {
            partials[t] += partials[t + off];
        }
    }
    let mean = partials[0] * (1.0 / dm as f32);
    let rstd = 1.0 / (mean + RMS_EPS).sqrt();
    hidden
        .iter()
        .zip(norm_w)
        .map(|(h, g)| (h * rstd) * g)
        .collect()
}

/// The kernels' fma-ordered f16-row matvec (same order as the fused
/// sampler's reference).  `lm_head_f32` is `[vocab][d_model]`
/// row-major f32 - an exact-value cast of the kernel's f16 weights.
fn row_dot(h: &[f32], lm_head_f32: &[f32], tok: usize) -> f32 {
    let dm = h.len();
    let row = &lm_head_f32[tok * dm..(tok + 1) * dm];
    let mut dot = 0f32;
    for d in 0..dm {
        dot = row[d].mul_add(h[d], dot);
    }
    dot
}

/// Running state of the streaming online-softmax pass.
struct OnlineSoftmax {
    m: f32,
    sum: f32,
    sel: u32,
}

/// The kernels' streaming pass, element order and arithmetic mirrored
/// exactly: per tile the scores are staged first (the SMEM write), then
/// merged serially with strict-`>` first-max-wins tie-breaks.
fn online_pass(h: &[f32], lm_head_f32: &[f32], vocab: usize) -> OnlineSoftmax {
    let log2e = std::f32::consts::LOG2_E;
    let mut st = OnlineSoftmax {
        m: f32::NEG_INFINITY,
        sum: 0.0,
        sel: 0,
    };
    let mut tile = 0usize;
    while tile < vocab {
        let cnt = (vocab - tile).min(TILE as usize);
        let mut scores = vec![0f32; cnt];
        for (t, sc) in scores.iter_mut().enumerate() {
            *sc = row_dot(h, lm_head_f32, tile + t);
        }
        for (i, &s) in scores.iter().enumerate() {
            if s > st.m {
                st.sum *= ((st.m - s) * log2e).exp2();
                st.m = s;
                st.sel = (tile + i) as u32;
            }
            st.sum += ((s - st.m) * log2e).exp2();
        }
        tile += TILE as usize;
    }
    st
}

fn validate_cpu_inputs(
    cfg: &SpecSamplerConfig,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
) -> (usize, usize) {
    let dm = cfg.d_model as usize;
    let vocab = cfg.vocab_size as usize;
    assert_eq!(cfg.vocab_tile, TILE, "vocab_tile must be {TILE}");
    assert!((1..=8192).contains(&dm), "d_model must be in 1..=8192");
    assert!(vocab >= 1, "vocab_size must be >= 1");
    assert_eq!(hidden.len(), dm, "hidden must be [1, d_model]");
    assert_eq!(norm_w.len(), dm, "norm_w (gamma) must be [d_model]");
    assert_eq!(
        lm_head_f32.len(),
        vocab * dm,
        "lm_head must be [vocab][d_model] row-major"
    );
    (dm, vocab)
}

/// CPU mirror of `nsl_cfie_draft_sample`: returns
/// `(argmax token, p(argmax))`.  Same accumulation order as the kernel
/// so parity is exact-token + tight-float (see module docs).
pub fn cpu_reference_draft_sample(
    cfg: &SpecSamplerConfig,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
) -> (u32, f32) {
    let (_, vocab) = validate_cpu_inputs(cfg, hidden, norm_w, lm_head_f32);
    let h = rms_norm_kernel_order(hidden, norm_w);
    let st = online_pass(&h, lm_head_f32, vocab);
    (st.sel, 1.0 / st.sum)
}

/// CPU mirror of `nsl_cfie_verify_probs`: returns the softmaxed prob
/// row `[vocab]` the rejection kernel consumes.  Pass 2 recomputes
/// the matvec per token exactly like the kernel does (the price of
/// never staging the logits row).
pub fn cpu_reference_verify_probs(
    cfg: &SpecSamplerConfig,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
) -> Vec<f32> {
    let (_, vocab) = validate_cpu_inputs(cfg, hidden, norm_w, lm_head_f32);
    let log2e = std::f32::consts::LOG2_E;
    let h = rms_norm_kernel_order(hidden, norm_w);
    let st = online_pass(&h, lm_head_f32, vocab);
    (0..vocab)
        .map(|tok| {
            let dot = row_dot(&h, lm_head_f32, tok);
            ((dot - st.m) * log2e).exp2() / st.sum
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
    use crate::cfie_speculative_ptx::{cpu_reference_reject, RejectionConfig};

    fn cfg(d_model: u32, vocab_size: u32) -> SpecSamplerConfig {
        SpecSamplerConfig {
            d_model,
            vocab_size,
            vocab_tile: 128,
            sm_version: 80,
        }
    }

    /// Reference config from the CFIE paper's NSL-Coder example (the
    /// draft model shares the target's vocab per the spec sub-block).
    fn paper_cfg() -> SpecSamplerConfig {
        cfg(512, 49_152)
    }

    fn f16r(v: f32) -> f32 {
        half::f16::from_f32(v).to_f32()
    }

    /// f16-rounded pseudo-random LM head (exact-value f32 cast, the
    /// kernels' weight contract).
    fn lm_head(vocab: usize, dm: usize) -> Vec<f32> {
        (0..vocab * dm)
            .map(|i| f16r(((i * 7 + 3) % 23) as f32 * 0.07 - 0.7))
            .collect()
    }

    fn hidden_pattern(dm: usize) -> Vec<f32> {
        (0..dm).map(|i| (i as f32 * 0.37).sin()).collect()
    }

    fn gamma_pattern(dm: usize) -> Vec<f32> {
        (0..dm).map(|i| 1.0 + i as f32 * 0.01).collect()
    }

    fn param_lines(ptx: &str, entry: &str) -> Vec<String> {
        let start = ptx.find(entry).unwrap();
        let end = start + ptx[start..].find(')').unwrap();
        ptx[start..end]
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                l.starts_with(".param")
                    .then(|| l.trim_end_matches(',').to_string())
            })
            .collect()
    }

    // -- structural ------------------------------------------------------

    #[test]
    fn draft_param_list_is_exactly_the_six_params() {
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        assert_eq!(
            param_lines(&ptx, ".visible .entry nsl_cfie_draft_sample("),
            vec![
                ".param .u64 hidden_ptr",
                ".param .u64 norm_w_ptr",
                ".param .u64 lm_head_ptr",
                ".param .u64 out_token_ptr",
                ".param .u64 out_prob_ptr",
                ".param .u64 rng_seed",
            ]
        );
        assert_eq!(ptx.matches("ld.param").count(), 6);
    }

    #[test]
    fn verify_param_list_is_exactly_the_four_params() {
        let ptx = emit_verify_probs_ptx(&paper_cfg());
        assert_eq!(
            param_lines(&ptx, ".visible .entry nsl_cfie_verify_probs("),
            vec![
                ".param .u64 hidden_ptr",
                ".param .u64 norm_w_ptr",
                ".param .u64 lm_head_ptr",
                ".param .u64 out_probs_ptr",
            ]
        );
        assert_eq!(ptx.matches("ld.param").count(), 4);
    }

    #[test]
    fn draft_stores_exactly_token_and_prob() {
        // The draft sampler's whole output is 8 bytes: token + prob.
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        assert_eq!(ptx.matches("st.global").count(), 2);
        assert!(ptx.contains("st.global.u32 [%rd_outtok], %r_sel;"));
        assert!(ptx.contains("st.global.f32 [%rd_outprob], %f_p;"));
    }

    #[test]
    fn verify_single_global_store_is_the_prob_row() {
        // One st.global.f32 in the pass-2 loop body - executed once
        // per vocab entry, the row the reject kernel consumes.
        let ptx = emit_verify_probs_ptx(&paper_cfg());
        assert_eq!(ptx.matches("st.global").count(), 1);
        assert!(ptx.contains("st.global.f32 [%rd_a], %f_t0;"));
        // Normalization is div.rn so the CPU '/' mirrors it exactly.
        assert!(ptx.contains("div.rn.f32 %f_t0, %f_t0, %f_sum;"));
    }

    #[test]
    fn no_mad_lo_ascii_only_and_line_width_both_kernels() {
        for ptx in [
            emit_draft_sample_ptx(&paper_cfg()),
            emit_verify_probs_ptx(&paper_cfg()),
        ] {
            assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
            assert!(
                ptx.bytes().all(|b| b < 128),
                "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
            );
            assert!(!ptx.ends_with('\0'), "String PTX carries no trailing NUL");
            assert!(
                ptx.lines().all(|l| l.len() <= 132),
                "PTX lines must stay within 132 columns"
            );
        }
    }

    #[test]
    fn draft_is_greedy_no_prng_no_temperature() {
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        // rng_seed is ABI symmetry only: no xorshift64* constants, no
        // RNG mixing, no temperature epilogue on the scores.
        assert!(!ptx.contains("0x2545F4914F6CDD1D"));
        assert!(!ptx.contains("0x9E3779B97F4A7C15"));
        assert!(!ptx.contains("RNG"));
        assert!(ptx.contains("ACCEPTED for ABI symmetry; UNUSED"));
        assert!(ptx.contains("ld.param.u64 %rd_seed, [rng_seed];"));
    }

    #[test]
    fn rmsnorm_and_flash_softmax_sections_present_in_both() {
        for ptx in [
            emit_draft_sample_ptx(&paper_cfg()),
            emit_verify_probs_ptx(&paper_cfg()),
        ] {
            assert!(ptx.contains("rsqrt.approx.f32"));
            assert!(ptx.contains(&f32_imm(RMS_EPS)));
            assert!(ptx.contains("ex2.approx.f32"));
            // The online-merge rescale multiply (flash softmax).
            assert!(ptx.contains("mul.f32 %f_sum, %f_sum, %f_t0;"));
        }
    }

    #[test]
    fn header_matches_sm_version_convention() {
        for emit_ptx in [
            emit_draft_sample_ptx as fn(&SpecSamplerConfig) -> String,
            emit_verify_probs_ptx as fn(&SpecSamplerConfig) -> String,
        ] {
            let ptx80 = emit_ptx(&paper_cfg());
            assert!(ptx80.starts_with("//"));
            assert!(ptx80.contains(".version 7.0\n.target sm_80\n.address_size 64"));
            let mut c = paper_cfg();
            c.sm_version = 90;
            assert!(emit_ptx(&c).contains(".version 8.4\n.target sm_90"));
            c.sm_version = 100;
            assert!(emit_ptx(&c).contains(".version 8.6\n.target sm_100"));
        }
    }

    #[test]
    fn meta_reports_launch_shape() {
        let (ptx, meta) = emit_draft_sample(&paper_cfg());
        assert_eq!(meta.kernel_name, DRAFT_SAMPLE_KERNEL_NAME);
        assert_eq!(meta.block_dim, 128);
        // hidden(512 f32) + scores(128 f32) + rstd.
        assert_eq!(meta.smem_bytes, 512 * 4 + 128 * 4 + 4);
        assert!(ptx.contains(&format!(
            ".shared .align 4 .b8 cfie_draft_sample_smem[{}];",
            meta.smem_bytes
        )));

        let (ptx, meta) = emit_verify_probs(&paper_cfg());
        assert_eq!(meta.kernel_name, VERIFY_PROBS_KERNEL_NAME);
        assert_eq!(meta.block_dim, 128);
        // hidden + scores + rstd + max + sum.
        assert_eq!(meta.smem_bytes, 512 * 4 + 128 * 4 + 12);
        assert!(ptx.contains(&format!(
            ".shared .align 4 .b8 cfie_verify_probs_smem[{}];",
            meta.smem_bytes
        )));
    }

    #[test]
    fn registration_kinds_match_the_cycle13_abi() {
        assert_eq!(DRAFT_SAMPLE_KERNEL_KIND, 7);
        assert_eq!(VERIFY_PROBS_KERNEL_KIND, 8);
    }

    // -- refusals ---------------------------------------------------------

    #[test]
    #[should_panic(expected = "vocab_tile")]
    fn draft_vocab_tile_not_128_panics() {
        let mut c = paper_cfg();
        c.vocab_tile = 256;
        let _ = emit_draft_sample(&c);
    }

    #[test]
    #[should_panic(expected = "d_model")]
    fn verify_d_model_over_8192_panics() {
        let mut c = paper_cfg();
        c.d_model = 8193;
        let _ = emit_verify_probs(&c);
    }

    #[test]
    #[should_panic(expected = "vocab_size")]
    fn draft_zero_vocab_panics() {
        let mut c = paper_cfg();
        c.vocab_size = 0;
        let _ = emit_draft_sample(&c);
    }

    // -- cpu references ----------------------------------------------------

    #[test]
    fn cpu_draft_greedy_matches_fused_sampler_greedy() {
        let (dm, vocab) = (16usize, 48usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let mut w = lm_head(vocab, dm);
        // Plant a unique max at row 37: align the row with hidden's
        // signs (RMSNorm preserves signs - rstd and gamma positive).
        for d in 0..dm {
            w[37 * dm + d] = f16r(if hidden[d] >= 0.0 { 4.0 } else { -4.0 });
        }
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(tok, 37);
        assert!(prob > 0.0 && prob <= 1.0, "p(argmax) must be a probability");

        // Cross-check: the fused sampler's greedy cpu_reference with
        // top_k == vocab sees every token, so its argmax is global.
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: vocab as u32,
            ..Default::default()
        };
        let prog = emit_program(
            params,
            LmHeadShape {
                d_model: dm as u32,
                vocab_size: vocab as u32,
                vocab_tile: 128,
                dtype_bytes: 2,
            },
        );
        let fused = crate::cfie_sample_ptx::cpu_reference(&prog, &hidden, &gamma, &w, None, 7);
        assert_eq!(tok, fused, "draft greedy must agree with the fused sampler");
    }

    #[test]
    fn cpu_draft_prob_matches_f64_full_softmax() {
        // vocab 300 = 2 full tiles + a 44-wide tail tile.
        let (dm, vocab) = (32usize, 300usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);

        // Independent f64 pipeline (plain RMSNorm + softmax).
        let mut ss = 0f64;
        for &h in &hidden {
            ss += h as f64 * h as f64;
        }
        let rstd = 1.0 / (ss / dm as f64 + RMS_EPS as f64).sqrt();
        let h64: Vec<f64> = hidden
            .iter()
            .zip(&gamma)
            .map(|(h, g)| *h as f64 * rstd * *g as f64)
            .collect();
        let logits: Vec<f64> = (0..vocab)
            .map(|t| (0..dm).map(|d| w[t * dm + d] as f64 * h64[d]).sum())
            .collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|x| (x - max).exp()).sum();
        let argmax = logits
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |acc, (i, &x)| {
                if x > acc.1 {
                    (i, x)
                } else {
                    acc
                }
            })
            .0;
        assert_eq!(tok as usize, argmax);
        let p64 = (logits[argmax] - max).exp() / sum;
        assert!(
            (prob as f64 - p64).abs() < 1e-4,
            "draft prob {prob} vs f64 softmax {p64}"
        );
    }

    #[test]
    fn rescale_after_mass_pinned_by_planted_ascending_maxima() {
        // Verify-1 should-fix: pin the rescale-after-mass path
        // STRUCTURALLY, not incidentally.  Five planted rows with
        // strictly ascending dots sit deep into accumulation (tiles 0,
        // 1, 2, 3, 4 of a 640-token vocab), so the running max updates
        // five times AFTER substantial mass has accumulated - each
        // update must rescale the online sum or p(argmax) inflates by
        // ~exp(delta) per missed rescale, far outside the 1e-4 bound
        // against the independent f64 softmax below.
        let (dm, vocab) = (32usize, 640usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let mut w = lm_head(vocab, dm);
        let planted = [5usize, 150, 290, 400, 560];
        for (j, &tok) in planted.iter().enumerate() {
            let amp = 0.5 + j as f32 * 0.5; // strictly ascending dots
            for d in 0..dm {
                w[tok * dm + d] = f16r(if hidden[d] >= 0.0 { amp } else { -amp });
            }
        }
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(
            tok as usize,
            planted[planted.len() - 1],
            "the last (largest) planted max must win"
        );

        // Independent f64 pipeline (plain RMSNorm + softmax).
        let mut ss = 0f64;
        for &h in &hidden {
            ss += h as f64 * h as f64;
        }
        let rstd = 1.0 / (ss / dm as f64 + RMS_EPS as f64).sqrt();
        let h64: Vec<f64> = hidden
            .iter()
            .zip(&gamma)
            .map(|(h, g)| *h as f64 * rstd * *g as f64)
            .collect();
        let logits: Vec<f64> = (0..vocab)
            .map(|t| (0..dm).map(|d| w[t * dm + d] as f64 * h64[d]).sum())
            .collect();
        // The planted dots really are strictly ascending running maxima
        // (the fixture's premise - assert it so a weight tweak cannot
        // silently degrade this test back to zero rescales).
        let mut prev = f64::NEG_INFINITY;
        for &t in &planted {
            assert!(
                logits[t] > prev,
                "planted maxima must strictly ascend (token {t})"
            );
            let run_max = logits[..t]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                logits[t] > run_max,
                "token {t} must beat every earlier logit (a real max update)"
            );
            prev = logits[t];
        }
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|x| (x - max).exp()).sum();
        let p64 = (logits[tok as usize] - max).exp() / sum;
        assert!(
            (prob as f64 - p64).abs() < 1e-4,
            "planted-maxima prob {prob} vs f64 softmax {p64}"
        );

        // The verify row agrees on the same fixture (same online pass).
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert!((row[tok as usize] as f64 - p64).abs() < 1e-4);
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row must sum to ~1, got {total}");
    }

    #[test]
    fn cpu_verify_probs_row_matches_f64_softmax_and_normalizes() {
        let (dm, vocab) = (32usize, 300usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(row.len(), vocab);
        assert!(row.iter().all(|&p| p >= 0.0));
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row must sum to ~1, got {total}");

        // Row argmax == the draft sampler's token (same weights).
        let (tok, _) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        let row_argmax = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &p)| {
                if p > acc.1 {
                    (i, p)
                } else {
                    acc
                }
            })
            .0;
        assert_eq!(row_argmax, tok as usize);

        // Per-element agreement with the independent f64 softmax.
        let mut ss = 0f64;
        for &h in &hidden {
            ss += h as f64 * h as f64;
        }
        let rstd = 1.0 / (ss / dm as f64 + RMS_EPS as f64).sqrt();
        let h64: Vec<f64> = hidden
            .iter()
            .zip(&gamma)
            .map(|(h, g)| *h as f64 * rstd * *g as f64)
            .collect();
        let logits: Vec<f64> = (0..vocab)
            .map(|t| (0..dm).map(|d| w[t * dm + d] as f64 * h64[d]).sum())
            .collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|x| (x - max).exp()).sum();
        for (i, &p) in row.iter().enumerate() {
            let p64 = (logits[i] - max).exp() / sum;
            assert!(
                (p as f64 - p64).abs() < 1e-5,
                "row[{i}] = {p} vs f64 {p64}"
            );
        }
    }

    #[test]
    fn cpu_draft_prob_bitwise_equals_verify_row_at_token() {
        // The lossless self-speculation anchor: with draft == target
        // weights, p_draft == p_target[tok] BIT-FOR-BIT, so the reject
        // ratio is exactly 1.0 and every draft token accepts.
        let (dm, vocab) = (16usize, 200usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(
            prob.to_bits(),
            row[tok as usize].to_bits(),
            "p_draft must equal p_target[argmax] bitwise (draft == target)"
        );
    }

    #[test]
    fn verify_rows_feed_cpu_reference_reject_all_accept_self_speculation() {
        // Frozen-ABI cross-check: the rows this module's reference
        // produces are exactly what cfie_speculative_ptx's reject
        // reference consumes.  Self-speculation (draft == target)
        // must accept all K for EVERY seed.
        let (dm, vocab) = (8usize, 32usize);
        let c = cfg(dm as u32, vocab as u32);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let hiddens = [hidden_pattern(dm), (0..dm).map(|i| (i as f32 * 0.61).cos()).collect()];

        let mut target_probs = Vec::new();
        let mut draft_probs = Vec::new();
        let mut draft_tokens = Vec::new();
        for h in &hiddens {
            let (tok, p) = cpu_reference_draft_sample(&c, h, &gamma, &w);
            let row = cpu_reference_verify_probs(&c, h, &gamma, &w);
            assert_eq!(p.to_bits(), row[tok as usize].to_bits());
            target_probs.extend_from_slice(&row);
            draft_probs.push(p);
            draft_tokens.push(tok);
        }

        let rcfg = RejectionConfig {
            k_tokens: 2,
            vocab_size: vocab as u32,
            sm_version: 80,
        };
        for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
            let (acc, corr) =
                cpu_reference_reject(&rcfg, &target_probs, &draft_probs, &draft_tokens, seed);
            assert_eq!(acc, 2, "seed {seed}: self-speculation must accept all K");
            assert_eq!(corr, u32::MAX, "seed {seed}: all-accept sentinel expected");
        }
    }

    #[test]
    fn verify_rows_feed_cpu_reference_reject_rejection_path() {
        // Inflated draft probs force ratio < 1: the reject walk must
        // consume this module's rows without tripping its layout
        // asserts, and the Leviathan residual (p_target - p_draft on
        // the drafted token, clamped) must never resample that token.
        let (dm, vocab) = (8usize, 32usize);
        let c = cfg(dm as u32, vocab as u32);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let hiddens = [hidden_pattern(dm), (0..dm).map(|i| (i as f32 * 0.61).cos()).collect()];

        let mut target_probs = Vec::new();
        let mut draft_tokens = Vec::new();
        for h in &hiddens {
            let (tok, _) = cpu_reference_draft_sample(&c, h, &gamma, &w);
            target_probs.extend_from_slice(&cpu_reference_verify_probs(&c, h, &gamma, &w));
            draft_tokens.push(tok);
        }
        let draft_probs = [1.0f32, 1.0];
        assert!(
            target_probs[draft_tokens[0] as usize] < 1.0,
            "test premise: ratio at position 0 must be < 1"
        );

        let rcfg = RejectionConfig {
            k_tokens: 2,
            vocab_size: vocab as u32,
            sm_version: 80,
        };
        let mut saw_rejection = false;
        for seed in 0..64u64 {
            let (acc, corr) =
                cpu_reference_reject(&rcfg, &target_probs, &draft_probs, &draft_tokens, seed);
            assert!((0..=2).contains(&acc), "seed {seed}");
            if acc < 2 {
                saw_rejection = true;
                assert!(corr < vocab as u32, "seed {seed}: correction in vocab");
                assert_ne!(
                    corr, draft_tokens[acc as usize],
                    "seed {seed}: residual zeroes the drafted token"
                );
            }
        }
        assert!(saw_rejection, "ratio < 1 must reject for some seed");
    }

    #[test]
    fn cpu_draft_finds_max_in_tail_tile() {
        // vocab 130 = one full tile + a 2-wide tail; plant the max at
        // token 129 so the tail-tile guard path is the winner.
        let (dm, vocab) = (4usize, 130usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = vec![1.0f32; dm];
        let mut w = lm_head(vocab, dm);
        for d in 0..dm {
            w[129 * dm + d] = f16r(if hidden[d] >= 0.0 { 4.0 } else { -4.0 });
        }
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(tok, 129);
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(prob.to_bits(), row[129].to_bits());
    }

    #[test]
    fn cpu_draft_vocab_one_is_certain() {
        // Single-token vocab: p(argmax) must be exactly 1.0 (the
        // online sum is exactly ex2(0) == 1).
        let dm = 4usize;
        let c = cfg(dm as u32, 1);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(1, dm);
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(tok, 0);
        assert_eq!(prob.to_bits(), 1.0f32.to_bits());
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].to_bits(), 1.0f32.to_bits());
    }

    // -- ptxas validation (skips silently when no validator present) --

    #[test]
    fn ptxas_validates_draft_sample_paper_config() {
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                eprintln!("[skip] cfie draft-sample ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie draft-sample PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }

    #[test]
    fn ptxas_validates_verify_probs_paper_config() {
        let ptx = emit_verify_probs_ptx(&paper_cfg());
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                eprintln!("[skip] cfie verify-probs ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie verify-probs PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }
}
