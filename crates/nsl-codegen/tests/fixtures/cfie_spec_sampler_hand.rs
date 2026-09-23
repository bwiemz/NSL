// Frozen fixture (roadmap A2 step 9): the hand-written PTX emitter for the
// CFIE draft-sampler and verify-prob-row kernels, exactly as `crates/
// nsl-codegen/src/cfie_spec_sampler_ptx.rs` held it at c1640243, before the
// kernels moved onto KIR. Lines below this comment are that file's first
// 558 lines, unedited; `cfie_spec_sampler_kir_equivalence.rs` includes it
// with `#[path]` (supplying the one `crate::` item it names) and runs its
// output against the KIR kernels'. Do not change it: it is the
// pre-migration behaviour the equivalence claim is about.
//
// Not scanned by the hand-PTX freeze: nothing under `tests/` is.

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
    writeln!(w, ".version {}", crate::gpu_specs::ptx_isa_for_sm(cfg.sm_version)).unwrap();
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
    writeln!(w, ".version {}", crate::gpu_specs::ptx_isa_for_sm(cfg.sm_version)).unwrap();
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
