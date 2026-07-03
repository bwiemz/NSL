//! CFIE Feature 2: fused decode-sample PTX emitter.
//!
//! The paper's decode-tail claim: the six-launch pipeline (RMSNorm,
//! LM-head matmul, softmax, top-k, top-p, multinomial) becomes ONE
//! kernel where the `[1, vocab]` logits tensor never touches HBM.
//! Only the sampled token id (4 bytes) is written back.
//!
//! Consumes the structured [`FusedSampleProgram`] built by
//! `cfie_fused_sample::emit_program` — the ops actually present drive
//! which sections are emitted (RmsNorm, Argmax vs SoftmaxTopK /
//! NucleusFilter / MultinomialSample).
//!
//! Launch shape: single CTA (grid = 1), 128 threads — the batch=1
//! latency path.  Algorithm:
//!   1. cooperative load of hidden `[1, d_model]` (f32) into SMEM;
//!   2. RMSNorm in SMEM when the program has the op (parallel SMEM
//!      reduction of the sum of squares, rsqrt(mean + eps), gamma);
//!   3. tile loop over vocab in chunks of 128: thread `t` owns row
//!      `tile_base + t`, computes dot(x, W[row]) with f16 loads +
//!      f32 accumulate, scales by the baked 1/temperature, applies
//!      the grammar bitmask hook when compiled in, stores to SMEM;
//!   4. thread 0 merges the tile into a k-entry candidate list in
//!      SMEM (replace-min insertion, serial — correctness first);
//!   5. thread 0: softmax over the k candidates, optional nucleus
//!      filter (insertion sort desc + cumulative cutoff at the baked
//!      top_p), multinomial via xorshift64* seeded from `rng_seed`.
//!      Greedy programs argmax the candidate list directly.
//!   6. the ONLY global store of the kernel writes the token id.
//!
//! Determinism: the PRNG is xorshift64* over the u64 seed param —
//! the sampled token is a pure function of (weights, hidden, seed),
//! which keeps the kernel M46-friendly (no curand state, no clock).
//!
//! `cpu_reference` mirrors the kernel's arithmetic order (same
//! strided partial sums + tree reduction, same fma dot order, same
//! replace-min/sort/walk tie-breaks, same xorshift64*).  The kernel
//! uses `rsqrt.approx` / `ex2.approx` where the CPU uses exact libm;
//! exact GPU parity is verified in a later GPU cycle.

use crate::cfie_fused_sample::{FusedSampleOp, FusedSampleProgram};
use std::fmt::Write;

/// Threads per CTA == vocab tile width (thread t owns row tile_base+t).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

/// Baked RMSNorm epsilon (paper stage 1).
const RMS_EPS: f32 = 1e-5;

pub const KERNEL_NAME: &str = "nsl_cfie_fused_sample";

pub fn kernel_name() -> &'static str {
    KERNEL_NAME
}

/// Compile-time configuration for the fused sampler kernel.
#[derive(Debug, Clone)]
pub struct FusedSampleKernelConfig {
    pub d_model: u32,
    pub vocab_size: u32,
    pub vocab_tile: u32,
    pub top_k: u32,
    pub sm_version: u32,
    /// Number of grammar DFA states; 0 = no grammar hook emitted.
    pub grammar_states: u32,
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct FusedSampleMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
}

/// Mirrors `gpu_specs::GpuSpec::ptx_version`: sm_100+ -> 8.6 (Blackwell),
/// sm_90+ -> 8.4 (Hopper wgmma/TMA), else 7.0 baseline.
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

fn has_op(program: &FusedSampleProgram, pred: impl Fn(&FusedSampleOp) -> bool) -> bool {
    program.ops.iter().any(pred)
}

/// Baked 1/temperature — taken from the program's MatmulTile epilogue.
fn temperature_recip(program: &FusedSampleProgram) -> f32 {
    program
        .ops
        .iter()
        .find_map(|op| match op {
            FusedSampleOp::MatmulTile {
                temperature_recip, ..
            } => Some(*temperature_recip),
            _ => None,
        })
        .expect("FusedSampleProgram has no MatmulTile op")
}

fn nucleus_top_p(program: &FusedSampleProgram) -> Option<f32> {
    program.ops.iter().find_map(|op| match op {
        FusedSampleOp::NucleusFilter { top_p } => Some(*top_p),
        _ => None,
    })
}

/// Emit the min-scan over the k-entry candidate list (thread 0 only).
/// Result: `%f_min` = min value, `%r_minpos` = its index; strict `<`
/// with first-min-wins — the CPU reference mirrors this tie-break.
fn emit_min_scan(w: &mut String, label: &str, k: u32, topk_val_off: u32) {
    writeln!(w, "    ld.shared.f32 %f_min, [%r_sbase+{}];", topk_val_off).unwrap();
    writeln!(w, "    mov.u32 %r_minpos, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 1;").unwrap();
    writeln!(w, "{}_LOOP:", label).unwrap();
    writeln!(w, "    setp.ge.u32 %p_c, %r_j, {};", k).unwrap();
    writeln!(w, "    @%p_c bra {}_DONE;", label).unwrap();
    writeln!(w, "    mul.lo.u32 %r_t3, %r_j, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t3, %r_t3, %r_sbase;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_t0, [%r_t3+{}];", topk_val_off).unwrap();
    writeln!(w, "    setp.lt.f32 %p_d, %f_t0, %f_min;").unwrap();
    writeln!(w, "    @!%p_d bra {}_NEXT;", label).unwrap();
    writeln!(w, "    mov.f32 %f_min, %f_t0;").unwrap();
    writeln!(w, "    mov.u32 %r_minpos, %r_j;").unwrap();
    writeln!(w, "{}_NEXT:", label).unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra {}_LOOP;", label).unwrap();
    writeln!(w, "{}_DONE:", label).unwrap();
}

/// Emit the fused decode-sample kernel for `program` under `cfg`.
pub fn emit(
    program: &FusedSampleProgram,
    cfg: &FusedSampleKernelConfig,
) -> (String, FusedSampleMeta) {
    assert_eq!(
        cfg.vocab_tile, TILE,
        "vocab_tile must be {} so thread t owns row tile_base+t",
        TILE
    );
    assert!(
        cfg.top_k >= 1 && cfg.top_k <= 64,
        "top_k must be in 1..=64 (serial candidate list in SMEM)"
    );
    assert!(
        cfg.d_model >= 1 && cfg.d_model <= 8192,
        "d_model must be in 1..=8192 (hidden state staged in static SMEM)"
    );
    assert!(cfg.vocab_size >= 1, "vocab_size must be >= 1");
    assert_eq!(
        program.shape.d_model, cfg.d_model,
        "program shape d_model mismatch with cfg"
    );
    assert_eq!(
        program.shape.vocab_size, cfg.vocab_size,
        "program shape vocab_size mismatch with cfg"
    );
    assert_eq!(
        program.shape.vocab_tile, cfg.vocab_tile,
        "program shape vocab_tile mismatch with cfg"
    );
    assert_eq!(
        program.params.top_k, cfg.top_k,
        "program params top_k mismatch with cfg"
    );

    let has_rms = has_op(program, |op| matches!(op, FusedSampleOp::RmsNorm));
    let greedy = has_op(program, |op| matches!(op, FusedSampleOp::Argmax));
    let top_p = nucleus_top_p(program);
    let inv_temp = temperature_recip(program);
    let grammar_hook = cfg.grammar_states > 0;

    let dm = cfg.d_model;
    let vocab = cfg.vocab_size;
    let k = cfg.top_k;
    // f16 LM-head row stride: W is [vocab, d_model] ROW-major, one
    // contiguous d_model-long f16 row per vocab entry.
    let w_row_bytes = dm as u64 * 2;
    // Grammar bitmask row: one bit per token, rows indexed by DFA state.
    let mask_row_bytes = vocab.div_ceil(8);

    // SMEM layout (f32 unless noted):
    //   [hidden: d_model][scores: TILE][topk_val: k][topk_idx: k u32][rstd: 1]
    // The scores region doubles as the RMSNorm reduction scratch.
    let scores_off = dm * 4;
    let topk_val_off = scores_off + TILE * 4;
    let topk_idx_off = topk_val_off + k * 4;
    let rms_off = topk_idx_off + k * 4;
    let smem_bytes = rms_off + 4;

    let neg_inf = f32_imm(f32::NEG_INFINITY);
    let zero = f32_imm(0.0);
    let log2e = f32_imm(std::f32::consts::LOG2_E);
    let inv_temp_imm = f32_imm(inv_temp);
    let inv_dm = f32_imm(1.0 / dm as f32);
    let eps = f32_imm(RMS_EPS);
    // r in [0,1): top 24 bits of the xorshift64* output over 2^24.
    let two_neg24 = f32_imm(1.0 / 16_777_216.0);

    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(w, "// {} - CFIE fused decode-sample (paper Feature 2).", KERNEL_NAME).unwrap();
    writeln!(
        w,
        "// One CTA, {} threads; logits stay in SMEM/registers, only the",
        BLOCK_DIM
    )
    .unwrap();
    writeln!(w, "// sampled token id (4 bytes) is written to HBM.").unwrap();
    writeln!(w, "// LM-head layout: f16 [vocab, d_model], ROW-major per vocab row.").unwrap();
    writeln!(w, "// Baked constants:").unwrap();
    writeln!(w, "//   d_model          = {}", dm).unwrap();
    writeln!(w, "//   vocab_size       = {}", vocab).unwrap();
    writeln!(w, "//   vocab_tile       = {}", TILE).unwrap();
    writeln!(w, "//   top_k            = {}", k).unwrap();
    writeln!(w, "//   temperature_recip= {} ({})", inv_temp, inv_temp_imm).unwrap();
    if let Some(tp) = top_p {
        writeln!(w, "//   top_p            = {} ({})", tp, f32_imm(tp)).unwrap();
    }
    if has_rms {
        writeln!(w, "//   rms_eps          = {} ({})", RMS_EPS, eps).unwrap();
    }
    if grammar_hook {
        writeln!(
            w,
            "//   grammar: {} states x {} mask bytes/row (1 bit/token)",
            cfg.grammar_states, mask_row_bytes
        )
        .unwrap();
    }
    writeln!(
        w,
        "// PRNG: xorshift64* over rng_seed - deterministic given seed (M46)."
    )
    .unwrap();
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", ptx_version_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".shared .align 4 .b8 cfie_sample_smem[{}];", smem_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", KERNEL_NAME).unwrap();
    writeln!(w, "    .param .u64 hidden_ptr,").unwrap();
    writeln!(w, "    .param .u64 norm_w_ptr,").unwrap();
    writeln!(w, "    .param .u64 lm_head_ptr,").unwrap();
    writeln!(w, "    .param .u64 out_token_ptr,").unwrap();
    writeln!(w, "    .param .u64 rng_seed,").unwrap();
    writeln!(w, "    .param .u64 grammar_mask_ptr,").unwrap();
    writeln!(w, "    .param .u32 grammar_state").unwrap();
    writeln!(w, ")").unwrap();
    writeln!(w, "{{").unwrap();
    writeln!(w, "    .reg .pred %p_a, %p_b, %p_c, %p_d, %p_t0;").unwrap();
    writeln!(w, "    .reg .b16 %h_w;").unwrap();
    writeln!(
        w,
        "    .reg .f32 %f_h, %f_w, %f_dot, %f_s, %f_ss, %f_rstd, %f_g, %f_min, %f_max, %f_p, %f_sum, %f_cum, %f_key, %f_tgt, %f_ks, %f_r, %f_t0, %f_t1;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_tid, %r_sbase, %r_i, %r_j, %r_d, %r_tile, %r_tok, %r_cnt, %r_minpos, %r_sel, %r_gstate, %r_kidx, %r_t0, %r_t1, %r_t2, %r_t3;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_hidden, %rd_norm, %rd_w, %rd_out, %rd_seed, %rd_mask, %rd_a, %rd_x, %rd_t0, %rd_t1;"
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_hidden, [hidden_ptr];").unwrap();
    writeln!(w, "    // ignored when the program lacks the RmsNorm op").unwrap();
    writeln!(w, "    ld.param.u64 %rd_norm, [norm_w_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_w, [lm_head_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_out, [out_token_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_seed, [rng_seed];").unwrap();
    writeln!(w, "    // 0 when no grammar; Phase B wires the live mask").unwrap();
    writeln!(w, "    ld.param.u64 %rd_mask, [grammar_mask_ptr];").unwrap();
    writeln!(w, "    ld.param.u32 %r_gstate, [grammar_state];").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_sbase, cfie_sample_smem;").unwrap();
    writeln!(w).unwrap();
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

    if has_rms {
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

    writeln!(w, "    // 3. candidate list init: threads t < k").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_tid, {};", k).unwrap();
    writeln!(w, "    @%p_a bra TK_INIT_DONE;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    mov.f32 %f_t0, {};", neg_inf).unwrap();
    writeln!(w, "    st.shared.f32 [%r_t0+{}], %f_t0;", topk_val_off).unwrap();
    writeln!(w, "    mov.u32 %r_t1, 0;").unwrap();
    writeln!(w, "    st.shared.u32 [%r_t0+{}], %r_t1;", topk_idx_off).unwrap();
    writeln!(w, "TK_INIT_DONE:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // 4. vocab tile loop: thread t scores row tile_base + t").unwrap();
    writeln!(w, "    mov.u32 %r_tile, 0;").unwrap();
    writeln!(w, "TILE_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_tile, {};", vocab).unwrap();
    writeln!(w, "    @%p_a bra TILES_DONE;").unwrap();
    writeln!(w, "    add.u32 %r_tok, %r_tile, %r_tid;").unwrap();
    writeln!(w, "    mov.f32 %f_s, {};", neg_inf).unwrap();
    writeln!(w, "    // tail-tile guard: lanes past vocab keep -inf").unwrap();
    writeln!(w, "    setp.ge.u32 %p_b, %r_tok, {};", vocab).unwrap();
    writeln!(w, "    @%p_b bra SCORE_STORE;").unwrap();
    writeln!(w, "    // f16 row: lm_head_ptr + tok * d_model * 2").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t0, %r_tok, {};", w_row_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_a, %rd_w, %rd_t0;").unwrap();
    writeln!(w, "    mov.f32 %f_dot, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_d, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_t1, %r_sbase;").unwrap();
    writeln!(w, "DOT_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_b, %r_d, {};", dm).unwrap();
    writeln!(w, "    @%p_b bra DOT_DONE;").unwrap();
    writeln!(w, "    ld.global.b16 %h_w, [%rd_a];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f_w, %h_w;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_h, [%r_t1];").unwrap();
    writeln!(w, "    fma.rn.f32 %f_dot, %f_w, %f_h, %f_dot;").unwrap();
    writeln!(w, "    add.u64 %rd_a, %rd_a, 2;").unwrap();
    writeln!(w, "    add.u32 %r_t1, %r_t1, 4;").unwrap();
    writeln!(w, "    add.u32 %r_d, %r_d, 1;").unwrap();
    writeln!(w, "    bra DOT_LOOP;").unwrap();
    writeln!(w, "DOT_DONE:").unwrap();
    writeln!(w, "    // baked temperature epilogue").unwrap();
    writeln!(w, "    mul.f32 %f_s, %f_dot, {};", inv_temp_imm).unwrap();

    if grammar_hook {
        writeln!(w, "    // grammar bitmask hook: bit (state, token) == 0 -> -inf").unwrap();
        writeln!(w, "    setp.eq.u64 %p_b, %rd_mask, 0;").unwrap();
        writeln!(w, "    @%p_b bra GRAMMAR_DONE;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t0, %r_gstate, {};", mask_row_bytes).unwrap();
        writeln!(w, "    shr.u32 %r_t2, %r_tok, 3;").unwrap();
        writeln!(w, "    add.u32 %r_t0, %r_t0, %r_t2;").unwrap();
        writeln!(w, "    cvt.u64.u32 %rd_t0, %r_t0;").unwrap();
        writeln!(w, "    add.u64 %rd_t1, %rd_mask, %rd_t0;").unwrap();
        writeln!(w, "    ld.global.u8 %r_t3, [%rd_t1];").unwrap();
        writeln!(w, "    and.b32 %r_t2, %r_tok, 7;").unwrap();
        writeln!(w, "    shr.u32 %r_t3, %r_t3, %r_t2;").unwrap();
        writeln!(w, "    and.b32 %r_t3, %r_t3, 1;").unwrap();
        writeln!(w, "    setp.ne.u32 %p_b, %r_t3, 0;").unwrap();
        writeln!(w, "    @%p_b bra GRAMMAR_DONE;").unwrap();
        writeln!(w, "    mov.f32 %f_s, {};", neg_inf).unwrap();
        writeln!(w, "GRAMMAR_DONE:").unwrap();
    }

    writeln!(w, "SCORE_STORE:").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t0+{}], %f_s;", scores_off).unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // 5. thread 0: replace-min merge of the tile into the list").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra MERGE_DONE;").unwrap();
    writeln!(w, "    sub.u32 %r_cnt, {}, %r_tile;", vocab).unwrap();
    writeln!(w, "    min.u32 %r_cnt, %r_cnt, {};", TILE).unwrap();
    emit_min_scan(w, "MS0", k, topk_val_off);
    writeln!(w, "    mov.u32 %r_i, 0;").unwrap();
    writeln!(w, "MERGE_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_i, %r_cnt;").unwrap();
    writeln!(w, "    @%p_a bra MERGE_DONE;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t0, %r_i, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_s, [%r_t0+{}];", scores_off).unwrap();
    writeln!(w, "    setp.gt.f32 %p_b, %f_s, %f_min;").unwrap();
    writeln!(w, "    @!%p_b bra MERGE_NEXT;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t1, %r_minpos, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t1, %r_t1, %r_sbase;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t1+{}], %f_s;", topk_val_off).unwrap();
    writeln!(w, "    add.u32 %r_t2, %r_tile, %r_i;").unwrap();
    writeln!(w, "    st.shared.u32 [%r_t1+{}], %r_t2;", topk_idx_off).unwrap();
    emit_min_scan(w, "MS1", k, topk_val_off);
    writeln!(w, "MERGE_NEXT:").unwrap();
    writeln!(w, "    add.u32 %r_i, %r_i, 1;").unwrap();
    writeln!(w, "    bra MERGE_LOOP;").unwrap();
    writeln!(w, "MERGE_DONE:").unwrap();
    writeln!(w, "    // scores SMEM is rewritten next tile; sync before loop back").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    add.u32 %r_tile, %r_tile, {};", TILE).unwrap();
    writeln!(w, "    bra TILE_LOOP;").unwrap();
    writeln!(w, "TILES_DONE:").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // 6. selection is serial on thread 0; others exit").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra EXIT;").unwrap();

    if greedy {
        writeln!(w, "    // greedy: argmax over the candidate list, no softmax/RNG").unwrap();
        writeln!(w, "    ld.shared.f32 %f_max, [%r_sbase+{}];", topk_val_off).unwrap();
        writeln!(w, "    ld.shared.u32 %r_sel, [%r_sbase+{}];", topk_idx_off).unwrap();
        writeln!(w, "    mov.u32 %r_j, 1;").unwrap();
        writeln!(w, "AM_LOOP:").unwrap();
        writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
        writeln!(w, "    @%p_a bra STORE_TOKEN;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_t0, [%r_t0+{}];", topk_val_off).unwrap();
        writeln!(w, "    setp.gt.f32 %p_b, %f_t0, %f_max;").unwrap();
        writeln!(w, "    @!%p_b bra AM_NEXT;").unwrap();
        writeln!(w, "    mov.f32 %f_max, %f_t0;").unwrap();
        writeln!(w, "    ld.shared.u32 %r_sel, [%r_t0+{}];", topk_idx_off).unwrap();
        writeln!(w, "AM_NEXT:").unwrap();
        writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
        writeln!(w, "    bra AM_LOOP;").unwrap();
    } else {
        writeln!(w, "    // stable-softmax max over the k candidates").unwrap();
        writeln!(w, "    ld.shared.f32 %f_max, [%r_sbase+{}];", topk_val_off).unwrap();
        writeln!(w, "    mov.u32 %r_j, 1;").unwrap();
        writeln!(w, "MX_LOOP:").unwrap();
        writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
        writeln!(w, "    @%p_a bra MX_DONE;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_t0, [%r_t0+{}];", topk_val_off).unwrap();
        writeln!(w, "    setp.gt.f32 %p_b, %f_t0, %f_max;").unwrap();
        writeln!(w, "    @!%p_b bra MX_NEXT;").unwrap();
        writeln!(w, "    mov.f32 %f_max, %f_t0;").unwrap();
        writeln!(w, "MX_NEXT:").unwrap();
        writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
        writeln!(w, "    bra MX_LOOP;").unwrap();
        writeln!(w, "MX_DONE:").unwrap();
        writeln!(w, "    // softmax over k only (not vocab): p = exp2((v - max) * log2e)").unwrap();
        writeln!(w, "    mov.f32 %f_sum, {};", zero).unwrap();
        writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
        writeln!(w, "SM_LOOP:").unwrap();
        writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
        writeln!(w, "    @%p_a bra SM_DONE;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_t0, [%r_t0+{}];", topk_val_off).unwrap();
        writeln!(w, "    sub.f32 %f_t0, %f_t0, %f_max;").unwrap();
        writeln!(w, "    mul.f32 %f_t0, %f_t0, {};", log2e).unwrap();
        writeln!(w, "    ex2.approx.f32 %f_t0, %f_t0;").unwrap();
        writeln!(w, "    st.shared.f32 [%r_t0+{}], %f_t0;", topk_val_off).unwrap();
        writeln!(w, "    add.f32 %f_sum, %f_sum, %f_t0;").unwrap();
        writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
        writeln!(w, "    bra SM_LOOP;").unwrap();
        writeln!(w, "SM_DONE:").unwrap();

        if let Some(tp) = top_p {
            writeln!(w, "    // nucleus: insertion sort desc (stable, strict-lt shift)").unwrap();
            writeln!(w, "    mov.u32 %r_j, 1;").unwrap();
            writeln!(w, "SORT_OUTER:").unwrap();
            writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
            writeln!(w, "    @%p_a bra SORT_DONE;").unwrap();
            writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
            writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
            writeln!(w, "    ld.shared.f32 %f_key, [%r_t0+{}];", topk_val_off).unwrap();
            writeln!(w, "    ld.shared.u32 %r_kidx, [%r_t0+{}];", topk_idx_off).unwrap();
            writeln!(w, "    mov.u32 %r_i, %r_j;").unwrap();
            writeln!(w, "SORT_INNER:").unwrap();
            writeln!(w, "    setp.eq.u32 %p_b, %r_i, 0;").unwrap();
            writeln!(w, "    @%p_b bra SORT_PLACE;").unwrap();
            writeln!(w, "    sub.u32 %r_t1, %r_i, 1;").unwrap();
            writeln!(w, "    mul.lo.u32 %r_t1, %r_t1, 4;").unwrap();
            writeln!(w, "    add.u32 %r_t1, %r_t1, %r_sbase;").unwrap();
            writeln!(w, "    ld.shared.f32 %f_t0, [%r_t1+{}];", topk_val_off).unwrap();
            writeln!(w, "    setp.lt.f32 %p_c, %f_t0, %f_key;").unwrap();
            writeln!(w, "    @!%p_c bra SORT_PLACE;").unwrap();
            writeln!(w, "    ld.shared.u32 %r_t2, [%r_t1+{}];", topk_idx_off).unwrap();
            writeln!(w, "    mul.lo.u32 %r_t3, %r_i, 4;").unwrap();
            writeln!(w, "    add.u32 %r_t3, %r_t3, %r_sbase;").unwrap();
            writeln!(w, "    st.shared.f32 [%r_t3+{}], %f_t0;", topk_val_off).unwrap();
            writeln!(w, "    st.shared.u32 [%r_t3+{}], %r_t2;", topk_idx_off).unwrap();
            writeln!(w, "    sub.u32 %r_i, %r_i, 1;").unwrap();
            writeln!(w, "    bra SORT_INNER;").unwrap();
            writeln!(w, "SORT_PLACE:").unwrap();
            writeln!(w, "    mul.lo.u32 %r_t3, %r_i, 4;").unwrap();
            writeln!(w, "    add.u32 %r_t3, %r_t3, %r_sbase;").unwrap();
            writeln!(w, "    st.shared.f32 [%r_t3+{}], %f_key;", topk_val_off).unwrap();
            writeln!(w, "    st.shared.u32 [%r_t3+{}], %r_kidx;", topk_idx_off).unwrap();
            writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
            writeln!(w, "    bra SORT_OUTER;").unwrap();
            writeln!(w, "SORT_DONE:").unwrap();
            writeln!(w, "    // cumulative prob until > top_p (crossing entry kept);").unwrap();
            writeln!(w, "    // baked top_p immediate, tail zeroed").unwrap();
            writeln!(w, "    mov.f32 %f_cum, {};", zero).unwrap();
            writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
            writeln!(w, "NUC_LOOP:").unwrap();
            writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
            writeln!(w, "    @%p_a bra NUC_DONE;").unwrap();
            writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
            writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
            writeln!(w, "    ld.shared.f32 %f_p, [%r_t0+{}];", topk_val_off).unwrap();
            writeln!(w, "    div.rn.f32 %f_t0, %f_p, %f_sum;").unwrap();
            writeln!(w, "    add.f32 %f_cum, %f_cum, %f_t0;").unwrap();
            writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
            writeln!(w, "    setp.gt.f32 %p_b, %f_cum, {};", f32_imm(tp)).unwrap();
            writeln!(w, "    @!%p_b bra NUC_LOOP;").unwrap();
            writeln!(w, "    mov.f32 %f_t1, {};", zero).unwrap();
            writeln!(w, "ZERO_LOOP:").unwrap();
            writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
            writeln!(w, "    @%p_a bra NUC_DONE;").unwrap();
            writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
            writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
            writeln!(w, "    st.shared.f32 [%r_t0+{}], %f_t1;", topk_val_off).unwrap();
            writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
            writeln!(w, "    bra ZERO_LOOP;").unwrap();
            writeln!(w, "NUC_DONE:").unwrap();
        }

        writeln!(w, "    // kept probability mass (== sum when no nucleus filter)").unwrap();
        writeln!(w, "    mov.f32 %f_ks, {};", zero).unwrap();
        writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
        writeln!(w, "KS_LOOP:").unwrap();
        writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
        writeln!(w, "    @%p_a bra KS_DONE;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_t0, [%r_t0+{}];", topk_val_off).unwrap();
        writeln!(w, "    add.f32 %f_ks, %f_ks, %f_t0;").unwrap();
        writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
        writeln!(w, "    bra KS_LOOP;").unwrap();
        writeln!(w, "KS_DONE:").unwrap();
        writeln!(w, "    // xorshift64* PRNG: deterministic given rng_seed (M46)").unwrap();
        writeln!(w, "    mov.u64 %rd_x, %rd_seed;").unwrap();
        writeln!(w, "    setp.ne.u64 %p_a, %rd_x, 0;").unwrap();
        writeln!(w, "    @%p_a bra RNG_MIX;").unwrap();
        writeln!(w, "    // zero seed would be a fixed point; substitute golden gamma").unwrap();
        writeln!(w, "    mov.u64 %rd_x, 0x9E3779B97F4A7C15;").unwrap();
        writeln!(w, "RNG_MIX:").unwrap();
        writeln!(w, "    shr.b64 %rd_t0, %rd_x, 12;").unwrap();
        writeln!(w, "    xor.b64 %rd_x, %rd_x, %rd_t0;").unwrap();
        writeln!(w, "    shl.b64 %rd_t0, %rd_x, 25;").unwrap();
        writeln!(w, "    xor.b64 %rd_x, %rd_x, %rd_t0;").unwrap();
        writeln!(w, "    shr.b64 %rd_t0, %rd_x, 27;").unwrap();
        writeln!(w, "    xor.b64 %rd_x, %rd_x, %rd_t0;").unwrap();
        writeln!(w, "    mov.u64 %rd_t0, 0x2545F4914F6CDD1D;").unwrap();
        writeln!(w, "    mul.lo.u64 %rd_x, %rd_x, %rd_t0;").unwrap();
        writeln!(w, "    // r in [0,1): top 24 bits over 2^24 (f32 mantissa exact)").unwrap();
        writeln!(w, "    shr.b64 %rd_x, %rd_x, 40;").unwrap();
        writeln!(w, "    cvt.u32.u64 %r_t0, %rd_x;").unwrap();
        writeln!(w, "    cvt.rn.f32.u32 %f_r, %r_t0;").unwrap();
        writeln!(w, "    mul.f32 %f_r, %f_r, {};", two_neg24).unwrap();
        writeln!(w, "    mul.f32 %f_tgt, %f_r, %f_ks;").unwrap();
        writeln!(w, "    // multinomial: walk the cumulative distribution").unwrap();
        writeln!(w, "    ld.shared.u32 %r_sel, [%r_sbase+{}];", topk_idx_off).unwrap();
        writeln!(w, "    mov.f32 %f_cum, {};", zero).unwrap();
        writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
        writeln!(w, "WALK_LOOP:").unwrap();
        writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
        writeln!(w, "    @%p_a bra STORE_TOKEN;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t0, %r_j, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t0, %r_t0, %r_sbase;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_p, [%r_t0+{}];", topk_val_off).unwrap();
        writeln!(w, "    add.f32 %f_cum, %f_cum, %f_p;").unwrap();
        writeln!(w, "    // zero-prob entries never selected; last live entry is").unwrap();
        writeln!(w, "    // the fp-drift fallback").unwrap();
        writeln!(w, "    setp.gt.f32 %p_b, %f_p, {};", zero).unwrap();
        writeln!(w, "    @!%p_b bra WALK_NEXT;").unwrap();
        writeln!(w, "    ld.shared.u32 %r_sel, [%r_t0+{}];", topk_idx_off).unwrap();
        writeln!(w, "    setp.ge.f32 %p_c, %f_cum, %f_tgt;").unwrap();
        writeln!(w, "    @%p_c bra STORE_TOKEN;").unwrap();
        writeln!(w, "WALK_NEXT:").unwrap();
        writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
        writeln!(w, "    bra WALK_LOOP;").unwrap();
    }

    writeln!(w, "STORE_TOKEN:").unwrap();
    writeln!(w, "    // the ONLY global store of the kernel").unwrap();
    writeln!(w, "    st.global.u32 [%rd_out], %r_sel;").unwrap();
    writeln!(w, "EXIT:").unwrap();
    writeln!(w, "    ret;").unwrap();
    writeln!(w, "}}").unwrap();

    let meta = FusedSampleMeta {
        kernel_name: KERNEL_NAME.to_string(),
        smem_bytes,
        block_dim: BLOCK_DIM,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit`].
pub fn emit_fused_sample_ptx(
    program: &FusedSampleProgram,
    cfg: &FusedSampleKernelConfig,
) -> String {
    emit(program, cfg).0
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// xorshift64* — the kernel's PRNG, bit-for-bit.
fn xorshift64star(seed: u64) -> u64 {
    // Zero seed is a fixed point of xorshift; the kernel substitutes the
    // golden-gamma constant, so mirror that here.
    let mut x = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn min_scan(vals: &[f32]) -> (f32, usize) {
    let mut mv = vals[0];
    let mut mp = 0usize;
    for (j, &v) in vals.iter().enumerate().skip(1) {
        if v < mv {
            mv = v;
            mp = j;
        }
    }
    (mv, mp)
}

/// CPU mirror of the fused kernel.  `lm_head_f32` is `[vocab][d_model]`
/// row-major, f32 (an exact-value cast of the kernel's f16 weights).
/// `grammar_mask` is `(bit rows, current DFA state)` — bit
/// `rows[state * ceil(vocab/8) + token/8] >> (token % 8)` gates the
/// token, matching the kernel's hook.
///
/// Mirrors the kernel exactly: same strided partial sums + tree
/// reduction for RMSNorm, same fma dot order, same replace-min /
/// argmax / sort / walk tie-breaks, same xorshift64*.  The kernel's
/// `rsqrt.approx` / `ex2.approx` are approximate where this uses exact
/// libm — exact GPU parity is verified in a later GPU cycle.
pub fn cpu_reference(
    program: &FusedSampleProgram,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
    grammar_mask: Option<(&[u8], u32)>,
    seed: u64,
) -> u32 {
    let dm = program.shape.d_model as usize;
    let vocab = program.shape.vocab_size as usize;
    let k = program.params.top_k as usize;
    assert!((1..=64).contains(&k), "top_k must be in 1..=64");
    assert!((1..=8192).contains(&dm), "d_model must be in 1..=8192");
    assert_eq!(program.shape.vocab_tile, TILE, "vocab_tile must be {TILE}");
    assert_eq!(hidden.len(), dm, "hidden must be [1, d_model]");
    assert_eq!(
        lm_head_f32.len(),
        vocab * dm,
        "lm_head must be [vocab][d_model] row-major"
    );

    let has_rms = has_op(program, |op| matches!(op, FusedSampleOp::RmsNorm));
    let greedy = has_op(program, |op| matches!(op, FusedSampleOp::Argmax));
    let top_p = nucleus_top_p(program);
    let inv_temp = temperature_recip(program);

    let mut h = hidden.to_vec();
    if has_rms {
        assert_eq!(norm_w.len(), dm, "norm_w (gamma) must be [d_model]");
        // Per-thread strided partial sums, then the kernel's SMEM tree.
        let mut partials = [0f32; TILE as usize];
        for (t, part) in partials.iter_mut().enumerate() {
            let mut s = 0f32;
            let mut i = t;
            while i < dm {
                s = h[i].mul_add(h[i], s);
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
        for (i, hv) in h.iter_mut().enumerate() {
            *hv = (*hv * rstd) * norm_w[i];
        }
    }

    let mask_row_bytes = vocab.div_ceil(8);
    let mut tv = vec![f32::NEG_INFINITY; k];
    let mut ti = vec![0u32; k];
    let mut tile = 0usize;
    while tile < vocab {
        let cnt = (vocab - tile).min(TILE as usize);
        let mut scores = vec![f32::NEG_INFINITY; cnt];
        for (t, score) in scores.iter_mut().enumerate() {
            let tok = tile + t;
            let row = &lm_head_f32[tok * dm..(tok + 1) * dm];
            let mut dot = 0f32;
            for d in 0..dm {
                dot = row[d].mul_add(h[d], dot);
            }
            let mut s = dot * inv_temp;
            if let Some((mask, state)) = grammar_mask {
                let byte = mask[state as usize * mask_row_bytes + tok / 8];
                if (byte >> (tok & 7)) & 1 == 0 {
                    s = f32::NEG_INFINITY;
                }
            }
            *score = s;
        }
        // Thread-0 replace-min merge, identical insertion order.
        let (mut mv, mut mp) = min_scan(&tv);
        for (i, &s) in scores.iter().enumerate() {
            if s > mv {
                tv[mp] = s;
                ti[mp] = (tile + i) as u32;
                let r = min_scan(&tv);
                mv = r.0;
                mp = r.1;
            }
        }
        tile += TILE as usize;
    }

    if greedy {
        let mut mx = tv[0];
        let mut sel = ti[0];
        for j in 1..k {
            if tv[j] > mx {
                mx = tv[j];
                sel = ti[j];
            }
        }
        return sel;
    }

    // Softmax over the k candidates only.
    let mut mx = tv[0];
    for j in 1..k {
        if tv[j] > mx {
            mx = tv[j];
        }
    }
    let mut sum = 0f32;
    for v in tv.iter_mut() {
        let p = ((*v - mx) * std::f32::consts::LOG2_E).exp2();
        *v = p;
        sum += p;
    }

    if let Some(tp) = top_p {
        // Insertion sort descending (stable: shift only while strictly
        // smaller than the key).
        for j in 1..k {
            let key = tv[j];
            let kidx = ti[j];
            let mut i = j;
            while i > 0 && tv[i - 1] < key {
                tv[i] = tv[i - 1];
                ti[i] = ti[i - 1];
                i -= 1;
            }
            tv[i] = key;
            ti[i] = kidx;
        }
        // Cumulative prob until > top_p; the crossing entry is kept,
        // the tail is zeroed.
        let mut cum = 0f32;
        let mut j = 0usize;
        while j < k {
            cum += tv[j] / sum;
            j += 1;
            if cum > tp {
                break;
            }
        }
        for z in tv.iter_mut().skip(j) {
            *z = 0.0;
        }
    }

    let mut ks = 0f32;
    for &v in tv.iter() {
        ks += v;
    }
    let r = ((xorshift64star(seed) >> 40) as u32) as f32 * (1.0 / 16_777_216.0);
    let target = r * ks;

    let mut sel = ti[0];
    let mut cum = 0f32;
    for j in 0..k {
        let p = tv[j];
        cum += p;
        if p > 0.0 {
            sel = ti[j];
            if cum >= target {
                break;
            }
        }
    }
    sel
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_fused_sample::{
        emit_program, LmHeadShape, SamplingParams, SamplingStrategy,
    };

    fn shape(d_model: u32, vocab_size: u32) -> LmHeadShape {
        LmHeadShape {
            d_model,
            vocab_size,
            vocab_tile: 128,
            dtype_bytes: 2,
        }
    }

    /// Reference config from the CFIE paper's NSL-Coder example.
    fn paper_cfg() -> FusedSampleKernelConfig {
        FusedSampleKernelConfig {
            d_model: 512,
            vocab_size: 49_152,
            vocab_tile: 128,
            top_k: 50,
            sm_version: 80,
            grammar_states: 0,
        }
    }

    fn paper_program() -> crate::cfie_fused_sample::FusedSampleProgram {
        emit_program(SamplingParams::default(), shape(512, 49_152))
    }

    // ── structural ─────────────────────────────────────────────────

    #[test]
    fn param_list_is_exactly_the_seven_params() {
        let ptx = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        let start = ptx.find(".visible .entry nsl_cfie_fused_sample(").unwrap();
        let end = start + ptx[start..].find(')').unwrap();
        let params: Vec<&str> = ptx[start..end]
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                l.starts_with(".param").then(|| l.trim_end_matches(','))
            })
            .collect();
        assert_eq!(
            params,
            vec![
                ".param .u64 hidden_ptr",
                ".param .u64 norm_w_ptr",
                ".param .u64 lm_head_ptr",
                ".param .u64 out_token_ptr",
                ".param .u64 rng_seed",
                ".param .u64 grammar_mask_ptr",
                ".param .u32 grammar_state",
            ]
        );
        assert_eq!(ptx.matches("ld.param").count(), 7);
    }

    #[test]
    fn exactly_one_global_store() {
        // The [1, vocab] logits never touch HBM: the token id write is
        // the kernel's only global store.
        let ptx = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert_eq!(ptx.matches("st.global").count(), 1);
        assert!(ptx.contains("st.global.u32 [%rd_out], %r_sel;"));
        // Greedy variant too.
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let ptx = emit_fused_sample_ptx(&prog, &paper_cfg());
        assert_eq!(ptx.matches("st.global").count(), 1);
    }

    #[test]
    fn no_mad_lo_and_ascii_only() {
        let ptx = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
        assert!(
            ptx.bytes().all(|b| b < 128),
            "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
        );
    }

    #[test]
    fn baked_temperature_and_top_p_immediates_present() {
        let ptx = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        // Default temperature 0.7 -> baked 1/0.7 epilogue multiplier.
        let inv_temp = f32_imm(1.0f32 / 0.7f32);
        assert!(ptx.contains(&format!("mul.f32 %f_s, %f_dot, {};", inv_temp)));
        // Default top_p 0.9 -> baked nucleus threshold.
        let top_p = f32_imm(0.9f32);
        assert!(ptx.contains(&format!("setp.gt.f32 %p_b, %f_cum, {};", top_p)));
        // RMSNorm epsilon baked (default program has the op).
        assert!(ptx.contains(&f32_imm(1e-5)));
    }

    #[test]
    fn grammar_hook_present_only_when_grammar_states_positive() {
        let no_grammar = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(!no_grammar.contains("ld.global.u8"));
        assert!(!no_grammar.contains("GRAMMAR_DONE"));

        let params = SamplingParams {
            grammar_masked: true,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let mut cfg = paper_cfg();
        cfg.grammar_states = 4;
        let ptx = emit_fused_sample_ptx(&prog, &cfg);
        assert!(ptx.contains("ld.global.u8"));
        // Baked mask row stride: ceil(49152 / 8) = 6144 bytes/state.
        assert!(ptx.contains("mul.lo.u32 %r_t0, %r_gstate, 6144;"));
        // Runtime null-ptr guard keeps the hook inert until Phase B.
        assert!(ptx.contains("setp.eq.u64 %p_b, %rd_mask, 0;"));
    }

    #[test]
    fn greedy_program_skips_softmax_sort_and_rng() {
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let ptx = emit_fused_sample_ptx(&prog, &paper_cfg());
        assert!(!ptx.contains("ex2.approx"));
        assert!(!ptx.contains("SORT_OUTER"));
        assert!(!ptx.contains("RNG_MIX"));
        assert!(ptx.contains("AM_LOOP"));

        let full = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(full.contains("ex2.approx"));
        assert!(full.contains("SORT_OUTER"));
        assert!(full.contains("RNG_MIX"));
    }

    #[test]
    fn rms_norm_section_gated_on_program_op() {
        let with_rms = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(with_rms.contains("rsqrt.approx.f32"));

        let params = SamplingParams {
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let ptx = emit_fused_sample_ptx(&prog, &paper_cfg());
        assert!(!ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn header_matches_sm_version_convention() {
        let ptx80 = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(ptx80.starts_with("//"));
        assert!(ptx80.contains(".version 7.0\n.target sm_80\n.address_size 64"));
        let mut cfg = paper_cfg();
        cfg.sm_version = 90;
        assert!(emit_fused_sample_ptx(&paper_program(), &cfg)
            .contains(".version 8.4\n.target sm_90"));
        cfg.sm_version = 100;
        assert!(emit_fused_sample_ptx(&paper_program(), &cfg)
            .contains(".version 8.6\n.target sm_100"));
    }

    #[test]
    fn meta_reports_launch_shape() {
        let (ptx, meta) = emit(&paper_program(), &paper_cfg());
        assert_eq!(meta.kernel_name, kernel_name());
        assert_eq!(meta.block_dim, 128);
        // hidden(512 f32) + scores(128 f32) + topk_val(50) + topk_idx(50) + rstd.
        assert_eq!(meta.smem_bytes, 512 * 4 + 128 * 4 + 50 * 4 + 50 * 4 + 4);
        assert!(ptx.contains(&format!(
            ".shared .align 4 .b8 cfie_sample_smem[{}];",
            meta.smem_bytes
        )));
    }

    #[test]
    #[should_panic(expected = "vocab_tile")]
    fn vocab_tile_not_128_panics() {
        let mut cfg = paper_cfg();
        cfg.vocab_tile = 256;
        let prog = emit_program(
            SamplingParams::default(),
            LmHeadShape {
                d_model: 512,
                vocab_size: 49_152,
                vocab_tile: 256,
                dtype_bytes: 2,
            },
        );
        let _ = emit(&prog, &cfg);
    }

    #[test]
    #[should_panic(expected = "top_k")]
    fn top_k_over_64_panics() {
        let mut cfg = paper_cfg();
        cfg.top_k = 65;
        let params = SamplingParams {
            top_k: 65,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let _ = emit(&prog, &cfg);
    }

    #[test]
    #[should_panic(expected = "d_model")]
    fn d_model_over_8192_panics() {
        let mut cfg = paper_cfg();
        cfg.d_model = 8193;
        let prog = emit_program(SamplingParams::default(), shape(8193, 49_152));
        let _ = emit(&prog, &cfg);
    }

    #[test]
    #[should_panic(expected = "mismatch")]
    fn program_shape_mismatch_panics() {
        // cfg says d_model 512, program says 256.
        let prog = emit_program(SamplingParams::default(), shape(256, 49_152));
        let _ = emit(&prog, &paper_cfg());
    }

    // ── cpu_reference ──────────────────────────────────────────────

    fn small_weights(vocab: usize, dm: usize) -> Vec<f32> {
        (0..vocab * dm)
            .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1 - 0.5)
            .collect()
    }

    #[test]
    fn cpu_reference_is_deterministic_for_fixed_seed() {
        let (dm, vocab) = (16usize, 256u32);
        let prog = emit_program(SamplingParams::default(), shape(dm as u32, vocab));
        let hidden: Vec<f32> = (0..dm).map(|i| (i as f32 * 0.37).sin()).collect();
        let gamma: Vec<f32> = (0..dm).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let w = small_weights(vocab as usize, dm);
        let t1 = cpu_reference(&prog, &hidden, &gamma, &w, None, 0xDEAD_BEEF);
        let t2 = cpu_reference(&prog, &hidden, &gamma, &w, None, 0xDEAD_BEEF);
        assert_eq!(t1, t2);
        assert!(t1 < vocab);
    }

    #[test]
    fn cpu_reference_greedy_matches_naive_argmax() {
        let (dm, vocab) = (4usize, 10usize);
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: 10,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab as u32));
        let hidden = [0.3f32, -1.2, 0.7, 0.05];
        let mut w = small_weights(vocab, dm);
        // Force a unique max at row 6: align the row with hidden's signs.
        for d in 0..dm {
            w[6 * dm + d] = if hidden[d] >= 0.0 { 5.0 } else { -5.0 };
        }
        let naive = (0..vocab)
            .map(|t| {
                (0..dm)
                    .map(|d| w[t * dm + d] * hidden[d])
                    .sum::<f32>()
            })
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (t, s)| {
                if s > acc.1 {
                    (t, s)
                } else {
                    acc
                }
            })
            .0;
        assert_eq!(naive, 6);
        for seed in [0u64, 1, 42, u64::MAX] {
            let tok = cpu_reference(&prog, &hidden, &[], &w, None, seed);
            assert_eq!(tok as usize, naive, "greedy must ignore the seed");
        }
    }

    #[test]
    fn cpu_reference_grammar_mask_excludes_tokens() {
        let (dm, vocab) = (2usize, 8u32);
        let params = SamplingParams {
            top_k: 4,
            grammar_masked: true,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab));
        let hidden = [0.9f32, -0.4];
        let w = small_weights(vocab as usize, dm);
        // One mask row (state 0), only token 5 allowed.
        let mask = [0b0010_0000u8];
        for seed in 0..32u64 {
            let tok = cpu_reference(&prog, &hidden, &[], &w, Some((&mask, 0)), seed);
            assert_eq!(tok, 5, "grammar mask must exclude every other token");
        }
    }

    #[test]
    fn cpu_reference_top_p_excludes_tail() {
        // Dominant logit: p(token 0) ~= 0.9999 > top_p = 0.5, so the
        // nucleus keeps only token 0 for every seed.
        let (dm, vocab) = (1usize, 4u32);
        let params = SamplingParams {
            strategy: SamplingStrategy::TopKTopP,
            temperature: 1.0,
            top_k: 4,
            top_p: 0.5,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab));
        let hidden = [1.0f32];
        let w = [10.0f32, 0.0, 0.0, 0.0];
        for seed in 0..64u64 {
            let tok = cpu_reference(&prog, &hidden, &[], &w, None, seed);
            assert_eq!(tok, 0, "nucleus tail must never be sampled");
        }
    }

    #[test]
    fn cpu_reference_uniform_gamma_preserves_greedy_argmax() {
        // RMSNorm with gamma == 1 is a uniform positive rescale of the
        // hidden state, so the greedy argmax is invariant.
        let (dm, vocab) = (8usize, 12u32);
        let base = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: 12,
            ..Default::default()
        };
        let hidden: Vec<f32> = (0..dm).map(|i| (i as f32 * 0.61).cos()).collect();
        let mut w = small_weights(vocab as usize, dm);
        for d in 0..dm {
            w[9 * dm + d] = 2.0; // unique max at row 9
        }
        let gamma = vec![1.0f32; dm];

        let prog_on = emit_program(base, shape(dm as u32, vocab));
        let prog_off = emit_program(
            SamplingParams {
                rms_norm: false,
                ..base
            },
            shape(dm as u32, vocab),
        );
        let on = cpu_reference(&prog_on, &hidden, &gamma, &w, None, 7);
        let off = cpu_reference(&prog_off, &hidden, &[], &w, None, 7);
        assert_eq!(on, off);
    }

    #[test]
    fn cpu_reference_multinomial_spreads_over_seeds() {
        // Uniform logits: different seeds must be able to pick
        // different tokens.
        let (dm, vocab) = (1usize, 4u32);
        let params = SamplingParams {
            strategy: SamplingStrategy::TopK,
            temperature: 1.0,
            top_k: 4,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab));
        let hidden = [1.0f32];
        let w = [0.0f32; 4];
        let mut seen = std::collections::BTreeSet::new();
        for seed in 1..=32u64 {
            seen.insert(cpu_reference(&prog, &hidden, &[], &w, None, seed));
        }
        assert!(seen.len() > 1, "multinomial must not collapse to one token");
        assert!(seen.iter().all(|&t| t < vocab));
    }

    // ── ptxas validation (skips silently when no validator present) ──

    #[test]
    fn ptxas_validates_paper_config() {
        let mut checks = vec![(paper_program(), paper_cfg())];
        // Grammar-hook variant exercises the mask address arithmetic.
        let params = SamplingParams {
            grammar_masked: true,
            ..Default::default()
        };
        let mut gcfg = paper_cfg();
        gcfg.grammar_states = 4;
        checks.push((emit_program(params, shape(512, 49_152)), gcfg));

        for (prog, cfg) in checks {
            let ptx = emit_fused_sample_ptx(&prog, &cfg);
            match crate::ptxas_validation::validate_ptx(&ptx) {
                Ok(()) => {}
                Err(msg) if msg.contains("nvcc not available") => {
                    eprintln!(
                        "[skip] cfie fused-sample ptxas validation - no validator: {msg}"
                    );
                }
                Err(msg) => panic!(
                    "cfie fused-sample PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
                ),
            }
        }
    }
}
