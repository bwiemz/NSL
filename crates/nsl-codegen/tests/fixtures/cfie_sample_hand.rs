// Frozen fixture (roadmap A2 step 9): the hand-written PTX emitter for the
// CFIE fused decode-sample kernel, exactly as `crates/nsl-codegen/src/
// cfie_sample_ptx.rs` held it at 250aeeb2, before the kernel moved onto KIR.
// Lines below this comment are that file's first 659 lines, unedited;
// `cfie_sample_kir_equivalence.rs` includes it with `#[path]` (supplying
// the two `crate::` paths it names) and runs its output against the KIR
// kernel's. Do not change it: it is the pre-migration behaviour the
// equivalence claim is about.
//
// Not scanned by the hand-PTX freeze: nothing under `tests/` is.

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
    writeln!(w, ".version {}", crate::gpu_specs::ptx_isa_for_sm(cfg.sm_version)).unwrap();
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
