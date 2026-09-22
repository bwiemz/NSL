// Frozen fixture (roadmap A2 step 9): the hand-written PTX emitter for the
// CFIE decode-attention kernel, exactly as `crates/nsl-codegen/src/
// cfie_decode_attention.rs` held it at 0aac07c0, before the kernel moved
// onto KIR. Lines below this comment are that file's first 399 lines,
// unedited; `cfie_decode_attn_kir_equivalence.rs` includes it with
// `#[path]` and runs its output against the KIR kernel's. Do not change
// it: it is the pre-migration behaviour the equivalence claim is about.
//
// Not scanned by the hand-PTX freeze: nothing under `tests/` is.

//! CFIE Feature 1: direct-indexing decode-attention PTX emitter.
//!
//! The paper's core claim: because the KV-cache layout
//! `[n_layers][2][max_tokens][n_kv_heads][head_dim]` is fixed at compile
//! time (see `cfie_kv_plan::DirectLayout`), the decode-attention kernel
//! addresses K/V by pure arithmetic over strides baked as PTX immediates.
//! No block table, no indirection load, no CPU-side page mapping on the
//! decode path.
//!
//! ONE kernel handles every layer and batch slot: `layer_idx`/`slot_idx`
//! are runtime params, but every stride that multiplies them is an
//! immediate constant.  The global token pool is partitioned contiguously
//! per slot: slot `s` owns tokens
//! `[s*per_slot_max_tokens, (s+1)*per_slot_max_tokens)`.
//!
//! Thread mapping (flash-decode, one CTA per Q head, 128 threads):
//!   pass 1: thread `t` computes dot(Q, K[tile_base+t]) for its token,
//!           scaled by 1/sqrt(head_dim), score stored to SMEM;
//!   pass 2: thread 0 performs the online-softmax update serially over
//!           the tile's scores (running max m, running sum l, publishes
//!           the rescale factor exp(m_old - m_new) to SMEM);
//!   pass 3: thread `d < head_dim` rescales its accumulator and
//!           accumulates output element `d` across the tile's tokens.
//! Simplest-correct scheme per the Tier A spec; coalescing/vectorization
//! is a later tier's concern.

use std::fmt::Write;

/// Threads per CTA and softmax tile width (tokens processed per tile).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

pub const KERNEL_NAME: &str = "nsl_cfie_decode_attn";

pub fn kernel_name() -> &'static str {
    KERNEL_NAME
}

/// Compile-time layout + launch configuration for the decode kernel.
#[derive(Debug, Clone)]
pub struct DecodeAttentionConfig {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub per_slot_max_tokens: u32,
    pub max_slots: u32,
    /// Bytes per stored KV element (2 = f16; the only supported v1 dtype).
    pub kv_dtype_bytes: u32,
    pub sm_version: u32,
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct DecodeAttentionMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
    pub grid_dim_is_n_heads: bool,
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

/// Emit the direct-indexing decode-attention kernel.
pub fn emit(cfg: &DecodeAttentionConfig) -> (String, DecodeAttentionMeta) {
    assert!(cfg.n_layers >= 1, "n_layers must be >= 1");
    assert!(
        cfg.n_heads >= 1 && cfg.n_kv_heads >= 1,
        "n_heads and n_kv_heads must be >= 1"
    );
    assert_eq!(
        cfg.n_heads % cfg.n_kv_heads,
        0,
        "GQA requires n_heads divisible by n_kv_heads"
    );
    assert!(
        cfg.head_dim >= 1 && cfg.head_dim <= BLOCK_DIM,
        "pass 3 maps one thread per output element; head_dim must be in 1..={}",
        BLOCK_DIM
    );
    assert_eq!(
        cfg.kv_dtype_bytes, 2,
        "v1 loads KV via ld.global.b16 + cvt.f32.f16; only f16 (2 bytes) supported"
    );
    assert!(
        cfg.per_slot_max_tokens >= 1 && cfg.max_slots >= 1,
        "per_slot_max_tokens and max_slots must be >= 1"
    );

    // Baked strides in ELEMENTS of the contiguous layout
    // [n_layers][2][max_tokens][n_kv_heads][head_dim].
    let token_stride = cfg.n_kv_heads as u64 * cfg.head_dim as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    // The kernel's global token index is a u32 register (byte addressing
    // is 64-bit, but the token count itself must not wrap).
    assert!(
        max_tokens <= u32::MAX as u64,
        "global token pool (max_slots * per_slot_max_tokens = {max_tokens}) must fit in u32"
    );
    let kv_half_stride = max_tokens * token_stride;
    let layer_stride = 2 * kv_half_stride;

    let dtype = cfg.kv_dtype_bytes as u64;
    let token_stride_bytes = token_stride * dtype;
    let kv_half_stride_bytes = kv_half_stride * dtype;
    let layer_stride_bytes = layer_stride * dtype;
    let head_row_bytes = cfg.head_dim as u64 * dtype;

    let group = cfg.n_heads / cfg.n_kv_heads;
    let inv_sqrt_hd = f32_imm(1.0f32 / (cfg.head_dim as f32).sqrt());
    let log2e = f32_imm(std::f32::consts::LOG2_E);
    let neg_inf = f32_imm(f32::NEG_INFINITY);
    let zero = f32_imm(0.0);

    // SMEM layout (f32): [q: head_dim][scores: TILE][rescale: 1][l: 1].
    let scores_off = cfg.head_dim * 4;
    let rescale_off = scores_off + TILE * 4;
    let l_off = rescale_off + 4;
    let smem_bytes = l_off + 4;

    let hd = cfg.head_dim;
    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(
        w,
        "// {} - CFIE direct-indexing decode attention (flash-decode).",
        KERNEL_NAME
    )
    .unwrap();
    writeln!(
        w,
        "// KV pool layout [n_layers={}][2][max_tokens={}][n_kv_heads={}][head_dim={}], f16.",
        cfg.n_layers, max_tokens, cfg.n_kv_heads, cfg.head_dim
    )
    .unwrap();
    writeln!(w, "// Baked layout constants (elements):").unwrap();
    writeln!(w, "//   token_stride        = {}", token_stride).unwrap();
    writeln!(w, "//   kv_half_stride      = {}", kv_half_stride).unwrap();
    writeln!(w, "//   layer_stride        = {}", layer_stride).unwrap();
    writeln!(w, "//   per_slot_max_tokens = {}", cfg.per_slot_max_tokens).unwrap();
    writeln!(w, "//   max_tokens          = {}", max_tokens).unwrap();
    writeln!(w, "//   gqa_group_size      = {}", group).unwrap();
    writeln!(
        w,
        "// No block table: every KV address is arithmetic over these immediates."
    )
    .unwrap();
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", ptx_version_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".shared .align 4 .b8 cfie_smem[{}];", smem_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", KERNEL_NAME).unwrap();
    writeln!(w, "    .param .u64 q_ptr,").unwrap();
    writeln!(w, "    .param .u64 kv_base,").unwrap();
    writeln!(w, "    .param .u64 out_ptr,").unwrap();
    writeln!(w, "    .param .u32 layer_idx,").unwrap();
    writeln!(w, "    .param .u32 slot_idx,").unwrap();
    writeln!(w, "    .param .u32 seq_len").unwrap();
    writeln!(w, ")").unwrap();
    writeln!(w, "{{").unwrap();
    writeln!(
        w,
        "    .reg .pred %p_qd, %p_done, %p_val, %p_d, %p_t0, %p_j, %p_j2, %p_nd, %p_t1, %p_no, %p_lz;"
    )
    .unwrap();
    writeln!(w, "    .reg .b16 %h_k, %h_v;").unwrap();
    writeln!(
        w,
        "    .reg .f32 %f_q, %f_k, %f_v, %f_p, %f_s, %f_dot, %f_acc, %f_m, %f_l, %f_tm, %f_rs, %f_rs2, %f_t1, %f_lf, %f_o, %f_t0;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_tid, %r_head, %r_kvhead, %r_layer, %r_slot, %r_seqlen, %r_sbase, %r_slotbase, %r_hoff, %r_tile, %r_rem, %r_tcnt, %r_tok, %r_g, %r_g0, %r_d, %r_qsm, %r_j, %r_sp, %r_t1, %r_t2, %r_t3, %r_t4;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_q, %rd_kv, %rd_out, %rd_kplane, %rd_vplane, %rd_hoff, %rd_koff, %rd_kaddr, %rd_voff, %rd_vaddr, %rd_t0, %rd_t1, %rd_t2, %rd_t3;"
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_q, [q_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_kv, [kv_base];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_out, [out_ptr];").unwrap();
    writeln!(w, "    ld.param.u32 %r_layer, [layer_idx];").unwrap();
    writeln!(w, "    ld.param.u32 %r_slot, [slot_idx];").unwrap();
    writeln!(w, "    ld.param.u32 %r_seqlen, [seq_len];").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_head, %ctaid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_sbase, cfie_smem;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // GQA: kv_head = q_head / group_size (baked divisor)").unwrap();
    writeln!(w, "    div.u32 %r_kvhead, %r_head, {};", group).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // K plane base (bytes): kv_base + layer_idx * layer_stride_bytes").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_t0, %r_layer;").unwrap();
    writeln!(w, "    mul.lo.u64 %rd_kplane, %rd_t0, {};", layer_stride_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_kplane, %rd_kv, %rd_kplane;").unwrap();
    writeln!(w, "    // V plane = K plane + kv_half_stride_bytes").unwrap();
    writeln!(w, "    add.u64 %rd_vplane, %rd_kplane, {};", kv_half_stride_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // slot's first global token: slot_idx * per_slot_max_tokens").unwrap();
    writeln!(
        w,
        "    mul.lo.u32 %r_slotbase, %r_slot, {};",
        cfg.per_slot_max_tokens
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // byte offset of kv_head's row inside one token record").unwrap();
    writeln!(w, "    mul.lo.u32 %r_hoff, %r_kvhead, {};", head_row_bytes).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_hoff, %r_hoff;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // load this head's Q row (f32) into SMEM").unwrap();
    writeln!(w, "    setp.lt.u32 %p_qd, %r_tid, {};", hd).unwrap();
    writeln!(w, "    @!%p_qd bra Q_DONE;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t1, %r_head, {};", hd).unwrap();
    writeln!(w, "    add.u32 %r_t1, %r_t1, %r_tid;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t1, %r_t1, 4;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_t1, %r_t1;").unwrap();
    writeln!(w, "    add.u64 %rd_t1, %rd_q, %rd_t1;").unwrap();
    writeln!(w, "    ld.global.f32 %f_t0, [%rd_t1];").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t2, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t2, %r_t2, %r_sbase;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t2], %f_t0;").unwrap();
    writeln!(w, "Q_DONE:").unwrap();
    writeln!(w, "    mov.f32 %f_acc, {};", zero).unwrap();
    writeln!(w, "    mov.f32 %f_m, {};", neg_inf).unwrap();
    writeln!(w, "    mov.f32 %f_l, {};", zero).unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tile, 0;").unwrap();
    writeln!(w, "TILE_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_done, %r_tile, %r_seqlen;").unwrap();
    writeln!(w, "    @%p_done bra LOOP_END;").unwrap();
    writeln!(w, "    sub.u32 %r_rem, %r_seqlen, %r_tile;").unwrap();
    writeln!(w, "    min.u32 %r_tcnt, %r_rem, {};", TILE).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // pass 1: thread t scores token tile_base + t").unwrap();
    writeln!(w, "    add.u32 %r_tok, %r_tile, %r_tid;").unwrap();
    writeln!(w, "    // tail-tile guard: last tile covers seq_len % {} tokens", TILE).unwrap();
    writeln!(w, "    setp.lt.u32 %p_val, %r_tok, %r_seqlen;").unwrap();
    writeln!(w, "    @!%p_val bra SCORE_DONE;").unwrap();
    writeln!(w, "    add.u32 %r_g, %r_slotbase, %r_tok;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_koff, %r_g, {};", token_stride_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_kaddr, %rd_kplane, %rd_koff;").unwrap();
    writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, %rd_hoff;").unwrap();
    writeln!(w, "    mov.f32 %f_dot, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_d, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_qsm, %r_sbase;").unwrap();
    writeln!(w, "DOT_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_d, %r_d, {};", hd).unwrap();
    writeln!(w, "    @%p_d bra DOT_DONE;").unwrap();
    writeln!(w, "    ld.global.b16 %h_k, [%rd_kaddr];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f_k, %h_k;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_q, [%r_qsm];").unwrap();
    writeln!(w, "    fma.rn.f32 %f_dot, %f_k, %f_q, %f_dot;").unwrap();
    writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, {};", dtype).unwrap();
    writeln!(w, "    add.u32 %r_qsm, %r_qsm, 4;").unwrap();
    writeln!(w, "    add.u32 %r_d, %r_d, 1;").unwrap();
    writeln!(w, "    bra DOT_LOOP;").unwrap();
    writeln!(w, "DOT_DONE:").unwrap();
    writeln!(w, "    // scale by 1/sqrt(head_dim)").unwrap();
    writeln!(w, "    mul.f32 %f_dot, %f_dot, {};", inv_sqrt_hd).unwrap();
    writeln!(w, "    mul.lo.u32 %r_t3, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r_t3, %r_t3, %r_sbase;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_t3+{}], %f_dot;", scores_off).unwrap();
    writeln!(w, "SCORE_DONE:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // pass 2: online softmax, thread 0 serial over the tile").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra SOFTMAX_DONE;").unwrap();
    writeln!(w, "    mov.f32 %f_tm, %f_m;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_sp, %r_sbase;").unwrap();
    writeln!(w, "MAX_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_j, %r_j, %r_tcnt;").unwrap();
    writeln!(w, "    @%p_j bra MAX_DONE;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_s, [%r_sp+{}];", scores_off).unwrap();
    writeln!(w, "    max.f32 %f_tm, %f_tm, %f_s;").unwrap();
    writeln!(w, "    add.u32 %r_sp, %r_sp, 4;").unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra MAX_LOOP;").unwrap();
    writeln!(w, "MAX_DONE:").unwrap();
    writeln!(w, "    // rescale = exp(m_old - m_new); exp(-inf) = 0 on first tile").unwrap();
    writeln!(w, "    sub.f32 %f_t1, %f_m, %f_tm;").unwrap();
    writeln!(w, "    mul.f32 %f_t1, %f_t1, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f_rs, %f_t1;").unwrap();
    writeln!(w, "    mul.f32 %f_l, %f_l, %f_rs;").unwrap();
    writeln!(w, "    mov.f32 %f_m, %f_tm;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_sp, %r_sbase;").unwrap();
    writeln!(w, "P_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_j, %r_j, %r_tcnt;").unwrap();
    writeln!(w, "    @%p_j bra P_DONE;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_s, [%r_sp+{}];", scores_off).unwrap();
    writeln!(w, "    sub.f32 %f_s, %f_s, %f_m;").unwrap();
    writeln!(w, "    mul.f32 %f_s, %f_s, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f_s, %f_s;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sp+{}], %f_s;", scores_off).unwrap();
    writeln!(w, "    add.f32 %f_l, %f_l, %f_s;").unwrap();
    writeln!(w, "    add.u32 %r_sp, %r_sp, 4;").unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra P_LOOP;").unwrap();
    writeln!(w, "P_DONE:").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sbase+{}], %f_rs;", rescale_off).unwrap();
    writeln!(w, "SOFTMAX_DONE:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // pass 3: rescale accumulator, add P*V; thread d owns out[d]").unwrap();
    writeln!(w, "    setp.ge.u32 %p_nd, %r_tid, {};", hd).unwrap();
    writeln!(w, "    @%p_nd bra ACC_TAIL;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_rs2, [%r_sbase+{}];", rescale_off).unwrap();
    writeln!(w, "    mul.f32 %f_acc, %f_acc, %f_rs2;").unwrap();
    writeln!(w, "    add.u32 %r_g0, %r_slotbase, %r_tile;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_voff, %r_g0, {};", token_stride_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vplane, %rd_voff;").unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, %rd_hoff;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t2, %r_tid, {};", dtype).unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, %rd_t2;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_sp, %r_sbase;").unwrap();
    writeln!(w, "ACC_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_j2, %r_j, %r_tcnt;").unwrap();
    writeln!(w, "    @%p_j2 bra ACC_TAIL;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_p, [%r_sp+{}];", scores_off).unwrap();
    writeln!(w, "    ld.global.b16 %h_v, [%rd_vaddr];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f_v, %h_v;").unwrap();
    writeln!(w, "    fma.rn.f32 %f_acc, %f_p, %f_v, %f_acc;").unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, {};", token_stride_bytes).unwrap();
    writeln!(w, "    add.u32 %r_sp, %r_sp, 4;").unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra ACC_LOOP;").unwrap();
    writeln!(w, "ACC_TAIL:").unwrap();
    writeln!(w, "    // scores SMEM is rewritten next tile; sync before loop back").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    add.u32 %r_tile, %r_tile, {};", TILE).unwrap();
    writeln!(w, "    bra TILE_LOOP;").unwrap();
    writeln!(w, "LOOP_END:").unwrap();
    writeln!(w, "    // thread 0 publishes the final softmax denominator").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t1, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t1 bra L_PUB;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sbase+{}], %f_l;", l_off).unwrap();
    writeln!(w, "L_PUB:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    setp.ge.u32 %p_no, %r_tid, {};", hd).unwrap();
    writeln!(w, "    @%p_no bra EXIT;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_lf, [%r_sbase+{}];", l_off).unwrap();
    writeln!(w, "    // seq_len == 0 leaves l == 0; write 0 instead of NaN").unwrap();
    writeln!(w, "    mov.f32 %f_o, {};", zero).unwrap();
    writeln!(w, "    setp.gt.f32 %p_lz, %f_lf, {};", zero).unwrap();
    writeln!(w, "    @!%p_lz bra STORE_OUT;").unwrap();
    writeln!(w, "    div.rn.f32 %f_o, %f_acc, %f_lf;").unwrap();
    writeln!(w, "STORE_OUT:").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t4, %r_head, {};", hd).unwrap();
    writeln!(w, "    add.u32 %r_t4, %r_t4, %r_tid;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t4, %r_t4, 4;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_t3, %r_t4;").unwrap();
    writeln!(w, "    add.u64 %rd_t3, %rd_out, %rd_t3;").unwrap();
    writeln!(w, "    st.global.f32 [%rd_t3], %f_o;").unwrap();
    writeln!(w, "EXIT:").unwrap();
    writeln!(w, "    ret;").unwrap();
    writeln!(w, "}}").unwrap();

    let meta = DecodeAttentionMeta {
        kernel_name: KERNEL_NAME.to_string(),
        smem_bytes,
        block_dim: BLOCK_DIM,
        grid_dim_is_n_heads: true,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit`].
pub fn emit_decode_attention_ptx(cfg: &DecodeAttentionConfig) -> String {
    emit(cfg).0
}
