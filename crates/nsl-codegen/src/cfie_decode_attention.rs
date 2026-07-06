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

/// CPU reference for GPU-parity tests.  Layouts: `q` is
/// `[n_heads][head_dim]`, `k`/`v` are `[seq_len][n_kv_heads][head_dim]`
/// (i.e. one slot's tokens, contiguous — the kernel's per-token record).
/// Returns `[n_heads][head_dim]` f32.
pub fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Vec<f32> {
    assert!(n_kv_heads >= 1 && n_heads.is_multiple_of(n_kv_heads));
    let (nh, nkv, hd, sl) = (
        n_heads as usize,
        n_kv_heads as usize,
        head_dim as usize,
        seq_len as usize,
    );
    assert_eq!(q.len(), nh * hd, "q must be [n_heads][head_dim]");
    assert_eq!(k.len(), sl * nkv * hd, "k must be [seq_len][n_kv_heads][head_dim]");
    assert_eq!(v.len(), sl * nkv * hd, "v must be [seq_len][n_kv_heads][head_dim]");

    let group = nh / nkv;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; nh * hd];
    if sl == 0 {
        return out;
    }
    for h in 0..nh {
        let kvh = h / group;
        let qrow = &q[h * hd..(h + 1) * hd];
        let mut scores = Vec::with_capacity(sl);
        let mut m = f32::NEG_INFINITY;
        for t in 0..sl {
            let krow = &k[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
            let dot: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum();
            let s = dot * scale;
            scores.push(s);
            if s > m {
                m = s;
            }
        }
        let mut l = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            l += *s;
        }
        for t in 0..sl {
            let p = scores[t] / l;
            let vrow = &v[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
            for d in 0..hd {
                out[h * hd + d] += p * vrow[d];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference config from the CFIE paper's NSL-Coder example.
    fn paper_cfg() -> DecodeAttentionConfig {
        DecodeAttentionConfig {
            n_layers: 8,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 128,
            per_slot_max_tokens: 2048,
            max_slots: 64,
            kv_dtype_bytes: 2,
            sm_version: 80,
        }
    }

    #[test]
    fn param_list_is_exactly_the_six_direct_params() {
        let ptx = emit_decode_attention_ptx(&paper_cfg());
        let start = ptx.find(".visible .entry nsl_cfie_decode_attn(").unwrap();
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
                ".param .u64 q_ptr",
                ".param .u64 kv_base",
                ".param .u64 out_ptr",
                ".param .u32 layer_idx",
                ".param .u32 slot_idx",
                ".param .u32 seq_len",
            ]
        );
    }

    #[test]
    fn no_block_table_and_no_runtime_stride_loads() {
        let ptx = emit_decode_attention_ptx(&paper_cfg());
        assert!(!ptx.contains("block_table"));
        assert!(!ptx.contains("stride_ptr"));
        // Exactly the six declared params are ever loaded — any additional
        // ld.param would mean a stride reached the kernel at runtime.
        assert_eq!(ptx.matches("ld.param").count(), 6);
    }

    #[test]
    fn baked_stride_immediates_present() {
        let cfg = paper_cfg();
        let ptx = emit_decode_attention_ptx(&cfg);
        // elements: token_stride = 4*128 = 512; max_tokens = 64*2048 = 131072;
        // kv_half = 131072*512 = 67108864; layer = 2*kv_half = 134217728.
        assert!(ptx.contains("//   token_stride        = 512"));
        assert!(ptx.contains("//   kv_half_stride      = 67108864"));
        assert!(ptx.contains("//   layer_stride        = 134217728"));
        // byte immediates in the address arithmetic (x2 for f16)
        assert!(ptx.contains("mul.wide.u32 %rd_koff, %r_g, 1024;"));
        assert!(ptx.contains("mul.lo.u64 %rd_kplane, %rd_t0, 268435456;"));
        assert!(ptx.contains("add.u64 %rd_vplane, %rd_kplane, 134217728;"));
        assert!(ptx.contains("mul.lo.u32 %r_slotbase, %r_slot, 2048;"));
    }

    #[test]
    fn no_mad_lo_and_ascii_only() {
        let ptx = emit_decode_attention_ptx(&paper_cfg());
        assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
        assert!(
            ptx.bytes().all(|b| b < 128),
            "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
        );
    }

    #[test]
    fn tail_tile_guard_references_seq_len() {
        let ptx = emit_decode_attention_ptx(&paper_cfg());
        assert!(ptx.contains("setp.lt.u32 %p_val, %r_tok, %r_seqlen;"));
        assert!(ptx.contains("min.u32 %r_tcnt, %r_rem, 128;"));
    }

    #[test]
    fn header_matches_sm_version_convention() {
        let ptx80 = emit_decode_attention_ptx(&paper_cfg());
        assert!(ptx80.starts_with("//"));
        assert!(ptx80.contains(".version 7.0\n.target sm_80\n.address_size 64"));
        let mut cfg = paper_cfg();
        cfg.sm_version = 90;
        assert!(emit_decode_attention_ptx(&cfg).contains(".version 8.4\n.target sm_90"));
        cfg.sm_version = 100;
        assert!(emit_decode_attention_ptx(&cfg).contains(".version 8.6\n.target sm_100"));
    }

    #[test]
    fn meta_reports_launch_shape() {
        let (_, meta) = emit(&paper_cfg());
        assert_eq!(meta.kernel_name, kernel_name());
        assert_eq!(meta.block_dim, 128);
        assert!(meta.grid_dim_is_n_heads);
        // q(128 f32) + scores(128 f32) + rescale + l = 512 + 512 + 8.
        assert_eq!(meta.smem_bytes, 1032);
    }

    #[test]
    fn smem_scales_with_head_dim() {
        let mut cfg = paper_cfg();
        cfg.head_dim = 64;
        let (ptx, meta) = emit(&cfg);
        assert_eq!(meta.smem_bytes, 64 * 4 + 128 * 4 + 8);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 cfie_smem[{}];", meta.smem_bytes)));
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn heads_not_divisible_by_kv_heads_panics() {
        let mut cfg = paper_cfg();
        cfg.n_kv_heads = 3;
        let _ = emit(&cfg);
    }

    #[test]
    #[should_panic(expected = "head_dim")]
    fn head_dim_over_block_dim_panics() {
        let mut cfg = paper_cfg();
        cfg.head_dim = 256;
        let _ = emit(&cfg);
    }

    #[test]
    #[should_panic(expected = "f16")]
    fn non_f16_kv_dtype_panics() {
        let mut cfg = paper_cfg();
        cfg.kv_dtype_bytes = 4;
        let _ = emit(&cfg);
    }

    // ── cpu_reference ──────────────────────────────────────────────

    #[test]
    fn cpu_reference_two_token_one_head_hand_computed() {
        // head_dim=2, seq_len=2, single head.
        // q = [1, 0]; k0 = [1, 0], k1 = [0, 1]; v0 = [1, 2], v1 = [3, 4].
        // scale = 1/sqrt(2); s0 = 0.70710678, s1 = 0.
        // p0 = e^s0 / (e^s0 + 1) = 0.6697615; p1 = 0.3302385.
        // out = [p0*1 + p1*3, p0*2 + p1*4] = [1.6604770, 2.6604770].
        let q = [1.0f32, 0.0];
        let k = [1.0f32, 0.0, 0.0, 1.0];
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let out = cpu_reference(&q, &k, &v, 1, 1, 2, 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.660_477).abs() < 1e-4, "out[0] = {}", out[0]);
        assert!((out[1] - 2.660_477).abs() < 1e-4, "out[1] = {}", out[1]);
        // p0 + p1 == 1 implies out[1] - out[0] == 1 exactly (v deltas are 1).
        assert!((out[1] - out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cpu_reference_gqa_heads_sharing_kv_head_match() {
        // n_heads=4, n_kv_heads=2 (group=2): heads {0,1} -> kv 0, {2,3} -> kv 1.
        let (nh, nkv, hd, sl) = (4u32, 2u32, 2usize, 3usize);
        // identical Q rows within each group, different across groups
        let q = [0.9f32, 0.5, 0.9, 0.5, 0.3, -0.7, 0.3, -0.7];
        let mut k = vec![0.0f32; sl * nkv as usize * hd];
        let mut v = vec![0.0f32; sl * nkv as usize * hd];
        for (i, x) in k.iter_mut().enumerate() {
            *x = (i as f32 * 0.37).sin();
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = (i as f32 * 0.53).cos();
        }
        let out = cpu_reference(&q, &k, &v, nh, nkv, hd as u32, sl as u32);
        assert_eq!(&out[0..hd], &out[hd..2 * hd], "heads 0,1 share kv head 0");
        assert_eq!(&out[2 * hd..3 * hd], &out[3 * hd..4 * hd], "heads 2,3 share kv head 1");
        assert_ne!(&out[0..hd], &out[2 * hd..3 * hd], "different kv heads must differ");
    }

    #[test]
    fn cpu_reference_empty_sequence_is_zeros() {
        let q = [1.0f32, 2.0];
        let out = cpu_reference(&q, &[], &[], 1, 1, 2, 0);
        assert_eq!(out, vec![0.0f32, 0.0]);
    }

    #[test]
    fn cpu_reference_single_token_returns_v_row() {
        // seq_len=1: softmax weight is exactly 1, out == v row.
        let q = [0.25f32, -3.0, 7.5];
        let k = [1.0f32, 2.0, 3.0];
        let v = [4.0f32, 5.0, 6.0];
        let out = cpu_reference(&q, &k, &v, 1, 1, 3, 1);
        assert_eq!(out, vec![4.0, 5.0, 6.0]);
    }

    // ── ptxas validation (skips silently when no validator present) ──

    #[test]
    fn ptxas_validates_paper_config() {
        let cfg = paper_cfg();
        let ptx = emit_decode_attention_ptx(&cfg);
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                eprintln!("[skip] cfie decode-attn ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie decode-attn PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }
}
