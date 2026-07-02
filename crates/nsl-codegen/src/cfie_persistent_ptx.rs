//! CFIE Feature 4 (audit gap G16): persistent decode-block kernel.
//!
//! Paper SS6: ONE CTA executes ONE transformer layer's decode step for
//! ONE token — grid = 1, one kernel launch per layer per token ("32
//! launches instead of ~1000").  The block runs the full layer on-chip:
//!
//!   RMSNorm1 -> Q/K/V matvecs -> RoPE(q, k) -> KV-pool append ->
//!   flash-decode attention over pos+1 tokens -> W_o + residual ->
//!   RMSNorm2 -> silu(gate)*up FFN -> W_down + residual -> x_out.
//!
//! The KV pool uses the SAME baked layout as `cfie_decode_attention`
//! (`[n_layers][2][max_tokens][n_kv_heads][head_dim]`, f16, strides as
//! PTX immediates) — the two emitters share stride derivation and a
//! structural test asserts the header constants stay equal.
//!
//! Simplest-correct over occupancy (Tier-A convention): block_dim 128,
//! scalar loops, thread-per-output-element matvecs that stride by the
//! block when the output dimension exceeds 128.  `bar.sync 0` guards
//! every SMEM hazard; per PTX semantics it also orders this CTA's
//! global KV-pool stores before the attention pass reads them
//! (membar.cta effect — sufficient because grid = 1).
//!
//! Approximations (documented per house style):
//!   * RoPE angle: freq = theta^(-i2/head_dim) computed as
//!     ex2(i2 * -log2(theta)/head_dim) with `ex2.approx.f32`, then
//!     `sin.approx.f32` / `cos.approx.f32` on angle = pos * freq.
//!   * softmax exponentials: `ex2.approx.f32` with a baked log2(e).
//!   * silu(g) = g * sigmoid(g) with sigmoid = 1/(1 + ex2(-g*log2(e))).

use std::fmt::Write;

/// Threads per CTA; also the attention softmax tile width and the FFN
/// gate/up staging tile width.
const BLOCK_DIM: u32 = 128;
const ATTN_TILE: u32 = 128;
const FFN_TILE: u32 = 128;

pub const KERNEL_NAME: &str = "nsl_cfie_decode_block";

pub fn kernel_name() -> &'static str {
    KERNEL_NAME
}

/// Compile-time model + KV-pool configuration for the decode block.
#[derive(Debug, Clone)]
pub struct DecodeBlockConfig {
    pub d_model: u32,
    pub head_dim: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub d_ff: u32,
    pub per_slot_max_tokens: u32,
    pub max_slots: u32,
    pub n_layers: u32,
    /// RoPE base (10000.0 default).
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub eps: f32,
    pub sm_version: u32,
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct DecodeBlockMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
}

/// Mirrors `gpu_specs::GpuSpec::ptx_version` (same convention as
/// `cfie_decode_attention`).
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

/// Emit the persistent decode-block kernel.
///
/// Launch shape: grid = 1 CTA, block = 128 threads; one launch per
/// layer per decode step.  `pos` drives both the RoPE angle and the
/// KV-append offset; seq_len after the append is `pos + 1`.
pub fn emit(cfg: &DecodeBlockConfig) -> (String, DecodeBlockMeta) {
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
        cfg.head_dim >= 2 && cfg.head_dim <= BLOCK_DIM,
        "attention pass 3 maps one thread per output element and RoPE \
         rotates even/odd pairs; head_dim must be even and in 2..={}",
        BLOCK_DIM
    );
    assert_eq!(
        cfg.head_dim % 2,
        0,
        "RoPE pair rotation requires an even head_dim"
    );
    assert!(
        cfg.d_model >= 1 && cfg.d_model <= 8192,
        "d_model must be in 1..=8192 (SMEM residual-stream buffers)"
    );
    assert!(
        cfg.d_ff >= 1 && cfg.d_ff <= 32768,
        "d_ff must be in 1..=32768 (u32 weight-row byte arithmetic)"
    );
    assert!(
        cfg.n_heads * cfg.head_dim <= 8192,
        "n_heads * head_dim must be <= 8192 (u32 weight-row byte arithmetic)"
    );
    assert!(
        cfg.per_slot_max_tokens >= 1 && cfg.max_slots >= 1,
        "per_slot_max_tokens and max_slots must be >= 1"
    );

    let d = cfg.d_model;
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let nkv = cfg.n_kv_heads;
    let dff = cfg.d_ff;
    let nhd = nh * hd; // q/attention width
    let group = nh / nkv;

    // Baked KV-pool strides — identical derivation to
    // `cfie_decode_attention` (elements of the contiguous layout
    // [n_layers][2][max_tokens][n_kv_heads][head_dim], f16).
    let token_stride = nkv as u64 * hd as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    assert!(
        max_tokens <= u32::MAX as u64,
        "global token pool (max_slots * per_slot_max_tokens = {max_tokens}) must fit in u32"
    );
    let kv_half_stride = max_tokens * token_stride;
    let layer_stride = 2 * kv_half_stride;
    let tsb = token_stride * 2; // bytes (f16)
    let kv_half_bytes = kv_half_stride * 2;
    let layer_bytes = layer_stride * 2;

    // SMEM layout (f32 unless noted), byte offsets:
    //   X  [d_model]      residual stream (in/out of both sub-blocks)
    //   XN [d_model]      RMSNorm output (input to matvecs)
    //   Q  [nh*hd]        rotated query rows
    //   AO [nh*hd]        attention output rows
    //   RED[128]          norm tree-reduction scratch
    //   SC [128]          attention score tile
    //   RSC[1] LL[1]      online-softmax rescale + denominator
    //   H  [128]          FFN silu(gate)*up staging tile
    //   Y  [d_model]      FFN down-projection accumulator
    let x_off = 0u32;
    let xn_off = x_off + d * 4;
    let q_off = xn_off + d * 4;
    let ao_off = q_off + nhd * 4;
    let red_off = ao_off + nhd * 4;
    let sc_off = red_off + BLOCK_DIM * 4;
    let rsc_off = sc_off + ATTN_TILE * 4;
    let ll_off = rsc_off + 4;
    let h_off = ll_off + 4;
    let y_off = h_off + FFN_TILE * 4;
    let smem_bytes = y_off + d * 4;

    let inv_d = f32_imm(1.0f32 / d as f32);
    let eps = f32_imm(cfg.eps);
    let one = f32_imm(1.0);
    let zero = f32_imm(0.0);
    let neg_inf = f32_imm(f32::NEG_INFINITY);
    let log2e = f32_imm(std::f32::consts::LOG2_E);
    let neg_log2e = f32_imm(-std::f32::consts::LOG2_E);
    let inv_sqrt_hd = f32_imm(1.0f32 / (hd as f32).sqrt());
    // freq = theta^(-i2/hd) = ex2(i2 * -log2(theta)/hd), i2 = even
    // element index within the head (== 2 * pair index).
    let c_rope = f32_imm(-(cfg.rope_theta.log2()) / hd as f32);

    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(
        w,
        "// {} - CFIE persistent decode block (one CTA = one layer, one token).",
        KERNEL_NAME
    )
    .unwrap();
    writeln!(
        w,
        "// Launch: grid=1, block={} - one launch per layer per decode step.",
        BLOCK_DIM
    )
    .unwrap();
    writeln!(
        w,
        "// Model: d_model={} head_dim={} n_heads={} n_kv_heads={} d_ff={}",
        d, hd, nh, nkv, dff
    )
    .unwrap();
    writeln!(
        w,
        "// KV pool layout [n_layers={}][2][max_tokens={}][n_kv_heads={}][head_dim={}], f16.",
        cfg.n_layers, max_tokens, nkv, hd
    )
    .unwrap();
    writeln!(w, "// Baked layout constants (elements):").unwrap();
    writeln!(w, "//   token_stride        = {}", token_stride).unwrap();
    writeln!(w, "//   kv_half_stride      = {}", kv_half_stride).unwrap();
    writeln!(w, "//   layer_stride        = {}", layer_stride).unwrap();
    writeln!(w, "//   per_slot_max_tokens = {}", cfg.per_slot_max_tokens).unwrap();
    writeln!(w, "//   max_tokens          = {}", max_tokens).unwrap();
    writeln!(w, "//   gqa_group_size      = {}", group).unwrap();
    writeln!(w, "// rope_theta={} eps={}", cfg.rope_theta, cfg.eps).unwrap();
    writeln!(
        w,
        "// Approx ops: ex2/sin/cos .approx.f32 (RoPE + softmax + silu)."
    )
    .unwrap();
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", ptx_version_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".shared .align 4 .b8 cfie_blk_smem[{}];", smem_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", KERNEL_NAME).unwrap();
    writeln!(w, "    .param .u64 x_in_ptr,").unwrap();
    writeln!(w, "    .param .u64 x_out_ptr,").unwrap();
    writeln!(w, "    .param .u64 wq_ptr,").unwrap();
    writeln!(w, "    .param .u64 wk_ptr,").unwrap();
    writeln!(w, "    .param .u64 wv_ptr,").unwrap();
    writeln!(w, "    .param .u64 wo_ptr,").unwrap();
    writeln!(w, "    .param .u64 w_gate_ptr,").unwrap();
    writeln!(w, "    .param .u64 w_up_ptr,").unwrap();
    writeln!(w, "    .param .u64 w_down_ptr,").unwrap();
    writeln!(w, "    .param .u64 norm1_w_ptr,").unwrap();
    writeln!(w, "    .param .u64 norm2_w_ptr,").unwrap();
    writeln!(w, "    .param .u64 kv_base,").unwrap();
    writeln!(w, "    .param .u32 layer_idx,").unwrap();
    writeln!(w, "    .param .u32 slot_idx,").unwrap();
    writeln!(w, "    .param .u32 pos").unwrap();
    writeln!(w, ")").unwrap();
    writeln!(w, "{{").unwrap();
    writeln!(w, "    .reg .pred %p<4>;").unwrap();
    writeln!(w, "    .reg .b16 %h<2>;").unwrap();
    writeln!(w, "    .reg .f32 %f<18>;").unwrap();
    writeln!(w, "    .reg .f32 %f_acc, %f_m, %f_l;").unwrap();
    writeln!(w, "    .reg .u32 %r<18>;").unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_tid, %r_sb, %r_pos, %r_sl, %r_slotbase, %r_layer, %r_slot;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_hh, %r_qb, %r_tile, %r_tcnt, %r_tb, %r_fcnt;"
    )
    .unwrap();
    writeln!(w, "    .reg .u64 %rd<14>;").unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_xin, %rd_xout, %rd_wq, %rd_wk, %rd_wv, %rd_wo;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_wg, %rd_wu, %rd_wd, %rd_n1, %rd_n2, %rd_kv;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_kplane, %rd_vplane, %rd_krec, %rd_vrec, %rd_hoff;"
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_xin, [x_in_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_xout, [x_out_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_wq, [wq_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_wk, [wk_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_wv, [wv_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_wo, [wo_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_wg, [w_gate_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_wu, [w_up_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_wd, [w_down_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_n1, [norm1_w_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_n2, [norm2_w_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_kv, [kv_base];").unwrap();
    writeln!(w, "    ld.param.u32 %r_layer, [layer_idx];").unwrap();
    writeln!(w, "    ld.param.u32 %r_slot, [slot_idx];").unwrap();
    writeln!(w, "    ld.param.u32 %r_pos, [pos];").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_sb, cfie_blk_smem;").unwrap();
    writeln!(w, "    // seq_len after the KV append below").unwrap();
    writeln!(w, "    add.u32 %r_sl, %r_pos, 1;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // K/V plane bases (bytes) + this slot's token range").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd0, %r_layer;").unwrap();
    writeln!(w, "    mul.lo.u64 %rd_kplane, %rd0, {};", layer_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_kplane, %rd_kv, %rd_kplane;").unwrap();
    writeln!(w, "    add.u64 %rd_vplane, %rd_kplane, {};", kv_half_bytes).unwrap();
    writeln!(
        w,
        "    mul.lo.u32 %r_slotbase, %r_slot, {};",
        cfg.per_slot_max_tokens
    )
    .unwrap();
    writeln!(w, "    // append target: token record (layer, slot, pos)").unwrap();
    writeln!(w, "    add.u32 %r0, %r_slotbase, %r_pos;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd0, %r0, {};", tsb).unwrap();
    writeln!(w, "    add.u64 %rd_krec, %rd_kplane, %rd0;").unwrap();
    writeln!(w, "    add.u64 %rd_vrec, %rd_vplane, %rd0;").unwrap();
    writeln!(w).unwrap();

    // ── phase 1: load x into SMEM ─────────────────────────────────
    writeln!(w, "    // phase 1: x -> SMEM (strided over the block)").unwrap();
    writeln!(w, "    mov.u32 %r0, %r_tid;").unwrap();
    writeln!(w, "LX:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r0, {};", d).unwrap();
    writeln!(w, "    @%p0 bra LX_D;").unwrap();
    writeln!(w, "    mul.lo.u32 %r1, %r0, 4;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd0, %r1;").unwrap();
    writeln!(w, "    add.u64 %rd0, %rd_xin, %rd0;").unwrap();
    writeln!(w, "    ld.global.f32 %f0, [%rd0];").unwrap();
    writeln!(w, "    add.u32 %r2, %r1, %r_sb;").unwrap();
    writeln!(w, "    st.shared.f32 [%r2+{}], %f0;", x_off).unwrap();
    writeln!(w, "    add.u32 %r0, %r0, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra LX;").unwrap();
    writeln!(w, "LX_D:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();

    // RMSNorm emitter (X -> XN), used for norm1 and norm2.
    let emit_rmsnorm = |w: &mut String, prefix: &str, weight_reg: &str| {
        writeln!(w, "    // RMSNorm({}) X -> XN: tree reduction over 128 partials", prefix)
            .unwrap();
        writeln!(w, "    mov.f32 %f1, {};", zero).unwrap();
        writeln!(w, "    mov.u32 %r0, %r_tid;").unwrap();
        writeln!(w, "{}_SS:", prefix).unwrap();
        writeln!(w, "    setp.ge.u32 %p0, %r0, {};", d).unwrap();
        writeln!(w, "    @%p0 bra {}_SSD;", prefix).unwrap();
        writeln!(w, "    mul.lo.u32 %r1, %r0, 4;").unwrap();
        writeln!(w, "    add.u32 %r2, %r1, %r_sb;").unwrap();
        writeln!(w, "    ld.shared.f32 %f0, [%r2+{}];", x_off).unwrap();
        writeln!(w, "    fma.rn.f32 %f1, %f0, %f0, %f1;").unwrap();
        writeln!(w, "    add.u32 %r0, %r0, {};", BLOCK_DIM).unwrap();
        writeln!(w, "    bra {}_SS;", prefix).unwrap();
        writeln!(w, "{}_SSD:", prefix).unwrap();
        writeln!(w, "    mul.lo.u32 %r3, %r_tid, 4;").unwrap();
        writeln!(w, "    add.u32 %r3, %r3, %r_sb;").unwrap();
        writeln!(w, "    st.shared.f32 [%r3+{}], %f1;", red_off).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        let mut s = BLOCK_DIM / 2;
        while s >= 1 {
            writeln!(w, "    setp.ge.u32 %p0, %r_tid, {};", s).unwrap();
            writeln!(w, "    @%p0 bra {}_R{};", prefix, s).unwrap();
            writeln!(w, "    ld.shared.f32 %f0, [%r3+{}];", red_off).unwrap();
            writeln!(w, "    ld.shared.f32 %f2, [%r3+{}];", red_off + s * 4).unwrap();
            writeln!(w, "    add.f32 %f0, %f0, %f2;").unwrap();
            writeln!(w, "    st.shared.f32 [%r3+{}], %f0;", red_off).unwrap();
            writeln!(w, "{}_R{}:", prefix, s).unwrap();
            writeln!(w, "    bar.sync 0;").unwrap();
            s /= 2;
        }
        writeln!(w, "    ld.shared.f32 %f0, [%r_sb+{}];", red_off).unwrap();
        writeln!(w, "    mul.f32 %f0, %f0, {};", inv_d).unwrap();
        writeln!(w, "    add.f32 %f0, %f0, {};", eps).unwrap();
        writeln!(w, "    sqrt.rn.f32 %f0, %f0;").unwrap();
        writeln!(w, "    mov.f32 %f1, {};", one).unwrap();
        writeln!(w, "    div.rn.f32 %f1, %f1, %f0;").unwrap();
        writeln!(w, "    mov.u32 %r0, %r_tid;").unwrap();
        writeln!(w, "{}_NM:", prefix).unwrap();
        writeln!(w, "    setp.ge.u32 %p0, %r0, {};", d).unwrap();
        writeln!(w, "    @%p0 bra {}_NMD;", prefix).unwrap();
        writeln!(w, "    mul.lo.u32 %r2, %r0, 4;").unwrap();
        writeln!(w, "    add.u32 %r4, %r2, %r_sb;").unwrap();
        writeln!(w, "    ld.shared.f32 %f0, [%r4+{}];", x_off).unwrap();
        writeln!(w, "    cvt.u64.u32 %rd0, %r2;").unwrap();
        writeln!(w, "    add.u64 %rd0, {}, %rd0;", weight_reg).unwrap();
        writeln!(w, "    ld.global.f32 %f2, [%rd0];").unwrap();
        writeln!(w, "    mul.f32 %f0, %f0, %f1;").unwrap();
        writeln!(w, "    mul.f32 %f0, %f0, %f2;").unwrap();
        writeln!(w, "    st.shared.f32 [%r4+{}], %f0;", xn_off).unwrap();
        writeln!(w, "    add.u32 %r0, %r0, {};", BLOCK_DIM).unwrap();
        writeln!(w, "    bra {}_NM;", prefix).unwrap();
        writeln!(w, "{}_NMD:", prefix).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w).unwrap();
    };

    // ── phase 2: RMSNorm1 ─────────────────────────────────────────
    emit_rmsnorm(w, "N1", "%rd_n1");

    // Dual-row f16 matvec + RoPE pair rotation, shared by the Q and K
    // pair loops.  `store` emits the two rotated outputs.
    let emit_pair_loop = |w: &mut String,
                          prefix: &str,
                          pairs: u32,
                          wreg: &str,
                          store: &dyn Fn(&mut String)| {
        writeln!(w, "    mov.u32 %r10, %r_tid;").unwrap();
        writeln!(w, "{}:", prefix).unwrap();
        writeln!(w, "    setp.ge.u32 %p0, %r10, {};", pairs).unwrap();
        writeln!(w, "    @%p0 bra {}_D;", prefix).unwrap();
        writeln!(w, "    shl.b32 %r11, %r10, 1;").unwrap();
        writeln!(w, "    // dual dot: weight rows e0 and e0+1 against XN").unwrap();
        writeln!(w, "    mul.lo.u32 %r12, %r11, {};", d * 2).unwrap();
        writeln!(w, "    cvt.u64.u32 %rd10, %r12;").unwrap();
        writeln!(w, "    add.u64 %rd10, {}, %rd10;", wreg).unwrap();
        writeln!(w, "    add.u64 %rd11, %rd10, {};", d * 2).unwrap();
        writeln!(w, "    mov.f32 %f10, {};", zero).unwrap();
        writeln!(w, "    mov.f32 %f11, {};", zero).unwrap();
        writeln!(w, "    mov.u32 %r13, 0;").unwrap();
        writeln!(w, "    mov.u32 %r14, %r_sb;").unwrap();
        writeln!(w, "{}_DOT:", prefix).unwrap();
        writeln!(w, "    setp.ge.u32 %p1, %r13, {};", d).unwrap();
        writeln!(w, "    @%p1 bra {}_DOTD;", prefix).unwrap();
        writeln!(w, "    ld.shared.f32 %f12, [%r14+{}];", xn_off).unwrap();
        writeln!(w, "    ld.global.b16 %h0, [%rd10];").unwrap();
        writeln!(w, "    cvt.f32.f16 %f13, %h0;").unwrap();
        writeln!(w, "    fma.rn.f32 %f10, %f13, %f12, %f10;").unwrap();
        writeln!(w, "    ld.global.b16 %h1, [%rd11];").unwrap();
        writeln!(w, "    cvt.f32.f16 %f13, %h1;").unwrap();
        writeln!(w, "    fma.rn.f32 %f11, %f13, %f12, %f11;").unwrap();
        writeln!(w, "    add.u64 %rd10, %rd10, 2;").unwrap();
        writeln!(w, "    add.u64 %rd11, %rd11, 2;").unwrap();
        writeln!(w, "    add.u32 %r14, %r14, 4;").unwrap();
        writeln!(w, "    add.u32 %r13, %r13, 1;").unwrap();
        writeln!(w, "    bra {}_DOT;", prefix).unwrap();
        writeln!(w, "{}_DOTD:", prefix).unwrap();
        writeln!(w, "    // RoPE: i2 = e0 % head_dim; angle = pos * ex2(i2 * c_rope)").unwrap();
        writeln!(w, "    div.u32 %r12, %r11, {};", hd).unwrap();
        writeln!(w, "    mul.lo.u32 %r12, %r12, {};", hd).unwrap();
        writeln!(w, "    sub.u32 %r12, %r11, %r12;").unwrap();
        writeln!(w, "    cvt.rn.f32.u32 %f12, %r12;").unwrap();
        writeln!(w, "    mul.f32 %f12, %f12, {};", c_rope).unwrap();
        writeln!(w, "    ex2.approx.f32 %f12, %f12;").unwrap();
        writeln!(w, "    cvt.rn.f32.u32 %f13, %r_pos;").unwrap();
        writeln!(w, "    mul.f32 %f12, %f13, %f12;").unwrap();
        writeln!(w, "    sin.approx.f32 %f14, %f12;").unwrap();
        writeln!(w, "    cos.approx.f32 %f15, %f12;").unwrap();
        writeln!(w, "    // (e', o') = (e*cos - o*sin, e*sin + o*cos)").unwrap();
        writeln!(w, "    mul.f32 %f16, %f10, %f15;").unwrap();
        writeln!(w, "    mul.f32 %f17, %f11, %f14;").unwrap();
        writeln!(w, "    sub.f32 %f16, %f16, %f17;").unwrap();
        writeln!(w, "    mul.f32 %f17, %f10, %f14;").unwrap();
        writeln!(w, "    fma.rn.f32 %f17, %f11, %f15, %f17;").unwrap();
        store(w);
        writeln!(w, "    add.u32 %r10, %r10, {};", BLOCK_DIM).unwrap();
        writeln!(w, "    bra {};", prefix).unwrap();
        writeln!(w, "{}_D:", prefix).unwrap();
    };

    // ── phase 3a: Q = rope(Wq @ xn) -> SMEM ───────────────────────
    writeln!(w, "    // phase 3a: Q pairs (thread p owns elements 2p, 2p+1)").unwrap();
    emit_pair_loop(w, "QP", nhd / 2, "%rd_wq", &|w| {
        writeln!(w, "    mul.lo.u32 %r12, %r11, 4;").unwrap();
        writeln!(w, "    add.u32 %r12, %r12, %r_sb;").unwrap();
        writeln!(w, "    st.shared.f32 [%r12+{}], %f16;", q_off).unwrap();
        writeln!(w, "    st.shared.f32 [%r12+{}], %f17;", q_off + 4).unwrap();
    });
    writeln!(w).unwrap();

    // ── phase 3b: K = rope(Wk @ xn) -> KV pool append (f16) ───────
    writeln!(w, "    // phase 3b: K pairs, appended straight to the pool").unwrap();
    emit_pair_loop(w, "KP", nkv * hd / 2, "%rd_wk", &|w| {
        writeln!(w, "    mul.lo.u32 %r12, %r11, 2;").unwrap();
        writeln!(w, "    cvt.u64.u32 %rd12, %r12;").unwrap();
        writeln!(w, "    add.u64 %rd12, %rd_krec, %rd12;").unwrap();
        writeln!(w, "    cvt.rn.f16.f32 %h0, %f16;").unwrap();
        writeln!(w, "    st.global.b16 [%rd12], %h0;").unwrap();
        writeln!(w, "    cvt.rn.f16.f32 %h1, %f17;").unwrap();
        writeln!(w, "    st.global.b16 [%rd12+2], %h1;").unwrap();
    });
    writeln!(w).unwrap();

    // ── phase 3c: V = Wv @ xn -> KV pool append (f16, no rotation) ─
    writeln!(w, "    // phase 3c: V elements (no rotation)").unwrap();
    writeln!(w, "    mov.u32 %r10, %r_tid;").unwrap();
    writeln!(w, "VP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r10, {};", nkv * hd).unwrap();
    writeln!(w, "    @%p0 bra VP_D;").unwrap();
    writeln!(w, "    mul.lo.u32 %r12, %r10, {};", d * 2).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd10, %r12;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd_wv, %rd10;").unwrap();
    writeln!(w, "    mov.f32 %f10, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r13, 0;").unwrap();
    writeln!(w, "    mov.u32 %r14, %r_sb;").unwrap();
    writeln!(w, "VP_DOT:").unwrap();
    writeln!(w, "    setp.ge.u32 %p1, %r13, {};", d).unwrap();
    writeln!(w, "    @%p1 bra VP_DOTD;").unwrap();
    writeln!(w, "    ld.shared.f32 %f12, [%r14+{}];", xn_off).unwrap();
    writeln!(w, "    ld.global.b16 %h0, [%rd10];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f13, %h0;").unwrap();
    writeln!(w, "    fma.rn.f32 %f10, %f13, %f12, %f10;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, 2;").unwrap();
    writeln!(w, "    add.u32 %r14, %r14, 4;").unwrap();
    writeln!(w, "    add.u32 %r13, %r13, 1;").unwrap();
    writeln!(w, "    bra VP_DOT;").unwrap();
    writeln!(w, "VP_DOTD:").unwrap();
    writeln!(w, "    mul.lo.u32 %r12, %r10, 2;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd12, %r12;").unwrap();
    writeln!(w, "    add.u64 %rd12, %rd_vrec, %rd12;").unwrap();
    writeln!(w, "    cvt.rn.f16.f32 %h0, %f10;").unwrap();
    writeln!(w, "    st.global.b16 [%rd12], %h0;").unwrap();
    writeln!(w, "    add.u32 %r10, %r10, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra VP;").unwrap();
    writeln!(w, "VP_D:").unwrap();
    writeln!(
        w,
        "    // publishes Q SMEM + orders the KV-pool stores before the"
    )
    .unwrap();
    writeln!(w, "    // attention reads (bar.sync has membar.cta effect)").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();

    // ── phase 4: flash-decode attention, Q heads sequential ───────
    writeln!(
        w,
        "    // phase 4: 3-pass flash-decode per Q head (sequential heads,"
    )
    .unwrap();
    writeln!(
        w,
        "    // same algorithm as {}); seq_len = pos+1",
        crate::cfie_decode_attention::KERNEL_NAME
    )
    .unwrap();
    writeln!(w, "    mov.u32 %r_hh, 0;").unwrap();
    writeln!(w, "AH:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r_hh, {};", nh).unwrap();
    writeln!(w, "    @%p0 bra AH_D;").unwrap();
    writeln!(w, "    // kv_head = q_head / group_size (baked divisor)").unwrap();
    writeln!(w, "    div.u32 %r10, %r_hh, {};", group).unwrap();
    writeln!(w, "    mul.lo.u32 %r10, %r10, {};", hd * 2).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_hoff, %r10;").unwrap();
    writeln!(w, "    mul.lo.u32 %r11, %r_hh, {};", hd * 4).unwrap();
    writeln!(w, "    add.u32 %r_qb, %r11, %r_sb;").unwrap();
    writeln!(w, "    mov.f32 %f_acc, {};", zero).unwrap();
    writeln!(w, "    mov.f32 %f_m, {};", neg_inf).unwrap();
    writeln!(w, "    mov.f32 %f_l, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_tile, 0;").unwrap();
    writeln!(w, "AT_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r_tile, %r_sl;").unwrap();
    writeln!(w, "    @%p0 bra AT_END;").unwrap();
    writeln!(w, "    mov.u32 %r12, %r_sl;").unwrap();
    writeln!(w, "    sub.u32 %r12, %r12, %r_tile;").unwrap();
    writeln!(w, "    min.u32 %r_tcnt, %r12, {};", ATTN_TILE).unwrap();
    writeln!(w, "    // pass 1: thread t scores token tile_base + t").unwrap();
    writeln!(w, "    add.u32 %r13, %r_tile, %r_tid;").unwrap();
    writeln!(w, "    setp.lt.u32 %p1, %r13, %r_sl;").unwrap();
    writeln!(w, "    @!%p1 bra ASC_D;").unwrap();
    writeln!(w, "    add.u32 %r14, %r_slotbase, %r13;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd10, %r14, {};", tsb).unwrap();
    writeln!(w, "    add.u64 %rd10, %rd_kplane, %rd10;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, %rd_hoff;").unwrap();
    writeln!(w, "    mov.f32 %f10, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r15, 0;").unwrap();
    writeln!(w, "    mov.u32 %r16, %r_qb;").unwrap();
    writeln!(w, "ADOT:").unwrap();
    writeln!(w, "    setp.ge.u32 %p2, %r15, {};", hd).unwrap();
    writeln!(w, "    @%p2 bra ADOT_D;").unwrap();
    writeln!(w, "    ld.global.b16 %h0, [%rd10];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f11, %h0;").unwrap();
    writeln!(w, "    ld.shared.f32 %f12, [%r16+{}];", q_off).unwrap();
    writeln!(w, "    fma.rn.f32 %f10, %f11, %f12, %f10;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, 2;").unwrap();
    writeln!(w, "    add.u32 %r16, %r16, 4;").unwrap();
    writeln!(w, "    add.u32 %r15, %r15, 1;").unwrap();
    writeln!(w, "    bra ADOT;").unwrap();
    writeln!(w, "ADOT_D:").unwrap();
    writeln!(w, "    mul.f32 %f10, %f10, {};", inv_sqrt_hd).unwrap();
    writeln!(w, "    mul.lo.u32 %r15, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r15, %r15, %r_sb;").unwrap();
    writeln!(w, "    st.shared.f32 [%r15+{}], %f10;", sc_off).unwrap();
    writeln!(w, "ASC_D:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    // pass 2: online softmax, thread 0 serial over the tile").unwrap();
    writeln!(w, "    setp.ne.u32 %p1, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p1 bra ASM_D;").unwrap();
    writeln!(w, "    mov.f32 %f10, %f_m;").unwrap();
    writeln!(w, "    mov.u32 %r15, 0;").unwrap();
    writeln!(w, "    mov.u32 %r16, %r_sb;").unwrap();
    writeln!(w, "AMAX:").unwrap();
    writeln!(w, "    setp.ge.u32 %p2, %r15, %r_tcnt;").unwrap();
    writeln!(w, "    @%p2 bra AMAX_D;").unwrap();
    writeln!(w, "    ld.shared.f32 %f11, [%r16+{}];", sc_off).unwrap();
    writeln!(w, "    max.f32 %f10, %f10, %f11;").unwrap();
    writeln!(w, "    add.u32 %r16, %r16, 4;").unwrap();
    writeln!(w, "    add.u32 %r15, %r15, 1;").unwrap();
    writeln!(w, "    bra AMAX;").unwrap();
    writeln!(w, "AMAX_D:").unwrap();
    writeln!(w, "    // rescale = exp(m_old - m_new); exp(-inf) = 0 on first tile").unwrap();
    writeln!(w, "    sub.f32 %f11, %f_m, %f10;").unwrap();
    writeln!(w, "    mul.f32 %f11, %f11, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f12, %f11;").unwrap();
    writeln!(w, "    mul.f32 %f_l, %f_l, %f12;").unwrap();
    writeln!(w, "    mov.f32 %f_m, %f10;").unwrap();
    writeln!(w, "    mov.u32 %r15, 0;").unwrap();
    writeln!(w, "    mov.u32 %r16, %r_sb;").unwrap();
    writeln!(w, "AP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p2, %r15, %r_tcnt;").unwrap();
    writeln!(w, "    @%p2 bra AP_D;").unwrap();
    writeln!(w, "    ld.shared.f32 %f11, [%r16+{}];", sc_off).unwrap();
    writeln!(w, "    sub.f32 %f11, %f11, %f_m;").unwrap();
    writeln!(w, "    mul.f32 %f11, %f11, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f11, %f11;").unwrap();
    writeln!(w, "    st.shared.f32 [%r16+{}], %f11;", sc_off).unwrap();
    writeln!(w, "    add.f32 %f_l, %f_l, %f11;").unwrap();
    writeln!(w, "    add.u32 %r16, %r16, 4;").unwrap();
    writeln!(w, "    add.u32 %r15, %r15, 1;").unwrap();
    writeln!(w, "    bra AP;").unwrap();
    writeln!(w, "AP_D:").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sb+{}], %f12;", rsc_off).unwrap();
    writeln!(w, "ASM_D:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    // pass 3: rescale accumulator, add P*V; thread d owns out[d]").unwrap();
    writeln!(w, "    setp.ge.u32 %p1, %r_tid, {};", hd).unwrap();
    writeln!(w, "    @%p1 bra AACC_T;").unwrap();
    writeln!(w, "    ld.shared.f32 %f10, [%r_sb+{}];", rsc_off).unwrap();
    writeln!(w, "    mul.f32 %f_acc, %f_acc, %f10;").unwrap();
    writeln!(w, "    add.u32 %r14, %r_slotbase, %r_tile;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd10, %r14, {};", tsb).unwrap();
    writeln!(w, "    add.u64 %rd10, %rd_vplane, %rd10;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, %rd_hoff;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd11, %r_tid, 2;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, %rd11;").unwrap();
    writeln!(w, "    mov.u32 %r15, 0;").unwrap();
    writeln!(w, "    mov.u32 %r16, %r_sb;").unwrap();
    writeln!(w, "AACC:").unwrap();
    writeln!(w, "    setp.ge.u32 %p2, %r15, %r_tcnt;").unwrap();
    writeln!(w, "    @%p2 bra AACC_T;").unwrap();
    writeln!(w, "    ld.shared.f32 %f11, [%r16+{}];", sc_off).unwrap();
    writeln!(w, "    ld.global.b16 %h0, [%rd10];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f12, %h0;").unwrap();
    writeln!(w, "    fma.rn.f32 %f_acc, %f11, %f12, %f_acc;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, {};", tsb).unwrap();
    writeln!(w, "    add.u32 %r16, %r16, 4;").unwrap();
    writeln!(w, "    add.u32 %r15, %r15, 1;").unwrap();
    writeln!(w, "    bra AACC;").unwrap();
    writeln!(w, "AACC_T:").unwrap();
    writeln!(w, "    // score tile SMEM rewritten next tile").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    add.u32 %r_tile, %r_tile, {};", ATTN_TILE).unwrap();
    writeln!(w, "    bra AT_LOOP;").unwrap();
    writeln!(w, "AT_END:").unwrap();
    writeln!(w, "    setp.ne.u32 %p0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p0 bra ALP;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sb+{}], %f_l;", ll_off).unwrap();
    writeln!(w, "ALP:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r_tid, {};", hd).unwrap();
    writeln!(w, "    @%p0 bra AH_NEXT;").unwrap();
    writeln!(w, "    ld.shared.f32 %f10, [%r_sb+{}];", ll_off).unwrap();
    writeln!(w, "    // l > 0 always holds (seq_len = pos+1 >= 1); keep the guard").unwrap();
    writeln!(w, "    mov.f32 %f11, {};", zero).unwrap();
    writeln!(w, "    setp.gt.f32 %p1, %f10, {};", zero).unwrap();
    writeln!(w, "    @!%p1 bra AH_ST;").unwrap();
    writeln!(w, "    div.rn.f32 %f11, %f_acc, %f10;").unwrap();
    writeln!(w, "AH_ST:").unwrap();
    writeln!(w, "    mul.lo.u32 %r12, %r_hh, {};", hd).unwrap();
    writeln!(w, "    add.u32 %r12, %r12, %r_tid;").unwrap();
    writeln!(w, "    mul.lo.u32 %r12, %r12, 4;").unwrap();
    writeln!(w, "    add.u32 %r12, %r12, %r_sb;").unwrap();
    writeln!(w, "    st.shared.f32 [%r12+{}], %f11;", ao_off).unwrap();
    writeln!(w, "AH_NEXT:").unwrap();
    writeln!(w, "    // rescale/l/score SMEM reused by the next head").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    add.u32 %r_hh, %r_hh, 1;").unwrap();
    writeln!(w, "    bra AH;").unwrap();
    writeln!(w, "AH_D:").unwrap();
    writeln!(w).unwrap();

    // ── phase 5: W_o matvec + residual into X ─────────────────────
    writeln!(w, "    // phase 5: x += Wo @ attn_out (thread d owns X[d])").unwrap();
    writeln!(w, "    mov.u32 %r10, %r_tid;").unwrap();
    writeln!(w, "WO:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r10, {};", d).unwrap();
    writeln!(w, "    @%p0 bra WO_D;").unwrap();
    writeln!(w, "    mul.lo.u32 %r11, %r10, {};", nhd * 2).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd10, %r11;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd_wo, %rd10;").unwrap();
    writeln!(w, "    mov.f32 %f10, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r12, 0;").unwrap();
    writeln!(w, "    mov.u32 %r13, %r_sb;").unwrap();
    writeln!(w, "WO_DOT:").unwrap();
    writeln!(w, "    setp.ge.u32 %p1, %r12, {};", nhd).unwrap();
    writeln!(w, "    @%p1 bra WO_DOTD;").unwrap();
    writeln!(w, "    ld.global.b16 %h0, [%rd10];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f11, %h0;").unwrap();
    writeln!(w, "    ld.shared.f32 %f12, [%r13+{}];", ao_off).unwrap();
    writeln!(w, "    fma.rn.f32 %f10, %f11, %f12, %f10;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, 2;").unwrap();
    writeln!(w, "    add.u32 %r13, %r13, 4;").unwrap();
    writeln!(w, "    add.u32 %r12, %r12, 1;").unwrap();
    writeln!(w, "    bra WO_DOT;").unwrap();
    writeln!(w, "WO_DOTD:").unwrap();
    writeln!(w, "    mul.lo.u32 %r14, %r10, 4;").unwrap();
    writeln!(w, "    add.u32 %r14, %r14, %r_sb;").unwrap();
    writeln!(w, "    ld.shared.f32 %f12, [%r14+{}];", x_off).unwrap();
    writeln!(w, "    add.f32 %f12, %f12, %f10;").unwrap();
    writeln!(w, "    st.shared.f32 [%r14+{}], %f12;", x_off).unwrap();
    writeln!(w, "    add.u32 %r10, %r10, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra WO;").unwrap();
    writeln!(w, "WO_D:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w).unwrap();

    // ── phase 6: RMSNorm2 ─────────────────────────────────────────
    emit_rmsnorm(w, "N2", "%rd_n2");

    // ── phase 7: FFN (tiled gate/up + down accumulation) ──────────
    writeln!(w, "    // phase 7: FFN. Y accumulator zeroed, then per 128-wide").unwrap();
    writeln!(w, "    // d_ff tile: h = silu(gate)*up staged in SMEM, down-").unwrap();
    writeln!(w, "    // projection accumulated into Y (thread d owns Y[d]).").unwrap();
    writeln!(w, "    mov.u32 %r10, %r_tid;").unwrap();
    writeln!(w, "    mov.f32 %f10, {};", zero).unwrap();
    writeln!(w, "YZ:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r10, {};", d).unwrap();
    writeln!(w, "    @%p0 bra YZ_D;").unwrap();
    writeln!(w, "    mul.lo.u32 %r11, %r10, 4;").unwrap();
    writeln!(w, "    add.u32 %r11, %r11, %r_sb;").unwrap();
    writeln!(w, "    st.shared.f32 [%r11+{}], %f10;", y_off).unwrap();
    writeln!(w, "    add.u32 %r10, %r10, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra YZ;").unwrap();
    writeln!(w, "YZ_D:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    mov.u32 %r_tb, 0;").unwrap();
    writeln!(w, "FT:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r_tb, {};", dff).unwrap();
    writeln!(w, "    @%p0 bra FT_D;").unwrap();
    writeln!(w, "    mov.u32 %r10, {};", dff).unwrap();
    writeln!(w, "    sub.u32 %r10, %r10, %r_tb;").unwrap();
    writeln!(w, "    min.u32 %r_fcnt, %r10, {};", FFN_TILE).unwrap();
    writeln!(w, "    // gate/up dual matvec for row tb + tid").unwrap();
    writeln!(w, "    setp.ge.u32 %p1, %r_tid, %r_fcnt;").unwrap();
    writeln!(w, "    @%p1 bra FH_D;").unwrap();
    writeln!(w, "    add.u32 %r11, %r_tb, %r_tid;").unwrap();
    writeln!(w, "    mul.lo.u32 %r12, %r11, {};", d * 2).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd10, %r12;").unwrap();
    writeln!(w, "    add.u64 %rd11, %rd_wg, %rd10;").unwrap();
    writeln!(w, "    add.u64 %rd12, %rd_wu, %rd10;").unwrap();
    writeln!(w, "    mov.f32 %f10, {};", zero).unwrap();
    writeln!(w, "    mov.f32 %f11, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r13, 0;").unwrap();
    writeln!(w, "    mov.u32 %r14, %r_sb;").unwrap();
    writeln!(w, "FGU:").unwrap();
    writeln!(w, "    setp.ge.u32 %p2, %r13, {};", d).unwrap();
    writeln!(w, "    @%p2 bra FGU_D;").unwrap();
    writeln!(w, "    ld.shared.f32 %f12, [%r14+{}];", xn_off).unwrap();
    writeln!(w, "    ld.global.b16 %h0, [%rd11];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f13, %h0;").unwrap();
    writeln!(w, "    fma.rn.f32 %f10, %f13, %f12, %f10;").unwrap();
    writeln!(w, "    ld.global.b16 %h1, [%rd12];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f13, %h1;").unwrap();
    writeln!(w, "    fma.rn.f32 %f11, %f13, %f12, %f11;").unwrap();
    writeln!(w, "    add.u64 %rd11, %rd11, 2;").unwrap();
    writeln!(w, "    add.u64 %rd12, %rd12, 2;").unwrap();
    writeln!(w, "    add.u32 %r14, %r14, 4;").unwrap();
    writeln!(w, "    add.u32 %r13, %r13, 1;").unwrap();
    writeln!(w, "    bra FGU;").unwrap();
    writeln!(w, "FGU_D:").unwrap();
    writeln!(w, "    // silu(g) = g * sigmoid(g); sigmoid = 1/(1 + ex2(-g*log2e))").unwrap();
    writeln!(w, "    mul.f32 %f12, %f10, {};", neg_log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f12, %f12;").unwrap();
    writeln!(w, "    add.f32 %f12, %f12, {};", one).unwrap();
    writeln!(w, "    mov.f32 %f13, {};", one).unwrap();
    writeln!(w, "    div.rn.f32 %f13, %f13, %f12;").unwrap();
    writeln!(w, "    mul.f32 %f13, %f10, %f13;").unwrap();
    writeln!(w, "    mul.f32 %f13, %f13, %f11;").unwrap();
    writeln!(w, "    mul.lo.u32 %r15, %r_tid, 4;").unwrap();
    writeln!(w, "    add.u32 %r15, %r15, %r_sb;").unwrap();
    writeln!(w, "    st.shared.f32 [%r15+{}], %f13;", h_off).unwrap();
    writeln!(w, "FH_D:").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    // down accumulation: Y[d] += W_down[d, tb..tb+fcnt] . h").unwrap();
    writeln!(w, "    mov.u32 %r10, %r_tid;").unwrap();
    writeln!(w, "FD:").unwrap();
    writeln!(w, "    setp.ge.u32 %p1, %r10, {};", d).unwrap();
    writeln!(w, "    @%p1 bra FD_D;").unwrap();
    writeln!(w, "    mul.lo.u32 %r11, %r10, {};", dff * 2).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd10, %r11;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd_wd, %rd10;").unwrap();
    writeln!(w, "    mul.lo.u32 %r12, %r_tb, 2;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd11, %r12;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, %rd11;").unwrap();
    writeln!(w, "    mov.f32 %f10, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r13, 0;").unwrap();
    writeln!(w, "    mov.u32 %r14, %r_sb;").unwrap();
    writeln!(w, "FDD:").unwrap();
    writeln!(w, "    setp.ge.u32 %p2, %r13, %r_fcnt;").unwrap();
    writeln!(w, "    @%p2 bra FDD_D;").unwrap();
    writeln!(w, "    ld.shared.f32 %f11, [%r14+{}];", h_off).unwrap();
    writeln!(w, "    ld.global.b16 %h0, [%rd10];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f12, %h0;").unwrap();
    writeln!(w, "    fma.rn.f32 %f10, %f11, %f12, %f10;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd10, 2;").unwrap();
    writeln!(w, "    add.u32 %r14, %r14, 4;").unwrap();
    writeln!(w, "    add.u32 %r13, %r13, 1;").unwrap();
    writeln!(w, "    bra FDD;").unwrap();
    writeln!(w, "FDD_D:").unwrap();
    writeln!(w, "    mul.lo.u32 %r15, %r10, 4;").unwrap();
    writeln!(w, "    add.u32 %r15, %r15, %r_sb;").unwrap();
    writeln!(w, "    ld.shared.f32 %f11, [%r15+{}];", y_off).unwrap();
    writeln!(w, "    add.f32 %f11, %f11, %f10;").unwrap();
    writeln!(w, "    st.shared.f32 [%r15+{}], %f11;", y_off).unwrap();
    writeln!(w, "    add.u32 %r10, %r10, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra FD;").unwrap();
    writeln!(w, "FD_D:").unwrap();
    writeln!(w, "    // h tile rewritten next iteration").unwrap();
    writeln!(w, "    bar.sync 0;").unwrap();
    writeln!(w, "    add.u32 %r_tb, %r_tb, {};", FFN_TILE).unwrap();
    writeln!(w, "    bra FT;").unwrap();
    writeln!(w, "FT_D:").unwrap();
    writeln!(w).unwrap();

    // ── phase 8: x_out = X + Y ────────────────────────────────────
    writeln!(w, "    // phase 8: x_out = x + ffn (second residual)").unwrap();
    writeln!(w, "    mov.u32 %r10, %r_tid;").unwrap();
    writeln!(w, "SO:").unwrap();
    writeln!(w, "    setp.ge.u32 %p0, %r10, {};", d).unwrap();
    writeln!(w, "    @%p0 bra SO_D;").unwrap();
    writeln!(w, "    mul.lo.u32 %r11, %r10, 4;").unwrap();
    writeln!(w, "    add.u32 %r12, %r11, %r_sb;").unwrap();
    writeln!(w, "    ld.shared.f32 %f10, [%r12+{}];", x_off).unwrap();
    writeln!(w, "    ld.shared.f32 %f11, [%r12+{}];", y_off).unwrap();
    writeln!(w, "    add.f32 %f10, %f10, %f11;").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd10, %r11;").unwrap();
    writeln!(w, "    add.u64 %rd10, %rd_xout, %rd10;").unwrap();
    writeln!(w, "    st.global.f32 [%rd10], %f10;").unwrap();
    writeln!(w, "    add.u32 %r10, %r10, {};", BLOCK_DIM).unwrap();
    writeln!(w, "    bra SO;").unwrap();
    writeln!(w, "SO_D:").unwrap();
    writeln!(w, "    ret;").unwrap();
    writeln!(w, "}}").unwrap();

    let meta = DecodeBlockMeta {
        kernel_name: KERNEL_NAME.to_string(),
        smem_bytes,
        block_dim: BLOCK_DIM,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit`].
pub fn emit_decode_block_ptx(cfg: &DecodeBlockConfig) -> String {
    emit(cfg).0
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let rms = (ss / d as f32 + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v / rms * g).collect()
}

fn rope_rotate(row: &mut [f32], head_dim: usize, pos: u32, theta: f32) {
    for h in 0..row.len() / head_dim {
        let base = h * head_dim;
        let mut i = 0;
        while i < head_dim {
            let freq = theta.powf(-(i as f32) / head_dim as f32);
            let ang = pos as f32 * freq;
            let (s, c) = ang.sin_cos();
            let e = row[base + i];
            let o = row[base + i + 1];
            row[base + i] = e * c - o * s;
            row[base + i + 1] = e * s + o * c;
            i += 2;
        }
    }
}

fn matvec(w: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let cols = x.len();
    assert_eq!(w.len(), rows * cols, "weight shape mismatch");
    (0..rows)
        .map(|r| w[r * cols..(r + 1) * cols].iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

/// Full-block CPU reference.  Weights are f32 row-major `[out, in]`
/// (the kernel's f16 storage is a GPU-parity concern, not modelled
/// here).  `kv_k`/`kv_v` hold this slot's token records
/// `[pos][n_kv_heads][head_dim]`; the call appends token `pos` (so
/// chained calls with pos = 0, 1, ... mirror the kernel's KV append)
/// and attends over `pos + 1` tokens.  Returns the new residual stream
/// `[d_model]`.
#[allow(clippy::too_many_arguments)]
pub fn cpu_reference(
    cfg: &DecodeBlockConfig,
    x: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    wo: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    norm1_w: &[f32],
    norm2_w: &[f32],
    kv_k: &mut Vec<f32>,
    kv_v: &mut Vec<f32>,
    pos: u32,
) -> Vec<f32> {
    let d = cfg.d_model as usize;
    let hd = cfg.head_dim as usize;
    let nh = cfg.n_heads as usize;
    let nkv = cfg.n_kv_heads as usize;
    let dff = cfg.d_ff as usize;
    let group = nh / nkv;
    assert_eq!(x.len(), d, "x must be [d_model]");
    assert_eq!(norm1_w.len(), d);
    assert_eq!(norm2_w.len(), d);
    assert_eq!(kv_k.len(), pos as usize * nkv * hd, "kv_k must hold pos tokens");
    assert_eq!(kv_v.len(), pos as usize * nkv * hd, "kv_v must hold pos tokens");

    // Attention sub-block.
    let xn = rmsnorm(x, norm1_w, cfg.eps);
    let mut q = matvec(wq, &xn, nh * hd);
    rope_rotate(&mut q, hd, pos, cfg.rope_theta);
    let mut k_new = matvec(wk, &xn, nkv * hd);
    rope_rotate(&mut k_new, hd, pos, cfg.rope_theta);
    let v_new = matvec(wv, &xn, nkv * hd);
    kv_k.extend_from_slice(&k_new);
    kv_v.extend_from_slice(&v_new);

    let sl = pos as usize + 1;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut ao = vec![0.0f32; nh * hd];
    for h in 0..nh {
        let kvh = h / group;
        let qrow = &q[h * hd..(h + 1) * hd];
        let mut scores: Vec<f32> = (0..sl)
            .map(|t| {
                let krow = &kv_k[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
                qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale
            })
            .collect();
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut l = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            l += *s;
        }
        for t in 0..sl {
            let p = scores[t] / l;
            let vrow = &kv_v[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
            for e in 0..hd {
                ao[h * hd + e] += p * vrow[e];
            }
        }
    }
    let proj = matvec(wo, &ao, d);
    let x1: Vec<f32> = x.iter().zip(&proj).map(|(a, b)| a + b).collect();

    // FFN sub-block.
    let xn2 = rmsnorm(&x1, norm2_w, cfg.eps);
    let gate = matvec(w_gate, &xn2, dff);
    let up = matvec(w_up, &xn2, dff);
    let h: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    let y = matvec(w_down, &h, d);
    x1.iter().zip(&y).map(|(a, b)| a + b).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Paper-shaped NSL-Coder config (d_model 512 / 8 heads of 64 /
    /// GQA 4 KV heads / d_ff 1408) on the sm_80 baseline target.
    fn paper_cfg() -> DecodeBlockConfig {
        DecodeBlockConfig {
            d_model: 512,
            head_dim: 64,
            n_heads: 8,
            n_kv_heads: 4,
            d_ff: 1408,
            per_slot_max_tokens: 2048,
            max_slots: 64,
            n_layers: 8,
            rope_theta: 10000.0,
            eps: 1e-5,
            sm_version: 80,
        }
    }

    #[test]
    fn param_list_is_exactly_the_fifteen_block_params() {
        let ptx = emit_decode_block_ptx(&paper_cfg());
        let start = ptx.find(".visible .entry nsl_cfie_decode_block(").unwrap();
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
                ".param .u64 x_in_ptr",
                ".param .u64 x_out_ptr",
                ".param .u64 wq_ptr",
                ".param .u64 wk_ptr",
                ".param .u64 wv_ptr",
                ".param .u64 wo_ptr",
                ".param .u64 w_gate_ptr",
                ".param .u64 w_up_ptr",
                ".param .u64 w_down_ptr",
                ".param .u64 norm1_w_ptr",
                ".param .u64 norm2_w_ptr",
                ".param .u64 kv_base",
                ".param .u32 layer_idx",
                ".param .u32 slot_idx",
                ".param .u32 pos",
            ]
        );
        // Exactly the declared params are ever loaded.
        assert_eq!(ptx.matches("ld.param").count(), 15);
    }

    #[test]
    fn no_mad_lo_and_ascii_only() {
        let ptx = emit_decode_block_ptx(&paper_cfg());
        assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
        assert!(
            ptx.bytes().all(|b| b < 128),
            "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
        );
    }

    #[test]
    fn kv_layout_constants_match_decode_attention_emitter() {
        // The block kernel appends into the SAME pool the standalone
        // decode-attention kernel reads: their baked stride header
        // lines must be byte-identical for a matching config.
        let cfg = paper_cfg();
        let attn = crate::cfie_decode_attention::DecodeAttentionConfig {
            n_layers: cfg.n_layers,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            per_slot_max_tokens: cfg.per_slot_max_tokens,
            max_slots: cfg.max_slots,
            kv_dtype_bytes: 2,
            sm_version: cfg.sm_version,
        };
        let block_ptx = emit_decode_block_ptx(&cfg);
        let attn_ptx = crate::cfie_decode_attention::emit_decode_attention_ptx(&attn);
        let strides = |ptx: &str| -> Vec<String> {
            ptx.lines()
                .filter(|l| {
                    // header-constant lines only ("//   name = value"),
                    // not body comments mentioning the byte strides
                    l.starts_with("//   ")
                        && (l.contains("token_stride")
                            || l.contains("kv_half_stride")
                            || l.contains("layer_stride")
                            || l.contains("max_tokens "))
                })
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        let b = strides(&block_ptx);
        // token_stride, kv_half_stride, layer_stride,
        // per_slot_max_tokens, max_tokens.
        assert_eq!(b.len(), 5, "block header must bake all five constants");
        assert_eq!(b, strides(&attn_ptx));
    }

    #[test]
    fn baked_immediates_and_launch_shape() {
        let cfg = paper_cfg();
        let (ptx, meta) = emit(&cfg);
        // token_stride = 4*64 = 256 elems -> 512 bytes; layer stride =
        // 2 * 64*2048*256 * 2 bytes = 134217728 bytes.
        assert!(ptx.contains("//   token_stride        = 256"));
        assert!(ptx.contains("mul.lo.u64 %rd_kplane, %rd0, 134217728;"));
        assert!(ptx.contains("mul.lo.u32 %r_slotbase, %r_slot, 2048;"));
        assert!(ptx.contains(".version 7.0\n.target sm_80\n.address_size 64"));
        assert_eq!(meta.kernel_name, kernel_name());
        assert_eq!(meta.block_dim, 128);
        // SMEM: (3*d_model + 2*nh*hd + 128 + 128 + 2 + 128) * 4.
        let expected = (3 * 512 + 2 * 512 + 128 + 128 + 2 + 128) * 4;
        assert_eq!(meta.smem_bytes, expected);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 cfie_blk_smem[{}];", expected)));
    }

    #[test]
    fn rope_and_silu_use_documented_approx_ops() {
        let ptx = emit_decode_block_ptx(&paper_cfg());
        assert!(ptx.contains("sin.approx.f32"));
        assert!(ptx.contains("cos.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        // seq_len derives from pos, not a separate param.
        assert!(ptx.contains("add.u32 %r_sl, %r_pos, 1;"));
    }

    #[test]
    #[should_panic(expected = "even")]
    fn odd_head_dim_panics() {
        let mut cfg = paper_cfg();
        cfg.head_dim = 63;
        let _ = emit(&cfg);
    }

    #[test]
    #[should_panic(expected = "d_model")]
    fn oversized_d_model_panics() {
        let mut cfg = paper_cfg();
        cfg.d_model = 8193;
        let _ = emit(&cfg);
    }

    #[test]
    #[should_panic(expected = "d_ff")]
    fn oversized_d_ff_panics() {
        let mut cfg = paper_cfg();
        cfg.d_ff = 32769;
        let _ = emit(&cfg);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn heads_not_divisible_by_kv_heads_panics() {
        let mut cfg = paper_cfg();
        cfg.n_kv_heads = 3;
        let _ = emit(&cfg);
    }

    #[test]
    #[should_panic(expected = "fit in u32")]
    fn token_pool_overflow_panics() {
        let mut cfg = paper_cfg();
        cfg.max_slots = 1 << 21;
        cfg.per_slot_max_tokens = 1 << 12;
        let _ = emit(&cfg);
    }

    // ── cpu_reference ──────────────────────────────────────────────

    fn tiny_cfg() -> DecodeBlockConfig {
        DecodeBlockConfig {
            d_model: 4,
            head_dim: 2,
            n_heads: 1,
            n_kv_heads: 1,
            d_ff: 8,
            per_slot_max_tokens: 8,
            max_slots: 1,
            n_layers: 1,
            rope_theta: 10000.0,
            eps: 0.0,
            sm_version: 80,
        }
    }

    /// Zeroed weight set for the tiny config; tests override the
    /// pieces they exercise.
    struct TinyW {
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        wg: Vec<f32>,
        wu: Vec<f32>,
        wd: Vec<f32>,
        n1: Vec<f32>,
        n2: Vec<f32>,
    }

    fn tiny_weights() -> TinyW {
        TinyW {
            wq: vec![0.0; 2 * 4],
            wk: vec![0.0; 2 * 4],
            wv: vec![0.0; 2 * 4],
            wo: vec![0.0; 4 * 2],
            wg: vec![0.0; 8 * 4],
            wu: vec![0.0; 8 * 4],
            wd: vec![0.0; 4 * 8],
            n1: vec![1.0; 4],
            n2: vec![1.0; 4],
        }
    }

    #[test]
    fn cpu_reference_tiny_pos0_hand_computed() {
        // x = [1,-1,1,-1]: RMS = 1 and eps = 0, so with unit norm
        // weights RMSNorm is exactly the identity (the identity-ish
        // norm case).  pos = 0: one token, softmax weight exactly 1,
        // RoPE angle = 0 (identity rotation) => attn out == v row.
        //   v = Wv @ x = [x0, x1] = [1, -1]  (rows pick elements 0/1)
        //   proj = Wo @ v = [1, -1, 0, 0]    (rows: e0; e1; e0+e1; 0)
        //   FFN weights all zero => out = x + proj = [2, -2, 1, -1].
        let cfg = tiny_cfg();
        let mut w = tiny_weights();
        w.wv = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ];
        w.wo = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0, //
            0.0, 0.0,
        ];
        let x = [1.0f32, -1.0, 1.0, -1.0];
        let (mut kk, mut kv) = (Vec::new(), Vec::new());
        let out = cpu_reference(
            &cfg, &x, &w.wq, &w.wk, &w.wv, &w.wo, &w.wg, &w.wu, &w.wd, &w.n1, &w.n2, &mut kk,
            &mut kv, 0,
        );
        let expect = [2.0f32, -2.0, 1.0, -1.0];
        for (o, e) in out.iter().zip(&expect) {
            assert!((o - e).abs() < 1e-6, "out = {out:?}");
        }
        // KV append happened: one token of k (zeros) and v.
        assert_eq!(kk, vec![0.0, 0.0]);
        assert_eq!(kv, vec![1.0, -1.0]);
    }

    #[test]
    fn cpu_reference_tiny_pos1_uniform_softmax_over_two_tokens() {
        // Continue from pos 0 with x' = [1,1,1,1] (RMS = 1 again).
        // Wk = 0 => every key is zero => all scores 0 => softmax is
        // uniform 1/2 over the two tokens.
        //   v0 = [1,-1] (from pos 0), v1 = [x'0, x'1] = [1, 1]
        //   attn = (v0+v1)/2 = [1, 0]; proj = Wo @ [1,0] = [1,0,1,0]
        //   out = x' + proj = [2, 1, 2, 1].
        let cfg = tiny_cfg();
        let mut w = tiny_weights();
        w.wv = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ];
        w.wo = vec![
            1.0, 0.0, //
            0.0, 1.0, //
            1.0, 1.0, //
            0.0, 0.0,
        ];
        // wq nonzero to prove scores stay 0 through zero keys.
        w.wq = vec![0.5; 2 * 4];
        let x0 = [1.0f32, -1.0, 1.0, -1.0];
        let (mut kk, mut kv) = (Vec::new(), Vec::new());
        let _ = cpu_reference(
            &cfg, &x0, &w.wq, &w.wk, &w.wv, &w.wo, &w.wg, &w.wu, &w.wd, &w.n1, &w.n2, &mut kk,
            &mut kv, 0,
        );
        let x1 = [1.0f32, 1.0, 1.0, 1.0];
        let out = cpu_reference(
            &cfg, &x1, &w.wq, &w.wk, &w.wv, &w.wo, &w.wg, &w.wu, &w.wd, &w.n1, &w.n2, &mut kk,
            &mut kv, 1,
        );
        let expect = [2.0f32, 1.0, 2.0, 1.0];
        for (o, e) in out.iter().zip(&expect) {
            assert!((o - e).abs() < 1e-6, "out = {out:?}");
        }
        assert_eq!(kk.len(), 2 * 2, "two tokens of keys appended");
    }

    #[test]
    fn cpu_reference_tiny_ffn_silu_hand_computed() {
        // Zero attention weights => x1 = x = [1,-1,1,-1] (unit RMS,
        // eps 0, unit norms => xn2 = x1).  Only FFN row 0 is live:
        //   gate0 = x[0] = 1, up0 = x[1] = -1
        //   h0 = silu(1) * -1 = -(1/(1+e^-1)) = -0.73105857
        //   y = [h0, 0, 0, 0]; out = x + y.
        let cfg = tiny_cfg();
        let mut w = tiny_weights();
        w.wg[0] = 1.0; // gate row 0 = [1,0,0,0]
        w.wu[1] = 1.0; // up row 0 = [0,1,0,0]
        w.wd[0] = 1.0; // down row 0 = [1,0,0,0,0,0,0,0]
        let x = [1.0f32, -1.0, 1.0, -1.0];
        let (mut kk, mut kv) = (Vec::new(), Vec::new());
        let out = cpu_reference(
            &cfg, &x, &w.wq, &w.wk, &w.wv, &w.wo, &w.wg, &w.wu, &w.wd, &w.n1, &w.n2, &mut kk,
            &mut kv, 0,
        );
        let silu1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let expect = [1.0 - silu1, -1.0, 1.0, -1.0];
        for (o, e) in out.iter().zip(&expect) {
            assert!((o - e).abs() < 1e-6, "out = {out:?}");
        }
    }

    #[test]
    fn cpu_reference_zero_ffn_matches_decode_attention_composition() {
        // Invariance: with all FFN weights zero the block reduces to
        // x + Wo @ attention(rope(Wq xn), pool).  The attention factor
        // is cross-checked against the INDEPENDENT
        // cfie_decode_attention::cpu_reference implementation.
        let cfg = DecodeBlockConfig {
            d_model: 8,
            head_dim: 4,
            n_heads: 2,
            n_kv_heads: 1,
            d_ff: 16,
            per_slot_max_tokens: 8,
            max_slots: 1,
            n_layers: 1,
            rope_theta: 10000.0,
            eps: 1e-5,
            sm_version: 80,
        };
        let d = 8usize;
        let (hd, nh, nkv) = (4usize, 2usize, 1usize);
        let gen = |n: usize, f: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * f).sin() * 0.5).collect()
        };
        let wq = gen(nh * hd * d, 0.31);
        let wk = gen(nkv * hd * d, 0.47);
        let wv = gen(nkv * hd * d, 0.59);
        let wo = gen(d * nh * hd, 0.73);
        let zeros_g = vec![0.0f32; 16 * d];
        let zeros_d = vec![0.0f32; d * 16];
        let n1 = gen(d, 0.83).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let n2 = vec![1.0f32; d];
        let x = gen(d, 1.13);
        // Two pre-existing tokens in the pool.
        let pos = 2u32;
        let mut kk = gen(pos as usize * nkv * hd, 0.91);
        let mut kv = gen(pos as usize * nkv * hd, 1.07);

        let out = cpu_reference(
            &cfg, &x, &wq, &wk, &wv, &wo, &zeros_g, &zeros_g, &zeros_d, &n1, &n2, &mut kk,
            &mut kv, pos,
        );

        // Independent recomputation of the attention-only path.
        let xn = rmsnorm(&x, &n1, cfg.eps);
        let mut q = matvec(&wq, &xn, nh * hd);
        rope_rotate(&mut q, hd, pos, cfg.rope_theta);
        // kk/kv already contain the appended token from the call above.
        let ao = crate::cfie_decode_attention::cpu_reference(
            &q, &kk, &kv, nh as u32, nkv as u32, hd as u32, pos + 1,
        );
        let proj = matvec(&wo, &ao, d);
        for i in 0..d {
            let e = x[i] + proj[i];
            assert!(
                (out[i] - e).abs() < 1e-5,
                "elem {i}: block {} vs composition {e}",
                out[i]
            );
        }
    }

    #[test]
    fn cpu_reference_rmsnorm_scale_invariance() {
        // RMSNorm with eps = 0 is scale-invariant: scaling x by c
        // leaves xn (and thus q/k/v, attention, FFN) unchanged, so
        // out(c*x) - c*x == out(x) - x.
        let cfg = DecodeBlockConfig { eps: 0.0, ..tiny_cfg() };
        let mut w = tiny_weights();
        w.wv = vec![
            0.3, -0.2, 0.7, 0.1, //
            -0.5, 0.4, 0.2, 0.6,
        ];
        w.wo = vec![
            0.2, -0.1, //
            0.4, 0.3, //
            -0.6, 0.5, //
            0.1, 0.9,
        ];
        w.wg[2] = 0.8;
        w.wu[6] = -0.4;
        w.wd[9] = 0.7;
        let x = [0.5f32, -1.5, 2.0, 1.0];
        let xs: Vec<f32> = x.iter().map(|v| v * 3.0).collect();
        let (mut k1, mut v1) = (Vec::new(), Vec::new());
        let o1 = cpu_reference(
            &cfg, &x, &w.wq, &w.wk, &w.wv, &w.wo, &w.wg, &w.wu, &w.wd, &w.n1, &w.n2, &mut k1,
            &mut v1, 0,
        );
        let (mut k2, mut v2) = (Vec::new(), Vec::new());
        let o2 = cpu_reference(
            &cfg, &xs, &w.wq, &w.wk, &w.wv, &w.wo, &w.wg, &w.wu, &w.wd, &w.n1, &w.n2, &mut k2,
            &mut v2, 0,
        );
        assert_eq!(k1, k2, "keys must be scale-invariant");
        for i in 0..4 {
            let d1 = o1[i] - x[i];
            let d2 = o2[i] - xs[i];
            assert!((d1 - d2).abs() < 1e-5, "delta {i}: {d1} vs {d2}");
        }
    }

    // ── ptxas validation (skips silently when no validator present) ──

    #[test]
    fn ptxas_validates_paper_config() {
        let cfg = paper_cfg();
        let ptx = emit_decode_block_ptx(&cfg);
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                eprintln!("[skip] cfie decode-block ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie decode-block PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }
}
