//! CFIE Feature 5: per-layer KV-quant decode-attention PTX emitters.
//!
//! The paper's claim: "Layer 3's attention kernel reads INT8 K/V and
//! dequantizes in registers; layer 0's kernel reads FP16 directly.  No
//! runtime precision dispatch."  This module realises that by emitting
//! ONE kernel PER LAYER (`nsl_cfie_decode_attn_l{layer}`) — the same
//! flash-decode algorithm as `cfie_decode_attention`, except each
//! layer's K and V load paths are specialized at compile time to that
//! layer's `KvPrecision` decision from `cfie_kv_quant::KvQuantPlan`:
//!
//!   * `Fp16`: `ld.global.b16` + `cvt.f32.f16` (identical to the base
//!     kernel's load path).
//!   * `Int8`: `ld.global.s8` + `cvt.rn.f32.s8` + `mul.f32` by a
//!     per-(layer, kv-half) dequant scale — dequantized in registers.
//!   * `Int4` / `Bf16`: refused in v1 (loud assert).
//!
//! ## What is compile-time vs runtime — the honest split
//!
//! The LOAD PATH (instruction sequence, element width, every address
//! stride and the layer's pool base offset) is baked into the PTX as
//! immediates — this is the paper's "no runtime precision dispatch"
//! claim: no branch in the kernel or on the host decode path ever
//! inspects a precision tag.  The int8 dequant scale VALUE, however,
//! is runtime data: it is a symmetric per-tensor scale computed when
//! the cache half is written, and reaches the kernel as a `.f32`
//! kernel parameter (`k_scale` / `v_scale`).  Baking the scale would
//! require knowing activation magnitudes at compile time, which the
//! paper does not claim.  FP16 layers declare but never load the scale
//! params so every layer shares one 7-param launch ABI.
//!
//! ## Layout consequence: the pool becomes layer-dependent
//!
//! An INT8 half stores 1 byte/element where FP16 stores 2, so the
//! uniform `[n_layers][2][max_tokens][n_kv_heads][head_dim]` layout of
//! `cfie_kv_plan::DirectLayout` no longer has a single layer stride.
//! Instead each layer's K half and V half get their own baked base
//! OFFSET into the pool, computed by summing the byte sizes of all
//! preceding halves in layer order (K then V per layer).  See
//! [`pool_layout`] / [`total_pool_bytes`].  An all-FP16 plan
//! reproduces the base kernel's uniform derivation exactly.
//!
//! Plan/serve wiring is out of scope here; this module only exposes
//! the emitters, layout math, and CPU references.

use std::fmt::Write;

use crate::cfie_kv_quant::KvPrecision;

/// Threads per CTA and softmax tile width — must match
/// `cfie_decode_attention` (same flash-decode scheme).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

/// Kernel name for one layer's specialized decode-attention kernel.
pub fn kernel_name_for_layer(layer_idx: u32) -> String {
    format!("nsl_cfie_decode_attn_l{layer_idx}")
}

/// Compile-time layout + per-layer precision configuration.
#[derive(Debug, Clone)]
pub struct QuantDecodeAttentionConfig {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub per_slot_max_tokens: u32,
    pub max_slots: u32,
    pub sm_version: u32,
    /// `(K precision, V precision)` per layer; `len() == n_layers`.
    pub layer_precisions: Vec<(KvPrecision, KvPrecision)>,
}

/// Host-readable launch metadata emitted alongside one layer's PTX.
#[derive(Debug, Clone)]
pub struct QuantDecodeAttentionMeta {
    pub kernel_name: String,
    pub layer_idx: u32,
    pub k_precision: KvPrecision,
    pub v_precision: KvPrecision,
    /// Baked byte offset of this layer's K half from the pool base.
    pub k_offset_bytes: u64,
    /// Baked byte offset of this layer's V half from the pool base.
    pub v_offset_bytes: u64,
    pub smem_bytes: u32,
    pub block_dim: u32,
    pub grid_dim_is_n_heads: bool,
    /// Whether the kernel actually loads `k_scale` / `v_scale`.  Both
    /// params are always DECLARED (uniform launch ABI); FP16 halves
    /// ignore theirs.
    pub k_scale_param_used: bool,
    pub v_scale_param_used: bool,
}

/// Per-layer baked pool offsets (bytes from the pool base pointer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPoolOffsets {
    pub k_offset_bytes: u64,
    pub v_offset_bytes: u64,
    pub k_elem_bytes: u32,
    pub v_elem_bytes: u32,
}

/// Mirrors `gpu_specs::GpuSpec::ptx_version` (duplicated here because
/// the base emitter keeps its copy private): sm_100+ -> 8.6, sm_90+ ->
/// 8.4, else 7.0 baseline.
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

/// Bytes per stored element for a supported precision; refuses the
/// precisions the v1 kernel cannot load.
fn elem_bytes(p: KvPrecision) -> u32 {
    match p {
        KvPrecision::Fp16 => 2,
        KvPrecision::Int8 => 1,
        KvPrecision::Int4 => panic!(
            "Int4 KV halves are not supported by the per-layer decode kernel v1: \
             sub-byte addressing needs a packed load path (2 elems/byte); \
             re-plan with Fp16/Int8 or extend the emitter"
        ),
        KvPrecision::Bf16 => panic!(
            "Bf16 KV halves are not supported by the per-layer decode kernel v1: \
             only Fp16 and Int8 load paths are emitted; re-plan with Fp16/Int8"
        ),
    }
}

fn validate(cfg: &QuantDecodeAttentionConfig) {
    assert!(cfg.n_layers >= 1, "n_layers must be >= 1");
    assert_eq!(
        cfg.layer_precisions.len(),
        cfg.n_layers as usize,
        "layer_precisions must have exactly one (K, V) entry per layer"
    );
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
    assert!(
        cfg.per_slot_max_tokens >= 1 && cfg.max_slots >= 1,
        "per_slot_max_tokens and max_slots must be >= 1"
    );
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    assert!(
        max_tokens <= u32::MAX as u64,
        "global token pool (max_slots * per_slot_max_tokens = {max_tokens}) must fit in u32"
    );
    // Refuse unsupported precisions up front (loud, before any emission).
    for &(kp, vp) in &cfg.layer_precisions {
        let _ = elem_bytes(kp);
        let _ = elem_bytes(vp);
    }
}

/// Compute each layer's baked K/V half offsets by summing the byte
/// sizes of all preceding halves (layer order, K half then V half).
pub fn pool_layout(cfg: &QuantDecodeAttentionConfig) -> Vec<LayerPoolOffsets> {
    validate(cfg);
    let token_stride = cfg.n_kv_heads as u64 * cfg.head_dim as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    let half_elems = max_tokens * token_stride;
    let mut cursor = 0u64;
    let mut out = Vec::with_capacity(cfg.n_layers as usize);
    for &(kp, vp) in &cfg.layer_precisions {
        let (kb, vb) = (elem_bytes(kp), elem_bytes(vp));
        let k_offset_bytes = cursor;
        cursor += half_elems * kb as u64;
        let v_offset_bytes = cursor;
        cursor += half_elems * vb as u64;
        out.push(LayerPoolOffsets {
            k_offset_bytes,
            v_offset_bytes,
            k_elem_bytes: kb,
            v_elem_bytes: vb,
        });
    }
    out
}

/// Total pool allocation in bytes for the mixed-precision layout.
pub fn total_pool_bytes(cfg: &QuantDecodeAttentionConfig) -> u64 {
    validate(cfg);
    let token_stride = cfg.n_kv_heads as u64 * cfg.head_dim as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    let half_elems = max_tokens * token_stride;
    cfg.layer_precisions
        .iter()
        .map(|&(kp, vp)| half_elems * (elem_bytes(kp) as u64 + elem_bytes(vp) as u64))
        .sum()
}

/// Emit the specialized decode-attention kernel for one layer.
pub fn emit_layer(
    cfg: &QuantDecodeAttentionConfig,
    layer_idx: u32,
) -> (String, QuantDecodeAttentionMeta) {
    validate(cfg);
    assert!(
        layer_idx < cfg.n_layers,
        "layer_idx {} out of range (n_layers = {})",
        layer_idx,
        cfg.n_layers
    );
    let (kp, vp) = cfg.layer_precisions[layer_idx as usize];
    let offsets = pool_layout(cfg)[layer_idx as usize];
    let k_elem = offsets.k_elem_bytes as u64;
    let v_elem = offsets.v_elem_bytes as u64;

    // Strides re-derived identically to cfie_decode_attention: the
    // contiguous per-token record is [n_kv_heads][head_dim].
    let token_stride = cfg.n_kv_heads as u64 * cfg.head_dim as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    let k_token_stride_bytes = token_stride * k_elem;
    let v_token_stride_bytes = token_stride * v_elem;
    let k_head_row_bytes = cfg.head_dim as u64 * k_elem;
    let v_head_row_bytes = cfg.head_dim as u64 * v_elem;

    let group = cfg.n_heads / cfg.n_kv_heads;
    let inv_sqrt_hd = f32_imm(1.0f32 / (cfg.head_dim as f32).sqrt());
    let log2e = f32_imm(std::f32::consts::LOG2_E);
    let neg_inf = f32_imm(f32::NEG_INFINITY);
    let zero = f32_imm(0.0);

    // SMEM layout (f32): [q: head_dim][scores: TILE][rescale: 1][l: 1]
    // — identical to the base kernel.
    let scores_off = cfg.head_dim * 4;
    let rescale_off = scores_off + TILE * 4;
    let l_off = rescale_off + 4;
    let smem_bytes = l_off + 4;

    let hd = cfg.head_dim;
    let name = kernel_name_for_layer(layer_idx);
    let k_is_i8 = kp == KvPrecision::Int8;
    let v_is_i8 = vp == KvPrecision::Int8;

    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(
        w,
        "// {} - CFIE per-layer KV-quant decode attention (flash-decode).",
        name
    )
    .unwrap();
    writeln!(
        w,
        "// Layer {} of {}: K={}, V={} (precision baked; no runtime dispatch).",
        layer_idx,
        cfg.n_layers,
        kp.as_str(),
        vp.as_str()
    )
    .unwrap();
    writeln!(
        w,
        "// Int8 dequant scale VALUES arrive as runtime .f32 params; the load",
    )
    .unwrap();
    writeln!(
        w,
        "// path, element widths and pool offsets below are all immediates.",
    )
    .unwrap();
    writeln!(w, "// Baked layout constants:").unwrap();
    writeln!(w, "//   token_stride        = {}", token_stride).unwrap();
    writeln!(w, "//   per_slot_max_tokens = {}", cfg.per_slot_max_tokens).unwrap();
    writeln!(w, "//   max_tokens          = {}", max_tokens).unwrap();
    writeln!(w, "//   gqa_group_size      = {}", group).unwrap();
    writeln!(w, "//   k_offset_bytes      = {}", offsets.k_offset_bytes).unwrap();
    writeln!(w, "//   v_offset_bytes      = {}", offsets.v_offset_bytes).unwrap();
    writeln!(w, "//   k_elem_bytes        = {}", k_elem).unwrap();
    writeln!(w, "//   v_elem_bytes        = {}", v_elem).unwrap();
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", ptx_version_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".shared .align 4 .b8 cfie_smem[{}];", smem_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", name).unwrap();
    writeln!(w, "    .param .u64 q_ptr,").unwrap();
    writeln!(w, "    .param .u64 kv_base,").unwrap();
    writeln!(w, "    .param .u64 out_ptr,").unwrap();
    writeln!(w, "    .param .u32 slot_idx,").unwrap();
    writeln!(w, "    .param .u32 seq_len,").unwrap();
    writeln!(w, "    .param .f32 k_scale,").unwrap();
    writeln!(w, "    .param .f32 v_scale").unwrap();
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
        "    .reg .f32 %f_q, %f_k, %f_v, %f_p, %f_s, %f_dot, %f_acc, %f_m, %f_l, %f_tm, %f_rs, %f_rs2, %f_t1, %f_lf, %f_o, %f_t0, %f_ks, %f_vs;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_tid, %r_head, %r_kvhead, %r_slot, %r_seqlen, %r_sbase, %r_slotbase, %r_hoff, %r_tile, %r_rem, %r_tcnt, %r_tok, %r_g, %r_g0, %r_d, %r_qsm, %r_j, %r_sp, %r_t1, %r_t2, %r_t3, %r_t4;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_q, %rd_kv, %rd_out, %rd_kplane, %rd_vplane, %rd_khoff, %rd_vhoff, %rd_koff, %rd_kaddr, %rd_voff, %rd_vaddr, %rd_t1, %rd_t2, %rd_t3;"
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_q, [q_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_kv, [kv_base];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_out, [out_ptr];").unwrap();
    writeln!(w, "    ld.param.u32 %r_slot, [slot_idx];").unwrap();
    writeln!(w, "    ld.param.u32 %r_seqlen, [seq_len];").unwrap();
    if k_is_i8 {
        writeln!(w, "    // runtime symmetric dequant scale for the int8 K half").unwrap();
        writeln!(w, "    ld.param.f32 %f_ks, [k_scale];").unwrap();
    }
    if v_is_i8 {
        writeln!(w, "    // runtime symmetric dequant scale for the int8 V half").unwrap();
        writeln!(w, "    ld.param.f32 %f_vs, [v_scale];").unwrap();
    }
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_head, %ctaid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_sbase, cfie_smem;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // GQA: kv_head = q_head / group_size (baked divisor)").unwrap();
    writeln!(w, "    div.u32 %r_kvhead, %r_head, {};", group).unwrap();
    writeln!(w).unwrap();
    writeln!(
        w,
        "    // per-layer baked pool offsets: layer {} K half at +{}, V half at +{}",
        layer_idx, offsets.k_offset_bytes, offsets.v_offset_bytes
    )
    .unwrap();
    writeln!(w, "    add.u64 %rd_kplane, %rd_kv, {};", offsets.k_offset_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_vplane, %rd_kv, {};", offsets.v_offset_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // slot's first global token: slot_idx * per_slot_max_tokens").unwrap();
    writeln!(
        w,
        "    mul.lo.u32 %r_slotbase, %r_slot, {};",
        cfg.per_slot_max_tokens
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(
        w,
        "    // byte offsets of kv_head's row inside one K / V token record"
    )
    .unwrap();
    writeln!(w, "    mul.lo.u32 %r_hoff, %r_kvhead, {};", k_head_row_bytes).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_khoff, %r_hoff;").unwrap();
    writeln!(w, "    mul.lo.u32 %r_hoff, %r_kvhead, {};", v_head_row_bytes).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_vhoff, %r_hoff;").unwrap();
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
    writeln!(w, "    mul.wide.u32 %rd_koff, %r_g, {};", k_token_stride_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_kaddr, %rd_kplane, %rd_koff;").unwrap();
    writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, %rd_khoff;").unwrap();
    writeln!(w, "    mov.f32 %f_dot, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_d, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_qsm, %r_sbase;").unwrap();
    writeln!(w, "DOT_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_d, %r_d, {};", hd).unwrap();
    writeln!(w, "    @%p_d bra DOT_DONE;").unwrap();
    if k_is_i8 {
        writeln!(w, "    // K load path: int8, dequantized in registers").unwrap();
        writeln!(w, "    ld.global.s8 %h_k, [%rd_kaddr];").unwrap();
        writeln!(w, "    cvt.rn.f32.s8 %f_k, %h_k;").unwrap();
        writeln!(w, "    mul.f32 %f_k, %f_k, %f_ks;").unwrap();
    } else {
        writeln!(w, "    // K load path: fp16, read directly").unwrap();
        writeln!(w, "    ld.global.b16 %h_k, [%rd_kaddr];").unwrap();
        writeln!(w, "    cvt.f32.f16 %f_k, %h_k;").unwrap();
    }
    writeln!(w, "    ld.shared.f32 %f_q, [%r_qsm];").unwrap();
    writeln!(w, "    fma.rn.f32 %f_dot, %f_k, %f_q, %f_dot;").unwrap();
    writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, {};", k_elem).unwrap();
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
    writeln!(w, "    mul.wide.u32 %rd_voff, %r_g0, {};", v_token_stride_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vplane, %rd_voff;").unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, %rd_vhoff;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t2, %r_tid, {};", v_elem).unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, %rd_t2;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_sp, %r_sbase;").unwrap();
    writeln!(w, "ACC_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_j2, %r_j, %r_tcnt;").unwrap();
    writeln!(w, "    @%p_j2 bra ACC_TAIL;").unwrap();
    writeln!(w, "    ld.shared.f32 %f_p, [%r_sp+{}];", scores_off).unwrap();
    if v_is_i8 {
        writeln!(w, "    // V load path: int8, dequantized in registers").unwrap();
        writeln!(w, "    ld.global.s8 %h_v, [%rd_vaddr];").unwrap();
        writeln!(w, "    cvt.rn.f32.s8 %f_v, %h_v;").unwrap();
        writeln!(w, "    mul.f32 %f_v, %f_v, %f_vs;").unwrap();
    } else {
        writeln!(w, "    // V load path: fp16, read directly").unwrap();
        writeln!(w, "    ld.global.b16 %h_v, [%rd_vaddr];").unwrap();
        writeln!(w, "    cvt.f32.f16 %f_v, %h_v;").unwrap();
    }
    writeln!(w, "    fma.rn.f32 %f_acc, %f_p, %f_v, %f_acc;").unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, {};", v_token_stride_bytes).unwrap();
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

    let meta = QuantDecodeAttentionMeta {
        kernel_name: name,
        layer_idx,
        k_precision: kp,
        v_precision: vp,
        k_offset_bytes: offsets.k_offset_bytes,
        v_offset_bytes: offsets.v_offset_bytes,
        smem_bytes,
        block_dim: BLOCK_DIM,
        grid_dim_is_n_heads: true,
        k_scale_param_used: k_is_i8,
        v_scale_param_used: v_is_i8,
    };
    (p, meta)
}

/// Emit one specialized kernel per layer, in layer order.
pub fn emit_all(cfg: &QuantDecodeAttentionConfig) -> Vec<(String, QuantDecodeAttentionMeta)> {
    validate(cfg);
    (0..cfg.n_layers).map(|l| emit_layer(cfg, l)).collect()
}

// ---------------------------------------------------------------------------
// CPU reference (mirrors the kernel's register dequant)
// ---------------------------------------------------------------------------

/// Symmetric per-tensor int8 quantization with `scale = max_abs / 127`.
/// All-zero input maps to `(zeros, scale = 1.0)`.
pub fn quantize_symmetric_i8(vals: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs == 0.0 {
        return (vec![0i8; vals.len()], 1.0);
    }
    let scale = max_abs / 127.0;
    let q = vals
        .iter()
        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (q, scale)
}

/// One KV half as the CPU reference consumes it.  `Fp16` is modelled
/// as f32 (the base `cpu_reference` does the same); `Int8` mirrors the
/// kernel's `i8 -> f32 * scale` register dequant.
#[derive(Debug, Clone, Copy)]
pub enum CpuKvHalf<'a> {
    Fp16(&'a [f32]),
    Int8 { data: &'a [i8], scale: f32 },
}

fn dequant_half(h: CpuKvHalf<'_>) -> Vec<f32> {
    match h {
        CpuKvHalf::Fp16(d) => d.to_vec(),
        CpuKvHalf::Int8 { data, scale } => {
            data.iter().map(|&q| q as f32 * scale).collect()
        }
    }
}

/// CPU reference for one layer's specialized kernel: dequantize each
/// half exactly as the kernel does, then run the identical attention
/// math as `cfie_decode_attention::cpu_reference`.  Layouts match the
/// base reference: `q` is `[n_heads][head_dim]`, each half is
/// `[seq_len][n_kv_heads][head_dim]`.
pub fn cpu_reference_layer(
    q: &[f32],
    k: CpuKvHalf<'_>,
    v: CpuKvHalf<'_>,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Vec<f32> {
    let kf = dequant_half(k);
    let vf = dequant_half(v);
    crate::cfie_decode_attention::cpu_reference(q, &kf, &vf, n_heads, n_kv_heads, head_dim, seq_len)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_decode_attention::{
        cpu_reference, emit_decode_attention_ptx, DecodeAttentionConfig,
    };

    /// Mixed 4-layer fixture: paper-shaped edge FP16, middle INT8.
    fn mixed_cfg() -> QuantDecodeAttentionConfig {
        QuantDecodeAttentionConfig {
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 64,
            per_slot_max_tokens: 256,
            max_slots: 4,
            sm_version: 80,
            layer_precisions: vec![
                (KvPrecision::Fp16, KvPrecision::Fp16),
                (KvPrecision::Fp16, KvPrecision::Int8),
                (KvPrecision::Int8, KvPrecision::Int8),
                (KvPrecision::Fp16, KvPrecision::Fp16),
            ],
        }
    }

    fn all_fp16_cfg() -> QuantDecodeAttentionConfig {
        let mut cfg = mixed_cfg();
        cfg.layer_precisions =
            vec![(KvPrecision::Fp16, KvPrecision::Fp16); cfg.n_layers as usize];
        cfg
    }

    /// Base-kernel config with dims matching [`mixed_cfg`] (f16 KV).
    fn matching_base_cfg() -> DecodeAttentionConfig {
        DecodeAttentionConfig {
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 64,
            per_slot_max_tokens: 256,
            max_slots: 4,
            kv_dtype_bytes: 2,
            sm_version: 80,
        }
    }

    /// Parse `//   <key><spaces>= <number>` header comments.
    fn comment_value(ptx: &str, key: &str) -> u64 {
        let prefix = format!("//   {key}");
        let line = ptx
            .lines()
            .find(|l| l.starts_with(&prefix) && l.contains('='))
            .unwrap_or_else(|| panic!("no header comment starting with {prefix:?}"));
        line.rsplit('=').next().unwrap().trim().parse().unwrap()
    }

    /// Extract the immediate operand of the first line starting with
    /// `prefix` (after trimming), e.g. the baked stride of a mul.wide.
    fn trailing_imm(ptx: &str, prefix: &str) -> u64 {
        let line = ptx
            .lines()
            .map(str::trim)
            .find(|l| l.starts_with(prefix))
            .unwrap_or_else(|| panic!("no instruction starting with {prefix:?}"));
        line.strip_prefix(prefix)
            .unwrap()
            .trim()
            .trim_end_matches(';')
            .parse()
            .unwrap()
    }

    // ── pool layout ────────────────────────────────────────────────

    #[test]
    fn pool_layout_mixed_hand_computed() {
        // Small fixture: token_stride = 2*4 = 8 elems; max_tokens =
        // 2*8 = 16; half = 128 elems -> fp16 half 256 B, int8 half 128 B.
        let cfg = QuantDecodeAttentionConfig {
            n_layers: 4,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: 4,
            per_slot_max_tokens: 8,
            max_slots: 2,
            sm_version: 80,
            layer_precisions: vec![
                (KvPrecision::Fp16, KvPrecision::Fp16),
                (KvPrecision::Fp16, KvPrecision::Int8),
                (KvPrecision::Int8, KvPrecision::Int8),
                (KvPrecision::Fp16, KvPrecision::Fp16),
            ],
        };
        let lo = pool_layout(&cfg);
        assert_eq!(lo.len(), 4);
        // L0: fp16/fp16 -> k@0, v@256
        assert_eq!((lo[0].k_offset_bytes, lo[0].v_offset_bytes), (0, 256));
        assert_eq!((lo[0].k_elem_bytes, lo[0].v_elem_bytes), (2, 2));
        // L1: fp16/int8 -> k@512, v@768
        assert_eq!((lo[1].k_offset_bytes, lo[1].v_offset_bytes), (512, 768));
        assert_eq!((lo[1].k_elem_bytes, lo[1].v_elem_bytes), (2, 1));
        // L2: int8/int8 -> k@896, v@1024
        assert_eq!((lo[2].k_offset_bytes, lo[2].v_offset_bytes), (896, 1024));
        // L3: fp16/fp16 -> k@1152, v@1408
        assert_eq!((lo[3].k_offset_bytes, lo[3].v_offset_bytes), (1152, 1408));
        assert_eq!(total_pool_bytes(&cfg), 1664);
    }

    #[test]
    fn all_fp16_layout_reproduces_base_kernel_uniform_derivation() {
        let cfg = all_fp16_cfg();
        let base_ptx = emit_decode_attention_ptx(&matching_base_cfg());
        // Base derivation, read back from its own PTX header (elements).
        let kv_half_elems = comment_value(&base_ptx, "kv_half_stride");
        let layer_elems = comment_value(&base_ptx, "layer_stride");
        let kv_half_bytes = kv_half_elems * 2;
        let layer_bytes = layer_elems * 2;
        let lo = pool_layout(&cfg);
        for (l, off) in lo.iter().enumerate() {
            assert_eq!(off.k_offset_bytes, l as u64 * layer_bytes);
            assert_eq!(off.v_offset_bytes, off.k_offset_bytes + kv_half_bytes);
        }
        assert_eq!(total_pool_bytes(&cfg), cfg.n_layers as u64 * layer_bytes);
    }

    // ── cross-module stride consistency (string inspection only) ──

    #[test]
    fn strides_match_base_kernel_via_ptx_inspection() {
        let base_ptx = emit_decode_attention_ptx(&matching_base_cfg());
        let (l0_ptx, _) = emit_layer(&all_fp16_cfg(), 0);
        // Element token stride, from both headers.
        assert_eq!(
            comment_value(&base_ptx, "token_stride"),
            comment_value(&l0_ptx, "token_stride"),
        );
        // Byte token stride baked into the K address arithmetic.
        let base_kstride = trailing_imm(&base_ptx, "mul.wide.u32 %rd_koff, %r_g,");
        let quant_kstride = trailing_imm(&l0_ptx, "mul.wide.u32 %rd_koff, %r_g,");
        assert_eq!(base_kstride, quant_kstride);
        // V-side token stride baked into the accumulate loop.
        let base_vstride = trailing_imm(&base_ptx, "mul.wide.u32 %rd_voff, %r_g0,");
        let quant_vstride = trailing_imm(&l0_ptx, "mul.wide.u32 %rd_voff, %r_g0,");
        assert_eq!(base_vstride, quant_vstride);
        // Slot base immediate.
        let base_slot = trailing_imm(&base_ptx, "mul.lo.u32 %r_slotbase, %r_slot,");
        let quant_slot = trailing_imm(&l0_ptx, "mul.lo.u32 %r_slotbase, %r_slot,");
        assert_eq!(base_slot, quant_slot);
    }

    // ── structural: per-precision load paths ──────────────────────

    #[test]
    fn fp16_layer_reads_f16_directly_and_has_no_s8_loads() {
        let (ptx, meta) = emit_layer(&mixed_cfg(), 0);
        assert!(ptx.contains(".visible .entry nsl_cfie_decode_attn_l0("));
        assert!(ptx.contains("cvt.f32.f16"));
        assert!(!ptx.contains("ld.global.s8"));
        assert!(!ptx.contains("cvt.rn.f32.s8"));
        // Scale params are declared (uniform ABI) but never loaded.
        assert!(ptx.contains(".param .f32 k_scale"));
        assert!(ptx.contains(".param .f32 v_scale"));
        assert_eq!(ptx.matches("ld.param").count(), 5);
        assert!(!meta.k_scale_param_used && !meta.v_scale_param_used);
    }

    #[test]
    fn int8_layer_loads_s8_and_dequantizes_with_scale_params() {
        let (ptx, meta) = emit_layer(&mixed_cfg(), 2);
        assert!(ptx.contains(".visible .entry nsl_cfie_decode_attn_l2("));
        assert_eq!(ptx.matches("ld.global.s8").count(), 2, "K and V loads");
        assert_eq!(ptx.matches("cvt.rn.f32.s8").count(), 2);
        assert!(ptx.contains("ld.param.f32 %f_ks, [k_scale];"));
        assert!(ptx.contains("ld.param.f32 %f_vs, [v_scale];"));
        assert!(ptx.contains("mul.f32 %f_k, %f_k, %f_ks;"));
        assert!(ptx.contains("mul.f32 %f_v, %f_v, %f_vs;"));
        // 5 direct params + 2 scales.
        assert_eq!(ptx.matches("ld.param").count(), 7);
        // No f16 conversion anywhere in a pure-int8 layer.
        assert!(!ptx.contains("cvt.f32.f16"));
        assert!(meta.k_scale_param_used && meta.v_scale_param_used);
    }

    #[test]
    fn mixed_layer_specializes_k_and_v_independently() {
        // Layer 1: FP16 K, INT8 V.
        let (ptx, meta) = emit_layer(&mixed_cfg(), 1);
        assert!(ptx.contains("cvt.f32.f16 %f_k, %h_k;"), "K path stays f16");
        assert!(ptx.contains("ld.global.s8 %h_v, [%rd_vaddr];"), "V path is s8");
        assert!(ptx.contains("mul.f32 %f_v, %f_v, %f_vs;"));
        assert!(!ptx.contains("mul.f32 %f_k, %f_k, %f_ks;"));
        assert_eq!(ptx.matches("ld.param").count(), 6, "only v_scale loaded");
        assert!(!meta.k_scale_param_used && meta.v_scale_param_used);
    }

    #[test]
    fn per_layer_offsets_are_distinct_baked_immediates() {
        let cfg = mixed_cfg();
        let all = emit_all(&cfg);
        assert_eq!(all.len(), cfg.n_layers as usize);
        let mut k_offs = Vec::new();
        let mut v_offs = Vec::new();
        for (l, (ptx, meta)) in all.iter().enumerate() {
            assert_eq!(meta.kernel_name, kernel_name_for_layer(l as u32));
            let k = trailing_imm(ptx, "add.u64 %rd_kplane, %rd_kv,");
            let v = trailing_imm(ptx, "add.u64 %rd_vplane, %rd_kv,");
            assert_eq!(k, meta.k_offset_bytes);
            assert_eq!(v, meta.v_offset_bytes);
            k_offs.push(k);
            v_offs.push(v);
        }
        // Offsets strictly increase in (k, v) interleaved order.
        let mut interleaved: Vec<u64> = Vec::new();
        for i in 0..k_offs.len() {
            interleaved.push(k_offs[i]);
            interleaved.push(v_offs[i]);
        }
        assert!(interleaved.windows(2).all(|w| w[0] < w[1]));
        // And no ld.param of a layer index anywhere: the layer is baked.
        for (ptx, _) in &all {
            assert!(!ptx.contains("layer_idx"));
        }
    }

    #[test]
    fn no_mad_lo_and_ascii_only_all_layers() {
        for (ptx, _) in emit_all(&mixed_cfg()) {
            assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
            assert!(
                ptx.bytes().all(|b| b < 128),
                "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
            );
        }
    }

    #[test]
    fn header_and_meta_match_base_kernel_launch_shape() {
        let (ptx, meta) = emit_layer(&mixed_cfg(), 3);
        assert!(ptx.starts_with("//"));
        assert!(ptx.contains(".version 7.0\n.target sm_80\n.address_size 64"));
        assert_eq!(meta.block_dim, 128);
        assert!(meta.grid_dim_is_n_heads);
        // Same SMEM formula as the base kernel: q + scores + rescale + l.
        assert_eq!(meta.smem_bytes, 64 * 4 + 128 * 4 + 8);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 cfie_smem[{}];", meta.smem_bytes)));
    }

    // ── refusals ───────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "Int4")]
    fn int4_layer_refused_loudly() {
        let mut cfg = mixed_cfg();
        cfg.layer_precisions[2] = (KvPrecision::Int4, KvPrecision::Int8);
        let _ = emit_layer(&cfg, 2);
    }

    #[test]
    #[should_panic(expected = "Bf16")]
    fn bf16_layer_refused_loudly() {
        let mut cfg = mixed_cfg();
        cfg.layer_precisions[1] = (KvPrecision::Fp16, KvPrecision::Bf16);
        let _ = pool_layout(&cfg);
    }

    #[test]
    #[should_panic(expected = "layer_precisions")]
    fn precision_list_length_mismatch_refused() {
        let mut cfg = mixed_cfg();
        cfg.layer_precisions.pop();
        let _ = emit_all(&cfg);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn layer_index_out_of_range_refused() {
        let _ = emit_layer(&mixed_cfg(), 4);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn gqa_divisibility_still_enforced() {
        let mut cfg = mixed_cfg();
        cfg.n_kv_heads = 3;
        let _ = emit_layer(&cfg, 0);
    }

    // ── cpu reference + quantization ──────────────────────────────

    #[test]
    fn quantize_symmetric_maps_extremes_to_127() {
        let (q, s) = quantize_symmetric_i8(&[0.5, -1.0, 0.0, 1.0]);
        assert_eq!(q, vec![64, -127, 0, 127]);
        assert!((s - 1.0 / 127.0).abs() < 1e-9);
        let (qz, sz) = quantize_symmetric_i8(&[0.0, 0.0]);
        assert_eq!(qz, vec![0, 0]);
        assert_eq!(sz, 1.0);
    }

    #[test]
    fn fp16_halves_pass_through_to_base_reference_exactly() {
        let (nh, nkv, hd, sl) = (2u32, 1u32, 4u32, 5u32);
        let q: Vec<f32> = (0..nh * hd).map(|i| ((i as f32) * 0.37).sin()).collect();
        let k: Vec<f32> = (0..sl * nkv * hd).map(|i| ((i as f32) * 0.53).cos()).collect();
        let v: Vec<f32> = (0..sl * nkv * hd).map(|i| ((i as f32) * 0.29).sin()).collect();
        let base = cpu_reference(&q, &k, &v, nh, nkv, hd, sl);
        let out = cpu_reference_layer(
            &q,
            CpuKvHalf::Fp16(&k),
            CpuKvHalf::Fp16(&v),
            nh,
            nkv,
            hd,
            sl,
        );
        assert_eq!(out, base);
    }

    #[test]
    fn int8_roundtrip_attention_close_to_f32_reference() {
        let (nh, nkv, hd, sl) = (4u32, 2u32, 8u32, 16u32);
        let q: Vec<f32> = (0..nh * hd).map(|i| ((i as f32) * 0.37).sin()).collect();
        let k: Vec<f32> = (0..sl * nkv * hd).map(|i| ((i as f32) * 0.53).cos()).collect();
        let v: Vec<f32> = (0..sl * nkv * hd).map(|i| ((i as f32) * 0.29).sin()).collect();
        let base = cpu_reference(&q, &k, &v, nh, nkv, hd, sl);
        let (kq, ks) = quantize_symmetric_i8(&k);
        let (vq, vs) = quantize_symmetric_i8(&v);
        let out = cpu_reference_layer(
            &q,
            CpuKvHalf::Int8 { data: &kq, scale: ks },
            CpuKvHalf::Int8 { data: &vq, scale: vs },
            nh,
            nkv,
            hd,
            sl,
        );
        assert_eq!(out.len(), base.len());
        let mut max_err = 0.0f32;
        for (a, b) in out.iter().zip(&base) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.05, "int8 roundtrip max abs err {max_err}");
        assert!(max_err > 0.0, "quantization must actually perturb the output");
    }

    // ── ptxas validation (skips silently when no validator present) ──

    #[test]
    fn ptxas_validates_mixed_four_layer_config() {
        for (ptx, meta) in emit_all(&mixed_cfg()) {
            match crate::ptxas_validation::validate_ptx(&ptx) {
                Ok(()) => {}
                Err(msg) if msg.contains("nvcc not available") => {
                    eprintln!(
                        "[skip] cfie kv-quant ptxas validation ({}) - no validator: {msg}",
                        meta.kernel_name
                    );
                }
                Err(msg) => panic!(
                    "cfie kv-quant PTX rejected for {}:\n{msg}\n\nEmitted PTX:\n{ptx}",
                    meta.kernel_name
                ),
            }
        }
    }
}
