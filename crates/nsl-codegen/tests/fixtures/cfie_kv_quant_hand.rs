// Frozen fixture (roadmap A2 step 9): the hand-written PTX emitter for the
// CFIE per-layer KV-quant decode-attention kernels, exactly as `crates/
// nsl-codegen/src/cfie_kv_quant_ptx.rs` held it at 188083eb, before the
// kernels moved onto KIR. Lines below this comment are that file's first
// 557 lines, unedited; `cfie_kv_quant_kir_equivalence.rs` includes it with
// `#[path]` (supplying the two `crate::` items it names) and runs its
// output against the KIR kernels'. Do not change it: it is the
// pre-migration behaviour the equivalence claim is about.
//
// Not scanned by the hand-PTX freeze: nothing under `tests/` is.

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
    writeln!(w, ".version {}", crate::gpu_specs::ptx_isa_for_sm(cfg.sm_version)).unwrap();
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

