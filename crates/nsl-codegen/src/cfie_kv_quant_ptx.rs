//! CFIE Feature 5: per-layer KV-quant decode-attention kernels, built as KIR.
//!
//! The paper's claim: "Layer 3's attention kernel reads INT8 K/V and
//! dequantizes in registers; layer 0's kernel reads FP16 directly.  No
//! runtime precision dispatch."  This module realises that by emitting
//! ONE kernel PER LAYER (`nsl_cfie_decode_attn_l{layer}`) — the same
//! flash-decode algorithm as `cfie_decode_attention`, except each
//! layer's K and V load paths are specialized at compile time to that
//! layer's `KvPrecision` decision from `cfie_kv_quant::KvQuantPlan`:
//!
//!   * `Fp16`: an f16 load widened to f32 (the base kernel's load path).
//!   * `Int8`: an s8 load, `cvt.rn.f32.s8`, and a multiply by a
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
//! paper does not claim.  FP16 layers declare but never read the scale
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
//! An f16 half must start on an even byte, and one can only land on an
//! odd byte after an int8 half with an odd element count
//! (`max_tokens * n_kv_heads * head_dim`). Such a layout is refused (see
//! `validate`): the load would be misaligned, which the GPU faults on.
//!
//! ## KIR (roadmap A2 step 9)
//!
//! Each layer's kernel was hand-assembled PTX text until A2 step 9; it is
//! now the flash-decode [`KernelIR`] `cfie_decode_attention` builds, over
//! a [`PoolLayout::Baked`] pool: the kernel names each half by its baked
//! byte offset from `kv_base` and reads it in the half's element type.
//! The algorithm and the order of every floating-point operation are the
//! hand kernels'; `tests/cfie_kv_quant_kir_equivalence.rs` runs the frozen
//! hand emitter and this one side by side on a PTX interpreter and
//! requires the same output bits. Like the base kernel, the module now
//! targets the KIR floor (`.version 7.0` / `.target sm_70`) and the driver
//! JIT-compiles it forward, so the configuration has no `sm_version`.
//!
//! Plan/serve wiring is out of scope here; this module only exposes
//! the emitters, layout math, and CPU references.

use std::fmt::Write;

use crate::backend_ptx::lower_kir_to_ptx;
use crate::cfie_decode_attention::{build_flash_decode, BakedHalf, FlashDecode, PoolLayout};
use crate::cfie_kv_quant::KvPrecision;
use crate::kernel_ir::KernelIR;

/// Threads per CTA and softmax tile width — must match
/// `cfie_decode_attention` (same flash-decode scheme).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

/// Kernel name for one layer's specialized decode-attention kernel.
pub fn kernel_name_for_layer(layer_idx: u32) -> String {
    format!("nsl_cfie_decode_attn_l{layer_idx}")
}

/// Compile-time layout + per-layer precision configuration.
///
/// There is no `sm_version`: each kernel targets the KIR floor and the
/// driver JIT-compiles it for the device it is loaded on (see the module
/// docs).
#[derive(Debug, Clone)]
pub struct QuantDecodeAttentionConfig {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub per_slot_max_tokens: u32,
    pub max_slots: u32,
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
    /// Whether the kernel actually reads `k_scale` / `v_scale`.  Both
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

/// Each layer's K/V half offsets: the byte sizes of all preceding halves
/// summed (layer order, K half then V half). No checks; see
/// [`pool_layout`].
fn offsets(cfg: &QuantDecodeAttentionConfig) -> Vec<LayerPoolOffsets> {
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
    // An f16 half on an odd byte would be a misaligned 2-byte load.
    for (l, o) in offsets(cfg).iter().enumerate() {
        for (half, off, elem) in [("K", o.k_offset_bytes, o.k_elem_bytes), ("V", o.v_offset_bytes, o.v_elem_bytes)] {
            assert!(
                off % elem as u64 == 0,
                "layer {l}'s {half} half would start at byte {off}, not a multiple of its \
                 {elem}-byte element: an int8 half before it has an odd element count \
                 (max_tokens * n_kv_heads * head_dim); re-plan the precisions or the geometry"
            );
        }
    }
}

/// Compute each layer's baked K/V half offsets by summing the byte
/// sizes of all preceding halves (layer order, K half then V half).
pub fn pool_layout(cfg: &QuantDecodeAttentionConfig) -> Vec<LayerPoolOffsets> {
    validate(cfg);
    offsets(cfg)
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

fn check_layer(cfg: &QuantDecodeAttentionConfig, layer_idx: u32) {
    validate(cfg);
    assert!(
        layer_idx < cfg.n_layers,
        "layer_idx {} out of range (n_layers = {})",
        layer_idx,
        cfg.n_layers
    );
}

/// Build one layer's specialized decode-attention kernel as KIR.
///
/// The flash-decode kernel `cfie_decode_attention::build` documents, over
/// this layer's two halves: each at its baked byte offset from `kv_base`,
/// read as f16, or as int8 dequantized by its half's scale param.
pub fn build_layer(cfg: &QuantDecodeAttentionConfig, layer_idx: u32) -> KernelIR {
    check_layer(cfg, layer_idx);
    let (kp, vp) = cfg.layer_precisions[layer_idx as usize];
    let o = offsets(cfg)[layer_idx as usize];
    let name = kernel_name_for_layer(layer_idx);
    build_flash_decode(&FlashDecode {
        name: &name,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        per_slot_max_tokens: cfg.per_slot_max_tokens,
        max_slots: cfg.max_slots,
        pool: PoolLayout::Baked {
            k: BakedHalf { offset_bytes: o.k_offset_bytes, int8: kp == KvPrecision::Int8 },
            v: BakedHalf { offset_bytes: o.v_offset_bytes, int8: vp == KvPrecision::Int8 },
        },
    })
}

/// The `//` header: the layer's precisions and what its kernel bakes.
/// The hand emitter's lines, unchanged.
fn header_comment(cfg: &QuantDecodeAttentionConfig, layer_idx: u32) -> String {
    let (kp, vp) = cfg.layer_precisions[layer_idx as usize];
    let o = offsets(cfg)[layer_idx as usize];
    let token_stride = cfg.n_kv_heads as u64 * cfg.head_dim as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    let mut w = String::new();
    writeln!(w, "//").unwrap();
    writeln!(
        w,
        "// {} - CFIE per-layer KV-quant decode attention (flash-decode).",
        kernel_name_for_layer(layer_idx)
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
    writeln!(w, "// Int8 dequant scale VALUES arrive as runtime .f32 params; the load").unwrap();
    writeln!(w, "// path, element widths and pool offsets below are all immediates.").unwrap();
    writeln!(w, "// Baked layout constants:").unwrap();
    writeln!(w, "//   token_stride        = {}", token_stride).unwrap();
    writeln!(w, "//   per_slot_max_tokens = {}", cfg.per_slot_max_tokens).unwrap();
    writeln!(w, "//   max_tokens          = {}", max_tokens).unwrap();
    writeln!(w, "//   gqa_group_size      = {}", cfg.n_heads / cfg.n_kv_heads).unwrap();
    writeln!(w, "//   k_offset_bytes      = {}", o.k_offset_bytes).unwrap();
    writeln!(w, "//   v_offset_bytes      = {}", o.v_offset_bytes).unwrap();
    writeln!(w, "//   k_elem_bytes        = {}", o.k_elem_bytes).unwrap();
    writeln!(w, "//   v_elem_bytes        = {}", o.v_elem_bytes).unwrap();
    writeln!(w, "//").unwrap();
    w
}

/// Emit the specialized decode-attention kernel for one layer: build,
/// verify, lower, and prefix the `//` header.
///
/// The returned text carries no NUL: the serve path appends the one
/// `cuModuleLoadData` needs when it embeds the module.
///
/// # Panics
///
/// On a configuration outside the contract (see `validate`), on a
/// `layer_idx` out of range, and if the built kernel fails KIR
/// verification — a bug in the builder, not a condition a caller can
/// provoke.
pub fn emit_layer(
    cfg: &QuantDecodeAttentionConfig,
    layer_idx: u32,
) -> (String, QuantDecodeAttentionMeta) {
    let ir = build_layer(cfg, layer_idx);
    let name = kernel_name_for_layer(layer_idx);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{name} failed KIR verification: {errors:?}");
    }
    let smem_bytes = ir
        .smem_layout
        .total_bytes()
        .expect("a verified layout has a size");
    let module = lower_kir_to_ptx(&ir);
    let module = module.strip_suffix(&[0]).unwrap_or(&module);
    let module = std::str::from_utf8(module).expect("the KIR printer emits ASCII");

    let mut p = header_comment(cfg, layer_idx);
    p.push_str(module);

    let (kp, vp) = cfg.layer_precisions[layer_idx as usize];
    let o = offsets(cfg)[layer_idx as usize];
    let meta = QuantDecodeAttentionMeta {
        kernel_name: name,
        layer_idx,
        k_precision: kp,
        v_precision: vp,
        k_offset_bytes: o.k_offset_bytes,
        v_offset_bytes: o.v_offset_bytes,
        smem_bytes,
        block_dim: BLOCK_DIM,
        grid_dim_is_n_heads: true,
        k_scale_param_used: kp == KvPrecision::Int8,
        v_scale_param_used: vp == KvPrecision::Int8,
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
        cpu_reference, emit_decode_attention_ptx, kv_strides, DecodeAttentionConfig,
    };
    use crate::kernel_ir::{AddressSpace, ConstValue, KirConst, KirOp, KirType};

    /// Mixed 4-layer fixture: paper-shaped edge FP16, middle INT8.
    fn mixed_cfg() -> QuantDecodeAttentionConfig {
        QuantDecodeAttentionConfig {
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 64,
            per_slot_max_tokens: 256,
            max_slots: 4,
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

    fn ops(ir: &KernelIR) -> impl Iterator<Item = &KirOp> {
        ir.blocks.iter().flat_map(|b| b.ops.iter())
    }

    /// How many global loads read an element of type `ty`.
    fn global_loads_of(ir: &KernelIR, ty: &KirType) -> usize {
        ops(ir)
            .filter(|op| {
                matches!(op, KirOp::Load(d, _, AddressSpace::Global) if ir.var_types.get(d) == Some(ty))
            })
            .count()
    }

    /// How many multiplies read the entry param named `name`.
    fn multiplies_by_param(ir: &KernelIR, name: &str) -> usize {
        let param = ir.params.iter().find(|p| p.name == name).expect("a declared param").id;
        ops(ir)
            .filter(|op| matches!(op, KirOp::Mul(_, a, b) if *a == param || *b == param))
            .count()
    }

    fn u64_constants(ir: &KernelIR) -> Vec<u64> {
        ops(ir)
            .filter_map(|op| match op {
                KirOp::Const(_, KirConst { value: ConstValue::U64(v), .. }) => Some(*v),
                _ => None,
            })
            .collect()
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

    // ── cross-module stride consistency ────────────────────────────

    #[test]
    fn strides_match_the_base_kernel() {
        let base_ptx = emit_decode_attention_ptx(&matching_base_cfg());
        let (l0_ptx, _) = emit_layer(&all_fp16_cfg(), 0);
        let base = kv_strides(&matching_base_cfg());
        // The element token stride, in both headers and in the one
        // function the base kernel bakes it from.
        assert_eq!(comment_value(&base_ptx, "token_stride"), base.token_stride);
        assert_eq!(comment_value(&l0_ptx, "token_stride"), base.token_stride);
        // Both kernels are one builder (roadmap A2 step 9): the quant
        // kernel indexes each half in its own element type, so it bakes
        // the same element stride the base kernel does, and the same slot
        // base.
        let ir = build_layer(&all_fp16_cfg(), 0);
        assert!(u64_constants(&ir).contains(&base.token_stride));
        let per_slot = matching_base_cfg().per_slot_max_tokens;
        assert!(ops(&ir).any(|op| matches!(
            op,
            KirOp::Const(_, KirConst { value: ConstValue::U32(v), .. }) if *v == per_slot
        )));
    }

    // ── structural: per-precision load paths ──────────────────────

    #[test]
    fn fp16_layer_reads_f16_directly_and_never_multiplies_by_a_scale() {
        let (ptx, meta) = emit_layer(&mixed_cfg(), 0);
        let ir = build_layer(&mixed_cfg(), 0);
        assert!(ptx.contains(".visible .entry nsl_cfie_decode_attn_l0("));
        assert_eq!(global_loads_of(&ir, &KirType::F16), 2, "K and V");
        assert_eq!(global_loads_of(&ir, &KirType::I8), 0);
        assert!(ptx.contains("cvt.f32.f16"));
        assert!(!ptx.contains("ld.global.s8"));
        assert!(!ptx.contains("cvt.rn.f32.s8"));
        // Scale params are declared (uniform ABI) but no multiply reads them.
        assert!(ptx.contains(".param .f32 param_k_scale"));
        assert!(ptx.contains(".param .f32 param_v_scale"));
        assert_eq!(multiplies_by_param(&ir, "k_scale"), 0);
        assert_eq!(multiplies_by_param(&ir, "v_scale"), 0);
        assert!(!meta.k_scale_param_used && !meta.v_scale_param_used);
    }

    #[test]
    fn int8_layer_loads_s8_and_dequantizes_with_scale_params() {
        let (ptx, meta) = emit_layer(&mixed_cfg(), 2);
        let ir = build_layer(&mixed_cfg(), 2);
        assert!(ptx.contains(".visible .entry nsl_cfie_decode_attn_l2("));
        assert_eq!(global_loads_of(&ir, &KirType::I8), 2, "K and V");
        assert_eq!(global_loads_of(&ir, &KirType::F16), 0);
        assert_eq!(ptx.matches("ld.global.s8").count(), 2);
        assert_eq!(ptx.matches("cvt.rn.f32.s8").count(), 2);
        // Each half dequantized by its own scale, once.
        assert_eq!(multiplies_by_param(&ir, "k_scale"), 1);
        assert_eq!(multiplies_by_param(&ir, "v_scale"), 1);
        // No f16 conversion anywhere in a pure-int8 layer.
        assert!(!ptx.contains("cvt.f32.f16"));
        assert!(meta.k_scale_param_used && meta.v_scale_param_used);
    }

    #[test]
    fn mixed_layer_specializes_k_and_v_independently() {
        // Layer 1: FP16 K, INT8 V.
        let (_, meta) = emit_layer(&mixed_cfg(), 1);
        let ir = build_layer(&mixed_cfg(), 1);
        assert_eq!(global_loads_of(&ir, &KirType::F16), 1, "K path stays f16");
        assert_eq!(global_loads_of(&ir, &KirType::I8), 1, "V path is s8");
        assert_eq!(multiplies_by_param(&ir, "k_scale"), 0);
        assert_eq!(multiplies_by_param(&ir, "v_scale"), 1);
        assert!(!meta.k_scale_param_used && meta.v_scale_param_used);
    }

    #[test]
    fn per_layer_offsets_are_distinct_baked_immediates() {
        let cfg = mixed_cfg();
        let all = emit_all(&cfg);
        assert_eq!(all.len(), cfg.n_layers as usize);
        let mut interleaved: Vec<u64> = Vec::new();
        for (l, (ptx, meta)) in all.iter().enumerate() {
            assert_eq!(meta.kernel_name, kernel_name_for_layer(l as u32));
            let ir = build_layer(&cfg, l as u32);
            let consts = u64_constants(&ir);
            assert!(consts.contains(&meta.k_offset_bytes), "layer {l} bakes its K offset");
            assert!(consts.contains(&meta.v_offset_bytes), "layer {l} bakes its V offset");
            assert_eq!(comment_value(ptx, "k_offset_bytes"), meta.k_offset_bytes);
            assert_eq!(comment_value(ptx, "v_offset_bytes"), meta.v_offset_bytes);
            // The layer is baked: no layer index is a parameter.
            assert!(ir.params.iter().all(|p| p.name != "layer_idx"));
            assert!(!ptx.contains("layer_idx"));
            interleaved.push(meta.k_offset_bytes);
            interleaved.push(meta.v_offset_bytes);
        }
        // Offsets strictly increase in (k, v) interleaved order.
        assert!(interleaved.windows(2).all(|w| w[0] < w[1]));
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
        // The KIR floor, whatever GPU serves it (see the module docs).
        assert!(ptx.contains(".version 7.0\n.target sm_70\n.address_size 64"));
        assert_eq!(meta.block_dim, 128);
        assert!(meta.grid_dim_is_n_heads);
        // Same SMEM formula as the base kernel: q + scores + rescale + l.
        assert_eq!(meta.smem_bytes, 64 * 4 + 128 * 4 + 8);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 shared_mem[{}];", meta.smem_bytes)));
        // No trailing NUL: the serve path appends the one it needs.
        assert!(!ptx.ends_with('\0'));
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
    #[should_panic(expected = "not a multiple of its 2-byte element")]
    fn an_f16_half_after_an_odd_int8_half_is_refused() {
        // half = max_tokens(3) * n_kv_heads(1) * head_dim(1) = 3 elements:
        // the int8 K half is 3 bytes, so the f16 V half would start on
        // byte 3 — a misaligned 2-byte load, which the GPU faults on.
        let cfg = QuantDecodeAttentionConfig {
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            per_slot_max_tokens: 3,
            max_slots: 1,
            layer_precisions: vec![(KvPrecision::Int8, KvPrecision::Fp16)],
        };
        let _ = pool_layout(&cfg);
    }

    #[test]
    fn an_odd_int8_half_is_fine_when_nothing_f16_follows_it() {
        let cfg = QuantDecodeAttentionConfig {
            n_layers: 2,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            per_slot_max_tokens: 3,
            max_slots: 1,
            layer_precisions: vec![
                (KvPrecision::Fp16, KvPrecision::Fp16),
                (KvPrecision::Int8, KvPrecision::Int8),
            ],
        };
        assert_eq!(total_pool_bytes(&cfg), 3 * 2 * 2 + 3 * 2);
        assert_eq!(emit_all(&cfg).len(), 2);
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
                    nsl_log::nsl_log!(INFO, "skip", 
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
