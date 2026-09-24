//! CFIE Feature 1: direct-indexing decode-attention kernel, built as KIR.
//!
//! The paper's core claim: because the KV-cache layout
//! `[n_layers][2][max_tokens][n_kv_heads][head_dim]` is fixed at compile
//! time (see `cfie_kv_plan::DirectLayout`), the decode-attention kernel
//! addresses K/V by pure arithmetic over strides baked as immediates.
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
//!
//! ## KIR (roadmap A2 step 9)
//!
//! The kernel was hand-assembled PTX text until A2 step 9; it is now a
//! [`KernelIR`] that the verifier checks before `nsl_kir`'s printer lowers
//! it. The algorithm, and the order of every floating-point operation in
//! it, is the hand kernel's — `tests/cfie_decode_attn_kir_equivalence.rs`
//! runs the frozen hand emitter and this one side by side on a PTX
//! interpreter and requires the same output bits. What changed:
//!
//! * Addresses are element indices through `PtrOffset` rather than byte
//!   offsets, so the baked immediates are element strides (the `//`
//!   header lines always were); the pointer's element type does the
//!   scaling.
//! * Loop-carried values (the tile cursor, the accumulator, the running
//!   max and sum, each loop's index) are block parameters.
//! * Shared memory is an [`SmemLayout`] of four f32 regions — `q`,
//!   `scores`, `rescale`, `l` — at the offsets the hand kernel used.
//! * The module targets the KIR floor (`.version 7.0` / `.target sm_70`)
//!   instead of the serving GPU: the driver JIT-compiles it forward to any
//!   newer part. The hand header paired `.target sm_{N}` with a PTX ISA
//!   that cannot name every `N` the GPU table holds (sm_86/87/89 need ISA
//!   7.1/7.4/7.8 and were given 7.0; sm_120 needs 8.7 and was given 8.6),
//!   and nothing in the kernel needs more than sm_70.

use std::fmt::Write;

use crate::backend_ptx::lower_kir_to_ptx;
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp,
    KirTerminator, KirType, SmemLayout, SmemRegion, VarId,
};

/// Threads per CTA and softmax tile width (tokens processed per tile).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

pub const KERNEL_NAME: &str = "nsl_cfie_decode_attn";

pub fn kernel_name() -> &'static str {
    KERNEL_NAME
}

/// Compile-time layout + launch configuration for the decode kernel.
///
/// There is no `sm_version`: the module targets the KIR floor and the
/// driver JIT-compiles it for the device it is loaded on (see the module
/// docs).
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
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct DecodeAttentionMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
    pub grid_dim_is_n_heads: bool,
}

/// The strides the kernel bakes, in ELEMENTS of the contiguous layout
/// `[n_layers][2][max_tokens][n_kv_heads][head_dim]` — the numbers the
/// `//` header prints and every sibling kernel reading the same pool must
/// agree with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvStrides {
    /// One token's record: `n_kv_heads * head_dim`.
    pub token_stride: u64,
    /// Tokens in the global pool: `max_slots * per_slot_max_tokens`.
    pub max_tokens: u64,
    /// The K plane (and the V plane) of one layer: `max_tokens * token_stride`.
    pub kv_half_stride: u64,
    /// One layer: `2 * kv_half_stride`.
    pub layer_stride: u64,
    /// Query heads per KV head.
    pub gqa_group: u32,
}

/// The pool strides for `cfg`, in elements.
pub fn kv_strides(cfg: &DecodeAttentionConfig) -> KvStrides {
    let token_stride = cfg.n_kv_heads as u64 * cfg.head_dim as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    let kv_half_stride = max_tokens * token_stride;
    KvStrides {
        token_stride,
        max_tokens,
        kv_half_stride,
        layer_stride: 2 * kv_half_stride,
        gqa_group: cfg.n_heads / cfg.n_kv_heads,
    }
}

/// The configuration contract. Panics name the violated condition; every
/// caller either checks the same conditions first (`serve.rs`) or is a
/// test.
fn check(cfg: &DecodeAttentionConfig) {
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
        "v1 reads the KV pool as f16 only: kv_dtype_bytes must be 2"
    );
    assert!(
        cfg.per_slot_max_tokens >= 1 && cfg.max_slots >= 1,
        "per_slot_max_tokens and max_slots must be >= 1"
    );
    // The kernel's global token index is a u32 register (addressing is
    // 64-bit, but the token count itself must not wrap).
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    assert!(
        max_tokens <= u32::MAX as u64,
        "global token pool (max_slots * per_slot_max_tokens = {max_tokens}) must fit in u32"
    );
}

/// Index of each shared region in [`smem_layout`], in declaration order.
const R_Q: u32 = 0;
const R_SCORES: u32 = 1;
const R_RESCALE: u32 = 2;
const R_L: u32 = 3;

/// `[q: head_dim][scores: TILE][rescale: 1][l: 1]`, all f32. Every region
/// is 4-aligned and a multiple of 4 long, so the offsets are the hand
/// kernel's packed ones.
fn smem_layout(head_dim: u32) -> SmemLayout {
    let f32_region = |name: &str, elems: u32| SmemRegion {
        name: name.to_string(),
        bytes: elems * 4,
        align: 4,
        elem: KirType::F32,
    };
    SmemLayout {
        regions: vec![
            f32_region("q", head_dim),
            f32_region("scores", TILE),
            f32_region("rescale", 1),
            f32_region("l", 1),
        ],
        dynamic: false,
    }
}

pub(crate) fn ptr(elem: KirType, space: AddressSpace) -> KirType {
    KirType::Ptr(Box::new(elem), space)
}

pub(crate) fn konst(b: &mut KirBuilder, value: ConstValue) -> VarId {
    let ty = match value {
        ConstValue::U32(_) => KirType::U32,
        ConstValue::U64(_) => KirType::U64,
        ConstValue::F32(_) => KirType::F32,
        _ => unreachable!("the kernel's constants are u32, u64 and f32"),
    };
    let dst = b.new_typed_var(ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

/// `dst = op(x, y)` with `dst` of type `ty`.
pub(crate) fn op2(
    b: &mut KirBuilder,
    ty: KirType,
    op: fn(VarId, VarId, VarId) -> KirOp,
    x: VarId,
    y: VarId,
) -> VarId {
    let dst = b.new_typed_var(ty);
    b.emit(op(dst, x, y));
    dst
}

pub(crate) fn cmp(b: &mut KirBuilder, x: VarId, y: VarId, how: CmpOp) -> VarId {
    let dst = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(dst, x, y, how));
    dst
}

/// Zero-extend a u32 to u64.
pub(crate) fn widen(b: &mut KirBuilder, x: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(dst, x, KirType::U64));
    dst
}

/// `&base[index]`, where `base` points at `elem` in `space`.
pub(crate) fn at(b: &mut KirBuilder, elem: KirType, space: AddressSpace, base: VarId, index: VarId) -> VarId {
    let dst = b.new_typed_var(ptr(elem, space));
    b.emit(KirOp::PtrOffset(dst, base, index));
    dst
}

pub(crate) fn load(b: &mut KirBuilder, ty: KirType, addr: VarId, space: AddressSpace) -> VarId {
    let dst = b.new_typed_var(ty);
    b.emit(KirOp::Load(dst, addr, space));
    dst
}

/// How one half (K or V) of the pool is read: where its elements start
/// and how one becomes an f32.
#[derive(Clone, Copy)]
pub(crate) struct HalfReader {
    /// Points at element 0 of the half's addressing: the pool itself for
    /// the uniform layout (the plane offset is then an element index), the
    /// half's own first element for a baked one.
    base: VarId,
    /// Element index of the half inside `base`, if it is not 0.
    plane: Option<VarId>,
    /// `None`: f16, widened. `Some(scale)`: int8, converted and multiplied
    /// by `scale` — dequantized in registers.
    int8_scale: Option<VarId>,
}

impl HalfReader {
    /// The K and V halves of layer `layer_idx` in the uniform f16 pool at
    /// `kv_base`: K at `layer_idx * 2 * kv_half_stride` elements, V a
    /// half-stride later. Emitted into the current block.
    pub(crate) fn uniform_f16(
        b: &mut KirBuilder,
        kv_base: VarId,
        layer_idx: VarId,
        kv_half_stride: u64,
    ) -> (HalfReader, HalfReader) {
        let layer = widen(b, layer_idx);
        let layer_stride = konst(b, ConstValue::U64(2 * kv_half_stride));
        let k_plane = op2(b, KirType::U64, KirOp::Mul, layer, layer_stride);
        let kv_half = konst(b, ConstValue::U64(kv_half_stride));
        let v_plane = op2(b, KirType::U64, KirOp::Add, k_plane, kv_half);
        (
            HalfReader { base: kv_base, plane: Some(k_plane), int8_scale: None },
            HalfReader { base: kv_base, plane: Some(v_plane), int8_scale: None },
        )
    }

    fn elem(&self) -> KirType {
        if self.int8_scale.is_some() { KirType::I8 } else { KirType::F16 }
    }

    /// The element at index `index` (relative to `base`), as f32.
    fn load(&self, b: &mut KirBuilder, index: VarId) -> VarId {
        let elem = self.elem();
        let addr = at(b, elem.clone(), AddressSpace::Global, self.base, index);
        let raw = load(b, elem, addr, AddressSpace::Global);
        let wide = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Cast(wide, raw, KirType::F32));
        match self.int8_scale {
            None => wide,
            Some(scale) => op2(b, KirType::F32, KirOp::Mul, wide, scale),
        }
    }

    /// `row` shifted into this half's plane.
    pub(crate) fn in_plane(&self, b: &mut KirBuilder, row: VarId) -> VarId {
        match self.plane {
            Some(plane) => op2(b, KirType::U64, KirOp::Add, plane, row),
            None => row,
        }
    }
}

/// The pool the shared flash-decode builder reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PoolLayout {
    /// `[n_layers][2][max_tokens][n_kv_heads][head_dim]`, f16 throughout;
    /// the layer is the runtime `layer_idx` param. This module's kernel.
    UniformF16,
    /// One layer, its K and V halves at baked byte offsets from `kv_base`,
    /// each f16 or int8; an int8 half is dequantized by the runtime
    /// `k_scale` / `v_scale` param. `cfie_kv_quant_ptx`'s per-layer
    /// kernels.
    Baked { k: BakedHalf, v: BakedHalf },
}

/// One half of a [`PoolLayout::Baked`] layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BakedHalf {
    /// Byte offset of the half's first element from `kv_base`. A multiple
    /// of the element size, which the caller checks.
    pub offset_bytes: u64,
    pub int8: bool,
}

/// What the shared flash-decode builder needs: the entry name, the
/// geometry, and the pool it reads.
#[derive(Debug, Clone)]
pub(crate) struct FlashDecode<'a> {
    pub name: &'a str,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub per_slot_max_tokens: u32,
    pub max_slots: u32,
    pub pool: PoolLayout,
}

/// Build the direct-indexing decode-attention kernel as KIR.
///
/// The CFG, with each block's parameters (every other value is defined
/// once and reaches its uses by dominance):
///
/// ```text
/// entry                    strides, kv_head, region pointers; tid < head_dim ?
/// q_load                   q_smem[tid] = q[head*head_dim + tid]
/// q_done                   bar; -> tile_head(0, 0.0, -inf, 0.0)
/// tile_head(tile, acc, m, l)          tile >= seq_len ? loop_end : tile_body
/// tile_body                tcnt = min(seq_len - tile, TILE); tok < seq_len ?
///   score                  K row of token tok      -> dot_head(0, 0.0)
///   dot_head(d, dot)       d >= head_dim ? dot_done : dot_body
///   dot_body               dot = fma(k[d], q_smem[d], dot)  -> dot_head
///   dot_done               scores[tid] = dot * 1/sqrt(head_dim)
/// score_done               bar; tid != 0 ? softmax_done(m, l) : max_head(0, m)
///   max_head(j, tm)        j >= tcnt ? max_done : max_body
///   max_body               tm = max(tm, scores[j])      -> max_head
///   max_done               rs = exp(m - tm)             -> p_head(0, l * rs)
///   p_head(j, lsum)        j >= tcnt ? p_done : p_body
///   p_body                 scores[j] = exp(scores[j] - tm); lsum += it
///   p_done                 rescale = rs                 -> softmax_done(tm, lsum)
/// softmax_done(m', l')     bar; tid >= head_dim ? acc_tail(acc) : acc_start
///   acc_start              acc *= rescale; V row        -> acc_head(0, acc)
///   acc_head(j, a)         j >= tcnt ? acc_tail(a) : acc_body
///   acc_body               a = fma(scores[j], v[j][tid], a)  -> acc_head
/// acc_tail(acc')           bar; -> tile_head(tile + TILE, acc', m', l')
/// loop_end                 tid != 0 ? l_pub : l_store
///   l_store                l_smem = l
/// l_pub                    bar; tid >= head_dim ? exit : out_load
///   out_load               lf = l_smem; lf > 0 ? out_div : store_out(0.0)
///   out_div                -> store_out(acc / lf)
///   store_out(o)           out[head*head_dim + tid] = o
/// exit                     ret
/// ```
///
/// Only thread 0's `m`/`l` are meaningful — the other threads carry their
/// initial values through `softmax_done`, as the hand kernel's untouched
/// registers did — and only thread 0's `l` is published.
pub fn build(cfg: &DecodeAttentionConfig) -> KernelIR {
    check(cfg);
    build_flash_decode(&FlashDecode {
        name: KERNEL_NAME,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        per_slot_max_tokens: cfg.per_slot_max_tokens,
        max_slots: cfg.max_slots,
        pool: PoolLayout::UniformF16,
    })
}

/// The flash-decode kernel [`build`] documents, over either pool layout.
///
/// The two layouts differ only in the entry's parameters and in where a
/// K or V element is read from; the CFG, the shared regions and every
/// floating-point operation are the same:
///
/// * [`PoolLayout::UniformF16`] takes `(q_ptr, kv_base, out_ptr,
///   layer_idx, slot_idx, seq_len)` and indexes the pool in f16 elements,
///   the layer's K plane at `layer_idx * layer_stride` and V a half-stride
///   later.
/// * [`PoolLayout::Baked`] takes `(q_ptr, kv_base, out_ptr, slot_idx,
///   seq_len, k_scale, v_scale)`. `kv_base` is a byte pointer; each half
///   starts at its baked byte offset and is indexed in its own element
///   type. An int8 element is converted and multiplied by its half's
///   scale param before it enters the dot product or the accumulator. An
///   f16 half never reads its scale; the param is declared anyway, so
///   every layer's kernel shares one launch ABI.
///
/// The caller has checked the geometry (and, for a baked layout, the
/// offsets' alignment).
pub(crate) fn build_flash_decode(spec: &FlashDecode<'_>) -> KernelIR {
    use KirType::U32;

    let (mut b, e) = begin_flash_decode(spec);
    let row = op2(&mut b, U32, KirOp::Mul, e.head, e.ctx.head_dim);
    load_q_row(&mut b, &e.ctx, e.q_ptr, row);
    let (acc, _m, l) = prefix_pass(&mut b, &e.ctx, e.seq_len);
    publish_output(&mut b, &e.ctx, acc, l, (e.out_ptr, AddressSpace::Global), row);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// The entry of a flash-decode kernel, and the handles its sections need.
pub(crate) struct FlashEntry {
    pub ctx: FlashCtx,
    pub q_ptr: VarId,
    pub out_ptr: VarId,
    pub seq_len: VarId,
    /// This CTA's Q head.
    pub head: VarId,
}

/// Values the entry block defines once and every section reads, reaching
/// each use by dominance. A kernel with its own entry (the persistent
/// decode block) builds one per head, from values its head loop defines.
pub(crate) struct FlashCtx {
    pub tid: VarId,
    /// u32 `0` and `1`.
    pub zero: VarId,
    pub one: VarId,
    /// u32 `head_dim`.
    pub head_dim: VarId,
    /// u64 elements per token record.
    pub token_stride: VarId,
    /// u32: the slot's first global token.
    pub slot_base: VarId,
    /// u64: `kv_head`'s row inside one token record.
    pub head_off: VarId,
    /// f32 `1/sqrt(head_dim)`.
    pub inv_sqrt_hd: VarId,
    pub k_half: HalfReader,
    pub v_half: HalfReader,
    /// f32 shared pointers: the Q row the scores read, the score tile, the
    /// published rescale factor and the published `l`.
    pub q_smem: VarId,
    pub scores: VarId,
    pub rescale: VarId,
    pub l_smem: VarId,
}

/// One tile of the flash-decode loop, relative to the slot's first token.
pub(crate) struct TileSpan {
    /// u32: the tile's first token (pass 3 walks V from here).
    pub first: VarId,
    /// u32: the token this thread scores in pass 1.
    pub tok: VarId,
    /// bool: whether this thread scores a token at all.
    pub scores: VarId,
    /// u32: how many tokens the tile holds.
    pub tcnt: VarId,
}

/// Rewrites a thread's scaled pass-1 score before it is stored (the
/// verify kernel's tree mask). Runs in the block that stores the score.
pub(crate) type ScoreHook<'a> = &'a dyn Fn(&mut KirBuilder, VarId) -> VarId;

/// Start a flash-decode kernel: the params in FFI order, the shared
/// layout, and the entry block's strides, `kv_head`, and region pointers.
/// Returns with the entry block current and unterminated.
pub(crate) fn begin_flash_decode(spec: &FlashDecode<'_>) -> (KirBuilder, FlashEntry) {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32};

    let token_stride_elems = spec.n_kv_heads as u64 * spec.head_dim as u64;
    let hd = spec.head_dim;
    let mut b = KirBuilder::new(spec.name);

    // The direct params, in FFI order (the launcher marshals them
    // positionally).
    let q_ptr = b.add_param("q_ptr", ptr(F32, Global), Global);
    let kv_elem = match spec.pool {
        PoolLayout::UniformF16 => KirType::F16,
        // A byte pointer: each half's offset is in bytes.
        PoolLayout::Baked { .. } => KirType::I8,
    };
    let kv_base = b.add_param("kv_base", ptr(kv_elem, Global), Global);
    let out_ptr = b.add_param("out_ptr", ptr(F32, Global), Global);
    let layer_idx = match spec.pool {
        PoolLayout::UniformF16 => Some(b.add_param("layer_idx", U32, Global)),
        PoolLayout::Baked { .. } => None,
    };
    let slot_idx = b.add_param("slot_idx", U32, Global);
    let seq_len = b.add_param("seq_len", U32, Global);
    let scales = match spec.pool {
        PoolLayout::UniformF16 => None,
        PoolLayout::Baked { .. } => Some((
            b.add_param("k_scale", F32, Global),
            b.add_param("v_scale", F32, Global),
        )),
    };

    b.set_smem_layout(smem_layout(hd));
    b.set_workgroup_size([BLOCK_DIM, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let head = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(head, 0));
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let head_dim = konst(&mut b, ConstValue::U32(hd));
    let token_stride = konst(&mut b, ConstValue::U64(token_stride_elems));

    // GQA: kv_head = q_head / group (baked divisor).
    let group = konst(&mut b, ConstValue::U32(spec.n_heads / spec.n_kv_heads));
    let kv_head = op2(&mut b, U32, KirOp::Div, head, group);

    let (k_half, v_half) = match (spec.pool, layer_idx, scales) {
        (PoolLayout::UniformF16, Some(layer_idx), None) => {
            // K plane of this layer, and V = K + kv_half_stride (elements).
            let kv_half_stride = spec.max_slots as u64 * spec.per_slot_max_tokens as u64 * token_stride_elems;
            HalfReader::uniform_f16(&mut b, kv_base, layer_idx, kv_half_stride)
        }
        (PoolLayout::Baked { k, v }, None, Some((k_scale, v_scale))) => {
            // Each half's first element: kv_base + its baked byte offset,
            // then addressed in the half's own element type.
            let half = |b: &mut KirBuilder, h: BakedHalf, scale: VarId| {
                let offset = konst(b, ConstValue::U64(h.offset_bytes));
                let start = at(b, KirType::I8, Global, kv_base, offset);
                let elem = if h.int8 { KirType::I8 } else { KirType::F16 };
                let base_ty = ptr(elem, Global);
                let base = b.new_typed_var(base_ty.clone());
                b.emit(KirOp::Cast(base, start, base_ty));
                HalfReader { base, plane: None, int8_scale: h.int8.then_some(scale) }
            };
            (half(&mut b, k, k_scale), half(&mut b, v, v_scale))
        }
        _ => unreachable!("the params follow the pool layout"),
    };

    // The slot's first global token.
    let per_slot = konst(&mut b, ConstValue::U32(spec.per_slot_max_tokens));
    let slot_base = op2(&mut b, U32, KirOp::Mul, slot_idx, per_slot);

    // kv_head's row inside one token record.
    let head_off32 = op2(&mut b, U32, KirOp::Mul, kv_head, head_dim);
    let head_off = widen(&mut b, head_off32);
    let inv_sqrt_hd = konst(&mut b, ConstValue::F32(1.0f32 / (hd as f32).sqrt()));

    let region = |b: &mut KirBuilder, r: u32| {
        let dst = b.new_typed_var(ptr(F32, Shared));
        b.emit(KirOp::SharedRegion { dst, region: r });
        dst
    };
    let q_smem = region(&mut b, R_Q);
    let scores = region(&mut b, R_SCORES);
    let rescale = region(&mut b, R_RESCALE);
    let l_smem = region(&mut b, R_L);

    let ctx = FlashCtx {
        tid,
        zero,
        one,
        head_dim,
        token_stride,
        slot_base,
        head_off,
        inv_sqrt_hd,
        k_half,
        v_half,
        q_smem,
        scores,
        rescale,
        l_smem,
    };
    (b, FlashEntry { ctx, q_ptr, out_ptr, seq_len, head })
}

/// `q_smem[tid] = q[row + tid]` for `tid < head_dim`, then a barrier.
/// `row` is the Q row's first element. Terminates the current block and
/// returns with the post-barrier block current.
pub(crate) fn load_q_row(b: &mut KirBuilder, c: &FlashCtx, q_ptr: VarId, row: VarId) {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32};

    let q_load = b.new_block();
    let q_done = b.new_block();
    let loads_q = cmp(b, c.tid, c.head_dim, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(loads_q, KirEdge::to(q_load), KirEdge::to(q_done)));

    b.set_block(q_load);
    let q_index = op2(b, U32, KirOp::Add, row, c.tid);
    let q_addr = at(b, F32, Global, q_ptr, q_index);
    let q_val = load(b, F32, q_addr, Global);
    let q_slot = at(b, F32, Shared, c.q_smem, c.tid);
    b.emit(KirOp::Store(q_slot, q_val, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(q_done)));

    b.set_block(q_done);
    b.emit(KirOp::Barrier);
}

/// The tile loop over the slot's first `seq_len` tokens, from a fresh
/// state (`acc = 0`, `m = -inf`, `l = 0`). Terminates the current block
/// and returns with the loop's exit block current; the returned
/// `(acc, m, l)` are the loop head's parameters, which dominate it.
pub(crate) fn prefix_pass(b: &mut KirBuilder, c: &FlashCtx, seq_len: VarId) -> (VarId, VarId, VarId) {
    use KirType::{F32, U32};

    let tile_head = b.new_block();
    let tile_body = b.new_block();
    let loop_end = b.new_block();
    let tile = b.add_block_param(tile_head, U32);
    let acc = b.add_block_param(tile_head, F32);
    let m = b.add_block_param(tile_head, F32);
    let l = b.add_block_param(tile_head, F32);

    let f_zero = konst(b, ConstValue::F32(0.0));
    let f_neg_inf = konst(b, ConstValue::F32(f32::NEG_INFINITY));
    b.terminate(KirTerminator::Branch(KirEdge::with(
        tile_head,
        vec![c.zero, f_zero, f_neg_inf, f_zero],
    )));

    b.set_block(tile_head);
    let tiles_done = cmp(b, tile, seq_len, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(tiles_done, KirEdge::to(loop_end), KirEdge::to(tile_body)));

    b.set_block(tile_body);
    let remaining = op2(b, U32, KirOp::Sub, seq_len, tile);
    let tile_width = konst(b, ConstValue::U32(TILE));
    // Tail-tile guard: the last tile covers seq_len % TILE tokens.
    let tcnt = op2(b, U32, KirOp::Min, remaining, tile_width);
    let tok = op2(b, U32, KirOp::Add, tile, c.tid);
    let scores = cmp(b, tok, seq_len, CmpOp::Lt);
    let span = TileSpan { first: tile, tok, scores, tcnt };
    let (acc_next, m_next, l_next) = flash_tile(b, c, &span, (acc, m, l), None);

    let tile_next = op2(b, U32, KirOp::Add, tile, tile_width);
    b.terminate(KirTerminator::Branch(KirEdge::with(
        tile_head,
        vec![tile_next, acc_next, m_next, l_next],
    )));

    b.set_block(loop_end);
    (acc, m, l)
}

/// One flash-decode tile: pass 1 scores, pass 2 folds them into the
/// running softmax on thread 0, pass 3 rescales the accumulator and adds
/// P*V. `state` is `(acc, m, l)` coming in. Terminates the current block
/// and returns with the tile's closing block current, after its barrier
/// and unterminated, and the `(acc, m, l)` going out.
///
/// Only thread 0's `m`/`l` are meaningful: the other threads carry
/// theirs through unchanged, as the hand kernel's untouched registers did.
pub(crate) fn flash_tile(
    b: &mut KirBuilder,
    c: &FlashCtx,
    span: &TileSpan,
    state: (VarId, VarId, VarId),
    score_hook: Option<ScoreHook<'_>>,
) -> (VarId, VarId, VarId) {
    use AddressSpace::Shared;
    use KirType::{F32, U32, U64};

    let (acc, m, l) = state;
    let tcnt = span.tcnt;
    let score = b.new_block();
    let dot_head = b.new_block();
    let dot_body = b.new_block();
    let dot_done = b.new_block();
    let score_done = b.new_block();
    let max_head = b.new_block();
    let max_body = b.new_block();
    let max_done = b.new_block();
    let p_head = b.new_block();
    let p_body = b.new_block();
    let p_done = b.new_block();
    let softmax_done = b.new_block();
    let acc_start = b.new_block();
    let acc_head = b.new_block();
    let acc_body = b.new_block();
    let acc_tail = b.new_block();

    let d = b.add_block_param(dot_head, U32);
    let dot = b.add_block_param(dot_head, F32);
    let max_j = b.add_block_param(max_head, U32);
    let tm = b.add_block_param(max_head, F32);
    let p_j = b.add_block_param(p_head, U32);
    let lsum = b.add_block_param(p_head, F32);
    let m_next = b.add_block_param(softmax_done, F32);
    let l_next = b.add_block_param(softmax_done, F32);
    let acc_j = b.add_block_param(acc_head, U32);
    let a = b.add_block_param(acc_head, F32);
    let acc_next = b.add_block_param(acc_tail, F32);

    b.terminate(KirTerminator::CondBranch(
        span.scores,
        KirEdge::to(score),
        KirEdge::to(score_done),
    ));

    // ── pass 1: thread t scores token `tok` ──────────────────────────
    b.set_block(score);
    let g = op2(b, U32, KirOp::Add, c.slot_base, span.tok);
    let g_wide = widen(b, g);
    let k_tok = op2(b, U64, KirOp::Mul, g_wide, c.token_stride);
    let k_tok_plane = c.k_half.in_plane(b, k_tok);
    let k_row = op2(b, U64, KirOp::Add, k_tok_plane, c.head_off);
    let f_zero_dot = konst(b, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(dot_head, vec![c.zero, f_zero_dot])));

    b.set_block(dot_head);
    let dot_complete = cmp(b, d, c.head_dim, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(dot_complete, KirEdge::to(dot_done), KirEdge::to(dot_body)));

    b.set_block(dot_body);
    let d_wide = widen(b, d);
    let k_index = op2(b, U64, KirOp::Add, k_row, d_wide);
    let k_val = c.k_half.load(b, k_index);
    let q_elem = at(b, F32, Shared, c.q_smem, d);
    let q_d = load(b, F32, q_elem, Shared);
    let dot_acc = b.new_typed_var(F32);
    b.emit(KirOp::Fma(dot_acc, k_val, q_d, dot));
    let d_next = op2(b, U32, KirOp::Add, d, c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(dot_head, vec![d_next, dot_acc])));

    b.set_block(dot_done);
    let scaled = op2(b, F32, KirOp::Mul, dot, c.inv_sqrt_hd);
    let scaled = match score_hook {
        Some(hook) => hook(b, scaled),
        None => scaled,
    };
    let score_slot = at(b, F32, Shared, c.scores, c.tid);
    b.emit(KirOp::Store(score_slot, scaled, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(score_done)));

    // ── pass 2: online softmax, thread 0 serial over the tile ────────
    b.set_block(score_done);
    b.emit(KirOp::Barrier);
    let not_thread0 = cmp(b, c.tid, c.zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(
        not_thread0,
        KirEdge::with(softmax_done, vec![m, l]),
        KirEdge::with(max_head, vec![c.zero, m]),
    ));

    b.set_block(max_head);
    let max_complete = cmp(b, max_j, tcnt, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(max_complete, KirEdge::to(max_done), KirEdge::to(max_body)));

    b.set_block(max_body);
    let max_elem = at(b, F32, Shared, c.scores, max_j);
    let max_score = load(b, F32, max_elem, Shared);
    let tm_next = op2(b, F32, KirOp::Max, tm, max_score);
    let max_j_next = op2(b, U32, KirOp::Add, max_j, c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(max_head, vec![max_j_next, tm_next])));

    // rescale = exp(m_old - m_new); exp(-inf) = 0 on the first tile.
    b.set_block(max_done);
    let m_delta = op2(b, F32, KirOp::Sub, m, tm);
    let rs = b.new_typed_var(F32);
    b.emit(KirOp::Exp(rs, m_delta));
    let l_rescaled = op2(b, F32, KirOp::Mul, l, rs);
    b.terminate(KirTerminator::Branch(KirEdge::with(p_head, vec![c.zero, l_rescaled])));

    b.set_block(p_head);
    let p_complete = cmp(b, p_j, tcnt, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(p_complete, KirEdge::to(p_done), KirEdge::to(p_body)));

    b.set_block(p_body);
    let p_elem = at(b, F32, Shared, c.scores, p_j);
    let p_score = load(b, F32, p_elem, Shared);
    let p_shift = op2(b, F32, KirOp::Sub, p_score, tm);
    let p = b.new_typed_var(F32);
    b.emit(KirOp::Exp(p, p_shift));
    b.emit(KirOp::Store(p_elem, p, Shared));
    let lsum_next = op2(b, F32, KirOp::Add, lsum, p);
    let p_j_next = op2(b, U32, KirOp::Add, p_j, c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(p_head, vec![p_j_next, lsum_next])));

    b.set_block(p_done);
    b.emit(KirOp::Store(c.rescale, rs, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::with(softmax_done, vec![tm, lsum])));

    // ── pass 3: rescale the accumulator, add P*V; thread d owns out[d] ─
    b.set_block(softmax_done);
    b.emit(KirOp::Barrier);
    let no_output = cmp(b, c.tid, c.head_dim, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        no_output,
        KirEdge::with(acc_tail, vec![acc]),
        KirEdge::to(acc_start),
    ));

    b.set_block(acc_start);
    let rs_shared = load(b, F32, c.rescale, Shared);
    let acc_rescaled = op2(b, F32, KirOp::Mul, acc, rs_shared);
    let g0 = op2(b, U32, KirOp::Add, c.slot_base, span.first);
    let g0_wide = widen(b, g0);
    let v_tok = op2(b, U64, KirOp::Mul, g0_wide, c.token_stride);
    let v_tok_plane = c.v_half.in_plane(b, v_tok);
    let v_head_row = op2(b, U64, KirOp::Add, v_tok_plane, c.head_off);
    let tid_wide = widen(b, c.tid);
    let v_col = op2(b, U64, KirOp::Add, v_head_row, tid_wide);
    b.terminate(KirTerminator::Branch(KirEdge::with(acc_head, vec![c.zero, acc_rescaled])));

    b.set_block(acc_head);
    let acc_complete = cmp(b, acc_j, tcnt, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        acc_complete,
        KirEdge::with(acc_tail, vec![a]),
        KirEdge::to(acc_body),
    ));

    b.set_block(acc_body);
    let w_elem = at(b, F32, Shared, c.scores, acc_j);
    let weight = load(b, F32, w_elem, Shared);
    let j_wide = widen(b, acc_j);
    let v_step = op2(b, U64, KirOp::Mul, j_wide, c.token_stride);
    let v_index = op2(b, U64, KirOp::Add, v_col, v_step);
    let v_val = c.v_half.load(b, v_index);
    let a_next = b.new_typed_var(F32);
    b.emit(KirOp::Fma(a_next, weight, v_val, a));
    let acc_j_next = op2(b, U32, KirOp::Add, acc_j, c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(acc_head, vec![acc_j_next, a_next])));

    // The scores region is rewritten by whatever runs next; sync first.
    b.set_block(acc_tail);
    b.emit(KirOp::Barrier);
    (acc_next, m_next, l_next)
}

/// Thread 0 publishes `l`; after a barrier, thread `d < head_dim` stores
/// `out[row + d] = acc / l` (0 when `l` is not positive, so an empty
/// prefix writes 0 rather than NaN). `out_ptr` points at f32 in
/// `out_space`: global for the attention kernels, shared for the
/// persistent decode block's attention-output rows. Terminates the current
/// block and returns with a fresh block, where every path meets, current.
pub(crate) fn publish_output(
    b: &mut KirBuilder,
    c: &FlashCtx,
    acc: VarId,
    l: VarId,
    (out_ptr, out_space): (VarId, AddressSpace),
    row: VarId,
) {
    use AddressSpace::Shared;
    use KirType::{F32, U32};

    let l_store = b.new_block();
    let l_pub = b.new_block();
    let out_load = b.new_block();
    let out_div = b.new_block();
    let store_out = b.new_block();
    let done = b.new_block();
    let o = b.add_block_param(store_out, F32);

    let skips_publish = cmp(b, c.tid, c.zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(skips_publish, KirEdge::to(l_pub), KirEdge::to(l_store)));

    b.set_block(l_store);
    b.emit(KirOp::Store(c.l_smem, l, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(l_pub)));

    b.set_block(l_pub);
    b.emit(KirOp::Barrier);
    let writes_nothing = cmp(b, c.tid, c.head_dim, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(writes_nothing, KirEdge::to(done), KirEdge::to(out_load)));

    b.set_block(out_load);
    let l_final = load(b, F32, c.l_smem, Shared);
    let f_zero_out = konst(b, ConstValue::F32(0.0));
    let positive = cmp(b, l_final, f_zero_out, CmpOp::Gt);
    b.terminate(KirTerminator::CondBranch(
        positive,
        KirEdge::to(out_div),
        KirEdge::with(store_out, vec![f_zero_out]),
    ));

    b.set_block(out_div);
    let normalized = op2(b, F32, KirOp::Div, acc, l_final);
    b.terminate(KirTerminator::Branch(KirEdge::with(store_out, vec![normalized])));

    b.set_block(store_out);
    let out_index = op2(b, U32, KirOp::Add, row, c.tid);
    let out_addr = at(b, F32, out_space, out_ptr, out_index);
    b.emit(KirOp::Store(out_addr, o, out_space));
    b.terminate(KirTerminator::Branch(KirEdge::to(done)));

    b.set_block(done);
}

/// The `//` header: what the kernel bakes, in elements. Sibling kernels
/// reading the same pool (`cfie_kv_quant_ptx`, `cfie_persistent_ptx`,
/// `cfie_speculative_ptx`) compare their `//   <name> = <value>` lines
/// with these byte for byte.
fn header_comment(cfg: &DecodeAttentionConfig, s: &KvStrides) -> String {
    let mut w = String::new();
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
        cfg.n_layers, s.max_tokens, cfg.n_kv_heads, cfg.head_dim
    )
    .unwrap();
    writeln!(w, "// Baked layout constants (elements):").unwrap();
    writeln!(w, "//   token_stride        = {}", s.token_stride).unwrap();
    writeln!(w, "//   kv_half_stride      = {}", s.kv_half_stride).unwrap();
    writeln!(w, "//   layer_stride        = {}", s.layer_stride).unwrap();
    writeln!(w, "//   per_slot_max_tokens = {}", cfg.per_slot_max_tokens).unwrap();
    writeln!(w, "//   max_tokens          = {}", s.max_tokens).unwrap();
    writeln!(w, "//   gqa_group_size      = {}", s.gqa_group).unwrap();
    writeln!(
        w,
        "// No block table: every KV address is arithmetic over these immediates."
    )
    .unwrap();
    writeln!(w, "//").unwrap();
    w
}

/// Emit the direct-indexing decode-attention kernel: build, verify, lower,
/// and prefix the `//` header.
///
/// The returned text carries no NUL: the serve path appends the one
/// `cuModuleLoadData` needs when it embeds the module.
///
/// # Panics
///
/// On a configuration outside the contract (see the asserts in `check`),
/// and if the built kernel fails KIR verification — a bug in this module,
/// not a condition a caller can provoke.
pub fn emit(cfg: &DecodeAttentionConfig) -> (String, DecodeAttentionMeta) {
    let ir = build(cfg);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{KERNEL_NAME} failed KIR verification: {errors:?}");
    }
    let smem_bytes = ir
        .smem_layout
        .total_bytes()
        .expect("a verified layout has a size");
    let module = lower_kir_to_ptx(&ir);
    let module = module.strip_suffix(&[0]).unwrap_or(&module);
    let module = std::str::from_utf8(module).expect("the KIR printer emits ASCII");

    let mut p = header_comment(cfg, &kv_strides(cfg));
    p.push_str(module);

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
        }
    }

    /// Geometries the structural checks sweep: the paper config, GQA and
    /// MHA, a head_dim that is not a power of two, the smallest pool.
    fn sweep() -> Vec<DecodeAttentionConfig> {
        let mut out = vec![paper_cfg()];
        for (n_heads, n_kv_heads, head_dim, per_slot, slots) in
            [(4, 4, 64, 256, 2), (6, 2, 40, 300, 3), (1, 1, 1, 1, 1), (32, 8, 128, 4096, 16)]
        {
            out.push(DecodeAttentionConfig {
                n_layers: 3,
                n_heads,
                n_kv_heads,
                head_dim,
                per_slot_max_tokens: per_slot,
                max_slots: slots,
                kv_dtype_bytes: 2,
            });
        }
        out
    }

    /// Lines of `ptx` that set a register to the immediate `value`.
    fn movs_of(ptx: &str, ty: &str, value: impl std::fmt::Display) -> usize {
        let suffix = format!(", {value};");
        ptx.lines()
            .filter(|l| l.trim_start().starts_with(&format!("mov.{ty} ")) && l.ends_with(&suffix))
            .count()
    }

    #[test]
    fn every_geometry_verifies() {
        for cfg in sweep() {
            let ir = build(&cfg);
            if let Err(errors) = crate::kir_verify::verify(&ir) {
                panic!("{cfg:?} failed verification: {errors:?}");
            }
        }
    }

    #[test]
    fn param_list_is_exactly_the_six_direct_params() {
        let ir = build(&paper_cfg());
        let f32_global = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        let params: Vec<(&str, &KirType)> =
            ir.params.iter().map(|p| (p.name.as_str(), &p.ty)).collect();
        assert_eq!(
            params,
            vec![
                ("q_ptr", &f32_global),
                ("kv_base", &KirType::Ptr(Box::new(KirType::F16), AddressSpace::Global)),
                ("out_ptr", &f32_global),
                ("layer_idx", &KirType::U32),
                ("slot_idx", &KirType::U32),
                ("seq_len", &KirType::U32),
            ]
        );
        // And the printed entry carries them in that order, at the widths
        // the launcher marshals.
        let ptx = emit_decode_attention_ptx(&paper_cfg());
        assert!(ptx.contains(
            ".visible .entry nsl_cfie_decode_attn(.param .u64 param_q_ptr, .param .u64 param_kv_base, \
             .param .u64 param_out_ptr, .param .u32 param_layer_idx, .param .u32 param_slot_idx, \
             .param .u32 param_seq_len)"
        ), "{ptx}");
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
    fn baked_strides_are_immediates() {
        let cfg = paper_cfg();
        let s = kv_strides(&cfg);
        // elements: token_stride = 4*128 = 512; max_tokens = 64*2048 = 131072;
        // kv_half = 131072*512 = 67108864; layer = 2*kv_half = 134217728.
        assert_eq!(
            s,
            KvStrides {
                token_stride: 512,
                max_tokens: 131_072,
                kv_half_stride: 67_108_864,
                layer_stride: 134_217_728,
                gqa_group: 2,
            }
        );
        let ptx = emit_decode_attention_ptx(&cfg);
        assert!(ptx.contains("//   token_stride        = 512\n"));
        assert!(ptx.contains("//   kv_half_stride      = 67108864\n"));
        assert!(ptx.contains("//   layer_stride        = 134217728\n"));
        // The same numbers drive the address arithmetic, as element
        // strides the f16 pointer scales by 2 — never a runtime load.
        assert_eq!(movs_of(&ptx, "u64", s.token_stride), 1, "{ptx}");
        assert_eq!(movs_of(&ptx, "u64", s.kv_half_stride), 1, "{ptx}");
        assert_eq!(movs_of(&ptx, "u64", s.layer_stride), 1, "{ptx}");
        assert_eq!(movs_of(&ptx, "u32", cfg.per_slot_max_tokens), 1, "{ptx}");
        assert_eq!(movs_of(&ptx, "u32", s.gqa_group), 1, "{ptx}");
        assert!(ptx.contains("mul.lo.u64 "), "f16 element indices scale to bytes");
    }

    #[test]
    fn no_mad_lo_no_nul_and_ascii_only() {
        for cfg in sweep() {
            let ptx = emit_decode_attention_ptx(&cfg);
            assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
            assert!(
                ptx.bytes().all(|b| b != 0 && b < 128),
                "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX) and \
                 NUL-free (the serve path appends the terminator)"
            );
        }
    }

    #[test]
    fn tail_tile_guard_clamps_to_seq_len() {
        // `tcnt = min(seq_len - tile, TILE)` and `tok < seq_len`: the last
        // tile covers seq_len % TILE tokens. The differential test runs
        // the ragged tile; this pins the two comparisons' operands.
        let ir = build(&paper_cfg());
        let seq_len = ir.params[5].id;
        let ops: Vec<&KirOp> = ir.blocks.iter().flat_map(|b| b.ops.iter()).collect();
        let consts: std::collections::HashMap<VarId, u32> = ops
            .iter()
            .filter_map(|op| match op {
                KirOp::Const(d, KirConst { value: ConstValue::U32(v), .. }) => Some((*d, *v)),
                _ => None,
            })
            .collect();
        assert!(ops.iter().any(|op| matches!(op,
            KirOp::Min(_, _, t) if consts.get(t) == Some(&TILE))));
        assert!(ops.iter().any(|op| matches!(op,
            KirOp::Cmp(_, _, n, CmpOp::Lt) if *n == seq_len)));
        assert!(ops.iter().any(|op| matches!(op,
            KirOp::Cmp(_, _, n, CmpOp::Ge) if *n == seq_len)));
    }

    #[test]
    fn header_is_the_kir_floor_whatever_the_device() {
        // The module no longer names the serving GPU: the driver
        // JIT-compiles sm_70 PTX forward (see the module docs).
        let ptx = emit_decode_attention_ptx(&paper_cfg());
        assert!(ptx.starts_with("//\n// nsl_cfie_decode_attn - "));
        assert!(ptx.contains("//\n.version 7.0\n.target sm_70\n.address_size 64\n"), "{ptx}");
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
        assert!(ptx.contains(&format!(".shared .align 4 .b8 shared_mem[{}];", meta.smem_bytes)));
        // The regions sit where the hand kernel's offsets put them.
        let layout = build(&cfg).smem_layout;
        let offsets: Vec<u32> = (0..4).map(|i| layout.offset_of(i).unwrap()).collect();
        assert_eq!(offsets, vec![0, 64 * 4, 64 * 4 + 128 * 4, 64 * 4 + 128 * 4 + 4]);
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

    #[test]
    #[should_panic(expected = "must fit in u32")]
    fn token_pool_over_u32_panics() {
        let mut cfg = paper_cfg();
        cfg.max_slots = 1 << 16;
        cfg.per_slot_max_tokens = 1 << 16;
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
                nsl_log::nsl_log!(INFO, "skip", "[skip] cfie decode-attn ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie decode-attn PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }
}
