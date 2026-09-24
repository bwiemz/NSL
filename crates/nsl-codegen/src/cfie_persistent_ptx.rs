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
//! immediates) — the two emitters share stride derivation and a
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
//!
//! ## KIR (roadmap A2 step 9)
//!
//! The kernel was hand-assembled PTX text until A2 step 9; it is now a
//! [`KernelIR`] ([`build`]) that the verifier checks before `nsl_kir`'s
//! printer lowers it. The phases, the barriers and the order of every
//! floating-point operation are the hand kernel's —
//! `tests/cfie_persistent_kir_equivalence.rs` runs the frozen hand emitter
//! and this one side by side on the cooperative-CTA interpreter and
//! requires the same output bits. It is assembled from its siblings'
//! sections where they are the same computation:
//!
//! * Attention is `cfie_decode_attention`'s flash-decode tile loop and
//!   output publish, run once per Q head over a [`FlashCtx`] this kernel's
//!   head loop builds (the head's Q row and attention-output row live in
//!   this kernel's shared memory, so the output is published to shared
//!   rather than global memory).
//! * Both RMSNorms start with `cfie_spec_sampler_ptx`'s sum-of-squares
//!   tree; the finish (every thread divides by `sqrt(mean + eps)`, and the
//!   result goes to its own row) is this kernel's.
//!
//! Addresses are element indices through `PtrOffset` rather than byte
//! offsets; loop-carried values are block parameters; shared memory is an
//! [`SmemLayout`] of ten f32 regions at the hand kernel's offsets. The
//! RoPE frequency is a bare `ex2`, [`KirOp::Exp2`] (`KirOp::Exp` is `e^x`
//! and scales by `log2(e)` first). The module targets the KIR floor
//! (`sm_70`), so [`DecodeBlockConfig`] has no `sm_version`.

use std::fmt::Write;

use crate::backend_ptx::lower_kir_to_ptx;
use crate::cfie_decode_attention::{
    at, cmp, konst, load, op2, prefix_pass, ptr, publish_output, widen, FlashCtx, HalfReader,
};
use crate::cfie_spec_sampler_ptx::{sum_of_squares_tree, SquareSum};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// Threads per CTA; also the attention softmax tile width and the FFN
/// gate/up staging tile width.
const BLOCK_DIM: u32 = 128;
const FFN_TILE: u32 = 128;

pub const KERNEL_NAME: &str = "nsl_cfie_decode_block";

pub fn kernel_name() -> &'static str {
    KERNEL_NAME
}

/// Compile-time model + KV-pool configuration for the decode block.
///
/// There is no `sm_version`: the module targets the KIR floor and the
/// driver JIT-compiles it for the device it is loaded on.
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
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct DecodeBlockMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
}

/// The configuration contract. Panics name the violated condition; every
/// caller either checks the same conditions first (`serve.rs`) or is a
/// test.
fn check(cfg: &DecodeBlockConfig) {
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
        "d_ff must be in 1..=32768 (u32 weight-row arithmetic)"
    );
    assert!(
        cfg.n_heads * cfg.head_dim <= 8192,
        "n_heads * head_dim must be <= 8192 (u32 weight-row arithmetic)"
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
}

/// The pool strides for `cfg`, in elements — `cfie_decode_attention`'s
/// derivation, so both kernels address one pool.
fn kv_strides(cfg: &DecodeBlockConfig) -> crate::cfie_decode_attention::KvStrides {
    crate::cfie_decode_attention::kv_strides(&crate::cfie_decode_attention::DecodeAttentionConfig {
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        per_slot_max_tokens: cfg.per_slot_max_tokens,
        max_slots: cfg.max_slots,
        kv_dtype_bytes: 2,
    })
}

/// Index of each shared region in [`smem_layout`], in declaration order.
const R_X: u32 = 0;
const R_XN: u32 = 1;
const R_Q: u32 = 2;
const R_AO: u32 = 3;
const R_RED: u32 = 4;
const R_SC: u32 = 5;
const R_RSC: u32 = 6;
const R_LL: u32 = 7;
const R_H: u32 = 8;
const R_Y: u32 = 9;

/// All f32, 4-aligned and a multiple of 4 long, so the offsets are the
/// hand kernel's packed ones:
///
/// ```text
/// x   [d_model]   residual stream (in/out of both sub-blocks)
/// xn  [d_model]   RMSNorm output (input to the matvecs)
/// q   [nh*hd]     rotated query rows
/// ao  [nh*hd]     attention output rows
/// red [128]       norm tree-reduction scratch
/// sc  [128]       attention score tile
/// rsc [1], ll [1] online-softmax rescale and denominator
/// h   [128]       FFN silu(gate)*up staging tile
/// y   [d_model]   FFN down-projection accumulator
/// ```
fn smem_layout(cfg: &DecodeBlockConfig) -> SmemLayout {
    let f32_region = |name: &str, elems: u32| SmemRegion {
        name: name.to_string(),
        bytes: elems * 4,
        align: 4,
        elem: KirType::F32,
    };
    let nhd = cfg.n_heads * cfg.head_dim;
    SmemLayout {
        regions: vec![
            f32_region("x", cfg.d_model),
            f32_region("xn", cfg.d_model),
            f32_region("q", nhd),
            f32_region("ao", nhd),
            f32_region("red", BLOCK_DIM),
            f32_region("sc", BLOCK_DIM),
            f32_region("rsc", 1),
            f32_region("ll", 1),
            f32_region("h", FFN_TILE),
            f32_region("y", cfg.d_model),
        ],
        dynamic: false,
    }
}

/// Bytes of static shared memory the kernel for `cfg` declares. The KIR
/// verifier refuses a static layout past the 48 KB per-CTA cap, so a
/// caller that downgrades on a big footprint (`serve.rs`) asks this before
/// calling [`emit`].
pub fn smem_bytes(cfg: &DecodeBlockConfig) -> u32 {
    check(cfg);
    smem_layout(cfg)
        .total_bytes()
        .expect("`check` bounds every region, so the layout has a size")
}

/// Values the entry block defines once and every phase reads.
struct Block {
    tid: VarId,
    zero: VarId,
    one: VarId,
    /// u32 `BLOCK_DIM`: every strided loop's step.
    stride: VarId,
    /// u32 and u64 `d_model`.
    d: VarId,
    d_wide: VarId,
    f_zero: VarId,
    /// f32 `1.0`.
    f_one: VarId,
    /// f32 `pos` (the RoPE angle's factor).
    pos_f: VarId,
    /// f32 shared pointers, one per region.
    x: VarId,
    xn: VarId,
    ao: VarId,
    red: VarId,
    h: VarId,
    y: VarId,
}

/// `for (i = tid; i < end; i += BLOCK_DIM) body(i)`, entered from the
/// current block; the builder is left in the loop's exit block.
fn strided(b: &mut KirBuilder, k: &Block, end: VarId, body: impl FnOnce(&mut KirBuilder, VarId)) {
    let head = b.new_block();
    let body_block = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(head, KirType::U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k.tid])));

    b.set_block(head);
    let finished = cmp(b, i, end, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body_block)));

    b.set_block(body_block);
    body(b, i);
    let next = op2(b, KirType::U32, KirOp::Add, i, k.stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![next])));

    b.set_block(done);
}

/// One f16 weight row: `weights[start..]`, `start` a u64 element index.
#[derive(Clone, Copy)]
struct Row {
    weights: VarId,
    start: VarId,
}

/// The dot products of `rows` with `x[0..len]` (f32, shared), in one loop,
/// as the hand kernel's matvecs: per `j`, each row's
/// `acc = fma(w[j], x[j], acc)` — `fma(x[j], w[j], acc)` when `x_first`
/// (the FFN down projection's operand order). Entered from the current
/// block; the builder is left in the loop's exit block, and the returned
/// values are the finished dots, one per row.
fn dot_rows(b: &mut KirBuilder, k: &Block, x: VarId, len: VarId, rows: &[Row], x_first: bool) -> Vec<VarId> {
    use AddressSpace::{Global, Shared};
    use KirType::{F16, F32, U32, U64};

    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let j = b.add_block_param(head, U32);
    let accs: Vec<VarId> = rows.iter().map(|_| b.add_block_param(head, F32)).collect();
    let mut init = vec![k.zero];
    init.extend(rows.iter().map(|_| k.f_zero));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, init)));

    b.set_block(head);
    let finished = cmp(b, j, len, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let x_addr = at(b, F32, Shared, x, j);
    let xj = load(b, F32, x_addr, Shared);
    let j_wide = widen(b, j);
    let mut next = Vec::with_capacity(rows.len() + 1);
    for (row, &acc) in rows.iter().zip(&accs) {
        let index = op2(b, U64, KirOp::Add, row.start, j_wide);
        let w_addr = at(b, F16, Global, row.weights, index);
        let w_raw = load(b, F16, w_addr, Global);
        let w = b.new_typed_var(F32);
        b.emit(KirOp::Cast(w, w_raw, F32));
        let acc_next = b.new_typed_var(F32);
        let (p, q) = if x_first { (xj, w) } else { (w, xj) };
        b.emit(KirOp::Fma(acc_next, p, q, acc));
        next.push(acc_next);
    }
    let j_next = op2(b, U32, KirOp::Add, j, k.one);
    next.insert(0, j_next);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, next)));

    b.set_block(done);
    accs
}

/// `xn = rmsnorm(x) * weight` over `d_model`: the shared sum-of-squares
/// tree into `red`, then every thread takes `inv = 1 / sqrt(red[0] / d +
/// eps)` and scales its strided elements, `xn[i] = (x[i] * inv) *
/// weight[i]`. Leaves the builder after the closing barrier.
fn build_rmsnorm(b: &mut KirBuilder, k: &Block, cfg: &DecodeBlockConfig, weight: VarId) {
    use AddressSpace::{Global, Shared};
    use KirType::F32;

    sum_of_squares_tree(
        b,
        SquareSum { tid: k.tid, len: k.d, stride: k.stride, f_zero: k.f_zero, src: k.x, scratch: k.red },
    );
    let total = load(b, F32, k.red, Shared);
    let inv_d = konst(b, ConstValue::F32(1.0f32 / cfg.d_model as f32));
    let mean = op2(b, F32, KirOp::Mul, total, inv_d);
    let eps = konst(b, ConstValue::F32(cfg.eps));
    let shifted = op2(b, F32, KirOp::Add, mean, eps);
    let root = b.new_typed_var(F32);
    b.emit(KirOp::Sqrt(root, shifted));
    let inv = op2(b, F32, KirOp::Div, k.f_one, root);

    strided(b, k, k.d, |b, i| {
        let x_addr = at(b, F32, Shared, k.x, i);
        let xi = load(b, F32, x_addr, Shared);
        let w_addr = at(b, F32, Global, weight, i);
        let w = load(b, F32, w_addr, Global);
        let scaled = op2(b, F32, KirOp::Mul, xi, inv);
        let normed = op2(b, F32, KirOp::Mul, scaled, w);
        let xn_addr = at(b, F32, Shared, k.xn, i);
        b.emit(KirOp::Store(xn_addr, normed, Shared));
    });
    b.emit(KirOp::Barrier);
}

/// Thread `p` of each strided round owns elements `e0 = 2p` and `e0 + 1`
/// of a `pairs * 2`-wide projection: the dual dot of weight rows `e0` and
/// `e0 + 1` with `xn`, rotated by RoPE at `pos` (`i2 = e0 % head_dim`,
/// `angle = pos * 2^(i2 * c_rope)`), handed to `store(e0, even, odd)`.
/// Leaves the builder in the loop's exit block.
fn rope_pairs(
    b: &mut KirBuilder,
    k: &Block,
    cfg: &DecodeBlockConfig,
    pairs: u32,
    weights: VarId,
    store: impl Fn(&mut KirBuilder, VarId, VarId, VarId),
) {
    use KirType::{F32, U32, U64};

    let end = konst(b, ConstValue::U32(pairs));
    strided(b, k, end, |b, p| {
        let e0 = op2(b, U32, KirOp::Shl, p, k.one);
        let row0 = op2(b, U32, KirOp::Mul, e0, k.d);
        let start0 = widen(b, row0);
        let start1 = op2(b, U64, KirOp::Add, start0, k.d_wide);
        let dots = dot_rows(
            b,
            k,
            k.xn,
            k.d,
            &[Row { weights, start: start0 }, Row { weights, start: start1 }],
            false,
        );
        let (e, o) = (dots[0], dots[1]);

        let head_dim = konst(b, ConstValue::U32(cfg.head_dim));
        let i2 = op2(b, U32, KirOp::Rem, e0, head_dim);
        let i2_f = b.new_typed_var(F32);
        b.emit(KirOp::Cast(i2_f, i2, F32));
        let c_rope = konst(b, ConstValue::F32(-(cfg.rope_theta.log2()) / cfg.head_dim as f32));
        let exponent = op2(b, F32, KirOp::Mul, i2_f, c_rope);
        let freq = b.new_typed_var(F32);
        b.emit(KirOp::Exp2(freq, exponent));
        let angle = op2(b, F32, KirOp::Mul, k.pos_f, freq);
        let sin = b.new_typed_var(F32);
        b.emit(KirOp::Sin(sin, angle));
        let cos = b.new_typed_var(F32);
        b.emit(KirOp::Cos(cos, angle));

        // (e', o') = (e*cos - o*sin, e*sin + o*cos)
        let e_cos = op2(b, F32, KirOp::Mul, e, cos);
        let o_sin = op2(b, F32, KirOp::Mul, o, sin);
        let even = op2(b, F32, KirOp::Sub, e_cos, o_sin);
        let e_sin = op2(b, F32, KirOp::Mul, e, sin);
        let odd = b.new_typed_var(F32);
        b.emit(KirOp::Fma(odd, o, cos, e_sin));
        store(b, e0, even, odd);
    });
}

/// `pool[index] = f16(value)`.
fn store_half(b: &mut KirBuilder, pool: VarId, index: VarId, value: VarId) {
    let half = b.new_typed_var(KirType::F16);
    b.emit(KirOp::Cast(half, value, KirType::F16));
    let addr = at(b, KirType::F16, AddressSpace::Global, pool, index);
    b.emit(KirOp::Store(addr, half, AddressSpace::Global));
}

/// Build the persistent decode-block kernel as KIR.
///
/// The phases, in order (each strided loop is `for i = tid; i < n;
/// i += 128`; `bar` is a CTA barrier):
///
/// ```text
/// 1  x[i] = x_in[i]                                          bar
/// 2  xn = rmsnorm(x) * norm1_w        (sum-of-squares tree)  bar x9
/// 3a q[2p], q[2p+1]     = rope(wq rows 2p, 2p+1 . xn)
/// 3b k_pool[pos][2p..]  = f16(rope(wk rows . xn))
/// 3c v_pool[pos][e]     = f16(wv row e . xn)                 bar
/// 4  per Q head hh:   (acc, m, l) = flash tile loop over pos+1 tokens
///                     ao[hh*hd + tid] = acc / l               bar x3 per tile, x2 per head
/// 5  x[i] += wo row i . ao                                   bar
/// 6  xn = rmsnorm(x) * norm2_w                               bar x9
/// 7  y[i] = 0                                                bar
///    per 128-row d_ff tile tb:
///      h[tid] = silu(w_gate row . xn) * (w_up row . xn)      bar
///      y[i] += w_down[i, tb..tb+fcnt] . h                    bar
/// 8  x_out[i] = x[i] + y[i]
/// ```
///
/// Only thread 0's softmax state is meaningful (see
/// `cfie_decode_attention::flash_tile`).
pub fn build(cfg: &DecodeBlockConfig) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F16, F32, U32, U64};

    check(cfg);
    let hd = cfg.head_dim;
    let nhd = cfg.n_heads * hd;
    let kv_rows = cfg.n_kv_heads * hd;
    let strides = kv_strides(cfg);

    let mut b = KirBuilder::new(KERNEL_NAME);
    // The params, in FFI order (the launcher marshals them positionally).
    let x_in = b.add_param("x_in_ptr", ptr(F32, Global), Global);
    let x_out = b.add_param("x_out_ptr", ptr(F32, Global), Global);
    let wq = b.add_param("wq_ptr", ptr(F16, Global), Global);
    let wk = b.add_param("wk_ptr", ptr(F16, Global), Global);
    let wv = b.add_param("wv_ptr", ptr(F16, Global), Global);
    let wo = b.add_param("wo_ptr", ptr(F16, Global), Global);
    let w_gate = b.add_param("w_gate_ptr", ptr(F16, Global), Global);
    let w_up = b.add_param("w_up_ptr", ptr(F16, Global), Global);
    let w_down = b.add_param("w_down_ptr", ptr(F16, Global), Global);
    let norm1_w = b.add_param("norm1_w_ptr", ptr(F32, Global), Global);
    let norm2_w = b.add_param("norm2_w_ptr", ptr(F32, Global), Global);
    let kv_base = b.add_param("kv_base", ptr(F16, Global), Global);
    let layer_idx = b.add_param("layer_idx", U32, Global);
    let slot_idx = b.add_param("slot_idx", U32, Global);
    let pos = b.add_param("pos", U32, Global);

    b.set_smem_layout(smem_layout(cfg));
    b.set_workgroup_size([BLOCK_DIM, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let region = |b: &mut KirBuilder, r: u32| {
        let dst = b.new_typed_var(ptr(F32, Shared));
        b.emit(KirOp::SharedRegion { dst, region: r });
        dst
    };
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let pos_f = b.new_typed_var(F32);
    b.emit(KirOp::Cast(pos_f, pos, F32));
    let k = Block {
        tid,
        zero,
        one,
        stride: konst(&mut b, ConstValue::U32(BLOCK_DIM)),
        d: konst(&mut b, ConstValue::U32(cfg.d_model)),
        d_wide: konst(&mut b, ConstValue::U64(cfg.d_model as u64)),
        f_zero: konst(&mut b, ConstValue::F32(0.0)),
        f_one: konst(&mut b, ConstValue::F32(1.0)),
        pos_f,
        x: region(&mut b, R_X),
        xn: region(&mut b, R_XN),
        ao: region(&mut b, R_AO),
        red: region(&mut b, R_RED),
        h: region(&mut b, R_H),
        y: region(&mut b, R_Y),
    };
    let q_smem = region(&mut b, R_Q);
    let scores = region(&mut b, R_SC);
    let rescale = region(&mut b, R_RSC);
    let l_smem = region(&mut b, R_LL);

    // seq_len after the KV append below.
    let seq_len = op2(&mut b, U32, KirOp::Add, pos, one);
    // K/V planes of this layer, and this slot's token range.
    let (k_half, v_half) = HalfReader::uniform_f16(&mut b, kv_base, layer_idx, strides.kv_half_stride);
    let per_slot = konst(&mut b, ConstValue::U32(cfg.per_slot_max_tokens));
    let slot_base = op2(&mut b, U32, KirOp::Mul, slot_idx, per_slot);
    let token_stride = konst(&mut b, ConstValue::U64(strides.token_stride));
    // The append target: token record (layer, slot, pos).
    let rec = op2(&mut b, U32, KirOp::Add, slot_base, pos);
    let rec_wide = widen(&mut b, rec);
    let rec_row = op2(&mut b, U64, KirOp::Mul, rec_wide, token_stride);
    let k_rec = k_half.in_plane(&mut b, rec_row);
    let v_rec = v_half.in_plane(&mut b, rec_row);

    // ── phase 1: x -> SMEM ──────────────────────────────────────────
    strided(&mut b, &k, k.d, |b, i| {
        let src = at(b, F32, Global, x_in, i);
        let xi = load(b, F32, src, Global);
        let dst = at(b, F32, Shared, k.x, i);
        b.emit(KirOp::Store(dst, xi, Shared));
    });
    b.emit(KirOp::Barrier);

    // ── phase 2: RMSNorm1 ───────────────────────────────────────────
    build_rmsnorm(&mut b, &k, cfg, norm1_w);

    // ── phase 3a: Q = rope(Wq @ xn) -> SMEM ─────────────────────────
    rope_pairs(&mut b, &k, cfg, nhd / 2, wq, |b, e0, even, odd| {
        let e_addr = at(b, F32, Shared, q_smem, e0);
        b.emit(KirOp::Store(e_addr, even, Shared));
        let e1 = op2(b, U32, KirOp::Add, e0, one);
        let o_addr = at(b, F32, Shared, q_smem, e1);
        b.emit(KirOp::Store(o_addr, odd, Shared));
    });

    // ── phase 3b: K = rope(Wk @ xn), appended to the pool (f16) ────
    rope_pairs(&mut b, &k, cfg, kv_rows / 2, wk, |b, e0, even, odd| {
        let e0_wide = widen(b, e0);
        let index = op2(b, U64, KirOp::Add, k_rec, e0_wide);
        store_half(b, kv_base, index, even);
        let next = konst(b, ConstValue::U64(1));
        let index1 = op2(b, U64, KirOp::Add, index, next);
        store_half(b, kv_base, index1, odd);
    });

    // ── phase 3c: V = Wv @ xn, appended to the pool (f16) ───────────
    let v_end = konst(&mut b, ConstValue::U32(kv_rows));
    strided(&mut b, &k, v_end, |b, e| {
        let row = op2(b, U32, KirOp::Mul, e, k.d);
        let start = widen(b, row);
        let dot = dot_rows(b, &k, k.xn, k.d, &[Row { weights: wv, start }], false)[0];
        let e_wide = widen(b, e);
        let index = op2(b, U64, KirOp::Add, v_rec, e_wide);
        store_half(b, kv_base, index, dot);
    });
    // Publishes q and orders the pool stores before the attention reads
    // (bar.sync has membar.cta effect).
    b.emit(KirOp::Barrier);

    // ── phase 4: flash-decode attention, Q heads sequential ─────────
    let heads_head = b.new_block();
    let head_body = b.new_block();
    let heads_done = b.new_block();
    let hh = b.add_block_param(heads_head, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(heads_head, vec![zero])));

    b.set_block(heads_head);
    let n_heads = konst(&mut b, ConstValue::U32(cfg.n_heads));
    let heads_finished = cmp(&mut b, hh, n_heads, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(heads_finished, KirEdge::to(heads_done), KirEdge::to(head_body)));

    b.set_block(head_body);
    // kv_head = q_head / group (baked divisor).
    let group = konst(&mut b, ConstValue::U32(cfg.n_heads / cfg.n_kv_heads));
    let kv_head = op2(&mut b, U32, KirOp::Div, hh, group);
    let head_dim = konst(&mut b, ConstValue::U32(hd));
    let head_off32 = op2(&mut b, U32, KirOp::Mul, kv_head, head_dim);
    let head_off = widen(&mut b, head_off32);
    let row = op2(&mut b, U32, KirOp::Mul, hh, head_dim);
    let q_row = at(&mut b, F32, Shared, q_smem, row);
    let inv_sqrt_hd = konst(&mut b, ConstValue::F32(1.0f32 / (hd as f32).sqrt()));
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
        q_smem: q_row,
        scores,
        rescale,
        l_smem,
    };
    let (acc, _m, l) = prefix_pass(&mut b, &ctx, seq_len);
    publish_output(&mut b, &ctx, acc, l, (k.ao, Shared), row);
    // rescale / l / scores are reused by the next head.
    b.emit(KirOp::Barrier);
    let hh_next = op2(&mut b, U32, KirOp::Add, hh, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(heads_head, vec![hh_next])));

    // ── phase 5: x += Wo @ attn_out (thread d owns x[d]) ────────────
    b.set_block(heads_done);
    let nhd_c = konst(&mut b, ConstValue::U32(nhd));
    strided(&mut b, &k, k.d, |b, i| {
        let row = op2(b, U32, KirOp::Mul, i, nhd_c);
        let start = widen(b, row);
        let dot = dot_rows(b, &k, k.ao, nhd_c, &[Row { weights: wo, start }], false)[0];
        let x_addr = at(b, F32, Shared, k.x, i);
        let xi = load(b, F32, x_addr, Shared);
        let sum = op2(b, F32, KirOp::Add, xi, dot);
        b.emit(KirOp::Store(x_addr, sum, Shared));
    });
    b.emit(KirOp::Barrier);

    // ── phase 6: RMSNorm2 ───────────────────────────────────────────
    build_rmsnorm(&mut b, &k, cfg, norm2_w);

    // ── phase 7: FFN (tiled gate/up + down accumulation) ────────────
    strided(&mut b, &k, k.d, |b, i| {
        let y_addr = at(b, F32, Shared, k.y, i);
        b.emit(KirOp::Store(y_addr, k.f_zero, Shared));
    });
    b.emit(KirOp::Barrier);

    let ft_head = b.new_block();
    let ft_body = b.new_block();
    let gate_up = b.new_block();
    let h_done = b.new_block();
    let ft_done = b.new_block();
    let tb = b.add_block_param(ft_head, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(ft_head, vec![zero])));

    b.set_block(ft_head);
    let d_ff = konst(&mut b, ConstValue::U32(cfg.d_ff));
    let ffn_finished = cmp(&mut b, tb, d_ff, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(ffn_finished, KirEdge::to(ft_done), KirEdge::to(ft_body)));

    b.set_block(ft_body);
    let remaining = op2(&mut b, U32, KirOp::Sub, d_ff, tb);
    let ffn_tile = konst(&mut b, ConstValue::U32(FFN_TILE));
    let fcnt = op2(&mut b, U32, KirOp::Min, remaining, ffn_tile);
    let idle = cmp(&mut b, tid, fcnt, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(idle, KirEdge::to(h_done), KirEdge::to(gate_up)));

    // gate/up dual matvec for row tb + tid, then h = silu(gate) * up.
    b.set_block(gate_up);
    let ff_row = op2(&mut b, U32, KirOp::Add, tb, tid);
    let ff_row_d = op2(&mut b, U32, KirOp::Mul, ff_row, k.d);
    let start = widen(&mut b, ff_row_d);
    let dots = dot_rows(
        &mut b,
        &k,
        k.xn,
        k.d,
        &[Row { weights: w_gate, start }, Row { weights: w_up, start }],
        false,
    );
    let (gate, up) = (dots[0], dots[1]);
    // silu(g) = g * sigmoid(g); sigmoid = 1/(1 + ex2(-g*log2e)).
    let neg_log2e = konst(&mut b, ConstValue::F32(-std::f32::consts::LOG2_E));
    let scaled = op2(&mut b, F32, KirOp::Mul, gate, neg_log2e);
    let e = b.new_typed_var(F32);
    b.emit(KirOp::Exp2(e, scaled));
    let den = op2(&mut b, F32, KirOp::Add, e, k.f_one);
    let sig = op2(&mut b, F32, KirOp::Div, k.f_one, den);
    let silu = op2(&mut b, F32, KirOp::Mul, gate, sig);
    let hv = op2(&mut b, F32, KirOp::Mul, silu, up);
    let h_addr = at(&mut b, F32, Shared, k.h, tid);
    b.emit(KirOp::Store(h_addr, hv, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(h_done)));

    // down accumulation: y[i] += w_down[i, tb..tb+fcnt] . h
    b.set_block(h_done);
    b.emit(KirOp::Barrier);
    strided(&mut b, &k, k.d, |b, i| {
        let row = op2(b, U32, KirOp::Mul, i, d_ff);
        let row_wide = widen(b, row);
        let col = widen(b, tb);
        let start = op2(b, U64, KirOp::Add, row_wide, col);
        let dot = dot_rows(b, &k, k.h, fcnt, &[Row { weights: w_down, start }], true)[0];
        let y_addr = at(b, F32, Shared, k.y, i);
        let yi = load(b, F32, y_addr, Shared);
        let sum = op2(b, F32, KirOp::Add, yi, dot);
        b.emit(KirOp::Store(y_addr, sum, Shared));
    });
    // The h tile is rewritten next iteration.
    b.emit(KirOp::Barrier);
    let tb_next = op2(&mut b, U32, KirOp::Add, tb, ffn_tile);
    b.terminate(KirTerminator::Branch(KirEdge::with(ft_head, vec![tb_next])));

    // ── phase 8: x_out = x + ffn (second residual) ──────────────────
    b.set_block(ft_done);
    strided(&mut b, &k, k.d, |b, i| {
        let x_addr = at(b, F32, Shared, k.x, i);
        let xi = load(b, F32, x_addr, Shared);
        let y_addr = at(b, F32, Shared, k.y, i);
        let yi = load(b, F32, y_addr, Shared);
        let sum = op2(b, F32, KirOp::Add, xi, yi);
        let out = at(b, F32, Global, x_out, i);
        b.emit(KirOp::Store(out, sum, Global));
    });
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// The `//` header: the hand kernel's lines, unchanged. The stride lines
/// are compared with `cfie_decode_attention`'s byte for byte.
fn header_comment(cfg: &DecodeBlockConfig) -> String {
    let s = kv_strides(cfg);
    let mut w = String::new();
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
        cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff
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
    writeln!(w, "// rope_theta={} eps={}", cfg.rope_theta, cfg.eps).unwrap();
    writeln!(
        w,
        "// Approx ops: ex2/sin/cos .approx.f32 (RoPE + softmax + silu)."
    )
    .unwrap();
    writeln!(w, "//").unwrap();
    w
}

/// Emit the persistent decode-block kernel: build, verify, lower, and
/// prefix the `//` header.
///
/// Launch shape: grid = 1 CTA, block = 128 threads; one launch per
/// layer per decode step.  `pos` drives both the RoPE angle and the
/// KV-append offset; seq_len after the append is `pos + 1`.
///
/// The returned text carries no NUL: the serve path appends the one
/// `cuModuleLoadData` needs when it embeds the module.
///
/// # Panics
///
/// On a configuration outside the contract (see the asserts in `check`),
/// on a shared footprint past the 48 KB static cap (check [`smem_bytes`]
/// first), and if the built kernel otherwise fails KIR verification — a
/// bug in this module, not a condition a caller can provoke.
pub fn emit(cfg: &DecodeBlockConfig) -> (String, DecodeBlockMeta) {
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

    let mut p = header_comment(cfg);
    p.push_str(module);
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
    use crate::kernel_ir::{AddressSpace, ConstValue, KernelIR, KirOp, KirType};

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
        }
    }

    fn ops(ir: &KernelIR) -> Vec<&KirOp> {
        ir.blocks.iter().flat_map(|b| b.ops.iter()).collect()
    }

    fn u64_consts(ir: &KernelIR) -> Vec<u64> {
        ops(ir)
            .into_iter()
            .filter_map(|op| match op {
                KirOp::Const(_, crate::kernel_ir::KirConst { value: ConstValue::U64(v), .. }) => Some(*v),
                _ => None,
            })
            .collect()
    }

    fn u32_consts(ir: &KernelIR) -> Vec<u32> {
        ops(ir)
            .into_iter()
            .filter_map(|op| match op {
                KirOp::Const(_, crate::kernel_ir::KirConst { value: ConstValue::U32(v), .. }) => Some(*v),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn param_list_is_exactly_the_fifteen_block_params() {
        let ir = build(&paper_cfg());
        let names: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
        let pointers = [
            "x_in_ptr",
            "x_out_ptr",
            "wq_ptr",
            "wk_ptr",
            "wv_ptr",
            "wo_ptr",
            "w_gate_ptr",
            "w_up_ptr",
            "w_down_ptr",
            "norm1_w_ptr",
            "norm2_w_ptr",
            "kv_base",
        ];
        let scalars = ["layer_idx", "slot_idx", "pos"];
        assert_eq!(names, [&pointers[..], &scalars[..]].concat());
        let ptx = emit_decode_block_ptx(&paper_cfg());
        for name in pointers {
            assert!(ptx.contains(&format!(".param .u64 param_{name}")), "{name}");
        }
        for name in scalars {
            assert!(ptx.contains(&format!(".param .u32 param_{name}")), "{name}");
        }
        // Exactly the declared params are ever loaded.
        assert_eq!(ptx.matches("ld.param").count(), 15);
    }

    #[test]
    fn the_global_stores_are_the_output_row_and_the_kv_append() {
        // f32 stores: x_out only. f16 stores: the K pair (two) and the V
        // element — the record at (layer, slot, pos).
        let ir = build(&paper_cfg());
        let stores: Vec<Option<&KirType>> = ops(&ir)
            .into_iter()
            .filter_map(|op| match op {
                KirOp::Store(_, v, AddressSpace::Global) => Some(ir.var_types.get(v)),
                _ => None,
            })
            .collect();
        assert_eq!(stores.iter().filter(|t| **t == Some(&KirType::F32)).count(), 1);
        assert_eq!(stores.iter().filter(|t| **t == Some(&KirType::F16)).count(), 3);
        assert_eq!(stores.len(), 4);
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
        // token_stride = 4*64 = 256 elements; the layer stride = 2 * 64*2048*256
        // = 67108864 elements (the hand kernel's 134217728 bytes).
        assert!(ptx.contains("//   token_stride        = 256"));
        let ir = build(&cfg);
        let wide = u64_consts(&ir);
        for stride in [256u64, 33_554_432, 67_108_864] {
            assert!(wide.contains(&stride), "{stride} in {wide:?}");
        }
        assert!(u32_consts(&ir).contains(&2048), "per_slot_max_tokens");
        assert_eq!(meta.kernel_name, kernel_name());
        assert_eq!(meta.block_dim, 128);
        // SMEM: (3*d_model + 2*nh*hd + 128 + 128 + 2 + 128) * 4.
        let expected = (3 * 512 + 2 * 512 + 128 + 128 + 2 + 128) * 4;
        assert_eq!(meta.smem_bytes, expected);
        assert!(ptx.contains(&format!("[{}];", expected)), "one static shared block of {expected} bytes");
    }

    #[test]
    fn smem_bytes_is_what_emit_declares_and_answers_past_the_cap() {
        let cfg = paper_cfg();
        assert_eq!(smem_bytes(&cfg), emit(&cfg).1.smem_bytes);
        // serve asks before emitting: a footprint past 48 KB is answered,
        // not refused (d_model 4096: 3 * 16 KB of residual rows alone).
        let big = DecodeBlockConfig { d_model: 4096, ..paper_cfg() };
        assert!(smem_bytes(&big) > 48 * 1024);
    }

    #[test]
    fn module_targets_the_kir_floor_after_the_header() {
        let ptx = emit_decode_block_ptx(&paper_cfg());
        assert!(ptx.starts_with("//"));
        let directives: Vec<&str> = ptx
            .lines()
            .filter(|l| l.starts_with(".version") || l.starts_with(".target") || l.starts_with(".address_size"))
            .collect();
        assert_eq!(directives.len(), 3, "{directives:?}");
        assert!(directives[1].starts_with(".target sm_70"), "{directives:?}");
        assert_eq!(directives[2], ".address_size 64");
    }

    #[test]
    fn rope_and_silu_use_documented_approx_ops() {
        let ptx = emit_decode_block_ptx(&paper_cfg());
        assert!(ptx.contains("sin.approx.f32"));
        assert!(ptx.contains("cos.approx.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        let ir = build(&paper_cfg());
        let count = |f: fn(&KirOp) -> bool| ops(&ir).into_iter().filter(|op| f(op)).count();
        // RoPE's frequency (the Q and K pair loops) and silu's sigmoid are
        // bare 2^x; the softmax's two exponentials are e^x.
        assert_eq!(count(|op| matches!(op, KirOp::Exp2(..))), 3);
        assert_eq!(count(|op| matches!(op, KirOp::Exp(..))), 2);
        assert_eq!(count(|op| matches!(op, KirOp::Sin(..))), 2);
        // seq_len derives from pos, not a separate param.
        assert!(!ir.params.iter().any(|p| p.name == "seq_len"));
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
        };
        let d = 8usize;
        let (hd, nh, nkv) = (4usize, 2usize, 1usize);
        let generator = |n: usize, f: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * f).sin() * 0.5).collect()
        };
        let wq = generator(nh * hd * d, 0.31);
        let wk = generator(nkv * hd * d, 0.47);
        let wv = generator(nkv * hd * d, 0.59);
        let wo = generator(d * nh * hd, 0.73);
        let zeros_g = vec![0.0f32; 16 * d];
        let zeros_d = vec![0.0f32; d * 16];
        let n1 = generator(d, 0.83).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let n2 = vec![1.0f32; d];
        let x = generator(d, 1.13);
        // Two pre-existing tokens in the pool.
        let pos = 2u32;
        let mut kk = generator(pos as usize * nkv * hd, 0.91);
        let mut kv = generator(pos as usize * nkv * hd, 1.07);

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
                nsl_log::nsl_log!(INFO, "skip", "[skip] cfie decode-block ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie decode-block PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }
}
