// crates/nsl-kir/src/kernels/flash_attn/mod.rs
//! Tensor-core flash attention for the training path (forward and
//! backward), built on KIR.
//!
//! These kernels serve `scaled_dot_product_attention` on f32 tensors laid
//! out `[batch, heads, seq, head_dim]` (row-major, contiguous), causal or
//! not, with grouped-query attention native: K and V may carry
//! `kv_heads` heads, each shared by `heads / kv_heads` consecutive query
//! heads. Sequence lengths need not be a multiple of any tile: every load
//! is clamped into the tensor, every key past `seq` is masked out of the
//! softmax, and every store past `seq` is skipped.
//!
//! # Precision contract
//!
//! Inputs and outputs are f32. The two products per tile run on the tensor
//! cores as `mma.sync.m16n8k16` with **f16 operands rounded to nearest from
//! f32 and f32 accumulation** — the operand precision NSL's flash attention
//! has used since it gained tensor cores, so the forward's agreement with an
//! exact reference is unchanged by these kernels in kind. In detail:
//!
//! - Forward: `Q`, `K`, `V` are rounded to f16 as they reach the tensor
//!   cores; the scores, the online softmax (running max, rescale, sum) and
//!   the output accumulator are f32, `exp2` is `ex2.approx.f32`; the softmax
//!   weights `P` are rounded to f16 for `P·V`. The output is normalised and
//!   **stored as f32** (the kernels these replace rounded it to f16 first),
//!   and the natural log-sum-exp of the scaled scores is stored per row.
//! - Backward: gradients are not activations. `dO` sits around 1e-6..1e-8
//!   in a real step, below f16's smallest normal (6.1e-5), so it is
//!   multiplied by an exact power of two before any f16 rounding and the
//!   results are multiplied back by its reciprocal as they leave the f32
//!   accumulators (see [`bwd`]: per row for `dQ`, per batch and kv-head
//!   group for `dK`/`dV`). `D = rowsum(dO ∘ O)` is computed in f32 and
//!   scaled by the same power. `dS` is clamped to f16's finite range before
//!   it is rounded. Accumulation is f32 throughout, and no output is formed
//!   with atomics: each `dQ`, `dK` and `dV` element has exactly one writer,
//!   so the backward is bit-reproducible run to run.
//!
//! What f16 operands cost against f32 (TF32) matmuls is the operands'
//! 11-bit significand; `tools/refcheck` measures the end-to-end effect
//! against PyTorch (see its README).
//!
//! # Against the kernels these replace
//!
//! The operand precision is the same (f16, round to nearest, f32
//! accumulation), so the f32 / TF32 training path's agreement with PyTorch
//! is unchanged in kind. Three things differ, each toward the exact answer
//! or toward reproducibility:
//!
//! - the forward output is stored as f32 instead of being rounded to f16;
//! - the gradient prescale is per row (dQ) and per batch and kv-head group
//!   (dK/dV) instead of per (batch-head, 64-row query tile), with the same
//!   target binade;
//! - dQ is formed without atomics, so two runs give the same bits.
//!
//! Measured 2026-11-14 with `tools/refcheck/grad_check.py` at Coder-50M,
//! production flags, TF32: median 1.21e-3 / worst 2.10e-3 relative error
//! against PyTorch here, against 1.22e-3 / 2.95e-3 for the kernels these
//! replace; with `NSL_MATMUL_BF16=1` 3.93e-3 / 5.62e-3 against 3.92e-3 /
//! 5.74e-3 (the bf16 GEMMs set that floor). Kernel-level checks against f64
//! are the `tc_*` tests in nsl-runtime's `attention_tc.rs`.

pub mod bwd;
pub mod fwd;

use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KirBuilder, KirConst, KirOp, KirType, MmaOperandTy, MmaShape,
    ShuffleMode, VarId,
};

/// `log2(e)`, for turning `e^x` into `2^(x · log2 e)`.
pub const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// `ln(2)`, for turning a base-2 log back into a natural one.
pub const LN_2: f32 = std::f32::consts::LN_2;

/// f16's largest finite magnitude. `dS` is clamped to it before rounding.
pub const F16_MAX: f32 = 65504.0;

/// Shared-memory tile rows are padded by 8 halves (16 bytes): with a row
/// stride of `(head_dim + 8) · 2` bytes, the eight 16-byte rows one
/// `ldmatrix` 8x8 matrix reads start 4 banks apart and cover all 32 banks
/// once, so the loads are conflict-free for head_dim 64 and 128.
pub const SMEM_ROW_PAD: u32 = 8;

/// The packed f16 pair type every fragment register and every shared tile
/// word carries.
pub(crate) fn h2() -> KirType {
    MmaOperandTy::F16.fragment_ty()
}

pub(crate) fn f32_global() -> KirType {
    KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
}

/// The global pointer type of an attention operand stored as `io` (option
/// (a) of bf16 activation storage: Q, K, V, O, dO, dQ, dK, dV in bf16; the
/// log-sum-exp, `D` and the per-head maxima stay f32).
pub(crate) fn io_global(io: crate::kernels::storage::Storage) -> KirType {
    io.ptr()
}

pub(crate) fn u32_global() -> KirType {
    KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global)
}

pub(crate) fn h2_shared() -> KirType {
    KirType::Ptr(Box::new(h2()), AddressSpace::Shared)
}

pub(crate) fn f32_shared() -> KirType {
    KirType::Ptr(Box::new(KirType::F32), AddressSpace::Shared)
}

/// A thin layer over [`KirBuilder`] for straight-line tensor-core code:
/// each method emits one op into the current block and returns its typed
/// destination.
pub(crate) struct G {
    pub b: KirBuilder,
}

impl G {
    pub fn new(name: &str) -> Self {
        G { b: KirBuilder::new(name) }
    }

    pub fn var(&mut self, ty: KirType) -> VarId {
        self.b.new_typed_var(ty)
    }

    fn ty(&self, v: VarId) -> KirType {
        self.b.var_type(v).expect("flash_attn: every value is typed")
    }

    pub fn konst(&mut self, ty: KirType, value: ConstValue) -> VarId {
        let d = self.var(ty.clone());
        self.b.emit(KirOp::Const(d, KirConst { ty, value }));
        d
    }

    pub fn u32c(&mut self, x: u32) -> VarId {
        self.konst(KirType::U32, ConstValue::U32(x))
    }

    pub fn u64c(&mut self, x: u64) -> VarId {
        self.konst(KirType::U64, ConstValue::U64(x))
    }

    pub fn f32c(&mut self, x: f32) -> VarId {
        self.konst(KirType::F32, ConstValue::F32(x))
    }

    /// A homogeneous two-operand op typed like `a`.
    pub fn op2(&mut self, op: fn(VarId, VarId, VarId) -> KirOp, a: VarId, b: VarId) -> VarId {
        let d = self.var(self.ty(a));
        self.b.emit(op(d, a, b));
        d
    }

    pub fn add(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Add, a, b)
    }

    pub fn sub(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Sub, a, b)
    }

    pub fn mul(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Mul, a, b)
    }

    pub fn div(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Div, a, b)
    }

    pub fn min(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Min, a, b)
    }

    pub fn max(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Max, a, b)
    }

    pub fn and(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::And, a, b)
    }

    pub fn or(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Or, a, b)
    }

    pub fn rem(&mut self, a: VarId, b: VarId) -> VarId {
        self.op2(KirOp::Rem, a, b)
    }

    pub fn shr(&mut self, a: VarId, amount: VarId) -> VarId {
        self.op2(KirOp::Shr, a, amount)
    }

    pub fn shl(&mut self, a: VarId, amount: VarId) -> VarId {
        self.op2(KirOp::Shl, a, amount)
    }

    /// `a + c` for a constant `c` of `a`'s type (u32).
    pub fn add_u32c(&mut self, a: VarId, c: u32) -> VarId {
        if c == 0 {
            return a;
        }
        let k = self.u32c(c);
        self.add(a, k)
    }

    pub fn mul_u32c(&mut self, a: VarId, c: u32) -> VarId {
        let k = self.u32c(c);
        self.mul(a, k)
    }

    pub fn fma(&mut self, a: VarId, b: VarId, c: VarId) -> VarId {
        let d = self.var(self.ty(a));
        self.b.emit(KirOp::Fma(d, a, b, c));
        d
    }

    pub fn neg(&mut self, a: VarId) -> VarId {
        let d = self.var(self.ty(a));
        self.b.emit(KirOp::Neg(d, a));
        d
    }

    pub fn abs(&mut self, a: VarId) -> VarId {
        let d = self.var(self.ty(a));
        self.b.emit(KirOp::Abs(d, a));
        d
    }

    pub fn exp2(&mut self, a: VarId) -> VarId {
        let d = self.var(KirType::F32);
        self.b.emit(KirOp::Exp2(d, a));
        d
    }

    pub fn log2(&mut self, a: VarId) -> VarId {
        let d = self.var(KirType::F32);
        self.b.emit(KirOp::Log2(d, a));
        d
    }

    pub fn cast(&mut self, a: VarId, ty: KirType) -> VarId {
        let d = self.var(ty.clone());
        self.b.emit(KirOp::Cast(d, a, ty));
        d
    }

    pub fn bitcast(&mut self, a: VarId, ty: KirType) -> VarId {
        let d = self.var(ty);
        self.b.emit(KirOp::Bitcast(d, a));
        d
    }

    pub fn cmp(&mut self, a: VarId, b: VarId, op: CmpOp) -> VarId {
        let d = self.var(KirType::Bool);
        self.b.emit(KirOp::Cmp(d, a, b, op));
        d
    }

    pub fn sel(&mut self, cond: VarId, t: VarId, f: VarId) -> VarId {
        let d = self.var(self.ty(t));
        self.b.emit(KirOp::Select(d, cond, t, f));
        d
    }

    /// `base + off` elements of `base`'s pointee (u32 or u64 `off`).
    pub fn ptr(&mut self, base: VarId, off: VarId) -> VarId {
        let d = self.var(self.ty(base));
        self.b.emit(KirOp::PtrOffset(d, base, off));
        d
    }

    pub fn ptr_c(&mut self, base: VarId, off: u32) -> VarId {
        if off == 0 {
            return base;
        }
        let k = self.u32c(off);
        self.ptr(base, k)
    }

    pub fn load(&mut self, p: VarId, space: AddressSpace) -> VarId {
        let pointee = match self.ty(p) {
            KirType::Ptr(t, _) => *t,
            other => panic!("flash_attn: load through a non-pointer {other:?}"),
        };
        let d = self.var(pointee);
        self.b.emit(KirOp::Load(d, p, space));
        d
    }

    /// `ld.<space>.v<N>` of `n` (2 or 4) pointee-typed values.
    pub fn load_vec(&mut self, p: VarId, n: usize, space: AddressSpace) -> Vec<VarId> {
        let pointee = match self.ty(p) {
            KirType::Ptr(t, _) => *t,
            other => panic!("flash_attn: load through a non-pointer {other:?}"),
        };
        let dsts: Vec<VarId> = (0..n).map(|_| self.var(pointee.clone())).collect();
        self.b.emit(KirOp::LoadVec { dsts: dsts.clone(), ptr: p, space });
        dsts
    }

    /// [`G::load_vec`] as f32 values: a bf16 pointee is widened
    /// (`cvt.f32.bf16`, exact). For an f32 pointee this is `load_vec` itself,
    /// so an f32 kernel emits exactly what it emitted before.
    pub fn load_vec_f32(&mut self, p: VarId, n: usize, space: AddressSpace) -> Vec<VarId> {
        let vals = self.load_vec(p, n, space);
        vals.into_iter()
            .map(|v| {
                if self.ty(v) == KirType::Bf16 {
                    let d = self.var(KirType::F32);
                    self.b.emit(KirOp::Cast(d, v, KirType::F32));
                    d
                } else {
                    v
                }
            })
            .collect()
    }

    /// f32 `vals` in the element type `p` points to: rounded once to nearest
    /// even (`cvt.rn.bf16.f32`) for a bf16 pointee, unchanged (no op) for an
    /// f32 one. For a store through `p`.
    pub fn narrow_to(&mut self, p: VarId, vals: Vec<VarId>) -> Vec<VarId> {
        let pointee = match self.ty(p) {
            KirType::Ptr(t, _) => *t,
            other => panic!("flash_attn: store through a non-pointer {other:?}"),
        };
        if pointee != KirType::Bf16 {
            return vals;
        }
        vals.into_iter()
            .map(|v| {
                let h = self.var(KirType::Bf16);
                self.b.emit(KirOp::Cast(h, v, KirType::Bf16));
                h
            })
            .collect()
    }

    pub fn store_vec(&mut self, p: VarId, vals: Vec<VarId>, space: AddressSpace) {
        self.b.emit(KirOp::StoreVec { ptr: p, vals, space });
    }

    /// A side effect that runs only where `pred` holds.
    pub fn when(&mut self, pred: VarId, op: KirOp) {
        self.b.emit(KirOp::Predicated { pred, negate: false, op: Box::new(op) });
    }

    /// `cvt.rn.f16x2.f32`: `lo` in the low half (the fragment's first
    /// column), `hi` in the high half.
    pub fn pack(&mut self, lo: VarId, hi: VarId) -> VarId {
        let d = self.var(h2());
        self.b.emit(KirOp::CvtPack { dst: d, hi, lo, ty: MmaOperandTy::F16 });
        d
    }

    pub fn ldmatrix_x4(&mut self, addr: VarId, trans: bool) -> [VarId; 4] {
        let d = [self.var(h2()), self.var(h2()), self.var(h2()), self.var(h2())];
        self.b.emit(KirOp::LdMatrix { dst: d.to_vec(), addr, trans });
        d
    }

    /// `d = a · b + c` on one m16n8k16 tile, f16 operands, f32 accumulator.
    pub fn mma(&mut self, a: [VarId; 4], b: [VarId; 2], c: [VarId; 4]) -> [VarId; 4] {
        let d = [
            self.var(KirType::F32),
            self.var(KirType::F32),
            self.var(KirType::F32),
            self.var(KirType::F32),
        ];
        self.b.emit(KirOp::Mma {
            shape: MmaShape::M16N8K16,
            a_ty: MmaOperandTy::F16,
            d: d.to_vec(),
            a: a.to_vec(),
            b: b.to_vec(),
            c: c.to_vec(),
        });
        d
    }

    /// `shfl.sync.bfly`: the value held by lane `lane ^ mask`.
    pub fn shfl_xor(&mut self, val: VarId, mask: u32) -> VarId {
        let m = self.u32c(mask);
        let d = self.var(self.ty(val));
        self.b.emit(KirOp::WarpShuffle { dst: d, val, lane: m, mode: ShuffleMode::Xor, width: 32 });
        d
    }

    /// Reduce `val` over the four lanes of a quad (the lanes that share
    /// an accumulator row) with `op`.
    pub fn quad_reduce(&mut self, val: VarId, op: fn(VarId, VarId, VarId) -> KirOp) -> VarId {
        let o1 = self.shfl_xor(val, 1);
        let v1 = self.op2(op, val, o1);
        let o2 = self.shfl_xor(v1, 2);
        self.op2(op, v1, o2)
    }

    /// The max of a non-empty list (`max.f32` keeps the non-NaN operand).
    pub fn max_all(&mut self, vals: &[VarId]) -> VarId {
        let mut acc = vals[0];
        for &v in &vals[1..] {
            acc = self.max(acc, v);
        }
        acc
    }

    pub fn sum_all(&mut self, vals: &[VarId]) -> VarId {
        let mut acc = vals[0];
        for &v in &vals[1..] {
            acc = self.add(acc, v);
        }
        acc
    }
}

/// The launch grid of a kernel with `tiles` sequence tiles per `rows`
/// (`batch · heads` or `batch · kv_heads`) row: `[tiles, rows, 1]`, or with
/// `lpt` (largest processing time first) `[rows, tiles, 1]`.
///
/// Under causal masking a CTA's work grows (forward, dQ) or shrinks (dK/dV)
/// with its tile index, by up to the tile count. The kernels already map
/// `blockIdx` so that within a row the heaviest tile comes first; but the
/// block scheduler dispatches in `blockIdx` order, x fastest, so with x over
/// tiles the heaviest CTAs of the last rows are dispatched only after every
/// earlier row's light ones, and finish last. With x over rows, every row's
/// heaviest tile is dispatched in the first wave and the lightest CTAs fill
/// the end: the greedy longest-first schedule. It matters most for the
/// grouped dK/dV kernel, whose CTAs each walk a whole group of query heads,
/// so there are `heads / kv_heads` times fewer, longer ones. Which CTA does
/// which tile is unchanged, so the results are the same bits.
pub(crate) fn grid_order(lpt: bool, tiles: u32, rows: u32) -> [u32; 3] {
    if lpt { [rows, tiles, 1] } else { [tiles, rows, 1] }
}

/// The `(tile, row)` block indices of a kernel launched with
/// [`grid_order`]`(lpt, ..)`.
pub(crate) fn tile_and_row(g: &mut G, lpt: bool) -> (VarId, VarId) {
    let (tile_axis, row_axis) = if lpt { (1, 0) } else { (0, 1) };
    let tile = g.var(KirType::U32);
    g.b.emit(KirOp::BlockIdx(tile, tile_axis));
    let row = g.var(KirType::U32);
    g.b.emit(KirOp::BlockIdx(row, row_axis));
    (tile, row)
}

/// The power of two that lifts a gradient whose largest magnitude has the
/// f32 bit pattern `max_bits` (non-negative, so its biased exponent is
/// `max_bits >> 23`) into `[2^GRAD_SCALE_EXP, 2^(GRAD_SCALE_EXP + 1))`, and
/// its reciprocal, as f32 values: `(gs, 1/gs)`, both exact powers of two
/// and both normal.
///
/// A maximum below `2^(GRAD_EXP_MIN - 127)` — an all-zero gradient
/// included — is scaled as if it were exactly that; an infinite or NaN one
/// as if it were the largest finite exponent (the non-finite value still
/// propagates).
pub(crate) fn grad_scale_from_max_bits(g: &mut G, max_bits: VarId) -> (VarId, VarId) {
    let e = {
        let sh = g.u32c(23);
        g.shr(max_bits, sh)
    };
    let lo = g.u32c(GRAD_EXP_MIN);
    let hi = g.u32c(GRAD_EXP_MAX);
    let e = g.max(e, lo);
    let e = g.min(e, hi);
    // gs = 2^(GRAD_SCALE_EXP - (e - 127)) has biased exponent
    // 254 + GRAD_SCALE_EXP - e; 1/gs has e - GRAD_SCALE_EXP.
    let top = g.u32c(254 + GRAD_SCALE_EXP);
    let gs_e = g.sub(top, e);
    let gi = g.u32c(GRAD_SCALE_EXP);
    let inv_e = g.sub(e, gi);
    let sh = g.u32c(23);
    let gs_bits = g.shl(gs_e, sh);
    let inv_bits = g.shl(inv_e, sh);
    let gs = g.bitcast(gs_bits, KirType::F32);
    let inv = g.bitcast(inv_bits, KirType::F32);
    (gs, inv)
}

/// Biased-exponent target for the backward's gradient prescale: the
/// largest `dO` element of a scaled group lands in `[64, 128)`. That leaves
/// 20 binades of normal f16 range below it for the group's smaller elements
/// and for `dS`, and 9 above it before `dS` reaches f16's maximum (where it
/// is clamped). The same target as the kernels these replace
/// (`BWD_GRAD_SCALE_EXP` in nsl-codegen's `flash_attention.rs`).
pub const GRAD_SCALE_EXP: u32 = 6;

/// Smallest biased exponent the prescale derives a scale from; keeps `gs`
/// and `1/gs` normal f32 values.
pub const GRAD_EXP_MIN: u32 = GRAD_SCALE_EXP + 1;

/// Largest biased exponent of a finite f32.
pub const GRAD_EXP_MAX: u32 = 254;

/// The f64 value of [`grad_scale_from_max_bits`] for a maximum `max`, for
/// host-side references: `(gs, 1/gs)`.
pub fn grad_scale_for_max(max: f32) -> (f64, f64) {
    let bits = max.abs().to_bits();
    let e = (bits >> 23).clamp(GRAD_EXP_MIN, GRAD_EXP_MAX) as i32;
    let p = GRAD_SCALE_EXP as i32 - (e - 127);
    (2f64.powi(p), 2f64.powi(-p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grad_scale_puts_the_maximum_in_the_target_binade() {
        for &m in &[1e-8f32, 3.7e-7, 1e-6, 2.5e-3, 1.0, 64.0, 127.9, 1e4] {
            let (gs, inv) = grad_scale_for_max(m);
            let scaled = m as f64 * gs;
            assert!((64.0..128.0).contains(&scaled), "max {m} scaled to {scaled}");
            assert_eq!(gs * inv, 1.0);
        }
        // An all-zero gradient keeps normal, mutually reciprocal scales.
        let (gs, inv) = grad_scale_for_max(0.0);
        assert!(gs.is_normal() && (gs as f32).is_normal() && (inv as f32).is_normal());
        assert_eq!(gs * inv, 1.0);
    }
}
