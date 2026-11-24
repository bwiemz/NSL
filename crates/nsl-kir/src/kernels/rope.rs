// crates/nsl-kir/src/kernels/rope.rs
//! Fused rotary position embedding (the `rope` builtin) and its adjoint.
//!
//! For `x` of shape `[b, h, s, d]` read through ANY strides (the head view
//! `q.reshape(..).transpose(1, 2)` is not contiguous) and the `[s, d]` f32
//! tables `cos`, `sin` (rows 0..s of the cached tables):
//!
//! ```text
//! forward   y[j]  = x[j]·cos[j] + rot(x)[j]·sin[j]      rot(x) = [-x2, x1]
//! adjoint   x̄[j] = ȳ[j]·cos[j] + rotᵀ(ȳ∘sin)[j]       rotᵀ(z) = [z2, -z1]
//! ```
//!
//! One thread per output element of the contiguous `[b, h, s, d]` result.
//! The arithmetic is f32 and is exactly the decomposed graph's, operation for
//! operation and rounding for rounding (`mul.rn`, `mul.rn`, `add.rn`, which
//! ptxas never contracts into an `fma`; the
//! negation and the half swap are exact), so the f32 instantiation is
//! bit-identical to `x * cos_4d + rotate_half(x) * sin_4d` and to its
//! adjoint. The stored dtype is the input's and output's [`Storage`]: a bf16
//! output is rounded once, on store.

use super::elementwise::{f32_op2, index_and_bound, verified_ptx, ELEMENTWISE_BLOCK};
use super::storage::{load_as_f32, store_from_f32, variant_name, Storage};
use crate::kernel_ir::{
    AddressSpace, CmpOp, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// Which half of the pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RopeDir {
    /// `y = x·cos + rot(x)·sin`.
    Forward,
    /// `x̄ = ȳ·cos + rotᵀ(ȳ·sin)`.
    Backward,
}

impl RopeDir {
    pub const ALL: [RopeDir; 2] = [RopeDir::Forward, RopeDir::Backward];

    /// The all-f32 entry name.
    pub fn kernel_name(self) -> &'static str {
        match self {
            RopeDir::Forward => "nsl_rope_f32",
            RopeDir::Backward => "nsl_rope_backward_f32",
        }
    }
}

/// The entry name of `dir`'s instantiation.
pub fn kernel_name(dir: RopeDir, input: Storage, out: Storage) -> String {
    variant_name(dir.kernel_name(), &[input], out)
}

fn u64v(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let d = b.new_typed_var(KirType::U64);
    b.emit(op(d, x, y));
    d
}

/// Parameters, in launch order: `x` (input storage), `cos`, `sin` (f32
/// `[s, d]`), `y` (output storage, contiguous `[b, h, s, d]`), `n` = b·h·s·d,
/// `d`, `s`, `h`, and `x`'s element strides `stride_b, stride_h, stride_s,
/// stride_d` — every scalar a `u64`.
pub fn build(dir: RopeDir, input: Storage, out: Storage) -> KernelIR {
    use AddressSpace::Global;
    let f32p = KirType::Ptr(Box::new(KirType::F32), Global);
    let mut b = KirBuilder::new(&kernel_name(dir, input, out));
    let x = b.add_param("x", input.ptr(), Global);
    let cos = b.add_param("cos", f32p.clone(), Global);
    let sin = b.add_param("sin", f32p, Global);
    let y = b.add_param("y", out.ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let d = b.add_param("d", KirType::U64, Global);
    let s = b.add_param("s", KirType::U64, Global);
    let h = b.add_param("h", KirType::U64, Global);
    let st_b = b.add_param("stride_b", KirType::U64, Global);
    let st_h = b.add_param("stride_h", KirType::U64, Global);
    let st_s = b.add_param("stride_s", KirType::U64, Global);
    let st_d = b.add_param("stride_d", KirType::U64, Global);
    let (i, _, exit) = index_and_bound(&mut b, n);

    // i = ((bb·h + hh)·s + row)·d + j, decomposed in 32 bits (the launcher
    // admits fewer than 2^31 elements; 64-bit div/rem is several times
    // slower), the offsets widened back to 64 bits.
    let to = |b: &mut KirBuilder, v: VarId, ty: KirType| {
        let d = b.new_typed_var(ty.clone());
        b.emit(KirOp::Cast(d, v, ty));
        d
    };
    let (i32_, d32, s32, h32) = (to(&mut b, i, KirType::U32), to(&mut b, d, KirType::U32), to(&mut b, s, KirType::U32), to(&mut b, h, KirType::U32));
    let u32v = |b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId| {
        let r = b.new_typed_var(KirType::U32);
        b.emit(op(r, x, y));
        r
    };
    let j32 = u32v(&mut b, KirOp::Rem, i32_, d32);
    let t = u32v(&mut b, KirOp::Div, i32_, d32);
    let row32 = u32v(&mut b, KirOp::Rem, t, s32);
    let t2 = u32v(&mut b, KirOp::Div, t, s32);
    let hh32 = u32v(&mut b, KirOp::Rem, t2, h32);
    let bb32 = u32v(&mut b, KirOp::Div, t2, h32);
    let (j, row, hh, bb) = (
        to(&mut b, j32, KirType::U64),
        to(&mut b, row32, KirType::U64),
        to(&mut b, hh32, KirType::U64),
        to(&mut b, bb32, KirType::U64),
    );
    let ob = u64v(&mut b, KirOp::Mul, bb, st_b);
    let oh = u64v(&mut b, KirOp::Mul, hh, st_h);
    let os = u64v(&mut b, KirOp::Mul, row, st_s);
    let base = u64v(&mut b, KirOp::Add, ob, oh);
    let base = u64v(&mut b, KirOp::Add, base, os);
    let trow = u64v(&mut b, KirOp::Mul, row, d);
    let two = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Const(two, crate::kernel_ir::KirConst { ty: KirType::U64, value: crate::kernel_ir::ConstValue::U64(2) }));
    let half = u64v(&mut b, KirOp::Div, d, two);

    // This element and its partner across the half split.
    let xj = {
        let o = u64v(&mut b, KirOp::Mul, j, st_d);
        let o = u64v(&mut b, KirOp::Add, base, o);
        load_as_f32(&mut b, x, o, input)
    };
    let tj = u64v(&mut b, KirOp::Add, trow, j);
    let cj = load_as_f32(&mut b, cos, tj, Storage::F32);
    let first = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(first, j, half, CmpOp::Lt));
    let first_block = b.new_block();
    let second_block = b.new_block();
    b.terminate(KirTerminator::CondBranch(first, KirEdge::to(first_block), KirEdge::to(second_block)));

    for (block, in_first) in [(first_block, true), (second_block, false)] {
        b.set_block(block);
        let jp = if in_first { u64v(&mut b, KirOp::Add, j, half) } else { u64v(&mut b, KirOp::Sub, j, half) };
        let xp = {
            let o = u64v(&mut b, KirOp::Mul, jp, st_d);
            let o = u64v(&mut b, KirOp::Add, base, o);
            load_as_f32(&mut b, x, o, input)
        };
        let a = f32_op2(&mut b, KirOp::MulRn, xj, cj);
        let rot = match dir {
            RopeDir::Forward => {
                // rot(x)[j] = -x[j + d/2] in the first half, x[j - d/2] after; then · sin[j].
                let sj = load_as_f32(&mut b, sin, tj, Storage::F32);
                let r = if in_first { f32_neg(&mut b, xp) } else { xp };
                f32_op2(&mut b, KirOp::MulRn, r, sj)
            }
            RopeDir::Backward => {
                // rotᵀ(ȳ·sin)[j] = (ȳ·sin)[j + d/2] in the first half, -(ȳ·sin)[j - d/2] after.
                let tp = u64v(&mut b, KirOp::Add, trow, jp);
                let sp = load_as_f32(&mut b, sin, tp, Storage::F32);
                let z = f32_op2(&mut b, KirOp::MulRn, xp, sp);
                if in_first { z } else { f32_neg(&mut b, z) }
            }
        };
        let v = f32_op2(&mut b, KirOp::AddRn, a, rot);
        store_from_f32(&mut b, y, i, v, out);
        b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
    }

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([ELEMENTWISE_BLOCK, 1, 1]);
    b.set_launch_bounds(ELEMENTWISE_BLOCK, None);
    b.finalize()
}

fn f32_neg(b: &mut KirBuilder, v: VarId) -> VarId {
    let d = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Neg(d, v));
    d
}

/// [`build`] lowered to a NUL-terminated PTX module.
pub fn ptx(dir: RopeDir, input: Storage, out: Storage) -> Vec<u8> {
    verified_ptx(build(dir, input, out))
}

/// Rows of `d` elements each kernel thread of the dense-rows kernel
/// ([`build_rows`]) handles, in pairs: `ROWS_PAIRS` first-half elements and
/// their `d / 2`-away partners, so each operand moves in 16-byte accesses.
pub const ROWS_PAIRS: u32 = 8;

/// The entry name of `dir`'s dense-rows instantiation ([`build_rows`]).
pub fn rows_kernel_name(dir: RopeDir, input: Storage, out: Storage) -> String {
    let base = match dir {
        RopeDir::Forward => "nsl_rope_rows_f32",
        RopeDir::Backward => "nsl_rope_rows_backward_f32",
    };
    variant_name(base, &[input], out)
}

/// The dense-rows RoPE: the same function as [`build`], for an `x` whose
/// elements are one dense buffer of `d`-element rows (one `(batch, head,
/// position)` each, in any row order), written to `y` in the SAME layout.
///
/// Row `r` is at position `(r / pos_div) % s`: `pos_div = 1` for the
/// contiguous `[b, h, s, d]` layout and `pos_div = h` for the attention
/// layout `[b, s, h, d]` (the head view `q.reshape(b, s, h, d).transpose(1,
/// 2)` of a projection is that buffer itself). So the head view is read and
/// written in place, row by row, and the output keeps it: no transpose.
///
/// Each thread handles [`ROWS_PAIRS`] consecutive first-half elements of one
/// row and their partners `d / 2` on: every operand (x, the two table rows,
/// y) moves in 16-byte vector accesses (`ld/st.global.v4.b32` for bf16,
/// `.v4.f32` for f32), against one 2- or 4-byte access per element in
/// [`build`]. The arithmetic is [`build`]'s, operation for operation
/// (`mul.rn`, `mul.rn`, `add.rn`; the negation exact): the results are the
/// same bits. bf16 is widened exactly (`x << 16`) and rounded once to nearest
/// even on store (`cvt.rn.bf16x2.f32`).
///
/// Parameters: `x`, `cos`, `sin` (f32 `[s, d]`), `y`, `chunks` = elements /
/// (2 · ROWS_PAIRS), `d`, `s`, `pos_div`, every scalar a `u32` (the launcher
/// admits fewer than 2^31 elements). `d` must be a multiple of `4 ·
/// ROWS_PAIRS` and every pointer 16-byte aligned.
pub fn build_rows(dir: RopeDir, input: Storage, out: Storage) -> KernelIR {
    use AddressSpace::Global;
    let u32p = KirType::Ptr(Box::new(KirType::U32), Global);
    let f32p = KirType::Ptr(Box::new(KirType::F32), Global);
    // bf16 operands move as pairs: read as u32 words, written as packed
    // bf16x2 words.
    let pair_ptr = KirType::Ptr(Box::new(crate::kernel_ir::MmaOperandTy::Bf16.fragment_ty()), Global);
    let mut b = KirBuilder::new(&rows_kernel_name(dir, input, out));
    let x = b.add_param("x", if input == Storage::Bf16 { u32p } else { f32p.clone() }, Global);
    let cos = b.add_param("cos", f32p.clone(), Global);
    let sin = b.add_param("sin", f32p.clone(), Global);
    let y = b.add_param("y", if out == Storage::Bf16 { pair_ptr } else { f32p.clone() }, Global);
    let chunks = b.add_param("chunks", KirType::U32, Global);
    let d = b.add_param("d", KirType::U32, Global);
    let s = b.add_param("s", KirType::U32, Global);
    let pos_div = b.add_param("pos_div", KirType::U32, Global);

    let entry = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let c = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(c, 0));
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, c, chunks, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));
    b.set_block(body);

    let u32c = |b: &mut KirBuilder, v: u32| {
        let k = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Const(k, crate::kernel_ir::KirConst { ty: KirType::U32, value: crate::kernel_ir::ConstValue::U32(v) }));
        k
    };
    let u32op = |b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId| {
        let r = b.new_typed_var(KirType::U32);
        b.emit(op(r, x, y));
        r
    };
    // chunk c -> row, first pair p0 = 8·(c % (d/16)), position.
    let one = u32c(&mut b, 1);
    let half = u32op(&mut b, KirOp::Shr, d, one);
    let per_row = {
        let sh = u32c(&mut b, ROWS_PAIRS.trailing_zeros());
        u32op(&mut b, KirOp::Shr, half, sh)
    };
    let row = u32op(&mut b, KirOp::Div, c, per_row);
    let k = u32op(&mut b, KirOp::Rem, c, per_row);
    let p0 = {
        let sh = u32c(&mut b, ROWS_PAIRS.trailing_zeros());
        u32op(&mut b, KirOp::Shl, k, sh)
    };
    let pos = {
        let t = u32op(&mut b, KirOp::Div, row, pos_div);
        u32op(&mut b, KirOp::Rem, t, s)
    };
    let lo = {
        let r = u32op(&mut b, KirOp::Mul, row, d);
        u32op(&mut b, KirOp::Add, r, p0)
    };
    let hi = u32op(&mut b, KirOp::Add, lo, half);
    let t_lo = {
        let r = u32op(&mut b, KirOp::Mul, pos, d);
        u32op(&mut b, KirOp::Add, r, p0)
    };
    let t_hi = u32op(&mut b, KirOp::Add, t_lo, half);

    // Every load first: x's two runs, then the four table runs.
    let x_lo = load_run(&mut b, x, lo, input);
    let x_hi = load_run(&mut b, x, hi, input);
    let c_lo = load_run(&mut b, cos, t_lo, Storage::F32);
    let c_hi = load_run(&mut b, cos, t_hi, Storage::F32);
    let s_lo = load_run(&mut b, sin, t_lo, Storage::F32);
    let s_hi = load_run(&mut b, sin, t_hi, Storage::F32);

    let mut y_lo = Vec::with_capacity(ROWS_PAIRS as usize);
    let mut y_hi = Vec::with_capacity(ROWS_PAIRS as usize);
    for i in 0..ROWS_PAIRS as usize {
        let (a_lo, a_hi) = (f32_op2(&mut b, KirOp::MulRn, x_lo[i], c_lo[i]), f32_op2(&mut b, KirOp::MulRn, x_hi[i], c_hi[i]));
        let (rot_lo, rot_hi) = match dir {
            // [`build`]: y[j] = x[j]·cos[j] + (-x[j+h])·sin[j] (j < h);
            // y[j] = x[j]·cos[j] + x[j-h]·sin[j] (j >= h).
            RopeDir::Forward => {
                let nx = f32_neg(&mut b, x_hi[i]);
                (f32_op2(&mut b, KirOp::MulRn, nx, s_lo[i]), f32_op2(&mut b, KirOp::MulRn, x_lo[i], s_hi[i]))
            }
            // [`build`]: x̄[j] = ȳ[j]·cos[j] + ȳ[j+h]·sin[j+h] (j < h);
            // x̄[j] = ȳ[j]·cos[j] + -(ȳ[j-h]·sin[j-h]) (j >= h).
            RopeDir::Backward => {
                let z_lo = f32_op2(&mut b, KirOp::MulRn, x_hi[i], s_hi[i]);
                let z_hi = f32_op2(&mut b, KirOp::MulRn, x_lo[i], s_lo[i]);
                (z_lo, f32_neg(&mut b, z_hi))
            }
        };
        y_lo.push(f32_op2(&mut b, KirOp::AddRn, a_lo, rot_lo));
        y_hi.push(f32_op2(&mut b, KirOp::AddRn, a_hi, rot_hi));
    }
    store_run(&mut b, y, lo, out, &y_lo);
    store_run(&mut b, y, hi, out, &y_hi);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([ELEMENTWISE_BLOCK, 1, 1]);
    b.set_launch_bounds(ELEMENTWISE_BLOCK, None);
    b.finalize()
}

/// `ROWS_PAIRS` consecutive elements from element `at` of `base` (stored as
/// `st`) as f32: two `.v4.f32` loads, or for bf16 one `.v4.b32` load of
/// packed pairs widened exactly (the low half of a word is the first
/// element: `lo << 16`, `hi & 0xffff0000`).
fn load_run(b: &mut KirBuilder, base: VarId, at: VarId, st: Storage) -> Vec<VarId> {
    let n = ROWS_PAIRS as usize;
    match st {
        Storage::F32 => (0..n / 4)
            .flat_map(|g| {
                let off = add_u32c(b, at, 4 * g as u32);
                let p = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
                b.emit(KirOp::PtrOffset(p, base, off));
                let dsts: Vec<VarId> = (0..4).map(|_| b.new_typed_var(KirType::F32)).collect();
                b.emit(KirOp::LoadVec { dsts: dsts.clone(), ptr: p, space: AddressSpace::Global });
                dsts
            })
            .collect(),
        Storage::Bf16 => {
            // `at` counts bf16 elements; the word pointer counts pairs.
            let w_at = word_index(b, at);
            let p = b.new_typed_var(KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global));
            b.emit(KirOp::PtrOffset(p, base, w_at));
            let words: Vec<VarId> = (0..n / 2).map(|_| b.new_typed_var(KirType::U32)).collect();
            b.emit(KirOp::LoadVec { dsts: words.clone(), ptr: p, space: AddressSpace::Global });
            let sixteen = {
                let k = b.new_typed_var(KirType::U32);
                b.emit(KirOp::Const(k, crate::kernel_ir::KirConst { ty: KirType::U32, value: crate::kernel_ir::ConstValue::U32(16) }));
                k
            };
            let mask = {
                let k = b.new_typed_var(KirType::U32);
                b.emit(KirOp::Const(k, crate::kernel_ir::KirConst { ty: KirType::U32, value: crate::kernel_ir::ConstValue::U32(0xFFFF_0000) }));
                k
            };
            words
                .into_iter()
                .flat_map(|w| {
                    let lo_bits = b.new_typed_var(KirType::U32);
                    b.emit(KirOp::Shl(lo_bits, w, sixteen));
                    let hi_bits = b.new_typed_var(KirType::U32);
                    b.emit(KirOp::And(hi_bits, w, mask));
                    [lo_bits, hi_bits].map(|bits| {
                        let f = b.new_typed_var(KirType::F32);
                        b.emit(KirOp::Bitcast(f, bits));
                        f
                    })
                })
                .collect()
        }
    }
}

/// Store `vals` (f32) at element `at` of `base` as `st`: two `.v4.f32`
/// stores, or for bf16 pairs rounded to nearest even and packed
/// (`cvt.rn.bf16x2.f32`) into one `.v4.b32` store.
fn store_run(b: &mut KirBuilder, base: VarId, at: VarId, st: Storage, vals: &[VarId]) {
    match st {
        Storage::F32 => {
            for (g, group) in vals.chunks(4).enumerate() {
                let off = add_u32c(b, at, 4 * g as u32);
                let p = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
                b.emit(KirOp::PtrOffset(p, base, off));
                b.emit(KirOp::StoreVec { ptr: p, vals: group.to_vec(), space: AddressSpace::Global });
            }
        }
        Storage::Bf16 => {
            let pair_ty = crate::kernel_ir::MmaOperandTy::Bf16.fragment_ty();
            let words: Vec<VarId> = vals
                .chunks(2)
                .map(|pair| {
                    let packed = b.new_typed_var(pair_ty.clone());
                    b.emit(KirOp::CvtPack { dst: packed, hi: pair[1], lo: pair[0], ty: crate::kernel_ir::MmaOperandTy::Bf16 });
                    packed
                })
                .collect();
            let w_at = word_index(b, at);
            let p = b.new_typed_var(KirType::Ptr(Box::new(pair_ty), AddressSpace::Global));
            b.emit(KirOp::PtrOffset(p, base, w_at));
            b.emit(KirOp::StoreVec { ptr: p, vals: words, space: AddressSpace::Global });
        }
    }
}

/// The packed-pair word holding bf16 element `at` (even).
fn word_index(b: &mut KirBuilder, at: VarId) -> VarId {
    let k = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(k, crate::kernel_ir::KirConst { ty: KirType::U32, value: crate::kernel_ir::ConstValue::U32(1) }));
    let w = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Shr(w, at, k));
    w
}

fn add_u32c(b: &mut KirBuilder, x: VarId, c: u32) -> VarId {
    if c == 0 {
        return x;
    }
    let k = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(k, crate::kernel_ir::KirConst { ty: KirType::U32, value: crate::kernel_ir::ConstValue::U32(c) }));
    let r = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Add(r, x, k));
    r
}

/// [`build_rows`] lowered to a NUL-terminated PTX module.
pub fn rows_ptx(dir: RopeDir, input: Storage, out: Storage) -> Vec<u8> {
    verified_ptx(build_rows(dir, input, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_instantiation_verifies_and_rounds_only_on_a_bf16_store() {
        for dir in RopeDir::ALL {
            for input in Storage::ALL {
                for out in Storage::ALL {
                    let text = String::from_utf8(ptx(dir, input, out)).unwrap();
                    let name = kernel_name(dir, input, out);
                    assert!(text.contains(&format!(".visible .entry {name}(")), "{name}");
                    assert_eq!(text.contains("cvt.rn.bf16.f32"), out == Storage::Bf16, "{name}: rounding");
                    // Explicitly rounded: ptxas may not contract into an fma,
                    // which would break bit identity with the decomposed graph.
                    assert!(text.contains("mul.rn.f32") && text.contains("add.rn.f32"), "{name}: unrounded arithmetic");
                    assert!(!text.contains("fma."), "{name}: an fma");
                }
            }
        }
    }

    /// The dense-rows kernels: every operand moves in 16-byte vector
    /// accesses, the arithmetic is explicitly rounded (no fma), and a bf16
    /// output is rounded only by the packed store conversion.
    #[test]
    fn every_rows_instantiation_verifies_and_moves_16_byte_vectors() {
        for dir in RopeDir::ALL {
            for input in Storage::ALL {
                for out in Storage::ALL {
                    let text = String::from_utf8(rows_ptx(dir, input, out)).unwrap();
                    let name = rows_kernel_name(dir, input, out);
                    assert!(text.contains(&format!(".visible .entry {name}(")), "{name}");
                    assert_eq!(text.contains("cvt.rn.bf16x2.f32"), out == Storage::Bf16, "{name}: rounding");
                    assert!(!text.contains("cvt.rn.bf16.f32"), "{name}: a scalar bf16 store");
                    assert!(text.contains("mul.rn.f32") && text.contains("add.rn.f32"), "{name}: unrounded arithmetic");
                    assert!(!text.contains("fma."), "{name}: an fma");
                    // Loads: x (2 runs) and the 4 table runs; f32 runs are two v4 accesses.
                    let x_loads = if input == Storage::Bf16 { 2 } else { 4 };
                    let lds = text.matches("ld.global.v4").count();
                    assert_eq!(lds, x_loads + 8, "{name}: vector loads\n{text}");
                    let sts = text.matches("st.global.v4").count();
                    assert_eq!(sts, if out == Storage::Bf16 { 2 } else { 4 }, "{name}: vector stores");
                    assert!(!text.contains("ld.global.f32") && !text.contains("ld.global.b16"), "{name}: a scalar load");
                }
            }
        }
    }
}
