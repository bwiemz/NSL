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

    // i = ((bb·h + hh)·s + row)·d + j
    let j = u64v(&mut b, KirOp::Rem, i, d);
    let t = u64v(&mut b, KirOp::Div, i, d);
    let row = u64v(&mut b, KirOp::Rem, t, s);
    let t2 = u64v(&mut b, KirOp::Div, t, s);
    let hh = u64v(&mut b, KirOp::Rem, t2, h);
    let bb = u64v(&mut b, KirOp::Div, t2, h);
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
}
