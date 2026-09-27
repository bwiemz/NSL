// crates/nsl-kir/src/kernels/dequant.rs
//! The runtime's integer dequantization kernels from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5).
//!
//! - `nsl_dequant_int8_per_head_f32(inp, out, scales, n, head_stride)`:
//!   `out[i] = f32(inp[i]) · scales[i / head_stride]`.
//! - `nsl_dequant_int8_per_token_f32(inp, out, scales, n, head_stride,
//!   head_dim)`: the same with the token's scale,
//!   `scales[(i % head_stride) / head_dim]`, for the `[heads, block,
//!   head_dim]` KV layout.
//! - `nsl_dequant_int4_per_group_f32(inp, out, scales, zero_points, n,
//!   group_size)`: two unsigned nibbles per byte, low nibble first.
//!   `out[i] = fma(nibble(i), scales[g], zero_points[g])` with
//!   `g = i / group_size`: one rounding, as the hand kernel's `fma.rn`.
//!
//! `inp` is `s8` for the int8 pair (a `.s8` load, sign-extended, then
//! `cvt.rn.f32.s8`, exact) and packed `u8` for int4 (a `.u8` load and
//! [`KirType::U8`]). The int8 product is a bare `mul.f32`: it has nothing to
//! contract with. Each kernel runs one thread per output element,
//! `ceil(n / 256)` blocks of 256, in the hand kernel's order: the index in
//! 32 bits widened, a 64-bit bound, and the same loads.

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// The int8 pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8Scale {
    /// `nsl_dequant_int8_per_head_f32`: one scale per `head_stride`
    /// elements.
    PerHead,
    /// `nsl_dequant_int8_per_token_f32`: one scale per `head_dim` elements
    /// within each head.
    PerToken,
}

impl Int8Scale {
    pub const ALL: [Int8Scale; 2] = [Int8Scale::PerHead, Int8Scale::PerToken];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            Int8Scale::PerHead => "nsl_dequant_int8_per_head_f32",
            Int8Scale::PerToken => "nsl_dequant_int8_per_token_f32",
        }
    }
}

/// `nsl_dequant_int4_per_group_f32`.
pub const INT4_PER_GROUP_NAME: &str = "nsl_dequant_int4_per_group_f32";

fn u64_op2(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::U64);
    b.emit(op(dst, x, y));
    dst
}

fn int_const(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = b.new_typed_var(ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

/// `base[i]` for an array of the one-byte integer `elem`.
fn load_byte(b: &mut KirBuilder, base: VarId, i: VarId, elem: KirType) -> VarId {
    let addr = b.new_typed_var(KirType::Ptr(Box::new(elem.clone()), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = b.new_typed_var(elem);
    b.emit(KirOp::Load(v, addr, AddressSpace::Global));
    v
}

/// Build the int8 dequantization kernel `scale` names.
///
/// ```text
/// entry:  i (u32, widened); if i >= n { exit }
/// body:   s = scales[i / head_stride]                 (per head)
///         s = scales[(i % head_stride) / head_dim]    (per token)
///         out[i] = f32(inp[i]) · s
/// ```
pub fn build_int8(scale: Int8Scale) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(scale.kernel_name());
    let inp = b.add_param("inp", KirType::Ptr(Box::new(KirType::I8), Global), Global);
    let out = b.add_param("out", f32_ptr(), Global);
    let scales = b.add_param("scales", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let head_stride = b.add_param("head_stride", KirType::U64, Global);
    let head_dim = match scale {
        Int8Scale::PerHead => None,
        Int8Scale::PerToken => Some(b.add_param("head_dim", KirType::U64, Global)),
    };

    let (i, _, exit) = index_and_bound(&mut b, n);
    let slot = match head_dim {
        None => u64_op2(&mut b, KirOp::Div, i, head_stride),
        Some(head_dim) => {
            let within = u64_op2(&mut b, KirOp::Rem, i, head_stride);
            u64_op2(&mut b, KirOp::Div, within, head_dim)
        }
    };
    let s = load_f32(&mut b, scales, slot);
    let q = load_byte(&mut b, inp, i, KirType::I8);
    let x = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Cast(x, q, KirType::F32));
    let y = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Mul(y, x, s));
    store_f32(&mut b, out, i, y);
    finish(b, exit)
}

/// [`build_int8`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn int8_ptx(scale: Int8Scale) -> Vec<u8> {
    verified_ptx(build_int8(scale))
}

/// Build `nsl_dequant_int4_per_group_f32`.
///
/// ```text
/// entry:  i (u32, widened); if i >= n { exit }
/// body:   w = u32(inp[i >> 1]); if i & 1 != 0 { br high } else { br low }
/// low:    br apply(w & 15)
/// high:   br apply((w >> 4) & 15)
/// apply(q): g = i / group_size
///         out[i] = fma(f32(q), scales[g], zero_points[g])
/// ```
pub fn build_int4_per_group() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(INT4_PER_GROUP_NAME);
    let inp = b.add_param("inp", KirType::Ptr(Box::new(KirType::U8), Global), Global);
    let out = b.add_param("out", f32_ptr(), Global);
    let scales = b.add_param("scales", f32_ptr(), Global);
    let zero_points = b.add_param("zero_points", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let group_size = b.add_param("group_size", KirType::U64, Global);

    let (i, _, exit) = index_and_bound(&mut b, n);
    let one32 = int_const(&mut b, KirType::U32, ConstValue::U32(1));
    let byte_idx = u64_op2(&mut b, KirOp::Shr, i, one32);
    let byte = load_byte(&mut b, inp, byte_idx, KirType::U8);
    let one = int_const(&mut b, KirType::U64, ConstValue::U64(1));
    let parity = u64_op2(&mut b, KirOp::And, i, one);
    let zero = int_const(&mut b, KirType::U64, ConstValue::U64(0));
    let odd = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(odd, parity, zero, CmpOp::Ne));
    let w = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(w, byte, KirType::U32));

    let low = b.new_block();
    let high = b.new_block();
    let apply = b.new_block();
    let nibble = b.add_block_param(apply, KirType::U32);
    b.terminate(KirTerminator::CondBranch(odd, KirEdge::to(high), KirEdge::to(low)));

    let fifteen = |b: &mut KirBuilder| int_const(b, KirType::U32, ConstValue::U32(15));
    let and32 = |b: &mut KirBuilder, x: VarId, y: VarId| {
        let dst = b.new_typed_var(KirType::U32);
        b.emit(KirOp::And(dst, x, y));
        dst
    };

    b.set_block(low);
    let mask = fifteen(&mut b);
    let q = and32(&mut b, w, mask);
    b.terminate(KirTerminator::Branch(KirEdge::with(apply, vec![q])));

    b.set_block(high);
    let four = int_const(&mut b, KirType::U32, ConstValue::U32(4));
    let hi = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Shr(hi, w, four));
    let mask = fifteen(&mut b);
    let q = and32(&mut b, hi, mask);
    b.terminate(KirTerminator::Branch(KirEdge::with(apply, vec![q])));

    b.set_block(apply);
    let x = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Cast(x, nibble, KirType::F32));
    let g = u64_op2(&mut b, KirOp::Div, i, group_size);
    let s = load_f32(&mut b, scales, g);
    let z = load_f32(&mut b, zero_points, g);
    let y = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Fma(y, x, s, z));
    store_f32(&mut b, out, i, y);
    finish(b, exit)
}

/// [`build_int4_per_group`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn int4_per_group_ptx() -> Vec<u8> {
    verified_ptx(build_int4_per_group())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dequant_kernels_verify_and_spell_their_loads() {
        for scale in Int8Scale::ALL {
            let ptx = String::from_utf8(int8_ptx(scale)).unwrap();
            assert!(ptx.contains(&format!(".visible .entry {}(", scale.kernel_name())), "{ptx}");
            assert_eq!(ptx.matches("ld.global.s8 ").count(), 1, "{ptx}");
            assert_eq!(ptx.matches("cvt.rn.f32.s8 ").count(), 1, "{ptx}");
            assert_eq!(ptx.matches("mul.f32 ").count(), 1, "{ptx}");
            assert_eq!(ptx.matches("div.u64 ").count(), 1, "{ptx}");
            assert_eq!(ptx.matches("rem.u64 ").count(), usize::from(scale == Int8Scale::PerToken), "{ptx}");
        }
        let ptx = String::from_utf8(int4_per_group_ptx()).unwrap();
        assert!(ptx.contains(".visible .entry nsl_dequant_int4_per_group_f32("), "{ptx}");
        for (form, count) in [("ld.global.u8 ", 1), ("cvt.u32.u8 ", 1), ("cvt.rn.f32.u32 ", 1), ("fma.rn.f32 ", 1), ("div.u64 ", 1)] {
            assert_eq!(ptx.matches(form).count(), count, "{form}\n{ptx}");
        }
        assert!(!ptx.contains("mul.f32") && !ptx.contains("add.f32"), "one rounding: {ptx}");
    }
}
