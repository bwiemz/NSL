// crates/nsl-kir/src/kernels/dropout.rs
//! The runtime's GPU dropout kernel from `cuda/fused_kernels.rs`, as KIR
//! (new-roadmap item 5).
//!
//! `nsl_dropout_f32(inp, out, mask, len, threshold, scale, seed)`: inverted
//! dropout with a counter-based hash, one thread per element.
//!
//! ```text
//! c  = seed + i                        (64-bit)
//! lo = u32(c);  hi = u32(c >> 32)
//! r_k(x) = x·M_k;  x ^= x >> S_k       (M, S = 0x9E3779B9/16, 0x85EBCA6B/13, 0xC2B2AE35/16)
//! g = r_3(r_2(r_1(hi)))                (the key of this 2^32-element block)
//! h = r_3(r_2(r_1(lo) + g))
//! keep = h < threshold                 (unsigned)
//! out[i]  = inp[i] · (keep ? scale : 0)
//! mask[i] = keep ? 1 : 0
//! ```
//!
//! The draw is a pure function of the 64-bit counter `seed + i` (the host
//! advances `seed` by `len` per launch), so the mask does not depend on the
//! launch shape and no two elements of a run share a counter.
//!
//! Both halves of the counter enter the hash. The kernel used to hash
//! `u32(seed + i)` alone, so the mask stream repeated exactly every 2^32
//! elements: a model that draws a few billion dropout decisions per step
//! saw the same masks again within a handful of steps. Now each block of
//! 2^32 counters is keyed by the hash of its high word, added between the
//! first and second rounds. For a counter below 2^32 the key is 0 (`r_k`
//! maps 0 to 0) and the hash is the old one, so runs that never leave the
//! first block draw the masks they always did.
//!
//! The product is a bare `mul.f32` by the selected factor, as in the hand
//! kernel: a dropped NaN input stays NaN (`NaN · 0`), and a dropped
//! infinity becomes NaN. Blocks of
//! [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK).

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirOp, KirType, VarId};

/// The entry name the runtime launches.
pub const DROPOUT_NAME: &str = "nsl_dropout_f32";

/// The hash's three multipliers, in order.
pub const DROPOUT_HASH_MULTIPLIERS: [u32; 3] = [0x9E37_79B9, 0x85EB_CA6B, 0xC2B2_AE35];
/// The xorshift after each multiply.
pub const DROPOUT_HASH_SHIFTS: [u32; 3] = [16, 13, 16];

fn u32_const(b: &mut KirBuilder, v: u32) -> VarId {
    let dst = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(dst, KirConst { ty: KirType::U32, value: ConstValue::U32(v) }));
    dst
}

fn f32_const(b: &mut KirBuilder, v: f32) -> VarId {
    let dst = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(dst, KirConst { ty: KirType::F32, value: ConstValue::F32(v) }));
    dst
}

fn u32_op2(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::U32);
    b.emit(op(dst, x, y));
    dst
}

/// One round of the hash: `x·M_k`, then `x ^= x >> S_k`.
fn hash_round(b: &mut KirBuilder, x: VarId, round: usize) -> VarId {
    let m = u32_const(b, DROPOUT_HASH_MULTIPLIERS[round]);
    let x = u32_op2(b, KirOp::Mul, x, m);
    let s = u32_const(b, DROPOUT_HASH_SHIFTS[round]);
    let shifted = u32_op2(b, KirOp::Shr, x, s);
    u32_op2(b, KirOp::Xor, x, shifted)
}

/// Build `nsl_dropout_f32` in the hand kernel's instruction order.
pub fn build_dropout() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(DROPOUT_NAME);
    let inp = b.add_param("inp", f32_ptr(), Global);
    let out = b.add_param("out", f32_ptr(), Global);
    let mask = b.add_param("mask", f32_ptr(), Global);
    let len = b.add_param("len", KirType::U64, Global);
    let threshold = b.add_param("threshold", KirType::U32, Global);
    let scale = b.add_param("scale", KirType::F32, Global);
    let seed = b.add_param("seed", KirType::U64, Global);

    let (i, _, exit) = index_and_bound(&mut b, len);
    let ctr = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Add(ctr, seed, i));
    let lo = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(lo, ctr, KirType::U32));
    let word = u32_const(&mut b, 32);
    let upper = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Shr(upper, ctr, word));
    let hi = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(hi, upper, KirType::U32));
    // The block key: all three rounds on the counter's high word.
    let mut key = hi;
    for round in 0..3 {
        key = hash_round(&mut b, key, round);
    }
    // The element hash: round 1 on the low word, the key added, rounds 2, 3.
    let mut h = hash_round(&mut b, lo, 0);
    h = u32_op2(&mut b, KirOp::Add, h, key);
    h = hash_round(&mut b, h, 1);
    h = hash_round(&mut b, h, 2);
    let keep = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(keep, h, threshold, CmpOp::Lt));

    let x = load_f32(&mut b, inp, i);
    let zero = f32_const(&mut b, 0.0);
    let factor = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Select(factor, keep, scale, zero));
    let y = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Mul(y, x, factor));
    let one = f32_const(&mut b, 1.0);
    let zero = f32_const(&mut b, 0.0);
    let m = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Select(m, keep, one, zero));
    store_f32(&mut b, out, i, y);
    store_f32(&mut b, mask, i, m);
    finish(b, exit)
}

/// [`build_dropout`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn dropout_ptx() -> Vec<u8> {
    verified_ptx(build_dropout())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropout_verifies_and_spells_its_hash() {
        let ptx = String::from_utf8(dropout_ptx()).unwrap();
        assert!(ptx.contains(".visible .entry nsl_dropout_f32("), "{ptx}");
        assert_eq!(ptx.matches("mul.lo.u32 ").count(), 6 + 1, "six hash multiplies (key and element) and the global index: {ptx}");
        assert_eq!(ptx.matches("xor.b32 ").count(), 6, "{ptx}");
        assert_eq!(ptx.matches("cvt.u32.u64 ").count(), 2, "both halves of the counter are read: {ptx}");
        assert_eq!(ptx.matches("shr.u64 ").count(), 1, "the high word: {ptx}");
        assert_eq!(ptx.matches("setp.lt.u32 ").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("selp.f32 ").count(), 2, "{ptx}");
        assert_eq!(ptx.matches("mul.f32 ").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("st.global.f32 ").count(), 2, "{ptx}");
        for m in DROPOUT_HASH_MULTIPLIERS {
            assert_eq!(ptx.matches(&format!(", {m};")).count(), 2, "{m:#x}: {ptx}");
        }
    }
}
