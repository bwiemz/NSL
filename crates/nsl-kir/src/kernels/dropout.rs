// crates/nsl-kir/src/kernels/dropout.rs
//! The runtime's GPU dropout kernel from `cuda/fused_kernels.rs`, as KIR
//! (new-roadmap item 5).
//!
//! `nsl_dropout_f32(inp, out, mask, len, threshold, scale, seed, seed_ptr)`:
//! inverted dropout with a counter-based hash, one thread per element.
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
//! # Where the counter base comes from
//!
//! The two forwards take the base by value (`seed`) when `seed_ptr` is null,
//! and otherwise read it from the two u32 words at `seed_ptr`, low word first.
//! The runtime passes a pointer only inside a `--cuda-graphs` region. There
//! a by-value base changes every step, so the region's recorded argument
//! bytes never repeat and the region never captures. Instead the base goes
//! through a device word that a hooked upload writes, and the graph's memcpy
//! node refreshes it at each replay. Outside a region the base stays by value,
//! so there is no upload and no host stall. Every thread takes the same branch.
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
//!
//! # Regenerating the mask instead of storing it
//!
//! Because the draw is a pure function of the counter, the backward does not
//! need the f32 mask: it needs the counter base. Two more kernels use that
//! (the source-AD default; `NSL_DROPOUT_REGEN=0` keeps the mask):
//!
//! * [`DROPOUT_KEY_NAME`] is `nsl_dropout_f32` with the mask store replaced by
//!   a 32-byte key that the thread with global index 0 writes
//!   ([`DropoutKeyWord`]: magic, the counter base, the threshold, the scale's
//!   bits, the length). Both kernels come from one builder, so the
//!   instructions that compute `out` are the same code.
//! * [`DROPOUT_BWD_NAME`] reads the base from the key, draws each keep bit
//!   with the same hash, and writes `dx = (dy · m) · scale` (`m` = 1 or 0,
//!   two `mul.rn.f32`): the operations, in the order, of the stored-mask
//!   backward (`fused_ew` `mul(i0,i1);mul(p0,imm)`, or `nsl_mul_f32` twice).
//!   Every element is therefore bit-identical to that backward's. A key whose
//!   magic, threshold, scale or length disagree with the launch's makes the
//!   whole gradient NaN: a key read in the wrong place must not train.
//! * [`DROPOUT_APPLY_NAME`] recomputes the forward's output from its input
//!   and its key (the forward's instructions, the base read from the key), so
//!   checkpointing can replay a dropout instead of saving its output. Same
//!   key check, same poison.

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp,
    KirTerminator, KirType, VarId,
};

/// The entry name the runtime launches.
pub const DROPOUT_NAME: &str = "nsl_dropout_f32";
/// The forward that writes a key instead of a mask.
pub const DROPOUT_KEY_NAME: &str = "nsl_dropout_key_f32";
/// The backward that regenerates the mask from a key.
pub const DROPOUT_BWD_NAME: &str = "nsl_dropout_bwd_f32";
/// The forward's output again, from its input and its key (a recompute).
pub const DROPOUT_APPLY_NAME: &str = "nsl_dropout_apply_f32";

/// The first word of every key: the bytes `DROK` read as a little-endian
/// u32. Neither a zero-filled buffer (a shape stub) nor an f32 mask (words
/// 0 and 0x3F800000) can hold it.
pub const DROPOUT_KEY_MAGIC: u32 = 0x4B4F_5244;
/// Words (u32) in a key; the runtime allocates `4 *` this many bytes.
pub const DROPOUT_KEY_WORDS: usize = 8;

/// The key's layout, one u32 per word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum DropoutKeyWord {
    /// [`DROPOUT_KEY_MAGIC`].
    Magic = 0,
    /// Low and high words of the launch's counter base (`seed`).
    SeedLo = 1,
    SeedHi = 2,
    /// The keep threshold the forward compared against.
    Threshold = 3,
    /// The f32 bits of the forward's scale `1/(1-p)`.
    ScaleBits = 4,
    /// Low and high words of the element count.
    LenLo = 5,
    LenHi = 6,
    /// Zero.
    Reserved = 7,
}

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

/// Where a forward launch leaves what its backward needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Saved {
    /// `nsl_dropout_f32`: an f32 mask, one element per element.
    Mask,
    /// `nsl_dropout_key_f32`: a [`DROPOUT_KEY_WORDS`]-word key.
    Key,
    /// `nsl_dropout_apply_f32`: nothing saved; the base is read from a key.
    Replay,
}

/// `keep = H(seed + i) < threshold`, the kernel's draw for element `i`.
fn emit_keep(b: &mut KirBuilder, seed: VarId, i: VarId, threshold: VarId) -> VarId {
    let ctr = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Add(ctr, seed, i));
    let lo = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(lo, ctr, KirType::U32));
    let word = u32_const(b, 32);
    let upper = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Shr(upper, ctr, word));
    let hi = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(hi, upper, KirType::U32));
    // The block key: all three rounds on the counter's high word.
    let mut key = hi;
    for round in 0..3 {
        key = hash_round(b, key, round);
    }
    // The element hash: round 1 on the low word, the key added, rounds 2, 3.
    let mut h = hash_round(b, lo, 0);
    h = u32_op2(b, KirOp::Add, h, key);
    h = hash_round(b, h, 1);
    h = hash_round(b, h, 2);
    let keep = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(keep, h, threshold, CmpOp::Lt));
    keep
}

fn u32_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global)
}

fn u64_const(b: &mut KirBuilder, v: u64) -> VarId {
    let dst = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Const(
        dst,
        KirConst {
            ty: KirType::U64,
            value: ConstValue::U64(v),
        },
    ));
    dst
}

fn to_u32(b: &mut KirBuilder, x: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(dst, x, KirType::U32));
    dst
}

/// The high 32 bits of a u64.
fn high_word(b: &mut KirBuilder, x: VarId) -> VarId {
    let word = u32_const(b, 32);
    let upper = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Shr(upper, x, word));
    to_u32(b, upper)
}

fn key_addr(b: &mut KirBuilder, key: VarId, w: DropoutKeyWord) -> VarId {
    let off = u64_const(b, w as u64);
    let addr = b.new_typed_var(u32_ptr());
    b.emit(KirOp::PtrOffset(addr, key, off));
    addr
}

/// `hi << 32 | lo`.
fn join_u64(b: &mut KirBuilder, lo: VarId, hi: VarId) -> VarId {
    let lo64 = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(lo64, lo, KirType::U64));
    let hi64 = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(hi64, hi, KirType::U64));
    let shift = u32_const(b, 32);
    let hi_shifted = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Shl(hi_shifted, hi64, shift));
    let v = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Or(v, hi_shifted, lo64));
    v
}

/// The forward's counter base: `seed` when `seed_ptr` is null, otherwise the
/// two u32 words at `seed_ptr` (low first). See the module's "Where the counter
/// base comes from". Branches from the current block and leaves the builder in
/// the join block, whose parameter it returns.
fn resolve_seed(b: &mut KirBuilder, seed: VarId, seed_ptr: VarId) -> VarId {
    use AddressSpace::Global;
    let load = b.new_block();
    let join = b.new_block();
    let base = b.add_block_param(join, KirType::U64);
    let null = b.new_typed_var(u32_ptr());
    b.emit(KirOp::Const(
        null,
        KirConst {
            ty: u32_ptr(),
            value: ConstValue::U64(0),
        },
    ));
    let by_value = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(by_value, seed_ptr, null, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(
        by_value,
        KirEdge::with(join, vec![seed]),
        KirEdge::to(load),
    ));

    b.set_block(load);
    let lo = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Load(lo, seed_ptr, Global));
    let one = u64_const(b, 1);
    let hi_addr = b.new_typed_var(u32_ptr());
    b.emit(KirOp::PtrOffset(hi_addr, seed_ptr, one));
    let hi = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Load(hi, hi_addr, Global));
    let loaded = join_u64(b, lo, hi);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![loaded])));

    b.set_block(join);
    base
}

/// Entry for the key kernel: resolve the counter base, let the thread with
/// global index 0 write the key, then exit every thread past `len`. Returns
/// `(i, seed)` and `exit`, with the builder in the body block, as
/// `index_and_bound` does.
fn index_write_key_and_bound(
    b: &mut KirBuilder,
    len: VarId,
    key: VarId,
    threshold: VarId,
    scale: VarId,
    seed: VarId,
    seed_ptr: VarId,
) -> ((VarId, VarId), BlockId) {
    use AddressSpace::Global;
    let entry = b.new_block();
    let write = b.new_block();
    let check = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let i32_ = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(i32_, 0));
    let i = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(i, i32_, KirType::U64));
    // Resolved before the key is written: the key records the base used.
    let seed = resolve_seed(b, seed, seed_ptr);
    let zero = u64_const(b, 0);
    let first = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(first, i, zero, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(
        first,
        KirEdge::to(write),
        KirEdge::to(check),
    ));

    // Written before the bound check, so a launch of any length writes it.
    b.set_block(write);
    let scale_bits = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Bitcast(scale_bits, scale));
    let words = [
        (DropoutKeyWord::Magic, u32_const(b, DROPOUT_KEY_MAGIC)),
        (DropoutKeyWord::SeedLo, to_u32(b, seed)),
        (DropoutKeyWord::SeedHi, high_word(b, seed)),
        (DropoutKeyWord::Threshold, threshold),
        (DropoutKeyWord::ScaleBits, scale_bits),
        (DropoutKeyWord::LenLo, to_u32(b, len)),
        (DropoutKeyWord::LenHi, high_word(b, len)),
        (DropoutKeyWord::Reserved, u32_const(b, 0)),
    ];
    for (w, v) in words {
        let addr = key_addr(b, key, w);
        b.emit(KirOp::Store(addr, v, Global));
    }
    b.terminate(KirTerminator::Branch(KirEdge::to(check)));

    b.set_block(check);
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, i, len, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(
        past,
        KirEdge::to(exit),
        KirEdge::to(body),
    ));
    b.set_block(body);
    ((i, seed), exit)
}

/// One builder for both forwards, so `out` is computed by the same code.
/// The mask kernel keeps the hand kernel's instruction order exactly.
fn build_forward(saved: Saved) -> KernelIR {
    use AddressSpace::Global;
    let name = match saved {
        Saved::Mask => DROPOUT_NAME,
        Saved::Key => DROPOUT_KEY_NAME,
        Saved::Replay => DROPOUT_APPLY_NAME,
    };
    let mut b = KirBuilder::new(name);
    let inp = b.add_param("inp", f32_ptr(), Global);
    let out = b.add_param("out", f32_ptr(), Global);
    let sink = match saved {
        Saved::Mask => b.add_param("mask", f32_ptr(), Global),
        Saved::Key | Saved::Replay => b.add_param("key", u32_ptr(), Global),
    };
    let len = b.add_param("len", KirType::U64, Global);
    let threshold = b.add_param("threshold", KirType::U32, Global);
    let scale = b.add_param("scale", KirType::F32, Global);
    let seed_params = match saved {
        Saved::Mask | Saved::Key => Some((
            b.add_param("seed", KirType::U64, Global),
            b.add_param("seed_ptr", u32_ptr(), Global),
        )),
        Saved::Replay => None,
    };

    let ((i, seed), exit, valid) = match (saved, seed_params) {
        (Saved::Mask, Some((seed, seed_ptr))) => {
            let (i, _, exit) = index_and_bound(&mut b, len);
            let seed = resolve_seed(&mut b, seed, seed_ptr);
            ((i, seed), exit, None)
        }
        (Saved::Key, Some((seed, seed_ptr))) => {
            let (is, exit) =
                index_write_key_and_bound(&mut b, len, sink, threshold, scale, seed, seed_ptr);
            (is, exit, None)
        }
        (Saved::Replay, None) => {
            let (i, _, exit) = index_and_bound(&mut b, len);
            let (seed, valid) = load_key_and_check(&mut b, sink, len, threshold, scale);
            ((i, seed), exit, Some(valid))
        }
        _ => unreachable!("the forwards take a seed and the replay reads it from the key"),
    };
    let keep = emit_keep(&mut b, seed, i, threshold);

    let x = load_f32(&mut b, inp, i);
    let zero = f32_const(&mut b, 0.0);
    let factor = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Select(factor, keep, scale, zero));
    let y = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Mul(y, x, factor));
    match saved {
        Saved::Mask => {
            let one = f32_const(&mut b, 1.0);
            let zero = f32_const(&mut b, 0.0);
            let m = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Select(m, keep, one, zero));
            store_f32(&mut b, out, i, y);
            store_f32(&mut b, sink, i, m);
        }
        Saved::Key => store_f32(&mut b, out, i, y),
        Saved::Replay => {
            let valid = valid.expect("the replay checks its key");
            let nan = f32_const(&mut b, f32::NAN);
            let r = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Select(r, valid, y, nan));
            store_f32(&mut b, out, i, r);
        }
    }
    finish(b, exit)
}

/// Read a key's counter base and check the rest of it against the launch:
/// `(seed, valid)`, `valid` true when the magic, threshold, scale bits and
/// length all agree. Every thread reads the same 28 bytes.
fn load_key_and_check(
    b: &mut KirBuilder,
    key: VarId,
    len: VarId,
    threshold: VarId,
    scale: VarId,
) -> (VarId, VarId) {
    use AddressSpace::Global;
    let word = |b: &mut KirBuilder, w: DropoutKeyWord| {
        let addr = key_addr(b, key, w);
        let v = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Load(v, addr, Global));
        v
    };
    let magic = word(b, DropoutKeyWord::Magic);
    let seed_lo = word(b, DropoutKeyWord::SeedLo);
    let seed_hi = word(b, DropoutKeyWord::SeedHi);
    let key_threshold = word(b, DropoutKeyWord::Threshold);
    let key_scale = word(b, DropoutKeyWord::ScaleBits);
    let len_lo = word(b, DropoutKeyWord::LenLo);
    let len_hi = word(b, DropoutKeyWord::LenHi);

    let seed = join_u64(b, seed_lo, seed_hi);

    let scale_bits = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Bitcast(scale_bits, scale));
    let len_lo_want = to_u32(b, len);
    let len_hi_want = high_word(b, len);
    let magic_want = u32_const(b, DROPOUT_KEY_MAGIC);
    let mut valid = None;
    for (got, want) in [
        (magic, magic_want),
        (key_threshold, threshold),
        (key_scale, scale_bits),
        (len_lo, len_lo_want),
        (len_hi, len_hi_want),
    ] {
        let eq = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(eq, got, want, CmpOp::Eq));
        valid = Some(match valid {
            None => eq,
            Some(acc) => {
                let both = b.new_typed_var(KirType::Bool);
                b.emit(KirOp::And(both, acc, eq));
                both
            }
        });
    }
    (seed, valid.expect("five checks"))
}

/// Build `nsl_dropout_f32(inp, out, mask, len, threshold, scale, seed,
/// seed_ptr)`. The draw and the product keep the hand kernel's instruction
/// order. The counter base is resolved first (see the module docs).
pub fn build_dropout() -> KernelIR {
    build_forward(Saved::Mask)
}

/// Build `nsl_dropout_key_f32(inp, out, key, len, threshold, scale, seed,
/// seed_ptr)`: `nsl_dropout_f32`'s `out`, and the key instead of the mask.
/// The key records the base the launch actually used.
pub fn build_dropout_key() -> KernelIR {
    build_forward(Saved::Key)
}

/// Build `nsl_dropout_apply_f32(inp, out, key, len, threshold, scale)`: the
/// key kernel's `out` again, its base read from the key it wrote. A recompute
/// of the forward's output, bit for bit when `inp` is the forward's input.
pub fn build_dropout_apply() -> KernelIR {
    build_forward(Saved::Replay)
}

/// Build `nsl_dropout_bwd_f32(grad, dx, key, len, threshold, scale)`: the
/// stored-mask backward `(dy · m) · scale` with `m` drawn again from the
/// key's counter base. `threshold` and `scale` are the launch's (the host
/// derives them from `p` as the forward did); the key's copies and its magic
/// and length only have to agree with them.
pub fn build_dropout_bwd() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(DROPOUT_BWD_NAME);
    let grad = b.add_param("grad", f32_ptr(), Global);
    let dx = b.add_param("dx", f32_ptr(), Global);
    let key = b.add_param("key", u32_ptr(), Global);
    let len = b.add_param("len", KirType::U64, Global);
    let threshold = b.add_param("threshold", KirType::U32, Global);
    let scale = b.add_param("scale", KirType::F32, Global);

    let (i, _, exit) = index_and_bound(&mut b, len);
    let (seed, valid) = load_key_and_check(&mut b, key, len, threshold, scale);
    let keep = emit_keep(&mut b, seed, i, threshold);

    let g = load_f32(&mut b, grad, i);
    let one = f32_const(&mut b, 1.0);
    let zero = f32_const(&mut b, 0.0);
    let m = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Select(m, keep, one, zero));
    let masked = b.new_typed_var(KirType::F32);
    b.emit(KirOp::MulRn(masked, g, m));
    let y = b.new_typed_var(KirType::F32);
    b.emit(KirOp::MulRn(y, masked, scale));

    let nan = f32_const(&mut b, f32::NAN);
    let r = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Select(r, valid, y, nan));
    store_f32(&mut b, dx, i, r);
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

/// [`build_dropout_key`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn dropout_key_ptx() -> Vec<u8> {
    verified_ptx(build_dropout_key())
}

/// [`build_dropout_apply`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn dropout_apply_ptx() -> Vec<u8> {
    verified_ptx(build_dropout_apply())
}

/// [`build_dropout_bwd`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn dropout_bwd_ptx() -> Vec<u8> {
    verified_ptx(build_dropout_bwd())
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

    #[test]
    fn the_key_kernel_computes_out_with_the_mask_kernels_instructions() {
        let mask = String::from_utf8(dropout_ptx()).unwrap();
        let key = String::from_utf8(dropout_key_ptx()).unwrap();
        assert!(
            key.contains(".visible .entry nsl_dropout_key_f32("),
            "{key}"
        );
        // The draw and the product are the same instructions in both: the
        // hash (6 multiplies + the index), one compare, the factor select
        // and one multiply. So is the base prologue: one null test and the
        // two words read through `seed_ptr`. The key kernel has no mask
        // select and stores `out` once as f32 and the key as eight u32 words.
        for (form, in_mask, in_key) in [
            ("mul.lo.u32 ", 7, 7),
            ("xor.b32 ", 6, 6),
            ("setp.lt.u32 ", 1, 1),
            ("selp.f32 ", 2, 1),
            ("mul.f32 ", 1, 1),
            ("st.global.f32 ", 2, 1),
            ("st.global.u32 ", 0, DROPOUT_KEY_WORDS),
            ("ld.global.u32 ", 2, 2),
            // The null test; the key kernel's second is `i == 0`.
            ("setp.eq.u64 ", 1, 2),
        ] {
            assert_eq!(
                mask.matches(form).count(),
                in_mask,
                "mask kernel {form}: {mask}"
            );
            assert_eq!(
                key.matches(form).count(),
                in_key,
                "key kernel {form}: {key}"
            );
        }
        for never in ["fma.", "mul.rn.f32", "add.f32"] {
            assert!(!key.contains(never), "{never}: {key}");
        }
        // The magic is written, once.
        assert_eq!(
            key.matches(&format!(", {DROPOUT_KEY_MAGIC};")).count(),
            1,
            "{key}"
        );
    }

    #[test]
    fn the_backward_regenerates_with_the_same_hash_and_two_rounded_multiplies() {
        let ptx = String::from_utf8(dropout_bwd_ptx()).unwrap();
        assert!(
            ptx.contains(".visible .entry nsl_dropout_bwd_f32("),
            "{ptx}"
        );
        assert_eq!(
            ptx.matches("mul.lo.u32 ").count(),
            7,
            "the hash and the index: {ptx}"
        );
        assert_eq!(ptx.matches("xor.b32 ").count(), 6, "{ptx}");
        assert_eq!(ptx.matches("setp.lt.u32 ").count(), 1, "{ptx}");
        // dx = (dy * m) * scale, both rounded to nearest; no fused multiply-add.
        assert_eq!(ptx.matches("mul.rn.f32 ").count(), 2, "{ptx}");
        assert!(!ptx.contains("fma.") && !ptx.contains("mul.f32 "), "{ptx}");
        // Seven key words read, the gradient read, dx written; the key and the
        // gradient are never written.
        assert_eq!(ptx.matches("ld.global.u32 ").count(), 7, "{ptx}");
        assert_eq!(ptx.matches("ld.global.f32 ").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("st.global.").count(), 1, "{ptx}");
        // Five equality checks on the key.
        assert_eq!(ptx.matches("setp.eq.u32 ").count(), 5, "{ptx}");
        for m in DROPOUT_HASH_MULTIPLIERS {
            assert_eq!(ptx.matches(&format!(", {m};")).count(), 2, "{m:#x}: {ptx}");
        }
    }

    #[test]
    fn the_replay_draws_from_the_key_and_multiplies_as_the_forward() {
        let fwd = String::from_utf8(dropout_key_ptx()).unwrap();
        let ptx = String::from_utf8(dropout_apply_ptx()).unwrap();
        assert!(
            ptx.contains(".visible .entry nsl_dropout_apply_f32("),
            "{ptx}"
        );
        // The forward's draw and product: the hash, one compare, the factor
        // select and one plain `mul.f32` (as `nsl_dropout_f32`, never `.rn`,
        // never fused), then the poison select.
        assert_eq!(ptx.matches("mul.lo.u32 ").count(), 7, "{ptx}");
        assert_eq!(ptx.matches("setp.lt.u32 ").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("mul.f32 ").count(), 1, "{ptx}");
        assert_eq!(fwd.matches("mul.f32 ").count(), 1, "{fwd}");
        assert!(
            !ptx.contains("fma.") && !ptx.contains("mul.rn.f32"),
            "{ptx}"
        );
        assert_eq!(
            ptx.matches("selp.f32 ").count(),
            2,
            "factor and poison: {ptx}"
        );
        // The key is read (seven words) and checked (five compares), never written.
        assert_eq!(ptx.matches("ld.global.u32 ").count(), 7, "{ptx}");
        assert_eq!(ptx.matches("setp.eq.u32 ").count(), 5, "{ptx}");
        assert!(!ptx.contains("st.global.u32"), "{ptx}");
        assert_eq!(ptx.matches("st.global.f32 ").count(), 1, "{ptx}");
        assert!(
            !ptx.contains("param_seed"),
            "the base comes from the key: {ptx}"
        );
    }

    /// Both forwards take the base by value and through a pointer, the pointer
    /// last (the runtime's argument arrays depend on the order), and branch on
    /// the pointer once. The key kernel writes the base it resolved, not
    /// the by-value parameter: under a pointer that parameter is 0, and a key
    /// recording 0 would make the backward redraw a different mask.
    #[test]
    fn the_forwards_resolve_the_base_from_a_value_or_a_pointer() {
        // The null test, plus the key kernel's `i == 0`.
        for (name, ptx, eqs) in [
            (DROPOUT_NAME, String::from_utf8(dropout_ptx()).unwrap(), 1),
            (
                DROPOUT_KEY_NAME,
                String::from_utf8(dropout_key_ptx()).unwrap(),
                2,
            ),
        ] {
            let sig = ptx
                .split_once(&format!(".visible .entry {name}("))
                .and_then(|(_, r)| r.split_once(')'))
                .map(|(s, _)| s.to_string())
                .unwrap_or_else(|| panic!("{name}: no entry: {ptx}"));
            let params: Vec<&str> = sig
                .split(',')
                .map(|p| p.split_whitespace().last().unwrap_or(""))
                .collect();
            assert_eq!(params.len(), 8, "{name}: {sig}");
            assert!(params[6].ends_with("seed"), "{name}: {sig}");
            assert!(params[7].ends_with("seed_ptr"), "{name}: {sig}");
            assert_eq!(ptx.matches("setp.eq.u64 ").count(), eqs, "{name}: {ptx}");
        }
        for ir in [build_dropout(), build_dropout_key()] {
            let seed = ir
                .params
                .iter()
                .find(|p| p.name == "seed")
                .map(|p| p.id)
                .expect("the forward takes a seed");
            // The by-value parameter reaches the hash and the key's base words
            // only through the join's block parameter: no op reads it.
            for op in ir.blocks.iter().flat_map(|blk| blk.ops.iter()) {
                let reads = match op {
                    KirOp::Add(_, a, c) | KirOp::Shr(_, a, c) => *a == seed || *c == seed,
                    KirOp::Cast(_, s, _) => *s == seed,
                    KirOp::Store(_, v, _) => *v == seed,
                    _ => false,
                };
                assert!(!reads, "{}: {op:?} reads the by-value seed", ir.name);
            }
            // ... and it does reach the join, as the null edge's argument.
            let passed = ir.blocks.iter().any(|blk| {
                matches!(
                    &blk.terminator,
                    Some(KirTerminator::CondBranch(_, t, _)) if t.args == vec![seed]
                )
            });
            assert!(
                passed,
                "{}: the by-value seed never reaches the join",
                ir.name
            );
        }
    }

    #[test]
    fn the_key_layout_is_eight_distinct_words() {
        use DropoutKeyWord::*;
        let all = [
            Magic, SeedLo, SeedHi, Threshold, ScaleBits, LenLo, LenHi, Reserved,
        ];
        assert_eq!(all.len(), DROPOUT_KEY_WORDS);
        for (k, w) in all.iter().enumerate() {
            assert_eq!(*w as usize, k);
        }
        // Neither a zeroed buffer nor an f32 mask word can pass as the magic.
        assert!(![0u32, 1.0f32.to_bits()].contains(&DROPOUT_KEY_MAGIC));
        assert_eq!(&DROPOUT_KEY_MAGIC.to_le_bytes(), b"DROK");
    }
}
