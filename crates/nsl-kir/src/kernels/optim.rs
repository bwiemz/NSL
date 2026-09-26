// crates/nsl-kir/src/kernels/optim.rs
//! The runtime's fused optimizer-step kernels, as KIR (roadmap A2 step 11,
//! new-roadmap item 5).
//!
//! `nsl_fase_fused_adamw_step_f32` is the FASE-Deferred AdamW/Adam step for
//! one parameter: one launch in place of the ~15-launch decomposed
//! `UpdateProgram`, and bit-exact with it. Per element, in the decomposed
//! program's rounding order (every arithmetic operation `.rn`, so ptxas
//! cannot contract a multiply into the add that reads it):
//!
//! ```text
//! m'  = rn(rn(m·β₁) + rn(mp·(1-β₁)))                  -> m
//! v'  = rn(rn(v·β₂) + rn(rn(mp·mp)·(1-β₂)))           -> v
//! u   = div.approx(rn(m'·bc1), rn(sqrt.rn(rn(v'·bc2)) + ε))
//! adj = rn(u·(-lr));  if has_wd != 0: adj = rn(adj + rn(θ·(-lr·wd)))
//! θ'  = rn(θ + adj)                                    -> θ
//! ```
//!
//! The quotient is `div.approx.f32` (`KirOp::DivApprox`) because the
//! decomposed program divides with `nsl_div_f32`, which is `div.approx`.
//! `mp` (the accumulated gradient) is only read.
//!
//! Signature `(theta, m, v, mp, n, b1, omb1, b2, omb2, eps, neg_lr,
//! neg_lr_wd, bc1, bc2, has_wd)`: four f32 pointers, `n` a `.u64`, nine
//! `.f32` scalars and a `.u32` flag. One thread per element, blocks of
//! [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK).

use super::elementwise::{f32_op1, f32_op2, f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// The entry name the runtime launches.
pub const FASE_ADAMW_STEP_NAME: &str = "nsl_fase_fused_adamw_step_f32";

/// The AdamW hyperparameters and the decay flag, in the kernels' parameter
/// order: `b1, omb1, b2, omb2, eps, neg_lr, neg_lr_wd, bc1, bc2, has_wd`.
struct AdamwScalars {
    b1: VarId,
    omb1: VarId,
    b2: VarId,
    omb2: VarId,
    eps: VarId,
    neg_lr: VarId,
    neg_lr_wd: VarId,
    bc1: VarId,
    bc2: VarId,
    has_wd: VarId,
}

fn add_adamw_scalars(b: &mut KirBuilder) -> AdamwScalars {
    use AddressSpace::Global;
    let mut scalar = |name: &str| b.add_param(name, KirType::F32, Global);
    let (b1, omb1, b2, omb2, eps) = (scalar("b1"), scalar("omb1"), scalar("b2"), scalar("omb2"), scalar("eps"));
    let (neg_lr, neg_lr_wd, bc1, bc2) = (scalar("neg_lr"), scalar("neg_lr_wd"), scalar("bc1"), scalar("bc2"));
    let has_wd = b.add_param("has_wd", KirType::U32, Global);
    AdamwScalars { b1, omb1, b2, omb2, eps, neg_lr, neg_lr_wd, bc1, bc2, has_wd }
}

/// The per-element AdamW update every kernel here shares, in the hand
/// kernels' order: `m'` and `v'` computed and stored at `m[i]` / `v[i]`,
/// then the step, then the decay branch. Returns `θ' = rn(θ + adj)`, with
/// the builder left in the join block that defines it.
#[allow(clippy::too_many_arguments)]
fn adamw_update(
    b: &mut KirBuilder,
    s: &AdamwScalars,
    (th, mi, vi, g): (VarId, VarId, VarId, VarId),
    m: VarId,
    v: VarId,
    i: VarId,
) -> VarId {
    // m' = rn(rn(m·β₁) + rn(mp·(1-β₁)))
    let m_decay = f32_op2(b, KirOp::MulRn, mi, s.b1);
    let m_new_part = f32_op2(b, KirOp::MulRn, g, s.omb1);
    let m_next = f32_op2(b, KirOp::AddRn, m_decay, m_new_part);
    store_f32(b, m, i, m_next);

    // v' = rn(rn(v·β₂) + rn(rn(mp·mp)·(1-β₂)))
    let g2 = f32_op2(b, KirOp::MulRn, g, g);
    let v_new_part = f32_op2(b, KirOp::MulRn, g2, s.omb2);
    let v_decay = f32_op2(b, KirOp::MulRn, vi, s.b2);
    let v_next = f32_op2(b, KirOp::AddRn, v_decay, v_new_part);
    store_f32(b, v, i, v_next);

    // u = div.approx(rn(m'·bc1), rn(sqrt.rn(rn(v'·bc2)) + ε)); adj = rn(u·(-lr))
    let m_hat = f32_op2(b, KirOp::MulRn, m_next, s.bc1);
    let v_hat = f32_op2(b, KirOp::MulRn, v_next, s.bc2);
    let root = f32_op1(b, KirOp::Sqrt, v_hat);
    let denom = f32_op2(b, KirOp::AddRn, root, s.eps);
    let u = f32_op2(b, KirOp::DivApprox, m_hat, denom);
    let adj = f32_op2(b, KirOp::MulRn, u, s.neg_lr);

    let wd = b.new_block();
    let join = b.new_block();
    let adj_in = b.add_block_param(join, KirType::F32);
    let zero = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(zero, KirConst { ty: KirType::U32, value: ConstValue::U32(0) }));
    let no_wd = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(no_wd, s.has_wd, zero, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(no_wd, KirEdge::with(join, vec![adj]), KirEdge::to(wd)));

    // Decoupled weight decay: adj += rn(θ·(-lr·wd)).
    b.set_block(wd);
    let decay = f32_op2(b, KirOp::MulRn, th, s.neg_lr_wd);
    let adj_wd = f32_op2(b, KirOp::AddRn, adj, decay);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![adj_wd])));

    b.set_block(join);
    f32_op2(b, KirOp::AddRn, th, adj_in)
}

/// Build `nsl_fase_fused_adamw_step_f32` in the hand kernel's instruction
/// order.
///
/// ```text
/// entry:  i = blockIdx.x*blockDim.x + threadIdx.x (u32, widened)
///         if i >= n { br exit } else { br body }
/// body:   θ, m, v, mp = loads; m', v' stored; adj = rn(u·(-lr))
///         if has_wd == 0 { br join(adj) } else { br wd }
/// wd:     br join(rn(adj + rn(θ·(-lr·wd))))
/// join(adj): θ[i] = rn(θ + adj); br exit
/// exit:   ret
/// ```
pub fn build_fase_adamw_step() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(FASE_ADAMW_STEP_NAME);
    let theta = b.add_param("theta", f32_ptr(), Global);
    let m = b.add_param("m", f32_ptr(), Global);
    let v = b.add_param("v", f32_ptr(), Global);
    let mp = b.add_param("mp", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let s = add_adamw_scalars(&mut b);

    let (i, _, exit) = index_and_bound(&mut b, n);
    let th = load_f32(&mut b, theta, i);
    let mi = load_f32(&mut b, m, i);
    let vi = load_f32(&mut b, v, i);
    let g = load_f32(&mut b, mp, i);
    let th_next = adamw_update(&mut b, &s, (th, mi, vi, g), m, v, i);
    store_f32(&mut b, theta, i, th_next);
    finish(b, exit)
}

/// [`build_fase_adamw_step`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_step_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_step())
}

/// The multi-tensor entry name the runtime launches.
pub const FASE_ADAMW_MULTI_NAME: &str = "nsl_fase_fused_adamw_multi_f32";

/// `nsl_fase_fused_adamw_multi_f32`: [`build_fase_adamw_step`] over every
/// parameter in one launch, on a FLAT grid.
///
/// Signature `(ttab, mtab, vtab, mptab, ntab, b1, omb1, b2, omb2, eps,
/// neg_lr, neg_lr_wd, bc1, bc2, has_wd, bptab, bbtab, mp_scale)`. The four
/// `*tab` pointer tables hold one device pointer per parameter, `ntab` one
/// `u32` length; `bptab[b]` / `bbtab[b]` are the parameter and element
/// offset of block `b`, built once on the host from the shape list. The
/// kernel reads neither `%ntid` nor `%nctaid`: its element is `bbtab[b] +
/// threadIdx.x`, so `blockDim.x` must be the block size the tables were
/// built with.
///
/// Per element it is the single-parameter step, with two differences:
///
/// - `mp_scale` folds the two-phase-clip Phase-B pre-scale into the read,
///   `g = rn(mp * mp_scale)`, and is branched AROUND when it equals 1.0 so
///   the unclipped path keeps NaN payloads bit-for-bit.
/// - After θ is stored, `mp[e] = 0`: the separate `nsl_tensor_zero_inplace`
///   pass folded in.
///
/// ```text
/// entry:  p = bptab[b]; e = bbtab[b] + tid (u32); if e >= ntab[p] exit
/// body:   θ, m, v, mp bases from the tables; loads at e
///         if mp_scale == 1.0 { br join(g) } else { br scale }
/// scale:  br join(rn(g·mp_scale))
/// join(g): the single-parameter body; wd branch; θ[e] stored; mp[e] = 0
/// ```
pub fn build_fase_adamw_multi() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(FASE_ADAMW_MULTI_NAME);
    let table = || KirType::Ptr(Box::new(f32_ptr()), Global);
    let u32_ptr = || KirType::Ptr(Box::new(KirType::U32), Global);
    let ttab = b.add_param("ttab", table(), Global);
    let mtab = b.add_param("mtab", table(), Global);
    let vtab = b.add_param("vtab", table(), Global);
    let mptab = b.add_param("mptab", table(), Global);
    let ntab = b.add_param("ntab", u32_ptr(), Global);
    let s = add_adamw_scalars(&mut b);
    let bptab = b.add_param("bptab", u32_ptr(), Global);
    let bbtab = b.add_param("bbtab", u32_ptr(), Global);
    let mp_scale = b.add_param("mp_scale", KirType::F32, Global);

    let entry = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    // `ptr[idx]` for a u32 table and a u64 index.
    let load_at = |b: &mut KirBuilder, base: VarId, idx: VarId, elem: KirType| {
        let addr = b.new_typed_var(KirType::Ptr(Box::new(elem.clone()), Global));
        b.emit(KirOp::PtrOffset(addr, base, idx));
        let v = b.new_typed_var(elem);
        b.emit(KirOp::Load(v, addr, Global));
        v
    };
    let widen = |b: &mut KirBuilder, x: VarId| {
        let w = b.new_typed_var(KirType::U64);
        b.emit(KirOp::Cast(w, x, KirType::U64));
        w
    };
    let blk = b.new_typed_var(KirType::U32);
    b.emit(KirOp::BlockIdx(blk, 0));
    let blk = widen(&mut b, blk);
    let param = load_at(&mut b, bptab, blk, KirType::U32);
    let first = load_at(&mut b, bbtab, blk, KirType::U32);
    let tid = b.new_typed_var(KirType::U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let e32 = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Add(e32, first, tid));
    let param = widen(&mut b, param);
    let len = load_at(&mut b, ntab, param, KirType::U32);
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, e32, len, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));

    b.set_block(body);
    let theta = load_at(&mut b, ttab, param, f32_ptr());
    let m = load_at(&mut b, mtab, param, f32_ptr());
    let v = load_at(&mut b, vtab, param, f32_ptr());
    let mp = load_at(&mut b, mptab, param, f32_ptr());
    let i = widen(&mut b, e32);
    let th = load_f32(&mut b, theta, i);
    let mi = load_f32(&mut b, m, i);
    let vi = load_f32(&mut b, v, i);
    let g_raw = load_f32(&mut b, mp, i);

    // Phase-B clip pre-scale, skipped at exactly 1.0.
    let scale = b.new_block();
    let scaled = b.new_block();
    let g = b.add_block_param(scaled, KirType::F32);
    let one = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(one, KirConst { ty: KirType::F32, value: ConstValue::F32(1.0) }));
    let unscaled = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(unscaled, mp_scale, one, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(unscaled, KirEdge::with(scaled, vec![g_raw]), KirEdge::to(scale)));
    b.set_block(scale);
    let g_scaled = f32_op2(&mut b, KirOp::MulRn, g_raw, mp_scale);
    b.terminate(KirTerminator::Branch(KirEdge::with(scaled, vec![g_scaled])));

    b.set_block(scaled);
    let th_next = adamw_update(&mut b, &s, (th, mi, vi, g), m, v, i);
    store_f32(&mut b, theta, i, th_next);
    // The accumulated gradient is consumed: zero it (was a separate launch).
    let fzero = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(fzero, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
    store_f32(&mut b, mp, i, fzero);
    finish(b, exit)
}

/// [`build_fase_adamw_multi`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_multi_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_multi())
}

/// The single-parameter SR-BF16 step's entry name.
pub const FASE_ADAMW_STEP_BF16SR_NAME: &str = "nsl_fase_fused_adamw_step_bf16sr";

/// The standalone SR-BF16 rounding probe's entry name.
pub const SR_BF16_ROUND_PROBE_NAME: &str = "nsl_sr_bf16_round_probe";

/// splitmix64's counter increment, the host's `sr_bf16::SR_STEP_SALT`.
pub const SR_SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// splitmix64's two finalizer multipliers.
const SR_SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SR_SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;

fn int_const(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = b.new_typed_var(ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn int_op2(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(ty);
    b.emit(op(dst, x, y));
    dst
}

/// `x op c` for a `u32` constant `c`.
fn u32_imm(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, c: u32) -> VarId {
    let c = int_const(b, KirType::U32, ConstValue::U32(c));
    int_op2(b, KirType::U32, op, x, c)
}

/// The SR-BF16 rounding tail: the f32 bit pattern `bits` (a `U32`) to bf16
/// storage bits, stochastically rounded. It is the hand kernels' tail and
/// the host's `sr_bf16::sr_bf16_round_counter`, bit for bit.
///
/// ```text
/// sign = (bits & 0x80000000) >> 16
/// if bits & 0x7f800000 == 0x7f800000:              special
///     NaN (mantissa != 0) -> sign | 0x7fc0; ±∞ -> sign | 0x7f80
/// z = splitmix64(key + (ctr_base + i)·γ)            dither
/// r = bits + (z & 0xffff)
/// if r & 0x7f800000 == 0x7f800000: sign | 0x7f7f    saturate
/// else r >> 16                                      truncate
/// ```
///
/// Returns the bits, a `U32`, as the parameter of the block the four paths
/// join in; the builder is left there.
fn sr_bf16_bits(b: &mut KirBuilder, bits: VarId, i: VarId, key: VarId, ctr_base: VarId) -> VarId {
    let sign = u32_imm(b, KirOp::And, bits, 0x8000_0000);
    let sign = u32_imm(b, KirOp::Shr, sign, 16);
    let exp = u32_imm(b, KirOp::And, bits, 0x7F80_0000);
    let all_ones = int_const(b, KirType::U32, ConstValue::U32(0x7F80_0000));
    let special = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(special, exp, all_ones, CmpOp::Eq));
    let dither = b.new_block();
    let truncate = b.new_block();
    let saturate = b.new_block();
    let special_blk = b.new_block();
    let inf = b.new_block();
    let qnan = b.new_block();
    let store = b.new_block();
    let out = b.add_block_param(store, KirType::U32);
    b.terminate(KirTerminator::CondBranch(special, KirEdge::to(special_blk), KirEdge::to(dither)));

    // The dither: the low 16 bits of splitmix64(key, ctr_base + i).
    b.set_block(dither);
    let u64_imm = |b: &mut KirBuilder, c: u64| int_const(b, KirType::U64, ConstValue::U64(c));
    let ctr = int_op2(b, KirType::U64, KirOp::Add, ctr_base, i);
    let gamma = u64_imm(b, SR_SPLITMIX_GAMMA);
    let z = int_op2(b, KirType::U64, KirOp::Mul, ctr, gamma);
    let mut z = int_op2(b, KirType::U64, KirOp::Add, key, z);
    for (shift, mult) in [(30, Some(SR_SPLITMIX_M1)), (27, Some(SR_SPLITMIX_M2)), (31, None)] {
        let amount = int_const(b, KirType::U32, ConstValue::U32(shift));
        let hi = int_op2(b, KirType::U64, KirOp::Shr, z, amount);
        z = int_op2(b, KirType::U64, KirOp::Xor, z, hi);
        if let Some(mult) = mult {
            let mult = u64_imm(b, mult);
            z = int_op2(b, KirType::U64, KirOp::Mul, z, mult);
        }
    }
    let low = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(low, z, KirType::U32));
    let d = u32_imm(b, KirOp::And, low, 0xFFFF);
    let r = int_op2(b, KirType::U32, KirOp::Add, bits, d);
    let r_exp = u32_imm(b, KirOp::And, r, 0x7F80_0000);
    let all_ones = int_const(b, KirType::U32, ConstValue::U32(0x7F80_0000));
    let overflow = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(overflow, r_exp, all_ones, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(overflow, KirEdge::to(saturate), KirEdge::to(truncate)));

    b.set_block(truncate);
    let t = u32_imm(b, KirOp::Shr, r, 16);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    // Rounding carried into the all-ones exponent: ±max-normal.
    b.set_block(saturate);
    let t = u32_imm(b, KirOp::Or, sign, 0x7F7F);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    // The input was ±∞ or NaN: propagate, quieting a NaN.
    b.set_block(special_blk);
    let mant = u32_imm(b, KirOp::And, bits, 0x007F_FFFF);
    let zero = int_const(b, KirType::U32, ConstValue::U32(0));
    let nan = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(nan, mant, zero, CmpOp::Ne));
    b.terminate(KirTerminator::CondBranch(nan, KirEdge::to(qnan), KirEdge::to(inf)));

    b.set_block(inf);
    let t = u32_imm(b, KirOp::Or, sign, 0x7F80);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    b.set_block(qnan);
    let t = u32_imm(b, KirOp::Or, sign, 0x7FC0);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    b.set_block(store);
    out
}

fn u16_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::U16), AddressSpace::Global)
}

/// `base[i] = bits` for a `u16` array: `cvt.u16.u32`, `st.global.u16`.
fn store_u16(b: &mut KirBuilder, base: VarId, i: VarId, bits: VarId) {
    let h = b.new_typed_var(KirType::U16);
    b.emit(KirOp::Cast(h, bits, KirType::U16));
    let addr = b.new_typed_var(u16_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, h, AddressSpace::Global));
}

/// `nsl_fase_fused_adamw_step_bf16sr`: [`build_fase_adamw_step`] with a
/// bf16 θ and a stochastically rounded θ store (P4 item 17, SR-BF16).
///
/// Signature `(theta, m, v, mp, n, b1, omb1, b2, omb2, eps, neg_lr,
/// neg_lr_wd, bc1, bc2, has_wd, sr_key, sr_ctr_base)`. `theta` holds bf16
/// bits; `m`, `v`, `mp` are f32. The θ load widens exactly (`bits << 16`,
/// a `mov.b32` into f32). The update is the f32 step's, rounding for
/// rounding. The store is [`sr_bf16_bits`] with the counter
/// `sr_ctr_base + i` and the launch key `sr_key` (the host's
/// `sr_step_key(seed, step)`).
pub fn build_fase_adamw_step_bf16sr() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(FASE_ADAMW_STEP_BF16SR_NAME);
    let theta = b.add_param("theta", u16_ptr(), Global);
    let m = b.add_param("m", f32_ptr(), Global);
    let v = b.add_param("v", f32_ptr(), Global);
    let mp = b.add_param("mp", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let s = add_adamw_scalars(&mut b);
    let key = b.add_param("sr_key", KirType::U64, Global);
    let ctr_base = b.add_param("sr_ctr_base", KirType::U64, Global);

    let (i, _, exit) = index_and_bound(&mut b, n);
    // θ: bf16 bits, widened exactly.
    let addr = b.new_typed_var(u16_ptr());
    b.emit(KirOp::PtrOffset(addr, theta, i));
    let h = b.new_typed_var(KirType::U16);
    b.emit(KirOp::Load(h, addr, Global));
    let w = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(w, h, KirType::U32));
    let w = u32_imm(&mut b, KirOp::Shl, w, 16);
    let th = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Bitcast(th, w));
    let mi = load_f32(&mut b, m, i);
    let vi = load_f32(&mut b, v, i);
    let g = load_f32(&mut b, mp, i);
    let th_next = adamw_update(&mut b, &s, (th, mi, vi, g), m, v, i);

    let bits = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Bitcast(bits, th_next));
    let out = sr_bf16_bits(&mut b, bits, i, key, ctr_base);
    store_u16(&mut b, theta, i, out);
    finish(b, exit)
}

/// [`build_fase_adamw_step_bf16sr`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_step_bf16sr_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_step_bf16sr())
}

/// `nsl_sr_bf16_round_probe(src, dst, n, sr_key, sr_ctr_base)`: the SR-BF16
/// tail alone, `dst[i] = sr_bf16_bits(src[i])`, over f32 bit patterns
/// (`src` is read as `u32`). The parity gate drives it with adversarial
/// inputs (max-normal, subnormals, ±∞, NaN) against the host reference,
/// without the optimizer arithmetic in the way.
pub fn build_sr_bf16_round_probe() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(SR_BF16_ROUND_PROBE_NAME);
    let src = b.add_param("src", KirType::Ptr(Box::new(KirType::U32), Global), Global);
    let dst = b.add_param("dst", u16_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let key = b.add_param("sr_key", KirType::U64, Global);
    let ctr_base = b.add_param("sr_ctr_base", KirType::U64, Global);

    let (i, _, exit) = index_and_bound(&mut b, n);
    let addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::U32), Global));
    b.emit(KirOp::PtrOffset(addr, src, i));
    let bits = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Load(bits, addr, Global));
    let out = sr_bf16_bits(&mut b, bits, i, key, ctr_base);
    store_u16(&mut b, dst, i, out);
    finish(b, exit)
}

/// [`build_sr_bf16_round_probe`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn sr_bf16_round_probe_ptx() -> Vec<u8> {
    verified_ptx(build_sr_bf16_round_probe())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sr_bf16_kernels_verify_and_share_one_tail() {
        for (ir, name) in [
            (build_sr_bf16_round_probe(), SR_BF16_ROUND_PROBE_NAME),
            (build_fase_adamw_step_bf16sr(), FASE_ADAMW_STEP_BF16SR_NAME),
        ] {
            crate::kir_verify::verify(&ir).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(ir.name, name);
        }
        let probe = String::from_utf8(sr_bf16_round_probe_ptx()).unwrap();
        let step = String::from_utf8(fase_adamw_step_bf16sr_ptx()).unwrap();
        for p in [&probe, &step] {
            for c in [SR_SPLITMIX_GAMMA, SR_SPLITMIX_M1, SR_SPLITMIX_M2] {
                assert_eq!(p.matches(&format!(", {c};")).count(), 1, "{c:#x}");
            }
            assert_eq!(p.matches("st.global.u16 ").count(), 1);
            assert_eq!(p.matches("cvt.u16.u32 ").count(), 1);
        }
        assert_eq!(step.matches("mov.b32 ").count(), 2, "θ in, θ' out");
        assert_eq!(step.matches("ld.global.u16 ").count(), 1);
        assert_eq!(step.matches("div.approx.f32").count(), 1);
    }

    #[test]
    fn fase_adamw_step_verifies_and_spells_its_rounding() {
        let ptx = String::from_utf8(fase_adamw_step_ptx()).unwrap();
        assert!(ptx.contains(".visible .entry nsl_fase_fused_adamw_step_f32("), "{ptx}");
        assert_eq!(ptx.matches("div.approx.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("sqrt.rn.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("mul.rn.f32").count(), 9, "{ptx}");
        assert_eq!(ptx.matches("add.rn.f32").count(), 5, "{ptx}");
        assert!(!ptx.contains("fma") && !ptx.contains("div.rn"), "{ptx}");
    }

    #[test]
    fn fase_adamw_multi_verifies_and_spells_its_rounding() {
        let ptx = String::from_utf8(fase_adamw_multi_ptx()).unwrap();
        assert!(ptx.contains(".visible .entry nsl_fase_fused_adamw_multi_f32("), "{ptx}");
        assert_eq!(ptx.matches("div.approx.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("sqrt.rn.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("mul.rn.f32").count(), 10, "{ptx}");
        assert_eq!(ptx.matches("add.rn.f32").count(), 5, "{ptx}");
        assert!(!ptx.contains("fma") && !ptx.contains("div.rn"), "{ptx}");
        assert!(!ptx.contains("%ntid"), "the element is bbtab[b] + tid: {ptx}");
    }
}
