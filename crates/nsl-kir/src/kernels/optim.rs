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
    let mut scalar = |name: &str| b.add_param(name, KirType::F32, Global);
    let b1 = scalar("b1");
    let omb1 = scalar("omb1");
    let b2 = scalar("b2");
    let omb2 = scalar("omb2");
    let eps = scalar("eps");
    let neg_lr = scalar("neg_lr");
    let neg_lr_wd = scalar("neg_lr_wd");
    let bc1 = scalar("bc1");
    let bc2 = scalar("bc2");
    let has_wd = b.add_param("has_wd", KirType::U32, Global);

    let (i, _, exit) = index_and_bound(&mut b, n);
    let th = load_f32(&mut b, theta, i);
    let mi = load_f32(&mut b, m, i);
    let vi = load_f32(&mut b, v, i);
    let g = load_f32(&mut b, mp, i);

    // m' = rn(rn(m·β₁) + rn(mp·(1-β₁)))
    let m_decay = f32_op2(&mut b, KirOp::MulRn, mi, b1);
    let m_new_part = f32_op2(&mut b, KirOp::MulRn, g, omb1);
    let m_next = f32_op2(&mut b, KirOp::AddRn, m_decay, m_new_part);
    store_f32(&mut b, m, i, m_next);

    // v' = rn(rn(v·β₂) + rn(rn(mp·mp)·(1-β₂)))
    let g2 = f32_op2(&mut b, KirOp::MulRn, g, g);
    let v_new_part = f32_op2(&mut b, KirOp::MulRn, g2, omb2);
    let v_decay = f32_op2(&mut b, KirOp::MulRn, vi, b2);
    let v_next = f32_op2(&mut b, KirOp::AddRn, v_decay, v_new_part);
    store_f32(&mut b, v, i, v_next);

    // u = div.approx(rn(m'·bc1), rn(sqrt.rn(rn(v'·bc2)) + ε)); adj = rn(u·(-lr))
    let m_hat = f32_op2(&mut b, KirOp::MulRn, m_next, bc1);
    let v_hat = f32_op2(&mut b, KirOp::MulRn, v_next, bc2);
    let root = f32_op1(&mut b, KirOp::Sqrt, v_hat);
    let denom = f32_op2(&mut b, KirOp::AddRn, root, eps);
    let u = f32_op2(&mut b, KirOp::DivApprox, m_hat, denom);
    let adj = f32_op2(&mut b, KirOp::MulRn, u, neg_lr);

    let wd = b.new_block();
    let join = b.new_block();
    let adj_in = b.add_block_param(join, KirType::F32);
    let zero = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(zero, KirConst { ty: KirType::U32, value: ConstValue::U32(0) }));
    let no_wd = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(no_wd, has_wd, zero, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(no_wd, KirEdge::with(join, vec![adj]), KirEdge::to(wd)));

    // Decoupled weight decay: adj += rn(θ·(-lr·wd)).
    b.set_block(wd);
    let decay = f32_op2(&mut b, KirOp::MulRn, th, neg_lr_wd);
    let adj_wd = f32_op2(&mut b, KirOp::AddRn, adj, decay);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![adj_wd])));

    b.set_block(join);
    let th_next = f32_op2(&mut b, KirOp::AddRn, th, adj_in);
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
    let mut scalar = |name: &str| b.add_param(name, KirType::F32, Global);
    let b1 = scalar("b1");
    let omb1 = scalar("omb1");
    let b2 = scalar("b2");
    let omb2 = scalar("omb2");
    let eps = scalar("eps");
    let neg_lr = scalar("neg_lr");
    let neg_lr_wd = scalar("neg_lr_wd");
    let bc1 = scalar("bc1");
    let bc2 = scalar("bc2");
    let has_wd = b.add_param("has_wd", KirType::U32, Global);
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
    let m_decay = f32_op2(&mut b, KirOp::MulRn, mi, b1);
    let m_new_part = f32_op2(&mut b, KirOp::MulRn, g, omb1);
    let m_next = f32_op2(&mut b, KirOp::AddRn, m_decay, m_new_part);
    store_f32(&mut b, m, i, m_next);
    let g2 = f32_op2(&mut b, KirOp::MulRn, g, g);
    let v_new_part = f32_op2(&mut b, KirOp::MulRn, g2, omb2);
    let v_decay = f32_op2(&mut b, KirOp::MulRn, vi, b2);
    let v_next = f32_op2(&mut b, KirOp::AddRn, v_decay, v_new_part);
    store_f32(&mut b, v, i, v_next);
    let m_hat = f32_op2(&mut b, KirOp::MulRn, m_next, bc1);
    let v_hat = f32_op2(&mut b, KirOp::MulRn, v_next, bc2);
    let root = f32_op1(&mut b, KirOp::Sqrt, v_hat);
    let denom = f32_op2(&mut b, KirOp::AddRn, root, eps);
    let u = f32_op2(&mut b, KirOp::DivApprox, m_hat, denom);
    let adj = f32_op2(&mut b, KirOp::MulRn, u, neg_lr);

    let wd = b.new_block();
    let join = b.new_block();
    let adj_in = b.add_block_param(join, KirType::F32);
    let zero = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(zero, KirConst { ty: KirType::U32, value: ConstValue::U32(0) }));
    let no_wd = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(no_wd, has_wd, zero, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(no_wd, KirEdge::with(join, vec![adj]), KirEdge::to(wd)));

    b.set_block(wd);
    let decay = f32_op2(&mut b, KirOp::MulRn, th, neg_lr_wd);
    let adj_wd = f32_op2(&mut b, KirOp::AddRn, adj, decay);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![adj_wd])));

    b.set_block(join);
    let th_next = f32_op2(&mut b, KirOp::AddRn, th, adj_in);
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

#[cfg(test)]
mod tests {
    use super::*;

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
