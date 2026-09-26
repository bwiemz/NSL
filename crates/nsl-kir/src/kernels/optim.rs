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
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType};

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
}
