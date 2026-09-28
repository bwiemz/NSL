// crates/nsl-kir/src/kernels/rmsnorm_dgamma.rs
//! The runtime's two-launch fused RMSNorm gamma backward from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5). Both kernels run a
//! thread per output on blocks of
//! [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK), over a
//! row-major `[rows, cols]` view:
//!
//! - `nsl_rmsnorm_rinv_rows_f32(x, rinv, rows, cols, eps)`: a thread per
//!   row, `rinv[r] = 1 / sqrt(Σ_j x[r, j]² / cols + eps)`. The sum of
//!   squares is `fma.rn`-accumulated from `+0.0` in column order; `cols` is
//!   converted with `cvt.rn.f32.u64`; the mean, `+ eps`, the square root
//!   and the reciprocal are each correctly rounded (`div.rn`, `add.rn`,
//!   `sqrt.rn`, `div.rn`).
//! - `nsl_rmsnorm_dgamma_f32(dy, x, rinv, dgamma, rows, cols)`: a thread
//!   per column, `dgamma[j] = Σ_i (dy[i, j] · x[i, j]) · rinv[i]`, summed in
//!   row order from `+0.0` with every multiply and add `.rn`. The fixed
//!   order makes the result bit-deterministic run to run.
//!
//! Each keeps the hand kernel's loads and rounding. Each loop leaves its
//! header for a block the header dominates, so the exit edge carries no
//! copies.

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// Which of the two launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormDgammaOp {
    /// The per-row `1 / rms`.
    RinvRows,
    /// The per-column gamma gradient.
    Dgamma,
}

impl RmsNormDgammaOp {
    pub const ALL: [RmsNormDgammaOp; 2] = [RmsNormDgammaOp::RinvRows, RmsNormDgammaOp::Dgamma];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            RmsNormDgammaOp::RinvRows => "nsl_rmsnorm_rinv_rows_f32",
            RmsNormDgammaOp::Dgamma => "nsl_rmsnorm_dgamma_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            RmsNormDgammaOp::RinvRows => &["x", "rinv", "rows", "cols", "eps"],
            RmsNormDgammaOp::Dgamma => &["dy", "x", "rinv", "dgamma", "rows", "cols"],
        }
    }
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = b.new_typed_var(ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn op2(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(ty);
    b.emit(op(dst, x, y));
    dst
}

/// `acc = step(acc, k)` for `k` in `0..n`, from `+0.0`, entered from the
/// current block. Returns `acc` with the builder in a block the loop header
/// dominates.
fn accumulate(b: &mut KirBuilder, n: VarId, step: impl Fn(&mut KirBuilder, VarId, VarId) -> VarId) -> (VarId, BlockId) {
    use KirType::{F32, U64};
    let head = b.new_block();
    let body = b.new_block();
    let out = b.new_block();
    let zero = konst(b, U64, ConstValue::U64(0));
    let acc0 = konst(b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, acc0])));
    let k = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);

    b.set_block(head);
    let end = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(end, k, n, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(out), KirEdge::to(body)));

    b.set_block(body);
    let acc_next = step(b, acc, k);
    let one = konst(b, U64, ConstValue::U64(1));
    let k_next = op2(b, U64, KirOp::Add, k, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, acc_next])));

    b.set_block(out);
    (acc, out)
}

/// ```text
/// entry: r = global id; if r >= rows { exit }
/// loop:  s = fma(x[r·cols + j], x[r·cols + j], s) for j in 0..cols
/// done:  rinv[r] = 1 / sqrt(s / f32(cols) + eps)
/// ```
fn build_rinv_rows() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let names = RmsNormDgammaOp::RinvRows.param_names();
    let mut b = KirBuilder::new(RmsNormDgammaOp::RinvRows.kernel_name());
    let x = b.add_param(names[0], f32_ptr(), Global);
    let rinv = b.add_param(names[1], f32_ptr(), Global);
    let rows = b.add_param(names[2], U64, Global);
    let cols = b.add_param(names[3], U64, Global);
    let eps = b.add_param(names[4], F32, Global);

    let (r, _, exit) = index_and_bound(&mut b, rows);
    let base = op2(&mut b, U64, KirOp::Mul, r, cols);
    let (s, _) = accumulate(&mut b, cols, |b, s, j| {
        let at = op2(b, U64, KirOp::Add, base, j);
        let v = load_f32(b, x, at);
        let next = b.new_typed_var(F32);
        b.emit(KirOp::Fma(next, v, v, s));
        next
    });
    let n = b.new_typed_var(F32);
    b.emit(KirOp::Cast(n, cols, F32));
    let mean = op2(&mut b, F32, KirOp::Div, s, n);
    let shifted = op2(&mut b, F32, KirOp::AddRn, mean, eps);
    let root = b.new_typed_var(F32);
    b.emit(KirOp::Sqrt(root, shifted));
    let one = konst(&mut b, F32, ConstValue::F32(1.0));
    let inv = op2(&mut b, F32, KirOp::Div, one, root);
    store_f32(&mut b, rinv, r, inv);
    finish(b, exit)
}

/// ```text
/// entry: j = global id; if j >= cols { exit }
/// head(i, at, acc): if i >= rows { done }        (at = i·cols + j)
/// body:  acc += (dy[at] · x[at]) · rinv[i]; at += cols; i += 1
/// done:  dgamma[j] = acc
/// ```
fn build_dgamma() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let names = RmsNormDgammaOp::Dgamma.param_names();
    let mut b = KirBuilder::new(RmsNormDgammaOp::Dgamma.kernel_name());
    let dy = b.add_param(names[0], f32_ptr(), Global);
    let x = b.add_param(names[1], f32_ptr(), Global);
    let rinv = b.add_param(names[2], f32_ptr(), Global);
    let dgamma = b.add_param(names[3], f32_ptr(), Global);
    let rows = b.add_param(names[4], U64, Global);
    let cols = b.add_param(names[5], U64, Global);

    let (j, _, exit) = index_and_bound(&mut b, cols);
    // The loop carries the element offset `i·cols + j` and steps it by
    // `cols`, as the hand kernel stepped its two pointers, instead of
    // forming the product each trip.
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    let acc0 = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, j, acc0])));
    let i = b.add_block_param(head, U64);
    let at = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);

    b.set_block(head);
    let end = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(end, i, rows, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let g = load_f32(&mut b, dy, at);
    let v = load_f32(&mut b, x, at);
    let ri = load_f32(&mut b, rinv, i);
    let p = op2(&mut b, F32, KirOp::MulRn, g, v);
    let q = op2(&mut b, F32, KirOp::MulRn, p, ri);
    let acc_next = op2(&mut b, F32, KirOp::AddRn, acc, q);
    let at_next = op2(&mut b, U64, KirOp::Add, at, cols);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let i_next = op2(&mut b, U64, KirOp::Add, i, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, at_next, acc_next])));

    b.set_block(done);
    store_f32(&mut b, dgamma, j, acc);
    finish(b, exit)
}

/// Build `op` as KIR.
pub fn build(op: RmsNormDgammaOp) -> KernelIR {
    match op {
        RmsNormDgammaOp::RinvRows => build_rinv_rows(),
        RmsNormDgammaOp::Dgamma => build_dgamma(),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: RmsNormDgammaOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: RmsNormDgammaOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in RmsNormDgammaOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}: NUL-terminated");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
        }
    }

    /// Every float operation is explicitly rounded, so none fuses or
    /// approximates.
    #[test]
    fn every_float_operation_is_correctly_rounded() {
        let r = text(RmsNormDgammaOp::RinvRows);
        assert_eq!(r.matches("fma.rn.f32").count(), 1, "{r}");
        assert_eq!(r.matches("div.rn.f32").count(), 2, "{r}");
        assert_eq!(r.matches("sqrt.rn.f32").count(), 1, "{r}");
        assert_eq!(r.matches("add.rn.f32").count(), 1, "{r}");
        assert!(r.contains("cvt.rn.f32.u64"), "{r}");
        let d = text(RmsNormDgammaOp::Dgamma);
        assert_eq!(d.matches("mul.rn.f32").count(), 2, "{d}");
        assert_eq!(d.matches("add.rn.f32").count(), 1, "{d}");
        for p in [&r, &d] {
            assert!(!p.contains("approx"), "{p}");
            assert!(!p.lines().any(|l| l.contains(".f32 ") && (l.contains(" add.f32") || l.contains(" mul.f32"))), "{p}");
        }
    }

    /// Each loop's bound test branches straight out, with no edge copies
    /// printed around it.
    #[test]
    fn the_loop_exits_carry_no_copies() {
        for op in RmsNormDgammaOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
