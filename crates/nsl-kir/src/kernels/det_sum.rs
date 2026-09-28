// crates/nsl-kir/src/kernels/det_sum.rs
//! The runtime's deterministic sum kernels from `cuda/fused_kernels.rs`, as
//! KIR (new-roadmap item 5).
//!
//! All three sum in ascending index order with one thread per result, so
//! the result does not depend on scheduling:
//!
//! - `nsl_det_global_sum_f32(inp, out, len)`: one thread, launched as a
//!   single one-thread block, adds `inp[0..len]` in order into `out[0]`.
//! - `nsl_det_sum_dim_f32(inp, out, outer, reduce_size, inner)`: a
//!   one-thread block per output, `t = %ctaid.x`. Output `t = o·inner + i`
//!   adds `inp[(o·reduce_size + r)·inner + i]` for `r` in
//!   `0..reduce_size`, in order, into `out[t]`. A block at or past
//!   `outer·inner` does nothing.
//! - `nsl_sum_dim_short_f32(inp, out, outer, reduce_size, inner)`: the same
//!   sum on blocks of [`SUM_DIM_SHORT_BLOCK`] threads, the output being the
//!   thread's global index `%ctaid.x · %ntid.x + %tid.x`. Consecutive
//!   threads take consecutive `i`, so the loads stay coalesced when `inner`
//!   is large. It replaced a shared-memory tree per output for reductions
//!   too short to fill one.
//!
//! The accumulator starts at `+0.0`, and every add is `add.rn.f32`: the
//! hand kernels' `add.f32` rounds the same way, and the explicit rounding
//! keeps a future multiply from being contracted into it.
//!
//! Each loop is `head(i, acc)`, which tests the bound and leaves for a block
//! the header dominates (so the exit edge carries no copies), and a body
//! that branches back with `(i + 1, acc + x)`.

use super::elementwise::{f32_ptr, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// The block the deterministic kernels are launched with: one thread.
pub const DET_SUM_BLOCK: u32 = 1;

/// The block `nsl_sum_dim_short_f32` is launched with.
pub const SUM_DIM_SHORT_BLOCK: u32 = 256;

/// Which deterministic sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetSumOp {
    /// `out[0] = inp[0] + inp[1] + … + inp[len - 1]`.
    Global,
    /// The sum over the middle axis of an `[outer, reduce_size, inner]` view,
    /// a one-thread block per output.
    Dim,
    /// [`DetSumOp::Dim`] with a thread per output on blocks of
    /// [`SUM_DIM_SHORT_BLOCK`].
    DimShort,
}

impl DetSumOp {
    pub const ALL: [DetSumOp; 3] = [DetSumOp::Global, DetSumOp::Dim, DetSumOp::DimShort];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            DetSumOp::Global => "nsl_det_global_sum_f32",
            DetSumOp::Dim => "nsl_det_sum_dim_f32",
            DetSumOp::DimShort => "nsl_sum_dim_short_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            DetSumOp::Global => &["inp", "out", "len"],
            DetSumOp::Dim | DetSumOp::DimShort => &["inp", "out", "outer", "reduce_size", "inner"],
        }
    }

    /// The block the runtime launches the kernel with.
    pub fn block(self) -> u32 {
        match self {
            DetSumOp::Global | DetSumOp::Dim => DET_SUM_BLOCK,
            DetSumOp::DimShort => SUM_DIM_SHORT_BLOCK,
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

/// The ascending accumulation `acc = Σ inp[at(r)]` for `r` in `0..n`,
/// entered from the current block. Returns `(acc, out)` with the builder in
/// `out`, a block the loop header dominates.
fn accumulate(b: &mut KirBuilder, inp: VarId, n: VarId, at: impl Fn(&mut KirBuilder, VarId) -> VarId) -> (VarId, BlockId) {
    use KirType::{F32, U64};
    let head = b.new_block();
    let body = b.new_block();
    let out = b.new_block();
    let zero = konst(b, U64, ConstValue::U64(0));
    let acc0 = konst(b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, acc0])));
    let r = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);

    b.set_block(head);
    let end = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(end, r, n, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(out), KirEdge::to(body)));

    b.set_block(body);
    let a = at(b, r);
    let x = load_f32(b, inp, a);
    let acc_next = op2(b, F32, KirOp::AddRn, acc, x);
    let one = konst(b, U64, ConstValue::U64(1));
    let r_next = op2(b, U64, KirOp::Add, r, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![r_next, acc_next])));

    b.set_block(out);
    (acc, out)
}

fn build_global() -> KernelIR {
    use AddressSpace::Global;
    let names = DetSumOp::Global.param_names();
    let mut b = KirBuilder::new(DetSumOp::Global.kernel_name());
    let inp = b.add_param(names[0], f32_ptr(), Global);
    let out = b.add_param(names[1], f32_ptr(), Global);
    let len = b.add_param(names[2], KirType::U64, Global);
    let entry = b.new_block();
    b.set_block(entry);
    let (acc, _) = accumulate(&mut b, inp, len, |_, r| r);
    b.emit(KirOp::Store(out, acc, Global));
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([DET_SUM_BLOCK, 1, 1]);
    b.finalize()
}

/// `Dim` (`t = %ctaid.x`) or `DimShort` (`t` = the global thread index).
fn build_dim(op: DetSumOp) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{U32, U64};
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let inp = b.add_param(names[0], f32_ptr(), Global);
    let out = b.add_param(names[1], f32_ptr(), Global);
    let outer = b.add_param(names[2], U64, Global);
    let reduce = b.add_param(names[3], U64, Global);
    let inner = b.add_param(names[4], U64, Global);
    let entry = b.new_block();
    let setup = b.new_block();
    let exit = b.new_block();

    b.set_block(entry);
    let t32 = b.new_typed_var(U32);
    b.emit(match op {
        DetSumOp::DimShort => KirOp::GlobalId(t32, 0),
        _ => KirOp::BlockIdx(t32, 0),
    });
    let t = b.new_typed_var(U64);
    b.emit(KirOp::Cast(t, t32, U64));
    let total = op2(&mut b, U64, KirOp::Mul, outer, inner);
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, t, total, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(setup)));

    b.set_block(setup);
    let o = op2(&mut b, U64, KirOp::Div, t, inner);
    let i = op2(&mut b, U64, KirOp::Rem, t, inner);
    let base = op2(&mut b, U64, KirOp::Mul, o, reduce);
    let base = op2(&mut b, U64, KirOp::Mul, base, inner);
    let base = op2(&mut b, U64, KirOp::Add, base, i);
    let (acc, _) = accumulate(&mut b, inp, reduce, |b, r| {
        let step = op2(b, U64, KirOp::Mul, r, inner);
        op2(b, U64, KirOp::Add, base, step)
    });
    store_f32(&mut b, out, t, acc);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([op.block(), 1, 1]);
    b.finalize()
}

/// Build `op` as KIR.
pub fn build(op: DetSumOp) -> KernelIR {
    match op {
        DetSumOp::Global => build_global(),
        DetSumOp::Dim | DetSumOp::DimShort => build_dim(op),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: DetSumOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: DetSumOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in DetSumOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}: NUL-terminated");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
            assert!(!p.contains(".maxntid"), "{p}");
        }
    }

    /// One load and one explicitly rounded add per iteration. Only the
    /// short kernel reads a thread index: the deterministic ones' result is
    /// the block's (or the launch's).
    #[test]
    fn the_loop_is_one_rounded_add_per_element() {
        for op in DetSumOp::ALL {
            let p = text(op);
            assert_eq!(p.matches("ld.global.f32").count(), 1, "{p}");
            assert_eq!(p.matches("add.rn.f32").count(), 1, "{p}");
            assert!(!p.contains("fma"), "{p}");
            assert_eq!(p.contains("%tid"), op == DetSumOp::DimShort, "{p}");
        }
        assert!(!text(DetSumOp::Global).contains("%ctaid"));
        assert!(text(DetSumOp::Dim).contains("%ctaid.x"));
        assert!(text(DetSumOp::DimShort).contains("%ctaid.x") && text(DetSumOp::DimShort).contains("%ntid.x"));
    }

    /// The loop's bound test branches straight out, with no edge copies
    /// printed around it.
    #[test]
    fn the_loop_exit_carries_no_copies() {
        for op in DetSumOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
