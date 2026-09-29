// crates/nsl-kir/src/kernels/softmax.rs
//! The runtime's row softmax and log-softmax from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5): `nsl_softmax_f32(inp, out, rows, cols)` and
//! `nsl_log_softmax_f32(...)`.
//!
//! One block of [`SOFTMAX_BLOCK`] threads per row (`%ctaid.x`); a block at
//! or past `rows` does nothing. Each thread strides over the row's columns
//! (`k = tid, tid + 256, …`) in three passes:
//!
//! 1. **Max.** Each thread folds its columns with `max.f32` from `-inf` and
//!    stores its partial to `smax[tid]`. After a barrier, thread 0 folds
//!    `smax[1 .. %ntid.x]` into its own partial, in order, and stores the row
//!    max to `smax[0]`. Every thread reads it back after a second barrier.
//! 2. **Sum.** Each thread adds `e = 2^((x - max) · log2 e)` over its columns
//!    (`sub`, `mul`, then a bare `ex2.approx`). The softmax also stores `e` to
//!    `out`. Thread 0 folds `ssum[1 .. %ntid.x]` the same way, then stores
//!    `1 / sum` (`rcp.approx`, softmax) or `ln sum = lg2(sum) · ln 2`
//!    (log-softmax) to `ssum[0]`.
//! 3. **Finish.** Softmax: `out[k] *= 1 / sum`. Log-softmax: `out[k] = (x -
//!    max) - ln sum`.
//!
//! Every add, subtract and multiply rounds explicitly (`.rn`); none of the
//! hand kernel's pairs could contract (each multiply feeds `ex2`, a store or
//! a shared-memory round trip, never an add). The order of every combine is
//! the hand kernel's, so the result is the same bit for bit. Each loop
//! leaves its header for a block the header dominates. Thread 0 reads
//! `%ntid.x` once, before its fold: read in the fold's header, as the first
//! draft did, ptxas left the fold rolled for sm_90 and sm_120 where it
//! unrolls the hand kernel's.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// The block the runtime launches with: one per row. The column loops
/// stride by it.
pub const SOFTMAX_BLOCK: u32 = 256;

/// `log2 e` and `ln 2` as the hand kernels spell them.
const LOG2_E: f32 = f32::from_bits(0x3FB8_AA3B);
const LN_2: f32 = f32::from_bits(0x3F31_7218);

/// Which kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoftmaxOp {
    /// `out = exp(x - max) / Σ exp(x - max)`.
    Softmax,
    /// `out = (x - max) - ln Σ exp(x - max)`.
    LogSoftmax,
}

impl SoftmaxOp {
    pub const ALL: [SoftmaxOp; 2] = [SoftmaxOp::Softmax, SoftmaxOp::LogSoftmax];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            SoftmaxOp::Softmax => "nsl_softmax_f32",
            SoftmaxOp::LogSoftmax => "nsl_log_softmax_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub const PARAMS: [&'static str; 4] = ["inp", "out", "rows", "cols"];
}

fn var(b: &mut KirBuilder, ty: KirType) -> VarId {
    b.new_typed_var(ty)
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn op1(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId) -> KirOp, x: VarId) -> VarId {
    let dst = var(b, ty);
    b.emit(op(dst, x));
    dst
}

fn op2(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = var(b, ty);
    b.emit(op(dst, x, y));
    dst
}

fn cmp(b: &mut KirBuilder, x: VarId, y: VarId, op: CmpOp) -> VarId {
    let dst = var(b, KirType::Bool);
    b.emit(KirOp::Cmp(dst, x, y, op));
    dst
}

fn load_at(b: &mut KirBuilder, space: AddressSpace, base: VarId, i: VarId) -> VarId {
    let addr = var(b, KirType::Ptr(Box::new(KirType::F32), space));
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = var(b, KirType::F32);
    b.emit(KirOp::Load(v, addr, space));
    v
}

fn store_at(b: &mut KirBuilder, space: AddressSpace, base: VarId, i: VarId, v: VarId) {
    let addr = var(b, KirType::Ptr(Box::new(KirType::F32), space));
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, v, space));
}

/// A column loop from the current block: `k = tid, tid + 256, … < cols`,
/// carrying `acc` (if any) through `body(b, k, acc) -> acc'`. Returns the
/// block the loop exits to, current on return, and the exit's `acc`.
fn column_loop(
    b: &mut KirBuilder,
    tid: VarId,
    cols: VarId,
    acc0: Option<VarId>,
    body: impl FnOnce(&mut KirBuilder, VarId, Option<VarId>) -> Option<VarId>,
) -> Option<VarId> {
    use KirType::{F32, U64};
    let head = b.new_block();
    let step = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let acc = acc0.map(|_| b.add_block_param(head, F32));
    let mut args = vec![tid];
    args.extend(acc0);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, args)));

    b.set_block(head);
    let end = cmp(b, k, cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(exit), KirEdge::to(step)));

    b.set_block(step);
    let acc_next = body(b, k, acc);
    let stride = konst(b, U64, ConstValue::U64(u64::from(SOFTMAX_BLOCK)));
    let k_next = op2(b, U64, KirOp::Add, k, stride);
    let mut args = vec![k_next];
    args.extend(acc_next);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, args)));

    b.set_block(exit);
    acc
}

/// From the current block, holding the thread's `partial`: store it to
/// `sm[tid]`, barrier; thread 0 folds `sm[1 .. %ntid.x]` into it with
/// `combine` and stores `finish(total)` to `sm[0]`; barrier; everyone reads
/// `sm[0]`, which is returned (the current block is then the one after).
fn block_fold(
    b: &mut KirBuilder,
    sm: VarId,
    tid32: VarId,
    partial: VarId,
    combine: fn(VarId, VarId, VarId) -> KirOp,
    finish: impl FnOnce(&mut KirBuilder, VarId) -> VarId,
) -> VarId {
    use AddressSpace::Shared;
    use KirType::{F32, U32};
    let first = b.new_block();
    let head = b.new_block();
    let step = b.new_block();
    let done = b.new_block();
    let after = b.new_block();
    let i = b.add_block_param(head, U32);
    let acc = b.add_block_param(head, F32);

    store_at(b, Shared, sm, tid32, partial);
    b.emit(KirOp::Barrier);
    let zero = konst(b, U32, ConstValue::U32(0));
    let not_first = cmp(b, tid32, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(after), KirEdge::to(first)));

    b.set_block(first);
    let ntid = var(b, U32);
    b.emit(KirOp::BlockDim(ntid, 0));
    let one = konst(b, U32, ConstValue::U32(1));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![one, partial])));

    b.set_block(head);
    let end = cmp(b, i, ntid, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(step)));

    b.set_block(step);
    let other = load_at(b, Shared, sm, i);
    let acc_next = op2(b, F32, combine, acc, other);
    let one = konst(b, U32, ConstValue::U32(1));
    let i_next = op2(b, U32, KirOp::Add, i, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, acc_next])));

    b.set_block(done);
    let total = finish(b, acc);
    b.emit(KirOp::Store(sm, total, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(after)));

    b.set_block(after);
    b.emit(KirOp::Barrier);
    let v = var(b, F32);
    b.emit(KirOp::Load(v, sm, Shared));
    v
}

/// ```text
/// entry: row = %ctaid.x; if row >= rows { exit }
/// setup: in = inp + row·cols; o = out + row·cols
/// pass 1: m = fold max over in[k];  mx = block_fold(smax, max)
/// pass 2: e = ex2((in[k] - mx) · log2 e) [softmax: o[k] = e]; s += e
///         v = block_fold(ssum, add, then rcp (softmax) or lg2 · ln 2)
/// pass 3: softmax: o[k] = o[k] · v;  log-softmax: o[k] = (in[k] - mx) - v
/// ```
pub fn build(op: SoftmaxOp) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(op.kernel_name());
    let inp = b.add_param(SoftmaxOp::PARAMS[0], f32_ptr(), Global);
    let out = b.add_param(SoftmaxOp::PARAMS[1], f32_ptr(), Global);
    let rows = b.add_param(SoftmaxOp::PARAMS[2], U64, Global);
    let cols = b.add_param(SoftmaxOp::PARAMS[3], U64, Global);
    let region = |name: &str| SmemRegion { name: name.to_string(), bytes: SOFTMAX_BLOCK * 4, align: 4, elem: F32 };
    b.set_smem_layout(SmemLayout { regions: vec![region("smax"), region("ssum")], dynamic: false });

    let entry = b.new_block();
    let setup = b.new_block();
    let exit: BlockId = b.new_block();

    b.set_block(entry);
    let row32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(row32, 0));
    let row = var(&mut b, U64);
    b.emit(KirOp::Cast(row, row32, U64));
    let past = cmp(&mut b, row, rows, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(setup)));

    b.set_block(setup);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = var(&mut b, U64);
    b.emit(KirOp::Cast(tid, tid32, U64));
    let base = op2(&mut b, U64, KirOp::Mul, row, cols);
    let in_row = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(in_row, inp, base));
    let out_row = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(out_row, out, base));
    let smax = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: smax, region: 0 });
    let ssum = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: ssum, region: 1 });
    let neg_inf = konst(&mut b, F32, ConstValue::F32(f32::NEG_INFINITY));

    // Pass 1: the row max.
    let m = column_loop(&mut b, tid, cols, Some(neg_inf), |b, k, acc| {
        let x = load_at(b, Global, in_row, k);
        Some(op2(b, F32, KirOp::Max, acc.expect("acc"), x))
    })
    .expect("acc");
    let mx = block_fold(&mut b, smax, tid32, m, KirOp::Max, |_, t| t);

    // Pass 2: the exponentials and their sum.
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    let s = column_loop(&mut b, tid, cols, Some(zero), |b, k, acc| {
        let x = load_at(b, Global, in_row, k);
        let d = op2(b, F32, KirOp::SubRn, x, mx);
        let log2e = konst(b, F32, ConstValue::F32(LOG2_E));
        let scaled = op2(b, F32, KirOp::MulRn, d, log2e);
        let e = op1(b, F32, KirOp::Exp2, scaled);
        if op == SoftmaxOp::Softmax {
            store_at(b, Global, out_row, k, e);
        }
        Some(op2(b, F32, KirOp::AddRn, acc.expect("acc"), e))
    })
    .expect("acc");
    let v = block_fold(&mut b, ssum, tid32, s, KirOp::AddRn, |b, total| match op {
        SoftmaxOp::Softmax => op1(b, F32, KirOp::Rcp, total),
        SoftmaxOp::LogSoftmax => {
            let l2 = op1(b, F32, KirOp::Log2, total);
            let ln2 = konst(b, F32, ConstValue::F32(LN_2));
            op2(b, F32, KirOp::MulRn, l2, ln2)
        }
    });

    // Pass 3: the finish.
    column_loop(&mut b, tid, cols, None, |b, k, _| {
        let y = match op {
            SoftmaxOp::Softmax => {
                let e = load_at(b, Global, out_row, k);
                op2(b, F32, KirOp::MulRn, e, v)
            }
            SoftmaxOp::LogSoftmax => {
                let x = load_at(b, Global, in_row, k);
                let d = op2(b, F32, KirOp::SubRn, x, mx);
                op2(b, F32, KirOp::SubRn, d, v)
            }
        };
        store_at(b, Global, out_row, k, y);
        None
    });
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([SOFTMAX_BLOCK, 1, 1]);
    b.set_launch_bounds(SOFTMAX_BLOCK, None);
    b.finalize()
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: SoftmaxOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: SoftmaxOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in SoftmaxOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in SoftmaxOp::PARAMS {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
            assert!(p.contains(".shared .align 4 .b8 shared_mem[2048]"), "{p}");
        }
    }

    /// The exponential is a bare `ex2` after an explicit multiply by
    /// `log2 e`; the finish is `rcp.approx` or `lg2 · ln 2`; four barriers.
    #[test]
    fn the_arithmetic_follows_the_hand_kernels() {
        for op in SoftmaxOp::ALL {
            let p = text(op);
            assert_eq!(p.matches("ex2.approx.f32").count(), 1, "{p}");
            assert_eq!(p.matches("0f3FB8AA3B").count(), 1, "{p}");
            assert_eq!(p.matches("bar.sync 0;").count(), 4, "{p}");
            assert_eq!(p.matches("max.f32").count(), 2, "{p}");
            assert!(!p.contains("fma") && !p.contains("add.f32") && !p.contains("mul.f32 "), "{p}");
            let (rcp, lg2) = (p.contains("rcp.approx.f32"), p.contains("lg2.approx.f32"));
            assert_eq!((rcp, lg2), (op == SoftmaxOp::Softmax, op == SoftmaxOp::LogSoftmax), "{p}");
            assert_eq!(p.contains("0f3F317218"), op == SoftmaxOp::LogSoftmax, "{p}");
        }
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        for op in SoftmaxOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
