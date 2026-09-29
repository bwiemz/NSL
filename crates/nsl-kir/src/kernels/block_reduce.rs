// crates/nsl-kir/src/kernels/block_reduce.rs
//! The runtime's shared-memory tree reductions from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5). Each is one block of
//! [`BLOCK_REDUCE_BLOCK`] threads per result:
//!
//! - `nsl_global_sum_f32(inp, out, n)`: the sum of `inp[0..n]` into
//!   `out[0]`, launched as one block.
//! - `nsl_sum_dim_f32(inp, out, outer, reduce_size, inner)` and
//!   `nsl_max_dim_f32(...)`: the sum (or max) over the middle axis of an
//!   `[outer, reduce_size, inner]` view, a block per output `t = %ctaid.x`
//!   (`t = o·inner + i`, reading `inp[(o·reduce_size + r)·inner + i]`). A
//!   block at or past `outer·inner` does nothing.
//!
//! Thread `k` folds `k, k + 256, …` into a partial from the identity (`+0.0`
//! or `-inf`); the partials are stored to shared memory and combined by a
//! tree (`s[k] ⊕= s[k + h]` for `h = 128, 64, …, 1`, a barrier after each
//! level), and thread 0 writes `s[0]`. The order of every combine is the
//! hand kernels', so the result is the same bit for bit.
//!
//! The sums add with `add.rn.f32`, which rounds as the hand kernels'
//! `add.f32` does and keeps a multiply from contracting into it; the max is
//! `max.f32`, as before. The per-dim loop carries its element offset and
//! steps it by `256 · inner`, where the hand kernel formed `k · inner` on
//! every trip. Each accumulation loop leaves its header for a block the
//! header dominates, so the exit edge carries no copies.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// The block every kernel is launched with. The loops stride by it and the
/// shared buffer holds one partial per thread.
pub const BLOCK_REDUCE_BLOCK: u32 = 256;

/// Which reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReduceOp {
    /// `out[0] = Σ inp[0..n]`.
    GlobalSum,
    /// The sum over the middle axis, a block per output.
    SumDim,
    /// The max over the middle axis, a block per output.
    MaxDim,
}

impl BlockReduceOp {
    pub const ALL: [BlockReduceOp; 3] = [BlockReduceOp::GlobalSum, BlockReduceOp::SumDim, BlockReduceOp::MaxDim];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            BlockReduceOp::GlobalSum => "nsl_global_sum_f32",
            BlockReduceOp::SumDim => "nsl_sum_dim_f32",
            BlockReduceOp::MaxDim => "nsl_max_dim_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            BlockReduceOp::GlobalSum => &["inp", "out", "n"],
            BlockReduceOp::SumDim | BlockReduceOp::MaxDim => &["inp", "out", "outer", "reduce_size", "inner"],
        }
    }

    fn is_max(self) -> bool {
        self == BlockReduceOp::MaxDim
    }

    /// The fold's identity: `+0.0` for a sum, `-inf` for the max.
    fn identity(self) -> f32 {
        if self.is_max() {
            f32::NEG_INFINITY
        } else {
            0.0
        }
    }

    fn combine(self) -> fn(VarId, VarId, VarId) -> KirOp {
        if self.is_max() {
            KirOp::Max
        } else {
            KirOp::AddRn
        }
    }
}

fn var(b: &mut KirBuilder, ty: KirType) -> VarId {
    b.new_typed_var(ty)
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn op2(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = var(b, ty);
    b.emit(op(dst, x, y));
    dst
}

fn cast(b: &mut KirBuilder, src: VarId, ty: KirType) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Cast(dst, src, ty));
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

/// From the current block, which holds the thread's `partial` (in a block
/// that the accumulation loop's header dominates): store it to `sm[tid]`,
/// run the tree, and have thread 0 store `sm[0]` to `out_at`. Ends in
/// `exit`, returning nothing.
fn tree_and_write(b: &mut KirBuilder, op: BlockReduceOp, sm: VarId, tid: VarId, partial: VarId, out_at: VarId, exit: BlockId) {
    use AddressSpace::{Global, Shared};
    use KirType::U32;
    let tree = b.new_block();
    let step = b.new_block();
    let add = b.new_block();
    let skip = b.new_block();
    let done = b.new_block();
    let write = b.new_block();
    let half = b.add_block_param(tree, U32);

    store_at(b, Shared, sm, tid, partial);
    b.emit(KirOp::Barrier);
    let first = konst(b, U32, ConstValue::U32(BLOCK_REDUCE_BLOCK / 2));
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![first])));

    b.set_block(tree);
    let one = konst(b, U32, ConstValue::U32(1));
    let over = cmp(b, half, one, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(over, KirEdge::to(done), KirEdge::to(step)));

    b.set_block(step);
    let idle = cmp(b, tid, half, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(idle, KirEdge::to(skip), KirEdge::to(add)));

    b.set_block(add);
    let mine = load_at(b, Shared, sm, tid);
    let other = op2(b, U32, KirOp::Add, tid, half);
    let theirs = load_at(b, Shared, sm, other);
    let both = op2(b, KirType::F32, op.combine(), mine, theirs);
    store_at(b, Shared, sm, tid, both);
    b.terminate(KirTerminator::Branch(KirEdge::to(skip)));

    b.set_block(skip);
    b.emit(KirOp::Barrier);
    let next = op2(b, U32, KirOp::Shr, half, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![next])));

    b.set_block(done);
    let zero = konst(b, U32, ConstValue::U32(0));
    let not_first = cmp(b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(exit), KirEdge::to(write)));

    b.set_block(write);
    let total = var(b, KirType::F32);
    b.emit(KirOp::Load(total, sm, Shared));
    b.emit(KirOp::Store(out_at, total, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
}

fn smem() -> SmemLayout {
    SmemLayout {
        regions: vec![SmemRegion {
            name: "sdata".to_string(),
            bytes: BLOCK_REDUCE_BLOCK * 4,
            align: 4,
            elem: KirType::F32,
        }],
        dynamic: false,
    }
}

/// ```text
/// entry: tid; head(k = tid, acc = 0)
/// head(k, acc): if k >= n { red }
/// body:  acc += inp[k]; k += 256
/// red:   the tree; thread 0: out[0] = sdata[0]
/// ```
fn build_global() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let op = BlockReduceOp::GlobalSum;
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let inp = b.add_param(names[0], f32_ptr(), Global);
    let out = b.add_param(names[1], f32_ptr(), Global);
    let n = b.add_param(names[2], U64, Global);
    b.set_smem_layout(smem());

    let entry = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let red = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);

    b.set_block(entry);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = cast(&mut b, tid32, U64);
    let sm = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: sm, region: 0 });
    let init = konst(&mut b, F32, ConstValue::F32(op.identity()));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![tid, init])));

    b.set_block(head);
    let end = cmp(&mut b, k, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(red), KirEdge::to(body)));

    b.set_block(body);
    let x = load_at(&mut b, Global, inp, k);
    let acc_next = op2(&mut b, F32, op.combine(), acc, x);
    let stride = konst(&mut b, U64, ConstValue::U64(u64::from(BLOCK_REDUCE_BLOCK)));
    let k_next = op2(&mut b, U64, KirOp::Add, k, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, acc_next])));

    b.set_block(red);
    tree_and_write(&mut b, op, sm, tid32, acc, out, exit);
    b.set_workgroup_size([BLOCK_REDUCE_BLOCK, 1, 1]);
    b.set_launch_bounds(BLOCK_REDUCE_BLOCK, None);
    b.finalize()
}

/// ```text
/// entry: t = %ctaid.x; if t >= outer·inner { exit }
/// setup: base = (t / inner)·reduce_size·inner + t % inner
///        head(k = tid, at = base + tid·inner, acc = identity)
/// head(k, at, acc): if k >= reduce_size { red }
/// body:  acc ⊕= inp[at]; k += 256; at += 256·inner
/// red:   the tree; thread 0: out[t] = sdata[0]
/// ```
fn build_dim(op: BlockReduceOp) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let inp = b.add_param(names[0], f32_ptr(), Global);
    let out = b.add_param(names[1], f32_ptr(), Global);
    let outer = b.add_param(names[2], U64, Global);
    let reduce = b.add_param(names[3], U64, Global);
    let inner = b.add_param(names[4], U64, Global);
    b.set_smem_layout(smem());

    let entry = b.new_block();
    let setup = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let red = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let at = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);

    b.set_block(entry);
    let t32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(t32, 0));
    let t = cast(&mut b, t32, U64);
    let total = op2(&mut b, U64, KirOp::Mul, outer, inner);
    let past = cmp(&mut b, t, total, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(setup)));

    b.set_block(setup);
    let o = op2(&mut b, U64, KirOp::Div, t, inner);
    let i = op2(&mut b, U64, KirOp::Rem, t, inner);
    let base = op2(&mut b, U64, KirOp::Mul, o, reduce);
    let base = op2(&mut b, U64, KirOp::Mul, base, inner);
    let base = op2(&mut b, U64, KirOp::Add, base, i);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = cast(&mut b, tid32, U64);
    let first = op2(&mut b, U64, KirOp::Mul, tid, inner);
    let at0 = op2(&mut b, U64, KirOp::Add, base, first);
    let stride = konst(&mut b, U64, ConstValue::U64(u64::from(BLOCK_REDUCE_BLOCK)));
    let step = op2(&mut b, U64, KirOp::Mul, stride, inner);
    let sm = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: sm, region: 0 });
    let init = konst(&mut b, F32, ConstValue::F32(op.identity()));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![tid, at0, init])));

    b.set_block(head);
    let end = cmp(&mut b, k, reduce, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(red), KirEdge::to(body)));

    b.set_block(body);
    let x = load_at(&mut b, Global, inp, at);
    let acc_next = op2(&mut b, F32, op.combine(), acc, x);
    let k_next = op2(&mut b, U64, KirOp::Add, k, stride);
    let at_next = op2(&mut b, U64, KirOp::Add, at, step);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, at_next, acc_next])));

    b.set_block(red);
    let out_at = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(out_at, out, t));
    tree_and_write(&mut b, op, sm, tid32, acc, out_at, exit);
    b.set_workgroup_size([BLOCK_REDUCE_BLOCK, 1, 1]);
    b.set_launch_bounds(BLOCK_REDUCE_BLOCK, None);
    b.finalize()
}

/// Build `op` as KIR.
pub fn build(op: BlockReduceOp) -> KernelIR {
    match op {
        BlockReduceOp::GlobalSum => build_global(),
        BlockReduceOp::SumDim | BlockReduceOp::MaxDim => build_dim(op),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: BlockReduceOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: BlockReduceOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in BlockReduceOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}: NUL-terminated");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
            assert!(p.contains(".shared .align 4 .b8 shared_mem[1024]"), "{p}");
        }
    }

    /// One global load, one combine in the loop and one in the tree, two
    /// barriers; the sums round explicitly and never fuse.
    #[test]
    fn the_combines_follow_the_op() {
        for op in BlockReduceOp::ALL {
            let p = text(op);
            assert_eq!(p.matches("ld.global.f32").count(), 1, "{p}");
            assert_eq!(p.matches("ld.shared.f32").count(), 3, "{p}");
            assert_eq!(p.matches("bar.sync 0;").count(), 2, "{p}");
            let (want, not) = if op.is_max() { ("max.f32", "add.rn.f32") } else { ("add.rn.f32", "max.f32") };
            assert_eq!(p.matches(want).count(), 2, "{p}");
            assert!(!p.contains(not) && !p.contains("fma"), "{p}");
            assert_eq!(p.contains("%ctaid.x"), op != BlockReduceOp::GlobalSum, "{p}");
        }
        assert!(text(BlockReduceOp::MaxDim).contains("0fFF800000"));
    }

    /// The per-dim loop carries its offset: no multiply inside it.
    #[test]
    fn no_conditional_edge_carries_copies() {
        for op in BlockReduceOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
