// crates/nsl-kir/src/kernels/sum_sq.rs
//! The runtime's f64-accumulated sum of squares from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5): `nsl_sum_sq_f64_acc_f32(inp, out, n)`.
//!
//! Gradient clipping compares `sqrt(Σ g²)` against `max_norm`, so the sum is
//! accumulated in f64: an f32 sum that runs low can decline a clip the host
//! path would make. The kernel is grid-strided on blocks of
//! [`SUM_SQ_BLOCK`] threads. Thread `k` of the launch (`%ctaid.x · %ntid.x +
//! %tid.x`) folds `inp[k], inp[k + s], …` for the stride `s = %ntid.x ·
//! %nctaid.x`, squaring each value in f64 and adding it with one rounding
//! (`fma.rn.f64`). The block's partials then take a shared-memory tree
//! (`s[t] += s[t + h]`, `h = 128 … 1`, a barrier per level), and thread 0
//! writes the block's partial to `out[%ctaid.x]`. The host sums the partials
//! in block order, so the result is the same run to run.
//!
//! The order of every add is the hand kernel's, so the partials are the same
//! bit for bit. The loop leaves its header for a block the header dominates,
//! so the exit edge carries no copies.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// The block the kernel is launched with. The shared buffer holds one f64
/// partial per thread.
pub const SUM_SQ_BLOCK: u32 = 256;

/// The `.visible .entry` name. Pinned: the runtime launches by it.
pub const KERNEL_NAME: &str = "nsl_sum_sq_f64_acc_f32";

/// The parameter names, in launch order. Pinned: the hand module's.
pub const PARAM_NAMES: [&str; 3] = ["inp", "out", "n"];

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

fn load_at(b: &mut KirBuilder, elem: KirType, space: AddressSpace, base: VarId, i: VarId) -> VarId {
    let addr = var(b, KirType::Ptr(Box::new(elem.clone()), space));
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = var(b, elem);
    b.emit(KirOp::Load(v, addr, space));
    v
}

fn store_at(b: &mut KirBuilder, elem: KirType, space: AddressSpace, base: VarId, i: VarId, v: VarId) {
    let addr = var(b, KirType::Ptr(Box::new(elem), space));
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, v, space));
}

/// ```text
/// entry: k = global id; s = %ntid.x · %nctaid.x; head(k, acc = 0.0)
/// head(k, acc): if k >= n { red }
/// body:  x = f64(inp[k]); acc = fma(x, x, acc); k += s
/// red:   ssq[tid] = acc; barrier
/// tree(h = 128): if h < 1 { done }
///        if tid < h { ssq[tid] += ssq[tid + h] }; barrier; h >>= 1
/// done:  if tid == 0 { out[%ctaid.x] = ssq[0] }
/// ```
pub fn build() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, F64, U32, U64};
    let mut b = KirBuilder::new(KERNEL_NAME);
    let inp = b.add_param(PARAM_NAMES[0], f32_ptr(), Global);
    let out = b.add_param(PARAM_NAMES[1], KirType::Ptr(Box::new(F64), Global), Global);
    let n = b.add_param(PARAM_NAMES[2], U64, Global);
    b.set_smem_layout(SmemLayout {
        regions: vec![SmemRegion { name: "ssq".to_string(), bytes: SUM_SQ_BLOCK * 8, align: 8, elem: F64 }],
        dynamic: false,
    });

    let entry = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let red = b.new_block();
    let tree = b.new_block();
    let step = b.new_block();
    let add = b.new_block();
    let skip = b.new_block();
    let done = b.new_block();
    let write = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F64);
    let half = b.add_block_param(tree, U32);

    b.set_block(entry);
    let g32 = var(&mut b, U32);
    b.emit(KirOp::GlobalId(g32, 0));
    let start = cast(&mut b, g32, U64);
    let ntid = var(&mut b, U32);
    b.emit(KirOp::BlockDim(ntid, 0));
    let nctaid = var(&mut b, U32);
    b.emit(KirOp::GridDim(nctaid, 0));
    let stride32 = op2(&mut b, U32, KirOp::Mul, ntid, nctaid);
    let stride = cast(&mut b, stride32, U64);
    let tid = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let sm = var(&mut b, KirType::Ptr(Box::new(F64), Shared));
    b.emit(KirOp::SharedRegion { dst: sm, region: 0 });
    let zero = konst(&mut b, F64, ConstValue::F64(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![start, zero])));

    b.set_block(head);
    let end = cmp(&mut b, k, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(red), KirEdge::to(body)));

    b.set_block(body);
    let x32 = load_at(&mut b, F32, Global, inp, k);
    let x = cast(&mut b, x32, F64);
    let acc_next = var(&mut b, F64);
    b.emit(KirOp::Fma(acc_next, x, x, acc));
    let k_next = op2(&mut b, U64, KirOp::Add, k, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, acc_next])));

    b.set_block(red);
    store_at(&mut b, F64, Shared, sm, tid, acc);
    b.emit(KirOp::Barrier);
    let first = konst(&mut b, U32, ConstValue::U32(SUM_SQ_BLOCK / 2));
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![first])));

    b.set_block(tree);
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let over = cmp(&mut b, half, one, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(over, KirEdge::to(done), KirEdge::to(step)));

    b.set_block(step);
    let idle = cmp(&mut b, tid, half, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(idle, KirEdge::to(skip), KirEdge::to(add)));

    b.set_block(add);
    let mine = load_at(&mut b, F64, Shared, sm, tid);
    let other = op2(&mut b, U32, KirOp::Add, tid, half);
    let theirs = load_at(&mut b, F64, Shared, sm, other);
    let both = op2(&mut b, F64, KirOp::AddRn, mine, theirs);
    store_at(&mut b, F64, Shared, sm, tid, both);
    b.terminate(KirTerminator::Branch(KirEdge::to(skip)));

    b.set_block(skip);
    b.emit(KirOp::Barrier);
    let next = op2(&mut b, U32, KirOp::Shr, half, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![next])));

    b.set_block(done);
    let u_zero = konst(&mut b, U32, ConstValue::U32(0));
    let not_first = cmp(&mut b, tid, u_zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(exit), KirEdge::to(write)));

    b.set_block(write);
    let total = var(&mut b, F64);
    b.emit(KirOp::Load(total, sm, Shared));
    let block = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(block, 0));
    store_at(&mut b, F64, Global, out, block, total);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([SUM_SQ_BLOCK, 1, 1]);
    b.set_launch_bounds(SUM_SQ_BLOCK, None);
    b.finalize()
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx() -> Vec<u8> {
    verified_ptx(build())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text() -> String {
        String::from_utf8(ptx()).expect("ASCII")
    }

    #[test]
    fn the_kernel_verifies_and_keeps_its_entry_and_parameters() {
        let p = text();
        assert!(p.ends_with('\0'));
        assert!(p.contains(&format!(".visible .entry {KERNEL_NAME}(")), "{p}");
        for name in PARAM_NAMES {
            assert!(p.contains(&format!("[param_{name}]")), "{name}\n{p}");
        }
        assert!(p.contains(".shared .align 8 .b8 shared_mem[2048]"), "{p}");
    }

    /// The square-accumulate is one f64 `fma`, the tree adds in f64, and the
    /// stride comes from the grid.
    #[test]
    fn the_accumulation_is_f64() {
        let p = text();
        assert_eq!(p.matches("cvt.f64.f32").count(), 1, "{p}");
        assert_eq!(p.matches("fma.rn.f64").count(), 1, "{p}");
        assert_eq!(p.matches("add.rn.f64").count(), 1, "{p}");
        assert_eq!(p.matches("st.global.f64").count(), 1, "{p}");
        assert_eq!(p.matches("bar.sync 0;").count(), 2, "{p}");
        assert!(p.contains("%nctaid.x") && p.contains("%ntid.x"), "{p}");
        assert!(!p.contains("fma.rn.f32") && !p.contains("add.rn.f32"), "{p}");
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        let p = text();
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
