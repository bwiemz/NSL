// crates/nsl-kir/src/kernels/tensor_stats.rs
//! The runtime's single-block tensor statistics from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5):
//! `nsl_tensor_stats_f32(inp, out, n)` writes `out[0..4] = [min, max, Σx,
//! Σx²]` over `inp[0..n]`.
//!
//! It is launched as one block of [`TENSOR_STATS_BLOCK`] threads. Thread `k`
//! folds `inp[k], inp[k + 256], …` into four partials, starting from `+inf`,
//! `-inf`, `+0.0` and `+0.0`. The partials go to four shared arrays and are
//! combined by one tree (`s[k] ⊕= s[k + h]` for `h = 128, …, 1`, a barrier
//! after each level, the four statistics in that order at each step).
//! Thread 0 then writes `s[0]` of each array. The order of every combine is
//! the hand kernel's.
//!
//! The sums round explicitly (`add.rn.f32`, and `mul.rn.f32` for the
//! square). The hand kernel's `mul.f32` + `add.f32` let ptxas contract the
//! square into its accumulate as one `fma` (an `FFMA` per element on
//! sm_80/90/120), so on hardware its `Σx²` rounded once per element where
//! the PTX reads twice. `nsl_muon_batch_sumsq_f32` squares and adds with
//! two roundings and is documented to be bit-identical to this sum (the
//! sequential Frobenius path). With the rounding explicit here, that holds
//! on hardware too, and the result no longer depends on what ptxas decides.
//! `min.f32` and `max.f32` are the hand kernel's.
//!
//! The loop leaves its header for a block the header dominates, so the exit
//! edge carries no copies.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// The block the kernel is launched with (one block). The loop strides by
/// it and each shared array holds one partial per thread.
pub const TENSOR_STATS_BLOCK: u32 = 256;

/// The `.visible .entry` name. Pinned: the runtime launches by it.
pub const KERNEL_NAME: &str = "nsl_tensor_stats_f32";

/// The parameter names, in launch order. Pinned: the hand module's.
pub const PARAM_NAMES: [&str; 3] = ["inp", "out", "n"];

/// The four statistics, in the order of `out` and of the shared arrays.
const STATS: [&str; 4] = ["smin", "smax", "ssum", "ssq"];

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

/// How statistic `i` combines two values: min, max, then two sums.
fn combine(i: usize) -> fn(VarId, VarId, VarId) -> KirOp {
    [KirOp::Min, KirOp::Max, KirOp::AddRn, KirOp::AddRn][i]
}

/// ```text
/// entry: tid; head(k = tid, mn = +inf, mx = -inf, s = 0, q = 0)
/// head(k, mn, mx, s, q): if k >= n { red }
/// body:  x = inp[k]; mn = min(mn, x); mx = max(mx, x); s += x; q += x·x;
///        k += 256
/// red:   smin[tid], smax[tid], ssum[tid], ssq[tid] = mn, mx, s, q; barrier
/// tree(h = 128): if h < 1 { done }
///        if tid < h { each array: a[tid] ⊕= a[tid + h] }; barrier; h >>= 1
/// done:  if tid == 0 { out[0..4] = smin[0], smax[0], ssum[0], ssq[0] }
/// ```
pub fn build() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(KERNEL_NAME);
    let inp = b.add_param(PARAM_NAMES[0], f32_ptr(), Global);
    let out = b.add_param(PARAM_NAMES[1], f32_ptr(), Global);
    let n = b.add_param(PARAM_NAMES[2], U64, Global);
    b.set_smem_layout(SmemLayout {
        regions: STATS
            .iter()
            .map(|name| SmemRegion { name: name.to_string(), bytes: TENSOR_STATS_BLOCK * 4, align: 4, elem: F32 })
            .collect(),
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
    let acc: Vec<VarId> = (0..4).map(|_| b.add_block_param(head, F32)).collect();
    let half = b.add_block_param(tree, U32);

    b.set_block(entry);
    let tid = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let start = cast(&mut b, tid, U64);
    let sm: Vec<VarId> = (0..STATS.len())
        .map(|i| {
            let p = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
            b.emit(KirOp::SharedRegion { dst: p, region: i as u32 });
            p
        })
        .collect();
    let init: Vec<VarId> = [f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0]
        .into_iter()
        .map(|v| konst(&mut b, F32, ConstValue::F32(v)))
        .collect();
    let mut args = vec![start];
    args.extend(init);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, args)));

    b.set_block(head);
    let end = cmp(&mut b, k, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(red), KirEdge::to(body)));

    b.set_block(body);
    let x = load_at(&mut b, Global, inp, k);
    let mn = op2(&mut b, F32, KirOp::Min, acc[0], x);
    let mx = op2(&mut b, F32, KirOp::Max, acc[1], x);
    let s = op2(&mut b, F32, KirOp::AddRn, acc[2], x);
    let sq = op2(&mut b, F32, KirOp::MulRn, x, x);
    let q = op2(&mut b, F32, KirOp::AddRn, acc[3], sq);
    let stride = konst(&mut b, U64, ConstValue::U64(u64::from(TENSOR_STATS_BLOCK)));
    let k_next = op2(&mut b, U64, KirOp::Add, k, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, mn, mx, s, q])));

    b.set_block(red);
    for (&p, &a) in sm.iter().zip(&acc) {
        store_at(&mut b, Shared, p, tid, a);
    }
    b.emit(KirOp::Barrier);
    let first = konst(&mut b, U32, ConstValue::U32(TENSOR_STATS_BLOCK / 2));
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![first])));

    b.set_block(tree);
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let over = cmp(&mut b, half, one, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(over, KirEdge::to(done), KirEdge::to(step)));

    b.set_block(step);
    let idle = cmp(&mut b, tid, half, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(idle, KirEdge::to(skip), KirEdge::to(add)));

    b.set_block(add);
    let other = op2(&mut b, U32, KirOp::Add, tid, half);
    for (i, &p) in sm.iter().enumerate() {
        let mine = load_at(&mut b, Shared, p, tid);
        let theirs = load_at(&mut b, Shared, p, other);
        let both = op2(&mut b, F32, combine(i), mine, theirs);
        store_at(&mut b, Shared, p, tid, both);
    }
    b.terminate(KirTerminator::Branch(KirEdge::to(skip)));

    b.set_block(skip);
    b.emit(KirOp::Barrier);
    let next = op2(&mut b, U32, KirOp::Shr, half, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![next])));

    b.set_block(done);
    let zero = konst(&mut b, U32, ConstValue::U32(0));
    let not_first = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(exit), KirEdge::to(write)));

    b.set_block(write);
    for (i, &p) in sm.iter().enumerate() {
        let total = var(&mut b, F32);
        b.emit(KirOp::Load(total, p, Shared));
        let slot = konst(&mut b, U64, ConstValue::U64(i as u64));
        store_at(&mut b, Global, out, slot, total);
    }
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([TENSOR_STATS_BLOCK, 1, 1]);
    b.set_launch_bounds(TENSOR_STATS_BLOCK, None);
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
        assert!(p.contains(".shared .align 4 .b8 shared_mem[4096]"), "{p}");
    }

    /// Each statistic combines once in the loop and once in the tree; the
    /// sums and the square round explicitly, so nothing fuses.
    #[test]
    fn the_combines_round_explicitly() {
        let p = text();
        assert_eq!(p.matches("min.f32").count(), 2, "{p}");
        assert_eq!(p.matches("max.f32").count(), 2, "{p}");
        assert_eq!(p.matches("add.rn.f32").count(), 4, "{p}");
        assert_eq!(p.matches("mul.rn.f32").count(), 1, "{p}");
        assert_eq!(p.matches("ld.global.f32").count(), 1, "{p}");
        assert_eq!(p.matches("st.global.f32").count(), 4, "{p}");
        assert_eq!(p.matches("bar.sync 0;").count(), 2, "{p}");
        assert!(!p.contains("fma") && !p.contains("add.f32") && !p.contains("mul.f32"), "{p}");
        assert!(p.contains("0f7F800000") && p.contains("0fFF800000"), "{p}");
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        let p = text();
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
