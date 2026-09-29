// crates/nsl-kir/src/kernels/ce_bwd.rs
//! The runtime's GPU cross-entropy backward kernels from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5). After the softmax
//! kernel, two launches turn the softmax output into the gradient in place,
//! with no host readback:
//!
//! - `nsl_ce_bwd_count_f32(tgt, n, tgt_i32, scratch)`: one block of
//!   [`CE_BWD_BLOCK`] threads counts the valid targets (`t >= 0`) among
//!   `tgt[0..n]` and writes `max(count, 1)` as f32 to `scratch[0]`. Thread
//!   `k` counts `k, k + 256, …`; the partial counts are summed by a
//!   shared-memory tree, and thread 0 writes the result.
//! - `nsl_ce_bwd_finish_f32(sm, tgt, scratch, gop, go_imm, go_mode, tgt_i32,
//!   total, cols)`: a thread per element of the `[rows, cols]` softmax `sm`.
//!   Element `(i, j)` becomes `0` when `t_i < 0`, and otherwise
//!   `(sm[i, j] - [j == t_i]) · (go / scratch[0])`, where `go` is `gop[0]`
//!   when `go_mode == 1` and `go_imm` otherwise. `gop` is not read in the
//!   second case: the host passes a null pointer then.
//!
//! A target is read as f32 and truncated toward zero (`cvt.rzi.s32.f32`),
//! or as s32 when `tgt_i32 != 0`: the CPU arm's `read_index`. The hand
//! kernels branched between the two loads. These load the four bytes once
//! and choose between the two readings with a `selp`; the value is the
//! same.
//!
//! The float arithmetic is explicitly rounded (`sub.rn`, `div.rn`,
//! `mul.rn`), as the hand kernel's was, so ptxas contracts nothing: the
//! gradient matches the CPU arm's separately rounded steps. The one-hot
//! subtraction is computed for every valid element and selected, where the
//! hand kernel branched around it; the selected value is the same.
//!
//! The count loop leaves its header for a block the header dominates, so
//! the exit edge carries no copies, and the finish kernel's `go` join is
//! entered by two unconditional edges.

use super::elementwise::{f32_ptr, finish, verified_ptx, ELEMENTWISE_BLOCK};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    RoundMode, SmemLayout, SmemRegion, VarId,
};

/// The block both kernels are launched with. The count kernel strides by
/// it and sizes its shared buffer to it.
pub const CE_BWD_BLOCK: u32 = ELEMENTWISE_BLOCK;

/// Which kernel of the pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CeBwdOp {
    /// `scratch[0] = max(#valid targets, 1)` as f32.
    Count,
    /// The in-place gradient over the softmax output.
    Finish,
}

impl CeBwdOp {
    pub const ALL: [CeBwdOp; 2] = [CeBwdOp::Count, CeBwdOp::Finish];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            CeBwdOp::Count => "nsl_ce_bwd_count_f32",
            CeBwdOp::Finish => "nsl_ce_bwd_finish_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            CeBwdOp::Count => &["tgt", "n", "tgt_i32", "scratch"],
            CeBwdOp::Finish => &["sm", "tgt", "scratch", "gop", "go_imm", "go_mode", "tgt_i32", "total", "cols"],
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

fn select(b: &mut KirBuilder, ty: KirType, c: VarId, t: VarId, f: VarId) -> VarId {
    let dst = var(b, ty);
    b.emit(KirOp::Select(dst, c, t, f));
    dst
}

/// `base[i]` for an element type `elem` in `space` (`i` any integer width).
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

/// Target `tgt[i]` as s32: the four bytes read as f32 and truncated toward
/// zero, or read as s32 when `as_i32`.
fn target(b: &mut KirBuilder, tgt: VarId, i: VarId, as_i32: VarId) -> VarId {
    use KirType::{F32, I32, U32};
    let raw = load_at(b, U32, AddressSpace::Global, tgt, i);
    let f = var(b, F32);
    b.emit(KirOp::Bitcast(f, raw));
    let from_f = var(b, I32);
    b.emit(KirOp::CastRounded { dst: from_f, src: f, ty: I32, mode: RoundMode::Rz });
    let from_i = var(b, I32);
    b.emit(KirOp::Bitcast(from_i, raw));
    select(b, I32, as_i32, from_i, from_f)
}

fn tgt_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global)
}

/// ```text
/// entry: tid; as_i32 = tgt_i32 != 0
/// head(k, cnt): if k >= n { red }
/// body:  cnt += tgt[k] >= 0 ? 1 : 0; k += 256
/// red:   cnt_smem[tid] = cnt; barrier
/// tree(h = 128): if h < 1 { done }
///        if tid < h { cnt_smem[tid] += cnt_smem[tid + h] }; barrier; h >>= 1
/// done:  if tid == 0 { scratch[0] = f32(max(cnt_smem[0], 1)) }
/// ```
fn build_count() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, I32, U32};
    let names = CeBwdOp::Count.param_names();
    let mut b = KirBuilder::new(CeBwdOp::Count.kernel_name());
    let tgt = b.add_param(names[0], tgt_ptr(), Global);
    let n = b.add_param(names[1], U32, Global);
    let tgt_i32 = b.add_param(names[2], U32, Global);
    let scratch = b.add_param(names[3], f32_ptr(), Global);
    b.set_smem_layout(SmemLayout {
        regions: vec![SmemRegion { name: "cnt".to_string(), bytes: CE_BWD_BLOCK * 4, align: 4, elem: U32 }],
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
    let k = b.add_block_param(head, U32);
    let cnt = b.add_block_param(head, U32);
    let half = b.add_block_param(tree, U32);

    b.set_block(entry);
    let tid = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let u_zero = konst(&mut b, U32, ConstValue::U32(0));
    let as_i32 = cmp(&mut b, tgt_i32, u_zero, CmpOp::Ne);
    let sm = var(&mut b, KirType::Ptr(Box::new(U32), Shared));
    b.emit(KirOp::SharedRegion { dst: sm, region: 0 });
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![tid, u_zero])));

    b.set_block(head);
    let end = cmp(&mut b, k, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(red), KirEdge::to(body)));

    b.set_block(body);
    let t = target(&mut b, tgt, k, as_i32);
    let s_zero = konst(&mut b, I32, ConstValue::I32(0));
    let valid = cmp(&mut b, t, s_zero, CmpOp::Ge);
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let zero = konst(&mut b, U32, ConstValue::U32(0));
    let inc = select(&mut b, U32, valid, one, zero);
    let cnt_next = op2(&mut b, U32, KirOp::Add, cnt, inc);
    let stride = konst(&mut b, U32, ConstValue::U32(CE_BWD_BLOCK));
    let k_next = op2(&mut b, U32, KirOp::Add, k, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, cnt_next])));

    b.set_block(red);
    store_at(&mut b, U32, Shared, sm, tid, cnt);
    b.emit(KirOp::Barrier);
    let first = konst(&mut b, U32, ConstValue::U32(CE_BWD_BLOCK / 2));
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![first])));

    b.set_block(tree);
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let over = cmp(&mut b, half, one, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(over, KirEdge::to(done), KirEdge::to(step)));

    b.set_block(step);
    let idle = cmp(&mut b, tid, half, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(idle, KirEdge::to(skip), KirEdge::to(add)));

    b.set_block(add);
    let mine = load_at(&mut b, U32, Shared, sm, tid);
    let other = op2(&mut b, U32, KirOp::Add, tid, half);
    let theirs = load_at(&mut b, U32, Shared, sm, other);
    let sum = op2(&mut b, U32, KirOp::Add, mine, theirs);
    store_at(&mut b, U32, Shared, sm, tid, sum);
    b.terminate(KirTerminator::Branch(KirEdge::to(skip)));

    b.set_block(skip);
    b.emit(KirOp::Barrier);
    let next = op2(&mut b, U32, KirOp::Shr, half, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![next])));

    b.set_block(done);
    let not_first = cmp(&mut b, tid, u_zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(exit), KirEdge::to(write)));

    b.set_block(write);
    let total = var(&mut b, U32);
    b.emit(KirOp::Load(total, sm, Shared));
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let denom = op2(&mut b, U32, KirOp::Max, total, one);
    let f = cast(&mut b, denom, F32);
    b.emit(KirOp::Store(scratch, f, Global));
    finish(b, exit)
}

/// ```text
/// entry: g = global id (u32); if g >= total { exit }
/// body:  i = g / cols; j = g - i·cols; t = tgt[i]
///        if t < 0 { zero } else { valid }
/// zero:  sm[g] = 0
/// valid: v = sm[g]; v = j == t ? v - 1 : v
///        go = go_mode == 1 ? gop[0] : go_imm      (gop read only then)
/// scale: sm[g] = v · (go / scratch[0])
/// ```
fn build_finish() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, I32, U32, U64};
    let names = CeBwdOp::Finish.param_names();
    let mut b = KirBuilder::new(CeBwdOp::Finish.kernel_name());
    let sm = b.add_param(names[0], f32_ptr(), Global);
    let tgt = b.add_param(names[1], tgt_ptr(), Global);
    let scratch = b.add_param(names[2], f32_ptr(), Global);
    let gop = b.add_param(names[3], f32_ptr(), Global);
    let go_imm = b.add_param(names[4], F32, Global);
    let go_mode = b.add_param(names[5], U32, Global);
    let tgt_i32 = b.add_param(names[6], U32, Global);
    let total = b.add_param(names[7], U32, Global);
    let cols = b.add_param(names[8], U32, Global);

    let entry = b.new_block();
    let body = b.new_block();
    let zero_blk = b.new_block();
    let valid_blk = b.new_block();
    let dev = b.new_block();
    let imm = b.new_block();
    let scale = b.new_block();
    let exit = b.new_block();
    let go = b.add_block_param(scale, F32);

    b.set_block(entry);
    let g = var(&mut b, U32);
    b.emit(KirOp::GlobalId(g, 0));
    let past = cmp(&mut b, g, total, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));

    b.set_block(body);
    let row = op2(&mut b, U32, KirOp::Div, g, cols);
    let base = op2(&mut b, U32, KirOp::Mul, row, cols);
    let col = op2(&mut b, U32, KirOp::Sub, g, base);
    let u_zero = konst(&mut b, U32, ConstValue::U32(0));
    let as_i32 = cmp(&mut b, tgt_i32, u_zero, CmpOp::Ne);
    let t = target(&mut b, tgt, row, as_i32);
    let at = cast(&mut b, g, U64);
    let s_zero = konst(&mut b, I32, ConstValue::I32(0));
    let invalid = cmp(&mut b, t, s_zero, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(invalid, KirEdge::to(zero_blk), KirEdge::to(valid_blk)));

    b.set_block(zero_blk);
    let f_zero = konst(&mut b, F32, ConstValue::F32(0.0));
    store_at(&mut b, F32, Global, sm, at, f_zero);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(valid_blk);
    let v = load_at(&mut b, F32, Global, sm, at);
    let tu = var(&mut b, U32);
    b.emit(KirOp::Bitcast(tu, t));
    let hit = cmp(&mut b, col, tu, CmpOp::Eq);
    let f_one = konst(&mut b, F32, ConstValue::F32(1.0));
    let v_hit = op2(&mut b, F32, KirOp::SubRn, v, f_one);
    let d = select(&mut b, F32, hit, v_hit, v);
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let from_dev = cmp(&mut b, go_mode, one, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(from_dev, KirEdge::to(dev), KirEdge::to(imm)));

    b.set_block(dev);
    let go_dev = var(&mut b, F32);
    b.emit(KirOp::Load(go_dev, gop, Global));
    b.terminate(KirTerminator::Branch(KirEdge::with(scale, vec![go_dev])));

    b.set_block(imm);
    b.terminate(KirTerminator::Branch(KirEdge::with(scale, vec![go_imm])));

    b.set_block(scale);
    let denom = var(&mut b, F32);
    b.emit(KirOp::Load(denom, scratch, Global));
    let q = op2(&mut b, F32, KirOp::Div, go, denom);
    let out = op2(&mut b, F32, KirOp::MulRn, d, q);
    store_at(&mut b, F32, Global, sm, at, out);
    finish(b, exit)
}

/// Build `op` as KIR.
pub fn build(op: CeBwdOp) -> KernelIR {
    match op {
        CeBwdOp::Count => build_count(),
        CeBwdOp::Finish => build_finish(),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: CeBwdOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: CeBwdOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in CeBwdOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}: NUL-terminated");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
        }
    }

    /// A target is loaded once and read both ways; the float steps are all
    /// explicitly rounded, and only the finish kernel does float arithmetic.
    #[test]
    fn targets_are_read_both_ways_and_the_float_steps_round() {
        for op in CeBwdOp::ALL {
            let p = text(op);
            assert_eq!(p.matches("ld.global.u32").count(), 1, "{p}");
            assert_eq!(p.matches("cvt.rzi.s32.f32").count(), 1, "{p}");
            assert!(p.contains("setp.ge.s32") || p.contains("setp.lt.s32"), "{p}");
            assert!(!p.contains("fma") && !p.contains("approx"), "{p}");
        }
        let fin = text(CeBwdOp::Finish);
        for want in ["sub.rn.f32", "div.rn.f32", "mul.rn.f32", "div.u32"] {
            assert_eq!(fin.matches(want).count(), 1, "{want}\n{fin}");
        }
        let count = text(CeBwdOp::Count);
        assert_eq!(count.matches("bar.sync").count(), 2, "{count}");
        assert!(count.contains("max.u32") && count.contains("cvt.rn.f32.u32"), "{count}");
        assert!(!count.contains(".rn.f32 %f"), "{count}");
    }

    /// No loop exit or join edge prints copies inline around a branch.
    #[test]
    fn no_conditional_edge_carries_copies() {
        for op in CeBwdOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
