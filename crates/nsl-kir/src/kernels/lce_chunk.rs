// crates/nsl-kir/src/kernels/lce_chunk.rs
//! The GEMM-chunked fused linear cross-entropy's per-chunk kernels from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5). The large-vocab
//! fused linear-CE runs a cuBLAS GEMM per vocabulary chunk of `cols`
//! columns starting at `chunk_start`; these two are the glue around it:
//!
//! - `nsl_lce_chunk_stats_f32(logits, bias, targets, mstate, sstate,
//!   tlstate, rows, cols, chunk_start, has_bias)`: one block of
//!   [`LCE_CHUNK_BLOCK`] threads per row (`%ctaid.x`; a block at or past
//!   `rows` does nothing) folds the row's `[cols]` logits chunk into the
//!   running online-softmax state. With `val_j = logits[row, j] (+
//!   bias[chunk_start + j] when has_bias != 0)`:
//!   1. each thread folds `max` over `j = tid, tid + 256, …` from `-inf` and
//!      stores it to `sdata[tid]`; after a barrier thread 0 folds
//!      `sdata[1 .. 256]` into it in order, takes `m_new = max(m_old,
//!      block max)` with `m_old = mstate[row]`, and publishes `m_new` in
//!      `sdata[0]`; after a second barrier every thread reads it;
//!   2. each thread sums `ex2((val_j - m_new) · log2 e)` from `+0`; after a
//!      barrier (so no thread still reads `sdata[0]`) the partials go to
//!      `sdata`, and after another thread 0 sums `sdata[1 .. 256]` into its
//!      own in order, stores `sstate[row] = fma(s_old, ex2((m_old - m_new) ·
//!      log2 e), sum)` and `mstate[row] = m_new`, and, when the row's target
//!      `t` (an `s64`) lies in `[chunk_start, chunk_start + cols)`, stores
//!      `tlstate[row] = val_{t - chunk_start}`.
//! - `nsl_lce_chunk_dlogits_f32(logits, bias, targets, lse, dbias, rows,
//!   cols, chunk_start, scale, has_bias)`: grid-strided over the `rows ·
//!   cols` elements (`%nctaid.x` blocks of 256), overwriting the chunk in
//!   place with `dl = t_r < 0 ? 0 : (ex2((val - lse[r]) · log2 e) - [t_r ==
//!   chunk_start + j]) · scale`, where `r = idx / cols` and `j = idx - r ·
//!   cols`. When `has_bias != 0` each `dl` is also added into
//!   `dbias[chunk_start + j]` (`red.global.add.f32`).
//!
//! The bias is read only when `has_bias != 0` (the pointer may be null
//! then), behind a branch, as in the hand kernels. Every add, subtract and
//! multiply rounds explicitly (`.rn`): none of the hand kernels' pairs
//! could contract (each multiply feeds `ex2`, a store or the explicit
//! `fma`), so the result is the same. The stats kernel's thread-0 folds
//! run to the constant 256, as the hand kernel's did, and its one shared
//! buffer serves both passes behind the same four barriers. The dlogits
//! kernel selects the one-hot `- 1` where the hand kernel branched around
//! it; the selected value is the same. No conditional edge carries copies.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// The block both kernels are launched with. The stats kernel's column
/// loops stride by it and its thread-0 folds run to it.
pub const LCE_CHUNK_BLOCK: u32 = 256;

/// The stats kernel's register cap (`.maxnreg`): left alone, ptxas gives it
/// 34 on sm_80 where the hand kernel had 32, which would drop a 256-thread
/// block from 64 warps to 48. At 32 nothing spills.
pub const STATS_MAX_REGISTERS: u32 = 32;

/// `log2 e` as the hand kernels spell it (`0f3FB8AA3B`).
pub const LOG2_E_BITS: u32 = 0x3FB8_AA3B;

/// Which kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LceChunkOp {
    /// The per-chunk online-softmax state update.
    Stats,
    /// The per-chunk softmax gradient, in place.
    Dlogits,
}

impl LceChunkOp {
    pub const ALL: [LceChunkOp; 2] = [LceChunkOp::Stats, LceChunkOp::Dlogits];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            LceChunkOp::Stats => "nsl_lce_chunk_stats_f32",
            LceChunkOp::Dlogits => "nsl_lce_chunk_dlogits_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            LceChunkOp::Stats => {
                &["logits", "bias", "targets", "mstate", "sstate", "tlstate", "rows", "cols", "chunk_start", "has_bias"]
            }
            LceChunkOp::Dlogits => {
                &["logits", "bias", "targets", "lse", "dbias", "rows", "cols", "chunk_start", "scale", "has_bias"]
            }
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

fn load_at(b: &mut KirBuilder, ty: KirType, space: AddressSpace, base: VarId, i: VarId) -> VarId {
    let addr = var(b, KirType::Ptr(Box::new(ty.clone()), space));
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = var(b, ty);
    b.emit(KirOp::Load(v, addr, space));
    v
}

fn store_at(b: &mut KirBuilder, space: AddressSpace, base: VarId, i: VarId, v: VarId) {
    let addr = var(b, KirType::Ptr(Box::new(KirType::F32), space));
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, v, space));
}

/// `logits[k] (+ bias[kb] when has_bias)`: the bias is loaded only on its
/// branch. Leaves the builder in the join block.
fn value(b: &mut KirBuilder, logits: VarId, k: VarId, bias: VarId, kb: VarId, no_bias: VarId) -> VarId {
    use AddressSpace::Global;
    use KirType::F32;
    let with_bias = b.new_block();
    let join = b.new_block();
    let v = b.add_block_param(join, F32);
    let x = load_at(b, F32, Global, logits, k);
    let skip = b.new_block();
    b.terminate(KirTerminator::CondBranch(no_bias, KirEdge::to(skip), KirEdge::to(with_bias)));

    b.set_block(with_bias);
    let bv = load_at(b, F32, Global, bias, kb);
    let sum = op2(b, F32, KirOp::AddRn, x, bv);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![sum])));

    b.set_block(skip);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![x])));

    b.set_block(join);
    v
}

/// A column loop from the current block: `k = tid, tid + 256, … < cols`,
/// carrying an `f32` through `body(b, k, acc) -> acc'`. Returns the carried
/// value at the exit, whose block is current on return.
fn column_loop(
    b: &mut KirBuilder,
    tid: VarId,
    cols: VarId,
    acc0: VarId,
    body: impl FnOnce(&mut KirBuilder, VarId, VarId) -> VarId,
) -> VarId {
    use KirType::{F32, U64};
    let head = b.new_block();
    let step = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![tid, acc0])));

    b.set_block(head);
    let end = cmp(b, k, cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(exit), KirEdge::to(step)));

    b.set_block(step);
    let acc_next = body(b, k, acc);
    let stride = konst(b, U64, ConstValue::U64(u64::from(LCE_CHUNK_BLOCK)));
    let k_next = op2(b, U64, KirOp::Add, k, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, acc_next])));

    b.set_block(exit);
    acc
}

/// Thread 0's in-order fold of `sdata[1 .. 256]` into `partial` with
/// `combine`, from the current block. Returns the total; the builder is in
/// the block after the fold.
fn thread0_fold(b: &mut KirBuilder, sdata: VarId, partial: VarId, combine: fn(VarId, VarId, VarId) -> KirOp) -> VarId {
    use AddressSpace::Shared;
    use KirType::{F32, U32};
    let head = b.new_block();
    let step = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(head, U32);
    let acc = b.add_block_param(head, F32);
    let one = konst(b, U32, ConstValue::U32(1));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![one, partial])));

    b.set_block(head);
    let n = konst(b, U32, ConstValue::U32(LCE_CHUNK_BLOCK));
    let end = cmp(b, i, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(step)));

    b.set_block(step);
    let other = load_at(b, F32, Shared, sdata, i);
    let acc_next = op2(b, F32, combine, acc, other);
    let one = konst(b, U32, ConstValue::U32(1));
    let i_next = op2(b, U32, KirOp::Add, i, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, acc_next])));

    b.set_block(done);
    acc
}

/// `2^((x - m) · log2 e)`: a `sub.rn`, a `mul.rn` and a bare `ex2.approx`.
fn exp_shifted(b: &mut KirBuilder, x: VarId, m: VarId) -> VarId {
    use KirType::F32;
    let d = op2(b, F32, KirOp::SubRn, x, m);
    let log2_e = konst(b, F32, ConstValue::F32(f32::from_bits(LOG2_E_BITS)));
    let scaled = op2(b, F32, KirOp::MulRn, d, log2_e);
    op1(b, F32, KirOp::Exp2, scaled)
}

/// ```text
/// entry:  row = %ctaid.x; if row >= rows { exit }
/// setup:  lr = logits + row·cols; bc = bias + chunk_start
/// pass 1: m = max over val(k); sdata[tid] = m; bar
///         tid 0: m = fold max sdata[1..256]; m_old = mstate[row];
///                sdata[0] = max(m_old, m)
///         bar; mn = sdata[0]
/// pass 2: s = Σ ex2((val(k) - mn) · log2 e); bar; sdata[tid] = s; bar
///         tid 0: s = fold add sdata[1..256]
///                sstate[row] = fma(sstate[row], ex2((m_old - mn) · log2 e), s)
///                mstate[row] = mn
///                t = targets[row]; if chunk_start <= t < chunk_start + cols:
///                    tlstate[row] = val(t - chunk_start)
/// ```
fn build_stats() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, I64, U32, U64};
    let op = LceChunkOp::Stats;
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let logits = b.add_param(names[0], f32_ptr(), Global);
    let bias = b.add_param(names[1], f32_ptr(), Global);
    let targets = b.add_param(names[2], KirType::Ptr(Box::new(I64), Global), Global);
    let mstate = b.add_param(names[3], f32_ptr(), Global);
    let sstate = b.add_param(names[4], f32_ptr(), Global);
    let tlstate = b.add_param(names[5], f32_ptr(), Global);
    let rows = b.add_param(names[6], U64, Global);
    let cols = b.add_param(names[7], U64, Global);
    let chunk_start = b.add_param(names[8], U64, Global);
    let has_bias = b.add_param(names[9], U32, Global);
    b.set_smem_layout(SmemLayout {
        regions: vec![SmemRegion { name: "sdata".to_string(), bytes: LCE_CHUNK_BLOCK * 4, align: 4, elem: F32 }],
        dynamic: false,
    });

    let entry = b.new_block();
    let setup = b.new_block();
    let exit: BlockId = b.new_block();

    b.set_block(entry);
    let row32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(row32, 0));
    let row = cast(&mut b, row32, U64);
    let past = cmp(&mut b, row, rows, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(setup)));

    b.set_block(setup);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = cast(&mut b, tid32, U64);
    let base = op2(&mut b, U64, KirOp::Mul, row, cols);
    let logits_row = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(logits_row, logits, base));
    let bias_chunk = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(bias_chunk, bias, chunk_start));
    let zero32 = konst(&mut b, U32, ConstValue::U32(0));
    let no_bias = cmp(&mut b, has_bias, zero32, CmpOp::Eq);
    let sdata = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: sdata, region: 0 });
    let neg_inf = konst(&mut b, F32, ConstValue::F32(f32::NEG_INFINITY));

    // Pass 1: the block max, merged with the running max by thread 0.
    let m = column_loop(&mut b, tid, cols, neg_inf, |b, k, acc| {
        let v = value(b, logits_row, k, bias_chunk, k, no_bias);
        op2(b, F32, KirOp::Max, acc, v)
    });
    store_at(&mut b, Shared, sdata, tid32, m);
    b.emit(KirOp::Barrier);
    let first = b.new_block();
    let others = b.new_block();
    let wait = b.new_block();
    let m_old = b.add_block_param(wait, F32);
    let zero32 = konst(&mut b, U32, ConstValue::U32(0));
    let not_first = cmp(&mut b, tid32, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(others), KirEdge::to(first)));

    b.set_block(first);
    let block_max = thread0_fold(&mut b, sdata, m, KirOp::Max);
    let old = load_at(&mut b, F32, Global, mstate, row);
    let m_new = op2(&mut b, F32, KirOp::Max, old, block_max);
    b.emit(KirOp::Store(sdata, m_new, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::with(wait, vec![old])));

    b.set_block(others);
    // Only thread 0 reads `m_old` after the passes; the others carry `-inf`.
    b.terminate(KirTerminator::Branch(KirEdge::with(wait, vec![neg_inf])));

    b.set_block(wait);
    b.emit(KirOp::Barrier);
    let mn = var(&mut b, F32);
    b.emit(KirOp::Load(mn, sdata, Shared));

    // Pass 2: the block sum of the exponentials against the new max.
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    let s = column_loop(&mut b, tid, cols, zero, |b, k, acc| {
        let v = value(b, logits_row, k, bias_chunk, k, no_bias);
        let e = exp_shifted(b, v, mn);
        op2(b, F32, KirOp::AddRn, acc, e)
    });
    b.emit(KirOp::Barrier);
    store_at(&mut b, Shared, sdata, tid32, s);
    b.emit(KirOp::Barrier);
    let last = b.new_block();
    let zero32 = konst(&mut b, U32, ConstValue::U32(0));
    let not_first = cmp(&mut b, tid32, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(exit), KirEdge::to(last)));

    b.set_block(last);
    let sum = thread0_fold(&mut b, sdata, s, KirOp::AddRn);
    let s_old = load_at(&mut b, F32, Global, sstate, row);
    let rescale = exp_shifted(&mut b, m_old, mn);
    let s_new = var(&mut b, F32);
    b.emit(KirOp::Fma(s_new, s_old, rescale, sum));
    store_at(&mut b, Global, sstate, row, s_new);
    store_at(&mut b, Global, mstate, row, mn);

    // The target logit, when the target falls in this chunk.
    let t = load_at(&mut b, I64, Global, targets, row);
    let cs = cast(&mut b, chunk_start, I64);
    let above = b.new_block();
    let inside = b.new_block();
    let before = cmp(&mut b, t, cs, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(before, KirEdge::to(exit), KirEdge::to(above)));

    b.set_block(above);
    let rel = op2(&mut b, I64, KirOp::Sub, t, cs);
    let cols_s = cast(&mut b, cols, I64);
    let beyond = cmp(&mut b, rel, cols_s, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(beyond, KirEdge::to(exit), KirEdge::to(inside)));

    b.set_block(inside);
    let j = cast(&mut b, rel, U64);
    let tl = value(&mut b, logits_row, j, bias_chunk, j, no_bias);
    store_at(&mut b, Global, tlstate, row, tl);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([LCE_CHUNK_BLOCK, 1, 1]);
    b.set_launch_bounds(LCE_CHUNK_BLOCK, None);
    b.set_max_registers(STATS_MAX_REGISTERS);
    b.finalize()
}

/// ```text
/// entry: idx = global id; stride = %ntid.x · %nctaid.x; total = rows · cols
/// head(idx): if idx >= total { exit }
///        r = idx / cols; j = idx - r · cols
///        val = logits[idx] (+ bias[chunk_start + j]); t = targets[r]
///        dl = t < 0 ? 0
///                   : (e = ex2((val - lse[r]) · log2 e);
///                      (t != chunk_start + j ? e : e - 1) · scale)
///        logits[idx] = dl; if has_bias { dbias[chunk_start + j] += dl }
///        head(idx + stride)
/// ```
fn build_dlogits() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, I64, U32, U64};
    let op = LceChunkOp::Dlogits;
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let logits = b.add_param(names[0], f32_ptr(), Global);
    let bias = b.add_param(names[1], f32_ptr(), Global);
    let targets = b.add_param(names[2], KirType::Ptr(Box::new(I64), Global), Global);
    let lse = b.add_param(names[3], f32_ptr(), Global);
    let dbias = b.add_param(names[4], f32_ptr(), Global);
    let rows = b.add_param(names[5], U64, Global);
    let cols = b.add_param(names[6], U64, Global);
    let chunk_start = b.add_param(names[7], U64, Global);
    let scale = b.add_param(names[8], F32, Global);
    let has_bias = b.add_param(names[9], U32, Global);

    let entry = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let invalid = b.new_block();
    let valid = b.new_block();
    let store = b.new_block();
    let reduce = b.new_block();
    let next = b.new_block();
    let exit = b.new_block();
    let idx = b.add_block_param(head, U64);
    let dl = b.add_block_param(store, F32);

    b.set_block(entry);
    let g32 = var(&mut b, U32);
    b.emit(KirOp::GlobalId(g32, 0));
    let start = cast(&mut b, g32, U64);
    let ntid = var(&mut b, U32);
    b.emit(KirOp::BlockDim(ntid, 0));
    let nctaid = var(&mut b, U32);
    b.emit(KirOp::GridDim(nctaid, 0));
    let stride32 = op2(&mut b, U32, KirOp::Mul, nctaid, ntid);
    let stride = cast(&mut b, stride32, U64);
    let total = op2(&mut b, U64, KirOp::Mul, rows, cols);
    let zero32 = konst(&mut b, U32, ConstValue::U32(0));
    let no_bias = cmp(&mut b, has_bias, zero32, CmpOp::Eq);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![start])));

    b.set_block(head);
    let end = cmp(&mut b, idx, total, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(exit), KirEdge::to(body)));

    b.set_block(body);
    let r = op2(&mut b, U64, KirOp::Div, idx, cols);
    let rc = op2(&mut b, U64, KirOp::Mul, r, cols);
    let j = op2(&mut b, U64, KirOp::Sub, idx, rc);
    let col = op2(&mut b, U64, KirOp::Add, chunk_start, j);
    let v = value(&mut b, logits, idx, bias, col, no_bias);
    let t = load_at(&mut b, I64, Global, targets, r);
    let zero64 = konst(&mut b, I64, ConstValue::I64(0));
    let ignored = cmp(&mut b, t, zero64, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(ignored, KirEdge::to(invalid), KirEdge::to(valid)));

    b.set_block(invalid);
    let none = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![none])));

    b.set_block(valid);
    let l = load_at(&mut b, F32, Global, lse, r);
    let p = exp_shifted(&mut b, v, l);
    let col_s = cast(&mut b, col, I64);
    let other = cmp(&mut b, t, col_s, CmpOp::Ne);
    let one = konst(&mut b, F32, ConstValue::F32(1.0));
    let hit = op2(&mut b, F32, KirOp::SubRn, p, one);
    let g = var(&mut b, F32);
    b.emit(KirOp::Select(g, other, p, hit));
    let scaled = op2(&mut b, F32, KirOp::MulRn, g, scale);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![scaled])));

    b.set_block(store);
    store_at(&mut b, Global, logits, idx, dl);
    b.terminate(KirTerminator::CondBranch(no_bias, KirEdge::to(next), KirEdge::to(reduce)));

    b.set_block(reduce);
    let at = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(at, dbias, col));
    b.emit(KirOp::AtomicAdd(at, dl, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(next)));

    b.set_block(next);
    let idx_next = op2(&mut b, U64, KirOp::Add, idx, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![idx_next])));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([LCE_CHUNK_BLOCK, 1, 1]);
    b.set_launch_bounds(LCE_CHUNK_BLOCK, None);
    b.finalize()
}

/// Build one kernel.
pub fn build(op: LceChunkOp) -> KernelIR {
    match op {
        LceChunkOp::Stats => build_stats(),
        LceChunkOp::Dlogits => build_dlogits(),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: LceChunkOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: LceChunkOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn the_kernels_verify_and_keep_their_entries_and_parameters() {
        for op in LceChunkOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
        }
        assert!(text(LceChunkOp::Stats).contains(&format!(".maxnreg {STATS_MAX_REGISTERS}")));
    }

    /// The stats kernel: one shared buffer, four barriers, `max` folds, one
    /// `fma` for the rescale, two `ex2`s; the dlogits kernel: one `ex2`, one
    /// `red.global.add.f32`, a `selp` for the one-hot. No unrounded float
    /// add, subtract or multiply in either.
    #[test]
    fn the_arithmetic_follows_the_hand_kernels() {
        let s = text(LceChunkOp::Stats);
        assert!(s.contains(".shared .align 4 .b8 shared_mem[1024]"), "{s}");
        assert_eq!(s.matches("bar.sync 0;").count(), 4, "{s}");
        assert_eq!(s.matches("max.f32").count(), 3, "{s}");
        assert_eq!(s.matches("fma.rn.f32").count(), 1, "{s}");
        assert_eq!(s.matches("ex2.approx.f32").count(), 2, "{s}");
        assert_eq!(s.matches("ld.global.s64").count(), 1, "{s}");
        let d = text(LceChunkOp::Dlogits);
        assert_eq!(d.matches("ex2.approx.f32").count(), 1, "{d}");
        assert_eq!(d.matches("red.global.add.f32").count(), 1, "{d}");
        assert_eq!(d.matches("selp.f32").count(), 1, "{d}");
        assert!(d.contains("%nctaid.x"), "{d}");
        for p in [s, d] {
            for bare in ["add.f32", "sub.f32", "mul.f32"] {
                assert!(!p.lines().any(|l| l.trim_start().starts_with(bare)), "{bare}\n{p}");
            }
            assert_eq!(p.matches("0f3FB8AA3B").count(), p.matches("ex2.approx.f32").count(), "{p}");
        }
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        for op in LceChunkOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
