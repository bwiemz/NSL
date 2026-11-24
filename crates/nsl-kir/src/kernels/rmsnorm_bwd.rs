// crates/nsl-kir/src/kernels/rmsnorm_bwd.rs
//! The RMSNorm backward as ONE pass over `dy` and `x`: the input gradient of
//! [`super::rmsnorm_dx`] and the per-chunk gamma-gradient column sums of
//! [`super::rmsnorm_dgamma`], from a single read of each row.
//!
//! `nsl_rmsnorm_bwd_fused[_add]_f32_s<S>(dy, x, gamma, dxout, [res,] partial,
//! rows, cols, eps, chunk_rows)`: one block of [`FUSED_BLOCK`] threads per
//! chunk of `chunk_rows` rows (`%ctaid.x`), `S = ceil(cols / 256)` column
//! slots per thread (`k = tid + 256·s`), the kernel specialised by `S` so a
//! thread holds its slots' `x`, `dy`, `γ` (and `res`) and gamma-gradient sums
//! in registers. For each row of the chunk, in order:
//!
//! 1. **Reduce.** Each thread folds `S1 = Σ x²` and `S2 = Σ (dy · γ) · x`
//!    over its slots with `fma.rn` from `+0.0` — the very partials both
//!    separate kernels form — and stores them to shared memory. Thread 0 then
//!    folds them as [`super::rmsnorm_dx`] does (serially, `1 .. %ntid.x`) into
//!    `rinv = min(rsqrt(S1 / cols + eps), 1e12)` and `S2`, and ALSO as
//!    `nsl_rmsnorm_rinv_rows_f32` does (warp butterflies, then the fixed tree
//!    of eight) into `rinv_g = 1 / sqrt(S1 / f32(cols) + eps)`, correctly
//!    rounded.
//! 2. **Write.** Every thread writes `dx = (γ · dy) · rinv - x · coeff`
//!    (`+ res`), exactly as the dx kernel does, and adds `(dy · x) · rinv_g`
//!    to its slot's running column sum, exactly as
//!    `nsl_rmsnorm_dgamma_partial_f32` does.
//!
//! At the end each thread stores its column sums to `partial[c · cols + k]`,
//! and `nsl_rmsnorm_dgamma_final_f32` adds the chunks as before.
//!
//! **Bits.** `dx` is bit-identical to `nsl_rmsnorm_dx_bwd[_add]_f32`: the same
//! operations on the same values in the same order. Each `partial` row is
//! bit-identical to the partial kernel's for a chunk of the same rows: its
//! `rinv_g` is the rinv kernel's, and the column sum adds the same products
//! in the same row order. What differs from the three-kernel path is the
//! chunking ([`fused_plan`] instead of [`super::rmsnorm_dgamma::dgamma_plan`]):
//! a block here needs every column of its rows, so it cannot take that
//! plan's few tall chunks (32 at Coder-1B) without starving the GPU. The
//! module's error bound is [`super::rmsnorm_dgamma`]'s with this plan's
//! depth; the result is deterministic for the same reason (a partition that
//! is a function of the shape, fixed summation orders, no atomics).
//!
//! The last slot is the only one that can run past `cols` (every earlier
//! one ends below `256·(S-1) < cols`): it loads a clamped in-bounds index,
//! keeps its sums by `selp`, and stores under a branch.

use super::elementwise::{f32_ptr, verified_ptx};
use super::rmsnorm_dgamma::{WARPS, tree_of_eight, warp_sum};
use super::rmsnorm_dx::{RINV_MAX, RMSNORM_DX_BLOCK};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator,
    KirType, SmemLayout, SmemRegion, VarId,
};

/// A block's threads: the dx kernel's, so the per-thread partials (and the
/// serial fold over them) are its.
pub const FUSED_BLOCK: u32 = RMSNORM_DX_BLOCK;

/// The most column slots a thread holds (`cols ≤ 256 · 16 = 4096`). Wider
/// rows take the three-kernel path.
pub const MAX_SLOTS: u32 = 16;

/// How many blocks [`fused_plan`] aims for. Measured on the RTX PRO 4500
/// (sm_120, 82 SMs, ~896 GB/s) with L2 evicted before each launch
/// (2026-11-24, `nsl-audit/research/work/PERF-R3/normbwd`): at Coder-1B's
/// `[4096, 2048]` 193 us at 256 blocks (16 rows), 197 at 512, 203 at 1024
/// and 207 at 2048; at Coder-500M's `[2048, 1280]` 68 us from 256 to 1024
/// blocks and 78 at 128. The dx kernel alone takes 184 and 64 us there; the
/// three launches this replaces 321 and 115.
pub const FUSED_TARGET_BLOCKS: u64 = 256;

/// The fewest rows a chunk holds (the partials stay at most half of one
/// input).
pub const FUSED_MIN_CHUNK_ROWS: u64 = 2;

/// How a fused launch splits the rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedPlan {
    /// Rows per block (the last may hold fewer).
    pub chunk_rows: u64,
    /// The grid, and the final kernel's partial count: `ceil(rows / chunk_rows)`.
    pub chunks: u64,
}

impl FusedPlan {
    /// The most roundings a term of `dgamma` goes through: two products,
    /// `chunk_rows` adds in its chunk, `ceil(chunks / 8)` in its warp of the
    /// final kernel and three in its tree (as [`super::rmsnorm_dgamma::DgammaPlan::depth`]).
    pub fn depth(&self) -> u64 {
        2 + self.chunk_rows + self.chunks.div_ceil(u64::from(WARPS)) + 3
    }
}

/// The partition for `[rows, cols]`: about [`FUSED_TARGET_BLOCKS`] chunks of
/// at least [`FUSED_MIN_CHUNK_ROWS`] rows. A function of the rows alone.
pub fn fused_plan(rows: u64) -> FusedPlan {
    fused_plan_with(
        rows,
        rows.div_ceil(FUSED_TARGET_BLOCKS).max(FUSED_MIN_CHUNK_ROWS),
    )
}

/// The partition with `chunk_rows` rows a chunk (at least one).
pub fn fused_plan_with(rows: u64, chunk_rows: u64) -> FusedPlan {
    let chunk_rows = chunk_rows.max(1);
    FusedPlan {
        chunk_rows,
        chunks: rows.div_ceil(chunk_rows),
    }
}

/// The slots a thread needs for `cols` columns, or `None` past [`MAX_SLOTS`]
/// (and for an empty row).
pub fn slots_for(cols: u64) -> Option<u32> {
    let s = cols.div_ceil(u64::from(FUSED_BLOCK));
    (1..=u64::from(MAX_SLOTS)).contains(&s).then_some(s as u32)
}

/// The `.visible .entry` name of the kernel with `slots` slots.
pub fn kernel_name(add: bool, slots: u32) -> String {
    format!(
        "nsl_rmsnorm_bwd_fused{}_f32_s{slots}",
        if add { "_add" } else { "" }
    )
}

/// The parameter names, in launch order.
pub fn params(add: bool) -> &'static [&'static str] {
    if add {
        &[
            "dy",
            "x",
            "gamma",
            "dxout",
            "res",
            "partial",
            "rows",
            "cols",
            "eps",
            "chunk_rows",
        ]
    } else {
        &[
            "dy",
            "x",
            "gamma",
            "dxout",
            "partial",
            "rows",
            "cols",
            "eps",
            "chunk_rows",
        ]
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

fn op2(
    b: &mut KirBuilder,
    ty: KirType,
    op: fn(VarId, VarId, VarId) -> KirOp,
    x: VarId,
    y: VarId,
) -> VarId {
    let dst = var(b, ty);
    b.emit(op(dst, x, y));
    dst
}

fn fma(b: &mut KirBuilder, x: VarId, y: VarId, acc: VarId) -> VarId {
    let dst = var(b, KirType::F32);
    b.emit(KirOp::Fma(dst, x, y, acc));
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

/// `total / cols` with `div.approx`, `cols` converted with `cvt.rn` (the dx
/// kernel's spelling).
fn per_column(b: &mut KirBuilder, total: VarId, cols: VarId) -> VarId {
    let n = var(b, KirType::F32);
    b.emit(KirOp::Cast(n, cols, KirType::F32));
    op2(b, KirType::F32, KirOp::DivApprox, total, n)
}

fn region(name: &str, floats: u32) -> SmemRegion {
    SmemRegion {
        name: name.to_string(),
        bytes: floats * 4,
        align: 4,
        elem: KirType::F32,
    }
}

/// ```text
/// entry:  c = %ctaid.x; r0 = c·chunk_rows; if r0 >= rows { exit }
/// setup:  k_s = tid + 256·s; the last slot's index clamped to 0 past cols
///         r1 = min(r0 + chunk_rows, rows); row_head(r0, acc_s = +0.0)
/// row:    x_s, dy_s, γ_s (, res_s) = the row's slot values
///         s1, s2 = the dx kernel's per-thread folds; ssq[tid] = s1; sdwx[tid] = s2
///         w = warp butterfly of s1; lane 0: swarp[warp] = w
/// sync1:  barrier; thread 0: serial fold -> rinv, S2; tree(swarp) -> rinv_g;
///         bcast = [rinv, S2, rinv_g]
/// sync2:  barrier; dx_s = (γ·dy)·rinv - x·coeff (+ res_s); acc_s += (dy·x)·rinv_g
///         store dx_s; row_head(i + 1, acc_s)
/// done:   partial[c·cols + k_s] = acc_s
/// ```
pub fn build(add: bool, slots: u32) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    assert!(
        (1..=MAX_SLOTS).contains(&slots),
        "rmsnorm_bwd: {slots} slots"
    );
    let names = params(add);
    let mut b = KirBuilder::new(&kernel_name(add, slots));
    let dy = b.add_param(names[0], f32_ptr(), Global);
    let x = b.add_param(names[1], f32_ptr(), Global);
    let gamma = b.add_param(names[2], f32_ptr(), Global);
    let dxout = b.add_param(names[3], f32_ptr(), Global);
    let res = add.then(|| b.add_param(names[4], f32_ptr(), Global));
    let at = names.len() - 5;
    let partial = b.add_param(names[at], f32_ptr(), Global);
    let rows = b.add_param(names[at + 1], U64, Global);
    let cols = b.add_param(names[at + 2], U64, Global);
    let eps = b.add_param(names[at + 3], F32, Global);
    let chunk_rows = b.add_param(names[at + 4], U64, Global);
    b.set_smem_layout(SmemLayout {
        regions: vec![
            region("ssq", FUSED_BLOCK),
            region("sdwx", FUSED_BLOCK),
            region("swarp", WARPS),
            region("bcast", 4),
        ],
        dynamic: false,
    });
    let n = slots as usize;
    let last = n - 1;

    let entry = b.new_block();
    let setup = b.new_block();
    let row_head = b.new_block();
    let row = b.new_block();
    let put = b.new_block();
    let sync1 = b.new_block();
    let first = b.new_block();
    let fold_head = b.new_block();
    let fold_step = b.new_block();
    let fold_done = b.new_block();
    let sync2 = b.new_block();
    let st_last = b.new_block();
    let cont = b.new_block();
    let done = b.new_block();
    let st_partial = b.new_block();
    let exit = b.new_block();
    let i = b.add_block_param(row_head, U64);
    let acc: Vec<VarId> = (0..n).map(|_| b.add_block_param(row_head, F32)).collect();
    let fi = b.add_block_param(fold_head, U32);
    let t1 = b.add_block_param(fold_head, F32);
    let t2 = b.add_block_param(fold_head, F32);

    b.set_block(entry);
    let c32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(c32, 0));
    let c = var(&mut b, U64);
    b.emit(KirOp::Cast(c, c32, U64));
    let r0 = op2(&mut b, U64, KirOp::Mul, c, chunk_rows);
    let past = cmp(&mut b, r0, rows, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        past,
        KirEdge::to(exit),
        KirEdge::to(setup),
    ));

    b.set_block(setup);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = var(&mut b, U64);
    b.emit(KirOp::Cast(tid, tid32, U64));
    let lane_mask = konst(&mut b, U32, ConstValue::U32(31));
    let lane = op2(&mut b, U32, KirOp::And, tid32, lane_mask);
    let five = konst(&mut b, U32, ConstValue::U32(5));
    let warp = op2(&mut b, U32, KirOp::Shr, tid32, five);
    let r_end = op2(&mut b, U64, KirOp::Add, r0, chunk_rows);
    let r1 = op2(&mut b, U64, KirOp::Min, r_end, rows);
    let ks: Vec<VarId> = (0..slots)
        .map(|s| {
            let off = konst(&mut b, U64, ConstValue::U64(u64::from(s * FUSED_BLOCK)));
            op2(&mut b, U64, KirOp::Add, tid, off)
        })
        .collect();
    let valid_last = cmp(&mut b, ks[last], cols, CmpOp::Lt);
    let zero64 = konst(&mut b, U64, ConstValue::U64(0));
    let k_last = select(&mut b, U64, valid_last, ks[last], zero64);
    let idx: Vec<VarId> = (0..n)
        .map(|s| if s == last { k_last } else { ks[s] })
        .collect();
    let shared = |b: &mut KirBuilder, r: u32| {
        let p = var(b, KirType::Ptr(Box::new(F32), Shared));
        b.emit(KirOp::SharedRegion { dst: p, region: r });
        p
    };
    let ssq = shared(&mut b, 0);
    let sdwx = shared(&mut b, 1);
    let swarp = shared(&mut b, 2);
    let bcast = shared(&mut b, 3);
    let zeros: Vec<VarId> = (0..n)
        .map(|_| konst(&mut b, F32, ConstValue::F32(0.0)))
        .collect();
    let mut args = vec![r0];
    args.extend(zeros);
    b.terminate(KirTerminator::Branch(KirEdge::with(row_head, args)));

    b.set_block(row_head);
    let end = cmp(&mut b, i, r1, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        end,
        KirEdge::to(done),
        KirEdge::to(row),
    ));

    // Pass 1: the slot values and the dx kernel's per-thread partials.
    b.set_block(row);
    let base = op2(&mut b, U64, KirOp::Mul, i, cols);
    let row_ptr = |b: &mut KirBuilder, p: VarId| {
        let r = var(b, f32_ptr());
        b.emit(KirOp::PtrOffset(r, p, base));
        r
    };
    let dy_row = row_ptr(&mut b, dy);
    let x_row = row_ptr(&mut b, x);
    let dx_row = row_ptr(&mut b, dxout);
    let res_row = res.map(|p| row_ptr(&mut b, p));
    let xv: Vec<VarId> = idx
        .iter()
        .map(|&k| load_at(&mut b, Global, x_row, k))
        .collect();
    let dv: Vec<VarId> = idx
        .iter()
        .map(|&k| load_at(&mut b, Global, dy_row, k))
        .collect();
    let gv: Vec<VarId> = idx
        .iter()
        .map(|&k| load_at(&mut b, Global, gamma, k))
        .collect();
    // The residual is read with the row, not after the barriers: its load
    // latency then overlaps the reductions (measured 2-12 us a launch at the
    // recipe shapes).
    let rv: Option<Vec<VarId>> = res_row.map(|rr| {
        idx.iter()
            .map(|&k| load_at(&mut b, Global, rr, k))
            .collect()
    });
    let mut s1 = konst(&mut b, F32, ConstValue::F32(0.0));
    let mut s2 = konst(&mut b, F32, ConstValue::F32(0.0));
    for s in 0..n {
        let s1n = fma(&mut b, xv[s], xv[s], s1);
        let dg = op2(&mut b, F32, KirOp::MulRn, dv[s], gv[s]);
        let s2n = fma(&mut b, dg, xv[s], s2);
        if s == last {
            s1 = select(&mut b, F32, valid_last, s1n, s1);
            s2 = select(&mut b, F32, valid_last, s2n, s2);
        } else {
            (s1, s2) = (s1n, s2n);
        }
    }
    store_at(&mut b, Shared, ssq, tid32, s1);
    store_at(&mut b, Shared, sdwx, tid32, s2);
    let wsum = warp_sum(&mut b, s1);
    let zero32 = konst(&mut b, U32, ConstValue::U32(0));
    let not_lane0 = cmp(&mut b, lane, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(
        not_lane0,
        KirEdge::to(sync1),
        KirEdge::to(put),
    ));

    b.set_block(put);
    store_at(&mut b, Shared, swarp, warp, wsum);
    b.terminate(KirTerminator::Branch(KirEdge::to(sync1)));

    b.set_block(sync1);
    b.emit(KirOp::Barrier);
    let not_first = cmp(&mut b, tid32, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(
        not_first,
        KirEdge::to(sync2),
        KirEdge::to(first),
    ));

    // Thread 0: the dx kernel's serial fold, then the rinv kernel's tree.
    b.set_block(first);
    let ntid = var(&mut b, U32);
    b.emit(KirOp::BlockDim(ntid, 0));
    let one32 = konst(&mut b, U32, ConstValue::U32(1));
    b.terminate(KirTerminator::Branch(KirEdge::with(
        fold_head,
        vec![one32, s1, s2],
    )));

    b.set_block(fold_head);
    let fend = cmp(&mut b, fi, ntid, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        fend,
        KirEdge::to(fold_done),
        KirEdge::to(fold_step),
    ));

    b.set_block(fold_step);
    let o1 = load_at(&mut b, Shared, ssq, fi);
    let t1n = op2(&mut b, F32, KirOp::AddRn, t1, o1);
    let o2 = load_at(&mut b, Shared, sdwx, fi);
    let t2n = op2(&mut b, F32, KirOp::AddRn, t2, o2);
    let one32b = konst(&mut b, U32, ConstValue::U32(1));
    let fi_next = op2(&mut b, U32, KirOp::Add, fi, one32b);
    b.terminate(KirTerminator::Branch(KirEdge::with(
        fold_head,
        vec![fi_next, t1n, t2n],
    )));

    b.set_block(fold_done);
    let m = per_column(&mut b, t1, cols);
    let v = op2(&mut b, F32, KirOp::AddRn, m, eps);
    let r = op1(&mut b, F32, KirOp::Rsqrt, v);
    let cap = konst(&mut b, F32, ConstValue::F32(RINV_MAX));
    let rc = op2(&mut b, F32, KirOp::Min, r, cap);
    let b0 = konst(&mut b, U32, ConstValue::U32(0));
    let b1 = konst(&mut b, U32, ConstValue::U32(1));
    let b2 = konst(&mut b, U32, ConstValue::U32(2));
    store_at(&mut b, Shared, bcast, b0, rc);
    store_at(&mut b, Shared, bcast, b1, t2);
    let tsum = tree_of_eight(&mut b, swarp, zero32, 1);
    let nf = var(&mut b, F32);
    b.emit(KirOp::Cast(nf, cols, F32));
    let mean = op2(&mut b, F32, KirOp::Div, tsum, nf);
    let shifted = op2(&mut b, F32, KirOp::AddRn, mean, eps);
    let root = var(&mut b, F32);
    b.emit(KirOp::Sqrt(root, shifted));
    let onef = konst(&mut b, F32, ConstValue::F32(1.0));
    let rinv_g = op2(&mut b, F32, KirOp::Div, onef, root);
    store_at(&mut b, Shared, bcast, b2, rinv_g);
    b.terminate(KirTerminator::Branch(KirEdge::to(sync2)));

    // Pass 2: dx as the dx kernel writes it, and the column sums.
    b.set_block(sync2);
    b.emit(KirOp::Barrier);
    let c0 = konst(&mut b, U32, ConstValue::U32(0));
    let c1 = konst(&mut b, U32, ConstValue::U32(1));
    let c2 = konst(&mut b, U32, ConstValue::U32(2));
    let rinv = load_at(&mut b, Shared, bcast, c0);
    let s2_all = load_at(&mut b, Shared, bcast, c1);
    let rg = load_at(&mut b, Shared, bcast, c2);
    let r2 = op2(&mut b, F32, KirOp::MulRn, rinv, rinv);
    let r3 = op2(&mut b, F32, KirOp::MulRn, r2, rinv);
    let cc = op2(&mut b, F32, KirOp::MulRn, r3, s2_all);
    let coeff = per_column(&mut b, cc, cols);
    let mut acc_next = Vec::with_capacity(n);
    let mut out_last = None;
    for s in 0..n {
        let a = op2(&mut b, F32, KirOp::MulRn, gv[s], dv[s]);
        let a = op2(&mut b, F32, KirOp::MulRn, a, rinv);
        let xc = op2(&mut b, F32, KirOp::MulRn, xv[s], coeff);
        let mut out = op2(&mut b, F32, KirOp::SubRn, a, xc);
        if let Some(rv) = &rv {
            out = op2(&mut b, F32, KirOp::AddRn, out, rv[s]);
        }
        let p = op2(&mut b, F32, KirOp::MulRn, dv[s], xv[s]);
        let q = op2(&mut b, F32, KirOp::MulRn, p, rg);
        let an = op2(&mut b, F32, KirOp::AddRn, acc[s], q);
        if s == last {
            acc_next.push(select(&mut b, F32, valid_last, an, acc[s]));
            out_last = Some(out);
        } else {
            acc_next.push(an);
            store_at(&mut b, Global, dx_row, idx[s], out);
        }
    }
    b.terminate(KirTerminator::CondBranch(
        valid_last,
        KirEdge::to(st_last),
        KirEdge::to(cont),
    ));

    b.set_block(st_last);
    store_at(
        &mut b,
        Global,
        dx_row,
        k_last,
        out_last.expect("a last slot"),
    );
    b.terminate(KirTerminator::Branch(KirEdge::to(cont)));

    b.set_block(cont);
    let one64 = konst(&mut b, U64, ConstValue::U64(1));
    let i_next = op2(&mut b, U64, KirOp::Add, i, one64);
    let mut args = vec![i_next];
    args.extend(acc_next);
    b.terminate(KirTerminator::Branch(KirEdge::with(row_head, args)));

    // The chunk's column sums.
    b.set_block(done);
    let prow = op2(&mut b, U64, KirOp::Mul, c, cols);
    for s in 0..last {
        let at = op2(&mut b, U64, KirOp::Add, prow, ks[s]);
        store_at(&mut b, Global, partial, at, acc[s]);
    }
    b.terminate(KirTerminator::CondBranch(
        valid_last,
        KirEdge::to(st_partial),
        KirEdge::to(exit),
    ));

    b.set_block(st_partial);
    let at_last = op2(&mut b, U64, KirOp::Add, prow, ks[last]);
    store_at(&mut b, Global, partial, at_last, acc[last]);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([FUSED_BLOCK, 1, 1]);
    b.set_launch_bounds(FUSED_BLOCK, None);
    b.finalize()
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(add: bool, slots: u32) -> Vec<u8> {
    verified_ptx(build(add, slots))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(add: bool, slots: u32) -> String {
        String::from_utf8(ptx(add, slots)).expect("ASCII")
    }

    #[test]
    fn every_variant_verifies_and_keeps_its_entry_and_parameters() {
        for add in [false, true] {
            for slots in 1..=MAX_SLOTS {
                let p = text(add, slots);
                assert!(p.ends_with('\0'), "{add} {slots}");
                assert!(
                    p.contains(&format!(".visible .entry {}(", kernel_name(add, slots))),
                    "{p}"
                );
                for name in params(add) {
                    assert!(
                        p.contains(&format!("[param_{name}]")),
                        "{add} {slots}: {name}\n{p}"
                    );
                }
                assert!(p.contains(&format!(".maxntid {FUSED_BLOCK}, 1, 1")), "{p}");
            }
        }
    }

    /// Per slot: the dx kernel's two `fma`s and its write arithmetic, the
    /// partial kernel's two products and add; per row: one `rsqrt` clamped by
    /// `min`, two `div.approx`, the rinv kernel's two `div.rn`, one `sqrt.rn`,
    /// five butterfly shuffles, and two barriers.
    #[test]
    fn the_arithmetic_is_both_kernels() {
        for add in [false, true] {
            for slots in [1u32, 5, 8] {
                let p = text(add, slots);
                let s = slots as usize;
                assert_eq!(p.matches("fma.rn.f32").count(), 2 * s, "{p}");
                assert_eq!(p.matches("rsqrt.approx.f32").count(), 1, "{p}");
                assert_eq!(p.matches("min.f32").count(), 1, "{p}");
                assert_eq!(p.matches("div.approx.f32").count(), 2, "{p}");
                assert_eq!(p.matches("div.rn.f32").count(), 2, "{p}");
                assert_eq!(p.matches("sqrt.rn.f32").count(), 1, "{p}");
                assert_eq!(p.matches("shfl.sync.bfly.b32").count(), 5, "{p}");
                assert_eq!(p.matches("bar.sync 0;").count(), 2, "{p}");
                assert_eq!(p.matches("sub.rn.f32").count(), s, "{p}");
                // add.rn: two in the serial fold, five butterfly, seven tree,
                // two `+ eps`, one column sum a slot, one residual a slot.
                let adds = 2 + 5 + 7 + 2 + s + if add { s } else { 0 };
                assert_eq!(p.matches("add.rn.f32").count(), adds, "{p}");
                assert!(
                    !p.contains("add.f32") && !p.contains("mul.f32 ") && !p.contains("sub.f32"),
                    "{p}"
                );
            }
        }
    }

    #[test]
    fn the_plan_and_the_slots() {
        // Coder-1B: 4096 rows -> 256 chunks of 16. Coder-500M: 2048 -> 256
        // of 8. Coder-50M: 1024 -> 256 of 4. Fewer rows: chunks of 2.
        assert_eq!(
            fused_plan(4096),
            FusedPlan {
                chunk_rows: 16,
                chunks: 256
            }
        );
        assert_eq!(
            fused_plan(2048),
            FusedPlan {
                chunk_rows: 8,
                chunks: 256
            }
        );
        assert_eq!(
            fused_plan(1024),
            FusedPlan {
                chunk_rows: 4,
                chunks: 256
            }
        );
        assert_eq!(
            fused_plan(37),
            FusedPlan {
                chunk_rows: 2,
                chunks: 19
            }
        );
        assert_eq!(
            fused_plan(1),
            FusedPlan {
                chunk_rows: 2,
                chunks: 1
            }
        );
        assert_eq!(
            fused_plan(0),
            FusedPlan {
                chunk_rows: 2,
                chunks: 0
            }
        );
        assert_eq!(fused_plan(4096).depth(), 2 + 16 + 32 + 3);
        assert_eq!(
            fused_plan_with(10, 0),
            FusedPlan {
                chunk_rows: 1,
                chunks: 10
            }
        );
        assert_eq!(slots_for(2048), Some(8));
        assert_eq!(slots_for(1280), Some(5));
        assert_eq!(slots_for(1), Some(1));
        assert_eq!(slots_for(4096), Some(16));
        assert_eq!(slots_for(4097), None);
        assert_eq!(slots_for(0), None);
    }
}
