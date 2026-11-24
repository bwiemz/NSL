// crates/nsl-kir/src/kernels/rmsnorm_dgamma.rs
//! The fused RMSNorm gamma backward, `dgamma[j] = Σ_i dy[i, j] · x[i, j] /
//! rms_i` with `rms_i = sqrt(Σ_j x[i, j]² / cols + eps)`, over a row-major
//! `[rows, cols]` view, as three launches built in KIR:
//!
//! 1. `nsl_rmsnorm_rinv_rows_f32(x, rinv, rows, cols, eps)`: one block of
//!    [`RINV_BLOCK`] threads per row (`%ctaid.x`). Thread `t` folds its
//!    columns `t, t + 256, …` with `fma.rn` from `+0.0`; each warp adds its
//!    32 partials by a butterfly (`shfl.sync.bfly` 16, 8, 4, 2, 1, each an
//!    `add.rn`), lane 0 of warp `w` stores the warp's sum to `swarp[w]`, and
//!    after a barrier thread 0 adds the eight in a fixed tree,
//!    `((w0 + w1) + (w2 + w3)) + ((w4 + w5) + (w6 + w7))`, and stores
//!    `rinv[r] = 1 / sqrt(s / f32(cols) + eps)` (`div.rn`, `add.rn`,
//!    `sqrt.rn`, `div.rn`).
//! 2. `nsl_rmsnorm_dgamma_partial_f32(dy, x, rinv, partial, rows, cols,
//!    chunk_rows)`: a thread per column `j = %ctaid.x · 256 + %tid.x` and a
//!    grid row per chunk `c = %ctaid.y` of `chunk_rows` rows. The thread
//!    adds `(dy · x) · rinv[i]` (`mul.rn`, `mul.rn`, `add.rn`) for the
//!    chunk's rows in order from `+0.0` and stores `partial[c · cols + j]`.
//!    Consecutive threads read consecutive columns, so every load of a warp
//!    is one 128-byte line.
//! 3. `nsl_rmsnorm_dgamma_final_f32(partial, dgamma, chunks, cols)`: a block
//!    of [`FINAL_BLOCK`] threads per [`FINAL_COLS`] columns. Lane `l` of warp
//!    `w` owns column `j = %ctaid.x · 32 + l` and adds the partials of chunks
//!    `w, w + 8, …` in order from `+0.0`; the eight warp sums go through
//!    shared memory and, after a barrier, warp 0 adds them in the same fixed
//!    tree as kernel 1 and stores `dgamma[j]`.
//!
//! Every block reaches every barrier: a block past the work exits as a
//! whole before its first one, and a lane past the columns runs no trips but
//! still stores its `+0.0` and waits.
//!
//! **Deterministic.** The partition is [`dgamma_plan`], a function of
//! `(rows, cols)` alone, and every sum has a fixed order: no atomics, no
//! dependence on scheduling or on the device. Two runs give the same bits.
//!
//! **Error bounds** (`u = 2^-24`, `γ_n = n·u / (1 - n·u)`; Higham, *Accuracy
//! and Stability of Numerical Algorithms*, §3.1 and §4.2). A term that goes
//! through `n` roundings is perturbed by a factor `1 + θ_n`, `|θ_n| ≤ γ_n`.
//!
//! - *Sum of squares.* `fma.rn` folds `v²` exactly and rounds once per step,
//!   so a term of thread `t`'s fold is rounded at most `ceil(cols / 256)`
//!   times there, then 5 times in the butterfly and 3 in the tree:
//!   `ŝ = s · (1 + θ_{m+8})`, `m = ceil(cols / 256)`, since every term is
//!   non-negative. The division by `f32(cols)` (exact for `cols ≤ 2^24`) and
//!   the add of `eps ≥ 0` round once each, so `ŝ / cols + eps` carries
//!   `1 + θ_{m+10}`, and `rinv̂ / rinv = (1 + δ_4) / ((1 + δ_3) · sqrt(1 +
//!   θ_{m+10}))`, `|δ| ≤ u`: `|rinv̂ / rinv - 1| ≤ (1 + u) / ((1 - u) ·
//!   sqrt(1 - γ_{m+10})) - 1` ([`rinv_relative_bound`]).
//! - *Gamma gradient.* Each product rounds twice; a term then goes through
//!   at most `chunk_rows` adds in its chunk, `ceil(chunks / 8)` in its warp
//!   of kernel 3 and 3 in the tree. The terms have either sign, so the bound
//!   is absolute: `|dĝ_j - Σ_i t_ij| ≤ γ_D · Σ_i |t_ij|` with `t_ij = dy · x ·
//!   rinv[i]` for the `rinv` the kernel was given and `D = 2 + chunk_rows +
//!   ceil(chunks / 8) + 3` ([`DgammaPlan::depth`]). With `rinv` from kernel 1,
//!   `rinv[i]` itself carries `1 + ρ_i`, `|ρ_i| ≤` the bound above.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    ShuffleMode, SmemLayout, SmemRegion, VarId,
};

/// Kernel 1's block: one per row.
pub const RINV_BLOCK: u32 = 256;

/// Kernel 2's block: one thread per column.
pub const PARTIAL_BLOCK: u32 = 256;

/// Kernel 3's block: [`FINAL_COLS`] columns by [`FINAL_WARPS`] warps.
pub const FINAL_BLOCK: u32 = 256;

/// The columns one kernel-3 block owns: a lane each.
pub const FINAL_COLS: u32 = 32;

/// The warps of a kernel-1 or kernel-3 block, whose sums the fixed tree adds.
pub const WARPS: u32 = 8;

/// How many kernel-2 blocks [`dgamma_plan`] aims for. Measured on the RTX
/// PRO 4500 (sm_120, 82 SMs) with the inputs evicted from L2 before each
/// launch (2026-11-15): at Coder-1B's `[4096, 2048]` the kernel took 153,
/// 156, 166, 172, 179 and 184 µs at 256, 128, 512, 1024, 2048 and 4096
/// blocks (longer per-thread row runs stream better); at Coder-500M's
/// `[2048, 1280]` 30.6-33.3 µs from 160 to 1280 blocks and 57 at 80; at
/// Coder-50M's `[1024, 512]` 4.6 µs at 256 blocks, 5.8 at 128. Absolute
/// times moved by 2x between runs on this shared card; the order held. The
/// partials are `chunks · cols` floats, about 256 KiB at this target.
pub const PARTIAL_TARGET_BLOCKS: u64 = 256;

/// The fewest rows a chunk holds, so the partials are at most an eighth of
/// one input.
pub const MIN_CHUNK_ROWS: u64 = 8;

/// Which launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormDgammaOp {
    /// The per-row `1 / rms`.
    RinvRows,
    /// The per-chunk column sums.
    Partial,
    /// The column sums of the partials.
    Final,
}

impl RmsNormDgammaOp {
    pub const ALL: [RmsNormDgammaOp; 3] = [RmsNormDgammaOp::RinvRows, RmsNormDgammaOp::Partial, RmsNormDgammaOp::Final];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            RmsNormDgammaOp::RinvRows => "nsl_rmsnorm_rinv_rows_f32",
            RmsNormDgammaOp::Partial => "nsl_rmsnorm_dgamma_partial_f32",
            RmsNormDgammaOp::Final => "nsl_rmsnorm_dgamma_final_f32",
        }
    }

    /// The parameter names, in launch order.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            RmsNormDgammaOp::RinvRows => &["x", "rinv", "rows", "cols", "eps"],
            RmsNormDgammaOp::Partial => &["dy", "x", "rinv", "partial", "rows", "cols", "chunk_rows"],
            RmsNormDgammaOp::Final => &["partial", "dgamma", "chunks", "cols"],
        }
    }

    /// The block the runtime launches the kernel with.
    pub fn block(self) -> u32 {
        match self {
            RmsNormDgammaOp::RinvRows => RINV_BLOCK,
            RmsNormDgammaOp::Partial => PARTIAL_BLOCK,
            RmsNormDgammaOp::Final => FINAL_BLOCK,
        }
    }
}

/// How the gamma gradient's rows are split between kernel 2's grid rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DgammaPlan {
    /// Kernel 2's grid width, `ceil(cols / 256)`.
    pub col_blocks: u64,
    /// Rows per chunk (the last chunk may hold fewer).
    pub chunk_rows: u64,
    /// Kernel 2's grid height and kernel 3's partial count,
    /// `ceil(rows / chunk_rows)`; 0 when `rows` is.
    pub chunks: u64,
}

impl DgammaPlan {
    /// The most roundings a term of `dgamma` goes through: two products,
    /// `chunk_rows` adds in its chunk, `ceil(chunks / 8)` in its warp of
    /// kernel 3, three in the tree. The `γ` index of the module's bound.
    pub fn depth(&self) -> u64 {
        2 + self.chunk_rows + self.chunks.div_ceil(u64::from(WARPS)) + 3
    }

    /// Kernel 3's grid width, `ceil(cols / 32)`.
    pub fn final_blocks(cols: u64) -> u64 {
        cols.div_ceil(u64::from(FINAL_COLS))
    }
}

/// The partition for `[rows, cols]`: chunks of at least [`MIN_CHUNK_ROWS`]
/// rows, as many as make about [`PARTIAL_TARGET_BLOCKS`] kernel-2 blocks. A
/// function of the shape alone, so the summation order (and the result) is
/// the same on every run and every device.
pub fn dgamma_plan(rows: u64, cols: u64) -> DgammaPlan {
    let col_blocks = cols.div_ceil(u64::from(PARTIAL_BLOCK)).max(1);
    let target_chunks = PARTIAL_TARGET_BLOCKS.div_ceil(col_blocks).max(1);
    let chunk_rows = rows.div_ceil(target_chunks).max(MIN_CHUNK_ROWS);
    DgammaPlan { col_blocks, chunk_rows, chunks: rows.div_ceil(chunk_rows) }
}

/// The f32 unit roundoff, `2^-24`.
pub const UNIT_ROUNDOFF: f64 = 1.0 / (1u64 << 24) as f64;

/// `γ_n = n·u / (1 - n·u)`.
pub fn gamma(n: u64) -> f64 {
    let nu = n as f64 * UNIT_ROUNDOFF;
    assert!(nu < 1.0, "γ_{n} is undefined");
    nu / (1.0 - nu)
}

/// The bound on `|rinv̂ / rinv - 1|` for a row of `cols` columns (the
/// module docs derive it): `(1 + u) / ((1 - u) · sqrt(1 - γ_{m+10})) - 1`,
/// `m = ceil(cols / 256)`.
pub fn rinv_relative_bound(cols: u64) -> f64 {
    let m = cols.div_ceil(u64::from(RINV_BLOCK));
    let u = UNIT_ROUNDOFF;
    (1.0 + u) / ((1.0 - u) * (1.0 - gamma(m + 10)).sqrt()) - 1.0
}

fn var(b: &mut KirBuilder, ty: KirType) -> VarId {
    b.new_typed_var(ty)
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn u32c(b: &mut KirBuilder, v: u32) -> VarId {
    konst(b, KirType::U32, ConstValue::U32(v))
}

fn u64c(b: &mut KirBuilder, v: u64) -> VarId {
    konst(b, KirType::U64, ConstValue::U64(v))
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

fn add_rn(b: &mut KirBuilder, x: VarId, y: VarId) -> VarId {
    op2(b, KirType::F32, KirOp::AddRn, x, y)
}

/// The warp-shared region of `n` floats every block-wide sum goes through.
fn smem(n: u32) -> SmemLayout {
    SmemLayout {
        regions: vec![SmemRegion { name: "swarp".to_string(), bytes: n * 4, align: 4, elem: KirType::F32 }],
        dynamic: false,
    }
}

/// `v` summed over its warp by a butterfly: `v += shfl.bfly(v, d)` for `d =
/// 16, 8, 4, 2, 1`. Every lane ends with the same bits (IEEE addition is
/// commutative, and each lane adds the same two values at every level).
pub(crate) fn warp_sum(b: &mut KirBuilder, v: VarId) -> VarId {
    let mut acc = v;
    for d in [16u32, 8, 4, 2, 1] {
        let lane = u32c(b, d);
        let other = var(b, KirType::F32);
        b.emit(KirOp::WarpShuffle { dst: other, val: acc, lane, mode: ShuffleMode::Xor, width: 32 });
        acc = add_rn(b, acc, other);
    }
    acc
}

/// The eight warp sums at `sm[w · stride + at]`, `w = 0..8`, added as
/// `((w0 + w1) + (w2 + w3)) + ((w4 + w5) + (w6 + w7))`.
pub(crate) fn tree_of_eight(b: &mut KirBuilder, sm: VarId, at: VarId, stride: u32) -> VarId {
    let mut level: Vec<VarId> = (0..WARPS)
        .map(|w| {
            let off = u32c(b, w * stride);
            let i = op2(b, KirType::U32, KirOp::Add, off, at);
            load_at(b, AddressSpace::Shared, sm, i)
        })
        .collect();
    while level.len() > 1 {
        level = level.chunks(2).map(|p| add_rn(b, p[0], p[1])).collect();
    }
    level[0]
}

/// ```text
/// entry: r = %ctaid.x; if r >= rows { exit }
/// setup: head(k = tid, at = r·cols + tid, s = +0.0)
/// head:  if k >= cols { red }
/// body:  s = fma(x[at], x[at], s); k += 256; at += 256
/// red:   s = warp butterfly sum; if lane != 0 { sync }
/// put:   swarp[warp] = s
/// sync:  barrier; if tid != 0 { exit }
/// fin:   t = tree(swarp); rinv[r] = 1 / sqrt(t / f32(cols) + eps)
/// ```
fn build_rinv_rows() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let op = RmsNormDgammaOp::RinvRows;
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let x = b.add_param(names[0], f32_ptr(), Global);
    let rinv = b.add_param(names[1], f32_ptr(), Global);
    let rows = b.add_param(names[2], U64, Global);
    let cols = b.add_param(names[3], U64, Global);
    let eps = b.add_param(names[4], F32, Global);
    b.set_smem_layout(smem(WARPS));

    let entry = b.new_block();
    let setup = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let red = b.new_block();
    let put = b.new_block();
    let sync = b.new_block();
    let fin = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let at = b.add_block_param(head, U64);
    let s = b.add_block_param(head, F32);

    b.set_block(entry);
    let r32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(r32, 0));
    let r = cast(&mut b, r32, U64);
    let past = cmp(&mut b, r, rows, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(setup)));

    b.set_block(setup);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = cast(&mut b, tid32, U64);
    let base = op2(&mut b, U64, KirOp::Mul, r, cols);
    let at0 = op2(&mut b, U64, KirOp::Add, base, tid);
    let stride = u64c(&mut b, u64::from(RINV_BLOCK));
    let sm = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: sm, region: 0 });
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![tid, at0, zero])));

    b.set_block(head);
    let end = cmp(&mut b, k, cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(red), KirEdge::to(body)));

    b.set_block(body);
    let v = load_at(&mut b, Global, x, at);
    let s_next = var(&mut b, F32);
    b.emit(KirOp::Fma(s_next, v, v, s));
    let k_next = op2(&mut b, U64, KirOp::Add, k, stride);
    let at_next = op2(&mut b, U64, KirOp::Add, at, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, at_next, s_next])));

    b.set_block(red);
    let total = warp_sum(&mut b, s);
    let lane_mask = u32c(&mut b, 31);
    let lane = op2(&mut b, U32, KirOp::And, tid32, lane_mask);
    let five = u32c(&mut b, 5);
    let warp = op2(&mut b, U32, KirOp::Shr, tid32, five);
    let zero32 = u32c(&mut b, 0);
    let not_lane0 = cmp(&mut b, lane, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_lane0, KirEdge::to(sync), KirEdge::to(put)));

    b.set_block(put);
    store_at(&mut b, Shared, sm, warp, total);
    b.terminate(KirTerminator::Branch(KirEdge::to(sync)));

    b.set_block(sync);
    b.emit(KirOp::Barrier);
    let not_first = cmp(&mut b, tid32, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(exit), KirEdge::to(fin)));

    b.set_block(fin);
    let t = tree_of_eight(&mut b, sm, zero32, 1);
    let n = cast(&mut b, cols, F32);
    let mean = op2(&mut b, F32, KirOp::Div, t, n);
    let shifted = add_rn(&mut b, mean, eps);
    let root = var(&mut b, F32);
    b.emit(KirOp::Sqrt(root, shifted));
    let one = konst(&mut b, F32, ConstValue::F32(1.0));
    let inv = op2(&mut b, F32, KirOp::Div, one, root);
    store_at(&mut b, Global, rinv, r, inv);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    finish(b, exit, RINV_BLOCK)
}

/// ```text
/// entry:  j = %ctaid.x·256 + %tid.x; if j >= cols { exit }
/// chunk:  c = %ctaid.y; r0 = c·chunk_rows; if r0 >= rows { exit }
/// setup:  r1 = min(r0 + chunk_rows, rows); head(i = r0, at = r0·cols + j, acc = +0.0)
/// head:   if i >= r1 { done }
/// body:   acc += (dy[at] · x[at]) · rinv[i]; at += cols; i += 1
/// done:   partial[c·cols + j] = acc
/// ```
fn build_partial() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let op = RmsNormDgammaOp::Partial;
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let dy = b.add_param(names[0], f32_ptr(), Global);
    let x = b.add_param(names[1], f32_ptr(), Global);
    let rinv = b.add_param(names[2], f32_ptr(), Global);
    let partial = b.add_param(names[3], f32_ptr(), Global);
    let rows = b.add_param(names[4], U64, Global);
    let cols = b.add_param(names[5], U64, Global);
    let chunk_rows = b.add_param(names[6], U64, Global);

    let entry = b.new_block();
    let chunk = b.new_block();
    let setup = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let exit = b.new_block();
    let i = b.add_block_param(head, U64);
    let at = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);

    b.set_block(entry);
    let j32 = var(&mut b, U32);
    b.emit(KirOp::GlobalId(j32, 0));
    let j = cast(&mut b, j32, U64);
    let past = cmp(&mut b, j, cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(chunk)));

    b.set_block(chunk);
    let c32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(c32, 1));
    let c = cast(&mut b, c32, U64);
    let r0 = op2(&mut b, U64, KirOp::Mul, c, chunk_rows);
    let empty = cmp(&mut b, r0, rows, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(empty, KirEdge::to(exit), KirEdge::to(setup)));

    b.set_block(setup);
    let r_end = op2(&mut b, U64, KirOp::Add, r0, chunk_rows);
    let r1 = op2(&mut b, U64, KirOp::Min, r_end, rows);
    let first = op2(&mut b, U64, KirOp::Mul, r0, cols);
    let at0 = op2(&mut b, U64, KirOp::Add, first, j);
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    let one = u64c(&mut b, 1);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![r0, at0, zero])));

    b.set_block(head);
    let end = cmp(&mut b, i, r1, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let g = load_at(&mut b, Global, dy, at);
    let v = load_at(&mut b, Global, x, at);
    let ri = load_at(&mut b, Global, rinv, i);
    let p = op2(&mut b, F32, KirOp::MulRn, g, v);
    let q = op2(&mut b, F32, KirOp::MulRn, p, ri);
    let acc_next = add_rn(&mut b, acc, q);
    let at_next = op2(&mut b, U64, KirOp::Add, at, cols);
    let i_next = op2(&mut b, U64, KirOp::Add, i, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, at_next, acc_next])));

    b.set_block(done);
    let row = op2(&mut b, U64, KirOp::Mul, c, cols);
    let out = op2(&mut b, U64, KirOp::Add, row, j);
    store_at(&mut b, Global, partial, out, acc);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    finish(b, exit, PARTIAL_BLOCK)
}

/// ```text
/// entry: lane = tid & 31; warp = tid >> 5; j = %ctaid.x·32 + lane
///        c0 = j < cols ? warp : chunks      (a lane past the columns runs no trips)
///        head(c = c0, at = c0·cols + j, acc = +0.0)
/// head:  if c >= chunks { red }
/// body:  acc += partial[at]; c += 8; at += 8·cols
/// red:   swarp[tid] = acc; barrier; if tid >= 32 { exit }
/// live:  if j >= cols { exit }
/// fin:   dgamma[j] = tree(swarp[w·32 + lane])
/// ```
fn build_final() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let op = RmsNormDgammaOp::Final;
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let partial = b.add_param(names[0], f32_ptr(), Global);
    let dgamma = b.add_param(names[1], f32_ptr(), Global);
    let chunks = b.add_param(names[2], U64, Global);
    let cols = b.add_param(names[3], U64, Global);
    b.set_smem_layout(smem(FINAL_BLOCK));

    let entry = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let red = b.new_block();
    let live = b.new_block();
    let fin = b.new_block();
    let exit = b.new_block();
    let c = b.add_block_param(head, U64);
    let at = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);

    b.set_block(entry);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let lane_mask = u32c(&mut b, FINAL_COLS - 1);
    let lane32 = op2(&mut b, U32, KirOp::And, tid32, lane_mask);
    let five = u32c(&mut b, 5);
    let warp32 = op2(&mut b, U32, KirOp::Shr, tid32, five);
    let blk32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(blk32, 0));
    let blk = cast(&mut b, blk32, U64);
    let width = u64c(&mut b, u64::from(FINAL_COLS));
    let first_col = op2(&mut b, U64, KirOp::Mul, blk, width);
    let lane = cast(&mut b, lane32, U64);
    let j = op2(&mut b, U64, KirOp::Add, first_col, lane);
    let in_cols = cmp(&mut b, j, cols, CmpOp::Lt);
    let warp = cast(&mut b, warp32, U64);
    let c0 = var(&mut b, U64);
    b.emit(KirOp::Select(c0, in_cols, warp, chunks));
    let first = op2(&mut b, U64, KirOp::Mul, c0, cols);
    let at0 = op2(&mut b, U64, KirOp::Add, first, j);
    let warps = u64c(&mut b, u64::from(WARPS));
    let step = op2(&mut b, U64, KirOp::Mul, warps, cols);
    let sm = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: sm, region: 0 });
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c0, at0, zero])));

    b.set_block(head);
    let end = cmp(&mut b, c, chunks, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(red), KirEdge::to(body)));

    b.set_block(body);
    let p = load_at(&mut b, Global, partial, at);
    let acc_next = add_rn(&mut b, acc, p);
    let c_next = op2(&mut b, U64, KirOp::Add, c, warps);
    let at_next = op2(&mut b, U64, KirOp::Add, at, step);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c_next, at_next, acc_next])));

    b.set_block(red);
    store_at(&mut b, Shared, sm, tid32, acc);
    b.emit(KirOp::Barrier);
    let lanes = u32c(&mut b, FINAL_COLS);
    let not_warp0 = cmp(&mut b, tid32, lanes, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(not_warp0, KirEdge::to(exit), KirEdge::to(live)));

    b.set_block(live);
    let past = cmp(&mut b, j, cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(fin)));

    b.set_block(fin);
    let total = tree_of_eight(&mut b, sm, lane32, FINAL_COLS);
    store_at(&mut b, Global, dgamma, j, total);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    finish(b, exit, FINAL_BLOCK)
}

fn finish(mut b: KirBuilder, exit: BlockId, block: u32) -> KernelIR {
    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([block, 1, 1]);
    b.set_launch_bounds(block, None);
    b.finalize()
}

/// Build `op` as KIR.
pub fn build(op: RmsNormDgammaOp) -> KernelIR {
    match op {
        RmsNormDgammaOp::RinvRows => build_rinv_rows(),
        RmsNormDgammaOp::Partial => build_partial(),
        RmsNormDgammaOp::Final => build_final(),
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
            assert!(p.contains(&format!(".maxntid {}, 1, 1", op.block())), "{p}");
        }
    }

    /// Every float operation is explicitly rounded, so none fuses or
    /// approximates, and the counts are the documented ones: kernel 1 has
    /// one `fma`, five butterfly adds, seven tree adds and `+ eps`; kernel 2
    /// two products and the accumulate; kernel 3 the accumulate and the tree.
    #[test]
    fn every_float_operation_is_correctly_rounded() {
        let r = text(RmsNormDgammaOp::RinvRows);
        assert_eq!(r.matches("fma.rn.f32").count(), 1, "{r}");
        assert_eq!(r.matches("add.rn.f32").count(), 5 + 7 + 1, "{r}");
        assert_eq!(r.matches("div.rn.f32").count(), 2, "{r}");
        assert_eq!(r.matches("sqrt.rn.f32").count(), 1, "{r}");
        assert_eq!(r.matches("shfl.sync.bfly.b32").count(), 5, "{r}");
        assert!(r.contains("cvt.rn.f32.u64"), "{r}");
        let p = text(RmsNormDgammaOp::Partial);
        assert_eq!(p.matches("mul.rn.f32").count(), 2, "{p}");
        assert_eq!(p.matches("add.rn.f32").count(), 1, "{p}");
        let f = text(RmsNormDgammaOp::Final);
        assert_eq!(f.matches("add.rn.f32").count(), 1 + 7, "{f}");
        for k in [&r, &p, &f] {
            assert!(!k.contains("approx") && !k.contains(" atom.") && !k.contains(" red."), "{k}");
            assert!(!k.lines().any(|l| l.contains(".f32 ") && (l.contains(" add.f32") || l.contains(" mul.f32"))), "{k}");
        }
    }

    /// The two kernels with a block-wide sum have one barrier and the
    /// shared region its warps write.
    #[test]
    fn the_block_sums_have_one_barrier_and_their_region() {
        for (op, bytes) in [(RmsNormDgammaOp::RinvRows, WARPS * 4), (RmsNormDgammaOp::Final, FINAL_BLOCK * 4)] {
            let p = text(op);
            assert_eq!(p.matches("bar.sync 0;").count(), 1, "{p}");
            assert!(p.contains(&format!(".shared .align 4 .b8 shared_mem[{bytes}]")), "{p}");
        }
        let p = text(RmsNormDgammaOp::Partial);
        assert!(!p.contains("bar.sync") && !p.contains(".shared"), "{p}");
        assert!(p.contains("%ctaid.y"), "{p}");
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

    /// The plan at the recipes' shapes (`rows` = batch · sequence), worked
    /// by hand from its definition.
    #[test]
    fn the_plan_at_the_recipe_shapes() {
        // Coder-1B: 2 × 2048 rows of 2048. 8 column blocks, 32 chunks of 128.
        assert_eq!(dgamma_plan(4096, 2048), DgammaPlan { col_blocks: 8, chunk_rows: 128, chunks: 32 });
        // Coder-500M: 2 × 1024 rows of 1280. 5 column blocks; ceil(256 / 5)
        // = 52 target chunks of ceil(2048 / 52) = 40 rows; 52 chunks.
        assert_eq!(dgamma_plan(2048, 1280), DgammaPlan { col_blocks: 5, chunk_rows: 40, chunks: 52 });
        // Coder-50M: 1 × 1024 rows of 512: 128 target chunks of 8 rows.
        assert_eq!(dgamma_plan(1024, 512), DgammaPlan { col_blocks: 2, chunk_rows: 8, chunks: 128 });
        // Tails: one row; no rows; a column count past the target.
        assert_eq!(dgamma_plan(1, 1), DgammaPlan { col_blocks: 1, chunk_rows: 8, chunks: 1 });
        assert_eq!(dgamma_plan(0, 7), DgammaPlan { col_blocks: 1, chunk_rows: 8, chunks: 0 });
        assert_eq!(dgamma_plan(100, 300_000), DgammaPlan { col_blocks: 1172, chunk_rows: 100, chunks: 1 });
        // The depth at Coder-1B: 2 + 128 + 4 + 3.
        assert_eq!(dgamma_plan(4096, 2048).depth(), 137);
    }

    #[test]
    fn the_bounds_are_the_derived_ones() {
        let u = UNIT_ROUNDOFF;
        assert_eq!(gamma(1), u / (1.0 - u));
        // cols = 2048: m = 8, so γ_18 under the square root.
        let want = (1.0 + u) / ((1.0 - u) * (1.0 - gamma(18)).sqrt()) - 1.0;
        assert_eq!(rinv_relative_bound(2048), want);
        assert!(want > 10.0 * u && want < 12.0 * u, "{want}");
    }
}
