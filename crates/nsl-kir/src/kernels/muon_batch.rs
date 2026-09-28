// crates/nsl-kir/src/kernels/muon_batch.rs
//! The runtime's batched Muon Newton-Schulz kernels from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5).
//!
//! `nsl_runtime::muon_batch` updates `k` same-shape `r × c` matrices with a
//! fixed launch sequence over persistent workspaces. It reaches the
//! matrices through device pointer tables (a `u64` array of data pointers
//! per group), so `k` tensors at arbitrary addresses batch without being
//! repacked. Every kernel but the reduction runs a thread per element on a
//! `(ceil(n / 256), k)` grid of 256-thread blocks, with `%ctaid.y` naming the
//! matrix:
//!
//! - `nsl_muon_batch_mom_f32(mtab, gtab, mu, n)`: `m = mu·m + g` in place.
//! - `nsl_muon_batch_sumsq_f32(mtab, gtab, mu, nest, n, norms)`: one
//!   256-thread block per matrix sums `u²` into `norms[matrix]`, where `u =
//!   nest ? g + mu·m : m`. Each thread accumulates a stride-256 slice, then a
//!   128-step shared-memory tree adds the partials. The order is fixed, so
//!   the sum is deterministic.
//! - `nsl_muon_batch_pack_f32(mtab, gtab, mu, nest, norms, ybase, r, c, tr,
//!   eps)`: `Y = u · (1 / (sqrt(norms[matrix]) + eps))`, written to the
//!   matrix's slice of the workspace. With `tr` set, it is transposed on the
//!   way (`[r, c]` → `[c, r]`).
//! - `nsl_muon_batch_poly_f32(abase, aabase, nsa, nsb, nsc, rdim, r2)`: the
//!   polynomial combine in place over the Gram workspace, `A = nsb·A +
//!   nsc·AA`, plus `nsa` on the diagonal.
//! - `nsl_muon_batch_update_f32(ptab, ybase, r, c, tr, decay, step)`: `p =
//!   decay·p − step·o`, with `o` read from the workspace (transposed back
//!   when `tr` is set).
//!
//! Every multiply and add is explicitly rounded (`MulRn`, `AddRn`, `SubRn`),
//! as in the hand kernels, so no pair fuses into an `fma`. The momentum
//! update is bit-exact against the stdlib `muon_step` arm, and the sums
//! against the sequential Frobenius path. Each kernel keeps its hand
//! kernel's order of loads, operations and stores.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator,
    KirType, SmemLayout, SmemRegion, VarId,
};

/// The block every kernel is launched with. The reduction strides by it and
/// sizes its shared buffer to it.
pub const MUON_BATCH_BLOCK: u32 = 256;

/// One of the five kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuonBatchOp {
    Mom,
    Sumsq,
    Pack,
    Poly,
    Update,
}

impl MuonBatchOp {
    pub const ALL: [MuonBatchOp; 5] =
        [MuonBatchOp::Mom, MuonBatchOp::Sumsq, MuonBatchOp::Pack, MuonBatchOp::Poly, MuonBatchOp::Update];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            MuonBatchOp::Mom => "nsl_muon_batch_mom_f32",
            MuonBatchOp::Sumsq => "nsl_muon_batch_sumsq_f32",
            MuonBatchOp::Pack => "nsl_muon_batch_pack_f32",
            MuonBatchOp::Poly => "nsl_muon_batch_poly_f32",
            MuonBatchOp::Update => "nsl_muon_batch_update_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            MuonBatchOp::Mom => &["mtab", "gtab", "mu", "n"],
            MuonBatchOp::Sumsq => &["mtab", "gtab", "mu", "nest", "n", "norms"],
            MuonBatchOp::Pack => &["mtab", "gtab", "mu", "nest", "norms", "ybase", "r", "c", "tr", "eps"],
            MuonBatchOp::Poly => &["abase", "aabase", "nsa", "nsb", "nsc", "rdim", "r2"],
            MuonBatchOp::Update => &["ptab", "ybase", "r", "c", "tr", "decay", "step"],
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

/// `base[i]` for an element type `elem` (`base` a global pointer to it).
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

/// A pointer table: `u64` data pointers to f32 arrays.
fn table() -> KirType {
    KirType::Ptr(Box::new(f32_ptr()), AddressSpace::Global)
}

/// `%ctaid.dim`, widened.
fn matrix(b: &mut KirBuilder, dim: u8) -> VarId {
    let m = var(b, KirType::U32);
    b.emit(KirOp::BlockIdx(m, dim));
    cast(b, m, KirType::U64)
}

/// `i = %ctaid.x · %ntid.x + %tid.x` (u32); `if i >= n { exit }`. Returns
/// `(i, body, exit)` with the builder in `body`.
fn element(b: &mut KirBuilder, entry: BlockId, n: impl FnOnce(&mut KirBuilder) -> VarId) -> (VarId, BlockId, BlockId) {
    let body = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let n = n(b);
    let i = var(b, KirType::U32);
    b.emit(KirOp::GlobalId(i, 0));
    let past = cmp(b, i, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));
    b.set_block(body);
    (i, body, exit)
}

/// `u = nest ? g + mu·m : m`, branching on `nest_on` from the current
/// block. Returns `u` with the builder in the join block.
fn update_direction(b: &mut KirBuilder, nest_on: VarId, x: VarId, gp: VarId, i: VarId, mu: VarId) -> VarId {
    use AddressSpace::Global;
    let nest = b.new_block();
    let join = b.new_block();
    let u = b.add_block_param(join, KirType::F32);
    b.terminate(KirTerminator::CondBranch(nest_on, KirEdge::to(nest), KirEdge::with(join, vec![x])));
    b.set_block(nest);
    let g = load_at(b, KirType::F32, Global, gp, i);
    let t = op2(b, KirType::F32, KirOp::MulRn, x, mu);
    let s = op2(b, KirType::F32, KirOp::AddRn, g, t);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![s])));
    b.set_block(join);
    u
}

/// The workspace index of element `i`: `i` itself, or with `tr` set its
/// transpose `(i % c)·r + i / c`, from the current block. Returns it with
/// the builder in the join block.
fn workspace_index(b: &mut KirBuilder, tr: VarId, i: VarId, r: VarId, c: VarId) -> VarId {
    let tpose = b.new_block();
    let join = b.new_block();
    let idx = b.add_block_param(join, KirType::U32);
    let zero = konst(b, KirType::U32, ConstValue::U32(0));
    let plain = cmp(b, tr, zero, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(plain, KirEdge::with(join, vec![i]), KirEdge::to(tpose)));
    b.set_block(tpose);
    let row = op2(b, KirType::U32, KirOp::Div, i, c);
    let start = op2(b, KirType::U32, KirOp::Mul, row, c);
    let col = op2(b, KirType::U32, KirOp::Sub, i, start);
    let t = op2(b, KirType::U32, KirOp::Mul, col, r);
    let t = op2(b, KirType::U32, KirOp::Add, t, row);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![t])));
    b.set_block(join);
    idx
}

/// `matrix · per + idx`, in 64 bits: an element's place in a `[k, per]`
/// workspace.
fn slot(b: &mut KirBuilder, mat: VarId, per: VarId, idx: VarId) -> VarId {
    let per = cast(b, per, KirType::U64);
    let base = op2(b, KirType::U64, KirOp::Mul, mat, per);
    let idx = cast(b, idx, KirType::U64);
    op2(b, KirType::U64, KirOp::Add, base, idx)
}

fn finish(mut b: KirBuilder, exit: BlockId) -> KernelIR {
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([MUON_BATCH_BLOCK, 1, 1]);
    b.set_launch_bounds(MUON_BATCH_BLOCK, None);
    b.finalize()
}

/// Build `op` as KIR.
pub fn build(op: MuonBatchOp) -> KernelIR {
    match op {
        MuonBatchOp::Mom => build_mom(),
        MuonBatchOp::Sumsq => build_sumsq(),
        MuonBatchOp::Pack => build_pack(),
        MuonBatchOp::Poly => build_poly(),
        MuonBatchOp::Update => build_update(),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: MuonBatchOp) -> Vec<u8> {
    verified_ptx(build(op))
}

/// ```text
/// entry: i; if i >= n { exit }
/// body:  mp = mtab[ctaid.y]; gp = gtab[ctaid.y]; mp[i] = mu·mp[i] + gp[i]
/// ```
fn build_mom() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(MuonBatchOp::Mom.kernel_name());
    let mtab = b.add_param("mtab", table(), Global);
    let gtab = b.add_param("gtab", table(), Global);
    let mu = b.add_param("mu", F32, Global);
    let n = b.add_param("n", U32, Global);
    let entry = b.new_block();
    let (i, _, exit) = element(&mut b, entry, |_| n);
    let mat = matrix(&mut b, 1);
    let mp = load_at(&mut b, f32_ptr(), Global, mtab, mat);
    let gp = load_at(&mut b, f32_ptr(), Global, gtab, mat);
    let iw = cast(&mut b, i, U64);
    let m = load_at(&mut b, F32, Global, mp, iw);
    let g = load_at(&mut b, F32, Global, gp, iw);
    let t = op2(&mut b, F32, KirOp::MulRn, m, mu);
    let v = op2(&mut b, F32, KirOp::AddRn, t, g);
    store_at(&mut b, F32, Global, mp, iw, v);
    finish(b, exit)
}

/// ```text
/// entry:        mat = ctaid.x; mp, gp; nest ? br nacc(tid, 0) : br pacc(tid, 0)
/// pacc(i, s):   if i >= n { br pout }; x = mp[i]; br pacc(i + 256, s + x·x)
/// pout:         sm[tid] = s; br red
/// nacc(i, s):   if i >= n { br nout }
///               u = gp[i] + mu·mp[i]; br nacc(i + 256, s + u·u)
/// nout:         sm[tid] = s; br red
/// red:          bar; br tree(128)
/// tree(h):      if h < 1 { done }; if tid >= h { skip }
/// add:          sm[tid] = sm[tid] + sm[tid + h]
/// skip:         bar; br tree(h >> 1)
/// done:         if tid == 0 { norms[mat] = sm[0] }
/// ```
///
/// The hand kernel tests `nest` inside its loop. Here the test is hoisted,
/// and each setting gets its own straight-line loop. ptxas unrolls these
/// as it did the hand kernel's predicated loop, keeping several loads in
/// flight. The additions happen in the same order either way.
///
/// Each loop leaves through an exit block of its own, which publishes the
/// partial. The exit edge then carries no block arguments, so its copies
/// are not printed inline in the loop header. With them inline, ptxas does
/// not see a loop it can unroll (the reduction is one block per matrix and
/// latency-bound, so that matters).
fn build_sumsq() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(MuonBatchOp::Sumsq.kernel_name());
    let mtab = b.add_param("mtab", table(), Global);
    let gtab = b.add_param("gtab", table(), Global);
    let mu = b.add_param("mu", F32, Global);
    let nest = b.add_param("nest", U32, Global);
    let n = b.add_param("n", U32, Global);
    let norms = b.add_param("norms", f32_ptr(), Global);
    b.set_smem_layout(SmemLayout {
        regions: vec![SmemRegion { name: "mbss".to_string(), bytes: MUON_BATCH_BLOCK * 4, align: 4, elem: F32 }],
        dynamic: false,
    });

    let entry = b.new_block();
    let red = b.new_block();
    let tree = b.new_block();
    let step = b.new_block();
    let add = b.new_block();
    let skip = b.new_block();
    let done = b.new_block();
    let write = b.new_block();
    let exit = b.new_block();
    let half = b.add_block_param(tree, U32);

    b.set_block(entry);
    let mat = matrix(&mut b, 0);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let mp = load_at(&mut b, f32_ptr(), Global, mtab, mat);
    let gp = load_at(&mut b, f32_ptr(), Global, gtab, mat);
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    let tid = cast(&mut b, tid32, U64);
    let n64 = cast(&mut b, n, U64);
    let u_zero = konst(&mut b, U32, ConstValue::U32(0));
    let nest_on = cmp(&mut b, nest, u_zero, CmpOp::Ne);
    let stride = konst(&mut b, U64, ConstValue::U64(MUON_BATCH_BLOCK as u64));
    let sm = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: sm, region: 0 });

    // One accumulation loop per `nest` setting: `u = x` or `u = g + mu·x`.
    let accumulate = |b: &mut KirBuilder, with_g: bool| {
        let head = b.new_block();
        let body = b.new_block();
        let out = b.new_block();
        let i = b.add_block_param(head, U64);
        let s = b.add_block_param(head, F32);
        b.set_block(head);
        let end = cmp(b, i, n64, CmpOp::Ge);
        b.terminate(KirTerminator::CondBranch(end, KirEdge::to(out), KirEdge::to(body)));
        b.set_block(body);
        let x = load_at(b, F32, Global, mp, i);
        let u = if with_g {
            let g = load_at(b, F32, Global, gp, i);
            let t = op2(b, F32, KirOp::MulRn, x, mu);
            op2(b, F32, KirOp::AddRn, g, t)
        } else {
            x
        };
        let sq = op2(b, F32, KirOp::MulRn, u, u);
        let s_next = op2(b, F32, KirOp::AddRn, s, sq);
        let i_next = op2(b, U64, KirOp::Add, i, stride);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, s_next])));
        b.set_block(out);
        store_at(b, F32, Shared, sm, tid32, s);
        b.terminate(KirTerminator::Branch(KirEdge::to(red)));
        head
    };
    let nacc = accumulate(&mut b, true);
    let pacc = accumulate(&mut b, false);
    b.set_block(entry);
    b.terminate(KirTerminator::CondBranch(
        nest_on,
        KirEdge::with(nacc, vec![tid, zero]),
        KirEdge::with(pacc, vec![tid, zero]),
    ));

    b.set_block(red);
    b.emit(KirOp::Barrier);
    let first = konst(&mut b, U32, ConstValue::U32(MUON_BATCH_BLOCK / 2));
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![first])));

    b.set_block(tree);
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let over = cmp(&mut b, half, one, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(over, KirEdge::to(done), KirEdge::to(step)));

    b.set_block(step);
    let idle = cmp(&mut b, tid32, half, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(idle, KirEdge::to(skip), KirEdge::to(add)));

    b.set_block(add);
    let mine = load_at(&mut b, F32, Shared, sm, tid32);
    let other = op2(&mut b, U32, KirOp::Add, tid32, half);
    let theirs = load_at(&mut b, F32, Shared, sm, other);
    let sum = op2(&mut b, F32, KirOp::AddRn, mine, theirs);
    store_at(&mut b, F32, Shared, sm, tid32, sum);
    b.terminate(KirTerminator::Branch(KirEdge::to(skip)));

    b.set_block(skip);
    b.emit(KirOp::Barrier);
    let next = op2(&mut b, U32, KirOp::Shr, half, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(tree, vec![next])));

    b.set_block(done);
    let not_first = cmp(&mut b, tid32, u_zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(exit), KirEdge::to(write)));

    b.set_block(write);
    let total = var(&mut b, F32);
    b.emit(KirOp::Load(total, sm, Shared));
    store_at(&mut b, F32, Global, norms, mat, total);
    finish(b, exit)
}

/// ```text
/// entry: n = r·c; i; if i >= n { exit }
/// body:  mp, gp; x = mp[i]; u = nest ? gp[i] + mu·x : x
/// norm:  y = u · (1 / (sqrt(norms[mat]) + eps)); d = tr ? transpose(i) : i
/// store: ybase[mat·n + d] = y
/// ```
fn build_pack() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(MuonBatchOp::Pack.kernel_name());
    let mtab = b.add_param("mtab", table(), Global);
    let gtab = b.add_param("gtab", table(), Global);
    let mu = b.add_param("mu", F32, Global);
    let nest = b.add_param("nest", U32, Global);
    let norms = b.add_param("norms", f32_ptr(), Global);
    let ybase = b.add_param("ybase", f32_ptr(), Global);
    let r = b.add_param("r", U32, Global);
    let c = b.add_param("c", U32, Global);
    let tr = b.add_param("tr", U32, Global);
    let eps = b.add_param("eps", F32, Global);
    let entry = b.new_block();
    let mut n = None;
    let (i, _, exit) = element(&mut b, entry, |b| *n.insert(op2(b, U32, KirOp::Mul, r, c)));
    let n = n.expect("the element count");
    let mat = matrix(&mut b, 1);
    let mp = load_at(&mut b, f32_ptr(), Global, mtab, mat);
    let gp = load_at(&mut b, f32_ptr(), Global, gtab, mat);
    let iw = cast(&mut b, i, U64);
    let x = load_at(&mut b, F32, Global, mp, iw);
    let u_zero = konst(&mut b, U32, ConstValue::U32(0));
    let nest_on = cmp(&mut b, nest, u_zero, CmpOp::Ne);
    let u = update_direction(&mut b, nest_on, x, gp, iw, mu);
    let nv = load_at(&mut b, F32, Global, norms, mat);
    let root = var(&mut b, F32);
    b.emit(KirOp::Sqrt(root, nv));
    let denom = op2(&mut b, F32, KirOp::AddRn, root, eps);
    let one = konst(&mut b, F32, ConstValue::F32(1.0));
    let inv = op2(&mut b, F32, KirOp::Div, one, denom);
    let y = op2(&mut b, F32, KirOp::MulRn, u, inv);
    let d = workspace_index(&mut b, tr, i, r, c);
    let at = slot(&mut b, mat, n, d);
    store_at(&mut b, F32, Global, ybase, at, y);
    finish(b, exit)
}

/// ```text
/// entry: i; if i >= r2 { exit }
/// body:  o = ctaid.y·r2 + i; v = nsb·A[o] + nsc·AA[o]
///        A[o] = (i / rdim == i % rdim) ? v + nsa : v
/// ```
fn build_poly() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32};
    let mut b = KirBuilder::new(MuonBatchOp::Poly.kernel_name());
    let abase = b.add_param("abase", f32_ptr(), Global);
    let aabase = b.add_param("aabase", f32_ptr(), Global);
    let nsa = b.add_param("nsa", F32, Global);
    let nsb = b.add_param("nsb", F32, Global);
    let nsc = b.add_param("nsc", F32, Global);
    let rdim = b.add_param("rdim", U32, Global);
    let r2 = b.add_param("r2", U32, Global);
    let entry = b.new_block();
    let (i, _, exit) = element(&mut b, entry, |_| r2);
    let mat = matrix(&mut b, 1);
    let o = slot(&mut b, mat, r2, i);
    let a = load_at(&mut b, F32, Global, abase, o);
    let aa = load_at(&mut b, F32, Global, aabase, o);
    let t1 = op2(&mut b, F32, KirOp::MulRn, a, nsb);
    let t2 = op2(&mut b, F32, KirOp::MulRn, aa, nsc);
    let v = op2(&mut b, F32, KirOp::AddRn, t1, t2);
    let row = op2(&mut b, U32, KirOp::Div, i, rdim);
    let start = op2(&mut b, U32, KirOp::Mul, row, rdim);
    let col = op2(&mut b, U32, KirOp::Sub, i, start);
    let diag = cmp(&mut b, row, col, CmpOp::Eq);
    let shifted = op2(&mut b, F32, KirOp::AddRn, v, nsa);
    let w = var(&mut b, F32);
    b.emit(KirOp::Select(w, diag, shifted, v));
    store_at(&mut b, F32, Global, abase, o, w);
    finish(b, exit)
}

/// ```text
/// entry: n = r·c; i; if i >= n { exit }
/// body:  pp = ptab[ctaid.y]; s = tr ? transpose(i) : i
/// load:  o = ybase[mat·n + s]; pp[i] = decay·pp[i] − step·o
/// ```
fn build_update() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(MuonBatchOp::Update.kernel_name());
    let ptab = b.add_param("ptab", table(), Global);
    let ybase = b.add_param("ybase", f32_ptr(), Global);
    let r = b.add_param("r", U32, Global);
    let c = b.add_param("c", U32, Global);
    let tr = b.add_param("tr", U32, Global);
    let decay = b.add_param("decay", F32, Global);
    let step = b.add_param("step", F32, Global);
    let entry = b.new_block();
    let mut n = None;
    let (i, _, exit) = element(&mut b, entry, |b| *n.insert(op2(b, U32, KirOp::Mul, r, c)));
    let n = n.expect("the element count");
    let mat = matrix(&mut b, 1);
    let pp = load_at(&mut b, f32_ptr(), Global, ptab, mat);
    let s = workspace_index(&mut b, tr, i, r, c);
    let at = slot(&mut b, mat, n, s);
    let o = load_at(&mut b, F32, Global, ybase, at);
    let iw = cast(&mut b, i, U64);
    let p = load_at(&mut b, F32, Global, pp, iw);
    let t1 = op2(&mut b, F32, KirOp::MulRn, p, decay);
    let t2 = op2(&mut b, F32, KirOp::MulRn, step, o);
    let v = op2(&mut b, F32, KirOp::SubRn, t1, t2);
    store_at(&mut b, F32, Global, pp, iw, v);
    finish(b, exit)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: MuonBatchOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in MuonBatchOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}: NUL-terminated");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
        }
    }

    /// Every float multiply and add is explicitly rounded, so none fuses.
    #[test]
    fn no_float_op_can_contract() {
        for op in MuonBatchOp::ALL {
            let p = text(op);
            for bare in ["mul.f32", "add.f32", "sub.f32", "fma."] {
                assert!(!p.contains(bare), "{op:?}: {bare}\n{p}");
            }
        }
    }

    /// Each accumulation loop's bound test branches straight out (`@%p bra`
    /// to an exit block), with no edge copies printed inline in the header.
    /// With copies inline, ptxas stops unrolling the loop.
    #[test]
    fn the_reduction_loops_exit_without_inline_copies() {
        let p = text(MuonBatchOp::Sumsq);
        let lines: Vec<&str> = p.lines().map(str::trim).collect();
        let heads: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("setp.ge.u64 ")).collect();
        assert_eq!(heads.len(), 2, "{p}");
        for i in heads {
            assert!(lines[i + 1].starts_with("@%p") && lines[i + 1].contains(" bra "), "{}\n{p}", lines[i + 1]);
            assert!(lines[i + 2].starts_with("bra "), "{}\n{p}", lines[i + 2]);
        }
    }

    #[test]
    fn only_the_reduction_uses_shared_memory_and_barriers() {
        for op in MuonBatchOp::ALL {
            let p = text(op);
            assert_eq!(p.contains("bar.sync"), op == MuonBatchOp::Sumsq, "{op:?}");
            assert_eq!(p.contains(".shared"), op == MuonBatchOp::Sumsq, "{op:?}");
        }
    }
}
