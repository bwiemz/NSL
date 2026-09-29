// crates/nsl-kir/src/kernels/norm.rs
//! The runtime's row LayerNorm and RMSNorm forwards from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5):
//! `nsl_layernorm_f32(inp, out, gamma, beta, rows, cols, eps)` and
//! `nsl_rmsnorm_f32(inp, out, gamma, rows, cols, eps)`.
//!
//! One block of [`NORM_BLOCK`] threads per row (`%ctaid.x`); a block at or
//! past `rows` does nothing. Each thread strides over the row's columns
//! (`k = tid, tid + 256, …`). A statistic is reduced by storing each
//! thread's partial to shared memory; after a barrier, thread 0 folds the
//! partials `1 .. %ntid.x` into its own, in order, finishes the statistic
//! and stores it to the region's first slot, which every thread reads after
//! a second barrier.
//!
//! - **LayerNorm:** `mean = Σx / cols` (`div.approx`), then `inv = rsqrt(Σ(x
//!   - mean)² / cols + eps)`, then `out = ((x - mean) · inv) · gamma + beta`.
//! - **RMSNorm:** `inv = rsqrt(Σx² / cols + eps)`, then `out = (x · inv) ·
//!   gamma`.
//!
//! The hand LayerNorm reduced both statistics through one 256-float region.
//! Thread 0 stored its variance partial to the slot that held the mean while
//! other warps could still be about to read the mean (there is no barrier
//! between the read and the next pass's store), so a warp that fell a pass
//! behind normalized with thread 0's variance partial in place of the mean.
//! The mean and the variance have a region each here, as the softmax's max
//! and sum do.
//!
//! Every add, subtract and multiply rounds explicitly (`.rn`), as the hand
//! PTX spells them. ptxas contracted the hand kernels' squares and
//! `· gamma + beta` into `fma` on hardware; these kernels round twice, as
//! the PTX (and the CPU reference) do. Thread 0 reads `%ntid.x` once, before
//! its fold, so ptxas unrolls the fold as it does the hand kernels'.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// The block the runtime launches with: one per row. The column loops
/// stride by it.
pub const NORM_BLOCK: u32 = 256;

/// Which kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormOp {
    /// `out = ((x - mean) · rsqrt(var + eps)) · gamma + beta`.
    LayerNorm,
    /// `out = (x · rsqrt(mean(x²) + eps)) · gamma`.
    RmsNorm,
}

impl NormOp {
    pub const ALL: [NormOp; 2] = [NormOp::LayerNorm, NormOp::RmsNorm];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            NormOp::LayerNorm => "nsl_layernorm_f32",
            NormOp::RmsNorm => "nsl_rmsnorm_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn params(self) -> &'static [&'static str] {
        match self {
            NormOp::LayerNorm => &["inp", "out", "gamma", "beta", "rows", "cols", "eps"],
            NormOp::RmsNorm => &["inp", "out", "gamma", "rows", "cols", "eps"],
        }
    }

    /// The shared regions: one per statistic.
    fn regions(self) -> &'static [&'static str] {
        match self {
            NormOp::LayerNorm => &["smean", "svar"],
            NormOp::RmsNorm => &["ssq"],
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
/// exit's `acc`; the exit block is current on return.
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
    let stride = konst(b, U64, ConstValue::U64(u64::from(NORM_BLOCK)));
    let k_next = op2(b, U64, KirOp::Add, k, stride);
    let mut args = vec![k_next];
    args.extend(acc_next);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, args)));

    b.set_block(exit);
    acc
}

/// From the current block, holding the thread's `partial` sum: store it to
/// `sm[tid]`, barrier; thread 0 reads `%ntid.x`, adds `sm[1 .. %ntid.x]` to
/// it in order and stores `finish(total)` to `sm[0]`; barrier; everyone
/// reads `sm[0]`, which is returned (the current block is then the one
/// after).
fn block_sum(
    b: &mut KirBuilder,
    sm: VarId,
    tid32: VarId,
    partial: VarId,
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
    let acc_next = op2(b, F32, KirOp::AddRn, acc, other);
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

/// `total / cols` with `div.approx`, `cols` converted with `cvt.rn`.
fn per_column(b: &mut KirBuilder, total: VarId, cols: VarId) -> VarId {
    let n = var(b, KirType::F32);
    b.emit(KirOp::Cast(n, cols, KirType::F32));
    op2(b, KirType::F32, KirOp::DivApprox, total, n)
}

/// `rsqrt(total / cols + eps)`.
fn inv_std(b: &mut KirBuilder, total: VarId, cols: VarId, eps: VarId) -> VarId {
    let m = per_column(b, total, cols);
    let v = op2(b, KirType::F32, KirOp::AddRn, m, eps);
    op1(b, KirType::F32, KirOp::Rsqrt, v)
}

/// ```text
/// entry: row = %ctaid.x; if row >= rows { exit }
/// setup: in = inp + row·cols; o = out + row·cols
/// LayerNorm: mean = block_sum(smean, Σ in[k]) / cols
///            inv  = rsqrt(block_sum(svar, Σ (in[k] - mean)²) / cols + eps)
///            o[k] = ((in[k] - mean) · inv) · gamma[k] + beta[k]
/// RMSNorm:   inv  = rsqrt(block_sum(ssq, Σ in[k]²) / cols + eps)
///            o[k] = (in[k] · inv) · gamma[k]
/// ```
pub fn build(op: NormOp) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let names = op.params();
    let mut b = KirBuilder::new(op.kernel_name());
    let inp = b.add_param(names[0], f32_ptr(), Global);
    let out = b.add_param(names[1], f32_ptr(), Global);
    let gamma = b.add_param(names[2], f32_ptr(), Global);
    let beta = (op == NormOp::LayerNorm).then(|| b.add_param(names[3], f32_ptr(), Global));
    let at = names.len() - 3;
    let rows = b.add_param(names[at], U64, Global);
    let cols = b.add_param(names[at + 1], U64, Global);
    let eps = b.add_param(names[at + 2], F32, Global);
    let regions = op
        .regions()
        .iter()
        .map(|name| SmemRegion { name: name.to_string(), bytes: NORM_BLOCK * 4, align: 4, elem: F32 })
        .collect();
    b.set_smem_layout(SmemLayout { regions, dynamic: false });

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
    let sm: Vec<VarId> = (0..op.regions().len() as u32)
        .map(|region| {
            let p = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
            b.emit(KirOp::SharedRegion { dst: p, region });
            p
        })
        .collect();
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));

    match op {
        NormOp::LayerNorm => {
            let s = column_loop(&mut b, tid, cols, Some(zero), |b, k, acc| {
                let x = load_at(b, Global, in_row, k);
                Some(op2(b, F32, KirOp::AddRn, acc.expect("acc"), x))
            })
            .expect("acc");
            let mean = block_sum(&mut b, sm[0], tid32, s, |b, total| per_column(b, total, cols));
            let zero = konst(&mut b, F32, ConstValue::F32(0.0));
            let q = column_loop(&mut b, tid, cols, Some(zero), |b, k, acc| {
                let x = load_at(b, Global, in_row, k);
                let d = op2(b, F32, KirOp::SubRn, x, mean);
                let sq = op2(b, F32, KirOp::MulRn, d, d);
                Some(op2(b, F32, KirOp::AddRn, acc.expect("acc"), sq))
            })
            .expect("acc");
            let inv = block_sum(&mut b, sm[1], tid32, q, |b, total| inv_std(b, total, cols, eps));
            let beta = beta.expect("LayerNorm has beta");
            column_loop(&mut b, tid, cols, None, |b, k, _| {
                let x = load_at(b, Global, in_row, k);
                let d = op2(b, F32, KirOp::SubRn, x, mean);
                let n = op2(b, F32, KirOp::MulRn, d, inv);
                let g = load_at(b, Global, gamma, k);
                let scaled = op2(b, F32, KirOp::MulRn, n, g);
                let bt = load_at(b, Global, beta, k);
                let y = op2(b, F32, KirOp::AddRn, scaled, bt);
                store_at(b, Global, out_row, k, y);
                None
            });
        }
        NormOp::RmsNorm => {
            let q = column_loop(&mut b, tid, cols, Some(zero), |b, k, acc| {
                let x = load_at(b, Global, in_row, k);
                let sq = op2(b, F32, KirOp::MulRn, x, x);
                Some(op2(b, F32, KirOp::AddRn, acc.expect("acc"), sq))
            })
            .expect("acc");
            let inv = block_sum(&mut b, sm[0], tid32, q, |b, total| inv_std(b, total, cols, eps));
            column_loop(&mut b, tid, cols, None, |b, k, _| {
                let x = load_at(b, Global, in_row, k);
                let n = op2(b, F32, KirOp::MulRn, x, inv);
                let g = load_at(b, Global, gamma, k);
                let y = op2(b, F32, KirOp::MulRn, n, g);
                store_at(b, Global, out_row, k, y);
                None
            });
        }
    }
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([NORM_BLOCK, 1, 1]);
    b.set_launch_bounds(NORM_BLOCK, None);
    b.finalize()
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: NormOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: NormOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in NormOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.params() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
            let bytes = 1024 * op.regions().len();
            assert!(p.contains(&format!(".shared .align 4 .b8 shared_mem[{bytes}]")), "{p}");
        }
    }

    /// One `rsqrt`, one `div.approx` per statistic, two barriers per
    /// statistic, and no `fma` or contractible spelling.
    #[test]
    fn the_arithmetic_follows_the_hand_kernels() {
        for op in NormOp::ALL {
            let p = text(op);
            let stats = op.regions().len();
            assert_eq!(p.matches("rsqrt.approx.f32").count(), 1, "{p}");
            assert_eq!(p.matches("div.approx.f32").count(), stats, "{p}");
            assert_eq!(p.matches("cvt.rn.f32.u64").count(), stats, "{p}");
            assert_eq!(p.matches("bar.sync 0;").count(), 2 * stats, "{p}");
            assert!(!p.contains("fma") && !p.contains("add.f32") && !p.contains("mul.f32 "), "{p}");
            assert!(!p.contains("sub.f32"), "{p}");
        }
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        for op in NormOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
