// crates/nsl-kir/src/kernels/rmsnorm_dx.rs
//! The fused RMSNorm input-gradient backward from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5): `nsl_rmsnorm_dx_bwd_f32(dy, x, gamma, dxout,
//! rows, cols, eps)` and its residual-folding twin
//! `nsl_rmsnorm_dx_bwd_add_f32(dy, x, gamma, dxout, res, rows, cols, eps)`.
//!
//! One block of [`RMSNORM_DX_BLOCK`] threads per row (`%ctaid.x`); a block
//! at or past `rows` does nothing. Each thread strides over the row's
//! columns (`k = tid, tid + 256, …`) twice:
//!
//! 1. **Reduce.** Each thread accumulates `S1 = Σ x²` and `S2 = Σ (dy · γ) ·
//!    x` with `fma.rn`, as the hand kernel spells them, and stores them to
//!    `ssq[tid]` and `sdwx[tid]`. After a barrier, thread 0 adds the
//!    partials `1 .. %ntid.x` to its own, in order (both in one loop), and
//!    stores `rinv = min(rsqrt(S1 / cols + eps), 1e12)` to `ssq[0]` and `S2`
//!    to `sdwx[0]`, which every thread reads after a second barrier. The
//!    clamp is the CPU and tape-AD underflow guard: with `eps = 0`, a
//!    near-zero row cannot inject `inf`.
//! 2. **Write.** Every thread forms `coeff = ((rinv · rinv) · rinv) · S2 /
//!    cols` and writes `dx = (γ · dy) · rinv - x · coeff`, plus `res` for the
//!    residual-folding twin (`add.rn`, which rounds as the separate add it
//!    replaced did).
//!
//! Every other add, subtract and multiply rounds explicitly (`.rn`). ptxas
//! fused two of the hand kernel's pairs on hardware: `S1 / cols + eps` (the
//! `div.approx` is a multiply by the reciprocal) and `(γ · dy) · rinv - …`
//! (the multiply by `rinv` into the subtraction). These kernels round twice,
//! as the PTX (and the CPU reference) do. Thread 0 reads `%ntid.x` once,
//! before its fold, so ptxas unrolls the fold as it does the hand kernel's,
//! and the entry caps registers at [`MAX_REGISTERS`].

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// The block the runtime launches with: one per row. The column loops
/// stride by it.
pub const RMSNORM_DX_BLOCK: u32 = 256;

/// The clamp on `rinv`, `1e12`, as the hand kernel spells it
/// (`0f5368D4A5`).
pub const RINV_MAX: f32 = f32::from_bits(0x5368_D4A5);

/// The register cap: the hand kernel's count, and the most a 256-thread
/// block can use for full occupancy on sm_80 and sm_90.
pub const MAX_REGISTERS: u32 = 32;

/// Which kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmsNormDxOp {
    /// `dx = (γ · dy) · rinv - x · coeff`.
    Dx,
    /// `dx + res`, the residual gradient folded in.
    DxAdd,
}

impl RmsNormDxOp {
    pub const ALL: [RmsNormDxOp; 2] = [RmsNormDxOp::Dx, RmsNormDxOp::DxAdd];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            RmsNormDxOp::Dx => "nsl_rmsnorm_dx_bwd_f32",
            RmsNormDxOp::DxAdd => "nsl_rmsnorm_dx_bwd_add_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn params(self) -> &'static [&'static str] {
        match self {
            RmsNormDxOp::Dx => &["dy", "x", "gamma", "dxout", "rows", "cols", "eps"],
            RmsNormDxOp::DxAdd => &["dy", "x", "gamma", "dxout", "res", "rows", "cols", "eps"],
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

/// `total / cols` with `div.approx`, `cols` converted with `cvt.rn`.
fn per_column(b: &mut KirBuilder, total: VarId, cols: VarId) -> VarId {
    let n = var(b, KirType::F32);
    b.emit(KirOp::Cast(n, cols, KirType::F32));
    op2(b, KirType::F32, KirOp::DivApprox, total, n)
}

/// ```text
/// entry: row = %ctaid.x; if row >= rows { exit }
/// setup: dyr, xr, dxr (, resr) = the row's bases
/// acc:   k = tid, +256 < cols: s1 = fma(x, x, s1); s2 = fma(dy · γ, x, s2)
/// fold:  ssq[tid] = s1; sdwx[tid] = s2; barrier
///        thread 0: s1 += ssq[i]; s2 += sdwx[i] for i in 1 .. %ntid.x
///                  ssq[0] = min(rsqrt(s1 / cols + eps), 1e12); sdwx[0] = s2
///        barrier; rinv = ssq[0]; s2 = sdwx[0]
/// write: coeff = ((rinv · rinv) · rinv) · s2 / cols
///        k = tid, +256 < cols: dx[k] = (γ · dy) · rinv - x · coeff (+ res[k])
/// ```
pub fn build(op: RmsNormDxOp) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let names = op.params();
    let mut b = KirBuilder::new(op.kernel_name());
    let dy = b.add_param(names[0], f32_ptr(), Global);
    let x = b.add_param(names[1], f32_ptr(), Global);
    let gamma = b.add_param(names[2], f32_ptr(), Global);
    let dxout = b.add_param(names[3], f32_ptr(), Global);
    let res = (op == RmsNormDxOp::DxAdd).then(|| b.add_param(names[4], f32_ptr(), Global));
    let at = names.len() - 3;
    let rows = b.add_param(names[at], U64, Global);
    let cols = b.add_param(names[at + 1], U64, Global);
    let eps = b.add_param(names[at + 2], F32, Global);
    let region = |name: &str| SmemRegion { name: name.to_string(), bytes: RMSNORM_DX_BLOCK * 4, align: 4, elem: F32 };
    b.set_smem_layout(SmemLayout { regions: vec![region("ssq"), region("sdwx")], dynamic: false });

    let entry = b.new_block();
    let setup = b.new_block();
    let acc_head = b.new_block();
    let acc_step = b.new_block();
    let acc_done = b.new_block();
    let first = b.new_block();
    let fold_head = b.new_block();
    let fold_step = b.new_block();
    let fold_done = b.new_block();
    let after = b.new_block();
    let wr_head = b.new_block();
    let wr_step = b.new_block();
    let wr_done = b.new_block();
    let exit: BlockId = b.new_block();
    let k = b.add_block_param(acc_head, U64);
    let s1 = b.add_block_param(acc_head, F32);
    let s2 = b.add_block_param(acc_head, F32);
    let i = b.add_block_param(fold_head, U32);
    let t1 = b.add_block_param(fold_head, F32);
    let t2 = b.add_block_param(fold_head, F32);
    let j = b.add_block_param(wr_head, U64);

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
    let row_ptr = |b: &mut KirBuilder, p: VarId| {
        let r = var(b, f32_ptr());
        b.emit(KirOp::PtrOffset(r, p, base));
        r
    };
    let dy_row = row_ptr(&mut b, dy);
    let x_row = row_ptr(&mut b, x);
    let dx_row = row_ptr(&mut b, dxout);
    let res_row = res.map(|p| row_ptr(&mut b, p));
    let ssq = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: ssq, region: 0 });
    let sdwx = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion { dst: sdwx, region: 1 });
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    let zero2 = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(acc_head, vec![tid, zero, zero2])));

    // Pass 1: S1 = Σ x², S2 = Σ (dy · γ) · x.
    b.set_block(acc_head);
    let end = cmp(&mut b, k, cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(acc_done), KirEdge::to(acc_step)));

    b.set_block(acc_step);
    let xv = load_at(&mut b, Global, x_row, k);
    let dv = load_at(&mut b, Global, dy_row, k);
    let gv = load_at(&mut b, Global, gamma, k);
    let s1n = fma(&mut b, xv, xv, s1);
    let dg = op2(&mut b, F32, KirOp::MulRn, dv, gv);
    let s2n = fma(&mut b, dg, xv, s2);
    let stride = konst(&mut b, U64, ConstValue::U64(u64::from(RMSNORM_DX_BLOCK)));
    let kn = op2(&mut b, U64, KirOp::Add, k, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(acc_head, vec![kn, s1n, s2n])));

    // The fold.
    b.set_block(acc_done);
    store_at(&mut b, Shared, ssq, tid32, s1);
    store_at(&mut b, Shared, sdwx, tid32, s2);
    b.emit(KirOp::Barrier);
    let zero32 = konst(&mut b, U32, ConstValue::U32(0));
    let not_first = cmp(&mut b, tid32, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(after), KirEdge::to(first)));

    b.set_block(first);
    let ntid = var(&mut b, U32);
    b.emit(KirOp::BlockDim(ntid, 0));
    let one = konst(&mut b, U32, ConstValue::U32(1));
    b.terminate(KirTerminator::Branch(KirEdge::with(fold_head, vec![one, s1, s2])));

    b.set_block(fold_head);
    let fend = cmp(&mut b, i, ntid, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(fend, KirEdge::to(fold_done), KirEdge::to(fold_step)));

    b.set_block(fold_step);
    let o1 = load_at(&mut b, Shared, ssq, i);
    let t1n = op2(&mut b, F32, KirOp::AddRn, t1, o1);
    let o2 = load_at(&mut b, Shared, sdwx, i);
    let t2n = op2(&mut b, F32, KirOp::AddRn, t2, o2);
    let one = konst(&mut b, U32, ConstValue::U32(1));
    let i_next = op2(&mut b, U32, KirOp::Add, i, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(fold_head, vec![i_next, t1n, t2n])));

    b.set_block(fold_done);
    let m = per_column(&mut b, t1, cols);
    let v = op2(&mut b, F32, KirOp::AddRn, m, eps);
    let r = op1(&mut b, F32, KirOp::Rsqrt, v);
    let cap = konst(&mut b, F32, ConstValue::F32(RINV_MAX));
    let rc = op2(&mut b, F32, KirOp::Min, r, cap);
    b.emit(KirOp::Store(ssq, rc, Shared));
    b.emit(KirOp::Store(sdwx, t2, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(after)));

    b.set_block(after);
    b.emit(KirOp::Barrier);
    let rinv = var(&mut b, F32);
    b.emit(KirOp::Load(rinv, ssq, Shared));
    let s2_all = var(&mut b, F32);
    b.emit(KirOp::Load(s2_all, sdwx, Shared));
    let r2 = op2(&mut b, F32, KirOp::MulRn, rinv, rinv);
    let r3 = op2(&mut b, F32, KirOp::MulRn, r2, rinv);
    let c = op2(&mut b, F32, KirOp::MulRn, r3, s2_all);
    let coeff = per_column(&mut b, c, cols);
    b.terminate(KirTerminator::Branch(KirEdge::with(wr_head, vec![tid])));

    // Pass 2: dx = (γ · dy) · rinv - x · coeff (+ res).
    b.set_block(wr_head);
    let wend = cmp(&mut b, j, cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(wend, KirEdge::to(wr_done), KirEdge::to(wr_step)));

    b.set_block(wr_step);
    let xv = load_at(&mut b, Global, x_row, j);
    let dv = load_at(&mut b, Global, dy_row, j);
    let gv = load_at(&mut b, Global, gamma, j);
    let a = op2(&mut b, F32, KirOp::MulRn, gv, dv);
    let a = op2(&mut b, F32, KirOp::MulRn, a, rinv);
    let xc = op2(&mut b, F32, KirOp::MulRn, xv, coeff);
    let mut out = op2(&mut b, F32, KirOp::SubRn, a, xc);
    if let Some(res_row) = res_row {
        let rv = load_at(&mut b, Global, res_row, j);
        out = op2(&mut b, F32, KirOp::AddRn, out, rv);
    }
    store_at(&mut b, Global, dx_row, j, out);
    let stride = konst(&mut b, U64, ConstValue::U64(u64::from(RMSNORM_DX_BLOCK)));
    let jn = op2(&mut b, U64, KirOp::Add, j, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(wr_head, vec![jn])));

    b.set_block(wr_done);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([RMSNORM_DX_BLOCK, 1, 1]);
    b.set_launch_bounds(RMSNORM_DX_BLOCK, None);
    // Left to itself ptxas gives the plain kernel 35–36 registers where the
    // hand kernel had 32, which drops a 256-thread block's occupancy on
    // sm_80 and sm_90 from 64 warps to 48. At 32 it spills nothing and
    // unrolls as before.
    b.set_max_registers(MAX_REGISTERS);
    b.finalize()
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: RmsNormDxOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: RmsNormDxOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in RmsNormDxOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'), "{op:?}");
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.params() {
                assert!(p.contains(&format!("[param_{name}]")), "{op:?}: {name}\n{p}");
            }
            assert!(p.contains(".shared .align 4 .b8 shared_mem[2048]"), "{p}");
            assert!(p.contains(&format!(".maxnreg {MAX_REGISTERS}")), "{p}");
        }
    }

    /// The two explicit `fma`s, one `rsqrt` clamped by `min`, two
    /// `div.approx`, two barriers, and no contractible spelling.
    #[test]
    fn the_arithmetic_follows_the_hand_kernel() {
        for op in RmsNormDxOp::ALL {
            let p = text(op);
            assert_eq!(p.matches("fma.rn.f32").count(), 2, "{p}");
            assert_eq!(p.matches("rsqrt.approx.f32").count(), 1, "{p}");
            assert_eq!(p.matches("min.f32").count(), 1, "{p}");
            assert_eq!(p.matches("0f5368D4A5").count(), 1, "{p}");
            assert_eq!(p.matches("div.approx.f32").count(), 2, "{p}");
            assert_eq!(p.matches("bar.sync 0;").count(), 2, "{p}");
            assert_eq!(p.matches("sub.rn.f32").count(), 1, "{p}");
            let adds = if op == RmsNormDxOp::DxAdd { 4 } else { 3 };
            assert_eq!(p.matches("add.rn.f32").count(), adds, "{p}");
            assert!(!p.contains("add.f32") && !p.contains("mul.f32 ") && !p.contains("sub.f32"), "{p}");
        }
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        for op in RmsNormDxOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
