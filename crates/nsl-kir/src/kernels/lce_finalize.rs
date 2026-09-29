// crates/nsl-kir/src/kernels/lce_finalize.rs
//! The GEMM-chunked fused linear cross-entropy's finalize kernel from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5):
//! `nsl_lce_finalize_f32(mstate, sstate, tlstate, targets, loss, lse, rows)`.
//!
//! The chunk kernels leave an online-softmax state per row: the running max
//! `m`, the scaled sum `s` and the target's logit `tl`. The finalize turns it
//! into the row's log-sum-exp and loss:
//!
//! ```text
//! lse[r]  = m[r] + ln(s[r])
//! loss[r] = targets[r] >= 0 ? lse[r] - tl[r] : 0
//! ```
//!
//! The kernel is grid-strided over rows. Thread `k` of the launch
//! (`%ctaid.x · %ntid.x + %tid.x`) takes `r = k, k + s, …` for the stride
//! `s = %ntid.x · %nctaid.x`. `ln(s)` is `lg2.approx(s)` times `ln 2`, folded
//! into the add as one `fma.rn.f32` (`lse = lg2(s) · ln 2 + m`), as the hand
//! kernel does. [`KirOp::Log2`](crate::kernel_ir::KirOp::Log2) is the bare
//! `lg2.approx.f32`: `KirOp::Log` multiplies by `ln 2` separately, which
//! would round twice. The target is read as `s64`, and `tl` is loaded only
//! for a valid row. The loss subtracts with `sub.rn.f32`, which rounds as the
//! hand kernel's `sub.f32` does.
//!
//! The loop leaves its header for a block the header dominates, so the exit
//! edge carries no copies.

use super::elementwise::{f32_ptr, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// The block the runtime launches with. The stride comes from the launch,
/// so this only bounds the block (`.maxntid`).
pub const LCE_FINALIZE_BLOCK: u32 = 256;

/// The `.visible .entry` name. Pinned: the runtime launches by it.
pub const KERNEL_NAME: &str = "nsl_lce_finalize_f32";

/// The parameter names, in launch order. Pinned: the hand module's.
pub const PARAM_NAMES: [&str; 7] = ["mstate", "sstate", "tlstate", "targets", "loss", "lse", "rows"];

/// `ln 2` as the hand kernel spells it (`0f3F317218`).
pub const LN_2: f32 = f32::from_bits(0x3F31_7218);

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

fn load_at(b: &mut KirBuilder, elem: KirType, base: VarId, i: VarId) -> VarId {
    let addr = var(b, KirType::Ptr(Box::new(elem.clone()), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = var(b, elem);
    b.emit(KirOp::Load(v, addr, AddressSpace::Global));
    v
}

fn store_at(b: &mut KirBuilder, base: VarId, i: VarId, v: VarId) {
    let addr = var(b, f32_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, v, AddressSpace::Global));
}

/// ```text
/// entry: k = global id; s = %ntid.x · %nctaid.x; head(k)
/// head(k): if k >= rows { exit }
/// body:  lse = fma(lg2(sstate[k]), ln 2, mstate[k]); lse_out[k] = lse
///        if targets[k] < 0 { join(0) } else { valid }
/// valid: join(lse - tlstate[k])
/// join(loss): loss_out[k] = loss; head(k + s)
/// ```
pub fn build() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, I64, U32, U64};
    let mut b = KirBuilder::new(KERNEL_NAME);
    let mstate = b.add_param(PARAM_NAMES[0], f32_ptr(), Global);
    let sstate = b.add_param(PARAM_NAMES[1], f32_ptr(), Global);
    let tlstate = b.add_param(PARAM_NAMES[2], f32_ptr(), Global);
    let targets = b.add_param(PARAM_NAMES[3], KirType::Ptr(Box::new(I64), Global), Global);
    let loss = b.add_param(PARAM_NAMES[4], f32_ptr(), Global);
    let lse_out = b.add_param(PARAM_NAMES[5], f32_ptr(), Global);
    let rows = b.add_param(PARAM_NAMES[6], U64, Global);

    let entry = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let invalid = b.new_block();
    let valid = b.new_block();
    let join = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let row_loss = b.add_block_param(join, F32);

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
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![start])));

    b.set_block(head);
    let end = cmp(&mut b, k, rows, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(exit), KirEdge::to(body)));

    b.set_block(body);
    let m = load_at(&mut b, F32, mstate, k);
    let s = load_at(&mut b, F32, sstate, k);
    let l2 = var(&mut b, F32);
    b.emit(KirOp::Log2(l2, s));
    let ln2 = konst(&mut b, F32, ConstValue::F32(LN_2));
    let lse = var(&mut b, F32);
    b.emit(KirOp::Fma(lse, l2, ln2, m));
    store_at(&mut b, lse_out, k, lse);
    let t = load_at(&mut b, I64, targets, k);
    let zero = konst(&mut b, I64, ConstValue::I64(0));
    let ignored = cmp(&mut b, t, zero, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(ignored, KirEdge::to(invalid), KirEdge::to(valid)));

    b.set_block(invalid);
    let none = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![none])));

    b.set_block(valid);
    let tl = load_at(&mut b, F32, tlstate, k);
    let diff = op2(&mut b, F32, KirOp::SubRn, lse, tl);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![diff])));

    b.set_block(join);
    store_at(&mut b, loss, k, row_loss);
    let k_next = op2(&mut b, U64, KirOp::Add, k, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next])));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([LCE_FINALIZE_BLOCK, 1, 1]);
    b.set_launch_bounds(LCE_FINALIZE_BLOCK, None);
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
    }

    /// `ln(s)` is one bare `lg2.approx` with `ln 2` folded into the add by
    /// one `fma`; the target is signed; the stride comes from the grid.
    #[test]
    fn the_log_folds_into_one_fma() {
        let p = text();
        assert_eq!(p.matches("lg2.approx.f32").count(), 1, "{p}");
        assert_eq!(p.matches("fma.rn.f32").count(), 1, "{p}");
        assert_eq!(p.matches("0f3F317218").count(), 1, "{p}");
        assert!(!p.contains("mul.f32") && !p.contains("mul.rn.f32"), "{p}");
        assert_eq!(p.matches("sub.rn.f32").count(), 1, "{p}");
        assert_eq!(p.matches("ld.global.s64").count(), 1, "{p}");
        assert_eq!(p.matches("setp.lt.s64").count(), 1, "{p}");
        assert_eq!(p.matches("ld.global.f32").count(), 3, "{p}");
        assert_eq!(p.matches("st.global.f32").count(), 2, "{p}");
        assert!(p.contains("%nctaid.x") && p.contains("%ntid.x"), "{p}");
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        let p = text();
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
