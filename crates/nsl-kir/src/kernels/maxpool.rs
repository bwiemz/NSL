// crates/nsl-kir/src/kernels/maxpool.rs
//! The runtime's 2-D max pooling forward from `cuda/fused_kernels.rs`, as
//! KIR (new-roadmap item 5): `nsl_maxpool2d_f32(inp, out, argmax, N, C, H,
//! W, kH, kW, stride, padding, H_out, W_out, total)`.
//!
//! A thread per output element (blocks of
//! [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK)); a thread
//! at or past `total` does nothing. The flat index `i` splits as `ow = i %
//! W_out`, `oh = (i / W_out) % H_out`, `c = (i / W_out / H_out) % C` and
//! `n = i / W_out / H_out / C`. The thread walks the window `ky = 0 .. kH`,
//! `kx = 0 .. kW` in order. A tap at `ih = oh · stride + ky - padding`,
//! `iw = ow · stride + kx - padding` is skipped when it falls in the padding
//! (`oh · stride + ky < padding`, likewise for the width) or past the input
//! (`ih >= H`, `iw >= W`). Otherwise `x = inp[((n · C + c) · H + ih) · W +
//! iw]` replaces the running max, and its flat index the running argmax,
//! unless `x <= max`. The running pair starts at `(-inf, 0)`. So a NaN tap
//! replaces the max (no comparison with it is true) and the next tap
//! replaces the NaN, as in the hand kernel. The thread stores the max to
//! `out[i]` and the argmax, as `u64`, to `argmax[i]`.
//!
//! The update is two `selp`s on `x <= max` where the hand kernel branched
//! around two moves; it picks the same pair. Every loop leaves its header
//! for a block the header dominates, and every skip reaches the step through
//! one unconditional edge, so no conditional edge carries copies.

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// The `.visible .entry` name. Pinned: the runtime launches by it.
pub const KERNEL_NAME: &str = "nsl_maxpool2d_f32";

/// The parameter names, in launch order. Pinned: the hand module's.
pub const PARAM_NAMES: [&str; 14] =
    ["inp", "out", "argmax", "N", "C", "H", "W", "kH", "kW", "stride", "padding", "H_out", "W_out", "total"];

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

fn u64_op(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    op2(b, KirType::U64, op, x, y)
}

fn cmp(b: &mut KirBuilder, x: VarId, y: VarId, op: CmpOp) -> VarId {
    let dst = var(b, KirType::Bool);
    b.emit(KirOp::Cmp(dst, x, y, op));
    dst
}

/// ```text
/// entry:  i = global id; if i >= total { exit }
///         ow, oh, c, n = i split by W_out, H_out, C
/// ky(ky, m, a): if ky >= kH { write }
/// kx(kx, m, a): if kx >= kW { ky(ky + 1, m, a) }
///         ih0 = oh · stride + ky; iw0 = ow · stride + kx
///         skip if ih0 < padding, iw0 < padding,
///                 ih0 - padding >= H, iw0 - padding >= W
///         j = ((n · C + c) · H + ih) · W + iw; x = inp[j]
///         keep = x <= m; step(keep ? m : x, keep ? a : j)
/// skip:   step(m, a)
/// step(m, a): kx(kx + 1, m, a)
/// write:  out[i] = m; argmax[i] = a
/// ```
pub fn build() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let mut b = KirBuilder::new(KERNEL_NAME);
    let inp = b.add_param(PARAM_NAMES[0], f32_ptr(), Global);
    let out = b.add_param(PARAM_NAMES[1], f32_ptr(), Global);
    let argmax = b.add_param(PARAM_NAMES[2], KirType::Ptr(Box::new(U64), Global), Global);
    let dims: Vec<VarId> = PARAM_NAMES[3..].iter().map(|name| b.add_param(name, U64, Global)).collect();
    let [_n, c_dim, h, w, kh, kw, stride, padding, h_out, w_out, total] = dims[..] else {
        unreachable!("eleven dimensions")
    };

    let (i, _, exit) = index_and_bound(&mut b, total);
    let ky_head = b.new_block();
    let kx_start = b.new_block();
    let kx_head = b.new_block();
    let ky_step = b.new_block();
    let tap = b.new_block();
    let in_w = b.new_block();
    let in_h = b.new_block();
    let below_w = b.new_block();
    let load = b.new_block();
    let skip = b.new_block();
    let step = b.new_block();
    let write = b.new_block();
    let ky = b.add_block_param(ky_head, U64);
    let m = b.add_block_param(ky_head, F32);
    let a = b.add_block_param(ky_head, U64);
    let kx = b.add_block_param(kx_head, U64);
    let m2 = b.add_block_param(kx_head, F32);
    let a2 = b.add_block_param(kx_head, U64);
    let m3 = b.add_block_param(step, F32);
    let a3 = b.add_block_param(step, U64);

    let ow = u64_op(&mut b, KirOp::Rem, i, w_out);
    let t = u64_op(&mut b, KirOp::Div, i, w_out);
    let oh = u64_op(&mut b, KirOp::Rem, t, h_out);
    let t2 = u64_op(&mut b, KirOp::Div, t, h_out);
    let c = u64_op(&mut b, KirOp::Rem, t2, c_dim);
    let n = u64_op(&mut b, KirOp::Div, t2, c_dim);
    let neg_inf = konst(&mut b, F32, ConstValue::F32(f32::NEG_INFINITY));
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(ky_head, vec![zero, neg_inf, zero])));

    b.set_block(ky_head);
    let ky_end = cmp(&mut b, ky, kh, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(ky_end, KirEdge::to(write), KirEdge::to(kx_start)));

    b.set_block(kx_start);
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(kx_head, vec![zero, m, a])));

    b.set_block(kx_head);
    let kx_end = cmp(&mut b, kx, kw, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(kx_end, KirEdge::to(ky_step), KirEdge::to(tap)));

    b.set_block(ky_step);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let ky_next = u64_op(&mut b, KirOp::Add, ky, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(ky_head, vec![ky_next, m2, a2])));

    b.set_block(tap);
    let ohs = u64_op(&mut b, KirOp::Mul, oh, stride);
    let ih0 = u64_op(&mut b, KirOp::Add, ohs, ky);
    let ows = u64_op(&mut b, KirOp::Mul, ow, stride);
    let iw0 = u64_op(&mut b, KirOp::Add, ows, kx);
    let pad_h = cmp(&mut b, ih0, padding, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(pad_h, KirEdge::to(skip), KirEdge::to(in_w)));

    b.set_block(in_w);
    let pad_w = cmp(&mut b, iw0, padding, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(pad_w, KirEdge::to(skip), KirEdge::to(in_h)));

    b.set_block(in_h);
    let ih = u64_op(&mut b, KirOp::Sub, ih0, padding);
    let iw = u64_op(&mut b, KirOp::Sub, iw0, padding);
    let past_h = cmp(&mut b, ih, h, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past_h, KirEdge::to(skip), KirEdge::to(below_w)));

    b.set_block(below_w);
    let past_w = cmp(&mut b, iw, w, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past_w, KirEdge::to(skip), KirEdge::to(load)));

    b.set_block(load);
    let nc = u64_op(&mut b, KirOp::Mul, n, c_dim);
    let plane = u64_op(&mut b, KirOp::Add, nc, c);
    let rows = u64_op(&mut b, KirOp::Mul, plane, h);
    let row = u64_op(&mut b, KirOp::Add, rows, ih);
    let cols = u64_op(&mut b, KirOp::Mul, row, w);
    let j = u64_op(&mut b, KirOp::Add, cols, iw);
    let x = load_f32(&mut b, inp, j);
    let keep = cmp(&mut b, x, m2, CmpOp::Le);
    let m_new = var(&mut b, F32);
    b.emit(KirOp::Select(m_new, keep, m2, x));
    let a_new = var(&mut b, U64);
    b.emit(KirOp::Select(a_new, keep, a2, j));
    b.terminate(KirTerminator::Branch(KirEdge::with(step, vec![m_new, a_new])));

    b.set_block(skip);
    b.terminate(KirTerminator::Branch(KirEdge::with(step, vec![m2, a2])));

    b.set_block(step);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let kx_next = u64_op(&mut b, KirOp::Add, kx, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(kx_head, vec![kx_next, m3, a3])));

    b.set_block(write);
    store_f32(&mut b, out, i, m);
    let at = var(&mut b, KirType::Ptr(Box::new(U64), Global));
    b.emit(KirOp::PtrOffset(at, argmax, i));
    b.emit(KirOp::Store(at, a, Global));
    finish(b, exit)
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

    /// Three `rem` and three `div` split the index; the window test is
    /// `setp.le.f32`; the update is two `selp`; the argmax is stored as
    /// `u64`.
    #[test]
    fn the_arithmetic_follows_the_hand_kernel() {
        let p = text();
        assert_eq!(p.matches("rem.u64").count(), 3, "{p}");
        assert_eq!(p.matches("div.u64").count(), 3, "{p}");
        assert_eq!(p.matches("setp.le.f32").count(), 1, "{p}");
        assert_eq!(p.matches("selp.").count(), 2, "{p}");
        assert_eq!(p.matches("st.global.u64").count(), 1, "{p}");
        assert_eq!(p.matches("st.global.f32").count(), 1, "{p}");
        assert_eq!(p.matches("0fFF800000").count(), 1, "{p}");
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        let p = text();
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
