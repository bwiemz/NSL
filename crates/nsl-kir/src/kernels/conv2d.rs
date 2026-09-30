// crates/nsl-kir/src/kernels/conv2d.rs
//! The runtime's direct 2-D convolution forward from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5): `nsl_conv2d_f32(inp, wt, bias, out, N, C_in,
//! H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w, H_out, W_out,
//! total)`, NCHW input `[N, C_in, H, W]`, weight `[C_out, C_in, kH, kW]`,
//! output `[N, C_out, H_out, W_out]`.
//!
//! A thread per output element (blocks of
//! [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK)); a thread
//! at or past `total` does nothing. The flat index `i` splits as `ow = i %
//! W_out`, `oh = (i / W_out) % H_out`, `co = (i / W_out / H_out) % C_out`
//! and `n = i / W_out / H_out / C_out`. The thread walks `ci = 0 .. C_in`,
//! `ky = 0 .. kH`, `kx = 0 .. kW` in order. A tap at `ih = oh · stride_h +
//! ky - pad_h`, `iw = ow · stride_w + kx - pad_w` is skipped when it falls
//! in the padding (`oh · stride_h + ky < pad_h`, likewise for the width) or
//! past the input (`ih >= H`, `iw >= W`). Otherwise the accumulator, from
//! `0`, becomes `fma(inp[((n · C_in + ci) · H + ih) · W + iw], wt[((co ·
//! C_in + ci) · kH + ky) · kW + kx], acc)`, the hand kernel's explicit
//! `fma.rn`. When `bias` is not null, `bias[co]` is added (`add.f32`: no
//! product to contract with). The thread stores the sum to `out[i]`.
//!
//! Every loop leaves its header for a block the header dominates, and every
//! skip reaches the step through one unconditional edge, so no conditional
//! edge carries copies.

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// The `.visible .entry` name. Pinned: the runtime launches by it.
pub const KERNEL_NAME: &str = "nsl_conv2d_f32";

/// The parameter names, in launch order. Pinned: the hand module's.
pub const PARAM_NAMES: [&str; 18] = [
    "inp", "wt", "bias", "out", "N", "C_in", "H", "W", "C_out", "kH", "kW", "stride_h", "stride_w", "pad_h", "pad_w",
    "H_out", "W_out", "total",
];

fn var(b: &mut KirBuilder, ty: KirType) -> VarId {
    b.new_typed_var(ty)
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn u64_op(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = var(b, KirType::U64);
    b.emit(op(dst, x, y));
    dst
}

fn cmp(b: &mut KirBuilder, x: VarId, y: VarId, op: CmpOp) -> VarId {
    let dst = var(b, KirType::Bool);
    b.emit(KirOp::Cmp(dst, x, y, op));
    dst
}

/// `((a · b + c) · d + e) · f + g`: a row-major flat index.
#[allow(clippy::too_many_arguments)]
fn flat(b: &mut KirBuilder, a: VarId, bd: VarId, c: VarId, d: VarId, e: VarId, f: VarId, g: VarId) -> VarId {
    let ab = u64_op(b, KirOp::Mul, a, bd);
    let abc = u64_op(b, KirOp::Add, ab, c);
    let abcd = u64_op(b, KirOp::Mul, abc, d);
    let abcde = u64_op(b, KirOp::Add, abcd, e);
    let abcdef = u64_op(b, KirOp::Mul, abcde, f);
    u64_op(b, KirOp::Add, abcdef, g)
}

/// ```text
/// entry:  i = global id; if i >= total { exit }
///         ow, oh, co, n = i split by W_out, H_out, C_out
/// ci(ci, acc):   if ci >= C_in { bias }
/// ky(ky, acc):   if ky >= kH { ci(ci + 1, acc) }
/// kx(kx, acc):   if kx >= kW { ky(ky + 1, acc) }
///         ih0 = oh · stride_h + ky; iw0 = ow · stride_w + kx
///         skip if ih0 < pad_h, iw0 < pad_w,
///                 ih0 - pad_h >= H, iw0 - pad_w >= W
///         step(fma(inp[n, ci, ih, iw], wt[co, ci, ky, kx], acc))
/// skip:   step(acc)
/// step(acc): kx(kx + 1, acc)
/// bias:   if bias == null { write(acc) } else { write(acc + bias[co]) }
/// write(r): out[i] = r
/// ```
pub fn build() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let mut b = KirBuilder::new(KERNEL_NAME);
    let inp = b.add_param(PARAM_NAMES[0], f32_ptr(), Global);
    let wt = b.add_param(PARAM_NAMES[1], f32_ptr(), Global);
    let bias = b.add_param(PARAM_NAMES[2], f32_ptr(), Global);
    let out = b.add_param(PARAM_NAMES[3], f32_ptr(), Global);
    let dims: Vec<VarId> = PARAM_NAMES[4..].iter().map(|name| b.add_param(name, U64, Global)).collect();
    let [_n, c_in, h, w, c_out, kh, kw, stride_h, stride_w, pad_h, pad_w, h_out, w_out, total] = dims[..] else {
        unreachable!("fourteen dimensions")
    };

    let (i, _, exit) = index_and_bound(&mut b, total);
    let ci_head = b.new_block();
    let ky_start = b.new_block();
    let ky_head = b.new_block();
    let ci_step = b.new_block();
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
    let bias_test = b.new_block();
    let add_bias = b.new_block();
    let no_bias = b.new_block();
    let write = b.new_block();
    let ci = b.add_block_param(ci_head, U64);
    let acc = b.add_block_param(ci_head, F32);
    let ky = b.add_block_param(ky_head, U64);
    let acc1 = b.add_block_param(ky_head, F32);
    let kx = b.add_block_param(kx_head, U64);
    let acc2 = b.add_block_param(kx_head, F32);
    let acc3 = b.add_block_param(step, F32);
    let r = b.add_block_param(write, F32);

    let ow = u64_op(&mut b, KirOp::Rem, i, w_out);
    let t = u64_op(&mut b, KirOp::Div, i, w_out);
    let oh = u64_op(&mut b, KirOp::Rem, t, h_out);
    let t2 = u64_op(&mut b, KirOp::Div, t, h_out);
    let co = u64_op(&mut b, KirOp::Rem, t2, c_out);
    let n = u64_op(&mut b, KirOp::Div, t2, c_out);
    let zero_f = konst(&mut b, F32, ConstValue::F32(0.0));
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(ci_head, vec![zero, zero_f])));

    b.set_block(ci_head);
    let ci_end = cmp(&mut b, ci, c_in, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(ci_end, KirEdge::to(bias_test), KirEdge::to(ky_start)));

    b.set_block(ky_start);
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(ky_head, vec![zero, acc])));

    b.set_block(ky_head);
    let ky_end = cmp(&mut b, ky, kh, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(ky_end, KirEdge::to(ci_step), KirEdge::to(kx_start)));

    b.set_block(ci_step);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let ci_next = u64_op(&mut b, KirOp::Add, ci, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(ci_head, vec![ci_next, acc1])));

    b.set_block(kx_start);
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(kx_head, vec![zero, acc1])));

    b.set_block(kx_head);
    let kx_end = cmp(&mut b, kx, kw, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(kx_end, KirEdge::to(ky_step), KirEdge::to(tap)));

    b.set_block(ky_step);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let ky_next = u64_op(&mut b, KirOp::Add, ky, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(ky_head, vec![ky_next, acc2])));

    b.set_block(tap);
    let ohs = u64_op(&mut b, KirOp::Mul, oh, stride_h);
    let ih0 = u64_op(&mut b, KirOp::Add, ohs, ky);
    let ows = u64_op(&mut b, KirOp::Mul, ow, stride_w);
    let iw0 = u64_op(&mut b, KirOp::Add, ows, kx);
    let in_pad_h = cmp(&mut b, ih0, pad_h, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(in_pad_h, KirEdge::to(skip), KirEdge::to(in_w)));

    b.set_block(in_w);
    let in_pad_w = cmp(&mut b, iw0, pad_w, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(in_pad_w, KirEdge::to(skip), KirEdge::to(in_h)));

    b.set_block(in_h);
    let ih = u64_op(&mut b, KirOp::Sub, ih0, pad_h);
    let iw = u64_op(&mut b, KirOp::Sub, iw0, pad_w);
    let past_h = cmp(&mut b, ih, h, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past_h, KirEdge::to(skip), KirEdge::to(below_w)));

    b.set_block(below_w);
    let past_w = cmp(&mut b, iw, w, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past_w, KirEdge::to(skip), KirEdge::to(load)));

    b.set_block(load);
    let j = flat(&mut b, n, c_in, ci, h, ih, w, iw);
    let x = load_f32(&mut b, inp, j);
    let k = flat(&mut b, co, c_in, ci, kh, ky, kw, kx);
    let wv = load_f32(&mut b, wt, k);
    let acc_new = var(&mut b, F32);
    b.emit(KirOp::Fma(acc_new, x, wv, acc2));
    b.terminate(KirTerminator::Branch(KirEdge::with(step, vec![acc_new])));

    b.set_block(skip);
    b.terminate(KirTerminator::Branch(KirEdge::with(step, vec![acc2])));

    b.set_block(step);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let kx_next = u64_op(&mut b, KirOp::Add, kx, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(kx_head, vec![kx_next, acc3])));

    b.set_block(bias_test);
    let null = konst(&mut b, f32_ptr(), ConstValue::U64(0));
    let no = cmp(&mut b, bias, null, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(no, KirEdge::to(no_bias), KirEdge::to(add_bias)));

    b.set_block(add_bias);
    let bv = load_f32(&mut b, bias, co);
    let biased = var(&mut b, F32);
    b.emit(KirOp::Add(biased, acc, bv));
    b.terminate(KirTerminator::Branch(KirEdge::with(write, vec![biased])));

    b.set_block(no_bias);
    b.terminate(KirTerminator::Branch(KirEdge::with(write, vec![acc])));

    b.set_block(write);
    store_f32(&mut b, out, i, r);
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

    /// Three `rem` and three `div` split the index; one `fma.rn` per tap,
    /// one `add.f32` for the bias, and nothing else in f32 arithmetic; the
    /// null test is a 64-bit `setp.eq`.
    #[test]
    fn the_arithmetic_follows_the_hand_kernel() {
        let p = text();
        assert_eq!(p.matches("rem.u64").count(), 3, "{p}");
        assert_eq!(p.matches("div.u64").count(), 3, "{p}");
        assert_eq!(p.matches("fma.rn.f32").count(), 1, "{p}");
        assert_eq!(p.matches("add.f32").count(), 1, "{p}");
        assert_eq!(p.matches("mul.f32").count(), 0, "{p}");
        assert_eq!(p.matches("setp.eq.u64").count(), 1, "{p}");
        assert_eq!(p.matches("ld.global.f32").count(), 3, "{p}");
        assert_eq!(p.matches("st.global.f32").count(), 1, "{p}");
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        let p = text();
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
