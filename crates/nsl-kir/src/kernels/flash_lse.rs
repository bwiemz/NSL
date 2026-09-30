// crates/nsl-kir/src/kernels/flash_lse.rs
//! The runtime's device-resident flash-attention log-sum-exp kernels from
//! `cuda/fused_kernels.rs`, as KIR (new-roadmap item 5). The flash backward
//! recomputes each attention row's `lse` from Q and K with them:
//!
//! - `nsl_flash_lse_f32(q, k, lse, total, seq, hd, scale, causal)`: Q and K
//!   are `[batch, heads, seq, hd]` (the MHA layout).
//! - `nsl_flash_lse_gqa_f32(…, heads, kv_heads)`: K is `[batch, kv_heads,
//!   seq, hd]`, and Q head `h` of batch `b` reads kv-head `(bh / heads) ·
//!   kv_heads + (bh % heads) / (heads / kv_heads)`, the consecutive-block
//!   grouping of the CPU backward.
//!
//! A thread per `(batch, head, query row)`, `i < total = b · h · s`, on
//! blocks of [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK).
//! Row `qi = i % seq` of head `bh = i / seq` sees keys `j < qi + 1` when
//! `causal != 0` and `j < seq` otherwise. The thread walks them twice, as
//! the hand kernels do: first for the row max of `score_j = (Σ_d q_d ·
//! k_jd) · scale` (from `-inf`, replaced when `score > max`, so a NaN score
//! never replaces it), then for `Σ_j 2^((score_j - max) · log2 e)`
//! (`ex2.approx`). It stores `lse[i] = max + lg2(sum) · ln 2`
//! (`lg2.approx`).
//!
//! Every add, subtract and multiply rounds explicitly (`.rn`): the hand
//! kernels' comment asks for mul-then-add "to mirror the CPU reference"
//! (`compute_logsumexp_gqa`, which rounds each step), but ptxas contracted
//! their dot-product terms, `score · scale - max` and `lg2 · ln 2 + max`
//! into `FFMA` on hardware. These kernels round twice, as the PTX text and
//! the CPU reference do.
//!
//! Every loop leaves its header for a block the header dominates, so no
//! conditional edge carries copies.

use super::elementwise::{f32_ptr, finish, index_and_bound, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// `log2(e)` as the hand kernels spell it (`0f3FB8AA3B`).
pub const LOG2_E_BITS: u32 = 0x3FB8_AA3B;
/// `ln(2)` as the hand kernels spell it (`0f3F317218`).
pub const LN_2_BITS: u32 = 0x3F31_7218;

/// Which K layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LseOp {
    /// K has Q's head count.
    Mha,
    /// K has `kv_heads` heads, each shared by `heads / kv_heads` Q heads.
    Gqa,
}

impl LseOp {
    pub const ALL: [LseOp; 2] = [LseOp::Mha, LseOp::Gqa];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            LseOp::Mha => "nsl_flash_lse_f32",
            LseOp::Gqa => "nsl_flash_lse_gqa_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            LseOp::Mha => &["q", "k", "lse", "total", "seq", "hd", "scale", "causal"],
            LseOp::Gqa => &["q", "k", "lse", "total", "seq", "hd", "scale", "causal", "heads", "kv_heads"],
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

fn cmp(b: &mut KirBuilder, x: VarId, y: VarId, op: CmpOp) -> VarId {
    let dst = var(b, KirType::Bool);
    b.emit(KirOp::Cmp(dst, x, y, op));
    dst
}

fn f32_const(b: &mut KirBuilder, bits: u32) -> VarId {
    konst(b, KirType::F32, ConstValue::F32(f32::from_bits(bits)))
}

/// `score = (Σ_d q[d] · kr[d]) · scale`, the sum from `+0` in `d` order,
/// walking both rows by pointer. Starts in the current block and leaves
/// the builder in the block after the loop.
///
/// ```text
/// d(d, acc, pq, pk): if d >= hd { done }
///         d(d + 1, acc + q · k, pq + 1, pk + 1)
/// done:   score = acc · scale
/// ```
fn score(b: &mut KirBuilder, qp: VarId, kr: VarId, hd: VarId, scale: VarId) -> VarId {
    use KirType::{F32, U64};
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let d = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);
    let pq = b.add_block_param(head, f32_ptr());
    let pk = b.add_block_param(head, f32_ptr());
    let zero_f = konst(b, F32, ConstValue::F32(0.0));
    let zero = konst(b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, zero_f, qp, kr])));

    b.set_block(head);
    let end = cmp(b, d, hd, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let x = var(b, F32);
    b.emit(KirOp::Load(x, pq, AddressSpace::Global));
    let y = var(b, F32);
    b.emit(KirOp::Load(y, pk, AddressSpace::Global));
    let prod = op2(b, F32, KirOp::MulRn, x, y);
    let acc_next = op2(b, F32, KirOp::AddRn, acc, prod);
    let one = konst(b, U64, ConstValue::U64(1));
    let pq_next = var(b, f32_ptr());
    b.emit(KirOp::PtrOffset(pq_next, pq, one));
    let pk_next = var(b, f32_ptr());
    b.emit(KirOp::PtrOffset(pk_next, pk, one));
    let d_next = op2(b, U64, KirOp::Add, d, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![d_next, acc_next, pq_next, pk_next])));

    b.set_block(done);
    op2(b, F32, KirOp::MulRn, acc, scale)
}

/// One pass over the keys: `j = 0 .. n_keys` with a carried `f32`, each
/// step folding that key's score in with `fold`. Starts in the current
/// block; returns the carried value and leaves the builder in the block
/// after the loop.
#[allow(clippy::too_many_arguments)]
fn key_pass(
    b: &mut KirBuilder,
    start: VarId,
    n_keys: VarId,
    qp: VarId,
    kb: VarId,
    hd: VarId,
    scale: VarId,
    fold: impl FnOnce(&mut KirBuilder, VarId, VarId) -> VarId,
) -> VarId {
    use KirType::{F32, U64};
    let head: BlockId = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let j = b.add_block_param(head, U64);
    let carried = b.add_block_param(head, F32);
    let zero = konst(b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, start])));

    b.set_block(head);
    let end = cmp(b, j, n_keys, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let jh = op2(b, U64, KirOp::Mul, j, hd);
    let kr = var(b, f32_ptr());
    b.emit(KirOp::PtrOffset(kr, kb, jh));
    let s = score(b, qp, kr, hd, scale);
    let next = fold(b, carried, s);
    let one = konst(b, U64, ConstValue::U64(1));
    let j_next = op2(b, U64, KirOp::Add, j, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![j_next, next])));

    b.set_block(done);
    carried
}

/// ```text
/// entry:  i = global id; if i >= total { exit }
///         qi = i % seq; bh = i / seq
///         qp = q + i · hd; kb = k + kbh · seq · hd
///         n_keys = causal != 0 ? qi + 1 : seq
///         max = pass(-inf, (m, s) => s > m ? s : m)
///         sum = pass(0, (acc, s) => acc + ex2((s - max) · log2 e))
///         lse[i] = max + lg2(sum) · ln 2
/// ```
/// where `kbh = bh` for [`LseOp::Mha`] and `(bh / heads) · kv_heads + (bh %
/// heads) / (heads / kv_heads)` for [`LseOp::Gqa`].
pub fn build(op: LseOp) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name());
    let q = b.add_param(names[0], f32_ptr(), Global);
    let k = b.add_param(names[1], f32_ptr(), Global);
    let lse = b.add_param(names[2], f32_ptr(), Global);
    let total = b.add_param(names[3], U64, Global);
    let seq = b.add_param(names[4], U64, Global);
    let hd = b.add_param(names[5], U64, Global);
    let scale = b.add_param(names[6], F32, Global);
    let causal = b.add_param(names[7], U64, Global);
    let gqa = (op == LseOp::Gqa).then(|| (b.add_param(names[8], U64, Global), b.add_param(names[9], U64, Global)));

    let (i, _, exit) = index_and_bound(&mut b, total);
    let qi = op2(&mut b, U64, KirOp::Rem, i, seq);
    let bh = op2(&mut b, U64, KirOp::Div, i, seq);
    let q_off = op2(&mut b, U64, KirOp::Mul, i, hd);
    let kbh = match gqa {
        None => bh,
        Some((heads, kv_heads)) => {
            let groups = op2(&mut b, U64, KirOp::Div, heads, kv_heads);
            let batch = op2(&mut b, U64, KirOp::Div, bh, heads);
            let h = op2(&mut b, U64, KirOp::Rem, bh, heads);
            let kv_h = op2(&mut b, U64, KirOp::Div, h, groups);
            let base = op2(&mut b, U64, KirOp::Mul, batch, kv_heads);
            op2(&mut b, U64, KirOp::Add, base, kv_h)
        }
    };
    let k_rows = op2(&mut b, U64, KirOp::Mul, kbh, seq);
    let k_off = op2(&mut b, U64, KirOp::Mul, k_rows, hd);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let upto = op2(&mut b, U64, KirOp::Add, qi, one);
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    let is_causal = cmp(&mut b, causal, zero, CmpOp::Ne);
    let n_keys = var(&mut b, U64);
    b.emit(KirOp::Select(n_keys, is_causal, upto, seq));
    let qp = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(qp, q, q_off));
    let kb = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(kb, k, k_off));

    let neg_inf = konst(&mut b, F32, ConstValue::F32(f32::NEG_INFINITY));
    let max = key_pass(&mut b, neg_inf, n_keys, qp, kb, hd, scale, |b, m, s| {
        let greater = cmp(b, s, m, CmpOp::Gt);
        let m_next = var(b, F32);
        b.emit(KirOp::Select(m_next, greater, s, m));
        m_next
    });
    let zero_f = konst(&mut b, F32, ConstValue::F32(0.0));
    let sum = key_pass(&mut b, zero_f, n_keys, qp, kb, hd, scale, |b, acc, s| {
        let shifted = op2(b, F32, KirOp::SubRn, s, max);
        let log2_e = f32_const(b, LOG2_E_BITS);
        let exponent = op2(b, F32, KirOp::MulRn, shifted, log2_e);
        let e = var(b, F32);
        b.emit(KirOp::Exp2(e, exponent));
        op2(b, F32, KirOp::AddRn, acc, e)
    });
    let l2 = var(&mut b, F32);
    b.emit(KirOp::Log2(l2, sum));
    let ln_2 = f32_const(&mut b, LN_2_BITS);
    let ln = op2(&mut b, F32, KirOp::MulRn, l2, ln_2);
    let result = op2(&mut b, F32, KirOp::AddRn, max, ln);
    store_f32(&mut b, lse, i, result);
    finish(b, exit)
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: LseOp) -> Vec<u8> {
    verified_ptx(build(op))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: LseOp) -> String {
        String::from_utf8(ptx(op)).expect("ASCII")
    }

    #[test]
    fn the_kernels_verify_and_keep_their_entries_and_parameters() {
        for op in LseOp::ALL {
            let p = text(op);
            assert!(p.ends_with('\0'));
            assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name())), "{p}");
            for name in op.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{name}\n{p}");
            }
            assert!(p.contains("ld.param.f32"), "scale is an f32 parameter\n{p}");
        }
    }

    /// Two dot-product loops (one per pass), each a `mul.rn` then an
    /// `add.rn`; no unrounded float arithmetic; one `ex2.approx`, one
    /// `lg2.approx`; the two constants spelled as the hand kernels spell
    /// them.
    #[test]
    fn the_arithmetic_follows_the_hand_kernels() {
        for op in LseOp::ALL {
            let p = text(op);
            assert_eq!(p.matches("mul.rn.f32").count(), 6, "{p}");
            assert_eq!(p.matches("add.rn.f32").count(), 4, "{p}");
            assert_eq!(p.matches("sub.rn.f32").count(), 1, "{p}");
            for bare in ["mul.f32", "add.f32", "sub.f32", "fma."] {
                assert!(!p.contains(bare), "{bare}\n{p}");
            }
            assert_eq!(p.matches("ex2.approx.f32").count(), 1, "{p}");
            assert_eq!(p.matches("lg2.approx.f32").count(), 1, "{p}");
            assert_eq!(p.matches("setp.gt.f32").count(), 1, "{p}");
            assert_eq!(p.matches("0f3FB8AA3B").count(), 1, "{p}");
            assert_eq!(p.matches("0f3F317218").count(), 1, "{p}");
            assert_eq!(p.matches("ld.global.f32").count(), 4, "{p}");
            assert_eq!(p.matches("st.global.f32").count(), 1, "{p}");
            let (rem, div) = match op {
                LseOp::Mha => (1, 1),
                LseOp::Gqa => (2, 4),
            };
            assert_eq!(p.matches("rem.u64").count(), rem, "{p}");
            assert_eq!(p.matches("div.u64").count(), div, "{p}");
        }
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        for op in LseOp::ALL {
            let p = text(op);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
        }
    }
}
