// crates/nsl-kir/src/kernels/det_scatter.rs
//! The runtime's deterministic scatter-add from `cuda/fused_kernels.rs`, as
//! KIR (new-roadmap item 5): `nsl_det_scatter_add_f32(src, indices, input,
//! out, num_indices, embed_dim, vocab_size)`.
//!
//! It is output-centric and uses no atomics. On a `(vocab_size, embed_dim)`
//! grid of 16 × 16 blocks, thread `(row, col)` owns `out[row, col]`. It
//! starts from `input[row, col]`, walks every position `i` in order, adds
//! `src[i, col]` wherever `idx(i) == row`, and writes the sum once. The
//! order of the adds is fixed, so the result is the same on every run.
//!
//! The indices are f32, converted with `cvt.rzi.u64.f32` as in the hand
//! kernel. That conversion truncates toward zero and saturates, so a
//! negative index, and a NaN, become 0 and add into row 0. The kernel keeps
//! that behaviour; the gate pins it.
//!
//! The sum adds with `add.rn.f32`, which rounds as the hand kernel's
//! `add.f32` does (no multiply feeds it). The loop leaves its header for a
//! block the header dominates, and no conditional edge in it carries block
//! arguments, so no copies print around a predicated branch.

use super::elementwise::{f32_ptr, load_f32, store_f32, verified_ptx};
use super::lookup::{coordinate, LOOKUP_BLOCK};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// The block the runtime launches with: 16 × 16.
pub const DET_SCATTER_BLOCK: [u32; 2] = LOOKUP_BLOCK;

/// The `.visible .entry` name. Pinned: the runtime launches by it.
pub const KERNEL_NAME: &str = "nsl_det_scatter_add_f32";

/// The parameter names, in launch order. Pinned: the hand module's.
pub const PARAM_NAMES: [&str; 7] = ["src", "indices", "input", "out", "num_indices", "embed_dim", "vocab_size"];

fn var(b: &mut KirBuilder, ty: KirType) -> VarId {
    b.new_typed_var(ty)
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

/// `row · cols + j`.
fn flat(b: &mut KirBuilder, row: VarId, cols: VarId, j: VarId) -> VarId {
    let t = op2(b, KirType::U64, KirOp::Mul, row, cols);
    op2(b, KirType::U64, KirOp::Add, t, j)
}

/// ```text
/// entry:      row = x; col = y; if row >= vocab_size { exit }
/// cols:       if col >= embed_dim { exit }
/// start:      at = row·embed + col; walk(0, input[at])
/// walk(i, s): if i >= num_indices { done }
/// body:       if u64(indices[i]) != row { skip } else { hit }
/// hit:        next(s + src[i·embed + col])
/// skip:       next(s)
/// next(s'):   walk(i + 1, s')
/// done:       out[at] = s
/// ```
pub fn build() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let mut b = KirBuilder::new(KERNEL_NAME);
    let src = b.add_param(PARAM_NAMES[0], f32_ptr(), Global);
    let indices = b.add_param(PARAM_NAMES[1], f32_ptr(), Global);
    let input = b.add_param(PARAM_NAMES[2], f32_ptr(), Global);
    let out = b.add_param(PARAM_NAMES[3], f32_ptr(), Global);
    let n = b.add_param(PARAM_NAMES[4], U64, Global);
    let embed = b.add_param(PARAM_NAMES[5], U64, Global);
    let vocab = b.add_param(PARAM_NAMES[6], U64, Global);

    let entry = b.new_block();
    let cols = b.new_block();
    let start = b.new_block();
    let walk = b.new_block();
    let body = b.new_block();
    let hit = b.new_block();
    let skip = b.new_block();
    let next = b.new_block();
    let done = b.new_block();
    let exit = b.new_block();
    let i = b.add_block_param(walk, U64);
    let s = b.add_block_param(walk, F32);
    let s_joined = b.add_block_param(next, F32);

    b.set_block(entry);
    let row = coordinate(&mut b, 0);
    let col = coordinate(&mut b, 1);
    let past = cmp(&mut b, row, vocab, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(cols)));

    b.set_block(cols);
    let past = cmp(&mut b, col, embed, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(start)));

    b.set_block(start);
    let at = flat(&mut b, row, embed, col);
    let s0 = load_f32(&mut b, input, at);
    let i0 = var(&mut b, U64);
    b.emit(KirOp::Const(i0, KirConst { ty: U64, value: ConstValue::U64(0) }));
    b.terminate(KirTerminator::Branch(KirEdge::with(walk, vec![i0, s0])));

    b.set_block(walk);
    let end = cmp(&mut b, i, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let raw = load_f32(&mut b, indices, i);
    let r = var(&mut b, U64);
    b.emit(KirOp::Cast(r, raw, U64));
    let other = cmp(&mut b, r, row, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(other, KirEdge::to(skip), KirEdge::to(hit)));

    b.set_block(hit);
    let from = flat(&mut b, i, embed, col);
    let g = load_f32(&mut b, src, from);
    let s_hit = op2(&mut b, F32, KirOp::AddRn, s, g);
    b.terminate(KirTerminator::Branch(KirEdge::with(next, vec![s_hit])));

    b.set_block(skip);
    b.terminate(KirTerminator::Branch(KirEdge::with(next, vec![s])));

    b.set_block(next);
    let one = var(&mut b, U64);
    b.emit(KirOp::Const(one, KirConst { ty: U64, value: ConstValue::U64(1) }));
    let i_next = op2(&mut b, U64, KirOp::Add, i, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(walk, vec![i_next, s_joined])));

    b.set_block(done);
    store_f32(&mut b, out, at, s);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([DET_SCATTER_BLOCK[0], DET_SCATTER_BLOCK[1], 1]);
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

    /// The index converts as the hand kernel's does (unsigned, truncating,
    /// saturating); the sum rounds explicitly; the grid is two-dimensional.
    #[test]
    fn the_index_and_sum_follow_the_hand_kernel() {
        let p = text();
        assert_eq!(p.matches("cvt.rzi.u64.f32").count(), 1, "{p}");
        assert_eq!(p.matches("add.rn.f32").count(), 1, "{p}");
        assert!(!p.contains("fma") && !p.contains("add.f32"), "{p}");
        assert_eq!(p.matches("ld.global.f32").count(), 3, "{p}");
        assert_eq!(p.matches("st.global.f32").count(), 1, "{p}");
        assert!(p.contains("%ctaid.y") && p.contains("%tid.y") && p.contains("%ntid.y"), "{p}");
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        let p = text();
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
