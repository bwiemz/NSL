// crates/nsl-kir/src/kernels/bmm.rs
//! The runtime's strided batched matmul from `cuda/fused_kernels.rs`, as KIR
//! (new-roadmap item 5): `nsl_bmm_f32(a, b, c, M, N, K, batch_count,
//! stride_A, stride_B, stride_C)` computes `C[z] = A[z] · B[z]` for every
//! batch slice `z = %ctaid.z` in one launch, where slice `z` of each operand
//! starts `z · stride` elements in (a stride of 0 broadcasts one matrix to
//! every slice).
//!
//! Blocks of [`BMM_BLOCK`] × [`BMM_BLOCK`] threads over a `(ceil(N / 16),
//! ceil(M / 16), batch_count)` grid; thread `(x, y)` of block `(bx, by)`
//! computes `row = by · %ntid.y + y`, `col = bx · %ntid.x + x`. A thread
//! with `row >= M`, `col >= N` or `z >= batch_count` does nothing (tested in
//! that order, as the hand kernel did). The others sum `Σ_k A[row, k] ·
//! B[k, col]` from `+0` in `k` order with the hand kernel's explicit
//! `fma.rn`, and store it to `C[row, col]`, all indexed row-major within the
//! slice.
//!
//! The `k` loop walks `A`'s row and `B`'s column by pointer (one element
//! and `N` elements a step), where the hand kernel formed `row · K + k` and
//! `k · N + col` on every trip: ptxas reduced those to the same walk, and
//! carrying it keeps the kernel at the hand kernel's register count. The
//! loop leaves its header for a block the header dominates, so its exit
//! edge carries no copies.

use super::elementwise::{f32_ptr, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// The block edge: the runtime launches `BMM_BLOCK × BMM_BLOCK` threads.
pub const BMM_BLOCK: u32 = 16;

/// The `.visible .entry` name. Pinned: the runtime launches by it.
pub const KERNEL_NAME: &str = "nsl_bmm_f32";

/// The parameter names, in launch order. Pinned: the hand module's.
pub const PARAM_NAMES: [&str; 10] = ["a", "b", "c", "M", "N", "K", "batch_count", "stride_A", "stride_B", "stride_C"];

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

/// `%ctaid.d · %ntid.d + %tid.d`, in 32 bits.
fn global_index(b: &mut KirBuilder, dim: u8) -> VarId {
    use KirType::U32;
    let block = var(b, U32);
    b.emit(KirOp::BlockIdx(block, dim));
    let width = var(b, U32);
    b.emit(KirOp::BlockDim(width, dim));
    let base = op2(b, U32, KirOp::Mul, block, width);
    let tid = var(b, U32);
    b.emit(KirOp::ThreadId(tid, dim));
    op2(b, U32, KirOp::Add, base, tid)
}

/// ```text
/// entry: row = by·ntid.y + y; col = bx·ntid.x + x; z = %ctaid.z
///        if row >= M or col >= N or z >= batch_count { exit }
///        A = a + z·stride_A; B = b + z·stride_B; C = c + z·stride_C
/// k(k, acc, pa, pb): if k >= K { write }     (pa = &A[row, 0], pb = &B[0, col])
///        k(k + 1, fma(*pa, *pb, acc), pa + 1, pb + N)
/// write: C[row·N + col] = acc
/// ```
pub fn build() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(KERNEL_NAME);
    let a = b.add_param(PARAM_NAMES[0], f32_ptr(), Global);
    let bm = b.add_param(PARAM_NAMES[1], f32_ptr(), Global);
    let c = b.add_param(PARAM_NAMES[2], f32_ptr(), Global);
    let dims: Vec<VarId> = PARAM_NAMES[3..].iter().map(|name| b.add_param(name, U64, Global)).collect();
    let [m, n, k_dim, batch_count, stride_a, stride_b, stride_c] = dims[..] else {
        unreachable!("seven dimensions")
    };

    let entry = b.new_block();
    let col_ok = b.new_block();
    let batch_ok = b.new_block();
    let setup = b.new_block();
    let head = b.new_block();
    let body = b.new_block();
    let write = b.new_block();
    let exit = b.new_block();
    let k = b.add_block_param(head, U64);
    let acc = b.add_block_param(head, F32);
    let pa = b.add_block_param(head, f32_ptr());
    let pb = b.add_block_param(head, f32_ptr());

    b.set_block(entry);
    let row32 = global_index(&mut b, 1);
    let col32 = global_index(&mut b, 0);
    let z32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(z32, 2));
    let row = cast(&mut b, row32, U64);
    let col = cast(&mut b, col32, U64);
    let z = cast(&mut b, z32, U64);
    let past_row = cmp(&mut b, row, m, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past_row, KirEdge::to(exit), KirEdge::to(col_ok)));

    b.set_block(col_ok);
    let past_col = cmp(&mut b, col, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past_col, KirEdge::to(exit), KirEdge::to(batch_ok)));

    b.set_block(batch_ok);
    let past_batch = cmp(&mut b, z, batch_count, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past_batch, KirEdge::to(exit), KirEdge::to(setup)));

    b.set_block(setup);
    let slice = |b: &mut KirBuilder, base: VarId, stride: VarId| {
        let off = op2(b, U64, KirOp::Mul, z, stride);
        let p = var(b, f32_ptr());
        b.emit(KirOp::PtrOffset(p, base, off));
        p
    };
    let a_slice = slice(&mut b, a, stride_a);
    let b_slice = slice(&mut b, bm, stride_b);
    let c_slice = slice(&mut b, c, stride_c);
    // A's row and B's column, walked by pointer: `A[row, k]` steps by one
    // element, `B[k, col]` by `N`.
    let row_k = op2(&mut b, U64, KirOp::Mul, row, k_dim);
    let a_row = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(a_row, a_slice, row_k));
    let b_col = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(b_col, b_slice, col));
    let zero_f = konst(&mut b, F32, ConstValue::F32(0.0));
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, zero_f, a_row, b_col])));

    b.set_block(head);
    let end = cmp(&mut b, k, k_dim, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(write), KirEdge::to(body)));

    b.set_block(body);
    let av = var(&mut b, F32);
    b.emit(KirOp::Load(av, pa, Global));
    let bv = var(&mut b, F32);
    b.emit(KirOp::Load(bv, pb, Global));
    let acc_next = var(&mut b, F32);
    b.emit(KirOp::Fma(acc_next, av, bv, acc));
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let pa_next = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(pa_next, pa, one));
    let pb_next = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(pb_next, pb, n));
    let k_next = op2(&mut b, U64, KirOp::Add, k, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, acc_next, pa_next, pb_next])));

    b.set_block(write);
    let row_n = op2(&mut b, U64, KirOp::Mul, row, n);
    let ci = op2(&mut b, U64, KirOp::Add, row_n, col);
    store_f32(&mut b, c_slice, ci, acc);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([BMM_BLOCK, BMM_BLOCK, 1]);
    b.set_launch_bounds(BMM_BLOCK * BMM_BLOCK, None);
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

    /// The three grid axes and two block axes are read; one `fma.rn` per
    /// step and no other float arithmetic; two loads and one store.
    #[test]
    fn the_arithmetic_follows_the_hand_kernel() {
        let p = text();
        for reg in ["%ctaid.x", "%ctaid.y", "%ctaid.z", "%ntid.x", "%ntid.y", "%tid.x", "%tid.y"] {
            assert!(p.contains(reg), "{reg}\n{p}");
        }
        assert_eq!(p.matches("fma.rn.f32").count(), 1, "{p}");
        for bare in ["add.f32", "mul.f32", "sub.f32"] {
            assert!(!p.contains(bare), "{bare}\n{p}");
        }
        assert_eq!(p.matches("ld.global.f32").count(), 2, "{p}");
        assert_eq!(p.matches("st.global.f32").count(), 1, "{p}");
    }

    #[test]
    fn no_conditional_edge_carries_copies() {
        let p = text();
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
