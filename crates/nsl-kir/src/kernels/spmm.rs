// crates/nsl-kir/src/kernels/spmm.rs
//! The runtime's sparse matrix-matrix kernels from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5). Each computes `C = A · B` for a sparse `A`
//! and a dense row-major f32 `B` with `N` columns:
//!
//! - `nsl_csr_spmm_f32(row_ptrs, col_indices, values, B, C, M, N)`: a block
//!   of [`SPMM_BLOCK`] threads per (row, 256 output columns): row
//!   `%ctaid.x`, column `%ctaid.y · %ntid.x + %tid.x`. The host stages the
//!   indices as `u32`. The thread walks `row_ptrs[r] .. row_ptrs[r + 1]` in
//!   order, accumulating `sum = fma(values[k], B[col_indices[k], j], sum)`
//!   from `+0.0`, and stores `C[r, j] = sum`.
//! - `nsl_coo_spmm_f32(row_indices, col_indices, values, B, C, N, nnz)`: a
//!   thread per nonzero, 64-bit indices. Nonzero `k` adds `values[k] ·
//!   B[col_indices[k], j]` (`mul.rn`) into `C[row_indices[k], j]` atomically
//!   for every `j` in order; the host zeroed `C`. The atomic is
//!   `red.global.add.f32` where the hand kernel's was `atom` with an unread
//!   result; the two add alike.
//! - `nsl_bsr_spmm_f32(row_ptrs, col_indices, values, B, C, N, block_rows,
//!   block_cols, nblk_rows)`: a two-dimensional block per (block row, output
//!   columns): block row `%ctaid.x`, row within it `%tid.y`, column
//!   `%ctaid.y · %ntid.x + %tid.x`. The thread walks the block row's blocks
//!   in order and, within each, its row's `block_cols` values in order,
//!   accumulating with `fma` as the CSR kernel does. The runtime sizes the
//!   block from the shapes (`min(256, N) × block_rows`), so this kernel,
//!   like the hand one, declares no launch bounds.
//!
//! Each keeps the hand kernel's order of loads and its rounding. Every loop
//! leaves its header for a block the header dominates, so no exit edge
//! carries copies.

use super::elementwise::{f32_ptr, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType,
    VarId,
};

/// The block the runtime launches the CSR and COO kernels with.
pub const SPMM_BLOCK: u32 = 256;

/// Which sparse format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmmFormat {
    /// Compressed sparse rows, `u32` row pointers and column indices.
    Csr,
    /// Coordinate list, `i64` row and column indices.
    Coo,
    /// Block sparse rows, `u32` block-row pointers and block-column indices.
    Bsr,
}

impl SpmmFormat {
    pub const ALL: [SpmmFormat; 3] = [SpmmFormat::Csr, SpmmFormat::Coo, SpmmFormat::Bsr];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            SpmmFormat::Csr => "nsl_csr_spmm_f32",
            SpmmFormat::Coo => "nsl_coo_spmm_f32",
            SpmmFormat::Bsr => "nsl_bsr_spmm_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            SpmmFormat::Csr => &["row_ptrs", "col_indices", "values", "B", "C", "M", "N"],
            SpmmFormat::Coo => &["row_indices", "col_indices", "values", "B", "C", "N", "nnz"],
            SpmmFormat::Bsr => &["row_ptrs", "col_indices", "values", "B", "C", "N", "block_rows", "block_cols", "nblk_rows"],
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

fn special(b: &mut KirBuilder, op: fn(VarId, u8) -> KirOp, dim: u8) -> VarId {
    let dst = var(b, KirType::U32);
    b.emit(op(dst, dim));
    dst
}

fn widen(b: &mut KirBuilder, x: VarId) -> VarId {
    let dst = var(b, KirType::U64);
    b.emit(KirOp::Cast(dst, x, KirType::U64));
    dst
}

fn index_ptr(elem: KirType) -> KirType {
    KirType::Ptr(Box::new(elem), AddressSpace::Global)
}

/// `base[i]` widened to `u64`, for an index array of `elem` (`U32` or `U64`).
fn load_index(b: &mut KirBuilder, elem: KirType, base: VarId, i: VarId) -> VarId {
    let addr = var(b, index_ptr(elem.clone()));
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = var(b, elem.clone());
    b.emit(KirOp::Load(v, addr, AddressSpace::Global));
    if elem == KirType::U64 {
        return v;
    }
    widen(b, v)
}

/// `%ctaid.y · %ntid.x + %tid.x`, in 32 bits as the hand kernels form it,
/// widened.
fn output_column(b: &mut KirBuilder) -> VarId {
    use KirType::U32;
    let cy = special(b, KirOp::BlockIdx, 1);
    let nx = special(b, KirOp::BlockDim, 0);
    let base = op2(b, U32, KirOp::Mul, cy, nx);
    let tx = special(b, KirOp::ThreadId, 0);
    let col = op2(b, U32, KirOp::Add, base, tx);
    widen(b, col)
}

/// `a · n + j`, the row-major index of `[a, j]` in a matrix of `n` columns.
fn at(b: &mut KirBuilder, a: VarId, n: VarId, j: VarId) -> VarId {
    let row = op2(b, KirType::U64, KirOp::Mul, a, n);
    op2(b, KirType::U64, KirOp::Add, row, j)
}

/// From the current block: exit if `x >= bound`, else continue in a new
/// block (current on return).
fn guard(b: &mut KirBuilder, x: VarId, bound: VarId, exit: BlockId) {
    let next = b.new_block();
    let past = cmp(b, x, bound, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(next)));
    b.set_block(next);
}

/// `row_ptrs[r]` and `row_ptrs[r + 1]`, widened.
fn row_span(b: &mut KirBuilder, row_ptrs: VarId, r: VarId) -> (VarId, VarId) {
    let lo = load_index(b, KirType::U32, row_ptrs, r);
    let one = konst(b, KirType::U64, ConstValue::U64(1));
    let r1 = op2(b, KirType::U64, KirOp::Add, r, one);
    let hi = load_index(b, KirType::U32, row_ptrs, r1);
    (lo, hi)
}

fn finish(mut b: KirBuilder, exit: BlockId, bounded: bool) -> KernelIR {
    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    if bounded {
        b.set_workgroup_size([SPMM_BLOCK, 1, 1]);
        b.set_launch_bounds(SPMM_BLOCK, None);
    }
    b.finalize()
}

/// ```text
/// entry: r = %ctaid.x; j = %ctaid.y · %ntid.x + %tid.x
///        if r >= M or j >= N { exit }
///        lo, hi = row_ptrs[r], row_ptrs[r + 1]
/// head(k, sum): if k >= hi { write }
/// step:  sum = fma(values[k], B[col_indices[k] · N + j], sum); k += 1
/// write: C[r · N + j] = sum
/// ```
fn build_csr() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let names = SpmmFormat::Csr.param_names();
    let mut b = KirBuilder::new(SpmmFormat::Csr.kernel_name());
    let row_ptrs = b.add_param(names[0], index_ptr(U32), Global);
    let col_indices = b.add_param(names[1], index_ptr(U32), Global);
    let values = b.add_param(names[2], f32_ptr(), Global);
    let bm = b.add_param(names[3], f32_ptr(), Global);
    let cm = b.add_param(names[4], f32_ptr(), Global);
    let m = b.add_param(names[5], U64, Global);
    let n = b.add_param(names[6], U64, Global);

    let entry = b.new_block();
    let exit = b.new_block();
    let head = b.new_block();
    let step = b.new_block();
    let write = b.new_block();
    let k = b.add_block_param(head, U64);
    let sum = b.add_block_param(head, F32);

    b.set_block(entry);
    let r32 = special(&mut b, KirOp::BlockIdx, 0);
    let r = widen(&mut b, r32);
    let j = output_column(&mut b);
    guard(&mut b, r, m, exit);
    guard(&mut b, j, n, exit);
    let (lo, hi) = row_span(&mut b, row_ptrs, r);
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![lo, zero])));

    b.set_block(head);
    let end = cmp(&mut b, k, hi, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(write), KirEdge::to(step)));

    b.set_block(step);
    let c = load_index(&mut b, U32, col_indices, k);
    let v = load_f32(&mut b, values, k);
    let bi = at(&mut b, c, n, j);
    let bv = load_f32(&mut b, bm, bi);
    let next = var(&mut b, F32);
    b.emit(KirOp::Fma(next, v, bv, sum));
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let k_next = op2(&mut b, U64, KirOp::Add, k, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, next])));

    b.set_block(write);
    let ci = at(&mut b, r, n, j);
    store_f32(&mut b, cm, ci, sum);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
    finish(b, exit, true)
}

/// ```text
/// entry: t = global id; if t >= nnz { exit }
///        r = row_indices[t]; c = col_indices[t]; v = values[t]
/// head(j): if j >= N { exit }
/// step:  atomic C[r · N + j] += v · B[c · N + j]; j += 1
/// ```
fn build_coo() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let names = SpmmFormat::Coo.param_names();
    let mut b = KirBuilder::new(SpmmFormat::Coo.kernel_name());
    let row_indices = b.add_param(names[0], index_ptr(U64), Global);
    let col_indices = b.add_param(names[1], index_ptr(U64), Global);
    let values = b.add_param(names[2], f32_ptr(), Global);
    let bm = b.add_param(names[3], f32_ptr(), Global);
    let cm = b.add_param(names[4], f32_ptr(), Global);
    let n = b.add_param(names[5], U64, Global);
    let nnz = b.add_param(names[6], U64, Global);

    let entry = b.new_block();
    let exit = b.new_block();
    let head = b.new_block();
    let step = b.new_block();
    let j = b.add_block_param(head, U64);

    b.set_block(entry);
    let t32 = var(&mut b, U32);
    b.emit(KirOp::GlobalId(t32, 0));
    let t = widen(&mut b, t32);
    guard(&mut b, t, nnz, exit);
    let r = load_index(&mut b, U64, row_indices, t);
    let c = load_index(&mut b, U64, col_indices, t);
    let v = load_f32(&mut b, values, t);
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero])));

    b.set_block(head);
    let end = cmp(&mut b, j, n, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(exit), KirEdge::to(step)));

    b.set_block(step);
    let bi = at(&mut b, c, n, j);
    let bv = load_f32(&mut b, bm, bi);
    let p = op2(&mut b, F32, KirOp::MulRn, v, bv);
    let ci = at(&mut b, r, n, j);
    let addr = var(&mut b, f32_ptr());
    b.emit(KirOp::PtrOffset(addr, cm, ci));
    b.emit(KirOp::AtomicAdd(addr, p, Global));
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let j_next = op2(&mut b, U64, KirOp::Add, j, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![j_next])));
    finish(b, exit, true)
}

/// ```text
/// entry: br = %ctaid.x; sr = %tid.y; j = %ctaid.y · %ntid.x + %tid.x
///        if br >= nblk_rows or sr >= block_rows or j >= N { exit }
///        lo, hi = row_ptrs[br], row_ptrs[br + 1]
/// outer(blk, sum): if blk >= hi { write }
/// body:  bc = col_indices[blk]
///        off = blk · (block_rows · block_cols) + sr · block_cols
/// inner(sc, sum): if sc >= block_cols { next }
/// step:  sum = fma(values[off + sc], B[(bc · block_cols + sc) · N + j], sum)
///        sc += 1
/// next:  outer(blk + 1, sum)
/// write: C[(br · block_rows + sr) · N + j] = sum
/// ```
fn build_bsr() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let names = SpmmFormat::Bsr.param_names();
    let mut b = KirBuilder::new(SpmmFormat::Bsr.kernel_name());
    let row_ptrs = b.add_param(names[0], index_ptr(U32), Global);
    let col_indices = b.add_param(names[1], index_ptr(U32), Global);
    let values = b.add_param(names[2], f32_ptr(), Global);
    let bm = b.add_param(names[3], f32_ptr(), Global);
    let cm = b.add_param(names[4], f32_ptr(), Global);
    let n = b.add_param(names[5], U64, Global);
    let block_rows = b.add_param(names[6], U64, Global);
    let block_cols = b.add_param(names[7], U64, Global);
    let nblk_rows = b.add_param(names[8], U64, Global);

    let entry = b.new_block();
    let exit = b.new_block();
    let outer = b.new_block();
    let body = b.new_block();
    let inner = b.new_block();
    let step = b.new_block();
    let next = b.new_block();
    let write = b.new_block();
    let blk = b.add_block_param(outer, U64);
    let sum = b.add_block_param(outer, F32);
    let sc = b.add_block_param(inner, U64);
    let isum = b.add_block_param(inner, F32);

    b.set_block(entry);
    let br32 = special(&mut b, KirOp::BlockIdx, 0);
    let br = widen(&mut b, br32);
    let sr32 = special(&mut b, KirOp::ThreadId, 1);
    let sr = widen(&mut b, sr32);
    let j = output_column(&mut b);
    guard(&mut b, br, nblk_rows, exit);
    guard(&mut b, sr, block_rows, exit);
    guard(&mut b, j, n, exit);
    let (lo, hi) = row_span(&mut b, row_ptrs, br);
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(outer, vec![lo, zero])));

    b.set_block(outer);
    let end = cmp(&mut b, blk, hi, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(write), KirEdge::to(body)));

    b.set_block(body);
    let bc = load_index(&mut b, U32, col_indices, blk);
    let block_size = op2(&mut b, U64, KirOp::Mul, block_rows, block_cols);
    let first = op2(&mut b, U64, KirOp::Mul, blk, block_size);
    let within = op2(&mut b, U64, KirOp::Mul, sr, block_cols);
    let off = op2(&mut b, U64, KirOp::Add, first, within);
    let zero = konst(&mut b, U64, ConstValue::U64(0));
    b.terminate(KirTerminator::Branch(KirEdge::with(inner, vec![zero, sum])));

    b.set_block(inner);
    let end = cmp(&mut b, sc, block_cols, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(next), KirEdge::to(step)));

    b.set_block(step);
    let vi = op2(&mut b, U64, KirOp::Add, off, sc);
    let v = load_f32(&mut b, values, vi);
    let k = at(&mut b, bc, block_cols, sc);
    let bi = at(&mut b, k, n, j);
    let bv = load_f32(&mut b, bm, bi);
    let acc = var(&mut b, F32);
    b.emit(KirOp::Fma(acc, v, bv, isum));
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let sc_next = op2(&mut b, U64, KirOp::Add, sc, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(inner, vec![sc_next, acc])));

    b.set_block(next);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let blk_next = op2(&mut b, U64, KirOp::Add, blk, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(outer, vec![blk_next, isum])));

    b.set_block(write);
    let row = at(&mut b, br, block_rows, sr);
    let ci = at(&mut b, row, n, j);
    store_f32(&mut b, cm, ci, sum);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
    finish(b, exit, false)
}

/// Build `format`'s kernel as KIR.
pub fn build(format: SpmmFormat) -> KernelIR {
    match format {
        SpmmFormat::Csr => build_csr(),
        SpmmFormat::Coo => build_coo(),
        SpmmFormat::Bsr => build_bsr(),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(format: SpmmFormat) -> Vec<u8> {
    verified_ptx(build(format))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(format: SpmmFormat) -> String {
        String::from_utf8(ptx(format)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for format in SpmmFormat::ALL {
            let p = text(format);
            assert!(p.ends_with('\0'), "{format:?}: NUL-terminated");
            assert!(p.contains(&format!(".visible .entry {}(", format.kernel_name())), "{p}");
            for name in format.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{format:?}: {name}\n{p}");
            }
            // Only the fixed-block kernels bound their launch.
            assert_eq!(p.contains(".maxntid 256, 1, 1"), format != SpmmFormat::Bsr, "{p}");
        }
    }

    #[test]
    fn the_index_widths_and_the_arithmetic_follow_the_format() {
        let csr = text(SpmmFormat::Csr);
        assert_eq!(csr.matches("ld.global.u32").count(), 3, "{csr}");
        assert_eq!(csr.matches("fma.rn.f32").count(), 1, "{csr}");
        assert!(csr.contains("%ctaid.y") && !csr.contains("red.") && !csr.contains("mul.rn.f32"), "{csr}");
        let coo = text(SpmmFormat::Coo);
        assert_eq!(coo.matches("ld.global.u64").count(), 2, "{coo}");
        assert_eq!(coo.matches("mul.rn.f32").count(), 1, "{coo}");
        assert_eq!(coo.matches("red.global.add.f32").count(), 1, "{coo}");
        assert!(!coo.contains("fma") && !coo.contains("%ctaid.y"), "{coo}");
        let bsr = text(SpmmFormat::Bsr);
        assert_eq!(bsr.matches("ld.global.u32").count(), 3, "{bsr}");
        assert_eq!(bsr.matches("fma.rn.f32").count(), 1, "{bsr}");
        assert!(bsr.contains("%tid.y") && bsr.contains("%ctaid.y") && !bsr.contains("red."), "{bsr}");
    }

    #[test]
    fn no_loop_exit_carries_copies() {
        for format in SpmmFormat::ALL {
            let p = text(format);
            assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{format:?}\n{p}");
        }
    }
}
