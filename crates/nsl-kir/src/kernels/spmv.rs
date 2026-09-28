// crates/nsl-kir/src/kernels/spmv.rs
//! The runtime's sparse matrix-vector kernels from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5). Both compute `y = A · x` for an `M × K`
//! sparse `A` and a dense f32 `x`, on blocks of
//! [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK):
//!
//! - `nsl_csr_spmv_f32(row_ptrs, col_indices, values, x, y, M)`: a thread
//!   per row. The host stages `row_ptrs` and `col_indices` as `u32`. Row
//!   `r` walks `row_ptrs[r] .. row_ptrs[r + 1]` in order, accumulating
//!   `sum = fma(values[k], x[col_indices[k]], sum)` from `+0.0`, and stores
//!   `y[r] = sum`.
//! - `nsl_coo_spmv_f32(row_indices, col_indices, values, x, y, nnz)`: a
//!   thread per nonzero. The indices are 64-bit (the host copies them as it
//!   holds them). Nonzero `k` adds `values[k] · x[col_indices[k]]`
//!   (`mul.rn`) into `y[row_indices[k]]` atomically; the host zeroed `y`.
//!
//! Each keeps the hand kernel's order of loads and its rounding: one fused
//! multiply-add per CSR term, an unfused product then an atomic add per COO
//! term. The COO atomic is `red.global.add.f32` where the hand kernel's was
//! `atom` with an unread result; the two add alike. The CSR loop leaves its
//! header for a block the header dominates, so the exit edge carries no
//! copies.

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// Which sparse format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpmvFormat {
    /// Compressed sparse rows, `u32` row pointers and column indices.
    Csr,
    /// Coordinate list, `i64` row and column indices.
    Coo,
}

impl SpmvFormat {
    pub const ALL: [SpmvFormat; 2] = [SpmvFormat::Csr, SpmvFormat::Coo];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            SpmvFormat::Csr => "nsl_csr_spmv_f32",
            SpmvFormat::Coo => "nsl_coo_spmv_f32",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            SpmvFormat::Csr => &["row_ptrs", "col_indices", "values", "x", "y", "M"],
            SpmvFormat::Coo => &["row_indices", "col_indices", "values", "x", "y", "nnz"],
        }
    }
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = b.new_typed_var(ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn op2(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(ty);
    b.emit(op(dst, x, y));
    dst
}

/// `base[i]` widened to `u64`, for an index array of `elem` (`U32` or `U64`).
fn load_index(b: &mut KirBuilder, elem: KirType, base: VarId, i: VarId) -> VarId {
    let addr = b.new_typed_var(KirType::Ptr(Box::new(elem.clone()), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = b.new_typed_var(elem.clone());
    b.emit(KirOp::Load(v, addr, AddressSpace::Global));
    if elem == KirType::U64 {
        return v;
    }
    let w = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(w, v, KirType::U64));
    w
}

fn index_ptr(elem: KirType) -> KirType {
    KirType::Ptr(Box::new(elem), AddressSpace::Global)
}

/// ```text
/// entry: r = global id; if r >= M { exit }
/// body:  lo = row_ptrs[r]; hi = row_ptrs[r + 1]
/// head(k, sum): if k >= hi { write }
/// step:  sum = fma(values[k], x[col_indices[k]], sum); k += 1
/// write: y[r] = sum
/// ```
fn build_csr() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};
    let names = SpmvFormat::Csr.param_names();
    let mut b = KirBuilder::new(SpmvFormat::Csr.kernel_name());
    let row_ptrs = b.add_param(names[0], index_ptr(U32), Global);
    let col_indices = b.add_param(names[1], index_ptr(U32), Global);
    let values = b.add_param(names[2], f32_ptr(), Global);
    let x = b.add_param(names[3], f32_ptr(), Global);
    let y = b.add_param(names[4], f32_ptr(), Global);
    let m = b.add_param(names[5], U64, Global);

    let (r, _, exit) = index_and_bound(&mut b, m);
    let head = b.new_block();
    let step = b.new_block();
    let write = b.new_block();
    let lo = load_index(&mut b, U32, row_ptrs, r);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let r1 = op2(&mut b, U64, KirOp::Add, r, one);
    let hi = load_index(&mut b, U32, row_ptrs, r1);
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![lo, zero])));
    let k = b.add_block_param(head, U64);
    let sum = b.add_block_param(head, F32);

    b.set_block(head);
    let end = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(end, k, hi, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(end, KirEdge::to(write), KirEdge::to(step)));

    b.set_block(step);
    let col = load_index(&mut b, U32, col_indices, k);
    let v = load_f32(&mut b, values, k);
    let xc = load_f32(&mut b, x, col);
    let next = b.new_typed_var(F32);
    b.emit(KirOp::Fma(next, v, xc, sum));
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let k_next = op2(&mut b, U64, KirOp::Add, k, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next, next])));

    b.set_block(write);
    store_f32(&mut b, y, r, sum);
    finish(b, exit)
}

/// ```text
/// entry: k = global id; if k >= nnz { exit }
/// body:  r = row_indices[k]; c = col_indices[k]
///        atomic y[r] += values[k] · x[c]
/// ```
fn build_coo() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let names = SpmvFormat::Coo.param_names();
    let mut b = KirBuilder::new(SpmvFormat::Coo.kernel_name());
    let row_indices = b.add_param(names[0], index_ptr(U64), Global);
    let col_indices = b.add_param(names[1], index_ptr(U64), Global);
    let values = b.add_param(names[2], f32_ptr(), Global);
    let x = b.add_param(names[3], f32_ptr(), Global);
    let y = b.add_param(names[4], f32_ptr(), Global);
    let nnz = b.add_param(names[5], U64, Global);

    let (k, _, exit) = index_and_bound(&mut b, nnz);
    let row = load_index(&mut b, U64, row_indices, k);
    let col = load_index(&mut b, U64, col_indices, k);
    let v = load_f32(&mut b, values, k);
    let xc = load_f32(&mut b, x, col);
    let p = op2(&mut b, F32, KirOp::MulRn, v, xc);
    let at = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(at, y, row));
    b.emit(KirOp::AtomicAdd(at, p, Global));
    finish(b, exit)
}

/// Build `format`'s kernel as KIR.
pub fn build(format: SpmvFormat) -> KernelIR {
    match format {
        SpmvFormat::Csr => build_csr(),
        SpmvFormat::Coo => build_coo(),
    }
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(format: SpmvFormat) -> Vec<u8> {
    verified_ptx(build(format))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(format: SpmvFormat) -> String {
        String::from_utf8(ptx(format)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for format in SpmvFormat::ALL {
            let p = text(format);
            assert!(p.ends_with('\0'), "{format:?}: NUL-terminated");
            assert!(p.contains(&format!(".visible .entry {}(", format.kernel_name())), "{p}");
            for name in format.param_names() {
                assert!(p.contains(&format!("[param_{name}]")), "{format:?}: {name}\n{p}");
            }
        }
    }

    #[test]
    fn the_index_widths_and_the_arithmetic_follow_the_format() {
        let csr = text(SpmvFormat::Csr);
        assert_eq!(csr.matches("ld.global.u32").count(), 3, "{csr}");
        assert_eq!(csr.matches("fma.rn.f32").count(), 1, "{csr}");
        assert!(!csr.contains("red.") && !csr.contains("mul.rn.f32"), "{csr}");
        let coo = text(SpmvFormat::Coo);
        assert_eq!(coo.matches("ld.global.u64").count(), 2, "{coo}");
        assert_eq!(coo.matches("mul.rn.f32").count(), 1, "{coo}");
        assert_eq!(coo.matches("red.global.add.f32").count(), 1, "{coo}");
        assert!(!coo.contains("fma"), "{coo}");
    }

    /// The CSR loop's bound test branches straight out, with no edge copies
    /// printed around it.
    #[test]
    fn the_csr_loop_exit_carries_no_copies() {
        let p = text(SpmvFormat::Csr);
        assert!(!p.contains("_t:") && !p.contains("_f:") && !p.contains("_else"), "{p}");
    }
}
