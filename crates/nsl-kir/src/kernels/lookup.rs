// crates/nsl-kir/src/kernels/lookup.rs
//! The runtime's 2-D-block row-lookup kernels from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5).
//!
//! Four kernels copy one row of a table per index, `out[i, j] =
//! table[idx(i), j]`, with a thread per output element on a
//! `(ceil(rows / 16), ceil(cols / 16))` grid of 16 × 16 blocks. The thread's
//! row `i` comes from `x` (`%ctaid.x · %ntid.x + %tid.x`) and its column `j`
//! from `y`:
//!
//! - `nsl_embedding_f32(weight, indices, out, seq_len, embed_dim)` and
//!   `nsl_embedding_i32idx(...)`: the embedding lookup. The index is not
//!   range-checked; the host validated it.
//! - `nsl_gather_f32(input, indices, out, num_indices, inner_dim,
//!   input_rows)` and `nsl_gather_i32idx(...)`: NSL's dim-0 gather. An index
//!   at or past `input_rows` leaves its output row unwritten.
//!
//! An f32 index is truncated to `u64` (`cvt.rzi.u64.f32`: toward zero,
//! saturating, a negative or NaN index giving 0). An i32 index is
//! sign-extended (`cvt.u64.s32`), so the gather's unsigned test skips a
//! negative one.
//!
//! Each is the hand kernel's instruction sequence: both coordinates in 32
//! bits widened, the row bound and then the column bound, the index, and
//! for the gather its bound, then the load and the store. The addresses
//! are `(row · cols + j) · 4`, as the hand kernels form them.

use super::elementwise::{f32_ptr, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// The block the runtime launches these kernels with: 16 × 16.
pub const LOOKUP_BLOCK: [u32; 2] = [16, 16];

/// Which table lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupOp {
    /// `out[i, j] = weight[idx(i), j]`, no index bound.
    Embedding,
    /// `out[i, j] = input[idx(i), j]` when `idx(i) < input_rows`.
    Gather,
}

/// How the index buffer holds its indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexDtype {
    /// f32 values, truncated (`cvt.rzi.u64.f32`).
    F32,
    /// i32 values, sign-extended (`cvt.u64.s32`).
    I32,
}

impl LookupOp {
    pub const ALL: [LookupOp; 2] = [LookupOp::Embedding, LookupOp::Gather];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self, idx: IndexDtype) -> &'static str {
        match (self, idx) {
            (LookupOp::Embedding, IndexDtype::F32) => "nsl_embedding_f32",
            (LookupOp::Embedding, IndexDtype::I32) => "nsl_embedding_i32idx",
            (LookupOp::Gather, IndexDtype::F32) => "nsl_gather_f32",
            (LookupOp::Gather, IndexDtype::I32) => "nsl_gather_i32idx",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            LookupOp::Embedding => &["weight", "indices", "out", "seq_len", "embed_dim"],
            LookupOp::Gather => &["input", "indices", "out", "num_indices", "inner_dim", "input_rows"],
        }
    }
}

impl IndexDtype {
    pub const ALL: [IndexDtype; 2] = [IndexDtype::F32, IndexDtype::I32];
}

fn u64_op2(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::U64);
    b.emit(op(dst, x, y));
    dst
}

/// `%ctaid.d · %ntid.d + %tid.d`, widened to 64 bits.
pub(super) fn coordinate(b: &mut KirBuilder, dim: u8) -> VarId {
    let c32 = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(c32, dim));
    let c = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(c, c32, KirType::U64));
    c
}

/// Build `op` with `idx` indices.
///
/// ```text
/// entry:   i = x coordinate; j = y coordinate
///          if i >= rows { exit } else { col }
/// col:     if j >= cols { exit } else { body }
/// body:    r = u64(indices[i])
///          [gather: if r >= input_rows { exit } else { copy }]
/// copy:    out[i·cols + j] = table[r·cols + j]
/// ```
pub fn build_lookup(op: LookupOp, idx: IndexDtype) -> KernelIR {
    use AddressSpace::Global;
    let names = op.param_names();
    let mut b = KirBuilder::new(op.kernel_name(idx));
    let table = b.add_param(names[0], f32_ptr(), Global);
    let idx_ty = match idx {
        IndexDtype::F32 => KirType::F32,
        IndexDtype::I32 => KirType::I32,
    };
    let idx_ptr_ty = KirType::Ptr(Box::new(idx_ty.clone()), Global);
    let indices = b.add_param(names[1], idx_ptr_ty.clone(), Global);
    let out = b.add_param(names[2], f32_ptr(), Global);
    let rows = b.add_param(names[3], KirType::U64, Global);
    let cols = b.add_param(names[4], KirType::U64, Global);
    let table_rows = (op == LookupOp::Gather).then(|| b.add_param(names[5], KirType::U64, Global));

    let entry = b.new_block();
    let col = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();

    b.set_block(entry);
    let i = coordinate(&mut b, 0);
    let j = coordinate(&mut b, 1);
    let past_row = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past_row, i, rows, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past_row, KirEdge::to(exit), KirEdge::to(col)));

    b.set_block(col);
    let past_col = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past_col, j, cols, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past_col, KirEdge::to(exit), KirEdge::to(body)));

    b.set_block(body);
    let addr = b.new_typed_var(idx_ptr_ty);
    b.emit(KirOp::PtrOffset(addr, indices, i));
    let raw = b.new_typed_var(idx_ty);
    b.emit(KirOp::Load(raw, addr, Global));
    let r = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(r, raw, KirType::U64));
    if let Some(table_rows) = table_rows {
        let copy = b.new_block();
        let oob = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(oob, r, table_rows, CmpOp::Ge));
        b.terminate(KirTerminator::CondBranch(oob, KirEdge::to(exit), KirEdge::to(copy)));
        b.set_block(copy);
    }
    let src = u64_op2(&mut b, KirOp::Mul, r, cols);
    let src = u64_op2(&mut b, KirOp::Add, src, j);
    let v = load_f32(&mut b, table, src);
    let dst = u64_op2(&mut b, KirOp::Mul, i, cols);
    let dst = u64_op2(&mut b, KirOp::Add, dst, j);
    store_f32(&mut b, out, dst, v);
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([LOOKUP_BLOCK[0], LOOKUP_BLOCK[1], 1]);
    b.finalize()
}

/// [`build_lookup`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn lookup_ptx(op: LookupOp, idx: IndexDtype) -> Vec<u8> {
    verified_ptx(build_lookup(op, idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptx(op: LookupOp, idx: IndexDtype) -> String {
        String::from_utf8(lookup_ptx(op, idx)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in LookupOp::ALL {
            for idx in IndexDtype::ALL {
                let p = ptx(op, idx);
                assert!(p.ends_with('\0'), "{op:?} {idx:?}: NUL-terminated");
                assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name(idx))), "{p}");
                for name in op.param_names() {
                    assert!(p.contains(&format!("[param_{name}]")), "{op:?} {idx:?}: {name}\n{p}");
                }
                assert!(p.contains("%tid.y") && p.contains("%ntid.y") && p.contains("%ctaid.y"), "{p}");
                // No launch bound: the block is 16 × 16, and `.maxntid 256, 1, 1`
                // would describe a different shape.
                assert!(!p.contains(".maxntid"), "{p}");
            }
        }
    }

    #[test]
    fn the_index_conversion_follows_the_dtype() {
        for op in LookupOp::ALL {
            let f = ptx(op, IndexDtype::F32);
            assert!(f.contains("ld.global.f32") && f.contains("cvt.rzi.u64.f32"), "{f}");
            let i = ptx(op, IndexDtype::I32);
            assert!(i.contains("ld.global.s32") && i.contains("cvt.u64.s32"), "{i}");
        }
    }

    #[test]
    fn only_the_gather_bounds_its_index() {
        for idx in IndexDtype::ALL {
            assert_eq!(ptx(LookupOp::Embedding, idx).matches("setp.ge.u64").count(), 2);
            assert_eq!(ptx(LookupOp::Gather, idx).matches("setp.ge.u64").count(), 3);
        }
    }
}
