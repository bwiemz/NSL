// crates/nsl-kir/src/kernels/embedding_bwd.rs
//! The runtime's embedding backward kernels from `cuda/fused_kernels.rs`,
//! as KIR (new-roadmap item 5).
//!
//! Four kernels scatter `grad [seq_len, embed_dim]` into a pre-zeroed
//! `out [vocab, embed_dim]`, one per index buffer dtype (f32 or i32, see
//! [`IndexDtype`]) and strategy. Both strategies run on a 16 × 16 block,
//! with a thread per `(x, y)`:
//!
//! - **Atomic** (`nsl_embedding_bwd_f32`, `nsl_embedding_bwd_i32idx`): thread
//!   `(i, j)` on a `(seq_len, embed_dim)` grid adds `grad[i, j]` into
//!   `out[idx(i), j]` with `red.global.add.f32`. An index that is negative or
//!   at or past `vocab` is skipped, like the CPU reference's `tok <
//!   vocab_size` check. The order of the additions is the hardware's.
//! - **Deterministic** (`nsl_embedding_bwd_det_f32`,
//!   `nsl_embedding_bwd_det_i32idx`): thread `(v, j)` on a `(vocab,
//!   embed_dim)` grid owns `out[v, j]`. It walks every position `t` in order,
//!   adding `grad[t, j]` wherever `idx(t) == v`, and writes the sum once.
//!   That is the CPU reference's summation order, so the result is bit-exact.
//!
//! An f32 index is truncated toward zero to a signed 64-bit integer
//! (`cvt.rzi.s64.f32`: saturating, NaN to 0). An i32 index is sign-extended
//! (`cvt.s64.s32`).
//!
//! Each is the hand kernel's instruction sequence, with two differences:
//!
//! - **The vocab bound.** The atomic kernels test `idx < 0` signed, as the
//!   hand kernels do, then take the index's bits as unsigned
//!   (`mov.b64`) for the `vocab` bound and the address. The hand kernels
//!   compared `idx >= vocab` signed. For an index already known to be
//!   non-negative, and a `vocab` below 2^63, the two agree.
//! - **The loop exit.** The deterministic kernels' loop leaves through a
//!   block of its own that writes the sum. That keeps block arguments off
//!   the exit edge, whose copies would otherwise print inline in the loop
//!   header and stop ptxas unrolling the loop (see `muon_batch`).

use super::elementwise::{f32_ptr, load_f32, store_f32, verified_ptx};
use super::lookup::{coordinate, IndexDtype, LOOKUP_BLOCK};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// The two scatter strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingBwd {
    /// `red.global.add` from a `(seq_len, embed_dim)` grid.
    Atomic,
    /// A per-output-element loop over the positions, from a `(vocab,
    /// embed_dim)` grid.
    Deterministic,
}

impl EmbeddingBwd {
    pub const ALL: [EmbeddingBwd; 2] = [EmbeddingBwd::Atomic, EmbeddingBwd::Deterministic];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self, idx: IndexDtype) -> &'static str {
        match (self, idx) {
            (EmbeddingBwd::Atomic, IndexDtype::F32) => "nsl_embedding_bwd_f32",
            (EmbeddingBwd::Atomic, IndexDtype::I32) => "nsl_embedding_bwd_i32idx",
            (EmbeddingBwd::Deterministic, IndexDtype::F32) => "nsl_embedding_bwd_det_f32",
            (EmbeddingBwd::Deterministic, IndexDtype::I32) => "nsl_embedding_bwd_det_i32idx",
        }
    }

    /// The parameter names, in launch order. Pinned: the hand modules'.
    pub const PARAMS: [&'static str; 6] = ["grad", "indices", "out", "seq_len", "embed_dim", "vocab"];
}

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

/// `idx(t)` as a signed 64-bit integer: an f32 truncated, an i32
/// sign-extended.
fn index_at(b: &mut KirBuilder, indices: VarId, idx: IndexDtype, t: VarId) -> VarId {
    use AddressSpace::Global;
    let ty = match idx {
        IndexDtype::F32 => KirType::F32,
        IndexDtype::I32 => KirType::I32,
    };
    let addr = var(b, KirType::Ptr(Box::new(ty.clone()), Global));
    b.emit(KirOp::PtrOffset(addr, indices, t));
    let raw = var(b, ty);
    b.emit(KirOp::Load(raw, addr, Global));
    let r = var(b, KirType::I64);
    b.emit(KirOp::Cast(r, raw, KirType::I64));
    r
}

/// The same bits, typed unsigned.
fn unsigned(b: &mut KirBuilder, x: VarId) -> VarId {
    let u = var(b, KirType::U64);
    b.emit(KirOp::Bitcast(u, x));
    u
}

/// `row · cols + j`.
fn flat(b: &mut KirBuilder, row: VarId, cols: VarId, j: VarId) -> VarId {
    let t = op2(b, KirType::U64, KirOp::Mul, row, cols);
    op2(b, KirType::U64, KirOp::Add, t, j)
}

/// Build `op` with `idx` indices.
///
/// Atomic:
///
/// ```text
/// entry:  i = x; j = y; if i >= seq_len { exit }
/// col:    if j >= embed_dim { exit }
/// body:   r = idx(i); if r < 0 { exit }
/// range:  u = bits(r); if u >= vocab { exit }
/// add:    red.add out[u·embed + j], grad[i·embed + j]
/// ```
///
/// Deterministic:
///
/// ```text
/// entry:      v = x; j = y; if v >= vocab { exit }
/// col:        if j >= embed_dim { exit }; br walk(0, 0.0)
/// walk(t, s): if t >= seq_len { done }
/// body:       if bits(idx(t)) == v { hit } else { skip }
/// hit:        br next(s + grad[t·embed + j])
/// skip:       br next(s)
/// next(s'):   br walk(t + 1, s')
/// done:       out[v·embed + j] = s
/// ```
///
/// Every conditional branch in the loop carries no block arguments, so no
/// edge copies print inline around a predicated branch; with them, ptxas
/// neither if-converts nor unrolls the loop the way it does the hand
/// kernel's.
pub fn build(op: EmbeddingBwd, idx: IndexDtype) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, I64, U64};
    let mut b = KirBuilder::new(op.kernel_name(idx));
    let grad = b.add_param("grad", f32_ptr(), Global);
    let idx_ty = match idx {
        IndexDtype::F32 => F32,
        IndexDtype::I32 => KirType::I32,
    };
    let indices = b.add_param("indices", KirType::Ptr(Box::new(idx_ty), Global), Global);
    let out = b.add_param("out", f32_ptr(), Global);
    let seq_len = b.add_param("seq_len", U64, Global);
    let embed = b.add_param("embed_dim", U64, Global);
    let vocab = b.add_param("vocab", U64, Global);

    let entry = b.new_block();
    let col = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let x = coordinate(&mut b, 0);
    let j = coordinate(&mut b, 1);
    let x_bound = match op {
        EmbeddingBwd::Atomic => seq_len,
        EmbeddingBwd::Deterministic => vocab,
    };
    let past = cmp(&mut b, x, x_bound, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(col)));
    b.set_block(col);
    let past = cmp(&mut b, j, embed, CmpOp::Ge);

    match op {
        EmbeddingBwd::Atomic => {
            let body = b.new_block();
            let range = b.new_block();
            let add = b.new_block();
            b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));

            b.set_block(body);
            let r = index_at(&mut b, indices, idx, x);
            let zero = var(&mut b, I64);
            b.emit(KirOp::Const(zero, KirConst { ty: I64, value: ConstValue::I64(0) }));
            let negative = cmp(&mut b, r, zero, CmpOp::Lt);
            b.terminate(KirTerminator::CondBranch(negative, KirEdge::to(exit), KirEdge::to(range)));

            b.set_block(range);
            let u = unsigned(&mut b, r);
            let too_big = cmp(&mut b, u, vocab, CmpOp::Ge);
            b.terminate(KirTerminator::CondBranch(too_big, KirEdge::to(exit), KirEdge::to(add)));

            b.set_block(add);
            let src = flat(&mut b, x, embed, j);
            let g = load_f32(&mut b, grad, src);
            let dst = flat(&mut b, u, embed, j);
            let addr = var(&mut b, f32_ptr());
            b.emit(KirOp::PtrOffset(addr, out, dst));
            b.emit(KirOp::AtomicAdd(addr, g, Global));
            b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
        }
        EmbeddingBwd::Deterministic => {
            let walk = b.new_block();
            let body = b.new_block();
            let hit = b.new_block();
            let skip = b.new_block();
            let next = b.new_block();
            let done = b.new_block();
            let t = b.add_block_param(walk, U64);
            let s = b.add_block_param(walk, F32);
            let s_joined = b.add_block_param(next, F32);
            let t0 = var(&mut b, U64);
            b.emit(KirOp::Const(t0, KirConst { ty: U64, value: ConstValue::U64(0) }));
            let s0 = var(&mut b, F32);
            b.emit(KirOp::Const(s0, KirConst { ty: F32, value: ConstValue::F32(0.0) }));
            b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::with(walk, vec![t0, s0])));

            b.set_block(walk);
            let end = cmp(&mut b, t, seq_len, CmpOp::Ge);
            b.terminate(KirTerminator::CondBranch(end, KirEdge::to(done), KirEdge::to(body)));

            b.set_block(body);
            let r = index_at(&mut b, indices, idx, t);
            let u = unsigned(&mut b, r);
            let other = cmp(&mut b, u, x, CmpOp::Ne);
            b.terminate(KirTerminator::CondBranch(other, KirEdge::to(skip), KirEdge::to(hit)));

            b.set_block(hit);
            let src = flat(&mut b, t, embed, j);
            let g = load_f32(&mut b, grad, src);
            let s_hit = op2(&mut b, F32, KirOp::Add, s, g);
            b.terminate(KirTerminator::Branch(KirEdge::with(next, vec![s_hit])));

            b.set_block(skip);
            b.terminate(KirTerminator::Branch(KirEdge::with(next, vec![s])));

            b.set_block(next);
            let one = var(&mut b, U64);
            b.emit(KirOp::Const(one, KirConst { ty: U64, value: ConstValue::U64(1) }));
            let t_next = op2(&mut b, U64, KirOp::Add, t, one);
            b.terminate(KirTerminator::Branch(KirEdge::with(walk, vec![t_next, s_joined])));

            b.set_block(done);
            let dst = flat(&mut b, x, embed, j);
            store_f32(&mut b, out, dst, s);
            b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
        }
    }

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([LOOKUP_BLOCK[0], LOOKUP_BLOCK[1], 1]);
    b.finalize()
}

/// [`build`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(op: EmbeddingBwd, idx: IndexDtype) -> Vec<u8> {
    verified_ptx(build(op, idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(op: EmbeddingBwd, idx: IndexDtype) -> String {
        String::from_utf8(ptx(op, idx)).expect("ASCII")
    }

    #[test]
    fn every_kernel_verifies_and_keeps_its_entry_and_parameters() {
        for op in EmbeddingBwd::ALL {
            for idx in IndexDtype::ALL {
                let p = text(op, idx);
                assert!(p.ends_with('\0'), "{op:?} {idx:?}: NUL-terminated");
                assert!(p.contains(&format!(".visible .entry {}(", op.kernel_name(idx))), "{p}");
                for name in EmbeddingBwd::PARAMS {
                    assert!(p.contains(&format!("[param_{name}]")), "{op:?} {idx:?}: {name}\n{p}");
                }
                assert!(!p.contains(".maxntid"), "{p}");
            }
        }
    }

    #[test]
    fn the_index_widening_follows_the_dtype() {
        for op in EmbeddingBwd::ALL {
            assert!(text(op, IndexDtype::F32).contains("cvt.rzi.s64.f32"));
            let i = text(op, IndexDtype::I32);
            assert!(i.contains("ld.global.s32") && i.contains("cvt.s64.s32"), "{i}");
        }
    }

    #[test]
    fn only_the_atomic_kernels_use_an_atomic() {
        for idx in IndexDtype::ALL {
            assert!(text(EmbeddingBwd::Atomic, idx).contains("red.global.add.f32"));
            let d = text(EmbeddingBwd::Deterministic, idx);
            assert!(!d.contains("red.") && !d.contains("atom."), "{d}");
        }
    }

    /// The deterministic loop's bound test branches straight out, with no
    /// edge copies inline in the header (they stop ptxas unrolling).
    #[test]
    fn the_deterministic_loop_exits_without_inline_copies() {
        for idx in IndexDtype::ALL {
            let p = text(EmbeddingBwd::Deterministic, idx);
            let lines: Vec<&str> = p.lines().map(str::trim).collect();
            // The first two `setp.ge.u64` are the grid bounds; the loop's is the third.
            let heads: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("setp.ge.u64 ")).collect();
            assert_eq!(heads.len(), 3, "{p}");
            let h = heads[2];
            assert!(lines[h + 1].starts_with("@%p") && lines[h + 2].starts_with("bra "), "{p}");
        }
    }
}
