// crates/nsl-kir/src/kernels/lce_rows.rs
//! The row-chunked fused linear cross-entropy's softmax kernel: the loss of a
//! chunk of token rows AND the gradient of their logits, in one pass over
//! logits the runtime computed for the whole vocabulary of those rows.
//!
//! `nsl_lce_rows_grad_{f32,bf16}(logits, targets, loss, rows, cols, scale)`
//! runs one block of [`LCE_ROWS_BLOCK`] threads per row (`%ctaid.x`; a block
//! at or past `rows` does nothing). With `z_j = logits[row, j]` widened to f32
//! (the `bf16` instantiation stores the logits bf16) and `t = targets[row]`
//! (an `s64`; a negative target is the ignore index):
//!
//! 1. each thread folds `max` over `j = tid, tid + 256, …` from `-inf`; thread
//!    0 folds `sdata[1 .. 256]` into its own in order and publishes the row
//!    max `m` in `sdata[0]`;
//! 2. each thread sums `ex2((z_j - m) · log2 e)` from `+0`; thread 0 folds the
//!    partials in order into `s`, takes `lse = fma(lg2(s), ln 2, m)` (as
//!    `nsl_lce_finalize_f32` does), stores `loss[row] = t < 0 ? 0 : lse -
//!    z_t` (reading `z_t` before any thread writes the row) and publishes
//!    `lse` in `sdata[0]`;
//! 3. each thread overwrites its columns in place with the gradient of the
//!    row's loss times `scale`:
//!
//!    ```text
//!    dl_j = t < 0 ? 0 : (ex2((z_j - lse) · log2 e) - [j == t]) · scale
//!    ```
//!
//!    rounded once to the logits' storage (`cvt.rn.bf16.f32` for bf16).
//!
//! The arithmetic is the GEMM-chunked path's (`lce_chunk`, `lce_finalize`)
//! restated over a whole row: the same `ex2.approx` / `lg2.approx` family, the
//! same explicitly rounded add, subtract and multiply, thread 0's folds in
//! thread order. What differs is that the row's max and sum come from the
//! whole row at once instead of an online fold over vocabulary chunks, so the
//! LSE is not bit-identical to that path's.

use super::elementwise::{f32_ptr, verified_ptx};
use super::lce_chunk::{
    LCE_CHUNK_BLOCK, cast, cmp, column_loop, exp_shifted, konst, load_at, op2, store_at,
    thread0_fold, var,
};
use super::storage::{Storage, load_as_f32, store_from_f32};
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator,
    KirType, SmemLayout, SmemRegion, VarId,
};

/// The block the kernel is launched with. Its column loops stride by it and
/// thread 0's folds run to it (the `lce_chunk` helpers it reuses).
pub const LCE_ROWS_BLOCK: u32 = LCE_CHUNK_BLOCK;

/// `ln 2` as `nsl_lce_finalize_f32` spells it (`0f3F317218`).
pub const LN_2: f32 = f32::from_bits(0x3F31_7218);

/// The parameter names, in launch order. Pinned: the runtime launches by
/// position.
pub const PARAM_NAMES: [&str; 6] = ["logits", "targets", "loss", "rows", "cols", "scale"];

/// The `.visible .entry` name of the instantiation whose logits are stored
/// `s`. Pinned: the runtime launches by it.
pub fn kernel_name(s: Storage) -> &'static str {
    match s {
        Storage::F32 => "nsl_lce_rows_grad_f32",
        Storage::Bf16 => "nsl_lce_rows_grad_bf16",
    }
}

/// Thread 0 runs `first`; every thread then meets at a barrier in the block
/// this returns (current on return). `first` must leave the builder in a
/// block it has not terminated.
fn thread0_section(b: &mut KirBuilder, tid32: VarId, first: impl FnOnce(&mut KirBuilder)) {
    use KirType::U32;
    let first_block = b.new_block();
    let wait = b.new_block();
    let zero32 = konst(b, U32, ConstValue::U32(0));
    let not_first = cmp(b, tid32, zero32, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(
        not_first,
        KirEdge::to(wait),
        KirEdge::to(first_block),
    ));

    b.set_block(first_block);
    first(b);
    b.terminate(KirTerminator::Branch(KirEdge::to(wait)));

    b.set_block(wait);
    b.emit(KirOp::Barrier);
}

/// ```text
/// entry:  row = %ctaid.x; if row >= rows { exit }
/// setup:  lr = logits + row·cols; t = targets[row]; valid = t >= 0
/// pass 1: m = max over z(k); sdata[tid] = m; bar
///         tid 0: sdata[0] = fold max sdata[1..256] (onto m); bar; mn = sdata[0]
/// pass 2: s = Σ ex2((z(k) - mn) · log2 e); bar; sdata[tid] = s; bar
///         tid 0: s = fold add sdata[1..256]; lse = fma(lg2(s), ln 2, mn)
///                loss[row] = valid ? lse - z(t) : 0; sdata[0] = lse
///         bar; lse = sdata[0]
/// pass 3: z(k) = valid ? (e = ex2((z(k) - lse) · log2 e); (k != t ? e : e - 1) · scale) : 0
/// ```
fn build(s: Storage) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, I64, U32, U64};
    let mut b = KirBuilder::new(kernel_name(s));
    let logits = b.add_param(PARAM_NAMES[0], s.ptr(), Global);
    let targets = b.add_param(PARAM_NAMES[1], KirType::Ptr(Box::new(I64), Global), Global);
    let loss = b.add_param(PARAM_NAMES[2], f32_ptr(), Global);
    let rows = b.add_param(PARAM_NAMES[3], U64, Global);
    let cols = b.add_param(PARAM_NAMES[4], U64, Global);
    let scale = b.add_param(PARAM_NAMES[5], F32, Global);
    b.set_smem_layout(SmemLayout {
        regions: vec![SmemRegion {
            name: "sdata".to_string(),
            bytes: LCE_ROWS_BLOCK * 4,
            align: 4,
            elem: F32,
        }],
        dynamic: false,
    });

    let entry = b.new_block();
    let setup = b.new_block();
    let exit: BlockId = b.new_block();

    b.set_block(entry);
    let row32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(row32, 0));
    let row = cast(&mut b, row32, U64);
    let past = cmp(&mut b, row, rows, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        past,
        KirEdge::to(exit),
        KirEdge::to(setup),
    ));

    b.set_block(setup);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = cast(&mut b, tid32, U64);
    let base = op2(&mut b, U64, KirOp::Mul, row, cols);
    let logits_row = var(&mut b, s.ptr());
    b.emit(KirOp::PtrOffset(logits_row, logits, base));
    let sdata = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion {
        dst: sdata,
        region: 0,
    });
    let t = load_at(&mut b, I64, Global, targets, row);
    let zero64 = konst(&mut b, I64, ConstValue::I64(0));
    let valid = cmp(&mut b, t, zero64, CmpOp::Ge);
    let neg_inf = konst(&mut b, F32, ConstValue::F32(f32::NEG_INFINITY));

    // Pass 1: the row max.
    let m = column_loop(&mut b, tid, cols, neg_inf, |b, k, acc| {
        let z = load_as_f32(b, logits_row, k, s);
        op2(b, F32, KirOp::Max, acc, z)
    });
    store_at(&mut b, Shared, sdata, tid32, m);
    b.emit(KirOp::Barrier);
    thread0_section(&mut b, tid32, |b| {
        let row_max = thread0_fold(b, sdata, m, KirOp::Max);
        b.emit(KirOp::Store(sdata, row_max, Shared));
    });
    let mn = var(&mut b, F32);
    b.emit(KirOp::Load(mn, sdata, Shared));

    // Pass 2: the sum of the exponentials, the LSE and the row's loss.
    let zero = konst(&mut b, F32, ConstValue::F32(0.0));
    let partial = column_loop(&mut b, tid, cols, zero, |b, k, acc| {
        let z = load_as_f32(b, logits_row, k, s);
        let e = exp_shifted(b, z, mn);
        op2(b, F32, KirOp::AddRn, acc, e)
    });
    // No thread may still read `sdata[0]` (the max) when the partials land.
    b.emit(KirOp::Barrier);
    store_at(&mut b, Shared, sdata, tid32, partial);
    b.emit(KirOp::Barrier);
    thread0_section(&mut b, tid32, |b| {
        let sum = thread0_fold(b, sdata, partial, KirOp::AddRn);
        let l2 = var(b, F32);
        b.emit(KirOp::Log2(l2, sum));
        let ln2 = konst(b, F32, ConstValue::F32(LN_2));
        let lse = var(b, F32);
        b.emit(KirOp::Fma(lse, l2, ln2, mn));
        // The target's logit, read before pass 3 overwrites the row.
        let with_target = b.new_block();
        let ignored = b.new_block();
        let store = b.new_block();
        let row_loss = b.add_block_param(store, F32);
        b.terminate(KirTerminator::CondBranch(
            valid,
            KirEdge::to(with_target),
            KirEdge::to(ignored),
        ));

        b.set_block(with_target);
        let tj = cast(b, t, U64);
        let zt = load_as_f32(b, logits_row, tj, s);
        let d = op2(b, F32, KirOp::SubRn, lse, zt);
        b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![d])));

        b.set_block(ignored);
        let none = konst(b, F32, ConstValue::F32(0.0));
        b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![none])));

        b.set_block(store);
        store_at(b, Global, loss, row, row_loss);
        b.emit(KirOp::Store(sdata, lse, Shared));
    });
    let lse = var(&mut b, F32);
    b.emit(KirOp::Load(lse, sdata, Shared));

    // Pass 3: the gradient, in place. The loop carries a value it does not
    // use (the helper's shape); every thread writes only its own columns.
    let _ = column_loop(&mut b, tid, cols, zero, |b, k, acc| {
        let z = load_as_f32(b, logits_row, k, s);
        let p = exp_shifted(b, z, lse);
        let ks = cast(b, k, I64);
        let other = cmp(b, t, ks, CmpOp::Ne);
        let one = konst(b, F32, ConstValue::F32(1.0));
        let hit = op2(b, F32, KirOp::SubRn, p, one);
        let g = var(b, F32);
        b.emit(KirOp::Select(g, other, p, hit));
        let scaled = op2(b, F32, KirOp::MulRn, g, scale);
        let none = konst(b, F32, ConstValue::F32(0.0));
        let dl = var(b, F32);
        b.emit(KirOp::Select(dl, valid, scaled, none));
        store_from_f32(b, logits_row, k, dl, s);
        acc
    });
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([LCE_ROWS_BLOCK, 1, 1]);
    b.set_launch_bounds(LCE_ROWS_BLOCK, None);
    b.finalize()
}

/// The kernel whose logits are stored `s`, as KIR.
pub fn build_for(s: Storage) -> KernelIR {
    build(s)
}

/// [`build_for`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn ptx(s: Storage) -> Vec<u8> {
    verified_ptx(build(s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: Storage) -> String {
        String::from_utf8(ptx(s)).expect("ASCII")
    }

    #[test]
    fn both_instantiations_verify_and_keep_their_entry_and_parameters() {
        for s in Storage::ALL {
            let p = text(s);
            assert!(p.ends_with('\0'), "{s:?}");
            assert!(
                p.contains(&format!(".visible .entry {}(", kernel_name(s))),
                "{p}"
            );
            for name in PARAM_NAMES {
                assert!(p.contains(&format!("[param_{name}]")), "{s:?}: {name}\n{p}");
            }
        }
    }

    /// Two `max` folds and an `ex2` per pass over the row, one `lg2` with `ln
    /// 2` folded into one `fma`, three barriers in the passes plus the two
    /// thread-0 joins, no unrounded float add, subtract or multiply.
    #[test]
    fn the_arithmetic_is_the_chunked_paths() {
        for s in Storage::ALL {
            let p = text(s);
            assert!(p.contains(".shared .align 4 .b8 shared_mem[1024]"), "{p}");
            assert_eq!(p.matches("bar.sync 0;").count(), 5, "{s:?}\n{p}");
            assert_eq!(p.matches("max.f32").count(), 2, "{s:?}\n{p}");
            assert_eq!(p.matches("ex2.approx.f32").count(), 2, "{s:?}\n{p}");
            assert_eq!(p.matches("lg2.approx.f32").count(), 1, "{s:?}\n{p}");
            assert_eq!(p.matches("fma.rn.f32").count(), 1, "{s:?}\n{p}");
            assert_eq!(p.matches("0f3F317218").count(), 1, "{s:?}\n{p}");
            assert_eq!(p.matches("ld.global.s64").count(), 1, "{s:?}\n{p}");
            for bare in ["add.f32", "sub.f32", "mul.f32"] {
                assert!(
                    !p.lines().any(|l| l.trim_start().starts_with(bare)),
                    "{bare}\n{p}"
                );
            }
        }
        let b = text(Storage::Bf16);
        assert!(b.contains("cvt.rn.bf16.f32"), "{b}");
        assert!(b.contains("cvt.f32.bf16"), "{b}");
        assert!(!text(Storage::F32).contains("bf16"));
    }
}
