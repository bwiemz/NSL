// crates/nsl-kir/src/kernels/data_movement.rs
//! The runtime's data-movement kernels from `cuda/fused_kernels.rs`, as KIR
//! (new-roadmap item 5).
//!
//! Four kernels that move f32 values and do almost no arithmetic:
//!
//! - `nsl_bias_add_f32(inp, bias, out, total, cols)`:
//!   `out[i] = inp[i] + bias[i % cols]`.
//! - `nsl_gather_dim_f32(input, indices, out, outer, gather_dim_size, inner)`:
//!   NSL's dimension-removing gather. With `o = i / inner` and
//!   `k = i % inner`, `out[i] = input[(o·gather_dim_size + idx)·inner + k]`,
//!   where `idx = indices[o]` read as f32 and truncated (`cvt.rzi.u64.f32`).
//!   An index at or past `gather_dim_size` writes 0 instead of reading out of
//!   bounds; the host has already rejected it.
//! - `nsl_strided_copy_f32(src, dst, shape, src_strides, dst_strides, ndim,
//!   total)`: materialises a strided view. The flat output index is split
//!   into coordinates by `dst_strides` (a zero stride skips its dimension),
//!   each coordinate is reduced modulo `shape[d]` (the broadcast case), and
//!   `src_strides` turns them back into a source offset.
//! - `nsl_slice_f32(..., slice_dim, slice_start)`: the same walk, with
//!   `slice_start` added to the `slice_dim` coordinate.
//!
//! Each is launched one thread per output element, `ceil(n / 256)` blocks
//! of 256, and each is the hand kernel's instruction sequence: the index in
//! 32 bits widened, a 64-bit bound, the same loads in the same order, and
//! bare `add.f32` for the bias (no multiply to contract with).

use super::elementwise::{f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// `nsl_bias_add_f32`.
pub const BIAS_ADD_NAME: &str = "nsl_bias_add_f32";
/// `nsl_gather_dim_f32`.
pub const GATHER_DIM_NAME: &str = "nsl_gather_dim_f32";

/// The strided-walk pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StridedOp {
    /// `nsl_strided_copy_f32`.
    Copy,
    /// `nsl_slice_f32`: the copy with `slice_start` added on `slice_dim`.
    Slice,
}

impl StridedOp {
    pub const ALL: [StridedOp; 2] = [StridedOp::Copy, StridedOp::Slice];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            StridedOp::Copy => "nsl_strided_copy_f32",
            StridedOp::Slice => "nsl_slice_f32",
        }
    }
}

fn u64_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::U64), AddressSpace::Global)
}

fn u64_op2(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::U64);
    b.emit(op(dst, x, y));
    dst
}

fn u64_const(b: &mut KirBuilder, v: u64) -> VarId {
    let dst = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Const(dst, KirConst { ty: KirType::U64, value: ConstValue::U64(v) }));
    dst
}

/// `base[i]` for a `u64` array.
fn load_u64(b: &mut KirBuilder, base: VarId, i: VarId) -> VarId {
    let addr = b.new_typed_var(u64_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Load(v, addr, AddressSpace::Global));
    v
}

/// Build `nsl_bias_add_f32`: `out[i] = inp[i] + bias[i % cols]`.
pub fn build_bias_add() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(BIAS_ADD_NAME);
    let inp = b.add_param("inp", f32_ptr(), Global);
    let bias = b.add_param("bias", f32_ptr(), Global);
    let out = b.add_param("out", f32_ptr(), Global);
    let total = b.add_param("total", KirType::U64, Global);
    let cols = b.add_param("cols", KirType::U64, Global);

    let (i, _, exit) = index_and_bound(&mut b, total);
    let x = load_f32(&mut b, inp, i);
    let col = u64_op2(&mut b, KirOp::Rem, i, cols);
    let bv = load_f32(&mut b, bias, col);
    let y = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Add(y, x, bv));
    store_f32(&mut b, out, i, y);
    finish(b, exit)
}

/// [`build_bias_add`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn bias_add_ptx() -> Vec<u8> {
    verified_ptx(build_bias_add())
}

/// Build `nsl_gather_dim_f32`.
///
/// ```text
/// entry:  i (u32, widened); if i >= outer·inner { exit }
/// body:   o = i / inner; k = i % inner; idx = rzi(indices[o])
///         if idx >= gather_dim_size { br store(0.0) } else { br load }
/// load:   br store(input[(o·gds)·inner + idx·inner + k])
/// store(v): out[i] = v
/// ```
pub fn build_gather_dim() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(GATHER_DIM_NAME);
    let input = b.add_param("input", f32_ptr(), Global);
    let indices = b.add_param("indices", f32_ptr(), Global);
    let out = b.add_param("out", f32_ptr(), Global);
    let outer = b.add_param("outer", KirType::U64, Global);
    let gds = b.add_param("gather_dim_size", KirType::U64, Global);
    let inner = b.add_param("inner", KirType::U64, Global);

    // The bound is `outer · inner`, computed before the index is compared.
    let entry = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let i32_ = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(i32_, 0));
    let i = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(i, i32_, KirType::U64));
    let total = u64_op2(&mut b, KirOp::Mul, outer, inner);
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, i, total, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));

    b.set_block(body);
    let o = u64_op2(&mut b, KirOp::Div, i, inner);
    let k = u64_op2(&mut b, KirOp::Rem, i, inner);
    let fidx = load_f32(&mut b, indices, o);
    let idx = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(idx, fidx, KirType::U64));
    let load = b.new_block();
    let store = b.new_block();
    let v = b.add_block_param(store, KirType::F32);
    let zero = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(zero, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
    let oob = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(oob, idx, gds, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(oob, KirEdge::with(store, vec![zero]), KirEdge::to(load)));

    b.set_block(load);
    let base = u64_op2(&mut b, KirOp::Mul, o, gds);
    let base = u64_op2(&mut b, KirOp::Mul, base, inner);
    let row = u64_op2(&mut b, KirOp::Mul, idx, inner);
    let off = u64_op2(&mut b, KirOp::Add, base, row);
    let off = u64_op2(&mut b, KirOp::Add, off, k);
    let x = load_f32(&mut b, input, off);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![x])));

    b.set_block(store);
    store_f32(&mut b, out, i, v);
    finish(b, exit)
}

/// [`build_gather_dim`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn gather_dim_ptx() -> Vec<u8> {
    verified_ptx(build_gather_dim())
}

/// Build `nsl_strided_copy_f32` or `nsl_slice_f32`: the per-dimension walk
/// as a loop whose block parameters are `(dim, remaining, src_offset)`.
///
/// ```text
/// entry:   i; if i >= total { exit }; br head(0, i, 0)
/// head(d, rem, off): if d >= ndim { br copy } else { br dim }
/// dim:     ds = dst_strides[d]; if ds == 0 { br next(rem, off) } else { br walk }
/// walk:    c = (rem / ds) % shape[d]      [slice: + slice_start if d == slice_dim]
///          br next(rem % ds, off + c·src_strides[d])
/// next(rem, off): br head(d + 1, rem, off)
/// copy:    dst[i] = src[off]
/// ```
pub fn build_strided(op: StridedOp) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(op.kernel_name());
    let src = b.add_param("src", f32_ptr(), Global);
    let dst = b.add_param("dst", f32_ptr(), Global);
    let shape = b.add_param("shape", u64_ptr(), Global);
    let src_strides = b.add_param("src_strides", u64_ptr(), Global);
    let dst_strides = b.add_param("dst_strides", u64_ptr(), Global);
    let ndim = b.add_param("ndim", KirType::U64, Global);
    let total = b.add_param("total", KirType::U64, Global);
    let slice = (op == StridedOp::Slice).then(|| {
        let dim = b.add_param("slice_dim", KirType::U64, Global);
        let start = b.add_param("slice_start", KirType::U64, Global);
        (dim, start)
    });

    let (i, _, exit) = index_and_bound(&mut b, total);
    let head = b.new_block();
    let dim_blk = b.new_block();
    let walk = b.new_block();
    let next = b.new_block();
    let copy = b.new_block();
    let d = b.add_block_param(head, KirType::U64);
    let rem = b.add_block_param(head, KirType::U64);
    let off = b.add_block_param(head, KirType::U64);
    let n_rem = b.add_block_param(next, KirType::U64);
    let n_off = b.add_block_param(next, KirType::U64);
    let zero = u64_const(&mut b, 0);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero, i, zero])));

    b.set_block(head);
    let done = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(done, d, ndim, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(done, KirEdge::to(copy), KirEdge::to(dim_blk)));

    b.set_block(dim_blk);
    let ds = load_u64(&mut b, dst_strides, d);
    let zero = u64_const(&mut b, 0);
    let flat = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(flat, ds, zero, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(flat, KirEdge::with(next, vec![rem, off]), KirEdge::to(walk)));

    b.set_block(walk);
    let c = u64_op2(&mut b, KirOp::Div, rem, ds);
    let rem2 = u64_op2(&mut b, KirOp::Rem, rem, ds);
    let extent = load_u64(&mut b, shape, d);
    let mut c = u64_op2(&mut b, KirOp::Rem, c, extent);
    if let Some((slice_dim, start)) = slice {
        let shifted = u64_op2(&mut b, KirOp::Add, c, start);
        let here = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(here, d, slice_dim, CmpOp::Eq));
        let sel = b.new_typed_var(KirType::U64);
        b.emit(KirOp::Select(sel, here, shifted, c));
        c = sel;
    }
    let ss = load_u64(&mut b, src_strides, d);
    let step = u64_op2(&mut b, KirOp::Mul, c, ss);
    let off2 = u64_op2(&mut b, KirOp::Add, off, step);
    b.terminate(KirTerminator::Branch(KirEdge::with(next, vec![rem2, off2])));

    b.set_block(next);
    let one = u64_const(&mut b, 1);
    let d2 = u64_op2(&mut b, KirOp::Add, d, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![d2, n_rem, n_off])));

    b.set_block(copy);
    let x = load_f32(&mut b, src, off);
    store_f32(&mut b, dst, i, x);
    finish(b, exit)
}

/// [`build_strided`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn strided_ptx(op: StridedOp) -> Vec<u8> {
    verified_ptx(build_strided(op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kir_verify::verify;

    #[test]
    fn the_data_movement_kernels_verify_and_keep_their_signatures() {
        let cases: [(KernelIR, &str, &[&str]); 4] = [
            (build_bias_add(), BIAS_ADD_NAME, &["inp", "bias", "out", "total", "cols"]),
            (build_gather_dim(), GATHER_DIM_NAME, &["input", "indices", "out", "outer", "gather_dim_size", "inner"]),
            (
                build_strided(StridedOp::Copy),
                "nsl_strided_copy_f32",
                &["src", "dst", "shape", "src_strides", "dst_strides", "ndim", "total"],
            ),
            (
                build_strided(StridedOp::Slice),
                "nsl_slice_f32",
                &["src", "dst", "shape", "src_strides", "dst_strides", "ndim", "total", "slice_dim", "slice_start"],
            ),
        ];
        for (ir, name, params) in cases {
            verify(&ir).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(ir.name, name);
            let names: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(names, params, "{name}");
        }
        let gd = String::from_utf8(gather_dim_ptx()).unwrap();
        assert!(gd.contains("cvt.rzi.u64.f32 "), "{gd}");
        assert!(gd.contains("div.u64 ") && gd.contains("rem.u64 "), "{gd}");
        let bias = String::from_utf8(bias_add_ptx()).unwrap();
        assert_eq!(bias.matches("add.f32 ").count(), 1);
        assert!(!bias.contains("fma") && !bias.contains(".rn."), "{bias}");
    }
}
