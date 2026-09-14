// crates/nsl-kir/src/kernels/cast.rs
//! The four element-wise precision-cast kernels, as KIR (roadmap A2 step 7).
//!
//! These replace `nsl_codegen::precision_cast_ptx` — 436 lines of PTX text
//! assembled by `push_str` — and the four `static` PTX strings the runtime
//! carried alongside it. The runtime could not call the codegen emitter
//! (`nsl-codegen` depends on `nsl-runtime`, so the reverse edge is a
//! cycle), so the bytes were transcribed into `nsl-runtime` and a
//! byte-for-byte parity test kept the copy honest. `nsl-kir` is a leaf
//! crate: both sides can call this module instead, and the copy is gone.
//!
//! ## The kernels
//!
//! | entry | cast | rounding |
//! |---|---|---|
//! | `nsl_cast_f32_to_bf16` | f32 -> bf16 | round-to-nearest-even |
//! | `nsl_cast_bf16_to_f32` | bf16 -> f32 | exact (widening) |
//! | `nsl_cast_f32_to_fp16` | f32 -> f16  | round-to-nearest-even |
//! | `nsl_cast_fp16_to_f32` | f16  -> f32 | exact (widening) |
//!
//! Each is a 1-D grid-stride loop over `numel` elements with the FFI
//! signature `(src_ptr: .u64, dst_ptr: .u64, numel: .u64)`, launched with
//! `block = (CAST_BLOCK_DIM_X, 1, 1)`. The caller may clamp `gridDim.x` to
//! the hardware limit; the grid-stride loop still covers every element.
//!
//! ## Why the index is 64-bit
//!
//! `numel` arrives as a `u64`, and a tensor can exceed 2^32 elements (4.3B
//! bf16 is 8.6 GB — well inside an 80 GB part). A 32-bit induction
//! variable would wrap and leave the tail of `dst` holding whatever the
//! allocator last put there, silently. The index, the stride and the
//! bounds check are all `U64`; only the seed (`blockIdx*blockDim + tid`)
//! and the stride product are computed in 32 bits, which is sound because
//! both are bounded by the CUDA grid limits.
//!
//! Emitting that shape is what required the printer fix for 64-bit
//! parameters and 64-bit `PtrOffset` indices: before it, `numel` was
//! loaded with `ld.param.u32` into a register the comparison never read,
//! and the address arithmetic took its index from the 32-bit file.

use crate::backend_ptx::lower_kir_to_ptx;
#[cfg(test)]
use crate::feature_set::FeatureSet;
use crate::kernel_ir::{
    AddressSpace, CmpOp, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType,
};

/// Block dim every cast kernel is launched with. The kernel itself is
/// agnostic — the grid-stride loop reads `%ntid.x` — but the launcher and
/// the `.maxntid` bound must agree, so the constant lives here with the
/// kernel rather than being restated on the runtime side.
pub const CAST_BLOCK_DIM_X: u32 = 256;

/// Which precision cast a kernel performs.
///
/// The four are separate entries rather than one kernel branching on a
/// dtype argument: the conversion mnemonic is fixed per pair, and the bf16
/// pairs need a higher PTX ISA than the f16 pairs, so conflating them
/// would force every cast onto the stricter header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    F32ToBf16,
    Bf16ToF32,
    F32ToFp16,
    Fp16ToF32,
}

impl CastKind {
    /// Every kind, for exhaustive iteration in tests and in the runtime's
    /// build-at-first-use table.
    pub const ALL: [CastKind; 4] = [
        CastKind::F32ToBf16,
        CastKind::Bf16ToF32,
        CastKind::F32ToFp16,
        CastKind::Fp16ToF32,
    ];

    /// The `.visible .entry` name. Pinned: the runtime looks the kernel up
    /// in the loaded module by exactly this string.
    pub fn kernel_name(self) -> &'static str {
        match self {
            CastKind::F32ToBf16 => "nsl_cast_f32_to_bf16",
            CastKind::Bf16ToF32 => "nsl_cast_bf16_to_f32",
            CastKind::F32ToFp16 => "nsl_cast_f32_to_fp16",
            CastKind::Fp16ToF32 => "nsl_cast_fp16_to_f32",
        }
    }

    /// The element type read from `src_ptr`.
    pub fn src_ty(self) -> KirType {
        match self {
            CastKind::F32ToBf16 | CastKind::F32ToFp16 => KirType::F32,
            CastKind::Bf16ToF32 => KirType::Bf16,
            CastKind::Fp16ToF32 => KirType::F16,
        }
    }

    /// The element type written to `dst_ptr`.
    pub fn dst_ty(self) -> KirType {
        match self {
            CastKind::F32ToBf16 => KirType::Bf16,
            CastKind::F32ToFp16 => KirType::F16,
            CastKind::Bf16ToF32 | CastKind::Fp16ToF32 => KirType::F32,
        }
    }
}

/// Build the kernel for `kind` as KIR.
///
/// The shape is one grid-stride loop:
///
/// ```text
/// entry:                       seed = blockIdx.x*blockDim.x + threadIdx.x  (u32 -> u64)
///                              stride = gridDim.x*blockDim.x               (u32 -> u64)
///                              br loop(seed)
/// loop(idx: u64):              done = idx >= numel
///                              if done { br exit } else { br body }
/// body:                        dst[idx] = cvt(src[idx])
///                              br loop(idx + stride)
/// exit:                        ret
/// ```
///
/// `idx` is a block parameter, which is how a reassigned loop variable is
/// expressed in SSA — the back edge passes the next value rather than
/// writing the register a second time.
pub fn build(kind: CastKind) -> KernelIR {
    let src_ty = kind.src_ty();
    let dst_ty = kind.dst_ty();

    let mut b = KirBuilder::new(kind.kernel_name());

    let src = b.add_param(
        "src_ptr",
        KirType::Ptr(Box::new(src_ty.clone()), AddressSpace::Global),
        AddressSpace::Global,
    );
    let dst = b.add_param(
        "dst_ptr",
        KirType::Ptr(Box::new(dst_ty.clone()), AddressSpace::Global),
        AddressSpace::Global,
    );
    let numel = b.add_param("numel", KirType::U64, AddressSpace::Global);

    let entry = b.new_block();
    let loop_head = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();

    // The induction variable: `u64`, so a tensor with more than 2^32
    // elements is iterated to the end rather than modulo 2^32.
    let idx = b.add_block_param(loop_head, KirType::U64);

    // ── entry: seed and stride ───────────────────────────────────────
    b.set_block(entry);

    // seed = blockIdx.x * blockDim.x + threadIdx.x. `GlobalId` lowers to
    // `mul.lo.u32` + `add.u32` — never `mad.lo.u32`, which is invalid at
    // PTX ISA 7.0 (see the printer's own guard test).
    let seed32 = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(seed32, 0));
    let seed = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(seed, seed32, KirType::U64));

    // stride = gridDim.x * blockDim.x. Both factors are u32 by the CUDA
    // launch limits, so the product is computed in 32 bits and widened.
    let gdim = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GridDim(gdim, 0));
    let bdim = b.new_typed_var(KirType::U32);
    b.emit(KirOp::BlockDim(bdim, 0));
    let stride32 = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Mul(stride32, gdim, bdim));
    let stride = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(stride, stride32, KirType::U64));

    b.terminate(KirTerminator::Branch(KirEdge::with(loop_head, vec![seed])));

    // ── loop head: the bounds check ──────────────────────────────────
    b.set_block(loop_head);
    let done = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(done, idx, numel, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(
        done,
        KirEdge::to(exit),
        KirEdge::to(body),
    ));

    // ── body: one element ────────────────────────────────────────────
    b.set_block(body);

    // `PtrOffset` scales by the pointee size, so the same index serves both
    // sides even though the element widths differ (4 bytes one way, 2 the
    // other).
    let src_addr = b.new_typed_var(KirType::Ptr(Box::new(src_ty.clone()), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(src_addr, src, idx));
    let val = b.new_typed_var(src_ty);
    b.emit(KirOp::Load(val, src_addr, AddressSpace::Global));

    // Narrowing picks up `.rn` (round-to-nearest-even) from the printer's
    // default; widening rounds not at all, because every bf16 and f16 value
    // is exactly representable in f32.
    let out = b.new_typed_var(dst_ty.clone());
    b.emit(KirOp::Cast(out, val, dst_ty.clone()));

    let dst_addr = b.new_typed_var(KirType::Ptr(Box::new(dst_ty), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(dst_addr, dst, idx));
    b.emit(KirOp::Store(dst_addr, out, AddressSpace::Global));

    let next = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Add(next, idx, stride));
    b.terminate(KirTerminator::Branch(KirEdge::with(loop_head, vec![next])));

    // ── exit ─────────────────────────────────────────────────────────
    b.set_block(exit);
    b.terminate(KirTerminator::Return);

    b.set_workgroup_size([CAST_BLOCK_DIM_X, 1, 1]);
    b.set_launch_bounds(CAST_BLOCK_DIM_X, None);
    // `required_features` is not declared here: the builder infers
    // `BF16_ARITHMETIC` from the bf16 `Cast`, which is what moves the
    // header to PTX ISA 7.8 / sm_80 for those two kernels. The f16 pairs
    // require nothing extra, so they stay on the lower floor rather than
    // being dragged up by an association they do not have.
    b.finalize()
}

/// Build `kind` and lower it to a NUL-terminated PTX module.
///
/// The terminator is `lower_kir_to_ptx`'s, and it is what
/// `cuModuleLoadData` requires — it reads the buffer as a C string.
///
/// # Panics
///
/// If the built kernel fails verification. That is a bug in this module,
/// not a condition a caller can provoke or recover from: the four kernels
/// are fixed, take no user input, and are verified by this crate's tests
/// on every build.
pub fn ptx(kind: CastKind) -> Vec<u8> {
    let ir = build(kind);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!(
            "cast kernel `{}` failed KIR verification: {:?}",
            kind.kernel_name(),
            errors
        );
    }
    lower_kir_to_ptx(&ir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kir_verify::verify;

    /// Every kernel this module builds must satisfy the verifier. `ptx`
    /// panics if one does not, so this is the check that turns that panic
    /// into a named test failure rather than a surprise at first cast.
    #[test]
    fn all_four_kernels_verify() {
        for kind in CastKind::ALL {
            let ir = build(kind);
            if let Err(errors) = verify(&ir) {
                panic!("{:?} failed verification: {errors:?}", kind);
            }
        }
    }

    /// The FFI contract: three parameters, `(src, dst, numel)`, with the
    /// pointer element types the cast reads and writes. The runtime
    /// marshals arguments positionally, so a reordering here would be a
    /// silent memory-corruption bug at launch.
    #[test]
    fn the_parameter_list_is_src_dst_numel() {
        for kind in CastKind::ALL {
            let ir = build(kind);
            let names: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(names, ["src_ptr", "dst_ptr", "numel"], "{:?}", kind);
            assert_eq!(
                ir.params[0].ty,
                KirType::Ptr(Box::new(kind.src_ty()), AddressSpace::Global),
                "{:?}: source pointer element type",
                kind
            );
            assert_eq!(
                ir.params[1].ty,
                KirType::Ptr(Box::new(kind.dst_ty()), AddressSpace::Global),
                "{:?}: destination pointer element type",
                kind
            );
            // `numel` must stay 64-bit: a u32 count silently truncates a
            // tensor of more than 2^32 elements.
            assert_eq!(ir.params[2].ty, KirType::U64, "{:?}: numel is u64", kind);
        }
    }

    /// The induction variable is a block parameter typed `U64`. This is
    /// the property that makes the loop cover a tensor larger than 2^32
    /// elements, and it is invisible in the PTX unless you know which
    /// register the allocator gave it — so pin it on the IR instead.
    #[test]
    fn the_loop_index_is_a_64_bit_block_parameter() {
        for kind in CastKind::ALL {
            let ir = build(kind);
            let carrying: Vec<&crate::kernel_ir::KirBlock> =
                ir.blocks.iter().filter(|b| !b.params.is_empty()).collect();
            assert_eq!(carrying.len(), 1, "{:?}: exactly one block carries the index", kind);
            assert_eq!(carrying[0].params.len(), 1, "{:?}: one loop-carried value", kind);
            let idx = &carrying[0].params[0];
            assert_eq!(idx.ty, KirType::U64, "{:?}: the loop index must be u64", kind);
            assert_eq!(
                ir.var_types.get(&idx.id),
                Some(&KirType::U64),
                "{:?}: the index's recorded type must agree with its declaration",
                kind
            );
        }
    }

    /// Only the bf16 pairs may raise the header. Asserting the f16 pairs
    /// require *nothing* is the half that matters: an over-broad feature
    /// set would push them to sm_80 and narrow the devices they load on.
    #[test]
    fn only_the_bf16_pairs_require_a_feature() {
        for kind in [CastKind::F32ToBf16, CastKind::Bf16ToF32] {
            assert!(
                build(kind).required_features.contains(FeatureSet::BF16_ARITHMETIC),
                "{:?} needs the bf16 feature",
                kind
            );
        }
        for kind in [CastKind::F32ToFp16, CastKind::Fp16ToF32] {
            assert!(
                build(kind).required_features.is_empty(),
                "{:?} must not require a feature it does not use",
                kind
            );
        }
    }

    /// The entry names are an ABI the runtime looks up by string.
    #[test]
    fn the_entry_names_are_pinned() {
        assert_eq!(CastKind::F32ToBf16.kernel_name(), "nsl_cast_f32_to_bf16");
        assert_eq!(CastKind::Bf16ToF32.kernel_name(), "nsl_cast_bf16_to_f32");
        assert_eq!(CastKind::F32ToFp16.kernel_name(), "nsl_cast_f32_to_fp16");
        assert_eq!(CastKind::Fp16ToF32.kernel_name(), "nsl_cast_fp16_to_f32");
        for kind in CastKind::ALL {
            assert_eq!(build(kind).name, kind.kernel_name(), "{:?}", kind);
        }
    }

    /// The lowered module is NUL-terminated and ASCII: `cuModuleLoadData`
    /// reads it as a C string, and non-ASCII trips `CUDA_ERROR_INVALID_PTX`
    /// under the JIT.
    #[test]
    fn the_lowered_modules_are_nul_terminated_ascii() {
        for kind in CastKind::ALL {
            let bytes = ptx(kind);
            assert_eq!(bytes.last(), Some(&0), "{:?} must be NUL-terminated", kind);
            assert!(
                !bytes[..bytes.len() - 1].contains(&0),
                "{:?} must hold exactly one NUL, at the end",
                kind
            );
            assert!(bytes.is_ascii(), "{:?} must be ASCII-only", kind);
        }
    }
}
