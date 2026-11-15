// crates/nsl-kir/src/kernels/storage.rs
//! The storage dtype of a kernel operand under bf16 activation storage.
//!
//! A storage-generic kernel family takes one [`Storage`] per tensor operand.
//! Arithmetic is always f32: a bf16 operand is widened on load
//! (`ld.global.b16` + `cvt.f32.bf16`, exact) and a bf16 result is rounded
//! once, to nearest even, on store (`cvt.rn.bf16.f32` + `st.global.b16`).
//! An f32 operand is loaded and stored exactly as the f32-only kernels did,
//! with the same KIR op sequence, so a family's all-f32 instantiation prints
//! the bytes it printed before it became generic
//! (`tests/f32_ptx_golden.rs`).

use crate::kernel_ir::{AddressSpace, KirBuilder, KirOp, KirType, VarId};

/// How one tensor operand is stored in global memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Storage {
    F32,
    Bf16,
}

impl Storage {
    /// Both, for exhaustive iteration in tests and kernel tables.
    pub const ALL: [Storage; 2] = [Storage::F32, Storage::Bf16];

    /// The element type in memory.
    pub fn elem(self) -> KirType {
        match self {
            Storage::F32 => KirType::F32,
            Storage::Bf16 => KirType::Bf16,
        }
    }

    /// The global pointer type of an operand stored this way.
    pub fn ptr(self) -> KirType {
        KirType::Ptr(Box::new(self.elem()), AddressSpace::Global)
    }

    /// Bytes per element.
    pub fn bytes(self) -> usize {
        match self {
            Storage::F32 => 4,
            Storage::Bf16 => 2,
        }
    }

    /// The letter a kernel name spells this storage with.
    pub fn letter(self) -> char {
        match self {
            Storage::F32 => 'f',
            Storage::Bf16 => 'b',
        }
    }
}

/// The entry name of `base`'s instantiation for `inputs` and `out`.
///
/// All-f32 keeps `base` itself (the f32 kernel keeps its pinned name).
/// Otherwise the trailing `_f32` is replaced by the operand storages, inputs
/// then output: `nsl_mul_f32` with bf16, bf16 -> bf16 is `nsl_mul_bb_b`,
/// `nsl_add_f32` with f32, bf16 -> f32 is `nsl_add_fb_f`.
pub fn variant_name(base: &str, inputs: &[Storage], out: Storage) -> String {
    if inputs.iter().all(|s| *s == Storage::F32) && out == Storage::F32 {
        return base.to_string();
    }
    let stem = base.strip_suffix("_f32").unwrap_or(base);
    let ins: String = inputs.iter().map(|s| s.letter()).collect();
    format!("{stem}_{ins}_{}", out.letter())
}

/// `base[i]` as an f32 value. For [`Storage::F32`] this is the f32 kernels'
/// own sequence (a typed pointer offset, then `ld.global.f32`); for
/// [`Storage::Bf16`] the bf16 load is widened with `cvt.f32.bf16`, which is
/// exact.
pub fn load_as_f32(b: &mut KirBuilder, base: VarId, i: VarId, s: Storage) -> VarId {
    let addr = b.new_typed_var(s.ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    match s {
        Storage::F32 => {
            let v = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Load(v, addr, AddressSpace::Global));
            v
        }
        Storage::Bf16 => {
            let h = b.new_typed_var(KirType::Bf16);
            b.emit(KirOp::Load(h, addr, AddressSpace::Global));
            let v = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Cast(v, h, KirType::F32));
            v
        }
    }
}

/// `base[i] = v` for an f32 `v`. [`Storage::Bf16`] rounds it once, to
/// nearest even (`cvt.rn.bf16.f32`), before the 16-bit store.
pub fn store_from_f32(b: &mut KirBuilder, base: VarId, i: VarId, v: VarId, s: Storage) {
    let stored = match s {
        Storage::F32 => v,
        Storage::Bf16 => {
            let h = b.new_typed_var(KirType::Bf16);
            b.emit(KirOp::Cast(h, v, KirType::Bf16));
            h
        }
    };
    let addr = b.new_typed_var(s.ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, stored, AddressSpace::Global));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_ptx::lower_kir_to_ptx;
    use crate::kernel_ir::{CmpOp, KirEdge, KirTerminator};

    #[test]
    fn names_keep_the_f32_kernel_and_spell_every_operand_otherwise() {
        use Storage::*;
        assert_eq!(variant_name("nsl_mul_f32", &[F32, F32], F32), "nsl_mul_f32");
        assert_eq!(variant_name("nsl_mul_f32", &[Bf16, Bf16], Bf16), "nsl_mul_bb_b");
        assert_eq!(variant_name("nsl_add_f32", &[F32, Bf16], F32), "nsl_add_fb_f");
        assert_eq!(variant_name("nsl_silu_f32", &[Bf16], Bf16), "nsl_silu_b_b");
        assert_eq!(variant_name("nsl_rmsnorm_f32", &[F32], Bf16), "nsl_rmsnorm_f_b");
    }

    /// A one-element copy kernel through the helpers: a bf16 load is widened
    /// by `cvt.f32.bf16`, a bf16 store narrowed by `cvt.rn.bf16.f32`, and the
    /// 16-bit accesses are `.b16`.
    #[test]
    fn a_bf16_operand_is_widened_on_load_and_rounded_to_nearest_even_on_store() {
        for (src, dst) in [(Storage::Bf16, Storage::F32), (Storage::F32, Storage::Bf16)] {
            let mut b = KirBuilder::new("k");
            let a = b.add_param("a", src.ptr(), AddressSpace::Global);
            let c = b.add_param("c", dst.ptr(), AddressSpace::Global);
            let n = b.add_param("n", KirType::U64, AddressSpace::Global);
            let entry = b.new_block();
            let body = b.new_block();
            let exit = b.new_block();
            b.set_block(entry);
            let i32_ = b.new_typed_var(KirType::U32);
            b.emit(KirOp::GlobalId(i32_, 0));
            let i = b.new_typed_var(KirType::U64);
            b.emit(KirOp::Cast(i, i32_, KirType::U64));
            let past = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Cmp(past, i, n, CmpOp::Ge));
            b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));
            b.set_block(body);
            let v = load_as_f32(&mut b, a, i, src);
            store_from_f32(&mut b, c, i, v, dst);
            b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
            b.set_block(exit);
            b.terminate(KirTerminator::Return);
            let ir = b.finalize();
            crate::kir_verify::verify(&ir).expect("verifies");
            let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
            assert!(ptx.contains(".version 7.8"), "bf16 needs PTX 7.8:\n{ptx}");
            if src == Storage::Bf16 {
                assert!(ptx.contains("ld.global.b16"), "{ptx}");
                assert!(ptx.contains("cvt.f32.bf16"), "{ptx}");
                assert!(ptx.contains("st.global.f32"), "{ptx}");
            } else {
                assert!(ptx.contains("ld.global.f32"), "{ptx}");
                assert!(ptx.contains("cvt.rn.bf16.f32"), "{ptx}");
                assert!(ptx.contains("st.global.b16"), "{ptx}");
            }
        }
    }
}
