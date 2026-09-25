// crates/nsl-kir/src/kernels/elementwise.rs
//! The runtime's elementwise f32 kernels, as KIR (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::kernels` carried every elementwise kernel as a
//! hand-written PTX module; they move here family by family. The binary
//! arithmetic family is first: `nsl_add_f32`, `nsl_sub_f32` and
//! `nsl_mul_f32`, each `c[i] = a[i] op b[i]` over `n` f32 elements.
//!
//! `nsl_div_f32` stays hand-written for now. It divides with
//! `div.approx.f32`, which is within 2 ulp and returns 0 for a divisor
//! whose magnitude is in `(2^126, 2^128)`; KIR's f32 division is the IEEE
//! `div.rn.f32`. Moving it would change what GPU division returns, and the
//! A2 migrations preserve behaviour, so that change is its own step.
//!
//! ## The kernels
//!
//! Signature `(a, b, c, n)`, every parameter a `.u64` (`a`, `b`, `c` are f32
//! pointers), launched one thread per element with a block of
//! [`ELEMENTWISE_BLOCK`] threads and `ceil(n / block)` blocks. A thread past
//! `n` exits. The index is `blockIdx.x * blockDim.x + threadIdx.x`, taken in
//! 32 bits and widened, as the hand kernels did; the bound and the
//! addresses are 64-bit.

use crate::backend_ptx::lower_kir_to_ptx;
use crate::kernel_ir::{
    AddressSpace, CmpOp, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType, VarId,
};

/// The block every elementwise kernel is launched with
/// (`gpu_elementwise_binary` and its in-place twin), declared as the
/// kernels' `.maxntid`.
pub const ELEMENTWISE_BLOCK: u32 = 256;

/// A binary arithmetic kernel, `c[i] = a[i] op b[i]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
}

impl BinaryOp {
    pub const ALL: [BinaryOp; 3] = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            BinaryOp::Add => "nsl_add_f32",
            BinaryOp::Sub => "nsl_sub_f32",
            BinaryOp::Mul => "nsl_mul_f32",
        }
    }

    fn op(self) -> fn(VarId, VarId, VarId) -> KirOp {
        match self {
            BinaryOp::Add => KirOp::Add,
            BinaryOp::Sub => KirOp::Sub,
            BinaryOp::Mul => KirOp::Mul,
        }
    }
}

fn f32_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
}

/// Build `op` as KIR.
///
/// ```text
/// entry:  i = blockIdx.x*blockDim.x + threadIdx.x (u32, widened)
///         if i >= n { br exit } else { br body }
/// body:   c[i] = a[i] op b[i]
/// exit:   ret
/// ```
pub fn build_binary(op: BinaryOp) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(op.kernel_name());

    let a = b.add_param("a", f32_ptr(), Global);
    let bp = b.add_param("b", f32_ptr(), Global);
    let c = b.add_param("c", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);

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
    let load = |b: &mut KirBuilder, base: VarId| {
        let addr = b.new_typed_var(f32_ptr());
        b.emit(KirOp::PtrOffset(addr, base, i));
        let v = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(v, addr, Global));
        v
    };
    let x = load(&mut b, a);
    let y = load(&mut b, bp);
    let z = b.new_typed_var(KirType::F32);
    b.emit((op.op())(z, x, y));
    let out = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(out, c, i));
    b.emit(KirOp::Store(out, z, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);

    b.set_workgroup_size([ELEMENTWISE_BLOCK, 1, 1]);
    b.set_launch_bounds(ELEMENTWISE_BLOCK, None);
    b.finalize()
}

/// Build `op` and lower it to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module, since the
/// kernels take no input.
pub fn binary_ptx(op: BinaryOp) -> Vec<u8> {
    let ir = build_binary(op);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("elementwise kernel `{}` failed KIR verification: {errors:?}", op.kernel_name());
    }
    lower_kir_to_ptx(&ir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kir_verify::verify;

    #[test]
    fn every_binary_kernel_verifies() {
        for op in BinaryOp::ALL {
            if let Err(errors) = verify(&build_binary(op)) {
                panic!("{op:?} failed verification: {errors:?}");
            }
        }
    }

    #[test]
    fn the_signature_and_names_are_pinned() {
        let names: Vec<&str> = BinaryOp::ALL.iter().map(|o| o.kernel_name()).collect();
        assert_eq!(names, ["nsl_add_f32", "nsl_sub_f32", "nsl_mul_f32"]);
        for op in BinaryOp::ALL {
            let ir = build_binary(op);
            assert_eq!(ir.name, op.kernel_name());
            let params: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(params, ["a", "b", "c", "n"], "{op:?}");
            assert_eq!(ir.params[3].ty, KirType::U64, "{op:?}: n is u64");
        }
    }

    /// NUL-terminated ASCII at the ISA floor, with the arithmetic in the
    /// hand kernels' spelling: a bare `add.f32` / `sub.f32` / `mul.f32` has
    /// no contraction partner here, so it rounds as the IEEE operation does.
    #[test]
    fn the_modules_are_nul_terminated_ascii_with_the_expected_op() {
        for (op, mnemonic) in [(BinaryOp::Add, "add.f32 "), (BinaryOp::Sub, "sub.f32 "), (BinaryOp::Mul, "mul.f32 ")] {
            let bytes = binary_ptx(op);
            assert_eq!(bytes.last(), Some(&0), "{op:?}");
            assert!(!bytes[..bytes.len() - 1].contains(&0), "{op:?}");
            assert!(bytes.is_ascii(), "{op:?}");
            let text = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
            assert!(text.starts_with(".version 7.0\n.target sm_70\n"), "{op:?}");
            assert_eq!(text.matches(mnemonic).count(), 1, "{op:?}");
            assert!(text.contains(&format!(".visible .entry {}(", op.kernel_name())), "{op:?}");
        }
    }
}
