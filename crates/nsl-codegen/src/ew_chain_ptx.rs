//! The fused adjoint elementwise-chain kernel (MFU campaign C3), as KIR
//! (new-roadmap item 5).
//!
//! `ew_chain_fusion` collapses a short single-reader run of the adjoint
//! tape's elementwise ops into one `Passthrough("fused_ew:v1:…")` op; the
//! lowerer ships this kernel for its [`ChainSig`], and the runtime
//! (`nsl_fused_ew_chain`) launches it when every input is a GPU-resident,
//! contiguous f32 tensor of one length, one thread per element on a
//! 256-thread block.
//!
//! Signature `(out, in0, …, in{N-1}, n)`, every parameter a `.u64`: the
//! output pointer first, then the inputs in slot order, then the length.
//! Thread `i = %ctaid.x · %ntid.x + %tid.x` (taken in 32 bits and widened)
//! exits when `i >= n`. The others load `in_k[i]` for every input, in slot
//! order, then run the steps register to register in tape order and store
//! the last step's result to `out[i]`:
//!
//! - `Add`, `Sub` and `Mul` are `add.rn` / `sub.rn` / `mul.rn`. The explicit
//!   rounding is load-bearing: the chain replaces standalone `nsl_add_f32`-
//!   class kernels whose single ops round once each, and a bare `add.f32`
//!   or `mul.f32` next to its partner in one kernel is one ptxas may
//!   contract into an `FFMA`, which rounds once for both.
//! - `Div` is `div.approx.f32`, as `nsl_div_f32` is: the chain is held
//!   bit-exact to the decomposed kernels, and theirs is the approximate
//!   divide.
//! - `Neg` is `neg.f32`, which is exact.
//! - `RtsCheck` emits nothing: on the fast path the runtime has already
//!   checked that the like-reference's shape is the uniform one, so the
//!   `reduce_to_shape` it stands for is an identity, and the step's result
//!   is its left operand.
//!
//! An operand is an input slot's value, an earlier step's result, or an f32
//! immediate whose bits the signature carries, materialized with a
//! `mov.f32` of those bits (the hand emitter wrote it inline as an operand;
//! the value is the same).

use crate::backend_ptx::lower_kir_to_ptx;
use crate::ew_chain_fusion::{ChainSig, EwOpcode, Operand};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator,
    KirType, VarId,
};

/// The block `gpu_fused_ew_launch` launches the kernel with.
pub const EW_CHAIN_BLOCK: u32 = 256;

fn f32_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
}

/// Build `sig`'s kernel as KIR, named `kname`.
///
/// ```text
/// entry: i = %ctaid.x·%ntid.x + %tid.x (u32, widened)
///        if i >= n { exit }
/// body:  x_k = in_k[i] for every input k
///        s_j = step j over x, earlier s and immediates
///        out[i] = s_last
/// exit:  ret
/// ```
///
/// # Panics
///
/// If `sig` has no steps, a binary step has no right operand, or an operand
/// names an input slot past `n_inputs` or a step at or after its own: the
/// fuser never builds such a signature.
pub fn build(sig: &ChainSig, kname: &str) -> KernelIR {
    use AddressSpace::Global;
    assert!(!sig.steps.is_empty(), "a fused chain has at least one step");
    let mut b = KirBuilder::new(kname);
    let out = b.add_param("out", f32_ptr(), Global);
    let inputs: Vec<VarId> =
        (0..sig.n_inputs).map(|k| b.add_param(&format!("in{k}"), f32_ptr(), Global)).collect();
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
    let x: Vec<VarId> = inputs
        .iter()
        .map(|&base| {
            let addr = b.new_typed_var(f32_ptr());
            b.emit(KirOp::PtrOffset(addr, base, i));
            let v = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Load(v, addr, Global));
            v
        })
        .collect();

    let mut steps: Vec<VarId> = Vec::with_capacity(sig.steps.len());
    for (j, step) in sig.steps.iter().enumerate() {
        let operand = |b: &mut KirBuilder, o: Operand| match o {
            Operand::Input(k) => x[k as usize],
            Operand::Prev(p) => {
                assert!((p as usize) < j, "step {j} reads step {p}, not an earlier one");
                steps[p as usize]
            }
            Operand::Imm(bits) => {
                let c = b.new_typed_var(KirType::F32);
                b.emit(KirOp::Const(c, KirConst { ty: KirType::F32, value: ConstValue::F32(f32::from_bits(bits)) }));
                c
            }
        };
        let lhs = operand(&mut b, step.lhs);
        let result = match step.op {
            EwOpcode::RtsCheck => lhs,
            EwOpcode::Neg => {
                let dst = b.new_typed_var(KirType::F32);
                b.emit(KirOp::Neg(dst, lhs));
                dst
            }
            EwOpcode::Add | EwOpcode::Sub | EwOpcode::Mul | EwOpcode::Div => {
                let rhs = operand(&mut b, step.rhs.expect("binary chain step carries a rhs"));
                let dst = b.new_typed_var(KirType::F32);
                b.emit(match step.op {
                    EwOpcode::Add => KirOp::AddRn(dst, lhs, rhs),
                    EwOpcode::Sub => KirOp::SubRn(dst, lhs, rhs),
                    EwOpcode::Mul => KirOp::MulRn(dst, lhs, rhs),
                    _ => KirOp::DivApprox(dst, lhs, rhs),
                });
                dst
            }
        };
        steps.push(result);
    }

    let addr = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(addr, out, i));
    b.emit(KirOp::Store(addr, *steps.last().expect("at least one step"), Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// [`build`] lowered to a NUL-terminated PTX module (`cuModuleLoadData`
/// reads it as a C string).
///
/// # Panics
///
/// If the built kernel fails KIR verification: a bug in this module.
pub fn emit(sig: &ChainSig, kname: &str) -> Vec<u8> {
    let ir = build(sig, kname);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{} failed KIR verification: {errors:?}", ir.name);
    }
    lower_kir_to_ptx(&ir)
}
