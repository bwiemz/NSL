// crates/nsl-kir/src/kernels/elementwise.rs
//! The runtime's elementwise f32 kernels, as KIR (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::kernels` carried every elementwise kernel as a
//! hand-written PTX module; they move here family by family. The binary
//! arithmetic family is first: `nsl_add_f32`, `nsl_sub_f32` and
//! `nsl_mul_f32`, each `c[i] = a[i] op b[i]` over `n` f32 elements.
//!
//! The unary family follows: `nsl_{neg,relu,exp,log,sqrt,abs,sign,sigmoid,
//! sin,cos,silu,gelu}_f32` (`c[i] = f(a[i])`) and `nsl_clamp_f32`
//! (`c[i] = min(max(a[i], lo), hi)`), each in the hand kernel's instruction
//! sequence: `exp` as `ex2.approx` of `x * log2(e)`, `log` as `lg2.approx`
//! times `ln 2`, the sigmoid family through `ex2.approx` and `rcp.approx`,
//! `sqrt` IEEE-rounded. GELU is the sigmoid approximation `x·σ(k·x)` with
//! `k` = [`GELU_SLOPE`], the slope `nsl_gelu_backward_srcad_f32`
//! differentiates with. `nsl_tanh_f32` stays hand-written: it divides with
//! `div.approx.f32`.
//!
//! The scalar-operand family, `nsl_{mul,add,sub}_scalar_f32`
//! (`c[i] = a[i] op s`, `s` an `.f32` parameter), is the binary kernels with
//! the second load replaced by the parameter. `nsl_div_scalar_f32` stays
//! hand-written with `nsl_div_f32`, for the reason below.
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
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator,
    KirType, VarId,
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

/// A scalar-operand kernel, `c[i] = a[i] op s`, with signature
/// `(a, c, s, n)`: `s` is an `.f32` parameter, the rest `.u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarOp {
    Mul,
    Add,
    Sub,
}

impl ScalarOp {
    pub const ALL: [ScalarOp; 3] = [ScalarOp::Mul, ScalarOp::Add, ScalarOp::Sub];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            ScalarOp::Mul => "nsl_mul_scalar_f32",
            ScalarOp::Add => "nsl_add_scalar_f32",
            ScalarOp::Sub => "nsl_sub_scalar_f32",
        }
    }

    fn op(self) -> fn(VarId, VarId, VarId) -> KirOp {
        match self {
            ScalarOp::Mul => KirOp::Mul,
            ScalarOp::Add => KirOp::Add,
            ScalarOp::Sub => KirOp::Sub,
        }
    }
}

/// Build `op` as KIR: [`build_binary`]'s shape with `b[i]` replaced by the
/// parameter `s`, in the hand kernels' operand order (`a[i] op s`).
pub fn build_scalar(op: ScalarOp) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(op.kernel_name());

    let a = b.add_param("a", f32_ptr(), Global);
    let c = b.add_param("c", f32_ptr(), Global);
    let s = b.add_param("s", KirType::F32, Global);
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
    let src = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(src, a, i));
    let x = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Load(x, src, Global));
    let z = b.new_typed_var(KirType::F32);
    b.emit((op.op())(z, x, s));
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
/// If the built kernel fails verification: a bug in this module.
pub fn scalar_ptx(op: ScalarOp) -> Vec<u8> {
    let ir = build_scalar(op);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("elementwise kernel `{}` failed KIR verification: {errors:?}", op.kernel_name());
    }
    lower_kir_to_ptx(&ir)
}

/// A unary kernel, `c[i] = f(a[i])`, with signature `(a, c, n)`; `Clamp`
/// adds two `.f32` bounds, `(a, c, n, lo, hi)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Relu,
    Exp,
    Log,
    Sqrt,
    Abs,
    Sign,
    Sigmoid,
    Sin,
    Cos,
    Silu,
    Gelu,
    Clamp,
}

/// `log2(e)`, the hand kernels' `0f3FB8AA3B`.
const LOG2_E: u32 = 0x3FB8_AA3B;

/// GELU's slope `k` in `x·σ(k·x)`: 1.702f, `0f3FD9DB23`. The runtime's
/// source-AD backward, `nsl_gelu_backward_srcad_f32`, is hand-written with
/// the same constant; a CPU-lane gate in `nsl-runtime` (`gelu_slope_drift`)
/// holds the two together.
pub const GELU_SLOPE: u32 = 0x3FD9_DB23;

impl UnaryOp {
    pub const ALL: [UnaryOp; 13] = [
        UnaryOp::Neg,
        UnaryOp::Relu,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Abs,
        UnaryOp::Sign,
        UnaryOp::Sigmoid,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::Silu,
        UnaryOp::Gelu,
        UnaryOp::Clamp,
    ];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "nsl_neg_f32",
            UnaryOp::Relu => "nsl_relu_f32",
            UnaryOp::Exp => "nsl_exp_f32",
            UnaryOp::Log => "nsl_log_f32",
            UnaryOp::Sqrt => "nsl_sqrt_f32",
            UnaryOp::Abs => "nsl_abs_f32",
            UnaryOp::Sign => "nsl_sign_f32",
            UnaryOp::Sigmoid => "nsl_sigmoid_f32",
            UnaryOp::Sin => "nsl_sin_f32",
            UnaryOp::Cos => "nsl_cos_f32",
            UnaryOp::Silu => "nsl_silu_f32",
            UnaryOp::Gelu => "nsl_gelu_f32",
            UnaryOp::Clamp => "nsl_clamp_f32",
        }
    }
}

fn f32_const(b: &mut KirBuilder, bits: u32) -> VarId {
    let dst = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(dst, KirConst { ty: KirType::F32, value: ConstValue::F32(f32::from_bits(bits)) }));
    dst
}

fn f32_op1(b: &mut KirBuilder, op: fn(VarId, VarId) -> KirOp, x: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::F32);
    b.emit(op(dst, x));
    dst
}

fn f32_op2(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(KirType::F32);
    b.emit(op(dst, x, y));
    dst
}

/// `1 / (1 + 2^(-x * log2 e))`: the hand kernels' sigmoid, `-x` first.
fn sigmoid_of(b: &mut KirBuilder, x: VarId) -> VarId {
    let negated = f32_op1(b, KirOp::Neg, x);
    let log2e = f32_const(b, LOG2_E);
    let scaled = f32_op2(b, KirOp::Mul, negated, log2e);
    let e = f32_op1(b, KirOp::Exp2, scaled);
    let one = f32_const(b, 1.0f32.to_bits());
    let denom = f32_op2(b, KirOp::Add, e, one);
    f32_op1(b, KirOp::Rcp, denom)
}

/// Build `op` as KIR: the binary kernels' shape with one input.
pub fn build_unary(op: UnaryOp) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(op.kernel_name());

    let a = b.add_param("a", f32_ptr(), Global);
    let c = b.add_param("c", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let bounds = (op == UnaryOp::Clamp).then(|| {
        let lo = b.add_param("lo", KirType::F32, Global);
        let hi = b.add_param("hi", KirType::F32, Global);
        (lo, hi)
    });

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
    let src = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(src, a, i));
    let x = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Load(x, src, Global));
    let y = match op {
        UnaryOp::Neg => f32_op1(&mut b, KirOp::Neg, x),
        UnaryOp::Relu => {
            let zero = f32_const(&mut b, 0);
            f32_op2(&mut b, KirOp::Max, x, zero)
        }
        UnaryOp::Exp => f32_op1(&mut b, KirOp::Exp, x),
        UnaryOp::Log => f32_op1(&mut b, KirOp::Log, x),
        UnaryOp::Sqrt => f32_op1(&mut b, KirOp::Sqrt, x),
        UnaryOp::Abs => f32_op1(&mut b, KirOp::Abs, x),
        UnaryOp::Sign => {
            // 1 if x > 0, -1 if x < 0, else 0 (NaN included).
            let zero = f32_const(&mut b, 0);
            let pos = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Cmp(pos, x, zero, CmpOp::Gt));
            let neg = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Cmp(neg, x, zero, CmpOp::Lt));
            let one = f32_const(&mut b, 1.0f32.to_bits());
            let minus_one = f32_const(&mut b, (-1.0f32).to_bits());
            let t = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Select(t, pos, one, zero));
            let y = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Select(y, neg, minus_one, t));
            y
        }
        UnaryOp::Sigmoid => sigmoid_of(&mut b, x),
        UnaryOp::Sin => f32_op1(&mut b, KirOp::Sin, x),
        UnaryOp::Cos => f32_op1(&mut b, KirOp::Cos, x),
        UnaryOp::Silu => {
            let s = sigmoid_of(&mut b, x);
            f32_op2(&mut b, KirOp::Mul, x, s)
        }
        UnaryOp::Gelu => {
            let slope = f32_const(&mut b, GELU_SLOPE);
            let kx = f32_op2(&mut b, KirOp::Mul, x, slope);
            let s = sigmoid_of(&mut b, kx);
            f32_op2(&mut b, KirOp::Mul, x, s)
        }
        UnaryOp::Clamp => {
            let (lo, hi) = bounds.expect("clamp has bounds");
            let t = f32_op2(&mut b, KirOp::Max, x, lo);
            f32_op2(&mut b, KirOp::Min, t, hi)
        }
    };
    let dst = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(dst, c, i));
    b.emit(KirOp::Store(dst, y, Global));
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
/// If the built kernel fails verification: a bug in this module.
pub fn unary_ptx(op: UnaryOp) -> Vec<u8> {
    let ir = build_unary(op);
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

    #[test]
    fn every_unary_kernel_verifies_and_keeps_its_signature() {
        for op in UnaryOp::ALL {
            let ir = build_unary(op);
            if let Err(errors) = verify(&ir) {
                panic!("{op:?} failed verification: {errors:?}");
            }
            assert_eq!(ir.name, op.kernel_name());
            let params: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            if op == UnaryOp::Clamp {
                assert_eq!(params, ["a", "c", "n", "lo", "hi"]);
                assert_eq!((ir.params[3].ty.clone(), ir.params[4].ty.clone()), (KirType::F32, KirType::F32));
            } else {
                assert_eq!(params, ["a", "c", "n"], "{op:?}");
            }
            let bytes = unary_ptx(op);
            assert_eq!(bytes.last(), Some(&0), "{op:?}");
            assert!(bytes.is_ascii(), "{op:?}");
        }
    }

    /// The approximate forms are the hand kernels': `ex2`/`lg2`/`sin`/`cos`/
    /// `rcp` `.approx`, `sqrt.rn`, and no division at all.
    #[test]
    fn the_unary_kernels_use_the_hand_kernels_math_forms() {
        let text = |op| String::from_utf8(unary_ptx(op)).unwrap();
        for (op, forms) in [
            (UnaryOp::Exp, &["mul.f32 ", "ex2.approx.f32 "][..]),
            (UnaryOp::Log, &["lg2.approx.f32 ", "mul.f32 "][..]),
            (UnaryOp::Sqrt, &["sqrt.rn.f32 "][..]),
            (UnaryOp::Sigmoid, &["neg.f32 ", "ex2.approx.f32 ", "rcp.approx.f32 "][..]),
            (UnaryOp::Gelu, &["0f3FD9DB23", "neg.f32 ", "ex2.approx.f32 ", "rcp.approx.f32 "][..]),
            (UnaryOp::Sin, &["sin.approx.f32 "][..]),
            (UnaryOp::Cos, &["cos.approx.f32 "][..]),
            (UnaryOp::Sign, &["setp.gt.f32 ", "setp.lt.f32 ", "selp.f32 "][..]),
        ] {
            let t = text(op);
            for f in forms {
                assert!(t.contains(f), "{op:?} lacks {f}");
            }
        }
        for op in UnaryOp::ALL {
            assert!(!text(op).contains("div."), "{op:?}");
        }
    }

    #[test]
    fn every_scalar_kernel_verifies_and_keeps_its_signature() {
        let names: Vec<&str> = ScalarOp::ALL.iter().map(|o| o.kernel_name()).collect();
        assert_eq!(names, ["nsl_mul_scalar_f32", "nsl_add_scalar_f32", "nsl_sub_scalar_f32"]);
        for (op, mnemonic) in [(ScalarOp::Mul, "mul.f32 "), (ScalarOp::Add, "add.f32 "), (ScalarOp::Sub, "sub.f32 ")] {
            let ir = build_scalar(op);
            if let Err(errors) = verify(&ir) {
                panic!("{op:?} failed verification: {errors:?}");
            }
            let params: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(params, ["a", "c", "s", "n"], "{op:?}");
            assert_eq!(ir.params[2].ty, KirType::F32, "{op:?}: s is f32");
            assert_eq!(ir.params[3].ty, KirType::U64, "{op:?}: n is u64");
            let bytes = scalar_ptx(op);
            assert_eq!(bytes.last(), Some(&0), "{op:?}");
            assert!(bytes.is_ascii(), "{op:?}");
            let text = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
            assert_eq!(text.matches(mnemonic).count(), 1, "{op:?}");
            assert!(!text.contains("div."), "{op:?}");
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
