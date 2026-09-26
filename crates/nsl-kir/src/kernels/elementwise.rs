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
//! Two kernels that must match a decomposed computation bit for bit spell
//! their arithmetic with `KirOp::{AddRn, MulRn}`, which print `.rn` so ptxas
//! cannot contract a multiply and the add that reads it into one `fma`:
//! `nsl_scalar_mul_add_inplace_f32` (`m[i] += g[i] * s`, replacing a
//! scalar multiply then an add) and `nsl_muon_scale_inv_frob_f32`
//! (`c[i] = x[i] * (1 / (sqrt(stats[3]) + 1e-7))`).
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

/// The thread index and the bounds branch every elementwise kernel opens
/// with: `i = blockIdx.x * blockDim.x + threadIdx.x` taken in 32 bits and
/// widened, then `if i >= n { exit } else { body }`. Returns `(i, body,
/// exit)` with the builder in `body`.
fn index_and_bound(b: &mut KirBuilder, n: VarId) -> (VarId, crate::kernel_ir::BlockId, crate::kernel_ir::BlockId) {
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
    (i, body, exit)
}

/// Close `body` into `exit`, return from `exit`, and set the launch shape.
fn finish(mut b: KirBuilder, exit: crate::kernel_ir::BlockId) -> KernelIR {
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([ELEMENTWISE_BLOCK, 1, 1]);
    b.set_launch_bounds(ELEMENTWISE_BLOCK, None);
    b.finalize()
}

fn load_f32(b: &mut KirBuilder, base: VarId, i: VarId) -> VarId {
    let addr = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    let v = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Load(v, addr, AddressSpace::Global));
    v
}

fn store_f32(b: &mut KirBuilder, base: VarId, i: VarId, v: VarId) {
    let addr = b.new_typed_var(f32_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, v, AddressSpace::Global));
}

fn verified_ptx(ir: KernelIR) -> Vec<u8> {
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("elementwise kernel `{}` failed KIR verification: {errors:?}", ir.name);
    }
    lower_kir_to_ptx(&ir)
}

/// `nsl_scalar_mul_add_inplace_f32(m, g, s, n)`: `m[i] = m[i] + g[i] * s`,
/// the FASE accumulate epilogue. It replaces `nsl_mul_scalar_f32` then
/// `nsl_add_f32`, whose intermediate rounds to f32 through memory, so the
/// multiply and the add are both explicitly rounded (`MulRn`, `AddRn`): a
/// contracted `fma` would round once and differ by up to an ulp. The order
/// is the hand kernel's: load `g[i]`, scale, load `m[i]`, add, store.
pub fn build_scalar_mul_add_inplace() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new("nsl_scalar_mul_add_inplace_f32");
    let m = b.add_param("m", f32_ptr(), Global);
    let g = b.add_param("g", f32_ptr(), Global);
    let s = b.add_param("s", KirType::F32, Global);
    let n = b.add_param("n", KirType::U64, Global);
    let (i, _, exit) = index_and_bound(&mut b, n);
    let gi = load_f32(&mut b, g, i);
    let scaled = f32_op2(&mut b, KirOp::MulRn, gi, s);
    let mi = load_f32(&mut b, m, i);
    let sum = f32_op2(&mut b, KirOp::AddRn, mi, scaled);
    store_f32(&mut b, m, i, sum);
    finish(b, exit)
}

/// [`build_scalar_mul_add_inplace`], lowered to a NUL-terminated module.
pub fn scalar_mul_add_inplace_ptx() -> Vec<u8> {
    verified_ptx(build_scalar_mul_add_inplace())
}

/// `1e-7f`, the Muon pre-normalization epsilon (`0f33D6BF95`).
const MUON_EPS: u32 = 0x33D6_BF95;

/// `nsl_muon_scale_inv_frob_f32(x, c, stats, n)`:
/// `c[i] = x[i] * (1 / (sqrt(stats[3]) + 1e-7))`, with `stats` the 4-slot
/// output of `nsl_tensor_stats_f32` (slot 3 the raw sum of squares), read
/// on the device so Muon's Newton-Schulz pre-normalization needs no host
/// sync. Every thread recomputes the scale: `sqrt.rn`, an `AddRn` of the
/// epsilon, `div.rn` of 1, then an `MulRn`, in the hand kernel's order.
pub fn build_muon_scale_inv_frob() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new("nsl_muon_scale_inv_frob_f32");
    let x = b.add_param("x", f32_ptr(), Global);
    let c = b.add_param("c", f32_ptr(), Global);
    let stats = b.add_param("stats", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let (i, _, exit) = index_and_bound(&mut b, n);
    let slot3 = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Const(slot3, KirConst { ty: KirType::U64, value: ConstValue::U64(3) }));
    let sumsq = load_f32(&mut b, stats, slot3);
    let norm = f32_op1(&mut b, KirOp::Sqrt, sumsq);
    let eps = f32_const(&mut b, MUON_EPS);
    let denom = f32_op2(&mut b, KirOp::AddRn, norm, eps);
    let one = f32_const(&mut b, 1.0f32.to_bits());
    let inv = f32_op2(&mut b, KirOp::Div, one, denom);
    let xi = load_f32(&mut b, x, i);
    let y = f32_op2(&mut b, KirOp::MulRn, xi, inv);
    store_f32(&mut b, c, i, y);
    finish(b, exit)
}

/// [`build_muon_scale_inv_frob`], lowered to a NUL-terminated module.
pub fn muon_scale_inv_frob_ptx() -> Vec<u8> {
    verified_ptx(build_muon_scale_inv_frob())
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

/// An activation-backward kernel: the gradient of a unary activation, one
/// thread per element. Each reads the upstream gradient and one saved
/// tensor, `(grad, input | saved, out, n)`; `SwigluGate` reads a second
/// operand, `(grad, up, input, out, n)`.
///
/// Two families share the shapes. The tape-AD kernels (`Relu`, `Sigmoid`,
/// `Tanh`, `Silu`) spell their arithmetic bare, so ptxas may
/// contract a multiply and the add that reads it, as it did for the hand
/// kernels. The source-AD kernels (`*Srcad`, `SwigluGate`) replace a chain
/// of separate launches and must match it bit for bit, so every derivative
/// operation is explicitly rounded (`MulRn`/`AddRn`/`SubRn`); their sigmoid
/// stays bare, as in `nsl_sigmoid_f32`, since `ex2.approx`/`rcp.approx`
/// leave no contractible pair. `nsl_gelu_backward_f32` (the tape-AD tanh
/// approximation) stays hand-written: it divides with `div.approx.f32`.
/// `nsl_clamp_backward_f32` follows once the interpreter its gate runs on
/// models `and.pred`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackwardOp {
    /// `out = input > 0 ? grad : 0`.
    Relu,
    /// `out = grad * (y * (1 - y))`, `y` the saved sigmoid output.
    Sigmoid,
    /// `out = grad * (1 - y * y)`, `y` the saved tanh output.
    Tanh,
    /// `s = σ(x)`; `out = grad * (s + s * (x * (1 - s)))`.
    Silu,
    /// Source-AD `grad * (y * (1 - y))`, every operation rounded.
    SigmoidSrcad,
    /// Source-AD `grad * (1 - y * y)`, every operation rounded.
    TanhSrcad,
    /// Source-AD `s = σ(x)`; `grad * (s * (1 + x * (1 - s)))`.
    SiluSrcad,
    /// Source-AD `kx = k·x`, `s = σ(kx)`; `grad * (s * (1 + kx * (1 - s)))`,
    /// `k` = [`GELU_SLOPE`]: the exact derivative of `nsl_gelu_f32`.
    GeluSrcad,
    /// `t = grad * up`, then [`BackwardOp::SiluSrcad`] of `t` at `input`:
    /// the SwiGLU gate adjoint, bit-exact with the multiply launch and the
    /// silu-backward launch it replaces.
    SwigluGate,
}

impl BackwardOp {
    pub const ALL: [BackwardOp; 9] = [
        BackwardOp::Relu,
        BackwardOp::Sigmoid,
        BackwardOp::Tanh,
        BackwardOp::Silu,
        BackwardOp::SigmoidSrcad,
        BackwardOp::TanhSrcad,
        BackwardOp::SiluSrcad,
        BackwardOp::GeluSrcad,
        BackwardOp::SwigluGate,
    ];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            BackwardOp::Relu => "nsl_relu_backward_f32",
            BackwardOp::Sigmoid => "nsl_sigmoid_backward_f32",
            BackwardOp::Tanh => "nsl_tanh_backward_f32",
            BackwardOp::Silu => "nsl_silu_backward_f32",
            BackwardOp::SigmoidSrcad => "nsl_sigmoid_backward_srcad_f32",
            BackwardOp::TanhSrcad => "nsl_tanh_backward_srcad_f32",
            BackwardOp::SiluSrcad => "nsl_silu_backward_srcad_f32",
            BackwardOp::GeluSrcad => "nsl_gelu_backward_srcad_f32",
            BackwardOp::SwigluGate => "nsl_swiglu_gate_backward_f32",
        }
    }

    /// The parameter names, in order. Pinned: the launchers pass arguments
    /// positionally, and the tape-AD sigmoid/tanh kernels call their saved
    /// operand `saved` (it is the forward's output, not its input).
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            BackwardOp::Sigmoid | BackwardOp::Tanh => &["grad", "saved", "out", "n"],
            BackwardOp::SwigluGate => &["grad", "up", "input", "out", "n"],
            _ => &["grad", "input", "out", "n"],
        }
    }
}

/// `s·(1 + x·(1 − s))` in explicitly rounded steps, `s` already computed:
/// the source-AD derivative tail shared by SiLU, GELU and the SwiGLU gate.
fn srcad_silu_tail(b: &mut KirBuilder, x: VarId, s: VarId) -> VarId {
    let one = f32_const(b, 1.0f32.to_bits());
    let t = f32_op2(b, KirOp::SubRn, one, s);
    let t = f32_op2(b, KirOp::MulRn, x, t);
    let one = f32_const(b, 1.0f32.to_bits());
    let t = f32_op2(b, KirOp::AddRn, one, t);
    f32_op2(b, KirOp::MulRn, s, t)
}

/// Build `op` as KIR, in the hand kernel's instruction order: the index and
/// bound, the loads in parameter order (the SwiGLU gate scales `grad` by
/// `up` before it loads `input`), the derivative, one store.
pub fn build_backward(op: BackwardOp) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(op.kernel_name());
    let names = op.param_names();
    let mut params = Vec::with_capacity(names.len());
    for &name in names {
        let ty = if name == "n" { KirType::U64 } else { f32_ptr() };
        params.push(b.add_param(name, ty, Global));
    }
    let param = |name: &str| params[names.iter().position(|n| *n == name).expect("a declared parameter")];
    let (i, _, exit) = index_and_bound(&mut b, param("n"));

    let mut g = load_f32(&mut b, param("grad"), i);
    if op == BackwardOp::SwigluGate {
        let up = load_f32(&mut b, param("up"), i);
        g = f32_op2(&mut b, KirOp::MulRn, g, up);
    }
    let saved = if matches!(op, BackwardOp::Sigmoid | BackwardOp::Tanh) { "saved" } else { "input" };
    let x = load_f32(&mut b, param(saved), i);

    let y = match op {
        BackwardOp::Relu => {
            let zero = f32_const(&mut b, 0);
            let pos = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Cmp(pos, x, zero, CmpOp::Gt));
            let zero = f32_const(&mut b, 0);
            let y = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Select(y, pos, g, zero));
            y
        }
        BackwardOp::Sigmoid => {
            let one = f32_const(&mut b, 1.0f32.to_bits());
            let t = f32_op2(&mut b, KirOp::Sub, one, x);
            let t = f32_op2(&mut b, KirOp::Mul, x, t);
            f32_op2(&mut b, KirOp::Mul, g, t)
        }
        BackwardOp::Tanh => {
            let t = f32_op2(&mut b, KirOp::Mul, x, x);
            let one = f32_const(&mut b, 1.0f32.to_bits());
            let t = f32_op2(&mut b, KirOp::Sub, one, t);
            f32_op2(&mut b, KirOp::Mul, g, t)
        }
        BackwardOp::Silu => {
            let s = sigmoid_of(&mut b, x);
            let one = f32_const(&mut b, 1.0f32.to_bits());
            let t = f32_op2(&mut b, KirOp::Sub, one, s);
            let t = f32_op2(&mut b, KirOp::Mul, x, t);
            let t = f32_op2(&mut b, KirOp::Mul, s, t);
            let t = f32_op2(&mut b, KirOp::Add, s, t);
            f32_op2(&mut b, KirOp::Mul, g, t)
        }
        BackwardOp::SigmoidSrcad => {
            let one = f32_const(&mut b, 1.0f32.to_bits());
            let t = f32_op2(&mut b, KirOp::SubRn, one, x);
            let t = f32_op2(&mut b, KirOp::MulRn, x, t);
            f32_op2(&mut b, KirOp::MulRn, g, t)
        }
        BackwardOp::TanhSrcad => {
            let t = f32_op2(&mut b, KirOp::MulRn, x, x);
            let one = f32_const(&mut b, 1.0f32.to_bits());
            let t = f32_op2(&mut b, KirOp::SubRn, one, t);
            f32_op2(&mut b, KirOp::MulRn, g, t)
        }
        BackwardOp::SiluSrcad | BackwardOp::SwigluGate => {
            let s = sigmoid_of(&mut b, x);
            let t = srcad_silu_tail(&mut b, x, s);
            f32_op2(&mut b, KirOp::MulRn, g, t)
        }
        BackwardOp::GeluSrcad => {
            let slope = f32_const(&mut b, GELU_SLOPE);
            let kx = f32_op2(&mut b, KirOp::MulRn, x, slope);
            let s = sigmoid_of(&mut b, kx);
            let t = srcad_silu_tail(&mut b, kx, s);
            f32_op2(&mut b, KirOp::MulRn, g, t)
        }
    };
    store_f32(&mut b, param("out"), i, y);
    finish(b, exit)
}

/// Build `op` and lower it to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn backward_ptx(op: BackwardOp) -> Vec<u8> {
    verified_ptx(build_backward(op))
}

/// The RoPE `rotate_half` pair over the last dimension: with `col = i %
/// last_dim` and `half = last_dim / 2`, element `i` takes the partner
/// `half` away (`i + half` in the first half, `i - half` in the second).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotateHalfOp {
    /// `nsl_rotate_half_f32`: `out[..h] = -in[h..]`, `out[h..] = in[..h]`.
    Plain,
    /// `nsl_rotate_half_neg_f32`, the fused RoPE backward
    /// `neg(rotate_half(x))`: `out[..h] = in[h..]`, `out[h..] = -in[..h]`.
    /// A negation is a sign-bit flip, so this is bit-identical to the
    /// two-launch composition it replaces.
    Neg,
}

impl RotateHalfOp {
    pub const ALL: [RotateHalfOp; 2] = [RotateHalfOp::Plain, RotateHalfOp::Neg];

    pub fn kernel_name(self) -> &'static str {
        match self {
            RotateHalfOp::Plain => "nsl_rotate_half_f32",
            RotateHalfOp::Neg => "nsl_rotate_half_neg_f32",
        }
    }
}

/// `nsl_rotate_half[_neg]_f32(a, c, n, last_dim, half)`, as the hand kernels
/// branch: `col < half` takes the first-half arm, anything else the second.
/// [`RotateHalfOp::Plain`] negates in the first arm, [`RotateHalfOp::Neg`]
/// in the second.
pub fn build_rotate_half(op: RotateHalfOp) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(op.kernel_name());
    let a = b.add_param("a", f32_ptr(), Global);
    let c = b.add_param("c", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let last_dim = b.add_param("last_dim", KirType::U64, Global);
    let half = b.add_param("half", KirType::U64, Global);
    let (i, _, exit) = index_and_bound(&mut b, n);

    let col = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Rem(col, i, last_dim));
    let first = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(first, col, half, CmpOp::Lt));
    let first_block = b.new_block();
    let second_block = b.new_block();
    b.terminate(KirTerminator::CondBranch(first, KirEdge::to(first_block), KirEdge::to(second_block)));

    for (block, from_second_half) in [(second_block, false), (first_block, true)] {
        b.set_block(block);
        let src = b.new_typed_var(KirType::U64);
        b.emit(if from_second_half { KirOp::Add(src, i, half) } else { KirOp::Sub(src, i, half) });
        let v = load_f32(&mut b, a, src);
        // Plain negates what the first half reads; Neg what the second does.
        let negate = from_second_half == (op == RotateHalfOp::Plain);
        let v = if negate { f32_op1(&mut b, KirOp::Neg, v) } else { v };
        store_f32(&mut b, c, i, v);
        b.terminate(KirTerminator::Branch(KirEdge::to(exit)));
    }

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([ELEMENTWISE_BLOCK, 1, 1]);
    b.set_launch_bounds(ELEMENTWISE_BLOCK, None);
    b.finalize()
}

/// [`build_rotate_half`] lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn rotate_half_ptx(op: RotateHalfOp) -> Vec<u8> {
    verified_ptx(build_rotate_half(op))
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

    /// The two kernels that must not be contracted spell their multiply and
    /// add `.rn` and nothing bare; Muon's division and square root are the
    /// IEEE forms.
    #[test]
    fn the_uncontracted_kernels_spell_rn() {
        for (ir, names) in [
            (build_scalar_mul_add_inplace(), ["m", "g", "s", "n"]),
            (build_muon_scale_inv_frob(), ["x", "c", "stats", "n"]),
        ] {
            verify(&ir).unwrap_or_else(|e| panic!("{}: {e:?}", ir.name));
            let params: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(params, names, "{}", ir.name);
        }
        let sma = String::from_utf8(scalar_mul_add_inplace_ptx()).unwrap();
        assert_eq!(sma.matches("mul.rn.f32 ").count(), 1, "{sma}");
        assert_eq!(sma.matches("add.rn.f32 ").count(), 1, "{sma}");
        assert!(!sma.contains("mul.f32 ") && !sma.contains("add.f32 ") && !sma.contains("fma"), "{sma}");
        let muon = String::from_utf8(muon_scale_inv_frob_ptx()).unwrap();
        for form in ["sqrt.rn.f32 ", "add.rn.f32 ", "div.rn.f32 ", "mul.rn.f32 ", "0f33D6BF95"] {
            assert_eq!(muon.matches(form).count(), 1, "{form}: {muon}");
        }
        assert!(!muon.contains("mul.f32 ") && !muon.contains("add.f32 ") && !muon.contains("fma"), "{muon}");
    }

    #[test]
    fn every_backward_kernel_verifies_and_keeps_its_signature() {
        for op in BackwardOp::ALL {
            let ir = build_backward(op);
            if let Err(errors) = verify(&ir) {
                panic!("{op:?} failed verification: {errors:?}");
            }
            assert_eq!(ir.name, op.kernel_name());
            let params: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(params, op.param_names(), "{op:?}");
            assert_eq!(ir.params.last().map(|p| p.ty.clone()), Some(KirType::U64), "{op:?}: n is u64");
            let bytes = backward_ptx(op);
            assert_eq!(bytes.last(), Some(&0), "{op:?}");
            assert!(bytes.is_ascii(), "{op:?}");
        }
    }

    #[test]
    fn the_rotate_half_kernels_verify_and_keep_their_signature() {
        for op in RotateHalfOp::ALL {
            let ir = build_rotate_half(op);
            if let Err(errors) = verify(&ir) {
                panic!("{op:?} failed verification: {errors:?}");
            }
            assert_eq!(ir.name, op.kernel_name());
            let params: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
            assert_eq!(params, ["a", "c", "n", "last_dim", "half"], "{op:?}");
            let text = String::from_utf8(rotate_half_ptx(op)).expect("ASCII");
            for mnemonic in ["rem.u64 ", "setp.lt.u64 ", "neg.f32 "] {
                assert_eq!(text.matches(mnemonic).count(), 1, "{op:?} {mnemonic}");
            }
        }
    }

    /// The tape-AD kernels are bare, the source-AD ones explicitly rounded
    /// in every derivative operation; GELU's backward carries the forward's
    /// slope.
    #[test]
    fn the_backward_kernels_spell_their_rounding() {
        let text = |op| String::from_utf8(backward_ptx(op)).unwrap();
        for op in [BackwardOp::Relu, BackwardOp::Sigmoid, BackwardOp::Tanh, BackwardOp::Silu] {
            assert!(!text(op).contains(".rn."), "{op:?}");
        }
        for (op, rounded) in [
            (BackwardOp::SigmoidSrcad, 3),
            (BackwardOp::TanhSrcad, 3),
            (BackwardOp::SiluSrcad, 5),
            (BackwardOp::GeluSrcad, 6),
            (BackwardOp::SwigluGate, 6),
        ] {
            let t = text(op);
            assert_eq!(t.matches(".rn.f32 ").count(), rounded, "{op:?}: {t}");
        }
        assert!(text(BackwardOp::GeluSrcad).contains(&format!("0f{GELU_SLOPE:08X}")));
        for op in BackwardOp::ALL {
            let t = text(op);
            assert!(!t.contains("div.") && !t.contains("fma."), "{op:?}");
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
