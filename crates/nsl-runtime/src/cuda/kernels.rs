//! Static PTX kernel strings for GPU elementwise operations.
//! All strings are null-terminated (end with `\0`) for CUDA driver API.

// PTX constants are loaded at runtime by name via the CUDA driver API;
// Rust's dead-code analysis cannot see these usages.

// --- Binary ops ---

// `nsl_add_f32`, `nsl_sub_f32` and `nsl_mul_f32` are built from KIR by
// `nsl_kir::kernels::elementwise` (roadmap A2 step 11); `nsl-codegen`'s
// `elementwise_binary_kir_equivalence` gate holds them to the hand-written
// modules they replace. Each module is built once, on first use, and kept:
// `kernel_launch` keys its module cache on the buffer's address.

use nsl_kir::kernels::elementwise::BinaryOp;

fn binary_module(op: BinaryOp) -> &'static str {
    static MODULES: std::sync::OnceLock<[String; 3]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        // The printer emits ASCII only, so this cannot fail for a kernel
        // `nsl-kir` builds.
        BinaryOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::binary_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = BinaryOp::ALL.iter().position(|o| *o == op).expect("every BinaryOp is in ALL");
    &modules[slot]
}

/// `nsl_add_f32`: `c[i] = a[i] + b[i]`, NUL-terminated.
pub(crate) fn add_f32_ptx() -> &'static str {
    binary_module(BinaryOp::Add)
}

/// `nsl_sub_f32`: `c[i] = a[i] - b[i]`, NUL-terminated.
pub(crate) fn sub_f32_ptx() -> &'static str {
    binary_module(BinaryOp::Sub)
}

/// `nsl_mul_f32`: `c[i] = a[i] * b[i]`, NUL-terminated.
pub(crate) fn mul_f32_ptx() -> &'static str {
    binary_module(BinaryOp::Mul)
}

// The scalar-operand family, `c[i] = a[i] op s`, likewise built by
// `nsl_kir::kernels::elementwise` (its `elementwise_scalar_kir_equivalence`
// gate). `nsl_div_scalar_f32` (`div.approx.f32`) stays hand-written below.

use nsl_kir::kernels::elementwise::ScalarOp;

pub(crate) fn scalar_module(op: ScalarOp) -> &'static str {
    static MODULES: std::sync::OnceLock<[String; 3]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        ScalarOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::scalar_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = ScalarOp::ALL.iter().position(|o| *o == op).expect("every ScalarOp is in ALL");
    &modules[slot]
}

/// `nsl_mul_scalar_f32`: `c[i] = a[i] * s`, NUL-terminated.
pub(crate) fn mul_scalar_f32_ptx() -> &'static str {
    scalar_module(ScalarOp::Mul)
}

/// `nsl_add_scalar_f32`: `c[i] = a[i] + s`, NUL-terminated.
pub(crate) fn add_scalar_f32_ptx() -> &'static str {
    scalar_module(ScalarOp::Add)
}

/// `nsl_sub_scalar_f32`: `c[i] = a[i] - s`, NUL-terminated.
pub(crate) fn sub_scalar_f32_ptx() -> &'static str {
    scalar_module(ScalarOp::Sub)
}

// Two kernels that must match a decomposed computation bit for bit, built by
// `nsl_kir::kernels::elementwise` with explicitly rounded arithmetic
// (`KirOp::{AddRn, MulRn}` print `.rn`, which ptxas never contracts into an
// `fma`); `elementwise_rn_kir_equivalence` holds them to the hand-written
// modules they replace.

/// `nsl_scalar_mul_add_inplace_f32(m, g, s, n)`: `m[i] = m[i] + g[i] * s`,
/// the FASE accumulate epilogue, bit-exact with `nsl_mul_scalar_f32` then
/// `nsl_add_f32`. NUL-terminated.
pub(crate) fn scalar_mul_add_inplace_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::elementwise::scalar_mul_add_inplace_ptx()).expect("PTX must be ASCII")
    })
}

/// `nsl_muon_scale_inv_frob_f32(x, c, stats, n)`:
/// `c[i] = x[i] * (1 / (sqrt(stats[3]) + 1e-7))`, with `stats` the output of
/// `nsl_tensor_stats_f32` read on the device. NUL-terminated.
pub(crate) fn muon_scale_inv_frob_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::elementwise::muon_scale_inv_frob_ptx()).expect("PTX must be ASCII")
    })
}

// The unary family, `c[i] = f(a[i])`, likewise built by
// `nsl_kir::kernels::elementwise` (its `elementwise_unary_kir_equivalence`
// gate). `nsl_tanh_f32` (`div.approx.f32`) stays hand-written below.

use nsl_kir::kernels::elementwise::UnaryOp;

pub(crate) fn unary_module(op: UnaryOp) -> &'static str {
    static MODULES: std::sync::OnceLock<[String; 13]> = std::sync::OnceLock::new();
    let modules = MODULES.get_or_init(|| {
        UnaryOp::ALL.map(|op| {
            String::from_utf8(nsl_kir::kernels::elementwise::unary_ptx(op)).expect("PTX must be ASCII")
        })
    });
    let slot = UnaryOp::ALL.iter().position(|o| *o == op).expect("every UnaryOp is in ALL");
    &modules[slot]
}

/// `nsl_neg_f32`: `c[i] = -a[i]`, NUL-terminated.
pub(crate) fn neg_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Neg)
}

/// `nsl_relu_f32`: `c[i] = max(a[i], 0)`, NUL-terminated.
pub(crate) fn relu_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Relu)
}

/// `nsl_exp_f32`: `c[i] = e^a[i] (ex2.approx)`, NUL-terminated.
pub(crate) fn exp_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Exp)
}

/// `nsl_log_f32`: `c[i] = ln a[i] (lg2.approx)`, NUL-terminated.
pub(crate) fn log_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Log)
}

/// `nsl_sqrt_f32`: `c[i] = sqrt(a[i]) (IEEE)`, NUL-terminated.
pub(crate) fn sqrt_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sqrt)
}

/// `nsl_abs_f32`: `c[i] = |a[i]|`, NUL-terminated.
pub(crate) fn abs_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Abs)
}

/// `nsl_sign_f32`: `c[i] = sign(a[i]) (0 for 0 and NaN)`, NUL-terminated.
pub(crate) fn sign_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sign)
}

/// `nsl_sigmoid_f32`: `c[i] = 1 / (1 + e^-a[i])`, NUL-terminated.
pub(crate) fn sigmoid_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sigmoid)
}

/// `nsl_sin_f32`: `c[i] = sin a[i] (sin.approx)`, NUL-terminated.
pub(crate) fn sin_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Sin)
}

/// `nsl_cos_f32`: `c[i] = cos a[i] (cos.approx)`, NUL-terminated.
pub(crate) fn cos_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Cos)
}

/// `nsl_silu_f32`: `c[i] = a[i] * sigmoid(a[i])`, NUL-terminated.
pub(crate) fn silu_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Silu)
}

/// `nsl_gelu_f32`: `c[i] = a[i] * sigmoid(1.702 * a[i])`, NUL-terminated. The
/// slope is `nsl_kir::kernels::elementwise::GELU_SLOPE`, the one
/// `GELU_BACKWARD_SRCAD_F32_PTX` differentiates with (`super::gelu_slope_drift`).
pub(crate) fn gelu_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Gelu)
}

/// `nsl_clamp_f32`: `c[i] = min(max(a[i], lo), hi)`, NUL-terminated.
pub(crate) fn clamp_f32_ptx() -> &'static str {
    unary_module(UnaryOp::Clamp)
}

// `nsl_div_f32` stays hand-written: it divides with `div.approx.f32`, and
// KIR's f32 division is `div.rn.f32`, so moving it would change what GPU
// division returns (see `nsl_kir::kernels::elementwise`).
pub(crate) const DIV_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_div_f32(\n\
    .param .u64 a, .param .u64 b, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<8>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [b];\n\
    ld.param.u64 %rd3, [c];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    div.approx.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

pub(crate) const ROTATE_HALF_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_rotate_half_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n, .param .u64 last_dim, .param .u64 half\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<14>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.u64 %rd3, [n];\n\
    ld.param.u64 %rd4, [last_dim];\n\
    ld.param.u64 %rd5, [half];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd3;\n\
    @%p1 bra DONE;\n\
    rem.u64 %rd7, %rd6, %rd4;\n\
    shl.b64 %rd8, %rd6, 2;\n\
    setp.lt.u64 %p2, %rd7, %rd5;\n\
    @%p2 bra FIRST_HALF;\n\
    sub.u64 %rd9, %rd6, %rd5;\n\
    shl.b64 %rd10, %rd9, 2;\n\
    add.u64 %rd11, %rd1, %rd10;\n\
    ld.global.f32 %fs1, [%rd11];\n\
    add.u64 %rd12, %rd2, %rd8;\n\
    st.global.f32 [%rd12], %fs1;\n\
    bra DONE;\n\
FIRST_HALF:\n\
    add.u64 %rd9, %rd6, %rd5;\n\
    shl.b64 %rd10, %rd9, 2;\n\
    add.u64 %rd11, %rd1, %rd10;\n\
    ld.global.f32 %fs1, [%rd11];\n\
    neg.f32 %fs2, %fs1;\n\
    add.u64 %rd12, %rd2, %rd8;\n\
    st.global.f32 [%rd12], %fs2;\n\
DONE: ret;\n\
}\0";

// Fused RoPE backward `neg(rotate_half(x))` (mfu-fusion C2). Clone of
// ROTATE_HALF_F32_PTX with the negation MOVED to match the composition
// being replaced, not the textbook rotate_half formula:
//   rotate_half:  out[..h] = -in[h..],  out[h..] =  in[..h]   (neg on the
//                 FIRST-half branch above)
//   then neg:     out[..h] =  in[h..],  out[h..] = -in[..h]
// So this kernel stores the FIRST-half outputs raw (the two negations
// cancel) and negates the SECOND-half outputs. f32 negation is a pure
// sign-bit flip, so one flip here is bit-identical to running
// nsl_rotate_half_f32 then nsl_neg_f32 (which nsl_tensor_rotate_half's CPU
// arm plus nsl_tensor_neg also compose to, elementwise).
pub(crate) const ROTATE_HALF_NEG_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_rotate_half_neg_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n, .param .u64 last_dim, .param .u64 half\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<14>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.u64 %rd3, [n];\n\
    ld.param.u64 %rd4, [last_dim];\n\
    ld.param.u64 %rd5, [half];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd3;\n\
    @%p1 bra DONE;\n\
    rem.u64 %rd7, %rd6, %rd4;\n\
    shl.b64 %rd8, %rd6, 2;\n\
    setp.lt.u64 %p2, %rd7, %rd5;\n\
    @%p2 bra FIRST_HALF;\n\
    sub.u64 %rd9, %rd6, %rd5;\n\
    shl.b64 %rd10, %rd9, 2;\n\
    add.u64 %rd11, %rd1, %rd10;\n\
    ld.global.f32 %fs1, [%rd11];\n\
    neg.f32 %fs2, %fs1;\n\
    add.u64 %rd12, %rd2, %rd8;\n\
    st.global.f32 [%rd12], %fs2;\n\
    bra DONE;\n\
FIRST_HALF:\n\
    add.u64 %rd9, %rd6, %rd5;\n\
    shl.b64 %rd10, %rd9, 2;\n\
    add.u64 %rd11, %rd1, %rd10;\n\
    ld.global.f32 %fs1, [%rd11];\n\
    add.u64 %rd12, %rd2, %rd8;\n\
    st.global.f32 [%rd12], %fs1;\n\
DONE: ret;\n\
}\0";




// --- Matrix multiplication ---
//
// The f32 single-matmul PTX kernel (`nsl_matmul_f32`) was deleted 2026-04-21
// as part of the cuBLAS swap (spec docs/superpowers/specs/2026-04-21-matmul-
// cublas-swap-design.md). f32 single matmul now dispatches to cuBLAS sgemm
// via `cuda::cublas_inner::sgemm_row_major` — see `cuda::gpu_matmul_f32`.
// The batched f32 path (`BMM_F32_PTX`/`nsl_bmm_f32`) is out of scope per
// spec §6 and remains unchanged.

// Scalar-RHS div (mfu-fusion C3 scalar sweep): out[i] = a[i] / s, with s a
// .f32 kernel param so ONE kernel serves every immediate value. Replaces the
// decomposed Div(x, Constant) chain: scalar CPU tensor + synchronous HtoD +
// full-size broadcast materialize + nsl_div_f32.
//
// ARITHMETIC INSTRUCTION IS LOAD-BEARING: `div.approx.f32`, copied verbatim
// from DIV_F32_PTX's nsl_div_f32 (NOT the correctly-rounded `div.rn.f32`) —
// bit-exactness with the decomposed baseline requires the IDENTICAL opcode,
// and the baseline kernel this replaces divides with div.approx. A
// single-instruction kernel has no mul+add pair, so FMA contraction (the
// usual reason for .rn) cannot arise here.
pub(crate) const DIV_SCALAR_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_div_scalar_f32(\n\
    .param .u64 a, .param .u64 c, .param .f32 s, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.f32 %fs2, [s];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd1, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    div.approx.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";


// --- Backward kernels for activation functions ---

/// relu_backward: out[i] = input[i] > 0 ? grad[i] : 0
pub(crate) const RELU_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_relu_backward_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p<2>;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    mov.f32 %fs3, 0f00000000;\n\
    setp.gt.f32 %p1, %fs2, %fs3;\n\
    selp.f32 %fs3, %fs1, 0f00000000, %p1;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

/// sigmoid_backward: out[i] = grad[i] * saved[i] * (1 - saved[i])
/// saved[i] is the sigmoid output
pub(crate) const SIGMOID_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sigmoid_backward_f32(\n\
    .param .u64 grad, .param .u64 saved, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [saved];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    sub.f32 %fs3, 0f3F800000, %fs2;\n\
    mul.f32 %fs3, %fs2, %fs3;\n\
    mul.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

/// tanh_backward: out[i] = grad[i] * (1 - saved[i] * saved[i])
/// saved[i] is the tanh output
pub(crate) const TANH_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_tanh_backward_f32(\n\
    .param .u64 grad, .param .u64 saved, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [saved];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    mul.f32 %fs3, %fs2, %fs2;\n\
    sub.f32 %fs3, 0f3F800000, %fs3;\n\
    mul.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

/// gelu_backward using tanh approximation derivative
/// k = 0.0356774*x^3 + 0.797885*x
/// sech2 = 1 - tanh(k)^2
/// out[i] = grad[i] * 0.5 * (1 + tanh(k) + x * sech2 * (0.107032*x + 0.797885))
pub(crate) const GELU_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_gelu_backward_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<12>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    mul.f32 %fs3, %fs2, %fs2;\n\
    mul.f32 %fs3, %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3D124925;\n\
    mul.f32 %fs4, %fs2, 0f3F4C422A;\n\
    add.f32 %fs3, %fs3, %fs4;\n\
    add.f32 %fs4, %fs3, %fs3;\n\
    mul.f32 %fs4, %fs4, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs4, %fs4;\n\
    add.f32 %fs5, %fs4, 0f3F800000;\n\
    sub.f32 %fs4, %fs4, 0f3F800000;\n\
    div.approx.f32 %fs6, %fs4, %fs5;\n\
    mul.f32 %fs7, %fs6, %fs6;\n\
    sub.f32 %fs7, 0f3F800000, %fs7;\n\
    mul.f32 %fs8, %fs2, %fs2;\n\
    mul.f32 %fs8, %fs8, 0f3DD8ECA1;\n\
    add.f32 %fs8, %fs8, 0f3F4C422A;\n\
    mul.f32 %fs8, %fs2, %fs8;\n\
    mul.f32 %fs8, %fs7, %fs8;\n\
    add.f32 %fs8, %fs6, %fs8;\n\
    add.f32 %fs8, 0f3F800000, %fs8;\n\
    mul.f32 %fs8, 0f3F000000, %fs8;\n\
    mul.f32 %fs8, %fs1, %fs8;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs8;\n\
DONE: ret;\n\
}\0";

/// silu_backward: sig = 1/(1+exp(-x)); out[i] = grad[i] * (sig + x*sig*(1-sig))
pub(crate) const SILU_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_silu_backward_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<8>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.f32 %fs4, %fs2, %fs4;\n\
    mul.f32 %fs4, %fs3, %fs4;\n\
    add.f32 %fs4, %fs3, %fs4;\n\
    mul.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs4;\n\
DONE: ret;\n\
}\0";

// Source-AD SiLU backward, fused (Milestone C · p4 slice 2). Collapses the 6
// separate adjoint kernels source-AD emits for `SiluBackward` — Sigmoid, Sub,
// Mul, Add, Mul, Mul — into ONE launch, and is BIT-EXACT with them.
//
// Computes, per element, in source-AD's exact operation order:
//   s  = sigmoid(a)                 (identical instructions to nsl_sigmoid_f32)
//   t1 = 1.0 - s
//   t2 = a * t1
//   t3 = 1.0 + t2
//   t4 = s * t3
//   out = grad * t4     => grad * s*(1 + a*(1-s))
//
// This differs from `SILU_BACKWARD_F32_PTX` above (the tape-AD kernel) only in
// operation ORDER: that one computes grad*(s + s*a*(1-s)), which is the same
// value but rounds differently. Matching source-AD's order is what makes this
// byte-identical to the decomposed path it replaces.
//
// LOAD-BEARING `.rn` on the derivative ops (`t2`→`t3` is a mul feeding an add):
// ptxas would otherwise contract the register-dependent mul+add into a single
// `fma` (one rounding) and diverge by ~1 ULP. `.rn` (round-to-nearest, already
// the default) forbids contraction so each op rounds independently — exactly
// like the 6 separate kernels, whose intermediates round to f32 through memory.
// The sigmoid ops stay plain `.f32` to match nsl_sigmoid_f32 byte-for-byte (they
// contain no contractible mul+add pair — ex2.approx/rcp.approx break the chain).
// P5 item 20 slice B — fused SwiGLU GATE backward. For f = silu(g) * u the
// adjoint pair is  t = dy * u  (Mul) followed by silu_backward(t, g); this
// kernel folds the Mul into the silu-backward launch. BIT-EXACT with the
// two-kernel path: t = mul.rn(dy, u) exactly as the standalone Mul kernel
// rounds it, then the identical SILU_BACKWARD_SRCAD instruction sequence
// (sigmoid via ex2/rcp.approx, then .rn ops that forbid fma-contraction).
pub(crate) const SWIGLU_GATE_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_swiglu_gate_backward_f32(\n\
    .param .u64 grad, .param .u64 up, .param .u64 input,\n\
    .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<10>;\n\
    .reg .f32 %fs<6>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [up];\n\
    ld.param.u64 %rd3, [input];\n\
    ld.param.u64 %rd4, [out];\n\
    ld.param.u64 %rd5, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %fs1, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs5, [%rd8];\n\
    mul.rn.f32 %fs1, %fs1, %fs5;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs2, [%rd8];\n\
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.rn.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs4, %fs2, %fs4;\n\
    add.rn.f32 %fs4, 0f3F800000, %fs4;\n\
    mul.rn.f32 %fs4, %fs3, %fs4;\n\
    mul.rn.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd9, %rd4, %rd7;\n\
    st.global.f32 [%rd9], %fs4;\n\
DONE: ret;\n\
}\0";

pub(crate) const SILU_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_silu_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.rn.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs4, %fs2, %fs4;\n\
    add.rn.f32 %fs4, 0f3F800000, %fs4;\n\
    mul.rn.f32 %fs4, %fs3, %fs4;\n\
    mul.rn.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs4;\n\
DONE: ret;\n\
}\0";

// Source-AD SIGMOID backward (Milestone C · p4 slice 3). Fuses the three
// adjoint kernels source-AD emits for `SigmoidBackward` — Sub, Mul, Mul — into
// ONE launch, and is BIT-EXACT with them. Input `y` is the sigmoid OUTPUT (the
// adjoint carries `op.result`, not the pre-activation).
//
// Computes, per element, in source-AD's exact operation order:
//   t1 = 1.0 - y
//   t2 = y * t1
//   out = grad * t2     => grad * y*(1 - y)
//
// `.rn` on every op (round-to-nearest, already the default) forbids ptxas from
// contracting any register-dependent mul+add into a single `fma` — each op
// rounds independently, exactly like the three separate kernels whose
// intermediates round to f32 through global memory. (Sigmoid backward has no
// mul-feeding-add pair, but `.rn` documents intent and is zero-cost.)
pub(crate) const SIGMOID_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sigmoid_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    sub.rn.f32 %fs3, 0f3F800000, %fs2;\n\
    mul.rn.f32 %fs3, %fs2, %fs3;\n\
    mul.rn.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

// Source-AD TANH backward (Milestone C · p4 slice 3). Fuses the three adjoint
// kernels source-AD emits for `TanhBackward` — Mul, Sub, Mul — into ONE launch,
// and is BIT-EXACT with them. Input `y` is the tanh OUTPUT.
//
// Computes, per element, in source-AD's exact operation order:
//   t1 = y * y
//   t2 = 1.0 - t1
//   out = grad * t2     => grad * (1 - y*y)
//
// LOAD-BEARING `.rn`: `t1 = y*y` feeds `t2 = 1.0 - t1`, a mul-feeding-sub that
// ptxas would otherwise contract into a single-rounding `fma(-y, y, 1.0)` (~1
// ULP drift). `.rn` on the mul forces `y*y` to round to f32 first — exactly as
// the decomposed path does when it stores `y_sq` to global memory before the
// subtract — and `.rn` on the sub keeps them separate.
pub(crate) const TANH_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_tanh_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    mul.rn.f32 %fs3, %fs2, %fs2;\n\
    sub.rn.f32 %fs3, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

// Source-AD GELU backward (Milestone C · p4 GELU fix). One launch computing the
// EXACT derivative of the GPU forward `nsl_gelu_f32` (gelu(x) = x·σ(1.702x),
// sigmoid approximation):
//
//   kx  = 1.702 * x                 (0f3FD9DB23 = 1.702f, matches nsl_tensor_scalar(1.702,1))
//   s   = σ(kx)                     (identical instructions to nsl_sigmoid_f32)
//   out = grad * s*(1 + kx*(1-s))
//
// This REPLACES the source-AD 7-op expansion of `AdjointExpr::GeluBackward`,
// which was numerically WRONG: its internal temp `kx` (refcount 1) was
// FBIP-mutated in place by the expansion's own `Sigmoid(kx)` during the adjoint
// pass (where the in-place-suppression guard is deliberately clear), so the
// later `Mul(kx, 1-s)` read σ(kx) and the whole thing computed s·(1+s·(1-s)).
// Fusing eliminates the temp — no aliasing is possible inside one kernel.
//
// NOTE the CPU forward gelu uses the TANH approximation, so the CPU path of
// `nsl_tensor_gelu_backward` computes the tanh-approx derivative instead — each
// device gets the derivative of the forward it actually ran. `.rn` on the
// derivative ops blocks ptxas fma-contraction (family convention; the kx·(1-s)
// mul feeding the 1+· add is a contractible pair).
pub(crate) const GELU_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_gelu_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    mul.rn.f32 %fs2, %fs2, 0f3FD9DB23;\n\
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.rn.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs4, %fs2, %fs4;\n\
    add.rn.f32 %fs4, 0f3F800000, %fs4;\n\
    mul.rn.f32 %fs4, %fs3, %fs4;\n\
    mul.rn.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs4;\n\
DONE: ret;\n\
}\0";

// Fused per-parameter FASE-Deferred AdamW/Adam optimizer step (Milestone C ·
// p9). ONE launch replacing the ~15-launch interpreted `UpdateProgram`
// (`fase_emit_final_step`), BIT-EXACT with it. Per element, in the program's
// exact rounding order (each `rn` mirrors one decomposed kernel's store):
//
//   m'  = rn(rn(m·β₁)   + rn(mp·(1-β₁)))          [ScalarMulAdd]
//   v'  = rn(rn(v·β₂)   + rn((1-β₂)·rn(mp·mp)))   [SquaredAccumulate]
//   m̂   = rn(m'·bc1inv);  v̂ = rn(v'·bc2inv)       [ScalarMulByBc ×2]
//   t   = rn(sqrt.rn(v̂) + ε)                      [SqrtPlusEps; the +0.0 copy
//                                                   is identity for v̂ ≥ +0]
//   u   = div.approx(m̂, t)                        [Div — matches DIV_F32_PTX's
//                                                   div.approx.f32 exactly]
//   adj = rn(u·(-lr)); if wd≠0: adj = rn(adj + rn(θ·(-lr·wd)))   [Update]
//   θ'  = rn(θ + adj)
//
// `.rn` everywhere (except the deliberate div.approx and sqrt.rn, which match
// the decomposed kernels' instructions) forbids ptxas fma-contraction so each
// op rounds independently — exactly like the decomposed path whose
// intermediates round to f32 through global memory. Scalars arrive already
// converted f64→f32 by the FFI, the same `as f32` conversion every
// nsl_tensor_*_scalar op performs at its launch boundary. m_partial is only
// READ (the zero-for-next-window stays with the existing per-param emission).
pub(crate) const FASE_FUSED_ADAMW_STEP_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_fase_fused_adamw_step_f32(\n\
    .param .u64 theta, .param .u64 m, .param .u64 v, .param .u64 mp, .param .u64 n,\n\
    .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,\n\
    .param .f32 eps, .param .f32 neg_lr, .param .f32 neg_lr_wd,\n\
    .param .f32 bc1, .param .f32 bc2, .param .u32 has_wd\n\
) {\n\
    .reg .u32 %r<6>;\n\
    .reg .u64 %rd<10>;\n\
    .reg .f32 %fs<16>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [theta];\n\
    ld.param.u64 %rd2, [m];\n\
    ld.param.u64 %rd3, [v];\n\
    ld.param.u64 %rd4, [mp];\n\
    ld.param.u64 %rd5, [n];\n\
    ld.param.f32 %fs1, [b1];\n\
    ld.param.f32 %fs2, [omb1];\n\
    ld.param.f32 %fs3, [b2];\n\
    ld.param.f32 %fs4, [omb2];\n\
    ld.param.f32 %fs5, [eps];\n\
    ld.param.f32 %fs6, [neg_lr];\n\
    ld.param.f32 %fs7, [neg_lr_wd];\n\
    ld.param.f32 %fs8, [bc1];\n\
    ld.param.f32 %fs9, [bc2];\n\
    ld.param.u32 %r4, [has_wd];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %fs10, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs11, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs12, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs13, [%rd8];\n\
    mul.rn.f32 %fs14, %fs11, %fs1;\n\
    mul.rn.f32 %fs15, %fs13, %fs2;\n\
    add.rn.f32 %fs11, %fs14, %fs15;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    st.global.f32 [%rd8], %fs11;\n\
    mul.rn.f32 %fs14, %fs13, %fs13;\n\
    mul.rn.f32 %fs14, %fs14, %fs4;\n\
    mul.rn.f32 %fs12, %fs12, %fs3;\n\
    add.rn.f32 %fs12, %fs12, %fs14;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    st.global.f32 [%rd8], %fs12;\n\
    mul.rn.f32 %fs14, %fs11, %fs8;\n\
    mul.rn.f32 %fs15, %fs12, %fs9;\n\
    sqrt.rn.f32 %fs15, %fs15;\n\
    add.rn.f32 %fs15, %fs15, %fs5;\n\
    div.approx.f32 %fs14, %fs14, %fs15;\n\
    mul.rn.f32 %fs14, %fs14, %fs6;\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra SKIPWD;\n\
    mul.rn.f32 %fs15, %fs10, %fs7;\n\
    add.rn.f32 %fs14, %fs14, %fs15;\n\
SKIPWD:\n\
    add.rn.f32 %fs10, %fs10, %fs14;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    st.global.f32 [%rd8], %fs10;\n\
DONE: ret;\n\
}\0";

// Fusion-queue item 1: MULTI-TENSOR fused AdamW step. One launch updates
// every parameter, with base pointers read from device pointer tables and
// per-parameter length from ntab. The arithmetic body is byte-for-byte the
// FASE_FUSED_ADAMW_STEP_F32_PTX sequence (same roundings, same div.approx),
// so per-element results are BIT-IDENTICAL to the per-param launches. The
// shared tail's m_partial zeroing is folded in (store 0 after the read) —
// value-identical to the separate nsl_tensor_zero_inplace pass it replaces.
//
// (Roadmap item 8 replaced the original `grid.y = parameter index` mapping;
// the doc comment below describes what it does now. This paragraph used to
// end "blocks past a shorter param's end exit immediately", which was the
// waste item 8 removed.)
/// Multi-parameter fused AdamW, FLAT grid (roadmap item 8).
///
/// Was a rectangular grid: `(ceil(max_n/256), k, 1)`, i.e. every parameter got
/// enough blocks for the LARGEST parameter and the surplus exited on the
/// bounds guard. Measured on Coder-50M's real parameter list (74 params,
/// 38.5M elements, largest 16.4M): 4,736,000 blocks launched for 150,562
/// blocks of work — 96.8% exited immediately, a 31.5x overshoot.
///
/// Now the host precomputes two u32 tables indexed by BLOCK id — `bptab[b]`
/// is the parameter that block `b` works on, `bbtab[b]` is the element offset
/// of that block's first thread within the parameter — so the grid is
/// `(sum(ceil(n_i/256)), 1, 1)` with no wasted blocks and no binary search in
/// the kernel. The tables depend only on the shape list, so they are built
/// once and cached until it changes.
///
/// Dropping grid.y also removes the 65535-parameter cap.
///
/// CONTRACT: the kernel no longer reads `%ntid.x`, so `blockDim.x` at launch
/// MUST equal the `block` argument `build_block_tables` was called with. Both
/// come from the single `let block = 256i64` in
/// `gpu_fase_fused_adamw_step_multi`, so they cannot currently desync — but if
/// they ever did, a smaller `blockDim` silently skips elements (weights stop
/// updating) and a larger one applies AdamW twice to the overlap. Neither is
/// caught by any test, because no test can vary the two independently.
///
/// `mp_scale` folds the two-phase-clip Phase B pre-scale
/// (`nsl_tensor_mul_scalar_inplace(m_partial, clip_factor)`) into the read:
/// `g = rn(mp[i] * mp_scale)` — the same single f32 rounding the in-place
/// scale-then-read performs, so the clip path stays bit-identical to the
/// per-param loop it replaces. `mp_scale == 1.0` branches AROUND the
/// multiply so the non-clip path keeps exact bit-identity (a `mul.rn` by
/// 1.0 would canonicalize NaN payloads, which "bit-identical" must not).
pub(crate) const FASE_FUSED_ADAMW_MULTI_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_fase_fused_adamw_multi_f32(\n\
    .param .u64 ttab, .param .u64 mtab, .param .u64 vtab, .param .u64 mptab,\n\
    .param .u64 ntab,\n\
    .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,\n\
    .param .f32 eps, .param .f32 neg_lr, .param .f32 neg_lr_wd,\n\
    .param .f32 bc1, .param .f32 bc2, .param .u32 has_wd,\n\
    .param .u64 bptab, .param .u64 bbtab, .param .f32 mp_scale\n\
) {\n\
    .reg .u32 %r<10>;\n\
    .reg .u64 %rd<12>;\n\
    .reg .f32 %fs<17>;\n\
    .reg .pred %p<4>;\n\
    ld.param.u64 %rd1, [ttab];\n\
    ld.param.u64 %rd2, [mtab];\n\
    ld.param.u64 %rd3, [vtab];\n\
    ld.param.u64 %rd4, [mptab];\n\
    ld.param.u64 %rd5, [ntab];\n\
    ld.param.f32 %fs1, [b1];\n\
    ld.param.f32 %fs2, [omb1];\n\
    ld.param.f32 %fs3, [b2];\n\
    ld.param.f32 %fs4, [omb2];\n\
    ld.param.f32 %fs5, [eps];\n\
    ld.param.f32 %fs6, [neg_lr];\n\
    ld.param.f32 %fs7, [neg_lr_wd];\n\
    ld.param.f32 %fs8, [bc1];\n\
    ld.param.f32 %fs9, [bc2];\n\
    ld.param.u32 %r4, [has_wd];\n\
    ld.param.u64 %rd10, [bptab];\n\
    ld.param.u64 %rd11, [bbtab];\n\
    ld.param.f32 %fs16, [mp_scale];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd10, %rd7;\n\
    ld.global.u32 %r5, [%rd8];\n\
    add.u64 %rd8, %rd11, %rd7;\n\
    ld.global.u32 %r7, [%rd8];\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r7, %r1;\n\
    cvt.u64.u32 %rd6, %r5;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd5, %rd7;\n\
    ld.global.u32 %r6, [%rd8];\n\
    setp.ge.u32 %p1, %r3, %r6;\n\
    @%p1 bra MDONE;\n\
    shl.b64 %rd7, %rd6, 3;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.u64 %rd1, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.u64 %rd2, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.u64 %rd3, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.u64 %rd4, [%rd8];\n\
    cvt.u64.u32 %rd6, %r3;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %fs10, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs11, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs12, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs13, [%rd8];\n\
    setp.eq.f32 %p3, %fs16, 0f3F800000;\n\
    @%p3 bra MNOSC;\n\
    mul.rn.f32 %fs13, %fs13, %fs16;\n\
MNOSC:\n\
    mul.rn.f32 %fs14, %fs11, %fs1;\n\
    mul.rn.f32 %fs15, %fs13, %fs2;\n\
    add.rn.f32 %fs11, %fs14, %fs15;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    st.global.f32 [%rd8], %fs11;\n\
    mul.rn.f32 %fs14, %fs13, %fs13;\n\
    mul.rn.f32 %fs14, %fs14, %fs4;\n\
    mul.rn.f32 %fs12, %fs12, %fs3;\n\
    add.rn.f32 %fs12, %fs12, %fs14;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    st.global.f32 [%rd8], %fs12;\n\
    mul.rn.f32 %fs14, %fs11, %fs8;\n\
    mul.rn.f32 %fs15, %fs12, %fs9;\n\
    sqrt.rn.f32 %fs15, %fs15;\n\
    add.rn.f32 %fs15, %fs15, %fs5;\n\
    div.approx.f32 %fs14, %fs14, %fs15;\n\
    mul.rn.f32 %fs14, %fs14, %fs6;\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra MSKIPWD;\n\
    mul.rn.f32 %fs15, %fs10, %fs7;\n\
    add.rn.f32 %fs14, %fs14, %fs15;\n\
MSKIPWD:\n\
    add.rn.f32 %fs10, %fs10, %fs14;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    st.global.f32 [%rd8], %fs10;\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    mov.f32 %fs14, 0f00000000;\n\
    st.global.f32 [%rd8], %fs14;\n\
MDONE: ret;\n\
}\0";

/// Roadmap item 8, bf16-SR arm: MULTI-parameter fused SR-BF16 AdamW step on
/// the SAME flat grid as `nsl_fase_fused_adamw_multi_f32` (bptab/bbtab block
/// tables, per-param base pointers from device tables), with the SR-BF16
/// arithmetic body and rounding tail of `nsl_fase_fused_adamw_step_bf16sr`.
///
/// BIT-IDENTITY to the per-param SR loop it replaces holds because the SR
/// dither is a pure function of (sr_key, ctrtab[p] + element) — the same
/// counters the per-param launches use (`param_idx << SR_PARAM_SHIFT`, set
/// at registration) — so the draw for every (param, element, step) is
/// independent of launch shape. The arithmetic body is byte-for-byte the
/// per-param kernel's sequence.
///
/// Contract deltas vs the f32 multi kernel, both deliberate:
///   - NO `mp_scale`: the per-param SR entry has no clip fold (SR composes
///     with FASE-Deferred accumulation; the clip path is refused upstream),
///     and adding one here would create an arm the per-param path cannot
///     mirror.
///   - NO in-kernel m_partial zero: the per-param SR kernel leaves mp to the
///     FASE-Deferred lifecycle; mirroring that keeps the replacement
///     observationally identical.
///
/// `ntab` is u32 — enforced at the batching entry itself
/// (`bf16sr_multi_impl`'s `u32::try_from(len)` aborts on overflow;
/// registration only bounds len below 2^40 for the counter scheme). The f32
/// multi demotes oversize params to its sequential arm instead; both are
/// unreachable on hardware where a >4G-element param's moments alone
/// exceed VRAM. `ctrtab` is u64 per-param SR counter bases. Same
/// `blockDim.x == build_block_tables block` contract as the f32 multi
/// kernel — one constant feeds both.
pub(crate) const FASE_FUSED_ADAMW_MULTI_BF16SR_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_fase_fused_adamw_multi_bf16sr(\n\
    .param .u64 ttab, .param .u64 mtab, .param .u64 vtab, .param .u64 mptab,\n\
    .param .u64 ntab,\n\
    .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,\n\
    .param .f32 eps, .param .f32 neg_lr, .param .f32 neg_lr_wd,\n\
    .param .f32 bc1, .param .f32 bc2, .param .u32 has_wd,\n\
    .param .u64 sr_key, .param .u64 ctrtab,\n\
    .param .u64 bptab, .param .u64 bbtab\n\
) {\n\
    .reg .u32 %r<12>;\n\
    .reg .u64 %rd<18>;\n\
    .reg .f32 %fs<16>;\n\
    .reg .b16 %rs<3>;\n\
    .reg .pred %p<6>;\n\
    ld.param.u64 %rd1, [ttab];\n\
    ld.param.u64 %rd2, [mtab];\n\
    ld.param.u64 %rd3, [vtab];\n\
    ld.param.u64 %rd4, [mptab];\n\
    ld.param.u64 %rd5, [ntab];\n\
    ld.param.f32 %fs1, [b1];\n\
    ld.param.f32 %fs2, [omb1];\n\
    ld.param.f32 %fs3, [b2];\n\
    ld.param.f32 %fs4, [omb2];\n\
    ld.param.f32 %fs5, [eps];\n\
    ld.param.f32 %fs6, [neg_lr];\n\
    ld.param.f32 %fs7, [neg_lr_wd];\n\
    ld.param.f32 %fs8, [bc1];\n\
    ld.param.f32 %fs9, [bc2];\n\
    ld.param.u32 %r4, [has_wd];\n\
    ld.param.u64 %rd15, [bptab];\n\
    ld.param.u64 %rd16, [bbtab];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd15, %rd7;\n\
    ld.global.u32 %r5, [%rd8];\n\
    add.u64 %rd8, %rd16, %rd7;\n\
    ld.global.u32 %r7, [%rd8];\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r7, %r1;\n\
    cvt.u64.u32 %rd6, %r5;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd5, %rd7;\n\
    ld.global.u32 %r6, [%rd8];\n\
    setp.ge.u32 %p1, %r3, %r6;\n\
    @%p1 bra MSDONE;\n\
    shl.b64 %rd7, %rd6, 3;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.u64 %rd1, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.u64 %rd2, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.u64 %rd3, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.u64 %rd4, [%rd8];\n\
    ld.param.u64 %rd10, [ctrtab];\n\
    add.u64 %rd8, %rd10, %rd7;\n\
    ld.global.u64 %rd11, [%rd8];\n\
    cvt.u64.u32 %rd6, %r3;\n\
    add.u64 %rd11, %rd11, %rd6;\n\
    // theta: bf16 load, widen exactly (bits<<16)\n\
    shl.b64 %rd9, %rd6, 1;\n\
    add.u64 %rd8, %rd1, %rd9;\n\
    ld.global.u16 %rs1, [%rd8];\n\
    cvt.u32.u16 %r5, %rs1;\n\
    shl.b32 %r5, %r5, 16;\n\
    mov.b32 %fs10, %r5;\n\
    // m/v/mp: f32 buffers, byte offset = idx*4\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs11, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs12, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs13, [%rd8];\n\
    mul.rn.f32 %fs14, %fs11, %fs1;\n\
    mul.rn.f32 %fs15, %fs13, %fs2;\n\
    add.rn.f32 %fs11, %fs14, %fs15;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    st.global.f32 [%rd8], %fs11;\n\
    mul.rn.f32 %fs14, %fs13, %fs13;\n\
    mul.rn.f32 %fs14, %fs14, %fs4;\n\
    mul.rn.f32 %fs12, %fs12, %fs3;\n\
    add.rn.f32 %fs12, %fs12, %fs14;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    st.global.f32 [%rd8], %fs12;\n\
    mul.rn.f32 %fs14, %fs11, %fs8;\n\
    mul.rn.f32 %fs15, %fs12, %fs9;\n\
    sqrt.rn.f32 %fs15, %fs15;\n\
    add.rn.f32 %fs15, %fs15, %fs5;\n\
    div.approx.f32 %fs14, %fs14, %fs15;\n\
    mul.rn.f32 %fs14, %fs14, %fs6;\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra MSSKIPWD;\n\
    mul.rn.f32 %fs15, %fs10, %fs7;\n\
    add.rn.f32 %fs14, %fs14, %fs15;\n\
MSSKIPWD:\n\
    add.rn.f32 %fs10, %fs10, %fs14;\n\
    // ---- SR tail: theta_new (f32 in %fs10) -> bf16 bits in %r8 ----\n\
    mov.b32 %r5, %fs10;\n\
    and.b32 %r9, %r5, 0x80000000;\n\
    shr.u32 %r9, %r9, 16;\n\
    and.b32 %r6, %r5, 0x7f800000;\n\
    setp.eq.u32 %p3, %r6, 0x7f800000;\n\
    @%p3 bra MSSPECIAL;\n\
    // dither = low 16 bits of splitmix64(sr_key, ctrtab[p] + idx)\n\
    ld.param.u64 %rd10, [sr_key];\n\
    mov.u64 %rd12, 0x9E3779B97F4A7C15;\n\
    mul.lo.u64 %rd13, %rd11, %rd12;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shr.u64 %rd14, %rd13, 30;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0xBF58476D1CE4E5B9;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 27;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0x94D049BB133111EB;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 31;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    cvt.u32.u64 %r7, %rd13;\n\
    and.b32 %r7, %r7, 0xffff;\n\
    add.u32 %r6, %r5, %r7;\n\
    and.b32 %r7, %r6, 0x7f800000;\n\
    setp.eq.u32 %p4, %r7, 0x7f800000;\n\
    @%p4 bra MSSATURATE;\n\
    shr.u32 %r8, %r6, 16;\n\
    bra MSSTORE;\n\
MSSATURATE:\n\
    or.b32 %r8, %r9, 0x7f7f;\n\
    bra MSSTORE;\n\
MSSPECIAL:\n\
    and.b32 %r6, %r5, 0x007fffff;\n\
    setp.ne.u32 %p4, %r6, 0;\n\
    @%p4 bra MSQNAN;\n\
    or.b32 %r8, %r9, 0x7f80;\n\
    bra MSSTORE;\n\
MSQNAN:\n\
    or.b32 %r8, %r9, 0x7fc0;\n\
MSSTORE:\n\
    add.u64 %rd8, %rd1, %rd9;\n\
    cvt.u16.u32 %rs2, %r8;\n\
    st.global.u16 [%rd8], %rs2;\n\
MSDONE: ret;\n\
}\0";

// P4 item 17: fused FASE-Deferred AdamW step with a BF16 AUTHORITATIVE theta
// and counter-based stochastic rounding (SR-BF16, no FP32 master copy).
//
// Identical update arithmetic and rounding order to
// FASE_FUSED_ADAMW_STEP_F32_PTX (m/v/mp stay f32 buffers; every intermediate
// rounds .rn exactly like the decomposed kernels), with two differences:
//   1. theta loads as bf16 (widen = bits<<16, exact) and stores as bf16.
//   2. The theta store is STOCHASTICALLY rounded: a splitmix64 hash of
//      (key, ctr_base + element) produces a 16-bit dither added to the f32
//      result's raw bits before truncating the low 16 bits. `key` arrives
//      precomputed as seed ^ (step * SR_STEP_SALT) from the host so the
//      kernel is a pure function of its counters — bit-identical to the CPU
//      reference `sr_bf16::sr_bf16_round_counter`.
//
// Explicit edge policy (mirrors sr_bf16.rs, the reference):
//   - arithmetic Inf propagates (0x7F80|sign), NaN forces quiet 0x7FC0|sign;
//   - rounding-induced overflow (dither carries into exponent 0xFF)
//     SATURATES to +-max-normal 0x7F7F|sign;
//   - subnormal underflow is gradual via the same bit arithmetic.
pub(crate) const FASE_FUSED_ADAMW_STEP_BF16SR_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_fase_fused_adamw_step_bf16sr(\n\
    .param .u64 theta, .param .u64 m, .param .u64 v, .param .u64 mp, .param .u64 n,\n\
    .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,\n\
    .param .f32 eps, .param .f32 neg_lr, .param .f32 neg_lr_wd,\n\
    .param .f32 bc1, .param .f32 bc2, .param .u32 has_wd,\n\
    .param .u64 sr_key, .param .u64 sr_ctr_base\n\
) {\n\
    .reg .u32 %r<12>;\n\
    .reg .u64 %rd<16>;\n\
    .reg .f32 %fs<16>;\n\
    .reg .b16 %rs<3>;\n\
    .reg .pred %p<5>;\n\
    ld.param.u64 %rd1, [theta];\n\
    ld.param.u64 %rd2, [m];\n\
    ld.param.u64 %rd3, [v];\n\
    ld.param.u64 %rd4, [mp];\n\
    ld.param.u64 %rd5, [n];\n\
    ld.param.f32 %fs1, [b1];\n\
    ld.param.f32 %fs2, [omb1];\n\
    ld.param.f32 %fs3, [b2];\n\
    ld.param.f32 %fs4, [omb2];\n\
    ld.param.f32 %fs5, [eps];\n\
    ld.param.f32 %fs6, [neg_lr];\n\
    ld.param.f32 %fs7, [neg_lr_wd];\n\
    ld.param.f32 %fs8, [bc1];\n\
    ld.param.f32 %fs9, [bc2];\n\
    ld.param.u32 %r4, [has_wd];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    // theta: bf16 load, widen exactly (bits<<16)\n\
    shl.b64 %rd9, %rd6, 1;\n\
    add.u64 %rd8, %rd1, %rd9;\n\
    ld.global.u16 %rs1, [%rd8];\n\
    cvt.u32.u16 %r5, %rs1;\n\
    shl.b32 %r5, %r5, 16;\n\
    mov.b32 %fs10, %r5;\n\
    // m/v/mp: f32 buffers, byte offset = idx*4\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs11, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs12, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs13, [%rd8];\n\
    mul.rn.f32 %fs14, %fs11, %fs1;\n\
    mul.rn.f32 %fs15, %fs13, %fs2;\n\
    add.rn.f32 %fs11, %fs14, %fs15;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    st.global.f32 [%rd8], %fs11;\n\
    mul.rn.f32 %fs14, %fs13, %fs13;\n\
    mul.rn.f32 %fs14, %fs14, %fs4;\n\
    mul.rn.f32 %fs12, %fs12, %fs3;\n\
    add.rn.f32 %fs12, %fs12, %fs14;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    st.global.f32 [%rd8], %fs12;\n\
    mul.rn.f32 %fs14, %fs11, %fs8;\n\
    mul.rn.f32 %fs15, %fs12, %fs9;\n\
    sqrt.rn.f32 %fs15, %fs15;\n\
    add.rn.f32 %fs15, %fs15, %fs5;\n\
    div.approx.f32 %fs14, %fs14, %fs15;\n\
    mul.rn.f32 %fs14, %fs14, %fs6;\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra SKIPWD;\n\
    mul.rn.f32 %fs15, %fs10, %fs7;\n\
    add.rn.f32 %fs14, %fs14, %fs15;\n\
SKIPWD:\n\
    add.rn.f32 %fs10, %fs10, %fs14;\n\
    // ---- SR tail: theta_new (f32 in %fs10) -> bf16 bits in %r8 ----\n\
    mov.b32 %r5, %fs10;\n\
    and.b32 %r9, %r5, 0x80000000;\n\
    shr.u32 %r9, %r9, 16;\n\
    and.b32 %r6, %r5, 0x7f800000;\n\
    setp.eq.u32 %p3, %r6, 0x7f800000;\n\
    @%p3 bra SPECIAL;\n\
    // dither = low 16 bits of splitmix64(sr_key, sr_ctr_base + idx)\n\
    ld.param.u64 %rd10, [sr_key];\n\
    ld.param.u64 %rd11, [sr_ctr_base];\n\
    add.u64 %rd11, %rd11, %rd6;\n\
    mov.u64 %rd12, 0x9E3779B97F4A7C15;\n\
    mul.lo.u64 %rd13, %rd11, %rd12;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shr.u64 %rd14, %rd13, 30;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0xBF58476D1CE4E5B9;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 27;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0x94D049BB133111EB;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 31;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    cvt.u32.u64 %r7, %rd13;\n\
    and.b32 %r7, %r7, 0xffff;\n\
    add.u32 %r6, %r5, %r7;\n\
    and.b32 %r7, %r6, 0x7f800000;\n\
    setp.eq.u32 %p4, %r7, 0x7f800000;\n\
    @%p4 bra SATURATE;\n\
    shr.u32 %r8, %r6, 16;\n\
    bra STORE;\n\
SATURATE:\n\
    or.b32 %r8, %r9, 0x7f7f;\n\
    bra STORE;\n\
SPECIAL:\n\
    and.b32 %r6, %r5, 0x007fffff;\n\
    setp.ne.u32 %p4, %r6, 0;\n\
    @%p4 bra QNAN;\n\
    or.b32 %r8, %r9, 0x7f80;\n\
    bra STORE;\n\
QNAN:\n\
    or.b32 %r8, %r9, 0x7fc0;\n\
STORE:\n\
    add.u64 %rd8, %rd1, %rd9;\n\
    cvt.u16.u32 %rs2, %r8;\n\
    st.global.u16 [%rd8], %rs2;\n\
DONE: ret;\n\
}\0";

// P4 item 17: standalone SR-BF16 rounding probe — the SAME dither hash and
// rounding tail as FASE_FUSED_ADAMW_STEP_BF16SR_PTX, applied to an arbitrary
// f32 buffer. Exists so the parity gate can assert the GPU tail is
// bit-identical to the CPU reference (`sr_bf16::sr_bf16_round_counter`) on
// adversarial inputs (max-normal, subnormal, Inf, NaN) without needing to
// reproduce div.approx-dependent optimizer arithmetic on the host.
pub(crate) const SR_BF16_ROUND_PROBE_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sr_bf16_round_probe(\n\
    .param .u64 src, .param .u64 dst, .param .u64 n,\n\
    .param .u64 sr_key, .param .u64 sr_ctr_base\n\
) {\n\
    .reg .u32 %r<12>;\n\
    .reg .u64 %rd<16>;\n\
    .reg .b16 %rs<3>;\n\
    .reg .pred %p<5>;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd5, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.u32 %r5, [%rd8];\n\
    and.b32 %r9, %r5, 0x80000000;\n\
    shr.u32 %r9, %r9, 16;\n\
    and.b32 %r6, %r5, 0x7f800000;\n\
    setp.eq.u32 %p3, %r6, 0x7f800000;\n\
    @%p3 bra SPECIAL;\n\
    ld.param.u64 %rd10, [sr_key];\n\
    ld.param.u64 %rd11, [sr_ctr_base];\n\
    add.u64 %rd11, %rd11, %rd6;\n\
    mov.u64 %rd12, 0x9E3779B97F4A7C15;\n\
    mul.lo.u64 %rd13, %rd11, %rd12;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shr.u64 %rd14, %rd13, 30;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0xBF58476D1CE4E5B9;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 27;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0x94D049BB133111EB;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 31;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    cvt.u32.u64 %r7, %rd13;\n\
    and.b32 %r7, %r7, 0xffff;\n\
    add.u32 %r6, %r5, %r7;\n\
    and.b32 %r7, %r6, 0x7f800000;\n\
    setp.eq.u32 %p4, %r7, 0x7f800000;\n\
    @%p4 bra SATURATE;\n\
    shr.u32 %r8, %r6, 16;\n\
    bra STORE;\n\
SATURATE:\n\
    or.b32 %r8, %r9, 0x7f7f;\n\
    bra STORE;\n\
SPECIAL:\n\
    and.b32 %r6, %r5, 0x007fffff;\n\
    setp.ne.u32 %p4, %r6, 0;\n\
    @%p4 bra QNAN;\n\
    or.b32 %r8, %r9, 0x7f80;\n\
    bra STORE;\n\
QNAN:\n\
    or.b32 %r8, %r9, 0x7fc0;\n\
STORE:\n\
    shl.b64 %rd9, %rd6, 1;\n\
    add.u64 %rd8, %rd2, %rd9;\n\
    cvt.u16.u32 %rs2, %r8;\n\
    st.global.u16 [%rd8], %rs2;\n\
DONE: ret;\n\
}\0";

/// clamp_backward: out[i] = (input[i] >= min_val && input[i] <= max_val) ? grad[i] : 0
pub(crate) const CLAMP_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_clamp_backward_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out,\n\
    .param .f32 min_val, .param .f32 max_val, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<6>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.f32 %fs4, [min_val];\n\
    ld.param.f32 %fs5, [max_val];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    setp.ge.f32 %p1, %fs2, %fs4;\n\
    setp.le.f32 %p2, %fs2, %fs5;\n\
    and.pred %p1, %p1, %p2;\n\
    selp.f32 %fs3, %fs1, 0f00000000, %p1;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

/// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
pub(crate) const TANH_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_tanh_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd1, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    min.f32 %fs1, %fs1, 0f42300000;\n\
    max.f32 %fs1, %fs1, 0fC2300000;\n\
    add.f32 %fs2, %fs1, %fs1;\n\
    mul.f32 %fs2, %fs2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs2, %fs2;\n\
    add.f32 %fs3, %fs2, 0f3F800000;\n\
    sub.f32 %fs2, %fs2, 0f3F800000;\n\
    div.approx.f32 %fs1, %fs2, %fs3;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

/// Every hand-written PTX module in this file, paired with its constant name.
///
/// Consumed by the `ptxas` gate in `super::tests`, which assembles each one.
/// These modules are only ever fed to `cuModuleLoadData` at runtime, so a syntax
/// error in them is invisible until a kernel launch fails on a real GPU.
#[cfg(test)]
pub(crate) const ALL_PTX: &[(&str, &str)] = &[
    ("DIV_F32_PTX", DIV_F32_PTX),
    ("ROTATE_HALF_F32_PTX", ROTATE_HALF_F32_PTX),
    ("ROTATE_HALF_NEG_F32_PTX", ROTATE_HALF_NEG_F32_PTX),
    ("DIV_SCALAR_F32_PTX", DIV_SCALAR_F32_PTX),
    ("RELU_BACKWARD_F32_PTX", RELU_BACKWARD_F32_PTX),
    ("SIGMOID_BACKWARD_F32_PTX", SIGMOID_BACKWARD_F32_PTX),
    ("TANH_BACKWARD_F32_PTX", TANH_BACKWARD_F32_PTX),
    ("GELU_BACKWARD_F32_PTX", GELU_BACKWARD_F32_PTX),
    ("SILU_BACKWARD_F32_PTX", SILU_BACKWARD_F32_PTX),
    ("SWIGLU_GATE_BACKWARD_F32_PTX", SWIGLU_GATE_BACKWARD_F32_PTX),
    ("SILU_BACKWARD_SRCAD_F32_PTX", SILU_BACKWARD_SRCAD_F32_PTX),
    ("SIGMOID_BACKWARD_SRCAD_F32_PTX", SIGMOID_BACKWARD_SRCAD_F32_PTX),
    ("TANH_BACKWARD_SRCAD_F32_PTX", TANH_BACKWARD_SRCAD_F32_PTX),
    ("GELU_BACKWARD_SRCAD_F32_PTX", GELU_BACKWARD_SRCAD_F32_PTX),
    ("FASE_FUSED_ADAMW_STEP_F32_PTX", FASE_FUSED_ADAMW_STEP_F32_PTX),
    ("FASE_FUSED_ADAMW_MULTI_F32_PTX", FASE_FUSED_ADAMW_MULTI_F32_PTX),
    ("FASE_FUSED_ADAMW_MULTI_BF16SR_PTX", FASE_FUSED_ADAMW_MULTI_BF16SR_PTX),
    ("FASE_FUSED_ADAMW_STEP_BF16SR_PTX", FASE_FUSED_ADAMW_STEP_BF16SR_PTX),
    ("SR_BF16_ROUND_PROBE_PTX", SR_BF16_ROUND_PROBE_PTX),
    ("CLAMP_BACKWARD_F32_PTX", CLAMP_BACKWARD_F32_PTX),
    ("TANH_F32_PTX", TANH_F32_PTX),
];
