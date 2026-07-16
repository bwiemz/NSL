//! Static PTX kernel strings for GPU elementwise operations.
//! All strings are null-terminated (end with `\0`) for CUDA driver API.

// PTX constants are loaded at runtime by name via the CUDA driver API;
// Rust's dead-code analysis cannot see these usages.
#![allow(dead_code)]

// --- Binary ops ---

pub(crate) const ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_add_f32(\n\
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
    add.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

pub(crate) const SUB_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sub_f32(\n\
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
    sub.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

pub(crate) const MUL_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_mul_f32(\n\
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
    mul.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

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

// --- Unary ops ---

pub(crate) const NEG_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_neg_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    neg.f32 %fs1, %fs1;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

pub(crate) const RELU_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_relu_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    mov.f32 %fs2, 0f00000000;\n\
    max.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

// --- Scalar ops ---

pub(crate) const MUL_SCALAR_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_mul_scalar_f32(\n\
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
    mul.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

// Fused scaled-add (FASE accumulate epilogue, Milestone C · p4):
//   m[i] = m[i] + (g[i] * s)      (m read-write, g read-only, s a host f32)
// Bit-exact replacement for `nsl_mul_scalar_f32` then `nsl_add_f32`, saving one
// launch and the scaled-grad temp.
//
// LOAD-BEARING `.rn`: the two GPU kernels it replaces double-round (the mul
// result is stored to global memory as f32, then reloaded and added). Within a
// single kernel, ptxas would by default CONTRACT a register-dependent
// `mul.f32`+`add.f32` into one `fma.f32` (a single rounding), diverging from the
// decomposed path by up to 1 ULP. The explicit `.rn` (round-to-nearest) modifier
// on `mul.rn.f32`/`add.rn.f32` forbids that contraction — same numerics as plain
// `.f32` (round-to-nearest is already the default), but each op rounds
// independently, reproducing the two-kernel double-rounding element-for-element.
pub(crate) const SCALAR_MUL_ADD_INPLACE_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_scalar_mul_add_inplace_f32(\n\
    .param .u64 m, .param .u64 g, .param .f32 s, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<8>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [m];\n\
    ld.param.u64 %rd2, [g];\n\
    ld.param.f32 %fs3, [s];\n\
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
    add.u64 %rd6, %rd2, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    mul.rn.f32 %fs1, %fs1, %fs3;\n\
    add.u64 %rd7, %rd1, %rd5;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    add.rn.f32 %fs2, %fs2, %fs1;\n\
    st.global.f32 [%rd7], %fs2;\n\
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

pub(crate) const ADD_SCALAR_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_add_scalar_f32(\n\
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
    add.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

// --- Unary math ops: exp, log, sqrt, abs, sign ---

/// exp(x) = 2^(x * log2(e))  using ex2.approx
pub(crate) const EXP_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_exp_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    mul.f32 %fs2, %fs1, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

/// ln(x) = log2(x) * ln(2)  using lg2.approx
pub(crate) const LOG_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_log_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    lg2.approx.f32 %fs2, %fs1;\n\
    mul.f32 %fs1, %fs2, 0f3F317218;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

pub(crate) const SQRT_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sqrt_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    sqrt.rn.f32 %fs1, %fs1;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

pub(crate) const ABS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_abs_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    abs.f32 %fs1, %fs1;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

/// sign(x): 1.0 if x>0, -1.0 if x<0, 0.0 if x==0
pub(crate) const SIGN_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sign_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p<3>;\n\
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
    mov.f32 %fs2, 0f00000000;\n\
    setp.gt.f32 %p1, %fs1, %fs2;\n\
    setp.lt.f32 %p2, %fs1, %fs2;\n\
    selp.f32 %fs1, 0f3F800000, 0f00000000, %p1;\n\
    selp.f32 %fs2, 0fBF800000, %fs1, %p2;\n\
    mov.f32 %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

// --- Activation functions ---

/// sigmoid(x) = 1 / (1 + exp(-x))
pub(crate) const SIGMOID_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sigmoid_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    neg.f32 %fs2, %fs1;\n\
    mul.f32 %fs2, %fs2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs2, %fs2;\n\
    add.f32 %fs2, %fs2, 0f3F800000;\n\
    rcp.approx.f32 %fs1, %fs2;\n\
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
//   s  = sigmoid(a)                 (identical instructions to SIGMOID_F32_PTX)
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
// The sigmoid ops stay plain `.f32` to match SIGMOID_F32_PTX byte-for-byte (they
// contain no contractible mul+add pair — ex2.approx/rcp.approx break the chain).
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

/// sin(x) using sin.approx.f32
pub(crate) const SIN_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sin_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    sin.approx.f32 %fs1, %fs1;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

/// cos(x) using cos.approx.f32
pub(crate) const COS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_cos_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
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
    cos.approx.f32 %fs1, %fs1;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

/// gelu(x) = x * sigmoid(1.702 * x)  [sigmoid approximation]
/// sigmoid(1.702*x) = 1 / (1 + exp(-1.702*x))
pub(crate) const GELU_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_gelu_f32(\n\
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
    mul.f32 %fs2, %fs1, 0f3FD9999A;\n\
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    mul.f32 %fs1, %fs1, %fs3;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
pub(crate) const SILU_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_silu_f32(\n\
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
    neg.f32 %fs2, %fs1;\n\
    mul.f32 %fs2, %fs2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs2, %fs2;\n\
    add.f32 %fs2, %fs2, 0f3F800000;\n\
    rcp.approx.f32 %fs2, %fs2;\n\
    mul.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

/// clamp(x, lo, hi): max(lo, min(x, hi))
pub(crate) const CLAMP_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_clamp_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n, .param .f32 lo, .param .f32 hi\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.u64 %rd3, [n];\n\
    ld.param.f32 %fs2, [lo];\n\
    ld.param.f32 %fs3, [hi];\n\
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
    max.f32 %fs1, %fs1, %fs2;\n\
    min.f32 %fs1, %fs1, %fs3;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
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
    ("ADD_F32_PTX", ADD_F32_PTX),
    ("SUB_F32_PTX", SUB_F32_PTX),
    ("MUL_F32_PTX", MUL_F32_PTX),
    ("DIV_F32_PTX", DIV_F32_PTX),
    ("ROTATE_HALF_F32_PTX", ROTATE_HALF_F32_PTX),
    ("NEG_F32_PTX", NEG_F32_PTX),
    ("RELU_F32_PTX", RELU_F32_PTX),
    ("MUL_SCALAR_F32_PTX", MUL_SCALAR_F32_PTX),
    ("SCALAR_MUL_ADD_INPLACE_F32_PTX", SCALAR_MUL_ADD_INPLACE_F32_PTX),
    ("ADD_SCALAR_F32_PTX", ADD_SCALAR_F32_PTX),
    ("EXP_F32_PTX", EXP_F32_PTX),
    ("LOG_F32_PTX", LOG_F32_PTX),
    ("SQRT_F32_PTX", SQRT_F32_PTX),
    ("ABS_F32_PTX", ABS_F32_PTX),
    ("SIGN_F32_PTX", SIGN_F32_PTX),
    ("SIGMOID_F32_PTX", SIGMOID_F32_PTX),
    ("RELU_BACKWARD_F32_PTX", RELU_BACKWARD_F32_PTX),
    ("SIGMOID_BACKWARD_F32_PTX", SIGMOID_BACKWARD_F32_PTX),
    ("TANH_BACKWARD_F32_PTX", TANH_BACKWARD_F32_PTX),
    ("GELU_BACKWARD_F32_PTX", GELU_BACKWARD_F32_PTX),
    ("SILU_BACKWARD_F32_PTX", SILU_BACKWARD_F32_PTX),
    ("SILU_BACKWARD_SRCAD_F32_PTX", SILU_BACKWARD_SRCAD_F32_PTX),
    ("CLAMP_BACKWARD_F32_PTX", CLAMP_BACKWARD_F32_PTX),
    ("SIN_F32_PTX", SIN_F32_PTX),
    ("COS_F32_PTX", COS_F32_PTX),
    ("GELU_F32_PTX", GELU_F32_PTX),
    ("SILU_F32_PTX", SILU_F32_PTX),
    ("CLAMP_F32_PTX", CLAMP_F32_PTX),
    ("TANH_F32_PTX", TANH_F32_PTX),
];
