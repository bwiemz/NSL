//! The hand-written `nsl_tanh_f32` and `nsl_gelu_backward_f32`, frozen
//! (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::kernels::{TANH_F32_PTX, GELU_BACKWARD_F32_PTX}` as
//! they stood when the kernels moved to
//! `nsl_kir::kernels::elementwise::{build_tanh, build_gelu_backward}`,
//! verbatim (their comments aside). They are the reference side of
//! `tanh_gelu_backward_kir_equivalence.rs` and are not compiled into
//! anything else.

pub const TANH_F32_PTX: &str = "\
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

pub const GELU_BACKWARD_F32_PTX: &str = "\
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
