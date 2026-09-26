//! The hand-written RoPE `rotate_half` kernels, frozen (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::kernels::{ROTATE_HALF,ROTATE_HALF_NEG}_F32_PTX` as
//! they stood when the kernels moved to
//! `nsl_kir::kernels::elementwise::build_rotate_half`, verbatim (their
//! comments aside). They are the reference side of
//! `rotate_half_kir_equivalence.rs` and are not compiled into anything else.

pub const ROTATE_HALF_F32_PTX: &str = "\
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

pub const ROTATE_HALF_NEG_F32_PTX: &str = "\
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
