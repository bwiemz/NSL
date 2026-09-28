//! The hand-written embedding backward kernels, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{EMBEDDING_BWD_F32_PTX,
//! EMBEDDING_BWD_I32IDX_PTX, EMBEDDING_BWD_DET_F32_PTX,
//! EMBEDDING_BWD_DET_I32IDX_PTX}` as they stood when the kernels moved to
//! `nsl_kir::kernels::embedding_bwd`, verbatim (their comments aside). They
//! are the reference side of `embedding_bwd_kir_equivalence.rs` and are not
//! compiled into anything else.

pub const EMBEDDING_BWD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_bwd_f32(\n\
    .param .u64 grad, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim, .param .u64 vocab\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<3>;\n\
    .reg .pred %p<3>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    ld.param.u64 %rd12, [vocab];\n\
    cvt.u64.u32 %rd6, %r1;\n\
    cvt.u64.u32 %rd7, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra DONE;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd8, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd8;\n\
    ld.global.f32 %f1, [%rd8];\n\
    cvt.rzi.s64.f32 %rd9, %f1;\n\
    setp.lt.s64 %p2, %rd9, 0;\n\
    @%p2 bra DONE;\n\
    setp.ge.s64 %p2, %rd9, %rd12;\n\
    @%p2 bra DONE;\n\
    mul.lo.u64 %rd11, %rd6, %rd5;\n\
    add.u64 %rd11, %rd11, %rd7;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f2, [%rd11];\n\
    mul.lo.u64 %rd10, %rd9, %rd5;\n\
    add.u64 %rd10, %rd10, %rd7;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd3, %rd10;\n\
    red.global.add.f32 [%rd10], %f2;\n\
DONE: ret;\n\
}\0";

pub const EMBEDDING_BWD_I32IDX_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_bwd_i32idx(\n\
    .param .u64 grad, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim, .param .u64 vocab\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<7>;\n\
    .reg .f32 %f<3>;\n\
    .reg .pred %p<3>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    ld.param.u64 %rd12, [vocab];\n\
    cvt.u64.u32 %rd6, %r1;\n\
    cvt.u64.u32 %rd7, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra DONE;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd8, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd8;\n\
    ld.global.s32 %r5, [%rd8];\n\
    cvt.s64.s32 %rd9, %r5;\n\
    setp.lt.s64 %p2, %rd9, 0;\n\
    @%p2 bra DONE;\n\
    setp.ge.s64 %p2, %rd9, %rd12;\n\
    @%p2 bra DONE;\n\
    mul.lo.u64 %rd11, %rd6, %rd5;\n\
    add.u64 %rd11, %rd11, %rd7;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f2, [%rd11];\n\
    mul.lo.u64 %rd10, %rd9, %rd5;\n\
    add.u64 %rd10, %rd10, %rd7;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd3, %rd10;\n\
    red.global.add.f32 [%rd10], %f2;\n\
DONE: ret;\n\
}\0";

pub const EMBEDDING_BWD_DET_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_bwd_det_f32(\n\
    .param .u64 grad, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim, .param .u64 vocab\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    ld.param.u64 %rd6, [vocab];\n\
    cvt.u64.u32 %rd7, %r1;\n\
    cvt.u64.u32 %rd8, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd6;\n\
    @%p1 bra DONE;\n\
    setp.ge.u64 %p1, %rd8, %rd5;\n\
    @%p1 bra DONE;\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd9, 0;\n\
LOOP:\n\
    setp.ge.u64 %p2, %rd9, %rd4;\n\
    @%p2 bra WRITE;\n\
    shl.b64 %rd10, %rd9, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    ld.global.f32 %f2, [%rd10];\n\
    cvt.rzi.s64.f32 %rd11, %f2;\n\
    setp.ne.s64 %p3, %rd11, %rd7;\n\
    @%p3 bra NEXT;\n\
    mul.lo.u64 %rd12, %rd9, %rd5;\n\
    add.u64 %rd12, %rd12, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd1, %rd12;\n\
    ld.global.f32 %f3, [%rd12];\n\
    add.f32 %f1, %f1, %f3;\n\
NEXT:\n\
    add.u64 %rd9, %rd9, 1;\n\
    bra LOOP;\n\
WRITE:\n\
    mul.lo.u64 %rd13, %rd7, %rd5;\n\
    add.u64 %rd13, %rd13, %rd8;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd3, %rd13;\n\
    st.global.f32 [%rd13], %f1;\n\
DONE: ret;\n\
}\0";

pub const EMBEDDING_BWD_DET_I32IDX_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_bwd_det_i32idx(\n\
    .param .u64 grad, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim, .param .u64 vocab\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    ld.param.u64 %rd6, [vocab];\n\
    cvt.u64.u32 %rd7, %r1;\n\
    cvt.u64.u32 %rd8, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd6;\n\
    @%p1 bra DONE;\n\
    setp.ge.u64 %p1, %rd8, %rd5;\n\
    @%p1 bra DONE;\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd9, 0;\n\
LOOP:\n\
    setp.ge.u64 %p2, %rd9, %rd4;\n\
    @%p2 bra WRITE;\n\
    shl.b64 %rd10, %rd9, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    ld.global.s32 %r5, [%rd10];\n\
    cvt.s64.s32 %rd11, %r5;\n\
    setp.ne.s64 %p3, %rd11, %rd7;\n\
    @%p3 bra NEXT;\n\
    mul.lo.u64 %rd12, %rd9, %rd5;\n\
    add.u64 %rd12, %rd12, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd1, %rd12;\n\
    ld.global.f32 %f3, [%rd12];\n\
    add.f32 %f1, %f1, %f3;\n\
NEXT:\n\
    add.u64 %rd9, %rd9, 1;\n\
    bra LOOP;\n\
WRITE:\n\
    mul.lo.u64 %rd13, %rd7, %rd5;\n\
    add.u64 %rd13, %rd13, %rd8;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd3, %rd13;\n\
    st.global.f32 [%rd13], %f1;\n\
DONE: ret;\n\
}\0";
