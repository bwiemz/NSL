//! The hand-written 2-D-block row-lookup kernels, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{EMBEDDING_F32_PTX,
//! EMBEDDING_I32IDX_PTX, GATHER_F32_PTX, GATHER_I32IDX_PTX}` as they stood
//! when the kernels moved to `nsl_kir::kernels::lookup`, verbatim (their
//! comments aside; the PTX `//` lines are kept). They are the reference side
//! of `lookup_kir_equivalence.rs` and are not compiled into anything else.

pub const EMBEDDING_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_f32(\n\
    .param .u64 weight, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p1;\n\
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
    ld.param.u64 %rd1, [weight];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    cvt.u64.u32 %rd6, %r1;\n\
    cvt.u64.u32 %rd7, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra DONE;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd8, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd8;\n\
    ld.global.f32 %f1, [%rd8];\n\
    cvt.rzi.u64.f32 %rd9, %f1;\n\
    mul.lo.u64 %rd10, %rd9, %rd5;\n\
    add.u64 %rd10, %rd10, %rd7;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd1, %rd10;\n\
    ld.global.f32 %f1, [%rd10];\n\
    mul.lo.u64 %rd11, %rd6, %rd5;\n\
    add.u64 %rd11, %rd11, %rd7;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd3, %rd11;\n\
    st.global.f32 [%rd11], %f1;\n\
DONE: ret;\n\
}\0";

pub const EMBEDDING_I32IDX_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_i32idx(\n\
    .param .u64 weight, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p1;\n\
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
    ld.param.u64 %rd1, [weight];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
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
    mul.lo.u64 %rd10, %rd9, %rd5;\n\
    add.u64 %rd10, %rd10, %rd7;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd1, %rd10;\n\
    ld.global.f32 %f1, [%rd10];\n\
    mul.lo.u64 %rd11, %rd6, %rd5;\n\
    add.u64 %rd11, %rd11, %rd7;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd3, %rd11;\n\
    st.global.f32 [%rd11], %f1;\n\
DONE: ret;\n\
}\0";

pub const GATHER_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_gather_f32(\n\
    .param .u64 input, .param .u64 indices, .param .u64 out,\n\
    .param .u64 num_indices, .param .u64 inner_dim, .param .u64 input_rows\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<3>;\n\
    // i = blockIdx.x * blockDim.x + threadIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    // j = blockIdx.y * blockDim.y + threadIdx.y\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    // Load params\n\
    ld.param.u64 %rd1, [input];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [num_indices];\n\
    ld.param.u64 %rd5, [inner_dim];\n\
    ld.param.u64 %rd6, [input_rows];\n\
    // Bounds check\n\
    cvt.u64.u32 %rd7, %r1;\n\
    cvt.u64.u32 %rd8, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra G_DONE;\n\
    setp.ge.u64 %p2, %rd8, %rd5;\n\
    @%p2 bra G_DONE;\n\
    // Load index: idx = (int)indices[i]\n\
    shl.b64 %rd9, %rd7, 2;\n\
    add.u64 %rd9, %rd2, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    cvt.rzi.u64.f32 %rd10, %f1;\n\
    // Bounds check: idx < input_rows\n\
    setp.ge.u64 %p1, %rd10, %rd6;\n\
    @%p1 bra G_DONE;\n\
    // Load input[idx, j]\n\
    mul.lo.u64 %rd11, %rd10, %rd5;\n\
    add.u64 %rd11, %rd11, %rd8;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    // Store out[i, j]\n\
    mul.lo.u64 %rd12, %rd7, %rd5;\n\
    add.u64 %rd12, %rd12, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd3, %rd12;\n\
    st.global.f32 [%rd12], %f1;\n\
G_DONE: ret;\n\
}\0";

pub const GATHER_I32IDX_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_gather_i32idx(\n\
    .param .u64 input, .param .u64 indices, .param .u64 out,\n\
    .param .u64 num_indices, .param .u64 inner_dim, .param .u64 input_rows\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
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
    ld.param.u64 %rd1, [input];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [num_indices];\n\
    ld.param.u64 %rd5, [inner_dim];\n\
    ld.param.u64 %rd6, [input_rows];\n\
    cvt.u64.u32 %rd7, %r1;\n\
    cvt.u64.u32 %rd8, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra G_DONE;\n\
    setp.ge.u64 %p2, %rd8, %rd5;\n\
    @%p2 bra G_DONE;\n\
    // Load index as i32: idx = indices[i]\n\
    shl.b64 %rd9, %rd7, 2;\n\
    add.u64 %rd9, %rd2, %rd9;\n\
    ld.global.s32 %r5, [%rd9];\n\
    cvt.s64.s32 %rd10, %r5;\n\
    setp.ge.u64 %p1, %rd10, %rd6;\n\
    @%p1 bra G_DONE;\n\
    mul.lo.u64 %rd11, %rd10, %rd5;\n\
    add.u64 %rd11, %rd11, %rd8;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    mul.lo.u64 %rd12, %rd7, %rd5;\n\
    add.u64 %rd12, %rd12, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd3, %rd12;\n\
    st.global.f32 [%rd12], %f1;\n\
G_DONE: ret;\n\
}\0";
