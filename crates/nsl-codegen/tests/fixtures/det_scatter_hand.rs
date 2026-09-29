//! The hand-written deterministic scatter-add, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::DET_SCATTER_ADD_F32_PTX` as it stood
//! when the kernel moved to `nsl_kir::kernels::det_scatter`, verbatim (its
//! comments aside; the PTX `//` lines are kept). It is the reference side of
//! `det_scatter_kir_equivalence.rs` and is not compiled into anything else.

pub const DET_SCATTER_ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_det_scatter_add_f32(\n\
    .param .u64 src, .param .u64 indices, .param .u64 input, .param .u64 out,\n\
    .param .u64 num_indices, .param .u64 embed_dim, .param .u64 vocab_size\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    // row = blockIdx.x * blockDim.x + threadIdx.x  (output row index)\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    // col = blockIdx.y * blockDim.y + threadIdx.y  (column index)\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    // Load params\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [input];\n\
    ld.param.u64 %rd4, [out];\n\
    ld.param.u64 %rd5, [num_indices];\n\
    ld.param.u64 %rd6, [embed_dim];\n\
    ld.param.u64 %rd7, [vocab_size];\n\
    // Bounds check: row < vocab_size, col < embed_dim\n\
    cvt.u64.u32 %rd8, %r1;\n\
    cvt.u64.u32 %rd9, %r3;\n\
    setp.ge.u64 %p1, %rd8, %rd7;\n\
    @%p1 bra DSA_DONE;\n\
    setp.ge.u64 %p2, %rd9, %rd6;\n\
    @%p2 bra DSA_DONE;\n\
    // acc = input[row, col] = input[row * embed_dim + col]\n\
    mul.lo.u64 %rd10, %rd8, %rd6;\n\
    add.u64 %rd10, %rd10, %rd9;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd11, %rd3, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    // Loop: for i = 0..num_indices\n\
    mov.u64 %rd12, 0;\n\
DSA_LOOP:\n\
    setp.ge.u64 %p3, %rd12, %rd5;\n\
    @%p3 bra DSA_STORE;\n\
    // Load indices[i]\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd13, %rd2, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    cvt.rzi.u64.f32 %rd14, %f2;\n\
    // if indices[i] == row, acc += src[i, col]\n\
    setp.ne.u64 %p3, %rd14, %rd8;\n\
    @%p3 bra DSA_NEXT;\n\
    // Load src[i, col] = src[i * embed_dim + col]\n\
    mul.lo.u64 %rd15, %rd12, %rd6;\n\
    add.u64 %rd15, %rd15, %rd9;\n\
    shl.b64 %rd15, %rd15, 2;\n\
    add.u64 %rd15, %rd1, %rd15;\n\
    ld.global.f32 %f3, [%rd15];\n\
    add.f32 %f1, %f1, %f3;\n\
DSA_NEXT:\n\
    add.u64 %rd12, %rd12, 1;\n\
    bra DSA_LOOP;\n\
DSA_STORE:\n\
    // out[row, col] = acc\n\
    shl.b64 %rd16, %rd10, 2;\n\
    add.u64 %rd16, %rd4, %rd16;\n\
    st.global.f32 [%rd16], %f1;\n\
DSA_DONE: ret;\n\
}\0";
