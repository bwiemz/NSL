//! The hand-written shared-memory tree reductions, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{GLOBAL_SUM_F32_PTX, SUM_DIM_F32_PTX,
//! MAX_DIM_F32_PTX}` as they stood when the kernels moved to
//! `nsl_kir::kernels::block_reduce`, verbatim (their comments aside; the PTX
//! `//` lines are kept). They are the reference side of
//! `block_reduce_kir_equivalence.rs` and are not compiled into anything
//! else.

pub const GLOBAL_SUM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_global_sum_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 n\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd4, %r2;\n\
    // Partial sum via stride loop\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd5, %rd4;\n\
GSUM_LOOP:\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra GSUM_DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd6, %rd1, %rd6;\n\
    ld.global.f32 %f2, [%rd6];\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd5, %rd5, 256;\n\
    bra GSUM_LOOP;\n\
GSUM_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    mov.u32 %r4, 128;\n\
GREDUCE_LOOP:\n\
    setp.lt.u32 %p1, %r4, 1;\n\
    @%p1 bra GREDUCE_DONE;\n\
    setp.ge.u32 %p2, %r2, %r4;\n\
    @%p2 bra GSKIP;\n\
    mul.lo.u32 %r5, %r2, 4;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    ld.shared.f32 %f2, [%r7];\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f2, %f2, %f3;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    st.shared.f32 [%r7], %f2;\n\
GSKIP:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra GREDUCE_LOOP;\n\
GREDUCE_DONE:\n\
    setp.ne.u32 %p1, %r2, 0;\n\
    @%p1 bra GDONE;\n\
    ld.shared.f32 %f1, [sdata];\n\
    st.global.f32 [%rd2], %f1;\n\
GDONE: ret;\n\
}\0";

pub const SUM_DIM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_sum_dim_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 outer, .param .u64 reduce_size, .param .u64 inner\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [outer];\n\
    ld.param.u64 %rd4, [reduce_size];\n\
    ld.param.u64 %rd5, [inner];\n\
    // block_id = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    // total_out = outer * inner\n\
    mul.lo.u64 %rd7, %rd3, %rd5;\n\
    setp.ge.u64 %p1, %rd6, %rd7;\n\
    @%p1 bra DONE;\n\
    // outer_idx = block_id / inner\n\
    div.u64 %rd8, %rd6, %rd5;\n\
    // inner_idx = block_id % inner\n\
    rem.u64 %rd9, %rd6, %rd5;\n\
    // base_in = (outer_idx * reduce_size * inner + inner_idx) * 4\n\
    mul.lo.u64 %rd10, %rd8, %rd4;\n\
    mul.lo.u64 %rd10, %rd10, %rd5;\n\
    add.u64 %rd10, %rd10, %rd9;\n\
    // stride_in = inner (elements between consecutive reduce elements)\n\
    // tid\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd11, %r2;\n\
    // Partial sum via stride loop\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd12, %rd11;\n\
SUM_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd4;\n\
    @%p2 bra SUM_DONE;\n\
    // addr = (base_in + k * inner) * 4\n\
    mul.lo.u64 %rd13, %rd12, %rd5;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd1, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra SUM_LOOP;\n\
SUM_DONE:\n\
    // Store partial sum to shared memory\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Tree reduction in shared memory\n\
    mov.u32 %r4, 128;\n\
REDUCE_LOOP:\n\
    setp.lt.u32 %p2, %r4, 1;\n\
    @%p2 bra REDUCE_DONE;\n\
    setp.ge.u32 %p3, %r2, %r4;\n\
    @%p3 bra SKIP_REDUCE;\n\
    mul.lo.u32 %r5, %r2, 4;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    ld.shared.f32 %f2, [%r7];\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f2, %f2, %f3;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    st.shared.f32 [%r7], %f2;\n\
SKIP_REDUCE:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra REDUCE_LOOP;\n\
REDUCE_DONE:\n\
    // Thread 0 writes result\n\
    setp.ne.u32 %p2, %r2, 0;\n\
    @%p2 bra DONE;\n\
    ld.shared.f32 %f1, [sdata];\n\
    // out[block_id] = sum\n\
    shl.b64 %rd14, %rd6, 2;\n\
    add.u64 %rd14, %rd2, %rd14;\n\
    st.global.f32 [%rd14], %f1;\n\
DONE: ret;\n\
}\0";

pub const MAX_DIM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_max_dim_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 outer, .param .u64 reduce_size, .param .u64 inner\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [outer];\n\
    ld.param.u64 %rd4, [reduce_size];\n\
    ld.param.u64 %rd5, [inner];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    mul.lo.u64 %rd7, %rd3, %rd5;\n\
    setp.ge.u64 %p1, %rd6, %rd7;\n\
    @%p1 bra DONE;\n\
    div.u64 %rd8, %rd6, %rd5;\n\
    rem.u64 %rd9, %rd6, %rd5;\n\
    mul.lo.u64 %rd10, %rd8, %rd4;\n\
    mul.lo.u64 %rd10, %rd10, %rd5;\n\
    add.u64 %rd10, %rd10, %rd9;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd11, %r2;\n\
    // Initialize with -inf\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd12, %rd11;\n\
MAX_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd4;\n\
    @%p2 bra MAX_DONE;\n\
    mul.lo.u64 %rd13, %rd12, %rd5;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd1, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    max.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra MAX_LOOP;\n\
MAX_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    mov.u32 %r4, 128;\n\
REDUCE_LOOP:\n\
    setp.lt.u32 %p2, %r4, 1;\n\
    @%p2 bra REDUCE_DONE;\n\
    setp.ge.u32 %p3, %r2, %r4;\n\
    @%p3 bra SKIP_REDUCE;\n\
    mul.lo.u32 %r5, %r2, 4;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    ld.shared.f32 %f2, [%r7];\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    max.f32 %f2, %f2, %f3;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    st.shared.f32 [%r7], %f2;\n\
SKIP_REDUCE:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra REDUCE_LOOP;\n\
REDUCE_DONE:\n\
    setp.ne.u32 %p2, %r2, 0;\n\
    @%p2 bra DONE;\n\
    ld.shared.f32 %f1, [sdata];\n\
    shl.b64 %rd14, %rd6, 2;\n\
    add.u64 %rd14, %rd2, %rd14;\n\
    st.global.f32 [%rd14], %f1;\n\
DONE: ret;\n\
}\0";
