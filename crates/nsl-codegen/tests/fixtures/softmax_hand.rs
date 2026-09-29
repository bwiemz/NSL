//! The hand-written row softmax and log-softmax, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::SOFTMAX_F32_PTX` and
//! `LOG_SOFTMAX_F32_PTX` as they stood when the kernels moved to
//! `nsl_kir::kernels::softmax`, verbatim (their comments aside; the PTX `//`
//! lines are kept). They are the reference side of
//! `softmax_kir_equivalence.rs` and are not compiled into anything else.

pub const SOFTMAX_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_softmax_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 rows, .param .u64 cols\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<8>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 smax[256];\n\
    .shared .f32 ssum[256];\n\
    .reg .u32 %r8;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [rows];\n\
    ld.param.u64 %rd4, [cols];\n\
    // row = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd5, %r1;\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra DONE;\n\
    // tid = threadIdx.x\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd6, %r2;\n\
    // row_base = row * cols * 4\n\
    mul.lo.u64 %rd7, %rd5, %rd4;\n\
    shl.b64 %rd7, %rd7, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    add.u64 %rd9, %rd2, %rd7;\n\
    // --- Pass 1: find row max ---\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd10, %rd6;\n\
MAX_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra MAX_DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd11, %rd8, %rd11;\n\
    ld.global.f32 %f2, [%rd11];\n\
    max.f32 %f1, %f1, %f2;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra MAX_LOOP;\n\
MAX_DONE:\n\
    // Store local max to shared memory\n\
    cvt.u32.u64 %r3, %rd6;\n\
    mul.lo.u32 %r3, %r3, 4;\n\
    mov.u32 %r8, smax; add.u32 %r8, %r8, %r3; st.shared.f32 [%r8], %f1;\n\
    bar.sync 0;\n\
    // Reduce shared memory max (thread 0 only, simple sequential)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra SKIP_MAX_REDUCE;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r5, %ntid.x;\n\
REDUCE_MAX:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra DONE_MAX_REDUCE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r8, smax; add.u32 %r8, %r8, %r6; ld.shared.f32 %f3, [%r8];\n\
    max.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra REDUCE_MAX;\n\
DONE_MAX_REDUCE:\n\
    st.shared.f32 [smax], %f1;\n\
SKIP_MAX_REDUCE:\n\
    bar.sync 0;\n\
    // Load global max\n\
    ld.shared.f32 %f1, [smax];\n\
    // --- Pass 2: exp(x - max) and partial sum ---\n\
    mov.f32 %f4, 0f00000000;\n\
    mov.u64 %rd10, %rd6;\n\
EXP_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra EXP_DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd12, %rd8, %rd11;\n\
    ld.global.f32 %f2, [%rd12];\n\
    sub.f32 %f2, %f2, %f1;\n\
    // exp(x) = 2^(x * log2(e))\n\
    mul.f32 %f2, %f2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f2, %f2;\n\
    add.u64 %rd12, %rd9, %rd11;\n\
    st.global.f32 [%rd12], %f2;\n\
    add.f32 %f4, %f4, %f2;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra EXP_LOOP;\n\
EXP_DONE:\n\
    // Store partial sum to shared memory\n\
    cvt.u32.u64 %r3, %rd6;\n\
    mul.lo.u32 %r3, %r3, 4;\n\
    mov.u32 %r8, ssum; add.u32 %r8, %r8, %r3; st.shared.f32 [%r8], %f4;\n\
    bar.sync 0;\n\
    // Reduce shared memory sum (thread 0)\n\
    @%p3 bra SKIP_SUM_REDUCE;\n\
    mov.u32 %r4, 1;\n\
REDUCE_SUM:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra DONE_SUM_REDUCE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r8, ssum; add.u32 %r8, %r8, %r6; ld.shared.f32 %f5, [%r8];\n\
    add.f32 %f4, %f4, %f5;\n\
    add.u32 %r4, %r4, 1;\n\
    bra REDUCE_SUM;\n\
DONE_SUM_REDUCE:\n\
    rcp.approx.f32 %f4, %f4;\n\
    st.shared.f32 [ssum], %f4;\n\
SKIP_SUM_REDUCE:\n\
    bar.sync 0;\n\
    // Load 1/sum\n\
    ld.shared.f32 %f4, [ssum];\n\
    // --- Pass 3: divide by sum ---\n\
    mov.u64 %rd10, %rd6;\n\
DIV_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd12, %rd9, %rd11;\n\
    ld.global.f32 %f6, [%rd12];\n\
    mul.f32 %f6, %f6, %f4;\n\
    st.global.f32 [%rd12], %f6;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra DIV_LOOP;\n\
DONE: ret;\n\
}\0";

pub const LOG_SOFTMAX_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_log_softmax_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 rows, .param .u64 cols\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<8>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 smax[256];\n\
    .shared .f32 ssum[256];\n\
    .reg .u32 %r8;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [rows];\n\
    ld.param.u64 %rd4, [cols];\n\
    // row = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd5, %r1;\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra DONE;\n\
    // tid = threadIdx.x\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd6, %r2;\n\
    // row_base = row * cols * 4\n\
    mul.lo.u64 %rd7, %rd5, %rd4;\n\
    shl.b64 %rd7, %rd7, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    add.u64 %rd9, %rd2, %rd7;\n\
    // --- Pass 1: find row max ---\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd10, %rd6;\n\
MAX_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra MAX_DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd11, %rd8, %rd11;\n\
    ld.global.f32 %f2, [%rd11];\n\
    max.f32 %f1, %f1, %f2;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra MAX_LOOP;\n\
MAX_DONE:\n\
    // Store local max to shared memory\n\
    cvt.u32.u64 %r3, %rd6;\n\
    mul.lo.u32 %r3, %r3, 4;\n\
    mov.u32 %r8, smax; add.u32 %r8, %r8, %r3; st.shared.f32 [%r8], %f1;\n\
    bar.sync 0;\n\
    // Reduce shared memory max (thread 0 only, simple sequential)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra SKIP_MAX_REDUCE;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r5, %ntid.x;\n\
REDUCE_MAX:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra DONE_MAX_REDUCE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r8, smax; add.u32 %r8, %r8, %r6; ld.shared.f32 %f3, [%r8];\n\
    max.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra REDUCE_MAX;\n\
DONE_MAX_REDUCE:\n\
    st.shared.f32 [smax], %f1;\n\
SKIP_MAX_REDUCE:\n\
    bar.sync 0;\n\
    // Load global max\n\
    ld.shared.f32 %f1, [smax];\n\
    // --- Pass 2: exp(x - max) and partial sum ---\n\
    mov.f32 %f4, 0f00000000;\n\
    mov.u64 %rd10, %rd6;\n\
EXP_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra EXP_DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd12, %rd8, %rd11;\n\
    ld.global.f32 %f2, [%rd12];\n\
    sub.f32 %f2, %f2, %f1;\n\
    // exp(x) = 2^(x * log2(e))\n\
    mul.f32 %f2, %f2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f2, %f2;\n\
    add.f32 %f4, %f4, %f2;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra EXP_LOOP;\n\
EXP_DONE:\n\
    // Store partial sum to shared memory\n\
    cvt.u32.u64 %r3, %rd6;\n\
    mul.lo.u32 %r3, %r3, 4;\n\
    mov.u32 %r8, ssum; add.u32 %r8, %r8, %r3; st.shared.f32 [%r8], %f4;\n\
    bar.sync 0;\n\
    // Reduce shared memory sum (thread 0)\n\
    @%p3 bra SKIP_SUM_REDUCE;\n\
    mov.u32 %r4, 1;\n\
REDUCE_SUM:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra DONE_SUM_REDUCE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r8, ssum; add.u32 %r8, %r8, %r6; ld.shared.f32 %f5, [%r8];\n\
    add.f32 %f4, %f4, %f5;\n\
    add.u32 %r4, %r4, 1;\n\
    bra REDUCE_SUM;\n\
DONE_SUM_REDUCE:\n\
    // Compute log(sum) using log2(sum) / log2(e) = log2(sum) * ln(2)\n\
    lg2.approx.f32 %f4, %f4;\n\
    mul.f32 %f4, %f4, 0f3F317218;\n\
    st.shared.f32 [ssum], %f4;\n\
SKIP_SUM_REDUCE:\n\
    bar.sync 0;\n\
    // Load log(sum)\n\
    ld.shared.f32 %f4, [ssum];\n\
    // --- Pass 3: out[i] = (x[i] - max) - log(sum) ---\n\
    mov.u64 %rd10, %rd6;\n\
OUT_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd12, %rd8, %rd11;\n\
    ld.global.f32 %f6, [%rd12];\n\
    sub.f32 %f6, %f6, %f1;\n\
    sub.f32 %f6, %f6, %f4;\n\
    add.u64 %rd13, %rd9, %rd11;\n\
    st.global.f32 [%rd13], %f6;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra OUT_LOOP;\n\
DONE: ret;\n\
}\0";
