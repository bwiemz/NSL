//! The hand-written single-block tensor statistics, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::TENSOR_STATS_F32_PTX` as it stood when
//! the kernel moved to `nsl_kir::kernels::tensor_stats`, verbatim (its
//! comments aside; the PTX `//` lines are kept). It is the reference side of
//! `tensor_stats_kir_equivalence.rs` and is not compiled into anything else.

pub const TENSOR_STATS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_tensor_stats_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<12>;\n\
    .reg .pred %p<4>;\n\
    // 4 shared arrays of 256 floats: smin[256], smax[256], ssum[256], ssq[256]\n\
    .shared .f32 smin[256];\n\
    .shared .f32 smax[256];\n\
    .shared .f32 ssum[256];\n\
    .shared .f32 ssq[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd4, %r2;\n\
    // Init accumulators: min=+inf, max=-inf, sum=0, sq=0\n\
    mov.f32 %f1, 0f7F800000;\n\
    mov.f32 %f2, 0fFF800000;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.f32 %f4, 0f00000000;\n\
    mov.u64 %rd5, %rd4;\n\
TS_LOOP:\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra TS_REDUCE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd6, %rd1, %rd6;\n\
    ld.global.f32 %f5, [%rd6];\n\
    min.f32 %f1, %f1, %f5;\n\
    max.f32 %f2, %f2, %f5;\n\
    add.f32 %f3, %f3, %f5;\n\
    mul.f32 %f6, %f5, %f5;\n\
    add.f32 %f4, %f4, %f6;\n\
    add.u64 %rd5, %rd5, 256;\n\
    bra TS_LOOP;\n\
TS_REDUCE:\n\
    // Store per-thread results to shared memory\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, smin;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    mov.u32 %r7, smax;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f2;\n\
    mov.u32 %r7, ssum;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f3;\n\
    mov.u32 %r7, ssq;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f4;\n\
    bar.sync 0;\n\
    // Tree reduction: stride from 128 down to 1\n\
    mov.u32 %r4, 128;\n\
TS_RED_LOOP:\n\
    setp.lt.u32 %p1, %r4, 1;\n\
    @%p1 bra TS_RED_DONE;\n\
    setp.ge.u32 %p2, %r2, %r4;\n\
    @%p2 bra TS_RED_SKIP;\n\
    mul.lo.u32 %r5, %r2, 4;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 4;\n\
    // min reduction\n\
    mov.u32 %r7, smin;\n\
    add.u32 %r8, %r7, %r5;\n\
    ld.shared.f32 %f7, [%r8];\n\
    add.u32 %r9, %r7, %r6;\n\
    ld.shared.f32 %f8, [%r9];\n\
    min.f32 %f7, %f7, %f8;\n\
    st.shared.f32 [%r8], %f7;\n\
    // max reduction\n\
    mov.u32 %r7, smax;\n\
    add.u32 %r8, %r7, %r5;\n\
    ld.shared.f32 %f7, [%r8];\n\
    add.u32 %r9, %r7, %r6;\n\
    ld.shared.f32 %f8, [%r9];\n\
    max.f32 %f7, %f7, %f8;\n\
    st.shared.f32 [%r8], %f7;\n\
    // sum reduction\n\
    mov.u32 %r7, ssum;\n\
    add.u32 %r8, %r7, %r5;\n\
    ld.shared.f32 %f7, [%r8];\n\
    add.u32 %r9, %r7, %r6;\n\
    ld.shared.f32 %f8, [%r9];\n\
    add.f32 %f7, %f7, %f8;\n\
    st.shared.f32 [%r8], %f7;\n\
    // sum_sq reduction\n\
    mov.u32 %r7, ssq;\n\
    add.u32 %r8, %r7, %r5;\n\
    ld.shared.f32 %f7, [%r8];\n\
    add.u32 %r9, %r7, %r6;\n\
    ld.shared.f32 %f8, [%r9];\n\
    add.f32 %f7, %f7, %f8;\n\
    st.shared.f32 [%r8], %f7;\n\
TS_RED_SKIP:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra TS_RED_LOOP;\n\
TS_RED_DONE:\n\
    // Thread 0 writes final results: out[0..3] = min, max, sum, sum_sq\n\
    setp.ne.u32 %p1, %r2, 0;\n\
    @%p1 bra TS_DONE;\n\
    ld.shared.f32 %f1, [smin];\n\
    st.global.f32 [%rd2], %f1;\n\
    ld.shared.f32 %f2, [smax];\n\
    add.u64 %rd7, %rd2, 4;\n\
    st.global.f32 [%rd7], %f2;\n\
    ld.shared.f32 %f3, [ssum];\n\
    add.u64 %rd7, %rd2, 8;\n\
    st.global.f32 [%rd7], %f3;\n\
    ld.shared.f32 %f4, [ssq];\n\
    add.u64 %rd7, %rd2, 12;\n\
    st.global.f32 [%rd7], %f4;\n\
TS_DONE: ret;\n\
}\0";
