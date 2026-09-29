//! The hand-written row LayerNorm and RMSNorm forwards, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::LAYERNORM_F32_PTX` and
//! `RMSNORM_F32_PTX` as they stood when the kernels moved to
//! `nsl_kir::kernels::norm`, verbatim (their comments aside; the PTX `//`
//! lines are kept). They are the reference side of
//! `norm_kir_equivalence.rs` and are not compiled into anything else. The
//! LayerNorm reuses one shared slot for the mean and the variance partial
//! without a barrier between them, a race the gate shows.

pub const LAYERNORM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_layernorm_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 gamma, .param .u64 beta,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<12>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [beta];\n\
    ld.param.u64 %rd5, [rows];\n\
    ld.param.u64 %rd6, [cols];\n\
    ld.param.f32 %f10, [eps];\n\
    // row = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra LN_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
    // row_base = row * cols * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
    add.u64 %rd11, %rd2, %rd9;\n\
    // --- Pass 1: compute partial sum for mean ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd12, %rd8;\n\
LN_MEAN_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd6;\n\
    @%p2 bra LN_MEAN_DONE;\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra LN_MEAN_LOOP;\n\
LN_MEAN_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Reduce sum for mean (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra LN_SKIP_MEAN;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r5, %ntid.x;\n\
LN_RMEAN:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra LN_DMEAN;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LN_RMEAN;\n\
LN_DMEAN:\n\
    // mean = sum / cols\n\
    cvt.rn.f32.u64 %f4, %rd6;\n\
    div.approx.f32 %f1, %f1, %f4;\n\
    st.shared.f32 [sdata], %f1;\n\
LN_SKIP_MEAN:\n\
    bar.sync 0;\n\
    // Load mean\n\
    ld.shared.f32 %f5, [sdata];\n\
    // --- Pass 2: compute partial sum of (x - mean)^2 for variance ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd12, %rd8;\n\
LN_VAR_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd6;\n\
    @%p2 bra LN_VAR_DONE;\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    sub.f32 %f2, %f2, %f5;\n\
    mul.f32 %f2, %f2, %f2;\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra LN_VAR_LOOP;\n\
LN_VAR_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Reduce sum for variance (thread 0)\n\
    @%p3 bra LN_SKIP_VAR;\n\
    mov.u32 %r4, 1;\n\
LN_RVAR:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra LN_DVAR;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LN_RVAR;\n\
LN_DVAR:\n\
    // var = sum_sq / cols\n\
    cvt.rn.f32.u64 %f4, %rd6;\n\
    div.approx.f32 %f1, %f1, %f4;\n\
    // inv_std = rsqrt(var + eps)\n\
    add.f32 %f1, %f1, %f10;\n\
    rsqrt.approx.f32 %f1, %f1;\n\
    st.shared.f32 [sdata], %f1;\n\
LN_SKIP_VAR:\n\
    bar.sync 0;\n\
    // Load inv_std\n\
    ld.shared.f32 %f6, [sdata];\n\
    // --- Pass 3: normalize, scale, shift ---\n\
    mov.u64 %rd12, %rd8;\n\
LN_NORM_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd6;\n\
    @%p2 bra LN_DONE;\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd14, %rd10, %rd13;\n\
    ld.global.f32 %f2, [%rd14];\n\
    sub.f32 %f2, %f2, %f5;\n\
    mul.f32 %f2, %f2, %f6;\n\
    // gamma[j]\n\
    add.u64 %rd15, %rd3, %rd13;\n\
    ld.global.f32 %f7, [%rd15];\n\
    mul.f32 %f2, %f2, %f7;\n\
    // beta[j]\n\
    add.u64 %rd15, %rd4, %rd13;\n\
    ld.global.f32 %f8, [%rd15];\n\
    add.f32 %f2, %f2, %f8;\n\
    // Store\n\
    add.u64 %rd15, %rd11, %rd13;\n\
    st.global.f32 [%rd15], %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra LN_NORM_LOOP;\n\
LN_DONE: ret;\n\
}\0";

pub const RMSNORM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 gamma,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<10>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [rows];\n\
    ld.param.u64 %rd5, [cols];\n\
    ld.param.f32 %f8, [eps];\n\
    // row = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra RMS_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd7, %r2;\n\
    // row_base = row * cols * 4\n\
    mul.lo.u64 %rd8, %rd6, %rd5;\n\
    shl.b64 %rd8, %rd8, 2;\n\
    add.u64 %rd9, %rd1, %rd8;\n\
    add.u64 %rd10, %rd2, %rd8;\n\
    // --- Pass 1: compute partial sum of x^2 ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd11, %rd7;\n\
RMS_SQ_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra RMS_SQ_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    add.u64 %rd12, %rd9, %rd12;\n\
    ld.global.f32 %f2, [%rd12];\n\
    mul.f32 %f3, %f2, %f2;\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra RMS_SQ_LOOP;\n\
RMS_SQ_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Reduce sum_sq (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra RMS_SKIP_REDUCE;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r5, %ntid.x;\n\
RMS_RLOOP:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra RMS_RDONE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f4, [%r7];\n\
    add.f32 %f1, %f1, %f4;\n\
    add.u32 %r4, %r4, 1;\n\
    bra RMS_RLOOP;\n\
RMS_RDONE:\n\
    // rms_inv = rsqrt(sum_sq / cols + eps)\n\
    cvt.rn.f32.u64 %f5, %rd5;\n\
    div.approx.f32 %f1, %f1, %f5;\n\
    add.f32 %f1, %f1, %f8;\n\
    rsqrt.approx.f32 %f1, %f1;\n\
    st.shared.f32 [sdata], %f1;\n\
RMS_SKIP_REDUCE:\n\
    bar.sync 0;\n\
    // Load rms_inv\n\
    ld.shared.f32 %f6, [sdata];\n\
    // --- Pass 2: normalize and scale ---\n\
    mov.u64 %rd11, %rd7;\n\
RMS_NORM_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra RMS_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    add.u64 %rd13, %rd9, %rd12;\n\
    ld.global.f32 %f2, [%rd13];\n\
    mul.f32 %f2, %f2, %f6;\n\
    // gamma[j]\n\
    add.u64 %rd14, %rd3, %rd12;\n\
    ld.global.f32 %f7, [%rd14];\n\
    mul.f32 %f2, %f2, %f7;\n\
    // Store\n\
    add.u64 %rd14, %rd10, %rd12;\n\
    st.global.f32 [%rd14], %f2;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra RMS_NORM_LOOP;\n\
RMS_DONE: ret;\n\
}\0";
