//! The hand-written f64-accumulated sum of squares, frozen (new-roadmap item
//! 5).
//!
//! `nsl_runtime::cuda::fused_kernels::SUM_SQ_F64_ACC_F32_PTX` as it stood when
//! the kernel moved to `nsl_kir::kernels::sum_sq`, verbatim (its comments
//! aside; the PTX `//` lines are kept). It is the reference side of
//! `sum_sq_kir_equivalence.rs` and is not compiled into anything else.

pub const SUM_SQ_F64_ACC_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_sum_sq_f64_acc_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<16>;\n\
    .reg .f32 %f<4>;\n\
    .reg .f64 %fd<6>;\n\
    .reg .pred %p<4>;\n\
    .shared .align 8 .f64 ssq[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r2, %tid.x;\n\
    mov.u32 %r10, %ctaid.x;\n\
    mov.u32 %r11, %ntid.x;\n\
    mov.u32 %r12, %nctaid.x;\n\
    // start = ctaid*ntid + tid\n\
    mul.lo.u32 %r13, %r10, %r11;\n\
    add.u32 %r13, %r13, %r2;\n\
    cvt.u64.u32 %rd5, %r13;\n\
    // stride = ntid*nctaid\n\
    mul.lo.u32 %r14, %r11, %r12;\n\
    cvt.u64.u32 %rd7, %r14;\n\
    mov.f64 %fd1, 0d0000000000000000;\n\
SSQ_LOOP:\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra SSQ_REDUCE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd6, %rd1, %rd6;\n\
    ld.global.f32 %f1, [%rd6];\n\
    cvt.f64.f32 %fd2, %f1;\n\
    fma.rn.f64 %fd1, %fd2, %fd2, %fd1;\n\
    add.u64 %rd5, %rd5, %rd7;\n\
    bra SSQ_LOOP;\n\
SSQ_REDUCE:\n\
    mul.lo.u32 %r3, %r2, 8;\n\
    mov.u32 %r7, ssq;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f64 [%r7], %fd1;\n\
    bar.sync 0;\n\
    mov.u32 %r4, 128;\n\
SSQ_RED_LOOP:\n\
    setp.lt.u32 %p1, %r4, 1;\n\
    @%p1 bra SSQ_RED_DONE;\n\
    setp.ge.u32 %p2, %r2, %r4;\n\
    @%p2 bra SSQ_RED_SKIP;\n\
    mul.lo.u32 %r5, %r2, 8;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 8;\n\
    mov.u32 %r7, ssq;\n\
    add.u32 %r8, %r7, %r5;\n\
    ld.shared.f64 %fd3, [%r8];\n\
    add.u32 %r9, %r7, %r6;\n\
    ld.shared.f64 %fd4, [%r9];\n\
    add.f64 %fd3, %fd3, %fd4;\n\
    st.shared.f64 [%r8], %fd3;\n\
SSQ_RED_SKIP:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra SSQ_RED_LOOP;\n\
SSQ_RED_DONE:\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra SSQ_DONE;\n\
    ld.shared.f64 %fd5, [ssq];\n\
    // out[ctaid]\n\
    cvt.u64.u32 %rd8, %r10;\n\
    shl.b64 %rd8, %rd8, 3;\n\
    add.u64 %rd9, %rd2, %rd8;\n\
    st.global.f64 [%rd9], %fd5;\n\
SSQ_DONE: ret;\n\
}\0";
