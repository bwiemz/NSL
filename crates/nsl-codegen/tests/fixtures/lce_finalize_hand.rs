//! The hand-written fused linear cross-entropy finalize kernel, frozen
//! (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::LCE_FINALIZE_F32_PTX` as it stood when
//! the kernel moved to `nsl_kir::kernels::lce_finalize`, verbatim (its
//! comments aside; the PTX `//` lines are kept). It is the reference side of
//! `lce_finalize_kir_equivalence.rs` and is not compiled into anything else.

pub const LCE_FINALIZE_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_lce_finalize_f32(\n\
    .param .u64 mstate, .param .u64 sstate, .param .u64 tlstate,\n\
    .param .u64 targets, .param .u64 loss, .param .u64 lse,\n\
    .param .u64 rows\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<8>;\n\
    .reg .s64 %tg<2>;\n\
    .reg .pred %p<4>;\n\
    ld.param.u64 %rd1, [mstate];\n\
    ld.param.u64 %rd2, [sstate];\n\
    ld.param.u64 %rd3, [tlstate];\n\
    ld.param.u64 %rd4, [targets];\n\
    ld.param.u64 %rd5, [loss];\n\
    ld.param.u64 %rd6, [lse];\n\
    ld.param.u64 %rd7, [rows];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %tid.x;\n\
    mul.lo.u32 %r4, %r1, %r2;\n\
    add.u32 %r4, %r4, %r3;\n\
    cvt.u64.u32 %rd8, %r4;\n\
    mov.u32 %r1, %nctaid.x;\n\
    mul.lo.u32 %r4, %r1, %r2;\n\
    cvt.u64.u32 %rd9, %r4;\n\
LCF_LOOP:\n\
    setp.ge.u64 %p1, %rd8, %rd7;\n\
    @%p1 bra LCF_DONE;\n\
    shl.b64 %rd10, %rd8, 2;\n\
    add.u64 %rd11, %rd1, %rd10;\n\
    ld.global.f32 %f1, [%rd11];\n\
    add.u64 %rd12, %rd2, %rd10;\n\
    ld.global.f32 %f2, [%rd12];\n\
    // lse = m + log(s) = m + lg2(s) * ln(2)\n\
    lg2.approx.f32 %f2, %f2;\n\
    fma.rn.f32 %f3, %f2, 0f3F317218, %f1;\n\
    add.u64 %rd13, %rd6, %rd10;\n\
    st.global.f32 [%rd13], %f3;\n\
    // loss = valid ? lse - tl : 0\n\
    shl.b64 %rd14, %rd8, 3;\n\
    add.u64 %rd14, %rd4, %rd14;\n\
    ld.global.s64 %tg0, [%rd14];\n\
    setp.lt.s64 %p2, %tg0, 0;\n\
    mov.f32 %f5, 0f00000000;\n\
    @%p2 bra LCF_STORE;\n\
    add.u64 %rd15, %rd3, %rd10;\n\
    ld.global.f32 %f4, [%rd15];\n\
    sub.f32 %f5, %f3, %f4;\n\
LCF_STORE:\n\
    add.u64 %rd15, %rd5, %rd10;\n\
    st.global.f32 [%rd15], %f5;\n\
    add.u64 %rd8, %rd8, %rd9;\n\
    bra LCF_LOOP;\n\
LCF_DONE: ret;\n\
}\0";
