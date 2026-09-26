//! The hand-written kernel `nsl_fase_fused_adamw_multi_f32`, frozen
//! (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::kernels::FASE_FUSED_ADAMW_MULTI_F32_PTX` as it stood
//! when the kernel moved to `nsl_kir::kernels::optim`, verbatim. It is the
//! reference side of `fase_adamw_multi_kir_equivalence.rs` and is not
//! compiled into anything else.

pub const FASE_FUSED_ADAMW_MULTI_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_fase_fused_adamw_multi_f32(\n\
    .param .u64 ttab, .param .u64 mtab, .param .u64 vtab, .param .u64 mptab,\n\
    .param .u64 ntab,\n\
    .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,\n\
    .param .f32 eps, .param .f32 neg_lr, .param .f32 neg_lr_wd,\n\
    .param .f32 bc1, .param .f32 bc2, .param .u32 has_wd,\n\
    .param .u64 bptab, .param .u64 bbtab, .param .f32 mp_scale\n\
) {\n\
    .reg .u32 %r<10>;\n\
    .reg .u64 %rd<12>;\n\
    .reg .f32 %fs<17>;\n\
    .reg .pred %p<4>;\n\
    ld.param.u64 %rd1, [ttab];\n\
    ld.param.u64 %rd2, [mtab];\n\
    ld.param.u64 %rd3, [vtab];\n\
    ld.param.u64 %rd4, [mptab];\n\
    ld.param.u64 %rd5, [ntab];\n\
    ld.param.f32 %fs1, [b1];\n\
    ld.param.f32 %fs2, [omb1];\n\
    ld.param.f32 %fs3, [b2];\n\
    ld.param.f32 %fs4, [omb2];\n\
    ld.param.f32 %fs5, [eps];\n\
    ld.param.f32 %fs6, [neg_lr];\n\
    ld.param.f32 %fs7, [neg_lr_wd];\n\
    ld.param.f32 %fs8, [bc1];\n\
    ld.param.f32 %fs9, [bc2];\n\
    ld.param.u32 %r4, [has_wd];\n\
    ld.param.u64 %rd10, [bptab];\n\
    ld.param.u64 %rd11, [bbtab];\n\
    ld.param.f32 %fs16, [mp_scale];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd10, %rd7;\n\
    ld.global.u32 %r5, [%rd8];\n\
    add.u64 %rd8, %rd11, %rd7;\n\
    ld.global.u32 %r7, [%rd8];\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r7, %r1;\n\
    cvt.u64.u32 %rd6, %r5;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd5, %rd7;\n\
    ld.global.u32 %r6, [%rd8];\n\
    setp.ge.u32 %p1, %r3, %r6;\n\
    @%p1 bra MDONE;\n\
    shl.b64 %rd7, %rd6, 3;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.u64 %rd1, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.u64 %rd2, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.u64 %rd3, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.u64 %rd4, [%rd8];\n\
    cvt.u64.u32 %rd6, %r3;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %fs10, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs11, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs12, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs13, [%rd8];\n\
    setp.eq.f32 %p3, %fs16, 0f3F800000;\n\
    @%p3 bra MNOSC;\n\
    mul.rn.f32 %fs13, %fs13, %fs16;\n\
MNOSC:\n\
    mul.rn.f32 %fs14, %fs11, %fs1;\n\
    mul.rn.f32 %fs15, %fs13, %fs2;\n\
    add.rn.f32 %fs11, %fs14, %fs15;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    st.global.f32 [%rd8], %fs11;\n\
    mul.rn.f32 %fs14, %fs13, %fs13;\n\
    mul.rn.f32 %fs14, %fs14, %fs4;\n\
    mul.rn.f32 %fs12, %fs12, %fs3;\n\
    add.rn.f32 %fs12, %fs12, %fs14;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    st.global.f32 [%rd8], %fs12;\n\
    mul.rn.f32 %fs14, %fs11, %fs8;\n\
    mul.rn.f32 %fs15, %fs12, %fs9;\n\
    sqrt.rn.f32 %fs15, %fs15;\n\
    add.rn.f32 %fs15, %fs15, %fs5;\n\
    div.approx.f32 %fs14, %fs14, %fs15;\n\
    mul.rn.f32 %fs14, %fs14, %fs6;\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra MSKIPWD;\n\
    mul.rn.f32 %fs15, %fs10, %fs7;\n\
    add.rn.f32 %fs14, %fs14, %fs15;\n\
MSKIPWD:\n\
    add.rn.f32 %fs10, %fs10, %fs14;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    st.global.f32 [%rd8], %fs10;\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    mov.f32 %fs14, 0f00000000;\n\
    st.global.f32 [%rd8], %fs14;\n\
MDONE: ret;\n\
}\0";
