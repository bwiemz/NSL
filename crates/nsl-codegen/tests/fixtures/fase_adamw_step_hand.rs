//! The hand-written kernel `nsl_fase_fused_adamw_step_f32`, frozen
//! (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::kernels::FASE_FUSED_ADAMW_STEP_F32_PTX` as it stood
//! when the kernel moved to `nsl_kir::kernels::optim`, verbatim. It is the
//! reference side of `fase_adamw_step_kir_equivalence.rs` and is not
//! compiled into anything else.

pub const FASE_FUSED_ADAMW_STEP_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_fase_fused_adamw_step_f32(\n\
    .param .u64 theta, .param .u64 m, .param .u64 v, .param .u64 mp, .param .u64 n,\n\
    .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,\n\
    .param .f32 eps, .param .f32 neg_lr, .param .f32 neg_lr_wd,\n\
    .param .f32 bc1, .param .f32 bc2, .param .u32 has_wd\n\
) {\n\
    .reg .u32 %r<6>;\n\
    .reg .u64 %rd<10>;\n\
    .reg .f32 %fs<16>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [theta];\n\
    ld.param.u64 %rd2, [m];\n\
    ld.param.u64 %rd3, [v];\n\
    ld.param.u64 %rd4, [mp];\n\
    ld.param.u64 %rd5, [n];\n\
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
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %fs10, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs11, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs12, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs13, [%rd8];\n\
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
    @%p2 bra SKIPWD;\n\
    mul.rn.f32 %fs15, %fs10, %fs7;\n\
    add.rn.f32 %fs14, %fs14, %fs15;\n\
SKIPWD:\n\
    add.rn.f32 %fs10, %fs10, %fs14;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    st.global.f32 [%rd8], %fs10;\n\
DONE: ret;\n\
}\0";
