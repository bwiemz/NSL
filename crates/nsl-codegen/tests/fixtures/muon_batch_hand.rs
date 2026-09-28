//! The hand-written batched Muon Newton-Schulz kernels, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{MUON_BATCH_MOM_F32_PTX,
//! MUON_BATCH_SUMSQ_F32_PTX, MUON_BATCH_PACK_F32_PTX,
//! MUON_BATCH_POLY_F32_PTX, MUON_BATCH_UPDATE_F32_PTX}` as they stood when
//! the kernels moved to `nsl_kir::kernels::muon_batch`, verbatim (their
//! comments aside; the PTX `//` lines are kept). They are the reference side
//! of `muon_batch_kir_equivalence.rs` and are not compiled into anything
//! else.

pub const MUON_BATCH_MOM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_muon_batch_mom_f32(\n\
    .param .u64 mtab, .param .u64 gtab, .param .f32 mu, .param .u32 n\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<6>;\n\
    .reg .pred %p<2>;\n\
    ld.param.u64 %rd1, [mtab];\n\
    ld.param.u64 %rd2, [gtab];\n\
    ld.param.f32 %f1, [mu];\n\
    ld.param.u32 %r1, [n];\n\
    mov.u32 %r2, %ctaid.x;\n\
    mov.u32 %r3, %ntid.x;\n\
    mov.u32 %r4, %tid.x;\n\
    mul.lo.u32 %r5, %r2, %r3;\n\
    add.u32 %r5, %r5, %r4;\n\
    setp.ge.u32 %p1, %r5, %r1;\n\
    @%p1 bra MBM_EXIT;\n\
    mov.u32 %r6, %ctaid.y;\n\
    cvt.u64.u32 %rd3, %r6;\n\
    shl.b64 %rd4, %rd3, 3;\n\
    add.u64 %rd5, %rd1, %rd4;\n\
    ld.global.u64 %rd6, [%rd5];\n\
    add.u64 %rd5, %rd2, %rd4;\n\
    ld.global.u64 %rd7, [%rd5];\n\
    cvt.u64.u32 %rd8, %r5;\n\
    shl.b64 %rd8, %rd8, 2;\n\
    add.u64 %rd9, %rd6, %rd8;\n\
    add.u64 %rd10, %rd7, %rd8;\n\
    ld.global.f32 %f2, [%rd9];\n\
    ld.global.f32 %f3, [%rd10];\n\
    mul.rn.f32 %f4, %f2, %f1;\n\
    add.rn.f32 %f5, %f4, %f3;\n\
    st.global.f32 [%rd9], %f5;\n\
MBM_EXIT:\n\
    ret;\n\
}\0";

pub const MUON_BATCH_SUMSQ_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_muon_batch_sumsq_f32(\n\
    .param .u64 mtab, .param .u64 gtab, .param .f32 mu,\n\
    .param .u32 nest, .param .u32 n, .param .u64 norms\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<14>;\n\
    .reg .f32 %f<10>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 mbss[256];\n\
    ld.param.u64 %rd1, [mtab];\n\
    ld.param.u64 %rd2, [gtab];\n\
    ld.param.f32 %f1, [mu];\n\
    ld.param.u32 %r1, [nest];\n\
    ld.param.u32 %r2, [n];\n\
    ld.param.u64 %rd3, [norms];\n\
    mov.u32 %r3, %ctaid.x;\n\
    mov.u32 %r4, %tid.x;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    shl.b64 %rd5, %rd4, 3;\n\
    add.u64 %rd6, %rd1, %rd5;\n\
    ld.global.u64 %rd7, [%rd6];\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    ld.global.u64 %rd8, [%rd6];\n\
    mov.f32 %f2, 0f00000000;\n\
    cvt.u64.u32 %rd9, %r4;\n\
    cvt.u64.u32 %rd10, %r2;\n\
    setp.ne.u32 %p3, %r1, 0;\n\
MBS_LOOP:\n\
    setp.ge.u64 %p1, %rd9, %rd10;\n\
    @%p1 bra MBS_RED;\n\
    shl.b64 %rd11, %rd9, 2;\n\
    add.u64 %rd12, %rd7, %rd11;\n\
    ld.global.f32 %f3, [%rd12];\n\
    mov.f32 %f4, %f3;\n\
    @!%p3 bra MBS_ACC;\n\
    add.u64 %rd13, %rd8, %rd11;\n\
    ld.global.f32 %f5, [%rd13];\n\
    mul.rn.f32 %f6, %f3, %f1;\n\
    add.rn.f32 %f4, %f5, %f6;\n\
MBS_ACC:\n\
    mul.rn.f32 %f7, %f4, %f4;\n\
    add.rn.f32 %f2, %f2, %f7;\n\
    add.u64 %rd9, %rd9, 256;\n\
    bra MBS_LOOP;\n\
MBS_RED:\n\
    mul.lo.u32 %r5, %r4, 4;\n\
    mov.u32 %r6, mbss;\n\
    add.u32 %r6, %r6, %r5;\n\
    st.shared.f32 [%r6], %f2;\n\
    bar.sync 0;\n\
    mov.u32 %r7, 128;\n\
MBS_TREE:\n\
    setp.lt.u32 %p1, %r7, 1;\n\
    @%p1 bra MBS_DONE;\n\
    setp.ge.u32 %p2, %r4, %r7;\n\
    @%p2 bra MBS_SKIP;\n\
    mov.u32 %r8, mbss;\n\
    mul.lo.u32 %r9, %r4, 4;\n\
    add.u32 %r10, %r8, %r9;\n\
    add.u32 %r11, %r4, %r7;\n\
    mul.lo.u32 %r11, %r11, 4;\n\
    add.u32 %r12, %r8, %r11;\n\
    ld.shared.f32 %f8, [%r10];\n\
    ld.shared.f32 %f9, [%r12];\n\
    add.rn.f32 %f8, %f8, %f9;\n\
    st.shared.f32 [%r10], %f8;\n\
MBS_SKIP:\n\
    bar.sync 0;\n\
    shr.u32 %r7, %r7, 1;\n\
    bra MBS_TREE;\n\
MBS_DONE:\n\
    setp.ne.u32 %p1, %r4, 0;\n\
    @%p1 bra MBS_EXIT;\n\
    mov.u32 %r13, mbss;\n\
    ld.shared.f32 %f2, [%r13];\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd3, %rd5;\n\
    st.global.f32 [%rd6], %f2;\n\
MBS_EXIT:\n\
    ret;\n\
}\0";

pub const MUON_BATCH_PACK_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_muon_batch_pack_f32(\n\
    .param .u64 mtab, .param .u64 gtab, .param .f32 mu, .param .u32 nest,\n\
    .param .u64 norms, .param .u64 ybase, .param .u32 r, .param .u32 c,\n\
    .param .u32 tr, .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<18>;\n\
    .reg .u32 %r<16>;\n\
    .reg .f32 %f<14>;\n\
    .reg .pred %p<4>;\n\
    ld.param.u64 %rd1, [mtab];\n\
    ld.param.u64 %rd2, [gtab];\n\
    ld.param.f32 %f1, [mu];\n\
    ld.param.u32 %r1, [nest];\n\
    ld.param.u64 %rd3, [norms];\n\
    ld.param.u64 %rd4, [ybase];\n\
    ld.param.u32 %r2, [r];\n\
    ld.param.u32 %r3, [c];\n\
    ld.param.u32 %r4, [tr];\n\
    ld.param.f32 %f2, [eps];\n\
    mov.u32 %r5, %ctaid.x;\n\
    mov.u32 %r6, %ntid.x;\n\
    mov.u32 %r7, %tid.x;\n\
    mul.lo.u32 %r8, %r5, %r6;\n\
    add.u32 %r8, %r8, %r7;\n\
    mul.lo.u32 %r9, %r2, %r3;\n\
    setp.ge.u32 %p1, %r8, %r9;\n\
    @%p1 bra MBP_EXIT;\n\
    mov.u32 %r10, %ctaid.y;\n\
    cvt.u64.u32 %rd5, %r10;\n\
    shl.b64 %rd6, %rd5, 3;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.u64 %rd8, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.u64 %rd9, [%rd7];\n\
    cvt.u64.u32 %rd10, %r8;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd12, %rd8, %rd11;\n\
    ld.global.f32 %f3, [%rd12];\n\
    mov.f32 %f4, %f3;\n\
    setp.ne.u32 %p2, %r1, 0;\n\
    @!%p2 bra MBP_NORM;\n\
    add.u64 %rd13, %rd9, %rd11;\n\
    ld.global.f32 %f5, [%rd13];\n\
    mul.rn.f32 %f6, %f3, %f1;\n\
    add.rn.f32 %f4, %f5, %f6;\n\
MBP_NORM:\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    ld.global.f32 %f7, [%rd7];\n\
    sqrt.rn.f32 %f8, %f7;\n\
    add.rn.f32 %f8, %f8, %f2;\n\
    mov.f32 %f9, 0f3F800000;\n\
    div.rn.f32 %f10, %f9, %f8;\n\
    mul.rn.f32 %f11, %f4, %f10;\n\
    // destination index: identity, or on-the-fly transpose col*r + row\n\
    mov.u32 %r11, %r8;\n\
    setp.eq.u32 %p3, %r4, 0;\n\
    @%p3 bra MBP_STORE;\n\
    div.u32 %r12, %r8, %r3;\n\
    mul.lo.u32 %r13, %r12, %r3;\n\
    sub.u32 %r14, %r8, %r13;\n\
    mul.lo.u32 %r11, %r14, %r2;\n\
    add.u32 %r11, %r11, %r12;\n\
MBP_STORE:\n\
    cvt.u64.u32 %rd14, %r9;\n\
    mul.lo.u64 %rd15, %rd5, %rd14;\n\
    cvt.u64.u32 %rd16, %r11;\n\
    add.u64 %rd15, %rd15, %rd16;\n\
    shl.b64 %rd15, %rd15, 2;\n\
    add.u64 %rd17, %rd4, %rd15;\n\
    st.global.f32 [%rd17], %f11;\n\
MBP_EXIT:\n\
    ret;\n\
}\0";

pub const MUON_BATCH_POLY_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_muon_batch_poly_f32(\n\
    .param .u64 abase, .param .u64 aabase, .param .f32 nsa,\n\
    .param .f32 nsb, .param .f32 nsc, .param .u32 rdim, .param .u32 r2\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<14>;\n\
    .reg .f32 %f<10>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [abase];\n\
    ld.param.u64 %rd2, [aabase];\n\
    ld.param.f32 %f1, [nsa];\n\
    ld.param.f32 %f2, [nsb];\n\
    ld.param.f32 %f3, [nsc];\n\
    ld.param.u32 %r1, [rdim];\n\
    ld.param.u32 %r2, [r2];\n\
    mov.u32 %r3, %ctaid.x;\n\
    mov.u32 %r4, %ntid.x;\n\
    mov.u32 %r5, %tid.x;\n\
    mul.lo.u32 %r6, %r3, %r4;\n\
    add.u32 %r6, %r6, %r5;\n\
    setp.ge.u32 %p1, %r6, %r2;\n\
    @%p1 bra MBY_EXIT;\n\
    mov.u32 %r7, %ctaid.y;\n\
    cvt.u64.u32 %rd3, %r7;\n\
    cvt.u64.u32 %rd4, %r2;\n\
    mul.lo.u64 %rd5, %rd3, %rd4;\n\
    cvt.u64.u32 %rd6, %r6;\n\
    add.u64 %rd5, %rd5, %rd6;\n\
    shl.b64 %rd5, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd5;\n\
    add.u64 %rd8, %rd2, %rd5;\n\
    ld.global.f32 %f4, [%rd7];\n\
    ld.global.f32 %f5, [%rd8];\n\
    mul.rn.f32 %f6, %f4, %f2;\n\
    mul.rn.f32 %f7, %f5, %f3;\n\
    add.rn.f32 %f8, %f6, %f7;\n\
    div.u32 %r8, %r6, %r1;\n\
    mul.lo.u32 %r9, %r8, %r1;\n\
    sub.u32 %r10, %r6, %r9;\n\
    setp.eq.u32 %p2, %r8, %r10;\n\
    @%p2 add.rn.f32 %f8, %f8, %f1;\n\
    st.global.f32 [%rd7], %f8;\n\
MBY_EXIT:\n\
    ret;\n\
}\0";

pub const MUON_BATCH_UPDATE_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_muon_batch_update_f32(\n\
    .param .u64 ptab, .param .u64 ybase, .param .u32 r, .param .u32 c,\n\
    .param .u32 tr, .param .f32 decay, .param .f32 step\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<16>;\n\
    .reg .f32 %f<8>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [ptab];\n\
    ld.param.u64 %rd2, [ybase];\n\
    ld.param.u32 %r1, [r];\n\
    ld.param.u32 %r2, [c];\n\
    ld.param.u32 %r3, [tr];\n\
    ld.param.f32 %f1, [decay];\n\
    ld.param.f32 %f2, [step];\n\
    mov.u32 %r4, %ctaid.x;\n\
    mov.u32 %r5, %ntid.x;\n\
    mov.u32 %r6, %tid.x;\n\
    mul.lo.u32 %r7, %r4, %r5;\n\
    add.u32 %r7, %r7, %r6;\n\
    mul.lo.u32 %r8, %r1, %r2;\n\
    setp.ge.u32 %p1, %r7, %r8;\n\
    @%p1 bra MBU_EXIT;\n\
    mov.u32 %r9, %ctaid.y;\n\
    cvt.u64.u32 %rd3, %r9;\n\
    shl.b64 %rd4, %rd3, 3;\n\
    add.u64 %rd5, %rd1, %rd4;\n\
    ld.global.u64 %rd6, [%rd5];\n\
    // source index in Y: identity, or transpose col*r + row\n\
    mov.u32 %r10, %r7;\n\
    setp.eq.u32 %p2, %r3, 0;\n\
    @%p2 bra MBU_LOAD;\n\
    div.u32 %r11, %r7, %r2;\n\
    mul.lo.u32 %r12, %r11, %r2;\n\
    sub.u32 %r13, %r7, %r12;\n\
    mul.lo.u32 %r10, %r13, %r1;\n\
    add.u32 %r10, %r10, %r11;\n\
MBU_LOAD:\n\
    cvt.u64.u32 %rd7, %r8;\n\
    mul.lo.u64 %rd8, %rd3, %rd7;\n\
    cvt.u64.u32 %rd9, %r10;\n\
    add.u64 %rd8, %rd8, %rd9;\n\
    shl.b64 %rd8, %rd8, 2;\n\
    add.u64 %rd10, %rd2, %rd8;\n\
    ld.global.f32 %f3, [%rd10];\n\
    cvt.u64.u32 %rd11, %r7;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd12, %rd6, %rd11;\n\
    ld.global.f32 %f4, [%rd12];\n\
    mul.rn.f32 %f5, %f4, %f1;\n\
    mul.rn.f32 %f6, %f2, %f3;\n\
    sub.rn.f32 %f7, %f5, %f6;\n\
    st.global.f32 [%rd12], %f7;\n\
MBU_EXIT:\n\
    ret;\n\
}\0";
