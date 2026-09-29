//! The hand-written GPU cross-entropy backward kernels, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{CE_BWD_COUNT_F32_PTX,
//! CE_BWD_FINISH_F32_PTX}` as they stood when the kernels moved to
//! `nsl_kir::kernels::ce_bwd`, verbatim (their comments aside). They are the
//! reference side of `ce_bwd_kir_equivalence.rs` and are not compiled into
//! anything else.

pub const CE_BWD_COUNT_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_ce_bwd_count_f32(\n\
    .param .u64 tgt, .param .u32 n, .param .u32 tgt_i32, .param .u64 scratch\n\
) {\n\
    .reg .u64 %rd<8>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f<4>;\n\
    .reg .s32 %s<4>;\n\
    .reg .pred %p<4>;\n\
    .shared .u32 cnt[256];\n\
    ld.param.u64 %rd1, [tgt];\n\
    ld.param.u32 %r1, [n];\n\
    ld.param.u32 %r2, [tgt_i32];\n\
    ld.param.u64 %rd2, [scratch];\n\
    mov.u32 %r3, %tid.x;\n\
    mov.u32 %r4, 0;\n\
    mov.u32 %r5, %r3;\n\
    setp.ne.u32 %p3, %r2, 0;\n\
CEC_LOOP:\n\
    setp.ge.u32 %p1, %r5, %r1;\n\
    @%p1 bra CEC_RED;\n\
    cvt.u64.u32 %rd3, %r5;\n\
    shl.b64 %rd4, %rd3, 2;\n\
    add.u64 %rd5, %rd1, %rd4;\n\
    @%p3 bra CEC_I32;\n\
    ld.global.f32 %f1, [%rd5];\n\
    cvt.rzi.s32.f32 %s1, %f1;\n\
    bra CEC_TEST;\n\
CEC_I32:\n\
    ld.global.s32 %s1, [%rd5];\n\
CEC_TEST:\n\
    setp.lt.s32 %p2, %s1, 0;\n\
    @%p2 bra CEC_NEXT;\n\
    add.u32 %r4, %r4, 1;\n\
CEC_NEXT:\n\
    add.u32 %r5, %r5, 256;\n\
    bra CEC_LOOP;\n\
CEC_RED:\n\
    mul.lo.u32 %r6, %r3, 4;\n\
    mov.u32 %r7, cnt;\n\
    add.u32 %r7, %r7, %r6;\n\
    st.shared.u32 [%r7], %r4;\n\
    bar.sync 0;\n\
    mov.u32 %r8, 128;\n\
CEC_TREE:\n\
    setp.lt.u32 %p1, %r8, 1;\n\
    @%p1 bra CEC_DONE;\n\
    setp.ge.u32 %p2, %r3, %r8;\n\
    @%p2 bra CEC_SKIP;\n\
    mov.u32 %r7, cnt;\n\
    mul.lo.u32 %r6, %r3, 4;\n\
    add.u32 %r9, %r7, %r6;\n\
    add.u32 %r10, %r3, %r8;\n\
    mul.lo.u32 %r10, %r10, 4;\n\
    add.u32 %r11, %r7, %r10;\n\
    ld.shared.u32 %r6, [%r9];\n\
    ld.shared.u32 %r10, [%r11];\n\
    add.u32 %r6, %r6, %r10;\n\
    st.shared.u32 [%r9], %r6;\n\
CEC_SKIP:\n\
    bar.sync 0;\n\
    shr.u32 %r8, %r8, 1;\n\
    bra CEC_TREE;\n\
CEC_DONE:\n\
    setp.ne.u32 %p1, %r3, 0;\n\
    @%p1 bra CEC_EXIT;\n\
    mov.u32 %r7, cnt;\n\
    ld.shared.u32 %r6, [%r7];\n\
    max.u32 %r6, %r6, 1;\n\
    cvt.rn.f32.u32 %f2, %r6;\n\
    st.global.f32 [%rd2], %f2;\n\
CEC_EXIT:\n\
    ret;\n\
}\0";

pub const CE_BWD_FINISH_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_ce_bwd_finish_f32(\n\
    .param .u64 sm, .param .u64 tgt, .param .u64 scratch, .param .u64 gop,\n\
    .param .f32 go_imm, .param .u32 go_mode, .param .u32 tgt_i32,\n\
    .param .u32 total, .param .u32 cols\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f<10>;\n\
    .reg .s32 %s<4>;\n\
    .reg .pred %p<6>;\n\
    ld.param.u64 %rd1, [sm];\n\
    ld.param.u64 %rd2, [tgt];\n\
    ld.param.u64 %rd3, [scratch];\n\
    ld.param.u64 %rd4, [gop];\n\
    ld.param.f32 %f1, [go_imm];\n\
    ld.param.u32 %r1, [go_mode];\n\
    ld.param.u32 %r2, [tgt_i32];\n\
    ld.param.u32 %r3, [total];\n\
    ld.param.u32 %r4, [cols];\n\
    mov.u32 %r5, %ctaid.x;\n\
    mov.u32 %r6, %ntid.x;\n\
    mul.lo.u32 %r7, %r5, %r6;\n\
    mov.u32 %r5, %tid.x;\n\
    add.u32 %r7, %r7, %r5;\n\
    setp.ge.u32 %p1, %r7, %r3;\n\
    @%p1 bra CEF_EXIT;\n\
    div.u32 %r8, %r7, %r4;\n\
    mul.lo.u32 %r9, %r8, %r4;\n\
    sub.u32 %r10, %r7, %r9;\n\
    cvt.u64.u32 %rd5, %r8;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    setp.ne.u32 %p2, %r2, 0;\n\
    @%p2 bra CEF_I32;\n\
    ld.global.f32 %f2, [%rd7];\n\
    cvt.rzi.s32.f32 %s1, %f2;\n\
    bra CEF_HAVE;\n\
CEF_I32:\n\
    ld.global.s32 %s1, [%rd7];\n\
CEF_HAVE:\n\
    cvt.u64.u32 %rd5, %r7;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd8, %rd1, %rd6;\n\
    setp.lt.s32 %p3, %s1, 0;\n\
    @!%p3 bra CEF_VALID;\n\
    mov.f32 %f3, 0f00000000;\n\
    st.global.f32 [%rd8], %f3;\n\
    bra CEF_EXIT;\n\
CEF_VALID:\n\
    ld.global.f32 %f4, [%rd8];\n\
    cvt.u32.s32 %r11, %s1;\n\
    setp.eq.u32 %p4, %r10, %r11;\n\
    @!%p4 bra CEF_SCALE;\n\
    mov.f32 %f5, 0f3F800000;\n\
    sub.rn.f32 %f4, %f4, %f5;\n\
CEF_SCALE:\n\
    mov.f32 %f6, %f1;\n\
    setp.ne.u32 %p5, %r1, 1;\n\
    @%p5 bra CEF_GO;\n\
    ld.global.f32 %f6, [%rd4];\n\
CEF_GO:\n\
    ld.global.f32 %f7, [%rd3];\n\
    div.rn.f32 %f8, %f6, %f7;\n\
    mul.rn.f32 %f4, %f4, %f8;\n\
    st.global.f32 [%rd8], %f4;\n\
CEF_EXIT:\n\
    ret;\n\
}\0";
