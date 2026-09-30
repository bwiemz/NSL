//! The hand-written GEMM-chunked fused linear-CE chunk kernels, frozen
//! (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::LCE_CHUNK_STATS_F32_PTX` and
//! `LCE_CHUNK_DLOGITS_F32_PTX` as they stood when the kernels moved to
//! `nsl_kir::kernels::lce_chunk`, verbatim (their comments aside; the PTX
//! `//` lines are kept). They are the reference side of
//! `lce_chunk_kir_equivalence.rs` and are not compiled into anything else.

pub const LCE_CHUNK_STATS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_lce_chunk_stats_f32(\n\
    .param .u64 logits, .param .u64 bias, .param .u64 targets,\n\
    .param .u64 mstate, .param .u64 sstate, .param .u64 tlstate,\n\
    .param .u64 rows, .param .u64 cols, .param .u64 chunk_start,\n\
    .param .u32 has_bias\n\
) {\n\
    .reg .u64 %rd<24>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<14>;\n\
    .reg .s64 %tg<2>;\n\
    .reg .pred %p<8>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [logits];\n\
    ld.param.u64 %rd2, [bias];\n\
    ld.param.u64 %rd3, [targets];\n\
    ld.param.u64 %rd4, [mstate];\n\
    ld.param.u64 %rd5, [sstate];\n\
    ld.param.u64 %rd6, [tlstate];\n\
    ld.param.u64 %rd7, [rows];\n\
    ld.param.u64 %rd8, [cols];\n\
    ld.param.u64 %rd9, [chunk_start];\n\
    ld.param.u32 %r5, [has_bias];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd10, %r1;\n\
    setp.ge.u64 %p1, %rd10, %rd7;\n\
    @%p1 bra LCS_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd11, %r2;\n\
    // row_base_bytes = row * cols * 4\n\
    mul.lo.u64 %rd12, %rd10, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd1, %rd12;\n\
    // bias chunk base (bytes), used only when has_bias\n\
    shl.b64 %rd13, %rd9, 2;\n\
    add.u64 %rd13, %rd2, %rd13;\n\
    // --- Pass 1: block max of val_j = logits[row,j] (+ bias[cs+j]) ---\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd14, %rd11;\n\
LCS_MAX_LOOP:\n\
    setp.ge.u64 %p2, %rd14, %rd8;\n\
    @%p2 bra LCS_MAX_DONE;\n\
    shl.b64 %rd15, %rd14, 2;\n\
    add.u64 %rd16, %rd12, %rd15;\n\
    ld.global.f32 %f2, [%rd16];\n\
    setp.eq.u32 %p3, %r5, 0;\n\
    @%p3 bra LCS_MAX_NOB;\n\
    add.u64 %rd17, %rd13, %rd15;\n\
    ld.global.f32 %f3, [%rd17];\n\
    add.f32 %f2, %f2, %f3;\n\
LCS_MAX_NOB:\n\
    max.f32 %f1, %f1, %f2;\n\
    add.u64 %rd14, %rd14, 256;\n\
    bra LCS_MAX_LOOP;\n\
LCS_MAX_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    setp.ne.u32 %p4, %r2, 0;\n\
    @%p4 bra LCS_MAX_WAIT;\n\
    // thread 0: reduce max, merge with m_old, publish m_new\n\
    mov.u32 %r4, 1;\n\
LCS_MAX_RED:\n\
    setp.ge.u32 %p2, %r4, 256;\n\
    @%p2 bra LCS_MAX_RED_DONE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f4, [%r7];\n\
    max.f32 %f1, %f1, %f4;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LCS_MAX_RED;\n\
LCS_MAX_RED_DONE:\n\
    // m_old / s_old from global state (kept in thread-0 registers)\n\
    shl.b64 %rd18, %rd10, 2;\n\
    add.u64 %rd19, %rd4, %rd18;\n\
    ld.global.f32 %f5, [%rd19];\n\
    max.f32 %f6, %f5, %f1;\n\
    st.shared.f32 [sdata], %f6;\n\
LCS_MAX_WAIT:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f6, [sdata];\n\
    // --- Pass 2: block sum of exp(val_j - m_new) ---\n\
    mov.f32 %f7, 0f00000000;\n\
    mov.u64 %rd14, %rd11;\n\
LCS_SUM_LOOP:\n\
    setp.ge.u64 %p2, %rd14, %rd8;\n\
    @%p2 bra LCS_SUM_DONE;\n\
    shl.b64 %rd15, %rd14, 2;\n\
    add.u64 %rd16, %rd12, %rd15;\n\
    ld.global.f32 %f2, [%rd16];\n\
    setp.eq.u32 %p3, %r5, 0;\n\
    @%p3 bra LCS_SUM_NOB;\n\
    add.u64 %rd17, %rd13, %rd15;\n\
    ld.global.f32 %f3, [%rd17];\n\
    add.f32 %f2, %f2, %f3;\n\
LCS_SUM_NOB:\n\
    sub.f32 %f2, %f2, %f6;\n\
    mul.f32 %f2, %f2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f2, %f2;\n\
    add.f32 %f7, %f7, %f2;\n\
    add.u64 %rd14, %rd14, 256;\n\
    bra LCS_SUM_LOOP;\n\
LCS_SUM_DONE:\n\
    bar.sync 0;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f7;\n\
    bar.sync 0;\n\
    setp.ne.u32 %p4, %r2, 0;\n\
    @%p4 bra LCS_DONE;\n\
    // thread 0: reduce sum, rescale running s, store state\n\
    mov.u32 %r4, 1;\n\
LCS_SUM_RED:\n\
    setp.ge.u32 %p2, %r4, 256;\n\
    @%p2 bra LCS_SUM_RED_DONE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f4, [%r7];\n\
    add.f32 %f7, %f7, %f4;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LCS_SUM_RED;\n\
LCS_SUM_RED_DONE:\n\
    // s_new = s_old * exp(m_old - m_new) + chunk_sum\n\
    add.u64 %rd20, %rd5, %rd18;\n\
    ld.global.f32 %f8, [%rd20];\n\
    sub.f32 %f9, %f5, %f6;\n\
    mul.f32 %f9, %f9, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f9, %f9;\n\
    fma.rn.f32 %f7, %f8, %f9, %f7;\n\
    st.global.f32 [%rd20], %f7;\n\
    add.u64 %rd19, %rd4, %rd18;\n\
    st.global.f32 [%rd19], %f6;\n\
    // target logit gather: t in [chunk_start, chunk_start+cols)\n\
    shl.b64 %rd21, %rd10, 3;\n\
    add.u64 %rd21, %rd3, %rd21;\n\
    ld.global.s64 %tg0, [%rd21];\n\
    cvt.s64.u64 %tg1, %rd9;\n\
    setp.lt.s64 %p5, %tg0, %tg1;\n\
    @%p5 bra LCS_DONE;\n\
    sub.s64 %tg0, %tg0, %tg1;\n\
    cvt.s64.u64 %tg1, %rd8;\n\
    setp.ge.s64 %p6, %tg0, %tg1;\n\
    @%p6 bra LCS_DONE;\n\
    cvt.u64.s64 %rd22, %tg0;\n\
    shl.b64 %rd15, %rd22, 2;\n\
    add.u64 %rd16, %rd12, %rd15;\n\
    ld.global.f32 %f10, [%rd16];\n\
    setp.eq.u32 %p3, %r5, 0;\n\
    @%p3 bra LCS_TL_NOB;\n\
    add.u64 %rd17, %rd13, %rd15;\n\
    ld.global.f32 %f11, [%rd17];\n\
    add.f32 %f10, %f10, %f11;\n\
LCS_TL_NOB:\n\
    add.u64 %rd23, %rd6, %rd18;\n\
    st.global.f32 [%rd23], %f10;\n\
LCS_DONE: ret;\n\
}\0";

pub const LCE_CHUNK_DLOGITS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_lce_chunk_dlogits_f32(\n\
    .param .u64 logits, .param .u64 bias, .param .u64 targets,\n\
    .param .u64 lse, .param .u64 dbias,\n\
    .param .u64 rows, .param .u64 cols, .param .u64 chunk_start,\n\
    .param .f32 scale, .param .u32 has_bias\n\
) {\n\
    .reg .u64 %rd<22>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<10>;\n\
    .reg .s64 %tg<3>;\n\
    .reg .pred %p<8>;\n\
    ld.param.u64 %rd1, [logits];\n\
    ld.param.u64 %rd2, [bias];\n\
    ld.param.u64 %rd3, [targets];\n\
    ld.param.u64 %rd4, [lse];\n\
    ld.param.u64 %rd5, [dbias];\n\
    ld.param.u64 %rd6, [rows];\n\
    ld.param.u64 %rd7, [cols];\n\
    ld.param.u64 %rd8, [chunk_start];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u32 %r4, [has_bias];\n\
    // idx = ctaid.x * ntid.x + tid.x; stride = ntid.x * nctaid.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %tid.x;\n\
    mul.lo.u32 %r5, %r1, %r2;\n\
    add.u32 %r5, %r5, %r3;\n\
    cvt.u64.u32 %rd9, %r5;\n\
    mov.u32 %r1, %nctaid.x;\n\
    mul.lo.u32 %r5, %r1, %r2;\n\
    cvt.u64.u32 %rd10, %r5;\n\
    mul.lo.u64 %rd11, %rd6, %rd7;\n\
LCD_LOOP:\n\
    setp.ge.u64 %p1, %rd9, %rd11;\n\
    @%p1 bra LCD_DONE;\n\
    // r = idx / cols; j = idx % cols\n\
    div.u64 %rd12, %rd9, %rd7;\n\
    mul.lo.u64 %rd13, %rd12, %rd7;\n\
    sub.u64 %rd13, %rd9, %rd13;\n\
    // val = logits[idx] (+ bias[chunk_start + j])\n\
    shl.b64 %rd14, %rd9, 2;\n\
    add.u64 %rd14, %rd1, %rd14;\n\
    ld.global.f32 %f2, [%rd14];\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra LCD_NOB;\n\
    add.u64 %rd15, %rd8, %rd13;\n\
    shl.b64 %rd15, %rd15, 2;\n\
    add.u64 %rd15, %rd2, %rd15;\n\
    ld.global.f32 %f3, [%rd15];\n\
    add.f32 %f2, %f2, %f3;\n\
LCD_NOB:\n\
    // t = targets[r]; valid = t >= 0\n\
    shl.b64 %rd16, %rd12, 3;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
    ld.global.s64 %tg0, [%rd16];\n\
    setp.lt.s64 %p3, %tg0, 0;\n\
    @!%p3 bra LCD_VALID;\n\
    mov.f32 %f6, 0f00000000;\n\
    bra LCD_STORE;\n\
LCD_VALID:\n\
    // p = exp(val - lse[r])\n\
    shl.b64 %rd17, %rd12, 2;\n\
    add.u64 %rd17, %rd4, %rd17;\n\
    ld.global.f32 %f4, [%rd17];\n\
    sub.f32 %f2, %f2, %f4;\n\
    mul.f32 %f2, %f2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f5, %f2;\n\
    // onehot subtract\n\
    add.u64 %rd18, %rd8, %rd13;\n\
    cvt.s64.u64 %tg1, %rd18;\n\
    setp.ne.s64 %p4, %tg0, %tg1;\n\
    @%p4 bra LCD_SCALE;\n\
    mov.f32 %f7, 0f3F800000;\n\
    sub.f32 %f5, %f5, %f7;\n\
LCD_SCALE:\n\
    mul.f32 %f6, %f5, %f1;\n\
LCD_STORE:\n\
    st.global.f32 [%rd14], %f6;\n\
    // dbias[chunk_start + j] += dl (bias configs only)\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra LCD_NEXT;\n\
    add.u64 %rd19, %rd8, %rd13;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd5, %rd19;\n\
    red.global.add.f32 [%rd19], %f6;\n\
LCD_NEXT:\n\
    add.u64 %rd9, %rd9, %rd10;\n\
    bra LCD_LOOP;\n\
LCD_DONE: ret;\n\
}\0";
