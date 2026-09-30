//! The hand-written flash-attention log-sum-exp kernels, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::FLASH_LSE_F32_PTX` and
//! `FLASH_LSE_GQA_F32_PTX` as they stood when the kernels moved to
//! `nsl_kir::kernels::flash_lse`, verbatim (their comments aside; the PTX
//! `//` lines are kept). They are the reference side of
//! `flash_lse_kir_equivalence.rs` and are not compiled into anything else.

pub const FLASH_LSE_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_flash_lse_f32(\n\
    .param .u64 q, .param .u64 k, .param .u64 lse,\n\
    .param .u64 total, .param .u64 seq, .param .u64 hd,\n\
    .param .f32 scale, .param .u64 causal\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<16>;\n\
    .reg .pred %p<8>;\n\
\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd1, %r1;\n\
\n\
    ld.param.u64 %rd2, [q];\n\
    ld.param.u64 %rd3, [k];\n\
    ld.param.u64 %rd4, [lse];\n\
    ld.param.u64 %rd5, [total];\n\
    ld.param.u64 %rd6, [seq];\n\
    ld.param.u64 %rd7, [hd];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u64 %rd8, [causal];\n\
\n\
    setp.ge.u64 %p1, %rd1, %rd5;\n\
    @%p1 bra LSE_DONE;\n\
\n\
    rem.u64 %rd9, %rd1, %rd6;\n\
    div.u64 %rd10, %rd1, %rd6;\n\
\n\
    mul.lo.u64 %rd11, %rd1, %rd7;\n\
    mul.lo.u64 %rd12, %rd10, %rd6;\n\
    mul.lo.u64 %rd12, %rd12, %rd7;\n\
\n\
    add.u64 %rd13, %rd9, 1;\n\
    setp.ne.u64 %p2, %rd8, 0;\n\
    selp.u64 %rd14, %rd13, %rd6, %p2;\n\
\n\
    shl.b64 %rd15, %rd11, 2;\n\
    add.u64 %rd15, %rd2, %rd15;\n\
    shl.b64 %rd16, %rd12, 2;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
\n\
    // Pass 1: row max\n\
    mov.f32 %f2, 0fFF800000;\n\
    mov.u64 %rd17, 0;\n\
LSE_MAX_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSE_MAX_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSE_MAX_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSE_MAX_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSE_MAX_D;\n\
LSE_MAX_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    setp.gt.f32 %p5, %f3, %f2;\n\
    @%p5 mov.f32 %f2, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSE_MAX_J;\n\
LSE_MAX_DONE:\n\
\n\
    // Pass 2: sum exp(score - max)\n\
    mov.f32 %f7, 0f00000000;\n\
    mov.u64 %rd17, 0;\n\
LSE_SUM_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSE_SUM_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSE_SUM_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSE_SUM_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSE_SUM_D;\n\
LSE_SUM_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    sub.f32 %f3, %f3, %f2;\n\
    mul.f32 %f3, %f3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f3, %f3;\n\
    add.f32 %f7, %f7, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSE_SUM_J;\n\
LSE_SUM_DONE:\n\
\n\
    lg2.approx.f32 %f8, %f7;\n\
    mul.f32 %f8, %f8, 0f3F317218;\n\
    add.f32 %f8, %f2, %f8;\n\
\n\
    shl.b64 %rd22, %rd1, 2;\n\
    add.u64 %rd22, %rd4, %rd22;\n\
    st.global.f32 [%rd22], %f8;\n\
LSE_DONE:\n\
    ret;\n\
}\n\
\0";

pub const FLASH_LSE_GQA_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_flash_lse_gqa_f32(\n\
    .param .u64 q, .param .u64 k, .param .u64 lse,\n\
    .param .u64 total, .param .u64 seq, .param .u64 hd,\n\
    .param .f32 scale, .param .u64 causal,\n\
    .param .u64 heads, .param .u64 kv_heads\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<16>;\n\
    .reg .pred %p<8>;\n\
\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd1, %r1;\n\
\n\
    ld.param.u64 %rd2, [q];\n\
    ld.param.u64 %rd3, [k];\n\
    ld.param.u64 %rd4, [lse];\n\
    ld.param.u64 %rd5, [total];\n\
    ld.param.u64 %rd6, [seq];\n\
    ld.param.u64 %rd7, [hd];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u64 %rd8, [causal];\n\
    ld.param.u64 %rd23, [heads];\n\
    ld.param.u64 %rd24, [kv_heads];\n\
\n\
    setp.ge.u64 %p1, %rd1, %rd5;\n\
    @%p1 bra LSEG_DONE;\n\
\n\
    rem.u64 %rd9, %rd1, %rd6;\n\
    div.u64 %rd10, %rd1, %rd6;\n\
\n\
    mul.lo.u64 %rd11, %rd1, %rd7;\n\
    // kbh = (bh / heads) * kv_heads + (bh % heads) / groups\n\
    div.u64 %rd25, %rd23, %rd24;\n\
    div.u64 %rd26, %rd10, %rd23;\n\
    rem.u64 %rd27, %rd10, %rd23;\n\
    div.u64 %rd27, %rd27, %rd25;\n\
    mul.lo.u64 %rd12, %rd26, %rd24;\n\
    add.u64 %rd12, %rd12, %rd27;\n\
    mul.lo.u64 %rd12, %rd12, %rd6;\n\
    mul.lo.u64 %rd12, %rd12, %rd7;\n\
\n\
    add.u64 %rd13, %rd9, 1;\n\
    setp.ne.u64 %p2, %rd8, 0;\n\
    selp.u64 %rd14, %rd13, %rd6, %p2;\n\
\n\
    shl.b64 %rd15, %rd11, 2;\n\
    add.u64 %rd15, %rd2, %rd15;\n\
    shl.b64 %rd16, %rd12, 2;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
\n\
    // Pass 1: row max\n\
    mov.f32 %f2, 0fFF800000;\n\
    mov.u64 %rd17, 0;\n\
LSEG_MAX_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSEG_MAX_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSEG_MAX_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSEG_MAX_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSEG_MAX_D;\n\
LSEG_MAX_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    setp.gt.f32 %p5, %f3, %f2;\n\
    @%p5 mov.f32 %f2, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSEG_MAX_J;\n\
LSEG_MAX_DONE:\n\
\n\
    // Pass 2: sum exp(score - max)\n\
    mov.f32 %f7, 0f00000000;\n\
    mov.u64 %rd17, 0;\n\
LSEG_SUM_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSEG_SUM_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSEG_SUM_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSEG_SUM_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSEG_SUM_D;\n\
LSEG_SUM_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    sub.f32 %f3, %f3, %f2;\n\
    mul.f32 %f3, %f3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f3, %f3;\n\
    add.f32 %f7, %f7, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSEG_SUM_J;\n\
LSEG_SUM_DONE:\n\
\n\
    lg2.approx.f32 %f8, %f7;\n\
    mul.f32 %f8, %f8, 0f3F317218;\n\
    add.f32 %f8, %f2, %f8;\n\
\n\
    shl.b64 %rd22, %rd1, 2;\n\
    add.u64 %rd22, %rd4, %rd22;\n\
    st.global.f32 [%rd22], %f8;\n\
LSEG_DONE:\n\
    ret;\n\
}\n\
\0";
