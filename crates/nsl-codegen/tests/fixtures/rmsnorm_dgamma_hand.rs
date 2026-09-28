//! The hand-written fused RMSNorm gamma-backward kernels, frozen
//! (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{RMSNORM_RINV_ROWS_F32_PTX,
//! RMSNORM_DGAMMA_F32_PTX}` as they stood when the kernels moved to
//! `nsl_kir::kernels::rmsnorm_dgamma`, verbatim (their comments aside; the
//! PTX `//` lines are kept). They are the reference side of
//! `rmsnorm_dgamma_kir_equivalence.rs` and are not compiled into anything
//! else.

pub const RMSNORM_RINV_ROWS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_rinv_rows_f32(\n\
    .param .u64 x, .param .u64 rinv,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<10>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [x];\n\
    ld.param.u64 %rd2, [rinv];\n\
    ld.param.u64 %rd3, [rows];\n\
    ld.param.u64 %rd4, [cols];\n\
    ld.param.f32 %f1, [eps];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %tid.x;\n\
    mul.lo.u32 %r4, %r1, %r2;\n\
    add.u32 %r4, %r4, %r3;\n\
    cvt.u64.u32 %rd5, %r4;\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra RINV_DONE;\n\
    // row base = row * cols * 4\n\
    mul.lo.u64 %rd6, %rd5, %rd4;\n\
    shl.b64 %rd6, %rd6, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    mov.f32 %f2, 0f00000000;\n\
    mov.u64 %rd8, 0;\n\
RINV_ACC:\n\
    setp.ge.u64 %p2, %rd8, %rd4;\n\
    @%p2 bra RINV_ACC_DONE;\n\
    shl.b64 %rd9, %rd8, 2;\n\
    add.u64 %rd10, %rd7, %rd9;\n\
    ld.global.f32 %f3, [%rd10];\n\
    fma.rn.f32 %f2, %f3, %f3, %f2;\n\
    add.u64 %rd8, %rd8, 1;\n\
    bra RINV_ACC;\n\
RINV_ACC_DONE:\n\
    cvt.rn.f32.u64 %f4, %rd4;\n\
    div.rn.f32 %f5, %f2, %f4;\n\
    add.rn.f32 %f6, %f5, %f1;\n\
    sqrt.rn.f32 %f7, %f6;\n\
    mov.f32 %f8, 0f3F800000;\n\
    div.rn.f32 %f9, %f8, %f7;\n\
    shl.b64 %rd11, %rd5, 2;\n\
    add.u64 %rd11, %rd2, %rd11;\n\
    st.global.f32 [%rd11], %f9;\n\
RINV_DONE:\n\
    ret;\n\
}\n\
\0";

pub const RMSNORM_DGAMMA_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_dgamma_f32(\n\
    .param .u64 dy, .param .u64 x, .param .u64 rinv,\n\
    .param .u64 dgamma,\n\
    .param .u64 rows, .param .u64 cols\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<8>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [dy];\n\
    ld.param.u64 %rd2, [x];\n\
    ld.param.u64 %rd3, [rinv];\n\
    ld.param.u64 %rd4, [dgamma];\n\
    ld.param.u64 %rd5, [rows];\n\
    ld.param.u64 %rd6, [cols];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %tid.x;\n\
    mul.lo.u32 %r4, %r1, %r2;\n\
    add.u32 %r4, %r4, %r3;\n\
    cvt.u64.u32 %rd7, %r4;\n\
    setp.ge.u64 %p1, %rd7, %rd6;\n\
    @%p1 bra DG_DONE;\n\
    // col offset bytes = j * 4; row stride bytes = cols * 4\n\
    shl.b64 %rd8, %rd7, 2;\n\
    shl.b64 %rd9, %rd6, 2;\n\
    add.u64 %rd10, %rd1, %rd8;\n\
    add.u64 %rd11, %rd2, %rd8;\n\
    mov.u64 %rd12, 0;\n\
    mov.f32 %f1, 0f00000000;\n\
DG_ACC:\n\
    setp.ge.u64 %p2, %rd12, %rd5;\n\
    @%p2 bra DG_ACC_DONE;\n\
    ld.global.f32 %f2, [%rd10];\n\
    ld.global.f32 %f3, [%rd11];\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd14, %rd3, %rd13;\n\
    ld.global.f32 %f4, [%rd14];\n\
    mul.rn.f32 %f5, %f2, %f3;\n\
    mul.rn.f32 %f6, %f5, %f4;\n\
    add.rn.f32 %f1, %f1, %f6;\n\
    add.u64 %rd10, %rd10, %rd9;\n\
    add.u64 %rd11, %rd11, %rd9;\n\
    add.u64 %rd12, %rd12, 1;\n\
    bra DG_ACC;\n\
DG_ACC_DONE:\n\
    add.u64 %rd15, %rd4, %rd8;\n\
    st.global.f32 [%rd15], %f1;\n\
DG_DONE:\n\
    ret;\n\
}\n\
\0";
