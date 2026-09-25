//! The hand-written kernels `nsl_scalar_mul_add_inplace_f32` and
//! `nsl_muon_scale_inv_frob_f32`, frozen (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::kernels::{SCALAR_MUL_ADD_INPLACE,MUON_SCALE_INV_FROB}_F32_PTX`
//! as they stood when the kernels moved to `nsl_kir::kernels::elementwise`,
//! verbatim (their comments aside). They are the reference side of
//! `elementwise_rn_kir_equivalence.rs` and are not compiled into anything
//! else.

pub const SCALAR_MUL_ADD_INPLACE_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_scalar_mul_add_inplace_f32(\n\
    .param .u64 m, .param .u64 g, .param .f32 s, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<8>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [m];\n\
    ld.param.u64 %rd2, [g];\n\
    ld.param.f32 %fs3, [s];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    mul.rn.f32 %fs1, %fs1, %fs3;\n\
    add.u64 %rd7, %rd1, %rd5;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    add.rn.f32 %fs2, %fs2, %fs1;\n\
    st.global.f32 [%rd7], %fs2;\n\
DONE: ret;\n\
}\0";

pub const MUON_SCALE_INV_FROB_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_muon_scale_inv_frob_f32(\n\
    .param .u64 x, .param .u64 c, .param .u64 stats, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [x];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.u64 %rd3, [stats];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    add.u64 %rd6, %rd3, 12;\n\
    ld.global.f32 %f1, [%rd6];\n\
    sqrt.rn.f32 %f1, %f1;\n\
    add.rn.f32 %f1, %f1, 0f33D6BF95;\n\
    mov.f32 %f2, 0f3F800000;\n\
    div.rn.f32 %f1, %f2, %f1;\n\
    shl.b64 %rd7, %rd5, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %f3, [%rd8];\n\
    mul.rn.f32 %f3, %f3, %f1;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    st.global.f32 [%rd8], %f3;\n\
DONE: ret;\n\
}\0";
