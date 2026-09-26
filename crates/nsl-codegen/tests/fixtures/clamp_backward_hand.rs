//! The hand-written clamp-backward kernel, frozen (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::kernels::CLAMP_BACKWARD_F32_PTX` as it stood when the
//! kernel moved to `nsl_kir::kernels::elementwise::build_clamp_backward`,
//! verbatim (its comment aside). It is the reference side of
//! `clamp_backward_kir_equivalence.rs` and is not compiled into anything else.

pub const CLAMP_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_clamp_backward_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out,\n\
    .param .f32 min_val, .param .f32 max_val, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<6>;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.f32 %fs4, [min_val];\n\
    ld.param.f32 %fs5, [max_val];\n\
    ld.param.u64 %rd4, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd5, %r3;\n\
    setp.ge.u64 %p1, %rd5, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd7, %rd1, %rd6;\n\
    ld.global.f32 %fs1, [%rd7];\n\
    add.u64 %rd7, %rd2, %rd6;\n\
    ld.global.f32 %fs2, [%rd7];\n\
    setp.ge.f32 %p1, %fs2, %fs4;\n\
    setp.le.f32 %p2, %fs2, %fs5;\n\
    and.pred %p1, %p1, %p2;\n\
    selp.f32 %fs3, %fs1, 0f00000000, %p1;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";
