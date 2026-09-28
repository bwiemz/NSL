//! The hand-written deterministic sum kernels, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{DET_GLOBAL_SUM_F32_PTX,
//! DET_SUM_DIM_F32_PTX}` as they stood when the kernels moved to
//! `nsl_kir::kernels::det_sum`, verbatim (their comments aside; the PTX `//`
//! lines are kept). They are the reference side of
//! `det_sum_kir_equivalence.rs` and are not compiled into anything else.

pub const DET_GLOBAL_SUM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_det_global_sum_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 len\n\
) {\n\
    .reg .u64 %rd<6>;\n\
    .reg .f32 %f<3>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [len];\n\
    // acc = 0.0\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd4, 0;\n\
DET_SUM_LOOP:\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DET_SUM_DONE;\n\
    // Load inp[i]\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd5, %rd1, %rd5;\n\
    ld.global.f32 %f2, [%rd5];\n\
    // acc += inp[i] (deterministic order: 0, 1, 2, ...)\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd4, %rd4, 1;\n\
    bra DET_SUM_LOOP;\n\
DET_SUM_DONE:\n\
    st.global.f32 [%rd2], %f1;\n\
    ret;\n\
}\0";

pub const DET_SUM_DIM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_det_sum_dim_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 outer, .param .u64 reduce_size, .param .u64 inner\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f<3>;\n\
    .reg .pred %p<3>;\n\
    // tid = blockIdx.x (one thread per output element)\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd0, %r1;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [outer];\n\
    ld.param.u64 %rd4, [reduce_size];\n\
    ld.param.u64 %rd5, [inner];\n\
    // total_outputs = outer * inner\n\
    mul.lo.u64 %rd6, %rd3, %rd5;\n\
    setp.ge.u64 %p1, %rd0, %rd6;\n\
    @%p1 bra DET_SDIM_DONE;\n\
    // Compute o = tid / inner, i = tid % inner\n\
    div.u64 %rd7, %rd0, %rd5;\n\
    rem.u64 %rd8, %rd0, %rd5;\n\
    // Sequential accumulate: sum inp[o * reduce_size * inner + r * inner + i] for r in 0..reduce_size\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd9, 0;\n\
    // base = (o * reduce_size * inner + i) * 4\n\
    mul.lo.u64 %rd10, %rd7, %rd4;\n\
    mul.lo.u64 %rd10, %rd10, %rd5;\n\
    add.u64 %rd10, %rd10, %rd8;\n\
    // stride = inner\n\
DET_SDIM_LOOP:\n\
    setp.ge.u64 %p2, %rd9, %rd4;\n\
    @%p2 bra DET_SDIM_STORE;\n\
    // addr = inp + (base + r * inner) * 4\n\
    mul.lo.u64 %rd11, %rd9, %rd5;\n\
    add.u64 %rd11, %rd10, %rd11;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd12, %rd1, %rd11;\n\
    ld.global.f32 %f2, [%rd12];\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd9, %rd9, 1;\n\
    bra DET_SDIM_LOOP;\n\
DET_SDIM_STORE:\n\
    // out[tid] = acc\n\
    shl.b64 %rd13, %rd0, 2;\n\
    add.u64 %rd13, %rd2, %rd13;\n\
    st.global.f32 [%rd13], %f1;\n\
DET_SDIM_DONE: ret;\n\
}\0";
