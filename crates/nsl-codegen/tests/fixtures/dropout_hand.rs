//! The hand-written kernel `nsl_dropout_f32`, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::DROPOUT_F32_PTX` as it stood when the
//! kernel moved to `nsl_kir::kernels::dropout`, verbatim. It is the
//! reference side of `dropout_kir_equivalence.rs` and is not compiled into
//! anything else.

pub const DROPOUT_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_dropout_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 mask,\n\
    .param .u64 len, .param .u32 threshold, .param .f32 scale, .param .u64 seed\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<3>;\n\
    // Global thread index\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd0, %r1;\n\
    // Load params\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [mask];\n\
    ld.param.u64 %rd4, [len];\n\
    ld.param.u32 %r3, [threshold];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u64 %rd5, [seed];\n\
    // Bounds check\n\
    setp.ge.u64 %p1, %rd0, %rd4;\n\
    @%p1 bra DROP_DONE;\n\
    // Philox-style hash: hash = (seed + idx) * 0x9E3779B9; hash ^= hash >> 16; hash *= 0x85EBCA6B\n\
    add.u64 %rd6, %rd5, %rd0;\n\
    cvt.u32.u64 %r4, %rd6;\n\
    mul.lo.u32 %r4, %r4, 0x9E3779B9;\n\
    shr.u32 %r5, %r4, 16;\n\
    xor.b32 %r4, %r4, %r5;\n\
    mul.lo.u32 %r4, %r4, 0x85EBCA6B;\n\
    shr.u32 %r5, %r4, 13;\n\
    xor.b32 %r4, %r4, %r5;\n\
    mul.lo.u32 %r4, %r4, 0xC2B2AE35;\n\
    shr.u32 %r5, %r4, 16;\n\
    xor.b32 %r4, %r4, %r5;\n\
    // Compare: keep = (hash < threshold)\n\
    setp.lt.u32 %p2, %r4, %r3;\n\
    // Load input\n\
    shl.b64 %rd7, %rd0, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %f2, [%rd8];\n\
    // Compute output and mask\n\
    selp.f32 %f3, %f1, 0f00000000, %p2;\n\
    mul.f32 %f2, %f2, %f3;\n\
    // mask = keep ? 1.0 : 0.0\n\
    selp.f32 %f3, 0f3F800000, 0f00000000, %p2;\n\
    // Store output\n\
    add.u64 %rd9, %rd2, %rd7;\n\
    st.global.f32 [%rd9], %f2;\n\
    // Store mask\n\
    add.u64 %rd9, %rd3, %rd7;\n\
    st.global.f32 [%rd9], %f3;\n\
DROP_DONE: ret;\n\
}\0";
