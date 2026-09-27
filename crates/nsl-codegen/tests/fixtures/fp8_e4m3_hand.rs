//! The hand-written kernel `nsl_dequant_fp8_e4m3_f32`, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::DEQUANT_FP8_E4M3_F32_PTX` as it stood
//! when the kernel moved to `nsl_kir::kernels::dequant::build_fp8_e4m3`,
//! verbatim. It is the reference side of `fp8_e4m3_kir_equivalence.rs` and is
//! not compiled into anything else.

pub const DEQUANT_FP8_E4M3_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_dequant_fp8_e4m3_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u64 %rd<8>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f1;\n\
    .reg .u16 %rh1;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DQFP8_DONE;\n\
    // Load u8 value\n\
    add.u64 %rd5, %rd1, %rd4;\n\
    ld.global.u8 %rh1, [%rd5];\n\
    cvt.u32.u16 %r4, %rh1;\n\
    // Extract sign (bit 7), exp (bits 6-3), mantissa (bits 2-0)\n\
    shr.u32 %r5, %r4, 7;\n\
    and.b32 %r5, %r5, 1;\n\
    shl.b32 %r5, %r5, 31;\n\
    shr.u32 %r6, %r4, 3;\n\
    and.b32 %r6, %r6, 15;\n\
    and.b32 %r7, %r4, 7;\n\
    // NaN: S.1111.111\n\
    and.b32 %r8, %r4, 127;\n\
    setp.eq.u32 %p2, %r8, 127;\n\
    @%p2 bra DQFP8_NAN;\n\
    // Zero and subnormals: exp == 0\n\
    setp.eq.u32 %p2, %r6, 0;\n\
    @%p2 bra DQFP8_SUB;\n\
    // Normal: sign<<31 | (exp-7+127)<<23 | mantissa<<20\n\
    add.u32 %r6, %r6, 120;\n\
    shl.b32 %r6, %r6, 23;\n\
    shl.b32 %r7, %r7, 20;\n\
    or.b32 %r8, %r5, %r6;\n\
    or.b32 %r8, %r8, %r7;\n\
    mov.b32 %f1, %r8;\n\
    bra DQFP8_STORE;\n\
DQFP8_SUB:\n\
    // m * 2^-9 is exact in f32; OR in the sign so zero keeps it\n\
    cvt.rn.f32.u32 %f1, %r7;\n\
    mul.f32 %f1, %f1, 0f3B000000;\n\
    mov.b32 %r8, %f1;\n\
    or.b32 %r8, %r8, %r5;\n\
    mov.b32 %f1, %r8;\n\
    bra DQFP8_STORE;\n\
DQFP8_NAN:\n\
    mov.b32 %f1, 0f7FC00000;\n\
DQFP8_STORE:\n\
    shl.b64 %rd6, %rd4, 2;\n\
    add.u64 %rd6, %rd2, %rd6;\n\
    st.global.f32 [%rd6], %f1;\n\
DQFP8_DONE: ret;\n\
}\0";
