//! The hand-written integer dequantization kernels, frozen (new-roadmap
//! item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{DEQUANT_INT8_PER_HEAD_F32_PTX,
//! DEQUANT_INT8_PER_TOKEN_F32_PTX, DEQUANT_INT4_PER_GROUP_F32_PTX}` as they
//! stood when the kernels moved to `nsl_kir::kernels::dequant`, verbatim.
//! They are the reference side of `dequant_kir_equivalence.rs` and are not
//! compiled into anything else.

pub const DEQUANT_INT8_PER_HEAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_dequant_int8_per_head_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 scales,\n\
    .param .u64 n, .param .u64 head_stride\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .s16 %rs1;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [scales];\n\
    ld.param.u64 %rd4, [n];\n\
    ld.param.u64 %rd5, [head_stride];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra DQ8H_DONE;\n\
    // head_index = i / head_stride\n\
    div.u64 %rd7, %rd6, %rd5;\n\
    // Load scale for this head\n\
    shl.b64 %rd8, %rd7, 2;\n\
    add.u64 %rd8, %rd3, %rd8;\n\
    ld.global.f32 %f1, [%rd8];\n\
    // Load i8 value, convert to f32, multiply by scale\n\
    add.u64 %rd9, %rd1, %rd6;\n\
    ld.global.s8 %rs1, [%rd9];\n\
    cvt.rn.f32.s16 %f2, %rs1;\n\
    mul.f32 %f3, %f2, %f1;\n\
    // Store f32 result\n\
    shl.b64 %rd10, %rd6, 2;\n\
    add.u64 %rd10, %rd2, %rd10;\n\
    st.global.f32 [%rd10], %f3;\n\
DQ8H_DONE: ret;\n\
}\0";

pub const DEQUANT_INT8_PER_TOKEN_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_dequant_int8_per_token_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 scales,\n\
    .param .u64 n, .param .u64 head_stride, .param .u64 head_dim\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .s16 %rs1;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [scales];\n\
    ld.param.u64 %rd4, [n];\n\
    ld.param.u64 %rd5, [head_stride];\n\
    ld.param.u64 %rd6, [head_dim];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd7, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra DQ8T_DONE;\n\
    // token_index = (i % head_stride) / head_dim\n\
    rem.u64 %rd8, %rd7, %rd5;\n\
    div.u64 %rd8, %rd8, %rd6;\n\
    // Load scale\n\
    shl.b64 %rd9, %rd8, 2;\n\
    add.u64 %rd9, %rd3, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    // Load i8, convert, multiply\n\
    add.u64 %rd10, %rd1, %rd7;\n\
    ld.global.s8 %rs1, [%rd10];\n\
    cvt.rn.f32.s16 %f2, %rs1;\n\
    mul.f32 %f3, %f2, %f1;\n\
    // Store\n\
    shl.b64 %rd11, %rd7, 2;\n\
    add.u64 %rd11, %rd2, %rd11;\n\
    st.global.f32 [%rd11], %f3;\n\
DQ8T_DONE: ret;\n\
}\0";

pub const DEQUANT_INT4_PER_GROUP_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_dequant_int4_per_group_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 scales, .param .u64 zero_points,\n\
    .param .u64 n, .param .u64 group_size\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<5>;\n\
    .reg .u16 %rh1;\n\
    .reg .pred %p<3>;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [scales];\n\
    ld.param.u64 %rd4, [zero_points];\n\
    ld.param.u64 %rd5, [n];\n\
    ld.param.u64 %rd6, [group_size];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd7, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DQ4G_DONE;\n\
    // byte_idx = i / 2\n\
    shr.u64 %rd8, %rd7, 1;\n\
    add.u64 %rd9, %rd1, %rd8;\n\
    ld.global.u8 %rh1, [%rd9];\n\
    // Check if even (low nibble) or odd (high nibble)\n\
    and.b64 %rd10, %rd7, 1;\n\
    setp.ne.u64 %p2, %rd10, 0;\n\
    cvt.u32.u16 %r4, %rh1;\n\
    @%p2 bra DQ4G_HIGH;\n\
    and.b32 %r4, %r4, 15;\n\
    bra DQ4G_APPLY;\n\
DQ4G_HIGH:\n\
    shr.u32 %r4, %r4, 4;\n\
    and.b32 %r4, %r4, 15;\n\
DQ4G_APPLY:\n\
    cvt.rn.f32.u32 %f1, %r4;\n\
    // group = i / group_size\n\
    div.u64 %rd11, %rd7, %rd6;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    add.u64 %rd13, %rd3, %rd12;\n\
    ld.global.f32 %f2, [%rd13];\n\
    add.u64 %rd13, %rd4, %rd12;\n\
    ld.global.f32 %f3, [%rd13];\n\
    // output = nibble * scale + zero_point\n\
    fma.rn.f32 %f4, %f1, %f2, %f3;\n\
    shl.b64 %rd12, %rd7, 2;\n\
    add.u64 %rd12, %rd2, %rd12;\n\
    st.global.f32 [%rd12], %f4;\n\
DQ4G_DONE: ret;\n\
}\0";
