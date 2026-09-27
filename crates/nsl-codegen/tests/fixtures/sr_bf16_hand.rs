//! The hand-written SR-BF16 kernels, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::kernels::{FASE_FUSED_ADAMW_STEP_BF16SR_PTX,
//! SR_BF16_ROUND_PROBE_PTX}` as they stood when the kernels moved to
//! `nsl_kir::kernels::optim::{build_fase_adamw_step_bf16sr,
//! build_sr_bf16_round_probe}`, verbatim (their comments aside; the PTX
//! `//` lines inside the step are kept). They are the reference side of
//! `sr_bf16_kir_equivalence.rs` and are not compiled into anything else.

pub const FASE_FUSED_ADAMW_STEP_BF16SR_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_fase_fused_adamw_step_bf16sr(\n\
    .param .u64 theta, .param .u64 m, .param .u64 v, .param .u64 mp, .param .u64 n,\n\
    .param .f32 b1, .param .f32 omb1, .param .f32 b2, .param .f32 omb2,\n\
    .param .f32 eps, .param .f32 neg_lr, .param .f32 neg_lr_wd,\n\
    .param .f32 bc1, .param .f32 bc2, .param .u32 has_wd,\n\
    .param .u64 sr_key, .param .u64 sr_ctr_base\n\
) {\n\
    .reg .u32 %r<12>;\n\
    .reg .u64 %rd<16>;\n\
    .reg .f32 %fs<16>;\n\
    .reg .b16 %rs<3>;\n\
    .reg .pred %p<5>;\n\
    ld.param.u64 %rd1, [theta];\n\
    ld.param.u64 %rd2, [m];\n\
    ld.param.u64 %rd3, [v];\n\
    ld.param.u64 %rd4, [mp];\n\
    ld.param.u64 %rd5, [n];\n\
    ld.param.f32 %fs1, [b1];\n\
    ld.param.f32 %fs2, [omb1];\n\
    ld.param.f32 %fs3, [b2];\n\
    ld.param.f32 %fs4, [omb2];\n\
    ld.param.f32 %fs5, [eps];\n\
    ld.param.f32 %fs6, [neg_lr];\n\
    ld.param.f32 %fs7, [neg_lr_wd];\n\
    ld.param.f32 %fs8, [bc1];\n\
    ld.param.f32 %fs9, [bc2];\n\
    ld.param.u32 %r4, [has_wd];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    // theta: bf16 load, widen exactly (bits<<16)\n\
    shl.b64 %rd9, %rd6, 1;\n\
    add.u64 %rd8, %rd1, %rd9;\n\
    ld.global.u16 %rs1, [%rd8];\n\
    cvt.u32.u16 %r5, %rs1;\n\
    shl.b32 %r5, %r5, 16;\n\
    mov.b32 %fs10, %r5;\n\
    // m/v/mp: f32 buffers, byte offset = idx*4\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs11, [%rd8];\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs12, [%rd8];\n\
    add.u64 %rd8, %rd4, %rd7;\n\
    ld.global.f32 %fs13, [%rd8];\n\
    mul.rn.f32 %fs14, %fs11, %fs1;\n\
    mul.rn.f32 %fs15, %fs13, %fs2;\n\
    add.rn.f32 %fs11, %fs14, %fs15;\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    st.global.f32 [%rd8], %fs11;\n\
    mul.rn.f32 %fs14, %fs13, %fs13;\n\
    mul.rn.f32 %fs14, %fs14, %fs4;\n\
    mul.rn.f32 %fs12, %fs12, %fs3;\n\
    add.rn.f32 %fs12, %fs12, %fs14;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    st.global.f32 [%rd8], %fs12;\n\
    mul.rn.f32 %fs14, %fs11, %fs8;\n\
    mul.rn.f32 %fs15, %fs12, %fs9;\n\
    sqrt.rn.f32 %fs15, %fs15;\n\
    add.rn.f32 %fs15, %fs15, %fs5;\n\
    div.approx.f32 %fs14, %fs14, %fs15;\n\
    mul.rn.f32 %fs14, %fs14, %fs6;\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra SKIPWD;\n\
    mul.rn.f32 %fs15, %fs10, %fs7;\n\
    add.rn.f32 %fs14, %fs14, %fs15;\n\
SKIPWD:\n\
    add.rn.f32 %fs10, %fs10, %fs14;\n\
    // ---- SR tail: theta_new (f32 in %fs10) -> bf16 bits in %r8 ----\n\
    mov.b32 %r5, %fs10;\n\
    and.b32 %r9, %r5, 0x80000000;\n\
    shr.u32 %r9, %r9, 16;\n\
    and.b32 %r6, %r5, 0x7f800000;\n\
    setp.eq.u32 %p3, %r6, 0x7f800000;\n\
    @%p3 bra SPECIAL;\n\
    // dither = low 16 bits of splitmix64(sr_key, sr_ctr_base + idx)\n\
    ld.param.u64 %rd10, [sr_key];\n\
    ld.param.u64 %rd11, [sr_ctr_base];\n\
    add.u64 %rd11, %rd11, %rd6;\n\
    mov.u64 %rd12, 0x9E3779B97F4A7C15;\n\
    mul.lo.u64 %rd13, %rd11, %rd12;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shr.u64 %rd14, %rd13, 30;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0xBF58476D1CE4E5B9;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 27;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0x94D049BB133111EB;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 31;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    cvt.u32.u64 %r7, %rd13;\n\
    and.b32 %r7, %r7, 0xffff;\n\
    add.u32 %r6, %r5, %r7;\n\
    and.b32 %r7, %r6, 0x7f800000;\n\
    setp.eq.u32 %p4, %r7, 0x7f800000;\n\
    @%p4 bra SATURATE;\n\
    shr.u32 %r8, %r6, 16;\n\
    bra STORE;\n\
SATURATE:\n\
    or.b32 %r8, %r9, 0x7f7f;\n\
    bra STORE;\n\
SPECIAL:\n\
    and.b32 %r6, %r5, 0x007fffff;\n\
    setp.ne.u32 %p4, %r6, 0;\n\
    @%p4 bra QNAN;\n\
    or.b32 %r8, %r9, 0x7f80;\n\
    bra STORE;\n\
QNAN:\n\
    or.b32 %r8, %r9, 0x7fc0;\n\
STORE:\n\
    add.u64 %rd8, %rd1, %rd9;\n\
    cvt.u16.u32 %rs2, %r8;\n\
    st.global.u16 [%rd8], %rs2;\n\
DONE: ret;\n\
}\0";

pub const SR_BF16_ROUND_PROBE_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sr_bf16_round_probe(\n\
    .param .u64 src, .param .u64 dst, .param .u64 n,\n\
    .param .u64 sr_key, .param .u64 sr_ctr_base\n\
) {\n\
    .reg .u32 %r<12>;\n\
    .reg .u64 %rd<16>;\n\
    .reg .b16 %rs<3>;\n\
    .reg .pred %p<5>;\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd5, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.u32 %r5, [%rd8];\n\
    and.b32 %r9, %r5, 0x80000000;\n\
    shr.u32 %r9, %r9, 16;\n\
    and.b32 %r6, %r5, 0x7f800000;\n\
    setp.eq.u32 %p3, %r6, 0x7f800000;\n\
    @%p3 bra SPECIAL;\n\
    ld.param.u64 %rd10, [sr_key];\n\
    ld.param.u64 %rd11, [sr_ctr_base];\n\
    add.u64 %rd11, %rd11, %rd6;\n\
    mov.u64 %rd12, 0x9E3779B97F4A7C15;\n\
    mul.lo.u64 %rd13, %rd11, %rd12;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shr.u64 %rd14, %rd13, 30;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0xBF58476D1CE4E5B9;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 27;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    mov.u64 %rd12, 0x94D049BB133111EB;\n\
    mul.lo.u64 %rd13, %rd13, %rd12;\n\
    shr.u64 %rd14, %rd13, 31;\n\
    xor.b64 %rd13, %rd13, %rd14;\n\
    cvt.u32.u64 %r7, %rd13;\n\
    and.b32 %r7, %r7, 0xffff;\n\
    add.u32 %r6, %r5, %r7;\n\
    and.b32 %r7, %r6, 0x7f800000;\n\
    setp.eq.u32 %p4, %r7, 0x7f800000;\n\
    @%p4 bra SATURATE;\n\
    shr.u32 %r8, %r6, 16;\n\
    bra STORE;\n\
SATURATE:\n\
    or.b32 %r8, %r9, 0x7f7f;\n\
    bra STORE;\n\
SPECIAL:\n\
    and.b32 %r6, %r5, 0x007fffff;\n\
    setp.ne.u32 %p4, %r6, 0;\n\
    @%p4 bra QNAN;\n\
    or.b32 %r8, %r9, 0x7f80;\n\
    bra STORE;\n\
QNAN:\n\
    or.b32 %r8, %r9, 0x7fc0;\n\
STORE:\n\
    shl.b64 %rd9, %rd6, 1;\n\
    add.u64 %rd8, %rd2, %rd9;\n\
    cvt.u16.u32 %rs2, %r8;\n\
    st.global.u16 [%rd8], %rs2;\n\
DONE: ret;\n\
}\0";
