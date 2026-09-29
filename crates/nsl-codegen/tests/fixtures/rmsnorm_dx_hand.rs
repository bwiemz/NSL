//! The hand-written fused RMSNorm input-gradient backward pair, frozen
//! (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::RMSNORM_DX_BWD_F32_PTX` and
//! `RMSNORM_DX_BWD_ADD_F32_PTX` as they stood when the kernels moved to
//! `nsl_kir::kernels::rmsnorm_dx`, verbatim (their comments aside; the PTX
//! `//` lines are kept). They are the reference side of
//! `rmsnorm_dx_kir_equivalence.rs` and are not compiled into anything else.

pub const RMSNORM_DX_BWD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_dx_bwd_f32(\n\
    .param .u64 dy, .param .u64 x, .param .u64 gamma,\n\
    .param .u64 dxout,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<24>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f<24>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 ssq[256];\n\
    .shared .f32 sdwx[256];\n\
    ld.param.u64 %rd1, [dy];\n\
    ld.param.u64 %rd2, [x];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [dxout];\n\
    ld.param.u64 %rd5, [rows];\n\
    ld.param.u64 %rd6, [cols];\n\
    ld.param.f32 %f1, [eps];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DX_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
    // row_base_bytes = row * cols * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
    add.u64 %rd11, %rd2, %rd9;\n\
    add.u64 %rd12, %rd4, %rd9;\n\
    // --- Pass 1: local S1=sum(x*x), S2=sum(dy*gamma*x) ---\n\
    mov.f32 %f2, 0f00000000;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd13, %rd8;\n\
DX_ACC:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_ACC_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f4, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f5, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f6, [%rd17];\n\
    fma.rn.f32 %f2, %f4, %f4, %f2;\n\
    mul.f32 %f7, %f5, %f6;\n\
    fma.rn.f32 %f3, %f7, %f4, %f3;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_ACC;\n\
DX_ACC_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r4, ssq;\n\
    add.u32 %r4, %r4, %r3;\n\
    st.shared.f32 [%r4], %f2;\n\
    mov.u32 %r5, sdwx;\n\
    add.u32 %r5, %r5, %r3;\n\
    st.shared.f32 [%r5], %f3;\n\
    bar.sync 0;\n\
    // Reduce both (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra DX_SKIP;\n\
    mov.u32 %r6, 1;\n\
    mov.u32 %r7, %ntid.x;\n\
DX_RLOOP:\n\
    setp.ge.u32 %p2, %r6, %r7;\n\
    @%p2 bra DX_RDONE;\n\
    mul.lo.u32 %r8, %r6, 4;\n\
    mov.u32 %r9, ssq;\n\
    add.u32 %r9, %r9, %r8;\n\
    ld.shared.f32 %f8, [%r9];\n\
    add.f32 %f2, %f2, %f8;\n\
    mov.u32 %r10, sdwx;\n\
    add.u32 %r10, %r10, %r8;\n\
    ld.shared.f32 %f9, [%r10];\n\
    add.f32 %f3, %f3, %f9;\n\
    add.u32 %r6, %r6, 1;\n\
    bra DX_RLOOP;\n\
DX_RDONE:\n\
    // rms_inv = rsqrt(S1/cols + eps)\n\
    cvt.rn.f32.u64 %f10, %rd6;\n\
    div.approx.f32 %f11, %f2, %f10;\n\
    add.f32 %f11, %f11, %f1;\n\
    rsqrt.approx.f32 %f11, %f11;\n\
    // Clamp rms_inv <= 1e12 (i.e. rms >= 1e-12), matching the CPU/tape-AD\n\
    // underflow guard so eps=0 + a near-zero row cannot inject +Inf/NaN.\n\
    min.f32 %f11, %f11, 0f5368D4A5;\n\
    st.shared.f32 [ssq], %f11;\n\
    st.shared.f32 [sdwx], %f3;\n\
DX_SKIP:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f12, [ssq];\n\
    ld.shared.f32 %f13, [sdwx];\n\
    // coeff = S2 * rms_inv^3 / cols\n\
    mul.f32 %f14, %f12, %f12;\n\
    mul.f32 %f14, %f14, %f12;\n\
    mul.f32 %f14, %f14, %f13;\n\
    cvt.rn.f32.u64 %f15, %rd6;\n\
    div.approx.f32 %f14, %f14, %f15;\n\
    // --- Pass 2: dx_j = gamma_j*dy_j*rms_inv - x_j*coeff ---\n\
    mov.u64 %rd13, %rd8;\n\
DX_WR:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f16, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f17, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f18, [%rd17];\n\
    mul.f32 %f19, %f18, %f17;\n\
    mul.f32 %f19, %f19, %f12;\n\
    mul.f32 %f20, %f16, %f14;\n\
    sub.f32 %f19, %f19, %f20;\n\
    add.u64 %rd18, %rd12, %rd14;\n\
    st.global.f32 [%rd18], %f19;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_WR;\n\
DX_DONE: ret;\n\
}\0";

pub const RMSNORM_DX_BWD_ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_dx_bwd_add_f32(\n\
    .param .u64 dy, .param .u64 x, .param .u64 gamma,\n\
    .param .u64 dxout, .param .u64 res,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<24>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f<24>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 ssq[256];\n\
    .shared .f32 sdwx[256];\n\
    ld.param.u64 %rd1, [dy];\n\
    ld.param.u64 %rd2, [x];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [dxout];\n\
    ld.param.u64 %rd19, [res];\n\
    ld.param.u64 %rd5, [rows];\n\
    ld.param.u64 %rd6, [cols];\n\
    ld.param.f32 %f1, [eps];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DX_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
    // row_base_bytes = row * cols * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
    add.u64 %rd11, %rd2, %rd9;\n\
    add.u64 %rd12, %rd4, %rd9;\n\
    add.u64 %rd20, %rd19, %rd9;\n\
    // --- Pass 1: local S1=sum(x*x), S2=sum(dy*gamma*x) ---\n\
    mov.f32 %f2, 0f00000000;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd13, %rd8;\n\
DX_ACC:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_ACC_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f4, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f5, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f6, [%rd17];\n\
    fma.rn.f32 %f2, %f4, %f4, %f2;\n\
    mul.f32 %f7, %f5, %f6;\n\
    fma.rn.f32 %f3, %f7, %f4, %f3;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_ACC;\n\
DX_ACC_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r4, ssq;\n\
    add.u32 %r4, %r4, %r3;\n\
    st.shared.f32 [%r4], %f2;\n\
    mov.u32 %r5, sdwx;\n\
    add.u32 %r5, %r5, %r3;\n\
    st.shared.f32 [%r5], %f3;\n\
    bar.sync 0;\n\
    // Reduce both (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra DX_SKIP;\n\
    mov.u32 %r6, 1;\n\
    mov.u32 %r7, %ntid.x;\n\
DX_RLOOP:\n\
    setp.ge.u32 %p2, %r6, %r7;\n\
    @%p2 bra DX_RDONE;\n\
    mul.lo.u32 %r8, %r6, 4;\n\
    mov.u32 %r9, ssq;\n\
    add.u32 %r9, %r9, %r8;\n\
    ld.shared.f32 %f8, [%r9];\n\
    add.f32 %f2, %f2, %f8;\n\
    mov.u32 %r10, sdwx;\n\
    add.u32 %r10, %r10, %r8;\n\
    ld.shared.f32 %f9, [%r10];\n\
    add.f32 %f3, %f3, %f9;\n\
    add.u32 %r6, %r6, 1;\n\
    bra DX_RLOOP;\n\
DX_RDONE:\n\
    // rms_inv = rsqrt(S1/cols + eps)\n\
    cvt.rn.f32.u64 %f10, %rd6;\n\
    div.approx.f32 %f11, %f2, %f10;\n\
    add.f32 %f11, %f11, %f1;\n\
    rsqrt.approx.f32 %f11, %f11;\n\
    // Clamp rms_inv <= 1e12 (i.e. rms >= 1e-12), matching the CPU/tape-AD\n\
    // underflow guard so eps=0 + a near-zero row cannot inject +Inf/NaN.\n\
    min.f32 %f11, %f11, 0f5368D4A5;\n\
    st.shared.f32 [ssq], %f11;\n\
    st.shared.f32 [sdwx], %f3;\n\
DX_SKIP:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f12, [ssq];\n\
    ld.shared.f32 %f13, [sdwx];\n\
    // coeff = S2 * rms_inv^3 / cols\n\
    mul.f32 %f14, %f12, %f12;\n\
    mul.f32 %f14, %f14, %f12;\n\
    mul.f32 %f14, %f14, %f13;\n\
    cvt.rn.f32.u64 %f15, %rd6;\n\
    div.approx.f32 %f14, %f14, %f15;\n\
    // --- Pass 2: dx_j = gamma_j*dy_j*rms_inv - x_j*coeff ---\n\
    mov.u64 %rd13, %rd8;\n\
DX_WR:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f16, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f17, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f18, [%rd17];\n\
    mul.f32 %f19, %f18, %f17;\n\
    mul.f32 %f19, %f19, %f12;\n\
    mul.f32 %f20, %f16, %f14;\n\
    sub.f32 %f19, %f19, %f20;\n\
    add.u64 %rd21, %rd20, %rd14;\n\
    ld.global.f32 %f21, [%rd21];\n\
    add.rn.f32 %f19, %f19, %f21;\n\
    add.u64 %rd18, %rd12, %rd14;\n\
    st.global.f32 [%rd18], %f19;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_WR;\n\
DX_DONE: ret;\n\
}\0";
