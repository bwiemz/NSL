//! The hand-written 2-D max pooling forward, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::MAXPOOL2D_F32_PTX` as it stood when the
//! kernel moved to `nsl_kir::kernels::maxpool`, verbatim (its comments
//! aside; the PTX `//` lines are kept). It is the reference side of
//! `maxpool_kir_equivalence.rs` and is not compiled into anything else.

pub const MAXPOOL2D_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_maxpool2d_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 argmax,\n\
    .param .u64 N, .param .u64 C, .param .u64 H, .param .u64 W,\n\
    .param .u64 kH, .param .u64 kW,\n\
    .param .u64 stride, .param .u64 padding,\n\
    .param .u64 H_out, .param .u64 W_out, .param .u64 total\n\
) {\n\
    .reg .u64 %rd<24>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f<3>;\n\
    .reg .pred %p<4>;\n\
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
    ld.param.u64 %rd3, [argmax];\n\
    ld.param.u64 %rd4, [N];\n\
    ld.param.u64 %rd5, [C];\n\
    ld.param.u64 %rd6, [H];\n\
    ld.param.u64 %rd7, [W];\n\
    ld.param.u64 %rd8, [kH];\n\
    ld.param.u64 %rd9, [kW];\n\
    ld.param.u64 %rd10, [stride];\n\
    ld.param.u64 %rd11, [padding];\n\
    ld.param.u64 %rd12, [H_out];\n\
    ld.param.u64 %rd13, [W_out];\n\
    ld.param.u64 %rd14, [total];\n\
    // Bounds check\n\
    setp.ge.u64 %p1, %rd0, %rd14;\n\
    @%p1 bra MP_DONE;\n\
    // Decompose flat index -> (n, c, oh, ow)\n\
    rem.u64 %rd15, %rd0, %rd13;\n\
    div.u64 %rd16, %rd0, %rd13;\n\
    rem.u64 %rd17, %rd16, %rd12;\n\
    div.u64 %rd18, %rd16, %rd12;\n\
    rem.u64 %rd19, %rd18, %rd5;\n\
    div.u64 %rd20, %rd18, %rd5;\n\
    // max_val = -inf, max_idx = 0\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd21, 0;\n\
    // Loop over kernel window\n\
    mov.u64 %rd22, 0;\n\
MP_KY:\n\
    setp.ge.u64 %p1, %rd22, %rd8;\n\
    @%p1 bra MP_WRITE;\n\
    mov.u64 %rd23, 0;\n\
MP_KX:\n\
    setp.ge.u64 %p1, %rd23, %rd9;\n\
    @%p1 bra MP_KY_INC;\n\
    // ih = oh * stride + ky, iw = ow * stride + kx\n\
    mul.lo.u64 %rd16, %rd17, %rd10;\n\
    add.u64 %rd16, %rd16, %rd22;\n\
    mul.lo.u64 %rd18, %rd15, %rd10;\n\
    add.u64 %rd18, %rd18, %rd23;\n\
    // Padding check\n\
    setp.lt.u64 %p2, %rd16, %rd11;\n\
    @%p2 bra MP_KX_INC;\n\
    setp.lt.u64 %p2, %rd18, %rd11;\n\
    @%p2 bra MP_KX_INC;\n\
    sub.u64 %rd16, %rd16, %rd11;\n\
    sub.u64 %rd18, %rd18, %rd11;\n\
    setp.ge.u64 %p2, %rd16, %rd6;\n\
    @%p2 bra MP_KX_INC;\n\
    setp.ge.u64 %p2, %rd18, %rd7;\n\
    @%p2 bra MP_KX_INC;\n\
    // input_idx = n*C*H*W + c*H*W + ih*W + iw\n\
    mul.lo.u64 %rd16, %rd20, %rd5;\n\
    add.u64 %rd16, %rd16, %rd19;\n\
    mul.lo.u64 %rd16, %rd16, %rd6;\n\
    // ih was computed above but we used %rd16 -- recompute\n\
    mul.lo.u64 %rd18, %rd17, %rd10;\n\
    add.u64 %rd18, %rd18, %rd22;\n\
    sub.u64 %rd18, %rd18, %rd11;\n\
    add.u64 %rd16, %rd16, %rd18;\n\
    mul.lo.u64 %rd16, %rd16, %rd7;\n\
    mul.lo.u64 %rd18, %rd15, %rd10;\n\
    add.u64 %rd18, %rd18, %rd23;\n\
    sub.u64 %rd18, %rd18, %rd11;\n\
    add.u64 %rd16, %rd16, %rd18;\n\
    // Load input value\n\
    shl.b64 %rd18, %rd16, 2;\n\
    add.u64 %rd18, %rd1, %rd18;\n\
    ld.global.f32 %f2, [%rd18];\n\
    // Compare with max\n\
    setp.le.f32 %p3, %f2, %f1;\n\
    @%p3 bra MP_KX_INC;\n\
    mov.f32 %f1, %f2;\n\
    mov.u64 %rd21, %rd16;\n\
MP_KX_INC:\n\
    add.u64 %rd23, %rd23, 1;\n\
    bra MP_KX;\n\
MP_KY_INC:\n\
    add.u64 %rd22, %rd22, 1;\n\
    bra MP_KY;\n\
MP_WRITE:\n\
    // Store max value\n\
    shl.b64 %rd16, %rd0, 2;\n\
    add.u64 %rd16, %rd2, %rd16;\n\
    st.global.f32 [%rd16], %f1;\n\
    // Store argmax index (as u64)\n\
    shl.b64 %rd16, %rd0, 3;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
    st.global.u64 [%rd16], %rd21;\n\
MP_DONE: ret;\n\
}\0";
