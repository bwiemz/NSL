//! The hand-written 2-D convolution forward, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::CONV2D_F32_PTX` as it stood when the
//! kernel moved to `nsl_kir::kernels::conv2d`, verbatim (its comments
//! aside; the PTX `//` lines are kept). It is the reference side of
//! `conv2d_kir_equivalence.rs` and is not compiled into anything else.

pub const CONV2D_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_conv2d_f32(\n\
    .param .u64 inp, .param .u64 wt, .param .u64 bias, .param .u64 out,\n\
    .param .u64 N, .param .u64 C_in, .param .u64 H, .param .u64 W,\n\
    .param .u64 C_out, .param .u64 kH, .param .u64 kW,\n\
    .param .u64 stride_h, .param .u64 stride_w,\n\
    .param .u64 pad_h, .param .u64 pad_w,\n\
    .param .u64 H_out, .param .u64 W_out, .param .u64 total\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f<4>;\n\
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
    ld.param.u64 %rd2, [wt];\n\
    ld.param.u64 %rd3, [bias];\n\
    ld.param.u64 %rd4, [out];\n\
    ld.param.u64 %rd5, [N];\n\
    ld.param.u64 %rd6, [C_in];\n\
    ld.param.u64 %rd7, [H];\n\
    ld.param.u64 %rd8, [W];\n\
    ld.param.u64 %rd9, [C_out];\n\
    ld.param.u64 %rd10, [kH];\n\
    ld.param.u64 %rd11, [kW];\n\
    ld.param.u64 %rd12, [stride_h];\n\
    ld.param.u64 %rd13, [stride_w];\n\
    ld.param.u64 %rd14, [pad_h];\n\
    ld.param.u64 %rd15, [pad_w];\n\
    ld.param.u64 %rd16, [H_out];\n\
    ld.param.u64 %rd17, [W_out];\n\
    ld.param.u64 %rd18, [total];\n\
    // Bounds check\n\
    setp.ge.u64 %p1, %rd0, %rd18;\n\
    @%p1 bra CONV_DONE;\n\
    // Decompose flat index -> (n, co, oh, ow)\n\
    // ow = idx % W_out\n\
    rem.u64 %rd19, %rd0, %rd17;\n\
    // tmp = idx / W_out\n\
    div.u64 %rd20, %rd0, %rd17;\n\
    // oh = tmp % H_out\n\
    rem.u64 %rd21, %rd20, %rd16;\n\
    // tmp2 = tmp / H_out\n\
    div.u64 %rd22, %rd20, %rd16;\n\
    // co = tmp2 % C_out\n\
    rem.u64 %rd23, %rd22, %rd9;\n\
    // n = tmp2 / C_out\n\
    div.u64 %rd24, %rd22, %rd9;\n\
    // Accumulator = 0\n\
    mov.f32 %f1, 0f00000000;\n\
    // Triple loop: ci, ky, kx\n\
    mov.u64 %rd25, 0;\n\
CONV_CI:\n\
    setp.ge.u64 %p1, %rd25, %rd6;\n\
    @%p1 bra CONV_BIAS;\n\
    mov.u64 %rd26, 0;\n\
CONV_KY:\n\
    setp.ge.u64 %p1, %rd26, %rd10;\n\
    @%p1 bra CONV_CI_INC;\n\
    mov.u64 %rd27, 0;\n\
CONV_KX:\n\
    setp.ge.u64 %p1, %rd27, %rd11;\n\
    @%p1 bra CONV_KY_INC;\n\
    // ih = oh * stride_h + ky\n\
    mul.lo.u64 %rd28, %rd21, %rd12;\n\
    add.u64 %rd28, %rd28, %rd26;\n\
    // iw = ow * stride_w + kx\n\
    mul.lo.u64 %rd29, %rd19, %rd13;\n\
    add.u64 %rd29, %rd29, %rd27;\n\
    // Padding check: ih >= pad_h && iw >= pad_w && ih-pad_h < H && iw-pad_w < W\n\
    setp.lt.u64 %p2, %rd28, %rd14;\n\
    @%p2 bra CONV_KX_INC;\n\
    setp.lt.u64 %p2, %rd29, %rd15;\n\
    @%p2 bra CONV_KX_INC;\n\
    sub.u64 %rd28, %rd28, %rd14;\n\
    sub.u64 %rd29, %rd29, %rd15;\n\
    setp.ge.u64 %p2, %rd28, %rd7;\n\
    @%p2 bra CONV_KX_INC_RESTORE;\n\
    setp.ge.u64 %p2, %rd29, %rd8;\n\
    @%p2 bra CONV_KX_INC_RESTORE;\n\
    // input[n, ci, ih-pad, iw-pad]\n\
    mul.lo.u64 %rd30, %rd24, %rd6;\n\
    add.u64 %rd30, %rd30, %rd25;\n\
    mul.lo.u64 %rd30, %rd30, %rd7;\n\
    add.u64 %rd30, %rd30, %rd28;\n\
    mul.lo.u64 %rd30, %rd30, %rd8;\n\
    add.u64 %rd30, %rd30, %rd29;\n\
    shl.b64 %rd30, %rd30, 2;\n\
    add.u64 %rd30, %rd1, %rd30;\n\
    ld.global.f32 %f2, [%rd30];\n\
    // weight[co, ci, ky, kx]\n\
    mul.lo.u64 %rd31, %rd23, %rd6;\n\
    add.u64 %rd31, %rd31, %rd25;\n\
    mul.lo.u64 %rd31, %rd31, %rd10;\n\
    // Restore ky from pre-subtraction: ky is still in %rd26, kx in %rd27\n\
    add.u64 %rd31, %rd31, %rd26;\n\
    mul.lo.u64 %rd31, %rd31, %rd11;\n\
    add.u64 %rd31, %rd31, %rd27;\n\
    shl.b64 %rd31, %rd31, 2;\n\
    add.u64 %rd31, %rd2, %rd31;\n\
    ld.global.f32 %f3, [%rd31];\n\
    fma.rn.f32 %f1, %f2, %f3, %f1;\n\
    // Restore ih,iw for next iteration (we subtracted pad above)\n\
    add.u64 %rd28, %rd28, %rd14;\n\
    add.u64 %rd29, %rd29, %rd15;\n\
    bra CONV_KX_INC;\n\
CONV_KX_INC_RESTORE:\n\
    // Restore ih/iw after failed bounds check (pad was already subtracted)\n\
    add.u64 %rd28, %rd28, %rd14;\n\
    add.u64 %rd29, %rd29, %rd15;\n\
CONV_KX_INC:\n\
    add.u64 %rd27, %rd27, 1;\n\
    bra CONV_KX;\n\
CONV_KY_INC:\n\
    add.u64 %rd26, %rd26, 1;\n\
    bra CONV_KY;\n\
CONV_CI_INC:\n\
    add.u64 %rd25, %rd25, 1;\n\
    bra CONV_CI;\n\
CONV_BIAS:\n\
    // Add bias if non-null\n\
    setp.eq.u64 %p3, %rd3, 0;\n\
    @%p3 bra CONV_STORE;\n\
    shl.b64 %rd30, %rd23, 2;\n\
    add.u64 %rd30, %rd3, %rd30;\n\
    ld.global.f32 %f2, [%rd30];\n\
    add.f32 %f1, %f1, %f2;\n\
CONV_STORE:\n\
    // Store out[flat_idx]\n\
    shl.b64 %rd30, %rd0, 2;\n\
    add.u64 %rd30, %rd4, %rd30;\n\
    st.global.f32 [%rd30], %f1;\n\
CONV_DONE: ret;\n\
}\0";
