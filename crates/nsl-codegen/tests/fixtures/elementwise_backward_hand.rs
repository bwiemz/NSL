//! The hand-written activation-backward kernels, frozen (roadmap A2 step 11).
//!
//! `nsl_runtime::cuda::kernels::{RELU,SIGMOID,TANH,SILU}_BACKWARD_F32_PTX`,
//! `{SIGMOID,TANH,SILU,GELU}_BACKWARD_SRCAD_F32_PTX` and
//! `SWIGLU_GATE_BACKWARD_F32_PTX` as they stood when the kernels moved to
//! `nsl_kir::kernels::elementwise`, verbatim (their comments aside). They are
//! the reference side of `elementwise_backward_kir_equivalence.rs` and are not
//! compiled into anything else.


pub const RELU_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_relu_backward_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p<2>;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
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
    mov.f32 %fs3, 0f00000000;\n\
    setp.gt.f32 %p1, %fs2, %fs3;\n\
    selp.f32 %fs3, %fs1, 0f00000000, %p1;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

pub const SIGMOID_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sigmoid_backward_f32(\n\
    .param .u64 grad, .param .u64 saved, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [saved];\n\
    ld.param.u64 %rd3, [out];\n\
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
    sub.f32 %fs3, 0f3F800000, %fs2;\n\
    mul.f32 %fs3, %fs2, %fs3;\n\
    mul.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

pub const TANH_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_tanh_backward_f32(\n\
    .param .u64 grad, .param .u64 saved, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [saved];\n\
    ld.param.u64 %rd3, [out];\n\
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
    mul.f32 %fs3, %fs2, %fs2;\n\
    sub.f32 %fs3, 0f3F800000, %fs3;\n\
    mul.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

pub const SILU_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_silu_backward_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<8>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
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
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.f32 %fs4, %fs2, %fs4;\n\
    mul.f32 %fs4, %fs3, %fs4;\n\
    add.f32 %fs4, %fs3, %fs4;\n\
    mul.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs4;\n\
DONE: ret;\n\
}\0";

pub const SIGMOID_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_sigmoid_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
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
    sub.rn.f32 %fs3, 0f3F800000, %fs2;\n\
    mul.rn.f32 %fs3, %fs2, %fs3;\n\
    mul.rn.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

pub const TANH_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_tanh_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
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
    mul.rn.f32 %fs3, %fs2, %fs2;\n\
    sub.rn.f32 %fs3, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs3, %fs1, %fs3;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs3;\n\
DONE: ret;\n\
}\0";

pub const SILU_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_silu_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
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
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.rn.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs4, %fs2, %fs4;\n\
    add.rn.f32 %fs4, 0f3F800000, %fs4;\n\
    mul.rn.f32 %fs4, %fs3, %fs4;\n\
    mul.rn.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs4;\n\
DONE: ret;\n\
}\0";

pub const GELU_BACKWARD_SRCAD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_gelu_backward_srcad_f32(\n\
    .param .u64 grad, .param .u64 input, .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<9>;\n\
    .reg .f32 %fs<5>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [input];\n\
    ld.param.u64 %rd3, [out];\n\
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
    mul.rn.f32 %fs2, %fs2, 0f3FD9DB23;\n\
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.rn.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs4, %fs2, %fs4;\n\
    add.rn.f32 %fs4, 0f3F800000, %fs4;\n\
    mul.rn.f32 %fs4, %fs3, %fs4;\n\
    mul.rn.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd8, %rd3, %rd6;\n\
    st.global.f32 [%rd8], %fs4;\n\
DONE: ret;\n\
}\0";

pub const SWIGLU_GATE_BACKWARD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_swiglu_gate_backward_f32(\n\
    .param .u64 grad, .param .u64 up, .param .u64 input,\n\
    .param .u64 out, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<10>;\n\
    .reg .f32 %fs<6>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [grad];\n\
    ld.param.u64 %rd2, [up];\n\
    ld.param.u64 %rd3, [input];\n\
    ld.param.u64 %rd4, [out];\n\
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
    ld.global.f32 %fs1, [%rd8];\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.f32 %fs5, [%rd8];\n\
    mul.rn.f32 %fs1, %fs1, %fs5;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %fs2, [%rd8];\n\
    neg.f32 %fs3, %fs2;\n\
    mul.f32 %fs3, %fs3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %fs3, %fs3;\n\
    add.f32 %fs3, %fs3, 0f3F800000;\n\
    rcp.approx.f32 %fs3, %fs3;\n\
    sub.rn.f32 %fs4, 0f3F800000, %fs3;\n\
    mul.rn.f32 %fs4, %fs2, %fs4;\n\
    add.rn.f32 %fs4, 0f3F800000, %fs4;\n\
    mul.rn.f32 %fs4, %fs3, %fs4;\n\
    mul.rn.f32 %fs4, %fs1, %fs4;\n\
    add.u64 %rd9, %rd4, %rd7;\n\
    st.global.f32 [%rd9], %fs4;\n\
DONE: ret;\n\
}\0";
