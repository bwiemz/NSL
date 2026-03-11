//! Static PTX kernel strings for GPU elementwise operations.
//! All strings are null-terminated (end with `\0`) for CUDA driver API.

// --- Binary ops ---

pub(crate) const ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_add_f32(\n\
    .param .u64 a, .param .u64 b, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<8>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [b];\n\
    ld.param.u64 %rd3, [c];\n\
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
    add.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

pub(crate) const SUB_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_sub_f32(\n\
    .param .u64 a, .param .u64 b, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<8>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [b];\n\
    ld.param.u64 %rd3, [c];\n\
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
    sub.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

pub(crate) const MUL_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_mul_f32(\n\
    .param .u64 a, .param .u64 b, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<8>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [b];\n\
    ld.param.u64 %rd3, [c];\n\
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
    mul.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

pub(crate) const DIV_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_div_f32(\n\
    .param .u64 a, .param .u64 b, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<8>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [b];\n\
    ld.param.u64 %rd3, [c];\n\
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
    div.f32 %fs3, %fs1, %fs2;\n\
    add.u64 %rd7, %rd3, %rd6;\n\
    st.global.f32 [%rd7], %fs3;\n\
DONE: ret;\n\
}\0";

// --- Unary ops ---

pub(crate) const NEG_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_neg_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd1, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    neg.f32 %fs1, %fs1;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

pub(crate) const RELU_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_relu_f32(\n\
    .param .u64 a, .param .u64 c, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd1, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    mov.f32 %fs2, 0f00000000;\n\
    max.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

// --- Scalar ops ---

pub(crate) const MUL_SCALAR_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_mul_scalar_f32(\n\
    .param .u64 a, .param .u64 c, .param .f32 s, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.f32 %fs2, [s];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd1, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    mul.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";

pub(crate) const ADD_SCALAR_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_add_scalar_f32(\n\
    .param .u64 a, .param .u64 c, .param .f32 s, .param .u64 n\n\
) {\n\
    .reg .u32 %r<4>;\n\
    .reg .u64 %rd<7>;\n\
    .reg .f32 %fs<3>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [c];\n\
    ld.param.f32 %fs2, [s];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r3, %r3, %r1;\n\
    cvt.u64.u32 %rd4, %r3;\n\
    setp.ge.u64 %p1, %rd4, %rd3;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd5, %rd4, 2;\n\
    add.u64 %rd6, %rd1, %rd5;\n\
    ld.global.f32 %fs1, [%rd6];\n\
    add.f32 %fs1, %fs1, %fs2;\n\
    add.u64 %rd6, %rd2, %rd5;\n\
    st.global.f32 [%rd6], %fs1;\n\
DONE: ret;\n\
}\0";
