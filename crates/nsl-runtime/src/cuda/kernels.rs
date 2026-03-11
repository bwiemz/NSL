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

// --- Matrix multiplication ---

pub(crate) const MATMUL_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry nsl_matmul_f32(\n\
    .param .u64 a, .param .u64 b, .param .u64 c,\n\
    .param .u64 M, .param .u64 N, .param .u64 K\n\
) {\n\
    .reg .u32 %r<8>;\n\
    .reg .u64 %rd<16>;\n\
    .reg .f32 %fs<4>;\n\
    .reg .pred %p<3>;\n\
\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [b];\n\
    ld.param.u64 %rd3, [c];\n\
    ld.param.u64 %rd4, [M];\n\
    ld.param.u64 %rd5, [N];\n\
    ld.param.u64 %rd6, [K];\n\
\n\
    mov.u32 %r1, %ctaid.y;\n\
    mov.u32 %r2, %ntid.y;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.y;\n\
    add.u32 %r3, %r3, %r1;\n\
\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r4, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r4, %r4, %r1;\n\
\n\
    cvt.u64.u32 %rd7, %r3;\n\
    cvt.u64.u32 %rd8, %r4;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    setp.ge.u64 %p2, %rd8, %rd5;\n\
    @%p1 bra DONE;\n\
    @%p2 bra DONE;\n\
\n\
    mov.f32 %fs1, 0f00000000;\n\
    mov.u64 %rd9, 0;\n\
\n\
LOOP:\n\
    setp.ge.u64 %p1, %rd9, %rd6;\n\
    @%p1 bra WRITE;\n\
\n\
    mul.lo.u64 %rd10, %rd7, %rd6;\n\
    add.u64 %rd10, %rd10, %rd9;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd1, %rd10;\n\
    ld.global.f32 %fs2, [%rd10];\n\
\n\
    mul.lo.u64 %rd11, %rd9, %rd5;\n\
    add.u64 %rd11, %rd11, %rd8;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd2, %rd11;\n\
    ld.global.f32 %fs3, [%rd11];\n\
\n\
    fma.rn.f32 %fs1, %fs2, %fs3, %fs1;\n\
\n\
    add.u64 %rd9, %rd9, 1;\n\
    bra LOOP;\n\
\n\
WRITE:\n\
    mul.lo.u64 %rd10, %rd7, %rd5;\n\
    add.u64 %rd10, %rd10, %rd8;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd3, %rd10;\n\
    st.global.f32 [%rd10], %fs1;\n\
\n\
DONE:\n\
    ret;\n\
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
