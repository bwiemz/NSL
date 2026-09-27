//! The hand-written data-movement kernels, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{BIAS_ADD_F32_PTX, GATHER_DIM_F32_PTX,
//! STRIDED_COPY_F32_PTX, GPU_SLICE_F32_PTX}` as they stood when the kernels
//! moved to `nsl_kir::kernels::data_movement`, verbatim (their comments
//! aside; the PTX `//` lines are kept). They are the reference side of
//! `data_movement_kir_equivalence.rs` and are not compiled into anything
//! else.

pub const BIAS_ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_bias_add_f32(\n\
    .param .u64 inp, .param .u64 bias, .param .u64 out,\n\
    .param .u64 total, .param .u64 cols\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p1;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [bias];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [total];\n\
    ld.param.u64 %rd5, [cols];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd7, %rd6, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %f1, [%rd8];\n\
    rem.u64 %rd9, %rd6, %rd5;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd9, %rd2, %rd9;\n\
    ld.global.f32 %f2, [%rd9];\n\
    add.f32 %f3, %f1, %f2;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    st.global.f32 [%rd8], %f3;\n\
DONE: ret;\n\
}\0";

pub const GATHER_DIM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_gather_dim_f32(\n\
    .param .u64 input, .param .u64 indices, .param .u64 out,\n\
    .param .u64 outer, .param .u64 gather_dim_size, .param .u64 inner\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    ld.param.u64 %rd1, [input];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [outer];\n\
    ld.param.u64 %rd5, [gather_dim_size];\n\
    ld.param.u64 %rd6, [inner];\n\
    // gid = blockIdx.x * blockDim.x + threadIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %tid.x;\n\
    mul.lo.u32 %r4, %r1, %r2;\n\
    add.u32 %r4, %r4, %r3;\n\
    cvt.u64.u32 %rd7, %r4;\n\
    // total = outer * inner\n\
    mul.lo.u64 %rd8, %rd4, %rd6;\n\
    setp.ge.u64 %p1, %rd7, %rd8;\n\
    @%p1 bra GD_DONE;\n\
    // o = gid / inner ; k = gid % inner\n\
    div.u64 %rd9, %rd7, %rd6;\n\
    rem.u64 %rd10, %rd7, %rd6;\n\
    // idx = (u64)indices[o]\n\
    shl.b64 %rd11, %rd9, 2;\n\
    add.u64 %rd11, %rd2, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    cvt.rzi.u64.f32 %rd12, %f1;\n\
    // Out-of-range writes zero rather than reading out of bounds. The host has\n\
    // already rejected such indices; this only bounds the damage.\n\
    mov.f32 %f2, 0f00000000;\n\
    setp.ge.u64 %p2, %rd12, %rd5;\n\
    @%p2 bra GD_STORE;\n\
    // in_off = o * gather_dim_size * inner + idx * inner + k\n\
    mul.lo.u64 %rd13, %rd9, %rd5;\n\
    mul.lo.u64 %rd13, %rd13, %rd6;\n\
    mul.lo.u64 %rd14, %rd12, %rd6;\n\
    add.u64 %rd13, %rd13, %rd14;\n\
    add.u64 %rd13, %rd13, %rd10;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd1, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
GD_STORE:\n\
    shl.b64 %rd15, %rd7, 2;\n\
    add.u64 %rd15, %rd3, %rd15;\n\
    st.global.f32 [%rd15], %f2;\n\
GD_DONE:\n\
    ret;\n\
}\0";

pub const STRIDED_COPY_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_strided_copy_f32(\n\
    .param .u64 src, .param .u64 dst,\n\
    .param .u64 shape, .param .u64 src_strides, .param .u64 dst_strides,\n\
    .param .u64 ndim, .param .u64 total\n\
) {\n\
    .reg .u64 %rd<18>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f1;\n\
    .reg .pred %p<3>;\n\
    // Global thread index = flat contiguous output index\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd0, %r1;\n\
    // Load params\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [shape];\n\
    ld.param.u64 %rd4, [src_strides];\n\
    ld.param.u64 %rd5, [dst_strides];\n\
    ld.param.u64 %rd6, [ndim];\n\
    ld.param.u64 %rd7, [total];\n\
    // Bounds check\n\
    setp.ge.u64 %p1, %rd0, %rd7;\n\
    @%p1 bra SC_DONE;\n\
    // Decompose flat_idx into N-dim coords using dst_strides,\n\
    // then compute src offset using src_strides\n\
    // remaining = flat_idx\n\
    mov.u64 %rd8, %rd0;\n\
    // src_offset = 0\n\
    mov.u64 %rd9, 0;\n\
    // dim = 0\n\
    mov.u64 %rd10, 0;\n\
SC_DIM_LOOP:\n\
    setp.ge.u64 %p1, %rd10, %rd6;\n\
    @%p1 bra SC_LOAD;\n\
    // byte offset for dim index: dim * 8\n\
    shl.b64 %rd11, %rd10, 3;\n\
    // Load dst_strides[dim] for coordinate decomposition\n\
    add.u64 %rd12, %rd5, %rd11;\n\
    ld.global.u64 %rd13, [%rd12];\n\
    // Guard: if dst_stride == 0, skip this dim (shouldn't happen for contiguous)\n\
    setp.eq.u64 %p2, %rd13, 0;\n\
    @%p2 bra SC_DIM_INC;\n\
    // coord = remaining / dst_strides[dim]\n\
    div.u64 %rd14, %rd8, %rd13;\n\
    // remaining = remaining % dst_strides[dim]\n\
    rem.u64 %rd8, %rd8, %rd13;\n\
    // Clamp coord by shape[dim] (handles broadcast/expand edge cases)\n\
    add.u64 %rd15, %rd3, %rd11;\n\
    ld.global.u64 %rd16, [%rd15];\n\
    rem.u64 %rd14, %rd14, %rd16;\n\
    // Load src_strides[dim]\n\
    add.u64 %rd15, %rd4, %rd11;\n\
    ld.global.u64 %rd17, [%rd15];\n\
    // src_offset += coord * src_strides[dim]\n\
    mul.lo.u64 %rd17, %rd14, %rd17;\n\
    add.u64 %rd9, %rd9, %rd17;\n\
SC_DIM_INC:\n\
    add.u64 %rd10, %rd10, 1;\n\
    bra SC_DIM_LOOP;\n\
SC_LOAD:\n\
    // Load src[src_offset]\n\
    shl.b64 %rd11, %rd9, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    // Store dst[flat_idx]\n\
    shl.b64 %rd11, %rd0, 2;\n\
    add.u64 %rd11, %rd2, %rd11;\n\
    st.global.f32 [%rd11], %f1;\n\
SC_DONE: ret;\n\
}\0";

pub const GPU_SLICE_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_slice_f32(\n\
    .param .u64 src, .param .u64 dst,\n\
    .param .u64 shape, .param .u64 src_strides, .param .u64 dst_strides,\n\
    .param .u64 ndim, .param .u64 total,\n\
    .param .u64 slice_dim, .param .u64 slice_start\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f1;\n\
    .reg .pred %p<3>;\n\
    // Global thread index = flat contiguous output index\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd0, %r1;\n\
    // Load params\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [dst];\n\
    ld.param.u64 %rd3, [shape];\n\
    ld.param.u64 %rd4, [src_strides];\n\
    ld.param.u64 %rd5, [dst_strides];\n\
    ld.param.u64 %rd6, [ndim];\n\
    ld.param.u64 %rd7, [total];\n\
    ld.param.u64 %rd18, [slice_dim];\n\
    ld.param.u64 %rd19, [slice_start];\n\
    // Bounds check\n\
    setp.ge.u64 %p1, %rd0, %rd7;\n\
    @%p1 bra SL_DONE;\n\
    // Decompose flat_idx into N-dim coords using dst_strides,\n\
    // add slice_start to the slice_dim coord,\n\
    // then compute src offset using src_strides\n\
    mov.u64 %rd8, %rd0;  // remaining\n\
    mov.u64 %rd9, 0;     // src_offset\n\
    mov.u64 %rd10, 0;    // dim\n\
SL_DIM_LOOP:\n\
    setp.ge.u64 %p1, %rd10, %rd6;\n\
    @%p1 bra SL_LOAD;\n\
    shl.b64 %rd11, %rd10, 3;\n\
    // Load dst_strides[dim]\n\
    add.u64 %rd12, %rd5, %rd11;\n\
    ld.global.u64 %rd13, [%rd12];\n\
    setp.eq.u64 %p2, %rd13, 0;\n\
    @%p2 bra SL_DIM_INC;\n\
    // coord = remaining / dst_strides[dim]\n\
    div.u64 %rd14, %rd8, %rd13;\n\
    rem.u64 %rd8, %rd8, %rd13;\n\
    // Clamp coord by shape[dim]\n\
    add.u64 %rd15, %rd3, %rd11;\n\
    ld.global.u64 %rd16, [%rd15];\n\
    rem.u64 %rd14, %rd14, %rd16;\n\
    // If this is the slice dim, add slice_start to coord\n\
    setp.ne.u64 %p2, %rd10, %rd18;\n\
    @%p2 bra SL_NO_OFFSET;\n\
    add.u64 %rd14, %rd14, %rd19;\n\
SL_NO_OFFSET:\n\
    // Load src_strides[dim]\n\
    add.u64 %rd15, %rd4, %rd11;\n\
    ld.global.u64 %rd17, [%rd15];\n\
    // src_offset += coord * src_strides[dim]\n\
    mul.lo.u64 %rd17, %rd14, %rd17;\n\
    add.u64 %rd9, %rd9, %rd17;\n\
SL_DIM_INC:\n\
    add.u64 %rd10, %rd10, 1;\n\
    bra SL_DIM_LOOP;\n\
SL_LOAD:\n\
    // Load src[src_offset]\n\
    shl.b64 %rd11, %rd9, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    // Store dst[flat_idx]\n\
    shl.b64 %rd11, %rd0, 2;\n\
    add.u64 %rd11, %rd2, %rd11;\n\
    st.global.f32 [%rd11], %f1;\n\
SL_DONE: ret;\n\
}\0";
