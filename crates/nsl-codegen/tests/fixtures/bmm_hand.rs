//! The hand-written strided batched matmul, frozen (new-roadmap item 5).
//!
//! `nsl_runtime::cuda::fused_kernels::BMM_F32_PTX` as it stood when the
//! kernel moved to `nsl_kir::kernels::bmm`, verbatim (its comments aside;
//! the PTX `//` lines are kept). It is the reference side of
//! `bmm_kir_equivalence.rs` and is not compiled into anything else.

pub const BMM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_bmm_f32(\n\
    .param .u64 a, .param .u64 b, .param .u64 c,\n\
    .param .u64 M, .param .u64 N, .param .u64 K,\n\
    .param .u64 batch_count,\n\
    .param .u64 stride_A, .param .u64 stride_B, .param .u64 stride_C\n\
) {\n\
    .reg .u32 %r<8>;\n\
    .reg .u64 %rd<20>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    // Load params\n\
    ld.param.u64 %rd1, [a];\n\
    ld.param.u64 %rd2, [b];\n\
    ld.param.u64 %rd3, [c];\n\
    ld.param.u64 %rd4, [M];\n\
    ld.param.u64 %rd5, [N];\n\
    ld.param.u64 %rd6, [K];\n\
    ld.param.u64 %rd7, [batch_count];\n\
    ld.param.u64 %rd8, [stride_A];\n\
    ld.param.u64 %rd9, [stride_B];\n\
    ld.param.u64 %rd10, [stride_C];\n\
    // row = blockIdx.y * 16 + threadIdx.y\n\
    mov.u32 %r1, %ctaid.y;\n\
    mov.u32 %r2, %ntid.y;\n\
    mul.lo.u32 %r3, %r1, %r2;\n\
    mov.u32 %r1, %tid.y;\n\
    add.u32 %r3, %r3, %r1;\n\
    // col = blockIdx.x * 16 + threadIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r4, %r1, %r2;\n\
    mov.u32 %r1, %tid.x;\n\
    add.u32 %r4, %r4, %r1;\n\
    // batch = blockIdx.z\n\
    mov.u32 %r5, %ctaid.z;\n\
    cvt.u64.u32 %rd11, %r3;\n\
    cvt.u64.u32 %rd12, %r4;\n\
    cvt.u64.u32 %rd13, %r5;\n\
    // Bounds check: row < M, col < N, batch < batch_count\n\
    setp.ge.u64 %p1, %rd11, %rd4;\n\
    @%p1 bra BMM_DONE;\n\
    setp.ge.u64 %p2, %rd12, %rd5;\n\
    @%p2 bra BMM_DONE;\n\
    setp.ge.u64 %p3, %rd13, %rd7;\n\
    @%p3 bra BMM_DONE;\n\
    // A_base = a + batch * stride_A * 4\n\
    mul.lo.u64 %rd14, %rd13, %rd8;\n\
    shl.b64 %rd14, %rd14, 2;\n\
    add.u64 %rd14, %rd1, %rd14;\n\
    // B_base = b + batch * stride_B * 4\n\
    mul.lo.u64 %rd15, %rd13, %rd9;\n\
    shl.b64 %rd15, %rd15, 2;\n\
    add.u64 %rd15, %rd2, %rd15;\n\
    // C_base = c + batch * stride_C * 4\n\
    mul.lo.u64 %rd16, %rd13, %rd10;\n\
    shl.b64 %rd16, %rd16, 2;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
    // Accumulator = 0\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd17, 0;\n\
BMM_LOOP:\n\
    setp.ge.u64 %p1, %rd17, %rd6;\n\
    @%p1 bra BMM_WRITE;\n\
    // A[row, k] = A_base + (row * K + k) * 4\n\
    mul.lo.u64 %rd18, %rd11, %rd6;\n\
    add.u64 %rd18, %rd18, %rd17;\n\
    shl.b64 %rd18, %rd18, 2;\n\
    add.u64 %rd18, %rd14, %rd18;\n\
    ld.global.f32 %f2, [%rd18];\n\
    // B[k, col] = B_base + (k * N + col) * 4\n\
    mul.lo.u64 %rd19, %rd17, %rd5;\n\
    add.u64 %rd19, %rd19, %rd12;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd15, %rd19;\n\
    ld.global.f32 %f3, [%rd19];\n\
    // acc += A * B\n\
    fma.rn.f32 %f1, %f2, %f3, %f1;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra BMM_LOOP;\n\
BMM_WRITE:\n\
    // C[row, col] = C_base + (row * N + col) * 4\n\
    mul.lo.u64 %rd18, %rd11, %rd5;\n\
    add.u64 %rd18, %rd18, %rd12;\n\
    shl.b64 %rd18, %rd18, 2;\n\
    add.u64 %rd18, %rd16, %rd18;\n\
    st.global.f32 [%rd18], %f1;\n\
BMM_DONE: ret;\n\
}\0";
