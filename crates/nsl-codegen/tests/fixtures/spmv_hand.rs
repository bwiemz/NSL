//! The hand-written sparse matrix-vector kernels, frozen (new-roadmap item
//! 5).
//!
//! `nsl_runtime::cuda::fused_kernels::{CSR_SPMV_F32_PTX, COO_SPMV_F32_PTX}`
//! as they stood when the kernels moved to `nsl_kir::kernels::spmv`,
//! verbatim (their comments aside; the PTX `//` lines are kept). They are
//! the reference side of `spmv_kir_equivalence.rs` and are not compiled into
//! anything else.

pub const CSR_SPMV_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_csr_spmv_f32(\n\
    .param .u64 row_ptrs,\n\
    .param .u64 col_indices,\n\
    .param .u64 values,\n\
    .param .u64 x,\n\
    .param .u64 y,\n\
    .param .u64 M\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p1;\n\
    // row = blockIdx.x * blockDim.x + threadIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd0, %r1;\n\
    ld.param.u64 %rd1, [row_ptrs];\n\
    ld.param.u64 %rd2, [col_indices];\n\
    ld.param.u64 %rd3, [values];\n\
    ld.param.u64 %rd4, [x];\n\
    ld.param.u64 %rd5, [y];\n\
    ld.param.u64 %rd6, [M];\n\
    setp.ge.u64 %p1, %rd0, %rd6;\n\
    @%p1 bra SPMV_DONE;\n\
    // Load row_ptrs[row] and row_ptrs[row+1]\n\
    shl.b64 %rd7, %rd0, 2;\n\
    add.u64 %rd7, %rd1, %rd7;\n\
    ld.global.u32 %r1, [%rd7];\n\
    ld.global.u32 %r2, [%rd7+4];\n\
    cvt.u64.u32 %rd8, %r1;   // start\n\
    cvt.u64.u32 %rd9, %r2;   // end\n\
    mov.f32 %f1, 0f00000000; // sum = 0\n\
    mov.u64 %rd10, %rd8;\n\
SPMV_LOOP:\n\
    setp.ge.u64 %p1, %rd10, %rd9;\n\
    @%p1 bra SPMV_WRITE;\n\
    // col = col_indices[idx]\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd11, %rd2, %rd11;\n\
    ld.global.u32 %r3, [%rd11];\n\
    cvt.u64.u32 %rd12, %r3;\n\
    // val = values[idx]\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd11, %rd3, %rd11;\n\
    ld.global.f32 %f2, [%rd11];\n\
    // x[col]\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd13, %rd4, %rd13;\n\
    ld.global.f32 %f3, [%rd13];\n\
    fma.rn.f32 %f1, %f2, %f3, %f1;\n\
    add.u64 %rd10, %rd10, 1;\n\
    bra SPMV_LOOP;\n\
SPMV_WRITE:\n\
    shl.b64 %rd7, %rd0, 2;\n\
    add.u64 %rd7, %rd5, %rd7;\n\
    st.global.f32 [%rd7], %f1;\n\
SPMV_DONE: ret;\n\
}\0";


pub const COO_SPMV_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_coo_spmv_f32(\n\
    .param .u64 row_indices,\n\
    .param .u64 col_indices,\n\
    .param .u64 values,\n\
    .param .u64 x,\n\
    .param .u64 y,\n\
    .param .u64 nnz\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p1;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd0, %r1;\n\
    ld.param.u64 %rd1, [row_indices];\n\
    ld.param.u64 %rd2, [col_indices];\n\
    ld.param.u64 %rd3, [values];\n\
    ld.param.u64 %rd4, [x];\n\
    ld.param.u64 %rd5, [y];\n\
    ld.param.u64 %rd6, [nnz];\n\
    setp.ge.u64 %p1, %rd0, %rd6;\n\
    @%p1 bra COOV_DONE;\n\
    // Load row_indices[i], col_indices[i] (i64)\n\
    shl.b64 %rd7, %rd0, 3;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.s64 %rd9, [%rd8];   // row\n\
    add.u64 %rd8, %rd2, %rd7;\n\
    ld.global.s64 %rd10, [%rd8];  // col\n\
    // Load values[i] (f32)\n\
    shl.b64 %rd7, %rd0, 2;\n\
    add.u64 %rd8, %rd3, %rd7;\n\
    ld.global.f32 %f1, [%rd8];\n\
    // Load x[col]\n\
    shl.b64 %rd7, %rd10, 2;\n\
    add.u64 %rd7, %rd4, %rd7;\n\
    ld.global.f32 %f2, [%rd7];\n\
    // product = val * x[col]\n\
    mul.rn.f32 %f3, %f1, %f2;\n\
    // y[row] += product (atomic)\n\
    shl.b64 %rd7, %rd9, 2;\n\
    add.u64 %rd7, %rd5, %rd7;\n\
    atom.global.add.f32 %f2, [%rd7], %f3;\n\
COOV_DONE: ret;\n\
}\0";
