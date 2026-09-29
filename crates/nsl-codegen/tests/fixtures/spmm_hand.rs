//! The hand-written sparse matrix-matrix kernels, frozen (new-roadmap item
//! 5).
//!
//! `nsl_runtime::cuda::fused_kernels::CSR_SPMM_F32_PTX`, `COO_SPMM_F32_PTX`
//! and `BSR_SPMM_F32_PTX` as they stood when the kernels moved to
//! `nsl_kir::kernels::spmm`, verbatim (their comments aside; the PTX `//`
//! lines are kept). They are the reference side of
//! `spmm_kir_equivalence.rs` and are not compiled into anything else.

pub const CSR_SPMM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_csr_spmm_f32(\n\
    .param .u64 row_ptrs,\n\
    .param .u64 col_indices,\n\
    .param .u64 values,\n\
    .param .u64 B,\n\
    .param .u64 C,\n\
    .param .u64 M,\n\
    .param .u64 N\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<3>;\n\
    // row = blockIdx.x (one block per row)\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd0, %r1;  // rd0 = row\n\
    // col_base = blockIdx.y * blockDim.x + threadIdx.x\n\
    mov.u32 %r2, %ctaid.y;\n\
    mov.u32 %r3, %ntid.x;\n\
    mul.lo.u32 %r2, %r2, %r3;\n\
    mov.u32 %r3, %tid.x;\n\
    add.u32 %r2, %r2, %r3;\n\
    cvt.u64.u32 %rd1, %r2;  // rd1 = col\n\
    // Load params\n\
    ld.param.u64 %rd2, [row_ptrs];\n\
    ld.param.u64 %rd3, [col_indices];\n\
    ld.param.u64 %rd4, [values];\n\
    ld.param.u64 %rd5, [B];\n\
    ld.param.u64 %rd6, [C];\n\
    ld.param.u64 %rd7, [M];\n\
    ld.param.u64 %rd8, [N];\n\
    // Bounds check: row < M, col < N\n\
    setp.ge.u64 %p1, %rd0, %rd7;\n\
    @%p1 bra SP_DONE;\n\
    setp.ge.u64 %p1, %rd1, %rd8;\n\
    @%p1 bra SP_DONE;\n\
    // Load row_ptrs[row] and row_ptrs[row+1] (u32)\n\
    shl.b64 %rd9, %rd0, 2;          // row * 4 (u32 offset)\n\
    add.u64 %rd9, %rd2, %rd9;\n\
    ld.global.u32 %r4, [%rd9];      // r4 = row_start\n\
    ld.global.u32 %r5, [%rd9+4];    // r5 = row_end\n\
    cvt.u64.u32 %rd10, %r4;         // rd10 = row_start\n\
    cvt.u64.u32 %rd11, %r5;         // rd11 = row_end\n\
    // Accumulate: sum = 0\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd12, %rd10;           // rd12 = idx (loop variable)\n\
SP_NNZ_LOOP:\n\
    setp.ge.u64 %p1, %rd12, %rd11;\n\
    @%p1 bra SP_WRITE;\n\
    // Load col_indices[idx] (u32)\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd13, %rd3, %rd13;\n\
    ld.global.u32 %r4, [%rd13];     // r4 = k (column in A)\n\
    // Load values[idx] (f32)\n\
    shl.b64 %rd14, %rd12, 2;\n\
    add.u64 %rd14, %rd4, %rd14;\n\
    ld.global.f32 %f2, [%rd14];     // f2 = A_val\n\
    // Load B[k, col] = B[r4 * N + col]\n\
    cvt.u64.u32 %rd15, %r4;\n\
    mul.lo.u64 %rd15, %rd15, %rd8;  // k * N\n\
    add.u64 %rd15, %rd15, %rd1;     // k * N + col\n\
    shl.b64 %rd15, %rd15, 2;        // byte offset (* 4 for f32)\n\
    add.u64 %rd15, %rd5, %rd15;\n\
    ld.global.f32 %f3, [%rd15];     // f3 = B[k, col]\n\
    // sum += A_val * B[k, col]\n\
    fma.rn.f32 %f1, %f2, %f3, %f1;\n\
    // idx++\n\
    add.u64 %rd12, %rd12, 1;\n\
    bra SP_NNZ_LOOP;\n\
SP_WRITE:\n\
    // Store C[row, col] = C[row * N + col]\n\
    mul.lo.u64 %rd13, %rd0, %rd8;   // row * N\n\
    add.u64 %rd13, %rd13, %rd1;     // row * N + col\n\
    shl.b64 %rd13, %rd13, 2;        // byte offset\n\
    add.u64 %rd13, %rd6, %rd13;\n\
    st.global.f32 [%rd13], %f1;\n\
SP_DONE: ret;\n\
}\0";

pub const COO_SPMM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_coo_spmm_f32(\n\
    .param .u64 row_indices,\n\
    .param .u64 col_indices,\n\
    .param .u64 values,\n\
    .param .u64 B,\n\
    .param .u64 C,\n\
    .param .u64 N,\n\
    .param .u64 nnz\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<4>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p1;\n\
    // Global thread index\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd0, %r1;  // thread_id\n\
    // Load params\n\
    ld.param.u64 %rd1, [row_indices];\n\
    ld.param.u64 %rd2, [col_indices];\n\
    ld.param.u64 %rd3, [values];\n\
    ld.param.u64 %rd4, [B];\n\
    ld.param.u64 %rd5, [C];\n\
    ld.param.u64 %rd6, [N];\n\
    ld.param.u64 %rd7, [nnz];\n\
    // Each thread handles one nonzero element, broadcasting across B columns\n\
    // thread_id indexes the nonzero; we process all N columns for that nonzero\n\
    setp.ge.u64 %p1, %rd0, %rd7;\n\
    @%p1 bra COO_DONE;\n\
    // Load row_indices[thread_id] and col_indices[thread_id] (i64)\n\
    shl.b64 %rd8, %rd0, 3;         // * 8 (i64)\n\
    add.u64 %rd9, %rd1, %rd8;\n\
    ld.global.s64 %rd10, [%rd9];    // row\n\
    add.u64 %rd9, %rd2, %rd8;\n\
    ld.global.s64 %rd11, [%rd9];    // col (= k in A)\n\
    // Load values[thread_id] (f32 -- we store values as f32 on GPU)\n\
    shl.b64 %rd8, %rd0, 2;         // * 4 (f32)\n\
    add.u64 %rd9, %rd3, %rd8;\n\
    ld.global.f32 %f1, [%rd9];     // val = A[row, col]\n\
    // For each output column j in [0, N):\n\
    // C[row, j] += val * B[col, j]\n\
    // We iterate j with a simple loop (each thread does all N columns)\n\
    mov.u64 %rd12, 0;              // j = 0\n\
COO_COL_LOOP:\n\
    setp.ge.u64 %p1, %rd12, %rd6;\n\
    @%p1 bra COO_DONE;\n\
    // B[col, j] = B[col * N + j]\n\
    mul.lo.u64 %rd13, %rd11, %rd6;\n\
    add.u64 %rd13, %rd13, %rd12;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd4, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    // product = val * B[col, j]\n\
    mul.rn.f32 %f3, %f1, %f2;\n\
    // C[row, j] += product (atomic add for safety with multiple threads per row)\n\
    mul.lo.u64 %rd14, %rd10, %rd6;\n\
    add.u64 %rd14, %rd14, %rd12;\n\
    shl.b64 %rd14, %rd14, 2;\n\
    add.u64 %rd14, %rd5, %rd14;\n\
    atom.global.add.f32 %f2, [%rd14], %f3;\n\
    // j++\n\
    add.u64 %rd12, %rd12, 1;\n\
    bra COO_COL_LOOP;\n\
COO_DONE: ret;\n\
}\0";

pub const BSR_SPMM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_70\n\
.address_size 64\n\
\n\
.visible .entry nsl_bsr_spmm_f32(\n\
    .param .u64 row_ptrs,\n\
    .param .u64 col_indices,\n\
    .param .u64 values,\n\
    .param .u64 B,\n\
    .param .u64 C,\n\
    .param .u64 N,\n\
    .param .u64 block_rows,\n\
    .param .u64 block_cols,\n\
    .param .u64 nblk_rows\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<3>;\n\
    // block_row = blockIdx.x, sub_row = threadIdx.y, out_col = blockIdx.y*blockDim.x+threadIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd0, %r1;   // block_row\n\
    mov.u32 %r2, %tid.y;\n\
    cvt.u64.u32 %rd1, %r2;   // sub_row within block\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.x;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.x;\n\
    add.u32 %r3, %r3, %r4;\n\
    cvt.u64.u32 %rd2, %r3;   // out_col\n\
    // Load params\n\
    ld.param.u64 %rd3, [row_ptrs];\n\
    ld.param.u64 %rd4, [col_indices];\n\
    ld.param.u64 %rd5, [values];\n\
    ld.param.u64 %rd6, [B];\n\
    ld.param.u64 %rd7, [C];\n\
    ld.param.u64 %rd8, [N];\n\
    ld.param.u64 %rd9, [block_rows];\n\
    ld.param.u64 %rd10, [block_cols];\n\
    ld.param.u64 %rd11, [nblk_rows];\n\
    // Bounds check\n\
    setp.ge.u64 %p1, %rd0, %rd11;\n\
    @%p1 bra BSR_DONE;\n\
    setp.ge.u64 %p1, %rd1, %rd9;\n\
    @%p1 bra BSR_DONE;\n\
    setp.ge.u64 %p1, %rd2, %rd8;\n\
    @%p1 bra BSR_DONE;\n\
    // Load row_ptrs[block_row] and row_ptrs[block_row+1]\n\
    shl.b64 %rd12, %rd0, 2;\n\
    add.u64 %rd12, %rd3, %rd12;\n\
    ld.global.u32 %r1, [%rd12];\n\
    ld.global.u32 %r2, [%rd12+4];\n\
    cvt.u64.u32 %rd13, %r1;  // blk_start\n\
    cvt.u64.u32 %rd14, %r2;  // blk_end\n\
    // Accumulate\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd15, %rd13;    // blk_idx\n\
BSR_BLK_LOOP:\n\
    setp.ge.u64 %p1, %rd15, %rd14;\n\
    @%p1 bra BSR_WRITE;\n\
    // Load col_indices[blk_idx] (u32) = block_col\n\
    shl.b64 %rd16, %rd15, 2;\n\
    add.u64 %rd16, %rd4, %rd16;\n\
    ld.global.u32 %r3, [%rd16];\n\
    cvt.u64.u32 %rd17, %r3;  // block_col\n\
    // block_size = block_rows * block_cols\n\
    mul.lo.u64 %rd18, %rd9, %rd10;\n\
    // val_offset = blk_idx * block_size + sub_row * block_cols\n\
    mul.lo.u64 %rd16, %rd15, %rd18;\n\
    mul.lo.u64 %rd19, %rd1, %rd10;\n\
    add.u64 %rd16, %rd16, %rd19;\n\
    // Loop over sub_col in [0, block_cols)\n\
    mov.u64 %rd19, 0;\n\
BSR_SUBCOL_LOOP:\n\
    setp.ge.u64 %p2, %rd19, %rd10;\n\
    @%p2 bra BSR_BLK_INC;\n\
    // Load values[val_offset + sub_col]\n\
    add.u64 %rd18, %rd16, %rd19;\n\
    shl.b64 %rd18, %rd18, 2;\n\
    add.u64 %rd18, %rd5, %rd18;\n\
    ld.global.f32 %f2, [%rd18];\n\
    // k = block_col * block_cols + sub_col\n\
    mul.lo.u64 %rd18, %rd17, %rd10;\n\
    add.u64 %rd18, %rd18, %rd19;\n\
    // B[k, out_col]\n\
    mul.lo.u64 %rd18, %rd18, %rd8;\n\
    add.u64 %rd18, %rd18, %rd2;\n\
    shl.b64 %rd18, %rd18, 2;\n\
    add.u64 %rd18, %rd6, %rd18;\n\
    ld.global.f32 %f3, [%rd18];\n\
    fma.rn.f32 %f1, %f2, %f3, %f1;\n\
    add.u64 %rd19, %rd19, 1;\n\
    bra BSR_SUBCOL_LOOP;\n\
BSR_BLK_INC:\n\
    add.u64 %rd15, %rd15, 1;\n\
    bra BSR_BLK_LOOP;\n\
BSR_WRITE:\n\
    // output_row = block_row * block_rows + sub_row\n\
    mul.lo.u64 %rd16, %rd0, %rd9;\n\
    add.u64 %rd16, %rd16, %rd1;\n\
    // C[output_row, out_col]\n\
    mul.lo.u64 %rd16, %rd16, %rd8;\n\
    add.u64 %rd16, %rd16, %rd2;\n\
    shl.b64 %rd16, %rd16, 2;\n\
    add.u64 %rd16, %rd7, %rd16;\n\
    st.global.f32 [%rd16], %f1;\n\
BSR_DONE: ret;\n\
}\0";
