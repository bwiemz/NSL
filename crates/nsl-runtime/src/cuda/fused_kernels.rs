//! Fused GPU kernels for embedding lookup, bias_add, layernorm, and rmsnorm.
//! All PTX strings are null-terminated for the CUDA driver API.

// PTX constants are loaded at runtime by name via the CUDA driver API;
// Rust's dead-code analysis cannot see these usages.
#![allow(dead_code)]

// ---------------------------------------------------------------------------
// GPU Embedding Lookup
// Thread (i, j): copies weight[indices[i], j] -> out[i, j]
// Grid:  (ceil(seq_len/16), ceil(embed_dim/16), 1)
// Block: (16, 16, 1)
// Params: weight ptr, indices ptr, out ptr, seq_len (u64), embed_dim (u64)
// indices are stored as f32 (matching GPU dtype=1 convention)
// ---------------------------------------------------------------------------
pub(crate) const EMBEDDING_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_f32(\n\
    .param .u64 weight, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p1;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    ld.param.u64 %rd1, [weight];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    cvt.u64.u32 %rd6, %r1;\n\
    cvt.u64.u32 %rd7, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra DONE;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd8, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd8;\n\
    ld.global.f32 %f1, [%rd8];\n\
    cvt.rzi.u64.f32 %rd9, %f1;\n\
    mul.lo.u64 %rd10, %rd9, %rd5;\n\
    add.u64 %rd10, %rd10, %rd7;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd1, %rd10;\n\
    ld.global.f32 %f1, [%rd10];\n\
    mul.lo.u64 %rd11, %rd6, %rd5;\n\
    add.u64 %rd11, %rd11, %rd7;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd3, %rd11;\n\
    st.global.f32 [%rd11], %f1;\n\
DONE: ret;\n\
}\0";

/// GPU embedding lookup kernel with i32 integer indices.
/// Same as EMBEDDING_F32_PTX but reads indices via ld.global.s32 + cvt.s64.s32
/// instead of ld.global.f32 + cvt.rzi.u64.f32 to correctly handle i32 token IDs.
pub(crate) const EMBEDDING_I32IDX_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_embedding_i32idx(\n\
    .param .u64 weight, .param .u64 indices, .param .u64 out,\n\
    .param .u64 seq_len, .param .u64 embed_dim\n\
) {\n\
    .reg .u64 %rd<12>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p1;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    ld.param.u64 %rd1, [weight];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [seq_len];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    cvt.u64.u32 %rd6, %r1;\n\
    cvt.u64.u32 %rd7, %r3;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra DONE;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DONE;\n\
    shl.b64 %rd8, %rd6, 2;\n\
    add.u64 %rd8, %rd2, %rd8;\n\
    ld.global.s32 %r5, [%rd8];\n\
    cvt.s64.s32 %rd9, %r5;\n\
    mul.lo.u64 %rd10, %rd9, %rd5;\n\
    add.u64 %rd10, %rd10, %rd7;\n\
    shl.b64 %rd10, %rd10, 2;\n\
    add.u64 %rd10, %rd1, %rd10;\n\
    ld.global.f32 %f1, [%rd10];\n\
    mul.lo.u64 %rd11, %rd6, %rd5;\n\
    add.u64 %rd11, %rd11, %rd7;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd3, %rd11;\n\
    st.global.f32 [%rd11], %f1;\n\
DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Bias Add
// Thread i handles element out[i] = in[i] + bias[i % cols]
// Grid:  (ceil(rows*cols / 256), 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, bias ptr, out ptr, total (rows*cols, u64), cols (u64)
// ---------------------------------------------------------------------------
pub(crate) const BIAS_ADD_F32_PTX: &str = "\
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
    .reg .f32 %f<3>;\n\
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

// ---------------------------------------------------------------------------
// GPU Softmax (per-row, numerically stable)
// One thread block per row. Uses shared memory for max and sum reductions.
// Grid:  (num_rows, 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, out ptr, rows (u64), cols (u64)
// Algorithm: for each row — find max, subtract max + exp, sum, divide
// Each thread handles multiple columns via stride loop.
// ---------------------------------------------------------------------------
pub(crate) const SOFTMAX_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_softmax_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 rows, .param .u64 cols\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<8>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 smax[256];\n\
    .shared .f32 ssum[256];\n\
    .reg .u32 %r8;\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [rows];\n\
    ld.param.u64 %rd4, [cols];\n\
    // row = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd5, %r1;\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra DONE;\n\
    // tid = threadIdx.x\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd6, %r2;\n\
    // row_base = row * cols * 4\n\
    mul.lo.u64 %rd7, %rd5, %rd4;\n\
    shl.b64 %rd7, %rd7, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    add.u64 %rd9, %rd2, %rd7;\n\
    // --- Pass 1: find row max ---\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd10, %rd6;\n\
MAX_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra MAX_DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd11, %rd8, %rd11;\n\
    ld.global.f32 %f2, [%rd11];\n\
    max.f32 %f1, %f1, %f2;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra MAX_LOOP;\n\
MAX_DONE:\n\
    // Store local max to shared memory\n\
    cvt.u32.u64 %r3, %rd6;\n\
    mul.lo.u32 %r3, %r3, 4;\n\
    mov.u32 %r8, smax; add.u32 %r8, %r8, %r3; st.shared.f32 [%r8], %f1;\n\
    bar.sync 0;\n\
    // Reduce shared memory max (thread 0 only, simple sequential)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra SKIP_MAX_REDUCE;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r5, %ntid.x;\n\
REDUCE_MAX:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra DONE_MAX_REDUCE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r8, smax; add.u32 %r8, %r8, %r6; ld.shared.f32 %f3, [%r8];\n\
    max.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra REDUCE_MAX;\n\
DONE_MAX_REDUCE:\n\
    st.shared.f32 [smax], %f1;\n\
SKIP_MAX_REDUCE:\n\
    bar.sync 0;\n\
    // Load global max\n\
    ld.shared.f32 %f1, [smax];\n\
    // --- Pass 2: exp(x - max) and partial sum ---\n\
    mov.f32 %f4, 0f00000000;\n\
    mov.u64 %rd10, %rd6;\n\
EXP_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra EXP_DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd12, %rd8, %rd11;\n\
    ld.global.f32 %f2, [%rd12];\n\
    sub.f32 %f2, %f2, %f1;\n\
    // exp(x) = 2^(x * log2(e))\n\
    mul.f32 %f2, %f2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f2, %f2;\n\
    add.u64 %rd12, %rd9, %rd11;\n\
    st.global.f32 [%rd12], %f2;\n\
    add.f32 %f4, %f4, %f2;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra EXP_LOOP;\n\
EXP_DONE:\n\
    // Store partial sum to shared memory\n\
    cvt.u32.u64 %r3, %rd6;\n\
    mul.lo.u32 %r3, %r3, 4;\n\
    mov.u32 %r8, ssum; add.u32 %r8, %r8, %r3; st.shared.f32 [%r8], %f4;\n\
    bar.sync 0;\n\
    // Reduce shared memory sum (thread 0)\n\
    @%p3 bra SKIP_SUM_REDUCE;\n\
    mov.u32 %r4, 1;\n\
REDUCE_SUM:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra DONE_SUM_REDUCE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r8, ssum; add.u32 %r8, %r8, %r6; ld.shared.f32 %f5, [%r8];\n\
    add.f32 %f4, %f4, %f5;\n\
    add.u32 %r4, %r4, 1;\n\
    bra REDUCE_SUM;\n\
DONE_SUM_REDUCE:\n\
    rcp.approx.f32 %f4, %f4;\n\
    st.shared.f32 [ssum], %f4;\n\
SKIP_SUM_REDUCE:\n\
    bar.sync 0;\n\
    // Load 1/sum\n\
    ld.shared.f32 %f4, [ssum];\n\
    // --- Pass 3: divide by sum ---\n\
    mov.u64 %rd10, %rd6;\n\
DIV_LOOP:\n\
    setp.ge.u64 %p2, %rd10, %rd4;\n\
    @%p2 bra DONE;\n\
    shl.b64 %rd11, %rd10, 2;\n\
    add.u64 %rd12, %rd9, %rd11;\n\
    ld.global.f32 %f6, [%rd12];\n\
    mul.f32 %f6, %f6, %f4;\n\
    st.global.f32 [%rd12], %f6;\n\
    add.u64 %rd10, %rd10, 256;\n\
    bra DIV_LOOP;\n\
DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Per-dimension Sum Reduction
// One thread block per output element.
// Decomposes the reduction as: outer * reduce_size * inner
// Each block sums reduce_size elements spaced `inner` apart.
// Grid:  (outer * inner, 1, 1) — one block per output element
// Block: (256, 1, 1)
// Params: in ptr, out ptr, outer (u64), reduce_size (u64), inner (u64)
// ---------------------------------------------------------------------------
pub(crate) const SUM_DIM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_sum_dim_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 outer, .param .u64 reduce_size, .param .u64 inner\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [outer];\n\
    ld.param.u64 %rd4, [reduce_size];\n\
    ld.param.u64 %rd5, [inner];\n\
    // block_id = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    // total_out = outer * inner\n\
    mul.lo.u64 %rd7, %rd3, %rd5;\n\
    setp.ge.u64 %p1, %rd6, %rd7;\n\
    @%p1 bra DONE;\n\
    // outer_idx = block_id / inner\n\
    div.u64 %rd8, %rd6, %rd5;\n\
    // inner_idx = block_id % inner\n\
    rem.u64 %rd9, %rd6, %rd5;\n\
    // base_in = (outer_idx * reduce_size * inner + inner_idx) * 4\n\
    mul.lo.u64 %rd10, %rd8, %rd4;\n\
    mul.lo.u64 %rd10, %rd10, %rd5;\n\
    add.u64 %rd10, %rd10, %rd9;\n\
    // stride_in = inner (elements between consecutive reduce elements)\n\
    // tid\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd11, %r2;\n\
    // Partial sum via stride loop\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd12, %rd11;\n\
SUM_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd4;\n\
    @%p2 bra SUM_DONE;\n\
    // addr = (base_in + k * inner) * 4\n\
    mul.lo.u64 %rd13, %rd12, %rd5;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd1, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra SUM_LOOP;\n\
SUM_DONE:\n\
    // Store partial sum to shared memory\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Tree reduction in shared memory\n\
    mov.u32 %r4, 128;\n\
REDUCE_LOOP:\n\
    setp.lt.u32 %p2, %r4, 1;\n\
    @%p2 bra REDUCE_DONE;\n\
    setp.ge.u32 %p3, %r2, %r4;\n\
    @%p3 bra SKIP_REDUCE;\n\
    mul.lo.u32 %r5, %r2, 4;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    ld.shared.f32 %f2, [%r7];\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f2, %f2, %f3;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    st.shared.f32 [%r7], %f2;\n\
SKIP_REDUCE:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra REDUCE_LOOP;\n\
REDUCE_DONE:\n\
    // Thread 0 writes result\n\
    setp.ne.u32 %p2, %r2, 0;\n\
    @%p2 bra DONE;\n\
    ld.shared.f32 %f1, [sdata];\n\
    // out[block_id] = sum\n\
    shl.b64 %rd14, %rd6, 2;\n\
    add.u64 %rd14, %rd2, %rd14;\n\
    st.global.f32 [%rd14], %f1;\n\
DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Per-dimension Max Reduction
// Same structure as sum but uses max.f32 instead of add.f32.
// Grid:  (outer * inner, 1, 1) — one block per output element
// Block: (256, 1, 1)
// Params: in ptr, out ptr, outer (u64), reduce_size (u64), inner (u64)
// ---------------------------------------------------------------------------
pub(crate) const MAX_DIM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_max_dim_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 outer, .param .u64 reduce_size, .param .u64 inner\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [outer];\n\
    ld.param.u64 %rd4, [reduce_size];\n\
    ld.param.u64 %rd5, [inner];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    mul.lo.u64 %rd7, %rd3, %rd5;\n\
    setp.ge.u64 %p1, %rd6, %rd7;\n\
    @%p1 bra DONE;\n\
    div.u64 %rd8, %rd6, %rd5;\n\
    rem.u64 %rd9, %rd6, %rd5;\n\
    mul.lo.u64 %rd10, %rd8, %rd4;\n\
    mul.lo.u64 %rd10, %rd10, %rd5;\n\
    add.u64 %rd10, %rd10, %rd9;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd11, %r2;\n\
    // Initialize with -inf\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd12, %rd11;\n\
MAX_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd4;\n\
    @%p2 bra MAX_DONE;\n\
    mul.lo.u64 %rd13, %rd12, %rd5;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    shl.b64 %rd13, %rd13, 2;\n\
    add.u64 %rd13, %rd1, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    max.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra MAX_LOOP;\n\
MAX_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    mov.u32 %r4, 128;\n\
REDUCE_LOOP:\n\
    setp.lt.u32 %p2, %r4, 1;\n\
    @%p2 bra REDUCE_DONE;\n\
    setp.ge.u32 %p3, %r2, %r4;\n\
    @%p3 bra SKIP_REDUCE;\n\
    mul.lo.u32 %r5, %r2, 4;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    ld.shared.f32 %f2, [%r7];\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    max.f32 %f2, %f2, %f3;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    st.shared.f32 [%r7], %f2;\n\
SKIP_REDUCE:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra REDUCE_LOOP;\n\
REDUCE_DONE:\n\
    setp.ne.u32 %p2, %r2, 0;\n\
    @%p2 bra DONE;\n\
    ld.shared.f32 %f1, [sdata];\n\
    shl.b64 %rd14, %rd6, 2;\n\
    add.u64 %rd14, %rd2, %rd14;\n\
    st.global.f32 [%rd14], %f1;\n\
DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Global Sum Reduction (all elements to a single scalar)
// One block, shared memory tree reduction.
// Grid:  (1, 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, out ptr, n (u64)
// ---------------------------------------------------------------------------
pub(crate) const GLOBAL_SUM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_global_sum_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 n\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [n];\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd4, %r2;\n\
    // Partial sum via stride loop\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd5, %rd4;\n\
GSUM_LOOP:\n\
    setp.ge.u64 %p1, %rd5, %rd3;\n\
    @%p1 bra GSUM_DONE;\n\
    shl.b64 %rd6, %rd5, 2;\n\
    add.u64 %rd6, %rd1, %rd6;\n\
    ld.global.f32 %f2, [%rd6];\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd5, %rd5, 256;\n\
    bra GSUM_LOOP;\n\
GSUM_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    mov.u32 %r4, 128;\n\
GREDUCE_LOOP:\n\
    setp.lt.u32 %p1, %r4, 1;\n\
    @%p1 bra GREDUCE_DONE;\n\
    setp.ge.u32 %p2, %r2, %r4;\n\
    @%p2 bra GSKIP;\n\
    mul.lo.u32 %r5, %r2, 4;\n\
    add.u32 %r6, %r2, %r4;\n\
    mul.lo.u32 %r6, %r6, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    ld.shared.f32 %f2, [%r7];\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f2, %f2, %f3;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r5;\n\
    st.shared.f32 [%r7], %f2;\n\
GSKIP:\n\
    bar.sync 0;\n\
    shr.u32 %r4, %r4, 1;\n\
    bra GREDUCE_LOOP;\n\
GREDUCE_DONE:\n\
    setp.ne.u32 %p1, %r2, 0;\n\
    @%p1 bra GDONE;\n\
    ld.shared.f32 %f1, [sdata];\n\
    st.global.f32 [%rd2], %f1;\n\
GDONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU LayerNorm (per-row, fused mean + variance + normalize + scale + shift)
// One thread block per row. 2-pass algorithm for numerical stability.
// Grid:  (num_rows, 1, 1) — one block per row
// Block: (256, 1, 1)
// Params: in ptr, out ptr, gamma ptr, beta ptr, rows (u64), cols (u64), eps (f32)
//
// Pass 1: Compute mean = sum(x) / cols
// Pass 2: Compute var = sum((x - mean)^2) / cols, then normalize:
//         out = gamma * (x - mean) * rsqrt(var + eps) + beta
// ---------------------------------------------------------------------------
pub(crate) const LAYERNORM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_layernorm_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 gamma, .param .u64 beta,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<12>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [beta];\n\
    ld.param.u64 %rd5, [rows];\n\
    ld.param.u64 %rd6, [cols];\n\
    ld.param.f32 %f10, [eps];\n\
    // row = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra LN_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
    // row_base = row * cols * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
    add.u64 %rd11, %rd2, %rd9;\n\
    // --- Pass 1: compute partial sum for mean ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd12, %rd8;\n\
LN_MEAN_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd6;\n\
    @%p2 bra LN_MEAN_DONE;\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra LN_MEAN_LOOP;\n\
LN_MEAN_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Reduce sum for mean (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra LN_SKIP_MEAN;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r5, %ntid.x;\n\
LN_RMEAN:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra LN_DMEAN;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LN_RMEAN;\n\
LN_DMEAN:\n\
    // mean = sum / cols\n\
    cvt.rn.f32.u64 %f4, %rd6;\n\
    div.approx.f32 %f1, %f1, %f4;\n\
    st.shared.f32 [sdata], %f1;\n\
LN_SKIP_MEAN:\n\
    bar.sync 0;\n\
    // Load mean\n\
    ld.shared.f32 %f5, [sdata];\n\
    // --- Pass 2: compute partial sum of (x - mean)^2 for variance ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd12, %rd8;\n\
LN_VAR_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd6;\n\
    @%p2 bra LN_VAR_DONE;\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd13, %rd10, %rd13;\n\
    ld.global.f32 %f2, [%rd13];\n\
    sub.f32 %f2, %f2, %f5;\n\
    mul.f32 %f2, %f2, %f2;\n\
    add.f32 %f1, %f1, %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra LN_VAR_LOOP;\n\
LN_VAR_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Reduce sum for variance (thread 0)\n\
    @%p3 bra LN_SKIP_VAR;\n\
    mov.u32 %r4, 1;\n\
LN_RVAR:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra LN_DVAR;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f3, [%r7];\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LN_RVAR;\n\
LN_DVAR:\n\
    // var = sum_sq / cols\n\
    cvt.rn.f32.u64 %f4, %rd6;\n\
    div.approx.f32 %f1, %f1, %f4;\n\
    // inv_std = rsqrt(var + eps)\n\
    add.f32 %f1, %f1, %f10;\n\
    rsqrt.approx.f32 %f1, %f1;\n\
    st.shared.f32 [sdata], %f1;\n\
LN_SKIP_VAR:\n\
    bar.sync 0;\n\
    // Load inv_std\n\
    ld.shared.f32 %f6, [sdata];\n\
    // --- Pass 3: normalize, scale, shift ---\n\
    mov.u64 %rd12, %rd8;\n\
LN_NORM_LOOP:\n\
    setp.ge.u64 %p2, %rd12, %rd6;\n\
    @%p2 bra LN_DONE;\n\
    shl.b64 %rd13, %rd12, 2;\n\
    add.u64 %rd14, %rd10, %rd13;\n\
    ld.global.f32 %f2, [%rd14];\n\
    sub.f32 %f2, %f2, %f5;\n\
    mul.f32 %f2, %f2, %f6;\n\
    // gamma[j]\n\
    add.u64 %rd15, %rd3, %rd13;\n\
    ld.global.f32 %f7, [%rd15];\n\
    mul.f32 %f2, %f2, %f7;\n\
    // beta[j]\n\
    add.u64 %rd15, %rd4, %rd13;\n\
    ld.global.f32 %f8, [%rd15];\n\
    add.f32 %f2, %f2, %f8;\n\
    // Store\n\
    add.u64 %rd15, %rd11, %rd13;\n\
    st.global.f32 [%rd15], %f2;\n\
    add.u64 %rd12, %rd12, 256;\n\
    bra LN_NORM_LOOP;\n\
LN_DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU RMSNorm (per-row, fused rms + normalize + scale)
// One thread block per row. Simpler than LayerNorm (no mean subtraction).
// rms = sqrt(mean(x^2) + eps), out = gamma * x / rms
// Grid:  (num_rows, 1, 1) — one block per row
// Block: (256, 1, 1)
// Params: in ptr, out ptr, gamma ptr, rows (u64), cols (u64), eps (f32)
// ---------------------------------------------------------------------------
pub(crate) const RMSNORM_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_f32(\n\
    .param .u64 inp, .param .u64 out,\n\
    .param .u64 gamma,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<20>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<10>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [inp];\n\
    ld.param.u64 %rd2, [out];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [rows];\n\
    ld.param.u64 %rd5, [cols];\n\
    ld.param.f32 %f8, [eps];\n\
    // row = blockIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd6, %r1;\n\
    setp.ge.u64 %p1, %rd6, %rd4;\n\
    @%p1 bra RMS_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd7, %r2;\n\
    // row_base = row * cols * 4\n\
    mul.lo.u64 %rd8, %rd6, %rd5;\n\
    shl.b64 %rd8, %rd8, 2;\n\
    add.u64 %rd9, %rd1, %rd8;\n\
    add.u64 %rd10, %rd2, %rd8;\n\
    // --- Pass 1: compute partial sum of x^2 ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd11, %rd7;\n\
RMS_SQ_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra RMS_SQ_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    add.u64 %rd12, %rd9, %rd12;\n\
    ld.global.f32 %f2, [%rd12];\n\
    mul.f32 %f3, %f2, %f2;\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra RMS_SQ_LOOP;\n\
RMS_SQ_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    // Reduce sum_sq (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra RMS_SKIP_REDUCE;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r5, %ntid.x;\n\
RMS_RLOOP:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra RMS_RDONE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f4, [%r7];\n\
    add.f32 %f1, %f1, %f4;\n\
    add.u32 %r4, %r4, 1;\n\
    bra RMS_RLOOP;\n\
RMS_RDONE:\n\
    // rms_inv = rsqrt(sum_sq / cols + eps)\n\
    cvt.rn.f32.u64 %f5, %rd5;\n\
    div.approx.f32 %f1, %f1, %f5;\n\
    add.f32 %f1, %f1, %f8;\n\
    rsqrt.approx.f32 %f1, %f1;\n\
    st.shared.f32 [sdata], %f1;\n\
RMS_SKIP_REDUCE:\n\
    bar.sync 0;\n\
    // Load rms_inv\n\
    ld.shared.f32 %f6, [sdata];\n\
    // --- Pass 2: normalize and scale ---\n\
    mov.u64 %rd11, %rd7;\n\
RMS_NORM_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra RMS_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    add.u64 %rd13, %rd9, %rd12;\n\
    ld.global.f32 %f2, [%rd13];\n\
    mul.f32 %f2, %f2, %f6;\n\
    // gamma[j]\n\
    add.u64 %rd14, %rd3, %rd12;\n\
    ld.global.f32 %f7, [%rd14];\n\
    mul.f32 %f2, %f2, %f7;\n\
    // Store\n\
    add.u64 %rd14, %rd10, %rd12;\n\
    st.global.f32 [%rd14], %f2;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra RMS_NORM_LOOP;\n\
RMS_DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Scatter-Add (for embedding backward / gradient accumulation)
// Thread (i, j): atomicAdd(out[indices[i], j], src[i, j])
// Grid:  (ceil(num_indices / 16), ceil(embed_dim / 16), 1)
// Block: (16, 16, 1)
// Params: src ptr (grad), indices ptr, out ptr (grad_weight),
//         num_indices (u64), embed_dim (u64), vocab_size (u64)
//
// Uses atomicAdd (atom.global.add.f32) since multiple indices may alias
// the same row in the output — this is the standard embedding backward pattern.
//
// Target: sm_80 (Ampere base, compatible with Ada Lovelace sm_89, Hopper sm_90, Blackwell sm_100)
// ---------------------------------------------------------------------------
pub(crate) const SCATTER_ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_scatter_add_f32(\n\
    .param .u64 src, .param .u64 indices, .param .u64 out,\n\
    .param .u64 num_indices, .param .u64 embed_dim, .param .u64 vocab_size\n\
) {\n\
    .reg .u64 %rd<16>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f1;\n\
    .reg .pred %p<3>;\n\
    // i = blockIdx.x * blockDim.x + threadIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    // j = blockIdx.y * blockDim.y + threadIdx.y\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    // Load params\n\
    ld.param.u64 %rd1, [src];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [num_indices];\n\
    ld.param.u64 %rd5, [embed_dim];\n\
    ld.param.u64 %rd6, [vocab_size];\n\
    // Bounds check: i < num_indices, j < embed_dim\n\
    cvt.u64.u32 %rd7, %r1;\n\
    cvt.u64.u32 %rd8, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra SA_DONE;\n\
    setp.ge.u64 %p2, %rd8, %rd5;\n\
    @%p2 bra SA_DONE;\n\
    // Load index: idx = (int)indices[i]\n\
    shl.b64 %rd9, %rd7, 2;\n\
    add.u64 %rd9, %rd2, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    cvt.rzi.u64.f32 %rd10, %f1;\n\
    // Bounds check: idx < vocab_size\n\
    setp.ge.u64 %p1, %rd10, %rd6;\n\
    @%p1 bra SA_DONE;\n\
    // Load src[i, j] = src[i * embed_dim + j]\n\
    mul.lo.u64 %rd11, %rd7, %rd5;\n\
    add.u64 %rd11, %rd11, %rd8;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    // Atomic add: out[idx, j] += src[i, j]\n\
    // out_addr = out + (idx * embed_dim + j) * 4\n\
    mul.lo.u64 %rd12, %rd10, %rd5;\n\
    add.u64 %rd12, %rd12, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd3, %rd12;\n\
    atom.global.add.f32 %f1, [%rd12], %f1;\n\
SA_DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Gather (general dim-0 gather for any 2D+ tensor)
// Thread (i, j): out[i, j] = input[indices[i], j]
// Grid:  (ceil(num_indices / 16), ceil(inner_dim / 16), 1)
// Block: (16, 16, 1)
// Params: input ptr, indices ptr, out ptr,
//         num_indices (u64), inner_dim (u64), input_rows (u64)
//
// Identical to embedding lookup but with explicit bounds checking on input_rows.
// Target: sm_80 (compatible sm_89, sm_90, sm_100 Blackwell)
// ---------------------------------------------------------------------------
pub(crate) const GATHER_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_gather_f32(\n\
    .param .u64 input, .param .u64 indices, .param .u64 out,\n\
    .param .u64 num_indices, .param .u64 inner_dim, .param .u64 input_rows\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<3>;\n\
    // i = blockIdx.x * blockDim.x + threadIdx.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    // j = blockIdx.y * blockDim.y + threadIdx.y\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    // Load params\n\
    ld.param.u64 %rd1, [input];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [num_indices];\n\
    ld.param.u64 %rd5, [inner_dim];\n\
    ld.param.u64 %rd6, [input_rows];\n\
    // Bounds check\n\
    cvt.u64.u32 %rd7, %r1;\n\
    cvt.u64.u32 %rd8, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra G_DONE;\n\
    setp.ge.u64 %p2, %rd8, %rd5;\n\
    @%p2 bra G_DONE;\n\
    // Load index: idx = (int)indices[i]\n\
    shl.b64 %rd9, %rd7, 2;\n\
    add.u64 %rd9, %rd2, %rd9;\n\
    ld.global.f32 %f1, [%rd9];\n\
    cvt.rzi.u64.f32 %rd10, %f1;\n\
    // Bounds check: idx < input_rows\n\
    setp.ge.u64 %p1, %rd10, %rd6;\n\
    @%p1 bra G_DONE;\n\
    // Load input[idx, j]\n\
    mul.lo.u64 %rd11, %rd10, %rd5;\n\
    add.u64 %rd11, %rd11, %rd8;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    // Store out[i, j]\n\
    mul.lo.u64 %rd12, %rd7, %rd5;\n\
    add.u64 %rd12, %rd12, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd3, %rd12;\n\
    st.global.f32 [%rd12], %f1;\n\
G_DONE: ret;\n\
}\0";

/// GPU gather kernel with i32 integer indices.
/// Same as GATHER_F32_PTX but reads indices via ld.global.s32 + cvt.s64.s32.
pub(crate) const GATHER_I32IDX_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_gather_i32idx(\n\
    .param .u64 input, .param .u64 indices, .param .u64 out,\n\
    .param .u64 num_indices, .param .u64 inner_dim, .param .u64 input_rows\n\
) {\n\
    .reg .u64 %rd<14>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<2>;\n\
    .reg .pred %p<3>;\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    mov.u32 %r3, %ctaid.y;\n\
    mov.u32 %r4, %ntid.y;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r4, %tid.y;\n\
    add.u32 %r3, %r3, %r4;\n\
    ld.param.u64 %rd1, [input];\n\
    ld.param.u64 %rd2, [indices];\n\
    ld.param.u64 %rd3, [out];\n\
    ld.param.u64 %rd4, [num_indices];\n\
    ld.param.u64 %rd5, [inner_dim];\n\
    ld.param.u64 %rd6, [input_rows];\n\
    cvt.u64.u32 %rd7, %r1;\n\
    cvt.u64.u32 %rd8, %r3;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra G_DONE;\n\
    setp.ge.u64 %p2, %rd8, %rd5;\n\
    @%p2 bra G_DONE;\n\
    // Load index as i32: idx = indices[i]\n\
    shl.b64 %rd9, %rd7, 2;\n\
    add.u64 %rd9, %rd2, %rd9;\n\
    ld.global.s32 %r5, [%rd9];\n\
    cvt.s64.s32 %rd10, %r5;\n\
    setp.ge.u64 %p1, %rd10, %rd6;\n\
    @%p1 bra G_DONE;\n\
    mul.lo.u64 %rd11, %rd10, %rd5;\n\
    add.u64 %rd11, %rd11, %rd8;\n\
    shl.b64 %rd11, %rd11, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    mul.lo.u64 %rd12, %rd7, %rd5;\n\
    add.u64 %rd12, %rd12, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd3, %rd12;\n\
    st.global.f32 [%rd12], %f1;\n\
G_DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Conv2d (implicit GEMM, direct convolution)
// Each thread computes one output element: out[n, co, oh, ow]
// Grid:  (total_output_elements / 256 + 1, 1, 1) — 1D flat launch
// Block: (256, 1, 1)
// Params: input ptr, weight ptr, bias ptr (0=no bias), out ptr,
//         N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w,
//         H_out, W_out, total (all u64)
//
// Input layout: NCHW [N, C_in, H, W]
// Weight layout: [C_out, C_in, kH, kW]
// Output layout: NCHW [N, C_out, H_out, W_out]
//
// Target: sm_80 (Ampere base, compatible Ada sm_89, Hopper sm_90, Blackwell sm_100)
// ---------------------------------------------------------------------------
pub(crate) const CONV2D_F32_PTX: &str = "\
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

// ---------------------------------------------------------------------------
// GPU MaxPool2d
// Each thread computes one output element: out[n, c, oh, ow] = max over window
// Also stores argmax index for backward pass.
// Grid:  (total_output_elements / 256 + 1, 1, 1) — 1D flat launch
// Block: (256, 1, 1)
// Params: input ptr, out ptr, argmax ptr (i64 indices),
//         N, C, H, W, kH, kW, stride, padding, H_out, W_out, total (all u64)
//
// Target: sm_80 (compatible sm_89, sm_90, sm_100 Blackwell)
// ---------------------------------------------------------------------------
pub(crate) const MAXPOOL2D_F32_PTX: &str = "\
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
    // ih was computed above but we used %rd16 — recompute\n\
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

// ---------------------------------------------------------------------------
// GPU Dropout (inverted dropout with Philox-style PRNG)
// Each thread: generate random u32 via hash(seed + idx), compare with threshold,
// output = keep ? input * scale : 0
// Also writes mask (f32: 1.0 or 0.0) for backward pass.
// Grid:  (ceil(len / 256), 1, 1)
// Block: (256, 1, 1)
// Params: input ptr, out ptr, mask ptr, len (u64), threshold (u32), scale (f32), seed (u64)
//
// Uses a simple multiply-xorshift hash for per-element randomness.
// Not cryptographically secure but sufficient for dropout.
//
// Target: sm_80 (compatible sm_89, sm_90, sm_100 Blackwell)
// ---------------------------------------------------------------------------
pub(crate) const DROPOUT_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_dropout_f32(\n\
    .param .u64 inp, .param .u64 out, .param .u64 mask,\n\
    .param .u64 len, .param .u32 threshold, .param .f32 scale, .param .u64 seed\n\
) {\n\
    .reg .u64 %rd<10>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<4>;\n\
    .reg .pred %p<3>;\n\
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
    ld.param.u64 %rd3, [mask];\n\
    ld.param.u64 %rd4, [len];\n\
    ld.param.u32 %r3, [threshold];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u64 %rd5, [seed];\n\
    // Bounds check\n\
    setp.ge.u64 %p1, %rd0, %rd4;\n\
    @%p1 bra DROP_DONE;\n\
    // Philox-style hash: hash = (seed + idx) * 0x9E3779B9; hash ^= hash >> 16; hash *= 0x85EBCA6B\n\
    add.u64 %rd6, %rd5, %rd0;\n\
    cvt.u32.u64 %r4, %rd6;\n\
    mul.lo.u32 %r4, %r4, 0x9E3779B9;\n\
    shr.u32 %r5, %r4, 16;\n\
    xor.b32 %r4, %r4, %r5;\n\
    mul.lo.u32 %r4, %r4, 0x85EBCA6B;\n\
    shr.u32 %r5, %r4, 13;\n\
    xor.b32 %r4, %r4, %r5;\n\
    mul.lo.u32 %r4, %r4, 0xC2B2AE35;\n\
    shr.u32 %r5, %r4, 16;\n\
    xor.b32 %r4, %r4, %r5;\n\
    // Compare: keep = (hash < threshold)\n\
    setp.lt.u32 %p2, %r4, %r3;\n\
    // Load input\n\
    shl.b64 %rd7, %rd0, 2;\n\
    add.u64 %rd8, %rd1, %rd7;\n\
    ld.global.f32 %f2, [%rd8];\n\
    // Compute output and mask\n\
    selp.f32 %f3, %f1, 0f00000000, %p2;\n\
    mul.f32 %f2, %f2, %f3;\n\
    // mask = keep ? 1.0 : 0.0\n\
    selp.f32 %f3, 0f3F800000, 0f00000000, %p2;\n\
    // Store output\n\
    add.u64 %rd9, %rd2, %rd7;\n\
    st.global.f32 [%rd9], %f2;\n\
    // Store mask\n\
    add.u64 %rd9, %rd3, %rd7;\n\
    st.global.f32 [%rd9], %f3;\n\
DROP_DONE: ret;\n\
}\0";

// ---------------------------------------------------------------------------
// GPU Strided Batched Matmul (cuBLAS-style BMM)
// Single kernel launch handles ALL batch slices via blockIdx.z.
// Each thread computes one output element: C[b, row, col] = sum_k(A[ba, row, k] * B[bb, k, col])
//
// Grid:  (ceil(N/16), ceil(M/16), batch_count) — z dimension = batch
// Block: (16, 16, 1)
//
// Params: A ptr, B ptr, C ptr,
//         M, N, K (matrix dims, u64),
//         batch_count (u64),
//         stride_A (u64, elements per batch = M*K, or 0 for broadcast),
//         stride_B (u64, elements per batch = K*N, or 0 for broadcast),
//         stride_C (u64, elements per batch = M*N)
//
// Broadcast: stride=0 means tensor is shared across all batches (e.g. single weight matrix).
//
// Target: sm_80 (Ampere base, compatible Ada sm_89, Hopper sm_90, Blackwell sm_100)
// ---------------------------------------------------------------------------
pub(crate) const BMM_F32_PTX: &str = "\
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

// ---------------------------------------------------------------------------
// GPU Strided Copy (makes non-contiguous views contiguous on-device)
// Each thread copies one element: dst[flat_idx] = src[strided_offset(flat_idx)]
//
// For each flat output index, decompose into N-dim coordinates using the shape,
// then compute the source offset using the source (non-contiguous) strides.
// Shape and stride arrays live in GPU-accessible global memory.
//
// Grid:  (ceil(total / 256), 1, 1)
// Block: (256, 1, 1)
// Params: src_data ptr, dst_data ptr, shape ptr (i64[ndim]),
//         src_strides ptr (i64[ndim]), dst_strides ptr (i64[ndim]),
//         ndim (u64), total (u64)
//
// The shape-based decomposition handles edge cases (zero strides from expand,
// stride-0 broadcast dimensions) correctly because coord is clamped by
// shape[dim] via the mod operation.
//
// Target: sm_80 (compatible sm_89, sm_90, sm_100 Blackwell)
// ---------------------------------------------------------------------------
pub(crate) const STRIDED_COPY_F32_PTX: &str = "\
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
