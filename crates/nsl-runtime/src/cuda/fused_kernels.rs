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
.target sm_52\n\
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

// ---------------------------------------------------------------------------
// GPU Bias Add
// Thread i handles element out[i] = in[i] + bias[i % cols]
// Grid:  (ceil(rows*cols / 256), 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, bias ptr, out ptr, total (rows*cols, u64), cols (u64)
// ---------------------------------------------------------------------------
pub(crate) const BIAS_ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
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
.target sm_52\n\
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
    st.shared.f32 [smax + %r3], %f1;\n\
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
    ld.shared.f32 %f3, [smax + %r6];\n\
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
    st.shared.f32 [ssum + %r3], %f4;\n\
    bar.sync 0;\n\
    // Reduce shared memory sum (thread 0)\n\
    @%p3 bra SKIP_SUM_REDUCE;\n\
    mov.u32 %r4, 1;\n\
REDUCE_SUM:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra DONE_SUM_REDUCE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    ld.shared.f32 %f5, [ssum + %r6];\n\
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
.target sm_52\n\
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
    st.shared.f32 [sdata + %r3], %f1;\n\
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
    ld.shared.f32 %f2, [sdata + %r5];\n\
    ld.shared.f32 %f3, [sdata + %r6];\n\
    add.f32 %f2, %f2, %f3;\n\
    st.shared.f32 [sdata + %r5], %f2;\n\
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
.target sm_52\n\
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
    st.shared.f32 [sdata + %r3], %f1;\n\
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
    ld.shared.f32 %f2, [sdata + %r5];\n\
    ld.shared.f32 %f3, [sdata + %r6];\n\
    max.f32 %f2, %f2, %f3;\n\
    st.shared.f32 [sdata + %r5], %f2;\n\
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
.target sm_52\n\
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
    st.shared.f32 [sdata + %r3], %f1;\n\
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
    ld.shared.f32 %f2, [sdata + %r5];\n\
    ld.shared.f32 %f3, [sdata + %r6];\n\
    add.f32 %f2, %f2, %f3;\n\
    st.shared.f32 [sdata + %r5], %f2;\n\
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
.target sm_52\n\
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
    st.shared.f32 [sdata + %r3], %f1;\n\
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
    ld.shared.f32 %f3, [sdata + %r6];\n\
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
    st.shared.f32 [sdata + %r3], %f1;\n\
    bar.sync 0;\n\
    // Reduce sum for variance (thread 0)\n\
    @%p3 bra LN_SKIP_VAR;\n\
    mov.u32 %r4, 1;\n\
LN_RVAR:\n\
    setp.ge.u32 %p2, %r4, %r5;\n\
    @%p2 bra LN_DVAR;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    ld.shared.f32 %f3, [sdata + %r6];\n\
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
.target sm_52\n\
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
    st.shared.f32 [sdata + %r3], %f1;\n\
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
    ld.shared.f32 %f4, [sdata + %r6];\n\
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
