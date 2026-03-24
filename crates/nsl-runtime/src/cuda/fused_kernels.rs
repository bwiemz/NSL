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

// Note: LayerNorm and RMSNorm use a CPU-redirect path (see tensor/mod.rs).
// A full fused PTX layernorm requires shared-memory warp reductions which
// would add significant complexity and are deferred to a follow-up milestone.
// The CPU-redirect ensures correct device tagging without silent miscomputes.
