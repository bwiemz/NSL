//! The hand-written CSHA Tier B.1 pre-pass kernels, frozen (roadmap A2
//! step 11).
//!
//! `nsl_runtime::cuda::tier_b1_prepass::{CSHA_TIER_B1_PREPASS_X_PTX,
//! CSHA_TIER_B1_PREPASS_W_PTX}` as they stood when the kernels moved to
//! `nsl_kir::kernels::tier_b1_prepass`, verbatim. They are the reference
//! side of `tier_b1_prepass_kir_equivalence.rs` and are not compiled into
//! anything else.

// ---------------------------------------------------------------------------
// X pre-pass: RMSNorm + narrow + chunkify
//
// Grid:    (seq, 1, 1)     — one CTA per row
// Block:   (256, 1, 1)
// Inputs:
//   x_in      : f32 [seq, d_model]               (row-major)
//   gamma     : f32 [d_model]                    (RMSNorm scale)
//   x_out     : f16 [d_model/chunk, seq, chunk]  (chunks-major)
//   seq, d_model, chunk : u64
//   log2_chunk          : u32 (chunk is power of 2; from chunk_config::select)
//   eps                 : f32
//
// Each CTA processes one row of x. Pass 1: compute partial sum-of-squares
// across the row (reduction in sdata). Pass 2: each thread normalizes,
// applies gamma, narrows to f16, and writes to the chunkified output at
// `(chunk_idx, row, c) → byte (chunk_idx * seq * chunk + row * chunk + c) * 2`
// where chunk_idx = d >> log2_chunk and c = d & (chunk - 1).
// ---------------------------------------------------------------------------
pub const CSHA_TIER_B1_PREPASS_X_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry csha_tier_b1_prepass_x(\n\
    .param .u64 x_in,\n\
    .param .u64 gamma,\n\
    .param .u64 x_out,\n\
    .param .u64 seq,\n\
    .param .u64 d_model,\n\
    .param .u64 chunk,\n\
    .param .u32 log2_chunk,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<16>;\n\
    .reg .f32 %f<12>;\n\
    .reg .b16 %h1;\n\
    .reg .pred %p<5>;\n\
    .shared .f32 sdata[256];\n\
\n\
    ld.param.u64 %rd1, [x_in];\n\
    ld.param.u64 %rd2, [gamma];\n\
    ld.param.u64 %rd3, [x_out];\n\
    ld.param.u64 %rd4, [seq];\n\
    ld.param.u64 %rd5, [d_model];\n\
    ld.param.u64 %rd6, [chunk];\n\
    ld.param.u32 %r10, [log2_chunk];\n\
    ld.param.f32 %f8, [eps];\n\
\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd4;\n\
    @%p1 bra X_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
\n\
    // row_base_in = x_in + (row * d_model) * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd5;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
\n\
    // --- Pass 1: partial sum-of-squares ---\n\
    mov.f32 %f1, 0f00000000;\n\
    mov.u64 %rd11, %rd8;\n\
X_SQ_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra X_SQ_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    add.u64 %rd13, %rd10, %rd12;\n\
    ld.global.f32 %f2, [%rd13];\n\
    fma.rn.f32 %f1, %f2, %f2, %f1;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra X_SQ_LOOP;\n\
X_SQ_DONE:\n\
    // Store partial in sdata[tid]\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r5, sdata;\n\
    add.u32 %r5, %r5, %r3;\n\
    st.shared.f32 [%r5], %f1;\n\
    bar.sync 0;\n\
\n\
    // Thread 0 reduces and writes rms_inv to sdata[0]\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra X_SKIP_REDUCE;\n\
    mov.u32 %r4, 1;\n\
    mov.u32 %r6, %ntid.x;\n\
X_RLOOP:\n\
    setp.ge.u32 %p2, %r4, %r6;\n\
    @%p2 bra X_RDONE;\n\
    mul.lo.u32 %r7, %r4, 4;\n\
    mov.u32 %r5, sdata;\n\
    add.u32 %r5, %r5, %r7;\n\
    ld.shared.f32 %f3, [%r5];\n\
    add.f32 %f1, %f1, %f3;\n\
    add.u32 %r4, %r4, 1;\n\
    bra X_RLOOP;\n\
X_RDONE:\n\
    cvt.rn.f32.u64 %f4, %rd5;\n\
    div.approx.f32 %f1, %f1, %f4;\n\
    add.f32 %f1, %f1, %f8;\n\
    rsqrt.approx.f32 %f1, %f1;\n\
    st.shared.f32 [sdata], %f1;\n\
X_SKIP_REDUCE:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f6, [sdata];\n\
\n\
    // --- Pass 2: normalize, scale by gamma, narrow, chunkify ---\n\
    // chunk_band_size_bytes = seq * chunk * 2\n\
    mul.lo.u64 %rd14, %rd4, %rd6;\n\
    shl.b64 %rd14, %rd14, 1;\n\
    // row_off_bytes = row * chunk * 2\n\
    mul.lo.u64 %rd15, %rd7, %rd6;\n\
    shl.b64 %rd15, %rd15, 1;\n\
    // chunk_minus_one (mask for c = d & (chunk-1))\n\
    sub.u64 %rd16, %rd6, 1;\n\
\n\
    mov.u64 %rd11, %rd8;  // d = tid\n\
X_NORM_LOOP:\n\
    setp.ge.u64 %p2, %rd11, %rd5;\n\
    @%p2 bra X_DONE;\n\
    shl.b64 %rd12, %rd11, 2;\n\
    // Load x[row, d]\n\
    add.u64 %rd17, %rd10, %rd12;\n\
    ld.global.f32 %f2, [%rd17];\n\
    // Load gamma[d]\n\
    add.u64 %rd18, %rd2, %rd12;\n\
    ld.global.f32 %f7, [%rd18];\n\
    // f2 = x * rms_inv * gamma\n\
    mul.f32 %f2, %f2, %f6;\n\
    mul.f32 %f2, %f2, %f7;\n\
    // Narrow to f16\n\
    cvt.rn.f16.f32 %h1, %f2;\n\
    // chunk_idx = d >> log2_chunk\n\
    shr.u64 %rd19, %rd11, %r10;\n\
    // c = d & (chunk - 1); c_bytes = c << 1\n\
    and.b64 %rd20, %rd11, %rd16;\n\
    shl.b64 %rd20, %rd20, 1;\n\
    // out_byte = chunk_idx * chunk_band_size + row_off + c_bytes\n\
    mul.lo.u64 %rd21, %rd19, %rd14;\n\
    add.u64 %rd21, %rd21, %rd15;\n\
    add.u64 %rd21, %rd21, %rd20;\n\
    add.u64 %rd21, %rd3, %rd21;\n\
    st.global.b16 [%rd21], %h1;\n\
    add.u64 %rd11, %rd11, 256;\n\
    bra X_NORM_LOOP;\n\
X_DONE:\n\
    ret;\n\
}\0";

// ---------------------------------------------------------------------------
// W pre-pass: narrow + col-major-chunkify
//
// Grid:    (ceil(d_model * hd / 256), 1, 1)
// Block:   (256, 1, 1)
// Inputs:
//   w_in       : f32 [d_model, hd]                (row-major)
//   w_out      : f16 [d_model/chunk, hd, chunk]   (col-major within each chunk band)
//   d_model, hd, chunk : u64
//   log2_hd, log2_chunk : u32 (both powers of 2)
//
// Output layout per chunk band: `[hd, chunk]` col-major →
// byte `(chunk_idx * hd * chunk + n * chunk + k_in_chunk) * 2`
// where (d_row, n) is the input position and
// d_row = chunk_idx * chunk + k_in_chunk.
//
// Indexed by input: gid → (d_row, n) where d_row = gid >> log2_hd,
// n = gid & (hd - 1). Reads coalesce (adjacent gid → adjacent n in the
// same input row). Writes are NOT coalesced (adjacent gid → adjacent n
// in output, which are `chunk * 2` bytes apart in col-major-within-chunk
// storage) — acceptable because this is a one-time conversion at model
// load, not per-step.
// ---------------------------------------------------------------------------
pub const CSHA_TIER_B1_PREPASS_W_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry csha_tier_b1_prepass_w(\n\
    .param .u64 w_in,\n\
    .param .u64 w_out,\n\
    .param .u64 d_model,\n\
    .param .u64 hd,\n\
    .param .u64 chunk,\n\
    .param .u32 log2_hd,\n\
    .param .u32 log2_chunk\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f1;\n\
    .reg .b16 %h1;\n\
    .reg .pred %p<3>;\n\
\n\
    ld.param.u64 %rd1, [w_in];\n\
    ld.param.u64 %rd2, [w_out];\n\
    ld.param.u64 %rd3, [d_model];\n\
    ld.param.u64 %rd4, [hd];\n\
    ld.param.u64 %rd5, [chunk];\n\
    ld.param.u32 %r1, [log2_hd];\n\
    ld.param.u32 %r2, [log2_chunk];\n\
\n\
    // gid = ctaid.x * 256 + tid.x\n\
    mov.u32 %r3, %ctaid.x;\n\
    mov.u32 %r4, %ntid.x;\n\
    mul.lo.u32 %r3, %r3, %r4;\n\
    mov.u32 %r5, %tid.x;\n\
    add.u32 %r3, %r3, %r5;\n\
    cvt.u64.u32 %rd6, %r3;\n\
    // total = d_model * hd\n\
    mul.lo.u64 %rd7, %rd3, %rd4;\n\
    setp.ge.u64 %p1, %rd6, %rd7;\n\
    @%p1 bra W_DONE;\n\
\n\
    // d_row = gid >> log2_hd; n = gid & (hd - 1)\n\
    shr.u64 %rd8, %rd6, %r1;\n\
    sub.u64 %rd9, %rd4, 1;\n\
    and.b64 %rd10, %rd6, %rd9;\n\
\n\
    // Load w_in[d_row, n] = w_in + gid * 4\n\
    shl.b64 %rd11, %rd6, 2;\n\
    add.u64 %rd11, %rd1, %rd11;\n\
    ld.global.f32 %f1, [%rd11];\n\
    cvt.rn.f16.f32 %h1, %f1;\n\
\n\
    // chunk_idx = d_row >> log2_chunk; k_in_chunk = d_row & (chunk - 1)\n\
    shr.u64 %rd12, %rd8, %r2;\n\
    sub.u64 %rd13, %rd5, 1;\n\
    and.b64 %rd14, %rd8, %rd13;\n\
\n\
    // out_byte = (chunk_idx * hd * chunk + n * chunk + k_in_chunk) * 2\n\
    // chunk_band_size = hd * chunk\n\
    mul.lo.u64 %rd15, %rd4, %rd5;\n\
    mul.lo.u64 %rd16, %rd12, %rd15;          // chunk_idx * hd * chunk\n\
    mul.lo.u64 %rd17, %rd10, %rd5;            // n * chunk\n\
    add.u64 %rd16, %rd16, %rd17;\n\
    add.u64 %rd16, %rd16, %rd14;              // + k_in_chunk\n\
    shl.b64 %rd16, %rd16, 1;                  // * 2 bytes\n\
    add.u64 %rd16, %rd2, %rd16;\n\
    st.global.b16 [%rd16], %h1;\n\
W_DONE:\n\
    ret;\n\
}\0";
