//! Fused GPU kernels for embedding lookup, bias_add, layernorm, and rmsnorm.
//! All PTX strings are null-terminated for the CUDA driver API.

// PTX constants are loaded at runtime by name via the CUDA driver API;
// Rust's dead-code analysis cannot see these usages.

// ---------------------------------------------------------------------------
// GPU Embedding Lookup
// Thread (i, j): copies weight[indices[i], j] -> out[i, j]
// Grid:  (ceil(seq_len/16), ceil(embed_dim/16), 1)
// Block: (16, 16, 1)
// Params: weight ptr, indices ptr, out ptr, seq_len (u64), embed_dim (u64)
// indices are stored as f32 (matching GPU dtype=1 convention)
// ---------------------------------------------------------------------------
/// `nsl_embedding_f32` (header above). The four 2-D-block row lookups (the
/// embedding and dim-0 gather kernels, with f32 or i32 indices) are built by
/// `nsl_kir::kernels::lookup`; its `lookup_kir_equivalence` gate holds them
/// to the hand-written modules they replaced. NUL-terminated.
pub(crate) fn embedding_f32_ptx() -> &'static str {
    lookup_module(nsl_kir::kernels::lookup::LookupOp::Embedding, nsl_kir::kernels::lookup::IndexDtype::F32)
}

/// `nsl_embedding_i32idx`: `nsl_embedding_f32` for i32 token ids, read
/// with `ld.global.s32` and sign-extended instead of truncated from f32.
pub(crate) fn embedding_i32idx_ptx() -> &'static str {
    lookup_module(nsl_kir::kernels::lookup::LookupOp::Embedding, nsl_kir::kernels::lookup::IndexDtype::I32)
}

/// The KIR-built row-lookup module for `(op, idx)`, built once.
fn lookup_module(op: nsl_kir::kernels::lookup::LookupOp, idx: nsl_kir::kernels::lookup::IndexDtype) -> &'static str {
    use nsl_kir::kernels::lookup::{lookup_ptx, IndexDtype, LookupOp};
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 4] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let slot = match (op, idx) {
        (LookupOp::Embedding, IndexDtype::F32) => 0,
        (LookupOp::Embedding, IndexDtype::I32) => 1,
        (LookupOp::Gather, IndexDtype::F32) => 2,
        (LookupOp::Gather, IndexDtype::I32) => 3,
    };
    MODULES[slot].get_or_init(|| String::from_utf8(lookup_ptx(op, idx)).expect("PTX must be ASCII"))
}

// ---------------------------------------------------------------------------
// GPU Embedding Backward (scatter-add)
// Thread (i, j): red.global.add out[indices[i], j] += grad[i, j]
// Grid:  (ceil(seq_len/16), ceil(embed_dim/16), 1)
// Block: (16, 16, 1)
// Params: grad ptr, indices ptr, out ptr (pre-zeroed [vocab, embed]),
//         seq_len (u64), embed_dim (u64), vocab (u64)
// indices stored as f32 (GPU dtype=1 convention); converted via
// cvt.rzi.s64.f32 with explicit [0, vocab) guards so negative or
// out-of-range ids are skipped exactly like the CPU reference's
// `tok < vocab_size` check (negative usize-wrap skips there).
// f32 atomics make the row sum order nondeterministic across runs (same
// policy as the flash phase-2 backward); NSL_EMBEDDING_BWD_CPU=1 in the
// launcher restores the deterministic host scatter.
// ---------------------------------------------------------------------------
pub(crate) fn embedding_bwd_f32_ptx() -> &'static str {
    embedding_bwd_module(
        nsl_kir::kernels::embedding_bwd::EmbeddingBwd::Atomic,
        nsl_kir::kernels::lookup::IndexDtype::F32,
    )
}

/// The KIR-built embedding backward module for `(op, idx)`, built once. The
/// four kernels are built by `nsl_kir::kernels::embedding_bwd`; its
/// `embedding_bwd_kir_equivalence` gate holds them to the hand-written
/// modules they replaced. NUL-terminated.
fn embedding_bwd_module(
    op: nsl_kir::kernels::embedding_bwd::EmbeddingBwd,
    idx: nsl_kir::kernels::lookup::IndexDtype,
) -> &'static str {
    use nsl_kir::kernels::embedding_bwd::{ptx, EmbeddingBwd};
    use nsl_kir::kernels::lookup::IndexDtype;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 4] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let slot = match (op, idx) {
        (EmbeddingBwd::Atomic, IndexDtype::F32) => 0,
        (EmbeddingBwd::Atomic, IndexDtype::I32) => 1,
        (EmbeddingBwd::Deterministic, IndexDtype::F32) => 2,
        (EmbeddingBwd::Deterministic, IndexDtype::I32) => 3,
    };
    MODULES[slot].get_or_init(|| String::from_utf8(ptx(op, idx)).expect("PTX must be ASCII"))
}

/// GPU embedding backward with i32 integer indices (mirrors
/// `nsl_embedding_i32idx`'s ld.global.s32 + cvt.s64.s32 index read).
pub(crate) fn embedding_bwd_i32idx_ptx() -> &'static str {
    embedding_bwd_module(
        nsl_kir::kernels::embedding_bwd::EmbeddingBwd::Atomic,
        nsl_kir::kernels::lookup::IndexDtype::I32,
    )
}

// ---------------------------------------------------------------------------
// GPU Embedding Backward (DETERMINISTIC, M46)
// Thread (v, j) OWNS out[v, j] and computes it by looping every sequence
// position i in fixed order, accumulating grad[i, j] where ids[i] == v. No
// atomics, single writer per element -> bit-identical to the CPU reference
// (same per-row summation order: increasing i). Selected when
// deterministic_ops::is_deterministic() is set (--deterministic / M46).
// Grid:  (ceil(vocab/16), ceil(embed_dim/16), 1)   [NOTE: vocab on x, unlike
//         the atomicAdd variant which puts seq_len on x]
// Block: (16, 16, 1)
// Params: grad, indices, out (pre-zeroed [vocab, embed]), seq_len, embed_dim, vocab
// Cost O(vocab * embed * seq) -> opt-in only; the atomicAdd variant remains
// the production default.
// ---------------------------------------------------------------------------
pub(crate) fn embedding_bwd_det_f32_ptx() -> &'static str {
    embedding_bwd_module(
        nsl_kir::kernels::embedding_bwd::EmbeddingBwd::Deterministic,
        nsl_kir::kernels::lookup::IndexDtype::F32,
    )
}

/// Deterministic embedding backward with i32 integer indices (mirrors
/// `nsl_embedding_bwd_det_f32`; token read via ld.global.s32 + cvt.s64.s32).
pub(crate) fn embedding_bwd_det_i32idx_ptx() -> &'static str {
    embedding_bwd_module(
        nsl_kir::kernels::embedding_bwd::EmbeddingBwd::Deterministic,
        nsl_kir::kernels::lookup::IndexDtype::I32,
    )
}

// ---------------------------------------------------------------------------
// GPU Bias Add
// Thread i handles element out[i] = in[i] + bias[i % cols]
// Grid:  (ceil(rows*cols / 256), 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, bias ptr, out ptr, total (rows*cols, u64), cols (u64)
//
// `.reg .f32 %f<4>` is deliberate: `%f<N>` declares %f0..%f(N-1) and the body
// uses %f3 for the sum. This read `%f<3>` until 2026-07, so ptxas rejected the
// module with `Unknown symbol '%f3'` and every GPU bias_add launch failed.
// ---------------------------------------------------------------------------
/// `nsl_bias_add_f32(inp, bias, out, total, cols)`: `out[i] = inp[i] + bias[i % cols]`. Built by `nsl_kir::kernels::data_movement` (its
/// `data_movement_kir_equivalence` gate holds it to the hand-written module
/// it replaced). NUL-terminated.
pub(crate) fn bias_add_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::data_movement::bias_add_ptx()).expect("PTX must be ASCII"))
}

// ---------------------------------------------------------------------------
// GPU Softmax and Log-Softmax (per-row, numerically stable)
// One thread block per row. Uses shared memory for max and sum reductions.
// Grid:  (num_rows, 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, out ptr, rows (u64), cols (u64)
// Each thread handles multiple columns via stride loop: find the row max,
// sum exp(x - max), then scale by 1/sum (softmax) or subtract log(sum) from
// x - max (log-softmax).
// ---------------------------------------------------------------------------
/// `nsl_softmax_f32(inp, out, rows, cols)`. Built by `nsl_kir::kernels::softmax`
/// (its `softmax_kir_equivalence` gate holds it to the hand-written module it
/// replaced). NUL-terminated, built once.
pub(crate) fn softmax_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::softmax::ptx(nsl_kir::kernels::softmax::SoftmaxOp::Softmax))
            .expect("PTX must be ASCII")
    })
}

/// `nsl_log_softmax_f32(inp, out, rows, cols)`: `(x - max) - log(sum(exp(x -
/// max)))`. Built by `nsl_kir::kernels::softmax`, gated as
/// [`softmax_f32_ptx`] is. NUL-terminated, built once.
pub(crate) fn log_softmax_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| {
        String::from_utf8(nsl_kir::kernels::softmax::ptx(nsl_kir::kernels::softmax::SoftmaxOp::LogSoftmax))
            .expect("PTX must be ASCII")
    })
}

// ---------------------------------------------------------------------------
// GPU Per-dimension Sum Reduction — SHORT-AXIS variant.
//
// The general kernel below assigns one 256-thread block per output element,
// which is the right shape when the reduced axis is long. When it is short
// that shape is pathological: a tied [49152, 512] embedding gradient sums two
// contributions per output, so it launched 25,165,824 blocks — 6.4 billion
// threads to perform 25 million additions. Every thread past `reduce_size`
// exits immediately and the survivors still run a full eight-level shared
// memory tree reduction over a single live value. Profiling a Coder-50M step
// attributed 82.5% of ALL GPU kernel time to `nsl_sum_dim_f32`, and 47.8% of
// that to just three launches of this shape.
//
// Here each output element gets one THREAD, which walks the reduced axis
// serially. Consecutive threads take consecutive `inner_idx`, so for the
// contiguous case (inner large) the loads stay coalesced.
//
// Grid:  (ceil(outer * inner / 256), 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, out ptr, outer (u64), reduce_size (u64), inner (u64)
// ---------------------------------------------------------------------------
//
// Built by `nsl_kir::kernels::det_sum` (`DetSumOp::DimShort`): the
// deterministic per-dim sum's loop with the thread's global index as the
// output. Its `det_sum_kir_equivalence` gate holds it to the hand-written
// module it replaced.
pub(crate) fn sum_dim_short_f32_ptx() -> &'static str {
    det_sum_module(nsl_kir::kernels::det_sum::DetSumOp::DimShort)
}

// ---------------------------------------------------------------------------
// GPU Per-dimension Sum Reduction
// One thread block per output element.
// Decomposes the reduction as: outer * reduce_size * inner
// Each block sums reduce_size elements spaced `inner` apart.
// Grid:  (outer * inner, 1, 1) — one block per output element
// Block: (256, 1, 1)
// Params: in ptr, out ptr, outer (u64), reduce_size (u64), inner (u64)
// ---------------------------------------------------------------------------
/// The KIR-built tree-reduction module for `op`, built once. All three
/// kernels are built by `nsl_kir::kernels::block_reduce`; its
/// `block_reduce_kir_equivalence` gate holds them to the hand-written
/// modules they replaced. NUL-terminated.
fn block_reduce_module(op: nsl_kir::kernels::block_reduce::BlockReduceOp) -> &'static str {
    use nsl_kir::kernels::block_reduce::BlockReduceOp;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let slot = BlockReduceOp::ALL.iter().position(|o| *o == op).expect("a tree reduction");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::block_reduce::ptx(op)).expect("PTX must be ASCII"))
}

/// `nsl_sum_dim_f32(inp, out, outer, reduce_size, inner)`.
pub(crate) fn sum_dim_f32_ptx() -> &'static str {
    block_reduce_module(nsl_kir::kernels::block_reduce::BlockReduceOp::SumDim)
}

// ---------------------------------------------------------------------------
// GPU Per-dimension Max Reduction
// Same structure as sum but uses max.f32 instead of add.f32.
// Grid:  (outer * inner, 1, 1) — one block per output element
// Block: (256, 1, 1)
// Params: in ptr, out ptr, outer (u64), reduce_size (u64), inner (u64)
// ---------------------------------------------------------------------------
/// `nsl_max_dim_f32(inp, out, outer, reduce_size, inner)`, from `-inf`.
pub(crate) fn max_dim_f32_ptx() -> &'static str {
    block_reduce_module(nsl_kir::kernels::block_reduce::BlockReduceOp::MaxDim)
}

// ---------------------------------------------------------------------------
// GPU Global Sum Reduction (all elements to a single scalar)
// One block, shared memory tree reduction.
// Grid:  (1, 1, 1)
// Block: (256, 1, 1)
// Params: in ptr, out ptr, n (u64)
// ---------------------------------------------------------------------------
/// `nsl_global_sum_f32(inp, out, n)`.
pub(crate) fn global_sum_f32_ptx() -> &'static str {
    block_reduce_module(nsl_kir::kernels::block_reduce::BlockReduceOp::GlobalSum)
}

// ---------------------------------------------------------------------------
// GPU LayerNorm and RMSNorm forwards (per-row). One thread block per row.
// Grid:  (num_rows, 1, 1) — one block per row
// Block: (256, 1, 1)
// LayerNorm params: in ptr, out ptr, gamma ptr, beta ptr, rows (u64), cols (u64), eps (f32)
//   mean = sum(x) / cols; var = sum((x - mean)^2) / cols;
//   out = gamma * (x - mean) * rsqrt(var + eps) + beta
// RMSNorm params: in ptr, out ptr, gamma ptr, rows (u64), cols (u64), eps (f32)
//   out = gamma * x * rsqrt(mean(x^2) + eps)
// ---------------------------------------------------------------------------
/// `nsl_layernorm_f32(inp, out, gamma, beta, rows, cols, eps)`. Built by
/// `nsl_kir::kernels::norm` (its `norm_kir_equivalence` gate holds it to the
/// hand-written module it replaced, whose shared-slot race it fixes).
/// NUL-terminated, built once.
pub(crate) fn layernorm_f32_ptx() -> &'static str {
    norm_module(nsl_kir::kernels::norm::NormOp::LayerNorm)
}

/// `nsl_rmsnorm_f32(inp, out, gamma, rows, cols, eps)`. Built by
/// `nsl_kir::kernels::norm`, gated as [`layernorm_f32_ptx`] is.
/// NUL-terminated, built once.
pub(crate) fn rmsnorm_f32_ptx() -> &'static str {
    norm_module(nsl_kir::kernels::norm::NormOp::RmsNorm)
}

/// The KIR-built module for one of the two norms, built once.
fn norm_module(op: nsl_kir::kernels::norm::NormOp) -> &'static str {
    use nsl_kir::kernels::norm::NormOp;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 2] = [OnceLock::new(), OnceLock::new()];
    let slot = NormOp::ALL.iter().position(|o| *o == op).expect("a norm kernel");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::norm::ptx(op)).expect("PTX must be ASCII"))
}

// Fused RMSNorm INPUT-gradient (dx) backward, one block per row (blockDim=256).
// Computes, per row, both reductions in one pass —
//   S1 = Σ_j x_j²           (→ rms_inv = rsqrt(S1/N + eps))
//   S2 = Σ_j ȳ_j·γ_j·x_j    (the shared per-row scalar)
// then dx_j = γ_j·ȳ_j·rms_inv − x_j·S2·rms_inv³/N. Single output, no atomics.
// Matches the correct RMSNorm dx (NO mean-subtract), i.e. tape-AD's
// `rmsnorm_backward` dx term and the source-AD `RmsNormInputBackward`
// decomposition — validated to tolerance (approx rsqrt/div vs the CPU f64 ref).
// P5 item 20 slice A — fused RMSNorm GAMMA backward (2 launches, no
// full-size temps). Kernel 1: one thread per ROW computes
// rinv[row] = 1 / sqrt(mean(x_row^2) + eps) into a tiny [rows] scratch.
// Kernel 2: one thread per COLUMN j accumulates
// dgamma[j] = sum_rows(dy[i,j] * x[i,j] * rinv[i]) with a sequential row
// loop — a fixed summation order, so the result is bit-deterministic
// run-to-run (the old 7-op decomposition materialized three [rows, cols]
// temporaries and reduced through reduce_to_shape).

/// `nsl_rmsnorm_rinv_rows_f32(x, rinv, rows, cols, eps)`: a thread per row,
/// `rinv[r] = 1 / sqrt(Σ_j x[r, j]² / cols + eps)`, every step correctly
/// rounded.
pub(crate) fn rmsnorm_rinv_rows_f32_ptx() -> &'static str {
    rmsnorm_dgamma_module(nsl_kir::kernels::rmsnorm_dgamma::RmsNormDgammaOp::RinvRows)
}

/// `nsl_rmsnorm_dgamma_f32(dy, x, rinv, dgamma, rows, cols)`: a thread per
/// column, `dgamma[j] = Σ_i (dy[i, j] · x[i, j]) · rinv[i]` in row order.
pub(crate) fn rmsnorm_dgamma_f32_ptx() -> &'static str {
    rmsnorm_dgamma_module(nsl_kir::kernels::rmsnorm_dgamma::RmsNormDgammaOp::Dgamma)
}

/// The KIR-built module for one of the two launches, built once. Both are
/// built by `nsl_kir::kernels::rmsnorm_dgamma`; its
/// `rmsnorm_dgamma_kir_equivalence` gate holds them to the hand-written
/// modules they replaced. NUL-terminated.
fn rmsnorm_dgamma_module(op: nsl_kir::kernels::rmsnorm_dgamma::RmsNormDgammaOp) -> &'static str {
    use nsl_kir::kernels::rmsnorm_dgamma::RmsNormDgammaOp;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 2] = [OnceLock::new(), OnceLock::new()];
    let slot = RmsNormDgammaOp::ALL.iter().position(|o| *o == op).expect("an RMSNorm dgamma kernel");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::rmsnorm_dgamma::ptx(op)).expect("PTX must be ASCII"))
}

// P5 slice C — fused RMSNorm dx WITH residual-gradient fold. Identical to
// RMSNORM_DX_BWD_F32_PTX plus one epilogue `add.rn` of a same-shape residual
// gradient before the store — replaces the adjoint-accumulate Add that
// followed the dx op (bit-exact: IEEE add is commutative, and the standalone
// Add kernel performs the same single rn-rounded add through global memory).
pub(crate) const RMSNORM_DX_BWD_ADD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_dx_bwd_add_f32(\n\
    .param .u64 dy, .param .u64 x, .param .u64 gamma,\n\
    .param .u64 dxout, .param .u64 res,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<24>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f<24>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 ssq[256];\n\
    .shared .f32 sdwx[256];\n\
    ld.param.u64 %rd1, [dy];\n\
    ld.param.u64 %rd2, [x];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [dxout];\n\
    ld.param.u64 %rd19, [res];\n\
    ld.param.u64 %rd5, [rows];\n\
    ld.param.u64 %rd6, [cols];\n\
    ld.param.f32 %f1, [eps];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DX_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
    // row_base_bytes = row * cols * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
    add.u64 %rd11, %rd2, %rd9;\n\
    add.u64 %rd12, %rd4, %rd9;\n\
    add.u64 %rd20, %rd19, %rd9;\n\
    // --- Pass 1: local S1=sum(x*x), S2=sum(dy*gamma*x) ---\n\
    mov.f32 %f2, 0f00000000;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd13, %rd8;\n\
DX_ACC:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_ACC_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f4, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f5, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f6, [%rd17];\n\
    fma.rn.f32 %f2, %f4, %f4, %f2;\n\
    mul.f32 %f7, %f5, %f6;\n\
    fma.rn.f32 %f3, %f7, %f4, %f3;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_ACC;\n\
DX_ACC_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r4, ssq;\n\
    add.u32 %r4, %r4, %r3;\n\
    st.shared.f32 [%r4], %f2;\n\
    mov.u32 %r5, sdwx;\n\
    add.u32 %r5, %r5, %r3;\n\
    st.shared.f32 [%r5], %f3;\n\
    bar.sync 0;\n\
    // Reduce both (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra DX_SKIP;\n\
    mov.u32 %r6, 1;\n\
    mov.u32 %r7, %ntid.x;\n\
DX_RLOOP:\n\
    setp.ge.u32 %p2, %r6, %r7;\n\
    @%p2 bra DX_RDONE;\n\
    mul.lo.u32 %r8, %r6, 4;\n\
    mov.u32 %r9, ssq;\n\
    add.u32 %r9, %r9, %r8;\n\
    ld.shared.f32 %f8, [%r9];\n\
    add.f32 %f2, %f2, %f8;\n\
    mov.u32 %r10, sdwx;\n\
    add.u32 %r10, %r10, %r8;\n\
    ld.shared.f32 %f9, [%r10];\n\
    add.f32 %f3, %f3, %f9;\n\
    add.u32 %r6, %r6, 1;\n\
    bra DX_RLOOP;\n\
DX_RDONE:\n\
    // rms_inv = rsqrt(S1/cols + eps)\n\
    cvt.rn.f32.u64 %f10, %rd6;\n\
    div.approx.f32 %f11, %f2, %f10;\n\
    add.f32 %f11, %f11, %f1;\n\
    rsqrt.approx.f32 %f11, %f11;\n\
    // Clamp rms_inv <= 1e12 (i.e. rms >= 1e-12), matching the CPU/tape-AD\n\
    // underflow guard so eps=0 + a near-zero row cannot inject +Inf/NaN.\n\
    min.f32 %f11, %f11, 0f5368D4A5;\n\
    st.shared.f32 [ssq], %f11;\n\
    st.shared.f32 [sdwx], %f3;\n\
DX_SKIP:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f12, [ssq];\n\
    ld.shared.f32 %f13, [sdwx];\n\
    // coeff = S2 * rms_inv^3 / cols\n\
    mul.f32 %f14, %f12, %f12;\n\
    mul.f32 %f14, %f14, %f12;\n\
    mul.f32 %f14, %f14, %f13;\n\
    cvt.rn.f32.u64 %f15, %rd6;\n\
    div.approx.f32 %f14, %f14, %f15;\n\
    // --- Pass 2: dx_j = gamma_j*dy_j*rms_inv - x_j*coeff ---\n\
    mov.u64 %rd13, %rd8;\n\
DX_WR:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f16, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f17, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f18, [%rd17];\n\
    mul.f32 %f19, %f18, %f17;\n\
    mul.f32 %f19, %f19, %f12;\n\
    mul.f32 %f20, %f16, %f14;\n\
    sub.f32 %f19, %f19, %f20;\n\
    add.u64 %rd21, %rd20, %rd14;\n\
    ld.global.f32 %f21, [%rd21];\n\
    add.rn.f32 %f19, %f19, %f21;\n\
    add.u64 %rd18, %rd12, %rd14;\n\
    st.global.f32 [%rd18], %f19;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_WR;\n\
DX_DONE: ret;\n\
}\0";

pub(crate) const RMSNORM_DX_BWD_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_rmsnorm_dx_bwd_f32(\n\
    .param .u64 dy, .param .u64 x, .param .u64 gamma,\n\
    .param .u64 dxout,\n\
    .param .u64 rows, .param .u64 cols,\n\
    .param .f32 eps\n\
) {\n\
    .reg .u64 %rd<24>;\n\
    .reg .u32 %r<12>;\n\
    .reg .f32 %f<24>;\n\
    .reg .pred %p<4>;\n\
    .shared .f32 ssq[256];\n\
    .shared .f32 sdwx[256];\n\
    ld.param.u64 %rd1, [dy];\n\
    ld.param.u64 %rd2, [x];\n\
    ld.param.u64 %rd3, [gamma];\n\
    ld.param.u64 %rd4, [dxout];\n\
    ld.param.u64 %rd5, [rows];\n\
    ld.param.u64 %rd6, [cols];\n\
    ld.param.f32 %f1, [eps];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd7, %r1;\n\
    setp.ge.u64 %p1, %rd7, %rd5;\n\
    @%p1 bra DX_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd8, %r2;\n\
    // row_base_bytes = row * cols * 4\n\
    mul.lo.u64 %rd9, %rd7, %rd6;\n\
    shl.b64 %rd9, %rd9, 2;\n\
    add.u64 %rd10, %rd1, %rd9;\n\
    add.u64 %rd11, %rd2, %rd9;\n\
    add.u64 %rd12, %rd4, %rd9;\n\
    // --- Pass 1: local S1=sum(x*x), S2=sum(dy*gamma*x) ---\n\
    mov.f32 %f2, 0f00000000;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd13, %rd8;\n\
DX_ACC:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_ACC_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f4, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f5, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f6, [%rd17];\n\
    fma.rn.f32 %f2, %f4, %f4, %f2;\n\
    mul.f32 %f7, %f5, %f6;\n\
    fma.rn.f32 %f3, %f7, %f4, %f3;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_ACC;\n\
DX_ACC_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r4, ssq;\n\
    add.u32 %r4, %r4, %r3;\n\
    st.shared.f32 [%r4], %f2;\n\
    mov.u32 %r5, sdwx;\n\
    add.u32 %r5, %r5, %r3;\n\
    st.shared.f32 [%r5], %f3;\n\
    bar.sync 0;\n\
    // Reduce both (thread 0)\n\
    setp.ne.u32 %p3, %r2, 0;\n\
    @%p3 bra DX_SKIP;\n\
    mov.u32 %r6, 1;\n\
    mov.u32 %r7, %ntid.x;\n\
DX_RLOOP:\n\
    setp.ge.u32 %p2, %r6, %r7;\n\
    @%p2 bra DX_RDONE;\n\
    mul.lo.u32 %r8, %r6, 4;\n\
    mov.u32 %r9, ssq;\n\
    add.u32 %r9, %r9, %r8;\n\
    ld.shared.f32 %f8, [%r9];\n\
    add.f32 %f2, %f2, %f8;\n\
    mov.u32 %r10, sdwx;\n\
    add.u32 %r10, %r10, %r8;\n\
    ld.shared.f32 %f9, [%r10];\n\
    add.f32 %f3, %f3, %f9;\n\
    add.u32 %r6, %r6, 1;\n\
    bra DX_RLOOP;\n\
DX_RDONE:\n\
    // rms_inv = rsqrt(S1/cols + eps)\n\
    cvt.rn.f32.u64 %f10, %rd6;\n\
    div.approx.f32 %f11, %f2, %f10;\n\
    add.f32 %f11, %f11, %f1;\n\
    rsqrt.approx.f32 %f11, %f11;\n\
    // Clamp rms_inv <= 1e12 (i.e. rms >= 1e-12), matching the CPU/tape-AD\n\
    // underflow guard so eps=0 + a near-zero row cannot inject +Inf/NaN.\n\
    min.f32 %f11, %f11, 0f5368D4A5;\n\
    st.shared.f32 [ssq], %f11;\n\
    st.shared.f32 [sdwx], %f3;\n\
DX_SKIP:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f12, [ssq];\n\
    ld.shared.f32 %f13, [sdwx];\n\
    // coeff = S2 * rms_inv^3 / cols\n\
    mul.f32 %f14, %f12, %f12;\n\
    mul.f32 %f14, %f14, %f12;\n\
    mul.f32 %f14, %f14, %f13;\n\
    cvt.rn.f32.u64 %f15, %rd6;\n\
    div.approx.f32 %f14, %f14, %f15;\n\
    // --- Pass 2: dx_j = gamma_j*dy_j*rms_inv - x_j*coeff ---\n\
    mov.u64 %rd13, %rd8;\n\
DX_WR:\n\
    setp.ge.u64 %p2, %rd13, %rd6;\n\
    @%p2 bra DX_DONE;\n\
    shl.b64 %rd14, %rd13, 2;\n\
    add.u64 %rd15, %rd11, %rd14;\n\
    ld.global.f32 %f16, [%rd15];\n\
    add.u64 %rd16, %rd10, %rd14;\n\
    ld.global.f32 %f17, [%rd16];\n\
    add.u64 %rd17, %rd3, %rd14;\n\
    ld.global.f32 %f18, [%rd17];\n\
    mul.f32 %f19, %f18, %f17;\n\
    mul.f32 %f19, %f19, %f12;\n\
    mul.f32 %f20, %f16, %f14;\n\
    sub.f32 %f19, %f19, %f20;\n\
    add.u64 %rd18, %rd12, %rd14;\n\
    st.global.f32 [%rd18], %f19;\n\
    add.u64 %rd13, %rd13, 256;\n\
    bra DX_WR;\n\
DX_DONE: ret;\n\
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
// the PTX for `gpu_scatter_add_f32`, which has no caller either
#[allow(dead_code)]
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
// ---------------------------------------------------------------------------
// GPU gather along an ARBITRARY dimension (NSL semantics).
//
// `nsl_tensor_gather` only had a device kernel for dim 0 and fell back to a full
// CPU round trip otherwise — copy the whole tensor to the host, gather, copy the
// result back. `cross_entropy` gathers along dim 1, so every training step moved
// the entire [B*S, vocab] log-probability tensor across PCIe twice: 192 MB each
// way at B*S=1024, vocab=49152. Host profiling showed memcpy at 94% of
// attributed host time, ~1 GB per step.
//
// NSL's gather REMOVES the gathered dimension: for input shape S and dim d,
// `indices` has length outer = prod(S[..d]) and the output is S with d dropped.
//   out[o*inner + k] = input[o*gather_dim_size*inner + idx[o]*inner + k]
//
// Indices are read as f32, matching `nsl_gather_f32` above. The caller validates
// them on the host first (they are only `outer` elements, so the check costs a
// few KB), which keeps the abort-on-out-of-bounds contract the CPU path has.
//
// Grid:  (ceil(outer * inner / 256), 1, 1)
// Block: (256, 1, 1)
// ---------------------------------------------------------------------------
/// `nsl_gather_dim_f32`: the dimension-removing gather (header above). Built by `nsl_kir::kernels::data_movement` (its
/// `data_movement_kir_equivalence` gate holds it to the hand-written module
/// it replaced). NUL-terminated.
pub(crate) fn gather_dim_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::data_movement::gather_dim_ptx()).expect("PTX must be ASCII"))
}

/// `nsl_gather_f32`: see [`embedding_f32_ptx`].
pub(crate) fn gather_f32_ptx() -> &'static str {
    lookup_module(nsl_kir::kernels::lookup::LookupOp::Gather, nsl_kir::kernels::lookup::IndexDtype::F32)
}

/// `nsl_gather_i32idx`: `nsl_gather_f32` with i32 indices, read with
/// `ld.global.s32` and sign-extended, so a negative one fails the unsigned
/// row test.
pub(crate) fn gather_i32idx_ptx() -> &'static str {
    lookup_module(nsl_kir::kernels::lookup::LookupOp::Gather, nsl_kir::kernels::lookup::IndexDtype::I32)
}

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
    // ih was computed above but we used %rd16 -- recompute\n\
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
// Flash-attention log-sum-exp (device-resident)
// ---------------------------------------------------------------------------

/// Two-pass numerically-stable log-sum-exp over attention rows, computed
/// device-resident from Q and K (MHA layout [batch, heads, seq, head_dim];
/// the GPU flash backward only dispatches when kv_heads == heads, so K is
/// indexed with the same head count as Q). One thread per (batch, head,
/// query-row); each thread walks the K rows twice -- once for the row max,
/// once for sum(exp(score - max)) -- and writes lse = max + ln(sum). This is
/// the exact algorithm of `compute_logsumexp_gqa`, moved onto the GPU so the
/// flash backward stops round-tripping Q and K to the host and recomputing the
/// full attention scores on the CPU every call (an O(b*h*s^2*d) per-call
/// regression on the default decorator-free backward path). Scores use
/// mul+add (not fma) to mirror the CPU reference; exp/ln go through
/// ex2.approx/lg2.approx, so the result matches the CPU lse within GPU
/// transcendental tolerance -- consistent with how the backward kernel itself
/// recomputes P = exp(score - lse).
///
/// Params: q, k, lse (f32 device ptrs), total = b*h*s, seq, head_dim,
/// scale (f32), causal (0/1). Grid: ceil(total/256). Block: 256.
pub(crate) const FLASH_LSE_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_flash_lse_f32(\n\
    .param .u64 q, .param .u64 k, .param .u64 lse,\n\
    .param .u64 total, .param .u64 seq, .param .u64 hd,\n\
    .param .f32 scale, .param .u64 causal\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<16>;\n\
    .reg .pred %p<8>;\n\
\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd1, %r1;\n\
\n\
    ld.param.u64 %rd2, [q];\n\
    ld.param.u64 %rd3, [k];\n\
    ld.param.u64 %rd4, [lse];\n\
    ld.param.u64 %rd5, [total];\n\
    ld.param.u64 %rd6, [seq];\n\
    ld.param.u64 %rd7, [hd];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u64 %rd8, [causal];\n\
\n\
    setp.ge.u64 %p1, %rd1, %rd5;\n\
    @%p1 bra LSE_DONE;\n\
\n\
    rem.u64 %rd9, %rd1, %rd6;\n\
    div.u64 %rd10, %rd1, %rd6;\n\
\n\
    mul.lo.u64 %rd11, %rd1, %rd7;\n\
    mul.lo.u64 %rd12, %rd10, %rd6;\n\
    mul.lo.u64 %rd12, %rd12, %rd7;\n\
\n\
    add.u64 %rd13, %rd9, 1;\n\
    setp.ne.u64 %p2, %rd8, 0;\n\
    selp.u64 %rd14, %rd13, %rd6, %p2;\n\
\n\
    shl.b64 %rd15, %rd11, 2;\n\
    add.u64 %rd15, %rd2, %rd15;\n\
    shl.b64 %rd16, %rd12, 2;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
\n\
    // Pass 1: row max\n\
    mov.f32 %f2, 0fFF800000;\n\
    mov.u64 %rd17, 0;\n\
LSE_MAX_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSE_MAX_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSE_MAX_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSE_MAX_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSE_MAX_D;\n\
LSE_MAX_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    setp.gt.f32 %p5, %f3, %f2;\n\
    @%p5 mov.f32 %f2, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSE_MAX_J;\n\
LSE_MAX_DONE:\n\
\n\
    // Pass 2: sum exp(score - max)\n\
    mov.f32 %f7, 0f00000000;\n\
    mov.u64 %rd17, 0;\n\
LSE_SUM_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSE_SUM_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSE_SUM_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSE_SUM_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSE_SUM_D;\n\
LSE_SUM_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    sub.f32 %f3, %f3, %f2;\n\
    mul.f32 %f3, %f3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f3, %f3;\n\
    add.f32 %f7, %f7, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSE_SUM_J;\n\
LSE_SUM_DONE:\n\
\n\
    lg2.approx.f32 %f8, %f7;\n\
    mul.f32 %f8, %f8, 0f3F317218;\n\
    add.f32 %f8, %f2, %f8;\n\
\n\
    shl.b64 %rd22, %rd1, 2;\n\
    add.u64 %rd22, %rd4, %rd22;\n\
    st.global.f32 [%rd22], %f8;\n\
LSE_DONE:\n\
    ret;\n\
}\n\
\0";

/// Native-GQA variant of `nsl_flash_lse_f32` (P4, pretraining memory
/// reduction): identical two-pass algorithm, but K is `[batch, kv_heads,
/// seq, head_dim]` and each Q row reads its GROUP's kv-head —
/// `kbh = (bh / heads) * kv_heads + (bh % heads) / (heads / kv_heads)`, the
/// same consecutive-block mapping as `flash_attention_backward_cpu_gqa` and
/// the grouped Phase-2 kernel. This is what lets the native GQA backward
/// recompute the logsumexp WITHOUT materializing an expanded K (the expand
/// envelope's whole point was to avoid exactly this indexing).
///
/// Params: q, k, lse, total = b*h*s, seq, head_dim, scale (f32),
/// causal (0/1), heads, kv_heads. Grid: ceil(total/256). Block: 256.
/// Caller guarantees heads % kv_heads == 0 (the GQA dispatch gate).
pub(crate) const FLASH_LSE_GQA_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_flash_lse_gqa_f32(\n\
    .param .u64 q, .param .u64 k, .param .u64 lse,\n\
    .param .u64 total, .param .u64 seq, .param .u64 hd,\n\
    .param .f32 scale, .param .u64 causal,\n\
    .param .u64 heads, .param .u64 kv_heads\n\
) {\n\
    .reg .u64 %rd<32>;\n\
    .reg .u32 %r<8>;\n\
    .reg .f32 %f<16>;\n\
    .reg .pred %p<8>;\n\
\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mul.lo.u32 %r1, %r1, %r2;\n\
    mov.u32 %r2, %tid.x;\n\
    add.u32 %r1, %r1, %r2;\n\
    cvt.u64.u32 %rd1, %r1;\n\
\n\
    ld.param.u64 %rd2, [q];\n\
    ld.param.u64 %rd3, [k];\n\
    ld.param.u64 %rd4, [lse];\n\
    ld.param.u64 %rd5, [total];\n\
    ld.param.u64 %rd6, [seq];\n\
    ld.param.u64 %rd7, [hd];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u64 %rd8, [causal];\n\
    ld.param.u64 %rd23, [heads];\n\
    ld.param.u64 %rd24, [kv_heads];\n\
\n\
    setp.ge.u64 %p1, %rd1, %rd5;\n\
    @%p1 bra LSEG_DONE;\n\
\n\
    rem.u64 %rd9, %rd1, %rd6;\n\
    div.u64 %rd10, %rd1, %rd6;\n\
\n\
    mul.lo.u64 %rd11, %rd1, %rd7;\n\
    // kbh = (bh / heads) * kv_heads + (bh % heads) / groups\n\
    div.u64 %rd25, %rd23, %rd24;\n\
    div.u64 %rd26, %rd10, %rd23;\n\
    rem.u64 %rd27, %rd10, %rd23;\n\
    div.u64 %rd27, %rd27, %rd25;\n\
    mul.lo.u64 %rd12, %rd26, %rd24;\n\
    add.u64 %rd12, %rd12, %rd27;\n\
    mul.lo.u64 %rd12, %rd12, %rd6;\n\
    mul.lo.u64 %rd12, %rd12, %rd7;\n\
\n\
    add.u64 %rd13, %rd9, 1;\n\
    setp.ne.u64 %p2, %rd8, 0;\n\
    selp.u64 %rd14, %rd13, %rd6, %p2;\n\
\n\
    shl.b64 %rd15, %rd11, 2;\n\
    add.u64 %rd15, %rd2, %rd15;\n\
    shl.b64 %rd16, %rd12, 2;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
\n\
    // Pass 1: row max\n\
    mov.f32 %f2, 0fFF800000;\n\
    mov.u64 %rd17, 0;\n\
LSEG_MAX_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSEG_MAX_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSEG_MAX_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSEG_MAX_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSEG_MAX_D;\n\
LSEG_MAX_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    setp.gt.f32 %p5, %f3, %f2;\n\
    @%p5 mov.f32 %f2, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSEG_MAX_J;\n\
LSEG_MAX_DONE:\n\
\n\
    // Pass 2: sum exp(score - max)\n\
    mov.f32 %f7, 0f00000000;\n\
    mov.u64 %rd17, 0;\n\
LSEG_SUM_J:\n\
    setp.ge.u64 %p3, %rd17, %rd14;\n\
    @%p3 bra LSEG_SUM_DONE;\n\
    mov.f32 %f3, 0f00000000;\n\
    mov.u64 %rd18, 0;\n\
    mul.lo.u64 %rd19, %rd17, %rd7;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd16, %rd19;\n\
    mov.u64 %rd20, %rd15;\n\
    mov.u64 %rd21, %rd19;\n\
LSEG_SUM_D:\n\
    setp.ge.u64 %p4, %rd18, %rd7;\n\
    @%p4 bra LSEG_SUM_D_DONE;\n\
    ld.global.f32 %f4, [%rd20];\n\
    ld.global.f32 %f5, [%rd21];\n\
    mul.f32 %f6, %f4, %f5;\n\
    add.f32 %f3, %f3, %f6;\n\
    add.u64 %rd20, %rd20, 4;\n\
    add.u64 %rd21, %rd21, 4;\n\
    add.u64 %rd18, %rd18, 1;\n\
    bra LSEG_SUM_D;\n\
LSEG_SUM_D_DONE:\n\
    mul.f32 %f3, %f3, %f1;\n\
    sub.f32 %f3, %f3, %f2;\n\
    mul.f32 %f3, %f3, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f3, %f3;\n\
    add.f32 %f7, %f7, %f3;\n\
    add.u64 %rd17, %rd17, 1;\n\
    bra LSEG_SUM_J;\n\
LSEG_SUM_DONE:\n\
\n\
    lg2.approx.f32 %f8, %f7;\n\
    mul.f32 %f8, %f8, 0f3F317218;\n\
    add.f32 %f8, %f2, %f8;\n\
\n\
    shl.b64 %rd22, %rd1, 2;\n\
    add.u64 %rd22, %rd4, %rd22;\n\
    st.global.f32 [%rd22], %f8;\n\
LSEG_DONE:\n\
    ret;\n\
}\n\
\0";

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
/// `nsl_dropout_f32` (header above). Built by `nsl_kir::kernels::dropout`
/// (its `dropout_kir_equivalence` gate holds it to the hand-written module
/// it replaced). NUL-terminated.
pub(crate) fn dropout_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::dropout::dropout_ptx()).expect("PTX must be ASCII"))
}

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
/// GPU slice kernel: copies a contiguous sub-range along one dimension.
/// Like strided_copy but adds slice_start to the coordinate for the slice dimension.
/// Params: src, dst, shape(out), src_strides, dst_strides, ndim, total(out), slice_dim, slice_start
/// `nsl_slice_f32`: the strided walk with `slice_start` added on `slice_dim`. Built by `nsl_kir::kernels::data_movement` (its
/// `data_movement_kir_equivalence` gate holds it to the hand-written module
/// it replaced). NUL-terminated.
pub(crate) fn slice_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::data_movement::strided_ptx(nsl_kir::kernels::data_movement::StridedOp::Slice)).expect("PTX must be ASCII"))
}

/// `nsl_strided_copy_f32`: materialise a strided view (header above). Built by `nsl_kir::kernels::data_movement` (its
/// `data_movement_kir_equivalence` gate holds it to the hand-written module
/// it replaced). NUL-terminated.
pub(crate) fn strided_copy_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::data_movement::strided_ptx(nsl_kir::kernels::data_movement::StridedOp::Copy)).expect("PTX must be ASCII"))
}

// ---------------------------------------------------------------------------
// CSR Sparse Matrix-Dense Matrix Multiply (SpMM)
// C[M,N] = A_sparse[M,K] @ B_dense[K,N]
// Row-parallel: one thread block per output row, threads parallelize across N.
// Each thread accumulates the dot product for one output column.
// ---------------------------------------------------------------------------

/// `nsl_csr_spmm_f32(row_ptrs, col_indices, values, B, C, M, N)`: sparse A
/// (CSR, `u32` indices) @ dense B → dense C. Built by `nsl_kir::kernels::spmm`
/// (its `spmm_kir_equivalence` gate holds it to the hand-written module it
/// replaced). NUL-terminated, built once.
pub(crate) fn csr_spmm_f32_ptx() -> &'static str {
    spmm_module(nsl_kir::kernels::spmm::SpmmFormat::Csr)
}

/// The KIR-built module for one of the three SpMM kernels, built once.
fn spmm_module(format: nsl_kir::kernels::spmm::SpmmFormat) -> &'static str {
    use nsl_kir::kernels::spmm::SpmmFormat;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let slot = SpmmFormat::ALL.iter().position(|f| *f == format).expect("an SpMM kernel");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::spmm::ptx(format)).expect("PTX must be ASCII"))
}

// ---------------------------------------------------------------------------
// M50: COO SpMM — C[M,N] = A_coo[M,K] @ B[K,N]
// One thread per nonzero, looping over the N output columns and atomically
// accumulating into C. Simple but effective for unstructured sparsity.
// ---------------------------------------------------------------------------

/// `nsl_coo_spmm_f32(row_indices, col_indices, values, B, C, N, nnz)`: a
/// thread per nonzero (`i64` indices) atomically adds its row of products
/// into the zeroed C. Built by `nsl_kir::kernels::spmm`, gated as
/// [`csr_spmm_f32_ptx`] is. NUL-terminated, built once.
pub(crate) fn coo_spmm_f32_ptx() -> &'static str {
    spmm_module(nsl_kir::kernels::spmm::SpmmFormat::Coo)
}

// ---------------------------------------------------------------------------
// M50: BSR SpMM — C[M,N] = A_bsr[M,K] @ B[K,N]
// Block-parallel: each thread block handles one block row.
// BSR stores dense sub-blocks (block_rows x block_cols).
// ---------------------------------------------------------------------------

/// `nsl_bsr_spmm_f32(row_ptrs, col_indices, values, B, C, N, block_rows,
/// block_cols, nblk_rows)`: row_ptrs[nblk_rows+1], col_indices[nblocks],
/// values[nblocks*br*bc], B[K,N], C[M,N]. Built by `nsl_kir::kernels::spmm`,
/// gated as [`csr_spmm_f32_ptx`] is. NUL-terminated, built once.
pub(crate) fn bsr_spmm_f32_ptx() -> &'static str {
    spmm_module(nsl_kir::kernels::spmm::SpmmFormat::Bsr)
}

// ---------------------------------------------------------------------------
// M50: sparse matrix-vector products, y[M] = A[M,K] @ x[K].
// ---------------------------------------------------------------------------

/// The KIR-built SpMV module for `format`, built once. Both kernels are
/// built by `nsl_kir::kernels::spmv`; its `spmv_kir_equivalence` gate holds
/// them to the hand-written modules they replaced. NUL-terminated.
fn spmv_module(format: nsl_kir::kernels::spmv::SpmvFormat) -> &'static str {
    use nsl_kir::kernels::spmv::SpmvFormat;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 2] = [OnceLock::new(), OnceLock::new()];
    let slot = SpmvFormat::ALL.iter().position(|f| *f == format).expect("a sparse format");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::spmv::ptx(format)).expect("PTX must be ASCII"))
}

/// `nsl_csr_spmv_f32(row_ptrs, col_indices, values, x, y, M)`: one thread
/// per row, `u32` row pointers and column indices, `fma`-accumulated from
/// `+0.0` in nonzero order.
pub(crate) fn csr_spmv_f32_ptx() -> &'static str {
    spmv_module(nsl_kir::kernels::spmv::SpmvFormat::Csr)
}

/// `nsl_coo_spmv_f32(row_indices, col_indices, values, x, y, nnz)`: one
/// thread per nonzero, `i64` indices, `y[row] += value · x[col]` atomically
/// into a zeroed `y`.
pub(crate) fn coo_spmv_f32_ptx() -> &'static str {
    spmv_module(nsl_kir::kernels::spmv::SpmvFormat::Coo)
}

// ---------------------------------------------------------------------------
// M46b: the deterministic sums — one thread per result, adding in ascending
// index order, so the result does not depend on scheduling.
// ---------------------------------------------------------------------------

/// The KIR-built deterministic-sum module for `op`, built once. Both kernels
/// are built by `nsl_kir::kernels::det_sum`; its `det_sum_kir_equivalence`
/// gate holds them to the hand-written modules they replaced.
/// NUL-terminated.
fn det_sum_module(op: nsl_kir::kernels::det_sum::DetSumOp) -> &'static str {
    use nsl_kir::kernels::det_sum::DetSumOp;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 3] = [OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let slot = DetSumOp::ALL.iter().position(|o| *o == op).expect("a deterministic sum");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::det_sum::ptx(op)).expect("PTX must be ASCII"))
}

/// `nsl_det_global_sum_f32(inp, out, len)`: `out[0] = inp[0] + … +
/// inp[len - 1]`, in that order, by one thread. Grid (1, 1, 1), block
/// (1, 1, 1).
pub(crate) fn det_global_sum_f32_ptx() -> &'static str {
    det_sum_module(nsl_kir::kernels::det_sum::DetSumOp::Global)
}

/// `nsl_det_sum_dim_f32(inp, out, outer, reduce_size, inner)`: the sum over
/// the middle axis of an `[outer, reduce_size, inner]` view, one one-thread
/// block per output (`%ctaid.x`), adding in ascending order. Grid
/// (outer · inner, 1, 1), block (1, 1, 1).
pub(crate) fn det_sum_dim_f32_ptx() -> &'static str {
    det_sum_module(nsl_kir::kernels::det_sum::DetSumOp::Dim)
}

// ---------------------------------------------------------------------------
// M46c: Deterministic Scatter-Add (output-centric, no atomics)
// Thread (row, col): out[row, col] = input[row, col] + sum(src[i, col] for all i where indices[i] == row)
// Each thread owns exactly one output element and sequentially scans all input indices.
// This guarantees bit-identical results regardless of GPU scheduling.
//
// Grid:  (ceil(vocab_size / 16), ceil(embed_dim / 16), 1)
// Block: (16, 16, 1)
// Params: src ptr (grad), indices ptr, input ptr (base), out ptr,
//         num_indices (u64), embed_dim (u64), vocab_size (u64)
// ---------------------------------------------------------------------------
//
// Built by `nsl_kir::kernels::det_scatter`; its `det_scatter_kir_equivalence`
// gate holds it to the hand-written module it replaced. NUL-terminated,
// built once.
pub(crate) fn det_scatter_add_f32_ptx() -> &'static str {
    use std::sync::OnceLock;
    static MODULE: OnceLock<String> = OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::det_scatter::ptx()).expect("PTX must be ASCII"))
}

// ---------------------------------------------------------------------------
// M45b: Tensor statistics kernel — single-block reduction for min, max, sum,
// sum_of_squares. Output: out[0]=min, out[1]=max, out[2]=sum, out[3]=sum_sq.
// Grid: (1, 1, 1), Block: (256, 1, 1), SharedMem: 256 * 4 * 4 = 4096 bytes.
// Each thread strides across the input, maintaining local accumulators, then
// a shared-memory tree reduction combines per-thread results.
// ---------------------------------------------------------------------------
//
// Built by `nsl_kir::kernels::tensor_stats`; its `tensor_stats_kir_equivalence`
// gate holds it to the hand-written module it replaced. The sums round
// explicitly, so `Σx²` squares and adds with two roundings on hardware, as
// `nsl_muon_batch_sumsq_f32` does (the hand module's `mul.f32` + `add.f32`
// were contracted into an `fma` by ptxas). NUL-terminated, built once.
pub(crate) fn tensor_stats_f32_ptx() -> &'static str {
    use std::sync::OnceLock;
    static MODULE: OnceLock<String> = OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::tensor_stats::ptx()).expect("PTX must be ASCII"))
}

// ---------------------------------------------------------------------------
// Sum of squares over an f32 buffer, accumulated in DOUBLE, one partial per
// block.
//
// Output: out[blockIdx.x] = that block's f64 partial. The host sums the
// partials in f64, in block order, so the result is reproducible run to run.
// Block: (256, 1, 1), SharedMem: 256 * 8 = 2048 bytes. Grid is chosen by the
// caller (see `gpu_sum_sq_many_f32`) and MUST match the slot count.
//
// Why not slot 3 of `nsl_tensor_stats_f32`: that one accumulates in f32.
// Gradient clipping feeds the result into a `norm <= max_norm` comparison, so a
// systematically low sum does not merely round the answer, it can decline a clip
// the host f64 path would have made.
//
// Why grid-strided rather than a single block: f64 runs at 1/64 rate on GeForce,
// and a single 256-thread block accumulating a 25M-element gradient means about
// 98,000 serial f64 FMAs per thread on ONE SM. That measured 80 -> 112 ms per
// micro-batch, a 39% step-level regression -- an unacceptable price for the
// precision. Spreading the same work over 256 blocks cuts it to ~384 FMAs per
// thread. Note this is NOT the atomic variant that was tried and reverted
// earlier: partials go to per-block slots, so there is no contention and no
// ordering nondeterminism.
//
// `fma.rn.f64` squares and accumulates with a single rounding.
// ---------------------------------------------------------------------------
//
// Built by `nsl_kir::kernels::sum_sq`; its `sum_sq_kir_equivalence` gate
// holds it to the hand-written module it replaced. NUL-terminated, built once.
pub(crate) fn sum_sq_f64_acc_f32_ptx() -> &'static str {
    use std::sync::OnceLock;
    static MODULE: OnceLock<String> = OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::sum_sq::ptx()).expect("PTX must be ASCII"))
}

// ---------------------------------------------------------------------------
// M42b: KV-cache dequantization kernels (GPU)
// ---------------------------------------------------------------------------

// INT8 dequantization: output[i] = input_i8[i] * scales[head_index]
// Layout: [num_heads, block_size, head_dim] (head-major)
// Params: inp (i8*), out (f32*), scales (f32*), n (total elements),
//         head_stride (block_size * head_dim)
/// `nsl_dequant_int8_per_head_f32` (header above). Built by `nsl_kir::kernels::dequant` (its
/// `dequant_kir_equivalence` gate holds it to the hand-written module it
/// replaced). NUL-terminated.
pub(crate) fn dequant_int8_per_head_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::dequant::int8_ptx(nsl_kir::kernels::dequant::Int8Scale::PerHead)).expect("PTX must be ASCII"))
}

// INT8 per-token dequantization: output[i] = input_i8[i] * scales[token_index]
// Layout: [num_heads, block_size, head_dim] — token index = (i % head_stride) / head_dim
// Params: inp, out, scales, n, head_stride, head_dim
/// `nsl_dequant_int8_per_token_f32` (header above). Built by `nsl_kir::kernels::dequant` (its
/// `dequant_kir_equivalence` gate holds it to the hand-written module it
/// replaced). NUL-terminated.
pub(crate) fn dequant_int8_per_token_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::dequant::int8_ptx(nsl_kir::kernels::dequant::Int8Scale::PerToken)).expect("PTX must be ASCII"))
}

// INT4 per-group dequantization: unpack nibble, apply scale + zero_point
// output[i] = nibble(i) * scales[group] + zero_points[group]
// Params: inp (packed u8*), out (f32*), scales (f32*), zero_points (f32*),
//         n (total elements), group_size
/// `nsl_dequant_int4_per_group_f32` (header above). Built by `nsl_kir::kernels::dequant` (its
/// `dequant_kir_equivalence` gate holds it to the hand-written module it
/// replaced). NUL-terminated.
pub(crate) fn dequant_int4_per_group_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::dequant::int4_per_group_ptx()).expect("PTX must be ASCII"))
}

// FP8 E4M3 dequantization: bit manipulation to convert u8 → f32
// E4M3: 1 sign + 4 exponent + 3 mantissa, bias=7
// Params: inp (u8*), out (f32*), n
// E4M3 per the OCP FP8 spec: bias 7, no infinities, S.1111.111 the only NaN,
// and exponent field 0 the subnormals m/8 * 2^-6 (= m * 2^-9). The kernel used
// to send subnormal codes through the normal-number path, decoding code m as
// (1 + m/8) * 2^-7, and turn the NaN code into 480. The CPU decoder in
// `kv_compress::quantize` is the reference; `tests/fp8_e4m3_dequant_interp.rs`
// runs this module on the CTA interpreter over all 256 codes against it.
/// `nsl_dequant_fp8_e4m3_f32` (header above). Built by
/// `nsl_kir::kernels::dequant::build_fp8_e4m3` (its
/// `fp8_e4m3_kir_equivalence` gate holds it to the hand-written module it
/// replaced). NUL-terminated.
pub fn dequant_fp8_e4m3_f32_ptx() -> &'static str {
    static MODULE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::dequant::fp8_e4m3_ptx()).expect("PTX must be ASCII"))
}

// ── Muon batched Newton-Schulz kernels (perf-campaign items 1/3/4) ─────────
//
// Shape-grouped batched NS: k same-shape rank-2 matrices are processed by a
// fixed launch sequence over persistent workspaces. Matrices are addressed
// through DEVICE POINTER TABLES (a u64 array of data pointers uploaded once
// per group per step) so k tensors at arbitrary addresses batch without
// repacking. grid.y = matrix index everywhere except the sumsq reduction
// (one block per matrix, deterministic 256-lane tree — the same stride-256 +
// tree-128 order as TENSOR_STATS, so per-matrix sums are bit-identical to
// the sequential frobenius path on identical data).

/// m[i] = mu * m[i] + g[i], elementwise, in place over the pointer table.
/// Two-rounding mul+add matches the stdlib muon_step momentum update.
pub(crate) fn muon_batch_mom_f32_ptx() -> &'static str {
    muon_batch_module(nsl_kir::kernels::muon_batch::MuonBatchOp::Mom)
}

/// The KIR-built batched-Muon module for `op`, built once. The five
/// kernels are built by `nsl_kir::kernels::muon_batch`; its
/// `muon_batch_kir_equivalence` gate holds them to the hand-written modules
/// they replaced. NUL-terminated.
fn muon_batch_module(op: nsl_kir::kernels::muon_batch::MuonBatchOp) -> &'static str {
    use nsl_kir::kernels::muon_batch::MuonBatchOp;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 5] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];
    let slot = MuonBatchOp::ALL.iter().position(|o| *o == op).expect("a batched-Muon kernel");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::muon_batch::ptx(op)).expect("PTX must be ASCII"))
}

/// Per-matrix sum of squares of the update direction u into norms[i], where
/// u = nesterov ? g + mu*m : m (m already momentum-updated). One block per
/// matrix; deterministic stride-256 accumulate + 128-step shared tree.
pub(crate) fn muon_batch_sumsq_f32_ptx() -> &'static str {
    muon_batch_module(nsl_kir::kernels::muon_batch::MuonBatchOp::Sumsq)
}

/// Pack the normalized update into the wide-layout workspace:
///   Y[i] = u * (1 / (sqrt(norms[i]) + eps)), u as in the sumsq kernel.
/// tr=1 transposes on the fly (input [r,c] -> Y [c,r]) so tall matrices
/// never materialize a separate transpose pass.
pub(crate) fn muon_batch_pack_f32_ptx() -> &'static str {
    muon_batch_module(nsl_kir::kernels::muon_batch::MuonBatchOp::Pack)
}

/// Polynomial combine, in place over the batched Gram workspace:
///   A[i] := ns_b*A[i] + ns_c*AA[i] + (diagonal ? ns_a : 0)
/// Folding ns_a into the diagonal turns the reference x = ns_a*x + b@x into
/// ONE gemm (B' @ x with B' = b + ns_a*I) — same math, fewer passes.
pub(crate) fn muon_batch_poly_f32_ptx() -> &'static str {
    muon_batch_module(nsl_kir::kernels::muon_batch::MuonBatchOp::Poly)
}

/// Fused unpack + parameter update over the pointer table:
///   p[i] = decay * p[i] - step * o[i],  o read from the wide workspace
///   (transposed back on the fly when tr=1). decay = 1 - lr*wd, step =
///   lr * sqrt(max(1, rows/cols)) — both folded on the host.
pub(crate) fn muon_batch_update_f32_ptx() -> &'static str {
    muon_batch_module(nsl_kir::kernels::muon_batch::MuonBatchOp::Update)
}

// ── Fusion-queue item 2: GPU-native cross-entropy backward ─────────────────
//
// Replaces the per-step [N,C] logits DtoH + CPU softmax + HtoD publish with
// three on-device launches (existing softmax + these two). grad_output stays
// on device (read in-kernel) — no host readback anywhere, so the loss
// epilogue no longer taints cuda-graph capture regions.

/// The KIR-built CE backward module for `op`, built once. Both kernels are
/// built by `nsl_kir::kernels::ce_bwd`; its `ce_bwd_kir_equivalence` gate
/// holds them to the hand-written modules they replaced. NUL-terminated.
fn ce_bwd_module(op: nsl_kir::kernels::ce_bwd::CeBwdOp) -> &'static str {
    use nsl_kir::kernels::ce_bwd::CeBwdOp;
    use std::sync::OnceLock;
    static MODULES: [OnceLock<String>; 2] = [OnceLock::new(), OnceLock::new()];
    let slot = CeBwdOp::ALL.iter().position(|o| *o == op).expect("a CE backward kernel");
    MODULES[slot].get_or_init(|| String::from_utf8(nsl_kir::kernels::ce_bwd::ptx(op)).expect("PTX must be ASCII"))
}

/// Count valid (>= 0) targets into scratch[0] as f32 max(count, 1).
/// One 256-thread block; targets read as f32 or s32 per tgt_i32.
/// Valid test mirrors the CPU arm's read_index -> i64 >= 0 (truncate first).
pub(crate) fn ce_bwd_count_f32_ptx() -> &'static str {
    ce_bwd_module(nsl_kir::kernels::ce_bwd::CeBwdOp::Count)
}

/// Finish pass over the softmax output, in place:
///   out[i,j] = valid(t_i) ? (out[i,j] - onehot) * go / denom : 0
/// go read from a device scalar (go_mode=1) or the go_imm immediate
/// (go_mode=0; gop is not read then). denom read from scratch[0].
pub(crate) fn ce_bwd_finish_f32_ptx() -> &'static str {
    ce_bwd_module(nsl_kir::kernels::ce_bwd::CeBwdOp::Finish)
}

// ─── GEMM-chunked fused linear-CE companions (Sprint 2.5 wiring) ───────────
//
// The auto-substituted large-vocab fused linear-CE runs as a loop of cuBLAS
// GEMMs over vocab CHUNKS (never materializing the full [rows, V] logits);
// these three kernels are the per-chunk glue. Measured motivation: the v1
// scalar kernels at Coder-50M shape (rows=1024, V=49152, H=512) run 486 ms
// forward / 2222 ms backward against the composite's ~1.4 / ~2.9 ms GEMMs
// (see fused_linear_ce_perf_probe.rs) — the atomics-based scatter cannot
// compete at production vocab, so the GEMM path is the training path.
//
// exp/log go through ex2.approx / lg2.approx — the same approximation family
// the v1 fused-CE kernels use (their softmax uses ex2.approx too), validated
// against the CPU f64 reference at the same tolerances.

/// Per-chunk online-softmax state update, one 256-thread block per row.
///
/// Folds a [rows, cols] logits chunk (vocab columns chunk_start..+cols of
/// the full head) into three running [rows] state vectors:
///   m (running max), s (running Σexp rescaled to m), tl (target logit).
/// The caller initialises m = -inf (cuMemsetD32 0xFF800000), s = 0, tl = 0.
/// `bias` is the FULL [V] bias vector, indexed at chunk_start+j; ignored
/// when has_bias == 0 (the pointer may be null then).
///
/// Two smem passes per chunk: block-max, then block-Σexp relative to the
/// UPDATED max (thread 0 publishes m_new through sdata[0] between passes,
/// and rescales the running s by exp(m_old - m_new)).
pub(crate) const LCE_CHUNK_STATS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_lce_chunk_stats_f32(\n\
    .param .u64 logits, .param .u64 bias, .param .u64 targets,\n\
    .param .u64 mstate, .param .u64 sstate, .param .u64 tlstate,\n\
    .param .u64 rows, .param .u64 cols, .param .u64 chunk_start,\n\
    .param .u32 has_bias\n\
) {\n\
    .reg .u64 %rd<24>;\n\
    .reg .u32 %r<10>;\n\
    .reg .f32 %f<14>;\n\
    .reg .s64 %tg<2>;\n\
    .reg .pred %p<8>;\n\
    .shared .f32 sdata[256];\n\
    ld.param.u64 %rd1, [logits];\n\
    ld.param.u64 %rd2, [bias];\n\
    ld.param.u64 %rd3, [targets];\n\
    ld.param.u64 %rd4, [mstate];\n\
    ld.param.u64 %rd5, [sstate];\n\
    ld.param.u64 %rd6, [tlstate];\n\
    ld.param.u64 %rd7, [rows];\n\
    ld.param.u64 %rd8, [cols];\n\
    ld.param.u64 %rd9, [chunk_start];\n\
    ld.param.u32 %r5, [has_bias];\n\
    mov.u32 %r1, %ctaid.x;\n\
    cvt.u64.u32 %rd10, %r1;\n\
    setp.ge.u64 %p1, %rd10, %rd7;\n\
    @%p1 bra LCS_DONE;\n\
    mov.u32 %r2, %tid.x;\n\
    cvt.u64.u32 %rd11, %r2;\n\
    // row_base_bytes = row * cols * 4\n\
    mul.lo.u64 %rd12, %rd10, %rd8;\n\
    shl.b64 %rd12, %rd12, 2;\n\
    add.u64 %rd12, %rd1, %rd12;\n\
    // bias chunk base (bytes), used only when has_bias\n\
    shl.b64 %rd13, %rd9, 2;\n\
    add.u64 %rd13, %rd2, %rd13;\n\
    // --- Pass 1: block max of val_j = logits[row,j] (+ bias[cs+j]) ---\n\
    mov.f32 %f1, 0fFF800000;\n\
    mov.u64 %rd14, %rd11;\n\
LCS_MAX_LOOP:\n\
    setp.ge.u64 %p2, %rd14, %rd8;\n\
    @%p2 bra LCS_MAX_DONE;\n\
    shl.b64 %rd15, %rd14, 2;\n\
    add.u64 %rd16, %rd12, %rd15;\n\
    ld.global.f32 %f2, [%rd16];\n\
    setp.eq.u32 %p3, %r5, 0;\n\
    @%p3 bra LCS_MAX_NOB;\n\
    add.u64 %rd17, %rd13, %rd15;\n\
    ld.global.f32 %f3, [%rd17];\n\
    add.f32 %f2, %f2, %f3;\n\
LCS_MAX_NOB:\n\
    max.f32 %f1, %f1, %f2;\n\
    add.u64 %rd14, %rd14, 256;\n\
    bra LCS_MAX_LOOP;\n\
LCS_MAX_DONE:\n\
    mul.lo.u32 %r3, %r2, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f1;\n\
    bar.sync 0;\n\
    setp.ne.u32 %p4, %r2, 0;\n\
    @%p4 bra LCS_MAX_WAIT;\n\
    // thread 0: reduce max, merge with m_old, publish m_new\n\
    mov.u32 %r4, 1;\n\
LCS_MAX_RED:\n\
    setp.ge.u32 %p2, %r4, 256;\n\
    @%p2 bra LCS_MAX_RED_DONE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f4, [%r7];\n\
    max.f32 %f1, %f1, %f4;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LCS_MAX_RED;\n\
LCS_MAX_RED_DONE:\n\
    // m_old / s_old from global state (kept in thread-0 registers)\n\
    shl.b64 %rd18, %rd10, 2;\n\
    add.u64 %rd19, %rd4, %rd18;\n\
    ld.global.f32 %f5, [%rd19];\n\
    max.f32 %f6, %f5, %f1;\n\
    st.shared.f32 [sdata], %f6;\n\
LCS_MAX_WAIT:\n\
    bar.sync 0;\n\
    ld.shared.f32 %f6, [sdata];\n\
    // --- Pass 2: block sum of exp(val_j - m_new) ---\n\
    mov.f32 %f7, 0f00000000;\n\
    mov.u64 %rd14, %rd11;\n\
LCS_SUM_LOOP:\n\
    setp.ge.u64 %p2, %rd14, %rd8;\n\
    @%p2 bra LCS_SUM_DONE;\n\
    shl.b64 %rd15, %rd14, 2;\n\
    add.u64 %rd16, %rd12, %rd15;\n\
    ld.global.f32 %f2, [%rd16];\n\
    setp.eq.u32 %p3, %r5, 0;\n\
    @%p3 bra LCS_SUM_NOB;\n\
    add.u64 %rd17, %rd13, %rd15;\n\
    ld.global.f32 %f3, [%rd17];\n\
    add.f32 %f2, %f2, %f3;\n\
LCS_SUM_NOB:\n\
    sub.f32 %f2, %f2, %f6;\n\
    mul.f32 %f2, %f2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f2, %f2;\n\
    add.f32 %f7, %f7, %f2;\n\
    add.u64 %rd14, %rd14, 256;\n\
    bra LCS_SUM_LOOP;\n\
LCS_SUM_DONE:\n\
    bar.sync 0;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r3;\n\
    st.shared.f32 [%r7], %f7;\n\
    bar.sync 0;\n\
    setp.ne.u32 %p4, %r2, 0;\n\
    @%p4 bra LCS_DONE;\n\
    // thread 0: reduce sum, rescale running s, store state\n\
    mov.u32 %r4, 1;\n\
LCS_SUM_RED:\n\
    setp.ge.u32 %p2, %r4, 256;\n\
    @%p2 bra LCS_SUM_RED_DONE;\n\
    mul.lo.u32 %r6, %r4, 4;\n\
    mov.u32 %r7, sdata;\n\
    add.u32 %r7, %r7, %r6;\n\
    ld.shared.f32 %f4, [%r7];\n\
    add.f32 %f7, %f7, %f4;\n\
    add.u32 %r4, %r4, 1;\n\
    bra LCS_SUM_RED;\n\
LCS_SUM_RED_DONE:\n\
    // s_new = s_old * exp(m_old - m_new) + chunk_sum\n\
    add.u64 %rd20, %rd5, %rd18;\n\
    ld.global.f32 %f8, [%rd20];\n\
    sub.f32 %f9, %f5, %f6;\n\
    mul.f32 %f9, %f9, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f9, %f9;\n\
    fma.rn.f32 %f7, %f8, %f9, %f7;\n\
    st.global.f32 [%rd20], %f7;\n\
    add.u64 %rd19, %rd4, %rd18;\n\
    st.global.f32 [%rd19], %f6;\n\
    // target logit gather: t in [chunk_start, chunk_start+cols)\n\
    shl.b64 %rd21, %rd10, 3;\n\
    add.u64 %rd21, %rd3, %rd21;\n\
    ld.global.s64 %tg0, [%rd21];\n\
    cvt.s64.u64 %tg1, %rd9;\n\
    setp.lt.s64 %p5, %tg0, %tg1;\n\
    @%p5 bra LCS_DONE;\n\
    sub.s64 %tg0, %tg0, %tg1;\n\
    cvt.s64.u64 %tg1, %rd8;\n\
    setp.ge.s64 %p6, %tg0, %tg1;\n\
    @%p6 bra LCS_DONE;\n\
    cvt.u64.s64 %rd22, %tg0;\n\
    shl.b64 %rd15, %rd22, 2;\n\
    add.u64 %rd16, %rd12, %rd15;\n\
    ld.global.f32 %f10, [%rd16];\n\
    setp.eq.u32 %p3, %r5, 0;\n\
    @%p3 bra LCS_TL_NOB;\n\
    add.u64 %rd17, %rd13, %rd15;\n\
    ld.global.f32 %f11, [%rd17];\n\
    add.f32 %f10, %f10, %f11;\n\
LCS_TL_NOB:\n\
    add.u64 %rd23, %rd6, %rd18;\n\
    st.global.f32 [%rd23], %f10;\n\
LCS_DONE: ret;\n\
}\0";

/// Per-chunk softmax-gradient, grid-strided over rows*cols elements,
/// written IN PLACE over the logits chunk (which is dead after this).
///
///   dl[r,j] = valid(r) ? (exp(val - lse[r]) - [t_r == chunk_start+j]) * scale : 0
///
/// `scale` = grad_output / num_valid (host-folded). When has_bias != 0 the
/// per-element dl is also reduced into dbias[chunk_start+j] via
/// red.global.add.f32 (atomic-order nondeterministic, like every dbias in
/// this family; ~rows collisions per column).
pub(crate) const LCE_CHUNK_DLOGITS_F32_PTX: &str = "\
.version 7.0\n\
.target sm_80\n\
.address_size 64\n\
\n\
.visible .entry nsl_lce_chunk_dlogits_f32(\n\
    .param .u64 logits, .param .u64 bias, .param .u64 targets,\n\
    .param .u64 lse, .param .u64 dbias,\n\
    .param .u64 rows, .param .u64 cols, .param .u64 chunk_start,\n\
    .param .f32 scale, .param .u32 has_bias\n\
) {\n\
    .reg .u64 %rd<22>;\n\
    .reg .u32 %r<6>;\n\
    .reg .f32 %f<10>;\n\
    .reg .s64 %tg<3>;\n\
    .reg .pred %p<8>;\n\
    ld.param.u64 %rd1, [logits];\n\
    ld.param.u64 %rd2, [bias];\n\
    ld.param.u64 %rd3, [targets];\n\
    ld.param.u64 %rd4, [lse];\n\
    ld.param.u64 %rd5, [dbias];\n\
    ld.param.u64 %rd6, [rows];\n\
    ld.param.u64 %rd7, [cols];\n\
    ld.param.u64 %rd8, [chunk_start];\n\
    ld.param.f32 %f1, [scale];\n\
    ld.param.u32 %r4, [has_bias];\n\
    // idx = ctaid.x * ntid.x + tid.x; stride = ntid.x * nctaid.x\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ntid.x;\n\
    mov.u32 %r3, %tid.x;\n\
    mul.lo.u32 %r5, %r1, %r2;\n\
    add.u32 %r5, %r5, %r3;\n\
    cvt.u64.u32 %rd9, %r5;\n\
    mov.u32 %r1, %nctaid.x;\n\
    mul.lo.u32 %r5, %r1, %r2;\n\
    cvt.u64.u32 %rd10, %r5;\n\
    mul.lo.u64 %rd11, %rd6, %rd7;\n\
LCD_LOOP:\n\
    setp.ge.u64 %p1, %rd9, %rd11;\n\
    @%p1 bra LCD_DONE;\n\
    // r = idx / cols; j = idx % cols\n\
    div.u64 %rd12, %rd9, %rd7;\n\
    mul.lo.u64 %rd13, %rd12, %rd7;\n\
    sub.u64 %rd13, %rd9, %rd13;\n\
    // val = logits[idx] (+ bias[chunk_start + j])\n\
    shl.b64 %rd14, %rd9, 2;\n\
    add.u64 %rd14, %rd1, %rd14;\n\
    ld.global.f32 %f2, [%rd14];\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra LCD_NOB;\n\
    add.u64 %rd15, %rd8, %rd13;\n\
    shl.b64 %rd15, %rd15, 2;\n\
    add.u64 %rd15, %rd2, %rd15;\n\
    ld.global.f32 %f3, [%rd15];\n\
    add.f32 %f2, %f2, %f3;\n\
LCD_NOB:\n\
    // t = targets[r]; valid = t >= 0\n\
    shl.b64 %rd16, %rd12, 3;\n\
    add.u64 %rd16, %rd3, %rd16;\n\
    ld.global.s64 %tg0, [%rd16];\n\
    setp.lt.s64 %p3, %tg0, 0;\n\
    @!%p3 bra LCD_VALID;\n\
    mov.f32 %f6, 0f00000000;\n\
    bra LCD_STORE;\n\
LCD_VALID:\n\
    // p = exp(val - lse[r])\n\
    shl.b64 %rd17, %rd12, 2;\n\
    add.u64 %rd17, %rd4, %rd17;\n\
    ld.global.f32 %f4, [%rd17];\n\
    sub.f32 %f2, %f2, %f4;\n\
    mul.f32 %f2, %f2, 0f3FB8AA3B;\n\
    ex2.approx.f32 %f5, %f2;\n\
    // onehot subtract\n\
    add.u64 %rd18, %rd8, %rd13;\n\
    cvt.s64.u64 %tg1, %rd18;\n\
    setp.ne.s64 %p4, %tg0, %tg1;\n\
    @%p4 bra LCD_SCALE;\n\
    mov.f32 %f7, 0f3F800000;\n\
    sub.f32 %f5, %f5, %f7;\n\
LCD_SCALE:\n\
    mul.f32 %f6, %f5, %f1;\n\
LCD_STORE:\n\
    st.global.f32 [%rd14], %f6;\n\
    // dbias[chunk_start + j] += dl (bias configs only)\n\
    setp.eq.u32 %p2, %r4, 0;\n\
    @%p2 bra LCD_NEXT;\n\
    add.u64 %rd19, %rd8, %rd13;\n\
    shl.b64 %rd19, %rd19, 2;\n\
    add.u64 %rd19, %rd5, %rd19;\n\
    red.global.add.f32 [%rd19], %f6;\n\
LCD_NEXT:\n\
    add.u64 %rd9, %rd9, %rd10;\n\
    bra LCD_LOOP;\n\
LCD_DONE: ret;\n\
}\0";

/// Finalize the online-softmax state into per-row loss and lse,
/// grid-strided over rows.
///
///   lse[r]  = m[r] + log(s[r])
///   loss[r] = valid(r) ? (lse[r] - tl[r]) : 0        [= -(tl - lse)]
///
/// Built by `nsl_kir::kernels::lce_finalize`; its `lce_finalize_kir_equivalence`
/// gate holds it to the hand-written module it replaced. `ln(s)` is a bare
/// `lg2.approx` with `ln 2` folded into the add by one `fma`, as before.
/// NUL-terminated, built once.
pub(crate) fn lce_finalize_f32_ptx() -> &'static str {
    use std::sync::OnceLock;
    static MODULE: OnceLock<String> = OnceLock::new();
    MODULE.get_or_init(|| String::from_utf8(nsl_kir::kernels::lce_finalize::ptx()).expect("PTX must be ASCII"))
}

/// Every hand-written PTX module in this file, paired with its constant name.
///
/// Consumed by the `ptxas` gate in `super::tests`, which assembles each one.
/// These modules only ever reach `cuModuleLoadData` at runtime, so a syntax error
/// in them is invisible until a kernel launch fails on a real GPU.
#[cfg(test)]
pub(crate) const ALL_PTX: &[(&str, &str)] = &[
    ("LCE_CHUNK_STATS_F32_PTX", LCE_CHUNK_STATS_F32_PTX),
    ("LCE_CHUNK_DLOGITS_F32_PTX", LCE_CHUNK_DLOGITS_F32_PTX),
    ("RMSNORM_DX_BWD_ADD_F32_PTX", RMSNORM_DX_BWD_ADD_F32_PTX),
    ("RMSNORM_DX_BWD_F32_PTX", RMSNORM_DX_BWD_F32_PTX),
    ("SCATTER_ADD_F32_PTX", SCATTER_ADD_F32_PTX),
    ("CONV2D_F32_PTX", CONV2D_F32_PTX),
    ("MAXPOOL2D_F32_PTX", MAXPOOL2D_F32_PTX),
    ("FLASH_LSE_F32_PTX", FLASH_LSE_F32_PTX),
    ("FLASH_LSE_GQA_F32_PTX", FLASH_LSE_GQA_F32_PTX),
    ("BMM_F32_PTX", BMM_F32_PTX),
];
