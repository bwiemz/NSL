//! The draft model's sampler kernel, and the builder sections it shares
//! with the fused sampler (`cfie_sample_ptx`).
//!
//! `nsl_cfie_draft_sample` (registration kind 7) is a greedy argmax over
//! the DRAFT model's LM head: one streaming pass over the vocabulary
//! tiles keeps a running max and argmax. It also writes the argmax's
//! softmax probability (`1 / sum_final`, from an online sum of
//! `exp(x - max)` kept in the same pass). That number is a diagnostic:
//! the speculative driver does not read it, because a greedy draft
//! proposes its token with probability 1, and it is that 1, not the
//! softmax value, that the acceptance rule divides by. `rng_seed` is
//! accepted and unused.
//!
//! The kernel is sampler-family: NO KV access, grid = 1 CTA, block = 128,
//! static `.shared`. RMSNorm(hidden) with the final-norm gamma runs once
//! into SMEM (strided partial sums, SMEM tree reduction, rsqrt.approx,
//! gamma scale in place), the same section the fused sampler starts with.
//!
//! This module used to hold a second kernel, `nsl_cfie_verify_probs`
//! (kind 8), which wrote the target's softmax row for a rejection kernel.
//! Both are gone: that row was a temperature-1 softmax with no top-k or
//! top-p, so verification sampled a different distribution from the
//! configured sampler. Verification now runs the fused sampler itself
//! (`nsl_runtime::cfie::engine`, `speculative_rounds`).
//!
//! `cpu_reference_draft_sample` mirrors the kernel's arithmetic order
//! (same strided partial sums + tree reduction, same fma dot order, same
//! online max/sum merge, same division). The kernel uses `rsqrt.approx` /
//! `ex2.approx` where the CPU uses exact libm.
//!
//! ## KIR (roadmap A2 step 9)
//!
//! The kernel was hand-assembled PTX text until A2 step 9; it is now
//! [`KernelIR`] that the verifier checks before `nsl_kir`'s printer lowers
//! it. The algorithm, and the order of every floating-point operation in
//! it, is the hand kernel's — `tests/cfie_spec_sampler_kir_equivalence.rs`
//! runs the frozen hand emitter and this one side by side on a PTX
//! interpreter and requires the same output bits. As for
//! `cfie_decode_attention`: addresses are element indices through
//! `PtrOffset`, loop-carried values (the loop cursors, the sum of squares,
//! the running max / sum / argmax) are block parameters, shared memory is an
//! [`SmemLayout`] at the hand kernels' packed offsets, and the module
//! targets the KIR floor (`.version 7.0` / `.target sm_70`) instead of
//! the serving GPU, so [`SpecSamplerConfig`] has no `sm_version`.
//! `KirOp::Exp` prints the hand kernels' `mul` by log2(e) then
//! `ex2.approx`, `KirOp::Rsqrt` their `rsqrt.approx.f32`, and an f32
//! `KirOp::Div` their `div.rn.f32`.

use std::fmt::Write;

use crate::backend_ptx::lower_kir_to_ptx;
use crate::cfie_decode_attention::{at, cmp, konst, load, op2, ptr, widen};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};

/// Threads per CTA == vocab tile width (thread t owns row tile_base+t).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

pub const DRAFT_SAMPLE_KERNEL_NAME: &str = "nsl_cfie_draft_sample";

/// Registration kind the serve wiring passes to
/// `nsl_cfie_register_kernel` for the draft sampler (Cycle-13 frozen
/// ABI; kinds 0-5 are the Cycle-6 ABI in `cfie.rs`, kind 6 is the
/// draft decode_block emitted by `cfie_persistent_ptx`).
pub const DRAFT_SAMPLE_KERNEL_KIND: u8 = 7;
/// Compile-time configuration of the draft sampler - mirrors
/// `cfie_sample_ptx::FusedSampleKernelConfig` minus the sampling params
/// (the draft is fixed-function greedy) and minus `sm_version`: the
/// module targets the KIR floor and the driver JIT-compiles it forward.
#[derive(Debug, Clone)]
pub struct SpecSamplerConfig {
    pub d_model: u32,
    pub vocab_size: u32,
    pub vocab_tile: u32,
    /// The final RMSNorm's epsilon (the model's `norm_eps`).
    pub norm_eps: f32,
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct SpecSamplerMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
}

fn f32_imm(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

fn validate_config(cfg: &SpecSamplerConfig) {
    assert_eq!(
        cfg.vocab_tile, TILE,
        "vocab_tile must be {} so thread t owns row tile_base+t",
        TILE
    );
    assert!(
        cfg.d_model >= 1 && cfg.d_model <= 8192,
        "d_model must be in 1..=8192 (hidden state staged in static SMEM)"
    );
    assert!(cfg.vocab_size >= 1, "vocab_size must be >= 1");
}

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------

const R_HIDDEN: u32 = 0;
const R_SCORES: u32 = 1;
const R_RSTD: u32 = 2;

/// `[hidden: d_model][scores: TILE][rstd: 1]`; all f32, 4-aligned and a
/// multiple of 4 long, so the offsets are the hand kernel's packed ones.
fn smem_layout(d_model: u32) -> SmemLayout {
    let f32_region = |name: &str, elems: u32| SmemRegion {
        name: name.to_string(),
        bytes: elems * 4,
        align: 4,
        elem: KirType::F32,
    };
    let regions = vec![
        f32_region("hidden", d_model),
        f32_region("scores", TILE),
        f32_region("rstd", 1),
    ];
    SmemLayout { regions, dynamic: false }
}

// ---------------------------------------------------------------------------
// Builder sections, shared by both kernels
// ---------------------------------------------------------------------------

/// Values every section reads, made once in the entry block. Shared with
/// `cfie_sample_ptx`, whose fused sampler starts with the same sections.
pub(crate) struct Common {
    pub(crate) tid: VarId,
    pub(crate) hidden: VarId,
    pub(crate) norm_w: VarId,
    pub(crate) lm_head: VarId,
    pub(crate) hidden_smem: VarId,
    pub(crate) scores: VarId,
    pub(crate) rstd_smem: VarId,
    pub(crate) zero: VarId,
    pub(crate) one: VarId,
    pub(crate) tile_width: VarId,
    pub(crate) d_model: VarId,
    /// The same value as `d_model`, for the `1 / d_model` immediate.
    pub(crate) d_model_value: u32,
    pub(crate) d_model_wide: VarId,
    /// The final RMSNorm's epsilon.
    pub(crate) norm_eps: f32,
    pub(crate) vocab: VarId,
    pub(crate) f_zero: VarId,
    pub(crate) f_neg_inf: VarId,
}

fn shared_region(b: &mut KirBuilder, region: u32) -> VarId {
    let dst = b.new_typed_var(ptr(KirType::F32, AddressSpace::Shared));
    b.emit(KirOp::SharedRegion { dst, region });
    dst
}

/// `for (i = tid; i < d_model; i += TILE) body(i)`, entered from the
/// current block; the builder is left in the loop's exit block.
fn strided_loop(b: &mut KirBuilder, c: &Common, body: impl FnOnce(&mut KirBuilder, VarId)) {
    let head = b.new_block();
    let body_block = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(head, KirType::U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c.tid])));

    b.set_block(head);
    let finished = cmp(b, i, c.d_model, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body_block)));

    b.set_block(body_block);
    body(b, i);
    let next = op2(b, KirType::U32, KirOp::Add, i, c.tile_width);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![next])));

    b.set_block(done);
}

/// Steps 1 and 2: the hidden row into SMEM, RMSNorm'd in place with the
/// final-norm gamma. Leaves the builder after the closing barrier.
fn hidden_load_and_rmsnorm(b: &mut KirBuilder, c: &Common) {
    hidden_load(b, c);
    rmsnorm_in_place(b, c);
}

/// Step 1: cooperative strided load of hidden `[1, d_model]` f32 into
/// SMEM. Leaves the builder after the closing barrier.
pub(crate) fn hidden_load(b: &mut KirBuilder, c: &Common) {
    use AddressSpace::{Global, Shared};
    use KirType::F32;

    strided_loop(b, c, |b, i| {
        let src = at(b, F32, Global, c.hidden, i);
        let h = load(b, F32, src, Global);
        let dst = at(b, F32, Shared, c.hidden_smem, i);
        b.emit(KirOp::Store(dst, h, Shared));
    });
    b.emit(KirOp::Barrier);
}

/// Step 2: RMSNorm of the staged hidden row, in place, with the final-norm
/// gamma. Leaves the builder after the closing barrier.
pub(crate) fn rmsnorm_in_place(b: &mut KirBuilder, c: &Common) {
    use AddressSpace::{Global, Shared};
    use KirType::F32;

    // The scores region doubles as the reduction scratch before the tile
    // loop.
    sum_of_squares_tree(
        b,
        SquareSum {
            tid: c.tid,
            len: c.d_model,
            stride: c.tile_width,
            f_zero: c.f_zero,
            src: c.hidden_smem,
            scratch: c.scores,
        },
    );

    // Thread 0: rstd = rsqrt(sum_sq / d_model + eps).
    let rstd_compute = b.new_block();
    let rstd_done = b.new_block();
    let is_zero = cmp(b, c.tid, c.zero, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(is_zero, KirEdge::to(rstd_compute), KirEdge::to(rstd_done)));

    b.set_block(rstd_compute);
    let total = load(b, F32, c.scores, Shared);
    let inv_dm = konst(b, ConstValue::F32(1.0 / c.d_model_value as f32));
    let mean = op2(b, F32, KirOp::Mul, total, inv_dm);
    let eps = konst(b, ConstValue::F32(c.norm_eps));
    let shifted = op2(b, F32, KirOp::Add, mean, eps);
    let rstd = b.new_typed_var(F32);
    b.emit(KirOp::Rsqrt(rstd, shifted));
    b.emit(KirOp::Store(c.rstd_smem, rstd, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(rstd_done)));

    b.set_block(rstd_done);
    b.emit(KirOp::Barrier);
    let rstd = load(b, F32, c.rstd_smem, Shared);

    // Scale hidden in place: h = h * rstd * gamma.
    strided_loop(b, c, |b, i| {
        let g_addr = at(b, F32, Global, c.norm_w, i);
        let g = load(b, F32, g_addr, Global);
        let slot = at(b, F32, Shared, c.hidden_smem, i);
        let h = load(b, F32, slot, Shared);
        let scaled = op2(b, F32, KirOp::Mul, h, rstd);
        let normed = op2(b, F32, KirOp::Mul, scaled, g);
        b.emit(KirOp::Store(slot, normed, Shared));
    });
    b.emit(KirOp::Barrier);
}

/// The operands of [`sum_of_squares_tree`]: the thread id, the row length
/// and the block-wide stride (u32), f32 `0`, and two f32 shared pointers.
pub(crate) struct SquareSum {
    pub(crate) tid: VarId,
    pub(crate) len: VarId,
    pub(crate) stride: VarId,
    pub(crate) f_zero: VarId,
    pub(crate) src: VarId,
    pub(crate) scratch: VarId,
}

/// The sum of squares of `src[0..len]` across the CTA, in the hand
/// kernels' order: each thread accumulates `fma(x, x, ss)` over
/// `i = tid, tid + stride, ...`, stores its partial to `scratch[tid]`, and
/// a barrier-separated tree folds `scratch[tid] += scratch[tid + off]` for
/// `off = 64, 32, ..., 1` (so `scratch` holds `TILE` floats and the CTA is
/// `TILE` threads). Shared with `cfie_persistent_ptx`, whose two RMSNorms
/// start the same way. Leaves the builder after the last tree barrier,
/// with the total in `scratch[0]`.
pub(crate) fn sum_of_squares_tree(b: &mut KirBuilder, s: SquareSum) {
    use AddressSpace::Shared;
    use KirType::F32;

    // Per-thread strided partial sum of squares.
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(head, KirType::U32);
    let ss = b.add_block_param(head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![s.tid, s.f_zero])));

    b.set_block(head);
    let finished = cmp(b, i, s.len, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let slot = at(b, F32, Shared, s.src, i);
    let h = load(b, F32, slot, Shared);
    let ss_next = b.new_typed_var(F32);
    b.emit(KirOp::Fma(ss_next, h, h, ss));
    let i_next = op2(b, KirType::U32, KirOp::Add, i, s.stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, ss_next])));

    b.set_block(done);
    let mine = at(b, F32, Shared, s.scratch, s.tid);
    b.emit(KirOp::Store(mine, ss, Shared));
    b.emit(KirOp::Barrier);

    // Tree reduction: scratch[tid] += scratch[tid + off] for tid < off.
    for off in [64u32, 32, 16, 8, 4, 2, 1] {
        let add = b.new_block();
        let join = b.new_block();
        let offset = konst(b, ConstValue::U32(off));
        let active = cmp(b, s.tid, offset, CmpOp::Lt);
        b.terminate(KirTerminator::CondBranch(active, KirEdge::to(add), KirEdge::to(join)));

        b.set_block(add);
        let lo_addr = at(b, F32, Shared, s.scratch, s.tid);
        let lo = load(b, F32, lo_addr, Shared);
        let partner = op2(b, KirType::U32, KirOp::Add, s.tid, offset);
        let hi_addr = at(b, F32, Shared, s.scratch, partner);
        let hi = load(b, F32, hi_addr, Shared);
        let sum = op2(b, F32, KirOp::Add, lo, hi);
        b.emit(KirOp::Store(lo_addr, sum, Shared));
        b.terminate(KirTerminator::Branch(KirEdge::to(join)));

        b.set_block(join);
        b.emit(KirOp::Barrier);
    }
}

/// `dot(hidden_smem, lm_head[tok])` over the f16 row, in the hand
/// kernels' fma order: `dot = fma(w, h, dot)` for `d = 0..d_model`.
/// Entered from the current block; the builder is left in the loop's exit
/// block, and the returned value is the finished dot.
pub(crate) fn build_row_dot(b: &mut KirBuilder, c: &Common, tok: VarId) -> VarId {
    use AddressSpace::{Global, Shared};
    use KirType::{F16, F32, U32, U64};

    let tok_wide = widen(b, tok);
    let row = op2(b, U64, KirOp::Mul, tok_wide, c.d_model_wide);

    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let d = b.add_block_param(head, U32);
    let dot = b.add_block_param(head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c.zero, c.f_zero])));

    b.set_block(head);
    let finished = cmp(b, d, c.d_model, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let d_wide = widen(b, d);
    let index = op2(b, U64, KirOp::Add, row, d_wide);
    let w_addr = at(b, F16, Global, c.lm_head, index);
    let w_raw = load(b, F16, w_addr, Global);
    let w = b.new_typed_var(F32);
    b.emit(KirOp::Cast(w, w_raw, F32));
    let h_addr = at(b, F32, Shared, c.hidden_smem, d);
    let h = load(b, F32, h_addr, Shared);
    let dot_next = b.new_typed_var(F32);
    b.emit(KirOp::Fma(dot_next, w, h, dot));
    let d_next = op2(b, U32, KirOp::Add, d, c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![d_next, dot_next])));

    b.set_block(done);
    dot
}

/// The running state of the streaming pass once every tile is merged.
struct PassResult {
    sum: VarId,
    argmax: Option<VarId>,
}

/// Step 3: per tile every thread scores
/// its row into SMEM, then thread 0 merges the tile's `cnt` scores into
/// the running (max, sum[, argmax]) serially — per element
///   if s > m: sum *= exp(m - s); m = s; [sel = token;]
///   sum += exp(s - m)   (== 1.0 exactly on the max path).
/// Strict `>` keeps first-max-wins tie-breaks. Every thread carries the
/// state through the tile loop; only thread 0's is ever changed or read.
/// Leaves the builder in the loop's exit block.
fn streaming_pass(b: &mut KirBuilder, c: &Common, track_argmax: bool) -> PassResult {
    use AddressSpace::Shared;
    use KirType::{F32, U32};

    let tile_head = b.new_block();
    let tile_body = b.new_block();
    let scored = b.new_block();
    let merge_start = b.new_block();
    let merge_head = b.new_block();
    let merge_body = b.new_block();
    let new_max = b.new_block();
    let accumulate = b.new_block();
    let merged = b.new_block();
    let tiles_done = b.new_block();

    let state = |b: &mut KirBuilder, block| {
        let m = b.add_block_param(block, F32);
        let sum = b.add_block_param(block, F32);
        let sel = track_argmax.then(|| b.add_block_param(block, U32));
        (m, sum, sel)
    };
    let with_state = |mut args: Vec<VarId>, (m, sum, sel): (VarId, VarId, Option<VarId>)| {
        args.extend([m, sum]);
        args.extend(sel);
        args
    };

    let tile = b.add_block_param(tile_head, U32);
    let tile_state = state(b, tile_head);
    let s_scored = b.add_block_param(scored, F32);
    let i = b.add_block_param(merge_head, U32);
    let merge_state = state(b, merge_head);
    let acc_state = state(b, accumulate);
    let merged_state = state(b, merged);

    let initial = (c.f_neg_inf, c.f_zero, track_argmax.then_some(c.zero));
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, with_state(vec![c.zero], initial))));

    b.set_block(tile_head);
    let finished = cmp(b, tile, c.vocab, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(tiles_done), KirEdge::to(tile_body)));

    // Tail-tile guard: lanes past vocab keep -inf.
    b.set_block(tile_body);
    let tok = op2(b, U32, KirOp::Add, tile, c.tid);
    let past = cmp(b, tok, c.vocab, CmpOp::Ge);
    let dot_start = b.new_block();
    b.terminate(KirTerminator::CondBranch(
        past,
        KirEdge::with(scored, vec![c.f_neg_inf]),
        KirEdge::to(dot_start),
    ));
    b.set_block(dot_start);
    let dot = build_row_dot(b, c, tok);
    b.terminate(KirTerminator::Branch(KirEdge::with(scored, vec![dot])));

    b.set_block(scored);
    let mine = at(b, F32, Shared, c.scores, c.tid);
    b.emit(KirOp::Store(mine, s_scored, Shared));
    b.emit(KirOp::Barrier);
    let not_zero = cmp(b, c.tid, c.zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(
        not_zero,
        KirEdge::with(merged, with_state(vec![], tile_state)),
        KirEdge::to(merge_start),
    ));

    // Thread 0: online flash-softmax merge of the tile's cnt scores.
    b.set_block(merge_start);
    let remaining = op2(b, U32, KirOp::Sub, c.vocab, tile);
    let cnt = op2(b, U32, KirOp::Min, remaining, c.tile_width);
    b.terminate(KirTerminator::Branch(KirEdge::with(merge_head, with_state(vec![c.zero], tile_state))));

    b.set_block(merge_head);
    let merge_done = cmp(b, i, cnt, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        merge_done,
        KirEdge::with(merged, with_state(vec![], merge_state)),
        KirEdge::to(merge_body),
    ));

    b.set_block(merge_body);
    let s_addr = at(b, F32, Shared, c.scores, i);
    let s = load(b, F32, s_addr, Shared);
    let (m, sum, sel) = merge_state;
    let above = cmp(b, s, m, CmpOp::Gt);
    b.terminate(KirTerminator::CondBranch(
        above,
        KirEdge::to(new_max),
        KirEdge::with(accumulate, with_state(vec![], merge_state)),
    ));

    // New running max: rescale the online sum[, adopt the argmax].
    b.set_block(new_max);
    let gap = op2(b, F32, KirOp::Sub, m, s);
    let factor = b.new_typed_var(F32);
    b.emit(KirOp::Exp(factor, gap));
    let rescaled = op2(b, F32, KirOp::Mul, sum, factor);
    let adopted = sel.map(|_| op2(b, U32, KirOp::Add, tile, i));
    b.terminate(KirTerminator::Branch(KirEdge::with(accumulate, with_state(vec![], (s, rescaled, adopted)))));

    // exp(0) == 1.0 exactly on the max path.
    b.set_block(accumulate);
    let (am, asum, asel) = acc_state;
    let delta = op2(b, F32, KirOp::Sub, s, am);
    let term = b.new_typed_var(F32);
    b.emit(KirOp::Exp(term, delta));
    let sum_next = op2(b, F32, KirOp::Add, asum, term);
    let i_next = op2(b, U32, KirOp::Add, i, c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(
        merge_head,
        with_state(vec![i_next], (am, sum_next, asel)),
    )));

    // The scores region is rewritten next tile; sync before looping back.
    b.set_block(merged);
    b.emit(KirOp::Barrier);
    let tile_next = op2(b, U32, KirOp::Add, tile, c.tile_width);
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, with_state(vec![tile_next], merged_state))));

    b.set_block(tiles_done);
    let (_max, sum, argmax) = tile_state;
    PassResult { sum, argmax }
}

/// The builder, the params both kernels share (hidden, norm_w, lm_head)
/// already added in FFI order, and the entry block started.
fn begin(name: &str, cfg: &SpecSamplerConfig) -> (KirBuilder, VarId, VarId, VarId) {
    use AddressSpace::Global;
    use KirType::{F16, F32};

    let mut b = KirBuilder::new(name);
    let hidden = b.add_param("hidden_ptr", ptr(F32, Global), Global);
    let norm_w = b.add_param("norm_w_ptr", ptr(F32, Global), Global);
    let lm_head = b.add_param("lm_head_ptr", ptr(F16, Global), Global);
    b.set_smem_layout(smem_layout(cfg.d_model));
    b.set_workgroup_size([BLOCK_DIM, 1, 1]);
    (b, hidden, norm_w, lm_head)
}

/// The entry block's shared values. Call with the builder in the entry
/// block, after every param is added.
fn common(b: &mut KirBuilder, cfg: &SpecSamplerConfig, hidden: VarId, norm_w: VarId, lm_head: VarId) -> Common {
    common_at(b, cfg.d_model, cfg.vocab_size, cfg.norm_eps, [hidden, norm_w, lm_head], R_RSTD)
}

/// [`common`] for any kernel whose shared layout starts `[hidden][scores]`
/// (regions 0 and 1) and keeps `rstd` at region `rstd_region`. `pointers`
/// are the `hidden`, `norm_w` and `lm_head` params, in that order.
pub(crate) fn common_at(
    b: &mut KirBuilder,
    d_model: u32,
    vocab_size: u32,
    norm_eps: f32,
    pointers: [VarId; 3],
    rstd_region: u32,
) -> Common {
    let [hidden, norm_w, lm_head] = pointers;
    let tid = b.new_typed_var(KirType::U32);
    b.emit(KirOp::ThreadId(tid, 0));
    Common {
        tid,
        hidden,
        norm_w,
        lm_head,
        hidden_smem: shared_region(b, R_HIDDEN),
        scores: shared_region(b, R_SCORES),
        rstd_smem: shared_region(b, rstd_region),
        zero: konst(b, ConstValue::U32(0)),
        one: konst(b, ConstValue::U32(1)),
        tile_width: konst(b, ConstValue::U32(TILE)),
        d_model: konst(b, ConstValue::U32(d_model)),
        d_model_value: d_model,
        d_model_wide: konst(b, ConstValue::U64(d_model as u64)),
        norm_eps,
        vocab: konst(b, ConstValue::U32(vocab_size)),
        f_zero: konst(b, ConstValue::F32(0.0)),
        f_neg_inf: konst(b, ConstValue::F32(f32::NEG_INFINITY)),
    }
}

// ---------------------------------------------------------------------------
// Kind 7: nsl_cfie_draft_sample
// ---------------------------------------------------------------------------

/// The draft greedy sampler for `cfg`, as KIR.
pub fn build_draft_sample(cfg: &SpecSamplerConfig) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};

    validate_config(cfg);
    let (mut b, hidden, norm_w, lm_head) = begin(DRAFT_SAMPLE_KERNEL_NAME, cfg);
    let out_token = b.add_param("out_token_ptr", ptr(U32, Global), Global);
    let out_prob = b.add_param("out_prob_ptr", ptr(F32, Global), Global);
    // ACCEPTED for ABI symmetry with the fused sampler; UNUSED - v1
    // drafting is greedy.
    b.add_param("rng_seed", U64, Global);

    let entry = b.new_block();
    b.set_block(entry);
    let c = common(&mut b, cfg, hidden, norm_w, lm_head);

    hidden_load_and_rmsnorm(&mut b, &c);
    let pass = streaming_pass(&mut b, &c, true);

    // Thread 0 publishes; the kernel's only global stores.
    let publish = b.new_block();
    let exit = b.new_block();
    let not_zero = cmp(&mut b, c.tid, c.zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_zero, KirEdge::to(exit), KirEdge::to(publish)));

    // p(argmax) = 1 / sum_final (exp(x_argmax - max) == 1).
    b.set_block(publish);
    let f_one = konst(&mut b, ConstValue::F32(1.0));
    let p = op2(&mut b, F32, KirOp::Div, f_one, pass.sum);
    let argmax = pass.argmax.expect("the draft pass tracks the argmax");
    b.emit(KirOp::Store(out_token, argmax, Global));
    b.emit(KirOp::Store(out_prob, p, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// The `//` header the hand kernel carried, line for line.
fn header_comment(cfg: &SpecSamplerConfig) -> String {
    let eps = f32_imm(cfg.norm_eps);
    let mut w = String::new();
    writeln!(w, "//").unwrap();
    writeln!(w, "// {} - CFIE draft-model greedy sampler (Cycle 13, G15).", DRAFT_SAMPLE_KERNEL_NAME).unwrap();
    writeln!(w, "// One CTA, {} threads; ONE streaming pass over the vocab tiles keeps", BLOCK_DIM).unwrap();
    writeln!(w, "// a running max + argmax + online sum of exp(x - max) with rescale").unwrap();
    writeln!(w, "// (flash softmax).  p(argmax) = 1/sum_final because").unwrap();
    writeln!(w, "// exp(x_argmax - max) == 1 when the argmax attains the max.").unwrap();
    writeln!(w, "// Outputs: token id (u32) + p(argmax) (f32) - 8 bytes to HBM.").unwrap();
    writeln!(w, "// LM-head layout: f16 [vocab, d_model], ROW-major per vocab row.").unwrap();
    writeln!(w, "// rng_seed is ACCEPTED for ABI symmetry with the fused sampler but").unwrap();
    writeln!(w, "// UNUSED: v1 drafting is greedy (the paper's temperature 0.0).").unwrap();
    writeln!(w, "// Baked constants:").unwrap();
    writeln!(w, "//   d_model    = {}", cfg.d_model).unwrap();
    writeln!(w, "//   vocab_size = {}", cfg.vocab_size).unwrap();
    writeln!(w, "//   vocab_tile = {}", TILE).unwrap();
    writeln!(w, "//   rms_eps    = {} ({})", cfg.norm_eps, eps).unwrap();
    writeln!(w, "//").unwrap();
    w
}

/// Verify, lower and prefix the header. The returned text carries no NUL:
/// the serve path appends the one the driver wants when it embeds the
/// module.
fn emit(ir: &KernelIR, cfg: &SpecSamplerConfig) -> (String, SpecSamplerMeta) {
    if let Err(errors) = crate::kir_verify::verify(ir) {
        panic!("{} failed KIR verification: {errors:?}", ir.name);
    }
    let smem_bytes = ir.smem_layout.total_bytes().expect("a verified layout has a size");
    let module = lower_kir_to_ptx(ir);
    let module = module.strip_suffix(&[0]).unwrap_or(&module);
    let module = std::str::from_utf8(module).expect("the KIR printer emits ASCII");

    let mut p = header_comment(cfg);
    p.push_str(module);
    let meta = SpecSamplerMeta { kernel_name: ir.name.clone(), smem_bytes, block_dim: BLOCK_DIM };
    (p, meta)
}

/// Emit the draft greedy sampler kernel for `cfg`.
pub fn emit_draft_sample(cfg: &SpecSamplerConfig) -> (String, SpecSamplerMeta) {
    emit(&build_draft_sample(cfg), cfg)
}

/// PTX-only convenience wrapper around [`emit_draft_sample`].
pub fn emit_draft_sample_ptx(cfg: &SpecSamplerConfig) -> String {
    emit_draft_sample(cfg).0
}


// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

/// Kernel-order RMSNorm: per-thread strided partial sums, the SMEM
/// tree reduction, gamma scale - identical staging to the kernels
/// (and to `cfie_sample_ptx::cpu_reference`).  The kernel's
/// `rsqrt.approx` is exact libm here; token parity is unaffected (the
/// argmax compares dots computed from the same scaled hidden).
fn rms_norm_kernel_order(hidden: &[f32], norm_w: &[f32], norm_eps: f32) -> Vec<f32> {
    let dm = hidden.len();
    let mut partials = [0f32; TILE as usize];
    for (t, part) in partials.iter_mut().enumerate() {
        let mut s = 0f32;
        let mut i = t;
        while i < dm {
            s = hidden[i].mul_add(hidden[i], s);
            i += TILE as usize;
        }
        *part = s;
    }
    for off in [64usize, 32, 16, 8, 4, 2, 1] {
        for t in 0..off {
            partials[t] += partials[t + off];
        }
    }
    let mean = partials[0] * (1.0 / dm as f32);
    let rstd = 1.0 / (mean + norm_eps).sqrt();
    hidden
        .iter()
        .zip(norm_w)
        .map(|(h, g)| (h * rstd) * g)
        .collect()
}

/// The kernels' fma-ordered f16-row matvec (same order as the fused
/// sampler's reference).  `lm_head_f32` is `[vocab][d_model]`
/// row-major f32 - an exact-value cast of the kernel's f16 weights.
fn row_dot(h: &[f32], lm_head_f32: &[f32], tok: usize) -> f32 {
    let dm = h.len();
    let row = &lm_head_f32[tok * dm..(tok + 1) * dm];
    let mut dot = 0f32;
    for d in 0..dm {
        dot = row[d].mul_add(h[d], dot);
    }
    dot
}

/// Running state of the streaming online-softmax pass.
struct OnlineSoftmax {
    m: f32,
    sum: f32,
    sel: u32,
}

/// The kernels' streaming pass, element order and arithmetic mirrored
/// exactly: per tile the scores are staged first (the SMEM write), then
/// merged serially with strict-`>` first-max-wins tie-breaks.
fn online_pass(h: &[f32], lm_head_f32: &[f32], vocab: usize) -> OnlineSoftmax {
    let log2e = std::f32::consts::LOG2_E;
    let mut st = OnlineSoftmax {
        m: f32::NEG_INFINITY,
        sum: 0.0,
        sel: 0,
    };
    let mut tile = 0usize;
    while tile < vocab {
        let cnt = (vocab - tile).min(TILE as usize);
        let mut scores = vec![0f32; cnt];
        for (t, sc) in scores.iter_mut().enumerate() {
            *sc = row_dot(h, lm_head_f32, tile + t);
        }
        for (i, &s) in scores.iter().enumerate() {
            if s > st.m {
                st.sum *= ((st.m - s) * log2e).exp2();
                st.m = s;
                st.sel = (tile + i) as u32;
            }
            st.sum += ((s - st.m) * log2e).exp2();
        }
        tile += TILE as usize;
    }
    st
}

fn validate_cpu_inputs(
    cfg: &SpecSamplerConfig,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
) -> (usize, usize) {
    let dm = cfg.d_model as usize;
    let vocab = cfg.vocab_size as usize;
    assert_eq!(cfg.vocab_tile, TILE, "vocab_tile must be {TILE}");
    assert!((1..=8192).contains(&dm), "d_model must be in 1..=8192");
    assert!(vocab >= 1, "vocab_size must be >= 1");
    assert_eq!(hidden.len(), dm, "hidden must be [1, d_model]");
    assert_eq!(norm_w.len(), dm, "norm_w (gamma) must be [d_model]");
    assert_eq!(
        lm_head_f32.len(),
        vocab * dm,
        "lm_head must be [vocab][d_model] row-major"
    );
    (dm, vocab)
}

/// CPU mirror of `nsl_cfie_draft_sample`: returns
/// `(argmax token, p(argmax))`.  Same accumulation order as the kernel
/// so parity is exact-token + tight-float (see module docs).
pub fn cpu_reference_draft_sample(
    cfg: &SpecSamplerConfig,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
) -> (u32, f32) {
    let (_, vocab) = validate_cpu_inputs(cfg, hidden, norm_w, lm_head_f32);
    let h = rms_norm_kernel_order(hidden, norm_w, cfg.norm_eps);
    let st = online_pass(&h, lm_head_f32, vocab);
    (st.sel, 1.0 / st.sum)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};

    fn cfg(d_model: u32, vocab_size: u32) -> SpecSamplerConfig {
        SpecSamplerConfig {
            d_model,
            vocab_size,
            vocab_tile: 128,
            norm_eps: RMS_EPS,
        }
    }

    /// The epsilon the tests bake; `norm_eps_reaches_the_kernel` checks
    /// that another value is honoured.
    const RMS_EPS: f32 = 1e-5;

    /// Reference config from the CFIE paper's NSL-Coder example (the
    /// draft model shares the target's vocab per the spec sub-block).
    fn paper_cfg() -> SpecSamplerConfig {
        cfg(512, 49_152)
    }

    fn f16r(v: f32) -> f32 {
        half::f16::from_f32(v).to_f32()
    }

    /// f16-rounded pseudo-random LM head (exact-value f32 cast, the
    /// kernels' weight contract).
    fn lm_head(vocab: usize, dm: usize) -> Vec<f32> {
        (0..vocab * dm)
            .map(|i| f16r(((i * 7 + 3) % 23) as f32 * 0.07 - 0.7))
            .collect()
    }

    fn hidden_pattern(dm: usize) -> Vec<f32> {
        (0..dm).map(|i| (i as f32 * 0.37).sin()).collect()
    }

    fn gamma_pattern(dm: usize) -> Vec<f32> {
        (0..dm).map(|i| 1.0 + i as f32 * 0.01).collect()
    }

    /// Every op of every block, flattened (terminators aside).
    fn ops(ir: &KernelIR) -> Vec<&KirOp> {
        ir.blocks.iter().flat_map(|b| b.ops.iter()).collect()
    }

    fn count(ir: &KernelIR, pred: impl Fn(&KirOp) -> bool) -> usize {
        ops(ir).into_iter().filter(|op| pred(op)).count()
    }

    fn global_stores(ir: &KernelIR) -> Vec<(VarId, VarId)> {
        ops(ir)
            .into_iter()
            .filter_map(|op| match op {
                KirOp::Store(addr, val, AddressSpace::Global) => Some((*addr, *val)),
                _ => None,
            })
            .collect()
    }

    fn param_names(ir: &KernelIR) -> Vec<&str> {
        ir.params.iter().map(|p| p.name.as_str()).collect()
    }

    // -- structural ------------------------------------------------------

    #[test]
    fn draft_param_list_is_exactly_the_six_params() {
        let ir = build_draft_sample(&paper_cfg());
        assert_eq!(
            param_names(&ir),
            ["hidden_ptr", "norm_w_ptr", "lm_head_ptr", "out_token_ptr", "out_prob_ptr", "rng_seed"]
        );
        // Every param is 8 bytes: five device pointers and the u64 seed.
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        for name in param_names(&ir) {
            assert!(ptx.contains(&format!(".param .u64 param_{name}")), "{name}");
        }
    }

    #[test]
    fn draft_stores_exactly_token_and_prob() {
        // The draft sampler's whole output is 8 bytes: token + prob,
        // stored straight through the two output params.
        let ir = build_draft_sample(&paper_cfg());
        let out_token = ir.params[3].id;
        let out_prob = ir.params[4].id;
        let stores = global_stores(&ir);
        assert_eq!(stores.iter().map(|(a, _)| *a).collect::<Vec<_>>(), [out_token, out_prob]);
        assert_eq!(ir.var_types[&stores[0].1], KirType::U32, "the token is a u32");
        assert_eq!(ir.var_types[&stores[1].1], KirType::F32, "p(argmax) is an f32");
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        assert_eq!(ptx.matches("st.global").count(), 2);
        assert_eq!(ptx.matches("st.global.u32").count(), 1);
    }

    #[test]
    fn no_mad_lo_ascii_only_and_line_width() {
        for ptx in [emit_draft_sample_ptx(&paper_cfg())] {
            assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
            assert!(
                ptx.bytes().all(|b| b < 128),
                "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
            );
            assert!(!ptx.ends_with('\0'), "String PTX carries no trailing NUL");
            // The KIR printer writes the entry signature on one line, as
            // for every KIR kernel; the column limit was the hand
            // emitter's style for the rest, and still holds there.
            assert!(
                ptx.lines().filter(|l| !l.starts_with(".visible .entry ")).all(|l| l.len() <= 132),
                "PTX lines must stay within 132 columns"
            );
        }
    }

    #[test]
    fn draft_is_greedy_no_prng_no_temperature() {
        // rng_seed is ABI symmetry only: no op reads it, no xorshift64*
        // constants, no RNG mixing, no temperature epilogue on the scores.
        let ir = build_draft_sample(&paper_cfg());
        let seed = ir.params[5].id;
        let reads_seed = ir.blocks.iter().any(|b| {
            b.ops.iter().any(|op| crate::kir_verify::op_uses(op).contains(&seed))
                || b.terminator.as_ref().is_some_and(|t| crate::kir_verify::terminator_uses(t).contains(&seed))
        });
        assert!(!reads_seed, "the seed must not feed any op");
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        assert!(!ptx.contains("0x2545F4914F6CDD1D"));
        assert!(!ptx.contains("0x9E3779B97F4A7C15"));
        assert!(!ptx.contains("RNG"));
        assert!(ptx.contains("rng_seed is ACCEPTED for ABI symmetry"));
    }

    #[test]
    fn rmsnorm_and_flash_softmax_sections_present() {
        for (ir, exps) in [
            // Rescale + accumulate in the merge.
            (build_draft_sample(&paper_cfg()), 2),
        ] {
            assert_eq!(count(&ir, |op| matches!(op, KirOp::Rsqrt(..))), 1, "{}", ir.name);
            assert_eq!(count(&ir, |op| matches!(op, KirOp::Exp(..))), exps, "{}", ir.name);
            // One cooperative strided load, one norm scale, one score store
            // per tile; the 7-step tree reduction.
            let barriers = count(&ir, |op| matches!(op, KirOp::Barrier));
            assert!(barriers >= 13, "{}: {barriers} barriers", ir.name);
        }
        for ptx in [emit_draft_sample_ptx(&paper_cfg())] {
            assert!(ptx.contains("rsqrt.approx.f32"));
            assert!(ptx.contains(&f32_imm(RMS_EPS)));
            assert!(ptx.contains("ex2.approx.f32"));
        }
    }

    #[test]
    fn header_is_the_kir_floor() {
        // No sm_version: the modules target the KIR floor and the driver
        // JIT-compiles them forward (tests/cfie_ptx_headers_ptxas.rs
        // assembles them for every serving architecture).
        for ptx in [emit_draft_sample_ptx(&paper_cfg())] {
            assert!(ptx.starts_with("//"));
            assert!(ptx.contains(".version 7.0\n.target sm_70\n.address_size 64"));
        }
    }

    #[test]
    fn meta_reports_launch_shape() {
        let (ptx, meta) = emit_draft_sample(&paper_cfg());
        assert_eq!(meta.kernel_name, DRAFT_SAMPLE_KERNEL_NAME);
        assert_eq!(meta.block_dim, 128);
        // hidden(512 f32) + scores(128 f32) + rstd.
        assert_eq!(meta.smem_bytes, 512 * 4 + 128 * 4 + 4);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 shared_mem[{}];", meta.smem_bytes)));
    }

    #[test]
    fn registration_kind_matches_the_runtime() {
        assert_eq!(DRAFT_SAMPLE_KERNEL_KIND, 7);
    }

    /// The configured epsilon is the one in the module: the kernel used to
    /// bake 1e-5 whatever the model's `norm_eps` was.
    #[test]
    fn norm_eps_reaches_the_kernel() {
        let mut c = paper_cfg();
        c.norm_eps = 1e-6;
        let ptx = emit_draft_sample_ptx(&c);
        assert!(ptx.contains(&f32_imm(1e-6)), "the configured epsilon is baked");
        assert!(!ptx.contains(&f32_imm(1e-5)), "the old constant is not");
        // And it changes the answer where it should: a hidden row whose
        // mean square is comparable to the epsilon.
        let (dm, vocab) = (8usize, 16usize);
        let hidden = vec![1e-3f32; dm];
        let gamma = vec![1.0f32; dm];
        let w = lm_head(vocab, dm);
        let a = SpecSamplerConfig { d_model: dm as u32, vocab_size: vocab as u32, vocab_tile: 128, norm_eps: 1e-5 };
        let b = SpecSamplerConfig { norm_eps: 1e-6, ..a.clone() };
        let (_, pa) = cpu_reference_draft_sample(&a, &hidden, &gamma, &w);
        let (_, pb) = cpu_reference_draft_sample(&b, &hidden, &gamma, &w);
        // rstd = 1/sqrt(1e-6 + eps): 301.5 against 707.1, so the logits
        // differ by a factor 2.35 and the softmax peak with them.
        assert!((pa - pb).abs() > 1e-3, "{pa} vs {pb}");
    }

    // -- refusals ---------------------------------------------------------

    #[test]
    #[should_panic(expected = "vocab_tile")]
    fn draft_vocab_tile_not_128_panics() {
        let mut c = paper_cfg();
        c.vocab_tile = 256;
        let _ = emit_draft_sample(&c);
    }

    #[test]
    #[should_panic(expected = "vocab_size")]
    fn draft_zero_vocab_panics() {
        let mut c = paper_cfg();
        c.vocab_size = 0;
        let _ = emit_draft_sample(&c);
    }

    // -- cpu references ----------------------------------------------------

    #[test]
    fn cpu_draft_greedy_matches_fused_sampler_greedy() {
        let (dm, vocab) = (16usize, 48usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let mut w = lm_head(vocab, dm);
        // Plant a unique max at row 37: align the row with hidden's
        // signs (RMSNorm preserves signs - rstd and gamma positive).
        for d in 0..dm {
            w[37 * dm + d] = f16r(if hidden[d] >= 0.0 { 4.0 } else { -4.0 });
        }
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(tok, 37);
        assert!(prob > 0.0 && prob <= 1.0, "p(argmax) must be a probability");

        // Cross-check: the fused sampler's greedy cpu_reference with
        // top_k == vocab sees every token, so its argmax is global.
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: vocab as u32,
            ..Default::default()
        };
        let prog = emit_program(
            params,
            LmHeadShape {
                d_model: dm as u32,
                vocab_size: vocab as u32,
                vocab_tile: 128,
                dtype_bytes: 2,
            },
        );
        let fused = crate::cfie_sample_ptx::cpu_reference(&prog, &hidden, &gamma, &w, None, 7);
        assert_eq!(tok, fused, "draft greedy must agree with the fused sampler");
    }

    #[test]
    fn cpu_draft_prob_matches_f64_full_softmax() {
        // vocab 300 = 2 full tiles + a 44-wide tail tile.
        let (dm, vocab) = (32usize, 300usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);

        // Independent f64 pipeline (plain RMSNorm + softmax).
        let mut ss = 0f64;
        for &h in &hidden {
            ss += h as f64 * h as f64;
        }
        let rstd = 1.0 / (ss / dm as f64 + RMS_EPS as f64).sqrt();
        let h64: Vec<f64> = hidden
            .iter()
            .zip(&gamma)
            .map(|(h, g)| *h as f64 * rstd * *g as f64)
            .collect();
        let logits: Vec<f64> = (0..vocab)
            .map(|t| (0..dm).map(|d| w[t * dm + d] as f64 * h64[d]).sum())
            .collect();
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|x| (x - max).exp()).sum();
        let argmax = logits
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |acc, (i, &x)| {
                if x > acc.1 {
                    (i, x)
                } else {
                    acc
                }
            })
            .0;
        assert_eq!(tok as usize, argmax);
        let p64 = (logits[argmax] - max).exp() / sum;
        assert!(
            (prob as f64 - p64).abs() < 1e-4,
            "draft prob {prob} vs f64 softmax {p64}"
        );
    }

    #[test]
    fn rescale_after_mass_pinned_by_planted_ascending_maxima() {
        // Verify-1 should-fix: pin the rescale-after-mass path
        // STRUCTURALLY, not incidentally.  Five planted rows with
        // strictly ascending dots sit deep into accumulation (tiles 0,
        // 1, 2, 3, 4 of a 640-token vocab), so the running max updates
        // five times AFTER substantial mass has accumulated - each
        // update must rescale the online sum or p(argmax) inflates by
        // ~exp(delta) per missed rescale, far outside the 1e-4 bound
        // against the independent f64 softmax below.
        let (dm, vocab) = (32usize, 640usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let mut w = lm_head(vocab, dm);
        let planted = [5usize, 150, 290, 400, 560];
        for (j, &tok) in planted.iter().enumerate() {
            let amp = 0.5 + j as f32 * 0.5; // strictly ascending dots
            for d in 0..dm {
                w[tok * dm + d] = f16r(if hidden[d] >= 0.0 { amp } else { -amp });
            }
        }
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(
            tok as usize,
            planted[planted.len() - 1],
            "the last (largest) planted max must win"
        );

        // Independent f64 pipeline (plain RMSNorm + softmax).
        let mut ss = 0f64;
        for &h in &hidden {
            ss += h as f64 * h as f64;
        }
        let rstd = 1.0 / (ss / dm as f64 + RMS_EPS as f64).sqrt();
        let h64: Vec<f64> = hidden
            .iter()
            .zip(&gamma)
            .map(|(h, g)| *h as f64 * rstd * *g as f64)
            .collect();
        let logits: Vec<f64> = (0..vocab)
            .map(|t| (0..dm).map(|d| w[t * dm + d] as f64 * h64[d]).sum())
            .collect();
        // The planted dots really are strictly ascending running maxima
        // (the fixture's premise - assert it so a weight tweak cannot
        // silently degrade this test back to zero rescales).
        let mut prev = f64::NEG_INFINITY;
        for &t in &planted {
            assert!(
                logits[t] > prev,
                "planted maxima must strictly ascend (token {t})"
            );
            let run_max = logits[..t]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                logits[t] > run_max,
                "token {t} must beat every earlier logit (a real max update)"
            );
            prev = logits[t];
        }
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|x| (x - max).exp()).sum();
        let p64 = (logits[tok as usize] - max).exp() / sum;
        assert!(
            (prob as f64 - p64).abs() < 1e-4,
            "planted-maxima prob {prob} vs f64 softmax {p64}"
        );
    }

    #[test]
    fn cpu_draft_finds_max_in_tail_tile() {
        // vocab 130 = one full tile + a 2-wide tail; plant the max at
        // token 129 so the tail-tile guard path is the winner.
        let (dm, vocab) = (4usize, 130usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = vec![1.0f32; dm];
        let mut w = lm_head(vocab, dm);
        for d in 0..dm {
            w[129 * dm + d] = f16r(if hidden[d] >= 0.0 { 4.0 } else { -4.0 });
        }
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(tok, 129);
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn cpu_draft_vocab_one_is_certain() {
        // Single-token vocab: p(argmax) must be exactly 1.0 (the
        // online sum is exactly ex2(0) == 1).
        let dm = 4usize;
        let c = cfg(dm as u32, 1);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(1, dm);
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        assert_eq!(tok, 0);
        assert_eq!(prob.to_bits(), 1.0f32.to_bits());
    }

    // -- ptxas validation (skips silently when no validator present) --

    #[test]
    fn ptxas_validates_draft_sample_paper_config() {
        let ptx = emit_draft_sample_ptx(&paper_cfg());
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                nsl_log::nsl_log!(INFO, "skip", "[skip] cfie draft-sample ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie draft-sample PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }

}
