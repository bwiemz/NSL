//! CFIE Cycle 13 (G15 draft-model-in-binary): the two draft-side
//! sampler-family kernels.
//!
//!   * `nsl_cfie_draft_sample` (registration kind 7) - greedy argmax +
//!     p(chosen) over the DRAFT model's LM head.  The paper drafts K
//!     tokens at temperature 0.0, recording `draft_probs[k] =
//!     softmax(draft_logits)[chosen]`; greedy means chosen == argmax,
//!     so p(argmax) falls out of ONE streaming flash-softmax pass for
//!     free: `exp(x_argmax - max) == 1`, hence `p = 1 / sum_final`.
//!     The pass keeps a running max + argmax + online sum of
//!     `exp(x - max)` with rescale - the same trick the fused sampler
//!     (`cfie_sample_ptx`) already uses.  `rng_seed` is ACCEPTED for
//!     ABI symmetry with the fused sampler but UNUSED - v1 drafting is
//!     greedy per the paper.
//!
//!   * `nsl_cfie_verify_probs` (registration kind 8) - full softmaxed
//!     prob-ROW writer for the TARGET model.  The fused sampler never
//!     materializes probs by design (its whole point is that the
//!     `[1, vocab]` logits row never touches HBM), but the rejection
//!     kernel (`cfie_speculative_ptx::REJECT_KERNEL_NAME`, kind 4)
//!     consumes softmaxed f32 rows `target_probs[k][vocab]`.  This
//!     kernel EXISTS to materialize that row: pass 1 computes max +
//!     sum online (recomputing the matvec per tile), pass 2 recomputes
//!     the matvec per tile and stores `p_i = exp(x_i - max) / sum`.
//!     The 2x matvec is the price of never staging the logits row and
//!     is bounded by K <= 32 verify positions per round.
//!
//! Both kernels are sampler-family: NO KV access, grid = 1 CTA,
//! block = 128, static `.shared`.  RMSNorm(hidden) with the final-norm
//! gamma runs once into SMEM - the section is identical to
//! `cfie_sample_ptx` (strided partial sums, SMEM tree reduction,
//! rsqrt.approx, gamma scale in place).
//!
//! Lossless self-speculation anchor (the engine's determinism proof):
//! with draft == target weights both kernels compute bit-identical
//! (max, sum) - same fma dot order, same online-merge order - so
//! `p_draft = div(1, sum)` from kind 7 equals
//! `p_target[tok] = div(ex2(0), sum)` from kind 8 bit-for-bit
//! (`ex2(+-0) == 1.0` exactly per the PTX ISA).  The reject kernel's
//! ratio is then exactly 1.0 and every drafted token accepts
//! regardless of seed.  Both kernels are built from the same builder
//! sections below, so the shared order is one piece of code, not two
//! copies kept in step.
//!
//! `cpu_reference_*` mirror the kernels' arithmetic order exactly
//! (same strided partial sums + tree reduction, same fma dot order,
//! same online max/sum merge, same division).  The kernels use
//! `rsqrt.approx` / `ex2.approx` where the CPU uses exact libm - token
//! parity is expected-exact modulo rsqrt.approx rounding on knife-edge
//! argmax ties (the rstd delta perturbs the normalized hidden by <1
//! ulp-scale, so only dots equal to within that delta could flip), and
//! prob parity is tight-float.  The determinism contract does NOT rest
//! on CPU-vs-GPU parity - it rests on GPU-side kind-7 vs kind-8 BIT
//! identity (above), proven by the engine's self-speculation contract
//! (spec generate == plain generate).
//!
//! ## KIR (roadmap A2 step 9)
//!
//! Both kernels were hand-assembled PTX text until A2 step 9; they are now
//! [`KernelIR`] that the verifier checks before `nsl_kir`'s printer lowers
//! it. The algorithm, and the order of every floating-point operation in
//! it, is the hand kernels' — `tests/cfie_spec_sampler_kir_equivalence.rs`
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

/// Baked RMSNorm epsilon - same value as `cfie_sample_ptx` (paper
/// stage 1); the draft and target final norms share it.
const RMS_EPS: f32 = 1e-5;

pub const DRAFT_SAMPLE_KERNEL_NAME: &str = "nsl_cfie_draft_sample";
pub const VERIFY_PROBS_KERNEL_NAME: &str = "nsl_cfie_verify_probs";

/// Registration kind the serve wiring passes to
/// `nsl_cfie_register_kernel` for the draft sampler (Cycle-13 frozen
/// ABI; kinds 0-5 are the Cycle-6 ABI in `cfie.rs`, kind 6 is the
/// draft decode_block emitted by `cfie_persistent_ptx`).
pub const DRAFT_SAMPLE_KERNEL_KIND: u8 = 7;
/// Registration kind for the verify prob-row writer (Cycle-13 ABI).
pub const VERIFY_PROBS_KERNEL_KIND: u8 = 8;

/// Compile-time configuration shared by both kernels - mirrors
/// `cfie_sample_ptx::FusedSampleKernelConfig` minus the sampling
/// params (the draft is fixed-function greedy; the verify writer has
/// no sampling at all) and minus `sm_version`: the modules target the
/// KIR floor and the driver JIT-compiles them forward.
#[derive(Debug, Clone)]
pub struct SpecSamplerConfig {
    pub d_model: u32,
    pub vocab_size: u32,
    pub vocab_tile: u32,
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
/// Verify only: pass 1's (max, sum), published by thread 0.
const R_MAX: u32 = 3;
const R_SUM: u32 = 4;

/// `[hidden: d_model][scores: TILE][rstd: 1]`, plus `[max: 1][sum: 1]` for
/// the verify writer; all f32, 4-aligned and a multiple of 4 long, so the
/// offsets are the hand kernels' packed ones.
fn smem_layout(d_model: u32, verify: bool) -> SmemLayout {
    let f32_region = |name: &str, elems: u32| SmemRegion {
        name: name.to_string(),
        bytes: elems * 4,
        align: 4,
        elem: KirType::F32,
    };
    let mut regions = vec![
        f32_region("hidden", d_model),
        f32_region("scores", TILE),
        f32_region("rstd", 1),
    ];
    if verify {
        regions.push(f32_region("max", 1));
        regions.push(f32_region("sum", 1));
    }
    SmemLayout { regions, dynamic: false }
}

// ---------------------------------------------------------------------------
// Builder sections, shared by both kernels
// ---------------------------------------------------------------------------

/// Values every section reads, made once in the entry block.
struct Common {
    tid: VarId,
    hidden: VarId,
    norm_w: VarId,
    lm_head: VarId,
    hidden_smem: VarId,
    scores: VarId,
    rstd_smem: VarId,
    zero: VarId,
    one: VarId,
    tile_width: VarId,
    d_model: VarId,
    /// The same value as `d_model`, for the `1 / d_model` immediate.
    d_model_value: u32,
    d_model_wide: VarId,
    vocab: VarId,
    f_zero: VarId,
    f_neg_inf: VarId,
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
    use AddressSpace::{Global, Shared};
    use KirType::F32;

    // 1. cooperative strided load: hidden [1, d_model] f32 -> SMEM
    strided_loop(b, c, |b, i| {
        let src = at(b, F32, Global, c.hidden, i);
        let h = load(b, F32, src, Global);
        let dst = at(b, F32, Shared, c.hidden_smem, i);
        b.emit(KirOp::Store(dst, h, Shared));
    });
    b.emit(KirOp::Barrier);

    // 2. RMSNorm: per-thread strided partial sum of squares.
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(head, KirType::U32);
    let ss = b.add_block_param(head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c.tid, c.f_zero])));

    b.set_block(head);
    let finished = cmp(b, i, c.d_model, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let slot = at(b, F32, Shared, c.hidden_smem, i);
    let h = load(b, F32, slot, Shared);
    let ss_next = b.new_typed_var(F32);
    b.emit(KirOp::Fma(ss_next, h, h, ss));
    let i_next = op2(b, KirType::U32, KirOp::Add, i, c.tile_width);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![i_next, ss_next])));

    // The scores region doubles as the reduction scratch before the tile
    // loop.
    b.set_block(done);
    let mine = at(b, F32, Shared, c.scores, c.tid);
    b.emit(KirOp::Store(mine, ss, Shared));
    b.emit(KirOp::Barrier);

    // Tree reduction: scores[tid] += scores[tid + off] for tid < off.
    for off in [64u32, 32, 16, 8, 4, 2, 1] {
        let add = b.new_block();
        let join = b.new_block();
        let offset = konst(b, ConstValue::U32(off));
        let active = cmp(b, c.tid, offset, CmpOp::Lt);
        b.terminate(KirTerminator::CondBranch(active, KirEdge::to(add), KirEdge::to(join)));

        b.set_block(add);
        let lo_addr = at(b, F32, Shared, c.scores, c.tid);
        let lo = load(b, F32, lo_addr, Shared);
        let partner = op2(b, KirType::U32, KirOp::Add, c.tid, offset);
        let hi_addr = at(b, F32, Shared, c.scores, partner);
        let hi = load(b, F32, hi_addr, Shared);
        let sum = op2(b, F32, KirOp::Add, lo, hi);
        b.emit(KirOp::Store(lo_addr, sum, Shared));
        b.terminate(KirTerminator::Branch(KirEdge::to(join)));

        b.set_block(join);
        b.emit(KirOp::Barrier);
    }

    // Thread 0: rstd = rsqrt(sum_sq / d_model + eps).
    let rstd_compute = b.new_block();
    let rstd_done = b.new_block();
    let is_zero = cmp(b, c.tid, c.zero, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(is_zero, KirEdge::to(rstd_compute), KirEdge::to(rstd_done)));

    b.set_block(rstd_compute);
    let total = load(b, F32, c.scores, Shared);
    let inv_dm = konst(b, ConstValue::F32(1.0 / c.d_model_value as f32));
    let mean = op2(b, F32, KirOp::Mul, total, inv_dm);
    let eps = konst(b, ConstValue::F32(RMS_EPS));
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

/// `dot(hidden_smem, lm_head[tok])` over the f16 row, in the hand
/// kernels' fma order: `dot = fma(w, h, dot)` for `d = 0..d_model`.
/// Entered from the current block; the builder is left in the loop's exit
/// block, and the returned value is the finished dot.
fn build_row_dot(b: &mut KirBuilder, c: &Common, tok: VarId) -> VarId {
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
    max: VarId,
    sum: VarId,
    /// Draft only.
    argmax: Option<VarId>,
}

/// Step 3 (both kernels) / pass 1 (verify): per tile every thread scores
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
    let (max, sum, argmax) = tile_state;
    PassResult { max, sum, argmax }
}

/// The builder, the params both kernels share (hidden, norm_w, lm_head)
/// already added in FFI order, and the entry block started.
fn begin(name: &str, cfg: &SpecSamplerConfig, verify: bool) -> (KirBuilder, VarId, VarId, VarId) {
    use AddressSpace::Global;
    use KirType::{F16, F32};

    let mut b = KirBuilder::new(name);
    let hidden = b.add_param("hidden_ptr", ptr(F32, Global), Global);
    let norm_w = b.add_param("norm_w_ptr", ptr(F32, Global), Global);
    let lm_head = b.add_param("lm_head_ptr", ptr(F16, Global), Global);
    b.set_smem_layout(smem_layout(cfg.d_model, verify));
    b.set_workgroup_size([BLOCK_DIM, 1, 1]);
    (b, hidden, norm_w, lm_head)
}

/// The entry block's shared values. Call with the builder in the entry
/// block, after every param is added.
fn common(b: &mut KirBuilder, cfg: &SpecSamplerConfig, hidden: VarId, norm_w: VarId, lm_head: VarId) -> Common {
    let tid = b.new_typed_var(KirType::U32);
    b.emit(KirOp::ThreadId(tid, 0));
    Common {
        tid,
        hidden,
        norm_w,
        lm_head,
        hidden_smem: shared_region(b, R_HIDDEN),
        scores: shared_region(b, R_SCORES),
        rstd_smem: shared_region(b, R_RSTD),
        zero: konst(b, ConstValue::U32(0)),
        one: konst(b, ConstValue::U32(1)),
        tile_width: konst(b, ConstValue::U32(TILE)),
        d_model: konst(b, ConstValue::U32(cfg.d_model)),
        d_model_value: cfg.d_model,
        d_model_wide: konst(b, ConstValue::U64(cfg.d_model as u64)),
        vocab: konst(b, ConstValue::U32(cfg.vocab_size)),
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
    let (mut b, hidden, norm_w, lm_head) = begin(DRAFT_SAMPLE_KERNEL_NAME, cfg, false);
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
// Kind 8: nsl_cfie_verify_probs
// ---------------------------------------------------------------------------

/// The target prob-row writer for `cfg`, as KIR.
pub fn build_verify_probs(cfg: &SpecSamplerConfig) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32};

    validate_config(cfg);
    let (mut b, hidden, norm_w, lm_head) = begin(VERIFY_PROBS_KERNEL_NAME, cfg, true);
    let out = b.add_param("out_probs_ptr", ptr(F32, Global), Global);

    let entry = b.new_block();
    b.set_block(entry);
    let c = common(&mut b, cfg, hidden, norm_w, lm_head);
    let max_smem = shared_region(&mut b, R_MAX);
    let sum_smem = shared_region(&mut b, R_SUM);

    hidden_load_and_rmsnorm(&mut b, &c);
    // Pass 1: identical merge order to nsl_cfie_draft_sample — the
    // self-speculation anchor relies on bit-identical (max, sum).
    let pass = streaming_pass(&mut b, &c, false);

    // Thread 0 publishes (max, sum); every thread reloads them.
    let publish = b.new_block();
    let published = b.new_block();
    let is_zero = cmp(&mut b, c.tid, c.zero, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(is_zero, KirEdge::to(publish), KirEdge::to(published)));

    b.set_block(publish);
    b.emit(KirOp::Store(max_smem, pass.max, Shared));
    b.emit(KirOp::Store(sum_smem, pass.sum, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(published)));

    b.set_block(published);
    b.emit(KirOp::Barrier);
    let m = load(&mut b, F32, max_smem, Shared);
    let sum = load(&mut b, F32, sum_smem, Shared);

    // Pass 2: recompute the matvec per tile and store the row. No SMEM
    // writes, so no barriers inside the loop.
    let tile_head = b.new_block();
    let tile_body = b.new_block();
    let dot_start = b.new_block();
    let next = b.new_block();
    let exit = b.new_block();
    let tile = b.add_block_param(tile_head, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, vec![c.zero])));

    b.set_block(tile_head);
    let finished = cmp(&mut b, tile, c.vocab, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(exit), KirEdge::to(tile_body)));

    // Tail-tile guard: lanes past vocab store nothing.
    b.set_block(tile_body);
    let tok = op2(&mut b, U32, KirOp::Add, tile, c.tid);
    let past = cmp(&mut b, tok, c.vocab, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(next), KirEdge::to(dot_start)));

    // p = exp(x - max) / sum; div.rn matches the CPU '/'.
    b.set_block(dot_start);
    let dot = build_row_dot(&mut b, &c, tok);
    let shifted = op2(&mut b, F32, KirOp::Sub, dot, m);
    let e = b.new_typed_var(F32);
    b.emit(KirOp::Exp(e, shifted));
    let p = op2(&mut b, F32, KirOp::Div, e, sum);
    let dst = at(&mut b, F32, Global, out, tok);
    b.emit(KirOp::Store(dst, p, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(next)));

    b.set_block(next);
    let tile_next = op2(&mut b, U32, KirOp::Add, tile, c.tile_width);
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, vec![tile_next])));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

// ---------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------

/// The `//` header the hand kernels carried, line for line.
fn header_comment(cfg: &SpecSamplerConfig, verify: bool) -> String {
    let eps = f32_imm(RMS_EPS);
    let mut w = String::new();
    writeln!(w, "//").unwrap();
    if verify {
        writeln!(w, "// {} - CFIE target prob-row writer (Cycle 13, G15).", VERIFY_PROBS_KERNEL_NAME).unwrap();
        writeln!(w, "// One CTA, {} threads; TWO passes over the vocab tiles:", BLOCK_DIM).unwrap();
        writeln!(w, "//   pass 1: online max + exp-sum (recomputing the matvec per tile),").unwrap();
        writeln!(w, "//   pass 2: recompute the matvec, store p_i = exp(x_i - max)/sum.").unwrap();
        writeln!(w, "// This kernel EXISTS to materialize softmaxed f32 rows for the").unwrap();
        writeln!(w, "// rejection kernel (nsl_cfie_spec_reject) - the fused sampler never").unwrap();
        writeln!(w, "// writes probs by design.  The 2x matvec is the price and is").unwrap();
        writeln!(w, "// bounded by K <= 32 verify positions per round.").unwrap();
        writeln!(w, "// LM-head layout: f16 [vocab, d_model], ROW-major per vocab row.").unwrap();
    } else {
        writeln!(w, "// {} - CFIE draft-model greedy sampler (Cycle 13, G15).", DRAFT_SAMPLE_KERNEL_NAME).unwrap();
        writeln!(w, "// One CTA, {} threads; ONE streaming pass over the vocab tiles keeps", BLOCK_DIM).unwrap();
        writeln!(w, "// a running max + argmax + online sum of exp(x - max) with rescale").unwrap();
        writeln!(w, "// (flash softmax).  p(argmax) = 1/sum_final because").unwrap();
        writeln!(w, "// exp(x_argmax - max) == 1 when the argmax attains the max.").unwrap();
        writeln!(w, "// Outputs: token id (u32) + p(argmax) (f32) - 8 bytes to HBM.").unwrap();
        writeln!(w, "// LM-head layout: f16 [vocab, d_model], ROW-major per vocab row.").unwrap();
        writeln!(w, "// rng_seed is ACCEPTED for ABI symmetry with the fused sampler but").unwrap();
        writeln!(w, "// UNUSED: v1 drafting is greedy (the paper's temperature 0.0).").unwrap();
    }
    writeln!(w, "// Baked constants:").unwrap();
    writeln!(w, "//   d_model    = {}", cfg.d_model).unwrap();
    writeln!(w, "//   vocab_size = {}", cfg.vocab_size).unwrap();
    writeln!(w, "//   vocab_tile = {}", TILE).unwrap();
    writeln!(w, "//   rms_eps    = {} ({})", RMS_EPS, eps).unwrap();
    writeln!(w, "//").unwrap();
    w
}

/// Verify, lower and prefix the header. The returned text carries no NUL:
/// the serve path appends the one the driver wants when it embeds the
/// module.
fn emit(ir: &KernelIR, cfg: &SpecSamplerConfig, verify: bool) -> (String, SpecSamplerMeta) {
    if let Err(errors) = crate::kir_verify::verify(ir) {
        panic!("{} failed KIR verification: {errors:?}", ir.name);
    }
    let smem_bytes = ir.smem_layout.total_bytes().expect("a verified layout has a size");
    let module = lower_kir_to_ptx(ir);
    let module = module.strip_suffix(&[0]).unwrap_or(&module);
    let module = std::str::from_utf8(module).expect("the KIR printer emits ASCII");

    let mut p = header_comment(cfg, verify);
    p.push_str(module);
    let meta = SpecSamplerMeta { kernel_name: ir.name.clone(), smem_bytes, block_dim: BLOCK_DIM };
    (p, meta)
}

/// Emit the draft greedy sampler kernel for `cfg`.
pub fn emit_draft_sample(cfg: &SpecSamplerConfig) -> (String, SpecSamplerMeta) {
    emit(&build_draft_sample(cfg), cfg, false)
}

/// PTX-only convenience wrapper around [`emit_draft_sample`].
pub fn emit_draft_sample_ptx(cfg: &SpecSamplerConfig) -> String {
    emit_draft_sample(cfg).0
}

/// Emit the target prob-row writer kernel for `cfg`.
pub fn emit_verify_probs(cfg: &SpecSamplerConfig) -> (String, SpecSamplerMeta) {
    emit(&build_verify_probs(cfg), cfg, true)
}

/// PTX-only convenience wrapper around [`emit_verify_probs`].
pub fn emit_verify_probs_ptx(cfg: &SpecSamplerConfig) -> String {
    emit_verify_probs(cfg).0
}


// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

/// Kernel-order RMSNorm: per-thread strided partial sums, the SMEM
/// tree reduction, gamma scale - identical staging to the kernels
/// (and to `cfie_sample_ptx::cpu_reference`).  The kernel's
/// `rsqrt.approx` is exact libm here; token parity is unaffected (the
/// argmax compares dots computed from the same scaled hidden).
fn rms_norm_kernel_order(hidden: &[f32], norm_w: &[f32]) -> Vec<f32> {
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
    let rstd = 1.0 / (mean + RMS_EPS).sqrt();
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
    let h = rms_norm_kernel_order(hidden, norm_w);
    let st = online_pass(&h, lm_head_f32, vocab);
    (st.sel, 1.0 / st.sum)
}

/// CPU mirror of `nsl_cfie_verify_probs`: returns the softmaxed prob
/// row `[vocab]` the rejection kernel consumes.  Pass 2 recomputes
/// the matvec per token exactly like the kernel does (the price of
/// never staging the logits row).
pub fn cpu_reference_verify_probs(
    cfg: &SpecSamplerConfig,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
) -> Vec<f32> {
    let (_, vocab) = validate_cpu_inputs(cfg, hidden, norm_w, lm_head_f32);
    let log2e = std::f32::consts::LOG2_E;
    let h = rms_norm_kernel_order(hidden, norm_w);
    let st = online_pass(&h, lm_head_f32, vocab);
    (0..vocab)
        .map(|tok| {
            let dot = row_dot(&h, lm_head_f32, tok);
            ((dot - st.m) * log2e).exp2() / st.sum
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
    use crate::cfie_speculative_ptx::{cpu_reference_reject, RejectionConfig};

    fn cfg(d_model: u32, vocab_size: u32) -> SpecSamplerConfig {
        SpecSamplerConfig {
            d_model,
            vocab_size,
            vocab_tile: 128,
        }
    }

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
    fn verify_param_list_is_exactly_the_four_params() {
        let ir = build_verify_probs(&paper_cfg());
        assert_eq!(param_names(&ir), ["hidden_ptr", "norm_w_ptr", "lm_head_ptr", "out_probs_ptr"]);
        let ptx = emit_verify_probs_ptx(&paper_cfg());
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
    fn verify_single_global_store_is_the_prob_row() {
        // One global store in the pass-2 loop body - executed once per
        // vocab entry, the row the reject kernel consumes.
        let ir = build_verify_probs(&paper_cfg());
        let stores = global_stores(&ir);
        assert_eq!(stores.len(), 1);
        assert_eq!(ir.var_types[&stores[0].1], KirType::F32);
        // Normalization is an f32 Div (div.rn) so the CPU '/' mirrors it
        // exactly.
        assert_eq!(count(&ir, |op| matches!(op, KirOp::Div(..))), 1);
        assert!(emit_verify_probs_ptx(&paper_cfg()).contains("div.rn.f32"));
    }

    #[test]
    fn no_mad_lo_ascii_only_and_line_width_both_kernels() {
        for ptx in [
            emit_draft_sample_ptx(&paper_cfg()),
            emit_verify_probs_ptx(&paper_cfg()),
        ] {
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
    fn rmsnorm_and_flash_softmax_sections_present_in_both() {
        for (ir, exps) in [
            // Rescale + accumulate in the merge.
            (build_draft_sample(&paper_cfg()), 2),
            // The same two in pass 1, plus pass 2's p = exp(x - max).
            (build_verify_probs(&paper_cfg()), 3),
        ] {
            assert_eq!(count(&ir, |op| matches!(op, KirOp::Rsqrt(..))), 1, "{}", ir.name);
            assert_eq!(count(&ir, |op| matches!(op, KirOp::Exp(..))), exps, "{}", ir.name);
            // One cooperative strided load, one norm scale, one score store
            // per tile; the 7-step tree reduction.
            let barriers = count(&ir, |op| matches!(op, KirOp::Barrier));
            assert!(barriers >= 13, "{}: {barriers} barriers", ir.name);
        }
        for ptx in [
            emit_draft_sample_ptx(&paper_cfg()),
            emit_verify_probs_ptx(&paper_cfg()),
        ] {
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
        for ptx in [
            emit_draft_sample_ptx(&paper_cfg()),
            emit_verify_probs_ptx(&paper_cfg()),
        ] {
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

        let (ptx, meta) = emit_verify_probs(&paper_cfg());
        assert_eq!(meta.kernel_name, VERIFY_PROBS_KERNEL_NAME);
        assert_eq!(meta.block_dim, 128);
        // hidden + scores + rstd + max + sum.
        assert_eq!(meta.smem_bytes, 512 * 4 + 128 * 4 + 12);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 shared_mem[{}];", meta.smem_bytes)));
    }

    #[test]
    fn registration_kinds_match_the_cycle13_abi() {
        assert_eq!(DRAFT_SAMPLE_KERNEL_KIND, 7);
        assert_eq!(VERIFY_PROBS_KERNEL_KIND, 8);
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
    #[should_panic(expected = "d_model")]
    fn verify_d_model_over_8192_panics() {
        let mut c = paper_cfg();
        c.d_model = 8193;
        let _ = emit_verify_probs(&c);
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

        // The verify row agrees on the same fixture (same online pass).
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert!((row[tok as usize] as f64 - p64).abs() < 1e-4);
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row must sum to ~1, got {total}");
    }

    #[test]
    fn cpu_verify_probs_row_matches_f64_softmax_and_normalizes() {
        let (dm, vocab) = (32usize, 300usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(row.len(), vocab);
        assert!(row.iter().all(|&p| p >= 0.0));
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row must sum to ~1, got {total}");

        // Row argmax == the draft sampler's token (same weights).
        let (tok, _) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        let row_argmax = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &p)| {
                if p > acc.1 {
                    (i, p)
                } else {
                    acc
                }
            })
            .0;
        assert_eq!(row_argmax, tok as usize);

        // Per-element agreement with the independent f64 softmax.
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
        for (i, &p) in row.iter().enumerate() {
            let p64 = (logits[i] - max).exp() / sum;
            assert!(
                (p as f64 - p64).abs() < 1e-5,
                "row[{i}] = {p} vs f64 {p64}"
            );
        }
    }

    #[test]
    fn cpu_draft_prob_bitwise_equals_verify_row_at_token() {
        // The lossless self-speculation anchor: with draft == target
        // weights, p_draft == p_target[tok] BIT-FOR-BIT, so the reject
        // ratio is exactly 1.0 and every draft token accepts.
        let (dm, vocab) = (16usize, 200usize);
        let c = cfg(dm as u32, vocab as u32);
        let hidden = hidden_pattern(dm);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let (tok, prob) = cpu_reference_draft_sample(&c, &hidden, &gamma, &w);
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(
            prob.to_bits(),
            row[tok as usize].to_bits(),
            "p_draft must equal p_target[argmax] bitwise (draft == target)"
        );
    }

    #[test]
    fn verify_rows_feed_cpu_reference_reject_all_accept_self_speculation() {
        // Frozen-ABI cross-check: the rows this module's reference
        // produces are exactly what cfie_speculative_ptx's reject
        // reference consumes.  Self-speculation (draft == target)
        // must accept all K for EVERY seed.
        let (dm, vocab) = (8usize, 32usize);
        let c = cfg(dm as u32, vocab as u32);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let hiddens = [hidden_pattern(dm), (0..dm).map(|i| (i as f32 * 0.61).cos()).collect()];

        let mut target_probs = Vec::new();
        let mut draft_probs = Vec::new();
        let mut draft_tokens = Vec::new();
        for h in &hiddens {
            let (tok, p) = cpu_reference_draft_sample(&c, h, &gamma, &w);
            let row = cpu_reference_verify_probs(&c, h, &gamma, &w);
            assert_eq!(p.to_bits(), row[tok as usize].to_bits());
            target_probs.extend_from_slice(&row);
            draft_probs.push(p);
            draft_tokens.push(tok);
        }

        let rcfg = RejectionConfig {
            k_tokens: 2,
            vocab_size: vocab as u32,
        };
        for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
            let (acc, corr) =
                cpu_reference_reject(&rcfg, &target_probs, &draft_probs, &draft_tokens, seed);
            assert_eq!(acc, 2, "seed {seed}: self-speculation must accept all K");
            assert_eq!(corr, u32::MAX, "seed {seed}: all-accept sentinel expected");
        }
    }

    #[test]
    fn verify_rows_feed_cpu_reference_reject_rejection_path() {
        // Inflated draft probs force ratio < 1: the reject walk must
        // consume this module's rows without tripping its layout
        // asserts, and the Leviathan residual (p_target - p_draft on
        // the drafted token, clamped) must never resample that token.
        let (dm, vocab) = (8usize, 32usize);
        let c = cfg(dm as u32, vocab as u32);
        let gamma = gamma_pattern(dm);
        let w = lm_head(vocab, dm);
        let hiddens = [hidden_pattern(dm), (0..dm).map(|i| (i as f32 * 0.61).cos()).collect()];

        let mut target_probs = Vec::new();
        let mut draft_tokens = Vec::new();
        for h in &hiddens {
            let (tok, _) = cpu_reference_draft_sample(&c, h, &gamma, &w);
            target_probs.extend_from_slice(&cpu_reference_verify_probs(&c, h, &gamma, &w));
            draft_tokens.push(tok);
        }
        let draft_probs = [1.0f32, 1.0];
        assert!(
            target_probs[draft_tokens[0] as usize] < 1.0,
            "test premise: ratio at position 0 must be < 1"
        );

        let rcfg = RejectionConfig {
            k_tokens: 2,
            vocab_size: vocab as u32,
        };
        let mut saw_rejection = false;
        for seed in 0..64u64 {
            let (acc, corr) =
                cpu_reference_reject(&rcfg, &target_probs, &draft_probs, &draft_tokens, seed);
            assert!((0..=2).contains(&acc), "seed {seed}");
            if acc < 2 {
                saw_rejection = true;
                assert!(corr < vocab as u32, "seed {seed}: correction in vocab");
                assert_ne!(
                    corr, draft_tokens[acc as usize],
                    "seed {seed}: residual zeroes the drafted token"
                );
            }
        }
        assert!(saw_rejection, "ratio < 1 must reject for some seed");
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
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(prob.to_bits(), row[129].to_bits());
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
        let row = cpu_reference_verify_probs(&c, &hidden, &gamma, &w);
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].to_bits(), 1.0f32.to_bits());
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

    #[test]
    fn ptxas_validates_verify_probs_paper_config() {
        let ptx = emit_verify_probs_ptx(&paper_cfg());
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                nsl_log::nsl_log!(INFO, "skip", "[skip] cfie verify-probs ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie verify-probs PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }
}
