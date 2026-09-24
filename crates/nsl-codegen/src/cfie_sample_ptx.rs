//! CFIE Feature 2: fused decode-sample PTX emitter.
//!
//! The paper's decode-tail claim: the six-launch pipeline (RMSNorm,
//! LM-head matmul, softmax, top-k, top-p, multinomial) becomes ONE
//! kernel where the `[1, vocab]` logits tensor never touches HBM.
//! Only the sampled token id (4 bytes) is written back.
//!
//! Consumes the structured [`FusedSampleProgram`] built by
//! `cfie_fused_sample::emit_program` — the ops actually present drive
//! which sections are emitted (RmsNorm, Argmax vs SoftmaxTopK /
//! NucleusFilter / MultinomialSample).
//!
//! Launch shape: single CTA (grid = 1), 128 threads — the batch=1
//! latency path.  Algorithm:
//!   1. cooperative load of hidden `[1, d_model]` (f32) into SMEM;
//!   2. RMSNorm in SMEM when the program has the op (parallel SMEM
//!      reduction of the sum of squares, rsqrt(mean + eps), gamma);
//!   3. tile loop over vocab in chunks of 128: thread `t` owns row
//!      `tile_base + t`, computes dot(x, W[row]) with f16 loads +
//!      f32 accumulate, scales by the baked 1/temperature, applies
//!      the grammar bitmask hook when compiled in, stores to SMEM;
//!   4. thread 0 merges the tile into a k-entry candidate list in
//!      SMEM (replace-min insertion, serial — correctness first);
//!   5. thread 0: softmax over the k candidates, optional nucleus
//!      filter (insertion sort desc + cumulative cutoff at the baked
//!      top_p), multinomial via xorshift64* seeded from `rng_seed`.
//!      Greedy programs argmax the candidate list directly.
//!   6. the ONLY global store of the kernel writes the token id.
//!
//! Determinism: the PRNG is xorshift64* over the u64 seed param —
//! the sampled token is a pure function of (weights, hidden, seed),
//! which keeps the kernel M46-friendly (no curand state, no clock).
//!
//! `cpu_reference` mirrors the kernel's arithmetic order (same
//! strided partial sums + tree reduction, same fma dot order, same
//! replace-min/sort/walk tie-breaks, same xorshift64*).  The kernel
//! uses `rsqrt.approx` / `ex2.approx` where the CPU uses exact libm;
//! exact GPU parity is verified in a later GPU cycle.
//!
//! ## KIR (roadmap A2 step 9)
//!
//! The kernel is built as [`KernelIR`] ([`build`]) and lowered by
//! `nsl_kir`'s printer. It shares its first sections with
//! `cfie_spec_sampler_ptx` (the hidden-row load, the in-place RMSNorm and
//! the f16 row dot, through [`Common`]) and its xorshift64* draw with
//! `cfie_speculative_ptx`. `tests/cfie_sample_kir_equivalence.rs` runs it
//! against the frozen hand emitter on the cooperative-CTA interpreter and
//! requires the same output bits, and `cpu_reference`'s token. It targets
//! the KIR floor (`sm_70`), so [`FusedSampleKernelConfig`] has no
//! `sm_version`.

use crate::backend_ptx::lower_kir_to_ptx;
use crate::cfie_decode_attention::{at, cmp, konst, load, op2, ptr};
use crate::cfie_fused_sample::{FusedSampleOp, FusedSampleProgram};
use crate::cfie_spec_sampler_ptx::Common;
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType,
    SmemLayout, SmemRegion, VarId,
};
use std::fmt::Write;

/// Threads per CTA == vocab tile width (thread t owns row tile_base+t).
const TILE: u32 = 128;
const BLOCK_DIM: u32 = TILE;

/// Baked RMSNorm epsilon (paper stage 1).
const RMS_EPS: f32 = 1e-5;

pub const KERNEL_NAME: &str = "nsl_cfie_fused_sample";

pub fn kernel_name() -> &'static str {
    KERNEL_NAME
}

/// Compile-time configuration for the fused sampler kernel.
#[derive(Debug, Clone)]
pub struct FusedSampleKernelConfig {
    pub d_model: u32,
    pub vocab_size: u32,
    pub vocab_tile: u32,
    pub top_k: u32,
    /// Number of grammar DFA states; 0 = no grammar hook emitted.
    pub grammar_states: u32,
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct FusedSampleMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
}

fn f32_imm(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

fn has_op(program: &FusedSampleProgram, pred: impl Fn(&FusedSampleOp) -> bool) -> bool {
    program.ops.iter().any(pred)
}

/// Baked 1/temperature — taken from the program's MatmulTile epilogue.
fn temperature_recip(program: &FusedSampleProgram) -> f32 {
    program
        .ops
        .iter()
        .find_map(|op| match op {
            FusedSampleOp::MatmulTile {
                temperature_recip, ..
            } => Some(*temperature_recip),
            _ => None,
        })
        .expect("FusedSampleProgram has no MatmulTile op")
}

fn nucleus_top_p(program: &FusedSampleProgram) -> Option<f32> {
    program.ops.iter().find_map(|op| match op {
        FusedSampleOp::NucleusFilter { top_p } => Some(*top_p),
        _ => None,
    })
}

/// Emit the min-scan over the k-entry candidate list: `(min, pos)` with
/// strict `<`, first min wins (the CPU reference mirrors the tie-break).
/// Entered from the current block; the builder is left in the loop's exit
/// block, where the returned pair is the scan's result.
fn build_min_scan(b: &mut KirBuilder, s: &Sampler) -> (VarId, VarId) {
    use AddressSpace::Shared;
    use KirType::{F32, U32};

    let v0 = load(b, F32, s.topk_val, Shared);
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let j = b.add_block_param(head, U32);
    let min = b.add_block_param(head, F32);
    let pos = b.add_block_param(head, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![s.c.one, v0, s.c.zero])));

    b.set_block(head);
    let finished = cmp(b, j, s.k, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let addr = at(b, F32, Shared, s.topk_val, j);
    let v = load(b, F32, addr, Shared);
    let lower = cmp(b, v, min, CmpOp::Lt);
    let min_next = select(b, F32, lower, v, min);
    let pos_next = select(b, U32, lower, j, pos);
    let j_next = op2(b, U32, KirOp::Add, j, s.c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![j_next, min_next, pos_next])));

    b.set_block(done);
    (min, pos)
}

fn select(b: &mut KirBuilder, ty: KirType, cond: VarId, yes: VarId, no: VarId) -> VarId {
    let dst = b.new_typed_var(ty);
    b.emit(KirOp::Select(dst, cond, yes, no));
    dst
}

/// `for (j = from; j < k; j++) body(j)`, entered from the current block;
/// the builder is left in the loop's exit block.
fn k_loop(b: &mut KirBuilder, s: &Sampler, from: VarId, body: impl FnOnce(&mut KirBuilder, VarId)) {
    let head = b.new_block();
    let body_block = b.new_block();
    let done = b.new_block();
    let j = b.add_block_param(head, KirType::U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![from])));

    b.set_block(head);
    let finished = cmp(b, j, s.k, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body_block)));

    b.set_block(body_block);
    body(b, j);
    let next = op2(b, KirType::U32, KirOp::Add, j, s.c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![next])));

    b.set_block(done);
}

/// `sum over j < k of topk_val[j]`, in index order, from `0.0`. The loop
/// carries the sum; the builder is left in the loop's exit block.
fn build_k_sum(b: &mut KirBuilder, s: &Sampler, rewrite: Option<VarId>) -> VarId {
    use AddressSpace::Shared;
    use KirType::{F32, U32};

    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let j = b.add_block_param(head, U32);
    let sum = b.add_block_param(head, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![s.c.zero, s.c.f_zero])));

    b.set_block(head);
    let finished = cmp(b, j, s.k, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let addr = at(b, F32, Shared, s.topk_val, j);
    let v = load(b, F32, addr, Shared);
    // With `rewrite = Some(max)`, the entry becomes exp(v - max) in place
    // and that is what is summed: the softmax pass.
    let term = match rewrite {
        Some(max) => {
            let shifted = op2(b, F32, KirOp::Sub, v, max);
            let p = b.new_typed_var(F32);
            b.emit(KirOp::Exp(p, shifted));
            b.emit(KirOp::Store(addr, p, Shared));
            p
        }
        None => v,
    };
    let sum_next = op2(b, F32, KirOp::Add, sum, term);
    let j_next = op2(b, U32, KirOp::Add, j, s.c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![j_next, sum_next])));

    b.set_block(done);
    sum
}

/// The values the sampler's own sections read, beside the shared
/// [`Common`] ones.
struct Sampler {
    c: Common,
    topk_val: VarId,
    topk_idx: VarId,
    /// u32 `top_k`.
    k: VarId,
}

const R_TOPK_VAL: u32 = 2;
const R_TOPK_IDX: u32 = 3;
const R_RSTD: u32 = 4;

/// `[hidden: d_model][scores: TILE][topk_val: k][topk_idx: k u32][rstd: 1]`.
/// Every region is 4-aligned and a multiple of 4 long, so the offsets are
/// the hand kernel's packed ones.
fn smem_layout(d_model: u32, k: u32) -> SmemLayout {
    let region = |name: &str, elems: u32, elem: KirType| SmemRegion {
        name: name.to_string(),
        bytes: elems * 4,
        align: 4,
        elem,
    };
    SmemLayout {
        regions: vec![
            region("hidden", d_model, KirType::F32),
            region("scores", TILE, KirType::F32),
            region("topk_val", k, KirType::F32),
            region("topk_idx", k, KirType::U32),
            region("rstd", 1, KirType::F32),
        ],
        dynamic: false,
    }
}

fn validate(program: &FusedSampleProgram, cfg: &FusedSampleKernelConfig) {
    assert_eq!(
        cfg.vocab_tile, TILE,
        "vocab_tile must be {} so thread t owns row tile_base+t",
        TILE
    );
    assert!(
        cfg.top_k >= 1 && cfg.top_k <= 64,
        "top_k must be in 1..=64 (serial candidate list in SMEM)"
    );
    assert!(
        cfg.d_model >= 1 && cfg.d_model <= 8192,
        "d_model must be in 1..=8192 (hidden state staged in static SMEM)"
    );
    assert!(cfg.vocab_size >= 1, "vocab_size must be >= 1");
    assert_eq!(
        program.shape.d_model, cfg.d_model,
        "program shape d_model mismatch with cfg"
    );
    assert_eq!(
        program.shape.vocab_size, cfg.vocab_size,
        "program shape vocab_size mismatch with cfg"
    );
    assert_eq!(
        program.shape.vocab_tile, cfg.vocab_tile,
        "program shape vocab_tile mismatch with cfg"
    );
    assert_eq!(
        program.params.top_k, cfg.top_k,
        "program params top_k mismatch with cfg"
    );
}

/// Build the fused decode-sample kernel for `program` under `cfg` as KIR.
///
/// The sections, in order (each one's barriers are the hand kernel's):
///
/// ```text
/// hidden_load          hidden row -> SMEM                      (shared with
/// rmsnorm_in_place     only with the RmsNorm op                 cfie_spec_sampler_ptx)
/// candidate init       topk_val[t] = -inf, topk_idx[t] = 0 for t < k; bar
/// tile loop            thread t: s = dot(h, W[tile+t]) * (1/T), or -inf past
///                      the vocab; the grammar hook turns s to -inf where the
///                      state's mask bit is clear; scores[t] = s; bar;
///                      thread 0: replace-min merge of the tile's scores into
///                      the candidate list (strict >, min re-scanned after
///                      each replacement); bar
/// thread 0 selection   greedy: argmax (strict >, first wins)
///                      sampling: max; softmax in place; [nucleus: insertion
///                      sort desc, cumulative cutoff at top_p, tail zeroed];
///                      kept mass; xorshift64* draw (cfie_speculative_ptx's
///                      build_prng_draw); multinomial CDF walk
/// store                out_token = sel, the kernel's only global store
/// ```
pub fn build(program: &FusedSampleProgram, cfg: &FusedSampleKernelConfig) -> KernelIR {
    use crate::cfie_spec_sampler_ptx::{build_row_dot, common_at, hidden_load, rmsnorm_in_place};
    use AddressSpace::{Global, Shared};
    use KirType::{F16, F32, I8, U32, U64};

    validate(program, cfg);
    let has_rms = has_op(program, |op| matches!(op, FusedSampleOp::RmsNorm));
    let greedy = has_op(program, |op| matches!(op, FusedSampleOp::Argmax));
    let top_p = nucleus_top_p(program);
    let inv_temp = temperature_recip(program);
    let grammar_hook = cfg.grammar_states > 0;

    let mut b = KirBuilder::new(KERNEL_NAME);
    let hidden = b.add_param("hidden_ptr", ptr(F32, Global), Global);
    // Read only when the program has the RmsNorm op.
    let norm_w = b.add_param("norm_w_ptr", ptr(F32, Global), Global);
    let lm_head = b.add_param("lm_head_ptr", ptr(F16, Global), Global);
    let out_token = b.add_param("out_token_ptr", ptr(U32, Global), Global);
    let rng_seed = b.add_param("rng_seed", U64, Global);
    // Null when no grammar is live; read only through the hook.
    let grammar_mask = b.add_param("grammar_mask_ptr", ptr(I8, Global), Global);
    let grammar_state = b.add_param("grammar_state", U32, Global);
    b.set_smem_layout(smem_layout(cfg.d_model, cfg.top_k));
    b.set_workgroup_size([BLOCK_DIM, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let c = common_at(&mut b, cfg.d_model, cfg.vocab_size, [hidden, norm_w, lm_head], R_RSTD);
    let region = |b: &mut KirBuilder, r: u32, elem: KirType| {
        let dst = b.new_typed_var(ptr(elem, Shared));
        b.emit(KirOp::SharedRegion { dst, region: r });
        dst
    };
    let topk_val = region(&mut b, R_TOPK_VAL, F32);
    let topk_idx = region(&mut b, R_TOPK_IDX, U32);
    let k = konst(&mut b, ConstValue::U32(cfg.top_k));
    let s = Sampler { c, topk_val, topk_idx, k };
    let c = &s.c;

    hidden_load(&mut b, c);
    if has_rms {
        rmsnorm_in_place(&mut b, c);
    }

    // Candidate list init: threads t < k.
    let init = b.new_block();
    let init_done = b.new_block();
    let inits = cmp(&mut b, c.tid, s.k, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(inits, KirEdge::to(init), KirEdge::to(init_done)));
    b.set_block(init);
    let val_slot = at(&mut b, F32, Shared, s.topk_val, c.tid);
    b.emit(KirOp::Store(val_slot, c.f_neg_inf, Shared));
    let idx_slot = at(&mut b, U32, Shared, s.topk_idx, c.tid);
    b.emit(KirOp::Store(idx_slot, c.zero, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(init_done)));
    b.set_block(init_done);
    b.emit(KirOp::Barrier);

    // The vocab tile loop.
    let tile_head = b.new_block();
    let tile_body = b.new_block();
    let score = b.new_block();
    let score_store = b.new_block();
    let tiles_done = b.new_block();
    let tile = b.add_block_param(tile_head, U32);
    let s_final = b.add_block_param(score_store, F32);
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, vec![c.zero])));

    b.set_block(tile_head);
    let all_tiles = cmp(&mut b, tile, c.vocab, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(all_tiles, KirEdge::to(tiles_done), KirEdge::to(tile_body)));

    // Tail-tile guard: lanes past the vocab keep -inf.
    b.set_block(tile_body);
    let tok = op2(&mut b, U32, KirOp::Add, tile, c.tid);
    let in_vocab = cmp(&mut b, tok, c.vocab, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(
        in_vocab,
        KirEdge::to(score),
        KirEdge::with(score_store, vec![c.f_neg_inf]),
    ));

    b.set_block(score);
    let dot = build_row_dot(&mut b, c, tok);
    let inv_t = konst(&mut b, ConstValue::F32(inv_temp));
    let scaled = op2(&mut b, F32, KirOp::Mul, dot, inv_t);
    if grammar_hook {
        // Bit (state, token) of the mask: clear -> -inf. A null mask
        // pointer means no grammar is live, and nothing is read.
        let hook = b.new_block();
        let mask_addr = b.new_typed_var(U64);
        b.emit(KirOp::Cast(mask_addr, grammar_mask, U64));
        let null = konst(&mut b, ConstValue::U64(0));
        let no_mask = cmp(&mut b, mask_addr, null, CmpOp::Eq);
        b.terminate(KirTerminator::CondBranch(
            no_mask,
            KirEdge::with(score_store, vec![scaled]),
            KirEdge::to(hook),
        ));

        b.set_block(hook);
        let row_bytes = konst(&mut b, ConstValue::U32(cfg.vocab_size.div_ceil(8)));
        let row = op2(&mut b, U32, KirOp::Mul, grammar_state, row_bytes);
        let three = konst(&mut b, ConstValue::U32(3));
        let byte_in_row = op2(&mut b, U32, KirOp::Shr, tok, three);
        let byte_index = op2(&mut b, U32, KirOp::Add, row, byte_in_row);
        let byte_addr = at(&mut b, I8, Global, grammar_mask, byte_index);
        let byte = load(&mut b, I8, byte_addr, Global);
        // Sign extension leaves the byte's own 8 bits as they were, and
        // only those are tested.
        let byte32 = b.new_typed_var(U32);
        b.emit(KirOp::Cast(byte32, byte, U32));
        let seven = konst(&mut b, ConstValue::U32(7));
        let bit_index = op2(&mut b, U32, KirOp::And, tok, seven);
        let shifted = op2(&mut b, U32, KirOp::Shr, byte32, bit_index);
        let bit = op2(&mut b, U32, KirOp::And, shifted, c.one);
        let allowed = cmp(&mut b, bit, c.zero, CmpOp::Ne);
        let gated = select(&mut b, F32, allowed, scaled, c.f_neg_inf);
        b.terminate(KirTerminator::Branch(KirEdge::with(score_store, vec![gated])));
    } else {
        b.terminate(KirTerminator::Branch(KirEdge::with(score_store, vec![scaled])));
    }

    b.set_block(score_store);
    let score_slot = at(&mut b, F32, Shared, c.scores, c.tid);
    b.emit(KirOp::Store(score_slot, s_final, Shared));
    b.emit(KirOp::Barrier);

    // Thread 0: replace-min merge of the tile's `cnt` scores.
    let merge_start = b.new_block();
    let merge_done = b.new_block();
    let not_zero = cmp(&mut b, c.tid, c.zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_zero, KirEdge::to(merge_done), KirEdge::to(merge_start)));

    b.set_block(merge_start);
    let remaining = op2(&mut b, U32, KirOp::Sub, c.vocab, tile);
    let cnt = op2(&mut b, U32, KirOp::Min, remaining, c.tile_width);
    let (min0, pos0) = build_min_scan(&mut b, &s);
    let merge_head = b.new_block();
    let merge_body = b.new_block();
    let replace = b.new_block();
    let merge_next = b.new_block();
    let i = b.add_block_param(merge_head, U32);
    let min = b.add_block_param(merge_head, F32);
    let pos = b.add_block_param(merge_head, U32);
    let min_n = b.add_block_param(merge_next, F32);
    let pos_n = b.add_block_param(merge_next, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(merge_head, vec![c.zero, min0, pos0])));

    b.set_block(merge_head);
    let merged = cmp(&mut b, i, cnt, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(merged, KirEdge::to(merge_done), KirEdge::to(merge_body)));

    b.set_block(merge_body);
    let cand_addr = at(&mut b, F32, Shared, c.scores, i);
    let cand = load(&mut b, F32, cand_addr, Shared);
    let beats = cmp(&mut b, cand, min, CmpOp::Gt);
    b.terminate(KirTerminator::CondBranch(
        beats,
        KirEdge::to(replace),
        KirEdge::with(merge_next, vec![min, pos]),
    ));

    b.set_block(replace);
    let val_at = at(&mut b, F32, Shared, s.topk_val, pos);
    b.emit(KirOp::Store(val_at, cand, Shared));
    let token = op2(&mut b, U32, KirOp::Add, tile, i);
    let idx_at = at(&mut b, U32, Shared, s.topk_idx, pos);
    b.emit(KirOp::Store(idx_at, token, Shared));
    let (min1, pos1) = build_min_scan(&mut b, &s);
    b.terminate(KirTerminator::Branch(KirEdge::with(merge_next, vec![min1, pos1])));

    b.set_block(merge_next);
    let i_next = op2(&mut b, U32, KirOp::Add, i, c.one);
    b.terminate(KirTerminator::Branch(KirEdge::with(merge_head, vec![i_next, min_n, pos_n])));

    // The scores region is rewritten next tile; sync before looping back.
    b.set_block(merge_done);
    b.emit(KirOp::Barrier);
    let tile_next = op2(&mut b, U32, KirOp::Add, tile, c.tile_width);
    b.terminate(KirTerminator::Branch(KirEdge::with(tile_head, vec![tile_next])));

    // Selection is serial on thread 0; the others exit.
    b.set_block(tiles_done);
    let select_blk = b.new_block();
    let store = b.new_block();
    let exit = b.new_block();
    let sel_final = b.add_block_param(store, U32);
    let not_zero = cmp(&mut b, c.tid, c.zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_zero, KirEdge::to(exit), KirEdge::to(select_blk)));
    b.set_block(select_blk);

    if greedy {
        // Argmax over the candidate list: strict >, first wins.
        let v0 = load(&mut b, F32, s.topk_val, Shared);
        let i0 = load(&mut b, U32, s.topk_idx, Shared);
        let head = b.new_block();
        let body = b.new_block();
        let j = b.add_block_param(head, U32);
        let max = b.add_block_param(head, F32);
        let sel = b.add_block_param(head, U32);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c.one, v0, i0])));

        b.set_block(head);
        let finished = cmp(&mut b, j, s.k, CmpOp::Ge);
        b.terminate(KirTerminator::CondBranch(
            finished,
            KirEdge::with(store, vec![sel]),
            KirEdge::to(body),
        ));

        b.set_block(body);
        let v_addr = at(&mut b, F32, Shared, s.topk_val, j);
        let v = load(&mut b, F32, v_addr, Shared);
        let idx_addr = at(&mut b, U32, Shared, s.topk_idx, j);
        let idx = load(&mut b, U32, idx_addr, Shared);
        let higher = cmp(&mut b, v, max, CmpOp::Gt);
        let max_next = select(&mut b, F32, higher, v, max);
        let sel_next = select(&mut b, U32, higher, idx, sel);
        let j_next = op2(&mut b, U32, KirOp::Add, j, c.one);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![j_next, max_next, sel_next])));
    } else {
        // Stable-softmax max over the k candidates.
        let v0 = load(&mut b, F32, s.topk_val, Shared);
        let head = b.new_block();
        let body = b.new_block();
        let done = b.new_block();
        let j = b.add_block_param(head, U32);
        let max = b.add_block_param(head, F32);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c.one, v0])));
        b.set_block(head);
        let finished = cmp(&mut b, j, s.k, CmpOp::Ge);
        b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));
        b.set_block(body);
        let v_addr = at(&mut b, F32, Shared, s.topk_val, j);
        let v = load(&mut b, F32, v_addr, Shared);
        let higher = cmp(&mut b, v, max, CmpOp::Gt);
        let max_next = select(&mut b, F32, higher, v, max);
        let j_next = op2(&mut b, U32, KirOp::Add, j, c.one);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![j_next, max_next])));
        b.set_block(done);

        // Softmax over k only (not vocab): p = exp(v - max), in place.
        let sum = build_k_sum(&mut b, &s, Some(max));

        if let Some(tp) = top_p {
            build_nucleus(&mut b, &s, sum, tp);
        }

        // Kept probability mass (== sum when no nucleus filter).
        let kept = build_k_sum(&mut b, &s, None);

        // xorshift64* over rng_seed; a zero seed would be a fixed point, so
        // it is replaced by the golden gamma.
        let null_seed = konst(&mut b, ConstValue::U64(0));
        let seed_zero = cmp(&mut b, rng_seed, null_seed, CmpOp::Eq);
        let golden = konst(&mut b, ConstValue::U64(0x9E37_79B9_7F4A_7C15));
        let x = select(&mut b, U64, seed_zero, golden, rng_seed);
        let (_state, r) = crate::cfie_speculative_ptx::build_prng_draw(&mut b, x);
        let target = op2(&mut b, F32, KirOp::Mul, r, kept);

        // Multinomial: walk the cumulative distribution. Zero-probability
        // entries are never selected; the last live entry is the fp-drift
        // fallback.
        let i0 = load(&mut b, U32, s.topk_idx, Shared);
        let head = b.new_block();
        let body = b.new_block();
        let live = b.new_block();
        let next = b.new_block();
        let j = b.add_block_param(head, U32);
        let cum = b.add_block_param(head, F32);
        let sel = b.add_block_param(head, U32);
        let sel_n = b.add_block_param(next, U32);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c.zero, c.f_zero, i0])));

        b.set_block(head);
        let finished = cmp(&mut b, j, s.k, CmpOp::Ge);
        b.terminate(KirTerminator::CondBranch(
            finished,
            KirEdge::with(store, vec![sel]),
            KirEdge::to(body),
        ));

        b.set_block(body);
        let p_addr = at(&mut b, F32, Shared, s.topk_val, j);
        let p = load(&mut b, F32, p_addr, Shared);
        let cum_next = op2(&mut b, F32, KirOp::Add, cum, p);
        let positive = cmp(&mut b, p, c.f_zero, CmpOp::Gt);
        b.terminate(KirTerminator::CondBranch(
            positive,
            KirEdge::to(live),
            KirEdge::with(next, vec![sel]),
        ));

        b.set_block(live);
        let idx_addr = at(&mut b, U32, Shared, s.topk_idx, j);
        let idx = load(&mut b, U32, idx_addr, Shared);
        let reached = cmp(&mut b, cum_next, target, CmpOp::Ge);
        b.terminate(KirTerminator::CondBranch(
            reached,
            KirEdge::with(store, vec![idx]),
            KirEdge::with(next, vec![idx]),
        ));

        b.set_block(next);
        let j_next = op2(&mut b, U32, KirOp::Add, j, c.one);
        b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![j_next, cum_next, sel_n])));
    }

    // The kernel's only global store.
    b.set_block(store);
    b.emit(KirOp::Store(out_token, sel_final, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// The nucleus filter over the softmaxed candidates: insertion sort
/// descending (stable, strict `<` shift), then the cumulative share
/// `p / sum` until it exceeds `top_p` (the crossing entry is kept) and the
/// tail zeroed. Leaves the builder in the filter's exit block.
fn build_nucleus(b: &mut KirBuilder, s: &Sampler, sum: VarId, top_p: f32) {
    use AddressSpace::Shared;
    use KirType::{F32, U32};
    let c = &s.c;

    // Insertion sort, j = 1..k.
    k_loop(b, s, c.one, |b, j| {
        let key_addr = at(b, F32, Shared, s.topk_val, j);
        let key = load(b, F32, key_addr, Shared);
        let kidx_addr = at(b, U32, Shared, s.topk_idx, j);
        let kidx = load(b, U32, kidx_addr, Shared);

        let inner = b.new_block();
        let compare = b.new_block();
        let shift = b.new_block();
        let place = b.new_block();
        let i = b.add_block_param(inner, U32);
        let at_i = b.add_block_param(place, U32);
        b.terminate(KirTerminator::Branch(KirEdge::with(inner, vec![j])));

        b.set_block(inner);
        let at_front = cmp(b, i, c.zero, CmpOp::Eq);
        b.terminate(KirTerminator::CondBranch(
            at_front,
            KirEdge::with(place, vec![i]),
            KirEdge::to(compare),
        ));

        b.set_block(compare);
        let prev_i = op2(b, U32, KirOp::Sub, i, c.one);
        let prev_addr = at(b, F32, Shared, s.topk_val, prev_i);
        let prev = load(b, F32, prev_addr, Shared);
        let smaller = cmp(b, prev, key, CmpOp::Lt);
        b.terminate(KirTerminator::CondBranch(
            smaller,
            KirEdge::to(shift),
            KirEdge::with(place, vec![i]),
        ));

        b.set_block(shift);
        let prev_idx_addr = at(b, U32, Shared, s.topk_idx, prev_i);
        let prev_idx = load(b, U32, prev_idx_addr, Shared);
        let dst_val = at(b, F32, Shared, s.topk_val, i);
        b.emit(KirOp::Store(dst_val, prev, Shared));
        let dst_idx = at(b, U32, Shared, s.topk_idx, i);
        b.emit(KirOp::Store(dst_idx, prev_idx, Shared));
        b.terminate(KirTerminator::Branch(KirEdge::with(inner, vec![prev_i])));

        b.set_block(place);
        let put_val = at(b, F32, Shared, s.topk_val, at_i);
        b.emit(KirOp::Store(put_val, key, Shared));
        let put_idx = at(b, U32, Shared, s.topk_idx, at_i);
        b.emit(KirOp::Store(put_idx, kidx, Shared));
    });

    // Cumulative share until > top_p; the tail from there is zeroed.
    let head = b.new_block();
    let body = b.new_block();
    let zero_from = b.new_block();
    let done = b.new_block();
    let j = b.add_block_param(head, U32);
    let cum = b.add_block_param(head, F32);
    let z = b.add_block_param(zero_from, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![c.zero, c.f_zero])));

    b.set_block(head);
    let finished = cmp(b, j, s.k, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(finished, KirEdge::to(done), KirEdge::to(body)));

    b.set_block(body);
    let p_addr = at(b, F32, Shared, s.topk_val, j);
    let p = load(b, F32, p_addr, Shared);
    let share = op2(b, F32, KirOp::Div, p, sum);
    let cum_next = op2(b, F32, KirOp::Add, cum, share);
    let j_next = op2(b, U32, KirOp::Add, j, c.one);
    let top_p = konst(b, ConstValue::F32(top_p));
    let over = cmp(b, cum_next, top_p, CmpOp::Gt);
    b.terminate(KirTerminator::CondBranch(
        over,
        KirEdge::with(zero_from, vec![j_next]),
        KirEdge::with(head, vec![j_next, cum_next]),
    ));

    b.set_block(zero_from);
    k_loop(b, s, z, |b, j| {
        let addr = at(b, F32, Shared, s.topk_val, j);
        b.emit(KirOp::Store(addr, c.f_zero, Shared));
    });
    b.terminate(KirTerminator::Branch(KirEdge::to(done)));

    b.set_block(done);
}

/// The `//` header the hand kernel carried, line for line.
fn header_comment(program: &FusedSampleProgram, cfg: &FusedSampleKernelConfig) -> String {
    let has_rms = has_op(program, |op| matches!(op, FusedSampleOp::RmsNorm));
    let top_p = nucleus_top_p(program);
    let inv_temp = temperature_recip(program);
    let mut w = String::new();
    writeln!(w, "//").unwrap();
    writeln!(w, "// {} - CFIE fused decode-sample (paper Feature 2).", KERNEL_NAME).unwrap();
    writeln!(w, "// One CTA, {} threads; logits stay in SMEM/registers, only the", BLOCK_DIM).unwrap();
    writeln!(w, "// sampled token id (4 bytes) is written to HBM.").unwrap();
    writeln!(w, "// LM-head layout: f16 [vocab, d_model], ROW-major per vocab row.").unwrap();
    writeln!(w, "// Baked constants:").unwrap();
    writeln!(w, "//   d_model          = {}", cfg.d_model).unwrap();
    writeln!(w, "//   vocab_size       = {}", cfg.vocab_size).unwrap();
    writeln!(w, "//   vocab_tile       = {}", TILE).unwrap();
    writeln!(w, "//   top_k            = {}", cfg.top_k).unwrap();
    writeln!(w, "//   temperature_recip= {} ({})", inv_temp, f32_imm(inv_temp)).unwrap();
    if let Some(tp) = top_p {
        writeln!(w, "//   top_p            = {} ({})", tp, f32_imm(tp)).unwrap();
    }
    if has_rms {
        writeln!(w, "//   rms_eps          = {} ({})", RMS_EPS, f32_imm(RMS_EPS)).unwrap();
    }
    if cfg.grammar_states > 0 {
        writeln!(
            w,
            "//   grammar: {} states x {} mask bytes/row (1 bit/token)",
            cfg.grammar_states,
            cfg.vocab_size.div_ceil(8)
        )
        .unwrap();
    }
    writeln!(w, "// PRNG: xorshift64* over rng_seed - deterministic given seed (M46).").unwrap();
    writeln!(w, "//").unwrap();
    w
}

/// Emit the fused decode-sample kernel for `program` under `cfg`: build,
/// verify, lower, and prefix the `//` header.
///
/// Launch shape: grid = 1, block = 128. The module targets the KIR floor
/// (`sm_70`), so [`FusedSampleKernelConfig`] has no `sm_version`. The
/// returned text carries no NUL.
pub fn emit(program: &FusedSampleProgram, cfg: &FusedSampleKernelConfig) -> (String, FusedSampleMeta) {
    let ir = build(program, cfg);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{KERNEL_NAME} failed KIR verification: {errors:?}");
    }
    let smem_bytes = ir.smem_layout.total_bytes().expect("a verified layout has a size");
    let module = lower_kir_to_ptx(&ir);
    let module = module.strip_suffix(&[0]).unwrap_or(&module);
    let module = std::str::from_utf8(module).expect("the KIR printer emits ASCII");
    let mut p = header_comment(program, cfg);
    p.push_str(module);

    let meta = FusedSampleMeta {
        kernel_name: KERNEL_NAME.to_string(),
        smem_bytes,
        block_dim: BLOCK_DIM,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit`].
pub fn emit_fused_sample_ptx(
    program: &FusedSampleProgram,
    cfg: &FusedSampleKernelConfig,
) -> String {
    emit(program, cfg).0
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// xorshift64* — the kernel's PRNG, bit-for-bit.
fn xorshift64star(seed: u64) -> u64 {
    // Zero seed is a fixed point of xorshift; the kernel substitutes the
    // golden-gamma constant, so mirror that here.
    let mut x = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn min_scan(vals: &[f32]) -> (f32, usize) {
    let mut mv = vals[0];
    let mut mp = 0usize;
    for (j, &v) in vals.iter().enumerate().skip(1) {
        if v < mv {
            mv = v;
            mp = j;
        }
    }
    (mv, mp)
}

/// CPU mirror of the fused kernel.  `lm_head_f32` is `[vocab][d_model]`
/// row-major, f32 (an exact-value cast of the kernel's f16 weights).
/// `grammar_mask` is `(bit rows, current DFA state)` — bit
/// `rows[state * ceil(vocab/8) + token/8] >> (token % 8)` gates the
/// token, matching the kernel's hook.
///
/// Mirrors the kernel exactly: same strided partial sums + tree
/// reduction for RMSNorm, same fma dot order, same replace-min /
/// argmax / sort / walk tie-breaks, same xorshift64*.  The kernel's
/// `rsqrt.approx` / `ex2.approx` are approximate where this uses exact
/// libm — exact GPU parity is verified in a later GPU cycle.
pub fn cpu_reference(
    program: &FusedSampleProgram,
    hidden: &[f32],
    norm_w: &[f32],
    lm_head_f32: &[f32],
    grammar_mask: Option<(&[u8], u32)>,
    seed: u64,
) -> u32 {
    let dm = program.shape.d_model as usize;
    let vocab = program.shape.vocab_size as usize;
    let k = program.params.top_k as usize;
    assert!((1..=64).contains(&k), "top_k must be in 1..=64");
    assert!((1..=8192).contains(&dm), "d_model must be in 1..=8192");
    assert_eq!(program.shape.vocab_tile, TILE, "vocab_tile must be {TILE}");
    assert_eq!(hidden.len(), dm, "hidden must be [1, d_model]");
    assert_eq!(
        lm_head_f32.len(),
        vocab * dm,
        "lm_head must be [vocab][d_model] row-major"
    );

    let has_rms = has_op(program, |op| matches!(op, FusedSampleOp::RmsNorm));
    let greedy = has_op(program, |op| matches!(op, FusedSampleOp::Argmax));
    let top_p = nucleus_top_p(program);
    let inv_temp = temperature_recip(program);

    let mut h = hidden.to_vec();
    if has_rms {
        assert_eq!(norm_w.len(), dm, "norm_w (gamma) must be [d_model]");
        // Per-thread strided partial sums, then the kernel's SMEM tree.
        let mut partials = [0f32; TILE as usize];
        for (t, part) in partials.iter_mut().enumerate() {
            let mut s = 0f32;
            let mut i = t;
            while i < dm {
                s = h[i].mul_add(h[i], s);
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
        for (i, hv) in h.iter_mut().enumerate() {
            *hv = (*hv * rstd) * norm_w[i];
        }
    }

    let mask_row_bytes = vocab.div_ceil(8);
    let mut tv = vec![f32::NEG_INFINITY; k];
    let mut ti = vec![0u32; k];
    let mut tile = 0usize;
    while tile < vocab {
        let cnt = (vocab - tile).min(TILE as usize);
        let mut scores = vec![f32::NEG_INFINITY; cnt];
        for (t, score) in scores.iter_mut().enumerate() {
            let tok = tile + t;
            let row = &lm_head_f32[tok * dm..(tok + 1) * dm];
            let mut dot = 0f32;
            for d in 0..dm {
                dot = row[d].mul_add(h[d], dot);
            }
            let mut s = dot * inv_temp;
            if let Some((mask, state)) = grammar_mask {
                let byte = mask[state as usize * mask_row_bytes + tok / 8];
                if (byte >> (tok & 7)) & 1 == 0 {
                    s = f32::NEG_INFINITY;
                }
            }
            *score = s;
        }
        // Thread-0 replace-min merge, identical insertion order.
        let (mut mv, mut mp) = min_scan(&tv);
        for (i, &s) in scores.iter().enumerate() {
            if s > mv {
                tv[mp] = s;
                ti[mp] = (tile + i) as u32;
                let r = min_scan(&tv);
                mv = r.0;
                mp = r.1;
            }
        }
        tile += TILE as usize;
    }

    if greedy {
        let mut mx = tv[0];
        let mut sel = ti[0];
        for j in 1..k {
            if tv[j] > mx {
                mx = tv[j];
                sel = ti[j];
            }
        }
        return sel;
    }

    // Softmax over the k candidates only.
    let mut mx = tv[0];
    for j in 1..k {
        if tv[j] > mx {
            mx = tv[j];
        }
    }
    let mut sum = 0f32;
    for v in tv.iter_mut() {
        let p = ((*v - mx) * std::f32::consts::LOG2_E).exp2();
        *v = p;
        sum += p;
    }

    if let Some(tp) = top_p {
        // Insertion sort descending (stable: shift only while strictly
        // smaller than the key).
        for j in 1..k {
            let key = tv[j];
            let kidx = ti[j];
            let mut i = j;
            while i > 0 && tv[i - 1] < key {
                tv[i] = tv[i - 1];
                ti[i] = ti[i - 1];
                i -= 1;
            }
            tv[i] = key;
            ti[i] = kidx;
        }
        // Cumulative prob until > top_p; the crossing entry is kept,
        // the tail is zeroed.
        let mut cum = 0f32;
        let mut j = 0usize;
        while j < k {
            cum += tv[j] / sum;
            j += 1;
            if cum > tp {
                break;
            }
        }
        for z in tv.iter_mut().skip(j) {
            *z = 0.0;
        }
    }

    let mut ks = 0f32;
    for &v in tv.iter() {
        ks += v;
    }
    let r = ((xorshift64star(seed) >> 40) as u32) as f32 * (1.0 / 16_777_216.0);
    let target = r * ks;

    let mut sel = ti[0];
    let mut cum = 0f32;
    for j in 0..k {
        let p = tv[j];
        cum += p;
        if p > 0.0 {
            sel = ti[j];
            if cum >= target {
                break;
            }
        }
    }
    sel
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_fused_sample::{
        emit_program, LmHeadShape, SamplingParams, SamplingStrategy,
    };

    fn shape(d_model: u32, vocab_size: u32) -> LmHeadShape {
        LmHeadShape {
            d_model,
            vocab_size,
            vocab_tile: 128,
            dtype_bytes: 2,
        }
    }

    /// Reference config from the CFIE paper's NSL-Coder example.
    fn paper_cfg() -> FusedSampleKernelConfig {
        FusedSampleKernelConfig {
            d_model: 512,
            vocab_size: 49_152,
            vocab_tile: 128,
            top_k: 50,
            grammar_states: 0,
        }
    }

    fn paper_program() -> crate::cfie_fused_sample::FusedSampleProgram {
        emit_program(SamplingParams::default(), shape(512, 49_152))
    }

    // ── structural ─────────────────────────────────────────────────

    /// Every op of `ir`, in block order.
    fn ops(ir: &KernelIR) -> Vec<&KirOp> {
        ir.blocks.iter().flat_map(|b| b.ops.iter()).collect()
    }

    fn f32_consts(ir: &KernelIR) -> Vec<u32> {
        ops(ir)
            .into_iter()
            .filter_map(|op| match op {
                KirOp::Const(_, crate::kernel_ir::KirConst { value: ConstValue::F32(v), .. }) => Some(v.to_bits()),
                _ => None,
            })
            .collect()
    }

    fn u32_consts(ir: &KernelIR) -> Vec<u32> {
        ops(ir)
            .into_iter()
            .filter_map(|op| match op {
                KirOp::Const(_, crate::kernel_ir::KirConst { value: ConstValue::U32(v), .. }) => Some(*v),
                _ => None,
            })
            .collect()
    }

    fn greedy_program() -> crate::cfie_fused_sample::FusedSampleProgram {
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            ..Default::default()
        };
        emit_program(params, shape(512, 49_152))
    }

    #[test]
    fn param_list_is_exactly_the_seven_params() {
        let ir = build(&paper_program(), &paper_cfg());
        let names: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "hidden_ptr",
                "norm_w_ptr",
                "lm_head_ptr",
                "out_token_ptr",
                "rng_seed",
                "grammar_mask_ptr",
                "grammar_state",
            ]
        );
        let ptx = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        for (name, ty) in [
            ("hidden_ptr", "u64"),
            ("norm_w_ptr", "u64"),
            ("lm_head_ptr", "u64"),
            ("out_token_ptr", "u64"),
            ("rng_seed", "u64"),
            ("grammar_mask_ptr", "u64"),
            ("grammar_state", "u32"),
        ] {
            assert!(ptx.contains(&format!(".param .{ty} param_{name}")), "{name}");
        }
    }

    #[test]
    fn exactly_one_global_store() {
        // The [1, vocab] logits never touch HBM: the token id write is
        // the kernel's only global store, sampling and greedy alike.
        for prog in [paper_program(), greedy_program()] {
            let ir = build(&prog, &paper_cfg());
            let global_stores = ops(&ir)
                .into_iter()
                .filter(|op| matches!(op, KirOp::Store(_, _, AddressSpace::Global)))
                .count();
            assert_eq!(global_stores, 1);
            let ptx = emit_fused_sample_ptx(&prog, &paper_cfg());
            assert_eq!(ptx.matches("st.global").count(), 1);
            assert!(ptx.contains("st.global.u32 "));
        }
    }

    #[test]
    fn no_mad_lo_and_ascii_only() {
        let ptx = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
        assert!(
            ptx.bytes().all(|b| b < 128),
            "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
        );
    }

    #[test]
    fn baked_temperature_and_top_p_immediates_present() {
        let consts = f32_consts(&build(&paper_program(), &paper_cfg()));
        // Default temperature 0.7 -> baked 1/0.7 epilogue multiplier.
        assert!(consts.contains(&(1.0f32 / 0.7f32).to_bits()));
        // Default top_p 0.9 -> baked nucleus threshold.
        assert!(consts.contains(&0.9f32.to_bits()));
        // RMSNorm epsilon baked (default program has the op).
        assert!(consts.contains(&1e-5f32.to_bits()));
    }

    #[test]
    fn grammar_hook_present_only_when_grammar_states_positive() {
        let byte_loads = |ir: &KernelIR| {
            ops(ir)
                .into_iter()
                .filter(|op| matches!(op, KirOp::Load(dst, _, _) if ir.var_types.get(dst) == Some(&KirType::I8)))
                .count()
        };
        let no_grammar = build(&paper_program(), &paper_cfg());
        assert_eq!(byte_loads(&no_grammar), 0);
        assert!(!u32_consts(&no_grammar).contains(&6144));

        let params = SamplingParams {
            grammar_masked: true,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let mut cfg = paper_cfg();
        cfg.grammar_states = 4;
        let ir = build(&prog, &cfg);
        assert_eq!(byte_loads(&ir), 1);
        // Baked mask row stride: ceil(49152 / 8) = 6144 bytes/state.
        assert!(u32_consts(&ir).contains(&6144));
        // The runtime null-pointer guard keeps the hook inert until a mask
        // is bound: the byte load sits behind a branch on it.
        let ptx = emit_fused_sample_ptx(&prog, &cfg);
        assert!(ptx.contains("ld.global.s8 "));
        assert!(ptx.contains("setp.eq.u64 "));
    }

    #[test]
    fn greedy_program_skips_softmax_sort_and_rng() {
        let has_exp = |ir: &KernelIR| ops(ir).into_iter().any(|op| matches!(op, KirOp::Exp(..)));
        let has_xor = |ir: &KernelIR| ops(ir).into_iter().any(|op| matches!(op, KirOp::Xor(..)));
        let has_div = |ir: &KernelIR| ops(ir).into_iter().any(|op| matches!(op, KirOp::Div(..)));

        let greedy = build(&greedy_program(), &paper_cfg());
        assert!(!has_exp(&greedy), "no softmax");
        assert!(!has_div(&greedy), "no nucleus cutoff");
        assert!(!has_xor(&greedy), "no PRNG");

        let full = build(&paper_program(), &paper_cfg());
        assert!(has_exp(&full) && has_div(&full) && has_xor(&full));
    }

    #[test]
    fn rms_norm_section_gated_on_program_op() {
        let with_rms = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(with_rms.contains("rsqrt.approx.f32"));

        let params = SamplingParams {
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let ptx = emit_fused_sample_ptx(&prog, &paper_cfg());
        assert!(!ptx.contains("rsqrt.approx.f32"));
    }

    #[test]
    fn module_targets_the_kir_floor_after_the_header() {
        let ptx = emit_fused_sample_ptx(&paper_program(), &paper_cfg());
        assert!(ptx.starts_with("//"));
        let directives: Vec<&str> = ptx
            .lines()
            .filter(|l| l.starts_with(".version") || l.starts_with(".target") || l.starts_with(".address_size"))
            .collect();
        assert_eq!(directives.len(), 3, "{directives:?}");
        assert!(directives[1].starts_with(".target sm_70"), "{directives:?}");
        assert_eq!(directives[2], ".address_size 64");
    }

    #[test]
    fn meta_reports_launch_shape() {
        let (ptx, meta) = emit(&paper_program(), &paper_cfg());
        assert_eq!(meta.kernel_name, kernel_name());
        assert_eq!(meta.block_dim, 128);
        // hidden(512 f32) + scores(128 f32) + topk_val(50) + topk_idx(50) + rstd.
        assert_eq!(meta.smem_bytes, 512 * 4 + 128 * 4 + 50 * 4 + 50 * 4 + 4);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 shared_mem[{}];", meta.smem_bytes)));
    }

    #[test]
    #[should_panic(expected = "vocab_tile")]
    fn vocab_tile_not_128_panics() {
        let mut cfg = paper_cfg();
        cfg.vocab_tile = 256;
        let prog = emit_program(
            SamplingParams::default(),
            LmHeadShape {
                d_model: 512,
                vocab_size: 49_152,
                vocab_tile: 256,
                dtype_bytes: 2,
            },
        );
        let _ = emit(&prog, &cfg);
    }

    #[test]
    #[should_panic(expected = "top_k")]
    fn top_k_over_64_panics() {
        let mut cfg = paper_cfg();
        cfg.top_k = 65;
        let params = SamplingParams {
            top_k: 65,
            ..Default::default()
        };
        let prog = emit_program(params, shape(512, 49_152));
        let _ = emit(&prog, &cfg);
    }

    #[test]
    #[should_panic(expected = "d_model")]
    fn d_model_over_8192_panics() {
        let mut cfg = paper_cfg();
        cfg.d_model = 8193;
        let prog = emit_program(SamplingParams::default(), shape(8193, 49_152));
        let _ = emit(&prog, &cfg);
    }

    #[test]
    #[should_panic(expected = "mismatch")]
    fn program_shape_mismatch_panics() {
        // cfg says d_model 512, program says 256.
        let prog = emit_program(SamplingParams::default(), shape(256, 49_152));
        let _ = emit(&prog, &paper_cfg());
    }

    // ── cpu_reference ──────────────────────────────────────────────

    fn small_weights(vocab: usize, dm: usize) -> Vec<f32> {
        (0..vocab * dm)
            .map(|i| ((i * 7 + 3) % 11) as f32 * 0.1 - 0.5)
            .collect()
    }

    #[test]
    fn cpu_reference_is_deterministic_for_fixed_seed() {
        let (dm, vocab) = (16usize, 256u32);
        let prog = emit_program(SamplingParams::default(), shape(dm as u32, vocab));
        let hidden: Vec<f32> = (0..dm).map(|i| (i as f32 * 0.37).sin()).collect();
        let gamma: Vec<f32> = (0..dm).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let w = small_weights(vocab as usize, dm);
        let t1 = cpu_reference(&prog, &hidden, &gamma, &w, None, 0xDEAD_BEEF);
        let t2 = cpu_reference(&prog, &hidden, &gamma, &w, None, 0xDEAD_BEEF);
        assert_eq!(t1, t2);
        assert!(t1 < vocab);
    }

    #[test]
    fn cpu_reference_greedy_matches_naive_argmax() {
        let (dm, vocab) = (4usize, 10usize);
        let params = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: 10,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab as u32));
        let hidden = [0.3f32, -1.2, 0.7, 0.05];
        let mut w = small_weights(vocab, dm);
        // Force a unique max at row 6: align the row with hidden's signs.
        for d in 0..dm {
            w[6 * dm + d] = if hidden[d] >= 0.0 { 5.0 } else { -5.0 };
        }
        let naive = (0..vocab)
            .map(|t| {
                (0..dm)
                    .map(|d| w[t * dm + d] * hidden[d])
                    .sum::<f32>()
            })
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (t, s)| {
                if s > acc.1 {
                    (t, s)
                } else {
                    acc
                }
            })
            .0;
        assert_eq!(naive, 6);
        for seed in [0u64, 1, 42, u64::MAX] {
            let tok = cpu_reference(&prog, &hidden, &[], &w, None, seed);
            assert_eq!(tok as usize, naive, "greedy must ignore the seed");
        }
    }

    #[test]
    fn cpu_reference_grammar_mask_excludes_tokens() {
        let (dm, vocab) = (2usize, 8u32);
        let params = SamplingParams {
            top_k: 4,
            grammar_masked: true,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab));
        let hidden = [0.9f32, -0.4];
        let w = small_weights(vocab as usize, dm);
        // One mask row (state 0), only token 5 allowed.
        let mask = [0b0010_0000u8];
        for seed in 0..32u64 {
            let tok = cpu_reference(&prog, &hidden, &[], &w, Some((&mask, 0)), seed);
            assert_eq!(tok, 5, "grammar mask must exclude every other token");
        }
    }

    #[test]
    fn cpu_reference_top_p_excludes_tail() {
        // Dominant logit: p(token 0) ~= 0.9999 > top_p = 0.5, so the
        // nucleus keeps only token 0 for every seed.
        let (dm, vocab) = (1usize, 4u32);
        let params = SamplingParams {
            strategy: SamplingStrategy::TopKTopP,
            temperature: 1.0,
            top_k: 4,
            top_p: 0.5,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab));
        let hidden = [1.0f32];
        let w = [10.0f32, 0.0, 0.0, 0.0];
        for seed in 0..64u64 {
            let tok = cpu_reference(&prog, &hidden, &[], &w, None, seed);
            assert_eq!(tok, 0, "nucleus tail must never be sampled");
        }
    }

    #[test]
    fn cpu_reference_uniform_gamma_preserves_greedy_argmax() {
        // RMSNorm with gamma == 1 is a uniform positive rescale of the
        // hidden state, so the greedy argmax is invariant.
        let (dm, vocab) = (8usize, 12u32);
        let base = SamplingParams {
            strategy: SamplingStrategy::Greedy,
            temperature: 0.0,
            top_k: 12,
            ..Default::default()
        };
        let hidden: Vec<f32> = (0..dm).map(|i| (i as f32 * 0.61).cos()).collect();
        let mut w = small_weights(vocab as usize, dm);
        for d in 0..dm {
            w[9 * dm + d] = 2.0; // unique max at row 9
        }
        let gamma = vec![1.0f32; dm];

        let prog_on = emit_program(base, shape(dm as u32, vocab));
        let prog_off = emit_program(
            SamplingParams {
                rms_norm: false,
                ..base
            },
            shape(dm as u32, vocab),
        );
        let on = cpu_reference(&prog_on, &hidden, &gamma, &w, None, 7);
        let off = cpu_reference(&prog_off, &hidden, &[], &w, None, 7);
        assert_eq!(on, off);
    }

    #[test]
    fn cpu_reference_multinomial_spreads_over_seeds() {
        // Uniform logits: different seeds must be able to pick
        // different tokens.
        let (dm, vocab) = (1usize, 4u32);
        let params = SamplingParams {
            strategy: SamplingStrategy::TopK,
            temperature: 1.0,
            top_k: 4,
            rms_norm: false,
            ..Default::default()
        };
        let prog = emit_program(params, shape(dm as u32, vocab));
        let hidden = [1.0f32];
        let w = [0.0f32; 4];
        let mut seen = std::collections::BTreeSet::new();
        for seed in 1..=32u64 {
            seen.insert(cpu_reference(&prog, &hidden, &[], &w, None, seed));
        }
        assert!(seen.len() > 1, "multinomial must not collapse to one token");
        assert!(seen.iter().all(|&t| t < vocab));
    }

    // ── ptxas validation (skips silently when no validator present) ──

    #[test]
    fn ptxas_validates_paper_config() {
        let mut checks = vec![(paper_program(), paper_cfg())];
        // Grammar-hook variant exercises the mask address arithmetic.
        let params = SamplingParams {
            grammar_masked: true,
            ..Default::default()
        };
        let mut gcfg = paper_cfg();
        gcfg.grammar_states = 4;
        checks.push((emit_program(params, shape(512, 49_152)), gcfg));

        for (prog, cfg) in checks {
            let ptx = emit_fused_sample_ptx(&prog, &cfg);
            match crate::ptxas_validation::validate_ptx(&ptx) {
                Ok(()) => {}
                Err(msg) if msg.contains("nvcc not available") => {
                    nsl_log::nsl_log!(INFO, "skip", 
                        "[skip] cfie fused-sample ptxas validation - no validator: {msg}"
                    );
                }
                Err(msg) => panic!(
                    "cfie fused-sample PTX rejected for paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
                ),
            }
        }
    }
}
