// Frozen fixture (roadmap A2 step 9): the hand-written PTX emitters for the
// CFIE speculative-decoding kernels, exactly as `crates/nsl-codegen/src/
// cfie_speculative_ptx.rs` held them at f2818f64, before the kernels moved
// onto KIR. Lines below this comment are that file's first 829 lines,
// unedited; the `cfie_speculative_kir_equivalence` gate includes it with
// `#[path]` (supplying the two `crate::` items it names) and runs its
// output against the KIR kernels'. Do not change it: it is the
// pre-migration behaviour the equivalence claim is about.
//
// Not scanned by the hand-PTX freeze: nothing under `tests/` is.

//! CFIE Feature 3, kernel half (audit gaps G13 + G14): compiled
//! speculative verification kernels.
//!
//! G14 — tree-mask verification attention: one launch scores all
//! `num_nodes` speculative positions (draft tree nodes, BFS-numbered by
//! `cfie_speculative::build_tree_mask`) against the committed KV prefix
//! plus the mask-allowed tree nodes.  The paper's claim vs the runtime
//! tree-parent params in `flash_attention.rs`: the ancestor mask is a
//! compile-time constant — ONE baked u64 immediate per node row, no
//! mask tensor parameter.  `num_nodes <= 33` (K+1 <= 33) so a row's
//! bits fit one u64.
//!
//! G13 — rejection-sampling epilogue (paper step 3): a single serial
//! CTA walks the K draft positions with the same xorshift64* PRNG as
//! `cfie_sample_ptx` (same golden-gamma zero-seed guard, advanced as a
//! sequential state), accepting token j iff
//! `r < p_target[j][tok_j] / p_draft[j]` (`p_draft <= 0` rejects — the
//! division guard).  On the first rejection it samples the correction
//! from the Leviathan residual: `q(x) = max(p_target(x) -
//! p_draft*[x == tok_j], 0)` renormalised.  `draft_probs` carries only
//! the drafted token's probability, so the residual is `p_target` with
//! the drafted token's entry reduced by that scalar and clamped at 0 —
//! faithful to Leviathan et al. 2023 given the per-token prob the
//! draft phase captures.  A later cycle fuses the target-probs input
//! with the verify matmul epilogue; v1 takes the softmaxed rows as a
//! kernel parameter.
//!
//! Host contract (decode-loop integration cycle):
//!   * verify: the `num_nodes` draft K/V rows (RoPE-rotated by the
//!     host at their tree positions) are appended to the baked KV pool
//!     at positions `seq_len .. seq_len + num_nodes` of (layer, slot)
//!     BEFORE launch; the pool layout/strides are byte-identical to
//!     `cfie_decode_attention` (cross-module test enforced).
//!   * reject, LINEAR chain (method != tree): after readback the host
//!     calls the Cycle-2 FFI
//!     `nsl_cfie_kv_slot_rollback(slot, k_tokens - accepted)` to
//!     discard the rejected draft KV entries (the correction token's
//!     KV is appended by the normal decode step that consumes it).
//!   * reject, TREE method: `num_nodes` rows were appended (not K),
//!     the accepted tokens form ONE root-to-leaf path through the
//!     tree, and the truncate-only rollback cannot compact a
//!     non-contiguous path.  The host must therefore (1) linearize the
//!     candidate path's per-position probs BEFORE invoking the reject
//!     kernel (its serial j = 0..K walk assumes one path), then
//!     (2) `nsl_cfie_kv_slot_rollback(slot, num_nodes)` to drop ALL
//!     appended tree rows, and (3) re-append the accepted path's K/V
//!     rows contiguously (advance + device-side copy — the values are
//!     already in the pool, no recompute).
//!     No runtime edits here — the decode-loop cycle wires these calls.

use crate::cfie_speculative::TreeMask;
use std::fmt::Write;

/// Threads per CTA and softmax tile width for the verify kernel.
const TILE: u32 = 128;
const VERIFY_BLOCK_DIM: u32 = TILE;
/// The rejection kernel is a serial thread-0 walk (correctness first);
/// one warp keeps the launch honest about its shape.
const REJECT_BLOCK_DIM: u32 = 32;

/// Row bits must fit a u64 immediate and K is clamped to 32 upstream.
const MAX_NODES: u32 = 33;

pub const VERIFY_KERNEL_NAME: &str = "nsl_cfie_spec_verify_attn";
pub const REJECT_KERNEL_NAME: &str = "nsl_cfie_spec_reject";

/// Compile-time layout + tree configuration for the verify kernel.
#[derive(Debug, Clone)]
pub struct VerifyAttentionConfig {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub per_slot_max_tokens: u32,
    pub max_slots: u32,
    /// Speculative positions verified per launch (K+1 <= 33).
    pub num_nodes: u32,
    /// Baked ancestor mask: bit `c` of `mask_bits[r]` set iff node `r`
    /// attends node `c` (rows from `cfie_speculative::TreeMask`).
    pub mask_bits: Vec<u64>,
    pub sm_version: u32,
}

/// Compile-time configuration for the rejection kernel.
#[derive(Debug, Clone)]
pub struct RejectionConfig {
    /// Draft tokens per speculative step (1..=32).
    pub k_tokens: u32,
    pub vocab_size: u32,
    pub sm_version: u32,
}

/// Host-readable launch metadata emitted alongside the PTX.
#[derive(Debug, Clone)]
pub struct SpecKernelMeta {
    pub kernel_name: String,
    pub smem_bytes: u32,
    pub block_dim: u32,
    /// Verify kernel: grid = n_heads CTAs.  Reject kernel: grid = 1.
    pub grid_dim_is_n_heads: bool,
}

fn f32_imm(v: f32) -> String {
    format!("0f{:08X}", v.to_bits())
}

/// Pack a tested BFS [`TreeMask`] into the per-row u64 immediates the
/// verify emitter bakes.
pub fn mask_bits_from_tree(mask: &TreeMask) -> Vec<u64> {
    assert!(
        mask.num_nodes >= 1 && mask.num_nodes <= 64,
        "tree mask rows must fit u64 bits (num_nodes = {})",
        mask.num_nodes
    );
    (0..mask.num_nodes)
        .map(|r| {
            let mut row = 0u64;
            for c in 0..mask.num_nodes {
                if mask.get(r, c) {
                    row |= 1u64 << c;
                }
            }
            row
        })
        .collect()
}

// ---------------------------------------------------------------------------
// G14: tree-mask verification attention
// ---------------------------------------------------------------------------

/// Emit the tree-mask verification attention kernel.
///
/// Launch shape: grid = n_heads CTAs, block = 128.  Node rows are
/// looped serially inside the CTA (unrolled at emission — the mask row
/// is a per-node immediate).  `q`/`out` are f32
/// `[num_nodes, n_heads, head_dim]`; `seq_len` is the committed prefix
/// length (the draft rows sit at pool positions
/// `seq_len .. seq_len + num_nodes`, appended by the host beforehand).
pub fn emit_verify_attention(cfg: &VerifyAttentionConfig) -> (String, SpecKernelMeta) {
    assert!(
        cfg.n_heads >= 1 && cfg.n_kv_heads >= 1,
        "n_heads and n_kv_heads must be >= 1"
    );
    assert_eq!(
        cfg.n_heads % cfg.n_kv_heads,
        0,
        "GQA requires n_heads divisible by n_kv_heads"
    );
    assert!(
        cfg.head_dim >= 1 && cfg.head_dim <= VERIFY_BLOCK_DIM,
        "pass 3 maps one thread per output element; head_dim must be in 1..={}",
        VERIFY_BLOCK_DIM
    );
    assert!(
        cfg.num_nodes >= 1 && cfg.num_nodes <= MAX_NODES,
        "num_nodes must be in 1..={} (K+1, row bits fit a u64 immediate)",
        MAX_NODES
    );
    assert_eq!(
        cfg.mask_bits.len(),
        cfg.num_nodes as usize,
        "mask_bits must have one row per node"
    );
    for (i, &row) in cfg.mask_bits.iter().enumerate() {
        assert!(
            row & (1u64 << i) != 0,
            "mask row {i} must include the self bit (softmax denominator)"
        );
        assert_eq!(
            row >> cfg.num_nodes,
            0,
            "mask row {i} sets bits beyond num_nodes"
        );
    }
    assert!(
        cfg.per_slot_max_tokens >= cfg.num_nodes && cfg.max_slots >= 1,
        "per_slot_max_tokens must fit the appended tree rows and max_slots must be >= 1"
    );

    // Baked strides in ELEMENTS — identical derivation to
    // `cfie_decode_attention` (contiguous layout
    // [n_layers][2][max_tokens][n_kv_heads][head_dim], f16).
    let token_stride = cfg.n_kv_heads as u64 * cfg.head_dim as u64;
    let max_tokens = cfg.max_slots as u64 * cfg.per_slot_max_tokens as u64;
    assert!(
        max_tokens <= u32::MAX as u64,
        "global token pool (max_slots * per_slot_max_tokens = {max_tokens}) must fit in u32"
    );
    let kv_half_stride = max_tokens * token_stride;
    let layer_stride = 2 * kv_half_stride;

    let dtype = 2u64; // f16 pool, the only supported v1 dtype
    let token_stride_bytes = token_stride * dtype;
    let kv_half_stride_bytes = kv_half_stride * dtype;
    let layer_stride_bytes = layer_stride * dtype;
    let head_row_bytes = cfg.head_dim as u64 * dtype;

    let group = cfg.n_heads / cfg.n_kv_heads;
    let inv_sqrt_hd = f32_imm(1.0f32 / (cfg.head_dim as f32).sqrt());
    let log2e = f32_imm(std::f32::consts::LOG2_E);
    let neg_inf = f32_imm(f32::NEG_INFINITY);
    let zero = f32_imm(0.0);

    // SMEM layout (f32): [q: head_dim][scores: TILE][rescale: 1][l: 1]
    // — same shape as cfie_decode_attention, reused per node row.
    let scores_off = cfg.head_dim * 4;
    let rescale_off = scores_off + TILE * 4;
    let l_off = rescale_off + 4;
    let smem_bytes = l_off + 4;

    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let nn = cfg.num_nodes;
    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(
        w,
        "// {} - CFIE speculative verification attention (tree mask baked).",
        VERIFY_KERNEL_NAME
    )
    .unwrap();
    writeln!(
        w,
        "// {} node rows verified per launch; each row's ancestor mask is a",
        nn
    )
    .unwrap();
    writeln!(
        w,
        "// compile-time u64 immediate - no mask tensor parameter."
    )
    .unwrap();
    writeln!(
        w,
        "// KV pool layout [n_layers][2][max_tokens={}][n_kv_heads={}][head_dim={}], f16.",
        max_tokens, cfg.n_kv_heads, cfg.head_dim
    )
    .unwrap();
    writeln!(
        w,
        "// Host appends the {} draft K/V rows at positions seq_len..seq_len+{}",
        nn, nn
    )
    .unwrap();
    writeln!(w, "// of (layer, slot) BEFORE launch.").unwrap();
    writeln!(w, "// Baked layout constants (elements):").unwrap();
    writeln!(w, "//   token_stride        = {}", token_stride).unwrap();
    writeln!(w, "//   kv_half_stride      = {}", kv_half_stride).unwrap();
    writeln!(w, "//   layer_stride        = {}", layer_stride).unwrap();
    writeln!(w, "//   per_slot_max_tokens = {}", cfg.per_slot_max_tokens).unwrap();
    writeln!(w, "//   max_tokens          = {}", max_tokens).unwrap();
    writeln!(w, "//   gqa_group_size      = {}", group).unwrap();
    writeln!(w, "// Baked mask rows (bit c of row r = node r attends node c):").unwrap();
    for (i, &row) in cfg.mask_bits.iter().enumerate() {
        writeln!(w, "//   node {:>2} mask = 0x{:016X}", i, row).unwrap();
    }
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", crate::gpu_specs::ptx_isa_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".shared .align 4 .b8 cfie_spec_smem[{}];", smem_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", VERIFY_KERNEL_NAME).unwrap();
    writeln!(w, "    .param .u64 q_ptr,").unwrap();
    writeln!(w, "    .param .u64 kv_base,").unwrap();
    writeln!(w, "    .param .u64 out_ptr,").unwrap();
    writeln!(w, "    .param .u32 layer_idx,").unwrap();
    writeln!(w, "    .param .u32 slot_idx,").unwrap();
    writeln!(w, "    .param .u32 seq_len").unwrap();
    writeln!(w, ")").unwrap();
    writeln!(w, "{{").unwrap();
    writeln!(
        w,
        "    .reg .pred %p_qd, %p_done, %p_val, %p_d, %p_t0, %p_j, %p_j2, %p_nd, %p_t1, %p_no, %p_lz, %p_m;"
    )
    .unwrap();
    writeln!(w, "    .reg .b16 %h_k, %h_v;").unwrap();
    writeln!(
        w,
        "    .reg .f32 %f_q, %f_k, %f_v, %f_p, %f_s, %f_dot, %f_acc, %f_m, %f_l, %f_tm, %f_rs, %f_rs2, %f_t1, %f_lf, %f_o, %f_t0;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u32 %r_tid, %r_head, %r_kvhead, %r_layer, %r_slot, %r_seqlen, %r_sbase, %r_slotbase, %r_hoff, %r_tile, %r_rem, %r_tcnt, %r_tok, %r_g, %r_g0, %r_d, %r_qsm, %r_j, %r_sp, %r_t1, %r_t2, %r_t3, %r_t4;"
    )
    .unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_q, %rd_kv, %rd_out, %rd_kplane, %rd_vplane, %rd_hoff, %rd_koff, %rd_kaddr, %rd_voff, %rd_vaddr, %rd_mask, %rd_mb, %rd_t0, %rd_t1, %rd_t2, %rd_t3;"
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_q, [q_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_kv, [kv_base];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_out, [out_ptr];").unwrap();
    writeln!(w, "    ld.param.u32 %r_layer, [layer_idx];").unwrap();
    writeln!(w, "    ld.param.u32 %r_slot, [slot_idx];").unwrap();
    writeln!(w, "    ld.param.u32 %r_seqlen, [seq_len];").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_head, %ctaid.x;").unwrap();
    writeln!(w, "    mov.u32 %r_sbase, cfie_spec_smem;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // GQA: kv_head = q_head / group_size (baked divisor)").unwrap();
    writeln!(w, "    div.u32 %r_kvhead, %r_head, {};", group).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // K plane base (bytes): kv_base + layer_idx * layer_stride_bytes").unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_t0, %r_layer;").unwrap();
    writeln!(w, "    mul.lo.u64 %rd_kplane, %rd_t0, {};", layer_stride_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_kplane, %rd_kv, %rd_kplane;").unwrap();
    writeln!(w, "    // V plane = K plane + kv_half_stride_bytes").unwrap();
    writeln!(w, "    add.u64 %rd_vplane, %rd_kplane, {};", kv_half_stride_bytes).unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // slot's first global token: slot_idx * per_slot_max_tokens").unwrap();
    writeln!(
        w,
        "    mul.lo.u32 %r_slotbase, %r_slot, {};",
        cfg.per_slot_max_tokens
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // byte offset of kv_head's row inside one token record").unwrap();
    writeln!(w, "    mul.lo.u32 %r_hoff, %r_kvhead, {};", head_row_bytes).unwrap();
    writeln!(w, "    cvt.u64.u32 %rd_hoff, %r_hoff;").unwrap();
    writeln!(w).unwrap();

    // One fully-emitted flash-decode pass per node row: the row's tree
    // mask is compile-time constant, so the node loop is unrolled at
    // emission (num_nodes <= 33).
    for (i, &mask_row) in cfg.mask_bits.iter().enumerate() {
        let i = i as u32;
        let l = |s: &str| format!("N{}_{}", i, s);
        writeln!(w, "    // ==== node row {} (mask 0x{:016X}) ====", i, mask_row).unwrap();
        writeln!(w, "    // load this node+head's Q row (f32) into SMEM").unwrap();
        writeln!(w, "    setp.lt.u32 %p_qd, %r_tid, {};", hd).unwrap();
        writeln!(w, "    @!%p_qd bra {};", l("QDONE")).unwrap();
        writeln!(w, "    mul.lo.u32 %r_t1, %r_head, {};", hd).unwrap();
        writeln!(w, "    add.u32 %r_t1, %r_t1, {};", i * nh * hd).unwrap();
        writeln!(w, "    add.u32 %r_t1, %r_t1, %r_tid;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t1, %r_t1, 4;").unwrap();
        writeln!(w, "    cvt.u64.u32 %rd_t1, %r_t1;").unwrap();
        writeln!(w, "    add.u64 %rd_t1, %rd_q, %rd_t1;").unwrap();
        writeln!(w, "    ld.global.f32 %f_t0, [%rd_t1];").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t2, %r_tid, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t2, %r_t2, %r_sbase;").unwrap();
        writeln!(w, "    st.shared.f32 [%r_t2], %f_t0;").unwrap();
        writeln!(w, "{}:", l("QDONE")).unwrap();
        writeln!(w, "    mov.f32 %f_acc, {};", zero).unwrap();
        writeln!(w, "    mov.f32 %f_m, {};", neg_inf).unwrap();
        writeln!(w, "    mov.f32 %f_l, {};", zero).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w).unwrap();
        writeln!(w, "    // prefix pass: 3-pass flash-decode over the committed tokens").unwrap();
        writeln!(w, "    mov.u32 %r_tile, 0;").unwrap();
        writeln!(w, "{}:", l("TILE")).unwrap();
        writeln!(w, "    setp.ge.u32 %p_done, %r_tile, %r_seqlen;").unwrap();
        writeln!(w, "    @%p_done bra {};", l("PREFIX_END")).unwrap();
        writeln!(w, "    sub.u32 %r_rem, %r_seqlen, %r_tile;").unwrap();
        writeln!(w, "    min.u32 %r_tcnt, %r_rem, {};", TILE).unwrap();
        writeln!(w, "    add.u32 %r_tok, %r_tile, %r_tid;").unwrap();
        writeln!(w, "    setp.lt.u32 %p_val, %r_tok, %r_seqlen;").unwrap();
        writeln!(w, "    @!%p_val bra {};", l("SCD")).unwrap();
        writeln!(w, "    add.u32 %r_g, %r_slotbase, %r_tok;").unwrap();
        writeln!(w, "    mul.wide.u32 %rd_koff, %r_g, {};", token_stride_bytes).unwrap();
        writeln!(w, "    add.u64 %rd_kaddr, %rd_kplane, %rd_koff;").unwrap();
        writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, %rd_hoff;").unwrap();
        writeln!(w, "    mov.f32 %f_dot, {};", zero).unwrap();
        writeln!(w, "    mov.u32 %r_d, 0;").unwrap();
        writeln!(w, "    mov.u32 %r_qsm, %r_sbase;").unwrap();
        writeln!(w, "{}:", l("DOT")).unwrap();
        writeln!(w, "    setp.ge.u32 %p_d, %r_d, {};", hd).unwrap();
        writeln!(w, "    @%p_d bra {};", l("DOTD")).unwrap();
        writeln!(w, "    ld.global.b16 %h_k, [%rd_kaddr];").unwrap();
        writeln!(w, "    cvt.f32.f16 %f_k, %h_k;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_q, [%r_qsm];").unwrap();
        writeln!(w, "    fma.rn.f32 %f_dot, %f_k, %f_q, %f_dot;").unwrap();
        writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, {};", dtype).unwrap();
        writeln!(w, "    add.u32 %r_qsm, %r_qsm, 4;").unwrap();
        writeln!(w, "    add.u32 %r_d, %r_d, 1;").unwrap();
        writeln!(w, "    bra {};", l("DOT")).unwrap();
        writeln!(w, "{}:", l("DOTD")).unwrap();
        writeln!(w, "    mul.f32 %f_dot, %f_dot, {};", inv_sqrt_hd).unwrap();
        writeln!(w, "    mul.lo.u32 %r_t3, %r_tid, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t3, %r_t3, %r_sbase;").unwrap();
        writeln!(w, "    st.shared.f32 [%r_t3+{}], %f_dot;", scores_off).unwrap();
        writeln!(w, "{}:", l("SCD")).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        emit_softmax_pass2(w, &l("SM"), "%r_tcnt", &log2e, rescale_off, scores_off);
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w, "    // pass 3: rescale accumulator, add P*V; thread d owns out[d]").unwrap();
        writeln!(w, "    setp.ge.u32 %p_nd, %r_tid, {};", hd).unwrap();
        writeln!(w, "    @%p_nd bra {};", l("ACT")).unwrap();
        writeln!(w, "    ld.shared.f32 %f_rs2, [%r_sbase+{}];", rescale_off).unwrap();
        writeln!(w, "    mul.f32 %f_acc, %f_acc, %f_rs2;").unwrap();
        writeln!(w, "    add.u32 %r_g0, %r_slotbase, %r_tile;").unwrap();
        emit_pv_accumulate(w, &l("ACC"), &l("ACT"), "%r_tcnt", token_stride_bytes, dtype, scores_off);
        writeln!(w, "{}:", l("ACT")).unwrap();
        writeln!(w, "    // scores SMEM rewritten next tile").unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w, "    add.u32 %r_tile, %r_tile, {};", TILE).unwrap();
        writeln!(w, "    bra {};", l("TILE")).unwrap();
        writeln!(w, "{}:", l("PREFIX_END")).unwrap();
        writeln!(w).unwrap();
        writeln!(
            w,
            "    // tree tile: the {} draft rows at pool positions seq_len..;",
            nn
        )
        .unwrap();
        writeln!(w, "    // disallowed nodes score -inf (exp -> 0 in the softmax)").unwrap();
        writeln!(w, "    setp.ge.u32 %p_val, %r_tid, {};", nn).unwrap();
        writeln!(w, "    @%p_val bra {};", l("XSCD")).unwrap();
        writeln!(w, "    add.u32 %r_tok, %r_seqlen, %r_tid;").unwrap();
        writeln!(w, "    add.u32 %r_g, %r_slotbase, %r_tok;").unwrap();
        writeln!(w, "    mul.wide.u32 %rd_koff, %r_g, {};", token_stride_bytes).unwrap();
        writeln!(w, "    add.u64 %rd_kaddr, %rd_kplane, %rd_koff;").unwrap();
        writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, %rd_hoff;").unwrap();
        writeln!(w, "    mov.f32 %f_dot, {};", zero).unwrap();
        writeln!(w, "    mov.u32 %r_d, 0;").unwrap();
        writeln!(w, "    mov.u32 %r_qsm, %r_sbase;").unwrap();
        writeln!(w, "{}:", l("XDOT")).unwrap();
        writeln!(w, "    setp.ge.u32 %p_d, %r_d, {};", hd).unwrap();
        writeln!(w, "    @%p_d bra {};", l("XDOTD")).unwrap();
        writeln!(w, "    ld.global.b16 %h_k, [%rd_kaddr];").unwrap();
        writeln!(w, "    cvt.f32.f16 %f_k, %h_k;").unwrap();
        writeln!(w, "    ld.shared.f32 %f_q, [%r_qsm];").unwrap();
        writeln!(w, "    fma.rn.f32 %f_dot, %f_k, %f_q, %f_dot;").unwrap();
        writeln!(w, "    add.u64 %rd_kaddr, %rd_kaddr, {};", dtype).unwrap();
        writeln!(w, "    add.u32 %r_qsm, %r_qsm, 4;").unwrap();
        writeln!(w, "    add.u32 %r_d, %r_d, 1;").unwrap();
        writeln!(w, "    bra {};", l("XDOT")).unwrap();
        writeln!(w, "{}:", l("XDOTD")).unwrap();
        writeln!(w, "    mul.f32 %f_dot, %f_dot, {};", inv_sqrt_hd).unwrap();
        writeln!(w, "    // node row {}'s baked ancestor mask, bit = this thread's node", i).unwrap();
        writeln!(w, "    mov.u64 %rd_mask, 0x{:016X};", mask_row).unwrap();
        writeln!(w, "    shr.b64 %rd_mb, %rd_mask, %r_tid;").unwrap();
        writeln!(w, "    and.b64 %rd_mb, %rd_mb, 1;").unwrap();
        writeln!(w, "    setp.ne.u64 %p_m, %rd_mb, 0;").unwrap();
        writeln!(w, "    @%p_m bra {};", l("XMOK")).unwrap();
        writeln!(w, "    mov.f32 %f_dot, {};", neg_inf).unwrap();
        writeln!(w, "{}:", l("XMOK")).unwrap();
        writeln!(w, "    mul.lo.u32 %r_t3, %r_tid, 4;").unwrap();
        writeln!(w, "    add.u32 %r_t3, %r_t3, %r_sbase;").unwrap();
        writeln!(w, "    st.shared.f32 [%r_t3+{}], %f_dot;", scores_off).unwrap();
        writeln!(w, "{}:", l("XSCD")).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w, "    mov.u32 %r_tcnt, {};", nn).unwrap();
        emit_softmax_pass2(w, &l("XSM"), "%r_tcnt", &log2e, rescale_off, scores_off);
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w, "    setp.ge.u32 %p_nd, %r_tid, {};", hd).unwrap();
        writeln!(w, "    @%p_nd bra {};", l("XACT")).unwrap();
        writeln!(w, "    ld.shared.f32 %f_rs2, [%r_sbase+{}];", rescale_off).unwrap();
        writeln!(w, "    mul.f32 %f_acc, %f_acc, %f_rs2;").unwrap();
        writeln!(w, "    add.u32 %r_g0, %r_slotbase, %r_seqlen;").unwrap();
        emit_pv_accumulate(w, &l("XACC"), &l("XACT"), "%r_tcnt", token_stride_bytes, dtype, scores_off);
        writeln!(w, "{}:", l("XACT")).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w).unwrap();
        writeln!(w, "    // node {} output: thread 0 publishes l, thread d stores out[d]", i).unwrap();
        writeln!(w, "    setp.ne.u32 %p_t1, %r_tid, 0;").unwrap();
        writeln!(w, "    @%p_t1 bra {};", l("LPUB")).unwrap();
        writeln!(w, "    st.shared.f32 [%r_sbase+{}], %f_l;", l_off).unwrap();
        writeln!(w, "{}:", l("LPUB")).unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w, "    setp.ge.u32 %p_no, %r_tid, {};", hd).unwrap();
        writeln!(w, "    @%p_no bra {};", l("END")).unwrap();
        writeln!(w, "    ld.shared.f32 %f_lf, [%r_sbase+{}];", l_off).unwrap();
        writeln!(w, "    // self bit guarantees l > 0; keep the guard (house style)").unwrap();
        writeln!(w, "    mov.f32 %f_o, {};", zero).unwrap();
        writeln!(w, "    setp.gt.f32 %p_lz, %f_lf, {};", zero).unwrap();
        writeln!(w, "    @!%p_lz bra {};", l("STO")).unwrap();
        writeln!(w, "    div.rn.f32 %f_o, %f_acc, %f_lf;").unwrap();
        writeln!(w, "{}:", l("STO")).unwrap();
        writeln!(w, "    mul.lo.u32 %r_t4, %r_head, {};", hd).unwrap();
        writeln!(w, "    add.u32 %r_t4, %r_t4, {};", i * nh * hd).unwrap();
        writeln!(w, "    add.u32 %r_t4, %r_t4, %r_tid;").unwrap();
        writeln!(w, "    mul.lo.u32 %r_t4, %r_t4, 4;").unwrap();
        writeln!(w, "    cvt.u64.u32 %rd_t3, %r_t4;").unwrap();
        writeln!(w, "    add.u64 %rd_t3, %rd_out, %rd_t3;").unwrap();
        writeln!(w, "    st.global.f32 [%rd_t3], %f_o;").unwrap();
        writeln!(w, "{}:", l("END")).unwrap();
        writeln!(w, "    // Q/scores SMEM reused by the next node row").unwrap();
        writeln!(w, "    bar.sync 0;").unwrap();
        writeln!(w).unwrap();
    }
    writeln!(w, "    ret;").unwrap();
    writeln!(w, "}}").unwrap();

    let meta = SpecKernelMeta {
        kernel_name: VERIFY_KERNEL_NAME.to_string(),
        smem_bytes,
        block_dim: VERIFY_BLOCK_DIM,
        grid_dim_is_n_heads: true,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit_verify_attention`].
pub fn emit_verify_attention_ptx(cfg: &VerifyAttentionConfig) -> String {
    emit_verify_attention(cfg).0
}

/// Pass 2 of the flash-decode tile: thread-0 serial online softmax
/// (identical algorithm to `cfie_decode_attention`).
fn emit_softmax_pass2(
    w: &mut String,
    label: &str,
    tcnt_reg: &str,
    log2e: &str,
    rescale_off: u32,
    scores_off: u32,
) {
    writeln!(w, "    // pass 2: online softmax, thread 0 serial over the tile").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra {}_DONE;", label).unwrap();
    writeln!(w, "    mov.f32 %f_tm, %f_m;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_sp, %r_sbase;").unwrap();
    writeln!(w, "{}_MAX:", label).unwrap();
    writeln!(w, "    setp.ge.u32 %p_j, %r_j, {};", tcnt_reg).unwrap();
    writeln!(w, "    @%p_j bra {}_MAXD;", label).unwrap();
    writeln!(w, "    ld.shared.f32 %f_s, [%r_sp+{}];", scores_off).unwrap();
    writeln!(w, "    max.f32 %f_tm, %f_tm, %f_s;").unwrap();
    writeln!(w, "    add.u32 %r_sp, %r_sp, 4;").unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra {}_MAX;", label).unwrap();
    writeln!(w, "{}_MAXD:", label).unwrap();
    writeln!(w, "    // rescale = exp(m_old - m_new); exp(-inf) = 0 on first tile").unwrap();
    writeln!(w, "    sub.f32 %f_t1, %f_m, %f_tm;").unwrap();
    writeln!(w, "    mul.f32 %f_t1, %f_t1, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f_rs, %f_t1;").unwrap();
    writeln!(w, "    mul.f32 %f_l, %f_l, %f_rs;").unwrap();
    writeln!(w, "    mov.f32 %f_m, %f_tm;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_sp, %r_sbase;").unwrap();
    writeln!(w, "{}_P:", label).unwrap();
    writeln!(w, "    setp.ge.u32 %p_j, %r_j, {};", tcnt_reg).unwrap();
    writeln!(w, "    @%p_j bra {}_PD;", label).unwrap();
    writeln!(w, "    ld.shared.f32 %f_s, [%r_sp+{}];", scores_off).unwrap();
    writeln!(w, "    sub.f32 %f_s, %f_s, %f_m;").unwrap();
    writeln!(w, "    mul.f32 %f_s, %f_s, {};", log2e).unwrap();
    writeln!(w, "    ex2.approx.f32 %f_s, %f_s;").unwrap();
    writeln!(w, "    st.shared.f32 [%r_sp+{}], %f_s;", scores_off).unwrap();
    writeln!(w, "    add.f32 %f_l, %f_l, %f_s;").unwrap();
    writeln!(w, "    add.u32 %r_sp, %r_sp, 4;").unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra {}_P;", label).unwrap();
    writeln!(w, "{}_PD:", label).unwrap();
    writeln!(w, "    st.shared.f32 [%r_sbase+{}], %f_rs;", rescale_off).unwrap();
    writeln!(w, "{}_DONE:", label).unwrap();
}

/// Pass-3 P*V accumulation body: assumes `%r_g0` already holds the
/// tile's first global token; masked entries carry p == 0 so their V
/// rows contribute nothing.
fn emit_pv_accumulate(
    w: &mut String,
    label: &str,
    exit_label: &str,
    tcnt_reg: &str,
    token_stride_bytes: u64,
    dtype: u64,
    scores_off: u32,
) {
    writeln!(w, "    mul.wide.u32 %rd_voff, %r_g0, {};", token_stride_bytes).unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vplane, %rd_voff;").unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, %rd_hoff;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t2, %r_tid, {};", dtype).unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, %rd_t2;").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "    mov.u32 %r_sp, %r_sbase;").unwrap();
    writeln!(w, "{}:", label).unwrap();
    writeln!(w, "    setp.ge.u32 %p_j2, %r_j, {};", tcnt_reg).unwrap();
    writeln!(w, "    @%p_j2 bra {};", exit_label).unwrap();
    writeln!(w, "    ld.shared.f32 %f_p, [%r_sp+{}];", scores_off).unwrap();
    writeln!(w, "    ld.global.b16 %h_v, [%rd_vaddr];").unwrap();
    writeln!(w, "    cvt.f32.f16 %f_v, %h_v;").unwrap();
    writeln!(w, "    fma.rn.f32 %f_acc, %f_p, %f_v, %f_acc;").unwrap();
    writeln!(w, "    add.u64 %rd_vaddr, %rd_vaddr, {};", token_stride_bytes).unwrap();
    writeln!(w, "    add.u32 %r_sp, %r_sp, 4;").unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra {};", label).unwrap();
}

// ---------------------------------------------------------------------------
// G13: rejection-sampling epilogue
// ---------------------------------------------------------------------------

/// xorshift64* state advance + [0,1) draw — the PRNG idiom shared with
/// `cfie_sample_ptx` (state = post-shift value, output = state * M,
/// r = top 24 bits over 2^24).
fn emit_prng_draw(w: &mut String, two_neg24: &str) {
    writeln!(w, "    shr.b64 %rd_t0, %rd_x, 12;").unwrap();
    writeln!(w, "    xor.b64 %rd_x, %rd_x, %rd_t0;").unwrap();
    writeln!(w, "    shl.b64 %rd_t0, %rd_x, 25;").unwrap();
    writeln!(w, "    xor.b64 %rd_x, %rd_x, %rd_t0;").unwrap();
    writeln!(w, "    shr.b64 %rd_t0, %rd_x, 27;").unwrap();
    writeln!(w, "    xor.b64 %rd_x, %rd_x, %rd_t0;").unwrap();
    writeln!(w, "    mov.u64 %rd_t0, 0x2545F4914F6CDD1D;").unwrap();
    writeln!(w, "    mul.lo.u64 %rd_t1, %rd_x, %rd_t0;").unwrap();
    writeln!(w, "    shr.b64 %rd_t1, %rd_t1, 40;").unwrap();
    writeln!(w, "    cvt.u32.u64 %r_t0, %rd_t1;").unwrap();
    writeln!(w, "    cvt.rn.f32.u32 %f_r, %r_t0;").unwrap();
    writeln!(w, "    mul.f32 %f_r, %f_r, {};", two_neg24).unwrap();
}

/// Emit the rejection-sampling kernel (paper step 3).
///
/// Launch shape: grid = 1, block = 32; the walk is serial on thread 0
/// (correctness first), all other threads exit immediately — no SMEM,
/// so no barriers are required.  `out_accepted` (i32) = number of
/// accepted draft tokens; `out_correction_token` = the residual sample
/// at the first rejection, or the `0xFFFFFFFF` sentinel when all K
/// accept (the host then samples the K+1-th token normally via the
/// fused sampler).
pub fn emit_rejection_kernel(cfg: &RejectionConfig) -> (String, SpecKernelMeta) {
    assert!(
        cfg.k_tokens >= 1 && cfg.k_tokens <= 32,
        "k_tokens must be in 1..=32 (matches the serve-side clamp)"
    );
    assert!(cfg.vocab_size >= 1, "vocab_size must be >= 1");
    assert!(
        (cfg.k_tokens as u64) * (cfg.vocab_size as u64) <= u32::MAX as u64,
        "k_tokens * vocab_size must fit in u32 (row index arithmetic)"
    );

    let k = cfg.k_tokens;
    let vocab = cfg.vocab_size;
    let zero = f32_imm(0.0);
    let two_neg24 = f32_imm(1.0 / 16_777_216.0);

    let mut p = String::new();
    let w = &mut p;

    writeln!(w, "//").unwrap();
    writeln!(
        w,
        "// {} - CFIE speculative rejection-sampling epilogue.",
        REJECT_KERNEL_NAME
    )
    .unwrap();
    writeln!(
        w,
        "// Serial thread-0 walk over K={} draft positions, vocab={}.",
        k, vocab
    )
    .unwrap();
    writeln!(w, "// Accept j iff r < p_target[j][tok_j] / p_draft[j];").unwrap();
    writeln!(w, "// p_draft <= 0 rejects (division guard).  First rejection").unwrap();
    writeln!(w, "// samples the Leviathan residual max(p_target - p_draft*").unwrap();
    writeln!(w, "// [x == tok_j], 0) renormalised; empty residual falls back").unwrap();
    writeln!(w, "// to the drafted token (then the argmax of p_target).").unwrap();
    writeln!(w, "// All-accept writes the 0xFFFFFFFF correction sentinel.").unwrap();
    writeln!(
        w,
        "// PRNG: xorshift64* over rng_seed - deterministic given seed (M46)."
    )
    .unwrap();
    writeln!(
        w,
        "// Host rollback: linear chain rollback(slot, K - accepted);"
    )
    .unwrap();
    writeln!(
        w,
        "// tree method: rollback(slot, num_nodes) + re-append accepted path."
    )
    .unwrap();
    writeln!(w, "//").unwrap();
    writeln!(w, ".version {}", crate::gpu_specs::ptx_isa_for_sm(cfg.sm_version)).unwrap();
    writeln!(w, ".target sm_{}", cfg.sm_version).unwrap();
    writeln!(w, ".address_size 64").unwrap();
    writeln!(w).unwrap();
    writeln!(w, ".visible .entry {}(", REJECT_KERNEL_NAME).unwrap();
    writeln!(w, "    .param .u64 target_probs_ptr,").unwrap();
    writeln!(w, "    .param .u64 draft_probs_ptr,").unwrap();
    writeln!(w, "    .param .u64 draft_tokens_ptr,").unwrap();
    writeln!(w, "    .param .u64 rng_seed,").unwrap();
    writeln!(w, "    .param .u64 out_accepted_ptr,").unwrap();
    writeln!(w, "    .param .u64 out_correction_token_ptr").unwrap();
    writeln!(w, ")").unwrap();
    writeln!(w, "{{").unwrap();
    writeln!(w, "    .reg .pred %p_a, %p_b, %p_c, %p_d, %p_t0;").unwrap();
    writeln!(w, "    .reg .f32 %f_r, %f_d, %f_t, %f_ratio, %f_tot, %f_tgt, %f_cum;").unwrap();
    writeln!(w, "    .reg .u32 %r_tid, %r_j, %r_v, %r_tok, %r_sel, %r_t0, %r_t1;").unwrap();
    writeln!(
        w,
        "    .reg .u64 %rd_tp, %rd_dp, %rd_dt, %rd_seed, %rd_oa, %rd_oc, %rd_x, %rd_row, %rd_t0, %rd_t1, %rd_t2;"
    )
    .unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    ld.param.u64 %rd_tp, [target_probs_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_dp, [draft_probs_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_dt, [draft_tokens_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_seed, [rng_seed];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_oa, [out_accepted_ptr];").unwrap();
    writeln!(w, "    ld.param.u64 %rd_oc, [out_correction_token_ptr];").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // serial kernel: only thread 0 works; no SMEM, no barriers").unwrap();
    writeln!(w, "    mov.u32 %r_tid, %tid.x;").unwrap();
    writeln!(w, "    setp.ne.u32 %p_t0, %r_tid, 0;").unwrap();
    writeln!(w, "    @%p_t0 bra EXIT;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // xorshift64* PRNG: deterministic given rng_seed (M46)").unwrap();
    writeln!(w, "    mov.u64 %rd_x, %rd_seed;").unwrap();
    writeln!(w, "    setp.ne.u64 %p_a, %rd_x, 0;").unwrap();
    writeln!(w, "    @%p_a bra SEEDED;").unwrap();
    writeln!(w, "    // zero seed would be a fixed point; substitute golden gamma").unwrap();
    writeln!(w, "    mov.u64 %rd_x, 0x9E3779B97F4A7C15;").unwrap();
    writeln!(w, "SEEDED:").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "    // acceptance walk over the K draft positions").unwrap();
    writeln!(w, "    mov.u32 %r_j, 0;").unwrap();
    writeln!(w, "ACC_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_j, {};", k).unwrap();
    writeln!(w, "    @%p_a bra ALL_ACCEPT;").unwrap();
    emit_prng_draw(w, &two_neg24);
    writeln!(w, "    mul.wide.u32 %rd_t0, %r_j, 4;").unwrap();
    writeln!(w, "    add.u64 %rd_t2, %rd_dp, %rd_t0;").unwrap();
    writeln!(w, "    ld.global.f32 %f_d, [%rd_t2];").unwrap();
    writeln!(w, "    add.u64 %rd_t2, %rd_dt, %rd_t0;").unwrap();
    writeln!(w, "    ld.global.u32 %r_tok, [%rd_t2];").unwrap();
    writeln!(w, "    // p_draft <= 0 => reject (division guard)").unwrap();
    writeln!(w, "    setp.gt.f32 %p_b, %f_d, {};", zero).unwrap();
    writeln!(w, "    @!%p_b bra REJECT;").unwrap();
    writeln!(w, "    // p_target = target_probs[j * vocab + tok_j]").unwrap();
    writeln!(w, "    mul.lo.u32 %r_t1, %r_j, {};", vocab).unwrap();
    writeln!(w, "    add.u32 %r_t1, %r_t1, %r_tok;").unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t0, %r_t1, 4;").unwrap();
    writeln!(w, "    add.u64 %rd_t2, %rd_tp, %rd_t0;").unwrap();
    writeln!(w, "    ld.global.f32 %f_t, [%rd_t2];").unwrap();
    writeln!(w, "    div.rn.f32 %f_ratio, %f_t, %f_d;").unwrap();
    writeln!(w, "    setp.lt.f32 %p_c, %f_r, %f_ratio;").unwrap();
    writeln!(w, "    @!%p_c bra REJECT;").unwrap();
    writeln!(w, "    add.u32 %r_j, %r_j, 1;").unwrap();
    writeln!(w, "    bra ACC_LOOP;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "ALL_ACCEPT:").unwrap();
    writeln!(w, "    mov.u32 %r_t0, {};", k).unwrap();
    writeln!(w, "    st.global.u32 [%rd_oa], %r_t0;").unwrap();
    writeln!(w, "    // sentinel: host samples the K+1-th token normally").unwrap();
    writeln!(w, "    mov.u32 %r_t0, 4294967295;").unwrap();
    writeln!(w, "    st.global.u32 [%rd_oc], %r_t0;").unwrap();
    writeln!(w, "    bra EXIT;").unwrap();
    writeln!(w).unwrap();
    writeln!(w, "REJECT:").unwrap();
    writeln!(w, "    // accepted = j; correction from row j's Leviathan residual").unwrap();
    writeln!(w, "    st.global.u32 [%rd_oa], %r_j;").unwrap();
    writeln!(w, "    // negative p_draft is garbage input; clamp for the residual").unwrap();
    writeln!(w, "    max.f32 %f_d, %f_d, {};", zero).unwrap();
    writeln!(w, "    mul.lo.u32 %r_t1, %r_j, {};", vocab).unwrap();
    writeln!(w, "    mul.wide.u32 %rd_t0, %r_t1, 4;").unwrap();
    writeln!(w, "    add.u64 %rd_row, %rd_tp, %rd_t0;").unwrap();
    writeln!(w, "    // pass 1: total residual mass").unwrap();
    writeln!(w, "    mov.f32 %f_tot, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_v, 0;").unwrap();
    writeln!(w, "    mov.u64 %rd_t2, %rd_row;").unwrap();
    writeln!(w, "TOT_LOOP:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_v, {};", vocab).unwrap();
    writeln!(w, "    @%p_a bra TOT_DONE;").unwrap();
    writeln!(w, "    ld.global.f32 %f_t, [%rd_t2];").unwrap();
    writeln!(w, "    setp.ne.u32 %p_b, %r_v, %r_tok;").unwrap();
    writeln!(w, "    @%p_b bra TOT_Q;").unwrap();
    writeln!(w, "    sub.f32 %f_t, %f_t, %f_d;").unwrap();
    writeln!(w, "TOT_Q:").unwrap();
    writeln!(w, "    max.f32 %f_t, %f_t, {};", zero).unwrap();
    writeln!(w, "    add.f32 %f_tot, %f_tot, %f_t;").unwrap();
    writeln!(w, "    add.u64 %rd_t2, %rd_t2, 4;").unwrap();
    writeln!(w, "    add.u32 %r_v, %r_v, 1;").unwrap();
    writeln!(w, "    bra TOT_LOOP;").unwrap();
    writeln!(w, "TOT_DONE:").unwrap();
    writeln!(w, "    // empty residual => target mass sat on the drafted token;").unwrap();
    writeln!(w, "    // fall back to it (the argmax of p_target)").unwrap();
    writeln!(w, "    setp.gt.f32 %p_a, %f_tot, {};", zero).unwrap();
    writeln!(w, "    @%p_a bra RESAMPLE;").unwrap();
    writeln!(w, "    st.global.u32 [%rd_oc], %r_tok;").unwrap();
    writeln!(w, "    bra EXIT;").unwrap();
    writeln!(w, "RESAMPLE:").unwrap();
    emit_prng_draw(w, &two_neg24);
    writeln!(w, "    mul.f32 %f_tgt, %f_r, %f_tot;").unwrap();
    writeln!(w, "    // walk the residual CDF; last positive entry is the").unwrap();
    writeln!(w, "    // fp-drift fallback (total > 0 guarantees one exists)").unwrap();
    writeln!(w, "    mov.u32 %r_sel, %r_tok;").unwrap();
    writeln!(w, "    mov.f32 %f_cum, {};", zero).unwrap();
    writeln!(w, "    mov.u32 %r_v, 0;").unwrap();
    writeln!(w, "    mov.u64 %rd_t2, %rd_row;").unwrap();
    writeln!(w, "WALK:").unwrap();
    writeln!(w, "    setp.ge.u32 %p_a, %r_v, {};", vocab).unwrap();
    writeln!(w, "    @%p_a bra WALK_DONE;").unwrap();
    writeln!(w, "    ld.global.f32 %f_t, [%rd_t2];").unwrap();
    writeln!(w, "    setp.ne.u32 %p_b, %r_v, %r_tok;").unwrap();
    writeln!(w, "    @%p_b bra W_Q;").unwrap();
    writeln!(w, "    sub.f32 %f_t, %f_t, %f_d;").unwrap();
    writeln!(w, "W_Q:").unwrap();
    writeln!(w, "    max.f32 %f_t, %f_t, {};", zero).unwrap();
    writeln!(w, "    add.f32 %f_cum, %f_cum, %f_t;").unwrap();
    writeln!(w, "    setp.gt.f32 %p_c, %f_t, {};", zero).unwrap();
    writeln!(w, "    @!%p_c bra W_NEXT;").unwrap();
    writeln!(w, "    mov.u32 %r_sel, %r_v;").unwrap();
    writeln!(w, "    setp.ge.f32 %p_d, %f_cum, %f_tgt;").unwrap();
    writeln!(w, "    @%p_d bra WALK_DONE;").unwrap();
    writeln!(w, "W_NEXT:").unwrap();
    writeln!(w, "    add.u64 %rd_t2, %rd_t2, 4;").unwrap();
    writeln!(w, "    add.u32 %r_v, %r_v, 1;").unwrap();
    writeln!(w, "    bra WALK;").unwrap();
    writeln!(w, "WALK_DONE:").unwrap();
    writeln!(w, "    st.global.u32 [%rd_oc], %r_sel;").unwrap();
    writeln!(w, "EXIT:").unwrap();
    writeln!(w, "    ret;").unwrap();
    writeln!(w, "}}").unwrap();

    let meta = SpecKernelMeta {
        kernel_name: REJECT_KERNEL_NAME.to_string(),
        smem_bytes: 0,
        block_dim: REJECT_BLOCK_DIM,
        grid_dim_is_n_heads: false,
    };
    (p, meta)
}

/// PTX-only convenience wrapper around [`emit_rejection_kernel`].
pub fn emit_rejection_ptx(cfg: &RejectionConfig) -> String {
    emit_rejection_kernel(cfg).0
}

