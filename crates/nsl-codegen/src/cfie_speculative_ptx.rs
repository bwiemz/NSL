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
//!
//! ## KIR (roadmap A2 step 9)
//!
//! The rejection kernel is built as [`KernelIR`] ([`build_rejection`]) and
//! lowered by `nsl_kir`'s printer; `tests/cfie_speculative_kir_equivalence.rs`
//! runs it against the frozen hand emitter on a PTX interpreter and
//! requires the same output bits, and the CPU reference's answer exactly.
//! It targets the KIR floor (`sm_70`), so [`RejectionConfig`] has no
//! `sm_version`. The verify attention kernel is still hand-assembled; it
//! moves in the next slice.

use crate::backend_ptx::lower_kir_to_ptx;
use crate::cfie_decode_attention::{at, cmp, konst, load, op2, ptr};
use crate::cfie_speculative::TreeMask;
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator,
    KirType, VarId,
};
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

/// Compile-time configuration for the rejection kernel. No
/// `sm_version`: the kernel is KIR and targets the backend floor
/// (roadmap A2 step 9).
#[derive(Debug, Clone)]
pub struct RejectionConfig {
    /// Draft tokens per speculative step (1..=32).
    pub k_tokens: u32,
    pub vocab_size: u32,
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

/// One xorshift64* step and its [0,1) draw — the PRNG idiom shared with
/// `cfie_sample_ptx`, in its order: the state advances by three shift-xors
/// (`x ^= x >> 12; x ^= x << 25; x ^= x >> 27`) and becomes the new state;
/// the output is `state * M`, and `r` is its top 24 bits over 2^24, exact
/// in an f32 mantissa. Returns `(state, r)`.
fn build_prng_draw(b: &mut KirBuilder, x: VarId) -> (VarId, VarId) {
    use KirType::{F32, U32, U64};

    let mut x = x;
    for (shift, op) in [
        (12u32, KirOp::Shr as fn(VarId, VarId, VarId) -> KirOp),
        (25, KirOp::Shl),
        (27, KirOp::Shr),
    ] {
        let amount = konst(b, ConstValue::U32(shift));
        let shifted = op2(b, U64, op, x, amount);
        x = op2(b, U64, KirOp::Xor, x, shifted);
    }
    let m = konst(b, ConstValue::U64(0x2545_F491_4F6C_DD1D));
    let out = op2(b, U64, KirOp::Mul, x, m);
    let forty = konst(b, ConstValue::U32(40));
    let top = op2(b, U64, KirOp::Shr, out, forty);
    let top32 = b.new_typed_var(U32);
    b.emit(KirOp::Cast(top32, top, U32));
    let wide = b.new_typed_var(F32);
    b.emit(KirOp::Cast(wide, top32, F32));
    let two_neg24 = konst(b, ConstValue::F32(1.0 / 16_777_216.0));
    let r = op2(b, F32, KirOp::Mul, wide, two_neg24);
    (x, r)
}

/// The rejection-sampling kernel for `cfg`, as KIR (roadmap A2 step 9).
///
/// A serial walk on thread 0 (every other thread returns at once), no
/// shared memory, so no barriers. The branches and the order of every
/// floating-point operation are the hand kernel's: the acceptance walk
/// (`r < p_target / p_draft`, a non-positive `p_draft` rejecting before
/// the division), then at the first rejection the residual's total mass,
/// the empty-residual fallback to the drafted token, and the CDF walk
/// whose last positive entry is the fp-drift fallback. Loop-carried
/// values — the PRNG state, the loop cursors, the running mass and the
/// selected token — are block parameters.
pub fn build_rejection(cfg: &RejectionConfig) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, U32, U64};

    validate_rejection(cfg);
    let mut b = KirBuilder::new(REJECT_KERNEL_NAME);
    let target_probs = b.add_param("target_probs_ptr", ptr(F32, Global), Global);
    let draft_probs = b.add_param("draft_probs_ptr", ptr(F32, Global), Global);
    let draft_tokens = b.add_param("draft_tokens_ptr", ptr(U32, Global), Global);
    let rng_seed = b.add_param("rng_seed", U64, Global);
    let out_accepted = b.add_param("out_accepted_ptr", ptr(U32, Global), Global);
    let out_correction = b.add_param("out_correction_token_ptr", ptr(U32, Global), Global);
    b.set_workgroup_size([REJECT_BLOCK_DIM, 1, 1]);

    let entry = b.new_block();
    let seed = b.new_block();
    let acc_head = b.new_block();
    let acc_body = b.new_block();
    let acc_ratio = b.new_block();
    let acc_next = b.new_block();
    let all_accept = b.new_block();
    let reject = b.new_block();
    let tot_head = b.new_block();
    let tot_body = b.new_block();
    let tot_sub = b.new_block();
    let tot_clamp = b.new_block();
    let tot_done = b.new_block();
    let fallback = b.new_block();
    let resample = b.new_block();
    let walk_head = b.new_block();
    let walk_body = b.new_block();
    let walk_sub = b.new_block();
    let walk_clamp = b.new_block();
    let walk_select = b.new_block();
    let walk_next = b.new_block();
    let walk_done = b.new_block();
    let exit = b.new_block();

    let j = b.add_block_param(acc_head, U32);
    let x = b.add_block_param(acc_head, U64);
    let v = b.add_block_param(tot_head, U32);
    let tot = b.add_block_param(tot_head, F32);
    let tot_t = b.add_block_param(tot_clamp, F32);
    let wv = b.add_block_param(walk_head, U32);
    let cum = b.add_block_param(walk_head, F32);
    let sel = b.add_block_param(walk_head, U32);
    let walk_t = b.add_block_param(walk_clamp, F32);
    let next_cum = b.add_block_param(walk_next, F32);
    let next_sel = b.add_block_param(walk_next, U32);
    let chosen = b.add_block_param(walk_done, U32);

    // Serial kernel: only thread 0 works.
    b.set_block(entry);
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let k = konst(&mut b, ConstValue::U32(cfg.k_tokens));
    let vocab = konst(&mut b, ConstValue::U32(cfg.vocab_size));
    let f_zero = konst(&mut b, ConstValue::F32(0.0));
    let not_zero = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_zero, KirEdge::to(exit), KirEdge::to(seed)));

    // xorshift64*: a zero seed would be a fixed point; substitute the
    // golden gamma.
    b.set_block(seed);
    let zero64 = konst(&mut b, ConstValue::U64(0));
    let golden = konst(&mut b, ConstValue::U64(0x9E37_79B9_7F4A_7C15));
    let seeded = cmp(&mut b, rng_seed, zero64, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(
        seeded,
        KirEdge::with(acc_head, vec![zero, rng_seed]),
        KirEdge::with(acc_head, vec![zero, golden]),
    ));

    // Acceptance walk over the K draft positions.
    b.set_block(acc_head);
    let walked = cmp(&mut b, j, k, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(walked, KirEdge::to(all_accept), KirEdge::to(acc_body)));

    b.set_block(acc_body);
    let (x_next, r) = build_prng_draw(&mut b, x);
    let d_addr = at(&mut b, F32, Global, draft_probs, j);
    let d = load(&mut b, F32, d_addr, Global);
    let tok_addr = at(&mut b, U32, Global, draft_tokens, j);
    let tok = load(&mut b, U32, tok_addr, Global);
    // p_draft <= 0 rejects (division guard).
    let positive = cmp(&mut b, d, f_zero, CmpOp::Gt);
    b.terminate(KirTerminator::CondBranch(positive, KirEdge::to(acc_ratio), KirEdge::to(reject)));

    // p_target = target_probs[j * vocab + tok_j].
    b.set_block(acc_ratio);
    let row_j = op2(&mut b, U32, KirOp::Mul, j, vocab);
    let t_index = op2(&mut b, U32, KirOp::Add, row_j, tok);
    let t_addr = at(&mut b, F32, Global, target_probs, t_index);
    let t = load(&mut b, F32, t_addr, Global);
    let ratio = op2(&mut b, F32, KirOp::Div, t, d);
    let accept = cmp(&mut b, r, ratio, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(accept, KirEdge::to(acc_next), KirEdge::to(reject)));

    b.set_block(acc_next);
    let j_next = op2(&mut b, U32, KirOp::Add, j, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(acc_head, vec![j_next, x_next])));

    // All K accepted: the sentinel tells the host to sample the K+1-th
    // token normally.
    b.set_block(all_accept);
    b.emit(KirOp::Store(out_accepted, k, Global));
    let sentinel = konst(&mut b, ConstValue::U32(u32::MAX));
    b.emit(KirOp::Store(out_correction, sentinel, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    // accepted = j; the correction comes from row j's Leviathan residual.
    // A negative p_draft is garbage input; clamp it for the residual.
    b.set_block(reject);
    b.emit(KirOp::Store(out_accepted, j, Global));
    let d_clamped = op2(&mut b, F32, KirOp::Max, d, f_zero);
    let row = op2(&mut b, U32, KirOp::Mul, j, vocab);
    b.terminate(KirTerminator::Branch(KirEdge::with(tot_head, vec![zero, f_zero])));

    // The residual at vocab entry `at_v`: p_target, less the clamped
    // p_draft at the drafted token, clamped at 0. Entered from the current
    // block; leaves the builder in `clamp` with the entry in its param.
    let residual = |b: &mut KirBuilder, at_v: VarId, sub: BlockId, clamp: BlockId| {
        let index = op2(b, U32, KirOp::Add, row, at_v);
        let addr = at(b, F32, Global, target_probs, index);
        let p = load(b, F32, addr, Global);
        let other = cmp(b, at_v, tok, CmpOp::Ne);
        b.terminate(KirTerminator::CondBranch(
            other,
            KirEdge::with(clamp, vec![p]),
            KirEdge::to(sub),
        ));
        b.set_block(sub);
        let reduced = op2(b, F32, KirOp::Sub, p, d_clamped);
        b.terminate(KirTerminator::Branch(KirEdge::with(clamp, vec![reduced])));
        b.set_block(clamp);
    };

    // Pass 1: the residual's total mass.
    b.set_block(tot_head);
    let tot_walked = cmp(&mut b, v, vocab, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(tot_walked, KirEdge::to(tot_done), KirEdge::to(tot_body)));

    b.set_block(tot_body);
    residual(&mut b, v, tot_sub, tot_clamp);
    let q = op2(&mut b, F32, KirOp::Max, tot_t, f_zero);
    let tot_next = op2(&mut b, F32, KirOp::Add, tot, q);
    let v_next = op2(&mut b, U32, KirOp::Add, v, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(tot_head, vec![v_next, tot_next])));

    // An empty residual means the target mass sat on the drafted token;
    // fall back to it (the argmax of p_target).
    b.set_block(tot_done);
    let has_mass = cmp(&mut b, tot, f_zero, CmpOp::Gt);
    b.terminate(KirTerminator::CondBranch(has_mass, KirEdge::to(resample), KirEdge::to(fallback)));

    b.set_block(fallback);
    b.emit(KirOp::Store(out_correction, tok, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(resample);
    let (_, r2) = build_prng_draw(&mut b, x_next);
    let target = op2(&mut b, F32, KirOp::Mul, r2, tot);
    b.terminate(KirTerminator::Branch(KirEdge::with(walk_head, vec![zero, f_zero, tok])));

    // Walk the residual's CDF; the last positive entry is the fp-drift
    // fallback (a positive total guarantees one exists).
    b.set_block(walk_head);
    let walk_walked = cmp(&mut b, wv, vocab, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        walk_walked,
        KirEdge::with(walk_done, vec![sel]),
        KirEdge::to(walk_body),
    ));

    b.set_block(walk_body);
    residual(&mut b, wv, walk_sub, walk_clamp);
    let wq = op2(&mut b, F32, KirOp::Max, walk_t, f_zero);
    let cum_next = op2(&mut b, F32, KirOp::Add, cum, wq);
    let positive_q = cmp(&mut b, wq, f_zero, CmpOp::Gt);
    b.terminate(KirTerminator::CondBranch(
        positive_q,
        KirEdge::to(walk_select),
        KirEdge::with(walk_next, vec![cum_next, sel]),
    ));

    b.set_block(walk_select);
    let reached = cmp(&mut b, cum_next, target, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(
        reached,
        KirEdge::with(walk_done, vec![wv]),
        KirEdge::with(walk_next, vec![cum_next, wv]),
    ));

    b.set_block(walk_next);
    let wv_next = op2(&mut b, U32, KirOp::Add, wv, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(walk_head, vec![wv_next, next_cum, next_sel])));

    b.set_block(walk_done);
    b.emit(KirOp::Store(out_correction, chosen, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

fn validate_rejection(cfg: &RejectionConfig) {
    assert!(
        cfg.k_tokens >= 1 && cfg.k_tokens <= 32,
        "k_tokens must be in 1..=32 (matches the serve-side clamp)"
    );
    assert!(cfg.vocab_size >= 1, "vocab_size must be >= 1");
    assert!(
        (cfg.k_tokens as u64) * (cfg.vocab_size as u64) <= u32::MAX as u64,
        "k_tokens * vocab_size must fit in u32 (row index arithmetic)"
    );
}

/// The `//` header the hand kernel carried, line for line.
fn rejection_header(cfg: &RejectionConfig) -> String {
    let mut w = String::new();
    writeln!(w, "//").unwrap();
    writeln!(w, "// {} - CFIE speculative rejection-sampling epilogue.", REJECT_KERNEL_NAME).unwrap();
    writeln!(w, "// Serial thread-0 walk over K={} draft positions, vocab={}.", cfg.k_tokens, cfg.vocab_size)
        .unwrap();
    writeln!(w, "// Accept j iff r < p_target[j][tok_j] / p_draft[j];").unwrap();
    writeln!(w, "// p_draft <= 0 rejects (division guard).  First rejection").unwrap();
    writeln!(w, "// samples the Leviathan residual max(p_target - p_draft*").unwrap();
    writeln!(w, "// [x == tok_j], 0) renormalised; empty residual falls back").unwrap();
    writeln!(w, "// to the drafted token (then the argmax of p_target).").unwrap();
    writeln!(w, "// All-accept writes the 0xFFFFFFFF correction sentinel.").unwrap();
    writeln!(w, "// PRNG: xorshift64* over rng_seed - deterministic given seed (M46).").unwrap();
    writeln!(w, "// Host rollback: linear chain rollback(slot, K - accepted);").unwrap();
    writeln!(w, "// tree method: rollback(slot, num_nodes) + re-append accepted path.").unwrap();
    writeln!(w, "//").unwrap();
    w
}

/// Emit the rejection-sampling kernel (paper step 3): build, verify, lower,
/// and prefix the `//` header.
///
/// Launch shape: grid = 1, block = 32; the walk is serial on thread 0
/// (correctness first), all other threads exit immediately — no SMEM,
/// so no barriers are required.  `out_accepted` (i32) = number of
/// accepted draft tokens; `out_correction_token` = the residual sample
/// at the first rejection, or the `0xFFFFFFFF` sentinel when all K
/// accept (the host then samples the K+1-th token normally via the
/// fused sampler). The module targets the KIR floor (`sm_70`), so
/// [`RejectionConfig`] has no `sm_version`. The returned text carries no
/// NUL.
pub fn emit_rejection_kernel(cfg: &RejectionConfig) -> (String, SpecKernelMeta) {
    let ir = build_rejection(cfg);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{} failed KIR verification: {errors:?}", ir.name);
    }
    let module = lower_kir_to_ptx(&ir);
    let module = module.strip_suffix(&[0]).unwrap_or(&module);
    let module = std::str::from_utf8(module).expect("the KIR printer emits ASCII");
    let mut p = rejection_header(cfg);
    p.push_str(module);

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

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

/// CPU reference for the verify kernel.  Layouts: `q` is
/// `[num_nodes][n_heads][head_dim]`; `k`/`v` are one slot's token
/// records `[seq_len + num_nodes][n_kv_heads][head_dim]` — the prefix
/// followed by the appended draft rows (the kernel's host contract).
/// Returns `[num_nodes][n_heads][head_dim]` f32.
pub fn cpu_reference_verify(
    cfg: &VerifyAttentionConfig,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: u32,
) -> Vec<f32> {
    let nn = cfg.num_nodes as usize;
    let nh = cfg.n_heads as usize;
    let nkv = cfg.n_kv_heads as usize;
    let hd = cfg.head_dim as usize;
    let sl = seq_len as usize;
    assert!(nkv >= 1 && nh.is_multiple_of(nkv));
    assert_eq!(cfg.mask_bits.len(), nn, "mask_bits must have one row per node");
    assert_eq!(q.len(), nn * nh * hd, "q must be [num_nodes][n_heads][head_dim]");
    assert_eq!(
        k.len(),
        (sl + nn) * nkv * hd,
        "k must be [seq_len + num_nodes][n_kv_heads][head_dim]"
    );
    assert_eq!(v.len(), k.len(), "v must match k's layout");

    let group = nh / nkv;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; nn * nh * hd];
    for node in 0..nn {
        let mask = cfg.mask_bits[node];
        for h in 0..nh {
            let kvh = h / group;
            let qrow = &q[(node * nh + h) * hd..(node * nh + h) * hd + hd];
            // Allowed token set: the whole committed prefix + the
            // mask-allowed tree nodes.
            let mut idx = Vec::with_capacity(sl + nn);
            idx.extend(0..sl);
            for c in 0..nn {
                if mask & (1u64 << c) != 0 {
                    idx.push(sl + c);
                }
            }
            let mut scores = Vec::with_capacity(idx.len());
            let mut m = f32::NEG_INFINITY;
            for &t in &idx {
                let krow = &k[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
                let dot: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum();
                let s = dot * scale;
                scores.push(s);
                if s > m {
                    m = s;
                }
            }
            let mut l = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - m).exp();
                l += *s;
            }
            for (pos, &t) in idx.iter().enumerate() {
                let p = scores[pos] / l;
                let vrow = &v[(t * nkv + kvh) * hd..(t * nkv + kvh) * hd + hd];
                for d in 0..hd {
                    out[(node * nh + h) * hd + d] += p * vrow[d];
                }
            }
        }
    }
    out
}

/// The kernel's sequential xorshift64* draw, bit-for-bit: state
/// advances via the three shift-xors; output = state * M; r = top 24
/// bits over 2^24 (f32-mantissa exact).  Zero seed substitutes the
/// golden-gamma constant (same guard as `cfie_sample_ptx`).
fn prng_draw(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let out = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    ((out >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

/// CPU mirror of the rejection kernel: same PRNG stepping, same f32
/// operation order, same clamp/fallback rules.  Returns
/// `(accepted, correction_token)`; `correction_token == u32::MAX` iff
/// all K accept.
pub fn cpu_reference_reject(
    cfg: &RejectionConfig,
    target_probs: &[f32],
    draft_probs: &[f32],
    draft_tokens: &[u32],
    seed: u64,
) -> (i32, u32) {
    let k = cfg.k_tokens as usize;
    let vocab = cfg.vocab_size as usize;
    assert_eq!(target_probs.len(), k * vocab, "target_probs must be [k][vocab]");
    assert_eq!(draft_probs.len(), k, "draft_probs must be [k]");
    assert_eq!(draft_tokens.len(), k, "draft_tokens must be [k]");
    assert!(
        draft_tokens.iter().all(|&t| (t as usize) < vocab),
        "draft tokens must index the vocab"
    );

    let mut state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    for j in 0..k {
        let r = prng_draw(&mut state);
        let d = draft_probs[j];
        let tok = draft_tokens[j] as usize;
        let accept = d > 0.0 && r < target_probs[j * vocab + tok] / d;
        if accept {
            continue;
        }
        // First rejection: sample row j's Leviathan residual.
        let d = d.max(0.0);
        let row = &target_probs[j * vocab..(j + 1) * vocab];
        let residual = |vv: usize| -> f32 {
            let p = if vv == tok { row[vv] - d } else { row[vv] };
            p.max(0.0)
        };
        let mut total = 0.0f32;
        for vv in 0..vocab {
            total += residual(vv);
        }
        if total <= 0.0 {
            // Empty residual: target mass sat on the drafted token.
            return (j as i32, tok as u32);
        }
        let target = prng_draw(&mut state) * total;
        let mut sel = tok as u32;
        let mut cum = 0.0f32;
        for vv in 0..vocab {
            let qv = residual(vv);
            cum += qv;
            if qv > 0.0 {
                sel = vv as u32;
                if cum >= target {
                    break;
                }
            }
        }
        return (j as i32, sel);
    }
    (k as i32, u32::MAX)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfie_speculative::build_tree_mask;

    /// Reference verify config: paper KV shape + the 6-node width-2
    /// tree 0 -> (1, 2); 1 -> (3, 4); 2 -> (5).
    fn tree6_mask_bits() -> Vec<u64> {
        vec![
            0b000001, // root: self
            0b000011, // 1: self + 0
            0b000101, // 2: self + 0
            0b001011, // 3: self + 1 + 0
            0b010011, // 4: self + 1 + 0
            0b100101, // 5: self + 2 + 0
        ]
    }

    fn paper_verify_cfg() -> VerifyAttentionConfig {
        VerifyAttentionConfig {
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 128,
            per_slot_max_tokens: 2048,
            max_slots: 64,
            num_nodes: 6,
            mask_bits: tree6_mask_bits(),
            sm_version: 80,
        }
    }

    fn reject_cfg(k: u32, vocab: u32) -> RejectionConfig {
        RejectionConfig {
            k_tokens: k,
            vocab_size: vocab,
        }
    }

    // ── verify: structural ─────────────────────────────────────────

    #[test]
    fn verify_param_list_is_exactly_the_six_direct_params() {
        let ptx = emit_verify_attention_ptx(&paper_verify_cfg());
        let start = ptx.find(".visible .entry nsl_cfie_spec_verify_attn(").unwrap();
        let end = start + ptx[start..].find(')').unwrap();
        let params: Vec<&str> = ptx[start..end]
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                l.starts_with(".param").then(|| l.trim_end_matches(','))
            })
            .collect();
        assert_eq!(
            params,
            vec![
                ".param .u64 q_ptr",
                ".param .u64 kv_base",
                ".param .u64 out_ptr",
                ".param .u32 layer_idx",
                ".param .u32 slot_idx",
                ".param .u32 seq_len",
            ]
        );
        assert_eq!(ptx.matches("ld.param").count(), 6);
    }

    #[test]
    fn verify_mask_rows_are_baked_immediates_not_a_parameter() {
        let cfg = paper_verify_cfg();
        let ptx = emit_verify_attention_ptx(&cfg);
        // The paper's claim vs flash_attention.rs runtime tree-parent
        // params: no mask reaches the kernel at runtime.
        assert!(!ptx.contains("mask_ptr"));
        assert!(!ptx.contains("tree_parent"));
        // One baked u64 immediate per node row, exact values.
        for &row in &cfg.mask_bits {
            assert!(
                ptx.contains(&format!("mov.u64 %rd_mask, 0x{:016X};", row)),
                "mask row 0x{row:016X} must be a baked immediate"
            );
        }
        assert_eq!(
            ptx.matches("mov.u64 %rd_mask, 0x").count(),
            cfg.num_nodes as usize
        );
    }

    #[test]
    fn verify_no_mad_lo_and_ascii_only() {
        let ptx = emit_verify_attention_ptx(&paper_verify_cfg());
        assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
        assert!(
            ptx.bytes().all(|b| b < 128),
            "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
        );
    }

    #[test]
    fn verify_kv_stride_header_matches_decode_attention_emitter() {
        // The verify kernel reads the SAME pool the decode kernels
        // maintain: baked stride header lines must be byte-identical.
        let cfg = paper_verify_cfg();
        let attn = crate::cfie_decode_attention::DecodeAttentionConfig {
            n_layers: 8,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            per_slot_max_tokens: cfg.per_slot_max_tokens,
            max_slots: cfg.max_slots,
            kv_dtype_bytes: 2,
        };
        let verify_ptx = emit_verify_attention_ptx(&cfg);
        let attn_ptx = crate::cfie_decode_attention::emit_decode_attention_ptx(&attn);
        let strides = |ptx: &str| -> Vec<String> {
            ptx.lines()
                .filter(|l| {
                    l.starts_with("//   ")
                        && (l.contains("token_stride")
                            || l.contains("kv_half_stride")
                            || l.contains("layer_stride")
                            || l.contains("max_tokens "))
                })
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        let v = strides(&verify_ptx);
        // token_stride, kv_half_stride, layer_stride,
        // per_slot_max_tokens, max_tokens.
        assert_eq!(v.len(), 5, "verify header must bake all five constants");
        assert_eq!(v, strides(&attn_ptx));
    }

    #[test]
    fn verify_meta_reports_launch_shape_and_smem() {
        let (ptx, meta) = emit_verify_attention(&paper_verify_cfg());
        assert_eq!(meta.kernel_name, VERIFY_KERNEL_NAME);
        assert_eq!(meta.block_dim, 128);
        assert!(meta.grid_dim_is_n_heads);
        // q(128 f32) + scores(128 f32) + rescale + l = 512 + 512 + 8.
        assert_eq!(meta.smem_bytes, 1032);
        assert!(ptx.contains(&format!(".shared .align 4 .b8 cfie_spec_smem[{}];", meta.smem_bytes)));
    }

    #[test]
    fn mask_bits_from_tree_matches_treemask_semantics() {
        let m = build_tree_mask(3, 2); // 7 nodes
        let bits = mask_bits_from_tree(&m);
        assert_eq!(bits.len(), 7);
        for r in 0..7u32 {
            for c in 0..7u32 {
                assert_eq!(
                    bits[r as usize] & (1u64 << c) != 0,
                    m.get(r, c),
                    "bit ({r},{c}) must mirror TreeMask::get"
                );
            }
        }
    }

    // ── verify: refusals ───────────────────────────────────────────

    #[test]
    #[should_panic(expected = "num_nodes")]
    fn verify_num_nodes_over_33_panics() {
        let mut cfg = paper_verify_cfg();
        cfg.num_nodes = 34;
        cfg.mask_bits = (0..34u64).map(|i| 1u64 << i).collect();
        let _ = emit_verify_attention(&cfg);
    }

    #[test]
    #[should_panic(expected = "one row per node")]
    fn verify_mask_row_count_mismatch_panics() {
        let mut cfg = paper_verify_cfg();
        cfg.mask_bits.pop();
        let _ = emit_verify_attention(&cfg);
    }

    #[test]
    #[should_panic(expected = "self bit")]
    fn verify_mask_row_without_self_bit_panics() {
        let mut cfg = paper_verify_cfg();
        cfg.mask_bits[3] = 0b000011; // row 3 lost its self bit
        let _ = emit_verify_attention(&cfg);
    }

    #[test]
    #[should_panic(expected = "beyond num_nodes")]
    fn verify_mask_bits_beyond_num_nodes_panic() {
        let mut cfg = paper_verify_cfg();
        cfg.mask_bits[0] |= 1u64 << 40;
        let _ = emit_verify_attention(&cfg);
    }

    #[test]
    #[should_panic(expected = "divisible")]
    fn verify_gqa_indivisible_panics() {
        let mut cfg = paper_verify_cfg();
        cfg.n_kv_heads = 3;
        let _ = emit_verify_attention(&cfg);
    }

    #[test]
    #[should_panic(expected = "appended tree rows")]
    fn verify_slot_too_small_for_tree_panics() {
        let mut cfg = paper_verify_cfg();
        cfg.per_slot_max_tokens = 4; // < num_nodes = 6
        let _ = emit_verify_attention(&cfg);
    }

    // ── verify: cpu reference ──────────────────────────────────────

    fn tiny_verify_cfg(num_nodes: u32, mask_bits: Vec<u64>) -> VerifyAttentionConfig {
        VerifyAttentionConfig {
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 2,
            per_slot_max_tokens: 64,
            max_slots: 1,
            num_nodes,
            mask_bits,
            sm_version: 80,
        }
    }

    fn fill(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
        (0..n).map(f).collect()
    }

    #[test]
    fn cpu_verify_root_with_empty_prefix_returns_its_own_v_row() {
        // seq_len = 0: node 0 attends only itself -> softmax weight 1.
        let m = build_tree_mask(2, 2); // 3 nodes
        let cfg = tiny_verify_cfg(m.num_nodes, mask_bits_from_tree(&m));
        let q = fill(3 * 2, |i| (i as f32 * 0.31).sin());
        let k = fill(3 * 2, |i| (i as f32 * 0.17).cos());
        let v = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let out = cpu_reference_verify(&cfg, &q, &k, &v, 0);
        assert_eq!(&out[0..2], &[10.0, 20.0], "root row must be v[node 0]");
    }

    #[test]
    fn cpu_verify_nodes_attend_prefix_plus_ancestors_only() {
        // build_tree_mask(3, 2): node 3's ancestors are {1, 0}; node 2
        // is a sibling branch.  Perturbing node 2's K/V must leave node
        // 3's output bit-identical; perturbing node 1's must change it.
        let m = build_tree_mask(3, 2); // 7 nodes
        let cfg = tiny_verify_cfg(m.num_nodes, mask_bits_from_tree(&m));
        let nn = 7usize;
        let sl = 3usize;
        let q = fill(nn * 2, |i| (i as f32 * 0.23).sin());
        let k = fill((sl + nn) * 2, |i| (i as f32 * 0.37).cos());
        let v = fill((sl + nn) * 2, |i| (i as f32 * 0.53).sin());
        let base = cpu_reference_verify(&cfg, &q, &k, &v, sl as u32);

        // Perturb sibling node 2 (pool position sl + 2).
        let mut k2 = k.clone();
        let mut v2 = v.clone();
        k2[(sl + 2) * 2] += 5.0;
        v2[(sl + 2) * 2 + 1] -= 7.0;
        let out2 = cpu_reference_verify(&cfg, &q, &k2, &v2, sl as u32);
        assert_eq!(
            &out2[3 * 2..4 * 2],
            &base[3 * 2..4 * 2],
            "node 3 must not see sibling node 2"
        );
        // But node 2's own row must have changed (sanity).
        assert_ne!(&out2[2 * 2..3 * 2], &base[2 * 2..3 * 2]);

        // Perturb ancestor node 1 (pool position sl + 1).
        let mut k1 = k.clone();
        k1[(sl + 1) * 2] += 5.0;
        let out1 = cpu_reference_verify(&cfg, &q, &k1, &v, sl as u32);
        assert_ne!(
            &out1[3 * 2..4 * 2],
            &base[3 * 2..4 * 2],
            "node 3 must see ancestor node 1"
        );

        // Every node sees the committed prefix.  (Dim 1: q rows start
        // at sin(0) = 0, so a dim-0 bump would be invisible to node 0.)
        let mut kp = k.clone();
        kp[1] += 3.0;
        let outp = cpu_reference_verify(&cfg, &q, &kp, &v, sl as u32);
        for node in 0..nn {
            assert_ne!(
                &outp[node * 2..node * 2 + 2],
                &base[node * 2..node * 2 + 2],
                "node {node} must attend the prefix"
            );
        }
    }

    #[test]
    fn cpu_verify_linear_chain_equals_causal_decode_attention() {
        // width = 1 chain: node i attends prefix + nodes 0..=i, which
        // is exactly causal attention at position seq_len + i.
        let m = build_tree_mask(4, 1); // 4-node chain
        let cfg = tiny_verify_cfg(m.num_nodes, mask_bits_from_tree(&m));
        let nn = 4usize;
        let sl = 5usize;
        let hd = 2usize;
        let q = fill(nn * hd, |i| (i as f32 * 0.29).sin());
        let k = fill((sl + nn) * hd, |i| (i as f32 * 0.41).cos());
        let v = fill((sl + nn) * hd, |i| (i as f32 * 0.61).sin());
        let out = cpu_reference_verify(&cfg, &q, &k, &v, sl as u32);
        for i in 0..nn {
            let causal_len = sl + i + 1;
            let expect = crate::cfie_decode_attention::cpu_reference(
                &q[i * hd..(i + 1) * hd],
                &k[..causal_len * hd],
                &v[..causal_len * hd],
                1,
                1,
                hd as u32,
                causal_len as u32,
            );
            for d in 0..hd {
                assert!(
                    (out[i * hd + d] - expect[d]).abs() < 1e-5,
                    "chain node {i} dim {d}: {} vs causal {}",
                    out[i * hd + d],
                    expect[d]
                );
            }
        }
    }

    // ── reject: structural ─────────────────────────────────────────

    #[test]
    fn reject_param_list_is_exactly_the_six_params() {
        let ir = build_rejection(&reject_cfg(5, 49_152));
        let names: Vec<&str> = ir.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "target_probs_ptr",
                "draft_probs_ptr",
                "draft_tokens_ptr",
                "rng_seed",
                "out_accepted_ptr",
                "out_correction_token_ptr",
            ]
        );
        // Every param is 8 bytes: five device pointers and the u64 seed.
        let ptx = emit_rejection_ptx(&reject_cfg(5, 49_152));
        for name in names {
            assert!(ptx.contains(&format!(".param .u64 param_{name}")), "{name}");
        }
    }

    #[test]
    fn reject_prng_idiom_and_sentinel_present() {
        use crate::kernel_ir::KirConst;
        let ir = build_rejection(&reject_cfg(5, 32_000));
        let mut u64s = Vec::new();
        let mut u32s = Vec::new();
        for op in ir.blocks.iter().flat_map(|b| b.ops.iter()) {
            match op {
                KirOp::Const(_, KirConst { value: ConstValue::U64(v), .. }) => u64s.push(*v),
                KirOp::Const(_, KirConst { value: ConstValue::U32(v), .. }) => u32s.push(*v),
                _ => {}
            }
        }
        // Same xorshift64* constants + golden-gamma guard as the fused
        // sampler, and the all-accept correction sentinel.
        assert!(u64s.contains(&0x2545_F491_4F6C_DD1D));
        assert!(u64s.contains(&0x9E37_79B9_7F4A_7C15));
        assert!(u32s.contains(&u32::MAX));
        // One ratio division, reached only past the division guard.
        let divs = ir
            .blocks
            .iter()
            .flat_map(|b| b.ops.iter())
            .filter(|op| matches!(op, KirOp::Div(..)))
            .count();
        assert_eq!(divs, 1);
        assert!(emit_rejection_ptx(&reject_cfg(5, 32_000)).contains("div.rn.f32"));
    }

    #[test]
    fn reject_no_mad_lo_and_ascii_only() {
        let ptx = emit_rejection_ptx(&reject_cfg(5, 49_152));
        assert!(!ptx.contains("mad."), "mad.lo.u32 is invalid at PTX ISA 7.0");
        assert!(
            ptx.bytes().all(|b| b < 128),
            "PTX must be ASCII-only (Unicode -> CUDA_ERROR_INVALID_PTX)"
        );
    }

    #[test]
    fn reject_meta_reports_serial_launch_shape() {
        let (_, meta) = emit_rejection_kernel(&reject_cfg(5, 32_000));
        assert_eq!(meta.kernel_name, REJECT_KERNEL_NAME);
        assert_eq!(meta.block_dim, 32);
        assert_eq!(meta.smem_bytes, 0);
        assert!(!meta.grid_dim_is_n_heads);
    }

    #[test]
    #[should_panic(expected = "k_tokens")]
    fn reject_k_over_32_panics() {
        let _ = emit_rejection_kernel(&reject_cfg(33, 100));
    }

    // ── reject: cpu reference ──────────────────────────────────────

    /// Row-major [k][vocab] target probs with prob `p` on the drafted
    /// token and the remainder spread over the rest.
    fn target_rows(k: usize, vocab: usize, tokens: &[u32], p_on_draft: f32) -> Vec<f32> {
        let mut rows = vec![0.0f32; k * vocab];
        for j in 0..k {
            let rest = (1.0 - p_on_draft) / (vocab as f32 - 1.0);
            for v in 0..vocab {
                rows[j * vocab + v] = if v as u32 == tokens[j] { p_on_draft } else { rest };
            }
        }
        rows
    }

    #[test]
    fn cpu_reject_all_accept_returns_k_and_sentinel() {
        let cfg = reject_cfg(3, 8);
        let tokens = [1u32, 4, 6];
        // ratio = 0.9 / 0.5 = 1.8 > any r in [0,1) -> every draw accepts.
        let target = target_rows(3, 8, &tokens, 0.9);
        let draft = [0.5f32; 3];
        for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
            let (acc, corr) = cpu_reference_reject(&cfg, &target, &draft, &tokens, seed);
            assert_eq!(acc, 3, "seed {seed}");
            assert_eq!(corr, u32::MAX, "seed {seed}");
        }
    }

    #[test]
    fn cpu_reject_reports_first_rejection_index() {
        let cfg = reject_cfg(3, 8);
        let tokens = [1u32, 4, 6];
        let mut target = target_rows(3, 8, &tokens, 0.9);
        // Position 1's drafted token gets zero target mass -> ratio 0,
        // r >= 0 always rejects there; position 0 still accepts.
        for v in 0..8 {
            target[8 + v] = if v == 4 { 0.0 } else { 1.0 / 7.0 };
        }
        let draft = [0.5f32; 3];
        for seed in [1u64, 7, 99, 12345] {
            let (acc, corr) = cpu_reference_reject(&cfg, &target, &draft, &tokens, seed);
            assert_eq!(acc, 1, "seed {seed}: must reject exactly at index 1");
            assert_ne!(corr, 4, "seed {seed}: zero-mass draft token cannot be resampled");
            assert!(corr < 8);
        }
    }

    #[test]
    fn cpu_reject_zero_draft_prob_rejects_despite_high_target_prob() {
        let cfg = reject_cfg(2, 8);
        let tokens = [3u32, 5];
        let target = target_rows(2, 8, &tokens, 0.9);
        // Division guard: p_draft = 0 must reject at index 0 without
        // evaluating the ratio.
        let draft = [0.0f32, 0.5];
        let (acc, corr) = cpu_reference_reject(&cfg, &target, &draft, &tokens, 7);
        assert_eq!(acc, 0);
        assert!(corr < 8);
    }

    #[test]
    fn cpu_reject_residual_never_returns_dominated_draft_token() {
        // p_target(draft) = 0.2 <= p_draft = 0.9 -> the residual zeroes
        // the drafted token; no seed may resample it.
        let cfg = reject_cfg(1, 16);
        let tokens = [7u32];
        let target = target_rows(1, 16, &tokens, 0.2);
        let draft = [0.9f32];
        let mut saw_rejection = false;
        for seed in 0..500u64 {
            let (acc, corr) = cpu_reference_reject(&cfg, &target, &draft, &tokens, seed);
            if acc == 0 {
                saw_rejection = true;
                assert_ne!(corr, 7, "seed {seed}: residual q(draft) == 0");
                assert!(corr < 16);
            }
        }
        assert!(saw_rejection, "ratio 0.2/0.9 must reject for some seed");
    }

    #[test]
    fn cpu_reject_empty_residual_falls_back_to_draft_token() {
        // All target mass on the drafted token and p_draft >= p_target:
        // residual is empty -> documented fallback returns the token.
        let cfg = reject_cfg(1, 4);
        let tokens = [2u32];
        let mut target = vec![0.0f32; 4];
        target[2] = 1.0;
        let draft = [2.0f32]; // ratio 0.5: some seeds reject
        let mut saw_rejection = false;
        for seed in 0..64u64 {
            let (acc, corr) = cpu_reference_reject(&cfg, &target, &draft, &tokens, seed);
            if acc == 0 {
                saw_rejection = true;
                assert_eq!(corr, 2, "empty residual must fall back to the draft token");
            }
        }
        assert!(saw_rejection);
    }

    #[test]
    fn cpu_reject_deterministic_given_seed() {
        let cfg = reject_cfg(4, 32);
        let tokens = [3u32, 9, 20, 31];
        let target = target_rows(4, 32, &tokens, 0.4);
        let draft = [0.8f32, 0.7, 0.9, 0.6];
        for seed in [0u64, 5, 0xABCD] {
            let a = cpu_reference_reject(&cfg, &target, &draft, &tokens, seed);
            let b = cpu_reference_reject(&cfg, &target, &draft, &tokens, seed);
            assert_eq!(a, b, "seed {seed}: sampled pair must be a pure function of inputs");
        }
    }

    // ── ptxas validation (skips silently when no validator present) ──

    #[test]
    fn ptxas_validates_verify_tree6() {
        let ptx = emit_verify_attention_ptx(&paper_verify_cfg());
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                nsl_log::nsl_log!(INFO, "skip", "[skip] cfie spec-verify ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie spec-verify PTX rejected for the 6-node tree config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }

    #[test]
    fn ptxas_validates_reject_paper_config() {
        let ptx = emit_rejection_ptx(&reject_cfg(5, 49_152));
        match crate::ptxas_validation::validate_ptx(&ptx) {
            Ok(()) => {}
            Err(msg) if msg.contains("nvcc not available") => {
                nsl_log::nsl_log!(INFO, "skip", "[skip] cfie spec-reject ptxas validation - no validator: {msg}");
            }
            Err(msg) => panic!(
                "cfie spec-reject PTX rejected for the paper config:\n{msg}\n\nEmitted PTX:\n{ptx}"
            ),
        }
    }
}
