//! Per-document CTA prelude — Phase 0 delta for the G2 Strategy 3 kernel.
//!
//! Emits the block-index computation that replaces the standard
//! `%q_start = bid_x * block_q` with a per-doc load:
//!
//! ```ptx
//! // per-doc CTA: q_start = doc_starts[bid_x]
//! // q_end   = doc_starts[bid_x + 1]
//! ```
//!
//! This module is called by `per_doc_cta::synthesize_per_doc_cta_forward`
//! AFTER the standard `prelude::emit` runs (which declares all registers and
//! loads scalar params).  The emit here OVERWRITES `%q_start` and sets the
//! new `%k_max` to `doc_end` rather than `seq_len`.
//!
//! # Register contract after this runs
//!
//! `%q_start` (u64) — start token position for this CTA's document.
//! `%k_max`   (u64) — end token position (one past the last KV token).
//! `%r_pdoc_len` (u32) — document length = doc_end - doc_start (u32 scratch;
//! NOT a permanent register; reused by Q-load bounds predicate).
//!
//! # Param dependency
//!
//! The kernel param `doc_starts_ptr` must be present in the PTX param block.
//! For the per-doc kernel we add this param unconditionally (unlike the Tier A
//! `segment_masked && rope_q` gating).  This is added in
//! `per_doc_cta::build_per_doc_param_list`.

use crate::flash_attention::FlashAttentionConfig;

/// Emit the per-doc CTA prelude block (Phase 0 delta).
///
/// Preconditions (set by the standard `prelude::emit` before this runs):
///   `%bid_x` (u32) = blockIdx.x = document index for this CTA.
///   Standard scalar params already loaded into registers.
///   `doc_starts_ptr` param declared in the kernel signature.
///
/// Postconditions:
///   `%q_start` (u64) = doc_starts[bid_x]   (replaces `bid_x * block_q`).
///   `%k_max`   (u64) = doc_starts[bid_x+1] (replaces `seq_len`).
///   `%r_pdoc_len` (u32) declared and set = doc_end - doc_start.
pub fn emit(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("\n    // per-doc CTA: q_start = doc_starts[bid_x]\n");
    ptx.push_str("    // q_end = doc_starts[bid_x + 1]\n");
    ptx.push_str("    .reg .u64 %rd_pdoc_base, %rd_pdoc_addr;\n");
    ptx.push_str("    .reg .s32 %r_pdoc_start, %r_pdoc_end;\n");
    ptx.push_str("    .reg .u32 %r_pdoc_len;\n");

    // Load doc_starts device pointer from kernel params.
    ptx.push_str("    ld.param.u64 %rd_pdoc_base, [doc_starts_ptr];\n");

    // Address of doc_starts[bid_x] = doc_starts_ptr + bid_x * 4 (i32 = 4 bytes).
    ptx.push_str("    cvt.u64.u32 %rd_pdoc_addr, %bid_x;\n");
    ptx.push_str("    shl.b64 %rd_pdoc_addr, %rd_pdoc_addr, 2;  // bid_x * 4\n");
    ptx.push_str("    add.u64 %rd_pdoc_addr, %rd_pdoc_base, %rd_pdoc_addr;\n");
    ptx.push_str("    ld.global.s32 %r_pdoc_start, [%rd_pdoc_addr];\n");

    // Address of doc_starts[bid_x + 1].
    ptx.push_str("    add.u64 %rd_pdoc_addr, %rd_pdoc_addr, 4;  // next i32\n");
    ptx.push_str("    ld.global.s32 %r_pdoc_end, [%rd_pdoc_addr];\n");

    // doc_len = doc_end - doc_start (i32 arithmetic, then widen to u32).
    ptx.push_str("    sub.s32 %r_pdoc_len, %r_pdoc_end, %r_pdoc_start;\n");

    // Overwrite %q_start: doc_start (s32 sign-extend to u64 for addressing).
    ptx.push_str("    cvt.u64.s32 %q_start, %r_pdoc_start;\n");
    ptx.push_str("    // %q_start now = doc_start (per-doc CTA override)\n");

    // Overwrite %k_max: doc_end (the KV loop sweeps [doc_start, doc_end) only).
    ptx.push_str("    cvt.u64.s32 %k_max, %r_pdoc_end;\n");
    ptx.push_str("    // %k_max now = doc_end (per-doc CTA override)\n");
}
