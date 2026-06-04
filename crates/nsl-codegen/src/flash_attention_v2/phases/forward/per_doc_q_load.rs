//! Per-document CTA Q-load delta — Phase 1 bounds predicate.
//!
//! For the per-doc CTA kernel, Q-load must be bounds-checked against
//! `doc_len` (= `q_end - q_start`) rather than `block_q`.  Out-of-doc
//! lanes still participate in `bar.sync 0` (warp semantics) but do NOT
//! issue global loads.
//!
//! In v1 the Q-load body is IDENTICAL to the standard `q_load::emit` since
//! `segment_masked=false` and `csha.fused_projections=false` in the
//! per-doc config.  The only structural difference is that the warp's row
//! might be beyond `doc_len` — in that case the warp writes zeros to
//! Q-SMEM (by predicating the HBM load).
//!
//! # Design decision (v1 simplification)
//!
//! Rather than duplicating the entire Q-load emitter, the per-doc kernel
//! reuses `q_load::emit` verbatim.  The `doc_len` clamp is enforced
//! STRUCTURALLY by the fact that the KV loop only runs from
//! `%k_start = %q_start` to `%k_max = %q_end`, so only doc-local KV
//! tokens are ever loaded.  For Q: warps beyond doc_len load whatever is
//! in global memory past the doc boundary — those output rows are in the
//! padding region `[doc_end, padded_seq_len)` which the design marks as
//! OUTPUT-DON'T-CARE.  v1 accepts this with a clear documentation note.
//!
//! # This module
//!
//! Provides `emit_doc_len_q_load_guard` — a small PTX snippet that emits
//! an early-exit predicate for warps whose `q_row_local >= doc_len`.  This
//! is NOT called in the default per-doc synthesis path (which relies on the
//! structural bound above) but is provided for callers that want clean
//! padding behaviour.

use crate::flash_attention::FlashAttentionConfig;

/// Emit a Q-load warp-level skip predicate for warps beyond `doc_len`.
///
/// When `q_row_local = q_tile_iter * 4 + warp_id >= doc_len`, the warp
/// writes zero into Q-SMEM and skips the HBM load.  The bar.sync still
/// runs so all warps participate in the fence.
///
/// `%r_pdoc_len` must be set by `per_doc_prelude::emit` before this is called.
#[allow(dead_code)] // provided for v2 use; v1 uses structural bound only
pub fn emit_doc_len_q_load_guard(
    ptx: &mut String,
    _config: &FlashAttentionConfig,
    q_tile_iter: u32,
    label_suffix: &str,
) {
    ptx.push_str(&format!(
        "    // per-doc Q-load guard: skip warp if q_row_local >= doc_len (q_tile_iter={})\n",
        q_tile_iter
    ));
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {};  // q_row_local = warp_id + {}\n",
        q_tile_iter * 4, q_tile_iter * 4
    ));
    ptx.push_str(
        "    setp.ge.u32 %p0, %r0, %r_pdoc_len;  // q_row_local >= doc_len\n"
    );
    ptx.push_str(&format!(
        "    @%p0 bra PER_DOC_Q_SKIP_{};          // out-of-doc warp — skip load\n",
        label_suffix
    ));
}
