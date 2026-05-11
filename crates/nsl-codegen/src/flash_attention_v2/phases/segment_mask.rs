//! PCA Tier A — shared segment-mask predicate emitter (spec §5.1).
//!
//! The existing forward `s_compute.rs` causal block sets a mask
//! predicate (`%p0` in the kernel's register scheme) to `TRUE` iff
//! `k_global > q_row_global`. This helper extends that predicate
//! in place with a cross-segment disjunction:
//!
//! ```text
//!   new %p0 = old %p0  OR  (segment_ids[q_row_global] != segment_ids[k_global])
//! ```
//!
//! Both the forward `s_compute.rs` and the backward `ds_compute.rs`
//! call this helper with their respective mask-predicate registers
//! and their respective position registers. The emitted PTX substring
//! MUST be byte-identical across callers (invariant §4 #1/#2).
//!
//! # Convention — mask-predicate (§5.1 amendment)
//!
//! Spec §5 originally wrote this helper in allow-predicate convention
//! (`pred TRUE` = keep cell). The real FA2 emitter uses mask-predicate
//! convention (`pred TRUE` = force to `-inf`). This helper matches
//! the real emitter. The two conventions are semantically equivalent
//! by De Morgan.
//!
//! # Residency
//!
//! Tier A exercises only `SegmentResidency::Shared` up to the ceiling
//! defined by `pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET` (the FA
//! prelude's `seg_smem` allocation is sized to the same budget). The
//! `Streamed` branch emits a `trap` marker so mis-planned residency
//! surfaces at runtime rather than silently producing wrong gradients.

use crate::pca_segment::SegmentResidency;

/// Emit PTX that extends `mask_pred_inout` with a cross-segment
/// disjunction.  Does not return a new register — the caller's
/// existing mask predicate is updated in place.
///
/// # Arguments
///
/// * `ptx` — the kernel's PTX string buffer to append to.
/// * `q_pos_reg` — name of the u64 register holding
///   `q_row_global` (e.g. `"%rd35"`). Flows at u64 width directly
///   from tile-offset arithmetic.
/// * `k_pos_reg` — name of the u64 register holding `k_global`
///   (e.g. `"%rd34"`).
/// * `segment_ids_reg` — name of the u64 register holding the
///   SMEM generic-space base address of `segment_ids` (e.g. `"%seg_base"`).
///   Must be the full 64-bit generic address from `cvta.shared.u64` — do NOT
///   truncate to u32 (Blackwell sm_120 produces wrong values when the label is
///   at shared offset 0).
/// * `residency` — `Shared` uses `ld.shared.u16`; `Streamed` emits
///   a `trap` marker (out of scope for Tier A).
/// * `mask_pred_inout` — name of the u1 predicate register that
///   already holds the causal-mask predicate (e.g. `"%p0"`);
///   this helper OR-extends it in place.
///
/// The emitted PTX allocates three internal scratch registers
/// (`%rd_q_SEGMASK`, `%rd_k_SEGMASK`, `%p_seg_SEGMASK`) whose names
/// are fixed rather than monotonically seeded — multiple calls in
/// the same kernel would collide.  Tier A never calls this helper
/// more than once per kernel, so fixed names are fine; if a future
/// tier needs multiple calls, introduce a seed parameter then.
pub fn emit_segment_mask_predicate(
    ptx:             &mut String,
    q_pos_reg:       &str,
    k_pos_reg:       &str,
    segment_ids_reg: &str,
    residency:       SegmentResidency,
    mask_pred_inout: &str,
) {
    ptx.push_str("    // PCA Tier A: extend %p0 with cross-segment disjunction (spec 5.1)\n");
    emit_segment_id_load(ptx, q_pos_reg, segment_ids_reg, residency, 'q');
    emit_segment_id_load(ptx, k_pos_reg, segment_ids_reg, residency, 'k');
    ptx.push_str("    setp.ne.u16    %p_seg_SEGMASK, %rs_q_SEGMASK, %rs_k_SEGMASK;\n");
    ptx.push_str(&format!(
        "    or.pred        {pred}, {pred}, %p_seg_SEGMASK;\n",
        pred = mask_pred_inout
    ));
}

/// Private sub-helper: compute a u64 SMEM address and `ld.shared.u16`.
/// Uses full u64 arithmetic throughout to avoid the `cvt.u32.u64` truncation
/// bug on Blackwell (sm_120) where `seg_smem` at shared offset 0 produces a
/// generic address whose low 32 bits are non-zero (0x13E00 observed on
/// RTX 5070 Ti), causing CUDA_ERROR_ILLEGAL_ADDRESS at runtime.
///
/// Residency-aware: `Shared` = `ld.shared.u16`, `Streamed` = `trap;` (Tier A out of scope).
fn emit_segment_id_load(
    ptx:             &mut String,
    pos_reg:         &str,
    segment_ids_reg: &str,
    residency:       SegmentResidency,
    tag:             char,
) {
    let rd_off = format!("%rd_{tag}_SEGMASK");
    let rs = format!("%rs_{tag}_SEGMASK");
    // Use u64 width throughout: avoid cvt.u32.u64 which truncates the Blackwell
    // generic address incorrectly when the label is at shared offset 0.
    ptx.push_str(&format!(
        "    and.b64        {rd_off}, {pos_reg}, 0xFFFFFFFF; // {tag}_global low-32 as u64\n"
    ));
    ptx.push_str(&format!(
        "    shl.b64        {rd_off}, {rd_off}, 1;                // {tag}*sizeof(u16)\n"
    ));
    match residency {
        SegmentResidency::Shared => {
            ptx.push_str(&format!(
                "    add.u64        {rd_off}, {segment_ids_reg}, {rd_off};     // &seg[{tag}]\n"
            ));
            ptx.push_str(&format!(
                "    ld.shared.u16  {rs}, [{rd_off}];\n"
            ));
        }
        SegmentResidency::Streamed => {
            ptx.push_str("    // SegmentResidency::Streamed is out of scope for PCA Tier A\n");
            ptx.push_str(&format!(
                "    trap;                                        // planner mis-routed {tag}\n"
            ));
            // Still emit a dummy ld so the reg name is defined
            // even on the trap path (keeps downstream PTX parsable).
            ptx.push_str(&format!(
                "    ld.shared.u16  {rs}, [{segment_ids_reg}];    // unreachable\n"
            ));
        }
    }
}
