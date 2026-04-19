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
//! Tier A exercises only `SegmentResidency::Shared` at seq ≤ 2048
//! on the 4096 B SMEM budget. The `Streamed` branch emits a `trap`
//! marker so mis-planned residency surfaces at runtime rather than
//! silently producing wrong gradients.

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
/// * `segment_ids_reg` — name of the u32 register holding the
///   SMEM base address of `segment_ids` (e.g. `"%seg_base"`).
/// * `residency` — `Shared` uses `ld.shared.u16`; `Streamed` emits
///   a `trap` marker (out of scope for Tier A).
/// * `mask_pred_inout` — name of the u1 predicate register that
///   already holds the causal-mask predicate (e.g. `"%p0"`);
///   this helper OR-extends it in place.
///
/// The emitted PTX allocates three internal scratch registers
/// (`%r_sq_SEGMASK`, `%r_sk_SEGMASK`, `%p_seg_SEGMASK`) whose names
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
    ptx.push_str("    // PCA Tier A: extend %p0 with cross-segment disjunction (spec §5.1)\n");
    emit_segment_id_load(ptx, q_pos_reg, segment_ids_reg, residency, 'q');
    emit_segment_id_load(ptx, k_pos_reg, segment_ids_reg, residency, 'k');
    ptx.push_str("    setp.ne.u16    %p_seg_SEGMASK, %rs_q_SEGMASK, %rs_k_SEGMASK;\n");
    ptx.push_str(&format!(
        "    or.pred        {pred}, {pred}, %p_seg_SEGMASK;\n",
        pred = mask_pred_inout
    ));
}

/// Private sub-helper: emit the u64 → u32 truncation, offset-by-2,
/// add to SMEM base, and `ld.shared.u16`.  Produces the 4-instruction
/// prelude per position.  Residency-aware: `Shared` = `ld.shared.u16`,
/// `Streamed` = `trap;` (Tier A out of scope).
fn emit_segment_id_load(
    ptx:             &mut String,
    pos_reg:         &str,
    segment_ids_reg: &str,
    residency:       SegmentResidency,
    tag:             char,
) {
    let r_off = format!("%r_{tag}_SEGMASK");
    let rs = format!("%rs_{tag}_SEGMASK");
    ptx.push_str(&format!(
        "    cvt.u32.u64    {r_off}, {pos_reg};            // {tag}_global low-32\n"
    ));
    ptx.push_str(&format!(
        "    shl.b32        {r_off}, {r_off}, 1;                 // {tag}*sizeof(u16)\n"
    ));
    match residency {
        SegmentResidency::Shared => {
            ptx.push_str(&format!(
                "    add.u32        {r_off}, {segment_ids_reg}, {r_off};      // &seg[{tag}]\n"
            ));
            ptx.push_str(&format!(
                "    ld.shared.u16  {rs}, [{r_off}];\n"
            ));
        }
        SegmentResidency::Streamed => {
            ptx.push_str(&format!(
                "    // SegmentResidency::Streamed is out of scope for PCA Tier A\n"
            ));
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
