//! insta snapshot of the PTX emitted by `emit_segment_mask_predicate`.
//! Spec §7.1 primary test.
//!
//! Two snapshot cases: one for `Shared` residency (the Tier A path),
//! one for `Streamed` residency (out-of-scope trap marker, verified
//! to ensure the planner can rely on a panic-on-mis-route contract).

use nsl_codegen::flash_attention_v2::phases::segment_mask::emit_segment_mask_predicate;
use nsl_codegen::pca_segment::SegmentResidency;

fn snapshot_with(residency: SegmentResidency) -> String {
    let mut ptx = String::new();
    emit_segment_mask_predicate(
        &mut ptx,
        "%rd35",     // q_row_global (u64)
        "%rd34",     // k_global     (u64)
        "%seg_base", // SMEM base    (u32)
        residency,
        "%p0", // caller's mask predicate — helper OR-extends
    );
    ptx
}

#[test]
fn snapshot_shared() {
    insta::assert_snapshot!(snapshot_with(SegmentResidency::Shared));
}

#[test]
fn snapshot_streamed_emits_trap() {
    insta::assert_snapshot!(snapshot_with(SegmentResidency::Streamed));
}
