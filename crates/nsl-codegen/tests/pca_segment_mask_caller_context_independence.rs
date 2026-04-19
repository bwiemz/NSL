//! Spec §7.2: the helper's PTX emission must depend ONLY on its
//! arguments, not on caller-context state.  Catches the "helper
//! secretly consults caller state" bug that snapshot tests cannot
//! catch by themselves (snapshots of both forward and backward
//! kernels would update together, hiding the drift).

use nsl_codegen::flash_attention_v2::phases::segment_mask::emit_segment_mask_predicate;
use nsl_codegen::pca_segment::SegmentResidency;

/// Simulated caller contexts — deliberately vacuous; the test
/// asserts the helper ignores them.
enum CallerContext {
    ForwardScompute,
    BackwardDsCompute,
}

fn capture_helper_emission_in(ctx: CallerContext) -> String {
    // Caller-context-dependent prelude.  The helper's emission MUST
    // NOT change based on what came before it in the kernel.
    let mut ptx = match ctx {
        CallerContext::ForwardScompute => String::from(
            "// forward s_compute caller prelude — irrelevant to helper\n",
        ),
        CallerContext::BackwardDsCompute => String::from(
            "// backward ds_compute caller prelude — irrelevant to helper\n",
        ),
    };
    let prelude_len = ptx.len();

    emit_segment_mask_predicate(
        &mut ptx,
        "%rd35",
        "%rd34",
        "%seg_base",
        SegmentResidency::Shared,
        "%p0",
    );

    // Return only the helper's contribution, stripping the prelude.
    ptx[prelude_len..].to_string()
}

#[test]
fn segment_mask_predicate_byte_identical_across_callers() {
    let fwd = capture_helper_emission_in(CallerContext::ForwardScompute);
    let bwd = capture_helper_emission_in(CallerContext::BackwardDsCompute);
    assert_eq!(
        fwd, bwd,
        "segment mask predicate must emit byte-identically in forward and \
         backward contexts (spec §4 invariants #1 and #2)"
    );
}
