// Verifies the PR #102 stderr-contract string survives the
// PruneNotImplemented → WholeBlockPruneNotImplemented rename.
// Spec §5.5: the string "ir_rewrite_not_implemented" is emitted via
// an explicit function return, NOT via a debug-formatted variant name.

use nsl_codegen::wggo_overrides::whole_block_prune_not_implemented_reason;

#[test]
fn whole_block_prune_reason_string_preserves_pr102_contract() {
    assert_eq!(
        whole_block_prune_not_implemented_reason(),
        "ir_rewrite_not_implemented",
        "PR #102's stderr-contract string must not change on rename"
    );
}
