// Verifies the 7 new OverrideRejectReason variants exist per spec §6.3,
// using the OverrideRejectReason enum in place of the spec's DiagnosticCode
// (the codebase has no parallel DiagnosticCode enum).

use nsl_codegen::wggo_overrides::OverrideRejectReason;

#[test]
fn prune_refusal_variants_exist() {
    // Each variant should construct (unit-style, no fields — we'll use
    // these as discriminants for the structural test assertions Task 15
    // wires up).
    let _ = OverrideRejectReason::PruneCrossLayerParam;
    let _ = OverrideRejectReason::PruneNoResidualAdd;
    let _ = OverrideRejectReason::PruneParallelResidualBranches;
    let _ = OverrideRejectReason::PruneAmbiguousPatternMatch;
    let _ = OverrideRejectReason::PruneEmptyClosure;
    let _ = OverrideRejectReason::PruneWholeBlockUnsupported;
    let _ = OverrideRejectReason::PruneConflictingDecisions;
}
