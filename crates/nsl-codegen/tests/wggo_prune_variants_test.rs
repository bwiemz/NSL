// Verifies the prune OverrideRejectReason variants exist per spec §6.3,
// using the OverrideRejectReason enum in place of the spec's DiagnosticCode
// (the codebase has no parallel DiagnosticCode enum).
//
// `PruneWholeBlockUnsupported` (spec §3.6) is gone: whole-block prune is
// implemented by the v2 chain-collapse, so no plan can reach a "Block is
// unsupported" refusal any more. Its v2 replacement is
// `PruneBrokenResidualChain` — the refusal for a block whose residual Adds
// cannot be collapsed.

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
    let _ = OverrideRejectReason::PruneBrokenResidualChain;
    let _ = OverrideRejectReason::PruneConflictingDecisions;
}
