// Task 15 Layer 4: verify format_refusal produces three-part spec §3 output
// AND diagnostic_code maps each refusal variant to the matching
// OverrideRejectReason::Prune* discriminant.

use nsl_codegen::wggo_graph::LayerRole;
use nsl_codegen::wggo_overrides::OverrideRejectReason;
use nsl_codegen::wggo_prune::{
    diagnostic_code, format_refusal, format_success_stderr, PruneRefusal, PruneRewrite,
};

#[test]
fn refusal_cross_layer_param_three_part_format() {
    let r = PruneRefusal::CrossLayerParam {
        layer_name: "blocks.7.attn".into(),
        layer_role: LayerRole::Attention,
        param_name: "blocks.7.attn.wq".into(),
        param_var: 200,
        external_consumer: 42,
        external_op_kind: "Mul".into(),
    };
    let text = format_refusal(&r);
    assert!(
        text.starts_with("prune: layer has cross-layer parameter sharing"),
        "expected three-part header; got: {text}"
    );
    assert!(text.contains("requested:"));
    assert!(text.contains("expected:"));
    assert!(text.contains("found:"));
    assert!(text.contains("blocks.7.attn"));
    assert!(text.contains("op_id=42"));
    assert_eq!(
        diagnostic_code(&r),
        OverrideRejectReason::PruneCrossLayerParam
    );
}

#[test]
fn refusal_no_residual_add_three_part_format() {
    let r = PruneRefusal::NoResidualAdd {
        layer_name: "blocks.3.attn".into(),
        layer_role: LayerRole::Attention,
        closure_size: 5,
    };
    let text = format_refusal(&r);
    assert!(text.starts_with("prune: layer is not residual-structured"));
    assert!(text.contains("closure has 5 ops"));
    assert_eq!(
        diagnostic_code(&r),
        OverrideRejectReason::PruneNoResidualAdd
    );
}

#[test]
fn refusal_parallel_branches_three_part_format() {
    let r = PruneRefusal::ParallelResidualBranches {
        layer_name: "blocks.2.attn".into(),
        layer_role: LayerRole::Attention,
        add_ops: vec![12, 18],
    };
    let text = format_refusal(&r);
    assert!(text.contains("parallel residual branches"));
    assert!(text.contains("2 residual Adds"));
    assert_eq!(
        diagnostic_code(&r),
        OverrideRejectReason::PruneParallelResidualBranches
    );
}

#[test]
fn refusal_ambiguous_pattern_three_part_format() {
    let r = PruneRefusal::AmbiguousPatternMatch {
        layer_name: "blocks.4.ffn".into(),
        layer_role: LayerRole::Ffn,
        h_before_var: 100,
        candidate_adds: vec![8, 15],
    };
    let text = format_refusal(&r);
    assert!(text.contains("multiple candidate residual boundaries"));
    assert!(text.contains("2 candidate Adds"));
    assert!(text.contains("VarId 100"));
    assert_eq!(
        diagnostic_code(&r),
        OverrideRejectReason::PruneAmbiguousPatternMatch
    );
}

#[test]
fn refusal_empty_closure_three_part_format() {
    let r = PruneRefusal::EmptyClosure {
        layer_name: "blocks.99.attn".into(),
        layer_role: LayerRole::Attention,
        prefix: "blocks.99.attn.".into(),
    };
    let text = format_refusal(&r);
    assert!(text.contains("no parameters match"));
    assert!(text.contains("blocks.99.attn."));
    assert_eq!(diagnostic_code(&r), OverrideRejectReason::PruneEmptyClosure);
}

#[test]
fn refusal_whole_block_unsupported_three_part_format() {
    let r = PruneRefusal::WholeBlockUnsupported {
        layer_name: "blocks.7".into(),
    };
    let text = format_refusal(&r);
    assert!(text.contains("whole-block pruning"));
    assert!(text.contains("not supported in v1"));
    assert!(text.contains("blocks.7.attn"));
    assert!(text.contains("blocks.7.ffn"));
    assert!(text.contains("planned:"));
    assert_eq!(
        diagnostic_code(&r),
        OverrideRejectReason::PruneWholeBlockUnsupported
    );
}

#[test]
fn refusal_conflicting_decisions_three_part_format() {
    let r = PruneRefusal::ConflictingPruneDecisions {
        decision_a: "blocks.1.attn".into(),
        decision_b: "blocks.2.attn".into(),
        reason: "closures overlap on ops: [3, 5]".into(),
    };
    let text = format_refusal(&r);
    assert!(text.contains("two prune decisions in the same plan conflict"));
    assert!(text.contains("blocks.1.attn AND prune blocks.2.attn"));
    assert_eq!(
        diagnostic_code(&r),
        OverrideRejectReason::PruneConflictingDecisions
    );
}

#[test]
fn success_stderr_format_matches_spec() {
    // Spec §6.1: [prune] layer=N name=<name> role=<role> applied=true
    // closure_size=K ops_deleted=K residual_add_op=ID
    let rewrite = PruneRewrite {
        layer_name: "blocks.7.attn".into(),
        layer_role: LayerRole::Attention,
        h_before_var: 100,
        h_after_var: 202,
        residual_add_op: 42,
        closure_ops: vec![10, 11, 12, 13],
        ops_deleted: 5, // 4 closure ops + 1 residual Add
    };
    let line = format_success_stderr(&rewrite, /*layer_index=*/ 7, /*ops_deleted=*/ 5);
    assert!(line.contains("layer=7"));
    assert!(line.contains("name=blocks.7.attn"));
    assert!(line.contains("role=Attention"));
    assert!(line.contains("applied=true"));
    assert!(line.contains("closure_size=4"));
    assert!(line.contains("ops_deleted=5"));
    assert!(line.contains("residual_add_op=42"));
    // Separator convention: key=value, no colons.
    assert!(
        !line.contains(":"),
        "format should use = not : ; got {line}"
    );
}

#[test]
fn multi_rewrite_stderr_reports_per_rewrite_ops_deleted_not_aggregate() {
    // Regression: stmt.rs previously passed wggo_prune_result.ops_deleted
    // (aggregate) to format_success_stderr for each rewrite, so multi-prune
    // plans reported identical (combined) op counts on every line.
    //
    // This test verifies per-rewrite ops_deleted is the source of truth.
    let rewrite_a = PruneRewrite {
        layer_name: "blocks.1.attn".into(),
        layer_role: LayerRole::Attention,
        h_before_var: 100,
        h_after_var: 110,
        residual_add_op: 9,
        closure_ops: vec![1, 2, 3],
        ops_deleted: 4, // 3 + 1
    };
    let rewrite_b = PruneRewrite {
        layer_name: "blocks.2.ffn".into(),
        layer_role: LayerRole::Ffn,
        h_before_var: 200,
        h_after_var: 220,
        residual_add_op: 19,
        closure_ops: vec![11, 12, 13, 14, 15],
        ops_deleted: 6, // 5 + 1
    };
    let line_a = format_success_stderr(&rewrite_a, 1, rewrite_a.ops_deleted);
    let line_b = format_success_stderr(&rewrite_b, 2, rewrite_b.ops_deleted);
    assert!(
        line_a.contains("ops_deleted=4"),
        "line_a expected ops_deleted=4; got: {line_a}"
    );
    assert!(
        line_b.contains("ops_deleted=6"),
        "line_b expected ops_deleted=6; got: {line_b}"
    );
    // If a caller accidentally passes the aggregate (10) to both, both lines
    // would contain ops_deleted=10 — catch that.
    assert!(!line_a.contains("ops_deleted=10"));
    assert!(!line_b.contains("ops_deleted=10"));
}
