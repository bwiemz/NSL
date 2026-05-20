use nsl_codegen::wrga_fusion::{build_fusion_plan, FusionPlan};
use std::collections::HashMap;

fn nid(n: u32) -> nsl_ast::NodeId {
    nsl_ast::NodeId(n)
}

#[test]
fn fusion_plan_exposes_fused_node_groups_field() {
    let plan = build_fusion_plan(&[], None);
    let _: &HashMap<nsl_ast::NodeId, Vec<nsl_ast::NodeId>> = &plan.fused_node_groups;
    assert!(plan.fused_node_groups.is_empty());
}

#[test]
fn non_fused_op_returns_singleton_via_lookup() {
    let plan = FusionPlan {
        decisions: vec![],
        fused_node_groups: HashMap::new(),
    };
    let n = nid(42);
    assert_eq!(plan.constituents_of(n), vec![n]);
}

#[test]
fn fused_root_returns_all_constituents() {
    let mut groups = HashMap::new();
    let root = nid(10);
    let c1 = nid(11);
    let c2 = nid(12);
    let c3 = nid(13);
    groups.insert(root, vec![c1, c2, c3]);
    let plan = FusionPlan {
        decisions: vec![],
        fused_node_groups: groups,
    };
    assert_eq!(plan.constituents_of(root), vec![c1, c2, c3]);
}
