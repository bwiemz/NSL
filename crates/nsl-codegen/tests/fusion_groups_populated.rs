//! Task 1: apply_epilogue_fusion populates FusionPlan.fused_node_groups.
//! Task 2: apply_reduction_fusion populates FusionPlan.fused_node_groups.

use nsl_ast::NodeId as AstNodeId;
use nsl_codegen::epilogue_fusion::{apply_epilogue_fusion, EpilogueChain, EpilogueOp, MatmulKind};
use nsl_codegen::fusion_graph::{FusionGraph, FusionOp};
use nsl_codegen::reduction_fusion::{apply_reduction_fusion, detect_reduction_patterns};
use nsl_codegen::wrga_fusion::FusionPlan;
use nsl_semantic::types::DType;

#[test]
fn epilogue_fusion_populates_fused_node_groups() {
    // Build a minimal graph: matmul + bias input + relu, so node indices exist.
    let mut graph = FusionGraph::new();
    let a = graph.add_node(FusionOp::Input, vec![]);
    let b = graph.add_node(FusionOp::Input, vec![]);
    let bias_in = graph.add_node(FusionOp::Input, vec![]);
    let matmul = graph.add_node(FusionOp::Matmul, vec![a, b]);
    let bias = graph.add_node(FusionOp::Elementwise("add".into()), vec![matmul, bias_in]);
    let relu = graph.add_node(FusionOp::Elementwise("relu".into()), vec![bias]);

    let chain = EpilogueChain {
        matmul_node: matmul,
        matmul_kind: MatmulKind::Standard,
        epilogue_ops: vec![
            EpilogueOp::BiasAdd {
                bias_node: bias,
                broadcast_dim: 0,
            },
            EpilogueOp::Activation("relu".into()),
        ],
        output_node: relu,
        eliminated_nodes: vec![bias, relu],
    };

    let mut plan = FusionPlan::default();
    apply_epilogue_fusion(&mut graph, &[chain], 0, &mut plan);

    let constituents = plan
        .fused_node_groups
        .get(&AstNodeId(matmul))
        .expect("matmul root should be keyed into fused_node_groups");
    assert_eq!(
        constituents,
        &vec![AstNodeId(matmul), AstNodeId(bias), AstNodeId(relu)]
    );
}

#[test]
fn reduction_fusion_populates_fused_node_groups() {
    // Build a stable-softmax graph: exp(x - reduce_max(x)) / reduce_sum(exp(...))
    let mut g = FusionGraph::new();
    let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
    let rmax = g.add_node(FusionOp::Reduction("reduce_max".into()), vec![x]);
    g.set_type_info(rmax, vec![1024], DType::F32);
    let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, rmax]);
    let exp = g.add_node(FusionOp::Elementwise("exp".into()), vec![sub]);
    let rsum = g.add_node(FusionOp::Reduction("reduce_sum".into()), vec![exp]);
    g.set_type_info(rsum, vec![1024], DType::F32);
    let div = g.add_named_node(
        "out".into(),
        FusionOp::Elementwise("div".into()),
        vec![exp, rsum],
    );
    g.mark_graph_output(div);
    g.build_consumers();

    let matches = detect_reduction_patterns(&g);
    assert_eq!(matches.len(), 1);
    let root = matches[0].root_node;
    let all_matched = matches[0].all_matched_nodes.clone();

    let mut plan = FusionPlan::default();
    apply_reduction_fusion(&mut g, &matches, 0, &mut plan);

    let constituents = plan
        .fused_node_groups
        .get(&AstNodeId(root))
        .expect("reduction root should be keyed into fused_node_groups");
    let expected: Vec<AstNodeId> = all_matched.iter().copied().map(AstNodeId).collect();
    assert_eq!(constituents, &expected);
}
