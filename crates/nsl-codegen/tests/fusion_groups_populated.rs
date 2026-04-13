//! Task 1: apply_epilogue_fusion populates FusionPlan.fused_node_groups.

use nsl_ast::NodeId as AstNodeId;
use nsl_codegen::epilogue_fusion::{
    apply_epilogue_fusion, EpilogueChain, EpilogueOp, MatmulKind,
};
use nsl_codegen::fusion_graph::{FusionGraph, FusionOp};
use nsl_codegen::wrga_fusion::FusionPlan;

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
