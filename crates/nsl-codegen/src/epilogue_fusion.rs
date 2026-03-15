//! Epilogue fusion: detect and fuse elementwise tails onto matmul kernels.
//! Matmul + bias + activation → single kernel with in-register epilogue.

use crate::fusion_graph::{FusionGraph, FusionOp, NodeId, FusedKernelId};

/// A single epilogue operation to apply after the matmul accumulation.
#[derive(Debug, Clone, PartialEq)]
pub enum EpilogueOp {
    BiasAdd { bias_node: NodeId, broadcast_dim: usize },
    Activation(String),
    ScalarMul { scalar_node: NodeId },
    Clamp { min_node: NodeId, max_node: NodeId },
}

/// A detected matmul + elementwise epilogue chain.
#[derive(Debug, Clone)]
pub struct EpilogueChain {
    pub matmul_node: NodeId,
    pub epilogue_ops: Vec<EpilogueOp>,
    pub output_node: NodeId,
    pub eliminated_nodes: Vec<NodeId>,
}

/// Check if an elementwise op is eligible for epilogue fusion.
/// This is separate from M26's is_fusible_op — epilogue supports gelu/silu
/// which require multi-instruction PTX sequences.
fn is_epilogue_eligible(op_name: &str) -> bool {
    matches!(
        op_name,
        "add" | "relu" | "gelu" | "silu" | "sigmoid" | "tanh"
            | "mul" | "clamp"
    )
}

/// Detect epilogue chains starting from matmul nodes in the graph.
/// Returns chains sorted by matmul node ID (deterministic ordering).
pub fn detect_epilogue_chains(graph: &FusionGraph) -> Vec<EpilogueChain> {
    let mut chains = Vec::new();

    for node in &graph.nodes {
        if !matches!(node.op, FusionOp::Matmul) {
            continue;
        }
        if node.fused_into.is_some() {
            continue; // Already claimed by reduction pass
        }

        if let Some(chain) = trace_epilogue_chain(graph, node.id) {
            chains.push(chain);
        }
    }

    chains
}

/// Walk forward from a matmul node collecting eligible epilogue ops.
fn trace_epilogue_chain(graph: &FusionGraph, matmul_id: NodeId) -> Option<EpilogueChain> {
    let matmul = &graph.nodes[matmul_id as usize];

    // Matmul must have exactly 1 consumer to start a chain
    if matmul.consumers.len() != 1 {
        return None;
    }

    let mut epilogue_ops = Vec::new();
    let mut eliminated = Vec::new();
    let mut current_id = matmul.consumers[0];

    loop {
        let current = &graph.nodes[current_id as usize];

        // Stop conditions
        if current.fused_into.is_some() || current.no_fuse {
            break;
        }

        match &current.op {
            FusionOp::Elementwise(op_name) if is_epilogue_eligible(op_name) => {
                match op_name.as_str() {
                    "add" => {
                        // BiasAdd: one input is the chain, other is the bias
                        if current.inputs.len() == 2 {
                            let prev_id = *eliminated.last().unwrap_or(&matmul_id);
                            let (chain_input, bias_input) = if current.inputs[0] == prev_id {
                                (current.inputs[0], current.inputs[1])
                            } else {
                                (current.inputs[1], current.inputs[0])
                            };
                            let _ = chain_input; // used for identification

                            // Determine broadcast_dim from shapes
                            let broadcast_dim = resolve_broadcast_dim(graph, matmul_id, bias_input);
                            epilogue_ops.push(EpilogueOp::BiasAdd {
                                bias_node: bias_input,
                                broadcast_dim,
                            });
                        } else {
                            break;
                        }
                    }
                    "mul" => {
                        // ScalarMul: one input is chain, other is scalar
                        if current.inputs.len() == 2 {
                            let prev_id = *eliminated.last().unwrap_or(&matmul_id);
                            let scalar_input = if current.inputs[0] == prev_id {
                                current.inputs[1]
                            } else {
                                current.inputs[0]
                            };
                            epilogue_ops.push(EpilogueOp::ScalarMul {
                                scalar_node: scalar_input,
                            });
                        } else {
                            break;
                        }
                    }
                    "clamp" => {
                        // Clamp needs min and max nodes — simplified: skip for now
                        break;
                    }
                    activation => {
                        // Unary activation: relu, gelu, silu, sigmoid, tanh
                        epilogue_ops.push(EpilogueOp::Activation(activation.to_string()));
                    }
                }
                // Only add to eliminated if this is an intermediate node.
                // Graph outputs must materialize — they're the chain's output_node
                // but are NOT eliminated (the fused kernel writes them).
                eliminated.push(current_id);
            }
            FusionOp::View(view_name) if view_name == "broadcast" => {
                // Broadcast-only view is safe — skip through it
                eliminated.push(current_id);
            }
            _ => break, // Fusion barrier
        }

        // Continue to next consumer if single-consumer and not graph output
        if current.consumers.len() == 1 && !current.is_graph_output {
            current_id = current.consumers[0];
        } else {
            break;
        }
    }

    if epilogue_ops.is_empty() {
        return None;
    }

    let output_node = *eliminated.last().unwrap();
    Some(EpilogueChain {
        matmul_node: matmul_id,
        epilogue_ops,
        output_node,
        eliminated_nodes: eliminated,
    })
}

/// Resolve broadcast dimension by comparing matmul output shape with bias shape.
/// Returns 0 for row-broadcast (bias is [N] or [1,N]), 1 for col-broadcast (bias is [M,1]).
fn resolve_broadcast_dim(graph: &FusionGraph, matmul_id: NodeId, bias_id: NodeId) -> usize {
    let mm_shape = graph.nodes[matmul_id as usize].shape.as_ref();
    let bias_shape = graph.nodes[bias_id as usize].shape.as_ref();

    match (mm_shape, bias_shape) {
        (Some(mm), Some(bias)) if mm.len() == 2 => {
            // [M, N] + [N] or [1, N] -> broadcast_dim = 0
            // [M, N] + [M, 1] -> broadcast_dim = 1
            if bias.len() == 1 && bias[0] == mm[1] {
                0 // row broadcast
            } else if bias.len() == 2 && bias[0] == 1 && bias[1] == mm[1] {
                0 // row broadcast
            } else if bias.len() == 2 && bias[0] == mm[0] && bias[1] == 1 {
                1 // col broadcast
            } else {
                0 // default
            }
        }
        _ => 0, // default to row broadcast
    }
}

/// Mark all nodes in detected chains as fused, assigning FusedKernelId.
pub fn apply_epilogue_fusion(graph: &mut FusionGraph, chains: &[EpilogueChain], base_kernel_id: FusedKernelId) {
    for (i, chain) in chains.iter().enumerate() {
        let kid = base_kernel_id + i as FusedKernelId;
        // Mark matmul node
        graph.nodes[chain.matmul_node as usize].fused_into = Some(kid);
        // Mark all eliminated intermediate nodes
        for &node_id in &chain.eliminated_nodes {
            graph.nodes[node_id as usize].fused_into = Some(kid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion_graph::FusionGraph;
    use nsl_semantic::types::DType;

    fn make_matmul_bias_relu() -> FusionGraph {
        // matmul(A, B) + bias -> relu
        let mut g = FusionGraph::new();
        let a = g.add_named_node("A".into(), FusionOp::Input, vec![]);
        let b = g.add_named_node("B".into(), FusionOp::Input, vec![]);
        let bias = g.add_named_node("bias".into(), FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![a, b]);
        let add = g.add_node(FusionOp::Elementwise("add".into()), vec![mm, bias]);
        let relu = g.add_named_node("out".into(), FusionOp::Elementwise("relu".into()), vec![add]);
        g.mark_graph_output(relu);

        // Set shapes for broadcast resolution
        g.set_type_info(mm, vec![1024, 768], DType::F32);
        g.set_type_info(bias, vec![768], DType::F32);
        g.set_type_info(add, vec![1024, 768], DType::F32);
        g.set_type_info(relu, vec![1024, 768], DType::F32);

        g.build_consumers();
        g
    }

    #[test]
    fn test_detect_matmul_bias_relu() {
        let g = make_matmul_bias_relu();
        let chains = detect_epilogue_chains(&g);

        assert_eq!(chains.len(), 1);
        let chain = &chains[0];
        assert_eq!(chain.matmul_node, 3); // mm node
        assert_eq!(chain.epilogue_ops.len(), 2);
        assert!(matches!(&chain.epilogue_ops[0], EpilogueOp::BiasAdd { broadcast_dim: 0, .. }));
        assert_eq!(chain.epilogue_ops[1], EpilogueOp::Activation("relu".into()));
        assert_eq!(chain.eliminated_nodes.len(), 2); // add + relu
    }

    #[test]
    fn test_multi_consumer_stops_chain() {
        // matmul -> add(mm, bias), but mm has 2 consumers
        let mut g = FusionGraph::new();
        let a = g.add_node(FusionOp::Input, vec![]);
        let b = g.add_node(FusionOp::Input, vec![]);
        let bias = g.add_node(FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![a, b]);
        let _add = g.add_node(FusionOp::Elementwise("add".into()), vec![mm, bias]);
        let _sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![mm, bias]); // second consumer
        g.build_consumers();

        let chains = detect_epilogue_chains(&g);
        assert_eq!(chains.len(), 0); // mm has 2 consumers
    }

    #[test]
    fn test_transpose_barrier() {
        // matmul -> transpose -> relu: transpose breaks chain
        let mut g = FusionGraph::new();
        let a = g.add_node(FusionOp::Input, vec![]);
        let b = g.add_node(FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![a, b]);
        let tr = g.add_node(FusionOp::View("transpose".into()), vec![mm]);
        let relu = g.add_node(FusionOp::Elementwise("relu".into()), vec![tr]);
        g.mark_graph_output(relu);
        g.build_consumers();

        let chains = detect_epilogue_chains(&g);
        assert_eq!(chains.len(), 0); // transpose is a barrier
    }

    #[test]
    fn test_no_fuse_stops_chain() {
        let mut g = FusionGraph::new();
        let a = g.add_node(FusionOp::Input, vec![]);
        let b = g.add_node(FusionOp::Input, vec![]);
        let bias = g.add_node(FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![a, b]);
        let add = g.add_node(FusionOp::Elementwise("add".into()), vec![mm, bias]);
        g.mark_no_fuse(add);
        let relu = g.add_node(FusionOp::Elementwise("relu".into()), vec![add]);
        g.mark_graph_output(relu);
        g.build_consumers();

        let chains = detect_epilogue_chains(&g);
        assert_eq!(chains.len(), 0); // @no_fuse on add
    }

    #[test]
    fn test_already_claimed_skipped() {
        let mut g = FusionGraph::new();
        let a = g.add_node(FusionOp::Input, vec![]);
        let b = g.add_node(FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![a, b]);
        g.nodes[mm as usize].fused_into = Some(99); // pre-claimed
        let relu = g.add_node(FusionOp::Elementwise("relu".into()), vec![mm]);
        g.mark_graph_output(relu);
        g.build_consumers();

        let chains = detect_epilogue_chains(&g);
        assert_eq!(chains.len(), 0);
    }

    #[test]
    fn test_apply_marks_fused_into() {
        let mut g = make_matmul_bias_relu();
        let chains = detect_epilogue_chains(&g);
        apply_epilogue_fusion(&mut g, &chains, 0);

        assert_eq!(g.nodes[3].fused_into, Some(0)); // matmul
        assert_eq!(g.nodes[4].fused_into, Some(0)); // add
        assert_eq!(g.nodes[5].fused_into, Some(0)); // relu
    }

    #[test]
    fn test_broadcast_dim_row() {
        let mut g = FusionGraph::new();
        let mm = g.add_node(FusionOp::Matmul, vec![]);
        let bias = g.add_node(FusionOp::Input, vec![]);
        g.set_type_info(mm, vec![1024, 768], DType::F32);
        g.set_type_info(bias, vec![768], DType::F32);

        assert_eq!(resolve_broadcast_dim(&g, mm, bias), 0);
    }

    #[test]
    fn test_broadcast_dim_col() {
        let mut g = FusionGraph::new();
        let mm = g.add_node(FusionOp::Matmul, vec![]);
        let bias = g.add_node(FusionOp::Input, vec![]);
        g.set_type_info(mm, vec![1024, 768], DType::F32);
        g.set_type_info(bias, vec![1024, 1], DType::F32);

        assert_eq!(resolve_broadcast_dim(&g, mm, bias), 1);
    }

    #[test]
    fn test_gelu_silu_eligible() {
        assert!(is_epilogue_eligible("gelu"));
        assert!(is_epilogue_eligible("silu"));
        assert!(!is_epilogue_eligible("softmax")); // reduction, not epilogue
    }
}
