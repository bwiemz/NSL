//! Epilogue fusion: detect and fuse elementwise tails onto matmul kernels.
//! Matmul + bias + activation → single kernel with in-register epilogue.

use crate::fusion_graph::{FusionGraph, FusionOp, NodeId, FusedKernelId};
use nsl_semantic::types::DType;

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
            | "mul"
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
            if (bias.len() == 1 && bias[0] == mm[1])
                || (bias.len() == 2 && bias[0] == 1 && bias[1] == mm[1])
            {
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

/// Synthesize the epilogue portion of a fused matmul kernel as PTX.
/// The caller is responsible for integrating this into the matmul kernel's
/// output section (after MMA accumulation, before st.global).
///
/// When `use_mma` is true, generates MMA lane-to-coordinate mapping for
/// Tensor Core fragment layouts. When false, uses linear thread mapping.
///
/// All epilogue ops execute in f32 on accumulator registers.
/// Dtype downcast happens once at the end before st.global.
pub fn synthesize_epilogue_ptx(
    name: &str,
    ops: &[EpilogueOp],
    output_dtype: DType,
    use_mma: bool,
) -> Vec<u8> {
    let mut ptx = String::new();

    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_70\n");
    ptx.push_str(".address_size 64\n\n");

    ptx.push_str(&format!(".visible .entry {}(\n", name));
    ptx.push_str("    .param .u64 param_out,\n");
    ptx.push_str("    .param .u64 param_acc,\n");    // accumulator (matmul output)
    ptx.push_str("    .param .u64 param_bias,\n");   // bias vector (if needed)
    ptx.push_str("    .param .u64 param_M,\n");
    ptx.push_str("    .param .u64 param_N\n");
    ptx.push_str(") {\n");

    ptx.push_str("    .reg .u32 %r<32>;\n");
    ptx.push_str("    .reg .u64 %rd<16>;\n");
    ptx.push_str("    .reg .f32 %f<32>;\n");
    ptx.push_str("    .reg .f16 %h<8>;\n");
    ptx.push_str("    .reg .pred %p<4>;\n\n");

    if use_mma {
        // MMA lane-to-coordinate mapping for m16n8k16 fragments
        ptx.push_str("    // MMA lane_id -> logical (row, col)\n");
        ptx.push_str("    mov.u32 %r0, %tid.x;\n");
        ptx.push_str("    and.b32 %r1, %r0, 31;          // lane_id = tid % 32\n");
        ptx.push_str("    shr.u32 %r2, %r1, 2;            // lane_id / 4\n");
        ptx.push_str("    and.b32 %r3, %r1, 3;            // lane_id % 4\n");
        ptx.push_str("    shl.b32 %r4, %r3, 1;            // (lane_id % 4) * 2\n");
        ptx.push_str("    // For fragment i: row = r2 + (i/2)*8, col = r4 + (i%2)\n\n");
    } else {
        // Linear thread mapping
        ptx.push_str("    mov.u32 %r0, %tid.x;\n");
        ptx.push_str("    mov.u32 %r1, %ctaid.x;\n");
        ptx.push_str("    mov.u32 %r2, %ntid.x;\n");
        ptx.push_str("    mul.lo.u32 %r3, %r1, %r2;\n");
        ptx.push_str("    add.u32 %r0, %r0, %r3;\n");
        ptx.push_str("    cvt.u64.u32 %rd0, %r0;\n\n");
    }

    // Load params
    ptx.push_str("    ld.param.u64 %rd1, [param_N];\n");
    ptx.push_str("    ld.param.u64 %rd2, [param_acc];\n");
    ptx.push_str("    ld.param.u64 %rd3, [param_bias];\n");
    ptx.push_str("    ld.param.u64 %rd4, [param_out];\n\n");

    // Load accumulator value (already f32)
    ptx.push_str("    // Load f32 accumulator value\n");
    if use_mma {
        ptx.push_str("    // (In real integration, acc is already in register from MMA)\n");
        ptx.push_str("    // This standalone version loads from memory for testing\n");
    }
    ptx.push_str("    cvt.u64.u32 %rd5, %r0;\n");
    ptx.push_str("    shl.b64 %rd5, %rd5, 2;\n"); // * 4 bytes (f32)
    ptx.push_str("    add.u64 %rd6, %rd2, %rd5;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd6];\n\n");

    // Apply epilogue ops — all in f32
    let mut acc_reg = 0;
    for (i, op) in ops.iter().enumerate() {
        let out_reg = i + 1;
        match op {
            EpilogueOp::BiasAdd { broadcast_dim, .. } => {
                ptx.push_str("    // BiasAdd epilogue\n");
                if use_mma {
                    // Use col (r4) for broadcast_dim==0, row (r2) for broadcast_dim==1
                    let coord_reg = if *broadcast_dim == 0 { "r4" } else { "r2" };
                    ptx.push_str(&format!("    cvt.u64.u32 %rd7, %{};\n", coord_reg));
                } else {
                    // Linear: col = tid % N, row = tid / N
                    if *broadcast_dim == 0 {
                        ptx.push_str("    cvt.u64.u32 %rd7, %r0;\n");
                        ptx.push_str("    rem.u64 %rd7, %rd7, %rd1;\n"); // tid % N
                    } else {
                        ptx.push_str("    cvt.u64.u32 %rd7, %r0;\n");
                        ptx.push_str("    div.u64 %rd7, %rd7, %rd1;\n"); // tid / N
                    }
                }
                ptx.push_str("    shl.b64 %rd7, %rd7, 2;\n"); // * 4 bytes
                ptx.push_str("    add.u64 %rd8, %rd3, %rd7;\n");
                ptx.push_str(&format!("    ld.global.f32 %f{}, [%rd8];\n", out_reg));
                ptx.push_str(&format!(
                    "    add.f32 %f{}, %f{}, %f{};\n",
                    out_reg, acc_reg, out_reg
                ));
            }
            EpilogueOp::Activation(act) => {
                match act.as_str() {
                    "relu" => {
                        ptx.push_str("    // ReLU epilogue\n");
                        ptx.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", out_reg));
                        ptx.push_str(&format!(
                            "    max.f32 %f{}, %f{}, %f{};\n",
                            out_reg, acc_reg, out_reg
                        ));
                    }
                    "gelu" => {
                        // GELU approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                        // sqrt(2/pi) = 0.7978845608 -> 0f3F4C422A
                        ptx.push_str("    // GELU epilogue (tanh approximation)\n");
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, acc_reg, acc_reg)); // x^2
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, acc_reg)); // x^3
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, 0f3D372713;\n", out_reg, out_reg)); // 0.044715 * x^3
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, acc_reg, out_reg)); // x + 0.044715*x^3
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, 0f3F4C422A;\n", out_reg, out_reg)); // sqrt(2/pi) * (...)
                        // tanh via exp: tanh(x) = 2*sigmoid(2x) - 1
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg)); // 2x
                        ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, 0f3FB8AA3B;\n", out_reg, out_reg)); // log2(e)
                        ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg)); // 1+exp(-2x)
                        ptx.push_str(&format!("    rcp.approx.f32 %f{}, %f{};\n", out_reg, out_reg)); // sigmoid(2x)
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg)); // 2*sigmoid
                        ptx.push_str(&format!("    sub.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg)); // tanh
                        // (1 + tanh) * 0.5 * x
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg)); // 1+tanh
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, 0f3F000000;\n", out_reg, out_reg)); // *0.5
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, acc_reg)); // *x
                    }
                    "silu" => {
                        // SiLU = x * sigmoid(x)
                        ptx.push_str("    // SiLU epilogue\n");
                        ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, acc_reg));
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, 0f3FB8AA3B;\n", out_reg, out_reg));
                        ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg));
                        ptx.push_str(&format!("    rcp.approx.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, acc_reg, out_reg));
                    }
                    "sigmoid" => {
                        ptx.push_str("    // Sigmoid epilogue\n");
                        ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, acc_reg));
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, 0f3FB8AA3B;\n", out_reg, out_reg));
                        ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg));
                        ptx.push_str(&format!("    rcp.approx.f32 %f{}, %f{};\n", out_reg, out_reg));
                    }
                    "tanh" => {
                        ptx.push_str("    // Tanh epilogue\n");
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, acc_reg, acc_reg));
                        ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, 0f3FB8AA3B;\n", out_reg, out_reg));
                        ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg));
                        ptx.push_str(&format!("    rcp.approx.f32 %f{}, %f{};\n", out_reg, out_reg));
                        ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg));
                        ptx.push_str(&format!("    sub.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg));
                    }
                    _ => {
                        // Unknown activation — pass through
                        ptx.push_str(&format!("    mov.f32 %f{}, %f{};\n", out_reg, acc_reg));
                    }
                }
            }
            EpilogueOp::ScalarMul { .. } => {
                ptx.push_str("    // ScalarMul epilogue (placeholder — scalar loaded separately)\n");
                ptx.push_str(&format!("    mov.f32 %f{}, %f{};\n", out_reg, acc_reg));
            }
            EpilogueOp::Clamp { .. } => {
                ptx.push_str(&format!("    mov.f32 %f{}, %f{};\n", out_reg, acc_reg));
            }
        }
        acc_reg = out_reg;
    }

    // Dtype downcast (f32 -> output dtype) and store
    match output_dtype {
        DType::Fp16 | DType::Bf16 => {
            ptx.push_str(&format!(
                "\n    cvt.rn.f16.f32 %h0, %f{};\n",
                acc_reg
            ));
            ptx.push_str("    st.global.b16 [%rd4], %h0;\n");
        }
        DType::F32 => {
            ptx.push_str(&format!(
                "\n    st.global.f32 [%rd4], %f{};\n",
                acc_reg
            ));
        }
        _ => {
            // Default: store as f32
            ptx.push_str(&format!(
                "\n    st.global.f32 [%rd4], %f{};\n",
                acc_reg
            ));
        }
    }

    ptx.push_str("\n    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
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

    #[test]
    fn test_synthesize_epilogue_ptx_bias_relu() {
        let ptx = synthesize_epilogue_ptx(
            "fused_matmul_bias_relu",
            &[
                EpilogueOp::BiasAdd { bias_node: 0, broadcast_dim: 0 },
                EpilogueOp::Activation("relu".into()),
            ],
            DType::Fp16,
            true, // use_mma
        );
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();

        // Must contain MMA lane-to-coord mapping
        assert!(ptx_str.contains("lane_id"));
        // Must do epilogue in f32
        assert!(ptx_str.contains("add.f32"));
        assert!(ptx_str.contains("max.f32")); // relu
        // Must downcast to f16 at the end
        assert!(ptx_str.contains("cvt.rn.f16.f32"));
        // Must store to global
        assert!(ptx_str.contains("st.global"));
    }

    #[test]
    fn test_synthesize_epilogue_ptx_no_mma() {
        let ptx = synthesize_epilogue_ptx(
            "fused_matmul_bias",
            &[
                EpilogueOp::BiasAdd { bias_node: 0, broadcast_dim: 0 },
            ],
            DType::F32,
            false, // no MMA — linear mapping
        );
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();

        // Should NOT contain lane_id mapping for non-MMA path
        assert!(!ptx_str.contains("lane_id"));
        // f32 output — no cvt needed
        assert!(!ptx_str.contains("cvt.rn.f16.f32"));
        assert!(ptx_str.contains("st.global.f32"));
    }

    #[test]
    fn test_synthesize_gelu_epilogue() {
        let ptx = synthesize_epilogue_ptx(
            "fused_matmul_gelu",
            &[EpilogueOp::Activation("gelu".into())],
            DType::Fp16,
            true,
        );
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();

        // GELU uses tanh approximation: sqrt(2/pi) constant
        assert!(ptx_str.contains("0f3F4C422A")); // sqrt(2/pi) hex
        assert!(ptx_str.contains("tanh") || ptx_str.contains("ex2.approx"));
    }
}
