//! Expression DAG for graph-level fusion analysis.
//! Builds a consumer-counting DAG via ANF linearization of function bodies.

use std::collections::HashMap;
use nsl_semantic::types::DType;

pub type NodeId = u32;
pub type FusedKernelId = u32;

/// Classification of operations for fusion analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionOp {
    Input,
    Matmul,
    Elementwise(String),
    Reduction(String),
    View(String),
    FlashAttention,
    MoEDispatch,   // M32: fusion barrier — never fuse into or out of
    Other,
}

/// A node in the fusion DAG.
#[derive(Debug, Clone)]
pub struct FusionNode {
    pub id: NodeId,
    pub name: Option<String>,
    pub op: FusionOp,
    pub inputs: Vec<NodeId>,
    pub consumers: Vec<NodeId>,
    pub is_graph_output: bool,
    pub fused_into: Option<FusedKernelId>,
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<DType>,
    pub no_fuse: bool,
    /// Scalar constant value, if this node represents a compile-time f32 constant.
    /// Set by the frontend when it lowers a literal into the fusion graph.
    pub const_value: Option<f32>,
}

/// The fusion analysis DAG for a single function.
pub struct FusionGraph {
    pub nodes: Vec<FusionNode>,
    pub name_to_node: HashMap<String, NodeId>,
}

impl Default for FusionGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionGraph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            name_to_node: HashMap::new(),
        }
    }

    /// Add a node and return its ID.
    pub fn add_node(&mut self, op: FusionOp, inputs: Vec<NodeId>) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(FusionNode {
            id,
            name: None,
            op,
            inputs,
            consumers: Vec::new(),
            is_graph_output: false,
            fused_into: None,
            shape: None,
            dtype: None,
            no_fuse: false,
            const_value: None,
        });
        id
    }

    /// Add a scalar constant node (e.g., a literal used as a ScalarMul argument).
    pub fn add_const_node(&mut self, value: f32) -> NodeId {
        let id = self.add_node(FusionOp::Other, vec![]);
        self.nodes[id as usize].const_value = Some(value);
        id
    }

    /// Add a named node (from a let-binding).
    pub fn add_named_node(&mut self, name: String, op: FusionOp, inputs: Vec<NodeId>) -> NodeId {
        let id = self.add_node(op, inputs);
        self.nodes[id as usize].name = Some(name.clone());
        self.name_to_node.insert(name, id);
        id
    }

    /// Mark a node as a graph output (referenced by return).
    pub fn mark_graph_output(&mut self, id: NodeId) {
        self.nodes[id as usize].is_graph_output = true;
    }

    /// Mark a node with @no_fuse.
    pub fn mark_no_fuse(&mut self, id: NodeId) {
        self.nodes[id as usize].no_fuse = true;
    }

    /// Set shape and dtype on a node.
    pub fn set_type_info(&mut self, id: NodeId, shape: Vec<usize>, dtype: DType) {
        let node = &mut self.nodes[id as usize];
        node.shape = Some(shape);
        node.dtype = Some(dtype);
    }

    /// Build consumer back-links from input forward-links.
    /// Must be called after all nodes are added.
    pub fn build_consumers(&mut self) {
        // Clear existing consumers
        for node in &mut self.nodes {
            node.consumers.clear();
        }
        // Build from inputs
        for i in 0..self.nodes.len() {
            let id = i as NodeId;
            let inputs: Vec<NodeId> = self.nodes[i].inputs.clone();
            for input_id in inputs {
                self.nodes[input_id as usize].consumers.push(id);
            }
        }
    }

    /// Check if a node is a single-consumer intermediate that can be eliminated
    /// by fusing it into its sole downstream consumer's kernel.
    /// Requires: exactly 1 consumer, not a graph output, not @no_fuse,
    /// not FlashAttention, not MoEDispatch, and not already claimed by a fusion pass.
    pub fn is_single_consumer_intermediate(&self, node_id: NodeId) -> bool {
        let node = &self.nodes[node_id as usize];
        node.consumers.len() == 1
            && !node.is_graph_output
            && !node.no_fuse
            && !matches!(node.op, FusionOp::FlashAttention | FusionOp::MoEDispatch)
            && node.fused_into.is_none()
    }

    /// Compute bytes for one materialization of a node's output.
    pub fn node_bytes(&self, node_id: NodeId) -> u64 {
        let node = &self.nodes[node_id as usize];
        match (&node.shape, &node.dtype) {
            (Some(shape), Some(dtype)) => {
                let elements: u64 = shape.iter().map(|&d| d as u64).product();
                elements * dtype.byte_width() as u64
            }
            _ => 0,
        }
    }

    /// Lookup a node by let-binding name.
    pub fn lookup(&self, name: &str) -> Option<NodeId> {
        self.name_to_node.get(name).copied()
    }
}

/// Classify a builtin function name into a FusionOp.
pub fn classify_op(name: &str) -> FusionOp {
    match name {
        "matmul" => FusionOp::Matmul,
        "add" | "sub" | "mul" | "div" | "pow" | "neg" | "abs"
        | "relu" | "gelu" | "silu" | "sigmoid" | "tanh"
        | "exp" | "log" | "sqrt" | "sign" | "clamp" => {
            FusionOp::Elementwise(name.to_string())
        }
        "sum" | "mean" | "reduce_max" | "reduce_min" | "var" => {
            FusionOp::Reduction(name.to_string())
        }
        "softmax" | "layernorm" | "rmsnorm" => {
            FusionOp::Reduction(name.to_string())
        }
        "transpose" | "reshape" | "broadcast" | "expand" => {
            FusionOp::View(name.to_string())
        }
        "scaled_dot_product_attention" => FusionOp::FlashAttention,
        _ => FusionOp::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anf_linearization_simple() {
        // relu(matmul(A, B) + C) should produce 6 nodes
        let mut g = FusionGraph::new();
        let a = g.add_named_node("A".into(), FusionOp::Input, vec![]);
        let b = g.add_named_node("B".into(), FusionOp::Input, vec![]);
        let c = g.add_named_node("C".into(), FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![a, b]);
        let add = g.add_node(FusionOp::Elementwise("add".into()), vec![mm, c]);
        let relu = g.add_named_node("out".into(), FusionOp::Elementwise("relu".into()), vec![add]);
        g.mark_graph_output(relu);
        g.build_consumers();

        assert_eq!(g.nodes.len(), 6);
        assert_eq!(g.nodes[mm as usize].consumers, vec![add]);
        assert_eq!(g.nodes[add as usize].consumers, vec![relu]);
        assert!(g.nodes[relu as usize].is_graph_output);
    }

    #[test]
    fn test_consumer_counting() {
        // x consumed by both add and sub -> 2 consumers
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let y = g.add_named_node("y".into(), FusionOp::Input, vec![]);
        let add = g.add_node(FusionOp::Elementwise("add".into()), vec![x, y]);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, y]);
        // add has exactly 1 consumer (out), sub has 0
        let out = g.add_node(FusionOp::Elementwise("relu".into()), vec![add]);
        g.mark_graph_output(out);
        g.build_consumers();

        assert_eq!(g.nodes[x as usize].consumers.len(), 2);
        assert!(!g.is_single_consumer_intermediate(x)); // multi-consumer -> not fusible
        assert!(g.is_single_consumer_intermediate(add)); // exactly 1 consumer -> fusible
        assert!(!g.is_single_consumer_intermediate(sub)); // 0 consumers -> not fusible (no downstream to fuse into)
    }

    #[test]
    fn test_graph_output_not_fusible() {
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let relu = g.add_named_node("out".into(), FusionOp::Elementwise("relu".into()), vec![x]);
        g.mark_graph_output(relu);
        g.build_consumers();

        // relu is graph output — even with 0 consumers, is_fusible checks is_graph_output
        assert!(!g.is_single_consumer_intermediate(relu));
    }

    #[test]
    fn test_flash_attention_barrier() {
        let mut g = FusionGraph::new();
        let x = g.add_node(FusionOp::Input, vec![]);
        let fa = g.add_node(FusionOp::FlashAttention, vec![x]);
        g.build_consumers();

        assert!(!g.is_single_consumer_intermediate(fa));
    }

    #[test]
    fn test_no_fuse_annotation() {
        let mut g = FusionGraph::new();
        let x = g.add_node(FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![x]);
        g.mark_no_fuse(mm);
        g.build_consumers();

        assert!(!g.is_single_consumer_intermediate(mm));
    }

    #[test]
    fn test_node_bytes() {
        let mut g = FusionGraph::new();
        let x = g.add_node(FusionOp::Input, vec![]);
        g.set_type_info(x, vec![1024, 768], DType::F32);

        assert_eq!(g.node_bytes(x), 1024 * 768 * 4);
    }

    #[test]
    fn test_node_bytes_unknown() {
        let mut g = FusionGraph::new();
        let x = g.add_node(FusionOp::Input, vec![]);
        // No type info set
        assert_eq!(g.node_bytes(x), 0);
    }

    #[test]
    fn test_classify_op() {
        assert_eq!(classify_op("matmul"), FusionOp::Matmul);
        assert_eq!(classify_op("relu"), FusionOp::Elementwise("relu".into()));
        assert_eq!(classify_op("gelu"), FusionOp::Elementwise("gelu".into()));
        assert_eq!(classify_op("sum"), FusionOp::Reduction("sum".into()));
        assert_eq!(classify_op("softmax"), FusionOp::Reduction("softmax".into()));
        assert_eq!(classify_op("transpose"), FusionOp::View("transpose".into()));
        assert_eq!(classify_op("scaled_dot_product_attention"), FusionOp::FlashAttention);
        assert_eq!(classify_op("some_unknown"), FusionOp::Other);
    }

    #[test]
    fn test_lookup_by_name() {
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        assert_eq!(g.lookup("x"), Some(x));
        assert_eq!(g.lookup("y"), None);
    }

    #[test]
    fn test_fused_into_blocks_fusion() {
        let mut g = FusionGraph::new();
        let x = g.add_node(FusionOp::Input, vec![]);
        let relu = g.add_node(FusionOp::Elementwise("relu".into()), vec![x]);
        // Add a consumer so relu has exactly 1 consumer -> fusible
        let out = g.add_node(FusionOp::Elementwise("sigmoid".into()), vec![relu]);
        g.mark_graph_output(out);
        g.build_consumers();

        // Before marking as fused
        assert!(g.is_single_consumer_intermediate(relu));

        // After marking
        g.nodes[relu as usize].fused_into = Some(0);
        assert!(!g.is_single_consumer_intermediate(relu));
    }

    #[test]
    fn test_pass_ordering_reduction_before_epilogue() {
        // Softmax pattern should be claimed by reduction pass,
        // preventing the elementwise pass from stealing exp/div nodes.
        use crate::epilogue_fusion::detect_epilogue_chains;
        use crate::reduction_fusion::{detect_reduction_patterns, apply_reduction_fusion};

        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let rmax = g.add_node(FusionOp::Reduction("reduce_max".into()), vec![x]);
        g.set_type_info(rmax, vec![1024], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, rmax]);
        let exp = g.add_node(FusionOp::Elementwise("exp".into()), vec![sub]);
        let rsum = g.add_node(FusionOp::Reduction("reduce_sum".into()), vec![exp]);
        g.set_type_info(rsum, vec![1024], DType::F32);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![exp, rsum]);
        g.mark_graph_output(div);
        g.build_consumers();

        // Pass 1: Reduction fusion claims the softmax nodes
        let reduction_matches = detect_reduction_patterns(&g);
        assert_eq!(reduction_matches.len(), 1);
        apply_reduction_fusion(&mut g, &reduction_matches, 0);

        // Pass 2: Epilogue fusion should find nothing (no matmul, and nodes are claimed)
        let epilogue_chains = detect_epilogue_chains(&g);
        assert_eq!(epilogue_chains.len(), 0);

        // Verify claimed nodes
        assert!(g.nodes[div as usize].fused_into.is_some());
        assert!(g.nodes[exp as usize].fused_into.is_some());
    }

    #[test]
    fn test_pass_ordering_mixed_graph() {
        // Graph with matmul+relu feeding into softmax.
        // Reduction claims softmax nodes (div, exp, rsum, sub, rmax).
        // Epilogue then finds matmul+relu as a 1-op epilogue chain:
        // relu is not claimed by reduction, so epilogue fusion captures it.
        // The chain stops at relu (2 consumers: rmax + sub), but relu is still
        // added as an Activation epilogue op before the consumer check halts iteration.
        use crate::epilogue_fusion::detect_epilogue_chains;
        use crate::reduction_fusion::{detect_reduction_patterns, apply_reduction_fusion};

        let mut g = FusionGraph::new();
        // Matmul + relu path
        let a = g.add_named_node("A".into(), FusionOp::Input, vec![]);
        let w = g.add_named_node("W".into(), FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![a, w]);
        g.set_type_info(mm, vec![1024, 768], DType::F32);
        let relu = g.add_node(FusionOp::Elementwise("relu".into()), vec![mm]);

        // Softmax path (on relu output)
        let rmax = g.add_node(FusionOp::Reduction("reduce_max".into()), vec![relu]);
        g.set_type_info(rmax, vec![1024], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![relu, rmax]);
        let exp = g.add_node(FusionOp::Elementwise("exp".into()), vec![sub]);
        let rsum = g.add_node(FusionOp::Reduction("reduce_sum".into()), vec![exp]);
        g.set_type_info(rsum, vec![1024], DType::F32);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![exp, rsum]);
        g.mark_graph_output(div);
        g.build_consumers();

        // Pass 1: Reduction claims softmax nodes (div, exp, rsum, sub, rmax)
        let red = detect_reduction_patterns(&g);
        apply_reduction_fusion(&mut g, &red, 0);

        // Pass 2: Epilogue finds matmul+relu (relu is unclaimed; 2 consumers halt
        // iteration but relu is already collected as an Activation epilogue op).
        let epi = detect_epilogue_chains(&g);
        assert_eq!(epi.len(), 1);
        assert_eq!(epi[0].matmul_node, mm);
        assert_eq!(epi[0].output_node, relu);

        // Softmax nodes remain claimed by reduction kernel 0
        assert!(g.nodes[div as usize].fused_into.is_some());
        assert!(g.nodes[exp as usize].fused_into.is_some());
    }

    #[test]
    fn test_full_pipeline_transformer_block() {
        // Simulates a transformer attention block:
        // matmul(Q, K^T) -> softmax -> matmul(attn, V) -> bias + relu
        use crate::epilogue_fusion::{detect_epilogue_chains, apply_epilogue_fusion};
        use crate::reduction_fusion::{detect_reduction_patterns, apply_reduction_fusion};

        let mut g = FusionGraph::new();
        let q = g.add_named_node("Q".into(), FusionOp::Input, vec![]);
        let k = g.add_named_node("K".into(), FusionOp::Input, vec![]);
        let v = g.add_named_node("V".into(), FusionOp::Input, vec![]);
        let bias = g.add_named_node("bias".into(), FusionOp::Input, vec![]);
        g.set_type_info(bias, vec![64], DType::F32);

        // QK^T
        let qk = g.add_node(FusionOp::Matmul, vec![q, k]);
        g.set_type_info(qk, vec![32, 64, 64], DType::F32);

        // Softmax(QK^T)
        let rmax = g.add_node(FusionOp::Reduction("reduce_max".into()), vec![qk]);
        g.set_type_info(rmax, vec![32, 64, 1], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![qk, rmax]);
        let exp = g.add_node(FusionOp::Elementwise("exp".into()), vec![sub]);
        let rsum = g.add_node(FusionOp::Reduction("reduce_sum".into()), vec![exp]);
        g.set_type_info(rsum, vec![32, 64, 1], DType::F32);
        let div = g.add_node(FusionOp::Elementwise("div".into()), vec![exp, rsum]);

        // attn @ V
        let av = g.add_node(FusionOp::Matmul, vec![div, v]);
        g.set_type_info(av, vec![32, 64, 64], DType::F32);

        // bias + relu
        let add = g.add_node(FusionOp::Elementwise("add".into()), vec![av, bias]);
        let relu = g.add_named_node("out".into(), FusionOp::Elementwise("relu".into()), vec![add]);
        g.mark_graph_output(relu);
        g.build_consumers();

        // Pass 1: Reduction fusion
        let red_matches = detect_reduction_patterns(&g);
        let softmax_count = red_matches.iter().filter(|m| m.pattern == "softmax").count();
        assert_eq!(softmax_count, 1);
        apply_reduction_fusion(&mut g, &red_matches, 0);

        // Pass 2: Epilogue fusion
        let epi_chains = detect_epilogue_chains(&g);
        // qk matmul has 2 consumers (rmax + sub) -> no epilogue
        // av matmul has 1 consumer (add) -> should detect epilogue chain
        assert_eq!(epi_chains.len(), 1);
        assert_eq!(epi_chains[0].matmul_node, av);
        apply_epilogue_fusion(&mut g, &epi_chains, 100);

        // Verify: softmax nodes claimed by kernel 0, epilogue by kernel 100
        assert!(g.nodes[rmax as usize].fused_into.is_some());
        assert!(g.nodes[div as usize].fused_into.is_some());
        assert_eq!(g.nodes[av as usize].fused_into, Some(100));
        assert_eq!(g.nodes[add as usize].fused_into, Some(100));
        assert_eq!(g.nodes[relu as usize].fused_into, Some(100));

        // Unclaimed: q, k, v, bias (inputs), qk (matmul with multi-consumer)
        assert!(g.nodes[q as usize].fused_into.is_none());
        assert!(g.nodes[qk as usize].fused_into.is_none());
    }
}
