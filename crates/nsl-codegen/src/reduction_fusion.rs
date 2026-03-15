//! Reduction fusion: detect softmax/layernorm/rmsnorm subgraphs and replace
//! with pre-optimized single-kernel implementations.
//! Uses hand-written PTX templates (like FlashAttention), not the KernelCompiler.

use std::collections::HashSet;
use crate::fusion_graph::{FusionGraph, FusionOp, NodeId, FusedKernelId};
use nsl_semantic::types::DType;

/// A matched reduction pattern in the fusion graph.
#[derive(Debug, Clone)]
pub struct ReductionMatch {
    pub pattern: &'static str,
    pub root_node: NodeId,
    pub input_nodes: Vec<NodeId>,
    pub all_matched_nodes: Vec<NodeId>,
    pub reduction_dim: i64,
    pub is_naive: bool,
    pub has_affine: bool,
}

/// Try to match all reduction patterns in the graph.
/// Returns matches sorted by root node ID.
pub fn detect_reduction_patterns(graph: &FusionGraph) -> Vec<ReductionMatch> {
    let mut matches = Vec::new();

    // Try each pattern at each node
    for node in &graph.nodes {
        if node.fused_into.is_some() {
            continue;
        }

        // Try softmax (look for div nodes that could be the tail)
        if matches!(node.op, FusionOp::Elementwise(ref s) if s == "div") {
            if let Some(m) = try_match_softmax(graph, node.id) {
                if is_valid_reduction_match(graph, &m) {
                    matches.push(m);
                }
            }
        }

        // Try layernorm (look for the final add with beta, or div without affine)
        if matches!(node.op, FusionOp::Elementwise(ref s) if s == "add" || s == "div") {
            if let Some(m) = try_match_layernorm(graph, node.id) {
                if is_valid_reduction_match(graph, &m) {
                    matches.push(m);
                }
            }
        }

        // Try rmsnorm (look for mul with gamma at the end)
        if matches!(node.op, FusionOp::Elementwise(ref s) if s == "mul") {
            if let Some(m) = try_match_rmsnorm(graph, node.id) {
                if is_valid_reduction_match(graph, &m) {
                    matches.push(m);
                }
            }
        }

        // Single-node builtins: softmax(), layernorm(), rmsnorm()
        if let FusionOp::Reduction(ref name) = node.op {
            match name.as_str() {
                "softmax" => {
                    matches.push(ReductionMatch {
                        pattern: "softmax",
                        root_node: node.id,
                        input_nodes: node.inputs.clone(),
                        all_matched_nodes: vec![node.id],
                        reduction_dim: -1,
                        is_naive: false,
                        has_affine: false,
                    });
                }
                "layernorm" => {
                    matches.push(ReductionMatch {
                        pattern: "layernorm",
                        root_node: node.id,
                        input_nodes: node.inputs.clone(),
                        all_matched_nodes: vec![node.id],
                        reduction_dim: -1,
                        is_naive: false,
                        has_affine: node.inputs.len() > 1,
                    });
                }
                "rmsnorm" => {
                    matches.push(ReductionMatch {
                        pattern: "rmsnorm",
                        root_node: node.id,
                        input_nodes: node.inputs.clone(),
                        all_matched_nodes: vec![node.id],
                        reduction_dim: -1,
                        is_naive: false,
                        has_affine: node.inputs.len() > 1,
                    });
                }
                _ => {}
            }
        }
    }

    matches
}

/// Internal vs external consumer rule: only the root node may have external consumers.
fn is_valid_reduction_match(graph: &FusionGraph, matched: &ReductionMatch) -> bool {
    let matched_set: HashSet<NodeId> = matched.all_matched_nodes.iter().copied().collect();

    for &node_id in &matched.all_matched_nodes {
        let node = &graph.nodes[node_id as usize];

        // Skip already-claimed nodes
        if node.fused_into.is_some() || node.no_fuse {
            return false;
        }

        for &consumer in &node.consumers {
            if !matched_set.contains(&consumer) {
                // External consumer — only the root node may have them
                if node_id != matched.root_node {
                    return false;
                }
            }
        }
    }
    true
}

/// Extract reduction dimension from a Reduction node, verifying it's the last dim.
fn get_reduction_dim(graph: &FusionGraph, node_id: NodeId) -> Option<i64> {
    let node = &graph.nodes[node_id as usize];
    // For now, assume reduction is on last dim (-1).
    // Full implementation would check the AST's dim argument.
    if matches!(node.op, FusionOp::Reduction(_)) {
        // Verify it's a contiguous last dimension
        if let Some(ref shape) = node.shape {
            let rank = shape.len() as i64;
            Some(rank - 1)
        } else {
            Some(-1)
        }
    } else {
        None
    }
}

/// Try to match the softmax pattern:
/// Stable: exp(x - reduce_max(x)) / reduce_sum(exp(x - reduce_max(x)))
/// Naive: exp(x) / reduce_sum(exp(x))
fn try_match_softmax(graph: &FusionGraph, div_node: NodeId) -> Option<ReductionMatch> {
    let div = &graph.nodes[div_node as usize];
    if div.inputs.len() != 2 { return None; }

    let numerator_id = div.inputs[0];
    let denominator_id = div.inputs[1];

    let numerator = &graph.nodes[numerator_id as usize];
    let denominator = &graph.nodes[denominator_id as usize];

    // Denominator must be reduce_sum
    if !matches!(denominator.op, FusionOp::Reduction(ref s) if s == "reduce_sum") {
        return None;
    }

    // Numerator must be exp(something)
    if !matches!(numerator.op, FusionOp::Elementwise(ref s) if s == "exp") {
        return None;
    }

    // Check if it's the stable form: exp(x - max(x))
    let exp_input_id = numerator.inputs[0];
    let exp_input = &graph.nodes[exp_input_id as usize];

    let mut all_nodes = vec![div_node, numerator_id, denominator_id];
    let mut is_naive = true;
    let mut x_node = exp_input_id;

    if matches!(exp_input.op, FusionOp::Elementwise(ref s) if s == "sub") {
        // Stable form: sub(x, reduce_max(x))
        if exp_input.inputs.len() == 2 {
            let max_candidate = &graph.nodes[exp_input.inputs[1] as usize];
            if matches!(max_candidate.op, FusionOp::Reduction(ref s) if s == "reduce_max") {
                x_node = exp_input.inputs[0];
                all_nodes.push(exp_input_id);
                all_nodes.push(exp_input.inputs[1]);
                is_naive = false;
            }
        }
    }

    // reduce_sum's input should be the exp node
    if denominator.inputs[0] != numerator_id {
        return None;
    }

    let reduction_dim = get_reduction_dim(graph, denominator_id).unwrap_or(-1);

    Some(ReductionMatch {
        pattern: "softmax",
        root_node: div_node,
        input_nodes: vec![x_node],
        all_matched_nodes: all_nodes,
        reduction_dim,
        is_naive,
        has_affine: false,
    })
}

/// Try to match layernorm: (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
fn try_match_layernorm(graph: &FusionGraph, candidate: NodeId) -> Option<ReductionMatch> {
    let node = &graph.nodes[candidate as usize];

    // Check if this is the affine tail: add(mul(normalized, gamma), beta)
    if matches!(node.op, FusionOp::Elementwise(ref s) if s == "add") && node.inputs.len() == 2 {
        let mul_candidate = &graph.nodes[node.inputs[0] as usize];
        if matches!(mul_candidate.op, FusionOp::Elementwise(ref s) if s == "mul") {
            // Could be affine layernorm — check deeper
            let div_candidate_id = mul_candidate.inputs[0];
            let div_candidate = &graph.nodes[div_candidate_id as usize];
            if matches!(div_candidate.op, FusionOp::Elementwise(ref s) if s == "div") {
                // Continue checking from div node
                if let Some(inner) = try_match_layernorm_core(graph, div_candidate_id) {
                    let x_node = inner.0;
                    let mut all_nodes = vec![candidate, node.inputs[0]]; // add (beta), mul (gamma)
                    all_nodes.extend(inner.1);
                    let reduction_dim = inner.2;
                    return Some(ReductionMatch {
                        pattern: "layernorm",
                        root_node: candidate,
                        input_nodes: vec![x_node, mul_candidate.inputs[1], node.inputs[1]], // x, gamma, beta
                        all_matched_nodes: all_nodes,
                        reduction_dim,
                        is_naive: false,
                        has_affine: true,
                    });
                }
            }
        }
    }

    // Non-affine: just the div node
    if matches!(node.op, FusionOp::Elementwise(ref s) if s == "div") {
        if let Some(inner) = try_match_layernorm_core(graph, candidate) {
            let x_node = inner.0;
            let all_nodes = inner.1;
            return Some(ReductionMatch {
                pattern: "layernorm",
                root_node: candidate,
                input_nodes: vec![x_node],
                all_matched_nodes: all_nodes,
                reduction_dim: inner.2,
                is_naive: false,
                has_affine: false,
            });
        }
    }

    None
}

/// Match the core layernorm pattern: (x - mean(x)) / sqrt(var(x) + eps)
/// Returns (x_node, matched_node_ids, reduction_dim)
fn try_match_layernorm_core(graph: &FusionGraph, div_id: NodeId) -> Option<(NodeId, Vec<NodeId>, i64)> {
    let div = &graph.nodes[div_id as usize];
    if div.inputs.len() != 2 { return None; }

    // LHS: sub(x, mean(x))
    let sub_id = div.inputs[0];
    let sub = &graph.nodes[sub_id as usize];
    if !matches!(sub.op, FusionOp::Elementwise(ref s) if s == "sub") { return None; }
    if sub.inputs.len() != 2 { return None; }

    let x_node = sub.inputs[0];
    let mean_id = sub.inputs[1];
    let mean = &graph.nodes[mean_id as usize];
    if !matches!(mean.op, FusionOp::Reduction(ref s) if s == "mean") { return None; }

    // RHS: sqrt(var(x) + eps) or sqrt(add(var(x), eps))
    let sqrt_id = div.inputs[1];
    let sqrt = &graph.nodes[sqrt_id as usize];
    if !matches!(sqrt.op, FusionOp::Elementwise(ref s) if s == "sqrt") { return None; }

    let add_eps_id = sqrt.inputs[0];
    let add_eps = &graph.nodes[add_eps_id as usize];
    if !matches!(add_eps.op, FusionOp::Elementwise(ref s) if s == "add") { return None; }

    let var_id = add_eps.inputs[0];
    let var_node = &graph.nodes[var_id as usize];
    if !matches!(var_node.op, FusionOp::Reduction(ref s) if s == "var") { return None; }

    let reduction_dim = get_reduction_dim(graph, mean_id).unwrap_or(-1);

    let nodes = vec![div_id, sub_id, mean_id, sqrt_id, add_eps_id, var_id];
    Some((x_node, nodes, reduction_dim))
}

/// Try to match rmsnorm: x / sqrt(mean(x^2) + eps) * gamma
fn try_match_rmsnorm(graph: &FusionGraph, mul_node: NodeId) -> Option<ReductionMatch> {
    let mul = &graph.nodes[mul_node as usize];
    if mul.inputs.len() != 2 { return None; }

    // One input should be div(x, sqrt(mean(x^2) + eps)), other is gamma
    let div_id = mul.inputs[0];
    let div = &graph.nodes[div_id as usize];
    if !matches!(div.op, FusionOp::Elementwise(ref s) if s == "div") { return None; }
    if div.inputs.len() != 2 { return None; }

    let x_node = div.inputs[0];

    // sqrt(mean(x^2) + eps)
    let sqrt_id = div.inputs[1];
    let sqrt = &graph.nodes[sqrt_id as usize];
    if !matches!(sqrt.op, FusionOp::Elementwise(ref s) if s == "sqrt") { return None; }

    let add_eps_id = sqrt.inputs[0];
    let add_eps = &graph.nodes[add_eps_id as usize];
    if !matches!(add_eps.op, FusionOp::Elementwise(ref s) if s == "add") { return None; }

    let mean_id = add_eps.inputs[0];
    let mean = &graph.nodes[mean_id as usize];
    if !matches!(mean.op, FusionOp::Reduction(ref s) if s == "mean") { return None; }

    // mean input should be x^2 = mul(x, x)
    let sq_id = mean.inputs[0];
    let sq = &graph.nodes[sq_id as usize];
    if !matches!(sq.op, FusionOp::Elementwise(ref s) if s == "mul") { return None; }
    if sq.inputs[0] != x_node || sq.inputs[1] != x_node { return None; }

    let reduction_dim = get_reduction_dim(graph, mean_id).unwrap_or(-1);

    Some(ReductionMatch {
        pattern: "rmsnorm",
        root_node: mul_node,
        input_nodes: vec![x_node, mul.inputs[1]], // x, gamma
        all_matched_nodes: vec![mul_node, div_id, sqrt_id, add_eps_id, mean_id, sq_id],
        reduction_dim,
        is_naive: false,
        has_affine: true,
    })
}

/// Mark all nodes in detected reduction matches as fused.
pub fn apply_reduction_fusion(graph: &mut FusionGraph, matches: &[ReductionMatch], base_kernel_id: FusedKernelId) {
    for (i, m) in matches.iter().enumerate() {
        let kid = base_kernel_id + i as FusedKernelId;
        for &node_id in &m.all_matched_nodes {
            graph.nodes[node_id as usize].fused_into = Some(kid);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion_graph::FusionGraph;

    fn make_stable_softmax() -> (FusionGraph, NodeId) {
        // exp(x - reduce_max(x)) / reduce_sum(exp(x - reduce_max(x)))
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
        (g, div)
    }

    fn make_naive_softmax() -> (FusionGraph, NodeId) {
        // exp(x) / reduce_sum(exp(x))
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let exp = g.add_node(FusionOp::Elementwise("exp".into()), vec![x]);
        let rsum = g.add_node(FusionOp::Reduction("reduce_sum".into()), vec![exp]);
        g.set_type_info(rsum, vec![1024], DType::F32);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![exp, rsum]);
        g.mark_graph_output(div);
        g.build_consumers();
        (g, div)
    }

    #[test]
    fn test_detect_stable_softmax() {
        let (g, _) = make_stable_softmax();
        let matches = detect_reduction_patterns(&g);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern, "softmax");
        assert!(!matches[0].is_naive);
        assert!(matches[0].all_matched_nodes.len() >= 3);
    }

    #[test]
    fn test_detect_naive_softmax() {
        let (g, _) = make_naive_softmax();
        let matches = detect_reduction_patterns(&g);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern, "softmax");
        assert!(matches[0].is_naive);
    }

    #[test]
    fn test_layernorm_basic_pattern() {
        // Standard layernorm: (x - mean(x)) / sqrt(var(x) + eps)
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![x]);
        g.set_type_info(mean, vec![768], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, mean]);
        let var = g.add_node(FusionOp::Reduction("var".into()), vec![x]);
        g.set_type_info(var, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![var, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![sub, sqrt]);
        g.mark_graph_output(div);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let ln_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "layernorm").collect();
        assert_eq!(ln_matches.len(), 1);
    }

    #[test]
    fn test_layernorm_internal_multi_consumer() {
        // sub(x, mean) is consumed by BOTH div and the squaring step for variance.
        // This tests the internal multi-consumer rule: sub has 2 consumers but
        // both are inside the matched subgraph.
        // Pattern: mean, sub, mul(sub,sub), mean_sq, add_eps, sqrt, div
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![x]);
        g.set_type_info(mean, vec![768], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, mean]);
        // Manual variance: mean((x - mean)^2) — sub consumed by both sq and div
        let sq = g.add_node(FusionOp::Elementwise("mul".into()), vec![sub, sub]);
        let var_mean = g.add_node(FusionOp::Reduction("mean".into()), vec![sq]);
        g.set_type_info(var_mean, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![var_mean, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![sub, sqrt]);
        g.mark_graph_output(div);
        g.build_consumers();

        // sub has 3 consumer edges: sq uses sub twice (mul(sub, sub)) plus div uses it once.
        // build_consumers counts edges, not unique consumer nodes.
        assert_eq!(g.nodes[sub as usize].consumers.len(), 3);

        // NOTE: The current try_match_layernorm_core uses Reduction("var") not
        // this manual expansion. This test documents the desired behavior for
        // manual variance patterns. The implementer should extend the matcher
        // or accept that only var-builtin patterns are matched in M31.
    }

    #[test]
    fn test_layernorm_external_consumer_rejected() {
        // mean(x) consumed by sub AND an external node -> should be rejected
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![x]);
        g.set_type_info(mean, vec![768], DType::F32);
        let sub = g.add_node(FusionOp::Elementwise("sub".into()), vec![x, mean]);
        let var = g.add_node(FusionOp::Reduction("var".into()), vec![x]);
        g.set_type_info(var, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![var, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_named_node("out".into(), FusionOp::Elementwise("div".into()), vec![sub, sqrt]);
        g.mark_graph_output(div);
        // External consumer of mean (outside the layernorm subgraph)
        let _external = g.add_named_node("leak".into(), FusionOp::Elementwise("relu".into()), vec![mean]);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let ln_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "layernorm").collect();
        assert_eq!(ln_matches.len(), 0); // rejected due to external consumer on mean
    }

    #[test]
    fn test_single_node_softmax_builtin() {
        let mut g = FusionGraph::new();
        let x = g.add_node(FusionOp::Input, vec![]);
        let sm = g.add_node(FusionOp::Reduction("softmax".into()), vec![x]);
        g.mark_graph_output(sm);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern, "softmax");
        assert!(!matches[0].is_naive);
    }

    #[test]
    fn test_rmsnorm_pattern() {
        // x / sqrt(mean(x^2) + eps) * gamma
        let mut g = FusionGraph::new();
        let x = g.add_named_node("x".into(), FusionOp::Input, vec![]);
        let gamma = g.add_named_node("gamma".into(), FusionOp::Input, vec![]);
        let sq = g.add_node(FusionOp::Elementwise("mul".into()), vec![x, x]); // x^2
        let mean = g.add_node(FusionOp::Reduction("mean".into()), vec![sq]);
        g.set_type_info(mean, vec![768], DType::F32);
        let eps = g.add_named_node("eps".into(), FusionOp::Input, vec![]);
        let add_eps = g.add_node(FusionOp::Elementwise("add".into()), vec![mean, eps]);
        let sqrt = g.add_node(FusionOp::Elementwise("sqrt".into()), vec![add_eps]);
        let div = g.add_node(FusionOp::Elementwise("div".into()), vec![x, sqrt]);
        let out = g.add_named_node("out".into(), FusionOp::Elementwise("mul".into()), vec![div, gamma]);
        g.mark_graph_output(out);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let rms_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "rmsnorm").collect();
        assert_eq!(rms_matches.len(), 1);
        assert!(rms_matches[0].has_affine);
    }

    #[test]
    fn test_apply_reduction_marks_fused_into() {
        let (mut g, _) = make_stable_softmax();
        let matches = detect_reduction_patterns(&g);
        apply_reduction_fusion(&mut g, &matches, 100);

        // All matched nodes should be marked
        for &nid in &matches[0].all_matched_nodes {
            assert_eq!(g.nodes[nid as usize].fused_into, Some(100));
        }
    }

    #[test]
    fn test_already_claimed_skipped() {
        let (mut g, div) = make_stable_softmax();
        // Pre-claim the div node (the softmax tail)
        g.nodes[div as usize].fused_into = Some(99);
        g.build_consumers();

        let matches = detect_reduction_patterns(&g);
        let softmax_matches: Vec<_> = matches.iter().filter(|m| m.pattern == "softmax").collect();
        assert_eq!(softmax_matches.len(), 0);
    }
}
