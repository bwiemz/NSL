//! WGGO — Wengert Graph Global Optimization: layer-level graph representation.
//!
//! Transforms the elementary-operation Wengert list into a coarser
//! *layer graph* that WGGO's hierarchical optimizer operates on.  A layer
//! is a contiguous group of Wengert ops that share a name prefix such as
//! `blocks.N.*`.  The rationale (Section 4.2 of the paper): a 500 000-node
//! Wengert graph is intractable for ILP, but ~30 layers × ~30 per-layer
//! variables is trivial.
//!
//! The output is an [`OptGraph`] consumed by `wggo_dp` (Level 1) and
//! `wggo_ilp` (Level 2).  Layer grouping is deterministic — two
//! invocations on the same Wengert list produce byte-identical graphs.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::wengert::{PrimalOp, VarId, WengertList};

/// Role of a layer in the transformer.  The role drives which decision
/// variables apply: `Attention` layers pay CSHA decisions, `Ffn` layers
/// don't.  `Other` is a catch-all for embeddings, norms, the LM head, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LayerRole {
    Attention,
    Ffn,
    /// Fused attention + FFN (`TransformerBlock` wrapper).
    Block,
    Embedding,
    LmHead,
    Other,
}

impl LayerRole {
    pub fn as_str(self) -> &'static str {
        match self {
            LayerRole::Attention => "attention",
            LayerRole::Ffn => "ffn",
            LayerRole::Block => "block",
            LayerRole::Embedding => "embedding",
            LayerRole::LmHead => "lm_head",
            LayerRole::Other => "other",
        }
    }
}

/// Coarse descriptor for one transformer layer.
///
/// Carries the information Level-1 DP and Level-2 ILP need:
///   * `ops`            — indices into the source Wengert list
///   * `params`         — names of parameters owned by this layer
///   * `role`           — transformer role
///   * `depends_on`     — layer indices whose outputs feed this layer
#[derive(Debug, Clone, Serialize)]
pub struct Layer {
    pub index: u32,
    pub name: String,
    pub role: LayerRole,
    pub op_indices: Vec<u32>,
    pub param_names: Vec<String>,
    pub depends_on: Vec<u32>,
}

/// The full layer graph WGGO optimises over.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OptGraph {
    pub layers: Vec<Layer>,
    /// Total number of Wengert ops grouped into layers (for reporting).
    pub total_ops: u32,
}

impl OptGraph {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, idx: usize) -> Option<&Layer> {
        self.layers.get(idx)
    }

    pub fn ops_grouped(&self) -> u32 {
        self.layers.iter().map(|l| l.op_indices.len() as u32).sum()
    }
}

/// Extract the "layer prefix" from a parameter / op name.
///
/// We recognise three common shapes:
///   * `blocks.N.<rest>`   → prefix is `blocks.N`
///   * `layers.N.<rest>`   → prefix is `layers.N`
///   * `h.N.<rest>`        → prefix is `h.N`  (GPT-2-style)
///
/// Everything else is bucketed under a synthetic `other` layer.
pub fn layer_prefix(name: &str) -> Option<String> {
    for pat in ["blocks.", "layers.", "h."] {
        if let Some(rest) = name.strip_prefix(pat) {
            // Everything up to (and including) the next dot.
            if let Some(end) = rest.find('.') {
                return Some(format!("{}{}", pat, &rest[..end]));
            }
        }
    }
    None
}

/// Infer a layer's role from its canonical name.
pub fn infer_role(name: &str) -> LayerRole {
    let lname = name.to_ascii_lowercase();
    if lname.contains("embed") || lname == "tok_embeddings" || lname == "wte" {
        LayerRole::Embedding
    } else if lname.contains("lm_head") || lname.contains("output") {
        LayerRole::LmHead
    } else if lname.contains("ffn") || lname.contains("mlp") {
        LayerRole::Ffn
    } else if lname.contains("attn") {
        LayerRole::Attention
    } else if lname.starts_with("blocks.") || lname.starts_with("layers.") || lname.starts_with("h.") {
        LayerRole::Block
    } else {
        LayerRole::Other
    }
}

/// Build the optimisation graph from a Wengert list.
///
/// The pass walks every op; for each `Param(name)` it assigns the
/// producing op to the layer whose prefix matches.  Non-param ops inherit
/// the layer of the last-seen param in their input set; this covers the
/// common case where an op consumes a param (or the output of a previous
/// op in the same layer).  Floating ops (no param ancestor) land in the
/// synthetic `other` layer.
pub fn build(list: &WengertList) -> OptGraph {
    // 1) Partition param ops into layer buckets.
    let mut layer_by_prefix: BTreeMap<String, Layer> = BTreeMap::new();
    let mut op_to_layer: Vec<Option<String>> = vec![None; list.ops.len()];

    for (idx, op) in list.ops.iter().enumerate() {
        if let PrimalOp::Param(name) = &op.op {
            let prefix = layer_prefix(name).unwrap_or_else(|| "other".to_string());
            let entry = layer_by_prefix
                .entry(prefix.clone())
                .or_insert_with(|| Layer {
                    index: 0,
                    name: prefix.clone(),
                    role: infer_role(&prefix),
                    op_indices: Vec::new(),
                    param_names: Vec::new(),
                    depends_on: Vec::new(),
                });
            entry.param_names.push(name.clone());
            entry.op_indices.push(idx as u32);
            op_to_layer[idx] = Some(prefix);
        }
    }

    // 2) Propagate layer assignment to non-param ops: an op inherits the
    //    layer of its earliest-named input ancestor.
    let var_to_op: BTreeMap<VarId, usize> = list
        .ops
        .iter()
        .enumerate()
        .map(|(i, op)| (op.result, i))
        .collect();
    for (idx, op) in list.ops.iter().enumerate() {
        if op_to_layer[idx].is_some() {
            continue;
        }
        let mut chosen: Option<String> = None;
        for &inp in &op.inputs {
            if let Some(&producer_idx) = var_to_op.get(&inp) {
                if let Some(layer) = op_to_layer[producer_idx].as_ref() {
                    chosen = Some(layer.clone());
                    break;
                }
            }
        }
        let layer_name = chosen.unwrap_or_else(|| "other".to_string());
        let entry = layer_by_prefix
            .entry(layer_name.clone())
            .or_insert_with(|| Layer {
                index: 0,
                name: layer_name.clone(),
                role: infer_role(&layer_name),
                op_indices: Vec::new(),
                param_names: Vec::new(),
                depends_on: Vec::new(),
            });
        entry.op_indices.push(idx as u32);
        op_to_layer[idx] = Some(layer_name);
    }

    // 3) Sort layers by first op index to preserve execution order.
    let mut layers: Vec<Layer> = layer_by_prefix.into_values().collect();
    layers.sort_by_key(|l| l.op_indices.first().copied().unwrap_or(u32::MAX));
    for (i, layer) in layers.iter_mut().enumerate() {
        layer.index = i as u32;
    }

    // 4) Dependency inference: layer j depends on layer i iff at least one
    //    op in j consumes a VarId produced in layer i.
    let layer_of: BTreeMap<u32, u32> = layers
        .iter()
        .flat_map(|l| l.op_indices.iter().map(move |&op_idx| (op_idx, l.index)))
        .collect();
    let mut dependencies: Vec<std::collections::BTreeSet<u32>> =
        vec![Default::default(); layers.len()];
    for op in list.ops.iter() {
        let op_idx = var_to_op.get(&op.result).copied().unwrap_or(usize::MAX);
        let Some(&consumer_layer) = layer_of.get(&(op_idx as u32)) else {
            continue;
        };
        for &inp in &op.inputs {
            if let Some(&producer_op) = var_to_op.get(&inp) {
                if let Some(&producer_layer) = layer_of.get(&(producer_op as u32)) {
                    if producer_layer != consumer_layer {
                        dependencies[consumer_layer as usize].insert(producer_layer);
                    }
                }
            }
        }
    }
    for (i, deps) in dependencies.into_iter().enumerate() {
        layers[i].depends_on = deps.into_iter().collect();
    }

    let total_ops = list.ops.len() as u32;
    OptGraph { layers, total_ops }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use std::collections::HashMap;

    fn op(id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: o,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn two_block_mlp() -> WengertList {
        // x (#0) input
        // blocks.0.wq (#1) → matmul (#2)
        // blocks.1.wq (#3) → matmul (#4)
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            op(2, 2, PrimalOp::Matmul, vec![1, 0]),
            op(3, 3, PrimalOp::Param("blocks.1.attn.wq".into()), vec![]),
            op(4, 4, PrimalOp::Matmul, vec![3, 2]),
        ];
        WengertList {
            ops,
            output: 4,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    #[test]
    fn layer_prefix_handles_standard_patterns() {
        assert_eq!(
            layer_prefix("blocks.6.attn.wq"),
            Some("blocks.6".to_string())
        );
        assert_eq!(
            layer_prefix("layers.12.norm"),
            Some("layers.12".to_string())
        );
        assert_eq!(layer_prefix("h.3.mlp.fc"), Some("h.3".to_string()));
        assert_eq!(layer_prefix("embedding.weight"), None);
    }

    #[test]
    fn build_groups_by_block_prefix() {
        let w = two_block_mlp();
        let g = build(&w);
        // Two real layers + possibly "other" bucket for the raw input.
        let real_layer_names: Vec<&str> = g.layers.iter().map(|l| l.name.as_str()).collect();
        assert!(real_layer_names.contains(&"blocks.0"));
        assert!(real_layer_names.contains(&"blocks.1"));
    }

    #[test]
    fn layers_are_indexed_by_execution_order() {
        let w = two_block_mlp();
        let g = build(&w);
        for (i, l) in g.layers.iter().enumerate() {
            assert_eq!(l.index, i as u32);
        }
    }

    #[test]
    fn dependency_inference_links_block_n_to_block_n_minus_1() {
        let w = two_block_mlp();
        let g = build(&w);
        // blocks.1 consumes matmul output of blocks.0 → must depend on it.
        let b1 = g.layers.iter().find(|l| l.name == "blocks.1").unwrap();
        let b0 = g.layers.iter().find(|l| l.name == "blocks.0").unwrap();
        assert!(b1.depends_on.contains(&b0.index));
    }

    #[test]
    fn inferred_role_matches_known_names() {
        assert_eq!(infer_role("blocks.6.attn"), LayerRole::Attention);
        assert_eq!(infer_role("blocks.6.ffn"), LayerRole::Ffn);
        assert_eq!(infer_role("tok_embeddings"), LayerRole::Embedding);
        assert_eq!(infer_role("lm_head"), LayerRole::LmHead);
        assert_eq!(infer_role("unknown"), LayerRole::Other);
    }

    #[test]
    fn empty_list_produces_empty_graph() {
        let empty = WengertList {
            ops: Vec::new(),
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let g = build(&empty);
        assert_eq!(g.num_layers(), 0);
        assert_eq!(g.total_ops, 0);
    }

    #[test]
    fn ops_grouped_equals_total_ops_when_layers_cover_graph() {
        let w = two_block_mlp();
        let g = build(&w);
        // Every op must be assigned to exactly one layer.
        assert_eq!(g.ops_grouped(), g.total_ops);
    }
}
