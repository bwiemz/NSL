//! CSHA plan → kernel-configuration bridge.
//!
//! Translates a [`CshaPlan`] (produced by `csha::run`) into two concrete
//! artefacts the rest of the compiler pipeline can consume:
//!
//!   * A `Vec<FlashAttentionConfig>` — one specialised attention kernel
//!     per layer, carrying the `CshaExtras` that drive PTX emission in
//!     [`crate::flash_attention`].
//!   * A `FusionGraphMarks` set — `(layer, node_id, kind)` tuples that
//!     tell the fusion-graph walker which nodes have been claimed by a
//!     CSHA-fused kernel and must therefore not be re-fused by
//!     `epilogue_fusion` / `reduction_fusion` separately.
//!
//! The module is intentionally pure: it takes immutable inputs and
//! returns data structures.  The actual graph mutation is done by the
//! caller via [`apply_marks_to_graph`].

use serde::Serialize;

use crate::csha::CshaPlan;
use crate::csha_boundary::{BoundaryChain, ProjKind};
use crate::csha_pipeline::{FusionLevel, LayerPlan};
use crate::csha_specialize::LayerSpec;
use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use crate::fusion_graph::{FusionGraph, FusionOp};

/// One flash-attention kernel specialisation produced by the bridge.
#[derive(Debug, Clone, Serialize)]
pub struct KernelSpec {
    /// Layer prefix (e.g. `blocks.3`) this kernel targets.
    pub layer: String,
    /// Fusion level realised by the kernel.
    pub level: FusionLevel,
    /// Symbolic kernel name — unique per specialisation.
    pub kernel_name: String,
    /// Shared-memory requirement in bytes.
    pub smem_bytes: u32,
}

/// Result of [`bridge`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct BridgeResult {
    pub kernels: Vec<KernelSpec>,
    pub marks: Vec<FusionMark>,
    /// Per-layer FlashAttentionConfig that downstream passes can query
    /// to get the right specialisation for a particular layer.
    pub configs: std::collections::BTreeMap<String, FlashAttentionConfigPayload>,
}

/// Serializable mirror of the bits of `FlashAttentionConfig` a downstream
/// consumer cares about.  The original struct isn't `Serialize` because it
/// is mostly a PTX-builder input; this projection makes the bridge output
/// testable and report-able without dragging PTX machinery into the
/// `Serialize` path.
#[derive(Debug, Clone, Serialize)]
pub struct FlashAttentionConfigPayload {
    pub block_q: i64,
    pub block_kv: i64,
    pub head_dim: i64,
    pub gqa_group_size: u32,
    pub csha_level: u8,
    pub fused_rmsnorm: bool,
    pub fused_projections: bool,
    pub fused_output_proj: bool,
    pub active_heads: u32,
}

impl From<&FlashAttentionConfig> for FlashAttentionConfigPayload {
    fn from(c: &FlashAttentionConfig) -> Self {
        let csha = c.csha.as_ref();
        Self {
            block_q: c.block_q,
            block_kv: c.block_kv,
            head_dim: c.head_dim,
            gqa_group_size: c.gqa_group_size,
            csha_level: csha.map(|x| x.level).unwrap_or(0),
            fused_rmsnorm: csha.map(|x| x.fused_rmsnorm).unwrap_or(false),
            fused_projections: csha.map(|x| x.fused_projections).unwrap_or(false),
            fused_output_proj: csha.map(|x| x.fused_output_proj).unwrap_or(false),
            active_heads: csha.map(|x| x.active_heads).unwrap_or(0),
        }
    }
}

/// A `FusionGraph` node claim produced by CSHA.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FusionMark {
    /// Layer prefix.
    pub layer: String,
    /// Which projection the mark belongs to (or `None` for the attention
    /// output kernel itself).
    pub kind: Option<ProjKind>,
    /// Name of the parameter whose consumer should be fused.
    pub param_name: String,
    /// What the mark means semantically.
    pub role: MarkRole,
}

/// Why CSHA is claiming a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum MarkRole {
    /// Norm node folded as prologue of the Q/K/V matmul.
    NormPrologue,
    /// RoPE node folded as epilogue of the Q/K matmul.
    RoPEEpilogue,
    /// Output-projection matmul absorbed into the attention kernel's
    /// epilogue (Level 2+).
    OutputProjEpilogue,
}

// ---------------------------------------------------------------------------
// Bridge
// ---------------------------------------------------------------------------

/// Build the downstream artefacts from a CSHA plan.
pub fn bridge(plan: &CshaPlan, shape_head_dim: i64) -> BridgeResult {
    let mut out = BridgeResult::default();
    if plan.per_layer.is_empty() {
        return out;
    }

    for layer_plan in &plan.per_layer {
        let cfg = build_flash_config(layer_plan, &plan.specialization, shape_head_dim);
        let kernel_name = crate::flash_attention::flash_attention_kernel_name(&cfg);
        let smem = crate::flash_attention::shared_mem_bytes(&cfg);

        out.kernels.push(KernelSpec {
            layer: layer_plan.layer.clone(),
            level: layer_plan.level,
            kernel_name,
            smem_bytes: smem,
        });
        out.configs
            .insert(layer_plan.layer.clone(), (&cfg).into());
    }

    // Emit fusion-graph marks from the per-chain boundary scan so
    // `epilogue_fusion` / `reduction_fusion` know these nodes are
    // already spoken for.
    for chain in &plan.boundary.chains {
        if let Some(ref layer) = chain.layer {
            out.marks.push(FusionMark {
                layer: layer.clone(),
                kind: Some(chain.kind),
                param_name: chain.weight_param.clone(),
                role: MarkRole::NormPrologue,
            });
            if chain.rope_op.is_some() {
                out.marks.push(FusionMark {
                    layer: layer.clone(),
                    kind: Some(chain.kind),
                    param_name: chain.weight_param.clone(),
                    role: MarkRole::RoPEEpilogue,
                });
            }
        }
    }

    // Layer-level output-projection marks (Level 2+).
    for layer_plan in &plan.per_layer {
        if matches!(layer_plan.level, FusionLevel::Pipeline | FusionLevel::Block) {
            out.marks.push(FusionMark {
                layer: layer_plan.layer.clone(),
                kind: None,
                param_name: format!("{}.attn.wo", layer_plan.layer),
                role: MarkRole::OutputProjEpilogue,
            });
        }
    }

    out.marks.sort_by(|a, b| {
        (a.layer.as_str(), a.kind.map(|k| k as u8), a.role as u8)
            .cmp(&(b.layer.as_str(), b.kind.map(|k| k as u8), b.role as u8))
    });
    out
}

fn build_flash_config(
    layer: &LayerPlan,
    spec: &crate::csha_specialize::SpecializationPlan,
    shape_head_dim: i64,
) -> FlashAttentionConfig {
    let head_dim = shape_head_dim.max(layer.tiles.head_dim as i64);
    let active_heads = spec
        .get(&layer.layer)
        .map(|s| s.n_active_heads)
        .unwrap_or(0);

    let csha = match layer.level {
        FusionLevel::None => None,
        FusionLevel::Boundary => {
            let mut e = CshaExtras::level1(1e-5);
            e.active_heads = active_heads;
            Some(e)
        }
        FusionLevel::Pipeline => {
            let mut e = CshaExtras::level2(1e-5, (head_dim as u32) * (active_heads.max(1)));
            e.active_heads = active_heads;
            Some(e)
        }
        FusionLevel::Block => {
            let mut e = CshaExtras::level3(1e-5, (head_dim as u32) * (active_heads.max(1)));
            e.active_heads = active_heads;
            Some(e)
        }
    };

    FlashAttentionConfig {
        block_q: layer.tiles.block_q as i64,
        block_kv: layer.tiles.block_kv as i64,
        head_dim,
        causal: true, // autoregressive is the common case; overridable later
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        csha,
    }
}

/// Apply CSHA fusion marks to a `FusionGraph`, tagging matmul / RMSNorm /
/// RoPE nodes that belong to CSHA-fused kernels so the independent
/// fusion passes skip them.
///
/// Returns the number of nodes successfully marked.  The search is by
/// parameter name — the graph's name_to_node map must contain the
/// weight params for the marks to apply.
pub fn apply_marks_to_graph(graph: &mut FusionGraph, marks: &[FusionMark]) -> usize {
    let mut n_applied = 0;
    for m in marks {
        let Some(&param_node) = graph.name_to_node.get(&m.param_name) else {
            continue;
        };
        // The matmul that consumes this param is the Q/K/V projection
        // we want to claim.  Walk the param's consumers and find the
        // first Matmul.
        let consumers: Vec<u32> = graph.nodes[param_node as usize].consumers.clone();
        let mm = consumers
            .into_iter()
            .find(|&c| matches!(graph.nodes[c as usize].op, FusionOp::Matmul));
        let Some(mm_id) = mm else { continue };
        if graph.nodes[mm_id as usize].fused_into.is_none() {
            // Reserve kernel-id 0xFF_00_00_00 | role as a CSHA marker.
            // This avoids collisions with real `FusedKernelId`s (which
            // grow from 0) and makes it trivial to detect in
            // downstream passes.
            let csha_tag: u32 = 0xFF00_0000 | m.role as u32;
            graph.nodes[mm_id as usize].fused_into = Some(csha_tag);
            n_applied += 1;
        }
    }
    n_applied
}

/// Returns `true` if a given `fused_into` value was produced by
/// `apply_marks_to_graph` (i.e. high byte = 0xFF).
pub fn is_csha_fused(kid: u32) -> bool {
    (kid & 0xFF00_0000) == 0xFF00_0000
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csha::{run, CshaInput, CshaMode};
    use crate::csha_specialize::SpecConfig;
    use crate::wengert::{PrimalOp, WengertList, WengertOp};
    use crate::wggo_cost::LayerShape;
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

    fn attn_wengert() -> WengertList {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            op(2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            op(3, 3, PrimalOp::Matmul, vec![1, 2]),
            op(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            op(5, 5, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
            op(6, 6, PrimalOp::Matmul, vec![1, 5]),
            op(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            op(8, 8, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
            op(9, 9, PrimalOp::Matmul, vec![1, 8]),
        ];
        WengertList {
            ops,
            output: 9,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    fn toy_plan(mode: CshaMode) -> CshaPlan {
        let w = attn_wengert();
        run(CshaInput {
            mode,
            target: "H100",
            wengert: &w,
            weights: None,
            shape: LayerShape {
                batch: 1,
                seq: 1024,
                d_model: 512,
                head_dim: 64,
                n_kv_heads: 4,
                dtype_bytes: 2,
            },
            n_heads: 8,
            spec_cfg: SpecConfig::default(),
            wggo_overrides: None,
        })
    }

    #[test]
    fn bridge_auto_produces_one_kernel_per_layer() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64);
        assert_eq!(r.kernels.len(), 1);
        assert_eq!(r.kernels[0].layer, "blocks.0");
        assert_ne!(r.kernels[0].level, FusionLevel::None);
    }

    #[test]
    fn bridge_emits_qkv_norm_and_rope_marks() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64);
        // 3 chains × (norm + rope) = 6, except V has no rope so 5 marks
        // from boundary chains.  Plus per-layer output-proj mark
        // (Pipeline or Block).
        let norm_marks = r
            .marks
            .iter()
            .filter(|m| m.role == MarkRole::NormPrologue)
            .count();
        let rope_marks = r
            .marks
            .iter()
            .filter(|m| m.role == MarkRole::RoPEEpilogue)
            .count();
        assert_eq!(norm_marks, 3);
        assert_eq!(rope_marks, 2);
    }

    #[test]
    fn bridge_off_mode_produces_nothing() {
        let plan = toy_plan(CshaMode::Off);
        let r = bridge(&plan, 64);
        assert!(r.kernels.is_empty());
        assert!(r.marks.is_empty());
        assert!(r.configs.is_empty());
    }

    #[test]
    fn bridge_forced_boundary_emits_no_output_proj_mark() {
        let plan = toy_plan(CshaMode::Boundary);
        let r = bridge(&plan, 64);
        let n_output = r
            .marks
            .iter()
            .filter(|m| m.role == MarkRole::OutputProjEpilogue)
            .count();
        assert_eq!(n_output, 0);
    }

    #[test]
    fn kernel_names_encode_csha_level() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64);
        for k in &r.kernels {
            assert!(k.kernel_name.contains("cshaL"), "name={}", k.kernel_name);
        }
    }

    #[test]
    fn smem_bytes_are_positive() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64);
        for k in &r.kernels {
            assert!(k.smem_bytes > 0);
        }
    }

    #[test]
    fn csha_fused_tag_is_distinguishable() {
        assert!(is_csha_fused(0xFF00_0000));
        assert!(is_csha_fused(0xFF00_0002));
        assert!(!is_csha_fused(0));
        assert!(!is_csha_fused(1));
        assert!(!is_csha_fused(0x7FFF_FFFF));
    }

    #[test]
    fn apply_marks_tags_matmul_consumers() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64);

        // Minimal fusion graph: one param node consumed by one matmul.
        let mut g = FusionGraph::new();
        let p = g.add_named_node(
            "blocks.0.attn.wq".into(),
            FusionOp::Input,
            vec![],
        );
        let mm = g.add_node(FusionOp::Matmul, vec![p]);
        g.nodes[p as usize].consumers.push(mm);

        let n = apply_marks_to_graph(&mut g, &r.marks);
        assert!(n >= 1);
        let fused = g.nodes[mm as usize].fused_into.unwrap();
        assert!(is_csha_fused(fused));
    }

    #[test]
    fn apply_marks_is_idempotent() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64);
        let mut g = FusionGraph::new();
        let p = g.add_named_node("blocks.0.attn.wq".into(), FusionOp::Input, vec![]);
        let mm = g.add_node(FusionOp::Matmul, vec![p]);
        g.nodes[p as usize].consumers.push(mm);

        let n1 = apply_marks_to_graph(&mut g, &r.marks);
        let n2 = apply_marks_to_graph(&mut g, &r.marks);
        // Second pass shouldn't re-mark already-fused nodes.
        assert!(n1 >= 1);
        assert_eq!(n2, 0);
    }

    #[test]
    fn config_payload_round_trips_csha_level() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64);
        let (_layer, cfg) = r.configs.iter().next().unwrap();
        assert!(cfg.csha_level >= 1 && cfg.csha_level <= 3);
        assert!(cfg.fused_rmsnorm);
    }
}
