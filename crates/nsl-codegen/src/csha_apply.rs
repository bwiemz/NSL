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
    /// A.1: per-layer `CshaExtras` — the runtime-marshalled form that
    /// the FA FFI dispatch path consumes. Parallel to `configs` but
    /// carries the full extras struct (non-serializable `f32` fields)
    /// rather than the serializable payload.
    #[serde(skip)]
    pub extras: std::collections::BTreeMap<String, CshaExtras>,
}

impl BridgeResult {
    /// A.1 convenience: pick a representative `CshaExtras` when the
    /// caller cannot identify which layer a particular FA launch
    /// belongs to. Returns the extras for the first layer in sorted
    /// order, or `None` if no CSHA-active layers were planned.
    ///
    /// Superseded in A.2.0 by [`Self::extras_for_layer`] /
    /// [`Self::extras_at_index`]; kept for back-compat with tests.
    pub fn first_extras(&self) -> Option<&CshaExtras> {
        self.extras.values().next()
    }

    /// A.2.0: look up per-layer CSHA extras by name. Layer names come
    /// from the Wengert `Param` prefix scan (e.g. `blocks.0`,
    /// `layers.7`). Returns `None` when the layer isn't in the plan or
    /// CSHA didn't fuse it.
    pub fn extras_for_layer(&self, layer: &str) -> Option<&CshaExtras> {
        self.extras.get(layer)
    }

    /// A.2.0: positional lookup — returns the extras for the Nth layer
    /// in sorted-key order. Used by the FA call-site as a stopgap when
    /// a proper layer name isn't derivable from context, under the
    /// assumption that FA calls and planner layers appear in matching
    /// source order. `None` when `index` is out of range.
    ///
    /// A.2.1 will replace callers of this with
    /// [`Self::extras_for_layer`] once the Wengert `Param` → enclosing
    /// function-name mapping is wired through `compile_flash_attention_call`.
    pub fn extras_at_index(&self, index: usize) -> Option<&CshaExtras> {
        self.extras.values().nth(index)
    }

    /// Number of CSHA-active layers. `0` when the plan produced no
    /// fused kernels (CSHA off / no Wengert matches / infeasible SMEM).
    pub fn active_layer_count(&self) -> usize {
        self.extras.len()
    }

    /// A.2.1a: positional layer-name lookup — returns the layer name for
    /// the Nth entry in sorted-key order. Parallel to
    /// [`Self::extras_at_index`] so the FA call site can reach both the
    /// extras and the layer name from a single ordinal, in order to
    /// query `marks_for_layer` for weight-param resolution.
    pub fn layer_at_index(&self, index: usize) -> Option<&str> {
        self.extras.keys().nth(index).map(|s| s.as_str())
    }

    /// A.2.1a: iterate the subset of `marks` belonging to a given layer.
    /// Used by the FA call site to resolve `param_name` strings
    /// (e.g. `"TransformerBlock.wq"`) against `FuncState.weights_by_name`
    /// and thread the resulting Cranelift Values into
    /// `nsl_flash_attention_csha`.
    pub fn marks_for_layer<'a>(
        &'a self,
        layer: &'a str,
    ) -> impl Iterator<Item = &'a FusionMark> + 'a {
        self.marks.iter().filter(move |m| m.layer == layer)
    }

    /// A.2.1e: candidate norm-weight parameter names for a CSHA layer.
    ///
    /// RMSNorm's gamma isn't carried as an explicit Wengert input — the
    /// runtime RMSNorm op loads it from the enclosing module field. To
    /// resolve the Cranelift `Value` at the FA call site we probe
    /// `FuncState.weights_by_name` against a set of conventional keys
    /// built from the bridge's layer prefix and well-known field names.
    ///
    /// Returns the candidates in priority order. The caller picks the
    /// first one that hits `weights_by_name`; all others are silently
    /// skipped. Keeping the list explicit (rather than walking the
    /// module AST) keeps A.2.1e self-contained and plan-pure; if a
    /// real model names its norm weight something exotic, the FA
    /// FFI's null-check fallback still fires and no harm is done.
    ///
    /// Conventions covered (priority order):
    ///   1. LLaMA / Mistral style — `<layer>.attn_norm.weight`
    ///   2. Generic — `<layer>.norm.weight`
    ///   3. HuggingFace transformers — `<layer>.input_layernorm.weight`
    ///   4. GPT-2 style — `<layer>.ln_1.weight`
    ///   5. Alt naming — `<layer>.attention_norm.weight`
    ///
    /// All five are also tried with `.gamma` in place of `.weight` for
    /// frameworks that expose the RMSNorm scale under its canonical name.
    pub fn norm_weight_candidates(layer: &str) -> Vec<String> {
        let fields = [
            "attn_norm",
            "norm",
            "input_layernorm",
            "ln_1",
            "attention_norm",
        ];
        let suffixes = ["weight", "gamma"];
        let mut out = Vec::with_capacity(fields.len() * suffixes.len());
        for f in fields {
            for s in suffixes {
                out.push(format!("{}.{}.{}", layer, f, s));
            }
        }
        out
    }

    /// A.2.1b: name-based extras resolver — given the Cranelift
    /// function name currently being compiled (typically mangled as
    /// `"ModelName__method_name"`, or for loop-unrolled per-block
    /// methods `"ModelName__method__blocks_0"`), try to derive a
    /// layer key that matches one of the bridge's per-layer entries.
    ///
    /// Heuristic: scan the function name for each `extras` key as a
    /// substring after normalising dots to underscores (bridge uses
    /// `"blocks.0"`, mangled names use `"blocks_0"`). Return the first
    /// match. Callers should fall back to `extras_at_index` when this
    /// returns `None`.
    ///
    /// This is an incremental step away from pure positional matching:
    /// in multi-layer transformers where FA call order doesn't align
    /// with plan layer order (e.g. iterative `for block in self.blocks`
    /// after loop-unrolling), a name-based match picks the right
    /// layer's extras even if the ordinal would not.
    pub fn extras_for_current_function(
        &self,
        fn_name: &str,
    ) -> Option<(&str, &CshaExtras)> {
        let needle = fn_name.replace('.', "_");
        for (key, extras) in &self.extras {
            let normalised = key.replace('.', "_");
            if needle.contains(&normalised) {
                return Some((key.as_str(), extras));
            }
        }
        None
    }
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

/// A.2.1d: collect the Wengert op indices claimed by every boundary
/// chain in `plan` — RMSNorm prologue, Q/K/V projection matmul, and
/// (optional) RoPE epilogue. Returned set is what
/// `Compiler.csha_claimed_ops` is populated from at the same stmt.rs
/// hook that sets `last_csha_bridge`.
///
/// Factored as a free function so the population logic is pure and
/// directly unit-testable without constructing a full `Compiler`.
pub fn collect_claimed_ops(plan: &CshaPlan) -> std::collections::HashSet<u32> {
    let mut out = std::collections::HashSet::new();
    for chain in &plan.boundary.chains {
        out.insert(chain.norm_op);
        out.insert(chain.matmul_op);
        if let Some(rope_op) = chain.rope_op {
            out.insert(rope_op);
        }
    }
    out
}

/// A.2.1f: resolve each CSHA-active layer's `x` input VarId — the
/// Wengert VarId feeding the RMSNorm op at the start of the boundary
/// chain. Pure plan-level helper: consumes the `CshaPlan`'s
/// `BoundaryChain.norm_op` indices and looks up each op's first input
/// in the supplied Wengert list.
///
/// Returned map is keyed by layer name (matching `BridgeResult.extras`
/// / `BridgeResult.marks_for_layer`). When multiple chains for the
/// same layer disagree on the norm input (shouldn't happen — Q/K/V
/// typically share a single RMSNorm), the first-seen VarId wins.
///
/// **Why this helper doesn't thread the Cranelift `Value` itself:**
/// Cranelift Values are `FunctionBuilder`-scoped. The Wengert VarId →
/// Value map is built inside the train-block function body's lowering
/// (`wengert_lower::compile_wengert_ops`), while the FA call site
/// fires inside the separately-compiled `@flash_attention` function
/// body. Using Values across FunctionBuilder boundaries is unsound.
/// Full x_ptr threading at the FA call site needs either (a) to
/// happen inside the train-block function's lowering when FA ops are
/// decomposed, or (b) a tensor-pointer-sharing mechanism that
/// survives across function boundaries (e.g. a name-keyed global or
/// per-call stash populated before the `@flash_attention` function is
/// invoked). Both are architectural changes beyond the scope of this
/// helper, which stays plan-pure so the infrastructure is ready when
/// that design lands.
pub fn collect_norm_input_varids(
    plan: &CshaPlan,
    wengert: &crate::wengert::WengertList,
) -> std::collections::HashMap<String, u32> {
    let mut out = std::collections::HashMap::new();
    for chain in &plan.boundary.chains {
        let Some(layer) = chain.layer.as_ref() else {
            continue;
        };
        let Some(norm_op) = wengert.ops.get(chain.norm_op as usize) else {
            continue;
        };
        let Some(&x_varid) = norm_op.inputs.first() else {
            continue;
        };
        out.entry(layer.clone()).or_insert(x_varid);
    }
    out
}

/// Build the downstream artefacts from a CSHA plan.
///
/// `seen` and `diagnostics` are threaded through to `flash_attention_selector`
/// so that fallback warnings are dedup'd across the whole compilation unit and
/// emitted by the caller via `eprintln!`.  Pass `&mut FallbackSeen::new()` and
/// `&mut Vec::new()` from free-function callers that have no compiler context.
pub fn bridge(
    plan: &CshaPlan,
    shape_head_dim: i64,
    seen: &mut crate::flash_attention_selector::FallbackSeen,
    diagnostics: &mut Vec<String>,
) -> BridgeResult {
    let mut out = BridgeResult::default();
    if plan.per_layer.is_empty() {
        return out;
    }

    for layer_plan in &plan.per_layer {
        let pat = plan.patterns.get(&layer_plan.layer);
        let cfg = build_flash_config(layer_plan, &plan.specialization, pat, shape_head_dim);
        let kernel_name = crate::flash_attention_selector::flash_attention_kernel_name_selected_with_diag(
            &cfg, seen, diagnostics,
        );
        let smem = crate::flash_attention_selector::shared_mem_bytes_selected_with_diag(
            &cfg, seen, diagnostics,
        );

        out.kernels.push(KernelSpec {
            layer: layer_plan.layer.clone(),
            level: layer_plan.level,
            kernel_name,
            smem_bytes: smem,
        });
        if let Some(ref extras) = cfg.csha {
            out.extras
                .insert(layer_plan.layer.clone(), extras.clone());
        }
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
    pattern: Option<&crate::csha_patterns::PatternDecision>,
    shape_head_dim: i64,
) -> FlashAttentionConfig {
    let head_dim = shape_head_dim.max(layer.tiles.head_dim as i64);
    let active_heads = spec
        .get(&layer.layer)
        .map(|s| s.n_active_heads)
        .unwrap_or(0);
    // Causal defaults to true (autoregressive LLMs); the pattern pass
    // flips it to false for bidirectional models.
    let causal = pattern
        .map(|p| !matches!(p.causal_mask, crate::csha_patterns::CausalMaskStrategy::None))
        .unwrap_or(true);
    let gqa_group_size: u32 = match pattern.map(|p| p.gqa) {
        Some(crate::csha_patterns::GqaStrategy::ZeroCopyStride { group_size }) => group_size,
        _ => 1,
    };

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
        causal,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size,
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

    /// A.2.1c: single-class variant of `attn_wengert` — params use
    /// `Model.member` naming rather than `blocks.N.attn.member`. Before
    /// the A.2.1c fallback, `chain.layer` was None for these names and
    /// no `FusionMark`s were produced.
    fn flat_attn_wengert() -> WengertList {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            op(2, 2, PrimalOp::Param("TransformerBlock.wq".into()), vec![]),
            op(3, 3, PrimalOp::Matmul, vec![1, 2]),
            op(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            op(5, 5, PrimalOp::Param("TransformerBlock.wk".into()), vec![]),
            op(6, 6, PrimalOp::Matmul, vec![1, 5]),
            op(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            op(8, 8, PrimalOp::Param("TransformerBlock.wv".into()), vec![]),
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
            pattern_cfg: crate::csha_patterns::PatternConfig::default(),
        })
    }

    #[test]
    fn bridge_auto_produces_one_kernel_per_layer() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert_eq!(r.kernels.len(), 1);
        assert_eq!(r.kernels[0].layer, "blocks.0");
        assert_ne!(r.kernels[0].level, FusionLevel::None);
    }

    #[test]
    fn a1_bridge_populates_extras_per_csha_active_layer() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        // One kernel, one extras entry — keyed by layer name.
        assert_eq!(r.extras.len(), 1);
        let extras = r
            .extras
            .get("blocks.0")
            .expect("expected extras for blocks.0");
        assert!(extras.level >= 1, "CSHA level must be set for auto mode");
        assert!(
            extras.fused_rmsnorm,
            "boundary fusion implies fused RMSNorm prologue"
        );
    }

    #[test]
    fn a1_bridge_extras_empty_when_csha_off() {
        let plan = toy_plan(CshaMode::Off);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert!(
            r.extras.is_empty(),
            "Off mode must not populate the extras side-table"
        );
    }

    #[test]
    fn a1_first_extras_returns_some_when_plan_active() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert!(r.first_extras().is_some());
    }

    #[test]
    fn a1_first_extras_returns_none_when_plan_off() {
        let plan = toy_plan(CshaMode::Off);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert!(r.first_extras().is_none());
    }

    // ── A.2.0: per-layer extras lookup ────────────────────────────

    #[test]
    fn a20_extras_for_layer_returns_extras_for_matching_name() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        let extras = r
            .extras_for_layer("blocks.0")
            .expect("blocks.0 should be present in the single-layer plan");
        assert!(extras.level >= 1);
    }

    #[test]
    fn a20_extras_for_layer_returns_none_for_unknown_layer() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert!(r.extras_for_layer("blocks.9999").is_none());
    }

    #[test]
    fn a20_extras_at_index_positional_lookup() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert!(r.extras_at_index(0).is_some());
        assert!(r.extras_at_index(1).is_none(), "toy plan has one layer");
    }

    #[test]
    fn a20_active_layer_count_matches_extras_len() {
        let plan_on = toy_plan(CshaMode::Auto);
        let r_on = bridge(&plan_on, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert_eq!(r_on.active_layer_count(), r_on.extras.len());
        assert!(r_on.active_layer_count() >= 1);

        let plan_off = toy_plan(CshaMode::Off);
        let r_off = bridge(&plan_off, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert_eq!(r_off.active_layer_count(), 0);
    }

    #[test]
    fn a21a_layer_at_index_matches_extras_at_index() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        // In-order traversal of `extras` yields the same layer name
        // that `layer_at_index` returns for each ordinal — the FA
        // call site relies on this parallel structure to reach both
        // `CshaExtras` and the layer name from a single ordinal.
        let layer = r
            .layer_at_index(0)
            .expect("toy plan must have at least one CSHA-active layer");
        assert_eq!(layer, "blocks.0");
        assert!(r.extras_at_index(0).is_some());
        assert!(r.layer_at_index(1).is_none());
    }

    #[test]
    fn a21d_collect_claimed_ops_covers_norm_matmul_rope_per_chain() {
        // `attn_wengert` has three boundary chains (Q, K, V). Each
        // chain contributes (norm_op, matmul_op) always and rope_op
        // for Q and K. V has no RoPE.
        //
        // Concretely in the toy Wengert list:
        //   op 1 — RMSNorm (shared across chains, so three chains
        //          all have norm_op = 1)
        //   op 2 — Param blocks.0.attn.wq
        //   op 3 — Matmul (Q)
        //   op 4 — RoPE (Q)
        //   op 5 — Param blocks.0.attn.wk
        //   op 6 — Matmul (K)
        //   op 7 — RoPE (K)
        //   op 8 — Param blocks.0.attn.wv
        //   op 9 — Matmul (V)    ← no RoPE consumer
        //
        // Claimed set must contain {1, 3, 4, 6, 7, 9} — three matmuls,
        // two RoPEs, one shared RMSNorm; exactly six entries.
        let plan = toy_plan(CshaMode::Auto);
        let claimed = collect_claimed_ops(&plan);
        assert_eq!(claimed.len(), 6, "got {:?}", claimed);
        assert!(claimed.contains(&1), "RMSNorm op missing");
        assert!(claimed.contains(&3), "Q matmul missing");
        assert!(claimed.contains(&4), "Q RoPE missing");
        assert!(claimed.contains(&6), "K matmul missing");
        assert!(claimed.contains(&7), "K RoPE missing");
        assert!(claimed.contains(&9), "V matmul missing");
        // V has no rope — op indices 2/5/8 are the Param leaves (not
        // claimed as fusion targets; the bridge tracks them via
        // `chain.weight_param` instead).
        assert!(!claimed.contains(&2));
        assert!(!claimed.contains(&5));
        assert!(!claimed.contains(&8));
    }

    #[test]
    fn a21d_collect_claimed_ops_empty_when_csha_off() {
        let plan = toy_plan(CshaMode::Off);
        let claimed = collect_claimed_ops(&plan);
        assert!(
            claimed.is_empty(),
            "Off mode must produce no claimed ops; got {:?}",
            claimed
        );
    }

    #[test]
    fn a21c_flat_model_params_produce_marks_via_last_dot_fallback() {
        // Single-class model: params named `"TransformerBlock.wq"` lack
        // a canonical `blocks.N` prefix. Pre-A.2.1c, `chain.layer` was
        // None for these and `bridge()` emitted zero `FusionMark`s —
        // meaning the FA call site's `marks_for_layer` iterator was
        // empty and Wq/Wk/Wv/Wo threading always fell back to null.
        //
        // After A.2.1c, `layer_key_with_fallback` strips `.wq`/`.wk`/etc.
        // to give `"TransformerBlock"` as the layer key, so marks are
        // emitted and wq/wk/wv/wo threading actually resolves.
        let w = flat_attn_wengert();
        let plan = run(CshaInput {
            mode: CshaMode::Auto,
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
            pattern_cfg: crate::csha_patterns::PatternConfig::default(),
        });
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());

        // At least one mark must be emitted, and all of them must
        // carry the class-level fallback layer key.
        assert!(
            !r.marks.is_empty(),
            "flat-model chains must still emit marks via last-dot fallback"
        );
        for m in &r.marks {
            assert_eq!(m.layer, "TransformerBlock");
        }

        // The extras map must also key on the fallback layer — the FA
        // call site's name-based resolver (A.2.1b) relies on this
        // consistency to look up extras by the enclosing function name.
        assert!(r.extras.contains_key("TransformerBlock"));
    }

    #[test]
    fn a21f_collect_norm_input_varids_extracts_rmsnorm_input_per_layer() {
        // Toy three-chain plan (Q/K/V) shares a single RMSNorm whose
        // input is VarId 0 (the `Input("x")` op at index 0). All three
        // chains point at norm_op=1 (the RMSNorm). The helper returns
        // `layer → x_varid`, first-seen wins when chains for the same
        // layer agree.
        let w = attn_wengert();
        let plan = toy_plan(CshaMode::Auto);
        let norm_inputs = collect_norm_input_varids(&plan, &w);
        // Toy plan yields one CSHA-active layer `blocks.0` with three
        // chains, all referencing the same RMSNorm whose input is 0.
        assert_eq!(norm_inputs.len(), 1);
        assert_eq!(norm_inputs.get("blocks.0").copied(), Some(0));
    }

    #[test]
    fn a21f_collect_norm_input_varids_empty_when_csha_off() {
        // Off-mode short-circuits in `run` → no per_layer entries and
        // no boundary chains. Helper returns an empty map, matching
        // the null-fallback contract at the FA call site.
        let w = attn_wengert();
        let plan = toy_plan(CshaMode::Off);
        let norm_inputs = collect_norm_input_varids(&plan, &w);
        assert!(norm_inputs.is_empty());
    }

    #[test]
    fn a21f_collect_norm_input_varids_skips_chains_without_layer() {
        // `BoundaryChain.layer = None` is the "floating chain"
        // case (`layer_key_with_fallback` returned None, typically
        // for top-level parameters). These chains still exist in the
        // scan but don't contribute to per-layer x-input resolution —
        // the FA call site keys on layer name, so unlayered chains
        // are inaccessible anyway.
        use crate::csha_boundary::{BoundaryChain, BoundaryScan, ProjKind};
        let mut plan = toy_plan(CshaMode::Off);
        plan.boundary = BoundaryScan {
            chains: vec![BoundaryChain {
                layer: None,
                kind: ProjKind::Q,
                norm_op: 1,
                matmul_op: 3,
                rope_op: None,
                weight_param: "unrooted.wq".to_string(),
            }],
        };
        let norm_inputs = collect_norm_input_varids(&plan, &attn_wengert());
        assert!(norm_inputs.is_empty());
    }

    #[test]
    fn a21e_norm_weight_candidates_covers_common_conventions() {
        let cands = BridgeResult::norm_weight_candidates("blocks.0");
        // LLaMA-style (priority 1) must come first.
        assert_eq!(cands[0], "blocks.0.attn_norm.weight");
        // Every convention is tried with both `.weight` and `.gamma`.
        assert!(cands.contains(&"blocks.0.norm.weight".to_string()));
        assert!(cands.contains(&"blocks.0.norm.gamma".to_string()));
        assert!(cands.contains(&"blocks.0.input_layernorm.weight".to_string()));
        assert!(cands.contains(&"blocks.0.ln_1.weight".to_string()));
        assert!(cands.contains(&"blocks.0.attention_norm.weight".to_string()));
        // Exactly 5 field names × 2 suffixes = 10 candidates total.
        assert_eq!(cands.len(), 10);
    }

    #[test]
    fn a21e_norm_weight_candidates_respects_flat_model_layer_fallback() {
        // A.2.1c's last-dot fallback produces layer keys like
        // `TransformerBlock` for single-class models. The conventions
        // helper must compose cleanly with that — no canonical prefix
        // required.
        let cands = BridgeResult::norm_weight_candidates("TransformerBlock");
        assert_eq!(cands[0], "TransformerBlock.attn_norm.weight");
        assert!(cands.contains(&"TransformerBlock.input_layernorm.weight".to_string()));
    }

    #[test]
    fn a21b_extras_for_current_function_matches_mangled_name() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        // Cranelift mangling typically produces `ModelName__method`
        // or `ModelName__method__blocks_0`. With dot→underscore
        // normalisation, "blocks.0" matches "blocks_0" inside the
        // mangled name.
        let (layer, extras) = r
            .extras_for_current_function("TransformerBlock__forward__blocks_0")
            .expect("mangled fn name should resolve to blocks.0");
        assert_eq!(layer, "blocks.0");
        assert!(extras.level >= 1);
    }

    #[test]
    fn a21b_extras_for_current_function_matches_plain_name() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        // An un-mangled debug-build name still matches — the bridge
        // key `blocks.0` is a substring of `"Model.forward.blocks.0"`
        // under the dot-preserving view.
        let hit = r.extras_for_current_function("Model_forward_blocks_0");
        assert!(hit.is_some());
    }

    #[test]
    fn a21b_extras_for_current_function_returns_none_when_no_layer_match() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        // Function name without any `blocks.N` shape — name-based
        // resolver declines; caller falls back to ordinal.
        assert!(r.extras_for_current_function("MLP__forward").is_none());
    }

    #[test]
    fn a21b_extras_for_current_function_none_when_csha_off() {
        let plan = toy_plan(CshaMode::Off);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert!(r
            .extras_for_current_function("anything__blocks_0")
            .is_none());
    }

    #[test]
    fn a21a_marks_for_layer_filters_by_layer_name() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        // Every mark yielded by `marks_for_layer("blocks.0")` must
        // carry that same layer name — the FA call site uses this
        // filter to scope `weights_by_name` lookups to the current
        // layer's Wq/Wk/Wv/Wo params.
        let count = r.marks_for_layer("blocks.0").count();
        assert!(count >= 1, "toy plan must produce at least one mark");
        for m in r.marks_for_layer("blocks.0") {
            assert_eq!(m.layer, "blocks.0");
        }
        // Unknown layer yields no marks.
        assert_eq!(r.marks_for_layer("blocks.9999").count(), 0);
    }

    #[test]
    fn bridge_emits_qkv_norm_and_rope_marks() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
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
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        assert!(r.kernels.is_empty());
        assert!(r.marks.is_empty());
        assert!(r.configs.is_empty());
    }

    #[test]
    fn bridge_forced_boundary_emits_no_output_proj_mark() {
        let plan = toy_plan(CshaMode::Boundary);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
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
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        for k in &r.kernels {
            assert!(k.kernel_name.contains("cshaL"), "name={}", k.kernel_name);
        }
    }

    #[test]
    fn smem_bytes_are_positive() {
        let plan = toy_plan(CshaMode::Auto);
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
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
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());

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
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
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
        let r = bridge(&plan, 64, &mut crate::flash_attention_selector::FallbackSeen::new(), &mut Vec::new());
        let (_layer, cfg) = r.configs.iter().next().unwrap();
        assert!(cfg.csha_level >= 1 && cfg.csha_level <= 3);
        assert!(cfg.fused_rmsnorm);
    }
}
