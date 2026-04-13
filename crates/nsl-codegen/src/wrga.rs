//! WRGA compiler pass entry point: Wengert-Pruned Roofline-Guided Adaptation.
//!
//! Composes the five innovations in `wrga_prune` + `wrga_roofline` +
//! `wrga_spectral` + `wrga_fusion` + `wrga_memory` into a single driver that
//! produces a [`WrgaPlan`] from a model's Wengert list, weight map, and
//! hardware target.
//!
//! The driver is intentionally *pure*: it consumes immutable inputs and
//! returns a plan that the rest of the codegen pipeline applies.  The
//! `nsl build --wrga-report` subcommand serializes the plan to a
//! human-readable string via [`WrgaPlan::render_report`].

use std::collections::{BTreeMap, BTreeSet, HashMap};

use nsl_ast::block::WrgaMode;

use crate::gpu_specs::{default_gpu, find_gpu};
use crate::wengert::{PrimalOp, VarId, WengertList};
use crate::weight_aware::WeightMap;
use crate::wrga_fusion::{build_fusion_plan, FusionPlan};
use crate::wrga_memory::{plan_memory, MemoryPlan, SizeHints};
use crate::wrga_prune::{prune, PruneResult};
use crate::wrga_roofline::{place_adapters, AdapterPlacement, AdapterSite, SiteKind};
use crate::wrga_spectral::{allocate_ranks, analyse_weight_map, RankAllocation, SpectralAnalysis};

// Re-export B.2 Task 2b observation surfaces so consumers can use
// `crate::wrga::{InitKind, InitStrategy}`.
pub use crate::wrga_adapter_inject::{InitKind, InitStrategy};

/// Input to the WRGA driver.
#[derive(Clone)]
pub struct WrgaInput<'a> {
    pub mode: WrgaMode,
    /// Parameter names to *exclude* from freezing (i.e. these are trainable).
    /// Supports the same glob dialect as `wrga_prune::glob_match` —
    /// e.g. `blocks.6.*`.
    pub trainable_patterns: Vec<&'a str>,
    /// Explicit adapter target layer globs for `mode=manual`.  Ignored for
    /// `auto` (which runs placement automatically) and `hybrid` (which uses
    /// the layer scope instead).
    pub manual_adapter_targets: Vec<&'a str>,
    /// Explicit layer scope for `mode=hybrid`.
    pub hybrid_layers: Vec<&'a str>,
    /// Forward Wengert list (from source-AD extraction).
    pub wengert: &'a WengertList,
    /// VarId of the scalar loss (the adjoint seed).
    pub loss_output: VarId,
    /// Optional pre-trained weights for spectral analysis.
    pub weights: Option<&'a WeightMap>,
    /// Target GPU name.  Looked up in `gpu_specs::GPU_DATABASE`.  Empty /
    /// unknown falls back to `default_gpu()`.
    pub target: &'a str,
    /// Total adapter parameter budget.  A budget of 0 means "unbounded"; in
    /// that case WRGA falls back to `r_max`-sized adapters on every site that
    /// the roofline picks.
    pub budget_params: usize,
    /// Rank band.
    pub r_min: usize,
    pub r_max: usize,
    /// Deterministic seed for randomized SVD.
    pub seed: u64,
}

/// Compiled WRGA plan — consumed by downstream codegen stages.
#[derive(Debug, Clone)]
pub struct WrgaPlan {
    pub mode: WrgaMode,
    pub target_gpu: String,
    pub prune: PruneResult,
    pub placements: Vec<AdapterPlacement>,
    pub spectral: Vec<SpectralAnalysis>,
    pub ranks: Vec<RankAllocation>,
    pub fusion: FusionPlan,
    pub memory: MemoryPlan,
}

impl WrgaPlan {
    /// Total adapter parameter count after rank allocation.
    pub fn adapter_params(&self) -> usize {
        self.ranks.iter().map(|r| r.adapter_params).sum()
    }

    /// Render a human-readable report, matching the layout of
    /// `nsl build --wrga-report`.
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        writeln!(s, "=== WRGA Compilation Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target hardware: {}", self.target_gpu).unwrap();
        writeln!(s).unwrap();
        writeln!(
            s,
            "Backward pruning: {} ops → {} ops ({:.1}% eliminated)",
            self.prune.stats.forward_ops_total,
            self.prune.stats.backward_ops_retained,
            100.0 * self.prune.stats.backward_elimination_ratio()
        )
        .unwrap();
        writeln!(
            s,
            "Activation memory: {} tensors → {} tensors ({:.1}% eliminated)",
            self.prune.stats.full_backward_saved_activations,
            self.prune.stats.pruned_saved_activations,
            100.0 * self.prune.stats.activation_elimination_ratio()
        )
        .unwrap();
        writeln!(
            s,
            "Frozen parameters: {}  |  Trainable targets: {}",
            self.prune.stats.frozen_params, self.prune.stats.gradient_targets
        )
        .unwrap();
        writeln!(s).unwrap();
        writeln!(
            s,
            "Adapter parameters: {}  |  Sites: {}  |  Fused: {}/{} ({:.1}%)",
            self.adapter_params(),
            self.fusion.total_count(),
            self.fusion.fused_count(),
            self.fusion.total_count(),
            100.0 * self.fusion.fusion_ratio()
        )
        .unwrap();
        writeln!(
            s,
            "Memory plan: {} live activations → {} slots (reuse {:.1}%)",
            self.memory.stats.live_activations,
            self.memory.stats.slots_used,
            100.0 * self.memory.stats.reuse_ratio()
        )
        .unwrap();
        s
    }
}

/// Run the full WRGA driver.
pub fn run(input: WrgaInput) -> WrgaPlan {
    // ── Stage 1: PEFT topology extraction ─────────────────────────────────
    let trainable = select_trainable(input.wengert, &input);

    // ── Stage 2: Wengert pruning (dead gradient elimination) ──────────────
    let prune_result = prune(input.wengert, &trainable, input.loss_output);

    // ── Stage 3: Roofline analysis + placement ────────────────────────────
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);
    let sites = infer_sites_from_wengert(input.wengert, &trainable, &input);
    let placements = place_adapters(&sites, gpu, input.r_min, input.r_max);

    // ── Stage 4: Spectral rank allocation (if weights supplied) ───────────
    let (spectral, ranks) = run_spectral(input.weights, &placements, &input);

    // ── Stage 5: Fusion integration ───────────────────────────────────────
    let rank_overrides: Vec<usize> =
        ranks.iter().map(|r| r.rank).collect();
    let fusion = build_fusion_plan(&placements, Some(&rank_overrides));

    // ── Stage 6: Memory plan ──────────────────────────────────────────────
    let size_hints: SizeHints = HashMap::new(); // callers can enrich later
    let memory = plan_memory(
        input.wengert,
        &prune_result.activation_live,
        &size_hints,
        &{
            let mut s = BTreeSet::new();
            s.insert(input.loss_output);
            s
        },
    );

    WrgaPlan {
        mode: input.mode,
        target_gpu: gpu.name.to_string(),
        prune: prune_result,
        placements,
        spectral,
        ranks,
        fusion,
        memory,
    }
}

// ---------------------------------------------------------------------------
// Stage helpers
// ---------------------------------------------------------------------------

/// Determine the set of trainable parameter VarIds per [`WrgaMode`].
fn select_trainable(list: &WengertList, input: &WrgaInput) -> BTreeSet<VarId> {
    use crate::wrga_prune::glob_match;
    let mut out = BTreeSet::new();
    for op in &list.ops {
        let name = match &op.op {
            PrimalOp::Param(n) => n.as_str(),
            _ => continue,
        };
        let included = match input.mode {
            WrgaMode::Auto => {
                // If the caller supplied patterns, honour them; otherwise all
                // parameters are trainable.  `Auto` is expected to be paired
                // with an upstream `@freeze(exclude=...)` that resolves into
                // `trainable_patterns` — but we don't require it.
                if input.trainable_patterns.is_empty() {
                    true
                } else {
                    input
                        .trainable_patterns
                        .iter()
                        .any(|p| glob_match(p, name))
                }
            }
            WrgaMode::Manual => {
                // Manual mode: only explicit adapter targets receive gradients.
                input
                    .manual_adapter_targets
                    .iter()
                    .any(|p| glob_match(p, name))
            }
            WrgaMode::Hybrid => {
                // Hybrid: the layer scope defines trainability; treat each
                // layer prefix as `blocks.N.*`.
                input
                    .hybrid_layers
                    .iter()
                    .any(|layer| name.starts_with(layer))
            }
        };
        if included {
            out.insert(op.result);
        }
    }
    out
}

/// Infer candidate adapter sites by walking the Wengert list and attaching a
/// [`SiteKind`] to each trainable parameter based on its downstream op.
///
/// This is a simple heuristic — the real compiler has richer shape info via
/// the type-map — but it's good enough for the driver to produce a deterministic
/// placement plan even in unit tests.
fn infer_sites_from_wengert(
    list: &WengertList,
    trainable: &BTreeSet<VarId>,
    _input: &WrgaInput,
) -> Vec<AdapterSite> {
    let mut out = Vec::new();
    // Index param VarId → name for friendly site names.
    let mut param_name: BTreeMap<VarId, String> = BTreeMap::new();
    for op in &list.ops {
        if let PrimalOp::Param(n) = &op.op {
            param_name.insert(op.result, n.clone());
        }
    }

    // For each op that consumes a trainable param, decide a site kind from
    // the op type.  Matmul → Matmul, LayerNorm/RMSNorm → Norm, Softmax →
    // Softmax, Embedding → Embedding, others → skip.
    for op in &list.ops {
        let Some(&trainable_id) = op.inputs.iter().find(|v| trainable.contains(v)) else {
            continue;
        };
        let name = param_name
            .get(&trainable_id)
            .cloned()
            .unwrap_or_else(|| format!("var_{trainable_id}"));
        let kind = match op.op {
            PrimalOp::Matmul => SiteKind::Matmul,
            PrimalOp::LayerNorm { .. } | PrimalOp::RMSNorm { .. } => SiteKind::Norm,
            PrimalOp::Softmax { .. } | PrimalOp::LogSoftmax { .. } => SiteKind::Softmax,
            PrimalOp::Embedding => SiteKind::Embedding,
            _ => continue,
        };
        // No shape info in the Wengert list; use a modest default so the
        // cost model still produces a meaningful AI for the placement.
        out.push(AdapterSite {
            name,
            kind,
            shape: vec![512, 512],
            dtype_bytes: 2,
            batch: 1,
            seq: 128,
        });
    }
    out
}

fn run_spectral(
    weights: Option<&WeightMap>,
    placements: &[AdapterPlacement],
    input: &WrgaInput,
) -> (Vec<SpectralAnalysis>, Vec<RankAllocation>) {
    let Some(wm) = weights else {
        // Without weights, fall back to uniform r_max (clamped by budget) per
        // non-skipped placement.
        let ranks: Vec<RankAllocation> = placements
            .iter()
            .map(|p| RankAllocation {
                name: p.name.clone(),
                rank: p.suggested_rank.max(input.r_min).min(input.r_max),
                effective_rank: 0.0,
                adapter_params: p.suggested_rank * 1024, // rough guess
            })
            .collect();
        return (Vec::new(), ranks);
    };

    // Collect analyses only for sites we actually plan to adapt.
    let active_names: BTreeSet<&str> = placements
        .iter()
        .filter(|p| !matches!(p.adapter, crate::wrga_roofline::AdapterKind::Skip))
        .map(|p| p.name.as_str())
        .collect();
    let mut spectral = analyse_weight_map(wm, input.r_max.max(8), 5, 8, input.seed);
    spectral.retain(|s| active_names.iter().any(|n| s.name == *n));

    // Cross-reference: per-site roofline slack as the allocator weight.
    let mut slack = HashMap::new();
    for p in placements {
        slack.insert(p.name.clone(), p.roofline_slack.max(1e-6));
    }

    // Derive a parameter budget: if the user specified one, use it; otherwise
    // let every site claim `r_max` (i.e. R_total = r_max · sites).
    let budget = if input.budget_params > 0 {
        input.budget_params
    } else {
        spectral.len().max(1) * input.r_max
    };
    let ranks = allocate_ranks(&spectral, budget, input.r_min, input.r_max, Some(&slack));

    (spectral, ranks)
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

    fn tiny_transformer_like() -> WengertList {
        // x (#0)          input
        // wq (#1)         param  blocks.6.wq
        // wk (#2)         param  blocks.6.wk
        // wv (#3)         param  blocks.7.wq   (trainable layer)
        // q = wq @ x      (#4)
        // k = wk @ x      (#5)  <- frozen path
        // v = wv @ x      (#6)  <- trainable path
        // y = q + v       (#7)
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("blocks.6.wq".into()), vec![]),
            op(2, 2, PrimalOp::Param("blocks.6.wk".into()), vec![]),
            op(3, 3, PrimalOp::Param("blocks.7.wq".into()), vec![]),
            op(4, 4, PrimalOp::Matmul, vec![1, 0]),
            op(5, 5, PrimalOp::Matmul, vec![2, 0]),
            op(6, 6, PrimalOp::Matmul, vec![3, 0]),
            op(7, 7, PrimalOp::Add, vec![4, 6]),
        ];
        WengertList {
            ops,
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    #[test]
    fn auto_mode_produces_plan_without_weights() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
        };
        let plan = run(input);
        // Exactly one param (blocks.7.wq) is trainable → exactly one site.
        assert_eq!(plan.prune.stats.gradient_targets, 1);
        assert_eq!(plan.prune.stats.frozen_params, 2);
        // The report must render without panicking.
        let report = plan.render_report();
        assert!(report.contains("WRGA Compilation Report"));
        assert!(report.contains("Mode: auto"));
    }

    #[test]
    fn manual_mode_only_adapts_explicit_targets() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Manual,
            trainable_patterns: Vec::new(),
            manual_adapter_targets: vec!["blocks.6.wq"],
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
        };
        let plan = run(input);
        assert_eq!(plan.prune.stats.gradient_targets, 1);
    }

    #[test]
    fn hybrid_mode_layer_scope() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Hybrid,
            trainable_patterns: Vec::new(),
            manual_adapter_targets: Vec::new(),
            hybrid_layers: vec!["blocks.6"],
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
        };
        let plan = run(input);
        // Both blocks.6.wq and blocks.6.wk match the layer scope.
        assert_eq!(plan.prune.stats.gradient_targets, 2);
        assert_eq!(plan.prune.stats.frozen_params, 1);
    }

    #[test]
    fn render_report_is_deterministic() {
        let w = tiny_transformer_like();
        let make = || WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
        };
        assert_eq!(run(make()).render_report(), run(make()).render_report());
    }
}
