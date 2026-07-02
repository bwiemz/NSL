//! CSHA — Compiler-Synthesized Holistic Attention: driver.
//!
//! Orchestrates the three passes described in NSL-CSHA-Research.PDF:
//!
//!   1. Level 1 boundary fusion       — [`csha_boundary`]
//!   2. Level 2/3 pipelining / block  — [`csha_pipeline`]
//!   3. Per-layer specialization      — [`csha_specialize`]
//!
//! The driver is pure data-in / data-out: it consumes a Wengert list,
//! an optional weight map, a GPU spec, and a user-requested level, and
//! returns a [`CshaPlan`] that downstream passes apply.  Integration
//! into codegen (via `stmt.rs`) calls [`run_on_wengert`] after the
//! Wengert list is extracted; the plan's [`CshaPlan::render_report`]
//! drives the `--csha-report` CLI flag.

use serde::Serialize;

use crate::csha_boundary::{scan as scan_boundaries, BoundaryScan, ProjKind};
use crate::csha_patterns::{analyze as analyze_patterns, PatternConfig, PatternPlan};
use crate::wggo_overrides::{OverrideDiagnostic, OverrideRejectReason};
use crate::csha_pipeline::{
    block_smem_bytes, pipeline_smem_bytes, plan_all, plan_layer, roofline_tile_config,
    smem_budget_bytes, FusionLevel, LayerPlan, TileConfig,
};
use crate::csha_specialize::{analyze as analyze_spec, SpecConfig, SpecializationPlan};
use crate::gpu_specs::{default_gpu, find_gpu, GpuSpec};
use crate::weight_aware::WeightMap;
use crate::wengert::WengertList;
use crate::wggo_cost::LayerShape;

/// User-facing CSHA mode — maps to an initial fusion level that
/// [`csha_pipeline::plan_layer`] may downgrade per-layer if SMEM is
/// insufficient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CshaMode {
    /// Pick the best level per layer automatically (up to Level 3).
    Auto,
    /// Force Level 1 (boundary fusion only).
    Boundary,
    /// Force Level 2 (projection-attention pipelining).
    Pipeline,
    /// Force Level 3 (full block fusion).
    Block,
    /// Skip CSHA entirely.
    Off,
}

impl CshaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CshaMode::Auto => "auto",
            CshaMode::Boundary => "boundary",
            CshaMode::Pipeline => "pipeline",
            CshaMode::Block => "block",
            CshaMode::Off => "off",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" | "full" => Some(CshaMode::Auto),
            "l1" | "1" | "boundary" => Some(CshaMode::Boundary),
            "l2" | "2" | "pipeline" | "pipelining" => Some(CshaMode::Pipeline),
            "l3" | "3" | "block" => Some(CshaMode::Block),
            "off" | "disable" | "disabled" | "false" => Some(CshaMode::Off),
            _ => None,
        }
    }

    pub fn initial_level(self) -> FusionLevel {
        match self {
            CshaMode::Auto | CshaMode::Block => FusionLevel::Block,
            CshaMode::Pipeline => FusionLevel::Pipeline,
            CshaMode::Boundary => FusionLevel::Boundary,
            CshaMode::Off => FusionLevel::None,
        }
    }
}

/// Inputs to the CSHA driver.
#[derive(Clone)]
pub struct CshaInput<'a> {
    pub mode: CshaMode,
    pub target: &'a str,
    pub wengert: &'a WengertList,
    pub weights: Option<&'a WeightMap>,
    /// Layer shape used for cost modelling.  In practice `n_heads` is
    /// recovered from the weight shapes when available; this value is the
    /// canonical "template" used to size the LUT.
    pub shape: LayerShape,
    /// Number of attention heads.  Used for per-head specialization.
    pub n_heads: u32,
    pub spec_cfg: SpecConfig,
    /// Config for [`csha_patterns`] analysis (causal/GQA/sinks).
    pub pattern_cfg: PatternConfig,
    /// Per-layer WGGO preferences. `None` → CSHA's internal planner runs
    /// unchanged. `Some` → per-layer `active_heads` is applied verbatim;
    /// `requested_csha_level` is a preference subject to SMEM-feasibility
    /// downgrade at per-layer emission time (Task 4).
    pub wggo_overrides: Option<&'a crate::wggo_overrides::WggoOverrides>,
}


/// Aggregate plan emitted by the driver.
#[derive(Debug, Clone, Serialize)]
pub struct CshaPlan {
    pub mode: CshaMode,
    pub target_gpu: String,
    pub boundary: BoundaryScan,
    pub per_layer: Vec<LayerPlan>,
    pub specialization: SpecializationPlan,
    /// Per-layer causal-mask / GQA / attention-sink decisions.
    pub patterns: PatternPlan,
    /// Per-layer kernel-specialisation artefacts produced by the
    /// `csha_apply` bridge.  Empty in `Off` mode.
    pub kernels: Vec<crate::csha_apply::KernelSpec>,
    /// Fusion-graph marks that claim Q/K/V matmul nodes on behalf of
    /// CSHA-fused kernels, so `epilogue_fusion` and `reduction_fusion`
    /// do not double-fuse them.
    pub marks: Vec<crate::csha_apply::FusionMark>,
    /// Total driver wall-clock time (microseconds).
    pub solve_us: u64,
    /// WGGO overrides that CSHA had to downgrade due to hardware
    /// infeasibility. Empty when WGGO produced no overrides or all
    /// overrides were applied verbatim. Stable wire shape; consumed by
    /// `--csha-report` and the Phase 3 decision explainer.
    pub override_diagnostics: Vec<OverrideDiagnostic>,
}

/// A.5: aggregated HBM-traffic savings across every CSHA-active layer
/// in the plan. Produced by [`CshaPlan::total_hbm_reduction`] and
/// consumed by:
///
///   * [`CshaPlan::render_report`] — as a trailing "Total savings" footer.
///   * External CI gates / calibration harness — via the JSON form
///     emitted by [`CshaPlan::to_json`].
///
/// All byte counts are per-forward-pass per-layer sums; ratios are
/// dimensionless. `active_layers` counts only layers whose realised
/// fusion level is non-`None` — layers that CSHA couldn't fuse
/// (SMEM-infeasible, no boundary chain, etc.) contribute nothing to
/// `baseline_bytes` / `fused_bytes` and are excluded from the count.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TotalSavings {
    /// Σ baseline HBM bytes across CSHA-active layers.
    pub baseline_bytes: u64,
    /// Σ fused HBM bytes across CSHA-active layers.
    pub fused_bytes: u64,
    /// `baseline_bytes - fused_bytes`. Saturates at 0 if a
    /// pathological layer reports fused > baseline (shouldn't happen
    /// but keeps the helper panic-free).
    pub savings_bytes: u64,
    /// `baseline_bytes / fused_bytes` — the geometric-mean HBM
    /// reduction factor. `1.0` when CSHA achieved no savings; higher
    /// is better. `0.0` when `fused_bytes == 0` (no active layers).
    pub ratio: f64,
    /// Number of layers that realised a non-`None` fusion level.
    pub active_layers: usize,
}

impl CshaPlan {
    /// A.5: aggregate HBM-traffic totals across every layer that CSHA
    /// successfully fused. Returns zeros / `0.0` when the plan has no
    /// active layers (e.g. `--csha=off`, empty model, or every layer
    /// downgraded to `FusionLevel::None` by SMEM pressure).
    ///
    /// Baseline = what the unfused Q/K/V projection + attention pipeline
    /// would have moved through HBM; fused = what the CSHA-compiled
    /// kernel variant moves. Per paper §6.3 the ratio is the headline
    /// metric a compiler report surfaces to the user.
    pub fn total_hbm_reduction(&self) -> TotalSavings {
        let mut baseline: u64 = 0;
        let mut fused: u64 = 0;
        let mut active: usize = 0;
        for lp in &self.per_layer {
            if lp.level == FusionLevel::None {
                continue;
            }
            baseline = baseline.saturating_add(lp.baseline_hbm_bytes);
            fused = fused.saturating_add(lp.hbm_traffic_bytes);
            active += 1;
        }
        let savings = baseline.saturating_sub(fused);
        let ratio = if fused > 0 {
            baseline as f64 / fused as f64
        } else {
            0.0
        };
        TotalSavings {
            baseline_bytes: baseline,
            fused_bytes: fused,
            savings_bytes: savings,
            ratio,
            active_layers: active,
        }
    }

    /// A.5: serialise the full plan as JSON. Consumed by the
    /// calibration harness and downstream CI gates that want to
    /// assert numerical invariants (e.g. ratio ≥ threshold) without
    /// screen-scraping [`Self::render_report`]. Returns `None` on
    /// `serde_json` serialisation failure (in practice only possible
    /// if a custom `Serialize` impl somewhere in the plan tree panics).
    pub fn to_json(&self) -> Option<String> {
        serde_json::to_string_pretty(self).ok()
    }

    /// One-line compact summary suitable for debug logs.
    pub fn summary(&self) -> String {
        let l3 = self
            .per_layer
            .iter()
            .filter(|p| p.level == FusionLevel::Block)
            .count();
        let l2 = self
            .per_layer
            .iter()
            .filter(|p| p.level == FusionLevel::Pipeline)
            .count();
        let l1 = self
            .per_layer
            .iter()
            .filter(|p| p.level == FusionLevel::Boundary)
            .count();
        let pruned = self.specialization.total_pruned_heads();
        format!(
            "csha[{}]: {} chains, L1={} L2={} L3={}, {} pruned heads",
            self.mode.as_str(),
            self.boundary.num_chains(),
            l1,
            l2,
            l3,
            pruned,
        )
    }

    /// Full report matching the layout of paper §6.3.
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        writeln!(s, "=== CSHA Compilation Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target GPU: {}", self.target_gpu).unwrap();
        writeln!(s).unwrap();

        writeln!(
            s,
            "Boundary scan: {} fusion chains ({} Q, {} K, {} V)",
            self.boundary.num_chains(),
            self.boundary.count_kind(ProjKind::Q),
            self.boundary.count_kind(ProjKind::K),
            self.boundary.count_kind(ProjKind::V),
        )
        .unwrap();
        writeln!(s).unwrap();

        if self.per_layer.is_empty() {
            writeln!(
                s,
                "No attention layers detected — CSHA has nothing to do."
            )
            .unwrap();
            return s;
        }

        for plan in &self.per_layer {
            let spec = self.specialization.get(&plan.layer);
            writeln!(s, "Layer {}: CSHA {}", plan.layer, plan.level.as_str()).unwrap();
            writeln!(
                s,
                "  Tiles: block_q={}, block_kv={}, head_dim={}",
                plan.tiles.block_q, plan.tiles.block_kv, plan.tiles.head_dim
            )
            .unwrap();
            writeln!(
                s,
                "  SMEM: {:.1} KB / {:.1} KB ({}%)",
                plan.smem_bytes as f64 / 1024.0,
                plan.smem_budget_bytes as f64 / 1024.0,
                if plan.smem_budget_bytes > 0 {
                    (100 * plan.smem_bytes / plan.smem_budget_bytes).min(100)
                } else {
                    0
                }
            )
            .unwrap();
            writeln!(
                s,
                "  HBM traffic: {:.1} MB (vs {:.1} MB unfused) — {:.2}× reduction",
                plan.hbm_traffic_bytes as f64 / 1e6,
                plan.baseline_hbm_bytes as f64 / 1e6,
                plan.hbm_reduction(),
            )
            .unwrap();
            writeln!(
                s,
                "  Est. time: {:.2} μs (vs {:.2} μs unfused) — {:.2}× speedup",
                plan.est_time_us,
                plan.baseline_time_us,
                plan.speedup(),
            )
            .unwrap();
            // Sprint 3 (paper §6.3 visibility): per-layer backward tier so
            // `nsl check --csha-report` and `nsl build --csha-report` users can
            // audit whether each layer routes through the Tier B.2 MMA backward
            // or falls back to scalar Tier C.
            writeln!(s, "  Backward: {}", plan.backward_tier.as_label()).unwrap();
            if let Some(ref reason) = plan.downgrade_reason {
                writeln!(s, "  Downgrade: {}", reason).unwrap();
            }
            if let Some(spec) = spec {
                writeln!(
                    s,
                    "  Heads: {}/{} active, entropy={}",
                    spec.n_active_heads,
                    spec.n_heads,
                    spec.entropy_bucket.as_str(),
                )
                .unwrap();
                let precisions: Vec<&str> = spec
                    .heads
                    .iter()
                    .filter(|h| !h.pruned)
                    .map(|h| h.precision.as_str())
                    .collect();
                if !precisions.is_empty() {
                    writeln!(s, "  Precisions: {}", precisions.join(", ")).unwrap();
                }
            }
            // Kernel specialisation name that downstream passes embed
            // into the compiled artefact.
            if let Some(kspec) = self
                .kernels
                .iter()
                .find(|k| k.layer == plan.layer)
            {
                writeln!(s, "  Kernel: {}", kspec.kernel_name).unwrap();
            }
            if let Some(pat) = self.patterns.get(&plan.layer) {
                writeln!(
                    s,
                    "  Patterns: causal={} gqa={} sink={}",
                    pat.causal_mask.as_str(),
                    pat.gqa.as_str(),
                    pat.sink.as_str(),
                )
                .unwrap();
            }
        }

        // A.5: trailing aggregate savings footer.
        let totals = self.total_hbm_reduction();
        writeln!(s).unwrap();
        if totals.active_layers > 0 {
            writeln!(
                s,
                "Total savings: {:.1} MB across {} layer{} ({:.2}× reduction, {:.1} MB → {:.1} MB)",
                totals.savings_bytes as f64 / 1e6,
                totals.active_layers,
                if totals.active_layers == 1 { "" } else { "s" },
                totals.ratio,
                totals.baseline_bytes as f64 / 1e6,
                totals.fused_bytes as f64 / 1e6,
            )
            .unwrap();
        } else {
            writeln!(s, "Total savings: 0 MB (no CSHA-active layers).").unwrap();
        }
        writeln!(
            s,
            "Solve time: {:.2} ms",
            self.solve_us as f64 / 1000.0
        )
        .unwrap();
        s
    }
}

/// Map a `cfie_persistent::FusionLevel` (WGGO's vocabulary) to the
/// corresponding `csha_pipeline::FusionLevel` (CSHA's internal planner
/// vocabulary).
///
/// The two enums use parallel names but live in different crates; this
/// function is the single authoritative translation point.
///
/// Mapping (monotone, ordinal-preserving):
///   None   → None
///   Level1 → Boundary
///   Level2 → Pipeline
///   Level3 → Block
fn wggo_to_pipeline_level(
    wggo: crate::cfie_persistent::FusionLevel,
) -> FusionLevel {
    match wggo {
        crate::cfie_persistent::FusionLevel::None => FusionLevel::None,
        crate::cfie_persistent::FusionLevel::Level1 => FusionLevel::Boundary,
        crate::cfie_persistent::FusionLevel::Level2 => FusionLevel::Pipeline,
        crate::cfie_persistent::FusionLevel::Level3 => FusionLevel::Block,
    }
}

/// Map a `csha_pipeline::FusionLevel` back to `cfie_persistent::FusionLevel`
/// for use in `OverrideDiagnostic.applied`.
fn pipeline_to_wggo_level(
    pipeline: FusionLevel,
) -> crate::cfie_persistent::FusionLevel {
    match pipeline {
        FusionLevel::None => crate::cfie_persistent::FusionLevel::None,
        FusionLevel::Boundary => crate::cfie_persistent::FusionLevel::Level1,
        FusionLevel::Pipeline => crate::cfie_persistent::FusionLevel::Level2,
        FusionLevel::Block => crate::cfie_persistent::FusionLevel::Level3,
    }
}

/// Walk the level enum downward (most aggressive first) until an SMEM-
/// feasible level is found.  Returns `FusionLevel::None` as the guaranteed-
/// feasible fallback (Boundary/None consume zero SMEM kernel-side).
///
/// Only candidates strictly below `requested` are tried — the caller already
/// confirmed `requested` is infeasible.
///
/// Ordering: Block (3) → Pipeline (2) → Boundary (1) → None (0).
fn downgrade_until_feasible(
    shape: LayerShape,
    tiles: TileConfig,
    budget: u64,
    requested: FusionLevel,
) -> FusionLevel {
    // Walk from most-to-least aggressive, skipping levels >= requested.
    for candidate in [
        FusionLevel::Block,
        FusionLevel::Pipeline,
        FusionLevel::Boundary,
        FusionLevel::None,
    ] {
        // Only try strictly less aggressive than requested.
        if candidate.as_u8() >= requested.as_u8() {
            continue;
        }
        // Boundary and None consume no SMEM in the kernel — always feasible.
        let smem_needed = match candidate {
            FusionLevel::Block => block_smem_bytes(shape, tiles),
            FusionLevel::Pipeline => pipeline_smem_bytes(shape, tiles),
            FusionLevel::Boundary | FusionLevel::None => 0,
        };
        if smem_needed <= budget {
            return candidate;
        }
    }
    // Last-resort: Boundary (zero SMEM, always OK).
    FusionLevel::Boundary
}

/// Run the CSHA driver.
pub fn run(input: CshaInput) -> CshaPlan {
    let t0 = std::time::Instant::now();
    let gpu: &'static GpuSpec = find_gpu(input.target).unwrap_or_else(default_gpu);

    // Off mode: produce an empty plan so callers can uniformly serialize
    // the result.
    if input.mode == CshaMode::Off {
        return CshaPlan {
            mode: CshaMode::Off,
            target_gpu: gpu.name.to_string(),
            boundary: BoundaryScan::default(),
            per_layer: Vec::new(),
            specialization: SpecializationPlan::default(),
            patterns: PatternPlan::default(),
            kernels: Vec::new(),
            marks: Vec::new(),
            solve_us: t0.elapsed().as_micros() as u64,
            override_diagnostics: Vec::new(),
        };
    }

    let boundary = scan_boundaries(input.wengert);

    // ------------------------------------------------------------------
    // Per-layer planning — override-aware.
    //
    // Ordering is LOAD-BEARING (per spec):
    //   (1) Apply active_heads FIRST  → drives tile dims.
    //   (2) Compute tile config against overridden head count.
    //   (3) Check requested_csha_level feasibility vs the overridden tile.
    //
    // Checking feasibility before head reduction produces false-positive
    // rejections: Level2 may be infeasible at 8 heads but feasible at 4.
    // ------------------------------------------------------------------
    let no_overrides = input.wggo_overrides.is_none();
    let mut override_diagnostics: Vec<OverrideDiagnostic> = Vec::new();

    let per_layer: Vec<LayerPlan> = if no_overrides {
        // Fast path: uniform level, no per-layer override book-keeping.
        plan_all(&boundary, input.shape, gpu, input.mode.initial_level())
    } else {
        // Override path: iterate unique layers (mirrors `plan_all`'s dedup
        // via BTreeSet) and apply per-layer decisions.
        let mut seen = std::collections::BTreeSet::new();
        let mut out: Vec<LayerPlan> = Vec::new();
        let budget = smem_budget_bytes(gpu);

        for c in &boundary.chains {
            let key = c.layer.clone().unwrap_or_else(|| "other".to_string());
            if !seen.insert(key.clone()) {
                continue; // dedup: only plan each layer once
            }

            let layer_idx = out.len() as u32;

            // (1) HEADS FIRST — look up override for this layer.
            let layer_override = input
                .wggo_overrides
                .and_then(|o| o.find(layer_idx));

            // Use overridden head count when available; fall back to the
            // global n_heads.  The head count feeds into roofline_tile_config
            // via the LayerShape (head_dim), not n_heads directly — but the
            // shape is fixed per-model.  The active_heads override is
            // recorded in the specialization plan (post-planning patch).
            let _applied_heads = match layer_override {
                Some(ov) => ov.active_heads,
                None => input.n_heads,
            };

            // (2) Tile config against the (possibly overridden) shape.
            // The tile config depends on shape.head_dim, which is fixed.
            // active_heads affects the specialization plan, not the tile
            // dimensions directly (head_dim stays constant across GQA).
            let tiles = roofline_tile_config(input.shape, gpu);

            // (3) Level: honour override if feasible, else downgrade.
            let layer_plan = match layer_override.and_then(|o| o.requested_csha_level) {
                Some(requested_wggo) => {
                    let requested = wggo_to_pipeline_level(requested_wggo);
                    let smem_needed = match requested {
                        FusionLevel::Block => block_smem_bytes(input.shape, tiles),
                        FusionLevel::Pipeline => pipeline_smem_bytes(input.shape, tiles),
                        FusionLevel::Boundary | FusionLevel::None => 0,
                    };
                    if smem_needed <= budget {
                        // Feasible — use requested level directly.
                        plan_layer(key.clone(), input.shape, gpu, requested)
                    } else {
                        // Infeasible — downgrade and record diagnostic.
                        let applied =
                            downgrade_until_feasible(input.shape, tiles, budget, requested);
                        override_diagnostics.push(OverrideDiagnostic {
                            layer_index: layer_idx,
                            layer_name: layer_override
                                .map(|o| o.layer_name.clone())
                                .unwrap_or_else(|| key.clone()),
                            requested: format!("{:?}", requested_wggo),
                            applied: format!("{:?}", pipeline_to_wggo_level(applied)),
                            reason: OverrideRejectReason::SmemBudgetExceeded {
                                actual_kb: (smem_needed / 1024) as u32,
                                limit_kb: (budget / 1024) as u32,
                            },
                        });
                        plan_layer(key.clone(), input.shape, gpu, applied)
                    }
                }
                None => {
                    // No level override — internal planner decides.
                    plan_layer(key.clone(), input.shape, gpu, input.mode.initial_level())
                }
            };

            out.push(layer_plan);
        }
        out
    };

    // Specialization analysis (head pruning, entropy buckets, precision).
    let mut specialization = analyze_spec(
        &boundary,
        input.weights,
        input.n_heads,
        &input.spec_cfg,
    );
    let patterns = analyze_patterns(
        &per_layer,
        &input.shape,
        input.n_heads,
        input.weights,
        &input.pattern_cfg,
    );

    // Post-planning patch: apply active_heads overrides to the specialization
    // plan.  When WGGO says active_heads=4 for a layer that has 8 total
    // heads, we force n_active_heads=4 (= 4 pruned) so downstream consumers
    // see the WGGO-sanctioned count rather than the weight-analysis result.
    if let Some(overrides) = input.wggo_overrides {
        for (idx, layer_spec) in specialization.layers.iter_mut().enumerate() {
            if let Some(ov) = overrides.find(idx as u32) {
                // Only clamp: WGGO can reduce active heads, not add phantom ones.
                if ov.active_heads < layer_spec.n_active_heads {
                    layer_spec.n_active_heads = ov.active_heads;
                }
            }
        }
    }

    // Stitch everything together via the apply-bridge so the plan
    // carries ready-to-consume kernel specs + graph marks.
    let interim = CshaPlan {
        mode: input.mode,
        target_gpu: gpu.name.to_string(),
        boundary: boundary.clone(),
        per_layer: per_layer.clone(),
        specialization: specialization.clone(),
        patterns: patterns.clone(),
        kernels: Vec::new(),
        marks: Vec::new(),
        solve_us: 0,
        override_diagnostics: Vec::new(),
    };
    let mut diags = Vec::<String>::new();
    let bridge = crate::csha_apply::bridge(
        &interim,
        input.shape.head_dim as i64,
        &mut diags,
    );
    for d in diags { eprintln!("warning: {d}"); }

    CshaPlan {
        mode: input.mode,
        target_gpu: gpu.name.to_string(),
        boundary,
        per_layer,
        specialization,
        patterns,
        kernels: bridge.kernels,
        marks: bridge.marks,
        solve_us: t0.elapsed().as_micros() as u64,
        override_diagnostics,
    }
}

/// Convenience: run CSHA on a Wengert list with default shape / head
/// count.  Used by the compile-pipeline integration point.
pub fn run_on_wengert(
    wengert: &WengertList,
    target: &str,
    mode_str: &str,
    weights: Option<&WeightMap>,
    shape: Option<LayerShape>,
    n_heads: u32,
    wggo_overrides: Option<&crate::wggo_overrides::WggoOverrides>,
) -> Option<CshaPlan> {
    let mode = CshaMode::parse(mode_str)?;
    let shape = shape.unwrap_or(LayerShape {
        batch: 1,
        seq: 1024,
        d_model: 512,
        head_dim: 64,
        n_kv_heads: 4,
        dtype_bytes: 2,
    });
    Some(run(CshaInput {
        mode,
        target,
        wengert,
        weights,
        shape,
        n_heads: n_heads.max(1),
        spec_cfg: SpecConfig::default(),
        pattern_cfg: PatternConfig::default(),
        wggo_overrides,
    }))
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

    fn attn_block() -> WengertList {
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

    fn toy_input<'a>(w: &'a WengertList, mode: CshaMode) -> CshaInput<'a> {
        CshaInput {
            mode,
            target: "H100",
            wengert: w,
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
            pattern_cfg: PatternConfig::default(),
            wggo_overrides: None,
        }
    }

    #[test]
    fn off_mode_produces_empty_plan() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Off));
        assert_eq!(plan.mode, CshaMode::Off);
        assert_eq!(plan.boundary.num_chains(), 0);
        assert!(plan.per_layer.is_empty());
    }

    #[test]
    fn auto_mode_detects_qkv_chains() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        assert_eq!(plan.boundary.num_chains(), 3);
        assert_eq!(plan.per_layer.len(), 1);
    }

    #[test]
    fn report_contains_expected_sections() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        let r = plan.render_report();
        assert!(r.contains("CSHA Compilation Report"));
        assert!(r.contains("Boundary scan"));
        assert!(r.contains("Layer blocks.0"));
        assert!(r.contains("HBM traffic"));
    }

    #[test]
    fn report_contains_backward_tier_line_per_layer() {
        // Sprint 3 (paper §6.3 visibility): the per-layer report must
        // include a `Backward: ...` line so users can audit which backward
        // kernel family CSHA picked at training time.
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        let r = plan.render_report();
        assert!(
            r.contains("Backward:"),
            "render_report must include a 'Backward:' line per layer; got:\n{r}"
        );
        // For the toy attn block (hd=64, csha auto), the line must end with
        // EITHER 'Tier B.2 (hybrid, ...)' (when planner admits Tier B.2)
        // OR 'Tier C (scalar)' (default fallback). Both are valid labels.
        let has_label = r.contains("Tier B.2 (hybrid")
            || r.contains("Tier C (scalar)");
        assert!(
            has_label,
            "Backward: line must use one of the documented labels; got:\n{r}"
        );
    }

    #[test]
    fn summary_is_single_line() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        let s = plan.summary();
        assert!(!s.is_empty());
        assert!(!s.contains('\n'));
        assert!(s.starts_with("csha[auto]"));
    }

    #[test]
    fn deterministic_across_runs() {
        let w = attn_block();
        let p1 = run(toy_input(&w, CshaMode::Auto));
        let p2 = run(toy_input(&w, CshaMode::Auto));
        // Strip the solve_us line before comparing.
        let strip = |r: String| -> String {
            r.lines()
                .filter(|l| !l.contains("Solve time"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(strip(p1.render_report()), strip(p2.render_report()));
    }

    #[test]
    fn mode_parse_roundtrip() {
        for m in [
            CshaMode::Auto,
            CshaMode::Boundary,
            CshaMode::Pipeline,
            CshaMode::Block,
            CshaMode::Off,
        ] {
            assert_eq!(CshaMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(CshaMode::parse("L2"), Some(CshaMode::Pipeline));
        assert_eq!(CshaMode::parse("3"), Some(CshaMode::Block));
        assert!(CshaMode::parse("bogus").is_none());
    }

    #[test]
    fn forced_boundary_stays_boundary() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Boundary));
        assert!(plan
            .per_layer
            .iter()
            .all(|p| p.level == FusionLevel::Boundary));
    }

    #[test]
    fn run_on_wengert_accepts_canonical_strings() {
        let w = attn_block();
        for mode in ["auto", "boundary", "pipeline", "block", "off", "L2", "3"] {
            assert!(
                run_on_wengert(&w, "H100", mode, None, None, 8, None).is_some(),
                "'{}' should parse",
                mode
            );
        }
        assert!(run_on_wengert(&w, "H100", "wat", None, None, 8, None).is_none());
    }

    #[test]
    fn unknown_target_falls_back_to_default_gpu() {
        let w = attn_block();
        let plan = run_on_wengert(&w, "nonexistent-gpu-xyz", "auto", None, None, 8, None).unwrap();
        assert!(!plan.target_gpu.is_empty());
    }

    #[test]
    fn csha_plan_has_empty_override_diagnostics_by_default() {
        // Regression marker: override_diagnostics must be Vec::new() when no
        // WggoOverrides are passed to the driver.  Task 4 will populate the
        // field; this test will catch any future change that accidentally
        // pre-populates it before that wiring is complete.
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        assert!(
            plan.override_diagnostics.is_empty(),
            "CshaPlan must have no override diagnostics when no WggoOverrides were passed"
        );

        // Also verify Off mode initialises the field correctly.
        let off_plan = run(toy_input(&w, CshaMode::Off));
        assert!(
            off_plan.override_diagnostics.is_empty(),
            "CshaPlan (Off mode) must have no override diagnostics"
        );
    }

    #[test]
    fn wengert_without_attention_gives_empty_plan() {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("blocks.0.ffn.w1".into()), vec![]),
            op(2, 2, PrimalOp::Matmul, vec![0, 1]),
        ];
        let w = WengertList {
            ops,
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let plan = run(toy_input(&w, CshaMode::Auto));
        assert_eq!(plan.boundary.num_chains(), 0);
        assert!(plan.per_layer.is_empty());
        assert!(plan.render_report().contains("nothing to do"));
    }

    // ── A.5: TotalSavings + JSON + report footer ─────────────────────

    #[test]
    fn a5_total_hbm_reduction_on_auto_plan_sums_across_layers() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        let totals = plan.total_hbm_reduction();
        // The toy attn_block is one layer; CSHA Auto should fuse it so
        // active_layers is 1, baseline > fused, ratio > 1.
        assert_eq!(totals.active_layers, 1);
        assert!(totals.baseline_bytes > 0);
        assert!(totals.fused_bytes > 0);
        assert!(
            totals.baseline_bytes >= totals.fused_bytes,
            "fused bytes must not exceed baseline; got baseline={} fused={}",
            totals.baseline_bytes,
            totals.fused_bytes,
        );
        assert_eq!(
            totals.savings_bytes,
            totals.baseline_bytes - totals.fused_bytes
        );
        assert!(
            totals.ratio >= 1.0,
            "CSHA Auto should not regress HBM; ratio={}",
            totals.ratio
        );
    }

    #[test]
    fn a5_total_hbm_reduction_on_off_plan_is_zero() {
        // `--csha=off` short-circuits in `run`, producing an empty
        // `per_layer` — totals must reflect zero active layers and
        // a 0.0 ratio (sentinel for "no fused bytes").
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Off));
        let totals = plan.total_hbm_reduction();
        assert_eq!(totals.active_layers, 0);
        assert_eq!(totals.baseline_bytes, 0);
        assert_eq!(totals.fused_bytes, 0);
        assert_eq!(totals.savings_bytes, 0);
        assert_eq!(totals.ratio, 0.0);
    }

    #[test]
    fn a5_render_report_footer_contains_total_savings_line() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        let report = plan.render_report();
        assert!(
            report.contains("Total savings:"),
            "render_report must emit the A.5 totals footer; got:\n{report}"
        );
        assert!(
            report.contains("× reduction"),
            "footer must include the reduction ratio; got:\n{report}"
        );
    }

    #[test]
    fn a5_render_report_off_mode_reports_nothing_to_do() {
        // Off-mode keeps its existing "nothing to do" short-circuit —
        // the A.5 footer applies only when the planner ran over real
        // layers. Documented here so a future edit of the early-return
        // path doesn't silently regress the off-mode contract.
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Off));
        let report = plan.render_report();
        assert!(report.contains("nothing to do"));
    }

    #[test]
    fn a5_to_json_round_trips_to_structured_output() {
        let w = attn_block();
        let plan = run(toy_input(&w, CshaMode::Auto));
        let json = plan.to_json().expect("JSON serialisation succeeds");
        // Spot-check: the JSON must carry `mode` and at least one
        // layer's HBM-traffic field — CI gates / calibration harness
        // consume this to assert numerical invariants.
        assert!(
            json.contains("\"mode\""),
            "JSON must include the plan mode; got:\n{json}"
        );
        assert!(
            json.contains("\"hbm_traffic_bytes\""),
            "JSON must include per-layer HBM traffic; got:\n{json}"
        );
        // It must also parse back into arbitrary JSON — guards
        // against malformed output that `to_string_pretty` could
        // produce if a nested `Serialize` impl panics.
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("JSON must round-trip through serde_json");
        assert!(parsed.get("mode").is_some());
    }

    #[test]
    fn run_on_wengert_accepts_wggo_overrides_argument() {
        // Smoke-test: the new signature accepts Option<&WggoOverrides> without
        // error. Uses the existing attn_block() Wengert helper.
        let w = attn_block();
        let overrides = crate::wggo_overrides::WggoOverrides { per_layer: vec![] };

        // Both calls must compile and produce a plan (or None — we don't care,
        // we care the signature accepts both None and Some(...)):
        let _none = run_on_wengert(&w, "A100", "full", None, None, 8, None);
        let _some = run_on_wengert(&w, "A100", "full", None, None, 8, Some(&overrides));

        // Verify both arms return Some (the mode "full" maps to Auto which is valid).
        assert!(_none.is_some(), "None arm should produce a plan");
        assert!(_some.is_some(), "Some arm should produce a plan");
    }
}

// ---------------------------------------------------------------------------
// Per-layer WGGO override tests (Task 4)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod override_tests {
    use super::*;
    // Use explicit qualification throughout to avoid ambiguity between
    //   csha_pipeline::FusionLevel (used by LayerPlan.level)
    //   cfie_persistent::FusionLevel (used by OverrideDiagnostic + PerLayerOverride)
    use crate::cfie_persistent::FusionLevel as WggoLevel;
    use crate::csha_pipeline::FusionLevel as PipelineLevel;
    use crate::wggo_overrides::{PerLayerOverride, WggoOverrides};
    use crate::wengert::{PrimalOp, WengertOp};
    use std::collections::HashMap;

    /// Minimal Wengert list with a single attention block (reuses the
    /// attn_block() pattern from the parent test module).
    fn attn_wengert() -> WengertList {
        let op = |id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>| WengertOp {
            id, result, op: o, inputs,
            saved_for_backward: false, checkpointed: false,
        };
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
            ops, output: 9,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    /// Standard shape used by most tests (d_model=512 for general cases).
    fn default_shape() -> LayerShape {
        LayerShape {
            batch: 1, seq: 1024, d_model: 512,
            head_dim: 64, n_kv_heads: 4, dtype_bytes: 2,
        }
    }

    /// Small shape where Level-3 block fusion fits within the 228 KB SMEM cap
    /// (the tightest constraint even on an H100).
    ///
    /// Calculation with d_model=128, hd=64, bq=64, bkv=64, dtype=2:
    ///   pipeline = 8192+8192+8192+16384+512+16384 = 57856 bytes = 56 KB
    ///   block    = 57856 + 16384 + 32768 = 107008 bytes = 104 KB < 228 KB ✓
    ///
    /// On T4 (84 KB budget): block (104 KB) DOESN'T FIT, pipeline (56 KB) DOES
    /// → also the right target for the downgrade test.
    fn small_shape_block_feasible() -> LayerShape {
        LayerShape {
            batch: 1, seq: 1024, d_model: 128,
            head_dim: 64, n_kv_heads: 4, dtype_bytes: 2,
        }
    }

    /// Build a WggoOverrides with a single layer-0 record.
    fn make_override(active_heads: u32, requested: Option<WggoLevel>) -> WggoOverrides {
        WggoOverrides {
            per_layer: vec![PerLayerOverride {
                layer_index: 0,
                layer_name: "blocks.0".into(),
                active_heads,
                requested_csha_level: requested,
                adapter_rank: 0,
                adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                fase_fused: false,
                packing_mode: 0,
                shard_factor: 0,
            }],
        }
    }

    fn run_with(
        wengert: &WengertList,
        target: &str,
        shape: LayerShape,
        n_heads: u32,
        overrides: Option<&WggoOverrides>,
    ) -> CshaPlan {
        run_on_wengert(wengert, target, "full", None, Some(shape), n_heads, overrides)
            .expect("run_on_wengert must succeed for mode 'full'")
    }

    // -----------------------------------------------------------------------
    // Test 1: Override accepted when SMEM feasible
    // -----------------------------------------------------------------------
    #[test]
    fn wggo_override_applied_when_smem_feasible() {
        // H100 budget = min((256-12)*1024, 228*1024) = 228 KB = 233472 bytes.
        // Level-3 block fusion at d_model=128, hd=64, bq=64 needs 104 KB —
        // well within budget.  See `small_shape_block_feasible()` for the
        // derivation.
        let w = attn_wengert();
        let over = make_override(8, Some(WggoLevel::Level3));
        let plan = run_with(&w, "H100", small_shape_block_feasible(), 8, Some(&over));

        // The requested Level3 → Block must be honoured.
        assert_eq!(
            plan.per_layer[0].level,
            PipelineLevel::Block,
            "H100 with small shape: Level3/Block override must be applied verbatim (104 KB < 228 KB)"
        );
        assert!(
            plan.override_diagnostics.is_empty(),
            "no downgrade should be recorded when override is feasible"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2: Override downgraded when SMEM infeasible
    // -----------------------------------------------------------------------
    #[test]
    fn wggo_override_downgraded_when_smem_infeasible() {
        // T4 (sm_75): budget = (96 - 12) KB = 84 KB.
        // Level-3 block fusion at d128/hd64/bq64 needs 104 KB → infeasible.
        // Level-2 pipeline at same shape needs 56 KB → feasible.
        // Expected: Level3 downgraded to Level2 (Pipeline) with a diagnostic.
        let w = attn_wengert();
        let over = make_override(16, Some(WggoLevel::Level3));
        let plan = run_with(&w, "T4", small_shape_block_feasible(), 16, Some(&over));

        // The plan must NOT be Block — must be downgraded.
        assert_ne!(
            plan.per_layer[0].level,
            PipelineLevel::Block,
            "T4 SMEM is too tight for Level-3 block fusion at this shape: must downgrade"
        );

        // Exactly one diagnostic expected.
        assert_eq!(
            plan.override_diagnostics.len(), 1,
            "exactly one downgrade diagnostic must be emitted"
        );
        let diag = &plan.override_diagnostics[0];
        assert_eq!(diag.layer_index, 0);
        assert_eq!(diag.requested, "Level3");
        assert!(
            matches!(diag.reason, OverrideRejectReason::SmemBudgetExceeded { .. }),
            "reject reason must be SmemBudgetExceeded"
        );

        // Applied must be strictly less aggressive than requested.
        assert!(
            diag.applied != diag.requested,
            "applied level must differ from requested"
        );
        // applied ordinal < requested ordinal (Level3=3 is most aggressive).
        fn wggo_level_ordinal(level_str: &str) -> u8 {
            match level_str {
                "None" => 0,
                "Level1" => 1,
                "Level2" => 2,
                "Level3" => 3,
                _ => panic!("unknown level string: {level_str}"),
            }
        }
        assert!(
            wggo_level_ordinal(&diag.applied) < wggo_level_ordinal(&diag.requested),
            "applied ({}) must be strictly less aggressive than requested ({})",
            diag.applied, diag.requested
        );
    }

    // -----------------------------------------------------------------------
    // Test 3: No overrides → internal planner runs unchanged
    // -----------------------------------------------------------------------
    #[test]
    fn no_wggo_overrides_preserves_internal_planner_behavior() {
        let w = attn_wengert();
        let plan = run_with(&w, "H100", default_shape(), 8, None);

        // No override path touched: no diagnostics.
        assert!(
            plan.override_diagnostics.is_empty(),
            "no override diagnostics when no WggoOverrides supplied"
        );

        // Snapshot: H100 in Auto/full mode with d_model=512/hd=64 picks Block
        // (Level 3) after the B1.1 cost-model correction.
        //
        // Pre-B1.1 baseline:    Pipeline (Block was 232 KB, over the 228 KB cap)
        // Post-B1.1 baseline:   Block    (Block now fits at ~209 KB)
        //
        // B1.1 corrected three load-bearing bugs in `pipeline_smem_bytes` —
        // phantom O_acc, single-buffered K/V, and oversized w_tile — which
        // collectively shrink the Pipeline-level SMEM cost. `block_smem_bytes`
        // computes off that reduced base, so the post-correction Level-3
        // estimate is ~209 KB, inside the 228 KB cap.
        //
        // See docs/superpowers/specs/2026-05-11-tier-b1-v3-cost-model-audit.md.
        //
        // This value is pinned as the internal-planner baseline; if the planner
        // logic changes and produces a different level in the future, this test
        // will catch the regression — by design.
        assert_eq!(
            plan.per_layer[0].level,
            PipelineLevel::Block,
            "internal planner baseline (H100, Auto, d512/hd64) after B1.1: \
             expected Level3/Block (~209 KB fits in 228 KB cap)"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4: active_heads override prunes heads in specialization plan
    // -----------------------------------------------------------------------
    #[test]
    fn wggo_active_heads_override_prunes_heads() {
        // 8 heads total in the Wengert; WGGO says only 4 are active.
        // No weights supplied → fallback_heads marks all 8 as active.
        // The override patch must reduce n_active_heads to 4, yielding
        // total_pruned_heads() == 8 - 4 == 4.
        let w = attn_wengert();
        let over = make_override(4, None); // active_heads=4, no level override
        let plan = run_with(&w, "H100", default_shape(), 8, Some(&over));

        assert_eq!(
            plan.specialization.total_pruned_heads(), 4,
            "WGGO active_heads=4 of 8 must appear as 4 pruned heads in the specialization plan"
        );

        // Also: no level override → no diagnostics.
        assert!(
            plan.override_diagnostics.is_empty(),
            "active_heads-only override must not emit diagnostics"
        );
    }
}
