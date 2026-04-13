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
use crate::csha_pipeline::{plan_all, FusionLevel, LayerPlan};
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
}

/// Aggregate plan emitted by the driver.
#[derive(Debug, Clone, Serialize)]
pub struct CshaPlan {
    pub mode: CshaMode,
    pub target_gpu: String,
    pub boundary: BoundaryScan,
    pub per_layer: Vec<LayerPlan>,
    pub specialization: SpecializationPlan,
    /// Per-layer kernel-specialisation artefacts produced by the
    /// `csha_apply` bridge.  Empty in `Off` mode.
    pub kernels: Vec<crate::csha_apply::KernelSpec>,
    /// Fusion-graph marks that claim Q/K/V matmul nodes on behalf of
    /// CSHA-fused kernels, so `epilogue_fusion` and `reduction_fusion`
    /// do not double-fuse them.
    pub marks: Vec<crate::csha_apply::FusionMark>,
    /// Total driver wall-clock time (microseconds).
    pub solve_us: u64,
}

impl CshaPlan {
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
        }

        writeln!(s).unwrap();
        writeln!(
            s,
            "Solve time: {:.2} ms",
            self.solve_us as f64 / 1000.0
        )
        .unwrap();
        s
    }
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
            kernels: Vec::new(),
            marks: Vec::new(),
            solve_us: t0.elapsed().as_micros() as u64,
        };
    }

    let boundary = scan_boundaries(input.wengert);
    let per_layer = plan_all(&boundary, input.shape, gpu, input.mode.initial_level());
    let specialization = analyze_spec(
        &boundary,
        input.weights,
        input.n_heads,
        &input.spec_cfg,
    );

    // Stitch everything together via the apply-bridge so the plan
    // carries ready-to-consume kernel specs + graph marks.
    let interim = CshaPlan {
        mode: input.mode,
        target_gpu: gpu.name.to_string(),
        boundary: boundary.clone(),
        per_layer: per_layer.clone(),
        specialization: specialization.clone(),
        kernels: Vec::new(),
        marks: Vec::new(),
        solve_us: 0,
    };
    let bridge = crate::csha_apply::bridge(&interim, input.shape.head_dim as i64);

    CshaPlan {
        mode: input.mode,
        target_gpu: gpu.name.to_string(),
        boundary,
        per_layer,
        specialization,
        kernels: bridge.kernels,
        marks: bridge.marks,
        solve_us: t0.elapsed().as_micros() as u64,
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
                run_on_wengert(&w, "H100", mode, None, None, 8).is_some(),
                "'{}' should parse",
                mode
            );
        }
        assert!(run_on_wengert(&w, "H100", "wat", None, None, 8).is_none());
    }

    #[test]
    fn unknown_target_falls_back_to_default_gpu() {
        let w = attn_block();
        let plan = run_on_wengert(&w, "nonexistent-gpu-xyz", "auto", None, None, 8).unwrap();
        assert!(!plan.target_gpu.is_empty());
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
}
