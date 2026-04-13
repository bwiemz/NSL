//! `nsl profile` — predictive performance profiling.
//!
//! Parses a source file, walks its typed ops through the cost model for a
//! target GPU, and renders either a formatted table or a machine-readable
//! JSON report. See `docs/superpowers/plans/2026-04-12-nsl-dev-tools-phase1.md`.

use std::path::PathBuf;

use nsl_codegen::cost_model::format_perf_table;
use nsl_codegen::gpu_specs::{find_gpu, GpuSpec, GPU_DATABASE};
use nsl_codegen::profiling::shape_env::ShapeEnv;
use nsl_codegen::profiling::types::{EntryKind, ProfileReport};
use nsl_codegen::profiling::walker::walk_ops;

#[derive(Debug, Clone)]
pub struct ProfileArgs {
    pub file: PathBuf,
    pub target: String,
    pub dtype: String,
    pub batch: u64,
    pub seq: u64,
    pub dim: Vec<String>,
    pub fusion: bool,
    pub memory: bool,
    pub entry: String,
    pub json: bool,
    /// Run WGGO `Full` mode and attach/append the decision explanation.
    pub explain_wggo: bool,
}

pub fn run_profile(args: &ProfileArgs) -> Result<String, String> {
    let gpu = find_gpu(&args.target).ok_or_else(|| {
        let available: Vec<&str> = GPU_DATABASE.iter().map(|g| g.name).collect();
        format!(
            "unknown GPU target: {}; available: {:?}",
            args.target, available
        )
    })?;

    let src = std::fs::read_to_string(&args.file).map_err(|e| {
        format!("could not read '{}': {}", args.file.display(), e)
    })?;
    let input = crate::shape_debug::ShapeDebugInput::from_source(
        &src,
        args.file.to_str().unwrap_or("<file>"),
    )?;

    let mut env = ShapeEnv::with_defaults();
    env.set("batch", args.batch);
    env.set("seq", args.seq);
    for f in &args.dim {
        env.parse_dim_flag(f)?;
    }

    let entry = EntryKind::parse_flag(&args.entry)
        .ok_or_else(|| "bad --entry value (use auto|train|fn:<name>)".to_string())?;

    let mut report = walk_ops(
        &input.module,
        &input.analysis,
        &input.interner,
        entry,
        &env,
        gpu,
        &args.dtype,
    )?;

    if args.fusion {
        report.fusion = Some(nsl_codegen::wrga_fusion::build_fusion_plan(&[], None));
    }

    if args.memory {
        // TODO: replace with real WengertList when source-AD integration is
        // stable. For Phase 1 we synthesize a minimal plausible plan from the
        // walker's op list: each op produces an activation of `bytes_written`
        // bytes, birthed at its program-point index and living for a small
        // sliding window (2 steps) to approximate activation reuse.
        use nsl_codegen::wrga_memory::{MemoryPlan, MemoryPlanStats, SlotAssignment};
        const LIFETIME_WINDOW: u32 = 2;
        let n = report.ops.len() as u32;
        let assignments: Vec<SlotAssignment> = report
            .ops
            .iter()
            .enumerate()
            .map(|(i, op)| {
                let birth = i as u32;
                let death = (birth + LIFETIME_WINDOW).min(n.max(1));
                SlotAssignment {
                    var: birth,
                    slot: birth,
                    size_bytes: op.bytes_written,
                    birth,
                    death,
                }
            })
            .collect();
        let plan = MemoryPlan {
            assignments,
            stats: MemoryPlanStats::default(),
        };
        let tl = nsl_codegen::profiling::memory_timeline::build(
            &nsl_codegen::profiling::memory_timeline::MemoryTimelineInput {
                plan: &plan,
                phase_markers: vec![],
            },
        );
        report.memory_timeline = Some(tl);
    }

    if args.explain_wggo {
        // Phase 3 Task 4: the CLI profile path doesn't yet carry a real
        // source-AD WengertList, so we synthesize a minimal two-block
        // attention-shaped list. This is sufficient for the decision-trace
        // renderer (Task 3) to produce a complete per-layer explanation.
        // When Phase-4 source-AD lands in `nsl profile`, replace with the
        // real wengert extraction.
        let wengert = synth_two_block_wengert();
        let plan = nsl_codegen::wggo::run_on_wengert(
            &wengert,
            &args.target,
            "full",
            1,
        )
        .ok_or_else(|| "WGGO mode 'full' rejected — internal error".to_string())?;
        if args.json {
            report.wggo_explain = Some(plan);
        } else {
            let explain = crate::wggo_explain::render_explain(&plan);
            // Defer to render_text below; stash via a helper string.
            let text = render_text(&report, gpu);
            let mut out = text;
            out.push('\n');
            out.push_str(&explain);
            return Ok(out);
        }
    }

    if args.json {
        return serde_json::to_string_pretty(&report).map_err(|e| e.to_string());
    }

    Ok(render_text(&report, gpu))
}

/// Synthesize a two-block attention Wengert list suitable for WGGO
/// driver input. Mirrors the test fixture used by Task 2's decision-trace
/// tests. Used only by `--explain-wggo` until real AD extraction is wired.
fn synth_two_block_wengert() -> nsl_codegen::wengert::WengertList {
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
    use std::collections::HashMap;
    let op = |id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>| WengertOp {
        id,
        result,
        op: o,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };
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

fn render_text(r: &ProfileReport, gpu: &GpuSpec) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== NSL Predictive Profile ===\nTarget: {} ({:.0} TFLOPS fp16, {:.2} TB/s HBM)\nDtype: {}\n\n",
        gpu.name,
        gpu.peak_fp16_tflops,
        gpu.peak_bandwidth_gbs / 1000.0,
        r.dtype
    ));
    out.push_str(&format_perf_table(&r.ops, gpu, &r.dtype));
    out.push_str(&format!(
        "\nTotals: {} FLOPs, {} HBM bytes, {:.2} μs estimated\n",
        r.total_flops, r.total_hbm_bytes, r.total_estimated_us
    ));
    if let Some(fp) = &r.fusion {
        out.push_str("\nFusion summary:\n");
        for d in &fp.decisions {
            out.push_str(&format!(
                "  {} → {:?}  ({} extra HBM bytes)  — {}\n",
                d.site, d.target, d.extra_hbm_bytes, d.rationale
            ));
        }
    }
    if let Some(tl) = &r.memory_timeline {
        out.push_str(&nsl_codegen::profiling::memory_timeline::render(tl));
    }
    if !r.recommendations.is_empty() {
        out.push_str("\nRecommendations:\n");
        for (i, rec) in r.recommendations.iter().enumerate() {
            out.push_str(&format!("  [{}] {} {}\n", i + 1, rec.code, rec.message));
        }
    }
    out
}
