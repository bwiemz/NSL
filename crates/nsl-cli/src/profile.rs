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

    // TODO: memory timeline — Task 5 will populate `report.memory_timeline`
    // and render it here when `args.memory` is true.
    let _ = args.memory;

    if args.json {
        return serde_json::to_string_pretty(&report).map_err(|e| e.to_string());
    }

    Ok(render_text(&report, gpu))
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
    if !r.recommendations.is_empty() {
        out.push_str("\nRecommendations:\n");
        for (i, rec) in r.recommendations.iter().enumerate() {
            out.push_str(&format!("  [{}] {} {}\n", i + 1, rec.code, rec.message));
        }
    }
    out
}
