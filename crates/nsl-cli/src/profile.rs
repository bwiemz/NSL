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
    /// Write a self-contained HTML report (inline-SVG roofline) here.
    pub html: Option<PathBuf>,
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

    // Dev-tools paper completion: run a real capture compile when a real
    // WengertList materially improves the requested output (--explain-wggo
    // explains the USER'S model instead of a synthetic one; --memory gets a
    // real-liveness timeline with what-if lines). The capture survives
    // downstream codegen errors, so a minimal single-file profile compile
    // that fails late (e.g. on optimizer stdlib symbols) still yields the
    // extracted list. Programs without a source-AD train block fall back to
    // the labeled synthetic/approximate paths below.
    let captures = if args.explain_wggo || args.memory {
        try_capture_real(&input, &args.target)
    } else {
        None
    };
    let real_wengert = captures
        .as_ref()
        .and_then(|c| c.train_wengert.clone())
        .filter(|w| !w.ops.is_empty());

    if args.fusion {
        // Prefer the fusion plan the capture compile actually produced;
        // fall back to the (empty) placeholder plan otherwise.
        report.fusion = Some(
            captures
                .as_ref()
                .and_then(|c| c.fusion.clone())
                .unwrap_or_else(|| nsl_codegen::wrga_fusion::build_fusion_plan(&[], None)),
        );
    }

    // Pre-rendered real-liveness timeline (render_text uses it in place of
    // the approximate renderer when the real path succeeded).
    let mut real_timeline_text: Option<String> = None;
    if args.memory {
        if let (Some(c), Some(w)) = (&captures, &real_wengert) {
            if let Some(rt) =
                nsl_codegen::profiling::real_timeline::build_training_timeline(
                    w,
                    &c.var_size_hints,
                )
            {
                report.memory_timeline = Some(rt.entries.clone());
                report.memory_timeline_approximate = Some(false);
                report.memory_what_if = Some(rt.what_if.clone());
                report.memory_peak_bytes = Some(rt.peak_bytes);
                report.memory_unsized_vars = Some(rt.unsized_vars);
                report.memory_total_vars = Some(rt.sized_vars + rt.unsized_vars);
                real_timeline_text =
                    Some(nsl_codegen::profiling::real_timeline::render(&rt, 48));
            }
        }
    }

    if args.memory && real_timeline_text.is_none() {
        // APPROXIMATION fallback, and labeled as such in both the render and
        // the JSON (`memory_timeline_approximate: true`): each op's
        // activation is given a fixed 2-step lifetime, so the timeline shows
        // a moving sum of bytes_written — NOT real activation liveness. Used
        // when the program has no source-AD train block to extract (the real
        // path above handles the training case).
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
        report.memory_timeline_approximate = Some(true);
    }

    // Dev-tools paper completion (paper section 3.1 "Roofline Plot:
    // [generated as SVG/HTML in build output]"): write the self-contained
    // HTML report when requested. The report is complete at this point for
    // HTML purposes (ops, fusion, timeline, recommendations).
    if let Some(html_path) = &args.html {
        let html = crate::profile_render::render_html(&report, gpu);
        std::fs::write(html_path, html).map_err(|e| {
            format!("could not write HTML report '{}': {}", html_path.display(), e)
        })?;
        eprintln!("[nsl] HTML profile report written to {}", html_path.display());
    }

    if args.explain_wggo {
        // Dev-tools paper completion: explain the USER'S model when a real
        // source-AD WengertList was captured; the synthetic two-block
        // attention list remains only as the labeled fallback for programs
        // without a train block.
        let used_real = real_wengert.is_some();
        let wengert = match &real_wengert {
            Some(w) => w.clone(),
            None => synth_two_block_wengert(),
        };
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
            let text = render_text(&report, gpu, real_timeline_text.as_deref());
            let mut out = text;
            out.push('\n');
            if !used_real {
                out.push_str(
                    "note: no source-AD train block captured from this program — the \
                     explanation below covers a SYNTHETIC two-block attention model.\n",
                );
            }
            out.push_str(&explain);
            return Ok(out);
        }
    }

    if args.json {
        return serde_json::to_string_pretty(&report).map_err(|e| e.to_string());
    }

    Ok(render_text(&report, gpu, real_timeline_text.as_deref()))
}

/// Attempt the real capture compile for `nsl profile`'s real path. Returns
/// `None` (with a stderr note) when the module has no source-AD train block
/// or the pipeline errored before extraction.
///
/// The CompileOptions are enriched with the SAME decorator bridges the real
/// build paths apply (standalone.rs / shared_lib.rs / zk.rs): without them a
/// `@checkpoint`/`@lora`/`@wrga`/`@csha` program would extract a list the
/// production compile never lowers (unpruned, non-checkpoint-aware), and we
/// would then label an inaccurate timeline "real". Populating `wrga_inputs`
/// also makes the captured `fusion` plan non-empty (the profile fusion
/// summary reflects the real WRGA plan instead of an empty placeholder).
fn try_capture_real(
    input: &crate::shape_debug::ShapeDebugInput,
    target: &str,
) -> Option<nsl_codegen::profiling::captures::ProfileCaptures> {
    let analysis = &input.analysis;
    let mut options = nsl_codegen::CompileOptions {
        source_ad: true,
        target: target.to_string(),
        ..Default::default()
    };
    options.wrga_inputs = Some(crate::analysis_bridges::analysis_to_wrga_inputs(
        analysis,
        &options.wrga_check,
    ));
    options.fused_ce_configs = crate::analysis_bridges::analysis_to_fused_ce_configs(analysis);
    options.fused_kl_ce_configs = crate::analysis_bridges::analysis_to_fused_kl_ce_configs(analysis);
    options.pca_user_strategies = crate::analysis_bridges::analysis_to_pca_user_strategies(analysis);
    options.csha_configs = crate::analysis_bridges::analysis_to_csha_configs(analysis);
    options.checkpoint_policies = crate::analysis_bridges::analysis_to_checkpoint_policies(analysis);
    options.weight_index_map = analysis.weight_index_map.clone();
    let (captures, result) = nsl_codegen::compile_with_profile_captures(
        &input.module,
        &input.interner,
        &input.analysis.type_map,
        &options,
    );
    if captures.is_none() {
        match &result {
            Err(e) => eprintln!(
                "note: real train-block extraction unavailable ({}); using the \
                 labeled synthetic/approximate profile paths",
                e.message
            ),
            Ok(_) => eprintln!(
                "note: no source-AD train block in this program; using the \
                 labeled synthetic/approximate profile paths"
            ),
        }
    }
    captures
}

/// Synthesize a two-block attention Wengert list suitable for WGGO
/// driver input. Mirrors the test fixture used by Task 2's decision-trace
/// tests. FALLBACK ONLY: `--explain-wggo` uses the real captured train-block
/// WengertList when the program has one (and says so in the output when it
/// does not).
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

fn render_text(r: &ProfileReport, gpu: &GpuSpec, real_timeline_text: Option<&str>) -> String {
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
        if fp.decisions.is_empty() {
            // Refuse-loudly rather than print a bare header: the fusion plan
            // is only populated from a real WRGA compile, which runs on the
            // capture path (--memory / --explain-wggo). Say so instead of
            // implying "no fusion opportunities exist".
            out.push_str(
                "  (no fusion decisions captured — the fusion plan is populated only\n  \
                 from a real compile; re-run with --memory or --explain-wggo, or with\n  \
                 @wrga/@adapter decorators present, to see WRGA fusion decisions)\n",
            );
        }
        for d in &fp.decisions {
            out.push_str(&format!(
                "  {} → {:?}  ({} extra HBM bytes)  — {}\n",
                d.site, d.target, d.extra_hbm_bytes, d.rationale
            ));
        }
    }
    if let Some(rt_text) = real_timeline_text {
        // Real-liveness path: the pre-rendered timeline already carries the
        // phase markers, peak, what-if lines, and honesty notes.
        out.push_str(rt_text);
    } else if let Some(tl) = &r.memory_timeline {
        out.push_str(&nsl_codegen::profiling::memory_timeline::render(tl));
        if r.memory_timeline_approximate == Some(true) {
            out.push_str(
                "\n  NOTE: APPROXIMATE timeline — synthesized from a fixed 2-step\n  \
                 activation-lifetime heuristic, not real liveness analysis. For\n  \
                 training programs the true peak (saved-for-backward activations)\n  \
                 is typically much higher than shown.\n",
            );
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
