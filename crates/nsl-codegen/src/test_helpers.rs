//! Test-only helpers for the profiling pre-pass and codegen wiring
//! (Dev Tools Phase 2).
//!
//! Gated behind `cfg(any(test, feature = "test-helpers"))` so the module is
//! only compiled during tests or when a downstream crate explicitly opts in.
#![cfg(any(test, feature = "test-helpers"))]

use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

/// Parsing output kept alive for the duration of a pre-pass test.
///
/// The returned `Compiler` stores `&Interner` / `&TypeMap` references into
/// this struct, so callers must keep the bundle alive for as long as they
/// use the compiler.
pub struct PrePassBundle {
    pub interner: Interner,
    pub type_map: TypeMap,
}

/// Parse + semantic-analyze `src`, then run the Phase 2 kernel-profile
/// pre-pass (and *only* the pre-pass) against a fresh `Compiler`.  Returns
/// the bundle plus snapshots of the three pre-pass outputs:
///
/// * whether `manifest_builder` was populated
/// * whether `fusion_plan_for_profile` was populated
/// * the `prediction_map` length
///
/// This avoids handing out a `Compiler<'a>` whose lifetime would collide
/// with the temporary `Interner`/`TypeMap` at call sites.
pub fn run_pre_pass_only(
    src: &str,
    opts: &crate::CompileOptions,
) -> Result<PrePassResult, String> {
    use nsl_errors::{FileId, Level};

    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, Level::Error)) {
        return Err(format!(
            "lex errors: {:?}",
            lex_diags
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        ));
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        return Err(format!(
            "parse errors: {:?}",
            parsed
                .diagnostics
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        ));
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);

    // Build a Compiler and run only the pre-pass logic (mirrors the body of
    // `entry_points::run_profile_pre_pass`, which is private).
    let type_map = analysis.type_map;
    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, opts)
        .map_err(|e| format!("Compiler::new failed: {}", e.message))?;

    let mut prediction_map_len = 0usize;
    let mut manifest_builder_set = false;
    let mut fusion_plan_set = false;
    let mut source_text_out = String::new();

    if opts.profile_kernels {
        use crate::gpu_specs::find_gpu;
        use crate::profiling::instrument::ManifestBuilder;
        use crate::profiling::shape_env::ShapeEnv;
        use crate::profiling::types::EntryKind;
        use crate::profiling::walker::walk_ops;

        // Phase 2.5 Task 4: install source-text/name up front, mirroring
        // `run_profile_pre_pass`.  Runs independently of walker success.
        compiler.source_text = match &opts.profile_source_text {
            Some(s) => s.clone(),
            None => opts
                .profile_source_file_name
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .unwrap_or_default(),
        };
        compiler.source_file_name = opts
            .profile_source_file_name
            .clone()
            .unwrap_or_default();
        source_text_out = compiler.source_text.clone();

        let env = ShapeEnv::with_defaults();
        let target_gpu = opts.target_gpu.as_str();
        let dtype = opts.dtype.as_str();
        let gpu = find_gpu(target_gpu)
            .ok_or_else(|| format!("unknown GPU target: {}", target_gpu))?;

        let synth_analysis = nsl_semantic::AnalysisResult {
            diagnostics: Vec::new(),
            type_map: type_map.clone(),
            scopes: nsl_semantic::scope::ScopeMap::new(),
            ownership_info: std::collections::HashMap::new(),
            wrga_configs: Vec::new(),
            freeze_configs: Vec::new(),
            adapter_configs: Vec::new(),
        };
        // walk_ops may fail on trivial test inputs (no fn/train block).  The
        // production path is non-fatal; mirror that so the source-text
        // fallback can be exercised independently of walker success.
        if let Ok(report) = walk_ops(
            &parsed.module,
            &synth_analysis,
            &interner,
            EntryKind::Auto,
            &env,
            gpu,
            dtype,
        ) {
            compiler.prediction_map = report
                .ops
                .iter()
                .filter_map(|op| op.origin_node.map(|nid| (nid, op.clone())))
                .collect();
        }
        compiler.manifest_builder = Some(ManifestBuilder::new(target_gpu, dtype));
        // Phase 2.5 Task 3: mirror `run_profile_pre_pass` seeding so
        // `fusion_plan_for_profile` is `Some(...)` after the pre-pass.
        let seeded = compiler
            .last_wrga_plan
            .as_ref()
            .map(|p| p.fusion.clone())
            .unwrap_or_default();
        compiler.fusion_plan_for_profile = Some(seeded);

        prediction_map_len = compiler.prediction_map.len();
        manifest_builder_set = compiler.manifest_builder.is_some();
        fusion_plan_set = compiler.fusion_plan_for_profile.is_some();
    }

    Ok(PrePassResult {
        prediction_map_len,
        manifest_builder_set,
        fusion_plan_set,
        source_text: source_text_out,
    })
}

/// Summary of the pre-pass outputs, safe to return across the `Compiler`
/// lifetime boundary.
#[derive(Debug, Clone)]
pub struct PrePassResult {
    pub prediction_map_len: usize,
    pub manifest_builder_set: bool,
    pub fusion_plan_set: bool,
    /// Phase 2.5 Task 4: the `source_text` the pre-pass installed on the
    /// `Compiler` (explicit `profile_source_text`, else disk read from
    /// `profile_source_file_name`, else empty).
    pub source_text: String,
}
