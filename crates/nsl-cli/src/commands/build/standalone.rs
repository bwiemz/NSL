//! M52: `nsl build --standalone` — embed or sidecar safetensors weights into a
//! self-contained executable.
//!
//! Extracted verbatim from the former monolithic `build.rs`; behavior is
//! unchanged.

use std::path::PathBuf;
use std::process;

use super::reports::{check_wrga_report_preconditions, emit_wrga_report};

pub(crate) fn run_build_standalone(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    weights: &std::path::Path,
    embed_mode: crate::standalone::EmbedMode,
    embed_threshold: u64,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    // 1. Read weights from safetensors
    let tensors = crate::standalone::read_safetensors(weights).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        process::exit(1);
    });

    // 2. Serialize to .nslweights format
    let nslweights_data = crate::standalone::serialize_nslweights(&tensors);

    // 3. Decide embed vs sidecar
    let embedded = match embed_mode {
        crate::standalone::EmbedMode::Always => true,
        crate::standalone::EmbedMode::Never => false,
        crate::standalone::EmbedMode::Auto => (nslweights_data.len() as u64) <= embed_threshold,
    };

    // 4. Run frontend (lex, parse, semantic analysis)
    let file_pb = file.to_path_buf();
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(&file_pb, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so `@freeze`/`@adapter`/`@wrga`
    // take effect in codegen on the standalone build path.
    // Task 4 (B.2): if `--wrga-report` is set, fail fast when decorators are
    // present without `--source-ad`.
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs =
        Some(crate::pipeline::analysis_to_wrga_inputs(&analysis, &options.wrga_check));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    options.pca_user_strategies = crate::pipeline::analysis_to_pca_user_strategies(&analysis);
    // Sprint 2 (paper §6.2): forward @csha decorator configs so per-model
    // disable/level/target overrides take effect on the standalone build path.
    options.csha_configs = crate::pipeline::analysis_to_csha_configs(&analysis);
    // Cycle-10 §5.3 Task 6: route @checkpoint(policy=...) policies from
    // EffectChecker into CompileOptions so WengertExtractor::with_checkpoint_policies
    // can stamp the prologue + emit a PrologueRecompute marker.
    options.checkpoint_policies = crate::pipeline::analysis_to_checkpoint_policies(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

    // 5. Determine output path
    let output_path = if let Some(out) = output {
        out.to_path_buf()
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    let sidecar_path = output_path.with_extension("nslweights");

    // Pass only the filename (not full path) to codegen — the runtime resolves
    // the sidecar relative to the executable, so absolute paths would break
    // portability when the binary is moved.
    let sidecar_name = sidecar_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "model.nslweights".to_string());

    let config = nsl_codegen::StandaloneConfig {
        embedded,
        sidecar_path: sidecar_name,
    };

    // 6. Compile with standalone config.
    // Task 4 (B.2): use the `_returning_plan` variant so the WRGA plan is
    // observable (and reportable) even if later codegen fails.
    let (bytes_res, wrga_plan) = nsl_codegen::compile_standalone_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        config,
        false,
        options,
    );
    emit_wrga_report(&wrga_plan, wrga_report, &options.wrga_check);
    let obj_bytes = bytes_res.unwrap_or_else(|e| {
        eprintln!("codegen error: {e}");
        process::exit(1);
    });

    // 7. Write main object file
    let temp_dir = std::env::temp_dir().join(format!("nsl_standalone_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let main_obj_path = temp_dir.join("main.o");
    if let Err(e) = std::fs::write(&main_obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    let mut obj_paths: Vec<PathBuf> = vec![main_obj_path];

    // 8. Handle embedded weights or sidecar
    if embedded {
        // Create weight object containing the nslweights data
        let weight_obj_bytes = nsl_codegen::create_weight_object(&nslweights_data).unwrap_or_else(|e| {
            eprintln!("error: could not create weight object: {e}");
            process::exit(1);
        });
        let weight_obj_path = temp_dir.join("weights.o");
        if let Err(e) = std::fs::write(&weight_obj_path, &weight_obj_bytes) {
            eprintln!("error: could not write weight object file: {e}");
            process::exit(1);
        }
        obj_paths.push(weight_obj_path);
    } else {
        // Write sidecar .nslweights file
        crate::standalone::write_nslweights_sidecar_raw(&nslweights_data, &sidecar_path).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            process::exit(1);
        });
    }

    // 9. Link all objects
    match nsl_codegen::linker::link_multi(&obj_paths, &output_path) {
        Ok(()) => {
            // 10. Clean up temp object files
            for obj in &obj_paths {
                let _ = std::fs::remove_file(obj);
            }
            let _ = std::fs::remove_dir(&temp_dir);

            println!("Built {} (standalone{})", output_path.display(),
                if embedded { ", weights embedded" } else { ", sidecar weights" });
            if !embedded {
                println!("  Sidecar: {}", sidecar_path.display());
            }
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}
