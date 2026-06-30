//! M55: `nsl build --zk-circuit` and the `nsl zk` subcommands (stats / prove /
//! verify) for the folding ZK backend.
//!
//! Extracted verbatim from the former monolithic `build.rs`; behavior is
//! unchanged.

use std::path::PathBuf;
use std::process;

use super::reports::{check_wrga_report_preconditions, emit_wrga_report};

/// M55: Build with --zk-circuit. Runs normal compilation and then invokes
/// `zk::compile_zk()` for each @zk_proof-decorated function found.
pub(crate) fn run_build_zk(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    zk_weights: Option<&std::path::Path>,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so `@freeze`/`@adapter`/`@wrga`
    // take effect in codegen on the ZK build path.
    // Task 4 (B.2): if `--wrga-report` is set, fail fast when decorators are
    // present without `--source-ad` (mirroring the single/multi build paths).
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs = Some(crate::pipeline::analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    options.pca_user_strategies = crate::pipeline::analysis_to_pca_user_strategies(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

    // Task 4 (B.2): use the `_returning_plan` variant so the WRGA plan is
    // observable (and reportable) even if later codegen fails.
    let (bytes_res, zk_proof_fns, zk_results, wrga_plan) =
        nsl_codegen::compile_with_zk_info_returning_plan(
            &parse_result.module,
            &interner,
            &analysis.type_map,
            dump_ir,
            options,
        );

    // Emit the WRGA report (if requested) before reporting any codegen error.
    emit_wrga_report(&wrga_plan, wrga_report);

    let obj_bytes = match bytes_res {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    // M55c: Write ZK proof files alongside the binary
    if zk_proof_fns.is_empty() {
        eprintln!("[nsl/zk] no @zk_proof functions found — ZK circuit output skipped");
    } else {
        eprintln!(
            "[nsl/zk] found {} @zk_proof function(s):",
            zk_proof_fns.len()
        );

        if let Some(wpath) = zk_weights {
            eprintln!("[nsl/zk]   weights: {}", wpath.display());
        }

        for (fn_name, result) in &zk_results {
            let report = nsl_codegen::zk::stats::format_stats(&result.stats, fn_name);
            eprint!("{}", report);

            // Write proof file if proof was generated
            if let Some(ref proof) = result.proof {
                let proof_path = file.with_extension(format!("{}.proof", fn_name));
                if let Err(e) = std::fs::write(&proof_path, &proof.data) {
                    eprintln!("[nsl/zk] error writing proof: {e}");
                } else {
                    eprintln!("[nsl/zk]   proof: {} ({} bytes, {} folds)",
                        proof_path.display(), proof.data.len(), proof.num_folds);
                }

                // Write public inputs as JSON
                let pi_path = file.with_extension(format!("{}.public.json", fn_name));
                let pi_entries: Vec<String> = proof.public_inputs.iter()
                    .map(|v| format!("{:?}", v))
                    .collect();
                let pi_json = format!(
                    "{{\"public_inputs\":[{}],\"num_folds\":{}}}",
                    pi_entries.join(","), proof.num_folds
                );
                if let Err(e) = std::fs::write(&pi_path, &pi_json) {
                    eprintln!("[nsl/zk] error writing public inputs: {e}");
                } else {
                    eprintln!("[nsl/zk]   public inputs: {}", pi_path.display());
                }
            }
        }
    }

    // Write normal object / link as usual.
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("error: invalid input filename '{}'", file.display());
            process::exit(1);
        });
    let obj_path = file.with_file_name(format!("{stem}.o"));

    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    if emit_obj {
        println!("Wrote {}", obj_path.display());
        return;
    }

    let exe_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    match nsl_codegen::linker::link(&obj_path, &exe_path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
            println!("Built {}", exe_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

/// M55c: Handle `nsl zk <subcommand>`.
pub(crate) fn run_zk_cmd(cmd: crate::args::ZkCmd) {
    match cmd {
        crate::args::ZkCmd::Stats { file } => {
            match std::fs::read(&file) {
                Ok(data) => {
                    if data.len() < 12 {
                        eprintln!("[nsl/zk] invalid proof file: too short ({} bytes)", data.len());
                        process::exit(1);
                    }
                    let num_folds = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let instance_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                    let num_rounds = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                    println!("Proof stats:");
                    println!("  File:        {}", file.display());
                    println!("  Size:        {} bytes", data.len());
                    println!("  Folds:       {}", num_folds);
                    println!("  Instance:    {} elements", instance_len);
                    println!("  SC rounds:   {}", num_rounds);
                }
                Err(e) => {
                    eprintln!("error reading {}: {e}", file.display());
                    process::exit(1);
                }
            }
        }
        crate::args::ZkCmd::Prove { file, pk: _, input: _, output } => {
            // For the folding backend, proofs are generated during compilation.
            eprintln!("[nsl/zk] For the folding backend, proofs are generated during `nsl build --zk-circuit`.");
            eprintln!("[nsl/zk] The proof file is written alongside the binary as <file>.<fn_name>.proof");
            if let Some(ref out) = output {
                eprintln!("[nsl/zk] Requested output: {}", out.display());
            }
            eprintln!("[nsl/zk] To generate a proof, run: nsl build --zk-circuit {}", file.display());
        }
        crate::args::ZkCmd::Verify { vk: _, proof, public: _ } => {
            match std::fs::read(&proof) {
                Ok(data) => {
                    if data.len() < 12 {
                        eprintln!("INVALID: proof file too short ({} bytes)", data.len());
                        process::exit(1);
                    }

                    let num_folds = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let instance_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
                    let num_rounds = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

                    // Reconstruct ZkProof for verification
                    let zk_proof = nsl_codegen::zk::backend::ZkProof {
                        data: data.clone(),
                        num_folds,
                        public_inputs: vec![vec![0u8; 4]; instance_len],
                        public_outputs: Vec::new(),
                    };

                    use nsl_codegen::zk::backend::FoldingBackend;
                    type M31Prover = nsl_codegen::zk::folding::FoldingProver<nsl_codegen::zk::field_m31::Mersenne31Field>;

                    match M31Prover::verify(&zk_proof, &[]) {
                        Ok(true) => {
                            println!("VERIFIED: proof is valid ({} folds, {} sumcheck rounds)",
                                num_folds, num_rounds);
                        }
                        Ok(false) => {
                            println!("INVALID: proof verification failed");
                            process::exit(1);
                        }
                        Err(e) => {
                            eprintln!("VERIFICATION ERROR: {e}");
                            process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("error reading {}: {e}", proof.display());
                    process::exit(1);
                }
            }
        }
    }
}
