//! CFIE Tier-A wiring (audit gaps G1-G6): end-to-end proof that a
//! serve block with CFIE config drives the orchestrator through the
//! REAL pipeline (lex -> parse -> semantic -> codegen), populating
//! `last_cfie_plan` with inputs resolved from the serve config + GPU
//! database — the exact reachability the CFIE audit found missing.
//!
//! Serve blocks live inside `fn main()` here because the debug entry
//! point compiles function bodies, not synthesized top-level mains
//! (same convention as `wrga_freeze_end_to_end.rs`).

use nsl_codegen::cfie::CfiePlan;
use nsl_codegen::CompileOptions;

const CFIE_SERVE_SRC: &str = r#"
fn main():
    serve Inference:
        max_batch: 64
        max_seq: 2048
        kv_layout: "static"
        kv_quant: "auto"
        target_gpu: "h100"
        n_layers: 8
        n_kv_heads: 4
        head_dim: 128
        d_model: 512
        d_ff: 1408
        vocab_size: 49152

        sampling:
            temperature: 0.7
            top_k: 50
            top_p: 0.9
            fused: true

        speculative:
            tokens: 5
            method: "tree"
            tree_width: 2

        @endpoint
        fn generate(prompt: str) -> str:
            let x = 0
"#;

const LEGACY_SERVE_SRC: &str = r#"
fn main():
    serve Legacy:
        max_batch: 32
        kv_blocks: 2048

        @endpoint
        fn handle(prompt: str) -> str:
            let x = 0
"#;

fn compile_cfie(src: &str, opts: &CompileOptions) -> Option<CfiePlan> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed.diagnostics.is_empty(),
        "parse must be clean: {:?}",
        parsed.diagnostics
    );
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "input must type-check: {:?}",
        analysis.diagnostics
    );
    nsl_codegen::debug_compile_and_return_cfie_plan_from_ast(
        &parsed.module,
        &interner,
        &analysis.type_map,
        opts,
    )
    .expect("compile must succeed")
}

#[test]
fn cfie_serve_block_populates_plan_from_real_inputs() {
    let plan = compile_cfie(CFIE_SERVE_SRC, &CompileOptions::default())
        .expect("kv_layout key must activate CFIE");

    // Mode: implicit activation via serve keys => Full.
    assert_eq!(plan.mode, nsl_codegen::cfie::CfieMode::Full);

    // GPU resolved from the serve-block `target_gpu` key via gpu_specs.
    assert_eq!(plan.target_gpu, "H100-SXM");

    // The paper's worked example: 8 layers x 4 KV heads x 128 head_dim
    // x 2 bytes -> 16 KiB/token; H100's 80 GB fits the static layout.
    assert!(
        plan.kv.uses_direct_indexing(),
        "static layout must fit on H100: {}",
        plan.kv.rationale
    );

    // Fused sampler was configured from the sampling: section.
    assert!(plan.sampling.is_fused());
    assert_eq!(plan.sampling.params.top_k, 50);

    // Speculative section reached the plan.
    let spec = plan.speculative.as_ref().expect("speculative plan");
    assert_eq!(spec.config.k_tokens, 5);
    assert_eq!(spec.config.tree_width, 2);

    // Per-layer KV quant ran (kv_quant: "auto").
    assert_eq!(plan.kv_quant.layers.len(), 8);

    // Launch accounting is populated.
    assert!(plan.kernel_launches_per_token_cfie < plan.kernel_launches_per_token_baseline);

    // The report renders with the real GPU + all six sections.
    let report = plan.render_report();
    assert!(report.contains("CFIE Inference Build Report"));
    assert!(report.contains("H100-SXM"));
    assert!(report.contains("[6] Grammar DFA: disabled"));
}

#[test]
fn legacy_serve_block_does_not_activate_cfie() {
    let plan = compile_cfie(LEGACY_SERVE_SRC, &CompileOptions::default());
    assert!(
        plan.is_none(),
        "a serve block without CFIE keys must leave the M29 path untouched"
    );
}

#[test]
fn cli_mode_override_forces_cfie_off() {
    let opts = CompileOptions {
        cfie: nsl_codegen::CfieOptions {
            mode_override: Some("off".to_string()),
            report_path: None,
        },
        ..Default::default()
    };
    let plan = compile_cfie(CFIE_SERVE_SRC, &opts);
    assert!(plan.is_none(), "--cfie off must disable CFIE entirely");
}

#[test]
fn cli_mode_override_forces_cfie_on_for_legacy_serve() {
    let opts = CompileOptions {
        cfie: nsl_codegen::CfieOptions {
            mode_override: Some("full".to_string()),
            report_path: None,
        },
        ..Default::default()
    };
    let plan = compile_cfie(LEGACY_SERVE_SRC, &opts)
        .expect("--cfie full must activate CFIE even without serve keys");
    assert_eq!(plan.mode, nsl_codegen::cfie::CfieMode::Full);
}

#[test]
fn cfie_decorator_on_serve_block_activates_and_targets() {
    let src = r#"
fn main():
    @cfie(mode=full, target=a100)
    serve Decorated:
        max_batch: 16

        @endpoint
        fn handle(prompt: str) -> str:
            let x = 0
"#;
    let plan = compile_cfie(src, &CompileOptions::default())
        .expect("@cfie decorator must activate CFIE");
    assert_eq!(
        plan.target_gpu, "A100-SXM",
        "@cfie(target=a100) must drive GPU resolution"
    );
}

#[test]
fn grammar_section_fails_compile_with_g12_refusal() {
    let src = r#"
fn main():
    serve WithGrammar:
        kv_layout: "static"

        grammar:
            schema: "output_schema.json"

        @endpoint
        fn handle(prompt: str) -> str:
            let x = 0
"#;
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    let res = nsl_codegen::debug_compile_and_return_cfie_plan_from_ast(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &CompileOptions::default(),
    );
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("grammar: section must refuse until G12 lands"),
    };
    assert!(
        err.message.contains("G12"),
        "refusal must cite the audit gap: {}",
        err.message
    );
}

#[test]
fn report_file_is_written_when_requested() {
    let dir = std::env::temp_dir().join("nsl_cfie_report_test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("cfie_report.txt");
    let _ = std::fs::remove_file(&path);

    let opts = CompileOptions {
        cfie: nsl_codegen::CfieOptions {
            mode_override: None,
            report_path: Some(path.clone()),
        },
        ..Default::default()
    };
    let _plan = compile_cfie(CFIE_SERVE_SRC, &opts).expect("CFIE active");
    let contents = std::fs::read_to_string(&path).expect("report file must exist");
    assert!(contents.contains("CFIE Inference Build Report"));
    assert!(
        contents.contains("Model-shape provenance"),
        "report must attribute input sources: {contents}"
    );
    let _ = std::fs::remove_file(&path);
}
