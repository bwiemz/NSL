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

    // Feature 1 / G7: the static layout must come with an emitted
    // direct-indexing decode-attention kernel (real PTX, baked strides).
    let kernel = plan
        .decode_attention_kernel
        .as_deref()
        .expect("static layout must emit the decode-attention kernel");
    assert_eq!(kernel, "nsl_cfie_decode_attn");
    let ptx = plan
        .decode_attention_ptx
        .as_deref()
        .expect("PTX must be stored on the plan");
    assert!(ptx.contains(".visible .entry nsl_cfie_decode_attn"));
    assert!(
        !ptx.contains("block_table"),
        "direct indexing must not reference a block table"
    );

    // Feature 4 / G16: the paper config (d_model 512, head_dim 128 ->
    // 4 derived heads, 4 KV heads, d_ff 1408) fits Level-3 block
    // fusion on H100 (256 KB SMEM/SM vs ~8 KB modelled residency), so
    // plan() selects level3 and the serve wiring must emit the
    // persistent decode-block kernel alongside the plan.
    assert_eq!(plan.persistent.fusion.as_str(), "level3");
    let blk = plan
        .decode_block_kernel
        .as_deref()
        .expect("level3 + static layout must emit the decode-block kernel");
    assert_eq!(blk, "nsl_cfie_decode_block");
    let blk_ptx = plan
        .decode_block_ptx
        .as_deref()
        .expect("decode-block PTX must be stored on the plan");
    assert!(blk_ptx.contains(".visible .entry nsl_cfie_decode_block"));
    // Same baked KV layout as the decode-attention kernel.
    assert!(blk_ptx.contains("//   token_stride        = 512"));

    // Feature 3 / G13+G14: the tree draft (tokens=5, width=2) yields a
    // BFS tree of 1+2+4+8+16 = 31 nodes; both verification kernels are
    // emitted against the same static KV pool.
    let tree = spec.tree_mask.as_ref().expect("tree method builds a mask");
    assert_eq!(tree.num_nodes, 31);
    let vk = plan
        .spec_verify_kernel
        .as_deref()
        .expect("tree speculative + static layout must emit the verify kernel");
    assert_eq!(vk, "nsl_cfie_spec_verify_attn");
    let vptx = plan
        .spec_verify_ptx
        .as_deref()
        .expect("verify PTX must be stored on the plan");
    assert!(vptx.contains(".visible .entry nsl_cfie_spec_verify_attn"));
    // One baked u64 mask immediate per node row — no mask parameter.
    assert_eq!(vptx.matches("mov.u64 %rd_mask, 0x").count(), 31);
    assert!(!vptx.contains("mask_ptr"));
    // Same baked KV layout as the decode kernels.
    assert!(vptx.contains("//   token_stride        = 512"));
    let rk = plan
        .spec_reject_kernel
        .as_deref()
        .expect("speculative must emit the rejection epilogue");
    assert_eq!(rk, "nsl_cfie_spec_reject");
    assert!(plan
        .spec_reject_ptx
        .as_deref()
        .expect("reject PTX must be stored on the plan")
        .contains(".visible .entry nsl_cfie_spec_reject"));

    // Feature 5 / G18: kv_quant "auto" mixes INT8 into the middle
    // layers, so the per-layer decode-attention kernel family must be
    // emitted (one kernel per layer, precision baked into each).
    assert_eq!(
        plan.quant_attention_kernels.len(),
        8,
        "auto kv_quant on the 8-layer fixture must emit one kernel per layer"
    );
    assert!(plan
        .quant_attention_kernels
        .iter()
        .any(|(name, ptx)| name == "nsl_cfie_decode_attn_l0"
            && ptx.contains(".visible .entry nsl_cfie_decode_attn_l0")));
    // At least one INT8 layer dequantizes in registers.
    assert!(
        plan.quant_attention_kernels
            .iter()
            .any(|(_, ptx)| ptx.contains("cvt.rn.f32.s8")),
        "auto plan must produce at least one INT8 layer kernel"
    );

    // The report renders with the real GPU + all six sections.
    let report = plan.render_report();
    assert!(report.contains("CFIE Inference Build Report"));
    assert!(report.contains("H100-SXM"));
    assert!(report.contains("direct-index decode attention: nsl_cfie_decode_attn emitted"));
    assert!(report.contains("persistent decode block: nsl_cfie_decode_block emitted"));
    assert!(report.contains("verify attention: nsl_cfie_spec_verify_attn emitted"));
    assert!(report.contains("rejection epilogue: nsl_cfie_spec_reject emitted"));
    assert!(report.contains("per-layer decode-attention kernels: 8 emitted"));
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
fn grammar_schema_without_tokenizer_fails_compile_with_g12_refusal() {
    // Schema without a tokenizer vocab cannot be token-projected; the
    // refusal must tell the user to add `tokenizer:` (audit gap G12).
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
        Ok(_) => panic!("schema without tokenizer must refuse"),
    };
    assert!(
        err.message.contains("G12") && err.message.contains("tokenizer"),
        "refusal must cite the audit gap and the fix: {}",
        err.message
    );
}

#[test]
fn grammar_schema_with_tokenizer_bakes_mask_into_plan() {
    let dir = std::env::temp_dir().join("nsl_cfie_grammar_e2e");
    let _ = std::fs::create_dir_all(&dir);
    let schema_path = dir.join("schema.json");
    std::fs::write(&schema_path, r#"{"type": "boolean"}"#).unwrap();
    let vocab_path = dir.join("vocab.txt");
    std::fs::write(&vocab_path, "true\nfalse\ntr\nue\nx\n").unwrap();

    // Forward slashes keep the NSL string literal free of escapes.
    let schema_str = schema_path.display().to_string().replace('\\', "/");
    let vocab_str = vocab_path.display().to_string().replace('\\', "/");
    let src = format!(
        r#"
fn main():
    serve WithGrammar:
        kv_layout: "static"
        target_gpu: "h100"
        n_layers: 8
        n_kv_heads: 4
        head_dim: 128
        d_model: 512
        vocab_size: 5

        grammar:
            schema: "{schema_str}"
            tokenizer: "{vocab_str}"

        @endpoint
        fn handle(prompt: str) -> str:
            let x = 0
"#
    );
    let plan = compile_cfie(&src, &CompileOptions::default())
        .expect("grammar section must activate CFIE");

    // The token-level DFA reached the plan (5-token vocab).
    let dfa = plan.grammar.as_ref().expect("token DFA on the plan");
    assert_eq!(dfa.vocab_size, 5);
    assert!(plan.sampling.params.grammar_masked);

    // G11: the mask global is baked as an initialized .global fragment.
    let mask = plan
        .grammar_mask_ptx
        .as_deref()
        .expect("grammar mask PTX on the plan");
    assert!(mask.starts_with(".global .align 1 .b8 nsl_cfie_grammar_mask["));
    assert!(mask.bytes().all(|b| b < 128), "PTX must be ASCII-only");

    // Report shows the DFA section plus the baked-mask line.
    let report = plan.render_report();
    assert!(report.contains("[6] Grammar DFA:"));
    assert!(report.contains("mask baked into module image"));

    let _ = std::fs::remove_file(&schema_path);
    let _ = std::fs::remove_file(&vocab_path);
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
