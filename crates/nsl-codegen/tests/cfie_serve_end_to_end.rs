//! CFIE Tier-A wiring (audit gaps G1-G6): end-to-end proof that a
//! serve block with CFIE config drives the orchestrator through the
//! REAL pipeline (lex -> parse -> semantic -> codegen), populating
//! `last_cfie_plan` with inputs resolved from the serve config + GPU
//! database — the exact reachability the CFIE audit found missing.
//!
//! Serve blocks live inside `fn main()` here because the debug entry
//! point compiles function bodies, not synthesized top-level mains
//! (same convention as `wrga_freeze_end_to_end.rs`).

use nsl_codegen::cfie::{
    choose_kernel_family, kernel_registrations, CfieKernelFamily, CfieLaunchSpec, CfiePlan,
};
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

    // Feature 2 / Cycle 6: the fused decode-sample kernel is emitted
    // (vocab 49152 is a multiple of the 128 tile, top_k 50 in 1..=64,
    // d_model 512 <= 8192) with the single-CTA launch geometry.
    assert_eq!(
        plan.fused_sample_kernel.as_deref(),
        Some("nsl_cfie_fused_sample")
    );
    let fs_ptx = plan
        .fused_sample_ptx
        .as_deref()
        .expect("fused-sample PTX must be stored on the plan");
    assert!(fs_ptx.contains(".visible .entry nsl_cfie_fused_sample"));
    assert!(
        !fs_ptx.contains("nsl_cfie_grammar_mask"),
        "no grammar section -> no mask splice"
    );
    assert_eq!(
        plan.fused_sample_launch,
        Some(CfieLaunchSpec {
            grid_x: 1,
            block_x: 128,
            smem_dyn_bytes: 0
        })
    );

    // Cycle 6 family chooser: kv_quant "auto" mixed INT8 in, so the
    // QUANT family wins registration — kind 5 per layer + the kind-1
    // sampler; the emitted decode-block/spec kernels stay on the plan
    // but are NOT registered (mixed-precision pool layout).
    assert_eq!(choose_kernel_family(&plan), CfieKernelFamily::Quant);
    let regs = kernel_registrations(&plan);
    assert_eq!(
        regs.iter().map(|r| (r.kind, r.layer_idx)).collect::<Vec<_>>(),
        (0..8u32).map(|l| (5u8, l)).chain([(1u8, 0u32)]).collect::<Vec<_>>(),
        "quant family: 8 per-layer kernels + fused sampler, nothing else"
    );
    assert!(regs.iter().all(|r| r.smem_dyn == 0));
    assert!(
        regs.iter().all(|r| r.kind != 2),
        "decode_block emitted on the plan but must NOT register in the quant family"
    );
    // grid_x = n_heads (4 = d_model 512 / head_dim 128) for the
    // attention kernels; 1 for the sampler.
    assert!(regs.iter().filter(|r| r.kind == 5).all(|r| r.grid_x == 4));
    assert_eq!(regs.last().unwrap().grid_x, 1);

    // G22: the cost model populated the estimate, and the report prints
    // the two paper lines (latency + throughput).  Parse the numbers
    // rather than pinning exact floats — assert cfie < baseline, both
    // positive, throughput positive, and ASCII-only ("us", never a
    // Unicode micro sign).
    let est = plan
        .cost_estimate
        .as_ref()
        .expect("run_cfie_for_serve must populate the cost estimate");
    assert!(est.cfie_us_per_token > 0.0);
    assert!(est.baseline_us_per_token > 0.0);
    assert!(
        est.cfie_us_per_token < est.baseline_us_per_token,
        "CFIE latency ({}) must beat baseline ({})",
        est.cfie_us_per_token,
        est.baseline_us_per_token
    );
    assert!(est.throughput_tok_s > 0.0);

    // The report renders with the real GPU + all six sections.
    let report = plan.render_report();
    assert!(report.contains("CFIE Inference Build Report"));
    assert!(report.contains("H100-SXM"));

    // G22 report lines, well-formed.
    let latency_line = report
        .lines()
        .find(|l| l.starts_with("Estimated decode latency:"))
        .expect("report must contain the latency line");
    // The estimate lines must be ASCII-only: the paper's microsecond
    // renders as "us", never a Unicode micro sign.
    assert!(
        latency_line.bytes().all(|b| b < 128),
        "latency line must be ASCII-only (us, never a Unicode micro sign): {latency_line}"
    );
    assert!(latency_line.contains("us/token"));
    assert!(latency_line.contains("baseline"));
    assert!(latency_line.contains("bandwidth-roofline model"));
    // Parse the two floats out of "... 82.3us/token (vs 210.1us baseline) ..."
    let nums: Vec<f64> = latency_line
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .filter(|s| !s.is_empty() && s.contains('.'))
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    assert!(
        nums.len() >= 2,
        "latency line must expose two parseable floats: {latency_line}"
    );
    let (rep_cfie, rep_baseline) = (nums[0], nums[1]);
    assert!(rep_cfie > 0.0 && rep_baseline > 0.0);
    assert!(
        rep_cfie < rep_baseline,
        "rendered CFIE latency ({rep_cfie}) must be < baseline ({rep_baseline})"
    );
    let tput_line = report
        .lines()
        .find(|l| l.starts_with("Estimated throughput"))
        .expect("report must contain the throughput line");
    assert!(
        tput_line.bytes().all(|b| b < 128),
        "throughput line must be ASCII-only: {tput_line}"
    );
    assert!(tput_line.contains("tok/s"));
    // This fixture sets max_batch: 64, so the throughput line reports
    // the scheduler's max_active (64), not the paper's 32.
    assert!(
        tput_line.contains("batch=64"),
        "throughput batch must match the fixture's max_batch: {tput_line}"
    );
    assert!(est.batch == 64);
    // At least one digit in the throughput figure.
    assert!(tput_line.chars().any(|c| c.is_ascii_digit()));
    assert!(report.contains("cost-model assumptions:"));
    assert!(report.contains("direct-index decode attention: nsl_cfie_decode_attn emitted"));
    assert!(report.contains("persistent decode block: nsl_cfie_decode_block emitted"));
    assert!(report.contains("verify attention: nsl_cfie_spec_verify_attn emitted"));
    assert!(report.contains("rejection epilogue: nsl_cfie_spec_reject emitted"));
    assert!(report.contains("per-layer decode-attention kernels: 8 emitted"));
    assert!(report.contains("[6] Grammar DFA: disabled"));
    // Cycle 6 report lines: fused sampler, truthful wiring status, and
    // the quant-family exclusivity note.
    assert!(report.contains("fused sampler kernel: nsl_cfie_fused_sample emitted"));
    assert!(report.contains("registered at serve init"));
    assert!(report.contains("host decode loop = nsl_cfie_decode_step"));
    assert!(report.contains("Kernel family: quant"));
    assert!(report.contains(
        "decode-block/spec kernels NOT registered (mixed-precision pool layout is incompatible)"
    ));
    assert!(
        !report.contains("launch wiring pending"),
        "the pending-wiring honesty notes must be replaced by truthful lines"
    );
}

#[test]
fn uniform_fp16_kv_quant_registers_uniform_family() {
    // Same paper config but kv_quant "uniform_fp16": every layer stays
    // FP16 -> no per-layer quant kernels -> the UNIFORM family
    // registers (decode_attn + decode_block + spec verify/reject +
    // fused sampler).
    let src = r#"
fn main():
    serve Inference:
        max_batch: 64
        max_seq: 2048
        kv_layout: "static"
        kv_quant: "uniform_fp16"
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
    let plan = compile_cfie(src, &CompileOptions::default()).expect("CFIE active");
    assert!(
        plan.quant_attention_kernels.is_empty(),
        "uniform_fp16 must not emit the per-layer quant family"
    );
    assert_eq!(choose_kernel_family(&plan), CfieKernelFamily::Uniform);
    assert!(plan.decode_block_kernel.is_some());

    let regs = kernel_registrations(&plan);
    assert_eq!(
        regs.iter().map(|r| r.kind).collect::<Vec<_>>(),
        vec![0, 2, 3, 4, 1],
        "uniform family: decode_attn + decode_block + verify + reject + sampler"
    );
    assert!(regs.iter().all(|r| r.layer_idx == 0));
    // Attention-style kernels launch one CTA per head (4 heads);
    // decode_block and the sampler are single-CTA; reject is one warp.
    assert_eq!(regs[0].grid_x, 4);
    assert_eq!((regs[1].grid_x, regs[1].block_x), (1, 128));
    assert_eq!(regs[2].grid_x, 4);
    assert_eq!((regs[3].grid_x, regs[3].block_x), (1, 32));
    assert_eq!((regs[4].grid_x, regs[4].block_x), (1, 128));

    let report = plan.render_report();
    assert!(report.contains("Kernel family: uniform"));
    assert!(!report.contains("mixed-precision pool layout is incompatible"));
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

    // Cycle 6 miss path: vocab 5 is not a multiple of the 128-wide
    // sampler tile, so the fused-sample kernel is skipped and the mask
    // stays unbound (the report says so instead of overclaiming).
    assert!(plan.fused_sample_kernel.is_none());
    assert!(report.contains("fused sampler not emitted this build - mask unbound"));

    let _ = std::fs::remove_file(&schema_path);
    let _ = std::fs::remove_file(&vocab_path);
}

#[test]
fn grammar_mask_is_spliced_into_fused_sampler_module() {
    // A 128-token vocab satisfies the sampler's tile precondition, so
    // the fused-sample kernel is emitted AND the grammar mask .global
    // fragment is spliced into its module (Cycle 6) — the host resolves
    // "nsl_cfie_grammar_mask" via cuModuleGetGlobal at engine finalize.
    let dir = std::env::temp_dir().join("nsl_cfie_grammar_splice_e2e");
    let _ = std::fs::create_dir_all(&dir);
    let schema_path = dir.join("schema.json");
    std::fs::write(&schema_path, r#"{"type": "boolean"}"#).unwrap();
    let vocab_path = dir.join("vocab128.txt");
    let mut vocab = String::from("true\nfalse\n");
    for i in 0..126 {
        vocab.push_str(&format!("tok{i}\n"));
    }
    std::fs::write(&vocab_path, vocab).unwrap();

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
        vocab_size: 128

        sampling:
            temperature: 0.7
            top_k: 50
            fused: true

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

    assert!(plan.grammar.is_some());
    assert!(plan.sampling.params.grammar_masked);
    assert_eq!(
        plan.fused_sample_kernel.as_deref(),
        Some("nsl_cfie_fused_sample")
    );
    let fs_ptx = plan
        .fused_sample_ptx
        .as_deref()
        .expect("fused-sample PTX on the plan");
    assert!(
        fs_ptx.contains("nsl_cfie_grammar_mask["),
        "mask .global must be spliced into the sampler module"
    );
    // Module-scope placement: header, then mask, then the kernel entry.
    let addr = fs_ptx.find(".address_size 64").expect("module header");
    let mask = fs_ptx.find("nsl_cfie_grammar_mask").unwrap();
    let entry = fs_ptx.find(".visible .entry nsl_cfie_fused_sample").unwrap();
    assert!(addr < mask && mask < entry);
    // The spliced module registers as the kind-1 kernel.
    let regs = kernel_registrations(&plan);
    let samp = regs
        .iter()
        .find(|r| r.kind == 1)
        .expect("fused sampler must register");
    assert!(samp.ptx.contains("nsl_cfie_grammar_mask["));

    let report = plan.render_report();
    assert!(report.contains("spliced into the fused sampler module"));

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

// ---------------------------------------------------------------------------
// CFIE Cycle 11: endpoint wiring emission
// ---------------------------------------------------------------------------

/// Compile `src` to an object file and return the set of runtime-function
/// symbols that are *actually called* — i.e. targeted by at least one
/// relocation.  Membership proves the compile emitted a real call, not
/// merely that `declare_runtime_functions` imported the symbol (it imports
/// every runtime function unconditionally).  Mirrors the relocation-walk
/// in `csha_gap_a_forward_saves.rs`.
fn called_runtime_symbols(src: &str) -> std::collections::HashSet<String> {
    use object::{Object, ObjectSection, ObjectSymbol};

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

    let obj_bytes = nsl_codegen::compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        &CompileOptions::default(),
    )
    .expect("compile_module must succeed");

    let file = object::File::parse(&*obj_bytes).expect("object::File::parse");
    let name_by_index: std::collections::HashMap<_, _> = file
        .symbols()
        .filter_map(|s| {
            let n = s.name().ok()?;
            Some((s.index(), nsl_codegen::linker::strip_host_symbol_prefix(n).to_string()))
        })
        .collect();

    let mut called = std::collections::HashSet::new();
    for section in file.sections() {
        for (_offset, reloc) in section.relocations() {
            if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                if let Some(name) = name_by_index.get(&idx) {
                    called.insert(name.clone());
                }
            }
        }
    }
    called
}

/// A CFIE serve block whose endpoint calls `generate()` must emit the
/// Cycle-11 model-binding + generation driver calls: `nsl_model_create`
/// (load the served weights), `nsl_cfie_bind_model` (upload + record the
/// shape), and `nsl_cfie_generate` (drive the decode loop) — and it must
/// NOT fall back to the dead M29 `nsl_serve_enqueue` path (the pre-Cycle-11
/// bug that used the model handle as `prompt_len`).
#[test]
fn cfie_serve_generate_emits_the_driver_calls() {
    // Full CFIE config so the decode path is wired (kernels emitted ->
    // engine finalize -> the endpoint init + generate() are emitted).
    let src = r#"
fn main():
    serve Inference:
        max_batch: 64
        max_seq: 2048
        kv_layout: "static"
        kv_quant: "uniform_fp16"
        target_gpu: "h100"
        n_layers: 8
        n_kv_heads: 4
        head_dim: 128
        d_model: 512
        d_ff: 1408
        vocab_size: 49152
        weights: "model.safetensors"
        tokenizer: "tokenizer.json"
        prompt: "Hello"
        max_new_tokens: 16
        eos_token_id: 2

        sampling:
            temperature: 0.7
            top_k: 50
            fused: true

        @endpoint
        fn generate_endpoint(prompt: str) -> str:
            let n = generate(prompt, prompt, 0)
            print(n)
"#;
    let called = called_runtime_symbols(src);

    for sym in [
        "nsl_model_create",
        "nsl_cfie_bind_model",
        "nsl_cfie_generate",
        "nsl_tokenizer_load",
        // Cycle 12 — runtime prompt encode (tokenizer + prompt are both
        // configured, so the serve init encodes the prompt through the
        // real tokenizer and bridges the tensor to the i64 prompt ABI)...
        "nsl_tokenizer_encode",
        "nsl_cfie_tensor_to_tokens",
        // ...and the decode-and-print text tail on generate()'s output.
        "nsl_cfie_tokens_to_tensor",
        "nsl_tokenizer_decode",
        "nsl_print_str",
        // Free discipline: the encode tensor, the decode string, and the
        // bridge tensor are all released.
        "nsl_tensor_free",
        "nsl_string_free",
    ] {
        assert!(
            called.contains(sym),
            "CFIE serve generate() must emit a call to {sym}; called runtime symbols: {called:?}"
        );
    }
    // The old broken path must be gone.
    assert!(
        !called.contains("nsl_serve_enqueue"),
        "generate() must NOT enqueue into the dead M29 path (pre-Cycle-11 bug)"
    );
    // Cycle 13: WITHOUT a speculative draft, none of the draft-in-binary
    // FFIs may be emitted — no behavior change for existing programs.
    for sym in [
        "nsl_cfie_bind_draft_model",
        "nsl_cfie_draft_pool_alloc",
        "nsl_cfie_speculative_generate",
    ] {
        assert!(
            !called.contains(sym),
            "no draft_weights -> {sym} must not be emitted; called: {called:?}"
        );
    }
}

/// A CFIE serve block WITHOUT a tokenizer still wires model binding +
/// generation; only `nsl_tokenizer_load` is absent.
#[test]
fn cfie_serve_generate_without_tokenizer_still_binds_and_generates() {
    let src = r#"
fn main():
    serve Inference:
        max_batch: 64
        max_seq: 2048
        kv_layout: "static"
        kv_quant: "uniform_fp16"
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
            fused: true

        @endpoint
        fn generate_endpoint(prompt: str) -> str:
            let n = generate(prompt, prompt, 0)
            print(n)
"#;
    let called = called_runtime_symbols(src);
    assert!(called.contains("nsl_model_create"));
    assert!(called.contains("nsl_cfie_bind_model"));
    assert!(called.contains("nsl_cfie_generate"));
    assert!(
        !called.contains("nsl_tokenizer_load"),
        "no tokenizer: key -> no tokenizer load emitted"
    );
    // Cycle 12: with no tokenizer there is no runtime prompt encode and
    // no text-decode tail — the count-print behavior is preserved and
    // none of the bridge FFIs are reachable.
    for sym in [
        "nsl_tokenizer_encode",
        "nsl_tokenizer_decode",
        "nsl_cfie_tensor_to_tokens",
        "nsl_cfie_tokens_to_tensor",
    ] {
        assert!(
            !called.contains(sym),
            "no tokenizer: key -> {sym} must not be emitted; called: {called:?}"
        );
    }
    assert!(!called.contains("nsl_serve_enqueue"));
    // Cycle 13: no draft -> no draft-in-binary FFIs.
    assert!(!called.contains("nsl_cfie_bind_draft_model"));
    assert!(!called.contains("nsl_cfie_speculative_generate"));
}

// ---------------------------------------------------------------------------
// CFIE Cycle 13 (G15): draft-model-in-binary emission
// ---------------------------------------------------------------------------

/// Full CFIE config + a speculative draft compiled into the binary.
/// Uniform family (the kind-2 verify chain requirement), method
/// "standard" (the linear-K driver), K = 3, a 2-layer half-width draft.
const CFIE_DRAFT_SERVE_SRC: &str = r#"
fn main():
    serve Inference:
        max_batch: 64
        max_seq: 2048
        kv_layout: "static"
        kv_quant: "uniform_fp16"
        target_gpu: "h100"
        n_layers: 8
        n_kv_heads: 4
        head_dim: 128
        d_model: 512
        d_ff: 1408
        vocab_size: 49152
        weights: "model.safetensors"
        max_new_tokens: 16
        eos_token_id: 2

        sampling:
            temperature: 0.7
            top_k: 50
            fused: true

        speculative:
            tokens: 3
            method: "standard"
            draft_weights: "draft.safetensors"
            draft_n_layers: 2
            draft_d_model: 256
            draft_n_heads: 2
            draft_n_kv_heads: 1
            draft_head_dim: 128
            draft_d_ff: 704

        @endpoint
        fn generate_endpoint(prompt: str) -> str:
            let n = generate(prompt, prompt, 0)
            print(n)
"#;

/// The draft fixture populates the kind-6/7/8 kernels + the resolved
/// draft info on the plan, registers the full uniform family
/// [0,2,3,4,6,7,8,1], and renders the truthful report lines.
#[test]
fn cfie_draft_serve_emits_kind_6_7_8_and_draft_info() {
    let plan = compile_cfie(CFIE_DRAFT_SERVE_SRC, &CompileOptions::default())
        .expect("draft fixture must activate CFIE");

    // Resolved draft info: explicit draft_* keys + shared vocab + the
    // TARGET's per-slot capacity in the pool sizing.
    let draft = plan
        .speculative_draft
        .as_ref()
        .expect("draft_weights must populate speculative_draft");
    assert_eq!(draft.weights_path, "draft.safetensors");
    assert_eq!(
        (draft.n_layers, draft.d_model, draft.n_heads, draft.n_kv_heads),
        (2, 256, 2, 1)
    );
    assert_eq!((draft.head_dim, draft.d_ff), (128, 704));
    assert_eq!(draft.vocab_size, 49152, "vocab is the target's (shared)");
    assert_eq!(draft.k_tokens, 3);
    assert_eq!(draft.target_n_layers, 8);
    let per_slot = plan
        .kv
        .direct
        .as_ref()
        .expect("static layout")
        .per_sequence_max_tokens as i64;
    assert_eq!(
        draft.pool_bytes,
        2 * 2 * per_slot * 1 * 128 * 2,
        "draft pool = n_layers * 2 * per_slot(TARGET) * n_kv * hd * 2 (f16)"
    );

    // Kind 6: the decode-block emitter instantiated with the DRAFT dims.
    let dblk = plan
        .draft_block_ptx
        .as_deref()
        .expect("draft decode-block PTX on the plan");
    assert_eq!(plan.draft_block_kernel.as_deref(), Some("nsl_cfie_decode_block"));
    assert!(dblk.contains(".visible .entry nsl_cfie_decode_block"));
    assert_ne!(
        dblk,
        plan.decode_block_ptx.as_deref().expect("target block PTX"),
        "draft block bakes DRAFT dims — it must differ from the target block"
    );
    // Kinds 7/8: the Cycle-13 sampler family.
    assert_eq!(
        plan.draft_sample_kernel.as_deref(),
        Some("nsl_cfie_draft_sample")
    );
    assert!(plan
        .draft_sample_ptx
        .as_deref()
        .expect("kind-7 PTX")
        .contains(".visible .entry nsl_cfie_draft_sample"));
    assert_eq!(
        plan.verify_probs_kernel.as_deref(),
        Some("nsl_cfie_verify_probs")
    );
    assert!(plan
        .verify_probs_ptx
        .as_deref()
        .expect("kind-8 PTX")
        .contains(".visible .entry nsl_cfie_verify_probs"));

    // Registration list: the uniform family + the draft trio, sampler
    // last (the Cycle-6 convention).
    assert_eq!(choose_kernel_family(&plan), CfieKernelFamily::Uniform);
    let regs = kernel_registrations(&plan);
    assert_eq!(
        regs.iter().map(|r| r.kind).collect::<Vec<_>>(),
        vec![0, 2, 3, 4, 6, 7, 8, 1],
        "uniform family + kinds 6/7/8 when the draft is compiled in"
    );
    // Kinds 6/7/8 are single-CTA block-128 launches.
    for k in [6u8, 7, 8] {
        let r = regs.iter().find(|r| r.kind == k).unwrap();
        assert_eq!((r.grid_x, r.block_x, r.smem_dyn), (1, 128, 0));
    }

    // Report truthing: the [3] section names the draft kernels, the
    // driver, and the per-round launch accounting
    // (K*(draft_layers+1) + K*(target_layers+1) + 1 = 3*3 + 3*9 + 1).
    let report = plan.render_report();
    assert!(report.contains("draft decode block: nsl_cfie_decode_block emitted (kind 6"));
    assert!(report.contains("draft greedy sampler: nsl_cfie_draft_sample emitted (kind 7"));
    assert!(report.contains("verify prob-row writer: nsl_cfie_verify_probs emitted (kind 8"));
    assert!(report.contains("speculative decode driver: nsl_cfie_speculative_generate"));
    assert!(report.contains("draft model bound at serve init from 'draft.safetensors'"));
    assert!(
        report.contains("K=3, 37 launches per round"),
        "per-round launch accounting must be truthful: {report}"
    );
    assert!(
        !report.contains("G15"),
        "the old G15 refusal text must be gone from the report"
    );
}

/// The draft fixture's binary emits the Cycle-13 serve-init calls
/// (draft model_create -> bind_draft -> pool alloc) and generate()
/// rewrites to the SPECULATIVE driver — the plain driver must be gone.
#[test]
fn cfie_draft_serve_emits_bind_draft_pool_and_speculative_generate() {
    let called = called_runtime_symbols(CFIE_DRAFT_SERVE_SRC);
    for sym in [
        "nsl_model_create",
        "nsl_cfie_bind_model",
        "nsl_cfie_bind_draft_model",
        "nsl_cfie_draft_pool_alloc",
        "nsl_cfie_speculative_generate",
    ] {
        assert!(
            called.contains(sym),
            "draft serve must emit a call to {sym}; called: {called:?}"
        );
    }
    assert!(
        !called.contains("nsl_cfie_generate"),
        "with a compiled-in draft, generate() must drive the speculative \
         loop INSTEAD of nsl_cfie_generate; called: {called:?}"
    );
    assert!(!called.contains("nsl_serve_enqueue"));
}

/// draft_weights under the QUANT family cannot wire the kind-2 verify
/// chain — the compile must refuse loudly, never ship a binary whose
/// generate() refuses at runtime.
#[test]
fn cfie_draft_with_quant_family_refuses_at_compile_time() {
    let src = CFIE_DRAFT_SERVE_SRC.replace("\"uniform_fp16\"", "\"auto\"");
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(&src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(parsed.diagnostics.is_empty());
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    let res = nsl_codegen::debug_compile_and_return_cfie_plan_from_ast(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &CompileOptions::default(),
    );
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("draft_weights + quant family must refuse at compile time"),
    };
    assert!(
        err.message.contains("draft_weights") && err.message.contains("uniform_fp16"),
        "refusal must name the key and the fix: {}",
        err.message
    );
}

/// `generate()` called OUTSIDE a CFIE serve context has no bound model to
/// drive, so it must refuse at compile time with a clear message — never
/// silently emit a broken enqueue.
#[test]
fn generate_outside_cfie_serve_context_refuses() {
    let src = r#"
fn main():
    let tok = tokenizer_load("t.json")
    let ids = tokenizer_encode(tok, "hi")
    let out = generate(0, ids, 0)
    print(out)
"#;
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    // Semantics accept `generate` (registered builtin); codegen must refuse.
    let res = nsl_codegen::compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        &CompileOptions::default(),
    );
    let err = match res {
        Err(e) => e,
        Ok(_) => panic!("generate() outside a CFIE serve block must refuse"),
    };
    assert!(
        err.message.contains("CFIE-active serve block"),
        "refusal must explain the CFIE serve requirement: {}",
        err.message
    );
}
