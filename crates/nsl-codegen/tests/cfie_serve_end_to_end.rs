//! CFIE serve wiring, end to end: a serve block with CFIE config driven
//! through the REAL pipeline (lex -> parse -> semantic -> codegen).
//!
//! Three things are checked here that the unit tests of the planners cannot
//! see:
//!
//! * what a build contains — which kernels are emitted and registered,
//!   which runtime functions the endpoint's `generate()` calls;
//! * that every configuration which used to build a binary whose
//!   `generate()` always refused at run time is now a compile error that
//!   says why, or (where the configuration is fine) now builds the decode
//!   path;
//! * that the build report states what the build contains.
//!
//! Serve blocks live inside `fn main()` here because the debug entry
//! point compiles function bodies, not synthesized top-level mains
//! (same convention as `wrga_freeze_end_to_end.rs`).

use nsl_codegen::cfie::{kernel_registrations, CfieMode, CfiePlan};
use nsl_codegen::CompileOptions;

/// A serve block: the model of the shipped examples (8 layers, d_model
/// 512), with `{extra}` spliced into the config and `{body}` as the
/// endpoint's body.
fn serve_src(config: &str, sections: &str, body: &str) -> String {
    format!(
        r#"
fn main():
    serve Inference:
        max_batch: 64
        max_seq: 2048
        target_gpu: "h100"
        n_layers: 8
        n_kv_heads: 4
        head_dim: 128
        d_model: 512
        d_ff: 1408
        vocab_size: 49152
{config}
{sections}
        @endpoint
        fn complete(prompt: str) -> str:
{body}
"#
    )
}

const STATIC: &str = "        kv_layout: \"static\"\n";
const SAMPLING: &str = "        sampling:\n            temperature: 0.7\n            top_k: 50\n            top_p: 0.9\n            fused: true\n";
const NO_GENERATE: &str = "            let x = 0";
const GENERATE: &str = "            let n = generate(prompt, prompt, 0)\n            print(n)";
const DRAFT: &str = "        speculative:\n            tokens: 3\n            method: \"standard\"\n            draft_weights: \"draft.safetensors\"\n            draft_n_layers: 2\n            draft_d_model: 256\n            draft_n_heads: 2\n            draft_n_kv_heads: 1\n            draft_head_dim: 128\n            draft_d_ff: 704\n";

const LEGACY_SERVE_SRC: &str = r#"
fn main():
    serve Legacy:
        max_batch: 32
        kv_blocks: 2048

        @endpoint
        fn handle(prompt: str) -> str:
            let x = 0
"#;

/// What compiling `src` gives: the CFIE plan (`None` when CFIE is not
/// active), or the first error — the checker's, else codegen's.
fn try_compile(src: &str, opts: &CompileOptions) -> Result<Option<CfiePlan>, String> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(parsed.diagnostics.is_empty(), "parse must be clean: {:?}", parsed.diagnostics);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if let Some(d) = analysis.diagnostics.iter().find(|d| matches!(d.level, nsl_errors::Level::Error)) {
        return Err(format!("checker: {}", d.message));
    }
    // The real entry point decides whether the program compiles. (The
    // plan-returning debug entry below hands back a plan even when
    // codegen failed after planning.)
    nsl_codegen::compile_module(&parsed.module, &interner, &analysis.type_map, "", false, opts)
        .map_err(|e| format!("codegen: {}", e.message))?;
    nsl_codegen::debug_compile_and_return_cfie_plan_from_ast(&parsed.module, &interner, &analysis.type_map, opts)
        .map_err(|e| format!("codegen: {}", e.message))
}

fn compile_cfie(src: &str, opts: &CompileOptions) -> Option<CfiePlan> {
    try_compile(src, opts).unwrap_or_else(|e| panic!("compile must succeed: {e}"))
}

fn plan_of(src: &str) -> CfiePlan {
    compile_cfie(src, &CompileOptions::default()).expect("the serve block must activate CFIE")
}

fn error_of(src: &str) -> String {
    match try_compile(src, &CompileOptions::default()) {
        Err(e) => e,
        Ok(_) => panic!("this program must not compile:\n{src}"),
    }
}

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
    assert!(parsed.diagnostics.is_empty(), "parse must be clean: {:?}", parsed.diagnostics);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis.diagnostics.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
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
            if let object::RelocationTarget::Symbol(idx) = reloc.target()
                && let Some(name) = name_by_index.get(&idx)
            {
                called.insert(name.clone());
            }
        }
    }
    called
}

// ---------------------------------------------------------------------------
// What a build contains
// ---------------------------------------------------------------------------

#[test]
fn a_static_serve_block_builds_the_decode_path_from_real_inputs() {
    let plan = plan_of(&serve_src(STATIC, SAMPLING, NO_GENERATE));

    // Mode: implicit activation via serve keys => Full.  GPU resolved
    // from the serve-block `target_gpu` key via gpu_specs.
    assert_eq!(plan.mode, CfieMode::Full);
    assert_eq!(plan.target_gpu, "H100-SXM");

    // 8 layers x 4 KV heads x 128 head_dim x 2 (K and V) x 2 bytes is
    // 16 KiB per token; 64 sequences of 2,048 tokens fit an H100.
    assert!(plan.kv.uses_direct_indexing(), "{}", plan.kv.rationale);
    assert_eq!(plan.kv.direct.as_ref().unwrap().per_sequence_max_tokens, 2048);

    // The decode path: both kernels, and nothing else.
    assert!(plan.decode_path_emitted(), "{:?}", plan.generate_unavailable());
    let regs = kernel_registrations(&plan);
    assert_eq!(
        regs.iter().map(|r| (r.kind, r.name.as_str())).collect::<Vec<_>>(),
        [(2, "nsl_cfie_decode_block"), (1, "nsl_cfie_fused_sample")],
        "every registered kernel is one the decode loop launches"
    );
    assert!(regs.iter().all(|r| r.grid_x == 1 && r.block_x == 128 && r.layer_idx == 0));
    assert!(plan.draft_block.emitted.is_none() && plan.speculative_draft.is_none());

    // One decode-block launch per layer and one sampler launch.
    assert_eq!(plan.kernel_launches_per_token(), Some(9));
    assert_eq!(plan.kernel_launches_per_token_baseline, 12 * 8 + 6);
}

/// `kv_layout: "static"` with no `kv_quant` key used to select per-layer
/// INT8 read kernels, which excluded the decode block, so `generate()`
/// refused every call at run time.  It now builds the decode path, and the
/// endpoint's `generate()` calls the driver.
#[test]
fn static_with_no_kv_quant_key_builds_a_generate_that_runs() {
    let src = serve_src(STATIC, SAMPLING, GENERATE);
    let plan = plan_of(&src);
    assert!(plan.decode_path_emitted(), "{:?}", plan.generate_unavailable());
    assert!(plan.kv_quant.layers.is_empty(), "no precision plan without kv_quant: \"auto\"");
    let report = plan.render_report();
    assert!(report.contains("[5] KV precision: FP16 in every layer"), "{report}");
    assert!(report.contains("generate(): compiles to serve-init model binding"), "{report}");

    let called = called_runtime_symbols(&src);
    for sym in [
        "nsl_cfie_kv_slots_init",
        "nsl_cfie_register_kernel",
        "nsl_cfie_kv_pool_alloc",
        "nsl_cfie_engine_finalize",
        "nsl_model_create",
        "nsl_cfie_bind_model",
        "nsl_cfie_generate",
        "nsl_cfie_engine_destroy",
    ] {
        assert!(called.contains(sym), "expected a call to {sym}; called: {called:?}");
    }
    // The old broken path must be gone, and without a draft none of the
    // speculative FFIs may be emitted.
    for sym in [
        "nsl_serve_enqueue",
        "nsl_cfie_bind_draft_model",
        "nsl_cfie_draft_pool_alloc",
        "nsl_cfie_speculative_generate",
    ] {
        assert!(!called.contains(sym), "{sym} must not be called; called: {called:?}");
    }
}

#[test]
fn a_tokenizer_and_prompt_add_the_runtime_encode_and_the_text_tail() {
    let config = format!(
        "{STATIC}        weights: \"model.safetensors\"\n        tokenizer: \"tokenizer.json\"\n        prompt: \"Hello\"\n        max_new_tokens: 16\n        eos_token_id: 2\n"
    );
    let called = called_runtime_symbols(&serve_src(&config, SAMPLING, GENERATE));
    for sym in [
        "nsl_tokenizer_load",
        "nsl_tokenizer_encode",
        "nsl_cfie_tensor_to_tokens",
        "nsl_cfie_generate",
        "nsl_cfie_tokens_to_tensor",
        "nsl_tokenizer_decode",
        "nsl_print_str",
        "nsl_tensor_free",
        "nsl_string_free",
    ] {
        assert!(called.contains(sym), "expected a call to {sym}; called: {called:?}");
    }
    // Without a tokenizer the binding and the driver are still there.
    let called = called_runtime_symbols(&serve_src(STATIC, SAMPLING, GENERATE));
    assert!(called.contains("nsl_cfie_generate") && called.contains("nsl_cfie_bind_model"));
    assert!(!called.contains("nsl_tokenizer_load"));
}

#[test]
fn a_draft_model_adds_its_two_kernels_and_the_speculative_driver() {
    let src = serve_src(STATIC, &format!("{SAMPLING}\n{DRAFT}"), GENERATE);
    let plan = plan_of(&src);
    let regs = kernel_registrations(&plan);
    assert_eq!(
        regs.iter().map(|r| (r.kind, r.name.as_str())).collect::<Vec<_>>(),
        [
            (2, "nsl_cfie_decode_block"),
            (6, "nsl_cfie_decode_block"),
            (7, "nsl_cfie_draft_sample"),
            (1, "nsl_cfie_fused_sample"),
        ]
    );
    let draft = plan.speculative_draft.as_ref().expect("draft info");
    assert_eq!((draft.n_layers, draft.d_model, draft.k_tokens, draft.target_n_layers), (2, 256, 3, 8));
    // [n_layers][2][per_slot][n_kv][head_dim] f16: 2 * 2 * 2048 * 1 * 128 * 2.
    assert_eq!(draft.pool_bytes, 2 * 2 * 2048 * 128 * 2);

    let report = plan.render_report();
    assert!(report.contains("[3] Speculative decoding: K=3, greedy draft model from 'draft.safetensors'"), "{report}");
    // 3 draft steps of 2 + 1 launches and 3 target steps of 8 + 1.
    assert!(report.contains("up to 36 launches per round for at most 3 tokens, against 9 per token without a draft"), "{report}");
    assert!(report.contains("SLOWER than plain decoding"), "{report}");
    assert!(!report.contains("speedup"), "{report}");

    let called = called_runtime_symbols(&src);
    for sym in ["nsl_cfie_bind_draft_model", "nsl_cfie_draft_pool_alloc", "nsl_cfie_speculative_generate"] {
        assert!(called.contains(sym), "expected a call to {sym}; called: {called:?}");
    }
    assert!(!called.contains("nsl_cfie_generate"), "the speculative driver replaces the plain one");
}

/// `norm_eps` reached the decode block only; the fused sampler and the
/// draft sampler baked 1e-5 for the final norm whatever the model said.
#[test]
fn norm_eps_reaches_every_kernel_that_normalises() {
    let f32_imm = |v: f32| format!("0f{:08X}", v.to_bits());
    let src = serve_src(&format!("{STATIC}        norm_eps: 0.000001\n"), &format!("{SAMPLING}\n{DRAFT}"), NO_GENERATE);
    let plan = plan_of(&src);
    for kernel in [&plan.decode_block, &plan.fused_sample, &plan.draft_block, &plan.draft_sample] {
        let e = kernel.emitted.as_ref().expect("emitted");
        assert!(e.ptx.contains(&f32_imm(1e-6)), "{}: the configured epsilon is not baked", e.name);
        assert!(!e.ptx.contains(&f32_imm(1e-5)), "{}: the old constant is still baked", e.name);
    }
    // And the default is still 1e-5 everywhere.
    let plan = plan_of(&serve_src(STATIC, SAMPLING, NO_GENERATE));
    for kernel in [&plan.decode_block, &plan.fused_sample] {
        assert!(kernel.emitted.as_ref().unwrap().ptx.contains(&f32_imm(1e-5)));
    }
}

// ---------------------------------------------------------------------------
// Configurations that used to build a binary whose generate() refused
// ---------------------------------------------------------------------------

/// A model too wide for the one-block kernel (d_model 4096).  The planner
/// says "level3"; the kernel's static shared memory is past what one
/// thread block may declare, so no decode block is emitted.  With a
/// `generate()` in the endpoint that is a compile error; without one the
/// build succeeds and the report says what is missing.
#[test]
fn a_model_too_wide_for_the_decode_block_cannot_compile_generate() {
    let wide = |body: &str| {
        serve_src(STATIC, SAMPLING, body)
            .replace("d_model: 512", "d_model: 4096")
            .replace("head_dim: 128", "head_dim: 128\n        n_heads: 32")
    };
    let err = error_of(&wide(GENERATE));
    assert!(err.contains("generate() cannot run in this serve block"), "{err}");
    assert!(err.contains("the decode-block kernel was not emitted"), "{err}");
    assert!(err.contains("static shared memory would be"), "{err}");
    assert!(err.contains("at most 49152"), "{err}");

    let plan = plan_of(&wide(NO_GENERATE));
    assert!(!plan.decode_block.is_emitted());
    assert!(plan.fused_sample.is_emitted(), "the sampler alone still fits");
    assert!(kernel_registrations(&plan).is_empty(), "and is not registered: nothing would launch it");
    assert_eq!(plan.kernel_launches_per_token(), None);
    let report = plan.render_report();
    assert!(report.contains("[4] Decode block (planner: fusion=level3): NOT emitted — its static shared memory would be"), "{report}");
    assert!(report.contains("generate(): NOT available in this build"), "{report}");
    assert!(!report.contains("nsl_cfie_decode_block emitted"), "{report}");
}

/// `top_k: 100` used to pass, skip the sampler kernel with a note at the
/// end of the report, and leave a `generate()` that refused.
#[test]
fn a_top_k_past_the_samplers_capacity_is_a_compile_error() {
    let err = error_of(&serve_src(STATIC, &SAMPLING.replace("top_k: 50", "top_k: 100"), GENERATE));
    assert!(err.contains("top_k must be an integer in [1, 64]"), "{err}");
}

/// A vocabulary the sampler kernel cannot tile: no sampler, so no
/// `generate()` — said at compile time.
#[test]
fn a_vocabulary_the_sampler_cannot_tile_cannot_compile_generate() {
    let err = error_of(&serve_src(STATIC, SAMPLING, GENERATE).replace("vocab_size: 49152", "vocab_size: 50000"));
    assert!(err.contains("the fused sampler kernel was not emitted"), "{err}");
    assert!(err.contains("a positive multiple of 128 (is 50000)"), "{err}");
}

/// An envelope the card cannot hold.  `kv_layout: "static"` used to plan
/// "static_with_bump", print that a GPU-side allocator would handle it,
/// and emit kernels for a 128 GiB pool on an 80 GB card.
#[test]
fn a_static_layout_that_does_not_fit_is_a_compile_error() {
    let big = |layout: &str, body: &str| {
        serve_src(&format!("        kv_layout: \"{layout}\"\n"), SAMPLING, body)
            .replace("max_batch: 64", "max_batch: 256")
            .replace("max_seq: 2048", "max_seq: 32768")
    };
    let err = error_of(&big("static", NO_GENERATE));
    assert!(err.contains("kv_layout: \"static\" cannot be honoured on H100-SXM"), "{err}");
    assert!(err.contains("8388608 tokens need 128.0 GiB of KV"), "{err}");
    assert!(err.contains("GiB is available"), "{err}");

    // "auto" may fall back: the plan is paged and has no decode block,
    // and the report says so instead of naming an allocator.
    let plan = plan_of(&big("auto", NO_GENERATE));
    assert!(!plan.kv.uses_direct_indexing());
    assert_eq!(plan.kv.kind.as_str(), "paged");
    assert!(!plan.decode_block.is_emitted());
    let report = plan.render_report();
    assert!(report.contains("NOT emitted — the KV layout is paged"), "{report}");
    assert!(!report.contains("bump"), "{report}");
    // And generate() on that plan is a compile error, not a dead binary.
    let err = error_of(&big("auto", GENERATE));
    assert!(err.contains("the KV layout is paged"), "{err}");
}

/// `kv_layout: "paged"` was accepted and then ignored (the planner chose
/// by size alone).
#[test]
fn a_requested_paged_layout_is_honoured() {
    let plan = plan_of(&serve_src("        kv_layout: \"paged\"\n", SAMPLING, NO_GENERATE));
    assert_eq!(plan.kv.kind.as_str(), "paged");
    assert!(plan.kv.rationale.contains("was requested"), "{}", plan.kv.rationale);
    assert!(!plan.decode_block.is_emitted());
}

#[test]
fn requests_nothing_honours_do_not_compile() {
    for (config, sections, needle) in [
        (format!("{STATIC}        kv_quant: \"uniform_int8\"\n"), SAMPLING.to_string(), "INT8 KV is not supported"),
        (
            STATIC.to_string(),
            format!("{SAMPLING}\n        grammar:\n            schema: \"s.json\"\n            tokenizer: \"v.txt\"\n"),
            "`grammar:` is not supported",
        ),
        (
            STATIC.to_string(),
            format!("{SAMPLING}\n        speculative:\n            tokens: 5\n"),
            "needs draft_weights",
        ),
        (
            STATIC.to_string(),
            format!("{SAMPLING}\n{}", DRAFT.replace("\"standard\"", "\"tree\"")),
            "method \"tree\" is not supported",
        ),
        (
            STATIC.to_string(),
            "        sampling:\n            temperature: 0.7\n            fused: true\n".to_string(),
            "a non-zero temperature needs top_k",
        ),
    ] {
        let err = error_of(&serve_src(&config, &sections, NO_GENERATE));
        assert!(err.contains(needle), "expected `{needle}` in: {err}");
    }
}

/// A draft model needs the target's decode path; when that is missing the
/// speculative section is refused with the reason.
#[test]
fn a_draft_model_without_a_decode_path_is_refused() {
    let src = serve_src("        kv_layout: \"paged\"\n", &format!("{SAMPLING}\n{DRAFT}"), NO_GENERATE);
    let err = error_of(&src);
    assert!(err.contains("CFIE speculative (draft_weights 'draft.safetensors')"), "{err}");
    assert!(err.contains("the KV layout is paged"), "{err}");
}

/// `generate()` OUTSIDE a CFIE serve context has no bound model: the
/// compile must REFUSE with a clear message rather than silently emit a
/// broken enqueue.
#[test]
fn generate_outside_cfie_serve_context_refuses() {
    let src = r#"
fn main():
    let tok = tokenizer_load("t.json")
    let ids = tokenizer_encode(tok, "hi")
    let out = generate(0, ids, 0)
    print(out)
"#;
    let err = error_of(src);
    assert!(err.contains("CFIE-active serve block"), "{err}");
}

// ---------------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------------

/// The numbers the report prints for the shipped example's model are the
/// build's own.  It used to print "Kernel launches per token: 407 (vs 500
/// baseline)", an estimated latency of 2069.4 us of which 2,035 were
/// 407 x 5, a throughput at batch 64, and "Continuous batching: on".
#[test]
fn the_report_counts_what_the_build_does() {
    let report = plan_of(&serve_src(STATIC, SAMPLING, NO_GENERATE)).render_report();
    assert!(report.contains("Kernel launches per generated token: 9 (8 decode-block + 1 sampler"), "{report}");
    assert!(report.contains("would make 102 (12 per layer + 6 for the sampling tail)"), "{report}");
    assert!(report.contains("[2] Sampler: temperature 0.7, top_k 50, top_p 0.9 (from the sampling: section)"), "{report}");
    assert!(report.contains("Requests: one sequence at a time"), "{report}");
    assert!(report.contains("Speed: not measured."), "{report}");
    for gone in ["407", "500 baseline", "us/token", "tok/s", "Continuous batching", "Estimated", "Kernel family"] {
        assert!(!report.contains(gone), "the report still prints `{gone}`:\n{report}");
    }
    // One statement about the decode block, and it is "emitted".
    assert_eq!(report.matches("nsl_cfie_decode_block emitted").count(), 1, "{report}");
    for absent in ["NOT emitted", "NOT registered", "NOT available"] {
        assert!(!report.contains(absent), "`{absent}` in:\n{report}");
    }
}

#[test]
fn kv_quant_auto_prints_a_labelled_advisory_and_changes_nothing() {
    let auto = plan_of(&serve_src(&format!("{STATIC}        kv_quant: \"auto\"\n"), SAMPLING, NO_GENERATE));
    let fp16 = plan_of(&serve_src(&format!("{STATIC}        kv_quant: \"uniform_fp16\"\n"), SAMPLING, NO_GENERATE));
    assert!(!auto.kv_quant.layers.is_empty());
    assert!(fp16.kv_quant.layers.is_empty());
    let report = auto.render_report();
    assert!(report.contains("[5] KV precision: FP16 in every layer"), "{report}");
    assert!(report.contains("advisory, NOT applied"), "{report}");
    // The two builds carry the same kernels.
    let ptx = |p: &CfiePlan| kernel_registrations(p).into_iter().map(|r| r.ptx).collect::<Vec<_>>();
    assert_eq!(ptx(&auto), ptx(&fp16));
}

#[test]
fn defaulted_sampler_settings_are_named_as_defaults() {
    let report = plan_of(&serve_src(STATIC, "", NO_GENERATE)).render_report();
    assert!(report.contains("(DEFAULTS — no sampling: section set them)"), "{report}");
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
    let _plan = compile_cfie(&serve_src(STATIC, SAMPLING, NO_GENERATE), &opts).expect("CFIE active");
    let contents = std::fs::read_to_string(&path).expect("report file must exist");
    assert!(contents.contains("CFIE Inference Build Report"));
    assert!(contents.contains("Model-shape provenance"), "report must attribute input sources: {contents}");
    assert!(contents.contains("Kernel wiring: KV-slot init + kernel registration + engine finalize"), "{contents}");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Activation
// ---------------------------------------------------------------------------

#[test]
fn legacy_serve_block_does_not_activate_cfie() {
    let plan = compile_cfie(LEGACY_SERVE_SRC, &CompileOptions::default());
    assert!(plan.is_none(), "a serve block without CFIE keys must leave the M29 path untouched");
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
    let plan = compile_cfie(&serve_src(STATIC, SAMPLING, NO_GENERATE), &opts);
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
    assert_eq!(plan.mode, CfieMode::Full);
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
    let plan = compile_cfie(src, &CompileOptions::default()).expect("@cfie decorator must activate CFIE");
    assert_eq!(plan.target_gpu, "A100-SXM", "@cfie(target=a100) must drive GPU resolution");
}
