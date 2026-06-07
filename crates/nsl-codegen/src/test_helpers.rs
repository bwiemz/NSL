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
            csha_configs: Vec::new(),
            weight_index_map: std::collections::HashMap::new(),
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

/// A1 regression test helper: build a minimal `@flash_attention`-decorated NSL
/// snippet, run `compile_flash_attention_kernels` against a `Compiler` whose
/// `compile_options.target` is set to `target`, and return the `gpu_sm` stored
/// in the resulting `FlashAttentionCompileContext`.
///
/// This exercises the **non-autotune call-site at ~line 756** in
/// `compiler/kernel.rs` (the path where
/// `parse_gpu_sm_from_target(&self.compile_options.target)` is called).
/// If that call-site were reverted to `gpu_sm: 80`, the return value would
/// always be `80` regardless of `target`, causing the regression test to fail
/// for every SM other than `sm_80`.
///
/// # Panics
/// Panics when `target` is not a recognised `sm_<N>` string, or when
/// `Compiler::new` fails.
pub fn flash_sm_for_compile_target(target: &str) -> u32 {
    use nsl_errors::FileId;

    // Parse a minimal NSL snippet that contains @flash_attention but NO
    // @autotune.  The absence of @autotune means compile_flash_attention_kernels
    // always takes the single-config path and hits the gpu_sm assignment at
    // ~line 756 in compiler/kernel.rs.
    let src = "@flash_attention\nfn forward():\n    pass\n";
    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let stmts = parsed.module.stmts.clone();

    let type_map: TypeMap = TypeMap::new();

    // Build a Compiler whose compile_options.target is set to `target`.
    // This is the field read by parse_gpu_sm_from_target at the three
    // fixed call-sites in kernel.rs.
    let opts = crate::CompileOptions {
        target: target.to_string(),
        ..Default::default()
    };

    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new failed in flash_sm_for_compile_target");

    compiler
        .compile_flash_attention_kernels(&stmts)
        .expect("compile_flash_attention_kernels failed in flash_sm_for_compile_target");

    compiler
        .kernels
        .flash_attention_context
        .expect("flash_attention_context not set after compile_flash_attention_kernels")
        .config
        .gpu_sm
}

/// Gap B observation helper: what `compile_flash_attention_kernels`
/// wired onto the `FlashAttentionCompileContext` when given a program
/// that (optionally) contains a `@train` block.
///
/// Returns `(has_forward_with_saves, has_backward_ptx, training_config_level)`
/// where the third field is `Some(level)` iff the training CSHA config
/// was recorded (meaning the pre-scan saw `@train`).  Used by the
/// Gap B integration test to assert the forward PTX actually grew the
/// save-write codepaths and that Gap C/D's expected backward-PTX
/// DataId arrived on the context.
pub fn flash_gap_b_context_for_source(src: &str) -> (bool, bool, Option<u8>) {
    use nsl_errors::FileId;

    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let stmts = parsed.module.stmts.clone();
    let type_map: TypeMap = TypeMap::new();
    // `parse_gpu_sm_from_target` panics on anything that isn't `sm_<N>`;
    // the default is `"cuda"` (valid at link time, not PTX time), so we
    // override to `sm_80` which matches the other test harnesses.
    let opts = crate::CompileOptions {
        target: "sm_80".to_string(),
        ..Default::default()
    };

    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new failed in flash_gap_b_context_for_source");

    compiler
        .compile_flash_attention_kernels(&stmts)
        .expect("compile_flash_attention_kernels failed in flash_gap_b_context_for_source");

    match compiler.kernels.flash_attention_context {
        Some(ctx) => (
            ctx.csha_forward_with_saves_ptx_id.is_some(),
            ctx.csha_backward_ptx_data_id.is_some(),
            ctx.csha_training_config
                .as_ref()
                .and_then(|c| c.csha.as_ref().map(|e| e.level)),
        ),
        None => (false, false, None),
    }
}

/// Gap F observation helper: does `compile_flash_attention_kernels`
/// detect `@flash_attention` on a `ModelMember::Method`, and does it
/// honour an optional `head_dim=N` decorator argument?
///
/// Returns `(context_set, head_dim, has_backward_ptx)` where:
/// - `context_set` is `true` iff a `FlashAttentionCompileContext`
///   landed on the compiler (proof the scanner descended into the
///   method decorators — pre-F.1 this stayed `false`).
/// - `head_dim` is the value that the v2 emitter will see
///   (`None` when no context was built).
/// - `has_backward_ptx` is `true` iff the Tier C SMEM validator
///   accepted the config and embedded the fused-backward PTX
///   (at hd=64 this is always false; at hd=32 it is true).
pub fn flash_gap_f_context_for_source(
    src: &str,
) -> (bool, Option<i64>, bool) {
    use nsl_errors::FileId;

    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let stmts = parsed.module.stmts.clone();
    let type_map: TypeMap = TypeMap::new();
    let opts = crate::CompileOptions {
        target: "sm_80".to_string(),
        ..Default::default()
    };

    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new failed in flash_gap_f_context_for_source");

    compiler
        .compile_flash_attention_kernels(&stmts)
        .expect("compile_flash_attention_kernels failed in flash_gap_f_context_for_source");

    match compiler.kernels.flash_attention_context {
        Some(ctx) => (
            true,
            Some(ctx.config.head_dim),
            ctx.csha_backward_ptx_data_id.is_some(),
        ),
        None => (false, None, false),
    }
}

/// Sprint 1 cycle-3 (paper §4 tree-mask) observation helper: does the
/// `@tree_mask` decorator on an `@flash_attention` fn reach
/// `FlashAttentionConfig::tree_mask`, and does the kernel-name +
/// synthesized PTX pick up the tree-mask variant?
///
/// Returns `(context_set, tree_mask_flag, kernel_name, ptx_contains_dfs_enter_ptr)`:
/// - `context_set`     — `true` iff a compile context was built (proof
///                       the extraction site ran at all).
/// - `tree_mask_flag`  — `Some(ctx.config.tree_mask)`; the load-bearing
///                       end-to-end thread.
/// - `kernel_name`     — what the runtime dispatcher will look up; pins
///                       the variant tag (`_t1_` for tree_mask=true) per
///                       `flash_attention.rs::test_tree_mask_variant`.
/// - `ptx_contains_dfs_enter_base` — `true` iff the synthesized PTX
///                       loads the `%dfs_enter_base` register from the
///                       `dfs_enter_ptr` kernel parameter. The parameter
///                       itself is emitted unconditionally
///                       (`flash_attention.rs:449`), but the register
///                       load (`flash_attention.rs:584`/`635`) is gated
///                       on `config.tree_mask` — so probing for the
///                       register gives the ground-truth proof that the
///                       M33 ancestor-check code path was taken.
pub fn flash_tree_mask_context_for_source(
    src: &str,
) -> (bool, Option<bool>, Option<String>, bool) {
    use nsl_errors::FileId;

    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let stmts = parsed.module.stmts.clone();
    let type_map: TypeMap = TypeMap::new();
    let opts = crate::CompileOptions {
        target: "sm_80".to_string(),
        ..Default::default()
    };

    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new failed in flash_tree_mask_context_for_source");

    compiler
        .compile_flash_attention_kernels(&stmts)
        .expect("compile_flash_attention_kernels failed in flash_tree_mask_context_for_source");

    match compiler.kernels.flash_attention_context {
        Some(ctx) => {
            let cfg = &ctx.config;
            let kernel_name = crate::flash_attention::flash_attention_kernel_name(cfg);
            let ptx = crate::flash_attention::synthesize_flash_attention_ptx(cfg);
            // Strip the trailing null byte that `synthesize_flash_attention_ptx`
            // appends (cudarc expects a null-terminated PTX blob).
            let ptx_body = if ptx.last() == Some(&0) {
                &ptx[..ptx.len() - 1]
            } else {
                &ptx[..]
            };
            let ptx_str = std::str::from_utf8(ptx_body).unwrap_or("");
            // Probe for the register, not the parameter: the param
            // `dfs_enter_ptr` is declared unconditionally; only the
            // register load `%dfs_enter_base` is gated on
            // `config.tree_mask` — so the register's presence is the
            // ground-truth proof that the M33 ancestor-check fired.
            let has_dfs = ptx_str.contains("%dfs_enter_base");
            (true, Some(cfg.tree_mask), Some(kernel_name), has_dfs)
        }
        None => (false, None, None, false),
    }
}

/// Sprint 2 cycle-3 (paper §3.2 paged KV) observation helper: does the
/// `@paged_kv` decorator on an `@flash_attention` fn reach
/// `FlashAttentionConfig::paged`, and does the kernel-name + synthesized
/// PTX pick up the paged-KV variant?
///
/// Returns `(context_set, paged_flag, kernel_name, ptx_contains_paged_block_table_indirection)`:
/// - `context_set`     — `true` iff a compile context was built (proof
///                       the extraction site ran at all).
/// - `paged_flag`      — `Some(ctx.config.paged)`; the load-bearing
///                       end-to-end thread.
/// - `kernel_name`     — what the runtime dispatcher will look up; pins
///                       the variant tag (`_p1_` for paged=true) per
///                       `flash_attention_kernel_name`'s format
///                       `flash_attn_p{paged}_r..._g..._c..._t..._q..._kv...`.
/// - `ptx_contains_paged_block_table_indirection` — `true` iff the
///                       synthesized PTX emits the paged-only K block-table
///                       indirection comment + divide. The KERNEL PARAMETER
///                       `block_table_ptr` is declared unconditionally
///                       (`flash_attention.rs:438`) and the `ld.param.u64`
///                       for it is also unconditional (`flash_attention.rs:622`),
///                       just like Sprint 1's `dfs_enter_ptr`. Only the
///                       block-table indirection comment + divide block
///                       (`flash_attention.rs:1622-1649`) is gated on
///                       `config.paged`. Probing for that comment gives
///                       the ground-truth proof that the paged code path
///                       fired — probing for the parameter name would
///                       be a Sprint-1-style false positive.
///
/// Note on `block_size`: the `@paged_kv(block_size=N)` argument is
/// parsed and used at compile time for the `block_kv % block_size`
/// alignment validation (`compiler/kernel.rs:1167`), but is NOT
/// threaded onto `FlashAttentionConfig` — it remains a launch-time
/// runtime parameter (the kernel's `.param .u64 block_size`). It is
/// therefore not observable through the compile context; the runtime
/// dispatch (not codegen) wires the launch arg. The fixture uses a
/// bare `@paged_kv` and this helper does not return a block_size.
pub fn flash_paged_kv_context_for_source(
    src: &str,
) -> (bool, Option<bool>, Option<String>, bool) {
    use nsl_errors::FileId;

    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let stmts = parsed.module.stmts.clone();
    let type_map: TypeMap = TypeMap::new();
    let opts = crate::CompileOptions {
        target: "sm_80".to_string(),
        ..Default::default()
    };

    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new failed in flash_paged_kv_context_for_source");

    compiler
        .compile_flash_attention_kernels(&stmts)
        .expect("compile_flash_attention_kernels failed in flash_paged_kv_context_for_source");

    match compiler.kernels.flash_attention_context {
        Some(ctx) => {
            let cfg = &ctx.config;
            let kernel_name = crate::flash_attention::flash_attention_kernel_name(cfg);
            let ptx = crate::flash_attention::synthesize_flash_attention_ptx(cfg);
            let ptx_body = if ptx.last() == Some(&0) {
                &ptx[..ptx.len() - 1]
            } else {
                &ptx[..]
            };
            let ptx_str = std::str::from_utf8(ptx_body).unwrap_or("");
            // Probe for the paged-only block-table indirection comment,
            // not the unconditional parameter. The comment
            // `"Paged: block table indirection per physical block"` is
            // emitted ONLY when `config.paged=true`
            // (`flash_attention.rs:1623` inside `if config.paged { .. }`).
            // The companion division `div.u64 %rd36, %k_start, %rd11`
            // (logical_block = k_start / block_size) is also paged-only
            // and would be a valid alternate probe; the comment is more
            // self-documenting in the test failure message.
            let has_paged_indirection =
                ptx_str.contains("Paged: block table indirection per physical block");
            (true, Some(cfg.paged), Some(kernel_name), has_paged_indirection)
        }
        None => (false, None, None, false),
    }
}

/// Sprint 2 cycle-4 (paper §4.3 attention sinks) observation helper: does
/// the `@attention_sink(tokens=N)` decorator on an `@flash_attention` fn
/// reach `FlashAttentionConfig::num_sink_tokens`?
///
/// Returns `(context_set, num_sink_tokens_flag)`:
/// - `context_set`           — `true` iff a compile context was built
///                             (proof the extraction site ran at all).
/// - `num_sink_tokens_flag`  — `Some(ctx.config.num_sink_tokens)`; the
///                             load-bearing end-to-end thread.
///
/// IMPORTANT — v0 API surface only:
/// This helper INTENTIONALLY does NOT probe the synthesized PTX. The
/// `num_sink_tokens` field is wired through the decorator-extraction
/// loop into `FlashAttentionConfig`, but the SMEM-layout codegen that
/// would actually materialize the sink cache (paper §4.3 Phase 5 v1)
/// is DEFERRED to a future sprint. Until that lands, the kernel emits
/// no sink-specific PTX — probing for a sink register would be either
/// a false positive (matching unrelated text) or a guaranteed false
/// negative. The integration test pins decorator → config wiring only,
/// per the explicit cycle-4 Sprint 2 scope.
///
/// When the SMEM emission lands, EXTEND this helper to return a 3-tuple
/// adding a `ptx_contains_sink_register: bool` field, and update the
/// integration test to assert it (mirrors the cycle-3 Sprint 1 review-fix
/// pattern in commit `0a987a73` — params unconditional, register loads
/// gated on the flag).
pub fn flash_attention_sink_context_for_source(
    src: &str,
) -> (bool, Option<u32>) {
    use nsl_errors::FileId;

    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let stmts = parsed.module.stmts.clone();
    let type_map: TypeMap = TypeMap::new();
    let opts = crate::CompileOptions {
        target: "sm_80".to_string(),
        ..Default::default()
    };

    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new failed in flash_attention_sink_context_for_source");

    compiler
        .compile_flash_attention_kernels(&stmts)
        .expect("compile_flash_attention_kernels failed in flash_attention_sink_context_for_source");

    match compiler.kernels.flash_attention_context {
        Some(ctx) => (true, Some(ctx.config.num_sink_tokens)),
        None => (false, None),
    }
}

/// Sprint 2 cycle-5 (paper §4.3 silent-gap closure) observation helper:
/// fallible variant of [`flash_attention_sink_context_for_source`] that
/// returns the compile error string instead of panicking. Used by the
/// integration test to assert that `@attention_sink(tokens=N)` with
/// `N > 0` is REFUSED at codegen — the cycle-4 v0 API surface left a
/// silent correctness gap (decorator parsed, config set, but no SMEM
/// emission → user output was rope-effectively-off with no warning).
///
/// On success returns `Ok((context_set, num_sink_tokens_flag))` with
/// the same semantics as the panicking helper. On compile failure
/// returns `Err(error_message)` for substring assertions.
pub fn try_flash_attention_sink_context_for_source(
    src: &str,
) -> Result<(bool, Option<u32>), String> {
    use nsl_errors::FileId;

    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let stmts = parsed.module.stmts.clone();
    let type_map: TypeMap = TypeMap::new();
    let opts = crate::CompileOptions {
        target: "sm_80".to_string(),
        ..Default::default()
    };

    let mut compiler = crate::compiler::Compiler::new(&interner, &type_map, &opts)
        .map_err(|e| format!("Compiler::new failed: {e}"))?;

    compiler
        .compile_flash_attention_kernels(&stmts)
        .map_err(|e| format!("{e}"))?;

    Ok(match compiler.kernels.flash_attention_context {
        Some(ctx) => (true, Some(ctx.config.num_sink_tokens)),
        None => (false, None),
    })
}
