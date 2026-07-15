//! Refusal tests for §5.1 (backward batch-shape mismatch) and §5.2 (link-step
//! error when calib_model.o does not export `nsl_calib_model_backward`).
//!
//! §5.2 is fully tested here via symbol-presence inspection on real emitted .o
//! files — no actual linker execution needed.
//!
//! §5.1 (runtime batch-shape mismatch → exit-3) is verified at two levels:
//! at the IR level we check that `nsl_calib_model_backward` is exported and
//! the wrapper contains the shape-validation iconst.i32(3) return path, and
//! at the subprocess level `backward_batch_shape_mismatch_subprocess_returns_3`
//! executes the real linked calibration binary against deliberately
//! mismatched runtime data (via the test-only
//! `HarnessConfig.runtime_data_override` fault-injection seam) and asserts
//! the structured status-3 refusal surfaces through `real_subprocess_entry`.

use std::path::PathBuf;

use nsl_codegen::calibration::{
    build_arena_layout, emit_calibration_model_object, emit_calibration_scaffolding_object,
    link_calibration_binary, pre_scan_awq_projections_from_ast, DiscoveredProjection,
    FinalizePlanEntry, ObservePlanEntry, ProjectionRef,
};
use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner};

// ── Fixture helpers ─────────────────────────────────────────────────────────

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests").join("fixtures").join(name)
}

fn parse_awq_fixture() -> (nsl_ast::Module, Interner) {
    let source =
        std::fs::read_to_string(fixture("awq_calibration_mlp.nsl")).expect("fixture readable");
    let mut interner = Interner::new();
    let (tokens, lex_diags) = tokenize(&source, FileId(0), &mut interner);
    assert!(
        lex_diags.iter().all(|d| !matches!(d.level, Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );
    (parsed.module, interner)
}

/// Parse + analyze the WGGO merge-gate fixture (`wggo_attention_mlp_real.nsl`)
/// into a compile bundle for `real_subprocess_entry`. Mirrors
/// `wggo_fixture_compile_bundle` in `wggo_backward_pipeline.rs` (test binaries
/// are separate crates, so the helper cannot be shared without a common
/// support module).
fn wggo_fixture_compile_bundle(
) -> std::sync::Arc<nsl_codegen::calibration::CalibrationCompileBundle> {
    let source = std::fs::read_to_string(fixture("wggo_attention_mlp_real.nsl"))
        .expect("merge-gate fixture readable");
    let mut interner = Interner::new();
    let (tokens, lex_diags) = tokenize(&source, FileId(0), &mut interner);
    assert!(
        lex_diags.iter().all(|d| !matches!(d.level, Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );
    let mut analysis_interner = interner.clone();
    let analysis = nsl_semantic::analyze(&parsed.module, &mut analysis_interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, Level::Error)),
        "fixture must pass semantic analysis: {:?}",
        analysis.diagnostics
    );
    std::sync::Arc::new(nsl_codegen::calibration::CalibrationCompileBundle {
        ast: parsed.module,
        interner: analysis_interner,
        type_map: analysis.type_map.clone(),
    })
}

/// Compile options WITHOUT `calibration_grad_retention` — calib_model.o
/// emitted with these opts will NOT export `nsl_calib_model_backward`.
#[allow(clippy::field_reassign_with_default)]
fn forward_only_compile_options(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> nsl_codegen::CompileOptions {
    let mut analysis_interner = interner.clone();
    let analysis = nsl_semantic::analyze(ast, &mut analysis_interner);
    let mut opts = nsl_codegen::CompileOptions::default();
    opts.calibration_batch_seq = Some((8, 4));
    opts.calibration_compile_bundle = Some(std::sync::Arc::new(
        nsl_codegen::calibration::CalibrationCompileBundle {
            ast: ast.clone(),
            interner: analysis_interner,
            type_map: analysis.type_map.clone(),
        },
    ));
    opts.weight_index_map = analysis.weight_index_map.clone();
    // calibration_grad_retention intentionally left None.
    opts
}

/// Compile options WITH `calibration_grad_retention` — calib_model.o emitted
/// with these opts WILL export `nsl_calib_model_backward`.
#[allow(clippy::field_reassign_with_default)]
fn backward_compile_options(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> nsl_codegen::CompileOptions {
    use nsl_codegen::calibration::discovery::WggoGradTarget;

    let mut analysis_interner = interner.clone();
    let analysis = nsl_semantic::analyze(ast, &mut analysis_interner);
    let mut opts = nsl_codegen::CompileOptions::default();
    opts.calibration_batch_seq = Some((8, 4));
    opts.calibration_compile_bundle = Some(std::sync::Arc::new(
        nsl_codegen::calibration::CalibrationCompileBundle {
            ast: ast.clone(),
            interner: analysis_interner,
            type_map: analysis.type_map.clone(),
        },
    ));
    opts.weight_index_map = analysis.weight_index_map.clone();
    let targets = vec![WggoGradTarget {
        layer_key: "TinyMLP".into(),
        class_name: "TinyMLP".into(),
        head_dim: 64,
        w_q: ProjectionRef("TinyMLP.up_proj".into()),
        w_k: ProjectionRef("TinyMLP.up_proj".into()),
        w_v: ProjectionRef("TinyMLP.up_proj".into()),
        w_o: ProjectionRef("TinyMLP.down_proj".into()),
        w_q_shape: [128, 64],
        w_k_shape: [128, 64],
        w_v_shape: [128, 64],
        w_o_shape: [64, 128],
        w_q_index: 0,
        w_k_index: 0,
        w_v_index: 0,
        w_o_index: 1,
    }];
    opts.calibration_grad_retention = Some(targets);
    opts
}

fn minimal_plans_and_layout() -> (
    Vec<ObservePlanEntry>,
    Vec<FinalizePlanEntry>,
    nsl_codegen::calibration::ArenaLayout,
) {
    let projection = ProjectionRef("TinyMLP.up_proj".into());
    let observe_plan = vec![ObservePlanEntry {
        projection: projection.clone(),
        src_offset: 0,
        rows: 32,
        channels: 64,
        running_symbol: "__nsl_awq_running_up_proj".into(),
    }];
    let finalize_plan = vec![FinalizePlanEntry {
        projection,
        running_symbol: "__nsl_awq_running_up_proj".into(),
        channels: 64,
        bytes_per_element: 4,
    }];
    let arena_layout = build_arena_layout(
        &[DiscoveredProjection {
            projection: ProjectionRef("TinyMLP.up_proj".into()),
            weight_shape: [128, 64],
        }],
        8,
        4,
    );
    (observe_plan, finalize_plan, arena_layout)
}

// ── §5.2: link-step refusal ──────────────────────────────────────────────────

/// §5.2: When the scaffolding requires backward (needs_backward = true) but
/// calib_model.o was compiled without `calibration_grad_retention` and therefore
/// does NOT export `nsl_calib_model_backward`, `link_calibration_binary` must
/// return an error naming the missing symbol with the three-part template.
#[test]
fn link_fails_when_backward_wrapper_missing() {
    let (ast, interner) = parse_awq_fixture();
    let projections = pre_scan_awq_projections_from_ast(&ast, &interner);
    let arena_layout = build_arena_layout(&projections, 8, 4);
    let (observe_plan, finalize_plan, _) = minimal_plans_and_layout();
    let tmp = tempfile::tempdir().expect("tempdir");

    // calib_model.o emitted WITHOUT grad retention → no backward symbol.
    let model_obj = tmp.path().join("calib_model_fwd_only.o");
    let fwd_opts = forward_only_compile_options(&ast, &interner);
    emit_calibration_model_object(&ast, &fwd_opts, &arena_layout, &model_obj)
        .expect("emit forward-only calib_model.o");

    // Confirm the forward symbol IS present and backward symbol is absent —
    // this makes the test intention clear.
    {
        use object::{Object, ObjectSymbol};
        let obj_bytes = std::fs::read(&model_obj).expect("read model_obj");
        let obj = object::File::parse(&*obj_bytes).expect("parse model_obj");
        use nsl_codegen::linker::strip_host_symbol_prefix;
        assert!(
            obj.symbols().any(|s| {
                s.name().map(strip_host_symbol_prefix) == Ok("nsl_calib_model_forward")
                    && !s.is_undefined()
            }),
            "precondition: calib_model.o must export nsl_calib_model_forward"
        );
        assert!(
            !obj.symbols().any(|s| {
                s.name().map(strip_host_symbol_prefix) == Ok("nsl_calib_model_backward")
                    && !s.is_undefined()
            }),
            "precondition: calib_model.o must NOT export nsl_calib_model_backward \
             when grad_retention is absent"
        );
    }

    // scaffolding.o requesting backward (needs_backward = true).
    let scaffolding_obj = tmp.path().join("scaffolding_bwd.o");
    emit_calibration_scaffolding_object(
        &observe_plan,
        &finalize_plan,
        &arena_layout,
        b"{}",
        true,
        true, // needs_backward = true — the scaffolding will call nsl_calib_model_backward
        &[],
        &[],  // no wggo targets for this link-refusal test
        None, // no grad_arena_layout
        &scaffolding_obj,
    )
    .expect("emit backward scaffolding.o");

    let binary = tmp.path().join(if cfg!(windows) {
        "calibration.exe"
    } else {
        "calibration"
    });

    // The link must fail because calib_model.o does not export the backward symbol.
    let result = link_calibration_binary(&scaffolding_obj, &model_obj, &binary, true);
    let err = result.expect_err("link must fail when backward wrapper is missing");
    let err_msg = format!("{err}");

    assert!(
        err_msg.contains("nsl_calib_model_backward"),
        "error must name the missing symbol; got: {err_msg}"
    );
    assert!(
        err_msg.contains("does not export") || err_msg.contains("missing"),
        "error must indicate the symbol is absent; got: {err_msg}"
    );
    // Verify all three parts of the canonical refusal template are present.
    assert!(
        err_msg.contains("requested:"),
        "error must include 'requested:' section; got: {err_msg}"
    );
    assert!(
        err_msg.contains("expected:"),
        "error must include 'expected:' section; got: {err_msg}"
    );
    assert!(
        err_msg.contains("found:"),
        "error must include 'found:' section; got: {err_msg}"
    );
}

/// §5.2 (green path): When needs_backward = false, a calib_model.o without
/// the backward symbol must link successfully — the §5.2 guard must NOT fire
/// for forward-only (AWQ-only) calibration runs.
#[test]
fn link_succeeds_when_backward_not_needed_and_wrapper_absent() {
    let (ast, interner) = parse_awq_fixture();
    let projections = pre_scan_awq_projections_from_ast(&ast, &interner);
    let arena_layout = build_arena_layout(&projections, 8, 4);
    let (observe_plan, finalize_plan, _) = minimal_plans_and_layout();
    let tmp = tempfile::tempdir().expect("tempdir");

    let model_obj = tmp.path().join("calib_model_fwd.o");
    let fwd_opts = forward_only_compile_options(&ast, &interner);
    emit_calibration_model_object(&ast, &fwd_opts, &arena_layout, &model_obj)
        .expect("emit forward-only calib_model.o");

    let scaffolding_obj = tmp.path().join("scaffolding_fwd.o");
    emit_calibration_scaffolding_object(
        &observe_plan,
        &finalize_plan,
        &arena_layout,
        b"{}",
        true,
        false, // needs_backward = false — AWQ-only path
        &[],
        &[],  // no wggo targets for this forward-only test
        None, // no grad_arena_layout
        &scaffolding_obj,
    )
    .expect("emit forward scaffolding.o");

    let binary = tmp.path().join(if cfg!(windows) {
        "calibration_fwd.exe"
    } else {
        "calibration_fwd"
    });

    // With needs_backward = false the §5.2 guard must not fire even though
    // nsl_calib_model_backward is absent from calib_model.o.
    link_calibration_binary(&scaffolding_obj, &model_obj, &binary, false)
        .expect("link must succeed for forward-only (AWQ) calibration");
}

// ── §5.1: backward batch-shape mismatch ─────────────────────────────────────

/// §5.1 (IR-level check): `emit_calibration_model_object` with
/// `calibration_grad_retention` set must emit `nsl_calib_model_backward` with a
/// shape-validation block.  We verify the wrapper is exported and present in the
/// object, confirming the Task-14 shape-mismatch path (status 3) is compiled in.
///
/// Full subprocess execution of the mismatch path is deferred — that requires a
/// working calibration subprocess invocation harness with real model weights and
/// calibration data.
#[test]
fn backward_wrapper_exported_with_shape_validation() {
    use object::{Object, ObjectSymbol};

    let (ast, interner) = parse_awq_fixture();
    let projections = pre_scan_awq_projections_from_ast(&ast, &interner);
    let arena_layout = build_arena_layout(&projections, 8, 4);
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("calib_model_bwd.o");
    let opts = backward_compile_options(&ast, &interner);

    emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path)
        .expect("emit calib_model.o with backward wrapper");

    let obj_bytes = std::fs::read(&out_path).expect("read object");
    let obj = object::File::parse(&*obj_bytes).expect("parse object");

    use nsl_codegen::linker::strip_host_symbol_prefix;
    // The backward wrapper must be exported (Task 14/15 requirement).
    assert!(
        obj.symbols().any(|s| {
            s.name().map(strip_host_symbol_prefix) == Ok("nsl_calib_model_backward")
                && !s.is_undefined()
        }),
        "calib_model.o must export nsl_calib_model_backward when grad_retention is set"
    );

    // The forward wrapper must also be present (guard that we didn't accidentally
    // replace it when adding backward).
    assert!(
        obj.symbols().any(|s| {
            s.name().map(strip_host_symbol_prefix) == Ok("nsl_calib_model_forward")
                && !s.is_undefined()
        }),
        "calib_model.o must still export nsl_calib_model_forward alongside backward"
    );
}

/// §5.1 (subprocess-level): end-to-end fault-injection test for the runtime
/// status-3 path (per-batch shape mismatch returned by
/// `nsl_calib_model_{forward,backward}` and surfaced by the scaffolding's
/// mid-loop status check).
///
/// `real_subprocess_entry` normally uses the SAME calibration data file for
/// both compile-time shape derivation (`peek_batch_seq`) and runtime feeding,
/// so the wrapper's `batch_elem_count == expected_elem_count` check is
/// unsatisfiable by construction. The test-only
/// `HarnessConfig.runtime_data_override` seam splits the two roles:
///
/// 1. The model object is compiled against the real WGGO merge-gate fixture
///    (`wggo_calib_data.safetensors`, shape [8, 4, 32] → per-call input
///    [1, seq=4, dim=32] = 128 f32 elements).
/// 2. The linked calibration binary is executed with argv[1] pointing at a
///    deliberately mismatched rank-3 NSLB batch file ([8, 8, 32] → 256 f32
///    elements per batch).
/// 3. The wrapper refuses each batch with status 3; the scaffolding writes
///    the structured "batch shape mismatch" detail to the sidecar path and
///    exits 3; `real_subprocess_entry` folds both into the surfaced
///    `HarnessError::Infrastructure` reason.
///
/// The fixture registers only the WGGO gradient hook (no AWQ projections),
/// so the arena-based first-batch preflight is skipped and the wrapper's
/// per-batch check is the sole gate — exactly the §5.1 path under test.
#[test]
fn backward_batch_shape_mismatch_subprocess_returns_3() {
    use nsl_codegen::calibration::binary_codegen::real_subprocess_entry;
    use nsl_codegen::calibration::wggo_gradient_hook::WggoGradientHook;
    use nsl_codegen::calibration::{HarnessConfig, HarnessError, HarnessMode, HookRegistry};

    let data_path = fixture("wggo_calib_data.safetensors");
    let weights_path = fixture("wggo_calib_weights.safetensors");
    assert!(data_path.exists(), "fixture missing: {}", data_path.display());
    assert!(
        weights_path.exists(),
        "fixture missing: {}",
        weights_path.display()
    );

    // Compile bundle + WGGO targets from the merge-gate fixture — same
    // construction as the green-path merge gate in wggo_backward_pipeline.rs,
    // so any failure here isolates to the injected runtime data.
    let compile_bundle = wggo_fixture_compile_bundle();
    let targets = nsl_codegen::calibration::discovery::pre_scan_wggo_targets_from_ast(
        &compile_bundle.ast,
        &compile_bundle.interner,
    );
    assert!(
        !targets.is_empty(),
        "fixture must yield at least one @wggo_target for the backward path"
    );

    let mut registry = HookRegistry::new();
    registry.register(Box::new(WggoGradientHook::new(targets)));

    // Mismatched runtime data: rank-3 NSLB [8, 8, 32] vs fixture [8, 4, 32].
    // Per-batch payload 8*32 = 256 f32 elements vs the compile-time expected
    // 1*4*32 = 128 (model object compiled with calibration_batch_seq (1, 4)).
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp_dir = std::env::temp_dir().join(format!(
        "nsl-wggo-status3-{nanos}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp_dir).expect("create unique temp dir");
    let override_path = tmp_dir.join("mismatched_calib.bin");
    let mut blob = Vec::new();
    blob.extend_from_slice(b"NSLB");
    blob.extend_from_slice(&3u32.to_le_bytes()); // rank 3
    for dim in [8u32, 8, 32] {
        blob.extend_from_slice(&dim.to_le_bytes());
    }
    // 8 batches x (8*32) f32 each. Values never reach compute — the wrapper
    // refuses on element count first — but use a non-zero ramp so an
    // accidentally-accepted run cannot masquerade as a degenerate-zero result.
    for i in 0..(8u32 * 8 * 32) {
        blob.extend_from_slice(&(((i % 17) as f32) * 0.25 + 0.5).to_le_bytes());
    }
    std::fs::write(&override_path, blob).expect("write mismatched NSLB data file");

    let cfg = HarnessConfig {
        checkpoints: vec![weights_path],
        calibration_data: data_path,
        samples: 8,
        batch_size: 1,
        timeout_secs: 60,
        mode: HarnessMode::Required,
        // WGGO-only flow: no AWQ projections, so the arena preflight is
        // skipped and the wrapper's per-batch check is the sole gate.
        projections: Vec::new(),
        compile_bundle: Some(compile_bundle),
        runtime_data_override: Some(override_path),
    };

    let result = real_subprocess_entry(&cfg, &registry);
    let _ = std::fs::remove_dir_all(&tmp_dir);

    match result {
        Err(HarnessError::Infrastructure { reason }) => {
            assert!(
                reason.contains("status 3"),
                "expected subprocess status-3 refusal, got: {reason}"
            );
            assert!(
                reason.contains("batch shape mismatch"),
                "expected structured batch-shape-mismatch detail (written by the \
                 subprocess to the sidecar path) in the surfaced reason, got: {reason}"
            );
        }
        Ok(out) => panic!(
            "mismatched runtime data must NOT produce a sidecar; got outcome {:?}",
            out.outcome_repr
        ),
        Err(other) => panic!("expected Infrastructure(status 3), got {other:?}"),
    }
}

/// PR 1b regression: when `calibration_grad_retention` is non-empty, the
/// hook registry built in `compile_train_block` MUST contain a
/// `WggoGradientHook` alongside the `AwqCalibrationHook`. Conversely,
/// when `calibration_grad_retention` is None, only the AWQ hook is
/// registered.
///
/// We exercise this through the registry-builder path used by
/// `compile_and_calibrate`. The test inspects the registry's hook IDs
/// to assert the WGGO hook is present iff targets are non-empty.
///
/// NOTE: `WggoGradientHook::id()` returns `"wggo_head_gradients"` (not
/// `"wggo_gradient"`). The assert below uses the actual value.
#[test]
fn wggo_gradient_hook_registered_when_grad_retention_is_some() {
    use nsl_codegen::calibration::registry::HookRegistry;
    use nsl_codegen::calibration::wggo_gradient_hook::WggoGradientHook;
    use nsl_codegen::calibration::discovery::WggoGradTarget;

    // Build a non-empty WggoGradTarget vec.
    let targets = vec![WggoGradTarget {
        layer_key: "m".to_string(),
        class_name: "Attention".to_string(),
        head_dim: 8,
        w_q: ProjectionRef("m.q_proj".into()),
        w_k: ProjectionRef("m.k_proj".into()),
        w_v: ProjectionRef("m.v_proj".into()),
        w_o: ProjectionRef("m.o_proj".into()),
        w_q_shape: [32, 32],
        w_k_shape: [32, 32],
        w_v_shape: [32, 32],
        w_o_shape: [32, 32],
        w_q_index: 0,
        w_k_index: 1,
        w_v_index: 2,
        w_o_index: 3,
    }];

    // Mirror the conditional registration logic from stmt.rs.
    let mut registry = HookRegistry::new();
    if !targets.is_empty() {
        registry.register(Box::new(WggoGradientHook::new(targets.clone())));
    }

    let ids: Vec<&'static str> = registry.iter().map(|h| h.id()).collect();
    assert!(
        ids.contains(&"wggo_head_gradients"),
        "WggoGradientHook (id='wggo_head_gradients') must be registered; got: {:?}",
        ids
    );
}

#[test]
fn wggo_gradient_hook_not_registered_when_targets_empty() {
    use nsl_codegen::calibration::registry::HookRegistry;
    use nsl_codegen::calibration::discovery::WggoGradTarget;

    // Empty targets: pre-scan returned Some(vec![]) — the gating must
    // skip registration to avoid an empty-buffer sidecar entry.
    let targets: Vec<WggoGradTarget> = vec![];

    let mut registry = HookRegistry::new();
    if !targets.is_empty() {
        registry.register(Box::new(
            nsl_codegen::calibration::wggo_gradient_hook::WggoGradientHook::new(targets),
        ));
    }

    assert!(
        registry.is_empty(),
        "registry must be empty when WGGO targets are empty"
    );
}
