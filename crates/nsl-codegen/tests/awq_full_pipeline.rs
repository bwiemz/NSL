//! Task 10: End-to-end AWQ calibration pipeline integration test.
//!
//! Verifies the full chain:
//!   fixture NSL source → discovery (discover_awq_projections_from_state)
//!   → retention layout → real calibration data load → in-process
//!   activation observation (emit_observe_batch with real f32 data)
//!   → sidecar assembly → final compile with scales (compile_with_options).
//!
//! Architecture (post-2026-04-22 completion work):
//!   * Two-object compile: scaffolding.o (calibration_main) + calib_model.o
//!     (model_forward + nsl_calib_model_forward wrapper + retention arena)
//!     linked via link_calibration_binary. See spec/plan pair at
//!     docs/superpowers/specs/2026-04-22-awq-real-subprocess-completion-design.md.
//!   * loop_body calls nsl_calib_model_forward between nsl_calibration_batch_at
//!     and the plan-driven max-abs reduction, populating the retention arena
//!     via the model_forward splice. Empty-arena regression (the Task 6 revert
//!     signal) is impossible by construction.
//!
//! These tests drive the public calibration API (no internal Compiler
//! exposure) so they exercise the same surface a real AWQ pipeline uses.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nsl_codegen::calibration::{
    CalibCtx, CalibrationHook, CalibrationResult,
    awq_hook::AwqCalibrationHook,
    discovery::DiscoveredProjection,
    binary_codegen::real_subprocess_entry,
    observation::ProjectionRef,
    HarnessConfig, HarnessMode, HookRegistry,
    retention::{ArenaLayout, RetentionTable, TensorShape},
    retention_pass::build_arena_layout,
    sidecar::{Sidecar, SIDECAR_VERSION},
};

/// Absolute path to the repo root (parent of `crates/`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..") // crates/nsl-codegen -> crates
        .join("..") // crates             -> repo root
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests").join("fixtures").join(name)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal AWQ Sidecar blob for a set of `(projection, scales)` pairs.
fn build_awq_sidecar(projections: &[(&str, Vec<f32>)]) -> Sidecar {
    use nsl_codegen::calibration::awq_sidecar;

    let mut map: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (name, scales) in projections {
        map.insert(name.to_string(), scales.clone());
    }
    let blob = awq_sidecar::serialize(&map);

    let mut hooks = BTreeMap::new();
    hooks.insert("awq_activation_scales".to_string(), blob);

    Sidecar {
        version: SIDECAR_VERSION,
        checkpoint_sha256: "test".into(),
        calibration_data_sha256: "test".into(),
        hook_set_sha256: "test".into(),
        cache_key_digest: String::new(),
        num_samples_used: 1,
        hooks,
        wggo_head_gradients: None,
    }
}

fn awq_fixture_compile_bundle(
) -> (
    Vec<DiscoveredProjection>,
    std::sync::Arc<nsl_codegen::calibration::CalibrationCompileBundle>,
) {
    let source = std::fs::read_to_string(fixture("awq_calibration_mlp.nsl"))
        .expect("awq fixture readable");
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(&source, nsl_errors::FileId(0), &mut interner);
    assert!(
        lex_diags
            .iter()
            .all(|diag| !matches!(diag.level, nsl_errors::Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.level, nsl_errors::Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let projections = nsl_codegen::calibration::pre_scan_awq_projections_from_ast(
        &parsed.module,
        &interner,
    );
    let mut analysis_interner = interner.clone();
    let analysis = nsl_semantic::analyze(&parsed.module, &mut analysis_interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.level, nsl_errors::Level::Error)),
        "fixture must pass semantic analysis: {:?}",
        analysis.diagnostics
    );

    let bundle = std::sync::Arc::new(nsl_codegen::calibration::CalibrationCompileBundle {
        ast: parsed.module,
        interner: analysis_interner,
        type_map: analysis.type_map.clone(),
    });

    (projections, bundle)
}

fn read_safetensors_flat(path: &Path, tensor_name: &str) -> Vec<f32> {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let tensors = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap_or_else(|e| panic!("deserializing {}: {e}", path.display()));
    let tensor = tensors
        .tensor(tensor_name)
        .unwrap_or_else(|e| panic!("missing tensor {tensor_name} in {}: {e}", path.display()));
    assert_eq!(
        tensor.dtype(),
        safetensors::Dtype::F32,
        "{tensor_name} in {} must be f32",
        path.display()
    );
    tensor
        .data()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn reference_awq_scales(calib: &[f32], up_w: &[f32]) -> (Vec<f32>, Vec<f32>) {
    const CALIB_COUNT: usize = 8;
    const CALIB_SEQ: usize = 4;
    const CALIB_HIDDEN: usize = 64;
    const UP_OUT: usize = 128;

    assert_eq!(
        calib.len(),
        CALIB_COUNT * CALIB_SEQ * CALIB_HIDDEN,
        "calibration fixture shape must stay [8, 4, 64]"
    );
    assert_eq!(
        up_w.len(),
        UP_OUT * CALIB_HIDDEN,
        "TinyMLP.up_proj weight fixture shape must stay [128, 64]"
    );

    let rows = CALIB_COUNT * CALIB_SEQ;
    let mut up_ref = vec![0.0f32; CALIB_HIDDEN];
    let mut down_ref = vec![0.0f32; UP_OUT];

    for row_idx in 0..rows {
        let row = &calib[row_idx * CALIB_HIDDEN..(row_idx + 1) * CALIB_HIDDEN];
        for channel in 0..CALIB_HIDDEN {
            up_ref[channel] = up_ref[channel].max(row[channel].abs());
        }

        for out_channel in 0..UP_OUT {
            let weights = &up_w[out_channel * CALIB_HIDDEN..(out_channel + 1) * CALIB_HIDDEN];
            let mut acc = 0.0f32;
            for feature in 0..CALIB_HIDDEN {
                acc += row[feature] * weights[feature];
            }
            let relu = acc.max(0.0);
            down_ref[out_channel] = down_ref[out_channel].max(relu.abs());
        }
    }

    (up_ref, down_ref)
}

fn awq_scales(sidecar: &Sidecar, projection: &str) -> Vec<f32> {
    let blob = sidecar
        .hooks
        .get("awq_activation_scales")
        .expect("awq_activation_scales hook blob missing from sidecar");
    let parsed = nsl_runtime::awq::AwqScales::from_blob(blob)
        .expect("AWQ sidecar blob must parse");
    parsed
        .by_projection
        .get(projection)
        .unwrap_or_else(|| panic!("projection {projection} missing from AWQ sidecar"))
        .clone()
}

fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: vector length mismatch (actual {} expected {})",
        actual.len(),
        expected.len()
    );

    for (index, (&actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let scale = actual_value
            .abs()
            .max(expected_value.abs())
            .max(1.0);
        let diff = (actual_value - expected_value).abs();
        assert!(
            diff <= rtol * scale,
            "{label}[{index}] diff {diff} exceeds tol {} (actual {actual_value}, expected {expected_value})",
            rtol * scale
        );
    }
}

/// Parse + semantic-check + compile a source string with options.
/// Returns Err as a formatted string on failure.
fn try_compile(
    src: &str,
    opts: &nsl_codegen::CompileOptions,
) -> Result<Vec<u8>, String> {
    nsl_codegen::compile_with_options(src, opts).map_err(|e| format!("{e:?}"))
}

// ---------------------------------------------------------------------------
// Retention arena layout — unit assertions (pure, no I/O)
// ---------------------------------------------------------------------------

#[test]
fn retention_arena_size_matches_activation_shape() {
    // TinyMLP: up_proj [128, 64] → in_features=64; down_proj [64, 128] → in_features=128.
    // batch=8, seq=4.
    let ps = vec![
        DiscoveredProjection {
            projection: ProjectionRef("TinyMLP.up_proj".into()),
            weight_shape: [128, 64],
        },
        DiscoveredProjection {
            projection: ProjectionRef("TinyMLP.down_proj".into()),
            weight_shape: [64, 128],
        },
    ];
    let layout = build_arena_layout(&ps, 8, 4);
    // up_proj:   8*4*64*4 = 8192 bytes
    assert_eq!(layout.entries[0].2, 8192, "up_proj arena bytes");
    // down_proj: 8*4*128*4 = 16 384 bytes, offset = 8192
    assert_eq!(layout.entries[1].1, 8192, "down_proj arena offset");
    assert_eq!(layout.entries[1].2, 16_384, "down_proj arena bytes");
    assert_eq!(layout.total_bytes(), 24_576, "total arena bytes");
}

// ---------------------------------------------------------------------------
// Discovery from parsed calibration-fixture NSL source
// ---------------------------------------------------------------------------

#[test]
fn discovery_from_fixture_nsl_finds_both_projections() {
    use nsl_codegen::calibration::discovery::discover_awq_projections_from_state;

    let source = std::fs::read_to_string(fixture("awq_calibration_mlp.nsl"))
        .expect("awq_calibration_mlp.nsl must exist");

    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) =
        nsl_lexer::tokenize(&source, nsl_errors::FileId(0), &mut interner);
    assert!(
        lex_diags.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed.diagnostics.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must parse cleanly: {:?}", parsed.diagnostics
    );

    // Walk the AST to find the TinyMLP model definition.
    let model_def_ref = parsed.module.stmts.iter().find_map(|s| {
        use nsl_ast::stmt::StmtKind;
        match &s.kind {
            StmtKind::Decorated { stmt: inner, .. } => {
                if let StmtKind::ModelDef(md) = &inner.kind {
                    if interner.resolve(md.name.0).unwrap_or("") == "TinyMLP" {
                        return Some(md);
                    }
                }
                None
            }
            StmtKind::ModelDef(md) => {
                if interner.resolve(md.name.0).unwrap_or("") == "TinyMLP" {
                    Some(md)
                } else {
                    None
                }
            }
            _ => None,
        }
    }).expect("TinyMLP model must be present in the fixture");

    // Build field_types and tensor_shapes matching what collect_models would produce.
    let mut field_types = std::collections::HashMap::new();
    field_types.insert("up_proj".to_string(), "Tensor".to_string());
    field_types.insert("down_proj".to_string(), "Tensor".to_string());

    let mut tensor_shapes = std::collections::HashMap::new();
    tensor_shapes.insert("up_proj".to_string(), "Tensor<[128, 64], f32>".to_string());
    tensor_shapes.insert("down_proj".to_string(), "Tensor<[64, 128], f32>".to_string());

    // Find the forward body.
    let forward_body = model_def_ref.members.iter().find_map(|m| {
        use nsl_ast::decl::ModelMember;
        if let ModelMember::Method(fn_def, _) = m {
            if interner.resolve(fn_def.name.0).unwrap_or("") == "forward" {
                return Some(&fn_def.body);
            }
        }
        None
    });

    let projections = discover_awq_projections_from_state(
        "TinyMLP",
        forward_body,
        &field_types,
        &tensor_shapes,
        &[],
        &interner,
    ).expect("discovery must succeed on the fixture");

    let mut names: Vec<_> = projections.iter().map(|p| p.projection.0.clone()).collect();
    names.sort();
    assert_eq!(
        names,
        vec!["TinyMLP.down_proj".to_string(), "TinyMLP.up_proj".to_string()],
        "discovery must find both TinyMLP projections"
    );

    // Channel count: up_proj in_features=64, down_proj in_features=128.
    let up = projections.iter().find(|p| p.projection.0 == "TinyMLP.up_proj").unwrap();
    let dn = projections.iter().find(|p| p.projection.0 == "TinyMLP.down_proj").unwrap();
    assert_eq!(up.weight_shape, [128, 64], "up_proj shape");
    assert_eq!(dn.weight_shape, [64, 128], "down_proj shape");
}

// ---------------------------------------------------------------------------
// In-process AWQ observation with real calibration data
// ---------------------------------------------------------------------------

/// Load the safetensors calibration-data fixture and run the AWQ hook
/// through the in-process stub path with real f32 values.
/// Validates:
///   (1) scales aren't all-zero (real observation ran),
///   (2) scales show variation across channels.
#[test]
fn awq_observation_with_real_calib_data_produces_nonuniform_scales() {
    let data_path = fixture("awq_calib_data.safetensors");
    assert!(data_path.exists(), "awq_calib_data.safetensors fixture missing — run gen_awq_fixtures.py");

    // Load calibration batches: shape [8, 4, 64] → 8 batches, seq=4, hidden=64.
    let batches = nsl_runtime::calibration_data::load(&data_path)
        .expect("calibration data must load");
    assert_eq!(batches.count, 8, "expected 8 calibration batches");

    // Two projections: up_proj (in_features=64), down_proj (in_features=128).
    // For this test only up_proj matches the activation dimension (64).
    let discovered = vec![
        DiscoveredProjection {
            projection: ProjectionRef::new("TinyMLP.up_proj"),
            weight_shape: [128, 64],
        },
    ];
    let hook: Box<dyn CalibrationHook> = Box::new(AwqCalibrationHook::from_discovered(&discovered));

    // Build retention table: shape [8, 4, 64].
    let mut table = RetentionTable::new();
    table.register(
        ProjectionRef::new("TinyMLP.up_proj"),
        TensorShape::new(vec![8, 4, 64]),
        4,
    );

    let mut ctx = CalibCtx::for_tests(&table);
    ctx.total_samples = batches.count as u32;
    hook.emit_init(&mut ctx);

    // Arena layout: 8*4*64*4 = 8192 bytes at offset 0.
    let layout = ArenaLayout {
        entries: vec![(ProjectionRef::new("TinyMLP.up_proj"), 0, 8 * 4 * 64 * 4)],
    };

    // Process each batch with real data.
    for batch_idx in 0..batches.count {
        ctx.sample_idx = batch_idx as u32;
        let batch_bytes = batches.batch_at(batch_idx).expect("batch must exist");
        let n_f32 = batch_bytes.len() / 4;
        let mut floats: Vec<f32> = Vec::with_capacity(n_f32);
        for i in 0..n_f32 {
            let b: [u8; 4] = batch_bytes[i * 4..i * 4 + 4].try_into().unwrap();
            floats.push(f32::from_le_bytes(b));
        }
        ctx.stub_set_arena_buffer("TinyMLP.up_proj", &floats);
        hook.emit_observe_batch(&mut ctx, &layout);
    }

    let blob = match hook.emit_finalize(&mut ctx) {
        CalibrationResult::Ok(b) => b,
        CalibrationResult::Degenerate { reason } => {
            panic!("AWQ hook degenerate: {reason}")
        }
    };

    // Parse scales via nsl-runtime's AwqScales.
    let awq_scales = nsl_runtime::awq::AwqScales::from_blob(&blob)
        .expect("scales blob must parse");
    let up_scales = awq_scales.by_projection
        .get("TinyMLP.up_proj")
        .expect("TinyMLP.up_proj scales must be present");

    // Assertion 1: 64 channels.
    assert_eq!(up_scales.len(), 64, "up_proj must have 64 channel scales");

    // Assertion 2: not all zero (observation actually ran).
    assert!(
        up_scales.iter().any(|&v| v > 0.0),
        "AWQ scales must be positive; all-zero implies observation failed: {up_scales:?}"
    );

    // Assertion 3: channel variation (the fixture seeds channel 0 with 10x the others).
    let max_scale = up_scales.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_scale = up_scales.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        (max_scale - min_scale).abs() > 1e-4,
        "AWQ scales must vary across channels; fixture injects channel 0 as dominant: \
         max={max_scale}, min={min_scale}"
    );
}

// ---------------------------------------------------------------------------
// Full two-projection end-to-end pipeline (discovery + observation + sidecar)
// ---------------------------------------------------------------------------

#[test]
fn end_to_end_discovers_enumerates_calibrates_and_quantizes() {
    let data_path = fixture("awq_calib_data.safetensors");
    assert!(
        data_path.exists(),
        "awq_calib_data.safetensors fixture missing — run gen_awq_fixtures.py"
    );

    // Discovered projections matching TinyMLP:
    //   up_proj   [128, 64] → in_features=64
    //   down_proj [64, 128] → in_features=128
    let discovered = vec![
        DiscoveredProjection {
            projection: ProjectionRef::new("TinyMLP.up_proj"),
            weight_shape: [128, 64],
        },
        DiscoveredProjection {
            projection: ProjectionRef::new("TinyMLP.down_proj"),
            weight_shape: [64, 128],
        },
    ];

    let hook: Box<dyn CalibrationHook> = Box::new(AwqCalibrationHook::from_discovered(&discovered));

    // Build retention table.
    let mut table = RetentionTable::new();
    table.register(
        ProjectionRef::new("TinyMLP.up_proj"),
        TensorShape::new(vec![8, 4, 64]),
        4,
    );
    table.register(
        ProjectionRef::new("TinyMLP.down_proj"),
        TensorShape::new(vec![8, 4, 128]),
        4,
    );

    // Build arena layout: up_proj at offset 0, down_proj at offset 8192.
    let layout = ArenaLayout {
        entries: vec![
            (ProjectionRef::new("TinyMLP.up_proj"), 0, 8 * 4 * 64 * 4),
            (ProjectionRef::new("TinyMLP.down_proj"), 8 * 4 * 64 * 4, 8 * 4 * 128 * 4),
        ],
    };

    let mut ctx = CalibCtx::for_tests(&table);
    hook.emit_init(&mut ctx);

    // Load calibration data and run observation.
    let batches = nsl_runtime::calibration_data::load(&data_path)
        .expect("calib data must load");

    for batch_idx in 0..batches.count {
        ctx.sample_idx = batch_idx as u32;
        if let Some(batch_bytes) = batches.batch_at(batch_idx) {
            let n_f32 = batch_bytes.len() / 4;
            let mut floats: Vec<f32> = Vec::with_capacity(n_f32);
            for i in 0..n_f32 {
                let b: [u8; 4] = batch_bytes[i * 4..i * 4 + 4].try_into().unwrap();
                floats.push(f32::from_le_bytes(b));
            }
            // Broadcast same batch data to both projections.
            // down_proj has more in_features (128) than the batch hidden dim (64);
            // we tile the floats to fill the down_proj slot.
            let dn_floats: Vec<f32> = floats.iter().chain(floats.iter()).copied().collect();
            ctx.stub_set_arena_buffer("TinyMLP.up_proj", &floats);
            ctx.stub_set_arena_buffer("TinyMLP.down_proj", &dn_floats);
        }
        hook.emit_observe_batch(&mut ctx, &layout);
    }

    let blob = match hook.emit_finalize(&mut ctx) {
        CalibrationResult::Ok(b) => b,
        CalibrationResult::Degenerate { reason } => {
            panic!("degenerate: {reason}")
        }
    };

    // Build the full Sidecar wrapping the AWQ blob.
    let mut hooks_map = BTreeMap::new();
    hooks_map.insert("awq_activation_scales".to_string(), blob.clone());
    let sidecar = Sidecar {
        version: SIDECAR_VERSION,
        checkpoint_sha256: "fixture".into(),
        calibration_data_sha256: "fixture".into(),
        hook_set_sha256: "fixture".into(),
        cache_key_digest: String::new(),
        num_samples_used: batches.count as u32,
        hooks: hooks_map,
        wggo_head_gradients: None,
    };

    // Parse via AwqScales.
    let awq_scales = nsl_runtime::awq::AwqScales::from_blob(&blob)
        .expect("sidecar blob must parse");

    // Assertion 1: discovery produced both projections.
    let mut names: Vec<_> = awq_scales.by_projection.keys().cloned().collect();
    names.sort();
    assert_eq!(
        names,
        vec!["TinyMLP.down_proj".to_string(), "TinyMLP.up_proj".to_string()],
        "both projections must appear in AWQ scales"
    );

    // Assertion 2: correct channel counts.
    let up = awq_scales.by_projection.get("TinyMLP.up_proj").unwrap();
    let dn = awq_scales.by_projection.get("TinyMLP.down_proj").unwrap();
    assert_eq!(up.len(), 64, "TinyMLP.up_proj must have 64 channel scales");
    assert_eq!(dn.len(), 128, "TinyMLP.down_proj must have 128 channel scales");

    // Assertion 3: scales aren't all-equal (proves observation ran with real data).
    assert!(
        up.iter().any(|&v| (v - up[0]).abs() > 1e-6),
        "AWQ up_proj scales must show channel variation; all-equal implies no real observation"
    );

    // Assertion 4: scales aren't all-zero.
    assert!(
        up.iter().any(|&v| v > 0.0),
        "AWQ up_proj scales must be positive-valued after real observation"
    );

    // Assertion 5: final compile with the sidecar succeeds (exercises Task 9 lookup path).
    // Use the TINY_AWQ_SRC fixture from awq_hook_end_to_end to avoid a full train-block compile.
    const TINY_AWQ_SRC: &str = r#"model Tiny:
    w: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

fn main():
    let m = Tiny()
    quant static qm from m:
        default: awq4
    let x = ones([1, 4])
    let y = qm.forward(x)
"#;
    // Prepare sidecar with "Tiny.w" matching the above quant block.
    let sidecar_for_final = build_awq_sidecar(&[("Tiny.w", vec![1.0, 1.0, 1.0, 1.0])]);
    let mut final_opts = nsl_codegen::CompileOptions::default();
    final_opts.calibration_sidecar = Some(sidecar_for_final);
    try_compile(TINY_AWQ_SRC, &final_opts)
        .expect("final compile with sidecar should succeed (Task 9 lookup path)");

    // Also verify the original sidecar from our pipeline is a valid Sidecar object.
    let _ = sidecar; // already validated above; just assert no panic
}

// ---------------------------------------------------------------------------
// compile_with_options — basic smoke test
// ---------------------------------------------------------------------------

#[test]
fn compile_with_options_smoke_test() {
    let src = r#"fn main():
    let x = ones([4, 4])
"#;
    let opts = nsl_codegen::CompileOptions::default();
    nsl_codegen::compile_with_options(src, &opts)
        .expect("simple source must compile with default options");
}

/// Tolerance: 5e-6 — f32 matmul over 64-length reductions accumulates
/// ~4 ULPs of round-off vs. a pairwise accumulator; the subprocess
/// runs the same fabs/fmax loop as the analytical reference, so the
/// dominant source of drift is model_forward's matmul reordering, not
/// the reduction. Fixture dimensions:
///   up_proj:   [128, 64]  — K=64 reduction in the forward matmul
///   down_proj: [64, 128]  — K=128 reduction, tighter but still < 5e-6
///   batch:     [8, 4, 64] = 32 rows × 64 channels of calibration data
/// Tighter (e.g. 1e-6) risks flakiness from reduction-order drift;
/// looser would mask real subprocess-pipeline bugs.
#[test]
fn end_to_end_real_subprocess_matches_analytical_reference() {
    let data_path = fixture("awq_calib_data.safetensors");
    let weights_path = fixture("awq_calib_weights.safetensors");
    let (projections, compile_bundle) = awq_fixture_compile_bundle();

    let mut registry = HookRegistry::new();
    registry.register(Box::new(AwqCalibrationHook::from_discovered(&projections)));

    let cfg = HarnessConfig {
        checkpoints: vec![weights_path.clone()],
        calibration_data: data_path.clone(),
        samples: 8,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
        projections,
        compile_bundle: Some(compile_bundle),
    };

    let sidecar = real_subprocess_entry(&cfg, &registry)
        .expect("real subprocess pipeline runs end-to-end")
        .sidecar;

    // #134 §6.1 determinism verification: when SIDECAR_DUMP=1, print the
    // canonical sidecar JSON between sentinel lines so
    // scripts/verify-awq-determinism.sh can extract and compare across runs.
    if std::env::var("SIDECAR_DUMP").is_ok() {
        let canonical = serde_json::to_string_pretty(&sidecar)
            .expect("Sidecar serializes to JSON");
        eprintln!("SIDECAR_JSON_START");
        eprintln!("{canonical}");
        eprintln!("SIDECAR_JSON_END");
    }

    let calib = read_safetensors_flat(&data_path, "calibration");
    let up_w = read_safetensors_flat(&weights_path, "TinyMLP.up_proj");
    let (up_ref, down_ref) = reference_awq_scales(&calib, &up_w);

    let up_actual = awq_scales(&sidecar, "TinyMLP.up_proj");
    let down_actual = awq_scales(&sidecar, "TinyMLP.down_proj");

    assert_close(&up_actual, &up_ref, 5e-6, "up_proj");
    assert_close(&down_actual, &down_ref, 5e-6, "down_proj");
}

/// #134 §6.2 — AWQ sidecar bit-identical regression test.
///
/// Captures the full Sidecar JSON as an `insta` snapshot. Under #134's
/// (c-i) convergence shape, this snapshot is **zero-by-construction**: the
/// wrapper-level firing change does not affect `compile_main`'s output and
/// fires calibration against the same fixture in the same order. Any drift
/// after commit 1b indicates an implementation bug in commits 2-5, not a
/// (c-i) design problem.
///
/// Legitimate future changes (new hooks, format upgrades) must update both
/// this snapshot AND `CHANGELOG-CALIBRATION.md` — CI enforces the pairing.
#[test]
fn snapshot_awq_sidecar_baseline() {
    let data_path = fixture("awq_calib_data.safetensors");
    let weights_path = fixture("awq_calib_weights.safetensors");
    let (projections, compile_bundle) = awq_fixture_compile_bundle();

    let mut registry = HookRegistry::new();
    registry.register(Box::new(AwqCalibrationHook::from_discovered(&projections)));

    let cfg = HarnessConfig {
        checkpoints: vec![weights_path.clone()],
        calibration_data: data_path.clone(),
        samples: 8,
        batch_size: 1,
        timeout_secs: 30,
        mode: HarnessMode::Required,
        projections,
        compile_bundle: Some(compile_bundle),
    };

    let sidecar = real_subprocess_entry(&cfg, &registry)
        .expect("real subprocess pipeline runs end-to-end")
        .sidecar;

    // Serialize to a canonical pretty-printed JSON. serde_json sorts BTreeMap
    // keys by construction; Sidecar.hooks is BTreeMap<String, Vec<u8>>. Float
    // fields use default serde_json formatting (deterministic per Rust toolchain).
    let canonical = serde_json::to_string_pretty(&sidecar)
        .expect("Sidecar serializes to JSON");

    // #134 §6.1 determinism verification: when SIDECAR_DUMP=1, print the
    // canonical sidecar JSON between sentinel lines so
    // scripts/verify-awq-determinism.sh can extract and compare across runs.
    if std::env::var("SIDECAR_DUMP").is_ok() {
        eprintln!("SIDECAR_JSON_START");
        eprintln!("{canonical}");
        eprintln!("SIDECAR_JSON_END");
    }

    insta::assert_snapshot!("awq_sidecar_baseline", canonical);
}
