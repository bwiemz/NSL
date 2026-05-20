use std::path::PathBuf;

use nsl_codegen::calibration::{
    build_arena_layout, emit_calibration_model_object, emit_calibration_scaffolding_object,
    link_calibration_binary, pre_scan_awq_projections_from_ast, FinalizePlanEntry,
    ObservePlanEntry, ProjectionRef,
};
use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner};

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
        lex_diags
            .iter()
            .all(|diag| !matches!(diag.level, Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.level, Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    (parsed.module, interner)
}

fn awq_fixture_compile_options(
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
    opts
}

fn awq_plans() -> (Vec<ObservePlanEntry>, Vec<FinalizePlanEntry>) {
    let projection = ProjectionRef("TinyMLP.up_proj".into());
    (
        vec![ObservePlanEntry {
            projection: projection.clone(),
            src_offset: 0,
            rows: 32,
            channels: 64,
            running_symbol: "__nsl_awq_running_up_proj".into(),
        }],
        vec![FinalizePlanEntry {
            projection,
            running_symbol: "__nsl_awq_running_up_proj".into(),
            channels: 64,
            bytes_per_element: 4, // AWQ f32 max-abs running buffer
        }],
    )
}

#[test]
fn two_object_link_produces_binary_with_resolved_wrapper_symbol() {
    let (ast, interner) = parse_awq_fixture();
    let projections = pre_scan_awq_projections_from_ast(&ast, &interner);
    let arena_layout = build_arena_layout(&projections, 8, 4);
    let (observe_plan, finalize_plan) = awq_plans();
    let tmp = tempfile::tempdir().expect("tempdir");
    let opts = awq_fixture_compile_options(&ast, &interner);

    let model_obj = tmp.path().join("calib_model.o");
    emit_calibration_model_object(&ast, &opts, &arena_layout, &model_obj)
        .expect("emit calib_model.o");

    let scaffolding_obj = tmp.path().join("scaffolding.o");
    emit_calibration_scaffolding_object(
        &observe_plan,
        &finalize_plan,
        &arena_layout,
        b"{}",
        true,
        false, // needs_backward = false for AWQ-only test
        &[],   // no per-step backward symbols for AWQ-only
        &[],   // no wggo targets for AWQ-only test
        None,  // no grad_arena_layout
        &scaffolding_obj,
    )
    .expect("emit scaffolding.o");

    let binary = tmp.path().join(if cfg!(windows) {
        "calibration.exe"
    } else {
        "calibration"
    });
    link_calibration_binary(&scaffolding_obj, &model_obj, &binary, false)
        .expect("link two-object calibration binary");

    assert!(binary.exists(), "linked calibration binary must exist");

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let nm_output = std::process::Command::new("nm")
            .arg(&binary)
            .output()
            .expect("nm runs");
        let stdout = String::from_utf8_lossy(&nm_output.stdout);
        assert!(
            !stdout
                .lines()
                .any(|line| line.contains("U nsl_calib_model_forward")),
            "nsl_calib_model_forward must be resolved, not undefined: {stdout}"
        );
    }
}

#[test]
fn missing_wrapper_symbol_at_link_emits_three_part_error() {
    let (observe_plan, finalize_plan) = awq_plans();
    let arena_layout = build_arena_layout(
        &[nsl_codegen::calibration::DiscoveredProjection {
            projection: ProjectionRef("TinyMLP.up_proj".into()),
            weight_shape: [128, 64],
        }],
        8,
        4,
    );
    let tmp = tempfile::tempdir().expect("tempdir");

    let scaffolding_obj = tmp.path().join("scaffolding.o");
    emit_calibration_scaffolding_object(
        &observe_plan,
        &finalize_plan,
        &arena_layout,
        b"{}",
        true,
        false, // needs_backward = false; tests forward-path link failure
        &[],   // no per-step backward symbols
        &[],   // no wggo targets for this link-test
        None,  // no grad_arena_layout
        &scaffolding_obj,
    )
    .expect("emit scaffolding.o");

    let binary = tmp.path().join(if cfg!(windows) {
        "calibration.exe"
    } else {
        "calibration"
    });
    let err = link_calibration_binary(&scaffolding_obj, &scaffolding_obj, &binary, false)
        .expect_err("link should fail without calib_model.o wrapper export");

    let err_str = format!("{err}");
    assert!(err_str.contains("calibration: model-forward wrapper missing from calib_model.o"));
    assert!(err_str.contains("requested:"));
    assert!(err_str.contains("expected:"));
    assert!(err_str.contains("found:"));
}
