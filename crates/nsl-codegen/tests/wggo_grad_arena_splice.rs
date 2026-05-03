//! Spec §4.2 — source-AD splice into __nsl_calib_grad_arena.
//!
//! For a fixture with one Attention block (Q, K, V, O projections), the
//! calibration object's `model_backward` body must reference
//! `__nsl_calib_grad_arena` ≥4 times (one relocation per distinct weight
//! that the on_param_grad callback writes a grad slice to).

use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner};
use nsl_codegen::calibration::{
    observation::ProjectionRef,
    retention_pass::build_arena_layout,
};
use nsl_codegen::calibration::binary_codegen::emit_calibration_model_object;

// ── Helper: parse the four-projection fixture ─────────────────────────────────

fn fixture_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("wggo_attn_4proj.nsl")
}

fn parse_attn4_fixture() -> (nsl_ast::Module, Interner) {
    let src = std::fs::read_to_string(fixture_path()).expect("fixture readable");
    let mut interner = Interner::new();
    let (tokens, lex_diags) = tokenize(&src, FileId(0), &mut interner);
    assert!(
        lex_diags.iter().all(|d| !matches!(d.level, Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed.diagnostics.iter().all(|d| !matches!(d.level, Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );
    (parsed.module, interner)
}

// ── Helper: build CompileOptions with backward enabled ───────────────────────

fn opts_with_backward(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> nsl_codegen::CompileOptions {
    use nsl_codegen::calibration::discovery::WggoGradTarget;

    let mut analysis_interner = interner.clone();
    let analysis = nsl_semantic::analyze(ast, &mut analysis_interner);

    let mut opts = nsl_codegen::CompileOptions::default();
    opts.calibration_batch_seq = Some((4, 4));
    opts.calibration_compile_bundle = Some(std::sync::Arc::new(
        nsl_codegen::calibration::CalibrationCompileBundle {
            ast: ast.clone(),
            interner: analysis_interner,
            type_map: analysis.type_map.clone(),
        },
    ));
    opts.weight_index_map = analysis.weight_index_map.clone();

    // Four distinct weight projections with compatible square shapes (16x16).
    let targets = vec![WggoGradTarget {
        layer_key: "TinyAttn4".into(),
        class_name: "TinyAttn4".into(),
        head_dim: 4,
        w_q: ProjectionRef("TinyAttn4.q_proj".into()),
        w_k: ProjectionRef("TinyAttn4.k_proj".into()),
        w_v: ProjectionRef("TinyAttn4.v_proj".into()),
        w_o: ProjectionRef("TinyAttn4.o_proj".into()),
        w_q_shape: [16, 16],
        w_k_shape: [16, 16],
        w_v_shape: [16, 16],
        w_o_shape: [16, 16],
        w_q_index: 0,
        w_k_index: 1,
        w_v_index: 2,
        w_o_index: 3,
    }];
    opts.calibration_grad_retention = Some(targets);
    opts
}

// ── Helper: count relocations targeting __nsl_calib_grad_arena ───────────────

fn count_grad_arena_refs(obj_bytes: &[u8]) -> usize {
    use object::{Object, ObjectSection, ObjectSymbol, SectionKind};
    let obj = object::File::parse(obj_bytes).expect("object::File::parse");
    let mut count = 0;
    for sec in obj.sections() {
        if sec.kind() != SectionKind::Text {
            continue;
        }
        for (_offset, reloc) in sec.relocations() {
            if let object::RelocationTarget::Symbol(sym_idx) = reloc.target() {
                if let Ok(sym) = obj.symbol_by_index(sym_idx) {
                    if sym.name() == Ok("__nsl_calib_grad_arena") {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

// ── Test ──────────────────────────────────────────────────────────────────────

/// Spec §4.2: the compiled `model_backward` body must reference
/// `__nsl_calib_grad_arena` at least once per distinct W_* projection.
///
/// With the `wggo_attn_4proj.nsl` fixture (q_proj, k_proj, v_proj, o_proj),
/// the on_param_grad callback fires 4 times and each call emits a
/// `emit_splice_memcpy` that references the arena global.  We count
/// relocations in the text section targeting the symbol.
#[test]
fn model_backward_emits_grad_arena_memcpy_for_each_w_star() {
    let (ast, interner) = parse_attn4_fixture();
    let projections = nsl_codegen::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
    let arena_layout = build_arena_layout(&projections, 4, 4);
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("calib_model_grad.o");
    let opts = opts_with_backward(&ast, &interner);

    emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path)
        .expect("emit_calibration_model_object succeeds with 4-proj backward");

    let obj_bytes = std::fs::read(&out_path).expect("object readable");
    let count = count_grad_arena_refs(&obj_bytes);
    assert!(
        count >= 4,
        "expected ≥4 relocations targeting __nsl_calib_grad_arena (one per W_*); got {count}"
    );
}
