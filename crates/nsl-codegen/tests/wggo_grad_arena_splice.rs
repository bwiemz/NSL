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
    parse_src(&std::fs::read_to_string(fixture_path()).expect("fixture readable"))
}

fn parse_src(src: &str) -> (nsl_ast::Module, Interner) {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = tokenize(src, FileId(0), &mut interner);
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
    opts.calibration.batch_seq = Some((4, 4));
    opts.calibration.compile_bundle = Some(std::sync::Arc::new(
        nsl_codegen::calibration::CalibrationCompileBundle {
            ast: ast.clone(),
            interner: analysis_interner,
            type_map: analysis.type_map.clone(),
        },
    ));
    opts.weights.index_map = analysis.weight_index_map.clone();

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
    opts.calibration.grad_retention = Some(targets);
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
            if let object::RelocationTarget::Symbol(sym_idx) = reloc.target()
                && let Ok(sym) = obj.symbol_by_index(sym_idx)
                && sym.name().map(nsl_codegen::linker::strip_host_symbol_prefix)
                    == Ok("__nsl_calib_grad_arena")
            {
                count += 1;
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

/// The fixture with an RMSNorm after the projections, its eps a model field
/// (`eps_decl`). Source AD reads a norm's eps field at run time
/// (`NormEps::Var`); a calibration binary binds no model struct, so it binds
/// the field's declared literal -- and refuses a field it cannot evaluate
/// rather than training against some other eps.
fn norm_eps_fixture(eps_decl: &str) -> String {
    format!(
        r#"@quantize(dtype="awq4")
model TinyAttn4:
    q_proj: Tensor = zeros([16, 16])
    k_proj: Tensor = zeros([16, 16])
    v_proj: Tensor = zeros([16, 16])
    o_proj: Tensor = zeros([16, 16])
    g: Tensor = ones([16])
    {eps_decl}

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=4)
    fn forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x |> q_proj |> k_proj |> v_proj |> o_proj, self.g, self.eps)

fn main():
    let m = TinyAttn4()
"#
    )
}

fn emit(src: &str) -> Result<(), String> {
    let (ast, interner) = parse_src(src);
    let projections = nsl_codegen::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
    let arena_layout = build_arena_layout(&projections, 4, 4);
    let tmp = tempfile::tempdir().expect("tempdir");
    let opts = opts_with_backward(&ast, &interner);
    emit_calibration_model_object(&ast, &opts, &arena_layout, &tmp.path().join("calib.o"))
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn model_backward_binds_a_norm_eps_field_to_its_declared_value() {
    emit(&norm_eps_fixture("eps: float = 0.5"))
        .expect("a literal eps field is bound to its declared value");
    let err = emit(&norm_eps_fixture("eps: float = 0.25 * 2.0"))
        .expect_err("an eps field calibration cannot evaluate must be refused, not defaulted");
    assert!(err.contains("eps"), "the refusal must name the field: {err}");
}
