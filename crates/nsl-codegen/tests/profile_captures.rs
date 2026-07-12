//! Dev-tools paper completion: `compile_with_profile_captures` must yield
//! the REAL train-block WengertList for `nsl profile`'s real path — and must
//! yield it EVEN WHEN the minimal single-module compile errors downstream
//! (the documented optimizer-stdlib-symbol failure mode of profile/check
//! grade compiles). The real-liveness timeline must then build from it.

use nsl_codegen::profiling::real_timeline;

const TRAIN_SRC: &str = r#"model Tiny:
    w1: Tensor = randn([32, 16])
    w2: Tensor = randn([16, 4])

    fn forward(self, x: Tensor) -> Tensor:
        let h = x @ self.w1
        let y = h @ self.w2
        return y

let m = Tiny()

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let x = randn([8, 32])
        let out = m.forward(x)
        let loss = sum(out)
"#;

fn parse_and_analyze(
    src: &str,
) -> (nsl_ast::Module, nsl_lexer::Interner, nsl_semantic::AnalysisResult) {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _d) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    (parsed.module, interner, analysis)
}

#[test]
fn captures_survive_compile_even_on_downstream_error() {
    let (module, interner, analysis) = parse_and_analyze(TRAIN_SRC);
    let options = nsl_codegen::CompileOptions {
        source_ad: true,
        target: "h100".to_string(),
        ..Default::default()
    };
    let (captures, result) = nsl_codegen::compile_with_profile_captures(
        &module,
        &interner,
        &analysis.type_map,
        &options,
    );
    // The compile itself may fail (this minimal module has no resolved
    // optimizer stdlib import) — the captures must exist regardless.
    let captures = captures.unwrap_or_else(|| {
        panic!(
            "expected profile captures even when compile result is {:?}",
            result.as_ref().map(|_| "ok").map_err(|e| &e.message)
        )
    });
    let wengert = captures
        .train_wengert
        .as_ref()
        .expect("train block must yield a captured WengertList");
    assert!(
        !wengert.ops.is_empty(),
        "captured WengertList must be non-empty"
    );
    // The real model params must appear (not the synthetic blocks.N names).
    let param_names: Vec<&str> = wengert
        .ops
        .iter()
        .filter_map(|op| match &op.op {
            nsl_codegen::wengert::PrimalOp::Param(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        param_names.iter().any(|n| n.contains("w1")),
        "expected the user's params in the captured list, got {param_names:?}"
    );

    // Size hints: the fixture's shapes are fully concrete, so at least the
    // params should carry byte sizes.
    assert!(
        !captures.var_size_hints.is_empty(),
        "concrete-shaped fixture must produce size hints"
    );

    // The real-liveness timeline must build with forward AND backward ops
    // and mark the phase boundary.
    let rt = real_timeline::build_training_timeline(wengert, &captures.var_size_hints)
        .expect("timeline must build from a non-empty primal list");
    assert!(rt.forward_ops > 0 && rt.backward_ops > 0);
    let rendered = real_timeline::render(&rt, 48);
    assert!(rendered.contains("backward begins"));
    assert!(rendered.contains("Peak:"));
}

#[test]
fn no_train_block_yields_no_captures() {
    let src = "fn main() -> int:\n    return 0\n";
    let (module, interner, analysis) = parse_and_analyze(src);
    let options = nsl_codegen::CompileOptions {
        source_ad: true,
        target: "h100".to_string(),
        ..Default::default()
    };
    let (captures, result) = nsl_codegen::compile_with_profile_captures(
        &module,
        &interner,
        &analysis.type_map,
        &options,
    );
    assert!(
        captures.is_none(),
        "no train block, no captures (got {captures:?})"
    );
    assert!(result.is_ok(), "trivial module must compile");
}
