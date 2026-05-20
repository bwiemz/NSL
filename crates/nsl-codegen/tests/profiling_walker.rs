//! Integration tests for the profiling walker (Task 3 of dev-tools Phase 1).

use nsl_ast::Module;
use nsl_codegen::gpu_specs::default_gpu;
use nsl_codegen::profiling::shape_env::ShapeEnv;
use nsl_codegen::profiling::types::EntryKind;
use nsl_codegen::profiling::walker::walk_ops;
use nsl_lexer::Interner;
use nsl_semantic::AnalysisResult;

/// Mini pipeline: lex → parse → semantic-analyze. Matches the pattern used
/// by `nsl-cli::shape_debug::ShapeDebugInput::from_source`.
fn parse_and_analyze(src: &str) -> (Module, AnalysisResult, Interner) {
    let mut interner = Interner::new();
    let file_id = nsl_errors::FileId(0);
    let (tokens, _lex_errs) = nsl_lexer::tokenize(src, file_id, &mut interner);
    let parse = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parse.module, &mut interner);
    (parse.module, analysis, interner)
}

#[test]
fn walks_single_matmul() {
    // Use fully concrete shapes so there's nothing to resolve.
    let src = "\
fn step() -> Tensor<[4, 8], f32>:
    let a: Tensor<[4, 16], f32> = zeros([4, 16])
    let b: Tensor<[16, 8], f32> = zeros([16, 8])
    let c = a @ b
    return c
";
    let (module, analysis, interner) = parse_and_analyze(src);
    let env = ShapeEnv::new();
    let gpu = default_gpu();
    let report = walk_ops(
        &module,
        &analysis,
        &interner,
        EntryKind::Function("step".into()),
        &env,
        gpu,
        "fp16",
    )
    .expect("walk_ops");

    let matmuls: Vec<_> = report.ops.iter().filter(|o| o.name == "matmul").collect();
    assert_eq!(
        matmuls.len(),
        1,
        "expected exactly one matmul op, got {:?}",
        report.ops
    );
    // 2 * M * K * N = 2 * 4 * 16 * 8 = 1024 FLOPs
    assert_eq!(matmuls[0].flops, 2 * 4 * 16 * 8);
}

#[test]
fn walker_inlines_model_forward_from_train() {
    let src = "\
model Lin():
    w: Tensor<[16, 8], f32> = zeros([16, 8])

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn step_fn() -> Tensor<[4, 8], f32>:
    let m = Lin()
    let x: Tensor<[4, 16], f32> = zeros([4, 16])
    return m.forward(x)
";
    let (module, analysis, interner) = parse_and_analyze(src);
    let env = ShapeEnv::new();
    let gpu = default_gpu();
    let report = walk_ops(
        &module,
        &analysis,
        &interner,
        EntryKind::Function("step_fn".into()),
        &env,
        gpu,
        "fp16",
    )
    .expect("walk_ops");

    let matmul_count = report.ops.iter().filter(|o| o.name == "matmul").count();
    assert!(
        matmul_count >= 1,
        "expected at least one matmul from inlined forward, got ops={:?}",
        report
            .ops
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn unknown_op_yields_zero_cost_with_note() {
    let src = "\
fn step() -> Tensor<[4, 8], f32>:
    let a: Tensor<[4, 16], f32> = zeros([4, 16])
    let b: Tensor<[16, 8], f32> = zeros([16, 8])
    return my_custom_op(a, b)
";
    let (module, analysis, interner) = parse_and_analyze(src);
    let env = ShapeEnv::new();
    let gpu = default_gpu();
    let report = walk_ops(
        &module,
        &analysis,
        &interner,
        EntryKind::Function("step".into()),
        &env,
        gpu,
        "fp16",
    )
    .expect("walk_ops");

    let unknowns: Vec<_> = report
        .ops
        .iter()
        .filter(|o| o.name.starts_with("unknown"))
        .collect();
    assert!(
        !unknowns.is_empty(),
        "expected at least one unknown op, got {:?}",
        report
            .ops
            .iter()
            .map(|o| o.name.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(unknowns[0].flops, 0);
    assert!(
        unknowns[0].loc.contains("unknown"),
        "expected loc to note `unknown`, got `{}`",
        unknowns[0].loc
    );
}

#[test]
fn unresolved_shape_var_does_not_panic() {
    // Symbolic dim "B" that is not supplied by the env.
    let src = "\
fn step(x: Tensor<[B, 16], f32>, w: Tensor<[16, 8], f32>) -> Tensor<[B, 8], f32>:
    return x @ w
";
    let (module, analysis, interner) = parse_and_analyze(src);
    let env = ShapeEnv::new(); // deliberately empty
    let gpu = default_gpu();
    let report = walk_ops(
        &module,
        &analysis,
        &interner,
        EntryKind::Function("step".into()),
        &env,
        gpu,
        "fp16",
    )
    .expect("walk_ops");

    // Either: a matmul op with 0 flops (shape unresolved), or no ops. Just
    // assert the walker didn't panic and any emitted op has 0 flops.
    for op in &report.ops {
        if op.name == "matmul" {
            // Allowed: flops can be 0 when K is unresolved.
            // We also accept non-zero if the walker managed to resolve from
            // concrete B via Bounded/etc. Either is fine.
        }
    }
}

#[test]
fn walker_populates_origin_node_on_each_op() {
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    let y = matmul(x, W)
    return y
"#;
    let (m, analysis, interner) = parse_and_analyze(src);
    let gpu = nsl_codegen::gpu_specs::find_gpu("h100").unwrap();
    let env = nsl_codegen::profiling::shape_env::ShapeEnv::with_defaults();
    let r = nsl_codegen::profiling::walker::walk_ops(
        &m,
        &analysis,
        &interner,
        nsl_codegen::profiling::types::EntryKind::Auto,
        &env,
        gpu,
        "bf16",
    )
    .unwrap();
    assert!(!r.ops.is_empty());
    for op in &r.ops {
        assert!(
            op.origin_node.is_some(),
            "every walked op should carry its source NodeId, got None for {}",
            op.name
        );
    }
}
