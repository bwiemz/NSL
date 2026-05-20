//! Phase 2.5 Task 3 regression: epilogue fusion against the Compiler-owned
//! plan must be visible through `Compiler::fusion_constituents`.
//!
//! Guards against the throwaway-plan pattern where fusion populates a
//! `FusionPlan` that never flows back to the Compiler. Also asserts that
//! the pre-pass seeds `fusion_plan_for_profile = Some(...)` (Part B of the
//! task).
#![cfg(feature = "test-helpers")]

use nsl_ast::NodeId as AstNodeId;
use nsl_codegen::epilogue_fusion::{apply_epilogue_fusion, EpilogueChain, EpilogueOp, MatmulKind};
use nsl_codegen::fusion_graph::{FusionGraph, FusionOp};
use nsl_codegen::wrga_fusion::FusionPlan;
use nsl_codegen::CompileOptions;
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

/// Part B coverage: the pre-pass must seed `fusion_plan_for_profile = Some(...)`
/// so that later fusion passes have something to populate in place.
#[test]
fn pre_pass_seeds_fusion_plan_as_some() {
    let src = "fn forward(x: int) -> int:\n    return x\n";
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    let result =
        nsl_codegen::test_helpers::run_pre_pass_only(src, &opts).expect("pre-pass should succeed");
    assert!(
        result.manifest_builder_set,
        "manifest_builder should be populated on the profile path"
    );
    assert!(
        result.fusion_plan_set,
        "pre-pass must seed fusion_plan_for_profile = Some(...)"
    );
}

/// Part A+C coverage: a plan obtained via `profile_fusion_plan_mut` and
/// mutated through `apply_epilogue_fusion` must be visible through
/// `fusion_constituents`. This is the critical round-trip: if fusion writes
/// into a scratch plan that never flows back, the assertion here fails.
#[test]
fn seeded_fusion_plan_round_trips_through_fusion_constituents() {
    let interner = Interner::new();
    let type_map = TypeMap::new();
    let opts = CompileOptions::default();
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed");

    // Simulate the pre-pass seeding step (Part B) by installing an empty plan.
    compiler.fusion_plan_for_profile = Some(FusionPlan::default());

    // Build the same minimal graph Task 1's test used: matmul + bias + relu.
    let mut graph = FusionGraph::new();
    let a = graph.add_node(FusionOp::Input, vec![]);
    let b = graph.add_node(FusionOp::Input, vec![]);
    let bias_in = graph.add_node(FusionOp::Input, vec![]);
    let matmul = graph.add_node(FusionOp::Matmul, vec![a, b]);
    let bias = graph.add_node(FusionOp::Elementwise("add".into()), vec![matmul, bias_in]);
    let relu = graph.add_node(FusionOp::Elementwise("relu".into()), vec![bias]);

    let chain = EpilogueChain {
        matmul_node: matmul,
        matmul_kind: MatmulKind::Standard,
        epilogue_ops: vec![
            EpilogueOp::BiasAdd {
                bias_node: bias,
                broadcast_dim: 0,
            },
            EpilogueOp::Activation("relu".into()),
        ],
        output_node: relu,
        eliminated_nodes: vec![bias, relu],
    };

    // THE critical step: thread the Compiler-owned plan (not a throwaway)
    // into apply_epilogue_fusion.
    let plan = compiler.profile_fusion_plan_mut().expect("seeded above");
    apply_epilogue_fusion(&mut graph, &[chain], 0, plan);

    // Read back through the Compiler API — this asserts the mutations landed
    // in the same FusionPlan instance that fusion_constituents reads.
    let constituents = compiler.fusion_constituents(AstNodeId(matmul));
    assert_eq!(
        constituents,
        vec![AstNodeId(matmul), AstNodeId(bias), AstNodeId(relu)],
        "fusion_constituents must read the Compiler-owned plan populated in place"
    );
}
