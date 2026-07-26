//! P0.2 gradient-integrity guard — positive trigger test.
//!
//! PR #396 showed the Wengert lowerer could meet an adjoint op whose input
//! VarId was never produced (a ghost merged into a live accumulation `Add`)
//! and SILENTLY skip it, dropping a real parameter gradient with no crash and
//! no diagnostic. The guard now converts that into a hard `CodegenError` when
//! the op is LIVE (its result is in `compiler.grad_live_results`, i.e.
//! structurally reachable from a needed parameter gradient), while still
//! allowing dead/ghost ops to be skipped.
//!
//! This drives the REAL lowerer (`compile_wengert_ops`) with a hand-built
//! adjoint containing a live op that reads an unresolved input, and asserts
//! the loud error — the exact behavior that would have caught #396 at compile
//! time. It complements the pure-function unit tests
//! (`source_ad::reachable_result_vars`, `wengert_lower::describe_producer_chain`).
#![cfg(feature = "test-helpers")]

use cranelift_codegen::ir::{Function, Signature, UserFuncName};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use nsl_codegen::context::FuncState;
use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
use nsl_codegen::wengert_lower::compile_wengert_ops;
use nsl_codegen::CompileOptions;
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;
use std::collections::{HashMap, HashSet};

fn op(id: u32, result: u32, kind: PrimalOp, inputs: Vec<u32>) -> WengertOp {
    WengertOp {
        id,
        result,
        op: kind,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    }
}

#[test]
fn live_op_with_unresolved_input_is_a_compile_error() {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions::default();
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed with default options");

    // A minimal, valid Cranelift function context. The lowerer errors on the
    // first (unresolved, live) op before it emits any IR, so an empty entry
    // block is all we need.
    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(CallConv::Fast),
    ));
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let mut state = FuncState::new();
    let entry = builder.create_block();
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    state.current_block = Some(entry);

    // Adjoint: the needed parameter gradient VarId 100 = Add(50, 6). Neither
    // input resolves: VarId 50 has a producer op LATER in the list (a Neg of
    // unresolved VarId 5 — a cascade), and VarId 6 has NO producer at all (a
    // leaf ghost adjoint never populated by accumulate_adjoint). The Add is op
    // 0 so the guard fires before the lowerer emits any IR / touches an FFI
    // (the Neg op 1 is never lowered). `primal_vars` is empty.
    let wengert = WengertList {
        ops: vec![
            op(0, 100, PrimalOp::Add, vec![50, 6]),
            op(1, 50, PrimalOp::Neg, vec![5]),
        ],
        output: 100,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    let primal_vars: HashMap<u32, cranelift_codegen::ir::Value> = HashMap::new();

    // Arm the guard: 100 and 50 are live (reachable from the needed grad 100).
    compiler.grad_live_results = Some(HashSet::from([100u32, 50u32]));

    let result =
        compile_wengert_ops(&mut compiler, &mut builder, &mut state, &wengert, &primal_vars, None);

    let err = match result {
        Ok(_) => panic!("live op with an unresolved input must be a hard error, got Ok"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("live gradient op has an unresolved input"),
        "diagnostic must name the failure mode:\n{msg}"
    );
    assert!(msg.contains("result VarId: 100"), "must name the result VarId:\n{msg}");
    assert!(
        msg.contains("unresolved input(s): [50, 6]"),
        "must name the unresolved inputs:\n{msg}"
    );
    assert!(msg.contains("producer chain"), "must include a producer chain:\n{msg}");
    assert!(msg.contains("VarId 50"), "chain must trace the cascade via VarId 50:\n{msg}");
    assert!(
        msg.contains("leaf ghost"),
        "VarId 6 has no producer → chain must flag the leaf ghost:\n{msg}"
    );
}

#[test]
fn dead_ghost_op_with_unresolved_input_is_still_skipped() {
    // Same shape, but the op is NOT in `grad_live_results` (dead/ghost). The
    // lowerer must skip it silently (no error) — the guard only fires on LIVE
    // ops. With the guard disarmed entirely (`None`), the pre-#396 skip
    // behavior is preserved for every non-training lowering.
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions::default();
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed");

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(CallConv::Fast),
    ));
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let mut state = FuncState::new();
    let entry = builder.create_block();
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    state.current_block = Some(entry);

    // A single ghost op (result 200) reading unresolved input 5. Nothing maps
    // 5; 200 is NOT live.
    let wengert = WengertList {
        ops: vec![op(0, 200, PrimalOp::Add, vec![5, 6])],
        output: 200,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    let primal_vars: HashMap<u32, cranelift_codegen::ir::Value> = HashMap::new();

    // Disarmed (None) — the default for forward / free-list / calibration.
    compiler.grad_live_results = None;
    let result =
        compile_wengert_ops(&mut compiler, &mut builder, &mut state, &wengert, &primal_vars, None);
    assert!(
        result.is_ok(),
        "a dead/ghost op with an unresolved input must be skipped, not errored"
    );
}

/// Item 7 regression: `--fuse-wgrad-accum` must NOT be able to turn the guard
/// above into a silent gradient drop.
///
/// The fusion elides the chain's `Transpose` and `Matmul` and consults their
/// operands at the `reduce_to_shape` instead. Both elided ops are in
/// `grad_live_results` (`reachable_result_vars` walks back from the needed
/// param adjoints through `op.inputs`), so unfused, an unmapped `x` fails the
/// compile on the transpose. The first version of the fused path returned
/// `continue` there — which would compile clean and train that weight at
/// exactly zero gradient, the precise #396 regression this file exists to
/// prevent. It must be an error on BOTH paths.
#[test]
fn fused_wgrad_chain_with_an_unresolved_operand_is_still_a_compile_error() {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let mut opts = CompileOptions::default();
    opts.fuse_wgrad_accum = true;
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed");

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(CallConv::Fast),
    ));
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let mut state = FuncState::new();
    let entry = builder.create_block();
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    state.current_block = Some(entry);

    // The canonical weight-gradient chain, with the upstream gradient (VarId 1)
    // NEVER produced — the ghost-adjoint case. x=0, g=1, w=2.
    let wengert = WengertList {
        ops: vec![
            op(
                0,
                10,
                PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX },
                vec![0],
            ),
            op(1, 11, PrimalOp::Matmul, vec![10, 1]),
            op(
                2,
                12,
                PrimalOp::Passthrough("reduce_to_shape".into()),
                vec![11, 2],
            ),
        ],
        output: 12,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    // `x` (0) and `w` (2) resolve; `g` (1) does not.
    let mut primal_vars: HashMap<u32, cranelift_codegen::ir::Value> = HashMap::new();
    let zero = {
        use cranelift_codegen::ir::InstBuilder;
        builder.ins().iconst(cranelift_codegen::ir::types::I64, 0)
    };
    primal_vars.insert(0, zero);
    primal_vars.insert(2, zero);

    compiler.grad_live_results = Some(HashSet::from([10u32, 11u32, 12u32]));
    let param_adj: HashSet<u32> = HashSet::from([12u32]);
    let mut cb = |_: &mut nsl_codegen::compiler::Compiler,
                  _: u32,
                  _: nsl_codegen::wengert_lower::ParamGradSource,
                  _: bool,
                  _: &mut FunctionBuilder|
     -> Result<(), nsl_codegen::error::CodegenError> {
        panic!("the hook must not fire: the chain's operand never resolved");
    };

    let result = compile_wengert_ops(
        &mut compiler,
        &mut builder,
        &mut state,
        &wengert,
        &primal_vars,
        Some((&param_adj, &mut cb)),
    );
    let err = match result {
        Ok(_) => panic!(
            "a fused wgrad chain with an unresolved operand must be a hard \
             error — returning Ok would silently zero this parameter's gradient"
        ),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("fused weight-gradient chain has an unresolved operand"),
        "diagnostic must name the fused failure mode:\n{msg}"
    );
    assert!(
        msg.contains("unresolved input(s): [1]"),
        "must name the unresolved operand (g = VarId 1):\n{msg}"
    );
    assert!(
        msg.contains("--fuse-wgrad-accum"),
        "must tell the user which flag to drop for the precise diagnostic:\n{msg}"
    );
}

/// ANTI-VACUITY for the test above: the SAME tape with every operand resolved
/// must lower cleanly and fire the hook once. Without this, the assertion
/// above would still pass if the fusion had simply stopped matching the
/// pattern (or if `plan` rejected the chain for an unrelated reason).
#[test]
fn fused_wgrad_chain_with_resolved_operands_fires_the_hook() {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let mut opts = CompileOptions::default();
    opts.fuse_wgrad_accum = true;
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed");

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, 0),
        Signature::new(CallConv::Fast),
    ));
    let mut fb_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
    let mut state = FuncState::new();
    let entry = builder.create_block();
    builder.switch_to_block(entry);
    builder.seal_block(entry);
    state.current_block = Some(entry);

    let wengert = WengertList {
        ops: vec![
            op(
                0,
                10,
                PrimalOp::Transpose { dim0: usize::MAX - 1, dim1: usize::MAX },
                vec![0],
            ),
            op(1, 11, PrimalOp::Matmul, vec![10, 1]),
            op(
                2,
                12,
                PrimalOp::Passthrough("reduce_to_shape".into()),
                vec![11, 2],
            ),
        ],
        output: 12,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    let zero = {
        use cranelift_codegen::ir::InstBuilder;
        builder.ins().iconst(cranelift_codegen::ir::types::I64, 0)
    };
    let primal_vars: HashMap<u32, cranelift_codegen::ir::Value> =
        HashMap::from([(0, zero), (1, zero), (2, zero)]);

    compiler.grad_live_results = Some(HashSet::from([10u32, 11u32, 12u32]));
    let param_adj: HashSet<u32> = HashSet::from([12u32]);
    let mut fired = 0usize;
    let mut cb = |_: &mut nsl_codegen::compiler::Compiler,
                  vid: u32,
                  src: nsl_codegen::wengert_lower::ParamGradSource,
                  still_needed: bool,
                  _: &mut FunctionBuilder|
     -> Result<(), nsl_codegen::error::CodegenError> {
        assert_eq!(vid, 12, "hook must fire on the param adjoint");
        assert!(!still_needed, "a fused chain has no later reader by construction");
        assert!(
            matches!(src, nsl_codegen::wengert_lower::ParamGradSource::FusedWgrad { .. }),
            "hook must receive the chain's OPERANDS, not a materialized gradient"
        );
        fired += 1;
        Ok(())
    };

    compile_wengert_ops(
        &mut compiler,
        &mut builder,
        &mut state,
        &wengert,
        &primal_vars,
        Some((&param_adj, &mut cb)),
    )
    .expect("a fully-resolved fused chain must lower cleanly");
    assert_eq!(fired, 1, "the hook must fire exactly once for one chain");
}
