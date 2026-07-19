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
