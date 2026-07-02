//! CFTP v6 — `wengert_lower` precision-cast insertion IR pin.
//!
//! Sprint v6 closes the Sprint v5 Finding 7 buffer-conformance gap by
//! inserting an explicit `nsl_tensor_to_{bf16,fp16}` cast call before the
//! `nsl_fused_linear_ce_{forward,forward_large,backward}` FFI when the
//! active `@fused_lm_ce(dtype = "...")` decorator selects a non-f32
//! emitter dtype. This test pins the IR shape of that insertion.
//!
//! Contract:
//!   * F32 hint (or absent) -> ZERO calls to either cast wrapper.  This is
//!     the byte-identity guard for the v1 F32 path.
//!   * FP16 hint -> exactly THREE calls to `nsl_tensor_to_fp16` (x / W / bias),
//!     ZERO calls to `nsl_tensor_to_bf16`, and each cast call's result must
//!     flow into a subsequent `nsl_tensor_data_ptr` call before the main
//!     forward / backward FFI.
//!   * BF16 hint -> exactly THREE calls to `nsl_tensor_to_bf16` (x / W / bias),
//!     ZERO calls to `nsl_tensor_to_fp16`, same data_ptr flow requirement.
//!   * Large-vocab forward path -> same three-cast insertion (the cast site
//!     is BEFORE the small/large branch in `lower_fused_linear_ce_forward`).
//!   * Backward extract path -> same three-cast insertion (mirror of forward).
//!
//! The IR walk searches by FFI symbol NAME (resolved via the Compiler's
//! `runtime_fns` registry) so future Cranelift IR layout changes don't break
//! the contract silently.

#![cfg(feature = "test-helpers")]

use cranelift_codegen::ir::{
    instructions::InstructionData, AbiParam, ExternalName, Function, InstBuilder,
    UserFuncName,
};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};

use nsl_codegen::context::FuncState;
use nsl_codegen::wengert::{PrimalOp, VarId, WengertList, WengertOp, WengertType};
use nsl_codegen::wengert_lower::compile_wengert_ops;
use nsl_codegen::{
    CompileOptions, FusedCeDecoratorConfig, FusedCeDtypeHint,
};
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

use std::collections::HashMap;

const SMALL_VOCAB: u32 = 4096;
const LARGE_VOCAB: u32 = 16384;
const HIDDEN: u32 = 128;
const BATCH: u32 = 2;
const SEQ: u32 = 32;
const TILE: u32 = 1024;

fn cfg_for_dtype_hint(
    hint: Option<FusedCeDtypeHint>,
    vocab_size: u32,
) -> FusedCeDecoratorConfig {
    FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(TILE),
        vocab_size: Some(vocab_size),
        hidden_size: Some(HIDDEN),
        batch_size: Some(BATCH),
        seq_len: Some(SEQ),
        dtype: hint,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    }
}

fn build_forward_wengert(vocab_size: u32, is_large: bool) -> WengertList {
    let mut var_types: HashMap<VarId, WengertType> = HashMap::new();
    for v in 1..=5u32 {
        var_types.insert(v, WengertType::Tensor);
    }
    WengertList {
        ops: vec![
            WengertOp {
                id: 0,
                result: 1,
                op: PrimalOp::Input("x".to_string()),
                inputs: vec![],
                saved_for_backward: false,
                checkpointed: false,
            },
            WengertOp {
                id: 1,
                result: 2,
                op: PrimalOp::Input("W".to_string()),
                inputs: vec![],
                saved_for_backward: false,
                checkpointed: false,
            },
            WengertOp {
                id: 2,
                result: 3,
                op: PrimalOp::Input("bias".to_string()),
                inputs: vec![],
                saved_for_backward: false,
                checkpointed: false,
            },
            WengertOp {
                id: 3,
                result: 4,
                op: PrimalOp::Input("targets".to_string()),
                inputs: vec![],
                saved_for_backward: false,
                checkpointed: false,
            },
            WengertOp {
                id: 4,
                result: 5,
                op: PrimalOp::FusedLinearCe {
                    vocab_size,
                    hidden_size: HIDDEN,
                    batch_size: BATCH,
                    seq_len: SEQ,
                    vocab_tile: TILE,
                    ignore_index: -100,
                    is_large,
                },
                inputs: vec![1, 2, 3, 4],
                saved_for_backward: false,
                checkpointed: false,
            },
        ],
        output: 5,
        var_names: HashMap::new(),
        var_types,
    }
}

/// Build a forward WengertList AND chain a `FusedLinearCeBackwardExtract`
/// node off of it (component=0) so the backward lowering executes. The
/// backward path is the second site we need to pin.
fn build_forward_plus_backward_wengert(vocab_size: u32) -> WengertList {
    let mut var_types: HashMap<VarId, WengertType> = HashMap::new();
    for v in 1..=7u32 {
        var_types.insert(v, WengertType::Tensor);
    }
    WengertList {
        ops: vec![
            WengertOp { id: 0, result: 1, op: PrimalOp::Input("x".into()), inputs: vec![], saved_for_backward: false, checkpointed: false },
            WengertOp { id: 1, result: 2, op: PrimalOp::Input("W".into()), inputs: vec![], saved_for_backward: false, checkpointed: false },
            WengertOp { id: 2, result: 3, op: PrimalOp::Input("bias".into()), inputs: vec![], saved_for_backward: false, checkpointed: false },
            WengertOp { id: 3, result: 4, op: PrimalOp::Input("targets".into()), inputs: vec![], saved_for_backward: false, checkpointed: false },
            WengertOp { id: 4, result: 5, op: PrimalOp::Input("grad_seed".into()), inputs: vec![], saved_for_backward: false, checkpointed: false },
            WengertOp {
                id: 5,
                result: 6,
                op: PrimalOp::FusedLinearCe {
                    vocab_size,
                    hidden_size: HIDDEN,
                    batch_size: BATCH,
                    seq_len: SEQ,
                    vocab_tile: TILE,
                    ignore_index: -100,
                    is_large: false,
                },
                inputs: vec![1, 2, 3, 4],
                saved_for_backward: false,
                checkpointed: false,
            },
            WengertOp {
                id: 6,
                result: 7,
                op: PrimalOp::FusedLinearCeBackwardExtract {
                    component: 0,
                    vocab_size,
                    hidden_size: HIDDEN,
                    batch_size: BATCH,
                    seq_len: SEQ,
                    vocab_tile: TILE,
                    ignore_index: -100,
                },
                // [output_bar, x, W, bias, targets, fwd_result]
                inputs: vec![5, 1, 2, 3, 4, 6],
                saved_for_backward: false,
                checkpointed: false,
            },
        ],
        output: 7,
        var_names: HashMap::new(),
        var_types,
    }
}

/// Drive the wengert lowering for `wengert` with `n_inputs` Cranelift
/// function params, returning the lowered `Function` plus a name -> FuncId
/// index map for the target FFI symbols.
fn lower_and_resolve(
    hint: Option<FusedCeDtypeHint>,
    vocab_size: u32,
    wengert: WengertList,
    n_inputs: usize,
    target_symbols: &[&str],
) -> (Function, HashMap<String, u32>) {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions {
        fused_ce_configs: vec![cfg_for_dtype_hint(hint, vocab_size)],
        ..CompileOptions::default()
    };
    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new must succeed");
    compiler
        .declare_runtime_functions()
        .expect("declare_runtime_functions must succeed");

    use cranelift_codegen::ir::types as cl_types;
    let mut sig = compiler.module.make_signature();
    sig.call_conv = compiler.call_conv;
    for _ in 0..n_inputs {
        sig.params.push(AbiParam::new(cl_types::I64));
    }
    sig.returns.push(AbiParam::new(cl_types::I64));

    let func_index = compiler.next_func_index();
    compiler
        .module
        .declare_function("test_v6_cast_lower", Linkage::Local, &sig)
        .expect("declare test fn");

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, func_index),
        sig,
    ));
    let mut fbctx = FunctionBuilderContext::new();
    {
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);
        let params = builder.block_params(entry).to_vec();
        let mut primal_vars: HashMap<VarId, cranelift_codegen::ir::Value> =
            HashMap::new();
        for (i, p) in params.iter().enumerate() {
            primal_vars.insert((i + 1) as VarId, *p);
        }
        let mut state = FuncState::default();
        let out_var = wengert.output;
        let lowered = compile_wengert_ops(
            &mut compiler,
            &mut builder,
            &mut state,
            &wengert,
            &primal_vars,
            None,
        )
        .expect("compile_wengert_ops must succeed under v6 cast insertion");
        let res = lowered
            .var_map
            .get(&out_var)
            .copied()
            .expect("output VarId must be in var_map");
        builder.ins().return_(&[res]);
        builder.finalize();
    }

    let mut name_to_index: HashMap<String, u32> = HashMap::new();
    for sym in target_symbols {
        if let Some((func_id, _)) = compiler.registry.runtime_fns.get(*sym).cloned() {
            name_to_index.insert((*sym).to_string(), func_id.as_u32());
        }
    }
    (ctx.func, name_to_index)
}

fn collect_calls(
    func: &Function,
) -> Vec<(u32, Vec<cranelift_codegen::ir::Value>, cranelift_codegen::ir::Inst)> {
    let mut out = Vec::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if let InstructionData::Call { func_ref, args, .. } = &func.dfg.insts[inst] {
                let ext = &func.dfg.ext_funcs[*func_ref];
                if let ExternalName::User(user_ref) = &ext.name {
                    let arg_list = args.as_slice(&func.dfg.value_lists).to_vec();
                    let user_name = func
                        .params
                        .user_named_funcs()
                        .get(*user_ref)
                        .expect("user_named_funcs entry for ext_func");
                    out.push((user_name.index, arg_list, inst));
                }
            }
        }
    }
    out
}

fn count_calls(func: &Function, sym_index: Option<u32>) -> usize {
    let Some(idx) = sym_index else { return 0 };
    collect_calls(func)
        .into_iter()
        .filter(|(i, _, _)| *i == idx)
        .count()
}

// ------------------------------ Forward small-vocab ------------------------------

#[test]
fn f32_hint_emits_no_precision_cast_forward_small() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::F32),
        SMALL_VOCAB,
        build_forward_wengert(SMALL_VOCAB, false),
        4,
        &["nsl_tensor_to_bf16", "nsl_tensor_to_fp16", "nsl_fused_linear_ce_forward"],
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()),
        0,
        "F32 hint MUST NOT emit nsl_tensor_to_bf16 calls (byte-identity guard).",
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()),
        0,
        "F32 hint MUST NOT emit nsl_tensor_to_fp16 calls (byte-identity guard).",
    );
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_forward").copied()) >= 1,
        "forward FFI must still fire under F32 hint",
    );
}

#[test]
fn absent_hint_emits_no_precision_cast_forward_small() {
    let (func, idx) = lower_and_resolve(
        None,
        SMALL_VOCAB,
        build_forward_wengert(SMALL_VOCAB, false),
        4,
        &["nsl_tensor_to_bf16", "nsl_tensor_to_fp16"],
    );
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()), 0);
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()), 0);
}

#[test]
fn fp16_hint_emits_three_to_fp16_calls_forward_small() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::F16),
        SMALL_VOCAB,
        build_forward_wengert(SMALL_VOCAB, false),
        4,
        &["nsl_tensor_to_bf16", "nsl_tensor_to_fp16", "nsl_fused_linear_ce_forward"],
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()),
        3,
        "FP16 hint MUST emit exactly 3 nsl_tensor_to_fp16 calls (x/W/bias).",
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()),
        0,
        "FP16 hint MUST NOT emit nsl_tensor_to_bf16 calls.",
    );
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_forward").copied()) >= 1,
        "forward FFI must still fire after cast insertion",
    );
}

#[test]
fn bf16_hint_emits_three_to_bf16_calls_forward_small() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::Bf16),
        SMALL_VOCAB,
        build_forward_wengert(SMALL_VOCAB, false),
        4,
        &["nsl_tensor_to_bf16", "nsl_tensor_to_fp16", "nsl_fused_linear_ce_forward"],
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()),
        3,
        "BF16 hint MUST emit exactly 3 nsl_tensor_to_bf16 calls (x/W/bias).",
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()),
        0,
        "BF16 hint MUST NOT emit nsl_tensor_to_fp16 calls.",
    );
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_forward").copied()) >= 1,
        "forward FFI must still fire after cast insertion",
    );
}

// ------------------------------ Forward large-vocab ------------------------------

#[test]
fn bf16_hint_emits_three_to_bf16_calls_forward_large() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::Bf16),
        LARGE_VOCAB,
        build_forward_wengert(LARGE_VOCAB, true),
        4,
        &[
            "nsl_tensor_to_bf16",
            "nsl_tensor_to_fp16",
            "nsl_fused_linear_ce_forward_large",
        ],
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()),
        3,
        "BF16 hint on the large-vocab path MUST also emit 3 cast calls — the \
         insertion site is BEFORE the is_large branch in \
         lower_fused_linear_ce_forward.",
    );
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()), 0);
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_forward_large").copied())
            >= 1,
        "large-vocab forward FFI must still fire",
    );
}

#[test]
fn f32_hint_emits_no_precision_cast_forward_large() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::F32),
        LARGE_VOCAB,
        build_forward_wengert(LARGE_VOCAB, true),
        4,
        &[
            "nsl_tensor_to_bf16",
            "nsl_tensor_to_fp16",
            "nsl_fused_linear_ce_forward_large",
        ],
    );
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()), 0);
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()), 0);
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_forward_large").copied())
            >= 1,
    );
}

// ------------------------------ Backward extract ------------------------------

#[test]
fn f32_hint_emits_no_precision_cast_backward() {
    // The backward extract is fired off a chained FusedLinearCeBackwardExtract;
    // the forward must run first (it populates compiler.fused_ce_fwd_lse).
    // Under F32 hint, NEITHER forward NOR backward should emit any cast call.
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::F32),
        SMALL_VOCAB,
        build_forward_plus_backward_wengert(SMALL_VOCAB),
        5,
        &[
            "nsl_tensor_to_bf16",
            "nsl_tensor_to_fp16",
            "nsl_fused_linear_ce_backward",
        ],
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()),
        0,
        "F32 hint MUST NOT emit cast calls on the backward path either.",
    );
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()), 0);
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_backward").copied()) >= 1,
        "backward FFI must still fire under F32 hint",
    );
}

/// CFTP v6 Findings 10/14: the forward+backward dispatch emits exactly
/// 3 cast calls — the forward casts are CACHED on the Compiler keyed by
/// the forward result Value and the backward extract REUSES them via
/// `compiler.fused_ce_fwd_casts.get(...)`. Without the cache the
/// implementation emitted 6 casts (forward 3 + backward 3), doubling
/// the docstring's quoted HBM cost. Pin the cached-reuse contract.
#[test]
fn fp16_hint_emits_three_to_fp16_calls_per_dispatch_with_backward_cache_reuse() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::F16),
        SMALL_VOCAB,
        build_forward_plus_backward_wengert(SMALL_VOCAB),
        5,
        &[
            "nsl_tensor_to_fp16",
            "nsl_tensor_to_bf16",
            "nsl_fused_linear_ce_forward",
            "nsl_fused_linear_ce_backward",
        ],
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()),
        3,
        "FP16 hint MUST emit exactly 3 cast calls per forward+backward dispatch \
         (Findings 10/14): forward emits the casts and stashes them in \
         `compiler.fused_ce_fwd_casts`; backward reuses the cached Values \
         instead of emitting 3 more. A regression to 6 means the backward \
         arm fell through to the cache-miss branch.",
    );
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()), 0);
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_forward").copied()) >= 1,
    );
    assert!(
        count_calls(&func, idx.get("nsl_fused_linear_ce_backward").copied()) >= 1,
    );
}

#[test]
fn bf16_hint_emits_three_to_bf16_calls_per_dispatch_with_backward_cache_reuse() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::Bf16),
        SMALL_VOCAB,
        build_forward_plus_backward_wengert(SMALL_VOCAB),
        5,
        &[
            "nsl_tensor_to_bf16",
            "nsl_tensor_to_fp16",
            "nsl_fused_linear_ce_forward",
            "nsl_fused_linear_ce_backward",
        ],
    );
    assert_eq!(
        count_calls(&func, idx.get("nsl_tensor_to_bf16").copied()),
        3,
        "BF16 hint MUST emit exactly 3 cast calls per forward+backward dispatch \
         (Findings 10/14 cache-reuse pin — same as fp16).",
    );
    assert_eq!(count_calls(&func, idx.get("nsl_tensor_to_fp16").copied()), 0);
}

// ------------------------------ Ordering: cast precedes data_ptr ------------------------------

/// Confirm that EVERY `nsl_tensor_to_bf16` call's result is consumed by a
/// subsequent `nsl_tensor_data_ptr` call. Without this guard, a future
/// refactor could emit the cast but never feed its result into the FFI
/// argument list — the kernel would silently read the f32 buffer.
#[test]
fn bf16_cast_results_flow_into_data_ptr_calls() {
    let (func, idx) = lower_and_resolve(
        Some(FusedCeDtypeHint::Bf16),
        SMALL_VOCAB,
        build_forward_wengert(SMALL_VOCAB, false),
        4,
        &["nsl_tensor_to_bf16", "nsl_tensor_data_ptr"],
    );
    let bf16_idx = *idx
        .get("nsl_tensor_to_bf16")
        .expect("nsl_tensor_to_bf16 must be registered");
    let data_ptr_idx = *idx
        .get("nsl_tensor_data_ptr")
        .expect("nsl_tensor_data_ptr must be registered");

    let calls = collect_calls(&func);
    let cast_results: Vec<_> = calls
        .iter()
        .filter(|(i, _, _)| *i == bf16_idx)
        .map(|(_, _, inst)| {
            // The Call inst has one result (the cast'd tensor pointer).
            func.dfg
                .inst_results(*inst)
                .first()
                .copied()
                .expect("cast call must produce a result")
        })
        .collect();
    assert_eq!(cast_results.len(), 3, "expected 3 cast results to track");

    let data_ptr_arg_consumers: std::collections::HashSet<_> = calls
        .iter()
        .filter(|(i, _, _)| *i == data_ptr_idx)
        .flat_map(|(_, args, _)| args.iter().copied())
        .collect();

    for cr in &cast_results {
        assert!(
            data_ptr_arg_consumers.contains(cr),
            "cast result {:?} not consumed by any nsl_tensor_data_ptr call — \
             the cast was emitted but its result never reached the FFI args; \
             kernel would silently read the f32 buffer.\nFunc:\n{}",
            cr,
            func.display(),
        );
    }
}
