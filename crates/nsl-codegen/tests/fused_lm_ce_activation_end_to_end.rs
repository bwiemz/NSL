//! CFTP §4.4 G3 v5 — `@fused_lm_ce(dtype = "...")` end-to-end ACTIVATION test.
//!
//! Sprint v5 lifts the v4-2 refusal at `wengert_lower::lower_fused_linear_ce_forward`
//! / `..._backward_extract` that gated this dispatch on `dtype == F32`.  This test
//! pins the post-activation contract by driving a complete wengert lowering
//! against a synthetic `WengertList` containing a single `PrimalOp::FusedLinearCe`,
//! then walking the resulting Cranelift IR for:
//!
//!   1. A `call` instruction targeting `nsl_fused_linear_ce_forward`
//!      (small-vocab path; the large-vocab path is covered by the
//!      `synthesize_*_ptx` byte-difference tests in
//!      `fused_linear_ce_dtype_activation.rs` plus the GPU-numerical pins
//!      at V=49152 in `fused_linear_ce_{fp16,bf16}_v49152_numerical.rs`).
//!   2. NO `call` to `nsl_tensor_logsoftmax` (the composite fallback path
//!      is mutually exclusive with the fused dispatch — a regression here
//!      means the AD-side substitution silently fell back).
//!   3. The terminal `dtype_tag` arg (last positional arg to the FFI) resolves
//!      to the right `iconst(I64, …)` constant for each `FusedCeDtypeHint`:
//!         * F32  -> 0  (pre-v4-2 byte-identical default)
//!         * F16  -> 1  (v3-2 emitters)
//!         * Bf16 -> 2  (v4-1 emitters; v5-activated through wengert)
//!
//! Assertion 3 is the load-bearing one: the v4-2 refusal hard-errored out
//! before the FFI emit, so the bf16/fp16 sentinels were unreachable from
//! the wengert dispatch.  Confirming the `iconst(I64, 2)` lands on a real
//! `call nsl_fused_linear_ce_forward` instruction *is* the v5 activation
//! contract — any future regression that re-introduces a refusal, silently
//! drops dtype to F32, or rewires the dispatch through `nsl_tensor_logsoftmax`
//! would flip exactly one of the three asserts in this file.
//!
//! The IR walk searches by FFI symbol NAME (resolved via the Compiler's
//! `runtime_fns` registry → `FuncId` → the corresponding ext_func in the
//! built `Function`) — NOT by instruction index — so future Cranelift IR
//! layout changes (re-ordering, intermediate ops added, etc.) do not break
//! the contract silently.  They would surface as one of the lookups
//! returning `None` with an explicit panic message.
//!
//! No CUDA / no JIT execution — pure Cranelift IR shape assertion.

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

/// Build a FusedCeDecoratorConfig for a small-vocab program.
fn cfg_for_dtype_hint(hint: Option<FusedCeDtypeHint>) -> FusedCeDecoratorConfig {
    FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: hint,
    }
}

/// Build a minimal WengertList with four Input leaves (x, W, bias, targets)
/// followed by a single `PrimalOp::FusedLinearCe`.
fn build_wengert_list() -> WengertList {
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
                    vocab_size: 4096,
                    hidden_size: 128,
                    batch_size: 2,
                    seq_len: 32,
                    vocab_tile: 1024,
                    ignore_index: -100,
                    is_large: false,
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

/// Build a Compiler with the supplied dtype hint, drive a single
/// `PrimalOp::FusedLinearCe` lowering, and return the resulting Cranelift
/// `Function`.  The function takes four `i64` parameters (one per Input
/// leaf) as a stand-in for `nsl_tensor_data_ptr`-shaped tensor handles.
fn lower_to_function(hint: Option<FusedCeDtypeHint>) -> Function {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions {
        fused_ce_configs: vec![cfg_for_dtype_hint(hint)],
        ..CompileOptions::default()
    };

    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new must succeed");
    compiler
        .declare_runtime_functions()
        .expect(
            "declare_runtime_functions must succeed — needed so \
             `nsl_fused_linear_ce_forward` is in `runtime_fns` for the \
             wengert lowering to resolve via compile_call_by_name.",
        );

    let mut sig = compiler.module.make_signature();
    sig.call_conv = compiler.call_conv;
    use cranelift_codegen::ir::types as cl_types;
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.returns.push(AbiParam::new(cl_types::I64));

    let func_index = compiler.next_func_index();
    compiler
        .module
        .declare_function("test_fused_lce_lower", Linkage::Local, &sig)
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
        let mut primal_vars: HashMap<VarId, cranelift_codegen::ir::Value> = HashMap::new();
        primal_vars.insert(1, params[0]);
        primal_vars.insert(2, params[1]);
        primal_vars.insert(3, params[2]);
        primal_vars.insert(4, params[3]);
        let wengert = build_wengert_list();
        let mut state = FuncState::default();
        let lowered = compile_wengert_ops(
            &mut compiler,
            &mut builder,
            &mut state,
            &wengert,
            &primal_vars,
            None,
        )
        .expect(
            "compile_wengert_ops must succeed for FusedLinearCe under any dtype hint — \
             v5 ACTIVATION: the v4-2 refusal on fp16/bf16 has been lifted.",
        );
        let res = lowered
            .var_map
            .get(&5)
            .copied()
            .expect("FusedLinearCe result VarId must be in var_map");
        builder.ins().return_(&[res]);
        builder.finalize();
    }
    ctx.func
}

/// Walk every call instruction in `func` and return the per-call argument
/// lists, paired with the `UserFuncName` index that identifies the callee.
/// The caller cross-references those indices against the Compiler's
/// runtime_fns table to resolve them to FFI symbol names.
fn collect_calls(
    func: &Function,
) -> Vec<(u32, Vec<cranelift_codegen::ir::Value>)> {
    let mut out = Vec::new();
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            if let InstructionData::Call { func_ref, args, .. } = &func.dfg.insts[inst] {
                let ext = &func.dfg.ext_funcs[*func_ref];
                if let ExternalName::User(user_ref) = &ext.name {
                    let arg_list = args.as_slice(&func.dfg.value_lists).to_vec();
                    // `user_ref` is a `UserExternalNameRef` — look up the
                    // underlying `UserExternalName` (namespace, index) and
                    // return `index` as the FuncId discriminator.  This
                    // matches the value `cranelift_module::declare_func_in_func`
                    // stamps into the FuncRef → FuncId namespace.
                    let user_name = func
                        .params
                        .user_named_funcs()
                        .get(*user_ref)
                        .expect("user_named_funcs entry for ext_func");
                    out.push((user_name.index, arg_list));
                }
            }
        }
    }
    out
}

/// Read the iconst constant produced by a Value, returning the literal i64
/// if the producing instruction is `Iconst`.  Used to confirm the dtype_tag
/// argument resolves to the expected sentinel.
fn iconst_value(func: &Function, v: cranelift_codegen::ir::Value) -> Option<i64> {
    let inst = func.dfg.value_def(v).inst()?;
    if let InstructionData::UnaryImm { opcode, imm } = &func.dfg.insts[inst] {
        if *opcode == cranelift_codegen::ir::Opcode::Iconst {
            return Some(imm.bits());
        }
    }
    None
}

/// Combined lowering + name-resolution helper.  Builds the Compiler, drives
/// the wengert lowering, then returns the func plus a mapping from FFI
/// symbol name → the discriminator value the Cranelift IR uses for calls
/// to that symbol.
fn lower_and_resolve(
    hint: Option<FusedCeDtypeHint>,
    target_symbols: &[&str],
) -> (Function, HashMap<String, u32>) {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions {
        fused_ce_configs: vec![cfg_for_dtype_hint(hint)],
        ..CompileOptions::default()
    };

    let mut compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new must succeed");
    compiler
        .declare_runtime_functions()
        .expect("declare_runtime_functions must succeed");

    let mut sig = compiler.module.make_signature();
    sig.call_conv = compiler.call_conv;
    use cranelift_codegen::ir::types as cl_types;
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.returns.push(AbiParam::new(cl_types::I64));
    let func_index = compiler.next_func_index();
    compiler
        .module
        .declare_function("test_fused_lce_lower", Linkage::Local, &sig)
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
        let mut primal_vars: HashMap<VarId, cranelift_codegen::ir::Value> = HashMap::new();
        primal_vars.insert(1, params[0]);
        primal_vars.insert(2, params[1]);
        primal_vars.insert(3, params[2]);
        primal_vars.insert(4, params[3]);
        let wengert = build_wengert_list();
        let mut state = FuncState::default();
        let lowered = compile_wengert_ops(
            &mut compiler,
            &mut builder,
            &mut state,
            &wengert,
            &primal_vars,
            None,
        )
        .expect("compile_wengert_ops must succeed under v5 activation");
        let res = lowered.var_map.get(&5).copied().expect("res Value");
        builder.ins().return_(&[res]);
        builder.finalize();
    }

    // Resolve each target symbol name → its discriminator (`UserExternalName.index`)
    // via the Compiler's runtime_fns registry.  cranelift_module's
    // `declare_func_in_func` writes `UserExternalName { namespace: 0, index: FuncId.as_u32() }`,
    // so the `index` we see at the call site equals the FuncId for that symbol.
    let mut name_to_index: HashMap<String, u32> = HashMap::new();
    for sym in target_symbols {
        if let Some((func_id, _)) = compiler.registry.runtime_fns.get(*sym).cloned() {
            name_to_index.insert((*sym).to_string(), func_id.as_u32());
        }
    }
    (ctx.func, name_to_index)
}

fn assert_dispatch_for_hint(
    hint: Option<FusedCeDtypeHint>,
    expected_dtype_tag: i64,
    hint_label: &str,
) {
    let (func, name_index) = lower_and_resolve(
        hint,
        &["nsl_fused_linear_ce_forward", "nsl_tensor_logsoftmax"],
    );

    let fwd_index = *name_index.get("nsl_fused_linear_ce_forward").unwrap_or_else(|| {
        panic!(
            "[{hint_label}] `nsl_fused_linear_ce_forward` not registered in \
             runtime_fns — declare_runtime_functions did not register it?"
        )
    });
    let logsoftmax_index = name_index.get("nsl_tensor_logsoftmax").copied();

    let calls = collect_calls(&func);
    let fwd_calls: Vec<_> = calls.iter().filter(|(idx, _)| *idx == fwd_index).collect();

    // 1. The forward FFI must have been called at least once.
    assert!(
        !fwd_calls.is_empty(),
        "[{hint_label}] expected ≥1 `call nsl_fused_linear_ce_forward` in lowered IR; \
         found 0.  Did the wengert dispatch refuse, or route through a different FFI?  \
         Lowered func dump:\n{}",
        func.display()
    );

    // 2. The composite-path symbol must NOT be present in any call.  When
    //    `nsl_tensor_logsoftmax` is not even registered for this lowering,
    //    it's trivially absent — only check if its symbol IS registered.
    if let Some(ls_index) = logsoftmax_index {
        let ls_calls: Vec<_> = calls.iter().filter(|(idx, _)| *idx == ls_index).collect();
        assert!(
            ls_calls.is_empty(),
            "[{hint_label}] unexpected `call nsl_tensor_logsoftmax` in lowered IR — \
             FusedLinearCe should NOT route through the composite path.  Func:\n{}",
            func.display()
        );
    }

    // 3. The terminal `dtype_tag` arg (last positional arg per the FFI
    //    signature: see nsl_runtime/src/fused_linear_ce.rs) must resolve
    //    to the expected iconst sentinel.
    let fwd_args = &fwd_calls[0].1;
    let dtype_tag_arg = *fwd_args.last().expect(
        "fwd FFI call must have a non-empty arg list (dtype_tag is the terminal arg)",
    );
    let dtype_tag_const = iconst_value(&func, dtype_tag_arg).unwrap_or_else(|| {
        panic!(
            "[{hint_label}] terminal arg to nsl_fused_linear_ce_forward must be an iconst \
             (the dtype_tag sentinel); got non-iconst-producing Value.  Func:\n{}",
            func.display()
        )
    });
    assert_eq!(
        dtype_tag_const, expected_dtype_tag,
        "[{hint_label}] dtype_tag sentinel mismatch: expected {expected_dtype_tag} \
         (FusedCeDtypeHint::{hint_label} → FFI sentinel), got {dtype_tag_const}.  \
         Mismatch = silent dtype corruption at runtime (kernel reads wrong byte width)."
    );
}

#[test]
fn f32_hint_propagates_dtype_tag_zero() {
    assert_dispatch_for_hint(Some(FusedCeDtypeHint::F32), 0, "F32");
}

#[test]
fn absent_hint_propagates_dtype_tag_zero() {
    // No dtype kwarg on the decorator → cfg.dtype = None → wengert dispatch
    // falls back to F32 (byte-identical with pre-v4-2 baseline).
    assert_dispatch_for_hint(None, 0, "<absent>");
}

#[test]
fn fp16_hint_propagates_dtype_tag_one_and_does_not_refuse() {
    // v5 activation: the v4-2 refusal previously hard-errored at this dispatch
    // for `Some(F16)`.  Successful lowering + `iconst(1)` on the FFI dtype_tag
    // arg = activation contract held.
    assert_dispatch_for_hint(Some(FusedCeDtypeHint::F16), 1, "F16");
}

#[test]
fn bf16_hint_propagates_dtype_tag_two_and_does_not_refuse() {
    // v5 activation: same as the fp16 test, but for the v4-1 bf16 emitters.
    // GPU numerical at V=49152 was validated by the prior sprint via direct
    // FFI tests (`fused_linear_ce_bf16_v49152_numerical.rs`) — this test
    // pins the wengert dispatch contract so the IR actually emits the
    // dtype_tag=2 call site that the runtime hot path consumes.
    assert_dispatch_for_hint(Some(FusedCeDtypeHint::Bf16), 2, "Bf16");
}

#[test]
fn unused_lowering_helper_is_reachable() {
    // The `lower_to_function` helper is kept around as a leaner alternative
    // entry point for future tests that just need the Function (no name
    // index map).  Touch it here so dead-code lints stay quiet without
    // adding `#[allow(dead_code)]` clutter.
    let func = lower_to_function(None);
    assert!(
        func.layout.blocks().next().is_some(),
        "lowered function must have at least one block"
    );
}
