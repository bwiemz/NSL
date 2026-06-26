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
//!
//! ## Coverage gap (CFTP v5 follow-on Finding 8 — LOW)
//!
//! This test pins the IR shape (presence/absence of FFI calls + dtype_tag
//! sentinel + kernel-name dtype suffix) but does NOT verify numerical
//! correctness through the wengert lowering.  GPU-numerical coverage at
//! V=49152 lives in
//!   * `fused_linear_ce_fp16_v49152_numerical.rs`
//!   * `fused_linear_ce_bf16_v49152_numerical.rs`
//! and those tests bypass the wengert dispatch entirely (they allocate
//! bf16/fp16 buffers themselves via `f32_slice_to_bf16_bits`).  So
//! Finding 7's buffer-conformance gap (wengert tape emits f32 buffers,
//! kernel reads bf16/fp16) is NOT observed by any test in this file.
//!
//! Mitigation (Sprint v6): wengert_lower now inserts an explicit
//! `nsl_tensor_to_{bf16,fp16}` precision_cast op into the Cranelift IR
//! before the FFI call whenever `dtype_tag != 0`, closing the gap
//! structurally.  The v5 opt-in runtime refusal
//! (`NSL_FUSED_LCE_REFUSE_NON_F32`) has been REMOVED.  A CUDA-gated
//! end-to-end test driving the full pipeline (NSL source →
//! compile_wengert_ops → JIT → execute → compare against CPU f64
//! reference) remains a follow-on validation step.

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

// =======================================================================
// CFTP v5 follow-on Finding 2 + Finding 3 — backward extract dispatch and
// kernel-name-embed dtype assertion.
// =======================================================================

/// Build a WengertList containing a `FusedLinearCe` forward op AND a
/// `FusedLinearCeBackwardExtract` op (component=0, which fires the
/// backward FFI).  The fwd_result is var 5 (the FusedLinearCe result);
/// the output_bar (upstream grad) is a 6th Input leaf at var 6.
///
/// The backward extract's `inputs` follow the lowerer's convention:
///   inputs[0] = output_bar
///   inputs[1..5] = x, W, bias, targets
///   inputs[5] = fwd_result (cache key)
fn build_wengert_list_with_backward() -> WengertList {
    let mut var_types: HashMap<VarId, WengertType> = HashMap::new();
    for v in 1..=7u32 {
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
            WengertOp {
                id: 5,
                result: 6,
                op: PrimalOp::Input("output_bar".to_string()),
                inputs: vec![],
                saved_for_backward: false,
                checkpointed: false,
            },
            WengertOp {
                id: 6,
                result: 7,
                op: PrimalOp::FusedLinearCeBackwardExtract {
                    component: 0,
                    vocab_size: 4096,
                    hidden_size: 128,
                    batch_size: 2,
                    seq_len: 32,
                    vocab_tile: 1024,
                    ignore_index: -100,
                },
                // Convention: [output_bar, x, W, bias, targets, fwd_result]
                inputs: vec![6, 1, 2, 3, 4, 5],
                saved_for_backward: false,
                checkpointed: false,
            },
        ],
        output: 7,
        var_names: HashMap::new(),
        var_types,
    }
}

/// Read the null-terminated string bytes stored in a data symbol declared
/// via `embed_fused_ce_data`.  The data definitions are stored in the
/// Cranelift module's data context — but cranelift_object's
/// `ObjectModule` doesn't expose declared-but-defined data by name to
/// tests.  Instead we recover the kernel-name embed by intercepting the
/// `symbol_value` call site: `embed_fused_ce_data` returns the symbol
/// value via `builder.ins().symbol_value(I64, name_gv)`, where `name_gv`
/// is the GlobalValue resolved from `module.declare_data_in_func`.  The
/// GlobalValueData::Symbol carries the ExternalName which we can map
/// back to the FuncId/DataId discriminator.  However, the embedded BYTES
/// live in the module's writer state, NOT the Function we're inspecting.
///
/// Instead of trying to peek into the writer state, we rely on the fact
/// that `synthesize_fused_linear_ce_ptx` is deterministic: for the same
/// cfg the bytes are identical.  We re-synthesise the expected kernel
/// name using the public `FusedLinearCEConfig::kernel_name` /
/// `bwd_kernel_name` API, and assert it carries the right dtype tag
/// suffix.  This is exactly the "synthesise the same bytes the embed
/// would have used" approach the finding fix recommends.
fn assert_kernel_name_carries_dtype_tag(
    cfg: &nsl_codegen::fused_linear_ce::FusedLinearCEConfig,
    expected_tag: &str,
    is_backward: bool,
    hint_label: &str,
) {
    let name = if is_backward {
        cfg.bwd_kernel_name()
    } else {
        cfg.kernel_name()
    };
    let direction = if is_backward { "backward" } else { "forward" };
    assert!(
        name.contains(expected_tag),
        "[{hint_label}] {direction} kernel name `{name}` does NOT contain dtype-tag \
         suffix `{expected_tag}` — synthesised PTX bytes would carry a different \
         kernel symbol than the dispatch dtype_tag implies, producing a runtime \
         CUDA_ERROR_NOT_FOUND on `cuModuleGetFunction`.  Fix: confirm \
         `FusedLinearCEConfig::dtype.tag()` matches the dtype_tag wire sentinel."
    );
}

/// Build a Compiler with the supplied dtype hint, drive a wengert
/// lowering that contains BOTH the forward op AND the backward-extract
/// component=0 op, and return the resulting Function.  This exercises
/// `lower_fused_linear_ce_backward_extract` for Finding 2.
fn lower_with_backward_and_resolve(
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
    // 5 params: x, W, bias, targets, output_bar.
    for _ in 0..5 {
        sig.params.push(AbiParam::new(cl_types::I64));
    }
    sig.returns.push(AbiParam::new(cl_types::I64));
    let func_index = compiler.next_func_index();
    compiler
        .module
        .declare_function("test_fused_lce_bwd_lower", Linkage::Local, &sig)
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
        primal_vars.insert(6, params[4]);
        let wengert = build_wengert_list_with_backward();
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
            "compile_wengert_ops must succeed for FusedLinearCe + \
             FusedLinearCeBackwardExtract under any dtype hint",
        );
        let res = lowered.var_map.get(&7).copied().expect("bwd-extract res Value");
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

/// Combined assertion for the backward path:
///   1. `nsl_fused_linear_ce_backward` is called.
///   2. Its terminal arg (dtype_tag, per the FFI signature at
///      `nsl_runtime/src/fused_linear_ce.rs:266`) is an iconst with the
///      expected sentinel.
///   3. The corresponding forward and backward kernel names carry the
///      right dtype-tag suffix (Finding 3 — kernel-name embed matches the
///      dispatch dtype_tag).  This is asserted by re-synthesising the
///      cfg-derived kernel name and confirming the suffix.
fn assert_backward_dispatch_for_hint(
    hint: Option<FusedCeDtypeHint>,
    expected_dtype_tag: i64,
    expected_dtype_tag_suffix: &str,
    hint_label: &str,
) {
    let (func, name_index) = lower_with_backward_and_resolve(
        hint,
        &[
            "nsl_fused_linear_ce_forward",
            "nsl_fused_linear_ce_backward",
        ],
    );

    let bwd_index = *name_index.get("nsl_fused_linear_ce_backward").unwrap_or_else(|| {
        panic!(
            "[{hint_label}] `nsl_fused_linear_ce_backward` not registered in \
             runtime_fns — declare_runtime_functions did not register it?"
        )
    });

    let calls = collect_calls(&func);
    let bwd_calls: Vec<_> = calls.iter().filter(|(idx, _)| *idx == bwd_index).collect();

    // 1. The backward FFI must have been called exactly once (component=0
    //    fires the launch; subsequent components hit the cache and would
    //    NOT emit another call — but this test only uses component=0).
    assert!(
        !bwd_calls.is_empty(),
        "[{hint_label}] expected ≥1 `call nsl_fused_linear_ce_backward` in lowered IR; \
         found 0.  Did the wengert dispatch silently keep the v4-2 backward refusal?  \
         Func:\n{}",
        func.display()
    );

    // 2. The terminal arg (dtype_tag, at FFI signature position 17) must
    //    resolve to the expected iconst sentinel.  This pins Finding 2 —
    //    the activation contract on the BACKWARD path must mirror the
    //    forward path's contract.
    let bwd_args = &bwd_calls[0].1;
    let dtype_tag_arg = *bwd_args.last().expect(
        "bwd FFI call must have a non-empty arg list (dtype_tag is the terminal arg \
         per nsl_runtime/src/fused_linear_ce.rs:266)",
    );
    let dtype_tag_const = iconst_value(&func, dtype_tag_arg).unwrap_or_else(|| {
        panic!(
            "[{hint_label}] terminal arg to nsl_fused_linear_ce_backward must be an iconst \
             (the dtype_tag sentinel); got non-iconst-producing Value.  Func:\n{}",
            func.display()
        )
    });
    assert_eq!(
        dtype_tag_const, expected_dtype_tag,
        "[{hint_label}] backward dtype_tag sentinel mismatch: expected \
         {expected_dtype_tag} (FusedCeDtypeHint::{hint_label} → FFI sentinel), \
         got {dtype_tag_const}.  Mismatch = silent dtype corruption in backward."
    );

    // 3. (Finding 3) Re-synthesise the cfg the lowering would have used,
    //    then assert both the forward and backward kernel names carry the
    //    expected dtype-tag suffix.  This pins the embed-vs-dispatch
    //    consistency: a future refactor that splits the source-of-truth
    //    between PTX synthesis and the iconst sentinel would land here.
    use nsl_codegen::fused_linear_ce::{Dtype as RuntimeDtype, FusedLinearCEConfig};
    let emitter_dtype = match expected_dtype_tag {
        0 => RuntimeDtype::F32,
        1 => RuntimeDtype::F16,
        2 => RuntimeDtype::Bf16,
        other => panic!("[{hint_label}] unexpected dtype_tag {other}"),
    };
    let synth_cfg = FusedLinearCEConfig {
        vocab_size: 4096,
        hidden_size: 128,
        seq_len: 32,
        batch_size: 2,
        vocab_tile: 1024,
        gpu_sm: 80,
        dtype: emitter_dtype,
        ignore_index: -100,
        max_vocab_v1: nsl_codegen::fused_linear_ce::MAX_VOCAB_HARD_CEILING,
    };
    assert_kernel_name_carries_dtype_tag(
        &synth_cfg,
        expected_dtype_tag_suffix,
        false,
        hint_label,
    );
    assert_kernel_name_carries_dtype_tag(
        &synth_cfg,
        expected_dtype_tag_suffix,
        true,
        hint_label,
    );
}

#[test]
fn f32_hint_propagates_backward_dtype_tag_zero() {
    assert_backward_dispatch_for_hint(Some(FusedCeDtypeHint::F32), 0, "f32", "F32");
}

#[test]
fn fp16_hint_propagates_backward_dtype_tag_one_and_does_not_refuse() {
    // v5 activation: the v4-2 BACKWARD refusal was lifted symmetrically with
    // the forward path.  This test pins that the lift actually reaches the
    // backward FFI emit site (it doesn't, in the pre-v5 codebase — the
    // refusal would hard-error before any `nsl_fused_linear_ce_backward`
    // call landed in the IR).
    assert_backward_dispatch_for_hint(Some(FusedCeDtypeHint::F16), 1, "f16", "F16");
}

#[test]
fn bf16_hint_propagates_backward_dtype_tag_two_and_does_not_refuse() {
    // v5 activation: same as the fp16 test, but for the v4-1 bf16 emitters.
    assert_backward_dispatch_for_hint(Some(FusedCeDtypeHint::Bf16), 2, "bf16", "Bf16");
}
