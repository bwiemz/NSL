//! CFTP v6 Sprint 3 — `@fused_lm_ce(dtype = "bf16")` end-to-end ACTIVATION
//! test.  Mirror of the fp16 e2e test for the v4-1 bf16 emitter path
//! (dtype_tag=2).
//!
//! See `fused_lm_ce_e2e_fp16_activation.rs` for the two-phase rationale
//! and structural-vs-numerical-coverage explanation.  Differences vs the
//! fp16 sibling:
//!
//!   * Decorator hint = `FusedCeDtypeHint::Bf16`.
//!   * Phase 1 asserts three `nsl_tensor_to_bf16` calls and dtype_tag=2.
//!   * Phase 2 uses `bf16::from_f32` for the host pre-rounding (the
//!     `half` crate's bf16 rounding matches the PTX `cvt.rn.bf16.f32`
//!     instruction the v4-1 bf16 emitters use).
//!
//! ## Numerical tolerance
//!
//! Bf16 trades 3 mantissa bits for 5 exponent bits vs fp16, so per-logit
//! precision is ~8e-3 RELATIVE (vs ~1e-3 for fp16) but the LSE rescale
//! chain has substantially wider dynamic range and no underflow risk.
//! The orchestrator's pinned tolerance `rel_err <= 5e-3` is shared with
//! fp16 — the aggregate `1/sqrt(num_valid * V)` averaging brings the
//! per-logit noise down well below this floor at V=4096.
//! Empirical baseline from
//! `fused_linear_ce_bf16_v49152_numerical::bf16_forward_backward_at_v49152_production_scale`:
//! mean_loss rel_err ≈ 2.8e-7 at V=49152, which extrapolates to ~1e-5 at
//! V=4096 — ~500× margin under the orchestrator tolerance.

#![cfg(feature = "test-helpers")]

mod common;

use common::fused_lce_cpu_f64::{cpu_lce_forward_f64, IGNORE_INDEX as CPU_IGNORE_INDEX};

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
use nsl_codegen::{CompileOptions, FusedCeDecoratorConfig, FusedCeDtypeHint};
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

use std::collections::HashMap;

const B: usize = 2;
const S: usize = 32;
const V: usize = 4096;
const H: usize = 128;
const VOCAB_TILE: u32 = 1024;
const IGNORE_INDEX: i64 = -100;

const _IGNORE_INDEX_MATCH: () = assert!(
    (IGNORE_INDEX as i32) == CPU_IGNORE_INDEX,
    "IGNORE_INDEX must match common::fused_lce_cpu_f64::IGNORE_INDEX"
);

fn cfg_bf16() -> FusedCeDecoratorConfig {
    FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(VOCAB_TILE),
        vocab_size: Some(V as u32),
        hidden_size: Some(H as u32),
        batch_size: Some(B as u32),
        seq_len: Some(S as u32),
        dtype: Some(FusedCeDtypeHint::Bf16),
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    }
}

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
                    vocab_size: V as u32,
                    hidden_size: H as u32,
                    batch_size: B as u32,
                    seq_len: S as u32,
                    vocab_tile: VOCAB_TILE,
                    ignore_index: IGNORE_INDEX,
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

fn lower_and_resolve(
    cfg: FusedCeDecoratorConfig,
    target_symbols: &[&str],
) -> (Function, HashMap<String, u32>) {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions {
        fused_ce_configs: vec![cfg],
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
    for _ in 0..4 {
        sig.params.push(AbiParam::new(cl_types::I64));
    }
    sig.returns.push(AbiParam::new(cl_types::I64));

    let func_index = compiler.next_func_index();
    compiler
        .module
        .declare_function("test_fused_lce_e2e_bf16", Linkage::Local, &sig)
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
            "compile_wengert_ops must succeed under @fused_lm_ce(dtype=\"bf16\") — \
             v6 ACTIVATION: the v4-2 refusal has been lifted and the v5 runtime \
             refusal has been removed in favour of structural precision_cast.",
        );
        let res = lowered.var_map.get(&5).copied().expect("FusedLinearCe result");
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

fn collect_calls(func: &Function) -> Vec<(u32, Vec<cranelift_codegen::ir::Value>)> {
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
                    out.push((user_name.index, arg_list));
                }
            }
        }
    }
    out
}

fn iconst_value(func: &Function, v: cranelift_codegen::ir::Value) -> Option<i64> {
    let inst = func.dfg.value_def(v).inst()?;
    if let InstructionData::UnaryImm { opcode, imm } = &func.dfg.insts[inst] {
        if *opcode == cranelift_codegen::ir::Opcode::Iconst {
            return Some(imm.bits());
        }
    }
    None
}

// ─── Phase 1: structural IR proof ─────────────────────────────────────────

/// Pin the structural contract for `@fused_lm_ce(dtype="bf16")`:
///   1. Three `nsl_tensor_to_bf16` calls (x / W / bias).
///   2. `nsl_fused_linear_ce_forward` is called.
///   3. Its terminal dtype_tag arg = iconst(I64, 2) — the Bf16 sentinel.
///   4. No `nsl_tensor_logsoftmax` (no composite fallback).
#[test]
fn bf16_decorator_emits_three_casts_plus_forward_with_dtype_tag_two() {
    let (func, name_index) = lower_and_resolve(
        cfg_bf16(),
        &[
            "nsl_fused_linear_ce_forward",
            "nsl_tensor_to_bf16",
            "nsl_tensor_logsoftmax",
        ],
    );

    let fwd_index = *name_index
        .get("nsl_fused_linear_ce_forward")
        .expect("nsl_fused_linear_ce_forward must be registered");
    let to_bf16_index = *name_index
        .get("nsl_tensor_to_bf16")
        .expect("nsl_tensor_to_bf16 must be registered");
    let logsoftmax_index = name_index.get("nsl_tensor_logsoftmax").copied();

    let calls = collect_calls(&func);

    let cast_calls: Vec<_> = calls
        .iter()
        .filter(|(idx, _)| *idx == to_bf16_index)
        .collect();
    assert_eq!(
        cast_calls.len(),
        3,
        "expected EXACTLY 3 `nsl_tensor_to_bf16` calls in lowered IR; found {}.  \
         Func:\n{}",
        cast_calls.len(),
        func.display()
    );

    let fwd_calls: Vec<_> = calls.iter().filter(|(idx, _)| *idx == fwd_index).collect();
    assert!(
        !fwd_calls.is_empty(),
        "expected ≥1 `nsl_fused_linear_ce_forward` call; got 0.  Func:\n{}",
        func.display()
    );

    let fwd_args = &fwd_calls[0].1;
    let dtype_tag_arg = *fwd_args.last().expect("non-empty arg list");
    let dtype_tag_const = iconst_value(&func, dtype_tag_arg).unwrap_or_else(|| {
        panic!(
            "terminal arg to nsl_fused_linear_ce_forward must be iconst; got non-iconst.  \
             Func:\n{}",
            func.display()
        )
    });
    assert_eq!(
        dtype_tag_const, 2,
        "expected dtype_tag=2 (Bf16) on `nsl_fused_linear_ce_forward`; got {dtype_tag_const}."
    );

    if let Some(ls_idx) = logsoftmax_index {
        let ls_calls: Vec<_> = calls.iter().filter(|(idx, _)| *idx == ls_idx).collect();
        assert!(
            ls_calls.is_empty(),
            "unexpected `nsl_tensor_logsoftmax` call under bf16 decorator.  Func:\n{}",
            func.display()
        );
    }
}

// ─── Phase 2: CUDA-gated GPU numerical proof ──────────────────────────────

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use half::bf16;
    use nsl_codegen::fused_linear_ce::{
        synthesize_fused_linear_ce_ptx, Dtype, FusedLinearCEConfig, MAX_VOCAB_HARD_CEILING,
    };
    use nsl_runtime::{
        nsl_cuda_init, nsl_fused_linear_ce_forward, nsl_test_cuda_alloc, nsl_test_cuda_d2h,
        nsl_test_cuda_free, nsl_test_cuda_h2d,
    };

    const DTYPE_TAG_BF16: i64 = 2;

    fn cuda_available() -> bool {
        if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
            eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
            return false;
        }
        let rc = unsafe { nsl_cuda_init() };
        if rc != 0 {
            eprintln!("skipping: nsl_cuda_init returned {rc}");
            false
        } else {
            true
        }
    }

    fn fill_seeded(dst: &mut [f32], seed: u64, lo: f32, hi: f32) {
        let mut s = seed;
        let span = hi - lo;
        for x in dst.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (s >> 33) as u32;
            *x = lo + ((u as f32) / (u32::MAX as f32)) * span;
        }
    }

    fn f32_slice_to_bf16_bits(src: &[f32], dst: &mut [u16]) {
        assert_eq!(src.len(), dst.len());
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = bf16::from_f32(*s).to_bits();
        }
    }

    fn round_through_bf16(src: &[f32]) -> Vec<f32> {
        src.iter().map(|&x| bf16::from_f32(x).to_f32()).collect()
    }

    fn f32_to_f64(src: &[f32]) -> Vec<f64> {
        src.iter().map(|&x| x as f64).collect()
    }

    fn i64_to_i32_targets(src: &[i64]) -> Vec<i32> {
        src.iter()
            .map(|&t| {
                if t == IGNORE_INDEX {
                    CPU_IGNORE_INDEX
                } else {
                    t as i32
                }
            })
            .collect()
    }

    /// End-to-end bf16 ACTIVATION numerical pin — small-vocab path.
    ///
    /// Tolerance: forward loss rel_err <= 5e-3 (orchestrator spec; shared
    /// with the fp16 sibling).
    #[test]
    #[ignore]
    fn bf16_decorator_e2e_loss_matches_cpu_f64_reference() {
        if !cuda_available() {
            return;
        }

        let cfg = FusedLinearCEConfig {
            vocab_size: V as u32,
            hidden_size: H as u32,
            seq_len: S as u32,
            batch_size: B as u32,
            vocab_tile: VOCAB_TILE,
            gpu_sm: 80,
            dtype: Dtype::Bf16,
            ignore_index: IGNORE_INDEX,
            max_vocab_v1: MAX_VOCAB_HARD_CEILING,
        };
        cfg.validate().expect("V=4096 bf16 cfg must validate");
        assert!(
            !cfg.is_large_vocab(),
            "V=4096 must route through the small-vocab path"
        );

        let rows = B * S;

        let mut targets = vec![0i64; rows];
        for (i, t) in targets.iter_mut().enumerate() {
            *t = if i % 8 == 0 {
                IGNORE_INDEX
            } else {
                ((i.wrapping_mul(37).wrapping_add(13)) % V) as i64
            };
        }
        let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count();
        eprintln!("bf16 e2e: rows={rows} num_valid={num_valid}");

        let mut x_f32 = vec![0f32; rows * H];
        let mut w_f32 = vec![0f32; V * H];
        let bias_f32 = vec![0f32; V];
        fill_seeded(&mut x_f32, 42, -0.3, 0.3);
        fill_seeded(&mut w_f32, 137, -0.05, 0.05);

        let x_ref = round_through_bf16(&x_f32);
        let w_ref = round_through_bf16(&w_f32);
        let bias_ref = round_through_bf16(&bias_f32);

        let mut x_h16 = vec![0u16; rows * H];
        let mut w_h16 = vec![0u16; V * H];
        let mut bias_h16 = vec![0u16; V];
        f32_slice_to_bf16_bits(&x_f32, &mut x_h16);
        f32_slice_to_bf16_bits(&w_f32, &mut w_h16);
        f32_slice_to_bf16_bits(&bias_f32, &mut bias_h16);

        let mut fwd_ptx = synthesize_fused_linear_ce_ptx(&cfg);
        fwd_ptx.push(0u8);
        let fwd_name = format!("{}\0", cfg.kernel_name());

        let x_bytes = (rows * H * 2) as i64;
        let w_bytes = (V * H * 2) as i64;
        let bias_bytes = (V * 2) as i64;
        let tgt_bytes = (rows * 8) as i64;
        let loss_bytes = (rows * 4) as i64;
        let lse_bytes = (rows * 4) as i64;

        let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
        let w_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
        let bias_dev = unsafe { nsl_test_cuda_alloc(bias_bytes) };
        let tgt_dev = unsafe { nsl_test_cuda_alloc(tgt_bytes) };
        let loss_dev = unsafe { nsl_test_cuda_alloc(loss_bytes) };
        let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };
        assert!(
            x_dev != 0 && w_dev != 0 && bias_dev != 0 && tgt_dev != 0
                && loss_dev != 0 && lse_dev != 0,
            "device alloc failed"
        );

        unsafe {
            nsl_test_cuda_h2d(x_dev, x_h16.as_ptr() as i64, x_bytes);
            nsl_test_cuda_h2d(w_dev, w_h16.as_ptr() as i64, w_bytes);
            nsl_test_cuda_h2d(bias_dev, bias_h16.as_ptr() as i64, bias_bytes);
            nsl_test_cuda_h2d(tgt_dev, targets.as_ptr() as i64, tgt_bytes);
        }

        let x_ref_f64 = f32_to_f64(&x_ref);
        let w_ref_f64 = f32_to_f64(&w_ref);
        let bias_ref_f64 = f32_to_f64(&bias_ref);
        let targets_i32 = i64_to_i32_targets(&targets);
        let cpu_fwd =
            cpu_lce_forward_f64(&x_ref_f64, &w_ref_f64, &bias_ref_f64, &targets_i32, rows, V, H);
        eprintln!("CPU f64 mean_loss = {:.6}", cpu_fwd.mean_loss);

        let smem = cfg.shared_mem_bytes();
        let rc = nsl_fused_linear_ce_forward(
            fwd_ptx.as_ptr() as i64,
            fwd_name.as_ptr() as i64,
            x_dev,
            w_dev,
            bias_dev,
            tgt_dev,
            loss_dev,
            lse_dev,
            B as i64,
            S as i64,
            V as i64,
            H as i64,
            smem as i64,
            DTYPE_TAG_BF16,
        );
        assert_eq!(rc, 0, "nsl_fused_linear_ce_forward (bf16) failed rc={rc}");

        let mut loss_gpu = vec![0f32; rows];
        unsafe {
            nsl_test_cuda_d2h(loss_gpu.as_mut_ptr() as i64, loss_dev, loss_bytes);
        }

        let mut gpu_loss_sum = 0f64;
        let mut gpu_nv = 0usize;
        for (i, &lv) in loss_gpu.iter().enumerate() {
            if targets[i] == IGNORE_INDEX {
                assert_eq!(lv, 0.0, "skip-identity: loss row={i} = {lv}");
                continue;
            }
            gpu_loss_sum += lv as f64;
            gpu_nv += 1;
        }
        let gpu_mean = gpu_loss_sum / gpu_nv.max(1) as f64;
        let rel_err =
            (gpu_mean - cpu_fwd.mean_loss).abs() / cpu_fwd.mean_loss.abs().max(1.0);
        eprintln!(
            "bf16 e2e fwd: gpu={:.6} cpu={:.6} rel_err={:.3e}",
            gpu_mean, cpu_fwd.mean_loss, rel_err
        );

        assert!(
            rel_err < 5e-3,
            "bf16 decorator-e2e mean_loss rel_err {rel_err:.3e} exceeds orchestrator \
             tolerance 5e-3; a real bf16 kernel-level regression (broken cvt.rn.bf16, \
             wrong rescale, dtype_tag mismatch, ISA<8.0 emit) is the most likely \
             cause."
        );

        unsafe {
            nsl_test_cuda_free(x_dev);
            nsl_test_cuda_free(w_dev);
            nsl_test_cuda_free(bias_dev);
            nsl_test_cuda_free(tgt_dev);
            nsl_test_cuda_free(loss_dev);
            nsl_test_cuda_free(lse_dev);
        }
    }
}
