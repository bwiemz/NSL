//! T7.1 — AD dispatcher fallback regression.
//!
//! When the Tier C backward validator rejects a CSHA chain's config
//! (e.g. SMEM budget exceeded under `Direction::Backward`), the AD
//! dispatcher MUST fall back to per-op adjoint rules — the one-kernel
//! fused path cannot be emitted. This test pins that behaviour so a
//! future regression in `csha_dispatch_for_op` (or in the validator's
//! direction-awareness) is caught early.
//!
//! Integration shape:
//!
//!   1. Build a CshaExtras + FlashAttentionConfig whose
//!      `Direction::Backward` validation fails (backward_extra_bytes
//!      pushes the total over the 99 KB cap).
//!   2. Build a minimal Wengert list modelling a single
//!      norm → matmul → rope chain (the four ops a CSHA claim covers).
//!   3. Pack into a FusionMark carrying that config.
//!   4. Invoke `csha_dispatch_for_op` once and — on Fallback —
//!      invoke `apply_ad_rule` per claimed op. Collect the adjoint
//!      count + the fallback diagnostic.
//!   5. Assert the per-op fallback fired (emitted count > 1, i.e. NOT
//!      a single fused call) and the diagnostic carries the strings
//!      the plan requires ("CSHA fused backward rejected", "bytes >",
//!      "falling back" [phrase synthesised by the caller]).

use nsl_codegen::ad_rules::{
    apply_ad_rule, csha_dispatch_for_op, AdjointExpr, CshaDispatchDecision,
    InputAdjoint,
};
use nsl_codegen::csha_apply::{FusionMark, MarkRole};
use nsl_codegen::csha_boundary::ProjKind;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::wengert::{PrimalOp, WengertOp};

fn backward_over_budget_config() -> FlashAttentionConfig {
    // Matches T2.1's `direction_backward_rejects_over_budget_with_detailed_diagnostic`
    // fixture: (64,64,64,8,64) — forward fits, backward's P+dQ+dK+dV
    // extra bytes push it past the 99 KB dynamic-SMEM cap.
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 64,
            ..CshaExtras::default()
        }),
    }
}

fn claimed_chain_ops() -> Vec<WengertOp> {
    // norm → matmul_q → rope_q (one of the Q/K/V legs). Enough to
    // exercise three distinct adjoint rules in the fallback path.
    vec![
        WengertOp {
            id: 0, result: 0, op: PrimalOp::Input("x".into()),
            inputs: vec![], saved_for_backward: true, checkpointed: false,
        },
        WengertOp {
            id: 1, result: 1, op: PrimalOp::RMSNorm { eps: 1e-5 },
            inputs: vec![0], saved_for_backward: true, checkpointed: false,
        },
        WengertOp {
            id: 2, result: 2, op: PrimalOp::Param("blocks.0.attn.wq".into()),
            inputs: vec![], saved_for_backward: false, checkpointed: false,
        },
        WengertOp {
            id: 3, result: 3, op: PrimalOp::Matmul,
            inputs: vec![1, 2], saved_for_backward: true, checkpointed: false,
        },
        WengertOp {
            id: 4, result: 4, op: PrimalOp::RoPE { dim: 64 },
            inputs: vec![3], saved_for_backward: true, checkpointed: false,
        },
    ]
}

fn mark_for_layer(cfg: FlashAttentionConfig) -> FusionMark {
    FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "blocks.0.attn.wq".into(),
        role: MarkRole::NormPrologue,
        config: Some(cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: None,
    }
}

#[derive(Debug)]
struct DispatchResult {
    emitted_op_count: usize,
    diagnostics: Vec<String>,
    emitted_adjoints: Vec<InputAdjoint>,
}

/// Simulates what the reverse-walk will do once T7.1's compiler-level
/// wire-up lands (currently deferred to @train integration). Walks
/// ops in reverse topological order. On the first encounter of a
/// claimed op:
///   - EmitFused: emit exactly one fused kernel call (represented
///     here by a single synthetic `InputAdjoint` so the count is 1).
///   - AlreadyEmitted: skip.
///   - Fallback: push the diagnostic and emit per-op adjoints for
///     every op in the chain via `apply_ad_rule`.
fn simulate_reverse_walk(ops: &[WengertOp], mark: &FusionMark) -> DispatchResult {
    let mut diagnostics: Vec<String> = Vec::new();
    let mut emitted: Vec<InputAdjoint> = Vec::new();

    // Walk in reverse (output op encountered first per spec §5.4).
    // Every op in this synthetic list is part of the claimed chain.
    let mut in_fallback = false;
    for op in ops.iter().rev() {
        match csha_dispatch_for_op(mark, op.id) {
            CshaDispatchDecision::EmitFused => {
                // Single synthetic adjoint representing the fused
                // kernel call. Real implementation would invoke
                // synthesize_backward + nsl_flash_attention_csha_backward.
                emitted.push(InputAdjoint {
                    input_var: 0,
                    expr: AdjointExpr::Identity(op.result),
                });
                // Flag is now set by csha_dispatch_for_op — subsequent
                // claimed ops return AlreadyEmitted and skip.
            }
            CshaDispatchDecision::AlreadyEmitted => {
                // Gradients for this op already produced by the earlier
                // fused emission. Nothing to do.
            }
            CshaDispatchDecision::Fallback { diagnostic } => {
                if !in_fallback {
                    diagnostics.push(format!(
                        "{diagnostic}; falling back to unfused backward"
                    ));
                    in_fallback = true;
                }
                // Per-op adjoint rules for the fallback path.
                let adjoints = apply_ad_rule(op, /*output_bar=*/ 100 + op.id);
                emitted.extend(adjoints);
            }
        }
    }

    DispatchResult {
        emitted_op_count: emitted.len(),
        diagnostics,
        emitted_adjoints: emitted,
    }
}

#[test]
fn ad_dispatcher_falls_back_when_backward_validator_rejects() {
    let cfg = backward_over_budget_config();
    // Sanity: confirm the config actually trips the backward validator
    // (if this fails, the T2.1 validator regressed and T7.1's harness
    // is silently testing the wrong path).
    use nsl_codegen::flash_attention_v2::smem_layout::{
        validate_scalar_v2_config, Direction,
    };
    assert!(
        validate_scalar_v2_config(&cfg, Direction::Forward).is_ok(),
        "harness config must pass forward validation"
    );
    assert!(
        validate_scalar_v2_config(&cfg, Direction::Backward).is_err(),
        "harness config must FAIL backward validation — \
         otherwise T7.1 is testing the wrong path"
    );

    let mark = mark_for_layer(cfg);
    let ops = claimed_chain_ops();
    let result = simulate_reverse_walk(&ops, &mark);

    // Per-op fallback: sum of adjoints across every op in the chain,
    // not a single fused-kernel adjoint.
    assert!(
        result.emitted_op_count > 1,
        "expected per-op fallback, got fused-shape (op_count={}): {:?}",
        result.emitted_op_count, result.emitted_adjoints
    );

    // Diagnostic surface — plan contract.
    assert!(
        !result.diagnostics.is_empty(),
        "fallback must surface a diagnostic"
    );
    let joined = result.diagnostics.join(" || ");
    assert!(
        joined.contains("CSHA fused backward rejected"),
        "diagnostic missing rejection phrase: {joined}"
    );
    assert!(
        joined.contains("bytes >"),
        "diagnostic missing byte-comparison: {joined}"
    );
    assert!(
        joined.contains("falling back"),
        "diagnostic missing fallback phrase: {joined}"
    );
    assert!(
        joined.contains("Backward"),
        "diagnostic must name direction: {joined}"
    );
    assert!(
        joined.contains("blocks.0"),
        "diagnostic must name the layer: {joined}"
    );

    // The mark's backward_emitted flag MUST stay false after a fallback
    // so the fallback path fires on every constituent op in the chain
    // (T5.2 contract).
    assert!(
        !mark.backward_emitted.get(),
        "validator-reject must leave backward_emitted=false so fallback \
         fires for all constituent ops"
    );
}

#[test]
fn ad_dispatcher_stays_single_emission_on_fused_accept() {
    // Counter-test: a config that DOES validate under Direction::Backward
    // should route to the fused path (single emission) — not degrade to
    // per-op fallback. Without this the T7.1 regression test could pass
    // trivially if the dispatcher always fell back.
    let ok_cfg = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
    };
    let mark = mark_for_layer(ok_cfg);
    let ops = claimed_chain_ops();
    let result = simulate_reverse_walk(&ops, &mark);

    // Exactly ONE synthetic fused-kernel adjoint, then AlreadyEmitted
    // short-circuits for the remaining claimed ops.
    assert_eq!(
        result.emitted_op_count, 1,
        "accept path should emit exactly one fused adjoint, got {}: {:?}",
        result.emitted_op_count, result.emitted_adjoints
    );
    assert!(
        result.diagnostics.is_empty(),
        "accept path should surface no fallback diagnostic, got {:?}",
        result.diagnostics
    );
    assert!(
        mark.backward_emitted.get(),
        "accept path must flip backward_emitted=true"
    );
}
