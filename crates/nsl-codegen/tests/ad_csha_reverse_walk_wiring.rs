//! T7.1 — CSHA fused backward reverse-walk wiring integration test.
//!
//! Verifies that `AdjointGenerator::generate` correctly dispatches
//! claimed Wengert ops through the CSHA backward dispatcher when
//! `CshaBackwardClaims` are set, and falls through to per-op AD
//! for unclaimed ops and fallback chains.

use nsl_codegen::csha_apply::{FusionMark, MarkRole};
use nsl_codegen::csha_boundary::ProjKind;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::source_ad::{AdjointGenerator, CshaBackwardClaims};
use nsl_codegen::wengert::{PrimalOp, VarId, WengertList, WengertOp};
use std::collections::HashMap;

fn op(id: u32, result: VarId, prim: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
    WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    }
}

/// Build a minimal Wengert list modelling:
///   op0: Input("x")        → var 0
///   op1: RMSNorm(0)        → var 1   [claimed]
///   op2: Param("wq")       → var 2
///   op3: Matmul(1, 2)      → var 3   [claimed]
///   op4: RoPE(3)           → var 4   [claimed]
///   op5: Add(4, 0)         → var 5   [NOT claimed — unclaimed op]
fn mixed_wengert() -> WengertList {
    let ops = vec![
        op(0, 0, PrimalOp::Input("x".into()), vec![]),
        op(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
        op(2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
        op(3, 3, PrimalOp::Matmul, vec![1, 2]),
        op(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
        op(5, 5, PrimalOp::Add, vec![4, 0]),
    ];
    WengertList {
        ops,
        output: 5,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    }
}

fn ok_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
    }
}

fn over_budget_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 64,
            ..CshaExtras::default()
        }),
    }
}

fn build_claims(config: FlashAttentionConfig) -> CshaBackwardClaims {
    // Chain: ops 1 (norm), 3 (matmul), 4 (rope) all map to chain 0.
    let mut op_to_chain = HashMap::new();
    op_to_chain.insert(1, 0usize); // RMSNorm
    op_to_chain.insert(3, 0); // Matmul
    op_to_chain.insert(4, 0); // RoPE

    let chain_marks = vec![FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "blocks.0.attn.wq".into(),
        role: MarkRole::NormPrologue,
        config: Some(config),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: None,
    }];

    CshaBackwardClaims {
        op_to_chain,
        chain_marks,
    }
}

/// When the backward validator rejects the config, the generator must
/// fall through to per-op AD for all claimed ops (producing adjoints
/// for RMSNorm, Matmul, RoPE) AND unclaimed ops (Add).
#[test]
fn fallback_chain_emits_per_op_adjoints_for_all_ops() {
    let primal = mixed_wengert();
    let claims = build_claims(over_budget_config());

    let mut gen = AdjointGenerator::new(10);
    gen.set_csha_claims(claims);
    let adjoint = gen.generate(&primal);

    // The fallback path should produce adjoints for ALL differentiable
    // ops: Add (2 adjoints), RoPE (1), Matmul (2), RMSNorm (1+).
    // Exact count depends on rule details, but must be > 1 (i.e. NOT
    // a single fused emission).
    assert!(
        adjoint.ops.len() > 2,
        "fallback path must produce per-op adjoints, got {} ops",
        adjoint.ops.len()
    );

    // Diagnostics should contain the rejection reason.
    let diags = gen.csha_diagnostics();
    assert!(!diags.is_empty(), "fallback must record a diagnostic");
    let joined = diags.join(" || ");
    assert!(
        joined.contains("CSHA fused backward rejected"),
        "diagnostic missing rejection phrase: {joined}"
    );
}

/// When the backward validator accepts the config, the generator
/// currently falls back to per-op AD (TODO stub) but the dispatch
/// logic fires correctly — no diagnostics, all ops produce adjoints.
///
/// Once T7.2 implements the real fused emission, this test should be
/// updated to verify that claimed ops are SKIPPED (only the unclaimed
/// Add produces per-op adjoints).
#[test]
fn accepted_chain_falls_back_with_todo_stub() {
    let primal = mixed_wengert();
    let claims = build_claims(ok_config());

    let mut gen = AdjointGenerator::new(10);
    gen.set_csha_claims(claims);
    let adjoint = gen.generate(&primal);

    // With the TODO stub, EmitFused resets backward_emitted and falls
    // through to per-op AD. So ALL ops still produce adjoints.
    assert!(
        adjoint.ops.len() > 2,
        "TODO-stub path must still produce per-op adjoints, got {} ops",
        adjoint.ops.len()
    );

    // No fallback diagnostics — the dispatcher accepted the chain.
    let diags = gen.csha_diagnostics();
    assert!(
        diags.is_empty(),
        "accepted chain should produce no fallback diagnostic, got {:?}",
        diags
    );
}

/// Without CSHA claims, `generate` behaves identically to before —
/// all ops go through per-op AD.
#[test]
fn no_claims_produces_normal_adjoints() {
    let primal = mixed_wengert();

    let mut gen = AdjointGenerator::new(10);
    // No set_csha_claims call.
    let adjoint = gen.generate(&primal);

    assert!(
        adjoint.ops.len() > 2,
        "unclaimed path must produce per-op adjoints, got {} ops",
        adjoint.ops.len()
    );
    assert!(gen.csha_diagnostics().is_empty());
    // No claims → no fused-backward events either.
    assert!(gen.csha_fused_events().is_empty());
}

/// T7.2: smoke config (hd=32, seq=32, block_q=block_kv=32, d_model=32)
/// records a `CshaFusedBackwardEvent{smoke_config=true}` so downstream
/// tools know which chains WOULD go fused once the kernel-launch codegen
/// lands.
#[test]
fn smoke_config_records_fused_event() {
    let primal = mixed_wengert();
    let claims = build_claims(ok_config()); // smoke config

    let mut gen = AdjointGenerator::new(10);
    gen.set_csha_claims(claims);
    let _adjoint = gen.generate(&primal);

    let events = gen.csha_fused_events();
    assert_eq!(
        events.len(),
        1,
        "smoke config should record one fused-backward event, got {:?}",
        events
    );
    let ev = &events[0];
    assert_eq!(ev.layer, "blocks.0");
    assert_eq!(ev.head_dim, 32);
    assert_eq!(ev.block_q, 32);
    assert_eq!(ev.block_kv, 32);
    assert!(
        ev.smoke_config,
        "hd=32 smoke config must mark smoke_config=true"
    );
}

/// T7.2: scope gate — configs bigger than the smoke shape still get an
/// event recorded (so reports can show the fused path WOULD have fired)
/// but with `smoke_config=false`. The kernel-launch codegen is pinned to
/// smoke config only in this build; everything else falls back to per-op AD.
#[test]
fn non_smoke_config_records_event_with_scope_gate() {
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    // hd=64 passes the backward validator but is outside the smoke scope.
    let hd64_ok_cfg = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 64,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 64,
            ..CshaExtras::default()
        }),
    };

    // Ensure the config passes the backward validator — otherwise this
    // test is silently exercising the fallback path.
    use nsl_codegen::flash_attention_v2::smem_layout::{validate_scalar_v2_config, Direction};
    assert!(
        validate_scalar_v2_config(&hd64_ok_cfg, Direction::Backward).is_ok(),
        "harness config must pass backward validation"
    );

    let primal = mixed_wengert();
    let claims = build_claims(hd64_ok_cfg);

    let mut gen = AdjointGenerator::new(10);
    gen.set_csha_claims(claims);
    let _adjoint = gen.generate(&primal);

    let events = gen.csha_fused_events();
    assert_eq!(events.len(), 1, "hd=64 chain should still record an event");
    let ev = &events[0];
    assert_eq!(ev.head_dim, 64);
    assert!(
        !ev.smoke_config,
        "hd=64 config must be flagged smoke_config=false (scope-gated)"
    );

    // Still no fallback diagnostics — the dispatcher accepted the chain;
    // the fallback is a scope-gate decision, not a validator rejection.
    assert!(
        gen.csha_diagnostics().is_empty(),
        "scope-gated fallback should not emit a rejection diagnostic"
    );
}

/// T7.2: validator-rejected configs should NOT record a fused event —
/// the dispatcher never reached EmitFused.
#[test]
fn fallback_chain_records_no_fused_event() {
    let primal = mixed_wengert();
    let claims = build_claims(over_budget_config());

    let mut gen = AdjointGenerator::new(10);
    gen.set_csha_claims(claims);
    let _adjoint = gen.generate(&primal);

    assert!(
        gen.csha_fused_events().is_empty(),
        "validator-rejected chain must not record a fused event, got {:?}",
        gen.csha_fused_events()
    );
    // The fallback diagnostic is the authoritative surface for this
    // case — events stay empty.
    assert!(!gen.csha_diagnostics().is_empty());
}

/// Gap I.1 regression: when the plan-level config carries fusion flags
/// (`fused_projections=true`) that break the backward SMEM validator OR
/// the `block_q == block_kv` fused-projections invariant, but a clamped
/// training config is supplied, the dispatcher must see the **training**
/// config and return `EmitFused` — not fall back via the plan config.
///
/// This pins the I.1 fix in `collect_chain_dispatch_map_with_wengert`:
/// `FusionMark.config` is sourced from `training_config` when present.
#[test]
fn gap_i1_training_config_clamps_plan_fusion_flags() {
    use nsl_codegen::ad_rules::{csha_dispatch_for_op, CshaDispatchDecision};
    use nsl_codegen::flash_attention_v2::smem_layout::{validate_scalar_v2_config, Direction};

    // Plan-level config: fused_projections + fused_output_proj at
    // head_dim=128 + d_model=128 blows the SMEM budget (116 KB+ exceeds
    // the 99 KB dynamic-SMEM cap on all supported archs) → validator
    // rejects.  Previously this test used `block_q != block_kv` to
    // force rejection; that asymmetry is now valid (see `flash_attention_v2/mod.rs`
    // Step 3 kv_iters decoupling), so we use the SMEM-budget path
    // instead — same test intent (plan-config validator-rejecting),
    // different trigger.
    let plan_cfg = FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 128,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            fused_projections: true,
            fused_output_proj: true,
            save_activations_for_backward: true,
            d_model: 128,
            ..CshaExtras::default()
        }),
    };
    // Plan config MUST fail the validator — otherwise this test isn't
    // actually exercising the clamp path.
    assert!(
        validate_scalar_v2_config(&plan_cfg, Direction::Backward).is_err(),
        "plan config must be validator-rejecting for the test to be meaningful"
    );

    // Training config: clamped — no fusion flags, block_q==block_kv==32.
    // This is what `compiler/kernel.rs:543-552` stamps onto
    // `csha_training_config`, which is the config the real backward
    // kernel is actually built from.
    let train_cfg = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: false,
            fused_output_proj: false,
            save_activations_for_backward: true,
            d_model: 32,
            rmsnorm_eps: 1e-5,
            ..CshaExtras::default()
        }),
    };
    // Training config MUST pass the validator — otherwise even the
    // clamp can't make the dispatcher hit `EmitFused`.
    assert!(
        validate_scalar_v2_config(&train_cfg, Direction::Backward).is_ok(),
        "training config must pass the backward validator"
    );

    // Simulate what `collect_chain_dispatch_map_with_wengert` does when
    // `training_config` is supplied: the mark carries the training
    // config, not the plan config.
    let clamped_mark = FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "blocks.0.attn.wq".into(),
        role: MarkRole::NormPrologue,
        config: Some(train_cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: None,
    };
    let plan_mark = FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "blocks.0.attn.wq".into(),
        role: MarkRole::NormPrologue,
        config: Some(plan_cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: None,
    };

    // With the plan config, the dispatcher MUST fall back.
    match csha_dispatch_for_op(&plan_mark, 0) {
        CshaDispatchDecision::Fallback { diagnostic } => {
            assert!(
                diagnostic.contains("CSHA fused backward rejected"),
                "expected plan-config rejection, got: {diagnostic}"
            );
        }
        other => panic!("plan config MUST be rejected by dispatcher; got {other:?}"),
    }

    // With the clamped training config, the dispatcher MUST emit fused.
    match csha_dispatch_for_op(&clamped_mark, 0) {
        CshaDispatchDecision::EmitFused => {}
        other => {
            panic!("I.1 regression — clamped training config MUST accept EmitFused; got {other:?}")
        }
    }
}

/// Verify that the `collect_chain_dispatch_map` helper correctly
/// maps boundary-chain op indices to chain-level marks.
#[test]
fn collect_chain_dispatch_map_maps_ops_to_chain_marks() {
    use nsl_codegen::csha::CshaMode;
    use nsl_codegen::csha_apply::{bridge, collect_chain_dispatch_map};
    use nsl_codegen::csha_specialize::SpecConfig;
    use nsl_codegen::wggo_cost::LayerShape;

    // Build a toy plan with one CSHA-active layer.
    let w = {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            op(2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            op(3, 3, PrimalOp::Matmul, vec![1, 2]),
            op(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            op(5, 5, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
            op(6, 6, PrimalOp::Matmul, vec![1, 5]),
            op(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            op(8, 8, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
            op(9, 9, PrimalOp::Matmul, vec![1, 8]),
        ];
        WengertList {
            ops,
            output: 9,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    };

    let plan = nsl_codegen::csha::run(nsl_codegen::csha::CshaInput {
        mode: CshaMode::Auto,
        target: "H100",
        wengert: &w,
        weights: None,
        shape: LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        },
        n_heads: 8,
        spec_cfg: SpecConfig::default(),
        pattern_cfg: nsl_codegen::csha_patterns::PatternConfig::default(),
        wggo_overrides: None,
    });
    let br = bridge(&plan, 64, &mut Vec::new());
    let (op_to_chain, chain_marks) = collect_chain_dispatch_map(&plan, &br);

    // Three chains (Q, K, V) → three chain marks.
    assert_eq!(
        chain_marks.len(),
        3,
        "expected 3 chains, got {}",
        chain_marks.len()
    );

    // All claimed op indices must be mapped.
    assert!(op_to_chain.contains_key(&1), "RMSNorm op 1 must be mapped");
    assert!(op_to_chain.contains_key(&3), "Q matmul op 3 must be mapped");
    assert!(op_to_chain.contains_key(&4), "Q RoPE op 4 must be mapped");
    assert!(op_to_chain.contains_key(&6), "K matmul op 6 must be mapped");
    assert!(op_to_chain.contains_key(&7), "K RoPE op 7 must be mapped");
    assert!(op_to_chain.contains_key(&9), "V matmul op 9 must be mapped");

    // Ops within the same chain must map to the same chain index.
    // Q chain: ops 1, 3, 4.
    let q_chain = op_to_chain[&3];
    assert_eq!(
        op_to_chain[&4], q_chain,
        "Q RoPE must share chain with Q matmul"
    );

    // Each chain mark should carry a config (from the bridge).
    for (i, mark) in chain_marks.iter().enumerate() {
        assert!(
            mark.config.is_some(),
            "chain {i} mark must carry a FlashAttentionConfig"
        );
        assert!(
            !mark.backward_emitted.get(),
            "chain {i} backward_emitted must start false"
        );
    }
}
