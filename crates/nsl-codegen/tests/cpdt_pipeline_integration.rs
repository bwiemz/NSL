//! Task 4: CPDT pipeline integration tests.
//!
//! Exercises the WGGO → CPDT propagation contract that
//! `invoke_cpdt_if_enabled` (stmt.rs) relies on:
//!
//!   1. An `AppliedPlan` with per-layer `shard_factor=K` feeds
//!      `WggoOverrides::min_shard_factor` → `Some(K)` → becomes
//!      `CpdtInput.wggo_recommended_shard`, which, when `num_gpus % K == 0`,
//!      drives the planner to pick a `ZeroConfig` with `s_p <= K`.
//!
//!   2. When `num_gpus` does not divide the recommendation, the planner
//!      emits a `ShardFactorIncompatibleWithWorldSize` diagnostic on
//!      `CpdtPlan.override_diagnostics`.
//!
//! These tests exercise the same public seam (`WggoOverrides::from_applied`,
//! `ModelSize::from_applied_plan`, `cpdt::run`) that `invoke_cpdt_if_enabled`
//! composes internally, so they verify the wiring contract without requiring
//! a full NSL compile path (`invoke_cpdt_if_enabled` is `pub(crate)` and the
//! `compiler.cpdt_plan` field is not exposed by any public compile helper).
//!
//! See `project_cpdt_pipeline_integration.md` for the scope note: driving
//! `cpdt_plan` out of a real compile requires either exporting a test helper
//! that returns the `Compiler` post-finalize or surfacing `cpdt_plan` on the
//! existing `compile_module_with_imports_best_effort_plan` tuple. Both are
//! larger API changes tracked separately.

use nsl_codegen::cpdt::{run as cpdt_run, CpdtInput, CpdtMode};
use nsl_codegen::cpdt_expert::ExpertConfig;
use nsl_codegen::cpdt_joint::JointConfig;
use nsl_codegen::cpdt_optim::AdamWHyperparams;
use nsl_codegen::cpdt_tier_apply::PrecisionConfig;
use nsl_codegen::cpdt_zero::{ClusterSpec, ModelSize};
use nsl_codegen::wggo_apply::{AppliedLayer, AppliedPlan};
use nsl_codegen::wggo_dp::LayerDecision as CoarseDecision;
use nsl_codegen::wggo_overrides::{OverrideRejectReason, WggoOverrides};

/// Build an `AppliedPlan` with `n_layers` layers, each carrying
/// `shard_factor=shard`. Uses realistic byte sizes so the ZeRO planner
/// doesn't trivially pick DDP.
fn applied_plan_with_shard(n_layers: usize, shard: u32) -> AppliedPlan {
    let layers = (0..n_layers)
        .map(|i| AppliedLayer {
            layer_index: i as u32,
            layer_name: format!("blocks.{i}"),
            coarse: CoarseDecision::KeepFull,
            pipeline_stage: 0,
            shard_factor: shard,
            active_heads: 8,
            ffn_width: 4096,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 8,
            optim_v_bits: 8,
            fase_fused: false,
            packing_mode: 0,
            estimated_us: 10.0,
            // Large-ish per-layer bytes so ZeRO sharding is meaningful,
            // but below the 80 GB/GPU default budget.
            param_bytes: 400_000_000,
            activation_bytes: 100_000_000,
        })
        .collect();
    AppliedPlan {
        layers,
        total_us: 0.0,
        peak_memory_bytes: 0,
    }
}

fn cpdt_input_from_plan<'a>(
    applied: &AppliedPlan,
    num_gpus: u32,
) -> CpdtInput<'a> {
    let overrides = WggoOverrides::from_applied(applied);
    CpdtInput {
        mode: CpdtMode::Full,
        model: ModelSize::from_applied_plan(applied),
        cluster: ClusterSpec {
            num_gpus,
            ..ClusterSpec::default()
        },
        weights: None,
        precision_cfg: PrecisionConfig::default(),
        adamw: AdamWHyperparams::default(),
        moe_shape: None,
        moe_router: None,
        moe_roofline_slack: 0.0,
        expert_cfg: ExpertConfig::default(),
        joint_cfg: JointConfig::default(),
        wggo_recommended_shard: overrides.min_shard_factor(),
    }
}

#[test]
fn wggo_shard_recommendation_flows_to_cpdt_plan() {
    // 8 layers, shard_factor=4 on each → min_shard_factor() == Some(4).
    let applied = applied_plan_with_shard(8, 4);
    let overrides = WggoOverrides::from_applied(&applied);
    assert_eq!(
        overrides.min_shard_factor(),
        Some(4),
        "fixture precondition: WGGO overrides must aggregate to Some(4)"
    );

    // num_gpus=8 divides 4 cleanly → Gate 1 passes.
    let input = cpdt_input_from_plan(&applied, 8);
    let plan = cpdt_run(input);

    let zero = plan
        .zero
        .as_ref()
        .expect("ZeRO planner must produce a tuple under CpdtMode::Full");
    assert!(
        zero.config.s_p <= 4,
        "WGGO recommendation must bound s_p: got s_p={}, expected ≤ 4",
        zero.config.s_p
    );
    // No divisibility diagnostic should fire on the clean-divide happy path.
    let has_incompat = plan.override_diagnostics.iter().any(|d| {
        matches!(
            d.reason,
            OverrideRejectReason::ShardFactorIncompatibleWithWorldSize { .. }
        )
    });
    assert!(
        !has_incompat,
        "world_size=8 divides recommended=4; must not emit ShardFactorIncompatibleWithWorldSize; got {:?}",
        plan.override_diagnostics
    );
}

#[test]
fn world_size_mismatch_emits_incompatible_diagnostic() {
    // Recommend shard_factor=4 on a cluster with num_gpus=5.
    // 5 % 4 != 0 → Gate 1 rejects → ShardFactorIncompatibleWithWorldSize.
    let applied = applied_plan_with_shard(8, 4);
    let input = cpdt_input_from_plan(&applied, 5);
    let plan = cpdt_run(input);

    let diag = plan
        .override_diagnostics
        .iter()
        .find(|d| {
            matches!(
                d.reason,
                OverrideRejectReason::ShardFactorIncompatibleWithWorldSize { .. }
            )
        })
        .unwrap_or_else(|| {
            panic!(
                "expected ShardFactorIncompatibleWithWorldSize; got {:?}",
                plan.override_diagnostics
            )
        });

    if let OverrideRejectReason::ShardFactorIncompatibleWithWorldSize {
        recommended,
        world_size,
    } = diag.reason
    {
        assert_eq!(recommended, 4);
        assert_eq!(world_size, 5);
    } else {
        unreachable!();
    }
}
