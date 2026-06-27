//! Phase 1 Item 2: `@cpdt(weight_aware=false)` runtime opt-out.
//!
//! Tests the compiler-side wiring:
//!   1. `Compiler::cpdt_weight_aware` defaults to `true`.
//!   2. Cascade contract: when weights are present (via CpdtInput) and
//!      `CpdtMode::Full`, `cpdt::run` produces a non-empty precision plan.
//!      When `CpdtInput.weights = None` (the shadow the opt-out installs),
//!      the precision plan is empty.
//!
//! The second test verifies the CASCADE the opt-out relies on — the
//! `invoke_cpdt_if_enabled` shadow itself (one-line `if/else`) is
//! trivially correct; the question is whether the downstream path
//! actually produces empty vs populated plans under `None` vs `Some`.
//! If the cascade were to change (e.g. cpdt::run decides to run plan_map
//! even with weights=None), Item 2's opt-out would silently fail; this
//! test guards against that.
//!
//! Design: docs/superpowers/specs/2026-04-20-cpdt-weight-aware-opt-out-design.md.

#![cfg(feature = "test-helpers")]

use std::collections::HashMap;
use std::path::PathBuf;

use nsl_codegen::cpdt::{run as cpdt_run, CpdtInput, CpdtMode};
use nsl_codegen::cpdt_expert::ExpertConfig;
use nsl_codegen::cpdt_joint::JointConfig;
use nsl_codegen::cpdt_optim::AdamWHyperparams;
use nsl_codegen::cpdt_tier_apply::PrecisionConfig;
use nsl_codegen::cpdt_zero::{ClusterSpec, ModelSize};
use nsl_codegen::weight_aware::WeightMap;
use nsl_codegen::wggo_apply::{AppliedLayer, AppliedPlan};
use nsl_codegen::wggo_dp::LayerDecision as CoarseDecision;
use nsl_codegen::wggo_overrides::WggoOverrides;
use nsl_codegen::CompileOptions;
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cpdt_calibration")
        .join(format!("{name}.safetensors"))
}

fn applied_plan_blocks(n_layers: usize) -> AppliedPlan {
    let layers = (0..n_layers)
        .map(|i| AppliedLayer {
            layer_index: i as u32,
            layer_name: format!("blocks.{i}"),
            coarse: CoarseDecision::KeepFull,
            pipeline_stage: 0,
            shard_factor: 1,
            shard_grads: 1,
            shard_optim: 1,
            active_heads: 8,
            ffn_width: 4096,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 8,
            optim_v_bits: 8,
            fase_fused: false,
            packing_mode: 0,
            estimated_us: 10.0,
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

fn cpdt_input<'a>(
    applied: &AppliedPlan,
    weights: Option<&'a WeightMap>,
) -> CpdtInput<'a> {
    let overrides = WggoOverrides::from_applied(applied);
    CpdtInput {
        mode: CpdtMode::Full,
        model: ModelSize::from_applied_plan(applied),
        cluster: ClusterSpec {
            num_gpus: 4,
            ..ClusterSpec::default()
        },
        weights,
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
fn compiler_default_cpdt_weight_aware_is_true() {
    let interner = Interner::new();
    let type_map: TypeMap = HashMap::new();
    let opts = CompileOptions::default();
    let compiler = nsl_codegen::compiler::Compiler::new(&interner, &type_map, &opts)
        .expect("Compiler::new should succeed with default options");
    assert_eq!(
        compiler.cpdt_weight_aware, true,
        "Compiler's cpdt_weight_aware must default to true; Phase 1 opt-out flips it \
         to false only when `@cpdt(weight_aware=false)` is seen by the decorator walker"
    );
}

#[test]
fn cascade_weights_some_populates_precision_plan() {
    // When CpdtInput.weights is Some(wm), cpdt::run's Full-mode branch
    // produces a non-empty precision plan. This is the behavior the
    // opt-out's "don't shadow to None" branch relies on.
    let wm = WeightMap::load(&fixture_path("calib_small"))
        .expect("calib_small fixture must be present");
    let applied = applied_plan_blocks(8);
    let input = cpdt_input(&applied, Some(&wm));
    let plan = cpdt_run(input);

    // calib_small has 74 tensors; weight-aware CPDT should produce a tier
    // assignment for approximately all of them. Lower bound (70) accommodates
    // minor fixture perturbations without the test becoming fragile.
    assert!(
        plan.precision.params.len() >= 70,
        "weight-aware CPDT should produce tier assignments for ~all tensors; \
         got {} params (expected >= 70 for calib_small's 74-tensor corpus)",
        plan.precision.params.len()
    );
}

#[test]
fn cascade_weights_none_produces_empty_precision_plan() {
    // When CpdtInput.weights is None (what the opt-out shadow installs),
    // cpdt::run returns PrecisionPlan::default() — zero params. This is
    // the contract the opt-out relies on: setting weight_aware=false
    // produces a build where the precision plan is empty.
    let applied = applied_plan_blocks(8);
    let input = cpdt_input(&applied, None);
    let plan = cpdt_run(input);

    assert_eq!(
        plan.precision.params.len(),
        0,
        "weight_aware=false should produce zero weight-derived tier assignments; \
         got {} (weight-map wiring not properly skipped? See cpdt::run's Full-mode branch.)",
        plan.precision.params.len()
    );
}
