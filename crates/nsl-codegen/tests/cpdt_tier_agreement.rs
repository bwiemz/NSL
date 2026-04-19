//! Phase 1 Commit 5 — `plan_map_noweights` and `compute_tier_agreement`
//! helpers power the passive `[cpdt] weight-aware tier agreement: X%`
//! diagnostic emitted from `invoke_cpdt_if_enabled`.

use std::path::PathBuf;

use nsl_codegen::cpdt_tier_apply::{
    compute_tier_agreement, plan_map, plan_map_noweights, PrecisionConfig, Tier,
};
use nsl_codegen::weight_aware::WeightMap;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cpdt_calibration")
        .join(format!("{name}.safetensors"))
}

#[test]
fn plan_map_noweights_produces_same_layer_set_as_plan_map() {
    let wm = WeightMap::load(&fixture("calib_tiny")).unwrap();
    let cfg = PrecisionConfig {
        n_layers: 2,
        ..Default::default()
    };
    let plan = plan_map(&wm, &cfg);
    let plan_nw = plan_map_noweights(&wm, &cfg);
    assert_eq!(
        plan.params.len(),
        plan_nw.params.len(),
        "both plans should produce a record for every tensor"
    );
    let names: Vec<&str> = plan.params.iter().map(|p| p.name.as_str()).collect();
    let names_nw: Vec<&str> = plan_nw.params.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names, names_nw,
        "plan_map_noweights must preserve tensor order for direct comparison"
    );
}

#[test]
fn tier_agreement_full_on_calib_tiny() {
    // On calib_tiny every tensor is kind-overridden (L=2 → every layer is
    // first-or-last; norms + embeddings also override), so weights-present
    // and weights-absent agree on every tensor.
    let wm = WeightMap::load(&fixture("calib_tiny")).unwrap();
    let cfg = PrecisionConfig {
        n_layers: 2,
        ..Default::default()
    };
    let plan = plan_map(&wm, &cfg);
    let plan_nw = plan_map_noweights(&wm, &cfg);
    let (agree_l, total_l, agree_p, total_p) = compute_tier_agreement(&plan, &plan_nw);
    assert_eq!(agree_l, total_l, "every layer should agree on calib_tiny");
    assert_eq!(agree_p, total_p, "every param should agree on calib_tiny");
    assert!(total_l > 0, "calib_tiny must produce at least one layer");
}

#[test]
fn tier_agreement_full_on_calib_small_by_construction() {
    // On calib_small, overridden tensors land High, generic tensors score
    // < CALIB_T2 under both paths (CALIB_K/numel also under T2), so both
    // paths agree on every tensor by construction of the placeholder
    // constants. If calibration tuning changes this, the assertion here
    // becomes a canary for the shift.
    let wm = WeightMap::load(&fixture("calib_small")).unwrap();
    let cfg = PrecisionConfig {
        n_layers: 8,
        ..Default::default()
    };
    let plan = plan_map(&wm, &cfg);
    let plan_nw = plan_map_noweights(&wm, &cfg);
    let (agree_l, total_l, _, _) = compute_tier_agreement(&plan, &plan_nw);
    assert_eq!(
        agree_l, total_l,
        "calibration currently produces identical tier sets; \
         recalibration will break this — update test when threshold tuning lands"
    );
}

#[test]
fn tier_agreement_ignores_unmatched_layers() {
    // Construct two PrecisionPlans with one shared and one unique tensor each
    // to confirm compute_tier_agreement only scores the intersection.
    let wm = WeightMap::load(&fixture("calib_tiny")).unwrap();
    let cfg = PrecisionConfig {
        n_layers: 2,
        ..Default::default()
    };
    let plan_a = plan_map(&wm, &cfg);
    let mut plan_b = plan_map(&wm, &cfg);
    // Rename one param in plan_b so it no longer matches plan_a's set.
    if let Some(p) = plan_b.params.iter_mut().next() {
        p.name = "alien.weight".to_string();
    }
    let (agree_l, total_l, _, _) = compute_tier_agreement(&plan_a, &plan_b);
    assert_eq!(
        total_l,
        plan_a.params.len() as u64 - 1,
        "only the intersection of names should be counted"
    );
    assert_eq!(agree_l, total_l, "remaining layers should still agree");
}

#[test]
fn plan_map_noweights_applies_embedding_stochastic_rounding_flag() {
    let wm = WeightMap::load(&fixture("calib_tiny")).unwrap();
    let cfg = PrecisionConfig {
        n_layers: 2,
        embedding_stochastic_rounding: true,
    };
    let plan_nw = plan_map_noweights(&wm, &cfg);
    let embed = plan_nw
        .params
        .iter()
        .find(|p| p.name == "tok_embeddings.weight")
        .expect("tok_embeddings.weight must be present in calib_tiny");
    assert_eq!(embed.tier, Tier::High);
    assert!(
        embed.stochastic_rounding,
        "embeddings should inherit stochastic rounding when the config enables it, \
         even on the no-weights path"
    );
}
