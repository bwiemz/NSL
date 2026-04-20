//! Phase 1 Commit 5 — `plan_map_noweights` and `compute_tier_agreement`
//! helpers power the passive `[cpdt] weight-aware tier agreement: X%`
//! diagnostic emitted from `invoke_cpdt_if_enabled`.

use std::path::PathBuf;

use nsl_codegen::cpdt_tier_apply::{
    compute_tier_agreement, plan_map, plan_map_noweights, ParamPrecision, PrecisionConfig,
    PrecisionPlan, Tier,
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
    // Phase 1 retune (2026-04-19): the previous assertion
    // (`agree_l == total_l`) held under the degenerate binary distribution.
    // Under the retuned constants, calib_small is effectively 2-tier (H + M)
    // because the pooled-across-fixtures thresholds are dominated by
    // calib_medium's larger tensors; calib_small's ffn_gate_up scores land
    // above T0 and all generic attn+ffn_gate_up go High, leaving ffn_down as
    // the sole Medium population. The test now asserts corpus-wide primary-
    // tier non-degeneracy (§6 rule) plus per-fixture sanity; see spec §4.2.
    let wm_small = WeightMap::load(&fixture("calib_small")).unwrap();
    let plan_small = plan_map(
        &wm_small,
        &PrecisionConfig { n_layers: 8, ..Default::default() },
    );
    let counts_small = tier_counts_of(&plan_small);

    // Per-fixture sanity on calib_small: at least one primary tier populated.
    // Catches the "all generics landed VeryLow" degeneracy that Phase 1 shipped.
    let small_primary_count =
        counts_small.high + counts_small.medium + counts_small.low;
    assert!(
        small_primary_count > 0,
        "calib_small has no primary-tier assignments (degenerate distribution)"
    );

    // Expected under pooled thresholds at K≈0.060 (per spec §3.3):
    //   High = 68, Medium = 6, Low = 0, VeryLow = 0.
    // Floors use slack so minor fixture perturbations don't break the test.
    assert!(
        counts_small.high >= 60,
        "H underpopulated on calib_small: {}",
        counts_small.high
    );
    assert!(
        counts_small.medium >= 4,
        "M underpopulated on calib_small: {}",
        counts_small.medium
    );
    assert_eq!(
        counts_small.very_low, 0,
        "VeryLow unexpectedly populated on calib_small: {}",
        counts_small.very_low
    );

    // Corpus-wide primary-tier non-degeneracy (§6 rule). calib_small alone
    // doesn't populate Low; calib_medium does. calib_medium is regen-at-test-
    // time — skip the corpus union check if the regenerated fixture is absent.
    // CI runs that regenerate calib_medium first catch the full rule; local
    // developer runs without regen see only the per-fixture sanity check.
    if let Some(plan_medium) = try_load_plan_medium() {
        let counts_medium = tier_counts_of(&plan_medium);
        let union_populated = |s: usize, m: usize| -> bool { s > 0 || m > 0 };
        assert!(
            union_populated(counts_small.high, counts_medium.high),
            "primary tier High not populated across corpus"
        );
        assert!(
            union_populated(counts_small.medium, counts_medium.medium),
            "primary tier Medium not populated across corpus"
        );
        assert!(
            union_populated(counts_small.low, counts_medium.low),
            "primary tier Low not populated across corpus; \
             calib_small: {}, calib_medium: {}",
            counts_small.low,
            counts_medium.low
        );
    } else {
        eprintln!(
            "note: calib_medium not present at target/cpdt_calibration/; \
             corpus-wide primary-tier check skipped. To enable, run \
             cpdt_fixture_generate --include-medium \
             --output-dir target/cpdt_calibration/"
        );
    }
}

#[derive(Default)]
struct TierCounts {
    high: usize,
    medium: usize,
    low: usize,
    very_low: usize,
}

fn tier_counts_of(plan: &PrecisionPlan) -> TierCounts {
    let mut c = TierCounts::default();
    for p in &plan.params {
        match p.tier {
            Tier::High => c.high += 1,
            Tier::Medium => c.medium += 1,
            Tier::Low => c.low += 1,
            Tier::VeryLow => c.very_low += 1,
        }
    }
    c
}

/// Load calib_medium from the regen-at-test-time directory if present.
/// Returns `None` if the file is absent so the canary test can skip the
/// corpus-union check gracefully.
fn try_load_plan_medium() -> Option<PrecisionPlan> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/cpdt_calibration/calib_medium.safetensors");
    if !p.is_file() {
        return None;
    }
    let wm = WeightMap::load(&p).ok()?;
    let plan = plan_map(
        &wm,
        &PrecisionConfig { n_layers: 16, ..Default::default() },
    );
    Some(plan)
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
fn tier_agreement_skips_zero_byte_tensors() {
    // Construct two tiny plans where one shared tensor has param_bytes == 0;
    // the zero-byte tensor must not contribute to the totals regardless of
    // whether its tiers agree.
    use nsl_codegen::cpdt_tier_apply::OptimPrecision;

    let shared_nonzero = ParamPrecision {
        name: "real.weight".to_string(),
        layer: Some(0),
        tier: Tier::High,
        m_precision: OptimPrecision::Fp32,
        v_precision: OptimPrecision::Fp32,
        stochastic_rounding: false,
        sensitivity_score: 1.0,
        param_bytes: 1024,
        optim_bytes: 8192,
    };
    let zero_byte = ParamPrecision {
        name: "empty.bias".to_string(),
        layer: Some(0),
        tier: Tier::VeryLow,
        m_precision: OptimPrecision::Int8,
        v_precision: OptimPrecision::Int8,
        stochastic_rounding: false,
        sensitivity_score: 0.0,
        param_bytes: 0,
        optim_bytes: 0,
    };
    let plan = PrecisionPlan {
        params: vec![shared_nonzero.clone(), zero_byte.clone()],
        total_optim_bytes: 8192,
        baseline_fp32_bytes: 8192,
    };
    // plan_nw disagrees on the zero-byte tensor; still shouldn't affect totals.
    let mut zero_byte_disagree = zero_byte.clone();
    zero_byte_disagree.tier = Tier::High;
    let plan_nw = PrecisionPlan {
        params: vec![shared_nonzero, zero_byte_disagree],
        total_optim_bytes: 8192,
        baseline_fp32_bytes: 8192,
    };
    let (agree_l, total_l, agree_p, total_p) = compute_tier_agreement(&plan, &plan_nw);
    assert_eq!(total_l, 1, "zero-byte tensor must not be counted");
    assert_eq!(agree_l, 1, "the real tensor agrees");
    assert_eq!(total_p, 1024);
    assert_eq!(agree_p, 1024);
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

#[test]
fn disagreement_source_matches_numel_collision() {
    // Spec §5.4: for every tensor p whose weights-present tier differs from
    // its no-weights tier, there must exist a numel-matched sibling q whose
    // weights-present tier is exactly the tier the nw path put p into. That's
    // the precise collision signature (SwiGLU gate_up/down at d_model × d_ffn).
    // Any disagreement without such a sibling is an unexpected source and
    // should fail the test.
    let fixtures = [("calib_tiny", 2u32), ("calib_small", 8u32)];
    for (fix, n_layers) in fixtures {
        let wm = WeightMap::load(&fixture(fix)).unwrap();
        let cfg = PrecisionConfig {
            n_layers,
            ..Default::default()
        };
        let plan = plan_map(&wm, &cfg);
        let plan_nw = plan_map_noweights(&wm, &cfg);
        let by_name_nw: std::collections::HashMap<&str, &ParamPrecision> = plan_nw
            .params
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();
        for p in &plan.params {
            let pnw = by_name_nw
                .get(p.name.as_str())
                .unwrap_or_else(|| panic!("{fix}: missing nw plan entry for {}", p.name));
            if p.tier == pnw.tier {
                continue;
            }
            let p_numel = wm.get(&p.name).unwrap().num_elements;
            let has_collision_partner = plan.params.iter().any(|q| {
                q.name != p.name
                    && wm.get(&q.name).unwrap().num_elements == p_numel
                    && q.tier == pnw.tier
            });
            assert!(
                has_collision_partner,
                "{fix}: disagreement on {} (wp={:?}, nw={:?}) has no \
                 numel-matched sibling with wp-tier == {:?} — unknown source \
                 of disagreement, investigate",
                p.name, p.tier, pnw.tier, pnw.tier
            );
        }
    }
}
