//! CPDT §3.3 paper-alignment integration test.
//!
//! Paper §3.3: "Q-Adam-mini discovered that embedding layers need stochastic
//! rounding for INT8 stability. ... In CPDT, the compiler knows which layers
//! are embeddings (from the model AST) and automatically generates stochastic
//! rounding kernels for those layers and deterministic rounding for
//! everything else."
//!
//! This integration test exercises the full chain end-to-end:
//!   1. Build a WeightMap with a mix of layer kinds
//!      (embedding, norm, FFN middle).
//!   2. Run `plan_map` (with the default `embedding_stochastic_rounding=true`).
//!   3. Verify the embedding param carries `stochastic_rounding=true` while
//!      non-embedding params carry `stochastic_rounding=false`.
//!   4. Run `emit_plan` and verify the QuantizedOptimProgram for the embedding
//!      records `stochastic_rounding=true`; everything else records `false`.
//!
//! Plus the "actually fires under INT8" gate: when a param's m/v precision is
//! INT8 AND it's flagged stochastic, the emitted `QuantStoreM`/`QuantStoreV`
//! ops carry `stochastic: true` — confirming the QuantizedOptimOp encoding
//! matches the paper's "stochastic rounding kernels for those layers" claim.

use nsl_codegen::cpdt_optim::{emit_step, AdamWHyperparams, QuantizedOptimOp, QuantizedOptimProgram};
use nsl_codegen::cpdt_tier_apply::{
    classify_param, plan_map, ParamPrecision, PrecisionConfig, Tier,
};
use nsl_codegen::weight_aware::{WeightDType, WeightEntry, WeightMap};

fn entry(name: &str) -> WeightEntry {
    // f32 weights, fixed shape — small fixture so tier classification is by name
    // alone (not magnitude / sensitivity heuristics).
    WeightEntry::new(name.to_string(), vec![0u8; 16], vec![2, 2], WeightDType::F32)
}

#[test]
fn embedding_flagged_for_stochastic_non_embedding_not() {
    let cfg = PrecisionConfig::default();
    let emb = classify_param(&entry("tok_embeddings.weight"), &cfg);
    let norm = classify_param(&entry("blocks.0.norm.weight"), &cfg);
    let ffn = classify_param(&entry("blocks.3.ffn.w_gate"), &cfg);

    assert!(emb.stochastic_rounding, "embedding must be flagged stochastic");
    assert!(!norm.stochastic_rounding, "norm must NOT be flagged stochastic");
    assert!(!ffn.stochastic_rounding, "FFN middle layer must NOT be flagged stochastic");

    // Sanity: tier_apply still pins embedding to Tier::High (paper §3.1 table).
    assert_eq!(emb.tier, Tier::High);
}

#[test]
fn opt_out_via_config_disables_embedding_stochastic() {
    let mut cfg = PrecisionConfig::default();
    cfg.embedding_stochastic_rounding = false;
    let emb = classify_param(&entry("tok_embeddings.weight"), &cfg);
    assert!(!emb.stochastic_rounding, "opt-out must silence the flag");
}

#[test]
fn embedding_detected_by_alt_names() {
    let cfg = PrecisionConfig::default();
    for name in &["wte.weight", "blocks.0.embed_in.weight", "embeddings.weight"] {
        let p = classify_param(&entry(name), &cfg);
        assert!(p.stochastic_rounding, "{name} must be detected as embedding");
    }
}

#[test]
fn plan_map_chain_propagates_flag_to_every_param() {
    let mut wm = WeightMap::default();
    wm.insert(entry("tok_embeddings.weight"));
    wm.insert(entry("blocks.0.norm.weight"));
    wm.insert(entry("blocks.3.ffn.w_gate"));
    wm.insert(entry("output.weight"));

    let plan = plan_map(&wm, &PrecisionConfig::default());
    let by_name: std::collections::HashMap<&str, &ParamPrecision> =
        plan.params.iter().map(|p| (p.name.as_str(), p)).collect();
    assert!(by_name["tok_embeddings.weight"].stochastic_rounding);
    assert!(!by_name["blocks.0.norm.weight"].stochastic_rounding);
    assert!(!by_name["blocks.3.ffn.w_gate"].stochastic_rounding);
    assert!(!by_name["output.weight"].stochastic_rounding);
}

fn force_param(name: &str, tier: Tier, stochastic: bool) -> ParamPrecision {
    let (m, v) = tier.precision();
    ParamPrecision {
        name: name.to_string(),
        layer: Some(0),
        tier,
        m_precision: m,
        v_precision: v,
        stochastic_rounding: stochastic,
        sensitivity_score: 0.0,
        param_bytes: 16,
        optim_bytes: 0,
    }
}

fn stochastic_store_count(prog: &QuantizedOptimProgram) -> usize {
    prog.ops
        .iter()
        .filter(|op| {
            matches!(op, QuantizedOptimOp::QuantStoreM { stochastic: true, .. })
                || matches!(op, QuantizedOptimOp::QuantStoreV { stochastic: true, .. })
        })
        .count()
}

#[test]
fn forced_int8_embedding_emits_stochastic_stores() {
    // If a future tier-override (or user policy) pushes an embedding down to
    // VeryLow (INT8 m + INT8 v), the stochastic flag MUST flow to BOTH stores.
    let p = force_param("tok_embeddings.weight", Tier::VeryLow, true);
    let prog = emit_step(&p, &AdamWHyperparams::default());
    assert_eq!(stochastic_store_count(&prog), 2);
    assert!(prog.stochastic_rounding);
}

#[test]
fn forced_low_tier_with_stochastic_only_int8_store_is_stochastic() {
    // Tier::Low = (INT8 m, FP16 v). The stochastic flag fires only on the
    // INT8 store (m); FP16 v is bit-exact, no SR needed (matches emit_step's
    // m_stochastic / v_stochastic gating).
    let p = force_param("tok_embeddings.weight", Tier::Low, true);
    let prog = emit_step(&p, &AdamWHyperparams::default());
    assert_eq!(stochastic_store_count(&prog), 1);
}

#[test]
fn fp32_embedding_records_flag_but_no_stochastic_store_fires() {
    // Real-world default: embeddings pin to Tier::High (FP32 m+v). The flag
    // is recorded on the ParamPrecision (paper-faithful: "the compiler knows")
    // but no QuantStore op carries stochastic=true because FP32 doesn't quantize.
    // This documents the "inert under default tier" property — the flag is
    // policy-correct even when no stochastic store fires.
    let p = force_param("tok_embeddings.weight", Tier::High, true);
    let prog = emit_step(&p, &AdamWHyperparams::default());
    assert!(prog.stochastic_rounding); // program-level flag preserved
    assert_eq!(stochastic_store_count(&prog), 0); // FP32 stores never stochastic
}

#[test]
fn non_embedding_int8_param_gets_deterministic_rounding() {
    // §3.3 second half: "deterministic rounding for everything else."
    let p = force_param("blocks.3.ffn.w_gate", Tier::VeryLow, false);
    let prog = emit_step(&p, &AdamWHyperparams::default());
    assert_eq!(stochastic_store_count(&prog), 0);
}
