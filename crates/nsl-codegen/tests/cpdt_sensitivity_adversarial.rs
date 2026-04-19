//! Phase 1 Commit 3 — adversarial correctness gate.
//!
//! Purpose: prove the weight-aware path is doing real work. Catches
//! silent-stub, mis-calibration, cross-layer contamination, clone
//! aliasing, and future kind-override regressions in one fixture.
//!
//! Ignored in Commit 3; unblocked in Commit 4.

use std::path::PathBuf;

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    LayerKind, CALIB_ALPHA, CALIB_T0,
};
use nsl_codegen::cpdt_tier_apply::Tier;
use nsl_codegen::weight_aware::{write_f64_as_dtype, WeightMap};

const TARGET_NAME: &str = "blocks.4.ffn.w_down.weight";
const N_LAYERS: u32 = 8;

fn score_layer(wm: &WeightMap, name: &str) -> (Tier, f64, LayerKind) {
    let entry = wm.get(name).expect("tensor missing");
    let layer = layer_of(name);
    let kind = classify_layer_kind(name, layer, N_LAYERS);
    let gm = gradient_magnitude_est(Some(entry));
    let pos = position_criticality(layer, N_LAYERS, CALIB_ALPHA);
    let elts = entry.num_elements.max(1) as f64;
    let score = gm * pos / elts;
    (assign_tier(score, kind), score, kind)
}

fn scale_entry_in_place(wm: &mut WeightMap, name: &str, factor: f64) {
    let entry = wm.get_mut(name).expect("tensor missing");
    let bw = entry.dtype.byte_width();
    let mut new_data = Vec::with_capacity(entry.data.len());
    for i in 0..entry.num_elements {
        let off = i * bw;
        let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
        let scaled = v * factor;
        let mut buf = vec![0u8; bw];
        write_f64_as_dtype(scaled, entry.dtype, &mut buf);
        new_data.extend_from_slice(&buf);
    }
    entry.data = new_data;
}

#[test]
fn adversarial_localized_tier_shift() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/cpdt_calibration/calib_small.safetensors");
    let original = WeightMap::load(&path)
        .unwrap_or_else(|e| panic!("calib_small fixture missing at {}: {e:?}", path.display()));

    // Precondition 1: target layer is not kind-overridden.
    let (_, _, target_kind) = score_layer(&original, TARGET_NAME);
    assert!(
        !target_kind.is_kind_overridden(),
        "adversarial target must be a scored (not kind-overridden) layer; \
         current target kind = {target_kind:?}. Revisit fixture selection."
    );

    // Clone + verify clone isolation.
    let mut clone = original.clone();
    let embed_rms_before = {
        let e = original.get("tok_embeddings.weight").unwrap();
        gradient_magnitude_est(Some(e))
    };

    // Compute multiplier from calibrated constants to guarantee Tier::High.
    let (baseline_tier, s_pre, _) = score_layer(&original, TARGET_NAME);
    let m = (CALIB_T0 / s_pre) * 1.5;
    eprintln!("adversarial: baseline tier {baseline_tier:?}, s_pre = {s_pre:.6}, M = {m:.3}");

    // Apply localized mutation.
    scale_entry_in_place(&mut clone, TARGET_NAME, m);

    // Precondition 2: clone isolation held (original embedding unchanged).
    let embed_rms_after = {
        let e = original.get("tok_embeddings.weight").unwrap();
        gradient_magnitude_est(Some(e))
    };
    assert_eq!(
        embed_rms_before, embed_rms_after,
        "WeightMap::clone aliased entries; adversarial fixture invariants broken."
    );

    // Strong assertion 1: target lands at Tier::High exactly.
    let (adv_tier, adv_score, _) = score_layer(&clone, TARGET_NAME);
    assert_eq!(
        adv_tier,
        Tier::High,
        "target layer did not reach Tier::High under {m:.2}× scaling (got {adv_tier:?}, score {adv_score:.6}). \
         Likely causes: silent-stub gradient_magnitude_est, mis-calibrated T0, broken scorer path."
    );

    // Strong assertion 2: every other scored layer's tier unchanged.
    let all_names: Vec<String> = original.entries().map(|(n, _)| n.clone()).collect();
    for name in all_names {
        if name == TARGET_NAME {
            continue;
        }
        let (orig_tier, _, _) = score_layer(&original, &name);
        let (clone_tier, _, _) = score_layer(&clone, &name);
        assert_eq!(
            clone_tier, orig_tier,
            "layer {name} tier changed unexpectedly under localized mutation of {TARGET_NAME}. \
             Scorer may be broadcasting weight effects across the model."
        );
    }
}
