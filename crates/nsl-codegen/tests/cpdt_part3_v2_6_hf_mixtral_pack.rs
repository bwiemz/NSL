//! CPDT Part III v2.6 integration test:
//! `pack_hf_mixtral_experts` produces a WeightMap that
//! `derive_v4_dims` resolves into the correct
//! `(hidden_dim, intermediate_dim)` tuple.
//!
//! Without this test the v2.6 packing primitive is decoupled from the
//! v4 SwiGLU lowering path — passing the unit tests in `moe_hf_pack`
//! alone would not catch a regression where the packed shape OR the
//! NSL naming convention drifts away from what `derive_v4_dims`
//! expects.

use nsl_codegen::moe::derive_v4_dims;
use nsl_codegen::moe_hf_pack::pack_hf_mixtral_experts;
use nsl_codegen::weight_aware::{WeightDType, WeightEntry, WeightMap};

fn ones_entry(name: &str, shape: Vec<usize>) -> WeightEntry {
    let num_elements = shape.iter().product::<usize>();
    let mut data = Vec::with_capacity(num_elements * 4);
    for _ in 0..num_elements {
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }
    WeightEntry::new(name.to_string(), data, shape, WeightDType::F32)
}

#[test]
fn hf_mixtral_pack_produces_weight_map_that_derive_v4_dims_resolves() {
    // Realistic Mixtral-ish scaled-down config:
    // hidden=8, intermediate=16, num_experts=4. HF stores per-expert
    // tensors as `[out_features, in_features]`:
    //   w1 (gate) = [intermediate=16, hidden=8]
    //   w3 (up)   = [intermediate=16, hidden=8]
    //   w2 (down) = [hidden=8, intermediate=16]
    let hidden = 8;
    let intermediate = 16;
    let num_experts = 4;

    let mut wm = WeightMap::new_for_test();

    // Router under the HF gate-style name. derive_v4_dims accepts BOTH
    // `router.weight` and `gate.weight` for the router (see moe.rs);
    // HF Mixtral uses the `gate.weight` form on the routing matrix.
    // Note: the routing-matrix is NOT in the per-expert HF pack and is
    // therefore loaded directly under its HF name.
    wm.insert(ones_entry("moe0.gate.weight", vec![hidden, num_experts]));

    // Per-expert HF projections.
    for e in 0..num_experts {
        wm.insert(ones_entry(
            &format!("moe0.experts.{}.w1.weight", e),
            vec![intermediate, hidden],
        ));
        wm.insert(ones_entry(
            &format!("moe0.experts.{}.w3.weight", e),
            vec![intermediate, hidden],
        ));
        wm.insert(ones_entry(
            &format!("moe0.experts.{}.w2.weight", e),
            vec![hidden, intermediate],
        ));
    }

    // Pack — HF prefix and NSL target prefix are the same string
    // (the simple case; non-identity target is exercised in the unit
    // tests inside moe_hf_pack).
    let outcome = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", num_experts)
        .expect("pack should succeed for a valid HF Mixtral layout");
    assert_eq!(outcome.inserted_names.len(), 3);
    assert_eq!(outcome.removed_names.len(), 3 * num_experts);

    // Now derive_v4_dims must resolve the packed entries.
    let dims = derive_v4_dims(&wm, "moe0", num_experts);
    assert_eq!(
        dims,
        Some((hidden, intermediate)),
        "derive_v4_dims must resolve packed HF-Mixtral entries into (hidden, intermediate)"
    );

    // Cross-check the packed shapes match what derive_v4_dims keyed on.
    let gate = wm.get("moe0.experts.gate.weight").expect("gate present");
    assert_eq!(gate.shape, vec![num_experts, hidden * intermediate]);
    let up = wm.get("moe0.experts.up.weight").expect("up present");
    assert_eq!(up.shape, vec![num_experts, hidden * intermediate]);
    let down = wm.get("moe0.experts.down.weight").expect("down present");
    assert_eq!(down.shape, vec![num_experts, intermediate * hidden]);
}

#[test]
fn hf_mixtral_pack_with_disagreeing_per_expert_shapes_does_not_drift_through_derive_v4() {
    // Variant of the happy-path test where one expert has a different
    // intermediate (16 vs 8). Pack must REFUSE, and the WeightMap must
    // be byte-identical to the pre-pack state so a subsequent retry
    // with a corrected fixture succeeds (no half-packed state).
    //
    // v2.6 adversarial-review fix F7: the original version of this
    // test only asserted one HF source present + one packed entry
    // absent — a strict subset of the atomicity contract. The fix
    // expands the assertions to all 6 HF sources still present + all
    // 3 packed entries absent + entry count unchanged, so a regression
    // that re-introduces per-projection mutation would be caught.
    let hidden = 8;
    let intermediate = 16;
    let num_experts = 2;

    let mut wm = WeightMap::new_for_test();
    wm.insert(ones_entry("moe0.gate.weight", vec![hidden, num_experts]));
    wm.insert(ones_entry(
        "moe0.experts.0.w1.weight",
        vec![intermediate, hidden],
    ));
    wm.insert(ones_entry(
        "moe0.experts.0.w3.weight",
        vec![intermediate, hidden],
    ));
    wm.insert(ones_entry("moe0.experts.0.w2.weight", vec![hidden, intermediate]));
    // Expert 1 has half the intermediate dim.
    wm.insert(ones_entry("moe0.experts.1.w1.weight", vec![8, hidden]));
    wm.insert(ones_entry(
        "moe0.experts.1.w3.weight",
        vec![intermediate, hidden],
    ));
    wm.insert(ones_entry("moe0.experts.1.w2.weight", vec![hidden, intermediate]));

    let pre_pack_entry_count = wm.len();

    let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", num_experts)
        .expect_err("disagreeing shapes must refuse");
    assert!(
        format!("{:?}", err).contains("ShapeMismatch"),
        "expected ShapeMismatch, got {:?}",
        err,
    );

    // Strengthened F7 atomicity assertions:
    // 1. ALL three packed entries absent.
    for nsl in ["gate", "up", "down"] {
        assert!(
            wm.get(&format!("moe0.experts.{}.weight", nsl)).is_none(),
            "atomicity violation: {} packed entry leaked",
            nsl,
        );
    }
    // 2. ALL 3*num_experts HF source entries still present.
    for proj in ["w1", "w2", "w3"] {
        for e in 0..num_experts {
            assert!(
                wm.get(&format!("moe0.experts.{}.{}.weight", e, proj)).is_some(),
                "atomicity violation: moe0.experts.{}.{}.weight was removed before refusal",
                e,
                proj,
            );
        }
    }
    // 3. WeightMap entry count is unchanged.
    assert_eq!(
        wm.len(),
        pre_pack_entry_count,
        "WeightMap entry count must be byte-identical to pre-pack state on refusal",
    );

    // And the pre-existing HF router is also untouched.
    let router = wm.get("moe0.gate.weight").expect("router unchanged");
    assert_eq!(router.shape, vec![hidden, num_experts]);
}

#[test]
fn hf_mixtral_pack_atomicity_holds_when_up_projection_shape_mismatches() {
    // v2.6 adversarial-review fix F1/F3/F6 (atomicity bug, integration
    // layer): the original integration test only triggered the failure
    // on w1 (Gate, the FIRST projection in HfProjection::ALL), which
    // bails before any mutation under either the old or the refactored
    // implementation. To catch the half-packed regression at the
    // integration layer, mismatch on w3 (Up, the SECOND projection) —
    // under the old per-projection-commit implementation, Gate would
    // have been packed + Gate sources removed before Up failed.
    let hidden = 8;
    let intermediate = 16;
    let num_experts = 2;
    let mut wm = WeightMap::new_for_test();
    wm.insert(ones_entry("moe0.gate.weight", vec![hidden, num_experts]));
    // Gate + Down consistent across both experts.
    for e in 0..num_experts {
        wm.insert(ones_entry(
            &format!("moe0.experts.{}.w1.weight", e),
            vec![intermediate, hidden],
        ));
        wm.insert(ones_entry(
            &format!("moe0.experts.{}.w2.weight", e),
            vec![hidden, intermediate],
        ));
    }
    // Up: expert 0 OK, expert 1 mismatched intermediate.
    wm.insert(ones_entry(
        "moe0.experts.0.w3.weight",
        vec![intermediate, hidden],
    ));
    wm.insert(ones_entry("moe0.experts.1.w3.weight", vec![8, hidden]));

    let pre = wm.len();
    let err = pack_hf_mixtral_experts(&mut wm, "moe0", "moe0", num_experts).unwrap_err();
    assert!(
        format!("{:?}", err).contains("ShapeMismatch")
            && format!("{:?}", err).contains("Up"),
        "expected ShapeMismatch on Up, got {:?}",
        err,
    );
    assert_eq!(wm.len(), pre);
    for nsl in ["gate", "up", "down"] {
        assert!(wm.get(&format!("moe0.experts.{}.weight", nsl)).is_none());
    }
    // Specifically: Gate's w1 sources MUST still be present — the
    // pre-refactor implementation would have removed them already.
    assert!(wm.get("moe0.experts.0.w1.weight").is_some());
    assert!(wm.get("moe0.experts.1.w1.weight").is_some());
}
