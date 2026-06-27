//! Phase 1 follow-up to PR #80 + PR #88 — layer-prefix validation unit tests
//! for `cpdt_sensitivity::validate`. See the design spec at
//! `docs/superpowers/specs/2026-04-20-cpdt-validate-body-design.md`.

use std::collections::HashMap;
use std::io::Write;

use nsl_codegen::cpdt_sensitivity::{validate, ValidationError};
use nsl_codegen::weight_aware::WeightMap;
use nsl_codegen::wggo_apply::AppliedPlan;
use nsl_codegen::wggo_dp::CoarseDecision;

/// Write a minimal safetensors file containing `tensor_names` (each a 1×1
/// f32 tensor) to a temp path, load it as a WeightMap, and return the pair
/// so the TempPath isn't dropped before the WeightMap is done reading it.
fn wm_with_names(tensor_names: &[&str]) -> (WeightMap, tempfile::TempPath) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    let mut owned: Vec<(String, Vec<u8>)> = Vec::new();
    for name in tensor_names {
        let bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        owned.push((name.to_string(), bytes));
    }
    let views: HashMap<String, TensorView<'_>> = owned
        .iter()
        .map(|(n, b)| (n.clone(), TensorView::new(Dtype::F32, vec![1], b.as_slice()).unwrap()))
        .collect();
    let bytes = serialize(&views, &None).unwrap();

    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&bytes).unwrap();
    let tp = tmp.into_temp_path();
    let wm = WeightMap::load(&tp).unwrap();
    (wm, tp)
}

/// Build an `AppliedPlan` with the given layer names. Fields irrelevant to
/// `validate` (costs, shard factors, etc.) are filled with defaults — the
/// validator only reads `layer_name`.
fn plan_with_layers(layer_names: &[&str]) -> AppliedPlan {
    let layers = layer_names
        .iter()
        .enumerate()
        .map(|(i, name)| nsl_codegen::wggo_apply::AppliedLayer {
            layer_index: i as u32,
            layer_name: (*name).to_string(),
            coarse: CoarseDecision::KeepFull,
            pipeline_stage: 0,
            shard_factor: 1,
            shard_grads: 1,
            shard_optim: 1,
            active_heads: 1,
            ffn_width: 0,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 32,
            optim_v_bits: 32,
            fase_fused: false,
            packing_mode: 0,
            estimated_us: 0.0,
            param_bytes: 0,
            activation_bytes: 0,
        })
        .collect();
    AppliedPlan {
        layers,
        total_us: 0.0,
        peak_memory_bytes: 0,
    }
}

#[test]
fn all_layers_matched_green() {
    // 3 hierarchical layers, each with at least one matching tensor.
    let (wm, _tp) = wm_with_names(&[
        "blocks.0.attn.wq.weight",
        "blocks.0.ffn.w_down.weight",
        "blocks.1.attn.wq.weight",
        "blocks.2.ffn.w_gate.weight",
    ]);
    let applied = plan_with_layers(&["blocks.0", "blocks.1", "blocks.2"]);
    assert!(validate(&wm, &applied).is_ok());
}

#[test]
fn single_missing_layer_red() {
    // blocks.0 and blocks.2 matched; blocks.1 has no matching tensors.
    let (wm, _tp) = wm_with_names(&[
        "blocks.0.attn.wq.weight",
        "blocks.2.ffn.w_down.weight",
    ]);
    let applied = plan_with_layers(&["blocks.0", "blocks.1", "blocks.2"]);
    let err = validate(&wm, &applied).expect_err("should fail");
    match err {
        ValidationError::LayersMissing {
            missing,
            total_layers_checked,
            ..
        } => {
            assert_eq!(missing, vec!["blocks.1".to_string()]);
            assert_eq!(total_layers_checked, 3);
        }
        other => panic!("expected LayersMissing, got {other:?}"),
    }
}

#[test]
fn multiple_missing_layers_red_aggregates() {
    // HuggingFace-format checkpoint loaded for an NSL-native model.
    // AppliedPlan has blocks.0..blocks.11; WeightMap has transformer.h.N.*
    // tensors. All 12 layers are missing; the error aggregates them and
    // includes `transformer.h` in the prefix summary.
    let mut names: Vec<String> = Vec::new();
    for i in 0..12 {
        names.push(format!("transformer.h.{i}.attn.c_attn.weight"));
        names.push(format!("transformer.h.{i}.mlp.c_fc.weight"));
    }
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let (wm, _tp) = wm_with_names(&name_refs);

    let block_names: Vec<String> = (0..12).map(|i| format!("blocks.{i}")).collect();
    let block_refs: Vec<&str> = block_names.iter().map(|s| s.as_str()).collect();
    let applied = plan_with_layers(&block_refs);

    let err = validate(&wm, &applied).expect_err("should fail");
    match err {
        ValidationError::LayersMissing {
            missing,
            total_layers_checked,
            weight_map_prefix_summary,
            weight_map_total_tensors,
        } => {
            assert_eq!(missing.len(), 12);
            assert!(
                missing.contains(&"blocks.0".to_string())
                    && missing.contains(&"blocks.11".to_string()),
                "aggregation should list all 12 missing layers, got {missing:?}"
            );
            assert_eq!(total_layers_checked, 12);
            assert_eq!(weight_map_total_tensors, 24);
            assert!(
                weight_map_prefix_summary
                    .iter()
                    .any(|p| p == "transformer.h"),
                "prefix summary should contain `transformer.h`, got {weight_map_prefix_summary:?}"
            );
        }
        other => panic!("expected LayersMissing, got {other:?}"),
    }

    // The Display impl should contain the diagnostic shape used in the spec.
    let display = format!("{}", validate(&wm, &applied).unwrap_err());
    assert!(
        display.contains("Missing layers (12 of 12)"),
        "display should report aggregated count: {display}"
    );
    assert!(
        display.contains("transformer.h"),
        "display should mention the WeightMap prefix: {display}"
    );
}

#[test]
fn empty_weightmap_red() {
    // A valid but empty WeightMap (e.g. the load succeeded but no tensors
    // matched the model's declarations — pathological but possible).
    let (wm, _tp) = wm_with_names(&["unused_scalar"]);
    let applied = plan_with_layers(&["blocks.0", "blocks.1", "blocks.2"]);
    let err = validate(&wm, &applied).expect_err("should fail");
    match err {
        ValidationError::LayersMissing {
            missing,
            total_layers_checked,
            weight_map_prefix_summary,
            weight_map_total_tensors,
        } => {
            assert_eq!(missing.len(), 3);
            assert_eq!(total_layers_checked, 3);
            assert_eq!(weight_map_total_tensors, 1);
            // `unused_scalar` has no dots → top_level_prefix returns the name itself.
            assert_eq!(weight_map_prefix_summary, vec!["unused_scalar".to_string()]);
        }
        other => panic!("expected LayersMissing, got {other:?}"),
    }
}

#[test]
fn exactly_eight_prefixes_does_not_emit_truncation_marker() {
    // Regression for an early bug: the "..." truncation marker would fire
    // whenever prefix count equaled PREFIX_SUMMARY_MAX (8) exactly, even
    // when nothing was actually truncated. Build a WeightMap with exactly 8
    // distinct top-level prefixes; assert the summary has 8 entries, no "...".
    let names_owned: Vec<String> = (0..8)
        .map(|i| format!("namespace{i}.tensor.weight"))
        .collect();
    let name_refs: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();
    let (wm, _tp) = wm_with_names(&name_refs);
    let applied = plan_with_layers(&["blocks.0"]);
    let err = validate(&wm, &applied).expect_err("should fail");
    match err {
        ValidationError::LayersMissing {
            weight_map_prefix_summary,
            ..
        } => {
            assert_eq!(
                weight_map_prefix_summary.len(),
                8,
                "exactly-8-prefixes case: no truncation, no '...' marker; got {weight_map_prefix_summary:?}"
            );
            assert!(
                !weight_map_prefix_summary.contains(&"...".to_string()),
                "should not contain truncation marker when nothing was truncated"
            );
        }
        other => panic!("expected LayersMissing, got {other:?}"),
    }
}

#[test]
fn nine_prefixes_emits_truncation_marker() {
    // Dual of the previous test: 9 prefixes exceed the cap of 8, so the
    // "..." marker should appear and the summary should total 9 entries
    // (8 truncated + the "..." marker).
    let names_owned: Vec<String> = (0..9)
        .map(|i| format!("namespace{i}.tensor.weight"))
        .collect();
    let name_refs: Vec<&str> = names_owned.iter().map(|s| s.as_str()).collect();
    let (wm, _tp) = wm_with_names(&name_refs);
    let applied = plan_with_layers(&["blocks.0"]);
    let err = validate(&wm, &applied).expect_err("should fail");
    match err {
        ValidationError::LayersMissing {
            weight_map_prefix_summary,
            ..
        } => {
            assert_eq!(
                weight_map_prefix_summary.len(),
                9,
                "9 distinct prefixes: 8 summary + 1 '...' marker; got {weight_map_prefix_summary:?}"
            );
            assert_eq!(
                weight_map_prefix_summary.last().map(|s| s.as_str()),
                Some("..."),
                "truncation marker should be the last entry"
            );
        }
        other => panic!("expected LayersMissing, got {other:?}"),
    }
}

#[test]
fn empty_applied_plan_is_ok() {
    // Documents intentional behavior: a plan with no layers produces no
    // layers to check, so `validate` returns Ok. Guards against a future
    // policy change that might want to error here.
    let (wm, _tp) = wm_with_names(&["any.tensor.weight"]);
    let applied = plan_with_layers(&[]);
    assert!(validate(&wm, &applied).is_ok());
}

#[test]
fn other_catchall_layer_is_skipped() {
    // The "other" catch-all (embeddings, norms, LM head) is heterogeneous;
    // Phase 1 skips it with no missing-layer error. Adds one more test beyond
    // the four core cases because the skip behavior is load-bearing for
    // Phase 1's scope boundary.
    let (wm, _tp) = wm_with_names(&["blocks.0.attn.wq.weight"]);
    let applied = plan_with_layers(&["blocks.0", "other"]);
    assert!(
        validate(&wm, &applied).is_ok(),
        "`other` layer_name should be skipped, not treated as missing"
    );
}
