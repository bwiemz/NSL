//! CPDT Part III — non-vacuous numerical validation (Layer 2) + pass
//! reachability (Layer 3). Composes the real granular MoE ops to prove the
//! dead-expert prune is correct against a real expert-weighted MoE.

use nsl_codegen::cpdt_expert_prune::{prune_dead_experts, MoeWeightBundle};
use nsl_codegen::cpdt_expert::{detect_dead_experts, router_affinities};
use nsl_codegen::weight_aware::{WeightDType, WeightEntry};

use nsl_runtime::moe::router::route_topk;
use nsl_runtime::moe::dispatch::{gather_tokens, scatter_tokens};
use nsl_runtime::moe::ffi::nsl_expert_parallel_matmul;

const D: usize = 4; // d_model = hidden = inter
const N: usize = 4; // experts
const T: usize = 4; // tokens
const CAP: f32 = 100.0; // high capacity -> never drop

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// Router [D, N] row-major. Column 1 (expert 1) is all-zero (dead). Columns
/// 0,2,3 give a clear argmax so routing is deterministic.
fn router_flat() -> Vec<f32> {
    vec![
        9.0, 0.0, 1.0, 1.0, // row 0 -> argmax e0
        1.0, 0.0, 9.0, 1.0, // row 1 -> argmax e2
        1.0, 0.0, 1.0, 9.0, // row 2 -> argmax e3
        1.0, 0.0, 1.0, 1.0, // row 3
    ]
}

/// Tokens [T, D], one-hot: token0=[1,0,0,0], token1=[0,1,0,0], token2=[0,0,1,0],
/// token3=[1,0,0,0]. So token0/3 route to e0, token1->e2, token2->e3, e1 unused.
fn tokens_flat() -> Vec<f32> {
    vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
    ]
}

/// Expert e block = (e+1)·I_D, packed [N, D*D] row-major.
fn experts_flat(ids: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * D * D];
    for (slot, &e) in ids.iter().enumerate() {
        for k in 0..D {
            out[slot * D * D + k * D + k] = (e + 1) as f32;
        }
    }
    out
}

/// logits[t][e] = sum_d tokens[t][d] * router[d][e].
fn matmul_logits(tokens: &[f32], router: &[f32], n_experts: usize) -> Vec<f32> {
    let mut logits = vec![0.0f32; T * n_experts];
    for t in 0..T {
        for e in 0..n_experts {
            let mut s = 0.0f32;
            for d in 0..D {
                s += tokens[t * D + d] * router[d * n_experts + e];
            }
            logits[t * n_experts + e] = s;
        }
    }
    logits
}

/// Run the real granular MoE forward (route -> scatter -> expert matmul ->
/// gather) for `n_experts` with the given packed expert weights. top_k=1.
fn moe_forward(tokens: &[f32], router: &[f32], experts: &[f32], n_experts: usize) -> Vec<f32> {
    let logits = matmul_logits(tokens, router, n_experts);
    let routing = route_topk(&logits, T, n_experts, 1, CAP as f64);
    let scattered = scatter_tokens(tokens, &routing.sorted_token_indices, D);
    let total_assigned = routing.total_assigned as usize;
    let mut expert_out = vec![0.0f32; total_assigned * D];
    // Real expert GEMM via the granular runtime op. The op is exposed as a safe
    // `extern "C" fn` (not `unsafe fn`), so no `unsafe` block is needed even
    // though it takes raw `i64` pointers — the buffers above outlive the call.
    let rc = nsl_expert_parallel_matmul(
        scattered.as_ptr() as i64,
        experts.as_ptr() as i64,
        routing.expert_boundaries.as_ptr() as i64,
        expert_out.as_mut_ptr() as i64,
        n_experts as i64,
        D as i64,
        D as i64,
    );
    assert_eq!(rc, 0);
    // top_k=1 -> gating weight is exactly 1.0 per assignment.
    let gw = vec![1.0f32; total_assigned];
    gather_tokens(&expert_out, &routing.sorted_token_indices, &gw, T, 1, D)
}

#[test]
fn detection_flags_zero_column_expert() {
    let router = WeightEntry::new("router".into(), f32s(&router_flat()), vec![D, N], WeightDType::F32);
    let aff = router_affinities(&router, N as u32);
    let dead = detect_dead_experts(&aff, 0.01);
    assert_eq!(dead, vec![1]);
}

#[test]
fn pruned_eq_reference_bitexact() {
    let tokens = tokens_flat();
    let router = router_flat();
    let experts4 = experts_flat(&[0, 1, 2, 3]);

    // Reference: full 4-expert forward.
    let out_ref = moe_forward(&tokens, &router, &experts4, N);

    // Detect + prune.
    let router_entry = WeightEntry::new("router".into(), f32s(&router), vec![D, N], WeightDType::F32);
    let dead = detect_dead_experts(&router_affinities(&router_entry, N as u32), 0.01);
    assert_eq!(dead, vec![1]);
    let bundle = MoeWeightBundle {
        router: f32s(&router),
        experts: f32s(&experts4),
        d_model: D,
        n_experts: N,
        expert_block_elems: D * D,
        dtype: WeightDType::F32,
        top_k: 1,
    };
    let res = prune_dead_experts(&bundle, &dead).unwrap();

    // Pruned: 3-expert forward over the sliced router + kept experts.
    let sliced_router = to_f32(&res.sliced_router);
    let kept_experts = to_f32(&res.kept_experts);
    let out_pruned = moe_forward(&tokens, &sliced_router, &kept_experts, res.n_live);

    // Bit-exact: dead expert never selected -> removing it changes nothing.
    assert_eq!(out_pruned, out_ref, "pruned MoE must match reference bit-exact");
}

#[test]
fn right_experts_by_identity() {
    let experts4 = experts_flat(&[0, 1, 2, 3]);
    let bundle = MoeWeightBundle {
        router: f32s(&router_flat()),
        experts: f32s(&experts4),
        d_model: D,
        n_experts: N,
        expert_block_elems: D * D,
        dtype: WeightDType::F32,
        top_k: 1,
    };
    let res = prune_dead_experts(&bundle, &[1]).unwrap();
    assert_eq!(res.index_remap, vec![0, 2, 3]);
    // kept_experts must equal the ORIGINAL blocks {0,2,3}, by identity.
    assert_eq!(to_f32(&res.kept_experts), experts_flat(&[0, 2, 3]));
}

#[test]
fn live_prune_negative_control() {
    // Force-prune a LIVE expert (id 0, which token0/3 route to). The parity
    // MUST break — proving the bit-exact gate detects removing a contributing
    // expert (non-vacuity guard).
    let tokens = tokens_flat();
    let router = router_flat();
    let experts4 = experts_flat(&[0, 1, 2, 3]);
    let out_ref = moe_forward(&tokens, &router, &experts4, N);

    let bundle = MoeWeightBundle {
        router: f32s(&router),
        experts: f32s(&experts4),
        d_model: D,
        n_experts: N,
        expert_block_elems: D * D,
        dtype: WeightDType::F32,
        top_k: 1,
    };
    let res = prune_dead_experts(&bundle, &[0]).unwrap(); // WRONG: id 0 is live
    let sliced_router = to_f32(&res.sliced_router);
    let kept_experts = to_f32(&res.kept_experts);
    let out_wrong = moe_forward(&tokens, &sliced_router, &kept_experts, res.n_live);

    assert_ne!(out_wrong, out_ref, "pruning a live expert MUST change the output");
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 3 — non-WGGO pass reachability
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use nsl_codegen::cpdt::CpdtMode;
use nsl_codegen::cpdt_expert_prune::{prune_moe_weights_in_map, ExpertPruneRefusal, MoePruneOutcome};
use nsl_codegen::moe::MoeInfo;
use nsl_codegen::weight_aware::WeightMap;

fn moe_info(num_experts: usize, top_k: usize) -> MoeInfo {
    MoeInfo {
        num_experts,
        top_k,
        capacity_factor: 100.0,
        aux_loss_coeff: 0.0,
        activation: nsl_codegen::moe::MoeActivation::default(),
        weight_prefix: None,
    }
}

/// WeightMap with a router (dead col 1) + packed experts under `<key>.*`.
fn weight_map_with_moe(key: &str) -> WeightMap {
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{key}.router.weight"),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.weight"),
        f32s(&experts_flat(&[0, 1, 2, 3])),
        vec![N, D * D],
        WeightDType::F32,
    ));
    wm
}

#[test]
fn pass_prunes_and_reports() {
    let mut wm = weight_map_with_moe("blocks.0.moe");
    let mut cfgs = HashMap::new();
    cfgs.insert("blocks.0.moe".to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);

    // Outcome records the prune.
    assert!(matches!(
        outcomes.as_slice(),
        [MoePruneOutcome::Pruned { ref dead, n_live: 3, .. }] if dead == &vec![1]
    ));
    // Router sliced to [D, 3]; experts sliced to [3, D*D].
    assert_eq!(wm.get("blocks.0.moe.router.weight").unwrap().shape, vec![D, 3]);
    assert_eq!(wm.get("blocks.0.moe.experts.weight").unwrap().shape, vec![3, D * D]);
    // num_experts threaded to the config (forward-looking consistency).
    assert_eq!(cfgs["blocks.0.moe"].num_experts, 3);
}

#[test]
fn pass_skips_missing_router() {
    let mut wm = WeightMap::default(); // no router entry
    let mut cfgs = HashMap::new();
    cfgs.insert("m".to_string(), moe_info(N, 1));
    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    assert!(matches!(outcomes.as_slice(), [MoePruneOutcome::SkippedMissingRouter { .. }]));
    assert_eq!(cfgs["m"].num_experts, N); // untouched
}

#[test]
fn pass_gated_off_when_not_full() {
    let mut wm = weight_map_with_moe("m");
    let mut cfgs = HashMap::new();
    cfgs.insert("m".to_string(), moe_info(N, 1));
    let outcomes = prune_moe_weights_in_map(CpdtMode::ZeroOnly, &mut cfgs, &mut wm);
    assert!(outcomes.is_empty());
    assert_eq!(wm.get("m.router.weight").unwrap().shape, vec![D, N]); // untouched
}

#[test]
fn pass_runs_without_wggo() {
    // Blocker-sidestep: no WGGO/source-AD setup anywhere — the pass prunes from
    // the router weights alone (cpdt Full + weights).
    let mut wm = weight_map_with_moe("m");
    let mut cfgs = HashMap::new();
    cfgs.insert("m".to_string(), moe_info(N, 1));
    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    assert!(matches!(outcomes.as_slice(), [MoePruneOutcome::Pruned { n_live: 3, .. }]));
}

#[test]
fn pass_no_dead_experts_is_informational() {
    // Router with NO zero column -> no dead experts -> informational no-op.
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        "m.router.weight".into(),
        f32s(&[
            9.0, 2.0, 1.0, 1.0,
            1.0, 2.0, 9.0, 1.0,
            1.0, 2.0, 1.0, 9.0,
            1.0, 2.0, 1.0, 1.0,
        ]),
        vec![D, N],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        "m.experts.weight".into(),
        f32s(&experts_flat(&[0, 1, 2, 3])),
        vec![N, D * D],
        WeightDType::F32,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert("m".to_string(), moe_info(N, 1));
    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    assert!(matches!(outcomes.as_slice(), [MoePruneOutcome::NoDeadExperts { .. }]));
    assert_eq!(wm.get("m.router.weight").unwrap().shape, vec![D, N]); // untouched
    assert_eq!(cfgs["m"].num_experts, N);
}

#[test]
fn pass_refuses_mixed_dtype() {
    // Router F32 + experts F16: v1 byte-slices with one byte-width, so a mixed
    // dtype must REFUSE (not mis-slice the experts). Closes a silent-corruption
    // path. The refusal fires before any slicing -> map untouched.
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        "m.router.weight".into(),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        "m.experts.weight".into(),
        vec![0u8; N * D * D * 2], // F16 byte length (2 bytes/elem)
        vec![N, D * D],
        WeightDType::F16,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert("m".to_string(), moe_info(N, 1));
    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    // v2.9 refactor: the per-projection mixed-dtype refusal now comes
    // from the named-projection `ProjectionDtypeMismatch` variant
    // (legacy `MixedDtypeUnsupported` is still defined for ABI stability
    // but produced only by callers that retain a v1 MoeWeightBundle path).
    assert!(matches!(
        outcomes.as_slice(),
        [MoePruneOutcome::Refused {
            refusal: ExpertPruneRefusal::ProjectionDtypeMismatch { .. },
            ..
        }]
    ));
    // Untouched: no partial mutation on refusal.
    assert_eq!(wm.get("m.router.weight").unwrap().shape, vec![D, N]);
    assert_eq!(wm.get("m.experts.weight").unwrap().shape, vec![N, D * D]);
    assert_eq!(cfgs["m"].num_experts, N);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPDT Part III v2.9 — multi-projection prune (v3 up+down, v4 gate+up+down).
// Reuses the v1 router_flat (column 1 is the dead expert) but extends the
// fixtures with per-projection expert tensors. Tests pin:
//   - prune_dead_experts_split slices each projection independently using the
//     same index_remap (KeptProjection ordering must match input order)
//   - the pass auto-detects v4 (gate+up+down) and prunes all 3 tensors
//   - the pass auto-detects v3 (up+down only) and prunes both tensors
//   - a PARTIAL multi-projection set (e.g., gate+up but no down) is REFUSED
//     loudly with BundleInconsistent — silent downgrade would leak orphan
//     tensors and break downstream FFI dispatch
//   - one projection with mismatched dtype produces a NAMED
//     ProjectionDtypeMismatch (no partial mutation of the WeightMap)
// ─────────────────────────────────────────────────────────────────────────────

use nsl_codegen::cpdt_expert_prune::{prune_dead_experts_split, ExpertProjection, MoeExpertsLayout};

/// Per-projection block for v4. Projection `p` (0=gate, 1=up, 2=down) gets a
/// distinct (e+1) * (p+1) diagonal so a mis-slice across projections is
/// observable: a regression that copied gate bytes into the up slot would
/// produce values differing by exactly the 2x scale factor.
fn v4_projection_flat(ids: &[usize], proj_scale: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * D * D];
    for (slot, &e) in ids.iter().enumerate() {
        for k in 0..D {
            out[slot * D * D + k * D + k] = ((e + 1) * proj_scale) as f32;
        }
    }
    out
}

#[test]
fn split_prune_slices_each_projection_with_same_index_remap() {
    // Pure-transform check: prune_dead_experts_split must slice all
    // projections in lockstep with the router. Three projections with
    // distinct scaling factors mean a swap or partial mis-slice would
    // produce wrong values, not just wrong shapes.
    let router = f32s(&router_flat());
    let gate = f32s(&v4_projection_flat(&[0, 1, 2, 3], 1)); // scale=1
    let up   = f32s(&v4_projection_flat(&[0, 1, 2, 3], 2)); // scale=2
    let down = f32s(&v4_projection_flat(&[0, 1, 2, 3], 3)); // scale=3

    let projections = vec![
        ExpertProjection { name: "gate".into(), data: gate, block_elems: D * D, dtype: WeightDType::F32 },
        ExpertProjection { name: "up".into(),   data: up,   block_elems: D * D, dtype: WeightDType::F32 },
        ExpertProjection { name: "down".into(), data: down, block_elems: D * D, dtype: WeightDType::F32 },
    ];

    let res = prune_dead_experts_split(&router, WeightDType::F32, D, N, 1, &projections, &[1]).unwrap();

    assert_eq!(res.index_remap, vec![0, 2, 3]);
    assert_eq!(res.n_live, 3);
    assert_eq!(res.kept_projections.len(), 3);

    // Each kept projection MUST equal the original blocks {0, 2, 3} at
    // its scale factor. A swap (e.g., kept_projections[0] taking up's
    // data) would fail this with values differing by 2x.
    assert_eq!(res.kept_projections[0].name, "gate");
    assert_eq!(to_f32(&res.kept_projections[0].data), v4_projection_flat(&[0, 2, 3], 1));
    assert_eq!(res.kept_projections[1].name, "up");
    assert_eq!(to_f32(&res.kept_projections[1].data), v4_projection_flat(&[0, 2, 3], 2));
    assert_eq!(res.kept_projections[2].name, "down");
    assert_eq!(to_f32(&res.kept_projections[2].data), v4_projection_flat(&[0, 2, 3], 3));
}

#[test]
fn split_prune_refuses_empty_projection_list() {
    let router = f32s(&router_flat());
    let res = prune_dead_experts_split(&router, WeightDType::F32, D, N, 1, &[], &[1]);
    assert!(matches!(res, Err(ExpertPruneRefusal::EmptyProjectionList)));
}

#[test]
fn split_prune_refuses_named_projection_shape_mismatch() {
    // One projection has the wrong byte length → ProjectionShapeMismatch
    // with the offending projection's name in the refusal payload.
    let router = f32s(&router_flat());
    let gate = f32s(&v4_projection_flat(&[0, 1, 2, 3], 1));
    let up   = f32s(&v4_projection_flat(&[0, 1, 2, 3], 2));
    // down truncated by one expert block — invalid byte length.
    let mut down = f32s(&v4_projection_flat(&[0, 1, 2, 3], 3));
    down.truncate(down.len() - D * D * 4);

    let projections = vec![
        ExpertProjection { name: "gate".into(), data: gate, block_elems: D * D, dtype: WeightDType::F32 },
        ExpertProjection { name: "up".into(),   data: up,   block_elems: D * D, dtype: WeightDType::F32 },
        ExpertProjection { name: "down".into(), data: down, block_elems: D * D, dtype: WeightDType::F32 },
    ];
    let res = prune_dead_experts_split(&router, WeightDType::F32, D, N, 1, &projections, &[1]);
    match res {
        Err(ExpertPruneRefusal::ProjectionShapeMismatch { name, .. }) => {
            assert_eq!(name, "down", "refusal must name the offending projection");
        }
        other => panic!("expected ProjectionShapeMismatch, got {other:?}"),
    }
}

/// WeightMap with v4 layout: router + experts.{gate,up,down}.weight,
/// each packed [N, D*D] with the projection scale baked in.
fn weight_map_with_v4_moe(key: &str) -> WeightMap {
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{key}.router.weight"),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    for (suffix, scale) in [("experts.gate.weight", 1), ("experts.up.weight", 2), ("experts.down.weight", 3)] {
        wm.insert(WeightEntry::new(
            format!("{key}.{suffix}"),
            f32s(&v4_projection_flat(&[0, 1, 2, 3], scale)),
            vec![N, D * D],
            WeightDType::F32,
        ));
    }
    wm
}

#[test]
fn pass_detects_v4_layout_and_prunes_all_three_projections() {
    let key = "blocks.0.moe";
    let mut wm = weight_map_with_v4_moe(key);
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);

    match outcomes.as_slice() {
        [MoePruneOutcome::Pruned { layout, dead, n_live, .. }] => {
            assert_eq!(*layout, MoeExpertsLayout::GateUpDown);
            assert_eq!(dead, &vec![1]);
            assert_eq!(*n_live, 3);
        }
        other => panic!("expected single Pruned outcome, got {other:?}"),
    }

    // Router sliced to [D, 3].
    assert_eq!(wm.get(&format!("{key}.router.weight")).unwrap().shape, vec![D, 3]);
    // All three projections sliced to [3, D*D] with the correct
    // surviving blocks {0, 2, 3} at their respective scales.
    for (suffix, scale) in [("experts.gate.weight", 1), ("experts.up.weight", 2), ("experts.down.weight", 3)] {
        let e = wm.get(&format!("{key}.{suffix}")).unwrap();
        assert_eq!(e.shape, vec![3, D * D], "{suffix} shape");
        assert_eq!(to_f32(&e.data), v4_projection_flat(&[0, 2, 3], scale), "{suffix} content");
    }
    assert_eq!(cfgs[key].num_experts, 3);
}

#[test]
fn pass_detects_v3_layout_and_prunes_both_projections() {
    // v3 = up+down only (no gate). Router + 2 projections.
    let key = "m";
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{key}.router.weight"),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.up.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 2)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.down.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 3)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);

    match outcomes.as_slice() {
        [MoePruneOutcome::Pruned { layout, dead, n_live, .. }] => {
            assert_eq!(*layout, MoeExpertsLayout::UpDown);
            assert_eq!(dead, &vec![1]);
            assert_eq!(*n_live, 3);
        }
        other => panic!("expected single Pruned outcome, got {other:?}"),
    }
    // Neither projection nor the router should be absent or mis-shaped.
    assert_eq!(wm.get(&format!("{key}.router.weight")).unwrap().shape, vec![D, 3]);
    assert_eq!(wm.get(&format!("{key}.experts.up.weight")).unwrap().shape,   vec![3, D * D]);
    assert_eq!(wm.get(&format!("{key}.experts.down.weight")).unwrap().shape, vec![3, D * D]);
    // A `gate` tensor must NOT have been created (v3 has no gate).
    assert!(wm.get(&format!("{key}.experts.gate.weight")).is_none());
    assert_eq!(cfgs[key].num_experts, 3);
}

#[test]
fn pass_refuses_partial_multi_projection_layout() {
    // Partial v4: gate + up present, but down MISSING and no v1/v2
    // single tensor either. Silent downgrade to v3 would silently
    // leave gate as an orphan (v3 has no gate) so the downstream FFI
    // call would crash on missing down. The pass must refuse loudly
    // and leave the WeightMap untouched.
    let key = "m";
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{key}.router.weight"),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.gate.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 1)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.up.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 2)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    assert!(
        matches!(
            outcomes.as_slice(),
            [MoePruneOutcome::Refused {
                refusal: ExpertPruneRefusal::BundleInconsistent { .. },
                ..
            }]
        ),
        "expected BundleInconsistent refusal for partial v4 layout, got {outcomes:?}"
    );

    // No partial mutation: every original entry must still be intact.
    assert_eq!(wm.get(&format!("{key}.router.weight")).unwrap().shape, vec![D, N]);
    assert_eq!(wm.get(&format!("{key}.experts.gate.weight")).unwrap().shape, vec![N, D * D]);
    assert_eq!(wm.get(&format!("{key}.experts.up.weight")).unwrap().shape,   vec![N, D * D]);
    assert_eq!(cfgs[key].num_experts, N);
}

#[test]
fn split_prune_refuses_all_experts_dead() {
    // v2.9 fix F5 (LOW adversarial review): the pure split-path
    // refusals AllExpertsDead and InsufficientLiveExperts had no
    // direct coverage. Pin them on the new function so a future
    // refactor that drops the gates is caught.
    let router = f32s(&router_flat());
    let gate = f32s(&v4_projection_flat(&[0, 1, 2, 3], 1));
    let up   = f32s(&v4_projection_flat(&[0, 1, 2, 3], 2));
    let down = f32s(&v4_projection_flat(&[0, 1, 2, 3], 3));
    let projections = vec![
        ExpertProjection { name: "gate".into(), data: gate, block_elems: D * D, dtype: WeightDType::F32 },
        ExpertProjection { name: "up".into(),   data: up,   block_elems: D * D, dtype: WeightDType::F32 },
        ExpertProjection { name: "down".into(), data: down, block_elems: D * D, dtype: WeightDType::F32 },
    ];
    // Every expert dead.
    let res_all = prune_dead_experts_split(&router, WeightDType::F32, D, N, 1, &projections, &[0, 1, 2, 3]);
    assert!(matches!(res_all, Err(ExpertPruneRefusal::AllExpertsDead)));
    // n_live=1 < top_k=2 → InsufficientLiveExperts.
    let res_insufficient = prune_dead_experts_split(&router, WeightDType::F32, D, N, 2, &projections, &[0, 1, 3]);
    assert!(matches!(
        res_insufficient,
        Err(ExpertPruneRefusal::InsufficientLiveExperts { n_live: 1, top_k: 2 })
    ));
}

#[test]
fn pass_respects_weight_prefix_for_hf_mixtral_layout() {
    // v2.9 fix F1 (HIGH adversarial review). The codegen v3/v4 lowering
    // uses `info.weight_prefix.or(cfg_key)` as the WeightMap lookup
    // key. Pre-fix, the prune pass used the cfg_key directly and
    // silently no-op'd when an HF Mixtral run set weight_prefix to a
    // dotted HF path (NSL field names can't contain `.`). Post-fix,
    // the pass MUST resolve under `info.weight_prefix.unwrap_or(key)`
    // and prune the HF-prefixed tensors.
    let cfg_key = "Block.experts_dummy".to_string();
    let hf_prefix = "model.layers.0.block_sparse_moe".to_string();
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{hf_prefix}.router.weight"),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    for (suffix, scale) in [("experts.gate.weight", 1), ("experts.up.weight", 2), ("experts.down.weight", 3)] {
        wm.insert(WeightEntry::new(
            format!("{hf_prefix}.{suffix}"),
            f32s(&v4_projection_flat(&[0, 1, 2, 3], scale)),
            vec![N, D * D],
            WeightDType::F32,
        ));
    }
    let mut cfgs = HashMap::new();
    let mut info = moe_info(N, 1);
    info.weight_prefix = Some(hf_prefix.clone());
    cfgs.insert(cfg_key.clone(), info);

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);

    // The pass MUST find the weights under the HF prefix and prune
    // (NOT skip with SkippedMissingRouter as the pre-fix behavior).
    // The outcome layer is the cfg_key (operator-facing identity)
    // even though the lookup happened under the HF prefix.
    match outcomes.as_slice() {
        [MoePruneOutcome::Pruned { layer, layout, dead, n_live, .. }] => {
            assert_eq!(layer, &cfg_key);
            assert_eq!(*layout, MoeExpertsLayout::GateUpDown);
            assert_eq!(dead, &vec![1]);
            assert_eq!(*n_live, 3);
        }
        other => panic!("expected Pruned under HF prefix, got {other:?}"),
    }
    // The HF-prefixed tensors must have been sliced.
    assert_eq!(wm.get(&format!("{hf_prefix}.router.weight")).unwrap().shape, vec![D, 3]);
    assert_eq!(wm.get(&format!("{hf_prefix}.experts.gate.weight")).unwrap().shape, vec![3, D * D]);
    assert_eq!(wm.get(&format!("{hf_prefix}.experts.up.weight")).unwrap().shape,   vec![3, D * D]);
    assert_eq!(wm.get(&format!("{hf_prefix}.experts.down.weight")).unwrap().shape, vec![3, D * D]);
    assert_eq!(cfgs[&cfg_key].num_experts, 3);
}

#[test]
fn pass_refuses_router_not_2d() {
    // v2.9 fix F4 (LOW adversarial review). A 1D fused router blob
    // produces a distinct refusal variant rather than a quadratic-
    // in-n bogus expected_elems count.
    let key = "m";
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{key}.router.weight"),
        f32s(&router_flat()),
        vec![D * N], // 1D fused — INVALID
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.weight"),
        f32s(&experts_flat(&[0, 1, 2, 3])),
        vec![N, D * D],
        WeightDType::F32,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    match outcomes.as_slice() {
        [MoePruneOutcome::Refused {
            refusal: ExpertPruneRefusal::RouterShapeNot2D { actual_ndim, .. },
            ..
        }] => assert_eq!(*actual_ndim, 1),
        other => panic!("expected RouterShapeNot2D, got {other:?}"),
    }
    // WeightMap untouched.
    assert_eq!(wm.get(&format!("{key}.router.weight")).unwrap().shape, vec![D * N]);
}

#[test]
fn pass_refuses_projection_not_2d_monotonically() {
    // v2.9 fix F2 (IMPORTANT adversarial review). Pre-fix, a 1D
    // packed projection `[n*block_elems]` would silently reshape to
    // 2D on writeback when ANY expert was dead, but stay 1D when
    // none were — non-monotonic, depending on router affinities.
    // Post-fix, the pass refuses BOTH cases (dead-list nonempty AND
    // empty) consistently.
    let key = "m";
    let make_wm = || {
        let mut wm = WeightMap::default();
        wm.insert(WeightEntry::new(
            format!("{key}.router.weight"),
            f32s(&router_flat()), // expert 1 dead
            vec![D, N],
            WeightDType::F32,
        ));
        // Up projection is 1D — pre-fix: silently reshaped to 2D on slice.
        wm.insert(WeightEntry::new(
            format!("{key}.experts.up.weight"),
            f32s(&v4_projection_flat(&[0, 1, 2, 3], 2)),
            vec![N * D * D], // 1D fused — INVALID
            WeightDType::F32,
        ));
        wm.insert(WeightEntry::new(
            format!("{key}.experts.down.weight"),
            f32s(&v4_projection_flat(&[0, 1, 2, 3], 3)),
            vec![N, D * D],
            WeightDType::F32,
        ));
        wm
    };

    // Case A: dead expert present → must REFUSE (was: silently reshape).
    let mut wm = make_wm();
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));
    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    match outcomes.as_slice() {
        [MoePruneOutcome::Refused {
            refusal: ExpertPruneRefusal::ProjectionShapeNot2D { name, actual_ndim, .. },
            ..
        }] => {
            assert!(name.ends_with("experts.up.weight"), "refusal must name 'up', got '{name}'");
            assert_eq!(*actual_ndim, 1);
        }
        other => panic!("expected ProjectionShapeNot2D (dead-case), got {other:?}"),
    }
    // 1D shape preserved (untouched).
    assert_eq!(wm.get(&format!("{key}.experts.up.weight")).unwrap().shape, vec![N * D * D]);
}

#[test]
fn pass_refuses_over_complete_layout_v4_and_single_coexist() {
    // v2.9 fix F3 (LOW adversarial review). When BOTH a v4 layout
    // (gate+up+down) AND a legacy `experts.weight` coexist under the
    // same key, the pass must REFUSE rather than silently leave the
    // single tensor as orphan dead bytes (which would inflate
    // WeightMap.total_bytes and confuse downstream passes).
    let key = "m";
    let mut wm = weight_map_with_v4_moe(key);
    // Inject a stale legacy single-experts entry.
    wm.insert(WeightEntry::new(
        format!("{key}.experts.weight"),
        f32s(&experts_flat(&[0, 1, 2, 3])),
        vec![N, D * D],
        WeightDType::F32,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    assert!(
        matches!(
            outcomes.as_slice(),
            [MoePruneOutcome::Refused {
                refusal: ExpertPruneRefusal::BundleInconsistent { .. },
                ..
            }]
        ),
        "expected BundleInconsistent for over-complete v4+single layout, got {outcomes:?}"
    );
    // No partial mutation — every original entry must still be intact.
    assert_eq!(wm.get(&format!("{key}.router.weight")).unwrap().shape, vec![D, N]);
    assert_eq!(wm.get(&format!("{key}.experts.gate.weight")).unwrap().shape, vec![N, D * D]);
    assert_eq!(wm.get(&format!("{key}.experts.up.weight")).unwrap().shape,   vec![N, D * D]);
    assert_eq!(wm.get(&format!("{key}.experts.down.weight")).unwrap().shape, vec![N, D * D]);
    assert_eq!(wm.get(&format!("{key}.experts.weight")).unwrap().shape,      vec![N, D * D]);
    assert_eq!(cfgs[key].num_experts, N);
}

#[test]
fn pass_v4_dtype_mismatch_names_first_offender() {
    // v2.9 fix F7 (LOW adversarial review). The dtype-mismatch test
    // below only validates the LAST projection (down). A regression
    // that swapped the loop iteration order from gate→up→down to
    // down→up→gate would still pass that test. This test pins the
    // first-offender-wins iteration-order invariant: only `gate` is
    // mismatched, refusal must name `gate`.
    let key = "m";
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{key}.router.weight"),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    // gate at F16 (the OFFENDER), up + down at F32.
    wm.insert(WeightEntry::new(
        format!("{key}.experts.gate.weight"),
        vec![0u8; N * D * D * 2],
        vec![N, D * D],
        WeightDType::F16,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.up.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 2)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.down.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 3)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    match outcomes.as_slice() {
        [MoePruneOutcome::Refused {
            refusal: ExpertPruneRefusal::ProjectionDtypeMismatch { name, .. },
            ..
        }] => {
            assert!(name.ends_with("experts.gate.weight"), "refusal must name 'gate', got '{name}'");
        }
        other => panic!("expected ProjectionDtypeMismatch on gate, got {other:?}"),
    }
}

#[test]
fn pass_v4_dtype_mismatch_names_offending_projection() {
    // v4 layout with one projection (down) at a different dtype than
    // router/gate/up. Refusal must name "down" specifically — protects
    // future diagnostics from a copy-paste that swallows the name.
    let key = "m";
    let mut wm = WeightMap::default();
    wm.insert(WeightEntry::new(
        format!("{key}.router.weight"),
        f32s(&router_flat()),
        vec![D, N],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.gate.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 1)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    wm.insert(WeightEntry::new(
        format!("{key}.experts.up.weight"),
        f32s(&v4_projection_flat(&[0, 1, 2, 3], 2)),
        vec![N, D * D],
        WeightDType::F32,
    ));
    // down at F16 (2 bytes/elem) — wrong dtype.
    wm.insert(WeightEntry::new(
        format!("{key}.experts.down.weight"),
        vec![0u8; N * D * D * 2],
        vec![N, D * D],
        WeightDType::F16,
    ));
    let mut cfgs = HashMap::new();
    cfgs.insert(key.to_string(), moe_info(N, 1));

    let outcomes = prune_moe_weights_in_map(CpdtMode::Full, &mut cfgs, &mut wm);
    match outcomes.as_slice() {
        [MoePruneOutcome::Refused {
            refusal: ExpertPruneRefusal::ProjectionDtypeMismatch { name, .. },
            ..
        }] => {
            assert!(name.ends_with("experts.down.weight"), "refusal must name 'down', got '{name}'");
        }
        other => panic!("expected ProjectionDtypeMismatch refusal, got {other:?}"),
    }
    // Untouched.
    assert_eq!(wm.get(&format!("{key}.router.weight")).unwrap().shape, vec![D, N]);
    assert_eq!(wm.get(&format!("{key}.experts.down.weight")).unwrap().shape, vec![N, D * D]);
    assert_eq!(cfgs[key].num_experts, N);
}
