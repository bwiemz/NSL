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
    assert!(matches!(
        outcomes.as_slice(),
        [MoePruneOutcome::Refused { refusal: ExpertPruneRefusal::MixedDtypeUnsupported { .. }, .. }]
    ));
    // Untouched: no partial mutation on refusal.
    assert_eq!(wm.get("m.router.weight").unwrap().shape, vec![D, N]);
    assert_eq!(wm.get("m.experts.weight").unwrap().shape, vec![N, D * D]);
    assert_eq!(cfgs["m"].num_experts, N);
}
