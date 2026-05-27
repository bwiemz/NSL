//! CPDT Part III — MoE dead-expert pruning (v1).
//!
//! Executes the dead-expert decision as a real weight-drop transform: slice the
//! router columns + drop the expert blocks for low-affinity experts, driven by
//! an `index_remap` single source of truth. A non-WGGO compile pass makes it
//! reachable with `--cpdt --weights` (no `--wggo`), sidestepping Part II's
//! WGGO/source-AD activation blocker. See
//! `docs/superpowers/specs/2026-05-27-cpdt-moe-dead-expert-pruning-v1-design.md`.

use crate::weight_aware::WeightDType;

/// Raw-byte MoE weight bundle for pruning. dtype-agnostic — slicing is by byte.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeWeightBundle {
    /// Router weights `[d_model, n_experts]` row-major (expert `e` = column `e`).
    pub router: Vec<u8>,
    /// Expert weights `[n_experts, expert_block_elems]` row-major (packed).
    pub experts: Vec<u8>,
    pub d_model: usize,
    pub n_experts: usize,
    /// Elements per expert block (e.g. hidden * intermediate).
    pub expert_block_elems: usize,
    pub dtype: WeightDType,
    pub top_k: usize,
}

/// Result of a successful prune. All three outputs derive from `index_remap`.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneResult {
    pub sliced_router: Vec<u8>,
    pub kept_experts: Vec<u8>,
    /// new contiguous slot `j` -> original expert id (ascending). Single source of truth.
    pub index_remap: Vec<u32>,
    pub n_live: usize,
    /// dead expert ids, sorted unique.
    pub dead_experts: Vec<u32>,
}

/// Precondition failures. On any refusal, nothing is produced (input untouched).
#[derive(Debug, Clone, PartialEq)]
pub enum ExpertPruneRefusal {
    RouterShapeMismatch { expected_elems: usize, actual_elems: usize },
    BundleInconsistent { reason: String },
    DeadIndexOutOfRange { index: u32, n_experts: usize },
    AllExpertsDead,
    InsufficientLiveExperts { n_live: usize, top_k: usize },
    /// v1 byte-slices router + experts with a single byte-width, so the two
    /// tensors must share a dtype. Mixed dtypes are refused (not mis-sliced);
    /// matched-dtype support is the common case and a separate-byte-width
    /// bundle is a documented follow-on. Produced only by the pass (the pure
    /// transform carries one dtype).
    MixedDtypeUnsupported { router_dtype: WeightDType, expert_dtype: WeightDType },
}

/// Prune dead experts from a MoE weight bundle: slice router columns + drop
/// expert blocks, both keyed by the `index_remap` single source of truth.
pub fn prune_dead_experts(
    bundle: &MoeWeightBundle,
    dead_experts: &[u32],
) -> Result<PruneResult, ExpertPruneRefusal> {
    let bw = bundle.dtype.byte_width();
    let n = bundle.n_experts;

    // Bundle consistency (byte-length checks).
    let expected_router = bundle.d_model * n * bw;
    if bundle.router.len() != expected_router {
        return Err(ExpertPruneRefusal::RouterShapeMismatch {
            expected_elems: bundle.d_model * n,
            actual_elems: bundle.router.len() / bw.max(1),
        });
    }
    let expected_experts = n * bundle.expert_block_elems * bw;
    if bundle.experts.len() != expected_experts {
        return Err(ExpertPruneRefusal::BundleInconsistent {
            reason: format!(
                "experts {} bytes != n_experts {} * block_elems {} * bw {}",
                bundle.experts.len(),
                n,
                bundle.expert_block_elems,
                bw
            ),
        });
    }

    // Dead-index range check.
    for &d in dead_experts {
        if d as usize >= n {
            return Err(ExpertPruneRefusal::DeadIndexOutOfRange { index: d, n_experts: n });
        }
    }

    // index_remap: survivors in ascending order (the single source of truth).
    let dead_set: std::collections::BTreeSet<u32> = dead_experts.iter().copied().collect();
    let index_remap: Vec<u32> = (0..n as u32).filter(|e| !dead_set.contains(e)).collect();
    let n_live = index_remap.len();
    if n_live == 0 {
        return Err(ExpertPruneRefusal::AllExpertsDead);
    }
    if n_live < bundle.top_k {
        return Err(ExpertPruneRefusal::InsufficientLiveExperts { n_live, top_k: bundle.top_k });
    }

    // Slice router: [d_model, n] -> [d_model, n_live], keep columns index_remap.
    let mut sliced_router = vec![0u8; bundle.d_model * n_live * bw];
    for r in 0..bundle.d_model {
        for (j, &orig) in index_remap.iter().enumerate() {
            let src = (r * n + orig as usize) * bw;
            let dst = (r * n_live + j) * bw;
            sliced_router[dst..dst + bw].copy_from_slice(&bundle.router[src..src + bw]);
        }
    }

    // Drop expert blocks: keep block index_remap[j].
    let blk = bundle.expert_block_elems * bw;
    let mut kept_experts = vec![0u8; n_live * blk];
    for (j, &orig) in index_remap.iter().enumerate() {
        let src = orig as usize * blk;
        let dst = j * blk;
        kept_experts[dst..dst + blk].copy_from_slice(&bundle.experts[src..src + blk]);
    }

    Ok(PruneResult {
        sliced_router,
        kept_experts,
        index_remap,
        n_live,
        dead_experts: dead_set.into_iter().collect(),
    })
}

use crate::weight_aware::{WeightEntry, WeightMap};
use std::collections::HashMap;

/// Per-MoE outcome of the prune pass (for reporting + tests).
#[derive(Debug, Clone, PartialEq)]
pub enum MoePruneOutcome {
    Pruned { layer: String, dead: Vec<u32>, n_live: usize, dropped_bytes: u64 },
    NoDeadExperts { layer: String },
    SkippedMissingRouter { layer: String },
    SkippedMissingExperts { layer: String },
    Refused { layer: String, refusal: ExpertPruneRefusal },
}

/// Affinity threshold below which an expert is pruned (mirrors
/// `ExpertConfig::default().dead_expert_threshold`).
const DEAD_EXPERT_THRESHOLD: f64 = 0.01;

/// Resolve a `WeightMap` entry by trying candidate name suffixes under `key`.
fn resolve<'a>(wm: &'a WeightMap, key: &str, suffixes: &[&str]) -> Option<&'a WeightEntry> {
    suffixes.iter().find_map(|s| wm.get(&format!("{key}.{s}")))
}

/// Core of the non-WGGO MoE dead-expert prune pass. Mutates `weight_map`
/// (slices router + experts in place) and `moe_configs` (threads `n_live` into
/// `num_experts` for forward-looking consistency). Returns per-MoE outcomes.
///
/// No-op unless `cpdt_mode == Full`. Opportunistic: missing router/experts ->
/// skip-with-outcome; found-but-inconsistent -> refusal-with-outcome; neither
/// mutates the map. This is the blocker-sidestep call: it depends only on the
/// router weights + the pure detection functions, never on WGGO/source-AD.
pub fn prune_moe_weights_in_map(
    cpdt_mode: crate::cpdt::CpdtMode,
    moe_configs: &mut HashMap<String, crate::moe::MoeInfo>,
    weight_map: &mut WeightMap,
) -> Vec<MoePruneOutcome> {
    use crate::cpdt::CpdtMode;
    if cpdt_mode != CpdtMode::Full {
        return Vec::new();
    }

    // Deterministic order over MoE layers.
    let mut keys: Vec<String> = moe_configs.keys().cloned().collect();
    keys.sort();

    let mut outcomes = Vec::new();
    for key in keys {
        let info = moe_configs[&key].clone();
        let n = info.num_experts;

        // Resolve router (drives detection + slice) and packed experts.
        let router_entry = match resolve(weight_map, &key, &["router.weight", "gate.weight", "router", "gate"]) {
            Some(e) => e.clone(),
            None => {
                outcomes.push(MoePruneOutcome::SkippedMissingRouter { layer: key });
                continue;
            }
        };
        let expert_name = ["experts.weight", "experts"]
            .iter()
            .map(|s| format!("{key}.{s}"))
            .find(|nm| weight_map.get(nm).is_some());
        let expert_name = match expert_name {
            Some(nm) => nm,
            None => {
                outcomes.push(MoePruneOutcome::SkippedMissingExperts { layer: key });
                continue;
            }
        };
        let expert_entry = weight_map.get(&expert_name).unwrap().clone();
        let router_name = router_entry.name.clone();

        // Found-but-inconsistent router shape -> refusal.
        if router_entry.shape.len() != 2 || router_entry.shape[1] != n {
            outcomes.push(MoePruneOutcome::Refused {
                layer: key,
                refusal: ExpertPruneRefusal::RouterShapeMismatch {
                    expected_elems: router_entry.shape.first().copied().unwrap_or(0) * n,
                    actual_elems: router_entry.num_elements,
                },
            });
            continue;
        }
        let d_model = router_entry.shape[0];

        // v1 byte-slices with a single byte-width -> router and experts must
        // share a dtype. Refuse mismatches rather than mis-slice the experts.
        if expert_entry.dtype != router_entry.dtype {
            outcomes.push(MoePruneOutcome::Refused {
                layer: key,
                refusal: ExpertPruneRefusal::MixedDtypeUnsupported {
                    router_dtype: router_entry.dtype,
                    expert_dtype: expert_entry.dtype,
                },
            });
            continue;
        }
        // Expert tensor must pack evenly into n blocks (clearer diagnostic than
        // the downstream byte-length refusal).
        if !expert_entry.num_elements.is_multiple_of(n.max(1)) {
            outcomes.push(MoePruneOutcome::Refused {
                layer: key,
                refusal: ExpertPruneRefusal::BundleInconsistent {
                    reason: format!(
                        "expert num_elements {} not divisible by n_experts {n}",
                        expert_entry.num_elements
                    ),
                },
            });
            continue;
        }
        let block_elems = expert_entry.num_elements / n.max(1);

        // Detect dead experts from the router weights alone.
        let affinities = crate::cpdt_expert::router_affinities(&router_entry, n as u32);
        let dead = crate::cpdt_expert::detect_dead_experts(&affinities, DEAD_EXPERT_THRESHOLD);

        let bundle = MoeWeightBundle {
            router: router_entry.data.clone(),
            experts: expert_entry.data.clone(),
            d_model,
            n_experts: n,
            expert_block_elems: block_elems,
            dtype: router_entry.dtype,
            top_k: info.top_k,
        };

        match prune_dead_experts(&bundle, &dead) {
            Err(refusal) => outcomes.push(MoePruneOutcome::Refused { layer: key, refusal }),
            Ok(res) if res.dead_experts.is_empty() => {
                outcomes.push(MoePruneOutcome::NoDeadExperts { layer: key });
            }
            Ok(res) => {
                let dropped_bytes = router_entry.data.len().saturating_sub(res.sliced_router.len()) as u64
                    + expert_entry.data.len().saturating_sub(res.kept_experts.len()) as u64;
                // Mutate the WeightMap entries in place (smaller bundle).
                let new_router = WeightEntry::new(
                    router_name.clone(),
                    res.sliced_router,
                    vec![d_model, res.n_live],
                    router_entry.dtype,
                );
                let new_experts = WeightEntry::new(
                    expert_name.clone(),
                    res.kept_experts,
                    vec![res.n_live, block_elems],
                    expert_entry.dtype,
                );
                weight_map.insert(new_router);
                weight_map.insert(new_experts);
                // Thread n_live into the config (forward-looking consistency).
                if let Some(cfg) = moe_configs.get_mut(&key) {
                    cfg.num_experts = res.n_live;
                }
                outcomes.push(MoePruneOutcome::Pruned {
                    layer: key,
                    dead: res.dead_experts,
                    n_live: res.n_live,
                    dropped_bytes,
                });
            }
        }
    }
    outcomes
}

/// Render a one-line report for each outcome (stderr).
pub fn report_outcomes(outcomes: &[MoePruneOutcome]) {
    for o in outcomes {
        match o {
            MoePruneOutcome::Pruned { layer, dead, n_live, dropped_bytes } => eprintln!(
                "[cpdt] moe '{layer}': pruned experts {dead:?} (affinity < {DEAD_EXPERT_THRESHOLD}) \
                 -> n_live={n_live}, dropped {dropped_bytes} bytes"
            ),
            MoePruneOutcome::NoDeadExperts { layer } => {
                eprintln!("[cpdt] moe '{layer}': no dead experts, nothing pruned")
            }
            MoePruneOutcome::SkippedMissingRouter { layer } => {
                eprintln!("[cpdt] moe '{layer}': skipped — no router weight found")
            }
            MoePruneOutcome::SkippedMissingExperts { layer } => {
                eprintln!("[cpdt] moe '{layer}': skipped — no expert weights found")
            }
            MoePruneOutcome::Refused { layer, refusal } => {
                eprintln!("[cpdt] moe '{layer}': prune refused — {refusal:?}")
            }
        }
    }
}

/// Compile-flow entry point for the MoE dead-expert prune pass.
///
/// STRUCTURAL GUARANTEE (blocker-sidestep): this is invoked directly from
/// `compile_returning_plan`, NOT from `stmt.rs::invoke_cpdt_if_enabled` (which
/// is gated on `wggo_applied`, source-AD only). It depends only on
/// `compiler.cpdt_mode` + `compiler.features.{moe_configs, weight_map}` — never
/// on `wggo_applied`. Keep this call site out of any `wggo_applied` guard so
/// the pass stays reachable with `--cpdt --weights` (no `--wggo`).
pub fn run_moe_prune_pass(compiler: &mut crate::compiler::Compiler) {
    // `cpdt_mode` is `Copy` — read it before borrowing `features` so the single
    // `&mut features` split-borrow below doesn't conflict with reading it.
    let cpdt_mode = compiler.cpdt_mode;
    let features = &mut compiler.features;
    let Some(weight_map) = features.weight_map.as_mut() else {
        return;
    };
    if features.moe_configs.is_empty() {
        return;
    }
    let outcomes = prune_moe_weights_in_map(cpdt_mode, &mut features.moe_configs, weight_map);
    report_outcomes(&outcomes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::WeightDType;

    /// Build little-endian f32 bytes.
    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// d_model=2, n_experts=4, 1 elem/expert. Distinct router columns + expert blocks.
    fn toy_bundle() -> MoeWeightBundle {
        MoeWeightBundle {
            router: f32s(&[
                10.0, 20.0, 30.0, 40.0, // row 0: e0 e1 e2 e3
                11.0, 21.0, 31.0, 41.0, // row 1
            ]),
            experts: f32s(&[100.0, 200.0, 300.0, 400.0]), // 4 experts × 1 elem
            d_model: 2,
            n_experts: 4,
            expert_block_elems: 1,
            dtype: WeightDType::F32,
            top_k: 1,
        }
    }

    #[test]
    fn prune_by_identity() {
        let b = toy_bundle();
        let res = prune_dead_experts(&b, &[1]).unwrap();
        assert_eq!(res.index_remap, vec![0, 2, 3]);
        assert_eq!(res.n_live, 3);
        assert_eq!(res.dead_experts, vec![1]);
    }

    #[test]
    fn internal_consistency() {
        let b = toy_bundle();
        let res = prune_dead_experts(&b, &[1]).unwrap();
        let bw = 4usize;
        // Router: j-th live column bytes == original column index_remap[j] bytes.
        for r in 0..b.d_model {
            for (j, &orig) in res.index_remap.iter().enumerate() {
                let got = &res.sliced_router[(r * res.n_live + j) * bw..(r * res.n_live + j) * bw + bw];
                let want = &b.router[(r * b.n_experts + orig as usize) * bw..(r * b.n_experts + orig as usize) * bw + bw];
                assert_eq!(got, want, "router col j={j} (orig {orig}) row {r}");
            }
        }
        // Experts: block j bytes == original block index_remap[j] bytes.
        let blk = b.expert_block_elems * bw;
        for (j, &orig) in res.index_remap.iter().enumerate() {
            let got = &res.kept_experts[j * blk..j * blk + blk];
            let want = &b.experts[orig as usize * blk..orig as usize * blk + blk];
            assert_eq!(got, want, "expert block j={j} (orig {orig})");
        }
    }

    #[test]
    fn no_dead_is_identity() {
        let b = toy_bundle();
        let res = prune_dead_experts(&b, &[]).unwrap();
        assert_eq!(res.index_remap, vec![0, 1, 2, 3]);
        assert_eq!(res.n_live, 4);
        assert_eq!(res.sliced_router, b.router);
        assert_eq!(res.kept_experts, b.experts);
    }

    #[test]
    fn refuse_router_shape() {
        let mut b = toy_bundle();
        b.router.truncate(b.router.len() - 4); // wrong byte length
        assert!(matches!(
            prune_dead_experts(&b, &[1]),
            Err(ExpertPruneRefusal::RouterShapeMismatch { .. })
        ));
    }

    #[test]
    fn refuse_bundle_inconsistent() {
        let mut b = toy_bundle();
        b.experts.truncate(b.experts.len() - 4); // experts bytes != n*block*bw
        assert!(matches!(
            prune_dead_experts(&b, &[1]),
            Err(ExpertPruneRefusal::BundleInconsistent { .. })
        ));
    }

    #[test]
    fn refuse_dead_oob() {
        let b = toy_bundle();
        assert_eq!(
            prune_dead_experts(&b, &[5]),
            Err(ExpertPruneRefusal::DeadIndexOutOfRange { index: 5, n_experts: 4 })
        );
    }

    #[test]
    fn refuse_all_dead() {
        let b = toy_bundle();
        assert_eq!(prune_dead_experts(&b, &[0, 1, 2, 3]), Err(ExpertPruneRefusal::AllExpertsDead));
    }

    #[test]
    fn refuse_insufficient_live() {
        let mut b = toy_bundle();
        b.top_k = 2;
        // 4 experts, dead [0,1,3] -> n_live=1 < top_k=2.
        assert_eq!(
            prune_dead_experts(&b, &[0, 1, 3]),
            Err(ExpertPruneRefusal::InsufficientLiveExperts { n_live: 1, top_k: 2 })
        );
    }

    #[test]
    fn refusal_mutates_nothing() {
        // The transform takes &bundle and returns owned data, so the input is
        // structurally immutable; assert it explicitly (clone-and-compare) so a
        // future &mut refactor that partially mutates before failing is caught.
        let b = toy_bundle();
        let snapshot = b.clone();
        let _ = prune_dead_experts(&b, &[5]); // refusal
        assert_eq!(b, snapshot);
    }
}
