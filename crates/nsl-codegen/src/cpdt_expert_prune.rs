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

/// CPDT Part III v2.9 — one per-expert projection in a multi-projection
/// MoE FFN (v3 = up+down, v4 = gate+up+down). Each projection is packed
/// `[n_experts, block_elems]` row-major in its own byte buffer; the
/// block_elems may differ across projections within the same MoE
/// (e.g. for v4 with hidden=8 / intermediate=16: gate and up have
/// 8*16=128 elems/block, down has 16*8=128 elems/block — equal here but
/// asymmetric configs are common).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertProjection {
    /// Human-readable name for diagnostics + WeightMap key on writeback
    /// (e.g., "experts.gate.weight", "experts.up.weight", "experts.down.weight").
    pub name: String,
    pub data: Vec<u8>,
    /// Elements per expert block for THIS projection.
    pub block_elems: usize,
    pub dtype: WeightDType,
}

/// CPDT Part III v2.9 — sliced output of one input projection. Same
/// length as the input `Vec<ExpertProjection>` passed to
/// `prune_dead_experts_split`, in the same order.
#[derive(Debug, Clone, PartialEq)]
pub struct KeptProjection {
    pub name: String,
    pub data: Vec<u8>,
    pub block_elems: usize,
    pub dtype: WeightDType,
}

/// CPDT Part III v2.9 — result of a successful multi-projection prune.
/// `sliced_router` mirrors v1 `PruneResult.sliced_router`; the per-
/// projection slices are returned in the same order as the input
/// projections so callers can pair input.name with output.name.
#[derive(Debug, Clone, PartialEq)]
pub struct SplitPruneResult {
    pub sliced_router: Vec<u8>,
    pub kept_projections: Vec<KeptProjection>,
    pub index_remap: Vec<u32>,
    pub n_live: usize,
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
    /// v2.9: `prune_dead_experts_split` requires at least one projection.
    /// An empty list is a programmer error (the caller should not have
    /// reached the split path) — refuse loudly rather than no-op.
    EmptyProjectionList,
    /// v2.9: one of the projections in the split-path has a byte length
    /// that doesn't equal `n_experts * block_elems * dtype.byte_width()`.
    /// Reported per-projection so the caller can identify which one.
    ProjectionShapeMismatch { name: String, expected_bytes: usize, actual_bytes: usize },
    /// v2.9: split-path projections must share dtype with the router.
    /// Mixed dtypes across projections are rejected here for the same
    /// reason as the legacy MixedDtypeUnsupported above (one byte-width
    /// drives the byte slice).
    ProjectionDtypeMismatch { name: String, projection_dtype: WeightDType, router_dtype: WeightDType },
    /// v2.9 fix F4 (LOW adversarial review): the router tensor's shape
    /// is not 2D. Distinct from `RouterShapeMismatch` because the
    /// "expected vs actual elems" diagnostic is meaningless when the
    /// tensor isn't 2D at all (a 1D fused router would produce a
    /// quadratic-in-n expected count). Reported with the actual ndim
    /// so the user can correct the upstream packing.
    RouterShapeNot2D { actual_ndim: usize, num_elements: usize },
    /// v2.9 fix F2 (IMPORTANT adversarial review): a projection tensor's
    /// shape is not 2D `[n_experts, block_elems]`. Without this gate
    /// the pass would silently reshape a 1D `[n*block_elems]` packed
    /// buffer into 2D on writeback (non-monotonic: only changes shape
    /// when at least one expert is dead). Refusal mirrors the router
    /// 2D check and names the offending projection.
    ProjectionShapeNot2D { name: String, actual_ndim: usize, num_elements: usize },
    /// CPDT Part III v2.13 — partial-bias bundle. The detected layout
    /// (UpDown for v3, GateUpDown for v4) has at least one bias entry
    /// present but at least one missing. Mirrors v2.12's `detect_v3_
    /// biases` partial-bias refusal: biases are all-or-nothing per
    /// layout family. If only some are present, the bundle is almost
    /// certainly the result of a stale or mis-packed save and silently
    /// pruning the present biases (while leaving the absent ones at
    /// the old `n_experts` row count) would produce silent corruption
    /// at the v2.12 codegen detection step (mismatched element count
    /// after the writeback). Refuse loudly with the present / missing
    /// names so the user can fix the bundle.
    ///
    /// v2.13 fix F11: dropped the redundant `layer` field — it
    /// duplicated `MoePruneOutcome::Refused.layer` with no consumer.
    PartialBiasBundle { present: Vec<String>, missing: Vec<String> },
    /// CPDT Part III v2.13 fix F1 (HIGH adversarial review) — bias
    /// entry has rank > 2. The slicing logic admits 1D `[n*dim]` and
    /// 2D `[n, dim]` layouts only; rank-3+ writeback can't pick a
    /// canonical shape without per-axis semantics that v2.13 doesn't
    /// model. Refuse loudly rather than silently flatten to 1D
    /// (which would re-introduce the non-monotonic shape-rewrite
    /// hazard that v2.9 fix F2 closed on the weight side).
    BiasShapeRankUnsupported { name: String, actual_ndim: usize, num_elements: usize },
    /// CPDT Part III v2.13 fix F2 (HIGH adversarial review) — the
    /// detected layout is Single (v1/v2 legacy packed format) but
    /// bias entries are present. Single does not support FFN biases
    /// (the legacy single-projection convention has no bias hook).
    /// Without this gate the prune would slice the single weight to
    /// `n_live` rows while leaving the orphan biases at the original
    /// `n_experts` count — silently inflating WeightMap.total_bytes()
    /// and confusing downstream passes that scan by total weight
    /// (mirrors the v2.9 F3 over-complete-layout invariant for the
    /// bias case).
    SingleLayoutWithBiasEntries { layer: String, orphan_biases_present: bool },
    /// CPDT Part III v2.13 fix F15 (IMPORTANT adversarial review) —
    /// a v3/v4 bias entry is present but the corresponding weight
    /// projection is absent (e.g., `experts.up.bias` shipped without
    /// `experts.up.weight`). Layout detection only inspects weight
    /// keys, so an orphan bias bundle silently falls into
    /// `SkippedMissingExperts` while leaving the orphan bias tensors
    /// in the WeightMap. v2.13 surfaces this loudly so downstream
    /// passes don't see polluted state.
    OrphanBiasWithoutWeight { layer: String, orphan_biases: Vec<String> },
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

/// CPDT Part III v2.9 — multi-projection dead-expert prune.
///
/// Slices `router` columns + each of `projections`'s expert blocks, all
/// keyed by the SAME `index_remap` single source of truth. v3 layout
/// passes 2 projections (`experts.{up,down}.weight`); v4 layout passes
/// 3 (`experts.{gate,up,down}.weight`). v1/v2 single-projection callers
/// should continue using `prune_dead_experts` — this function is the
/// generalized form.
///
/// Per-projection guarantee: each projection is sliced INDEPENDENTLY
/// using the same `index_remap`, so a v4 caller with mismatched
/// gate/up/down block_elems still gets consistent expert-id ordering
/// across all three outputs.
///
/// Refusals (no partial mutation — split-path is structurally pure;
/// inputs are taken by `&` and outputs are owned):
///   - `EmptyProjectionList` — `projections.is_empty()`
///   - `RouterShapeMismatch` — `router.len() != d_model * n_experts * bw`
///   - `ProjectionShapeMismatch` — projection.data.len() doesn't match
///     `n_experts * projection.block_elems * bw`
///   - `ProjectionDtypeMismatch` — projection.dtype != router_dtype
///   - `DeadIndexOutOfRange` — any dead index >= n_experts
///   - `AllExpertsDead` — every expert is in dead_experts
///   - `InsufficientLiveExperts` — n_live < top_k
pub fn prune_dead_experts_split(
    router: &[u8],
    router_dtype: WeightDType,
    d_model: usize,
    n_experts: usize,
    top_k: usize,
    projections: &[ExpertProjection],
    dead_experts: &[u32],
) -> Result<SplitPruneResult, ExpertPruneRefusal> {
    if projections.is_empty() {
        return Err(ExpertPruneRefusal::EmptyProjectionList);
    }
    let bw = router_dtype.byte_width();
    let n = n_experts;

    // Router byte-length check.
    let expected_router = d_model * n * bw;
    if router.len() != expected_router {
        return Err(ExpertPruneRefusal::RouterShapeMismatch {
            expected_elems: d_model * n,
            actual_elems: router.len() / bw.max(1),
        });
    }

    // Per-projection consistency: dtype + byte length. Validate ALL
    // projections BEFORE any allocation so a downstream failure can't
    // leave the caller with half-sliced state. (Inputs are immutable
    // refs — there is nothing to roll back — but explicit upfront
    // validation matches v2.6's two-phase-commit convention and gives
    // clearer diagnostics than a mid-slice panic.)
    for p in projections {
        if p.dtype != router_dtype {
            return Err(ExpertPruneRefusal::ProjectionDtypeMismatch {
                name: p.name.clone(),
                projection_dtype: p.dtype,
                router_dtype,
            });
        }
        let expected = n * p.block_elems * bw;
        if p.data.len() != expected {
            return Err(ExpertPruneRefusal::ProjectionShapeMismatch {
                name: p.name.clone(),
                expected_bytes: expected,
                actual_bytes: p.data.len(),
            });
        }
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
    if n_live < top_k {
        return Err(ExpertPruneRefusal::InsufficientLiveExperts { n_live, top_k });
    }

    // Slice router: [d_model, n] -> [d_model, n_live], keep columns index_remap.
    let mut sliced_router = vec![0u8; d_model * n_live * bw];
    for r in 0..d_model {
        for (j, &orig) in index_remap.iter().enumerate() {
            let src = (r * n + orig as usize) * bw;
            let dst = (r * n_live + j) * bw;
            sliced_router[dst..dst + bw].copy_from_slice(&router[src..src + bw]);
        }
    }

    // Slice each projection: keep block index_remap[j], in order.
    let kept_projections: Vec<KeptProjection> = projections
        .iter()
        .map(|p| {
            let blk = p.block_elems * bw;
            let mut kept = vec![0u8; n_live * blk];
            for (j, &orig) in index_remap.iter().enumerate() {
                let src = orig as usize * blk;
                let dst = j * blk;
                kept[dst..dst + blk].copy_from_slice(&p.data[src..src + blk]);
            }
            KeptProjection {
                name: p.name.clone(),
                data: kept,
                block_elems: p.block_elems,
                dtype: p.dtype,
            }
        })
        .collect();

    Ok(SplitPruneResult {
        sliced_router,
        kept_projections,
        index_remap,
        n_live,
        dead_experts: dead_set.into_iter().collect(),
    })
}

use crate::weight_aware::{WeightEntry, WeightMap};
use std::collections::HashMap;

/// CPDT Part III v2.9: which experts-tensor layout the pass detected
/// for a given MoE layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExpertsLayout {
    /// v1/v2: single packed `experts.weight` tensor.
    Single,
    /// v3: `experts.up.weight` + `experts.down.weight`.
    UpDown,
    /// v4: `experts.gate.weight` + `experts.up.weight` + `experts.down.weight`.
    GateUpDown,
}

/// Per-MoE outcome of the prune pass (for reporting + tests).
#[derive(Debug, Clone, PartialEq)]
pub enum MoePruneOutcome {
    Pruned { layer: String, layout: MoeExpertsLayout, dead: Vec<u32>, n_live: usize, dropped_bytes: u64 },
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

        // CPDT Part III v2.9 fix F1 (HIGH adversarial review) — mirror the
        // v3/v4 codegen lookup-key resolution from `expr/calls.rs`. The
        // codegen v4 lowering uses `lookup_key = weight_prefix.or(cfg_key)`
        // (see expr/calls.rs ≈ 1685); the pass MUST use the same key, else
        // an HF Mixtral run (post-v2.7 auto-pack writes tensors under the
        // HF prefix, NOT the moe_configs key) silently lands in
        // `SkippedMissingRouter` and the pass no-ops. The moe_configs key
        // itself stays load-bearing for `MoePruneOutcome.layer` (operator-
        // facing) and the `moe_configs.get_mut(&key)` writeback below.
        let lookup_key: &str = info.weight_prefix.as_deref().unwrap_or(&key);

        // Resolve router (drives detection + slice).
        let router_entry = match resolve(weight_map, lookup_key, &["router.weight", "gate.weight", "router", "gate"]) {
            Some(e) => e.clone(),
            None => {
                outcomes.push(MoePruneOutcome::SkippedMissingRouter { layer: key });
                continue;
            }
        };
        let router_name = router_entry.name.clone();

        // CPDT Part III v2.9 — layout detection. Precedence: v4 (gate+up+down) > v3 (up+down) > v1/v2 (single).
        //
        // Rationale for precedence: a v4 checkpoint contains all three
        // (gate, up, down); a v3 checkpoint has only up+down (no gate);
        // v1/v2 has a single packed `experts.weight`. Picking the
        // widest layout that fully resolves means an HF Mixtral
        // checkpoint (post-v2.7 auto-pack) gets pruned across all 3
        // projections, while a legacy v1/v2 packed-single-tensor file
        // still uses the original single-tensor path. PARTIAL v4
        // matches (e.g., gate + up present but down missing) are
        // structurally invalid — refuse loudly with PartialMultiProjLayout
        // rather than silently downgrade to v3 or v1/v2. v2.9 fix F3
        // (LOW adversarial review): OVER-COMPLETE matches (multi-proj
        // + legacy single coexisting) are also refused — the legacy
        // single would persist as dead orphan bytes after the multi-
        // proj prune, inflating WeightMap.total_bytes() and confusing
        // downstream passes that scan by total weight. Refuse rather
        // than silently leave the orphan.
        let gate_present = resolve(weight_map, lookup_key, &["experts.gate.weight", "experts.gate"]).is_some();
        let up_present   = resolve(weight_map, lookup_key, &["experts.up.weight",   "experts.up"]).is_some();
        let down_present = resolve(weight_map, lookup_key, &["experts.down.weight", "experts.down"]).is_some();
        let single_present = ["experts.weight", "experts"]
            .iter()
            .any(|s| weight_map.get(&format!("{lookup_key}.{s}")).is_some());

        // v2.13 fix F2 + F15 (adversarial review): pre-check whether
        // ANY bias entries are present before falling into the layout
        // match. Two distinct hazards:
        //
        //   F2 — Single + bias entries: Single (v1/v2) has no bias
        //   support. Silently pruning the single weight while leaving
        //   the bias tensors at original n_experts inflates
        //   total_bytes and pollutes downstream passes (the same
        //   class as v2.9 fix F3's over-complete weight refusal,
        //   mirrored for the bias case).
        //
        //   F15 — orphan biases without weights: e.g. up.bias present
        //   but up.weight absent. The layout match only inspects
        //   weight keys, so this falls into SkippedMissingExperts
        //   (or the ambiguous wildcard) while orphan biases stay in
        //   the WeightMap. Surface it loudly.
        //
        // Uses `any_v3_bias_entry_present` from moe.rs — the same
        // cheap presence-only detector v2.12 added (F2 there).
        let any_bias = crate::moe::any_v3_bias_entry_present(weight_map, lookup_key);
        let bias_entries_present_names = {
            let mut found: Vec<String> = Vec::new();
            for s in [
                "experts.up.bias",
                "experts.up_bias",
                "experts.down.bias",
                "experts.down_bias",
                "experts.gate.bias",
                "experts.gate_bias",
            ] {
                let full = format!("{lookup_key}.{s}");
                if weight_map.get(&full).is_some() {
                    found.push(full);
                }
            }
            found
        };

        let layout = match (gate_present, up_present, down_present, single_present) {
            (true,  true,  true,  false) => MoeExpertsLayout::GateUpDown,
            (false, true,  true,  false) => MoeExpertsLayout::UpDown,
            (false, false, false, true) => {
                // v2.13 fix F2: Single layout + bias entries → refuse.
                if any_bias {
                    outcomes.push(MoePruneOutcome::Refused {
                        layer: key.clone(),
                        refusal: ExpertPruneRefusal::SingleLayoutWithBiasEntries {
                            layer: key.clone(),
                            orphan_biases_present: true,
                        },
                    });
                    continue;
                }
                MoeExpertsLayout::Single
            }
            (false, false, false, false) => {
                // v2.13 fix F15: orphan biases without ANY weights
                // (the genuinely "biases shipped without their
                // projections" case). Surface specifically rather
                // than silently SkippedMissingExperts.
                if any_bias {
                    outcomes.push(MoePruneOutcome::Refused {
                        layer: key.clone(),
                        refusal: ExpertPruneRefusal::OrphanBiasWithoutWeight {
                            layer: key.clone(),
                            orphan_biases: bias_entries_present_names,
                        },
                    });
                    continue;
                }
                outcomes.push(MoePruneOutcome::SkippedMissingExperts { layer: key });
                continue;
            }
            // Partial multi-projection match OR over-complete (multi-proj
            // + legacy single coexisting). Refuse loudly — silent
            // downgrade in either direction leaves orphan tensors that
            // either crash downstream FFI dispatch (partial match) or
            // misreport total weight bytes (over-complete).
            _ => {
                let reason = format!(
                    "ambiguous multi-projection MoE layout under '{lookup_key}': \
                     gate={gate_present}, up={up_present}, down={down_present}, \
                     single={single_present}. v4 needs exactly gate+up+down, v3 \
                     needs exactly up+down, v1/v2 needs exactly the single \
                     `experts.weight`. A partial or over-complete set would leave \
                     orphan tensors after slicing.",
                );
                outcomes.push(MoePruneOutcome::Refused {
                    layer: key,
                    refusal: ExpertPruneRefusal::BundleInconsistent {
                        reason,
                    },
                });
                continue;
            }
        };

        // Found-but-inconsistent router shape -> refusal. v2.9 fix F4
        // (LOW adversarial review): split into two refusals so the 1D
        // case gets a meaningful diagnostic instead of a quadratic-
        // in-n bogus "expected" count.
        if router_entry.shape.len() != 2 {
            outcomes.push(MoePruneOutcome::Refused {
                layer: key,
                refusal: ExpertPruneRefusal::RouterShapeNot2D {
                    actual_ndim: router_entry.shape.len(),
                    num_elements: router_entry.num_elements,
                },
            });
            continue;
        }
        if router_entry.shape[1] != n {
            outcomes.push(MoePruneOutcome::Refused {
                layer: key,
                refusal: ExpertPruneRefusal::RouterShapeMismatch {
                    expected_elems: router_entry.shape[0] * n,
                    actual_elems: router_entry.num_elements,
                },
            });
            continue;
        }
        let d_model = router_entry.shape[0];

        // Resolve the per-projection WeightEntries for the detected layout.
        // Order matters for the SplitPruneResult.kept_projections list
        // and for the v3/v4 writeback below.
        let projection_specs: Vec<(&[&str], &str)> = match layout {
            MoeExpertsLayout::Single => vec![
                (&["experts.weight", "experts"][..], "experts.weight"),
            ],
            MoeExpertsLayout::UpDown => vec![
                (&["experts.up.weight", "experts.up"][..],     "experts.up.weight"),
                (&["experts.down.weight", "experts.down"][..], "experts.down.weight"),
            ],
            MoeExpertsLayout::GateUpDown => vec![
                (&["experts.gate.weight", "experts.gate"][..], "experts.gate.weight"),
                (&["experts.up.weight",   "experts.up"][..],   "experts.up.weight"),
                (&["experts.down.weight", "experts.down"][..], "experts.down.weight"),
            ],
        };
        let resolved: Vec<WeightEntry> = projection_specs
            .iter()
            .map(|(suffixes, _)| resolve(weight_map, lookup_key, suffixes).unwrap().clone())
            .collect();

        // All projections must (a) be 2D `[n_experts, block_elems]`,
        // (b) share dtype with router (single byte-width slice), and
        // (c) pack evenly into n. v2.9 fix F2 (IMPORTANT adversarial
        // review) — the 2D shape gate prevents silent 1D→2D reshape
        // on writeback that would otherwise be non-monotonic (only
        // changes shape when at least one expert is dead).
        let mut refusal: Option<ExpertPruneRefusal> = None;
        let mut block_elems_vec: Vec<usize> = Vec::with_capacity(resolved.len());
        for entry in &resolved {
            if entry.dtype != router_entry.dtype {
                refusal = Some(ExpertPruneRefusal::ProjectionDtypeMismatch {
                    name: entry.name.clone(),
                    projection_dtype: entry.dtype,
                    router_dtype: router_entry.dtype,
                });
                break;
            }
            if entry.shape.len() != 2 || entry.shape[0] != n {
                refusal = Some(ExpertPruneRefusal::ProjectionShapeNot2D {
                    name: entry.name.clone(),
                    actual_ndim: entry.shape.len(),
                    num_elements: entry.num_elements,
                });
                break;
            }
            if !entry.num_elements.is_multiple_of(n.max(1)) {
                refusal = Some(ExpertPruneRefusal::BundleInconsistent {
                    reason: format!(
                        "projection '{}' num_elements {} not divisible by n_experts {n}",
                        entry.name, entry.num_elements,
                    ),
                });
                break;
            }
            block_elems_vec.push(entry.num_elements / n.max(1));
        }
        if let Some(r) = refusal {
            outcomes.push(MoePruneOutcome::Refused { layer: key, refusal: r });
            continue;
        }

        // Detect dead experts from the router weights alone.
        let affinities = crate::cpdt_expert::router_affinities(&router_entry, n as u32);
        let dead = crate::cpdt_expert::detect_dead_experts(&affinities, DEAD_EXPERT_THRESHOLD);

        let projections: Vec<ExpertProjection> = resolved
            .iter()
            .zip(block_elems_vec.iter())
            .map(|(entry, &be)| ExpertProjection {
                name: entry.name.clone(),
                data: entry.data.clone(),
                block_elems: be,
                dtype: entry.dtype,
            })
            .collect();

        // CPDT Part III v2.13 — bias detection + all-or-nothing refusal.
        //
        // The slicing operation in `prune_dead_experts_split` is byte-
        // generic: any `[n_experts, block_elems]` packed buffer gets
        // sliced by the same index_remap. Bias tensors (shape
        // `[n_experts, dim]` packed, or 1D `[n_experts * dim]` flat
        // per v2.12's accept-both convention) fit that contract
        // exactly. So instead of duplicating the slicing logic, we
        // append biases as ADDITIONAL ExpertProjections in the same
        // Vec passed to `prune_dead_experts_split`. The kept_projections
        // result is then split back into weight + bias halves for
        // separate writeback (biases preserve their original shape;
        // weights are always 2D).
        //
        // All-or-nothing rule: if ANY bias is found for the layout's
        // expected set, ALL must be present. Partial bundles are
        // refused with `PartialBiasBundle` (mirrors v2.12's
        // `detect_v3_biases` partial refusal). Rationale: silently
        // pruning the present biases while leaving the absent ones at
        // their pre-prune `n_experts` count would cause v2.12 codegen
        // detection to fail downstream with a mismatched-element
        // diagnostic — better to refuse loudly at the prune site so
        // the user can fix the bundle in one place.
        //
        // Layout scope:
        //   Single (v1/v2): no bias support (legacy single-projection)
        //   UpDown (v3): {experts.up.bias, experts.down.bias}
        //   GateUpDown (v4): {experts.gate.bias, experts.up.bias,
        //                     experts.down.bias} — defensive forward-
        //                     compat (v2.12 doesn't activate v4 bias
        //                     from source yet; v2.next bias-FFI cycle
        //                     would). Slicing keeps the bundle
        //                     self-consistent either way.
        let bias_specs: Vec<(Vec<&str>, &str)> = match layout {
            MoeExpertsLayout::Single => Vec::new(),
            MoeExpertsLayout::UpDown => vec![
                (vec!["experts.up.bias", "experts.up_bias"], "experts.up.bias"),
                (vec!["experts.down.bias", "experts.down_bias"], "experts.down.bias"),
            ],
            MoeExpertsLayout::GateUpDown => vec![
                (vec!["experts.gate.bias", "experts.gate_bias"], "experts.gate.bias"),
                (vec!["experts.up.bias", "experts.up_bias"], "experts.up.bias"),
                (vec!["experts.down.bias", "experts.down_bias"], "experts.down.bias"),
            ],
        };
        let bias_resolved: Vec<Option<WeightEntry>> = bias_specs
            .iter()
            .map(|(suffixes, _)| resolve(weight_map, lookup_key, suffixes).cloned())
            .collect();
        let bias_present_count = bias_resolved.iter().filter(|o| o.is_some()).count();
        let bias_expected_count = bias_specs.len();

        if bias_present_count > 0 && bias_present_count < bias_expected_count {
            let present_names: Vec<String> = bias_specs
                .iter()
                .zip(bias_resolved.iter())
                .filter_map(|((_, canonical), r)| r.as_ref().map(|_| canonical.to_string()))
                .collect();
            let missing_names: Vec<String> = bias_specs
                .iter()
                .zip(bias_resolved.iter())
                .filter_map(|((_, canonical), r)| {
                    if r.is_none() { Some(canonical.to_string()) } else { None }
                })
                .collect();
            outcomes.push(MoePruneOutcome::Refused {
                layer: key.clone(),
                refusal: ExpertPruneRefusal::PartialBiasBundle {
                    present: present_names,
                    missing: missing_names,
                },
            });
            continue;
        }

        // Validate bias dtype + element count, then build bias
        // ExpertProjections. Track original shapes for shape-preserving
        // writeback after slicing.
        let mut bias_projections: Vec<ExpertProjection> = Vec::new();
        let mut bias_original_shapes: Vec<Vec<usize>> = Vec::new();
        let mut bias_refusal: Option<ExpertPruneRefusal> = None;
        for entry_opt in bias_resolved.iter() {
            let Some(entry) = entry_opt else { continue };
            if entry.dtype != router_entry.dtype {
                bias_refusal = Some(ExpertPruneRefusal::ProjectionDtypeMismatch {
                    name: entry.name.clone(),
                    projection_dtype: entry.dtype,
                    router_dtype: router_entry.dtype,
                });
                break;
            }
            // v2.13 fix F1 (HIGH adversarial review) — rank gate.
            // The writeback reconstructs shape from entry.shape rank
            // (1D → 1D, 2D → 2D); rank-3+ has no canonical writeback
            // semantics so refuse loudly rather than silently flatten
            // to 1D (which would re-introduce the non-monotonic shape-
            // rewrite hazard v2.9 fix F2 closed for weights).
            if entry.shape.len() != 1 && entry.shape.len() != 2 {
                bias_refusal = Some(ExpertPruneRefusal::BiasShapeRankUnsupported {
                    name: entry.name.clone(),
                    actual_ndim: entry.shape.len(),
                    num_elements: entry.num_elements,
                });
                break;
            }
            // v2.12 accept-both convention: 1D `[n*dim]` OR 2D `[n, dim]`.
            // The slicing only cares about total elem count divisibility
            // by n_experts. Shape on writeback is reconstructed from
            // the original entry.shape rank to preserve the user's
            // chosen layout (avoids non-monotonic 1D→2D rewrites).
            if !entry.num_elements.is_multiple_of(n.max(1)) {
                bias_refusal = Some(ExpertPruneRefusal::BundleInconsistent {
                    reason: format!(
                        "bias '{}' num_elements {} not divisible by n_experts {n}",
                        entry.name, entry.num_elements,
                    ),
                });
                break;
            }
            let block_elems = entry.num_elements / n.max(1);
            bias_projections.push(ExpertProjection {
                name: entry.name.clone(),
                data: entry.data.clone(),
                block_elems,
                dtype: entry.dtype,
            });
            bias_original_shapes.push(entry.shape.clone());
        }
        if let Some(r) = bias_refusal {
            outcomes.push(MoePruneOutcome::Refused { layer: key, refusal: r });
            continue;
        }

        // Concatenate biases to the weight projections list. The split
        // index (weight_proj_count) lets us separate the kept slices
        // back into weight + bias halves for differentiated writeback.
        let weight_proj_count = projections.len();
        let mut all_projections = projections;
        all_projections.extend(bias_projections);
        let bias_entry_names: Vec<String> = bias_resolved
            .iter()
            .filter_map(|o| o.as_ref().map(|e| e.name.clone()))
            .collect();

        match prune_dead_experts_split(
            &router_entry.data,
            router_entry.dtype,
            d_model,
            n,
            info.top_k,
            &all_projections,
            &dead,
        ) {
            Err(refusal) => outcomes.push(MoePruneOutcome::Refused { layer: key, refusal }),
            Ok(res) if res.dead_experts.is_empty() => {
                outcomes.push(MoePruneOutcome::NoDeadExperts { layer: key });
            }
            Ok(res) => {
                // v2.13: split kept_projections back into weight + bias
                // halves. The first `weight_proj_count` are the
                // original `resolved` weight entries; any remainder
                // are biases (in the same order as bias_resolved /
                // bias_original_shapes / bias_entry_names).
                let (weight_kept, bias_kept) = res.kept_projections.split_at(weight_proj_count);

                let dropped_router_bytes = router_entry.data.len().saturating_sub(res.sliced_router.len()) as u64;
                let dropped_weight_bytes: u64 = resolved
                    .iter()
                    .zip(weight_kept.iter())
                    .map(|(in_e, kp)| in_e.data.len().saturating_sub(kp.data.len()) as u64)
                    .sum();
                // v2.13: bias slicing also drops bytes. Iterate the
                // option vector to match the bias_kept order.
                let dropped_bias_bytes: u64 = bias_resolved
                    .iter()
                    .filter_map(|o| o.as_ref())
                    .zip(bias_kept.iter())
                    .map(|(in_e, kp)| in_e.data.len().saturating_sub(kp.data.len()) as u64)
                    .sum();
                let dropped_bytes = dropped_router_bytes + dropped_weight_bytes + dropped_bias_bytes;

                // Writeback: mutate WeightMap in place. Use the
                // RESOLVED entry's name (preserving the suffix the
                // weight file actually used — `.weight` or bare) so the
                // downstream codegen lookup at expr/calls.rs still
                // resolves them.
                let new_router = WeightEntry::new(
                    router_name.clone(),
                    res.sliced_router,
                    vec![d_model, res.n_live],
                    router_entry.dtype,
                );
                weight_map.insert(new_router);
                for (in_e, kp) in resolved.iter().zip(weight_kept.iter().cloned()) {
                    let new_entry = WeightEntry::new(
                        in_e.name.clone(),
                        kp.data,
                        vec![res.n_live, kp.block_elems],
                        kp.dtype,
                    );
                    weight_map.insert(new_entry);
                }
                // v2.13: bias writeback preserves the original shape
                // rank. A 1D `[n_experts * dim]` entry becomes 1D
                // `[n_live * dim]`; a 2D `[n_experts, dim]` entry
                // becomes 2D `[n_live, dim]`. This avoids the non-
                // monotonic shape-rewrite hazard that v2.9 fix F2
                // codified (changing rank only when at least one
                // expert is dead would be silent corruption).
                for ((name, orig_shape), kp) in bias_entry_names
                    .iter()
                    .zip(bias_original_shapes.iter())
                    .zip(bias_kept.iter().cloned())
                {
                    // v2.13 fix F1 (HIGH adversarial review): exhaustive
                    // match — the upstream rank gate refuses anything
                    // other than 1D or 2D, so a rank-3+ value reaching
                    // here is a structural invariant violation.
                    let new_shape: Vec<usize> = match orig_shape.len() {
                        1 => vec![res.n_live * kp.block_elems],
                        2 => vec![res.n_live, kp.block_elems],
                        _ => unreachable!(
                            "bias rank {} not in {{1,2}} — upstream `BiasShapeRankUnsupported` gate should have refused",
                            orig_shape.len(),
                        ),
                    };
                    let new_entry = WeightEntry::new(
                        name.clone(),
                        kp.data,
                        new_shape,
                        kp.dtype,
                    );
                    weight_map.insert(new_entry);
                }

                if let Some(cfg) = moe_configs.get_mut(&key) {
                    cfg.num_experts = res.n_live;
                }
                outcomes.push(MoePruneOutcome::Pruned {
                    layer: key,
                    layout,
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
            MoePruneOutcome::Pruned { layer, layout, dead, n_live, dropped_bytes } => eprintln!(
                "[cpdt] moe '{layer}' ({layout:?}): pruned experts {dead:?} (affinity < {DEAD_EXPERT_THRESHOLD}) \
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
            // v2.13 fix F12 (IMPORTANT adversarial review): dedicated
            // pretty-print arms for the v2.13 bias-related refusals.
            // Falls back to {refusal:?} Debug for the v2.9 variants
            // (which already have meaningful Debug output).
            MoePruneOutcome::Refused {
                layer,
                refusal: ExpertPruneRefusal::PartialBiasBundle { present, missing },
            } => eprintln!(
                "[cpdt] moe '{layer}': prune refused — partial bias bundle. present={present:?}, missing={missing:?}. \
                 v3/v4 biases are all-or-nothing per layout family (UpDown / GateUpDown). \
                 Fix by adding the missing bias tensors to the checkpoint, OR by removing the present biases entirely."
            ),
            MoePruneOutcome::Refused {
                layer,
                refusal: ExpertPruneRefusal::BiasShapeRankUnsupported { name, actual_ndim, num_elements },
            } => eprintln!(
                "[cpdt] moe '{layer}': prune refused — bias '{name}' has rank {actual_ndim} (num_elements={num_elements}). \
                 v2.13 supports 1D `[n_experts * dim]` or 2D `[n_experts, dim]` bias layouts only. \
                 Reshape the bias tensor to one of those layouts."
            ),
            MoePruneOutcome::Refused {
                layer,
                refusal: ExpertPruneRefusal::SingleLayoutWithBiasEntries { .. },
            } => eprintln!(
                "[cpdt] moe '{layer}': prune refused — Single layout (v1/v2 packed format) does not support FFN biases. \
                 Remove the bias tensors from the checkpoint, OR upgrade the bundle to a multi-projection layout (v3 UpDown / v4 GateUpDown)."
            ),
            MoePruneOutcome::Refused {
                layer,
                refusal: ExpertPruneRefusal::OrphanBiasWithoutWeight { orphan_biases, .. },
            } => eprintln!(
                "[cpdt] moe '{layer}': prune refused — orphan bias bundle. \
                 No expert weight projections were found under this layer, but bias entries exist: {orphan_biases:?}. \
                 Either add the matching weight tensors, or remove these orphan biases from the checkpoint."
            ),
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
