//! WGGO — CFIE inference decision axes for the Level-2 ILP (audit gap G20).
//!
//! The CFIE paper's implementation table promises "Inference decisions in
//! WGGO Level 2": the per-layer ILP co-decides the serving-side
//! configuration — persistent-kernel fusion level, KV-cache layout,
//! per-layer KV precision, and speculative decoding on/off — alongside its
//! training-side axes (heads / FFN / CSHA / rank / optimizer precision /
//! FASE / PCA).  This module supplies the decision vocabulary, the candidate
//! domains, and the decode-latency cost terms the ILP adds to its objective
//! (the same way the optimizer-step term of audit gap #2 composed in).
//!
//! # Honesty constraints (repo WGGO precedents — non-negotiable)
//!
//! * **Opt-in, default OFF.**  The axes only exist when the caller sets
//!   [`crate::wggo_ilp::LayerIlpConstraints::cfie_infer`] to `Some(config)`.
//!   With the gate off (the production default: `None`) the ILP candidate
//!   space, costs, plans, and reports are byte-identical to a build without
//!   this module — pinned by `tests/wggo_cfie_inference_decisions.rs`.
//!   Precedents: `DpConfig::zero_stage_search` (default OFF, gap #6), WRGA
//!   `snap_to_grid` (default OFF, gap #5), ILP `memory_budget` (report-only,
//!   gap #2/#3).
//! * **Advisory / report-only in this cycle.**  Chosen values land on
//!   [`crate::wggo::WggoPlan::cfie_inference`], the per-layer
//!   `DecisionTrace` (kind `CfieInference`), and the rendered WGGO report —
//!   they are NOT consumed by the CFIE serve planner and change no generated
//!   code.  The cross-subsystem hook is an explicit follow-up.
//!
//! # Surface note (scope-driven deviation from the G20 sketch)
//!
//! The audit sketch suggested carrying the chosen values on
//! `AppliedLayer`/`PerLayerOverride` like `adapter_placement`.  Those structs
//! (and `LayerDecision`/`LayerIlpSolution`) are literal-constructed by
//! non-WGGO consumers (CPDT/CSHA/WRGA/FASE test fixtures), so growing them is
//! a cross-subsystem edit.  The decisions surface instead on the plan the
//! WGGO driver alone constructs (`WggoPlan::cfie_inference`) — the same
//! advisory information, without touching consumer-owned struct layouts.
//! When the serve planner starts consuming these decisions, the override
//! carry-through can be added together with that wiring.
//!
//! # Cost model (per decoded token, per layer, microseconds)
//!
//! Batch-1 autoregressive decode is memory-bandwidth-bound plus CPU launch
//! overhead — the same two-term roofline `crate::cfie_cost` uses for the
//! serve report, evaluated per layer:
//!
//! ```text
//! bw          = peak_bandwidth_gbs * 1000            (bytes per us)
//! kv_seq      = max(max_seq / 2, 1)                  (representative half-full
//!                                                     cache, as cfie_cost::estimate)
//! weight_us   = param_bytes / bw                     (matvec decode touches every
//!                                                     weight element once)
//! kv_us       = kv_bytes_per_stored_token * kv_seq / bw
//! launch_us   = launches_per_layer(fusion) * PER_LAUNCH_US   (5 us, cfie_cost)
//! paged_us    = paged_overhead_us     if kv_layout == Paged  (block-table +
//!                                                     host sync share; default
//!                                                     reuses the 5 us constant)
//! dequant_us  = int8_dequant_us       if kv_precision == Int8
//! base        = weight_us + kv_us + launch_us + paged_us + dequant_us
//!
//! speculative on:  cost = base * (1 + draft_frac*K) / (1 + K*p / (1 + K))
//! speculative off: cost = base
//! ```
//!
//! The speculative denominator is the classical expected tokens/step from
//! Leviathan et al. 2023 — the same `1 + K*p/(1+K)` formula
//! `crate::cfie_speculative::emit_program` records as `expected_speedup`.
//! The numerator charges the draft model `draft_frac` of a target-model
//! token per drafted token (K per step).
//!
//! Memory: a `Static` layout must reserve the whole `max_seq` envelope up
//! front (that is what makes direct indexing possible), while `Paged` is
//! charged only its representative `kv_seq` working set.  The pool is added
//! to the layer's resident-memory check, which is the mechanism that makes
//! the layout axis a genuine decision: static is never slower, but under a
//! tight memory budget only paged fits.

use serde::Serialize;

pub use crate::cfie_kv_plan::LayoutKind;
pub use crate::cfie_kv_quant::KvPrecision;
pub use crate::cfie_persistent::FusionLevel;

use crate::cfie_cost::PER_LAUNCH_US;
use crate::wggo_dp::{CoarseDecision, InterLayerPlan};

/// Default in-register INT8 dequantization surcharge (us per token per
/// layer).  Dequantizing the KV block adds ALU work but no extra HBM
/// traffic; a twentieth of a kernel launch (0.25 us vs 5 us) is a coarse,
/// documented placeholder — override it per deployment via
/// [`CfieInferConfig::int8_dequant_us`].
pub const DEFAULT_INT8_DEQUANT_US: f64 = 0.25;

/// Opt-in configuration for the CFIE inference axes.  `Some(config)` on
/// [`crate::wggo_ilp::LayerIlpConstraints::cfie_infer`] turns the axes on;
/// `None` (the default) keeps the ILP byte-identical to today.
#[derive(Debug, Clone)]
pub struct CfieInferConfig {
    /// KV-cache geometry: grouped-query KV heads per layer.
    pub n_kv_heads: u32,
    /// KV-cache geometry: per-head dimension.
    pub head_dim: u32,
    /// Maximum sequence length the serve envelope must handle.  The KV
    /// read cost is evaluated at `max_seq / 2` (the same representative
    /// operating point `cfie_cost::estimate` uses); a static layout
    /// reserves the full `max_seq` pool.
    pub max_seq: u32,
    /// Deepest fusion level the candidate domain may offer.  The serve
    /// planner's SMEM feasibility analysis (`cfie_persistent::choose_fusion`)
    /// owns the real budget check; callers pass its verdict down as a cap.
    pub max_fusion_level: FusionLevel,
    /// Extra latency charged to a `Paged` KV layout (us per token per
    /// layer): the block-table indirection + host synchronization share the
    /// static direct-index layout eliminates.  Default reuses the 5 us
    /// launch constant (`cfie_cost::PER_LAUNCH_US`) so the penalty is the
    /// same, documented order of magnitude as one kernel launch.
    pub paged_overhead_us: f64,
    /// Extra latency charged to an `Int8` KV precision (us per token per
    /// layer) for in-register dequantization.  See
    /// [`DEFAULT_INT8_DEQUANT_US`].
    pub int8_dequant_us: f64,
    /// Speculative decoding: number of drafted tokens per step (K).
    pub spec_k: u32,
    /// Speculative decoding: expected draft acceptance rate p in [0, 1].
    pub spec_acceptance: f64,
    /// Draft-model cost as a fraction of a target-model token, charged per
    /// drafted token (K per step).
    pub spec_draft_cost_frac: f64,
}

impl Default for CfieInferConfig {
    fn default() -> Self {
        Self {
            // Matches the driver's default LayerShape (run_on_wengert).
            n_kv_heads: 4,
            head_dim: 64,
            // Matches cfie_kv_plan::KvBudget::default().
            max_seq: 2048,
            max_fusion_level: FusionLevel::Level3,
            paged_overhead_us: PER_LAUNCH_US,
            int8_dequant_us: DEFAULT_INT8_DEQUANT_US,
            // K=4 drafted tokens; p=0.7 acceptance; draft costs 15% of a
            // target token.  At these defaults speculative does NOT pay
            // ((1+0.15*4)=1.60 > (1+4*0.7/5)=1.56), so the conservative
            // spec-off choice wins until the caller supplies a measured
            // acceptance rate above ~0.75.
            spec_k: 4,
            spec_acceptance: 0.7,
            spec_draft_cost_frac: 0.15,
        }
    }
}

impl CfieInferConfig {
    /// Field-by-field equality (f64 via bit patterns) for the templated
    /// ILP's cache key — mirrors `wggo_ilp::constraints_eq`'s style so two
    /// layers whose gates differ never share a templated solution.
    pub fn key_eq(&self, other: &Self) -> bool {
        self.n_kv_heads == other.n_kv_heads
            && self.head_dim == other.head_dim
            && self.max_seq == other.max_seq
            && self.max_fusion_level == other.max_fusion_level
            && self.paged_overhead_us.to_bits() == other.paged_overhead_us.to_bits()
            && self.int8_dequant_us.to_bits() == other.int8_dequant_us.to_bits()
            && self.spec_k == other.spec_k
            && self.spec_acceptance.to_bits() == other.spec_acceptance.to_bits()
            && self.spec_draft_cost_frac.to_bits() == other.spec_draft_cost_frac.to_bits()
    }
}

/// One assignment of the four CFIE inference decision variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CfieInferenceChoice {
    /// Persistent-kernel fusion level.  G20 axis domain: {none, level2,
    /// level3} per the audit list (Level1 is a CSHA-side stepping stone the
    /// audit's inference axis does not enumerate).
    pub fusion_level: FusionLevel,
    /// KV-cache layout.  G20 axis domain: {paged, static}
    /// (`StaticWithBump` is the serve planner's fragmentation hybrid, not
    /// an ILP-level axis).
    pub kv_layout: LayoutKind,
    /// Per-layer KV precision.  G20 axis domain: {fp16, int8} — the
    /// vocabulary is `cfie_kv_quant::KvPrecision` (re-exported here);
    /// Bf16/Int4 stay serve-planner refinements.
    pub kv_precision: KvPrecision,
    /// Speculative decoding on/off.
    pub speculative: bool,
}

/// Per-layer advisory record surfaced on `WggoPlan::cfie_inference`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CfieLayerInference {
    pub layer_index: u32,
    pub layer_name: String,
    pub choice: CfieInferenceChoice,
    /// Human-readable statement of the cost model AND the constants this
    /// layer's decision was priced with (the `CfieCostEstimate.assumptions`
    /// precedent) — the report prints it so the advisory numbers are
    /// inspectable without reading code.
    pub note: String,
}

/// The assumptions line for a config: restates the decode cost model with
/// its actual constants (paged overhead, dequant surcharge, speculative
/// K / p / draft fraction, operating point).
pub fn model_note(cfg: &CfieInferConfig) -> String {
    format!(
        "decode_us/token/layer = param_bytes/bw + kv_bytes*kv_seq/bw + \
         launches*{launch:.0}us (+{paged:.2}us paged, +{deq:.2}us int8 dequant); \
         speculative x (1+{frac:.2}*K)/(1+K*p/(1+K)) [Leviathan 2023], K={k}, p={p:.2}; \
         kv_seq = max_seq/2 = {seq}",
        launch = PER_LAUNCH_US,
        paged = cfg.paged_overhead_us,
        deq = cfg.int8_dequant_us,
        frac = cfg.spec_draft_cost_frac,
        k = cfg.spec_k,
        p = cfg.spec_acceptance,
        seq = representative_kv_seq(cfg),
    )
}

/// Total order used to apply the `max_fusion_level` cap.
fn fusion_rank(level: FusionLevel) -> u8 {
    match level {
        FusionLevel::None => 0,
        FusionLevel::Level1 => 1,
        FusionLevel::Level2 => 2,
        FusionLevel::Level3 => 3,
    }
}

/// Kernel launches per layer per decode token at a given fusion level.
///
/// Mirrors the launch table inside `cfie_persistent::plan` (baseline 12 —
/// "norm + Q/K/V + RoPE + attention + O + FFN gate/up/down + 2x residual
/// adds"; Level1 = 6, Level2 = 2, Level3 = 1).  Mirrored (with this
/// documentation) because those constants are inline in `plan()`, not an
/// importable `pub fn` — G20 design note: "import if pub, else mirror +
/// document".
pub fn launches_per_layer(level: FusionLevel) -> u32 {
    match level {
        FusionLevel::None => 12,
        FusionLevel::Level1 => 6,
        FusionLevel::Level2 => 2,
        FusionLevel::Level3 => 1,
    }
}

/// Representative KV operating point: a half-full cache, exactly as
/// `cfie_cost::estimate` evaluates the serve report's KV term.
pub fn representative_kv_seq(cfg: &CfieInferConfig) -> u32 {
    (cfg.max_seq / 2).max(1)
}

/// KV bytes per stored token per layer at a precision: `2 (K+V) x
/// n_kv_heads x head_dim` elements, sized by the precision's exact byte
/// footprint (`KvPrecision::bytes_for_elems` — the same accounting
/// `cfie_kv_plan::KvShape::bytes_per_token_per_layer` uses for FP16).
pub fn kv_stored_bytes_per_token(cfg: &CfieInferConfig, precision: KvPrecision) -> u64 {
    let elems = 2u64 * (cfg.n_kv_heads as u64) * (cfg.head_dim as u64);
    precision.bytes_for_elems(elems)
}

/// HBM bytes the KV cache reserves for one sequence on this layer under a
/// layout choice.  `Static` (and the bump hybrid, if it ever enters the
/// domain) must pre-allocate the full `max_seq` envelope — that is the
/// price of direct indexing; `Paged` is charged its representative
/// working set (`kv_seq` tokens).  Added to the ILP's per-layer
/// resident-memory check when the gate is on.
pub fn kv_pool_bytes(choice: CfieInferenceChoice, cfg: &CfieInferConfig) -> u64 {
    let per_token = kv_stored_bytes_per_token(cfg, choice.kv_precision);
    match choice.kv_layout {
        LayoutKind::Static | LayoutKind::StaticWithBump => {
            per_token.saturating_mul(cfg.max_seq.max(1) as u64)
        }
        LayoutKind::Paged => per_token.saturating_mul(representative_kv_seq(cfg) as u64),
    }
}

/// Multiplicative factor speculative decoding applies to the per-token
/// decode cost:
///
/// ```text
/// (1 + draft_frac * K) / (1 + K * p / (1 + K))
/// ```
///
/// Denominator = expected tokens per verification step (Leviathan et al.
/// 2023; identical to `cfie_speculative::emit_program`'s
/// `expected_speedup`).  Numerator = one target step plus K drafted tokens
/// at `draft_frac` of a target token each.  Factor < 1 means speculative
/// decoding pays for itself.
pub fn spec_cost_factor(cfg: &CfieInferConfig) -> f64 {
    if cfg.spec_k == 0 {
        return 1.0;
    }
    let k = cfg.spec_k as f64;
    let speedup = 1.0 + (k * cfg.spec_acceptance) / (1.0 + k);
    let draft_overhead = 1.0 + cfg.spec_draft_cost_frac * k;
    draft_overhead / speedup.max(1e-9)
}

/// Decode latency (us per token) this layer contributes under a CFIE
/// inference choice.  See the module docs for the formula; `param_bytes`
/// is the candidate LUT entry's parameter footprint (matvec decode touches
/// every weight element once per token — the same "weights read once"
/// argument as `cfie_cost::weight_bytes_per_token`), and
/// `peak_bandwidth_gbs` is the LUT's HBM bandwidth so the term is
/// commensurate, in us, with the forward/backward/optimizer terms the ILP
/// already sums.
pub fn decode_us_per_token(
    choice: CfieInferenceChoice,
    cfg: &CfieInferConfig,
    param_bytes: u64,
    peak_bandwidth_gbs: f64,
) -> f64 {
    // GB/s = 1e9 bytes / 1e6 us => bytes/us = GB/s * 1000 (as cfie_cost).
    let bw = peak_bandwidth_gbs.max(1.0) * 1000.0;
    let kv_seq = representative_kv_seq(cfg) as u64;
    let weight_us = param_bytes as f64 / bw;
    let kv_us = (kv_stored_bytes_per_token(cfg, choice.kv_precision) as f64) * (kv_seq as f64) / bw;
    let launch_us = launches_per_layer(choice.fusion_level) as f64 * PER_LAUNCH_US;
    let paged_us = if matches!(choice.kv_layout, LayoutKind::Paged) {
        cfg.paged_overhead_us
    } else {
        0.0
    };
    let dequant_us = if matches!(choice.kv_precision, KvPrecision::Int8 | KvPrecision::Int4) {
        cfg.int8_dequant_us
    } else {
        0.0
    };
    let base = weight_us + kv_us + launch_us + paged_us + dequant_us;
    if choice.speculative {
        base * spec_cost_factor(cfg)
    } else {
        base
    }
}

/// Candidate domain for the ILP's CFIE axis.
///
/// * Gate off (`cfg == None`): the single `None` element — the solver's
///   loop collapses to one pass with a zero cost term, keeping the gate-off
///   candidate space (and node counts) byte-identical to today.
/// * Gate on: the cross-product of the four axes.  Ordering is
///   deterministic and puts the conservative value first on every axis
///   where cost ties are possible (Static before Paged, Fp16 before Int8,
///   spec-off before spec-on) — the solver keeps the FIRST candidate on a
///   tie (strict `<` improvement), so a risky value must be strictly
///   cheaper to win.  Fusion is enumerated deepest-first (ties impossible:
///   launch counts differ) so a good incumbent lands early.
pub fn enumerate_choices(cfg: Option<&CfieInferConfig>) -> Vec<Option<CfieInferenceChoice>> {
    let Some(cfg) = cfg else {
        return vec![None];
    };
    let cap = fusion_rank(cfg.max_fusion_level);
    let fusion_domain = [FusionLevel::Level3, FusionLevel::Level2, FusionLevel::None];
    let mut out = Vec::with_capacity(24);
    for &fusion_level in fusion_domain.iter().filter(|&&l| fusion_rank(l) <= cap) {
        for kv_layout in [LayoutKind::Static, LayoutKind::Paged] {
            for kv_precision in [KvPrecision::Fp16, KvPrecision::Int8] {
                for speculative in [false, true] {
                    out.push(Some(CfieInferenceChoice {
                        fusion_level,
                        kv_layout,
                        kv_precision,
                        speculative,
                    }));
                }
            }
        }
    }
    out
}

/// Greedy pick: the cheapest choice whose KV pool fits `pool_budget`
/// (remaining memory headroom after the layer's training-resident bytes).
/// Returns `None` when nothing fits — the greedy solver then simply
/// refuses to advise (no choice surfaced) rather than advising an
/// over-budget pool.
pub fn best_choice_fitting(
    cfg: &CfieInferConfig,
    param_bytes: u64,
    peak_bandwidth_gbs: f64,
    pool_budget: u64,
) -> Option<(CfieInferenceChoice, f64)> {
    let mut best: Option<(CfieInferenceChoice, f64)> = None;
    for choice in enumerate_choices(Some(cfg)).into_iter().flatten() {
        if kv_pool_bytes(choice, cfg) > pool_budget {
            continue;
        }
        let cost = decode_us_per_token(choice, cfg, param_bytes, peak_bandwidth_gbs);
        if best.as_ref().is_none_or(|(_, b)| cost < *b) {
            best = Some((choice, cost));
        }
    }
    best
}

/// Project the per-layer choices onto the advisory plan surface, skipping
/// layers the inter-layer DP pruned (mirroring `wggo_apply::apply`'s
/// forced-safe-defaults treatment of pruned layers).  `constraints`
/// supplies each layer's gate config for the record's model note; the
/// defensive first-element fallback mirrors the driver's `recost_total`
/// indexing (a caller may pass fewer constraints than layers).
pub fn surface_from_plan(
    inter: &InterLayerPlan,
    choices: &[Option<CfieInferenceChoice>],
    constraints: &[crate::wggo_ilp::LayerIlpConstraints],
) -> Vec<CfieLayerInference> {
    inter
        .layers
        .iter()
        .zip(choices.iter())
        .enumerate()
        .filter(|(_, (l, _))| l.decision != CoarseDecision::Prune)
        .filter_map(|(i, (l, ch))| {
            ch.map(|choice| {
                let note = constraints
                    .get(i)
                    .or_else(|| constraints.first())
                    .and_then(|c| c.cfie_infer.as_ref())
                    .map(model_note)
                    .unwrap_or_default();
                CfieLayerInference {
                    layer_index: l.layer_index,
                    layer_name: l.name.clone(),
                    choice,
                    note,
                }
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(
        fusion_level: FusionLevel,
        kv_layout: LayoutKind,
        kv_precision: KvPrecision,
        speculative: bool,
    ) -> CfieInferenceChoice {
        CfieInferenceChoice {
            fusion_level,
            kv_layout,
            kv_precision,
            speculative,
        }
    }

    #[test]
    fn gate_off_domain_is_the_single_none() {
        assert_eq!(enumerate_choices(None), vec![None]);
    }

    #[test]
    fn gate_on_domain_is_full_cross_product() {
        let cfg = CfieInferConfig::default();
        let choices = enumerate_choices(Some(&cfg));
        // 3 fusion x 2 layout x 2 precision x 2 spec = 24.
        assert_eq!(choices.len(), 24);
        assert!(choices.iter().all(|c| c.is_some()));
        // Conservative-first tie-break ordering within each fusion level.
        let first = choices[0].unwrap();
        assert_eq!(first.fusion_level, FusionLevel::Level3);
        assert_eq!(first.kv_layout, LayoutKind::Static);
        assert_eq!(first.kv_precision, KvPrecision::Fp16);
        assert!(!first.speculative);
    }

    #[test]
    fn fusion_cap_filters_domain() {
        let cfg = CfieInferConfig {
            max_fusion_level: FusionLevel::Level2,
            ..Default::default()
        };
        let choices = enumerate_choices(Some(&cfg));
        // 2 fusion (Level2, None) x 2 x 2 x 2 = 16; Level3 filtered out.
        assert_eq!(choices.len(), 16);
        assert!(choices
            .iter()
            .flatten()
            .all(|c| c.fusion_level != FusionLevel::Level3));
    }

    #[test]
    fn launch_table_mirrors_cfie_persistent_plan() {
        // Documentation of the mirrored values (the inline table in
        // cfie_persistent::plan)...
        assert_eq!(launches_per_layer(FusionLevel::None), 12);
        assert_eq!(launches_per_layer(FusionLevel::Level1), 6);
        assert_eq!(launches_per_layer(FusionLevel::Level2), 2);
        assert_eq!(launches_per_layer(FusionLevel::Level3), 1);

        // ...and the REAL drift pin: cross-assert against actual
        // cfie_persistent::plan() output, so a change to that inline
        // table fails HERE instead of silently diverging the WGGO cost
        // model.  Sweep model sizes / SMEM budgets chosen to land on
        // different fusion levels; whatever level each plan picks, its
        // launch count must equal this module's table entry for it.
        use crate::cfie_persistent::{plan as persistent_plan, GpuBudget, PersistentModel};
        let cases = [
            // (d_model, head_dim, n_heads, n_kv, d_ff, smem_per_sm)
            (64u32, 32u32, 2u32, 1u32, 128u32, 228 * 1024u32), // tiny model, H100 SMEM
            (512, 128, 4, 4, 1408, 228 * 1024),                // paper config
            (4096, 128, 32, 8, 14336, 164 * 1024),             // big model, A100 SMEM
            (8192, 128, 64, 8, 28672, 96 * 1024),              // bigger model, small SMEM
        ];
        let mut seen = std::collections::HashSet::new();
        for (d_model, head_dim, n_heads, n_kv_heads, d_ff, smem) in cases {
            let model = PersistentModel {
                d_model,
                head_dim,
                n_layers: 2,
                n_heads,
                n_kv_heads,
                d_ff,
                dtype_bytes: 2,
            };
            let budget = GpuBudget {
                smem_per_sm: smem,
                num_sms: 108,
                kernel_launch_us: 5.0,
            };
            let p = persistent_plan(&model, &budget, 4);
            assert_eq!(
                launches_per_layer(p.fusion),
                p.persistent_launches_per_layer,
                "wggo_cfie launch table diverged from cfie_persistent::plan at fusion={}",
                p.fusion.as_str()
            );
            assert_eq!(
                launches_per_layer(FusionLevel::None),
                p.baseline_launches_per_layer,
                "baseline launch count diverged from cfie_persistent::plan"
            );
            seen.insert(p.fusion.as_str());
        }
        assert!(
            seen.len() >= 2,
            "fixture sweep must exercise at least two fusion levels (saw {seen:?}) — \
             widen the sweep if choose_fusion's thresholds moved"
        );
    }

    #[test]
    fn kv_bytes_and_pools_hand_computed() {
        let cfg = CfieInferConfig {
            n_kv_heads: 4,
            head_dim: 64,
            max_seq: 4096,
            ..Default::default()
        };
        // 2 (K+V) * 4 heads * 64 dim = 512 elements/token/layer.
        //   fp16: 512 * 2 = 1024 B; int8: 512 * 1 = 512 B.
        assert_eq!(kv_stored_bytes_per_token(&cfg, KvPrecision::Fp16), 1024);
        assert_eq!(kv_stored_bytes_per_token(&cfg, KvPrecision::Int8), 512);
        // Static reserves the full 4096-token envelope; paged only the
        // representative 2048-token working set.
        let stat = choice(FusionLevel::Level3, LayoutKind::Static, KvPrecision::Fp16, false);
        let paged = choice(FusionLevel::Level3, LayoutKind::Paged, KvPrecision::Fp16, false);
        assert_eq!(kv_pool_bytes(stat, &cfg), 1024 * 4096); // 4,194,304
        assert_eq!(kv_pool_bytes(paged, &cfg), 1024 * 2048); // 2,097,152
    }

    #[test]
    fn decode_us_hand_computed() {
        // bw = 1000 GB/s => 1,000,000 bytes/us.  param = 1,000,000 B =>
        // weight_us = 1.0.  kv fp16 = 1024 B/token * kv_seq 1024 =
        // 1,048,576 B => kv_us = 1.048576.  fusion None => 12 * 5 = 60 us.
        // Static, fp16, no spec => base = 1.0 + 1.048576 + 60 = 62.048576.
        let cfg = CfieInferConfig {
            n_kv_heads: 4,
            head_dim: 64,
            max_seq: 2048,
            ..Default::default()
        };
        let c = choice(FusionLevel::None, LayoutKind::Static, KvPrecision::Fp16, false);
        let got = decode_us_per_token(c, &cfg, 1_000_000, 1000.0);
        assert!((got - 62.048576).abs() < 1e-9, "got {got}");

        // Paged adds the 5 us default overhead: 67.048576.
        let p = choice(FusionLevel::None, LayoutKind::Paged, KvPrecision::Fp16, false);
        let got_paged = decode_us_per_token(p, &cfg, 1_000_000, 1000.0);
        assert!((got_paged - 67.048576).abs() < 1e-9, "got {got_paged}");

        // Int8 halves the KV read (0.524288) and adds the dequant charge
        // (default 0.25): 1.0 + 0.524288 + 60 + 0.25 = 61.774288.
        let i = choice(FusionLevel::None, LayoutKind::Static, KvPrecision::Int8, false);
        let got_i8 = decode_us_per_token(i, &cfg, 1_000_000, 1000.0);
        assert!((got_i8 - 61.774288).abs() < 1e-9, "got {got_i8}");

        // Level3 drops launches to 1 * 5 us: 1.0 + 1.048576 + 5 = 7.048576.
        let l3 = choice(FusionLevel::Level3, LayoutKind::Static, KvPrecision::Fp16, false);
        let got_l3 = decode_us_per_token(l3, &cfg, 1_000_000, 1000.0);
        assert!((got_l3 - 7.048576).abs() < 1e-9, "got {got_l3}");
    }

    #[test]
    fn spec_factor_hand_computed() {
        // K=4, p=0.7, draft=0.15: (1 + 0.6) / (1 + 2.8/5) = 1.6/1.56 =
        // 1.02564... > 1 — speculative loses at the defaults.
        let cfg = CfieInferConfig::default();
        let f = spec_cost_factor(&cfg);
        assert!((f - 1.6 / 1.56).abs() < 1e-12, "got {f}");
        assert!(f > 1.0);

        // p=0.9: 1.6 / (1 + 3.6/5) = 1.6/1.72 = 0.93023... < 1 — wins.
        let cfg_hot = CfieInferConfig {
            spec_acceptance: 0.9,
            ..Default::default()
        };
        let f_hot = spec_cost_factor(&cfg_hot);
        assert!((f_hot - 1.6 / 1.72).abs() < 1e-12, "got {f_hot}");
        assert!(f_hot < 1.0);

        // K=0 degenerates to exactly 1 (no drafting, no benefit).
        let cfg_k0 = CfieInferConfig {
            spec_k: 0,
            ..Default::default()
        };
        assert_eq!(spec_cost_factor(&cfg_k0), 1.0);
    }

    #[test]
    fn best_choice_fitting_respects_pool_budget() {
        let cfg = CfieInferConfig {
            n_kv_heads: 4,
            head_dim: 64,
            max_seq: 4096,
            ..Default::default()
        };
        // Unlimited headroom: static wins (paged strictly slower).
        let (c, _) = best_choice_fitting(&cfg, 1_000_000, 1000.0, u64::MAX).unwrap();
        assert_eq!(c.kv_layout, LayoutKind::Static);
        // Headroom below every pool (smallest is paged-int8 at
        // 512 * 2048 = 1,048,576): refuse to advise.
        assert!(best_choice_fitting(&cfg, 1_000_000, 1000.0, 1_000_000).is_none());
        // Headroom that only admits paged-int8.
        let (c2, _) = best_choice_fitting(&cfg, 1_000_000, 1000.0, 1_048_576).unwrap();
        assert_eq!(c2.kv_layout, LayoutKind::Paged);
        assert_eq!(c2.kv_precision, KvPrecision::Int8);
    }

    #[test]
    fn key_eq_detects_field_drift() {
        let a = CfieInferConfig::default();
        assert!(a.key_eq(&CfieInferConfig::default()));
        let b = CfieInferConfig {
            spec_acceptance: 0.9,
            ..Default::default()
        };
        assert!(!a.key_eq(&b));
        let c = CfieInferConfig {
            max_fusion_level: FusionLevel::Level2,
            ..Default::default()
        };
        assert!(!a.key_eq(&c));
    }
}
