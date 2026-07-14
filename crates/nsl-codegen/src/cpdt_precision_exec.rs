//! CPDT precision-adaptive optimizer EXECUTION bridge (v1: FP16).
//!
//! Turns the per-parameter `PrecisionPlan` into ordered dtype codes aligned
//! with `param_paths` (the same ordered name list the codegen uses to build
//! `param_list`). v1 clamps INT8 tiers to FP16 — `clamp_int8_to_fp16` is the
//! SINGLE removal point for the INT8 ladder step (design doc §10).

use crate::cpdt_tier_apply::{OptimPrecision, ParamPrecision, PrecisionPlan};

/// Runtime dtype codes (mirror `nsl-runtime` `DTYPE_*`).
pub const DTYPE_F32: u16 = 1;
pub const DTYPE_FP16: u16 = 2;
pub const DTYPE_INT8: u16 = 4;

/// v1 clamp: map an optimizer-state precision to FP16 storage (Part II's FASE
/// wrapping handles only F32/FP16). This is the Part II FP16 PATH — the runtime
/// has INT8 blockwise ops now (§3.2 / Sprint 4), and the FASE-side cast wrapping
/// to consume them is a separate follow-on.
pub fn clamp_int8_to_fp16(p: OptimPrecision) -> u16 {
    match p {
        OptimPrecision::Fp32 => DTYPE_F32,
        OptimPrecision::Fp16 => DTYPE_FP16,
        OptimPrecision::Int8 => DTYPE_FP16,
    }
}

/// Paper-faithful dtype mapping for `OptimPrecision`. FP32 → DTYPE_F32, FP16 →
/// DTYPE_FP16, INT8 → DTYPE_INT8 (blockwise — runtime support shipped in §3.2).
/// Use this when the consumer (cast wrapper / FASE step) actually handles INT8
/// storage. `clamp_int8_to_fp16` remains the safe Part II default.
pub fn dtype_for_precision_full(p: OptimPrecision) -> u16 {
    match p {
        OptimPrecision::Fp32 => DTYPE_F32,
        OptimPrecision::Fp16 => DTYPE_FP16,
        OptimPrecision::Int8 => DTYPE_INT8,
    }
}

/// Find the precision for a param path. Matches CPDT plan names against the
/// codegen path with light normalization: exact match, or a trailing
/// `.weight`/`.bias` stripped from either side. Unmatched -> None.
fn precision_for_path<'a>(plan: &'a PrecisionPlan, path: &str) -> Option<&'a ParamPrecision> {
    plan.params.iter().find(|p| {
        p.name == path
            || p.name.strip_suffix(".weight").map(|s| s == path).unwrap_or(false)
            || p.name.strip_suffix(".bias").map(|s| s == path).unwrap_or(false)
            || path.strip_suffix(".weight").map(|s| s == p.name).unwrap_or(false)
            || path.strip_suffix(".bias").map(|s| s == p.name).unwrap_or(false)
    })
}

/// Build (m_dtype_codes, v_dtype_codes) in `param_paths` order. Each code is
/// `DTYPE_F32` or `DTYPE_FP16`. Unmatched paths -> (F32, F32) (safe fallback).
pub fn build_dtype_lists(plan: &PrecisionPlan, param_paths: &[String]) -> (Vec<u16>, Vec<u16>) {
    let mut m = Vec::with_capacity(param_paths.len());
    let mut v = Vec::with_capacity(param_paths.len());
    for path in param_paths {
        match precision_for_path(plan, path) {
            Some(pp) => {
                m.push(clamp_int8_to_fp16(pp.m_precision));
                v.push(clamp_int8_to_fp16(pp.v_precision));
            }
            None => {
                m.push(DTYPE_F32);
                v.push(DTYPE_F32);
            }
        }
    }
    (m, v)
}


/// Build (m_dtype_codes, v_dtype_codes) from WGGO's per-layer moment-bit
/// decisions. Returns `None` when no layer carries a sub-32-bit decision
/// (the caller falls through to the CPDT PrecisionPlan path / plain FP32).
///
/// Params are joined to layers with the SAME `wggo_graph::layer_prefix`
/// the graph builder used to bucket them (handles the model-variable
/// prefix and arbitrary container nesting); unmatched params stay F32.
/// Sub-32 bits map to FP16 storage in v1 — the same ladder step as
/// [`clamp_int8_to_fp16`].
pub fn build_dtype_lists_from_overrides(
    overrides: &crate::wggo_overrides::WggoOverrides,
    param_paths: &[String],
) -> Option<(Vec<u16>, Vec<u16>)> {
    let any_sub32 = overrides
        .per_layer
        .iter()
        .any(|l| l.optim_m_bits < 32 || l.optim_v_bits < 32);
    if !any_sub32 {
        return None;
    }
    let map_bits = |b: u8| -> u16 {
        if b < 32 {
            DTYPE_FP16
        } else {
            DTYPE_F32
        }
    };
    let mut m = Vec::with_capacity(param_paths.len());
    let mut v = Vec::with_capacity(param_paths.len());
    for path in param_paths {
        let layer = crate::wggo_graph::layer_prefix(path)
            .and_then(|lp| overrides.per_layer.iter().find(|l| l.layer_name == lp));
        match layer {
            Some(l) => {
                m.push(map_bits(l.optim_m_bits));
                v.push(map_bits(l.optim_v_bits));
            }
            None => {
                m.push(DTYPE_F32);
                v.push(DTYPE_F32);
            }
        }
    }
    // The Some/None boundary is decided AFTER the join: a plan whose only
    // sub-32 layers cannot be joined to any param (e.g. the synthetic
    // 'other' bucket, whose params have no layer_prefix) must NOT claim
    // activation — returning Some(all-F32) here would both lie in the
    // activation diagnostic and preempt an active per-param CPDT plan.
    if m.iter().all(|&c| c == DTYPE_F32) && v.iter().all(|&c| c == DTYPE_F32) {
        return None;
    }
    Some((m, v))
}

/// True when v1 precision-adaptive optimizer execution should activate.
///
/// Five conditions are required (design doc §6): CPDT Full mode, a non-empty
/// precision plan, weights present, FASE Deferred mode, AND `wrapped_path_active`.
///
/// `wrapped_path_active` is load-bearing for correctness. It rules out paths
/// that consume m/v WITHOUT a dequant→step→quant wrap. Three such paths
/// historically existed:
/// 1. The unified-dispatch arm hardcoding `wrap_precision=false`. **Closed
///    by S2** — threaded the wrap through.
/// 2. The unified-dispatch FullBuffer sub-arm passing raw s1/s2 to
///    `emit_stdlib_optim_call` (F32-only FFI). **Closed by S5** —
///    `emit_stdlib_optim_call` now itself takes `wrap_precision: bool` and
///    wraps in the same dequant→step→quant envelope as
///    `fase_emit_final_step`.
/// 3. The standalone FullBuffer-global call site at `stmt.rs:5927`
///    (no-WGGO, no mode table). The 4th condition (`fase_mode_is_deferred`)
///    of this gate already rejects activation at that path; the call site
///    passes `wrap_precision=false` for signature uniformity.
///
/// With all three paths covered, `wrapped_path_active` can be set
/// unconditionally to `true` (S5 lift). The parameter is retained as
/// defense-in-depth.
///
/// Closed follow-ons (Part II FP16 activation cycle, 2026-06-04):
/// - S1: source-AD MulElementwise broadcast crash (added third
///   `target` field to AdjointExpr + reduce_to_shape wrap).
/// - S2: unified-dispatch Deferred-arm wrap threading.
/// - S3: `--wggo*` flags on `nsl run`.
/// - S4: `wrapped_path_active` refinement to mode-table FullBuffer guard.
/// - S5: unified-dispatch FullBuffer-arm wrap threading +
///   `wrapped_path_active` lift to unconditional `true`.
///
/// Part II FP16 optimizer execution is now live end-to-end on every
/// optimizer-emission code path. Activation only requires CPDT Full mode +
/// a non-empty PrecisionPlan + FASE Deferred (AdamW with
/// grad_accumulation > 1). No remaining silent-fallback paths.
pub fn precision_active(
    cpdt_mode_is_full: bool,
    has_precision_plan: bool,
    weights_present: bool,
    fase_mode_is_deferred: bool,
    wrapped_path_active: bool,
) -> bool {
    cpdt_mode_is_full
        && has_precision_plan
        && weights_present
        && fase_mode_is_deferred
        && wrapped_path_active
}

/// Outcome of arbitrating WGGO's per-layer moment-bit decision against
/// CPDT's per-param `PrecisionPlan` for the optimizer-state (m/v) dtype
/// lists. Kept distinct (rather than collapsing straight to
/// `Option<(Vec<u16>, Vec<u16>)>`) so the caller can emit a diagnostic
/// that names which source(s) actually decided.
pub enum MomentPrecisionArbitration {
    /// WGGO decided sub-32-bit moments for this block but
    /// `--wggo-moment-precision` was not passed: stay conservative and
    /// drop BOTH sources (CPDT's plan included) rather than let CPDT
    /// silently diverge from WGGO's stated decision.
    NotLoweredNoOptIn,
    /// Both sources are live and opted in: merged conservatively (F32
    /// wins per param).
    Merged(Vec<u16>, Vec<u16>),
    /// Only WGGO decided (opted in).
    WggoOnly(Vec<u16>, Vec<u16>),
    /// Only CPDT decided. Independent of WGGO — applies unconditionally,
    /// including when WGGO ran on this block for an unrelated reason
    /// (e.g. structural pruning, CSHA fusion) and made no moment-bit
    /// decision of its own.
    CpdtOnly(Vec<u16>, Vec<u16>),
    /// Neither source decided anything; moments stay FP32.
    Inactive,
}

/// Arbitrate WGGO's per-layer moment-bit decision (`wggo_bits`, `None`
/// unless WGGO itself chose sub-32-bit moments for this train block)
/// against CPDT's per-param `PrecisionPlan` (`cpdt_lists`).
///
/// `wggo_bits.is_some()` is the correct gate for the opt-in requirement —
/// NOT "WGGO ran on this block at all". WGGO can be active for entirely
/// unrelated reasons (head pruning, CSHA fusion, packing) without ever
/// deciding anything about optimizer-moment precision; in that case
/// `wggo_bits` is `None` and an independent CPDT plan must not be
/// silently discarded just because WGGO happened to run.
pub fn arbitrate_moment_precision(
    wggo_bits: Option<(Vec<u16>, Vec<u16>)>,
    cpdt_lists: Option<(Vec<u16>, Vec<u16>)>,
    wggo_moment_precision_opt_in: bool,
) -> MomentPrecisionArbitration {
    match (wggo_bits, cpdt_lists) {
        (Some(_), _) if !wggo_moment_precision_opt_in => {
            MomentPrecisionArbitration::NotLoweredNoOptIn
        }
        (Some((wm, wv)), Some((cm, cv))) => {
            let merge = |a: &[u16], b: &[u16]| -> Vec<u16> {
                a.iter()
                    .zip(b)
                    .map(|(&x, &y)| if x == DTYPE_F32 || y == DTYPE_F32 { DTYPE_F32 } else { x })
                    .collect()
            };
            MomentPrecisionArbitration::Merged(merge(&wm, &cm), merge(&wv, &cv))
        }
        (Some((m, v)), None) => MomentPrecisionArbitration::WggoOnly(m, v),
        (None, Some((m, v))) => MomentPrecisionArbitration::CpdtOnly(m, v),
        (None, None) => MomentPrecisionArbitration::Inactive,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpdt_tier_apply::{ParamPrecision, PrecisionPlan, Tier};

    fn pp(name: &str, m: OptimPrecision, v: OptimPrecision) -> ParamPrecision {
        ParamPrecision {
            name: name.to_string(),
            layer: Some(0),
            tier: Tier::Low,
            m_precision: m,
            v_precision: v,
            stochastic_rounding: false,
            sensitivity_score: 0.0,
            param_bytes: 0,
            optim_bytes: 0,
        }
    }

    #[test]
    fn clamp_maps_int8_to_fp16() {
        assert_eq!(clamp_int8_to_fp16(OptimPrecision::Fp32), DTYPE_F32);
        assert_eq!(clamp_int8_to_fp16(OptimPrecision::Fp16), DTYPE_FP16);
        assert_eq!(clamp_int8_to_fp16(OptimPrecision::Int8), DTYPE_FP16);
    }

    #[test]
    fn full_maps_int8_to_dtype_int8() {
        // CPDT §3.2: runtime now supports INT8 blockwise storage (Sprint 4),
        // so the paper-faithful dtype mapping returns DTYPE_INT8.
        assert_eq!(dtype_for_precision_full(OptimPrecision::Fp32), DTYPE_F32);
        assert_eq!(dtype_for_precision_full(OptimPrecision::Fp16), DTYPE_FP16);
        assert_eq!(dtype_for_precision_full(OptimPrecision::Int8), DTYPE_INT8);
    }

    #[test]
    fn full_and_clamp_differ_only_on_int8() {
        for p in [OptimPrecision::Fp32, OptimPrecision::Fp16] {
            assert_eq!(dtype_for_precision_full(p), clamp_int8_to_fp16(p));
        }
        assert_ne!(
            dtype_for_precision_full(OptimPrecision::Int8),
            clamp_int8_to_fp16(OptimPrecision::Int8),
        );
    }

    #[test]
    fn lists_follow_param_paths_order_with_clamp_and_fallback() {
        let plan = PrecisionPlan {
            params: vec![
                pp("blocks.0.ffn.w_gate", OptimPrecision::Int8, OptimPrecision::Fp16),
                pp("norm.weight", OptimPrecision::Fp32, OptimPrecision::Fp32),
            ],
            total_optim_bytes: 0,
            baseline_fp32_bytes: 0,
        };
        let paths = vec![
            "norm".to_string(),                // matches "norm.weight" via suffix strip
            "embed".to_string(),               // unmatched -> F32,F32
            "blocks.0.ffn.w_gate".to_string(), // exact -> FP16,FP16 (INT8 clamped)
        ];
        let (m, v) = build_dtype_lists(&plan, &paths);
        assert_eq!(m, vec![DTYPE_F32, DTYPE_F32, DTYPE_FP16]);
        assert_eq!(v, vec![DTYPE_F32, DTYPE_F32, DTYPE_FP16]);
    }

    #[test]
    fn gate_requires_all_five_conditions() {
        assert!(precision_active(true, true, true, true, true));
        assert!(!precision_active(true, true, true, true, false)); // wrapped_path_active=false (kept as defense-in-depth post-S5; no production caller passes false today)
        assert!(!precision_active(true, true, true, false, true)); // not Deferred
        assert!(!precision_active(false, true, true, true, true)); // not Full
        assert!(!precision_active(true, false, true, true, true)); // empty plan
        assert!(!precision_active(true, true, false, true, true)); // no weights
    }

    fn override_layer(idx: u32, name: &str, m_bits: u8, v_bits: u8) -> crate::wggo_overrides::PerLayerOverride {
        crate::wggo_overrides::PerLayerOverride {
            layer_index: idx,
            layer_name: name.into(),
            active_heads: 8,
            requested_csha_level: None,
            adapter_rank: 0,
            adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
            fase_fused: true,
            packing_mode: 0,
            shard_factor: 0,
            optim_m_bits: m_bits,
            optim_v_bits: v_bits,
        }
    }

    #[test]
    fn overrides_all_32_bits_yield_none() {
        let o = crate::wggo_overrides::WggoOverrides {
            per_layer: vec![override_layer(0, "blocks.0", 32, 32)],
        };
        assert!(build_dtype_lists_from_overrides(&o, &["m.blocks.0.w".into()]).is_none());
    }

    #[test]
    fn overrides_sub32_bits_map_params_by_layer_prefix() {
        // blocks.0 keeps FP32 moments; blocks.1 gets FP16 (8-bit clamps to
        // FP16 in v1). Nested container ("m.encoder.blocks.1.w") and
        // model-var-prefixed paths both join; the unmatched embedding
        // param stays F32.
        let o = crate::wggo_overrides::WggoOverrides {
            per_layer: vec![
                override_layer(0, "blocks.0", 32, 32),
                override_layer(1, "blocks.1", 8, 16),
            ],
        };
        let paths: Vec<String> = vec![
            "m.blocks.0.w".into(),
            "m.encoder.blocks.1.w".into(),
            "m.embed".into(),
        ];
        let (m, v) = build_dtype_lists_from_overrides(&o, &paths).unwrap();
        assert_eq!(m, vec![DTYPE_F32, DTYPE_FP16, DTYPE_F32]);
        assert_eq!(v, vec![DTYPE_F32, DTYPE_FP16, DTYPE_F32]);
    }

    #[test]
    fn overrides_mixed_m_v_bits_are_independent() {
        let o = crate::wggo_overrides::WggoOverrides {
            per_layer: vec![override_layer(0, "blocks.0", 32, 16)],
        };
        let (m, v) =
            build_dtype_lists_from_overrides(&o, &["m.blocks.0.w".into()]).unwrap();
        assert_eq!(m, vec![DTYPE_F32]);
        assert_eq!(v, vec![DTYPE_FP16]);
    }

    fn dl(codes: &[u16]) -> Vec<u16> {
        codes.to_vec()
    }

    #[test]
    fn arbitration_cpdt_only_applies_even_when_wggo_bits_absent() {
        // Regression: WGGO can be active on a train block for reasons that
        // have nothing to do with moment precision (structural pruning,
        // CSHA fusion, packing), in which case `wggo_bits` is None. An
        // independent CPDT PrecisionPlan must still apply, regardless of
        // the --wggo-moment-precision opt-in, exactly as it did before
        // WGGO plans existed.
        let cpdt = Some((dl(&[DTYPE_FP16, DTYPE_F32]), dl(&[DTYPE_FP16, DTYPE_F32])));
        for opt_in in [false, true] {
            match arbitrate_moment_precision(None, cpdt.clone(), opt_in) {
                MomentPrecisionArbitration::CpdtOnly(m, v) => {
                    assert_eq!(m, vec![DTYPE_FP16, DTYPE_F32]);
                    assert_eq!(v, vec![DTYPE_FP16, DTYPE_F32]);
                }
                _ => panic!(
                    "expected CpdtOnly regardless of opt_in={opt_in}, got a different variant (discriminant only, no Debug)"
                ),
            }
        }
    }

    #[test]
    fn arbitration_wggo_decision_without_opt_in_drops_both_sources() {
        let wggo = Some((dl(&[DTYPE_FP16]), dl(&[DTYPE_FP16])));
        let cpdt = Some((dl(&[DTYPE_FP16]), dl(&[DTYPE_FP16])));
        assert!(matches!(
            arbitrate_moment_precision(wggo, cpdt, false),
            MomentPrecisionArbitration::NotLoweredNoOptIn
        ));
    }

    #[test]
    fn arbitration_wggo_only_requires_opt_in() {
        let wggo = Some((dl(&[DTYPE_FP16]), dl(&[DTYPE_F32])));
        assert!(matches!(
            arbitrate_moment_precision(wggo.clone(), None, false),
            MomentPrecisionArbitration::NotLoweredNoOptIn
        ));
        match arbitrate_moment_precision(wggo, None, true) {
            MomentPrecisionArbitration::WggoOnly(m, v) => {
                assert_eq!(m, vec![DTYPE_FP16]);
                assert_eq!(v, vec![DTYPE_F32]);
            }
            _ => panic!("expected WggoOnly when opted in"),
        }
    }

    #[test]
    fn arbitration_merge_prefers_f32_conservatively() {
        // param 0: WGGO says FP16, CPDT says F32 -> merged F32 (conservative).
        // param 1: both say FP16 -> merged FP16.
        let wggo = Some((dl(&[DTYPE_FP16, DTYPE_FP16]), dl(&[DTYPE_FP16, DTYPE_FP16])));
        let cpdt = Some((dl(&[DTYPE_F32, DTYPE_FP16]), dl(&[DTYPE_F32, DTYPE_FP16])));
        match arbitrate_moment_precision(wggo, cpdt, true) {
            MomentPrecisionArbitration::Merged(m, v) => {
                assert_eq!(m, vec![DTYPE_F32, DTYPE_FP16]);
                assert_eq!(v, vec![DTYPE_F32, DTYPE_FP16]);
            }
            _ => panic!("expected Merged"),
        }
    }

    #[test]
    fn arbitration_neither_source_is_inactive() {
        assert!(matches!(
            arbitrate_moment_precision(None, None, true),
            MomentPrecisionArbitration::Inactive
        ));
        assert!(matches!(
            arbitrate_moment_precision(None, None, false),
            MomentPrecisionArbitration::Inactive
        ));
    }
}
