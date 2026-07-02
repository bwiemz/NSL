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
}
