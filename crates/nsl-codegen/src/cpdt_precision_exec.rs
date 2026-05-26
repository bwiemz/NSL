//! CPDT precision-adaptive optimizer EXECUTION bridge (v1: FP16).
//!
//! Turns the per-parameter `PrecisionPlan` into ordered dtype codes aligned
//! with `param_paths` (the same ordered name list the codegen uses to build
//! `param_list`). v1 clamps INT8 tiers to FP16 — `clamp_int8_to_fp16` is the
//! SINGLE removal point for the INT8 ladder step (design doc §10).

use crate::cpdt_tier_apply::{OptimPrecision, ParamPrecision, PrecisionPlan};

/// Runtime dtype codes (mirror `nsl-runtime` `DTYPE_*`: F32=1, FP16=2).
pub const DTYPE_F32: u16 = 1;
pub const DTYPE_FP16: u16 = 2;

/// v1 clamp: map an optimizer-state precision to its v1 storage dtype.
/// FP32 stays FP32; FP16 stays FP16; INT8 clamps to FP16 (v1 defers INT8).
/// This is the SINGLE removal point for the INT8 ladder step.
pub fn clamp_int8_to_fp16(p: OptimPrecision) -> u16 {
    match p {
        OptimPrecision::Fp32 => DTYPE_F32,
        OptimPrecision::Fp16 => DTYPE_FP16,
        OptimPrecision::Int8 => DTYPE_FP16,
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
/// All four conditions are required (design doc §6): CPDT Full mode, a
/// non-empty precision plan, weights present, and FASE Deferred mode.
pub fn precision_active(
    cpdt_mode_is_full: bool,
    has_precision_plan: bool,
    weights_present: bool,
    fase_mode_is_deferred: bool,
) -> bool {
    cpdt_mode_is_full && has_precision_plan && weights_present && fase_mode_is_deferred
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
    fn gate_requires_all_four_conditions() {
        assert!(precision_active(true, true, true, true));
        assert!(!precision_active(true, true, true, false));
        assert!(!precision_active(false, true, true, true));
        assert!(!precision_active(true, false, true, true));
        assert!(!precision_active(true, true, false, true));
    }
}
