//! CPDT precision-adaptive optimizer EXECUTION bridge.
//!
//! Turns the per-parameter `PrecisionPlan` into ordered dtype codes aligned
//! with `param_paths` (the same ordered name list the codegen uses to build
//! `param_list`).
//!
//! # What a reduced-precision moment is stored as
//!
//! The plan's tiers name FP16 and INT8 (`OptimPrecision`). Neither is what
//! gets stored, because neither can hold Adam state for gradients of
//! realistic size:
//!
//! - **Second moment `v`: always f32.** `v` accumulates `(1-beta2) * g^2`.
//!   For `g = 1e-3` that increment is 1e-9, below half of f16's smallest
//!   subnormal (2.98e-8), so it stores as zero and stays zero for any
//!   element whose `|g|` never exceeds 5.5e-3 -- which is nearly every
//!   element of a model with a global gradient norm near 1. With `v = 0`
//!   the update is `m_hat / eps`: a constant gradient of 1e-3 moved a
//!   weight 10x further in 200 steps than f32 moments did. BF16 has the
//!   range but not the resolution: its relative spacing (2^-8) is coarser
//!   than `1 - beta2` (2^-10), so round-to-nearest stalls the average. A
//!   sound 16-bit `v` needs stochastic rounding or a transformed encoding;
//!   until one exists `v` is not narrowed.
//! - **First moment `m`: BF16.** `m` is of the size of the gradient itself,
//!   so in f16 it is subnormal below 6.1e-5 and zero below 3e-8. BF16 keeps
//!   f32's exponent range. Its spacing (2^-8 relative) is fine against
//!   `1 - beta1` (0.1): round-to-nearest holds `m` within a few percent of
//!   the f32 average for a constant gradient, and without bias for a noisy
//!   one.
//!
//! So a reduced tier saves 2 of the 8 moment bytes per element (25%), not
//! the 4 to 6 the tier table implies. [`stored_moment_bytes`] is the single
//! statement of that, used by the plan's byte accounting.

use crate::cpdt_tier_apply::{OptimPrecision, ParamPrecision, PrecisionPlan};

/// Runtime dtype codes. Not a mirror any more: the emitted optimizer-state
/// storage dtype IS the ABI's tag (`nsl_abi::wire::dtype`, roadmap A3),
/// which the runtime re-exports as its canonical `DTYPE_*`.
pub use nsl_abi::wire::dtype::{DTYPE_BF16, DTYPE_F32};

/// Storage dtype of a first moment whose plan asks for `p`: f32 when the
/// plan keeps it at full precision, BF16 for any reduced tier.
pub fn first_moment_storage(p: OptimPrecision) -> u16 {
    match p {
        OptimPrecision::Fp32 => DTYPE_F32,
        OptimPrecision::Fp16 | OptimPrecision::Int8 => DTYPE_BF16,
    }
}

/// Storage dtype of a second moment: f32 whatever the plan asks for (see
/// the module doc).
pub fn second_moment_storage(_p: OptimPrecision) -> u16 {
    DTYPE_F32
}

/// Bytes per element the pair `(m, v)` occupies as stored.
pub fn stored_moment_bytes(m: OptimPrecision, v: OptimPrecision) -> u64 {
    let width = |code: u16| if code == DTYPE_F32 { 4 } else { 2 };
    width(first_moment_storage(m)) + width(second_moment_storage(v))
}

/// Find the precision for a param path. Matches CPDT plan names against the
/// codegen path with light normalization: exact match, or a trailing
/// `.weight`/`.bias` stripped from either side. Unmatched -> None.
fn name_matches_path(name: &str, path: &str) -> bool {
    name == path
        || name.strip_suffix(".weight").map(|s| s == path).unwrap_or(false)
        || name.strip_suffix(".bias").map(|s| s == path).unwrap_or(false)
        || path.strip_suffix(".weight").map(|s| s == name).unwrap_or(false)
        || path.strip_suffix(".bias").map(|s| s == name).unwrap_or(false)
}

fn precision_for_path<'a>(plan: &'a PrecisionPlan, path: &str) -> Option<&'a ParamPrecision> {
    // Exact spelling first, then with the LEADING MODEL-VARIABLE SEGMENT
    // stripped. `param_paths` are enumerated as `<model var>.<field path>`
    // (`enumerate_model_tensor_paths`: `m.blocks.0.w`), while the plan's
    // names come from the weight file's keys, which carry no variable
    // prefix (`blocks.0.w`) — so the exact match alone joined NOTHING on
    // any real model, and every moment silently stayed F32 via the
    // unmatched-path fallback below. The WGGO-side join
    // (`build_dtype_lists_from_overrides`) handles the identical mismatch
    // via `wggo_graph::layer_prefix`; this is the same accommodation for
    // the per-param plan. One segment only: stripping repeatedly would let
    // `blocks.0.w` on some sibling container match a plan entry for a
    // different parameter.
    if let Some(pp) = plan.params.iter().find(|p| name_matches_path(&p.name, path)) {
        return Some(pp);
    }
    let stripped = path.split_once('.').map(|(_, rest)| rest)?;
    if let Some(pp) = plan.params.iter().find(|p| name_matches_path(&p.name, stripped)) {
        return Some(pp);
    }
    // Last, a plan name that carries a container prefix of its own, as a
    // checkpoint exported from another framework does (`model.blocks.0.w`
    // for the field path `blocks.0.w`): the name ends with the field path
    // at a dot boundary. Taken only when exactly one plan name does, since
    // two (`enc.blocks.0.w` and `dec.blocks.0.w`) cannot be told apart from
    // here. Such names used to join nothing and their parameters silently
    // kept FP32 moments.
    let mut prefixed = plan.params.iter().filter(|p| name_has_prefix_of(&p.name, stripped));
    match (prefixed.next(), prefixed.next()) {
        (Some(pp), None) => Some(pp),
        _ => None,
    }
}

/// True when `name`, or `name` without a trailing `.weight` / `.bias`, is
/// `<prefix>.<field_path>` for some non-empty prefix.
fn name_has_prefix_of(name: &str, field_path: &str) -> bool {
    let ends = |n: &str| {
        n.len() > field_path.len() + 1
            && n.ends_with(field_path)
            && n.as_bytes()[n.len() - field_path.len() - 1] == b'.'
    };
    ends(name)
        || name.strip_suffix(".weight").is_some_and(ends)
        || name.strip_suffix(".bias").is_some_and(ends)
}

/// The plan entries that join none of `param_paths`: tensors the weights
/// file names that this train block's model does not have (or that
/// [`precision_for_path`] cannot tell apart). Their precision decisions are
/// not applied; the caller reports them so that is not silent.
pub fn unjoined_plan_names<'a>(plan: &'a PrecisionPlan, param_paths: &[String]) -> Vec<&'a str> {
    let joined: std::collections::HashSet<*const ParamPrecision> = param_paths
        .iter()
        .filter_map(|p| precision_for_path(plan, p))
        .map(|pp| pp as *const ParamPrecision)
        .collect();
    plan.params
        .iter()
        .filter(|pp| !joined.contains(&(*pp as *const ParamPrecision)))
        .map(|pp| pp.name.as_str())
        .collect()
}

/// How many of this block's parameters the precision plan's names actually
/// join (see [`precision_for_path`]). The weights-only path has
/// no `cpdt_sensitivity::validate` (there is no AppliedPlan to check the
/// WeightMap against), so a wrong checkpoint would otherwise sail through
/// as "active: 0 moment buffer(s)" — activation with no effect, the exact
/// shape the join defect had. A non-empty plan joining ZERO params is the
/// certainly-wrong case the caller refuses on; a partial join is reported
/// by the caller from [`unjoined_plan_names`].
pub fn joined_param_count(plan: &PrecisionPlan, param_paths: &[String]) -> usize {
    param_paths
        .iter()
        .filter(|p| precision_for_path(plan, p).is_some())
        .count()
}

/// Build (m_dtype_codes, v_dtype_codes) in `param_paths` order. Each code is
/// a storage dtype from [`first_moment_storage`] / [`second_moment_storage`].
/// Unmatched paths -> (F32, F32).
pub fn build_dtype_lists(plan: &PrecisionPlan, param_paths: &[String]) -> (Vec<u16>, Vec<u16>) {
    let mut m = Vec::with_capacity(param_paths.len());
    let mut v = Vec::with_capacity(param_paths.len());
    for path in param_paths {
        match precision_for_path(plan, path) {
            Some(pp) => {
                m.push(first_moment_storage(pp.m_precision));
                v.push(second_moment_storage(pp.v_precision));
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
/// A sub-32-bit first moment is stored BF16; the second moment is stored
/// f32 whatever the plan carries (see the module doc).
pub fn build_dtype_lists_from_overrides(
    overrides: &crate::wggo_overrides::WggoOverrides,
    param_paths: &[String],
) -> Option<(Vec<u16>, Vec<u16>)> {
    if !overrides.per_layer.iter().any(|l| l.optim_m_bits < 32) {
        return None;
    }
    let m_storage = |bits: u8| if bits < 32 { DTYPE_BF16 } else { DTYPE_F32 };
    let mut m = Vec::with_capacity(param_paths.len());
    let mut v = Vec::with_capacity(param_paths.len());
    for path in param_paths {
        let layer = crate::wggo_graph::layer_prefix(path)
            .and_then(|lp| overrides.per_layer.iter().find(|l| l.layer_name == lp));
        match layer {
            Some(l) => {
                m.push(m_storage(l.optim_m_bits));
                v.push(DTYPE_F32);
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
    if m.iter().all(|&c| c == DTYPE_F32) {
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
    fn reduced_first_moment_is_bf16_and_second_moment_is_never_narrowed() {
        assert_eq!(first_moment_storage(OptimPrecision::Fp32), DTYPE_F32);
        assert_eq!(first_moment_storage(OptimPrecision::Fp16), DTYPE_BF16);
        assert_eq!(first_moment_storage(OptimPrecision::Int8), DTYPE_BF16);
        for p in [OptimPrecision::Fp32, OptimPrecision::Fp16, OptimPrecision::Int8] {
            assert_eq!(second_moment_storage(p), DTYPE_F32);
        }
    }

    #[test]
    fn stored_bytes_match_the_storage_dtypes() {
        // 4 + 4 at full precision; a reduced tier saves the first moment's
        // two bytes and nothing else.
        assert_eq!(stored_moment_bytes(OptimPrecision::Fp32, OptimPrecision::Fp32), 8);
        for (m, v) in [
            (OptimPrecision::Fp16, OptimPrecision::Fp32),
            (OptimPrecision::Int8, OptimPrecision::Fp16),
            (OptimPrecision::Int8, OptimPrecision::Int8),
        ] {
            assert_eq!(stored_moment_bytes(m, v), 6);
        }
    }

    /// The reason `v` is never narrowed, as arithmetic: the per-step
    /// increment of the second moment for a gradient of 1e-3 is below half
    /// of f16's smallest subnormal, so an f16 `v` stores it as zero.
    #[test]
    fn an_f16_second_moment_cannot_hold_a_realistic_increment() {
        let g = 1e-3_f64;
        let increment = (1.0 - 0.999) * g * g;
        let smallest_f16_subnormal = 2f64.powi(-24);
        assert!(increment < smallest_f16_subnormal / 2.0);
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
            "blocks.0.ffn.w_gate".to_string(), // exact -> BF16 m, F32 v
        ];
        let (m, v) = build_dtype_lists(&plan, &paths);
        assert_eq!(m, vec![DTYPE_F32, DTYPE_F32, DTYPE_BF16]);
        assert_eq!(v, vec![DTYPE_F32, DTYPE_F32, DTYPE_F32]);
    }

    /// A weights file whose keys carry a container prefix of their own
    /// (`model.blocks.0.w`, as an exported checkpoint spells them) joins the
    /// model's parameters. It used to join none, and those parameters
    /// silently kept FP32 moments.
    #[test]
    fn a_prefixed_weights_file_name_joins_its_parameter() {
        let plan = PrecisionPlan {
            params: vec![
                pp("model.blocks.0.w", OptimPrecision::Fp16, OptimPrecision::Fp32),
                pp("model.blocks.1.w.weight", OptimPrecision::Fp16, OptimPrecision::Fp32),
                pp("model.embed", OptimPrecision::Fp32, OptimPrecision::Fp32),
            ],
            total_optim_bytes: 0,
            baseline_fp32_bytes: 0,
        };
        let paths = vec![
            "m.embed".to_string(),
            "m.blocks.0.w".to_string(),
            "m.blocks.1.w".to_string(),
            "m.final_norm".to_string(),
        ];
        let (m, v) = build_dtype_lists(&plan, &paths);
        assert_eq!(m, vec![DTYPE_F32, DTYPE_BF16, DTYPE_BF16, DTYPE_F32]);
        assert_eq!(v, vec![DTYPE_F32; 4]);
        assert_eq!(joined_param_count(&plan, &paths), 3);
        assert!(unjoined_plan_names(&plan, &paths).is_empty());
    }

    /// The prefix match is a suffix test at a dot boundary, never a bare
    /// suffix: `xblocks.0.w` is not `blocks.0.w` under a prefix.
    #[test]
    fn a_name_that_merely_ends_like_the_path_does_not_join() {
        let plan = PrecisionPlan {
            params: vec![pp("xblocks.0.w", OptimPrecision::Fp16, OptimPrecision::Fp32)],
            total_optim_bytes: 0,
            baseline_fp32_bytes: 0,
        };
        let paths = vec!["m.blocks.0.w".to_string()];
        assert_eq!(build_dtype_lists(&plan, &paths).0, vec![DTYPE_F32]);
        assert_eq!(unjoined_plan_names(&plan, &paths), vec!["xblocks.0.w"]);
    }

    /// Two prefixed names for one field path cannot be told apart, so
    /// neither is applied, and both are reported.
    #[test]
    fn two_prefixed_names_for_one_parameter_join_neither() {
        let plan = PrecisionPlan {
            params: vec![
                pp("enc.blocks.0.w", OptimPrecision::Fp16, OptimPrecision::Fp32),
                pp("dec.blocks.0.w", OptimPrecision::Fp16, OptimPrecision::Fp32),
            ],
            total_optim_bytes: 0,
            baseline_fp32_bytes: 0,
        };
        let paths = vec!["m.blocks.0.w".to_string()];
        assert_eq!(build_dtype_lists(&plan, &paths).0, vec![DTYPE_F32]);
        assert_eq!(
            unjoined_plan_names(&plan, &paths),
            vec!["enc.blocks.0.w", "dec.blocks.0.w"]
        );
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
        // blocks.0 keeps FP32 moments; blocks.1 gets a BF16 first moment and
        // an f32 second moment. Nested container ("m.encoder.blocks.1.w") and
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
        assert_eq!(m, vec![DTYPE_F32, DTYPE_BF16, DTYPE_F32]);
        assert_eq!(v, vec![DTYPE_F32, DTYPE_F32, DTYPE_F32]);
    }

    #[test]
    fn overrides_asking_only_for_a_narrow_second_moment_change_nothing() {
        // A plan that keeps `m` at 32 bits and asks for a 16-bit `v` has
        // nothing left to lower: `v` is stored f32 regardless.
        let o = crate::wggo_overrides::WggoOverrides {
            per_layer: vec![override_layer(0, "blocks.0", 32, 16)],
        };
        assert!(build_dtype_lists_from_overrides(&o, &["m.blocks.0.w".into()]).is_none());
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
        let cpdt = Some((dl(&[DTYPE_BF16, DTYPE_F32]), dl(&[DTYPE_BF16, DTYPE_F32])));
        for opt_in in [false, true] {
            match arbitrate_moment_precision(None, cpdt.clone(), opt_in) {
                MomentPrecisionArbitration::CpdtOnly(m, v) => {
                    assert_eq!(m, vec![DTYPE_BF16, DTYPE_F32]);
                    assert_eq!(v, vec![DTYPE_BF16, DTYPE_F32]);
                }
                _ => panic!(
                    "expected CpdtOnly regardless of opt_in={opt_in}, got a different variant (discriminant only, no Debug)"
                ),
            }
        }
    }

    #[test]
    fn arbitration_wggo_decision_without_opt_in_drops_both_sources() {
        let wggo = Some((dl(&[DTYPE_BF16]), dl(&[DTYPE_BF16])));
        let cpdt = Some((dl(&[DTYPE_BF16]), dl(&[DTYPE_BF16])));
        assert!(matches!(
            arbitrate_moment_precision(wggo, cpdt, false),
            MomentPrecisionArbitration::NotLoweredNoOptIn
        ));
    }

    #[test]
    fn arbitration_wggo_only_requires_opt_in() {
        let wggo = Some((dl(&[DTYPE_BF16]), dl(&[DTYPE_F32])));
        assert!(matches!(
            arbitrate_moment_precision(wggo.clone(), None, false),
            MomentPrecisionArbitration::NotLoweredNoOptIn
        ));
        match arbitrate_moment_precision(wggo, None, true) {
            MomentPrecisionArbitration::WggoOnly(m, v) => {
                assert_eq!(m, vec![DTYPE_BF16]);
                assert_eq!(v, vec![DTYPE_F32]);
            }
            _ => panic!("expected WggoOnly when opted in"),
        }
    }

    #[test]
    fn arbitration_merge_prefers_f32_conservatively() {
        // param 0: WGGO says FP16, CPDT says F32 -> merged F32 (conservative).
        // param 1: both say FP16 -> merged FP16.
        let wggo = Some((dl(&[DTYPE_BF16, DTYPE_BF16]), dl(&[DTYPE_BF16, DTYPE_BF16])));
        let cpdt = Some((dl(&[DTYPE_F32, DTYPE_BF16]), dl(&[DTYPE_F32, DTYPE_BF16])));
        match arbitrate_moment_precision(wggo, cpdt, true) {
            MomentPrecisionArbitration::Merged(m, v) => {
                assert_eq!(m, vec![DTYPE_F32, DTYPE_BF16]);
                assert_eq!(v, vec![DTYPE_F32, DTYPE_BF16]);
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
