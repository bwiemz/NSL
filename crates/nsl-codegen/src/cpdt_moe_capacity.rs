//! CPDT §4.1 — roofline-derived MoE capacity-factor override.
//!
//! Paper §4.1: `capacity_factor = max(1.0, roofline_slack × top_k / n_experts)`.
//! The pure formula already lives in `cpdt_expert::capacity_factor`; this module
//! wires it into the compile flow by overriding `MoeInfo.capacity_factor` (which
//! the `moe_dispatch` lowering reads) for every `@moe` layer when CPDT is on.
//!
//! Non-WGGO call site (same blocker-sidestep as `cpdt_expert_prune`): reachable
//! with `--cpdt`, no `--wggo` required. The pass only runs in `CpdtMode::Full`
//! and is a deterministic no-op otherwise. The override is *compile-time
//! constant* — no runtime dispatch overhead.

use std::collections::HashMap;

use crate::cpdt::CpdtMode;
use crate::moe::MoeInfo;

/// Per-MoE outcome of the capacity-factor pass.
#[derive(Debug, Clone, PartialEq)]
pub enum MoeCapacityOutcome {
    /// `MoeInfo.capacity_factor` was overridden from `prev` to `new` for `layer`.
    Overridden { layer: String, prev: f32, new: f32 },
    /// `MoeInfo.capacity_factor` already matched the roofline value — no change.
    Unchanged { layer: String, value: f32 },
}

/// Pure formula — paper §4.1 (`cpdt_expert::capacity_factor` already encodes it,
/// re-exposed here as a typed wrapper so the pass + lowering share one number
/// independent of any planner state).
#[inline]
pub fn roofline_capacity(roofline_slack: f64, top_k: u32, n_experts: u32) -> f64 {
    crate::cpdt_expert::capacity_factor(roofline_slack, top_k, n_experts)
}

/// Override `moe_configs[*].capacity_factor` with the roofline-derived value for
/// every MoE layer. No-op unless `cpdt_mode == CpdtMode::Full`. Deterministic
/// iteration order (sorted keys) — matches the prune pass.
pub fn apply_capacity_overrides(
    cpdt_mode: CpdtMode,
    moe_configs: &mut HashMap<String, MoeInfo>,
    roofline_slack: f64,
) -> Vec<MoeCapacityOutcome> {
    if cpdt_mode != CpdtMode::Full {
        return Vec::new();
    }

    let mut keys: Vec<String> = moe_configs.keys().cloned().collect();
    keys.sort();

    let mut outcomes = Vec::with_capacity(keys.len());
    for key in keys {
        let info = moe_configs.get_mut(&key).expect("key from same map");
        let prev = info.capacity_factor;
        let new = roofline_capacity(roofline_slack, info.top_k as u32, info.num_experts as u32) as f32;
        if (prev - new).abs() < f32::EPSILON {
            outcomes.push(MoeCapacityOutcome::Unchanged { layer: key, value: new });
        } else {
            info.capacity_factor = new;
            outcomes.push(MoeCapacityOutcome::Overridden { layer: key, prev, new });
        }
    }
    outcomes
}

/// Render outcomes to stderr (one line per MoE) — matches the prune-pass report
/// convention so the CPDT report is a single coherent stream.
pub fn report_outcomes(outcomes: &[MoeCapacityOutcome]) {
    for o in outcomes {
        match o {
            MoeCapacityOutcome::Overridden { layer, prev, new } => eprintln!(
                "[cpdt] moe '{layer}': capacity_factor override {prev} -> {new} (roofline §4.1)"
            ),
            MoeCapacityOutcome::Unchanged { layer, value } => eprintln!(
                "[cpdt] moe '{layer}': capacity_factor {value} already matches roofline (§4.1, no-op)"
            ),
        }
    }
}

/// Compile-flow entry point.
///
/// STRUCTURAL: invoked from `compile_returning_plan`, NOT under the
/// `wggo_applied` guard. Depends only on `compiler.cpdt_mode` +
/// `compiler.features.moe_configs` + `compiler.cpdt_moe_roofline_slack` — never
/// on WGGO/source-AD. Same blocker-sidestep posture as
/// `cpdt_expert_prune::run_moe_prune_pass`.
pub fn run_moe_capacity_pass(compiler: &mut crate::compiler::Compiler) {
    if compiler.features.moe_configs.is_empty() {
        return;
    }
    let slack = compiler.cpdt_moe_roofline_slack;
    let outcomes = apply_capacity_overrides(
        compiler.cpdt_mode,
        &mut compiler.features.moe_configs,
        slack,
    );
    report_outcomes(&outcomes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(num_experts: usize, top_k: usize, capacity_factor: f32) -> MoeInfo {
        MoeInfo {
            num_experts,
            top_k,
            capacity_factor,
            aux_loss_coeff: 0.0,
            activation: crate::moe::MoeActivation::default(),
            weight_prefix: None,
        }
    }

    #[test]
    fn formula_matches_paper() {
        // §4.1: cap = max(1.0, slack × top_k / n_experts).
        assert_eq!(roofline_capacity(1.0, 2, 8), 1.0); // compute-bound clamps to 1.0
        assert_eq!(roofline_capacity(8.0, 2, 8), 2.0); // memory-bound: 8*2/8 = 2.0
        assert_eq!(roofline_capacity(16.0, 4, 8), 8.0); // 16*4/8 = 8.0
        assert_eq!(roofline_capacity(1.0, 0, 8), 1.0); // top_k=0 -> 0 -> clamp 1.0
        assert_eq!(roofline_capacity(1.0, 2, 0), 1.0); // n_experts=0 -> guard 1.0
    }

    #[test]
    fn override_changes_decorator_value() {
        let mut cfgs = HashMap::new();
        cfgs.insert("m".into(), info(8, 2, 1.25)); // decorator default 1.25
        let outcomes = apply_capacity_overrides(CpdtMode::Full, &mut cfgs, 8.0);
        assert!(matches!(outcomes.as_slice(), [MoeCapacityOutcome::Overridden { .. }]));
        // 8.0 * 2 / 8 = 2.0 — beat the user-specified 1.25.
        assert_eq!(cfgs["m"].capacity_factor, 2.0);
    }

    #[test]
    fn override_unchanged_when_already_roofline() {
        let mut cfgs = HashMap::new();
        cfgs.insert("m".into(), info(8, 2, 1.0)); // already roofline-derived under slack=1.0
        let outcomes = apply_capacity_overrides(CpdtMode::Full, &mut cfgs, 1.0);
        assert!(matches!(outcomes.as_slice(), [MoeCapacityOutcome::Unchanged { value, .. }] if (*value - 1.0).abs() < 1e-6));
        assert_eq!(cfgs["m"].capacity_factor, 1.0);
    }

    #[test]
    fn gated_off_when_not_full() {
        let mut cfgs = HashMap::new();
        cfgs.insert("m".into(), info(8, 2, 1.25));
        let outcomes = apply_capacity_overrides(CpdtMode::ZeroOnly, &mut cfgs, 8.0);
        assert!(outcomes.is_empty());
        // Untouched.
        assert_eq!(cfgs["m"].capacity_factor, 1.25);
    }

    #[test]
    fn deterministic_order_over_multiple_moes() {
        let mut cfgs = HashMap::new();
        cfgs.insert("z_late".into(), info(8, 2, 1.25));
        cfgs.insert("a_early".into(), info(8, 2, 1.25));
        cfgs.insert("m_mid".into(), info(8, 2, 1.25));
        let outcomes = apply_capacity_overrides(CpdtMode::Full, &mut cfgs, 8.0);
        let layers: Vec<&str> = outcomes
            .iter()
            .map(|o| match o {
                MoeCapacityOutcome::Overridden { layer, .. } | MoeCapacityOutcome::Unchanged { layer, .. } => layer.as_str(),
            })
            .collect();
        assert_eq!(layers, vec!["a_early", "m_mid", "z_late"]);
    }
}
