//! FASE — Fused Accumulation + Step + Epilogue.
//!
//! Core analysis pass: given a train-block configuration (optimizer, grad
//! accumulation count, clipping setting), produce a [`FasePlan`] that the
//! backward-codegen stage consumes.  The plan describes:
//!
//!   1. Whether to rewrite the backward at all (`accumulation > 1`).
//!   2. Whether to run in **Deferred** mode (first-moment accumulation) or
//!      **Full** mode (fall back to a standard gradient buffer — used when
//!      the optimizer doesn't match the FASE invariants, e.g. Lion).
//!   3. The mathematical update rule for the optimizer, already
//!      specialised to the accumulation count.
//!   4. The two-phase structure when `grad_clip` is enabled.
//!
//! The driver is pure — no state, no I/O — and produces the same output
//! given the same input.  PTX emission for the fused backward is a separate
//! concern handled by `fase_optimizer.rs` (update-rule codegen) and
//! `fase_memory.rs` (per-layer slot scheduling).

use serde::Serialize;

/// Supported optimizer algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FaseOptimizer {
    AdamW,
    Adam,
    Sgd,
    SgdMomentum,
    Lion,
    Unknown,
}

impl FaseOptimizer {
    pub fn parse(name: &str) -> Self {
        match name.to_ascii_lowercase().as_str() {
            "adamw" => FaseOptimizer::AdamW,
            "adam" => FaseOptimizer::Adam,
            "sgd" => FaseOptimizer::Sgd,
            "sgd_momentum" | "sgdmomentum" => FaseOptimizer::SgdMomentum,
            "lion" => FaseOptimizer::Lion,
            _ => FaseOptimizer::Unknown,
        }
    }
}

/// Execution mode chosen by the FASE planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FaseMode {
    /// Accumulation count is 1 — no rewrite needed.
    Passthrough,
    /// First-moment accumulator only (no separate gradient buffer).
    /// Available for AdamW/Adam and approximates their v_t via per-step
    /// squared-gradient averaging.
    Deferred,
    /// Full-gradient-buffer fallback.  Used when the optimizer's update
    /// rule can't be expressed as an incremental accumulator update (e.g.
    /// Lion's sign operation).  Still saves via per-layer memory
    /// scheduling even though the gradient buffer itself survives.
    FullBuffer,
}

/// User-visible configuration extracted from the `train(…)` block.
#[derive(Debug, Clone)]
pub struct FaseConfig {
    pub optimizer: FaseOptimizer,
    /// Number of micro-batches per optimizer step.  1 = no accumulation.
    pub accumulation: u32,
    /// Global gradient-norm clipping threshold.  `None` = no clipping.
    pub grad_clip: Option<f64>,
    /// Optimizer learning rate.
    pub lr: f64,
    /// AdamW/Adam β₁.
    pub beta1: f64,
    /// AdamW/Adam β₂.
    pub beta2: f64,
    /// AdamW/Adam ε.
    pub eps: f64,
    /// Weight decay (AdamW / SGD).
    pub weight_decay: f64,
    /// SGD momentum.
    pub momentum: f64,
    /// Whether AdamW's v_t is allowed to use the per-micro-batch average
    /// approximation described in Section 3.2 of the CFTP paper.  Default
    /// `true`; set to `false` to force FullBuffer mode.
    pub allow_v_approx: bool,
}

impl Default for FaseConfig {
    fn default() -> Self {
        Self {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 1,
            grad_clip: None,
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            allow_v_approx: true,
        }
    }
}

/// Backward-pass phases FASE emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BackwardPhase {
    /// Standard backward + passthrough optimizer (no accumulation).
    Standard,
    /// Accumulation micro-batch `i < N-1`: compute grad, fuse into
    /// accumulator, free grad.
    AccumulateOnly,
    /// Final micro-batch, single-phase (no clipping): compute grad,
    /// complete accumulator, run fused optimizer step, free grad.
    FinalFused,
    /// Final micro-batch, two-phase (clipping enabled):
    ///   Phase A: compute grad → add to m_partial → running-norm²
    ///   Phase B: read m_partial → apply clip → optimizer step
    FinalTwoPhase,
}

/// Per-parameter update-rule descriptor.  This is a recipe the downstream
/// emitter uses to generate PTX / Cranelift IR.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateRecipe {
    pub optimizer: FaseOptimizer,
    /// `1 / N` scaled into the accumulator per micro-batch.
    pub accum_scale: f64,
    /// Whether the second moment uses the batch-variance-aware
    /// approximation (per-step squared-gradient average).
    pub v_uses_approx: bool,
    /// Cached constants we fold into the generated code.
    pub one_minus_beta1: f64,
    pub one_minus_beta2: f64,
    pub lr: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
}

/// Aggregate FASE plan.
#[derive(Debug, Clone, Serialize)]
pub struct FasePlan {
    pub mode: FaseMode,
    pub accumulation: u32,
    pub backward_phases: Vec<BackwardPhase>,
    pub recipe: UpdateRecipe,
    /// Whether the caller must emit a two-phase backward for clipping.
    pub two_phase_clip: bool,
    /// Diagnostic explaining the chosen mode.
    pub rationale: String,
    /// Per-layer mode overrides. `None` ⇒ every layer uses `mode` (today's
    /// behavior, byte-identical). `Some(v)` ⇒ layer `i` uses `v[i]`; `mode`
    /// becomes an informational default.
    #[serde(default)]
    pub per_layer_mode: Option<Vec<FaseMode>>,
    /// WGGO recommendations that FASE had to clamp due to global feasibility.
    /// Empty when no overrides were supplied or all were applied verbatim.
    #[serde(default)]
    pub override_diagnostics: Vec<crate::wggo_overrides::OverrideDiagnostic>,
}

impl FasePlan {
    /// Returns the sequence of backward-phase tags in execution order.
    pub fn phase_sequence(&self) -> &[BackwardPhase] {
        &self.backward_phases
    }

    pub fn is_active(&self) -> bool {
        !matches!(self.mode, FaseMode::Passthrough)
    }

    /// Returns the mode for layer `i`. Falls back to `self.mode` when no
    /// per-layer override vector is present or when `i` is out of range.
    pub fn mode_for_layer(&self, i: usize) -> FaseMode {
        self.per_layer_mode
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(self.mode)
    }
}

/// Run the FASE planner.
pub fn plan(cfg: &FaseConfig) -> FasePlan {
    let accumulation = cfg.accumulation.max(1);

    // Accumulation count of 1 → no rewrite.
    if accumulation == 1 {
        return FasePlan {
            mode: FaseMode::Passthrough,
            accumulation: 1,
            backward_phases: vec![BackwardPhase::Standard],
            recipe: make_recipe(cfg, accumulation, false),
            two_phase_clip: false,
            rationale: "accumulation=1 — no FASE rewrite needed".to_string(),
            per_layer_mode: None,
            override_diagnostics: Vec::new(),
        };
    }

    // Decide Deferred vs FullBuffer based on optimizer.  Lion / Unknown →
    // FullBuffer because their update rules don't decompose into
    // incremental accumulator updates.
    let (mode, v_approx, rationale) = match (cfg.optimizer, cfg.allow_v_approx) {
        (FaseOptimizer::AdamW, true) | (FaseOptimizer::Adam, true) => (
            FaseMode::Deferred,
            true,
            format!(
                "{} supports deferred first-moment accumulation with batch-variance v approximation",
                optimizer_name(cfg.optimizer)
            ),
        ),
        (FaseOptimizer::AdamW, false) | (FaseOptimizer::Adam, false) => (
            FaseMode::FullBuffer,
            false,
            "v_t approximation disabled by config — fall back to full gradient buffer".to_string(),
        ),
        (FaseOptimizer::Sgd, _) | (FaseOptimizer::SgdMomentum, _) => (
            // SGD with momentum is trivially deferred — momentum is a linear
            // accumulator, identical to first-moment handling.  Plain SGD
            // has no accumulator state at all.
            FaseMode::Deferred,
            false,
            "SGD accumulator reduces to momentum/zero-state — deferred update is exact".to_string(),
        ),
        (FaseOptimizer::Lion, _) => (
            FaseMode::FullBuffer,
            false,
            "Lion uses sign(m + g) which cannot be decomposed incrementally — full buffer"
                .to_string(),
        ),
        (FaseOptimizer::Unknown, _) => (
            FaseMode::FullBuffer,
            false,
            "unrecognised optimizer — falling back to full gradient buffer".to_string(),
        ),
    };

    // Build the per-micro-batch phase sequence.
    let mut phases = Vec::with_capacity(accumulation as usize);
    for i in 0..accumulation {
        if i < accumulation - 1 {
            phases.push(BackwardPhase::AccumulateOnly);
        } else if cfg.grad_clip.is_some() {
            phases.push(BackwardPhase::FinalTwoPhase);
        } else {
            phases.push(BackwardPhase::FinalFused);
        }
    }

    FasePlan {
        mode,
        accumulation,
        backward_phases: phases,
        recipe: make_recipe(cfg, accumulation, v_approx),
        two_phase_clip: cfg.grad_clip.is_some(),
        rationale,
        per_layer_mode: None,
        override_diagnostics: Vec::new(),
    }
}

/// WGGO-aware variant of [`plan`]. Given a per-layer `fase_fused` vector,
/// returns a plan with `per_layer_mode` populated and any infeasible
/// overrides clamped + logged in `override_diagnostics`.
///
/// Semantics:
/// - Empty input (`wggo_fused_per_layer.is_empty()`) → same as `plan(cfg)`.
///   `per_layer_mode` stays `None`; no diagnostics.
/// - Passthrough global (accumulation=1) → overrides ignored. FASE isn't
///   rewriting the backward, so per-layer variation is meaningless.
///   `per_layer_mode` stays `None`; no diagnostics.
/// - Deferred global → overrides feasible either way; `per_layer_mode` is
///   `Some(vec![Deferred | FullBuffer, ...])` mapped from the input.
/// - FullBuffer global (Lion/Unknown/allow_v_approx=false) → `Deferred`
///   requests clamp to `FullBuffer` with a `FaseModeInfeasible` diagnostic.
///   `FullBuffer` requests apply verbatim.
pub fn plan_with_overrides(
    cfg: &FaseConfig,
    wggo_fused_per_layer: &[bool],
) -> FasePlan {
    let mut p = plan(cfg);

    if wggo_fused_per_layer.is_empty() || p.mode == FaseMode::Passthrough {
        return p;
    }

    let mut per_layer = Vec::with_capacity(wggo_fused_per_layer.len());
    let mut diagnostics = Vec::new();

    for (i, &fused) in wggo_fused_per_layer.iter().enumerate() {
        let requested = if fused { FaseMode::Deferred } else { FaseMode::FullBuffer };
        let applied = match (p.mode, requested) {
            (FaseMode::FullBuffer, FaseMode::Deferred) => {
                diagnostics.push(crate::wggo_overrides::OverrideDiagnostic {
                    layer_index: i as u32,
                    layer_name: format!("layer_{i}"),
                    reason: crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
                        optimizer: cfg.optimizer,
                        global_mode: p.mode,
                    },
                    requested: "Deferred".into(),
                    applied: "FullBuffer".into(),
                });
                FaseMode::FullBuffer
            }
            _ => requested,
        };
        per_layer.push(applied);
    }

    p.per_layer_mode = Some(per_layer);
    p.override_diagnostics = diagnostics;
    p
}

fn optimizer_name(o: FaseOptimizer) -> &'static str {
    match o {
        FaseOptimizer::AdamW => "AdamW",
        FaseOptimizer::Adam => "Adam",
        FaseOptimizer::Sgd => "SGD",
        FaseOptimizer::SgdMomentum => "SGD+Momentum",
        FaseOptimizer::Lion => "Lion",
        FaseOptimizer::Unknown => "unknown",
    }
}

fn make_recipe(cfg: &FaseConfig, n: u32, v_approx: bool) -> UpdateRecipe {
    let inv_n = 1.0 / n as f64;
    UpdateRecipe {
        optimizer: cfg.optimizer,
        accum_scale: inv_n,
        v_uses_approx: v_approx,
        one_minus_beta1: 1.0 - cfg.beta1,
        one_minus_beta2: 1.0 - cfg.beta2,
        lr: cfg.lr,
        eps: cfg.eps,
        weight_decay: cfg.weight_decay,
        beta1: cfg.beta1,
        beta2: cfg.beta2,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulation_one_is_passthrough() {
        let plan = plan(&FaseConfig {
            accumulation: 1,
            ..Default::default()
        });
        assert_eq!(plan.mode, FaseMode::Passthrough);
        assert_eq!(plan.backward_phases, vec![BackwardPhase::Standard]);
        assert!(!plan.is_active());
    }

    #[test]
    fn adamw_with_accumulation_is_deferred() {
        let plan = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        });
        assert_eq!(plan.mode, FaseMode::Deferred);
        assert!(plan.recipe.v_uses_approx);
        assert!(plan.is_active());
    }

    #[test]
    fn lion_falls_back_to_full_buffer() {
        let plan = plan(&FaseConfig {
            optimizer: FaseOptimizer::Lion,
            accumulation: 4,
            ..Default::default()
        });
        assert_eq!(plan.mode, FaseMode::FullBuffer);
        assert!(!plan.recipe.v_uses_approx);
    }

    #[test]
    fn sgd_is_deferred_and_exact() {
        let plan = plan(&FaseConfig {
            optimizer: FaseOptimizer::Sgd,
            accumulation: 4,
            ..Default::default()
        });
        assert_eq!(plan.mode, FaseMode::Deferred);
        assert!(!plan.recipe.v_uses_approx);
    }

    #[test]
    fn clip_triggers_two_phase_final() {
        let plan = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            grad_clip: Some(1.0),
            ..Default::default()
        });
        assert!(plan.two_phase_clip);
        // The final phase must be two-phase, intermediate ones accumulate-only.
        assert_eq!(plan.backward_phases[0], BackwardPhase::AccumulateOnly);
        assert_eq!(
            plan.backward_phases[plan.backward_phases.len() - 1],
            BackwardPhase::FinalTwoPhase
        );
    }

    #[test]
    fn accum_scale_matches_accumulation() {
        let plan = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 8,
            ..Default::default()
        });
        assert!((plan.recipe.accum_scale - 0.125).abs() < 1e-12);
    }

    #[test]
    fn v_approx_disable_forces_full_buffer() {
        let plan = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            allow_v_approx: false,
            ..Default::default()
        });
        assert_eq!(plan.mode, FaseMode::FullBuffer);
    }

    #[test]
    fn optimizer_parse_covers_known_names() {
        assert_eq!(FaseOptimizer::parse("AdamW"), FaseOptimizer::AdamW);
        assert_eq!(FaseOptimizer::parse("adam"), FaseOptimizer::Adam);
        assert_eq!(FaseOptimizer::parse("lion"), FaseOptimizer::Lion);
        assert_eq!(FaseOptimizer::parse("nonsense"), FaseOptimizer::Unknown);
    }

    #[test]
    fn phase_count_matches_accumulation() {
        let plan = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 7,
            ..Default::default()
        });
        assert_eq!(plan.phase_sequence().len(), 7);
        for p in &plan.backward_phases[..6] {
            assert_eq!(*p, BackwardPhase::AccumulateOnly);
        }
    }

    #[test]
    fn grad_clip_with_adamw_now_returns_deferred() {
        // Reverses the item #1 Task 5 temporary downgrade.  Item #3 emits
        // two-phase clip codegen, so grad_clip + Deferred is now valid.
        let cfg = FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: Some(1.0),
            allow_v_approx: true,
            ..Default::default()
        };
        let p = plan(&cfg);
        assert_eq!(p.mode, FaseMode::Deferred);
        assert!(p.two_phase_clip);
        assert!(matches!(
            p.backward_phases.last(),
            Some(BackwardPhase::FinalTwoPhase)
        ));
    }

    #[test]
    fn grad_clip_absent_keeps_deferred_for_adamw() {
        let cfg = FaseConfig {
            accumulation: 4,
            optimizer: FaseOptimizer::AdamW,
            grad_clip: None,
            allow_v_approx: true,
            ..Default::default()
        };
        let p = plan(&cfg);
        assert_eq!(p.mode, FaseMode::Deferred);
    }

    #[test]
    fn fase_plan_exposes_per_layer_mode_none_by_default() {
        let p = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        });
        assert!(p.per_layer_mode.is_none());
        assert!(p.override_diagnostics.is_empty());
    }

    #[test]
    fn mode_for_layer_falls_back_to_global_when_none() {
        let p = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        });
        assert_eq!(p.mode_for_layer(0), FaseMode::Deferred);
        assert_eq!(p.mode_for_layer(99), FaseMode::Deferred);
    }

    #[test]
    fn mode_for_layer_reads_per_layer_vector_when_some() {
        let mut p = plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        });
        p.per_layer_mode = Some(vec![FaseMode::FullBuffer, FaseMode::Deferred]);
        assert_eq!(p.mode_for_layer(0), FaseMode::FullBuffer);
        assert_eq!(p.mode_for_layer(1), FaseMode::Deferred);
        // Out-of-range falls back to global.
        assert_eq!(p.mode_for_layer(2), FaseMode::Deferred);
    }

    #[test]
    fn plan_with_overrides_empty_input_matches_plan() {
        let cfg = FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        };
        let baseline = plan(&cfg);
        let overridden = plan_with_overrides(&cfg, &[]);
        assert_eq!(overridden.mode, baseline.mode);
        assert_eq!(overridden.accumulation, baseline.accumulation);
        assert!(overridden.per_layer_mode.is_none());
        assert!(overridden.override_diagnostics.is_empty());
    }

    #[test]
    fn plan_with_overrides_passthrough_ignores_all() {
        let cfg = FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 1,
            ..Default::default()
        };
        let p = plan_with_overrides(&cfg, &[true, false, true]);
        assert_eq!(p.mode, FaseMode::Passthrough);
        assert!(p.per_layer_mode.is_none());
        assert!(p.override_diagnostics.is_empty());
    }

    #[test]
    fn plan_with_overrides_adamw_deferred_mixes_modes() {
        let cfg = FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        };
        let p = plan_with_overrides(&cfg, &[true, false, true, false]);
        assert_eq!(p.mode, FaseMode::Deferred);
        assert_eq!(
            p.per_layer_mode,
            Some(vec![
                FaseMode::Deferred,
                FaseMode::FullBuffer,
                FaseMode::Deferred,
                FaseMode::FullBuffer,
            ])
        );
        assert!(p.override_diagnostics.is_empty());
    }

    #[test]
    fn plan_with_overrides_lion_global_clamps_deferred_requests() {
        let cfg = FaseConfig {
            optimizer: FaseOptimizer::Lion,
            accumulation: 4,
            ..Default::default()
        };
        let p = plan_with_overrides(&cfg, &[true, false]);
        assert_eq!(p.mode, FaseMode::FullBuffer);
        assert_eq!(
            p.per_layer_mode,
            Some(vec![FaseMode::FullBuffer, FaseMode::FullBuffer])
        );
        assert_eq!(p.override_diagnostics.len(), 1);
        let diag = &p.override_diagnostics[0];
        assert_eq!(diag.layer_index, 0);
        assert_eq!(diag.requested, "Deferred");
        assert_eq!(diag.applied, "FullBuffer");
        assert!(matches!(
            diag.reason,
            crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
                optimizer: FaseOptimizer::Lion,
                global_mode: FaseMode::FullBuffer,
            }
        ));
    }

    #[test]
    fn plan_with_overrides_allow_v_approx_false_clamps() {
        let cfg = FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            allow_v_approx: false,
            ..Default::default()
        };
        let p = plan_with_overrides(&cfg, &[true]);
        assert_eq!(p.mode, FaseMode::FullBuffer);
        assert_eq!(p.per_layer_mode, Some(vec![FaseMode::FullBuffer]));
        assert_eq!(p.override_diagnostics.len(), 1);
    }

    #[test]
    fn fase_mode_infeasible_diagnostic_has_stderr_shape() {
        // Pins the FaseModeInfeasible diagnostic's field shape so the
        // stmt.rs stderr formatter has a stable contract. If stderr format
        // drifts, this test won't catch it — rely on CSHA/WRGA/CPDT
        // renderer symmetry and manual verification for the formatter.
        let cfg = FaseConfig {
            optimizer: FaseOptimizer::Lion,
            accumulation: 4,
            ..Default::default()
        };
        let p = plan_with_overrides(&cfg, &[true]);
        let diag = &p.override_diagnostics[0];

        assert_eq!(diag.layer_index, 0);
        assert_eq!(diag.requested.as_str(), "Deferred");
        assert_eq!(diag.applied.as_str(), "FullBuffer");
        match &diag.reason {
            crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
                optimizer,
                global_mode,
            } => {
                assert_eq!(*optimizer, FaseOptimizer::Lion);
                assert_eq!(*global_mode, FaseMode::FullBuffer);
            }
            _ => panic!("expected FaseModeInfeasible"),
        }
    }
}
