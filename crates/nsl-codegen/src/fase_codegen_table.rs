//! FASE Codegen Phase 2: per-parameter mode lookup table.
//!
//! Builds a compile-time `Vec<u8>` aligned with `param_paths` that the
//! backward loops in `stmt.rs` consult at runtime via a single byte
//! load. When WGGO is inactive (`plan.per_layer_mode.is_none()`) or
//! no overrides were supplied, returns `None` so callers fall back to
//! the existing global-mode dispatch (byte-identical to pre-Phase-2).
//!
//! Naming-convention note: `enumerate_model_tensor_paths` prefixes
//! every path with the model variable name (e.g. `m.blocks.0.attn.wq`),
//! but WGGO `layer_name` is bare (e.g. `blocks.0`). This helper strips
//! the `model_var_name.` prefix before calling
//! `WggoOverrides::find_by_layer_containing`. If the prefix isn't
//! present, the strip is a no-op via `strip_prefix(...).unwrap_or(path)`.

use crate::fase::{FaseMode, FasePlan};
use crate::wggo_overrides::WggoOverrides;

/// Per-parameter FASE mode encoding for the runtime dispatch table.
/// Encoded as `u8` for compact `.rodata` storage and single-byte loads.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamMode {
    Passthrough = 0,
    Deferred    = 1,
    FullBuffer  = 2,
}

impl From<FaseMode> for ParamMode {
    fn from(m: FaseMode) -> Self {
        match m {
            FaseMode::Passthrough => ParamMode::Passthrough,
            FaseMode::Deferred    => ParamMode::Deferred,
            FaseMode::FullBuffer  => ParamMode::FullBuffer,
        }
    }
}

/// Build a per-parameter mode table aligned with `param_paths`.
///
/// `model_var_name` is the model variable name that prefixes every
/// entry in `param_paths` (per `enumerate_model_tensor_paths`'s
/// recursion shape). It is stripped before each path is matched
/// against WGGO's layer registry.
///
/// Returns `None` when no WGGO per-layer dispatch is active — caller
/// should skip rodata emission and use the global-mode dispatch path.
pub fn build_param_mode_table(
    param_paths: &[String],
    model_var_name: &str,
    plan: &FasePlan,
    overrides: Option<&WggoOverrides>,
) -> Option<Vec<u8>> {
    let per_layer = plan.per_layer_mode.as_ref()?;
    let o = overrides?;

    let global = ParamMode::from(plan.mode) as u8;
    let prefix = format!("{model_var_name}.");

    let mut modes = Vec::with_capacity(param_paths.len());
    for path in param_paths {
        let bare = path.strip_prefix(&prefix).unwrap_or(path.as_str());
        let m = match o.find_by_layer_containing(bare) {
            Some(layer) => {
                let layer_idx = layer.layer_index as usize;
                per_layer
                    .get(layer_idx)
                    .copied()
                    .map(|fm| ParamMode::from(fm) as u8)
                    .unwrap_or(global)
            }
            None => global,
        };
        modes.push(m);
    }
    Some(modes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::FaseOptimizer;
    use crate::wggo_apply::{AppliedLayer, AppliedPlan};
    use crate::wggo_dp::CoarseDecision;

    fn applied_layer(idx: u32, name: &str) -> AppliedLayer {
        AppliedLayer {
            layer_index: idx,
            layer_name: name.into(),
            coarse: CoarseDecision::KeepFull,
            pipeline_stage: 0,
            shard_factor: 1,
            shard_grads: 1,
            shard_optim: 1,
            active_heads: 8,
            ffn_width: 4096,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 32,
            optim_v_bits: 32,
            fase_fused: false,
            packing_mode: 0,
            estimated_us: 1.0,
            param_bytes: 0,
            activation_bytes: 0,
        }
    }

    fn make_overrides(names: &[&str]) -> WggoOverrides {
        let plan = AppliedPlan {
            layers: names.iter().enumerate().map(|(i, n)| applied_layer(i as u32, n)).collect(),
            total_us: 0.0,
            peak_memory_bytes: 0,
        };
        WggoOverrides::from_applied(&plan)
    }

    fn deferred_plan() -> FasePlan {
        crate::fase::plan(&crate::fase::FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        })
    }

    #[test]
    fn returns_none_when_plan_has_no_per_layer_mode() {
        let p = deferred_plan();
        assert!(p.per_layer_mode.is_none());
        let paths = vec!["m.blocks.0.wq".into()];
        let o = make_overrides(&["blocks.0"]);
        assert!(build_param_mode_table(&paths, "m", &p, Some(&o)).is_none());
    }

    #[test]
    fn returns_none_when_overrides_absent() {
        let mut p = deferred_plan();
        p.per_layer_mode = Some(vec![FaseMode::Deferred]);
        let paths = vec!["m.blocks.0.wq".into()];
        assert!(build_param_mode_table(&paths, "m", &p, None).is_none());
    }

    #[test]
    fn maps_param_paths_via_layer_prefix_after_strip() {
        let mut p = deferred_plan();
        p.per_layer_mode = Some(vec![
            FaseMode::Deferred,
            FaseMode::FullBuffer,
            FaseMode::Deferred,
            FaseMode::FullBuffer,
        ]);
        // Paths carry the `m.` model-variable prefix, as
        // enumerate_model_tensor_paths emits.
        let paths: Vec<String> = vec![
            "m.blocks.0.wq", "m.blocks.0.wk", "m.blocks.0.wv",
            "m.blocks.1.wq", "m.blocks.1.wk", "m.blocks.1.wv",
            "m.blocks.2.wq", "m.blocks.2.wk", "m.blocks.2.wv",
            "m.blocks.3.wq", "m.blocks.3.wk", "m.blocks.3.wv",
        ].into_iter().map(String::from).collect();
        let o = make_overrides(&["blocks.0", "blocks.1", "blocks.2", "blocks.3"]);
        let modes = build_param_mode_table(&paths, "m", &p, Some(&o)).unwrap();
        assert_eq!(modes, vec![1,1,1, 2,2,2, 1,1,1, 2,2,2]);
    }

    #[test]
    fn unmatched_param_falls_back_to_global_mode() {
        let mut p = deferred_plan();
        p.per_layer_mode = Some(vec![FaseMode::Deferred]);
        // Global mode is Deferred (AdamW + accumulation=4); "m.embedding"
        // strips to "embedding" which doesn't match WGGO layer "blocks.0",
        // so it falls back to global Deferred (1).
        let paths: Vec<String> = vec!["m.embedding".into(), "m.blocks.0.wq".into()];
        let o = make_overrides(&["blocks.0"]);
        let modes = build_param_mode_table(&paths, "m", &p, Some(&o)).unwrap();
        assert_eq!(modes, vec![1, 1]);
    }

    #[test]
    fn from_fase_mode_round_trips() {
        assert_eq!(ParamMode::from(FaseMode::Passthrough) as u8, 0);
        assert_eq!(ParamMode::from(FaseMode::Deferred)    as u8, 1);
        assert_eq!(ParamMode::from(FaseMode::FullBuffer)  as u8, 2);
    }

    #[test]
    fn strip_is_noop_when_path_lacks_prefix() {
        let mut p = deferred_plan();
        p.per_layer_mode = Some(vec![FaseMode::Deferred]);
        // Path is already bare (no `m.` prefix). Strip returns the path
        // unchanged; matcher still works.
        let paths: Vec<String> = vec!["blocks.0.wq".into()];
        let o = make_overrides(&["blocks.0"]);
        let modes = build_param_mode_table(&paths, "m", &p, Some(&o)).unwrap();
        assert_eq!(modes, vec![1]);
    }
}
