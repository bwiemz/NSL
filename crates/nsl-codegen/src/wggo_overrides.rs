//! Consumer-facing translation of WGGO's AppliedPlan.
//!
//! Every downstream consumer (CSHA, WRGA, FASE, Prune, Sharding) reads the
//! fields it cares about and ignores the rest. WGGO's decisions are
//! "preferences, not mandates" — consumers check hardware feasibility
//! locally and may downgrade, recording diagnostics for the decision
//! explainer.

use crate::cfie_persistent::FusionLevel;
use serde::Serialize;

/// Consumer diagnostic: one override that was rejected or adjusted.
/// Shared across CSHA, WRGA, and future consumers of `WggoOverrides`.
/// `requested` and `applied` are stringified at the producer so the
/// decision explainer has a uniform wire format across consumers.
#[derive(Debug, Clone, Serialize)]
pub struct OverrideDiagnostic {
    pub layer_index: u32,
    pub layer_name: String,
    pub reason: OverrideRejectReason,
    pub requested: String,
    pub applied: String,
}

/// Why a consumer refused or adjusted a WGGO override.
#[derive(Debug, Clone, Serialize)]
pub enum OverrideRejectReason {
    // CSHA:
    SmemBudgetExceeded { actual_kb: u32, limit_kb: u32 },
    // WRGA:
    RankClampedToBounds { r_min: u32, r_max: u32 },
    RankForbiddenByWggo,
    BudgetExceededDowngraded { original_rank: u32, final_rank: u32 },
    // CPDT:
    /// WGGO's recommended shard factor doesn't divide world_size.
    ShardFactorIncompatibleWithWorldSize { recommended: u32, world_size: u32 },
    /// Memory budget required more aggressive sharding than WGGO recommended.
    ShardFactorOverriddenByMemory { recommended: u32, applied: u32 },
}

#[derive(Debug, Clone, Serialize)]
pub struct WggoOverrides {
    pub per_layer: Vec<PerLayerOverride>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerLayerOverride {
    pub layer_index: u32,
    pub layer_name: String,

    // CSHA fields (consumed by this plan):
    pub active_heads: u32,
    pub requested_csha_level: Option<FusionLevel>,

    // Future-consumer fields (populated now so the struct shape is stable;
    // their wiring plans land later without signature churn):
    pub adapter_rank: u64,   // WRGA
    pub fase_fused: bool,    // FASE
    pub packing_mode: u8,    // PCA / fusion
    /// CPDT: per-layer ZeRO shard factor recommendation. Aggregated to a
    /// single global recommendation via `WggoOverrides::min_shard_factor`.
    /// `0` is the uninitialized sentinel — meaning "no recommendation."
    pub shard_factor: u32,
}

impl WggoOverrides {
    /// Translate from WGGO's AppliedPlan.  Pure data transform.
    pub fn from_applied(applied: &crate::wggo_apply::AppliedPlan) -> Self {
        Self {
            per_layer: applied
                .layers
                .iter()
                .map(|l| PerLayerOverride {
                    layer_index: l.layer_index,
                    layer_name: l.layer_name.clone(),
                    active_heads: l.active_heads,
                    requested_csha_level: map_csha_level(l.csha_level),
                    adapter_rank: l.adapter_rank,
                    fase_fused: l.fase_fused,
                    packing_mode: l.packing_mode,
                    shard_factor: l.shard_factor,
                })
                .collect(),
        }
    }

    /// O(n) lookup by layer_index. `n` is layer count (32-128 in practice),
    /// so linear scan is fine.
    pub fn find(&self, layer_index: u32) -> Option<&PerLayerOverride> {
        self.per_layer.iter().find(|o| o.layer_index == layer_index)
    }

    /// Minimum shard factor across layers with a meaningful recommendation
    /// (`shard_factor > 0`).  Returns `None` when no override carries a
    /// recommendation.
    ///
    /// Uses MIN because the most-sensitive layer constrains the global
    /// decision: ZeRO's collective ops require uniform sharding, so a layer
    /// that only tolerates factor 2 forces every layer to factor 2.
    pub fn min_shard_factor(&self) -> Option<u32> {
        self.per_layer
            .iter()
            .filter_map(|p| if p.shard_factor > 0 { Some(p.shard_factor) } else { None })
            .min()
    }

    /// Look up the override for the layer containing a given projection
    /// name.  WRGA's `SpectralAnalysis.name` is per-projection (e.g.
    /// `blocks.0.attn.wq`); WGGO's `layer_name` is per-layer (e.g. `blocks.0`).
    /// All projections within a layer share its override.
    ///
    /// Match is **prefix + dot boundary**: `blocks.1` matches `blocks.1.attn.wq`
    /// but NOT `blocks.10.attn.wq`.
    pub fn find_by_layer_containing(&self, projection_name: &str) -> Option<&PerLayerOverride> {
        self.per_layer.iter().find(|o| {
            let ln = &o.layer_name;
            projection_name == ln
                || (projection_name.starts_with(ln.as_str())
                    && projection_name[ln.len()..].starts_with('.'))
        })
    }
}

fn map_csha_level(raw: u8) -> Option<FusionLevel> {
    match raw {
        0 => Some(FusionLevel::None),
        1 => Some(FusionLevel::Level1),
        2 => Some(FusionLevel::Level2),
        3 => Some(FusionLevel::Level3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_apply::{AppliedLayer, AppliedPlan};
    use crate::wggo_dp::LayerDecision as CoarseDecision;

    fn sample_applied() -> AppliedPlan {
        AppliedPlan {
            layers: vec![AppliedLayer {
                layer_index: 7,
                layer_name: "blocks.7".to_string(),
                coarse: CoarseDecision::KeepFull,
                pipeline_stage: 0,
                shard_factor: 1,
                active_heads: 4,
                ffn_width: 2048,
                csha_level: 2,
                adapter_rank: 16,
                optim_m_bits: 8,
                optim_v_bits: 16,
                fase_fused: true,
                packing_mode: 1,
                estimated_us: 42.0,
                param_bytes: 0,
                activation_bytes: 0,
            }],
            total_us: 42.0,
            peak_memory_bytes: 0,
        }
    }

    #[test]
    fn from_applied_translation_maps_all_fields() {
        let applied = sample_applied();
        let over = WggoOverrides::from_applied(&applied);
        assert_eq!(over.per_layer.len(), 1);
        let p = &over.per_layer[0];
        assert_eq!(p.layer_index, 7);
        assert_eq!(p.layer_name, "blocks.7");
        assert_eq!(p.active_heads, 4);
        assert_eq!(p.requested_csha_level, Some(FusionLevel::Level2));
        assert_eq!(p.adapter_rank, 16);
        assert!(p.fase_fused);
        assert_eq!(p.packing_mode, 1);
    }

    #[test]
    fn find_returns_record_by_layer_index() {
        let over = WggoOverrides::from_applied(&sample_applied());
        assert!(over.find(7).is_some());
        assert!(over.find(99).is_none());
    }

    #[test]
    fn map_csha_level_covers_all_valid_raws() {
        assert_eq!(map_csha_level(0), Some(FusionLevel::None));
        assert_eq!(map_csha_level(1), Some(FusionLevel::Level1));
        assert_eq!(map_csha_level(2), Some(FusionLevel::Level2));
        assert_eq!(map_csha_level(3), Some(FusionLevel::Level3));
        assert_eq!(map_csha_level(99), None);
    }

    #[test]
    fn override_diagnostic_has_string_requested_and_applied() {
        let d = OverrideDiagnostic {
            layer_index: 3,
            layer_name: "blocks.3".into(),
            reason: OverrideRejectReason::RankClampedToBounds { r_min: 2, r_max: 16 },
            requested: "32".to_string(),
            applied: "16".to_string(),
        };
        assert_eq!(d.requested, "32");
        assert_eq!(d.applied, "16");
    }

    #[test]
    fn override_reject_reason_covers_csha_and_wrga_variants() {
        let _csha = OverrideRejectReason::SmemBudgetExceeded { actual_kb: 52, limit_kb: 48 };
        let _wrga_clamp = OverrideRejectReason::RankClampedToBounds { r_min: 2, r_max: 16 };
        let _wrga_forbid = OverrideRejectReason::RankForbiddenByWggo;
        let _wrga_budget = OverrideRejectReason::BudgetExceededDowngraded {
            original_rank: 16, final_rank: 8,
        };
    }

    fn overrides_with_layers(layers: &[(&str, u32)]) -> WggoOverrides {
        WggoOverrides {
            per_layer: layers
                .iter()
                .enumerate()
                .map(|(i, (name, rank))| PerLayerOverride {
                    layer_index: i as u32,
                    layer_name: name.to_string(),
                    active_heads: 8,
                    requested_csha_level: None,
                    adapter_rank: *rank as u64,
                    fase_fused: false,
                    packing_mode: 0,
                    shard_factor: 0,
                })
                .collect(),
        }
    }

    #[test]
    fn find_by_layer_containing_matches_projections_in_layer() {
        let over = overrides_with_layers(&[("blocks.0", 8)]);
        assert_eq!(over.find_by_layer_containing("blocks.0.attn.wq").unwrap().adapter_rank, 8);
        assert_eq!(over.find_by_layer_containing("blocks.0.attn.wk").unwrap().adapter_rank, 8);
        assert_eq!(over.find_by_layer_containing("blocks.0.mlp.fc1").unwrap().adapter_rank, 8);
    }

    #[test]
    fn find_by_layer_containing_respects_dot_boundary() {
        let over = overrides_with_layers(&[("blocks.1", 4)]);
        assert!(over.find_by_layer_containing("blocks.10.attn.wq").is_none(),
            "'blocks.1' must NOT match 'blocks.10.attn.wq' — dot boundary required");
        assert_eq!(over.find_by_layer_containing("blocks.1.attn.wq").unwrap().adapter_rank, 4);
    }

    #[test]
    fn find_by_layer_containing_matches_bare_layer_name() {
        let over = overrides_with_layers(&[("blocks.2", 6)]);
        assert_eq!(over.find_by_layer_containing("blocks.2").unwrap().adapter_rank, 6);
    }

    #[test]
    fn find_by_layer_containing_returns_none_on_no_match() {
        let over = overrides_with_layers(&[("blocks.0", 8)]);
        assert!(over.find_by_layer_containing("blocks.99.attn.wq").is_none());
    }

    fn overrides_with_shards(shards: &[u32]) -> WggoOverrides {
        WggoOverrides {
            per_layer: shards.iter().enumerate().map(|(i, &s)| PerLayerOverride {
                layer_index: i as u32,
                layer_name: format!("blocks.{i}"),
                active_heads: 8,
                requested_csha_level: None,
                adapter_rank: 0,
                fase_fused: false,
                packing_mode: 0,
                shard_factor: s,
            }).collect(),
        }
    }

    #[test]
    fn min_shard_factor_returns_none_when_all_zero() {
        let over = overrides_with_shards(&[0, 0, 0]);
        assert!(over.min_shard_factor().is_none());
    }

    #[test]
    fn min_shard_factor_returns_minimum_of_nonzero_values() {
        let over = overrides_with_shards(&[8, 4, 2, 8]);
        assert_eq!(over.min_shard_factor(), Some(2));
    }

    #[test]
    fn min_shard_factor_skips_zero_sentinels() {
        let over = overrides_with_shards(&[8, 0, 4, 0]);
        assert_eq!(over.min_shard_factor(), Some(4));
    }

    #[test]
    fn shard_factor_reject_variants_construct() {
        let _incompat = OverrideRejectReason::ShardFactorIncompatibleWithWorldSize {
            recommended: 4, world_size: 2,
        };
        let _mem = OverrideRejectReason::ShardFactorOverriddenByMemory {
            recommended: 2, applied: 8,
        };
    }
}
