//! Consumer-facing translation of WGGO's AppliedPlan.
//!
//! Every downstream consumer (CSHA, WRGA, FASE, Prune, Sharding) reads the
//! fields it cares about and ignores the rest. WGGO's decisions are
//! "preferences, not mandates" — consumers check hardware feasibility
//! locally and may downgrade, recording diagnostics for the decision
//! explainer.

use crate::cfie_persistent::FusionLevel;
use serde::Serialize;

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
                })
                .collect(),
        }
    }

    /// O(n) lookup by layer_index. `n` is layer count (32-128 in practice),
    /// so linear scan is fine.
    pub fn find(&self, layer_index: u32) -> Option<&PerLayerOverride> {
        self.per_layer.iter().find(|o| o.layer_index == layer_index)
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
}
