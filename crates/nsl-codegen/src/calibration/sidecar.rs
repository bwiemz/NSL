//! Sidecar JSON format produced by the calibration subprocess and
//! consumed by each hook's decoder.  See spec §4.2.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Per-layer WGGO head-importance gradients accumulated during calibration.
///
/// Keyed by layer name (e.g. `"model.layers.0"`).  The outer
/// `WggoHeadGradients` wrapper exists so the sidecar field can be
/// `Option`-typed and omitted entirely when WGGO was not run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WggoHeadGradients {
    /// Maps each layer name to its per-head gradient summary.
    pub by_layer: BTreeMap<String, PerLayerGradient>,
}

/// Accumulated gradient signal for one transformer layer's attention heads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerLayerGradient {
    /// One score per attention head.  Higher ⇒ more important during
    /// calibration.  Length equals the number of heads in that layer.
    pub per_head_score: Vec<f32>,
    /// Number of calibration batches that contributed to these scores.
    pub batches_observed: u32,
}

/// Per-run serialised output written next to the checkpoint.  `hooks`
/// uses `BTreeMap` so serialisation is deterministic and diff-friendly.
///
/// Each hook owns the byte format under its key.  The sidecar itself
/// knows nothing about hook semantics — it only ferries bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sidecar {
    pub version: u32,
    pub checkpoint_sha256: String,
    pub calibration_data_sha256: String,
    pub hook_set_sha256: String,
    /// Full cache-key digest covering all checkpoints, the calibration
    /// data, the enabled hook set, and the sample/batch knobs.  When
    /// non-empty this is the canonical cache-invalidation signal —
    /// the individual `*_sha256` fields above are diagnostic only.
    /// Empty string for v1 sidecars (legacy compatibility).
    #[serde(default)]
    pub cache_key_digest: String,
    pub num_samples_used: u32,
    #[serde(with = "byte_map_b64")]
    pub hooks: BTreeMap<String, Vec<u8>>,
    /// WGGO head-importance gradients produced by the WGGO calibration
    /// hook.  `None` when WGGO was not enabled or the sidecar predates
    /// this field (backward-compatible: missing key deserialises as `None`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wggo_head_gradients: Option<WggoHeadGradients>,
}

/// Current on-disk format version.  Bump when a compatibility-breaking
/// change ships; caches with a lower version are discarded on load.
pub const SIDECAR_VERSION: u32 = 2;

/// serde helper: encode `Vec<u8>` values as base64 so the JSON stays
/// human-inspectable and embeddable in text tooling without escaping.
mod byte_map_b64 {
    use std::collections::BTreeMap;

    use base64::{engine::general_purpose::STANDARD as B64, Engine};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        map: &BTreeMap<String, Vec<u8>>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let encoded: BTreeMap<&String, String> =
            map.iter().map(|(k, v)| (k, B64.encode(v))).collect();
        encoded.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<String, Vec<u8>>, D::Error> {
        let encoded = BTreeMap::<String, String>::deserialize(d)?;
        encoded
            .into_iter()
            .map(|(k, v)| {
                B64.decode(v.as_bytes())
                    .map(|bytes| (k, bytes))
                    .map_err(serde::de::Error::custom)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload() -> Sidecar {
        let mut hooks = std::collections::BTreeMap::new();
        hooks.insert("identity".to_string(), vec![1u8, 2, 3, 4]);
        hooks.insert("wggo_head_gradient".to_string(), vec![0xff, 0xee]);
        Sidecar {
            version: SIDECAR_VERSION,
            checkpoint_sha256: "abc123".into(),
            calibration_data_sha256: "def456".into(),
            hook_set_sha256: "ghi789".into(),
            cache_key_digest: String::new(),
            num_samples_used: 200,
            hooks,
            wggo_head_gradients: None,
        }
    }

    #[test]
    fn sidecar_roundtrips_through_json() {
        let original = payload();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Sidecar = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, SIDECAR_VERSION);
        assert_eq!(parsed.hooks.get("identity"), Some(&vec![1, 2, 3, 4]));
        assert_eq!(
            parsed.hooks.get("wggo_head_gradient"),
            Some(&vec![0xff, 0xee])
        );
    }

    #[test]
    fn sidecar_preserves_key_order_via_btreemap() {
        let s = payload();
        let json = serde_json::to_string(&s).unwrap();
        let i = json.find("identity").unwrap();
        let w = json.find("wggo_head_gradient").unwrap();
        assert!(i < w, "identity should precede wggo_head_gradient in JSON");
    }

    #[test]
    fn missing_hook_key_returns_none() {
        let s = payload();
        assert!(s.hooks.get("unknown_hook").is_none());
    }

    #[test]
    fn sidecar_with_wggo_head_gradients_roundtrips() {
        let mut by_layer = std::collections::BTreeMap::new();
        by_layer.insert(
            "model.layers.0".to_string(),
            PerLayerGradient {
                per_head_score: vec![1.0f32, 2.0, 3.0, 4.0],
                batches_observed: 8,
            },
        );
        let mut sc = payload();
        sc.wggo_head_gradients = Some(WggoHeadGradients { by_layer });

        let bytes = serde_json::to_vec(&sc).unwrap();
        let back: Sidecar = serde_json::from_slice(&bytes).unwrap();

        let grads = back.wggo_head_gradients.as_ref().expect("grads present");
        let entry = &grads.by_layer["model.layers.0"];
        assert_eq!(entry.per_head_score, vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(entry.batches_observed, 8);
    }

    #[test]
    fn older_sidecar_without_wggo_grads_deserializes_cleanly() {
        let sc = payload();
        // Confirm wggo_head_gradients is None on a sidecar that never set it.
        let bytes = serde_json::to_vec(&sc).unwrap();
        let back: Sidecar = serde_json::from_slice(&bytes).unwrap();
        assert!(
            back.wggo_head_gradients.is_none(),
            "sidecar without wggo_head_gradients must deserialize with field = None"
        );
    }

    #[test]
    fn sidecar_without_wggo_field_in_json_deserializes_cleanly() {
        // Simulate a pre-field sidecar JSON (no wggo_head_gradients key at all).
        let json = r#"{
            "version": 2,
            "checkpoint_sha256": "abc",
            "calibration_data_sha256": "def",
            "hook_set_sha256": "ghi",
            "cache_key_digest": "",
            "num_samples_used": 10,
            "hooks": {}
        }"#;
        let sc: Sidecar = serde_json::from_str(json).unwrap();
        assert!(
            sc.wggo_head_gradients.is_none(),
            "older sidecar JSON without wggo_head_gradients key must yield None"
        );
    }
}
