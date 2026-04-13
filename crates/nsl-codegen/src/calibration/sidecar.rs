//! Sidecar JSON format produced by the calibration subprocess and
//! consumed by each hook's decoder.  See spec §4.2.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

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
    pub num_samples_used: u32,
    #[serde(with = "byte_map_b64")]
    pub hooks: BTreeMap<String, Vec<u8>>,
}

/// Current on-disk format version.  Bump when a compatibility-breaking
/// change ships; caches with a lower version are discarded on load.
pub const SIDECAR_VERSION: u32 = 1;

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
            version: 1,
            checkpoint_sha256: "abc123".into(),
            calibration_data_sha256: "def456".into(),
            hook_set_sha256: "ghi789".into(),
            num_samples_used: 200,
            hooks,
        }
    }

    #[test]
    fn sidecar_roundtrips_through_json() {
        let original = payload();
        let json = serde_json::to_string(&original).unwrap();
        let parsed: Sidecar = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, 1);
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
}
