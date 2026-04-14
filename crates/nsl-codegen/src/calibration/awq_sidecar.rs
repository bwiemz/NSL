//! Binary layout for AWQ calibration results written under the
//! `"awq_activation_scales"` sidecar key.  See design §8.

use std::collections::BTreeMap;

/// Current format version.  Readers reject other values.
pub const AWQ_SIDECAR_VERSION: u32 = 1;

/// Per-projection calibration result.
#[derive(Debug, Clone, PartialEq)]
pub struct AwqProjectionScales {
    pub name: String,
    pub scales: Vec<f32>,
}

#[derive(Debug)]
pub enum AwqSidecarError {
    TooSmall { need: usize, got: usize },
    UnsupportedVersion { got: u32 },
    Truncated { at: &'static str, offset: usize, need: usize, got: usize },
    BadUtf8 { offset: usize },
}

impl std::fmt::Display for AwqSidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooSmall { need, got } => write!(f, "blob too small: need {need}, got {got}"),
            Self::UnsupportedVersion { got } => write!(f, "unsupported AWQ sidecar version {got} (expected {AWQ_SIDECAR_VERSION})"),
            Self::Truncated { at, offset, need, got } => write!(f, "truncated at {at} offset {offset}: need {need} bytes, got {got}"),
            Self::BadUtf8 { offset } => write!(f, "invalid UTF-8 in projection name at offset {offset}"),
        }
    }
}

impl std::error::Error for AwqSidecarError {}

pub fn serialize(projections: &BTreeMap<String, Vec<f32>>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&AWQ_SIDECAR_VERSION.to_le_bytes());
    out.extend_from_slice(&(projections.len() as u32).to_le_bytes());
    for (name, scales) in projections {
        let nb = name.as_bytes();
        out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        out.extend_from_slice(nb);
        out.extend_from_slice(&(scales.len() as u32).to_le_bytes());
        for v in scales {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

pub fn deserialize(blob: &[u8]) -> Result<Vec<AwqProjectionScales>, AwqSidecarError> {
    if blob.len() < 8 {
        return Err(AwqSidecarError::TooSmall { need: 8, got: blob.len() });
    }
    let version = u32::from_le_bytes(blob[0..4].try_into().unwrap());
    if version != AWQ_SIDECAR_VERSION {
        return Err(AwqSidecarError::UnsupportedVersion { got: version });
    }
    let num_projections = u32::from_le_bytes(blob[4..8].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(num_projections);
    let mut cursor = 8;
    for _ in 0..num_projections {
        if blob.len() < cursor + 4 {
            return Err(AwqSidecarError::Truncated { at: "name_len", offset: cursor, need: 4, got: blob.len() - cursor });
        }
        let name_len = u32::from_le_bytes(blob[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if blob.len() < cursor + name_len {
            return Err(AwqSidecarError::Truncated { at: "name bytes", offset: cursor, need: name_len, got: blob.len() - cursor });
        }
        let name = std::str::from_utf8(&blob[cursor..cursor + name_len])
            .map_err(|_| AwqSidecarError::BadUtf8 { offset: cursor })?
            .to_string();
        cursor += name_len;
        if blob.len() < cursor + 4 {
            return Err(AwqSidecarError::Truncated { at: "channel_count", offset: cursor, need: 4, got: blob.len() - cursor });
        }
        let channel_count = u32::from_le_bytes(blob[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        let scale_bytes = channel_count * 4;
        if blob.len() < cursor + scale_bytes {
            return Err(AwqSidecarError::Truncated { at: "scales", offset: cursor, need: scale_bytes, got: blob.len() - cursor });
        }
        let mut scales = Vec::with_capacity(channel_count);
        for i in 0..channel_count {
            let off = cursor + i * 4;
            scales.push(f32::from_le_bytes(blob[off..off + 4].try_into().unwrap()));
        }
        cursor += scale_bytes;
        out.push(AwqProjectionScales { name, scales });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BTreeMap<String, Vec<f32>> {
        let mut m = BTreeMap::new();
        m.insert("blocks.0.attn.wq".into(), vec![1.0, 2.0, 3.5, 4.0]);
        m.insert("blocks.0.attn.wk".into(), vec![0.1, 0.2]);
        m
    }

    #[test]
    fn roundtrip_preserves_projections_and_scales() {
        let blob = serialize(&sample());
        let parsed = deserialize(&blob).unwrap();
        assert_eq!(parsed.len(), 2);
        // BTreeMap keeps keys sorted: wk < wq
        assert_eq!(parsed[0].name, "blocks.0.attn.wk");
        assert_eq!(parsed[0].scales, vec![0.1, 0.2]);
        assert_eq!(parsed[1].name, "blocks.0.attn.wq");
        assert_eq!(parsed[1].scales, vec![1.0, 2.0, 3.5, 4.0]);
    }

    #[test]
    fn rejects_unknown_version() {
        let mut blob = serialize(&sample());
        blob[0..4].copy_from_slice(&999u32.to_le_bytes());
        let err = deserialize(&blob).unwrap_err();
        assert!(matches!(err, AwqSidecarError::UnsupportedVersion { got: 999 }));
    }

    #[test]
    fn rejects_truncated_name() {
        let mut blob = serialize(&sample());
        blob.truncate(10); // header + partial name_len
        assert!(deserialize(&blob).is_err());
    }

    #[test]
    fn rejects_truncated_scales() {
        let full = serialize(&sample());
        let blob = &full[..full.len() - 2];
        assert!(deserialize(blob).is_err());
    }

    #[test]
    fn empty_projection_map_roundtrips() {
        let empty = BTreeMap::new();
        let blob = serialize(&empty);
        let parsed = deserialize(&blob).unwrap();
        assert_eq!(parsed.len(), 0);
    }
}
