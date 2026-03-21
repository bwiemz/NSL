//! Multi-modal data streaming — text, image, audio, video samples.
//!
//! Each shard can contain mixed-modality samples. The runtime groups samples
//! by modality for efficient batching (same-type samples pad/collate together).

// ---------------------------------------------------------------------------
// Sample types
// ---------------------------------------------------------------------------

/// A single training sample that may be any supported modality.
#[derive(Debug, Clone)]
pub enum Sample {
    /// Tokenized text sequence.
    Text { tokens: Vec<i64> },
    /// Raw image data (decoded pixels).
    Image { data: Vec<u8>, width: u32, height: u32, channels: u32 },
    /// Raw audio waveform.
    Audio { samples: Vec<f32>, sample_rate: u32 },
    /// Video frames.
    Video { frames: Vec<Vec<u8>>, width: u32, height: u32, fps: u32 },
}

impl Sample {
    /// Modality tag for grouping.
    pub fn modality(&self) -> Modality {
        match self {
            Sample::Text { .. } => Modality::Text,
            Sample::Image { .. } => Modality::Image,
            Sample::Audio { .. } => Modality::Audio,
            Sample::Video { .. } => Modality::Video,
        }
    }

    /// Serialized byte size (approximate, for memory budgeting).
    pub fn byte_size(&self) -> usize {
        match self {
            Sample::Text { tokens } => tokens.len() * 8,
            Sample::Image { data, .. } => data.len(),
            Sample::Audio { samples, .. } => samples.len() * 4,
            Sample::Video { frames, .. } => frames.iter().map(|f| f.len()).sum(),
        }
    }
}

/// Modality tag for batching decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
}

// ---------------------------------------------------------------------------
// Serialization — encode/decode samples for shard storage
// ---------------------------------------------------------------------------

/// Modality byte tag for wire format.
const TAG_TEXT: u8 = 0;
const TAG_IMAGE: u8 = 1;
const TAG_AUDIO: u8 = 2;
const TAG_VIDEO: u8 = 3;

/// Serialize a sample to bytes for shard storage.
pub fn serialize_sample(sample: &Sample) -> Vec<u8> {
    let mut buf = Vec::new();
    match sample {
        Sample::Text { tokens } => {
            buf.push(TAG_TEXT);
            buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
            for &t in tokens {
                buf.extend_from_slice(&t.to_le_bytes());
            }
        }
        Sample::Image { data, width, height, channels } => {
            buf.push(TAG_IMAGE);
            buf.extend_from_slice(&width.to_le_bytes());
            buf.extend_from_slice(&height.to_le_bytes());
            buf.extend_from_slice(&channels.to_le_bytes());
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
        }
        Sample::Audio { samples, sample_rate } => {
            buf.push(TAG_AUDIO);
            buf.extend_from_slice(&sample_rate.to_le_bytes());
            buf.extend_from_slice(&(samples.len() as u32).to_le_bytes());
            for &s in samples {
                buf.extend_from_slice(&s.to_le_bytes());
            }
        }
        Sample::Video { frames, width, height, fps } => {
            buf.push(TAG_VIDEO);
            buf.extend_from_slice(&width.to_le_bytes());
            buf.extend_from_slice(&height.to_le_bytes());
            buf.extend_from_slice(&fps.to_le_bytes());
            buf.extend_from_slice(&(frames.len() as u32).to_le_bytes());
            for frame in frames {
                buf.extend_from_slice(&(frame.len() as u32).to_le_bytes());
                buf.extend_from_slice(frame);
            }
        }
    }
    buf
}

/// Deserialize a sample from bytes.
pub fn deserialize_sample(buf: &[u8]) -> Option<Sample> {
    if buf.is_empty() { return None; }

    let tag = buf[0];
    let rest = &buf[1..];

    match tag {
        TAG_TEXT => {
            if rest.len() < 4 { return None; }
            let n = u32::from_le_bytes(rest[..4].try_into().ok()?) as usize;
            let data = &rest[4..];
            if data.len() < n * 8 { return None; }
            let tokens: Vec<i64> = (0..n)
                .map(|i| i64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap()))
                .collect();
            Some(Sample::Text { tokens })
        }
        TAG_IMAGE => {
            if rest.len() < 16 { return None; }
            let width = u32::from_le_bytes(rest[..4].try_into().ok()?);
            let height = u32::from_le_bytes(rest[4..8].try_into().ok()?);
            let channels = u32::from_le_bytes(rest[8..12].try_into().ok()?);
            let data_len = u32::from_le_bytes(rest[12..16].try_into().ok()?) as usize;
            let data = rest[16..16 + data_len].to_vec();
            Some(Sample::Image { data, width, height, channels })
        }
        TAG_AUDIO => {
            if rest.len() < 8 { return None; }
            let sample_rate = u32::from_le_bytes(rest[..4].try_into().ok()?);
            let n = u32::from_le_bytes(rest[4..8].try_into().ok()?) as usize;
            let data = &rest[8..];
            if data.len() < n * 4 { return None; }
            let samples: Vec<f32> = (0..n)
                .map(|i| f32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap()))
                .collect();
            Some(Sample::Audio { samples, sample_rate })
        }
        TAG_VIDEO => {
            if rest.len() < 16 { return None; }
            let width = u32::from_le_bytes(rest[..4].try_into().ok()?);
            let height = u32::from_le_bytes(rest[4..8].try_into().ok()?);
            let fps = u32::from_le_bytes(rest[8..12].try_into().ok()?);
            let num_frames = u32::from_le_bytes(rest[12..16].try_into().ok()?) as usize;
            let mut offset = 16;
            let mut frames = Vec::with_capacity(num_frames);
            for _ in 0..num_frames {
                if offset + 4 > rest.len() { return None; }
                let frame_len = u32::from_le_bytes(rest[offset..offset + 4].try_into().ok()?) as usize;
                offset += 4;
                if offset + frame_len > rest.len() { return None; }
                frames.push(rest[offset..offset + frame_len].to_vec());
                offset += frame_len;
            }
            Some(Sample::Video { frames, width, height, fps })
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Batching by modality
// ---------------------------------------------------------------------------

/// Group samples by modality for efficient batching.
pub fn group_by_modality(samples: Vec<Sample>) -> std::collections::HashMap<Modality, Vec<Sample>> {
    let mut groups: std::collections::HashMap<Modality, Vec<Sample>> = std::collections::HashMap::new();
    for sample in samples {
        groups.entry(sample.modality()).or_default().push(sample);
    }
    groups
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_sample_roundtrip() {
        let sample = Sample::Text { tokens: vec![1, 2, 3, 42, 100] };
        let bytes = serialize_sample(&sample);
        let restored = deserialize_sample(&bytes).unwrap();
        if let Sample::Text { tokens } = restored {
            assert_eq!(tokens, vec![1, 2, 3, 42, 100]);
        } else {
            panic!("expected Text sample");
        }
    }

    #[test]
    fn test_image_sample_roundtrip() {
        let sample = Sample::Image {
            data: vec![255, 0, 128, 64, 32, 16],
            width: 2, height: 1, channels: 3,
        };
        let bytes = serialize_sample(&sample);
        let restored = deserialize_sample(&bytes).unwrap();
        if let Sample::Image { data, width, height, channels } = restored {
            assert_eq!(data, vec![255, 0, 128, 64, 32, 16]);
            assert_eq!(width, 2);
            assert_eq!(height, 1);
            assert_eq!(channels, 3);
        } else {
            panic!("expected Image sample");
        }
    }

    #[test]
    fn test_audio_sample_roundtrip() {
        let sample = Sample::Audio {
            samples: vec![0.1, -0.5, 0.9],
            sample_rate: 16000,
        };
        let bytes = serialize_sample(&sample);
        let restored = deserialize_sample(&bytes).unwrap();
        if let Sample::Audio { samples, sample_rate } = restored {
            assert_eq!(samples.len(), 3);
            assert_eq!(sample_rate, 16000);
            assert!((samples[0] - 0.1).abs() < 1e-6);
        } else {
            panic!("expected Audio sample");
        }
    }

    #[test]
    fn test_video_sample_roundtrip() {
        let sample = Sample::Video {
            frames: vec![vec![1, 2, 3], vec![4, 5, 6]],
            width: 1, height: 1, fps: 30,
        };
        let bytes = serialize_sample(&sample);
        let restored = deserialize_sample(&bytes).unwrap();
        if let Sample::Video { frames, width, height, fps } = restored {
            assert_eq!(frames.len(), 2);
            assert_eq!(frames[0], vec![1, 2, 3]);
            assert_eq!(fps, 30);
            assert_eq!(width, 1);
            assert_eq!(height, 1);
        } else {
            panic!("expected Video sample");
        }
    }

    #[test]
    fn test_modality_grouping() {
        let samples = vec![
            Sample::Text { tokens: vec![1] },
            Sample::Image { data: vec![0], width: 1, height: 1, channels: 1 },
            Sample::Text { tokens: vec![2] },
            Sample::Audio { samples: vec![0.0], sample_rate: 16000 },
        ];
        let groups = group_by_modality(samples);
        assert_eq!(groups.get(&Modality::Text).unwrap().len(), 2);
        assert_eq!(groups.get(&Modality::Image).unwrap().len(), 1);
        assert_eq!(groups.get(&Modality::Audio).unwrap().len(), 1);
        assert!(!groups.contains_key(&Modality::Video));
    }

    #[test]
    fn test_byte_size_estimation() {
        let text = Sample::Text { tokens: vec![1, 2, 3] };
        assert_eq!(text.byte_size(), 24); // 3 * 8

        let img = Sample::Image { data: vec![0; 100], width: 10, height: 10, channels: 1 };
        assert_eq!(img.byte_size(), 100);
    }

    #[test]
    fn test_empty_sample_deserialization() {
        assert!(deserialize_sample(&[]).is_none());
        assert!(deserialize_sample(&[255]).is_none()); // unknown tag
    }
}
