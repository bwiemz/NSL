//! M48: Native multimodal primitives — component configs and compile-time constants.

// ---------------------------------------------------------------------------
// PatchEmbed
// ---------------------------------------------------------------------------

/// Configuration for PatchEmbed model component.
#[derive(Debug, Clone)]
pub struct PatchEmbedConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
}

impl PatchEmbedConfig {
    pub fn new(image_size: usize, patch_size: usize, embed_dim: usize) -> Result<Self, String> {
        if patch_size == 0 {
            return Err("patch_size must be > 0".into());
        }
        if !image_size.is_multiple_of(patch_size) {
            return Err(format!(
                "image_size {} not divisible by patch_size {}",
                image_size, patch_size
            ));
        }
        Ok(PatchEmbedConfig {
            image_size,
            patch_size,
            in_channels: 3,
            embed_dim,
        })
    }

    /// Number of patches: (image_size / patch_size)^2
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Flattened patch dimension: patch_size * patch_size * in_channels
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.in_channels
    }
}

// ---------------------------------------------------------------------------
// MelSpectrogram
// ---------------------------------------------------------------------------

/// Configuration for MelSpectrogram model component.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub sample_rate: usize,
}

impl MelConfig {
    pub fn new(n_fft: usize, hop_length: usize, n_mels: usize) -> Self {
        MelConfig {
            n_fft,
            hop_length,
            n_mels,
            sample_rate: 16000,
        }
    }

    /// Compute output time frames for a given input sample count.
    pub fn time_frames(&self, num_samples: usize) -> usize {
        if num_samples < self.n_fft {
            return 0;
        }
        (num_samples - self.n_fft) / self.hop_length + 1
    }

    /// Number of frequency bins in STFT output.
    pub fn freq_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// Convert frequency in Hz to mel scale.
pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to frequency in Hz.
pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Build a triangular mel filterbank matrix.
///
/// Returns a [n_mels x freq_bins] matrix (row-major) where each row is a
/// triangular filter centered on a mel-spaced frequency.
pub fn build_mel_filterbank(config: &MelConfig) -> Vec<f64> {
    let freq_bins = config.freq_bins();
    let fmax = config.sample_rate as f64 / 2.0;
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(fmax);

    // n_mels + 2 points (including endpoints)
    let mel_points: Vec<f64> = (0..=config.n_mels + 1)
        .map(|i| mel_to_hz(mel_low + (mel_high - mel_low) * i as f64 / (config.n_mels + 1) as f64))
        .collect();

    // Convert Hz points to FFT bin indices
    let bin_indices: Vec<f64> = mel_points
        .iter()
        .map(|&hz| hz * (config.n_fft as f64 + 1.0) / config.sample_rate as f64)
        .collect();

    // Build triangular filters
    let mut filterbank = vec![0.0f64; config.n_mels * freq_bins];
    for m in 0..config.n_mels {
        let left = bin_indices[m];
        let center = bin_indices[m + 1];
        let right = bin_indices[m + 2];

        for k in 0..freq_bins {
            let kf = k as f64;
            let val = if kf >= left && kf <= center && center > left {
                (kf - left) / (center - left)
            } else if kf > center && kf <= right && right > center {
                (right - kf) / (right - center)
            } else {
                0.0
            };
            filterbank[m * freq_bins + k] = val;
        }
    }
    filterbank
}

// ---------------------------------------------------------------------------
// CrossAttention
// ---------------------------------------------------------------------------

/// Configuration for cross-modal attention.
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    pub embed_dim: usize,
    pub num_heads: usize,
}

impl CrossAttentionConfig {
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self, String> {
        if num_heads == 0 {
            return Err("num_heads must be > 0".into());
        }
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(format!(
                "embed_dim {} not divisible by num_heads {}",
                embed_dim, num_heads
            ));
        }
        Ok(CrossAttentionConfig {
            embed_dim,
            num_heads,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

// ---------------------------------------------------------------------------
// Modality Classification
// ---------------------------------------------------------------------------

/// Input modality classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Vision, // 4D: [B, C, H, W]
    Audio,  // 2D float: [B, samples]
    Text,   // 2D int: [B, seq] or 3D: [B, seq, hidden]
}

impl Modality {
    pub fn name(&self) -> &'static str {
        match self {
            Modality::Vision => "vision",
            Modality::Audio => "audio",
            Modality::Text => "text",
        }
    }
}

/// Classify a tensor input's modality based on rank and dtype.
///
/// Heuristic:
/// - rank 4 → Vision ([B, C, H, W])
/// - rank 2 + float → Audio ([B, samples])
/// - rank 2 + integer → Text ([B, seq])
/// - rank 3 → Text ([B, seq, hidden])
pub fn classify_modality(rank: usize, is_integer_dtype: bool) -> Option<Modality> {
    match (rank, is_integer_dtype) {
        (4, _) => Some(Modality::Vision),
        (2, false) => Some(Modality::Audio),
        (2, true) => Some(Modality::Text),
        (3, _) => Some(Modality::Text),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_embed_config_valid() {
        let cfg = PatchEmbedConfig::new(224, 16, 768).unwrap();
        assert_eq!(cfg.num_patches(), 196);
        assert_eq!(cfg.patch_dim(), 768); // 16*16*3
    }

    #[test]
    fn patch_embed_config_invalid() {
        assert!(PatchEmbedConfig::new(224, 15, 768).is_err()); // not divisible
        assert!(PatchEmbedConfig::new(224, 0, 768).is_err()); // zero patch
    }

    #[test]
    fn mel_config_time_frames() {
        let cfg = MelConfig::new(1024, 256, 80);
        assert_eq!(cfg.time_frames(16000), 59); // (16000-1024)/256 + 1
        assert_eq!(cfg.time_frames(512), 0); // too short
        assert_eq!(cfg.freq_bins(), 513); // 1024/2 + 1
    }

    #[test]
    fn hz_mel_roundtrip() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((hz - back).abs() < 0.01);
    }

    #[test]
    fn mel_filterbank_shape() {
        let cfg = MelConfig::new(1024, 256, 80);
        let fb = build_mel_filterbank(&cfg);
        assert_eq!(fb.len(), 80 * 513); // n_mels * freq_bins
    }

    #[test]
    fn mel_filterbank_normalized() {
        let cfg = MelConfig::new(1024, 256, 40);
        let fb = build_mel_filterbank(&cfg);
        let freq_bins = cfg.freq_bins();
        // Each filter should have non-negative values
        assert!(fb.iter().all(|&v| v >= 0.0));
        // Each filter should have at least one non-zero value
        for m in 0..40 {
            let row = &fb[m * freq_bins..(m + 1) * freq_bins];
            assert!(row.iter().any(|&v| v > 0.0), "filter {m} is all zeros");
        }
    }

    #[test]
    fn cross_attention_config_valid() {
        let cfg = CrossAttentionConfig::new(768, 12).unwrap();
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn cross_attention_config_invalid() {
        assert!(CrossAttentionConfig::new(768, 13).is_err()); // not divisible
        assert!(CrossAttentionConfig::new(768, 0).is_err()); // zero heads
    }

    #[test]
    fn modality_classification() {
        assert_eq!(classify_modality(4, false), Some(Modality::Vision));
        assert_eq!(classify_modality(2, false), Some(Modality::Audio));
        assert_eq!(classify_modality(2, true), Some(Modality::Text));
        assert_eq!(classify_modality(3, false), Some(Modality::Text));
        assert_eq!(classify_modality(1, false), None);
    }
}
