//! M48: Multimodal preprocessing FFI stubs.

/// Patch embedding: unfold image into patches and project.
/// img_ptr: [B, C, H, W] tensor, weight_ptr: [patch_dim, embed_dim] tensor
#[no_mangle]
pub extern "C" fn nsl_patch_embed(img_ptr: i64, weight_ptr: i64, patch_size: i64) -> i64 {
    let _ = (img_ptr, weight_ptr, patch_size);
    0
}

/// Mel spectrogram: STFT + power spectrum + mel filterbank.
#[no_mangle]
pub extern "C" fn nsl_mel_spectrogram(
    audio_ptr: i64,
    n_fft: i64,
    hop_length: i64,
    n_mels: i64,
) -> i64 {
    let _ = (audio_ptr, n_fft, hop_length, n_mels);
    0
}

/// Cross-modal attention: Q from one modality, K/V from another.
#[no_mangle]
pub extern "C" fn nsl_cross_attention(
    q_ptr: i64,
    k_ptr: i64,
    v_ptr: i64,
    num_heads: i64,
) -> i64 {
    let _ = (q_ptr, k_ptr, v_ptr, num_heads);
    0
}

/// Bilinear image resize.
#[no_mangle]
pub extern "C" fn nsl_image_resize(img_ptr: i64, target_h: i64, target_w: i64) -> i64 {
    let _ = (img_ptr, target_h, target_w);
    0
}

/// Per-channel image normalization (ImageNet-style).
#[no_mangle]
pub extern "C" fn nsl_image_normalize(img_ptr: i64, mean_ptr: i64, std_ptr: i64) -> i64 {
    let _ = (img_ptr, mean_ptr, std_ptr);
    0
}

/// Short-time Fourier transform.
#[no_mangle]
pub extern "C" fn nsl_stft(audio_ptr: i64, n_fft: i64, hop_length: i64) -> i64 {
    let _ = (audio_ptr, n_fft, hop_length);
    0
}

/// Audio resampling.
#[no_mangle]
pub extern "C" fn nsl_audio_resample(audio_ptr: i64, orig_sr: i64, target_sr: i64) -> i64 {
    let _ = (audio_ptr, orig_sr, target_sr);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stubs_return_zero() {
        assert_eq!(nsl_patch_embed(0, 0, 16), 0);
        assert_eq!(nsl_mel_spectrogram(0, 1024, 256, 80), 0);
        assert_eq!(nsl_cross_attention(0, 0, 0, 8), 0);
        assert_eq!(nsl_image_resize(0, 224, 224), 0);
        assert_eq!(nsl_image_normalize(0, 0, 0), 0);
        assert_eq!(nsl_stft(0, 1024, 256), 0);
        assert_eq!(nsl_audio_resample(0, 44100, 16000), 0);
    }
}
