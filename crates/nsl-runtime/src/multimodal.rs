//! M48b: Multimodal preprocessing — CPU implementations.
//!
//! Image: patch embedding, bilinear resize, per-channel normalization.
//! Audio: STFT (Hann-windowed DFT), mel spectrogram, resampling.
//! Cross-modal: multi-head scaled dot-product attention.
//!
//! All functions operate on NslTensor (f32, dtype=1) via raw pointers.
//! GPU variants deferred to M48c.

use std::sync::atomic::AtomicI64;
use std::ffi::c_void;
use crate::memory::{checked_alloc, checked_alloc_zeroed};
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read tensor shape into a Vec.
fn shape_vec(t: &NslTensor) -> Vec<i64> {
    if t.ndim <= 0 { return vec![]; }
    unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize).to_vec() }
}

/// Allocate a new f32 NslTensor (dtype=1) with given shape, zero-filled.
fn alloc_f32_tensor(shape: &[i64]) -> i64 {
    let ndim = shape.len() as i64;
    let len: i64 = shape.iter().product();
    if len <= 0 { return 0; }

    let shape_ptr = checked_alloc(shape.len() * 8) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);
    let data = checked_alloc_zeroed((len as usize) * 4) as *mut c_void;

    let tensor = Box::new(NslTensor {
        data,
        shape: shape_ptr,
        strides,
        ndim,
        len,
        refcount: AtomicI64::new(1),
        device: 0,
        dtype: 1,
        owns_data: 1,
    });
    Box::into_raw(tensor) as i64
}

/// Get f32 data slice from tensor.
unsafe fn f32_data(t: &NslTensor) -> &[f32] {
    std::slice::from_raw_parts(t.data as *const f32, t.len as usize)
}

/// Get mutable f32 data slice from tensor pointer.
unsafe fn f32_data_mut(ptr: i64) -> &'static mut [f32] {
    let t = &*(ptr as *const NslTensor);
    std::slice::from_raw_parts_mut(t.data as *mut f32, t.len as usize)
}

// ---------------------------------------------------------------------------
// Image: Patch Embedding
// ---------------------------------------------------------------------------

/// Extract non-overlapping patches from [B,C,H,W] image, flatten, project.
/// img_ptr: [B, C, H, W] f32 tensor
/// weight_ptr: [patch_dim, embed_dim] f32 tensor
/// patch_size: patch height/width (square patches)
/// Returns [B, num_patches, embed_dim] f32 tensor, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_patch_embed(img_ptr: i64, weight_ptr: i64, patch_size: i64) -> i64 {
    if img_ptr == 0 || weight_ptr == 0 || patch_size <= 0 { return 0; }
    let img = unsafe { &*(img_ptr as *const NslTensor) };
    let wt = unsafe { &*(weight_ptr as *const NslTensor) };
    let is = shape_vec(img);
    let ws = shape_vec(wt);
    if is.len() != 4 || ws.len() != 2 { return 0; }

    let (b, c, h, w) = (is[0], is[1], is[2], is[3]);
    let ps = patch_size;
    if h % ps != 0 || w % ps != 0 { return 0; }

    let ph = h / ps;
    let pw = w / ps;
    let np = ph * pw;
    let pd = (c * ps * ps) as usize; // patch_dim
    let ed = ws[1] as usize;         // embed_dim
    if ws[0] as usize != pd { return 0; }

    let out_ptr = alloc_f32_tensor(&[b, np, ws[1]]);
    if out_ptr == 0 { return 0; }

    let id = unsafe { f32_data(img) };
    let wd = unsafe { f32_data(wt) };
    let od = unsafe { f32_data_mut(out_ptr) };

    for bi in 0..b as usize {
        for py in 0..ph as usize {
            for px in 0..pw as usize {
                let patch_idx = py * pw as usize + px;
                // Extract and flatten patch
                let mut patch = vec![0.0f32; pd];
                for ci in 0..c as usize {
                    for dy in 0..ps as usize {
                        for dx in 0..ps as usize {
                            let iy = py * ps as usize + dy;
                            let ix = px * ps as usize + dx;
                            let src = bi * (c * h * w) as usize + ci * (h * w) as usize + iy * w as usize + ix;
                            let pidx = ci * (ps * ps) as usize + dy * ps as usize + dx;
                            patch[pidx] = id[src];
                        }
                    }
                }
                // Matrix multiply: patch[pd] @ weight[pd, ed] -> out[ed]
                for e in 0..ed {
                    let mut sum = 0.0f32;
                    for p in 0..pd {
                        sum += patch[p] * wd[p * ed + e];
                    }
                    od[bi * (np as usize * ed) + patch_idx * ed + e] = sum;
                }
            }
        }
    }
    out_ptr
}

// ---------------------------------------------------------------------------
// Image: Bilinear Resize
// ---------------------------------------------------------------------------

/// Bilinear interpolation resize: [B, C, H, W] -> [B, C, target_h, target_w].
#[no_mangle]
pub extern "C" fn nsl_image_resize(img_ptr: i64, target_h: i64, target_w: i64) -> i64 {
    if img_ptr == 0 || target_h <= 0 || target_w <= 0 { return 0; }
    let img = unsafe { &*(img_ptr as *const NslTensor) };
    let is = shape_vec(img);
    if is.len() != 4 { return 0; }

    let (b, c, h, w) = (is[0] as usize, is[1] as usize, is[2] as usize, is[3] as usize);
    let (th, tw) = (target_h as usize, target_w as usize);

    let out_ptr = alloc_f32_tensor(&[b as i64, c as i64, target_h, target_w]);
    if out_ptr == 0 { return 0; }

    let id = unsafe { f32_data(img) };
    let od = unsafe { f32_data_mut(out_ptr) };

    let scale_h = if th > 1 { (h as f32 - 1.0) / (th as f32 - 1.0) } else { 0.0 };
    let scale_w = if tw > 1 { (w as f32 - 1.0) / (tw as f32 - 1.0) } else { 0.0 };

    for bi in 0..b {
        for ci in 0..c {
            for ty in 0..th {
                let sy = ty as f32 * scale_h;
                let y0 = (sy as usize).min(h - 1);
                let y1 = (y0 + 1).min(h - 1);
                let fy = sy - y0 as f32;

                for tx in 0..tw {
                    let sx = tx as f32 * scale_w;
                    let x0 = (sx as usize).min(w - 1);
                    let x1 = (x0 + 1).min(w - 1);
                    let fx = sx - x0 as f32;

                    let base = bi * c * h * w + ci * h * w;
                    let v00 = id[base + y0 * w + x0];
                    let v01 = id[base + y0 * w + x1];
                    let v10 = id[base + y1 * w + x0];
                    let v11 = id[base + y1 * w + x1];

                    let val = v00 * (1.0 - fy) * (1.0 - fx)
                            + v01 * (1.0 - fy) * fx
                            + v10 * fy * (1.0 - fx)
                            + v11 * fy * fx;

                    od[bi * c * th * tw + ci * th * tw + ty * tw + tx] = val;
                }
            }
        }
    }
    out_ptr
}

// ---------------------------------------------------------------------------
// Image: Per-Channel Normalization
// ---------------------------------------------------------------------------

/// Per-channel normalization: out = (img - mean) / std.
/// img_ptr: [B, C, H, W], mean_ptr/std_ptr: [C] tensors.
#[no_mangle]
pub extern "C" fn nsl_image_normalize(img_ptr: i64, mean_ptr: i64, std_ptr: i64) -> i64 {
    if img_ptr == 0 || mean_ptr == 0 || std_ptr == 0 { return 0; }
    let img = unsafe { &*(img_ptr as *const NslTensor) };
    let mt = unsafe { &*(mean_ptr as *const NslTensor) };
    let st = unsafe { &*(std_ptr as *const NslTensor) };
    let is = shape_vec(img);
    if is.len() != 4 { return 0; }

    let (b, c, h, w) = (is[0] as usize, is[1] as usize, is[2] as usize, is[3] as usize);
    let means = unsafe { f32_data(mt) };
    let stds = unsafe { f32_data(st) };
    if means.len() < c || stds.len() < c { return 0; }

    let out_ptr = alloc_f32_tensor(&is);
    if out_ptr == 0 { return 0; }

    let id = unsafe { f32_data(img) };
    let od = unsafe { f32_data_mut(out_ptr) };

    for bi in 0..b {
        for ci in 0..c {
            let m = means[ci];
            let s = if stds[ci].abs() < 1e-10 { 1.0 } else { stds[ci] };
            let base = bi * c * h * w + ci * h * w;
            for i in 0..(h * w) {
                od[base + i] = (id[base + i] - m) / s;
            }
        }
    }
    out_ptr
}

// ---------------------------------------------------------------------------
// Cross-Modal Attention
// ---------------------------------------------------------------------------

/// Multi-head scaled dot-product attention.
/// q_ptr: [B, Sq, D], k_ptr: [B, Sk, D], v_ptr: [B, Sk, D]
/// num_heads: number of attention heads (D must be divisible by num_heads)
/// Returns [B, Sq, D].
#[no_mangle]
pub extern "C" fn nsl_cross_attention(
    q_ptr: i64, k_ptr: i64, v_ptr: i64, num_heads: i64,
) -> i64 {
    if q_ptr == 0 || k_ptr == 0 || v_ptr == 0 || num_heads <= 0 { return 0; }
    let qt = unsafe { &*(q_ptr as *const NslTensor) };
    let kt = unsafe { &*(k_ptr as *const NslTensor) };
    let vt = unsafe { &*(v_ptr as *const NslTensor) };
    let qs = shape_vec(qt);
    let ks = shape_vec(kt);
    if qs.len() != 3 || ks.len() != 3 { return 0; }

    let (b, sq, d) = (qs[0] as usize, qs[1] as usize, qs[2] as usize);
    let sk = ks[1] as usize;
    let nh = num_heads as usize;
    if d % nh != 0 { return 0; }
    let dh = d / nh;

    let out_ptr = alloc_f32_tensor(&[b as i64, sq as i64, d as i64]);
    if out_ptr == 0 { return 0; }

    let qd = unsafe { f32_data(qt) };
    let kd = unsafe { f32_data(kt) };
    let vd = unsafe { f32_data(vt) };
    let od = unsafe { f32_data_mut(out_ptr) };

    let scale = 1.0 / (dh as f32).sqrt();

    for bi in 0..b {
        for hi in 0..nh {
            // For each query position
            for qi in 0..sq {
                // Compute attention scores
                let mut scores = vec![0.0f32; sk];
                for (ki, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for di in 0..dh {
                        let q_idx = bi * sq * d + qi * d + hi * dh + di;
                        let k_idx = bi * sk * d + ki * d + hi * dh + di;
                        dot += qd[q_idx] * kd[k_idx];
                    }
                    *score = dot * scale;
                }
                // Numerically stable softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                if sum_exp > 0.0 {
                    for s in &mut scores { *s /= sum_exp; }
                }
                // Weighted sum of values
                for di in 0..dh {
                    let mut val = 0.0f32;
                    for (ki, score) in scores.iter().enumerate() {
                        let v_idx = bi * sk * d + ki * d + hi * dh + di;
                        val += score * vd[v_idx];
                    }
                    od[bi * sq * d + qi * d + hi * dh + di] = val;
                }
            }
        }
    }
    out_ptr
}

// ---------------------------------------------------------------------------
// Audio: STFT
// ---------------------------------------------------------------------------

/// Short-time Fourier transform with Hann window.
/// audio_ptr: [B, T] f32 tensor
/// Returns [B, n_freq, n_frames, 2] (real + imag) or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_stft(audio_ptr: i64, n_fft: i64, hop_length: i64) -> i64 {
    if audio_ptr == 0 || n_fft <= 0 || hop_length <= 0 { return 0; }
    let at = unsafe { &*(audio_ptr as *const NslTensor) };
    let as_ = shape_vec(at);
    if as_.len() != 2 { return 0; }

    let (b, t) = (as_[0] as usize, as_[1] as usize);
    let nfft = n_fft as usize;
    let hop = hop_length as usize;
    let nfreq = nfft / 2 + 1;
    let nframes = if t >= nfft { (t - nfft) / hop + 1 } else { return 0; };

    // Precompute Hann window
    let window: Vec<f32> = (0..nfft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / nfft as f32).cos()))
        .collect();

    let out_ptr = alloc_f32_tensor(&[b as i64, nfreq as i64, nframes as i64, 2]);
    if out_ptr == 0 { return 0; }

    let ad = unsafe { f32_data(at) };
    let od = unsafe { f32_data_mut(out_ptr) };

    for bi in 0..b {
        for fi in 0..nframes {
            let offset = fi * hop;
            // Apply window and compute DFT for each frequency bin
            for k in 0..nfreq {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let freq = 2.0 * std::f32::consts::PI * k as f32 / nfft as f32;
                for n in 0..nfft {
                    let sample = ad[bi * t + offset + n] * window[n];
                    let angle = freq * n as f32;
                    re += sample * angle.cos();
                    im -= sample * angle.sin();
                }
                let idx = bi * nfreq * nframes * 2 + k * nframes * 2 + fi * 2;
                od[idx] = re;
                od[idx + 1] = im;
            }
        }
    }
    out_ptr
}

// ---------------------------------------------------------------------------
// Audio: Mel Spectrogram
// ---------------------------------------------------------------------------

/// Mel spectrogram: STFT → power spectrum → mel filterbank → log.
/// Returns [B, n_mels, n_frames] or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_mel_spectrogram(
    audio_ptr: i64, n_fft: i64, hop_length: i64, n_mels: i64,
) -> i64 {
    if audio_ptr == 0 || n_fft <= 0 || hop_length <= 0 || n_mels <= 0 { return 0; }

    // Compute STFT first
    let stft_ptr = nsl_stft(audio_ptr, n_fft, hop_length);
    if stft_ptr == 0 { return 0; }

    let st = unsafe { &*(stft_ptr as *const NslTensor) };
    let ss = shape_vec(st);
    let (b, nfreq, nframes) = (ss[0] as usize, ss[1] as usize, ss[2] as usize);
    let nm = n_mels as usize;
    let sd = unsafe { f32_data(st) };

    // Build mel filterbank (simplified triangular)
    let sr = 16000.0f32; // assumed sample rate
    let fmin = 0.0f32;
    let fmax = sr / 2.0;
    let mel_min = 2595.0 * (1.0 + fmin / 700.0).log10();
    let mel_max = 2595.0 * (1.0 + fmax / 700.0).log10();

    let mel_points: Vec<f32> = (0..nm + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (nm + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter()
        .map(|m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0))
        .collect();
    let bin_points: Vec<f32> = hz_points.iter()
        .map(|h| h * (n_fft + 1) as f32 / sr)
        .collect();

    // Build filterbank matrix [n_mels, nfreq]
    let mut filterbank = vec![0.0f32; nm * nfreq];
    for m in 0..nm {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for k in 0..nfreq {
            let kf = k as f32;
            if kf >= left && kf < center && center > left {
                filterbank[m * nfreq + k] = (kf - left) / (center - left);
            } else if kf >= center && kf <= right && right > center {
                filterbank[m * nfreq + k] = (right - kf) / (right - center);
            }
        }
    }

    let out_ptr = alloc_f32_tensor(&[b as i64, nm as i64, nframes as i64]);
    if out_ptr == 0 { return 0; }
    let od = unsafe { f32_data_mut(out_ptr) };

    for bi in 0..b {
        for m in 0..nm {
            for fi in 0..nframes {
                let mut mel_val = 0.0f32;
                for k in 0..nfreq {
                    let idx = bi * nfreq * nframes * 2 + k * nframes * 2 + fi * 2;
                    let re = sd[idx];
                    let im = sd[idx + 1];
                    let power = re * re + im * im;
                    mel_val += filterbank[m * nfreq + k] * power;
                }
                // Log mel with floor to avoid log(0)
                od[bi * nm * nframes + m * nframes + fi] = (mel_val + 1e-10).ln();
            }
        }
    }

    // Free intermediate STFT tensor
    crate::tensor::nsl_tensor_free(stft_ptr);
    out_ptr
}

// ---------------------------------------------------------------------------
// Audio: Resampling
// ---------------------------------------------------------------------------

/// Linear interpolation audio resampling: [B, T] -> [B, T_new].
#[no_mangle]
pub extern "C" fn nsl_audio_resample(audio_ptr: i64, orig_sr: i64, target_sr: i64) -> i64 {
    if audio_ptr == 0 || orig_sr <= 0 || target_sr <= 0 { return 0; }
    let at = unsafe { &*(audio_ptr as *const NslTensor) };
    let as_ = shape_vec(at);
    if as_.len() != 2 { return 0; }

    let (b, t) = (as_[0] as usize, as_[1] as usize);
    let t_new = ((t as f64) * target_sr as f64 / orig_sr as f64).round() as usize;
    if t_new == 0 { return 0; }

    let out_ptr = alloc_f32_tensor(&[b as i64, t_new as i64]);
    if out_ptr == 0 { return 0; }

    let ad = unsafe { f32_data(at) };
    let od = unsafe { f32_data_mut(out_ptr) };

    let ratio = orig_sr as f64 / target_sr as f64;
    for bi in 0..b {
        for i in 0..t_new {
            let src = i as f64 * ratio;
            let s0 = (src as usize).min(t - 1);
            let s1 = (s0 + 1).min(t - 1);
            let frac = (src - s0 as f64) as f32;
            od[bi * t_new + i] = ad[bi * t + s0] * (1.0 - frac) + ad[bi * t + s1] * frac;
        }
    }
    out_ptr
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_inputs_return_zero() {
        assert_eq!(nsl_patch_embed(0, 0, 16), 0);
        assert_eq!(nsl_mel_spectrogram(0, 1024, 256, 80), 0);
        assert_eq!(nsl_cross_attention(0, 0, 0, 8), 0);
        assert_eq!(nsl_image_resize(0, 224, 224), 0);
        assert_eq!(nsl_image_normalize(0, 0, 0), 0);
        assert_eq!(nsl_stft(0, 1024, 256), 0);
        assert_eq!(nsl_audio_resample(0, 44100, 16000), 0);
    }

    #[test]
    fn alloc_f32_tensor_basic() {
        let ptr = alloc_f32_tensor(&[2, 3]);
        assert_ne!(ptr, 0);
        let t = unsafe { &*(ptr as *const NslTensor) };
        assert_eq!(t.ndim, 2);
        assert_eq!(t.len, 6);
        assert_eq!(t.dtype, 1);
        crate::tensor::nsl_tensor_free(ptr);
    }

    #[test]
    fn image_normalize_basic() {
        // Create a [1, 2, 1, 1] image, [2] mean, [2] std
        let img_ptr = alloc_f32_tensor(&[1, 2, 1, 1]);
        let mean_ptr = alloc_f32_tensor(&[2]);
        let std_ptr = alloc_f32_tensor(&[2]);
        assert_ne!(img_ptr, 0);
        assert_ne!(mean_ptr, 0);
        assert_ne!(std_ptr, 0);

        // Set values: img = [10.0, 20.0], mean = [5.0, 10.0], std = [2.0, 5.0]
        unsafe {
            let id = f32_data_mut(img_ptr);
            id[0] = 10.0; id[1] = 20.0;
            let md = f32_data_mut(mean_ptr);
            md[0] = 5.0; md[1] = 10.0;
            let sd = f32_data_mut(std_ptr);
            sd[0] = 2.0; sd[1] = 5.0;
        }

        let out = nsl_image_normalize(img_ptr, mean_ptr, std_ptr);
        assert_ne!(out, 0);

        let od = unsafe { f32_data(&*(out as *const NslTensor)) };
        assert!((od[0] - 2.5).abs() < 0.01);  // (10-5)/2
        assert!((od[1] - 2.0).abs() < 0.01);  // (20-10)/5

        crate::tensor::nsl_tensor_free(img_ptr);
        crate::tensor::nsl_tensor_free(mean_ptr);
        crate::tensor::nsl_tensor_free(std_ptr);
        crate::tensor::nsl_tensor_free(out);
    }
}
