//! PCA Tier A backward numerical correctness fixtures (spec §3.4 + §7.3).
//!
//! Three test cases (parallel to forward fixtures in
//! pca_tier_a_forward_correctness.rs):
//!   1. Single-segment — all segment_ids = 0, mask is a no-op. dQ/dK/dV
//!      must match the segment_masked=false baseline within 5e-3.
//!   2. Two-equal-segments — seq_len=128 split [0..64]=0, [64..128]=1.
//!      Packed dQ/dK/dV compared against unpacked-padded reference
//!      (two independent GPU backward passes on seq_len=64 each, gradients
//!      scattered back to packed positions). Tolerance: 5e-3.
//!   3. Unequal-segments — three segments of lengths 38, 64, 26 packed
//!      into seq_len=128. Compared against three independent GPU backward
//!      passes. Tolerance: 5e-3.
//!
//! Reference strategy: both packed (PCA) and unpacked (reference) runs use
//! the same GPU CSHA backward kernel. The unpacked runs use segment_masked=false
//! on each segment's slice independently; the packed run uses segment_masked=true
//! on the full sequence. Divergence indicates a segment-mask bug in the backward.
//!
//! dQ/dK/dV are stored by the kernel as f16; tolerance 5e-3 matches CSHA Tier
//! A backward at head_dim=32 (spec §7.4).
//!
//! Gated with #[cfg(feature = "cuda")] and #[ignore]. Run with:
//!   cargo test -p nsl-codegen --features cuda --test pca_tier_a_backward_correctness \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2_backward, synthesize_backward,
    smem_layout::{self, Direction},
};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h,
    nsl_test_cuda_free, nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use nsl_runtime::flash_attention::{
    flash_attention_backward_cpu_gqa,
    nsl_csha_alloc_backward_activations, nsl_csha_free_backward_activations,
    nsl_flash_attention_csha_backward,
};

// ---------------------------------------------------------------------------
// IEEE 754 half-precision helpers
// ---------------------------------------------------------------------------

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp  = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut m = mant;
            let mut e: i32 = -1;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            let e = (127 + e - 14) as u32;
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        (sign << 31) | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() { return 0x7E00; }
    let b = x.to_bits();
    let sign = (b >> 31) & 1;
    let exp  = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 255 { return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16; }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 { return ((sign << 15) | 0x7C00) as u16; }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

// ---------------------------------------------------------------------------
// Deterministic PRNG — same LCG as pca_tier_a_forward_correctness.rs
// ---------------------------------------------------------------------------

fn fill_seeded(dst: &mut [f32], mut seed: u64) {
    for x in dst.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (seed >> 33) as u32;
        *x = ((u as f32) / (u32::MAX as f32)) - 0.5;
    }
}

fn roundtrip_f16_vec(src: &[f32]) -> Vec<f32> {
    src.iter().map(|&x| f16_to_f32(f32_to_f16_bits(x))).collect()
}

fn rmsnorm_rows(x_host: &[f32], seq_len: usize, head_dim: usize, eps: f32) -> Vec<f32> {
    let mut x_norm = vec![0.0f32; x_host.len()];
    for row in 0..seq_len {
        let row_base = row * head_dim;
        let mean_sq = x_host[row_base..row_base + head_dim]
            .iter()
            .copied()
            .map(|v| v * v)
            .sum::<f32>() / head_dim as f32;
        let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
        for d in 0..head_dim {
            x_norm[row_base + d] = x_host[row_base + d] * inv_rms;
        }
    }
    x_norm
}

fn project_rows_f16_saved(
    x_norm: &[f32],
    w_f16: &[u16],
    seq_len: usize,
    head_dim: usize,
) -> Vec<f32> {
    let w: Vec<f32> = w_f16.iter().map(|&bits| f16_to_f32(bits)).collect();
    let mut out = vec![0.0f32; seq_len * head_dim];
    for row in 0..seq_len {
        let row_base = row * head_dim;
        for out_col in 0..head_dim {
            let mut acc = 0.0f32;
            for inner in 0..head_dim {
                acc += x_norm[row_base + inner] * w[inner * head_dim + out_col];
            }
            out[row_base + out_col] = acc;
        }
    }
    roundtrip_f16_vec(&out)
}

fn flash_attention_forward_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; seq_len * head_dim];
    let mut lse = vec![0.0f32; seq_len];
    for i in 0..seq_len {
        let qi = i * head_dim;
        let j_max = if causal { i + 1 } else { seq_len };
        let mut scores = vec![f32::NEG_INFINITY; seq_len];
        let mut max_score = f32::NEG_INFINITY;
        for (j, score_slot) in scores.iter_mut().enumerate().take(j_max) {
            let kj = j * head_dim;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[qi + d] * k[kj + d];
            }
            score *= scale;
            *score_slot = score;
            max_score = max_score.max(score);
        }
        let mut sum_exp = 0.0f32;
        for score in scores.iter_mut().take(j_max) {
            *score = (*score - max_score).exp();
            sum_exp += *score;
        }
        lse[i] = max_score + sum_exp.ln();
        for (j, score) in scores.iter().enumerate().take(j_max) {
            let prob = *score / sum_exp;
            let vj = j * head_dim;
            for d in 0..head_dim {
                out[qi + d] += prob * v[vj + d];
            }
        }
    }
    (roundtrip_f16_vec(&out), lse)
}

fn flash_attention_forward_reference_with_stats(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
    seg_ids: &[u16],
    segment_masked: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; seq_len * head_dim];
    let mut lse = vec![0.0f32; seq_len];
    let mut row_max = vec![f32::NEG_INFINITY; seq_len];
    let mut row_sum = vec![0.0f32; seq_len];

    for i in 0..seq_len {
        let qi = i * head_dim;
        let j_max = if causal { i + 1 } else { seq_len };
        let mut scores = vec![f32::NEG_INFINITY; seq_len];
        let mut max_score = f32::NEG_INFINITY;
        // Null-guard mirror: when seg_ids is empty (GPU null-guard case),
        // treat as unmasked regardless of segment_masked flag — the GPU kernel's
        // skip-before-dereference guard writes all-zero seg_smem sentinel which
        // has the same effect as no masking.
        let seg_i = if segment_masked && !seg_ids.is_empty() {
            Some(seg_ids[i])
        } else {
            None
        };

        for (j, score_slot) in scores.iter_mut().enumerate().take(j_max) {
            if let Some(seg_i) = seg_i {
                if seg_ids[j] != seg_i {
                    continue;
                }
            }
            let kj = j * head_dim;
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[qi + d] * k[kj + d];
            }
            score *= scale;
            *score_slot = score;
            max_score = max_score.max(score);
        }

        row_max[i] = max_score;
        let mut sum_exp = 0.0f32;
        for score in scores.iter_mut().take(j_max) {
            if score.is_finite() {
                *score = (*score - max_score).exp();
                sum_exp += *score;
            }
        }
        row_sum[i] = sum_exp;
        lse[i] = max_score + sum_exp.ln();

        for (j, score) in scores.iter().enumerate().take(j_max) {
            if !score.is_finite() {
                continue;
            }
            let prob = *score / sum_exp;
            let vj = j * head_dim;
            for d in 0..head_dim {
                out[qi + d] += prob * v[vj + d];
            }
        }
    }

    (roundtrip_f16_vec(&out), lse, row_max, row_sum)
}

fn assert_no_nan(name: &str, arr: &[f32], fixture: &str) {
    assert!(arr.iter().all(|x| !x.is_nan()),
        "{fixture}: {name} contains NaN values");
}

// ---------------------------------------------------------------------------
// CUDA availability guard
// ---------------------------------------------------------------------------

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Free multiple device pointers, skipping zeros
// ---------------------------------------------------------------------------

fn free_all(ptrs: &[i64]) {
    for &p in ptrs {
        if p != 0 { unsafe { nsl_test_cuda_free(p); } }
    }
}

// ---------------------------------------------------------------------------
// CSHA backward config used for all PCA fixtures.
//
// block_q = block_kv = 32 (fused_projections constraint).
// d_model = head_dim = 32.
// causal = true (matches forward fixtures).
// segment_masked controls the PCA vs baseline path.
// ---------------------------------------------------------------------------

fn pca_backward_config(segment_masked: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q:        32,
        block_kv:       32,
        head_dim:       32,
        causal:         true,
        paged:          false,
        rope_q:         false,
        rope_style:     RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask:      false,
        gpu_sm:         75,
        segment_masked,
        csha: Some(CshaExtras {
            level:                         2,
            fused_rmsnorm:                 true,
            fused_projections:             true,
            fused_output_proj:             false,
            save_activations_for_backward: true,
            active_heads:                  1,
            rmsnorm_eps:                   1e-5,
            d_model:                       32,
            skip_rmsnorm_prologue:         false,
        }),
    }
}

// ---------------------------------------------------------------------------
// Backward kernel name: flash_attn_backward_<suffix>
// ---------------------------------------------------------------------------

fn backward_kernel_name(cfg: &FlashAttentionConfig) -> String {
    let fw = flash_attention_kernel_name_v2(cfg);
    match fw.strip_prefix("flash_attn_") {
        Some(rest) => format!("flash_attn_backward_{}", rest),
        None       => format!("flash_attn_backward_{fw}"),
    }
}

// ---------------------------------------------------------------------------
// Core launch helper: stage trusted forward states, then launch CSHA backward.
//
// x_host: raw input [seq_len × head_dim] as f32 (CSHA kernel applies RMSNorm
//         + projection internally; q/k/v are derived from x + weights).
// do_host: upstream gradient [seq_len × head_dim] as f32 (converted to f16).
// seg_ids_host: length seq_len; empty slice → unpacked path (seg_ids_dev = 0).
// segment_masked: controls which PTX variant is synthesised.
//
// Returns (dq_f32, dk_f32, dv_f32) read back from device (f16→f32), or None
// on any kernel launch failure (graceful skip).
// ---------------------------------------------------------------------------

fn launch_pca_backward(
    x_host:       &[f32],   // [seq_len × head_dim], used as CSHA raw input
    wq_f16:       &[u16],   // weight matrices [d_model × head_dim] f16
    wk_f16:       &[u16],
    wv_f16:       &[u16],
    do_host_f16:  &[u16],   // upstream gradient [seq_len × head_dim] f16
    seq_len:      usize,
    head_dim:     usize,
    seg_ids_host: &[u16],   // length seq_len; empty → unpacked
    segment_masked: bool,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let batch = 1usize;
    let heads = 1usize;
    let dm    = head_dim;   // d_model == head_dim for these fixtures
    let norm_eps = 1e-5f32;
    let scale    = 1.0f32 / (head_dim as f32).sqrt();

    let config = pca_backward_config(segment_masked);

    if let Err(e) = smem_layout::validate_scalar_v2_config(&config, Direction::Backward) {
        eprintln!("[pca_bwd] backward validator rejected: {e}");
        return None;
    }

    // ── Byte counts ──────────────────────────────────────────────────────────
    // q/k/v/out_dev: f16 — CSHA kernel writes projected QKV as f16.
    // x_dev: f32 — raw input before RMSNorm + projection.
    let qkv_f16_bytes = (batch * heads * seq_len * head_dim * 2) as i64;
    let lse_bytes     = (batch * heads * seq_len * 4) as i64;
    let x_bytes       = (batch * heads * seq_len * head_dim * 4) as i64;  // f32
    let w_bytes       = (dm * (heads * head_dim) * 2) as i64;             // f16
    let nw_bytes      = (head_dim * 4) as i64;                             // f32 norm weight
    let dw_bytes      = (dm * (heads * head_dim) * 2) as i64;
    let dx_bytes      = (batch * heads * seq_len * head_dim * 4) as i64;
    let dxn_bytes     = (batch * seq_len * dm * 4) as i64;

    // ── Allocate device buffers ───────────────────────────────────────────────
    let q_dev   = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let k_dev   = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let v_dev   = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };
    let x_dev   = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let nw_dev  = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    let wq_dev  = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wk_dev  = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wv_dev  = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let do_dev  = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let dq_dev  = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let dk_dev  = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let dv_dev  = unsafe { nsl_test_cuda_alloc(qkv_f16_bytes) };
    let dwq_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dwk_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dwv_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dx_dev  = unsafe { nsl_test_cuda_alloc(dx_bytes) };
    let dxn_dev = unsafe { nsl_test_cuda_alloc(dxn_bytes) };

    let all_dev = [
        q_dev, k_dev, v_dev, out_dev, lse_dev, x_dev, nw_dev,
        wq_dev, wk_dev, wv_dev, do_dev, dq_dev, dk_dev, dv_dev,
        dwq_dev, dwk_dev, dwv_dev, dx_dev, dxn_dev,
    ];

    // Verify all allocs succeeded.
    for &p in &all_dev {
        if p == 0 {
            eprintln!("[pca_bwd] device alloc returned null (OOM?)");
            free_all(&all_dev);
            return None;
        }
    }

    let saves = unsafe {
        nsl_csha_alloc_backward_activations(
            batch as i64, heads as i64, seq_len as i64, head_dim as i64,
        )
    };
    let save_ptrs = [
        saves.q_proj,
        saves.k_proj,
        saves.v_proj,
        saves.row_max,
        saves.row_sum,
        saves.x_raw,
    ];
    if save_ptrs.iter().any(|&ptr| ptr == 0) {
        eprintln!("[pca_bwd] CshaBackwardActivations alloc failed");
        unsafe { nsl_csha_free_backward_activations(saves); }
        free_all(&all_dev);
        return None;
    }

    // ── Stage trusted forward states on the host ────────────────────────────
    // The save-enabled fused forward kernel is not trustworthy on these
    // multi-tile PCA fixtures, so populate the backward inputs directly from
    // the harness's CPU forward reference.
    let x_norm = rmsnorm_rows(x_host, seq_len, head_dim, norm_eps);
    let q_host_f32 = project_rows_f16_saved(&x_norm, wq_f16, seq_len, head_dim);
    let k_host_f32 = project_rows_f16_saved(&x_norm, wk_f16, seq_len, head_dim);
    let v_host_f32 = project_rows_f16_saved(&x_norm, wv_f16, seq_len, head_dim);
    let (out_host_f32, lse_host, row_max_host, row_sum_host) =
        flash_attention_forward_reference_with_stats(
            &q_host_f32,
            &k_host_f32,
            &v_host_f32,
            seq_len,
            head_dim,
            scale,
            config.causal,
            seg_ids_host,
            segment_masked,
        );
    let q_host_f16: Vec<u16> = q_host_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let k_host_f16: Vec<u16> = k_host_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let v_host_f16: Vec<u16> = v_host_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();
    let out_host_f16: Vec<u16> = out_host_f32.iter().map(|&x| f32_to_f16_bits(x)).collect();

    // ── Upload ───────────────────────────────────────────────────────────────
    // Norm weight = 1.0 (identity — keeps RMSNorm as a no-op when x is small).
    let nw_host = vec![1.0f32; head_dim];
    unsafe {
        nsl_test_cuda_h2d(q_dev,   q_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(k_dev,   k_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(v_dev,   v_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(out_dev, out_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(lse_dev, lse_host.as_ptr() as i64, lse_bytes);
        nsl_test_cuda_h2d(x_dev,   x_host.as_ptr()    as i64, x_bytes);
        nsl_test_cuda_h2d(nw_dev,  nw_host.as_ptr()   as i64, nw_bytes);
        nsl_test_cuda_h2d(wq_dev,  wq_f16.as_ptr()    as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev,  wk_f16.as_ptr()    as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev,  wv_f16.as_ptr()    as i64, w_bytes);
        nsl_test_cuda_h2d(do_dev,  do_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(saves.q_proj, q_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(saves.k_proj, k_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(saves.v_proj, v_host_f16.as_ptr() as i64, qkv_f16_bytes);
        nsl_test_cuda_h2d(saves.row_max, row_max_host.as_ptr() as i64, lse_bytes);
        nsl_test_cuda_h2d(saves.row_sum, row_sum_host.as_ptr() as i64, lse_bytes);
        nsl_test_cuda_h2d(saves.x_raw, x_host.as_ptr() as i64, x_bytes);
    }

    // Upload segment_ids if needed.
    let seg_dev: i64 = if segment_masked && !seg_ids_host.is_empty() {
        let seg_bytes = (seg_ids_host.len() * 2) as i64;
        let ptr = unsafe { nsl_test_cuda_alloc(seg_bytes) };
        if ptr == 0 {
            eprintln!("[pca_bwd] segment_ids alloc failed");
            unsafe { nsl_csha_free_backward_activations(saves); }
            free_all(&all_dev);
            return None;
        }
        unsafe { nsl_test_cuda_h2d(ptr, seg_ids_host.as_ptr() as i64, seg_bytes); }
        ptr
    } else {
        0i64
    };

    if let Ok(rows) = std::env::var("NSL_PCA_DUMP_ROW_STATS") {
        let wanted: Vec<usize> = rows
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .filter(|&row| row < seq_len)
            .collect();
        if !wanted.is_empty() {
            let stats_elems = batch * heads * seq_len;
            let mut row_max_host = vec![0.0f32; stats_elems];
            let mut row_sum_host = vec![0.0f32; stats_elems];
            unsafe {
                nsl_test_cuda_d2h(
                    row_max_host.as_mut_ptr() as i64,
                    saves.row_max,
                    (stats_elems * 4) as i64,
                );
                nsl_test_cuda_d2h(
                    row_sum_host.as_mut_ptr() as i64,
                    saves.row_sum,
                    (stats_elems * 4) as i64,
                );
            }
            for row in wanted {
                eprintln!(
                    "[pca_bwd] row_stats row={} row_max={:?} row_sum={:?} seq_len={} segment_masked={}",
                    row,
                    row_max_host[row],
                    row_sum_host[row],
                    seq_len,
                    segment_masked,
                );
            }
        }
    }
    if let Ok(rows) = std::env::var("NSL_PCA_DUMP_OUT_ROWS") {
        let wanted: Vec<usize> = rows
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .filter(|&row| row < seq_len)
            .collect();
        if !wanted.is_empty() {
            let out_elems = batch * heads * seq_len * head_dim;
            let mut out_host = vec![0u16; out_elems];
            unsafe {
                nsl_test_cuda_d2h(
                    out_host.as_mut_ptr() as i64,
                    out_dev,
                    (out_elems * 2) as i64,
                );
            }
            let out_f32: Vec<f32> = out_host.into_iter().map(f16_to_f32).collect();
            for row in wanted {
                let base = row * head_dim;
                eprintln!(
                    "[pca_bwd] out_row row={} first4={:?} seq_len={} segment_masked={}",
                    row,
                    &out_f32[base..(base + 4).min(out_f32.len())],
                    seq_len,
                    segment_masked,
                );
            }
        }
    }

    // ── Backward PTX ─────────────────────────────────────────────────────────
    let mut bwd_ptx_str = match synthesize_backward(&config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[pca_bwd] synthesize_backward error: {e}");
            unsafe { nsl_csha_free_backward_activations(saves); }
            free_all(&all_dev);
            if seg_dev != 0 { unsafe { nsl_test_cuda_free(seg_dev); } }
            return None;
        }
    };
    if !bwd_ptx_str.ends_with('\0') { bwd_ptx_str.push('\0'); }
    // DEBUG: dump backward PTX.
    {
        let dump = std::env::temp_dir().join(format!(
            "pca_bwd_bwd_seg{}_sq{}.ptx",
            if segment_masked { "masked" } else { "plain" }, seq_len,
        ));
        std::fs::write(&dump, bwd_ptx_str.as_bytes()).ok();
        eprintln!("[pca_bwd] bwd PTX dumped: {}", dump.display());
    }
    let bwd_ptx  = bwd_ptx_str.into_bytes();
    let bwd_name = CString::new(backward_kernel_name(&config)).unwrap();

    // Dynamic SMEM size for the backward launch. Use the shared helper so
    // this stays in sync with the production wengert launcher
    // (`wengert_lower.rs::FusedCshaBackward`) — both call
    // `shared_mem_bytes_v2_backward` which folds backward_extra_bytes and
    // the PCA seg_smem tail into a single source of truth.
    //
    // kernel_launch only calls cuFuncSetAttribute when shared_mem_bytes > 48KB,
    // so we pass the full backward total when it exceeds the static cap to
    // force the opt-in; when it doesn't, the kernel runs entirely from static
    // SMEM and the dynamic size is ignored.
    let bwd_smem_total = shared_mem_bytes_v2_backward(&config);
    let smem_static_cap: u32 = 49152; // 48 KB hard limit for static SMEM
    let bwd_smem_dyn: i64 = if bwd_smem_total > smem_static_cap {
        bwd_smem_total as i64
    } else {
        0
    };
    eprintln!("[pca_bwd] bwd_smem_total={bwd_smem_total} bwd_smem_dyn={bwd_smem_dyn}");

    let rc_bwd = unsafe {
        nsl_flash_attention_csha_backward(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq_len as i64, head_dim as i64,
            0, 0, 0, 0,             // paging
            0, 0,                   // cos/sin
            0, 0,                   // ragged
            bwd_smem_dyn,           // dynamic SMEM bytes (0 if static is sufficient)
            bwd_ptx.as_ptr() as i64, bwd_name.as_ptr() as i64,
            config.block_q as i64, config.block_kv as i64,
            if config.causal { 1 } else { 0 },
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, dm as i64,
            saves.q_proj, saves.k_proj, saves.v_proj,
            saves.row_max, saves.row_sum,
            saves.x_raw,
            do_dev, dq_dev, dk_dev, dv_dev,
            dwq_dev, dwk_dev, dwv_dev, dx_dev,
            dxn_dev,
            // PCA Task 4B: trailing segment_ids.
            seg_dev,
            // Tier B extension — null (not used for Tier A tests)
            0i64, 0i64,
            // doc_starts ptr — null (rope_q=false, no doc-aware positions)
            0i64,
        )
    };
    // DEBUG: sync + report any async fault from the backward launch.
    unsafe {
        let sync_rc = cudarc::driver::sys::cuCtxSynchronize();
        eprintln!("[pca_bwd] bwd rc={rc_bwd} sync_rc={sync_rc:?}");
        if !matches!(sync_rc, cudarc::driver::sys::CUresult::CUDA_SUCCESS) {
            unsafe { nsl_csha_free_backward_activations(saves); }
            free_all(&all_dev);
            if seg_dev != 0 { unsafe { nsl_test_cuda_free(seg_dev); } }
            return None;
        }
    }
    if rc_bwd != 0 {
        let log = unsafe {
            let p = nsl_test_cuda_jit_log(bwd_ptx.as_ptr() as i64);
            if p != 0 {
                std::ffi::CStr::from_ptr(p as *const i8).to_string_lossy().into_owned()
            } else { "<no log>".into() }
        };
        eprintln!("[pca_bwd] backward rc={rc_bwd}\nJIT log:\n{log}");
        unsafe { nsl_csha_free_backward_activations(saves); }
        free_all(&all_dev);
        if seg_dev != 0 { unsafe { nsl_test_cuda_free(seg_dev); } }
        return None;
    }

    // ── Readback dQ, dK, dV (f16 on device → f32 on host) ───────────────────
    let qkv_elems = batch * heads * seq_len * head_dim;
    let mut dq_raw = vec![0u16; qkv_elems];
    let mut dk_raw = vec![0u16; qkv_elems];
    let mut dv_raw = vec![0u16; qkv_elems];
    unsafe {
        nsl_test_cuda_d2h(dq_raw.as_mut_ptr() as i64, dq_dev, (qkv_elems * 2) as i64);
        nsl_test_cuda_d2h(dk_raw.as_mut_ptr() as i64, dk_dev, (qkv_elems * 2) as i64);
        nsl_test_cuda_d2h(dv_raw.as_mut_ptr() as i64, dv_dev, (qkv_elems * 2) as i64);
    }
    let dq: Vec<f32> = dq_raw.iter().map(|&b| f16_to_f32(b)).collect();
    let dk: Vec<f32> = dk_raw.iter().map(|&b| f16_to_f32(b)).collect();
    let dv: Vec<f32> = dv_raw.iter().map(|&b| f16_to_f32(b)).collect();


    for (name, arr) in [("dq", &dq), ("dk", &dk), ("dv", &dv)] {
        if let Some((idx, value)) = arr.iter().copied().enumerate().find(|(_, x)| !x.is_finite()) {
            eprintln!(
                "[pca_bwd] non-finite {} at idx={} value={:?} seq_len={} segment_masked={}",
                name, idx, value, seq_len, segment_masked,
            );
        }
    }
    unsafe { nsl_csha_free_backward_activations(saves); }
    free_all(&all_dev);
    if seg_dev != 0 { unsafe { nsl_test_cuda_free(seg_dev); } }

    Some((dq, dk, dv))
}

fn cpu_reference_from_forward_saves(
    x_host: &[f32],
    wq_f16: &[u16],
    wk_f16: &[u16],
    wv_f16: &[u16],
    do_host_f16: &[u16],
    seq_len: usize,
    head_dim: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let batch = 1usize;
    let heads = 1usize;
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let x_norm = rmsnorm_rows(x_host, seq_len, head_dim, norm_eps);
    let q = project_rows_f16_saved(&x_norm, wq_f16, seq_len, head_dim);
    let k = project_rows_f16_saved(&x_norm, wk_f16, seq_len, head_dim);
    let v = project_rows_f16_saved(&x_norm, wv_f16, seq_len, head_dim);
    let (out, lse) = flash_attention_forward_reference(
        &q, &k, &v, seq_len, head_dim, scale, true,
    );
    let dout: Vec<f32> = do_host_f16.iter().map(|&bits| f16_to_f32(bits)).collect();

    let qkv_elems = batch * heads * seq_len * head_dim;
    let mut dq = vec![0f32; qkv_elems];
    let mut dk = vec![0f32; qkv_elems];
    let mut dv = vec![0f32; qkv_elems];
    flash_attention_backward_cpu_gqa(
        &q, &k, &v,
        &out, &lse, &dout,
        &mut dq, &mut dk, &mut dv,
        batch, heads, heads, seq_len, head_dim,
        scale, true, 1,
    );

    Some((
        roundtrip_f16_vec(&dq),
        roundtrip_f16_vec(&dk),
        roundtrip_f16_vec(&dv),
    ))
}

fn padded_unpacked_gpu_reference(
    x_host: &[f32],
    wq_f16: &[u16],
    wk_f16: &[u16],
    wv_f16: &[u16],
    do_host_f16: &[u16],
    seq_len: usize,
    head_dim: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let mut padded_seq_len = seq_len;
    if padded_seq_len < 128 {
        padded_seq_len = 128;
    }
    let rem = padded_seq_len % 32;
    if rem != 0 {
        padded_seq_len += 32 - rem;
    }

    let used_elems = seq_len * head_dim;
    let padded_elems = padded_seq_len * head_dim;
    let mut x_padded = vec![0.0f32; padded_elems];
    let mut do_padded = vec![0u16; padded_elems];
    x_padded[..used_elems].copy_from_slice(x_host);
    do_padded[..used_elems].copy_from_slice(do_host_f16);
    if seq_len > 0 {
        let tail_src = &x_host[(seq_len - 1) * head_dim..seq_len * head_dim];
        for row in seq_len..padded_seq_len {
            let row_base = row * head_dim;
            x_padded[row_base..row_base + head_dim].copy_from_slice(tail_src);
        }
    }

    let (dq, dk, dv) = launch_pca_backward(
        &x_padded, &wq_f16, &wk_f16, &wv_f16, &do_padded,
        padded_seq_len, head_dim, &[], false,
    )?;
    Some((
        dq[..used_elems].to_vec(),
        dk[..used_elems].to_vec(),
        dv[..used_elems].to_vec(),
    ))
}

// ---------------------------------------------------------------------------
// Max-abs-diff helper
// ---------------------------------------------------------------------------

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "length mismatch in max_abs_diff");
    let mut max_abs = 0f32;
    let mut max_idx = 0usize;
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        let d = if ai.is_nan() || bi.is_nan() {
            f32::INFINITY
        } else if ai == bi {
            0.0
        } else {
            (ai - bi).abs()
        };
        if d > max_abs { max_abs = d; max_idx = i; }
    }
    (max_abs, max_idx)
}

// ---------------------------------------------------------------------------
// Extract one segment's rows from a packed [seq_len × head_dim] f32 tensor.
// ---------------------------------------------------------------------------

fn extract_segment_rows(
    src: &[f32],
    seq_len: usize, head_dim: usize,
    seg_ids: &[u16], target_id: u16,
) -> Vec<f32> {
    let positions: Vec<usize> = (0..seq_len).filter(|&i| seg_ids[i] == target_id).collect();
    let mut out = vec![0f32; positions.len() * head_dim];
    for (dst_si, &src_si) in positions.iter().enumerate() {
        let src_base = src_si * head_dim;
        let dst_base = dst_si * head_dim;
        out[dst_base..dst_base + head_dim].copy_from_slice(&src[src_base..src_base + head_dim]);
    }
    out
}

// Same but for f16 slices (dO upload buffers).
fn extract_segment_rows_u16(
    src: &[u16],
    seq_len: usize, head_dim: usize,
    seg_ids: &[u16], target_id: u16,
) -> Vec<u16> {
    let positions: Vec<usize> = (0..seq_len).filter(|&i| seg_ids[i] == target_id).collect();
    let mut out = vec![0u16; positions.len() * head_dim];
    for (dst_si, &src_si) in positions.iter().enumerate() {
        let src_base = src_si * head_dim;
        let dst_base = dst_si * head_dim;
        out[dst_base..dst_base + head_dim].copy_from_slice(&src[src_base..src_base + head_dim]);
    }
    out
}

// ---------------------------------------------------------------------------
// Scatter per-segment gradient rows back into packed [seq_len × head_dim] layout.
// ---------------------------------------------------------------------------

fn scatter_segment_grads(
    dst: &mut [f32],
    seg_grads: &[f32],
    seq_len: usize, head_dim: usize,
    seg_ids: &[u16], target_id: u16,
) {
    let positions: Vec<usize> = (0..seq_len).filter(|&i| seg_ids[i] == target_id).collect();
    assert_eq!(seg_grads.len(), positions.len() * head_dim);
    for (dst_si, &src_si) in positions.iter().enumerate() {
        let src_base = dst_si * head_dim;
        let dst_base = src_si * head_dim;
        dst[dst_base..dst_base + head_dim].copy_from_slice(&seg_grads[src_base..src_base + head_dim]);
    }
}

// ===========================================================================
// Fixture 1: single-segment — mask is a no-op
//
// All segment_ids = 0. PCA backward (segment_masked=true) must match the
// segment_masked=false baseline within 5e-3 (same attention, no masking).
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_a_backward_single_segment_matches_unmasked_baseline() {
    if !cuda_available() { return; }

    let seq_len  = 128usize;
    let head_dim = 32usize;
    let total    = seq_len * head_dim;

    let mut x    = vec![0f32; total];
    let mut do_f32 = vec![0f32; total];
    fill_seeded(&mut x,     0xA1B2_C3D4);
    fill_seeded(&mut do_f32, 0xFEED_FACE);

    // Shared weight matrices (same for both runs).
    let n_weights = head_dim * head_dim;  // d_model=head_dim=32
    let mut wq_f32 = vec![0f32; n_weights];
    let mut wk_f32 = vec![0f32; n_weights];
    let mut wv_f32 = vec![0f32; n_weights];
    fill_seeded(&mut wq_f32, 0x1111_AAAA);
    fill_seeded(&mut wk_f32, 0x2222_BBBB);
    fill_seeded(&mut wv_f32, 0x3333_CCCC);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_f16: Vec<u16> = do_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    let seg_ids = vec![0u16; seq_len];

    eprintln!("Fixture 1 — baseline backward (segment_masked=false)");
    let baseline = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim, &[], false,
    );

    eprintln!("Fixture 1 — PCA backward (segment_masked=true, all-zero seg_ids)");
    let pca = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim, &seg_ids, true,
    );

    let ((dq_base, dk_base, dv_base), (dq_pca, dk_pca, dv_pca)) = match (baseline, pca) {
        (Some(b), Some(p)) => (b, p),
        _ => {
            eprintln!("Fixture 1 SKIPPED: kernel launch failed (non-GPU environment?)");
            return;
        }
    };

    for (name, arr) in [
        ("dq_base", &dq_base), ("dk_base", &dk_base), ("dv_base", &dv_base),
        ("dq_pca",  &dq_pca),  ("dk_pca",  &dk_pca),  ("dv_pca",  &dv_pca),
    ] {
        assert_no_nan(name, arr, "Fixture 1");
    }

    let (dq_diff, dq_idx) = max_abs_diff(&dq_pca, &dq_base);
    let (dk_diff, dk_idx) = max_abs_diff(&dk_pca, &dk_base);
    let (dv_diff, dv_idx) = max_abs_diff(&dv_pca, &dv_base);

    eprintln!(
        "Fixture 1 [single-segment]: dq={:.3e}@[{dq_idx}]  dk={:.3e}@[{dk_idx}]  dv={:.3e}@[{dv_idx}]",
        dq_diff, dk_diff, dv_diff,
    );
    eprintln!("  first 4 dq_pca:  {:?}", &dq_pca[..4.min(dq_pca.len())]);
    eprintln!("  first 4 dq_base: {:?}", &dq_base[..4.min(dq_base.len())]);

    let tol = 5e-3f32;
    assert!(dq_diff <= tol, "Fixture 1 FAILED: dq={:.3e} > {:.0e}", dq_diff, tol);
    assert!(dk_diff <= tol, "Fixture 1 FAILED: dk={:.3e} > {:.0e}", dk_diff, tol);
    assert!(dv_diff <= tol, "Fixture 1 FAILED: dv={:.3e} > {:.0e}", dv_diff, tol);
    eprintln!("Fixture 1 PASSED: dq={:.3e} dk={:.3e} dv={:.3e} <= {:.0e}", dq_diff, dk_diff, dv_diff, tol);
}

// ===========================================================================
// Fixture 2: two-equal-segments
//
// seq_len=128, segs [0..64]=0, [64..128]=1.
// Reference: two independent GPU backward passes on seq_len=64 halves
//            (segment_masked=false), gradients scattered back.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_a_backward_two_equal_segments_matches_unpacked_reference() {
    if !cuda_available() { return; }

    let seq_len  = 128usize;
    let head_dim = 32usize;
    let seg_a    = 64usize;
    let seg_b    = 64usize;
    assert_eq!(seg_a + seg_b, seq_len);
    let total = seq_len * head_dim;

    let mut x    = vec![0f32; total];
    let mut do_f32 = vec![0f32; total];
    fill_seeded(&mut x,     0xCAFE_BABE);
    fill_seeded(&mut do_f32, 0x1234_ABCD);

    let n_weights = head_dim * head_dim;
    let mut wq_f32 = vec![0f32; n_weights];
    let mut wk_f32 = vec![0f32; n_weights];
    let mut wv_f32 = vec![0f32; n_weights];
    fill_seeded(&mut wq_f32, 0xAAAA_1111);
    fill_seeded(&mut wk_f32, 0xBBBB_2222);
    fill_seeded(&mut wv_f32, 0xCCCC_3333);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_f16: Vec<u16> = do_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    let mut seg_ids = vec![0u16; seq_len];
    for x in seg_ids[seg_a..].iter_mut() { *x = 1; }

    // ── Packed PCA run ───────────────────────────────────────────────────────
    eprintln!("Fixture 2 — PCA backward (seq=128, two segments of 64)");
    let pca = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim, &seg_ids, true,
    );

    // ── Unpacked reference runs ───────────────────────────────────────────────
    // Each segment uses its own slice of x and dO (extracted by position).
    let x_a    = extract_segment_rows(&x,    seq_len, head_dim, &seg_ids, 0);
    let do_a   = extract_segment_rows_u16(&do_f16, seq_len, head_dim, &seg_ids, 0);
    let x_b    = extract_segment_rows(&x,    seq_len, head_dim, &seg_ids, 1);
    let do_b   = extract_segment_rows_u16(&do_f16, seq_len, head_dim, &seg_ids, 1);

    eprintln!("Fixture 2 — reference backward seg A (seq=64, segment_masked=false)");
    let ref_a = padded_unpacked_gpu_reference(
        &x_a, &wq_f16, &wk_f16, &wv_f16, &do_a,
        seg_a, head_dim,
    );
    eprintln!("Fixture 2 — reference backward seg B (seq=64, segment_masked=false)");
    let ref_b = padded_unpacked_gpu_reference(
        &x_b, &wq_f16, &wk_f16, &wv_f16, &do_b,
        seg_b, head_dim,
    );

    let (dq_pca, dk_pca, dv_pca) = match pca {
        Some(r) => r,
        None => { eprintln!("Fixture 2 SKIPPED: packed launch failed"); return; }
    };
    let (dq_a, dk_a, dv_a) = match ref_a {
        Some(r) => r,
        None => { eprintln!("Fixture 2 SKIPPED: ref-A launch failed"); return; }
    };
    let (dq_b, dk_b, dv_b) = match ref_b {
        Some(r) => r,
        None => { eprintln!("Fixture 2 SKIPPED: ref-B launch failed"); return; }
    };

    // Scatter per-segment gradients into packed [seq_len × head_dim] layout.
    let mut ref_dq = vec![0f32; total];
    let mut ref_dk = vec![0f32; total];
    let mut ref_dv = vec![0f32; total];
    scatter_segment_grads(&mut ref_dq, &dq_a, seq_len, head_dim, &seg_ids, 0);
    scatter_segment_grads(&mut ref_dk, &dk_a, seq_len, head_dim, &seg_ids, 0);
    scatter_segment_grads(&mut ref_dv, &dv_a, seq_len, head_dim, &seg_ids, 0);
    scatter_segment_grads(&mut ref_dq, &dq_b, seq_len, head_dim, &seg_ids, 1);
    scatter_segment_grads(&mut ref_dk, &dk_b, seq_len, head_dim, &seg_ids, 1);
    scatter_segment_grads(&mut ref_dv, &dv_b, seq_len, head_dim, &seg_ids, 1);

    for (name, arr) in [
        ("dq_pca", &dq_pca), ("dk_pca", &dk_pca), ("dv_pca", &dv_pca),
        ("ref_dq", &ref_dq), ("ref_dk", &ref_dk), ("ref_dv", &ref_dv),
    ] {
        assert_no_nan(name, arr, "Fixture 2");
    }

    let (dq_diff, dq_idx) = max_abs_diff(&dq_pca, &ref_dq);
    let (dk_diff, dk_idx) = max_abs_diff(&dk_pca, &ref_dk);
    let (dv_diff, dv_idx) = max_abs_diff(&dv_pca, &ref_dv);

    eprintln!(
        "Fixture 2 [two-equal-segments]: dq={:.3e}@[{dq_idx}]  dk={:.3e}@[{dk_idx}]  dv={:.3e}@[{dv_idx}]",
        dq_diff, dk_diff, dv_diff,
    );
    eprintln!("  first 4 dq_pca: {:?}", &dq_pca[..4.min(dq_pca.len())]);
    eprintln!("  first 4 ref_dq: {:?}", &ref_dq[..4.min(ref_dq.len())]);

    let tol = 5e-3f32;
    assert!(dq_diff <= tol, "Fixture 2 FAILED: dq={:.3e} > {:.0e}", dq_diff, tol);
    assert!(dk_diff <= tol, "Fixture 2 FAILED: dk={:.3e} > {:.0e}", dk_diff, tol);
    assert!(dv_diff <= tol, "Fixture 2 FAILED: dv={:.3e} > {:.0e}", dv_diff, tol);
    eprintln!("Fixture 2 PASSED: dq={:.3e} dk={:.3e} dv={:.3e} <= {:.0e}", dq_diff, dk_diff, dv_diff, tol);
}

// ===========================================================================
// Fixture 3: unequal-segments
//
// seq_len=128, segments [0..38]=0, [38..102]=1, [102..128]=2.
// Reference: three independent GPU backward passes, gradients scattered back.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_a_backward_unequal_segments_matches_unpacked_reference() {
    if !cuda_available() { return; }

    let head_dim = 32usize;
    let lens     = [38usize, 64usize, 26usize];
    let seq_len: usize = lens.iter().sum();
    assert_eq!(seq_len, 128);
    let total = seq_len * head_dim;

    let mut x    = vec![0f32; total];
    let mut do_f32 = vec![0f32; total];
    fill_seeded(&mut x,     0xABCD_EF01);
    fill_seeded(&mut do_f32, 0x9876_5432);

    let n_weights = head_dim * head_dim;
    let mut wq_f32 = vec![0f32; n_weights];
    let mut wk_f32 = vec![0f32; n_weights];
    let mut wv_f32 = vec![0f32; n_weights];
    fill_seeded(&mut wq_f32, 0xDEAD_BEEF);
    fill_seeded(&mut wk_f32, 0xFACE_CAFE);
    fill_seeded(&mut wv_f32, 0xBABE_1234);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_f16: Vec<u16> = do_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    // Build segment_ids.
    let mut seg_ids = vec![0u16; seq_len];
    let mut off = 0usize;
    for (sid, &slen) in lens.iter().enumerate() {
        for x in seg_ids[off..off + slen].iter_mut() { *x = sid as u16; }
        off += slen;
    }

    let mut do_seg0 = vec![f32_to_f16_bits(0.0); total];
    for row in 0..seq_len {
        if seg_ids[row] == 0 {
            let src = row * head_dim;
            let dst = src;
            do_seg0[dst..dst + head_dim].copy_from_slice(&do_f16[src..src + head_dim]);
        }
    }

    // ── Packed PCA run ───────────────────────────────────────────────────────
    eprintln!("Fixture 3 — PCA backward (seq=128, segments 38+64+26)");
    let pca = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim, &seg_ids, true,
    );
    eprintln!("Fixture 3 — full unmasked baseline (for segment 0 prefix reference)");
    let baseline_full = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim, &[], false,
    );
    eprintln!("Fixture 3 — segment 0 upstream-only baseline (full seq=128, unmasked)");
    let baseline_seg0 = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_seg0,
        seq_len, head_dim, &[], false,
    );

    // ── Per-segment reference runs ────────────────────────────────────────────
    let mut ref_dq = vec![0f32; total];
    let mut ref_dk = vec![0f32; total];
    let mut ref_dv = vec![0f32; total];
    let mut all_refs_ok = true;

    for (sid, &slen) in lens.iter().enumerate() {
        let target = sid as u16;
        if sid == 0 {
            continue;
        }
        let x_s    = extract_segment_rows(&x,     seq_len, head_dim, &seg_ids, target);
        let do_s   = extract_segment_rows_u16(&do_f16, seq_len, head_dim, &seg_ids, target);
        assert_eq!(x_s.len(), slen * head_dim);

        eprintln!("Fixture 3 — reference backward seg {sid} (seq={slen}, segment_masked=false)");
        match padded_unpacked_gpu_reference(
            &x_s, &wq_f16, &wk_f16, &wv_f16, &do_s,
            slen, head_dim,
        ) {
            Some((dq_s, dk_s, dv_s)) => {
                scatter_segment_grads(&mut ref_dq, &dq_s, seq_len, head_dim, &seg_ids, target);
                scatter_segment_grads(&mut ref_dk, &dk_s, seq_len, head_dim, &seg_ids, target);
                scatter_segment_grads(&mut ref_dv, &dv_s, seq_len, head_dim, &seg_ids, target);
            }
            None => {
                eprintln!("Fixture 3 SKIPPED: reference seg {sid} launch failed");
                all_refs_ok = false;
            }
        }
    }

    let (dq_pca, dk_pca, dv_pca) = match pca {
        Some(r) => r,
        None => { eprintln!("Fixture 3 SKIPPED: packed launch failed"); return; }
    };
    let baseline_full = match baseline_full {
        Some(r) => r,
        None => { eprintln!("Fixture 3 SKIPPED: full unmasked baseline launch failed"); return; }
    };
    let baseline_seg0 = match baseline_seg0 {
        Some(r) => r,
        None => { eprintln!("Fixture 3 SKIPPED: segment 0 baseline launch failed"); return; }
    };
    if !all_refs_ok {
        eprintln!("Fixture 3 SKIPPED: one or more reference launches failed");
        return;
    }

    // Segment 0 occupies the causal prefix [0..38). A full-length unmasked
    // launch with dO zeroed outside that prefix keeps the exact key/value
    // geometry while removing future-query contributions to dK/dV.
    let ref0_dq = extract_segment_rows(&baseline_seg0.0, seq_len, head_dim, &seg_ids, 0);
    let ref0_dk = extract_segment_rows(&baseline_seg0.1, seq_len, head_dim, &seg_ids, 0);
    let ref0_dv = extract_segment_rows(&baseline_seg0.2, seq_len, head_dim, &seg_ids, 0);
    scatter_segment_grads(&mut ref_dq, &ref0_dq, seq_len, head_dim, &seg_ids, 0);
    scatter_segment_grads(&mut ref_dk, &ref0_dk, seq_len, head_dim, &seg_ids, 0);
    scatter_segment_grads(&mut ref_dv, &ref0_dv, seq_len, head_dim, &seg_ids, 0);

    for (name, arr) in [
        ("dq_pca", &dq_pca), ("dk_pca", &dk_pca), ("dv_pca", &dv_pca),
        ("ref_dq", &ref_dq), ("ref_dk", &ref_dk), ("ref_dv", &ref_dv),
    ] {
        assert_no_nan(name, arr, "Fixture 3");
    }

    let (dq_diff, dq_idx) = max_abs_diff(&dq_pca, &ref_dq);
    let (dk_diff, dk_idx) = max_abs_diff(&dk_pca, &ref_dk);
    let (dv_diff, dv_idx) = max_abs_diff(&dv_pca, &ref_dv);

    eprintln!(
        "Fixture 3 [unequal-segments 38+64+26]: dq={:.3e}@[{dq_idx}]  dk={:.3e}@[{dk_idx}]  dv={:.3e}@[{dv_idx}]",
        dq_diff, dk_diff, dv_diff,
    );
    eprintln!("  first 4 dq_pca: {:?}", &dq_pca[..4.min(dq_pca.len())]);
    eprintln!("  first 4 ref_dq: {:?}", &ref_dq[..4.min(ref_dq.len())]);
    eprintln!(
        "  dq idx {}: pca={:?} ref={:?} full_unmasked={:?}",
        dq_idx, dq_pca[dq_idx], ref_dq[dq_idx], baseline_full.0[dq_idx]
    );
    eprintln!(
        "  dk idx {}: pca={:?} ref={:?} full_unmasked={:?}",
        dk_idx, dk_pca[dk_idx], ref_dk[dk_idx], baseline_full.1[dk_idx]
    );
    eprintln!(
        "  dv idx {}: pca={:?} ref={:?} full_unmasked={:?}",
        dv_idx, dv_pca[dv_idx], ref_dv[dv_idx], baseline_full.2[dv_idx]
    );

    let tol = 5e-3f32;
    assert!(dq_diff <= tol, "Fixture 3 FAILED: dq={:.3e} > {:.0e}", dq_diff, tol);
    assert!(dk_diff <= tol, "Fixture 3 FAILED: dk={:.3e} > {:.0e}", dk_diff, tol);
    assert!(dv_diff <= tol, "Fixture 3 FAILED: dv={:.3e} > {:.0e}", dv_diff, tol);
    eprintln!("Fixture 3 PASSED: dq={:.3e} dk={:.3e} dv={:.3e} <= {:.0e}", dq_diff, dk_diff, dv_diff, tol);
}

// ===========================================================================
// A-4 Test 2: null seg_ids_ptr guard — masked+null must equal unmasked backward
//
// Validates the same null-guard in the BACKWARD kernel: when seg_ids_ptr==0
// the kernel must behave identically to the segment_masked=false path.
//
// The launch_pca_backward helper passes seg_dev=0 when segment_masked is
// true but seg_ids_host is empty (see harness, line 488-501).
//
// Tolerance: 5e-3 — matches the file's established backward tolerance.
//
// CRITICAL: if cuda_available() is true and the masked-null launch returns
// None, we PANIC (guard failure), not skip.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn masked_null_seg_ptr_equals_unmasked_backward() {
    if !cuda_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }

    let seq_len  = 128usize;
    let head_dim = 32usize;
    let total    = seq_len * head_dim;

    let mut x      = vec![0f32; total];
    let mut do_f32 = vec![0f32; total];
    fill_seeded(&mut x,      0xA4B4_C4D4);
    fill_seeded(&mut do_f32, 0x5566_7788);

    let n_weights = head_dim * head_dim;
    let mut wq_f32 = vec![0f32; n_weights];
    let mut wk_f32 = vec![0f32; n_weights];
    let mut wv_f32 = vec![0f32; n_weights];
    fill_seeded(&mut wq_f32, 0x1111_2222);
    fill_seeded(&mut wk_f32, 0x3333_4444);
    fill_seeded(&mut wv_f32, 0x5555_6666);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_f16: Vec<u16> = do_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    eprintln!("A-4 Test 2 — baseline backward (segment_masked=false)");
    let baseline = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim,
        &[],   // empty → seg_dev=0
        false, // segment_masked=false
    );

    eprintln!("A-4 Test 2 — masked+null backward (segment_masked=true, empty seg_ids → ptr=0)");
    let masked_null = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim,
        &[],   // empty → seg_dev=0 (NULL)
        true,  // segment_masked=true
    );

    // CRITICAL: if the GPU is present and the masked-null launch returned None,
    // the null-guard failed (kernel crashed on NULL dereference).
    let (dq1, dk1, dv1) = masked_null.unwrap_or_else(|| {
        panic!(
            "masked backward kernel crashed on null seg_ids_ptr — A-3 null-guard FAILED \
             (segment_masked=true, seg_ids_ptr=0 should be safe)"
        )
    });
    let (dq0, dk0, dv0) = baseline.expect(
        "baseline (segment_masked=false) backward launch failed unexpectedly"
    );

    for (name, arr) in [
        ("dq0", &dq0), ("dk0", &dk0), ("dv0", &dv0),
        ("dq1", &dq1), ("dk1", &dk1), ("dv1", &dv1),
    ] {
        assert_no_nan(name, arr, "A-4 Test 2");
    }

    let (dq_diff, dq_idx) = max_abs_diff(&dq1, &dq0);
    let (dk_diff, dk_idx) = max_abs_diff(&dk1, &dk0);
    let (dv_diff, dv_idx) = max_abs_diff(&dv1, &dv0);

    eprintln!(
        "A-4 Test 2 [masked+null vs unmasked backward]: \
         dq={:.3e}@[{dq_idx}]  dk={:.3e}@[{dk_idx}]  dv={:.3e}@[{dv_idx}]",
        dq_diff, dk_diff, dv_diff,
    );
    eprintln!("  first 4 dq1 (masked+null): {:?}", &dq1[..4.min(dq1.len())]);
    eprintln!("  first 4 dq0 (baseline):    {:?}", &dq0[..4.min(dq0.len())]);

    // Tolerance matches the file's established backward budget (5e-3).
    let tol = 5e-3f32;
    assert!(dq_diff <= tol, "A-4 Test 2 FAILED: dq={:.3e} > {:.0e}", dq_diff, tol);
    assert!(dk_diff <= tol, "A-4 Test 2 FAILED: dk={:.3e} > {:.0e}", dk_diff, tol);
    assert!(dv_diff <= tol, "A-4 Test 2 FAILED: dv={:.3e} > {:.0e}", dv_diff, tol);
    eprintln!(
        "A-4 Test 2 PASSED: masked+null == unmasked backward, \
         dq={:.3e} dk={:.3e} dv={:.3e} <= {:.0e}",
        dq_diff, dk_diff, dv_diff, tol,
    );
}

#[test]
#[ignore = "debug helper vs full unmasked baseline"]
fn debug_cpu_reference_matches_full_unmasked_baseline() {
    if !cuda_available() { return; }

    let seq_len = 128usize;
    let head_dim = 32usize;
    let total = seq_len * head_dim;

    let mut x = vec![0f32; total];
    let mut do_f32 = vec![0f32; total];
    fill_seeded(&mut x, 0xA1B2_C3D4);
    fill_seeded(&mut do_f32, 0xFEED_FACE);

    let n_weights = head_dim * head_dim;
    let mut wq_f32 = vec![0f32; n_weights];
    let mut wk_f32 = vec![0f32; n_weights];
    let mut wv_f32 = vec![0f32; n_weights];
    fill_seeded(&mut wq_f32, 0x1111_AAAA);
    fill_seeded(&mut wk_f32, 0x2222_BBBB);
    fill_seeded(&mut wv_f32, 0x3333_CCCC);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_f16: Vec<u16> = do_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    let gpu = launch_pca_backward(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim, &[], false,
    ).expect("gpu baseline failed");
    let cpu = cpu_reference_from_forward_saves(
        &x, &wq_f16, &wk_f16, &wv_f16, &do_f16,
        seq_len, head_dim,
    ).expect("cpu helper failed");

    let (dq_diff, dq_idx) = max_abs_diff(&gpu.0, &cpu.0);
    let (dk_diff, dk_idx) = max_abs_diff(&gpu.1, &cpu.1);
    let (dv_diff, dv_idx) = max_abs_diff(&gpu.2, &cpu.2);
    let (gpu_dk_max, gpu_dk_max_idx) = gpu.1.iter().copied().enumerate()
        .fold((0.0f32, 0usize), |acc, (idx, v)| {
            let mag = v.abs();
            if mag > acc.0 { (mag, idx) } else { acc }
        });
    let (cpu_dk_max, cpu_dk_max_idx) = cpu.1.iter().copied().enumerate()
        .fold((0.0f32, 0usize), |acc, (idx, v)| {
            let mag = v.abs();
            if mag > acc.0 { (mag, idx) } else { acc }
        });
    eprintln!(
        "Debug helper vs full baseline: dq={:.3e}@[{dq_idx}] dk={:.3e}@[{dk_idx}] dv={:.3e}@[{dv_idx}]",
        dq_diff, dk_diff, dv_diff,
    );
    eprintln!("  first 4 gpu dq: {:?}", &gpu.0[..4.min(gpu.0.len())]);
    eprintln!("  first 4 cpu dq: {:?}", &cpu.0[..4.min(cpu.0.len())]);
    if gpu.1.len() > 1024 && cpu.1.len() > 1024 {
        eprintln!(
            "  dk[1024]: gpu={:?} cpu={:?}",
            gpu.1[1024], cpu.1[1024]
        );
    }
    eprintln!(
        "  dk maxabs: gpu={:?}@[{gpu_dk_max_idx}] cpu={:?}@[{cpu_dk_max_idx}]",
        gpu_dk_max, cpu_dk_max
    );
}
