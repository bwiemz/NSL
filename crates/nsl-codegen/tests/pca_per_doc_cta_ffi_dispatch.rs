//! PCA per-document CTA FFI dispatch integration test.
//!
//! Companion to `pca_per_doc_cta_forward_correctness.rs`: that test calls
//! `nsl_kernel_launch` directly, bypassing the production
//! `nsl_flash_attention_csha` FFI. This test exercises the FFI dispatch
//! path: the runtime detects the `_per_doc_cta` kernel-name suffix,
//! overrides `grid_x` to `num_docs_or_zero`, and forwards to the same
//! kernel.
//!
//! What this test proves:
//!   * `csha_is_per_doc_cta_kernel` correctly matches the
//!     `_per_doc_cta` suffix emitted by `per_doc_cta_kernel_name`.
//!   * The runtime overrides `grid_x` from
//!     `ceil(seq_len / block_q) = 2` (legacy topology with seq=128, bq=64)
//!     to `num_docs_or_zero = 4` (per-doc topology with 4 docs).
//!   * Numerical output matches the same CPU reference that the
//!     `nsl_kernel_launch` path validated, proving the FFI dispatch is
//!     a transparent re-route.
//!
//! Fixture: identical to `per_doc_cta_four_short_docs_matches_segmented_reference`:
//!   4 docs of lengths {32,28,24,16} packed into seq=128 (active=100),
//!   doc_starts = [0,32,60,84,100,128].  bq=64, hd=32, batch=1, heads=1.
//!
//! Tolerance: 5e-3 (matches the v1 test; identical kernel via identical
//! launch geometry — any tightening would have to apply to the v1 test too).
//!
//! Run with:
//!   cargo test -p nsl-codegen --features cuda --test pca_per_doc_cta_ffi_dispatch \
//!     -- --ignored --nocapture

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::per_doc_cta::{
    per_doc_cta_kernel_name, synthesize_per_doc_cta_forward,
};
use nsl_codegen::pca_detect::{DatasetPackingConfig, PcaDetection, PcaStrategy};
use nsl_codegen::pca_per_doc::{PerDocAdmitConfig, admit};
use std::ffi::CString;

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_d2h,
};
use nsl_runtime::flash_attention::nsl_flash_attention_csha;

// ---------------------------------------------------------------------------
// IEEE 754 half-precision decode (no std f16 support)
// ---------------------------------------------------------------------------

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp  = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
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

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

fn fill_seeded(dst: &mut [f32], mut seed: u64) {
    for x in dst.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (seed >> 33) as u32;
        *x = ((u as f32) / (u32::MAX as f32)) - 0.5;
    }
}

// ---------------------------------------------------------------------------
// CPU naive segmented attention (causal + per-document isolation)
// ---------------------------------------------------------------------------

fn naive_attention_segmented(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32],
    b: usize, h: usize, s: usize, d: usize,
    scale: f32,
    causal: bool,
    seg_ids: &[u16],
) {
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..s {
                let k_last = if causal { qi } else { s - 1 };
                let q_base = ((bi * h + hi) * s + qi) * d;
                let qv = &q[q_base..q_base + d];
                let q_seg = if seg_ids.is_empty() { 0u16 } else { seg_ids[qi] };

                let mut scores = vec![f32::NEG_INFINITY; s];
                let mut max_s = f32::NEG_INFINITY;

                for kj in 0..=k_last {
                    let k_seg = if seg_ids.is_empty() { 0u16 } else { seg_ids[kj] };
                    if !seg_ids.is_empty() && q_seg != k_seg { continue; }
                    let kv_base = ((bi * h + hi) * s + kj) * d;
                    let kv = &k[kv_base..kv_base + d];
                    let mut s_val = 0f32;
                    for dd in 0..d { s_val += qv[dd] * kv[dd]; }
                    s_val *= scale;
                    scores[kj] = s_val;
                    if s_val > max_s { max_s = s_val; }
                }

                if max_s == f32::NEG_INFINITY {
                    let out_base = ((bi * h + hi) * s + qi) * d;
                    for dd in 0..d { out[out_base + dd] = 0.0; }
                    continue;
                }

                let mut sum_exp = 0f32;
                for kj in 0..=k_last {
                    if scores[kj].is_finite() {
                        scores[kj] = (scores[kj] - max_s).exp();
                        sum_exp += scores[kj];
                    } else {
                        scores[kj] = 0.0;
                    }
                }

                let out_base = ((bi * h + hi) * s + qi) * d;
                for dd in 0..d {
                    let mut acc = 0f32;
                    for kj in 0..=k_last {
                        acc += (scores[kj] / sum_exp) * v[((bi * h + hi) * s + kj) * d + dd];
                    }
                    out[out_base + dd] = acc;
                }
            }
        }
    }
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0f32;
    let mut max_idx = 0usize;
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        let d = (ai - bi).abs();
        if d > max_abs { max_abs = d; max_idx = i; }
    }
    (max_abs, max_idx)
}

// ===========================================================================
// Helper: assert the FFI helper correctly classifies kernel names.
// Pure host-side check — does not require a live CUDA device.
// ===========================================================================

#[test]
fn ffi_kernel_name_classifier_recognizes_per_doc_cta_suffix() {
    // We don't have direct access to the runtime's private helper, but
    // we *can* verify that `per_doc_cta_kernel_name` produces a string
    // that the FFI's documented matcher (`name.contains("_per_doc_cta")`)
    // would accept.
    let cfg = FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 32,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, segment_masked: false, csha: None,
    };
    let kname = per_doc_cta_kernel_name(&cfg);
    assert!(
        kname.contains("_per_doc_cta"),
        "kernel name `{}` must carry the `_per_doc_cta` suffix the FFI matches on",
        kname,
    );
    // The matching tier_b1 precedent uses `_tier_b1` — make sure we
    // don't collide.
    assert!(
        !kname.contains("_tier_b1"),
        "kernel name `{}` must not collide with the `_tier_b1` suffix",
        kname,
    );
}

// ===========================================================================
// GPU test: production FFI dispatch matches CPU reference.
//
// 4 short docs packed into seq=128 (active=100), bq=64.  The legacy
// topology would launch grid_x=ceil(128/64)=2 CTAs; the per-doc dispatch
// launches grid_x=num_docs=4 CTAs.  If the FFI dispatch were broken
// (silently falling back to legacy grid_x), CTAs 2 and 3 would never
// fire and docs 2 and 3 of the output would be zero / NaN.  The
// numerical comparison detects this.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn per_doc_cta_ffi_dispatch_matches_segmented_reference() {
    if !cuda_available() { return; }

    // Same fixture as `pca_per_doc_cta_forward_correctness.rs`.
    let doc_lens = [32usize, 28, 24, 16];
    let active_seq_len: usize = doc_lens.iter().sum(); // 100
    let seq_len = 128usize; // padded
    let batch = 1usize;
    let heads = 1usize;
    let head_dim = 32usize;

    // Build doc_starts: [0, 32, 60, 84, 100, 128].
    let mut doc_starts_host = vec![0i32; doc_lens.len() + 2];
    let mut off = 0usize;
    for (i, &l) in doc_lens.iter().enumerate() {
        doc_starts_host[i] = off as i32;
        off += l;
    }
    doc_starts_host[doc_lens.len()] = active_seq_len as i32;
    doc_starts_host[doc_lens.len() + 1] = seq_len as i32;

    // Build seg_ids: position i in doc d gets seg_ids[i]=d.
    let mut seg_ids = vec![0u16; seq_len];
    let mut pos = 0usize;
    for (d, &l) in doc_lens.iter().enumerate() {
        for _ in 0..l {
            seg_ids[pos] = d as u16;
            pos += 1;
        }
    }

    // Generate Q/K/V data with the same seeds the v1 test uses so the
    // CPU references are bit-identical and any FFI-dispatch regression
    // can be diffed against the v1 baseline.
    let total = batch * heads * seq_len * head_dim;
    let mut q = vec![0f32; total];
    let mut k = vec![0f32; total];
    let mut v = vec![0f32; total];
    fill_seeded(&mut q, 0xDEAD_4D0C_u64);
    fill_seeded(&mut k, 0xBEEF_4D0C_u64);
    fill_seeded(&mut v, 0xCAFE_4D0C_u64);

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut cpu_ref = vec![0f32; total];
    naive_attention_segmented(
        &q, &k, &v, &mut cpu_ref,
        batch, heads, seq_len, head_dim,
        scale, true, &seg_ids,
    );

    let config = FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 32,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, segment_masked: false, csha: None,
    };

    let packing = DatasetPackingConfig {
        enabled: true,
        max_sequence_length: seq_len as u32,
        mean_doc_length: Some(25),
        doc_length_stddev: Some(0),
        separator_token_id: Some(2),
    };
    let det = PcaDetection {
        strategy: PcaStrategy::PerDocumentCta,
        expected_doc_fraction: 0.195,
        rationale: "ffi-dispatch test".to_string(),
        segment_id_bytes_per_batch: 256,
        eliminated_mask_bytes_per_batch: 32768,
    };
    let plan = admit(
        &det, &config, &packing,
        &PerDocAdmitConfig { enable_per_doc_cta: true, ..Default::default() },
    ).expect("plan must admit");

    // Device allocations.
    let f32_bytes = (total * std::mem::size_of::<f32>()) as i64;
    let f16_bytes = (total * std::mem::size_of::<u16>()) as i64;
    let lse_bytes = (batch * heads * seq_len * std::mem::size_of::<f32>()) as i64;
    let ds_bytes = (doc_starts_host.len() * std::mem::size_of::<i32>()) as i64;

    let q_dev   = nsl_test_cuda_alloc(f32_bytes);
    let k_dev   = nsl_test_cuda_alloc(f32_bytes);
    let v_dev   = nsl_test_cuda_alloc(f32_bytes);
    let out_dev = nsl_test_cuda_alloc(f16_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);
    let doc_starts_dev = nsl_test_cuda_alloc(ds_bytes);
    assert!(
        q_dev != 0 && k_dev != 0 && v_dev != 0 && out_dev != 0
            && lse_dev != 0 && doc_starts_dev != 0,
        "device alloc returned null",
    );

    nsl_test_cuda_h2d(q_dev,   q.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(k_dev,   k.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(v_dev,   v.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(doc_starts_dev, doc_starts_host.as_ptr() as i64, ds_bytes);

    // PTX synthesis — same emitter the v1 test calls.
    let mut ptx = synthesize_per_doc_cta_forward(&plan, &config);
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    ptx.push(0);

    let kernel_name = CString::new(per_doc_cta_kernel_name(&config)).unwrap();
    let smem = nsl_codegen::flash_attention_v2::smem_layout::total_bytes(&config) as i64;

    // num_docs sentinel walk — same helper the FFI's launchers
    // *would* use if Option 2 had been chosen.  Here it just feeds the
    // explicit `num_docs_or_zero` arg.
    let num_docs = nsl_codegen::pca_per_doc::compute_per_doc_grid_x(
        &doc_starts_host, active_seq_len as i32, 256,
    ) as i64;
    assert_eq!(num_docs, 4, "fixture must have exactly 4 docs");

    eprintln!(
        "[ffi-dispatch] kernel=\"{}\" num_docs={} legacy_grid_x_would_be={}",
        kernel_name.to_string_lossy(),
        num_docs,
        (seq_len as i64 + config.block_q as i64 - 1) / config.block_q as i64,
    );

    // ---- Launch via the production FFI --------------------------------
    //
    // Raw device pointers are accepted by `csha_tensor_data_ptr` (it
    // queries `cuPointerGetAttribute(MEMORY_TYPE)` to detect them and
    // passes them straight through — same path the existing
    // `csha_cuda_launch_classic` / `pca_tier_a_forward_correctness`
    // tests rely on).
    let rc = nsl_flash_attention_csha(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        batch as i64, heads as i64, seq_len as i64, head_dim as i64,
        0, 0, 0, 0,                         // paged
        0, 0,                               // cos, sin
        0, 0,                               // seq_ids, seq_lens
        smem,
        ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        config.block_q as i64, config.block_kv as i64,
        if config.causal { 1 } else { 0 },
        // CSHA extras (all null/zero — classic Q-from-HBM path).
        0, 0, 0, 0, 0, 0,
        1e-5f32.to_bits() as i64,
        0, 0,
        // PCA Tier A: segment_ids_ptr (0 — per-doc CTA gets isolation
        // from the launch grid, not seg-mask).
        0,
        // Tier B planner: disabled sentinel pair.
        0, 0,
        // PCA §4.3: doc_starts_ptr — REQUIRED for the per-doc kernel
        // (read by `per_doc_prelude::emit` to compute %q_start / %k_max).
        doc_starts_dev,
        // PCA per-doc CTA Strategy 3 v1: num_docs_or_zero — drives
        // `grid_x` override.  num_docs=4 launches 4 CTAs (one per doc)
        // instead of the legacy 2 CTAs (ceil(128/64)).
        num_docs,
    );

    if rc != 0 {
        nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
        nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
        nsl_test_cuda_free(doc_starts_dev);
        panic!("nsl_flash_attention_csha returned rc={} for per-doc dispatch", rc);
    }

    // Read back f16 output.
    let mut out_f16 = vec![0u16; total];
    nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, f16_bytes);
    let out_f32: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();

    nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
    nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
    nsl_test_cuda_free(doc_starts_dev);

    // Compare only the active region [0..active_seq_len * head_dim].
    let active_elems = batch * heads * active_seq_len * head_dim;
    let gpu_active = &out_f32[..active_elems];
    let cpu_active = &cpu_ref[..active_elems];

    assert!(
        gpu_active.iter().all(|x| x.is_finite()),
        "GPU output (active region) contains non-finite values — likely a \
         dispatch regression: doc-3 or doc-4 of the output is uninitialised",
    );
    assert!(
        cpu_active.iter().all(|x| x.is_finite()),
        "CPU ref (active region) contains non-finite values",
    );

    let (max_abs, max_idx) = max_abs_diff(gpu_active, cpu_active);
    eprintln!(
        "[ffi-dispatch] per-doc CTA via FFI vs CPU ref: max_abs={:.6} at idx={} gpu={} cpu={}",
        max_abs, max_idx, gpu_active[max_idx], cpu_active[max_idx],
    );

    assert!(
        max_abs <= 5e-3,
        "per-doc CTA FFI dispatch FAILED: max_abs={:.6} > 5e-3 tolerance",
        max_abs,
    );
    eprintln!("per-doc CTA FFI dispatch PASSED: max_abs={:.6} <= 5e-3", max_abs);
}

// ===========================================================================
// GPU test: per-doc dispatch with zero num_docs_or_zero MUST fail.
//
// Negative-control test for the runtime's `per_doc_cta && num_docs_or_zero
// <= 0` guard.  If this regresses (e.g. guard removed), grid_x silently
// becomes ceil(seq/bq)=2 instead of 4 and the user is debugging zero
// outputs in docs 2/3.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn per_doc_cta_ffi_dispatch_rejects_zero_num_docs() {
    if !cuda_available() { return; }

    let seq_len = 128usize;
    let head_dim = 32usize;
    let total = seq_len * head_dim;
    let f32_bytes = (total * std::mem::size_of::<f32>()) as i64;
    let f16_bytes = (total * std::mem::size_of::<u16>()) as i64;
    let lse_bytes = (seq_len * std::mem::size_of::<f32>()) as i64;
    let doc_starts_host = vec![0i32, 32, 60, 84, 100, 128];
    let ds_bytes = (doc_starts_host.len() * std::mem::size_of::<i32>()) as i64;

    let q_dev   = nsl_test_cuda_alloc(f32_bytes);
    let k_dev   = nsl_test_cuda_alloc(f32_bytes);
    let v_dev   = nsl_test_cuda_alloc(f32_bytes);
    let out_dev = nsl_test_cuda_alloc(f16_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);
    let doc_starts_dev = nsl_test_cuda_alloc(ds_bytes);
    assert!(
        q_dev != 0 && k_dev != 0 && v_dev != 0 && out_dev != 0
            && lse_dev != 0 && doc_starts_dev != 0,
        "device alloc returned null",
    );
    nsl_test_cuda_h2d(doc_starts_dev, doc_starts_host.as_ptr() as i64, ds_bytes);

    let config = FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 32,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, segment_masked: false, csha: None,
    };
    let packing = DatasetPackingConfig {
        enabled: true,
        max_sequence_length: seq_len as u32,
        mean_doc_length: Some(25),
        doc_length_stddev: Some(0),
        separator_token_id: Some(2),
    };
    let det = PcaDetection {
        strategy: PcaStrategy::PerDocumentCta,
        expected_doc_fraction: 0.195,
        rationale: "ffi-dispatch reject test".to_string(),
        segment_id_bytes_per_batch: 256,
        eliminated_mask_bytes_per_batch: 32768,
    };
    let plan = admit(
        &det, &config, &packing,
        &PerDocAdmitConfig { enable_per_doc_cta: true, ..Default::default() },
    ).expect("plan must admit");

    let mut ptx = synthesize_per_doc_cta_forward(&plan, &config);
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    ptx.push(0);
    let kernel_name = CString::new(per_doc_cta_kernel_name(&config)).unwrap();
    let smem = nsl_codegen::flash_attention_v2::smem_layout::total_bytes(&config) as i64;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Pass num_docs_or_zero=0 with a per-doc kernel name -> MUST return -1.
    let rc = nsl_flash_attention_csha(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        1, 1, seq_len as i64, head_dim as i64,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        smem,
        ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        config.block_q as i64, config.block_kv as i64,
        1,
        0, 0, 0, 0, 0, 0,
        1e-5f32.to_bits() as i64,
        0, 0,
        0,
        0, 0,
        doc_starts_dev,
        0, // num_docs_or_zero = 0 — mismatch with kernel name
    );

    nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
    nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
    nsl_test_cuda_free(doc_starts_dev);

    assert_eq!(
        rc, -1,
        "FFI must reject per-doc kernel name with num_docs_or_zero=0 \
         (got rc={}); otherwise grid_x silently misfires to ceil(seq/bq)",
        rc,
    );
}
