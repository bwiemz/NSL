//! PCA Tier A forward numerical correctness fixtures (spec §7.3).
//!
//! Three test cases:
//!   1. Single-segment — all positions in one segment, mask is a no-op.
//!      Result must match segment_masked=false baseline within 1e-5 (the
//!      segment predicate is always true, so no -inf clobbers occur and the
//!      numeric paths are identical save for f16 rounding in the identical
//!      positions).
//!   2. Two-equal-segments — seq_len=128 with segments [0:64]=0, [64:128]=1.
//!      Compared against unpacked-padded reference (two separate seq_len=64
//!      attention runs, outputs concatenated).
//!   3. Unequal-segments — three segments of lengths 38, 64, 26 packed into
//!      seq_len=128. Compared against unpacked-padded reference.
//!
//! Tolerance: 5e-3 (f16 mantissa + softmax renorm jitter — matches CSHA
//! Tier A at head_dim=32 per spec §7.4).
//!
//! Gated with #[cfg(feature = "cuda")] and #[ignore] per the
//! csha_cuda_launch_classic.rs pattern. Run with:
//!
//!   cargo test -p nsl-codegen --features cuda --test pca_tier_a_forward_correctness \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_selector::{
    flash_attention_kernel_name_selected, shared_mem_bytes_selected,
    synthesize_flash_attention_ptx_selected,
};
use std::ffi::CString;

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_d2h, nsl_test_cuda_jit_log,
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
// Deterministic PRNG — same LCG as csha_cuda_launch_classic.rs
// ---------------------------------------------------------------------------

fn fill_seeded(dst: &mut [f32], mut seed: u64) {
    for x in dst.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (seed >> 33) as u32;
        *x = ((u as f32) / (u32::MAX as f32)) - 0.5;
    }
}

// ---------------------------------------------------------------------------
// CPU naive attention reference (causal + segment-masked)
//
// Layout: Q/K/V/out are [B, H, S, D] row-major.
//
// The `seg_ids` slice has length S (one entry per position, indexed into
// the packed sequence).  When non-empty, positions (qi, kj) are masked
// where `seg_ids[qi] != seg_ids[kj]`.
// ---------------------------------------------------------------------------

fn naive_attention_segmented(
    q: &[f32], k: &[f32], v: &[f32],
    out: &mut [f32],
    b: usize, h: usize, s: usize, d: usize,
    scale: f32,
    causal: bool,
    seg_ids: &[u16],  // length s; empty = no segment masking
) {
    fn row<'a>(t: &'a [f32], bi: usize, hi: usize, si: usize, h: usize, s: usize, d: usize) -> &'a [f32] {
        let base = ((bi * h + hi) * s + si) * d;
        &t[base..base + d]
    }

    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..s {
                // Positions that are either causally visible and in the same
                // segment (or no segment masking) are attended to.
                let k_last = if causal { qi } else { s - 1 };
                let qv = row(q, bi, hi, qi, h, s, d);
                let q_seg = if seg_ids.is_empty() { 0u16 } else { seg_ids[qi] };

                let mut scores = vec![f32::NEG_INFINITY; s];
                let mut max_s = f32::NEG_INFINITY;

                for kj in 0..=k_last {
                    // Apply segment mask if provided.
                    let k_seg = if seg_ids.is_empty() { 0u16 } else { seg_ids[kj] };
                    if !seg_ids.is_empty() && q_seg != k_seg {
                        continue; // masked — score stays -inf
                    }
                    let kv = row(k, bi, hi, kj, h, s, d);
                    let mut s_val = 0f32;
                    for dd in 0..d {
                        s_val += qv[dd] * kv[dd];
                    }
                    s_val *= scale;
                    scores[kj] = s_val;
                    if s_val > max_s { max_s = s_val; }
                }

                // If all scores are -inf (row fully masked), output zeros.
                if max_s == f32::NEG_INFINITY {
                    let out_base = ((bi * h + hi) * s + qi) * d;
                    for dd in 0..d {
                        out[out_base + dd] = 0.0;
                    }
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

// ---------------------------------------------------------------------------
// CUDA availability guard (mirrors csha_cuda_launch_classic.rs)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Build a minimal PCA Tier A config for head_dim=32, causal, no RoPE.
// ---------------------------------------------------------------------------

fn fixture_config(segment_masked: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q:       64,
        block_kv:      64,
        head_dim:      32,
        causal:        true,
        paged:         false,
        rope_q:        false,
        rope_style:    RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask:     false,
        num_sink_tokens: 0,
        gpu_sm:        75,
        segment_masked,
        csha:          None,
        checkpoint: None,
    }
}

// ---------------------------------------------------------------------------
// f32 → f16 bit conversion helper (mirrors pca_tier_a_backward_correctness.rs).
// Needed for converting cos/sin tables to f16 before device upload.
// ---------------------------------------------------------------------------

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
// Parameterized config builder — lets callers force the extern-SMEM regime
// (total_bytes > 16 KB, e.g. head_dim=128, block_q=block_kv=128) and
// exercise RoPE by passing rope_q=true.
// ---------------------------------------------------------------------------

#[allow(dead_code)] // used by Task 3/5/7 GPU tests
fn fixture_config_sized(
    head_dim: i64, block_q: i64, block_kv: i64, rope_q: bool, segment_masked: bool,
) -> FlashAttentionConfig {
    let mut c = fixture_config(segment_masked); // reuse base fields (causal, gpu_sm, csha, etc.)
    c.head_dim = head_dim;
    c.block_q  = block_q;
    c.block_kv = block_kv;
    c.rope_q   = rope_q;
    c
}

// ---------------------------------------------------------------------------
// RoPE cos/sin table generator.
//
// Generates standard RoPE tables in the layout the CSHA kernel's
// `emit_rope_pair_sweep` reads:
//
//   cos/sin: [seq_len, half_dim] where half_dim = head_dim / 2
//   element dtype: f16 (the kernel does `ld.global.b16` + `cvt.f32.f16`)
//   address formula: (pos * half_dim + i) * 2  (bytes, 2 = sizeof(f16))
//
// Theta formula (matches cpu_reference_rope_single_doc / pca_rope_numerical):
//   theta = pos / 10000^(2*i / head_dim)
// which equals pos * 10000^(-(2*i / head_dim)).
//
// The returned Vec<u16> contains f16 bits, ready for nsl_test_cuda_h2d
// (pass `cos_f16.as_ptr() as i64` and byte size = seq_len * half_dim * 2).
// ---------------------------------------------------------------------------

#[allow(dead_code)] // used by Task 3/5/7 GPU tests
fn rope_cos_sin_tables_f16(seq_len: usize, head_dim: usize) -> (Vec<u16>, Vec<u16>) {
    let half = head_dim / 2;
    let mut cos_f16 = vec![0u16; seq_len * half];
    let mut sin_f16 = vec![0u16; seq_len * half];
    for pos in 0..seq_len {
        for i in 0..half {
            let theta = (pos as f64) * 10_000f64.powf(-(2.0 * i as f64) / head_dim as f64);
            cos_f16[pos * half + i] = f32_to_f16_bits(theta.cos() as f32);
            sin_f16[pos * half + i] = f32_to_f16_bits(theta.sin() as f32);
        }
    }
    (cos_f16, sin_f16)
}

// ---------------------------------------------------------------------------
// Core launch helper: upload Q/K/V + segment_ids, launch the kernel,
// read back f16 output, free device memory; return host f32 output.
//
// Returns None if the kernel fails to launch (with an eprintln diagnotic).
// ---------------------------------------------------------------------------

fn launch_pca(
    q_host: &[f32], k_host: &[f32], v_host: &[f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
    causal: bool,
    segment_masked: bool,
    seg_ids_host: &[u16],   // length seq_len; empty slice => pass ptr=0
) -> Option<Vec<f32>> {
    let total = batch * heads * seq_len * head_dim;
    let f32_bytes = (total * std::mem::size_of::<f32>()) as i64;
    let f16_bytes = (total * std::mem::size_of::<u16>()) as i64;
    let lse_bytes = (batch * heads * seq_len * std::mem::size_of::<f32>()) as i64;

    let config = fixture_config(segment_masked);
    let scale  = 1.0f32 / (head_dim as f32).sqrt();

    let q_dev   = nsl_test_cuda_alloc(f32_bytes);
    let k_dev   = nsl_test_cuda_alloc(f32_bytes);
    let v_dev   = nsl_test_cuda_alloc(f32_bytes);
    let out_dev = nsl_test_cuda_alloc(f16_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);

    assert!(q_dev != 0 && k_dev != 0 && v_dev != 0 && out_dev != 0 && lse_dev != 0,
        "device alloc returned null");

    nsl_test_cuda_h2d(q_dev,   q_host.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(k_dev,   k_host.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(v_dev,   v_host.as_ptr() as i64, f32_bytes);

    // Upload segment_ids if needed.
    let seg_ids_dev: i64 = if !seg_ids_host.is_empty() && segment_masked {
        let seg_bytes = (seg_ids_host.len() * std::mem::size_of::<u16>()) as i64;
        let ptr = nsl_test_cuda_alloc(seg_bytes);
        assert!(ptr != 0, "segment_ids device alloc returned null");
        nsl_test_cuda_h2d(ptr, seg_ids_host.as_ptr() as i64, seg_bytes);
        ptr
    } else {
        0i64 // null — unpacked path
    };

    // PTX synthesis.
    let mut ptx = synthesize_flash_attention_ptx_selected(&config);
    // Defensive: strip trailing NULs, ensure trailing newline, then null-terminate.
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    let dump = std::env::temp_dir().join(format!(
        "pca_tier_a_seg{}_sq{}.ptx",
        if segment_masked { "masked" } else { "plain" },
        seq_len,
    ));
    std::fs::write(&dump, &ptx).ok();
    eprintln!("PTX dumped to: {}", dump.display());
    ptx.push(0); // null-terminate for cuModuleLoadData

    let kernel_name = CString::new(flash_attention_kernel_name_selected(&config)).unwrap();
    let smem = shared_mem_bytes_selected(&config) as i64;

    let rc = nsl_flash_attention_csha(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq_len as i64, head_dim as i64,
            0, 0, 0, 0,         // paging
            0, 0, 0, 0,         // rope + ragged
            smem,
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            config.block_q as i64, config.block_kv as i64,
            if causal { 1i64 } else { 0i64 },
            // CSHA extras — all null/zero
            0, 0, 0, 0, 0, 0,
            1e-5f32.to_bits() as i64,
            0i64, 0i64,
            // PCA Tier A: segment_ids ptr (trailing)
            seg_ids_dev,
            // Tier B extension — null (not used for Tier A tests)
            0i64, 0i64,
            // doc_starts ptr — null (rope_q=false, no doc-aware positions)
            0i64,
            // PCA per-doc CTA Strategy 3 v1: num_docs_or_zero — 0 (legacy topology)
            0i64,
    );

    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                let cstr = std::ffi::CStr::from_ptr(log_ptr as *const i8);
                cstr.to_string_lossy().into_owned()
            }
        } else {
            "<no log>".into()
        };
        eprintln!("PCA kernel launch failed rc={} seq={} seg_masked={}\nJIT log:\n{}",
            rc, seq_len, segment_masked, log);
        // Free before returning.
        nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
        nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
        if seg_ids_dev != 0 { nsl_test_cuda_free(seg_ids_dev); }
        return None;
    }

    // Read back f16 output and convert to f32.
    let mut out_f16 = vec![0u16; total];
    nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, f16_bytes);
    let out_f32: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // Cleanup.
    nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
    nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
    if seg_ids_dev != 0 { nsl_test_cuda_free(seg_ids_dev); }

    Some(out_f32)
}

// ---------------------------------------------------------------------------
// Max-abs-diff diagnostic helper
// ---------------------------------------------------------------------------

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
// Fixture 1: single-segment — mask is a no-op
//
// All segment_ids = 0, so the segment predicate is always true.
// With causal masking also active, the kernel must produce output identical
// to the no-segment-mask baseline (within f16 rounding; tolerance 1e-5 is
// tighter than the usual 5e-3 because neither kernel path differs structurally
// — we tighten to 1e-4 to leave headroom for non-deterministic FP order in
// online softmax while still confirming no -inf clobbers).
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_a_forward_single_segment_matches_unmasked_baseline() {
    if !cuda_available() { return; }

    let batch = 1usize; let heads = 1usize;
    let seq_len = 128usize; let head_dim = 32usize;
    let total = batch * heads * seq_len * head_dim;

    let mut q = vec![0f32; total];
    let mut k = vec![0f32; total];
    let mut v = vec![0f32; total];
    fill_seeded(&mut q, 0xA1B2_C3D4);
    fill_seeded(&mut k, 0x1234_5678);
    fill_seeded(&mut v, 0xDEAD_BEEF);

    // segment_ids: all zeros — single segment, mask is a no-op.
    let seg_ids: Vec<u16> = vec![0u16; seq_len];

    eprintln!("Fixture 1 — running baseline (segment_masked=false)");
    let baseline = launch_pca(
        &q, &k, &v, batch, heads, seq_len, head_dim, true,
        false,      // segment_masked=false
        &[],        // no seg_ids needed
    );
    eprintln!("Fixture 1 — running PCA (segment_masked=true, all-zero seg_ids)");
    let pca_out = launch_pca(
        &q, &k, &v, batch, heads, seq_len, head_dim, true,
        true,       // segment_masked=true
        &seg_ids,   // all zeros
    );

    let (baseline, pca_out) = match (baseline, pca_out) {
        (Some(b), Some(p)) => (b, p),
        _ => {
            eprintln!("Fixture 1 SKIPPED: kernel launch failed (non-GPU environment?)");
            return;
        }
    };

    let all_finite_b = baseline.iter().all(|x| x.is_finite());
    let all_finite_p = pca_out.iter().all(|x| x.is_finite());
    assert!(all_finite_b, "baseline output contains non-finite values");
    assert!(all_finite_p, "PCA output contains non-finite values");

    let (max_abs, max_idx) = max_abs_diff(&baseline, &pca_out);
    eprintln!(
        "Fixture 1 [single-segment]: max_abs={:.6} at idx={} baseline[idx]={} pca[idx]={}",
        max_abs, max_idx, baseline[max_idx], pca_out[max_idx],
    );
    eprintln!("  first 4 baseline: {:?}", &baseline[..4.min(baseline.len())]);
    eprintln!("  first 4 pca:      {:?}", &pca_out[..4.min(pca_out.len())]);

    // Single-segment mask is a no-op; tolerate f16-rounding epsilon only.
    // 1e-4 is tighter than the cross-segment 5e-3 budget.
    assert!(
        max_abs <= 1e-4,
        "Fixture 1 FAILED: single-segment mask must be a no-op, got max_abs={:.6} (threshold 1e-4)",
        max_abs,
    );
    eprintln!("Fixture 1 PASSED: max_abs={:.6} <= 1e-4", max_abs);
}

// ===========================================================================
// Fixture 2: two-equal-segments — adversarial
//
// seq_len=128 split as [0..64]=segment 0, [64..128]=segment 1.
//
// Reference: run two separate attention computations:
//   (a) Q[:64], K[:64], V[:64] → out_0   (seq_len=64, causal)
//   (b) Q[64:], K[64:], V[64:] → out_1   (seq_len=64, causal)
//   concat → out_ref[0..128]
//
// Packed kernel output must match out_ref within 5e-3.
//
// This test satisfies spec §5 property 1 (cross-document attention
// prohibition): the CPU reference (naive_attention_segmented) enforces
// the segment boundary by design, and the packed kernel output must match
// it — any cross-segment leakage would produce a numerical divergence
// that exceeds the 5e-3 tolerance.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_a_forward_two_equal_segments_matches_unpacked_reference() {
    if !cuda_available() { return; }

    let batch = 1usize; let heads = 1usize;
    let seq_len = 128usize; let head_dim = 32usize;
    let seg_a = 64usize; let seg_b = 64usize;
    assert_eq!(seg_a + seg_b, seq_len);

    let total_full  = batch * heads * seq_len * head_dim;
    let total_half  = batch * heads * seg_a * head_dim;

    let mut q_full = vec![0f32; total_full];
    let mut k_full = vec![0f32; total_full];
    let mut v_full = vec![0f32; total_full];
    fill_seeded(&mut q_full, 0xCAFE_BABE);
    fill_seeded(&mut k_full, 0x0011_2233);
    fill_seeded(&mut v_full, 0xFFEE_DDCC);

    // segment_ids: [0..64]=0, [64..128]=1
    let mut seg_ids = vec![0u16; seq_len];
    for x in seg_ids[seg_a..].iter_mut() { *x = 1; }

    // Run packed kernel.
    eprintln!("Fixture 2 — running packed PCA kernel (seq=128, 2 segments of 64)");
    let pca_out = launch_pca(
        &q_full, &k_full, &v_full, batch, heads, seq_len, head_dim, true,
        true, &seg_ids,
    );

    // Build unpacked reference: two independent attention runs, each seq_len=64.
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let (q_a, k_a, v_a, q_b, k_b, v_b) = split_packed_qkv(
        &q_full, &k_full, &v_full,
        batch, heads, seq_len, head_dim,
        &[seg_a, seg_b],
    );

    let mut ref_a = vec![0f32; total_half];
    let mut ref_b = vec![0f32; total_half];

    naive_attention_segmented(&q_a, &k_a, &v_a, &mut ref_a, batch, heads, seg_a, head_dim, scale, true, &[]);
    naive_attention_segmented(&q_b, &k_b, &v_b, &mut ref_b, batch, heads, seg_b, head_dim, scale, true, &[]);

    // Concatenate reference outputs along the seq dim.
    let ref_out = interleave_unpacked_outputs(
        &[&ref_a, &ref_b],
        batch, heads, &[seg_a, seg_b], head_dim,
    );

    let pca_out = match pca_out {
        Some(p) => p,
        None => {
            eprintln!("Fixture 2 SKIPPED: kernel launch failed");
            return;
        }
    };

    let all_finite_p = pca_out.iter().all(|x| x.is_finite());
    let all_finite_r = ref_out.iter().all(|x| x.is_finite());
    assert!(all_finite_p, "PCA output contains non-finite values");
    assert!(all_finite_r, "reference output contains non-finite values");

    let (max_abs, max_idx) = max_abs_diff(&pca_out, &ref_out);
    eprintln!(
        "Fixture 2 [two-equal-segments]: max_abs={:.6} at idx={} pca={} ref={}",
        max_abs, max_idx, pca_out[max_idx], ref_out[max_idx],
    );
    eprintln!("  first 4 pca: {:?}", &pca_out[..4.min(pca_out.len())]);
    eprintln!("  first 4 ref: {:?}", &ref_out[..4.min(ref_out.len())]);

    assert!(
        max_abs <= 5e-3,
        "Fixture 2 FAILED: packed output diverges from unpacked reference, max_abs={:.6} (threshold 5e-3)",
        max_abs,
    );
    eprintln!("Fixture 2 PASSED: max_abs={:.6} <= 5e-3", max_abs);
}

// ===========================================================================
// Fixture 3: unequal-segments — adversarial
//
// seq_len=128, three segments: [0..38]=0, [38..102]=1, [102..128]=2.
// Lengths: 38, 64, 26 (irregular, tests boundary logic).
//
// Reference: three independent attention runs on seq_len=38, 64, 26;
// outputs concatenated.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_a_forward_unequal_segments_matches_unpacked_reference() {
    if !cuda_available() { return; }

    let batch = 1usize; let heads = 1usize;
    let head_dim = 32usize;
    let lens = [38usize, 64usize, 26usize];
    let seq_len: usize = lens.iter().sum(); // 128
    assert_eq!(seq_len, 128);

    let total_full = batch * heads * seq_len * head_dim;

    let mut q_full = vec![0f32; total_full];
    let mut k_full = vec![0f32; total_full];
    let mut v_full = vec![0f32; total_full];
    fill_seeded(&mut q_full, 0xABCD_EF01);
    fill_seeded(&mut k_full, 0x5A5A_5A5A);
    fill_seeded(&mut v_full, 0x1357_9BDF);

    // Build segment_ids: [0..38]=0, [38..102]=1, [102..128]=2.
    let mut seg_ids = vec![0u16; seq_len];
    let mut off = 0usize;
    for (seg_id, &seg_len) in lens.iter().enumerate() {
        for x in seg_ids[off..off + seg_len].iter_mut() { *x = seg_id as u16; }
        off += seg_len;
    }

    // Run packed kernel.
    eprintln!("Fixture 3 — running packed PCA kernel (seq=128, segments 38+64+26)");
    let pca_out = launch_pca(
        &q_full, &k_full, &v_full, batch, heads, seq_len, head_dim, true,
        true, &seg_ids,
    );

    // Build unpacked reference using the general extract_segment helper.
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let segment_refs: Vec<Vec<f32>> = lens.iter().enumerate().map(|(seg_id, &seg_len)| {
        let (q_s, k_s, v_s) = extract_segment(
            &q_full, &k_full, &v_full,
            batch, heads, seq_len, head_dim,
            &seg_ids, seg_id as u16,
        );
        assert_eq!(q_s.len(), batch * heads * seg_len * head_dim);
        let mut out_s = vec![0f32; batch * heads * seg_len * head_dim];
        naive_attention_segmented(&q_s, &k_s, &v_s, &mut out_s, batch, heads, seg_len, head_dim, scale, true, &[]);
        out_s
    }).collect();

    let seg_slices: Vec<&[f32]> = segment_refs.iter().map(|v| v.as_slice()).collect();
    let ref_out = interleave_unpacked_outputs(&seg_slices, batch, heads, &lens, head_dim);

    let pca_out = match pca_out {
        Some(p) => p,
        None => {
            eprintln!("Fixture 3 SKIPPED: kernel launch failed");
            return;
        }
    };

    let all_finite_p = pca_out.iter().all(|x| x.is_finite());
    let all_finite_r = ref_out.iter().all(|x| x.is_finite());
    assert!(all_finite_p, "PCA output contains non-finite values");
    assert!(all_finite_r, "reference output contains non-finite values");

    let (max_abs, max_idx) = max_abs_diff(&pca_out, &ref_out);
    eprintln!(
        "Fixture 3 [unequal-segments 38+64+26]: max_abs={:.6} at idx={} pca={} ref={}",
        max_abs, max_idx, pca_out[max_idx], ref_out[max_idx],
    );
    eprintln!("  first 4 pca: {:?}", &pca_out[..4.min(pca_out.len())]);
    eprintln!("  first 4 ref: {:?}", &ref_out[..4.min(ref_out.len())]);

    assert!(
        max_abs <= 5e-3,
        "Fixture 3 FAILED: packed output diverges from unpacked reference, max_abs={:.6} (threshold 5e-3)",
        max_abs,
    );
    eprintln!("Fixture 3 PASSED: max_abs={:.6} <= 5e-3", max_abs);
}

// ===========================================================================
// Split helpers
// ===========================================================================

/// Split a packed [B, H, S, D] tensor into per-segment slices.
/// Returns (q_a, k_a, v_a, q_b, k_b, v_b) for exactly 2 segments.
fn split_packed_qkv(
    q: &[f32], k: &[f32], v: &[f32],
    batch: usize, heads: usize, _seq: usize, head_dim: usize,
    lens: &[usize],
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(lens.len(), 2, "split_packed_qkv only supports 2 segments");
    let seg_a = lens[0]; let seg_b = lens[1];
    let mut q_a = vec![0f32; batch * heads * seg_a * head_dim];
    let mut k_a = vec![0f32; batch * heads * seg_a * head_dim];
    let mut v_a = vec![0f32; batch * heads * seg_a * head_dim];
    let mut q_b = vec![0f32; batch * heads * seg_b * head_dim];
    let mut k_b = vec![0f32; batch * heads * seg_b * head_dim];
    let mut v_b = vec![0f32; batch * heads * seg_b * head_dim];
    let full_seq = seg_a + seg_b;

    for bi in 0..batch {
        for hi in 0..heads {
            // Segment A
            for si in 0..seg_a {
                let src_base = ((bi * heads + hi) * full_seq + si) * head_dim;
                let dst_base = ((bi * heads + hi) * seg_a  + si) * head_dim;
                q_a[dst_base..dst_base + head_dim].copy_from_slice(&q[src_base..src_base + head_dim]);
                k_a[dst_base..dst_base + head_dim].copy_from_slice(&k[src_base..src_base + head_dim]);
                v_a[dst_base..dst_base + head_dim].copy_from_slice(&v[src_base..src_base + head_dim]);
            }
            // Segment B
            for si in 0..seg_b {
                let src_base = ((bi * heads + hi) * full_seq + seg_a + si) * head_dim;
                let dst_base = ((bi * heads + hi) * seg_b   + si) * head_dim;
                q_b[dst_base..dst_base + head_dim].copy_from_slice(&q[src_base..src_base + head_dim]);
                k_b[dst_base..dst_base + head_dim].copy_from_slice(&k[src_base..src_base + head_dim]);
                v_b[dst_base..dst_base + head_dim].copy_from_slice(&v[src_base..src_base + head_dim]);
            }
        }
    }
    (q_a, k_a, v_a, q_b, k_b, v_b)
}

/// General-purpose segment extractor.
///
/// `seg_ids` is the packed sequence segment ID vector (length `seq_len`).
/// Extracts all positions with `seg_ids[pos] == target_id` in order.
fn extract_segment(
    q: &[f32], k: &[f32], v: &[f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
    seg_ids: &[u16], target_id: u16,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let positions: Vec<usize> = (0..seq_len).filter(|&i| seg_ids[i] == target_id).collect();
    let seg_len = positions.len();
    let total = batch * heads * seg_len * head_dim;
    let mut q_s = vec![0f32; total];
    let mut k_s = vec![0f32; total];
    let mut v_s = vec![0f32; total];

    for bi in 0..batch {
        for hi in 0..heads {
            for (dst_si, &src_si) in positions.iter().enumerate() {
                let src_base = ((bi * heads + hi) * seq_len + src_si) * head_dim;
                let dst_base = ((bi * heads + hi) * seg_len + dst_si) * head_dim;
                q_s[dst_base..dst_base + head_dim].copy_from_slice(&q[src_base..src_base + head_dim]);
                k_s[dst_base..dst_base + head_dim].copy_from_slice(&k[src_base..src_base + head_dim]);
                v_s[dst_base..dst_base + head_dim].copy_from_slice(&v[src_base..src_base + head_dim]);
            }
        }
    }
    (q_s, k_s, v_s)
}

// ===========================================================================
// A-4 Test 1: null seg_ids_ptr guard — masked+null must equal unmasked forward
//
// Validates spec §3 null-guard: when seg_ids_ptr==0 the kernel writes
// an all-zero seg_smem sentinel (no segment masking applied), so the
// output must be numerically identical to the segment_masked=false kernel.
//
// The launch_pca helper passes 0 (NULL) for seg_ids_ptr when
// seg_ids_host is empty AND segment_masked is true (see harness, line
// 235-243). This exercises the `@%p_seg_null bra NULL_LD` skip guard
// added in A-3.
//
// Tolerance: 1e-5 — same kernel math, only the masking path is bypassed;
// f16 rounding is identical so outputs should be bit-exact or within
// a single ULP.
//
// CRITICAL (dQ-vacuous-gate lesson): if cuda_available() is true and the
// masked-null launch returns None, we PANIC (guard failure), not skip.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn masked_null_seg_ptr_equals_unmasked_forward() {
    if !cuda_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }

    let batch = 1usize; let heads = 1usize;
    let seq_len = 128usize; let head_dim = 32usize;
    let total = batch * heads * seq_len * head_dim;

    let mut q = vec![0f32; total];
    let mut k = vec![0f32; total];
    let mut v = vec![0f32; total];
    fill_seeded(&mut q, 0xA4_B4_C4_D4);
    fill_seeded(&mut k, 0x11_22_33_44);
    fill_seeded(&mut v, 0x55_66_77_88);

    eprintln!("A-4 Test 1 — baseline (segment_masked=false)");
    let baseline = launch_pca(
        &q, &k, &v, batch, heads, seq_len, head_dim, true,
        false, // segment_masked=false
        &[],   // no seg_ids
    );

    eprintln!("A-4 Test 1 — masked+null (segment_masked=true, empty seg_ids → ptr=0)");
    let masked_null = launch_pca(
        &q, &k, &v, batch, heads, seq_len, head_dim, true,
        true,  // segment_masked=true
        &[],   // empty → seg_ids_ptr=0 (NULL)
    );

    // CRITICAL: if the GPU is present and the masked-null launch returned None,
    // the null-guard failed (kernel crashed on NULL dereference).
    let masked_null = masked_null.unwrap_or_else(|| {
        panic!(
            "masked kernel crashed on null seg_ids_ptr — A-3 null-guard FAILED \
             (segment_masked=true, seg_ids_ptr=0 should be safe)"
        )
    });
    let baseline = baseline.expect("baseline (segment_masked=false) launch failed unexpectedly");

    let all_finite_b = baseline.iter().all(|x| x.is_finite());
    let all_finite_m = masked_null.iter().all(|x| x.is_finite());
    assert!(all_finite_b, "A-4 Test 1: baseline output contains non-finite values");
    assert!(all_finite_m, "A-4 Test 1: masked+null output contains non-finite values");

    let (max_abs, max_idx) = max_abs_diff(&masked_null, &baseline);
    eprintln!(
        "A-4 Test 1 [masked+null vs unmasked]: max_abs={:.2e} at idx={} \
         masked_null[idx]={} baseline[idx]={}",
        max_abs, max_idx, masked_null[max_idx], baseline[max_idx],
    );
    eprintln!("  first 4 baseline:    {:?}", &baseline[..4.min(baseline.len())]);
    eprintln!("  first 4 masked_null: {:?}", &masked_null[..4.min(masked_null.len())]);

    // Tight tolerance: same kernel, same math — only masking path differs.
    // With null seg_ids the guard forces all-zero seg_smem, which means the
    // segment predicate is always true (same as unmasked). f16 rounding is
    // identical, so we expect bit-exact or within 1e-5.
    let tol = 1e-5f32;
    assert!(
        max_abs <= tol,
        "A-4 Test 1 FAILED: masked+null output diverges from unmasked, \
         max_abs={:.2e} (threshold {:.0e}) — null-guard may be writing \
         non-zero sentinel or the guard branch is not taken",
        max_abs, tol,
    );
    eprintln!("A-4 Test 1 PASSED: masked+null == unmasked, max_abs={:.2e} <= {:.0e}", max_abs, tol);
}

// ===========================================================================
// Task 3 (reproduce-first): extern-regime forward — the Blackwell mixed-layout bug
//
// A large config forces total_bytes into the extern-shmem regime via the
// segment_masked +32 KB seg overhead. With the (pre-Task-4) static `.shared
// seg_smem`, this is the Blackwell static+extern crash. The test ASSERTS
// masked+null == unmasked (the fix's correctness); on pre-fix code the masked
// launch faults (the documented RED). Task 4's embed turns it GREEN.
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU; run ALONE (an illegal-address fault poisons the context)"]
fn masked_extern_config_equals_unmasked_forward() {
    if !cuda_available() { eprintln!("skipped: no CUDA device"); return; }

    let head_dim = 128usize; let block = 64i64;
    let batch = 1usize; let heads = 1usize; let seq_len = 128usize;

    // Sanity: confirm this config is in the extern regime AND fits the dynamic cap,
    // so the test exercises the right regime (not the static one the A-4 tests used).
    let cfg = fixture_config_sized(head_dim as i64, block, block, /*rope_q=*/false, /*segment_masked=*/true);
    let base = nsl_codegen::flash_attention_v2::smem_layout::total_bytes(&cfg);
    let layout = nsl_codegen::flash_attention_v2::smem_layout::pca_smem_layout(base, true, false);
    assert!(base <= 48 * 1024, "base total_bytes should fit static alone (got {base}) — the bug needs seg to force extern");
    assert!(layout.total > 48 * 1024, "config must force extern (layout.total={}); else not the buggy regime", layout.total);
    assert!(layout.total <= 99 * 1024, "config must fit the 99 KB dynamic cap (layout.total={})", layout.total);

    let total = batch * heads * seq_len * head_dim;
    let mut q = vec![0f32; total]; let mut k = vec![0f32; total]; let mut v = vec![0f32; total];
    fill_seeded(&mut q, 0xE0_E1_E2_E3);
    fill_seeded(&mut k, 0xD0_D1_D2_D3);
    fill_seeded(&mut v, 0xC0_C1_C2_C3);

    let baseline = launch_pca_ex(
        &q, &k, &v, batch, heads, seq_len, head_dim, block, block,
        /*rope_q=*/false, /*segment_masked=*/false, &[], /*cos_sin=*/None, /*doc_starts=*/None,
    );
    let masked_null = launch_pca_ex(
        &q, &k, &v, batch, heads, seq_len, head_dim, block, block,
        /*rope_q=*/false, /*segment_masked=*/true, &[], None, None, // empty seg_ids → seg_ids_ptr=0
    );

    let masked_null = masked_null.unwrap_or_else(|| panic!(
        "masked extern-config kernel crashed (seg_smem static + extern shmem[], sm_120) — \
         this is the documented pre-fix RED; Task 4's embed fix must make it Some"
    ));
    let baseline = baseline.expect("unmasked extern baseline launch failed unexpectedly");
    let (max_abs, _) = max_abs_diff(&masked_null, &baseline);
    assert!(max_abs < 1e-5, "masked+null != unmasked (extern): max_abs={max_abs:.2e}");
}

/// Concatenate per-segment outputs back into a packed [B, H, S, D] tensor.
///
/// `seg_outs[i]` is the output for segment i, shape [B, H, lens[i], D].
/// `lens[i]` is the length of segment i.
/// The segments are assumed to be contiguous in the packed sequence in
/// index order (seg 0 = positions 0..lens[0], seg 1 = positions lens[0]..lens[0]+lens[1], etc.).
fn interleave_unpacked_outputs(
    seg_outs: &[&[f32]],
    batch: usize, heads: usize,
    lens: &[usize],
    head_dim: usize,
) -> Vec<f32> {
    assert_eq!(seg_outs.len(), lens.len());
    let total_seq: usize = lens.iter().sum();
    let mut out = vec![0f32; batch * heads * total_seq * head_dim];

    let mut seq_offset = 0usize;
    for (seg_idx, (&seg_out, &seg_len)) in seg_outs.iter().zip(lens.iter()).enumerate() {
        let _ = seg_idx;
        for bi in 0..batch {
            for hi in 0..heads {
                for si in 0..seg_len {
                    let src_base = ((bi * heads + hi) * seg_len  + si) * head_dim;
                    let dst_base = ((bi * heads + hi) * total_seq + seq_offset + si) * head_dim;
                    out[dst_base..dst_base + head_dim]
                        .copy_from_slice(&seg_out[src_base..src_base + head_dim]);
                }
            }
        }
        seq_offset += seg_len;
    }
    out
}

// ---------------------------------------------------------------------------
// Extended forward launch helper — supports caller-chosen dims, rope_q=true,
// and optional cos/sin / doc_starts uploads.
//
// - cos_sin = Some((cos_f16, sin_f16)): upload both, pass real cos_ptr/sin_ptr.
//   Slices must be [seq_len * (head_dim/2)] f16 bits each.
//   Use rope_cos_sin_tables_f16() to generate them.
// - cos_sin = None: pass 0i64 for cos_ptr and sin_ptr.
// - doc_starts = Some(ds): upload [max_docs_per_row+1] i32 entries and
//   pass real doc_starts_ptr.
// - doc_starts = None: pass 0i64.
// - seg_ids empty AND segment_masked=false: seg_ids_ptr = 0.
//
// Dynamic SMEM: nsl_flash_attention_csha passes shared_mem_bytes to
// kernel_launch, which calls cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)
// for any non-zero value — so large/extern-SMEM configs are handled
// automatically via shared_mem_bytes_selected().
// ---------------------------------------------------------------------------

#[allow(dead_code)]              // used by Task 3/5/7 GPU tests
#[allow(clippy::too_many_arguments)]
fn launch_pca_ex(
    q: &[f32], k: &[f32], v: &[f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
    block_q: i64, block_kv: i64,
    rope_q: bool, segment_masked: bool, seg_ids: &[u16],
    cos_sin: Option<(&[u16], &[u16])>,
    doc_starts: Option<&[i32]>,
) -> Option<Vec<f32>> {
    let total    = batch * heads * seq_len * head_dim;
    let f32_bytes = (total * std::mem::size_of::<f32>()) as i64;
    let f16_bytes = (total * std::mem::size_of::<u16>()) as i64;
    let lse_bytes = (batch * heads * seq_len * std::mem::size_of::<f32>()) as i64;

    let config = fixture_config_sized(head_dim as i64, block_q, block_kv, rope_q, segment_masked);
    let scale  = 1.0f32 / (head_dim as f32).sqrt();

    let q_dev   = nsl_test_cuda_alloc(f32_bytes);
    let k_dev   = nsl_test_cuda_alloc(f32_bytes);
    let v_dev   = nsl_test_cuda_alloc(f32_bytes);
    let out_dev = nsl_test_cuda_alloc(f16_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);

    assert!(q_dev != 0 && k_dev != 0 && v_dev != 0 && out_dev != 0 && lse_dev != 0,
        "device alloc returned null");

    nsl_test_cuda_h2d(q_dev, q.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(k_dev, k.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(v_dev, v.as_ptr() as i64, f32_bytes);

    // Upload segment_ids if needed.
    let seg_ids_dev: i64 = if !seg_ids.is_empty() && segment_masked {
        let seg_bytes = std::mem::size_of_val(seg_ids) as i64;
        let ptr = nsl_test_cuda_alloc(seg_bytes);
        assert!(ptr != 0, "segment_ids device alloc returned null");
        nsl_test_cuda_h2d(ptr, seg_ids.as_ptr() as i64, seg_bytes);
        ptr
    } else {
        0i64
    };

    // Upload cos/sin tables (f16) if rope_q is active.
    let (cos_dev, sin_dev): (i64, i64) = if let Some((cos_f16, sin_f16)) = cos_sin {
        let cs_bytes = std::mem::size_of_val(cos_f16) as i64;
        let c_ptr = nsl_test_cuda_alloc(cs_bytes);
        let s_ptr = nsl_test_cuda_alloc(cs_bytes);
        assert!(c_ptr != 0 && s_ptr != 0, "cos/sin device alloc returned null");
        nsl_test_cuda_h2d(c_ptr, cos_f16.as_ptr() as i64, cs_bytes);
        nsl_test_cuda_h2d(s_ptr, sin_f16.as_ptr() as i64, std::mem::size_of_val(sin_f16) as i64);
        (c_ptr, s_ptr)
    } else {
        (0i64, 0i64)
    };

    // Upload doc_starts if provided.
    let doc_starts_dev: i64 = if let Some(ds) = doc_starts {
        let ds_bytes = std::mem::size_of_val(ds) as i64;
        let ptr = nsl_test_cuda_alloc(ds_bytes);
        assert!(ptr != 0, "doc_starts device alloc returned null");
        nsl_test_cuda_h2d(ptr, ds.as_ptr() as i64, ds_bytes);
        ptr
    } else {
        0i64
    };

    // PTX synthesis.
    let mut ptx = synthesize_flash_attention_ptx_selected(&config);
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    let dump = std::env::temp_dir().join(format!(
        "pca_ex_seg{}_hd{}_bq{}_rope{}.ptx",
        if segment_masked { "masked" } else { "plain" },
        head_dim, block_q, if rope_q { "true" } else { "false" },
    ));
    std::fs::write(&dump, &ptx).ok();
    eprintln!("PTX dumped to: {}", dump.display());
    ptx.push(0);

    let kernel_name = CString::new(flash_attention_kernel_name_selected(&config)).unwrap();
    let smem = shared_mem_bytes_selected(&config) as i64;

    let rc = nsl_flash_attention_csha(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        batch as i64, heads as i64, seq_len as i64, head_dim as i64,
        0, 0, 0, 0,         // paging
        cos_dev, sin_dev, 0, 0,  // cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr
        smem,
        ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        config.block_q, config.block_kv,
        1i64,               // causal=true (PCA always causal)
        // CSHA extras — all null/zero (no CSHA for Tier A PCA)
        0, 0, 0, 0, 0, 0,
        1e-5f32.to_bits() as i64,
        0i64, 0i64,
        // PCA Tier A: segment_ids ptr
        seg_ids_dev,
        // Tier B extension — null
        0i64, 0i64,
        // doc_starts ptr
        doc_starts_dev,
        // PCA per-doc CTA Strategy 3 v1: num_docs_or_zero — 0 (Tier A is not per-doc)
        0i64,
    );

    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                let cstr = std::ffi::CStr::from_ptr(log_ptr as *const i8);
                cstr.to_string_lossy().into_owned()
            }
        } else {
            "<no log>".into()
        };
        eprintln!("launch_pca_ex failed rc={} hd={} bq={} bkv={} rope={}\nJIT log:\n{}",
            rc, head_dim, block_q, block_kv, rope_q, log);
        nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
        nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
        if seg_ids_dev != 0 { nsl_test_cuda_free(seg_ids_dev); }
        if cos_dev != 0 { nsl_test_cuda_free(cos_dev); }
        if sin_dev != 0 { nsl_test_cuda_free(sin_dev); }
        if doc_starts_dev != 0 { nsl_test_cuda_free(doc_starts_dev); }
        return None;
    }

    // Read back f16 output and convert to f32.
    let mut out_f16 = vec![0u16; total];
    nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, f16_bytes);
    let out_f32: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // Cleanup.
    nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
    nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
    if seg_ids_dev != 0 { nsl_test_cuda_free(seg_ids_dev); }
    if cos_dev != 0 { nsl_test_cuda_free(cos_dev); }
    if sin_dev != 0 { nsl_test_cuda_free(sin_dev); }
    if doc_starts_dev != 0 { nsl_test_cuda_free(doc_starts_dev); }

    Some(out_f32)
}

// ===========================================================================
// rope_q=true, segment_masked=false: pure standard RoPE forward (no PCA SMEM).
//
// Validates the sin-address-aliasing fix (#2) — before the fix, rope_q=true
// forward faulted with CUDA_ERROR_ILLEGAL_ADDRESS (the sin address register
// was overwritten by the cos load before use).  After the fix the kernel
// launches cleanly.
//
// Uses segment_masked=false so NO PCA SMEM state (seg_smem / doc_starts) is
// involved — pure standard RoPE, zero cross-test GPU-state leakage surface.
//
// CPU reference: standard adjacent-pair RoPE applied to Q (position = row
// index, step_by(2) pairs), K unrotated (rope_q rotates Q only), then
// causal unmasked attention.  Tolerance 4e-2 (f16 + warp-shuffle approximation
// budget matching the Tier A forward suite).
//
// If the GPU output is all-finite and the launch succeeds, the primary goal
// (no crash) is met regardless of the CPU comparison result.
// ===========================================================================

/// CPU adjacent-pair RoPE for a single [seq_len, head_dim] Q matrix.
/// Mirrors the theta formula in pca_rope_numerical::cpu_reference_rope_single_doc.
fn cpu_rope_adjacent_pairs(q: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; q.len()];
    for pos in 0..seq_len {
        for d in (0..head_dim).step_by(2) {
            let theta = pos as f32 / 10_000f32.powf(d as f32 / head_dim as f32);
            let (c, s) = (theta.cos(), theta.sin());
            let x0 = q[pos * head_dim + d];
            let x1 = q[pos * head_dim + d + 1];
            out[pos * head_dim + d]     = x0 * c - x1 * s;
            out[pos * head_dim + d + 1] = x0 * s + x1 * c;
        }
    }
    out
}

/// CPU reference: apply standard RoPE to Q (adjacent-pair convention),
/// leave K unchanged (rope_q rotates Q only in the CFTP convention),
/// then run causal unmasked attention.
fn cpu_reference_rope_then_attention(
    q: &[f32], k: &[f32], v: &[f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
) -> Vec<f32> {
    // Rotate Q per-row; K/V pass through unchanged.
    let mut q_rot = q.to_vec();
    for bi in 0..batch {
        for hi in 0..heads {
            for pos in 0..seq_len {
                let base = ((bi * heads + hi) * seq_len + pos) * head_dim;
                for d in (0..head_dim).step_by(2) {
                    let theta = pos as f32 / 10_000f32.powf(d as f32 / head_dim as f32);
                    let (c, s) = (theta.cos(), theta.sin());
                    let x0 = q[base + d];
                    let x1 = q[base + d + 1];
                    q_rot[base + d]     = x0 * c - x1 * s;
                    q_rot[base + d + 1] = x0 * s + x1 * c;
                }
            }
        }
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0f32; batch * heads * seq_len * head_dim];
    naive_attention_segmented(&q_rot, k, v, &mut out, batch, heads, seq_len, head_dim, scale, true, &[]);
    out
}

#[test]
#[ignore = "requires CUDA GPU"]
fn rope_q_forward_matches_cpu_standard_rope() {
    if !cuda_available() { eprintln!("skipped: no CUDA device"); return; }
    let head_dim = 64usize; let block = 32i64;
    let batch = 1usize; let heads = 1usize; let seq_len = 64usize;
    let total = batch * heads * seq_len * head_dim;
    let mut q = vec![0f32; total]; let mut k = vec![0f32; total]; let mut v = vec![0f32; total];
    fill_seeded(&mut q, 0x9001); fill_seeded(&mut k, 0x9002); fill_seeded(&mut v, 0x9003);
    let (cos, sin) = rope_cos_sin_tables_f16(seq_len, head_dim);

    let gpu = launch_pca_ex(
        &q, &k, &v, batch, heads, seq_len, head_dim, block, block,
        /*rope_q=*/true, /*segment_masked=*/false, &[], Some((&cos, &sin)), None,
    ).expect("rope_q=true forward launch FAILED (must not crash -- validates the sin-aliasing fix)");

    let all_finite = gpu.iter().all(|x| x.is_finite());
    assert!(all_finite, "rope_q=true forward: output contains non-finite values after sin-aliasing fix");
    eprintln!("rope_q=true forward: kernel launched + output all-finite (sin-aliasing fix confirmed)");

    // Secondary: compare against CPU standard-RoPE reference.
    // rope_cos_sin_tables_f16 uses [seq_len, half_dim] f16 layout; the kernel's
    // HalfSplit inline rotation reads cos/sin from that table.  The CPU reference
    // uses the canonical adjacent-pair RoPE (same theta formula, f32 precision).
    // If the GPU HalfSplit convention differs from adjacent-pair within 4e-2,
    // the test passes.  If not, it is a real finding about the RoPE convention.
    let cpu = cpu_reference_rope_then_attention(&q, &k, &v, batch, heads, seq_len, head_dim);
    let (max_abs, idx) = max_abs_diff(&gpu, &cpu);
    eprintln!(
        "rope_q forward vs CPU standard-RoPE: max_abs={max_abs:.2e} at idx={idx} \
         gpu[idx]={:.4} cpu[idx]={:.4}",
        gpu[idx], cpu[idx],
    );
    eprintln!("  first 4 gpu: {:?}", &gpu[..4.min(gpu.len())]);
    eprintln!("  first 4 cpu: {:?}", &cpu[..4.min(cpu.len())]);
    assert!(
        max_abs < 4e-2,
        "rope_q forward != CPU standard-RoPE ref: max_abs={max_abs:.2e} at idx={idx} \
         (threshold 4e-2; if exceeded this is a real RoPE convention finding, not a tolerance issue)"
    );
    eprintln!("rope_q forward PASSED vs CPU standard-RoPE: max_abs={max_abs:.2e}");
}
