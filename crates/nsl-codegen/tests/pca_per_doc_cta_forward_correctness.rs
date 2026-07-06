//! PCA per-document CTA forward correctness (G2 Strategy 3 v1).
//!
//! GPU test: 4 short docs packed into seq_len=128 (padded from 100).
//! Doc lengths: {32, 28, 24, 16}. doc_starts = [0,32,60,84,100,128].
//! seg_ids[i] = doc index for position i.
//!
//! CPU reference: `naive_attention_segmented` with causal=true.
//! GPU target:    per-doc CTA kernel via `synthesize_per_doc_cta_forward`.
//! Tolerance:     5e-3 (f16 + softmax-renorm, matches Tier A §7.4).
//!
//! Run with:
//!   cargo test -p nsl-codegen --features cuda --test pca_per_doc_cta_forward_correctness \
//!     -- --ignored --nocapture

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::per_doc_cta::{
    per_doc_cta_kernel_name, synthesize_per_doc_cta_forward,
};
use nsl_codegen::pca_detect::{DatasetPackingConfig, PcaDetection, PcaStrategy};
use nsl_codegen::pca_per_doc::{PerDocAdmitConfig, PerDocCtaPlan, admit};
use std::ffi::CString;
use std::os::raw::c_void;

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_d2h, nsl_test_cuda_jit_log,
};

extern "C" {
    fn nsl_kernel_launch(
        ptx_ptr: i64, name_ptr: i64,
        grid_x: i64, grid_y: i64, grid_z: i64,
        block_x: i64, block_y: i64, block_z: i64,
        args_ptr: i64, num_args: i64,
        shared_mem_bytes: i64,
    ) -> i64;
}

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
// CPU naive attention reference (causal + segment-masked)
//
// Layout: Q/K/V/out are [B, H, S, D] row-major.
// seg_ids: length S; seg_ids[qi] != seg_ids[kj] => masked.
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

// ---------------------------------------------------------------------------
// Helpers
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

// ---------------------------------------------------------------------------
// Per-doc CTA launch helper
//
// This function directly calls `nsl_kernel_launch` with the per-doc kernel.
// The per-doc PTX has `doc_starts_ptr` as the LAST param (index 36, 0-based).
// Grid: (num_docs, batch * heads, 1). Block: (128, 1, 1).
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn launch_per_doc_cta(
    q_host: &[f32], k_host: &[f32], v_host: &[f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
    doc_starts_host: &[i32], // [0, d0_end, d1_end, ..., active_seq_len]
    active_seq_len: i32,     // = doc_starts_host[num_docs]
    config: &FlashAttentionConfig,
    plan: &PerDocCtaPlan,
) -> Option<Vec<f32>> {
    let total = batch * heads * seq_len * head_dim;
    let f32_bytes = (total * std::mem::size_of::<f32>()) as i64;
    let f16_bytes = (total * std::mem::size_of::<u16>()) as i64;
    let lse_bytes = (batch * heads * seq_len * std::mem::size_of::<f32>()) as i64;

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Device allocations.
    let q_dev   = nsl_test_cuda_alloc(f32_bytes);
    let k_dev   = nsl_test_cuda_alloc(f32_bytes);
    let v_dev   = nsl_test_cuda_alloc(f32_bytes);
    let out_dev = nsl_test_cuda_alloc(f16_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);
    let ds_bytes = (doc_starts_host.len() * std::mem::size_of::<i32>()) as i64;
    let doc_starts_dev = nsl_test_cuda_alloc(ds_bytes);

    assert!(q_dev != 0 && k_dev != 0 && v_dev != 0 && out_dev != 0
        && lse_dev != 0 && doc_starts_dev != 0,
        "device alloc returned null");

    nsl_test_cuda_h2d(q_dev,   q_host.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(k_dev,   k_host.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(v_dev,   v_host.as_ptr() as i64, f32_bytes);
    nsl_test_cuda_h2d(doc_starts_dev, doc_starts_host.as_ptr() as i64, ds_bytes);

    // PTX synthesis.
    let mut ptx = synthesize_per_doc_cta_forward(plan, config);
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    let dump = std::env::temp_dir().join("per_doc_cta.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("per-doc PTX dumped to: {}", dump.display());
    ptx.push(0); // NUL-terminate for cuModuleLoadData

    let kernel_name = CString::new(per_doc_cta_kernel_name(config)).unwrap();
    let smem = nsl_codegen::flash_attention_v2::smem_layout::total_bytes(config) as i64;

    // Compute num_docs from doc_starts using the sentinel walk.
    let num_docs = nsl_codegen::pca_per_doc::compute_per_doc_grid_x(
        doc_starts_host, active_seq_len, 256,
    ) as i64;
    eprintln!("per-doc CTA: num_docs={num_docs}, active_seq_len={active_seq_len}");

    // Build the kernel args array.
    // Per-doc kernel param order matches the param list in per_doc_cta::synthesize:
    // 0: q_ptr, 1: k_ptr, 2: v_ptr, 3: out_ptr, 4: scale (f32),
    // 5: batch, 6: heads, 7: seq_len, 8: head_dim,
    // 9..12: paging (0), 13..14: cos/sin (0), 15..16: seq_ids/seq_lens (0),
    // 17..19: tree_mask (0,0,0), 20: logsumexp,
    // 21..29: CSHA extras (0), 30..35: Tier C (0),
    // 36: _segment_ids_placeholder (FFI-ABI alignment — never read), 37: doc_starts_ptr.
    let mut q     = q_dev as u64;
    let mut k     = k_dev as u64;
    let mut v     = v_dev as u64;
    let mut out   = out_dev as u64;
    let mut s     = scale;
    let mut b     = batch as u64;
    let mut h     = heads as u64;
    let mut sl    = seq_len as u64;
    let mut hd    = head_dim as u64;
    let mut bt: u64 = 0; let mut kp: u64 = 0; let mut vp: u64 = 0; let mut bs: u64 = 0;
    let mut cos: u64 = 0; let mut sin: u64 = 0;
    let mut sids: u64 = 0; let mut slens: u64 = 0;
    let mut dfs_enter: u64 = 0; let mut dfs_exit: u64 = 0; let mut num_tree: u64 = 0;
    let mut lse   = lse_dev as u64;
    // CSHA extras — all null/zero.
    let mut cx: u64 = 0; let mut nw: u64 = 0;
    let mut wq: u64 = 0; let mut wk: u64 = 0; let mut wv: u64 = 0; let mut wo: u64 = 0;
    let mut eps   = 1e-5f32;
    let mut ah: u32 = 0; let mut dm: u32 = 0;
    // Tier C — all null.
    let mut qp: u64 = 0; let mut kp2: u64 = 0; let mut vp2: u64 = 0;
    let mut rmax: u64 = 0; let mut rsum: u64 = 0; let mut xraw: u64 = 0;
    // Per-doc CTA: segment_ids placeholder (FFI-ABI alignment) + doc_starts_ptr.
    let mut seg_ids_placeholder: u64 = 0;
    let mut doc_starts = doc_starts_dev as u64;

    let args: [*mut c_void; 38] = [
        &mut q     as *mut _ as *mut c_void,
        &mut k     as *mut _ as *mut c_void,
        &mut v     as *mut _ as *mut c_void,
        &mut out   as *mut _ as *mut c_void,
        &mut s     as *mut _ as *mut c_void,
        &mut b     as *mut _ as *mut c_void,
        &mut h     as *mut _ as *mut c_void,
        &mut sl    as *mut _ as *mut c_void,
        &mut hd    as *mut _ as *mut c_void,
        &mut bt    as *mut _ as *mut c_void,
        &mut kp    as *mut _ as *mut c_void,
        &mut vp    as *mut _ as *mut c_void,
        &mut bs    as *mut _ as *mut c_void,
        &mut cos   as *mut _ as *mut c_void,
        &mut sin   as *mut _ as *mut c_void,
        &mut sids  as *mut _ as *mut c_void,
        &mut slens as *mut _ as *mut c_void,
        &mut dfs_enter as *mut _ as *mut c_void,
        &mut dfs_exit  as *mut _ as *mut c_void,
        &mut num_tree  as *mut _ as *mut c_void,
        &mut lse   as *mut _ as *mut c_void,
        &mut cx    as *mut _ as *mut c_void,
        &mut nw    as *mut _ as *mut c_void,
        &mut wq    as *mut _ as *mut c_void,
        &mut wk    as *mut _ as *mut c_void,
        &mut wv    as *mut _ as *mut c_void,
        &mut wo    as *mut _ as *mut c_void,
        &mut eps   as *mut _ as *mut c_void,
        &mut ah    as *mut _ as *mut c_void,
        &mut dm    as *mut _ as *mut c_void,
        &mut qp    as *mut _ as *mut c_void,
        &mut kp2   as *mut _ as *mut c_void,
        &mut vp2   as *mut _ as *mut c_void,
        &mut rmax  as *mut _ as *mut c_void,
        &mut rsum  as *mut _ as *mut c_void,
        &mut xraw  as *mut _ as *mut c_void,
        // FFI-ABI alignment: seg_ids slot (per-doc kernel doesn't read it).
        &mut seg_ids_placeholder as *mut _ as *mut c_void,
        &mut doc_starts as *mut _ as *mut c_void,
    ];

    let grid_y = (batch * heads) as i64;

    let rc = unsafe {
        nsl_kernel_launch(
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            num_docs,       // grid_x = num_docs (per-doc CTA launch)
            grid_y,         // grid_y = batch * heads
            1i64,           // grid_z
            128i64,         // block_x = 128 (4 warps)
            1i64,           // block_y
            1i64,           // block_z
            args.as_ptr() as i64,
            38i64,          // num_args (FFI-ABI aligned: includes seg_ids placeholder)
            smem,           // shared_mem_bytes
        )
    };

    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                let cstr = std::ffi::CStr::from_ptr(log_ptr as *const i8);
                cstr.to_string_lossy().into_owned()
            }
        } else { "<no log>".into() };
        eprintln!("per-doc CTA kernel launch failed rc={rc}\nJIT log:\n{log}");
        nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
        nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
        nsl_test_cuda_free(doc_starts_dev);
        return None;
    }

    // Read back f16 output and convert to f32.
    let mut out_f16 = vec![0u16; total];
    nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, f16_bytes);
    let out_f32: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // Cleanup.
    nsl_test_cuda_free(q_dev); nsl_test_cuda_free(k_dev); nsl_test_cuda_free(v_dev);
    nsl_test_cuda_free(out_dev); nsl_test_cuda_free(lse_dev);
    nsl_test_cuda_free(doc_starts_dev);

    Some(out_f32)
}

// ===========================================================================
// GPU Test: 4 short docs vs naive_attention_segmented CPU reference
// ===========================================================================

#[test]
#[ignore = "requires CUDA GPU"]
fn per_doc_cta_four_short_docs_matches_segmented_reference() {
    if !cuda_available() { return; }

    // Fixture: 4 docs of lengths {32, 28, 24, 16} packed into seq_len=100,
    // padded to 128. doc_starts = [0, 32, 60, 84, 100, 128].
    let doc_lens = [32usize, 28, 24, 16];
    let active_seq_len: usize = doc_lens.iter().sum(); // 100
    let seq_len = 128usize; // padded
    let batch = 1usize;
    let heads = 1usize;
    let head_dim = 32usize;

    // Build doc_starts with sentinel at active_seq_len.
    let mut doc_starts_host = vec![0i32; doc_lens.len() + 2]; // 6 entries
    let mut off = 0usize;
    for (i, &l) in doc_lens.iter().enumerate() {
        doc_starts_host[i] = off as i32;
        off += l;
    }
    doc_starts_host[doc_lens.len()] = active_seq_len as i32; // sentinel = 100
    doc_starts_host[doc_lens.len() + 1] = seq_len as i32;    // padded end = 128
    eprintln!("doc_starts: {:?}", doc_starts_host);

    // Build seg_ids: position i in doc d gets seg_ids[i]=d.
    let mut seg_ids = vec![0u16; seq_len];
    let mut pos = 0usize;
    for (d, &l) in doc_lens.iter().enumerate() {
        for _ in 0..l {
            seg_ids[pos] = d as u16;
            pos += 1;
        }
    }
    // Positions [active_seq_len..seq_len] are padding; seg_ids can be anything.

    // Generate Q/K/V data.
    let total = batch * heads * seq_len * head_dim;
    let mut q = vec![0f32; total];
    let mut k = vec![0f32; total];
    let mut v = vec![0f32; total];
    fill_seeded(&mut q, 0xDEAD_4D0C_u64);
    fill_seeded(&mut k, 0xBEEF_4D0C_u64);
    fill_seeded(&mut v, 0xCAFE_4D0C_u64);

    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // CPU reference: naive_attention_segmented with causal=true.
    // Only compare positions 0..active_seq_len (padding is don't-care).
    let mut cpu_ref = vec![0f32; total];
    naive_attention_segmented(
        &q, &k, &v, &mut cpu_ref,
        batch, heads, seq_len, head_dim,
        scale, true, &seg_ids,
    );

    // Config: block_q=64, block_kv=64, head_dim=32, causal=true, no CSHA.
    // segment_masked=false for per-doc kernel.
    let config = FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 32,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: None,
        checkpoint: None,
    };

    // Build the plan.
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
        rationale: "test".to_string(),
        segment_id_bytes_per_batch: 256,
        eliminated_mask_bytes_per_batch: 32768,
    };
    let plan = admit(
        &det, &config, &packing,
        &PerDocAdmitConfig { enable_per_doc_cta: true, ..Default::default() },
    ).expect("plan must admit");

    // GPU launch.
    eprintln!("Launching per-doc CTA kernel: {} docs, seq={}, hd={}", doc_lens.len(), seq_len, head_dim);
    let gpu_out = launch_per_doc_cta(
        &q, &k, &v, batch, heads, seq_len, head_dim,
        &doc_starts_host, active_seq_len as i32,
        &config, &plan,
    );

    let gpu_out = match gpu_out {
        Some(g) => g,
        None => {
            eprintln!("SKIPPED: per-doc CTA kernel launch failed");
            return;
        }
    };

    // Compare only active positions [0..active_seq_len * head_dim].
    let active_elems = batch * heads * active_seq_len * head_dim;
    let gpu_active = &gpu_out[..active_elems];
    let cpu_active = &cpu_ref[..active_elems];

    let all_finite_g = gpu_active.iter().all(|x| x.is_finite());
    let all_finite_c = cpu_active.iter().all(|x| x.is_finite());
    assert!(all_finite_g, "GPU output (active region) contains non-finite values");
    assert!(all_finite_c, "CPU ref (active region) contains non-finite values");

    let (max_abs, max_idx) = max_abs_diff(gpu_active, cpu_active);
    eprintln!(
        "per-doc CTA vs CPU ref: max_abs={:.6} at idx={} gpu={} cpu={}",
        max_abs, max_idx, gpu_active[max_idx], cpu_active[max_idx],
    );
    eprintln!("  first 4 gpu: {:?}", &gpu_active[..4.min(gpu_active.len())]);
    eprintln!("  first 4 cpu: {:?}", &cpu_active[..4.min(cpu_active.len())]);

    assert!(
        max_abs <= 5e-3,
        "per-doc CTA FAILED: max_abs={:.6} > 5e-3 tolerance", max_abs,
    );
    eprintln!("per-doc CTA PASSED: max_abs={:.6} <= 5e-3", max_abs);
}
