//! PCA Stage C: GPU parity test for the `nsl_sdpa_fused_forward` runtime
//! FFI against an f64 CPU oracle (causal[-within-segment] softmax
//! attention).
//!
//! This exercises the EXACT production path of the decorator-free fused
//! SDPA dispatch: the PTX comes from
//! `pca_tier_b::emit_tier_b_variants_for_config(&config).base_ptx` (which
//! is what `ensure_sdpa_fwd_variant_table` embeds), the kernel name from
//! `emission.base_kernel_name`, and the dynamic-SMEM request from
//! `flash_attention_selector::shared_mem_bytes_selected` — then launches
//! through the runtime FFI with real GPU `NslTensor` handles, exactly as
//! the wengert_lower call site does.
//!
//! Historical note (the bug this test pins): the v2 forward epilogue
//! stores its output as f16 (`cvt.rn.f16.f32` + `st.global.b16` in
//! flash_attention_v2/phases/forward/finalize.rs). The first FFI version
//! pointed the kernel's out param at the final f32 tensor, so downstream
//! consumers read f16 bit pairs as f32 (~0 everywhere) and GPU packed
//! training produced a first-step loss of ~ln(vocab) (near-uniform
//! logits) instead of the decomposed-path 5.1085. The FFI now stages f16
//! and widens on-device; this test validates the full staging + widening
//! + list-return path against the oracle.
//!
//! Tolerance: 5e-3 on `out` (f16 mantissa + online-softmax renorm jitter,
//! matching pca_tier_a_forward_correctness.rs), 5e-3 on `lse` (f32 with
//! ex2/lg2 approximations).
//!
//! Gated with #[cfg(feature = "cuda")] and #[ignore] per the
//! csha_cuda_launch_classic.rs pattern. Run with:
//!
//!   cargo test -p nsl-codegen --features cuda \
//!     --test sdpa_fused_forward_gpu_parity -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_selector::shared_mem_bytes_selected;
use nsl_codegen::pca_tier_b::emit_tier_b_variants_for_config;

use nsl_runtime::flash_attention::nsl_sdpa_fused_forward;
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d, nsl_test_cuda_jit_log};

// ── Fixture geometry (per the Stage-C investigation spec) ──────────────────

const B: usize = 1;
const H: usize = 1;
const S: usize = 64;
const D: usize = 32;
const BLOCK_Q: i64 = 64;
const BLOCK_KV: i64 = 64;

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    nsl_cuda_init() == 0
}

/// Deterministic pseudo-random values in roughly [-0.5, 0.5] (LCG).
fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed as u64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

fn fixture_config(segment_masked: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: BLOCK_Q,
        block_kv: BLOCK_KV,
        head_dim: D as i64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 90,
        segment_masked,
        csha: None,
        checkpoint: None,
    }
}

/// Build a GPU f32 NslTensor of `shape` and fill it with `vals`.
fn gpu_tensor(shape: &[i64], vals: &[f32]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    assert_ne!(t, 0, "GPU tensor alloc failed");
    let dptr = nsl_tensor_data_ptr(t);
    nsl_test_cuda_h2d(dptr, vals.as_ptr() as i64, (vals.len() * 4) as i64);
    t
}

/// Build a HOST segment-id NslTensor [B, S]. `nsl_tensor_zeros_on(.., 0)`
/// allocates f32 storage (dtype=1) — the same layout packing.rs emits —
/// and the FFI's staging helper narrows the f32 ids to the u16 entries
/// the kernel reads.
fn host_segment_tensor(seg: &[u16]) -> i64 {
    assert_eq!(seg.len(), B * S);
    let shape_list = nsl_list_new();
    nsl_list_push(shape_list, B as i64);
    nsl_list_push(shape_list, S as i64);
    let t = nsl_tensor_zeros_on(shape_list, 0 /* CPU */);
    nsl_list_free(shape_list);
    assert_ne!(t, 0, "CPU tensor alloc failed");
    let dptr = nsl_tensor_data_ptr(t) as *mut f32;
    for (i, &v) in seg.iter().enumerate() {
        unsafe { *dptr.add(i) = v as f32 };
    }
    t
}

/// f64 oracle: causal softmax attention, optionally segment-masked
/// (position pair (i, j) visible iff j <= i AND same segment id).
/// Returns (out, lse) — lse is the natural logsumexp of the scaled
/// scores over each row's visible set (what the v2 kernel saves).
fn oracle_attention_f64(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seg: Option<&[u16]>, // [B*S]
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0f64 / (D as f64).sqrt();
    let mut out = vec![0.0f32; B * H * S * D];
    let mut lse = vec![0.0f32; B * H * S];
    for bb in 0..B {
        for hh in 0..H {
            let base = (bb * H + hh) * S * D;
            let lse_base = (bb * H + hh) * S;
            for i in 0..S {
                let visible: Vec<usize> = (0..=i)
                    .filter(|&j| match seg {
                        Some(sg) => sg[bb * S + i] == sg[bb * S + j],
                        None => true,
                    })
                    .collect();
                assert!(!visible.is_empty(), "row {i} has an empty visible set");
                let score = |j: usize| -> f64 {
                    let mut dot = 0.0f64;
                    for dd in 0..D {
                        dot += q[base + i * D + dd] as f64 * k[base + j * D + dd] as f64;
                    }
                    dot * scale
                };
                let m = visible
                    .iter()
                    .map(|&j| score(j))
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0f64;
                for &j in &visible {
                    denom += (score(j) - m).exp();
                }
                for dd in 0..D {
                    let mut acc = 0.0f64;
                    for &j in &visible {
                        let p = (score(j) - m).exp() / denom;
                        acc += p * v[base + j * D + dd] as f64;
                    }
                    out[base + i * D + dd] = acc as f32;
                }
                lse[lse_base + i] = (m + denom.ln()) as f32;
            }
        }
    }
    (out, lse)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0f32;
    let mut max_idx = 0usize;
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        let d = (ai - bi).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }
    (max_abs, max_idx)
}

/// Synthesize the production PTX pair, launch through the FFI, read back
/// (out, lse) as host f32. Panics with the JIT log on launch failure.
fn run_fused_forward(segment_masked: bool, seg: Option<&[u16]>) -> (Vec<f32>, Vec<f32>) {
    let total = B * H * S * D;
    let q_host = det_seq(0xA1B2_C3D4, total);
    let k_host = det_seq(0x1234_5678, total);
    let v_host = det_seq(0xDEAD_BEEF, total);

    let q_t = gpu_tensor(&[B as i64, H as i64, S as i64, D as i64], &q_host);
    let k_t = gpu_tensor(&[B as i64, H as i64, S as i64, D as i64], &k_host);
    let v_t = gpu_tensor(&[B as i64, H as i64, S as i64, D as i64], &v_host);
    let seg_t = match seg {
        Some(sg) => host_segment_tensor(sg),
        None => 0,
    };

    // EXACTLY what ensure_sdpa_fwd_variant_table embeds.
    let config = fixture_config(segment_masked);
    let emission = emit_tier_b_variants_for_config(&config);
    let mut ptx = emission.base_ptx.clone();
    while ptx.last() == Some(&0) {
        ptx.pop();
    }
    if ptx.last() != Some(&b'\n') {
        ptx.push(b'\n');
    }
    ptx.push(0); // null-terminate for cuModuleLoadData
    let kernel_name = CString::new(emission.base_kernel_name.clone()).unwrap();
    let smem = shared_mem_bytes_selected(&config) as i64;
    let scale = 1.0f32 / (D as f32).sqrt();

    eprintln!(
        "launching {} (segment_masked={segment_masked}, smem={smem})",
        emission.base_kernel_name
    );

    let list_ptr = nsl_sdpa_fused_forward(
        q_t,
        k_t,
        v_t,
        scale.to_bits() as i64,
        1, // causal (baked into the PTX; accepted for symmetry)
        seg_t,
        ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        0,
        0, // tier-b disabled sentinel
        BLOCK_Q,
        BLOCK_KV,
        smem,
    );
    if list_ptr == 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no log>".into()
        };
        panic!(
            "nsl_sdpa_fused_forward DECLINED (returned 0) for a valid GPU \
             fixture (segment_masked={segment_masked}) — JIT log:\n{log}"
        );
    }

    let out_t = nsl_list_get(list_ptr, 0);
    let lse_t = nsl_list_get(list_ptr, 1);
    assert_ne!(out_t, 0, "list slot 0 (out) is null");
    assert_ne!(lse_t, 0, "list slot 1 (lse) is null");

    let mut out_host = vec![0.0f32; total];
    nsl_test_cuda_d2h(
        out_host.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(out_t),
        (total * 4) as i64,
    );
    let total_lse = B * H * S;
    let mut lse_host = vec![0.0f32; total_lse];
    nsl_test_cuda_d2h(
        lse_host.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(lse_t),
        (total_lse * 4) as i64,
    );

    // Cleanup (list does not own elements; tensors own their device data).
    nsl_tensor_free(out_t);
    nsl_tensor_free(lse_t);
    nsl_list_free(list_ptr);
    nsl_tensor_free(q_t);
    nsl_tensor_free(k_t);
    nsl_tensor_free(v_t);
    if seg_t != 0 {
        nsl_tensor_free(seg_t);
    }

    (out_host, lse_host)
}

fn assert_matches_oracle(name: &str, seg: Option<&[u16]>, segment_masked: bool) {
    let total = B * H * S * D;
    let q_host = det_seq(0xA1B2_C3D4, total);
    let k_host = det_seq(0x1234_5678, total);
    let v_host = det_seq(0xDEAD_BEEF, total);
    let (oracle_out, oracle_lse) = oracle_attention_f64(&q_host, &k_host, &v_host, seg);

    let (gpu_out, gpu_lse) = run_fused_forward(segment_masked, seg);

    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "{name}: GPU out contains non-finite values"
    );
    let out_norm: f32 = gpu_out.iter().map(|x| x * x).sum();
    assert!(
        out_norm > 1e-6,
        "{name}: GPU out is (near-)all-zero — kernel wrote nothing or \
         the f16 widening path is broken (norm={out_norm})"
    );

    let (out_err, out_idx) = max_abs_diff(&oracle_out, &gpu_out);
    let (lse_err, lse_idx) = max_abs_diff(&oracle_lse, &gpu_lse);
    eprintln!(
        "{name}: out max_abs={out_err:.6} at idx={out_idx} \
         (oracle={}, gpu={}); lse max_abs={lse_err:.6} at idx={lse_idx} \
         (oracle={}, gpu={})",
        oracle_out[out_idx], gpu_out[out_idx], oracle_lse[lse_idx], gpu_lse[lse_idx],
    );
    assert!(
        out_err < 5e-3,
        "{name}: out mismatch vs f64 oracle: {out_err} at idx {out_idx}"
    );
    assert!(
        lse_err < 5e-3,
        "{name}: lse mismatch vs f64 oracle: {lse_err} at idx {lse_idx}"
    );
}

/// Unmasked (plain causal) fused forward vs oracle — isolates the base
/// kernel + FFI marshal from the segment arm.
#[test]
#[ignore = "requires CUDA GPU"]
fn sdpa_fused_forward_unmasked_matches_oracle() {
    if !cuda_available() {
        return;
    }
    assert_matches_oracle("unmasked", None, false);
}

/// Segment-masked fused forward vs oracle: two packed documents
/// (positions 0..32 = doc 0, 32..64 = doc 1). Cross-segment attention
/// must be fully suppressed — this is the Stage-C production
/// configuration (segmask=1 in the launch marker).
#[test]
#[ignore = "requires CUDA GPU"]
fn sdpa_fused_forward_segment_masked_matches_oracle() {
    if !cuda_available() {
        return;
    }
    let mut seg = vec![0u16; B * S];
    for i in 32..64 {
        seg[i] = 1;
    }
    assert_matches_oracle("segment-masked", Some(&seg), true);
}
