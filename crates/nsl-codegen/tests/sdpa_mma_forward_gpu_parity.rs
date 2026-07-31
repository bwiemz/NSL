//! GPU parity for the tensor-core SDPA forward (`mma_forward`) against an
//! f64 oracle, on MULTI-TILE geometries the original Stage-C fixture
//! (S=64 = one KV tile) never reaches:
//!
//!   * cross-tile online softmax (running max/sum rescale of O),
//!   * the causal early-exit KV bound,
//!   * the V^T-overwrites-K SMEM aliasing across loop iterations,
//!   * batch*head grid routing (B, H > 1),
//!   * head_dim 128 (largest table variant, highest register pressure).
//!
//! Each case also PINS which body it exercises via the kernel name: a
//! gpu_sm=90 config must launch `..._v2_mma` and gpu_sm=75 must launch
//! the scalar `..._v2` — so neither path can silently vanish from
//! coverage (the Tier B.1 meta-lesson: structural gates never catch what
//! only an e2e numerical gate can).
//!
//! Run: cargo test -p nsl-codegen --features cuda \
//!        --test sdpa_mma_forward_gpu_parity -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_selector::shared_mem_bytes_selected;
use nsl_codegen::pca_tier_b::emit_tier_b_variants_for_config;

use nsl_runtime::flash_attention::nsl_sdpa_fused_forward;
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d, nsl_test_cuda_jit_log};

#[derive(Clone, Copy)]
struct Geom {
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    causal: bool,
    gpu_sm: u32,
    expect_mma: bool,
    block_q: i64,
    block_kv: i64,
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    nsl_cuda_init() == 0
}

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

fn config_for(g: &Geom) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: g.block_q,
        block_kv: g.block_kv,
        head_dim: g.d as i64,
        causal: g.causal,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: g.gpu_sm,
        segment_masked: false,
        csha: None,
        checkpoint: None,
    }
}

fn gpu_tensor(shape: &[i64], vals: &[f32]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1);
    nsl_list_free(shape_list);
    assert_ne!(t, 0, "GPU tensor alloc failed");
    nsl_test_cuda_h2d(
        nsl_tensor_data_ptr(t),
        vals.as_ptr() as i64,
        (vals.len() * 4) as i64,
    );
    t
}

/// f64 oracle: softmax attention, causal per `g.causal`. lse is the
/// natural logsumexp of the scaled scores (finalize.rs definition).
fn oracle(g: &Geom, q: &[f32], k: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (b, h, s, d) = (g.b, g.h, g.s, g.d);
    let scale = 1.0f64 / (d as f64).sqrt();
    let mut out = vec![0.0f32; b * h * s * d];
    let mut lse = vec![0.0f32; b * h * s];
    for bb in 0..b {
        for hh in 0..h {
            let base = (bb * h + hh) * s * d;
            let lse_base = (bb * h + hh) * s;
            for i in 0..s {
                let hi = if g.causal { i } else { s - 1 };
                let score = |j: usize| -> f64 {
                    let mut dot = 0.0f64;
                    for dd in 0..d {
                        dot += q[base + i * d + dd] as f64 * k[base + j * d + dd] as f64;
                    }
                    dot * scale
                };
                let m = (0..=hi).map(score).fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0f64;
                for j in 0..=hi {
                    denom += (score(j) - m).exp();
                }
                for dd in 0..d {
                    let mut acc = 0.0f64;
                    for j in 0..=hi {
                        acc += (score(j) - m).exp() / denom * v[base + j * d + dd] as f64;
                    }
                    out[base + i * d + dd] = acc as f32;
                }
                lse[lse_base + i] = (m + denom.ln()) as f32;
            }
        }
    }
    (out, lse)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut max_abs = 0f32;
    let mut max_idx = 0usize;
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        let df = (ai - bi).abs();
        if df > max_abs {
            max_abs = df;
            max_idx = i;
        }
    }
    (max_abs, max_idx)
}

fn run_case(name: &str, g: Geom) {
    if !cuda_available() {
        return;
    }
    let total = g.b * g.h * g.s * g.d;
    let q_host = det_seq(0xA1B2_C3D4 ^ g.d as u32, total);
    let k_host = det_seq(0x1234_5678 ^ g.s as u32, total);
    let v_host = det_seq(0xDEAD_BEEF ^ g.gpu_sm, total);

    let config = config_for(&g);
    let emission = emit_tier_b_variants_for_config(&config);
    assert_eq!(
        emission.base_kernel_name.ends_with("_mma"),
        g.expect_mma,
        "{name}: dispatch pin — kernel {} (expect_mma={})",
        emission.base_kernel_name,
        g.expect_mma
    );
    let mut ptx = emission.base_ptx.clone();
    while ptx.last() == Some(&0) {
        ptx.pop();
    }
    if ptx.last() != Some(&b'\n') {
        ptx.push(b'\n');
    }
    ptx.push(0);
    let kernel_name = CString::new(emission.base_kernel_name.clone()).unwrap();
    let smem = shared_mem_bytes_selected(&config) as i64;
    let scale = 1.0f32 / (g.d as f32).sqrt();

    let dims = [g.b as i64, g.h as i64, g.s as i64, g.d as i64];
    let q_t = gpu_tensor(&dims, &q_host);
    let k_t = gpu_tensor(&dims, &k_host);
    let v_t = gpu_tensor(&dims, &v_host);

    eprintln!("{name}: launching {} (smem={smem})", emission.base_kernel_name);
    let list_ptr = nsl_sdpa_fused_forward(
        q_t,
        k_t,
        v_t,
        scale.to_bits() as i64,
        g.causal as i64,
        0,
        ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        0,
        0,
        config.block_q,
        config.block_kv,
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
        panic!("{name}: nsl_sdpa_fused_forward DECLINED — JIT log:\n{log}");
    }

    let out_t = nsl_list_get(list_ptr, 0);
    let lse_t = nsl_list_get(list_ptr, 1);
    let mut gpu_out = vec![0.0f32; total];
    nsl_test_cuda_d2h(
        gpu_out.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(out_t),
        (total * 4) as i64,
    );
    let mut gpu_lse = vec![0.0f32; g.b * g.h * g.s];
    nsl_test_cuda_d2h(
        gpu_lse.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(lse_t),
        ((g.b * g.h * g.s) * 4) as i64,
    );
    nsl_tensor_free(out_t);
    nsl_tensor_free(lse_t);
    nsl_list_free(list_ptr);
    nsl_tensor_free(q_t);
    nsl_tensor_free(k_t);
    nsl_tensor_free(v_t);

    assert!(
        gpu_out.iter().all(|x| x.is_finite()),
        "{name}: non-finite output"
    );
    let norm: f32 = gpu_out.iter().map(|x| x * x).sum();
    assert!(norm > 1e-6, "{name}: output near-all-zero (norm={norm})");

    let (oracle_out, oracle_lse) = oracle(&g, &q_host, &k_host, &v_host);
    let (out_err, out_idx) = max_abs_diff(&oracle_out, &gpu_out);
    let (lse_err, lse_idx) = max_abs_diff(&oracle_lse, &gpu_lse);
    eprintln!(
        "{name}: out max_abs={out_err:.6} at {out_idx} (oracle={}, gpu={}); \
         lse max_abs={lse_err:.6} at {lse_idx}",
        oracle_out[out_idx], gpu_out[out_idx]
    );
    assert!(out_err < 5e-3, "{name}: out mismatch {out_err} at {out_idx}");
    assert!(lse_err < 5e-3, "{name}: lse mismatch {lse_err} at {lse_idx}");
}

/// Production shape: hd=64, two KV tiles, multi-batch multi-head, causal.
/// Exercises the cross-tile online-softmax rescale and the early-exit.
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_multitile_hd64_causal() {
    run_case("mma_hd64_causal", Geom {
        b: 2, h: 2, s: 128, d: 64, causal: true, gpu_sm: 90, expect_mma: true,
        block_q: 64, block_kv: 64,
    });
}

/// Same geometry, no causal mask: full KV loop with every column live.
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_multitile_hd64_noncausal() {
    run_case("mma_hd64_noncausal", Geom {
        b: 2, h: 2, s: 128, d: 64, causal: false, gpu_sm: 90, expect_mma: true,
        block_q: 64, block_kv: 64,
    });
}

/// The scalar body stays covered: gpu_sm=75 must launch the non-_mma
/// kernel and still match the oracle on the same multi-tile geometry.
#[test]
#[ignore = "requires CUDA GPU"]
fn scalar_pinned_multitile_hd64_causal() {
    run_case("scalar_hd64_causal", Geom {
        b: 2, h: 2, s: 128, d: 64, causal: true, gpu_sm: 75, expect_mma: false,
        block_q: 64, block_kv: 64,
    });
}

/// Largest table variant: hd=128, three KV tiles. Highest register
/// pressure (16 O-tiles per lane) — correctness here implies ptxas found
/// a live allocation even if it spills.
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_multitile_hd128_causal() {
    run_case("mma_hd128_causal", Geom {
        b: 1, h: 2, s: 192, d: 128, causal: true, gpu_sm: 90, expect_mma: true,
        block_q: 64, block_kv: 64,
    });
}

/// Smallest table head_dim on the MMA path (hd=32, matches the original
/// Stage-C fixture head_dim at multi-tile seq).
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_multitile_hd32_causal() {
    run_case("mma_hd32_causal", Geom {
        b: 2, h: 2, s: 128, d: 32, causal: true, gpu_sm: 90, expect_mma: true,
        block_q: 64, block_kv: 64,
    });
}

/// block_q=32: m_tiles=2, so warps 2/3 redundantly recompute m-tiles 0/1
/// and are excluded from stores by %fm_pstore. The only GPU coverage of
/// the redundant-warp arm (the table picks (32,32) for short sequences).
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_multitile_bq32_redundant_warps_causal() {
    run_case("mma_bq32_causal", Geom {
        b: 2, h: 2, s: 96, d: 64, causal: true, gpu_sm: 90, expect_mma: true,
        block_q: 32, block_kv: 32,
    });
}
