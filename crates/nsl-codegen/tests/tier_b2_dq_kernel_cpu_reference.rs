//! Layer-1 dQ-kernel parity tests vs CPU reference.
//!
//! All tests are `#[ignore]` + require `feature="cuda"`. Manual GPU invocation:
//!     cargo test -p nsl-codegen --test tier_b2_dq_kernel_cpu_reference \
//!         --features cuda -- --ignored --nocapture
//!
//! - Test 1: D pre-pass standalone vs CPU rowsum (Task 7)
//! - Test 2: dQ smoke at canonical (Task 14)
//! - Test 3: dQ head_dim sweep (Task 14)
//!
//! GPU launcher helpers are STUBS for Phase 2 — real cudarc-based launchers wire
//! during PR review when manual GPU runs are performed. The `unimplemented!()`
//! bodies trip immediately when an #[ignore]'d test is actually run, which is
//! exactly the right behavior for "the test must be executed manually on GPU
//! hardware before closure."
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §6.1

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::d_prepass::synthesize_d_prepass;
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn tol_for_head_dim(hd: u32) -> f32 {
    if hd >= 128 {
        4e-2
    } else if hd >= 64 {
        2e-2
    } else {
        5e-3
    }
}

fn canonical_hd32_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

fn cfg(bq: i64, hd: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bq,
        head_dim: hd,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

// === Test 1: D pre-pass standalone (from Task 7) ===

#[test]
#[ignore]
fn tier_b2_d_prepass_vs_cpu_reduction() {
    // Test 1 of §6.1: D pre-pass standalone vs CPU rowsum(dO * O).
    // Validates the D pre-pass kernel in isolation, before dQ-kernel consumes D.
    // Tolerance: tol_for_head_dim(32) = 5e-3.

    let cfg_c = canonical_hd32_cfg();
    let batch = 1usize;
    let heads = 1usize;
    let seq = 32usize;
    let hd = cfg_c.head_dim as usize;

    // Seed-deterministic random inputs.
    let d_o_host: Vec<half::f16> = (0..batch * heads * seq * hd)
        .map(|i| half::f16::from_f32(((i as f32 * 0.137).sin() * 0.1) as f32))
        .collect();
    let o_host: Vec<half::f16> = (0..batch * heads * seq * hd)
        .map(|i| half::f16::from_f32(((i as f32 * 0.241).cos() * 0.1) as f32))
        .collect();

    // CPU reference: D[b,h,q] = sum_d (dO[b,h,q,d] * O[b,h,q,d])
    let mut d_ref = vec![0.0f32; batch * heads * seq];
    for b in 0..batch {
        for h in 0..heads {
            for q in 0..seq {
                let base = ((b * heads + h) * seq + q) * hd;
                let sum: f32 = (0..hd)
                    .map(|d| d_o_host[base + d].to_f32() * o_host[base + d].to_f32())
                    .sum();
                d_ref[(b * heads + h) * seq + q] = sum;
            }
        }
    }

    // GPU: emit + launch D pre-pass.
    let ptx = synthesize_d_prepass(&cfg_c).expect("D pre-pass synthesis");
    let d_gpu = run_d_prepass_on_gpu(&ptx, &d_o_host, &o_host, batch, heads, seq, hd);

    // Compare.
    let tol = tol_for_head_dim(hd as u32);
    let max_abs = d_ref
        .iter()
        .zip(d_gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_abs <= tol,
        "D pre-pass max_abs {} > tolerance {} for hd={}",
        max_abs,
        tol,
        hd
    );
}

// === Test 2: dQ-kernel smoke at canonical (Task 14) ===

#[test]
#[ignore]
fn tier_b2_dq_smoke_canonical() {
    // (bq=bkv=32, hd=32, non-causal, no RoPE). Tolerance: tol_for_head_dim(32) = 5e-3.
    let cfg_c = cfg(32, 32);
    let max_abs = run_dq_kernel_parity(&cfg_c, /*batch=*/ 1, /*heads=*/ 1, /*seq=*/ 32);
    let tol = tol_for_head_dim(cfg_c.head_dim as u32);
    assert!(
        max_abs <= tol,
        "dQ smoke canonical: max_abs {} > tol {} for hd={}",
        max_abs,
        tol,
        cfg_c.head_dim
    );
}

// === Test 3: dQ-kernel head_dim sweep (Task 14) ===

#[test]
#[ignore]
fn tier_b2_dq_head_dim_sweep() {
    for &hd in &[32i64, 64, 128] {
        let bq = if hd >= 128 { 64 } else { 32 };
        let cfg_c = cfg(bq, hd);
        let seq = if hd >= 128 { 128 } else { 64 };
        let max_abs = run_dq_kernel_parity(&cfg_c, 1, 1, seq);
        let tol = tol_for_head_dim(hd as u32);
        assert!(
            max_abs <= tol,
            "dQ sweep hd={}: max_abs {} > tol {}",
            hd,
            max_abs,
            tol
        );
    }
}

// === Test orchestrator ===

/// Returns max_abs(dQ_gpu - dQ_cpu_reference) over all elements.
/// Calls into the GPU launcher stubs; the stubs panic via unimplemented!() when
/// invoked, which means these tests fail loudly until the launchers are wired
/// during PR-prep / GPU validation.
fn run_dq_kernel_parity(
    cfg: &FlashAttentionConfig,
    batch: usize,
    heads: usize,
    seq: usize,
) -> f32 {
    let hd = cfg.head_dim as usize;
    let q: Vec<half::f16> = (0..batch * heads * seq * hd)
        .map(|i| half::f16::from_f32(((i as f32 * 0.137).sin() * 0.1) as f32))
        .collect();
    let k: Vec<half::f16> = (0..batch * heads * seq * hd)
        .map(|i| half::f16::from_f32(((i as f32 * 0.211).cos() * 0.1) as f32))
        .collect();
    let v: Vec<half::f16> = (0..batch * heads * seq * hd)
        .map(|i| half::f16::from_f32(((i as f32 * 0.317).sin() * 0.1) as f32))
        .collect();
    let d_o: Vec<half::f16> = (0..batch * heads * seq * hd)
        .map(|i| half::f16::from_f32(((i as f32 * 0.419).cos() * 0.1) as f32))
        .collect();

    let (row_max, row_sum, o) = run_b1_forward_for_test(&q, &k, &v, batch, heads, seq, hd);
    let d_prepass_ptx = synthesize_d_prepass(cfg).expect("d_prepass synth");
    let d = run_d_prepass_on_gpu(&d_prepass_ptx, &d_o, &o, batch, heads, seq, hd);
    let dq_ptx = synthesize_dq_kernel(cfg).expect("dq_kernel synth");
    let dq_gpu = run_dq_kernel_on_gpu(
        &dq_ptx, &q, &k, &v, &d_o, &row_max, &row_sum, &d, batch, heads, seq, hd,
    );
    let dq_ref =
        cpu_naive_dq(&q, &k, &v, &d_o, &row_max, &row_sum, &d, batch, heads, seq, hd);
    dq_gpu
        .iter()
        .zip(dq_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

// === CPU naive reference for dQ ===

fn cpu_naive_dq(
    q: &[half::f16],
    k: &[half::f16],
    v: &[half::f16],
    d_o: &[half::f16],
    row_max: &[f32],
    row_sum: &[f32],
    d: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
) -> Vec<f32> {
    // dQ[b,h,q,e] = sum_k dS[q,k] * K[k,e]
    //   where dS[q,k] = P[q,k] * (dP[q,k] - D[q])
    //         P[q,k]  = exp(QK^T[q,k] - row_max[q]) / row_sum[q]
    //         dP[q,k] = sum_e dO[q,e] * V[k,e]
    let mut dq = vec![0.0f32; batch * heads * seq * hd];
    let scale = 1.0f32 / (hd as f32).sqrt();
    for b in 0..batch {
        for h in 0..heads {
            for qi in 0..seq {
                let qbase = ((b * heads + h) * seq + qi) * hd;
                let rmax_idx = (b * heads + h) * seq + qi;
                let s_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        let kbase = ((b * heads + h) * seq + ki) * hd;
                        let s: f32 = (0..hd)
                            .map(|di| q[qbase + di].to_f32() * k[kbase + di].to_f32())
                            .sum();
                        s * scale
                    })
                    .collect();
                let p_row: Vec<f32> = s_row
                    .iter()
                    .map(|&s| (s - row_max[rmax_idx]).exp() / row_sum[rmax_idx])
                    .collect();
                let dp_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        let kbase = ((b * heads + h) * seq + ki) * hd;
                        (0..hd)
                            .map(|di| d_o[qbase + di].to_f32() * v[kbase + di].to_f32())
                            .sum()
                    })
                    .collect();
                let ds_row: Vec<f32> = (0..seq)
                    .map(|ki| p_row[ki] * (dp_row[ki] - d[rmax_idx]))
                    .collect();
                for ki in 0..seq {
                    let kbase = ((b * heads + h) * seq + ki) * hd;
                    for di in 0..hd {
                        dq[qbase + di] += ds_row[ki] * k[kbase + di].to_f32();
                    }
                }
            }
        }
    }
    dq
}

// === GPU launcher stubs (Task 14 scope; real launchers wire during PR-prep) ===

/// Stub for the B.1 forward FFI; returns (row_max, row_sum, O).
/// Real impl calls nsl_runtime's `nsl_flash_attention_csha_with_saves`.
fn run_b1_forward_for_test(
    _q: &[half::f16],
    _k: &[half::f16],
    _v: &[half::f16],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>, Vec<half::f16>) {
    let total_rows = batch * heads * seq;
    let _total_o = total_rows * hd;
    // Real impl: cudarc HBM alloc + nsl_flash_attention_csha_with_saves + dtoh copy.
    unimplemented!(
        "run_b1_forward_for_test stub — wire to nsl_flash_attention_csha_with_saves during GPU validation"
    );
    #[allow(unreachable_code)]
    (
        vec![0.0f32; total_rows],
        vec![1.0f32; total_rows],
        vec![half::f16::ZERO; _total_o],
    )
}

/// Stub for D pre-pass GPU launcher. Compiles + tests succeed only when wired.
fn run_d_prepass_on_gpu(
    _ptx: &str,
    _d_o: &[half::f16],
    _o: &[half::f16],
    batch: usize,
    heads: usize,
    seq: usize,
    _hd: usize,
) -> Vec<f32> {
    let total_rows = batch * heads * seq;
    let _ = total_rows;
    unimplemented!(
        "run_d_prepass_on_gpu stub — wire cudarc launcher during GPU validation"
    );
    #[allow(unreachable_code)]
    vec![0.0f32; batch * heads * seq]
}

/// Stub for dQ-kernel GPU launcher.
fn run_dq_kernel_on_gpu(
    _ptx: &str,
    _q: &[half::f16],
    _k: &[half::f16],
    _v: &[half::f16],
    _d_o: &[half::f16],
    _row_max: &[f32],
    _row_sum: &[f32],
    _d: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
) -> Vec<f32> {
    let _ = (batch, heads, seq, hd);
    unimplemented!(
        "run_dq_kernel_on_gpu stub — wire cudarc launcher during GPU validation"
    );
    #[allow(unreachable_code)]
    vec![0.0f32; batch * heads * seq * hd]
}
