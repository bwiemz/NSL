//! Layer-1 dQ-kernel parity tests vs CPU reference.
//!
//! All tests are `#[ignore]` + require `feature="cuda"`. Manual GPU invocation
//! required:
//!     cargo test -p nsl-codegen --test tier_b2_dq_kernel_cpu_reference \
//!         --features cuda -- --ignored --nocapture
//!
//! Test 1 (this task): D pre-pass standalone vs CPU rowsum.
//! Tests 2-3 (Task 14): dQ smoke + dQ head_dim sweep.
//!
//! GPU launcher helper `run_d_prepass_on_gpu` is a Task 14 deliverable; stubbed
//! here as `unimplemented!()`.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §6.1

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::d_prepass::synthesize_d_prepass;

/// Tiered tolerance inherited from V-B2-2 findings.
/// 5e-3 / 2e-2 / 4e-2 per head_dim regime (see Phase 1 spec §11.1).
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

#[test]
#[ignore]
fn tier_b2_d_prepass_vs_cpu_reduction() {
    // Test 1 of §6.1: D pre-pass standalone vs CPU rowsum(dO * O).
    // Validates the D pre-pass kernel in isolation, before dQ-kernel consumes D.
    // Tolerance: tol_for_head_dim(32) = 5e-3.

    let cfg = canonical_hd32_cfg();
    let batch = 1usize;
    let heads = 1usize;
    let seq = 32usize;
    let hd = cfg.head_dim as usize;

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
    let ptx = synthesize_d_prepass(&cfg).expect("D pre-pass synthesis");
    let d_gpu =
        run_d_prepass_on_gpu(&ptx, &d_o_host, &o_host, batch, heads, seq, hd);

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

/// GPU launcher — Task 14 will implement this. Allocates HBM via cudarc,
/// copies inputs, launches the emitted PTX, copies output back.
fn run_d_prepass_on_gpu(
    _ptx: &str,
    _d_o: &[half::f16],
    _o: &[half::f16],
    _batch: usize,
    _heads: usize,
    _seq: usize,
    _hd: usize,
) -> Vec<f32> {
    unimplemented!(
        "GPU launcher — Task 14 deliverable; this test is #[ignore]'d until then"
    )
}
