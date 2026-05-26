//! Layer-1 dQ-kernel parity tests vs CPU reference.
//!
//! All tests are `#[ignore]` + require `feature="cuda"`. Manual GPU invocation:
//!     cargo test -p nsl-codegen --test tier_b2_dq_kernel_cpu_reference \
//!         --features cuda -- --ignored --nocapture
//!
//! - Test 1: D pre-pass standalone vs CPU rowsum (Task 7) — **WIRED + launchable**
//! - Test 2: dQ smoke at canonical (Task 14) — CpuNaive source, seq=32
//! - Test 3a: `tier_b2_dq_sweep_cpu_naive_forward` — Phase 2.5 regression gate
//!   (CpuNaive forward, seq=64/128 multi-q-tile coverage)
//! - Test 3b: `tier_b2_dq_sweep_b1_forward` — Phase 2.6 closure gate
//!   (B.1 GPU forward + adapter, seq=32 single-block)
//!
//! Test 1 wires through `run_d_prepass_on_gpu` against the real `nsl_kernel_launch`
//! FFI; the D pre-pass kernel is data-mobile (real ld.global / fma / shfl /
//! st.global) and can be GPU-validated today.
//!
//! Tests 2 + 3 share `validate_dq_for_source`, which sources the forward outputs
//! from `nsl_test::diagnostic_mode::compute_forward_for_test` (CpuNaive via
//! `cpu_naive_forward`, or B1Forward via the GPU adapter `run_b1_forward_and_adapt`)
//! and compares the GPU dQ-kernel against `cpu_naive_backward_dq` (which recomputes
//! P/D from the forward's saved Q/K/V/O). `run_dq_kernel_on_gpu` is wired (H3, Phase
//! 2.5) to the real `nsl_kernel_launch` FFI — the dQ-kernel emitter is currently a
//! *structural scaffold* (sections, register decls, MMA chain shape, labels all
//! verified by ~20 ptxas/structural tests) but is **not yet fully data-mobile**;
//! T13 runs the first end-to-end B1Forward parity check on RTX 5070 Ti.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §6.1

#![cfg(feature = "cuda")]

use std::ffi::{c_void, CString};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::smem_layout::{
    tier_b2_dq_total_smem_bytes, tier_b2_effective_bq,
};
use nsl_codegen::flash_attention_v2::tier_b2::backward::d_prepass::synthesize_d_prepass;
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use nsl_test::cpu_naive_backward::cpu_naive_backward_dq;
use nsl_test::diagnostic_mode::{
    compute_forward_for_test, generate_d_o, generate_forward_inputs, FSource,
};

extern "C" {
    fn nsl_kernel_launch(
        ptx_ptr: i64,
        name_ptr: i64,
        grid_x: i64,
        grid_y: i64,
        grid_z: i64,
        block_x: i64,
        block_y: i64,
        block_z: i64,
        args_ptr: i64,
        num_args: i64,
        shared_mem_bytes: i64,
    ) -> i64;
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("[tier_b2] skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("[tier_b2] skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

/// Null-terminate a PTX string for cuModuleLoadData.
fn ptx_to_cstr_bytes(ptx: &str) -> Vec<u8> {
    let mut bytes = ptx.as_bytes().to_vec();
    if bytes.last() != Some(&b'\n') {
        bytes.push(b'\n');
    }
    bytes.push(0);
    bytes
}

fn tol_for_head_dim(hd: u32) -> f32 {
    if hd >= 128 {
        4e-2
    } else if hd >= 64 {
        2e-2
    } else {
        5e-3
    }
}

/// Tiered RELATIVE tolerance for the dQ parity gate (Phase 2.7 non-vacuous gate).
///
/// Scales with f16 MMA accumulation depth (~head_dim). Used as
/// `max_abs <= rel_tol * max|dq_ref|` so the gate is independent of the absolute
/// magnitude of dQ — a zero/degenerate output yields rel error ~= 1.0 and FAILS,
/// unlike the prior absolute-only gate which passed a zero-output kernel whenever
/// `max|dq_ref|` happened to fall below the absolute tolerance (the ×0.1-input
/// vacuity that hid the stats-stub bug; see project memory + dq.rs stats loads).
fn rel_tol_for_head_dim(hd: u32) -> f32 {
    if hd >= 128 {
        8e-2
    } else if hd >= 64 {
        5e-2
    } else {
        3e-2
    }
}

/// dO scale for the dQ parity gate. The shared `generate_d_o` emits ×0.1 values,
/// which makes the true dQ ~6e-5 — far below the absolute tolerance, rendering the
/// gate vacuous. dQ is linear in dO, so scaling dO lifts the dQ signal well above
/// the f16 noise floor, making the relative gate meaningful. Applied consistently
/// to the GPU D-pre-pass input, the GPU dQ-kernel input, and the CPU reference, so
/// D / dP / dS / dQ all scale together (self-consistent).
const DQ_GATE_DO_SCALE: f32 = 1024.0;

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
        // d_model=128 matches the T2.7 reference (tier_b1_save_activations_gpu)
        // single-block convention: chunk=d_model -> n_chunks=1. Leaving d_model=0
        // (bare ..Default::default()) defeats the B1Forward path's unwrap_or(128)
        // fallback (which only fires for csha=None) and zero-sizes every B.1 input.
        csha: Some(CshaExtras { level: 2, d_model: 128, ..Default::default() }),
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
        // d_model=128 matches the T2.7 reference (tier_b1_save_activations_gpu)
        // single-block convention: chunk=d_model -> n_chunks=1. Leaving d_model=0
        // (bare ..Default::default()) defeats the B1Forward path's unwrap_or(128)
        // fallback (which only fires for csha=None) and zero-sizes every B.1 input.
        csha: Some(CshaExtras { level: 2, d_model: 128, ..Default::default() }),
    }
}

// === Test 1: D pre-pass standalone (from Task 7) ===

#[test]
#[ignore]
fn tier_b2_d_prepass_vs_cpu_reduction() {
    // Test 1 of §6.1: D pre-pass standalone vs CPU rowsum(dO * O).
    // Validates the D pre-pass kernel in isolation, before dQ-kernel consumes D.
    // Tolerance: tol_for_head_dim(32) = 5e-3.

    if !cuda_available() {
        return;
    }

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

    eprintln!(
        "[tier_b2_d_prepass] batch={batch} heads={heads} seq={seq} hd={hd} \
         max_abs={max_abs:.6e} tol={tol:.6e}"
    );

    assert!(
        max_abs <= tol,
        "D pre-pass max_abs {} > tolerance {} for hd={}",
        max_abs,
        tol,
        hd
    );
}

// === Test 1b: D pre-pass at non-trivial grid + hd sweep ===

#[test]
#[ignore]
fn tier_b2_d_prepass_grid_dispatch_and_hd_sweep() {
    // Exercises the kernel's full grid (ceil(seq/32), heads, batch) and the
    // 32-row CTA strip handling for seq > 32. Sweeps hd ∈ {32, 64, 128} to
    // verify the per-row column loop scales correctly with head_dim. Reuses
    // the row-per-lane schedule; numerical match should remain near machine
    // epsilon for these magnitudes.

    if !cuda_available() {
        return;
    }

    struct Case {
        batch: usize,
        heads: usize,
        seq: usize,
        hd: i64,
    }
    let cases = [
        Case { batch: 1, heads: 1, seq: 64,  hd: 32  }, // 2 CTAs in x
        Case { batch: 1, heads: 2, seq: 96,  hd: 64  }, // 3 CTAs in x, 2 in y
        Case { batch: 2, heads: 1, seq: 128, hd: 128 }, // 4 CTAs in x, 2 in z
    ];

    for case in cases.iter() {
        let cfg_c = FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: case.hd,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
        };
        let hd = case.hd as usize;

        let d_o_host: Vec<half::f16> = (0..case.batch * case.heads * case.seq * hd)
            .map(|i| half::f16::from_f32(((i as f32 * 0.137).sin() * 0.1) as f32))
            .collect();
        let o_host: Vec<half::f16> = (0..case.batch * case.heads * case.seq * hd)
            .map(|i| half::f16::from_f32(((i as f32 * 0.241).cos() * 0.1) as f32))
            .collect();

        let mut d_ref = vec![0.0f32; case.batch * case.heads * case.seq];
        for b in 0..case.batch {
            for h in 0..case.heads {
                for q in 0..case.seq {
                    let base = ((b * case.heads + h) * case.seq + q) * hd;
                    let sum: f32 = (0..hd)
                        .map(|d| d_o_host[base + d].to_f32() * o_host[base + d].to_f32())
                        .sum();
                    d_ref[(b * case.heads + h) * case.seq + q] = sum;
                }
            }
        }

        let ptx = synthesize_d_prepass(&cfg_c).expect("D pre-pass synthesis");
        let d_gpu = run_d_prepass_on_gpu(
            &ptx, &d_o_host, &o_host, case.batch, case.heads, case.seq, hd,
        );

        let tol = tol_for_head_dim(hd as u32);
        let max_abs = d_ref
            .iter()
            .zip(d_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        eprintln!(
            "[tier_b2_d_prepass_sweep] batch={} heads={} seq={} hd={} \
             max_abs={:.6e} tol={:.6e}",
            case.batch, case.heads, case.seq, case.hd, max_abs, tol
        );

        assert!(
            max_abs <= tol,
            "D pre-pass sweep max_abs {} > tol {} (batch={}, heads={}, seq={}, hd={})",
            max_abs, tol, case.batch, case.heads, case.seq, case.hd,
        );
    }
}

// === Test 2: dQ-kernel smoke at canonical (Task 14) ===

#[test]
#[ignore]
fn tier_b2_dq_smoke_canonical() {
    // (bq=bkv=32, hd=32, non-causal, no RoPE) via the CpuNaive forward source.
    // Tolerance: tol_for_head_dim(32) = 5e-3.
    validate_dq_for_source(&cfg(32, 32), FSource::CpuNaive, 32);
}

// === Test 3: dQ-kernel sweeps — CpuNaive regression + B1Forward closure ===

#[test]
#[ignore]
fn tier_b2_dq_sweep_cpu_naive_forward() {
    // Phase 2.5 regression gate (refactored from tier_b2_dq_head_dim_sweep).
    // Preserves the original seq schedule (64/128) for multi-q-tile coverage.
    // Path-A schedule per spec §5.2: hd=128 uses bq=32 (SMEM-pressure fallback);
    // hd=32 and hd=64 use bq=64 (no fallback needed).
    for &hd in &[32i64, 64, 128] {
        let bq = if hd == 128 { 32 } else { 64 };
        let seq = if hd >= 128 { 128 } else { 64 };
        validate_dq_for_source(&cfg(bq, hd), FSource::CpuNaive, seq);
    }
}

#[test]
#[ignore]
fn tier_b2_dq_sweep_b1_forward() {
    // Phase 2.6 closure gate -- B.1 forward + adapter -> dQ-kernel end-to-end.
    // seq=32 (B.1 single-block precondition: launcher forces block_kv=32).
    //
    // INVOCATION: requires nsl-test's cuda feature (a dev-dependency), so run with
    //   cargo test -p nsl-codegen --features "cuda,nsl-test/cuda" \
    //       --test tier_b2_dq_kernel_cpu_reference tier_b2_dq_sweep_b1_forward \
    //       -- --ignored --nocapture
    // Plain --features cuda does NOT forward to nsl-test/cuda, so the B1Forward
    // path would hit the "requires feature='cuda'" panic.
    //
    // Full hd sweep restored: the B.1-forward hd>bkv softmax-stat-register bug
    // (tiles_per_warp_qkt vs tiles_per_warp_pv divergence) is FIXED -- the stats
    // are now re-keyed by absolute query row via a dedicated SMEM region
    // (commit e854bad8), GPU-validated at hd=32/64/128 in
    // tier_b1_save_activations_gpu.
    //
    // bq = bkv = 32 for ALL hd here: the B.1 single-block forward processes seq=32
    // as ONE block of block_kv=32, so the dQ kernel must use a matching bkv (a full
    // tile, bkv == seq). The dQ kernel has NO seq-boundary masking yet, so bkv > seq
    // (e.g. the old bq=64 with seq=32) sums OOB kv positions into dQ and yields
    // garbage -- the partial-tile / arbitrary-seq case is a documented Phase-4
    // follow-on (see dq.rs module doc). A realistic planner never picks bkv > seq.
    for &hd in &[32i64, 64, 128] {
        validate_dq_for_source(&cfg(32, hd), FSource::B1Forward, 32);
    }
}

// === Shared parity helper ===

/// Run the dQ-kernel parity check for one forward source at a given (cfg, seq).
/// Shared body for the CpuNaive regression gate + the B1Forward closure gate.
///
/// The comparator (`cpu_naive_backward_dq`) reads Q/K/V/O from `fwd` (the forward
/// outputs) -- NOT the raw test inputs -- so adapter stats-reshape + D-pre-pass
/// bugs are caught. D is sourced from the GPU D pre-pass; dQ from the GPU dQ-kernel.
fn validate_dq_for_source(cfg: &FlashAttentionConfig, source: FSource, seq: usize) {
    if !cuda_available() {
        return;
    }
    let batch = 1usize;
    let heads = 1usize;
    let hd = cfg.head_dim as usize;

    let inputs = generate_forward_inputs(cfg, source, seq);
    // Scale dO so the true dQ is substantial (Phase 2.7 non-vacuous gate). dQ is
    // linear in dO; the shared ×0.1 generator otherwise leaves dq_ref ~6e-5, below
    // tolerance. Used consistently below for the GPU D-pre-pass, the GPU dQ-kernel,
    // and the CPU reference, so D/dP/dS/dQ scale together.
    let d_o: Vec<half::f16> = generate_d_o(cfg, seq)
        .iter()
        .map(|x| half::f16::from_f32(x.to_f32() * DQ_GATE_DO_SCALE))
        .collect();

    // Comparator reads Q/K/V/O from fwd (Phase 2.6 comparator fix), NOT raw inputs.
    let fwd = compute_forward_for_test(&inputs, cfg, source, seq);

    let d_prepass_ptx = synthesize_d_prepass(cfg).expect("d_prepass synth");
    let d = run_d_prepass_on_gpu(&d_prepass_ptx, &d_o, &fwd.o, batch, heads, seq, hd);

    let dq_ptx = synthesize_dq_kernel(cfg).expect("dq_kernel synth");
    let dq_gpu = run_dq_kernel_on_gpu(
        &dq_ptx, cfg, &fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &d_o, &fwd.row_max,
        &fwd.row_sum, &d, batch, heads, seq, hd,
    );

    let dq_ref = cpu_naive_backward_dq(
        &fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, batch, heads, seq, cfg,
    );

    let max_abs = dq_gpu
        .iter()
        .zip(dq_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_ref = dq_ref.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let max_gpu = dq_gpu.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
    let rel_tol = rel_tol_for_head_dim(hd as u32);
    eprintln!(
        "[validate_dq FSource={:?}] hd={} bq={} seq={} max|dq_gpu|={:.6e} max|dq_ref|={:.6e} \
         max_abs={:.6e} rel={:.4e} rel_tol={:.4e}",
        source, hd, cfg.block_q, seq, max_gpu, max_ref, max_abs, rel, rel_tol
    );
    // Non-vacuous gate (Phase 2.7). The prior gate (`max_abs <= absolute_tol`)
    // passed a ZERO-output kernel whenever max|dq_ref| fell below tol — which it
    // did under the ×0.1 inputs, hiding the dq.rs stats-stub bug (dQ == 0).
    // (1) reference-magnitude floor: ensure the test actually exercises a
    //     substantial dQ (else the gate is meaningless).
    assert!(
        max_ref > 1e-3,
        "FSource={:?} hd={}: reference |dQ|={:.3e} too small for a meaningful gate \
         (raise DQ_GATE_DO_SCALE) — vacuous-gate guard",
        source, hd, max_ref
    );
    // (2) zero-output guard: a kernel that emits ~0 (e.g. stub softmax stats ->
    //     P=0 -> dQ=0) must FAIL, regardless of tolerance.
    assert!(
        max_gpu >= 0.25 * max_ref,
        "FSource={:?} hd={} seq={}: dQ-kernel output is near-zero \
         (max|dq_gpu|={:.3e} << max|dq_ref|={:.3e}) — kernel not computing dQ \
         (stub stats?). ZERO-OUTPUT GUARD.",
        source, hd, seq, max_gpu, max_ref
    );
    // (3) relative-error gate: catches magnitude/shape errors the absolute gate
    //     misses at small dq_ref.
    assert!(
        rel <= rel_tol,
        "FSource={:?} hd={} seq={}: relative error {:.4e} > rel_tol {:.4e} \
         (max_abs={:.3e}, max|dq_ref|={:.3e})",
        source, hd, seq, rel, rel_tol, max_abs, max_ref
    );
}

// === GPU launchers ===
//
// `run_d_prepass_on_gpu` is wired to the real `nsl_kernel_launch` FFI — the
// D pre-pass kernel is data-mobile and validates against the CPU reference
// today.
//
// `run_dq_kernel_on_gpu` is wired (H3, Phase 2.5) against the real
// `nsl_kernel_launch` FFI with the dQ-kernel's 12-param entry signature.
// The dQ-kernel emitter is currently a structural scaffold (sections + register
// decls + MMA chain + labels) with cp.async loads, HBM address derivation, dS
// SMEM scatter, col-major K re-stage, tile_skip predicate, MMA fragment row/col
// setup, and loop back-edges shipped as PTX comments rather than emitted
// instructions. H4 runs the first GPU parity check on the wired scaffold.

/// Phase-2.5 H3: dQ-kernel GPU launcher wired against the real `nsl_kernel_launch` FFI.
///
/// Mirrors `run_d_prepass_on_gpu` but extended for the dQ-kernel's 12-param entry
/// signature (Q, K, V, dO, row_max, row_sum, D, segment_ids, dQ_out, seq, heads, batch).
/// Takes `cfg` so it can derive effective_bq (Path-A schedule) and total SMEM bytes.
///
/// HBM layout:
///   - Q/K/V/dO: [batch, heads, seq, hd] f16 → 2 bytes/element
///   - row_max/row_sum/D: [batch, heads, seq] f32 → 4 bytes/element
///   - dQ_out: [batch, heads, seq, hd] f32 → 4 bytes/element
///   - segment_ids: null (0) — no segment masking in Phase 2.5
///
/// Grid: (ceil(seq / effective_bq), heads, batch). Block: (128, 1, 1) (4 warps).
/// Dynamic SMEM: tier_b2_dq_total_smem_bytes(cfg).
fn run_dq_kernel_on_gpu(
    ptx: &str,
    cfg: &FlashAttentionConfig,
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
    let in_4d_bytes = (batch * heads * seq * hd * 2) as i64; // f16
    let in_3d_bytes = (batch * heads * seq * 4) as i64;       // f32
    let out_bytes   = (batch * heads * seq * hd * 4) as i64;  // f32

    let q_dev    = nsl_test_cuda_alloc(in_4d_bytes);
    let k_dev    = nsl_test_cuda_alloc(in_4d_bytes);
    let v_dev    = nsl_test_cuda_alloc(in_4d_bytes);
    let do_dev   = nsl_test_cuda_alloc(in_4d_bytes);
    let rmax_dev = nsl_test_cuda_alloc(in_3d_bytes);
    let rsum_dev = nsl_test_cuda_alloc(in_3d_bytes);
    let d_dev    = nsl_test_cuda_alloc(in_3d_bytes);
    let dq_dev   = nsl_test_cuda_alloc(out_bytes);
    assert!(
        q_dev != 0 && k_dev != 0 && v_dev != 0 && do_dev != 0
            && rmax_dev != 0 && rsum_dev != 0 && d_dev != 0 && dq_dev != 0,
        "dQ-kernel device alloc failed"
    );

    nsl_test_cuda_h2d(q_dev,    q.as_ptr()       as i64, in_4d_bytes);
    nsl_test_cuda_h2d(k_dev,    k.as_ptr()       as i64, in_4d_bytes);
    nsl_test_cuda_h2d(v_dev,    v.as_ptr()       as i64, in_4d_bytes);
    nsl_test_cuda_h2d(do_dev,   d_o.as_ptr()     as i64, in_4d_bytes);
    nsl_test_cuda_h2d(rmax_dev, row_max.as_ptr() as i64, in_3d_bytes);
    nsl_test_cuda_h2d(rsum_dev, row_sum.as_ptr() as i64, in_3d_bytes);
    nsl_test_cuda_h2d(d_dev,    d.as_ptr()       as i64, in_3d_bytes);

    let ptx_bytes   = ptx_to_cstr_bytes(ptx);
    let kernel_name = CString::new("tier_b2_dq_kernel").unwrap();

    let mut q_p    = q_dev    as u64;
    let mut k_p    = k_dev    as u64;
    let mut v_p    = v_dev    as u64;
    let mut do_p   = do_dev   as u64;
    let mut rmax_p = rmax_dev as u64;
    let mut rsum_p = rsum_dev as u64;
    let mut d_p    = d_dev    as u64;
    let mut seg_p  = 0u64; // null — no segment masking in Phase 2.5
    let mut dq_p   = dq_dev   as u64;
    let mut seq_u32   = seq   as u32;
    let mut heads_u32 = heads as u32;
    let mut batch_u32 = batch as u32;

    let args: [*mut c_void; 12] = [
        &mut q_p    as *mut _ as *mut c_void,
        &mut k_p    as *mut _ as *mut c_void,
        &mut v_p    as *mut _ as *mut c_void,
        &mut do_p   as *mut _ as *mut c_void,
        &mut rmax_p as *mut _ as *mut c_void,
        &mut rsum_p as *mut _ as *mut c_void,
        &mut d_p    as *mut _ as *mut c_void,
        &mut seg_p  as *mut _ as *mut c_void,
        &mut dq_p   as *mut _ as *mut c_void,
        &mut seq_u32   as *mut _ as *mut c_void,
        &mut heads_u32 as *mut _ as *mut c_void,
        &mut batch_u32 as *mut _ as *mut c_void,
    ];

    let bq         = tier_b2_effective_bq(cfg) as i64;
    let grid_x     = ((seq as i64) + bq - 1) / bq;
    let smem_bytes = tier_b2_dq_total_smem_bytes(cfg) as i64;

    let rc = unsafe {
        nsl_kernel_launch(
            ptx_bytes.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            grid_x,
            heads as i64,
            batch as i64,
            128, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            smem_bytes,
        )
    };

    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx_bytes.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no JIT log>".into()
        };
        nsl_test_cuda_free(q_dev);
        nsl_test_cuda_free(k_dev);
        nsl_test_cuda_free(v_dev);
        nsl_test_cuda_free(do_dev);
        nsl_test_cuda_free(rmax_dev);
        nsl_test_cuda_free(rsum_dev);
        nsl_test_cuda_free(d_dev);
        nsl_test_cuda_free(dq_dev);
        panic!("dQ-kernel launch failed rc={rc}\nJIT log:\n{log}");
    }

    let mut dq_host = vec![0.0f32; batch * heads * seq * hd];
    nsl_test_cuda_d2h(dq_host.as_mut_ptr() as i64, dq_dev, out_bytes);
    nsl_test_cuda_free(q_dev);
    nsl_test_cuda_free(k_dev);
    nsl_test_cuda_free(v_dev);
    nsl_test_cuda_free(do_dev);
    nsl_test_cuda_free(rmax_dev);
    nsl_test_cuda_free(rsum_dev);
    nsl_test_cuda_free(d_dev);
    nsl_test_cuda_free(dq_dev);
    dq_host
}

/// D pre-pass GPU launcher.
///
/// Allocates HBM buffers for dO + O + D_out, copies dO/O host→device, calls
/// `nsl_kernel_launch` with grid (ceil(seq/32), heads, batch) and block (32,1,1),
/// copies D back, then frees. Returns the f32 D vector laid out [B, H, S].
fn run_d_prepass_on_gpu(
    ptx: &str,
    d_o: &[half::f16],
    o: &[half::f16],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
) -> Vec<f32> {
    assert_eq!(d_o.len(), batch * heads * seq * hd, "d_o size mismatch");
    assert_eq!(o.len(), batch * heads * seq * hd, "o size mismatch");

    let in_bytes = (batch * heads * seq * hd * 2) as i64;
    let out_bytes = (batch * heads * seq * 4) as i64;

    let d_o_dev = nsl_test_cuda_alloc(in_bytes);
    let o_dev = nsl_test_cuda_alloc(in_bytes);
    let d_out_dev = nsl_test_cuda_alloc(out_bytes);
    assert!(
        d_o_dev != 0 && o_dev != 0 && d_out_dev != 0,
        "D pre-pass device alloc failed"
    );

    nsl_test_cuda_h2d(d_o_dev, d_o.as_ptr() as i64, in_bytes);
    nsl_test_cuda_h2d(o_dev, o.as_ptr() as i64, in_bytes);

    let ptx_bytes = ptx_to_cstr_bytes(ptx);
    let kernel_name = CString::new("tier_b2_d_prepass").unwrap();

    let mut d_o_ptr_u64 = d_o_dev as u64;
    let mut o_ptr_u64 = o_dev as u64;
    let mut d_out_ptr_u64 = d_out_dev as u64;
    let mut seq_len_u32 = seq as u32;
    let mut heads_u32 = heads as u32;
    let args: [*mut c_void; 5] = [
        &mut d_o_ptr_u64 as *mut _ as *mut c_void,
        &mut o_ptr_u64 as *mut _ as *mut c_void,
        &mut d_out_ptr_u64 as *mut _ as *mut c_void,
        &mut seq_len_u32 as *mut _ as *mut c_void,
        &mut heads_u32 as *mut _ as *mut c_void,
    ];

    let grid_x = ((seq as i64) + 31) / 32;
    let rc = unsafe {
        nsl_kernel_launch(
            ptx_bytes.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            grid_x,
            heads as i64,
            batch as i64,
            32, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            0, // no dynamic SMEM
        )
    };

    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx_bytes.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no JIT log>".into()
        };
        nsl_test_cuda_free(d_o_dev);
        nsl_test_cuda_free(o_dev);
        nsl_test_cuda_free(d_out_dev);
        panic!("D pre-pass launch failed rc={}\nJIT log:\n{}", rc, log);
    }

    let mut d_host = vec![0.0f32; batch * heads * seq];
    nsl_test_cuda_d2h(d_host.as_mut_ptr() as i64, d_out_dev, out_bytes);
    nsl_test_cuda_free(d_o_dev);
    nsl_test_cuda_free(o_dev);
    nsl_test_cuda_free(d_out_dev);
    d_host
}
