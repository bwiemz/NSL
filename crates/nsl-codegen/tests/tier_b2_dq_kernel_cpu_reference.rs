//! Layer-1 dQ-kernel parity tests vs CPU reference.
//!
//! All tests are `#[ignore]` + require `feature="cuda"`. Manual GPU invocation:
//!     cargo test -p nsl-codegen --test tier_b2_dq_kernel_cpu_reference \
//!         --features cuda -- --ignored --nocapture
//!
//! - Test 1: D pre-pass standalone vs CPU rowsum (Task 7) — **WIRED + launchable**
//! - Test 2: dQ smoke at canonical (Task 14) — **gated on Phase 2.5 H3+H4**
//! - Test 3: dQ head_dim sweep (Task 14) — **gated on Phase 2.5 H3+H4**
//!
//! Test 1 wires through `run_d_prepass_on_gpu` against the real `nsl_kernel_launch`
//! FFI; the D pre-pass kernel is data-mobile (real ld.global / fma / shfl /
//! st.global) and can be GPU-validated today.
//!
//! Tests 2 + 3: `run_b1_forward_for_test` is now wired (H2, Phase 2.5) to the
//! CPU-naive forward (`nsl_test::cpu_naive_forward`), providing row_max / row_sum /
//! O as primary reference. `run_dq_kernel_on_gpu` is now wired (H3, Phase 2.5) to
//! the real `nsl_kernel_launch` FFI — the dQ-kernel emitter is currently a *structural
//! scaffold* (sections, register decls, MMA chain shape, labels all verified by ~20
//! ptxas/structural tests) but is **not data-mobile** — cp.async loads, HBM address
//! derivation, dS SMEM scatter, col-major K re-stage, tile_skip predicate loads, MMA
//! fragment row/col setup, and loop back-edges are placeholder comments. H4 runs the
//! first GPU parity check to determine numerical output from the scaffold.
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
    let rc = unsafe { nsl_cuda_init() };
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
    // (bq=bkv=32, hd=32, non-causal, no RoPE). Tolerance: tol_for_head_dim(32) = 5e-3.
    let cfg_c = cfg(32, 32);
    let max_abs = run_dq_kernel_parity(&cfg_c, /*batch=*/ 1, /*heads=*/ 1, /*seq=*/ 32);
    let tol = tol_for_head_dim(cfg_c.head_dim as u32);
    eprintln!(
        "[tier_b2_dq_smoke_canonical] bq=32 hd=32 seq=32 batch=1 heads=1 \
         max_abs={:.6e} tol={:.6e}",
        max_abs, tol
    );
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
        // Path-A schedule per spec §5.2: hd=128 uses bq=32 (SMEM-pressure fallback);
        // hd=32 and hd=64 use bq=64 (no fallback needed).
        // Old pre-Path-A code was `if hd >= 128 { 64 } else { 32 }` which hit the
        // 99 KB SMEM cap at hd=128 — the exact failure mode Path-A's fallback avoids.
        let bq = if hd == 128 { 32 } else { 64 };
        let cfg_c = cfg(bq, hd);
        let seq = if hd >= 128 { 128 } else { 64 };
        let max_abs = run_dq_kernel_parity(&cfg_c, 1, 1, seq);
        let tol = tol_for_head_dim(hd as u32);
        eprintln!(
            "[tier_b2_dq_head_dim_sweep] hd={} bq={} seq={} max_abs={:.6e} tol={:.6e}",
            hd, bq, seq, max_abs, tol
        );
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
/// Calls into the wired GPU launcher (H3, Phase 2.5); H4 will run the first real
/// GPU parity check to determine whether the structural scaffold produces correct
/// numerical output.
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
        &dq_ptx, cfg, &q, &k, &v, &d_o, &row_max, &row_sum, &d, batch, heads, seq, hd,
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

// === GPU launchers ===
//
// `run_d_prepass_on_gpu` is wired to the real `nsl_kernel_launch` FFI — the
// D pre-pass kernel is data-mobile and validates against the CPU reference
// today.
//
// `run_b1_forward_for_test` is wired (H2, Phase 2.5) to call the CPU-naive
// forward (H1) from `nsl_test::cpu_naive_forward`, producing row_max / row_sum
// / O as primary reference inputs for the dQ-kernel. This severs the Phase 2.6
// B.1-GPU-integration dependency: tests can validate the dQ-kernel's output
// against H1 outputs without requiring the full B.1 forward FFI stack. The
// CPU-naive forward also serves as the diagnostic-mode swap target in Phase 2.6.
//
// `run_dq_kernel_on_gpu` is now wired (H3, Phase 2.5) against the real
// `nsl_kernel_launch` FFI with the dQ-kernel's 12-param entry signature.
// The dQ-kernel emitter is currently a structural scaffold (sections + register
// decls + MMA chain + labels) with cp.async loads, HBM address derivation, dS
// SMEM scatter, col-major K re-stage, tile_skip predicate, MMA fragment row/col
// setup, and loop back-edges shipped as PTX comments rather than emitted
// instructions. H4 runs the first GPU parity check on the wired scaffold.

/// Phase-2.5 H2: primary-reference forward producing (row_max, row_sum, O) via
/// the CPU-naive forward (`nsl_test::cpu_naive_forward::cpu_naive_forward`).
///
/// This wiring severs the Phase 2.6 B.1-GPU-integration dependency: the dQ-kernel
/// tests can validate against H1 CPU-reference outputs without requiring the full
/// `nsl_flash_attention_csha_with_saves` FFI stack. Phase 2.6 will replace this with
/// B.1's actual GPU forward + adapter; the CPU-naive forward persists as the
/// diagnostic-mode swap target per spec §2.2 + §5.3.
///
/// Tests are non-causal at the dQ-kernel level — causal masking is handled by the
/// dQ-kernel's `tile_skip` predicate per spec §9.2.
fn run_b1_forward_for_test(
    q: &[half::f16],
    k: &[half::f16],
    v: &[half::f16],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
) -> (Vec<f32>, Vec<f32>, Vec<half::f16>) {
    // Phase 2.5: standalone validation uses CPU-naive forward (H1) to produce
    // row_max / row_sum / O, severing the Phase 2.6 B.1-integration dependency.
    // Tests are non-causal at the dQ-kernel level (causal masking is handled by
    // the dQ-kernel's tile_skip predicate per spec §9.2).
    let out = nsl_test::cpu_naive_forward::cpu_naive_forward(
        q, k, v, batch, heads, seq, hd, /*causal=*/false,
    );
    (out.row_max, out.row_sum, out.o)
}

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
