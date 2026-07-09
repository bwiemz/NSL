//! First GPU numeric parity gate for the PLAIN (non-CSHA) flash-attention
//! backward — the SDPA path real GQA pretraining actually exercises.
//!
//! # Why this file exists
//!
//! Real NSL transformers (stdlib `nn/gqa.nsl`, coder-rl/50m/7b, spec GPT2) use
//! standard multi-head attention: RMSNorm + Q/K/V projections lower as separate
//! generic ops, and only the SDPA core (`scaled_dot_product_attention`) becomes a
//! flash-attention kernel. `gqa.nsl` expands KV heads to the full head count
//! (`k5.expand(...).reshape([batch, nh, seq, hd])`) *before* the SDPA call, so at
//! the op the KV head count equals the Q head count (`kv_h == h`). The backward
//! FFI `nsl_flash_attention_backward` therefore routes to the GPU kernel
//! `flash_attention_backward_gpu` (its dispatch gate is `device>0 &&
//! phase1_ptx_ptr!=0 && kv_h==h`), which on sm_80+ (the RTX 5070 Ti is sm_120)
//! takes the MMA q-tile-loop path.
//!
//! That GPU path is *structurally* built for heads>1 (grid_x = batch*heads) and
//! multi-tile (grid_y = ceil(seq/block), q-tile loop with SMEM dK/dV accumulation
//! + atomic dQ), with no refusal — but it had **zero GPU numeric parity tests**:
//! every prior correctness test was CPU-only and single-tile (seq <= 16 <<
//! block_q=64). This file closes that gap: it validates the GPU backward against
//! the naive untiled CPU reference (`flash_attention_backward_cpu`, inherently
//! multi-tile-correct) at heads>1 AND seq>block_q.
//!
//! # Harness
//!
//! Deterministic q/k/v/dO -> naive CPU forward (out + logsumexp) -> CPU-reference
//! backward (oracle) and GPU FFI backward (system under test) consuming the same
//! `out`; the GPU auto-recomputes logsumexp from q/k (standard logsumexp, matches
//! the oracle's within tolerance). Compare dQ/dK/dV.
//!
//! block_q = block_kv = 64 are hardcoded by the runtime launcher, so "multi-tile"
//! means seq_len > 64; these tests use seq_len = 128 (2 tiles).
//!
//! # Running
//!
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test flash_attention_backward_gpu \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::flash_attention::{
    backward_shared_mem_bytes, flash_attention_bwd_d_kernel_name,
    flash_attention_bwd_main_kernel_name, synthesize_flash_attention_backward_ptx,
    FlashAttentionBackwardConfig,
};

use nsl_runtime::flash_attention::{flash_attention_backward_cpu, nsl_flash_attention_backward};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_zeros_on};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d, nsl_test_cuda_jit_log};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    nsl_cuda_init() == 0
}

/// Deterministic pseudo-random values in roughly [-0.5, 0.5] (LCG), matching the
/// generator used by the CSHA GPU tests so tolerances are comparable.
fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 65535.0) - 0.5
        })
        .collect()
}

/// Naive standard-attention forward. Returns (out[b*h*s*d], logsumexp[b*h*s]).
/// out[i] = sum_j softmax_j(scale * Q_i . K_j) * V_j ; lse[i] = logsumexp_j(scale * Q_i . K_j).
fn naive_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let bh_stride = h * s * d;
    let h_stride = s * d;
    let row_stride = d;
    let lse_bh_stride = h * s;
    let lse_h_stride = s;

    let mut out = vec![0.0f32; b * h * s * d];
    let mut lse = vec![0.0f32; b * h * s];

    for bi in 0..b {
        for hi in 0..h {
            let base = bi * bh_stride + hi * h_stride;
            let lse_base = bi * lse_bh_stride + hi * lse_h_stride;
            for i in 0..s {
                let q_row = base + i * row_stride;
                let j_max = if causal { i + 1 } else { s };

                // scores
                let mut scores = vec![0.0f32; j_max];
                let mut m = f32::NEG_INFINITY;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let k_row = base + j * row_stride;
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += q[q_row + dd] * k[k_row + dd];
                    }
                    let val = dot * scale;
                    *sc = val;
                    if val > m {
                        m = val;
                    }
                }
                // softmax denom + lse
                let mut denom = 0.0f32;
                for &sc in scores.iter() {
                    denom += (sc - m).exp();
                }
                lse[lse_base + i] = m + denom.ln();

                // weighted sum of V
                for (j, &sc) in scores.iter().enumerate() {
                    let p = (sc - m).exp() / denom;
                    let v_row = base + j * row_stride;
                    for dd in 0..d {
                        out[q_row + dd] += p * v[v_row + dd];
                    }
                }
            }
        }
    }
    (out, lse)
}

/// Create a GPU f32 tensor handle [shape] filled with `vals` (via managed alloc +
/// stream-ordered H2D copy).
fn gpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    let dptr = nsl_tensor_data_ptr(t);
    nsl_test_cuda_h2d(dptr, vals.as_ptr() as i64, (vals.len() * 4) as i64);
    t
}

/// Read `len` f32 elements back from a GPU tensor handle.
fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let dptr = nsl_tensor_data_ptr(t);
    let mut out = vec![0.0f32; len];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dptr, (len * 4) as i64);
    out
}

fn read_jit_log(ptx_ptr: i64) -> String {
    let log_ptr = nsl_test_cuda_jit_log(ptx_ptr);
    if log_ptr != 0 {
        unsafe {
            std::ffi::CStr::from_ptr(log_ptr as *const i8)
                .to_string_lossy()
                .into_owned()
        }
    } else {
        "<no log / JIT ok>".into()
    }
}

/// Diagnostic (not a gate): synthesize the backward PTX, report the declared
/// shared-memory size, and JIT-load each phase to surface the exact ptxas error.
#[test]
#[ignore = "diagnostic: prints backward PTX JIT log"]
fn diag_phase2_jit_log() {
    if !cuda_available() {
        return;
    }
    for (label, gpu_sm, head_dim) in [
        ("scalar_hd16", 75u32, 16i64),
        ("scalar_hd32", 75, 32),
        ("mma_hd32", 80, 32),
    ] {
        let cfg = FlashAttentionBackwardConfig {
            block_q: 64,
            block_kv: 64,
            head_dim,
            causal: false,
            gpu_sm,
            segment_masked: false,
        };
        let shmem = backward_shared_mem_bytes(&cfg);
        eprintln!(
            "[diag {label}] backward_shared_mem_bytes = {shmem} bytes ({:.1} KB)",
            shmem as f32 / 1024.0
        );
        let (ptx1, ptx2) = synthesize_flash_attention_backward_ptx(&cfg);
        let dir = std::env::temp_dir();
        std::fs::write(dir.join(format!("fa_bwd_phase2_{label}.ptx")), &ptx2).ok();
        eprintln!("[diag {label}] phase1 JIT: {}", read_jit_log(ptx1.as_ptr() as i64));
        eprintln!("[diag {label}] phase2 JIT: {}", read_jit_log(ptx2.as_ptr() as i64));
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn all_finite(a: &[f32]) -> bool {
    a.iter().all(|x| x.is_finite())
}

struct Grads {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

/// Run the plain flash-attention backward on GPU and via the CPU reference for
/// the same inputs. Returns (gpu, cpu) gradient triples. `gpu_sm` selects the
/// scalar (sm<80) vs MMA (sm>=80) backward code path.
fn run(
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    causal: bool,
    gpu_sm: u32,
) -> (Grads, Grads) {
    let scale = 1.0f32 / (d as f32).sqrt();
    let total = b * h * s * d;

    let q = det_seq(1, total);
    let k = det_seq(2, total);
    let v = det_seq(3, total);
    let dout = det_seq(4, total);

    let (out, lse) = naive_forward(&q, &k, &v, b, h, s, d, scale, causal);

    // ── CPU reference (oracle) ──
    let mut dq_c = vec![0.0f32; total];
    let mut dk_c = vec![0.0f32; total];
    let mut dv_c = vec![0.0f32; total];
    flash_attention_backward_cpu(
        &q, &k, &v, &out, &lse, &dout, &mut dq_c, &mut dk_c, &mut dv_c, b, h, s, d, scale, causal,
    );

    // ── GPU (system under test) ──
    let cfg = FlashAttentionBackwardConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: d as i64,
        causal,
        gpu_sm,
        segment_masked: false,
    };
    let (ptx1, ptx2) = synthesize_flash_attention_backward_ptx(&cfg);
    let name1 = CString::new(flash_attention_bwd_d_kernel_name(&cfg)).unwrap();
    let name2 = CString::new(flash_attention_bwd_main_kernel_name(&cfg)).unwrap();

    let shape = [b as i64, h as i64, s as i64, d as i64];
    let q_t = gpu_tensor(&q, &shape);
    let k_t = gpu_tensor(&k, &shape);
    let v_t = gpu_tensor(&v, &shape);
    let out_t = gpu_tensor(&out, &shape);
    let dout_t = gpu_tensor(&dout, &shape);

    let grads_list = nsl_flash_attention_backward(
        dout_t,
        q_t,
        k_t,
        v_t,
        out_t,
        0, // logsumexp_ptr = 0 -> runtime auto-computes
        scale.to_bits() as i64,
        b as i64,
        h as i64,
        s as i64,
        d as i64,
        causal as i64,
        ptx1.as_ptr() as i64,
        name1.as_ptr() as i64,
        ptx2.as_ptr() as i64,
        name2.as_ptr() as i64,
        0, // tier_b_ptx_ptr
        0, // tier_b_name_ptr
    );

    assert!(grads_list != 0, "backward FFI returned null list");
    let dq_g = read_gpu(nsl_list_get(grads_list, 0), total);
    let dk_g = read_gpu(nsl_list_get(grads_list, 1), total);
    let dv_g = read_gpu(nsl_list_get(grads_list, 2), total);

    // Keep PTX/name buffers alive until here (they back raw pointers passed above).
    let _ = (&ptx1, &ptx2, &name1, &name2);

    (
        Grads { dq: dq_g, dk: dk_g, dv: dv_g },
        Grads { dq: dq_c, dk: dk_c, dv: dv_c },
    )
}

/// Tolerance per head_dim: MMA uses tf32 for the matmuls, so error scales with
/// head_dim (mirrors the forward CSHA tiers, slightly looser for the backward's
/// extra accumulation).
fn tol_for(head_dim: usize) -> f32 {
    match head_dim {
        hd if hd >= 128 => 5e-2,
        hd if hd >= 64 => 3e-2,
        _ => 1.5e-2,
    }
}

fn max_abs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

/// Assert GPU/CPU parity for all three gradients:
///   1. GPU output is finite,
///   2. GPU output is non-trivially non-zero relative to the reference (guards
///      against the vacuous pass where a swallowed kernel-launch failure returns
///      the zero-initialized buffers — small grads can otherwise sit under `tol`
///      vs all-zeros), and
///   3. max_abs difference is within `tol`.
fn assert_parity(tag: &str, gpu: &Grads, cpu: &Grads, tol: f32) {
    let mut failures = Vec::new();
    for (name, g, c) in [
        ("dq", &gpu.dq, &cpu.dq),
        ("dk", &gpu.dk, &cpu.dk),
        ("dv", &gpu.dv, &cpu.dv),
    ] {
        assert!(all_finite(g), "{tag}: GPU {name} has non-finite values");
        let m = max_abs_diff(g, c);
        let ref_mag = max_abs(c);
        let gpu_mag = max_abs(g);
        eprintln!(
            "[{tag}] {name}: max_abs_diff = {m:.4e} (tol {tol:.1e}); |gpu|={gpu_mag:.3e} |ref|={ref_mag:.3e}"
        );
        // Non-triviality: a real reference gradient must be met by a real GPU
        // gradient of comparable magnitude, else the kernel silently no-op'd.
        if ref_mag > 1e-4 && gpu_mag < 0.3 * ref_mag {
            failures.push(format!(
                "{name} GPU output trivially small (|gpu|={gpu_mag:.3e} vs |ref|={ref_mag:.3e}) — likely a swallowed launch failure"
            ));
        }
        if m > tol {
            failures.push(format!("{name} max_abs_diff {m:.4e} > tol {tol:.1e}"));
        }
    }
    assert!(failures.is_empty(), "{tag}: {}", failures.join("; "));
}

/// TEMP localization diagnostic (not a gate): run the MMA path (sm=80) across
/// minimal shapes and print per-grad max_abs_diff + magnitudes + first samples,
/// to pinpoint whether the numerical bug is in the core single-tile/single-head
/// MMA, the multi-tile q-loop accumulation, or the heads mapping.
#[test]
#[ignore = "diagnostic: localizes MMA backward numerical bug"]
fn diag_mma_localize() {
    if !cuda_available() {
        return;
    }
    let shapes = [
        (1usize, 1usize, 64usize, 16usize, "core_h1_s64_hd16"),
        (1, 1, 64, 32, "core_h1_s64_hd32"),
        (1, 1, 128, 32, "multitile_h1_s128_hd32"),
        (1, 2, 64, 32, "heads_h2_s64_hd32"),
        (1, 4, 128, 32, "full_h4_s128_hd32"),
    ];
    for (b, h, s, d, tag) in shapes {
        let (gpu, cpu) = run(b, h, s, d, false, 80);
        eprintln!("── [{tag}]  b{b} h{h} s{s} hd{d} ──");
        for (name, g, c) in [
            ("dq", &gpu.dq, &cpu.dq),
            ("dk", &gpu.dk, &cpu.dk),
            ("dv", &gpu.dv, &cpu.dv),
        ] {
            let m = max_abs_diff(g, c);
            eprintln!(
                "   {name}: max_abs_diff={m:.4e}  |gpu|={:.4e} |ref|={:.4e}  finite={}",
                max_abs(g),
                max_abs(c),
                all_finite(g),
            );
            eprintln!(
                "        gpu[0..6]={:?}",
                g.iter().take(6).map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>()
            );
            eprintln!(
                "        ref[0..6]={:?}",
                c.iter().take(6).map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>()
            );
        }
    }
}

// ── Baseline: single-tile heads=1, scalar path (harness sanity) ─────────────

/// If this fails, the harness/oracle is wrong (not a multi-tile/heads bug):
/// seq=64=block_q is a single q-tile, heads=1, the simplest possible case.
/// hd16 scalar keeps shared memory under the 48 KB static cap.
#[test]
#[ignore = "requires CUDA GPU"]
fn plain_bwd_singletile_heads1_hd16_scalar() {
    if !cuda_available() {
        return;
    }
    let (gpu, cpu) = run(1, 1, 64, 16, false, 75);
    assert_parity("singletile_h1_hd16_scalar", &gpu, &cpu, tol_for(16));
}

// ── Scalar-path heads>1 + multi-tile (the gap, on the sm<80 code path) ───────

#[test]
#[ignore = "requires CUDA GPU"]
fn plain_bwd_multitile_heads4_hd16_scalar() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let (gpu, cpu) = run(1, 4, 128, 16, causal, 75);
        assert_parity(&format!("multitile_scalar_h4_hd16_c{}", causal as u8), &gpu, &cpu, tol_for(16));
    }
}

/// Scalar heads>1 + multi-tile at head_dim=32 — the largest head_dim whose
/// resident-tile shared-memory layout fits the sm_120 99 KB opt-in cap at
/// block=64 (70.5 KB). This is the current GPU-path ceiling.
#[test]
#[ignore = "requires CUDA GPU"]
fn plain_bwd_multitile_heads4_hd32_scalar() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let (gpu, cpu) = run(1, 4, 128, 32, causal, 75);
        assert_parity(&format!("multitile_scalar_h4_hd32_c{}", causal as u8), &gpu, &cpu, tol_for(32));
    }
}

/// batch>1 + heads>1 + multi-tile, scalar path: exercises the
/// grid_x = batch*heads mapping (batch and head both fold into blockIdx.x).
#[test]
#[ignore = "requires CUDA GPU"]
fn plain_bwd_multitile_batch2_heads4_hd16_scalar() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let (gpu, cpu) = run(2, 4, 128, 16, causal, 75);
        assert_parity(&format!("multitile_b2_h4_hd16_c{}", causal as u8), &gpu, &cpu, tol_for(16));
    }
}

// ── MMA-path (sm_80+ tensor-core) heads>1 + multi-tile parity ───────────────
//
// The GPU MMA backward (`flash_attn_bwd_main` on sm>=80 — the path the RTX 5070
// Ti / sm_120 takes) is now numerically correct: its m16n8k16 fragment loaders
// and C stores follow the hardware lane layout (row=laneid/4, col=2*(laneid%4)),
// and the per-warp `atom.add` accumulate (dV/dK/dQ) is gated to warp 0 — the CTA
// is block_q=64 threads = 2 warps, each of which independently computes the full
// MMA tile, so an unguarded atom.add over-counts by the warp count (an exact 2×).
// These mirror the scalar tests but at gpu_sm=80 so they exercise the tensor-core
// kernel. The non-triviality guard in `assert_parity` still catches a silent CPU
// fallback (which would match but never touch the GPU kernel), and the eprintln'd
// |gpu| vs |ref| makes any fallback visible.

#[test]
#[ignore = "requires CUDA GPU"]
fn mma_bwd_singletile_heads1_hd16() {
    if !cuda_available() {
        return;
    }
    let (gpu, cpu) = run(1, 1, 64, 16, false, 80);
    assert_parity("mma_singletile_h1_hd16", &gpu, &cpu, tol_for(16));
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mma_bwd_multitile_heads4_hd16() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let (gpu, cpu) = run(1, 4, 128, 16, causal, 80);
        assert_parity(
            &format!("mma_multitile_h4_hd16_c{}", causal as u8),
            &gpu,
            &cpu,
            tol_for(16),
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn mma_bwd_multitile_heads4_hd32() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let (gpu, cpu) = run(1, 4, 128, 32, causal, 80);
        assert_parity(
            &format!("mma_multitile_h4_hd32_c{}", causal as u8),
            &gpu,
            &cpu,
            tol_for(32),
        );
    }
}

/// batch>1 + heads>1 + multi-tile on the MMA path: exercises grid_x = batch*heads.
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_bwd_multitile_batch2_heads4_hd16() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let (gpu, cpu) = run(2, 4, 128, 16, causal, 80);
        assert_parity(
            &format!("mma_multitile_b2_h4_hd16_c{}", causal as u8),
            &gpu,
            &cpu,
            tol_for(16),
        );
    }
}

/// Regression guard: the sm_80+ MMA backward Phase 2 PTX must JIT cleanly. It
/// previously failed ptxas ("Arguments mismatch for instruction 'add'") because
/// the MMA tile loaders fed a `.u32` offset into `add.s64`; the u32→u64 widening
/// restored validity. If this regresses, the runtime silently falls back to the
/// CPU backward (correct but slow) — this catches it at the PTX level instead.
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_backward_phase2_ptx_is_valid() {
    if !cuda_available() {
        return;
    }
    let cfg = FlashAttentionBackwardConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 32,
        causal: false,
        gpu_sm: 80,
        segment_masked: false,
    };
    let (_ptx1, ptx2) = synthesize_flash_attention_backward_ptx(&cfg);
    let log = read_jit_log(ptx2.as_ptr() as i64);
    assert!(
        !(log.contains("error") || log.contains("mismatch")),
        "MMA backward Phase 2 PTX failed to JIT (regressed): {log}"
    );
}

/// Silent-failure guard: an MMA config whose kernel genuinely cannot launch must
/// NOT return zero gradients — the runtime must fall back to the correct CPU
/// backward. head_dim=64 exceeds the shared-memory opt-in cap at block=64 (see
/// `plain_bwd_shared_budget_ceiling_is_hd32`), so its GPU launch fails; the
/// fallback must still produce correct grads (zeros would fail the non-triviality
/// check in `assert_parity`).
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_hd64_falls_back_to_correct_grads_not_zeros() {
    if !cuda_available() {
        return;
    }
    let (gpu, cpu) = run(1, 1, 64, 64, false, 80);
    assert_parity("mma_hd64_fallback", &gpu, &cpu, tol_for(64));
}

/// head_dim >= 64 exceeds the resident-tile shared-memory budget at the runtime's
/// fixed block_q=block_kv=64 (hd64 needs ~118.5 KB > the sm_120 99 KB opt-in cap),
/// so the GPU kernel cannot launch and the runtime falls back to CPU. This host
/// assertion documents the ceiling; lifting it needs a block-size/layout redesign.
#[test]
fn plain_bwd_shared_budget_ceiling_is_hd32() {
    let mk = |hd: i64| FlashAttentionBackwardConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: hd,
        causal: false,
        gpu_sm: 80,
        segment_masked: false,
    };
    // sm_120 (RTX 5070 Ti) MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
    const OPTIN_CAP: u32 = 99 * 1024;
    assert!(
        backward_shared_mem_bytes(&mk(32)) <= OPTIN_CAP,
        "head_dim=32 must fit the opt-in cap"
    );
    assert!(
        backward_shared_mem_bytes(&mk(64)) > OPTIN_CAP,
        "head_dim=64 is expected to exceed the opt-in cap (block-size redesign needed)"
    );
}
