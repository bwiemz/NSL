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
//! Backward tile sizes are budget-selected per head_dim (`select_backward_blocks`),
//! not fixed at 64: hd<=32 -> 64/64 (MMA), hd64 -> 32/32 (MMA), hd128 -> 32/16
//! (MMA; asymmetric tiles are safe since the causal q-loop floor-start fix).
//! "Multi-tile" therefore means seq_len > block_q; these tests use seq_len = 128,
//! which is >= 2 tiles for every selected block_q. `run()` mirrors production by
//! keying its block sizes off the selector.
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
    backward_select_blocks, backward_shared_mem_bytes, flash_attention_bwd_d_kernel_name,
    flash_attention_bwd_main_kernel_name, synthesize_flash_attention_backward_ptx,
    FlashAttentionBackwardConfig,
};

use nsl_runtime::flash_attention::{
    flash_attention_backward_cpu, nsl_flash_attention_backward,
    nsl_test_flash_attention_backward_blocks, select_backward_blocks,
};
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
    // Match the production path: block sizes are budget-selected per head_dim, and
    // the production FFI (`nsl_flash_attention_backward`) independently derives the
    // same via `select_backward_blocks`, so the PTX we synthesize here and the grid/
    // SMEM the runtime computes agree (both keyed off head_dim).
    let (sel_bq, sel_bkv) = select_backward_blocks(d as i64);
    let cfg = FlashAttentionBackwardConfig {
        block_q: sel_bq,
        block_kv: sel_bkv,
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

/// Direct GPU backward probe at **explicit** `(block_q, block_kv)` via the test FFI
/// `nsl_test_flash_attention_backward_blocks` — synthesizes the Phase-1/2 PTX with
/// the same block sizes and does NOT fall back to CPU. Returns:
///   * `Some(Grads)` — the GPU kernel launched and produced gradients, or
///   * `None`        — the GPU launch failed (invalid PTX or SMEM request over the
///                     device opt-in cap), i.e. this config would fall back to CPU.
///
/// This is the discovery instrument for the block-size sweep: it classifies each
/// candidate as GPU-correct / GPU-wrong (compare the returned grads to the oracle)
/// / GPU-unavailable (`None`) without the production `select_backward_blocks` table
/// getting in the way.
#[allow(clippy::too_many_arguments)]
fn gpu_probe(
    b: usize,
    h: usize,
    s: usize,
    d: usize,
    causal: bool,
    gpu_sm: u32,
    block_q: i64,
    block_kv: i64,
) -> (Option<Grads>, Grads) {
    let scale = 1.0f32 / (d as f32).sqrt();
    let total = b * h * s * d;

    let q = det_seq(1, total);
    let k = det_seq(2, total);
    let v = det_seq(3, total);
    let dout = det_seq(4, total);

    let (out, lse) = naive_forward(&q, &k, &v, b, h, s, d, scale, causal);

    let mut dq_c = vec![0.0f32; total];
    let mut dk_c = vec![0.0f32; total];
    let mut dv_c = vec![0.0f32; total];
    flash_attention_backward_cpu(
        &q, &k, &v, &out, &lse, &dout, &mut dq_c, &mut dk_c, &mut dv_c, b, h, s, d, scale, causal,
    );

    let cfg = FlashAttentionBackwardConfig {
        block_q,
        block_kv,
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

    let grads_list = nsl_test_flash_attention_backward_blocks(
        block_q,
        block_kv,
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
    );

    let _ = (&ptx1, &ptx2, &name1, &name2);

    let gpu = if grads_list == 0 {
        None
    } else {
        Some(Grads {
            dq: read_gpu(nsl_list_get(grads_list, 0), total),
            dk: read_gpu(nsl_list_get(grads_list, 1), total),
            dv: read_gpu(nsl_list_get(grads_list, 2), total),
        })
    };
    (gpu, Grads { dq: dq_c, dk: dk_c, dv: dv_c })
}

/// EMPIRICAL block-size sweep (diagnostic, not a gate). For each candidate
/// `(gpu_sm, block_q, block_kv, head_dim)` it runs the GPU backward at those exact
/// tiles and classifies the result vs the CPU oracle: `OK` (within tol), `WRONG`
/// (with the observed max ratio, e.g. an exact 2× = redundant-warp bug), or
/// `UNAVAIL` (launch failed / over the SMEM opt-in cap → CPU fallback). This is
/// what decides which tiles `select_backward_blocks` may use for hd64/hd128.
#[test]
#[ignore = "diagnostic: sweeps backward block sizes for correctness"]
fn diag_block_size_sweep() {
    if !cuda_available() {
        return;
    }
    // (gpu_sm, block_q, block_kv, head_dim, note)
    let candidates = [
        // ── head_dim = 64 ──
        (80u32, 64i64, 64i64, 64usize, "MMA sym 64/64 (expect UNAVAIL: ~134KB)"),
        (80, 64, 32, 64, "MMA asym 64/32 (~84.5KB)"),
        (80, 32, 32, 64, "MMA sym 32/32, 1 warp (~59KB)"),
        (75, 64, 32, 64, "scalar asym 64/32"),
        (75, 32, 32, 64, "scalar sym 32/32"),
        // ── head_dim = 128 ──
        (80, 32, 16, 128, "MMA asym 32/16 (~70KB) <- PRODUCTION hd128 tiles (causal fixed by floor start)"),
        (80, 32, 32, 128, "MMA sym 32/32 (expect UNAVAIL: ~107KB)"),
        (80, 16, 16, 128, "sub-warp 16/16 at sm80: backward_uses_mma's block_q%32 gate routes to scalar (raw sub-warp MMA was NaN)"),
        (75, 32, 16, 128, "scalar asym 32/16"),
        (75, 16, 16, 128, "scalar sym 16/16 (~52KB)"),
    ];
    let (b, h, s) = (1usize, 2usize, 128usize);
    for causal in [false, true] {
        eprintln!("\n══════ causal={causal}  (b={b} h={h} s={s}) ══════");
        for (gpu_sm, bq, bkv, hd, note) in candidates {
            let (gpu, cpu) = gpu_probe(b, h, s, hd, causal, gpu_sm, bq, bkv);
            let path = if gpu_sm >= 80 { "MMA " } else { "scal" };
            match gpu {
                None => eprintln!(
                    "  sm{gpu_sm} {path} q{bq:<2} kv{bkv:<2} hd{hd:<3} -> UNAVAIL (CPU fallback)   | {note}"
                ),
                Some(g) => {
                    let tol = tol_for(hd);
                    let mut verdict = "OK   ";
                    let mut detail = String::new();
                    for (name, gg, cc) in [
                        ("dq", &g.dq, &cpu.dq),
                        ("dk", &g.dk, &cpu.dk),
                        ("dv", &g.dv, &cpu.dv),
                    ] {
                        let m = max_abs_diff(gg, cc);
                        let rmag = max_abs(cc);
                        let gmag = max_abs(gg);
                        let ratio = if rmag > 1e-6 { gmag / rmag } else { 0.0 };
                        if !all_finite(gg) {
                            verdict = "WRONG";
                            detail.push_str(&format!(" {name}=NaN"));
                        } else if m > tol {
                            verdict = "WRONG";
                            detail.push_str(&format!(" {name}(d={m:.2e},ratio={ratio:.3})"));
                        }
                    }
                    eprintln!(
                        "  sm{gpu_sm} {path} q{bq:<2} kv{bkv:<2} hd{hd:<3} -> {verdict}{detail:<40} | {note}"
                    );
                }
            }
        }
    }
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

// ── Block-size/shared-budget redesign: hd64/hd128 now run on GPU ────────────────
//
// Before the `select_backward_blocks` selector the backward used a fixed 64/64 tile,
// whose resident-tile SMEM footprint (~118 KB at hd64, ~230 KB at hd128) blew past
// the 99 KB opt-in cap on every sm_80..sm_120 device, so hd64/hd128 ALWAYS fell back
// to the (correct but slow) CPU backward. The selector shrinks the tiles per head_dim
// so the footprint fits and the GPU kernel launches. These gates use `gpu_probe` (the
// no-fallback test FFI): a `Some` result PROVES the GPU kernel ran — a CPU fallback
// would return `None`, so a fallback can't masquerade as a pass here.

/// head_dim=64 now runs on the **MMA (tensor-core)** GPU path. `select_backward_blocks(64)`
/// = (32,32): symmetric (dodges the asymmetric-causal dV bug), block_q=32 = one full
/// warp (MMA-eligible), ~59 KB MMA-inclusive < 99 KB cap. This is the coder-rl head_dim.
/// Exercises heads>1, batch>1, multi-tile (seq=128 > block_q=32), causal + non-causal.
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_hd64_runs_on_gpu_not_cpu_fallback() {
    if !cuda_available() {
        return;
    }
    let (bq, bkv) = select_backward_blocks(64);
    assert_eq!((bq, bkv), (32, 32), "selector regressed for hd64");
    for causal in [false, true] {
        let (gpu, cpu) = gpu_probe(2, 4, 128, 64, causal, 80, bq, bkv);
        let gpu = gpu.expect(
            "hd64 must LAUNCH on the GPU (32/32 fits the 99 KB opt-in cap); \
             `None` means it fell back to CPU — the redesign regressed",
        );
        assert_parity(&format!("mma_hd64_gpu_c{}", causal as u8), &gpu, &cpu, tol_for(64));
    }
}

/// head_dim=128 now runs on the **MMA (tensor-core)** GPU path.
/// `select_backward_blocks(128)` = (32,16): asymmetric — safe since the causal
/// q-loop start floors the tile ratio (`emit_bwd_causal_q_loop_start`) — with
/// block_q=32 = one full warp (MMA-eligible) at ~70 KB MMA-inclusive < 99 KB cap.
/// This is the coder7b head_dim, upgraded from the earlier 16/16 scalar tiles.
/// Exercises heads>1, multi-tile, causal + non-causal.
#[test]
#[ignore = "requires CUDA GPU"]
fn mma_hd128_runs_on_gpu_not_cpu_fallback() {
    if !cuda_available() {
        return;
    }
    let (bq, bkv) = select_backward_blocks(128);
    assert_eq!((bq, bkv), (32, 16), "selector regressed for hd128");
    for causal in [false, true] {
        let (gpu, cpu) = gpu_probe(1, 2, 128, 128, causal, 80, bq, bkv);
        let gpu = gpu.expect(
            "hd128 must LAUNCH on the GPU (32/16 MMA fits the cap); \
             `None` means it fell back to CPU — the redesign regressed",
        );
        assert_parity(&format!("mma_hd128_gpu_c{}", causal as u8), &gpu, &cpu, tol_for(128));
    }
}

// ── Asymmetric-causal regression gates (the floor-start fix) ────────────────────
//
// The causal q-loop start used to be `i_block = j_block`, which assumes
// block_q == block_kv; at asymmetric tiles it skipped live q-tiles and produced
// wrong dV/dK under a causal mask (found by `diag_block_size_sweep`, both paths).
// `emit_bwd_causal_q_loop_start` now floors `j*block_kv/block_q`. These gates pin
// the fix on BOTH code paths at both production and non-production tile shapes —
// via `gpu_probe`, so a CPU fallback cannot masquerade as a pass.

/// Asymmetric 64/32 at hd64, causal, on both the MMA (sm80) and scalar (sm75)
/// paths. 64/32 is not what the selector picks for hd64 (32/32), but it is the
/// exact shape the sweep first caught the bug at — the strongest regression pin.
#[test]
#[ignore = "requires CUDA GPU"]
fn asymmetric_causal_hd64_64_32_grads_correct_both_paths() {
    if !cuda_available() {
        return;
    }
    for (gpu_sm, path) in [(80u32, "mma"), (75, "scalar")] {
        for causal in [false, true] {
            let (gpu, cpu) = gpu_probe(1, 2, 128, 64, causal, gpu_sm, 64, 32);
            let gpu = gpu.expect("64/32 hd64 fits the cap; launch must succeed");
            assert_parity(
                &format!("asym_hd64_64_32_{path}_c{}", causal as u8),
                &gpu,
                &cpu,
                tol_for(64),
            );
        }
    }
}

/// Asymmetric 32/16 at hd128 on the scalar path (sm75) — the selector's hd128 tiles
/// but exercising the scalar q-loop's floor start (the MMA variant is covered by
/// `mma_hd128_runs_on_gpu_not_cpu_fallback`).
#[test]
#[ignore = "requires CUDA GPU"]
fn asymmetric_causal_hd128_32_16_scalar_grads_correct() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        let (gpu, cpu) = gpu_probe(1, 2, 128, 128, causal, 75, 32, 16);
        let gpu = gpu.expect("32/16 hd128 scalar fits the cap; launch must succeed");
        assert_parity(
            &format!("asym_hd128_32_16_scalar_c{}", causal as u8),
            &gpu,
            &cpu,
            tol_for(128),
        );
    }
}

/// Ragged-seq guard: the Phase-2 backward kernel has NO seq_len tail bounds —
/// a partial final q-tile computes phantom rows past seq_len that are NOT
/// causally masked (global_i >= s > every global_j), read the next batch-head's
/// Q/dO/D/L, pollute valid dV/dK rows, and atomicAdd garbage dQ into the
/// neighbor batch-head. The runtime therefore REFUSES the GPU launch for any
/// seq_len not a multiple of both tile sizes and falls back to the correct CPU
/// backward (adversarial-review finding on the hd128 32/16 selector change:
/// seq % 16 == 0 && seq % 32 != 0, e.g. 2000, was exact at the old 16/16 tiles
/// but gains a phantom half-tile at 32/16).
///
/// seq=80 (80 % 16 == 0, 80 % 32 == 16) is the cheap stand-in for that class:
/// the probe must refuse (None => loud CPU-fallback path), and the production
/// FFI must still return CORRECT gradients (via CPU), never silently-corrupt
/// GPU output. b=2/h=2 makes cross-batch-head corruption detectable if the
/// guard ever regresses and the kernel launches anyway.
#[test]
#[ignore = "requires CUDA GPU"]
fn ragged_seq_refuses_gpu_and_falls_back_correct() {
    if !cuda_available() {
        return;
    }
    for causal in [false, true] {
        // Probe at the production hd128 tiles: must refuse (not launch).
        let (gpu, _cpu) = gpu_probe(2, 2, 80, 128, causal, 80, 32, 16);
        assert!(
            gpu.is_none(),
            "ragged seq=80 must REFUSE the GPU backward at 32/16 tiles (c{})",
            causal as u8
        );
        // Production path: falls back to CPU and stays correct.
        let (grads, oracle) = run(1, 2, 80, 128, causal, 80);
        assert_parity(
            &format!("ragged_s80_hd128_cpu_fallback_c{}", causal as u8),
            &grads,
            &oracle,
            tol_for(128),
        );
    }
}

/// The runtime's `select_backward_blocks` mirror MUST equal codegen's authoritative
/// `backward_select_blocks` for every head_dim. They are baked into the Phase-2 kernel
/// name (codegen) and the launch grid + dynamic-SMEM request (runtime); if they ever
/// disagree, the runtime would launch a kernel whose tile geometry differs from the
/// grid/SMEM it set up — silent corruption. Host-only (no GPU needed).
#[test]
fn select_backward_blocks_matches_codegen() {
    for hd in [8i64, 16, 32, 48, 64, 96, 128, 256] {
        assert_eq!(
            select_backward_blocks(hd),
            nsl_codegen::flash_attention::backward_select_blocks(hd),
            "runtime/codegen backward block-size selectors disagree at head_dim={hd}",
        );
    }
}

/// The whole point of the redesign: after selection, EVERY production head_dim's
/// backward fits the 99 KB opt-in cap under the MMA-INCLUSIVE layout the runtime always
/// requests (it can't tell scalar vs MMA PTX apart), so the GPU kernel launches instead
/// of falling back to CPU. Also pins the MMA-eligibility of the tensor-core entries
/// (hd64 and hd128 must stay on tensor cores). Host-only. Replaces the old
/// `plain_bwd_shared_budget_ceiling_is_hd32`, whose ceiling this work lifts.
#[test]
fn selector_keeps_every_head_dim_within_opt_in_budget() {
    // sm_80..sm_120 MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
    const OPTIN_CAP: i64 = 99 * 1024; // 101376
    // MMA-inclusive SMEM — mirrors the runtime's Phase-2 request formula exactly
    // (K,V + Q,dO + dK,dV tiles, each padded head_dim; S + dP; D + L vectors).
    let mma_inclusive = |bq: i64, bkv: i64, hd: i64| -> i64 {
        let hdp = hd + 4; // BWD_PAD
        let tb = |r: i64, c: i64| r * c * 4;
        tb(bkv, hdp) * 2 + tb(bq, hdp) * 2 + tb(bkv, hdp) * 2 + tb(bq, bkv) * 2 + bq * 4 * 2
    };
    for hd in [16i64, 32, 64, 128] {
        let (bq, bkv) = select_backward_blocks(hd);
        let smem = mma_inclusive(bq, bkv, hd);
        assert!(
            smem <= OPTIN_CAP,
            "hd{hd} at {bq}/{bkv} needs {smem} B > {OPTIN_CAP} B opt-in cap — would fall back to CPU",
        );
        // Tensor-core eligibility for the real model head_dims (mirrors
        // `backward_uses_mma`): block_q a full warp multiple, block_kv on the
        // MMA_M=16 grid. hd64 = coder-rl, hd128 = coder7b — neither may silently
        // drop to the scalar path via a selector change.
        if hd == 64 || hd == 128 {
            assert!(
                bq % 32 == 0 && bkv % 16 == 0,
                "hd{hd} at {bq}/{bkv} lost MMA eligibility — tensor-core regression"
            );
        }
    }
    // hd=32 keeps the classic 64/64 tile (unchanged, byte-identical PTX for the common case).
    assert_eq!(select_backward_blocks(32), (64, 64));
    // hd128 uses asymmetric 32/16 (allowed since the causal floor-start fix).
    assert_eq!(select_backward_blocks(128), (32, 16));
    // hd>128 has no fitting tile even asymmetric: documented CPU fallback (64/64 overflows).
    assert_eq!(select_backward_blocks(256), (64, 64));
    assert!(mma_inclusive(64, 64, 256) > OPTIN_CAP, "hd256 is the documented CPU-fallback case");
}
