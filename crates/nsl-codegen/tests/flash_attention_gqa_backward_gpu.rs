//! GPU numeric parity gate for the GQA (kv_heads < heads) flash-attention
//! backward — the expand-KV envelope added by the scaling campaign.
//!
//! # Why this file exists
//!
//! The phase-1/phase-2 backward kernels index K/V by the fused `b*h` block id,
//! i.e. they are MHA-only. Before the scaling campaign, `nsl_flash_attention_backward`
//! routed any `kv_h != h` input to the (correct but ~50%-of-process-CPU) host
//! reference. The campaign added a device-side envelope: expand K/V to full heads
//! (consecutive-block mapping `kv_h_idx = h_idx / gqa_groups`, matching the CPU
//! reference), run the proven MHA kernels, then group-sum dK/dV back — the exact
//! `ReduceToShape` adjoint of the expansion. This file proves that envelope's
//! dQ/dK/dV match the segment-aware CPU GQA reference (`flash_attention_backward_cpu_gqa`)
//! at real GQA ratios and multi-tile seq lengths.
//!
//! It also covers the sibling campaign fix — the non-contiguous-view input
//! materialization — by passing transpose-view inputs and asserting the GPU
//! result still matches (the runtime now materializes a canonical copy instead
//! of falling back to CPU).
//!
//! # Running
//!
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test flash_attention_gqa_backward_gpu \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::flash_attention::{
    flash_attention_bwd_d_kernel_name, flash_attention_bwd_main_kernel_name,
    synthesize_flash_attention_backward_ptx, FlashAttentionBackwardConfig,
};

use nsl_runtime::flash_attention::{
    flash_attention_backward_cpu_gqa, nsl_flash_attention_backward, select_backward_blocks,
};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_transpose, nsl_tensor_zeros_on,
};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    nsl_cuda_init() == 0
}

/// Deterministic pseudo-random values in roughly [-0.5, 0.5] (LCG); same
/// generator as the MHA backward parity test so tolerances are comparable.
fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 65535.0) - 0.5
        })
        .collect()
}

/// Naive GQA forward. Q has `h` heads, K/V have `kv_h` heads; Q head `hi` reads
/// KV head `hi / (h/kv_h)`. Returns (out[b*h*s*d], lse[b*h*s]).
#[allow(clippy::too_many_arguments)]
fn naive_forward_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    h: usize,
    kv_h: usize,
    s: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> (Vec<f32>, Vec<f32>) {
    let groups = h / kv_h;
    let q_bh = h * s * d;
    let q_h = s * d;
    let kv_bh = kv_h * s * d;
    let kv_h_stride = s * d;
    let row = d;
    let lse_bh = h * s;
    let lse_h = s;

    let mut out = vec![0.0f32; b * h * s * d];
    let mut lse = vec![0.0f32; b * h * s];

    for bi in 0..b {
        for hi in 0..h {
            let kv_hi = hi / groups;
            let q_base = bi * q_bh + hi * q_h;
            let kv_base = bi * kv_bh + kv_hi * kv_h_stride;
            let lse_base = bi * lse_bh + hi * lse_h;
            for i in 0..s {
                let q_row = q_base + i * row;
                let j_max = if causal { i + 1 } else { s };
                let mut scores = vec![0.0f32; j_max];
                let mut m = f32::NEG_INFINITY;
                for (j, sc) in scores.iter_mut().enumerate() {
                    let k_row = kv_base + j * row;
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += q[q_row + dd] * k[k_row + dd];
                    }
                    *sc = dot * scale;
                    if *sc > m {
                        m = *sc;
                    }
                }
                let mut denom = 0.0f32;
                for &sc in &scores {
                    denom += (sc - m).exp();
                }
                lse[lse_base + i] = m + denom.ln();
                for (j, &sc) in scores.iter().enumerate() {
                    let p = (sc - m).exp() / denom;
                    let v_row = kv_base + j * row;
                    for dd in 0..d {
                        out[q_row + dd] += p * v[v_row + dd];
                    }
                }
            }
        }
    }
    (out, lse)
}

/// Create a GPU f32 tensor [shape] filled with `vals`.
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

fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let dptr = nsl_tensor_data_ptr(t);
    let mut out = vec![0.0f32; len];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dptr, (len * 4) as i64);
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn max_abs(a: &[f32]) -> f32 {
    a.iter().map(|x| x.abs()).fold(0.0, f32::max)
}
fn all_finite(a: &[f32]) -> bool {
    a.iter().all(|x| x.is_finite())
}

fn tol_for(head_dim: usize) -> f32 {
    match head_dim {
        hd if hd >= 128 => 5e-2,
        hd if hd >= 64 => 3e-2,
        _ => 1.5e-2,
    }
}

struct Grads {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

/// Synthesize backward PTX for `head_dim` and return the (ptx1, ptx2, name1,
/// name2) so the raw pointers stay alive across the FFI call at the call site.
fn synth(d: usize, causal: bool) -> (Vec<u8>, Vec<u8>, CString, CString) {
    let (bq, bkv) = select_backward_blocks(d as i64);
    let cfg = FlashAttentionBackwardConfig {
        block_q: bq,
        block_kv: bkv,
        head_dim: d as i64,
        causal,
        gpu_sm: 80, // MMA path (5070 Ti is sm_120 >= 80)
        segment_masked: false,
    };
    let (ptx1, ptx2) = synthesize_flash_attention_backward_ptx(&cfg);
    let name1 = CString::new(flash_attention_bwd_d_kernel_name(&cfg)).unwrap();
    let name2 = CString::new(flash_attention_bwd_main_kernel_name(&cfg)).unwrap();
    (ptx1, ptx2, name1, name2)
}

/// Run the GQA backward through the production FFI (which takes the expand-KV
/// envelope) and via the CPU GQA reference. When `view_inputs`, feed q/k/v as
/// transpose views to also exercise the campaign's canonical-copy materialization.
fn run_gqa(
    b: usize,
    h: usize,
    kv_h: usize,
    s: usize,
    d: usize,
    causal: bool,
    view_inputs: bool,
) -> (Grads, Grads) {
    let scale = 1.0f32 / (d as f32).sqrt();
    let groups = h / kv_h;
    let q_total = b * h * s * d;
    let kv_total = b * kv_h * s * d;

    let q = det_seq(1, q_total);
    let k = det_seq(2, kv_total);
    let v = det_seq(3, kv_total);
    let dout = det_seq(4, q_total);

    let (out, lse) = naive_forward_gqa(&q, &k, &v, b, h, kv_h, s, d, scale, causal);

    // ── CPU GQA reference (oracle) ──
    let mut dq_c = vec![0.0f32; q_total];
    let mut dk_c = vec![0.0f32; kv_total];
    let mut dv_c = vec![0.0f32; kv_total];
    flash_attention_backward_cpu_gqa(
        &q, &k, &v, &out, &lse, &dout, &mut dq_c, &mut dk_c, &mut dv_c, b, h, kv_h, s, d, scale,
        causal, groups, None,
    );

    // ── GPU (system under test) ──
    let (ptx1, ptx2, name1, name2) = synth(d, causal);

    let q_shape = [b as i64, h as i64, s as i64, d as i64];
    let kv_shape = [b as i64, kv_h as i64, s as i64, d as i64];

    // Optionally build inputs as transpose views: store the transposed layout
    // then transpose(1,2) back to logical [b,h,s,d] with non-canonical strides.
    let (q_t, k_t, v_t, out_t, dout_t, view_temps) = if view_inputs {
        // Store [b, s, h, d] contiguous, then transpose dims 1,2 -> a
        // non-contiguous [b, h, s, d] view.
        let permute = |src: &[f32], heads: usize| -> Vec<f32> {
            // src is [b, heads, s, d] logical row-major; produce [b, s, heads, d].
            let mut dst = vec![0.0f32; src.len()];
            for bi in 0..b {
                for hi in 0..heads {
                    for si in 0..s {
                        for dd in 0..d {
                            let s_idx = ((bi * heads + hi) * s + si) * d + dd;
                            let d_idx = ((bi * s + si) * heads + hi) * d + dd;
                            dst[d_idx] = src[s_idx];
                        }
                    }
                }
            }
            dst
        };
        let qv = gpu_tensor(&permute(&q, h), &[b as i64, s as i64, h as i64, d as i64]);
        let kv = gpu_tensor(&permute(&k, kv_h), &[b as i64, s as i64, kv_h as i64, d as i64]);
        let vv = gpu_tensor(&permute(&v, kv_h), &[b as i64, s as i64, kv_h as i64, d as i64]);
        let ov = gpu_tensor(&permute(&out, h), &[b as i64, s as i64, h as i64, d as i64]);
        let dv_ = gpu_tensor(&permute(&dout, h), &[b as i64, s as i64, h as i64, d as i64]);
        let qview = nsl_tensor_transpose(qv, 1, 2);
        let kview = nsl_tensor_transpose(kv, 1, 2);
        let vview = nsl_tensor_transpose(vv, 1, 2);
        let oview = nsl_tensor_transpose(ov, 1, 2);
        let dview = nsl_tensor_transpose(dv_, 1, 2);
        (
            qview,
            kview,
            vview,
            oview,
            dview,
            vec![qv, kv, vv, ov, dv_, qview, kview, vview, oview, dview],
        )
    } else {
        let qt = gpu_tensor(&q, &q_shape);
        let kt = gpu_tensor(&k, &kv_shape);
        let vt = gpu_tensor(&v, &kv_shape);
        let ot = gpu_tensor(&out, &q_shape);
        let dt = gpu_tensor(&dout, &q_shape);
        (qt, kt, vt, ot, dt, vec![qt, kt, vt, ot, dt])
    };

    let grads_list = nsl_flash_attention_backward(
        dout_t, q_t, k_t, v_t, out_t,
        0, // logsumexp = 0 -> runtime auto-recomputes
        scale.to_bits() as i64,
        b as i64, h as i64, s as i64, d as i64,
        causal as i64,
        ptx1.as_ptr() as i64, name1.as_ptr() as i64,
        ptx2.as_ptr() as i64, name2.as_ptr() as i64,
        0, 0, // tier_b sentinel
        0, // segment_ids (unmasked)
    );
    assert!(grads_list != 0, "GQA backward FFI returned null list");

    // dQ is [b,h,s,d]; dK/dV are reduced back to [b,kv_h,s,d] by the envelope.
    let dq_g = read_gpu(nsl_list_get(grads_list, 0), q_total);
    let dk_g = read_gpu(nsl_list_get(grads_list, 1), kv_total);
    let dv_g = read_gpu(nsl_list_get(grads_list, 2), kv_total);

    for t in view_temps {
        nsl_tensor_free(t);
    }
    let _ = (&ptx1, &ptx2, &name1, &name2);

    (
        Grads { dq: dq_g, dk: dk_g, dv: dv_g },
        Grads { dq: dq_c, dk: dk_c, dv: dv_c },
    )
}

fn assert_parity(tag: &str, gpu: &Grads, cpu: &Grads, tol: f32) {
    let mut failures = Vec::new();
    for (name, g, c) in [
        ("dq", &gpu.dq, &cpu.dq),
        ("dk", &gpu.dk, &cpu.dk),
        ("dv", &gpu.dv, &cpu.dv),
    ] {
        assert!(all_finite(g), "{tag}: GPU {name} non-finite");
        assert_eq!(g.len(), c.len(), "{tag}: {name} length mismatch (dK/dV must be reduced to kv_h)");
        let m = max_abs_diff(g, c);
        let ref_mag = max_abs(c);
        let gpu_mag = max_abs(g);
        eprintln!("[{tag}] {name}: max_abs_diff={m:.4e} (tol {tol:.1e}); |gpu|={gpu_mag:.3e} |ref|={ref_mag:.3e}");
        if ref_mag > 1e-4 && gpu_mag < 0.3 * ref_mag {
            failures.push(format!("{name} GPU trivially small (|gpu|={gpu_mag:.3e} vs |ref|={ref_mag:.3e})"));
        }
        if m > tol {
            failures.push(format!("{name} max_abs_diff {m:.4e} > tol {tol:.1e}"));
        }
    }
    assert!(failures.is_empty(), "{tag}: {}", failures.join("; "));
}

/// GQA ratios that real models use (8Q/2KV = GQA-4, 16Q/4KV = GQA-4, 8Q/1KV = MQA),
/// multi-tile seq, causal — the exact regime decorator-free pretraining hits.
#[test]
#[ignore = "GPU: requires CUDA device"]
fn gqa_backward_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }
    let cases = [
        (1usize, 8usize, 2usize, 128usize, 32usize, "gqa4_h8kv2_hd32"),
        (2, 8, 2, 128, 64, "gqa4_h8kv2_hd64_b2"),
        (1, 8, 1, 128, 64, "mqa_h8kv1_hd64"),
        (1, 16, 4, 128, 32, "gqa4_h16kv4_hd32"),
        (1, 4, 2, 256, 32, "gqa2_h4kv2_s256"),
    ];
    for (b, h, kv_h, s, d, tag) in cases {
        let (gpu, cpu) = run_gqa(b, h, kv_h, s, d, true, false);
        assert_parity(tag, &gpu, &cpu, tol_for(d));
    }
}

/// The sibling campaign fix: non-contiguous (transpose-view) inputs must now be
/// materialized to canonical device copies and stay on the GPU, matching the
/// stride-aware CPU reference — not silently routed to the whole-CPU backward.
#[test]
#[ignore = "GPU: requires CUDA device"]
fn gqa_backward_view_inputs_materialize_on_gpu() {
    if !cuda_available() {
        return;
    }
    // MHA (kv==h) and GQA both via view inputs — the materialization path is
    // shared, GQA additionally exercises expand-after-materialize.
    for (b, h, kv_h, s, d, tag) in [
        (1usize, 8usize, 8usize, 128usize, 64usize, "mha_view_h8_hd64"),
        (1, 8, 2, 128, 64, "gqa4_view_h8kv2_hd64"),
    ] {
        let (gpu, cpu) = run_gqa(b, h, kv_h, s, d, true, true);
        assert_parity(tag, &gpu, &cpu, tol_for(d));
    }
}
