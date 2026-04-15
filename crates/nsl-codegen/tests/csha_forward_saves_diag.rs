//! Diagnostic: verify the forward-with-saves emission writes correct
//! Q_proj / K_proj / V_proj / row_max / row_sum into the backward
//! activation arena. Compares against a minimal inline CPU reference
//! (RMSNorm → Wq/Wk/Wv matmul → scale → softmax stats).
//!
//! This test is ORTHOGONAL to the existing T6.3 backward numerical gate:
//! it stops after forward and never launches the backward kernel, so
//! it isolates whether the saves themselves are correct — a necessary
//! pre-condition for the backward gate to mean anything.
//!
//! Hypothesis map (from close-out memory):
//!   1. row_max / row_sum don't match softmax stats from the CPU path.
//!   2. Q/K/V projection saves use a different layout than backward reads.
//!   3. Backward applies the scale factor 1/sqrt(d) at the wrong place.
//!
//! Per-tensor max_abs_diff + first-8-element side-by-side dumps are
//! emitted via eprintln! so a human can triage which hypothesis is live.
//! No asserts — this is pure diagnostic.

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
#[allow(dead_code)]
mod csha_reference;

use std::ffi::CString;

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_flash_attention_ptx_v2,
    smem_layout::{self, needs_dynamic_smem, Direction},
};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h,
    nsl_test_cuda_free, nsl_test_cuda_h2d,
};
use nsl_runtime::flash_attention::{
    nsl_csha_alloc_backward_activations, nsl_csha_free_backward_activations,
    nsl_flash_attention_csha_with_saves,
};

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 { sign << 31 } else {
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

fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() { return 0x7E00; }
    let b = x.to_bits();
    let sign = (b >> 31) & 1;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 255 { return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16; }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 { return ((sign << 15) | 0x7C00) as u16; }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((s >> 16) as f32 / 65535.0) - 0.5
    }).collect()
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() { return false; }
    unsafe { nsl_cuda_init() == 0 }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0f32, f32::max)
}

fn dump_pair(name: &str, gpu: &[f32], cpu: &[f32], n: usize) {
    let n = n.min(gpu.len()).min(cpu.len());
    let mut buf = String::new();
    for i in 0..n {
        buf.push_str(&format!("    [{i:2}] gpu={:+.4} cpu={:+.4} diff={:+.3e}\n",
            gpu[i], cpu[i], gpu[i] - cpu[i]));
    }
    let mad = max_abs_diff(gpu, cpu);
    eprintln!("[diag] {name}: max_abs_diff={mad:.3e}  (elems={})\n{buf}", gpu.len());
}

#[test]
#[ignore]
fn forward_saves_match_cpu_reference() {
    if !cuda_available() {
        eprintln!("[diag] skipping — no CUDA (source-level audit in agent report)");
        return;
    }

    // Config matches t6_3_smoke_single_config: heads=1, d_model=hd=32,
    // seq=32, block_q=block_kv=32, no causal, no RoPE.
    let batch = 1usize;
    let heads = 1usize;
    let head_dim = 32usize;
    let d_model = 32usize;
    let seq = 32usize;
    let block_q = 32u32;
    let block_kv = 32u32;
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let kv_dim = heads * head_dim;

    let config = FlashAttentionConfig {
        block_q: block_q as i64, block_kv: block_kv as i64, head_dim: head_dim as i64,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
        csha: Some(CshaExtras {
            level: 2,
            fused_rmsnorm: true,
            fused_projections: true,
            fused_output_proj: false,
            save_activations_for_backward: true,
            active_heads: heads as u32,
            rmsnorm_eps: norm_eps,
            d_model: d_model as u32,
        }),
    };

    smem_layout::validate_scalar_v2_config(&config, Direction::Forward)
        .expect("validator should pass for 32x32x32 config");

    // Host data identical to t6_3_smoke.
    let x = det_seq(42, heads * seq * head_dim);
    let wq_f32 = det_seq(43, d_model * kv_dim);
    let wk_f32 = det_seq(44, d_model * kv_dim);
    let wv_f32 = det_seq(45, d_model * kv_dim);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let nw = vec![1.0f32; head_dim];
    let cos = vec![1.0f32; seq * head_dim / 2];
    let sin = vec![0.0f32; seq * head_dim / 2];

    // ── Inline CPU forward (matches csha_reference::forward_intermediates) ──
    // RMSNorm rows of x using norm_weight == 1.
    let mut x_norm = Vec::with_capacity(seq * d_model);
    for s in 0..seq {
        let row = &x[s * d_model..(s + 1) * d_model];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let r = (mean_sq + norm_eps).sqrt();
        x_norm.extend(row.iter().zip(&nw).map(|(v, w)| v / r * w));
    }
    // Q = x_norm @ Wq (RoPE is identity here: cos=1, sin=0).
    let matmul = |a: &[f32], b: &[f32], m: usize, k: usize, n: usize| -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k { sum += a[i*k+p] * b[p*n+j]; }
                c[i*n+j] = sum;
            }
        }
        c
    };
    let q_cpu = matmul(&x_norm, &wq_f32, seq, d_model, kv_dim);
    let k_cpu = matmul(&x_norm, &wk_f32, seq, d_model, kv_dim);
    let v_cpu = matmul(&x_norm, &wv_f32, seq, d_model, kv_dim);

    // row_max / row_sum: post-scale, post-mask (no mask), pre-softmax.
    // Shape on GPU: [batch, heads, seq] f32.
    let mut rowmax_cpu = vec![0f32; batch * heads * seq];
    let mut rowsum_cpu = vec![0f32; batch * heads * seq];
    for h in 0..heads {
        for i in 0..seq {
            let mut s_row = vec![0f32; seq];
            for j in 0..seq {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_cpu[i*kv_dim + h*head_dim + d]
                         * k_cpu[j*kv_dim + h*head_dim + d];
                }
                s_row[j] = dot * scale;
            }
            let rm = s_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let rs: f32 = s_row.iter().map(|&v| (v - rm).exp()).sum();
            let idx = h * seq + i; // batch=1
            rowmax_cpu[idx] = rm;
            rowsum_cpu[idx] = rs;
        }
    }

    // Reshape q_cpu/k_cpu/v_cpu from [seq, heads*head_dim] to GPU save layout
    // [batch, heads, seq, head_dim] (batch=1 so trivial).
    let reshape_to_save_layout = |t: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; batch * heads * seq * head_dim];
        for h in 0..heads {
            for i in 0..seq {
                for d in 0..head_dim {
                    // save layout: ((batch*heads + h)*seq + i)*head_dim + d
                    let dst = h * seq * head_dim + i * head_dim + d;
                    let src = i * kv_dim + h * head_dim + d;
                    out[dst] = t[src];
                }
            }
        }
        out
    };
    let q_cpu_save = reshape_to_save_layout(&q_cpu);
    let k_cpu_save = reshape_to_save_layout(&k_cpu);
    let v_cpu_save = reshape_to_save_layout(&v_cpu);

    // ── GPU forward-with-saves ──────────────────────────────────────────────
    unsafe { nsl_cuda_init(); }
    let qkv_bytes = (heads * seq * head_dim * 2) as i64;
    let lse_bytes = (batch * heads * seq * 4) as i64;
    let x_bytes = (heads * seq * head_dim * 4) as i64;
    let w_bytes = (d_model * kv_dim * 2) as i64;
    let nw_bytes = (head_dim * 4) as i64;
    let rope_bytes = (seq * head_dim / 2 * 4) as i64;

    let q_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let k_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let v_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };
    let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let nw_dev = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    let wq_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wk_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wv_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let cos_dev = unsafe { nsl_test_cuda_alloc(rope_bytes) };
    let sin_dev = unsafe { nsl_test_cuda_alloc(rope_bytes) };

    let saves = unsafe {
        nsl_csha_alloc_backward_activations(
            batch as i64, heads as i64, seq as i64, head_dim as i64,
        )
    };

    unsafe {
        nsl_test_cuda_h2d(x_dev,  x.as_ptr()  as i64, x_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(nw_dev, nw.as_ptr() as i64, nw_bytes);
        nsl_test_cuda_h2d(cos_dev, cos.as_ptr() as i64, rope_bytes);
        nsl_test_cuda_h2d(sin_dev, sin.as_ptr() as i64, rope_bytes);
    }

    let ptx = synthesize_flash_attention_ptx_v2(&config);
    let name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    let smem_total = smem_layout::total_bytes(&config);
    let smem_dyn = if needs_dynamic_smem(&config) { smem_total as i64 } else { 0 };

    let rc = unsafe {
        nsl_flash_attention_csha_with_saves(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, head_dim as i64,
            0, 0, 0, 0,
            cos_dev, sin_dev,
            0, 0,
            smem_dyn,
            ptx.as_ptr() as i64, name.as_ptr() as i64,
            block_q as i64, block_kv as i64,
            0,
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, d_model as i64,
            saves.q_proj, saves.k_proj, saves.v_proj,
            saves.row_max, saves.row_sum,
        )
    };
    assert_eq!(rc, 0, "forward-with-saves rc={rc}");

    // Read back saves.
    let qkv_elems = batch * heads * seq * head_dim;
    let rm_elems  = batch * heads * seq;
    let read_f16 = |dev: i64, n: usize| -> Vec<f32> {
        let mut raw = vec![0u16; n];
        unsafe { nsl_test_cuda_d2h(raw.as_mut_ptr() as i64, dev, (n * 2) as i64); }
        raw.iter().map(|&b| f16_to_f32(b)).collect()
    };
    let read_f32 = |dev: i64, n: usize| -> Vec<f32> {
        let mut out = vec![0f32; n];
        unsafe { nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dev, (n * 4) as i64); }
        out
    };
    let q_gpu = read_f16(saves.q_proj, qkv_elems);
    let k_gpu = read_f16(saves.k_proj, qkv_elems);
    let v_gpu = read_f16(saves.v_proj, qkv_elems);
    let rm_gpu = read_f32(saves.row_max, rm_elems);
    let rs_gpu = read_f32(saves.row_sum, rm_elems);

    dump_pair("q_proj", &q_gpu, &q_cpu_save, 8);
    dump_pair("k_proj", &k_gpu, &k_cpu_save, 8);
    dump_pair("v_proj", &v_gpu, &v_cpu_save, 8);
    dump_pair("row_max", &rm_gpu, &rowmax_cpu, 8);
    dump_pair("row_sum", &rs_gpu, &rowsum_cpu, 8);

    unsafe {
        nsl_csha_free_backward_activations(saves);
        for p in [q_dev, k_dev, v_dev, out_dev, lse_dev, x_dev, nw_dev,
                  wq_dev, wk_dev, wv_dev, cos_dev, sin_dev] {
            if p != 0 { nsl_test_cuda_free(p); }
        }
    }
}
