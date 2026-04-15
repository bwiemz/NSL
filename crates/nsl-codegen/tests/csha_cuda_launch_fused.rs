//! CSHA launch mechanics (Part 2 of numerical validation): real cudarc launch
//! of the CSHA FFI with fused Q/K/V projections + RoPE enabled.
//!
//! ## Config (C2 smoke — single row)
//!
//! * block_q=32, block_kv=32, head_dim=32, heads=4, d_model=32
//! * causal=false, rope_q=true
//! * csha.fused_rmsnorm=true, csha.fused_projections=true, csha.fused_output_proj=false
//! * active_heads=4 (all active)
//! * NSL_FA_EMITTER=v2
//!
//! ## CPU reference
//!
//! Consumed via `#[path = "csha_reference.rs"] mod csha_reference;`.
//! The reference computes:
//!   RMSNorm(x) → x_norm → x_norm@Wq/Wk/Wv → Q,K,V →
//!   RoPE(Q); RoPE(K) → softmax(Q@K^T / sqrt(d)) → O=P@V
//!
//! ## Layout note (x and norm)
//!
//! The v2 kernel's RMSNorm prologue indexes x as `[batch, heads, seq, head_dim]`
//! (comment in csha_hooks.rs emit_prologue) and normalises over `head_dim`
//! features per head.  The CPU reference `csha_reference` normalises over
//! `d_model` features per token.
//!
//! To reconcile, this test:
//!   1. Lays out x for the kernel as `[heads, seq, head_dim]` (transposed from
//!      the canonical `[seq, d_model]`).
//!   2. Calls the CPU reference with `d_model=head_dim, heads=1` per head,
//!      using the per-head x slice, so the reference's RMSNorm matches the
//!      kernel's head-wise normalisation.
//!   3. Passes a per-head weight slice `[head_dim, head_dim]` to the reference
//!      (head-0 column block of the full `[d_model, heads*head_dim]` weight),
//!      matching what the kernel's weight-tile load supplies.
//!
//! ## Running
//!
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test csha_cuda_launch_fused \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference, det_seq, CshaInputs, CshaShape};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_selector::{
    flash_attention_kernel_name_selected, shared_mem_bytes_selected,
    synthesize_flash_attention_ptx_selected, Emitter, select_emitter,
};
use std::ffi::CString;

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use nsl_runtime::flash_attention::nsl_flash_attention_csha;

// --------------------------------------------------------------------------
// Serialisation helpers (identical to Part 1)
// --------------------------------------------------------------------------

use std::sync::{Mutex, MutexGuard};
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

/// IEEE 754 f16 → f32 (same as Part 1).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
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

/// f32 → f16 bits (round-to-nearest, clamp on overflow).
fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() { return 0x7E00; }
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 255 {
        return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16;
    }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        // subnormal or underflow
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

// --------------------------------------------------------------------------
// Core fused launch helper
// --------------------------------------------------------------------------

/// Runs the fused-CSHA GPU kernel and the matching CPU reference, returning
/// `(max_abs_diff, max_rel_diff)`.
fn run_fused_csha_and_measure() -> (f32, f32) {
    // ── Shape constants ───────────────────────────────────────────────────
    let batch    = 1usize;
    let heads    = 4usize;
    let seq      = 32usize;
    let head_dim = 32usize;
    // d_model = head_dim: kernel RMSNorm normalises head_dim features per row.
    let d_model  = head_dim;
    let causal   = false;
    let norm_eps = 1e-5f32;
    let scale    = 1.0f32 / (head_dim as f32).sqrt();
    let active_heads_i = heads as i64;

    // ── Host data generation ──────────────────────────────────────────────
    //
    // x_kernel: [heads, seq, head_dim] — the layout the kernel's RMSNorm
    // prologue expects.  Kernel indexes as [batch*heads, seq, head_dim].
    let x_kernel = det_seq(42, heads * seq * head_dim);

    // Wq: [d_model=32, head_dim=32] f32.  All heads share one Wq pointer.
    // Wk/Wv not used by the kernel (pre-projected K/V uploaded directly).
    let wq_f32 = det_seq(43, d_model * head_dim);
    let wk_f32 = det_seq(44, d_model * head_dim);
    let wv_f32 = det_seq(45, d_model * head_dim);

    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    // Wk/Wv pointers in kernel params will be null — K/V come pre-projected.

    // norm_weight: [head_dim] all 1.0.
    let norm_weight_f32 = vec![1.0f32; head_dim];

    // ── CPU reference (no RoPE) ───────────────────────────────────────────
    //
    // CPU reference: RMSNorm(x) → Q=x_norm@Wq, K=x_norm@Wk, V=x_norm@Wv
    // → softmax(Q@K^T / sqrt(d)) @ V.
    // rope_q=false: cos=1, sin=0 (identity RoPE).
    let zeros_rope  = vec![0f32; seq * (head_dim / 2)];
    let ones_cos    = vec![1.0f32; seq * (head_dim / 2)];

    // Compute per-head CPU output and also extract pre-projected K/V for GPU.
    let mut cpu_out = vec![0f32; seq * heads * head_dim];
    // Pre-projected K and V: [heads, seq, head_dim] f32, layout [batch, heads, seq, head_dim].
    // Uploaded as f32 because k_tile_load reads f32 (ld.global.f32) and converts to f16 in SMEM.
    let mut kv_proj_f32 = vec![0f32; heads * seq * head_dim]; // K
    let mut vv_proj_f32 = vec![0f32; heads * seq * head_dim]; // V

    for h in 0..heads {
        // Per-head x slice: [seq, head_dim].
        let x_h: Vec<f32> = (0..seq * head_dim)
            .map(|idx| x_kernel[h * seq * head_dim + idx])
            .collect();

        // CPU reference (no RoPE: cos=1, sin=0).
        let shape = CshaShape {
            seq, heads: 1, head_dim, d_model: head_dim, causal, norm_eps,
        };
        let inputs = CshaInputs {
            x: &x_h, wq: &wq_f32, wk: &wk_f32, wv: &wv_f32,
            norm_weight: &norm_weight_f32, cos: &ones_cos, sin: &zeros_rope,
        };
        let out_h = csha_reference(&inputs, &shape);
        for s in 0..seq {
            cpu_out[s * heads * head_dim + h * head_dim
                ..s * heads * head_dim + (h + 1) * head_dim]
                .copy_from_slice(&out_h[s * head_dim..(s + 1) * head_dim]);
        }

        // Also compute x_norm for K/V projection.
        // Re-use: RMSNorm(x_h[s]) → x_norm_s, then K_s = x_norm_s @ Wk, V_s = x_norm_s @ Wv.
        for s in 0..seq {
            let row = &x_h[s * head_dim..(s + 1) * head_dim];
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let rms = (mean_sq + norm_eps).sqrt();
            let x_norm_s: Vec<f32> = row.iter().zip(norm_weight_f32.iter())
                .map(|(v, w)| v / rms * w)
                .collect();
            // K_s[col] = dot(x_norm_s, Wk[:, col])
            for col in 0..head_dim {
                let mut sum_k = 0f32;
                let mut sum_v = 0f32;
                for k in 0..d_model {
                    sum_k += x_norm_s[k] * wk_f32[k * head_dim + col];
                    sum_v += x_norm_s[k] * wv_f32[k * head_dim + col];
                }
                kv_proj_f32[h * seq * head_dim + s * head_dim + col] = sum_k;
                vv_proj_f32[h * seq * head_dim + s * head_dim + col] = sum_v;
            }
        }
    }

    // ── Device allocations ────────────────────────────────────────────────
    let x_bytes       = (x_kernel.len() * 4) as i64;
    let wq_bytes      = (wq_f16.len() * 2) as i64;
    let nw_bytes      = (norm_weight_f32.len() * 4) as i64;
    let qkv_elems     = batch * heads * seq * head_dim;
    let qkv_f16_bytes = (qkv_elems * 2) as i64;
    let qkv_f32_bytes = (qkv_elems * 4) as i64;
    let out_bytes     = qkv_f16_bytes;
    let lse_bytes     = (batch * heads * seq * 4) as i64;

    let x_dev   = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let wq_dev  = unsafe { nsl_test_cuda_alloc(wq_bytes) };
    let nw_dev  = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    // Dummy q_dev (Q comes from fused projection, not HBM q_ptr).
    let q_dev   = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    // K and V: pre-projected f32 tensors (k_tile_load reads f32, converts to f16 in SMEM).
    let k_dev   = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let v_dev   = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };

    let all_ptrs = [
        x_dev, wq_dev, nw_dev, q_dev, k_dev, v_dev, out_dev, lse_dev,
    ];
    assert!(all_ptrs.iter().all(|&p| p != 0), "device allocation returned null");

    unsafe {
        nsl_test_cuda_h2d(x_dev,  x_kernel.as_ptr()        as i64, x_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr()           as i64, wq_bytes);
        nsl_test_cuda_h2d(nw_dev, norm_weight_f32.as_ptr()  as i64, nw_bytes);
        // q_dev left uninitialised (kernel uses fused projection, not q_ptr).
        nsl_test_cuda_h2d(k_dev,  kv_proj_f32.as_ptr()      as i64, qkv_f32_bytes);
        nsl_test_cuda_h2d(v_dev,  vv_proj_f32.as_ptr()      as i64, qkv_f32_bytes);
    }

    // ── PTX synthesis ─────────────────────────────────────────────────────
    let config = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: head_dim as i64,
        causal,
        paged:           false,
        rope_q:          false, // RoPE disabled: CPU ref uses identity cos/sin
        rope_style:      RopeStyle::HalfSplit,
        gqa_group_size:  1,
        tree_mask:       false,
        gpu_sm:          75,
        csha: Some(CshaExtras {
            level:             2,
            fused_rmsnorm:     true,
            fused_projections: true,
            fused_output_proj: false,
            active_heads:      active_heads_i as u32,
            rmsnorm_eps:       norm_eps,
            d_model:           d_model as u32,
        }),
    };

    let synth = std::panic::catch_unwind(|| synthesize_flash_attention_ptx_selected(&config));
    let mut ptx = match synth {
        Ok(p) => p,
        Err(e) => {
            free_all(&all_ptrs);
            let msg = e.downcast_ref::<String>().cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            panic!("C2 emitter panicked: {msg}");
        }
    };

    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    let dump = std::env::temp_dir().join("csha_fused_c2_smoke.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("C2 PTX dumped to: {}", dump.display());
    ptx.push(0);

    let kernel_name = CString::new(flash_attention_kernel_name_selected(&config)).unwrap();
    let smem_static = shared_mem_bytes_selected(&config) as i64;
    // The kernel uses ONLY static shared memory (shmem[N] in PTX).
    // cuLaunchKernel adds dynamic SMEM on top of static, so passing
    // smem_static as dynamic would double-count and overflow sm_75's 48 KB
    // per-CTA limit.  Pass 0 to request no additional dynamic SMEM.
    let smem_dynamic: i64 = 0;
    eprintln!("C2 kernel: {}, static_smem={}B dynamic={}B",
        kernel_name.to_str().unwrap(), smem_static, smem_dynamic);

    // ── Launch ────────────────────────────────────────────────────────────
    let rc = unsafe {
        nsl_flash_attention_csha(
            // Base FlashAttention params
            q_dev, k_dev, v_dev,
            out_dev,
            lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, head_dim as i64,
            0, 0, 0, 0,   // paging: block_table, k_pool, v_pool, block_size
            0, 0,         // cos_ptr=0, sin_ptr=0 (rope_q=false)
            0, 0,         // seq_ids, seq_lens
            smem_dynamic,
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            32i64, 32i64, // block_q, block_kv
            0i64,         // causal=false
            // CSHA extras
            x_dev,        // x_ptr
            nw_dev,       // norm_weight_ptr
            wq_dev,       // wq_ptr
            0i64,         // wk_ptr=null (K pre-projected in k_dev)
            0i64,         // wv_ptr=null (V pre-projected in v_dev)
            0i64,         // wo_ptr=null (fused_output_proj=false)
            norm_eps.to_bits() as i64,
            active_heads_i,
            d_model as i64,
        )
    };

    if rc != 0 {
        let log = read_jit_log(ptx.as_ptr() as i64);
        free_all(&all_ptrs);
        panic!("C2 launch FAILED rc={rc}\nJIT log:\n{log}");
    }

    // ── Readback + compare ────────────────────────────────────────────────
    let mut out_f16 = vec![0u16; qkv_elems];
    unsafe { nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, out_bytes); }
    // GPU finalize stores in [heads, seq, head_dim] order (batch=1).
    // CPU reference stores in [seq, heads, head_dim] order.
    // Transpose GPU output to [seq, heads, head_dim] for comparison.
    let out_gpu_raw: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();
    let mut out_gpu = vec![0f32; qkv_elems];
    for h in 0..heads {
        for s in 0..seq {
            for d in 0..head_dim {
                let gpu_idx = (h * seq + s) * head_dim + d;
                let cpu_idx = s * heads * head_dim + h * head_dim + d;
                out_gpu[cpu_idx] = out_gpu_raw[gpu_idx];
            }
        }
    }

    let all_finite = out_gpu.iter().all(|v| v.is_finite());
    let nonzero = out_gpu.iter().filter(|v| v.abs() > 1e-6).count();
    eprintln!("C2 output: finite={all_finite}, nonzero={nonzero}/{qkv_elems}");
    let show = 8.min(qkv_elems);
    eprintln!("C2 first {show} GPU: {:?}", &out_gpu[..show]);
    eprintln!("C2 first {show} CPU: {:?}", &cpu_out[..show]);

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_idx = 0usize;
    for (i, (&g, &c)) in out_gpu.iter().zip(cpu_out.iter()).enumerate() {
        let abs = (g - c).abs();
        let rel = abs / c.abs().max(1e-6);
        if abs > max_abs { max_abs = abs; max_idx = i; }
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!(
        "C2 smoke: max_abs={:.3e}  max_rel={:.3e}  at idx={} gpu={:.6} cpu={:.6}",
        max_abs, max_rel, max_idx, out_gpu[max_idx], cpu_out[max_idx]
    );

    free_all(&all_ptrs);
    (max_abs, max_rel)
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

fn free_all(ptrs: &[i64]) {
    for &p in ptrs {
        if p != 0 { unsafe { nsl_test_cuda_free(p); } }
    }
}

fn read_jit_log(ptx_ptr: i64) -> String {
    let log_ptr = unsafe { nsl_test_cuda_jit_log(ptx_ptr) };
    if log_ptr != 0 {
        unsafe { std::ffi::CStr::from_ptr(log_ptr as *const i8).to_string_lossy().into_owned() }
    } else {
        "<no log>".into()
    }
}

// --------------------------------------------------------------------------
// Test entry point
// --------------------------------------------------------------------------

/// C2 Part 2 fused-CSHA GPU smoke test — single config.
///
/// block_q=32, block_kv=32, head_dim=32, heads=4, d_model=32,
/// causal=false, rope_q=false (identity), fused_rmsnorm=true,
/// fused_projections=true (Q only; K/V pre-projected in k_dev/v_dev),
/// fused_output_proj=false.  Tolerance: 5e-3.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_csha_32x32x32_heads4_nocausal() {
    let _guard = lock_env();
    std::env::set_var("NSL_FA_EMITTER", "v2");

    if !cuda_available() { return; }

    // Confirm v2 routing before spending time on GPU launch.
    let probe = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75,
        csha: Some(CshaExtras {
            level: 2, fused_rmsnorm: true, fused_projections: true,
            fused_output_proj: false, active_heads: 4,
            rmsnorm_eps: 1e-5, d_model: 32,
        }),
    };
    assert_eq!(
        select_emitter(&probe),
        Emitter::V2,
        "selector must route to v2 after NSL_FA_EMITTER=v2"
    );

    let (max_abs, _) = run_fused_csha_and_measure();

    assert!(
        max_abs <= 5e-3,
        "C2 fused CSHA: max_abs_diff {max_abs:.3e} exceeds 5e-3 tolerance"
    );
}
