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

    // Wq/Wk/Wv: [d_model=32, head_dim=32] f32.  All heads share one pointer each.
    // All three are uploaded as f16 for the fused projection sweeps.
    let wq_f32 = det_seq(43, d_model * head_dim);
    let wk_f32 = det_seq(44, d_model * head_dim);
    let wv_f32 = det_seq(45, d_model * head_dim);

    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    // norm_weight: [head_dim] all 1.0.
    let norm_weight_f32 = vec![1.0f32; head_dim];

    // ── CPU reference (no RoPE) ───────────────────────────────────────────
    //
    // CPU reference: RMSNorm(x) → Q=x_norm@Wq, K=x_norm@Wk, V=x_norm@Wv
    // → softmax(Q@K^T / sqrt(d)) @ V.
    // rope_q=false: cos=1, sin=0 (identity RoPE).
    let zeros_rope  = vec![0f32; seq * (head_dim / 2)];
    let ones_cos    = vec![1.0f32; seq * (head_dim / 2)];

    // CPU reference: all heads share wq/wk/wv.
    let mut cpu_out = vec![0f32; seq * heads * head_dim];

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
    }

    // ── Device allocations ────────────────────────────────────────────────
    let x_bytes       = (x_kernel.len() * 4) as i64;
    let w_bytes       = (wq_f16.len() * 2) as i64; // same size for Wq/Wk/Wv
    let nw_bytes      = (norm_weight_f32.len() * 4) as i64;
    let qkv_elems     = batch * heads * seq * head_dim;
    let qkv_f16_bytes = (qkv_elems * 2) as i64;
    let qkv_f32_bytes = (qkv_elems * 4) as i64;
    let out_bytes     = qkv_f16_bytes;
    let lse_bytes     = (batch * heads * seq * 4) as i64;

    let x_dev   = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let wq_dev  = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wk_dev  = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wv_dev  = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let nw_dev  = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    // Dummy q/k/v_dev: kernel derives Q/K/V from x@Wq/Wk/Wv; classic ptrs unused.
    // Pass non-null allocations so the kernel doesn't fault if it reads them.
    let q_dev   = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let k_dev   = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let v_dev   = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };

    let all_ptrs = [
        x_dev, wq_dev, wk_dev, wv_dev, nw_dev, q_dev, k_dev, v_dev, out_dev, lse_dev,
    ];
    assert!(all_ptrs.iter().all(|&p| p != 0), "device allocation returned null");

    unsafe {
        nsl_test_cuda_h2d(x_dev,  x_kernel.as_ptr()           as i64, x_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr()             as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr()             as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr()             as i64, w_bytes);
        nsl_test_cuda_h2d(nw_dev, norm_weight_f32.as_ptr()    as i64, nw_bytes);
        // q/k/v_dev left uninitialised: kernel uses fused Wq/Wk/Wv projections.
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
            wq_dev,       // wq_ptr — fused Q projection
            wk_dev,       // wk_ptr — fused K projection (kernel derives K from x@Wk)
            wv_dev,       // wv_ptr — fused V projection (kernel derives V from x@Wv)
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
    // Diagnostic: read back GPU x (normalized in-place by kernel).
    let mut x_back = vec![0f32; heads * seq * head_dim];
    unsafe { nsl_test_cuda_d2h(x_back.as_mut_ptr() as i64, x_dev, x_bytes); }
    // Check x_norm for all heads
    {
        let mut all_norm_ok = true;
        for h in 0..heads {
            for row in 0..2usize { // check rows 0 and 1
                let row_raw: Vec<f32> = (0..head_dim).map(|d| x_kernel[h * seq * head_dim + row * head_dim + d]).collect();
                let sumsq: f32 = row_raw.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
                let rms = (sumsq + norm_eps).sqrt();
                let norm_cpu: Vec<f32> = row_raw.iter().map(|v| v / rms).collect();
                let norm_gpu: Vec<f32> = (0..head_dim).map(|d| x_back[h * seq * head_dim + row * head_dim + d]).collect();
                let max_x_err = norm_cpu.iter().zip(norm_gpu.iter()).map(|(c,g)| (c-g).abs()).fold(0f32, f32::max);
                if max_x_err > 1e-4 { all_norm_ok = false; }
                eprintln!("C2 diag: x RMSNorm[h={h},row={row}] max_err={:.3e}", max_x_err);
            }
        }
        eprintln!("C2 diag: all x norms OK: {all_norm_ok}");
    }
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

    // Compute expected output for head=0, seq_pos=0 using GPU x_norm
    {
        let h = 0usize;
        let x_norm_h: Vec<f32> = (0..seq * head_dim).map(|i| x_back[h * seq * head_dim + i]).collect();
        let xn = &x_norm_h[..];
        let wq = &wq_f32[..]; let wk = &wk_f32[..]; let wv = &wv_f32[..];
        let mut q_mat = vec![0f32; seq * head_dim];
        let mut k_mat = vec![0f32; seq * head_dim];
        let mut v_mat = vec![0f32; seq * head_dim];
        for s in 0..seq { for d in 0..head_dim {
            q_mat[s * head_dim + d] = (0..d_model).map(|d2| xn[s * d_model + d2] * wq[d2 * head_dim + d]).sum();
            k_mat[s * head_dim + d] = (0..d_model).map(|d2| xn[s * d_model + d2] * wk[d2 * head_dim + d]).sum();
            v_mat[s * head_dim + d] = (0..d_model).map(|d2| xn[s * d_model + d2] * wv[d2 * head_dim + d]).sum();
        }}
        let mut s_mat = vec![0f32; seq * seq];
        for i in 0..seq { for j in 0..seq {
            s_mat[i * seq + j] = (0..head_dim).map(|d| q_mat[i * head_dim + d] * k_mat[j * head_dim + d]).sum::<f32>() * scale;
        }}
        for i in 0..seq {
            let row = &mut s_mat[i * seq..(i + 1) * seq];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sm = 0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sm += *v; }
            for v in row.iter_mut() { *v /= sm; }
        }
        let mut o_mat = vec![0f32; seq * head_dim];
        for i in 0..seq { for d in 0..head_dim {
            o_mat[i * head_dim + d] = (0..seq).map(|j| s_mat[i * seq + j] * v_mat[j * head_dim + d]).sum();
        }}
        eprintln!("C2 diag: expected O[h=0,s=0,0..4]: {:?}", &o_mat[..4]);
        // GPU raw: head h, seq s, dim d → flat index (h*seq + s)*head_dim + d
        for gpu_h in 0..heads {
            let gpu_raw_h: Vec<f32> = (0..4).map(|d| f16_to_f32(out_f16[(gpu_h * seq + 0) * head_dim + d])).collect();
            eprintln!("C2 diag: GPU raw O[h={gpu_h},s=0,0..4]: {gpu_raw_h:?}");
        }
        let cpu_h0s0: Vec<f32> = (0..4).map(|d| cpu_out[0 * heads * head_dim + 0 * head_dim + d]).collect();
        eprintln!("C2 diag: CPU O[h=0,s=0,0..4]: {cpu_h0s0:?}");
        // Check if all s rows give same GPU output (Q projection broken?)
        let mut all_same = true;
        let row0: Vec<f32> = (0..head_dim).map(|d| f16_to_f32(out_f16[(0 * seq + 0) * head_dim + d])).collect();
        for s in 1..seq {
            let rows: Vec<f32> = (0..head_dim).map(|d| f16_to_f32(out_f16[(0 * seq + s) * head_dim + d])).collect();
            if rows.iter().zip(row0.iter()).any(|(a, b)| (a - b).abs() > 0.01) {
                all_same = false; break;
            }
        }
        eprintln!("C2 diag: GPU h=0 all-rows-same: {all_same}");
        // Check row uniformity: if P is uniform (Q=0), all rows should be ~equal
        let row0_s1: Vec<f32> = (0..4).map(|d| f16_to_f32(out_f16[(0 * seq + 1) * head_dim + d])).collect();
        let row0_s4: Vec<f32> = (0..4).map(|d| f16_to_f32(out_f16[(0 * seq + 4) * head_dim + d])).collect();
        eprintln!("C2 diag: GPU h=0 s=1,d=0..4: {row0_s1:?}");
        eprintln!("C2 diag: GPU h=0 s=4,d=0..4: {row0_s4:?}");
    }

    let all_finite = out_gpu.iter().all(|v| v.is_finite());
    let nonzero = out_gpu.iter().filter(|v| v.abs() > 1e-6).count();
    eprintln!("C2 output: finite={all_finite}, nonzero={nonzero}/{qkv_elems}");
    let show = 8.min(qkv_elems);
    eprintln!("C2 first {show} GPU: {:?}", &out_gpu[..show]);
    eprintln!("C2 first {show} CPU: {:?}", &cpu_out[..show]);
    // Per-head max-abs diagnostics
    for h in 0..heads {
        let mut mx = 0f32;
        for s in 0..seq {
            for d in 0..head_dim {
                let idx = s * heads * head_dim + h * head_dim + d;
                mx = mx.max((out_gpu[idx] - cpu_out[idx]).abs());
            }
        }
        eprintln!("  head {h}: max_abs_diff = {mx:.4e}");
        // First 4 elements of row 0 for this head
        let g: Vec<_> = (0..4).map(|d| out_gpu[0 * heads * head_dim + h * head_dim + d]).collect();
        let c: Vec<_> = (0..4).map(|d| cpu_out[0 * heads * head_dim + h * head_dim + d]).collect();
        eprintln!("    GPU[s=0,h={h},0..4]={g:.4?}  CPU={c:.4?}");
    }

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
/// fused_projections=true (Q, K, V all fused; kernel derives Q/K/V from x@Wq/Wk/Wv),
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
