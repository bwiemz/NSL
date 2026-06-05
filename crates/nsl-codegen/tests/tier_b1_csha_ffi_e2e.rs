//! End-to-end production-FFI test for the CSHA Tier B.1 forward kernel.
//!
//! Where `tier_b1_full_kernel_e2e.rs` validates the kernel via direct
//! `nsl_kernel_launch` with manually-prepared chunkified inputs, this
//! test exercises the **production** path through
//! `nsl_flash_attention_csha`:
//!
//!   1. Caller supplies raw f32 `x` `[seq, d_model]`, f32 `gamma` `[d_model]`,
//!      and f32 `Wq/Wk/Wv` `[d_model, hd]` in row-major layout — the
//!      standard CSHA pipeline format.
//!   2. The FFI inspects the kernel name (`_tier_b1_chunk<N>` suffix) to
//!      detect Tier B.1 dispatch, runs the GPU pre-pass kernels (RMSNorm
//!      + narrow + chunkify on x; narrow + col-major chunkify on Wq/Wk/Wv),
//!      then launches the main Tier B.1 kernel with the chunkified
//!      pointers substituted in.
//!   3. Output is compared element-wise against `csha_reference` (the
//!      f32 CPU implementation).
//!
//! When this test passes, the Tier B.1 pipeline is usable by production
//! callers without requiring host-side pre-processing.
//!
//! Running:
//! ```bash
//! cargo test --package nsl-codegen --features cuda \
//!     --test tier_b1_csha_ffi_e2e -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference, CshaInputs, CshaShape};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2,
    synthesize_flash_attention_ptx_v2,
};
use nsl_codegen::flash_attention_v2::smem_layout::tier_b1_total_smem_bytes;
use nsl_runtime::flash_attention::nsl_flash_attention_csha;
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use std::ffi::CString;

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
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
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

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

fn canonical_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 2048,
            // The production FFI's pre-pass orchestration always feeds the
            // kernel a pre-RMSNormalized + chunkified x. The codegen
            // dispatch path in `synthesize_flash_attention_ptx_v2` forces
            // this flag to `true` when emitting Tier B.1 PTX (see
            // `flash_attention_v2::synthesize_flash_attention_ptx_v2_with_tier_b`),
            // so the caller's value here is overridden. We leave it `false`
            // to exercise the override path.
            skip_rmsnorm_prologue: false,
            ..CshaExtras::default()
        }),
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_through_csha_ffi_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }

    let config = canonical_config();
    let batch = 1usize;
    let heads = 1usize;
    let seq = config.block_q as usize;
    let head_dim = config.head_dim as usize;
    let kv_dim = heads * head_dim;
    let d_model = config
        .csha
        .as_ref()
        .map(|c| c.d_model as usize)
        .expect("canonical config has csha");
    let causal = config.causal;
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Host inputs (identical fixture to `tier_b1_full_kernel_e2e`).
    let mut x_host = vec![0f32; seq * d_model];
    for s in 0..seq {
        for d in 0..d_model {
            x_host[s * d_model + d] = ((s + d) as f32).sin() * 0.1 + 0.05;
        }
    }
    let mut wq_f32 = vec![0f32; d_model * kv_dim];
    let mut wk_f32 = vec![0f32; d_model * kv_dim];
    let mut wv_f32 = vec![0f32; d_model * kv_dim];
    for d in 0..d_model {
        for n in 0..kv_dim {
            wq_f32[d * kv_dim + n] = ((d * 7 + n) as f32).sin() * 0.05;
            wk_f32[d * kv_dim + n] = ((d * 11 + n) as f32).cos() * 0.05;
            wv_f32[d * kv_dim + n] = ((d * 13 + n) as f32).sin() * 0.05;
        }
    }
    let norm_weight = vec![1.0f32; d_model];

    // CPU reference.
    let cos = vec![1.0f32; seq * (head_dim / 2)];
    let sin = vec![0.0f32; seq * (head_dim / 2)];
    let cpu_out = csha_reference(
        &CshaInputs {
            x: &x_host,
            wq: &wq_f32,
            wk: &wk_f32,
            wv: &wv_f32,
            norm_weight: &norm_weight,
            cos: &cos,
            sin: &sin,
        },
        &CshaShape {
            seq, heads, head_dim, d_model, causal, norm_eps,
        },
    );

    // GPU buffers — all f32 raw inputs (standard CSHA pipeline format).
    let x_bytes = (x_host.len() * 4) as i64;
    let nw_bytes = (norm_weight.len() * 4) as i64;
    let w_bytes = (wq_f32.len() * 4) as i64;
    let qkv_elems = batch * heads * seq * head_dim;
    let qkv_f32_bytes = (qkv_elems * 4) as i64;
    let out_bytes = (qkv_elems * 2) as i64; // f16
    let lse_bytes = (batch * heads * seq * 4) as i64;

    let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let nw_dev = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    let wq_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wk_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wv_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let q_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let k_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let v_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };

    let all_ptrs = [x_dev, nw_dev, wq_dev, wk_dev, wv_dev, q_dev, k_dev, v_dev, out_dev, lse_dev];
    if !all_ptrs.iter().all(|&p| p != 0) {
        for &p in &all_ptrs { if p != 0 { unsafe { nsl_test_cuda_free(p) }; } }
        panic!("device alloc returned null");
    }

    unsafe {
        nsl_test_cuda_h2d(x_dev, x_host.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(nw_dev, norm_weight.as_ptr() as i64, nw_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f32.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f32.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f32.as_ptr() as i64, w_bytes);
    }

    // PTX synthesis — codegen will see Tier B.1 dispatch criteria and
    // emit the chunked variant with `_tier_b1_chunk<N>` in the name.
    // Crucially, the codegen also force-overrides
    // `skip_rmsnorm_prologue=true` for this path so the kernel does
    // NOT re-RMSNormalize (the runtime FFI's pre-pass already did).
    let mut ptx = synthesize_flash_attention_ptx_v2(&config);
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    let dump = std::env::temp_dir().join("tier_b1_csha_ffi_e2e.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("[csha-ffi] PTX dumped to {}", dump.display());
    ptx.push(0);
    let kernel_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    eprintln!("[csha-ffi] kernel name = {:?}", kernel_name);
    // Sanity: the name MUST encode the Tier B.1 chunk suffix; without it
    // the runtime's prepass orchestration is skipped.
    let name_str = kernel_name.to_string_lossy();
    assert!(
        name_str.contains("_tier_b1_chunk"),
        "Tier B.1 dispatch should have appended _tier_b1_chunk<N> to the kernel name; got {:?}",
        name_str
    );

    // Tier B.1 declares `.shared .b8 shmem[N]` statically (N <= 48KB
    // for canonical 32x32x32 — 45056 bytes fits the static cap). Pass
    // `shared_mem_bytes=0` so the runtime does NOT opt-in to dynamic
    // SMEM via cuFuncSetAttribute (which would also try to call
    // `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` and fail with
    // CUDA_ERROR_INVALID_VALUE on a static-SMEM kernel).
    let smem = 0i64;
    let _ = tier_b1_total_smem_bytes(&config, 128); // suppress unused-import warn
    let _ = shared_mem_bytes_v2; // suppress unused-import warn under cfg(cuda)

    // ---- Launch through the production FFI ----
    let rc = nsl_flash_attention_csha(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        batch as i64, heads as i64, seq as i64, head_dim as i64,
        0, 0, 0, 0,             // paging
        0, 0, 0, 0,             // rope + ragged
        smem,
        ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        config.block_q as i64, config.block_kv as i64,
        if causal { 1 } else { 0 },
        // CSHA extras: x, norm_weight, wq, wk, wv, wo, eps, ah, dm
        x_dev, nw_dev, wq_dev, wk_dev, wv_dev, 0,
        norm_eps.to_bits() as i64,
        0, d_model as i64,
        // segment_ids_ptr, tier_b sentinels, doc_starts
        0, 0, 0, 0,
    );

    if rc != 0 {
        let log_ptr = unsafe { nsl_test_cuda_jit_log(ptx.as_ptr() as i64) };
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no log>".into()
        };
        for &p in &all_ptrs { unsafe { nsl_test_cuda_free(p) }; }
        panic!("csha FFI launch rc={}\nJIT log:\n{}", rc, log);
    }

    // Readback + compare.
    let mut out_f16 = vec![0u16; qkv_elems];
    unsafe { nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, out_bytes) };
    let out_gpu: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();

    for &p in &all_ptrs { unsafe { nsl_test_cuda_free(p) }; }

    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    let mut max_idx = 0usize;
    let mut n_nan = 0usize;
    for (i, (&g, &c)) in out_gpu.iter().zip(cpu_out.iter()).enumerate() {
        if !g.is_finite() {
            n_nan += 1;
            continue;
        }
        let diff = (g - c).abs();
        sum_abs += diff;
        if diff > max_abs {
            max_abs = diff;
            max_idx = i;
        }
    }
    let mean_abs = sum_abs / qkv_elems as f32;
    eprintln!("\n[csha-ffi] ── numerical summary ──");
    eprintln!("[csha-ffi] total elems          = {}", qkv_elems);
    eprintln!("[csha-ffi] non-finite GPU elems = {}", n_nan);
    eprintln!("[csha-ffi] max_abs              = {:.4e}", max_abs);
    eprintln!("[csha-ffi] mean_abs             = {:.4e}", mean_abs);
    if max_idx < out_gpu.len() {
        eprintln!(
            "[csha-ffi] worst idx={} gpu={:.6} cpu={:.6}",
            max_idx, out_gpu[max_idx], cpu_out[max_idx]
        );
    }
    eprintln!("\n[csha-ffi] first 8 GPU: {:?}", &out_gpu[..8.min(out_gpu.len())]);
    eprintln!("[csha-ffi] first 8 CPU: {:?}", &cpu_out[..8.min(cpu_out.len())]);

    assert_eq!(n_nan, 0, "FFI path produced {} non-finite outputs", n_nan);
    // Same tolerance as `tier_b1_full_kernel_e2e` — the production FFI
    // adds an RMSNorm + narrow + chunkify GPU pre-pass which contributes
    // negligible additional noise (the f16 narrowing dominates).
    const MAX_ABS_TOL: f32 = 0.6;
    const MEAN_ABS_TOL: f32 = 0.25;
    assert!(
        max_abs <= MAX_ABS_TOL,
        "FFI-path max_abs {} > {} tolerance",
        max_abs, MAX_ABS_TOL
    );
    assert!(
        mean_abs <= MEAN_ABS_TOL,
        "FFI-path mean_abs {} > {} tolerance",
        mean_abs, MEAN_ABS_TOL
    );
    eprintln!(
        "[csha-ffi] PASS (max_abs={:.4e} ≤ {:.4e}; mean_abs={:.4e} ≤ {:.4e})",
        max_abs, MAX_ABS_TOL, mean_abs, MEAN_ABS_TOL
    );
}
