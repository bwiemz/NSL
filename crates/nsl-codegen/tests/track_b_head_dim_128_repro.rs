//! Track B head_dim=128 launch diagnostic — repro + regression gate.
//!
//! ## Purpose
//!
//! Synthesises the fused-CSHA PTX for a head_dim=128 config, dumps it to disk,
//! and launches it end-to-end via `nsl_flash_attention_csha`.  Used to:
//!   1. Diagnose the original rc=1 CUDA_ERROR_INVALID_VALUE failure.
//!   2. Serve as a regression gate so the fix is never silently reverted.
//!
//! ## Root cause summary (2026-04-15)
//!
//! ### Part A — static SMEM declaration blocker
//!
//! For block_q=32, block_kv=32, head_dim=128, d_model=128, fused_projections=true
//! the SMEM layout is:
//!   Q tile:         8192 B  (32 × 128 × 2)
//!   KV tile:        8192 B  (32 × 128 × 2)
//!   SP scratch:     4096 B  (8 iters × 4 warps × 32 × 4)
//!   Wq+Wk+Wv:     98304 B  (3 × 2 × 128 × 128)
//!   Softmax save:    256 B
//!   TOTAL:        119040 B  (116.25 KB)
//!
//! The static SMEM cap on all SM generations is 48 KB.  The PTX previously
//! declared `.shared .align 16 .b8 shmem[119040]` which the CUDA driver
//! rejected with CUDA_ERROR_INVALID_VALUE (rc=1) at cuModuleLoadData time.
//!
//! ### Part B — dynamic SMEM + device limit
//!
//! The fix switches to `.extern .shared .align 16 .b8 shmem[]` (module-scope,
//! NOT inside the function body — ptxas rejects `.extern .shared` there) and
//! calls `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, N)` before launch.
//!
//! However, `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` on the
//! RTX 5070 Ti (sm_120 / Blackwell) with CUDA 13.2 returns only 101,376 bytes
//! (99 KB) — not the 228 KB sometimes cited for sm_90+.  With d_model=128,
//! total SMEM = 116.25 KB > 99 KB, so `cuFuncSetAttribute` itself returns
//! CUDA_ERROR_INVALID_VALUE.
//!
//! ### Practical resolution
//!
//! Use d_model=32 with head_dim=128.  The weight matrix shrinks to [32, 128],
//! so Wq+Wk+Wv = 3 × 2 × 32 × 128 = 24,576 B instead of 98,304 B.
//!   TOTAL for d_model=32: ~44,288 B (43.25 KB) — fits static 48 KB cap.
//! The PTX uses a static `.shared` declaration (no dynamic SMEM needed).
//!
//! d_model=128 with head_dim=128 is physically impossible on this GPU.
//! The validator now rejects it explicitly with an "exceeds … dynamic SMEM
//! budget" error before any PTX is synthesised.
//!
//! ### Dynamic SMEM infrastructure (48-99 KB range)
//!
//! The dynamic SMEM infrastructure (`.extern .shared` at module scope +
//! `cuFuncSetAttribute` opt-in) is still useful for configs in the 48-99 KB
//! range, e.g. (64,64,64,d=64) at 56.5 KB.  That path is tested by
//! `csha_fused_matrix_sweep` in `csha_cuda_launch_fused.rs`.
//!
//! ## Running
//!
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test track_b_head_dim_128_repro \
//!     -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_selector::{
    flash_attention_kernel_name_selected,
    shared_mem_bytes_selected,
    synthesize_flash_attention_ptx_selected,
};
use nsl_codegen::flash_attention_v2::smem_layout::{needs_dynamic_smem, total_bytes, validate_scalar_v2_config, Direction};
use std::ffi::CString;

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_d2h,
};
use nsl_runtime::flash_attention::nsl_flash_attention_csha;

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

/// Build a CSHA config for head_dim=128, fused projections, with explicit d_model.
fn cfg_128(d_model: u32) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 128,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
            level: 2,
            fused_rmsnorm: true,
            fused_projections: true,
            fused_output_proj: false,
            active_heads: 4,
            rmsnorm_eps: 1e-5,
            d_model,
            save_activations_for_backward: false,
        }),
    }
}

/// Part A regression: d_model=128 must be rejected by the validator.
///
/// head_dim=128, d_model=128 → total SMEM = 116.25 KB > 99 KB device limit.
/// The validator must return an error so no PTX synthesis is attempted.
/// This ensures the CUDA_ERROR_INVALID_VALUE (rc=1) from the original bug is
/// caught statically, not at runtime.
#[test]
#[ignore = "requires CUDA GPU"]
fn track_b_d_model_128_rejected_by_validator() {
    // This test is GPU-ignored so it runs in the same test binary context but
    // can be individually targeted.  CUDA init not required for validator check.
    let config = cfg_128(128);
    let smem_total = total_bytes(&config);
    eprintln!(
        "[TrackB] d_model=128 smem_total={} bytes ({:.2} KB)",
        smem_total, smem_total as f32 / 1024.0
    );

    // Must require dynamic SMEM (> 48 KB static cap).
    assert!(
        needs_dynamic_smem(&config),
        "d_model=128 config ({} bytes) must exceed 48 KB static cap",
        smem_total
    );

    // Validator must reject it (exceeds 99 KB dynamic budget).
    let result = validate_scalar_v2_config(&config, Direction::Forward);
    assert!(
        result.is_err(),
        "d_model=128 ({} bytes = {:.2} KB) must be rejected by validator \
         (device limit 99 KB)",
        smem_total, smem_total as f32 / 1024.0
    );
    eprintln!("[TrackB] d_model=128 correctly rejected: {:?}", result.err());
}

/// Part B regression: d_model=32 with head_dim=128 must launch successfully.
///
/// head_dim=128, d_model=32 → total SMEM = ~44 KB → static SMEM (no dynamic opt-in).
/// This config exercises the full fused-projection path at head_dim=128 within the
/// device's static 48 KB SMEM budget.
///
/// Config: block_q=32, block_kv=32, head_dim=128, d_model=32, heads=4,
///         causal=false, fused_projections=true.
/// SMEM: ~44 KB < 48 KB static cap → static `.shared` declaration.
#[test]
#[ignore = "requires CUDA GPU"]
fn track_b_head_dim_128_d_model_32_launches() {
    if !cuda_available() { return; }

    let head_dim  = 128usize;
    let d_model   = 32usize;
    let block_q   = 32u32;
    let block_kv  = 32u32;
    let heads     = 4usize;
    let seq       = block_q as usize;
    let batch     = 1usize;
    let norm_eps  = 1e-5f32;
    let scale     = 1.0f32 / (head_dim as f32).sqrt();

    let config = cfg_128(d_model as u32);

    // Compute and report SMEM layout.
    let smem_total = total_bytes(&config);
    let is_dynamic = needs_dynamic_smem(&config);
    eprintln!(
        "[TrackB] d_model=32 head_dim=128 smem_total={} bytes ({:.2} KB) dynamic={}",
        smem_total, smem_total as f32 / 1024.0, is_dynamic
    );
    assert!(
        !is_dynamic,
        "d_model=32 head_dim=128 config ({} bytes) must fit in static 48 KB budget",
        smem_total
    );
    assert!(
        smem_total <= 48 * 1024,
        "d_model=32 head_dim=128 config ({} bytes) must be <= 48 KB",
        smem_total
    );

    // Validate config passes the validator.
    validate_scalar_v2_config(&config, Direction::Forward)
        .expect("d_model=32 head_dim=128 config must pass validator");

    // Synthesize PTX.
    let mut ptx = synthesize_flash_attention_ptx_selected(&config);
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    let dump = std::env::temp_dir().join("track_b_head_dim_128_d32.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("[TrackB] PTX dumped to: {}", dump.display());

    // Verify static `.shared` (NOT `.extern .shared`) since smem_total < 48 KB.
    let ptx_str = std::str::from_utf8(&ptx).unwrap_or("");
    assert!(
        ptx_str.contains(&format!(".shared .align 16 .b8 shmem[{}]", smem_total)),
        "d_model=32 head_dim=128 PTX must use static .shared[{}], found:\n{}",
        smem_total,
        &ptx_str[..ptx_str.len().min(500)]
    );
    assert!(
        !ptx_str.contains(".extern .shared"),
        "d_model=32 head_dim=128 PTX must NOT use .extern .shared (fits static budget)"
    );
    eprintln!("[TrackB] PTX SMEM declaration: OK (static .shared[{}])", smem_total);

    ptx.push(0);  // NUL-terminate for cuModuleLoadData.
    let kernel_name = CString::new(flash_attention_kernel_name_selected(&config)).unwrap();
    let smem_dynamic = 0i64;  // static SMEM — pass 0 so runtime skips cuFuncSetAttribute

    // Allocate device memory.
    let x_elems   = batch * heads * seq * head_dim;
    let w_elems   = d_model * head_dim;
    let qkv_elems = batch * heads * seq * head_dim;
    let x_dev  = unsafe { nsl_test_cuda_alloc((x_elems   * 4) as i64) };
    let wq_dev = unsafe { nsl_test_cuda_alloc((w_elems    * 2) as i64) };
    let wk_dev = unsafe { nsl_test_cuda_alloc((w_elems    * 2) as i64) };
    let wv_dev = unsafe { nsl_test_cuda_alloc((w_elems    * 2) as i64) };
    let nw_dev = unsafe { nsl_test_cuda_alloc((head_dim   * 4) as i64) };
    let q_dev  = unsafe { nsl_test_cuda_alloc((qkv_elems  * 4) as i64) };
    let k_dev  = unsafe { nsl_test_cuda_alloc((qkv_elems  * 4) as i64) };
    let v_dev  = unsafe { nsl_test_cuda_alloc((qkv_elems  * 4) as i64) };
    let out_dev = unsafe { nsl_test_cuda_alloc((qkv_elems * 2) as i64) };
    let lse_dev = unsafe { nsl_test_cuda_alloc((batch * heads * seq * 4) as i64) };

    let ptrs = [x_dev, wq_dev, wk_dev, wv_dev, nw_dev, q_dev, k_dev, v_dev, out_dev, lse_dev];
    if !ptrs.iter().all(|&p| p != 0) {
        for &p in &ptrs { if p != 0 { unsafe { nsl_test_cuda_free(p); } } }
        panic!("[TrackB] device allocation failed");
    }

    // Upload minimal host data.
    let x_f32: Vec<f32> = (0..x_elems).map(|i| (i as f32) / x_elems as f32).collect();
    let w_f16: Vec<u16> = (0..w_elems)
        .map(|i| f32_to_f16(if i % (d_model + 1) == 0 { 1.0 } else { 0.01 }))
        .collect();
    let nw_f32 = vec![1.0f32; head_dim];
    unsafe {
        nsl_test_cuda_h2d(x_dev,  x_f32.as_ptr()  as i64, (x_elems * 4) as i64);
        nsl_test_cuda_h2d(wq_dev, w_f16.as_ptr()  as i64, (w_elems * 2) as i64);
        nsl_test_cuda_h2d(wk_dev, w_f16.as_ptr()  as i64, (w_elems * 2) as i64);
        nsl_test_cuda_h2d(wv_dev, w_f16.as_ptr()  as i64, (w_elems * 2) as i64);
        nsl_test_cuda_h2d(nw_dev, nw_f32.as_ptr() as i64, (head_dim * 4) as i64);
    }

    // Launch — this tests that head_dim=128 with d_model=32 launches cleanly.
    let rc = unsafe {
        nsl_flash_attention_csha(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, head_dim as i64,
            0, 0, 0, 0,    // paging: block_table, k_pool, v_pool, block_size
            0, 0,          // cos_ptr=0, sin_ptr=0 (identity)
            0, 0,          // seq_ids, seq_lens
            smem_dynamic,
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            block_q as i64, block_kv as i64,
            0i64,          // causal=false
            x_dev,
            nw_dev,
            wq_dev, wk_dev, wv_dev,
            0i64,          // wo_ptr=null
            norm_eps.to_bits() as i64,
            heads as i64,
            d_model as i64,
        )
    };

    for &p in &ptrs { unsafe { nsl_test_cuda_free(p); } }

    assert_eq!(rc, 0, "[TrackB] nsl_flash_attention_csha rc={rc} (expected 0 = CUDA_SUCCESS)");
    eprintln!("[TrackB] head_dim=128 d_model=32 launch: OK (rc=0)");
}

/// Minimal f32 -> f16 helper (round-to-nearest).
fn f32_to_f16(x: f32) -> u16 {
    if x.is_nan() { return 0x7E00; }
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp  = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 255 {
        return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16;
    }
    let exp16 = exp - 127 + 15;
    if exp16 <= 0 {
        let shift = (1 - exp16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        return ((sign << 15) | ((shifted + 0x1000) >> 13)) as u16;
    }
    if exp16 >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    let m16 = (mant + 0x1000) >> 13;
    let overflow = (m16 >> 10) & 1;
    ((sign << 15) | (((exp16 as u32 + overflow) & 0x1F) << 10) | (m16 & 0x3FF)) as u16
}
