//! Diagnostic: read back the fused-backward `dx_norm` HBM buffer directly.
//!
//! H1 (from PR #83): the kernel-side SMEM→HBM cooperative copy in
//! `emit_drmsnorm` produces an all-zero HBM buffer for the toy pretrain
//! shape (hd=32, seq=32, d_model=32). If dx_norm is zero at runtime, the
//! AD-side `RmsNormGammaBackward` reads zeros for its upstream dy and
//! emits a zero dgamma — hence `w_norm` stays stuck under the fused
//! backward even though all adjoint wiring is proven correct.
//!
//! This test launches the same forward-with-saves + fused backward pair
//! used by `csha_cuda_backward::t6_3_smoke_single_config`, then reads the
//! 8th gradient output (`dx_norm`, f32 [batch, seq, d_model]) via
//! `nsl_test_cuda_d2h` and inspects it.
//!
//! Assertions:
//!   - buffer length matches `batch * seq * d_model * 4 bytes`
//!   - all elements are finite (no NaN/Inf)
//!   - at least one element is non-zero (kernel wrote SOMETHING)
//!
//! Run with:
//!   cargo test -p nsl-codegen --features cuda \
//!     --test csha_dx_norm_readback_diag -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::ffi::CString;

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_backward,
    synthesize_flash_attention_ptx_v2,
    smem_layout::{self, needs_dynamic_smem, Direction},
};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h,
    nsl_test_cuda_free, nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use nsl_runtime::flash_attention::{
    nsl_csha_alloc_backward_activations, nsl_csha_free_backward_activations,
    nsl_flash_attention_csha_backward, nsl_flash_attention_csha_with_saves,
};

// ── Float helpers (clone of csha_cuda_backward.rs helpers) ─────────────────

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
    nsl_cuda_init() == 0
}

fn free_all(ptrs: &[i64]) {
    for &p in ptrs { if p != 0 { nsl_test_cuda_free(p); } }
}

fn backward_kernel_name(cfg: &FlashAttentionConfig) -> String {
    let fw = flash_attention_kernel_name_v2(cfg);
    match fw.strip_prefix("flash_attn_") {
        Some(rest) => format!("flash_attn_backward_{}", rest),
        None => format!("flash_attn_backward_{fw}"),
    }
}

#[test]
#[ignore]
fn dx_norm_hbm_buffer_is_populated() {
    if !cuda_available() {
        eprintln!("[dx_norm diag] skipping — no CUDA");
        return;
    }

    // Mirror the e2e smoke shape: batch=1, seq=32, heads=1, head_dim=32,
    // d_model=32, block_q=block_kv=32. Inputs are det_seq (not ones) to
    // avoid the degenerate numerical case where x=1 makes rms=1 and the
    // closed-form dx reduction happens to cancel.
    let batch = 1usize;
    let seq = 32usize;
    let h = 1usize;
    let hd = 32usize;
    let dm = 32usize;
    let block_q: u32 = 32;
    let block_kv: u32 = 32;
    let causal = false;
    let rope_q = false;
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let kv_dim = h * hd;

    let config = FlashAttentionConfig {
        block_q: block_q as i64,
        block_kv: block_kv as i64,
        head_dim: hd as i64,
        causal,
        paged: false,
        rope_q,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            fused_rmsnorm: true,
            fused_projections: true,
            fused_output_proj: false,
            save_activations_for_backward: true,
            active_heads: h as u32,
            rmsnorm_eps: norm_eps,
            d_model: dm as u32,
            // Tier B.1 narrow-and-chunkify pre-pass not used here — keep
            // the in-kernel RMSNorm prologue active (default).
            skip_rmsnorm_prologue: false,
        }),
    };

    smem_layout::validate_scalar_v2_config(&config, Direction::Backward)
        .expect("validator should accept toy smoke config");

    // ── Host data ────────────────────────────────────────────────────────
    //
    // Mirror the failing e2e smoke exactly: `x = ones([1,1,32,32])`,
    // `wq/wk/wv = ones([32,32])`, `w_norm = ones([32])`. Any
    // deterministic-random alternative would obscure whether the bug is
    // shape-dependent or value-dependent.
    let diag_input = std::env::var("NSL_DX_NORM_DIAG_INPUT")
        .unwrap_or_else(|_| "ones".into());
    let (x, wq_f32, wk_f32, wv_f32): (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) =
        if diag_input == "random" {
            (
                det_seq(42, h * seq * hd),
                det_seq(43, dm * kv_dim),
                det_seq(44, dm * kv_dim),
                det_seq(45, dm * kv_dim),
            )
        } else {
            (
                vec![1.0f32; h * seq * hd],
                vec![1.0f32; dm * kv_dim],
                vec![1.0f32; dm * kv_dim],
                vec![1.0f32; dm * kv_dim],
            )
        };
    eprintln!("[dx_norm diag] input mode: {diag_input}");
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let nw = vec![1.0f32; hd];
    let cos_f32: Vec<f32> = vec![1.0f32; seq * hd / 2];
    let sin_f32: Vec<f32> = vec![0.0f32; seq * hd / 2];
    let cos_f16: Vec<u16> = cos_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let sin_f16: Vec<u16> = sin_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_host_f32 = det_seq(99, seq * kv_dim);
    let do_f16: Vec<u16> = do_host_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();

    // ── GPU allocations ──────────────────────────────────────────────────
    let _ = nsl_cuda_init();
    let qkv_bytes = (h * seq * hd * 2) as i64;
    let lse_bytes = (batch * h * seq * 4) as i64;
    let x_bytes = (h * seq * hd * 4) as i64;
    let w_bytes = (dm * kv_dim * 2) as i64;
    let nw_bytes = (hd * 4) as i64;
    let rope_bytes = (seq * hd / 2 * 2) as i64;
    let dw_bytes = (dm * kv_dim * 2) as i64;
    let dx_bytes = (h * seq * hd * 4) as i64;
    // f32, shape [batch, seq, d_model] — under this toy scope that's
    // 1 * 32 * 32 = 1024 elements * 4 bytes = 4096 bytes.
    let dxn_bytes = (batch * seq * dm * 4) as i64;
    let expected_dxn_elems = batch * seq * dm;

    let q_dev = nsl_test_cuda_alloc(qkv_bytes);
    let k_dev = nsl_test_cuda_alloc(qkv_bytes);
    let v_dev = nsl_test_cuda_alloc(qkv_bytes);
    let out_dev = nsl_test_cuda_alloc(qkv_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);
    let x_dev = nsl_test_cuda_alloc(x_bytes);
    let nw_dev = nsl_test_cuda_alloc(nw_bytes);
    let wq_dev = nsl_test_cuda_alloc(w_bytes);
    let wk_dev = nsl_test_cuda_alloc(w_bytes);
    let wv_dev = nsl_test_cuda_alloc(w_bytes);
    let cos_dev = nsl_test_cuda_alloc(rope_bytes);
    let sin_dev = nsl_test_cuda_alloc(rope_bytes);
    let do_dev = nsl_test_cuda_alloc(qkv_bytes);
    let dq_dev = nsl_test_cuda_alloc(qkv_bytes);
    let dk_dev = nsl_test_cuda_alloc(qkv_bytes);
    let dv_dev = nsl_test_cuda_alloc(qkv_bytes);
    let dwq_dev = nsl_test_cuda_alloc(dw_bytes);
    let dwk_dev = nsl_test_cuda_alloc(dw_bytes);
    let dwv_dev = nsl_test_cuda_alloc(dw_bytes);
    let dx_dev = nsl_test_cuda_alloc(dx_bytes);
    let dxn_dev = nsl_test_cuda_alloc(dxn_bytes);

    // SAFETY: alloc_backward_activations is marked unsafe because it
    // returns raw device pointers; we own them for the rest of the test.
    let saves = unsafe {
        nsl_csha_alloc_backward_activations(
            batch as i64, h as i64, seq as i64, hd as i64,
        )
    };
    let all_dev = [
        q_dev, k_dev, v_dev, out_dev, lse_dev, x_dev, nw_dev,
        wq_dev, wk_dev, wv_dev, cos_dev, sin_dev,
        do_dev, dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev, dx_dev,
        dxn_dev,
    ];

    nsl_test_cuda_h2d(x_dev, x.as_ptr() as i64, x_bytes);
    nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, w_bytes);
    nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, w_bytes);
    nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, w_bytes);
    nsl_test_cuda_h2d(nw_dev, nw.as_ptr() as i64, nw_bytes);
    nsl_test_cuda_h2d(cos_dev, cos_f16.as_ptr() as i64, rope_bytes);
    nsl_test_cuda_h2d(sin_dev, sin_f16.as_ptr() as i64, rope_bytes);
    nsl_test_cuda_h2d(do_dev, do_f16.as_ptr() as i64, qkv_bytes);

    // ── Forward with saves ──────────────────────────────────────────────
    let fwd_ptx = synthesize_flash_attention_ptx_v2(&config);
    let fwd_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    let fwd_smem_total = smem_layout::total_bytes(&config);
    let fwd_smem_dyn = if needs_dynamic_smem(&config) { fwd_smem_total as i64 } else { 0 };

    let rc_fwd = nsl_flash_attention_csha_with_saves(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        batch as i64, h as i64, seq as i64, hd as i64,
        0, 0, 0, 0,
        cos_dev, sin_dev,
        0, 0,
        fwd_smem_dyn,
        fwd_ptx.as_ptr() as i64, fwd_name.as_ptr() as i64,
        block_q as i64, block_kv as i64,
        if causal { 1 } else { 0 },
        x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
        0, norm_eps.to_bits() as i64,
        h as i64, dm as i64,
        saves.q_proj, saves.k_proj, saves.v_proj,
        saves.row_max, saves.row_sum,
        saves.x_raw,
        // PCA Tier A: segment_ids ptr (trailing) — 0 = unpacked launch.
        0i64,
        // Tier B extension — null (no Tier B dispatch for diag test).
        0i64, 0i64,
        // doc_starts ptr — null (no doc-aware RoPE for diag test).
        0i64,
        // PCA per-doc CTA Strategy 3 v1: num_docs_or_zero — 0 (legacy topology).
        0i64,
    );
    assert_eq!(rc_fwd, 0, "forward-with-saves rc={rc_fwd}");

    // ── Backward ────────────────────────────────────────────────────────
    let mut bwd_ptx_str = synthesize_backward(&config)
        .expect("synth backward should succeed");
    if !bwd_ptx_str.ends_with('\0') { bwd_ptx_str.push('\0'); }
    let bwd_ptx = bwd_ptx_str.into_bytes();
    let bwd_name = CString::new(backward_kernel_name(&config)).unwrap();

    let rc_bwd = nsl_flash_attention_csha_backward(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        batch as i64, h as i64, seq as i64, hd as i64,
        0, 0, 0, 0,
        cos_dev, sin_dev,
        0, 0,
        0,
        bwd_ptx.as_ptr() as i64, bwd_name.as_ptr() as i64,
        block_q as i64, block_kv as i64,
        if causal { 1 } else { 0 },
        x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
        0, norm_eps.to_bits() as i64,
        h as i64, dm as i64,
        saves.q_proj, saves.k_proj, saves.v_proj,
        saves.row_max, saves.row_sum,
        saves.x_raw,
        do_dev, dq_dev, dk_dev, dv_dev,
        dwq_dev, dwk_dev, dwv_dev, dx_dev,
        dxn_dev,
        // PCA Task 4B: trailing segment_ids — 0 = unpacked launch.
        0i64,
        // Tier B extension — null (no Tier B dispatch for diag test).
        0i64, 0i64,
        // doc_starts ptr — null (no doc-aware RoPE for diag test).
        0i64,
        // PCA per-doc CTA backward (Sprint 5): num_docs_or_zero — 0
        // means legacy per-q-block topology.
        0i64,
    );
    if rc_bwd != 0 {
        let log = unsafe {
            let p = nsl_test_cuda_jit_log(bwd_ptx.as_ptr() as i64);
            if p != 0 {
                std::ffi::CStr::from_ptr(p as *const i8)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "<no log>".into()
            }
        };
        unsafe { nsl_csha_free_backward_activations(saves); }
        free_all(&all_dev);
        panic!("backward rc={rc_bwd}\nJIT log:\n{log}");
    }

    // ── Read dx_norm back to host ───────────────────────────────────────
    let mut dxn_host = vec![0f32; expected_dxn_elems];
    nsl_test_cuda_d2h(
        dxn_host.as_mut_ptr() as i64,
        dxn_dev,
        (expected_dxn_elems * 4) as i64,
    );

    // ── Inspection ──────────────────────────────────────────────────────
    let sum: f64 = dxn_host.iter().map(|&v| v as f64).sum();
    let max_abs: f32 = dxn_host
        .iter()
        .map(|&v| v.abs())
        .fold(0f32, f32::max);
    let nonzero_count = dxn_host.iter().filter(|&&v| v != 0.0).count();
    let first8: Vec<f32> = dxn_host.iter().take(8).copied().collect();

    eprintln!(
        "[dx_norm diag] shape=[batch={batch}, seq={seq}, d_model={dm}] \
         elems={expected_dxn_elems} sum={sum:.6} max_abs={max_abs:.6e} \
         nonzero={nonzero_count}/{expected_dxn_elems}"
    );
    eprintln!("[dx_norm diag] first 8 values: {first8:?}");
    if nonzero_count < 16 && nonzero_count > 0 {
        // Show which indices hold non-zero values for a zero-sparsity
        // diagnosis — if only a handful of cells were written, that
        // narrows H1 to an offset/predicate issue rather than a
        // whole-buffer dropout.
        let locs: Vec<(usize, f32)> = dxn_host
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, &v)| (i, v))
            .collect();
        eprintln!("[dx_norm diag] non-zero positions: {locs:?}");
    }

    unsafe { nsl_csha_free_backward_activations(saves); }
    free_all(&all_dev);

    // ── Assertions ──────────────────────────────────────────────────────
    assert_eq!(
        dxn_host.len(),
        expected_dxn_elems,
        "dxn buffer length mismatch: {} vs {expected_dxn_elems}",
        dxn_host.len()
    );
    for (i, &v) in dxn_host.iter().enumerate() {
        assert!(v.is_finite(), "dx_norm[{i}] = {v} not finite");
    }
    assert!(
        nonzero_count > 0,
        "H1 CONFIRMED: dx_norm HBM buffer is entirely zero after backward \
         launch. Kernel-side SMEM->HBM cooperative copy in emit_drmsnorm \
         is broken (offset, predicate, sync, or SMEM source region)."
    );
}
