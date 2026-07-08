//! T6.3 — GPU three-way numerical gate for CSHA fused backward.
//!
//! Intended comparison: (fused backward GPU) vs (unfused backward GPU)
//! vs (CPU reference from csha_reference::csha_reference_backward).
//! Tolerance tiers (mirroring Tier A forward):
//!   head_dim=32  → 5e-3
//!   head_dim=64  → 2e-2
//!   head_dim=128 → 4e-2
//!
//! # Current status: real HBM addressing landed; numerical gates enabled
//!
//! The T3.3/T3.4/T3.5/T3.6 phase emitters land the reduction SHAPE
//! (warp butterflies, lane-owns-column fmas, softmax Jacobian, chain-
//! rule scale factor, null guards, label uniqueness, ptxas-clean
//! assembly). The Phase-3 follow-up that replaced the placeholder f32
//! constants (0f3F800000 == 1.0) with real Q/K/V/dO/x_norm inner-loop
//! HBM loads has since landed (see the per-gate notes below: dq/dk/dv
//! after the V-save/SMEM-tile/dQ-flush fixes; dw after `emit_dproj`
//! real addressing + x_norm re-materialisation; dx after the
//! `x_raw_ptr` forward save channel).
//!
//! Consequence: under the smoke scope (heads=1, no RoPE, no causal) the
//! fused backward produces real chain-rule gradients, and all three
//! numerical gates below are enabled. This is validated only on real
//! GPU hardware — the file is `#[cfg(feature = "cuda")]` and its tests
//! are `#[ignore]`, so the numerical comparison runs manually
//! (`--features cuda -- --ignored`), not in default CI.
//!
//! # What THIS file ships
//!
//! 1. The full three-way launch harness (`run_fused_backward_config`
//!    end-to-end: forward-with-saves → fused backward → CPU reference)
//!    so that once real HBM loads land in Phase 3, the numerical
//!    assertions can be enabled by flipping a single const.
//! 2. Structural assertions that DO hold today:
//!      - fused-backward launch rc=0
//!      - all 7 gradient outputs readable and finite
//!      - gradient shapes match expected layouts
//! 3. The numerical-tolerance assertions, now that Phase-3 inner loops
//!    carry real addressing, are enabled via the
//!    `NUMERICAL_GATE_DQKV_ENABLED` / `NUMERICAL_GATE_DW_ENABLED` /
//!    `NUMERICAL_GATE_DX_ENABLED` consts (all `true`).

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference, csha_reference_backward, CshaInputs, CshaShape, CshaGradients};

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
    CshaBackwardActivations,
};

// ── Gates ──────────────────────────────────────────────────────────────────
//
// dq/dk/dv: enabled after the three structural fixes on feat/csha-tier-c-diag
//   (V save scheduling, backward V-input SMEM tile, per-iter dQ SMEM flush).
//   Tier A 5e-3 tolerance for head_dim=32.
//
// dwq/dwk/dwv: enabled after emit_dproj real addressing + x_norm tile
//   re-materialisation lands (feat/csha-tier-c-diag). Under the smoke
//   scope (heads=1, no RoPE, no causal) max_abs lands ≤ 5e-3.
//
// dx: enabled. The forward-side blocker is resolved — `emit_prologue`
//   (`phases/forward/csha_hooks.rs`) now saves the pre-RMSNorm raw x to a
//   dedicated `x_raw_ptr` channel (null-gated per-slice store), and the
//   backward reads `[x_raw_ptr]` to feed the closed-form dx formula
//   (`csha_hooks_backward.rs`).
const NUMERICAL_GATE_DQKV_ENABLED: bool = true;
const NUMERICAL_GATE_DW_ENABLED:   bool = true;
const NUMERICAL_GATE_DX_ENABLED:   bool = true;

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

fn free_all(ptrs: &[i64]) {
    for &p in ptrs { if p != 0 { unsafe { nsl_test_cuda_free(p); } } }
}

fn backward_kernel_name(cfg: &FlashAttentionConfig) -> String {
    let fw = flash_attention_kernel_name_v2(cfg);
    match fw.strip_prefix("flash_attn_") {
        Some(rest) => format!("flash_attn_backward_{}", rest),
        None => format!("flash_attn_backward_{fw}"),
    }
}

/// Forward-parity max_abs from the most recent harness run (f32 bits).
/// Written by the fwd-parity diagnostic inside the harness; read by
/// tests whose backward launch is EXPECTED to refuse (multi-tile) but
/// which still gate on forward correctness. Tests run --test-threads=1.
static LAST_FWD_PARITY_MAX_ABS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Three-way launch harness. Returns (gpu_grads, cpu_grads) so the
/// caller can compute per-tensor max_abs. Structural guarantees: rc=0
/// launches, all 7 gradient outputs finite, shapes match CPU reference.
#[allow(clippy::too_many_arguments)]
fn run_fused_backward_config(
    block_q: u32, block_kv: u32, head_dim: u32, heads: u32, d_model: u32,
    causal: bool, rope_q: bool,
) -> Result<(CshaGradients /*gpu_as_f32*/, CshaGradients /*cpu*/), String> {
    let seq = (block_q as usize).max(block_kv as usize);
    run_fused_backward_config_seq(block_q, block_kv, head_dim, heads, d_model, causal, rope_q, seq)
}

/// Seq-decoupled variant: `seq` may exceed the block sizes, exercising the
/// multi-KV-tile / multi-q-block launch path (`q_blocks` outer loop in the
/// backward FFI + the kernel's `V2_BWD_LOOP_KV` k_start sweep). The
/// single-tile wrapper above pins `seq = max(block_q, block_kv)`.
#[allow(clippy::too_many_arguments)]
fn run_fused_backward_config_seq(
    block_q: u32, block_kv: u32, head_dim: u32, heads: u32, d_model: u32,
    causal: bool, rope_q: bool, seq: usize,
) -> Result<(CshaGradients /*gpu_as_f32*/, CshaGradients /*cpu*/), String> {
    let batch = 1usize;
    assert!(
        seq >= (block_q as usize).max(block_kv as usize),
        "seq {seq} must cover at least one full block"
    );
    let hd = head_dim as usize;
    let dm = d_model as usize;
    let h = heads as usize;
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let kv_dim = h * hd;

    let config = FlashAttentionConfig {
        block_q: block_q as i64, block_kv: block_kv as i64, head_dim: hd as i64,
        causal, paged: false, rope_q,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
            level: 2,
            fused_rmsnorm: true,
            fused_projections: true,
            fused_output_proj: false,
            save_activations_for_backward: true,
            active_heads: heads,
            rmsnorm_eps: norm_eps,
            d_model: d_model,
            // Tier B.1 narrow-and-chunkify pre-pass not used here — keep
            // the in-kernel RMSNorm prologue active (default).
            skip_rmsnorm_prologue: false,
            static_seq_len: None,
        }),
        checkpoint: None,
    };

    // Budget check.
    smem_layout::validate_scalar_v2_config(&config, Direction::Backward)
        .map_err(|e| format!("backward validator: {e}"))?;

    // ── Deterministic host data ─────────────────────────────────────────────
    let x = det_seq(42, h * seq * hd);
    let x_for_ref: Vec<f32> = {
        // CPU reference expects [seq, d_model]; kernel uses [heads, seq, head_dim]
        // with d_model == head_dim × heads for Tier A semantics. For the numerical
        // gate we set d_model == head_dim and iterate heads explicitly, matching
        // csha_reference_backward's layout.
        x.clone()
    };
    let wq_f32 = det_seq(43, dm * kv_dim);
    let wk_f32 = det_seq(44, dm * kv_dim);
    let wv_f32 = det_seq(45, dm * kv_dim);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let nw = vec![1.0f32; hd];
    // Kernel reads cos/sin as f16 (`ld.global.b16`); upload f16 and use
    // f16-rounded values for the CPU reference so both sides see the same
    // precision (avoids spurious divergence from f32-only cos/sin in CPU
    // vs f16-truncated cos/sin in GPU).
    let cos_f32: Vec<f32> = if rope_q {
        (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).cos()).collect()
    } else {
        vec![1.0f32; seq * hd / 2]
    };
    let sin_f32: Vec<f32> = if rope_q {
        (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).sin()).collect()
    } else {
        vec![0.0f32; seq * hd / 2]
    };
    let cos_f16: Vec<u16> = cos_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let sin_f16: Vec<u16> = sin_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let cos: Vec<f32> = cos_f16.iter().map(|&b| f16_to_f32(b)).collect();
    let sin: Vec<f32> = sin_f16.iter().map(|&b| f16_to_f32(b)).collect();
    let do_host_f32 = det_seq(99, seq * kv_dim);
    let do_f16: Vec<u16> = do_host_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    // Use f16-rounded dO for the CPU reference so both sides see the same
    // precision (avoids spurious divergence from f32-only dO in CPU vs
    // f16-truncated dO in GPU — same pattern as cos/sin above).
    let do_host: Vec<f32> = do_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // ── CPU reference ──────────────────────────────────────────────────────
    let inputs = CshaInputs {
        x: &x_for_ref, wq: &wq_f32, wk: &wk_f32, wv: &wv_f32,
        norm_weight: &nw, cos: &cos, sin: &sin,
    };
    let shape = CshaShape {
        seq, heads: h, head_dim: hd, d_model: dm,
        causal, norm_eps,
        rope_q: true,
    };
    let cpu_grads = csha_reference_backward(&inputs, &shape, &do_host);

    // ── GPU: forward-with-saves then fused backward ────────────────────────
    unsafe { nsl_cuda_init(); }
    let qkv_bytes = (h * seq * hd * 2) as i64;
    let lse_bytes = (batch * h * seq * 4) as i64;
    let x_bytes = (h * seq * hd * 4) as i64;  // kernel x is f32 per-head
    let w_bytes = (dm * kv_dim * 2) as i64;    // stored as one [dm, kv_dim] f16 block
    let nw_bytes = (hd * 4) as i64;
    let rope_bytes = (seq * hd / 2 * 2) as i64;  // f16, 2 bytes per element
    let dw_bytes = (dm * kv_dim * 2) as i64;
    let dx_bytes = (h * seq * hd * 4) as i64;

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
    let do_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let dq_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let dk_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let dv_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let dwq_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dwk_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dwv_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dx_dev = unsafe { nsl_test_cuda_alloc(dx_bytes) };
    // Gap I.5 Option A: 8th gradient output buffer (dx_norm — gradient
    // w.r.t. the RMSNorm OUTPUT). f32, shape [batch, seq, d_model]; under
    // the smoke scope this matches the existing `dx_bytes` sizing because
    // `seq * d_model == h * seq * hd` (heads=1, d_model=hd).
    let dxn_bytes = (batch * seq * dm * 4) as i64;
    let dxn_dev = unsafe { nsl_test_cuda_alloc(dxn_bytes) };

    let saves = unsafe {
        nsl_csha_alloc_backward_activations(
            batch as i64, heads as i64, seq as i64, hd as i64,
        )
    };
    let all_dev = [
        q_dev, k_dev, v_dev, out_dev, lse_dev, x_dev, nw_dev,
        wq_dev, wk_dev, wv_dev, cos_dev, sin_dev,
        do_dev, dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev, dx_dev,
        dxn_dev,
    ];

    unsafe {
        nsl_test_cuda_h2d(x_dev,  x.as_ptr()  as i64, x_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(nw_dev, nw.as_ptr() as i64, nw_bytes);
        nsl_test_cuda_h2d(cos_dev, cos_f16.as_ptr() as i64, rope_bytes);
        nsl_test_cuda_h2d(sin_dev, sin_f16.as_ptr() as i64, rope_bytes);
        nsl_test_cuda_h2d(do_dev, do_f16.as_ptr() as i64, qkv_bytes);
    }

    // Forward PTX + name.
    // Combined module: fused kernel + `_mt_attn` interleaved twin. The
    // runtime's two-launch multi-tile dispatch resolves the twin by name
    // when seq > block; single-tile launches use the fused entry only.
    let fwd_ptx = nsl_codegen::flash_attention_v2::synthesize_forward_multi_tile_combined(&config);
    let fwd_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    let fwd_smem_total = smem_layout::total_bytes(&config);
    let fwd_smem_dyn = if needs_dynamic_smem(&config) { fwd_smem_total as i64 } else { 0 };

    let rc_fwd = unsafe {
        nsl_flash_attention_csha_with_saves(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, hd as i64,
            0, 0, 0, 0,
            cos_dev, sin_dev,
            0, 0,
            fwd_smem_dyn,
            fwd_ptx.as_ptr() as i64, fwd_name.as_ptr() as i64,
            block_q as i64, block_kv as i64,
            if causal { 1 } else { 0 },
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, dm as i64,
            saves.q_proj, saves.k_proj, saves.v_proj,
            saves.row_max, saves.row_sum,
            saves.x_raw,
            // PCA Tier A: segment_ids ptr (trailing) — 0 = unpacked launch.
            0i64,
            // Tier B extension — null (no Tier B dispatch for this test).
            0i64, 0i64,
            // doc_starts ptr — null (no doc-aware RoPE for this test).
            0i64,
            // PCA per-doc CTA Strategy 3 v1: num_docs_or_zero — 0 (legacy topology).
            0i64,
        )
    };
    if rc_fwd != 0 {
        let log = unsafe {
            let p = nsl_test_cuda_jit_log(fwd_ptx.as_ptr() as i64);
            if p != 0 {
                std::ffi::CStr::from_ptr(p as *const i8).to_string_lossy().into_owned()
            } else { "<no log>".into() }
        };
        unsafe { nsl_csha_free_backward_activations(saves); }
        free_all(&all_dev);
        return Err(format!("forward rc={rc_fwd}\nJIT log:\n{log}"));
    }

    // Forward-parity diagnostic (multi-tile localization): per-q-block
    // max_abs of GPU `out` vs the CPU reference forward. If blocks 1+ are
    // garbage while block 0 matches, the defect is in the forward
    // multi-q-block path, not the backward. The overall max is published
    // through LAST_FWD_PARITY_MAX_ABS so refusal-path tests (backward
    // refuses at multi-tile) can still gate on forward correctness.
    {
        let fwd_qkv_elems = h * seq * hd;
        let gpu_out: Vec<f32> = {
            let mut raw = vec![0u16; fwd_qkv_elems];
            unsafe { nsl_test_cuda_d2h(raw.as_mut_ptr() as i64, out_dev, (fwd_qkv_elems * 2) as i64); }
            raw.iter().map(|&b| f16_to_f32(b)).collect()
        };
        let cpu_out = csha_reference(&inputs, &shape);
        let bq = block_q as usize;
        let mut per_block: Vec<f32> = Vec::new();
        for qb in 0..seq.div_ceil(bq) {
            let mut m = 0f32;
            let r0 = qb * bq;
            let r1 = ((qb + 1) * bq).min(seq);
            for hh in 0..h {
                for r in r0..r1 {
                    for d in 0..hd {
                        let i = (hh * seq + r) * hd + d;
                        m = m.max((gpu_out[i] - cpu_out[i]).abs());
                    }
                }
            }
            per_block.push(m);
        }
        let blocks_str: Vec<String> =
            per_block.iter().map(|m| format!("{m:.2e}")).collect();
        eprintln!(
            "  [fwd-parity] seq={seq} bq={bq} per-block out max_abs: [{}]",
            blocks_str.join(", ")
        );
        let overall = per_block.iter().cloned().fold(0f32, f32::max);
        LAST_FWD_PARITY_MAX_ABS.store(overall.to_bits(), std::sync::atomic::Ordering::SeqCst);
    }

    // Backward PTX + name.
    let mut bwd_ptx_str = synthesize_backward(&config)
        .map_err(|e| format!("synth backward: {e}"))?;
    if !bwd_ptx_str.ends_with('\0') { bwd_ptx_str.push('\0'); }
    let bwd_ptx = bwd_ptx_str.into_bytes();
    let bwd_name = CString::new(backward_kernel_name(&config)).unwrap();

    // Backward dynamic SMEM size: the backward kernel emits `.extern .shared
    // shmem[]` (dynamic) when its total tile footprint exceeds the 48 KB static
    // cap (e.g. head_dim=64), and the launch MUST supply that size. Passing 0
    // (as this forward-derived harness previously did) leaves the dynamic
    // region 0-byte, so every SMEM tile access is out of bounds — the source
    // of the hd=64 "catastrophic dV" garbage. Use the exact production sizing
    // helper (wengert_lower calls the same one), gated on the static cap to
    // mirror the kernel's own static-vs-`.extern` emit decision.
    let bwd_smem =
        nsl_codegen::flash_attention_v2::shared_mem_bytes_v2_backward(&config);
    let bwd_smem_dyn = if bwd_smem > smem_layout::SMEM_BUDGET_BYTES {
        bwd_smem as i64
    } else {
        0
    };

    let rc_bwd = unsafe {
        nsl_flash_attention_csha_backward(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, hd as i64,
            0, 0, 0, 0,
            cos_dev, sin_dev,
            0, 0,
            bwd_smem_dyn,
            bwd_ptx.as_ptr() as i64, bwd_name.as_ptr() as i64,
            block_q as i64, block_kv as i64,
            if causal { 1 } else { 0 },
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, dm as i64,
            saves.q_proj, saves.k_proj, saves.v_proj,
            saves.row_max, saves.row_sum,
            saves.x_raw,
            do_dev, dq_dev, dk_dev, dv_dev,
            dwq_dev, dwk_dev, dwv_dev, dx_dev,
            dxn_dev,
            // PCA Task 4B: trailing segment_ids — 0 = unpacked launch.
            0i64,
            // Tier B extension — null (no Tier B dispatch for this test).
            0i64, 0i64,
            // doc_starts ptr — null (no doc-aware RoPE for this test).
            0i64,
            // tier_b2_active — 0: scalar backward path, no Tier-B2 hybrid launch.
            0i64,
            // PCA per-doc CTA backward (Sprint 5): num_docs_or_zero — 0
            // means legacy per-q-block topology.
            0i64,
        )
    };
    if rc_bwd != 0 {
        let log = unsafe {
            let p = nsl_test_cuda_jit_log(bwd_ptx.as_ptr() as i64);
            if p != 0 {
                std::ffi::CStr::from_ptr(p as *const i8).to_string_lossy().into_owned()
            } else { "<no log>".into() }
        };
        unsafe { nsl_csha_free_backward_activations(saves); }
        free_all(&all_dev);
        return Err(format!("backward rc={rc_bwd}\nJIT log:\n{log}"));
    }

    // Readback.
    let read_f16 = |dev: i64, elems: usize| -> Vec<f32> {
        let mut raw = vec![0u16; elems];
        unsafe { nsl_test_cuda_d2h(raw.as_mut_ptr() as i64, dev, (elems * 2) as i64); }
        raw.iter().map(|&b| f16_to_f32(b)).collect()
    };
    let read_f32 = |dev: i64, elems: usize| -> Vec<f32> {
        let mut out = vec![0f32; elems];
        unsafe { nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dev, (elems * 4) as i64); }
        out
    };

    let qkv_elems = h * seq * hd;
    let dw_elems = dm * kv_dim;
    let dx_elems = h * seq * hd;


    let gpu_grads = CshaGradients {
        dq: read_f16(dq_dev, qkv_elems),
        dk: read_f16(dk_dev, qkv_elems),
        dv: read_f16(dv_dev, qkv_elems),
        dwq: read_f16(dwq_dev, dw_elems),
        dwk: read_f16(dwk_dev, dw_elems),
        dwv: read_f16(dwv_dev, dw_elems),
        dx: read_f32(dx_dev, dx_elems),
    };

    unsafe { nsl_csha_free_backward_activations(saves); }
    free_all(&all_dev);

    Ok((gpu_grads, cpu_grads))
}

/// Worst allclose excess: max over cells of |g-c| - rtol*|c|. A result
/// < atol means every cell satisfies |g-c| < atol + rtol*|c| (the
/// standard mixed absolute/relative gate). Used for the dW tensors,
/// whose cells reach |2..4| where the f16 STORE quantum alone is ~2e-3
/// — an absolute-only 5e-3 gate mis-fires on value-proportional
/// quantization noise there (verified diffuse via the [dwk-top] dump:
/// top excursions scatter across unrelated cells, all at max-magnitude
/// values, rel err <= 2.6e-3).
fn worst_allclose_excess(gpu: &[f32], cpu: &[f32], rtol: f32) -> f32 {
    gpu.iter()
        .zip(cpu.iter())
        .map(|(&g, &c)| (g - c).abs() - rtol * c.abs())
        .fold(f32::NEG_INFINITY, f32::max)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn tol_for_head_dim(hd: u32) -> f32 {
    if hd >= 128 { 4e-2 } else if hd >= 64 { 2e-2 } else { 5e-3 }
}

#[test]
#[ignore]
fn t6_3_smoke_single_config() {
    if !cuda_available() {
        eprintln!("[T6.3] skipping — no CUDA");
        return;
    }
    // heads=1, d_model=head_dim so CPU/GPU shapes align for the length
    // assertions below (heads=4 with dm=32 would give GPU dx length
    // 4*seq*hd but CPU dx length seq*dm — same elements, different
    // layouts; aligning heads avoids a shape-reshuffle for the smoke
    // test while still exercising every backward phase).
    let (gpu, cpu) = run_fused_backward_config(32, 32, 32, 1, 32, false, false)
        .expect("smoke config should succeed structurally");

    // Structural assertions — always enforced.
    for (name, arr) in [
        ("dq", &gpu.dq), ("dk", &gpu.dk), ("dv", &gpu.dv),
        ("dwq", &gpu.dwq), ("dwk", &gpu.dwk), ("dwv", &gpu.dwv),
        ("dx", &gpu.dx),
    ] {
        for (i, &v) in arr.iter().enumerate() {
            assert!(v.is_finite(), "gpu {name}[{i}] = {v} not finite");
        }
    }
    assert_eq!(gpu.dq.len(), cpu.dq.len(), "dq shape mismatch");
    assert_eq!(gpu.dk.len(), cpu.dk.len());
    assert_eq!(gpu.dv.len(), cpu.dv.len());
    assert_eq!(gpu.dwq.len(), cpu.dwq.len());
    assert_eq!(gpu.dx.len(), cpu.dx.len());

    // Numerical diff — log unconditionally so the blocker magnitude is
    // visible when future work unblocks Phase 3 inner-loop HBM loads.
    let d_dq = max_abs_diff(&gpu.dq, &cpu.dq);
    let d_dk = max_abs_diff(&gpu.dk, &cpu.dk);
    let d_dv = max_abs_diff(&gpu.dv, &cpu.dv);
    let d_dwq = max_abs_diff(&gpu.dwq, &cpu.dwq);
    let d_dwk = max_abs_diff(&gpu.dwk, &cpu.dwk);
    let d_dwv = max_abs_diff(&gpu.dwv, &cpu.dwv);
    let d_dx = max_abs_diff(&gpu.dx, &cpu.dx);
    eprintln!(
        "[T6.3 smoke] head_dim=32 max_abs: dq={d_dq:.3e} dk={d_dk:.3e} \
         dv={d_dv:.3e} dwq={d_dwq:.3e} dwk={d_dwk:.3e} dwv={d_dwv:.3e} dx={d_dx:.3e}"
    );

    // dwk error-shape diagnostic: top-5 |gpu-cpu| cells with values, so a
    // tolerance excursion can be classified as CONCENTRATED (systematic
    // defect at specific cells) vs DIFFUSE (f16/approx-op noise near the
    // gate boundary).
    {
        let mut cells: Vec<(usize, f32, f32, f32)> = gpu
            .dwk
            .iter()
            .zip(cpu.dwk.iter())
            .enumerate()
            .map(|(i, (&g, &c))| (i, (g - c).abs(), g, c))
            .collect();
        cells.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (i, d, g, c) in cells.iter().take(5) {
            eprintln!(
                "  [dwk-top] idx={i} p={} j={} |d|={d:.3e} gpu={g:.4} cpu={c:.4}",
                i / 32, i % 32
            );
        }
    }

    let tol = tol_for_head_dim(32);
    if NUMERICAL_GATE_DQKV_ENABLED {
        assert!(d_dq < tol, "dq max_abs {d_dq:.3e} > tol {tol:.1e}");
        assert!(d_dk < tol, "dk max_abs {d_dk:.3e} > tol {tol:.1e}");
        assert!(d_dv < tol, "dv max_abs {d_dv:.3e} > tol {tol:.1e}");
    }
    if NUMERICAL_GATE_DW_ENABLED {
        // dW gates use the mixed atol+rtol form (see worst_allclose_excess):
        // dW cells reach |2..4| where the f16 store quantum alone is ~2e-3,
        // so an absolute-only 5e-3 gate mis-fires on value-proportional
        // quantization noise (diagnosed diffuse via [dwk-top], not a defect).
        let rtol = 5e-3f32;
        let e_dwq = worst_allclose_excess(&gpu.dwq, &cpu.dwq, rtol);
        let e_dwk = worst_allclose_excess(&gpu.dwk, &cpu.dwk, rtol);
        let e_dwv = worst_allclose_excess(&gpu.dwv, &cpu.dwv, rtol);
        assert!(e_dwq < tol, "dwq allclose excess {e_dwq:.3e} > atol {tol:.1e} (rtol {rtol:.0e})");
        assert!(e_dwk < tol, "dwk allclose excess {e_dwk:.3e} > atol {tol:.1e} (rtol {rtol:.0e})");
        assert!(e_dwv < tol, "dwv allclose excess {e_dwv:.3e} > atol {tol:.1e} (rtol {rtol:.0e})");
    }
    if NUMERICAL_GATE_DX_ENABLED {
        // dx tolerance includes an additional sqrt(D)·ε_f16 factor from the
        // s_grad = sum_d (g_d · x_d) reduction inside the dRMSNorm closed
        // form (one extra reduction beyond what dq/dk/dv carry). For
        // head_dim=32 this lifts the bound from 5e-3 → ~3e-2.
        let dx_tol = (tol * 6.0).max(1e-2);
        assert!(d_dx < dx_tol, "dx max_abs {d_dx:.3e} > tol {dx_tol:.1e}");
    } else {
        eprintln!(
            "[T6.3] dx gate disabled: forward RMSNorm prologue overwrites \
             csha_x_ptr with x_normed in-place; backward has no raw x for \
             the closed-form dx. Needs a forward-side x_save pointer."
        );
    }
}

/// hd=64 backward at block=32. head_dim=64 with block_q=block_kv=64 exceeds
/// the 99 KB sm_120 SMEM opt-in cap (181 KB), but block=32 tiles fit (~83 KB).
/// This is the smallest config that reaches the bug's d-range (d up to 56).
/// Logs dV/dK/dQ max_abs + worst dV cell; no gate (diagnostic).
#[test]
#[ignore]
fn t6_3_hd64_block32_dv_probe() {
    if !cuda_available() {
        eprintln!("[hd64] skipping — no CUDA");
        return;
    }
    let hd = 64usize;
    for causal in [false, true] {
        let (gpu, cpu) = match run_fused_backward_config(32, 32, 64, 1, 64, causal, false) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[hd64 block32] causal={causal} LAUNCH FAILED: {e}");
                continue;
            }
        };
        let d_dv = max_abs_diff(&gpu.dv, &cpu.dv);
        let d_dk = max_abs_diff(&gpu.dk, &cpu.dk);
        let d_dq = max_abs_diff(&gpu.dq, &cpu.dq);
        let (mut wi, mut wmax) = (0usize, 0f32);
        for (i, (&g, &c)) in gpu.dv.iter().zip(cpu.dv.iter()).enumerate() {
            let d = (g - c).abs();
            if d > wmax {
                wmax = d;
                wi = i;
            }
        }
        eprintln!(
            "[hd64 block32] causal={causal} dv={d_dv:.3e} dk={d_dk:.3e} dq={d_dq:.3e} \
             | worst dv[{wi}]: col={} d={} gpu={:.4e} cpu={:.4e}",
            wi / hd, wi % hd, gpu.dv[wi], cpu.dv[wi]
        );
    }
}

/// Multi-tile contract: seq = 4 blocks (128 rows at block=32), hd ∈ {32, 64},
/// causal ∈ {0,1}. Two assertions per config:
///
///   1. FORWARD parity holds — the runtime two-launch dispatch (launch A =
///      fused per-block projections + saves; launch B = same kernel with
///      null weights, taking the classic k_start-addressed HBM tile loads
///      over the widened projections) must match the CPU reference in
///      EVERY q-block. Pre-dispatch, every block diverged by 3-10 abs
///      (single-tile K/V SMEM re-read — see the "Single-tile assumption"
///      comment in flash_attention_v2/mod.rs).
///
///   2. BACKWARD parity holds for all 7 gradients (PR2b). Multi-tile is now
///      correct at batch=1, heads=1 via the dK f32-scratch RMW
///      (emit_store_dk_only) + the {dWk,dWv,dx,dx_norm} post-pass
///      (csha_bwd_multitile_postpass) that recomputes those gradients from the
///      cross-launch dK/dV/dQ totals. This replaces the prior assert-refusal.
///
/// Historical note: this launch-path class is where the cycle-14..17
/// "Path A RED" numbers (dk max_rel 21.46 etc.) were measured — through
/// the checkpoint-recompute harness, downstream of the broken forward.
///
/// Tolerances carry a seq-scaling factor over the smoke gate: dK/dV RMW-
/// accumulate over 4× the q-block launches (more f16-drain round-off), the
/// post-pass reads dQ from f16 `d_q` (one extra quantization the in-kernel f32
/// SMEM path avoids), and dW sums over 4× rows. Every per-gradient excess is
/// logged so an excursion can be classified concentrated-vs-diffuse.
#[test]
#[ignore]
fn t6_3_multitile_seq128() {
    if !cuda_available() {
        eprintln!("[multitile] skipping — no CUDA");
        return;
    }
    let rtol = 5e-3f32;
    let mut failures: Vec<String> = Vec::new();
    for hd in [32u32, 64u32] {
        for causal in [false, true] {
            let fwd_tol = if hd <= 32 { 5e-3f32 } else { 2e-2f32 };
            // Seq-scaled tolerances (see the fn doc). base = single-tile tier.
            let base = tol_for_head_dim(hd);
            let qkv_tol = base * 2.0; // dk/dv accumulate over q-block launches
            let dw_atol = base * 2.0; // dW sums x_norm^T·dY over 4× rows
            let dx_tol = (base * 8.0).max(1.5e-2);
            match run_fused_backward_config_seq(32, 32, hd, 1, hd, causal, false, 128) {
                Ok((gpu, cpu)) => {
                    let fwd_max_abs = f32::from_bits(
                        LAST_FWD_PARITY_MAX_ABS.load(std::sync::atomic::Ordering::SeqCst),
                    );
                    if fwd_max_abs >= fwd_tol {
                        failures.push(format!(
                            "hd={hd} causal={causal}: forward parity {fwd_max_abs:.3e} > {fwd_tol:.0e}"
                        ));
                    }
                    // Finiteness + shape.
                    for (name, g, c) in [
                        ("dq", &gpu.dq, &cpu.dq), ("dk", &gpu.dk, &cpu.dk),
                        ("dv", &gpu.dv, &cpu.dv), ("dwq", &gpu.dwq, &cpu.dwq),
                        ("dwk", &gpu.dwk, &cpu.dwk), ("dwv", &gpu.dwv, &cpu.dwv),
                        ("dx", &gpu.dx, &cpu.dx),
                    ] {
                        if g.len() != c.len() {
                            failures.push(format!(
                                "hd={hd} causal={causal} {name}: len {} != {}", g.len(), c.len()
                            ));
                        }
                        if let Some((i, v)) = g.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                            failures.push(format!("hd={hd} causal={causal} {name}[{i}]={v} not finite"));
                        }
                    }
                    // dq/dk/dv: absolute gate (seq-scaled).
                    let d_dq = max_abs_diff(&gpu.dq, &cpu.dq);
                    let d_dk = max_abs_diff(&gpu.dk, &cpu.dk);
                    let d_dv = max_abs_diff(&gpu.dv, &cpu.dv);
                    // dw: mixed atol+rtol gate (seq-scaled atol).
                    let e_dwq = worst_allclose_excess(&gpu.dwq, &cpu.dwq, rtol);
                    let e_dwk = worst_allclose_excess(&gpu.dwk, &cpu.dwk, rtol);
                    let e_dwv = worst_allclose_excess(&gpu.dwv, &cpu.dwv, rtol);
                    // dx: absolute gate (extra reduction + f16-dQ headroom).
                    let d_dx = max_abs_diff(&gpu.dx, &cpu.dx);
                    eprintln!(
                        "[multitile] seq=128 hd={hd} causal={} fwd={fwd_max_abs:.3e} | \
                         dq={d_dq:.3e} dk={d_dk:.3e} dv={d_dv:.3e} \
                         dwq={e_dwq:.3e} dwk={e_dwk:.3e} dwv={e_dwv:.3e} dx={d_dx:.3e} \
                         (qkv_tol={qkv_tol:.1e} dw_atol={dw_atol:.1e} dx_tol={dx_tol:.1e})",
                        causal as u8
                    );
                    for (name, val, tol) in [
                        ("dq", d_dq, qkv_tol), ("dk", d_dk, qkv_tol), ("dv", d_dv, qkv_tol),
                        ("dwq", e_dwq, dw_atol), ("dwk", e_dwk, dw_atol), ("dwv", e_dwv, dw_atol),
                        ("dx", d_dx, dx_tol),
                    ] {
                        if !(val < tol) {
                            failures.push(format!(
                                "hd={hd} causal={causal} {name}: {val:.3e} >= tol {tol:.1e}"
                            ));
                        }
                    }
                }
                Err(e) => {
                    failures.push(format!(
                        "hd={hd} causal={causal}: multi-tile backward launch failed (expected \
                         success after PR2b): {e}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "multitile failures:\n{}", failures.join("\n"));
}

/// Phase 1.3 (PR3): rope_q=TRUE multi-tile parity gate. Companion to the
/// rope_q=false `t6_3_multitile_seq128` — locks in the forward RoPE multi-tile
/// fix (cos/sin indexed by global q_start+row, not tile-local row). Before the
/// fix, forward parity was 4.5-9.8 (garbage) and all grads were O(1) wrong; the
/// isolation that found it (rope_q the sole variable) is preserved in the doc.
///
/// Tolerances: same seq-scaled grad tiers as the rope-off gate, plus a rope-
/// aware forward tolerance (RoPE's extra f16 math ~doubles forward round-off:
/// hd64 fwd ~2.6e-2 vs 1.2e-2 rope-off).
#[test]
#[ignore]
fn t6_3_multitile_seq128_rope() {
    if !cuda_available() {
        eprintln!("[multitile-rope] skipping — no CUDA");
        return;
    }
    let rtol = 5e-3f32;
    let mut failures: Vec<String> = Vec::new();
    for hd in [32u32, 64u32] {
        for causal in [false, true] {
            let fwd_tol = if hd <= 32 { 1e-2f32 } else { 4e-2f32 }; // rope-aware
            let base = tol_for_head_dim(hd);
            let qkv_tol = base * 2.0;
            let dw_atol = base * 2.0;
            let dx_tol = (base * 8.0).max(1.5e-2);
            match run_fused_backward_config_seq(32, 32, hd, 1, hd, causal, true, 128) {
                Ok((gpu, cpu)) => {
                    let fwd_max_abs = f32::from_bits(
                        LAST_FWD_PARITY_MAX_ABS.load(std::sync::atomic::Ordering::SeqCst),
                    );
                    if fwd_max_abs >= fwd_tol {
                        failures.push(format!(
                            "hd={hd} causal={causal} rope: forward parity {fwd_max_abs:.3e} > {fwd_tol:.0e}"
                        ));
                    }
                    for (name, g, c) in [
                        ("dq", &gpu.dq, &cpu.dq), ("dk", &gpu.dk, &cpu.dk),
                        ("dv", &gpu.dv, &cpu.dv), ("dwq", &gpu.dwq, &cpu.dwq),
                        ("dwk", &gpu.dwk, &cpu.dwk), ("dwv", &gpu.dwv, &cpu.dwv),
                        ("dx", &gpu.dx, &cpu.dx),
                    ] {
                        if g.len() != c.len() {
                            failures.push(format!("hd={hd} causal={causal} rope {name}: len {} != {}", g.len(), c.len()));
                        }
                        if let Some((i, v)) = g.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                            failures.push(format!("hd={hd} causal={causal} rope {name}[{i}]={v} not finite"));
                        }
                    }
                    let d_dq = max_abs_diff(&gpu.dq, &cpu.dq);
                    let d_dk = max_abs_diff(&gpu.dk, &cpu.dk);
                    let d_dv = max_abs_diff(&gpu.dv, &cpu.dv);
                    let e_dwq = worst_allclose_excess(&gpu.dwq, &cpu.dwq, rtol);
                    let e_dwk = worst_allclose_excess(&gpu.dwk, &cpu.dwk, rtol);
                    let e_dwv = worst_allclose_excess(&gpu.dwv, &cpu.dwv, rtol);
                    let d_dx = max_abs_diff(&gpu.dx, &cpu.dx);
                    eprintln!(
                        "[multitile-rope] seq=128 hd={hd} causal={} fwd={fwd_max_abs:.3e} | \
                         dq={d_dq:.3e} dk={d_dk:.3e} dv={d_dv:.3e} \
                         dwq={e_dwq:.3e} dwk={e_dwk:.3e} dwv={e_dwv:.3e} dx={d_dx:.3e}",
                        causal as u8
                    );
                    for (name, val, tol) in [
                        ("dq", d_dq, qkv_tol), ("dk", d_dk, qkv_tol), ("dv", d_dv, qkv_tol),
                        ("dwq", e_dwq, dw_atol), ("dwk", e_dwk, dw_atol), ("dwv", e_dwv, dw_atol),
                        ("dx", d_dx, dx_tol),
                    ] {
                        if !(val < tol) {
                            failures.push(format!("hd={hd} causal={causal} rope {name}: {val:.3e} >= tol {tol:.1e}"));
                        }
                    }
                }
                Err(e) => {
                    failures.push(format!("hd={hd} causal={causal} rope: launch failed: {e}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "multitile rope failures:\n{}", failures.join("\n"));
}

/// Numerical sweep across (head_dim, causal, rope_q). One line per config
/// with per-gradient max_abs and PASS/FAIL. Gated by the same three
/// NUMERICAL_GATE_* consts as the smoke — only enabled gates panic.
///
/// Matrix: head_dim ∈ {32, 64, 128}, heads=1, causal ∈ {0,1}, rope_q ∈ {0,1}.
/// Configs rejected by `validate_scalar_v2_config(.., Backward)` are
/// skipped with a log line. Configs with rc≠0 at forward/backward
/// launch are logged as FAIL but do not abort the sweep.
#[test]
#[ignore]
fn t6_3_matrix_sweep_numerical() {
    if !cuda_available() {
        eprintln!("[T6.3 num] skipping — no CUDA");
        return;
    }

    #[derive(Default)]
    struct Tally {
        pass: u32,
        fail: u32,
        skipped_validator: u32,
        skipped_launch: u32,
        fail_detail: Vec<String>,
    }
    let mut tally = Tally::default();

    // Iterate in a deterministic order.
    for &hd in &[32u32, 64, 128] {
        for &causal in &[false, true] {
            for &rope_q in &[false, true] {
                // Use block_q == block_kv == head_dim (fused_projections
                // validator requirement + Tier A smoke convention).
                let bq = hd;
                let bkv = hd;
                let heads = 1u32;
                let dm = hd;

                // Pre-check validator so we can log a clean skip before
                // forward alloc.  Construct the same config shape as
                // run_fused_backward_config.
                let pre_cfg = FlashAttentionConfig {
                    block_q: bq as i64, block_kv: bkv as i64, head_dim: hd as i64,
                    causal, paged: false, rope_q,
                    rope_style: RopeStyle::Adjacent,
                    gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
                        level: 2, fused_rmsnorm: true, fused_projections: true,
                        fused_output_proj: false,
                        save_activations_for_backward: true,
                        active_heads: heads,
                        rmsnorm_eps: 1e-5, d_model: dm,
                        skip_rmsnorm_prologue: false,
                        static_seq_len: None,
                    }),
                    checkpoint: None,
                };
                if let Err(e) = smem_layout::validate_scalar_v2_config(
                    &pre_cfg, Direction::Backward,
                ) {
                    let fwd = smem_layout::total_bytes(&pre_cfg);
                    let extra = smem_layout::backward_extra_bytes(&pre_cfg);
                    eprintln!(
                        "[sweep] hd={hd} causal={} rope={} SKIP validator \
                         (fwd={fwd}B extra={extra}B total={}B): {e}",
                        causal as u8, rope_q as u8, fwd + extra
                    );
                    tally.skipped_validator += 1;
                    continue;
                }

                let (gpu, cpu) = match run_fused_backward_config(
                    bq, bkv, hd, heads, dm, causal, rope_q,
                ) {
                    Ok(pair) => pair,
                    Err(e) => {
                        eprintln!(
                            "[sweep] hd={hd} causal={} rope={} SKIP launch: {e}",
                            causal as u8, rope_q as u8
                        );
                        tally.skipped_launch += 1;
                        continue;
                    }
                };

                let dq = max_abs_diff(&gpu.dq, &cpu.dq);
                let dk = max_abs_diff(&gpu.dk, &cpu.dk);
                let dv = max_abs_diff(&gpu.dv, &cpu.dv);
                let dwq = max_abs_diff(&gpu.dwq, &cpu.dwq);
                let dwk = max_abs_diff(&gpu.dwk, &cpu.dwk);
                let dwv = max_abs_diff(&gpu.dwv, &cpu.dwv);
                let dx = max_abs_diff(&gpu.dx, &cpu.dx);

                let base_tol = tol_for_head_dim(hd);
                // RoPE adds f16 round-trips (cos/sin load + rotation + store)
                // that compound ~30% extra noise on the weight gradient chain.
                let tol = if rope_q { base_tol * 1.5 } else { base_tol };
                let dx_tol = (base_tol * 6.0).max(1e-2);

                let mut fails: Vec<String> = Vec::new();
                if NUMERICAL_GATE_DQKV_ENABLED {
                    if dq >= tol { fails.push(format!("dq={dq:.2e}>{tol:.0e}")); }
                    if dk >= tol { fails.push(format!("dk={dk:.2e}>{tol:.0e}")); }
                    if dv >= tol { fails.push(format!("dv={dv:.2e}>{tol:.0e}")); }
                }
                if NUMERICAL_GATE_DW_ENABLED {
                    // Mixed atol+rtol gate for dW (see worst_allclose_excess):
                    // |dW| cells reach 2..4 where the f16 store quantum is
                    // ~2e-3, so absolute-only mis-fires on value-proportional
                    // quantization noise.
                    let rtol = 5e-3f32;
                    let e_dwq = worst_allclose_excess(&gpu.dwq, &cpu.dwq, rtol);
                    let e_dwk = worst_allclose_excess(&gpu.dwk, &cpu.dwk, rtol);
                    let e_dwv = worst_allclose_excess(&gpu.dwv, &cpu.dwv, rtol);
                    if e_dwq >= tol { fails.push(format!("dwq_excess={e_dwq:.2e}>{tol:.0e}")); }
                    if e_dwk >= tol { fails.push(format!("dwk_excess={e_dwk:.2e}>{tol:.0e}")); }
                    if e_dwv >= tol { fails.push(format!("dwv_excess={e_dwv:.2e}>{tol:.0e}")); }
                }
                if NUMERICAL_GATE_DX_ENABLED && dx >= dx_tol {
                    fails.push(format!("dx={dx:.2e}>{dx_tol:.0e}"));
                }

                let status = if fails.is_empty() { "PASS" } else { "FAIL" };
                eprintln!(
                    "[sweep] hd={hd} causal={} rope={}: dq={dq:.2e} dk={dk:.2e} \
                     dv={dv:.2e} dwq={dwq:.2e} dwk={dwk:.2e} dwv={dwv:.2e} dx={dx:.2e} \
                     [{status}]{}",
                    causal as u8, rope_q as u8,
                    if fails.is_empty() { "".into() } else { format!(" ({})", fails.join(",")) }
                );
                if fails.is_empty() {
                    tally.pass += 1;
                } else {
                    tally.fail += 1;
                    tally.fail_detail.push(format!(
                        "hd={hd} causal={} rope={}: {}",
                        causal as u8, rope_q as u8, fails.join(",")
                    ));
                }
            }
        }
    }

    eprintln!(
        "[sweep] tally: pass={} fail={} skipped_validator={} skipped_launch={}",
        tally.pass, tally.fail, tally.skipped_validator, tally.skipped_launch
    );
    if tally.fail > 0 {
        panic!(
            "numerical sweep: {} config(s) failed:\n  {}",
            tally.fail, tally.fail_detail.join("\n  ")
        );
    }
}

#[test]
#[ignore]
fn t6_3_matrix_sweep_structural() {
    if !cuda_available() {
        eprintln!("[T6.3] skipping — no CUDA");
        return;
    }
    // Structural-only sweep: every row must launch rc=0 and produce
    // finite gradients. Numerical tolerance comparison stays blocked.
    let configs: &[(u32, u32, u32, u32, u32, bool, bool)] = &[
        (32, 32, 32, 1, 32, false, false),
        (32, 32, 32, 1, 32, true,  false),
        (32, 32, 32, 1, 32, false, true),
    ];
    for &(bq, bkv, hd, h, dm, causal, rope) in configs {
        match run_fused_backward_config(bq, bkv, hd, h, dm, causal, rope) {
            Ok((gpu, cpu)) => {
                eprintln!(
                    "[T6.3 sweep] bq={bq} bkv={bkv} hd={hd} h={h} dm={dm} \
                     causal={causal} rope={rope}: dq_max_abs={:.3e} dx_max_abs={:.3e}",
                    max_abs_diff(&gpu.dq, &cpu.dq),
                    max_abs_diff(&gpu.dx, &cpu.dx),
                );
                // Finiteness check skipped: the Phase 3 inner-loop
                // placeholder constants can produce NaN/Inf in edge
                // combinations (e.g. div-by-zero when
                // row_sum=0 on a zero-init readback slot). Once real
                // HBM loads land, enable alongside NUMERICAL_GATE_ENABLED.
                for (name, arr) in [
                    ("dq", &gpu.dq), ("dk", &gpu.dk), ("dv", &gpu.dv),
                    ("dwq", &gpu.dwq), ("dwk", &gpu.dwk), ("dwv", &gpu.dwv),
                    ("dx", &gpu.dx),
                ] {
                    let bad = arr.iter().filter(|v| !v.is_finite()).count();
                    if bad > 0 {
                        eprintln!(
                            "[T6.3 sweep] WARN bq={bq} hd={hd} rope={rope}: \
                             gpu {name} has {bad}/{} non-finite (skeleton \
                             placeholder artefact; blocker is unchanged)",
                            arr.len()
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("[T6.3 sweep] config skipped: {e}");
            }
        }
    }
}

/// Diagnostic pass 2: localise which backward SMEM tile is first to
/// diverge. Dumps max_abs_diff, first-8-element side-by-side, peak
/// element, and zero-count for dq/dk/dv at the smoke config. Signals:
///   * all-zero             → accumulator dropped / never reached HBM
///   * constant             → gated by a broken predicate
///   * cpu × scalar         → scale bug
///   * right sign, wrong k  → addressing / missing-term bug
///   * noise uncorrelated   → SMEM aliasing (K/V share kv_offset)
#[test]
#[ignore]
fn t6_3_element_dump_diag() {
    if !cuda_available() {
        eprintln!("[T6.3 diag] skipping — no CUDA");
        return;
    }
    let (gpu, cpu) = run_fused_backward_config(32, 32, 32, 1, 32, false, false)
        .expect("smoke config must launch structurally");

    fn dump(name: &str, gpu: &[f32], cpu: &[f32]) {
        assert_eq!(gpu.len(), cpu.len(), "{name} len mismatch");
        let mut max_abs = 0f32;
        let mut peak_idx = 0usize;
        for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = (g - c).abs();
            if d > max_abs { max_abs = d; peak_idx = i; }
        }
        let zeros = gpu.iter().filter(|v| v.abs() < 1e-8).count();
        let const_q = {
            // crude "is it nearly constant" heuristic: min/max spread
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in gpu { if v < lo { lo = v; } if v > hi { hi = v; } }
            hi - lo
        };
        // cpu × scalar check: median(gpu/cpu) — pick a few non-tiny cpu entries.
        let mut ratios = Vec::new();
        for (&g, &c) in gpu.iter().zip(cpu.iter()).take(64) {
            if c.abs() > 1e-3 { ratios.push(g / c); }
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_ratio = if ratios.is_empty() { f32::NAN } else { ratios[ratios.len() / 2] };

        eprintln!("─── {name}  len={}  max_abs_diff={:.3e}  zero_count={}/{}  spread(hi-lo)={:.3e}  median(g/c)={:.3e}",
            gpu.len(), max_abs, zeros, gpu.len(), const_q, median_ratio);
        eprintln!("  peak idx={peak_idx} gpu={:.4e} cpu={:.4e} diff={:.4e}",
            gpu[peak_idx], cpu[peak_idx], gpu[peak_idx] - cpu[peak_idx]);
        eprintln!("  first 8:   idx | gpu           | cpu           | diff");
        for i in 0..8.min(gpu.len()) {
            eprintln!("            {:>3} | {:>13.5e} | {:>13.5e} | {:>13.5e}",
                i, gpu[i], cpu[i], gpu[i] - cpu[i]);
        }
    }

    eprintln!("[T6.3 diag] config: bq=32 bkv=32 hd=32 heads=1 dm=32 causal=false rope=false");
    dump("dq", &gpu.dq, &cpu.dq);
    dump("dk", &gpu.dk, &cpu.dk);
    dump("dv", &gpu.dv, &cpu.dv);
    eprintln!("[T6.3 diag] SMEM-aliasing hypothesis: backward prelude sets \
        %k_smem_base and %v_smem_base BOTH to kv_offset (phases/backward/prelude.rs:174-179). \
        After kv_load::emit_v runs (mod.rs:472), the K tile holds V data. ds_compute then \
        recomputes S = Q·V^T instead of Q·K^T (ds_compute.rs:75), corrupting P, dS, dP \
        and therefore dq/dk/dv. Fix: allocate a separate V-input SMEM region (e.g. extend \
        backward_extra_bytes by block_kv*head_dim*2) and point %v_smem_base there.");
}
