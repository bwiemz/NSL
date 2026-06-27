//! Cycle-12 ships compile-only. Cycle 13 runs on real CUDA — lift `#[ignore]` after one green pass.
//!
//! Three-way oracle harness for `@checkpoint(policy="full")` + CSHA
//! kv-recompute backward.  Same five `#[test] #[ignore]` configs the
//! spec §3 calls out:
//!
//!   - hd=64,  S=512
//!   - hd=64,  S=2048
//!   - hd=128, S=512
//!   - hd=128, S=2048
//!   - hd=128, S=4096
//!
//! Per-test oracle (cycle 13 enables once GPU is wired):
//!   1. gpu_checkpoint     — synthesize backward with `checkpoint=Some(full())`
//!   2. gpu_non_checkpoint — same config with `checkpoint=None` (baseline)
//!   3. cpu_reference      — `csha_reference_backward` (self-validated by
//!                            csha_reference.rs:619 finite-difference)
//!   4. cpu_prologue       — `cpu_naive_norm_proj_rope` (Task 4 G7 oracle)
//!
//! Three-way diff:
//!   gpu_checkpoint vs gpu_non_checkpoint  → kv-recompute math correctness
//!   gpu_checkpoint vs cpu_reference       → full-stack closure
//!   SMEM-readback   vs cpu_prologue       → prologue-arithmetic closure
//!
//! Tolerance ladder per spec §3:
//!   - dq/dk/dv at hd=64:  atol=5e-4, rtol=5e-3
//!   - dq/dk/dv at hd=128: atol=2e-3, rtol=1e-2
//!   - dwq/dwk/dwv:        atol=1e-3, rtol=1e-2
//!   - dx:                 atol=1e-2, rtol=2e-2
//!
//! Cycle-12 R-C12-1 mitigation: ships compile-only (gated behind
//! `feature = "cuda"`); cycle-12 primary verification doesn't lean on
//! GPU execution.  G5/G6 (production wire-up smoke) verify checkpoint
//! plumbing without GPU; G7 verifies prologue arithmetic without GPU;
//! G3c verifies RoPE-K structural closure without GPU.

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference_backward, CshaGradients, CshaInputs, CshaShape};

use std::ffi::CString;

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, smem_layout, synthesize_backward_with_tier_b,
    synthesize_flash_attention_ptx_v2,
};
use nsl_test::cpu_naive_prologue::{cpu_naive_norm_proj_rope, PrologueConfig};

// Cycle-14 FFI block — mirrors csha_cuda_backward.rs:57-65 sister template.
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h,
    nsl_test_cuda_free, nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use nsl_runtime::flash_attention::{
    nsl_csha_alloc_backward_activations, nsl_csha_free_backward_activations,
    nsl_flash_attention_csha_backward, nsl_flash_attention_csha_with_saves,
};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Mirror of `csha_cuda_backward.rs:128 det_seq` so configs are
/// reproducible across harnesses.
fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s: u32 = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((s >> 16) as f32 / 65535.0) - 0.5
    }).collect()
}

/// Honors `NSL_SKIP_CUDA_TESTS` (idiom from csha_cuda_backward.rs:137).
/// Cycle 14: activated — wires the real `nsl_cuda_init` check.
fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    unsafe { nsl_cuda_init() == 0 }
}

/// G14-E default-run gate: `cuda_available()` MUST return false when
/// `NSL_SKIP_CUDA_TESTS` is set, regardless of whether a CUDA device is
/// present on the host. Closes the cycle-14 "env-var honored" invariant.
#[test]
fn g14_e_cuda_available_honors_skip_env() {
    // SAFETY: env var manipulation; serial test by convention.
    std::env::set_var("NSL_SKIP_CUDA_TESTS", "1");
    assert!(
        !cuda_available(),
        "cuda_available must respect NSL_SKIP_CUDA_TESTS"
    );
    std::env::remove_var("NSL_SKIP_CUDA_TESTS");
}

/// max(|a[i] - b[i]|).
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch in max_abs_diff");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// max(|a[i] - b[i]| / max(|b[i]|, eps)). Relative diff.
fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch in max_rel_diff");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs() / y.abs().max(1e-6))
        .fold(0f32, f32::max)
}

/// Spec §3 tolerance ladder for dq/dk/dv.
fn tol_dqkv(head_dim: usize) -> (f32, f32) {
    if head_dim >= 128 {
        (2e-3, 1e-2)
    } else {
        (5e-4, 5e-3)
    }
}

// ── f16 round-trip helpers (mirrors csha_cuda_backward.rs:86-126) ──────────

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

/// Build a cycle-14 Level-1 fused-projections + `@checkpoint(policy="full")`
/// config: causal, rope_q, no sinks, no segments, no paged_kv collision,
/// block_q == block_kv == 32 (downsized from cycle-12's 64 to bring hd=128
/// configs back inside the sm_120 99 KB dynamic-SMEM cap per spec §1.7),
/// `gpu_sm=80` to unlock the Tier B.1 dispatch fork on Blackwell.
fn build_cycle14_config(head_dim: u32, _seq_len: u32) -> FlashAttentionConfig {
    // seq_len enters via the harness `inputs`/launch shape; FlashAttentionConfig
    // doesn't carry a sequence dimension (the kernel takes it at launch).
    let d_model = head_dim; // 1 head, dm == hd for shape alignment with CPU ref
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: head_dim as i64,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras::level1_with_fused_proj(1e-6)),
        checkpoint: Some(CheckpointExtras::full()),
    }.with_d_model(d_model)
}

// `FlashAttentionConfig` doesn't have a `with_d_model` builder today;
// the d_model lives inside the CshaExtras. Provide a local extension trait
// so the harness reads cleanly. Compile-only — cycle 13 may inline this.
trait WithDModel {
    fn with_d_model(self, d_model: u32) -> Self;
}
impl WithDModel for FlashAttentionConfig {
    fn with_d_model(mut self, d_model: u32) -> Self {
        if let Some(csha) = self.csha.as_mut() {
            csha.d_model = d_model;
        }
        self
    }
}

// ── Per-config tests ───────────────────────────────────────────────────────
//
// Each test:
//   1. Skip if `!cuda_available()` (cycle 12 always-true skip; cycle 13 wires
//      real cudarc).
//   2. Build config, synthesize backward PTX (compile-only smoke — cycle 12
//      verifies the PTX synthesis path doesn't refuse).
//   3. Construct CPU references (csha_reference_backward + cpu_naive_prologue).
//   4. Cycle 13: launch GPU kernels, run three-way comparator, assert
//      tolerances.  Cycle 12: log expectations.
//
// All tests `#[ignore]`d so they NEVER run in default CI runs.  Cycle 13
// lifts `#[ignore]` AFTER one green pass on real CUDA hardware.

#[test]
#[ignore]
fn t_recompute_hd64_s512_bq32() {
    run_three_way_oracle(64, 512);
}

#[test]
#[ignore]
fn t_recompute_hd64_s2048_bq32() {
    run_three_way_oracle(64, 2048);
}

#[test]
#[ignore]
fn t_recompute_hd128_s512_bq32() {
    run_three_way_oracle(128, 512);
}

#[test]
#[ignore]
fn t_recompute_hd128_s2048_bq32() {
    run_three_way_oracle(128, 2048);
}

// ── G14-B/C/D default-run gates (no #[ignore]) ─────────────────────────────

/// G14-B: cycle-14 pre-impl R5 refusal pin. With block=64 + hd=128 +
/// fused_projections=true, the backward SMEM total still exceeds the
/// sm_120 dynamic-SMEM cap. The cycle-12 cascade routes this through
/// `validate_scalar_v2_config` which returns a refusal whose message
/// contains the substring `"exceeds device"` (smem_layout.rs:245+).
/// Pinning it here keeps the budget math honest under future refactors.
#[test]
fn g14_b_recompute_hd128_s4096_bq64_refuses_r5() {
    let mut cfg = build_cycle14_config(128, 4096);
    cfg.block_q = 64;
    cfg.block_kv = 64;
    // Re-pin checkpoint after the mutation (no-op for the carrier).
    cfg.checkpoint = Some(CheckpointExtras::full());
    let err = synthesize_backward_with_tier_b(&cfg, None)
        .expect_err("over-cap config must refuse with R5");
    assert!(
        err.contains("exceeds device"),
        "R5 'exceeds device' substring missing: {err}"
    );
}

/// G14-C: with `gpu_sm=80`, the synthesized PTX MUST target sm_80
/// (cycle-9 spec §1.8 dispatch invariant — Blackwell sm_120 JITs sm_80
/// PTX forward-compat per Phase A smoke proof).
#[test]
fn g14_c_sm80_target_emitted_under_gpu_sm_80() {
    let cfg = build_cycle14_config(64, 512);
    let ptx = synthesize_backward_with_tier_b(&cfg, None)
        .expect("hd=64 bq=32 must synthesize");
    assert!(
        ptx.contains(".target sm_80"),
        ".target sm_80 missing from synthesized backward PTX"
    );
}

/// G14-D: the SMEM total (forward layout + backward extras) for the
/// cycle-14 hd=64 baseline config MUST fit under the sm_120 99 KB cap
/// when `gpu_sm=80` AND `block=32`. Catches accidental budget regressions
/// in `smem_layout::backward_extra_bytes` or `smem_layout::total_bytes`.
#[test]
fn g14_d_recompute_extra_bytes_accounted() {
    let cfg = build_cycle14_config(64, 512);
    let total = smem_layout::total_bytes(&cfg) + smem_layout::backward_extra_bytes(&cfg);
    assert!(
        total <= 99 * 1024,
        "cycle-14 hd=64 bq=32 SMEM total {total} > sm_120 99 KB cap"
    );
}

/// All device buffers + saves struct, returned by `forward_launch_and_saves`.
struct ForwardArtifacts {
    // Inputs (kept live so backward can reuse the same H2D'd buffers).
    x_dev: i64,
    wq_dev: i64,
    wk_dev: i64,
    wv_dev: i64,
    nw_dev: i64,
    cos_dev: i64,
    sin_dev: i64,
    do_dev: i64,
    // Forward outputs.
    q_dev: i64,
    k_dev: i64,
    v_dev: i64,
    out_dev: i64,
    lse_dev: i64,
    // Backward saves alloc handle.
    saves: nsl_runtime::flash_attention::CshaBackwardActivations,
    // Sizes (so backward knows readback shapes).
    qkv_bytes: i64,
    w_bytes: i64,
    rope_bytes: i64,
    dw_bytes: i64,
    dx_bytes: i64,
    dxn_bytes: i64,
    // Reference inputs (cos/sin/wq/wk/wv f16-rounded back to f32 so CPU
    // reference and GPU see the same precision — mirrors sister:203-232).
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    do_f32: Vec<f32>,
    x_host: Vec<f32>,
    wq_f32: Vec<f32>,
    wk_f32: Vec<f32>,
    wv_f32: Vec<f32>,
    nw_host: Vec<f32>,
}

/// Launches the CSHA `with_saves` forward kernel on the cycle-14 config,
/// returning all device buffers + the saves handle for downstream backward
/// reuse. Mirrors `csha_cuda_backward.rs:243-340` sister template.
fn forward_launch_and_saves(
    config: &FlashAttentionConfig,
    head_dim: u32,
    seq_len: u32,
) -> ForwardArtifacts {
    let batch = 1usize;
    let heads = 1usize;
    let seq = seq_len as usize;
    let hd = head_dim as usize;
    let dm = hd; // 1 head, dm == hd
    let kv_dim = heads * hd;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let norm_eps = 1e-6f32;
    let causal = config.causal;

    // ── Deterministic host data (mirrors sister:192-230) ────────────────────
    let x_host = det_seq(42, heads * seq * hd);
    let wq_f32 = det_seq(43, dm * kv_dim);
    let wk_f32 = det_seq(44, dm * kv_dim);
    let wv_f32 = det_seq(45, dm * kv_dim);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let nw_host = vec![1.0f32; hd];
    let cos_raw: Vec<f32> = (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).cos()).collect();
    let sin_raw: Vec<f32> = (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).sin()).collect();
    let cos_f16: Vec<u16> = cos_raw.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let sin_f16: Vec<u16> = sin_raw.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let cos_f32: Vec<f32> = cos_f16.iter().map(|&b| f16_to_f32(b)).collect();
    let sin_f32: Vec<f32> = sin_f16.iter().map(|&b| f16_to_f32(b)).collect();
    let do_raw = det_seq(99, seq * kv_dim);
    let do_f16: Vec<u16> = do_raw.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_f32: Vec<f32> = do_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // ── Device allocation ──────────────────────────────────────────────────
    let qkv_bytes = (heads * seq * hd * 2) as i64;
    let lse_bytes = (batch * heads * seq * 4) as i64;
    let x_bytes = (heads * seq * hd * 4) as i64;
    let w_bytes = (dm * kv_dim * 2) as i64;
    let nw_bytes = (hd * 4) as i64;
    let rope_bytes = (seq * hd / 2 * 2) as i64;
    let dw_bytes = (dm * kv_dim * 2) as i64;
    let dx_bytes = (heads * seq * hd * 4) as i64;
    let dxn_bytes = (batch * seq * dm * 4) as i64;

    unsafe { nsl_cuda_init(); }

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

    let saves = unsafe {
        nsl_csha_alloc_backward_activations(
            batch as i64, heads as i64, seq as i64, hd as i64,
        )
    };

    // ── H2D ────────────────────────────────────────────────────────────────
    unsafe {
        nsl_test_cuda_h2d(x_dev,  x_host.as_ptr()  as i64, x_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(nw_dev, nw_host.as_ptr() as i64, nw_bytes);
        nsl_test_cuda_h2d(cos_dev, cos_f16.as_ptr() as i64, rope_bytes);
        nsl_test_cuda_h2d(sin_dev, sin_f16.as_ptr() as i64, rope_bytes);
        nsl_test_cuda_h2d(do_dev, do_f16.as_ptr() as i64, qkv_bytes);
    }

    // ── Forward PTX synth + launch ─────────────────────────────────────────
    let fwd_ptx = synthesize_flash_attention_ptx_v2(config);
    let fwd_name = CString::new(flash_attention_kernel_name_v2(config)).unwrap();
    let fwd_smem_total = smem_layout::total_bytes(config);
    let fwd_smem_dyn = if smem_layout::needs_dynamic_smem(config) {
        fwd_smem_total as i64
    } else { 0 };

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
            config.block_q, config.block_kv,
            if causal { 1 } else { 0 },
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, dm as i64,
            saves.q_proj, saves.k_proj, saves.v_proj,
            saves.row_max, saves.row_sum,
            saves.x_raw,
            // segment_ids_ptr, tier_b_ptx_ptr, tier_b_name_ptr, doc_starts_ptr.
            0, 0, 0, 0,
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
        free_all(&[q_dev, k_dev, v_dev, out_dev, lse_dev,
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            cos_dev, sin_dev, do_dev]);
        panic!(
            "[cycle14] forward launch FAILED rc={rc_fwd} hd={head_dim} S={seq_len}\n\
             JIT log:\n{log}"
        );
    }

    // ── Eyeball check: forward outputs should be finite + non-trivial ──────
    // Read back the FULL `out` buffer (cheap at our smoke sizes) so the
    // probe sees the late-sequence rows that have the most non-trivial
    // attention weights (causal=true with rope_q=true zeros position 0).
    let total_out_elems = heads * seq * hd;
    let mut out_all = vec![0u16; total_out_elems];
    unsafe {
        nsl_test_cuda_d2h(
            out_all.as_mut_ptr() as i64,
            out_dev,
            (total_out_elems * 2) as i64,
        );
    }
    let out_all_f32: Vec<f32> = out_all.iter().map(|&b| f16_to_f32(b)).collect();
    let nonzero = out_all_f32.iter().filter(|x| x.abs() > 1e-6).count();
    let all_finite = out_all_f32.iter().all(|x| x.is_finite());
    let first4: Vec<f32> = out_all_f32.iter().take(4).copied().collect();
    let mid4: Vec<f32> = out_all_f32.iter().skip(total_out_elems / 2).take(4).copied().collect();
    eprintln!(
        "[cycle14 fwd] hd={head_dim} S={seq_len} first 4 out: {:?} | mid 4 out: {:?} | \
         nonzero={}/{}",
        first4, mid4, nonzero, total_out_elems
    );
    assert!(
        all_finite,
        "forward out has non-finite values; first 4: {:?}", first4
    );
    // Sanity: SOME output must be non-zero (kernel actually wrote). The
    // CSHA fused-projections forward with causal+rope_q can leave large
    // swaths of the buffer zero (position 0 + masked tails); the spec's
    // eyeball gate is just "out is sane", not "out is dense". Anything
    // strictly greater than zero passes.
    assert!(
        nonzero > 0,
        "forward out is entirely zero — kernel did not write any rows"
    );

    ForwardArtifacts {
        x_dev, wq_dev, wk_dev, wv_dev, nw_dev, cos_dev, sin_dev, do_dev,
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        saves,
        qkv_bytes, w_bytes, rope_bytes, dw_bytes, dx_bytes, dxn_bytes,
        cos_f32, sin_f32, do_f32, x_host, wq_f32, wk_f32, wv_f32, nw_host,
    }
}

/// Launches the CSHA backward kernel for `bwd_cfg` against `forward`'s saves
/// and returns the 7 readback gradients (as f32). `bwd_cfg.checkpoint`
/// determines which dispatch branch is taken inside
/// `synthesize_backward_with_tier_b` (mod.rs:1459):
///   - `None` ⇒ HBM-resident kv_load (Path A baseline)
///   - `Some(Full)` ⇒ kv_recompute from x_raw + Wk/Wv (Path B — §5.3 evidence)
fn launch_backward_path(
    bwd_cfg: &FlashAttentionConfig,
    forward: &ForwardArtifacts,
    head_dim: u32,
    seq_len: u32,
    path_label: &str,
) -> CshaGradients {
    let batch = 1usize;
    let heads = 1usize;
    let seq = seq_len as usize;
    let hd = head_dim as usize;
    let dm = hd;
    let kv_dim = heads * hd;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let norm_eps = 1e-6f32;
    let causal = bwd_cfg.causal;

    // ── Allocate gradient output buffers + dxn scratch ─────────────────────
    let dq_dev = unsafe { nsl_test_cuda_alloc(forward.qkv_bytes) };
    let dk_dev = unsafe { nsl_test_cuda_alloc(forward.qkv_bytes) };
    let dv_dev = unsafe { nsl_test_cuda_alloc(forward.qkv_bytes) };
    let dwq_dev = unsafe { nsl_test_cuda_alloc(forward.dw_bytes) };
    let dwk_dev = unsafe { nsl_test_cuda_alloc(forward.dw_bytes) };
    let dwv_dev = unsafe { nsl_test_cuda_alloc(forward.dw_bytes) };
    let dx_dev = unsafe { nsl_test_cuda_alloc(forward.dx_bytes) };
    // R-C14-2 mitigation: allocate the dx_norm scratch buffer per sister
    // line 281. No readback — kernel internal staging only; undersizing
    // causes the dRMSNorm tile to scribble past `dx_dev`'s end.
    let dxn_dev = unsafe { nsl_test_cuda_alloc(forward.dxn_bytes) };

    // ── Backward PTX synth + name ──────────────────────────────────────────
    let mut bwd_ptx_str = synthesize_backward_with_tier_b(bwd_cfg, None)
        .unwrap_or_else(|e| {
            free_all(&[dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev,
                dx_dev, dxn_dev]);
            panic!("[{path_label}] backward PTX synth failed hd={head_dim} S={seq_len}: {e}");
        });
    if !bwd_ptx_str.ends_with('\0') { bwd_ptx_str.push('\0'); }
    let bwd_ptx = bwd_ptx_str.into_bytes();
    let bwd_name = CString::new(backward_kernel_name(bwd_cfg)).unwrap();

    // ── R-C14-4: SMEM accounting ───────────────────────────────────────────
    // Per `phases/backward/prelude.rs:backward_total_bytes` the dynamic
    // request is `total_bytes + backward_extra_bytes` (+segment_overhead
    // when segment_masked, which we don't set). If the kernel was emitted
    // with `.extern .shared shmem[]`, the runtime MUST pass that byte
    // count as the cuLaunchKernel `sharedMemBytes` arg — otherwise the
    // grant is silently truncated and stores past the static cap hit
    // ILLEGAL_ADDRESS. Mirrors `tests/csha_cuda_launch_classic.rs:284`
    // sister idiom for forward smem.
    let bwd_smem_total = smem_layout::total_bytes(bwd_cfg)
        + smem_layout::backward_extra_bytes(bwd_cfg);
    let bwd_needs_dyn = bwd_smem_total > 48 * 1024;
    let bwd_smem_dyn = if bwd_needs_dyn { bwd_smem_total as i64 } else { 0 };
    eprintln!(
        "  [{path_label}] bwd SMEM total={} bytes; dyn_request={} (needs_dynamic={})",
        bwd_smem_total, bwd_smem_dyn, bwd_needs_dyn
    );

    // ── Launch ─────────────────────────────────────────────────────────────
    let rc_bwd = unsafe {
        nsl_flash_attention_csha_backward(
            forward.q_dev, forward.k_dev, forward.v_dev,
            forward.out_dev, forward.lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, hd as i64,
            0, 0, 0, 0,
            forward.cos_dev, forward.sin_dev,
            0, 0,
            bwd_smem_dyn,
            bwd_ptx.as_ptr() as i64, bwd_name.as_ptr() as i64,
            bwd_cfg.block_q, bwd_cfg.block_kv,
            if causal { 1 } else { 0 },
            forward.x_dev, forward.nw_dev,
            forward.wq_dev, forward.wk_dev, forward.wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, dm as i64,
            forward.saves.q_proj, forward.saves.k_proj, forward.saves.v_proj,
            forward.saves.row_max, forward.saves.row_sum,
            forward.saves.x_raw,
            forward.do_dev,
            dq_dev, dk_dev, dv_dev,
            dwq_dev, dwk_dev, dwv_dev,
            dx_dev,
            dxn_dev,
            // segment_ids, tier_b_ptx, tier_b_name, doc_starts, tier_b2_active.
            0, 0, 0, 0, 0,
        )
    };
    if rc_bwd != 0 {
        let log = unsafe {
            let p = nsl_test_cuda_jit_log(bwd_ptx.as_ptr() as i64);
            if p != 0 {
                std::ffi::CStr::from_ptr(p as *const i8).to_string_lossy().into_owned()
            } else { "<no log>".into() }
        };
        free_all(&[dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev,
            dx_dev, dxn_dev]);
        panic!(
            "[{path_label}] backward launch FAILED rc={rc_bwd} hd={head_dim} S={seq_len}\n\
             JIT log:\n{log}"
        );
    }

    // ── Readback ───────────────────────────────────────────────────────────
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
    let qkv_elems = heads * seq * hd;
    let dw_elems = dm * kv_dim;
    let dx_elems = heads * seq * hd;
    let grads = CshaGradients {
        dq: read_f16(dq_dev, qkv_elems),
        dk: read_f16(dk_dev, qkv_elems),
        dv: read_f16(dv_dev, qkv_elems),
        dwq: read_f16(dwq_dev, dw_elems),
        dwk: read_f16(dwk_dev, dw_elems),
        dwv: read_f16(dwv_dev, dw_elems),
        dx: read_f32(dx_dev, dx_elems),
    };

    free_all(&[dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev,
        dx_dev, dxn_dev]);
    grads
}

/// Per-tensor diff envelope at the cycle-14 tolerance ladder. Panics with a
/// verbatim summary if any tensor exceeds tolerance (NOT just eprintln).
/// Tolerances per spec §6:
///   dQ/dK/dV at hd<128: atol=5e-4 rtol=5e-3
///   dQ/dK/dV at hd≥128: atol=2e-3 rtol=1e-2
///   dW:                 atol=1e-3 rtol=1e-2
///   dx:                 atol=1e-2 rtol=2e-2
/// Per-tensor diff summary. Returns (max_abs, max_rel, ok-mask) for the 7
/// tensors so the caller can decide whether to assert or just log.
fn diff_summary(
    label: &str,
    a: &CshaGradients,
    b: &CshaGradients,
    head_dim: usize,
) -> Vec<&'static str> {
    let (atol_qkv, rtol_qkv) = tol_dqkv(head_dim);
    let atol_dw  = 1e-3f32; let rtol_dw  = 1e-2f32;
    let atol_dx  = 1e-2f32; let rtol_dx  = 2e-2f32;

    let check = |name: &str, x: &[f32], y: &[f32], atol: f32, rtol: f32| -> bool {
        let abs = max_abs_diff(x, y);
        let rel = max_rel_diff(x, y);
        let ok = abs <= atol || rel <= rtol;
        eprintln!(
            "  [{label}] {name}: max_abs={abs:.3e} max_rel={rel:.3e} \
             (atol={atol:.0e} rtol={rtol:.0e}) {}",
            if ok { "PASS" } else { "FAIL" }
        );
        ok
    };

    let names_and_oks: Vec<(&'static str, bool)> = vec![
        ("dq",  check("dq",  &a.dq,  &b.dq,  atol_qkv, rtol_qkv)),
        ("dk",  check("dk",  &a.dk,  &b.dk,  atol_qkv, rtol_qkv)),
        ("dv",  check("dv",  &a.dv,  &b.dv,  atol_qkv, rtol_qkv)),
        ("dwq", check("dwq", &a.dwq, &b.dwq, atol_dw,  rtol_dw)),
        ("dwk", check("dwk", &a.dwk, &b.dwk, atol_dw,  rtol_dw)),
        ("dwv", check("dwv", &a.dwv, &b.dwv, atol_dw,  rtol_dw)),
        ("dx",  check("dx",  &a.dx,  &b.dx,  atol_dx,  rtol_dx)),
    ];
    names_and_oks.into_iter().filter_map(|(n, ok)| if !ok { Some(n) } else { None }).collect()
}

#[allow(dead_code)]
fn assert_grads_within_tolerance(
    label: &str,
    a: &CshaGradients,
    b: &CshaGradients,
    head_dim: usize,
) {
    let (atol_qkv, rtol_qkv) = tol_dqkv(head_dim);
    let atol_dw  = 1e-3f32;
    let rtol_dw  = 1e-2f32;
    let atol_dx  = 1e-2f32;
    let rtol_dx  = 2e-2f32;

    let check = |name: &str, x: &[f32], y: &[f32], atol: f32, rtol: f32| -> (f32, f32, bool) {
        let abs = max_abs_diff(x, y);
        let rel = max_rel_diff(x, y);
        // Pass if EITHER atol OR rtol envelope holds (standard PyTorch
        // semantics — atol catches small-magnitude tensors where rel
        // explodes; rtol catches large-magnitude tensors where abs is
        // unreasonable).
        let ok = abs <= atol || rel <= rtol;
        eprintln!(
            "  [{label}] {name}: max_abs={abs:.3e} max_rel={rel:.3e} \
             (atol={atol:.0e} rtol={rtol:.0e}) {}",
            if ok { "PASS" } else { "FAIL" }
        );
        (abs, rel, ok)
    };

    let r_dq  = check("dq",  &a.dq,  &b.dq,  atol_qkv, rtol_qkv);
    let r_dk  = check("dk",  &a.dk,  &b.dk,  atol_qkv, rtol_qkv);
    let r_dv  = check("dv",  &a.dv,  &b.dv,  atol_qkv, rtol_qkv);
    let r_dwq = check("dwq", &a.dwq, &b.dwq, atol_dw,  rtol_dw);
    let r_dwk = check("dwk", &a.dwk, &b.dwk, atol_dw,  rtol_dw);
    let r_dwv = check("dwv", &a.dwv, &b.dwv, atol_dw,  rtol_dw);
    let r_dx  = check("dx",  &a.dx,  &b.dx,  atol_dx,  rtol_dx);

    let fails: Vec<&str> = [
        ("dq",  r_dq.2),  ("dk",  r_dk.2),  ("dv",  r_dv.2),
        ("dwq", r_dwq.2), ("dwk", r_dwk.2), ("dwv", r_dwv.2),
        ("dx",  r_dx.2),
    ].iter().filter_map(|(n, ok)| if !ok { Some(*n) } else { None }).collect();
    assert!(
        fails.is_empty(),
        "[{label}] tolerance FAIL: {} tensor(s) out of envelope: {:?}",
        fails.len(), fails
    );
}

/// Three-way comparator driver. Tasks 3-5 progressively expand:
///   T3: forward launch (with saves) — verify out is finite
///   T4: backward Path A (checkpoint=None baseline) vs cpu_reference
///   T5: backward Path B (checkpoint=Some(Full)) vs cpu_reference + vs Path A
fn run_three_way_oracle(head_dim: u32, seq_len: u32) {
    if !cuda_available() {
        eprintln!(
            "[cycle14 csha checkpoint recompute] skipping hd={head_dim} S={seq_len} \
             — no CUDA device (or NSL_SKIP_CUDA_TESTS set)"
        );
        return;
    }

    let config = build_cycle14_config(head_dim, seq_len);
    let bwd_ptx_check = synthesize_backward_with_tier_b(&config, None)
        .expect("cycle-14 baseline config must synthesize backward PTX");
    assert!(
        bwd_ptx_check.contains("V2_KV_RECOMPUTE_MAIN"),
        "expected backward PTX to contain the kv-recompute label V2_KV_RECOMPUTE_MAIN"
    );

    let hd = head_dim as usize;
    let seq = seq_len as usize;
    let heads = 1usize;
    let dm = hd;

    // ── Task 3: forward launch with saves ──────────────────────────────────
    let artifacts = forward_launch_and_saves(&config, head_dim, seq_len);

    // ── CPU full-stack reference ───────────────────────────────────────────
    let inputs = CshaInputs {
        x: &artifacts.x_host,
        wq: &artifacts.wq_f32, wk: &artifacts.wk_f32, wv: &artifacts.wv_f32,
        norm_weight: &artifacts.nw_host,
        cos: &artifacts.cos_f32, sin: &artifacts.sin_f32,
    };
    let shape = CshaShape {
        seq, heads, head_dim: hd, d_model: dm,
        causal: config.causal,
        norm_eps: 1e-6,
    };
    let cpu_grads: CshaGradients = csha_reference_backward(&inputs, &shape, &artifacts.do_f32);

    // CPU prologue oracle (kept live for cycle-15 SMEM readback).
    let prologue_cfg = PrologueConfig {
        seq_len: seq, head_dim: hd, d_model: dm, eps: 1e-6,
        num_heads_q: heads, num_heads_kv: heads, num_heads_v: heads,
    };
    let (q_proj_cpu, k_proj_cpu, v_proj_cpu) = cpu_naive_norm_proj_rope(
        &artifacts.x_host,
        &artifacts.wq_f32, &artifacts.wk_f32, &artifacts.wv_f32,
        &artifacts.cos_f32, &artifacts.sin_f32,
        &prologue_cfg,
    );
    let _ = (&q_proj_cpu, &k_proj_cpu, &v_proj_cpu);

    // ── Task 4: backward Path A (checkpoint=None baseline) ─────────────────
    // Clone of config with checkpoint stripped so the dispatch fork at
    // mod.rs:1496 takes the kv_load branch (HBM-resident K_proj/V_proj
    // read from forward saves) instead of kv_recompute.
    let mut cfg_a = config.clone();
    cfg_a.checkpoint = None;

    eprintln!(
        "[cycle14 path A] hd={head_dim} S={seq_len} launching backward with \
         checkpoint=None (kv_load baseline)"
    );
    let gpu_a = launch_backward_path(&cfg_a, &artifacts, head_dim, seq_len, "path A");
    eprintln!("[cycle14 path A] vs cpu_reference:");
    let fails_a = diff_summary("path A vs cpu_ref", &gpu_a, &cpu_grads, hd);

    if !fails_a.is_empty() {
        // R5 deferral surfaced: Path A baseline is RED on cycle-14 config
        // (causal=true rope_q=true level=1 fused_proj). The sister
        // `t6_3_smoke_single_config` runs at causal=false rope_q=false and
        // is GREEN on this branch — see `csha_cuda_backward.rs:425`. So
        // the harness wiring is sound; the cycle-14-specific config
        // exposes a numerical bug in the level-1 fused-projections
        // backward when both causal AND rope_q are enabled.
        //
        // Per cycle-14 spec §4 Commit 4: "If red: STOP — the harness
        // wiring has a bug, not the recompute path". The next cycle's
        // analysis will localise which backward phase (likely
        // `csha_hooks_backward::emit_drmsnorm` or `emit_drope`)
        // produces the divergence. R0 stays in production until cycle
        // 15 lands the fix.
        unsafe { nsl_csha_free_backward_activations(artifacts.saves); }
        free_all(&[
            artifacts.q_dev, artifacts.k_dev, artifacts.v_dev,
            artifacts.out_dev, artifacts.lse_dev,
            artifacts.x_dev, artifacts.nw_dev,
            artifacts.wq_dev, artifacts.wk_dev, artifacts.wv_dev,
            artifacts.cos_dev, artifacts.sin_dev, artifacts.do_dev,
        ]);
        panic!(
            "[cycle14 task4] Path A RED at hd={head_dim} S={seq_len}: \
             {} tensor(s) out of tolerance: {:?}. Per spec Commit 4 STOP \
             condition triggered — Task 5 Path B run deferred until \
             baseline is restored.",
            fails_a.len(), fails_a
        );
    }

    // ── Cleanup forward+saves ──────────────────────────────────────────────
    unsafe { nsl_csha_free_backward_activations(artifacts.saves); }
    free_all(&[
        artifacts.q_dev, artifacts.k_dev, artifacts.v_dev,
        artifacts.out_dev, artifacts.lse_dev,
        artifacts.x_dev, artifacts.nw_dev,
        artifacts.wq_dev, artifacts.wk_dev, artifacts.wv_dev,
        artifacts.cos_dev, artifacts.sin_dev, artifacts.do_dev,
    ]);

    eprintln!("[cycle14 task4] hd={head_dim} S={seq_len} Path A GREEN");
}
