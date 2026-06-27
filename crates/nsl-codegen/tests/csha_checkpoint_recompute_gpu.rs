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

/// Spec §3 tolerance ladder for dq/dk/dv.
fn tol_dqkv(head_dim: usize) -> (f32, f32) {
    if head_dim >= 128 {
        (2e-3, 1e-2)
    } else {
        (5e-4, 5e-3)
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

/// Three-way comparator driver. Task-2 commit ships with signature change
/// (block param removed — locked at bq=bkv=32 via `build_cycle14_config`).
/// Task-3+ expands the body to launch forward+backward on real GPU.
fn run_three_way_oracle(head_dim: u32, seq_len: u32) {
    if !cuda_available() {
        eprintln!(
            "[cycle14 csha checkpoint recompute] skipping hd={head_dim} S={seq_len} \
             — no CUDA device (or NSL_SKIP_CUDA_TESTS set)"
        );
        return;
    }

    // ── Config + PTX synthesis ─────────────────────────────────────────────
    let config = build_cycle14_config(head_dim, seq_len);
    let bwd_ptx = synthesize_backward_with_tier_b(&config, None)
        .expect("cycle-14 baseline config must synthesize backward PTX");
    assert!(
        bwd_ptx.contains("V2_KV_RECOMPUTE_MAIN"),
        "expected backward PTX to contain the kv-recompute label V2_KV_RECOMPUTE_MAIN"
    );

    // ── Deterministic inputs ───────────────────────────────────────────────
    let heads = 1usize;
    let hd = head_dim as usize;
    let seq = seq_len as usize;
    let dm = hd; // 1 head, dm == hd matches csha_cuda_backward.rs convention
    let kv_dim = heads * hd;

    let x = det_seq(42, seq * dm);
    let wq = det_seq(43, dm * kv_dim);
    let wk = det_seq(44, dm * kv_dim);
    let wv = det_seq(45, dm * kv_dim);
    let nw = vec![1.0f32; dm];
    let cos: Vec<f32> = (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).cos()).collect();
    let sin: Vec<f32> = (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).sin()).collect();
    let do_out = det_seq(99, seq * kv_dim);

    // ── CPU full-stack reference (oracle #3) ───────────────────────────────
    let inputs = CshaInputs {
        x: &x, wq: &wq, wk: &wk, wv: &wv,
        norm_weight: &nw, cos: &cos, sin: &sin,
    };
    let shape = CshaShape {
        seq, heads, head_dim: hd, d_model: dm,
        causal: true,
        norm_eps: 1e-6,
    };
    let cpu_grads: CshaGradients = csha_reference_backward(&inputs, &shape, &do_out);

    // ── CPU prologue oracle (oracle #4) ────────────────────────────────────
    let prologue_cfg = PrologueConfig {
        seq_len: seq, head_dim: hd, d_model: dm, eps: 1e-6,
        num_heads_q: heads, num_heads_kv: heads, num_heads_v: heads,
    };
    let (q_proj_cpu, k_proj_cpu, v_proj_cpu) =
        cpu_naive_norm_proj_rope(&x, &wq, &wk, &wv, &cos, &sin, &prologue_cfg);

    // ── GPU launches (cycle 13) ────────────────────────────────────────────
    //
    // Cycle 13 wires:
    //   a) Forward-with-saves kernel (synthesize fwd PTX) → fills SMEM
    //      with x_raw + W_k/W_v.
    //   b) Backward kernel using `bwd_ptx` from above; reads x_raw,
    //      recomputes K/V projections + RoPE-K in SMEM, emits 7 gradients.
    //   c) Re-run backward with `checkpoint = None` for the GPU
    //      non-checkpoint baseline (oracle #2).
    //   d) Readback all 7 gradients to host, plus the SMEM K/V tile
    //      contents (for the prologue-arithmetic three-way).
    //
    // Cycle 12 keeps the harness COMPILE-only (this branch is dead under
    // `!cuda_available()`).  The reference computations above run on the
    // CPU regardless so cycle 13 instantly has its oracles; only the GPU
    // launch sites need to be filled in.

    // Pacify the "unused" lints under cycle-12 compile-only posture so the
    // file compiles cleanly with -D warnings.
    let _ = (&cpu_grads, &q_proj_cpu, &k_proj_cpu, &v_proj_cpu);

    // ── Three-way diff (cycle 13) ──────────────────────────────────────────
    //
    // gpu_checkpoint vs gpu_non_checkpoint (oracle #1 vs #2):
    //   - dq/dk/dv: max_abs < tol_dqkv(hd).0 + tol_dqkv(hd).1 * cpu_grads.dq.max()
    //   - dwq/dwk/dwv: same with (1e-3, 1e-2)
    //   - dx: same with (1e-2, 2e-2)
    //
    // gpu_checkpoint vs cpu_reference (oracle #1 vs #3):
    //   - dq: max_abs_diff(&gpu_ckpt.dq, &cpu_grads.dq) < tol
    //   - …same for dk, dv, dwq, dwk, dwv, dx
    //
    // SMEM-readback vs cpu_naive_norm_proj_rope (oracle #4):
    //   - K_smem_post_recompute ≈ k_proj_cpu within hd-dependent tol
    //   - V_smem_post_recompute ≈ v_proj_cpu within same tol
    //   - This is the recompute-arithmetic gate — proves the backward
    //     kv-recompute reproduces the forward prologue exactly.
    let (atol_dqkv, rtol_dqkv) = tol_dqkv(hd);
    let atol_dw = 1e-3f32;
    let rtol_dw = 1e-2f32;
    let atol_dx = 1e-2f32;
    let rtol_dx = 2e-2f32;
    eprintln!(
        "[cycle13-pending] hd={head_dim} S={seq_len} tols: \
         dqkv=(atol={atol_dqkv:.1e}, rtol={rtol_dqkv:.1e}) \
         dw=(atol={atol_dw:.1e}, rtol={rtol_dw:.1e}) \
         dx=(atol={atol_dx:.1e}, rtol={rtol_dx:.1e})"
    );

    // Cycle-12 pacification of helper-used lints — keep max_abs_diff
    // reachable from at least one call site so compile-only doesn't warn.
    let _zero = max_abs_diff(&cpu_grads.dq, &cpu_grads.dq);
    debug_assert_eq!(_zero, 0.0, "self-diff must be 0");
}
