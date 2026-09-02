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
// Only the `_packed` entry point is used here: it is a strict superset —
// `packing = None` is exactly `csha_reference_backward` — and the packed
// oracles need the same call site to take a fixture.
use csha_reference::{csha_reference_backward_packed, CshaGradients, CshaInputs, CshaShape};

use std::ffi::CString;

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2_backward, smem_layout,
    synthesize_backward_with_tier_b, synthesize_flash_attention_ptx_v2,
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
    unsafe { std::env::set_var("NSL_SKIP_CUDA_TESTS", "1") };
    assert!(
        !cuda_available(),
        "cuda_available must respect NSL_SKIP_CUDA_TESTS"
    );
    unsafe { std::env::remove_var("NSL_SKIP_CUDA_TESTS") };
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

/// Smallest `max_abs` a correct f16-storing kernel can achieve against a
/// reference of this magnitude.
///
/// Six of the seven compared tensors are read back as f16 while `atol` here is
/// calibrated for f32 (dq atol=5e-4 against a floor of ~2e-3), so the absolute
/// half of `abs <= atol || rel <= rtol` was unsatisfiable no matter how correct
/// the arithmetic — leaving the verdict to `max_rel`, which divides by
/// `y.abs().max(1e-6)` and so explodes on near-zero elements. Same diagnosis
/// and same remedy as `csha_cycle15_bug1_ablations.rs`.
///
/// This is the REPRESENTABLE-PRECISION quantum at the reference's magnitude,
/// deliberately NOT a round-trip error. Round-tripping is the natural
/// estimator and is what the ablation harness uses, but it degenerates to
/// exactly 0 for the `path B vs path A` comparison, whose reference is itself
/// already f16 and therefore f16-exact — silently reinstating the
/// unsatisfiable f32 atol for that one comparison. Two independent f16
/// computations of the same quantity each carry their own accumulation error,
/// so the bound has to come from the magnitude.
fn f16_floor(reference: &[f32]) -> f32 {
    // f16 has 10 explicit mantissa bits, so relative precision is 2^-11.
    const F16_REL_PRECISION: f32 = 1.0 / 2048.0;
    let max_mag = reference.iter().map(|x| x.abs()).fold(0f32, f32::max);
    max_mag * F16_REL_PRECISION
}

/// Accumulation amplification allowed over the f16 storage floor. Reductions
/// run over S (attention) and d_model (projections) and f16 error through an
/// N-term reduction grows ~sqrt(N); sqrt(512)=22.6 is the loose bound and 16
/// is a deliberately tighter PRE-REGISTERED choice, fixed before checking
/// whether Path B passes rather than fitted to it. Path B legitimately does
/// more arithmetic than Path A (it recomputes K/V instead of loading them),
/// so it is expected to sit somewhat higher than Path A within this bound.
const F16_ACCUM_AMP: f32 = 16.0;

/// Storage-aware absolute tolerance: never below the f16 floor.
fn atol_f16_aware(atol: f32, reference: &[f32]) -> f32 {
    atol.max(F16_ACCUM_AMP * f16_floor(reference))
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
    build_cycle14_config_styled(head_dim, _seq_len, RopeStyle::Adjacent)
}

fn build_cycle14_config_styled(head_dim: u32, _seq_len: u32, rope_style: RopeStyle) -> FlashAttentionConfig {
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
        rope_style,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras::level1_with_fused_proj(1e-6)),
        checkpoint: Some(CheckpointExtras::full()),
    }.with_d_model(d_model)
}

/// Same as `build_cycle14_config_styled` but with PCA document packing on.
///
/// `segment_masked` forces `block_q == block_kv == seq_len` usage at the call
/// site: the runtime refuses `multi_tile_fused && segment_ids_ptr != 0`
/// (`nsl-runtime/src/flash_attention.rs:2112`), so the CSHA fused segmented
/// path is single-tile only. That refusal is independent of R7 and is NOT
/// what this harness is validating.
fn build_segmask_config(head_dim: u32, rope_style: RopeStyle) -> FlashAttentionConfig {
    let mut cfg = build_cycle14_config_styled(head_dim, 0, rope_style);
    cfg.segment_masked = true;
    cfg
}

/// Host-side PCA packing fixture: `segment_ids[seq]` + `doc_starts[257]`.
///
/// Layout is fixed by the kernel, not by taste:
///   * `segment_ids` — one `u16` per absolute position (`seq * 2` bytes).
///   * `doc_starts`  — `[MAX_NUM_DOCS + 1] = [257]` `i32`, 1028 bytes per
///     batch row (`pca_rope::MAX_NUM_DOCS`). Slots past the last document
///     hold `-1` sentinels.
struct PackedFixture {
    seg_ids: Vec<u16>,
    doc_starts: Vec<i32>,
}

impl PackedFixture {
    /// `seq` positions split into `num_docs` equal documents.
    ///
    /// Equal splits keep the fixture readable; the property that matters is
    /// that at least one document starts at a NON-ZERO offset, so
    /// `effective_pos != abs_pos` for some rows. A single document (or
    /// `doc_starts` all zero) makes the reset a no-op and the test degenerate
    /// — that is exactly the trap the existing Tier-A "null-guard" tests fall
    /// into, and why they never exercised the reset arithmetic.
    fn equal_docs(seq: usize, num_docs: usize) -> Self {
        assert!(num_docs >= 1 && seq % num_docs == 0, "seq must divide evenly");
        let per = seq / num_docs;
        let seg_ids: Vec<u16> = (0..seq).map(|s| (s / per) as u16).collect();
        let mut doc_starts = vec![-1i32; 257];
        for d in 0..=num_docs {
            doc_starts[d] = (d * per) as i32;
        }
        assert!(
            num_docs == 1 || doc_starts[1] > 0,
            "fixture must place at least one document at a non-zero start, \
             else the RoPE reset is unobservable"
        );
        Self { seg_ids, doc_starts }
    }

    fn as_packed_docs(&self) -> csha_reference::PackedDocs<'_> {
        csha_reference::PackedDocs {
            segment_ids: &self.seg_ids,
            doc_starts: &self.doc_starts,
        }
    }
}

/// Upload a packing fixture, returning `(segment_ids_dev, doc_starts_dev)`.
/// `(0, 0)` when unpacked — the kernel's null-guards then yield identity
/// positions and a segment-blind mask.
fn upload_packing(packing: Option<&PackedFixture>) -> (i64, i64) {
    let Some(p) = packing else { return (0, 0) };
    let seg_bytes = (p.seg_ids.len() * 2) as i64;
    let ds_bytes = (p.doc_starts.len() * 4) as i64;
    let seg_dev = unsafe { nsl_test_cuda_alloc(seg_bytes) };
    let ds_dev = unsafe { nsl_test_cuda_alloc(ds_bytes) };
    if seg_dev == 0 || ds_dev == 0 {
        // Release whichever one succeeded before unwinding, so an OOM here does
        // not also leak. `free_all` null-guards, so this is safe either way.
        free_all(&[seg_dev, ds_dev]);
        panic!(
            "packing device alloc returned null (seg={seg_bytes}B -> {seg_dev}, \
             doc_starts={ds_bytes}B -> {ds_dev})"
        );
    }
    unsafe {
        nsl_test_cuda_h2d(seg_dev, p.seg_ids.as_ptr() as i64, seg_bytes);
        nsl_test_cuda_h2d(ds_dev, p.doc_starts.as_ptr() as i64, ds_bytes);
    }
    (seg_dev, ds_dev)
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
#[ignore = "requires CUDA GPU"]
fn t_recompute_hd64_s512_bq32() {
    run_three_way_oracle(64, 512);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn t_recompute_hd64_s2048_bq32() {
    run_three_way_oracle(64, 2048);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn t_recompute_hd128_s512_bq32() {
    run_three_way_oracle(128, 512);
}

#[test]
#[ignore = "requires CUDA GPU"]
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
    // checkpoint(full) + rope_q now refuses UP FRONT (Path B kv-recompute
    // has a known gross numerical error — deliberate deferral-refusal),
    // which would preempt the R5 SMEM refusal this test pins. The budget
    // math under test is orthogonal to RoPE — disable it.
    cfg.rope_q = false;
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
    let mut cfg = build_cycle14_config(64, 512);
    // See g14_b: rope_q + checkpoint(full) refuses up front by design;
    // the sm_80 target-emission invariant needs a successful synthesis.
    cfg.rope_q = false;
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
    // `w_bytes`/`rope_bytes` complete the readback-size record the harness
    // keeps per buffer; backward sizes its reads off `qkv_bytes` and `d*_bytes`.
    #[allow(dead_code)]
    w_bytes: i64,
    #[allow(dead_code)]
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
    packing: Option<&PackedFixture>,
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
    // Combined module: fused kernel + `_mt_attn` interleaved twin. These
    // configs are multi-tile (S >> block_q), so the runtime's two-launch
    // dispatch inside nsl_flash_attention_csha_with_saves resolves the twin by
    // name — a single-tile-only `synthesize_flash_attention_ptx_v2` module
    // lacks the twin and faults with rc=500 (CUDA_ERROR_NOT_FOUND).
    //
    // The checkpoint carrier is a backward-only concern. `flash_attention_
    // kernel_name_v2` encodes it into the entry name, so synthesizing the
    // forward from the checkpoint config would name the entry with a checkpoint
    // suffix the runtime's lookup key (also checkpoint-suffixed) matches — but
    // the multi-tile TWIN is emitted without that suffix, so the two-launch
    // dispatch's twin lookup misses and returns rc=500. Strip checkpoint for
    // the forward (name + PTX both), matching the sister csha_cuda_backward
    // harness which uses a checkpoint-free forward config.
    let mut fwd_config = config.clone();
    fwd_config.checkpoint = None;
    // The `_mt_attn` twin exists only for the multi-tile two-launch dispatch.
    // `segment_masked` configs are single-tile by construction (the runtime
    // refuses multi-tile + segmented at flash_attention.rs:2112), so they take
    // the canonical single-kernel synthesizer — the same one the PCA Tier-A
    // launches use. Non-segmented configs keep the combined module verbatim.
    let fwd_ptx = if fwd_config.segment_masked {
        assert!(
            seq_len as i64 <= fwd_config.block_q,
            "segment_masked forward is single-tile only (seq_len={seq_len} > \
             block_q={}); the runtime refuses the multi-tile segmented dispatch",
            fwd_config.block_q
        );
        synthesize_flash_attention_ptx_v2(&fwd_config)
    } else {
        nsl_codegen::flash_attention_v2::synthesize_forward_multi_tile_combined(&fwd_config)
    };
    let fwd_name = CString::new(flash_attention_kernel_name_v2(&fwd_config)).unwrap();
    let fwd_smem_total = smem_layout::total_bytes(config);
    let fwd_smem_dyn = if smem_layout::needs_dynamic_smem(config) {
        fwd_smem_total as i64
    } else { 0 };

    let (fwd_seg_dev, fwd_doc_dev) = upload_packing(packing);

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
            // segment_ids_ptr, tier_b_ptx_ptr, tier_b_name_ptr, doc_starts_ptr,
            // num_docs_or_zero (PCA per-doc CTA Strategy 3 v1 — added post-merge).
            // num_docs stays 0: that selects the per-doc-CTA grid, which is a
            // different (default-OFF) kernel variant, not what this validates.
            fwd_seg_dev, 0, 0, fwd_doc_dev, 0,
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
            cos_dev, sin_dev, do_dev, fwd_seg_dev, fwd_doc_dev]);
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
    // Free the packing buffers ONLY here — AFTER the D2H above, never right
    // after the launch.
    //
    // `kernel_launch` does NOT synchronize on the default path: its sync is
    // gated on `sync_mode_enabled()` and it ends with an explicit
    // "cuCtxSynchronize removed — same-stream kernels are implicitly ordered"
    // note (`nsl-runtime/src/cuda/mod.rs`). Same-stream ORDERING does not help
    // here, because `nsl_test_cuda_free` -> `inner::free_device` is an eager
    // `cuMemFree_v2`, and the runtime's own comment states the rule: "a raw
    // `cuMemFree` is NOT stream-ordered: it must not run until the kernels
    // reading the buffer have finished" — which is precisely why
    // `defer_free_device` exists.
    //
    // The forward kernel's warp-0 cooperative prologue is still reading these
    // two buffers (`ld.global.u16` of segment_ids, `ld.global.s32` of
    // doc_starts). Freeing before a barrier races the kernel, and if the VA is
    // recycled by the backward's very next alloc the symptom is garbage
    // segment ids — indistinguishable from the SMEM aliasing bug this change
    // exists to fix. The D2H above is the first real barrier.
    //
    // This mirrors the established pattern in
    // `pca_tier_a_forward_correctness::launch_pca_ex`, which frees its
    // seg_ids/doc_starts only after its readback.
    free_all(&[fwd_seg_dev, fwd_doc_dev]);

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
    packing: Option<&PackedFixture>,
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
    // Cycle 14 diagnostic: PTX ASCII audit. ptxas rc=218 with
    // "Unexpected non-ASCII character" is loud but doesn't say WHICH
    // character. List every non-ASCII byte with its line and byte offset
    // so the source code can be located surgically.
    for (line_idx, line) in bwd_ptx_str.lines().enumerate() {
        if line.chars().any(|c| !c.is_ascii()) {
            let bad: Vec<(usize, char)> = line.char_indices()
                .filter(|(_, c)| !c.is_ascii())
                .collect();
            eprintln!(
                "  [{path_label}] PTX line {} contains non-ASCII chars: {:?} \\\n    LINE: {}",
                line_idx + 1, bad, line.trim_end()
            );
        }
    }
    // Dump raw PTX to scratch dir for diagnostic when launch fails.
    let dump_path = std::env::temp_dir().join(format!(
        "cycle14_{path_label}_hd{head_dim}_S{seq_len}.ptx"
    ).replace(' ', "_"));
    std::fs::write(&dump_path, &bwd_ptx_str).ok();
    eprintln!("  [{path_label}] PTX dump: {}", dump_path.display());

    if !bwd_ptx_str.ends_with('\0') { bwd_ptx_str.push('\0'); }
    let bwd_ptx = bwd_ptx_str.into_bytes();
    let bwd_name = CString::new(backward_kernel_name(bwd_cfg)).unwrap();

    // ── R-C14-4: SMEM accounting ───────────────────────────────────────────
    // G16-2 fix: use shared_mem_bytes_v2_backward which adds recompute_extra_bytes
    // when checkpoint=Some(Full). The checkpoint path writes a recomputed x_norm
    // scratch tile starting at recompute_xnorm_offset (immediately after the
    // backward_total_bytes region). Omitting this grant caused
    // CUDA_ERROR_ILLEGAL_ADDRESS in emit_kv_recompute -> emit_one_recompute_matmul.
    let bwd_smem_total = shared_mem_bytes_v2_backward(bwd_cfg);
    let bwd_needs_dyn = bwd_smem_total > 48 * 1024;
    let bwd_smem_dyn = if bwd_needs_dyn { bwd_smem_total as i64 } else { 0 };
    eprintln!(
        "  [{path_label}] bwd SMEM total={} bytes; dyn_request={} (needs_dynamic={})",
        bwd_smem_total, bwd_smem_dyn, bwd_needs_dyn
    );

    // ── Launch ─────────────────────────────────────────────────────────────
    let (bwd_seg_dev, bwd_doc_dev) = upload_packing(packing);

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
            // segment_ids, tier_b_ptx, tier_b_name, doc_starts, tier_b2_active,
            // num_docs_or_zero (PCA per-doc CTA Sprint 5 — added post-merge).
            bwd_seg_dev, 0, 0, bwd_doc_dev, 0, 0,
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
            dx_dev, dxn_dev, bwd_seg_dev, bwd_doc_dev]);
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
        dx_dev, dxn_dev, bwd_seg_dev, bwd_doc_dev]);
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
        let atol = atol_f16_aware(atol, y);
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
        let atol = atol_f16_aware(atol, y);
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
    run_three_way_oracle_styled(head_dim, seq_len, RopeStyle::Adjacent)
}

fn run_three_way_oracle_styled(head_dim: u32, seq_len: u32, rope_style: RopeStyle) {
    let config = build_cycle14_config_styled(head_dim, seq_len, rope_style);
    run_three_way_oracle_cfg(config, head_dim, seq_len, None)
}

/// Three-way oracle over an explicit config + optional PCA packing.
///
/// Comparisons:
///   * `A_vs_cpu` — kv_load backward (checkpoint=None) vs the CPU reference
///   * `B_vs_cpu` — kv_recompute backward (@checkpoint) vs the CPU reference
///   * `B_vs_A`   — the two GPU paths against each other
///
/// `B_vs_A` is the load-bearing one for a recompute bug: it isolates the
/// substitution from any shared upstream disagreement with the CPU oracle.
fn run_three_way_oracle_cfg(
    config: FlashAttentionConfig,
    head_dim: u32,
    seq_len: u32,
    packing: Option<&PackedFixture>,
) {
    if !cuda_available() {
        eprintln!(
            "[cycle14 csha checkpoint recompute] skipping hd={head_dim} S={seq_len} \
             — no CUDA device (or NSL_SKIP_CUDA_TESTS set)"
        );
        return;
    }

    // A synthesizer refusal is a legitimate outcome, not a failure: hd=128
    // under @checkpoint needs 220 KB of SMEM against a 99 KB device cap, so
    // the config cannot exist on this hardware. Asserting here reported that
    // structural impossibility as a numerics failure and hid the real
    // disposition of the hd=64 shapes.
    let bwd_ptx_check = match synthesize_backward_with_tier_b(&config, None) {
        Ok(ptx) => ptx,
        Err(e) => {
            eprintln!(
                "[cycle14] hd={head_dim} S={seq_len} SKIPPED — config refused \
                 by synthesizer: {e}"
            );
            return;
        }
    };
    assert!(
        bwd_ptx_check.contains("V2_KV_RECOMPUTE_MAIN"),
        "expected backward PTX to contain the kv-recompute label V2_KV_RECOMPUTE_MAIN"
    );

    let hd = head_dim as usize;
    let seq = seq_len as usize;
    let heads = 1usize;
    let dm = hd;

    // ── Task 3: forward launch with saves ──────────────────────────────────
    let artifacts = forward_launch_and_saves(&config, head_dim, seq_len, packing);

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
        rope_q: true,
        rope_style: config.rope_style,
    };
    // The oracle MUST model the packing too: doc-relative RoPE positions AND
    // the block-diagonal (cross-document) mask. Comparing a packed kernel to
    // an unpacked reference would fail for reasons unrelated to the recompute.
    let packed_docs = packing.map(|p| p.as_packed_docs());
    let cpu_grads: CshaGradients = csha_reference_backward_packed(
        &inputs, &shape, &artifacts.do_f32, packed_docs.as_ref(),
    );

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
    let gpu_a = launch_backward_path(&cfg_a, &artifacts, head_dim, seq_len, "path A", packing);
    eprintln!("[cycle14 path A] vs cpu_reference:");
    let fails_a = diff_summary("path A vs cpu_ref", &gpu_a, &cpu_grads, hd);

    // ── NEGATIVE CONTROL: the packed comparison must be able to FAIL ───────
    //
    // A GREEN packed oracle proves nothing unless a kernel that IGNORED the
    // packing would have been caught. So re-score the SAME GPU gradients
    // against an UNPACKED reference (absolute RoPE positions, no cross-doc
    // mask) and require that to FAIL. If it passes, the fixture is degenerate
    // — every position would be in document 0 with doc_start 0 — and the
    // packed run was measuring nothing. This is the cycle-15 "no base-config
    // control" lesson and the cycle-18 degenerate-probe lesson applied at the
    // point where they actually bite.
    if packing.is_some() {
        let cpu_unpacked = csha_reference_backward_packed(
            &inputs, &shape, &artifacts.do_f32, None,
        );
        let control_fails = diff_summary(
            "CONTROL path A vs UNPACKED cpu_ref (MUST FAIL)", &gpu_a, &cpu_unpacked, hd,
        );
        assert!(
            !control_fails.is_empty(),
            "negative control PASSED: the GPU gradients also match an UNPACKED \
             reference, so this fixture does not discriminate doc-relative RoPE \
             positions or the cross-document mask. The packed GREEN above is \
             vacuous — fix the fixture before trusting it."
        );
        eprintln!(
            "[cycle14 control] packed oracle is discriminating — unpacked \
             reference fails on {control_fails:?} as required"
        );
    }

    // ── Task 5 diagnostic: Path B (checkpoint=Some(Full)) — §5.3 EVIDENCE ──
    // Even when Path A baseline is RED vs cpu_reference, the Path A vs
    // Path B comparison is still load-bearing: it isolates whether the
    // kv_recompute path emits the same gradients as the kv_load path.
    // If A and B agree (within recompute-arithmetic tolerance) but BOTH
    // disagree with cpu_reference, the bug is somewhere upstream of the
    // dispatch fork (in the shared backward emission); the §5.3
    // mechanism's substitution itself is structurally sound. If A and B
    // disagree, the substitution introduces additional error and the
    // cycle-11/12 recompute path needs investigation.
    eprintln!(
        "[cycle14 path B] hd={head_dim} S={seq_len} launching backward with \
         checkpoint=Some(Full) (kv_recompute — §5.3 EVIDENCE path)"
    );
    let mut cfg_b = config.clone();
    cfg_b.checkpoint = Some(CheckpointExtras::full());
    let gpu_b = launch_backward_path(&cfg_b, &artifacts, head_dim, seq_len, "path B", packing);

    eprintln!("[cycle14 path B] vs cpu_reference:");
    let fails_b_vs_cpu = diff_summary("path B vs cpu_ref", &gpu_b, &cpu_grads, hd);
    eprintln!("[cycle14 path B] vs Path A (recompute correctness):");
    let fails_b_vs_a = diff_summary("path B vs path A", &gpu_b, &gpu_a, hd);

    // ── Cleanup before any panic ───────────────────────────────────────────
    unsafe { nsl_csha_free_backward_activations(artifacts.saves); }
    free_all(&[
        artifacts.q_dev, artifacts.k_dev, artifacts.v_dev,
        artifacts.out_dev, artifacts.lse_dev,
        artifacts.x_dev, artifacts.nw_dev,
        artifacts.wq_dev, artifacts.wk_dev, artifacts.wv_dev,
        artifacts.cos_dev, artifacts.sin_dev, artifacts.do_dev,
    ]);

    // ── Disposition (per spec §2) ──────────────────────────────────────────
    // - Path B vs cpu_ref GREEN  ⇒ §5.3 mechanism end-to-end validated; R0 retires.
    // - Path B vs cpu_ref RED + Path A vs cpu_ref GREEN ⇒ cycle-11/12 recompute bug; R0 stays.
    // - Both RED but Path B vs Path A GREEN ⇒ shared-emission bug upstream; §5.3 substitution OK.
    // - Both RED + Path B vs Path A RED ⇒ harness wiring OR independent bugs; full diagnosis cycle 15.
    let cpu_a_ok = fails_a.is_empty();
    let cpu_b_ok = fails_b_vs_cpu.is_empty();
    let ab_ok    = fails_b_vs_a.is_empty();
    eprintln!(
        "[cycle14 disposition] hd={head_dim} S={seq_len}: A_vs_cpu={} B_vs_cpu={} B_vs_A={}",
        if cpu_a_ok { "GREEN" } else { "RED" },
        if cpu_b_ok { "GREEN" } else { "RED" },
        if ab_ok    { "GREEN" } else { "RED" },
    );

    if !cpu_a_ok {
        panic!(
            "[cycle14 task5] Path A RED at hd={head_dim} S={seq_len}: \
             A_vs_cpu fails={:?}; B_vs_cpu fails={:?}; B_vs_A fails={:?}. \
             Spec §2 disposition: depends on B_vs_A outcome (see eprintln above).",
            fails_a, fails_b_vs_cpu, fails_b_vs_a
        );
    }
    if !cpu_b_ok {
        panic!(
            "[cycle14 task5] Path B RED vs cpu_reference at hd={head_dim} S={seq_len}: \
             B_vs_cpu fails={:?}; B_vs_A fails={:?}. Per spec §2: real cycle-11/12 \
             recompute bug surfaced; R0 stays with refined wording.",
            fails_b_vs_cpu, fails_b_vs_a
        );
    }

    // NOTE: no second cleanup here. The "Cleanup before any panic" block above
    // already freed `artifacts.saves` and every device buffer unconditionally.
    // A duplicate teardown used to sit at this point and double-freed all of
    // them (cuMemFree_v2 -> CUDA_ERROR_INVALID_VALUE, SIGABRT). It was
    // unreachable for as long as Path B was broken, because the two panics
    // above always fired first — so fixing the kv-recompute numerics is what
    // exposed it. Latent bug hidden behind another bug.

    eprintln!("[cycle14 task4] hd={head_dim} S={seq_len} Path A GREEN");
}

/// SINGLE-TILE discriminator for Path B. S=32 with block_q=block_kv=32 is one
/// q tile and one kv tile, so `%q_start == %k_start` and every multi-tile
/// indexing concern collapses. Reading the matrix:
///
///   - single tile PASSES, multi-tile FAILS  -> the remaining bug is in the
///     per-tile indexing of the kv-recompute (which tile's rows / which
///     tile's x_norm), the same class as the `%q_start` vs `%k_start` fix.
///   - BOTH fail                             -> the recompute math itself is
///     wrong independent of tiling, and multi-tile is a red herring.
///
/// This is the same discriminator that settled the n1_p0 fault (there it
/// showed S=32 still faulted, killing the tiling hypothesis outright).
#[test]
#[ignore = "requires CUDA GPU"]
fn t_recompute_hd64_s32_bq32_single_tile_discriminator() {
    run_three_way_oracle(64, 32);
}

/// HalfSplit (LLaMA / Qwen) end-to-end oracle. `emit_rope_pair_sweep` used to
/// implement Adjacent only — `emit_rope_k_epilogue` asserted on anything else,
/// which is the second half of what R7 refused. The CPU reference was likewise
/// Adjacent-only and is now style-aware, so this compares a HalfSplit kernel
/// against a HalfSplit oracle rather than against the wrong pairing.
#[test]
#[ignore = "requires CUDA GPU"]
fn t_recompute_hd64_s512_bq32_halfsplit() {
    run_three_way_oracle_styled(64, 512, RopeStyle::HalfSplit);
}

/// HalfSplit at single tile, so a failure here is the pairing itself rather
/// than any tiling interaction.
#[test]
#[ignore = "requires CUDA GPU"]
fn t_recompute_hd64_s32_bq32_halfsplit_single_tile() {
    run_three_way_oracle_styled(64, 32, RopeStyle::HalfSplit);
}

/// hd=128 under @checkpoint needs 217472 bytes against a 101376-byte device
/// cap at block_q=32. The refusal message says "Reduce head_dim,
/// block_q/block_kv, or d_model" — this probe measures whether any smaller
/// tile actually fits, so "hd=128 is impossible" is a measurement rather than
/// an assumption. GPU-free: pure layout arithmetic + the validator.
#[test]
fn probe_hd128_checkpoint_smem_vs_block_q() {
    for bq in [8u32, 16, 32] {
        let mut cfg = build_cycle14_config(128, 512);
        cfg.block_q = bq as i64;
        cfg.block_kv = bq as i64;
        let total = smem_layout::total_bytes(&cfg);
        let verdict = match synthesize_backward_with_tier_b(&cfg, None) {
            Ok(_) => "SYNTHESIZES".to_string(),
            Err(e) => {
                let first = e.split(';').next().unwrap_or(&e).to_string();
                format!("REFUSED: {first}")
            }
        };
        eprintln!("[hd128-probe] block_q=block_kv={bq:>2}  total_bytes={total:>7}  {verdict}");
        assert!(
            verdict.starts_with("REFUSED"),
            "hd=128 + @checkpoint unexpectedly synthesizes at block_q={bq}. If the \
             layout genuinely shrank, this is good news — unblock the hd=128 \
             oracles and delete this assertion. If it did not, the SMEM \
             validator has been weakened."
        );
    }

    // The load-bearing fact, measured rather than assumed: at the SMALLEST
    // legal tile (block_kv must be one of [16,32,64,128], so 16), the FORWARD
    // component alone is 107648 bytes against a 101376-byte cap — before any
    // backward or recompute storage is added. So no admissible block_q makes
    // hd=128 + @checkpoint fit on a 99 KB device, and "reduce block_q/block_kv"
    // from the refusal message cannot rescue this particular config.
    let mut smallest = build_cycle14_config(128, 512);
    smallest.block_q = 16;
    smallest.block_kv = 16;
    let forward_only = smem_layout::total_bytes(&smallest);
    assert!(
        forward_only > smem_layout::SMEM_DYNAMIC_BUDGET_BYTES,
        "forward-only SMEM at hd=128 block_q=16 is {forward_only}, which now FITS \
         within {} — hd=128 + @checkpoint may have become viable; re-measure.",
        smem_layout::SMEM_DYNAMIC_BUDGET_BYTES
    );
}

// ── PCA packing under @checkpoint (R7's last arm) ──────────────────────────

/// The checkpoint kv-recompute's x_norm scratch must NOT alias `seg_smem`.
///
/// Both regions used to be anchored at exactly `total_bytes +
/// backward_extra_bytes`:
///   * `backward/prelude.rs` sets `%seg_base = %shmem_base + backward_total_bytes`
///   * `recompute_xnorm_offset` returned that same expression
///
/// so step 2 of the kv-recompute (`emit_prologue_recompute_from_raw`, which
/// writes `block_q * head_dim * 2` bytes of f16 x_norm) overwrote the segment
/// ids that step 5's RoPE-K epilogue then read back with `ld.shared.u16`. The
/// resulting `sid` is an f16 mantissa pattern up to 65535, which indexes
/// `smem_doc_starts[sid]` up to 262140 bytes into a 1028-byte array.
///
/// This is GPU-free arithmetic, so it holds this invariant even on machines
/// with no CUDA device.
#[test]
fn g_segmask_xnorm_scratch_does_not_alias_seg_smem() {
    let seg_budget = nsl_codegen::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32;

    for hd in [32u32, 64] {
        let cfg = build_segmask_config(hd, RopeStyle::Adjacent);
        let seg_base = nsl_codegen::flash_attention_v2::phases::backward::prelude::backward_total_bytes(&cfg);
        let xn_off = smem_layout::recompute_xnorm_offset(&cfg);
        let xn_len = smem_layout::recompute_extra_bytes(&cfg) as u32;
        eprintln!(
            "[segmask-alias] hd={hd:>3} seg_base={seg_base:>6} seg_len={seg_budget:>6} \
             xnorm=[{xn_off}, {})",
            xn_off + xn_len
        );
        assert!(
            xn_off >= seg_base + seg_budget,
            "hd={hd}: recompute x_norm scratch starts at {xn_off}, inside the \
             seg_smem region [{seg_base}, {}) — the kv-recompute would overwrite \
             segment_ids before the RoPE-K epilogue reads them back",
            seg_base + seg_budget
        );
        // And the whole scratch must still fit inside the grant the launcher asks for.
        let granted = shared_mem_bytes_v2_backward(&cfg);
        assert!(
            xn_off + xn_len <= granted,
            "hd={hd}: x_norm scratch ends at {} but the launcher only grants \
             {granted} bytes",
            xn_off + xn_len
        );
    }

    // Byte-identical for unpacked configs — every currently-green Path B gate
    // runs with segment_masked=false and must be unaffected by the fix above.
    for hd in [32u32, 64, 128] {
        let cfg = build_cycle14_config(hd, 512);
        assert_eq!(
            smem_layout::recompute_xnorm_offset(&cfg),
            smem_layout::total_bytes(&cfg) + smem_layout::backward_extra_bytes(&cfg),
            "unpacked hd={hd} offset moved — the aliasing fix must be a no-op \
             when !segment_masked"
        );
    }
}

/// Measures whether a packed `@checkpoint` config fits the 99 KB cap, and at
/// which head_dim. `segment_masked` costs a flat DEFAULT_SMEM_SEGMENT_BUDGET
/// (32 KB) on top of the backward layout, which is a large fraction of the
/// budget — so which shapes are admissible is a measurement, not a guess.
#[test]
fn probe_segmask_checkpoint_smem_budget() {
    for hd in [32u32, 64, 128] {
        let cfg = build_segmask_config(hd, RopeStyle::Adjacent);
        let bwd_total = nsl_codegen::flash_attention_v2::phases::backward::prelude::backward_total_bytes(&cfg);
        let granted = shared_mem_bytes_v2_backward(&cfg);
        let verdict = match synthesize_backward_with_tier_b(&cfg, None) {
            Ok(_) => "SYNTHESIZES".to_string(),
            Err(e) => format!("REFUSED: {}", e.split(';').next().unwrap_or(&e)),
        };
        eprintln!(
            "[segmask-budget] hd={hd:>3} bwd_total={bwd_total:>6} granted={granted:>6} \
             cap={} {verdict}",
            smem_layout::SMEM_DYNAMIC_BUDGET_BYTES
        );
    }
}

/// Non-vacuity wrapper for the packed oracles.
///
/// `run_three_way_oracle_cfg` SKIPS (and passes) when the synthesizer refuses.
/// That is right for structurally-impossible shapes like hd=128, but for these
/// gates a refusal is precisely the regression to catch: if R7 came back or R12
/// widened again, every packed test below would go green while running nothing.
/// Assert synthesis FIRST so the refusal path cannot masquerade as a pass.
fn run_segmask_oracle(
    cfg: FlashAttentionConfig,
    head_dim: u32,
    seq_len: u32,
    packing: &PackedFixture,
) {
    synthesize_backward_with_tier_b(&cfg, None).unwrap_or_else(|e| {
        panic!(
            "segment_masked + rope_q + @checkpoint must synthesize at hd={head_dim}; \
             got a refusal: {e}\nIf this is intentional, the packed oracles below are \
             no longer validating anything and must be re-scoped, not left green."
        )
    });
    run_three_way_oracle_cfg(cfg, head_dim, seq_len, Some(packing));
}

/// Three-way oracle for PCA document packing under `@checkpoint` — R7's last arm.
///
/// The shape is forced by two INDEPENDENT constraints, both measured rather
/// than assumed:
///   * **single tile** (`seq_len == block_q`) — the runtime refuses
///     `multi_tile_fused && segment_ids_ptr != 0`
///     (`nsl-runtime/src/flash_attention.rs:2112`). That refusal predates R7
///     and is not what this validates.
///   * **head_dim = 32** — `segment_masked` costs a flat 32 KB `seg_smem`
///     region, so hd=64 needs 127360 bytes against the 101376 cap. See
///     `probe_segmask_checkpoint_smem_budget` for the numbers.
///
/// The fixture packs 2 documents of 16, so rows 16..32 have
/// `effective_pos = abs_pos - 16` and the reset is actually observable. A
/// single document, or `doc_starts` all zero, would make this test pass
/// without exercising any of the reset arithmetic.
#[test]
#[ignore = "requires CUDA GPU"]
fn t_segmask_recompute_hd32_s32_two_docs() {
    let cfg = build_segmask_config(32, RopeStyle::Adjacent);
    let packing = PackedFixture::equal_docs(32, 2);
    run_segmask_oracle(cfg, 32, 32, &packing);
}

/// HalfSplit twin of `t_segmask_recompute_hd32_s32_two_docs`.
#[test]
#[ignore = "requires CUDA GPU"]
fn t_segmask_recompute_hd32_s32_two_docs_halfsplit() {
    let cfg = build_segmask_config(32, RopeStyle::HalfSplit);
    let packing = PackedFixture::equal_docs(32, 2);
    run_segmask_oracle(cfg, 32, 32, &packing);
}

/// Four unequal-ish documents, to catch an off-by-one that two equal 16-row
/// documents would not: doc boundaries at 8/16/24 land mid-warp rather than on
/// the tile's half.
#[test]
#[ignore = "requires CUDA GPU"]
fn t_segmask_recompute_hd32_s32_four_docs() {
    let cfg = build_segmask_config(32, RopeStyle::Adjacent);
    let packing = PackedFixture::equal_docs(32, 4);
    run_segmask_oracle(cfg, 32, 32, &packing);
}

/// The static-vs-dynamic SMEM routing predicate must agree with the array the
/// static path actually declares.
///
/// `backward_needs_dynamic_smem` decides the path from a byte count; the static
/// branch then emits `.shared` sized `backward_total_bytes + seg_overhead +
/// recompute_extra`. If the predicate omits `recompute_extra`, a config can be
/// routed static on a number BELOW the 48 KB hardware cap and then declare an
/// array ABOVE it — synthesis succeeds and ptxas rejects the module at JIT,
/// because `validate_scalar_v2_config` only checks the 99 KB DYNAMIC cap.
///
/// The witness below is the exact band where that happens (d_model=36 tunes
/// `backward_total_bytes` to just under the static cap).
#[test]
fn g_static_smem_routing_accounts_for_recompute_extra() {
    let mut cfg = build_cycle14_config(32, 32);
    cfg.block_q = 32;
    cfg.block_kv = 32;
    if let Some(e) = cfg.csha.as_mut() {
        e.d_model = 36;
    }

    let bwd_total =
        nsl_codegen::flash_attention_v2::phases::backward::prelude::backward_total_bytes(&cfg);
    let recompute_extra = smem_layout::recompute_extra_bytes(&cfg) as u32;
    let static_cap = smem_layout::SMEM_BUDGET_BYTES;
    let declared = bwd_total + recompute_extra;
    eprintln!(
        "[static-routing] bwd_total={bwd_total} recompute_extra={recompute_extra} \
         declared={declared} static_cap={static_cap}"
    );

    // Only meaningful while the witness actually straddles the cap.
    if !(bwd_total <= static_cap && declared > static_cap) {
        eprintln!(
            "[static-routing] witness no longer straddles the static cap \
             (bwd_total={bwd_total}, declared={declared}, cap={static_cap}) — \
             re-tune d_model rather than leaving this gate vacuous"
        );
        return;
    }

    // Straddling the cap, the config MUST be routed to the dynamic path.
    // `synthesize_backward_with_tier_b` is the observable: the emitted PTX must
    // use `.extern .shared`, never a static `.shared` array over the cap.
    let ptx = synthesize_backward_with_tier_b(&cfg, None)
        .expect("witness config must synthesize (it is under the 99 KB dynamic cap)");
    assert!(
        ptx.contains(".extern .shared"),
        "config straddling the static cap (bwd_total={bwd_total} <= {static_cap} < \
         declared={declared}) must route to the DYNAMIC smem path; it emitted a \
         static declaration that ptxas will reject at JIT"
    );
}

/// R12 must not be bypassable, because it now guards an SMEM offset collision.
///
/// After the aliasing fix, `recompute_xnorm_offset` and
/// `tier_b_range_table_offset(Backward)` are the SAME byte. R12 is what keeps a
/// packed checkpoint config from being emitted with Tier-B's range table and the
/// kv-recompute x_norm scratch stacked at one address. A test seam that
/// bypassed it would hand out memory-unsafe PTX, so the seam was removed.
#[test]
fn g_r12_tier_b_offset_collision_is_real_and_unbypassable() {
    let cfg = build_segmask_config(32, RopeStyle::Adjacent);
    let xn_off = smem_layout::recompute_xnorm_offset(&cfg);
    let tier_b_off = smem_layout::tier_b_range_table_offset(&cfg, smem_layout::Direction::Backward);
    eprintln!("[r12-collision] xnorm_off={xn_off} tier_b_off={tier_b_off}");
    assert_eq!(
        xn_off, tier_b_off,
        "if these ever differ the collision is gone and R12 may be re-examined \
         on its original (tile-active predicate) merits alone — update this gate \
         and the R12 comment together"
    );

    // With Tier-B admitted, R12 must refuse — with NO environment variable able
    // to turn it off.
    let seq_len = 4096u32;
    let residency = nsl_codegen::pca_segment::SegmentResidency::Shared;
    if !nsl_codegen::pca_tilerange::should_emit_tier_b(&cfg, seq_len as u64, residency) {
        eprintln!("[r12-collision] Tier-B not admitted at seq_len={seq_len}; re-pick a shape");
        return;
    }
    // SAFETY of this test: setting the old bypass var must have no effect.
    unsafe { std::env::set_var("NSL_VALIDATE_PATH_B", "1") };
    let err = synthesize_backward_with_tier_b(&cfg, Some((seq_len, residency)));
    unsafe { std::env::remove_var("NSL_VALIDATE_PATH_B") };
    let err = err.expect_err(
        "R12 must refuse segment_masked + Tier-B + @checkpoint even with \
         NSL_VALIDATE_PATH_B set — the seam was removed precisely because \
         bypassing R12 yields two SMEM regions at the same base",
    );
    assert!(
        err.contains("paged-segment-masked composition deferred"),
        "expected R12's message; got: {err}"
    );
}

/// Does a SMALLER tile let head_dim=64 packed+checkpoint through?
#[test]
fn probe_segmask_checkpoint_smem_vs_block_q() {
    for hd in [32u32, 64, 128] {
        for bq in [16i64, 32, 64] {
            let mut cfg = build_segmask_config(hd, RopeStyle::Adjacent);
            cfg.block_q = bq;
            cfg.block_kv = bq;
            let granted = shared_mem_bytes_v2_backward(&cfg);
            // `ds_compute` hard-asserts block_kv==32, so some shapes PANIC
            // rather than refusing. Catch it so the probe can report the whole
            // surface instead of dying on the first cell.
            let verdict = std::panic::catch_unwind(|| {
                match synthesize_backward_with_tier_b(&cfg, None) {
                    Ok(_) => "SYNTHESIZES".to_string(),
                    Err(e) => format!(
                        "REFUSED: {}",
                        e.split(';').next().unwrap_or(&e).chars().take(58).collect::<String>()
                    ),
                }
            })
            .unwrap_or_else(|_| "PANICS (emitter assert)".to_string());
            eprintln!("[segmask-shape] hd={hd:>3} bq=bkv={bq:>3} granted={granted:>7} {verdict}");
        }
    }

    // d_model is an independent knob (the SMEM weight tile is
    // 3 * head_dim * min(d_model, 256) * 2), so sweep it too: a small d_model
    // is the obvious way someone could try to squeeze an UNVALIDATED head_dim
    // under the cap.
    for hd in [32u32, 64, 128] {
        for dm in [8u32, 16, 32, 64, 128, 256] {
            let mut cfg = build_segmask_config(hd, RopeStyle::Adjacent);
            if let Some(e) = cfg.csha.as_mut() { e.d_model = dm; }
            let granted = shared_mem_bytes_v2_backward(&cfg);
            let verdict = std::panic::catch_unwind(|| {
                match synthesize_backward_with_tier_b(&cfg, None) {
                    Ok(_) => "SYNTHESIZES".to_string(),
                    Err(_) => "REFUSED".to_string(),
                }
            })
            .unwrap_or_else(|_| "PANICS".to_string());
            eprintln!("[segmask-dmodel] hd={hd:>3} d_model={dm:>4} granted={granted:>7} {verdict}");
            // R14 bounds this to the MEASURED envelope. Anything that
            // synthesizes here must have a green three-way oracle behind it.
            if hd != 32 {
                assert_ne!(
                    verdict, "SYNTHESIZES",
                    "head_dim={hd} d_model={dm} packed+@checkpoint SYNTHESIZES but has \
                     NO GPU validation (only head_dim=32 is measured). Either validate \
                     it on hardware and widen R14's list, or keep the refusal — do not \
                     let an unvalidated composition emit silently."
                );
            }
        }
    }
}
