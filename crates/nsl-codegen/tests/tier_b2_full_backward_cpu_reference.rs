//! Layer-1 full 7-gradient hybrid-backward parity gate vs CPU reference (Phase 3 T8).
//!
//! This is the central deliverable of CSHA Tier B.2 Phase 3 closure: it proves the
//! 4-kernel hybrid backward (`d_prepass` -> `dq` -> `dkdv` -> `proj_backward`),
//! launched through the production FFI `nsl_flash_attention_csha_backward` with
//! `tier_b2_active = 1`, produces ALL SEVEN gradients
//! (dQ, dK, dV, dWq, dWk, dWv, dx) correctly — each compared independently to a
//! CPU f64-precision ground-truth reference.
//!
//! All GPU tests are `#[ignore]` + require `feature="cuda"`. Manual GPU invocation
//! (the B1Forward path additionally needs nsl-test's cuda feature):
//!     cargo test -p nsl-codegen --features "cuda,nsl-test/cuda" \
//!         --test tier_b2_full_backward_cpu_reference -- --ignored --nocapture
//!
//! ## Gate (spec §8) — assert ALL 7, non-vacuously
//! For each gradient g in {dQ, dK, dV, dWq, dWk, dWv, dx}:
//!   - relative error vs the CPU f64 reference within the tiered tolerance for hd;
//!   - ZERO-OUTPUT GUARD: `max|g_gpu| >= 0.25 * max|g_ref|` (a dropped/zeroed
//!     gradient FAILS — specifically catches a dropped dWq/dWk/dWv/dx);
//!   - substantial inputs (the harness's normal generators + dO upscale so the
//!     reference magnitudes clear the f16 noise floor — avoids a vacuous gate).
//! dK uses a 1e-4 reference floor (it is ~3 orders smaller than dV).
//!
//! ## Smoke-intersection config (the ONLY validated regime)
//!   head_dim in {64, 128}, heads == 1, d_model == head_dim, seq == block_q,
//!   batch == 1, rope_q == false, causal == true, csha.level == 2, gpu_sm == 80.
//!   - hd=64  -> {heads=1, d_model=64,  seq=block_q=64}
//!   - hd=128 -> {heads=1, d_model=128, seq=block_q=32}  (block_q=32 keeps the
//!     backward SMEM under the 99 KB dynamic cap AND keeps the proj kernel in its
//!     `seq == block_q` single-q-block scope; this matches the dq/dkdv validated
//!     hd=128 tiling where effective_bq = min(block_q, 32) = 32).
//!
//! ## Design notes
//! - The attention saves (q/k/v_saved, o, row_max, row_sum) drive dQ/dK/dV. The
//!   projection inputs (x_raw, Wq/Wk/Wv, norm_weight) drive dWq/dWk/dWv/dx. The
//!   projection backward is a standalone chain-rule combining (dQ,dK,dV) with
//!   (x_raw, W, norm_weight); it need not be causally linked to the attention
//!   forward, so the harness generates the projection inputs independently.
//! - The proj-stage CPU reference (`cpu_naive_backward_proj`) is fed the *GPU*
//!   dQ/dK/dV read back from the hybrid, so dWq/dWk/dWv/dx parity isolates the
//!   proj kernel rather than compounding the (already-gated) dQ/dK/dV error.
//! - dtype reconciliation: the dq/dkdv kernels write dQ/dK/dV as f32 INTO
//!   PRIVATE F32 SCRATCH BUFFERS inside `csha_tier_b2_backward_launch` (Sprint
//!   1 T1.2); the proj kernel reads those same f32 scratches and writes
//!   dWq/dWk/dWv as f16, dx as f32. After proj_backward, the launch converts
//!   each f32 scratch into the production f16 destination buffers via
//!   `csha_bwd_convert_f32_to_f16`. This matches the wengert lowering's f16
//!   allocations (`nsl_tensor_zeros_f16_on`). The proj kernel reads
//!   x_raw/norm_weight as f32 (`ld.global.f32`) and Wq/Wk/Wv as f16
//!   (`ld.global.b16`). We therefore upload x_raw/norm_weight as f32 and W as
//!   f16, allocate the dQ/dK/dV test destinations as f16 (mirroring production
//!   wengert), and read them back as f16 → widen to f32 for the rel-error
//!   comparison. The CPU reference takes the same values as f16 (round-trip
//!   exact via `f16::to_f32`), so both paths see bit-identical numbers.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §8

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::shared_mem_bytes_v2_backward;
use nsl_codegen::flash_attention_v2::tier_b2::backward::synthesize_tier_b2_backward;
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d,
};
use nsl_test::cpu_naive_backward::{
    cpu_naive_backward_dkdv, cpu_naive_backward_dq, cpu_naive_backward_proj,
};
use nsl_test::diagnostic_mode::{
    compute_forward_for_test, generate_d_o, generate_forward_inputs, FSource,
};

extern "C" {
    /// Production hybrid-backward FFI (53-param). See `nsl-runtime`
    /// `flash_attention.rs::nsl_flash_attention_csha_backward`.
    #[allow(clippy::too_many_arguments)]
    fn nsl_flash_attention_csha_backward(
        q_ptr: i64, k_ptr: i64, v_ptr: i64,
        out_ptr: i64,
        logsumexp_ptr: i64,
        scale_bits: i64,
        batch: i64, heads: i64, seq_len: i64, head_dim: i64,
        block_table_ptr: i64,
        k_pool_ptr: i64, v_pool_ptr: i64,
        block_size: i64,
        cos_ptr: i64, sin_ptr: i64,
        seq_ids_ptr: i64, seq_lens_ptr: i64,
        shared_mem_bytes: i64,
        ptx_ptr: i64, name_ptr: i64,
        block_q: i64, block_kv: i64,
        causal: i64,
        x_ptr: i64, norm_weight_ptr: i64,
        wq_ptr: i64, wk_ptr: i64, wv_ptr: i64, wo_ptr: i64,
        rmsnorm_eps_bits: i64,
        active_heads: i64, d_model: i64,
        q_proj_ptr: i64, k_proj_ptr: i64, v_proj_ptr: i64,
        row_max_ptr: i64, row_sum_ptr: i64,
        x_raw_ptr: i64,
        do_ptr: i64,
        dq_ptr: i64, dk_ptr: i64, dv_ptr: i64,
        dwq_ptr: i64, dwk_ptr: i64, dwv_ptr: i64,
        dx_ptr: i64,
        dx_norm_ptr: i64,
        segment_ids_ptr: i64,
        tier_b_ptx_ptr: i64,
        tier_b_name_ptr: i64,
        doc_starts_ptr: i64,
        tier_b2_active: i64,
    ) -> i64;
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("[tier_b2_full] skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("[tier_b2_full] skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

/// Null-terminate a PTX string for cuModuleLoadData.
fn ptx_to_cstr_bytes(ptx: &str) -> Vec<u8> {
    let mut bytes = ptx.as_bytes().to_vec();
    if bytes.last() != Some(&b'\n') {
        bytes.push(b'\n');
    }
    bytes.push(0);
    bytes
}

/// Tiered RELATIVE tolerance for the attention gradients (dQ/dK/dV). Same
/// schedule as the dq/dkdv standalone gates (spec §6.1). Sprint 1 T1.2:
/// dQ/dK/dV now pass through an f32->f16 conversion in
/// `csha_tier_b2_backward_launch` before reaching the test buffers; the f16
/// quantization adds ~1e-3 rel error, well inside this schedule.
fn rel_tol_attn(hd: u32) -> f32 {
    if hd >= 128 {
        8e-2
    } else if hd >= 64 {
        5e-2
    } else {
        3e-2
    }
}

/// Tiered RELATIVE tolerance for the WEIGHT gradients (dWq/dWk/dWv).
///
/// The dproj kernel accumulates `dW = x_norm^T @ dY` in f32 but STORES the
/// result as f16 (`cvt.rn.f16.f32`, by design — weight gradients are consumed
/// f16 downstream) and recomputes x_norm with `rsqrt.approx.f32`. The GPU folds
/// the RMSNorm gain into x_norm = (x/rms)*gamma BEFORE the projection; the CPU
/// reference (`cpu_naive_backward_proj`) applies the SAME gamma in its dW path,
/// so the math agrees up to numerical precision.
///
/// Sprint 1 T1.2: the GPU proj_backward consumes dQ/dK/dV from F32 scratches
/// (full precision — the f32->f16 conversion to the production destination
/// buffers happens AFTER proj_backward finishes). The CPU reference, however,
/// is fed the f16-quantized dQ/dK/dV that we read back from device. That
/// asymmetric f16 round-trip on the proj inputs perturbs the CPU dW result by
/// up to ~1e-2 in the small-input B1Forward regime at hd=128 (measured on RTX
/// 5070 Ti: B1Forward hd=128 dWq rel = 1.06e-2; CpuNaive hd=128 dWk rel =
/// 3.4e-3; all others <= 3e-3). The 2e-2 bound covers the worst observed
/// magnitude with comfortable headroom while keeping the zero-output guard
/// (0.25x) as the real anti-drop check. The previous 3e-3 bound was the
/// pre-T1.2 ceiling when the test allocated dQ/dK/dV as f32 throughout — no
/// longer the production contract.
fn rel_tol_dweight(_hd: u32) -> f32 {
    2e-2
}

/// Tiered RELATIVE tolerance for dx (the input-tensor gradient).
///
/// dx is stored f32 and the dRMSNorm closed form is exact; the only error is
/// the rsqrt.approx in rms_inv and the f16->f32 reads of the GPU dQ/dK/dV that
/// feed it (the SAME values the CPU ref consumes), so dx matches to ~1e-6.
/// A tight 2e-2 schedule keeps this gradient honestly gated.
fn rel_tol_dx(hd: u32) -> f32 {
    if hd >= 128 {
        3e-2
    } else {
        2e-2
    }
}

/// dO upscale so the true gradients are substantial (non-vacuous gate). The
/// shared `generate_d_o` emits x0.1 values; the gradients are linear in dO, so
/// scaling lifts the whole chain (D/dP/dS/dQ/dK/dV and the proj gradients)
/// above the f16 noise floor self-consistently. Same constant as the
/// standalone dq/dkdv gates.
const GATE_DO_SCALE: f32 = 1024.0;

const RMSNORM_EPS: f32 = 1e-5;

/// Smoke-intersection config for a given head_dim.
///
/// Sprint 4 (paper §4.1 intra-tile causal masking): the dq/dkdv MMA kernels
/// now apply per-element causal masking BEFORE the softmax exp (set S=-INF
/// for kv_abs > q_abs when `config.causal`). The Phase-4 deferral has been
/// closed; the smoke config defaults to causal=false because the historical
/// gates were calibrated against that regime, but the Sprint 4 sweeps below
/// rerun the SAME gate with causal=true and the new per-element mask.
fn smoke_cfg(hd: i64, seq: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: seq,
        block_kv: seq,
        head_dim: hd,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: hd as u32, // d_model == head_dim (smoke)
            active_heads: 1,    // heads == 1 (smoke)
            ..Default::default()
        }),
    }
}

// ============================================================================
// Projection-input generators (independent of the attention forward).
// ============================================================================

/// Deterministic x_raw [batch=1, heads=1, seq, d_model] f16 (pre-RMSNorm input).
fn gen_x_raw(seq: usize, d_model: usize) -> Vec<half::f16> {
    (0..seq * d_model)
        .map(|i| half::f16::from_f32((i as f32 * 0.053).sin() * 0.5 + 0.1))
        .collect()
}

/// Deterministic weight [d_model, kv_dim] f16. `phase`/`freq` distinguish W{q,k,v}.
fn gen_weight(d_model: usize, kv_dim: usize, freq: f32, phase: f32) -> Vec<half::f16> {
    (0..d_model * kv_dim)
        .map(|i| half::f16::from_f32(((i as f32) * freq + phase).cos() * 0.25))
        .collect()
}

/// Deterministic norm_weight [d_model] f16 (RMSNorm gamma).
fn gen_norm_weight(d_model: usize) -> Vec<half::f16> {
    (0..d_model)
        .map(|i| half::f16::from_f32(1.0 + (i as f32 * 0.017).sin() * 0.1))
        .collect()
}

// ============================================================================
// GPU sweeps
// ============================================================================

#[test]
#[ignore]
fn tier_b2_full_backward_sweep_cpu_naive() {
    // Phase 3 T8 — full 7-gradient gate, CpuNaive forward.
    // hd=64  -> seq=block_q=64 ; hd=128 -> seq=block_q=32 (SMEM + proj scope).
    for &hd in &[64i64, 128] {
        let seq = if hd >= 128 { 32 } else { 64 };
        validate_full_backward_for_source(&smoke_cfg(hd, seq), FSource::CpuNaive, seq as usize);
    }
}

#[test]
#[ignore]
fn tier_b2_full_backward_sweep_b1_forward() {
    // Phase 3 T8 — full 7-gradient gate, B.1 GPU forward + adapter.
    // B.1 single-block precondition pins seq=32 for all hd; seq==block_q holds.
    for &hd in &[64i64, 128] {
        validate_full_backward_for_source(&smoke_cfg(hd, 32), FSource::B1Forward, 32);
    }
}

// ============================================================================
// Sprint 4: intra-tile causal element masking (paper §4.1)
// ============================================================================
//
// These sweeps rerun the full 7-gradient parity gate with causal=true and the
// new per-element causal mask emitted by dq.rs / dkdv.rs. The CPU naive
// reference at `cpu_naive_backward_*` already applies element-level masking
// (`k_limit = qi + 1`), so the GPU output must match it under causal=true.

#[test]
#[ignore]
fn tier_b2_full_backward_sweep_cpu_naive_causal() {
    // Sprint 4 — full 7-gradient gate, CpuNaive forward, causal=true.
    for &hd in &[64i64, 128] {
        let seq = if hd >= 128 { 32 } else { 64 };
        let mut cfg = smoke_cfg(hd, seq);
        cfg.causal = true;
        validate_full_backward_for_source(&cfg, FSource::CpuNaive, seq as usize);
    }
}

#[test]
#[ignore]
fn tier_b2_full_backward_sweep_b1_forward_causal() {
    // Sprint 4 — full 7-gradient gate, B.1 GPU forward + adapter, causal=true.
    // B.1 single-block precondition pins seq=32 for all hd; seq==block_q holds.
    for &hd in &[64i64, 128] {
        let mut cfg = smoke_cfg(hd, 32);
        cfg.causal = true;
        validate_full_backward_for_source(&cfg, FSource::B1Forward, 32);
    }
}

/// Sprint 10: structural codegen check that proj_backward's PTX includes
/// the inverse-RoPE phase when (and only when) `config.rope_q` is true.
/// Mirrors the Sprint-4 intra-tile-causal-mask emission check pattern.
///
/// This is the COMPILE-TIME half of the Sprint 10 deliverable. The
/// runtime/numerical half (full 7-gradient parity at rope_q=true) is
/// deferred until a rope_q-aware CPU naive forward/backward reference
/// lands; the existing CPU naive references at
/// `crates/nsl-test/src/cpu_naive_{forward,backward}.rs` do not yet apply
/// RoPE (no cos/sin inputs, no rotation step), so a numerical parity gate
/// can't be authored against them. Once the references gain a rope_q
/// branch, the existing `validate_full_backward_for_source` harness can
/// be re-used to add a rope_q=true sweep.
#[test]
fn tier_b2_proj_backward_emits_drope_when_rope_q_true() {
    use nsl_codegen::flash_attention_v2::tier_b2::backward::proj_backward::synthesize_proj_backward;

    // rope_q=true: the inverse-RoPE phase must be present, sequenced
    // strictly between emit_xnorm_recompute and emit_dproj.
    let mut cfg_rope = smoke_cfg(64, 64);
    cfg_rope.rope_q = true;
    let ptx_rope = synthesize_proj_backward(&cfg_rope).expect("synth ok");
    assert!(
        ptx_rope.contains("V2_BWD_DROPE_GUARD_0:")
            && ptx_rope.contains("V2_BWD_DROPE_Q_LOOP_0:")
            && ptx_rope.contains("V2_BWD_DROPE_K_LOOP_0:"),
        "rope_q=true: proj_backward PTX must include the inverse-RoPE phase \
         (V2_BWD_DROPE_GUARD/Q_LOOP/K_LOOP labels)"
    );

    // rope_q=false: zero dRoPE bytes — the codegen-side gate keeps the
    // emitted PTX byte-identical to the pre-Sprint-10 baseline.
    let ptx_no_rope = synthesize_proj_backward(&smoke_cfg(64, 64))
        .expect("synth ok");
    assert!(
        !ptx_no_rope.contains("V2_BWD_DROPE"),
        "rope_q=false: NO dRoPE label may appear (byte-identity guarantee)"
    );
    assert!(
        !ptx_no_rope.contains("Tier C dRoPE: rope_q=false"),
        "rope_q=false: emit_drope's internal no-op comment must NOT appear — \
         the call must be gated externally for byte identity"
    );
}

/// Sprint 4 (paper §4.1): structural verification that BOTH the tile-level
/// causal skip predicate AND the new intra-tile per-element mask are emitted
/// when `config.causal` is true. The intra-tile mask is gated entirely by the
/// compile-time `config.causal` flag — when false, the marker MUST NOT appear
/// (zero PTX overhead in the non-causal path).
#[test]
fn tier_b2_intra_tile_causal_mask_emission() {
    use nsl_codegen::flash_attention_v2::tier_b2::backward::dkdv::synthesize_dkdv_kernel;
    use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

    // causal=true: both kernels emit the tile-skip predicate AND the per-element mask.
    let mut cfg_causal = smoke_cfg(64, 64);
    cfg_causal.causal = true;
    let dq_c = synthesize_dq_kernel(&cfg_causal).expect("dq synth (causal)");
    let dkdv_c = synthesize_dkdv_kernel(&cfg_causal).expect("dkdv synth (causal)");

    for (name, ptx) in [("dq", &dq_c), ("dkdv", &dkdv_c)] {
        assert!(
            ptx.contains("%p_causal_active"),
            "{name} (causal=true): expected tile-level causal-skip predicate"
        );
        assert!(
            ptx.contains("V2_INTRA_TILE_CAUSAL_MASK"),
            "{name} (causal=true): expected intra-tile per-element causal mask marker"
        );
    }

    // causal=false: NO intra-tile mask emitted (zero overhead path).
    let cfg_noncausal = smoke_cfg(64, 64); // causal=false by default
    let dq_nc = synthesize_dq_kernel(&cfg_noncausal).expect("dq synth (non-causal)");
    let dkdv_nc = synthesize_dkdv_kernel(&cfg_noncausal).expect("dkdv synth (non-causal)");

    for (name, ptx) in [("dq", &dq_nc), ("dkdv", &dkdv_nc)] {
        assert!(
            !ptx.contains("V2_INTRA_TILE_CAUSAL_MASK"),
            "{name} (causal=false): intra-tile mask MUST NOT be emitted (zero-overhead gate)"
        );
    }
}

// ============================================================================
// The gate
// ============================================================================

/// Per-gradient relative-error + zero-output gate. Returns the (rel, max_gpu, max_ref).
#[allow(clippy::too_many_arguments)]
fn assert_gradient(
    name: &str,
    source: FSource,
    hd: usize,
    seq: usize,
    gpu: &[f32],
    reference: &[f32],
    rel_tol: f32,
    ref_floor: f32,
) {
    assert_eq!(
        gpu.len(),
        reference.len(),
        "{name}: gpu/ref length mismatch ({} vs {})",
        gpu.len(),
        reference.len()
    );
    let max_abs = gpu
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_ref = reference.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let max_gpu = gpu.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    let rel = if max_ref > 0.0 { max_abs / max_ref } else { max_abs };
    eprintln!(
        "[full_backward FSource={:?}] hd={} seq={} {}: \
         max|gpu|={:.6e} max|ref|={:.6e} max_abs={:.6e} rel={:.4e} rel_tol={:.4e}",
        source, hd, seq, name, max_gpu, max_ref, max_abs, rel, rel_tol
    );
    // (1) reference-magnitude floor: the gate must exercise a substantial gradient.
    assert!(
        max_ref > ref_floor,
        "FSource={:?} hd={} {}: reference magnitude {:.3e} below floor {:.3e} \
         (raise GATE_DO_SCALE) — vacuous-gate guard",
        source, hd, name, max_ref, ref_floor
    );
    // (2) zero-output guard: a dropped/zeroed gradient must FAIL.
    assert!(
        max_gpu >= 0.25 * max_ref,
        "FSource={:?} hd={} seq={} {}: GPU output near-zero \
         (max|gpu|={:.3e} << max|ref|={:.3e}) — ZERO-OUTPUT GUARD \
         (dropped/zeroed gradient).",
        source, hd, seq, name, max_gpu, max_ref
    );
    // (3) relative-error gate.
    assert!(
        rel <= rel_tol,
        "FSource={:?} hd={} seq={} {}: relative error {:.4e} > rel_tol {:.4e} \
         (max_abs={:.3e}, max|ref|={:.3e})",
        source, hd, seq, name, rel, rel_tol, max_abs, max_ref
    );
}

/// Build forward + saves, run the hybrid 4-kernel backward, run the CPU
/// references, and assert all 7 gradients.
fn validate_full_backward_for_source(cfg: &FlashAttentionConfig, source: FSource, seq: usize) {
    if !cuda_available() {
        return;
    }
    let batch = 1usize;
    let heads = 1usize;
    let hd = cfg.head_dim as usize;
    let d_model = cfg.csha.as_ref().unwrap().d_model as usize;
    assert_eq!(d_model, hd, "smoke scope requires d_model == head_dim");
    assert_eq!(cfg.block_q as usize, seq, "smoke scope requires seq == block_q");
    let kv_dim = heads * hd;

    // ---- Forward + saves (drive dQ/dK/dV) ----
    let inputs = generate_forward_inputs(cfg, source, seq);
    let d_o: Vec<half::f16> = generate_d_o(cfg, seq)
        .iter()
        .map(|x| half::f16::from_f32(x.to_f32() * GATE_DO_SCALE))
        .collect();
    let fwd = compute_forward_for_test(&inputs, cfg, source, seq);

    // ---- Projection inputs (drive dWq/dWk/dWv/dx) — independent of the forward ----
    let x_raw_f16 = gen_x_raw(seq, d_model);
    let wq_f16 = gen_weight(d_model, kv_dim, 0.031, 0.0);
    let wk_f16 = gen_weight(d_model, kv_dim, 0.037, 1.3);
    let wv_f16 = gen_weight(d_model, kv_dim, 0.041, 2.7);
    let norm_weight_f16 = gen_norm_weight(d_model);

    // ---- Run the hybrid 4-kernel backward on GPU ----
    let (dq_gpu, dk_gpu, dv_gpu, dwq_gpu, dwk_gpu, dwv_gpu, dx_gpu) = run_hybrid_backward_on_gpu(
        cfg, &fwd, &d_o, &x_raw_f16, &wq_f16, &wk_f16, &wv_f16, &norm_weight_f16, batch, heads, seq,
    );

    // ---- CPU references ----
    // Attention gradients: recompute P/dS from the forward saves (independent of kernel).
    let dq_ref = cpu_naive_backward_dq(
        &fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, batch, heads, seq, cfg,
    );
    let (dv_ref, dk_ref) = cpu_naive_backward_dkdv(
        &fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, batch, heads, seq, cfg,
    );

    // Projection gradients: fed the *GPU* dQ/dK/dV (isolates the proj kernel —
    // dWq/dWk/dWv/dx parity does not compound the already-gated dQ/dK/dV error).
    let (dwq_ref, dwk_ref, dwv_ref, dx_ref) = cpu_naive_backward_proj(
        &dq_gpu, &dk_gpu, &dv_gpu,
        &x_raw_f16, &wq_f16, &wk_f16, &wv_f16, &norm_weight_f16,
        RMSNORM_EPS, batch, heads, seq, hd, d_model,
    );

    let attn_tol = rel_tol_attn(hd as u32);
    let dw_tol = rel_tol_dweight(hd as u32);
    let dx_tol = rel_tol_dx(hd as u32);

    // ---- Assert all 7 gradients, non-vacuously ----
    // dQ / dV use the dV-calibrated 1e-3 floor; dK uses the 1e-4 floor (it is
    // intrinsically ~3 orders smaller than dV — the 1/sqrt(d)*(dP-D) factor).
    assert_gradient("dQ", source, hd, seq, &dq_gpu, &dq_ref, attn_tol, 1e-3);
    assert_gradient("dK", source, hd, seq, &dk_gpu, &dk_ref, attn_tol, 1e-4);
    assert_gradient("dV", source, hd, seq, &dv_gpu, &dv_ref, attn_tol, 1e-3);
    // dW floors track the attention gradient each weight gradient is contracted
    // against (dWx = x_norm^T @ dX): dWq~dQ, dWk~dK, dWv~dV. dK (hence dWk) is
    // intrinsically ~3 orders smaller than dV (the 1/sqrt(d)*(dP-D) factor); in
    // the B1Forward small-input regime dWq is also small. Use the dK-style 1e-4
    // floor for dWq/dWk (still well above the f16 noise floor) and the
    // dV-calibrated 1e-3 floor for dWv. The zero-output guard (0.25x) remains
    // the real anti-drop check for all three.
    assert_gradient("dWq", source, hd, seq, &dwq_gpu, &dwq_ref, dw_tol, 1e-4);
    assert_gradient("dWk", source, hd, seq, &dwk_gpu, &dwk_ref, dw_tol, 1e-4);
    assert_gradient("dWv", source, hd, seq, &dwv_gpu, &dwv_ref, dw_tol, 1e-3);
    assert_gradient("dx", source, hd, seq, &dx_gpu, &dx_ref, dx_tol, 1e-3);
}

// ============================================================================
// Hybrid backward GPU launcher
// ============================================================================

/// Construct the full FFI call (`tier_b2_active = 1`) for the 4-kernel hybrid
/// backward, returning all 7 gradients read back to host:
///   (dQ, dK, dV) — f32 [batch,heads,seq,hd]
///   (dWq, dWk, dWv) — f32 [d_model, kv_dim] (read back from f16 device buffers)
///   dx — f32 [batch,heads,seq,d_model]
#[allow(clippy::too_many_arguments)]
fn run_hybrid_backward_on_gpu(
    cfg: &FlashAttentionConfig,
    fwd: &nsl_test::cpu_naive_forward::ForwardOutputs,
    d_o: &[half::f16],
    x_raw_f16: &[half::f16],
    wq_f16: &[half::f16],
    wk_f16: &[half::f16],
    wv_f16: &[half::f16],
    norm_weight_f16: &[half::f16],
    batch: usize,
    heads: usize,
    seq: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = cfg.head_dim as usize;
    let d_model = cfg.csha.as_ref().unwrap().d_model as usize;
    let kv_dim = heads * hd;

    let qkv_elems = batch * heads * seq * hd;
    let stat_elems = batch * heads * seq;
    let xm_elems = batch * heads * seq * d_model; // x_raw / dx (heads==1 => == seq*d_model)
    let w_elems = d_model * kv_dim;

    let qkv_f16_bytes = (qkv_elems * 2) as i64;
    let stat_bytes = (stat_elems * 4) as i64;
    let xm_f32_bytes = (xm_elems * 4) as i64;
    let dx_bytes = (xm_elems * 4) as i64;
    let w_f16_bytes = (w_elems * 2) as i64;

    // --- Forward saves / inputs (device) ---
    let q_dev = nsl_test_cuda_alloc(qkv_f16_bytes);
    let k_dev = nsl_test_cuda_alloc(qkv_f16_bytes);
    let v_dev = nsl_test_cuda_alloc(qkv_f16_bytes);
    let o_dev = nsl_test_cuda_alloc(qkv_f16_bytes);
    let rmax_dev = nsl_test_cuda_alloc(stat_bytes);
    let rsum_dev = nsl_test_cuda_alloc(stat_bytes);
    let do_dev = nsl_test_cuda_alloc(qkv_f16_bytes);

    // --- Projection inputs (device) ---
    let xraw_dev = nsl_test_cuda_alloc(xm_f32_bytes); // f32 (kernel reads ld.global.f32)
    let nw_dev = nsl_test_cuda_alloc((d_model * 4) as i64); // f32
    let wq_dev = nsl_test_cuda_alloc(w_f16_bytes); // f16
    let wk_dev = nsl_test_cuda_alloc(w_f16_bytes);
    let wv_dev = nsl_test_cuda_alloc(w_f16_bytes);

    // --- Gradient outputs (device) ---
    // Sprint 1 T1.2: dq/dk/dv are allocated f16 to mirror the production
    // wengert lowering (`nsl_tensor_zeros_f16_on`). The launch internally uses
    // f32 scratches for the dq/dkdv/proj kernels and converts to f16 here via
    // `csha_bwd_convert_f32_to_f16` before returning.
    let dq_dev = nsl_test_cuda_alloc(qkv_f16_bytes); // f16 (production wengert contract)
    let dk_dev = nsl_test_cuda_alloc(qkv_f16_bytes);
    let dv_dev = nsl_test_cuda_alloc(qkv_f16_bytes);
    let dwq_dev = nsl_test_cuda_alloc(w_f16_bytes); // f16 (emit_dproj writes f16)
    let dwk_dev = nsl_test_cuda_alloc(w_f16_bytes);
    let dwv_dev = nsl_test_cuda_alloc(w_f16_bytes);
    let dx_dev = nsl_test_cuda_alloc(dx_bytes); // f32

    assert!(
        q_dev != 0 && k_dev != 0 && v_dev != 0 && o_dev != 0 && rmax_dev != 0 && rsum_dev != 0
            && do_dev != 0 && xraw_dev != 0 && nw_dev != 0 && wq_dev != 0 && wk_dev != 0
            && wv_dev != 0 && dq_dev != 0 && dk_dev != 0 && dv_dev != 0 && dwq_dev != 0
            && dwk_dev != 0 && dwv_dev != 0 && dx_dev != 0,
        "hybrid-backward device alloc failed"
    );

    // Upload forward saves / inputs.
    nsl_test_cuda_h2d(q_dev, fwd.q_saved.as_ptr() as i64, qkv_f16_bytes);
    nsl_test_cuda_h2d(k_dev, fwd.k_saved.as_ptr() as i64, qkv_f16_bytes);
    nsl_test_cuda_h2d(v_dev, fwd.v_saved.as_ptr() as i64, qkv_f16_bytes);
    nsl_test_cuda_h2d(o_dev, fwd.o.as_ptr() as i64, qkv_f16_bytes);
    nsl_test_cuda_h2d(rmax_dev, fwd.row_max.as_ptr() as i64, stat_bytes);
    nsl_test_cuda_h2d(rsum_dev, fwd.row_sum.as_ptr() as i64, stat_bytes);
    nsl_test_cuda_h2d(do_dev, d_o.as_ptr() as i64, qkv_f16_bytes);

    // Upload projection inputs (x_raw / norm_weight as f32; W as f16).
    let x_raw_f32: Vec<f32> = x_raw_f16.iter().map(|h| h.to_f32()).collect();
    let nw_f32: Vec<f32> = norm_weight_f16.iter().map(|h| h.to_f32()).collect();
    nsl_test_cuda_h2d(xraw_dev, x_raw_f32.as_ptr() as i64, xm_f32_bytes);
    nsl_test_cuda_h2d(nw_dev, nw_f32.as_ptr() as i64, (d_model * 4) as i64);
    nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, w_f16_bytes);
    nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, w_f16_bytes);
    nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, w_f16_bytes);

    // PTX module: 4 concatenated entries.
    let ptx = synthesize_tier_b2_backward(cfg).expect("synthesize_tier_b2_backward");
    let ptx_bytes = ptx_to_cstr_bytes(&ptx);

    // shared_mem_bytes for kernel 4 (proj_backward) — the scalar backward layout.
    let proj_smem = shared_mem_bytes_v2_backward(cfg) as i64;

    let scale_bits = (1.0f32 / (hd as f32).sqrt()).to_bits() as i64;
    let eps_bits = RMSNORM_EPS.to_bits() as i64;

    let rc = unsafe {
        nsl_flash_attention_csha_backward(
            // q/k/v_ptr — unused by the hybrid (it reads q_proj/k_proj/v_proj saves).
            0, 0, 0,
            o_dev,            // out_ptr (forward O — D-prepass needs it)
            0,                // logsumexp_ptr (unused by hybrid proj path)
            scale_bits,
            batch as i64, heads as i64, seq as i64, hd as i64,
            0,                // block_table_ptr
            0, 0,             // k_pool_ptr, v_pool_ptr
            0,                // block_size
            0, 0,             // cos_ptr, sin_ptr (rope_q=false)
            0,                // seq_ids_ptr
            0,                // seq_lens_ptr (per-q-block base; threaded internally)
            proj_smem,        // shared_mem_bytes (proj kernel)
            ptx_bytes.as_ptr() as i64,
            0,                // name_ptr (hybrid uses fixed per-kernel names)
            cfg.block_q, cfg.block_kv,
            if cfg.causal { 1 } else { 0 },
            0,                // x_ptr (csha_x_ptr — RMSNorm-overwritten; backward reads x_raw)
            nw_dev,           // norm_weight_ptr
            wq_dev, wk_dev, wv_dev,
            0,                // wo_ptr
            eps_bits,
            cfg.csha.as_ref().unwrap().active_heads as i64,
            d_model as i64,
            // Forward-saved activations (inputs to backward).
            q_dev, k_dev, v_dev,
            rmax_dev, rsum_dev,
            xraw_dev,         // x_raw_ptr (pre-RMSNorm input)
            do_dev,           // do_ptr
            dq_dev, dk_dev, dv_dev,
            dwq_dev, dwk_dev, dwv_dev,
            dx_dev,
            0,                // dx_norm_ptr (not gated here)
            0,                // segment_ids_ptr
            0, 0,             // tier_b_ptx_ptr, tier_b_name_ptr (sentinel: both 0)
            0,                // doc_starts_ptr
            1,                // tier_b2_active = 1 — HYBRID BRANCH
        )
    };

    let free_all = || {
        for p in [
            q_dev, k_dev, v_dev, o_dev, rmax_dev, rsum_dev, do_dev, xraw_dev, nw_dev, wq_dev,
            wk_dev, wv_dev, dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev, dx_dev,
        ] {
            nsl_test_cuda_free(p);
        }
    };

    if rc != 0 {
        free_all();
        panic!("hybrid backward FFI failed rc={rc} (hd={hd} seq={seq})");
    }

    // Read back gradients.
    // Sprint 1 T1.2: dq/dk/dv are f16 on device — read back as f16, widen to f32.
    let mut dq_f16 = vec![half::f16::ZERO; qkv_elems];
    let mut dk_f16 = vec![half::f16::ZERO; qkv_elems];
    let mut dv_f16 = vec![half::f16::ZERO; qkv_elems];
    nsl_test_cuda_d2h(dq_f16.as_mut_ptr() as i64, dq_dev, qkv_f16_bytes);
    nsl_test_cuda_d2h(dk_f16.as_mut_ptr() as i64, dk_dev, qkv_f16_bytes);
    nsl_test_cuda_d2h(dv_f16.as_mut_ptr() as i64, dv_dev, qkv_f16_bytes);
    let dq_host: Vec<f32> = dq_f16.iter().map(|h| h.to_f32()).collect();
    let dk_host: Vec<f32> = dk_f16.iter().map(|h| h.to_f32()).collect();
    let dv_host: Vec<f32> = dv_f16.iter().map(|h| h.to_f32()).collect();

    // dWq/dWk/dWv are f16 on device — read back as f16, widen to f32.
    let mut dwq_f16 = vec![half::f16::ZERO; w_elems];
    let mut dwk_f16 = vec![half::f16::ZERO; w_elems];
    let mut dwv_f16 = vec![half::f16::ZERO; w_elems];
    nsl_test_cuda_d2h(dwq_f16.as_mut_ptr() as i64, dwq_dev, w_f16_bytes);
    nsl_test_cuda_d2h(dwk_f16.as_mut_ptr() as i64, dwk_dev, w_f16_bytes);
    nsl_test_cuda_d2h(dwv_f16.as_mut_ptr() as i64, dwv_dev, w_f16_bytes);
    let dwq_host: Vec<f32> = dwq_f16.iter().map(|h| h.to_f32()).collect();
    let dwk_host: Vec<f32> = dwk_f16.iter().map(|h| h.to_f32()).collect();
    let dwv_host: Vec<f32> = dwv_f16.iter().map(|h| h.to_f32()).collect();

    let mut dx_host = vec![0.0f32; xm_elems];
    nsl_test_cuda_d2h(dx_host.as_mut_ptr() as i64, dx_dev, dx_bytes);

    free_all();

    (dq_host, dk_host, dv_host, dwq_host, dwk_host, dwv_host, dx_host)
}
