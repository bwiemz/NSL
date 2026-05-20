//! Pure-Rust FP32 reference implementation of the BitNet b1.58 forward pass.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §5.
//! Plan: `docs/superpowers/plans/2026-04-18-pca-tier-a-implementation.md` (Task 3).
//!
//! Included via `#[path]` from integration tests in this directory. Lives
//! under `tests/` rather than `src/bitnet/reference.rs` because `tests/`
//! integration tests are separate compilation units that cannot see
//! `#[cfg(test)]`-gated items from the lib crate (AWQ precedent — see
//! awq_full_pipeline.rs).
//!
//! ## Spec correction applied inline (documented for follow-up amendment)
//!
//! Spec §4.2 specified `scale = mean(|x[r,:]|)` (absmean) for activation
//! quantization. **This is wrong.** PI.2 verified against the reference
//! `utils_quant.py` in `1bitLLM/bitnet_b1_58-3B` (HF commit
//! `af89e318d78a70802061246bf037199d2fb97020`):
//!
//! - Activations (per-row, per-token): `scale = max(|x|)` (absmax).
//! - Weights (per-tensor): `scale = mean(|W|)` (absmean).
//!
//! The b1.58 paper conflates the two terms in surface-level prose; the
//! implementation in the released checkpoints is unambiguous. This file
//! encodes the absmax path for activations. Spec §4.2 will be amended
//! in a separate PR; the file-naming "absmean_quant.rs" downstream
//! (Task 5) is retained for spec traceability and will name the
//! function `absmax_quant` internally.
//!
//! ## Anchor strength
//!
//! Fixtures are NSL-reference self-anchored (spec §5.1 fallback path):
//! bitnet.cpp is not buildable on the Windows MSVC implementation
//! platform. The reference encodes the b1.58 paper's math directly; the
//! committed fixtures pin the reference's outputs so future kernel work
//! (Tasks 4-7) validates against the JSON, NOT this file. A future
//! Linux/macOS re-anchor against bitnet.cpp can re-write the fixture
//! values without touching the reference.

#![allow(dead_code)]

/// Per-row absmax scale. Returns 0.0 for an all-zero row (caller treats
/// quantization as a no-op and emits a zero output row).
///
/// Spec §4.2 CORRECTED: BitNet b1.58 uses per-row absmax (NOT absmean)
/// for activations.
pub fn absmax_scale_row(row: &[f32]) -> f32 {
    debug_assert!(
        row.iter().all(|x| x.is_finite()),
        "absmax_scale_row: row contains NaN/Inf"
    );
    row.iter()
        .map(|x| x.abs())
        .fold(0.0_f32, |acc, v| acc.max(v))
}

/// Quantize one row to int8 using its absmax scale.
///
/// Spec §4.2 CORRECTED: `q[r,k] = round(clip(x * 127 / scale, -127, 127))`.
/// Matches `1bitLLM/bitnet_b1_58-3B/utils_quant.py` reference.
pub fn quantize_row_int8(row: &[f32], scale: f32) -> Vec<i8> {
    if scale == 0.0 {
        return vec![0; row.len()];
    }
    let inv_scale = 127.0_f32 / scale;
    row.iter()
        .map(|x| {
            let scaled = x * inv_scale;
            let clipped = scaled.clamp(-127.0, 127.0);
            clipped.round() as i8
        })
        .collect()
}

/// Compute one element of the ternary GEMM output (FP32 accumulator).
///
/// `Y[r, c] = (scale[r] / 127.0) * Σ_k (X_q[r, k] * W_ternary[k, c])`
///
/// The `/127.0` dequantizes from the int8 range back to FP32 magnitude.
pub fn ternary_gemm_element(quantized_acts_row: &[i8], weights_col: &[i8], scale: f32) -> f32 {
    assert_eq!(
        quantized_acts_row.len(),
        weights_col.len(),
        "hidden dim mismatch: {} vs {}",
        quantized_acts_row.len(),
        weights_col.len()
    );
    let acc: i32 = quantized_acts_row
        .iter()
        .zip(weights_col.iter())
        .map(|(&a, &w)| (a as i32) * (w as i32))
        .sum();
    (scale / 127.0) * (acc as f32)
}

/// Full forward pass: returns `(scales, quantized_acts, gemm_output)`.
///
/// - `activations`: `[rows, hidden_dim]` row-major.
/// - `weights`: `[hidden_dim, out_dim]` row-major; ternary values in `{-1, 0, +1}`.
pub fn forward_reference(
    activations: &[Vec<f32>],
    weights: &[Vec<i8>],
) -> (Vec<f32>, Vec<Vec<i8>>, Vec<Vec<f32>>) {
    assert!(
        !activations.is_empty(),
        "activations must have at least one row"
    );
    assert!(!weights.is_empty(), "weights must be non-empty");
    let hidden_dim = activations[0].len();
    let out_dim = weights[0].len();
    assert_eq!(
        weights.len(),
        hidden_dim,
        "weights must be [hidden_dim, out_dim] = [{}, _], got [{}, _]",
        hidden_dim,
        weights.len()
    );

    let mut scales = Vec::with_capacity(activations.len());
    let mut q_acts = Vec::with_capacity(activations.len());
    let mut output = Vec::with_capacity(activations.len());

    for row in activations {
        assert_eq!(
            row.len(),
            hidden_dim,
            "all activation rows must share hidden_dim={}",
            hidden_dim
        );
        let scale = absmax_scale_row(row);
        let q = quantize_row_int8(row, scale);
        let mut row_output = Vec::with_capacity(out_dim);
        for c in 0..out_dim {
            let weights_col: Vec<i8> = (0..hidden_dim).map(|k| weights[k][c]).collect();
            row_output.push(ternary_gemm_element(&q, &weights_col, scale));
        }
        scales.push(scale);
        q_acts.push(q);
        output.push(row_output);
    }

    (scales, q_acts, output)
}

/// Simple deterministic LCG for cross-platform reproducible fixture inputs.
/// Avoids depending on `rand` crate behavior, which can vary across versions.
pub fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Generate the f10_large_hidden fixture inputs deterministically.
///
/// Uses a fixed LCG (seed=42) for cross-platform reproducibility, combined
/// with **Box-Muller basic form**: each output normal consumes two LCG draws,
/// and only the cosine branch is used (the sine companion `z1` is discarded).
/// This is mathematically correct but consumes 2× the LCG output of an
/// optimal implementation that emits both branches per pair. Documented
/// here so cross-implementation verification expects the same LCG-consumption
/// pattern.
///
/// Returns (activations: 128 floats, weights: 128×4 ternary matrix).
pub fn make_f10_inputs() -> (Vec<f32>, Vec<Vec<i8>>) {
    let mut state: u64 = 42;
    let acts: Vec<f32> = (0..128)
        .map(|_| {
            // Box-Muller-lite: combine two uniforms into a unit-variance normal.
            let u1 = (lcg_next(&mut state) >> 32) as f32 / (u32::MAX as f32);
            let u2 = (lcg_next(&mut state) >> 32) as f32 / (u32::MAX as f32);
            ((-2.0 * u1.max(1e-9).ln()).sqrt()) * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    let weights: Vec<Vec<i8>> = (0..128)
        .map(|_| {
            (0..4)
                .map(|_| (((lcg_next(&mut state) >> 32) % 3) as i8) - 1) // -1, 0, +1
                .collect()
        })
        .collect();
    (acts, weights)
}
