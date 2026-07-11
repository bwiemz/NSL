//! CPKD Innovation 4 (compile-time half): teacher LM-head spectral analysis.
//!
//! Selects the **effective vocabulary rank** of the teacher's LM head via the
//! paper's `energy_90` criterion:
//!
//! > `energy_90(A)` = the smallest `k` such that
//! > `Σ_{i≤k} σ_i² ≥ threshold · ‖A‖_F²`  (threshold defaults to 0.90).
//!
//! The denominator `‖A‖_F² = Σ_i σ_i²` is computed **exactly** by one streaming
//! O(mn) pass over the raw matrix — no factorization is needed for the total
//! energy.  The numerator uses the deterministic randomized SVD from
//! [`crate::wrga_spectral`] with a sketch rank that doubles (16, 32, 64, …)
//! until either the energy threshold is crossed or the cap is reached.
//!
//! ## Certified-lower-bound semantics
//!
//! The randomized SVD returns the singular values of `B = QᵀA` where `Q` is a
//! column-orthonormal sketch basis.  By projection interlacing,
//! `σ_i(QᵀA) ≤ σ_i(A)` for every `i`, so the cumulative sketch energy
//! `Σ_{i≤k} σ_i(B)²` is a **certified lower bound** on the true captured energy
//! `Σ_{i≤k} σ_i(A)²`.  When the bound crosses `threshold · ‖A‖_F²` at index
//! `k`, `Rank(k)` is *guaranteed* to capture at least `threshold` of the true
//! energy.  The selector may over-estimate the minimal `k` (the bound is
//! conservative), which is the safe direction for a compression decision: we
//! never keep too few directions, only possibly a few too many.
//!
//! ## Sketch-width ceiling (cap ≤ 192)
//!
//! `wrga_spectral`'s eigensolver is a max-pivot Jacobi sweep intended for
//! "tiny k×k" sketches (its module doc says ~32-dimensional); each rotation
//! pays an O(l²) pivot scan, so sketch widths past roughly `l ≈ 200` blow up.
//! With `oversample = 8` that puts the honest ceiling for the target rank at
//! `192`.  [`effective_vocab_rank`] therefore clamps `cap` to
//! [`MAX_SKETCH_CAP`] (and to `min(m, n)`); if the threshold is still not
//! crossed at the clamped cap the result is [`EffectiveRank::ExceedsCap`] —
//! the honest refusal.  LM heads are often near-full-rank (softmax
//! bottleneck), so `ExceedsCap` is an expected, meaningful outcome that
//! downstream reporting must surface as "compression disabled", never silently
//! degrade.
//!
//! ## Rank-collapse guard (why the bound stays certified)
//!
//! `wrga_spectral`'s Gram–Schmidt QR *normalizes* residual columns whose norm
//! is above `1e-30`.  When the sketch width `l` exceeds the numerical rank of
//! the matrix, the dependent columns leave ~ε-level rounding residuals that
//! get normalized into unit-norm junk vectors with O(1) overlap onto earlier
//! columns — `Q` is then **not** orthonormal and `σ_i(QᵀA) ≤ σ_i(A)` fails
//! (observed: 3× total-energy inflation on an exactly-rank-6 fixture).  Two
//! facts rescue certification without touching `wrga_spectral`:
//!
//! 1. For a *valid* certificate, `Σ_i σ_i(QᵀA)² = ‖QᵀA‖_F² ≤ ‖A‖_F²`.  Since
//!    we know `‖A‖_F²` exactly, any sketch whose total energy exceeds it has
//!    provably lost orthonormality and is discarded (never used for a bound).
//! 2. `B = QᵀA` has `rank(B) ≤ rank(A)` *regardless* of `Q`'s quality (its
//!    rows are combinations of `A`'s rows), so the count of non-negligible
//!    sketch values reads off the numerical rank `r < l`.  Re-sketching at
//!    exactly width `r` (oversample 0) keeps every QR residual well
//!    conditioned, restoring an orthonormal `Q` and a certified bound.
//!
//! The guard loop strictly shrinks the sketch width on each collapse, so it
//! terminates; if it cannot certify anything it refuses with `ExceedsCap`
//! rather than reporting an uncertified rank.
//!
//! ## Determinism
//!
//! All sketches use the fixed [`CPKD_SPECTRAL_SEED`]; LM-head name resolution
//! walks fixed candidate arrays (never HashMap iteration order).  Two
//! invocations over the same weights produce bit-identical results.
//!
//! ## Advisory in v1
//!
//! The selected rank feeds the CPKD distillation build report only; the fused
//! KL-CE kernel uses the full vocabulary in v1 (top-k logit compression is a
//! documented deferral).

use serde::Serialize;

use crate::weight_aware::{WeightDType, WeightEntry, WeightMap};
use crate::wrga_spectral::randomized_svd;

/// Fixed seed for the randomized SVD so two compiler invocations over the same
/// teacher checkpoint select the same effective vocabulary rank
/// (pattern: `cfie_kv_quant`'s `SPECTRAL_SEED = 0xCF1E_5EED`).
pub const CPKD_SPECTRAL_SEED: u64 = 0xC9CD_5EED;

/// Hard ceiling on the sketch target rank.  `wrga_spectral`'s Jacobi
/// eigensolver is only adequate for small sketches (O(l²) pivot scan per
/// rotation); with `oversample = 8` the sketch width at this cap is 200, the
/// documented practical limit.  [`effective_vocab_rank`] clamps its `cap`
/// argument to this value.
pub const MAX_SKETCH_CAP: usize = 192;

/// HMT oversampling used for every sketch (standard recommendation is 5–10).
const OVERSAMPLE: usize = 8;

/// First sketch rank in the doubling schedule 16, 32, 64, …
const INITIAL_SKETCH_RANK: usize = 16;

/// Relative slack on the energy-consistency guard: a certified sketch obeys
/// `Σ σ_i(QᵀA)² ≤ ‖A‖_F²`, so anything above `total · (1 + rtol)` proves the
/// sketch basis lost orthonormality (rank-collapse regime, see module docs).
const ENERGY_CONSISTENCY_RTOL: f64 = 1e-9;

/// Relative σ threshold used to count the numerical rank of a collapsed
/// sketch: values at or below `σ₁ · rtol` are junk-column noise, not real
/// spectral directions.
const RANK_COLLAPSE_RTOL: f64 = 1e-7;

/// LM-head parameter names tried first, in order.
const LM_HEAD_CANDIDATES: [&str; 4] =
    ["lm_head.weight", "output.weight", "lm_head", "head.weight"];

/// Tied-embedding fallbacks: checkpoints that tie the LM head to the token
/// embedding omit `lm_head.weight` entirely (see `bitnet/loader.rs` "lm_head
/// if untied"), so the embedding matrix *is* the head.
const TIED_EMBED_CANDIDATES: [&str; 5] = [
    "model.embed_tokens.weight",
    "embed_tokens.weight",
    "tok_embeddings.weight",
    "embedding.weight",
    "embed.weight",
];

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Outcome of the energy-threshold rank search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum EffectiveRank {
    /// Smallest `k` whose certified lower bound on captured energy crosses
    /// `threshold · ‖A‖_F²`.
    Rank(usize),
    /// The bound never crossed the threshold up to the (clamped) sketch cap.
    /// `cap` is the *effective* cap that was searched — the caller's request
    /// after clamping to [`MAX_SKETCH_CAP`] and `min(m, n)`.
    ExceedsCap { cap: usize },
}

/// Full record of one effective-vocabulary-rank selection, suitable for the
/// CPKD distillation build report.
#[derive(Debug, Clone, Serialize)]
pub struct SpectralRankResult {
    /// The selected rank, or the honest refusal.
    pub rank: EffectiveRank,
    /// Energy threshold the search targeted (e.g. 0.90).
    pub threshold: f64,
    /// Exact `‖A‖_F²` from the streaming pass (equals `Σ_i σ_i(A)²`).
    pub total_energy: f64,
    /// Certified lower bound on captured energy: at the crossing index for
    /// `Rank(k)`, or the best bound found when `ExceedsCap`.
    pub captured_energy: f64,
    /// Sketch singular values (descending) from the run that produced `rank`.
    pub singular_values: Vec<f64>,
}

// ---------------------------------------------------------------------------
// energy_90 rank selector
// ---------------------------------------------------------------------------

/// Select the effective vocabulary rank of the row-major `m × n` matrix `mat`.
///
/// Returns the smallest `k` (as [`EffectiveRank::Rank`]) whose **certified
/// lower bound** on captured spectral energy reaches
/// `threshold · ‖A‖_F²` — see the module docs for why `σ_i(QᵀA) ≤ σ_i(A)`
/// makes this guaranteed-sufficient (it may over-estimate the minimal `k`,
/// which is the safe direction).  The total energy is computed exactly by one
/// streaming pass; sketches double from 16 up to `cap` (clamped to
/// [`MAX_SKETCH_CAP`] and `min(m, n)`), `oversample = 8`, fixed seed
/// [`CPKD_SPECTRAL_SEED`].  If the clamped cap is exhausted without crossing,
/// the result is [`EffectiveRank::ExceedsCap`] with the best bound found.
///
/// Sketches whose total energy exceeds the exact `‖A‖_F²` have provably lost
/// basis orthonormality (rank-collapse regime — see the module docs) and are
/// never used for a bound; the search re-sketches at the matrix's numerical
/// rank instead, so every returned `Rank(k)` is certified.
///
/// A matrix with zero total energy (all zeros) yields `Rank(0)` without
/// running any sketch.
///
/// # Panics
///
/// Panics if `mat.len() < m * n` (caller bug — the public loader in
/// [`teacher_lm_head_rank`] validates byte lengths before decoding).
pub fn effective_vocab_rank(
    mat: &[f64],
    m: usize,
    n: usize,
    threshold: f64,
    cap: usize,
) -> SpectralRankResult {
    assert!(
        mat.len() >= m * n,
        "cpkd_spectral::effective_vocab_rank: mat has {} elements, need m*n = {}",
        mat.len(),
        m * n
    );
    let mat = &mat[..m * n];

    // Exact ‖A‖_F² = Σ σ_i² by streaming — one O(mn) pass, no factorization.
    let total_energy: f64 = mat.iter().map(|v| v * v).sum();

    if m == 0 || n == 0 || total_energy <= 0.0 {
        // Zero matrix: every direction carries zero energy; nothing to keep.
        return SpectralRankResult {
            rank: EffectiveRank::Rank(0),
            threshold,
            total_energy,
            captured_energy: 0.0,
            singular_values: Vec::new(),
        };
    }

    // Clamp the search cap: Jacobi sketch ceiling + rank(A) ≤ min(m, n).
    let effective_cap = cap.min(MAX_SKETCH_CAP).min(m.min(n)).max(1);
    let target = threshold * total_energy;

    let mut k = INITIAL_SKETCH_RANK.min(effective_cap);
    let mut oversample = OVERSAMPLE;
    let mut best_captured = 0.0_f64;
    let mut best_sv: Vec<f64> = Vec::new();
    loop {
        let sv = randomized_svd(mat, m, n, k, oversample, CPKD_SPECTRAL_SEED);
        let sketch_energy: f64 = sv.iter().map(|s| s * s).sum();

        // Energy-consistency guard: a certified sketch (orthonormal Q) obeys
        // ‖QᵀA‖_F² ≤ ‖A‖_F².  Violation ⇒ rank-collapse regime: the sketch
        // width exceeded the numerical rank and QR normalized junk columns
        // (module docs).  B = QᵀA still has rank ≤ rank(A), so the count of
        // non-negligible values reads off the numerical rank — re-sketch at
        // exactly that width (no oversampling) for a clean, certified basis.
        if sketch_energy > total_energy * (1.0 + ENERGY_CONSISTENCY_RTOL) {
            let sigma1 = sv.first().copied().unwrap_or(0.0);
            let l_used = (k + oversample).min(n).max(1);
            let r = sv
                .iter()
                .filter(|s| **s > sigma1 * RANK_COLLAPSE_RTOL)
                .count()
                .max(1);
            if r >= l_used {
                // The junk merged into significant values — the sketch width
                // cannot shrink, so nothing is certifiable.  Honest refusal
                // with the best certified bound found so far (possibly none).
                return SpectralRankResult {
                    rank: EffectiveRank::ExceedsCap { cap: effective_cap },
                    threshold,
                    total_energy,
                    captured_energy: best_captured,
                    singular_values: best_sv,
                };
            }
            k = r.min(effective_cap);
            oversample = 0;
            continue;
        }

        let mut cum = 0.0_f64;
        let mut crossing: Option<(usize, f64)> = None;
        for (i, s) in sv.iter().enumerate() {
            cum += s * s;
            if cum >= target {
                crossing = Some((i + 1, cum));
                break;
            }
        }
        if let Some((rank, captured)) = crossing {
            return SpectralRankResult {
                rank: EffectiveRank::Rank(rank),
                threshold,
                total_energy,
                captured_energy: captured,
                singular_values: sv,
            };
        }
        if cum > best_captured || best_sv.is_empty() {
            best_captured = cum;
            best_sv = sv;
        }
        // `oversample == 0` marks a collapse retry: the certified sketch at
        // the matrix's full numerical rank still did not cross, and any wider
        // sketch would only re-collapse — refuse rather than loop.
        if oversample == 0 || k >= effective_cap {
            return SpectralRankResult {
                rank: EffectiveRank::ExceedsCap { cap: effective_cap },
                threshold,
                total_energy,
                captured_energy: best_captured,
                singular_values: best_sv,
            };
        }
        k = (k * 2).min(effective_cap);
    }
}

// ---------------------------------------------------------------------------
// Teacher LM-head lookup + decode
// ---------------------------------------------------------------------------

/// Resolve the teacher's LM head in `wm` and run [`effective_vocab_rank`] on
/// it.  Returns `(matched_weight_name, result)`.
///
/// Name resolution walks [`LM_HEAD_CANDIDATES`] first, then the
/// tied-embedding fallbacks [`TIED_EMBED_CANDIDATES`] (tied checkpoints omit
/// `lm_head.weight` — the embedding matrix is the head).  Fixed arrays, so
/// resolution never depends on HashMap iteration order.
///
/// Refusals (`Err`, all loud):
/// * no candidate name present in the map;
/// * FP8 entries — `WeightDType::to_f64` decodes FP8 as an acknowledged
///   approximation (`bytes[0] / 127.0`), which yields garbage spectra;
/// * entries with fewer than 2 dims;
/// * raw data shorter than the shape requires.
///
/// Tensors with ndim > 2 are viewed as `[dim0, prod(rest)]` — the same
/// flattening convention as `wrga_spectral::analyse_weight_map`.
pub fn teacher_lm_head_rank(
    wm: &WeightMap,
    threshold: f64,
    cap: usize,
) -> Result<(String, SpectralRankResult), String> {
    let entry = find_lm_head(wm).ok_or_else(|| {
        format!(
            "CPKD spectral: no LM-head weight found in '{}'. Tried {:?}, then \
             tied-embedding fallbacks {:?}. Effective-vocab-rank selection \
             requires the teacher checkpoint to expose its output projection.",
            wm.source_path(),
            LM_HEAD_CANDIDATES,
            TIED_EMBED_CANDIDATES
        )
    })?;

    if matches!(entry.dtype, WeightDType::F8E4M3 | WeightDType::F8E5M2) {
        return Err(format!(
            "CPKD spectral: refusing FP8 LM head '{}' ({:?}): \
             WeightDType::to_f64 decodes FP8 as an acknowledged approximation \
             (bytes[0]/127.0), which produces garbage spectra. Provide an \
             F16/BF16/F32/F64 teacher checkpoint for rank selection.",
            entry.name, entry.dtype
        ));
    }
    if entry.shape.len() < 2 {
        return Err(format!(
            "CPKD spectral: LM-head candidate '{}' has {} dim(s) (shape {:?}); \
             spectral rank selection needs a matrix with >= 2 dims.",
            entry.name,
            entry.shape.len(),
            entry.shape
        ));
    }

    let m = entry.shape[0];
    let n: usize = entry.shape[1..].iter().product::<usize>().max(1);
    let bw = entry.dtype.byte_width();
    if entry.data.len() < m * n * bw {
        return Err(format!(
            "CPKD spectral: LM-head '{}' has {} raw bytes but shape {:?} at \
             {:?} needs {} — refusing truncated weight data.",
            entry.name,
            entry.data.len(),
            entry.shape,
            entry.dtype,
            m * n * bw
        ));
    }

    // Decode elementwise to f64 (loader pattern: cfie_kv_quant::sigma_max).
    let mut mat = Vec::with_capacity(m * n);
    for idx in 0..(m * n) {
        let off = idx * bw;
        mat.push(entry.dtype.to_f64(&entry.data[off..off + bw]));
    }

    let result = effective_vocab_rank(&mat, m, n, threshold, cap);
    Ok((entry.name.clone(), result))
}

/// First present candidate, primaries before tied-embedding fallbacks.
fn find_lm_head(wm: &WeightMap) -> Option<&WeightEntry> {
    LM_HEAD_CANDIDATES
        .iter()
        .chain(TIED_EMBED_CANDIDATES.iter())
        .find_map(|name| wm.get(name))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Exactly-rank-6 fixture: A = Σ scale_i · u_i v_iᵀ where u_i / v_i are
    /// disjoint-support (hence exactly orthogonal) block indicator vectors.
    /// True σ_i = scale_i · √(8·16); energies 128·scale_i².  Cumulative
    /// energy fractions: 0.4525, 0.7421, 0.9050, 0.9774, 0.9955, 1.0 — the
    /// exact minimal rank for threshold 0.90 is 3.
    fn low_rank_fixture() -> (Vec<f64>, usize, usize) {
        let (m, n) = (48usize, 96usize);
        let scales = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0];
        let mut a = vec![0.0; m * n];
        for (i, &scale) in scales.iter().enumerate() {
            for row in (8 * i)..(8 * i + 8) {
                for col in (16 * i)..(16 * i + 16) {
                    a[row * n + col] = scale;
                }
            }
        }
        (a, m, n)
    }

    fn identity(n: usize) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        a
    }

    fn f32_entry(name: &str, mat: &[f64], shape: Vec<usize>) -> WeightEntry {
        let mut data = Vec::with_capacity(mat.len() * 4);
        for &v in mat {
            data.extend_from_slice(&(v as f32).to_le_bytes());
        }
        WeightEntry::new(name.to_string(), data, shape, WeightDType::F32)
    }

    fn map_of(entries: Vec<WeightEntry>) -> WeightMap {
        let mut h = HashMap::new();
        for e in entries {
            h.insert(e.name.clone(), e);
        }
        WeightMap::from_entries(h)
    }

    // (a) exactly-low-rank matrix crosses the threshold at small k.
    #[test]
    fn low_rank_matrix_selects_small_rank() {
        let (a, m, n) = low_rank_fixture();
        let res = effective_vocab_rank(&a, m, n, 0.90, 64);
        match res.rank {
            EffectiveRank::Rank(k) => {
                assert!(k <= 8, "expected k <= 8 for exactly-rank-6 fixture, got {k}");
                assert!(k >= 3, "cannot cross 0.90 before the true minimum 3, got {k}");
            }
            EffectiveRank::ExceedsCap { cap } => {
                panic!("rank-6 fixture must not exceed cap {cap}")
            }
        }
        assert!(res.total_energy > 0.0);
        assert!(
            res.captured_energy / res.total_energy >= 0.90,
            "captured/total = {} < threshold",
            res.captured_energy / res.total_energy
        );
        // Exact streaming total: 128 · (100+64+36+16+4+1) = 28288.
        assert!((res.total_energy - 28288.0).abs() < 1e-6);
        assert_eq!(res.threshold, 0.90);
    }

    // (b) full-rank flat spectrum with a small cap refuses honestly.
    #[test]
    fn full_rank_identity_exceeds_cap() {
        let a = identity(64);
        let res = effective_vocab_rank(&a, 64, 64, 0.90, 16);
        assert_eq!(res.rank, EffectiveRank::ExceedsCap { cap: 16 });
        // Total energy of I_64 is exactly 64; the top-16 bound is ~16.
        assert!((res.total_energy - 64.0).abs() < 1e-9);
        assert!(res.captured_energy > 0.0);
        assert!(res.captured_energy < 0.90 * res.total_energy);
        assert!(!res.singular_values.is_empty());
    }

    // (c) determinism: same input -> bit-identical singular values.
    #[test]
    fn deterministic_across_invocations() {
        let (a, m, n) = low_rank_fixture();
        let r1 = effective_vocab_rank(&a, m, n, 0.90, 64);
        let r2 = effective_vocab_rank(&a, m, n, 0.90, 64);
        assert_eq!(r1.rank, r2.rank);
        assert_eq!(r1.singular_values.len(), r2.singular_values.len());
        for (x, y) in r1.singular_values.iter().zip(r2.singular_values.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "singular values not bit-identical");
        }
        assert_eq!(r1.captured_energy.to_bits(), r2.captured_energy.to_bits());
        assert_eq!(r1.total_energy.to_bits(), r2.total_energy.to_bits());
    }

    // (d) higher threshold never selects a smaller rank.
    #[test]
    fn threshold_monotonicity() {
        let (a, m, n) = low_rank_fixture();
        let lo = effective_vocab_rank(&a, m, n, 0.50, 64);
        let hi = effective_vocab_rank(&a, m, n, 0.95, 64);
        match (&lo.rank, &hi.rank) {
            (EffectiveRank::Rank(kl), EffectiveRank::Rank(kh)) => {
                assert!(kl <= kh, "rank(0.5) = {kl} must be <= rank(0.95) = {kh}");
            }
            other => panic!("both thresholds must resolve on the rank-6 fixture: {other:?}"),
        }
    }

    // (e) WeightMap lookup + decode via the primary lm_head name.
    #[test]
    fn teacher_lm_head_rank_finds_primary_name() {
        let (a, m, n) = low_rank_fixture();
        let wm = map_of(vec![
            f32_entry("lm_head.weight", &a, vec![m, n]),
            f32_entry("model.embed_tokens.weight", &identity(32), vec![32, 32]),
        ]);
        let (name, res) = teacher_lm_head_rank(&wm, 0.90, 64).expect("lookup must succeed");
        assert_eq!(name, "lm_head.weight");
        match res.rank {
            EffectiveRank::Rank(k) => assert!((3..=8).contains(&k), "got {k}"),
            other => panic!("expected Rank on rank-6 fixture, got {other:?}"),
        }
        assert!(res.captured_energy >= 0.90 * res.total_energy);
    }

    // (f) FP8 teacher heads are refused loudly.
    #[test]
    fn fp8_lm_head_is_refused() {
        let entry = WeightEntry::new(
            "lm_head.weight".to_string(),
            vec![0x40u8; 16],
            vec![4, 4],
            WeightDType::F8E4M3,
        );
        let wm = map_of(vec![entry]);
        let err = teacher_lm_head_rank(&wm, 0.90, 64).unwrap_err();
        assert!(err.contains("FP8"), "error must name FP8: {err}");
        assert!(err.contains("lm_head.weight"), "error must name the entry: {err}");
    }

    // (g) tied-embedding checkpoints resolve via the fallback list.
    #[test]
    fn tied_embedding_fallback_resolves() {
        let (a, m, n) = low_rank_fixture();
        let wm = map_of(vec![f32_entry("model.embed_tokens.weight", &a, vec![m, n])]);
        let (name, res) = teacher_lm_head_rank(&wm, 0.90, 64).expect("fallback must resolve");
        assert_eq!(name, "model.embed_tokens.weight");
        assert!(matches!(res.rank, EffectiveRank::Rank(_)));
    }

    #[test]
    fn missing_lm_head_errs_with_candidates() {
        let wm = map_of(vec![f32_entry("blocks.0.attn.wq", &identity(8), vec![8, 8])]);
        let err = teacher_lm_head_rank(&wm, 0.90, 64).unwrap_err();
        assert!(err.contains("lm_head.weight"), "error must list candidates: {err}");
        assert!(err.contains("embed_tokens"), "error must list fallbacks: {err}");
    }

    #[test]
    fn one_dimensional_lm_head_is_refused() {
        let entry = f32_entry("lm_head.weight", &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let wm = map_of(vec![entry]);
        let err = teacher_lm_head_rank(&wm, 0.90, 64).unwrap_err();
        assert!(err.contains(">= 2 dims"), "error must explain the dim refusal: {err}");
    }

    #[test]
    fn truncated_data_is_refused() {
        let entry = WeightEntry::new(
            "lm_head.weight".to_string(),
            vec![0u8; 8], // 2 f32s, but shape claims 16 elements
            vec![4, 4],
            WeightDType::F32,
        );
        let wm = map_of(vec![entry]);
        let err = teacher_lm_head_rank(&wm, 0.90, 64).unwrap_err();
        assert!(err.contains("raw bytes"), "error must name the truncation: {err}");
    }

    #[test]
    fn ndim3_is_flattened_like_wrga_spectral() {
        // Same data as the rank-6 fixture but shaped [48, 8, 12] — must be
        // viewed as [48, 96] and select the same small rank.
        let (a, m, n) = low_rank_fixture();
        let wm = map_of(vec![f32_entry("lm_head.weight", &a, vec![m, 8, 12])]);
        let (_, res3) = teacher_lm_head_rank(&wm, 0.90, 64).expect("3-D entry must flatten");
        let wm2 = map_of(vec![f32_entry("lm_head.weight", &a, vec![m, n])]);
        let (_, res2) = teacher_lm_head_rank(&wm2, 0.90, 64).expect("2-D entry");
        assert_eq!(res3.rank, res2.rank);
    }

    #[test]
    fn zero_matrix_selects_rank_zero() {
        let a = vec![0.0; 24 * 24];
        let res = effective_vocab_rank(&a, 24, 24, 0.90, 64);
        assert_eq!(res.rank, EffectiveRank::Rank(0));
        assert_eq!(res.total_energy, 0.0);
        assert_eq!(res.captured_energy, 0.0);
        assert!(res.singular_values.is_empty());
    }

    #[test]
    fn cap_is_clamped_to_matrix_and_jacobi_ceiling() {
        // cap 10_000 on a 32×32 identity: effective cap = min(192, 32) = 32;
        // the doubling schedule reaches the full sketch and must cross 0.90.
        let a = identity(32);
        let res = effective_vocab_rank(&a, 32, 32, 0.90, 10_000);
        match res.rank {
            EffectiveRank::Rank(k) => {
                // Exact crossing is ceil(0.9·32) = 29; the sketch bound may
                // land slightly later but can never exceed the matrix rank.
                assert!((29..=32).contains(&k), "got {k}");
            }
            EffectiveRank::ExceedsCap { cap } => {
                panic!("full-width sketch of I_32 must cross 0.90 (cap {cap})")
            }
        }
    }

    #[test]
    fn exceeds_cap_reports_effective_cap_when_clamped() {
        // cap 10_000 with a flat spectrum wider than MAX_SKETCH_CAP would be
        // needed — but min(m, n) = 64 clamps first, and a full-width sketch
        // crosses. Use cap 16 < min(m, n) to observe the reported clamp.
        let a = identity(64);
        let res = effective_vocab_rank(&a, 64, 64, 0.99, 16);
        assert_eq!(res.rank, EffectiveRank::ExceedsCap { cap: 16 });
    }

    #[test]
    #[should_panic(expected = "mat has")]
    fn short_slice_panics() {
        let a = vec![1.0; 10];
        let _ = effective_vocab_rank(&a, 4, 4, 0.90, 8);
    }

    // Regression for the rank-collapse guard: sketches wider than rank(A)
    // inflate σ (junk QR columns) — the guard must re-sketch at the numerical
    // rank so the reported values stay certified lower bounds.
    #[test]
    fn collapse_guard_keeps_bound_certified() {
        let (a, m, n) = low_rank_fixture();
        let res = effective_vocab_rank(&a, m, n, 0.90, 64);
        // Certified: reported sketch energy can never exceed the exact total.
        let sketch_energy: f64 = res.singular_values.iter().map(|s| s * s).sum();
        assert!(
            sketch_energy <= res.total_energy * (1.0 + 1e-9),
            "sketch energy {sketch_energy} exceeds exact total {} — bound not certified",
            res.total_energy
        );
        // Certified: σ₁(sketch) ≤ true σ₁ = 10·√128 ≈ 113.137 (fp slack).
        let sigma1_true = 10.0 * 128.0_f64.sqrt();
        assert!(
            res.singular_values[0] <= sigma1_true * (1.0 + 1e-6),
            "σ₁ sketch {} overshoots true σ₁ {sigma1_true}",
            res.singular_values[0]
        );
        // With the clean re-sketch, the crossing is the true minimum: the
        // cumulative energy fractions are .4525/.7421/.9050/… → k = 3.
        assert_eq!(res.rank, EffectiveRank::Rank(3));
    }

    #[test]
    fn result_serializes_to_json() {
        let (a, m, n) = low_rank_fixture();
        let res = effective_vocab_rank(&a, m, n, 0.90, 64);
        let json = serde_json::to_string(&res).expect("Serialize derive must work");
        assert!(json.contains("\"rank\""));
        assert!(json.contains("\"singular_values\""));
        let cap = effective_vocab_rank(&identity(64), 64, 64, 0.90, 16);
        let json_cap = serde_json::to_string(&cap).expect("ExceedsCap must serialize");
        assert!(json_cap.contains("ExceedsCap"));
    }
}
