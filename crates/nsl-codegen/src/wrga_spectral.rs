//! WRGA Innovation 3: Weight-Spectral Compile-Time Rank Allocation.
//!
//! Implements a **randomized SVD** (Halko–Martinsson–Tropp, 2011) for each 2-D
//! weight matrix pulled from a safetensors `WeightMap`, computes the spectral
//! entropy / effective rank, then allocates per-layer LoRA ranks under a total
//! parameter budget.
//!
//! Randomized SVD is the critical engineering decision flagged in Gemini's
//! comments: computing an exact truncated SVD for every weight matrix of a
//! multi-billion parameter model would make `nsl build` unusably slow.  The
//! randomized variant costs O(mnk) for a rank-`k` approximation instead of
//! O(mn · min(m,n)), which is between 10× and 100× faster in practice.
//!
//! The implementation is deterministic (fixed xorshift seed) so two invocations
//! of the compiler produce identical adapter configurations for the same
//! checkpoint.

use std::collections::HashMap;

use crate::weight_aware::{WeightDType, WeightMap};

// ---------------------------------------------------------------------------
// Deterministic xorshift RNG for reproducible spectral analysis
// ---------------------------------------------------------------------------

struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Avoid the degenerate 0 state.
        Xorshift64(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in (-1, 1).
    fn next_signed(&mut self) -> f64 {
        // Use top 53 bits of the u64 for a f64 mantissa.
        let u = (self.next_u64() >> 11) as f64;
        let r = u / (1u64 << 53) as f64; // [0, 1)
        2.0 * r - 1.0
    }

    /// Box-Muller standard normal sample.
    fn next_normal(&mut self) -> f64 {
        // Use two uniforms (0,1] to avoid log(0).
        let u1 = {
            let mut v = self.next_signed().abs();
            if v < 1e-300 {
                v = 1e-300;
            }
            v
        };
        let u2 = self.next_signed().abs();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Small dense linear-algebra primitives
// ---------------------------------------------------------------------------
//
// We roll our own rather than pulling in `ndarray`/`nalgebra` because the
// rest of the codegen crate avoids those dependencies.  These routines are
// adequate for the spectral-analysis sizes we care about (matrices up to
// a few thousand × a few thousand, reduced to a ~32-dimensional sketch).
//
// All matrices are stored row-major as flat `Vec<f64>`.

fn mat_mul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for l in 0..k {
            let aik = a[i * k + l];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[l * n + j];
            }
        }
    }
    out
}

/// Return `Aᵀ` (n × m) given row-major `A` (m × n).
fn transpose(a: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    out
}

/// Thin QR decomposition via modified Gram–Schmidt.  Returns `Q` (m × k)
/// in place of the input: its columns are orthonormal, except that a column
/// linearly dependent on the ones before it becomes zero.
///
/// "Dependent" is judged against the column's own length before projection:
/// what is left after removing its components along the earlier columns is
/// rounding error when it is some 10⁻¹⁰ of the original. Such a column used
/// to be normalised whenever its residual exceeded 10⁻³⁰ in absolute terms,
/// which every rounding residual does, and a normalised rounding residual
/// points in no particular direction: on a rank-5 matrix sketched with ten
/// columns, five columns of `Q` were noise, `QᵀQ` was off the identity by
/// 0.70, and the largest singular value came out 1.333 instead of 1.
///
/// Each column is projected twice; one pass of Gram–Schmidt loses
/// orthogonality in proportion to how much of the column it removes.
fn qr_orthonormalize(mat: &mut [f64], m: usize, k: usize) {
    const DEPENDENT: f64 = 1e-10;
    let col_norm = |mat: &[f64], j: usize| -> f64 {
        (0..m).map(|r| mat[r * k + j] * mat[r * k + j]).sum::<f64>().sqrt()
    };
    // Columns are stored row-major across rows — treat as k column vectors.
    for j in 0..k {
        let original = col_norm(mat, j);
        for _pass in 0..2 {
            // Subtract projections onto previous columns.
            for i in 0..j {
                let mut dot = 0.0;
                for r in 0..m {
                    dot += mat[r * k + i] * mat[r * k + j];
                }
                for r in 0..m {
                    mat[r * k + j] -= dot * mat[r * k + i];
                }
            }
        }
        // Normalize column j, or drop it.
        let norm = col_norm(mat, j);
        if original > 0.0 && norm > DEPENDENT * original {
            let inv = 1.0 / norm;
            for r in 0..m {
                mat[r * k + j] *= inv;
            }
        } else {
            for r in 0..m {
                mat[r * k + j] = 0.0;
            }
        }
    }
}

/// Compute singular values of `B` (rows × cols) via the eigendecomposition of
/// whichever Gram matrix is smaller: `BBᵀ` when rows ≤ cols, else `BᵀB`.
///
/// The nonzero eigenvalues of the two Gram matrices coincide, so either gives
/// the squared singular values of B; using the smaller one keeps the Jacobi
/// rotation workload bounded when one dimension (typically the sketch width
/// `l` in randomized SVD) is much smaller than the other.
///
/// Returns singular values in descending order.
fn small_singular_values(b: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let (gram, dim) = if rows <= cols {
        // BBᵀ is rows × rows (small).
        let bt = transpose(b, rows, cols);
        let bb_t = mat_mul(b, &bt, rows, cols, rows);
        (bb_t, rows)
    } else {
        // BᵀB is cols × cols.
        let bt = transpose(b, rows, cols);
        let bt_b = mat_mul(&bt, b, cols, rows, cols);
        (bt_b, cols)
    };
    let mut eigvals = symmetric_eigvals_jacobi(&gram, dim);
    // Numerical noise can produce small negatives; clamp then sqrt.
    for e in eigvals.iter_mut() {
        if *e < 0.0 {
            *e = 0.0;
        }
        *e = e.sqrt();
    }
    eigvals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    eigvals
}

/// Jacobi eigenvalue algorithm for a symmetric matrix `A` (n × n, row-major).
///
/// Returns the eigenvalues (unsorted).  Converges in O(n³) total work for our
/// tiny `k × k` matrices (k ≤ oversampled rank ≈ 32).
fn symmetric_eigvals_jacobi(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let max_iters = 50 * n * n + 32;
    let tol = 1e-12;
    for _ in 0..max_iters {
        // Find the largest off-diagonal |m[p,q]|.
        let mut p = 0usize;
        let mut q = 1usize;
        let mut best = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = m[i * n + j].abs();
                if v > best {
                    best = v;
                    p = i;
                    q = j;
                }
            }
        }
        if best <= tol {
            break;
        }
        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Rotate rows/columns p,q.
        for i in 0..n {
            if i != p && i != q {
                let aip = m[i * n + p];
                let aiq = m[i * n + q];
                m[i * n + p] = c * aip - s * aiq;
                m[i * n + q] = s * aip + c * aiq;
                m[p * n + i] = m[i * n + p];
                m[q * n + i] = m[i * n + q];
            }
        }
        m[p * n + p] = app - t * apq;
        m[q * n + q] = aqq + t * apq;
        m[p * n + q] = 0.0;
        m[q * n + p] = 0.0;
    }
    (0..n).map(|i| m[i * n + i]).collect()
}

// ---------------------------------------------------------------------------
// Randomized SVD entry point
// ---------------------------------------------------------------------------

/// Result of a randomized SVD on a single weight matrix.
#[derive(Debug, Clone)]
pub struct SpectralAnalysis {
    /// Parameter name (for reporting).
    pub name: String,
    /// Matrix shape `[m, n]`.
    pub shape: [usize; 2],
    /// Top-`k` singular values in descending order (σ₁ ≥ … ≥ σₖ).
    pub singular_values: Vec<f64>,
    /// Effective rank via spectral entropy: `exp(H(p))`
    /// where `p_j = σ_j / Σσ`.
    pub effective_rank: f64,
    /// Total number of truncated singular values (= `k`).
    pub truncated_rank: usize,
}

/// Compute a randomized truncated SVD of the 2-D matrix `mat` (row-major,
/// `m × n`) with target rank `target_rank`.
///
/// `oversample` extra dimensions are used internally to improve accuracy of the
/// top singular values — the standard HMT recommendation is 5–10.  A single
/// power iteration is performed to accelerate decay of the sketching error
/// (also standard for slowly-decaying spectra).
pub fn randomized_svd(
    mat: &[f64],
    m: usize,
    n: usize,
    target_rank: usize,
    oversample: usize,
    seed: u64,
) -> Vec<f64> {
    let k = target_rank.max(1).min(m.min(n));
    let l = (k + oversample).min(n).max(1);
    let mut rng = Xorshift64::new(seed);

    // 1. Sample an n×l Gaussian random matrix Ω.
    let mut omega = vec![0.0; n * l];
    for v in omega.iter_mut() {
        *v = rng.next_normal();
    }

    // 2. Y = A Ω (m × l)
    let mut y = mat_mul(mat, &omega, m, n, l);

    // 3. Two power iterations Y = (A Aᵀ)² Ω, reorthonormalising between
    //    each half-step to keep the sketch numerically stable.  Two
    //    iterations are the sweet spot for the slow-decay spectra typical of
    //    pre-trained transformer weights.
    for _ in 0..2 {
        qr_orthonormalize(&mut y, m, l);
        let at = transpose(mat, m, n);
        let z = mat_mul(&at, &y, n, m, l);
        y = mat_mul(mat, &z, m, n, l);
    }

    // 4. QR on Y to get an orthonormal basis Q (m × l) for range(A).
    qr_orthonormalize(&mut y, m, l);

    // 5. B = Qᵀ A  (l × n)
    let qt = transpose(&y, m, l);
    let b = mat_mul(&qt, mat, l, m, n);

    // 6. Singular values of B approximate those of A.  Return top `k`.
    let mut sv = small_singular_values(&b, l, n);
    sv.truncate(k);
    sv
}

/// Compute spectral entropy `H(p) = -Σ pᵢ log pᵢ` and the exponentiated form
/// (effective rank).  `p` is the normalised singular-value distribution.
pub fn effective_rank_from_singular_values(sv: &[f64]) -> f64 {
    let sum: f64 = sv.iter().sum();
    if sum <= 1e-30 {
        return 0.0;
    }
    let mut h = 0.0;
    for &s in sv {
        if s > 1e-30 {
            let p = s / sum;
            h -= p * p.ln();
        }
    }
    h.exp()
}

/// Decode a single weight element into f64, used by the spectral loader.
fn decode_element(dtype: WeightDType, bytes: &[u8]) -> f64 {
    dtype.to_f64(bytes)
}

/// Copy a (possibly 3-D+) tensor's *effective 2-D* view as a row-major f64
/// matrix.  Matrices with ndim != 2 are flattened by reshape `[dim0,
/// prod(rest)]`.
fn load_matrix_f64(entry: &crate::weight_aware::WeightEntry) -> Option<(Vec<f64>, usize, usize)> {
    if entry.shape.is_empty() {
        return None;
    }
    let m = entry.shape[0];
    let n: usize = entry.shape[1..].iter().product::<usize>().max(1);
    let bw = entry.dtype.byte_width();
    let mut data = Vec::with_capacity(m * n);
    for idx in 0..(m * n) {
        let off = idx * bw;
        if off + bw > entry.data.len() {
            return None;
        }
        data.push(decode_element(entry.dtype, &entry.data[off..off + bw]));
    }
    Some((data, m, n))
}

/// Per-weight spectral analysis over every 2-D tensor in `wm`.
///
/// Weights with `shape.len() < 2` or fewer than `min_dim` rows/cols are
/// skipped (embedding tables, biases, norm gammas).
pub fn analyse_weight_map(
    wm: &WeightMap,
    target_rank: usize,
    oversample: usize,
    min_dim: usize,
    seed: u64,
) -> Vec<SpectralAnalysis> {
    let mut out = Vec::new();
    // In name order: the map iterates in a hash order that differs from run
    // to run, and everything downstream keeps this order.
    let mut entries: Vec<_> = wm.entries().collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    for (name, entry) in entries {
        if entry.shape.len() < 2 {
            continue;
        }
        let m = entry.shape[0];
        let n: usize = entry.shape[1..].iter().product();
        if m < min_dim || n < min_dim {
            continue;
        }
        let (mat, m, n) = match load_matrix_f64(entry) {
            Some(v) => v,
            None => continue,
        };
        let k_try = target_rank.min(m.min(n));
        let sv = randomized_svd(&mat, m, n, k_try, oversample, seed);
        let r_eff = effective_rank_from_singular_values(&sv);
        out.push(SpectralAnalysis {
            name: name.clone(),
            shape: [m, n],
            singular_values: sv,
            effective_rank: r_eff,
            truncated_rank: k_try,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Rank allocation
// ---------------------------------------------------------------------------

/// A single weight matrix's portion of the global adapter-parameter budget.
#[derive(Debug, Clone, PartialEq)]
pub struct RankAllocation {
    pub name: String,
    /// LoRA rank chosen for this weight, clipped to `[r_min, r_max]`.
    pub rank: usize,
    /// Effective rank used as the weighting factor.
    pub effective_rank: f64,
    /// Approximate adapter parameter count: `rank * (m + n)`.
    pub adapter_params: usize,
}

/// The paper's declared per-weight adapter-rank domain: `r[l,w] ∈ {0,2,4,8,16}`
/// (§2.2).  `0` means "no adapter".
pub const RANK_GRID: [usize; 5] = [0, 2, 4, 8, 16];

/// Snap a raw integer rank DOWN to the largest [`RANK_GRID`] value it does not
/// exceed, so the realized `r[l,w]` lives in the paper's declared domain
/// (audit gap #5).
///
/// Flooring (rather than rounding to nearest) is deliberate: it never increases
/// a rank, so a total-parameter budget the caller already satisfied stays
/// satisfied.  Ranks at or above the grid top (`16`) map to `16`.
pub fn snap_rank_to_grid(rank: usize) -> usize {
    RANK_GRID
        .iter()
        .rev()
        .copied()
        .find(|&g| g <= rank)
        .unwrap_or(0)
}

/// The existing paper §2.3 per-site formula, extracted as a helper.
///
/// Returns the spectral rank for site `s`, derived from its effective-rank
/// share of `r_total`, clamped to `[r_min, r_max]`.
fn compute_spectral_rank(
    _s: &SpectralAnalysis,
    score: f64,
    total_score: f64,
    r_total: usize,
    r_min: usize,
    r_max: usize,
) -> usize {
    let share = r_total as f64 * (score / total_score);
    (share.floor() as usize).clamp(r_min, r_max.max(r_min))
}

/// Priority for downgrade queue: `effective_rank × slack`.
/// Lower priority → victimized first when over budget.
fn priority_for_alloc(
    alloc: &RankAllocation,
    per_site_slack: Option<&HashMap<String, f64>>,
) -> f64 {
    let slack = per_site_slack
        .and_then(|m| m.get(&alloc.name).copied())
        .unwrap_or(1.0);
    alloc.effective_rank * slack.max(0.0)
}

/// Allocate per-layer ranks.
///
/// Uses the formula from Section 2.3 of the WRGA proposal:
///
/// ```text
///   rᵢ = clip(⌊R_total · r_eff(Wᵢ) / Σ_j r_eff(Wⱼ)⌋, r_min, r_max)
/// ```
///
/// `r_total` is that formula's pool of RANKS. `param_budget` is a cap on the
/// total PARAMETER count, `Σ rᵢ · (mᵢ + nᵢ)`, enforced afterwards by
/// [`enforce_param_budget`]; `None` means no cap. The two are different
/// units: one argument used to serve as both, so a parameter budget was
/// handed out as that many ranks and bounded nothing, while with WGGO
/// overrides the rank pool was compared against a parameter count.
///
/// `per_site_slack` is the optional per-weight roofline multiplier (Innovation
/// 2 × Innovation 3); pass `None` for uniform weighting.
///
/// `overrides` is the optional WGGO override map.  When `Some`, per-layer
/// adapter ranks are honored subject to the `[r_min, r_max]` clamp and the
/// parameter budget.  When `None` (or when a layer has no override entry),
/// the spectral formula above is used.
///
/// Returns `(allocations, diagnostics)`.  `diagnostics` is empty when
/// `overrides` is `None`.
#[allow(clippy::too_many_arguments)]
pub fn allocate_ranks(
    spectral: &[SpectralAnalysis],
    r_total: usize,
    param_budget: Option<usize>,
    r_min: usize,
    r_max: usize,
    per_site_slack: Option<&HashMap<String, f64>>,
    overrides: Option<&crate::wggo_overrides::WggoOverrides>,
    snap_to_grid: bool,
) -> (Vec<RankAllocation>, Vec<crate::wggo_overrides::OverrideDiagnostic>) {
    use crate::wggo_overrides::{OverrideDiagnostic, OverrideRejectReason};

    if spectral.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // ── Pre-compute spectral scores (needed for both override + spectral paths) ─
    let scores: Vec<f64> = spectral
        .iter()
        .map(|s| {
            let base = s.effective_rank.max(0.0);
            let slack = per_site_slack
                .and_then(|m| m.get(&s.name).copied())
                .unwrap_or(1.0);
            base * slack.max(0.0)
        })
        .collect();
    let total_score: f64 = scores.iter().sum();

    let mut allocations: Vec<RankAllocation> = Vec::new();
    let mut diags: Vec<OverrideDiagnostic> = Vec::new();

    if total_score <= 1e-30 {
        // All weights are zero / analysis failed — fall back to uniform r_min.
        // Override forbids are still honored in this degenerate case, but clamp
        // and spectral paths both reduce to r_min anyway, so skip the override
        // dispatch and just check for hard-forbids.
        for s in spectral {
            let ov = overrides.and_then(|o| o.find_by_layer_containing(&s.name));
            if let Some(ov) = ov
                && ov.adapter_rank == 0
            {
                // Spectral would have placed r_min; that means it *would* have
                // placed, so emit a diagnostic.
                diags.push(OverrideDiagnostic {
                    layer_index: ov.layer_index,
                    layer_name: s.name.clone(),
                    reason: OverrideRejectReason::RankForbiddenByWggo,
                    requested: "0".to_string(),
                    applied: "no_adapter".to_string(),
                });
                continue; // excluded
            }
            allocations.push(RankAllocation {
                name: s.name.clone(),
                rank: r_min,
                effective_rank: 0.0,
                adapter_params: r_min * (s.shape[0] + s.shape[1]),
            });
        }
        if snap_to_grid {
            snap_allocations_to_grid(&mut allocations, spectral);
        }
        return (allocations, diags);
    }

    // ── Step 1: Per-projection allocation ──────────────────────────────────
    for (i, s) in spectral.iter().enumerate() {
        let ov = overrides.and_then(|o| o.find_by_layer_containing(&s.name));

        let target_rank = match ov {
            Some(ov) if ov.adapter_rank == 0 => {
                // Hard forbid.  Emit diagnostic only when spectral would have placed
                // (i.e., the layer has a non-zero effective-rank score — it is not
                // a degenerate / zero-weight projection that spectral would also skip).
                if scores[i] > 1e-30 {
                    diags.push(OverrideDiagnostic {
                        layer_index: ov.layer_index,
                        layer_name: s.name.clone(),
                        reason: OverrideRejectReason::RankForbiddenByWggo,
                        requested: "0".to_string(),
                        applied: "no_adapter".to_string(),
                    });
                }
                continue; // skip allocation entirely
            }
            Some(ov) => {
                let requested = ov.adapter_rank as usize;
                let clamped = requested.clamp(r_min, r_max.max(r_min));
                if requested != clamped {
                    diags.push(OverrideDiagnostic {
                        layer_index: ov.layer_index,
                        layer_name: s.name.clone(),
                        reason: OverrideRejectReason::RankClampedToBounds {
                            r_min: r_min as u32,
                            r_max: r_max as u32,
                        },
                        requested: requested.to_string(),
                        applied: clamped.to_string(),
                    });
                }
                clamped
            }
            None => compute_spectral_rank(s, scores[i], total_score, r_total, r_min, r_max),
        };

        allocations.push(RankAllocation {
            name: s.name.clone(),
            rank: target_rank,
            effective_rank: s.effective_rank,
            adapter_params: target_rank * (s.shape[0] + s.shape[1]),
        });
    }

    // ── Step 2: Parameter budget ───────────────────────────────────────────
    // Lowest `priority_for_alloc` (effective_rank × slack) loses rank first.
    // A diagnostic is emitted only for a layer that has a WGGO override
    // entry; a layer on the spectral formula is downgraded silently.
    if let Some(budget) = param_budget {
        let m_plus_n = |name: &str| -> usize {
            spectral
                .iter()
                .find(|s| s.name == name)
                .map_or(0, |s| s.shape[0] + s.shape[1])
        };
        let changed =
            enforce_param_budget(&mut allocations, budget, r_min, &m_plus_n, per_site_slack);
        for idx in changed {
            let name = &allocations[idx].name;
            let Some(ov) = overrides.and_then(|o| o.find_by_layer_containing(name)) else {
                continue;
            };
            let final_rank = allocations[idx].rank as u32;
            diags.push(OverrideDiagnostic {
                layer_index: ov.layer_index,
                layer_name: name.clone(),
                reason: OverrideRejectReason::BudgetExceededDowngraded {
                    original_rank: ov.adapter_rank as u32,
                    final_rank,
                },
                requested: ov.adapter_rank.to_string(),
                applied: final_rank.to_string(),
            });
        }
    }

    // ── Step 3: Paper-grid snap (audit gap #5, opt-in) ─────────────────────
    // Floor every realized rank onto the paper's declared domain {0,2,4,8,16}
    // (§2.2).  Snapping only *lowers* ranks, so the parameter budget enforced
    // by Step 2 stays satisfied — no second downgrade pass is needed.
    if snap_to_grid {
        snap_allocations_to_grid(&mut allocations, spectral);
    }

    (allocations, diags)
}

/// Lower ranks until the allocations hold at most `budget` parameters.
///
/// `m_plus_n(name)` is `m + n` of the weight an allocation sits on, so that a
/// rank `r` there costs `r · (m + n)` parameters; 0 means the shape is
/// unknown, and such an allocation is left alone (it is not in the count).
///
/// One rank is taken at a time from the allocation with the lowest priority
/// (`effective_rank × slack`) among those above `r_min`; on equal priority,
/// from the one with the highest rank, so equal sites stay level. When every
/// allocation is at `r_min` and the total is still over, the lowest-priority
/// ones lose their adapter altogether (rank 0), which is the only way left
/// to meet the budget.
///
/// Returns the indices of the allocations whose rank changed.
pub fn enforce_param_budget(
    allocations: &mut [RankAllocation],
    budget: usize,
    r_min: usize,
    m_plus_n: &dyn Fn(&str) -> usize,
    per_site_slack: Option<&HashMap<String, f64>>,
) -> Vec<usize> {
    use std::cmp::Ordering;
    let mut changed: Vec<usize> = Vec::new();
    let total = |allocations: &[RankAllocation]| -> usize {
        allocations.iter().map(|a| a.adapter_params).sum()
    };
    while total(allocations) > budget {
        // The victim: lowest priority, then highest rank, then first.
        let pick = |floor: usize| -> Option<usize> {
            allocations
                .iter()
                .enumerate()
                .filter(|(_, a)| a.rank > floor && m_plus_n(&a.name) > 0)
                .min_by(|(_, aa), (_, bb)| {
                    let pa = priority_for_alloc(aa, per_site_slack);
                    let pb = priority_for_alloc(bb, per_site_slack);
                    pa.partial_cmp(&pb)
                        .unwrap_or(Ordering::Equal)
                        .then(bb.rank.cmp(&aa.rank))
                })
                .map(|(i, _)| i)
        };
        let (idx, new_rank) = match pick(r_min) {
            Some(idx) => (idx, allocations[idx].rank - 1),
            None => match pick(0) {
                Some(idx) => (idx, 0),
                None => break, // nothing left to take
            },
        };
        let a = &mut allocations[idx];
        a.rank = new_rank;
        a.adapter_params = new_rank * m_plus_n(&a.name);
        if !changed.contains(&idx) {
            changed.push(idx);
        }
    }
    changed
}

/// Floor every allocation's rank onto [`RANK_GRID`] (audit gap #5) and keep
/// `adapter_params` consistent with the snapped rank.
///
/// Applied at *both* `allocate_ranks` exits (the degenerate zero-score path and
/// the normal path) so that `snap_to_grid = true` guarantees an on-grid result
/// regardless of which path produced the allocation.  `adapter_params` is
/// recomputed from `m + n` (looked up by name), which stays correct even when a
/// rank snaps to 0.  Assumes `r_min` is itself a grid value (it is `2` in
/// production); with an off-grid `r_min` the floor can drop a rank below it.
fn snap_allocations_to_grid(allocations: &mut [RankAllocation], spectral: &[SpectralAnalysis]) {
    for a in allocations.iter_mut() {
        let snapped = snap_rank_to_grid(a.rank);
        if snapped != a.rank {
            let m_plus_n = spectral
                .iter()
                .find(|s| s.name == a.name)
                .map(|s| s.shape[0] + s.shape[1])
                .unwrap_or(0);
            a.rank = snapped;
            a.adapter_params = snapped * m_plus_n;
        }
    }
}

/// Sum of adapter parameters across a plan.
pub fn total_adapter_params(plan: &[RankAllocation]) -> usize {
    plan.iter().map(|r| r.adapter_params).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a rank-`r` matrix A = U Σ Vᵀ with geometrically decaying
    /// singular values, so we can check that randomized SVD recovers them.
    fn rank_r_matrix(m: usize, n: usize, r: usize, decay: f64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Xorshift64::new(42);
        let mut u = vec![0.0; m * r];
        for v in u.iter_mut() {
            *v = rng.next_normal();
        }
        qr_orthonormalize(&mut u, m, r);
        let mut vt = vec![0.0; r * n];
        for v in vt.iter_mut() {
            *v = rng.next_normal();
        }
        // Orthonormalize rows of V by QR on its transpose.
        let mut v_mat = transpose(&vt, r, n);
        qr_orthonormalize(&mut v_mat, n, r);
        let vt = transpose(&v_mat, n, r);

        // Apply Σ to the rows of Vᵀ.
        let mut sigma = vec![1.0; r];
        for i in 0..r {
            sigma[i] = decay.powi(i as i32);
        }
        let mut sv = vt.clone();
        for i in 0..r {
            for j in 0..n {
                sv[i * n + j] *= sigma[i];
            }
        }
        let a = mat_mul(&u, &sv, m, r, n);
        (a, sigma)
    }

    /// A 40×30 matrix of rank exactly 5, built as U Σ Vᵀ with Σ = (1, ½, ¼,
    /// ⅛, 1⁄16): the sketch returns those five values. Its ten columns span
    /// only five directions, so five of them are linearly dependent; they
    /// used to be normalised into noise, which put σ₁ at 1.333.
    #[test]
    fn randomized_svd_recovers_the_singular_values_of_a_low_rank_matrix() {
        let (a, true_sigma) = rank_r_matrix(40, 30, 5, 0.5);
        assert_eq!(true_sigma, [1.0, 0.5, 0.25, 0.125, 0.0625]);
        let sv = randomized_svd(&a, 40, 30, 5, 5, 0xC0FFEE);
        assert_eq!(sv.len(), 5);
        for (i, (got, want)) in sv.iter().zip(&true_sigma).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "σ{}: sketch {got}, constructed {want} (all: {sv:?})",
                i + 1
            );
        }
    }

    /// Gram–Schmidt on columns (e₀, e₁, e₀ + e₁, e₂) of a 4×4 matrix: the
    /// third is dependent on the first two and comes out zero; the others
    /// are orthonormal.
    #[test]
    fn a_linearly_dependent_column_is_dropped_not_normalised() {
        let mut q = vec![
            1.0, 0.0, 1.0, 0.0, //
            0.0, 1.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        qr_orthonormalize(&mut q, 4, 4);
        let col = |j: usize| -> Vec<f64> { (0..4).map(|r| q[r * 4 + j]).collect() };
        let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let expect_norm = [1.0, 1.0, 0.0, 1.0];
        for i in 0..4 {
            for j in 0..4 {
                let want = if i == j { expect_norm[i] } else { 0.0 };
                let got = dot(&col(i), &col(j));
                assert!((got - want).abs() < 1e-12, "QᵀQ[{i}][{j}] = {got}, want {want}");
            }
        }
    }

    #[test]
    fn effective_rank_small_for_fast_decay() {
        // Fast decay → effective rank should be small.
        let sv = vec![10.0, 0.01, 1e-6, 1e-12];
        let r = effective_rank_from_singular_values(&sv);
        assert!(r < 2.0, "expected r_eff < 2, got {r}");
    }

    #[test]
    fn effective_rank_large_for_flat_spectrum() {
        // Flat spectrum → effective rank ≈ n.
        let sv = vec![1.0; 8];
        let r = effective_rank_from_singular_values(&sv);
        assert!((r - 8.0).abs() < 1e-6);
    }

    #[test]
    fn allocate_respects_budget_and_bounds() {
        let spectral = vec![
            SpectralAnalysis {
                name: "a".into(),
                shape: [64, 64],
                singular_values: vec![1.0; 8],
                effective_rank: 8.0,
                truncated_rank: 8,
            },
            SpectralAnalysis {
                name: "b".into(),
                shape: [64, 64],
                singular_values: vec![1.0, 1e-9, 1e-9, 1e-9],
                effective_rank: 1.0,
                truncated_rank: 4,
            },
        ];
        let (plan, diags) = allocate_ranks(&spectral, 16, None, 1, 16, None, None, false);
        assert!(diags.is_empty());
        assert_eq!(plan.len(), 2);
        // "a" has 8× the effective rank of "b" → gets most of the budget.
        assert!(plan[0].rank >= plan[1].rank);
        for r in &plan {
            assert!(r.rank >= 1);
            assert!(r.rank <= 16);
        }
    }

    #[test]
    fn allocate_zero_score_falls_back_to_rmin() {
        let spectral = vec![SpectralAnalysis {
            name: "dead".into(),
            shape: [16, 16],
            singular_values: vec![0.0; 4],
            effective_rank: 0.0,
            truncated_rank: 4,
        }];
        let (plan, _diags) = allocate_ranks(&spectral, 8, None, 2, 16, None, None, false);
        assert_eq!(plan[0].rank, 2);
    }

    #[test]
    fn allocate_with_roofline_slack_reweights() {
        let spectral = vec![
            SpectralAnalysis {
                name: "memory_bound".into(),
                shape: [64, 64],
                singular_values: vec![1.0; 4],
                effective_rank: 4.0,
                truncated_rank: 4,
            },
            SpectralAnalysis {
                name: "compute_bound".into(),
                shape: [64, 64],
                singular_values: vec![1.0; 4],
                effective_rank: 4.0,
                truncated_rank: 4,
            },
        ];
        let mut slack = HashMap::new();
        slack.insert("memory_bound".to_string(), 2.0);
        slack.insert("compute_bound".to_string(), 0.25);
        let (plan, _diags) = allocate_ranks(&spectral, 16, None, 1, 16, Some(&slack), None, false);
        // Memory-bound site should get more rank because of its larger slack.
        let mem = plan.iter().find(|r| r.name == "memory_bound").unwrap();
        let com = plan.iter().find(|r| r.name == "compute_bound").unwrap();
        assert!(mem.rank > com.rank);
    }

    fn six_equal_sites() -> Vec<SpectralAnalysis> {
        (0..6)
            .map(|i| SpectralAnalysis {
                name: format!("w{i}"),
                shape: [32, 32],
                singular_values: vec![1.0; 8],
                effective_rank: 8.0,
                truncated_rank: 8,
            })
            .collect()
    }

    /// Six equal 32×32 sites and a pool of 96 ranks: 16 each, 6,144
    /// parameters. A parameter budget above that changes nothing; 1,000
    /// leaves 15 ranks (960 parameters, each rank being 64). The budget used
    /// to be passed as the rank pool, where 1,000 and 100,000 both gave
    /// 6,144.
    #[test]
    fn a_parameter_budget_caps_the_parameter_count() {
        let spectral = six_equal_sites();
        let total = |budget: Option<usize>| -> (usize, Vec<usize>) {
            let (plan, _) = allocate_ranks(&spectral, 96, budget, 2, 16, None, None, false);
            (total_adapter_params(&plan), plan.iter().map(|r| r.rank).collect())
        };
        assert_eq!(total(None), (6_144, vec![16; 6]));
        assert_eq!(total(Some(100_000)), (6_144, vec![16; 6]));
        let (params, ranks) = total(Some(1_000));
        assert_eq!(params, 960);
        assert_eq!(ranks.iter().sum::<usize>(), 15);
        // Equal sites stay level: no rank differs from another by more than 1.
        assert!(ranks.iter().max().unwrap() - ranks.iter().min().unwrap() <= 1, "{ranks:?}");
    }

    /// When even `r_min` everywhere is over the budget, sites lose their
    /// adapter: 500 parameters pay for 7 ranks, three sites at rank 2.
    #[test]
    fn a_budget_below_the_minimum_ranks_drops_sites() {
        let spectral = six_equal_sites();
        let (plan, _) = allocate_ranks(&spectral, 96, Some(500), 2, 16, None, None, false);
        let mut ranks: Vec<usize> = plan.iter().map(|r| r.rank).collect();
        ranks.sort_unstable();
        assert_eq!(ranks, [0, 0, 0, 2, 2, 2]);
        assert_eq!(total_adapter_params(&plan), 384);
    }

    /// The lower-priority site gives up its ranks first.
    #[test]
    fn the_budget_takes_from_the_lowest_priority_site_first() {
        let mut spectral = six_equal_sites();
        spectral.truncate(2);
        spectral[0].effective_rank = 2.0;
        // Pool of 16 ranks: shares 3.2 and 12.8, floored to 3 and 12.
        let (free, _) = allocate_ranks(&spectral, 16, None, 2, 16, None, None, false);
        assert_eq!(free.iter().map(|r| r.rank).collect::<Vec<_>>(), [3, 12]);
        // One rank less than that fits: 14 × 64 = 896. Site 0 pays.
        let (capped, _) = allocate_ranks(&spectral, 16, Some(896), 2, 16, None, None, false);
        assert_eq!(capped.iter().map(|r| r.rank).collect::<Vec<_>>(), [2, 12]);
    }
}

// ---------------------------------------------------------------------------
// Override tests (Task 5)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod override_tests {
    use super::*;
    use crate::wggo_overrides::{OverrideRejectReason, PerLayerOverride, WggoOverrides};

    /// 4-layer spectral fixture with decreasing effective rank.
    /// All layers use shape [32, 32] so m+n = 64.
    fn spectral_fixture_4_layers() -> Vec<SpectralAnalysis> {
        vec![
            SpectralAnalysis {
                name: "blocks.0.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 10],
                effective_rank: 10.0,
                truncated_rank: 10,
            },
            SpectralAnalysis {
                name: "blocks.1.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 8],
                effective_rank: 8.0,
                truncated_rank: 8,
            },
            SpectralAnalysis {
                name: "blocks.2.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 6],
                effective_rank: 6.0,
                truncated_rank: 6,
            },
            SpectralAnalysis {
                name: "blocks.3.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 4],
                effective_rank: 4.0,
                truncated_rank: 4,
            },
        ]
    }

    /// Build a WggoOverrides where each entry's layer_name is the prefix
    /// ("blocks.0", "blocks.1", …) so `find_by_layer_containing` matches
    /// the full projection name ("blocks.0.attn.wq").
    fn overrides_with_ranks(entries: &[(&str, u64)]) -> WggoOverrides {
        WggoOverrides {
            per_layer: entries
                .iter()
                .enumerate()
                .map(|(i, (name, rank))| PerLayerOverride {
                    layer_index: i as u32,
                    layer_name: name.to_string(),
                    active_heads: 8,
                    requested_csha_level: None,
                    adapter_rank: *rank,
                    adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                    fase_fused: false,
                    packing_mode: 0,
                    shard_factor: 0,
                    optim_m_bits: 32,
                    optim_v_bits: 32,
                })
                .collect(),
        }
    }

    /// Sum of adapter_params across allocations.
    fn total_params(allocs: &[RankAllocation]) -> usize {
        allocs.iter().map(|a| a.adapter_params).sum()
    }

    // ── Test 1: No overrides → spectral behavior preserved (byte-identical) ─

    #[test]
    fn allocate_ranks_no_overrides_preserves_spectral_behavior() {
        let spectral = spectral_fixture_4_layers();
        let (alloc, diags) = allocate_ranks(&spectral, 256, None, 2, 16, None, None, false);
        assert!(diags.is_empty(), "no overrides → no diagnostics");
        assert_eq!(alloc.len(), 4);
        for a in &alloc {
            assert!(a.rank >= 2 && a.rank <= 16, "rank {} out of [2,16]", a.rank);
        }
    }

    // ── Test 2: Feasible WGGO ranks applied verbatim ──────────────────────

    #[test]
    fn allocate_ranks_honors_feasible_override() {
        let spectral = spectral_fixture_4_layers();
        let over = overrides_with_ranks(&[
            ("blocks.0", 8),
            ("blocks.1", 8),
            ("blocks.2", 4),
            ("blocks.3", 4),
        ]);
        // Budget large enough that no downgrade fires.
        let (alloc, diags) = allocate_ranks(&spectral, 100_000, None, 2, 16, None, Some(&over), false);
        assert!(diags.is_empty(), "all requests fit; no diagnostics, got: {diags:?}");
        let ranks: Vec<usize> = alloc.iter().map(|a| a.rank).collect();
        assert_eq!(ranks, vec![8, 8, 4, 4]);
    }

    // ── Test 3: Clamp above r_max ─────────────────────────────────────────

    #[test]
    fn allocate_ranks_clamps_rank_exceeding_r_max() {
        let spectral = spectral_fixture_4_layers();
        let over = overrides_with_ranks(&[
            ("blocks.0", 32), // > r_max=16 → clamped
            ("blocks.1", 8),
            ("blocks.2", 4),
            ("blocks.3", 4),
        ]);
        let (alloc, diags) = allocate_ranks(&spectral, 100_000, None, 2, 16, None, Some(&over), false);
        assert_eq!(alloc[0].rank, 16, "clamped to r_max");
        let clamp_diag = diags
            .iter()
            .find(|d| {
                d.layer_name == "blocks.0.attn.wq"
                    && matches!(
                        d.reason,
                        OverrideRejectReason::RankClampedToBounds { r_max: 16, .. }
                    )
            })
            .expect("expected clamp diagnostic for blocks.0.attn.wq");
        assert_eq!(clamp_diag.requested, "32");
        assert_eq!(clamp_diag.applied, "16");
    }

    // ── Test 4: Clamp below r_min ─────────────────────────────────────────

    #[test]
    fn allocate_ranks_clamps_rank_below_r_min() {
        let spectral = spectral_fixture_4_layers();
        let over = overrides_with_ranks(&[
            ("blocks.0", 1), // < r_min=2 → clamped
            ("blocks.1", 8),
            ("blocks.2", 4),
            ("blocks.3", 4),
        ]);
        let (alloc, diags) = allocate_ranks(&spectral, 100_000, None, 2, 16, None, Some(&over), false);
        assert_eq!(alloc[0].rank, 2, "clamped to r_min");
        assert!(
            diags.iter().any(|d| {
                d.layer_name == "blocks.0.attn.wq"
                    && matches!(
                        d.reason,
                        OverrideRejectReason::RankClampedToBounds { r_min: 2, .. }
                    )
            }),
            "expected r_min clamp diagnostic"
        );
    }

    // ── Test 5: adapter_rank == 0 forbids when spectral would have placed ─

    #[test]
    fn allocate_ranks_forbids_when_adapter_rank_zero_and_spectral_wanted_placement() {
        let spectral = spectral_fixture_4_layers();
        let over = overrides_with_ranks(&[
            ("blocks.0", 0), // hard forbid
            ("blocks.1", 8),
            ("blocks.2", 4),
            ("blocks.3", 4),
        ]);
        let (alloc, diags) = allocate_ranks(&spectral, 100_000, None, 2, 16, None, Some(&over), false);
        assert!(
            alloc.iter().all(|a| a.name != "blocks.0.attn.wq"),
            "blocks.0 projection must be excluded from allocation"
        );
        assert!(
            diags.iter().any(|d| {
                d.layer_name == "blocks.0.attn.wq"
                    && matches!(d.reason, OverrideRejectReason::RankForbiddenByWggo)
            }),
            "expected RankForbiddenByWggo diagnostic"
        );
    }

    // ── Test 6: adapter_rank == 0 is silent when spectral also rejected ───

    #[test]
    fn allocate_ranks_forbids_silently_when_spectral_also_rejected() {
        let mut spectral = spectral_fixture_4_layers();
        // Make blocks.0 degenerate so spectral_rank < r_min — spectral wouldn't place.
        // r_total is tiny so share ≈ 0 → spectral_rank = 0 < r_min=2.
        spectral[0].effective_rank = 0.0;
        spectral[0].singular_values = vec![0.0; 4];

        let over = overrides_with_ranks(&[
            ("blocks.0", 0), // forbid on a degenerate site
            ("blocks.1", 8),
            ("blocks.2", 4),
            ("blocks.3", 4),
        ]);
        // r_total large enough that non-degenerate layers aren't clamped.
        let (alloc, diags) = allocate_ranks(&spectral, 100_000, None, 2, 16, None, Some(&over), false);
        assert!(
            alloc.iter().all(|a| a.name != "blocks.0.attn.wq"),
            "blocks.0 excluded"
        );
        let blocks_0_diags = diags
            .iter()
            .filter(|d| d.layer_name == "blocks.0.attn.wq")
            .count();
        assert_eq!(
            blocks_0_diags, 0,
            "no diagnostic when spectral wouldn't have placed anyway"
        );
    }

    // ── Test 7: Budget-exceeded triggers downgrade diagnostics ───────────

    #[test]
    fn allocate_ranks_downgrades_layers_when_over_budget() {
        let spectral = spectral_fixture_4_layers();
        // Request rank 16 on all layers. shape=[32,32] → m+n=64.
        // Total at rank 16: 4 * 16 * 64 = 4096 params.
        // Set budget to half: 2048.
        let over = overrides_with_ranks(&[
            ("blocks.0", 16),
            ("blocks.1", 16),
            ("blocks.2", 16),
            ("blocks.3", 16),
        ]);
        let tight_budget = 4 * 16 * 64 / 2; // = 2048
        let (alloc, diags) = allocate_ranks(&spectral, 64, Some(tight_budget), 2, 16, None, Some(&over), false);
        assert!(
            total_params(&alloc) <= tight_budget,
            "final total {} must fit budget {}",
            total_params(&alloc),
            tight_budget
        );
        let downgrades: Vec<_> = diags
            .iter()
            .filter(|d| {
                matches!(
                    d.reason,
                    OverrideRejectReason::BudgetExceededDowngraded { .. }
                )
            })
            .collect();
        assert!(!downgrades.is_empty(), "expected ≥1 downgrade diagnostic");
        for d in &downgrades {
            if let OverrideRejectReason::BudgetExceededDowngraded {
                original_rank,
                final_rank,
            } = d.reason
            {
                assert_eq!(original_rank, 16, "original_rank must be 16");
                assert!(final_rank < 16, "final_rank {final_rank} must be < 16");
            }
        }
    }

    // ── Test 8: Downgrade targets lowest-priority layer first ────────────

    #[test]
    fn allocate_ranks_downgrade_targets_lowest_priority_first() {
        // Two layers, same shape [32,32] → m+n=64.
        // blocks.0 has effective_rank=2.0 (lower priority).
        // blocks.1 has effective_rank=10.0 (higher priority).
        // Both requested rank=8. total_at_8 = 2 * 8 * 64 = 1024.
        // Set budget = 1024 - 64 = 960 (exactly one rank-step under).
        // Expected: blocks.0 (lower priority) loses 1 rank → 7; blocks.1 stays at 8.
        let spectral = vec![
            SpectralAnalysis {
                name: "blocks.0.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 2],
                effective_rank: 2.0,
                truncated_rank: 2,
            },
            SpectralAnalysis {
                name: "blocks.1.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 10],
                effective_rank: 10.0,
                truncated_rank: 10,
            },
        ];
        let over = overrides_with_ranks(&[("blocks.0", 8), ("blocks.1", 8)]);
        let total_at_8 = 2 * 8 * 64_usize;  // 1024
        let tight = total_at_8 - 64;        // 960 — exactly 1 rank step under
        let (alloc, _diags) = allocate_ranks(&spectral, 32, Some(tight), 2, 16, None, Some(&over), false);
        let b0 = alloc.iter().find(|a| a.name == "blocks.0.attn.wq").unwrap();
        let b1 = alloc.iter().find(|a| a.name == "blocks.1.attn.wq").unwrap();
        assert_eq!(b0.rank, 7, "lower-priority layer loses 1 rank");
        assert_eq!(b1.rank, 8, "higher-priority layer keeps full rank");
    }

    // ── Test 9: paper-grid snap (audit gap #5) ───────────────────────────

    #[test]
    fn snap_rank_to_grid_floors_to_paper_grid() {
        // Paper §2.2 declares r[l,w] ∈ {0,2,4,8,16}.  Snapping floors each rank
        // DOWN to the largest grid value it does not exceed (budget-safe: never
        // increases the adapter-parameter total).
        assert_eq!(snap_rank_to_grid(0), 0);
        assert_eq!(snap_rank_to_grid(1), 0);
        assert_eq!(snap_rank_to_grid(2), 2);
        assert_eq!(snap_rank_to_grid(3), 2);
        assert_eq!(snap_rank_to_grid(4), 4);
        assert_eq!(snap_rank_to_grid(5), 4);
        assert_eq!(snap_rank_to_grid(7), 4);
        assert_eq!(snap_rank_to_grid(8), 8);
        assert_eq!(snap_rank_to_grid(15), 8);
        assert_eq!(snap_rank_to_grid(16), 16);
        // Ranks above the grid top clamp to the largest grid value.
        assert_eq!(snap_rank_to_grid(17), 16);
        assert_eq!(snap_rank_to_grid(1000), 16);
        // Every output is a member of the declared grid.
        for r in 0..=64 {
            assert!(RANK_GRID.contains(&snap_rank_to_grid(r)));
        }
    }

    #[test]
    fn allocate_ranks_snap_floors_realized_ranks_and_preserves_budget() {
        // Same scenario as the downgrade test: `blocks.0` downgrades to the
        // OFF-grid rank 7.  With `snap_to_grid = true` it must floor onto the
        // paper grid {0,2,4,8,16}.
        let spectral = vec![
            SpectralAnalysis {
                name: "blocks.0.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 2],
                effective_rank: 2.0,
                truncated_rank: 2,
            },
            SpectralAnalysis {
                name: "blocks.1.attn.wq".into(),
                shape: [32, 32],
                singular_values: vec![1.0; 10],
                effective_rank: 10.0,
                truncated_rank: 10,
            },
        ];
        let over = overrides_with_ranks(&[("blocks.0", 8), ("blocks.1", 8)]);
        let tight = 2 * 8 * 64_usize - 64; // 960 — forces blocks.0 down to 7

        // Opt-in is off by default: the off-grid rank 7 survives.
        let (off, _) = allocate_ranks(&spectral, 32, Some(tight), 2, 16, None, Some(&over), false);
        assert_eq!(
            off.iter().find(|a| a.name == "blocks.0.attn.wq").unwrap().rank,
            7,
            "snap=false leaves the realized rank off-grid"
        );

        // Opt-in on: 7 floors to 4; the on-grid 8 is unchanged.
        let (on, _) = allocate_ranks(&spectral, 32, Some(tight), 2, 16, None, Some(&over), true);
        let b0 = on.iter().find(|a| a.name == "blocks.0.attn.wq").unwrap();
        let b1 = on.iter().find(|a| a.name == "blocks.1.attn.wq").unwrap();
        assert_eq!(b0.rank, 4, "off-grid 7 floors to grid value 4");
        assert_eq!(b1.rank, 8, "on-grid 8 is unchanged");
        for a in &on {
            assert!(RANK_GRID.contains(&a.rank), "rank {} not on paper grid", a.rank);
            // adapter_params stays consistent with the snapped rank (m+n = 64).
            assert_eq!(a.adapter_params, a.rank * 64);
        }
        // Budget-safe: snapping DOWN never increases the total, and it still fits.
        let total_off: usize = off.iter().map(|a| a.adapter_params).sum();
        let total_on: usize = on.iter().map(|a| a.adapter_params).sum();
        assert!(total_on <= total_off, "snap must not increase total params");
        assert!(total_on <= tight, "snapped total must fit the budget");
    }
}
