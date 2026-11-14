//! CEP — weight-magnitude importance scoring.
//!
//! Per-component scores that order the pruning search:
//!
//! ```text
//! importance(head h in layer l) =
//!     ‖Wq_h‖ · ‖Wk_g‖ · ‖Wv_g‖ · ‖Wo_h‖        (g = h's key/value group)
//!     × spectral_energy(Wq) × roofline_slack × position_factor
//! importance(FFN neuron j in layer l) =
//!     ‖w_gate_j‖ · ‖w_up_j‖ · ‖w_down_j‖ × roofline_slack × position_factor
//! ```
//!
//! NSL stores a linear weight as `[in, out]` (the forward is `x @ w`), so a
//! head is a block of `head_dim` COLUMNS of `wq`, its key/value group a block
//! of columns of `wk`/`wv`, and a block of ROWS of `wo`. An FFN neuron is a
//! column of `w_gate` and `w_up` and a row of `w_down`.
//!
//! This is a magnitude criterion. It needs no forward pass, and it is known
//! to be a weak guide to which components a model needs; a component whose
//! score is exactly zero contributes nothing through that product, which is
//! the one case the score is reliable for.

use serde::Serialize;

use crate::cep_oracle::ModelSpec;
use crate::cep_weights::BlockIndex;
use crate::weight_aware::{WeightEntry, WeightMap};

/// Per-attention-head importance.
#[derive(Debug, Clone, Serialize)]
pub struct HeadImportance {
    pub layer: u32,
    pub head: u32,
    pub weight_magnitude: f64,
    pub spectral_energy: f64,
    pub roofline_slack: f64,
    pub position_factor: f64,
    pub score: f64,
}

/// Per-layer FFN importance, with the per-neuron magnitudes it sums.
#[derive(Debug, Clone, Serialize)]
pub struct FfnImportance {
    pub layer: u32,
    /// Sum of `neuron_magnitudes`.
    pub weight_magnitude: f64,
    pub position_factor: f64,
    pub roofline_slack: f64,
    pub score: f64,
    /// `‖w_gate_j‖ · ‖w_up_j‖ · ‖w_down_j‖` for each neuron `j`. Empty when
    /// no weights were available.
    #[serde(skip)]
    pub neuron_magnitudes: Vec<f64>,
}

/// Per-layer aggregate importance.
#[derive(Debug, Clone, Serialize)]
pub struct LayerImportance {
    pub layer: u32,
    pub attention_score: f64,
    pub ffn_score: f64,
    pub total_score: f64,
}

/// Aggregate importance table.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ImportanceTable {
    pub heads: Vec<HeadImportance>,
    pub ffns: Vec<FfnImportance>,
    pub layers: Vec<LayerImportance>,
}

impl ImportanceTable {
    pub fn num_heads(&self) -> usize {
        self.heads.len()
    }
    pub fn num_ffns(&self) -> usize {
        self.ffns.len()
    }
}

/// Per-site roofline-slack ratio: memory-bound sites (slack > 1) get a
/// bonus because their compute is "free" on the target GPU.
#[derive(Debug, Clone, Default)]
pub struct RooflineSlackTable {
    /// Keyed by layer index (`BTreeMap` semantics via sorted vec).
    pub per_layer: Vec<(u32, f64)>,
}

impl RooflineSlackTable {
    pub fn get(&self, layer: u32) -> f64 {
        self.per_layer
            .iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, s)| *s)
            .unwrap_or(1.0)
    }
}

/// Position factor: `1.0 + 0.3 × |2·l/(L−1) − 1|`.
///
/// A U-shaped curve over layers `0..L`: 1.3 at the first and last layer and
/// 1.0 in the middle. The proposal writes `2l/L`, which is not symmetric
/// (1.3 at layer 0 but 1.225 at layer 7 of 8); dividing by `L−1` is.
pub fn position_factor(layer: u32, n_layers: u32) -> f64 {
    if n_layers <= 1 {
        return 1.0;
    }
    let normalized = (layer as f64) / ((n_layers - 1) as f64);
    1.0 + 0.3 * (2.0 * normalized - 1.0).abs()
}

/// Read element `[r, c]` of a row-major 2-D entry.
fn at(entry: &WeightEntry, cols: usize, r: usize, c: usize) -> f64 {
    let bw = entry.dtype.byte_width();
    let off = (r * cols + c) * bw;
    entry.dtype.to_f64(&entry.data[off..off + bw])
}

/// Check that `entry` is a 2-D tensor of exactly `[rows, cols]` whose data
/// covers that shape, so the readers below cannot index out of bounds.
fn expect_shape(entry: &WeightEntry, rows: usize, cols: usize, what: &str) -> Result<(), String> {
    if entry.shape != [rows, cols] {
        return Err(format!(
            "CEP: '{}' has shape {:?}, expected [{rows}, {cols}] ({what}; NSL stores a linear \
             weight as [in, out])",
            entry.name, entry.shape
        ));
    }
    if entry.data.len() != rows * cols * entry.dtype.byte_width() {
        return Err(format!(
            "CEP: '{}' holds {} bytes, expected {} for shape [{rows}, {cols}]",
            entry.name,
            entry.data.len(),
            rows * cols * entry.dtype.byte_width()
        ));
    }
    Ok(())
}

/// Frobenius norm of columns `[c0, c1)` of a `[rows, cols]` entry.
fn column_block_norm(entry: &WeightEntry, c0: usize, c1: usize) -> f64 {
    let (rows, cols) = (entry.shape[0], entry.shape[1]);
    let mut sum_sq = 0.0_f64;
    for r in 0..rows {
        for c in c0..c1 {
            let v = at(entry, cols, r, c);
            sum_sq += v * v;
        }
    }
    sum_sq.sqrt()
}

/// Frobenius norm of rows `[r0, r1)` of a `[rows, cols]` entry.
fn row_block_norm(entry: &WeightEntry, r0: usize, r1: usize) -> f64 {
    let cols = entry.shape[1];
    let mut sum_sq = 0.0_f64;
    for r in r0..r1 {
        for c in 0..cols {
            let v = at(entry, cols, r, c);
            sum_sq += v * v;
        }
    }
    sum_sq.sqrt()
}

/// The four projections of one attention layer, in NSL's `[in, out]` layout,
/// with the dimensions that say where each head lives in them.
pub struct AttentionWeights<'a> {
    pub wq: &'a WeightEntry,
    pub wk: &'a WeightEntry,
    pub wv: &'a WeightEntry,
    pub wo: &'a WeightEntry,
    pub d_model: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

impl AttentionWeights<'_> {
    /// Refuse any tensor whose shape is not the one the dimensions imply.
    pub fn validate(&self) -> Result<(), String> {
        let d = self.d_model as usize;
        let q = (self.n_heads * self.head_dim) as usize;
        let kv = (self.n_kv_heads * self.head_dim) as usize;
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!(
                "CEP: n_heads ({}) must be a positive multiple of n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            ));
        }
        expect_shape(self.wq, d, q, "d_model x n_heads*head_dim")?;
        expect_shape(self.wk, d, kv, "d_model x n_kv_heads*head_dim")?;
        expect_shape(self.wv, d, kv, "d_model x n_kv_heads*head_dim")?;
        expect_shape(self.wo, q, d, "n_heads*head_dim x d_model")
    }

    /// `‖Wq_h‖ · ‖Wk_g‖ · ‖Wv_g‖ · ‖Wo_h‖` for head `head`, where `g` is the
    /// key/value head it shares (`head / (n_heads / n_kv_heads)`). Call
    /// [`Self::validate`] first.
    pub fn head_magnitude(&self, head: u32) -> f64 {
        let hd = self.head_dim as usize;
        let group = (self.n_heads / self.n_kv_heads) as usize;
        let h = head as usize;
        let g = h / group;
        column_block_norm(self.wq, h * hd, (h + 1) * hd)
            * column_block_norm(self.wk, g * hd, (g + 1) * hd)
            * column_block_norm(self.wv, g * hd, (g + 1) * hd)
            * row_block_norm(self.wo, h * hd, (h + 1) * hd)
    }
}

/// Per-neuron FFN magnitudes: `‖w_gate_j‖ · ‖w_up_j‖ · ‖w_down_j‖`, the L2
/// norms of column `j` of `w_gate` and `w_up` (`[d_model, d_ff]`) and of row
/// `j` of `w_down` (`[d_ff, d_model]`). A neuron with any factor zero
/// contributes nothing to `(silu(x·gate) ⊙ x·up)·down`.
pub fn ffn_neuron_magnitudes(
    w_gate: &WeightEntry,
    w_up: &WeightEntry,
    w_down: &WeightEntry,
    d_model: u32,
    d_ff: u32,
) -> Result<Vec<f64>, String> {
    let (d, ff) = (d_model as usize, d_ff as usize);
    expect_shape(w_gate, d, ff, "d_model x d_ff")?;
    expect_shape(w_up, d, ff, "d_model x d_ff")?;
    expect_shape(w_down, ff, d, "d_ff x d_model")?;
    Ok((0..ff)
        .map(|j| {
            column_block_norm(w_gate, j, j + 1)
                * column_block_norm(w_up, j, j + 1)
                * row_block_norm(w_down, j, j + 1)
        })
        .collect())
}

/// The `keep` highest-magnitude neurons, as ascending indices. Ties go to the
/// lower index, so the planner and the slicer always agree on the set.
pub fn ffn_survivors(magnitudes: &[f64], keep: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..magnitudes.len() as u32).collect();
    order.sort_by(|&a, &b| {
        magnitudes[b as usize]
            .partial_cmp(&magnitudes[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order.truncate(keep);
    order.sort_unstable();
    order
}

/// Approximate spectral energy via power iteration: ratio of the dominant
/// singular value's squared magnitude to the Frobenius norm squared.
///
/// Values close to 1.0 indicate the matrix is nearly rank-1 (all energy
/// in one direction); low values indicate diffuse, full-rank weights.
///
/// Uses 5 power iterations on a random-seeded vector — more than enough
/// to converge for importance scoring (we only need two-decimal accuracy).
pub fn spectral_energy(entry: &WeightEntry) -> f64 {
    if entry.shape.len() != 2 {
        return 0.0;
    }
    let rows = entry.shape[0];
    let cols = entry.shape[1];
    if rows == 0 || cols == 0 {
        return 0.0;
    }
    let bw = entry.dtype.byte_width();

    // Start with a simple deterministic seed vector (sin-based — no RNG,
    // same across runs).
    let mut v: Vec<f64> = (0..cols).map(|i| (i as f64 * 0.123 + 0.1).sin()).collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
    for x in v.iter_mut() {
        *x /= norm;
    }

    let read_at = |r: usize, c: usize| -> f64 {
        let idx = r * cols + c;
        let off = idx * bw;
        if off + bw > entry.data.len() {
            return 0.0;
        }
        entry.dtype.to_f64(&entry.data[off..off + bw])
    };

    // Power iteration: v ← AᵀA v; renormalise.
    for _ in 0..5 {
        // u = A v
        let mut u = vec![0.0; rows];
        for r in 0..rows {
            let mut s = 0.0;
            for c in 0..cols {
                s += read_at(r, c) * v[c];
            }
            u[r] = s;
        }
        // v = Aᵀ u
        let mut nv = vec![0.0; cols];
        for c in 0..cols {
            let mut s = 0.0;
            for r in 0..rows {
                s += read_at(r, c) * u[r];
            }
            nv[c] = s;
        }
        let nrm = nv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
        for x in nv.iter_mut() {
            *x /= nrm;
        }
        v = nv;
    }
    // Rayleigh quotient: ‖A v‖² / ‖v‖²   (v is already unit-norm).
    let mut rayleigh = 0.0_f64;
    for r in 0..rows {
        let mut s = 0.0;
        for c in 0..cols {
            s += read_at(r, c) * v[c];
        }
        rayleigh += s * s;
    }
    // Frobenius² = Σ aᵢⱼ².
    let mut frob_sq = 0.0_f64;
    for r in 0..rows {
        for c in 0..cols {
            let a = read_at(r, c);
            frob_sq += a * a;
        }
    }
    if frob_sq <= 1e-30 {
        0.0
    } else {
        (rayleigh / frob_sq).clamp(0.0, 1.0)
    }
}

/// Score every attention head and FFN neuron that `spec` describes.
///
/// Every layer must have its seven weight tensors (`attn.wq`, `wk`, `wv`,
/// `wo`, `ffn.w_gate`, `w_up`, `w_down`) in NSL's layout; a missing or
/// misshapen tensor is an error, never a silent score of zero.
pub fn analyse_weight_map(
    wm: &WeightMap,
    spec: &ModelSpec,
    slacks: &RooflineSlackTable,
) -> Result<ImportanceTable, String> {
    let index = BlockIndex::build(wm)?;
    let fetch = |layer: u32, rest: &str| {
        index
            .get(wm, layer, rest)
            .ok_or_else(|| format!("CEP: the weights file has no tensor 'blocks[{layer}].{rest}'"))
    };
    let mut heads = Vec::new();
    let mut ffns = Vec::new();
    let mut layers = Vec::new();
    for layer in 0..spec.n_layers {
        let l = layer as usize;
        let slack = slacks.get(layer);
        let pos = position_factor(layer, spec.n_layers);

        let attn = AttentionWeights {
            wq: fetch(layer, "attn.wq")?,
            wk: fetch(layer, "attn.wk")?,
            wv: fetch(layer, "attn.wv")?,
            wo: fetch(layer, "attn.wo")?,
            d_model: spec.d_model,
            n_heads: spec.n_heads[l],
            n_kv_heads: spec.n_kv_heads[l],
            head_dim: spec.head_dim[l],
        };
        attn.validate()?;
        // One value per layer: the concentration of the whole query
        // projection, not of any one head's slice.
        let spectral = spectral_energy(attn.wq);
        let mut attn_score = 0.0_f64;
        for head in 0..spec.n_heads[l] {
            let mag = attn.head_magnitude(head);
            let score = mag * spectral * slack * pos;
            attn_score += score;
            heads.push(HeadImportance {
                layer,
                head,
                weight_magnitude: mag,
                spectral_energy: spectral,
                roofline_slack: slack,
                position_factor: pos,
                score,
            });
        }

        let neuron_magnitudes = ffn_neuron_magnitudes(
            fetch(layer, "ffn.w_gate")?,
            fetch(layer, "ffn.w_up")?,
            fetch(layer, "ffn.w_down")?,
            spec.d_model,
            spec.d_ff[l],
        )?;
        let mag: f64 = neuron_magnitudes.iter().sum();
        let ffn_score = mag * slack * pos;
        ffns.push(FfnImportance {
            layer,
            weight_magnitude: mag,
            position_factor: pos,
            roofline_slack: slack,
            score: ffn_score,
            neuron_magnitudes,
        });

        layers.push(LayerImportance {
            layer,
            attention_score: attn_score,
            ffn_score,
            total_score: attn_score + ffn_score,
        });
    }
    Ok(ImportanceTable {
        heads,
        ffns,
        layers,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::{WeightDType, WeightEntry};
    use std::collections::HashMap;

    fn entry(name: &str, rows: usize, cols: usize, vals: &[f32]) -> WeightEntry {
        assert_eq!(vals.len(), rows * cols);
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        WeightEntry::new(name.to_string(), data, vec![rows, cols], WeightDType::F32)
    }

    fn close(got: f64, want: f64) {
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "got {got}, want {want}"
        );
    }

    #[test]
    fn position_factor_is_u_shaped() {
        assert!((position_factor(0, 8) - 1.3).abs() < 1e-9);
        assert!((position_factor(7, 8) - 1.3).abs() < 1e-9);
        // |2·3/7 − 1| = 1/7, so 1 + 0.3/7.
        assert!((position_factor(3, 8) - (1.0 + 0.3 / 7.0)).abs() < 1e-12);
        assert!((position_factor(4, 8) - (1.0 + 0.3 / 7.0)).abs() < 1e-12);
    }

    #[test]
    fn position_factor_handles_single_layer() {
        assert_eq!(position_factor(0, 1), 1.0);
    }

    /// d_model 2, two query heads sharing one key/value head, head_dim 1.
    /// `wq` is `[d_model, n_heads·head_dim]`, so head h is COLUMN h:
    ///
    /// ```text
    /// wq = | 3  0 |    wk = | 2 |    wv = | 1 |    wo = | 0  5 |
    ///      | 4  1 |         | 0 |         | 1 |         | 6  8 |
    /// ```
    ///
    /// head 0: ‖(3,4)‖·‖(2,0)‖·‖(1,1)‖·‖(0,5)‖ = 5·2·√2·5  = 50√2
    /// head 1: ‖(0,1)‖·‖(2,0)‖·‖(1,1)‖·‖(6,8)‖ = 1·2·√2·10 = 20√2
    ///
    /// Read along the other axis (rows of `wq`, columns of `wo`) the two
    /// heads would come out as 3·…·6 and √17·…·√89: neither number above.
    #[test]
    fn head_magnitude_reads_columns_of_wq_and_rows_of_wo() {
        let wq = entry("wq", 2, 2, &[3., 0., 4., 1.]);
        let wk = entry("wk", 2, 1, &[2., 0.]);
        let wv = entry("wv", 2, 1, &[1., 1.]);
        let wo = entry("wo", 2, 2, &[0., 5., 6., 8.]);
        let attn = AttentionWeights {
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            d_model: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 1,
        };
        attn.validate().unwrap();
        close(attn.head_magnitude(0), 50.0 * 2f64.sqrt());
        close(attn.head_magnitude(1), 20.0 * 2f64.sqrt());
    }

    /// Four query heads in two key/value groups (heads 0,1 -> group 0; heads
    /// 2,3 -> group 1), head_dim 1, d_model 1, every query and output entry
    /// 1. `wk = (3, 4)` and `wv = (1, 2)`, so the magnitudes are
    /// 3·1 = 3 for heads 0 and 1 and 4·2 = 8 for heads 2 and 3. Cutting
    /// `wk`/`wv` by query head instead of by group would index past them.
    #[test]
    fn head_magnitude_uses_the_key_value_group() {
        let wq = entry("wq", 1, 4, &[1., 1., 1., 1.]);
        let wk = entry("wk", 1, 2, &[3., 4.]);
        let wv = entry("wv", 1, 2, &[1., 2.]);
        let wo = entry("wo", 4, 1, &[1., 1., 1., 1.]);
        let attn = AttentionWeights {
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            d_model: 1,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 1,
        };
        attn.validate().unwrap();
        let mags: Vec<f64> = (0..4).map(|h| attn.head_magnitude(h)).collect();
        assert_eq!(mags, vec![3.0, 3.0, 8.0, 8.0]);
    }

    /// A head whose output rows are zero writes nothing into the residual
    /// stream whatever its query, key and value weights are.
    #[test]
    fn a_head_with_a_zero_output_projection_scores_zero() {
        let wq = entry("wq", 2, 2, &[3., 9., 4., 9.]);
        let wk = entry("wk", 2, 1, &[2., 0.]);
        let wv = entry("wv", 2, 1, &[1., 1.]);
        let wo = entry("wo", 2, 2, &[0., 5., 0., 0.]);
        let attn = AttentionWeights {
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            d_model: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 1,
        };
        assert_eq!(attn.head_magnitude(1), 0.0);
        assert!(attn.head_magnitude(0) > 0.0);
    }

    #[test]
    fn a_transposed_projection_is_refused() {
        // [n_heads·head_dim, d_model] = [4, 2] where NSL stores [2, 4].
        let wq = entry("blocks[0].attn.wq", 4, 2, &[0.; 8]);
        let wk = entry("wk", 2, 2, &[0.; 4]);
        let wv = entry("wv", 2, 2, &[0.; 4]);
        let wo = entry("wo", 4, 2, &[0.; 8]);
        let attn = AttentionWeights {
            wq: &wq,
            wk: &wk,
            wv: &wv,
            wo: &wo,
            d_model: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 2,
        };
        let err = attn.validate().unwrap_err();
        assert!(
            err.contains("blocks[0].attn.wq") && err.contains("[2, 4]"),
            "{err}"
        );
    }

    /// d_model 2, d_ff 3:
    ///
    /// ```text
    /// w_gate = | 3  0  1 |   w_up = | 1  2  0 |   w_down = | 2  0 |
    ///          | 4  0  1 |          | 0  2  0 |            | 1  1 |
    ///                                                      | 0  0 |
    /// ```
    ///
    /// neuron 0: 5 · 1 · 2 = 10; neuron 1: 0 · √8 · √2 = 0 (dead gate);
    /// neuron 2: √2 · 0 · 0 = 0. The gate column alone would rank neuron 2
    /// above neuron 1.
    #[test]
    fn ffn_magnitude_is_the_product_of_all_three_matrices() {
        let gate = entry("g", 2, 3, &[3., 0., 1., 4., 0., 1.]);
        let up = entry("u", 2, 3, &[1., 2., 0., 0., 2., 0.]);
        let down = entry("d", 3, 2, &[2., 0., 1., 1., 0., 0.]);
        let mags = ffn_neuron_magnitudes(&gate, &up, &down, 2, 3).unwrap();
        close(mags[0], 10.0);
        assert_eq!(mags[1], 0.0);
        assert_eq!(mags[2], 0.0);
    }

    #[test]
    fn ffn_survivors_keep_the_largest_and_break_ties_low() {
        assert_eq!(ffn_survivors(&[1.0, 5.0, 3.0, 5.0], 2), vec![1, 3]);
        assert_eq!(ffn_survivors(&[2.0, 2.0, 2.0], 2), vec![0, 1]);
        assert_eq!(ffn_survivors(&[1.0, 2.0], 5), vec![0, 1]);
        assert_eq!(ffn_survivors(&[1.0, 2.0], 0), Vec::<u32>::new());
    }

    /// diag(2, 1, 1, 1): σ₁² / ‖A‖_F² = 4 / 7. Power iteration contracts the
    /// other directions by (1/4) per step, so five steps are ample.
    #[test]
    fn spectral_energy_of_a_known_spectrum() {
        let mut vals = vec![0f32; 16];
        for (i, d) in [2.0f32, 1.0, 1.0, 1.0].iter().enumerate() {
            vals[i * 4 + i] = *d;
        }
        let se = spectral_energy(&entry("diag", 4, 4, &vals));
        assert!(
            (se - 4.0 / 7.0).abs() < 1e-3,
            "got {se}, want {}",
            4.0 / 7.0
        );
    }

    #[test]
    fn spectral_energy_is_one_for_rank_one() {
        // Outer product of (1, 2, 3) and (1, 1): one non-zero singular value.
        let se = spectral_energy(&entry("r1", 3, 2, &[1., 1., 2., 2., 3., 3.]));
        assert!((se - 1.0).abs() < 1e-9, "got {se}");
    }

    fn spec_2x2() -> ModelSpec {
        // d_model 2, one layer, 2 query heads, 1 key/value head, head_dim 1, d_ff 3.
        let mut s = ModelSpec::uniform(2, 1, 2, 1, 1, 3, 4);
        s.dtype_bytes = 4;
        s
    }

    fn weights_2x2(bracket: bool) -> WeightMap {
        let name = |rest: &str| {
            if bracket {
                format!("blocks[0].{rest}")
            } else {
                format!("blocks.0.{rest}")
            }
        };
        let mut m = HashMap::new();
        let mut put = |rest: &str, rows, cols, vals: &[f32]| {
            let n = name(rest);
            m.insert(n.clone(), entry(&n, rows, cols, vals));
        };
        put("attn.wq", 2, 2, &[3., 0., 4., 1.]);
        put("attn.wk", 2, 1, &[2., 0.]);
        put("attn.wv", 2, 1, &[1., 1.]);
        put("attn.wo", 2, 2, &[0., 5., 6., 8.]);
        put("ffn.w_gate", 2, 3, &[3., 0., 1., 4., 0., 1.]);
        put("ffn.w_up", 2, 3, &[1., 2., 0., 0., 2., 0.]);
        put("ffn.w_down", 3, 2, &[2., 0., 1., 1., 0., 0.]);
        WeightMap::from_entries(m)
    }

    /// The table for the hand-worked weights above, under both spellings of
    /// the layer name. With neutral slack and one layer (position factor 1)
    /// a head's score is its magnitude times the layer's spectral energy.
    #[test]
    fn analyse_scores_heads_and_neurons_under_both_name_spellings() {
        for bracket in [true, false] {
            let wm = weights_2x2(bracket);
            let t = analyse_weight_map(&wm, &spec_2x2(), &RooflineSlackTable::default()).unwrap();
            assert_eq!(t.heads.len(), 2);
            close(t.heads[0].weight_magnitude, 50.0 * 2f64.sqrt());
            close(t.heads[1].weight_magnitude, 20.0 * 2f64.sqrt());
            assert_eq!(t.heads[0].spectral_energy, t.heads[1].spectral_energy);
            close(
                t.heads[0].score,
                t.heads[0].weight_magnitude * t.heads[0].spectral_energy,
            );
            assert_eq!(t.ffns.len(), 1, "the FFN must be found by NSL's names");
            assert_eq!(t.ffns[0].neuron_magnitudes.len(), 3);
            close(t.ffns[0].weight_magnitude, 10.0);
            close(t.layers[0].ffn_score, 10.0);
        }
    }

    #[test]
    fn analyse_refuses_a_missing_tensor() {
        let mut wm = weights_2x2(true);
        wm.remove("blocks[0].ffn.w_down");
        let err = analyse_weight_map(&wm, &spec_2x2(), &RooflineSlackTable::default()).unwrap_err();
        assert!(err.contains("blocks[0].ffn.w_down"), "{err}");
    }

    #[test]
    fn roofline_slack_defaults_to_one_when_missing() {
        let tbl = RooflineSlackTable::default();
        assert_eq!(tbl.get(7), 1.0);
    }

    #[test]
    fn roofline_slack_lookup_by_layer() {
        let tbl = RooflineSlackTable {
            per_layer: vec![(0, 1.5), (1, 0.8), (2, 2.0)],
        };
        assert_eq!(tbl.get(0), 1.5);
        assert_eq!(tbl.get(2), 2.0);
        assert_eq!(tbl.get(99), 1.0);
    }
}
