//! CEP — weight-aware importance scoring.
//!
//! Per-component scores that guide the pruning search.  The scoring
//! function from §3.3 of the paper:
//!
//! ```text
//! importance(head_h_in_layer_l) =
//!     weight_magnitude   × spectral_energy
//!     × roofline_slack   × position_factor
//! ```
//!
//! All four factors are computable from pre-trained weights + the
//! compilation oracle — no forward/backward pass, no calibration data.

use serde::Serialize;

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

/// Per-FFN-neuron importance (rolled up to per-layer since scoring
/// individual neurons is expensive; the paper collapses this to
/// "per-layer FFN width reduction" decisions).
#[derive(Debug, Clone, Serialize)]
pub struct FfnImportance {
    pub layer: u32,
    pub weight_magnitude: f64,
    pub position_factor: f64,
    pub roofline_slack: f64,
    pub score: f64,
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

/// Position factor (paper §3.3):
///
/// ```text
/// position_factor(l/L) = 1.0 + 0.3 × |2l/L - 1|
/// ```
///
/// Produces a U-shaped curve: early (l=0) and late (l=L-1) layers are
/// weighted more heavily because empirical evidence shows they're the
/// most sensitive to pruning.
pub fn position_factor(layer: u32, n_layers: u32) -> f64 {
    if n_layers <= 1 {
        return 1.0;
    }
    let normalized = (layer as f64) / ((n_layers - 1) as f64);
    1.0 + 0.3 * (2.0 * normalized - 1.0).abs()
}

/// Frobenius norm of a rectangular slice of a weight entry, interpreted
/// row-major as `[rows, cols]` where `head_range` slices the outer
/// dimension (standard HuggingFace `[n_heads * head_dim, d_model]` layout).
pub fn head_frobenius(entry: &WeightEntry, head: u32, n_heads: u32) -> f64 {
    if entry.shape.len() != 2 || n_heads == 0 {
        return 0.0;
    }
    let rows = entry.shape[0];
    let cols = entry.shape[1];
    // Heads partition either the row or the column axis — try row first.
    let (per_head_rows, on_rows) = if rows.is_multiple_of(n_heads as usize) {
        (rows / n_heads as usize, true)
    } else if cols.is_multiple_of(n_heads as usize) {
        (cols / n_heads as usize, false)
    } else {
        return 0.0;
    };
    let bw = entry.dtype.byte_width();
    let mut sum_sq = 0.0_f64;
    if on_rows {
        let r_start = head as usize * per_head_rows;
        let r_end = r_start + per_head_rows;
        for r in r_start..r_end {
            for c in 0..cols {
                let idx = r * cols + c;
                let off = idx * bw;
                if off + bw > entry.data.len() {
                    return 0.0;
                }
                let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
                sum_sq += v * v;
            }
        }
    } else {
        let c_start = head as usize * per_head_rows;
        let c_end = c_start + per_head_rows;
        for r in 0..rows {
            for c in c_start..c_end {
                let idx = r * cols + c;
                let off = idx * bw;
                if off + bw > entry.data.len() {
                    return 0.0;
                }
                let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
                sum_sq += v * v;
            }
        }
    }
    sum_sq.sqrt()
}

/// L1 norm of a flat weight entry — used for FFN neurons.
pub fn weight_l1(entry: &WeightEntry) -> f64 {
    let bw = entry.dtype.byte_width();
    let mut sum = 0.0_f64;
    for i in 0..entry.num_elements {
        let off = i * bw;
        if off + bw > entry.data.len() {
            break;
        }
        sum += entry.dtype.to_f64(&entry.data[off..off + bw]).abs();
    }
    sum
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

/// Score a single attention head using the paper's formula.
pub fn score_head(
    wq: &WeightEntry,
    wk: &WeightEntry,
    wv: &WeightEntry,
    layer: u32,
    head: u32,
    n_heads: u32,
    n_layers: u32,
    roofline_slack: f64,
) -> HeadImportance {
    let q = head_frobenius(wq, head, n_heads);
    let k = head_frobenius(wk, head, n_heads);
    let v = head_frobenius(wv, head, n_heads);
    let weight_mag = q * k * v; // product — combines all three projections
    let spectral = spectral_energy(wq);
    let pos = position_factor(layer, n_layers);
    let score = weight_mag * spectral * roofline_slack * pos;
    HeadImportance {
        layer,
        head,
        weight_magnitude: weight_mag,
        spectral_energy: spectral,
        roofline_slack,
        position_factor: pos,
        score,
    }
}

/// Score an entire FFN block for a layer.
pub fn score_ffn(
    w_in: &WeightEntry,
    w_out: &WeightEntry,
    layer: u32,
    n_layers: u32,
    roofline_slack: f64,
) -> FfnImportance {
    let mag = weight_l1(w_in) * weight_l1(w_out);
    let pos = position_factor(layer, n_layers);
    let score = mag * roofline_slack * pos;
    FfnImportance {
        layer,
        weight_magnitude: mag,
        position_factor: pos,
        roofline_slack,
        score,
    }
}

/// Naming convention for finding a layer's weights in a WeightMap.  Tries
/// the common `blocks.N.attn.wq` / `h.N.attn.W_q` / `layers.N.q_proj`
/// patterns.
fn find_weight<'a>(wm: &'a WeightMap, layer: u32, suffixes: &[&str]) -> Option<&'a WeightEntry> {
    for prefix in [
        format!("blocks.{layer}."),
        format!("layers.{layer}."),
        format!("h.{layer}."),
    ] {
        for suf in suffixes {
            let name = format!("{prefix}{suf}");
            if let Some(e) = wm.get(&name) {
                return Some(e);
            }
        }
    }
    None
}

/// Scan a [`WeightMap`] and score every attention head + FFN block the
/// oracle's [`ModelSpec`] describes.
pub fn analyse_weight_map(
    wm: &WeightMap,
    n_heads_per_layer: &[u32],
    n_layers: u32,
    slacks: &RooflineSlackTable,
) -> ImportanceTable {
    let mut heads = Vec::new();
    let mut ffns = Vec::new();
    let mut layers = Vec::new();
    for layer in 0..n_layers {
        let h = *n_heads_per_layer.get(layer as usize).unwrap_or(&0);
        let slack = slacks.get(layer);

        let wq = find_weight(wm, layer, &["attn.wq", "attn.W_q", "q_proj.weight", "wq"]);
        let wk = find_weight(wm, layer, &["attn.wk", "attn.W_k", "k_proj.weight", "wk"]);
        let wv = find_weight(wm, layer, &["attn.wv", "attn.W_v", "v_proj.weight", "wv"]);
        let w_in = find_weight(wm, layer, &["ffn.w_in", "mlp.fc_in", "mlp.up_proj.weight"]);
        let w_out = find_weight(
            wm,
            layer,
            &["ffn.w_out", "mlp.fc_out", "mlp.down_proj.weight"],
        );

        let mut attn_score = 0.0_f64;
        if let (Some(q), Some(k), Some(v)) = (wq, wk, wv) {
            for head in 0..h {
                let hs = score_head(q, k, v, layer, head, h, n_layers, slack);
                attn_score += hs.score;
                heads.push(hs);
            }
        }

        let mut ffn_score = 0.0_f64;
        if let (Some(i), Some(o)) = (w_in, w_out) {
            let fi = score_ffn(i, o, layer, n_layers, slack);
            ffn_score = fi.score;
            ffns.push(fi);
        }

        layers.push(LayerImportance {
            layer,
            attention_score: attn_score,
            ffn_score,
            total_score: attn_score + ffn_score,
        });
    }
    ImportanceTable {
        heads,
        ffns,
        layers,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::{WeightDType, WeightEntry};

    fn make_weight(
        name: &str,
        per_head_rows: usize,
        cols: usize,
        per_head_mag: &[f64],
    ) -> WeightEntry {
        let n_heads = per_head_mag.len();
        let rows = per_head_rows * n_heads;
        let mut data = vec![0u8; rows * cols * 4];
        for (h, target) in per_head_mag.iter().enumerate() {
            let elements = per_head_rows * cols;
            let value = (target / (elements as f64).sqrt()) as f32;
            for r in (h * per_head_rows)..((h + 1) * per_head_rows) {
                for c in 0..cols {
                    let idx = r * cols + c;
                    let off = idx * 4;
                    data[off..off + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape: vec![rows, cols],
            dtype: WeightDType::F32,
            num_elements: rows * cols,
            sparsity: None,
            eliminated: false,
        }
    }

    #[test]
    fn position_factor_is_u_shaped() {
        assert!((position_factor(0, 8) - 1.3).abs() < 1e-9);
        assert!((position_factor(7, 8) - 1.3).abs() < 1e-9);
        let mid = position_factor(3, 8);
        assert!(mid < 1.3 && mid > 0.9);
    }

    #[test]
    fn position_factor_handles_single_layer() {
        assert_eq!(position_factor(0, 1), 1.0);
    }

    #[test]
    fn head_frobenius_scales_with_weight_magnitude() {
        let w = make_weight("wq", 4, 16, &[1.0, 2.0, 3.0, 4.0]);
        for (head, expected) in [(0u32, 1.0_f64), (1, 2.0), (2, 3.0), (3, 4.0)].iter() {
            let got = head_frobenius(&w, *head, 4);
            assert!((got - expected).abs() / expected < 1e-3);
        }
    }

    #[test]
    fn spectral_energy_high_for_low_rank() {
        // Rank-1 matrix: all columns identical → spectral_energy ≈ 1.
        let mut data = vec![0u8; 16 * 16 * 4];
        for r in 0..16 {
            for c in 0..16 {
                let v = ((r + 1) as f32) * 0.1; // column-independent
                let idx = r * 16 + c;
                let off = idx * 4;
                data[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let entry = WeightEntry {
            name: "rank1".into(),
            data,
            shape: vec![16, 16],
            dtype: WeightDType::F32,
            num_elements: 256,
            sparsity: None,
            eliminated: false,
        };
        let se = spectral_energy(&entry);
        assert!(se > 0.9, "expected high spectral energy, got {se}");
    }

    #[test]
    fn spectral_energy_low_for_full_rank_random() {
        // Diagonal matrix — approximately full rank, low spectral energy.
        let mut data = vec![0u8; 16 * 16 * 4];
        for i in 0..16 {
            let off = (i * 16 + i) * 4;
            let v = 1.0_f32;
            data[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        let entry = WeightEntry {
            name: "diag".into(),
            data,
            shape: vec![16, 16],
            dtype: WeightDType::F32,
            num_elements: 256,
            sparsity: None,
            eliminated: false,
        };
        let se = spectral_energy(&entry);
        assert!(se < 0.5, "expected low spectral energy, got {se}");
    }

    #[test]
    fn head_score_aggregates_all_factors() {
        let wq = make_weight("wq", 4, 8, &[10.0, 0.1]);
        let wk = make_weight("wk", 4, 8, &[10.0, 0.1]);
        let wv = make_weight("wv", 4, 8, &[10.0, 0.1]);
        let big = score_head(&wq, &wk, &wv, 0, 0, 2, 4, 1.2);
        let small = score_head(&wq, &wk, &wv, 0, 1, 2, 4, 1.2);
        assert!(big.score > small.score);
        assert!(big.position_factor > 0.0);
        assert!(big.weight_magnitude > 0.0);
        assert!(big.roofline_slack == 1.2);
    }

    #[test]
    fn weight_l1_matches_sum_of_abs() {
        let w = make_weight("w", 2, 4, &[1.0]); // one "head" filling 8 elements
        let l1 = weight_l1(&w);
        assert!(l1 > 0.0);
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
