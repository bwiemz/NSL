//! CEP Option 2 — SP1: weight slicing (pure data transform).

use crate::weight_aware::WeightEntry;

#[derive(Debug, Clone, PartialEq)]
pub enum CepSliceError {
    HeadIndexOutOfRange { layer: u32, head: u32, n_heads: u32 },
    ShapeMismatch { tensor: String, expected: String, found: String },
    MissingTensor { tensor: String },
    InconsistentFfn { layer: u32, detail: String },
    UnitOutOfRange { tensor: String, unit: u32, n_units: u32 },
}

impl std::fmt::Display for CepSliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CepSliceError::HeadIndexOutOfRange { layer, head, n_heads } =>
                write!(f, "CEP slice: layer {layer} head {head} out of range (n_heads={n_heads})"),
            CepSliceError::ShapeMismatch { tensor, expected, found } =>
                write!(f, "CEP slice: '{tensor}' shape mismatch\n  expected: {expected}\n  found:    {found}"),
            CepSliceError::MissingTensor { tensor } =>
                write!(f, "CEP slice: expected tensor '{tensor}' not found in weights"),
            CepSliceError::InconsistentFfn { layer, detail } =>
                write!(f, "CEP slice: layer {layer} FFN inconsistency: {detail}"),
            CepSliceError::UnitOutOfRange { tensor, unit, n_units } =>
                write!(f, "CEP slice: '{tensor}' survivor unit {unit} out of range (n_units={n_units})"),
        }
    }
}

/// Gather survivor blocks along `axis` (0 = rows, 1 = cols) of a row-major 2-D tensor.
/// A block is `block_width` consecutive rows/cols; `survivor_units` are 0-based block
/// indices (**strictly ascending** (enforced)). Returns a new contiguous WeightEntry; dtype-agnostic (byte copy).
pub fn slice_blocks(
    entry: &WeightEntry,
    axis: usize,
    n_units: u32,
    block_width: u32,
    survivor_units: &[u32],
) -> Result<WeightEntry, CepSliceError> {
    if entry.shape.len() != 2 {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: "2-D tensor".into(),
            found: format!("{:?}", entry.shape),
        });
    }
    let rows = entry.shape[0];
    let cols = entry.shape[1];
    let bw = entry.dtype.byte_width();
    // Data must be exactly rows*cols*bw bytes, else the byte gather below would index-panic.
    let expected_bytes = rows * cols * bw;
    if entry.data.len() != expected_bytes {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: format!("{expected_bytes} bytes ({rows}x{cols}x{bw})"),
            found: format!("{} bytes", entry.data.len()),
        });
    }
    // survivor_units must be strictly ascending: the gather preserves their order, and a
    // non-ascending (or duplicated) list would silently reorder/duplicate weight blocks.
    if survivor_units.windows(2).any(|w| w[0] >= w[1]) {
        return Err(CepSliceError::ShapeMismatch {
            tensor: entry.name.clone(),
            expected: "strictly ascending survivor_units".into(),
            found: format!("{survivor_units:?}"),
        });
    }
    let block = block_width as usize;
    for &u in survivor_units {
        if u >= n_units {
            return Err(CepSliceError::UnitOutOfRange { tensor: entry.name.clone(), unit: u, n_units });
        }
    }
    let n_surv = survivor_units.len();
    let mut out = Vec::new();
    let (new_rows, new_cols) = match axis {
        0 => {
            // Keep survivor row-blocks; each row is `cols * bw` bytes.
            out.reserve(survivor_units.len() * block * cols * bw);
            let row_bytes = cols * bw;
            for &u in survivor_units {
                let start = (u as usize) * block;
                for r in start..start + block {
                    let off = r * row_bytes;
                    out.extend_from_slice(&entry.data[off..off + row_bytes]);
                }
            }
            (n_surv * block, cols)
        }
        1 => {
            // Keep survivor col-blocks; gather per row.
            out.reserve(rows * survivor_units.len() * block * bw);
            for r in 0..rows {
                let row_off = r * cols * bw;
                for &u in survivor_units {
                    let c0 = (u as usize) * block;
                    let off = row_off + c0 * bw;
                    out.extend_from_slice(&entry.data[off..off + block * bw]);
                }
            }
            (rows, n_surv * block)
        }
        _ => {
            return Err(CepSliceError::ShapeMismatch {
                tensor: entry.name.clone(),
                expected: "axis 0 or 1".into(),
                found: format!("axis {axis}"),
            })
        }
    };
    let num_elements = new_rows * new_cols;
    Ok(WeightEntry {
        name: entry.name.clone(),
        data: out,
        shape: vec![new_rows, new_cols],
        dtype: entry.dtype,
        num_elements,
        sparsity: None,
        eliminated: false,
    })
}

/// Rank d_ff neurons by column L2 norm of w_gate (`[d_model, d_ff]`, neurons on axis-1);
/// return the top `new_d_ff` survivor indices in ascending order. dtype-agnostic via to_f64.
pub fn ffn_survivors_by_magnitude(
    w_gate: &WeightEntry,
    d_ff: u32,
    new_d_ff: u32,
) -> Result<Vec<u32>, CepSliceError> {
    if w_gate.shape.len() != 2 || w_gate.shape[1] != d_ff as usize {
        return Err(CepSliceError::ShapeMismatch {
            tensor: w_gate.name.clone(),
            expected: format!("[d_model, {d_ff}]"),
            found: format!("{:?}", w_gate.shape),
        });
    }
    let rows = w_gate.shape[0];
    let cols = d_ff as usize;
    let bw = w_gate.dtype.byte_width();
    let keep = (new_d_ff as usize).min(cols);

    let mut norms: Vec<(u32, f64)> = (0..cols)
        .map(|c| {
            let mut sumsq = 0.0_f64;
            for r in 0..rows {
                let off = (r * cols + c) * bw;
                let v = w_gate.dtype.to_f64(&w_gate.data[off..off + bw]);
                sumsq += v * v;
            }
            (c as u32, sumsq) // sqrt is monotonic; ranking on sumsq is equivalent
        })
        .collect();
    // Highest magnitude first; tie-break by lower index for determinism.
    norms.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
    });
    let mut survivors: Vec<u32> = norms.into_iter().take(keep).map(|(c, _)| c).collect();
    survivors.sort_unstable();
    Ok(survivors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::{WeightDType, WeightEntry};

    fn f32_entry(name: &str, shape: Vec<usize>, vals: &[f32]) -> WeightEntry {
        let mut data = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape,
            dtype: WeightDType::F32,
            num_elements: vals.len(),
            sparsity: None,
            eliminated: false,
        }
    }

    fn read_f32(e: &WeightEntry) -> Vec<f32> {
        e.data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    }

    #[test]
    fn slice_blocks_axis1_keeps_survivor_columns_by_identity() {
        // 2x4 row-major: rows [0,1,2,3] and [10,11,12,13]. block_width=1, keep cols {0,2,3}.
        let e = f32_entry("t", vec![2, 4], &[0., 1., 2., 3., 10., 11., 12., 13.]);
        let out = slice_blocks(&e, 1, 4, 1, &[0, 2, 3]).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(read_f32(&out), vec![0., 2., 3., 10., 12., 13.]);
    }

    #[test]
    fn slice_blocks_axis1_keeps_survivor_head_blocks() {
        // 1x4, block_width=2 (2 "heads" of width 2). Keep head 1 -> cols {2,3}.
        let e = f32_entry("t", vec![1, 4], &[0., 1., 2., 3.]);
        let out = slice_blocks(&e, 1, 2, 2, &[1]).unwrap();
        assert_eq!(out.shape, vec![1, 2]);
        assert_eq!(read_f32(&out), vec![2., 3.]);
    }

    #[test]
    fn slice_blocks_axis0_keeps_survivor_rows() {
        // 4x2 row-major. block_width=1, keep rows {1,3}.
        let e = f32_entry("t", vec![4, 2], &[0., 1., 2., 3., 4., 5., 6., 7.]);
        let out = slice_blocks(&e, 0, 4, 1, &[1, 3]).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert_eq!(read_f32(&out), vec![2., 3., 6., 7.]);
    }

    #[test]
    fn slice_blocks_rejects_out_of_range_unit() {
        let e = f32_entry("t", vec![2, 4], &[0.; 8]);
        assert!(slice_blocks(&e, 1, 4, 1, &[4]).is_err());
    }

    #[test]
    fn slice_blocks_rejects_non_ascending_units() {
        let e = f32_entry("t", vec![2, 4], &[0.; 8]);
        assert!(slice_blocks(&e, 1, 4, 1, &[2, 0]).is_err());
        // duplicates are also rejected (not strictly ascending)
        assert!(slice_blocks(&e, 1, 4, 1, &[1, 1]).is_err());
    }

    #[test]
    fn slice_blocks_rejects_truncated_data() {
        // shape claims 2x4 (32 bytes) but only 6 floats (24 bytes) of data.
        let e = f32_entry("t", vec![2, 4], &[0., 1., 2., 3., 4., 5.]);
        assert!(slice_blocks(&e, 1, 4, 1, &[0, 1]).is_err());
    }

    #[test]
    fn ffn_survivors_keep_highest_magnitude_columns() {
        // w_gate [2, 4]: column L2 norms ~ [sqrt(0+0), sqrt(1+1), sqrt(9+9), sqrt(4+4)]
        // = col0=0, col1=1.41, col2=4.24, col3=2.83. Keep top 2 -> cols {2, 3} (ascending).
        let w_gate = f32_entry("ffn.w_gate", vec![2, 4], &[
            0., 1., 3., 2.,
            0., 1., 3., 2.,
        ]);
        let surv = ffn_survivors_by_magnitude(&w_gate, 4, 2).unwrap();
        assert_eq!(surv, vec![2, 3]);
    }

    #[test]
    fn ffn_survivors_full_width_is_identity() {
        let w_gate = f32_entry("ffn.w_gate", vec![2, 3], &[1., 2., 3., 4., 5., 6.]);
        assert_eq!(ffn_survivors_by_magnitude(&w_gate, 3, 3).unwrap(), vec![0, 1, 2]);
    }
}
