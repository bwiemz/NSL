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
/// indices (ascending). Returns a new contiguous WeightEntry; dtype-agnostic (byte copy).
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
}
