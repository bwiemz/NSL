//! Sequence packing: continuous stream chunking with block-diagonal attention masks.
//!
//! Packs a flat token stream into fixed-size batches, building causal
//! block-diagonal attention masks that respect document boundaries (EOS tokens).

use crate::cpu::{create_tensor_with_shape_rs_dtype};
use crate::dict::{nsl_dict_new, nsl_dict_set_str};
use crate::string::nsl_str_from_rust;
use crate::tensor::NslTensor;

/// A packed batch of token sequences with attention mask.
pub struct PackedBatch {
    pub input_ids: Vec<i64>,
    pub labels: Vec<i64>,
    pub mask: Vec<f32>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Pack one batch from a continuous token stream.
///
/// Reads `batch_size * seq_len` tokens starting at `*cursor`, builds shifted
/// labels (with -100 at EOS and final positions), and a block-diagonal causal
/// attention mask that prevents cross-document attention.
///
/// Returns `None` when not enough tokens remain (epoch end).
// Safety: caller must ensure `data` points to a valid array of at least `data_len` elements.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn pack_batch(
    data: *const f64,
    data_len: usize,
    cursor: &mut usize,
    batch_size: usize,
    seq_len: usize,
    eos_token: i64,
) -> Option<PackedBatch> {
    let total = batch_size * seq_len;
    if *cursor + total > data_len {
        return None;
    }

    // Read tokens from stream
    let mut input_ids = Vec::with_capacity(total);
    for i in 0..total {
        let val = unsafe { *data.add(*cursor + i) } as i64;
        input_ids.push(val);
    }
    *cursor += total;

    // Build labels and mask per sequence in the batch
    let mut labels = vec![0i64; total];
    let mut mask = vec![-1e9f32; batch_size * seq_len * seq_len];

    for b in 0..batch_size {
        let offset = b * seq_len;

        // Labels: shifted by 1 within each sequence
        if seq_len > 0 {
            for i in 0..seq_len - 1 {
                labels[offset + i] = input_ids[offset + i + 1];
            }
            // Last position always gets -100
            labels[offset + seq_len - 1] = -100;
        }

        // Set -100 at EOS positions in labels
        for i in 0..seq_len {
            if input_ids[offset + i] == eos_token {
                labels[offset + i] = -100;
            }
        }

        // Assign doc_ids by scanning for EOS tokens
        let mut doc_ids = vec![0u32; seq_len];
        let mut current_doc: u32 = 0;
        for i in 0..seq_len {
            if i > 0 && input_ids[offset + i - 1] == eos_token {
                current_doc += 1;
            }
            doc_ids[i] = current_doc;
        }

        // Build block-diagonal causal mask for this sequence
        // mask[b][i][j] = 0.0 if doc_id[i] == doc_id[j] AND j <= i (same doc, causal)
        let mask_offset = b * seq_len * seq_len;
        for i in 0..seq_len {
            for j in 0..seq_len {
                if doc_ids[i] == doc_ids[j] && j <= i {
                    mask[mask_offset + i * seq_len + j] = 0.0;
                }
            }
        }
    }

    Some(PackedBatch {
        input_ids,
        labels,
        mask,
        batch_size,
        seq_len,
    })
}

/// Convert a `PackedBatch` into an NslDict with keys:
/// - `"input_ids"`: tensor of shape `[B, S]`
/// - `"labels"`: tensor of shape `[B, S]`
/// - `"attention_mask"`: tensor of shape `[B, S, S]`
pub fn packed_batch_to_dict(batch: &PackedBatch) -> i64 {
    let b = batch.batch_size as i64;
    let s = batch.seq_len as i64;

    // input_ids [B, S]
    let ids_ptr = create_tensor_with_shape_rs_dtype(&[b, s], 1);
    let ids_tensor = NslTensor::from_ptr(ids_ptr);
    let ids_data = ids_tensor.data_f32();
    for (i, &v) in batch.input_ids.iter().enumerate() {
        unsafe { *ids_data.add(i) = v as f32 };
    }

    // labels [B, S]
    let lbl_ptr = create_tensor_with_shape_rs_dtype(&[b, s], 1);
    let lbl_tensor = NslTensor::from_ptr(lbl_ptr);
    let lbl_data = lbl_tensor.data_f32();
    for (i, &v) in batch.labels.iter().enumerate() {
        unsafe { *lbl_data.add(i) = v as f32 };
    }

    // attention_mask [B, S, S]
    let mask_ptr = create_tensor_with_shape_rs_dtype(&[b, s, s], 1);
    let mask_tensor = NslTensor::from_ptr(mask_ptr);
    let mask_data = mask_tensor.data_f32();
    for (i, &v) in batch.mask.iter().enumerate() {
        unsafe { *mask_data.add(i) = v };
    }

    let dict = nsl_dict_new();
    let k_ids = nsl_str_from_rust("input_ids");
    let k_lbl = nsl_str_from_rust("labels");
    let k_mask = nsl_str_from_rust("attention_mask");
    nsl_dict_set_str(dict, k_ids, ids_ptr);
    nsl_dict_set_str(dict, k_lbl, lbl_ptr);
    nsl_dict_set_str(dict, k_mask, mask_ptr);

    dict
}

/// FFI: packing efficiency stub (always returns 1.0 for now).
#[no_mangle]
pub extern "C" fn nsl_packing_efficiency(_dl_ptr: i64) -> f64 {
    1.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::nsl_dict_len;

    #[test]
    fn test_pack_batch_basic() {
        // Stream: [10, 20, EOS=0, 30, 40, 50, EOS=0, 60]
        let stream: Vec<f64> = vec![10.0, 20.0, 0.0, 30.0, 40.0, 50.0, 0.0, 60.0];
        let mut cursor: usize = 0;

        let batch = pack_batch(
            stream.as_ptr(),
            stream.len(),
            &mut cursor,
            2,  // batch_size
            4,  // seq_len
            0,  // eos_token
        )
        .expect("should produce a batch");

        assert_eq!(cursor, 8);
        assert_eq!(batch.input_ids.len(), 8);
        assert_eq!(batch.labels.len(), 8);

        // Sequence 0: [10, 20, 0, 30] -> labels [20, 0, -100, ?]
        // EOS at index 2 -> labels[2] = -100
        // Last position labels[3] = -100
        assert_eq!(batch.labels[2], -100); // EOS position
        assert_eq!(batch.labels[3], -100); // last position

        // Sequence 1: [40, 50, 0, 60] -> labels [50, 0, -100, -100]
        assert_eq!(batch.labels[6], -100); // EOS position
        assert_eq!(batch.labels[7], -100); // last position
    }

    #[test]
    fn test_pack_batch_mask() {
        // Stream: [1, 2, EOS=0, 3], batch_size=1, seq_len=4
        let stream: Vec<f64> = vec![1.0, 2.0, 0.0, 3.0];
        let mut cursor: usize = 0;

        let batch = pack_batch(
            stream.as_ptr(),
            stream.len(),
            &mut cursor,
            1,  // batch_size
            4,  // seq_len
            0,  // eos_token
        )
        .expect("should produce a batch");

        // Doc 0: tokens at indices 0, 1, 2 (EOS at 2)
        // Doc 1: token at index 3
        // mask is [4][4], flattened row-major

        // mask[0][0] = 0.0 (same doc, causal: j<=i)
        assert_eq!(batch.mask[0], 0.0);

        // mask[1][0] = 0.0, mask[1][1] = 0.0 (same doc, causal)
        assert_eq!(batch.mask[4], 0.0);
        assert_eq!(batch.mask[4 + 1], 0.0);

        // mask[3][0] = -1e9, mask[3][1] = -1e9, mask[3][2] = -1e9 (cross-doc)
        assert_eq!(batch.mask[3 * 4], -1e9);
        assert_eq!(batch.mask[3 * 4 + 1], -1e9);
        assert_eq!(batch.mask[3 * 4 + 2], -1e9);

        // mask[3][3] = 0.0 (same doc, causal)
        assert_eq!(batch.mask[3 * 4 + 3], 0.0);
    }

    #[test]
    fn test_pack_batch_epoch_end() {
        let stream: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut cursor: usize = 0;

        let result = pack_batch(
            stream.as_ptr(),
            stream.len(),
            &mut cursor,
            1,  // batch_size
            4,  // seq_len (needs 4 tokens, only 3 available)
            0,
        );

        assert!(result.is_none());
        assert_eq!(cursor, 0); // cursor unchanged
    }

    #[test]
    fn test_packed_batch_to_dict() {
        let stream: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut cursor: usize = 0;

        let batch = pack_batch(
            stream.as_ptr(),
            stream.len(),
            &mut cursor,
            1,  // batch_size
            4,  // seq_len
            0,
        )
        .expect("should produce a batch");

        let dict = packed_batch_to_dict(&batch);
        assert_eq!(nsl_dict_len(dict), 3);

        // Verify input_ids tensor shape
        let k = nsl_str_from_rust("input_ids");
        let tensor_ptr = crate::dict::nsl_dict_get_str(dict, k);
        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(tensor.ndim, 2);
        unsafe {
            assert_eq!(*tensor.shape.add(0), 1); // batch_size
            assert_eq!(*tensor.shape.add(1), 4); // seq_len
        }
    }
}
