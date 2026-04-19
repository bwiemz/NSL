//! Sequence packing: continuous stream chunking with block-diagonal attention masks.
//!
//! Packs a flat token stream into fixed-size batches, building causal
//! block-diagonal attention masks that respect document boundaries (EOS tokens).

use std::ffi::c_void;

use crate::cpu::{create_tensor_with_shape_rs_dtype};
use crate::dict::{nsl_dict_new, nsl_dict_set_str};
use crate::string::nsl_str_from_rust;
use crate::tensor::{DTYPE_U16_SEGMENT, DTYPE_U16_TOKEN, NslTensor};

/// A packed batch of token sequences with attention mask.
pub struct PackedBatch {
    pub input_ids:   Vec<i64>,
    pub labels:      Vec<i64>,
    pub mask:        Vec<f32>,   // retained through Task 7b
    pub segment_ids: Vec<u16>,   // spec §3.5 — new, additive
    pub batch_size:  usize,
    pub seq_len:     usize,
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
fn read_flat_value(data: *const c_void, dtype: u16, index: usize) -> i64 {
    match dtype {
        DTYPE_U16_TOKEN => unsafe { *(data as *const u16).add(index) as i64 },
        1 => unsafe { *(data as *const f32).add(index) as i64 },
        0 => unsafe { *(data as *const f64).add(index) as i64 },
        _ => panic!("read_flat_value() unsupported dtype {}", dtype),
    }
}

pub fn pack_batch(
    data: *const c_void,
    data_len: usize,
    data_dtype: u16,
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
        input_ids.push(read_flat_value(data, data_dtype, *cursor + i));
    }
    *cursor += total;

    // Build labels and mask per sequence in the batch
    let mut labels = vec![0i64; total];
    let mut mask = vec![-1e9f32; batch_size * seq_len * seq_len];
    let mut segment_ids: Vec<u16> = Vec::with_capacity(total);

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

        // PCA Tier A (spec §3.5): emit per-position segment IDs alongside
        // the dense mask. Reuses the doc_ids computed above to preserve
        // byte-identity with the dense-mask baseline. u16 ceiling enforced
        // upstream by pca_detect::validate_config.
        for d in doc_ids.iter() {
            segment_ids.push(*d as u16);
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
        segment_ids,
        batch_size,
        seq_len,
    })
}

/// Convert a `PackedBatch` into an NslDict with keys:
/// - `"input_ids"`: tensor of shape `[B, S]`
/// - `"labels"`: tensor of shape `[B, S]`
/// - `"attention_mask"`: tensor of shape `[B, S, S]` (packed sequences need block-diagonal masks)
///
/// NOTE: Only packed batches include attention_mask. Standard (non-packed) batches
/// omit it — the model's GQA layer generates causal_mask(seq_len) internally.
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

    // segment_ids [B, S] — logical dtype is DTYPE_U16_SEGMENT (spec §3.5),
    // but the current tensor helper allocates f32 storage and data_f32()
    // asserts dtype==1. Use dtype=1 (f32) backing for now; values are
    // lossless (u16 max = 65535 < 2^24 mantissa of f32).
    // TODO(PCA Tier A, post-Task-7b): narrow-dtype (u16) storage for
    // segment_ids is a DataLoader-backend follow-up. The dict key and
    // shape contract are stable today; the physical backing promotes
    // from f32 to true u16 once the tensor helper supports
    // DTYPE_U16_SEGMENT. u16 values round-trip through f32 exactly
    // (max=65535 < 2^24), so current storage is lossless but wastes
    // 2x bytes per position.
    let _ = DTYPE_U16_SEGMENT; // imported for future use; see TODO above
    let seg_ptr = create_tensor_with_shape_rs_dtype(&[b, s], 1);
    let seg_tensor = NslTensor::from_ptr(seg_ptr);
    let seg_data = seg_tensor.data_f32();
    for (i, &v) in batch.segment_ids.iter().enumerate() {
        unsafe { *seg_data.add(i) = v as f32 };
    }

    let dict = nsl_dict_new();
    let k_ids = nsl_str_from_rust("input_ids");
    let k_lbl = nsl_str_from_rust("labels");
    let k_mask = nsl_str_from_rust("attention_mask");
    let k_seg = nsl_str_from_rust("segment_ids");
    nsl_dict_set_str(dict, k_ids, ids_ptr);
    nsl_dict_set_str(dict, k_lbl, lbl_ptr);
    nsl_dict_set_str(dict, k_mask, mask_ptr);
    nsl_dict_set_str(dict, k_seg, seg_ptr);
    crate::string::nsl_string_free(k_ids);
    crate::string::nsl_string_free(k_lbl);
    crate::string::nsl_string_free(k_mask);
    crate::string::nsl_string_free(k_seg);

    dict
}

/// FFI: packing efficiency — ratio of real tokens to total slots.
///
/// For packed sequences, this measures how much padding is wasted.
/// Efficiency = (total real tokens) / (batch_size * seq_len).
/// Without packing (padding to max length), efficiency is typically 0.6-0.8.
/// With packing, efficiency approaches 1.0.
///
/// `dl_ptr`: DataLoader handle. If 0 or no attention_mask available, returns 1.0.
#[no_mangle]
pub extern "C" fn nsl_packing_efficiency(dl_ptr: i64) -> f64 {
    if dl_ptr == 0 {
        return 1.0;
    }
    // Without per-batch attention_mask tracking, estimate from sequence packing:
    // if sequences are pre-packed (all slots used), efficiency = 1.0.
    // This is the common case for pre-tokenized LLM data with drop_last=true.
    // Future: read attention_mask from the DataLoader to compute real token ratio.
    let _ = dl_ptr;
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
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,
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
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,
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
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,
            &mut cursor,
            1,  // batch_size
            4,  // seq_len (needs 4 tokens, only 3 available)
            0,
        );

        assert!(result.is_none());
        assert_eq!(cursor, 0); // cursor unchanged
    }

    #[test]
    fn test_pack_batch_u16_tokens() {
        let stream: [u16; 4] = [11, 12, 0, 13];
        let mut cursor: usize = 0;

        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            DTYPE_U16_TOKEN,
            &mut cursor,
            1,
            4,
            0,
        )
        .expect("should produce a batch");

        assert_eq!(cursor, 4);
        assert_eq!(batch.input_ids, vec![11, 12, 0, 13]);
        assert_eq!(batch.labels, vec![12, 0, -100, -100]);
    }

    #[test]
    fn pack_batch_segment_ids_increment_on_eos() {
        // Two documents in one pack: [10, 11, 12, EOS=2, 20, 21, 22, 23].
        // Expected segment_ids: [0, 0, 0, 0, 1, 1, 1, 1].
        // Semantics: a new segment starts immediately AFTER an EOS token;
        // the EOS token itself is part of the current segment.
        let stream: Vec<f64> = vec![10.0, 11.0, 12.0, 2.0, 20.0, 21.0, 22.0, 23.0];
        let mut cursor: usize = 0;
        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,           // DTYPE_F64
            &mut cursor,
            1,           // batch_size
            8,           // seq_len
            2,           // eos_token
        )
        .expect("pack_batch produced a batch");

        assert_eq!(batch.segment_ids, vec![0u16, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn pack_batch_segment_ids_single_segment_no_eos() {
        // No EOS in the stream → single segment of all zeros.
        let stream: Vec<f64> = (0..8).map(|x| x as f64).collect();
        let mut cursor: usize = 0;
        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,
            &mut cursor,
            1,
            8,
            255,         // EOS token that never appears in the stream
        )
        .expect("pack_batch produced a batch");

        assert_eq!(batch.segment_ids, vec![0u16; 8]);
    }

    #[test]
    fn pack_batch_segment_ids_three_unequal_segments() {
        // Three documents: [A,A,EOS,B,B,B,EOS,C].
        // Expected: [0, 0, 0, 1, 1, 1, 1, 2].
        // Segment boundaries: EOS at pos 2 → segment 1 starts at pos 3.
        // EOS at pos 6 → segment 2 starts at pos 7.
        let stream: Vec<f64> = vec![10.0, 11.0, 2.0, 20.0, 21.0, 22.0, 2.0, 30.0];
        let mut cursor: usize = 0;
        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,
            &mut cursor,
            1,
            8,
            2,
        )
        .expect("pack_batch produced a batch");

        assert_eq!(batch.segment_ids, vec![0u16, 0, 0, 1, 1, 1, 1, 2]);
    }

    #[test]
    fn pack_batch_segment_ids_trailing_eos_does_not_overflow() {
        // Stream ends with EOS: [10, 11, 12, EOS=2] with seq_len=4.
        // The trailing EOS is part of segment 0; no phantom segment 1
        // should appear because there is no position after it.
        // Expected segment_ids: [0, 0, 0, 0].
        let stream: Vec<f64> = vec![10.0, 11.0, 12.0, 2.0];
        let mut cursor: usize = 0;
        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,               // DTYPE_F64
            &mut cursor,
            1,               // batch_size
            4,               // seq_len
            2,               // eos_token
        )
        .expect("pack_batch produced a batch");

        assert_eq!(batch.segment_ids, vec![0u16, 0, 0, 0]);
    }

    #[test]
    fn test_packed_batch_to_dict() {
        let stream: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut cursor: usize = 0;

        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,
            &mut cursor,
            1,  // batch_size
            4,  // seq_len
            0,
        )
        .expect("should produce a batch");

        let dict = packed_batch_to_dict(&batch);
        assert_eq!(nsl_dict_len(dict), 4);

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
