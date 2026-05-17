//! Sequence packing: continuous stream chunking with block-diagonal attention masks.
//!
//! Packs a flat token stream into fixed-size batches, building causal
//! block-diagonal attention masks that respect document boundaries (EOS tokens).

use std::ffi::c_void;

use crate::cpu::{create_tensor_with_shape_rs_dtype};
use crate::dict::{nsl_dict_new, nsl_dict_set_str};
use crate::string::nsl_str_from_rust;
use crate::tensor::{DTYPE_U16_SEGMENT, DTYPE_U16_TOKEN, NslTensor};

/// Maximum number of documents per packed batch.
///
/// This constant is the producer-side bound for the PCA §4.3 RoPE position
/// reset path. The `doc_starts` tensor emitted alongside `segment_ids` is
/// padded to `[MAX_NUM_DOCS + 1]` with sentinel `-1` for unused slots so
/// the GPU kernel can emit a constant-size SMEM load independent of the
/// per-batch document count.
///
/// **Source of truth:** `crates/nsl-codegen/src/pca_rope.rs::MAX_NUM_DOCS`
/// (declared there as `u32 = 256`). `nsl-runtime` cannot depend on
/// `nsl-codegen` directly (would introduce a circular crate dependency),
/// so the bound is duplicated here as `usize`. The two declarations MUST
/// agree numerically; a divergence is a producer/consumer ABI bug. T1's
/// const_assert on the codegen side keeps `(MAX_NUM_DOCS + 1) * 4` within
/// the per-CTA SMEM budget, so changing the value here without also
/// updating the codegen side will silently break the kernel.
pub const MAX_NUM_DOCS: usize = 256;

/// A packed batch of token sequences with attention mask.
pub struct PackedBatch {
    pub input_ids:   Vec<i64>,
    pub labels:      Vec<i64>,
    pub mask:        Vec<f32>,   // retained through Task 7b
    pub segment_ids: Vec<u16>,   // spec §3.5 — new, additive
    /// PCA §4.3 RoPE position-reset table, sibling to `segment_ids`.
    ///
    /// **Per-row layout** (spec v3, commit `2b8df4e6`): flat length
    /// `batch_size * (MAX_NUM_DOCS + 1)`, indexed as
    /// `doc_starts[r * 257 + k]` = start position of document `k` in batch
    /// row `r`, **relative to that row's local position 0** (not the flat
    /// batch_size * seq_len stream). Each row's 257-element subtable is
    /// independently accumulated during the EOS scan; unused slots are
    /// sentinel `-1`.
    ///
    /// The per-row shape is load-bearing for `batch_size > 1`: Tier A's
    /// `segment_ids` resets to 0 per batch row, so the kernel must index
    /// `doc_starts[batch_idx * 257 + segment_ids[idx]]` (handled in T6
    /// CTA prologue). A flat single-table layout would collapse all rows'
    /// document boundaries into row 0's positions.
    pub doc_starts:  Vec<i32>,
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

    // PCA §4.3 RoPE position-reset accumulator (spec v3, per-row layout).
    //
    // Flat shape `batch_size * (MAX_NUM_DOCS + 1)`, indexed as
    // `doc_starts[r * (MAX_NUM_DOCS+1) + k]` = start of document k in row r,
    // *relative to row r's local position 0*. Each row's 257-element
    // subtable is accumulated independently with its own cursor; sentinel
    // `-1` for unused slots so the kernel emits a constant-size SMEM load.
    //
    // Why per-row: Tier A's `segment_ids` resets to 0 at the start of each
    // batch row, so kernel indexing `doc_starts[batch_idx * 257 +
    // segment_ids[idx]]` requires each row to carry its own boundaries
    // expressed in row-local coordinates.
    let row_stride = MAX_NUM_DOCS + 1;
    let mut doc_starts = vec![-1i32; batch_size * row_stride];

    for b in 0..batch_size {
        let offset = b * seq_len;
        let row_base = b * row_stride;

        // Each new batch row opens doc 0 at local position 0; per-row cursor
        // resets here so row r's subtable accumulates independently of row r-1.
        doc_starts[row_base] = 0;
        let mut current_row_doc_idx: usize = 1;

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
                // PCA §4.3 per-row accumulator: EOS at local position i-1
                // ends the current doc; the next doc starts at local
                // position i (row-local, not flat). Write into this row's
                // 257-element subtable at offset row_base + current_row_doc_idx.
                // MAX_NUM_DOCS is enforced *per row* — exceeding it would
                // overflow this row's constant-size SMEM load on the kernel side.
                assert!(
                    current_row_doc_idx <= MAX_NUM_DOCS,
                    "num_docs > MAX_NUM_DOCS {} in batch row {}: too many EOS-delimited documents",
                    MAX_NUM_DOCS,
                    b
                );
                doc_starts[row_base + current_row_doc_idx] = i as i32;
                current_row_doc_idx += 1;
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

    // NOTE: doc_starts is now [batch_size, MAX_NUM_DOCS+1] flat (spec v3).
    // Each row's 257-element subtable stores ROW-LOCAL doc start positions
    // (relative to that row's local position 0, in [0, seq_len)), so kernel
    // indexing `doc_starts[batch_idx * 257 + segment_ids[idx]]` aligns with
    // Tier A's segment_ids that reset per batch row. The offline builder
    // `build_segment_ids_and_doc_starts` still uses flat-sequence semantics
    // for the single-row case it serves (see its docstring).
    Some(PackedBatch {
        input_ids,
        labels,
        mask,
        segment_ids,
        doc_starts,
        batch_size,
        seq_len,
    })
}

/// Construct segment_ids + doc_starts tensors for a packed batch with the
/// given per-document lengths.
///
/// **Single-row scope** (spec v3): this builder is the offline T2 path that
/// pre-computes per-doc lengths and emits the *flat single-row* layout
/// `doc_starts: [MAX_NUM_DOCS + 1]` with absolute positions. It is intended
/// for `batch_size == 1` callers that need to construct test fixtures or
/// drive a single-row consumer. The streaming `pack_batch` is the
/// multi-batch source of truth and emits per-row `[batch_size,
/// MAX_NUM_DOCS+1]` layout — do not mix outputs across the two paths.
///
/// Returns `(segment_ids, doc_starts)` where:
/// - `segment_ids` has length `doc_lengths.iter().sum()` (the packed length)
///   and contains the segment ID (`0..doc_lengths.len()`) for each position.
/// - `doc_starts` has fixed length `MAX_NUM_DOCS + 1 = 257` with valid
///   prefix-sum entries for slots `0..=doc_lengths.len()` and sentinel `-1`
///   for unused slots. The fixed length lets the GPU kernel emit a
///   constant-size SMEM load.
///
/// # Panics
///
/// Panics if `doc_lengths.len() > MAX_NUM_DOCS` (256). The panic message
/// starts with `"num_docs"` so callers can match it via
/// `#[should_panic(expected = "num_docs")]`.
pub fn build_segment_ids_and_doc_starts(doc_lengths: &[u32]) -> (Vec<u16>, Vec<i32>) {
    assert!(
        doc_lengths.len() <= MAX_NUM_DOCS,
        "num_docs {} > MAX_NUM_DOCS {}",
        doc_lengths.len(),
        MAX_NUM_DOCS
    );
    let total: u32 = doc_lengths.iter().sum();
    let mut segment_ids: Vec<u16> = Vec::with_capacity(total as usize);
    let mut doc_starts = vec![-1i32; MAX_NUM_DOCS + 1];
    let mut cursor: u32 = 0;
    doc_starts[0] = 0;
    for (k, &len) in doc_lengths.iter().enumerate() {
        for _ in 0..len {
            segment_ids.push(k as u16);
        }
        cursor += len;
        doc_starts[k + 1] = cursor as i32;
    }
    (segment_ids, doc_starts)
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

    // doc_starts [batch_size, MAX_NUM_DOCS + 1] — sibling to segment_ids
    // (PCA §4.3 RoPE position-reset path, per-row layout per spec v3).
    // Logical dtype is i32 (codegen-side `pca_rope::doc_starts_enabled`
    // consumes an i32-typed device tensor), but the current tensor helper
    // allocates f32 storage and data_f32() asserts dtype==1. We use f32
    // backing matching segment_ids' existing narrow-dtype TODO above;
    // i32 values in our range [−1, seq_len) round-trip through f32 exactly
    // (per-row local positions are bounded by seq_len, well below 2^24
    // mantissa).
    // TODO(PCA Tier A, post-Task-7b): promote to true i32 storage once the
    // tensor helper supports narrow integer dtypes; sibling to the
    // segment_ids TODO above (both unblock together).
    let ds_ptr = create_tensor_with_shape_rs_dtype(&[b, (MAX_NUM_DOCS + 1) as i64], 1);
    let ds_tensor = NslTensor::from_ptr(ds_ptr);
    let ds_data = ds_tensor.data_f32();
    for (i, &v) in batch.doc_starts.iter().enumerate() {
        unsafe { *ds_data.add(i) = v as f32 };
    }

    let dict = nsl_dict_new();
    let k_ids = nsl_str_from_rust("input_ids");
    let k_lbl = nsl_str_from_rust("labels");
    let k_mask = nsl_str_from_rust("attention_mask");
    let k_seg = nsl_str_from_rust("segment_ids");
    let k_ds = nsl_str_from_rust("doc_starts");
    nsl_dict_set_str(dict, k_ids, ids_ptr);
    nsl_dict_set_str(dict, k_lbl, lbl_ptr);
    nsl_dict_set_str(dict, k_mask, mask_ptr);
    nsl_dict_set_str(dict, k_seg, seg_ptr);
    nsl_dict_set_str(dict, k_ds, ds_ptr);
    crate::string::nsl_string_free(k_ids);
    crate::string::nsl_string_free(k_lbl);
    crate::string::nsl_string_free(k_mask);
    crate::string::nsl_string_free(k_seg);
    crate::string::nsl_string_free(k_ds);

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
    fn pack_emits_doc_starts_with_sentinel_padding() {
        // 3 documents of length [3, 2, 4] => doc_starts = [0, 3, 5, 9, -1, -1, ...]
        let doc_lengths: Vec<u32> = vec![3, 2, 4];
        let packed_len: u32 = doc_lengths.iter().sum();
        let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&doc_lengths);

        assert_eq!(segment_ids.len(), packed_len as usize);
        // first doc spans positions 0..3 with segment_id 0
        assert_eq!(&segment_ids[0..3], &[0u16, 0, 0]);
        // second doc spans positions 3..5 with segment_id 1
        assert_eq!(&segment_ids[3..5], &[1u16, 1]);
        // third doc spans positions 5..9 with segment_id 2
        assert_eq!(&segment_ids[5..9], &[2u16, 2, 2, 2]);

        // doc_starts fixed length MAX_NUM_DOCS+1 = 257
        assert_eq!(doc_starts.len(), 257);
        // valid slots
        assert_eq!(doc_starts[0], 0);
        assert_eq!(doc_starts[1], 3);
        assert_eq!(doc_starts[2], 5);
        assert_eq!(doc_starts[3], 9);
        // sentinel slots
        assert_eq!(doc_starts[4], -1);
        assert_eq!(doc_starts[256], -1);
    }

    #[test]
    #[should_panic(expected = "num_docs")]
    fn pack_rejects_too_many_docs() {
        // 257 single-token docs exceeds MAX_NUM_DOCS=256
        let doc_lengths: Vec<u32> = vec![1; 257];
        let _ = build_segment_ids_and_doc_starts(&doc_lengths);
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
        // input_ids, labels, attention_mask, segment_ids, doc_starts
        assert_eq!(nsl_dict_len(dict), 5);

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

    #[test]
    fn pack_batch_emits_doc_starts_during_eos_scan() {
        // T2.5/T2.6 — single-row pipeline test: pack_batch accumulates doc_starts
        // as a side effect of its EOS scan, in the per-row layout spec v3
        // expects (`[batch_size, MAX_NUM_DOCS+1]` flat; single-row batch ⇒
        // total length 1 * 257 = 257, all entries live at row 0).
        //
        // Stream lays out three EOS-delimited documents of lengths 3, 2, 4:
        //   doc A: [10, 11, EOS=2]                  (local positions 0..3)
        //   doc B: [20, EOS=2]                      (local positions 3..5)
        //   doc C: [30, 31, 32, 33]                 (local positions 5..9)
        // No trailing EOS — last segment runs to the end of the pack.
        let stream: Vec<f64> = vec![
            10.0, 11.0, 2.0,       // doc A — len 3, ends with EOS at pos 2
            20.0, 2.0,             // doc B — len 2, ends with EOS at pos 4
            30.0, 31.0, 32.0, 33.0 // doc C — len 4, no trailing EOS
        ];
        let mut cursor: usize = 0;
        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,                // DTYPE_F64
            &mut cursor,
            1,                // batch_size — single-row case
            9,                // seq_len
            2,                // eos_token
        )
        .expect("pack_batch produced a batch");

        // Segment IDs sanity-check: doc A spans [0..3], doc B [3..5], doc C [5..9].
        // EOS at pos 2 is part of doc 0; the next position (3) opens doc 1.
        // EOS at pos 4 is part of doc 1; the next position (5) opens doc 2.
        assert_eq!(
            batch.segment_ids,
            vec![0u16, 0, 0, 1, 1, 2, 2, 2, 2],
            "segment_ids precondition for the doc_starts shape this test asserts"
        );

        // Per-row sibling table per spec v3: batch_size * (MAX_NUM_DOCS+1).
        // For batch_size=1 the total length is 1 * 257 = 257.
        assert_eq!(batch.doc_starts.len(), 1 * (MAX_NUM_DOCS + 1));
        assert_eq!(batch.doc_starts.len(), 257);

        // Row 0 subtable: cumulative starts of docs 0, 1, 2 (row-local positions).
        // For batch_size=1 row-local positions coincide with the absolute pack
        // positions, so 9 == 3 + 2 + 4 still falls out cleanly.
        assert_eq!(batch.doc_starts[0], 0,  "row 0 doc 0 begins at local position 0");
        assert_eq!(batch.doc_starts[1], 3,  "row 0 doc 1 begins after doc A (len 3) including its EOS");
        assert_eq!(batch.doc_starts[2], 5,  "row 0 doc 2 begins after doc A + doc B (len 3+2)");

        // Slots 3..=256 are unused — sentinel -1 lets the kernel emit a
        // constant-size SMEM load independent of runtime num_docs.
        assert_eq!(batch.doc_starts[3],   -1, "first unused slot");
        assert_eq!(batch.doc_starts[100], -1);
        assert_eq!(batch.doc_starts[256], -1, "last slot");

        // End-to-end: packed_batch_to_dict must insert doc_starts into the
        // batch dict (key = "doc_starts") using the same allocation primitive
        // as segment_ids, with per-row shape [batch_size, MAX_NUM_DOCS+1].
        let dict = packed_batch_to_dict(&batch);
        assert_eq!(
            nsl_dict_len(dict),
            5,
            "expected input_ids/labels/attention_mask/segment_ids/doc_starts"
        );

        let k_ds = nsl_str_from_rust("doc_starts");
        let ds_ptr = crate::dict::nsl_dict_get_str(dict, k_ds);
        crate::string::nsl_string_free(k_ds);
        assert!(ds_ptr != 0, "doc_starts must be present in the batch dict");

        let ds_tensor = NslTensor::from_ptr(ds_ptr);
        assert_eq!(ds_tensor.ndim, 2, "doc_starts is 2-D [batch_size, MAX_NUM_DOCS+1]");
        unsafe {
            assert_eq!(*ds_tensor.shape.add(0), 1, "single-row batch");
            assert_eq!(*ds_tensor.shape.add(1), (MAX_NUM_DOCS + 1) as i64);
        }
        let ds_data = ds_tensor.data_f32();
        unsafe {
            assert_eq!(*ds_data.add(0), 0.0);
            assert_eq!(*ds_data.add(1), 3.0);
            assert_eq!(*ds_data.add(2), 5.0);
            assert_eq!(*ds_data.add(3), -1.0);
            assert_eq!(*ds_data.add(256), -1.0);
        }
    }

    #[test]
    fn pack_batch_doc_starts_no_eos_produces_single_doc() {
        // Pack with no EOS in the stream — only doc 0 exists, so doc_starts
        // is [0, -1, -1, ..., -1].
        let stream: Vec<f64> = (0..8).map(|x| x as f64).collect();
        let mut cursor: usize = 0;
        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,
            &mut cursor,
            1,
            8,
            255, // EOS token not present in stream
        )
        .expect("pack_batch produced a batch");

        // Per-row layout: batch_size=1, so length is exactly 257.
        assert_eq!(batch.doc_starts.len(), 257);
        assert_eq!(batch.doc_starts[0], 0);
        assert_eq!(batch.doc_starts[1], -1);
        assert_eq!(batch.doc_starts[256], -1);
    }

    #[test]
    fn pack_batch_doc_starts_per_row_layout_batch2() {
        // T2.6 — multi-batch correctness gap from T2.5.
        //
        // batch_size=2 with rows that have *different* doc layouts:
        //   row 0: [10, 11, EOS=2, 20, 21, 22]    — two docs: lengths 3, 3
        //   row 1: [30, 31, 32, 33, EOS=2, 40]    — two docs: lengths 5, 1
        //
        // Tier A's segment_ids resets to 0 per batch row, so each row needs
        // its own doc_starts subtable indexed *relative to that row's local
        // position 0*. The kernel reads doc_starts[row * 257 + segment_ids[idx]].
        //
        // Expected per-row subtables:
        //   row 0 (offset 0):   [0, 3, -1, -1, ..., -1]
        //   row 1 (offset 257): [0, 5, -1, -1, ..., -1]
        //
        // The independence is the load-bearing assertion: row 1's doc 1 starts
        // at local position 5, NOT at flat position 6 + 5 = 11. The pre-T2.6
        // implementation would have written 9 (= flat offset 6 + local 3) into
        // doc_starts[1] from row 0 *only* and left row 1's entries unset.
        let stream: Vec<f64> = vec![
            // row 0 — two docs: [10, 11, EOS=2] then [20, 21, 22]
            10.0, 11.0, 2.0, 20.0, 21.0, 22.0,
            // row 1 — two docs: [30, 31, 32, 33, EOS=2] then [40]
            30.0, 31.0, 32.0, 33.0, 2.0, 40.0,
        ];
        let mut cursor: usize = 0;
        let batch = pack_batch(
            stream.as_ptr() as *const c_void,
            stream.len(),
            0,         // DTYPE_F64
            &mut cursor,
            2,         // batch_size — the point of this test
            6,         // seq_len
            2,         // eos_token
        )
        .expect("pack_batch produced a batch");

        // Sanity: segment_ids resets per row (Tier A invariant).
        // Row 0: EOS at local pos 2 → segment_ids = [0,0,0,1,1,1]
        // Row 1: EOS at local pos 4 → segment_ids = [0,0,0,0,0,1]
        assert_eq!(
            batch.segment_ids,
            vec![0u16, 0, 0, 1, 1, 1,  0, 0, 0, 0, 0, 1],
            "segment_ids must reset to 0 at each new batch row"
        );

        // Per-row doc_starts layout: [batch_size, MAX_NUM_DOCS+1] flat.
        assert_eq!(batch.doc_starts.len(), 2 * (MAX_NUM_DOCS + 1));
        assert_eq!(batch.doc_starts.len(), 2 * 257);

        // Row 0 subtable lives at offset 0..257.
        assert_eq!(batch.doc_starts[0 * 257 + 0], 0, "row 0 doc 0 starts at local pos 0");
        assert_eq!(batch.doc_starts[0 * 257 + 1], 3, "row 0 doc 1 starts at local pos 3 (after EOS)");
        assert_eq!(batch.doc_starts[0 * 257 + 2], -1, "row 0 has only 2 docs; slot 2 is sentinel");
        assert_eq!(batch.doc_starts[0 * 257 + 256], -1, "row 0 last sentinel");

        // Row 1 subtable lives at offset 257..514 — *independent* of row 0.
        // Note row 1 doc 1 starts at LOCAL pos 5, not flat pos 11.
        assert_eq!(batch.doc_starts[1 * 257 + 0], 0, "row 1 doc 0 starts at local pos 0 (row-local!)");
        assert_eq!(batch.doc_starts[1 * 257 + 1], 5, "row 1 doc 1 starts at local pos 5 (row-local!)");
        assert_eq!(batch.doc_starts[1 * 257 + 2], -1, "row 1 has only 2 docs; slot 2 is sentinel");
        assert_eq!(batch.doc_starts[1 * 257 + 256], -1, "row 1 last sentinel");

        // End-to-end: packed_batch_to_dict emits a [batch_size, MAX_NUM_DOCS+1]
        // tensor, not [MAX_NUM_DOCS+1].
        let dict = packed_batch_to_dict(&batch);
        let k_ds = nsl_str_from_rust("doc_starts");
        let ds_ptr = crate::dict::nsl_dict_get_str(dict, k_ds);
        crate::string::nsl_string_free(k_ds);
        let ds_tensor = NslTensor::from_ptr(ds_ptr);
        assert_eq!(ds_tensor.ndim, 2, "doc_starts is now 2-D [batch_size, MAX_NUM_DOCS+1]");
        unsafe {
            assert_eq!(*ds_tensor.shape.add(0), 2, "batch_size dim");
            assert_eq!(*ds_tensor.shape.add(1), (MAX_NUM_DOCS + 1) as i64, "per-row table dim");
        }
        let ds_data = ds_tensor.data_f32();
        unsafe {
            // Row 0
            assert_eq!(*ds_data.add(0 * 257 + 0), 0.0);
            assert_eq!(*ds_data.add(0 * 257 + 1), 3.0);
            assert_eq!(*ds_data.add(0 * 257 + 2), -1.0);
            // Row 1
            assert_eq!(*ds_data.add(1 * 257 + 0), 0.0);
            assert_eq!(*ds_data.add(1 * 257 + 1), 5.0);
            assert_eq!(*ds_data.add(1 * 257 + 2), -1.0);
        }
    }
}
