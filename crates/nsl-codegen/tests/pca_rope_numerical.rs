//! CFTP §4.3 — Layer 2 numerical parity for RoPE position reset.
//!
//! Validates the math of effective_pos = q_pos - doc_starts[segment_ids[q_pos]]
//! at the CPU-reference level, independent of any kernel launch. Two
//! parity tests:
//!
//!   1. Single-doc parity: with doc_starts[0]=0 and segment_ids all 0,
//!      effective_pos == q_pos for every position. The output of RoPE
//!      with the reset formula must equal the output without the reset.
//!      Bit-exact.
//!
//!   2. Three-doc parity: a packed sequence with doc_lengths=[a, b, c]
//!      must produce per-document independent RoPE rotations. The packed
//!      output's per-document slice must equal a standalone RoPE rotation
//!      of that document starting at position 0. Bit-exact (CPU only —
//!      f16 tolerance applies on GPU but this Layer 2 test is f32).

use nsl_runtime::packing::build_segment_ids_and_doc_starts;

fn cpu_reference_rope_single_doc(q: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; q.len()];
    for pos in 0..seq_len {
        for d in (0..head_dim).step_by(2) {
            let theta = pos as f32 / 10_000f32.powf(d as f32 / head_dim as f32);
            let (cos, sin) = (theta.cos(), theta.sin());
            let x0 = q[pos * head_dim + d];
            let x1 = q[pos * head_dim + d + 1];
            out[pos * head_dim + d] = x0 * cos - x1 * sin;
            out[pos * head_dim + d + 1] = x0 * sin + x1 * cos;
        }
    }
    out
}

fn cpu_reference_rope_with_doc_starts(
    q: &[f32],
    segment_ids: &[u16],
    doc_starts: &[i32],
    head_dim: usize,
) -> Vec<f32> {
    let seq_len = segment_ids.len();
    let mut out = vec![0.0f32; q.len()];
    for pos in 0..seq_len {
        let sid = segment_ids[pos] as usize;
        let effective_pos = pos as i32 - doc_starts[sid];
        for d in (0..head_dim).step_by(2) {
            let theta = effective_pos as f32 / 10_000f32.powf(d as f32 / head_dim as f32);
            let (cos, sin) = (theta.cos(), theta.sin());
            let x0 = q[pos * head_dim + d];
            let x1 = q[pos * head_dim + d + 1];
            out[pos * head_dim + d] = x0 * cos - x1 * sin;
            out[pos * head_dim + d + 1] = x0 * sin + x1 * cos;
        }
    }
    out
}

#[test]
fn single_doc_with_doc_starts_matches_no_reset_path_bit_exact() {
    let seq_len = 16;
    let head_dim = 8;
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();

    let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&[seq_len as u32]);
    assert_eq!(doc_starts[0], 0);
    assert!(segment_ids.iter().all(|&s| s == 0));

    let no_reset = cpu_reference_rope_single_doc(&q, seq_len, head_dim);
    let reset = cpu_reference_rope_with_doc_starts(&q, &segment_ids, &doc_starts, head_dim);

    assert_eq!(reset, no_reset, "single-doc reset must be bit-exact identity");
}

#[test]
fn three_doc_packed_matches_per_doc_reference() {
    let head_dim = 8;
    let doc_lengths = vec![5u32, 3, 4];
    let total: u32 = doc_lengths.iter().sum();
    let q: Vec<f32> = (0..total as usize * head_dim).map(|i| (i as f32) * 0.01).collect();
    let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&doc_lengths);

    let packed_out = cpu_reference_rope_with_doc_starts(&q, &segment_ids, &doc_starts, head_dim);

    // Per-doc reference: split q by document, rotate each independently
    // starting from position 0, then concatenate.
    let mut reference: Vec<f32> = Vec::with_capacity(q.len());
    let mut cursor = 0usize;
    for &dlen in &doc_lengths {
        let doc_q = &q[cursor * head_dim..(cursor + dlen as usize) * head_dim];
        let rotated = cpu_reference_rope_single_doc(doc_q, dlen as usize, head_dim);
        reference.extend_from_slice(&rotated);
        cursor += dlen as usize;
    }

    assert_eq!(packed_out, reference);
}

#[test]
fn per_row_doc_starts_multi_batch_cpu_parity() {
    // Verify the per-row doc_starts layout that T2.6 introduced gives
    // independent rotations for each batch row.
    use nsl_runtime::packing::MAX_NUM_DOCS;
    let head_dim = 8;

    // Row 0 has 2 docs of length [3, 3]; row 1 has 2 docs of length [5, 1].
    // Each row's doc_starts table is independent.
    let row0_lengths = vec![3u32, 3];
    let row1_lengths = vec![5u32, 1];

    let (segs0, doc_starts0) = build_segment_ids_and_doc_starts(&row0_lengths);
    let (segs1, doc_starts1) = build_segment_ids_and_doc_starts(&row1_lengths);

    // Both rows have seq_len=6 (= sum of their own doc_lengths).
    assert_eq!(segs0.len(), 6);
    assert_eq!(segs1.len(), 6);
    // Row 0's doc_starts: [0, 3, 6, -1, ...]
    assert_eq!(doc_starts0[0], 0);
    assert_eq!(doc_starts0[1], 3);
    assert_eq!(doc_starts0[2], 6);
    assert_eq!(doc_starts0[3], -1);
    // Row 1's doc_starts: [0, 5, 6, -1, ...]
    assert_eq!(doc_starts1[0], 0);
    assert_eq!(doc_starts1[1], 5);
    assert_eq!(doc_starts1[2], 6);
    assert_eq!(doc_starts1[3], -1);

    // Each row's rotation matches its own per-doc reference.
    let q0: Vec<f32> = (0..6 * head_dim).map(|i| (i as f32) * 0.01).collect();
    let q1: Vec<f32> = (100..100 + 6 * head_dim).map(|i| (i as f32) * 0.01).collect();

    let row0_out = cpu_reference_rope_with_doc_starts(&q0, &segs0, &doc_starts0, head_dim);
    let row1_out = cpu_reference_rope_with_doc_starts(&q1, &segs1, &doc_starts1, head_dim);

    // Reference: each row is two independent docs starting at pos 0.
    let mut row0_ref = Vec::new();
    row0_ref.extend(cpu_reference_rope_single_doc(&q0[0..3 * head_dim], 3, head_dim));
    row0_ref.extend(cpu_reference_rope_single_doc(&q0[3 * head_dim..6 * head_dim], 3, head_dim));
    let mut row1_ref = Vec::new();
    row1_ref.extend(cpu_reference_rope_single_doc(&q1[0..5 * head_dim], 5, head_dim));
    row1_ref.extend(cpu_reference_rope_single_doc(&q1[5 * head_dim..6 * head_dim], 1, head_dim));

    assert_eq!(row0_out, row0_ref);
    assert_eq!(row1_out, row1_ref);

    // MAX_NUM_DOCS reachable through re-export.
    let _ = MAX_NUM_DOCS;
}
