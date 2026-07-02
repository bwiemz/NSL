use crate::context_parallel::partition::{gather_sequence, partition_sequence};
use crate::context_parallel::softmax::partial_softmax;

/// Naive full attention: softmax(Q @ K^T / sqrt(d)) @ V
/// Q,K,V: flat [seq_len, dim]
pub fn naive_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, dim: usize, causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * dim];

    for i in 0..seq_len {
        let mut scores = vec![f32::NEG_INFINITY; seq_len];
        let k_max = if causal { i + 1 } else { seq_len };
        for j in 0..k_max {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q[i * dim + d] * k[j * dim + d];
            }
            scores[j] = dot * scale;
        }

        let probs = crate::context_parallel::softmax::softmax(&scores[..k_max]);

        for d in 0..dim {
            let mut sum = 0.0f32;
            for j in 0..k_max {
                sum += probs[j] * v[j * dim + d];
            }
            out[i * dim + d] = sum;
        }
    }
    out
}

/// Ring attention CPU fallback: computes attention across multiple K/V chunks.
///
/// `q_local`: [local_seq_len, dim] — this GPU's queries
/// `k_chunks`: ordered K chunks from ring passes (local first, then remote)
/// `v_chunks`: corresponding V chunks
/// `chunk_global_starts`: global token offset for each chunk (for causal masking)
/// `q_global_start`: global token offset for this GPU's queries
/// Returns: [local_seq_len, dim] attention output
#[allow(clippy::too_many_arguments)] // mirrors FFI signature shape
pub fn ring_attention_cpu(
    q_local: &[f32],
    k_chunks: &[&[f32]],
    v_chunks: &[&[f32]],
    local_seq_len: usize,
    dim: usize,
    causal: bool,
    chunk_global_starts: &[usize],
    q_global_start: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; local_seq_len * dim];

    for qi in 0..local_seq_len {
        let q_global = q_global_start + qi;
        let mut global_max = f32::NEG_INFINITY;
        let mut global_sum = 0.0f32;
        let mut o_acc = vec![0.0f32; dim];

        for (pass, (k_chunk, v_chunk)) in k_chunks.iter().zip(v_chunks.iter()).enumerate() {
            let chunk_len = k_chunk.len() / dim;
            let kv_global_start = chunk_global_starts[pass];

            // Compute scores Q[qi] @ K_chunk^T with causal masking
            let mut scores = Vec::with_capacity(chunk_len);
            let mut has_valid = false;
            for kj in 0..chunk_len {
                let kv_global = kv_global_start + kj;
                if causal && kv_global > q_global {
                    scores.push(f32::NEG_INFINITY);
                } else {
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        dot += q_local[qi * dim + d] * k_chunk[kj * dim + d];
                    }
                    scores.push(dot * scale);
                    has_valid = true;
                }
            }

            // Skip chunk entirely if all positions are masked
            if !has_valid {
                continue;
            }

            // Partial softmax for this chunk
            let (chunk_max, chunk_sum, chunk_weights) = partial_softmax(&scores);

            // Online correction: rescale accumulated output
            let new_max = global_max.max(chunk_max);
            let old_correction = (global_max - new_max).exp();
            let new_correction = (chunk_max - new_max).exp();

            // Rescale accumulated output
            for val in o_acc.iter_mut() {
                *val *= old_correction;
            }
            global_sum = global_sum * old_correction + chunk_sum * new_correction;
            global_max = new_max;

            // Accumulate: O_acc += softmax_weights @ V_chunk
            for kj in 0..chunk_len {
                let w = chunk_weights[kj] * new_correction;
                for d in 0..dim {
                    o_acc[d] += w * v_chunk[kj * dim + d];
                }
            }
        }

        // Normalize — guard against zero sum (fully-masked queries in causal ring attention)
        if global_sum > 0.0 {
            for d in 0..dim {
                out[qi * dim + d] = o_acc[d] / global_sum;
            }
        }
        // else: out stays 0.0, which is correct for fully-masked query positions
    }
    out
}

/// M34 v1 single-node composer — end-to-end ring attention over one process.
///
/// Partitions Q, K, V by `ring_size`, runs `ring_attention_cpu` on each rank as
/// if it lived on its own GPU (with K/V chunks visited in ring order), then
/// gathers the per-rank outputs back into a `[seq_len, dim]` buffer.
///
/// Bit-identical to `naive_attention` up to floating-point non-associativity in
/// the online-softmax accumulation (see the tests below — tolerance 1e-4). The
/// point of running this on a single process is to exercise the partition +
/// online-softmax + gather composition end-to-end; the multi-device
/// distribution that would give this a wall-clock benefit is deferred until
/// the runtime gains real send/recv (NCCL, IPC, etc.). The insertion point is
/// inside the per-pass loop below — currently every K/V chunk is read from
/// this-process-local `partition_sequence` output; distribution would replace
/// each non-local read with a send-forward-recv-backward exchange via
/// `ring::ring_next` / `ring::ring_prev`.
pub fn run_ring_attention_full(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    dim: usize,
    ring_size: usize,
    causal: bool,
) -> Vec<f32> {
    assert!(ring_size >= 1, "ring_size must be >= 1");
    if ring_size == 1 {
        // Degenerate: one rank, one chunk, no ring. Naive attention.
        return naive_attention(q, k, v, seq_len, dim, causal);
    }
    assert_eq!(
        seq_len % ring_size, 0,
        "seq_len ({seq_len}) must be divisible by ring_size ({ring_size})",
    );
    let local_seq = seq_len / ring_size;

    let k_chunks: Vec<Vec<f32>> = (0..ring_size)
        .map(|r| partition_sequence(k, seq_len, dim, ring_size, r))
        .collect();
    let v_chunks: Vec<Vec<f32>> = (0..ring_size)
        .map(|r| partition_sequence(v, seq_len, dim, ring_size, r))
        .collect();

    let mut rank_outputs: Vec<Vec<f32>> = Vec::with_capacity(ring_size);
    for rank in 0..ring_size {
        let q_local = partition_sequence(q, seq_len, dim, ring_size, rank);
        // Ring visitation order: pass p pulls K/V from rank (rank + ring_size - p) % ring_size.
        // This matches types::RingConfig::source_rank.
        let mut k_refs: Vec<&[f32]> = Vec::with_capacity(ring_size);
        let mut v_refs: Vec<&[f32]> = Vec::with_capacity(ring_size);
        let mut chunk_starts: Vec<usize> = Vec::with_capacity(ring_size);
        for pass in 0..ring_size {
            let src = (rank + ring_size - pass) % ring_size;
            k_refs.push(&k_chunks[src]);
            v_refs.push(&v_chunks[src]);
            chunk_starts.push(src * local_seq);
        }
        let out = ring_attention_cpu(
            &q_local, &k_refs, &v_refs, local_seq, dim, causal, &chunk_starts, rank * local_seq,
        );
        rank_outputs.push(out);
    }
    gather_sequence(&rank_outputs, seq_len, dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_parallel::partition;

    #[test]
    fn test_ring_attention_matches_standard_2gpu() {
        // Q = K = V = [[1,0],[0,1],[1,1],[0,0]] (4 tokens, dim=2)
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = q.clone();
        let seq_len = 4;
        let dim = 2;
        let local_seq = seq_len / 2;

        let standard_out = naive_attention(&q, &k, &v, seq_len, dim, false);

        let q0 = partition::partition_sequence(&q, seq_len, dim, 2, 0);
        let k0 = partition::partition_sequence(&k, seq_len, dim, 2, 0);
        let v0 = partition::partition_sequence(&v, seq_len, dim, 2, 0);
        let k1 = partition::partition_sequence(&k, seq_len, dim, 2, 1);
        let v1 = partition::partition_sequence(&v, seq_len, dim, 2, 1);

        // GPU 0: chunks at global offsets [0, 2]
        let out0 = ring_attention_cpu(
            &q0, &[&k0, &k1], &[&v0, &v1], local_seq, dim, false, &[0, 2], 0,
        );

        let q1 = partition::partition_sequence(&q, seq_len, dim, 2, 1);
        // GPU 1: chunks at global offsets [2, 0]
        let out1 = ring_attention_cpu(
            &q1, &[&k1, &k0], &[&v1, &v0], local_seq, dim, false, &[2, 0], 2,
        );

        let ring_out = partition::gather_sequence(&[out0, out1], seq_len, dim);

        for (i, (a, b)) in standard_out.iter().zip(ring_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch at index {}: standard={}, ring={}", i, a, b
            );
        }
    }

    #[test]
    fn test_ring_attention_causal_matches_standard() {
        // 4 tokens, dim=2, causal=true, 2 GPUs
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5];
        let k = q.clone();
        let v = q.clone();
        let seq_len = 4;
        let dim = 2;
        let local_seq = seq_len / 2;

        let standard_out = naive_attention(&q, &k, &v, seq_len, dim, true);

        let q0 = partition::partition_sequence(&q, seq_len, dim, 2, 0);
        let k0 = partition::partition_sequence(&k, seq_len, dim, 2, 0);
        let v0 = partition::partition_sequence(&v, seq_len, dim, 2, 0);
        let k1 = partition::partition_sequence(&k, seq_len, dim, 2, 1);
        let v1 = partition::partition_sequence(&v, seq_len, dim, 2, 1);

        // GPU 0 (tokens 0,1): causal, only needs local chunk [0..2]
        // chunk 1 at global offset 2 is entirely in the future for tokens 0,1
        let out0 = ring_attention_cpu(
            &q0, &[&k0, &k1], &[&v0, &v1], local_seq, dim, true, &[0, 2], 0,
        );

        let q1 = partition::partition_sequence(&q, seq_len, dim, 2, 1);
        // GPU 1 (tokens 2,3): needs both chunks
        let out1 = ring_attention_cpu(
            &q1, &[&k1, &k0], &[&v1, &v0], local_seq, dim, true, &[2, 0], 2,
        );

        let ring_out = partition::gather_sequence(&[out0, out1], seq_len, dim);

        for (i, (a, b)) in standard_out.iter().zip(ring_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "causal mismatch at index {}: standard={}, ring={}", i, a, b
            );
        }
    }

    /// Deterministic pseudo-random seed so the matrix reproduces bit-for-bit
    /// across runs without pulling in a `rand` dep.
    fn seeded_qkv(seq_len: usize, dim: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // LCG — plenty for shaping values away from zero and giving each token
        // a distinctive score. Not statistically random; just non-degenerate.
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) & 0x7FFF) as f32 / 32768.0 - 0.5
        };
        let n = seq_len * dim;
        let q: Vec<f32> = (0..n).map(|_| next()).collect();
        let k: Vec<f32> = (0..n).map(|_| next()).collect();
        let v: Vec<f32> = (0..n).map(|_| next()).collect();
        (q, k, v)
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{label}: index {i}: naive={x}, ring={y}, diff={}",
                (x - y).abs(),
            );
        }
    }

    #[test]
    fn run_ring_full_ring_size_1_matches_naive_exactly() {
        // ring_size=1 must degenerate to naive_attention verbatim (same order of
        // operations, no online-softmax accumulation). This is the guarantee
        // that codegen can silence its "@context_parallel" warning for r=1.
        let (q, k, v) = seeded_qkv(8, 4, 0xC0FFEE);
        for &causal in &[false, true] {
            let naive = naive_attention(&q, &k, &v, 8, 4, causal);
            let ring = run_ring_attention_full(&q, &k, &v, 8, 4, 1, causal);
            assert_eq!(naive, ring, "ring_size=1 causal={causal} must be bit-identical");
        }
    }

    #[test]
    fn run_ring_full_matrix_matches_naive() {
        // (seq_len, dim, ring_size) sweeps ring_size ∈ {2, 4, 8} across shapes
        // where each divides evenly. Every entry is checked against
        // naive_attention with 1e-4 tolerance (online-softmax accumulation is
        // not bit-exact but numerically close).
        let cases: &[(usize, usize, usize)] = &[
            (4, 2, 2),
            (8, 4, 2),
            (8, 4, 4),
            (16, 8, 2),
            (16, 8, 4),
            (16, 8, 8),
            (12, 6, 3),  // odd ring_size
        ];
        for &(seq_len, dim, ring_size) in cases {
            for &causal in &[false, true] {
                let (q, k, v) = seeded_qkv(seq_len, dim, (seq_len * 31 + dim * 7 + ring_size) as u32);
                let naive = naive_attention(&q, &k, &v, seq_len, dim, causal);
                let ring = run_ring_attention_full(&q, &k, &v, seq_len, dim, ring_size, causal);
                assert_close(
                    &naive, &ring, 1e-4,
                    &format!("seq_len={seq_len} dim={dim} ring_size={ring_size} causal={causal}"),
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn run_ring_full_refuses_indivisible_seq_len() {
        let (q, k, v) = seeded_qkv(5, 2, 0xABCD);
        // 5 not divisible by 2 — partition would silently truncate; we panic instead.
        let _ = run_ring_attention_full(&q, &k, &v, 5, 2, 2, false);
    }
}
