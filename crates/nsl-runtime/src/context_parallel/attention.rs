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

        // Normalize
        for d in 0..dim {
            out[qi * dim + d] = o_acc[d] / global_sum;
        }
    }
    out
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
}
