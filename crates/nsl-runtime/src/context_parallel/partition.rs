/// Compute the token range for a given rank.
///
/// # Panics
/// Panics if `seq_len` is not evenly divisible by `ring_size`.
pub fn chunk_range(seq_len: usize, ring_size: usize, rank: usize) -> (usize, usize) {
    assert_eq!(
        seq_len % ring_size, 0,
        "seq_len ({}) must be divisible by ring_size ({})", seq_len, ring_size
    );
    let chunk_len = seq_len / ring_size;
    let start = rank * chunk_len;
    let end = start + chunk_len;
    (start, end)
}

/// Extract this rank's chunk from a full sequence.
/// `data`: flat [seq_len, hidden_dim]
pub fn partition_sequence(
    data: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ring_size: usize,
    rank: usize,
) -> Vec<f32> {
    let (start, end) = chunk_range(seq_len, ring_size, rank);
    let chunk_len = end - start;
    let mut chunk = vec![0.0f32; chunk_len * hidden_dim];
    for t in 0..chunk_len {
        let src_off = (start + t) * hidden_dim;
        let dst_off = t * hidden_dim;
        chunk[dst_off..dst_off + hidden_dim]
            .copy_from_slice(&data[src_off..src_off + hidden_dim]);
    }
    chunk
}

/// Gather chunks from all ranks into a full sequence.
/// `chunks`: one Vec per rank, each [chunk_len, hidden_dim]
pub fn gather_sequence(
    chunks: &[Vec<f32>],
    seq_len: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut full = vec![0.0f32; seq_len * hidden_dim];
    let chunk_len = seq_len / chunks.len();
    for (rank, chunk) in chunks.iter().enumerate() {
        let start = rank * chunk_len * hidden_dim;
        full[start..start + chunk.len()].copy_from_slice(chunk);
    }
    full
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_4way() {
        // 12 tokens, 4 ranks, hidden_dim=2
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let chunk = partition_sequence(&data, 12, 2, 4, 1); // rank 1
        // rank 1 gets tokens 3..6 = indices 6..12
        assert_eq!(chunk, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_partition_gather_roundtrip() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let hidden_dim = 2;
        let ring_size = 2;
        let seq_len = 8;

        let chunk0 = partition_sequence(&data, seq_len, hidden_dim, ring_size, 0);
        let chunk1 = partition_sequence(&data, seq_len, hidden_dim, ring_size, 1);

        let gathered = gather_sequence(&[chunk0, chunk1], seq_len, hidden_dim);
        assert_eq!(gathered, data);
    }

    #[test]
    fn test_chunk_boundaries() {
        let (start, end) = chunk_range(1000, 4, 2); // rank 2 of 4
        assert_eq!(start, 500);
        assert_eq!(end, 750);
    }
}
