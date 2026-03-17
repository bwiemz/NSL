/// Configuration for ring attention context parallelism.
#[derive(Debug, Clone)]
pub struct RingConfig {
    /// Number of GPUs in the ring.
    pub ring_size: usize,
    /// This GPU's rank in the ring (0..ring_size-1).
    pub ring_rank: usize,
    /// Tokens per GPU chunk: total_seq_len / ring_size.
    pub local_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Whether attention is causal (enables early termination).
    pub causal: bool,
}

impl RingConfig {
    /// Number of ring passes this GPU needs.
    /// Causal: ring_rank + 1 (GPU 0 needs only local, GPU N-1 needs all).
    /// Non-causal: ring_size (all GPUs need all K/V).
    pub fn num_passes(&self) -> usize {
        if self.causal {
            self.ring_rank + 1
        } else {
            self.ring_size
        }
    }

    /// Source rank for a given ring pass.
    /// Pass 0 = local (ring_rank). Pass p = (ring_rank + ring_size - p) % ring_size.
    pub fn source_rank(&self, pass: usize) -> usize {
        (self.ring_rank + self.ring_size - pass) % self.ring_size
    }
}

/// Per-ring-pass metadata.
#[derive(Debug, Clone)]
pub struct RingPassInfo {
    /// Which chunk of the global sequence is in the current K/V buffer.
    pub chunk_start: usize,
    /// Length of the K/V chunk.
    pub chunk_len: usize,
    /// Source rank that produced this K/V chunk.
    pub source_rank: usize,
    /// Whether this is the last pass.
    pub is_final: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_num_passes() {
        let cfg = RingConfig {
            ring_size: 4, ring_rank: 0, local_seq_len: 250,
            num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: true,
        };
        assert_eq!(cfg.num_passes(), 1); // GPU 0: local only

        let cfg3 = RingConfig { ring_rank: 3, ..cfg };
        assert_eq!(cfg3.num_passes(), 4); // GPU 3: all passes
    }

    #[test]
    fn test_noncausal_num_passes() {
        let cfg = RingConfig {
            ring_size: 4, ring_rank: 0, local_seq_len: 250,
            num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: false,
        };
        assert_eq!(cfg.num_passes(), 4); // all GPUs need all passes
    }

    #[test]
    fn test_source_rank_ring_order() {
        let cfg = RingConfig {
            ring_size: 4, ring_rank: 2, local_seq_len: 250,
            num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: false,
        };
        assert_eq!(cfg.source_rank(0), 2);
        assert_eq!(cfg.source_rank(1), 1);
        assert_eq!(cfg.source_rank(2), 0);
        assert_eq!(cfg.source_rank(3), 3);
    }
}
