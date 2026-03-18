// crates/nsl-runtime/src/kv_compress/sliding_window.rs
//! Sliding window KV-cache eviction with attention sinks.
//!
//! Retains the first `num_sinks` tokens (attention sinks) plus the most recent
//! `window_size` tokens. Everything in between is evicted at page granularity.

/// Manages sliding window eviction for one compression policy.
pub struct SlidingWindowManager {
    pub window_size: usize,
    pub num_sinks: usize,
    pub block_size: usize,
}

impl SlidingWindowManager {
    pub fn new(window_size: usize, num_sinks: usize, block_size: usize) -> Self {
        SlidingWindowManager { window_size, num_sinks, block_size }
    }

    /// Check if eviction is needed. Returns logical block indices to evict.
    ///
    /// A block is evicted only if ALL tokens in it fall in the eviction range
    /// (between sinks and the start of the window).
    pub fn check_eviction(&self, current_len: usize) -> Vec<usize> {
        if current_len <= self.num_sinks + self.window_size {
            return vec![];
        }

        let evict_start = self.num_sinks;
        let evict_end = current_len - self.window_size;

        let mut to_evict = Vec::new();
        // Only evict full blocks where every token is in [evict_start, evict_end)
        let start_block = evict_start.div_ceil(self.block_size); // ceil
        let end_block = evict_end / self.block_size; // floor

        for block_idx in start_block..end_block {
            let block_start_token = block_idx * self.block_size;
            let block_end_token = (block_idx + 1) * self.block_size;
            if block_start_token >= evict_start && block_end_token <= evict_end {
                to_evict.push(block_idx);
            }
        }
        to_evict
    }

    /// Active token count (sinks + window), capped at current_len.
    pub fn active_tokens(&self, current_len: usize) -> usize {
        if current_len <= self.num_sinks + self.window_size {
            current_len
        } else {
            self.num_sinks + self.window_size
        }
    }

    /// The attention config ranges for this window state.
    ///
    /// Returns (sink_end, window_start, window_end) for FlashAttention tile-skip.
    pub fn attention_ranges(&self, current_len: usize) -> (usize, usize, usize) {
        let sink_end = self.num_sinks.min(current_len);
        if current_len <= self.num_sinks + self.window_size {
            (sink_end, 0, current_len) // no eviction yet — attend to everything
        } else {
            let window_start = current_len - self.window_size;
            (sink_end, window_start, current_len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_eviction_when_within_budget() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        assert!(mgr.check_eviction(36).is_empty()); // exactly at budget
        assert!(mgr.check_eviction(20).is_empty());
        assert!(mgr.check_eviction(0).is_empty());
    }

    #[test]
    fn eviction_correct_blocks() {
        // window=32, sinks=4, block_size=8
        // At 100 tokens: sinks=[0..3], window=[68..99], evict=[4..67]
        // Block 0: tokens 0-7 (contains sinks) — NOT evicted
        // Block 1: tokens 8-15 — evicted (all in [4..67])
        // Block 2: tokens 16-23 — evicted
        // ...
        // Block 8: tokens 64-71 — NOT evicted (overlaps window start 68)
        let mgr = SlidingWindowManager::new(32, 4, 8);
        let evicted = mgr.check_eviction(100);
        // Blocks 1..8 (indices 1,2,3,4,5,6,7) — block 0 has sinks, block 8 straddles
        assert_eq!(evicted, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn partial_block_not_evicted() {
        // window=10, sinks=2, block_size=8
        // At 20 tokens: evict=[2..9], which is 8 tokens
        // Block 0: 0-7 (has sinks 0-1, also has evictable 2-7) — NOT evicted (sinks protect it)
        // Block 1: 8-15 (has token 8-9 in evict range, 10-15 in window) — NOT evicted
        let mgr = SlidingWindowManager::new(10, 2, 8);
        let evicted = mgr.check_eviction(20);
        assert!(evicted.is_empty()); // no full block is entirely in the evict range
    }

    #[test]
    fn large_eviction() {
        // window=16, sinks=0, block_size=4
        // At 100 tokens: evict=[0..83]
        // Blocks 0..20 are fully evicted (tokens 0..83 -> 84/4 = 21 blocks)
        let mgr = SlidingWindowManager::new(16, 0, 4);
        let evicted = mgr.check_eviction(100);
        assert_eq!(evicted.len(), 21); // blocks 0..20
    }

    #[test]
    fn active_tokens_capped() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        assert_eq!(mgr.active_tokens(10), 10);
        assert_eq!(mgr.active_tokens(36), 36);
        assert_eq!(mgr.active_tokens(100), 36); // sinks + window
        assert_eq!(mgr.active_tokens(10000), 36);
    }

    #[test]
    fn attention_ranges_before_eviction() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        let (sink_end, window_start, window_end) = mgr.attention_ranges(20);
        assert_eq!(sink_end, 4);
        assert_eq!(window_start, 0); // no eviction
        assert_eq!(window_end, 20);
    }

    #[test]
    fn attention_ranges_after_eviction() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        let (sink_end, window_start, window_end) = mgr.attention_ranges(100);
        assert_eq!(sink_end, 4);
        assert_eq!(window_start, 68);
        assert_eq!(window_end, 100);
    }
}
