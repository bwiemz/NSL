// crates/nsl-runtime/src/kv_compress/h2o.rs
//! Heavy Hitter Oracle (H2O) — attention-score-based KV eviction.
//!
//! Maintains cumulative attention scores per KV position. When the sequence
//! exceeds the budget, positions with the lowest scores are evicted (except sinks).

use std::collections::HashMap;

/// H2O eviction manager for attention-score-based KV pruning.
pub struct H2OManager {
    pub budget: usize,
    pub num_sinks: usize,
    pub block_size: usize,
    /// Per-sequence cumulative scores: scores[seq_id][token_pos] = cumulative attention weight.
    scores: HashMap<u64, Vec<f32>>,
}

impl H2OManager {
    pub fn new(budget: usize, num_sinks: usize, block_size: usize) -> Self {
        H2OManager {
            budget,
            num_sinks,
            block_size,
            scores: HashMap::new(),
        }
    }

    /// Accumulate attention scores for a sequence after one decode step.
    ///
    /// `scores_for_step`: [seq_len] attention weights from the latest query position,
    /// averaged across all layers and heads.
    pub fn accumulate_scores(&mut self, seq_id: u64, scores_for_step: &[f32]) {
        let cumulative = self.scores.entry(seq_id).or_default();

        // Extend if sequence grew
        if cumulative.len() < scores_for_step.len() {
            cumulative.resize(scores_for_step.len(), 0.0);
        }

        for (pos, &weight) in scores_for_step.iter().enumerate() {
            cumulative[pos] += weight;
        }
    }

    /// Check if eviction is needed. Returns logical block indices to evict.
    ///
    /// Evicts tokens with lowest cumulative scores (except sinks).
    /// Only evicts full blocks where ALL tokens are marked for eviction.
    pub fn check_eviction(&self, seq_id: u64, current_len: usize) -> Vec<usize> {
        if current_len <= self.budget {
            return vec![];
        }

        let scores = match self.scores.get(&seq_id) {
            Some(s) => s,
            None => return vec![],
        };

        // Collect non-sink positions with their scores
        let mut candidates: Vec<(usize, f32)> = scores.iter()
            .enumerate()
            .filter(|(pos, _)| *pos >= self.num_sinks && *pos < current_len)
            .map(|(pos, &score)| (pos, score))
            .collect();

        // Sort ascending by score (lowest first = evict first)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let num_to_evict = current_len - self.budget;
        let evict_set: std::collections::HashSet<usize> = candidates.iter()
            .take(num_to_evict)
            .map(|(pos, _)| *pos)
            .collect();

        // Find blocks where ALL tokens are in the evict set
        let num_blocks = current_len.div_ceil(self.block_size);
        let mut blocks_to_evict = Vec::new();
        for block_idx in 0..num_blocks {
            let block_start = block_idx * self.block_size;
            let block_end = ((block_idx + 1) * self.block_size).min(current_len);

            // Skip sink blocks
            if block_start < self.num_sinks {
                continue;
            }

            let all_evicted = (block_start..block_end).all(|pos| evict_set.contains(&pos));
            if all_evicted {
                blocks_to_evict.push(block_idx);
            }
        }
        blocks_to_evict
    }

    /// Remove tracking data for a completed sequence.
    pub fn remove_sequence(&mut self, seq_id: u64) {
        self.scores.remove(&seq_id);
    }

    /// Get cumulative scores for a sequence (for debugging/profiling).
    pub fn get_scores(&self, seq_id: u64) -> Option<&[f32]> {
        self.scores.get(&seq_id).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_eviction_within_budget() {
        let mgr = H2OManager::new(10, 2, 4);
        assert!(mgr.check_eviction(0, 5).is_empty());
        assert!(mgr.check_eviction(0, 10).is_empty());
    }

    #[test]
    fn score_accumulation() {
        let mut mgr = H2OManager::new(100, 0, 4);
        mgr.accumulate_scores(0, &[1.0, 2.0, 3.0]);
        mgr.accumulate_scores(0, &[0.5, 0.5, 0.5]);

        let scores = mgr.get_scores(0).unwrap();
        assert_eq!(scores, &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn score_extends_on_growth() {
        let mut mgr = H2OManager::new(100, 0, 4);
        mgr.accumulate_scores(0, &[1.0, 2.0]);
        mgr.accumulate_scores(0, &[0.5, 0.5, 3.0, 4.0]);

        let scores = mgr.get_scores(0).unwrap();
        assert_eq!(scores, &[1.5, 2.5, 3.0, 4.0]);
    }

    #[test]
    fn eviction_order_lowest_first() {
        // budget=4, sinks=0, block_size=1 (for precise per-token eviction)
        let mut mgr = H2OManager::new(4, 0, 1);

        // 6 tokens with scores: token 2 and 4 have lowest scores
        mgr.accumulate_scores(0, &[5.0, 3.0, 1.0, 4.0, 0.5, 6.0]);

        let evicted = mgr.check_eviction(0, 6);
        // Need to evict 2 tokens (6 - 4 = 2). Lowest: pos 4 (0.5), pos 2 (1.0)
        // With block_size=1, blocks = token positions
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&2));
        assert!(evicted.contains(&4));
    }

    #[test]
    fn sinks_protected_from_eviction() {
        // budget=3, sinks=2, block_size=1
        let mut mgr = H2OManager::new(3, 2, 1);

        // Sinks (pos 0,1) have lowest scores but must not be evicted
        mgr.accumulate_scores(0, &[0.1, 0.2, 5.0, 3.0, 1.0]);

        let evicted = mgr.check_eviction(0, 5);
        // Need to evict 2 (5 - 3 = 2). Candidates (non-sink): pos 2(5.0), 3(3.0), 4(1.0)
        // Lowest non-sink: pos 4(1.0), pos 3(3.0)
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&3));
        assert!(evicted.contains(&4));
        // Sinks must NOT be evicted
        assert!(!evicted.contains(&0));
        assert!(!evicted.contains(&1));
    }

    #[test]
    fn block_level_eviction_only_full_blocks() {
        // budget=8, sinks=0, block_size=4
        let mut mgr = H2OManager::new(8, 0, 4);

        // 16 tokens: tokens 0-3 low scores, 4-7 high, 8-11 low, 12-15 high
        let scores: Vec<f32> = (0..16).map(|i| {
            if i < 4 || (8..12).contains(&i) { 0.1 } else { 10.0 }
        }).collect();
        mgr.accumulate_scores(0, &scores);

        let evicted = mgr.check_eviction(0, 16);
        // Need to evict 8 tokens. Lowest 8: tokens 0-3 and 8-11.
        // Block 0 (tokens 0-3): all evicted -> evict block
        // Block 2 (tokens 8-11): all evicted -> evict block
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&0));
        assert!(evicted.contains(&2));
    }

    #[test]
    fn remove_sequence_clears_data() {
        let mut mgr = H2OManager::new(100, 0, 4);
        mgr.accumulate_scores(42, &[1.0, 2.0]);
        assert!(mgr.get_scores(42).is_some());
        mgr.remove_sequence(42);
        assert!(mgr.get_scores(42).is_none());
    }
}
