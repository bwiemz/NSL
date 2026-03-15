//! RaggedBatchBuilder: assemble flat token tensors with no padding.

pub struct RaggedBatch {
    pub token_ids: Vec<i64>,
    pub seq_start_offsets: Vec<u32>,
    pub seq_lengths: Vec<u32>,
    pub num_seqs: usize,
}

impl RaggedBatch {
    pub fn total_tokens(&self) -> usize {
        self.token_ids.len()
    }
}

pub struct RaggedBatchBuilder {
    token_ids: Vec<i64>,
    seq_start_offsets: Vec<u32>,
    seq_lengths: Vec<u32>,
}

impl RaggedBatchBuilder {
    pub fn new() -> Self {
        RaggedBatchBuilder {
            token_ids: Vec::new(),
            seq_start_offsets: Vec::new(),
            seq_lengths: Vec::new(),
        }
    }

    pub fn add_sequence(&mut self, tokens: &[i64]) {
        let offset = self.token_ids.len() as u32;
        self.seq_start_offsets.push(offset);
        self.seq_lengths.push(tokens.len() as u32);
        self.token_ids.extend_from_slice(tokens);
    }

    pub fn build(self) -> RaggedBatch {
        let num_seqs = self.seq_start_offsets.len();
        RaggedBatch {
            token_ids: self.token_ids,
            seq_start_offsets: self.seq_start_offsets,
            seq_lengths: self.seq_lengths,
            num_seqs,
        }
    }

    pub fn clear(&mut self) {
        self.token_ids.clear();
        self.seq_start_offsets.clear();
        self.seq_lengths.clear();
    }
}

impl Default for RaggedBatchBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_batch_basic() {
        let mut builder = RaggedBatchBuilder::new();
        builder.add_sequence(&[1, 2, 3]);
        builder.add_sequence(&[4, 5]);
        builder.add_sequence(&[6]);
        let batch = builder.build();
        assert_eq!(batch.total_tokens(), 6);
        assert_eq!(batch.num_seqs, 3);
        assert_eq!(batch.token_ids, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(batch.seq_start_offsets, vec![0, 3, 5]);
        assert_eq!(batch.seq_lengths, vec![3, 2, 1]);
    }

    #[test]
    fn ragged_batch_empty() {
        let batch = RaggedBatchBuilder::new().build();
        assert_eq!(batch.total_tokens(), 0);
        assert_eq!(batch.num_seqs, 0);
    }

    #[test]
    fn ragged_batch_single() {
        let mut builder = RaggedBatchBuilder::new();
        builder.add_sequence(&[10, 20, 30, 40]);
        let batch = builder.build();
        assert_eq!(batch.total_tokens(), 4);
        assert_eq!(batch.num_seqs, 1);
    }
}
