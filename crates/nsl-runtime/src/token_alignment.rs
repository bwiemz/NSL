//! M44: Precomputed token alignment for constrained decoding.
//!
//! Maps byte-level DFA states to valid token IDs by walking each token's
//! byte sequence through the DFA and recording which tokens lead to valid states.

/// Precomputed token-level transition table.
///
/// For each DFA state, stores the set of valid token IDs and their
/// resulting next states.
pub struct TokenAlignmentTable {
    /// Number of DFA states.
    pub num_states: usize,
    /// For each state: list of (token_id, next_state) pairs.
    /// Sorted by token_id for binary search.
    pub state_transitions: Vec<Vec<TokenTransition>>,
    /// For each state: bitmask of valid token IDs (vocab_size bits).
    /// Used for fast logit masking.
    pub state_masks: Vec<Vec<u64>>,
    /// Vocabulary size.
    pub vocab_size: usize,
}

#[derive(Clone, Debug)]
pub struct TokenTransition {
    pub token_id: u32,
    pub next_state: u32,
}

/// Build a token alignment table from a byte-level DFA and a BPE vocabulary.
///
/// For each DFA state and each token in the vocabulary:
/// 1. Walk the token's byte sequence through the DFA starting from that state
/// 2. If all bytes are valid transitions and the final state exists, token is valid
/// 3. Record (token_id, final_state) in the transition table
///
/// `vocab`: slice of byte sequences, one per token ID (index = token ID).
/// `dfa_transitions`: [state][byte] = Option<next_state>, from the compiled DFA.
/// `num_states`: total DFA states.
pub fn build_token_alignment(
    vocab: &[Vec<u8>],
    dfa_transitions: &[[Option<u32>; 256]],
    num_states: usize,
) -> TokenAlignmentTable {
    let vocab_size = vocab.len();
    let mask_words = vocab_size.div_ceil(64); // u64 words for bitmask

    let mut state_transitions = vec![Vec::new(); num_states];
    let mut state_masks = vec![vec![0u64; mask_words]; num_states];

    for state in 0..num_states {
        for (token_id, token_bytes) in vocab.iter().enumerate() {
            if token_bytes.is_empty() {
                continue;
            }

            // Walk through DFA
            let mut current = state as u32;
            let mut valid = true;
            for &b in token_bytes {
                match dfa_transitions[current as usize][b as usize] {
                    Some(next) => current = next,
                    None => {
                        valid = false;
                        break;
                    }
                }
            }

            if valid {
                state_transitions[state].push(TokenTransition {
                    token_id: token_id as u32,
                    next_state: current,
                });
                // Set bit in mask
                let word = token_id / 64;
                let bit = token_id % 64;
                state_masks[state][word] |= 1u64 << bit;
            }
        }
    }

    TokenAlignmentTable {
        num_states,
        state_transitions,
        state_masks,
        vocab_size,
    }
}

impl TokenAlignmentTable {
    /// Check if a token is valid in a given state.
    pub fn is_valid(&self, state: usize, token_id: usize) -> bool {
        if state >= self.num_states || token_id >= self.vocab_size {
            return false;
        }
        let word = token_id / 64;
        let bit = token_id % 64;
        (self.state_masks[state][word] >> bit) & 1 == 1
    }

    /// Get the next state after accepting a token.
    pub fn next_state(&self, state: usize, token_id: usize) -> Option<u32> {
        self.state_transitions[state]
            .iter()
            .find(|t| t.token_id == token_id as u32)
            .map(|t| t.next_state)
    }

    /// Count valid tokens for a state.
    pub fn valid_count(&self, state: usize) -> usize {
        self.state_masks[state]
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn make_simple_dfa() -> (Vec<[Option<u32>; 256]>, HashSet<u32>) {
        // DFA accepting "ab" or "ac"
        // State 0 -a-> State 1; State 1 -b-> State 2 (accept); State 1 -c-> State 3 (accept)
        let mut trans = vec![[None; 256]; 4];
        trans[0][b'a' as usize] = Some(1);
        trans[1][b'b' as usize] = Some(2);
        trans[1][b'c' as usize] = Some(3);
        let accept: HashSet<u32> = [2, 3].into_iter().collect();
        (trans, accept)
    }

    #[test]
    fn token_alignment_basic() {
        let (trans, _accept) = make_simple_dfa();
        let vocab = vec![
            b"a".to_vec(),  // token 0
            b"b".to_vec(),  // token 1
            b"c".to_vec(),  // token 2
            b"ab".to_vec(), // token 3
            b"ac".to_vec(), // token 4
            b"d".to_vec(),  // token 5 (invalid for this DFA)
        ];

        let table = build_token_alignment(&vocab, &trans, 4);

        // From state 0: only "a" (token 0) is valid (leads to state 1)
        assert!(table.is_valid(0, 0)); // "a"
        assert!(!table.is_valid(0, 1)); // "b" not valid from state 0
        assert!(!table.is_valid(0, 5)); // "d" not valid

        // Multi-byte tokens: "ab" (token 3) from state 0 should be valid (walks a->1, b->2)
        assert!(table.is_valid(0, 3)); // "ab"
        assert!(table.is_valid(0, 4)); // "ac"

        // From state 1: "b" (token 1) and "c" (token 2) are valid
        assert!(table.is_valid(1, 1)); // "b"
        assert!(table.is_valid(1, 2)); // "c"
        assert!(!table.is_valid(1, 0)); // "a" not valid from state 1
    }

    #[test]
    fn next_state_lookup() {
        let (trans, _accept) = make_simple_dfa();
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()];
        let table = build_token_alignment(&vocab, &trans, 4);

        assert_eq!(table.next_state(0, 0), Some(1)); // "a" from state 0 -> state 1
        assert_eq!(table.next_state(0, 2), Some(2)); // "ab" from state 0 -> state 2
        assert_eq!(table.next_state(1, 1), Some(2)); // "b" from state 1 -> state 2
    }

    #[test]
    fn valid_count() {
        let (trans, _accept) = make_simple_dfa();
        let vocab = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"ab".to_vec(),
            b"ac".to_vec(),
        ];
        let table = build_token_alignment(&vocab, &trans, 4);

        assert_eq!(table.valid_count(0), 3); // a, ab, ac
        assert_eq!(table.valid_count(1), 2); // b, c
    }

    #[test]
    fn empty_token_skipped() {
        let (trans, _accept) = make_simple_dfa();
        let vocab = vec![b"".to_vec(), b"a".to_vec()];
        let table = build_token_alignment(&vocab, &trans, 4);
        assert!(!table.is_valid(0, 0)); // empty token skipped
        assert!(table.is_valid(0, 1)); // "a" valid
    }
}
