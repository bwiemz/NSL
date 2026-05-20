//! CFIE — Grammar-in-kernel: compile a DFA directly into the sampling
//! kernel's PTX control flow.
//!
//! Paper §4 (grammar-constrained sampling) + Gemini's review: instead
//! of bouncing logits between GPU and a Python `Outlines` / `lm-format-
//! enforcer` DFA interpreter, CFIE bakes the entire transition table
//! into the fused sampler as a compile-time constant.  The kernel
//! masks invalid tokens in SMEM — zero CPU-GPU round-trips per step.
//!
//! The input is a user-supplied grammar (e.g., a JSON schema); we
//! translate it into a minimal [`CompiledDfa`] whose transition table
//! is the exact artifact the sampler kernel reads.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Compact, vocab-indexed DFA ready for emission into PTX.
///
/// `transitions[state * vocab_size + token] = next_state`, where
/// `REJECT` marks an invalid transition.  `is_accept[state]` is set
/// iff the state is a valid final state.
#[derive(Debug, Clone, Serialize)]
pub struct CompiledDfa {
    pub num_states: u32,
    pub vocab_size: u32,
    /// Flattened transition table.  `u32::MAX` is the reject sentinel.
    pub transitions: Vec<u32>,
    /// Accept-state bitset.
    pub is_accept: Vec<bool>,
    /// Initial state.
    pub start_state: u32,
    /// For reporting: total number of non-reject transitions.
    pub live_transitions: u32,
}

impl CompiledDfa {
    pub const REJECT: u32 = u32::MAX;

    pub fn transition(&self, state: u32, token: u32) -> u32 {
        if state >= self.num_states || token >= self.vocab_size {
            return Self::REJECT;
        }
        self.transitions[(state * self.vocab_size + token) as usize]
    }

    pub fn is_valid(&self, state: u32, token: u32) -> bool {
        self.transition(state, token) != Self::REJECT
    }

    pub fn is_final(&self, state: u32) -> bool {
        state < self.num_states && self.is_accept[state as usize]
    }

    /// Memory footprint of the table in bytes (u32 per entry + 1 byte
    /// per accept bit, packed).
    pub fn table_bytes(&self) -> u64 {
        (self.transitions.len() as u64) * 4 + (self.is_accept.len() as u64 / 8).max(1)
    }

    /// Density — fraction of `(state, token)` cells that are live
    /// transitions.  Used for reporting.
    pub fn density(&self) -> f64 {
        let total = (self.num_states as u64) * (self.vocab_size as u64);
        if total == 0 {
            return 0.0;
        }
        self.live_transitions as f64 / total as f64
    }
}

/// A single edge in an abstract grammar graph: from `from` via token
/// `token_id` to `to`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrammarEdge {
    pub from: u32,
    pub token_id: u32,
    pub to: u32,
}

/// Source-level grammar representation.  We keep the structure simple
/// to avoid coupling to any particular grammar front-end (regex, EBNF,
/// JSON-schema) — callers reduce their grammar to this edge list.
#[derive(Debug, Clone)]
pub struct GrammarSpec {
    pub num_states: u32,
    pub vocab_size: u32,
    pub start_state: u32,
    pub accept_states: Vec<u32>,
    pub edges: Vec<GrammarEdge>,
}

impl GrammarSpec {
    /// A convenience builder that produces a grammar accepting a single
    /// fixed token sequence — useful for schema "constant field"
    /// clauses and as a sanity-check test fixture.
    pub fn sequence(tokens: &[u32], vocab_size: u32) -> Self {
        let num_states = tokens.len() as u32 + 1;
        let edges = tokens
            .iter()
            .enumerate()
            .map(|(i, &tok)| GrammarEdge {
                from: i as u32,
                token_id: tok,
                to: i as u32 + 1,
            })
            .collect();
        Self {
            num_states,
            vocab_size,
            start_state: 0,
            accept_states: vec![num_states - 1],
            edges,
        }
    }
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    VocabTokenOutOfRange { edge: GrammarEdge },
    StateOutOfRange { edge: GrammarEdge },
    UnreachableAcceptState(u32),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::VocabTokenOutOfRange { edge } => {
                write!(f, "edge token {} out of vocab range", edge.token_id)
            }
            CompileError::StateOutOfRange { edge } => {
                write!(f, "edge references state out of range: {:?}", edge)
            }
            CompileError::UnreachableAcceptState(s) => {
                write!(f, "accept state {} is unreachable from start", s)
            }
        }
    }
}

/// Compile the grammar into the kernel-ready transition table.
pub fn compile(spec: &GrammarSpec) -> Result<CompiledDfa, CompileError> {
    // Validate edges up-front.
    for e in &spec.edges {
        if e.token_id >= spec.vocab_size {
            return Err(CompileError::VocabTokenOutOfRange { edge: *e });
        }
        if e.from >= spec.num_states || e.to >= spec.num_states {
            return Err(CompileError::StateOutOfRange { edge: *e });
        }
    }

    let total_cells = (spec.num_states as usize) * (spec.vocab_size as usize);
    let mut transitions = vec![CompiledDfa::REJECT; total_cells];
    let mut live = 0u32;
    // Keep track of the last-seen target for conflicting edges (NFA → DFA
    // determinisation isn't needed when the grammar is already a DFA;
    // conflicting edges are silently overwritten, matching the usual
    // "longest-match" semantics).
    for e in &spec.edges {
        let idx = (e.from as usize) * (spec.vocab_size as usize) + (e.token_id as usize);
        if transitions[idx] == CompiledDfa::REJECT {
            live += 1;
        }
        transitions[idx] = e.to;
    }

    // Reachability check for accept states (paper §4 requires that no
    // dead accept sits in the table).
    let mut reachable = BTreeSet::new();
    reachable.insert(spec.start_state);
    let mut frontier = vec![spec.start_state];
    while let Some(state) = frontier.pop() {
        for t in 0..spec.vocab_size {
            let next = transitions[(state as usize) * (spec.vocab_size as usize) + t as usize];
            if next != CompiledDfa::REJECT && reachable.insert(next) {
                frontier.push(next);
            }
        }
    }
    let mut is_accept = vec![false; spec.num_states as usize];
    for s in &spec.accept_states {
        if !reachable.contains(s) {
            return Err(CompileError::UnreachableAcceptState(*s));
        }
        if (*s as usize) < is_accept.len() {
            is_accept[*s as usize] = true;
        }
    }

    Ok(CompiledDfa {
        num_states: spec.num_states,
        vocab_size: spec.vocab_size,
        transitions,
        is_accept,
        start_state: spec.start_state,
        live_transitions: live,
    })
}

/// Minimisation: merge equivalent states.  Returns a new DFA with
/// `num_states ≤ input.num_states`.  Uses Hopcroft-style partition
/// refinement.
pub fn minimise(dfa: &CompiledDfa) -> CompiledDfa {
    let n = dfa.num_states as usize;
    let v = dfa.vocab_size as usize;
    // Initial partition: accept vs non-accept.
    let mut partition: Vec<usize> = (0..n)
        .map(|s| if dfa.is_accept[s] { 0 } else { 1 })
        .collect();
    let mut num_groups = if dfa.is_accept.iter().any(|b| *b) && dfa.is_accept.iter().any(|b| !b) {
        2
    } else {
        1
    };
    loop {
        // Signature of each state: (current group, [target-groups for each token]).
        let mut sig_to_group: BTreeMap<(usize, Vec<i64>), usize> = BTreeMap::new();
        let mut new_partition = vec![0usize; n];
        let mut next_id = 0usize;
        for s in 0..n {
            let mut targets = Vec::with_capacity(v);
            for t in 0..v {
                let next = dfa.transitions[s * v + t];
                targets.push(if next == CompiledDfa::REJECT {
                    -1
                } else {
                    partition[next as usize] as i64
                });
            }
            let key = (partition[s], targets);
            let entry = sig_to_group.entry(key).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            new_partition[s] = *entry;
        }
        if new_partition == partition {
            break;
        }
        partition = new_partition;
        num_groups = next_id;
    }

    // Build the minimised DFA.  Each new state = one partition group.
    let mut new_transitions = vec![CompiledDfa::REJECT; num_groups * v];
    let mut new_is_accept = vec![false; num_groups];
    let mut live = 0u32;
    for s in 0..n {
        let ns = partition[s];
        if dfa.is_accept[s] {
            new_is_accept[ns] = true;
        }
        for t in 0..v {
            let next = dfa.transitions[s * v + t];
            let cell = &mut new_transitions[ns * v + t];
            if next == CompiledDfa::REJECT {
                continue;
            }
            let mapped = partition[next as usize] as u32;
            if *cell == CompiledDfa::REJECT {
                live += 1;
                *cell = mapped;
            }
        }
    }
    CompiledDfa {
        num_states: num_groups as u32,
        vocab_size: dfa.vocab_size,
        transitions: new_transitions,
        is_accept: new_is_accept,
        start_state: partition[dfa.start_state as usize] as u32,
        live_transitions: live,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_grammar_accepts_exact_sequence_only() {
        let spec = GrammarSpec::sequence(&[5, 7, 3], 10);
        let dfa = compile(&spec).unwrap();
        assert_eq!(dfa.num_states, 4);
        assert_eq!(dfa.start_state, 0);
        assert!(dfa.is_final(3));
        // Valid: follow the sequence.
        assert_eq!(dfa.transition(0, 5), 1);
        assert_eq!(dfa.transition(1, 7), 2);
        assert_eq!(dfa.transition(2, 3), 3);
        // Invalid: every other transition is REJECT.
        assert!(!dfa.is_valid(0, 1));
        assert!(!dfa.is_valid(1, 5));
    }

    #[test]
    fn density_is_low_for_sparse_grammar() {
        let spec = GrammarSpec::sequence(&[1, 2, 3], 1000);
        let dfa = compile(&spec).unwrap();
        // 3 live edges / (4 states × 1000 vocab) = 0.00075
        assert!(dfa.density() < 0.01);
    }

    #[test]
    fn compile_rejects_out_of_range_tokens() {
        let spec = GrammarSpec {
            num_states: 2,
            vocab_size: 5,
            start_state: 0,
            accept_states: vec![1],
            edges: vec![GrammarEdge {
                from: 0,
                token_id: 10,
                to: 1,
            }],
        };
        assert!(matches!(
            compile(&spec),
            Err(CompileError::VocabTokenOutOfRange { .. })
        ));
    }

    #[test]
    fn compile_rejects_unreachable_accept() {
        let spec = GrammarSpec {
            num_states: 3,
            vocab_size: 3,
            start_state: 0,
            accept_states: vec![2], // no edge reaches state 2
            edges: vec![GrammarEdge {
                from: 0,
                token_id: 1,
                to: 1,
            }],
        };
        assert!(matches!(
            compile(&spec),
            Err(CompileError::UnreachableAcceptState(2))
        ));
    }

    #[test]
    fn minimise_merges_equivalent_states() {
        // Two states with identical outgoing transitions should merge.
        let spec = GrammarSpec {
            num_states: 4,
            vocab_size: 2,
            start_state: 0,
            accept_states: vec![3],
            edges: vec![
                GrammarEdge {
                    from: 0,
                    token_id: 0,
                    to: 1,
                },
                GrammarEdge {
                    from: 0,
                    token_id: 1,
                    to: 2,
                },
                GrammarEdge {
                    from: 1,
                    token_id: 0,
                    to: 3,
                },
                GrammarEdge {
                    from: 2,
                    token_id: 0,
                    to: 3,
                },
            ],
        };
        let dfa = compile(&spec).unwrap();
        let minimised = minimise(&dfa);
        assert!(minimised.num_states < dfa.num_states);
    }

    #[test]
    fn minimisation_preserves_language() {
        let spec = GrammarSpec::sequence(&[0, 1, 2], 3);
        let dfa = compile(&spec).unwrap();
        let m = minimise(&dfa);
        // Walk [0, 1, 2] through the minimised DFA.
        let mut state = m.start_state;
        for tok in [0u32, 1, 2] {
            state = m.transition(state, tok);
            assert_ne!(state, CompiledDfa::REJECT);
        }
        assert!(m.is_final(state));
    }

    #[test]
    fn table_bytes_reflects_size() {
        let spec = GrammarSpec::sequence(&[0, 1], 32);
        let dfa = compile(&spec).unwrap();
        assert_eq!(dfa.table_bytes(), (3 * 32 * 4) + 1);
    }

    #[test]
    fn live_transition_count_matches_edges() {
        let spec = GrammarSpec {
            num_states: 3,
            vocab_size: 3,
            start_state: 0,
            accept_states: vec![2],
            edges: vec![
                GrammarEdge {
                    from: 0,
                    token_id: 1,
                    to: 1,
                },
                GrammarEdge {
                    from: 1,
                    token_id: 2,
                    to: 2,
                },
            ],
        };
        let dfa = compile(&spec).unwrap();
        assert_eq!(dfa.live_transitions, 2);
    }

    #[test]
    fn duplicate_edges_are_overwritten_not_counted_twice() {
        let spec = GrammarSpec {
            num_states: 2,
            vocab_size: 2,
            start_state: 0,
            accept_states: vec![1],
            edges: vec![
                GrammarEdge {
                    from: 0,
                    token_id: 1,
                    to: 1,
                },
                GrammarEdge {
                    from: 0,
                    token_id: 1,
                    to: 1,
                }, // duplicate
            ],
        };
        let dfa = compile(&spec).unwrap();
        assert_eq!(dfa.live_transitions, 1);
    }
}
