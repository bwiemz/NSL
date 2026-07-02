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
//!
//! Audit gap G12: [`dfa_from_json_schema`] REUSES the existing M44
//! pipeline (`schema_convert::json_schema_to_grammar` ->
//! `grammar_compiler::compile_grammar`, byte-level) and projects the
//! byte DFA to token level against a [`TokenVocab`]: token `t` is
//! valid in state `s` iff feeding `t`'s bytes from `s` stays live; the
//! token-level target is the byte state reached.

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

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
// JSON-schema -> token-level DFA adapter (audit gap G12)
// ---------------------------------------------------------------------------

/// Memory guard: refuse token projections whose transition table would
/// exceed this many `(state, token)` cells (u32 each -> 64 MB).
const MAX_PROJECTION_ENTRIES: u64 = 16 * 1024 * 1024;

/// Minimal tokenizer vocabulary: token id == index.
#[derive(Debug, Clone)]
pub struct TokenVocab {
    pub tokens: Vec<String>,
}

impl TokenVocab {
    /// Parse a JSON array of token strings (`["true", "fal", ...]`).
    pub fn from_json_str(src: &str) -> Result<Self, String> {
        let value: serde_json::Value =
            serde_json::from_str(src).map_err(|e| format!("tokenizer JSON: {e}"))?;
        let arr = value
            .as_array()
            .ok_or_else(|| "tokenizer JSON must be an array of strings".to_string())?;
        let mut tokens = Vec::with_capacity(arr.len());
        for (i, v) in arr.iter().enumerate() {
            match v.as_str() {
                Some(s) => tokens.push(s.to_string()),
                None => return Err(format!("tokenizer JSON entry {i} is not a string")),
            }
        }
        Ok(Self { tokens })
    }

    /// Parse a plain-text vocab: one token string per line (token id ==
    /// line index).  Only a trailing `\r` is stripped -- tokens may
    /// contain leading/trailing spaces.
    pub fn from_lines(src: &str) -> Result<Self, String> {
        let tokens: Vec<String> = src
            .lines()
            .map(|l| l.strip_suffix('\r').unwrap_or(l).to_string())
            .collect();
        if tokens.is_empty() {
            return Err("tokenizer file has no tokens".to_string());
        }
        Ok(Self { tokens })
    }

    /// Load a vocab file: `.json` -> JSON string array, anything else
    /// (`.txt` convention) -> one token per line.
    pub fn load(path: &Path) -> Result<Self, String> {
        let src = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read tokenizer '{}': {e}", path.display()))?;
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        {
            Self::from_json_str(&src)
        } else {
            Self::from_lines(&src)
        }
    }
}

/// Compile a JSON schema into a token-level [`CompiledDfa`] by reusing
/// the M44 byte-level pipeline and projecting against `vocab`.
///
/// Token `t` is a live transition from state `s` iff walking `t`'s
/// bytes through the byte DFA from `s` never hits a reject; the
/// token-level target is the byte state reached.  Empty tokens make no
/// byte progress and are never valid emissions (their mask bits stay
/// 0).  Only byte states reachable via whole tokens become token-level
/// states, so every accept state in the result is reachable by
/// construction.
pub fn dfa_from_json_schema(
    schema_json: &str,
    vocab: &TokenVocab,
) -> Result<CompiledDfa, String> {
    if vocab.tokens.is_empty() {
        return Err("token vocabulary is empty".to_string());
    }
    let value: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|e| format!("grammar schema is not valid JSON: {e}"))?;
    let grammar = crate::schema_convert::json_schema_to_grammar(&value)
        .map_err(|e| format!("grammar schema -> grammar: {e}"))?;
    let byte_dfa = crate::grammar_compiler::compile_grammar(&grammar);

    // Memory guard BEFORE the O(states x vocab x token_len) walk; the
    // byte-level state count upper-bounds the token-level count.
    let entries = (byte_dfa.num_states() as u64) * (vocab.tokens.len() as u64);
    if entries > MAX_PROJECTION_ENTRIES {
        return Err(format!(
            "token projection needs {} entries ({} byte-DFA states x {} tokens), \
             over the {} cap (G12 memory guard)",
            entries,
            byte_dfa.num_states(),
            vocab.tokens.len(),
            MAX_PROJECTION_ENTRIES
        ));
    }

    // BFS over token-reachable byte states; renumber them densely.
    let mut byte_to_tok: BTreeMap<u32, u32> = BTreeMap::new();
    let mut order: Vec<u32> = vec![byte_dfa.start];
    byte_to_tok.insert(byte_dfa.start, 0);
    let mut edges: Vec<GrammarEdge> = Vec::new();
    let mut i = 0usize;
    while i < order.len() {
        let byte_state = order[i];
        let from = i as u32;
        i += 1;
        for (tid, token) in vocab.tokens.iter().enumerate() {
            if token.is_empty() {
                continue;
            }
            let mut st = byte_state;
            let mut alive = true;
            for &b in token.as_bytes() {
                match byte_dfa.transitions[st as usize][b as usize] {
                    Some(next) => st = next,
                    None => {
                        alive = false;
                        break;
                    }
                }
            }
            if !alive {
                continue;
            }
            let to = match byte_to_tok.get(&st) {
                Some(&id) => id,
                None => {
                    let id = order.len() as u32;
                    byte_to_tok.insert(st, id);
                    order.push(st);
                    id
                }
            };
            edges.push(GrammarEdge {
                from,
                token_id: tid as u32,
                to,
            });
        }
    }

    let accept_states: Vec<u32> = order
        .iter()
        .enumerate()
        .filter(|(_, bs)| byte_dfa.accept.contains(*bs))
        .map(|(id, _)| id as u32)
        .collect();
    if accept_states.is_empty() {
        return Err(
            "no token sequence in this vocabulary produces a string the schema accepts \
             (G12 token projection)"
                .to_string(),
        );
    }

    // Co-accessibility pruning: drop token edges entering states from
    // which no accept state is reachable by WHOLE-token steps.  A byte
    // path can stay alive toward an accept string that no vocab token
    // sequence can finish (e.g. token "t" toward "true" when only
    // "true" itself is in the vocab); once decode entered such a state
    // the baked mask would set every logit to -inf — sampler deadlock.
    let num_states = order.len();
    let mut co_accessible = vec![false; num_states];
    for &a in &accept_states {
        co_accessible[a as usize] = true;
    }
    let mut changed = true;
    while changed {
        changed = false;
        for e in &edges {
            if co_accessible[e.to as usize] && !co_accessible[e.from as usize] {
                co_accessible[e.from as usize] = true;
                changed = true;
            }
        }
    }
    if !co_accessible[0] {
        return Err(
            "schema accept states are unreachable through whole-token steps of this \
             vocabulary (G12 token projection)"
                .to_string(),
        );
    }
    let edges: Vec<GrammarEdge> = edges
        .into_iter()
        .filter(|e| co_accessible[e.from as usize] && co_accessible[e.to as usize])
        .collect();

    let spec = GrammarSpec {
        num_states: order.len() as u32,
        vocab_size: vocab.tokens.len() as u32,
        start_state: 0,
        accept_states,
        edges,
    };
    compile(&spec).map_err(|e| e.to_string())
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
                GrammarEdge { from: 0, token_id: 0, to: 1 },
                GrammarEdge { from: 0, token_id: 1, to: 2 },
                GrammarEdge { from: 1, token_id: 0, to: 3 },
                GrammarEdge { from: 2, token_id: 0, to: 3 },
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
                GrammarEdge { from: 0, token_id: 1, to: 1 },
                GrammarEdge { from: 1, token_id: 2, to: 2 },
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
                GrammarEdge { from: 0, token_id: 1, to: 1 },
                GrammarEdge { from: 0, token_id: 1, to: 1 }, // duplicate
            ],
        };
        let dfa = compile(&spec).unwrap();
        assert_eq!(dfa.live_transitions, 1);
    }

    // ── dfa_from_json_schema (G12 adapter) ─────────────────────────

    fn vocab(tokens: &[&str]) -> TokenVocab {
        TokenVocab {
            tokens: tokens.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Walk a token sequence; returns the final state or REJECT.
    fn walk(dfa: &CompiledDfa, tokens: &[u32]) -> u32 {
        let mut state = dfa.start_state;
        for &t in tokens {
            state = dfa.transition(state, t);
            if state == CompiledDfa::REJECT {
                return CompiledDfa::REJECT;
            }
        }
        state
    }

    #[test]
    fn boolean_schema_whole_token_projection() {
        let v = vocab(&["true", "false", "x"]);
        let dfa = dfa_from_json_schema(r#"{"type": "boolean"}"#, &v).unwrap();
        // "true" and "false" are single valid tokens landing on accept.
        let s_true = walk(&dfa, &[0]);
        assert!(dfa.is_final(s_true));
        let s_false = walk(&dfa, &[1]);
        assert!(dfa.is_final(s_false));
        // "x" matches no prefix of true/false.
        assert_eq!(walk(&dfa, &[2]), CompiledDfa::REJECT);
        // Nothing follows a complete boolean.
        for t in 0..3 {
            assert!(!dfa.is_valid(s_true, t));
        }
    }

    #[test]
    fn multi_token_composition_reaches_accept() {
        // "true" must be assembled from "tr" + "ue"; "fal" + "se" for false.
        let v = vocab(&["tr", "ue", "fal", "se"]);
        let dfa = dfa_from_json_schema(r#"{"type": "boolean"}"#, &v).unwrap();
        assert!(dfa.is_final(walk(&dfa, &[0, 1])));
        assert!(dfa.is_final(walk(&dfa, &[2, 3])));
        // Mixed halves violate the byte automaton.
        assert_eq!(walk(&dfa, &[0, 3]), CompiledDfa::REJECT);
        // Partial coverage: "tr" alone is live but not final.
        let mid = walk(&dfa, &[0]);
        assert_ne!(mid, CompiledDfa::REJECT);
        assert!(!dfa.is_final(mid));
    }

    #[test]
    fn empty_tokens_are_never_valid() {
        let v = vocab(&["true", "", "false"]);
        let dfa = dfa_from_json_schema(r#"{"type": "boolean"}"#, &v).unwrap();
        for s in 0..dfa.num_states {
            assert!(!dfa.is_valid(s, 1), "empty token must stay masked out");
        }
    }

    #[test]
    fn dead_end_token_states_are_pruned() {
        // "t" walks bytes alive toward "true", but from its landing
        // state no vocab token continues (only whole "true" and "x"
        // exist).  Without co-accessibility pruning the baked mask
        // would let decode enter that state and then mask EVERY token
        // to -inf — sampler deadlock.
        let v = vocab(&["t", "true", "x"]);
        let dfa = dfa_from_json_schema(r#"{"type": "boolean"}"#, &v).unwrap();
        assert!(
            !dfa.is_valid(dfa.start_state, 0),
            "token 't' leads to a dead end and must be pruned from the start state"
        );
        assert!(dfa.is_valid(dfa.start_state, 1), "'true' stays valid");
        // Every state reachable through valid tokens must keep at least
        // one way forward or be an accept state — no deadlocks.
        for s in 0..dfa.num_states {
            let reachable = s == dfa.start_state
                || (0..dfa.num_states)
                    .any(|p| (0..dfa.vocab_size).any(|t| dfa.transition(p, t) == s));
            if !reachable {
                continue;
            }
            let has_out = (0..dfa.vocab_size).any(|t| dfa.is_valid(s, t));
            assert!(
                dfa.is_final(s) || has_out,
                "reachable state {s} is a non-accept dead end"
            );
        }
    }

    #[test]
    fn vocab_that_cannot_reach_accept_errs() {
        let v = vocab(&["x", "y"]);
        let err = dfa_from_json_schema(r#"{"type": "boolean"}"#, &v).unwrap_err();
        assert!(err.contains("G12"), "refusal must cite the gap: {err}");
    }

    #[test]
    fn invalid_schema_json_errs() {
        let v = vocab(&["true"]);
        assert!(dfa_from_json_schema("{not json", &v).is_err());
        // Valid JSON but unsupported schema shape.
        assert!(dfa_from_json_schema(r#"{"type": "quaternion"}"#, &v).is_err());
    }

    #[test]
    fn empty_vocab_errs() {
        let v = TokenVocab { tokens: vec![] };
        assert!(dfa_from_json_schema(r#"{"type": "boolean"}"#, &v).is_err());
    }

    #[test]
    fn projection_cap_refuses_oversized_tables() {
        // boolean byte-DFA minimises to 8 states; 3M tokens pushes the
        // product (24M) over the 16M-entry guard before any projection
        // work happens.
        let v = TokenVocab {
            tokens: vec![String::new(); 3_000_000],
        };
        let err = dfa_from_json_schema(r#"{"type": "boolean"}"#, &v).unwrap_err();
        assert!(err.contains("G12 memory guard"), "cap must refuse: {err}");
    }

    #[test]
    fn enum_schema_projects_quoted_values() {
        // Enum values render as JSON strings -- quotes included.
        let v = vocab(&["\"red\"", "\"blue\"", "red"]);
        let dfa = dfa_from_json_schema(r#"{"enum": ["red", "blue"]}"#, &v).unwrap();
        assert!(dfa.is_final(walk(&dfa, &[0])));
        assert!(dfa.is_final(walk(&dfa, &[1])));
        // Unquoted "red" never starts a valid enum literal.
        assert_eq!(walk(&dfa, &[2]), CompiledDfa::REJECT);
    }

    // ── TokenVocab loading ─────────────────────────────────────────

    #[test]
    fn token_vocab_from_json_and_lines() {
        let v = TokenVocab::from_json_str(r#"["a", "bc", "d"]"#).unwrap();
        assert_eq!(v.tokens, vec!["a", "bc", "d"]);
        assert!(TokenVocab::from_json_str(r#"{"a": 1}"#).is_err());
        assert!(TokenVocab::from_json_str(r#"["a", 3]"#).is_err());

        let v = TokenVocab::from_lines("true\nfalse\r\nx\n").unwrap();
        assert_eq!(v.tokens, vec!["true", "false", "x"]);
        assert!(TokenVocab::from_lines("").is_err());
    }

    #[test]
    fn token_vocab_load_dispatches_on_extension() {
        let dir = tempfile::tempdir().unwrap();
        let txt = dir.path().join("vocab.txt");
        std::fs::write(&txt, "tr\nue\n").unwrap();
        let json = dir.path().join("vocab.json");
        std::fs::write(&json, r#"["tr", "ue"]"#).unwrap();

        let from_txt = TokenVocab::load(&txt).unwrap();
        let from_json = TokenVocab::load(&json).unwrap();
        assert_eq!(from_txt.tokens, from_json.tokens);
        assert!(TokenVocab::load(&dir.path().join("missing.txt")).is_err());
    }
}
