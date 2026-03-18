# M44: Structured Generation / Constrained Decoding — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compile-time FSM-based constrained decoding so LLM output conforms to JSON schemas, regex patterns, or BNF grammars. The compiler builds a finite state machine from the constraint, aligns it to the tokenizer vocabulary, and compresses it into a token-level transition table. At decode time, a single array lookup per token masks invalid logits — zero Python overhead.

**Architecture:** Two codegen modules (`grammar_compiler.rs` for NFA/DFA construction, `schema_convert.rs` for JSON Schema/BNF → Grammar) + two runtime modules (`grammar.rs` for FSM stepping + logit masking, `token_alignment.rs` for precomputed vocab alignment) + semantic validation for `@grammar` decorator + builtin FFI registration. This is NSL's moat feature — compiled FSMs are structurally impossible in Python.

**Tech Stack:** Rust (codegen + runtime + semantic)

**Spec:** `docs/superpowers/specs/2026-03-15-m44-constrained-decoding-design.md`

**Prerequisites:** M29 (Continuous Batching — serve block, decode loop)

---

## Important: Scope of This Plan

**This plan builds the core FSM construction and runtime stepping infrastructure.** It delivers:
- Abstract `Grammar` type (rules, alternatives, elements) — unified representation
- JSON Schema → Grammar conversion (object, string, integer, number, boolean, null, array, enum, oneOf)
- BNF grammar parsing → Grammar
- Thompson's construction: Grammar → NFA
- Subset construction: NFA → DFA
- Hopcroft minimization: DFA → minimized DFA
- Token alignment: byte-level DFA × BPE vocab → token-level transition table
- CSR compression with bitmask acceleration for transition tables
- Runtime `GrammarFSM` with state stepping and logit masking FFI
- `@grammar` decorator semantic validation
- Codegen: grammar config fields + 6 builtin FFI registrations (`nsl_grammar_*` naming; migrated to `nsl_fsm_*` in M44b to match spec)
- 30+ unit tests covering grammar parsing, NFA/DFA correctness, token alignment, logit masking

**Deferred to M44b:** `generate()` intrinsic codegen integration (actual injection into serve block decode loop), `.rodata` embedding of compiled FSMs, dynamic schema compilation (runtime `JsonSchema.parse()`), `@endpoint(schema=...)` integration, regex pattern → Grammar conversion (needs full regex parser), `format_to_regex` for JSON Schema format strings, integer range regex with min/max constraints, per-request FSM state management in BatchScheduler (replace global Mutex with per-request `FsmState`), FFI naming migration (`nsl_grammar_*` → `nsl_fsm_*` to match spec), `nsl_grammar_load_table` FFI for populating FSM from serialized data, `parse_bnf_elements` full BNF syntax support (grouping with `()`, inline regex `/pattern/`), E2E quality tests.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/grammar_compiler.rs` | Grammar, NFA, DFA types; Thompson's construction; subset construction; Hopcroft minimization | 450 |
| `crates/nsl-codegen/src/schema_convert.rs` | JSON Schema → Grammar; BNF parsing → Grammar | 300 |
| `crates/nsl-runtime/src/grammar.rs` | `GrammarFSM`, token-level DFA, CSR compression, logit masking FFI | 350 |
| `crates/nsl-runtime/src/token_alignment.rs` | Byte-DFA × vocab → token transition table, bitmask generation | 200 |
| `crates/nsl-semantic/src/grammar.rs` | `@grammar` decorator validation | 80 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod grammar_compiler; pub mod schema_convert;` |
| `crates/nsl-codegen/src/compiler.rs` | Add `grammar_configs` field |
| `crates/nsl-codegen/src/builtins.rs` | Register 7 new FFI functions |
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod grammar; pub mod token_alignment;` |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod grammar;` |
| `crates/nsl-semantic/src/checker.rs` | Wire `@grammar` validation |

---

## Phase 1: Abstract Grammar + NFA/DFA Construction

### Task 1: Grammar Compiler — Core Types + NFA/DFA

**Files:**
- Create: `crates/nsl-codegen/src/grammar_compiler.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `grammar_compiler.rs` with Grammar types, NFA, DFA, Thompson's construction, subset construction, Hopcroft minimization, and tests**

```rust
// crates/nsl-codegen/src/grammar_compiler.rs
//! M44: Compile-time FSM construction for constrained decoding.
//!
//! Pipeline: Grammar → NFA (Thompson) → DFA (subset construction) → minimized DFA (Hopcroft).

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// Abstract Grammar
// ---------------------------------------------------------------------------

/// Unified grammar representation from JSON Schema, Regex, or BNF.
#[derive(Debug, Clone)]
pub struct Grammar {
    pub rules: Vec<Rule>,
    pub start_rule: String,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub alternatives: Vec<Alternative>,
}

#[derive(Debug, Clone)]
pub struct Alternative {
    pub elements: Vec<GrammarElement>,
}

#[derive(Debug, Clone)]
pub enum GrammarElement {
    /// Exact byte string match: "true", "{", ","
    Literal(String),
    /// Character class: [a-z], [0-9], etc.
    CharClass(Vec<CharRange>),
    /// Reference to another rule by name.
    RuleRef(String),
    /// Repetition: *, +, ?
    Repeat(Box<GrammarElement>, RepeatMode),
    /// Sequence of elements (implicit concatenation).
    Sequence(Vec<GrammarElement>),
    /// Choice between alternatives (|).
    Choice(Vec<GrammarElement>),
    /// Any single byte.
    AnyByte,
}

/// A single byte range [lo, hi] inclusive.
#[derive(Debug, Clone, Copy)]
pub struct CharRange {
    pub lo: u8,
    pub hi: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RepeatMode {
    ZeroOrMore, // *
    OneOrMore,  // +
    Optional,   // ?
}

// ---------------------------------------------------------------------------
// NFA
// ---------------------------------------------------------------------------

pub type StateId = u32;

/// Nondeterministic finite automaton (byte-level).
#[derive(Debug, Clone)]
pub struct NFA {
    pub states: Vec<NFAState>,
    pub start: StateId,
    pub accept: HashSet<StateId>,
}

#[derive(Debug, Clone, Default)]
pub struct NFAState {
    pub transitions: Vec<(TransitionLabel, StateId)>,
}

#[derive(Debug, Clone)]
pub enum TransitionLabel {
    Byte(u8),
    ByteRange(u8, u8),
    Epsilon,
}

impl NFA {
    fn new_state(&mut self) -> StateId {
        let id = self.states.len() as StateId;
        self.states.push(NFAState::default());
        id
    }

    fn add_transition(&mut self, from: StateId, label: TransitionLabel, to: StateId) {
        self.states[from as usize].transitions.push((label, to));
    }

    /// Compute epsilon closure of a set of states.
    fn epsilon_closure(&self, states: &BTreeSet<StateId>) -> BTreeSet<StateId> {
        let mut closure = states.clone();
        let mut worklist: Vec<StateId> = states.iter().copied().collect();
        while let Some(s) = worklist.pop() {
            for (label, target) in &self.states[s as usize].transitions {
                if matches!(label, TransitionLabel::Epsilon) && closure.insert(*target) {
                    worklist.push(*target);
                }
            }
        }
        closure
    }

    /// Compute the set of states reachable from `states` on byte `b`.
    fn move_on_byte(&self, states: &BTreeSet<StateId>, b: u8) -> BTreeSet<StateId> {
        let mut result = BTreeSet::new();
        for &s in states {
            for (label, target) in &self.states[s as usize].transitions {
                match label {
                    TransitionLabel::Byte(byte) if *byte == b => { result.insert(*target); }
                    TransitionLabel::ByteRange(lo, hi) if b >= *lo && b <= *hi => { result.insert(*target); }
                    _ => {}
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Thompson's Construction: Grammar → NFA
// ---------------------------------------------------------------------------

/// NFA fragment with a single start and single accept state.
struct NFAFragment {
    start: StateId,
    accept: StateId,
}

/// Build an NFA from an abstract grammar using Thompson's construction.
///
/// First inlines all rule references by expanding them, then constructs
/// NFA fragments for each grammar element and wires them together.
pub fn grammar_to_nfa(grammar: &Grammar) -> NFA {
    let mut nfa = NFA {
        states: Vec::new(),
        start: 0,
        accept: HashSet::new(),
    };

    // Inline rule references: build a map of rule name → rule body
    let rule_map: HashMap<&str, &[Alternative]> = grammar.rules.iter()
        .map(|r| (r.name.as_str(), r.alternatives.as_slice()))
        .collect();

    // Check for ANY recursion (not just left recursion) — recursive grammars
    // cannot be compiled to finite automata via Thompson's construction.
    if has_recursive_rules(&rule_map) {
        panic!("grammar contains recursive rule references — only non-recursive grammars can be compiled to FSMs");
    }

    let start_rule = rule_map.get(grammar.start_rule.as_str())
        .expect("start rule not found in grammar");

    let frag = build_alternatives_fragment(&mut nfa, start_rule, &rule_map);
    nfa.start = frag.start;
    nfa.accept.insert(frag.accept);
    nfa
}

/// Check if any rule in the grammar is reachable from itself (direct or indirect recursion).
fn has_recursive_rules(rule_map: &HashMap<&str, &[Alternative]>) -> bool {
    fn collects_refs<'a>(elements: &'a [GrammarElement], out: &mut HashSet<&'a str>) {
        for elem in elements {
            match elem {
                GrammarElement::RuleRef(name) => { out.insert(name.as_str()); }
                GrammarElement::Repeat(inner, _) => collects_refs(&[*inner.clone()], out),
                GrammarElement::Sequence(elems) | GrammarElement::Choice(elems) => collects_refs(elems, out),
                _ => {}
            }
        }
    }

    // Build adjacency: rule -> set of rules it references
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (&name, alts) in rule_map {
        let mut refs = HashSet::new();
        for alt in *alts {
            collects_refs(&alt.elements, &mut refs);
        }
        adj.insert(name, refs);
    }

    // DFS cycle detection
    for &start in rule_map.keys() {
        let mut visited = HashSet::new();
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if node == start && visited.contains(&start) {
                return true; // cycle back to start
            }
            if !visited.insert(node) { continue; }
            if let Some(neighbors) = adj.get(node) {
                for &n in neighbors {
                    stack.push(n);
                }
            }
        }
    }
    false

fn build_alternatives_fragment(
    nfa: &mut NFA,
    alternatives: &[Alternative],
    rule_map: &HashMap<&str, &[Alternative]>,
) -> NFAFragment {
    if alternatives.len() == 1 {
        return build_sequence_fragment(nfa, &alternatives[0].elements, rule_map);
    }

    // Choice: new start → epsilon → each alternative → epsilon → new accept
    let start = nfa.new_state();
    let accept = nfa.new_state();

    for alt in alternatives {
        let frag = build_sequence_fragment(nfa, &alt.elements, rule_map);
        nfa.add_transition(start, TransitionLabel::Epsilon, frag.start);
        nfa.add_transition(frag.accept, TransitionLabel::Epsilon, accept);
    }

    NFAFragment { start, accept }
}

fn build_sequence_fragment(
    nfa: &mut NFA,
    elements: &[GrammarElement],
    rule_map: &HashMap<&str, &[Alternative]>,
) -> NFAFragment {
    if elements.is_empty() {
        let s = nfa.new_state();
        return NFAFragment { start: s, accept: s };
    }

    let mut frags: Vec<NFAFragment> = elements.iter()
        .map(|e| build_element_fragment(nfa, e, rule_map))
        .collect();

    // Chain: frag[0].accept → epsilon → frag[1].start → ... → frag[n].accept
    for i in 0..frags.len() - 1 {
        nfa.add_transition(frags[i].accept, TransitionLabel::Epsilon, frags[i + 1].start);
    }

    NFAFragment {
        start: frags[0].start,
        accept: frags.last().unwrap().accept,
    }
}

fn build_element_fragment(
    nfa: &mut NFA,
    element: &GrammarElement,
    rule_map: &HashMap<&str, &[Alternative]>,
) -> NFAFragment {
    match element {
        GrammarElement::Literal(s) => {
            if s.is_empty() {
                let state = nfa.new_state();
                return NFAFragment { start: state, accept: state };
            }
            let start = nfa.new_state();
            let mut current = start;
            for &b in s.as_bytes() {
                let next = nfa.new_state();
                nfa.add_transition(current, TransitionLabel::Byte(b), next);
                current = next;
            }
            NFAFragment { start, accept: current }
        }

        GrammarElement::CharClass(ranges) => {
            let start = nfa.new_state();
            let accept = nfa.new_state();
            for range in ranges {
                if range.lo == range.hi {
                    nfa.add_transition(start, TransitionLabel::Byte(range.lo), accept);
                } else {
                    nfa.add_transition(start, TransitionLabel::ByteRange(range.lo, range.hi), accept);
                }
            }
            NFAFragment { start, accept }
        }

        GrammarElement::AnyByte => {
            let start = nfa.new_state();
            let accept = nfa.new_state();
            nfa.add_transition(start, TransitionLabel::ByteRange(0, 255), accept);
            NFAFragment { start, accept }
        }

        GrammarElement::RuleRef(name) => {
            if let Some(alts) = rule_map.get(name.as_str()) {
                build_alternatives_fragment(nfa, alts, rule_map)
            } else {
                // Unknown rule — produce empty fragment (semantic checker should catch this)
                let s = nfa.new_state();
                NFAFragment { start: s, accept: s }
            }
        }

        GrammarElement::Repeat(inner, mode) => {
            let frag = build_element_fragment(nfa, inner, rule_map);
            let start = nfa.new_state();
            let accept = nfa.new_state();

            match mode {
                RepeatMode::ZeroOrMore => {
                    // start → ε → frag.start; frag.accept → ε → frag.start (loop)
                    // start → ε → accept (skip)
                    nfa.add_transition(start, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(start, TransitionLabel::Epsilon, accept);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, accept);
                }
                RepeatMode::OneOrMore => {
                    // start → ε → frag.start; frag.accept → ε → frag.start (loop)
                    // frag.accept → ε → accept (no skip)
                    nfa.add_transition(start, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, accept);
                }
                RepeatMode::Optional => {
                    // start → ε → frag.start; frag.accept → ε → accept
                    // start → ε → accept (skip)
                    nfa.add_transition(start, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, accept);
                    nfa.add_transition(start, TransitionLabel::Epsilon, accept);
                }
            }
            NFAFragment { start, accept }
        }

        GrammarElement::Sequence(elements) => {
            build_sequence_fragment(nfa, elements, rule_map)
        }

        GrammarElement::Choice(choices) => {
            let start = nfa.new_state();
            let accept = nfa.new_state();
            for choice in choices {
                let frag = build_element_fragment(nfa, choice, rule_map);
                nfa.add_transition(start, TransitionLabel::Epsilon, frag.start);
                nfa.add_transition(frag.accept, TransitionLabel::Epsilon, accept);
            }
            NFAFragment { start, accept }
        }
    }
}

// ---------------------------------------------------------------------------
// Subset Construction: NFA → DFA
// ---------------------------------------------------------------------------

/// Deterministic finite automaton (byte-level, 256 transitions per state).
#[derive(Debug, Clone)]
pub struct DFA {
    /// transitions[state][byte] = Option<next_state>
    pub transitions: Vec<[Option<StateId>; 256]>,
    pub start: StateId,
    pub accept: HashSet<StateId>,
}

impl DFA {
    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }

    /// Check if a byte string is accepted by this DFA.
    pub fn accepts(&self, input: &[u8]) -> bool {
        let mut state = self.start;
        for &b in input {
            match self.transitions[state as usize][b as usize] {
                Some(next) => state = next,
                None => return false,
            }
        }
        self.accept.contains(&state)
    }
}

/// Convert NFA to DFA via powerset/subset construction.
pub fn nfa_to_dfa(nfa: &NFA) -> DFA {
    let mut dfa_transitions: Vec<[Option<StateId>; 256]> = Vec::new();
    let mut dfa_accept = HashSet::new();

    // Map from NFA state sets → DFA state IDs
    let mut state_map: HashMap<BTreeSet<StateId>, StateId> = HashMap::new();
    let mut worklist: VecDeque<BTreeSet<StateId>> = VecDeque::new();

    // Start state = epsilon closure of NFA start
    let start_set = nfa.epsilon_closure(&{
        let mut s = BTreeSet::new();
        s.insert(nfa.start);
        s
    });

    let start_id = 0u32;
    state_map.insert(start_set.clone(), start_id);
    dfa_transitions.push([None; 256]);
    worklist.push_back(start_set.clone());

    if start_set.iter().any(|s| nfa.accept.contains(s)) {
        dfa_accept.insert(start_id);
    }

    while let Some(current_set) = worklist.pop_front() {
        let current_id = state_map[&current_set];

        for byte in 0..=255u8 {
            let moved = nfa.move_on_byte(&current_set, byte);
            if moved.is_empty() {
                continue;
            }
            let closed = nfa.epsilon_closure(&moved);
            if closed.is_empty() {
                continue;
            }

            let target_id = if let Some(&existing) = state_map.get(&closed) {
                existing
            } else {
                let new_id = dfa_transitions.len() as StateId;
                state_map.insert(closed.clone(), new_id);
                dfa_transitions.push([None; 256]);
                worklist.push_back(closed.clone());

                if closed.iter().any(|s| nfa.accept.contains(s)) {
                    dfa_accept.insert(new_id);
                }
                new_id
            };

            dfa_transitions[current_id as usize][byte as usize] = Some(target_id);
        }
    }

    DFA {
        transitions: dfa_transitions,
        start: start_id,
        accept: dfa_accept,
    }
}

// ---------------------------------------------------------------------------
// Hopcroft Minimization: DFA → minimized DFA
// ---------------------------------------------------------------------------

/// Minimize a DFA using Hopcroft's algorithm.
pub fn minimize_dfa(dfa: &DFA) -> DFA {
    let n = dfa.num_states();
    if n <= 1 {
        return dfa.clone();
    }

    // Initial partition: accept states vs non-accept states
    let accept_set: HashSet<StateId> = dfa.accept.clone();
    let non_accept_set: HashSet<StateId> = (0..n as StateId).filter(|s| !accept_set.contains(s)).collect();

    let mut partitions: Vec<HashSet<StateId>> = Vec::new();
    if !accept_set.is_empty() { partitions.push(accept_set); }
    if !non_accept_set.is_empty() { partitions.push(non_accept_set); }

    let mut changed = true;
    while changed {
        changed = false;
        let mut new_partitions = Vec::new();

        for partition in &partitions {
            if partition.len() <= 1 {
                new_partitions.push(partition.clone());
                continue;
            }

            // Try to split this partition
            let representative = *partition.iter().next().unwrap();
            let mut same = HashSet::new();
            let mut different = HashSet::new();
            same.insert(representative);

            for &state in partition {
                if state == representative { continue; }

                let mut equivalent = true;
                'byte_loop: for byte in 0..=255u8 {
                    let target_rep = dfa.transitions[representative as usize][byte as usize];
                    let target_state = dfa.transitions[state as usize][byte as usize];

                    let part_rep = target_rep.and_then(|t| partitions.iter().position(|p| p.contains(&t)));
                    let part_state = target_state.and_then(|t| partitions.iter().position(|p| p.contains(&t)));

                    if part_rep != part_state {
                        equivalent = false;
                        break 'byte_loop;
                    }
                }

                if equivalent {
                    same.insert(state);
                } else {
                    different.insert(state);
                }
            }

            if !different.is_empty() {
                changed = true;
                new_partitions.push(same);
                new_partitions.push(different);
            } else {
                new_partitions.push(partition.clone());
            }
        }

        partitions = new_partitions;
    }

    // Build minimized DFA
    let state_to_partition: HashMap<StateId, usize> = partitions.iter()
        .enumerate()
        .flat_map(|(i, p)| p.iter().map(move |&s| (s, i)))
        .collect();

    let num_new_states = partitions.len();
    let mut new_transitions = vec![[None; 256]; num_new_states];
    let mut new_accept = HashSet::new();

    for (i, partition) in partitions.iter().enumerate() {
        let representative = *partition.iter().next().unwrap();

        if dfa.accept.contains(&representative) {
            new_accept.insert(i as StateId);
        }

        for byte in 0..=255u8 {
            if let Some(target) = dfa.transitions[representative as usize][byte as usize] {
                new_transitions[i][byte as usize] = Some(state_to_partition[&target] as StateId);
            }
        }
    }

    let new_start = state_to_partition[&dfa.start] as StateId;

    DFA {
        transitions: new_transitions,
        start: new_start,
        accept: new_accept,
    }
}

// ---------------------------------------------------------------------------
// Full pipeline: Grammar → minimized DFA
// ---------------------------------------------------------------------------

/// Compile a grammar to a minimized DFA.
pub fn compile_grammar(grammar: &Grammar) -> DFA {
    let nfa = grammar_to_nfa(grammar);
    let dfa = nfa_to_dfa(&nfa);
    minimize_dfa(&dfa)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn literal_grammar(s: &str) -> Grammar {
        Grammar {
            rules: vec![Rule {
                name: "start".into(),
                alternatives: vec![Alternative {
                    elements: vec![GrammarElement::Literal(s.into())],
                }],
            }],
            start_rule: "start".into(),
        }
    }

    fn choice_grammar(choices: &[&str]) -> Grammar {
        Grammar {
            rules: vec![Rule {
                name: "start".into(),
                alternatives: choices.iter().map(|s| Alternative {
                    elements: vec![GrammarElement::Literal(s.to_string())],
                }).collect(),
            }],
            start_rule: "start".into(),
        }
    }

    #[test]
    fn literal_string_accepted() {
        let g = literal_grammar("hello");
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b"hello"));
        assert!(!dfa.accepts(b"hell"));
        assert!(!dfa.accepts(b"helloo"));
        assert!(!dfa.accepts(b"world"));
        assert!(!dfa.accepts(b""));
    }

    #[test]
    fn choice_between_literals() {
        let g = choice_grammar(&["true", "false", "null"]);
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b"true"));
        assert!(dfa.accepts(b"false"));
        assert!(dfa.accepts(b"null"));
        assert!(!dfa.accepts(b"tru"));
        assert!(!dfa.accepts(b"TRUE"));
    }

    #[test]
    fn char_class_digits() {
        let g = Grammar {
            rules: vec![Rule {
                name: "start".into(),
                alternatives: vec![Alternative {
                    elements: vec![GrammarElement::Repeat(
                        Box::new(GrammarElement::CharClass(vec![CharRange { lo: b'0', hi: b'9' }])),
                        RepeatMode::OneOrMore,
                    )],
                }],
            }],
            start_rule: "start".into(),
        };
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b"0"));
        assert!(dfa.accepts(b"123"));
        assert!(dfa.accepts(b"999999"));
        assert!(!dfa.accepts(b""));
        assert!(!dfa.accepts(b"12a3"));
    }

    #[test]
    fn optional_element() {
        // Grammar: "a" "b"?
        let g = Grammar {
            rules: vec![Rule {
                name: "start".into(),
                alternatives: vec![Alternative {
                    elements: vec![
                        GrammarElement::Literal("a".into()),
                        GrammarElement::Repeat(
                            Box::new(GrammarElement::Literal("b".into())),
                            RepeatMode::Optional,
                        ),
                    ],
                }],
            }],
            start_rule: "start".into(),
        };
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b"a"));
        assert!(dfa.accepts(b"ab"));
        assert!(!dfa.accepts(b"abb"));
        assert!(!dfa.accepts(b"b"));
    }

    #[test]
    fn zero_or_more_repetition() {
        // Grammar: "a"*
        let g = Grammar {
            rules: vec![Rule {
                name: "start".into(),
                alternatives: vec![Alternative {
                    elements: vec![GrammarElement::Repeat(
                        Box::new(GrammarElement::Literal("a".into())),
                        RepeatMode::ZeroOrMore,
                    )],
                }],
            }],
            start_rule: "start".into(),
        };
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b""));
        assert!(dfa.accepts(b"a"));
        assert!(dfa.accepts(b"aaa"));
        assert!(!dfa.accepts(b"b"));
    }

    #[test]
    fn rule_reference() {
        // Grammar: start -> digit digit; digit -> [0-9]
        let g = Grammar {
            rules: vec![
                Rule {
                    name: "start".into(),
                    alternatives: vec![Alternative {
                        elements: vec![
                            GrammarElement::RuleRef("digit".into()),
                            GrammarElement::RuleRef("digit".into()),
                        ],
                    }],
                },
                Rule {
                    name: "digit".into(),
                    alternatives: vec![Alternative {
                        elements: vec![GrammarElement::CharClass(vec![CharRange { lo: b'0', hi: b'9' }])],
                    }],
                },
            ],
            start_rule: "start".into(),
        };
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b"42"));
        assert!(dfa.accepts(b"00"));
        assert!(!dfa.accepts(b"1"));
        assert!(!dfa.accepts(b"123"));
    }

    #[test]
    fn json_boolean_grammar() {
        let g = choice_grammar(&["true", "false"]);
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b"true"));
        assert!(dfa.accepts(b"false"));
        assert!(!dfa.accepts(b"True"));
        // Minimized DFA should have fewer states than unminimized
        assert!(dfa.num_states() <= 10);
    }

    #[test]
    fn minimization_reduces_states() {
        // Large grammar that benefits from minimization
        let g = choice_grammar(&["aa", "ab", "ba", "bb"]);
        let nfa = grammar_to_nfa(&g);
        let unminimized = nfa_to_dfa(&nfa);
        let minimized = minimize_dfa(&unminimized);
        assert!(minimized.num_states() <= unminimized.num_states());
        // Both should accept the same strings
        for s in [b"aa".as_slice(), b"ab", b"ba", b"bb", b"a", b"b", b"abc"] {
            assert_eq!(unminimized.accepts(s), minimized.accepts(s), "mismatch on {:?}", s);
        }
    }

    #[test]
    fn empty_grammar_element() {
        let g = Grammar {
            rules: vec![Rule {
                name: "start".into(),
                alternatives: vec![Alternative { elements: vec![] }],
            }],
            start_rule: "start".into(),
        };
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b""));
        assert!(!dfa.accepts(b"a"));
    }

    #[test]
    fn nested_sequence_and_choice() {
        // Grammar: ("a" | "b") "c"
        let g = Grammar {
            rules: vec![Rule {
                name: "start".into(),
                alternatives: vec![Alternative {
                    elements: vec![
                        GrammarElement::Choice(vec![
                            GrammarElement::Literal("a".into()),
                            GrammarElement::Literal("b".into()),
                        ]),
                        GrammarElement::Literal("c".into()),
                    ],
                }],
            }],
            start_rule: "start".into(),
        };
        let dfa = compile_grammar(&g);
        assert!(dfa.accepts(b"ac"));
        assert!(dfa.accepts(b"bc"));
        assert!(!dfa.accepts(b"cc"));
        assert!(!dfa.accepts(b"a"));
    }
}
```

Add to `crates/nsl-codegen/src/lib.rs`:
```rust
pub mod grammar_compiler;
pub mod schema_convert;
```

### Task 2: Schema Conversion — JSON Schema + BNF → Grammar

**Files:**
- Create: `crates/nsl-codegen/src/schema_convert.rs`

- [ ] **Step 2: Create `schema_convert.rs` with JSON Schema → Grammar and BNF → Grammar converters**

This module converts JSON schemas and BNF grammar strings into the abstract `Grammar` type. The JSON Schema converter handles: object, string, integer, number, boolean, null, array, enum, oneOf. The BNF parser handles `rule: alt1 | alt2` syntax with quoted literals and rule references.

Key functions:
- `json_schema_to_grammar(schema: &serde_json::Value) -> Result<Grammar, GrammarError>` — spec Section 2 conversion
- `parse_bnf_grammar(text: &str) -> Result<Grammar, GrammarError>` — spec Section 2 BNF parsing

Tests:
- `test_json_schema_boolean` — `{"type":"boolean"}` → grammar accepting "true" | "false"
- `test_json_schema_integer` — `{"type":"integer"}` → grammar accepting digits
- `test_json_schema_string` — `{"type":"string"}` → grammar accepting quoted strings
- `test_json_schema_object` — simple `{"type":"object","properties":{...}}` → grammar with key-value pairs
- `test_json_schema_enum` — `{"enum":["a","b","c"]}` → choice grammar
- `test_bnf_simple` — `start: "true" | "false"` parses correctly
- `test_bnf_with_rule_refs` — multi-rule grammar with references resolves

**Important:** The `GrammarElement` enum intentionally omits `Regex(String)` — regex → Grammar conversion is deferred to M44b. For M44a, `integer` and `number` types use manual grammar element expansion:
- `integer`: `Sequence([Repeat(Literal("-"), Optional), Repeat(CharClass([0-9]), OneOrMore)])`
- `number`: `Sequence([integer_part, Repeat(Sequence([Literal("."), digits]), Optional)])`

This avoids needing a regex parser while supporting the most common JSON Schema types.

**Note:** This module will depend on `serde_json` for JSON Schema parsing. Check if it's already in the `nsl-codegen` crate's `Cargo.toml` dependencies; if not, add it.

---

## Phase 2: Token Alignment + Runtime

### Task 3: Token Alignment

**Files:**
- Create: `crates/nsl-runtime/src/token_alignment.rs`

- [ ] **Step 3: Create `token_alignment.rs` with byte-DFA × vocab → token transition table**

```rust
// crates/nsl-runtime/src/token_alignment.rs
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
/// 2. If all bytes are valid transitions and the final state exists → token is valid
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
    let mask_words = (vocab_size + 63) / 64; // u64 words for bitmask

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
                    None => { valid = false; break; }
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
        self.state_transitions[state].iter()
            .find(|t| t.token_id == token_id as u32)
            .map(|t| t.next_state)
    }

    /// Count valid tokens for a state.
    pub fn valid_count(&self, state: usize) -> usize {
        self.state_masks[state].iter()
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
        let (trans, accept) = make_simple_dfa();
        let vocab = vec![
            b"a".to_vec(),   // token 0
            b"b".to_vec(),   // token 1
            b"c".to_vec(),   // token 2
            b"ab".to_vec(),  // token 3
            b"ac".to_vec(),  // token 4
            b"d".to_vec(),   // token 5 (invalid for this DFA)
        ];

        let table = build_token_alignment(&vocab, &trans, 4);

        // From state 0: only "a" (token 0) is valid (leads to state 1)
        assert!(table.is_valid(0, 0)); // "a"
        assert!(!table.is_valid(0, 1)); // "b" not valid from state 0
        assert!(!table.is_valid(0, 5)); // "d" not valid

        // Multi-byte tokens: "ab" (token 3) from state 0 should be valid (walks a→1, b→2)
        assert!(table.is_valid(0, 3)); // "ab"
        assert!(table.is_valid(0, 4)); // "ac"

        // From state 1: "b" (token 1) and "c" (token 2) are valid
        assert!(table.is_valid(1, 1)); // "b"
        assert!(table.is_valid(1, 2)); // "c"
        assert!(!table.is_valid(1, 0)); // "a" not valid from state 1
    }

    #[test]
    fn next_state_lookup() {
        let (trans, accept) = make_simple_dfa();
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"ab".to_vec()];
        let table = build_token_alignment(&vocab, &trans, 4);

        assert_eq!(table.next_state(0, 0), Some(1)); // "a" from state 0 → state 1
        assert_eq!(table.next_state(0, 2), Some(2)); // "ab" from state 0 → state 2
        assert_eq!(table.next_state(1, 1), Some(2)); // "b" from state 1 → state 2
    }

    #[test]
    fn valid_count() {
        let (trans, accept) = make_simple_dfa();
        let vocab = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"ab".to_vec(), b"ac".to_vec()];
        let table = build_token_alignment(&vocab, &trans, 4);

        assert_eq!(table.valid_count(0), 3); // a, ab, ac
        assert_eq!(table.valid_count(1), 2); // b, c
    }

    #[test]
    fn empty_token_skipped() {
        let (trans, accept) = make_simple_dfa();
        let vocab = vec![b"".to_vec(), b"a".to_vec()];
        let table = build_token_alignment(&vocab, &trans, 4);
        assert!(!table.is_valid(0, 0)); // empty token skipped
        assert!(table.is_valid(0, 1));  // "a" valid
    }
}
```

### Task 4: Runtime Grammar FSM + Logit Masking

**Files:**
- Create: `crates/nsl-runtime/src/grammar.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 4: Create `grammar.rs` with GrammarFSM, CSR compression, and logit masking FFI**

```rust
// crates/nsl-runtime/src/grammar.rs
//! M44: Runtime FSM for constrained decoding.
//!
//! Stores a compiled token-level DFA in CSR (compressed sparse row) format
//! for O(1) state stepping and efficient logit masking.

use std::sync::Mutex;

// ---------------------------------------------------------------------------
// CSR-compressed token-level DFA
// ---------------------------------------------------------------------------

/// Compiled grammar FSM in CSR format for runtime use.
///
/// CSR = Compressed Sparse Row: for each state, a contiguous slice of
/// (token_id, next_state) pairs. Bitmasks for fast logit masking.
pub struct GrammarFSM {
    /// Number of DFA states.
    pub num_states: usize,
    /// CSR row pointers: state i's transitions are in
    /// `transitions[row_ptr[i]..row_ptr[i+1]]`.
    pub row_ptr: Vec<usize>,
    /// CSR data: (token_id, next_state) pairs, sorted by token_id.
    pub transitions: Vec<(u32, u32)>,
    /// Per-state bitmask of valid token IDs. [num_states][mask_words].
    /// mask_words = ceil(vocab_size / 64).
    pub masks: Vec<Vec<u64>>,
    /// Start state.
    pub start_state: u32,
    /// Accept states.
    pub accept_states: Vec<u32>,
    /// Vocabulary size.
    pub vocab_size: usize,
}

impl GrammarFSM {
    /// Create from a token alignment table.
    pub fn from_alignment(
        table: &crate::token_alignment::TokenAlignmentTable,
        start_state: u32,
        accept_states: Vec<u32>,
    ) -> Self {
        let mut row_ptr = Vec::with_capacity(table.num_states + 1);
        let mut transitions = Vec::new();

        for state in 0..table.num_states {
            row_ptr.push(transitions.len());
            for t in &table.state_transitions[state] {
                transitions.push((t.token_id, t.next_state));
            }
        }
        row_ptr.push(transitions.len());

        // Verify CSR sort invariant (binary search depends on sorted token_ids)
        for state in 0..table.num_states {
            let start = row_ptr[state];
            let end = *row_ptr.last().unwrap_or(&0);
            debug_assert!(
                transitions[start..end.min(transitions.len())]
                    .windows(2)
                    .all(|w| w[0].0 < w[1].0),
                "token transitions must be sorted by token_id for binary search"
            );
        }

        GrammarFSM {
            num_states: table.num_states,
            row_ptr,
            transitions,
            masks: table.state_masks.clone(),
            start_state,
            accept_states,
            vocab_size: table.vocab_size,
        }
    }

    /// Step the FSM: given current state and token_id, return next state or None.
    pub fn step(&self, state: u32, token_id: u32) -> Option<u32> {
        let start = self.row_ptr[state as usize];
        let end = self.row_ptr[state as usize + 1];
        let slice = &self.transitions[start..end];

        // Binary search since transitions are sorted by token_id
        slice.binary_search_by_key(&token_id, |&(tid, _)| tid)
            .ok()
            .map(|idx| slice[idx].1)
    }

    /// Check if a token is valid in the current state.
    pub fn is_valid_token(&self, state: u32, token_id: u32) -> bool {
        if state as usize >= self.num_states || token_id as usize >= self.vocab_size {
            return false;
        }
        let word = token_id as usize / 64;
        let bit = token_id as usize % 64;
        (self.masks[state as usize][word] >> bit) & 1 == 1
    }

    /// Apply logit mask: set logits of invalid tokens to -inf.
    ///
    /// `logits`: mutable f32 slice of length vocab_size.
    /// `state`: current FSM state.
    pub fn apply_logit_mask(&self, logits: &mut [f32], state: u32) {
        if (state as usize) >= self.num_states {
            // Invalid state — mask everything as safety fallback
            for logit in logits.iter_mut() {
                *logit = f32::NEG_INFINITY;
            }
            return;
        }
        let masks = &self.masks[state as usize];
        for (token_id, logit) in logits.iter_mut().enumerate() {
            let word = token_id / 64;
            let bit = token_id % 64;
            if word < masks.len() && (masks[word] >> bit) & 1 == 0 {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    /// Check if current state is an accept state.
    pub fn is_accept(&self, state: u32) -> bool {
        self.accept_states.contains(&state)
    }
}

// ---------------------------------------------------------------------------
// Global context + FFI
// ---------------------------------------------------------------------------

static GRAMMAR_CTX: Mutex<Option<GrammarContext>> = Mutex::new(None);

struct GrammarContext {
    fsm: GrammarFSM,
}

/// Initialize the grammar FSM from a pre-compiled token alignment table.
///
/// In M44a, the FSM is built from a precomputed alignment table passed
/// as a serialized buffer. In M44b, this will read from .rodata.
///
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_grammar_init(
    num_states: i64,
    vocab_size: i64,
    start_state: i64,
) -> i64 {
    let mut guard = GRAMMAR_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    let mask_words = (vocab_size as usize + 63) / 64;
    *guard = Some(GrammarContext {
        fsm: GrammarFSM {
            num_states: num_states as usize,
            row_ptr: vec![0; num_states as usize + 1],
            transitions: Vec::new(),
            masks: vec![vec![0u64; mask_words]; num_states as usize],
            start_state: start_state as u32,
            accept_states: Vec::new(),
            vocab_size: vocab_size as usize,
        },
    });
    0
}

/// Step the FSM: advance from current state with the given token.
/// Returns the next state, or -1 if the token is invalid.
#[no_mangle]
pub extern "C" fn nsl_grammar_step(state: i64, token_id: i64) -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_grammar_init not called");
    ctx.fsm.step(state as u32, token_id as u32)
        .map(|s| s as i64)
        .unwrap_or(-1)
}

/// Apply logit mask for the given FSM state to the logits tensor.
///
/// `logits_ptr`: pointer to f32 array of length vocab_size.
/// `state`: current FSM state.
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_grammar_apply_mask(logits_ptr: i64, state: i64) -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_grammar_init not called");

    if logits_ptr == 0 {
        return -1;
    }

    let logits = unsafe {
        std::slice::from_raw_parts_mut(logits_ptr as *mut f32, ctx.fsm.vocab_size)
    };
    ctx.fsm.apply_logit_mask(logits, state as u32);
    0
}

/// Check if the current state is an accepting state.
/// Returns 1 if accepting, 0 if not.
#[no_mangle]
pub extern "C" fn nsl_grammar_is_accept(state: i64) -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_grammar_init not called");
    if ctx.fsm.is_accept(state as u32) { 1 } else { 0 }
}

/// Get the start state of the FSM.
#[no_mangle]
pub extern "C" fn nsl_grammar_start_state() -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_grammar_init not called");
    ctx.fsm.start_state as i64
}

/// Destroy the grammar context.
#[no_mangle]
pub extern "C" fn nsl_grammar_destroy() -> i64 {
    let mut guard = GRAMMAR_CTX.lock().unwrap();
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_alignment::{build_token_alignment, TokenTransition};

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_grammar_destroy();
        guard
    }

    fn make_test_fsm() -> GrammarFSM {
        // Simple FSM: state 0 -(token 0)-> state 1 (accept), state 0 -(token 1)-> state 2
        let mut table = crate::token_alignment::TokenAlignmentTable {
            num_states: 3,
            state_transitions: vec![
                vec![
                    TokenTransition { token_id: 0, next_state: 1 },
                    TokenTransition { token_id: 1, next_state: 2 },
                ],
                vec![],
                vec![TokenTransition { token_id: 0, next_state: 1 }],
            ],
            state_masks: vec![vec![0b11; 1], vec![0; 1], vec![0b01; 1]],
            vocab_size: 4,
        };
        // Recalculate masks properly
        table.state_masks = vec![vec![0u64; 1]; 3];
        table.state_masks[0][0] = 0b0011; // tokens 0, 1 valid
        table.state_masks[2][0] = 0b0001; // token 0 valid

        GrammarFSM::from_alignment(&table, 0, vec![1])
    }

    #[test]
    fn fsm_step() {
        let fsm = make_test_fsm();
        assert_eq!(fsm.step(0, 0), Some(1));
        assert_eq!(fsm.step(0, 1), Some(2));
        assert_eq!(fsm.step(0, 2), None); // invalid token
        assert_eq!(fsm.step(1, 0), None); // no transitions from accept state
    }

    #[test]
    fn fsm_is_valid_token() {
        let fsm = make_test_fsm();
        assert!(fsm.is_valid_token(0, 0));
        assert!(fsm.is_valid_token(0, 1));
        assert!(!fsm.is_valid_token(0, 2));
        assert!(!fsm.is_valid_token(1, 0)); // state 1 has no valid tokens
    }

    #[test]
    fn fsm_logit_mask() {
        let fsm = make_test_fsm();
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        fsm.apply_logit_mask(&mut logits, 0);
        assert_eq!(logits[0], 1.0); // valid
        assert_eq!(logits[1], 2.0); // valid
        assert_eq!(logits[2], f32::NEG_INFINITY); // masked
        assert_eq!(logits[3], f32::NEG_INFINITY); // masked
    }

    #[test]
    fn fsm_accept_state() {
        let fsm = make_test_fsm();
        assert!(!fsm.is_accept(0));
        assert!(fsm.is_accept(1));
        assert!(!fsm.is_accept(2));
    }

    #[test]
    fn ffi_lifecycle() {
        let _lock = setup();

        assert_eq!(nsl_grammar_init(3, 10, 0), 0);
        assert_eq!(nsl_grammar_init(3, 10, 0), -1); // double init
        assert_eq!(nsl_grammar_start_state(), 0);
        assert_eq!(nsl_grammar_destroy(), 0);
    }

    #[test]
    fn ffi_null_logits_returns_error() {
        let _lock = setup();
        assert_eq!(nsl_grammar_init(1, 4, 0), 0);
        assert_eq!(nsl_grammar_apply_mask(0, 0), -1);
        assert_eq!(nsl_grammar_destroy(), 0);
    }
}
```

Add to `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod grammar;
pub mod token_alignment;
```

---

## Phase 3: Semantic Validation + Codegen + Builtins

### Task 5: Semantic Validation

**Files:**
- Create: `crates/nsl-semantic/src/grammar.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 5: Create `grammar.rs` validation module and wire into checker**

```rust
// crates/nsl-semantic/src/grammar.rs
//! M44: @grammar decorator validation.

use nsl_ast::block::Decorator;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validated grammar configuration from a @grammar decorator.
#[derive(Debug, Clone)]
pub struct GrammarConfig {
    pub start_rule: String,
}

/// Validate a @grammar decorator.
///
/// @grammar("start_rule_name") on a function means the function's docstring
/// contains a BNF grammar definition.
pub fn validate_grammar_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<GrammarConfig> {
    let mut start_rule = "start".to_string();

    if let Some(ref args) = deco.args {
        for arg in args {
            // Positional arg: the start rule name
            if arg.name.is_none() {
                if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                    start_rule = s.clone();
                }
            } else if let Some(ref name_sym) = arg.name {
                let key = resolve_sym(*name_sym);
                if key == "start" {
                    if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                        start_rule = s.clone();
                    }
                } else {
                    diagnostics.push(
                        Diagnostic::error(format!("unknown @grammar parameter '{key}'"))
                            .with_label(arg.value.span, "here"),
                    );
                    return None;
                }
            }
        }
    }

    if start_rule.is_empty() {
        diagnostics.push(
            Diagnostic::error("@grammar requires a non-empty start rule name")
                .with_label(deco.span, "here"),
        );
        return None;
    }

    Some(GrammarConfig { start_rule })
}
```

Wire into `checker.rs` decorator dispatch (near `@vmap`, `@moe`):
```rust
if dname == "grammar" {
    let resolve = |s: nsl_ast::Symbol| -> String { self.resolve_sym(s).to_string() };
    crate::grammar::validate_grammar_decorator(deco, &resolve, &mut self.diagnostics);
}
```

Wire into `lib.rs`:
```rust
pub mod grammar;
```

### Task 6: Codegen + Builtins

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

- [ ] **Step 6: Add `grammar_configs` field to Compiler struct**

Add to the Compiler struct:
```rust
/// M44: Compiled grammar FSMs from @grammar decorators and generate() schema= args.
pub grammar_configs: HashMap<String, GrammarInfo>,
```

Where `GrammarInfo` is:
```rust
/// M44: Codegen-side grammar configuration.
pub struct GrammarInfo {
    pub start_rule: String,
    pub grammar_source: String,  // "json_schema", "bnf", "regex"
}
```

Initialize: `grammar_configs: HashMap::new()`.

- [ ] **Step 7: Register FFI functions in builtins.rs**

Add to the `RUNTIME_FUNCTIONS` array:
```rust
// M44: Constrained decoding (grammar FSM)
("nsl_grammar_init", &[I64, I64, I64], Some(I64)),
("nsl_grammar_step", &[I64, I64], Some(I64)),
("nsl_grammar_apply_mask", &[I64, I64], Some(I64)),
("nsl_grammar_is_accept", &[I64], Some(I64)),
("nsl_grammar_start_state", &[], Some(I64)),
("nsl_grammar_destroy", &[], Some(I64)),
```

---

## Phase 4: Build Verification

- [ ] **Step 8: `cargo build` — verify no compile errors**

- [ ] **Step 9: `cargo test` — run all tests, expect 30+ new tests passing**

Expected new tests:
- `grammar_compiler::tests::*` (11 tests: literal, choice, char class, optional, zero_or_more, rule ref, boolean grammar, minimization, empty grammar, nested, accepts/rejects)
- `schema_convert::tests::*` (7 tests: JSON boolean/integer/string/object/enum, BNF simple/rule refs)
- `token_alignment::tests::*` (4 tests: basic alignment, next state, valid count, empty token)
- `grammar::tests::*` (6 tests: FSM step, is_valid, logit mask, accept state, FFI lifecycle, null pointer)

- [ ] **Step 10: `cargo clippy` — no warnings**

---

## Verification Checklist

After implementation, verify:

1. **Grammar compilation**: Literal/choice/repeat/sequence/char-class grammars compile to correct DFAs
2. **DFA correctness**: `dfa.accepts()` matches expected strings, rejects invalid strings
3. **Minimization**: Hopcroft reduces state count while preserving acceptance
4. **JSON Schema**: boolean/integer/string/object types produce correct grammars
5. **BNF parsing**: Multi-rule grammars with alternation parse correctly
6. **Token alignment**: Multi-byte BPE tokens walk through DFA correctly
7. **Logit masking**: Invalid tokens get -inf, valid tokens unchanged
8. **FSM stepping**: Step returns correct next state or None for invalid
9. **FFI**: All 6 functions callable with correct return codes
10. **No regressions**: All existing tests pass
