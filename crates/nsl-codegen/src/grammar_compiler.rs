// crates/nsl-codegen/src/grammar_compiler.rs
//! M44: Compile-time FSM construction for constrained decoding.
//!
//! Pipeline: Grammar -> NFA (Thompson) -> DFA (subset construction) -> minimized DFA (Hopcroft).

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
// Thompson's Construction: Grammar -> NFA
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

    // Inline rule references: build a map of rule name -> rule body
    let rule_map: HashMap<&str, &[Alternative]> = grammar.rules.iter()
        .map(|r| (r.name.as_str(), r.alternatives.as_slice()))
        .collect();

    // Check for ANY recursion (not just left recursion) -- recursive grammars
    // cannot be compiled to finite automata via Thompson's construction.
    if has_recursive_rules(&rule_map) {
        panic!("grammar contains recursive rule references -- only non-recursive grammars can be compiled to FSMs");
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
    fn collects_refs<'a>(element: &'a GrammarElement, out: &mut HashSet<&'a str>) {
        match element {
            GrammarElement::RuleRef(name) => { out.insert(name.as_str()); }
            GrammarElement::Repeat(inner, _) => collects_refs(inner, out),
            GrammarElement::Sequence(elems) | GrammarElement::Choice(elems) => {
                for elem in elems {
                    collects_refs(elem, out);
                }
            }
            _ => {}
        }
    }

    // Build adjacency: rule -> set of rules it references
    let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
    for (&name, alts) in rule_map {
        let mut refs = HashSet::new();
        for alt in *alts {
            for elem in &alt.elements {
                collects_refs(elem, &mut refs);
            }
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
}

fn build_alternatives_fragment(
    nfa: &mut NFA,
    alternatives: &[Alternative],
    rule_map: &HashMap<&str, &[Alternative]>,
) -> NFAFragment {
    if alternatives.len() == 1 {
        return build_sequence_fragment(nfa, &alternatives[0].elements, rule_map);
    }

    // Choice: new start -> epsilon -> each alternative -> epsilon -> new accept
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

    let frags: Vec<NFAFragment> = elements.iter()
        .map(|e| build_element_fragment(nfa, e, rule_map))
        .collect();

    // Chain: frag[0].accept -> epsilon -> frag[1].start -> ... -> frag[n].accept
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
                // Unknown rule -- produce empty fragment (semantic checker should catch this)
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
                    // start -> e -> frag.start; frag.accept -> e -> frag.start (loop)
                    // start -> e -> accept (skip)
                    nfa.add_transition(start, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(start, TransitionLabel::Epsilon, accept);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, accept);
                }
                RepeatMode::OneOrMore => {
                    // start -> e -> frag.start; frag.accept -> e -> frag.start (loop)
                    // frag.accept -> e -> accept (no skip)
                    nfa.add_transition(start, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, frag.start);
                    nfa.add_transition(frag.accept, TransitionLabel::Epsilon, accept);
                }
                RepeatMode::Optional => {
                    // start -> e -> frag.start; frag.accept -> e -> accept
                    // start -> e -> accept (skip)
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
// Subset Construction: NFA -> DFA
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

    // Map from NFA state sets -> DFA state IDs
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
// Hopcroft Minimization: DFA -> minimized DFA
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
// Full pipeline: Grammar -> minimized DFA
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
