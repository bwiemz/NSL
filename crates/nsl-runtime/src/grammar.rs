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
            let end = row_ptr[state + 1];
            debug_assert!(
                transitions[start..end]
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
        slice
            .binary_search_by_key(&token_id, |&(tid, _)| tid)
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
            // Invalid state -- mask everything as safety fallback
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
// Per-request grammar FSM state for constrained decoding (M44b)
// ---------------------------------------------------------------------------

/// Per-request grammar FSM state for constrained decoding.
#[derive(Clone, Debug)]
pub struct GrammarRequestState {
    pub current_state: u32,
    pub active: bool,
}

impl GrammarRequestState {
    pub fn new(start_state: u32) -> Self {
        GrammarRequestState { current_state: start_state, active: true }
    }
}

// ---------------------------------------------------------------------------
// Global context + FFI
// ---------------------------------------------------------------------------

pub(crate) static GRAMMAR_CTX: Mutex<Option<GrammarContext>> = Mutex::new(None);

pub(crate) struct GrammarContext {
    pub(crate) fsm: GrammarFSM,
}

/// Install a compiled grammar as the process's grammar context. Returns
/// false if one is already installed.
///
/// This is how a grammar gets into the runtime. There is no FFI for it:
/// no compiled program carries a token alignment table to hand over.
pub fn install_grammar(fsm: GrammarFSM) -> bool {
    let mut guard = GRAMMAR_CTX.lock().unwrap();
    if guard.is_some() {
        return false;
    }
    *guard = Some(GrammarContext { fsm });
    true
}

/// Refuses: always returns -1 and installs nothing.
///
/// The arguments are three numbers — a state count, a vocabulary size and
/// a start state — and no transitions, so there is no grammar to build
/// from them. This function used to install an automaton with that many
/// states, no transitions and all-zero masks, and return success; every
/// `nsl_grammar_apply_mask` then masked the whole vocabulary and every
/// `nsl_grammar_step` rejected. Nothing compiled calls it. A grammar is
/// installed with [`install_grammar`] from a real alignment table.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_grammar_init(num_states: i64, vocab_size: i64, start_state: i64) -> i64 {
    crate::nsl_log!(ERROR, "grammar",
        "nsl_grammar_init({num_states}, {vocab_size}, {start_state}) refused: it is given no transition table, so it cannot build a grammar (it used to install one that masked every token)"
    );
    -1
}

/// Step the FSM: advance from current state with the given token.
/// Returns the next state, or -1 if the token is invalid, the state is
/// out of range, or no grammar is installed.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_grammar_step(state: i64, token_id: i64) -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };
    if state < 0 || state as usize >= ctx.fsm.num_states || token_id < 0 {
        return -1;
    }
    ctx.fsm
        .step(state as u32, token_id as u32)
        .map(|s| s as i64)
        .unwrap_or(-1)
}

/// Apply logit mask for the given FSM state to the logits tensor.
///
/// `logits_ptr`: pointer to f32 array of length vocab_size.
/// `state`: current FSM state.
///
/// Returns 0 on success, -1 on error (null pointer, or no grammar
/// installed).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_grammar_apply_mask(logits_ptr: i64, state: i64) -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };

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
/// Returns 1 if accepting, 0 if not, -1 if no grammar is installed.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_grammar_is_accept(state: i64) -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    let Some(ctx) = guard.as_ref() else {
        return -1;
    };
    if ctx.fsm.is_accept(state as u32) {
        1
    } else {
        0
    }
}

/// Get the start state of the FSM, or -1 if no grammar is installed.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_grammar_start_state() -> i64 {
    let guard = GRAMMAR_CTX.lock().unwrap();
    match guard.as_ref() {
        Some(ctx) => ctx.fsm.start_state as i64,
        None => -1,
    }
}

/// Destroy the grammar context.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_grammar_destroy() -> i64 {
    let mut guard = GRAMMAR_CTX.lock().unwrap();
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token_alignment::TokenTransition;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_grammar_destroy();
        guard
    }

    fn make_test_fsm() -> GrammarFSM {
        // Simple FSM: state 0 -(token 0)-> state 1 (accept), state 0 -(token 1)-> state 2
        let table = crate::token_alignment::TokenAlignmentTable {
            num_states: 3,
            state_transitions: vec![
                vec![
                    TokenTransition {
                        token_id: 0,
                        next_state: 1,
                    },
                    TokenTransition {
                        token_id: 1,
                        next_state: 2,
                    },
                ],
                vec![],
                vec![TokenTransition {
                    token_id: 0,
                    next_state: 1,
                }],
            ],
            state_masks: vec![
                vec![0b0011], // tokens 0, 1 valid
                vec![0],      // no valid tokens
                vec![0b0001], // token 0 valid
            ],
            vocab_size: 4,
        };

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

        assert!(install_grammar(make_test_fsm()));
        assert!(!install_grammar(make_test_fsm()), "double install");
        assert_eq!(nsl_grammar_start_state(), 0);
        assert_eq!(nsl_grammar_step(0, 1), 2);
        assert_eq!(nsl_grammar_step(0, 2), -1);
        assert_eq!(nsl_grammar_step(99, 0), -1, "a state past the table");
        assert_eq!(nsl_grammar_step(-1, 0), -1);
        assert_eq!(nsl_grammar_is_accept(1), 1);
        assert_eq!(nsl_grammar_destroy(), 0);
    }

    #[test]
    fn ffi_null_logits_returns_error() {
        let _lock = setup();
        assert!(install_grammar(make_test_fsm()));
        assert_eq!(nsl_grammar_apply_mask(0, 0), -1);
        assert_eq!(nsl_grammar_destroy(), 0);
    }

    /// `nsl_grammar_init` cannot build a grammar from three numbers. It
    /// used to install one with no transitions and return 0, after which
    /// the mask for every state removed every token.
    #[test]
    fn init_from_counts_alone_is_refused_and_installs_nothing() {
        let _lock = setup();
        assert_eq!(nsl_grammar_init(3, 4, 0), -1);
        assert!(GRAMMAR_CTX.lock().unwrap().is_none(), "no context was installed");
        // With no grammar the other calls report an error; they used to
        // panic across the FFI boundary with a message that named
        // nsl_grammar_init.
        let mut untouched = [1.0f32, 2.0];
        assert_eq!(nsl_grammar_step(0, 0), -1);
        assert_eq!(nsl_grammar_apply_mask(untouched.as_mut_ptr() as i64, 0), -1);
        assert_eq!(untouched, [1.0, 2.0]);
        assert_eq!(nsl_grammar_is_accept(0), -1);
        assert_eq!(nsl_grammar_start_state(), -1);
        // With a real grammar the mask keeps the tokens the state allows.
        assert!(install_grammar(make_test_fsm()));
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(nsl_grammar_apply_mask(logits.as_mut_ptr() as i64, 0), 0);
        assert_eq!(logits, [1.0, 2.0, f32::NEG_INFINITY, f32::NEG_INFINITY]);
        assert_eq!(nsl_grammar_destroy(), 0);
    }
}
