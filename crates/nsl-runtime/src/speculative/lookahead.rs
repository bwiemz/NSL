//! Lookahead decoding: n-gram based candidate generation without a draft model.
//!
//! Implements Jacobi iteration-style speculation: maintains a pool of observed
//! n-grams from the prompt and generated text, and proposes continuations by
//! matching the current context against the pool. Useful as a fallback when
//! no draft model is available.

use std::collections::HashMap;

/// Lookahead decoding runner — generates speculative candidates from n-gram pools.
pub struct LookaheadRunner {
    /// N-gram size for matching (e.g., 3 = trigram).
    n_gram_size: usize,
    /// Maximum lookahead window (number of candidate tokens to propose).
    lookahead_window: usize,
    /// Pool of observed n-grams: context (n-1 tokens) → list of continuations.
    n_gram_pool: HashMap<Vec<i64>, Vec<i64>>,
}

impl LookaheadRunner {
    /// Create a new lookahead runner.
    pub fn new(n_gram_size: usize, lookahead_window: usize) -> Self {
        Self {
            n_gram_size: n_gram_size.max(2),
            lookahead_window: lookahead_window.max(1),
            n_gram_pool: HashMap::new(),
        }
    }

    /// Seed the n-gram pool from a prompt sequence.
    /// Extracts all n-grams from the prompt and records their continuations.
    pub fn seed_from_prompt(&mut self, prompt_tokens: &[i64]) {
        let n = self.n_gram_size;
        if prompt_tokens.len() < n {
            return;
        }
        for window in prompt_tokens.windows(n) {
            let context = window[..n - 1].to_vec();
            let continuation = window[n - 1];
            self.n_gram_pool
                .entry(context)
                .or_default()
                .push(continuation);
        }
    }

    /// Record a newly generated token for future n-gram matching.
    /// `recent_context` should be the last (n-1) tokens before this token.
    pub fn record_token(&mut self, recent_context: &[i64], token: i64) {
        let n = self.n_gram_size;
        if recent_context.len() >= n - 1 {
            let context = recent_context[recent_context.len() - (n - 1)..].to_vec();
            self.n_gram_pool.entry(context).or_default().push(token);
        }
    }

    /// Generate speculative candidates by n-gram matching.
    ///
    /// Given the current context (recent tokens), looks up matching n-grams
    /// and chains continuations to build a candidate sequence.
    ///
    /// Returns up to `lookahead_window` candidate token IDs.
    pub fn generate_candidates(&self, context: &[i64]) -> Vec<i64> {
        let n = self.n_gram_size;
        if context.len() < n - 1 {
            return Vec::new();
        }

        let mut candidates = Vec::with_capacity(self.lookahead_window);
        let mut current_ctx = context[context.len() - (n - 1)..].to_vec();

        for _ in 0..self.lookahead_window {
            match self.n_gram_pool.get(&current_ctx) {
                Some(continuations) if !continuations.is_empty() => {
                    // Pick the most recent continuation (last seen = most likely relevant)
                    let token = *continuations.last().unwrap();
                    candidates.push(token);
                    // Advance context window
                    current_ctx.remove(0);
                    current_ctx.push(token);
                }
                _ => break, // no match — stop extending
            }
        }

        candidates
    }

    /// Get the number of n-grams in the pool.
    pub fn pool_size(&self) -> usize {
        self.n_gram_pool.len()
    }

    /// Clear the n-gram pool.
    pub fn clear(&mut self) {
        self.n_gram_pool.clear();
    }
}

// ---------------------------------------------------------------------------
// FFI entry points
// ---------------------------------------------------------------------------

/// Create a new LookaheadRunner. Returns pointer as i64.
#[no_mangle]
pub extern "C" fn nsl_lookahead_init(ngram_size: i64, window: i64) -> i64 {
    let runner = Box::new(LookaheadRunner::new(ngram_size as usize, window as usize));
    Box::into_raw(runner) as i64
}

/// Seed the n-gram pool from prompt tokens.
#[no_mangle]
pub extern "C" fn nsl_lookahead_seed(runner_ptr: i64, tokens_ptr: i64, num_tokens: i64) -> i64 {
    if runner_ptr == 0 || tokens_ptr == 0 { return -1; }
    let runner = unsafe { &mut *(runner_ptr as *mut LookaheadRunner) };
    let tokens = unsafe {
        std::slice::from_raw_parts(tokens_ptr as *const i64, num_tokens as usize)
    };
    runner.seed_from_prompt(tokens);
    0
}

/// Generate candidates from the current context.
/// Writes candidates to `out_ptr` and returns the number written.
#[no_mangle]
pub extern "C" fn nsl_lookahead_generate(
    runner_ptr: i64,
    context_ptr: i64,
    context_len: i64,
    out_ptr: i64,
    max_out: i64,
) -> i64 {
    if runner_ptr == 0 || context_ptr == 0 || out_ptr == 0 { return 0; }
    let runner = unsafe { &*(runner_ptr as *const LookaheadRunner) };
    let context = unsafe {
        std::slice::from_raw_parts(context_ptr as *const i64, context_len as usize)
    };
    let candidates = runner.generate_candidates(context);
    let n = candidates.len().min(max_out as usize);
    let out = out_ptr as *mut i64;
    for (i, &tok) in candidates.iter().take(n).enumerate() {
        unsafe { *out.add(i) = tok; }
    }
    n as i64
}

/// Destroy a LookaheadRunner.
#[no_mangle]
pub extern "C" fn nsl_lookahead_destroy(runner_ptr: i64) -> i64 {
    if runner_ptr == 0 { return 0; }
    unsafe { drop(Box::from_raw(runner_ptr as *mut LookaheadRunner)); }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_and_generate() {
        let mut runner = LookaheadRunner::new(3, 5);
        // Prompt: [1, 2, 3, 4, 5]
        // 3-grams: [1,2]→3, [2,3]→4, [3,4]→5
        runner.seed_from_prompt(&[1, 2, 3, 4, 5]);
        assert_eq!(runner.pool_size(), 3);

        // Context ends with [3, 4] → should predict 5
        let candidates = runner.generate_candidates(&[1, 2, 3, 4]);
        assert_eq!(candidates, vec![5]);
    }

    #[test]
    fn test_chain_prediction() {
        let mut runner = LookaheadRunner::new(3, 5);
        runner.seed_from_prompt(&[1, 2, 3, 4, 5, 6]);
        // Context [3, 4] → 5, then [4, 5] → 6
        let candidates = runner.generate_candidates(&[1, 2, 3, 4]);
        assert_eq!(candidates, vec![5, 6]);
    }

    #[test]
    fn test_no_match_returns_empty() {
        let mut runner = LookaheadRunner::new(3, 5);
        runner.seed_from_prompt(&[1, 2, 3]);
        // Context [99, 100] → no match
        let candidates = runner.generate_candidates(&[99, 100]);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_record_token() {
        let mut runner = LookaheadRunner::new(3, 5);
        runner.record_token(&[10, 20], 30);
        let candidates = runner.generate_candidates(&[10, 20]);
        assert_eq!(candidates, vec![30]);
    }

    #[test]
    fn test_window_limit() {
        let mut runner = LookaheadRunner::new(2, 2); // bigram, window=2
        runner.seed_from_prompt(&[1, 2, 3, 4, 5]);
        // [1]→2, [2]→3, [3]→4, [4]→5
        // Context [3] → 4, [4] → 5, but window=2 so max 2 candidates
        let candidates = runner.generate_candidates(&[3]);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates, vec![4, 5]);
    }

    #[test]
    fn test_ffi_lifecycle() {
        let runner = nsl_lookahead_init(3, 5);
        assert_ne!(runner, 0);

        let tokens: Vec<i64> = vec![1, 2, 3, 4, 5];
        nsl_lookahead_seed(runner, tokens.as_ptr() as i64, tokens.len() as i64);

        let context: Vec<i64> = vec![3, 4];
        let mut out = vec![0i64; 5];
        let n = nsl_lookahead_generate(
            runner, context.as_ptr() as i64, context.len() as i64,
            out.as_mut_ptr() as i64, out.len() as i64,
        );
        assert_eq!(n, 1);
        assert_eq!(out[0], 5);

        nsl_lookahead_destroy(runner);
    }
}
