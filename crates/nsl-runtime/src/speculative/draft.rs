//! M33b: Draft model runner — autoregressive token generation for speculative decoding.
//!
//! The draft runner generates K candidate tokens by running a small model's
//! forward pass autoregressively. The resulting token sequence and logits
//! are then passed to `nsl_speculative_decode_step` for rejection sampling
//! against the target (verifier) model.
//!
//! Two modes:
//! - **Standard**: Separate draft model, K autoregressive steps
//! - **Medusa**: Single backbone + multiple prediction heads, 1 forward pass

use std::os::raw::c_void;
use std::sync::atomic::AtomicI64;

use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// DraftModelRunner — autoregressive draft generation
// ---------------------------------------------------------------------------

/// Runs a compiled draft model autoregressively to produce K candidate tokens.
///
/// The `forward_fn` is a pointer to a compiled NSL function with signature:
///   fn(input_tokens: NslTensor*, kv_cache: i64, seq_pos: i64) -> NslTensor*
/// where the returned tensor is [1, vocab_size] logits.
pub struct DraftModelRunner {
    /// Function pointer to the compiled draft model forward pass.
    /// Signature: (input_token_tensor_ptr, kv_cache_handle, seq_position) -> logits_tensor_ptr
    forward_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// Number of draft tokens to generate per speculative step.
    num_tokens: usize,
    /// Sampling temperature (0 = greedy argmax).
    temperature: f32,
    /// Vocabulary size for the draft model.
    vocab_size: usize,
}

/// Result of running the draft model K times autoregressively.
pub struct DraftSequence {
    /// The K draft token IDs: NslTensor [K] f64
    pub tokens_ptr: i64,
    /// The K sets of logits: NslTensor [K, vocab_size] f64
    pub logits_ptr: i64,
    /// How many tokens were actually drafted (may be < K if EOS hit).
    pub num_drafted: usize,
}

impl DraftModelRunner {
    /// Create a new draft model runner.
    pub fn new(
        forward_fn: extern "C" fn(i64, i64, i64) -> i64,
        num_tokens: usize,
        temperature: f32,
        vocab_size: usize,
    ) -> Self {
        Self { forward_fn, num_tokens, temperature, vocab_size }
    }

    /// Run the draft model autoregressively for up to K steps.
    ///
    /// Starting from the last token in `prompt_tokens`, generates `num_tokens`
    /// candidate tokens by running forward_fn in a loop.
    ///
    /// Returns a DraftSequence with token IDs and logits tensors.
    pub fn run_draft(&self, last_token: i64, kv_cache_handle: i64, start_pos: i64) -> DraftSequence {
        let k = self.num_tokens;
        let vocab = self.vocab_size;

        // Allocate output buffers
        let tokens_data = checked_alloc(k * std::mem::size_of::<f64>()) as *mut f64;
        let logits_data = checked_alloc(k * vocab * std::mem::size_of::<f64>()) as *mut f64;

        let mut current_token = last_token;
        let mut num_drafted = 0;

        for step in 0..k {
            // Create single-token input tensor
            let input_tensor = Self::make_token_tensor(current_token);
            let input_ptr = Box::into_raw(input_tensor) as i64;

            // Call draft model forward pass
            let logits_ptr = (self.forward_fn)(input_ptr, kv_cache_handle, start_pos + step as i64);

            // Read logits from returned tensor
            if logits_ptr == 0 {
                // Forward pass failed — stop drafting
                crate::tensor::nsl_tensor_free(input_ptr);
                break;
            }
            let logits_tensor = NslTensor::from_ptr(logits_ptr);
            let logits_len = logits_tensor.len as usize;
            let actual_vocab = logits_len.min(vocab);

            // Copy logits to our buffer
            if logits_tensor.dtype == 0 {
                let src = unsafe { std::slice::from_raw_parts(logits_tensor.data as *const f64, actual_vocab) };
                for (i, &v) in src.iter().enumerate() {
                    unsafe { *logits_data.add(step * vocab + i) = v };
                }
            } else {
                let src = unsafe { std::slice::from_raw_parts(logits_tensor.data as *const f32, actual_vocab) };
                for (i, &v) in src.iter().enumerate() {
                    unsafe { *logits_data.add(step * vocab + i) = v as f64 };
                }
            }
            // Zero-fill if logits shorter than vocab
            for i in actual_vocab..vocab {
                unsafe { *logits_data.add(step * vocab + i) = f64::NEG_INFINITY };
            }

            // Sample next token from logits
            let next_token = if self.temperature == 0.0 {
                self.greedy_sample(logits_data, step, vocab)
            } else {
                self.stochastic_sample(logits_data, step, vocab)
            };

            unsafe { *tokens_data.add(step) = next_token as f64 };
            current_token = next_token;
            num_drafted += 1;

            // Clean up input and logits tensors
            crate::tensor::nsl_tensor_free(input_ptr);
            crate::tensor::nsl_tensor_free(logits_ptr);
        }

        // Build output tensors — reallocate to exact size if num_drafted < k
        // to avoid dealloc size mismatch (allocator requires matching Layout).
        let (final_tokens, final_logits) = if num_drafted < k && num_drafted > 0 {
            // Copy to correctly-sized buffers
            let t = checked_alloc(num_drafted * std::mem::size_of::<f64>()) as *mut f64;
            unsafe { std::ptr::copy_nonoverlapping(tokens_data, t, num_drafted) };
            let l = checked_alloc(num_drafted * vocab * std::mem::size_of::<f64>()) as *mut f64;
            unsafe { std::ptr::copy_nonoverlapping(logits_data, l, num_drafted * vocab) };
            // Free oversized originals
            unsafe {
                crate::memory::checked_free(tokens_data as *mut u8, k * std::mem::size_of::<f64>());
                crate::memory::checked_free(logits_data as *mut u8, k * vocab * std::mem::size_of::<f64>());
            }
            (t, l)
        } else if num_drafted == 0 {
            // Nothing drafted — free originals, use 1-element placeholders
            unsafe {
                crate::memory::checked_free(tokens_data as *mut u8, k * std::mem::size_of::<f64>());
                crate::memory::checked_free(logits_data as *mut u8, k * vocab * std::mem::size_of::<f64>());
            }
            let t = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
            let l = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
            (t, l)
        } else {
            // num_drafted == k — buffers are exactly the right size
            (tokens_data, logits_data)
        };
        let tokens_tensor = Self::make_1d_tensor(final_tokens, num_drafted);
        let logits_tensor = Self::make_2d_tensor(final_logits, num_drafted, vocab);

        DraftSequence {
            tokens_ptr: Box::into_raw(tokens_tensor) as i64,
            logits_ptr: Box::into_raw(logits_tensor) as i64,
            num_drafted,
        }
    }

    /// Greedy sampling: return argmax of logits at the given step.
    fn greedy_sample(&self, logits_data: *const f64, step: usize, vocab: usize) -> i64 {
        let mut best_idx = 0i64;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..vocab {
            let v = unsafe { *logits_data.add(step * vocab + i) };
            if v > best_val {
                best_val = v;
                best_idx = i as i64;
            }
        }
        best_idx
    }

    /// Stochastic sampling with temperature.
    fn stochastic_sample(&self, logits_data: *const f64, step: usize, vocab: usize) -> i64 {
        let temp = self.temperature as f64;
        // Find max for numerical stability
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..vocab {
            let v = unsafe { *logits_data.add(step * vocab + i) };
            if v > max_val { max_val = v; }
        }

        // Compute softmax with temperature
        let mut probs: Vec<f64> = (0..vocab).map(|i| {
            let v = unsafe { *logits_data.add(step * vocab + i) };
            ((v - max_val) / temp).exp()
        }).collect();
        let sum: f64 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs { *p /= sum; }
        } else {
            probs.fill(1.0 / vocab as f64);
        }

        // Sample via cumulative distribution
        let seed = (step as u64).wrapping_mul(0x517cc1b727220a95) ^ 0x6c62272e07bb0142;
        let r = (seed % 10000) as f64 / 10000.0;
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r < cum { return i as i64; }
        }
        (vocab - 1) as i64
    }

    /// Create a 1-element tensor containing a single token ID.
    fn make_token_tensor(token: i64) -> Box<NslTensor> {
        let data = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
        unsafe { *data = token as f64 };

        let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = 1 };

        let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = 1 };

        Box::new(NslTensor {
            data: data as *mut c_void,
            shape, strides, ndim: 1, len: 1,
            refcount: AtomicI64::new(1),
            device: 0, dtype: 0, owns_data: 1, data_owner: 0,
        })
    }

    /// Create a 1D tensor from an existing data pointer.
    fn make_1d_tensor(data: *mut f64, len: usize) -> Box<NslTensor> {
        let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = len as i64 };
        let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = 1 };
        Box::new(NslTensor {
            data: data as *mut c_void,
            shape, strides, ndim: 1, len: len as i64,
            refcount: AtomicI64::new(1),
            device: 0, dtype: 0, owns_data: 1, data_owner: 0,
        })
    }

    /// Create a 2D tensor [rows, cols] from an existing data pointer.
    fn make_2d_tensor(data: *mut f64, rows: usize, cols: usize) -> Box<NslTensor> {
        let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = rows as i64; *shape.add(1) = cols as i64 };
        let strides = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = cols as i64; *strides.add(1) = 1 };
        Box::new(NslTensor {
            data: data as *mut c_void,
            shape, strides, ndim: 2, len: (rows * cols) as i64,
            refcount: AtomicI64::new(1),
            device: 0, dtype: 0, owns_data: 1, data_owner: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// MedusaDraftRunner — single forward pass with multiple prediction heads
// ---------------------------------------------------------------------------

/// Runs a backbone model with Medusa heads in a single forward pass to produce
/// a tree of candidate continuations.
pub struct MedusaDraftRunner {
    /// Function pointer to the backbone + Medusa heads forward pass.
    /// Signature: (input_ids_ptr, kv_cache_handle, seq_position) -> logits_tensor_ptr
    /// Returns: NslTensor [num_heads, vocab_size] with logits from each head.
    forward_fn: extern "C" fn(i64, i64, i64) -> i64,
    /// Number of Medusa heads.
    num_heads: usize,
    /// Tree branching factor at each depth.
    tree_width: usize,
    /// Vocabulary size.
    vocab_size: usize,
}

/// Result of Medusa draft generation.
pub struct MedusaDraftResult {
    /// Tree of candidate tokens with DFS timestamps.
    pub tree: super::types::SpeculativeTree,
    /// Flat logits from all heads: [num_heads, vocab_size].
    pub head_logits_ptr: i64,
}

impl MedusaDraftRunner {
    /// Create a new Medusa draft runner.
    pub fn new(
        forward_fn: extern "C" fn(i64, i64, i64) -> i64,
        num_heads: usize,
        tree_width: usize,
        vocab_size: usize,
    ) -> Self {
        Self { forward_fn, num_heads, tree_width, vocab_size }
    }

    /// Run backbone + Medusa heads and build a speculation tree.
    ///
    /// A single forward pass produces `num_heads` sets of logits.
    /// Each head predicts a token at a different future position.
    /// We build a tree of width `tree_width` from the top-k tokens at each head.
    pub fn run_draft(&self, last_token: i64, kv_cache_handle: i64, seq_pos: i64) -> MedusaDraftResult {
        // Create input tensor
        let input_tensor = DraftModelRunner::make_token_tensor(last_token);
        let input_ptr = Box::into_raw(input_tensor) as i64;

        // Single forward pass — returns [num_heads, vocab_size] logits
        let logits_ptr = (self.forward_fn)(input_ptr, kv_cache_handle, seq_pos);

        // Extract top-k tokens from each head to build the tree
        let tree = if logits_ptr != 0 {
            let logits_tensor = NslTensor::from_ptr(logits_ptr);
            let total = logits_tensor.len as usize;
            let num_heads = self.num_heads.min(total / self.vocab_size.max(1));

            // Read all head logits
            let all_logits: Vec<f64> = if logits_tensor.dtype == 0 {
                unsafe { std::slice::from_raw_parts(logits_tensor.data as *const f64, total) }.to_vec()
            } else {
                unsafe { std::slice::from_raw_parts(logits_tensor.data as *const f32, total) }
                    .iter().map(|&v| v as f64).collect()
            };

            // For each head, find top-k tokens
            let mut flat_tokens = Vec::new();
            let mut flat_logits = Vec::new();
            for h in 0..num_heads {
                let head_logits = &all_logits[h * self.vocab_size..(h + 1) * self.vocab_size];
                let mut indexed: Vec<(usize, f64)> = head_logits.iter().enumerate()
                    .map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

                for &(tok, logit) in indexed.iter().take(self.tree_width) {
                    flat_tokens.push(tok as i64);
                    flat_logits.push(logit as f32);
                }
            }

            super::tree::build_tree(
                num_heads,
                self.tree_width,
                &flat_tokens,
                &flat_logits,
            )
        } else {
            // Forward pass failed — return empty tree
            super::types::SpeculativeTree {
                nodes: Vec::new(),
                dfs_enter: Vec::new(),
                dfs_exit: Vec::new(),
                children: Vec::new(),
                tree_depth: 0,
                tree_width: 0,
            }
        };

        // Cleanup input tensor
        crate::tensor::nsl_tensor_free(input_ptr);

        MedusaDraftResult {
            tree,
            head_logits_ptr: logits_ptr,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Mock forward function: always returns logits with token 2 having the highest value.
    // NOTE: does NOT free the input tensor — the caller (DraftModelRunner) handles that.
    extern "C" fn mock_forward_always_2(_input_ptr: i64, _kv: i64, _pos: i64) -> i64 {
        // Create [1, 4] logits tensor where index 2 is the max
        let vocab = 4;
        let data = checked_alloc(vocab * std::mem::size_of::<f64>()) as *mut f64;
        unsafe {
            *data.add(0) = -1.0;  // token 0
            *data.add(1) = -2.0;  // token 1
            *data.add(2) = 5.0;   // token 2 — max
            *data.add(3) = -0.5;  // token 3
        }

        let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = vocab as i64 };
        let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = 1 };

        let tensor = Box::new(NslTensor {
            data: data as *mut c_void,
            shape, strides, ndim: 1, len: vocab as i64,
            refcount: AtomicI64::new(1),
            device: 0, dtype: 0, owns_data: 1, data_owner: 0,
        });
        Box::into_raw(tensor) as i64
    }

    #[test]
    fn draft_runner_greedy_always_predicts_same() {
        let runner = DraftModelRunner::new(mock_forward_always_2, 3, 0.0, 4);
        let result = runner.run_draft(0, 0, 0);

        assert_eq!(result.num_drafted, 3);

        // All 3 tokens should be token 2 (greedy argmax)
        let tokens_tensor = NslTensor::from_ptr(result.tokens_ptr);
        let tokens = unsafe { std::slice::from_raw_parts(tokens_tensor.data as *const f64, 3) };
        assert_eq!(tokens[0] as i64, 2);
        assert_eq!(tokens[1] as i64, 2);
        assert_eq!(tokens[2] as i64, 2);

        // Logits should have token 2 as the max at each position
        let logits_tensor = NslTensor::from_ptr(result.logits_ptr);
        let logits = unsafe { std::slice::from_raw_parts(logits_tensor.data as *const f64, 3 * 4) };
        for step in 0..3 {
            let row = &logits[step * 4..(step + 1) * 4];
            let argmax = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            assert_eq!(argmax, 2);
        }

        // Cleanup
        crate::tensor::nsl_tensor_free(result.tokens_ptr);
        crate::tensor::nsl_tensor_free(result.logits_ptr);
    }

    #[test]
    fn draft_runner_single_step() {
        let runner = DraftModelRunner::new(mock_forward_always_2, 1, 0.0, 4);
        let result = runner.run_draft(7, 0, 10);

        assert_eq!(result.num_drafted, 1);

        let tokens_tensor = NslTensor::from_ptr(result.tokens_ptr);
        let tokens = unsafe { std::slice::from_raw_parts(tokens_tensor.data as *const f64, 1) };
        assert_eq!(tokens[0] as i64, 2);

        crate::tensor::nsl_tensor_free(result.tokens_ptr);
        crate::tensor::nsl_tensor_free(result.logits_ptr);
    }

    #[test]
    fn draft_runner_stochastic_sampling() {
        // With high temperature, sampling is still valid (produces valid token IDs)
        let runner = DraftModelRunner::new(mock_forward_always_2, 5, 1.0, 4);
        let result = runner.run_draft(0, 0, 0);

        assert_eq!(result.num_drafted, 5);

        let tokens_tensor = NslTensor::from_ptr(result.tokens_ptr);
        let tokens = unsafe { std::slice::from_raw_parts(tokens_tensor.data as *const f64, 5) };
        for &t in tokens {
            let tok = t as i64;
            assert!(tok >= 0 && tok < 4, "token {} out of range [0,4)", tok);
        }

        crate::tensor::nsl_tensor_free(result.tokens_ptr);
        crate::tensor::nsl_tensor_free(result.logits_ptr);
    }

    // Mock forward for Medusa: returns [2, 4] logits (2 heads, vocab=4)
    // Head 0: token 1 is max, Head 1: token 3 is max
    extern "C" fn mock_medusa_forward(_input: i64, _kv: i64, _pos: i64) -> i64 {
        let num_heads = 2;
        let vocab = 4;
        let total = num_heads * vocab;
        let data = checked_alloc(total * std::mem::size_of::<f64>()) as *mut f64;
        unsafe {
            // Head 0: [0.1, 5.0, -1.0, 0.5] → token 1 is max
            *data.add(0) = 0.1; *data.add(1) = 5.0; *data.add(2) = -1.0; *data.add(3) = 0.5;
            // Head 1: [-2.0, 0.0, 1.0, 4.0] → token 3 is max
            *data.add(4) = -2.0; *data.add(5) = 0.0; *data.add(6) = 1.0; *data.add(7) = 4.0;
        }
        let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = num_heads as i64; *shape.add(1) = vocab as i64 };
        let strides = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = vocab as i64; *strides.add(1) = 1 };
        let tensor = Box::new(NslTensor {
            data: data as *mut c_void,
            shape, strides, ndim: 2, len: total as i64,
            refcount: AtomicI64::new(1),
            device: 0, dtype: 0, owns_data: 1, data_owner: 0,
        });
        Box::into_raw(tensor) as i64
    }

    #[test]
    fn medusa_runner_builds_tree() {
        let runner = MedusaDraftRunner::new(mock_medusa_forward, 2, 2, 4);
        let result = runner.run_draft(0, 0, 0);

        // Should have a tree with: root + 2 tokens from head 0 + 2*2 from head 1
        assert!(!result.tree.nodes.is_empty(), "tree should have nodes");
        assert!(result.tree.nodes.len() >= 3, "tree should have root + at least 2 children");

        // The top-1 token from head 0 should be token 1 (highest logit)
        // Find first non-root node at depth 1
        let depth1_nodes: Vec<_> = result.tree.nodes.iter()
            .filter(|n| n.depth == 1).collect();
        assert!(!depth1_nodes.is_empty(), "should have depth-1 nodes");
        assert_eq!(depth1_nodes[0].token_id, 1, "head 0's top token should be 1");

        // Clean up head logits tensor
        if result.head_logits_ptr != 0 {
            crate::tensor::nsl_tensor_free(result.head_logits_ptr);
        }
    }

    #[test]
    fn make_token_tensor_correct() {
        let tensor = DraftModelRunner::make_token_tensor(42);
        let ptr = Box::into_raw(tensor);
        let t = NslTensor::from_ptr(ptr as i64);
        assert_eq!(t.ndim, 1);
        assert_eq!(t.len, 1);
        let val = unsafe { *(t.data as *const f64) };
        assert_eq!(val as i64, 42);
        crate::tensor::nsl_tensor_free(ptr as i64);
    }
}
