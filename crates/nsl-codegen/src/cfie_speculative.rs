//! CFIE — the serve `speculative:` section's configuration.
//!
//! What exists is one method: a greedy draft model, compiled into the
//! same binary, proposes `k_tokens` tokens per round in a linear chain,
//! and the target confirms or corrects them
//! (`nsl_runtime::cfie::engine`, `speculative_rounds`). The output is the
//! token sequence plain decoding gives.
//!
//! This module used to plan more than that: tree, Medusa and lookahead
//! drafting, a tree attention mask for a batched verification pass, a
//! draft temperature, an acceptance-rate fallback and KV sharing. None
//! of them had a driver. The serve block now refuses them
//! (`nsl-semantic`'s `cfie.rs`), and the plan carries only what is used.
//! It also printed an "expected speedup" of `1 + K p / (1 + K)` at an
//! assumed acceptance rate of 0.6; the driver verifies one position per
//! target step, so speculation here costs time, and the report says so.

/// Compile-time speculative-decoding configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeConfig {
    /// Number of draft tokens proposed per round (1..=32).
    pub k_tokens: u32,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self { k_tokens: 5 }
    }
}
