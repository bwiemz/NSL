//! CFIE — compiled speculative decoding.
//!
//! Paper §5: fuses draft model + target model verification + rejection
//! sampling into a single persistent kernel chain.  The draft tree
//! structure, token count `K`, tree-width, and rejection logic are all
//! baked into the kernel control flow at compile time — no runtime
//! `DraftModelRunner` or Python-level tree manager.
//!
//! Gemini's review emphasises that keeping draft states in SMEM /
//! registers (not HBM) is what unlocks the theoretical speedup of
//! speculative decoding.  This module's output is a structured plan
//! that the downstream kernel emitter honours.

use serde::Serialize;

/// Drafting strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DraftMethod {
    /// Linear (autoregressive) draft of `K` tokens.
    Standard,
    /// Tree-structured draft (top-`tree_width` branches at each depth).
    Tree,
    /// Medusa-style multi-head draft.
    Medusa,
    /// Lookahead decoding (parallel n-gram guess).
    Lookahead,
}

impl DraftMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            DraftMethod::Standard => "standard",
            DraftMethod::Tree => "tree",
            DraftMethod::Medusa => "medusa",
            DraftMethod::Lookahead => "lookahead",
        }
    }
}

/// Compile-time speculative-decoding configuration.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    pub method: DraftMethod,
    /// Number of draft tokens proposed per target step.
    pub k_tokens: u32,
    /// For `Tree` method, number of branches per position.
    pub tree_width: u32,
    /// Draft-model temperature (0.0 = argmax).
    pub draft_temperature: f32,
    /// Fallback to target-only decoding when the draft's acceptance
    /// rate falls below this threshold.
    pub min_acceptance_rate: f32,
    /// Whether the draft shares the target's KV cache via
    /// Medusa-style head reuse.
    pub share_kv: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            method: DraftMethod::Standard,
            k_tokens: 5,
            tree_width: 1,
            draft_temperature: 0.0,
            min_acceptance_rate: 0.3,
            share_kv: false,
        }
    }
}

/// Compile-time tree-attention mask — one bit per (node, ancestor)
/// pair.  Packed row-major: `mask[i * n_nodes + j]` is `true` iff
/// node `i` is allowed to attend to node `j` (i.e. `j` is an
/// ancestor or equal to `i`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TreeMask {
    pub num_nodes: u32,
    pub max_depth: u32,
    pub width: u32,
    pub bits: Vec<bool>,
}

impl TreeMask {
    pub fn get(&self, q: u32, k: u32) -> bool {
        if q >= self.num_nodes || k >= self.num_nodes {
            return false;
        }
        self.bits[(q * self.num_nodes + k) as usize]
    }
    pub fn density(&self) -> f64 {
        if self.bits.is_empty() {
            return 0.0;
        }
        let ones = self.bits.iter().filter(|b| **b).count();
        ones as f64 / self.bits.len() as f64
    }
}

/// Per-phase op program for the fused speculative step.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum SpeculativeOp {
    /// Draft forward of step `k` into registers, producing a sampled
    /// token + its probability (for rejection sampling).
    DraftForward { k: u32 },
    /// Tree branch: at depth `k`, expand `tree_width` children from
    /// each current leaf.
    DraftBranch { depth: u32, width: u32 },
    /// Single target forward over `(input + drafted tokens)` with a
    /// compile-time attention mask.  Uses CSHA-fused attention.
    TargetForward { num_positions: u32 },
    /// Rejection sampling: for each draft token `k`, accept iff
    /// `random() < target_prob / draft_prob`.
    RejectionSample { k: u32 },
    /// Resample one token from the corrected distribution when a
    /// rejection occurs.
    ResampleCorrection,
    /// Advance the KV cache by `accepted` positions; discard rejected
    /// draft entries.
    KvCacheAdvance,
}

/// Complete fused speculative program.
#[derive(Debug, Clone)]
pub struct SpeculativeProgram {
    pub config: SpeculativeConfig,
    pub ops: Vec<SpeculativeOp>,
    pub tree_mask: Option<TreeMask>,
    /// Estimated per-step speedup over target-only decoding, given the
    /// expected acceptance rate.  Computed as `1 + (K · p) / (1 + K)`
    /// per the classical speculative-decoding analysis (Leviathan et
    /// al. 2023).
    pub expected_speedup: f64,
}

impl SpeculativeProgram {
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn uses_tree(&self) -> bool {
        matches!(self.config.method, DraftMethod::Tree)
    }
}

// ---------------------------------------------------------------------------
// Tree-mask construction
// ---------------------------------------------------------------------------

/// Build a binary-tree mask with given depth and branching factor.
///
/// Node numbering: breadth-first, root = 0, children of node `i` are at
/// `width*i + 1 .. width*i + width`.
pub fn build_tree_mask(depth: u32, width: u32) -> TreeMask {
    let depth = depth.max(1);
    let width = width.max(1);
    // Total nodes = sum_{d=0..depth-1} width^d = (width^depth - 1) / (width - 1).
    let mut nodes = 0u32;
    let mut layer = 1u32;
    for _ in 0..depth {
        nodes = nodes.saturating_add(layer);
        layer = layer.saturating_mul(width);
    }
    nodes = nodes.max(1);
    let mut bits = vec![false; (nodes as usize) * (nodes as usize)];

    // Compute each node's parent chain via BFS indexing.
    let parent = |i: u32| -> Option<u32> {
        if i == 0 {
            None
        } else {
            Some((i - 1) / width)
        }
    };

    for i in 0..nodes {
        // Every node attends to itself and all ancestors (causal within
        // the tree).
        let mut cur = Some(i);
        while let Some(c) = cur {
            bits[(i * nodes + c) as usize] = true;
            cur = parent(c);
        }
    }
    TreeMask {
        num_nodes: nodes,
        max_depth: depth,
        width,
        bits,
    }
}

// ---------------------------------------------------------------------------
// Program emission
// ---------------------------------------------------------------------------

fn expected_speedup(k: u32, acceptance: f32) -> f64 {
    if k == 0 {
        return 1.0;
    }
    // Simplified model: speedup = 1 + K·p / (1 + K).
    let p = acceptance as f64;
    let k = k as f64;
    1.0 + (k * p) / (1.0 + k)
}

/// Emit the fused program.
pub fn emit_program(config: SpeculativeConfig, expected_acceptance: f32) -> SpeculativeProgram {
    let mut ops = Vec::new();
    let tree_mask = match config.method {
        DraftMethod::Tree => Some(build_tree_mask(config.k_tokens, config.tree_width.max(1))),
        _ => None,
    };

    // Phase 1: draft.
    match config.method {
        DraftMethod::Standard | DraftMethod::Lookahead => {
            for k in 0..config.k_tokens {
                ops.push(SpeculativeOp::DraftForward { k });
            }
        }
        DraftMethod::Tree => {
            for depth in 0..config.k_tokens {
                ops.push(SpeculativeOp::DraftBranch {
                    depth,
                    width: config.tree_width.max(2),
                });
            }
        }
        DraftMethod::Medusa => {
            // Medusa uses additional LM heads — each head produces a
            // draft token in a single forward.  We still emit one
            // op per draft token for the scheduler's benefit.
            for k in 0..config.k_tokens {
                ops.push(SpeculativeOp::DraftForward { k });
            }
        }
    }

    // Phase 2: target verification.
    let num_positions = match config.method {
        DraftMethod::Tree => tree_mask
            .as_ref()
            .map(|m| m.num_nodes)
            .unwrap_or(config.k_tokens + 1),
        _ => config.k_tokens + 1,
    };
    ops.push(SpeculativeOp::TargetForward { num_positions });

    // Phase 3: rejection sampling + optional resample.
    for k in 0..config.k_tokens {
        ops.push(SpeculativeOp::RejectionSample { k });
    }
    ops.push(SpeculativeOp::ResampleCorrection);

    // Phase 4: KV cache advance.
    ops.push(SpeculativeOp::KvCacheAdvance);

    SpeculativeProgram {
        expected_speedup: expected_speedup(config.k_tokens, expected_acceptance),
        config,
        ops,
        tree_mask,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_mask_root_attends_only_to_self() {
        let m = build_tree_mask(3, 2);
        // Node 0 (root): mask[0,0] = true, everything else in row 0 = false.
        for k in 0..m.num_nodes {
            assert_eq!(m.get(0, k), k == 0);
        }
    }

    #[test]
    fn tree_mask_leaves_attend_to_path_to_root() {
        let m = build_tree_mask(3, 2);
        // depth 3, width 2 → 7 nodes.  Node 3 is a leaf; its ancestors
        // are 1 and 0.
        assert_eq!(m.num_nodes, 7);
        assert!(m.get(3, 3));
        assert!(m.get(3, 1));
        assert!(m.get(3, 0));
        assert!(!m.get(3, 2)); // sibling — not an ancestor
    }

    #[test]
    fn tree_mask_width_one_is_linear_chain() {
        let m = build_tree_mask(5, 1);
        assert_eq!(m.num_nodes, 5);
        // Standard causal pattern: each node attends to self + all
        // earlier.
        for q in 0..m.num_nodes {
            for k in 0..m.num_nodes {
                assert_eq!(m.get(q, k), k <= q);
            }
        }
    }

    #[test]
    fn tree_mask_density_below_dense_for_branching_tree() {
        let m = build_tree_mask(4, 2);
        // 15 nodes × 15 = 225 entries; a causal-linear mask would fill
        // 15 * 16 / 2 = 120.  A tree mask should have fewer.
        assert!(m.density() < 120.0 / 225.0);
    }

    #[test]
    fn standard_draft_emits_k_draft_ops() {
        let cfg = SpeculativeConfig {
            method: DraftMethod::Standard,
            k_tokens: 5,
            tree_width: 1,
            draft_temperature: 0.0,
            min_acceptance_rate: 0.3,
            share_kv: false,
        };
        let prog = emit_program(cfg, 0.7);
        let drafts = prog
            .ops
            .iter()
            .filter(|op| matches!(op, SpeculativeOp::DraftForward { .. }))
            .count();
        assert_eq!(drafts, 5);
    }

    #[test]
    fn tree_method_emits_draft_branches_plus_mask() {
        let cfg = SpeculativeConfig {
            method: DraftMethod::Tree,
            k_tokens: 3,
            tree_width: 2,
            ..Default::default()
        };
        let prog = emit_program(cfg, 0.6);
        assert!(prog.tree_mask.is_some());
        assert!(prog.uses_tree());
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, SpeculativeOp::DraftBranch { .. })));
    }

    #[test]
    fn target_forward_present_in_every_program() {
        let prog = emit_program(SpeculativeConfig::default(), 0.5);
        assert!(prog
            .ops
            .iter()
            .any(|op| matches!(op, SpeculativeOp::TargetForward { .. })));
    }

    #[test]
    fn expected_speedup_bounded_above_one() {
        let s_low = expected_speedup(5, 0.1);
        let s_high = expected_speedup(5, 0.9);
        assert!(s_low >= 1.0);
        assert!(s_high > s_low);
    }

    #[test]
    fn empty_k_yields_speedup_one() {
        assert_eq!(expected_speedup(0, 0.7), 1.0);
    }

    #[test]
    fn program_ends_with_kv_advance() {
        let prog = emit_program(SpeculativeConfig::default(), 0.7);
        assert!(matches!(
            prog.ops.last(),
            Some(SpeculativeOp::KvCacheAdvance)
        ));
    }

    #[test]
    fn method_names_are_stable() {
        assert_eq!(DraftMethod::Standard.as_str(), "standard");
        assert_eq!(DraftMethod::Tree.as_str(), "tree");
        assert_eq!(DraftMethod::Medusa.as_str(), "medusa");
        assert_eq!(DraftMethod::Lookahead.as_str(), "lookahead");
    }
}
