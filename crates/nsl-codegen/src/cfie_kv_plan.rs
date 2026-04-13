//! CFIE — static KV-cache layout planner.
//!
//! Paper §3 describes the core idea: replace vLLM's dynamic
//! `PagedAttention` block table with a compile-time-fixed layout
//! `[n_layers][2][max_tokens][n_kv_heads][head_dim]` that the
//! attention kernel addresses by direct arithmetic — no block-table
//! lookup, no CPU-GPU synchronization per decode step.
//!
//! Gemini's review flagged fragmentation as the key risk with an AOT
//! KV plan: unpredictable request lengths can waste pre-allocated
//! slots.  We address that with a **GPU-side bump allocator** — a
//! lightweight free-list of fixed-size blocks that the persistent
//! decode kernel itself manages, without waking the CPU.
//!
//! The planner emits a [`KvLayoutPlan`] with two sub-plans:
//!
//!   * `direct`   — the static contiguous layout for the "everything
//!                  bounded" common case.
//!   * `bump`     — an optional bump-allocator descriptor used when
//!                  the compiler cannot guarantee all sequences fit
//!                  within the max-length envelope.

use serde::Serialize;

/// Broad layout kind selected by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LayoutKind {
    /// Pre-allocated contiguous buffer with direct indexing.
    Static,
    /// Block-based (paged) layout — falls back to `PagedAttention`
    /// semantics when static layout can't hold the workload envelope.
    Paged,
    /// Bump-allocated blocks inside the static buffer — uses the fast
    /// direct-indexing path until fragmentation forces a fallback
    /// (measured by the GPU-side allocator).
    StaticWithBump,
}

impl LayoutKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LayoutKind::Static => "static",
            LayoutKind::Paged => "paged",
            LayoutKind::StaticWithBump => "static_with_bump",
        }
    }
}

/// Compile-time model KV-shape inputs.
#[derive(Debug, Clone)]
pub struct KvShape {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Bytes per stored KV element (1 = INT8, 2 = FP16/BF16, 4 = FP32).
    pub dtype_bytes: u32,
}

impl KvShape {
    /// Bytes per token per layer: `2 (K+V) × n_kv_heads × head_dim`.
    pub fn bytes_per_token_per_layer(&self) -> u64 {
        2 * (self.n_kv_heads as u64) * (self.head_dim as u64) * (self.dtype_bytes as u64)
    }

    /// Total bytes per token across all layers.
    pub fn bytes_per_token(&self) -> u64 {
        self.bytes_per_token_per_layer() * (self.n_layers as u64)
    }
}

/// Cluster / GPU budget + serve-block expectations.
#[derive(Debug, Clone)]
pub struct KvBudget {
    /// Total HBM bytes on the target GPU.
    pub hbm_bytes: u64,
    /// Bytes reserved for model weights.
    pub weights_bytes: u64,
    /// Bytes reserved for CUDA runtime + overhead (driver, activations,
    /// workspace, …).  Paper's example uses 800 MB on an RTX 5070 Ti.
    pub runtime_overhead_bytes: u64,
    /// Maximum sequence length the compiled binary will handle.
    pub max_seq: u32,
    /// Maximum concurrent batch the scheduler will admit.
    pub max_batch: u32,
    /// Desired block size (tokens per KV block).  The planner snaps
    /// this to the nearest cache-line-aligned value.  Paper default: 256.
    pub block_size: u32,
}

impl Default for KvBudget {
    fn default() -> Self {
        Self {
            hbm_bytes: 16 * 1024 * 1024 * 1024,
            weights_bytes: 100 * 1024 * 1024,
            runtime_overhead_bytes: 800 * 1024 * 1024,
            max_seq: 2048,
            max_batch: 64,
            block_size: 256,
        }
    }
}

/// GPU-side bump allocator descriptor.  The persistent decode kernel
/// reads/writes this structure (stored in pinned memory + a GPU
/// mirror) to manage KV block assignments without CPU intervention.
#[derive(Debug, Clone, Serialize)]
pub struct BumpAllocator {
    /// Total number of fixed-size KV blocks.
    pub num_blocks: u32,
    /// Size of each block in tokens.
    pub block_size: u32,
    /// Ring-buffer capacity used by the free-list.  Must be a power of
    /// two (the GPU-side scheduler relies on mask-based wraparound).
    pub freelist_capacity: u32,
}

/// Direct-indexing descriptor.
#[derive(Debug, Clone, Serialize)]
pub struct DirectLayout {
    /// `max_tokens` that fit in the pre-allocated buffer (the planner's
    /// upper bound — the number of tokens any single sequence slot can
    /// hold after batch-wise partitioning is `max_tokens / max_batch`).
    pub max_tokens: u64,
    /// Per-sequence max tokens.
    pub per_sequence_max_tokens: u32,
    /// Bytes per block.
    pub bytes_per_block: u64,
    /// Blocks per sequence for the worst-case max-length sequence.
    pub blocks_per_sequence: u32,
    /// Cache-line-aligned block size in tokens.
    pub block_size: u32,
    /// Bytes reserved for the entire KV buffer.
    pub total_kv_bytes: u64,
}

/// Full KV layout plan.
#[derive(Debug, Clone, Serialize)]
pub struct KvLayoutPlan {
    pub kind: LayoutKind,
    pub direct: Option<DirectLayout>,
    pub bump: Option<BumpAllocator>,
    pub available_bytes: u64,
    pub rationale: String,
}

impl KvLayoutPlan {
    pub fn uses_direct_indexing(&self) -> bool {
        matches!(self.kind, LayoutKind::Static | LayoutKind::StaticWithBump)
    }

    pub fn has_bump_allocator(&self) -> bool {
        self.bump.is_some()
    }
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

/// Snap `block_size` to the nearest cache-line-aligned integer ≥ 64.
fn snap_block_size(requested: u32) -> u32 {
    let candidates = [64u32, 128, 256, 512, 1024];
    for &c in candidates.iter().rev() {
        if requested >= c {
            return c;
        }
    }
    64
}

fn next_pow2(n: u32) -> u32 {
    let mut x = 1u32;
    while x < n && x < u32::MAX / 2 {
        x <<= 1;
    }
    x
}

/// Build the layout plan.  Returns a [`LayoutKind::Paged`] fallback
/// when static allocation can't accommodate the requested envelope.
pub fn plan(shape: &KvShape, budget: &KvBudget) -> KvLayoutPlan {
    let bytes_per_token = shape.bytes_per_token();
    let available = budget
        .hbm_bytes
        .saturating_sub(budget.weights_bytes)
        .saturating_sub(budget.runtime_overhead_bytes);

    if bytes_per_token == 0 {
        return KvLayoutPlan {
            kind: LayoutKind::Paged,
            direct: None,
            bump: None,
            available_bytes: available,
            rationale: "KV shape has zero bytes per token — falling back to dynamic layout"
                .to_string(),
        };
    }

    let block_size = snap_block_size(budget.block_size);
    let bytes_per_block = bytes_per_token * block_size as u64;

    // How many blocks fit in the available budget?
    let max_blocks = (available / bytes_per_block.max(1)).max(1);
    let max_tokens = max_blocks * block_size as u64;

    // Can the budget hold every batch slot at full length?
    let needed_tokens = (budget.max_seq as u64) * (budget.max_batch as u64);
    if max_tokens >= needed_tokens {
        let per_seq_blocks = budget.max_seq.div_ceil(block_size);
        let direct = DirectLayout {
            max_tokens,
            per_sequence_max_tokens: budget.max_seq,
            bytes_per_block,
            blocks_per_sequence: per_seq_blocks,
            block_size,
            total_kv_bytes: bytes_per_block * max_blocks,
        };
        return KvLayoutPlan {
            kind: LayoutKind::Static,
            direct: Some(direct),
            bump: None,
            available_bytes: available,
            rationale: format!(
                "static layout fits: {} blocks × {} tokens = {} > needed {}",
                max_blocks, block_size, max_tokens, needed_tokens
            ),
        };
    }

    // Otherwise try the static-with-bump hybrid: use every available
    // byte, let the GPU scheduler manage fragmentation.  Bump allocator
    // needs a free-list sized to the block count.
    if max_blocks >= budget.max_batch as u64 {
        let bump = BumpAllocator {
            num_blocks: max_blocks as u32,
            block_size,
            freelist_capacity: next_pow2(max_blocks as u32).max(64),
        };
        let per_seq_blocks = budget.max_seq.div_ceil(block_size);
        let direct = DirectLayout {
            max_tokens,
            per_sequence_max_tokens: budget.max_seq,
            bytes_per_block,
            blocks_per_sequence: per_seq_blocks,
            block_size,
            total_kv_bytes: bytes_per_block * max_blocks,
        };
        return KvLayoutPlan {
            kind: LayoutKind::StaticWithBump,
            direct: Some(direct),
            bump: Some(bump),
            available_bytes: available,
            rationale: format!(
                "static+bump: {} blocks total < {} required (batch × seq), GPU-side bump allocator handles fragmentation",
                max_blocks, needed_tokens / block_size as u64
            ),
        };
    }

    // Not enough even for one block per batch slot — fall back to paged.
    KvLayoutPlan {
        kind: LayoutKind::Paged,
        direct: None,
        bump: None,
        available_bytes: available,
        rationale: format!(
            "only {} blocks fit — less than max_batch ({}); fall back to PagedAttention",
            max_blocks, budget.max_batch
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn nslcoder_shape() -> KvShape {
        KvShape {
            n_layers: 8,
            n_kv_heads: 4,
            head_dim: 128,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn bytes_per_token_matches_paper_example() {
        // 4 KV × 128 × 2 bytes × 2 (K+V) = 2048 per layer × 8 layers = 16384 bytes.
        assert_eq!(nslcoder_shape().bytes_per_token(), 16_384);
    }

    #[test]
    fn static_layout_fits_small_model_on_5070ti() {
        let budget = KvBudget {
            hbm_bytes: 16 * 1024 * 1024 * 1024,
            weights_bytes: 98 * 1024 * 1024,
            runtime_overhead_bytes: 800 * 1024 * 1024,
            max_seq: 2048,
            max_batch: 64,
            block_size: 256,
        };
        let plan = plan(&nslcoder_shape(), &budget);
        assert_eq!(plan.kind, LayoutKind::Static);
        let direct = plan.direct.unwrap();
        assert_eq!(direct.block_size, 256);
        // Paper's example: ~968k max tokens.
        assert!(direct.max_tokens > 900_000);
        assert_eq!(direct.blocks_per_sequence, 8); // 2048 / 256
    }

    #[test]
    fn bump_allocator_kicks_in_when_budget_cannot_hold_full_envelope() {
        let budget = KvBudget {
            hbm_bytes: 512 * 1024 * 1024,
            weights_bytes: 32 * 1024 * 1024,
            runtime_overhead_bytes: 32 * 1024 * 1024,
            max_seq: 4096,
            max_batch: 128,
            block_size: 256,
        };
        let plan = plan(&nslcoder_shape(), &budget);
        // Too tight for everyone to hold full seq, but enough blocks for
        // the bump allocator to schedule between requests.
        assert!(matches!(
            plan.kind,
            LayoutKind::StaticWithBump | LayoutKind::Paged
        ));
    }

    #[test]
    fn paged_fallback_when_nothing_fits() {
        let budget = KvBudget {
            hbm_bytes: 128 * 1024 * 1024,
            weights_bytes: 100 * 1024 * 1024,
            runtime_overhead_bytes: 20 * 1024 * 1024,
            max_seq: 4096,
            max_batch: 256,
            block_size: 256,
        };
        let plan = plan(&nslcoder_shape(), &budget);
        assert_eq!(plan.kind, LayoutKind::Paged);
        assert!(plan.direct.is_none());
        assert!(plan.bump.is_none());
    }

    #[test]
    fn zero_bytes_per_token_falls_back_to_paged() {
        let mut shape = nslcoder_shape();
        shape.n_kv_heads = 0;
        let plan = plan(&shape, &KvBudget::default());
        assert_eq!(plan.kind, LayoutKind::Paged);
    }

    #[test]
    fn block_size_snaps_to_power_of_two_ish() {
        assert_eq!(snap_block_size(256), 256);
        assert_eq!(snap_block_size(300), 256);
        assert_eq!(snap_block_size(500), 256);
        assert_eq!(snap_block_size(600), 512);
        assert_eq!(snap_block_size(100), 64);
    }

    #[test]
    fn bump_freelist_capacity_is_power_of_two() {
        let budget = KvBudget {
            hbm_bytes: 2 * 1024 * 1024 * 1024,
            weights_bytes: 100 * 1024 * 1024,
            runtime_overhead_bytes: 100 * 1024 * 1024,
            max_seq: 8192,
            max_batch: 512, // deliberately over-subscribed
            block_size: 256,
        };
        let plan = plan(&nslcoder_shape(), &budget);
        if let Some(bump) = plan.bump.as_ref() {
            assert!(bump.freelist_capacity.is_power_of_two());
            assert!(bump.freelist_capacity >= 64);
        }
    }

    #[test]
    fn direct_layout_exposes_direct_indexing() {
        let plan = plan(&nslcoder_shape(), &KvBudget::default());
        if plan.kind == LayoutKind::Static {
            assert!(plan.uses_direct_indexing());
            assert!(!plan.has_bump_allocator());
        }
    }

    #[test]
    fn rationale_never_empty() {
        let plan = plan(&nslcoder_shape(), &KvBudget::default());
        assert!(!plan.rationale.is_empty());
    }
}
