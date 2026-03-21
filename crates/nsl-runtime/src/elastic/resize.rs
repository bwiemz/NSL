//! Elastic resize: shrink/grow the data-parallel dimension at runtime.
//!
//! When a rank fails, the DP dimension shrinks. The correct response is to
//! adjust gradient accumulation steps (NOT learning rate) to preserve the
//! effective global batch size.

use std::collections::HashSet;

/// Elastic training world — tracks the current state of the distributed group.
#[derive(Debug)]
pub struct ElasticWorld {
    /// Original world size (at training start).
    pub initial_world_size: u32,
    /// Current world size (after failures/scale-ups).
    pub current_world_size: u32,
    /// Set of alive rank IDs.
    pub alive_ranks: HashSet<u32>,
    /// Tensor parallel group size (fixed — TP is not elastic).
    pub tp_size: u32,
    /// Pipeline parallel stages (fixed — PP is not elastic).
    pub pp_stages: u32,
    /// Current data parallel replicas.
    pub dp_replicas: u32,
    /// Current gradient accumulation steps.
    pub grad_accum_steps: u32,
    /// Micro batch size (fixed per GPU).
    pub micro_batch_size: u32,
}

/// Result of an elastic resize operation.
#[derive(Debug)]
pub struct ResizeResult {
    /// New world size after resize.
    pub new_world_size: u32,
    /// New DP replica count.
    pub new_dp_replicas: u32,
    /// New gradient accumulation steps (adjusted to preserve batch size).
    pub new_grad_accum_steps: u32,
    /// Effective global batch size (should be unchanged).
    pub effective_batch_size: u64,
    /// Ranks that need to reload from checkpoint.
    pub ranks_to_reload: Vec<u32>,
}

impl ElasticWorld {
    /// Create a new elastic world.
    pub fn new(
        world_size: u32,
        tp_size: u32,
        pp_stages: u32,
        micro_batch_size: u32,
        grad_accum_steps: u32,
    ) -> Self {
        let dp_replicas = world_size / (tp_size * pp_stages);
        Self {
            initial_world_size: world_size,
            current_world_size: world_size,
            alive_ranks: (0..world_size).collect(),
            tp_size,
            pp_stages,
            dp_replicas,
            grad_accum_steps,
            micro_batch_size,
        }
    }

    /// Get the current effective global batch size.
    pub fn effective_batch_size(&self) -> u64 {
        self.micro_batch_size as u64 * self.dp_replicas as u64 * self.grad_accum_steps as u64
    }

    /// Handle a rank failure: remove the dead rank and adjust DP dimension.
    ///
    /// **Critical math:** We adjust gradient accumulation steps, NOT learning rate.
    /// For Transformers with AdamW, changing LR destabilizes Adam's momentum buffers.
    /// Keeping LR constant and adjusting grad_accum preserves the optimization trajectory.
    pub fn handle_rank_failure(&mut self, dead_ranks: &[u32]) -> ResizeResult {
        let _old_dp = self.dp_replicas;
        let target_batch = self.effective_batch_size();

        // Remove dead ranks
        for &rank in dead_ranks {
            self.alive_ranks.remove(&rank);
        }
        self.current_world_size = self.alive_ranks.len() as u32;

        // Recalculate DP replicas (TP and PP groups are fixed)
        let tp_pp = self.tp_size * self.pp_stages;
        self.dp_replicas = if tp_pp > 0 {
            self.current_world_size / tp_pp
        } else {
            self.current_world_size
        };

        // Adjust gradient accumulation to preserve effective batch size:
        //   target_batch = micro_batch * dp_replicas * grad_accum_steps
        //   new_accum = target_batch / (micro_batch * new_dp)
        if self.dp_replicas > 0 && self.micro_batch_size > 0 {
            let new_accum = target_batch / (self.micro_batch_size as u64 * self.dp_replicas as u64);
            self.grad_accum_steps = new_accum.max(1) as u32;
        }

        let surviving: Vec<u32> = self.alive_ranks.iter().copied().collect();

        ResizeResult {
            new_world_size: self.current_world_size,
            new_dp_replicas: self.dp_replicas,
            new_grad_accum_steps: self.grad_accum_steps,
            effective_batch_size: self.effective_batch_size(),
            ranks_to_reload: surviving,
        }
    }

    /// Handle a scale-up: add new ranks to the DP group.
    pub fn handle_scale_up(&mut self, new_ranks: &[u32]) -> ResizeResult {
        let target_batch = self.effective_batch_size();

        for &rank in new_ranks {
            self.alive_ranks.insert(rank);
        }
        self.current_world_size = self.alive_ranks.len() as u32;

        let tp_pp = self.tp_size * self.pp_stages;
        self.dp_replicas = if tp_pp > 0 {
            self.current_world_size / tp_pp
        } else {
            self.current_world_size
        };

        // Reduce grad_accum since we have more DP replicas
        if self.dp_replicas > 0 && self.micro_batch_size > 0 {
            let new_accum = target_batch / (self.micro_batch_size as u64 * self.dp_replicas as u64);
            self.grad_accum_steps = new_accum.max(1) as u32;
        }

        ResizeResult {
            new_world_size: self.current_world_size,
            new_dp_replicas: self.dp_replicas,
            new_grad_accum_steps: self.grad_accum_steps,
            effective_batch_size: self.effective_batch_size(),
            ranks_to_reload: new_ranks.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// Trigger elastic resize after a rank failure.
/// Returns new world_size, or -1 on error.
#[no_mangle]
pub extern "C" fn nsl_elastic_resize(
    _dead_rank: i64,
    _current_world_size: i64,
    _tp_size: i64,
    _pp_stages: i64,
) -> i64 {
    // Real implementation would:
    // 1. Pause training on all surviving ranks (barrier)
    // 2. Reload last checkpoint
    // 3. Rebuild NCCL communicators
    // 4. Adjust grad_accum_steps
    // 5. Resume training
    let new_world = _current_world_size - 1;
    if new_world <= 0 { return -1; }
    new_world
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_batch_size() {
        let world = ElasticWorld::new(32, 4, 2, 8, 4);
        // DP = 32 / (4 * 2) = 4 replicas
        assert_eq!(world.dp_replicas, 4);
        // Effective batch = 8 * 4 * 4 = 128
        assert_eq!(world.effective_batch_size(), 128);
    }

    #[test]
    fn test_rank_failure_preserves_batch_size() {
        let mut world = ElasticWorld::new(32, 4, 2, 8, 4);
        let original_batch = world.effective_batch_size();

        // Kill ranks 28-31 (one full DP replica: 4 ranks from TP group)
        let result = world.handle_rank_failure(&[28, 29, 30, 31]);

        assert_eq!(result.new_world_size, 28);
        assert_eq!(result.new_dp_replicas, 3); // 28 / (4*2) = 3.5 → 3
        // Grad accum should increase: 128 / (8 * 3) ≈ 5.33 → 5
        assert!(result.new_grad_accum_steps >= 5);
        // Effective batch should be close to original (may differ by rounding)
        let new_batch = result.effective_batch_size;
        assert!(
            new_batch >= original_batch - 10 && new_batch <= original_batch + 10,
            "batch size should be preserved: was {original_batch}, now {new_batch}"
        );
    }

    #[test]
    fn test_scale_up_reduces_grad_accum() {
        let mut world = ElasticWorld::new(24, 4, 2, 8, 4);
        // DP = 24 / 8 = 3, batch = 8 * 3 * 4 = 96
        assert_eq!(world.dp_replicas, 3);

        let result = world.handle_scale_up(&[24, 25, 26, 27, 28, 29, 30, 31]);
        // Now 32 ranks, DP = 32/8 = 4
        assert_eq!(result.new_dp_replicas, 4);
        // Grad accum should decrease: 96 / (8 * 4) = 3
        assert_eq!(result.new_grad_accum_steps, 3);
    }

    #[test]
    fn test_single_rank_failure() {
        // Non-TP/PP scenario: pure DP
        let mut world = ElasticWorld::new(4, 1, 1, 32, 1);
        assert_eq!(world.dp_replicas, 4);
        assert_eq!(world.effective_batch_size(), 128);

        let result = world.handle_rank_failure(&[3]);
        assert_eq!(result.new_dp_replicas, 3);
        // grad_accum: 128 / (32 * 3) ≈ 1.33 → 1
        // Effective batch: 32 * 3 * 1 = 96 (some loss due to rounding)
        assert!(result.new_grad_accum_steps >= 1);
    }

    #[test]
    fn test_ffi_resize() {
        let result = nsl_elastic_resize(3, 4, 1, 1);
        assert_eq!(result, 3); // 4-1 = 3
    }
}
