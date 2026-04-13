//! CPDT — per-layer communication schedule.
//!
//! Given a [`ZeroConfig`], emit an ordered list of async
//! allgather/reducescatter launches and matching waits so that every
//! layer's parameter gather overlaps with the previous layer's compute.
//!
//! The schedule is a compile-time constant — the downstream Cranelift
//! backend emits `nsl_cpdt_allgather` / `nsl_cpdt_reducescatter` runtime
//! calls in exactly this order.  No Python-level `register_comm_hook`
//! dispatch, no dynamic collective insertion.

use serde::Serialize;

use crate::cpdt_zero::ZeroConfig;

/// Kind of collective operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CollectiveKind {
    /// Replicate a sharded tensor across the group.
    AllGather,
    /// Sum + scatter gradients across the group.
    ReduceScatter,
    /// Full broadcast of a reduced result.
    AllReduce,
    /// Route per-token activations to per-expert owners.
    AllToAll,
}

impl CollectiveKind {
    pub fn as_str(self) -> &'static str {
        match self {
            CollectiveKind::AllGather => "allgather",
            CollectiveKind::ReduceScatter => "reducescatter",
            CollectiveKind::AllReduce => "allreduce",
            CollectiveKind::AllToAll => "alltoall",
        }
    }
}

/// Single scheduled communication op.
#[derive(Debug, Clone, Serialize)]
pub struct CommOp {
    pub layer: u32,
    pub kind: CollectiveKind,
    /// Shard group size participating in this op.
    pub group_size: u32,
    /// Whether the group spans an inter-node boundary.
    pub inter_node: bool,
    /// Bytes moved per rank.
    pub bytes: u64,
    /// Execution order index within the backward/forward pass (smaller =
    /// earlier).
    pub order: u32,
    /// Asynchronous launch marker — the runtime fires this op without
    /// blocking; a matching `wait` comes later.
    pub async_: bool,
    /// Human-readable rationale for the report.
    pub rationale: String,
}

/// Full ordered schedule.
#[derive(Debug, Clone, Default, Serialize)]
pub struct CommSchedule {
    pub forward: Vec<CommOp>,
    pub backward: Vec<CommOp>,
    pub optimizer: Vec<CommOp>,
}

impl CommSchedule {
    pub fn total_ops(&self) -> usize {
        self.forward.len() + self.backward.len() + self.optimizer.len()
    }
    pub fn total_bytes(&self) -> u64 {
        let f: u64 = self.forward.iter().map(|c| c.bytes).sum();
        let b: u64 = self.backward.iter().map(|c| c.bytes).sum();
        let o: u64 = self.optimizer.iter().map(|c| c.bytes).sum();
        f + b + o
    }
    pub fn async_count(&self) -> usize {
        self.forward.iter().filter(|c| c.async_).count()
            + self.backward.iter().filter(|c| c.async_).count()
            + self.optimizer.iter().filter(|c| c.async_).count()
    }
}

/// Build the schedule for a given `ZeroConfig` and per-layer param-byte
/// vector.  The forward pass prefetches parameters one layer ahead; the
/// backward pass reduces gradients as they're produced.
pub fn build_schedule(
    config: ZeroConfig,
    per_layer_param_bytes: &[u64],
) -> CommSchedule {
    let mut schedule = CommSchedule::default();
    let n = per_layer_param_bytes.len() as u32;
    let mut order = 0u32;

    // Forward pass: allgather params for layer l *while* layer l-1
    // computes.  We emit an async allgather at layer-0 start, then
    // one more per layer.
    if config.s_p > 1 {
        for layer in 0..n {
            let bytes = per_layer_param_bytes[layer as usize];
            schedule.forward.push(CommOp {
                layer,
                kind: CollectiveKind::AllGather,
                group_size: config.s_p,
                inter_node: config.mesh_p_inter,
                bytes,
                order,
                async_: true,
                rationale: format!("prefetch layer {layer} params"),
            });
            order += 1;
        }
    }

    // Backward pass: reducescatter the gradient of layer l as soon as
    // it's produced — runs asynchronously with layer l-1's backward
    // compute.  Iterate in reverse topological order.
    if config.s_g > 1 {
        let mut layer = n as i32 - 1;
        while layer >= 0 {
            let l = layer as u32;
            let bytes = per_layer_param_bytes[l as usize];
            schedule.backward.push(CommOp {
                layer: l,
                kind: CollectiveKind::ReduceScatter,
                group_size: config.s_g,
                inter_node: config.mesh_g_inter,
                bytes,
                order,
                async_: true,
                rationale: format!("scatter grad {l} while layer {} computes", layer.saturating_sub(1).max(0)),
            });
            order += 1;
            layer -= 1;
        }
    }

    // Optimizer pass: if optimizer-state shard factor is non-trivial
    // we need an allgather of the updated parameters before the next
    // forward pass.  One per layer, issued sequentially — the
    // compute-side optimizer step is short so async overlap is
    // marginal.
    if config.s_os > 1 {
        for layer in 0..n {
            let bytes = per_layer_param_bytes[layer as usize];
            schedule.optimizer.push(CommOp {
                layer,
                kind: CollectiveKind::AllGather,
                group_size: config.s_os,
                inter_node: config.mesh_os_inter,
                bytes,
                order,
                async_: false,
                rationale: format!("gather updated layer {layer} params"),
            });
            order += 1;
        }
    }

    schedule
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sizes() -> Vec<u64> {
        vec![6_000_000; 8]
    }

    #[test]
    fn ddp_config_produces_no_comm() {
        let cfg = ZeroConfig {
            s_p: 1,
            s_g: 1,
            s_os: 1,
            mesh_p_inter: false,
            mesh_g_inter: false,
            mesh_os_inter: false,
        };
        let sched = build_schedule(cfg, &sizes());
        assert_eq!(sched.total_ops(), 0);
        assert_eq!(sched.total_bytes(), 0);
    }

    #[test]
    fn zero_3_schedule_has_one_op_per_layer_per_phase() {
        let cfg = ZeroConfig::zero_3(8);
        let sched = build_schedule(cfg, &sizes());
        assert_eq!(sched.forward.len(), 8);
        assert_eq!(sched.backward.len(), 8);
        assert_eq!(sched.optimizer.len(), 8);
    }

    #[test]
    fn backward_ops_are_in_reverse_order() {
        let cfg = ZeroConfig::zero_2(8);
        let sched = build_schedule(cfg, &sizes());
        assert_eq!(sched.backward[0].layer, 7);
        assert_eq!(sched.backward[sched.backward.len() - 1].layer, 0);
    }

    #[test]
    fn forward_ops_async_backward_ops_async() {
        let cfg = ZeroConfig::zero_3(8);
        let sched = build_schedule(cfg, &sizes());
        for op in &sched.forward {
            assert!(op.async_);
        }
        for op in &sched.backward {
            assert!(op.async_);
        }
    }

    #[test]
    fn optimizer_ops_are_sync() {
        let cfg = ZeroConfig::zero_1(8);
        let sched = build_schedule(cfg, &sizes());
        for op in &sched.optimizer {
            assert!(!op.async_);
        }
    }

    #[test]
    fn inter_node_flag_carries_through() {
        let mut cfg = ZeroConfig::zero_3(8);
        cfg.mesh_g_inter = true;
        let sched = build_schedule(cfg, &sizes());
        assert!(sched.backward.iter().all(|op| op.inter_node));
        assert!(sched.forward.iter().all(|op| !op.inter_node));
    }

    #[test]
    fn async_count_matches_forward_plus_backward() {
        let sched = build_schedule(ZeroConfig::zero_3(8), &sizes());
        assert_eq!(sched.async_count(), sched.forward.len() + sched.backward.len());
    }

    #[test]
    fn collective_kind_names_are_stable() {
        assert_eq!(CollectiveKind::AllGather.as_str(), "allgather");
        assert_eq!(CollectiveKind::ReduceScatter.as_str(), "reducescatter");
        assert_eq!(CollectiveKind::AllReduce.as_str(), "allreduce");
        assert_eq!(CollectiveKind::AllToAll.as_str(), "alltoall");
    }

    #[test]
    fn totals_match_concatenation() {
        let sched = build_schedule(ZeroConfig::zero_3(8), &sizes());
        let total = sched.total_ops();
        assert_eq!(
            total,
            sched.forward.len() + sched.backward.len() + sched.optimizer.len()
        );
    }
}
