//! WRGA Innovation 5: Memory-Planned Activation Sharing.
//!
//! Consumes the VarId liveness information produced by `wrga_prune` and emits
//! a compact slot-assignment plan: each live activation is mapped to a
//! physical memory slot, with non-interfering activations sharing slots.
//!
//! This is a specialisation of `memory_planner.rs`'s interference-graph
//! colouring — we reuse the same greedy-colouring heuristic but drive it from
//! Wengert VarIds rather than full-program tensor allocations.  Because the
//! pruned backward graph is extremely sparse (85%+ of the forward ops carry no
//! adjoint), the resulting interference graph is nearly trivial to colour,
//! producing the 1.9 GB peak-memory estimate cited in Section 5.2.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::wengert::{VarId, WengertList};

/// Physical slot index assigned to an activation.
pub type SlotId = u32;

/// A single activation's slot assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotAssignment {
    pub var: VarId,
    pub slot: SlotId,
    /// Estimated byte size of the activation (computed from the Wengert var
    /// size table when available, else 0).
    pub size_bytes: u64,
    /// Lifetime `[birth, death]` in Wengert program-point units.  Two
    /// activations with disjoint lifetimes may share a slot.
    pub birth: u32,
    pub death: u32,
}

/// Summary of the memory plan for reporting.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryPlanStats {
    pub live_activations: usize,
    pub slots_used: usize,
    pub naive_peak_bytes: u64,
    pub planned_peak_bytes: u64,
}

impl MemoryPlanStats {
    pub fn reuse_ratio(&self) -> f64 {
        if self.live_activations == 0 {
            return 0.0;
        }
        1.0 - (self.slots_used as f64 / self.live_activations as f64)
    }

    pub fn byte_savings(&self) -> u64 {
        self.naive_peak_bytes.saturating_sub(self.planned_peak_bytes)
    }
}

/// Full memory plan.
#[derive(Debug, Clone, Default)]
pub struct MemoryPlan {
    pub assignments: Vec<SlotAssignment>,
    pub stats: MemoryPlanStats,
}

/// Compute `[birth, death]` ranges for each VarId in the *pruned* list.
///
/// Birth is the program point (op index) at which the VarId is produced;
/// death is the last point at which it is consumed.  VarIds in
/// `extra_live_at_end` (e.g. loss, gradient outputs) are kept alive to the
/// end so they aren't prematurely reused.
fn compute_var_ranges(
    list: &WengertList,
    interesting: &BTreeSet<VarId>,
    extra_live_at_end: &BTreeSet<VarId>,
) -> BTreeMap<VarId, (u32, u32)> {
    let last_pp = list.ops.len().saturating_sub(1) as u32;
    let mut births: HashMap<VarId, u32> = HashMap::new();
    let mut deaths: HashMap<VarId, u32> = HashMap::new();
    for (pp, op) in list.ops.iter().enumerate() {
        let pp = pp as u32;
        if interesting.contains(&op.result) {
            births.entry(op.result).or_insert(pp);
            deaths.entry(op.result).or_insert(pp);
        }
        for &inp in &op.inputs {
            if interesting.contains(&inp) {
                let d = deaths.entry(inp).or_insert(pp);
                if pp > *d {
                    *d = pp;
                }
            }
        }
    }
    for &v in extra_live_at_end {
        if interesting.contains(&v) {
            deaths.insert(v, last_pp);
        }
    }
    let mut out = BTreeMap::new();
    for (v, birth) in births.into_iter() {
        let death = deaths.get(&v).copied().unwrap_or(birth);
        out.insert(v, (birth, death));
    }
    out
}

/// Optional size hint table: `VarId → size_bytes`.  Call sites that know the
/// shape/dtype of a var can populate this; missing entries default to 0 and
/// the slab planner will treat them as "unknown but live".
pub type SizeHints = HashMap<VarId, u64>;

/// Build a slot-assignment plan for the given activation set.
///
/// Uses a standard greedy interval-graph colouring:
///   1. Sort activations by birth, ties broken by (-death, var).
///   2. For each activation, pick the *smallest* slot whose previous tenant
///      has already died by the activation's birth; if none fits, open a new
///      slot.
///   3. Record the assignment.
///
/// For interval graphs (which a Wengert straight-line liveness graph is) this
/// greedy policy is optimal: it produces the minimum number of slots.
pub fn plan_memory(
    list: &WengertList,
    activation_live: &BTreeSet<VarId>,
    sizes: &SizeHints,
    extra_live_at_end: &BTreeSet<VarId>,
) -> MemoryPlan {
    let ranges = compute_var_ranges(list, activation_live, extra_live_at_end);
    if ranges.is_empty() {
        return MemoryPlan::default();
    }

    // Sort by birth; ties by longer-death first (so longer-lived intervals get
    // placed into a fresh slot earlier — reduces later fragmentation).
    let mut ordered: Vec<(VarId, u32, u32)> = ranges
        .iter()
        .map(|(v, (b, d))| (*v, *b, *d))
        .collect();
    ordered.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.0.cmp(&b.0))
    });

    // For each slot, track the death-point of its current tenant + its max
    // size used (so a slot's size is the max of everything it ever held).
    let mut slot_free_at: Vec<u32> = Vec::new(); // "slot is free *after* this pp"
    let mut slot_sizes: Vec<u64> = Vec::new();
    let mut assignments: Vec<SlotAssignment> = Vec::with_capacity(ordered.len());
    let mut naive_peak: u64 = 0;

    for (var, birth, death) in ordered {
        let size = sizes.get(&var).copied().unwrap_or(0);
        naive_peak = naive_peak.saturating_add(size);

        // First-fit: pick the smallest existing slot that has already freed.
        let mut chosen = None;
        for (idx, &free_at) in slot_free_at.iter().enumerate() {
            if free_at < birth {
                chosen = Some(idx);
                break;
            }
        }
        let slot = match chosen {
            Some(idx) => {
                slot_free_at[idx] = death;
                if size > slot_sizes[idx] {
                    slot_sizes[idx] = size;
                }
                idx as SlotId
            }
            None => {
                slot_free_at.push(death);
                slot_sizes.push(size);
                (slot_free_at.len() - 1) as SlotId
            }
        };
        assignments.push(SlotAssignment {
            var,
            slot,
            size_bytes: size,
            birth,
            death,
        });
    }

    let planned_peak: u64 = slot_sizes.iter().sum();
    let stats = MemoryPlanStats {
        live_activations: assignments.len(),
        slots_used: slot_sizes.len(),
        naive_peak_bytes: naive_peak,
        planned_peak_bytes: planned_peak,
    };

    MemoryPlan {
        assignments,
        stats,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use std::collections::HashMap;

    fn op(id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>) -> WengertOp {
        WengertOp {
            id,
            result,
            op: o,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    /// Chain graph x → a → b → c → d (all relu).  All adjacent lifetimes
    /// should share a single slot.
    #[test]
    fn chain_graph_shares_slot() {
        let list = WengertList {
            ops: vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Relu, vec![0]),
                op(2, 2, PrimalOp::Relu, vec![1]),
                op(3, 3, PrimalOp::Relu, vec![2]),
                op(4, 4, PrimalOp::Relu, vec![3]),
            ],
            output: 4,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let live: BTreeSet<_> = [1u32, 2, 3, 4].iter().copied().collect();
        let plan = plan_memory(&list, &live, &HashMap::new(), &BTreeSet::new());
        // Classic interval-graph colouring: producer reads its input at the
        // same program point as it produces its output, so at each PP two
        // tensors coexist momentarily.  The minimum # of slots for a 4-step
        // chain is 2 — strictly better than the naive 4 slots.
        assert_eq!(plan.stats.slots_used, 2);
        assert_eq!(plan.stats.live_activations, 4);
        assert!((plan.stats.reuse_ratio() - 0.5).abs() < 1e-9);
    }

    /// Branching: x is consumed by both y and z which live in parallel.
    /// Two simultaneously-live intervals → two slots.
    #[test]
    fn branch_graph_needs_two_slots() {
        let list = WengertList {
            ops: vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Relu, vec![0]),
                op(2, 2, PrimalOp::Sigmoid, vec![0]),
                op(3, 3, PrimalOp::Add, vec![1, 2]),
            ],
            output: 3,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let live: BTreeSet<_> = [1u32, 2].iter().copied().collect();
        let plan = plan_memory(&list, &live, &HashMap::new(), &BTreeSet::new());
        assert_eq!(plan.stats.slots_used, 2);
    }

    #[test]
    fn size_hints_set_peak_bytes() {
        let list = WengertList {
            ops: vec![
                op(0, 0, PrimalOp::Input("x".into()), vec![]),
                op(1, 1, PrimalOp::Relu, vec![0]),
                op(2, 2, PrimalOp::Relu, vec![1]),
            ],
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let live: BTreeSet<_> = [1u32].iter().copied().collect();
        let mut sizes = HashMap::new();
        sizes.insert(1u32, 1024);
        let plan = plan_memory(&list, &live, &sizes, &BTreeSet::new());
        assert_eq!(plan.stats.naive_peak_bytes, 1024);
        assert_eq!(plan.stats.planned_peak_bytes, 1024);
    }

    #[test]
    fn empty_live_set_yields_empty_plan() {
        let list = WengertList {
            ops: vec![op(0, 0, PrimalOp::Input("x".into()), vec![])],
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let plan = plan_memory(&list, &BTreeSet::new(), &HashMap::new(), &BTreeSet::new());
        assert_eq!(plan.stats.live_activations, 0);
        assert_eq!(plan.stats.slots_used, 0);
    }

    #[test]
    fn reuse_ratio_zero_when_one_slot_per_live() {
        let stats = MemoryPlanStats {
            live_activations: 4,
            slots_used: 4,
            naive_peak_bytes: 400,
            planned_peak_bytes: 400,
        };
        assert!((stats.reuse_ratio() - 0.0).abs() < 1e-9);
        assert_eq!(stats.byte_savings(), 0);
    }

    #[test]
    fn reuse_ratio_half_when_two_to_one() {
        let stats = MemoryPlanStats {
            live_activations: 4,
            slots_used: 2,
            naive_peak_bytes: 400,
            planned_peak_bytes: 200,
        };
        assert!((stats.reuse_ratio() - 0.5).abs() < 1e-9);
        assert_eq!(stats.byte_savings(), 200);
    }
}
