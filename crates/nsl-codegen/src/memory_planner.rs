//! M36: Compile-time memory planning — liveness analysis, interference graph, slab assignment.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub type ProgramPoint = u32;
pub type TensorAllocId = u32;

/// How a tensor's size is known at compile time.
#[derive(Debug, Clone, PartialEq)]
pub enum SizeKind {
    /// Size is a compile-time constant.
    Static(u64),
    /// Size has a compile-time upper bound (from Dim::Bounded).
    Bounded { upper: u64, symbolic_expr: String },
    /// Size is truly dynamic — cannot be planned, falls back to runtime alloc.
    Dynamic,
}

/// A tensor allocation site discovered during liveness analysis.
#[derive(Debug, Clone)]
pub struct TensorAlloc {
    pub id: TensorAllocId,
    /// Human-readable name (from let-binding or synthetic).
    pub name: String,
    /// Size in bytes (computed from shape * dtype.byte_width()).
    pub size_bytes: u64,
    /// Program point where this tensor is first allocated.
    pub birth: ProgramPoint,
    /// Program point where this tensor is last used.
    pub death: ProgramPoint,
    /// Source location for diagnostics.
    pub source_loc: String,
    /// Whether this tensor's size is statically known or bounded.
    pub size_kind: SizeKind,
    /// True if this tensor is saved for the backward pass (DataRequired ops).
    /// When set, the slab planner must not reuse this tensor's slot until the
    /// backward pass completes, preventing pointer-aliasing bugs in gradient
    /// accumulation (where grad_map keys are tensor addresses).
    pub saved_for_backward: bool,
}

impl TensorAlloc {
    /// Returns the effective size for slab planning (upper bound for Bounded, exact for Static).
    pub fn plan_size(&self) -> Option<u64> {
        match &self.size_kind {
            SizeKind::Static(n) => Some(*n),
            SizeKind::Bounded { upper, .. } => Some(*upper),
            SizeKind::Dynamic => None,
        }
    }

    /// Returns true if this allocation can be slab-planned.
    pub fn is_plannable(&self) -> bool {
        !matches!(self.size_kind, SizeKind::Dynamic)
    }
}

// ---------------------------------------------------------------------------
// Liveness analysis
// ---------------------------------------------------------------------------

/// Walks program statements to record tensor allocation sites and lifetimes.
pub struct LivenessAnalyzer {
    allocs: Vec<TensorAlloc>,
    /// Maps variable name -> TensorAllocId for currently-live tensors.
    live_set: HashMap<String, TensorAllocId>,
    current_pp: ProgramPoint,
}

impl Default for LivenessAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl LivenessAnalyzer {
    pub fn new() -> Self {
        Self {
            allocs: Vec::new(),
            live_set: HashMap::new(),
            current_pp: 0,
        }
    }

    /// Record a tensor allocation at the current program point.
    pub fn record_alloc(&mut self, name: &str, size_bytes: u64, size_kind: SizeKind, loc: &str) {
        self.record_alloc_ex(name, size_bytes, size_kind, loc, false);
    }

    /// Record a tensor allocation with explicit saved-for-backward annotation.
    pub fn record_alloc_ex(&mut self, name: &str, size_bytes: u64, size_kind: SizeKind, loc: &str, saved_for_backward: bool) {
        let id = self.allocs.len() as TensorAllocId;
        self.allocs.push(TensorAlloc {
            id,
            name: name.to_string(),
            size_bytes,
            birth: self.current_pp,
            death: self.current_pp, // updated on each use
            source_loc: loc.to_string(),
            size_kind,
            saved_for_backward,
        });
        self.live_set.insert(name.to_string(), id);
    }

    /// Record a use of a tensor — extends its lifetime to the current program point.
    pub fn record_use(&mut self, name: &str) {
        if let Some(&id) = self.live_set.get(name) {
            self.allocs[id as usize].death = self.current_pp;
        }
    }

    /// Advance the program point counter.
    pub fn advance(&mut self) {
        self.current_pp += 1;
    }

    /// Current program point.
    pub fn current_pp(&self) -> ProgramPoint {
        self.current_pp
    }

    /// Consume the analyzer and return all discovered allocations.
    pub fn finish(self) -> Vec<TensorAlloc> {
        self.allocs
    }
}

// ---------------------------------------------------------------------------
// Interference graph
// ---------------------------------------------------------------------------

/// Returns true if two tensor lifetimes overlap (half-open intervals).
pub fn intervals_overlap(a: &TensorAlloc, b: &TensorAlloc) -> bool {
    a.birth < b.death && b.birth < a.death
}

/// Adjacency-list interference graph for tensor allocations.
pub struct InterferenceGraph {
    adj: Vec<HashSet<TensorAllocId>>,
    #[allow(dead_code)]
    num_tensors: usize,
}

impl InterferenceGraph {
    /// Build interference graph from a set of tensor allocations.
    /// Tensors marked `saved_for_backward` interfere with all other tensors
    /// to prevent slab slot reuse that could alias gradient accumulation keys.
    pub fn build(allocs: &[TensorAlloc]) -> Self {
        let n = allocs.len();
        let mut adj = vec![HashSet::new(); n];

        for i in 0..n {
            if !allocs[i].is_plannable() {
                continue;
            }
            for j in (i + 1)..n {
                if !allocs[j].is_plannable() {
                    continue;
                }
                // Saved-for-backward tensors must never share a slab slot
                // with any other tensor — their addresses are used as
                // gradient map keys and must remain unique.
                let force_interfere =
                    allocs[i].saved_for_backward || allocs[j].saved_for_backward;
                if force_interfere || intervals_overlap(&allocs[i], &allocs[j]) {
                    adj[i].insert(j as TensorAllocId);
                    adj[j].insert(i as TensorAllocId);
                }
            }
        }

        Self { adj, num_tensors: n }
    }

    /// Check if two tensors interfere.
    pub fn interferes(&self, a: TensorAllocId, b: TensorAllocId) -> bool {
        self.adj[a as usize].contains(&b)
    }
}

// ---------------------------------------------------------------------------
// Slab assignment
// ---------------------------------------------------------------------------

/// Align a byte offset up to the given alignment (must be power of 2).
pub fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// A memory slot within the slab — multiple non-overlapping tensors can share a slot.
#[derive(Debug, Clone)]
pub struct MemorySlot {
    pub id: u32,
    pub offset: u64,
    pub size: u64,
    pub assigned: Vec<TensorAllocId>,
}

/// The complete memory plan: slot layout + per-tensor offset assignments.
#[derive(Debug, Clone)]
pub struct SlabPlan {
    pub slots: Vec<MemorySlot>,
    /// Total slab size in bytes (including alignment padding).
    pub total_bytes: u64,
    /// TensorAllocId -> (slot_id, byte_offset_in_slab).
    pub assignments: HashMap<TensorAllocId, (u32, u64)>,
    /// Naive allocation total (sum of all planned tensor sizes) for savings report.
    pub naive_total: u64,
    /// Total alignment padding bytes (aligned_size - actual_size across all tensors).
    pub padding_bytes: u64,
}

impl SlabPlan {
    /// Fragmentation ratio: padding_bytes / total_bytes.
    /// 0.0 means no wasted alignment padding; higher = more fragmentation.
    pub fn fragmentation_ratio(&self) -> f64 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.padding_bytes as f64 / self.total_bytes as f64
        }
    }

    /// Memory savings as a fraction (0.0 to 1.0).
    pub fn savings_fraction(&self) -> f64 {
        if self.naive_total == 0 {
            return 0.0;
        }
        1.0 - (self.total_bytes as f64 / self.naive_total as f64)
    }
}

/// GPU alignment for slab offsets (256 bytes).
const SLAB_ALIGNMENT: u64 = 256;

// ---------------------------------------------------------------------------
// Rematerialization
// ---------------------------------------------------------------------------

/// A recomputation point: the tensor is freed early and recomputed later.
#[derive(Debug, Clone)]
pub struct RecompPoint {
    /// The original tensor allocation that is being rematerialized.
    pub original_alloc_id: TensorAllocId,
    /// Program point where the tensor is freed early (before natural death).
    pub evict_at: ProgramPoint,
    /// Program point where the tensor is recomputed.
    pub recompute_at: ProgramPoint,
    /// Name of the operation to re-execute.
    pub op_name: String,
    /// Tensor allocations of the inputs (must be live at recompute_at).
    pub input_alloc_ids: Vec<TensorAllocId>,
}

/// Descriptor of a tensor's producing operation, for rematerialization scoring.
#[derive(Debug, Clone)]
pub struct TensorOpInfo {
    pub alloc_id: TensorAllocId,
    pub op_name: String,
    pub input_alloc_ids: Vec<TensorAllocId>,
    pub dtype_bytes: u64,
}

/// Run the rematerialization pass: identify tensors to recompute and shorten their live ranges.
///
/// Returns a list of recomputation points and a modified set of allocations with shortened ranges.
/// The caller should use the modified allocations for interference graph construction and slab assignment.
pub fn rematerialize(
    allocs: &[TensorAlloc],
    op_infos: &[TensorOpInfo],
    threshold: f64,
) -> (Vec<TensorAlloc>, Vec<RecompPoint>) {
    use crate::cost_model::{score_remat_candidate, RecomputeClass};

    let mut modified_allocs: Vec<TensorAlloc> = allocs.to_vec();
    let mut recomp_points: Vec<RecompPoint> = Vec::new();

    // Score all candidates
    let mut candidates: Vec<(usize, crate::cost_model::RematCandidate)> = Vec::new();

    for info in op_infos {
        let idx = info.alloc_id as usize;
        if idx >= allocs.len() { continue; }
        let alloc = &allocs[idx];
        if !alloc.is_plannable() { continue; }

        let live_range = alloc.death.saturating_sub(alloc.birth);
        if live_range < 2 { continue; } // too short to benefit

        // Check if all inputs are live at a potential recompute point
        // (simplified: check if inputs are live at the original death - 1)
        let recompute_at = alloc.death.saturating_sub(1);
        let inputs_live = info.input_alloc_ids.iter().all(|&input_id| {
            let iid = input_id as usize;
            iid < allocs.len() && allocs[iid].death >= recompute_at
        });

        if let Some(candidate) = score_remat_candidate(
            info.alloc_id,
            alloc.size_bytes,
            live_range,
            &info.op_name,
            info.dtype_bytes,
            inputs_live,
        ) {
            if candidate.score >= threshold || candidate.recompute_class == RecomputeClass::Free {
                candidates.push((idx, candidate));
            }
        }
    }

    // Sort by score descending (best candidates first)
    candidates.sort_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));

    // Apply: shorten live ranges for accepted candidates
    for (idx, candidate) in &candidates {
        let alloc = &allocs[*idx];
        let evict_at = alloc.birth + (alloc.death - alloc.birth) / 3; // free after first third
        let recompute_at = alloc.death.saturating_sub(1);

        // Shorten the original allocation's death to evict_at
        modified_allocs[*idx].death = evict_at;

        // Find the op info for input IDs
        let info = op_infos.iter().find(|i| i.alloc_id == candidate.alloc_id);
        let input_ids = info.map(|i| i.input_alloc_ids.clone()).unwrap_or_default();
        let op_name = info.map(|i| i.op_name.clone()).unwrap_or_default();

        recomp_points.push(RecompPoint {
            original_alloc_id: candidate.alloc_id,
            evict_at,
            recompute_at,
            op_name,
            input_alloc_ids: input_ids,
        });
    }

    (modified_allocs, recomp_points)
}

/// Assign tensors to memory slots using best-fit-decreasing (BFD).
///
/// For each tensor (sorted by size descending), scans ALL existing non-conflicting
/// slots and picks the one with minimum wasted space (smallest slot that fits).
/// This produces tighter packing than first-fit-decreasing (FFD) by avoiding
/// placing small tensors into large slots when a closer-fitting slot exists.
///
/// Tensors with SizeKind::Dynamic are skipped.
pub fn plan_slab(allocs: &[TensorAlloc], interference: &InterferenceGraph) -> SlabPlan {
    // Sort tensor IDs by size (largest first) for decreasing order
    let mut sorted: Vec<TensorAllocId> = allocs
        .iter()
        .filter(|a| a.is_plannable())
        .map(|a| a.id)
        .collect();
    sorted.sort_by(|&a, &b| {
        allocs[b as usize]
            .size_bytes
            .cmp(&allocs[a as usize].size_bytes)
    });

    let naive_total: u64 = sorted
        .iter()
        .map(|&id| allocs[id as usize].size_bytes)
        .sum();

    let mut slots: Vec<MemorySlot> = Vec::new();
    let mut assignments = HashMap::new();
    let mut total_padding: u64 = 0;

    for &tensor_id in &sorted {
        let tensor = &allocs[tensor_id as usize];
        let aligned_size = align_up(tensor.size_bytes, SLAB_ALIGNMENT);
        total_padding += aligned_size - tensor.size_bytes;

        // Best-fit: scan ALL non-conflicting slots, pick smallest that fits
        let mut best_slot_idx: Option<usize> = None;
        let mut best_waste: u64 = u64::MAX;

        for (idx, slot) in slots.iter().enumerate() {
            let conflicts = slot
                .assigned
                .iter()
                .any(|&other_id| interference.interferes(tensor_id, other_id));
            if conflicts {
                continue;
            }

            // Slot must be at least as big as the tensor (slot.size is already the max
            // of all occupants, and we'll take the max again below)
            let slot_size_after = slot.size.max(tensor.size_bytes);
            let waste = slot_size_after - tensor.size_bytes;
            if waste < best_waste {
                best_waste = waste;
                best_slot_idx = Some(idx);
            }
        }

        match best_slot_idx {
            Some(idx) => {
                slots[idx].size = slots[idx].size.max(tensor.size_bytes);
                slots[idx].assigned.push(tensor_id);
            }
            None => {
                let slot_id = slots.len() as u32;
                slots.push(MemorySlot {
                    id: slot_id,
                    offset: 0,
                    size: tensor.size_bytes,
                    assigned: vec![tensor_id],
                });
            }
        }
    }

    // Compute byte offsets: each slot starts after the previous, aligned to SLAB_ALIGNMENT
    let mut current_offset: u64 = 0;
    for slot in &mut slots {
        slot.offset = align_up(current_offset, SLAB_ALIGNMENT);
        current_offset = slot.offset + slot.size;
        for &tid in &slot.assigned {
            assignments.insert(tid, (slot.id, slot.offset));
        }
    }

    SlabPlan {
        total_bytes: current_offset,
        slots,
        assignments,
        naive_total,
        padding_bytes: total_padding,
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Format a byte count as human-readable (B, KB, MB, GB).
pub fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Parse a human-readable size string (e.g., "8GB", "512MB") to bytes.
pub fn parse_vram_budget(s: &str) -> Option<u64> {
    let s = s.trim().to_uppercase();
    let (num_str, multiplier) = if s.ends_with("GB") {
        (&s[..s.len() - 2], 1024 * 1024 * 1024_u64)
    } else if s.ends_with("MB") {
        (&s[..s.len() - 2], 1024 * 1024_u64)
    } else if s.ends_with("KB") {
        (&s[..s.len() - 2], 1024_u64)
    } else if s.ends_with('B') {
        (&s[..s.len() - 1], 1_u64)
    } else {
        return None;
    };
    num_str
        .trim()
        .parse::<f64>()
        .ok()
        .map(|n| (n * multiplier as f64) as u64)
}

/// Format a memory plan report for stderr output.
pub fn format_memory_report(allocs: &[TensorAlloc], plan: &SlabPlan) -> String {
    let planned_count = plan.assignments.len();
    let dynamic_count = allocs
        .iter()
        .filter(|a| matches!(a.size_kind, SizeKind::Dynamic))
        .count();
    let total_count = allocs.len();

    let mut report = String::new();
    report.push_str("Memory Plan:\n");
    report.push_str(&format!(
        "  Planned (slab): {}\n",
        format_bytes(plan.total_bytes)
    ));
    report.push_str(&format!("  Slots: {}\n", plan.slots.len()));
    report.push_str(&format!(
        "  Tensor sites: {} ({} planned, {} dynamic)\n",
        total_count, planned_count, dynamic_count
    ));
    report.push_str(&format!(
        "  Naive total:  {}\n",
        format_bytes(plan.naive_total)
    ));
    report.push_str(&format!(
        "  Savings:      {:.1}%\n",
        plan.savings_fraction() * 100.0
    ));

    // Top consumers (sorted by size)
    let mut sorted: Vec<&TensorAlloc> = allocs.iter().filter(|a| a.is_plannable()).collect();
    sorted.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

    if !sorted.is_empty() {
        report.push_str("\n  Top consumers:\n");
        for (i, a) in sorted.iter().take(5).enumerate() {
            report.push_str(&format!(
                "    {}. {} ({}) [{}]\n",
                i + 1,
                a.name,
                format_bytes(a.size_bytes),
                a.source_loc
            ));
        }
    }

    report
}

/// Check VRAM budget and return error message if exceeded.
pub fn check_vram_budget(plan: &SlabPlan, budget_bytes: u64) -> Option<String> {
    if plan.total_bytes <= budget_bytes {
        return None;
    }

    let pct = (plan.total_bytes as f64 / budget_bytes as f64) * 100.0;
    Some(format!(
        "error: model exceeds VRAM budget\n\
         \x20 Budget:  {}\n\
         \x20 Planned: {} ({:.0}% of budget)\n",
        format_bytes(budget_bytes),
        format_bytes(plan.total_bytes),
        pct,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Task 1: TensorAlloc + SizeKind ---

    #[test]
    fn test_plan_size_static() {
        let a = TensorAlloc {
            id: 0, name: "x".into(), size_bytes: 1024,
            birth: 0, death: 5, source_loc: "test:1".into(),
            size_kind: SizeKind::Static(1024), saved_for_backward: false,
        };
        assert_eq!(a.plan_size(), Some(1024));
        assert!(a.is_plannable());
    }

    #[test]
    fn test_plan_size_bounded() {
        let a = TensorAlloc {
            id: 0, name: "x".into(), size_bytes: 4096,
            birth: 0, death: 5, source_loc: "test:1".into(),
            size_kind: SizeKind::Bounded { upper: 4096, symbolic_expr: "B*1024".into() }, saved_for_backward: false,
        };
        assert_eq!(a.plan_size(), Some(4096));
        assert!(a.is_plannable());
    }

    #[test]
    fn test_plan_size_dynamic() {
        let a = TensorAlloc {
            id: 0, name: "x".into(), size_bytes: 0,
            birth: 0, death: 5, source_loc: "test:1".into(),
            size_kind: SizeKind::Dynamic, saved_for_backward: false,
        };
        assert_eq!(a.plan_size(), None);
        assert!(!a.is_plannable());
    }

    // --- Task 2: LivenessAnalyzer ---

    #[test]
    fn test_liveness_linear() {
        let mut analyzer = LivenessAnalyzer::new();
        analyzer.record_alloc("a", 1024, SizeKind::Static(1024), "line:1");
        analyzer.advance();
        analyzer.record_alloc("b", 2048, SizeKind::Static(2048), "line:2");
        analyzer.record_use("a");
        analyzer.advance();
        analyzer.record_alloc("c", 512, SizeKind::Static(512), "line:3");
        analyzer.record_use("a");
        analyzer.advance();
        analyzer.record_use("b");
        analyzer.advance();
        analyzer.record_use("c");

        let allocs = analyzer.finish();
        assert_eq!(allocs.len(), 3);
        assert_eq!(allocs[0].birth, 0);
        assert_eq!(allocs[0].death, 2);
        assert_eq!(allocs[1].birth, 1);
        assert_eq!(allocs[1].death, 3);
        assert_eq!(allocs[2].birth, 2);
        assert_eq!(allocs[2].death, 4);
    }

    #[test]
    fn test_liveness_reuse_name() {
        let mut analyzer = LivenessAnalyzer::new();
        analyzer.record_alloc("x", 100, SizeKind::Static(100), "line:1");
        analyzer.advance();
        analyzer.record_use("x");
        analyzer.advance();
        analyzer.record_alloc("x", 200, SizeKind::Static(200), "line:3");
        analyzer.advance();
        analyzer.record_use("x");

        let allocs = analyzer.finish();
        assert_eq!(allocs.len(), 2);
        assert_eq!(allocs[0].size_bytes, 100);
        assert_eq!(allocs[0].death, 1);
        assert_eq!(allocs[1].size_bytes, 200);
        assert_eq!(allocs[1].birth, 2);
        assert_eq!(allocs[1].death, 3);
    }

    #[test]
    fn test_liveness_unused_tensor() {
        let mut analyzer = LivenessAnalyzer::new();
        analyzer.record_alloc("unused", 64, SizeKind::Static(64), "line:1");
        analyzer.advance();

        let allocs = analyzer.finish();
        assert_eq!(allocs.len(), 1);
        assert_eq!(allocs[0].birth, 0);
        assert_eq!(allocs[0].death, 0);
    }

    #[test]
    fn test_liveness_reassign_without_use() {
        let mut analyzer = LivenessAnalyzer::new();
        analyzer.record_alloc("x", 100, SizeKind::Static(100), "line:1");
        analyzer.advance();
        analyzer.record_alloc("x", 200, SizeKind::Static(200), "line:2");
        analyzer.advance();
        analyzer.record_use("x");

        let allocs = analyzer.finish();
        assert_eq!(allocs.len(), 2);
        assert_eq!(allocs[0].death, 0);
        assert_eq!(allocs[1].birth, 1);
        assert_eq!(allocs[1].death, 2);
    }

    // --- Task 3: InterferenceGraph ---

    #[test]
    fn test_intervals_overlap() {
        let a = TensorAlloc { id: 0, name: "a".into(), size_bytes: 100, birth: 0, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(100), saved_for_backward: false };
        let b = TensorAlloc { id: 1, name: "b".into(), size_bytes: 200, birth: 3, death: 8, source_loc: "".into(), size_kind: SizeKind::Static(200), saved_for_backward: false };
        let c = TensorAlloc { id: 2, name: "c".into(), size_bytes: 150, birth: 6, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(150), saved_for_backward: false };

        assert!(intervals_overlap(&a, &b));
        assert!(!intervals_overlap(&a, &c));
        assert!(intervals_overlap(&b, &c));
    }

    #[test]
    fn test_interference_graph_construction() {
        let allocs = vec![
            TensorAlloc { id: 0, name: "a".into(), size_bytes: 100, birth: 0, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(100), saved_for_backward: false },
            TensorAlloc { id: 1, name: "b".into(), size_bytes: 200, birth: 3, death: 8, source_loc: "".into(), size_kind: SizeKind::Static(200), saved_for_backward: false },
            TensorAlloc { id: 2, name: "c".into(), size_bytes: 150, birth: 6, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(150), saved_for_backward: false },
        ];
        let graph = InterferenceGraph::build(&allocs);

        assert!(graph.interferes(0, 1));
        assert!(!graph.interferes(0, 2));
        assert!(graph.interferes(1, 2));
    }

    #[test]
    fn test_interference_same_point_no_overlap() {
        let a = TensorAlloc { id: 0, name: "a".into(), size_bytes: 100, birth: 0, death: 3, source_loc: "".into(), size_kind: SizeKind::Static(100), saved_for_backward: false };
        let b = TensorAlloc { id: 1, name: "b".into(), size_bytes: 200, birth: 3, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(200), saved_for_backward: false };
        assert!(!intervals_overlap(&a, &b));
    }

    #[test]
    fn test_interference_graph_skips_dynamic() {
        let allocs = vec![
            TensorAlloc { id: 0, name: "static_a".into(), size_bytes: 100, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(100), saved_for_backward: false },
            TensorAlloc { id: 1, name: "dynamic_b".into(), size_bytes: 0, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Dynamic, saved_for_backward: false },
            TensorAlloc { id: 2, name: "static_c".into(), size_bytes: 200, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(200), saved_for_backward: false },
        ];
        let graph = InterferenceGraph::build(&allocs);

        assert!(graph.interferes(0, 2));
        assert!(!graph.interferes(0, 1));
        assert!(!graph.interferes(1, 2));
    }

    // --- Task 4: Slab assignment ---

    #[test]
    fn test_plan_slab_reuse() {
        let allocs = vec![
            TensorAlloc { id: 0, name: "a".into(), size_bytes: 1024, birth: 0, death: 3, source_loc: "".into(), size_kind: SizeKind::Static(1024), saved_for_backward: false },
            TensorAlloc { id: 1, name: "b".into(), size_bytes: 2048, birth: 1, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(2048), saved_for_backward: false },
            TensorAlloc { id: 2, name: "c".into(), size_bytes: 512, birth: 4, death: 7, source_loc: "".into(), size_kind: SizeKind::Static(512), saved_for_backward: false },
        ];
        let graph = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &graph);

        assert_eq!(plan.slots.len(), 2);
        assert!(plan.total_bytes >= 2048 + 1024);
        assert!(plan.total_bytes <= 2048 + 1024 + 512);

        let (slot_a, _) = plan.assignments[&0];
        let (slot_c, _) = plan.assignments[&2];
        assert_eq!(slot_a, slot_c);

        let (slot_b, _) = plan.assignments[&1];
        assert_ne!(slot_a, slot_b);
    }

    #[test]
    fn test_plan_slab_skips_dynamic() {
        let allocs = vec![
            TensorAlloc { id: 0, name: "static_t".into(), size_bytes: 1024, birth: 0, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(1024), saved_for_backward: false },
            TensorAlloc { id: 1, name: "dynamic_t".into(), size_bytes: 0, birth: 2, death: 4, source_loc: "".into(), size_kind: SizeKind::Dynamic, saved_for_backward: false },
        ];
        let graph = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &graph);

        assert_eq!(plan.assignments.len(), 1);
        assert!(plan.assignments.contains_key(&0));
        assert!(!plan.assignments.contains_key(&1));
    }

    #[test]
    fn test_plan_slab_empty() {
        let allocs: Vec<TensorAlloc> = Vec::new();
        let graph = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &graph);
        assert_eq!(plan.total_bytes, 0);
        assert!(plan.slots.is_empty());
    }

    #[test]
    fn test_plan_slab_all_overlap() {
        let allocs = vec![
            TensorAlloc { id: 0, name: "a".into(), size_bytes: 100, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(100), saved_for_backward: false },
            TensorAlloc { id: 1, name: "b".into(), size_bytes: 200, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(200), saved_for_backward: false },
            TensorAlloc { id: 2, name: "c".into(), size_bytes: 300, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(300), saved_for_backward: false },
        ];
        let graph = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &graph);

        assert_eq!(plan.slots.len(), 3);
        assert!(plan.total_bytes >= 600);
    }

    #[test]
    fn test_plan_slab_sequential_full_reuse() {
        let allocs = vec![
            TensorAlloc { id: 0, name: "a".into(), size_bytes: 1024, birth: 0, death: 2, source_loc: "".into(), size_kind: SizeKind::Static(1024), saved_for_backward: false },
            TensorAlloc { id: 1, name: "b".into(), size_bytes: 512, birth: 3, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(512), saved_for_backward: false },
            TensorAlloc { id: 2, name: "c".into(), size_bytes: 768, birth: 6, death: 8, source_loc: "".into(), size_kind: SizeKind::Static(768), saved_for_backward: false },
        ];
        let graph = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &graph);

        assert_eq!(plan.slots.len(), 1);
        assert!(plan.total_bytes >= 1024);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(255, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
    }

    // --- Task 8: Reporting ---

    #[test]
    fn test_format_memory_report() {
        let allocs = vec![
            TensorAlloc { id: 0, name: "weight_q".into(), size_bytes: 4096, birth: 0, death: 10, source_loc: "model.nsl:5".into(), size_kind: SizeKind::Static(4096), saved_for_backward: false },
            TensorAlloc { id: 1, name: "hidden".into(), size_bytes: 2048, birth: 2, death: 6, source_loc: "model.nsl:8".into(), size_kind: SizeKind::Static(2048), saved_for_backward: false },
            TensorAlloc { id: 2, name: "output".into(), size_bytes: 1024, birth: 7, death: 10, source_loc: "model.nsl:12".into(), size_kind: SizeKind::Static(1024), saved_for_backward: false },
        ];
        let graph = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &graph);

        let report = format_memory_report(&allocs, &plan);
        assert!(report.contains("Memory Plan"));
        assert!(report.contains("Planned (slab):"));
        assert!(report.contains("Savings:"));
        assert!(report.contains("weight_q"));
    }

    #[test]
    fn test_format_bytes_human() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    #[test]
    fn test_parse_vram_budget() {
        assert_eq!(parse_vram_budget("8GB"), Some(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_vram_budget("512MB"), Some(512 * 1024 * 1024));
        assert_eq!(parse_vram_budget("1024KB"), Some(1024 * 1024));
        assert_eq!(parse_vram_budget("100B"), Some(100));
        assert_eq!(parse_vram_budget("1.5GB"), Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64));
        assert_eq!(parse_vram_budget("0GB"), Some(0));
        assert_eq!(parse_vram_budget("garbage"), None);
        assert_eq!(parse_vram_budget(""), None);
        assert_eq!(parse_vram_budget("GB"), None);
    }

    #[test]
    fn test_savings_fraction_zero_allocs() {
        let plan = SlabPlan {
            slots: vec![], total_bytes: 0, assignments: HashMap::new(), naive_total: 0, padding_bytes: 0,
        };
        assert_eq!(plan.savings_fraction(), 0.0);
    }

    #[test]
    fn test_check_vram_budget_passes() {
        let plan = SlabPlan {
            slots: vec![], total_bytes: 1024, assignments: HashMap::new(), naive_total: 2048, padding_bytes: 0,
        };
        assert!(check_vram_budget(&plan, 2048).is_none());
    }

    #[test]
    fn test_check_vram_budget_fails() {
        let plan = SlabPlan {
            slots: vec![], total_bytes: 4096, assignments: HashMap::new(), naive_total: 4096, padding_bytes: 0,
        };
        let err = check_vram_budget(&plan, 2048);
        assert!(err.is_some());
        assert!(err.unwrap().contains("exceeds VRAM budget"));
    }

    // ── Best-fit-decreasing tests ─────────────────────────────────────

    #[test]
    fn test_bfd_prefers_tighter_slot() {
        // Scenario: slot A (1000 bytes), slot B (600 bytes) both available.
        // New tensor of 580 bytes should go to slot B (waste=20) not slot A (waste=420).
        //
        // Setup: 3 tensors, first two occupy separate slots (overlapping lifetimes),
        // third has non-overlapping lifetime with both and should pick the smaller slot.
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "big".into(), size_bytes: 1000, birth: 0, death: 5,
                source_loc: String::new(), size_kind: SizeKind::Static(1000), saved_for_backward: false,
            },
            TensorAlloc {
                id: 1, name: "medium".into(), size_bytes: 600, birth: 0, death: 5,
                source_loc: String::new(), size_kind: SizeKind::Static(600), saved_for_backward: false,
            },
            TensorAlloc {
                id: 2, name: "fits_medium".into(), size_bytes: 580, birth: 6, death: 10,
                source_loc: String::new(), size_kind: SizeKind::Static(580), saved_for_backward: false,
            },
        ];

        let ig = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &ig);

        // Tensor 2 should share a slot with tensor 1 (600, waste=20)
        // not tensor 0 (1000, waste=420)
        let (slot_t1, _) = plan.assignments[&1];
        let (slot_t2, _) = plan.assignments[&2];
        assert_eq!(slot_t2, slot_t1,
            "BFD should assign 580-byte tensor to 600-byte slot, not 1000-byte slot");

        // Only 2 slots needed (1000 + 600)
        assert_eq!(plan.slots.len(), 2);
    }

    #[test]
    fn test_bfd_reuses_slot_for_non_overlapping() {
        // Two tensors with non-overlapping lifetimes should share one slot
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "a".into(), size_bytes: 1024, birth: 0, death: 5,
                source_loc: String::new(), size_kind: SizeKind::Static(1024), saved_for_backward: false,
            },
            TensorAlloc {
                id: 1, name: "b".into(), size_bytes: 512, birth: 6, death: 10,
                source_loc: String::new(), size_kind: SizeKind::Static(512), saved_for_backward: false,
            },
        ];

        let ig = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &ig);

        assert_eq!(plan.slots.len(), 1, "non-overlapping tensors should share 1 slot");
        assert_eq!(plan.total_bytes, 1024, "slot size = max(1024, 512)");
    }

    #[test]
    fn test_fragmentation_zero_for_aligned_sizes() {
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "aligned".into(), size_bytes: 256, birth: 0, death: 5,
                source_loc: String::new(), size_kind: SizeKind::Static(256), saved_for_backward: false,
            },
            TensorAlloc {
                id: 1, name: "aligned2".into(), size_bytes: 512, birth: 0, death: 5,
                source_loc: String::new(), size_kind: SizeKind::Static(512), saved_for_backward: false,
            },
        ];

        let ig = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &ig);

        assert_eq!(plan.padding_bytes, 0, "aligned sizes should have zero padding");
        assert!((plan.fragmentation_ratio() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fragmentation_nonzero_for_unaligned_sizes() {
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "unaligned".into(), size_bytes: 100, birth: 0, death: 5,
                source_loc: String::new(), size_kind: SizeKind::Static(100), saved_for_backward: false,
            },
        ];

        let ig = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &ig);

        // 100 bytes -> 256 aligned, padding = 156
        assert_eq!(plan.padding_bytes, 156);
        assert!(plan.fragmentation_ratio() > 0.5);
    }

    // ── Rematerialization tests ──────────────────────────────────────

    #[test]
    fn test_remat_shortens_live_range() {
        // Tensor produced by relu (cheap), large, long live range → good candidate
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "input".into(), size_bytes: 4096, birth: 0, death: 100,
                source_loc: String::new(), size_kind: SizeKind::Static(4096), saved_for_backward: false,
            },
            TensorAlloc {
                id: 1, name: "relu_out".into(), size_bytes: 4096, birth: 5, death: 95,
                source_loc: String::new(), size_kind: SizeKind::Static(4096), saved_for_backward: false,
            },
        ];

        let op_infos = vec![
            TensorOpInfo {
                alloc_id: 1, op_name: "relu".into(),
                input_alloc_ids: vec![0], dtype_bytes: 4,
            },
        ];

        let (modified, recomp_points) = rematerialize(&allocs, &op_infos, crate::cost_model::REMAT_SCORE_THRESHOLD);

        // relu_out should be a remat candidate (cheap, large range)
        assert!(!recomp_points.is_empty(), "should have at least one recomp point");
        assert_eq!(recomp_points[0].original_alloc_id, 1);

        // Modified alloc should have shortened death
        assert!(modified[1].death < allocs[1].death,
            "death should be shortened: was {}, now {}", allocs[1].death, modified[1].death);
    }

    #[test]
    fn test_remat_rejects_matmul() {
        // matmul is expensive — should NOT be rematerialized
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "a".into(), size_bytes: 4096, birth: 0, death: 100,
                source_loc: String::new(), size_kind: SizeKind::Static(4096), saved_for_backward: false,
            },
            TensorAlloc {
                id: 1, name: "mm_out".into(), size_bytes: 4096, birth: 5, death: 95,
                source_loc: String::new(), size_kind: SizeKind::Static(4096), saved_for_backward: false,
            },
        ];

        let op_infos = vec![
            TensorOpInfo {
                alloc_id: 1, op_name: "matmul".into(),
                input_alloc_ids: vec![0], dtype_bytes: 4,
            },
        ];

        let (_, recomp_points) = rematerialize(&allocs, &op_infos, crate::cost_model::REMAT_SCORE_THRESHOLD);
        assert!(recomp_points.is_empty(), "matmul should NOT be rematerialized");
    }

    #[test]
    fn test_remat_free_ops_always_accepted() {
        // reshape is free — always accepted regardless of score
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "src".into(), size_bytes: 100, birth: 0, death: 100,
                source_loc: String::new(), size_kind: SizeKind::Static(100), saved_for_backward: false,
            },
            TensorAlloc {
                id: 1, name: "reshaped".into(), size_bytes: 100, birth: 5, death: 95,
                source_loc: String::new(), size_kind: SizeKind::Static(100), saved_for_backward: false,
            },
        ];

        let op_infos = vec![
            TensorOpInfo {
                alloc_id: 1, op_name: "reshape".into(),
                input_alloc_ids: vec![0], dtype_bytes: 4,
            },
        ];

        let (_, recomp_points) = rematerialize(&allocs, &op_infos, f64::MAX); // very high threshold
        assert!(!recomp_points.is_empty(), "free ops should always be accepted");
    }

    #[test]
    fn test_remat_inputs_must_be_live() {
        // Input dies before recompute point → cannot rematerialize
        let allocs = vec![
            TensorAlloc {
                id: 0, name: "input".into(), size_bytes: 4096, birth: 0, death: 10,
                source_loc: String::new(), size_kind: SizeKind::Static(4096), saved_for_backward: false,
            },
            TensorAlloc {
                id: 1, name: "relu_out".into(), size_bytes: 4096, birth: 5, death: 95,
                source_loc: String::new(), size_kind: SizeKind::Static(4096), saved_for_backward: false,
            },
        ];

        let op_infos = vec![
            TensorOpInfo {
                alloc_id: 1, op_name: "relu".into(),
                input_alloc_ids: vec![0], dtype_bytes: 4,
            },
        ];

        let (_, recomp_points) = rematerialize(&allocs, &op_infos, crate::cost_model::REMAT_SCORE_THRESHOLD);
        // Input dies at 10 but recompute point is at 94 — cannot remat
        assert!(recomp_points.is_empty(), "cannot remat if inputs are dead");
    }
}
