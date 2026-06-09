//! M36: Compile-time memory planning — liveness analysis, interference graph, slab assignment.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::wengert::VarId;

// ---------------------------------------------------------------------------
// WRGA Milestone B.2 Task 3: typed dual-keyspace allocation keys
// ---------------------------------------------------------------------------

/// Typed key into the real allocator. Enforces at the type level that a
/// caller cannot pass a `String` (weight name) where a `VarId` (activation
/// id) is expected, or vice versa. This is deliberately a plain enum rather
/// than a pair of newtypes so the keyspace boundary is checked at pattern
/// sites.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum AllocationKey {
    Weight(String),
    Activation(VarId),
}

/// Physical slot id assigned by the real allocator (distinct from the
/// `wrga_memory::SlotId` produced by the WRGA planner).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RealSlotId(pub u32);

// ---------------------------------------------------------------------------
// WRGA Milestone B.1 Task 4: `apply_wrga_hints`
// ---------------------------------------------------------------------------

/// Conservative coalescing pass: consume a `WrgaPlan`'s `MemoryPlan` as a
/// hint to the real memory planner.  Two WRGA-assigned vars may share a
/// real slot only when:
///   1. they share a WRGA slot (WRGA already proved non-overlap),
///   2. their `size_bytes` match exactly,
///   3. their `[birth, death]` intervals do not overlap.
///
/// The function records pre/post *real* slot counts on debug side-channels
/// for test observation and returns the post-hint slot count.  It does NOT
/// mutate the string-keyed `LivenessAnalyzer`: that allocator works on
/// AST-level names and has no natural mapping from Wengert `VarId`.  The
/// aggressive version — folding these merges into the live interference
/// graph — is scheduled for Milestone B.2 alongside the MMA epilogue.
pub fn apply_wrga_hints(plan: &crate::wrga::WrgaPlan) -> usize {
    let mem = &plan.memory;

    // Group WRGA assignments by WRGA slot id.
    let mut wrga_groups: HashMap<crate::wrga_memory::SlotId, Vec<&crate::wrga_memory::SlotAssignment>> =
        HashMap::new();
    for a in &mem.assignments {
        wrga_groups.entry(a.slot).or_default().push(a);
    }

    // Pre-hint real slot count: one slot per assignment (the "no coalescing"
    // baseline — equivalent to the allocator not yet having seen the hints).
    let pre = mem.assignments.len();

    // Post-hint: greedy union-find across each WRGA group, but only merging
    // pairs that pass the size+liveness safety gates.  This mirrors what
    // `try_merge_into_slot` would do on a VarId-keyed allocator.
    let all_slots: BTreeSet<_> = (0..mem.assignments.len()).collect();
    let mut parent: Vec<usize> = (0..mem.assignments.len()).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            let r = find(parent, parent[x]);
            parent[x] = r;
        }
        parent[x]
    }

    // Index assignments by their WRGA slot group for iteration.
    let mut by_slot: HashMap<crate::wrga_memory::SlotId, Vec<usize>> = HashMap::new();
    for (i, a) in mem.assignments.iter().enumerate() {
        by_slot.entry(a.slot).or_default().push(i);
    }
    for indices in by_slot.values() {
        if indices.len() < 2 {
            continue;
        }
        // Anchor at the first assignment; merge each compatible successor.
        let anchor = indices[0];
        for &other in &indices[1..] {
            if try_merge_pair(&mem.assignments, anchor, other) {
                let ra = find(&mut parent, anchor);
                let ro = find(&mut parent, other);
                if ra != ro {
                    parent[ro] = ra;
                }
            }
        }
    }

    let mut roots = BTreeSet::new();
    for s in &all_slots {
        let r = find(&mut parent, *s);
        roots.insert(r);
    }
    let post = roots.len();

    crate::debug_set_allocator_slot_count_pre_hint(pre);
    crate::debug_set_allocator_slot_count_post_hint(post);
    post
}

/// WRGA Milestone B.2 Task 3: free-function shim around
/// `LivenessAnalyzer::consume_hints`, mirroring `apply_wrga_hints`'s
/// call site in `stmt.rs`.
pub fn consume_hints(
    allocator: &mut LivenessAnalyzer,
    plan: &crate::wrga::WrgaPlan,
) -> usize {
    allocator.consume_hints(plan)
}

/// Safety gate for merging two WRGA slot assignments into a single real
/// slot.  Requires: WRGA already agreed they don't overlap (implicit via
/// shared WRGA slot), byte sizes equal, and `[birth, death]` intervals do
/// not overlap (belt-and-braces check).
fn try_merge_pair(
    assignments: &[crate::wrga_memory::SlotAssignment],
    a: usize,
    b: usize,
) -> bool {
    let aa = &assignments[a];
    let bb = &assignments[b];
    if aa.size_bytes != bb.size_bytes || aa.size_bytes == 0 {
        return false;
    }
    // Non-overlap (half-open intervals): death_a <= birth_b || death_b <= birth_a.
    // Using <= matches intervals_overlap's strict-< semantics so that adjacent
    // intervals ([0,5) and [5,10)) are correctly treated as non-overlapping.
    let disjoint = aa.death <= bb.birth || bb.death <= aa.birth;
    if !disjoint {
        return false;
    }
    true
}


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
    /// WRGA B.2 Task 3: typed-keyed real slot map.
    pub(crate) slots: HashMap<AllocationKey, RealSlotId>,
    pub(crate) slot_sizes: HashMap<RealSlotId, u64>,
    pub(crate) slot_liveness: HashMap<RealSlotId, Vec<(u32, u32)>>,
    pub(crate) next_slot_id: u32,
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
            slots: HashMap::new(),
            slot_sizes: HashMap::new(),
            slot_liveness: HashMap::new(),
            next_slot_id: 0,
        }
    }

    // -----------------------------------------------------------------
    // WRGA B.2 Task 3: typed-key allocation + consume_hints
    // -----------------------------------------------------------------

    /// Record a fresh activation allocation keyed by its Wengert VarId,
    /// giving it its own `RealSlotId`. This is the entry point the real
    /// allocator uses before `consume_hints` runs.
    pub fn record_activation_alloc(&mut self, var: VarId, size_bytes: u64) {
        let key = AllocationKey::Activation(var);
        let slot = RealSlotId(self.next_slot_id);
        self.next_slot_id += 1;
        self.slots.insert(key, slot);
        self.slot_sizes.insert(slot, size_bytes);
        self.slot_liveness.insert(slot, Vec::new());
    }

    pub fn real_slot_for_activation(&self, var: VarId) -> Option<RealSlotId> {
        self.slots.get(&AllocationKey::Activation(var)).copied()
    }

    /// Try to fold `var`'s real slot into `target`. Requires equal sizes
    /// and a live target slot; returns false otherwise.
    pub fn try_merge_activation_into_slot(
        &mut self,
        var: VarId,
        target: RealSlotId,
    ) -> bool {
        let Some(source) = self.real_slot_for_activation(var) else { return false; };
        if source == target { return false; }
        let Some(&src_size) = self.slot_sizes.get(&source) else { return false; };
        let Some(&tgt_size) = self.slot_sizes.get(&target) else { return false; };
        if src_size != tgt_size { return false; }
        self.slots.insert(AllocationKey::Activation(var), target);
        self.slot_sizes.remove(&source);
        if let Some(src_liv) = self.slot_liveness.remove(&source) {
            self.slot_liveness.entry(target).or_default().extend(src_liv);
        }
        true
    }

    /// Consume a WRGA `MemoryPlan` as merge hints on the real allocator.
    /// Folds activations that share a WRGA slot and whose real slots have
    /// equal size and disjoint liveness. Returns the post-hint distinct
    /// real-slot count (also recorded on the
    /// `ALLOC_SLOTS_POST_HINT` side-channel for test observation).
    pub fn consume_hints(&mut self, plan: &crate::wrga::WrgaPlan) -> usize {
        let mut by_wrga_slot: HashMap<crate::wrga_memory::SlotId, Vec<(VarId, u32, u32, u64)>> =
            HashMap::new();
        for a in &plan.memory.assignments {
            by_wrga_slot
                .entry(a.slot)
                .or_default()
                .push((a.var, a.birth, a.death, a.size_bytes));
        }

        crate::debug_bump_consume_hints_calls();

        for (_wrga_slot, group) in by_wrga_slot {
            if group.len() < 2 { continue; }
            let (first_var, first_birth, first_death, first_size) = group[0];
            let Some(target) = self.real_slot_for_activation(first_var) else { continue; };
            // Seed target liveness with the anchor's range.
            self.slot_liveness
                .entry(target)
                .or_default()
                .push((first_birth, first_death));
            for &(other_var, o_birth, o_death, o_size) in &group[1..] {
                if o_size != first_size { continue; }
                let overlap = self.slot_liveness
                    .get(&target)
                    .map(|ranges| ranges.iter().any(|&(b, d)| !(o_death <= b || d <= o_birth)))
                    .unwrap_or(false);
                if overlap { continue; }
                if self.try_merge_activation_into_slot(other_var, target) {
                    self.slot_liveness.entry(target).or_default().push((o_birth, o_death));
                }
            }
        }

        let distinct: BTreeSet<_> = self.slots.values().copied().collect();
        let post = distinct.len();
        crate::debug_set_allocator_slot_count_post_hint(post);
        post
    }

    /// Record a tensor allocation at the current program point.
    pub fn record_alloc(&mut self, name: &str, size_bytes: u64, size_kind: SizeKind, loc: &str) {
        self.record_alloc_ex(name, size_bytes, size_kind, loc, false);
    }

    /// Record a tensor allocation with explicit saved-for-backward annotation.
    pub fn record_alloc_ex(
        &mut self,
        name: &str,
        size_bytes: u64,
        size_kind: SizeKind,
        loc: &str,
        saved_for_backward: bool,
    ) {
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

    /// Extend the death of any tensor whose birth < loop_entry to at least loop_exit.
    /// This prevents the slab planner from reusing slots for tensors that are
    /// live across loop iterations (conservative: any tensor alive before the loop
    /// must survive the full loop body).
    pub fn extend_loop_live(&mut self, loop_entry: ProgramPoint, loop_exit: ProgramPoint) {
        for alloc in &mut self.allocs {
            if alloc.birth < loop_entry && alloc.death >= loop_entry && alloc.death < loop_exit {
                alloc.death = loop_exit;
            }
        }
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
                let force_interfere = allocs[i].saved_for_backward || allocs[j].saved_for_backward;
                if force_interfere || intervals_overlap(&allocs[i], &allocs[j]) {
                    adj[i].insert(j as TensorAllocId);
                    adj[j].insert(i as TensorAllocId);
                }
            }
        }

        Self {
            adj,
            num_tensors: n,
        }
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
        if idx >= allocs.len() {
            continue;
        }
        let alloc = &allocs[idx];
        if !alloc.is_plannable() {
            continue;
        }

        let live_range = alloc.death.saturating_sub(alloc.birth);
        if live_range < 2 {
            continue;
        } // too short to benefit

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
    candidates.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
    sorted.sort_by_key(|a| std::cmp::Reverse(a.size_bytes));

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
// AST-level liveness analysis
// ---------------------------------------------------------------------------

/// Compute the byte size of a tensor type from its shape and dtype.
/// Returns None if any dimension is non-concrete (symbolic, dynamic, etc.).
pub fn compute_tensor_bytes(
    shape: &nsl_semantic::types::Shape,
    dtype: &nsl_semantic::types::DType,
) -> Option<(u64, SizeKind)> {
    let mut total_elems: u64 = 1;
    let mut has_bounded = false;
    for dim in &shape.dims {
        match dim {
            nsl_semantic::types::Dim::Concrete(n) => {
                total_elems = total_elems.saturating_mul(*n as u64);
            }
            nsl_semantic::types::Dim::Bounded { upper_bound, .. } => {
                // Accumulate the upper bound and continue — do NOT early-return
                // here. Remaining Concrete dims after a Bounded dim must still
                // be multiplied in or the slab allocation is fatally undersized.
                total_elems = total_elems.saturating_mul(*upper_bound as u64);
                has_bounded = true;
            }
            _ => return None, // Dynamic / Wildcard / Symbolic — can't plan
        }
    }
    let bytes = total_elems * dtype_byte_size(dtype);
    if has_bounded {
        Some((
            bytes,
            SizeKind::Bounded {
                upper: bytes,
                symbolic_expr: format!("{:?}", shape),
            },
        ))
    } else {
        Some((bytes, SizeKind::Static(bytes)))
    }
}

/// Return the byte size of a DType.
fn dtype_byte_size(dtype: &nsl_semantic::types::DType) -> u64 {
    use nsl_semantic::types::DType;
    match dtype {
        DType::F64 | DType::Int64 => 8,
        DType::F32 | DType::Int32 => 4,
        DType::Fp16 | DType::Bf16 | DType::Int16 => 2,
        DType::Int8 | DType::Uint8 | DType::Bool => 1,
        DType::Fp8E4m3 | DType::Fp8E5m2 => 1,
        DType::Int4 => 1,                       // rounded up
        // M35.1 BitNet: ternary storage atom is 1 byte (packed 2-bit×4 trits; unpacked i8).
        DType::TernaryPacked | DType::TernaryUnpacked => 1,
        DType::Custom(_) | DType::Unknown => 4, // default to f32 size for unknown dtypes
    }
}

/// Walk top-level statements to discover tensor allocations and their lifetimes.
///
/// Returns a list of TensorAlloc records that can be fed into `plan_slab()`.
/// Only considers tensor bindings with statically-known shapes.
pub fn analyze_ast_liveness(
    stmts: &[nsl_ast::stmt::Stmt],
    type_map: &nsl_semantic::checker::TypeMap,
    interner: &nsl_lexer::Interner,
) -> Vec<TensorAlloc> {
    let mut analyzer = LivenessAnalyzer::new();
    let mut name_to_id: HashMap<String, TensorAllocId> = HashMap::new();

    walk_stmts(stmts, type_map, interner, &mut analyzer, &mut name_to_id);

    analyzer.finish()
}

fn walk_stmts(
    stmts: &[nsl_ast::stmt::Stmt],
    type_map: &nsl_semantic::checker::TypeMap,
    interner: &nsl_lexer::Interner,
    analyzer: &mut LivenessAnalyzer,
    name_to_id: &mut HashMap<String, TensorAllocId>,
) {
    for stmt in stmts {
        walk_stmt(stmt, type_map, interner, analyzer, name_to_id);
        analyzer.advance();
    }
}

fn walk_stmt(
    stmt: &nsl_ast::stmt::Stmt,
    type_map: &nsl_semantic::checker::TypeMap,
    interner: &nsl_lexer::Interner,
    analyzer: &mut LivenessAnalyzer,
    name_to_id: &mut HashMap<String, TensorAllocId>,
) {
    use nsl_ast::stmt::StmtKind;

    match &stmt.kind {
        StmtKind::VarDecl { pattern, value, .. } => {
            // Record uses in the RHS expression
            if let Some(expr) = value {
                walk_expr_uses(expr, type_map, interner, analyzer, name_to_id);
            }

            // Check if this binding is a tensor with static shape
            if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                if let Some(name_str) = interner.resolve(sym.0) {
                    let name = name_str.to_string();
                    // Type map stores types on expr IDs, not stmt IDs.
                    // Check the RHS expression's type for tensor shape info.
                    // Only plan GPU tensors (device = Cuda) — CPU tensors use heap.
                    let expr_ty = value.as_ref().and_then(|v| type_map.get(&v.id));
                    if let Some(ty) = expr_ty {
                        if let Some((shape, dtype, device)) = ty.as_tensor_parts() {
                            let is_gpu = matches!(device, nsl_semantic::types::Device::Cuda(_));
                            if is_gpu {
                                if let Some((size_bytes, size_kind)) =
                                    compute_tensor_bytes(shape, dtype)
                                {
                                    if size_bytes > 0 {
                                        let loc = format!("byte {}", stmt.span.start.0);
                                        analyzer.record_alloc(&name, size_bytes, size_kind, &loc);
                                        let id = analyzer.current_pp();
                                        name_to_id.insert(name, id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        StmtKind::Expr(expr) | StmtKind::Return(Some(expr)) => {
            walk_expr_uses(expr, type_map, interner, analyzer, name_to_id);
        }

        StmtKind::Assign { target, value, .. } => {
            walk_expr_uses(target, type_map, interner, analyzer, name_to_id);
            walk_expr_uses(value, type_map, interner, analyzer, name_to_id);
        }

        StmtKind::If {
            condition,
            then_block,
            elif_clauses,
            else_block,
        } => {
            walk_expr_uses(condition, type_map, interner, analyzer, name_to_id);
            walk_stmts(&then_block.stmts, type_map, interner, analyzer, name_to_id);
            for (cond, block) in elif_clauses {
                walk_expr_uses(cond, type_map, interner, analyzer, name_to_id);
                walk_stmts(&block.stmts, type_map, interner, analyzer, name_to_id);
            }
            if let Some(block) = else_block {
                walk_stmts(&block.stmts, type_map, interner, analyzer, name_to_id);
            }
        }

        StmtKind::For { body, iterable, .. } => {
            walk_expr_uses(iterable, type_map, interner, analyzer, name_to_id);
            let loop_entry = analyzer.current_pp();
            walk_stmts(&body.stmts, type_map, interner, analyzer, name_to_id);
            let loop_exit = analyzer.current_pp();
            // Extend death of any tensor live at loop entry to loop exit
            // (conservative: loop-carried values must survive the full loop)
            analyzer.extend_loop_live(loop_entry, loop_exit);
        }

        StmtKind::While { condition, body } => {
            walk_expr_uses(condition, type_map, interner, analyzer, name_to_id);
            let loop_entry = analyzer.current_pp();
            walk_stmts(&body.stmts, type_map, interner, analyzer, name_to_id);
            let loop_exit = analyzer.current_pp();
            analyzer.extend_loop_live(loop_entry, loop_exit);
        }

        _ => {} // FnDef, ModelDef, etc. handled separately
    }
}

/// Walk an expression to record uses of tensor variables.
fn walk_expr_uses(
    expr: &nsl_ast::expr::Expr,
    type_map: &nsl_semantic::checker::TypeMap,
    interner: &nsl_lexer::Interner,
    analyzer: &mut LivenessAnalyzer,
    name_to_id: &HashMap<String, TensorAllocId>,
) {
    use nsl_ast::expr::ExprKind;

    match &expr.kind {
        ExprKind::Ident(sym) => {
            if let Some(name) = interner.resolve(sym.0) {
                if name_to_id.contains_key(name) {
                    analyzer.record_use(name);
                }
            }
        }

        ExprKind::Call { callee, args, .. } => {
            walk_expr_uses(callee, type_map, interner, analyzer, name_to_id);
            for arg in args {
                walk_expr_uses(&arg.value, type_map, interner, analyzer, name_to_id);
            }
        }

        ExprKind::BinaryOp { left, right, .. } => {
            walk_expr_uses(left, type_map, interner, analyzer, name_to_id);
            walk_expr_uses(right, type_map, interner, analyzer, name_to_id);
        }

        ExprKind::UnaryOp { operand, .. } => {
            walk_expr_uses(operand, type_map, interner, analyzer, name_to_id);
        }

        ExprKind::MemberAccess { object, .. } => {
            walk_expr_uses(object, type_map, interner, analyzer, name_to_id);
        }

        ExprKind::Subscript { object, .. } => {
            walk_expr_uses(object, type_map, interner, analyzer, name_to_id);
        }

        ExprKind::ListLiteral(items) | ExprKind::TupleLiteral(items) => {
            for item in items {
                walk_expr_uses(item, type_map, interner, analyzer, name_to_id);
            }
        }

        ExprKind::Pipe { left, right, .. } => {
            walk_expr_uses(left, type_map, interner, analyzer, name_to_id);
            walk_expr_uses(right, type_map, interner, analyzer, name_to_id);
        }

        ExprKind::BlockExpr(block) => {
            let mut local_name_to_id = name_to_id.clone();
            walk_stmts(
                &block.stmts,
                type_map,
                interner,
                analyzer,
                &mut local_name_to_id,
            );
        }

        ExprKind::IfExpr {
            condition,
            then_expr,
            else_expr,
            ..
        } => {
            walk_expr_uses(condition, type_map, interner, analyzer, name_to_id);
            walk_expr_uses(then_expr, type_map, interner, analyzer, name_to_id);
            walk_expr_uses(else_expr, type_map, interner, analyzer, name_to_id);
        }

        _ => {} // Literals, etc.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod compute_tensor_bytes_tests {
    use super::*;
    use nsl_ast::Symbol;
    use nsl_semantic::types::{DType, Dim, Shape};

    fn make_sym(n: u32) -> Symbol {
        use string_interner::Symbol as SI;
        Symbol(SI::try_from_usize(n as usize).unwrap())
    }

    /// Bug: early-return on the first `Bounded` dim discards all subsequent
    /// `Concrete` dims from the product. For a shape like
    /// [Bounded{max=128}, Concrete(512), Concrete(768)] the slab would be
    /// allocated as 128*512*4 bytes instead of the correct 128*512*768*4,
    /// causing a 768x underallocation and a guaranteed buffer overrun.
    #[test]
    fn bounded_then_concrete_includes_all_dims() {
        let shape = Shape {
            dims: vec![
                Dim::Bounded { name: make_sym(0), upper_bound: 128 },
                Dim::Concrete(512),
                Dim::Concrete(768),
            ],
        };
        let (bytes, kind) = compute_tensor_bytes(&shape, &DType::F32)
            .expect("plannable shape must return Some");
        let expected = 128u64 * 512 * 768 * 4;
        assert_eq!(
            bytes, expected,
            "Bounded+Concrete+Concrete must include ALL dims; got {} want {}",
            bytes, expected
        );
        assert!(
            matches!(kind, SizeKind::Bounded { upper, .. } if upper == expected),
            "SizeKind must be Bounded with upper == {expected}"
        );
    }

    /// Concrete-then-Bounded-then-Concrete: same invariant from the other direction.
    #[test]
    fn concrete_bounded_concrete_includes_all_dims() {
        let shape = Shape {
            dims: vec![
                Dim::Concrete(4),
                Dim::Bounded { name: make_sym(1), upper_bound: 512 },
                Dim::Concrete(768),
            ],
        };
        let (bytes, kind) = compute_tensor_bytes(&shape, &DType::F32)
            .expect("plannable shape must return Some");
        let expected = 4u64 * 512 * 768 * 4;
        assert_eq!(bytes, expected, "got {bytes} want {expected}");
        assert!(matches!(kind, SizeKind::Bounded { .. }));
    }

    /// Purely concrete shape must remain Static.
    #[test]
    fn all_concrete_is_static() {
        let shape = Shape {
            dims: vec![Dim::Concrete(64), Dim::Concrete(128)],
        };
        let (bytes, kind) = compute_tensor_bytes(&shape, &DType::F32).unwrap();
        assert_eq!(bytes, 64 * 128 * 4);
        assert!(matches!(kind, SizeKind::Static(_)));
    }

    /// Shape with Symbolic dim must return None.
    #[test]
    fn symbolic_dim_returns_none() {
        let shape = Shape {
            dims: vec![Dim::Concrete(64), Dim::Symbolic(make_sym(2))],
        };
        assert!(compute_tensor_bytes(&shape, &DType::F32).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Task 1: TensorAlloc + SizeKind ---

    #[test]
    fn test_plan_size_static() {
        let a = TensorAlloc {
            id: 0,
            name: "x".into(),
            size_bytes: 1024,
            birth: 0,
            death: 5,
            source_loc: "test:1".into(),
            size_kind: SizeKind::Static(1024),
            saved_for_backward: false,
        };
        assert_eq!(a.plan_size(), Some(1024));
        assert!(a.is_plannable());
    }

    #[test]
    fn test_plan_size_bounded() {
        let a = TensorAlloc {
            id: 0,
            name: "x".into(),
            size_bytes: 4096,
            birth: 0,
            death: 5,
            source_loc: "test:1".into(),
            size_kind: SizeKind::Bounded {
                upper: 4096,
                symbolic_expr: "B*1024".into(),
            },
            saved_for_backward: false,
        };
        assert_eq!(a.plan_size(), Some(4096));
        assert!(a.is_plannable());
    }

    #[test]
    fn test_plan_size_dynamic() {
        let a = TensorAlloc {
            id: 0,
            name: "x".into(),
            size_bytes: 0,
            birth: 0,
            death: 5,
            source_loc: "test:1".into(),
            size_kind: SizeKind::Dynamic,
            saved_for_backward: false,
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
        let a = TensorAlloc {
            id: 0,
            name: "a".into(),
            size_bytes: 100,
            birth: 0,
            death: 5,
            source_loc: "".into(),
            size_kind: SizeKind::Static(100),
            saved_for_backward: false,
        };
        let b = TensorAlloc {
            id: 1,
            name: "b".into(),
            size_bytes: 200,
            birth: 3,
            death: 8,
            source_loc: "".into(),
            size_kind: SizeKind::Static(200),
            saved_for_backward: false,
        };
        let c = TensorAlloc {
            id: 2,
            name: "c".into(),
            size_bytes: 150,
            birth: 6,
            death: 10,
            source_loc: "".into(),
            size_kind: SizeKind::Static(150),
            saved_for_backward: false,
        };

        assert!(intervals_overlap(&a, &b));
        assert!(!intervals_overlap(&a, &c));
        assert!(intervals_overlap(&b, &c));
    }

    #[test]
    fn test_interference_graph_construction() {
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "a".into(),
                size_bytes: 100,
                birth: 0,
                death: 5,
                source_loc: "".into(),
                size_kind: SizeKind::Static(100),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "b".into(),
                size_bytes: 200,
                birth: 3,
                death: 8,
                source_loc: "".into(),
                size_kind: SizeKind::Static(200),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 2,
                name: "c".into(),
                size_bytes: 150,
                birth: 6,
                death: 10,
                source_loc: "".into(),
                size_kind: SizeKind::Static(150),
                saved_for_backward: false,
            },
        ];
        let graph = InterferenceGraph::build(&allocs);

        assert!(graph.interferes(0, 1));
        assert!(!graph.interferes(0, 2));
        assert!(graph.interferes(1, 2));
    }

    #[test]
    fn test_interference_same_point_no_overlap() {
        let a = TensorAlloc {
            id: 0,
            name: "a".into(),
            size_bytes: 100,
            birth: 0,
            death: 3,
            source_loc: "".into(),
            size_kind: SizeKind::Static(100),
            saved_for_backward: false,
        };
        let b = TensorAlloc {
            id: 1,
            name: "b".into(),
            size_bytes: 200,
            birth: 3,
            death: 5,
            source_loc: "".into(),
            size_kind: SizeKind::Static(200),
            saved_for_backward: false,
        };
        assert!(!intervals_overlap(&a, &b));
    }

    #[test]
    fn test_interference_graph_skips_dynamic() {
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "static_a".into(),
                size_bytes: 100,
                birth: 0,
                death: 10,
                source_loc: "".into(),
                size_kind: SizeKind::Static(100),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "dynamic_b".into(),
                size_bytes: 0,
                birth: 0,
                death: 10,
                source_loc: "".into(),
                size_kind: SizeKind::Dynamic,
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 2,
                name: "static_c".into(),
                size_bytes: 200,
                birth: 0,
                death: 10,
                source_loc: "".into(),
                size_kind: SizeKind::Static(200),
                saved_for_backward: false,
            },
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
            TensorAlloc {
                id: 0,
                name: "a".into(),
                size_bytes: 1024,
                birth: 0,
                death: 3,
                source_loc: "".into(),
                size_kind: SizeKind::Static(1024),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "b".into(),
                size_bytes: 2048,
                birth: 1,
                death: 5,
                source_loc: "".into(),
                size_kind: SizeKind::Static(2048),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 2,
                name: "c".into(),
                size_bytes: 512,
                birth: 4,
                death: 7,
                source_loc: "".into(),
                size_kind: SizeKind::Static(512),
                saved_for_backward: false,
            },
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
            TensorAlloc {
                id: 0,
                name: "static_t".into(),
                size_bytes: 1024,
                birth: 0,
                death: 5,
                source_loc: "".into(),
                size_kind: SizeKind::Static(1024),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "dynamic_t".into(),
                size_bytes: 0,
                birth: 2,
                death: 4,
                source_loc: "".into(),
                size_kind: SizeKind::Dynamic,
                saved_for_backward: false,
            },
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
            TensorAlloc {
                id: 0,
                name: "a".into(),
                size_bytes: 100,
                birth: 0,
                death: 10,
                source_loc: "".into(),
                size_kind: SizeKind::Static(100),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "b".into(),
                size_bytes: 200,
                birth: 0,
                death: 10,
                source_loc: "".into(),
                size_kind: SizeKind::Static(200),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 2,
                name: "c".into(),
                size_bytes: 300,
                birth: 0,
                death: 10,
                source_loc: "".into(),
                size_kind: SizeKind::Static(300),
                saved_for_backward: false,
            },
        ];
        let graph = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &graph);

        assert_eq!(plan.slots.len(), 3);
        assert!(plan.total_bytes >= 600);
    }

    #[test]
    fn test_plan_slab_sequential_full_reuse() {
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "a".into(),
                size_bytes: 1024,
                birth: 0,
                death: 2,
                source_loc: "".into(),
                size_kind: SizeKind::Static(1024),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "b".into(),
                size_bytes: 512,
                birth: 3,
                death: 5,
                source_loc: "".into(),
                size_kind: SizeKind::Static(512),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 2,
                name: "c".into(),
                size_bytes: 768,
                birth: 6,
                death: 8,
                source_loc: "".into(),
                size_kind: SizeKind::Static(768),
                saved_for_backward: false,
            },
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
            TensorAlloc {
                id: 0,
                name: "weight_q".into(),
                size_bytes: 4096,
                birth: 0,
                death: 10,
                source_loc: "model.nsl:5".into(),
                size_kind: SizeKind::Static(4096),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "hidden".into(),
                size_bytes: 2048,
                birth: 2,
                death: 6,
                source_loc: "model.nsl:8".into(),
                size_kind: SizeKind::Static(2048),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 2,
                name: "output".into(),
                size_bytes: 1024,
                birth: 7,
                death: 10,
                source_loc: "model.nsl:12".into(),
                size_kind: SizeKind::Static(1024),
                saved_for_backward: false,
            },
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
        assert_eq!(
            parse_vram_budget("1.5GB"),
            Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64)
        );
        assert_eq!(parse_vram_budget("0GB"), Some(0));
        assert_eq!(parse_vram_budget("garbage"), None);
        assert_eq!(parse_vram_budget(""), None);
        assert_eq!(parse_vram_budget("GB"), None);
    }

    #[test]
    fn test_savings_fraction_zero_allocs() {
        let plan = SlabPlan {
            slots: vec![],
            total_bytes: 0,
            assignments: HashMap::new(),
            naive_total: 0,
            padding_bytes: 0,
        };
        assert_eq!(plan.savings_fraction(), 0.0);
    }

    #[test]
    fn test_check_vram_budget_passes() {
        let plan = SlabPlan {
            slots: vec![],
            total_bytes: 1024,
            assignments: HashMap::new(),
            naive_total: 2048,
            padding_bytes: 0,
        };
        assert!(check_vram_budget(&plan, 2048).is_none());
    }

    #[test]
    fn test_check_vram_budget_fails() {
        let plan = SlabPlan {
            slots: vec![],
            total_bytes: 4096,
            assignments: HashMap::new(),
            naive_total: 4096,
            padding_bytes: 0,
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
                id: 0,
                name: "big".into(),
                size_bytes: 1000,
                birth: 0,
                death: 5,
                source_loc: String::new(),
                size_kind: SizeKind::Static(1000),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "medium".into(),
                size_bytes: 600,
                birth: 0,
                death: 5,
                source_loc: String::new(),
                size_kind: SizeKind::Static(600),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 2,
                name: "fits_medium".into(),
                size_bytes: 580,
                birth: 6,
                death: 10,
                source_loc: String::new(),
                size_kind: SizeKind::Static(580),
                saved_for_backward: false,
            },
        ];

        let ig = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &ig);

        // Tensor 2 should share a slot with tensor 1 (600, waste=20)
        // not tensor 0 (1000, waste=420)
        let (slot_t1, _) = plan.assignments[&1];
        let (slot_t2, _) = plan.assignments[&2];
        assert_eq!(
            slot_t2, slot_t1,
            "BFD should assign 580-byte tensor to 600-byte slot, not 1000-byte slot"
        );

        // Only 2 slots needed (1000 + 600)
        assert_eq!(plan.slots.len(), 2);
    }

    #[test]
    fn test_bfd_reuses_slot_for_non_overlapping() {
        // Two tensors with non-overlapping lifetimes should share one slot
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "a".into(),
                size_bytes: 1024,
                birth: 0,
                death: 5,
                source_loc: String::new(),
                size_kind: SizeKind::Static(1024),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "b".into(),
                size_bytes: 512,
                birth: 6,
                death: 10,
                source_loc: String::new(),
                size_kind: SizeKind::Static(512),
                saved_for_backward: false,
            },
        ];

        let ig = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &ig);

        assert_eq!(
            plan.slots.len(),
            1,
            "non-overlapping tensors should share 1 slot"
        );
        assert_eq!(plan.total_bytes, 1024, "slot size = max(1024, 512)");
    }

    #[test]
    fn test_fragmentation_zero_for_aligned_sizes() {
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "aligned".into(),
                size_bytes: 256,
                birth: 0,
                death: 5,
                source_loc: String::new(),
                size_kind: SizeKind::Static(256),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "aligned2".into(),
                size_bytes: 512,
                birth: 0,
                death: 5,
                source_loc: String::new(),
                size_kind: SizeKind::Static(512),
                saved_for_backward: false,
            },
        ];

        let ig = InterferenceGraph::build(&allocs);
        let plan = plan_slab(&allocs, &ig);

        assert_eq!(
            plan.padding_bytes, 0,
            "aligned sizes should have zero padding"
        );
        assert!((plan.fragmentation_ratio() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_fragmentation_nonzero_for_unaligned_sizes() {
        let allocs = vec![TensorAlloc {
            id: 0,
            name: "unaligned".into(),
            size_bytes: 100,
            birth: 0,
            death: 5,
            source_loc: String::new(),
            size_kind: SizeKind::Static(100),
            saved_for_backward: false,
        }];

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
                id: 0,
                name: "input".into(),
                size_bytes: 4096,
                birth: 0,
                death: 100,
                source_loc: String::new(),
                size_kind: SizeKind::Static(4096),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "relu_out".into(),
                size_bytes: 4096,
                birth: 5,
                death: 95,
                source_loc: String::new(),
                size_kind: SizeKind::Static(4096),
                saved_for_backward: false,
            },
        ];

        let op_infos = vec![TensorOpInfo {
            alloc_id: 1,
            op_name: "relu".into(),
            input_alloc_ids: vec![0],
            dtype_bytes: 4,
        }];

        let (modified, recomp_points) =
            rematerialize(&allocs, &op_infos, crate::cost_model::REMAT_SCORE_THRESHOLD);

        // relu_out should be a remat candidate (cheap, large range)
        assert!(
            !recomp_points.is_empty(),
            "should have at least one recomp point"
        );
        assert_eq!(recomp_points[0].original_alloc_id, 1);

        // Modified alloc should have shortened death
        assert!(
            modified[1].death < allocs[1].death,
            "death should be shortened: was {}, now {}",
            allocs[1].death,
            modified[1].death
        );
    }

    #[test]
    fn test_remat_rejects_matmul() {
        // matmul is expensive — should NOT be rematerialized
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "a".into(),
                size_bytes: 4096,
                birth: 0,
                death: 100,
                source_loc: String::new(),
                size_kind: SizeKind::Static(4096),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "mm_out".into(),
                size_bytes: 4096,
                birth: 5,
                death: 95,
                source_loc: String::new(),
                size_kind: SizeKind::Static(4096),
                saved_for_backward: false,
            },
        ];

        let op_infos = vec![TensorOpInfo {
            alloc_id: 1,
            op_name: "matmul".into(),
            input_alloc_ids: vec![0],
            dtype_bytes: 4,
        }];

        let (_, recomp_points) =
            rematerialize(&allocs, &op_infos, crate::cost_model::REMAT_SCORE_THRESHOLD);
        assert!(
            recomp_points.is_empty(),
            "matmul should NOT be rematerialized"
        );
    }

    #[test]
    fn test_remat_free_ops_always_accepted() {
        // reshape is free — always accepted regardless of score
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "src".into(),
                size_bytes: 100,
                birth: 0,
                death: 100,
                source_loc: String::new(),
                size_kind: SizeKind::Static(100),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "reshaped".into(),
                size_bytes: 100,
                birth: 5,
                death: 95,
                source_loc: String::new(),
                size_kind: SizeKind::Static(100),
                saved_for_backward: false,
            },
        ];

        let op_infos = vec![TensorOpInfo {
            alloc_id: 1,
            op_name: "reshape".into(),
            input_alloc_ids: vec![0],
            dtype_bytes: 4,
        }];

        let (_, recomp_points) = rematerialize(&allocs, &op_infos, f64::MAX); // very high threshold
        assert!(
            !recomp_points.is_empty(),
            "free ops should always be accepted"
        );
    }

    #[test]
    fn test_remat_inputs_must_be_live() {
        // Input dies before recompute point → cannot rematerialize
        let allocs = vec![
            TensorAlloc {
                id: 0,
                name: "input".into(),
                size_bytes: 4096,
                birth: 0,
                death: 10,
                source_loc: String::new(),
                size_kind: SizeKind::Static(4096),
                saved_for_backward: false,
            },
            TensorAlloc {
                id: 1,
                name: "relu_out".into(),
                size_bytes: 4096,
                birth: 5,
                death: 95,
                source_loc: String::new(),
                size_kind: SizeKind::Static(4096),
                saved_for_backward: false,
            },
        ];

        let op_infos = vec![TensorOpInfo {
            alloc_id: 1,
            op_name: "relu".into(),
            input_alloc_ids: vec![0],
            dtype_bytes: 4,
        }];

        let (_, recomp_points) =
            rematerialize(&allocs, &op_infos, crate::cost_model::REMAT_SCORE_THRESHOLD);
        // Input dies at 10 but recompute point is at 94 — cannot remat
        assert!(recomp_points.is_empty(), "cannot remat if inputs are dead");
    }
}

#[cfg(test)]
mod wrga_hints_unit_tests {
    use super::*;
    use crate::wrga_memory::SlotAssignment;

    fn mk_assignment(var: u32, slot: u32, size: u64, birth: u32, death: u32) -> SlotAssignment {
        SlotAssignment {
            var,
            slot,
            size_bytes: size,
            birth,
            death,
        }
    }

    #[test]
    fn try_merge_pair_refuses_size_mismatch() {
        let assignments = vec![
            mk_assignment(1, 0, 64, 0, 5),
            mk_assignment(2, 0, 128, 10, 15), // disjoint liveness, but size differs
        ];
        assert!(
            !try_merge_pair(&assignments, 0, 1),
            "different sizes must refuse"
        );
    }

    #[test]
    fn try_merge_pair_refuses_overlapping_liveness() {
        let assignments = vec![
            mk_assignment(1, 0, 64, 0, 10),
            mk_assignment(2, 0, 64, 5, 15), // overlap: 5..=10
        ];
        assert!(
            !try_merge_pair(&assignments, 0, 1),
            "overlapping liveness must refuse"
        );
    }

    #[test]
    fn try_merge_pair_accepts_disjoint_equal_size() {
        let assignments = vec![
            mk_assignment(1, 0, 64, 0, 5),
            mk_assignment(2, 0, 64, 10, 15),
        ];
        assert!(
            try_merge_pair(&assignments, 0, 1),
            "equal size + disjoint liveness must accept"
        );
    }

    #[test]
    fn try_merge_pair_refuses_zero_size() {
        let assignments = vec![
            mk_assignment(1, 0, 0, 0, 5),
            mk_assignment(2, 0, 0, 10, 15),
        ];
        assert!(
            !try_merge_pair(&assignments, 0, 1),
            "zero size must refuse (guard against unallocated)"
        );
    }

    // Half-open interval invariant: intervals [0, 5) and [5, 10) are adjacent
    // but NOT overlapping — the slab planner's intervals_overlap uses strict < for
    // this reason.  try_merge_pair must match that semantics (death <= birth means
    // disjoint) so that WRGA hints don't silently fail to coalesce adjacent-lifetime
    // activations that the slab planner would correctly place in the same slot.
    #[test]
    fn try_merge_pair_accepts_adjacent_intervals_death_eq_birth() {
        let assignments = vec![
            mk_assignment(1, 0, 64, 0, 5),
            mk_assignment(2, 0, 64, 5, 10), // birth == death of previous → adjacent, not overlapping
        ];
        assert!(
            try_merge_pair(&assignments, 0, 1),
            "adjacent intervals (death == birth) must be accepted: \
             [0,5) and [5,10) do not overlap under half-open semantics"
        );
    }
}

#[cfg(test)]
mod consume_hints_tests {
    use super::*;
    use crate::wrga_memory::{MemoryPlan, MemoryPlanStats, SlotAssignment};

    fn mk_plan_with_shared_slot() -> crate::wrga::WrgaPlan {
        let mut plan = crate::wrga::WrgaPlan::test_dummy();
        plan.memory = MemoryPlan {
            assignments: vec![
                SlotAssignment { var: 1, slot: 0, size_bytes: 64, birth: 0, death: 5 },
                SlotAssignment { var: 2, slot: 0, size_bytes: 64, birth: 10, death: 15 },
            ],
            stats: MemoryPlanStats::default(),
        };
        plan
    }

    #[test]
    fn consume_hints_merges_disjoint_equal_size_activations() {
        let mut a = LivenessAnalyzer::new();
        a.record_activation_alloc(1, 64);
        a.record_activation_alloc(2, 64);
        let plan = mk_plan_with_shared_slot();
        let post = a.consume_hints(&plan);
        assert_eq!(post, 1, "disjoint+equal-size should merge into one slot");
    }

    #[test]
    fn consume_hints_refuses_overlapping_liveness() {
        let mut a = LivenessAnalyzer::new();
        a.record_activation_alloc(1, 64);
        a.record_activation_alloc(2, 64);
        let mut plan = mk_plan_with_shared_slot();
        plan.memory.assignments[1].birth = 3; // overlaps var 1's 0..=5
        let post = a.consume_hints(&plan);
        assert_eq!(post, 2, "overlapping liveness must refuse to merge");
    }

    #[test]
    fn consume_hints_refuses_size_mismatch() {
        let mut a = LivenessAnalyzer::new();
        a.record_activation_alloc(1, 64);
        a.record_activation_alloc(2, 128);
        let plan = mk_plan_with_shared_slot();
        let post = a.consume_hints(&plan);
        assert_eq!(post, 2, "size mismatch must refuse to merge");
    }

    // Half-open interval invariant: consume_hints must merge adjacent-lifetime
    // activations ([0,5) and [5,10)) because the slab planner treats them as
    // non-overlapping.  The hint pass must agree or it silently degrades memory
    // reuse for the common sequential-backward-step pattern.
    #[test]
    fn consume_hints_merges_adjacent_lifetime_activations() {
        let mut a = LivenessAnalyzer::new();
        a.record_activation_alloc(1, 64);
        a.record_activation_alloc(2, 64);

        let mut plan = crate::wrga::WrgaPlan::test_dummy();
        plan.memory = crate::wrga_memory::MemoryPlan {
            assignments: vec![
                crate::wrga_memory::SlotAssignment { var: 1, slot: 0, size_bytes: 64, birth: 0, death: 5 },
                // birth == death of previous slot → adjacent, not overlapping (half-open)
                crate::wrga_memory::SlotAssignment { var: 2, slot: 0, size_bytes: 64, birth: 5, death: 10 },
            ],
            stats: crate::wrga_memory::MemoryPlanStats::default(),
        };

        let post = a.consume_hints(&plan);
        assert_eq!(
            post, 1,
            "adjacent-lifetime activations ([0,5) and [5,10)) must merge: \
             they are not overlapping under half-open interval semantics"
        );
    }
}
