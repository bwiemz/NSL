# M36: Compile-Time Memory Planning — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compile-time memory planning pass that statically analyzes tensor lifetimes, builds an interference graph, and assigns tensor allocations to offsets within a single pre-allocated memory slab — eliminating per-tensor allocation overhead and enabling compile-time VRAM budget guarantees.

**Architecture:** Three subsystems built bottom-up: (1) A codegen analysis pass (`memory_planner.rs`) that walks AST statements to extract tensor allocation sites, compute liveness intervals, build an interference graph, and assign slab offsets via first-fit-decreasing. (2) A minimal slab runtime (`slab.rs`) with `nsl_slab_alloc`/`nsl_slab_free`/`nsl_slab_offset` FFI for single-allocation memory pools. (3) Codegen integration that emits slab-backed tensor creation when a memory plan exists, plus CLI flags for `--vram-budget` and `nsl check --memory`.

**Tech Stack:** Rust (codegen analysis + runtime), Cranelift (codegen), clap (CLI)

**Spec:** `docs/superpowers/specs/2026-03-15-m36-memory-planning-design.md`

---

## Important: Scope of This Plan

**This plan builds the analysis infrastructure and runtime foundation for M36.** It delivers:
- The full memory planning algorithm (liveness, interference graph, slab assignment) as a standalone, tested library
- The slab runtime FFI (alloc/free/offset)
- CLI flags for `--vram-budget` and `--memory-report`
- Memory report formatting and VRAM budget checking

**Actual codegen slab emission** (replacing `checked_alloc` calls with `slab_ptr + offset` in `expr.rs`, and AST-walking integration in `compile_main()`) is deferred to **M36b**. This plan produces all the building blocks that M36b will wire together.

**Known dependency:** `CompileOptions` is not currently wired through `compile_entry()` (see `main.rs:190-195` TODO). The `--vram-budget` and `--memory-report` CLI flags are parsed and stored but will not reach the compiler until that wiring is completed (same status as existing `--fusion-report` and `--no-autotune` flags).

---

## Scope Note

The spec covers 8 deliverables. This plan orders them so each produces independently testable code:

1. **Tasks 1-4**: Core data structures + liveness analysis + interference graph + slab assignment (pure Rust, zero codegen dependency, fully unit-testable)
2. **Tasks 5-6**: Slab runtime FFI (nsl-runtime, independently testable)
3. **Tasks 7-8**: Codegen integration (compiler fields, memory report)
4. **Tasks 9-10**: CLI flags + E2E tests
5. **Task 11**: Full verification + clippy

**Deferred to M36b:** Automatic `@checkpoint` insertion (`--vram-budget-auto-checkpoint`), training-mode partial planning (tape-excluded activations), cross-function interprocedural memory planning, actual `cuMemAlloc` replacement on GPU path (CPU slab first), codegen slab emission in `expr.rs`/`stmt.rs`, AST-walking `LivenessAnalyzer` integration in `compile_main()`, branch/loop conservative lifetime extension, bounded dimension size computation from `type_map` shapes, `nsl check --memory` subcommand, program-exit slab cleanup (`nsl_slab_free` call site).

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/memory_planner.rs` | Core: TensorAlloc, LivenessAnalyzer, InterferenceGraph, SlabPlan, plan_slab(), memory report formatting | 500 |
| `crates/nsl-runtime/src/slab.rs` | Slab FFI: nsl_slab_alloc, nsl_slab_free, nsl_slab_offset | 100 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod memory_planner;`, extend `CompileOptions` with `vram_budget: Option<u64>`, `memory_report: bool` |
| `crates/nsl-codegen/src/builtins.rs` | Register 3 slab FFI functions |
| `crates/nsl-codegen/src/compiler.rs` | Add `slab_plan: Option<memory_planner::SlabPlan>` field; invoke memory planner in compilation flow |
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod slab;` |
| `crates/nsl-cli/src/main.rs` | Add `--vram-budget` flag to `build`/`run` subcommands |
| `crates/nsl-cli/tests/e2e.rs` | Add M36 E2E tests |

---

## Phase 1: Core Data Structures + Analysis

### Task 1: TensorAlloc + SizeKind Types

**Files:**
- Create: `crates/nsl-codegen/src/memory_planner.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `memory_planner.rs` with core types and basic tests**

```rust
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_size_static() {
        let a = TensorAlloc {
            id: 0, name: "x".into(), size_bytes: 1024,
            birth: 0, death: 5, source_loc: "test:1".into(),
            size_kind: SizeKind::Static(1024),
        };
        assert_eq!(a.plan_size(), Some(1024));
        assert!(a.is_plannable());
    }

    #[test]
    fn test_plan_size_bounded() {
        let a = TensorAlloc {
            id: 0, name: "x".into(), size_bytes: 4096,
            birth: 0, death: 5, source_loc: "test:1".into(),
            size_kind: SizeKind::Bounded { upper: 4096, symbolic_expr: "B*1024".into() },
        };
        assert_eq!(a.plan_size(), Some(4096));
        assert!(a.is_plannable());
    }

    #[test]
    fn test_plan_size_dynamic() {
        let a = TensorAlloc {
            id: 0, name: "x".into(), size_bytes: 0,
            birth: 0, death: 5, source_loc: "test:1".into(),
            size_kind: SizeKind::Dynamic,
        };
        assert_eq!(a.plan_size(), None);
        assert!(!a.is_plannable());
    }
}
```

- [ ] **Step 2: Add `pub mod memory_planner;` to codegen lib.rs**

After the existing `pub mod fp8;` line:

```rust
pub mod memory_planner;
```

- [ ] **Step 3: Verify compilation, run tests**

```bash
cargo test -p nsl-codegen memory_planner -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m36): add TensorAlloc and SizeKind core types for memory planning"
```

---

### Task 2: Liveness Analyzer

**Files:**
- Modify: `crates/nsl-codegen/src/memory_planner.rs`

- [ ] **Step 1: Write liveness analysis tests**

Add to the `tests` module:

```rust
#[test]
fn test_liveness_linear() {
    // Three sequential tensors: a lives [0,2], b lives [1,3], c lives [2,4]
    let mut analyzer = LivenessAnalyzer::new();
    analyzer.record_alloc("a", 1024, SizeKind::Static(1024), "line:1");
    analyzer.advance();
    analyzer.record_alloc("b", 2048, SizeKind::Static(2048), "line:2");
    analyzer.record_use("a");
    analyzer.advance();
    analyzer.record_alloc("c", 512, SizeKind::Static(512), "line:3");
    analyzer.record_use("a");  // last use of a
    analyzer.advance();
    analyzer.record_use("b");  // last use of b
    analyzer.advance();
    analyzer.record_use("c");  // last use of c

    let allocs = analyzer.finish();
    assert_eq!(allocs.len(), 3);
    assert_eq!(allocs[0].name, "a");
    assert_eq!(allocs[0].birth, 0);
    assert_eq!(allocs[0].death, 2);
    assert_eq!(allocs[1].name, "b");
    assert_eq!(allocs[1].birth, 1);
    assert_eq!(allocs[1].death, 3);
    assert_eq!(allocs[2].name, "c");
    assert_eq!(allocs[2].birth, 2);
    assert_eq!(allocs[2].death, 4);
}

#[test]
fn test_liveness_reuse_name() {
    // Variable reassigned: old alloc dies, new alloc starts
    let mut analyzer = LivenessAnalyzer::new();
    analyzer.record_alloc("x", 100, SizeKind::Static(100), "line:1");
    analyzer.advance();
    analyzer.record_use("x");
    analyzer.advance();
    analyzer.record_alloc("x", 200, SizeKind::Static(200), "line:3"); // reassign
    analyzer.advance();
    analyzer.record_use("x");

    let allocs = analyzer.finish();
    assert_eq!(allocs.len(), 2);
    assert_eq!(allocs[0].size_bytes, 100);
    assert_eq!(allocs[0].death, 1); // dies at last use before reassign
    assert_eq!(allocs[1].size_bytes, 200);
    assert_eq!(allocs[1].birth, 2);
    assert_eq!(allocs[1].death, 3);
}

#[test]
fn test_liveness_unused_tensor() {
    // Tensor allocated but never used — birth == death
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
    // Variable reassigned immediately — old alloc has birth == death
    let mut analyzer = LivenessAnalyzer::new();
    analyzer.record_alloc("x", 100, SizeKind::Static(100), "line:1");
    analyzer.advance();
    analyzer.record_alloc("x", 200, SizeKind::Static(200), "line:2"); // immediate reassign
    analyzer.advance();
    analyzer.record_use("x");

    let allocs = analyzer.finish();
    assert_eq!(allocs.len(), 2);
    assert_eq!(allocs[0].death, 0); // never used, birth == death
    assert_eq!(allocs[1].birth, 1);
    assert_eq!(allocs[1].death, 2);
}
```

- [ ] **Step 2: Implement LivenessAnalyzer**

Add above the `tests` module:

```rust
// ---------------------------------------------------------------------------
// Liveness analysis
// ---------------------------------------------------------------------------

/// Walks program statements to record tensor allocation sites and lifetimes.
pub struct LivenessAnalyzer {
    allocs: Vec<TensorAlloc>,
    /// Maps variable name → TensorAllocId for currently-live tensors.
    live_set: HashMap<String, TensorAllocId>,
    current_pp: ProgramPoint,
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
        let id = self.allocs.len() as TensorAllocId;
        self.allocs.push(TensorAlloc {
            id,
            name: name.to_string(),
            size_bytes,
            birth: self.current_pp,
            death: self.current_pp, // updated on each use
            source_loc: loc.to_string(),
            size_kind,
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
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p nsl-codegen memory_planner -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m36): add LivenessAnalyzer for tensor lifetime tracking"
```

---

### Task 3: Interference Graph

**Files:**
- Modify: `crates/nsl-codegen/src/memory_planner.rs`

- [ ] **Step 1: Write interference graph tests**

```rust
#[test]
fn test_intervals_overlap() {
    let a = TensorAlloc {
        id: 0, name: "a".into(), size_bytes: 100,
        birth: 0, death: 5, source_loc: "".into(),
        size_kind: SizeKind::Static(100),
    };
    let b = TensorAlloc {
        id: 1, name: "b".into(), size_bytes: 200,
        birth: 3, death: 8, source_loc: "".into(),
        size_kind: SizeKind::Static(200),
    };
    let c = TensorAlloc {
        id: 2, name: "c".into(), size_bytes: 150,
        birth: 6, death: 10, source_loc: "".into(),
        size_kind: SizeKind::Static(150),
    };

    // a [0,5] overlaps b [3,8]
    assert!(intervals_overlap(&a, &b));
    // a [0,5] does NOT overlap c [6,10]
    assert!(!intervals_overlap(&a, &c));
    // b [3,8] overlaps c [6,10]
    assert!(intervals_overlap(&b, &c));
}

#[test]
fn test_interference_graph_construction() {
    let allocs = vec![
        TensorAlloc { id: 0, name: "a".into(), size_bytes: 100, birth: 0, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(100) },
        TensorAlloc { id: 1, name: "b".into(), size_bytes: 200, birth: 3, death: 8, source_loc: "".into(), size_kind: SizeKind::Static(200) },
        TensorAlloc { id: 2, name: "c".into(), size_bytes: 150, birth: 6, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(150) },
    ];
    let graph = InterferenceGraph::build(&allocs);

    assert!(graph.interferes(0, 1)); // a ↔ b
    assert!(!graph.interferes(0, 2)); // a ↔ c: no overlap
    assert!(graph.interferes(1, 2)); // b ↔ c
}

#[test]
fn test_interference_same_point_no_overlap() {
    // birth == death of other → no overlap (half-open intervals)
    let a = TensorAlloc { id: 0, name: "a".into(), size_bytes: 100, birth: 0, death: 3, source_loc: "".into(), size_kind: SizeKind::Static(100) };
    let b = TensorAlloc { id: 1, name: "b".into(), size_bytes: 200, birth: 3, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(200) };
    // a dies at 3, b born at 3 → no overlap (a is dead when b starts)
    assert!(!intervals_overlap(&a, &b));
}

#[test]
fn test_interference_graph_skips_dynamic() {
    let allocs = vec![
        TensorAlloc { id: 0, name: "static_a".into(), size_bytes: 100, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(100) },
        TensorAlloc { id: 1, name: "dynamic_b".into(), size_bytes: 0, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Dynamic },
        TensorAlloc { id: 2, name: "static_c".into(), size_bytes: 200, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(200) },
    ];
    let graph = InterferenceGraph::build(&allocs);

    // static_a and static_c interfere (both [0,10])
    assert!(graph.interferes(0, 2));
    // dynamic_b should have NO interference edges
    assert!(!graph.interferes(0, 1));
    assert!(!graph.interferes(1, 2));
}
```

- [ ] **Step 2: Implement InterferenceGraph**

```rust
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
    num_tensors: usize,
}

impl InterferenceGraph {
    /// Build interference graph from a set of tensor allocations.
    pub fn build(allocs: &[TensorAlloc]) -> Self {
        let n = allocs.len();
        let mut adj = vec![HashSet::new(); n];

        for i in 0..n {
            if !allocs[i].is_plannable() { continue; }
            for j in (i + 1)..n {
                if !allocs[j].is_plannable() { continue; }
                if intervals_overlap(&allocs[i], &allocs[j]) {
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
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p nsl-codegen memory_planner -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m36): add interference graph with interval overlap detection"
```

---

### Task 4: Slab Assignment (First-Fit Decreasing)

**Files:**
- Modify: `crates/nsl-codegen/src/memory_planner.rs`

- [ ] **Step 1: Write slab assignment tests**

```rust
#[test]
fn test_plan_slab_reuse() {
    // a [0,3] 1024 bytes, b [1,5] 2048 bytes, c [4,7] 512 bytes
    // a and c don't overlap → should share a slot (1024 byte slot)
    // b overlaps with both → own slot (2048 byte slot)
    let allocs = vec![
        TensorAlloc { id: 0, name: "a".into(), size_bytes: 1024, birth: 0, death: 3, source_loc: "".into(), size_kind: SizeKind::Static(1024) },
        TensorAlloc { id: 1, name: "b".into(), size_bytes: 2048, birth: 1, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(2048) },
        TensorAlloc { id: 2, name: "c".into(), size_bytes: 512, birth: 4, death: 7, source_loc: "".into(), size_kind: SizeKind::Static(512) },
    ];
    let graph = InterferenceGraph::build(&allocs);
    let plan = plan_slab(&allocs, &graph);

    // b (largest) gets first slot, a and c share second slot
    assert_eq!(plan.slots.len(), 2);
    // Total should be 2048 (b's slot) + 1024 (shared a/c slot) + alignment padding
    assert!(plan.total_bytes >= 2048 + 1024);
    assert!(plan.total_bytes <= 2048 + 1024 + 512); // at most 2 * 256-byte alignment pads

    // a and c should be in the same slot
    let (slot_a, _) = plan.assignments[&0];
    let (slot_c, _) = plan.assignments[&2];
    assert_eq!(slot_a, slot_c);

    // b should be in a different slot
    let (slot_b, _) = plan.assignments[&1];
    assert_ne!(slot_a, slot_b);
}

#[test]
fn test_plan_slab_skips_dynamic() {
    let allocs = vec![
        TensorAlloc { id: 0, name: "static_t".into(), size_bytes: 1024, birth: 0, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(1024) },
        TensorAlloc { id: 1, name: "dynamic_t".into(), size_bytes: 0, birth: 2, death: 4, source_loc: "".into(), size_kind: SizeKind::Dynamic },
    ];
    let graph = InterferenceGraph::build(&allocs);
    let plan = plan_slab(&allocs, &graph);

    // Only static tensor is planned
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
    // Three tensors all alive at the same time — no reuse possible
    let allocs = vec![
        TensorAlloc { id: 0, name: "a".into(), size_bytes: 100, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(100) },
        TensorAlloc { id: 1, name: "b".into(), size_bytes: 200, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(200) },
        TensorAlloc { id: 2, name: "c".into(), size_bytes: 300, birth: 0, death: 10, source_loc: "".into(), size_kind: SizeKind::Static(300) },
    ];
    let graph = InterferenceGraph::build(&allocs);
    let plan = plan_slab(&allocs, &graph);

    assert_eq!(plan.slots.len(), 3); // each needs own slot
    assert!(plan.total_bytes >= 600); // at least sum of all sizes
}

#[test]
fn test_plan_slab_sequential_full_reuse() {
    // Three tensors fully sequential — maximum reuse
    let allocs = vec![
        TensorAlloc { id: 0, name: "a".into(), size_bytes: 1024, birth: 0, death: 2, source_loc: "".into(), size_kind: SizeKind::Static(1024) },
        TensorAlloc { id: 1, name: "b".into(), size_bytes: 512, birth: 3, death: 5, source_loc: "".into(), size_kind: SizeKind::Static(512) },
        TensorAlloc { id: 2, name: "c".into(), size_bytes: 768, birth: 6, death: 8, source_loc: "".into(), size_kind: SizeKind::Static(768) },
    ];
    let graph = InterferenceGraph::build(&allocs);
    let plan = plan_slab(&allocs, &graph);

    // All should share one slot (sized to max = 1024)
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
```

- [ ] **Step 2: Implement MemorySlot, SlabPlan, plan_slab, align_up**

```rust
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
    /// TensorAllocId → (slot_id, byte_offset_in_slab).
    pub assignments: HashMap<TensorAllocId, (u32, u64)>,
    /// Naive allocation total (sum of all planned tensor sizes) for savings report.
    pub naive_total: u64,
}

impl SlabPlan {
    /// Memory savings as a fraction (0.0 to 1.0).
    pub fn savings_fraction(&self) -> f64 {
        if self.naive_total == 0 { return 0.0; }
        1.0 - (self.total_bytes as f64 / self.naive_total as f64)
    }
}

/// GPU alignment for slab offsets (256 bytes).
const SLAB_ALIGNMENT: u64 = 256;

/// Assign tensors to memory slots using first-fit-decreasing.
/// Tensors with SizeKind::Dynamic are skipped.
pub fn plan_slab(allocs: &[TensorAlloc], interference: &InterferenceGraph) -> SlabPlan {
    // Sort tensor IDs by size (largest first) for first-fit-decreasing
    let mut sorted: Vec<TensorAllocId> = allocs.iter()
        .filter(|a| a.is_plannable())
        .map(|a| a.id)
        .collect();
    sorted.sort_by(|&a, &b| allocs[b as usize].size_bytes.cmp(&allocs[a as usize].size_bytes));

    let naive_total: u64 = sorted.iter().map(|&id| allocs[id as usize].size_bytes).sum();

    let mut slots: Vec<MemorySlot> = Vec::new();
    let mut assignments = HashMap::new();

    for &tensor_id in &sorted {
        let tensor = &allocs[tensor_id as usize];
        let mut assigned = false;

        for slot in &mut slots {
            let conflicts = slot.assigned.iter().any(|&other_id| {
                interference.interferes(tensor_id, other_id)
            });
            if !conflicts {
                slot.size = slot.size.max(tensor.size_bytes);
                slot.assigned.push(tensor_id);
                assigned = true;
                break;
            }
        }

        if !assigned {
            let slot_id = slots.len() as u32;
            slots.push(MemorySlot {
                id: slot_id,
                offset: 0, // computed after all assignments
                size: tensor.size_bytes,
                assigned: vec![tensor_id],
            });
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
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p nsl-codegen memory_planner -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m36): add slab assignment via first-fit-decreasing with interference graph"
```

---

## Phase 2: Slab Runtime

### Task 5: Slab Allocator Runtime FFI

**Files:**
- Create: `crates/nsl-runtime/src/slab.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Create `slab.rs` with allocator and tests**

```rust
//! M36: Slab memory allocator for compile-time planned tensor memory.
//!
//! A slab is a single contiguous allocation. Tensors are sub-allocated as
//! pointer offsets into the slab, with `owns_data = 0` so `nsl_tensor_free`
//! does not attempt to free them individually.

use crate::memory::{checked_alloc_zeroed, checked_free};

/// Allocate a contiguous memory slab (zeroed). Returns base pointer as i64.
/// On CPU (current implementation): uses checked_alloc_zeroed.
#[no_mangle]
pub extern "C" fn nsl_slab_alloc(size_bytes: i64) -> i64 {
    if size_bytes <= 0 {
        return 0;
    }
    let ptr = checked_alloc_zeroed(size_bytes as usize);
    ptr as i64
}

/// Free a previously allocated slab.
/// NOTE: Takes size_bytes (unlike spec's 1-param signature) because checked_free
/// requires the allocation size for Layout reconstruction.
#[no_mangle]
pub extern "C" fn nsl_slab_free(slab_ptr: i64, size_bytes: i64) {
    if slab_ptr == 0 || size_bytes <= 0 {
        return;
    }
    // SAFETY: slab_ptr was returned by nsl_slab_alloc (which uses checked_alloc_zeroed
    // with align=8), and size_bytes matches the original allocation size.
    unsafe {
        checked_free(slab_ptr as *mut u8, size_bytes as usize);
    }
}

/// Compute a pointer offset into the slab. Returns slab_ptr + offset.
#[no_mangle]
pub extern "C" fn nsl_slab_offset(slab_ptr: i64, offset: i64) -> i64 {
    slab_ptr + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slab_alloc_free() {
        let slab = nsl_slab_alloc(4096);
        assert_ne!(slab, 0);
        // Write to start and end to verify memory is accessible
        unsafe {
            *(slab as *mut u8) = 42;
            *((slab as *mut u8).add(4095)) = 99;
            assert_eq!(*(slab as *mut u8), 42);
            assert_eq!(*((slab as *mut u8).add(4095)), 99);
        }
        nsl_slab_free(slab, 4096);
    }

    #[test]
    fn test_slab_alloc_zero_size() {
        assert_eq!(nsl_slab_alloc(0), 0);
        assert_eq!(nsl_slab_alloc(-1), 0);
    }

    #[test]
    fn test_slab_offset() {
        let base = 0x1000_i64;
        assert_eq!(nsl_slab_offset(base, 0), 0x1000);
        assert_eq!(nsl_slab_offset(base, 256), 0x1100);
        assert_eq!(nsl_slab_offset(base, 1024), 0x1400);
    }

    #[test]
    fn test_slab_free_null() {
        // Should not crash
        nsl_slab_free(0, 100);
        nsl_slab_free(42, 0);
        nsl_slab_free(0, 0);
    }

    #[test]
    fn test_slab_zeroed() {
        let slab = nsl_slab_alloc(1024);
        let data = unsafe { std::slice::from_raw_parts(slab as *const u8, 1024) };
        assert!(data.iter().all(|&b| b == 0), "slab should be zeroed");
        nsl_slab_free(slab, 1024);
    }
}
```

- [ ] **Step 2: Add `pub mod slab;` to runtime lib.rs**

After the existing `pub mod context_parallel;` line:

```rust
pub mod slab;
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p nsl-runtime slab -- --nocapture
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m36): add slab allocator runtime with alloc/free/offset FFI"
```

---

### Task 6: Register Slab FFI Builtins

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`

- [ ] **Step 1: Add slab functions to RUNTIME_FUNCTIONS array**

After the M35 GPTQ block:

```rust
    // --- M36: Memory planning slab ---
    ("nsl_slab_alloc", &[types::I64], Some(types::I64)),
    ("nsl_slab_free", &[types::I64, types::I64], None),
    ("nsl_slab_offset", &[types::I64, types::I64], Some(types::I64)),
```

- [ ] **Step 2: Verify workspace compiles**

```bash
cargo check --workspace
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(m36): register slab allocator FFI builtins"
```

---

## Phase 3: Codegen Integration

### Task 7: Compiler Fields + CompileOptions Extension

**Files:**
- Modify: `crates/nsl-codegen/src/lib.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Extend CompileOptions**

In `crates/nsl-codegen/src/lib.rs`, add to `CompileOptions`:

```rust
pub struct CompileOptions {
    pub no_autotune: bool,
    pub autotune_fresh: bool,
    pub world_size: usize,
    pub fusion_report: bool,
    /// M36: VRAM budget in bytes (None = no limit, Some(n) = fail if plan exceeds n)
    pub vram_budget: Option<u64>,
    /// M36: Print memory plan report to stderr
    pub memory_report: bool,
}
```

Update `Default`:

```rust
impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            no_autotune: false,
            autotune_fresh: false,
            world_size: 1,
            fusion_report: false,
            vram_budget: None,
            memory_report: false,
        }
    }
}
```

- [ ] **Step 2: Add `slab_plan` to Compiler struct**

In `compiler.rs`, after the `fp8_compute_fns` field:

```rust
    /// M36: Computed memory plan for slab allocation (None if planning disabled/empty)
    pub slab_plan: Option<crate::memory_planner::SlabPlan>,
```

Initialize as `slab_plan: None` in `Compiler::new()`.

- [ ] **Step 3: Verify compilation**

```bash
cargo check --workspace
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(m36): add vram_budget/memory_report to CompileOptions, slab_plan to Compiler"
```

---

### Task 8: Memory Report Formatting

**Files:**
- Modify: `crates/nsl-codegen/src/memory_planner.rs`

- [ ] **Step 1: Write memory report test**

```rust
#[test]
fn test_format_memory_report() {
    let allocs = vec![
        TensorAlloc { id: 0, name: "weight_q".into(), size_bytes: 4096, birth: 0, death: 10, source_loc: "model.nsl:5".into(), size_kind: SizeKind::Static(4096) },
        TensorAlloc { id: 1, name: "hidden".into(), size_bytes: 2048, birth: 2, death: 6, source_loc: "model.nsl:8".into(), size_kind: SizeKind::Static(2048) },
        TensorAlloc { id: 2, name: "output".into(), size_bytes: 1024, birth: 7, death: 10, source_loc: "model.nsl:12".into(), size_kind: SizeKind::Static(1024) },
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
```

- [ ] **Step 2: Implement formatting functions**

```rust
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
        format!("{} B", bytes)
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
    num_str.trim().parse::<f64>().ok().map(|n| (n * multiplier as f64) as u64)
}

/// Format a memory plan report for stderr output.
pub fn format_memory_report(allocs: &[TensorAlloc], plan: &SlabPlan) -> String {
    let planned_count = plan.assignments.len();
    let dynamic_count = allocs.iter().filter(|a| matches!(a.size_kind, SizeKind::Dynamic)).count();
    let total_count = allocs.len();

    let mut report = String::new();
    report.push_str("Memory Plan:\n");
    report.push_str(&format!("  Planned (slab): {}\n", format_bytes(plan.total_bytes)));
    report.push_str(&format!("  Slots: {}\n", plan.slots.len()));
    report.push_str(&format!("  Tensor sites: {} ({} planned, {} dynamic)\n",
        total_count, planned_count, dynamic_count));
    report.push_str(&format!("  Naive total:  {}\n", format_bytes(plan.naive_total)));
    report.push_str(&format!("  Savings:      {:.1}%\n", plan.savings_fraction() * 100.0));

    // Top consumers (sorted by size)
    let mut sorted: Vec<&TensorAlloc> = allocs.iter()
        .filter(|a| a.is_plannable())
        .collect();
    sorted.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

    if !sorted.is_empty() {
        report.push_str("\n  Top consumers:\n");
        for (i, a) in sorted.iter().take(5).enumerate() {
            report.push_str(&format!("    {}. {} ({}) [{}]\n",
                i + 1, a.name, format_bytes(a.size_bytes), a.source_loc));
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
```

- [ ] **Step 3: Add parse_vram_budget test**

```rust
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
    assert_eq!(parse_vram_budget("GB"), None); // no number
}

#[test]
fn test_savings_fraction_zero_allocs() {
    let plan = SlabPlan {
        slots: vec![], total_bytes: 0, assignments: HashMap::new(), naive_total: 0,
    };
    assert_eq!(plan.savings_fraction(), 0.0);
}

#[test]
fn test_check_vram_budget_passes() {
    let plan = SlabPlan {
        slots: vec![], total_bytes: 1024, assignments: HashMap::new(), naive_total: 2048,
    };
    assert!(check_vram_budget(&plan, 2048).is_none()); // fits
}

#[test]
fn test_check_vram_budget_fails() {
    let plan = SlabPlan {
        slots: vec![], total_bytes: 4096, assignments: HashMap::new(), naive_total: 4096,
    };
    let err = check_vram_budget(&plan, 2048);
    assert!(err.is_some());
    assert!(err.unwrap().contains("exceeds VRAM budget"));
}
```

- [ ] **Step 4: Run tests, commit**

```bash
cargo test -p nsl-codegen memory_planner -- --nocapture
git commit -m "feat(m36): add memory report formatting and VRAM budget checking"
```

---

## Phase 4: CLI Integration

### Task 9: CLI Flags

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`

- [ ] **Step 1: Find the `Build` subcommand definition**

Look for the `Build` variant in the `Cli` enum (around line 165). It should have fields like `file: PathBuf`, `dump_ir: bool`, etc.

- [ ] **Step 2: Add `--vram-budget` and `--memory-report` flags**

Add to the `Build` subcommand:

```rust
        /// M36: VRAM budget (e.g., "8GB", "512MB") — fail if plan exceeds
        #[arg(long)]
        vram_budget: Option<String>,

        /// M36: Print memory plan report
        #[arg(long)]
        memory_report: bool,
```

Also add the same flags to the `Run` subcommand if it exists.

- [ ] **Step 3: Wire flags into CompileOptions construction**

Find where `CompileOptions` is constructed (around line 196) and add:

```rust
let compile_opts = nsl_codegen::CompileOptions {
    no_autotune,
    autotune_fresh,
    world_size: 1,
    fusion_report,
    vram_budget: vram_budget.as_deref()
        .and_then(nsl_codegen::memory_planner::parse_vram_budget),
    memory_report,
};
```

NOTE: The existing code has `_compile_opts` (unused). The plan mentions this isn't wired through yet. For now, keep the flags parsing working — actual threading through `compile_entry()` will happen when CompileOptions wiring is completed. If the `compile_entry` or equivalent function already accepts CompileOptions, wire it through. Otherwise, store it for later use.

- [ ] **Step 4: Verify compilation**

```bash
cargo check --workspace
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(m36): add --vram-budget and --memory-report CLI flags"
```

---

## Phase 5: E2E Tests + Verification

### Task 10: E2E Tests

**Files:**
- Create: `examples/m36_slab_basic.nsl`
- Create: `tests/expected/m36_slab_basic.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create basic memory planning test example**

This establishes a **correctness baseline** for when slab emission is wired in M36b. Currently the program runs without slab allocation (codegen emission is deferred), but the test locks in expected output so M36b can verify slab-backed execution produces identical results.

```nsl
# M36: Memory planning — basic tensor lifecycle test

let a = ones([2, 3])
let b = zeros([2, 3])
let c = a + b
print(c)

let d = ones([3, 2])
let e = c @ d
print(e)
```

- [ ] **Step 2: Create expected output**

Run the example without memory planning to capture expected output, or compute by hand:

```
tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
tensor([[3.0, 3.0], [3.0, 3.0]])
```

NOTE: The exact tensor output format depends on the runtime's `nsl_tensor_print`. Run the example once to capture actual output and save to `tests/expected/m36_slab_basic.txt`.

- [ ] **Step 3: Add E2E tests**

```rust
// ---------------------------------------------------------------------------
// M36: Compile-Time Memory Planning
// ---------------------------------------------------------------------------

#[test]
fn e2e_m36_slab_basic() {
    assert_output_matches("m36_slab_basic");
}
```

- [ ] **Step 4: Run E2E tests**

```bash
cargo test -p nsl-cli e2e_m36 -- --nocapture
```

- [ ] **Step 5: Commit**

```bash
git commit -m "test(m36): add E2E test for basic memory planning correctness"
```

---

### Task 11: Full Verification + Clippy

- [ ] **Step 1: Run all workspace lib tests**

```bash
cargo test --workspace --lib
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Run M36-specific tests**

```bash
cargo test -p nsl-codegen memory_planner -- --nocapture
cargo test -p nsl-runtime slab -- --nocapture
cargo test -p nsl-cli e2e_m36 -- --nocapture
```

- [ ] **Step 4: Fix any issues, commit**

```bash
git commit -m "chore(m36): fix clippy warnings and verify full test suite"
```

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | TensorAlloc + SizeKind types | 3 unit |
| 2 | LivenessAnalyzer | 4 unit |
| 3 | InterferenceGraph | 4 unit |
| 4 | Slab assignment (first-fit-decreasing) | 6 unit |
| 5 | Slab runtime FFI | 5 unit |
| 6 | Slab builtins registration | compile check |
| 7 | Compiler fields + CompileOptions | compile check |
| 8 | Memory report formatting + budget | 8 unit |
| 9 | CLI flags | compile check |
| 10 | E2E test (correctness baseline) | 1 E2E |
| 11 | Full verification | all tests |

**Total: 11 tasks, ~30 unit tests + 1 E2E test**

### Deferred to M36b
- Automatic `@checkpoint` insertion when VRAM budget exceeded (`--vram-budget-auto-checkpoint`)
- Training-mode partial planning (tape-excluded activations, optimizer state planning)
- Actual codegen slab emission (replacing `checked_alloc` with `slab_ptr + offset` in expr.rs)
- GPU `cuMemAlloc` slab on device (currently CPU-only)
- Cross-function interprocedural memory planning
- AST-walking integration in `compile_main()` to populate `LivenessAnalyzer` from real program
- Branch/loop lifetime extension (conservative analysis)
- Bounded dimension size computation from type_map shapes
