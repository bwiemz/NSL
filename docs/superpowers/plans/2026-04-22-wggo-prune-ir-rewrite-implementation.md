# WGGO Prune v1 — Sub-block IR Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the real WGGO Prune IR rewrite (v1, sub-block residual identity-alias) specified in `docs/superpowers/specs/2026-04-22-wggo-prune-ir-rewrite-design.md`, replacing PR #102's diagnostic stub with actual Wengert-list transformation.

**Architecture:** A new `wggo_prune.rs` module with a dry-run-then-commit `run()` entry point. Phase 1 validates all `CoarseDecision::Prune` decisions from the `AppliedPlan` by computing a parameter-anchored data-flow closure and pattern-matching the residual `Add(h_before, block_output)`. Phase 2 early-returns on any refusal (wengert unchanged, all refusals batched). Phase 3 commits: delete closure ops, repoint residual Add consumers to `h_before`, delete the Add. Caller in `stmt.rs` aggregates refusals, fails compilation on any, and passes `pruned_forward_var_ids` to `wrga_prune::prune()`.

**Tech Stack:** Rust 1.95.0, Cranelift, `insta` for snapshot tests, existing NSL Wengert IR (`WengertList`, `WengertOp`, `PrimalOp`), existing WGGO machinery (`AppliedPlan`, `LayerRole`, `CoarseDecision`), existing WRGA Prune (`wrga_prune::prune`).

**Reference spec:** [2026-04-22-wggo-prune-ir-rewrite-design.md](../specs/2026-04-22-wggo-prune-ir-rewrite-design.md) (commit `0407944c`, 581 lines). All task-level references to §N.M cite that spec.

---

## Codebase Anchors (pin these before starting)

These are concrete file:line locations the plan assumes. Verify each exists before Task 1; if any has moved, adjust the plan's references.

| Anchor | File:Line |
|---|---|
| `PrimalOp::Add` variant | `crates/nsl-codegen/src/wengert.rs:77` |
| `WengertOp` struct | `crates/nsl-codegen/src/wengert.rs:49-56` |
| `WengertList` struct | `crates/nsl-codegen/src/wengert.rs:315-321` |
| `VarId`, `OpId` type aliases (both `= u32`) | `crates/nsl-codegen/src/wengert.rs:5-6` |
| `AppliedLayer`, `AppliedPlan` | `crates/nsl-codegen/src/wggo_apply.rs:21-54` |
| `LayerRole` enum | `crates/nsl-codegen/src/wggo_graph.rs:24-32` |
| `LayerDecision` (aliased as `CoarseDecision`) | `crates/nsl-codegen/src/wggo_dp.rs:23-30` |
| `WeightMap` + `entries()` iterator | `crates/nsl-codegen/src/weight_aware.rs:377-386, 516-518` |
| `wrga_prune::prune()` entry (NOT `::run`) | `crates/nsl-codegen/src/wrga_prune.rs:190-194` |
| `OverrideRejectReason` enum | `crates/nsl-codegen/src/wggo_overrides.rs:27-65` |
| `collect_prune_diagnostics()` | `crates/nsl-codegen/src/wggo_overrides.rs:170-186` |
| `prune_not_implemented_reason()` | `crates/nsl-codegen/src/wggo_overrides.rs:192-194` |
| WGGO Prune insertion site (before WRGA invocation) | `crates/nsl-codegen/src/stmt.rs:4292` (before `invoke_wrga_if_enabled`) |
| `eliminate_by_backward_live` call-site | `crates/nsl-codegen/src/stmt.rs:4515-4523` |
| `insta::assert_snapshot!` pattern example | `crates/nsl-codegen/tests/fa_v2_snapshots.rs:27-30` |
| Snapshot file location | `crates/nsl-codegen/tests/snapshots/*.snap` |

**Note on `DiagnosticCode`:** the spec's §6.3 refers to `DiagnosticCode::Prune*` variants. No such enum exists in this crate; `OverrideRejectReason` in `wggo_overrides.rs` is the equivalent surface. **Add the 7 new refusal variants to `OverrideRejectReason`, not to a new enum.** Spec §6.3's wording should be read as "`OverrideRejectReason::Prune*`" in practice.

---

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `crates/nsl-codegen/src/wggo_prune.rs` | Public `run()`, `PruneRewriteResult`, `PruneRewrite`, `PruneRefusal`; internal closure computation, pattern-match, Phase 1 validator, Phase 3 mutation. ~400-600 LOC. |
| `crates/nsl-codegen/tests/wggo_prune_rewrite.rs` | Layer 3 numerical-equivalence tests + Layer 4 stderr+DiagnosticCode tests. |
| `crates/nsl-codegen/tests/fixtures/prune_rewrite_toy.nsl` | 4-block pre-norm transformer baseline. |
| `crates/nsl-codegen/tests/fixtures/prune_rewrite_toy_ref_*.nsl` | Reference variants (one per Layer 3 integration test — ~8 files). |
| `crates/nsl-codegen/tests/snapshots/wggo_prune_rewrite__*.snap` | Insta snapshots (Layer 1 + Layer 4 stderr). Auto-generated. |

**Modify:**

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Declare `pub mod wggo_prune;` |
| `crates/nsl-codegen/src/wggo_overrides.rs` | Rename `PruneNotImplemented` → `WholeBlockPruneNotImplemented`; rename `prune_not_implemented_reason()` → `whole_block_prune_not_implemented_reason()`; narrow `collect_prune_diagnostics()` to iterate only `LayerRole::Block` (skip sub-block prune decisions — those flow through `wggo_prune::run()`); add 7 new variants for prune refusals. |
| `crates/nsl-codegen/src/stmt.rs` | Insert `wggo_prune::run()` call before line 4292 (`invoke_wrga_if_enabled`). Aggregate refusals, fail compilation if any. Emit success-path stderr lines. Update the existing `OverrideRejectReason::PruneNotImplemented` match arm to use the renamed variant. |

---

## Execution Order

Tasks are grouped into four phases matching the spec's informative Commit structure (§10). Each phase ends with a commit. Within a phase, tasks run sequentially — later tasks depend on earlier ones.

- **Phase A** (Commit A): Scaffolding — Tasks 1–3
- **Phase B** (Commit B): Validation (closure + 7 refusals) — Tasks 4–11
- **Phase C** (Commit C): Mutation + integration tests — Tasks 12–14
- **Phase D** (Commit D): Diagnostics + Layer 4 tests — Tasks 15–16

---

# Phase A — Scaffolding (Commit A)

### Task 1: Module skeleton with public API types

**Files:**

- Create: `crates/nsl-codegen/src/wggo_prune.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod wggo_prune;`)

- [ ] **Step 1: Write the failing test** — assert the module and types exist and can be constructed.

Create `crates/nsl-codegen/tests/wggo_prune_skeleton_test.rs`:

```rust
// Compile-only skeleton check: the module, types, and entry point exist
// and have the signatures the spec mandates.

use nsl_codegen::wggo_prune::{
    PruneRewrite, PruneRewriteResult, PruneRefusal, run,
};
use std::collections::BTreeSet;

#[test]
fn types_are_constructible_and_run_is_callable() {
    // Empty PruneRewriteResult constructs.
    let _r = PruneRewriteResult {
        rewrites: Vec::<PruneRewrite>::new(),
        refusals: Vec::<PruneRefusal>::new(),
        pruned_forward_var_ids: BTreeSet::new(),
        ops_deleted: 0,
    };
    // (run() is callable with an empty plan — deferred until the full signature lands.)
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/wggo-prune-ir-rewrite
cargo test -p nsl-codegen --test wggo_prune_skeleton_test 2>&1 | tail -5
```

Expected: FAIL with "unresolved import `nsl_codegen::wggo_prune`" or similar.

- [ ] **Step 3: Write minimal implementation**

Create `crates/nsl-codegen/src/wggo_prune.rs`:

```rust
//! Layer-level Wengert rewriting driven by WGGO `CoarseDecision::Prune`.
//!
//! Distinct from `wrga_prune.rs`, which handles parameter-level `backward_live`
//! filtering for frozen adapter weights. This module removes whole layer
//! computations from the forward; `wrga_prune` then computes `backward_live`
//! on the already-reduced forward.
//!
//! Pipeline position: runs before `wrga_prune::prune()` in `stmt.rs`, and
//! therefore before source-AD's adjoint generation. The rewrite produces
//! the final forward Wengert that both WRGA Prune and source-AD will consume.
//!
//! Design principle: this module refuses transformations when preconditions
//! aren't met; it does not fall back to weaker transformations with different
//! semantics. See memory/feedback_transformation_precondition_refusal.md for
//! the generalized rule.

use std::collections::BTreeSet;

use crate::wengert::{OpId, VarId, WengertList};
use crate::weight_aware::WeightMap;
use crate::wggo_apply::AppliedPlan;
use crate::wggo_graph::LayerRole;

/// Outcome of `run()`. Either `rewrites` is populated and `refusals` is empty
/// (all prune decisions applied), or `refusals` is populated and `rewrites`
/// is empty (any refusal → nothing applied; `wengert` is unchanged).
pub struct PruneRewriteResult {
    pub rewrites: Vec<PruneRewrite>,
    pub refusals: Vec<PruneRefusal>,
    pub pruned_forward_var_ids: BTreeSet<VarId>,
    pub ops_deleted: usize,
}

/// Record of one layer successfully pruned.
pub struct PruneRewrite {
    pub layer_name: String,
    pub layer_role: LayerRole,
    pub h_before_var: VarId,
    pub h_after_var: VarId,
    pub residual_add_op: OpId,
    pub closure_ops: Vec<OpId>,
}

/// A refusal. One variant per precondition failure enumerated in spec §3.
pub enum PruneRefusal {
    CrossLayerParam {
        layer_name: String,
        layer_role: LayerRole,
        param_name: String,
        param_var: VarId,
        external_consumer: OpId,
        external_op_kind: String,
    },
    NoResidualAdd {
        layer_name: String,
        layer_role: LayerRole,
        closure_size: usize,
    },
    ParallelResidualBranches {
        layer_name: String,
        layer_role: LayerRole,
        add_ops: Vec<OpId>,
    },
    AmbiguousPatternMatch {
        layer_name: String,
        layer_role: LayerRole,
        h_before_var: VarId,
        candidate_adds: Vec<OpId>,
    },
    EmptyClosure {
        layer_name: String,
        layer_role: LayerRole,
        prefix: String,
    },
    WholeBlockUnsupported {
        layer_name: String,
    },
    ConflictingPruneDecisions {
        decision_a: String,
        decision_b: String,
        reason: String,
    },
}

/// Entry point. Dry-run-then-commit: validates all decisions first; applies
/// mutations only if all pass. On refusal, `wengert` is unchanged.
///
/// See spec §5.3 for the three-phase contract.
pub fn run(
    _wengert: &mut WengertList,
    _applied_plan: &AppliedPlan,
    _weight_map: &WeightMap,
) -> PruneRewriteResult {
    // Stub: Phase 1/2/3 land in Tasks 4–13.
    PruneRewriteResult {
        rewrites: Vec::new(),
        refusals: Vec::new(),
        pruned_forward_var_ids: BTreeSet::new(),
        ops_deleted: 0,
    }
}
```

Add to `crates/nsl-codegen/src/lib.rs` (search for an existing `pub mod wggo_overrides;` line and insert alphabetically):

```rust
pub mod wggo_prune;
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test -p nsl-codegen --test wggo_prune_skeleton_test 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Verify full crate still compiles cleanly**

```bash
cargo check -p nsl-codegen 2>&1 | tail -5
```

Expected: `Finished` with no errors. Warnings about `_wengert`/`_applied_plan`/`_weight_map` being unused are OK — they'll be used in Tasks 4+.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/tests/wggo_prune_skeleton_test.rs
git commit -m "feat(wggo-prune): module skeleton + public API types

Per spec §5.2. Types defined, run() stubbed to return empty result.
Phase 1 validator (closure + pattern-match + refusals) lands in Phase B;
Phase 3 mutation lands in Phase C.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Rename `PruneNotImplemented` → `WholeBlockPruneNotImplemented` with preserved stderr contract

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_overrides.rs` (5 occurrences per Grep)
- Modify: `crates/nsl-codegen/src/stmt.rs:4086` (the existing match arm)

- [ ] **Step 1: Write the failing test** — assert the new function name exists AND the stderr contract string is unchanged.

Add to `crates/nsl-codegen/tests/wggo_overrides_rename_test.rs` (new file):

```rust
// Verifies the PR #102 stderr-contract string survives the
// PruneNotImplemented → WholeBlockPruneNotImplemented rename.
// Spec §5.5: the string "ir_rewrite_not_implemented" is emitted via
// an explicit function return, NOT via a debug-formatted variant name.

use nsl_codegen::wggo_overrides::whole_block_prune_not_implemented_reason;

#[test]
fn whole_block_prune_reason_string_preserves_pr102_contract() {
    assert_eq!(
        whole_block_prune_not_implemented_reason(),
        "ir_rewrite_not_implemented",
        "PR #102's stderr-contract string must not change on rename"
    );
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-codegen --test wggo_overrides_rename_test 2>&1 | tail -5
```

Expected: FAIL — `whole_block_prune_not_implemented_reason` doesn't exist.

- [ ] **Step 3: Apply the rename**

In `crates/nsl-codegen/src/wggo_overrides.rs`:

Change line 64 from:

```rust
    PruneNotImplemented,
```

to:

```rust
    /// Whole-block prune (LayerRole::Block) — not yet implemented; see
    /// spec §3.6. Sub-block prune is handled by wggo_prune.rs.
    WholeBlockPruneNotImplemented,
```

Change line 181 (inside `collect_prune_diagnostics`) from:

```rust
            reason: OverrideRejectReason::PruneNotImplemented,
```

to:

```rust
            reason: OverrideRejectReason::WholeBlockPruneNotImplemented,
```

Change line 188 doc-comment prefix (the line before the fn) from `/// Stable reason-string for \`OverrideRejectReason::PruneNotImplemented\`,` to `/// Stable reason-string for \`OverrideRejectReason::WholeBlockPruneNotImplemented\`,`.

Rename the function at lines 192-194 from:

```rust
pub fn prune_not_implemented_reason() -> &'static str {
    "ir_rewrite_not_implemented"
}
```

to:

```rust
pub fn whole_block_prune_not_implemented_reason() -> &'static str {
    // Stable string preserved from PR #102; see spec §5.5.
    "ir_rewrite_not_implemented"
}
```

Rename line 357 (inside a test) from:

```rust
        assert!(matches!(diags[0].reason, OverrideRejectReason::PruneNotImplemented));
```

to:

```rust
        assert!(matches!(diags[0].reason, OverrideRejectReason::WholeBlockPruneNotImplemented));
```

Also update line 371 (test of reason string):

```rust
        assert_eq!(whole_block_prune_not_implemented_reason(), "ir_rewrite_not_implemented");
```

In `crates/nsl-codegen/src/stmt.rs:4086`, change the match arm from:

```rust
                                    crate::wggo_overrides::OverrideRejectReason::PruneNotImplemented => {
```

to:

```rust
                                    crate::wggo_overrides::OverrideRejectReason::WholeBlockPruneNotImplemented => {
```

Inside that arm, verify the stderr emission uses the renamed function:

```rust
let reason_str = crate::wggo_overrides::whole_block_prune_not_implemented_reason();
```

(not `format!("{:?}", variant)`). If the existing arm uses `format!("{:?}", ...)`, rewrite it to call `whole_block_prune_not_implemented_reason()` explicitly.

- [ ] **Step 4: Narrow `collect_prune_diagnostics` to `LayerRole::Block` only**

In `crates/nsl-codegen/src/wggo_overrides.rs:170-186`, change the filter from:

```rust
        .filter(|l| matches!(l.coarse, LayerDecision::Prune))
```

to:

```rust
        .filter(|l| {
            matches!(l.coarse, LayerDecision::Prune)
                && matches!(l.layer_role, crate::wggo_graph::LayerRole::Block)
        })
```

**Note:** if `AppliedLayer` doesn't directly carry `layer_role`, look up the role via `applied.layers` index or inspect the LayerRole-carrying sibling struct. Confirm the access path against `wggo_apply.rs` before editing. If `layer_role` isn't on `AppliedLayer`, either (a) add it by mirroring the planner's tagging, or (b) infer from `layer_name` conventions (`.attn`/`.ffn` suffix → sub-block, bare `blocks.N` → Block).

- [ ] **Step 5: Run all tests to verify rename is complete**

```bash
cargo test -p nsl-codegen --test wggo_overrides_rename_test 2>&1 | tail -5
cargo check -p nsl-codegen 2>&1 | tail -5
```

Expected: PASS + clean compile. No remaining references to `PruneNotImplemented` or `prune_not_implemented_reason`:

```bash
grep -rn "PruneNotImplemented\b" crates/nsl-codegen/src/ 2>&1 | head -5
grep -rn "prune_not_implemented_reason\b" crates/nsl-codegen/src/ 2>&1 | head -5
```

Expected: both greps return no results (or only `WholeBlockPruneNotImplemented` / `whole_block_prune_not_implemented_reason` which are the renamed forms).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wggo_overrides.rs crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/tests/wggo_overrides_rename_test.rs
git commit -m "refactor(wggo-overrides): rename PruneNotImplemented → WholeBlockPruneNotImplemented

Preserves PR #102's stderr-contract string (\"ir_rewrite_not_implemented\")
via explicit whole_block_prune_not_implemented_reason() return.
collect_prune_diagnostics() now filters to LayerRole::Block only; sub-block
prune decisions flow through wggo_prune::run() (landing in Phase B).

Spec §5.5 — rename was grep-verified internal-only (5 hits, all in
nsl-codegen; no serialized fixtures).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add 7 refusal variants to `OverrideRejectReason` + wire `stmt.rs` call-site

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_overrides.rs` (add 7 variants)
- Modify: `crates/nsl-codegen/src/stmt.rs` (insert `wggo_prune::run()` call before line 4292)

- [ ] **Step 1: Write the failing test** — assert the new variants exist and `wggo_prune::run()` is called before WRGA.

Create `crates/nsl-codegen/tests/wggo_prune_variants_test.rs`:

```rust
// Verifies the 7 new OverrideRejectReason variants exist per spec §6.3,
// using the OverrideRejectReason enum in place of the spec's DiagnosticCode
// (the codebase has no parallel DiagnosticCode enum).

use nsl_codegen::wggo_overrides::OverrideRejectReason;

#[test]
fn prune_refusal_variants_exist() {
    // Each variant should construct (fields are placeholders for now).
    let _ = OverrideRejectReason::PruneCrossLayerParam;
    let _ = OverrideRejectReason::PruneNoResidualAdd;
    let _ = OverrideRejectReason::PruneParallelResidualBranches;
    let _ = OverrideRejectReason::PruneAmbiguousPatternMatch;
    let _ = OverrideRejectReason::PruneEmptyClosure;
    let _ = OverrideRejectReason::PruneWholeBlockUnsupported;
    let _ = OverrideRejectReason::PruneConflictingDecisions;
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-codegen --test wggo_prune_variants_test 2>&1 | tail -5
```

Expected: FAIL — variants don't exist.

- [ ] **Step 3: Add the variants to `OverrideRejectReason`**

In `crates/nsl-codegen/src/wggo_overrides.rs`, append to the enum (after `WholeBlockPruneNotImplemented`):

```rust
    /// Prune refusal — spec §3.1. Cross-layer parameter consumption.
    PruneCrossLayerParam,
    /// Prune refusal — spec §3.2. Layer lacks a residual `Add`.
    PruneNoResidualAdd,
    /// Prune refusal — spec §3.3. Parallel residual branches detected.
    PruneParallelResidualBranches,
    /// Prune refusal — spec §3.4. Multiple Adds match the residual pattern ambiguously.
    PruneAmbiguousPatternMatch,
    /// Prune refusal — spec §3.5. No parameters match the requested layer prefix.
    PruneEmptyClosure,
    /// Prune refusal — spec §3.6. Whole-block prune (LayerRole::Block) unsupported in v1.
    /// Distinct from `WholeBlockPruneNotImplemented`: this variant is emitted by
    /// `wggo_prune::run()` via the new sub-block flow; `WholeBlockPruneNotImplemented`
    /// is emitted by the legacy `collect_prune_diagnostics` path. Both coexist
    /// during v1; v2 will consolidate.
    PruneWholeBlockUnsupported,
    /// Prune refusal — spec §3.7. Two prune decisions in the same plan conflict.
    PruneConflictingDecisions,
```

- [ ] **Step 4: Wire the `stmt.rs` call-site**

In `crates/nsl-codegen/src/stmt.rs`, find line 4292 (`let wrga_plan = crate::stmt::invoke_wrga_if_enabled(self, extractor.wengert_list());`).

Immediately BEFORE that line, insert:

```rust
// Spec §4: WGGO Prune runs before WRGA so wrga_prune::prune() sees the
// already-reduced forward. On any refusal, compile fails.
let wggo_prune_result = {
    let applied_plan = /* access the AppliedPlan for this compilation; the
       exact path depends on where WGGO output is threaded — look for
       `applied_plan` or `wggo_applied` in the surrounding function scope. If
       not readily accessible, fall back to calling wggo_apply::apply(...)
       directly with the WGGO intermediate plan. */ ;
    let weight_map = /* same accessor used by wrga_prune / source_ad; look
       for `weight_map` or `wm` in the surrounding scope. */ ;
    // `extractor.wengert_list()` currently returns `&WengertList`; add a
    // `wengert_list_mut()` accessor to `Extractor` if missing.
    crate::wggo_prune::run(
        extractor.wengert_list_mut(),
        applied_plan,
        weight_map,
    )
};

if !wggo_prune_result.refusals.is_empty() {
    // Spec §5.4 step 2: emit all refusals, fail compilation (no whack-a-mole).
    for refusal in &wggo_prune_result.refusals {
        // Stderr + structured OverrideRejectReason emission lands in Phase D
        // (Task 15). For now, print a bare placeholder line so the compile
        // failure is observable during Phase B development.
        eprintln!("[prune] refusal: {:?}", std::mem::discriminant(refusal));
    }
    return Err(/* existing compile-error type; mirror what invoke_wrga_if_enabled
       returns on error */);
}

// Else: each successful rewrite gets a stderr line in Phase D (Task 15).
// Thread pruned_forward_var_ids into wrga — see Phase C Task 12 for wiring.
```

**Implementer note:** the exact `Err(...)` construction and the accessor names (`applied_plan`, `weight_map`) depend on the surrounding function's control flow. Read lines 4200-4310 of `stmt.rs` carefully before editing; prefer mirroring the existing error-propagation pattern over inventing one. If the function returns `Result<_, E>`, use `E`; if it panics, panic with the same formatting convention.

**If `extractor.wengert_list_mut()` doesn't exist:** add it as a public method on `Extractor` returning `&mut WengertList`. One-line addition; mirror whatever `wengert_list()` (immutable) currently does.

- [ ] **Step 5: Run tests + full crate compile**

```bash
cargo test -p nsl-codegen --test wggo_prune_variants_test 2>&1 | tail -5
cargo check -p nsl-codegen 2>&1 | tail -5
```

Expected: variant test PASS; full crate compile clean (any `Result`-type mismatch on the new `return Err(...)` line surfaces here — fix by matching the function's existing error type).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wggo_overrides.rs crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/tests/wggo_prune_variants_test.rs
git commit -m "feat(wggo-prune): add 7 refusal variants + stmt.rs call-site wiring

OverrideRejectReason gains Prune{CrossLayerParam, NoResidualAdd,
ParallelResidualBranches, AmbiguousPatternMatch, EmptyClosure,
WholeBlockUnsupported, ConflictingDecisions} per spec §3.
stmt.rs now calls wggo_prune::run() before invoke_wrga_if_enabled so WRGA
sees the reduced forward (spec §4). run() is currently a stub; Phase B
lands closure + pattern-match + refusals; Phase C lands Phase 3 mutation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**End of Phase A.** At this point: module exists, types are real, call-site is wired, and the whole thing compiles but does nothing. Subsequent phases fill in behavior.

---

# Phase B — Validation (Commit B)

All Phase B tasks add to `wggo_prune.rs` and add/extend unit tests. Each task is a single TDD cycle: write failing test, implement the code to make it pass, commit.

**Shared test scaffolding — put this once near the top of the file added in Task 4 and reuse it in all subsequent Phase B tasks:**

```rust
// In crates/nsl-codegen/src/wggo_prune.rs (within #[cfg(test)] mod tests).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp, WengertList};
    use std::collections::HashMap;

    /// Build a minimal synthetic WengertList by hand, without going through
    /// the NSL compiler. Useful for closure/pattern-match unit tests.
    fn mk_wengert(ops: Vec<WengertOp>, output: VarId, var_names: &[(VarId, &str)]) -> WengertList {
        WengertList {
            ops,
            output,
            var_names: var_names.iter().map(|(v, s)| (*v, s.to_string())).collect(),
            var_types: HashMap::new(),
        }
    }

    /// Shorthand: non-param, non-Add op with one input.
    fn op_unary(id: OpId, result: VarId, input: VarId, kind: PrimalOp) -> WengertOp {
        WengertOp {
            id, result, op: kind, inputs: vec![input],
            saved_for_backward: false, checkpointed: false,
        }
    }

    fn op_add(id: OpId, result: VarId, a: VarId, b: VarId) -> WengertOp {
        WengertOp {
            id, result, op: PrimalOp::Add, inputs: vec![a, b],
            saved_for_backward: false, checkpointed: false,
        }
    }

    /// Build a minimal AppliedLayer for Prune with a given name and role.
    fn mk_prune_layer(idx: u32, name: &str, role: LayerRole) -> crate::wggo_apply::AppliedLayer {
        use crate::wggo_apply::AppliedLayer;
        use crate::wggo_dp::LayerDecision;
        AppliedLayer {
            layer_index: idx,
            layer_name: name.to_string(),
            coarse: LayerDecision::Prune,
            pipeline_stage: 0,
            shard_factor: 1,
            active_heads: 0,
            ffn_width: 0,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 0,
            optim_v_bits: 0,
            fase_fused: false,
            packing_mode: 0,
            estimated_us: 0.0,
            param_bytes: 0,
            activation_bytes: 0,
        }
    }

    fn mk_applied_plan(layers: Vec<crate::wggo_apply::AppliedLayer>) -> crate::wggo_apply::AppliedPlan {
        crate::wggo_apply::AppliedPlan {
            layers,
            total_us: 0.0,
            peak_memory_bytes: 0,
        }
    }
}
```

**Note:** if `AppliedLayer` doesn't carry `layer_role`, Task 2 Step 4's comment applies — the implementer has already decided how to carry or infer it. Adapt `mk_prune_layer` accordingly.

---

### Task 4: Closure computation (positive case) + internal `plan_rewrite` signature

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs` (add `plan_rewrite`, `compute_closure`, `PruneRewritePlan`, `PlanResult`)

- [ ] **Step 1: Write the failing test** — synthesize a 3-op block with a residual Add; closure should capture the 3 compute ops (NOT the Add).

Add to the `tests` module:

```rust
#[test]
fn closure_captures_transitive_compute_ops() {
    // Wengert list modeling: h_after = h_before + (h_before * blocks.7.attn.wq)
    //
    //   op0: PrimalOp::LoadParam(blocks.7.attn.wq)  → v0   (producer of param VarId)
    //   op1: PrimalOp::Mul  inputs=[v_hb, v0]        → v1   (block_output)
    //   op2: PrimalOp::Add  inputs=[v_hb, v1]        → v_ha (residual boundary)
    //
    // h_before = v_hb, h_after = v_ha, block_output = v1.
    // Expected closure: {op0, op1}. The Add (op2) is the BOUNDARY, not the closure.

    let v_hb: VarId = 100;
    let v0:   VarId = 200;
    let v1:   VarId = 201;
    let v_ha: VarId = 202;

    let ops = vec![
        // Param producer: use any leaf-like op variant; if PrimalOp has a
        // `LoadParam` or `Input` variant, use it. Otherwise use Mul with a
        // dummy VarId that's not declared elsewhere — the test only exercises
        // the closure walk, not the op's semantics.
        op_unary(0, v0, v_hb /* dummy input, not meaningful */, PrimalOp::Mul),
        op_unary(1, v1, v0, PrimalOp::Mul), // block_output
        op_add(2, v_ha, v_hb, v1),          // residual Add — the boundary
    ];

    let wengert = mk_wengert(
        ops,
        v_ha,
        &[(v_hb, "h_before"), (v0, "blocks.7.attn.wq"), (v_ha, "h_after")],
    );

    let layer = mk_prune_layer(7, "blocks.7.attn", LayerRole::Attention);

    // plan_rewrite is the internal Phase 1 validator.
    let result = plan_rewrite(&wengert, &layer, /* weight_map */ &empty_weight_map());

    match result {
        PlanResult::Ok(plan) => {
            assert_eq!(plan.closure_op_ids, vec![0, 1], "closure should include op0 (param producer) and op1 (compute), but NOT op2 (residual Add)");
            assert_eq!(plan.residual_add_op_id, 2);
            assert_eq!(plan.h_before_var, v_hb);
            assert_eq!(plan.h_after_var, v_ha);
        }
        PlanResult::Refused(r) => panic!("expected Ok, got Refused({:?})", r),
    }
}

fn empty_weight_map() -> crate::weight_aware::WeightMap {
    // Use whatever WeightMap constructor exists for synthetic tests.
    // If a `::empty()` or `::new()` isn't exposed, construct via the
    // default path; the test only needs entries() to be iterable.
    crate::weight_aware::WeightMap::empty_for_tests()
        // If no such method exists, add a `#[cfg(test)] pub fn empty_for_tests()`
        // helper inside weight_aware.rs that returns an empty map.
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests::closure_captures_transitive_compute_ops 2>&1 | tail -10
```

Expected: FAIL — `plan_rewrite`, `PlanResult`, `PruneRewritePlan` don't exist.

- [ ] **Step 3: Implement `plan_rewrite` with closure computation (positive case only)**

Add to `wggo_prune.rs` (outside the `tests` module):

```rust
/// Internal Phase 1 result: either a validated plan ready to commit, or
/// a refusal.
pub(crate) enum PlanResult {
    Ok(PruneRewritePlan),
    Refused(PruneRefusal),
}

/// Internal Phase 1 output: everything `apply_rewrite` needs to commit
/// the mutation without re-computing anything.
pub(crate) struct PruneRewritePlan {
    pub(crate) layer_name: String,
    pub(crate) layer_role: LayerRole,
    pub(crate) closure_op_ids: Vec<OpId>,   // topologically sorted; deleted in Phase 3
    pub(crate) residual_add_op_id: OpId,    // rewritten then deleted
    pub(crate) h_before_var: VarId,
    pub(crate) h_after_var: VarId,
    pub(crate) parameter_var_ids: BTreeSet<VarId>, // layer-N params (subset of closure outputs)
}

/// Phase 1 validator for a single `CoarseDecision::Prune` decision.
/// Does NOT mutate `wengert`.
pub(crate) fn plan_rewrite(
    wengert: &WengertList,
    layer: &crate::wggo_apply::AppliedLayer,
    weight_map: &WeightMap,
) -> PlanResult {
    // (a) Reject LayerRole::Block immediately — spec §3.6.
    if matches!(layer.layer_role, LayerRole::Block) {
        return PlanResult::Refused(PruneRefusal::WholeBlockUnsupported {
            layer_name: layer.layer_name.clone(),
        });
    }

    // (b) Find parameter VarIds matching `{layer_name}.` prefix.
    let prefix = format!("{}.", layer.layer_name);
    let param_var_ids: BTreeSet<VarId> = wengert
        .var_names
        .iter()
        .filter_map(|(v, name)| name.starts_with(&prefix).then_some(*v))
        .collect();

    if param_var_ids.is_empty() {
        return PlanResult::Refused(PruneRefusal::EmptyClosure {
            layer_name: layer.layer_name.clone(),
            layer_role: layer.layer_role,
            prefix: prefix.clone(),
        });
    }

    // (c) Compute the data-flow closure — transitive consumers of any layer-N
    //     parameter VarId, terminating at residual Add candidates.
    let closure_op_ids = compute_closure(wengert, &param_var_ids);

    // (d) Find the residual Add at the layer boundary + verify it's single-use
    //     per spec §1.3. Pattern-match happens in a helper (Tasks 5–8 extend
    //     it with the 4 refusal variants).
    let (residual_add_op_id, h_before_var, h_after_var) =
        match find_residual_add(wengert, &closure_op_ids, &param_var_ids) {
            Ok(triple) => triple,
            Err(refusal) => return PlanResult::Refused(refusal_with_context(refusal, layer)),
        };

    PlanResult::Ok(PruneRewritePlan {
        layer_name: layer.layer_name.clone(),
        layer_role: layer.layer_role,
        closure_op_ids,
        residual_add_op_id,
        h_before_var,
        h_after_var,
        parameter_var_ids: param_var_ids,
    })
}

/// Compute the transitive forward-closure of ops owned by the layer.
///
/// Spec §2.2: closure = { ops producing a layer-N parameter VarId } ∪ { ops
/// transitively dependent on layer-N outputs }, excluding the residual Add.
///
/// Returns OpIds in topological order (same order as `wengert.ops`).
pub(crate) fn compute_closure(wengert: &WengertList, param_var_ids: &BTreeSet<VarId>) -> Vec<OpId> {
    // "Tainted" VarIds: either a layer-N parameter, or a VarId produced by
    // a closure op. An op is in the closure iff any of its inputs is tainted
    // OR any of its inputs is a layer-N parameter VarId.
    let mut tainted_vars: BTreeSet<VarId> = param_var_ids.clone();
    let mut closure: Vec<OpId> = Vec::new();

    for op in &wengert.ops {
        // Param producer: result is a layer-N param VarId (in tainted_vars already).
        let produces_param = param_var_ids.contains(&op.result);
        // Compute op: any input is tainted.
        let reads_tainted = op.inputs.iter().any(|v| tainted_vars.contains(v));

        if produces_param || reads_tainted {
            // Is this the residual Add? If so, EXCLUDE from the closure
            // (per spec §2.2 three-category treatment).
            if matches!(op.op, PrimalOp::Add) && op.inputs.len() == 2 {
                let (a, b) = (op.inputs[0], op.inputs[1]);
                // Residual Add: one input is tainted (the block_output), the
                // other is NOT (the h_before — comes from the prior stream).
                let a_tainted = tainted_vars.contains(&a);
                let b_tainted = tainted_vars.contains(&b);
                if a_tainted != b_tainted {
                    // Residual boundary — don't add to closure, don't taint result.
                    continue;
                }
            }
            closure.push(op.id);
            tainted_vars.insert(op.result);
        }
    }

    closure
}

/// Stub for Task 5+. Returns `(residual_add_op_id, h_before_var, h_after_var)`
/// on success; a refusal enum without layer context on failure.
fn find_residual_add(
    wengert: &WengertList,
    closure: &[OpId],
    param_var_ids: &BTreeSet<VarId>,
) -> Result<(OpId, VarId, VarId), PartialRefusal> {
    // Walk ops NOT in closure; find any Add whose one input is tainted-by-closure
    // and whose other input is not.
    //
    // For Task 4 (positive case only): return the first match. Tasks 5–8 extend
    // this with refusal-case detection (zero matches, multiple parallel, ambiguous).

    let closure_set: BTreeSet<OpId> = closure.iter().copied().collect();
    let tainted: BTreeSet<VarId> = {
        let mut t: BTreeSet<VarId> = param_var_ids.clone();
        for op in &wengert.ops {
            if closure_set.contains(&op.id) {
                t.insert(op.result);
            }
        }
        t
    };

    for op in &wengert.ops {
        if closure_set.contains(&op.id) { continue; }
        if !matches!(op.op, PrimalOp::Add) { continue; }
        if op.inputs.len() != 2 { continue; }
        let (a, b) = (op.inputs[0], op.inputs[1]);
        let a_tainted = tainted.contains(&a);
        let b_tainted = tainted.contains(&b);
        // Residual pattern: exactly one input tainted (the block_output); the
        // other is h_before (external to the closure).
        if a_tainted != b_tainted {
            let (h_before, _block_output) = if a_tainted { (b, a) } else { (a, b) };
            return Ok((op.id, h_before, op.result));
        }
    }

    // Placeholder — Tasks 5–8 replace this with specific refusal construction.
    Err(PartialRefusal::NoResidualAdd)
}

/// Internal helper: refusal variants that `find_residual_add` can produce,
/// without yet knowing the layer_name/layer_role. `plan_rewrite` wraps them.
#[derive(Debug)]
enum PartialRefusal {
    NoResidualAdd,
    // Task 6 adds ParallelResidualBranches { add_ops: Vec<OpId> }
    // Task 7 adds AmbiguousPatternMatch { h_before: VarId, candidate_adds: Vec<OpId> }
}

fn refusal_with_context(
    partial: PartialRefusal,
    layer: &crate::wggo_apply::AppliedLayer,
) -> PruneRefusal {
    match partial {
        PartialRefusal::NoResidualAdd => PruneRefusal::NoResidualAdd {
            layer_name: layer.layer_name.clone(),
            layer_role: layer.layer_role,
            closure_size: 0, // overwritten by caller if known — refine in Task 5.
        },
    }
}
```

Also: if `LayerRole` doesn't `derive(Copy, Clone, Debug)`, add those derives in `wggo_graph.rs:24-32`. The tests need them; shouldn't be a breaking change.

If `WeightMap::empty_for_tests()` doesn't exist, add it in `weight_aware.rs`:

```rust
#[cfg(test)]
impl WeightMap {
    pub fn empty_for_tests() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            file_hash: [0u8; 32],
            source_path: String::new(),
            total_bytes: 0,
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests::closure_captures_transitive_compute_ops 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Run full crate compile**

```bash
cargo check -p nsl-codegen 2>&1 | tail -5
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs crates/nsl-codegen/src/wggo_graph.rs crates/nsl-codegen/src/weight_aware.rs
git commit -m "feat(wggo-prune): Phase 1 closure computation + positive pattern-match

plan_rewrite() + compute_closure() + find_residual_add() implement the
spec §2.2 closure rule and §1.3 pattern-match for the positive case.
Phase 1 refusals land in Tasks 5-11; Phase 3 mutation in Task 12.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Refusal §3.2 — NoResidualAdd

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — closure of 3 non-Add ops should produce `PruneRefusal::NoResidualAdd`.

Add to the `tests` module:

```rust
#[test]
fn no_residual_add_refusal() {
    // Closure: 3 Mul ops, NO Add anywhere. Non-residual architecture.
    let v_hb: VarId = 100;
    let v0:   VarId = 200;
    let v1:   VarId = 201;
    let v2:   VarId = 202;

    let ops = vec![
        op_unary(0, v0, v_hb, PrimalOp::Mul),
        op_unary(1, v1, v0, PrimalOp::Mul),
        op_unary(2, v2, v1, PrimalOp::Mul),
    ];
    let wengert = mk_wengert(ops, v2, &[(v_hb, "h_before"), (v0, "blocks.7.attn.wq")]);
    let layer = mk_prune_layer(7, "blocks.7.attn", LayerRole::Attention);

    match plan_rewrite(&wengert, &layer, &empty_weight_map()) {
        PlanResult::Refused(PruneRefusal::NoResidualAdd { layer_name, closure_size, .. }) => {
            assert_eq!(layer_name, "blocks.7.attn");
            assert_eq!(closure_size, 3, "all 3 Mul ops are in the closure (no Add to terminate at)");
        }
        other => panic!("expected NoResidualAdd refusal, got {:?}", std::mem::discriminant(&other_as_partial(other))),
    }
}

fn other_as_partial(r: PlanResult) -> PlanResult { r }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests::no_residual_add_refusal 2>&1 | tail -10
```

Expected: FAIL — `closure_size` in the emitted refusal is 0 (from the Task 4 stub), not 3.

- [ ] **Step 3: Fix the `closure_size` reporting**

In `plan_rewrite`, pass the computed `closure.len()` into `refusal_with_context`:

```rust
    // ... after computing closure_op_ids:
    let (residual_add_op_id, h_before_var, h_after_var) =
        match find_residual_add(wengert, &closure_op_ids, &param_var_ids) {
            Ok(triple) => triple,
            Err(partial) => {
                return PlanResult::Refused(refusal_with_context(
                    partial, layer, closure_op_ids.len(),
                ));
            }
        };
```

Update `refusal_with_context` signature:

```rust
fn refusal_with_context(
    partial: PartialRefusal,
    layer: &crate::wggo_apply::AppliedLayer,
    closure_size: usize,
) -> PruneRefusal {
    match partial {
        PartialRefusal::NoResidualAdd => PruneRefusal::NoResidualAdd {
            layer_name: layer.layer_name.clone(),
            layer_role: layer.layer_role,
            closure_size,
        },
        // Tasks 6, 7 extend this match.
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests::no_residual_add_refusal 2>&1 | tail -10
```

Expected: PASS. Run also the Task 4 test to confirm no regression:

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "feat(wggo-prune): refusal §3.2 — NoResidualAdd with closure_size

Spec §3.2. Non-residual architecture (SSM, Mamba, etc.) fails here at the
sub-block level; closure_size tells the user how big the closure was so
they can cross-check against their model topology.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Refusal §3.3 — ParallelResidualBranches

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — closure with two Adds against DIFFERENT `h_before` VarIds (parallel residuals) refuses.

```rust
#[test]
fn parallel_residuals_refusal() {
    // Two parallel residual paths sharing a common layer-N param:
    //   y1 = h_before_1 + (param * something)
    //   y2 = h_before_2 + (param * something)
    //
    // Both Adds pattern-match against different h_before values.

    let v_hb1: VarId = 100;
    let v_hb2: VarId = 110;
    let v_p:   VarId = 200; // shared layer-N param
    let v_b1:  VarId = 201;
    let v_b2:  VarId = 211;
    let v_y1:  VarId = 300;
    let v_y2:  VarId = 310;

    let ops = vec![
        op_unary(0, v_p, v_hb1, PrimalOp::Mul),  // param producer (dummy)
        op_unary(1, v_b1, v_p, PrimalOp::Mul),    // block_output branch 1
        op_add(2, v_y1, v_hb1, v_b1),             // residual branch 1
        op_unary(3, v_b2, v_p, PrimalOp::Mul),    // block_output branch 2 (reads same param)
        op_add(4, v_y2, v_hb2, v_b2),             // residual branch 2 (DISTINCT h_before)
    ];

    let wengert = mk_wengert(
        ops, v_y1,
        &[(v_hb1, "h_before_1"), (v_hb2, "h_before_2"), (v_p, "blocks.7.attn.wq")],
    );
    let layer = mk_prune_layer(7, "blocks.7.attn", LayerRole::Attention);

    match plan_rewrite(&wengert, &layer, &empty_weight_map()) {
        PlanResult::Refused(PruneRefusal::ParallelResidualBranches { layer_name, add_ops, .. }) => {
            assert_eq!(layer_name, "blocks.7.attn");
            assert_eq!(add_ops.len(), 2);
            assert!(add_ops.contains(&2) && add_ops.contains(&4));
        }
        other => panic!("expected ParallelResidualBranches, got {:?}", std::mem::discriminant(&other)),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `find_residual_add` returns `Ok(...)` for the first match and ignores the second.

- [ ] **Step 3: Extend `find_residual_add` to collect all candidates**

```rust
fn find_residual_add(
    wengert: &WengertList,
    closure: &[OpId],
    param_var_ids: &BTreeSet<VarId>,
) -> Result<(OpId, VarId, VarId), PartialRefusal> {
    let closure_set: BTreeSet<OpId> = closure.iter().copied().collect();
    let tainted: BTreeSet<VarId> = {
        let mut t: BTreeSet<VarId> = param_var_ids.clone();
        for op in &wengert.ops {
            if closure_set.contains(&op.id) {
                t.insert(op.result);
            }
        }
        t
    };

    // Collect ALL Adds that match the residual pattern.
    let mut candidates: Vec<(OpId, VarId, VarId)> = Vec::new(); // (add_op_id, h_before, h_after)
    for op in &wengert.ops {
        if closure_set.contains(&op.id) { continue; }
        if !matches!(op.op, PrimalOp::Add) { continue; }
        if op.inputs.len() != 2 { continue; }
        let (a, b) = (op.inputs[0], op.inputs[1]);
        let a_tainted = tainted.contains(&a);
        let b_tainted = tainted.contains(&b);
        if a_tainted != b_tainted {
            let h_before = if a_tainted { b } else { a };
            candidates.push((op.id, h_before, op.result));
        }
    }

    match candidates.len() {
        0 => Err(PartialRefusal::NoResidualAdd),
        1 => Ok(candidates[0]),
        _ => {
            // Multiple candidates: check if they all share the same h_before
            // (ambiguous pattern — Task 7) or have distinct h_before
            // (parallel branches — this task).
            let first_h_before = candidates[0].1;
            if candidates.iter().all(|(_, h, _)| *h == first_h_before) {
                Err(PartialRefusal::AmbiguousPatternMatch {
                    h_before: first_h_before,
                    candidate_adds: candidates.iter().map(|(op, _, _)| *op).collect(),
                })
            } else {
                Err(PartialRefusal::ParallelResidualBranches {
                    add_ops: candidates.iter().map(|(op, _, _)| *op).collect(),
                })
            }
        }
    }
}
```

Extend `PartialRefusal`:

```rust
#[derive(Debug)]
enum PartialRefusal {
    NoResidualAdd,
    ParallelResidualBranches { add_ops: Vec<OpId> },
    AmbiguousPatternMatch { h_before: VarId, candidate_adds: Vec<OpId> },
}
```

Extend `refusal_with_context`:

```rust
fn refusal_with_context(
    partial: PartialRefusal,
    layer: &crate::wggo_apply::AppliedLayer,
    closure_size: usize,
) -> PruneRefusal {
    match partial {
        PartialRefusal::NoResidualAdd => PruneRefusal::NoResidualAdd {
            layer_name: layer.layer_name.clone(),
            layer_role: layer.layer_role,
            closure_size,
        },
        PartialRefusal::ParallelResidualBranches { add_ops } => PruneRefusal::ParallelResidualBranches {
            layer_name: layer.layer_name.clone(),
            layer_role: layer.layer_role,
            add_ops,
        },
        PartialRefusal::AmbiguousPatternMatch { h_before, candidate_adds } => PruneRefusal::AmbiguousPatternMatch {
            layer_name: layer.layer_name.clone(),
            layer_role: layer.layer_role,
            h_before_var: h_before,
            candidate_adds,
        },
    }
}
```

- [ ] **Step 4: Run test to verify it passes (and no regression)**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests 2>&1 | tail -10
```

Expected: PASS for `parallel_residuals_refusal`, `no_residual_add_refusal`, and `closure_captures_transitive_compute_ops`.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "feat(wggo-prune): refusal §3.3 — ParallelResidualBranches

Detects ≥2 residual Adds with distinct h_before values. Extended
find_residual_add to collect all candidates; disambiguates vs §3.4
(shared h_before → ambiguous) in Task 7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Refusal §3.4 — AmbiguousPatternMatch

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — closure with two Adds both reading the SAME `h_before`. Already partially covered by Task 6's `find_residual_add` implementation; this test confirms the specific variant.

```rust
#[test]
fn ambiguous_pattern_match_refusal() {
    // Two Adds that both pattern-match against the same h_before.
    // E.g., an architecture with both pre-norm and post-norm residuals visible
    // at the layer boundary. Both match; we can't choose one.

    let v_hb: VarId = 100;
    let v_p:  VarId = 200;
    let v_b1: VarId = 201;
    let v_b2: VarId = 202;
    let v_y1: VarId = 300;
    let v_y2: VarId = 310;

    let ops = vec![
        op_unary(0, v_p, v_hb, PrimalOp::Mul),
        op_unary(1, v_b1, v_p, PrimalOp::Mul),
        op_unary(2, v_b2, v_p, PrimalOp::Mul),
        op_add(3, v_y1, v_hb, v_b1), // both Adds share h_before = v_hb
        op_add(4, v_y2, v_hb, v_b2),
    ];
    let wengert = mk_wengert(ops, v_y1, &[(v_hb, "h_before"), (v_p, "blocks.7.attn.wq")]);
    let layer = mk_prune_layer(7, "blocks.7.attn", LayerRole::Attention);

    match plan_rewrite(&wengert, &layer, &empty_weight_map()) {
        PlanResult::Refused(PruneRefusal::AmbiguousPatternMatch {
            layer_name, h_before_var, candidate_adds, ..
        }) => {
            assert_eq!(layer_name, "blocks.7.attn");
            assert_eq!(h_before_var, v_hb);
            assert_eq!(candidate_adds.len(), 2);
            assert!(candidate_adds.contains(&3) && candidate_adds.contains(&4));
        }
        other => panic!("expected AmbiguousPatternMatch, got {:?}", std::mem::discriminant(&other)),
    }
}
```

- [ ] **Step 2: Run test**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests::ambiguous_pattern_match_refusal 2>&1 | tail -10
```

Expected: PASS (Task 6's `find_residual_add` already dispatches to this case).

- [ ] **Step 3: No implementation changes needed — Task 6 covered this**

If the test happens to fail, the Task 6 disambiguation branch (`candidates.iter().all(|(_, h, _)| *h == first_h_before)`) is buggy. Inspect and fix.

- [ ] **Step 4: Run full test suite**

```bash
cargo test -p nsl-codegen --lib wggo_prune 2>&1 | tail -10
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit (only if this task added any code)**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "test(wggo-prune): refusal §3.4 — AmbiguousPatternMatch coverage

Confirms the §3.3-vs-§3.4 disambiguation in find_residual_add correctly
routes shared-h_before candidates to AmbiguousPatternMatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(If no code changed and only the test was added, rename commit type to `test(...)` as above.)

---

### Task 8: Refusal §3.5 — EmptyClosure

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — plan references a layer whose prefix matches no VarId.

```rust
#[test]
fn empty_closure_refusal() {
    // Wengert has parameters for blocks.0.* but user asks to prune blocks.99.attn.
    let v_hb: VarId = 100;
    let v_p:  VarId = 200;
    let v_y:  VarId = 300;

    let ops = vec![
        op_unary(0, v_p, v_hb, PrimalOp::Mul),
        op_add(1, v_y, v_hb, v_p),
    ];
    let wengert = mk_wengert(ops, v_y, &[(v_hb, "h_before"), (v_p, "blocks.0.attn.wq")]);
    let layer = mk_prune_layer(99, "blocks.99.attn", LayerRole::Attention);

    match plan_rewrite(&wengert, &layer, &empty_weight_map()) {
        PlanResult::Refused(PruneRefusal::EmptyClosure { layer_name, prefix, .. }) => {
            assert_eq!(layer_name, "blocks.99.attn");
            assert_eq!(prefix, "blocks.99.attn.");
        }
        other => panic!("expected EmptyClosure, got {:?}", std::mem::discriminant(&other)),
    }
}
```

- [ ] **Step 2: Run test**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests::empty_closure_refusal 2>&1 | tail -10
```

Expected: PASS — Task 4's `plan_rewrite` already handles the empty-params case. If it fails, the `if param_var_ids.is_empty()` branch was overlooked; add it back.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "test(wggo-prune): refusal §3.5 — EmptyClosure coverage

Confirms the empty-parameter-set early return in plan_rewrite correctly
emits EmptyClosure with the exact prefix string for user debugging.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Refusal §3.1 — CrossLayerParam

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — a layer-N parameter is consumed by an op outside the closure (cross-layer sharing).

```rust
#[test]
fn cross_layer_param_refusal() {
    // blocks.7.attn.wq is consumed both inside layer 7 AND by op elsewhere
    // (e.g., a tied-embedding op that reads the same weight from a different block).
    let v_hb: VarId = 100;
    let v_p:  VarId = 200; // blocks.7.attn.wq — shared
    let v_b:  VarId = 201;
    let v_y:  VarId = 300;
    let v_ext: VarId = 400; // consumed by external op

    let ops = vec![
        op_unary(0, v_p, v_hb, PrimalOp::Mul),
        op_unary(1, v_b, v_p, PrimalOp::Mul),       // in-layer consumer
        op_add(2, v_y, v_hb, v_b),                   // residual boundary
        // External consumer — some downstream op reads v_p directly.
        op_unary(3, v_ext, v_p, PrimalOp::Mul),     // OUTSIDE closure — triggers §3.1
    ];
    let wengert = mk_wengert(ops, v_ext, &[(v_hb, "h_before"), (v_p, "blocks.7.attn.wq")]);
    let layer = mk_prune_layer(7, "blocks.7.attn", LayerRole::Attention);

    match plan_rewrite(&wengert, &layer, &empty_weight_map()) {
        PlanResult::Refused(PruneRefusal::CrossLayerParam {
            layer_name, param_var, external_consumer, ..
        }) => {
            assert_eq!(layer_name, "blocks.7.attn");
            assert_eq!(param_var, v_p);
            assert_eq!(external_consumer, 3);
        }
        other => panic!("expected CrossLayerParam, got {:?}", std::mem::discriminant(&other)),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — current `plan_rewrite` doesn't check for external consumers; probably succeeds with wrong closure.

- [ ] **Step 3: Add cross-layer check after closure computation**

In `plan_rewrite`, immediately after computing `closure_op_ids`:

```rust
    // Spec §2.3 precondition #2 / §3.1: a layer-N parameter must not be
    // consumed by any op outside the closure.
    let closure_set: BTreeSet<OpId> = closure_op_ids.iter().copied().collect();
    for op in &wengert.ops {
        if closure_set.contains(&op.id) { continue; }
        for input in &op.inputs {
            if param_var_ids.contains(input) {
                let param_name = wengert.var_names.get(input).cloned().unwrap_or_default();
                return PlanResult::Refused(PruneRefusal::CrossLayerParam {
                    layer_name: layer.layer_name.clone(),
                    layer_role: layer.layer_role,
                    param_name,
                    param_var: *input,
                    external_consumer: op.id,
                    external_op_kind: format!("{:?}", op.op),
                });
            }
        }
    }
```

- [ ] **Step 4: Run test to verify passes + no regression**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests 2>&1 | tail -10
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "feat(wggo-prune): refusal §3.1 — CrossLayerParam

Spec §2.3 precondition #2: layer-N parameters consumed outside the closure
indicate cross-layer sharing (tied embeddings, shared ALiBi biases, etc.)
and break the closure's safe-to-delete invariant. Refuses with a pointer
to the specific external-consumer op for user debugging.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Refusal §3.6 — WholeBlockUnsupported via `plan_rewrite`

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — `LayerRole::Block` in plan triggers refusal.

```rust
#[test]
fn whole_block_refusal_from_planner() {
    // Layer with role=Block; plan_rewrite refuses immediately (spec §3.6).
    // This path is distinct from the legacy collect_prune_diagnostics one.

    let v_hb: VarId = 100;
    let v_p:  VarId = 200;
    let v_y:  VarId = 300;
    let ops = vec![
        op_unary(0, v_p, v_hb, PrimalOp::Mul),
        op_add(1, v_y, v_hb, v_p),
    ];
    let wengert = mk_wengert(ops, v_y, &[(v_hb, "h_before"), (v_p, "blocks.7.wq")]);
    let layer = mk_prune_layer(7, "blocks.7", LayerRole::Block);

    match plan_rewrite(&wengert, &layer, &empty_weight_map()) {
        PlanResult::Refused(PruneRefusal::WholeBlockUnsupported { layer_name }) => {
            assert_eq!(layer_name, "blocks.7");
        }
        other => panic!("expected WholeBlockUnsupported, got {:?}", std::mem::discriminant(&other)),
    }
}
```

- [ ] **Step 2: Run test**

Expected: PASS — Task 4's `plan_rewrite` already handles this at the very top. Test is a regression guard.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "test(wggo-prune): refusal §3.6 — WholeBlockUnsupported coverage

Regression guard for the early LayerRole::Block check in plan_rewrite.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Refusal §3.7 — ConflictingPruneDecisions + three-phase `run()` wiring

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — multiple layers in one plan; dry-run-then-commit contract; conflicts detected.

```rust
#[test]
fn conflicting_decisions_refusal() {
    // Two prune decisions whose closures overlap on the same parameter VarId.
    // Constructed artificially — in practice this is extremely rare under v1's
    // sub-block-only scope — but the variant exists for future-robustness.
    //
    // Test: two layers both claim the same param VarId in their prefixes.
    // The aggregator in `run()` catches the overlap and emits
    // ConflictingPruneDecisions.

    let v_hb: VarId = 100;
    let v_p:  VarId = 200;
    let v_b:  VarId = 201;
    let v_y:  VarId = 300;

    let ops = vec![
        op_unary(0, v_p, v_hb, PrimalOp::Mul),
        op_unary(1, v_b, v_p, PrimalOp::Mul),
        op_add(2, v_y, v_hb, v_b),
    ];
    // Both layer_names have prefixes that collide on v_p:
    //   "blocks.7"  prefix → "blocks.7."
    //   "blocks.7.attn" prefix → "blocks.7.attn."
    // Ambiguous: v_p's name starts with both.
    let mut wengert = mk_wengert(ops, v_y, &[(v_hb, "h_before"), (v_p, "blocks.7.attn.wq")]);
    let layers = vec![
        mk_prune_layer(7, "blocks.7.attn", LayerRole::Attention),
        // Artificial second decision with overlapping prefix: make both
        // valid matches by naming the param so both prefixes apply.
        mk_prune_layer(70, "blocks.7.attn.wq", LayerRole::Attention),
    ];
    let plan = mk_applied_plan(layers);

    let result = run(&mut wengert, &plan, &empty_weight_map());

    // Expect at least one ConflictingPruneDecisions refusal.
    let conflict = result.refusals.iter().find(|r| matches!(r, PruneRefusal::ConflictingPruneDecisions { .. }));
    assert!(conflict.is_some(), "expected a ConflictingPruneDecisions refusal; got refusals: {:?}",
        result.refusals.iter().map(|r| std::mem::discriminant(r)).collect::<Vec<_>>());

    // Dry-run invariant: wengert should be UNCHANGED (still has all 3 ops).
    assert_eq!(wengert.ops.len(), 3, "wengert should be untouched on refusal (spec §5.3 Phase 2)");
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `run()` is the Task 1 stub that returns empty result.

- [ ] **Step 3: Implement the three-phase `run()`**

Replace the stub in `wggo_prune.rs`:

```rust
pub fn run(
    wengert: &mut WengertList,
    applied_plan: &AppliedPlan,
    weight_map: &WeightMap,
) -> PruneRewriteResult {
    use crate::wggo_dp::LayerDecision;

    // Phase 1: validate each Prune decision without mutating.
    let mut plans: Vec<PruneRewritePlan> = Vec::new();
    let mut refusals: Vec<PruneRefusal> = Vec::new();
    for layer in &applied_plan.layers {
        if !matches!(layer.coarse, LayerDecision::Prune) { continue; }
        match plan_rewrite(wengert, layer, weight_map) {
            PlanResult::Ok(plan) => plans.push(plan),
            PlanResult::Refused(r) => refusals.push(r),
        }
    }

    // Phase 1b: cross-plan conflict detection. Two plans conflict if they
    // claim overlapping VarIds (parameter or closure output), or if their
    // residual Adds share output VarIds, etc. For v1's sub-block scope, the
    // likely conflict is one plan's parameter VarIds appearing in another's
    // closure.
    if refusals.is_empty() {
        for i in 0..plans.len() {
            for j in (i + 1)..plans.len() {
                let a = &plans[i];
                let b = &plans[j];
                let a_ops: BTreeSet<OpId> = a.closure_op_ids.iter().copied().collect();
                let b_ops: BTreeSet<OpId> = b.closure_op_ids.iter().copied().collect();
                let overlap: Vec<OpId> = a_ops.intersection(&b_ops).copied().collect();
                if !overlap.is_empty() {
                    refusals.push(PruneRefusal::ConflictingPruneDecisions {
                        decision_a: a.layer_name.clone(),
                        decision_b: b.layer_name.clone(),
                        reason: format!(
                            "closures overlap on ops: {:?} (same ops would be deleted by both rewrites)",
                            overlap
                        ),
                    });
                    break;
                }
                if a.h_after_var == b.h_after_var {
                    refusals.push(PruneRefusal::ConflictingPruneDecisions {
                        decision_a: a.layer_name.clone(),
                        decision_b: b.layer_name.clone(),
                        reason: format!(
                            "both rewrites target the same h_after VarId {:?}; aliasing is undefined",
                            a.h_after_var
                        ),
                    });
                    break;
                }
            }
        }
    }

    // Phase 2: early-return on any refusal.
    if !refusals.is_empty() {
        return PruneRewriteResult {
            rewrites: Vec::new(),
            refusals,
            pruned_forward_var_ids: BTreeSet::new(),
            ops_deleted: 0,
        };
    }

    // Phase 3: commit every plan.
    let mut rewrites: Vec<PruneRewrite> = Vec::with_capacity(plans.len());
    let mut pruned_forward_var_ids: BTreeSet<VarId> = BTreeSet::new();
    let mut ops_deleted: usize = 0;
    for plan in plans {
        let rewrite = apply_rewrite(wengert, plan);
        // Track pruned VarIds: every closure op's `result` VarId is now absent
        // from the reduced forward.
        for op_id in &rewrite.closure_ops {
            if let Some(op) = wengert.ops.iter().find(|o| o.id == *op_id) {
                pruned_forward_var_ids.insert(op.result);
            }
        }
        pruned_forward_var_ids.insert(rewrite.h_after_var);
        ops_deleted += rewrite.closure_ops.len() + 1; // +1 for the residual Add
        rewrites.push(rewrite);
    }

    // Finalize: actually remove the collected ops from wengert.
    // apply_rewrite marks them for removal; the actual deletion happens here
    // to keep op IDs stable during the rewrite phase.
    // Task 12 implements apply_rewrite + the finalize pass together.

    PruneRewriteResult {
        rewrites,
        refusals: Vec::new(),
        pruned_forward_var_ids,
        ops_deleted,
    }
}

// Stub — Task 12 implements Phase 3 mutation.
fn apply_rewrite(
    _wengert: &mut WengertList,
    plan: PruneRewritePlan,
) -> PruneRewrite {
    PruneRewrite {
        layer_name: plan.layer_name,
        layer_role: plan.layer_role,
        h_before_var: plan.h_before_var,
        h_after_var: plan.h_after_var,
        residual_add_op: plan.residual_add_op_id,
        closure_ops: plan.closure_op_ids,
    }
}
```

- [ ] **Step 4: Run test**

```bash
cargo test -p nsl-codegen --lib wggo_prune::tests 2>&1 | tail -10
```

Expected: all Phase B tests PASS. The conflict test specifically verifies both the refusal AND the "wengert untouched" invariant.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "feat(wggo-prune): Phase 1/2/3 orchestration + §3.7 ConflictingDecisions

run() now implements the three-phase contract (spec §5.3): validate all
decisions, detect cross-plan conflicts, early-return on refusal with
wengert unchanged, else apply all rewrites. Phase 3 mutation (apply_rewrite)
is still stubbed; Task 12 implements it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**End of Phase B.** All seven refusal variants detected; all Layer 2 unit tests green; `wengert` untouched on any refusal (verified by the conflict test).

---

# Phase C — Mutation + integration tests (Commit C)

### Task 12: Phase 3 mutation — op deletion, VarId repointing, Add removal

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs`

- [ ] **Step 1: Write the failing test** — after `run()` on a valid plan, the wengert list no longer contains the closure ops, and `h_after`'s original producer (the Add) is gone; consumers of `h_after` now reference `h_before`.

```rust
#[test]
fn apply_rewrite_deletes_closure_and_aliases_h_after() {
    // Wengert:
    //   op0: mul v0 = v_hb * ?            (param producer)
    //   op1: mul v1 = v_hb * v0           (block_output)
    //   op2: add v_ha = v_hb + v1         (residual Add)
    //   op3: mul v_out = v_ha * v_ha      (downstream consumer of h_after)
    //
    // After prune of blocks.7.attn:
    //   op3 remains, but its input changes from v_ha → v_hb.
    //   ops 0, 1, 2 are deleted.

    let v_hb: VarId = 100;
    let v0:   VarId = 200;
    let v1:   VarId = 201;
    let v_ha: VarId = 202;
    let v_out:VarId = 300;

    let ops = vec![
        op_unary(0, v0, v_hb, PrimalOp::Mul),
        op_unary(1, v1, v0, PrimalOp::Mul),
        op_add(2, v_ha, v_hb, v1),
        op_unary(3, v_out, v_ha, PrimalOp::Mul), // downstream consumer
    ];
    let mut wengert = mk_wengert(ops, v_out,
        &[(v_hb, "h_before"), (v0, "blocks.7.attn.wq"), (v_ha, "h_after")]);
    let plan = mk_applied_plan(vec![
        mk_prune_layer(7, "blocks.7.attn", LayerRole::Attention),
    ]);

    let result = run(&mut wengert, &plan, &empty_weight_map());

    // Zero refusals.
    assert!(result.refusals.is_empty(), "expected no refusals, got: {:?}",
        result.refusals.iter().map(|r| std::mem::discriminant(r)).collect::<Vec<_>>());
    assert_eq!(result.rewrites.len(), 1);
    assert_eq!(result.ops_deleted, 3, "closure=2 + residual Add=1");

    // Wengert has exactly one op left: the downstream consumer, but its input
    // should now be v_hb, not v_ha.
    assert_eq!(wengert.ops.len(), 1, "only op3 survives");
    let surviving = &wengert.ops[0];
    assert_eq!(surviving.id, 3);
    assert_eq!(surviving.inputs, vec![v_hb], "downstream consumer must be aliased to h_before");

    // pruned_forward_var_ids should include v0, v1, v_ha (everything the rewrite removed).
    assert!(result.pruned_forward_var_ids.contains(&v0));
    assert!(result.pruned_forward_var_ids.contains(&v1));
    assert!(result.pruned_forward_var_ids.contains(&v_ha));
}
```

- [ ] **Step 2: Run test to verify it fails**

Expected: FAIL — `apply_rewrite` in Task 11 is a stub that doesn't mutate.

- [ ] **Step 3: Implement Phase 3 mutation**

Replace the `apply_rewrite` stub in `wggo_prune.rs`:

```rust
fn apply_rewrite(
    wengert: &mut WengertList,
    plan: PruneRewritePlan,
) -> PruneRewrite {
    // Collect the set of OpIds we're about to delete.
    let mut to_delete: BTreeSet<OpId> = plan.closure_op_ids.iter().copied().collect();
    to_delete.insert(plan.residual_add_op_id);

    // Repoint every surviving op's inputs from h_after_var to h_before_var.
    for op in wengert.ops.iter_mut() {
        if to_delete.contains(&op.id) { continue; }
        for input in op.inputs.iter_mut() {
            if *input == plan.h_after_var {
                *input = plan.h_before_var;
            }
        }
    }
    // Repoint `wengert.output` too, if it pointed at h_after.
    if wengert.output == plan.h_after_var {
        wengert.output = plan.h_before_var;
    }

    // Remove the closure ops + residual Add from wengert.ops.
    wengert.ops.retain(|op| !to_delete.contains(&op.id));

    // Remove pruned VarIds from var_names / var_types for cleanliness.
    for op_id in &plan.closure_op_ids {
        // The op may be gone, but its result VarId entry might linger.
        // We don't have a quick lookup from op_id → VarId after delete, so
        // instead iterate over var_names and drop VarIds that no op produces.
        // (Alternative: capture the result VarIds in `plan` before delete.)
    }
    // Simpler: rebuild var_names keeping only VarIds still produced.
    let surviving_var_ids: BTreeSet<VarId> = wengert.ops.iter().map(|o| o.result).collect();
    wengert.var_names.retain(|v, _| surviving_var_ids.contains(v) || *v == wengert.output);
    wengert.var_types.retain(|v, _| surviving_var_ids.contains(v) || *v == wengert.output);
    // Note: h_before_var survives because it's produced by some op outside
    // the closure (the prior layer's residual Add output, or an initial input).

    PruneRewrite {
        layer_name: plan.layer_name,
        layer_role: plan.layer_role,
        h_before_var: plan.h_before_var,
        h_after_var: plan.h_after_var,
        residual_add_op: plan.residual_add_op_id,
        closure_ops: plan.closure_op_ids,
    }
}
```

Also update `run()`'s pruned-VarId tracking: it was incorrectly looking up ops in `wengert.ops` AFTER they'd been deleted. Move the lookup BEFORE `apply_rewrite`:

```rust
    for plan in plans {
        // Capture pruned VarIds BEFORE the mutation.
        let mut removed_vars: Vec<VarId> = wengert.ops.iter()
            .filter(|o| plan.closure_op_ids.contains(&o.id))
            .map(|o| o.result)
            .collect();
        removed_vars.push(plan.h_after_var);
        for v in &removed_vars {
            pruned_forward_var_ids.insert(*v);
        }

        let rewrite = apply_rewrite(wengert, plan);
        ops_deleted += rewrite.closure_ops.len() + 1; // +1 for the residual Add
        rewrites.push(rewrite);
    }
```

- [ ] **Step 4: Run test to verify passes + no regression**

```bash
cargo test -p nsl-codegen --lib wggo_prune 2>&1 | tail -10
```

Expected: all Phase B + Phase C test PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "feat(wggo-prune): Phase 3 mutation — delete + repoint + Add removal

apply_rewrite now (1) repoints every surviving op's consumers of h_after
to h_before, (2) repoints wengert.output if needed, (3) removes all
closure ops + the residual Add via wengert.ops.retain, (4) prunes stale
var_names/var_types entries. Pruned VarIds captured before deletion and
returned via pruned_forward_var_ids for the WRGA / source-AD handoff.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: 4-block test fixture + reference fixtures

**Files:**

- Create: `crates/nsl-codegen/tests/fixtures/prune_rewrite_toy.nsl`
- Create: 8 × `crates/nsl-codegen/tests/fixtures/prune_rewrite_toy_ref_*.nsl`

- [ ] **Step 1: Write the baseline fixture**

Create `crates/nsl-codegen/tests/fixtures/prune_rewrite_toy.nsl`:

```nsl
# Minimal 4-block pre-norm transformer for WGGO Prune IR-rewrite testing.
# Dimensions are tiny (d_model=32, d_ffn=64, 4 heads × head_dim=8) to keep
# CI fast; tests exercise structural rewriting, not numerical accuracy at scale.

from nsl.nn import rmsnorm, softmax

model TinyBlock:
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])
    wo: Tensor = ones([32, 32])
    w_ffn_up: Tensor = ones([32, 64])
    w_ffn_down: Tensor = ones([64, 32])
    norm_attn_g: Tensor = ones([32])
    norm_ffn_g: Tensor = ones([32])

    fn attn(self, h: Tensor) -> Tensor:
        let x = rmsnorm(h, self.norm_attn_g)
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        let a = softmax(q @ k.T)
        return (a @ v) @ self.wo

    fn ffn(self, h: Tensor) -> Tensor:
        let x = rmsnorm(h, self.norm_ffn_g)
        let u = x @ self.w_ffn_up
        return u @ self.w_ffn_down

    fn forward(self, h: Tensor) -> Tensor:
        let h1 = h + self.attn(h)        # attn residual
        let h2 = h1 + self.ffn(h1)        # ffn residual
        return h2

model TinyTransformer:
    blocks_0: TinyBlock = TinyBlock()
    blocks_1: TinyBlock = TinyBlock()
    blocks_2: TinyBlock = TinyBlock()
    blocks_3: TinyBlock = TinyBlock()

    fn forward(self, h: Tensor) -> Tensor:
        let h = self.blocks_0.forward(h)
        let h = self.blocks_1.forward(h)
        let h = self.blocks_2.forward(h)
        let h = self.blocks_3.forward(h)
        return h

let m = TinyTransformer()
let x = ones([1, 16, 32])
let y = m.forward(x)
```

**Implementer note on layer naming:** the WGGO planner emits `layer_name` values like `blocks.7.attn`. If NSL's codegen normalizes `blocks_0.attn` → `blocks.0.attn` in the Wengert `var_names` map, use that form. If not, adjust the fixture's field names to `blocks.0`, `blocks.1`, etc. (syntactically may require different NSL conventions). Verify by compiling the fixture and inspecting `wengert.var_names` for an expected parameter name. If the fixture's naming doesn't match the planner's output, the test's plan needs to use whatever `layer_name` form the extractor actually produces.

- [ ] **Step 2: Write reference fixtures**

For each planned Layer 3 test case, create a hand-written reference where the pruned sub-block is replaced by literal identity. Example for `blocks.1.attn` pruned:

Create `crates/nsl-codegen/tests/fixtures/prune_rewrite_toy_ref_blocks_1_attn.nsl`:

```nsl
# Reference for: prune blocks.1.attn in prune_rewrite_toy.nsl
# blocks.1's forward becomes: h1 = h (identity instead of h + attn(h))

from nsl.nn import rmsnorm, softmax

model TinyBlock:
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])
    wo: Tensor = ones([32, 32])
    w_ffn_up: Tensor = ones([32, 64])
    w_ffn_down: Tensor = ones([64, 32])
    norm_attn_g: Tensor = ones([32])
    norm_ffn_g: Tensor = ones([32])

    fn attn(self, h: Tensor) -> Tensor:
        let x = rmsnorm(h, self.norm_attn_g)
        let q = x @ self.wq
        let k = x @ self.wk
        let v = x @ self.wv
        let a = softmax(q @ k.T)
        return (a @ v) @ self.wo

    fn ffn(self, h: Tensor) -> Tensor:
        let x = rmsnorm(h, self.norm_ffn_g)
        let u = x @ self.w_ffn_up
        return u @ self.w_ffn_down

    fn forward_attn_pruned(self, h: Tensor) -> Tensor:
        # attn sub-block replaced by identity
        let h1 = h
        let h2 = h1 + self.ffn(h1)
        return h2

    fn forward(self, h: Tensor) -> Tensor:
        let h1 = h + self.attn(h)
        let h2 = h1 + self.ffn(h1)
        return h2

model TinyTransformer:
    blocks_0: TinyBlock = TinyBlock()
    blocks_1: TinyBlock = TinyBlock()
    blocks_2: TinyBlock = TinyBlock()
    blocks_3: TinyBlock = TinyBlock()

    fn forward(self, h: Tensor) -> Tensor:
        let h = self.blocks_0.forward(h)
        let h = self.blocks_1.forward_attn_pruned(h)
        let h = self.blocks_2.forward(h)
        let h = self.blocks_3.forward(h)
        return h

let m = TinyTransformer()
let x = ones([1, 16, 32])
let y = m.forward(x)
```

Create similarly-structured reference files for the 7 other test cases listed in spec §7.1:

| File | Pruned target |
|---|---|
| `prune_rewrite_toy_ref_blocks_0_attn.nsl` | `blocks.0.attn` |
| `prune_rewrite_toy_ref_blocks_0_ffn.nsl` | `blocks.0.ffn` |
| `prune_rewrite_toy_ref_blocks_1_attn.nsl` | `blocks.1.attn` (shown above) |
| `prune_rewrite_toy_ref_blocks_1_ffn.nsl` | `blocks.1.ffn` |
| `prune_rewrite_toy_ref_blocks_3_attn.nsl` | `blocks.3.attn` |
| `prune_rewrite_toy_ref_blocks_3_ffn.nsl` | `blocks.3.ffn` |
| `prune_rewrite_toy_ref_blocks_1_attn_and_ffn.nsl` | `blocks.1.attn` + `blocks.1.ffn` |
| `prune_rewrite_toy_ref_blocks_1_ffn_and_2_attn.nsl` | `blocks.1.ffn` + `blocks.2.attn` |

Each reference should swap the corresponding sub-block(s) for identity in the appropriate `blocks_N.forward` call.

- [ ] **Step 3: Verify the fixture compiles through NSL**

```bash
cargo build -p nsl-codegen --release 2>&1 | tail -3
# If there's a compiler invocation for NSL source, invoke it on the fixture
# to verify syntax. Look for existing test helpers like compile_training_to_object
# in crates/nsl-codegen/tests/csha_gap_f_toy_pretrain_smoke.rs:94-125.
```

Expected: no compile errors on the fixtures themselves.

- [ ] **Step 4: Commit**

```bash
git add -f crates/nsl-codegen/tests/fixtures/prune_rewrite_toy*.nsl
git commit -m "test(wggo-prune): 4-block toy fixture + 8 reference variants

Spec §7.1. Minimal pre-norm transformer (d_model=32, d_ffn=64, 4h × 8d,
vocab=64, seq=16, bs=1). Reference variants hand-code the pruned
sub-block as literal identity for bit-exact equivalence checks (§7.5 B1).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Layer 3 integration tests (bit-exact numerical equivalence + strong adjoint)

**Files:**

- Create: `crates/nsl-codegen/tests/wggo_prune_rewrite.rs`
- Create: `crates/nsl-codegen/tests/snapshots/wggo_prune_rewrite__*.snap` (auto-generated)

- [ ] **Step 1: Write the failing tests** — bit-exact forward + adjoint-reduction for each test case.

Create `crates/nsl-codegen/tests/wggo_prune_rewrite.rs`:

```rust
// Spec §7 test suite for WGGO Prune v1.
//
// Layers exercised here:
//   - Layer 1: IR skeleton snapshots (via insta) — stable pretty-print
//   - Layer 3: numerical equivalence (bit-exact) + strong adjoint assertion
//   - Layer 4: stderr snapshots + OverrideRejectReason assertions
//
// Layer 2 (closure + pattern-match unit tests) lives inside wggo_prune.rs.

use nsl_codegen::wggo_prune::{run, PruneRewriteResult, PruneRefusal};
use nsl_codegen::wggo_apply::AppliedPlan;
// ... additional imports based on the test-helper pattern from existing tests
// (see csha_gap_f_toy_pretrain_smoke.rs:94-125 for the compile harness shape).

/// Compile an .nsl source file and return (wengert, applied_plan, weight_map).
fn compile_fixture(path: &str) -> CompilationArtifacts {
    let src = std::fs::read_to_string(path).expect("fixture read");
    // Mirror the NSL pipeline from csha_gap_f_toy_pretrain_smoke.rs:
    // tokenize → parse → semantic → wggo-apply → extract-wengert.
    todo!("wire via existing helpers; see csha_gap_f_toy_pretrain_smoke.rs:94")
}

struct CompilationArtifacts {
    wengert: nsl_codegen::wengert::WengertList,
    applied_plan: AppliedPlan,
    weight_map: nsl_codegen::weight_aware::WeightMap,
}

/// Run forward from a wengert list at a fixed seed and return the output bytes.
fn run_forward(wengert: &nsl_codegen::wengert::WengertList, weight_map: &nsl_codegen::weight_aware::WeightMap) -> Vec<u8> {
    // Use the existing interpreter / Cranelift-compiled runner. If no such
    // harness exists at the nsl-codegen test layer, use the same path that
    // csha_gap_f_toy_pretrain_smoke.rs uses to evaluate forward output, then
    // extract the final tensor's bytes.
    todo!("wire via existing forward-runner harness")
}

#[test]
fn blocks_1_attn_forward_bit_exact_vs_identity_reference() {
    let mut art = compile_fixture("tests/fixtures/prune_rewrite_toy.nsl");
    let ref_art = compile_fixture("tests/fixtures/prune_rewrite_toy_ref_blocks_1_attn.nsl");

    // Synthesize a plan with Prune on blocks.1.attn and KeepFull everywhere else.
    // Start from art.applied_plan, override the blocks.1.attn entry.
    let plan = override_plan_with_prune(&art.applied_plan, "blocks.1.attn");

    let result = run(&mut art.wengert, &plan, &art.weight_map);

    assert!(result.refusals.is_empty(), "expected no refusals; got {:?}",
        result.refusals.iter().map(|r| std::mem::discriminant(r)).collect::<Vec<_>>());
    assert_eq!(result.rewrites.len(), 1);

    let out_pruned = run_forward(&art.wengert, &art.weight_map);
    let out_ref    = run_forward(&ref_art.wengert, &ref_art.weight_map);

    assert_eq!(out_pruned, out_ref, "bit-exact equivalence: pruned rewrite vs identity reference");
}

fn override_plan_with_prune(base: &AppliedPlan, target: &str) -> AppliedPlan {
    use nsl_codegen::wggo_dp::LayerDecision;
    let mut plan = base.clone();
    for layer in &mut plan.layers {
        if layer.layer_name == target {
            layer.coarse = LayerDecision::Prune;
        }
    }
    plan
}

/// Spec §7.5 strong adjoint assertion: after prune, pruned VarIds do not
/// appear in the backward adjoint tape as producers or consumers, and the
/// backward op count is strictly smaller than the no-prune baseline.
#[test]
fn blocks_1_attn_adjoint_reduced_and_pruned_vars_absent() {
    let mut art = compile_fixture("tests/fixtures/prune_rewrite_toy.nsl");
    let plan_no_prune = art.applied_plan.clone();
    let plan_prune    = override_plan_with_prune(&art.applied_plan, "blocks.1.attn");

    // Baseline adjoint (no prune):
    let mut wengert_baseline = art.wengert.clone();
    let _ = run(&mut wengert_baseline, &plan_no_prune, &art.weight_map);
    let adjoint_baseline = compile_adjoint(&wengert_baseline);

    // Pruned adjoint:
    let mut wengert_pruned = art.wengert.clone();
    let r = run(&mut wengert_pruned, &plan_prune, &art.weight_map);
    let adjoint_pruned = compile_adjoint(&wengert_pruned);

    assert!(adjoint_pruned.ops.len() < adjoint_baseline.ops.len(),
        "adjoint op count should strictly decrease after prune: baseline={} pruned={}",
        adjoint_baseline.ops.len(), adjoint_pruned.ops.len());

    for var_id in &r.pruned_forward_var_ids {
        let referenced = adjoint_pruned.ops.iter().any(|op|
            op.inputs.contains(var_id) || op.result == *var_id);
        assert!(!referenced, "pruned VarId {:?} still referenced in backward tape", var_id);
    }
}

fn compile_adjoint(wengert: &nsl_codegen::wengert::WengertList) -> nsl_codegen::wengert::WengertList {
    // Invoke the source-AD path. Use whichever existing helper the
    // csha/wrga tests use to produce an adjoint Wengert list.
    todo!("wire via existing source-AD helper")
}

/// Layer 4 text snapshot + structural OverrideRejectReason assertion.
#[test]
fn refusal_empty_closure_three_part_error() {
    let mut art = compile_fixture("tests/fixtures/prune_rewrite_toy.nsl");
    let plan = override_plan_with_prune(&art.applied_plan, "blocks.99.attn"); // bogus layer

    let result = run(&mut art.wengert, &plan, &art.weight_map);

    assert_eq!(result.refusals.len(), 1);
    assert!(matches!(result.refusals[0], PruneRefusal::EmptyClosure { .. }));

    // Text snapshot — format is fixed by spec §3.5. Task 15 wires the emitter
    // into stmt.rs; this test uses a local format_refusal helper that must
    // produce the same text.
    let rendered = nsl_codegen::wggo_prune::format_refusal(&result.refusals[0]);
    insta::assert_snapshot!("refusal_empty_closure", rendered);
}

// Similar tests for every other refusal case reachable from the supported
// fixture per spec §7.5 scope clause:
//   - refusal_whole_block_unsupported (by passing LayerRole::Block to a planner override)
```

**Layer 3 tests for remaining cases** (repeat the `*_forward_bit_exact_vs_identity_reference` pattern for each of the 8 reference fixtures):

```rust
#[test] fn blocks_0_attn_forward_bit_exact_vs_identity_reference() { /* ... */ }
#[test] fn blocks_0_ffn_forward_bit_exact_vs_identity_reference() { /* ... */ }
#[test] fn blocks_1_ffn_forward_bit_exact_vs_identity_reference() { /* ... */ }
#[test] fn blocks_3_attn_forward_bit_exact_vs_identity_reference() { /* ... */ }
#[test] fn blocks_3_ffn_forward_bit_exact_vs_identity_reference() { /* ... */ }
#[test] fn blocks_1_attn_and_ffn_forward_bit_exact_vs_identity_reference() { /* ... */ }
#[test] fn blocks_1_ffn_and_2_attn_forward_bit_exact_vs_identity_reference() { /* ... */ }
```

**Layer 1 snapshot tests** (spec §7.3) — after `run()`, format the Wengert list with the spec §7.3 stable pretty-print and snapshot:

```rust
#[test] fn layer1_snapshot_blocks_1_attn_pruned() {
    let mut art = compile_fixture("tests/fixtures/prune_rewrite_toy.nsl");
    let plan = override_plan_with_prune(&art.applied_plan, "blocks.1.attn");
    let result = run(&mut art.wengert, &plan, &art.weight_map);
    assert!(result.refusals.is_empty());

    let rendered = nsl_codegen::wggo_prune::pretty_print_stable(&art.wengert, &result);
    insta::assert_snapshot!("layer1_blocks_1_attn_pruned", rendered);
}
// ... one per (block_position × sub_block_role × before/after) cell, totaling ~20.
```

**Implementer note on `format_refusal` and `pretty_print_stable`:** these are new helpers in `wggo_prune.rs`. Task 15 implements the stderr-oriented `format_refusal`; spec §7.3 pins `pretty_print_stable` as symbolic-names + topologically-ordered placeholders, NOT global VarId numbers. If the implementer skips the VarId-stabilization and uses raw VarIds, Layer 1 snapshots will churn on every refactor. Re-read spec §7.3 carefully before implementing.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p nsl-codegen --test wggo_prune_rewrite 2>&1 | tail -20
```

Expected: tests fail with `todo!()` panics (the `compile_fixture`, `run_forward`, `compile_adjoint` helpers are unimplemented stubs at this point). That's fine; the next step implements them.

- [ ] **Step 3: Implement the test harness helpers**

Three helpers to implement:

1. `compile_fixture(path)` — mirror `csha_gap_f_toy_pretrain_smoke.rs:94-125`. Exact steps: read file → tokenize → parse → semantic → invoke wggo_apply to produce an AppliedPlan → invoke the Wengert extractor → return artifacts.
2. `run_forward(wengert, weight_map)` — use the existing CPU path used by other integration tests (the NSL interpreter or Cranelift-compiled runner). Return the output bytes.
3. `compile_adjoint(wengert)` — use the existing source-AD entry point. Look for `source_ad::run` or equivalent in `source_ad.rs`.

Also implement `wggo_prune::pretty_print_stable` and `wggo_prune::format_refusal` in `wggo_prune.rs` per spec §7.3 and §6.2 respectively. (`format_refusal` implementation lands more completely in Task 15; for Task 14 it can return a minimal-but-stable string with the three-part structure.)

Minimal `pretty_print_stable`:

```rust
/// Spec §7.3: VarId-number-stable pretty-print for Layer 1 snapshots.
/// Uses symbolic names where available; replaces anonymous VarIds with
/// topologically-ordered placeholders (%t1, %t2, ...).
pub fn pretty_print_stable(wengert: &WengertList, result: &PruneRewriteResult) -> String {
    let mut out = String::new();
    let mut placeholder_map: std::collections::HashMap<VarId, String> = std::collections::HashMap::new();
    let mut next_placeholder = 1usize;
    let mut name_var = |v: VarId| -> String {
        if let Some(n) = wengert.var_names.get(&v) {
            format!("%param:{}", n)
        } else {
            placeholder_map.entry(v).or_insert_with(|| {
                let name = format!("%t{}", next_placeholder);
                next_placeholder += 1;
                name
            }).clone()
        }
    };
    // Print wengert ops first.
    for op in &wengert.ops {
        let result_name = name_var(op.result);
        let input_names: Vec<String> = op.inputs.iter().map(|v| name_var(*v)).collect();
        out.push_str(&format!("{} = {:?}({})\n", result_name, op.op, input_names.join(", ")));
    }
    out.push_str(&format!("output = {}\n", name_var(wengert.output)));
    out.push_str("\n--- PruneRewriteResult ---\n");
    for rw in &result.rewrites {
        out.push_str(&format!("prune: layer={} role={:?} closure_ops={}\n",
            rw.layer_name, rw.layer_role, rw.closure_ops.len()));
    }
    out.push_str(&format!("ops_deleted={}\n", result.ops_deleted));
    out
}
```

**Note on borrow-checker:** the closure `name_var` borrows `placeholder_map` mutably and `wengert.var_names` immutably; the snippet above may need restructuring into a loop that threads explicit mutable state rather than a closure. If the borrow checker complains, inline `name_var` into each call site with `&mut placeholder_map, &mut next_placeholder` arguments.

Minimal `format_refusal`:

```rust
/// Spec §3 three-part refusal format. Task 15 hardens this; v1 produces
/// the minimal structured form.
pub fn format_refusal(r: &PruneRefusal) -> String {
    match r {
        PruneRefusal::EmptyClosure { layer_name, layer_role, prefix } => format!(
            "prune: no parameters match the requested layer prefix.\n  \
             requested:  prune {} (role={:?})\n  \
             expected:   at least one parameter VarId with var_name starting with `{}`\n  \
             found:      zero matching parameters in the WeightMap.",
            layer_name, layer_role, prefix
        ),
        // ... other variants — full implementation in Task 15.
        _ => format!("prune: refusal (TASK 15: implement full formatting) — {:?}", std::mem::discriminant(r)),
    }
}
```

- [ ] **Step 4: Run tests to verify they pass (or surface real bugs)**

```bash
cargo test -p nsl-codegen --test wggo_prune_rewrite 2>&1 | tail -30
```

Expected: Layer 3 tests PASS bit-exactly; Layer 1 snapshot tests generate `.snap.new` files (run `cargo insta review` or `cargo insta accept` to promote them). If ANY Layer 3 test fails bit-exactly, it's a real bug in Phase 3 mutation (Task 12) — diagnose by comparing the pruned Wengert to the reference's Wengert.

```bash
cargo insta accept --workspace
```

- [ ] **Step 5: Commit**

```bash
git add -f crates/nsl-codegen/tests/wggo_prune_rewrite.rs crates/nsl-codegen/tests/snapshots/wggo_prune_rewrite__*.snap
git add crates/nsl-codegen/src/wggo_prune.rs
git commit -m "test(wggo-prune): Layer 1 snapshots + Layer 3 bit-exact equivalence + strong adjoint

Spec §7. Eight Layer 3 integration tests exercise every (position × sub-block)
combination plus the two multi-prune cases; each compares bit-exactly against
a hand-written identity reference. Layer 1 snapshots capture the post-rewrite
Wengert in the §7.3 VarId-number-stable format. Strong adjoint assertion
verifies pruned VarIds are absent from the backward tape AND that backward op
count strictly decreases (§7.5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**End of Phase C.** Numerically-correct prune rewrite lands; Layer 1 and Layer 3 tests green. Diagnostics (success stderr + structured DiagnosticCode + refusal text snapshots) remain for Phase D.

---

# Phase D — Diagnostics (Commit D)

### Task 15: Full `format_refusal` + success stderr + OverrideRejectReason structured emission

**Files:**

- Modify: `crates/nsl-codegen/src/wggo_prune.rs` — complete `format_refusal` for all 7 variants + add `diagnostic_code(refusal: &PruneRefusal) -> OverrideRejectReason` mapping function
- Modify: `crates/nsl-codegen/src/stmt.rs` — at the Phase A Task 3 insertion point, replace the placeholder `eprintln!("[prune] refusal: ...")` with a real emission loop; add success-path stderr emission

- [ ] **Step 1: Write the failing test** — text snapshot + structural assertion for every refusal variant.

Extend `crates/nsl-codegen/tests/wggo_prune_rewrite.rs`:

```rust
use nsl_codegen::wggo_prune::{format_refusal, diagnostic_code};
use nsl_codegen::wggo_overrides::OverrideRejectReason;

#[test]
fn refusal_text_and_code_cross_layer_param() {
    let refusal = PruneRefusal::CrossLayerParam {
        layer_name: "blocks.7.attn".into(),
        layer_role: LayerRole::Attention,
        param_name: "blocks.7.attn.wq".into(),
        param_var: 200,
        external_consumer: 42,
        external_op_kind: "Mul".into(),
    };
    insta::assert_snapshot!("refusal_cross_layer_param", format_refusal(&refusal));
    assert_eq!(diagnostic_code(&refusal), OverrideRejectReason::PruneCrossLayerParam);
}

// Repeat for all 7 variants:
// refusal_text_and_code_no_residual_add
// refusal_text_and_code_parallel_residuals
// refusal_text_and_code_ambiguous_pattern
// refusal_text_and_code_empty_closure
// refusal_text_and_code_whole_block_unsupported
// refusal_text_and_code_conflicting_decisions

#[test]
fn success_path_stderr_format_matches_spec() {
    // Spec §6.1:
    //   [prune] layer=N name=blocks.N.attn role=Attention applied=true closure_size=K ops_deleted=K residual_add_op=ID
    let rewrite = PruneRewrite {
        layer_name: "blocks.7.attn".into(),
        layer_role: LayerRole::Attention,
        h_before_var: 100,
        h_after_var: 202,
        residual_add_op: 2,
        closure_ops: vec![0, 1, 2], // 3 ops for this example
    };
    let line = nsl_codegen::wggo_prune::format_success_stderr(&rewrite, /*layer_index=*/ 7, /*ops_deleted=*/ 3);
    insta::assert_snapshot!("success_stderr_blocks_7_attn", line);
    assert!(line.contains("applied=true"));
    assert!(line.contains("closure_size=3"));
    assert!(line.contains("ops_deleted=3"));
    assert!(line.contains("residual_add_op=2"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Expected: FAIL on the new helpers (`diagnostic_code`, `format_success_stderr`) and on the expanded `format_refusal`.

- [ ] **Step 3: Implement the helpers**

In `wggo_prune.rs`:

```rust
use crate::wggo_overrides::OverrideRejectReason;

/// Map a refusal variant to its structured diagnostic code. Spec §6.3.
pub fn diagnostic_code(r: &PruneRefusal) -> OverrideRejectReason {
    match r {
        PruneRefusal::CrossLayerParam { .. } => OverrideRejectReason::PruneCrossLayerParam,
        PruneRefusal::NoResidualAdd { .. } => OverrideRejectReason::PruneNoResidualAdd,
        PruneRefusal::ParallelResidualBranches { .. } => OverrideRejectReason::PruneParallelResidualBranches,
        PruneRefusal::AmbiguousPatternMatch { .. } => OverrideRejectReason::PruneAmbiguousPatternMatch,
        PruneRefusal::EmptyClosure { .. } => OverrideRejectReason::PruneEmptyClosure,
        PruneRefusal::WholeBlockUnsupported { .. } => OverrideRejectReason::PruneWholeBlockUnsupported,
        PruneRefusal::ConflictingPruneDecisions { .. } => OverrideRejectReason::PruneConflictingDecisions,
    }
}

/// Spec §6.1 success-path stderr line.
pub fn format_success_stderr(rewrite: &PruneRewrite, layer_index: u32, ops_deleted: usize) -> String {
    format!(
        "[prune] layer={} name={} role={:?} applied=true closure_size={} ops_deleted={} residual_add_op={}",
        layer_index,
        rewrite.layer_name,
        rewrite.layer_role,
        rewrite.closure_ops.len(),
        ops_deleted,
        rewrite.residual_add_op,
    )
}

/// Spec §3 three-part refusal message. One format per variant.
pub fn format_refusal(r: &PruneRefusal) -> String {
    match r {
        PruneRefusal::CrossLayerParam {
            layer_name, layer_role, param_name, param_var, external_consumer, external_op_kind,
        } => format!(
            "prune: layer has cross-layer parameter sharing (not supported in v1).\n  \
             requested:  prune {}  (role={:?})\n  \
             expected:   all parameters matching `{}.*` consumed only within\n              the layer's computational closure\n  \
             found:      parameter `{}` (VarId {}) is consumed by\n              op_id={} ({}), which is\n              outside the closure for {}",
            layer_name, layer_role, layer_name, param_name, param_var,
            external_consumer, external_op_kind, layer_name,
        ),
        PruneRefusal::NoResidualAdd { layer_name, layer_role, closure_size } => format!(
            "prune: layer is not residual-structured (no boundary Add found).\n  \
             requested:  prune {}  (role={:?})\n  \
             expected:   exactly one op in the closure matching Add(h_before, block_output)\n              with block_output ∈ closure ∧ block_output single-consumer\n  \
             found:      closure has {} ops but zero ops match the residual\n              pattern; the layer appears to be non-residual (SSM / Mamba /\n              non-standard architecture)",
            layer_name, layer_role, closure_size,
        ),
        PruneRefusal::ParallelResidualBranches { layer_name, layer_role, add_ops } => format!(
            "prune: layer has parallel residual branches (not supported in v1).\n  \
             requested:  prune {}  (role={:?})\n  \
             expected:   exactly one residual boundary Add\n  \
             found:      {} residual Adds detected at ops {:?}; each appears to\n              be a separate residual branch (distinct h_before values). Parallel\n              residual pruning requires branch-by-branch semantics not yet\n              specified.",
            layer_name, layer_role, add_ops.len(), add_ops,
        ),
        PruneRefusal::AmbiguousPatternMatch { layer_name, layer_role, h_before_var, candidate_adds } => format!(
            "prune: layer has multiple candidate residual boundaries (pattern-match ambiguous).\n  \
             requested:  prune {}  (role={:?})\n  \
             expected:   exactly one op matching the residual pattern\n  \
             found:      {} candidate Adds match the residual pattern against the same\n              h_before (VarId {}): ops {:?}.\n              Boundary disambiguation requires architecture-specific rules not\n              yet specified.",
            layer_name, layer_role, candidate_adds.len(), h_before_var, candidate_adds,
        ),
        PruneRefusal::EmptyClosure { layer_name, layer_role, prefix } => format!(
            "prune: no parameters match the requested layer prefix.\n  \
             requested:  prune {}  (role={:?})\n  \
             expected:   at least one parameter VarId with var_name starting\n              with `{}`\n  \
             found:      zero matching parameters in the WeightMap. Check layer name /\n              index; the requested layer does not exist in the compiled model.",
            layer_name, layer_role, prefix,
        ),
        PruneRefusal::WholeBlockUnsupported { layer_name } => format!(
            "prune: whole-block pruning (LayerRole::Block) is not supported in v1.\n  \
             requested:  prune {}  (role=Block)\n  \
             supported:  prune {}.attn  (role=Attention)\n              prune {}.ffn   (role=Ffn)\n  \
             workaround: emit two sub-block prune decisions for layer; their combined\n              effect is semantically equivalent to whole-block prune in standard\n              pre-norm transformer architectures (NOT equivalent for post-norm,\n              parallel, or scaled-residual architectures).\n  \
             planned:    whole-block prune tracked for v2 (chain-collapse transformation).",
            layer_name, layer_name, layer_name,
        ),
        PruneRefusal::ConflictingPruneDecisions { decision_a, decision_b, reason } => format!(
            "prune: two prune decisions in the same plan conflict.\n  \
             requested:  prune {} AND prune {} in the same plan\n  \
             expected:   each rewrite's closure and VarId aliasing is disjoint from every\n              other rewrite's\n  \
             found:      {}",
            decision_a, decision_b, reason,
        ),
    }
}
```

In `stmt.rs`, replace the Task 3 placeholder `eprintln!("[prune] refusal: ...")` with:

```rust
if !wggo_prune_result.refusals.is_empty() {
    for refusal in &wggo_prune_result.refusals {
        let text = crate::wggo_prune::format_refusal(refusal);
        let code = crate::wggo_prune::diagnostic_code(refusal);
        // Emit through the existing diagnostic infrastructure. Mirror how
        // WholeBlockPruneNotImplemented emits — it uses eprintln!(...) at
        // this same scope, and routes the reason through the existing
        // OverrideRejectReason enum. Emit the refusal text to stderr and
        // also attach the OverrideRejectReason to whatever diagnostics
        // structure the surrounding compilation collects.
        eprintln!("{}\n", text);
        // Attach `code` to the compilation's diagnostic envelope (mirror the
        // existing collect_prune_diagnostics path).
    }
    return Err(/* existing compile-error type, per Task 3 */);
}

// Success path: one stderr line per rewrite.
for (i, rw) in wggo_prune_result.rewrites.iter().enumerate() {
    let line = crate::wggo_prune::format_success_stderr(rw, /*layer_index=*/ i as u32, wggo_prune_result.ops_deleted);
    eprintln!("{}", line);
}
```

**Implementer note on `layer_index`:** the spec's stderr format uses `layer=N` where `N` is the plan's original `layer_index`. Carry that through via either (a) storing `layer_index` in `PruneRewrite` (requires adding a field) or (b) looking it up from the AppliedPlan before the `run()` call. Option (a) is cleaner — add `pub layer_index: u32` to `PruneRewrite` and plumb it through `plan_rewrite`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p nsl-codegen --test wggo_prune_rewrite 2>&1 | tail -30
cargo insta accept --workspace  # accept new refusal-text snapshots
```

Expected: all new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -f crates/nsl-codegen/src/wggo_prune.rs crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/tests/wggo_prune_rewrite.rs crates/nsl-codegen/tests/snapshots/wggo_prune_rewrite__refusal_*.snap crates/nsl-codegen/tests/snapshots/wggo_prune_rewrite__success_*.snap
git commit -m "feat(wggo-prune): Phase D — full refusal text + success stderr + diag code

format_refusal produces the spec §3 three-part error template for each of
the 7 variants. format_success_stderr emits the spec §6.1 line on each
successful rewrite. diagnostic_code maps each variant to the matching
OverrideRejectReason::Prune* for structural assertions.

stmt.rs now emits refusals through format_refusal + diagnostic_code and
success lines through format_success_stderr, replacing Phase A's placeholder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Merge-gate smoke test for mixed supported + unsupported plans

**Files:**

- Modify: `crates/nsl-codegen/tests/wggo_prune_rewrite.rs`

- [ ] **Step 1: Write the failing test** — plan with one supported and one unsupported decision emits BOTH in one compile pass (spec §11 criterion #7).

```rust
#[test]
fn mixed_plan_emits_all_refusals_in_one_pass() {
    // Plan: prune blocks.1.attn (supported) AND prune blocks.0 (LayerRole::Block — refused).
    // Expected: one refusal (WholeBlockUnsupported), zero rewrites (dry-run-then-commit).
    let mut art = compile_fixture("tests/fixtures/prune_rewrite_toy.nsl");
    let mut plan = art.applied_plan.clone();
    for layer in &mut plan.layers {
        if layer.layer_name == "blocks.1.attn" { layer.coarse = LayerDecision::Prune; }
        // Need a blocks.0 whole-block entry; if the planner doesn't produce
        // one, synthesize one:
    }
    plan.layers.push(AppliedLayer {
        layer_index: 999,
        layer_name: "blocks.0".into(),
        coarse: LayerDecision::Prune,
        // ... zero-fill other fields via ::default() if available, or mk_prune_layer helper.
        ..Default::default()
    });

    let result = run(&mut art.wengert, &plan, &art.weight_map);

    assert_eq!(result.rewrites.len(), 0, "dry-run-then-commit: no rewrites applied on refusal");
    assert!(result.refusals.iter().any(|r| matches!(r, PruneRefusal::WholeBlockUnsupported { .. })));
    // blocks.1.attn SHOULD have planned successfully (no refusal for it), but
    // its rewrite is held back by Phase 2's early-return.

    // Wengert UNCHANGED on refusal.
    let original_ops = art.wengert.ops.len();
    // (Compile fixture freshly to compare; re-reading is simplest.)
    let fresh = compile_fixture("tests/fixtures/prune_rewrite_toy.nsl");
    assert_eq!(art.wengert.ops.len(), fresh.wengert.ops.len());
}
```

- [ ] **Step 2: Run test**

Expected: PASS if Phase B/C/D are correctly wired. If blocks.1.attn's successful plan accidentally commits mutations before the refusal is aggregated, the test catches the violation — fix by ensuring Phase 2's early-return is strictly before any call to `apply_rewrite`.

- [ ] **Step 3: Run the full crate test suite + baseline comparison**

```bash
cargo test -p nsl-codegen 2>&1 | tail -20
```

Expected: all tests PASS. Compare against the main-branch baseline count:

```bash
git checkout main && cargo test -p nsl-codegen 2>&1 | grep -E "test result:" | tail -3
git checkout feat/wggo-prune-ir-rewrite
```

The feat-branch count should be main's count + the new tests added here (~20-30 new tests across Layer 1/2/3/4). Any net decrease in passing tests means a regression; diagnose and fix before proceeding.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/tests/wggo_prune_rewrite.rs
git commit -m "test(wggo-prune): merge-gate smoke — mixed plan emits all refusals in one pass

Spec §11 criterion #7. Plan with one supported + one unsupported prune
decision produces (a) the refusal for the unsupported one, (b) zero
rewrites applied (dry-run-then-commit), (c) wengert unchanged. Validates
the whole pipeline — Phase 1 validation, Phase 2 batched refusal
emission, Phase 3 early-return preservation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**End of Phase D. End of implementation.**

---

## Final Verification Checklist (for finishing-a-development-branch skill)

Before handing off:

- [ ] `cargo test -p nsl-codegen 2>&1 | grep "test result:"` — no failures.
- [ ] `cargo test -p nsl-codegen --lib wggo_prune 2>&1 | grep "test result:"` — all Layer 2 tests green.
- [ ] `cargo test -p nsl-codegen --test wggo_prune_rewrite 2>&1 | grep "test result:"` — all Layer 3/4 tests green.
- [ ] `cargo insta pending-snapshots --workspace 2>&1 | head -5` — no pending snapshots.
- [ ] `cargo check -p nsl-codegen 2>&1 | tail -5` — no errors; no new warnings beyond pre-existing baseline.
- [ ] Spec §11's seven merge-gate criteria all satisfied:
  - [ ] 1. All four test layers green
  - [ ] 2. Layer 3 tests pass bit-exactly (no tolerance)
  - [ ] 3. Each of 7 refusal variants verified by text snapshot + OverrideRejectReason assertion
  - [ ] 4. `wggo_prune::run()` returns empty rewrites + untouched wengert on any refusal (Task 16)
  - [ ] 5. Pipeline position verified (stmt.rs inserts call before WRGA)
  - [ ] 6. `reason=ir_rewrite_not_implemented` stderr survives for `LayerRole::Block` only (Task 2 preserved it)
  - [ ] 7. Mixed-plan smoke (Task 16)

After verification: use `superpowers:finishing-a-development-branch` skill for final PR / merge workflow.
