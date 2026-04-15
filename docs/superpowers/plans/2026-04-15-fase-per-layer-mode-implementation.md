# FASE Per-Layer Mode Override Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable WGGO's `PerLayerOverride.fase_fused` to flow into FASE's planner (always) and codegen (when feasible), making Consumer 3 of the 5-consumer rollout shipped end-to-end.

**Architecture:** Surgical additive refactor. `FasePlan` gains `per_layer_mode: Option<Vec<FaseMode>>` + `override_diagnostics: Vec<OverrideDiagnostic>`. New `plan_with_overrides(cfg, &[bool])` entry point clamps infeasible Deferred requests to FullBuffer. `mode_for_layer(i)` helper collapses per-layer vs global lookup. Memory schedule and codegen consume per-layer modes where practical.

**Tech Stack:** Rust, existing `crate::fase::*`, `crate::wggo_overrides::*`, Cranelift IR emitters in `stmt.rs` / `stmt_fase.rs`.

**Spec:** [docs/superpowers/specs/2026-04-15-fase-per-layer-mode-design.md](../specs/2026-04-15-fase-per-layer-mode-design.md)

**Branch:** `feat/fase-per-layer-refactor` (already created from synced `main` at `49c0dd9`).

## Task 0 Outcome: C (Codegen dispatch deferred)

The backward loop at `stmt.rs:4829-4876` is a runtime loop (param index `gai` via `nsl_list_get`) with a compile-time branch at line 4837 gating Deferred vs FullBuffer — all iterations take the same branch. Per-layer dispatch would require either a runtime lookup table keyed on param index, or loop unrolling at codegen — both substantial refactors beyond this plan's scope. The same pattern repeats at `stmt.rs:4903` (optimizer step). Task 7 ships outcome C: a TODO comment at `stmt.rs:3277` pointing to Phase 2 codegen work.

---

## File Inventory

**Modify:**
- `crates/nsl-codegen/src/fase.rs` — add fields, `plan_with_overrides`, `mode_for_layer`, unit tests.
- `crates/nsl-codegen/src/wggo_overrides.rs` — add `FaseModeInfeasible` reject reason.
- `crates/nsl-codegen/src/fase_memory.rs` — per-layer schedule iteration.
- `crates/nsl-codegen/src/stmt.rs` — wire WGGO → `plan_with_overrides` at FASE call site; stderr renderer.
- `crates/nsl-codegen/src/stmt_fase.rs` — per-layer codegen dispatch (if Task 0 confirms feasible).

**Create:**
- None. No new test files — all 9 tests fit in existing `#[cfg(test)] mod tests` blocks or an integration test beside existing ones.

---

## Task 0: Codegen threading feasibility (discovery)

**Files:** Read-only. No edits.

The spec assumes per-layer mode can be threaded into `stmt.rs`'s backward emitter. Before locking that in, confirm the mechanism.

- [ ] **Step 1: Read the FASE dispatch in stmt.rs**

```
cd c:/Users/bwiem/projects/NSL/.worktrees/fase-per-layer-refactor
grep -n "fase_plan\|fase::plan\|FaseMode\|fase_deferred" crates/nsl-codegen/src/stmt.rs
```

Read stmt.rs around lines 3265 (plan call) and 3277 (mode check). Inspect how the backward loop iterates (parameters? Wengert ops? layers?) and whether a layer index is in scope at the Deferred vs FullBuffer branch.

- [ ] **Step 2: Classify the threading model**

Write down the answer to: "When emitting a backward op, does the code know which WGGO-AppliedPlan layer it belongs to?"

Three outcomes:

- **(A)** Layer index is already in scope (e.g. iterated alongside an `AppliedLayer` vector). Per-layer codegen dispatch is cheap — Task 7 wires it directly.
- **(B)** Code iterates parameters or Wengert ops; layer index requires a new mapping (e.g. `param_name → layer_index` built from `AppliedPlan`). Task 7 adds the mapping + per-param lookup.
- **(C)** Layer structure is flattened beyond recovery at the FASE emitter. Per-layer codegen is a bigger refactor than this plan. Task 7 ships with codegen using `plan.mode` only (no per-layer dispatch) and leaves per-layer codegen as a documented follow-up. The planner + memory schedule changes still land.

- [ ] **Step 3: Record the decision**

Append a 3-line `## Task 0 Outcome` section to THIS PLAN FILE, noting which option (A/B/C) applies and what that means for Task 7's scope. The recording is the gate — subsequent tasks reference it.

- [ ] **Step 4: Commit if any discovery note was added**

If the outcome note was appended to the plan file, commit:

```
git add docs/superpowers/plans/2026-04-15-fase-per-layer-mode-implementation.md
git commit -m "docs(fase): record Task 0 outcome — codegen threading is <A|B|C>"
```

If no note was added (pure mental discovery), skip the commit.

---

## Task 1: Add `FaseModeInfeasible` reject reason

**Files:**
- Modify: `crates/nsl-codegen/src/wggo_overrides.rs`

- [ ] **Step 1: Read the existing `OverrideRejectReason` enum**

```
grep -n "pub enum OverrideRejectReason" crates/nsl-codegen/src/wggo_overrides.rs
```

Read the enum to confirm the variant naming style (PascalCase struct-variant with named fields, based on `ShardFactorIncompatibleWithWorldSize { recommended, world_size }`).

- [ ] **Step 2: Write the failing test**

Add at the bottom of `wggo_overrides.rs` `#[cfg(test)] mod tests` (find it with `grep -n "#\[cfg(test)\]" crates/nsl-codegen/src/wggo_overrides.rs`):

```rust
#[test]
fn fase_mode_infeasible_round_trips_debug() {
    let r = OverrideRejectReason::FaseModeInfeasible {
        optimizer: crate::fase::FaseOptimizer::Lion,
        global_mode: crate::fase::FaseMode::FullBuffer,
    };
    let s = format!("{:?}", r);
    assert!(s.contains("FaseModeInfeasible"));
    assert!(s.contains("Lion"));
    assert!(s.contains("FullBuffer"));
}
```

- [ ] **Step 3: Run to confirm failure**

```
cargo test -p nsl-codegen wggo_overrides::tests::fase_mode_infeasible_round_trips_debug
```

Expected: FAIL with "no variant named `FaseModeInfeasible`".

- [ ] **Step 4: Add the variant**

In `wggo_overrides.rs`, `pub enum OverrideRejectReason`, add:

```rust
/// WGGO requested Deferred mode on a layer whose global FASE plan is
/// FullBuffer because the optimizer does not support deferred accumulation
/// (Lion, Unknown, or AdamW/Adam with allow_v_approx=false).
FaseModeInfeasible {
    optimizer: crate::fase::FaseOptimizer,
    global_mode: crate::fase::FaseMode,
},
```

If the enum has `#[non_exhaustive]` or a match site elsewhere that exhaustively matches it, extend those matches to handle the new variant via `other => format!("{:?}", other)` or similar — find them with:

```
grep -rn "OverrideRejectReason::" crates/nsl-codegen/src/
```

- [ ] **Step 5: Verify test passes and full build is clean**

```
cargo test -p nsl-codegen wggo_overrides::tests::fase_mode_infeasible_round_trips_debug
cargo build -p nsl-codegen
```

Both must be clean (no E0004 non-exhaustive match errors).

- [ ] **Step 6: Commit**

```
git add crates/nsl-codegen/src/wggo_overrides.rs
git commit -m "feat(wggo): FaseModeInfeasible override reject reason

Carries (optimizer, global_mode) so consumer code can render a precise
stderr diagnostic when WGGO requests Deferred on a layer that the global
FASE plan cannot support."
```

---

## Task 2: `mode_for_layer` helper + new `FasePlan` fields

**Files:**
- Modify: `crates/nsl-codegen/src/fase.rs`

- [ ] **Step 1: Write failing tests for new fields + helper**

Add to `fase.rs` test module:

```rust
#[test]
fn fase_plan_exposes_per_layer_mode_none_by_default() {
    let p = plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    assert!(p.per_layer_mode.is_none());
    assert!(p.override_diagnostics.is_empty());
}

#[test]
fn mode_for_layer_falls_back_to_global_when_none() {
    let p = plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    assert_eq!(p.mode_for_layer(0), FaseMode::Deferred);
    assert_eq!(p.mode_for_layer(99), FaseMode::Deferred);
}

#[test]
fn mode_for_layer_reads_per_layer_vector_when_some() {
    let mut p = plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    p.per_layer_mode = Some(vec![FaseMode::FullBuffer, FaseMode::Deferred]);
    assert_eq!(p.mode_for_layer(0), FaseMode::FullBuffer);
    assert_eq!(p.mode_for_layer(1), FaseMode::Deferred);
    // Out-of-range falls back to global.
    assert_eq!(p.mode_for_layer(2), FaseMode::Deferred);
}
```

- [ ] **Step 2: Run to confirm failure**

```
cargo test -p nsl-codegen fase::tests::fase_plan_exposes_per_layer_mode_none_by_default
```

Expected: FAIL with "no field `per_layer_mode`".

- [ ] **Step 3: Add the fields**

In `fase.rs`, `pub struct FasePlan` (around line 144), add two fields after `rationale`:

```rust
/// Per-layer mode overrides. `None` ⇒ every layer uses `mode` (today's
/// behavior, byte-identical). `Some(v)` ⇒ layer `i` uses `v[i]`; `mode`
/// becomes an informational default.
#[serde(default)]
pub per_layer_mode: Option<Vec<FaseMode>>,
/// WGGO recommendations that FASE had to clamp due to global feasibility.
/// Empty when no overrides were supplied or all were applied verbatim.
#[serde(default)]
pub override_diagnostics: Vec<crate::wggo_overrides::OverrideDiagnostic>,
```

- [ ] **Step 4: Update all `FasePlan { ... }` literals**

The existing `plan()` function and any test helpers that construct `FasePlan { ... }` will now fail with E0063 missing-field errors. Find them:

```
grep -rn "FasePlan {" crates/nsl-codegen/src/
```

For each, add:

```rust
per_layer_mode: None,
override_diagnostics: Vec::new(),
```

- [ ] **Step 5: Add the helper**

In `fase.rs` `impl FasePlan`, add:

```rust
/// Returns the mode for layer `i`. Falls back to `self.mode` when no
/// per-layer override vector is present or when `i` is out of range.
pub fn mode_for_layer(&self, i: usize) -> FaseMode {
    self.per_layer_mode
        .as_ref()
        .and_then(|v| v.get(i).copied())
        .unwrap_or(self.mode)
}
```

- [ ] **Step 6: Verify tests pass**

```
cargo test -p nsl-codegen fase::tests
cargo build -p nsl-codegen
```

All three new tests + existing FASE tests must pass.

- [ ] **Step 7: Commit**

```
git add crates/nsl-codegen/src/fase.rs
git commit -m "feat(fase): FasePlan.per_layer_mode + mode_for_layer helper

Additive: per_layer_mode defaults to None (byte-identical to today).
mode_for_layer(i) collapses the per-layer-vs-global lookup into a single
call so consumers can use one code path."
```

---

## Task 3: `plan_with_overrides` planner entry point

**Files:**
- Modify: `crates/nsl-codegen/src/fase.rs`

- [ ] **Step 1: Write failing tests (5 planner cases)**

Add to `fase.rs` test module:

```rust
#[test]
fn plan_with_overrides_empty_input_matches_plan() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    };
    let baseline = plan(&cfg);
    let overridden = plan_with_overrides(&cfg, &[]);
    assert_eq!(overridden.mode, baseline.mode);
    assert_eq!(overridden.accumulation, baseline.accumulation);
    assert!(overridden.per_layer_mode.is_none());
    assert!(overridden.override_diagnostics.is_empty());
}

#[test]
fn plan_with_overrides_passthrough_ignores_all() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 1,  // Passthrough
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, true]);
    assert_eq!(p.mode, FaseMode::Passthrough);
    assert!(p.per_layer_mode.is_none());
    assert!(p.override_diagnostics.is_empty());
}

#[test]
fn plan_with_overrides_adamw_deferred_mixes_modes() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false, true, false]);
    assert_eq!(p.mode, FaseMode::Deferred);
    assert_eq!(
        p.per_layer_mode,
        Some(vec![
            FaseMode::Deferred,
            FaseMode::FullBuffer,
            FaseMode::Deferred,
            FaseMode::FullBuffer,
        ])
    );
    assert!(p.override_diagnostics.is_empty());
}

#[test]
fn plan_with_overrides_lion_global_clamps_deferred_requests() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::Lion,
        accumulation: 4,
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true, false]);
    assert_eq!(p.mode, FaseMode::FullBuffer);
    assert_eq!(
        p.per_layer_mode,
        Some(vec![FaseMode::FullBuffer, FaseMode::FullBuffer])
    );
    assert_eq!(p.override_diagnostics.len(), 1);
    let diag = &p.override_diagnostics[0];
    assert_eq!(diag.layer_index, 0);
    assert_eq!(diag.requested, "Deferred");
    assert_eq!(diag.applied, "FullBuffer");
    assert!(matches!(
        diag.reason,
        crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
            optimizer: FaseOptimizer::Lion,
            global_mode: FaseMode::FullBuffer,
        }
    ));
}

#[test]
fn plan_with_overrides_allow_v_approx_false_clamps() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        allow_v_approx: false,  // forces FullBuffer globally
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true]);
    assert_eq!(p.mode, FaseMode::FullBuffer);
    assert_eq!(p.per_layer_mode, Some(vec![FaseMode::FullBuffer]));
    assert_eq!(p.override_diagnostics.len(), 1);
}
```

- [ ] **Step 2: Run to confirm failure**

```
cargo test -p nsl-codegen fase::tests::plan_with_overrides_empty_input_matches_plan
```

Expected: FAIL with "function `plan_with_overrides` not found".

- [ ] **Step 3: Implement the function**

In `fase.rs`, near `pub fn plan`, add:

```rust
/// WGGO-aware variant of [`plan`]. Given a per-layer `fase_fused` vector,
/// returns a plan with `per_layer_mode` populated and any infeasible
/// overrides clamped + logged in `override_diagnostics`.
///
/// Empty input → same as `plan(cfg)`.
/// Passthrough global (accumulation=1) → overrides ignored; `per_layer_mode`
/// stays `None`.
pub fn plan_with_overrides(
    cfg: &FaseConfig,
    wggo_fused_per_layer: &[bool],
) -> FasePlan {
    let mut p = plan(cfg);

    if wggo_fused_per_layer.is_empty() || p.mode == FaseMode::Passthrough {
        return p;
    }

    let mut per_layer = Vec::with_capacity(wggo_fused_per_layer.len());
    let mut diagnostics = Vec::new();

    for (i, &fused) in wggo_fused_per_layer.iter().enumerate() {
        let requested = if fused { FaseMode::Deferred } else { FaseMode::FullBuffer };
        let applied = match (p.mode, requested) {
            // Global FullBuffer forbids Deferred — clamp + log.
            (FaseMode::FullBuffer, FaseMode::Deferred) => {
                diagnostics.push(crate::wggo_overrides::OverrideDiagnostic {
                    layer_index: i as u32,
                    layer_name: format!("layer_{i}"),
                    reason: crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
                        optimizer: cfg.optimizer,
                        global_mode: p.mode,
                    },
                    requested: "Deferred".into(),
                    applied: "FullBuffer".into(),
                });
                FaseMode::FullBuffer
            }
            // Every other combination is feasible.
            _ => requested,
        };
        per_layer.push(applied);
    }

    p.per_layer_mode = Some(per_layer);
    p.override_diagnostics = diagnostics;
    p
}
```

- [ ] **Step 4: Verify all 5 planner tests pass**

```
cargo test -p nsl-codegen fase::tests::plan_with_overrides
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```
git add crates/nsl-codegen/src/fase.rs
git commit -m "feat(fase): plan_with_overrides entry point for WGGO mode overrides

Delegates to plan(cfg) for the global shape, then clamps infeasible
Deferred requests (Lion, Unknown, allow_v_approx=false) to FullBuffer
and logs FaseModeInfeasible diagnostics. Empty input or Passthrough
global is a no-op with per_layer_mode=None."
```

---

## Task 4: `fase_memory.rs` per-layer schedule

**Files:**
- Modify: `crates/nsl-codegen/src/fase_memory.rs`

**Context:** The current `fase_breakdown` uses a single `match plan.mode { ... }`. Today, Deferred and FullBuffer produce identical `(accumulator_bytes, one_layer_grad)` tuples, so per-layer variation doesn't change numeric outputs. This task updates the function to be structurally per-layer — prepares the code for future refinement (e.g. if Deferred-mode m_partial sizing becomes layer-aware) without changing today's peaks.

- [ ] **Step 1: Write a test that all-Deferred per-layer matches global Deferred**

Add to `fase_memory.rs` test module:

```rust
#[test]
fn all_deferred_per_layer_matches_global_deferred_schedule() {
    let footprint = nslcoder_50m_footprint();
    let global = fase_plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    let mut overridden = global.clone();
    overridden.per_layer_mode = Some(vec![FaseMode::Deferred; footprint.num_layers()]);

    let s_global = schedule(&footprint, &global);
    let s_overridden = schedule(&footprint, &overridden);

    assert_eq!(s_global.fase.peak, s_overridden.fase.peak);
    assert_eq!(s_global.fase.gradients, s_overridden.fase.gradients);
}

#[test]
fn mixed_per_layer_modes_produce_valid_schedule() {
    let footprint = nslcoder_50m_footprint();
    let global = fase_plan(&FaseConfig {
        optimizer: FaseOptimizer::AdamW,
        accumulation: 4,
        ..Default::default()
    });
    let mut overridden = global.clone();
    let n = footprint.num_layers();
    let modes: Vec<FaseMode> = (0..n)
        .map(|i| if i % 2 == 0 { FaseMode::Deferred } else { FaseMode::FullBuffer })
        .collect();
    overridden.per_layer_mode = Some(modes);

    let s = schedule(&footprint, &overridden);
    // Peak must be positive and not less than a pure-Deferred schedule.
    let s_deferred = schedule(&footprint, &global);
    assert!(s.fase.peak >= s_deferred.fase.peak);
}
```

**Caveat:** `ModelFootprint::num_layers()` may not exist; check with `grep -n "num_layers\|pub fn\|impl ModelFootprint" crates/nsl-codegen/src/fase_memory.rs`. If absent, either add a trivial getter or count something you can infer (e.g. length of an existing field). If the second test requires a helper that doesn't exist, simplify: assert `s.fase.peak > 0` — the goal is only that mixed modes don't panic or produce zero.

- [ ] **Step 2: Run to confirm failure**

```
cargo test -p nsl-codegen fase_memory::tests::all_deferred_per_layer_matches_global_deferred_schedule
```

Expected: FAIL or PASS depending on whether the current global-mode path coincidentally handles `per_layer_mode = Some(...)`. If PASS without changes, the implementation step is a no-op — move to the structural update anyway.

- [ ] **Step 3: Refactor `fase_breakdown` to be per-layer-aware**

In `fase_memory.rs`, replace the current `match plan.mode { ... }` in `fase_breakdown` with a per-layer iteration that uses `plan.mode_for_layer(i)`. Because today's Deferred and FullBuffer produce identical bytes, the implementation can keep the per-mode tuple formula but evaluated per layer:

```rust
fn fase_breakdown(footprint: &ModelFootprint, plan: &FasePlan) -> MemoryBreakdown {
    let params = footprint.total_param_bytes();
    let opt_state = optimizer_bytes(footprint);
    let peak_activation_layer = footprint.peak_activation_layer_bytes();

    // Per-layer mode drives the accumulator vs full-buffer choice. Peak is
    // the maximum over layers of (accumulator + one-layer-grad + activation).
    let num_layers = footprint.num_layers();
    let mut max_working_bytes = 0u64;
    let mut accumulator_bytes = 0u64;
    let mut one_layer_grad = 0u64;

    for i in 0..num_layers {
        let mode = plan.mode_for_layer(i);
        let (acc_i, grad_i) = match mode {
            FaseMode::Passthrough => (params, 0),
            FaseMode::Deferred    => (params, footprint.max_param_bytes()),
            FaseMode::FullBuffer  => (params, footprint.max_param_bytes()),
        };
        accumulator_bytes = accumulator_bytes.max(acc_i);
        one_layer_grad    = one_layer_grad.max(grad_i);
        let working = acc_i + grad_i;
        max_working_bytes = max_working_bytes.max(working);
    }

    // Peak uses the global `plan.mode` only to decide whether to add
    // `one_layer_grad` at all (Passthrough has no grad residency).
    let peak = if plan.mode == FaseMode::Passthrough && plan.per_layer_mode.is_none() {
        params + accumulator_bytes + opt_state + peak_activation_layer
    } else {
        params + accumulator_bytes + one_layer_grad + opt_state + peak_activation_layer
    };

    MemoryBreakdown {
        params,
        gradients: accumulator_bytes,
        optimizer_state: opt_state,
        activations: peak_activation_layer,
        peak,
    }
}
```

If `ModelFootprint::num_layers()` doesn't exist, add a simple getter — whatever field already represents layer count (probably `per_layer_param_bytes.len()` or similar).

- [ ] **Step 4: Verify tests pass + no regression**

```
cargo test -p nsl-codegen fase_memory
```

All existing tests + both new ones must pass. If any existing test's assertion relied on a specific numeric peak that changed, investigate before relaxing the test — the refactor should be numerically identical when `per_layer_mode = None`.

- [ ] **Step 5: Commit**

```
git add crates/nsl-codegen/src/fase_memory.rs
git commit -m "refactor(fase): per-layer memory schedule iteration

fase_breakdown now iterates layers via plan.mode_for_layer(i) rather
than a single global match. Today's Deferred/FullBuffer peaks are
numerically unchanged (both produce identical tuples); structure is
ready for future per-layer refinement."
```

---

## Task 5: Wire WGGO → `plan_with_overrides` in stmt.rs

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

- [ ] **Step 1: Locate the FASE call site**

```
grep -n "fase::plan(&\|fase_plan = crate::fase::plan" crates/nsl-codegen/src/stmt.rs
```

Expect one call around line 3265 of the form:
```rust
let fase_plan = crate::fase::plan(&crate::fase::FaseConfig { ... });
```

- [ ] **Step 2: Replace with WGGO-aware branch**

Change the call site to:

```rust
let fase_cfg = crate::fase::FaseConfig { /* existing args */ };
let fase_plan = match self.wggo_overrides.as_ref() {
    Some(o) => {
        let fused: Vec<bool> = o.per_layer.iter().map(|p| p.fase_fused).collect();
        crate::fase::plan_with_overrides(&fase_cfg, &fused)
    }
    None => crate::fase::plan(&fase_cfg),
};
```

The exact `FaseConfig { ... }` literal stays unchanged. Only the planner call is swapped.

- [ ] **Step 3: Add the stderr diagnostic renderer**

Immediately after the planner call, render any diagnostics:

```rust
for diag in &fase_plan.override_diagnostics {
    let reason_str = match &diag.reason {
        crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
            optimizer,
            global_mode,
        } => format!(
            "{:?}_optimizer_global_mode_{:?}",
            optimizer, global_mode
        )
        .to_lowercase(),
        other => format!("{:?}", other),
    };
    eprintln!(
        "[fase] layer:{} wggo-override-rejected requested={} applied={} reason={}",
        diag.layer_index, diag.requested, diag.applied, reason_str
    );
}
```

Place this next to the existing `[wrga] ...` / `[csha] ...` stderr renderers for consistency — find them with `grep -n "wggo-override-rejected" crates/nsl-codegen/src/stmt.rs` and put the FASE block adjacent.

- [ ] **Step 4: Build clean**

```
cargo build -p nsl-codegen
cargo test -p nsl-codegen --lib
```

No new errors. Existing test count unchanged (plan call-site change is semantic-preserving when `wggo_overrides` is `None`).

- [ ] **Step 5: Commit**

```
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(fase): wire WGGO overrides into plan_with_overrides

When self.wggo_overrides is Some, derive a per-layer fase_fused Vec and
route through fase::plan_with_overrides. Render FaseModeInfeasible
diagnostics to stderr in the [fase] layer:N format matching CSHA/WRGA/CPDT."
```

---

## Task 6: Integration test for the `[fase] ...` stderr diagnostic

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs` test module OR create `crates/nsl-codegen/tests/fase_override_diagnostic.rs`.

Because the stmt.rs call site isn't directly unit-testable (requires a full compile), this test uses the `plan_with_overrides` path + the stderr formatter in isolation. The formatter logic is simple enough that an integration test of the actual stderr is low-value; instead, assert the diagnostic shape that the formatter consumes.

- [ ] **Step 1: Write a planner-level test asserting the diagnostic carries the needed fields**

Add to `fase.rs` test module (not a new file — the assertion is pure data):

```rust
#[test]
fn fase_mode_infeasible_diagnostic_has_stderr_shape() {
    let cfg = FaseConfig {
        optimizer: FaseOptimizer::Lion,
        accumulation: 4,
        ..Default::default()
    };
    let p = plan_with_overrides(&cfg, &[true]);
    let diag = &p.override_diagnostics[0];

    // Shape expected by the stmt.rs stderr renderer.
    assert_eq!(diag.layer_index, 0);
    assert_eq!(diag.requested.as_str(), "Deferred");
    assert_eq!(diag.applied.as_str(), "FullBuffer");
    match &diag.reason {
        crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
            optimizer,
            global_mode,
        } => {
            assert_eq!(*optimizer, FaseOptimizer::Lion);
            assert_eq!(*global_mode, FaseMode::FullBuffer);
        }
        _ => panic!("expected FaseModeInfeasible"),
    }
}
```

- [ ] **Step 2: Verify passes**

```
cargo test -p nsl-codegen fase::tests::fase_mode_infeasible_diagnostic_has_stderr_shape
```

Expected: PASS (all the needed types + behavior already shipped in Tasks 1-3).

- [ ] **Step 3: Commit**

```
git add crates/nsl-codegen/src/fase.rs
git commit -m "test(fase): diagnostic carries fields required by stderr renderer

Pins the FaseModeInfeasible diagnostic's field shape so the stmt.rs
stderr formatter has a stable contract. If stderr format drifts, this
test won't catch it — rely on the CSHA/WRGA/CPDT renderer symmetry and
manual verification."
```

---

## Task 7: Per-layer codegen dispatch (scope depends on Task 0 outcome)

**Files:** Depend on Task 0's outcome.

### 7.A — Outcome (A): Layer index in scope

- [ ] **Step 1:** In `stmt.rs` around line 3277, change `let fase_deferred = fase_plan.mode == FaseMode::Deferred;` to a per-layer lookup at each use site. Use `fase_plan.mode_for_layer(layer_idx)` where `layer_idx` is the in-scope variable.

- [ ] **Step 2:** Build + run full test suite: `cargo test -p nsl-codegen`. No regressions; the single-mode case (no overrides) must match today's behavior byte-for-byte.

- [ ] **Step 3:** Commit with message `"feat(fase): per-layer mode dispatch in backward codegen"`.

### 7.B — Outcome (B): Needs param-name → layer-index mapping

- [ ] **Step 1:** At the FASE call site, build a `HashMap<String, usize>` from `self.wggo_overrides.as_ref().map(|o| o.per_layer.iter().enumerate().flat_map(|(i, pl)| pl.contained_projections.iter().map(move |n| (n.clone(), i))).collect())` (or whatever projection-name field exists on `PerLayerOverride` — check with `grep -n "pub struct PerLayerOverride" crates/nsl-codegen/src/wggo_overrides.rs`).

- [ ] **Step 2:** At each backward emission site that today reads `fase_plan.mode`, look up the param name in the map and call `fase_plan.mode_for_layer(i)` using the resulting index. Fall back to global mode when the name isn't found.

- [ ] **Step 3:** Build + test. Commit.

### 7.C — Outcome (C): Codegen dispatch deferred

- [ ] **Step 1:** Leave `stmt.rs:3277` unchanged; the backward emitter continues using `fase_plan.mode`. The planner + memory schedule + stderr diagnostics still land (Tasks 1-6), so WGGO's `fase_fused` signal flows through the plan and is visible in diagnostics even though codegen doesn't yet branch per-layer.

- [ ] **Step 2:** Add a TODO comment near `stmt.rs:3277`:
  ```rust
  // TODO(consumer-3-phase-2): Per-layer codegen dispatch deferred —
  // backward emitter flattens layer identity beyond recovery at this
  // site. Planner + memory schedule + diagnostics already honor per-layer
  // modes; wire codegen when the emitter exposes layer indices.
  // See docs/superpowers/specs/2026-04-15-fase-per-layer-mode-design.md §9.
  ```

- [ ] **Step 3:** Commit with message `"docs(fase): note codegen dispatch deferred (Task 0 outcome C)"`.

---

## Task 8: Update memory file + push

- [ ] **Step 1: Update MEMORY.md and project_wggo_consumers.md**

Memory file lives outside the repo at `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/`.

In `project_wggo_consumers.md`:
- Change the Consumer 3 (FASE) section heading to include "(shipped 2026-04-15)" and document what landed.
- If Task 7 outcome was C, explicitly note that codegen dispatch is Phase 2.

In `MEMORY.md` top-of-file summary:
- Update the WGGO consumer rollout line to list FASE as shipped.

No commit (memory files aren't in git).

- [ ] **Step 2: Run full workspace test**

```
cd c:/Users/bwiem/projects/NSL/.worktrees/fase-per-layer-refactor
cargo test --workspace 2>&1 | tail -5
```

Expected: all pass, or only pre-existing infrastructure flakes (Windows file-lock races in `nsl run` e2e, documented in prior PRs).

- [ ] **Step 3: Push**

```
git push -u origin feat/fase-per-layer-refactor
```

- [ ] **Step 4: Prepare PR body**

Draft a PR body along the lines of:

```markdown
## Summary
- Adds `FasePlan.per_layer_mode` + `override_diagnostics`; new `plan_with_overrides` entry point.
- WGGO `fase_fused: bool` per layer flows into FASE's planner; infeasible Deferred requests (Lion / Unknown / allow_v_approx=false) clamp to FullBuffer + emit `FaseModeInfeasible` diagnostic.
- Stderr renderer matches CSHA/WRGA/CPDT format: `[fase] layer:N wggo-override-rejected requested=... applied=... reason=...`.

## Task 0 outcome
[Paste the recorded outcome from Step 3 of Task 0.]

## Out of scope
Per-layer optimizer hyperparams, per-layer accumulation, per-layer grad_clip. Codegen dispatch scope depends on Task 0 outcome (see commit for details).

## Test plan
- [ ] `cargo test -p nsl-codegen fase::tests` — 8 FASE tests pass
- [ ] `cargo test -p nsl-codegen fase_memory::tests` — all pass
- [ ] `cargo test -p nsl-codegen wggo_overrides::tests` — all pass
- [ ] Manual: compile an `@train` fixture with Lion optimizer + `--wggo auto`; assert `[fase] layer:N ...` diagnostic appears

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

User opens the PR on GitHub (gh token may be expired).

---

## Self-review checklist (run before claiming done)

- [ ] Every spec section (§3 data model, §4 planner, §5 consumers, §6 wiring, §7 testing) has ≥1 task implementing it.
- [ ] No `TBD` / `TODO` / `implement later` in plan text (Task 7.C does add a TODO comment to source code, which is intentional).
- [ ] Method / type names consistent across tasks: `plan_with_overrides`, `mode_for_layer`, `per_layer_mode`, `override_diagnostics`, `FaseModeInfeasible`.
- [ ] Every code-touching step shows the actual code (no "similar to Task N" hand-waves).
- [ ] Project's most-common mistake (E0063 missing-field initializers) is explicitly called out at Task 2 Step 4.
- [ ] Task 0 gate is load-bearing: Task 7 cannot be finalized until Task 0's outcome is recorded.
