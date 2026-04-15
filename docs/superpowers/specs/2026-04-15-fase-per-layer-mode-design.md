# FASE Per-Layer Mode Override — Design

**Date:** 2026-04-15
**Status:** Approved for implementation
**Branch (target):** `feat/fase-per-layer-refactor`
**Predecessor:** WGGO consumer rollout — [project_wggo_consumers.md](../../../C--Users-bwiem-projects-NSL/memory/project_wggo_consumers.md); CSHA / WRGA / CPDT shipped 2026-04-14 / 2026-04-15.

## 1. Goal

Enable WGGO's `PerLayerOverride.fase_fused: bool` to flow end-to-end into FASE's planner and codegen. After this lands, consumer 3 of the 5-consumer rollout is shipped; only Prune remains.

Phase 1 is a **surgical additive change**: the existing `plan(cfg) -> FasePlan` API stays intact and byte-identical when no overrides are supplied. Overrides enter through a new `plan_with_overrides` entry point.

## 2. Non-Goals (Explicitly Deferred)

- Per-layer optimizer hyperparameters (`lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `momentum`).
- Per-layer grad-clip thresholds.
- Per-layer accumulation counts or optimizer choice.
- Any change to `FaseConfig` itself — it stays a singleton.
- Runtime toggling of per-layer modes; decisions are compile-time only.

Rationale: every non-mode FASE field is genuinely global per-training-run. No realistic scenario exists where layer 3 uses AdamW while layer 7 uses SGD, or where different layers accumulate different counts. WGGO's `fase_fused: bool` expresses exactly one question per layer: *fused (Deferred) or separate (FullBuffer)*.

## 3. Data Model Changes

### 3.1 `FasePlan` gains two fields

File: `crates/nsl-codegen/src/fase.rs`

```rust
#[derive(Debug, Clone, Serialize)]
pub struct FasePlan {
    // existing fields unchanged:
    pub mode: FaseMode,                 // global default mode
    pub accumulation: u32,
    pub backward_phases: Vec<BackwardPhase>,
    pub recipe: UpdateRecipe,
    pub two_phase_clip: bool,
    pub rationale: String,

    // NEW:
    /// Per-layer mode overrides. `None` ⇒ every layer uses `mode` (today's
    /// behavior, byte-identical). `Some(v)` ⇒ layer `i` uses `v[i]`; `mode`
    /// becomes an informational default.
    pub per_layer_mode: Option<Vec<FaseMode>>,
    /// WGGO recommendations that FASE had to clamp due to global
    /// feasibility (e.g. Lion optimizer forbids Deferred). Empty when no
    /// overrides were supplied or all were applied verbatim.
    pub override_diagnostics: Vec<crate::wggo_overrides::OverrideDiagnostic>,
}
```

### 3.2 New diagnostic reject reason

File: `crates/nsl-codegen/src/wggo_overrides.rs`

Add to `OverrideRejectReason`:

```rust
/// WGGO requested Deferred mode on a layer whose global FASE plan is
/// FullBuffer because the optimizer does not support deferred accumulation
/// (Lion, Unknown, or AdamW/Adam with allow_v_approx=false).
FaseModeInfeasible {
    optimizer: crate::fase::FaseOptimizer,
    global_mode: crate::fase::FaseMode,
},
```

### 3.3 Mode-lookup helper on `FasePlan`

```rust
impl FasePlan {
    /// Returns the mode for layer `i`. Falls back to `self.mode` when no
    /// per-layer override vector is present. Callers that iterate layers
    /// without knowing whether overrides exist should use this helper.
    pub fn mode_for_layer(&self, i: usize) -> FaseMode {
        self.per_layer_mode
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(self.mode)
    }
}
```

`get(i)` (not indexing) protects against layer-count mismatches; out-of-range falls back to global mode rather than panicking.

## 4. New Planner Entry Point

```rust
pub fn plan(cfg: &FaseConfig) -> FasePlan                           // unchanged
pub fn plan_with_overrides(                                          // new
    cfg: &FaseConfig,
    wggo_fused_per_layer: &[bool],
) -> FasePlan
```

### 4.1 Behavior

`plan_with_overrides` delegates to `plan(cfg)` to compute the global plan, then post-processes:

1. **Empty overrides** (`wggo_fused_per_layer.is_empty()`) → return the global plan unchanged. `per_layer_mode = None`. Preserves the byte-identical fallback.
2. **Global mode is `Passthrough`** (accumulation=1) → all overrides ignored; `per_layer_mode = None`. FASE isn't rewriting the backward at all, so there's nothing to vary per-layer. No diagnostic; this is an expected no-op.
3. **Global mode is `Deferred`** → per-layer overrides are feasible in either direction. `fase_fused=true → Deferred`, `fase_fused=false → FullBuffer`. `per_layer_mode = Some(...)`.
4. **Global mode is `FullBuffer`** (Lion / Unknown / `allow_v_approx=false`) → layers requesting Deferred (`fase_fused=true`) clamp to FullBuffer and emit `FaseModeInfeasible`. Layers requesting FullBuffer are applied verbatim. `per_layer_mode = Some(...)`.

### 4.2 Diagnostic emission

For each clamped override, append to `override_diagnostics`:

```rust
OverrideDiagnostic {
    layer_index: i as u32,
    layer_name: format!("layer_{i}"),   // upstream plumbing provides names
    reason: OverrideRejectReason::FaseModeInfeasible {
        optimizer: cfg.optimizer,
        global_mode: global_plan.mode,
    },
    requested: "Deferred".into(),
    applied: "FullBuffer".into(),
}
```

(Layer names come from the caller in the wiring step — see §6.)

### 4.3 Empty-vs-passthrough distinction

Both `wggo_fused_per_layer.is_empty()` and `global_mode == Passthrough` produce `per_layer_mode = None`, but they're semantically different:
- Empty input → caller didn't supply overrides.
- Passthrough global → FASE is inactive, so overrides are meaningless.

Neither emits a diagnostic; both are no-ops. The distinction is only visible in the `rationale` string.

## 5. Consumer Updates

### 5.1 `fase_memory.rs` — per-layer schedule

Today the scheduler runs `match plan.mode { ... }` once and computes global peak memory. New shape: iterate per layer, reading `plan.mode_for_layer(i)`. Deferred layers contribute one-gradient-at-a-time bytes; FullBuffer layers contribute full gradient-buffer bytes.

```rust
let per_layer_bytes: Vec<u64> = (0..num_layers).map(|i| {
    match plan.mode_for_layer(i) {
        FaseMode::Passthrough => /* standard grad bytes */,
        FaseMode::Deferred    => /* one-grad-at-a-time bytes */,
        FaseMode::FullBuffer  => /* full grad-buffer bytes */,
    }
}).collect();
let peak = per_layer_bytes.iter().max().copied().unwrap_or(0);
```

Existing tests that pass `plan.mode == Deferred` globally continue to work — `mode_for_layer` returns the global mode when `per_layer_mode` is `None`.

### 5.2 `stmt_fase.rs` — per-layer backward codegen

The per-layer backward loop currently branches on `plan.mode` for the Deferred vs FullBuffer code path. Change the condition to `plan.mode_for_layer(i)`. No other logic changes — the recipe, phase sequence, and clip handling are all global and stay shared.

## 6. Wiring

### 6.1 Translation layer — WGGO → FASE input

In `stmt.rs`, at the FASE call site (currently calls `fase::plan(&cfg)`), add upstream:

```rust
let fase_plan = match self.wggo_overrides.as_ref() {
    Some(o) => {
        let fused: Vec<bool> = o.per_layer.iter().map(|p| p.fase_fused).collect();
        crate::fase::plan_with_overrides(&fase_cfg, &fused)
    }
    None => crate::fase::plan(&fase_cfg),
};
```

No new `Compiler` field. The existing FASE plan lifecycle absorbs the overrides — consumers (`fase_memory`, `stmt_fase`) read through `mode_for_layer`.

### 6.2 Stderr diagnostic renderer

Matching the CSHA/WRGA/CPDT precedent, after `plan_with_overrides` returns:

```rust
for diag in &fase_plan.override_diagnostics {
    let reason_str = match &diag.reason {
        OverrideRejectReason::FaseModeInfeasible { optimizer, global_mode } =>
            format!("{:?}_optimizer_global_mode_{:?}", optimizer, global_mode).to_lowercase(),
        other => format!("{:?}", other),
    };
    eprintln!(
        "[fase] layer:{} wggo-override-rejected requested={} applied={} reason={}",
        diag.layer_index, diag.requested, diag.applied, reason_str
    );
}
```

## 7. Testing

### 7.1 `fase.rs` unit tests (5)

1. `plan_with_overrides_empty_input_matches_plan` — calling with `&[]` produces a `FasePlan` byte-identical to `plan(cfg)` (except `per_layer_mode=None`, `override_diagnostics` empty).
2. `plan_with_overrides_passthrough_ignores_all` — `accumulation=1` + non-empty overrides → `per_layer_mode=None`, zero diagnostics.
3. `plan_with_overrides_adamw_mixes_modes` — AdamW/Deferred global + `[true, false, true, false]` → `per_layer_mode = Some(vec![Deferred, FullBuffer, Deferred, FullBuffer])`, zero diagnostics.
4. `plan_with_overrides_fullbuffer_global_clamps_deferred` — Lion/FullBuffer global + `[true, false]` → `per_layer_mode = Some(vec![FullBuffer, FullBuffer])`, 1 diagnostic on layer 0 with `FaseModeInfeasible { optimizer: Lion, global_mode: FullBuffer }`.
5. `plan_with_overrides_allow_v_approx_false_clamps` — AdamW + `allow_v_approx=false` (FullBuffer global) + `[true]` → clamp + diagnostic.

### 7.2 `fase_memory.rs` tests (2)

6. `per_layer_modes_produce_accurate_peak` — 4 layers, `per_layer_mode = [Deferred, FullBuffer, Deferred, FullBuffer]` — peak memory equals max of per-layer footprints (not sum), and FullBuffer layers dominate.
7. `all_deferred_matches_global_deferred_schedule` — `per_layer_mode = Some(vec![Deferred; N])` produces the same schedule as `per_layer_mode = None` with global `Deferred`.

### 7.3 `stmt_fase.rs` codegen test (1)

8. `per_layer_mode_routes_backward_codepath` — compile a 4-layer fixture with WGGO overrides `[true, false, true, false]`. Inspect the emitted Wengert list / IR to assert layers 0 and 2 use the Deferred code path, layers 1 and 3 use FullBuffer. If Wengert-inspection is awkward, fall back to a weaker test: `plan.mode_for_layer(i)` matches expectations after the pipeline runs.

### 7.4 Integration (1)

9. `fase_stderr_diagnostic_format_matches_peer_consumers` — compile with Lion optimizer + WGGO recommending Deferred on one layer. Assert stderr contains the `[fase] layer:N wggo-override-rejected requested=Deferred applied=FullBuffer reason=...` format.

**Total: 9 new tests** (5 planner + 2 memory + 1 codegen + 1 integration).

## 8. Architecture Diagram

```
┌────────────────────────────────┐
│  WGGO AppliedPlan              │
│  per-layer PerLayerOverride {  │
│     fase_fused: bool,          │
│     ...                        │
│  }                             │
└──────────────┬─────────────────┘
               │
               ▼ .per_layer.iter().map(|p| p.fase_fused).collect::<Vec<bool>>()
┌────────────────────────────────┐
│  stmt.rs FASE call site        │
│     if wggo_overrides.is_some()│
│       plan_with_overrides(...) │
│     else                       │
│       plan(...)                │
└──────────────┬─────────────────┘
               │
               ▼
┌────────────────────────────────┐   ┌──────────────────────────────┐
│  FasePlan {                    │──▶│  fase_memory.rs              │
│     mode: <global default>,    │   │    per-layer schedule via    │
│     per_layer_mode: Some(...), │   │    plan.mode_for_layer(i)    │
│     override_diagnostics: ...  │   └──────────────────────────────┘
│  }                             │
└──────────────┬─────────────────┘
               │                   ┌──────────────────────────────┐
               ├──────────────────▶│  stmt_fase.rs backward loop  │
               │                   │    per-layer code path via   │
               │                   │    plan.mode_for_layer(i)    │
               │                   └──────────────────────────────┘
               ▼
┌────────────────────────────────┐
│  Stderr renderer                │
│    [fase] layer:N               │
│    wggo-override-rejected ...   │
└────────────────────────────────┘
```

## 9. Risks & Open Questions

- **Risk:** `fase_memory.rs` may need a `num_layers` input that today is implicit in the global schedule shape. Mitigation: pass it through; if the existing function already takes a layer count or iterates a layer vector, this is a no-op.
- **Risk:** Layer-index alignment. WGGO's `AppliedPlan.layers[i]` indexes layers in whatever order WGGO discovered them. If FASE's codegen indexes layers in a different order, the overrides land on the wrong layers. Mitigation: assert `overrides.len() == fase_memory_layer_count` at the call site; hard-error on mismatch.
- **Open:** Does `stmt_fase.rs` need to know layer indices to make `mode_for_layer(i)` work? If it iterates the Wengert list rather than explicit layer indices, we need a layer-index lookup (or a mapping table). Defer the exact plumbing to the implementation plan.

## 10. Success Criteria

1. `plan_with_overrides(cfg, &[])` returns a plan byte-identical to `plan(cfg)` (modulo two new empty fields).
2. WGGO's `fase_fused: true` on a layer produces `Deferred` in `plan.per_layer_mode[i]` when feasible.
3. Lion optimizer + Deferred request produces a `FaseModeInfeasible` diagnostic and clamps to `FullBuffer`.
4. `fase_memory.rs` peak memory reflects per-layer mode choices (mixed plan has lower peak than all-FullBuffer, higher peak than all-Deferred).
5. `stmt_fase.rs` emits the correct code path per layer.
6. Stderr format matches CSHA/WRGA/CPDT precedent exactly: `[fase] layer:N wggo-override-rejected requested=... applied=... reason=...`.
7. All 9 new tests pass; no existing tests regress.
8. Memory file `project_wggo_consumers.md` updated to mark Consumer 3 fully shipped; only Prune pending.
