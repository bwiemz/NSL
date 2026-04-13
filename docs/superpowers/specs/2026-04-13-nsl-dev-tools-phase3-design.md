# NSL Dev Tools — Phase 3 WGGO Decision Explainer Design

**Date:** 2026-04-13
**Status:** Design approved, ready for implementation plan
**Builds on:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase2-5-design.md`
**Branch (continued):** `feat/dev-tools-phase1`
**Source:** `nsl dev tools.pdf` §3.3

## 1. Purpose

Implement the WGGO Decision Explainer (`nsl profile --explain-wggo`) — the third sub-feature of Tool 1 in the research document. For each model layer, render the WGGO/ILP solver's per-sub-decision rationale: which CEP head pruning, CSHA level, WRGA adapter rank, and CPDT precision was chosen and *why*, plus how any cross-decision conflicts were resolved.

After Phase 3, every WGGO decision the compiler makes is auditable from a single CLI invocation — closing the "torch.compile is opaque" gap the PDF cites.

## 2. Scope

**In:**

1. `DecisionTrace` struct + `DecisionKind` enum, populated by `solve_layer()` in `wggo_ilp.rs` for all four sub-decisions (CEP, CSHA, WRGA, CPDT) per layer.
2. New field `decision_trace: Vec<DecisionTrace>` on `LayerIlpSolution`.
3. Renderer `wggo_explain::render_explain(&WggoPlan) -> String` matching PDF §3.3 layout, including conflict resolutions from the existing `Resolution` enum.
4. CLI flag `--explain-wggo` on `nsl profile`. When set, forces `WggoMode::Full` and appends the explainer block below the per-op table.
5. JSON mode: `--json --explain-wggo` attaches the full `WggoPlan` under `report.wggo_explain`.

**Out:**

- Layer filtering (`--layer 3`).
- HTML output.
- Greedy / Off mode rationale (those modes don't run the ILP).
- Explanations for non-WGGO compile passes.

## 3. Architecture

Three discrete components.

| Component                          | Location                                                            |
|------------------------------------|---------------------------------------------------------------------|
| `DecisionTrace` + solver instrumentation | `crates/nsl-codegen/src/wggo_ilp.rs` (or sibling `wggo_trace.rs`) |
| Renderer                           | `crates/nsl-cli/src/wggo_explain.rs` (new)                          |
| CLI plumbing                       | `crates/nsl-cli/src/profile.rs` + `main.rs::Profile` arm            |

### 3.1 Reused Phase 1/2 infrastructure

- `nsl_codegen::wggo::WggoPlan { decisions, per_layer, resolutions, applied, ... }` (Phase 2 survey confirmed full struct).
- `LayerIlpSolution { LayerDecision, ... }` — Phase 3 adds `decision_trace` field.
- `Resolution` enum (`DowngradeCsha`, `RemoveWrgaAdapter`, `DeferFaseStep`, `AcceptNonUniformShard`, `NoChange`) — already populated by the conflict pass; renderer consumes as-is.
- `nsl_codegen::profiling::types::ProfileReport` — gains optional `wggo_explain: Option<WggoPlan>` field for JSON mode.
- `ProfileArgs` (Phase 1 Task 4) — gains `explain_wggo: bool`.

## 4. Component Designs

### 4.1 `DecisionTrace` + solver instrumentation

In `crates/nsl-codegen/src/wggo_ilp.rs`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionKind {
    CepHeadPrune,
    CshaLevel,
    WrgaAdapter,
    CpdtPrecision,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    pub kind: DecisionKind,
    pub chosen: String,
    pub runner_up: Option<String>,
    pub binding_constraint: Option<String>,
    pub metric_summary: String,
    pub cross_decision_note: Option<String>,
}
```

Add field to `LayerIlpSolution`:

```rust
pub decision_trace: Vec<DecisionTrace>,
```

Populate in `solve_layer()` after each sub-decision is finalized:

- **CEP head prune:**
  - `chosen = format!("Pruned {}/{} heads ({})", pruned_count, total_heads, pruned_indices_str)`
  - `metric_summary` lists importance values of the pruned heads (e.g. `"importance(h3)=0.12, importance(h6)=0.15 — lowest in layer"`).
  - `binding_constraint = Some(format!("min_retained_importance ≥ {:.2} — satisfied at {:.2}", threshold, achieved))`.
  - `runner_up = Some(format!("Prune {} heads (cost {:.1}μs)", next_count, next_cost))`.
- **CSHA level:**
  - `chosen = format!("Level {}", level)`.
  - `metric_summary = format!("With {} heads, SMEM for L2 = {}KB. Feasible on {} ({}KB limit).", heads, smem_kb, gpu_name, smem_limit_kb)`.
  - `runner_up = Some(format!("Level {} (saves {:.1}μs vs chosen)", alt_level, savings_us))`.
  - `cross_decision_note = Some(format!("L2 saves only {:.1}μs for this layer vs {:.1}μs on unpruned layer. ILP chose L1 because pruning already reduced the layer's cost by {:.0}%", saved, would_save, pct))` when CEP+CSHA interaction matters.
- **WRGA adapter:**
  - `chosen = format!("LoRA r={} on {}", rank, sites_str)`.
  - `metric_summary = format!("Layer {} has {} roofline slack ({}, {:.0}% utilization). Adapter compute is \"free\" — hidden by memory latency.", layer_idx, slack_label, bound, util_pct)`.
  - `binding_constraint = Some(format!("rank ≤ {} (max budget for layer)", rank_budget))`.
- **CPDT precision:**
  - `chosen = format!("INT{} m, FP{} v", m_bits, v_bits)`.
  - `metric_summary = format!("sensitivity({}) = {:.2} (low — middle layer, pruned heads removed the most sensitive components)", layer_idx, sens)`.
  - `binding_constraint = Some(format!("sensitivity ≤ {:.2} → INT{} tier", tier_threshold, m_bits))`.

When `WggoMode` is `Off` or `Greedy`, `decision_trace` stays empty. The renderer detects empty traces and prints a one-line skip message per layer.

#### 4.1.1 `runner_up` is best-effort

`runner_up: Option<String>` is intentionally optional. Populating it requires the ILP solver to retain the second-best candidate's identity and cost. Two ways to get it:

1. **If `solve_layer()` enumerates candidates internally** (likely — single-layer ILP has ~30 variables with pre-computed LUTs, so branch-and-bound already walks the candidate set): keep the top-2 by cost while iterating. Cost: a few lines added to the existing search loop.
2. **If the solver hands back only the optimal**: skip `runner_up` for the first pass — set `runner_up: None` everywhere. The renderer simply omits the `Runner-up:` line when `None`.

Implementer checks the actual solver structure (`crates/nsl-codegen/src/wggo_ilp.rs::solve_layer`) and picks #1 when cheap, #2 when not. **Do not block the phase on getting `runner_up` populated.** The high-value fields are `chosen`, `metric_summary`, `binding_constraint`, and `cross_decision_note` — those alone meet the PDF §3.3 explainability bar. `runner_up` is bonus context.

### 4.2 Renderer — `wggo_explain.rs`

```rust
//! Renders WGGO decision explanations (PDF §3.3) from a WggoPlan.

use nsl_codegen::wggo::{WggoPlan, LayerIlpSolution};
use nsl_codegen::wggo_ilp::{DecisionKind, DecisionTrace};
use nsl_codegen::wggo_conflicts::Resolution;

pub fn render_explain(plan: &WggoPlan) -> String {
    let mut out = String::from("=== WGGO Decision Explanation ===\n\n");
    if plan.per_layer.is_empty() {
        out.push_str("(no layers analyzed — was --wggo full active?)\n");
        return out;
    }
    for (idx, layer) in plan.per_layer.iter().enumerate() {
        render_layer(&mut out, idx, layer);
        for r in &plan.resolutions {
            if resolution_layer_idx(r) == Some(idx) {
                render_resolution(&mut out, r);
            }
        }
        out.push('\n');
    }
    out
}

fn render_layer(out: &mut String, layer_idx: usize, layer: &LayerIlpSolution) {
    out.push_str(&format!("Layer {} decisions:\n", layer_idx));
    if layer.decision_trace.is_empty() {
        out.push_str("  (no decisions traced — wggo mode not Full)\n");
        return;
    }
    for trace in &layer.decision_trace { render_trace(out, trace); }
}

fn render_trace(out: &mut String, t: &DecisionTrace) {
    let label = match t.kind {
        DecisionKind::CepHeadPrune  => "CEP",
        DecisionKind::CshaLevel     => "CSHA",
        DecisionKind::WrgaAdapter   => "WRGA",
        DecisionKind::CpdtPrecision => "CPDT",
    };
    out.push_str(&format!("  {}: {}\n", label, t.chosen));
    out.push_str(&format!("    Reason: {}\n", t.metric_summary));
    if let Some(c) = &t.binding_constraint { out.push_str(&format!("    Constraint: {}\n", c)); }
    if let Some(note) = &t.cross_decision_note { out.push_str(&format!("    BUT: {}\n", note)); }
    if let Some(ru) = &t.runner_up { out.push_str(&format!("    Runner-up: {}\n", ru)); }
    out.push('\n');
}

fn render_resolution(out: &mut String, r: &Resolution) {
    let (header, detail) = match r {
        Resolution::DowngradeCsha { layer: _, to_level }
            => ("CSHA downgrade", format!("→ Level {}", to_level)),
        Resolution::RemoveWrgaAdapter { layer: _ }
            => ("WRGA removed", String::new()),
        Resolution::DeferFaseStep { layer: _ }
            => ("FASE deferred", String::new()),
        Resolution::AcceptNonUniformShard { layer: _ }
            => ("Non-uniform shard accepted", String::new()),
        Resolution::NoChange
            => ("No conflict", String::new()),
    };
    out.push_str(&format!("  Conflict resolved: {} {}\n", header, detail));
}

fn resolution_layer_idx(r: &Resolution) -> Option<usize> {
    match r {
        Resolution::DowngradeCsha { layer, .. } |
        Resolution::RemoveWrgaAdapter { layer } |
        Resolution::DeferFaseStep { layer } |
        Resolution::AcceptNonUniformShard { layer } => Some(*layer as usize),
        Resolution::NoChange => None,
    }
}
```

Add `pub mod wggo_explain;` to `crates/nsl-cli/src/lib.rs`.

### 4.3 CLI plumbing

`ProfileArgs` gains:

```rust
pub explain_wggo: bool,
```

`ProfileReport` (in `crates/nsl-codegen/src/profiling/types.rs`) gains:

```rust
pub wggo_explain: Option<crate::wggo::WggoPlan>,
```

Defaults to `None`; populated only when `--explain-wggo` runs.

`run_profile()` adds, after the per-op walk:

```rust
if args.explain_wggo {
    let wggo_plan = nsl_codegen::wggo::run_wggo(
        &input.module, &input.analysis, &input.interner,
        nsl_codegen::wggo::WggoMode::Full,
        gpu,
    ).map_err(|e| format!("WGGO solve failed: {e}"))?;
    if args.json {
        report.wggo_explain = Some(wggo_plan);
    } else {
        text_output.push_str("\n");
        text_output.push_str(&crate::wggo_explain::render_explain(&wggo_plan));
    }
}
```

The exact `nsl_codegen::wggo::run_wggo` signature may differ — implementer adapts to whatever the existing WGGO entry function is (search for `WggoPlan` constructors in `crates/nsl-codegen/src/wggo*.rs`).

`main.rs` `Profile` variant adds `#[arg(long)] explain_wggo: bool` and threads it through.

## 5. Data flow

```
nsl profile --explain-wggo file.nsl
   │
   ▼
parse + analyze
   │
   ├── walk_ops → ProfileReport
   │
   └── if explain_wggo:
        │
        ▼
        wggo::run_wggo(... WggoMode::Full ...)
          │
          ├── per-layer: solve_layer() builds LayerIlpSolution
          │     ├── for each sub-decision (CEP/CSHA/WRGA/CPDT):
          │     │     push DecisionTrace { ... }
          │     └── decision_trace populated
          │
          └── conflict pass: populates resolutions: Vec<Resolution>
        │
        ▼
        wggo_explain::render_explain(&plan) (text mode)
        OR  report.wggo_explain = Some(plan) (json mode)
        │
        ▼
        text_output += explain_text  (text mode)
   │
   ▼
print or serialize
```

## 6. Error handling

- `WggoMode::Full` solve fails (infeasible constraints, GPU not in DB): hard error from `run_profile`.
- `decision_trace` empty for a layer: renderer prints `"(no decisions traced — wggo mode not Full)"` for that layer. Doesn't error.
- `--explain-wggo` with `--json`: serialize the full `WggoPlan` under `report.wggo_explain`. Already serializable from Phase 2.
- No layers in module: renderer prints `"(no layers analyzed — model has no recognized layer structure)"`.
- `--explain-wggo` without an analyzable model: hard error citing the missing structure.

## 7. Testing

- **Unit (`wggo_ilp.rs`):** `solve_layer` on a 2-layer mock with WGGO Full produces `LayerIlpSolution.decision_trace` with exactly 4 entries (one per `DecisionKind`), each having non-empty `chosen` and `metric_summary`.
- **Unit (`wggo_explain.rs`):** `render_explain` on a hand-built `WggoPlan` with one layer and four traces produces output containing `"Layer 0 decisions:"`, `"CEP:"`, `"CSHA:"`, `"WRGA:"`, `"CPDT:"`, plus the chosen strings and reason lines.
- **Unit (renderer, conflicts):** plan with `Resolution::DowngradeCsha { layer: 4, to_level: 1 }` produces a line containing `"Conflict resolved: CSHA downgrade → Level 1"`.
- **Unit (renderer, empty):** plan with `decision_trace.is_empty()` for a layer prints the skip message; doesn't panic.
- **Integration (`profile_cmd.rs`):** `nsl profile --explain-wggo tiny_transformer.nsl` exits 0; output contains `"=== WGGO Decision Explanation ==="` AND the existing per-op table.
- **Integration (JSON):** `--json --explain-wggo` produces parseable JSON with non-null `wggo_explain.per_layer[0].decision_trace`.
- **Regression:** all Phase 1/2/2.5 tests pass.

## 8. Non-goals

- Layer filtering on the CLI (`--layer 3`).
- HTML / SVG output.
- Greedy / Off mode rationale (those modes don't run the ILP — `decision_trace` stays empty by construction).
- Explanations for non-WGGO compile decisions (epilogue fusion, source-AD rewrites, etc.).
- Cost-of-explanation tracking (the trace adds a small per-layer overhead, not measured here).

## 9. File inventory

**Modified:**

- `crates/nsl-codegen/src/wggo_ilp.rs` — new types + `solve_layer()` instrumentation.
- `crates/nsl-codegen/src/wggo.rs` (or wherever `LayerIlpSolution` lives) — new field.
- `crates/nsl-codegen/src/profiling/types.rs` — `ProfileReport.wggo_explain: Option<WggoPlan>`.
- `crates/nsl-cli/src/profile.rs` — `--explain-wggo` wiring in `run_profile`.
- `crates/nsl-cli/src/main.rs` — clap arg.
- `crates/nsl-cli/src/lib.rs` — `pub mod wggo_explain;`.

**New:**

- `crates/nsl-cli/src/wggo_explain.rs` — renderer.
- Test files: `crates/nsl-codegen/tests/wggo_decision_trace.rs`, `crates/nsl-cli/tests/wggo_explain.rs`.

## 10. Follow-up phases

- **Phase 4** — Training Health Monitor (`nsl run --monitor` live view).
- **Phase 5** — Tensor Inspector (`@inspect` decorator + async dump collector). Stream-aware profiler (the ex-"Phase 2.6" item) folds in here as a prerequisite.
