# NSL Dev Tools — Phase 3 WGGO Decision Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `nsl profile --explain-wggo` per `nsl dev tools.pdf` §3.3 — for each layer, print the per-sub-decision rationale (CEP head pruning, CSHA level, WRGA adapter rank, CPDT precision) plus conflict resolutions, sourced from a new `DecisionTrace` populated during ILP solve.

**Architecture:** `solve_layer()` in `wggo_ilp.rs` populates a `Vec<DecisionTrace>` field on `LayerIlpSolution` covering all four sub-decision kinds. A new `wggo_explain` module in `nsl-cli` renders the `WggoPlan` to PDF §3.3 layout. CLI flag `--explain-wggo` on `nsl profile` triggers a `WggoMode::Full` solve and appends the explainer output below the per-op table; `--json` emits the full `WggoPlan` under `report.wggo_explain` instead.

**Tech Stack:** Rust workspace, `cargo` test, `serde` for plan serialization.

**Spec:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase3-design.md`

**Branch / worktree:** Continue on `feat/dev-tools-phase1` in `c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1`. No new worktree.

---

## Task 1: `DecisionTrace` + `DecisionKind` types and `LayerIlpSolution` field

**Files:**
- Modify: `crates/nsl-codegen/src/wggo_ilp.rs` (or sibling — check where `LayerIlpSolution` lives).
- Test: `crates/nsl-codegen/tests/wggo_decision_trace.rs` (new).

This task adds the data model only — no `solve_layer()` instrumentation yet (that's Task 2). Splits the work so the types ship cleanly first.

- [ ] **Step 1: Locate `LayerIlpSolution`**

```
grep -n "pub struct LayerIlpSolution\|pub LayerDecision" crates/nsl-codegen/src/wggo*.rs
```

Note its file and existing fields. Confirm `WggoPlan.per_layer: Vec<LayerIlpSolution>`.

- [ ] **Step 2: Write failing test**

Create `crates/nsl-codegen/tests/wggo_decision_trace.rs`:

```rust
use nsl_codegen::wggo_ilp::{DecisionKind, DecisionTrace};

#[test]
fn decision_trace_is_constructible_and_serializable() {
    let t = DecisionTrace {
        kind: DecisionKind::CshaLevel,
        chosen: "Level 1".into(),
        runner_up: Some("Level 2 (saves 0.3μs vs chosen)".into()),
        binding_constraint: Some("SMEM ≤ 228 KB".into()),
        metric_summary: "With 6 heads, SMEM for L2 = 84KB. Feasible on H100.".into(),
        cross_decision_note: Some("L2 saves only 1.8μs vs 2.3μs unpruned. Pruning already cut cost 25%.".into()),
    };
    let s = serde_json::to_string(&t).unwrap();
    let back: DecisionTrace = serde_json::from_str(&s).unwrap();
    assert_eq!(back.chosen, "Level 1");
    assert!(matches!(back.kind, DecisionKind::CshaLevel));
}

#[test]
fn decision_kind_round_trips_all_variants() {
    use DecisionKind::*;
    for k in [CepHeadPrune, CshaLevel, WrgaAdapter, CpdtPrecision] {
        let s = serde_json::to_string(&k).unwrap();
        let back: DecisionKind = serde_json::from_str(&s).unwrap();
        assert_eq!(format!("{back:?}"), format!("{k:?}"));
    }
}

#[test]
fn layer_ilp_solution_default_decision_trace_is_empty() {
    use nsl_codegen::wggo::LayerIlpSolution;
    let sol = LayerIlpSolution::default();
    assert!(sol.decision_trace.is_empty());
}
```

If `LayerIlpSolution::default()` doesn't exist or its constructor takes args, adapt to the actual constructor. The assertion is just `decision_trace.is_empty()` after default construction.

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test wggo_decision_trace 2>&1 | tail -15
```

Expected: unresolved import for `DecisionKind` / `DecisionTrace`; missing field on `LayerIlpSolution`.

- [ ] **Step 4: Add the types**

In `crates/nsl-codegen/src/wggo_ilp.rs` (or wherever ILP types live), add at the top of the public types section:

```rust
use serde::{Deserialize, Serialize};

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

- [ ] **Step 5: Add the field to `LayerIlpSolution`**

In whichever file holds `LayerIlpSolution`, add:

```rust
pub decision_trace: Vec<crate::wggo_ilp::DecisionTrace>,
```

(Adjust the path if `DecisionTrace` lives in the same module — use `Vec<DecisionTrace>`.)

Update every `LayerIlpSolution { ... }` literal to initialize the new field to `vec![]`. Find them:

```
grep -rn "LayerIlpSolution {" crates/nsl-codegen/src/
```

If the struct derives `Default`, the new field also needs to default to `vec![]` — which `Vec` does automatically — so existing `..Default::default()` callers stay working.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-codegen --test wggo_decision_trace 2>&1 | tail -10
cargo build -p nsl-codegen 2>&1 | tail -10
cargo test -p nsl-codegen --tests 2>&1 | tail -10
```

Expected: 3 passed; clean build; no regressions.

- [ ] **Step 7: Commit**

```
git add crates/nsl-codegen/src/wggo_ilp.rs \
        crates/nsl-codegen/src/wggo.rs \
        crates/nsl-codegen/tests/wggo_decision_trace.rs \
        $(grep -rln "LayerIlpSolution {" crates/nsl-codegen/src/)
git commit -m "feat(wggo): DecisionTrace types + LayerIlpSolution.decision_trace field"
```

---

## Task 2: Instrument `solve_layer()` to populate `decision_trace`

**Files:**
- Modify: `crates/nsl-codegen/src/wggo_ilp.rs::solve_layer` (or wherever the per-layer ILP runs).
- Test: extend `crates/nsl-codegen/tests/wggo_decision_trace.rs`.

All four sub-decisions instrumented in one task because they're structurally identical (same `DecisionTrace` shape, same write pattern). `runner_up` is best-effort per spec §4.1.1 — if `solve_layer` already enumerates candidates, capture the second-best while iterating; otherwise leave `runner_up: None`.

- [ ] **Step 1: Locate `solve_layer` body**

```
grep -n "fn solve_layer" crates/nsl-codegen/src/wggo_ilp.rs
sed -n '1,40p' crates/nsl-codegen/src/wggo_ilp.rs   # confirm structure / look for candidate enumeration
```

Note where each sub-decision is finalized in the body (likely four discrete blocks: CEP, CSHA, WRGA, CPDT) and whether candidates are enumerated (so `runner_up` is cheap) or pulled from a black-box solver.

- [ ] **Step 2: Write failing test**

Append to `crates/nsl-codegen/tests/wggo_decision_trace.rs`:

```rust
use nsl_codegen::wggo::{run_wggo, WggoMode};
use nsl_codegen::wggo_ilp::DecisionKind;

#[test]
fn solve_layer_populates_four_decision_traces_per_layer() {
    // Build the smallest possible 1-layer model the WGGO solver accepts.
    // Mirror whatever existing wggo unit tests use to construct test inputs.
    let plan = build_minimal_one_layer_plan();
    assert_eq!(plan.per_layer.len(), 1);
    let trace = &plan.per_layer[0].decision_trace;
    // Spec §4.1: all four sub-decisions instrumented.
    assert_eq!(trace.len(), 4, "expected 4 decision traces per layer, got {}", trace.len());

    // Each kind appears exactly once.
    let kinds: Vec<_> = trace.iter().map(|t| format!("{:?}", t.kind)).collect();
    for expected in &["CepHeadPrune", "CshaLevel", "WrgaAdapter", "CpdtPrecision"] {
        assert!(kinds.iter().any(|k| k == expected),
            "missing trace for kind {}, got {:?}", expected, kinds);
    }

    // High-value fields populated.
    for t in trace {
        assert!(!t.chosen.is_empty(), "{:?} missing chosen", t.kind);
        assert!(!t.metric_summary.is_empty(), "{:?} missing metric_summary", t.kind);
    }
}

fn build_minimal_one_layer_plan() -> nsl_codegen::wggo::WggoPlan {
    // Find an existing test helper in crates/nsl-codegen/src/wggo*.rs that
    // builds a small plan, or construct inline using whatever the solver's
    // public API requires (typically: a typed AST, AnalysisResult, GpuSpec,
    // and WggoMode). Reuse the parse_and_analyze helper from
    // crates/nsl-codegen/tests/profiling_walker.rs::parse_and_analyze.
    todo!("construct a minimal WggoPlan for testing; mirror existing wggo unit-test setup")
}
```

The `todo!()` is intentional — the right helper depends on what existing wggo tests use. Implementer reads them and adapts.

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test wggo_decision_trace solve_layer_populates 2>&1 | tail -10
```

Expected: panic from the `todo!()` (or zero-trace assertion failure once the helper is in place).

- [ ] **Step 4: Implement the helper**

Copy whatever pattern existing tests in `crates/nsl-codegen/src/wggo*.rs` (in their `#[cfg(test)] mod tests` blocks) use to build a `WggoPlan`. If none exist, the path is:

```rust
// 1. Use parse_and_analyze (from profiling_walker.rs tests) on a tiny model:
let src = r#"
model Tiny:
    Wq: Tensor<[512, 512], bf16>
    fn forward(self, x: Tensor<[1, 64, 512], bf16>) -> Tensor:
        return x @ self.Wq
"#;
let (m, analysis, interner) = parse_and_analyze(src);
let gpu = nsl_codegen::gpu_specs::find_gpu("h100").unwrap();
// 2. Call run_wggo (or whatever the actual entry function is — search wggo*.rs):
let plan = nsl_codegen::wggo::run_wggo(
    &m, &analysis, &interner, WggoMode::Full, gpu,
).unwrap();
plan
```

If `parse_and_analyze` isn't reachable from here (it's defined inline in `profiling_walker.rs` tests), copy it inline — duplicate is fine for a test helper.

- [ ] **Step 5: Instrument `solve_layer()`**

In `solve_layer()`, after each of the four sub-decisions is finalized, push a `DecisionTrace`. The exact local variables driving each decision depend on the solver's internal structure — read the existing body carefully, then add the four pushes near where the chosen value is computed. Templates (adapt local variable names to what's actually in scope):

**CEP (head pruning):**

```rust
{
    let chosen = format!(
        "Pruned {}/{} heads ({})",
        pruned_count, total_heads,
        pruned_indices.iter().map(|i| format!("h{i}")).collect::<Vec<_>>().join(", ")
    );
    let metric_summary = format!(
        "{} — lowest in layer",
        pruned_indices.iter()
            .map(|&i| format!("importance(h{i})={:.2}", head_importance[i]))
            .collect::<Vec<_>>().join(", ")
    );
    let binding_constraint = Some(format!(
        "min_retained_importance ≥ {:.2} — satisfied at {:.2}",
        cep_min_retained, cep_achieved_retained
    ));
    let runner_up = cep_runner_up_cost.map(|c|
        format!("Prune {} heads (cost {:.1}μs)", cep_runner_up_count, c));
    layer_solution.decision_trace.push(DecisionTrace {
        kind: DecisionKind::CepHeadPrune,
        chosen,
        runner_up,
        binding_constraint,
        metric_summary,
        cross_decision_note: None,
    });
}
```

**CSHA:**

```rust
{
    let chosen = format!("Level {}", csha_level);
    let metric_summary = format!(
        "With {} heads, SMEM for L2 = {}KB. Feasible on {} ({}KB limit).",
        retained_heads, csha_l2_smem_kb, gpu.name, gpu.shared_mem_per_block_kb
    );
    let runner_up = csha_runner_up.map(|(lvl, savings_us)|
        format!("Level {} (saves {:.1}μs vs chosen)", lvl, savings_us));
    let binding_constraint = Some(format!(
        "SMEM ≤ {}KB", gpu.shared_mem_per_block_kb
    ));
    let cross_decision_note = csha_pruning_interaction.map(|note| note.to_string());
    layer_solution.decision_trace.push(DecisionTrace {
        kind: DecisionKind::CshaLevel,
        chosen, runner_up, binding_constraint, metric_summary, cross_decision_note,
    });
}
```

**WRGA:**

```rust
{
    let chosen = format!(
        "LoRA r={} on {}",
        wrga_rank,
        wrga_sites.iter().cloned().collect::<Vec<_>>().join(", ")
    );
    let metric_summary = format!(
        "Layer {} has high roofline slack ({}, {:.0}% utilization). Adapter compute is \"free\" — hidden by memory latency.",
        layer_idx,
        if is_memory_bound { "memory-bound" } else { "compute-bound" },
        utilization_pct
    );
    let binding_constraint = Some(format!("rank ≤ {} (max budget for layer)", wrga_rank_budget));
    let runner_up = None;  // typically not enumerated
    layer_solution.decision_trace.push(DecisionTrace {
        kind: DecisionKind::WrgaAdapter,
        chosen, runner_up, binding_constraint, metric_summary, cross_decision_note: None,
    });
}
```

**CPDT:**

```rust
{
    let chosen = format!("INT{} m, FP{} v", cpdt_m_bits, cpdt_v_bits);
    let metric_summary = format!(
        "sensitivity({}) = {:.2} (low — middle layer, pruned heads removed the most sensitive components)",
        layer_idx, cpdt_sensitivity_score
    );
    let binding_constraint = Some(format!(
        "sensitivity ≤ {:.2} → INT{} tier",
        cpdt_tier_threshold, cpdt_m_bits
    ));
    layer_solution.decision_trace.push(DecisionTrace {
        kind: DecisionKind::CpdtPrecision,
        chosen, runner_up: None, binding_constraint, metric_summary, cross_decision_note: None,
    });
}
```

**Adapt local variable names to what's actually in scope.** The names above are illustrative. Where a value isn't computed today (e.g., `cpdt_sensitivity_score` may need to be derived from existing layer metrics), either compute it inline cheaply or substitute a concrete description. The high-value fields are `chosen` and `metric_summary` — make those informative even if the others end up generic.

For `runner_up`: per spec §4.1.1, only populate if the solver already enumerates candidates. Otherwise leave `None`.

Gate the pushes on `mode == WggoMode::Full` so Off / Greedy modes don't pay the cost.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-codegen --test wggo_decision_trace 2>&1 | tail -10
cargo test -p nsl-codegen --tests 2>&1 | tail -10
cargo build -p nsl-codegen 2>&1 | tail -10
```

Expected: the new `solve_layer_populates_four_decision_traces_per_layer` test passes; full suite green.

- [ ] **Step 7: Commit**

```
git add crates/nsl-codegen/src/wggo_ilp.rs \
        crates/nsl-codegen/tests/wggo_decision_trace.rs
git commit -m "feat(wggo): solve_layer instruments four DecisionTrace entries per layer"
```

---

## Task 3: Renderer module `wggo_explain`

**Files:**
- Create: `crates/nsl-cli/src/wggo_explain.rs`.
- Modify: `crates/nsl-cli/src/lib.rs` — add `pub mod wggo_explain;`.
- Test: `crates/nsl-cli/tests/wggo_explain.rs` (new).

- [ ] **Step 1: Write failing test**

Create `crates/nsl-cli/tests/wggo_explain.rs`:

```rust
use nsl_cli::wggo_explain::render_explain;
use nsl_codegen::wggo::{LayerIlpSolution, WggoPlan};
use nsl_codegen::wggo_ilp::{DecisionKind, DecisionTrace};
use nsl_codegen::wggo_conflicts::Resolution;

fn mk_trace(kind: DecisionKind, chosen: &str, reason: &str) -> DecisionTrace {
    DecisionTrace {
        kind,
        chosen: chosen.into(),
        runner_up: None,
        binding_constraint: None,
        metric_summary: reason.into(),
        cross_decision_note: None,
    }
}

fn mk_plan_with_one_layer() -> WggoPlan {
    let mut layer = LayerIlpSolution::default();
    layer.decision_trace.push(mk_trace(DecisionKind::CepHeadPrune, "Pruned 2/8 heads (h3, h6)",
        "importance(h3)=0.12, importance(h6)=0.15 — lowest in layer"));
    layer.decision_trace.push(mk_trace(DecisionKind::CshaLevel, "Level 1",
        "With 6 heads, SMEM for L2 = 84KB. Feasible on H100 (228KB limit)."));
    layer.decision_trace.push(mk_trace(DecisionKind::WrgaAdapter, "LoRA r=8 on Wq, Wk",
        "Layer 0 has high roofline slack (memory-bound, 32% utilization)."));
    layer.decision_trace.push(mk_trace(DecisionKind::CpdtPrecision, "INT8 m, FP16 v",
        "sensitivity(0) = 0.31 (low)"));

    let mut plan = WggoPlan::default();
    plan.per_layer.push(layer);
    plan
}

#[test]
fn renders_layer_header_and_all_four_kinds() {
    let plan = mk_plan_with_one_layer();
    let out = render_explain(&plan);
    assert!(out.contains("=== WGGO Decision Explanation ==="));
    assert!(out.contains("Layer 0 decisions:"));
    assert!(out.contains("CEP: Pruned 2/8 heads (h3, h6)"));
    assert!(out.contains("CSHA: Level 1"));
    assert!(out.contains("WRGA: LoRA r=8 on Wq, Wk"));
    assert!(out.contains("CPDT: INT8 m, FP16 v"));
    assert!(out.contains("Reason: importance(h3)=0.12"));
}

#[test]
fn renders_skip_message_for_empty_layer_trace() {
    let mut plan = WggoPlan::default();
    plan.per_layer.push(LayerIlpSolution::default());  // empty trace
    let out = render_explain(&plan);
    assert!(out.contains("Layer 0 decisions:"));
    assert!(out.contains("(no decisions traced — wggo mode not Full)"));
}

#[test]
fn renders_csha_downgrade_resolution() {
    let mut plan = mk_plan_with_one_layer();
    plan.resolutions.push(Resolution::DowngradeCsha { layer: 0, to_level: 1 });
    let out = render_explain(&plan);
    assert!(out.contains("Conflict resolved: CSHA downgrade → Level 1"));
}

#[test]
fn renders_wrga_removed_resolution() {
    let mut plan = mk_plan_with_one_layer();
    plan.resolutions.push(Resolution::RemoveWrgaAdapter { layer: 0 });
    let out = render_explain(&plan);
    assert!(out.contains("Conflict resolved: WRGA removed"));
}

#[test]
fn empty_plan_renders_helpful_note() {
    let plan = WggoPlan::default();
    let out = render_explain(&plan);
    assert!(out.contains("(no layers analyzed — was --wggo full active?)"));
}
```

If `WggoPlan::default()` or `LayerIlpSolution::default()` doesn't exist, add `#[derive(Default)]` (or impl manually). Resolution variants may use `u32` for `layer` rather than `usize` — adapt the test literals.

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-cli --test wggo_explain 2>&1 | tail -15
```

Expected: unresolved import for `nsl_cli::wggo_explain`.

- [ ] **Step 3: Implement the renderer**

Create `crates/nsl-cli/src/wggo_explain.rs`:

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
    for trace in &layer.decision_trace {
        render_trace(out, trace);
    }
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
    if let Some(c) = &t.binding_constraint {
        out.push_str(&format!("    Constraint: {}\n", c));
    }
    if let Some(note) = &t.cross_decision_note {
        out.push_str(&format!("    BUT: {}\n", note));
    }
    if let Some(ru) = &t.runner_up {
        out.push_str(&format!("    Runner-up: {}\n", ru));
    }
    out.push('\n');
}

fn render_resolution(out: &mut String, r: &Resolution) {
    let (header, detail) = match r {
        Resolution::DowngradeCsha { layer: _, to_level } =>
            ("CSHA downgrade", format!("→ Level {}", to_level)),
        Resolution::RemoveWrgaAdapter { layer: _ } =>
            ("WRGA removed", String::new()),
        Resolution::DeferFaseStep { layer: _ } =>
            ("FASE deferred", String::new()),
        Resolution::AcceptNonUniformShard { layer: _ } =>
            ("Non-uniform shard accepted", String::new()),
        Resolution::NoChange =>
            ("No conflict", String::new()),
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

Add to `crates/nsl-cli/src/lib.rs`:

```rust
pub mod wggo_explain;
```

- [ ] **Step 4: Run tests**

```
cargo test -p nsl-cli --test wggo_explain 2>&1 | tail -15
cargo build -p nsl-cli 2>&1 | tail -5
```

Expected: 5 passed; build clean.

- [ ] **Step 5: Commit**

```
git add crates/nsl-cli/src/wggo_explain.rs \
        crates/nsl-cli/src/lib.rs \
        crates/nsl-cli/tests/wggo_explain.rs
git commit -m "feat(cli): wggo_explain renderer matches PDF §3.3 layout"
```

---

## Task 4: CLI plumbing — `--explain-wggo` flag + `ProfileReport.wggo_explain`

**Files:**
- Modify: `crates/nsl-codegen/src/profiling/types.rs` — `ProfileReport.wggo_explain: Option<WggoPlan>`.
- Modify: `crates/nsl-cli/src/profile.rs` — `ProfileArgs.explain_wggo: bool` + `run_profile` invocation.
- Modify: `crates/nsl-cli/src/main.rs` — clap `#[arg(long)] explain_wggo: bool`.
- Test: extend `crates/nsl-cli/tests/profile_cmd.rs`.

- [ ] **Step 1: Add field to `ProfileReport`**

In `crates/nsl-codegen/src/profiling/types.rs`, add:

```rust
pub wggo_explain: Option<crate::wggo::WggoPlan>,
```

Default to `None` in any constructors. If `WggoPlan` doesn't already implement `Default`, derive it (Phase 2 added Serialize/Deserialize; Default should be feasible).

Update any `ProfileReport { ... }` literals (search via grep) to set `wggo_explain: None`.

- [ ] **Step 2: Add `explain_wggo` to `ProfileArgs`**

In `crates/nsl-cli/src/profile.rs`:

```rust
pub struct ProfileArgs {
    // ...existing fields...
    pub explain_wggo: bool,
}
```

Update the existing test helper `sample_args()` in `crates/nsl-cli/tests/profile_cmd.rs` to set `explain_wggo: false`.

- [ ] **Step 3: Write failing integration tests**

Append to `crates/nsl-cli/tests/profile_cmd.rs`:

```rust
#[test]
fn explain_wggo_flag_appends_explanation_block() {
    let mut args = sample_args();
    args.explain_wggo = true;
    let out = run_profile(&args).expect("profile should succeed");
    assert!(out.contains("=== NSL Predictive Profile ==="),
        "per-op table should still be present (additive flag)");
    assert!(out.contains("=== WGGO Decision Explanation ==="),
        "WGGO explanation block missing — flag did not append");
}

#[test]
fn explain_wggo_with_json_attaches_wggo_plan() {
    let mut args = sample_args();
    args.json = true;
    args.explain_wggo = true;
    let out = run_profile(&args).expect("profile should succeed");
    let v: serde_json::Value = serde_json::from_str(&out).expect("must be valid JSON");
    assert!(v["wggo_explain"].is_object() || v["wggo_explain"].is_null() == false,
        "wggo_explain should be populated in JSON mode, got {:?}", v["wggo_explain"]);
    // per_layer should exist with at least one entry.
    let per_layer = &v["wggo_explain"]["per_layer"];
    assert!(per_layer.is_array(), "per_layer should be array");
    assert!(per_layer.as_array().unwrap().len() >= 1, "expected at least one layer");
}
```

- [ ] **Step 4: Run — expect fail**

```
cargo test -p nsl-cli --test profile_cmd explain_wggo 2>&1 | tail -15
```

- [ ] **Step 5: Implement in `run_profile`**

In `crates/nsl-cli/src/profile.rs::run_profile`, after the per-op walk and before the renderer call, add:

```rust
if args.explain_wggo {
    // Find the actual run_wggo function — search wggo*.rs for entry points.
    // The exact signature may differ from the spec sketch.
    let wggo_plan = nsl_codegen::wggo::run_wggo(
        &input.module,
        &input.analysis,
        &input.interner,
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

If `nsl_codegen::wggo::run_wggo` doesn't exist by that exact name, find the actual entry function:

```
grep -n "fn run_wggo\|pub fn .*WggoPlan" crates/nsl-codegen/src/wggo*.rs
```

Use the real one. The CLI variant may need `target.clone()` instead of `gpu`, etc. — adapt to the existing signature. If the entry function takes a `Wengert` list rather than `(module, analysis, interner)`, build one upstream the same way other profile features do.

- [ ] **Step 6: Add the clap arg in `main.rs`**

In `Cli::Profile { ... }` variant:

```rust
#[arg(long)]
explain_wggo: bool,
```

In the dispatch arm:

```rust
let args = nsl_cli::profile::ProfileArgs {
    // ...existing fields...
    explain_wggo,
};
```

- [ ] **Step 7: Run tests**

```
cargo test -p nsl-cli --test profile_cmd 2>&1 | tail -15
cargo build -p nsl-cli 2>&1 | tail -5
cargo test --workspace --tests -- --test-threads=1 2>&1 | grep -E "^test result:|FAILED" | head -20
```

Expected: green across the workspace.

- [ ] **Step 8: Manual smoke**

```
cargo run -p nsl-cli -- profile --explain-wggo crates/nsl-cli/tests/fixtures/tiny_transformer.nsl | tail -30
cargo run -p nsl-cli -- profile --json --explain-wggo crates/nsl-cli/tests/fixtures/tiny_transformer.nsl | jq '.wggo_explain.per_layer[0].decision_trace | length'
```

Expected: text mode shows the explanation block; JSON mode shows `4` (or whatever the actual decision_trace length is).

- [ ] **Step 9: Commit**

```
git add crates/nsl-codegen/src/profiling/types.rs \
        crates/nsl-cli/src/profile.rs \
        crates/nsl-cli/src/main.rs \
        crates/nsl-cli/tests/profile_cmd.rs
git commit -m "feat(cli): nsl profile --explain-wggo runs WGGO Full + appends explainer"
```

---

## Task 5: Final verification

- [ ] **Step 1: Full workspace test**

```
cargo test --workspace --tests -- --test-threads=1 2>&1 | grep -E "^test result:|FAILED"
```

Expected: all green. Note: the known Windows parallel-file-lock issue on `e2e_m12_grad_basic_source_ad` is avoided by `--test-threads=1`.

- [ ] **Step 2: Acceptance smoke**

```
cargo run -p nsl-cli -- profile --explain-wggo crates/nsl-cli/tests/fixtures/tiny_transformer.nsl
```

Expected: text output ends with a `=== WGGO Decision Explanation ===` block listing per-layer CEP/CSHA/WRGA/CPDT decisions with Reason lines and (where applicable) Constraint, BUT, Runner-up lines.

- [ ] **Step 3: Branch status**

```
cd c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1
git log --oneline main..HEAD | head -5
```

Expected: 4 new Phase 3 commits on top of Phase 2.5.

- [ ] **Step 4: Don't push or open PR**

Per user instruction (2026-04-12), held local until all milestones ship.

---

## Self-Review

**Spec coverage:**

- §4.1 `DecisionTrace` types + `LayerIlpSolution` field → Task 1. ✅
- §4.1 `solve_layer` instrumentation across all four sub-decisions → Task 2. ✅
- §4.1.1 best-effort `runner_up` → Task 2 Step 5 explicitly notes "only populate if solver enumerates candidates; else None". ✅
- §4.2 renderer → Task 3. ✅
- §4.3 CLI plumbing (`--explain-wggo`, `ProfileReport.wggo_explain`, JSON mode) → Task 4. ✅
- §6 error handling — empty per_layer note (Task 3 Step 1 test); empty decision_trace skip (Task 3 Step 1 test); WGGO solve failure propagates as run_profile error (Task 4 Step 5 `?`). ✅
- §7 testing — unit (Tasks 1 + 2 + 3), integration (Task 4), regression (Task 5). ✅
- §8 non-goals respected — no layer filter, no HTML, no Greedy/Off rationale. ✅

**Placeholder scan:**

One `todo!()` in Task 2 Step 2's `build_minimal_one_layer_plan` helper, with explicit guidance to mirror existing wggo unit-test setup. Task 2 Step 4 fills it in. No other "TBD"/"add validation"/vague patterns.

**Type consistency:**

- `DecisionKind`, `DecisionTrace`, `LayerIlpSolution.decision_trace` consistent across Tasks 1, 2, 3.
- `WggoPlan.per_layer` and `WggoPlan.resolutions` consistent (Phase 2 survey-confirmed names).
- `ProfileReport.wggo_explain: Option<WggoPlan>` defined in Task 4 Step 1, populated in Task 4 Step 5, asserted in Task 4 Step 3 JSON test.
- `ProfileArgs.explain_wggo: bool` defined Task 4 Step 2, threaded through Task 4 Step 6.
- Renderer function `render_explain(&WggoPlan) -> String` consistent in Task 3 + Task 4.

**Scope:**

5 tasks (4 impl + 1 verification). Each produces testable working software. Final manual smoke matches PDF §3.3 expected output.
