# FASE Training-Report CLI — Design

**Date:** 2026-04-14
**Status:** Design approved, ready for implementation plan
**Depends on:** Items #1-#5 (all planner + codegen work that produces the data being reported)
**Scoped as:** Item #6 of the FASE roadmap (final observability item)

## Context

Items #1-#5 of the FASE roadmap built the compiler machinery that makes gradient accumulation, two-phase clipping, and consume-per-parameter gradient release work correctly.  Each of those items produces `Serialize`-annotated data (`FasePlan`, `ClipPlan`, `PcaDetection`, `fase_memory::Schedule`) that today is used only internally by codegen.

Item #6 surfaces that data via `nsl check --training-report[=<format>]`, producing a human-readable (or JSON) audit of the compiler's decisions for every train block in the file.  This is pure observability — no cost models, no hardware targeting, no estimation.  Every field in the report is a directly-traceable output of a pure-data planner already shipped in items #1-#5.

Cost-model outputs like "estimated peak VRAM on RTX 5070 Ti" and "+35% training throughput" belong in `nsl profile`, a separate predictive-profiler project with its own predicted-vs-actual feedback loop.  Mixing those estimates into `nsl check`'s decision-audit would erode trust in both tools when the numbers inevitably diverge from measured reality.

## Goals

1. Add a `--training-report[=<format>]` flag to the `nsl check` CLI subcommand. Values: `text` (default when flag present without value) or `json`.
2. Walk the parsed AST for all `train(...)` blocks.  For each, extract the compiler's planner decisions (FASE mode, clip plan, PCA detection, memory schedule) using the pure-data planners already shipped.
3. Format the collected data as plain text (default) or JSON (opt-in) on stdout.
4. Preserve `nsl check`'s existing exit behavior — the report is observational and does not affect the check's pass/fail.
5. Do NOT implement: hardware targeting, cost-model estimation, peak-VRAM prediction, throughput estimation, or cross-file analysis.

## Non-Goals

- Hardware database (`--target rtx5070ti` → GPU properties lookup).  Belongs in `nsl profile`.
- Peak-VRAM estimator or throughput estimator.  Belongs in `nsl profile` or an M37 Roofline milestone.
- Interpreting train blocks without successful semantic analysis.  `nsl check` already errors; `--training-report` is only attempted on success.
- Reporting on models without a train block (e.g., a file that only declares a model and no train block).  Output says "No train blocks found" and exits 0.
- Cross-file dataset resolution.  If a train block references a dataset declared in another file, the report notes the dataset as "unresolved" rather than traversing imports.
- Modifying or adding to the existing planner output.  Item #6 is reporting only.

## Design Decisions

### D1. Flag surface

Add a `TrainingReportFormat` `ValueEnum` to the `nsl-cli` crate:

```rust
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingReportFormat {
    Text,
    Json,
}
```

Extend the `Check` subcommand with:

```rust
/// Emit a training-pipeline decision audit for every train block in the file.
/// Pass without value for text output, or `--training-report=json` for JSON.
#[arg(long, num_args = 0..=1, default_missing_value = "text")]
training_report: Option<TrainingReportFormat>,
```

`clap`'s `num_args = 0..=1` + `default_missing_value` enables the `--training-report` vs `--training-report=<format>` ergonomic.

### D2. Report data structure

Lives in a new module `crates/nsl-codegen/src/training_report.rs`:

```rust
use crate::{fase, fase_memory, pca_detect};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TrainingReport {
    pub source_path: String,
    pub train_blocks: Vec<TrainBlockReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainBlockReport {
    /// Model identifier as it appears in the train block's `model = <ident>` arg.
    pub model_name: Option<String>,
    pub fase: FaseSection,
    pub pca: Option<PcaSection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FaseSection {
    pub plan: fase::FasePlan,
    pub memory: fase_memory::Schedule,
}

#[derive(Debug, Clone, Serialize)]
pub struct PcaSection {
    pub detection: pca_detect::PcaDetection,
    pub packing_config: pca_detect::DatasetPackingConfig,
}

impl std::fmt::Display for TrainingReport { /* text formatter */ }
```

All fields derive `Serialize`.  The few internal planner structs that may not already derive `Serialize` (inspect during Task 1 of the plan) get the derive added.

### D3. AST walker

The planners are pure data computations.  No codegen required.  A small module-level function:

```rust
pub fn build_report(
    ast: &nsl_ast::Program,
    source_path: &std::path::Path,
) -> TrainingReport
```

walks `ast`'s top-level items looking for `train(...)` blocks.  For each:

1. Extract `grad_accumulation` (integer literal; default 1 if absent).
2. Extract the optimizer call's identifier and keyword args to construct a `FaseConfig`.
3. Extract `grad_clip` (float literal; `None` if absent).
4. If a `data: source = <ident>` line is present, walk the AST for a top-level `dataset <ident>(...)` block and extract its packing config.
5. Call `fase::plan(&config)` → `FasePlan`.
6. Call `fase_memory::schedule(&plan, <param-count-estimate>)` → `Schedule`.
7. If packing is enabled in the dataset, call `pca_detect::detect(&packing_config)` → `PcaDetection`.
8. Assemble into a `TrainBlockReport`.

Where any field can't be resolved (e.g., `grad_accumulation` is a variable rather than a literal), note it as "unresolved" in the output rather than failing.

### D4. Text output shape

The text formatter produces sections separated by blank lines.  Example:

```
=== Training Pipeline Report ===
File: pretrain.nsl
Train blocks found: 1

[Block 1]
  Model: NSLCoder

  FASE (Fused Accumulation-Step Elimination):
    grad_accumulation: 4
    optimizer:         AdamW
    mode:              Deferred
    rationale:         AdamW supports deferred first-moment accumulation with batch-variance v approximation
    backward_phases:   AccumulateOnly × 3, FinalFused × 1
    two_phase_clip:    false
    memory:
      standard peak: 1,048,576 bytes (gradient buffer holds all N grads)
      FASE peak:     524,288 bytes (one grad live at a time)
      savings:       524,288 bytes (50.0%)

  PCA (Packed Causal Attention):
    packing: disabled
```

When `packing = true`:

```
  PCA (Packed Causal Attention):
    packing:              enabled
    strategy:             SegmentIdMasked
    max_sequence_length:  1024
    mean_doc_length:      340
    doc_length_stddev:    85
```

When no train blocks:

```
=== Training Pipeline Report ===
File: model_only.nsl
Train blocks found: 0

No train blocks found in model_only.nsl.
```

The text formatter is 2-column aligned at the 22-char colon position; the memory sub-block is 2-space indented under its parent.

### D5. JSON output

`serde_json::to_string_pretty(&report)` — directly serialized from the same `TrainingReport` struct.  One line of formatter code.  `serde_json` is already a workspace dependency (grep confirms during Task 1).

### D6. CLI plumbing

The `Check` handler in `crates/nsl-cli/src/main.rs`:

1. Runs the existing parse + semantic pipeline.
2. If that fails: return non-zero exit as today.
3. If `training_report` is `Some(format)`:
   - Call `nsl_codegen::training_report::build_report(&ast, &path)`.
   - Match on `format`:
     - `Text` → `println!("{}", report);`
     - `Json` → `println!("{}", serde_json::to_string_pretty(&report).unwrap());`
4. Exit 0.

### D7. Components touched

| File | Change |
|---|---|
| `crates/nsl-codegen/src/training_report.rs` (new) | Report structs, AST walker, text `Display` impl. ~200 LOC. |
| `crates/nsl-codegen/src/lib.rs` | `pub mod training_report;`. |
| `crates/nsl-codegen/src/fase.rs`, `fase_memory.rs`, `pca_detect.rs` | Add `Serialize` derive where missing.  Verify during Task 1; likely only 1-2 types need it. |
| `crates/nsl-cli/src/main.rs` | `TrainingReportFormat` enum, `--training-report` arg on `Check` subcommand, dispatch. ~40 LOC. |
| `crates/nsl-codegen/tests/fixtures/training_report_*.nsl` (3 new) | FASE Deferred fixture, FullBuffer-fallback fixture (Lion), PCA-with-packing fixture. |
| `crates/nsl-codegen/tests/training_report_test.rs` (new) | Integration tests — spawn `nsl check --training-report` and assert structural content (text + JSON). ~120 LOC. |
| memory note | Mark item #6 shipped. |

Total: ~360 LOC + 3 fixtures.

## Architecture

### Data flow

```
nsl check --training-report=text pretrain.nsl
  │
  ├─ lex + parse pretrain.nsl → AST
  ├─ semantic analysis → pass/fail
  │     │
  │     └─ fail: exit non-zero (unchanged behavior)
  │
  └─ training_report::build_report(&ast, &path):
        │
        ├─ for each top-level `train(...)` block in AST:
        │     extract config → FaseConfig + Option<DatasetPackingConfig>
        │     fase::plan(&config)         → FasePlan
        │     fase_memory::schedule(...)  → Schedule
        │     if packing: pca_detect::detect(...)  → PcaDetection
        │     push TrainBlockReport into report.train_blocks
        │
        └─ return TrainingReport
  │
  └─ match format:
        Text → println!("{}", report);
        Json → println!("{}", serde_json::to_string_pretty(&report)?);
  │
  └─ exit 0
```

### Invariants

- **Every reported field has a pure-data source.** No estimation, no lookup tables.
- **`nsl check`'s exit code is unchanged.** The flag only affects what's printed, not whether the check succeeds.
- **Report is emitted only on semantic-check success.** A failing file produces compiler errors, not a misleading partial report.
- **Graceful degradation on unresolvable fields.** Unresolved values are surfaced as "unresolved" in both text and JSON rather than causing a panic or non-zero exit.

## Testing

1. **Unit test (text formatter):** Construct a `TrainingReport` in Rust with a known FASE Deferred AdamW plan.  Assert the rendered string contains `"FASE mode: Deferred"`, `"grad_accumulation: 4"`, `"optimizer:         AdamW"`, and the memory savings lines.
2. **Unit test (JSON round-trip):** `serde_json::to_string_pretty(&report)` → `serde_json::from_str::<TrainingReport>`.  Assert equal-value round-trip.
3. **Integration test — FASE Deferred:** Spawn `nsl check --training-report <fixture>` via `env!("CARGO_BIN_EXE_nsl")`.  Capture stdout; assert presence of `"FASE mode: Deferred"`, the rationale string, and the memory section.
4. **Integration test — FullBuffer fallback:** Fixture uses `optimizer: Lion`.  Assert stdout contains `"FASE mode: FullBuffer"` and the Lion-specific rationale.
5. **Integration test — PCA section:** Fixture has a dataset block with `packing = true`.  Assert stdout contains `"PCA strategy: SegmentIdMasked"` (or whichever strategy the detector picks) and the packing config fields.
6. **Integration test — no train blocks:** Fixture contains only model declarations.  Assert stdout contains `"No train blocks found"` and exit code is 0.
7. **Integration test — JSON format:** Spawn `nsl check --training-report=json <fixture>`.  Pipe stdout through `serde_json::from_str::<serde_json::Value>` to verify valid JSON.  Assert `report["train_blocks"][0]["fase"]["plan"]["mode"]` equals `"Deferred"`.
8. **Exit-code test:** For a file that fails semantic analysis, `nsl check --training-report <fixture>` must still exit non-zero.

All integration tests use structural assertions (`contains`, JSON field access) rather than byte-exact snapshot matching, so format refinements don't cascade into test churn.

## Risks

1. **AST walker's train-block config parser.** The existing config-args parser lives deep in codegen's `stmt.rs` at around line 3012-3037.  Duplicating parsing in the report walker creates maintenance debt.  Mitigation during implementation: if the existing parser is pure and accessible, call it; if not, extract a shared helper that both codegen and the report walker call.  Implementation plan's Task 1 scouts this.
2. **Dataset-to-train-block linkage.** The `data: source = <ident>` reference resolution is currently only done in codegen.  Report walker needs a simpler lookup: find a top-level `dataset <ident>(...)` block in the same AST.  Cross-file references are out of scope; report "unresolved" for those.
3. **Serialize-derive gaps.** `FasePlan` and `PcaDetection` are annotated with `Serialize` per earlier exploration.  `fase_memory::Schedule` may not be.  Task 1 of the plan verifies; adds derives where needed.
4. **Clap flag ergonomics.** The `--training-report` vs `--training-report=json` syntax needs `num_args = 0..=1` plus `default_missing_value`.  Verify during implementation that the resulting clap-generated help text is unambiguous.

## Success Criteria

- `nsl check --training-report <fixture>` emits a well-formed text report on stdout matching the shape in D4.
- `nsl check --training-report=json <fixture>` emits valid JSON that round-trips through `serde_json`.
- Exit code of `nsl check` is unchanged by the flag's presence or absence.
- All 8 tests from the Testing section pass.
- Running `nsl check` without the flag produces byte-identical output to pre-item-#6 behavior.

## Follow-Ups

Item #6 closes the FASE roadmap.  All six items (#1 Deferred codegen, #2 numerical validation, #3 two-phase clip, #4 consume-per-param hook, #5 peak-memory regression, #6 training-report CLI) are shipped or scheduled.

Future-adjacent work:
- `nsl profile` — the predictive profiler that produces estimated peak-VRAM and throughput numbers with a predicted-vs-actual feedback loop.  Separate project.
- General-purpose M36 memory planner — subsumes item #4's hook as a special case when the full planner lands.  Separate project.
