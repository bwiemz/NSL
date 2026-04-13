# NSL Dev Tools — Phase 2.5 Cleanup Design

**Date:** 2026-04-13
**Status:** Design approved, ready for implementation plan
**Builds on:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase2-design.md`
**Branch (continued):** `feat/dev-tools-phase1`

## 1. Purpose

Close the two Phase 2.5 TODOs that have a tractable fix today:

1. Populate `FusionPlan.fused_node_groups` at the two fusion decision sites that have NodeIds in scope (epilogue fusion, reduction fusion) so fused kernels stop undercounting predictions by 2–4×.
2. Thread source-file context into every compile entry path that sets `profile_kernels=true`, not just `--monitor`, so emitted manifest spans have real line numbers in all cases.

After Phase 2.5, `--profile-kernels` on any compile path produces a manifest whose predictions for epilogue/reduction-fused kernels sum across constituents, and whose spans render against real source lines.

## 2. Scope

**In:**

1. `apply_epilogue_fusion` gains a `&mut FusionPlan` parameter and populates `plan.fused_node_groups` per chain. All call sites updated.
2. `apply_reduction_fusion` gains a `&mut FusionPlan` parameter and populates `plan.fused_node_groups` per match. All call sites updated.
3. `run_profile_pre_pass` stops hard-coding `fusion_plan_for_profile = None`. Reads from `compiler.last_wrga_plan.fusion` when present.
4. `run_profile_pre_pass` source-text fallback: when `profile_source_text` is empty but `profile_source_file_name` is set, read the file from disk.

**Out:**

- WRGA adapter fusion site (`wrga_fusion.rs::build_fusion_plan`). No NodeIds in scope — threading them through `AdapterPlacement` is 5–8 files.
- `FusionPlanBuilder` shared builder refactor.
- Pure-inference fusion-plan carrier.
- Multi-stream profiling.
- Real CUDA event verification on hardware.

## 3. Architecture

Four discrete edits. No new modules.

| Component                              | Location                                                          |
|----------------------------------------|-------------------------------------------------------------------|
| Epilogue-fusion plan populator         | `crates/nsl-codegen/src/epilogue_fusion.rs::apply_epilogue_fusion`|
| Reduction-fusion plan populator        | `crates/nsl-codegen/src/reduction_fusion.rs::apply_reduction_fusion`|
| Pre-pass fusion-plan wire-up           | `crates/nsl-codegen/src/compiler/entry_points.rs::run_profile_pre_pass` |
| Source-text disk fallback              | same function, same pre-pass                                      |

### 3.1 Reused Phase 2 infrastructure

- `FusionPlan { decisions, fused_node_groups: HashMap<NodeId, Vec<NodeId>> }` + `FusionPlan::constituents_of(root)` — Phase 2 Task 3.
- `Compiler::fusion_constituents(root) -> Vec<NodeId>` — Phase 2 Task 4.
- `CompileOptions { profile_source_text, profile_source_file_name, ... }` — Phase 2 Task 6.
- `Compiler::{source_text, source_file_name}` fields populated by pre-pass — Phase 2 Tasks 5/6.

## 4. Component Designs

### 4.1 Epilogue fusion populator

```rust
pub fn apply_epilogue_fusion(
    graph: &mut FusionGraph,
    chains: &[EpilogueChain],
    base_kernel_id: u32,
    plan: &mut FusionPlan,           // NEW
) -> Vec<FusedKernel> {
    let mut out = Vec::new();
    for chain in chains {
        // ...existing body that marks eliminated_nodes and builds fused kernel...

        let mut constituents = Vec::with_capacity(1 + chain.eliminated_nodes.len());
        constituents.push(chain.matmul_node);
        constituents.extend(chain.eliminated_nodes.iter().copied());
        plan.fused_node_groups.insert(chain.matmul_node, constituents);
    }
    out
}
```

All call sites pass a `&mut FusionPlan`. Tests that don't care about the plan use `&mut FusionPlan::default()`.

### 4.2 Reduction fusion populator

```rust
pub fn apply_reduction_fusion(
    graph: &mut FusionGraph,
    matches: &[ReductionMatch],
    base_kernel_id: u32,
    plan: &mut FusionPlan,           // NEW
) -> Vec<FusedKernel> {
    let mut out = Vec::new();
    for m in matches {
        // ...existing body...
        plan.fused_node_groups.insert(m.root_node, m.all_matched_nodes.clone());
    }
    out
}
```

### 4.3 Pre-pass fusion-plan wire-up

In `run_profile_pre_pass`, around the current `compiler.fusion_plan_for_profile = None;` line:

```rust
compiler.fusion_plan_for_profile = compiler.last_wrga_plan
    .as_ref()
    .map(|p| p.fusion.clone());
```

When `last_wrga_plan` is `None` (pure-inference compile), `fusion_plan_for_profile` remains `None` and `fusion_constituents(root)` returns `vec![root]`. For inference-only compiles, epilogue/reduction fusion still slightly under-predicts until Phase 3 adds a pure-inference fusion-plan carrier. Documented as Phase 3 follow-up.

### 4.4 Source-text disk fallback

In `run_profile_pre_pass`, after setting `manifest_builder`:

```rust
compiler.source_text = match &opts.profile_source_text {
    Some(s) => s.clone(),
    None => opts.profile_source_file_name.as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default(),
};
compiler.source_file_name = opts.profile_source_file_name
    .clone()
    .unwrap_or_default();
```

Monitor path unchanged (already sets both). Build path with `profile_kernels=true` now auto-reads the source from disk.

## 5. Data flow

```
opts.profile_kernels = true
  │
  ▼
run_profile_pre_pass:
  ├─ walk_ops → prediction_map
  ├─ manifest_builder = Some(...)
  ├─ source_text = opts.source || fs::read(opts.file) || ""    [§4.4]
  └─ fusion_plan_for_profile = last_wrga_plan.map(|p| p.fusion.clone())   [§4.3]
  │
  ▼
per compile_gpu_kernel_launch:
  constituents = fusion_constituents(origin_node)
  sum predictions over constituents
  reserve_id + record_kernel_at + emit begin/end hooks

apply_epilogue_fusion(plan: &mut)     [§4.1] → plan.fused_node_groups[matmul] = [matmul, ...elim]
apply_reduction_fusion(plan: &mut)    [§4.2] → plan.fused_node_groups[root] = all_matched
```

## 6. Error handling

- Source file missing on disk: empty `source_text`, line numbers degrade to 1 (same as pre-Phase-2.5 default). No error.
- `last_wrga_plan` is `None`: `fusion_plan_for_profile` stays `None`, `fusion_constituents` returns `vec![root]`. Epilogue/reduction fusion in pure-inference compiles under-predicts on fused kernels (Phase 3 follow-up).
- Call-site plan sharing: sites that previously built a fresh `FusionPlan` per-call now need to share one with the constituent-fusion passes. Isolated test harnesses use `FusionPlan::default()` and accept empty groups.

## 7. Testing

- **Unit (epilogue_fusion):** synthetic chain populates `fused_node_groups[matmul_node] == [matmul_node, ...epilogue_nodes]`.
- **Unit (reduction_fusion):** synthetic match populates `fused_node_groups[root_node] == all_matched_nodes`.
- **Unit (pre-pass wire-up):** mock `last_wrga_plan` with a non-empty `fusion.fused_node_groups`; `run_profile_pre_pass` copies it so `compiler.fusion_constituents(root)` returns the full Vec.
- **Unit (source fallback):** `run_profile_pre_pass` with `profile_source_file_name = Some(tempfile_path)` and `profile_source_text = None` populates `compiler.source_text` with the file contents.
- **Regression:** all existing `profiling_*`, `wrga_fusion_groups`, `epilogue_fusion`, `reduction_fusion`, and `monitor_e2e` tests still pass.

## 8. Non-goals

- WRGA adapter fusion site (`wrga_fusion.rs::build_fusion_plan`) — no NodeIds in scope. Phase 3.
- Unified `FusionPlanBuilder` threaded across all three fusion sites.
- Pure-inference fusion-plan carrier (a place for epilogue/reduction plans to live when `@train` isn't present). Phase 3.
- Real CUDA event verification on GPU hardware.
- Multi-stream profiling.

## 9. File inventory

**Modified:**

- `crates/nsl-codegen/src/epilogue_fusion.rs` — signature + populator (§4.1).
- `crates/nsl-codegen/src/reduction_fusion.rs` — signature + populator (§4.2).
- `crates/nsl-codegen/src/fusion_graph.rs` and any other call sites — pass `&mut FusionPlan`.
- `crates/nsl-codegen/src/compiler/entry_points.rs::run_profile_pre_pass` — §4.3 + §4.4.

Plus one or more test files in `crates/nsl-codegen/tests/` for the four unit tests in §7.

## 10. Follow-up phases

- **Phase 3:** WRGA adapter fusion NodeId threading (via `AdapterPlacement` enrichment).
- **Phase 3:** Pure-inference fusion-plan carrier.
- **Phase 3+:** Real CUDA-event timing verification on GPU.
- **Phase 4:** Multi-stream profiling.
- **Phase 5:** WGGO decision explainer.
- **Phase 6:** Training health monitor.
- **Phase 7:** `@inspect` tensor inspector.
