# NSL Dev Tools — Phase 2.5 Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close two Phase 2 TODOs: populate `FusionPlan.fused_node_groups` at the epilogue/reduction fusion decision sites, and have `run_profile_pre_pass` seed the Compiler-owned plan + fall back to reading source from disk when only the path is provided.

**Architecture:** `apply_epilogue_fusion` and `apply_reduction_fusion` each grow a `&mut FusionPlan` parameter and write constituent NodeIds inline. The Compiler owns a live `fusion_plan_for_profile: Option<FusionPlan>` that the pre-pass seeds with `Some(...)`. Call sites in profile builds pass `compiler.profile_fusion_plan_mut().unwrap()` so mutations land in the same instance `fusion_constituents` reads; non-profile callers pass a throwaway `&mut FusionPlan::default()`.

**Tech Stack:** Rust workspace (`cargo`), workspace tests via `cargo test`.

**Spec:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase2-5-design.md`

**Branch / worktree:** Continue on `feat/dev-tools-phase1` in `c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1`. No new worktree.

---

## Task 1: Epilogue fusion — populate `fused_node_groups`

**Files:**
- Modify: `crates/nsl-codegen/src/epilogue_fusion.rs::apply_epilogue_fusion` (signature + body).
- Modify: all call sites of `apply_epilogue_fusion` in the workspace.
- Test: `crates/nsl-codegen/tests/fusion_groups_populated.rs` (new).

- [ ] **Step 1: Locate signature + call sites**

```
grep -n "pub fn apply_epilogue_fusion\|apply_epilogue_fusion(" crates/nsl-codegen/src/ -r
```

Note the current signature (3 params: graph, chains, base_kernel_id) and every caller.

- [ ] **Step 2: Write failing test**

Create `crates/nsl-codegen/tests/fusion_groups_populated.rs`:

```rust
use std::collections::HashMap;
use nsl_codegen::epilogue_fusion::{apply_epilogue_fusion, EpilogueChain};
use nsl_codegen::fusion_graph::FusionGraph;
use nsl_codegen::wrga_fusion::FusionPlan;
use nsl_ast::NodeId;

#[test]
fn epilogue_fusion_populates_fused_node_groups() {
    // Build a minimal synthetic FusionGraph + EpilogueChain. If the actual
    // types require more setup (real graph nodes, valid kernel IDs), use
    // the existing test helpers in epilogue_fusion's unit tests as a model.
    // The key assertion: after apply_epilogue_fusion, plan.fused_node_groups
    // contains an entry for chain.matmul_node with value [matmul_node, ...eliminated_nodes].

    let mut graph = FusionGraph::default();
    let matmul = NodeId(100);
    let bias = NodeId(101);
    let relu = NodeId(102);
    let chain = EpilogueChain {
        matmul_node: matmul,
        eliminated_nodes: vec![bias, relu],
        // Any other fields: use Default::default() or whatever the existing
        // epilogue_fusion tests initialize them to.
        ..Default::default()
    };

    let mut plan = FusionPlan::default();
    let _kernels = apply_epilogue_fusion(&mut graph, &[chain], 0, &mut plan);

    let constituents = plan.fused_node_groups.get(&matmul)
        .expect("matmul root should be keyed into fused_node_groups");
    assert_eq!(constituents, &vec![matmul, bias, relu]);
}
```

If `EpilogueChain` doesn't impl `Default`, construct it explicitly — copy field initialization from an existing epilogue-fusion unit test. Don't invent shapes.

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test fusion_groups_populated 2>&1 | tail -15
```

Expected: compile error `function takes 3 arguments but 4 were supplied`.

- [ ] **Step 4: Extend the function signature**

In `crates/nsl-codegen/src/epilogue_fusion.rs`, change:

```rust
pub fn apply_epilogue_fusion(
    graph: &mut FusionGraph,
    chains: &[EpilogueChain],
    base_kernel_id: u32,
    plan: &mut crate::wrga_fusion::FusionPlan,   // NEW
) -> Vec<FusedKernel> {
    let mut out = Vec::new();
    for chain in chains {
        // ...existing body unchanged (marks eliminated_nodes, builds fused kernel)...

        // NEW: record the constituent NodeIds for the Phase 2 prediction sum.
        let mut constituents = Vec::with_capacity(1 + chain.eliminated_nodes.len());
        constituents.push(chain.matmul_node);
        constituents.extend(chain.eliminated_nodes.iter().copied());
        plan.fused_node_groups.insert(chain.matmul_node, constituents);
    }
    out
}
```

- [ ] **Step 5: Update every call site**

For each caller found in Step 1, pass a `&mut FusionPlan`:

- **Profile-enabled paths (Compiler-owned):** will be wired in Task 3. For now, if the call site is inside a Compiler method, pass `&mut FusionPlan::default()` and let Task 3 replace it.
- **Test-only / utility paths:** pass `&mut FusionPlan::default()`.

Typical call site patch:

```rust
let mut scratch = FusionPlan::default();
apply_epilogue_fusion(&mut graph, &chains, base_id, &mut scratch);
```

Resist the urge to plumb a real plan here — Task 3 does that. Minimize diff per task.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-codegen --test fusion_groups_populated 2>&1 | tail -10
cargo build -p nsl-codegen 2>&1 | tail -10
cargo test -p nsl-codegen --tests 2>&1 | tail -10
```

Expected: new test passes; build clean; no regressions.

- [ ] **Step 7: Commit**

```
git add crates/nsl-codegen/src/epilogue_fusion.rs \
        crates/nsl-codegen/tests/fusion_groups_populated.rs \
        $(grep -rln apply_epilogue_fusion crates/nsl-codegen/src/)
git commit -m "feat(fusion): apply_epilogue_fusion populates FusionPlan.fused_node_groups"
```

---

## Task 2: Reduction fusion — populate `fused_node_groups`

**Files:**
- Modify: `crates/nsl-codegen/src/reduction_fusion.rs::apply_reduction_fusion`.
- Modify: all call sites.
- Test: extend `crates/nsl-codegen/tests/fusion_groups_populated.rs`.

- [ ] **Step 1: Locate**

```
grep -n "pub fn apply_reduction_fusion\|apply_reduction_fusion(" crates/nsl-codegen/src/ -r
```

- [ ] **Step 2: Extend the test file**

Append to `crates/nsl-codegen/tests/fusion_groups_populated.rs`:

```rust
use nsl_codegen::reduction_fusion::{apply_reduction_fusion, ReductionMatch};

#[test]
fn reduction_fusion_populates_fused_node_groups() {
    let mut graph = FusionGraph::default();
    let root = NodeId(200);
    let exp = NodeId(201);
    let sum = NodeId(202);
    let div = NodeId(203);
    let m = ReductionMatch {
        root_node: root,
        all_matched_nodes: vec![exp, sum, div, root],
        ..Default::default()
    };

    let mut plan = FusionPlan::default();
    let _kernels = apply_reduction_fusion(&mut graph, &[m], 0, &mut plan);

    let constituents = plan.fused_node_groups.get(&root)
        .expect("root should be keyed into fused_node_groups");
    assert_eq!(constituents, &vec![exp, sum, div, root]);
}
```

Same caveat as Task 1: if `ReductionMatch` doesn't impl `Default`, copy field init from an existing reduction-fusion unit test.

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test fusion_groups_populated reduction_fusion_populates 2>&1 | tail -10
```

- [ ] **Step 4: Extend the function signature**

```rust
pub fn apply_reduction_fusion(
    graph: &mut FusionGraph,
    matches: &[ReductionMatch],
    base_kernel_id: u32,
    plan: &mut crate::wrga_fusion::FusionPlan,   // NEW
) -> Vec<FusedKernel> {
    let mut out = Vec::new();
    for m in matches {
        // ...existing body unchanged...
        plan.fused_node_groups.insert(m.root_node, m.all_matched_nodes.clone());
    }
    out
}
```

- [ ] **Step 5: Update every call site**

Same pattern as Task 1 — pass `&mut FusionPlan::default()` everywhere. Task 3 replaces for profile paths.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-codegen --test fusion_groups_populated 2>&1 | tail -10
cargo test -p nsl-codegen --tests 2>&1 | tail -10
```

Expected: 2 passed in the new test file; no regressions.

- [ ] **Step 7: Commit**

```
git add crates/nsl-codegen/src/reduction_fusion.rs \
        crates/nsl-codegen/tests/fusion_groups_populated.rs \
        $(grep -rln apply_reduction_fusion crates/nsl-codegen/src/)
git commit -m "feat(fusion): apply_reduction_fusion populates FusionPlan.fused_node_groups"
```

---

## Task 3: Pre-pass seeds `fusion_plan_for_profile` + profile-side call sites thread it

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` — add `profile_fusion_plan_mut` helper.
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs::run_profile_pre_pass` — seed `Some(...)`.
- Modify: profile-path call sites of `apply_epilogue_fusion` / `apply_reduction_fusion` — thread the Compiler-owned plan.
- Test: `crates/nsl-codegen/tests/fusion_plan_read_path.rs` (new; the regression test the spec §7 calls out).

- [ ] **Step 1: Add the helper**

In `crates/nsl-codegen/src/compiler/mod.rs`, in the `impl<'_> Compiler<'_>` block (near `fusion_constituents` at line 562):

```rust
/// Returns a mutable borrow of the Compiler-owned FusionPlan when profiling
/// is active. Fusion passes should call this and thread the borrow into
/// apply_epilogue_fusion / apply_reduction_fusion so that mutations land
/// in the same instance that fusion_constituents reads.
pub fn profile_fusion_plan_mut(&mut self) -> Option<&mut crate::wrga_fusion::FusionPlan> {
    self.fusion_plan_for_profile.as_mut()
}
```

- [ ] **Step 2: Update pre-pass seeding**

In `crates/nsl-codegen/src/compiler/entry_points.rs::run_profile_pre_pass`, locate the line `compiler.fusion_plan_for_profile = None;` (around line 82) and replace with:

```rust
// Seed the Compiler-owned plan so later fusion passes can write into the
// same instance fusion_constituents reads. Copy WRGA-level adapter-fusion
// groups if a recent @train compile produced them; otherwise start empty.
let seeded = compiler.last_wrga_plan
    .as_ref()
    .map(|p| p.fusion.clone())
    .unwrap_or_default();
compiler.fusion_plan_for_profile = Some(seeded);
```

- [ ] **Step 3: Write the read-path regression test**

Create `crates/nsl-codegen/tests/fusion_plan_read_path.rs`:

```rust
//! Regression test for Phase 2.5 spec §7: epilogue fusion running against
//! the Compiler-owned plan must make the groups visible to
//! Compiler::fusion_constituents. If this fails, profile_fusion_plan_mut()
//! is returning a wrong reference or a call site is still using a throwaway.
#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use nsl_codegen::wrga_fusion::FusionPlan;
use nsl_codegen::epilogue_fusion::{apply_epilogue_fusion, EpilogueChain};
use nsl_codegen::fusion_graph::FusionGraph;
use nsl_ast::NodeId;

#[test]
fn epilogue_fusion_against_compiler_plan_is_visible_to_fusion_constituents() {
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    return matmul(x, W)
"#;
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    opts.target_gpu = "h100".to_string();
    opts.dtype = "bf16".to_string();

    // run_pre_pass_only (from Phase 2 test_helpers) returns a summary or the
    // Compiler itself — see crates/nsl-codegen/src/test_helpers.rs. If it
    // returns a summary, extend the helper to also return a reference-owning
    // Compiler or restructure the test. Goal: get a live `&mut Compiler`
    // whose pre-pass has run.
    let result = nsl_codegen::test_helpers::run_pre_pass_only(src, &opts).unwrap();
    assert!(result.manifest_builder_set);
    assert!(result.fusion_plan_for_profile_set);
    // `result.fused_groups_after_epi` is a new field we add below.
    // Alternate structure: expose a closure-based helper that lets the test
    // mutate the Compiler during pre-pass.
}
```

**Caveat:** the current `run_pre_pass_only` returns a `PrePassResult` summary (per Phase 2 Task 4 report), not the `Compiler` itself, because of lifetime entanglement with a transient `Interner`/`TypeMap`. To test the read-path, add a second helper `run_pre_pass_and_fuse` in `crates/nsl-codegen/src/test_helpers.rs`:

```rust
#[cfg(any(test, feature = "test-helpers"))]
pub fn run_pre_pass_and_fuse(
    src: &str,
    opts: &crate::CompileOptions,
    matmul_node: nsl_ast::NodeId,
    eliminated: Vec<nsl_ast::NodeId>,
) -> Result<ReadPathProbe, String> {
    // 1. Run lex/parse/analyze + pre-pass (same as run_pre_pass_only).
    // 2. Get a &mut borrow of the Compiler.
    // 3. Call apply_epilogue_fusion with compiler.profile_fusion_plan_mut().unwrap()
    //    against a synthetic chain using the passed NodeIds.
    // 4. Return ReadPathProbe { constituents_from_compiler: compiler.fusion_constituents(matmul_node) }.
    todo!("helper mirrors run_pre_pass_only but runs apply_epilogue_fusion mid-flight")
}

#[cfg(any(test, feature = "test-helpers"))]
pub struct ReadPathProbe {
    pub constituents_from_compiler: Vec<nsl_ast::NodeId>,
}
```

Replace the test body with:

```rust
#[test]
fn epilogue_fusion_against_compiler_plan_is_visible_to_fusion_constituents() {
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    return matmul(x, W)
"#;
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    let matmul = NodeId(9000);
    let bias = NodeId(9001);
    let probe = nsl_codegen::test_helpers::run_pre_pass_and_fuse(
        src, &opts, matmul, vec![bias],
    ).unwrap();
    assert_eq!(probe.constituents_from_compiler, vec![matmul, bias],
        "fusion_constituents must read the Compiler-owned plan, not a throwaway");
}
```

If implementing `run_pre_pass_and_fuse` is genuinely intractable (construction issues surfaced in Phase 2 Task 4 persist), fall back to a simpler direct-API test that doesn't go through the codegen pipeline:

```rust
#[test]
fn seeded_fusion_plan_round_trips_through_fusion_constituents() {
    // Phase 2.5 minimal regression: verify that populating the Compiler-owned
    // plan via profile_fusion_plan_mut makes groups visible to fusion_constituents.
    // This doesn't exercise the full compile pipeline but catches the
    // throwaway-plan regression at the Compiler-API level.
    use nsl_codegen::compiler::Compiler;
    // Construct a minimally-initialized Compiler (same path run_pre_pass_only uses).
    let mut compiler = nsl_codegen::test_helpers::fresh_profile_compiler();
    compiler.fusion_plan_for_profile = Some(FusionPlan::default());

    let mut graph = FusionGraph::default();
    let matmul = NodeId(500);
    let bias = NodeId(501);
    let chain = EpilogueChain {
        matmul_node: matmul,
        eliminated_nodes: vec![bias],
        ..Default::default()
    };
    apply_epilogue_fusion(
        &mut graph, &[chain], 0,
        compiler.profile_fusion_plan_mut().expect("seeded"),
    );

    assert_eq!(compiler.fusion_constituents(matmul), vec![matmul, bias]);
}
```

`fresh_profile_compiler` is another test helper — returns a new `Compiler` with minimal valid state. If `Compiler::new` is hard to call, exposing an `#[cfg(any(test, feature = "test-helpers"))] fn new_for_profile_tests() -> Self` on `Compiler` that initializes only the profile-relevant fields is acceptable.

**Procedure guideline:** try the full `run_pre_pass_and_fuse` test first. If it takes more than 30 minutes to get the Compiler construction right, fall back to the simpler direct-API test. Report which you chose.

- [ ] **Step 4: Run — expect fail on first pass, pass on second**

```
cargo test -p nsl-codegen --features test-helpers --test fusion_plan_read_path 2>&1 | tail -10
```

If it already passes (rare — depends on whether Phase 2's `fusion_plan_for_profile = None` short-circuits `fusion_constituents` early), your test isn't catching the read-path bug. Strengthen it.

- [ ] **Step 5: Update profile-path call sites to thread the Compiler-owned plan**

Grep for every call of `apply_epilogue_fusion` and `apply_reduction_fusion` inside a Compiler method, and replace the `&mut FusionPlan::default()` scratch from Tasks 1–2 with a conditional that uses the Compiler-owned plan when profiling is on:

```rust
let mut scratch = FusionPlan::default();
let plan = self.profile_fusion_plan_mut().unwrap_or(&mut scratch);
apply_epilogue_fusion(&mut graph, &chains, base_id, plan);
```

Do the same for `apply_reduction_fusion`. If the call site is outside `Compiler` (e.g., in a free function), leave the scratch as-is.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-codegen --features test-helpers --test fusion_plan_read_path 2>&1 | tail -10
cargo test -p nsl-codegen --tests 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 7: Commit**

```
git add crates/nsl-codegen/src/compiler/mod.rs \
        crates/nsl-codegen/src/compiler/entry_points.rs \
        crates/nsl-codegen/src/test_helpers.rs \
        crates/nsl-codegen/tests/fusion_plan_read_path.rs \
        $(grep -rln "apply_epilogue_fusion\|apply_reduction_fusion" crates/nsl-codegen/src/ | grep -v tests/)
git commit -m "feat(codegen): pre-pass seeds Compiler-owned FusionPlan; fusion_constituents reads live mutations"
```

---

## Task 4: Source-text disk fallback in pre-pass

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs::run_profile_pre_pass`.
- Test: `crates/nsl-codegen/tests/profile_source_fallback.rs` (new).

- [ ] **Step 1: Write failing test**

Create `crates/nsl-codegen/tests/profile_source_fallback.rs`:

```rust
#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use std::io::Write;

#[test]
fn pre_pass_reads_source_from_disk_when_text_is_none() {
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    writeln!(tmp, "fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:").unwrap();
    writeln!(tmp, "    return x").unwrap();
    let path = tmp.path().to_path_buf();

    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    opts.profile_source_text = None;                         // <-- no text
    opts.profile_source_file_name = Some(path.display().to_string());

    // run_pre_pass_only should return a probe that lets us observe source_text.
    // If PrePassResult doesn't already expose source_text, extend it to.
    let result = nsl_codegen::test_helpers::run_pre_pass_only(
        // The helper reads opts.profile_source_text || fs::read(opts.profile_source_file_name).
        // For this test, pass an empty src argument and rely on the disk fallback.
        "", &opts,
    ).unwrap();

    assert!(result.source_text.contains("fn forward"),
        "expected disk-read source in compiler.source_text, got {:?}", result.source_text);
}

#[test]
fn pre_pass_prefers_explicit_text_over_disk() {
    let mut opts = CompileOptions::default();
    opts.profile_kernels = true;
    opts.profile_source_text = Some("# explicit override".to_string());
    opts.profile_source_file_name = Some("/nonexistent/path.nsl".to_string());

    let result = nsl_codegen::test_helpers::run_pre_pass_only("", &opts).unwrap();
    assert_eq!(result.source_text, "# explicit override");
}
```

If `PrePassResult` from Phase 2 Task 4 doesn't expose `source_text`, extend it:

```rust
// crates/nsl-codegen/src/test_helpers.rs
pub struct PrePassResult {
    pub prediction_map_len: usize,
    pub manifest_builder_set: bool,
    pub fusion_plan_for_profile_set: bool,
    pub source_text: String,          // NEW
}
```

Populate in `run_pre_pass_only` from `compiler.source_text`.

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-codegen --features test-helpers --test profile_source_fallback 2>&1 | tail -15
```

- [ ] **Step 3: Implement the fallback in pre-pass**

In `crates/nsl-codegen/src/compiler/entry_points.rs::run_profile_pre_pass`, add after `compiler.manifest_builder = Some(...)`:

```rust
// Source-text priority: explicit options → disk fallback → empty.
compiler.source_text = match &opts.profile_source_text {
    Some(s) => s.clone(),
    None => opts.profile_source_file_name
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_default(),
};
compiler.source_file_name = opts.profile_source_file_name
    .clone()
    .unwrap_or_default();
```

Place the snippet BEFORE Task 3's fusion-plan seeding so source context is ready when fusion passes emit.

- [ ] **Step 4: Run tests**

```
cargo test -p nsl-codegen --features test-helpers --test profile_source_fallback 2>&1 | tail -10
cargo test -p nsl-codegen --tests 2>&1 | tail -10
```

Expected: 2 passed (disk fallback + explicit override); no regressions.

- [ ] **Step 5: Commit**

```
git add crates/nsl-codegen/src/compiler/entry_points.rs \
        crates/nsl-codegen/src/test_helpers.rs \
        crates/nsl-codegen/tests/profile_source_fallback.rs
git commit -m "feat(codegen): pre-pass reads source from disk when profile_source_text unset"
```

---

## Task 5: Final verification

- [ ] **Step 1: Full workspace test**

```
cargo test --workspace --tests -- --test-threads=1 2>&1 | grep -E "^test result:|FAILED"
```

Expected: all green. The `--test-threads=1` avoids the known Windows parallel-file-lock issue on `e2e_m12_grad_basic_source_ad`.

- [ ] **Step 2: Smoke test real-world profile**

```
cargo run -p nsl-cli -- run --monitor crates/nsl-cli/tests/fixtures/tiny_transformer.nsl 2>&1 | head -30
```

Expected: the monitor output still works end-to-end (Phase 2 behavior preserved). Inspect `tiny_transformer.nsl-profile.json` — kernels should now have `source_span.start_line > 1` (real line numbers), and if the fixture exercises epilogue or reduction fusion, the predicted_us for fused kernels should reflect the sum of constituents.

- [ ] **Step 3: Confirm branch status**

```
cd c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1
git log --oneline main..HEAD | wc -l
```

Expected: 19 (Phase 1) + 9 (Phase 2) + 4 (Phase 2.5) = 32 commits on `feat/dev-tools-phase1`. Adjust expectation if earlier tasks merged multiple commits.

- [ ] **Step 4: Do not push or open PR**

Per user instruction (2026-04-12), held local until all milestones ship. No action here.

---

## Self-Review

**Spec coverage:**

- §4.1 epilogue populator → Task 1. ✅
- §4.2 reduction populator → Task 2. ✅
- §4.3 pre-pass seeding + `profile_fusion_plan_mut` + call-site threading → Task 3. ✅
- §4.4 source-text disk fallback → Task 4. ✅
- §6 error handling — throwaway-plan regression covered by Task 3 Step 3 read-path test. ✅
- §7 testing — each of the 4 required unit tests has a task. ✅
- §8 non-goals (WRGA adapter, FusionPlanBuilder, pure-inference carrier, multi-stream, GPU) — respected, no task attempts them. ✅

**Placeholder scan:**

One `todo!()` stub in Task 3 Step 3's `run_pre_pass_and_fuse` helper, explicitly flagged as "try the full test first; if >30 min, fall back to the direct-API test". The direct-API fallback has complete code. No other `TBD`/"later"/vague patterns.

**Type consistency:**

- `FusionPlan`, `fused_node_groups`, `constituents_of` consistent with Phase 2 Task 3's types.
- `profile_fusion_plan_mut` introduced in Task 3 Step 1, used in Task 3 Step 5 and the regression test.
- `PrePassResult.source_text` field added in Task 4 test; Task 3 test references `result.manifest_builder_set` / `fusion_plan_for_profile_set` (both from Phase 2 Task 4's existing shape).

**Scope:**

5 tasks (4 impl + 1 verification). Each produces testable working software. Final verification is workspace test + smoke.
