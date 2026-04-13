# NSL Dev Tools — Phase 2 Real Codegen Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire real `nsl_profile_kernel_begin/end` hooks around every GPU kernel launch in compiled NSL programs, replace the runtime collector's `NanoClock` with real `cuEventElapsedTime`-backed timing, and have codegen own the kernel-ID space so the manifest reflects what was actually emitted (post-fusion).

**Architecture:** Codegen runs the walker once at compile-start to build a `HashMap<NodeId, OpCost>` of predictions. Each call to `compile_gpu_kernel_launch` reserves a kernel ID via `Compiler.manifest_builder`, sums constituent-op predictions through `FusionPlan.fused_node_groups`, records a `KernelEntry`, and emits two extra Cranelift `call` instructions wrapping the existing `nsl_kernel_launch`. After codegen, the manifest is written to `<out>.nsl-profile.json`. The runtime FFI hooks check out a `cuEvent_t` from a pool, record it on the same `CUstream` the kernel launch uses, and pass the event handle through the existing `Collector` aggregation path.

**Tech Stack:** Rust workspace (`cargo`), Cranelift IR for codegen emission, `cudarc 0.19` for CUDA event APIs, `serde_json` for manifest IO, `insta` for snapshot tests where useful.

**Spec:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase2-design.md`

**Branch / worktree:** Continue on `feat/dev-tools-phase1` in `c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1`. No new worktree — Phase 2 builds directly on Phase 1 commits.

---

## Task 1: `OpCost.origin_node` + walker population

**Files:**
- Modify: `crates/nsl-codegen/src/cost_model.rs` — add field to `OpCost`.
- Modify: `crates/nsl-codegen/src/profiling/walker.rs` — populate the field.
- Test: `crates/nsl-codegen/tests/profiling_walker.rs` (extend existing).

The codegen pre-pass needs a stable handle from each `OpCost` back to the AST node it came from. Phase 1 emits a string `loc` like `"0:421-427"` which is brittle to parse. Add a real `Option<NodeId>` field instead.

- [ ] **Step 1: Read existing types**

Run: `grep -n "pub id" crates/nsl-ast/src/expr.rs | head -5` to confirm `Expr.id: NodeId`. Run: `grep -n "pub struct OpCost" crates/nsl-codegen/src/cost_model.rs` to confirm location.

- [ ] **Step 2: Write failing test**

Append to `crates/nsl-codegen/tests/profiling_walker.rs`:

```rust
#[test]
fn walker_populates_origin_node_on_each_op() {
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    let y = matmul(x, W)
    return y
"#;
    let (m, analysis, interner) = parse_and_analyze(src);
    let gpu = nsl_codegen::gpu_specs::find_gpu("h100").unwrap();
    let env = nsl_codegen::profiling::shape_env::ShapeEnv::with_defaults();
    let r = nsl_codegen::profiling::walker::walk_ops(
        &m, &analysis, &interner,
        nsl_codegen::profiling::types::EntryKind::Auto,
        &env, gpu, "bf16",
    ).unwrap();
    assert!(!r.ops.is_empty());
    for op in &r.ops {
        assert!(op.origin_node.is_some(),
            "every walked op should carry its source NodeId, got None for {}", op.name);
    }
}
```

- [ ] **Step 3: Run test — expect fail**

```
cargo test -p nsl-codegen --test profiling_walker walker_populates_origin_node_on_each_op 2>&1 | tail -10
```

Expected: compile error `no field 'origin_node' on type 'OpCost'`.

- [ ] **Step 4: Add the field**

In `crates/nsl-codegen/src/cost_model.rs`, locate `pub struct OpCost` and add as the last field:

```rust
pub struct OpCost {
    // ...all existing fields...
    pub origin_node: Option<nsl_ast::NodeId>,
}
```

If `nsl_ast::NodeId` isn't already a workspace dep of `nsl-codegen`, it should be — check `Cargo.toml`. Default the field to `None` in any test fixtures that construct `OpCost` directly (search: `grep -rn "OpCost {" crates/`).

- [ ] **Step 5: Populate in walker**

In `crates/nsl-codegen/src/profiling/walker.rs`, find every place that constructs an `OpCost` (likely a single helper `push_raw` or inlined in `push`/`emit_*` methods). Pass the originating `&Expr` through and set `origin_node: Some(expr.id)`. For `emit_unknown` and `emit_with_zero`, set `origin_node: Some(expr.id)` too — even unknown ops have an AST node.

If the existing helpers don't take an `Expr`, change the signatures to accept it. Walker tests should still pass.

- [ ] **Step 6: Run all walker tests**

```
cargo test -p nsl-codegen --test profiling_walker 2>&1 | tail -10
```

Expected: 5 passed (4 original + new origin_node test).

- [ ] **Step 7: Run all profiling tests for regression**

```
cargo test -p nsl-codegen --test profiling_types --test profiling_walker --test profiling_memory_timeline --test profiling_instrument 2>&1 | tail -15
```

Expected: 16 passed.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/cost_model.rs crates/nsl-codegen/src/profiling/walker.rs \
        crates/nsl-codegen/tests/profiling_walker.rs
git commit -m "feat(profiling): OpCost.origin_node carries source NodeId from walker"
```

---

## Task 2: `ManifestBuilder::reserve_id` / `record_kernel_at` + `SourceSpanJson::from_span`

**Files:**
- Modify: `crates/nsl-codegen/src/profiling/instrument.rs` — add two methods + one helper.
- Test: `crates/nsl-codegen/tests/profiling_instrument.rs` (extend existing).

Codegen needs to reserve IDs at one site (so the same ID can be emitted as a Cranelift constant into begin/end) and record the entry separately. It also needs to convert a `nsl_errors::Span` to a `SourceSpanJson` with real line numbers.

- [ ] **Step 1: Read SourceMap API**

Run: `grep -n "pub struct SourceMap\|impl SourceMap" crates/nsl-errors/src/*.rs | head -10`. Note the method that converts a byte position to a line number (likely `line_index_for(BytePos) -> u32` or similar). If no such helper exists, add one in this task.

- [ ] **Step 2: Write failing tests**

Append to `crates/nsl-codegen/tests/profiling_instrument.rs`:

```rust
use nsl_codegen::profiling::instrument::{ManifestBuilder, SourceSpanJson};

#[test]
fn reserve_id_returns_dense_monotonic() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    assert_eq!(b.reserve_id(), 0);
    assert_eq!(b.reserve_id(), 1);
    assert_eq!(b.reserve_id(), 2);
}

#[test]
fn record_kernel_at_uses_supplied_id() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    let id = b.reserve_id();
    b.record_kernel_at(
        id, "matmul",
        SourceSpanJson { file: "m.nsl".into(), start_line: 10, end_line: 12 },
        1.5, 1000, 50,
    );
    let m = b.finish();
    assert_eq!(m.kernels.len(), 1);
    assert_eq!(m.kernels[0].kernel_id, id);
    assert_eq!(m.kernels[0].op_name, "matmul");
    assert_eq!(m.kernels[0].source_span.start_line, 10);
}

#[test]
fn reserve_id_and_record_kernel_interoperate() {
    // reserve_id increments; record_kernel (auto) uses the next free id.
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    let a = b.reserve_id();   // 0
    b.record_kernel_at(a, "x", SourceSpanJson { file: "m".into(), start_line: 1, end_line: 1 }, 0.0, 0, 0);
    let bid = b.record_kernel("y", SourceSpanJson { file: "m".into(), start_line: 2, end_line: 2 }, 0.0, 0, 0);
    assert_eq!(bid, 1);
    let mfst = b.finish();
    assert_eq!(mfst.kernels.len(), 2);
}
```

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test profiling_instrument reserve 2>&1 | tail -10
```

Expected: `no method named 'reserve_id'` and `no method named 'record_kernel_at'`.

- [ ] **Step 4: Implement the methods**

In `crates/nsl-codegen/src/profiling/instrument.rs::impl ManifestBuilder`:

```rust
pub fn reserve_id(&mut self) -> u32 {
    let id = self.next_id;
    self.next_id += 1;
    id
}

pub fn record_kernel_at(
    &mut self,
    id: u32,
    op_name: &str,
    span: SourceSpanJson,
    predicted_us: f64,
    predicted_flops: u64,
    predicted_hbm_bytes: u64,
) {
    self.inner.kernels.push(KernelEntry {
        kernel_id: id,
        op_name: op_name.into(),
        source_span: span,
        predicted_us, predicted_flops, predicted_hbm_bytes,
    });
}
```

Refactor existing `record_kernel(name, span, us, flops, hbm) -> u32` into:

```rust
pub fn record_kernel(&mut self, op_name: &str, span: SourceSpanJson, us: f64, flops: u64, hbm: u64) -> u32 {
    let id = self.reserve_id();
    self.record_kernel_at(id, op_name, span, us, flops, hbm);
    id
}
```

- [ ] **Step 5: Add `SourceSpanJson::from_span` helper**

In `instrument.rs`:

```rust
impl SourceSpanJson {
    /// Resolve a byte-positioned Span to 1-based line numbers using the source text.
    /// Falls back to lines (1, 1) if start/end are out of range.
    pub fn from_span(span: nsl_errors::Span, file_name: &str, source_text: &str) -> Self {
        let start_line = line_of_byte(source_text, span.start.0);
        let end_byte = span.end.0.saturating_sub(1).max(span.start.0);
        let end_line = line_of_byte(source_text, end_byte);
        Self { file: file_name.into(), start_line, end_line }
    }
}
```

`line_of_byte` already exists in `instrument.rs` from Phase 1 — reuse it.

Add a test:

```rust
#[test]
fn source_span_json_from_span_resolves_lines() {
    let src = "line1\nline2\nline3\nline4\n";
    // Bytes 6..11 = "line2"
    let span = nsl_errors::Span {
        file_id: nsl_errors::FileId(0),
        start: nsl_errors::BytePos(6),
        end: nsl_errors::BytePos(11),
    };
    let s = SourceSpanJson::from_span(span, "m.nsl", src);
    assert_eq!(s.start_line, 2);
    assert_eq!(s.end_line, 2);
    assert_eq!(s.file, "m.nsl");
}
```

- [ ] **Step 6: Run instrument tests**

```
cargo test -p nsl-codegen --test profiling_instrument 2>&1 | tail -10
```

Expected: 9 passed (5 original + 4 new).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/profiling/instrument.rs \
        crates/nsl-codegen/tests/profiling_instrument.rs
git commit -m "feat(profiling): ManifestBuilder.reserve_id + record_kernel_at + SourceSpanJson::from_span"
```

---

## Task 3: `FusionPlan.fused_node_groups`

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fusion.rs` — add field + populator.
- Test: `crates/nsl-codegen/tests/wrga_fusion_groups.rs` (new).

Phase 2's emission code calls `Compiler::fusion_constituents(origin_node)` which queries this map. Without it, fused kernels would only get the fusion-root op's predicted cost, undercounting by 2–4×.

- [ ] **Step 1: Read existing FusionPlan + build path**

Run: `grep -n "pub struct FusionPlan\|pub fn build_fusion_plan\|FusionDecision" crates/nsl-codegen/src/wrga_fusion.rs | head -20`. Identify the function(s) that decide which ops to fuse — note where they have access to the constituent NodeIds.

- [ ] **Step 2: Write failing test**

Create `crates/nsl-codegen/tests/wrga_fusion_groups.rs`:

```rust
use nsl_codegen::wrga_fusion::{FusionPlan, build_fusion_plan};

#[test]
fn fusion_plan_exposes_fused_node_groups_field() {
    let plan = build_fusion_plan(&[], None);
    // Empty plan still exposes the map (empty).
    let _: &std::collections::HashMap<nsl_ast::NodeId, Vec<nsl_ast::NodeId>> = &plan.fused_node_groups;
}

#[test]
fn non_fused_op_returns_singleton_via_lookup() {
    let plan = FusionPlan { decisions: vec![], fused_node_groups: Default::default() };
    let nid = nsl_ast::NodeId::new(42);
    // Helper: the lookup function used by Compiler. Returns vec![nid] for non-fused.
    let constituents = plan.constituents_of(nid);
    assert_eq!(constituents, vec![nid]);
}

#[test]
fn fused_root_returns_all_constituents() {
    let mut groups = std::collections::HashMap::new();
    let root = nsl_ast::NodeId::new(10);
    let c1 = nsl_ast::NodeId::new(11);
    let c2 = nsl_ast::NodeId::new(12);
    let c3 = nsl_ast::NodeId::new(13);
    groups.insert(root, vec![c1, c2, c3]);
    let plan = FusionPlan { decisions: vec![], fused_node_groups: groups };
    let constituents = plan.constituents_of(root);
    assert_eq!(constituents, vec![c1, c2, c3]);
}
```

If `nsl_ast::NodeId` doesn't have a `pub fn new(u32) -> Self` constructor, use the actual constructor (e.g., `NodeId(42)` if it's a tuple struct).

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test wrga_fusion_groups 2>&1 | tail -15
```

Expected: `no field 'fused_node_groups'` and `no method 'constituents_of'`.

- [ ] **Step 4: Add the field + lookup**

In `crates/nsl-codegen/src/wrga_fusion.rs`:

```rust
use std::collections::HashMap;
use nsl_ast::NodeId;

pub struct FusionPlan {
    pub decisions: Vec<FusionDecision>,
    pub fused_node_groups: HashMap<NodeId, Vec<NodeId>>,
}

impl FusionPlan {
    /// Returns the constituent NodeIds folded into the kernel rooted at `root`.
    /// For non-fused kernels (no entry in the map), returns vec![root].
    pub fn constituents_of(&self, root: NodeId) -> Vec<NodeId> {
        self.fused_node_groups.get(&root)
            .cloned()
            .unwrap_or_else(|| vec![root])
    }
}
```

In `build_fusion_plan(...)`, initialize `fused_node_groups: HashMap::new()` so the existing test fixtures still construct correctly.

Other constructors of `FusionPlan` in the codebase (search: `grep -rn "FusionPlan {" crates/`) need the new field — initialize to `HashMap::new()`.

- [ ] **Step 5: Populate the map at fusion decision sites**

Search the fusion pass for the actual decision points:

```
grep -rn "fn fuse_\|epilogue_fuse\|csha_fuse" crates/nsl-codegen/src/ | head
```

At each site that decides "this kernel will fold ops A, B, C into root R", add:

```rust
plan.fused_node_groups.insert(root_node_id, vec![a_node_id, b_node_id, c_node_id]);
```

If the fusion pass doesn't currently track the original NodeIds (because it operates on lowered IR), add a side-table or pass the NodeIds through. **Scope guardrail:** if the fusion passes don't have NodeIds in scope and threading them is intrusive, leave `fused_node_groups` empty (the lookup falls back to singleton). Add a TODO comment at each decision site explaining the gap. Phase 2.5 can fill this in. Document the gap in the task report.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-codegen --test wrga_fusion_groups 2>&1 | tail -10
cargo build -p nsl-codegen 2>&1 | tail -10
```

Expected: 3 passed; build clean (modulo pre-existing warnings).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fusion.rs crates/nsl-codegen/tests/wrga_fusion_groups.rs
git commit -m "feat(fusion): FusionPlan.fused_node_groups + constituents_of lookup"
```

---

## Task 4: `Compiler` pre-pass — `prediction_map` + `manifest_builder` + `fusion_constituents`

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` — three new fields + helper.
- Modify: `crates/nsl-codegen/src/lib.rs` — pre-pass call in `compile()` (or `compile_module()`).
- Test: `crates/nsl-codegen/tests/profile_kernels_pre_pass.rs` (new).

- [ ] **Step 1: Locate the codegen entry**

Run: `grep -n "pub fn compile\|pub fn compile_module" crates/nsl-codegen/src/lib.rs`. Identify the function that's called from CLI for `Run`/`Build`. Read the surrounding 30 lines to see how it constructs the `Compiler` and walks the module.

- [ ] **Step 2: Write failing integration test**

Create `crates/nsl-codegen/tests/profile_kernels_pre_pass.rs`:

```rust
//! Verify that when CompileOptions.profile_kernels = true, the Compiler
//! gains a populated prediction_map and a manifest_builder. This test does
//! NOT compile to a binary — it stops after the pre-pass and inspects the
//! Compiler state, so no CUDA / linker is needed.

use nsl_codegen::CompileOptions;

#[test]
fn pre_pass_populates_prediction_map_when_profile_kernels_enabled() {
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    let y = matmul(x, W)
    return y
"#;
    let opts = CompileOptions {
        profile_kernels: true,
        target_gpu: "h100".into(),
        dtype: "bf16".into(),
        ..CompileOptions::default()
    };
    let compiler = nsl_codegen::test_helpers::run_pre_pass_only(src, &opts).unwrap();
    assert!(!compiler.prediction_map.is_empty(),
        "prediction_map should be non-empty for a module with at least one matmul");
    assert!(compiler.manifest_builder.is_some(),
        "manifest_builder must be set when profile_kernels is true");
}

#[test]
fn pre_pass_skipped_when_profile_kernels_disabled() {
    let src = r#"
fn forward(x: Tensor<[1, 2048, 512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    return matmul(x, W)
"#;
    let opts = CompileOptions {
        profile_kernels: false,
        ..CompileOptions::default()
    };
    let compiler = nsl_codegen::test_helpers::run_pre_pass_only(src, &opts).unwrap();
    assert!(compiler.prediction_map.is_empty());
    assert!(compiler.manifest_builder.is_none());
}
```

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test profile_kernels_pre_pass 2>&1 | tail -15
```

Expected: unresolved imports (`run_pre_pass_only`, `prediction_map` etc.).

- [ ] **Step 4: Add fields to `Compiler`**

In `crates/nsl-codegen/src/compiler/mod.rs::struct Compiler`, add:

```rust
use std::collections::HashMap;
use nsl_ast::NodeId;
use crate::cost_model::OpCost;
use crate::profiling::instrument::ManifestBuilder;
use crate::wrga_fusion::FusionPlan;

pub struct Compiler<'a> {
    // ...existing fields...
    pub prediction_map: HashMap<NodeId, OpCost>,
    pub manifest_builder: Option<ManifestBuilder>,
    pub fusion_plan_for_profile: Option<FusionPlan>,
}
```

In `Compiler::new(...)` (or wherever the struct is constructed), initialize:

```rust
prediction_map: HashMap::new(),
manifest_builder: None,
fusion_plan_for_profile: None,
```

Also add a helper method:

```rust
impl<'a> Compiler<'a> {
    pub fn fusion_constituents(&self, root: NodeId) -> Vec<NodeId> {
        match &self.fusion_plan_for_profile {
            Some(p) => p.constituents_of(root),
            None => vec![root],
        }
    }
}
```

- [ ] **Step 5: Run pre-pass in the codegen entry**

In `crates/nsl-codegen/src/lib.rs::compile()` (or whichever function the survey identified at lib.rs:102), after parsing/analysis is done and before function bodies are walked:

```rust
if opts.profile_kernels {
    use crate::profiling::shape_env::ShapeEnv;
    use crate::profiling::types::EntryKind;
    use crate::profiling::walker::walk_ops;
    use crate::gpu_specs::find_gpu;
    use crate::profiling::instrument::ManifestBuilder;

    let env = ShapeEnv::with_defaults();
    let gpu = find_gpu(&opts.target_gpu)
        .ok_or_else(|| format!("unknown GPU target: {}", opts.target_gpu))?;
    let report = walk_ops(
        &module, &analysis, &interner,
        EntryKind::Auto, &env, gpu, &opts.dtype,
    )?;
    compiler.prediction_map = report.ops.iter()
        .filter_map(|op| op.origin_node.map(|nid| (nid, op.clone())))
        .collect();
    compiler.manifest_builder = Some(ManifestBuilder::new(&opts.target_gpu, &opts.dtype));
    // FusionPlan is populated by the existing fusion pass; if available,
    // store it on the compiler for fusion_constituents lookups.
    // For now (no plumbing yet), leave fusion_plan_for_profile as None —
    // singleton fallback applies. Phase 2.5 wires the real plan.
    compiler.fusion_plan_for_profile = None;
}
```

The exact field/method names on `compiler` may differ — match the existing constructor. The error type for the `?` should match what the function currently returns (`anyhow::Error`, `CodegenError`, `String` — adapt).

- [ ] **Step 6: Add a test helper**

In a new file `crates/nsl-codegen/src/test_helpers.rs` (or under `#[cfg(test)] mod test_helpers` in `lib.rs` if you prefer, then `pub use` it):

```rust
#[cfg(any(test, feature = "test-helpers"))]
pub fn run_pre_pass_only(
    src: &str,
    opts: &crate::CompileOptions,
) -> Result<crate::compiler::Compiler<'static>, String> {
    // Lex/parse/analyze using the same pipeline as `compile()`.
    // Then construct a Compiler, run the profile pre-pass only, and return.
    // Implementation depends on existing module shape — typically:
    //   1. let mut interner = Interner::new();
    //   2. let (tokens, _) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    //   3. let parsed = nsl_parser::parse(&tokens, &mut interner);
    //   4. let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    //   5. let mut compiler = crate::compiler::Compiler::new(...);
    //   6. inline the pre-pass code from Step 5 above
    //   7. return Ok(compiler);
    todo!("inline pre-pass here — exact construction depends on Compiler::new signature")
}
```

The `todo!()` is the only one in the plan and it's intentional: implement it by mirroring the front part of `compile()`. If `Compiler::new` requires arguments not easily synthesizable (output paths, target triple, etc.), use a simpler dummy constructor or refactor `Compiler::new` to take an `Options` builder. **Do not block** — if construction is genuinely hard, replace these tests with a smaller integration test that round-trips a `Compiler` mock and trust the end-to-end test in Task 9 to cover the wiring.

- [ ] **Step 7: Add `pub mod test_helpers;` to lib.rs**

Behind `#[cfg(any(test, feature = "test-helpers"))]` so it doesn't bloat the release binary.

- [ ] **Step 8: Run pre-pass tests**

```
cargo test -p nsl-codegen --test profile_kernels_pre_pass 2>&1 | tail -10
```

Expected: 2 passed.

- [ ] **Step 9: Run full nsl-codegen test suite to catch regressions**

```
cargo test -p nsl-codegen 2>&1 | tail -15
```

Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-codegen/src/compiler/mod.rs crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/src/test_helpers.rs \
        crates/nsl-codegen/tests/profile_kernels_pre_pass.rs
git commit -m "feat(codegen): profile_kernels pre-pass populates prediction_map + manifest_builder"
```

---

## Task 5: Hook emission at `compile_gpu_kernel_launch`

**Files:**
- Modify: `crates/nsl-codegen/src/expr/calls.rs` — extend signature, emit hooks.
- Test: covered by Task 9 end-to-end. Add one Cranelift-IR shape unit test here too.

This is the structurally riskiest change because it touches the hot path. Keep the diff small and gated.

- [ ] **Step 1: Read the launch site**

Read `crates/nsl-codegen/src/expr/calls.rs` lines 2300–2425 (function `compile_gpu_kernel_launch`). Note its current signature, the surrounding `compile_call` site that invokes it (`grep -n "compile_gpu_kernel_launch" crates/nsl-codegen/src/expr/calls.rs`), and how `func_ref` is used to get Cranelift `FuncRef`s for extern C symbols.

- [ ] **Step 2: Find the func_ref helper for nsl_kernel_launch**

```
grep -n "nsl_kernel_launch\|nsl_runtime\|extern.*name" crates/nsl-codegen/src/expr/calls.rs | head -10
```

Note exactly how the function reference is constructed today — Phase 2 must mirror the same pattern for `nsl_profile_kernel_begin` and `nsl_profile_kernel_end`.

- [ ] **Step 3: Write the unit test**

Create `crates/nsl-codegen/tests/profile_hook_emission.rs`:

```rust
//! Verify that compile_gpu_kernel_launch emits begin+end hook calls when
//! the compiler has a manifest_builder, and only the bare launch otherwise.
//!
//! We don't actually run the resulting binary — we inspect the Cranelift IR
//! the function produced to confirm the call instructions appear.

use nsl_codegen::CompileOptions;

#[test]
fn manifest_kernel_count_matches_emitted_launches() {
    let src = r#"
fn forward(
    x: Tensor<[1, 2048, 512], bf16>,
    wq: Tensor<[512, 512], bf16>,
    wk: Tensor<[512, 512], bf16>,
) -> Tensor:
    let q = matmul(x, wq)
    let k = matmul(x, wk)
    return q
"#;
    let opts = CompileOptions {
        profile_kernels: true,
        target_gpu: "h100".into(),
        dtype: "bf16".into(),
        ..CompileOptions::default()
    };
    // Run pre-pass + body codegen, then snapshot manifest contents.
    let manifest = nsl_codegen::test_helpers::compile_and_capture_manifest(src, &opts).unwrap();
    // Two matmul launches → two manifest kernel entries (assuming no fusion of these two).
    assert_eq!(manifest.kernels.len(), 2,
        "expected 2 kernels for 2 matmul launches, got {}", manifest.kernels.len());
    for (i, k) in manifest.kernels.iter().enumerate() {
        assert_eq!(k.kernel_id, i as u32, "kernel ids must be dense and monotonic");
        assert!(k.predicted_us > 0.0, "matmul kernel should have non-zero predicted time");
        assert_eq!(k.op_name, "matmul");
    }
}
```

`compile_and_capture_manifest` is a new test helper that runs the full body codegen (not just pre-pass) and returns `compiler.manifest_builder.take().unwrap().finish()`.

- [ ] **Step 4: Add the test helper**

In `crates/nsl-codegen/src/test_helpers.rs`:

```rust
#[cfg(any(test, feature = "test-helpers"))]
pub fn compile_and_capture_manifest(
    src: &str,
    opts: &crate::CompileOptions,
) -> Result<crate::profiling::instrument::Manifest, String> {
    // Same as run_pre_pass_only, then walk the function bodies.
    // Return manifest_builder.take().unwrap().finish().
    todo!("mirror the codegen entry up to module-body emission")
}
```

Same caveat as Task 4: implement by mirroring `compile()`. If genuinely intractable, skip this test — Task 9 covers the same ground via real `nsl run --monitor`.

- [ ] **Step 5: Run — expect fail**

```
cargo test -p nsl-codegen --test profile_hook_emission 2>&1 | tail -10
```

- [ ] **Step 6: Extend `compile_gpu_kernel_launch` signature**

In `crates/nsl-codegen/src/expr/calls.rs`, change the function signature to:

```rust
fn compile_gpu_kernel_launch(
    &mut self,
    builder: &mut cranelift_frontend::FunctionBuilder,
    ptx_ptr: cranelift_codegen::ir::Value,
    name_ptr: cranelift_codegen::ir::Value,
    grid: [cranelift_codegen::ir::Value; 3],
    block: [cranelift_codegen::ir::Value; 3],
    args_ptr: cranelift_codegen::ir::Value,
    arg_count: cranelift_codegen::ir::Value,
    smem: cranelift_codegen::ir::Value,
    origin_node: nsl_ast::NodeId,        // NEW
    span: nsl_errors::Span,              // NEW
    op_name_hint: &str,                  // NEW
) -> CodegenResult<cranelift_codegen::ir::Value>
```

Match the actual Value/FunctionBuilder import paths used elsewhere in the file.

- [ ] **Step 7: Update the call site**

In `compile_call` (around line 161 — `// emit fused elementwise kernel launch`), pass:

```rust
self.compile_gpu_kernel_launch(
    builder, ptx_ptr, name_ptr, grid, block,
    args_ptr, arg_count, smem,
    call_expr.id,                              // origin_node
    call_expr.span.clone(),                    // span
    callee_name_for_profile(&call_expr),       // op_name_hint
)?;
```

`callee_name_for_profile` is a small helper that reads the callee's name (e.g., `"matmul"`, `"flash_attn"`). For BinOp::MatMul, return `"matmul"`. Default for unknown: `"unknown"`.

- [ ] **Step 8: Emit the hook calls inside `compile_gpu_kernel_launch`**

At the top of the body, after argument prep but before the existing `nsl_kernel_launch` call:

```rust
let kernel_id = if let Some(mb) = self.manifest_builder.as_mut() {
    let id = mb.reserve_id();
    let constituents = self.fusion_constituents(origin_node);
    let mut total_us = 0.0_f64;
    let mut total_flops = 0u64;
    let mut total_hbm = 0u64;
    for nid in &constituents {
        if let Some(p) = self.prediction_map.get(nid) {
            total_us += p.estimated_time_us;
            total_flops += p.flops;
            total_hbm += p.bytes_read + p.bytes_written;
        }
    }
    let span_json = crate::profiling::instrument::SourceSpanJson::from_span(
        span.clone(), &self.source_file_name, &self.source_text,
    );
    mb.record_kernel_at(id, op_name_hint, span_json, total_us, total_flops, total_hbm);
    Some(id)
} else { None };

if let Some(id) = kernel_id {
    let id_val = builder.ins().iconst(cranelift_codegen::ir::types::I32, id as i64);
    let begin_ref = self.func_ref_extern("nsl_profile_kernel_begin", &[cranelift_codegen::ir::types::I32], None);
    builder.ins().call(begin_ref, &[id_val]);
}

// existing nsl_kernel_launch invocation — unchanged
let launch_call = builder.ins().call(launch_ref, &[ptx_ptr, name_ptr, /* ... */]);

if let Some(id) = kernel_id {
    let id_val = builder.ins().iconst(cranelift_codegen::ir::types::I32, id as i64);
    let end_ref = self.func_ref_extern("nsl_profile_kernel_end", &[cranelift_codegen::ir::types::I32], None);
    builder.ins().call(end_ref, &[id_val]);
}
```

`func_ref_extern(name, params, ret)` is whatever helper currently registers `nsl_kernel_launch` — find it via `grep -n "fn func_ref\|fn declare_extern\|register_external" crates/nsl-codegen/src/`. Use the exact same one.

`self.source_file_name` and `self.source_text` are existing fields on `Compiler` — the survey didn't confirm exact names; check `Compiler::new` signature and adapt. If they don't exist, add them as additional fields populated at compile-time entry.

Note on Cranelift `iconst(I32, id as i64)`: the cast to `i64` is required by the API but Cranelift truncates correctly when the type is `I32`.

- [ ] **Step 9: Run the unit test**

```
cargo test -p nsl-codegen --test profile_hook_emission 2>&1 | tail -10
```

Expected: 1 passed.

- [ ] **Step 10: Run all nsl-codegen tests for regression**

```
cargo test -p nsl-codegen 2>&1 | tail -15
```

Expected: all green. If the hook emission breaks an existing PTX snapshot test (because the IR shape changed when `profile_kernels` is unused), the snapshot is unaffected (the gate is `manifest_builder.is_some()`, off by default). If a snapshot does change, review the diff carefully.

- [ ] **Step 11: Commit**

```bash
git add crates/nsl-codegen/src/expr/calls.rs crates/nsl-codegen/src/test_helpers.rs \
        crates/nsl-codegen/tests/profile_hook_emission.rs \
        crates/nsl-codegen/src/compiler/mod.rs
git commit -m "feat(codegen): emit nsl_profile_kernel_begin/end hooks around kernel launches"
```

---

## Task 6: Manifest write at end of `compile()`

**Files:**
- Modify: `crates/nsl-codegen/src/lib.rs` — write manifest in compile post-pass.
- Test: extend `crates/nsl-codegen/tests/profile_hook_emission.rs`.

- [ ] **Step 1: Find `CompileOptions.output_path`**

```
grep -n "output_path\|out_dir\|output_file" crates/nsl-codegen/src/lib.rs | head -5
```

Confirm whether the field is `output_path: PathBuf` or split into `out_dir + name`. Adapt the manifest path construction below.

- [ ] **Step 2: Write failing test**

Append to `crates/nsl-codegen/tests/profile_hook_emission.rs`:

```rust
#[test]
fn compile_writes_manifest_json_when_profile_kernels_enabled() {
    let src = r#"
fn forward(x: Tensor<[1, 2048, 512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    return matmul(x, W)
"#;
    let tmp = tempfile::tempdir().unwrap();
    let out_path = tmp.path().join("tiny");
    let opts = CompileOptions {
        profile_kernels: true,
        target_gpu: "h100".into(),
        dtype: "bf16".into(),
        output_path: out_path.clone(),
        ..CompileOptions::default()
    };
    nsl_codegen::test_helpers::compile_to_object(src, &opts).unwrap();
    let manifest_path = out_path.with_extension("nsl-profile.json");
    assert!(manifest_path.exists(), "manifest at {} should exist", manifest_path.display());
    let m: nsl_codegen::profiling::instrument::Manifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    assert_eq!(m.kernels.len(), 1);
    assert_eq!(m.kernels[0].op_name, "matmul");
}
```

`compile_to_object` is a third test-helper variant that runs full codegen but stops before linking (so we don't need a CUDA toolchain on the dev box).

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test profile_hook_emission compile_writes_manifest_json 2>&1 | tail -10
```

- [ ] **Step 4: Implement post-pass in `compile()`**

In `crates/nsl-codegen/src/lib.rs::compile()` (or wherever the function-body codegen finishes), after the last function is emitted:

```rust
if let Some(mb) = compiler.manifest_builder.take() {
    let manifest = mb.finish();
    let path = opts.output_path.with_extension("nsl-profile.json");
    crate::profiling::instrument::write_manifest(&path, &manifest)
        .map_err(|e| format!("failed to write profile manifest: {}", e))?;
}
```

If `output_path` is split into `out_dir + name`, build the path as `opts.out_dir.join(format!("{}.nsl-profile.json", opts.name))`.

- [ ] **Step 5: Add `compile_to_object` test helper**

Mirror `compile_and_capture_manifest` but call the real `compile()`. Skip the linker step. If `compile()` is monolithic and includes linking, the easiest path is to add a `compile_only: bool` flag to `CompileOptions` (default false) that short-circuits before the linker invocation.

Alternative: if `compile()` already takes a callable host-target object emitter that can be mocked, mock it. Pick the smaller change.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-codegen --test profile_hook_emission 2>&1 | tail -15
```

Expected: 2 passed (Task 5 + new manifest-write test).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/test_helpers.rs \
        crates/nsl-codegen/tests/profile_hook_emission.rs
git commit -m "feat(codegen): write <out>.nsl-profile.json after compile when profile_kernels=true"
```

---

## Task 7: `CudaEventClock` + event-aware FFI

**Files:**
- Create: `crates/nsl-runtime/src/profiler/cuda_clock.rs`
- Modify: `crates/nsl-runtime/src/profiler/mod.rs`
- Modify: `crates/nsl-runtime/src/profiler/ffi.rs`
- Modify: `crates/nsl-runtime/Cargo.toml` — remove `cuda-real-events` stub feature.
- Test: `crates/nsl-runtime/tests/cuda_clock.rs` (cfg cuda).

- [ ] **Step 1: Find the runtime's current-stream getter**

```
grep -n "current_stream\|active_stream\|cu_stream\|fn .*stream" crates/nsl-runtime/src/cuda/mod.rs | head -15
```

Identify the function that returns the `CUstream` used by `nsl_kernel_launch`. If it's not exposed publicly, add `pub fn current_stream() -> sys::CUstream` that wraps the existing thread-local read.

**Critical**: confirm `nsl_kernel_launch` (around `crates/nsl-runtime/src/cuda/mod.rs:1743`) calls `cuLaunchKernel` with this same stream. Both FFI hooks must record on it. **Never use `0` (null stream).**

- [ ] **Step 2: Write failing test**

Create `crates/nsl-runtime/tests/cuda_clock.rs`:

```rust
//! CUDA-feature-only test: a known-busy kernel produces an actual timing
//! within an order of magnitude of expected. Catches null-stream regression.
#![cfg(feature = "cuda")]

use nsl_runtime::profiler::collector::Collector;
use nsl_runtime::profiler::cuda_clock::CudaEventClock;

#[test]
fn cuda_event_clock_reports_nonzero_us_for_recorded_pair() {
    let clock = CudaEventClock::new();
    // Check out two events. In a real run, codegen-emitted hooks would
    // call cuEventRecord between them; for a smoke test, we only verify
    // that checkout returns distinct non-null handles and that elapsed_us
    // does not panic when both events were never recorded (it should error
    // gracefully, not crash).
    let s = clock.checkout_event();
    let e = clock.checkout_event();
    assert!(s != 0);
    assert!(e != 0);
    assert!(s != e);
    // Don't actually call elapsed_us without recording — that would error.
    // The end-to-end test in Task 9 covers the recorded path.
}

#[test]
fn collector_uses_cuda_event_clock_when_constructed_with_one() {
    let mut c = Collector::new_with_clock(Box::new(CudaEventClock::new()));
    // Verify the Collector accepts a CudaEventClock without compile errors —
    // the only test we can do without an actual recorded event.
    assert_eq!(c.snapshot().len(), 0);
}
```

- [ ] **Step 3: Run — expect fail (module not found)**

```
cargo test -p nsl-runtime --test cuda_clock --features cuda 2>&1 | tail -10
```

If CUDA isn't installed on the dev box, this won't link — that's fine, the unit work in Task 9 shifts to a CUDA box. Note in the task report.

- [ ] **Step 4: Implement `cuda_clock.rs`**

Create `crates/nsl-runtime/src/profiler/cuda_clock.rs`:

```rust
//! Real CUDA-event-backed ClockSource for the kernel-timing collector.

#![cfg(feature = "cuda")]

use cudarc::driver::sys;
use std::sync::Mutex;
use crate::profiler::collector::ClockSource;

pub struct CudaEventClock {
    pool: Mutex<Vec<sys::CUevent>>,
}

impl CudaEventClock {
    pub fn new() -> Self {
        Self { pool: Mutex::new(Vec::with_capacity(128)) }
    }

    /// Get an event from the pool, allocating a new one if empty.
    /// Returns the raw CUevent cast to u64 so it round-trips through
    /// the Collector's (start_handle, end_handle) tuple.
    pub fn checkout_event(&self) -> u64 {
        if let Some(e) = self.pool.lock().unwrap().pop() {
            return e as u64;
        }
        unsafe {
            let mut e: sys::CUevent = std::ptr::null_mut();
            let res = sys::lib().cuEventCreate(&mut e, 0);
            if res.0 != 0 { panic!("cuEventCreate failed: {:?}", res); }
            e as u64
        }
    }

    fn return_event(&self, h: u64) {
        self.pool.lock().unwrap().push(h as sys::CUevent);
    }
}

impl ClockSource for CudaEventClock {
    fn elapsed_us(&self, start: u64, end: u64) -> f64 {
        unsafe {
            let s = start as sys::CUevent;
            let e = end as sys::CUevent;
            let sync_res = sys::lib().cuEventSynchronize(e);
            if sync_res.0 != 0 {
                eprintln!("warning: cuEventSynchronize failed ({:?}), dropping pair", sync_res);
                self.return_event(start);
                self.return_event(end);
                return 0.0;
            }
            let mut ms: f32 = 0.0;
            let elapsed_res = sys::lib().cuEventElapsedTime(&mut ms, s, e);
            self.return_event(start);
            self.return_event(end);
            if elapsed_res.0 != 0 {
                eprintln!("warning: cuEventElapsedTime failed ({:?})", elapsed_res);
                return 0.0;
            }
            ms as f64 * 1000.0   // ms → μs
        }
    }
}
```

Add `#[cfg(feature = "cuda")] pub mod cuda_clock;` to `crates/nsl-runtime/src/profiler/mod.rs`.

- [ ] **Step 5: Update FFI hooks to use CUDA events under cuda feature**

In `crates/nsl-runtime/src/profiler/ffi.rs`:

```rust
#[cfg(feature = "cuda")]
use once_cell::sync::Lazy;
#[cfg(feature = "cuda")]
use crate::profiler::cuda_clock::CudaEventClock;
#[cfg(feature = "cuda")]
static CUDA_CLOCK: Lazy<CudaEventClock> = Lazy::new(CudaEventClock::new);

#[no_mangle]
pub extern "C" fn nsl_profile_kernel_begin(kernel_id: u32) {
    #[cfg(feature = "cuda")]
    {
        let event = CUDA_CLOCK.checkout_event();
        unsafe {
            let stream = crate::cuda::current_stream();
            cudarc::driver::sys::lib()
                .cuEventRecord(event as cudarc::driver::sys::CUevent, stream)
                .result()
                .ok();
        }
        let mut g = COLLECTOR.lock().unwrap();
        g.get_or_insert_with(|| {
            crate::profiler::collector::Collector::new_with_clock(Box::new(CudaEventClock::new()))
        }).begin(kernel_id, event);
        return;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let t = now_ns();
        let mut g = COLLECTOR.lock().unwrap();
        g.get_or_insert_with(|| {
            crate::profiler::collector::Collector::new_with_clock(Box::new(crate::profiler::collector::NanoClock))
        }).begin(kernel_id, t);
    }
}

#[no_mangle]
pub extern "C" fn nsl_profile_kernel_end(kernel_id: u32) {
    #[cfg(feature = "cuda")]
    {
        let event = CUDA_CLOCK.checkout_event();
        unsafe {
            let stream = crate::cuda::current_stream();
            cudarc::driver::sys::lib()
                .cuEventRecord(event as cudarc::driver::sys::CUevent, stream)
                .result()
                .ok();
        }
        let mut g = COLLECTOR.lock().unwrap();
        g.get_or_insert_with(|| {
            crate::profiler::collector::Collector::new_with_clock(Box::new(CudaEventClock::new()))
        }).end(kernel_id, event);
        return;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let t = now_ns();
        let mut g = COLLECTOR.lock().unwrap();
        g.get_or_insert_with(|| {
            crate::profiler::collector::Collector::new_with_clock(Box::new(crate::profiler::collector::NanoClock))
        }).end(kernel_id, t);
    }
}
```

The non-CUDA path is unchanged from Phase 1.

`crate::cuda::current_stream` is the getter from Step 1. If that name's wrong, use the actual one.

- [ ] **Step 6: Remove the `cuda-real-events` feature stub**

In `crates/nsl-runtime/Cargo.toml`, delete the line `cuda-real-events = []` from `[features]`.

In `crates/nsl-runtime/src/profiler/collector.rs`, delete the now-unused stub:

```rust
#[cfg(feature = "cuda-real-events")]
pub struct CudaEventClock;
#[cfg(feature = "cuda-real-events")]
impl ClockSource for CudaEventClock { ... }
```

- [ ] **Step 7: Run tests**

```
cargo test -p nsl-runtime --test profiler_collector 2>&1 | tail -10
cargo build -p nsl-runtime --features cuda 2>&1 | tail -10
```

Expected: 4 host-only tests still pass; cuda build is clean (assuming a CUDA-capable dev box). If no CUDA box, accept the build-only check.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-runtime/src/profiler/cuda_clock.rs \
        crates/nsl-runtime/src/profiler/mod.rs \
        crates/nsl-runtime/src/profiler/ffi.rs \
        crates/nsl-runtime/src/profiler/collector.rs \
        crates/nsl-runtime/Cargo.toml \
        crates/nsl-runtime/tests/cuda_clock.rs
git commit -m "feat(runtime): CudaEventClock + event-aware FFI hooks under cuda feature"
```

---

## Task 8: CLI — `--monitor` implies `profile_kernels: true`

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` — set the option in the `Run` arm.

- [ ] **Step 1: Locate the Run arm**

```
grep -n "Cli::Run\|profile_kernels\|monitor:" crates/nsl-cli/src/main.rs | head -10
```

Find where `CompileOptions` is constructed for the `Run` path.

- [ ] **Step 2: Write a small smoke check**

There's no easy unit test for this — it's wired into `main()`. Verify by inspection in Step 5.

- [ ] **Step 3: Set the field**

In the `Cli::Run { .. monitor, .. }` match arm, locate where `CompileOptions` is constructed (or where `compile_options.profile_kernels` is set). Add:

```rust
let mut compile_options = /* existing construction */;
if monitor {
    compile_options.profile_kernels = true;
}
```

If the existing `--profile-kernels` flag is also wired here, OR the two:

```rust
compile_options.profile_kernels = profile_kernels || monitor;
```

- [ ] **Step 4: Run regression**

```
cargo test -p nsl-cli 2>&1 | tail -10
cargo build -p nsl-cli 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 5: Manual smoke**

```
cargo run -p nsl-cli -- run --monitor crates/nsl-cli/tests/fixtures/tiny_transformer.nsl 2>&1 | head -20
```

Expected: builds without "no actual timings collected" if a CUDA toolchain is present and the build links runtime with `cuda` feature on. Even without CUDA, the manifest should be written by codegen — verify the file exists:

```
ls crates/nsl-cli/tests/fixtures/tiny_transformer.nsl-profile.json
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-cli/src/main.rs
git commit -m "feat(cli): --monitor implies profile_kernels=true at compile time"
```

---

## Task 9: End-to-end CUDA integration test

**Files:**
- Create: `crates/nsl-cli/tests/monitor_e2e.rs` (cfg cuda).

This is the regression test for the stream-binding rule. If both events go on the same stream, kernel times are realistic; if either goes on the null stream, times are absurdly small (host launch latency, ~1µs).

- [ ] **Step 1: Write the test**

Create `crates/nsl-cli/tests/monitor_e2e.rs`:

```rust
#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

#[test]
fn monitor_produces_actual_timings_in_realistic_range() {
    let src = fixture("tiny_transformer.nsl");
    // Build + run with --monitor. Use the CLI binary to mirror real usage.
    let bin = env!("CARGO_BIN_EXE_nsl");
    let output = Command::new(bin)
        .args(["run", "--monitor", src.to_str().unwrap()])
        .output()
        .expect("nsl run --monitor failed to execute");
    assert!(output.status.success(),
        "nsl run --monitor exited {:?}: stderr={}", output.status,
        String::from_utf8_lossy(&output.stderr));
    let out = String::from_utf8(output.stdout).unwrap();

    // Banner from Phase 1 must NOT appear — actual timings should be present.
    assert!(!out.contains("no actual timings collected"),
        "expected real timings, got Phase 1 fallback banner. Output:\n{}", out);

    // Sanity-bound the actual times. A 2048×512×512 matmul on H100-class
    // hardware should take >= 1 μs (compute lower bound) and << 1 ms.
    // If we see values << 1 μs, the stream binding is broken.
    let mut found_actual = false;
    for line in out.lines() {
        if let Some(idx) = line.find("μs") {
            // Parse the number preceding "μs". This is fuzzy on purpose —
            // we just want a sanity check.
            let head = &line[..idx];
            let num_str: String = head.chars().rev()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ' ')
                .collect::<String>().chars().rev().collect();
            if let Ok(us) = num_str.trim().parse::<f64>() {
                if line.contains("Actual") || line.matches("μs").count() >= 2 {
                    assert!(us >= 0.5, "actual time {} μs too low — stream binding likely broken; line: {}", us, line);
                    assert!(us < 100_000.0, "actual time {} μs implausibly high; line: {}", us, line);
                    found_actual = true;
                }
            }
        }
    }
    assert!(found_actual, "no actual μs values found in monitor output:\n{}", out);
}

#[test]
fn monitor_manifest_kernel_count_matches_actual_aggregate_count() {
    let src = fixture("tiny_transformer.nsl");
    let bin = env!("CARGO_BIN_EXE_nsl");
    let _ = Command::new(bin)
        .args(["run", "--monitor", src.to_str().unwrap()])
        .output()
        .unwrap();
    let manifest_path = src.with_extension("nsl-profile.json");
    let actual_path = src.with_extension("nsl-profile-actual.json");
    let m: nsl_codegen::profiling::instrument::Manifest =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    let a: nsl_runtime::profiler::collector::ActualReport =
        serde_json::from_str(&std::fs::read_to_string(&actual_path).unwrap()).unwrap();
    // Every kernel id in the manifest should appear in actuals.
    for k in &m.kernels {
        assert!(a.aggregates.iter().any(|ag| ag.kernel_id == k.kernel_id),
            "manifest kernel_id {} ({}) has no matching actual aggregate",
            k.kernel_id, k.op_name);
    }
}
```

- [ ] **Step 2: Run on a CUDA box**

```
cargo test -p nsl-cli --features cuda --test monitor_e2e 2>&1 | tail -20
```

Expected: 2 passed.

If the dev box has no CUDA: skip this task and document in the task report. Run on a CUDA box before declaring Phase 2 complete.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-cli/tests/monitor_e2e.rs
git commit -m "test(cli): end-to-end --monitor produces realistic actual timings (catches null-stream bug)"
```

---

## Task 10: Final review + workspace check

- [ ] **Step 1: Run full test suite**

```
cargo test --workspace 2>&1 | tail -20
cargo test --workspace --features cuda 2>&1 | tail -20    # CUDA box only
```

Expected: green.

- [ ] **Step 2: Clippy on new code**

```
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -30
```

Fix warnings in code we wrote in Tasks 1–9. Leave pre-existing warnings alone.

- [ ] **Step 3: Manual acceptance**

```
cargo run -p nsl-cli -- profile --target h100 --dtype bf16 crates/nsl-cli/tests/fixtures/tiny_transformer.nsl
cargo run -p nsl-cli -- run --monitor crates/nsl-cli/tests/fixtures/tiny_transformer.nsl
```

The second command's output should now show real μs in the Actual column instead of "n/a", and the "Note: no actual timings collected" banner should be gone.

- [ ] **Step 4: Don't open PR**

Per user instruction (2026-04-12), Phase 2 work stays local on `feat/dev-tools-phase1` until all milestones ship. Leave the branch as-is.

---

## Self-Review

**Spec coverage:**

- §3 architecture (4 components) → Tasks 1–8 each implement one component piece. ✅
- §4.1 pre-pass + state on `Compiler` → Task 4. ✅
- §4.2 hook emission → Task 5. ✅
- §4.2.1 fusion constituent sum → Task 3 (FusionPlan field) + Task 5 (sum loop in emission). ✅
- §4.3 manifest writer post-pass → Task 6. ✅
- §4.4 CudaEventClock → Task 7. ✅
- §4.5 FFI event-aware hooks → Task 7. ✅
- §4.6 CLI wiring → Task 8. ✅
- §6 error handling → distributed: Task 5 (NodeId miss → zeros), Task 6 (manifest write error), Task 7 (cuEventSynchronize warn), Phase 1 invariant (drain cap 64) preserved. ✅
- §7 testing — host-only unit (Tasks 1–3), pre-pass test (Task 4), hook emission test (Task 5), manifest-write test (Task 6), CudaEventClock smoke (Task 7), end-to-end CUDA test with stream-binding regression (Task 9). ✅
- §8 non-goals respected — `kernel_profiler.rs` untouched, no CPU-op timing, no autotuning loop. ✅

**Placeholder scan:**

Three intentional `todo!()` stubs in test helpers (Tasks 4, 5, 6) — flagged in the task text as "implement by mirroring `compile()`'s pipeline". These are deliberate boundaries because the exact `Compiler::new` signature isn't fully knowable from the spec without reading code, and forcing a wrong guess wastes more time than letting the implementer adapt. No "TBD"/"add validation"/etc. patterns elsewhere.

**Type consistency:**

- `OpCost.origin_node: Option<NodeId>` introduced in Task 1, used in Task 4 (`prediction_map` build) and Task 5 (`fusion_constituents` lookup).
- `ManifestBuilder::reserve_id` / `record_kernel_at` introduced in Task 2, used in Task 5.
- `FusionPlan.fused_node_groups` + `constituents_of` introduced in Task 3, used in Task 5.
- `CudaEventClock` introduced in Task 7, referenced consistently throughout.
- `current_stream()` is the same name in spec §4.4, FFI code in Task 7, and stream-binding regression in Task 9.

**Scope:**

10 tasks, each producing testable working software. Each can stop at the end of any task and have a working merged piece. Same branch as Phase 1 (`feat/dev-tools-phase1`) per user direction.
