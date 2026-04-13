# NSL Dev Tools — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship four compiler-native dev tools in ~1,400 LOC: `nsl check --shapes`, `nsl profile`, `nsl profile --memory`, and `nsl run --monitor` (with source-mapped kernel view).

**Architecture:** A single shared walker module (`nsl-codegen/src/profiling/`) produces a per-op `ProfileReport` from a typed AST. Each CLI feature is a thin frontend over the walker plus one existing subsystem (`MemoryPlan`, codegen kernel emitter, runtime collector). Instrumentation is gated by a new `profile: bool` plumbed through codegen — when off, the old code path runs unchanged.

**Tech Stack:** Rust workspace (`cargo`), `clap` for CLI, `serde_json` for manifests/reports, `insta` for snapshot tests, `cudarc` 0.19 for runtime CUDA events.

**Spec:** `docs/superpowers/specs/2026-04-12-nsl-dev-tools-phase1-design.md`

**Execution note:** Run this plan inside a dedicated worktree — see Task 0.

---

## Task 0: Worktree + baseline

**Files:** none (environment setup).

- [ ] **Step 1: Verify `.worktrees/` is gitignored**

```bash
cd c:/Users/bwiem/projects/NSL
git check-ignore -q .worktrees && echo "ok" || echo "NOT IGNORED"
```

If "NOT IGNORED", add `.worktrees/` to `.gitignore`, commit, then proceed.

- [ ] **Step 2: Create the worktree**

```bash
git worktree add .worktrees/dev-tools-phase1 -b feat/dev-tools-phase1
cd .worktrees/dev-tools-phase1
```

- [ ] **Step 3: Baseline test run**

```bash
cargo test --workspace --no-run
cargo test --workspace 2>&1 | tail -30
```

Record pass/fail count. Proceed only if baseline is green (or failures are pre-existing and documented in `project_outstanding_work.md`).

- [ ] **Step 4: Commit the worktree baseline marker**

```bash
git commit --allow-empty -m "chore: start dev-tools Phase 1 on feat/dev-tools-phase1"
```

---

## Task 1: Scaffold `profiling/` module + `profile: bool` codegen option

**Files:**
- Create: `crates/nsl-codegen/src/profiling/mod.rs`
- Create: `crates/nsl-codegen/src/profiling/types.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod profiling;`)
- Test: `crates/nsl-codegen/tests/profiling_types.rs`

Types in this module are the shared vocabulary used by every later task. Locking them in first avoids churn.

- [ ] **Step 1: Write failing test for `ProfileReport` construction**

Create `crates/nsl-codegen/tests/profiling_types.rs`:

```rust
use nsl_codegen::profiling::types::{EntryKind, ProfileReport, Recommendation};

#[test]
fn empty_report_round_trips_json() {
    let r = ProfileReport {
        target_gpu: "h100-sxm".to_string(),
        dtype: "bf16".to_string(),
        entry: EntryKind::Auto,
        ops: vec![],
        total_flops: 0,
        total_hbm_bytes: 0,
        total_estimated_us: 0.0,
        fusion: None,
        memory_timeline: None,
        recommendations: vec![],
    };
    let j = serde_json::to_string(&r).unwrap();
    let back: ProfileReport = serde_json::from_str(&j).unwrap();
    assert_eq!(back.target_gpu, "h100-sxm");
    assert_eq!(back.entry, EntryKind::Auto);
}

#[test]
fn entry_kind_parses_from_flag() {
    assert_eq!(EntryKind::parse_flag("auto"), Some(EntryKind::Auto));
    assert_eq!(EntryKind::parse_flag("train"), Some(EntryKind::Train));
    assert_eq!(EntryKind::parse_flag("fn:forward"),
               Some(EntryKind::Function("forward".into())));
    assert!(EntryKind::parse_flag("bogus").is_none());
}

#[test]
fn recommendation_has_code_and_message() {
    let rec = Recommendation::memory_bound_batch_hint("gate_proj");
    assert!(rec.code.starts_with("R0"));
    assert!(rec.message.contains("gate_proj"));
}
```

- [ ] **Step 2: Run — expect compile failure (types don't exist)**

```bash
cargo test -p nsl-codegen --test profiling_types 2>&1 | tail -15
```

Expected: "unresolved import `nsl_codegen::profiling`".

- [ ] **Step 3: Create the module and types**

`crates/nsl-codegen/src/profiling/mod.rs`:

```rust
pub mod types;
```

`crates/nsl-codegen/src/profiling/types.rs`:

```rust
use crate::cost_model::OpCost;
use crate::wrga_fusion::FusionPlan;
use crate::wrga_memory::MemoryPlan;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntryKind {
    Auto,
    Train,
    Function(String),
}

impl EntryKind {
    pub fn parse_flag(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Self::Auto),
            "train" => Some(Self::Train),
            other if other.starts_with("fn:") => {
                Some(Self::Function(other[3..].to_string()))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub code: String,       // R01..R99
    pub message: String,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Severity { Info, Warn }

impl Recommendation {
    pub fn memory_bound_batch_hint(op: &str) -> Self {
        Self {
            code: "R01".into(),
            message: format!("{op} is memory-bound (AI < 2). Batch size > 4 would improve utilization."),
            severity: Severity::Info,
        }
    }
    pub fn dominating_op(op: &str, pct: f64) -> Self {
        Self {
            code: "R02".into(),
            message: format!("{op} dominates ({pct:.1}% of total time). Consider INT4 quantization."),
            severity: Severity::Warn,
        }
    }
    pub fn fusion_strongly_recommended(kernels_saved: usize) -> Self {
        Self {
            code: "R03".into(),
            message: format!("Fusion eliminates {kernels_saved} kernel launches per forward. Strongly recommended."),
            severity: Severity::Info,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimelineEntry {
    pub program_point: u32,
    pub live_bytes: u64,
    pub phase: Option<String>, // e.g. "forward_start"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    pub target_gpu: String,
    pub dtype: String,
    pub entry: EntryKind,
    pub ops: Vec<OpCost>,
    pub total_flops: u64,
    pub total_hbm_bytes: u64,
    pub total_estimated_us: f64,
    pub fusion: Option<FusionPlan>,
    pub memory_timeline: Option<Vec<MemoryTimelineEntry>>,
    pub recommendations: Vec<Recommendation>,
}
```

Add `#[derive(Serialize, Deserialize)]` to `OpCost`, `BoundClassification` in `crates/nsl-codegen/src/cost_model.rs`, and to `FusionPlan`, `FusionDecision`, `FusionTarget`, `SiteKind` in `wrga_fusion.rs` / `wrga_roofline.rs` if not already present. If `serde` isn't in `nsl-codegen/Cargo.toml` yet, add `serde = { version = "1", features = ["derive"] }` and `serde_json = "1"`.

Add `pub mod profiling;` to `crates/nsl-codegen/src/lib.rs`.

- [ ] **Step 4: Run tests — expect pass**

```bash
cargo test -p nsl-codegen --test profiling_types
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/profiling/ crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/src/cost_model.rs crates/nsl-codegen/src/wrga_fusion.rs \
        crates/nsl-codegen/Cargo.toml crates/nsl-codegen/tests/profiling_types.rs
git commit -m "feat(profiling): scaffold profiling module with ProfileReport types"
```

---

## Task 2: `nsl check --shapes` (Feature 1)

**Files:**
- Create: `crates/nsl-cli/src/shape_debug.rs`
- Modify: `crates/nsl-cli/src/main.rs` (add `shapes: bool` flag on `Check` variant + dispatch)
- Test: `crates/nsl-cli/tests/shape_debug.rs`

The semantic pass already propagates tensor shapes. This task surfaces that propagation as a printed trace.

- [ ] **Step 1: Write failing test for the shape debugger core**

`crates/nsl-cli/tests/shape_debug.rs`:

```rust
use nsl_cli::shape_debug::{format_trace, ShapeDebugInput};

const SRC: &str = r#"
fn forward(x: Tensor<[batch=8, seq=2048, d=512], bf16>) -> Tensor:
    let h = layernorm(x)
    let y = matmul(h, W)
    return y
"#;

#[test]
fn trace_emits_one_line_per_typed_let() {
    let input = ShapeDebugInput::from_source(SRC, "model.nsl").unwrap();
    let out = format_trace(&input);
    assert!(out.contains("layernorm(x)"));
    assert!(out.contains("[8, 2048, 512]"));
    assert!(out.contains("matmul(h, W)"));
    assert!(out.lines().filter(|l| l.contains("✅")).count() >= 2);
}

#[test]
fn trace_final_line_reports_total_flops() {
    let input = ShapeDebugInput::from_source(SRC, "model.nsl").unwrap();
    let out = format_trace(&input);
    let last = out.lines().rev().find(|l| l.contains("Total FLOPs")).unwrap();
    assert!(last.contains("FLOP"));
}

#[test]
fn shape_mismatch_renders_error_block() {
    let bad = r#"
fn forward(x: Tensor<[8, 2048, 512], bf16>) -> Tensor:
    let y: Tensor<[8, 2048, 384], bf16> = x   # mismatch
    return y
"#;
    let input = ShapeDebugInput::from_source(bad, "bad.nsl").unwrap();
    let out = format_trace(&input);
    assert!(out.contains("❌"));
    assert!(out.contains("Expected"));
    assert!(out.contains("Cause"));
}
```

- [ ] **Step 2: Run — expect compile failure**

```bash
cargo test -p nsl-cli --test shape_debug 2>&1 | tail -10
```

Expected: "unresolved import `nsl_cli::shape_debug`".

- [ ] **Step 3: Implement `shape_debug.rs`**

`crates/nsl-cli/src/shape_debug.rs`:

```rust
//! Compile-time shape-propagation trace for `nsl check --shapes`.
//!
//! Read-only formatter over the output of the semantic analyzer. Walks the
//! typed AST and prints one line per top-level / let-bound expression that
//! produces a `Tensor<...>`.

use nsl_ast::{Module, stmt::Stmt, expr::Expr};
use nsl_errors::DiagnosticBag;
use std::fmt::Write;

pub struct ShapeDebugInput {
    pub module: Module,
    pub diagnostics: DiagnosticBag,
    pub source: String,
    pub file_name: String,
}

impl ShapeDebugInput {
    pub fn from_source(src: &str, file: &str) -> Result<Self, String> {
        // Reuse the existing driver pipeline: lex → parse → semantic-analyze.
        // The exact call is `nsl_cli::driver::analyze(src, file)` — see how
        // `Cli::Check` currently does it in `main.rs`.
        let (module, diagnostics) = crate::driver::analyze(src, file)
            .map_err(|e| format!("parse failed: {e}"))?;
        Ok(Self { module, diagnostics, source: src.to_string(), file_name: file.into() })
    }
}

pub fn format_trace(input: &ShapeDebugInput) -> String {
    let mut out = String::new();
    writeln!(out, "=== Shape Debugger ===").unwrap();

    // Print function signature's input shapes first.
    render_signature(&mut out, &input.module);

    writeln!(out, "\nPropagation:").unwrap();
    let mut total_flops: u64 = 0;
    for stmt in module_statements(&input.module) {
        render_stmt(&mut out, stmt, &input.diagnostics, &mut total_flops);
    }

    // Diagnostic errors (shape mismatches) → rich "Expected / Cause / Fix" block.
    for d in input.diagnostics.errors() {
        if d.is_shape_mismatch() {
            render_shape_error(&mut out, d, &input.source);
        }
    }

    if input.diagnostics.has_errors() {
        writeln!(out, "\n{} mismatch(es) detected.", input.diagnostics.error_count()).unwrap();
    } else {
        writeln!(out, "\nAll shapes valid. No mismatches detected.").unwrap();
    }
    writeln!(out, "Total FLOPs: {:.2} GFLOP per forward pass.",
             total_flops as f64 / 1e9).unwrap();
    out
}

fn module_statements(m: &Module) -> impl Iterator<Item = &Stmt> {
    m.stmts.iter()
}

fn render_signature(out: &mut String, m: &Module) {
    // For each FnDef at top level, print the params as "name: Tensor<shape>".
    for stmt in &m.stmts {
        if let Stmt::FnDef(f) = stmt {
            writeln!(out, "Input: {}", format_fn_params(f)).unwrap();
        }
    }
}

fn format_fn_params(f: &nsl_ast::decl::FnDef) -> String {
    f.params.iter()
        .map(|p| format!("{}: {}", p.name, p.ty.as_ref().map(|t| format!("{t:?}"))
            .unwrap_or_default()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn render_stmt(out: &mut String, stmt: &Stmt, diags: &DiagnosticBag, total_flops: &mut u64) {
    match stmt {
        Stmt::Let(l) => {
            let src_snippet = render_source_snippet(&l.expr);
            let shape = resolved_shape(&l.expr);
            let ok = diags.has_error_at(l.span) == false;
            let mark = if ok { "✅" } else { "❌" };
            writeln!(out, "  {:40} → {:20} {}", src_snippet, shape, mark).unwrap();
            *total_flops += estimate_flops(&l.expr);
        }
        Stmt::FnDef(f) => {
            for inner in &f.body.stmts {
                render_stmt(out, inner, diags, total_flops);
            }
        }
        _ => {}
    }
}

fn render_source_snippet(e: &Expr) -> String {
    // Best-effort: pretty-print the expression shorthand. If that isn't
    // available, fall back to `"<expr>"`.
    format!("{e}")  // Relies on Expr: Display in nsl-ast; add if missing.
}

fn resolved_shape(e: &Expr) -> String {
    // The semantic pass attaches a resolved type to each Expr via `e.inferred_type`.
    // Render it as `[d0, d1, ...]` shape only.
    e.inferred_type
        .as_ref()
        .map(|t| t.render_shape_only())
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn estimate_flops(e: &Expr) -> u64 {
    // Lightweight — call into `profiling::walker::op_flops_for_expr(e)` to
    // reuse the walker's cost logic. See Task 3.
    crate::profile_glue::op_flops_for_expr(e).unwrap_or(0)
}

fn render_shape_error(out: &mut String, d: &nsl_errors::Diagnostic, src: &str) {
    writeln!(out, "\n  ❌ MISMATCH").unwrap();
    writeln!(out, "    Expected: {}", d.expected_shape()).unwrap();
    writeln!(out, "    Cause:    {}", d.cause_summary()).unwrap();
    writeln!(out, "    Fix:      {}", d.suggested_fix()).unwrap();
    writeln!(out, "    Source:   {}", d.source_line(src)).unwrap();
}
```

Note: several methods referenced above (`inferred_type`, `render_shape_only`, `DiagnosticBag::has_error_at`, `Diagnostic::is_shape_mismatch`/`expected_shape`/`cause_summary`/`suggested_fix`/`source_line`) may need small additions in `nsl-ast` / `nsl-semantic` / `nsl-errors`. If any is missing, add a minimal implementation in the same commit — prefer adding thin accessors over changing semantics.

Add `pub mod shape_debug;` to `crates/nsl-cli/src/lib.rs` (create the lib target if the crate is binary-only — split `main.rs` so `lib.rs` holds shared modules and `main.rs` only holds the `main()` entry point).

- [ ] **Step 4: Wire the CLI flag**

In `crates/nsl-cli/src/main.rs`, on the `Check` variant add:

```rust
#[arg(long)]
shapes: bool,
```

Then in the `Cli::Check { .. shapes, .. }` match arm, if `shapes` is true:

```rust
if shapes {
    let src = std::fs::read_to_string(&file)?;
    let input = nsl_cli::shape_debug::ShapeDebugInput::from_source(&src, file.to_str().unwrap())?;
    println!("{}", nsl_cli::shape_debug::format_trace(&input));
    return Ok(());
}
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p nsl-cli --test shape_debug
```

Expected: 3 passed.

- [ ] **Step 6: Manual smoke test**

```bash
cargo run -p nsl-cli -- check --shapes examples/tiny_transformer.nsl
```

Expected: tabular shape trace with ✅ marks and final "Total FLOPs" line.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-cli/src/shape_debug.rs crates/nsl-cli/src/main.rs \
        crates/nsl-cli/src/lib.rs crates/nsl-cli/tests/shape_debug.rs
git commit -m "feat(cli): nsl check --shapes prints compile-time shape trace"
```

---

## Task 3: Shared walker core (`walk_ops` + `ShapeEnv`)

**Files:**
- Create: `crates/nsl-codegen/src/profiling/walker.rs`
- Create: `crates/nsl-codegen/src/profiling/shape_env.rs`
- Test: `crates/nsl-codegen/tests/profiling_walker.rs`

The walker is the shared backbone of Features 2–4. It takes a typed module plus a `ShapeEnv` plus `&GpuSpec` and emits a flat `Vec<OpCost>` in program order, with source spans attached.

- [ ] **Step 1: Write failing test for `ShapeEnv`**

Start `crates/nsl-codegen/tests/profiling_walker.rs`:

```rust
use nsl_codegen::profiling::shape_env::ShapeEnv;

#[test]
fn shape_env_resolves_defaults() {
    let env = ShapeEnv::with_defaults();
    assert_eq!(env.resolve("batch"), Some(1));
    assert_eq!(env.resolve("seq"), Some(2048));
}

#[test]
fn shape_env_overrides_from_flags() {
    let mut env = ShapeEnv::with_defaults();
    env.set("batch", 8);
    env.set("heads", 12);
    assert_eq!(env.resolve("batch"), Some(8));
    assert_eq!(env.resolve("heads"), Some(12));
}

#[test]
fn shape_env_unknown_returns_none() {
    let env = ShapeEnv::with_defaults();
    assert!(env.resolve("mystery_dim").is_none());
}
```

- [ ] **Step 2: Run — expect fail**

```bash
cargo test -p nsl-codegen --test profiling_walker 2>&1 | tail -10
```

- [ ] **Step 3: Implement `shape_env.rs`**

```rust
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct ShapeEnv(HashMap<String, u64>);

impl ShapeEnv {
    pub fn new() -> Self { Self::default() }

    pub fn with_defaults() -> Self {
        let mut m = HashMap::new();
        m.insert("batch".into(), 1);
        m.insert("seq".into(), 2048);
        Self(m)
    }

    pub fn set(&mut self, name: &str, value: u64) {
        self.0.insert(name.to_string(), value);
    }

    pub fn resolve(&self, name: &str) -> Option<u64> {
        self.0.get(name).copied()
    }

    pub fn parse_dim_flag(&mut self, flag: &str) -> Result<(), String> {
        let (k, v) = flag.split_once('=').ok_or_else(|| format!("expected name=N, got {flag}"))?;
        let n: u64 = v.parse().map_err(|_| format!("not a number: {v}"))?;
        self.set(k, n);
        Ok(())
    }
}
```

Add `pub mod shape_env;` to `profiling/mod.rs`.

- [ ] **Step 4: Run — expect 3 passed**

```bash
cargo test -p nsl-codegen --test profiling_walker
```

- [ ] **Step 5: Commit partial**

```bash
git add crates/nsl-codegen/src/profiling/shape_env.rs crates/nsl-codegen/src/profiling/mod.rs crates/nsl-codegen/tests/profiling_walker.rs
git commit -m "feat(profiling): ShapeEnv for resolving tensor dim flags"
```

- [ ] **Step 6: Add walker tests**

Append to `profiling_walker.rs`:

```rust
use nsl_codegen::profiling::types::{EntryKind, ProfileReport};
use nsl_codegen::profiling::walker::walk_ops;
use nsl_codegen::gpu_specs::find_gpu;

fn parse(src: &str) -> nsl_ast::Module {
    // Reuse the workspace's parse-and-analyze helper.
    nsl_cli::driver::analyze(src, "test.nsl").unwrap().0
}

#[test]
fn walks_single_matmul() {
    let src = r#"
fn forward(x: Tensor<[B=1, S=2048, D=512], bf16>, W: Tensor<[512, 512], bf16>) -> Tensor:
    let y = matmul(x, W)
    return y
"#;
    let m = parse(src);
    let gpu = find_gpu("h100").unwrap();
    let env = ShapeEnv::with_defaults();
    let r = walk_ops(&m, EntryKind::Auto, &env, gpu, "bf16").unwrap();
    assert_eq!(r.ops.len(), 1);
    assert_eq!(r.ops[0].name, "matmul");
    // 2 * M * K * N = 2 * 2048 * 512 * 512 = 1_073_741_824
    assert_eq!(r.ops[0].flops, 2 * 2048 * 512 * 512);
    assert!(r.total_flops > 0);
}

#[test]
fn walker_inlines_model_forward_from_train() {
    let src = r#"
model Tiny:
    W: Tensor<[512, 512], bf16>
    fn forward(self, x: Tensor<[B=1, S=2048, 512], bf16>) -> Tensor:
        return matmul(x, self.W)

train:
    model = Tiny()
    step:
        y = model.forward(x)
"#;
    let m = parse(src);
    let gpu = find_gpu("h100").unwrap();
    let env = ShapeEnv::with_defaults();
    let r = walk_ops(&m, EntryKind::Train, &env, gpu, "bf16").unwrap();
    assert!(r.ops.iter().any(|o| o.name == "matmul"), "matmul should be inlined from model.forward");
}

#[test]
fn unknown_op_yields_zero_cost_with_note() {
    let src = r#"
fn forward(x: Tensor<[B=1, 512], bf16>) -> Tensor:
    let y = mystery_op(x)   # not in cost_model
    return y
"#;
    let m = parse(src);
    let gpu = find_gpu("h100").unwrap();
    let env = ShapeEnv::with_defaults();
    let r = walk_ops(&m, EntryKind::Auto, &env, gpu, "bf16").unwrap();
    assert_eq!(r.ops.len(), 1);
    assert_eq!(r.ops[0].flops, 0);
    assert!(r.ops[0].loc.contains("unknown op") || r.ops[0].name.contains("mystery_op"));
}

#[test]
fn unresolved_shape_var_does_not_panic() {
    let src = r#"
fn forward(x: Tensor<[B, S, D], bf16>) -> Tensor:
    let h = layernorm(x)
    return h
"#;
    let m = parse(src);
    let gpu = find_gpu("h100").unwrap();
    let mut env = ShapeEnv::new();      // empty — no B, S, D
    let r = walk_ops(&m, EntryKind::Auto, &env, gpu, "bf16").unwrap();
    // Should produce an OpCost with note rather than panic.
    assert_eq!(r.ops.len(), 1);
    assert_eq!(r.ops[0].flops, 0);
}
```

- [ ] **Step 7: Run — expect fail**

```bash
cargo test -p nsl-codegen --test profiling_walker 2>&1 | tail -15
```

- [ ] **Step 8: Implement `walker.rs`**

```rust
//! Typed-AST walker that produces a ProfileReport by dispatching each op
//! through the existing per-op cost functions in cost_model.rs.

use crate::cost_model::{
    self, arithmetic_intensity, classify_op, BoundClassification, OpCost,
};
use crate::gpu_specs::GpuSpec;
use crate::profiling::shape_env::ShapeEnv;
use crate::profiling::types::{EntryKind, MemoryTimelineEntry, ProfileReport, Recommendation};
use nsl_ast::{expr::Expr, stmt::Stmt, Module};

pub fn walk_ops(
    module: &Module,
    entry: EntryKind,
    env: &ShapeEnv,
    gpu: &GpuSpec,
    dtype: &str,
) -> Result<ProfileReport, String> {
    let dtype_bytes = match dtype {
        "bf16" | "fp16" => 2,
        "fp8" => 1,
        "fp32" => 4,
        other => return Err(format!("unsupported dtype: {other}")),
    };

    let body = resolve_entry_body(module, &entry)?;
    let mut collector = WalkCtx { ops: vec![], env, gpu, dtype_bytes };
    collector.walk_block(&body);
    let ops = collector.ops;

    let total_flops = ops.iter().map(|o| o.flops).sum();
    let total_hbm = ops.iter().map(|o| o.bytes_read + o.bytes_written).sum();
    let total_us = ops.iter().map(|o| o.estimated_time_us).sum();

    let recommendations = build_recommendations(&ops, total_us);

    Ok(ProfileReport {
        target_gpu: gpu.name.to_string(),
        dtype: dtype.to_string(),
        entry,
        ops,
        total_flops,
        total_hbm_bytes: total_hbm,
        total_estimated_us: total_us,
        fusion: None,
        memory_timeline: None,
        recommendations,
    })
}

struct WalkCtx<'a> {
    ops: Vec<OpCost>,
    env: &'a ShapeEnv,
    gpu: &'a GpuSpec,
    dtype_bytes: u64,
}

impl<'a> WalkCtx<'a> {
    fn walk_block(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.walk_stmt(s);
        }
    }
    fn walk_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let(l) => self.walk_expr(&l.expr, Some(&l.name.to_string())),
            Stmt::Return(e) => self.walk_expr(e, None),
            Stmt::Expr(e) => self.walk_expr(e, None),
            Stmt::FnDef(f) => self.walk_block(&f.body.stmts),
            _ => {}
        }
    }
    fn walk_expr(&mut self, e: &Expr, binding: Option<&str>) {
        // Dispatch on call kind.
        if let Some((callee, args)) = e.as_call() {
            match callee.as_str() {
                "matmul" => self.emit_matmul(e, args),
                "flash_attention" | "flash_attn" => self.emit_flash_attn(e, args),
                "softmax" => self.emit_softmax(e, args),
                "layernorm" | "rmsnorm" => self.emit_norm(e, args),
                "embedding" => self.emit_embed(e, args),
                "model.forward" | "forward" => self.inline_forward(e, args),
                _ => self.emit_unknown(e, callee),
            }
        }
    }
    fn emit_matmul(&mut self, e: &Expr, args: &[Expr]) {
        // Resolve [M,K] × [K,N] from arg shapes + env.
        let (m, k, n) = match shape_2d_pair(&args[0], &args[1], self.env) {
            Some(t) => t,
            None => return self.emit_with_zero(e, "matmul", "unresolved shape"),
        };
        let (flops, br, bw) = cost_model::matmul_cost(m, k, n, self.dtype_bytes);
        self.push(e, "matmul", flops, br, bw);
    }
    fn emit_flash_attn(&mut self, e: &Expr, args: &[Expr]) {
        let dims = shape_bhsd(&args[0], self.env);
        let Some((b, h, s, d)) = dims else {
            return self.emit_with_zero(e, "flash_attn", "unresolved shape");
        };
        let (flops, br, bw) = cost_model::flash_attention_cost(b, h, s, d, self.dtype_bytes);
        self.push(e, "flash_attn", flops, br, bw);
    }
    fn emit_softmax(&mut self, e: &Expr, args: &[Expr]) {
        let Some((b, s)) = shape_bs(&args[0], self.env) else {
            return self.emit_with_zero(e, "softmax", "unresolved shape");
        };
        let (flops, br, bw) = cost_model::softmax_cost(b, s, self.dtype_bytes);
        self.push(e, "softmax", flops, br, bw);
    }
    fn emit_norm(&mut self, e: &Expr, args: &[Expr]) {
        let Some((b, s, d)) = shape_bsd(&args[0], self.env) else {
            return self.emit_with_zero(e, "layernorm", "unresolved shape");
        };
        let (flops, br, bw) = cost_model::layernorm_cost(b, s, d, self.dtype_bytes);
        self.push(e, "layernorm", flops, br, bw);
    }
    fn emit_embed(&mut self, e: &Expr, args: &[Expr]) {
        let Some((b, s, d)) = shape_bsd(&args[1], self.env) else {
            return self.emit_with_zero(e, "embedding", "unresolved shape");
        };
        let (flops, br, bw) = cost_model::embedding_cost(b, s, d, self.dtype_bytes);
        self.push(e, "embedding", flops, br, bw);
    }
    fn emit_unknown(&mut self, e: &Expr, callee: &str) {
        self.push_raw(OpCost {
            name: callee.to_string(),
            loc: format!("unknown op @ {}", span_label(e)),
            input_shapes: vec![],
            output_shape: String::new(),
            flops: 0,
            bytes_read: 0,
            bytes_written: 0,
            arithmetic_intensity: 0.0,
            classification: BoundClassification::Unknown,
            fused: false,
            estimated_time_us: 0.0,
        });
    }
    fn emit_with_zero(&mut self, e: &Expr, name: &str, reason: &str) {
        self.push_raw(OpCost {
            name: name.to_string(),
            loc: format!("{} ({})", span_label(e), reason),
            input_shapes: vec![],
            output_shape: String::new(),
            flops: 0, bytes_read: 0, bytes_written: 0,
            arithmetic_intensity: 0.0,
            classification: BoundClassification::Unknown,
            fused: false,
            estimated_time_us: 0.0,
        });
    }
    fn push(&mut self, e: &Expr, name: &str, flops: u64, br: u64, bw: u64) {
        let ai = arithmetic_intensity(flops, br, bw);
        let cls = classify_op(ai, self.gpu.crossover_fp16);
        let time_us = estimate_time_us(flops, br + bw, self.gpu, self.dtype_bytes);
        self.push_raw(OpCost {
            name: name.to_string(),
            loc: span_label(e),
            input_shapes: vec![],
            output_shape: String::new(),
            flops,
            bytes_read: br,
            bytes_written: bw,
            arithmetic_intensity: ai,
            classification: cls,
            fused: false,
            estimated_time_us: time_us,
        });
    }
    fn push_raw(&mut self, c: OpCost) { self.ops.push(c); }

    fn inline_forward(&mut self, _e: &Expr, _args: &[Expr]) {
        // Resolve the model reference, find its `forward` (or `forward_train`)
        // method, and walk its body. Recursion guard via a visited set keyed by
        // (model_name, method_name) to prevent infinite inlining.
        // See `resolve_model_method` helper.
    }
}

fn estimate_time_us(flops: u64, bytes: u64, gpu: &GpuSpec, dtype_bytes: u64) -> f64 {
    let compute_us = (flops as f64) / (gpu.peak_tflops(dtype_bytes) * 1e6);
    let mem_us = (bytes as f64) / (gpu.peak_bandwidth_gbs * 1e3);
    compute_us.max(mem_us)
}

fn span_label(e: &Expr) -> String {
    let s = e.span();
    format!("{}:{}-{}", s.file_id.0, s.start.0, s.end.0)
}

fn build_recommendations(ops: &[OpCost], total_us: f64) -> Vec<Recommendation> {
    let mut out = vec![];
    for op in ops {
        if matches!(op.classification, BoundClassification::MemoryBound) && op.arithmetic_intensity < 2.0 {
            out.push(Recommendation::memory_bound_batch_hint(&op.name));
        }
        if total_us > 0.0 && op.estimated_time_us / total_us > 0.10 {
            out.push(Recommendation::dominating_op(&op.name, 100.0 * op.estimated_time_us / total_us));
        }
    }
    out
}

// --- helpers (implementation sketch; concrete code fits existing Expr API) ---

fn resolve_entry_body(m: &Module, entry: &EntryKind) -> Result<Vec<Stmt>, String> {
    match entry {
        EntryKind::Auto => {
            if let Some(tb) = m.stmts.iter().find_map(|s| if let Stmt::Train(t) = s { Some(t) } else { None }) {
                return Ok(train_step_stmts(tb));
            }
            if let Some(f) = m.stmts.iter().find_map(|s| if let Stmt::FnDef(f) = s { Some(f) } else { None }) {
                return Ok(f.body.stmts.clone());
            }
            Err("no entry point found (need a fn or train block)".into())
        }
        EntryKind::Train => {
            let tb = m.stmts.iter().find_map(|s| if let Stmt::Train(t) = s { Some(t) } else { None })
                .ok_or_else(|| "no train block in module".to_string())?;
            Ok(train_step_stmts(tb))
        }
        EntryKind::Function(name) => {
            let f = m.stmts.iter().find_map(|s| match s {
                Stmt::FnDef(f) if f.name.as_str() == name => Some(f),
                _ => None,
            }).ok_or_else(|| format!("function not found: {name}"))?;
            Ok(f.body.stmts.clone())
        }
    }
}

fn train_step_stmts(tb: &nsl_ast::block::TrainBlock) -> Vec<Stmt> {
    tb.sections.iter()
        .filter_map(|sec| sec.step_body())
        .flatten()
        .cloned()
        .collect()
}

fn shape_2d_pair(_a: &Expr, _b: &Expr, _env: &ShapeEnv) -> Option<(u64, u64, u64)> {
    // Call `e.inferred_type()` on each arg, read its shape dims, resolve symbolic
    // names via `env.resolve(name)`. Returns None on any unresolved dim.
    // Concrete code depends on the nsl-semantic type representation — see
    // `TensorType::shape()` accessor (add one if missing).
    todo!("extract [M,K] × [K,N] from typed expressions")
}
fn shape_bhsd(_a: &Expr, _env: &ShapeEnv) -> Option<(u64,u64,u64,u64)> { todo!() }
fn shape_bs(_a: &Expr, _env: &ShapeEnv) -> Option<(u64,u64)> { todo!() }
fn shape_bsd(_a: &Expr, _env: &ShapeEnv) -> Option<(u64,u64,u64)> { todo!() }
```

Note on `todo!()`: the shape extraction helpers depend on the concrete typed-AST API in `nsl-semantic`. Before marking this task done, replace every `todo!()` with a real implementation that reads dims off `Expr::inferred_type()`. If the necessary accessor doesn't exist, add it in a small sibling commit in `nsl-semantic`.

- [ ] **Step 9: Run walker tests — iterate until all 4 pass**

```bash
cargo test -p nsl-codegen --test profiling_walker
```

Expected: 7 passed (3 shape_env + 4 walker).

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-codegen/src/profiling/walker.rs crates/nsl-codegen/src/profiling/mod.rs \
        crates/nsl-codegen/tests/profiling_walker.rs
git commit -m "feat(profiling): walker dispatches typed-AST ops through cost_model"
```

---

## Task 4: `nsl profile` subcommand + per-op table renderer

**Files:**
- Create: `crates/nsl-cli/src/profile.rs`
- Modify: `crates/nsl-cli/src/main.rs` (add `Profile` variant)
- Test: `crates/nsl-cli/tests/profile_cmd.rs`

- [ ] **Step 1: Write failing snapshot test**

`crates/nsl-cli/tests/profile_cmd.rs`:

```rust
use nsl_cli::profile::{run_profile, ProfileArgs};
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn snapshot_tiny_transformer_h100_bf16() {
    let args = ProfileArgs {
        file: fixture("tiny_transformer.nsl"),
        target: "h100".into(),
        dtype: "bf16".into(),
        batch: 1, seq: 2048,
        dim: vec![],
        fusion: true,
        memory: false,
        entry: "auto".into(),
        json: false,
    };
    let out = run_profile(&args).unwrap();
    insta::assert_snapshot!("tiny_transformer_h100_bf16", out);
}

#[test]
fn json_flag_emits_parseable_report() {
    let args = ProfileArgs {
        json: true,
        ..sample_args()
    };
    let out = run_profile(&args).unwrap();
    let _: serde_json::Value = serde_json::from_str(&out).expect("must be valid JSON");
}

#[test]
fn bad_gpu_errors_with_available_list() {
    let args = ProfileArgs { target: "nonsuch".into(), ..sample_args() };
    let err = run_profile(&args).unwrap_err();
    assert!(err.contains("nonsuch"));
    assert!(err.to_lowercase().contains("available"));
}

fn sample_args() -> ProfileArgs {
    ProfileArgs {
        file: fixture("tiny_transformer.nsl"),
        target: "h100".into(), dtype: "bf16".into(),
        batch: 1, seq: 2048, dim: vec![],
        fusion: true, memory: false, entry: "auto".into(), json: false,
    }
}
```

Also commit `crates/nsl-cli/tests/fixtures/tiny_transformer.nsl` — a minimal 2-layer transformer in NSL (≤30 lines) suitable for snapshotting.

- [ ] **Step 2: Run — expect fail**

```bash
cargo test -p nsl-cli --test profile_cmd 2>&1 | tail -15
```

- [ ] **Step 3: Implement `profile.rs`**

```rust
use crate::profile_glue;
use nsl_codegen::cost_model::format_perf_table;
use nsl_codegen::gpu_specs::{find_gpu, GPU_DATABASE};
use nsl_codegen::profiling::shape_env::ShapeEnv;
use nsl_codegen::profiling::types::{EntryKind, ProfileReport};
use nsl_codegen::profiling::walker::walk_ops;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ProfileArgs {
    pub file: PathBuf,
    pub target: String,
    pub dtype: String,
    pub batch: u64,
    pub seq: u64,
    pub dim: Vec<String>,       // "name=N" strings
    pub fusion: bool,
    pub memory: bool,
    pub entry: String,          // "auto" | "train" | "fn:<name>"
    pub json: bool,
}

pub fn run_profile(args: &ProfileArgs) -> Result<String, String> {
    let gpu = find_gpu(&args.target).ok_or_else(|| {
        let available: Vec<_> = GPU_DATABASE.iter().map(|g| g.name).collect();
        format!("unknown GPU target: {}; available: {:?}", args.target, available)
    })?;

    let src = std::fs::read_to_string(&args.file).map_err(|e| e.to_string())?;
    let (module, _) = crate::driver::analyze(&src, args.file.to_str().unwrap())?;

    let mut env = ShapeEnv::with_defaults();
    env.set("batch", args.batch);
    env.set("seq", args.seq);
    for f in &args.dim { env.parse_dim_flag(f)?; }

    let entry = EntryKind::parse_flag(&args.entry).ok_or("bad --entry value")?;
    let mut report = walk_ops(&module, entry, &env, gpu, &args.dtype)?;

    if args.fusion {
        // Attach existing FusionPlan — reuse WRGA's builder with empty adapter
        // placements as a default (no adapters inserted yet).
        report.fusion = Some(nsl_codegen::wrga_fusion::build_fusion_plan(&[], None));
    }
    if args.memory {
        report.memory_timeline = Some(profile_glue::memory_timeline(&module, &env)?);
    }
    if args.json {
        return Ok(serde_json::to_string_pretty(&report).unwrap());
    }
    Ok(render_text(&report, gpu))
}

fn render_text(r: &ProfileReport, gpu: &nsl_codegen::gpu_specs::GpuSpec) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== NSL Predictive Profile ===\nTarget: {} ({:.0} TFLOPS {}, {:.2} TB/s HBM)\n\n",
        gpu.name, gpu.peak_fp16_tflops, r.dtype, gpu.peak_bandwidth_gbs / 1000.0
    ));
    out.push_str(&format_perf_table(&r.ops, gpu, &r.dtype));
    if let Some(fp) = &r.fusion {
        out.push_str("\nFusion summary:\n");
        for d in &fp.decisions {
            out.push_str(&format!("  {} → {:?}  ({} extra HBM bytes)  — {}\n",
                d.site, d.target, d.extra_hbm_bytes, d.rationale));
        }
    }
    if let Some(tl) = &r.memory_timeline {
        out.push_str("\n");
        out.push_str(&nsl_codegen::profiling::memory_timeline::render(tl));
    }
    if !r.recommendations.is_empty() {
        out.push_str("\nRecommendations:\n");
        for (i, rec) in r.recommendations.iter().enumerate() {
            out.push_str(&format!("  [{}] {} {}\n", i + 1, rec.code, rec.message));
        }
    }
    out
}
```

Add a thin `crates/nsl-cli/src/profile_glue.rs` with `pub fn op_flops_for_expr(e: &Expr) -> Option<u64>` (single-op wrapper around `walker::walk_single_expr`) and `pub fn memory_timeline(...)` (hook for Task 5 — can return `Ok(vec![])` for now and get replaced).

- [ ] **Step 4: Wire clap**

In `main.rs` add:

```rust
Profile {
    file: PathBuf,
    #[arg(long, default_value = "h100")]
    target: String,
    #[arg(long, default_value = "bf16")]
    dtype: String,
    #[arg(long, default_value_t = 1)]
    batch: u64,
    #[arg(long, default_value_t = 2048)]
    seq: u64,
    #[arg(long)]
    dim: Vec<String>,
    #[arg(long, default_value_t = true)]
    fusion: bool,
    #[arg(long)]
    no_fusion: bool,
    #[arg(long)]
    memory: bool,
    #[arg(long, default_value = "auto")]
    entry: String,
    #[arg(long)]
    json: bool,
}
```

Dispatch:

```rust
Cli::Profile { file, target, dtype, batch, seq, dim, fusion, no_fusion, memory, entry, json } => {
    let args = nsl_cli::profile::ProfileArgs {
        file, target, dtype, batch, seq, dim,
        fusion: fusion && !no_fusion,
        memory, entry, json,
    };
    println!("{}", nsl_cli::profile::run_profile(&args)?);
}
```

- [ ] **Step 5: Run tests, review snapshot**

```bash
cargo test -p nsl-cli --test profile_cmd
cargo insta review
```

Expected: after accepting the snapshot, 3 passed.

- [ ] **Step 6: Manual smoke**

```bash
cargo run -p nsl-cli -- profile --target h100 --dtype bf16 --batch 1 --seq 2048 tests/fixtures/tiny_transformer.nsl
cargo run -p nsl-cli -- profile --json tests/fixtures/tiny_transformer.nsl | jq .target_gpu
```

Expected: second command prints `"h100-sxm"` (or similar).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-cli/src/profile.rs crates/nsl-cli/src/profile_glue.rs crates/nsl-cli/src/main.rs \
        crates/nsl-cli/tests/profile_cmd.rs crates/nsl-cli/tests/fixtures/tiny_transformer.nsl \
        crates/nsl-cli/tests/snapshots/
git commit -m "feat(cli): nsl profile emits per-op perf table + JSON output"
```

---

## Task 5: `nsl profile --memory` timeline

**Files:**
- Create: `crates/nsl-codegen/src/profiling/memory_timeline.rs`
- Modify: `crates/nsl-cli/src/profile_glue.rs` (replace stub)
- Test: `crates/nsl-codegen/tests/profiling_memory_timeline.rs`

- [ ] **Step 1: Write failing test**

```rust
use nsl_codegen::profiling::memory_timeline::{build, render, MemoryTimelineInput};
use nsl_codegen::wrga_memory::{MemoryPlan, MemoryPlanStats, SlotAssignment};

fn plan_with_intervals(intervals: &[(u32,u32,u64)]) -> MemoryPlan {
    let assignments = intervals.iter().enumerate().map(|(i, &(birth, death, sz))| {
        SlotAssignment {
            var: Default::default(),
            slot: Default::default(),
            size_bytes: sz, birth, death,
        }
    }).collect();
    MemoryPlan {
        assignments,
        stats: MemoryPlanStats { live_activations: 0, slots_used: 0, naive_peak_bytes: 0, planned_peak_bytes: 0 },
    }
}

#[test]
fn peak_at_overlap() {
    // Three slots: [0..5]=100, [2..8]=200, [4..6]=50. Peak at t=4..5 = 350.
    let plan = plan_with_intervals(&[(0,5,100),(2,8,200),(4,6,50)]);
    let tl = build(&MemoryTimelineInput { plan: &plan, phase_markers: vec![] });
    let peak = tl.iter().map(|e| e.live_bytes).max().unwrap();
    assert_eq!(peak, 350);
}

#[test]
fn renders_ascii_bar_chart() {
    let plan = plan_with_intervals(&[(0, 3, 1024*1024)]);
    let tl = build(&MemoryTimelineInput { plan: &plan, phase_markers: vec![] });
    let s = render(&tl);
    assert!(s.contains("MB"));
    assert!(s.contains("Peak"));
    assert!(s.contains("█") || s.contains("#"));
}

#[test]
fn phase_markers_annotate_rows() {
    let plan = plan_with_intervals(&[(0, 10, 512)]);
    let tl = build(&MemoryTimelineInput {
        plan: &plan,
        phase_markers: vec![(0, "forward_start".into()), (5, "loss".into())],
    });
    assert!(tl.iter().any(|e| e.phase.as_deref() == Some("forward_start")));
    assert!(tl.iter().any(|e| e.phase.as_deref() == Some("loss")));
}
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `memory_timeline.rs`**

```rust
use crate::profiling::types::MemoryTimelineEntry;
use crate::wrga_memory::MemoryPlan;

pub struct MemoryTimelineInput<'a> {
    pub plan: &'a MemoryPlan,
    pub phase_markers: Vec<(u32, String)>,   // (program_point, label)
}

pub fn build(input: &MemoryTimelineInput) -> Vec<MemoryTimelineEntry> {
    let max_pp = input.plan.assignments.iter().map(|s| s.death).max().unwrap_or(0);
    let mut out = Vec::with_capacity((max_pp + 1) as usize);
    for pp in 0..=max_pp {
        let live: u64 = input.plan.assignments.iter()
            .filter(|s| s.birth <= pp && pp < s.death)
            .map(|s| s.size_bytes)
            .sum();
        let phase = input.phase_markers.iter()
            .find(|(q, _)| *q == pp)
            .map(|(_, l)| l.clone());
        out.push(MemoryTimelineEntry { program_point: pp, live_bytes: live, phase });
    }
    out
}

pub fn render(tl: &[MemoryTimelineEntry]) -> String {
    let peak = tl.iter().map(|e| e.live_bytes).max().unwrap_or(0);
    let bar_width = 20;
    let mut out = String::from("=== Memory Timeline ===\n\n");
    out.push_str("Time (pp)  HBM Usage\n");
    for e in tl {
        let filled = if peak == 0 { 0 } else { (e.live_bytes * bar_width as u64 / peak) as usize };
        let bar = "█".repeat(filled) + &"░".repeat(bar_width - filled);
        let mb = e.live_bytes as f64 / (1024.0 * 1024.0);
        let phase = e.phase.as_deref().unwrap_or("");
        out.push_str(&format!("{:>4}       {} {:>7.1} MB  {}\n", e.program_point, bar, mb, phase));
    }
    let peak_mb = peak as f64 / (1024.0 * 1024.0);
    out.push_str(&format!("\nPeak: {:.1} MB\n", peak_mb));
    out
}
```

Add `pub mod memory_timeline;` to `profiling/mod.rs`.

In `profile_glue::memory_timeline`, build a Wengert list from the typed module (reuse the helper `nsl_codegen::wengert::build_from_module(&module)` if it exists — check `crates/nsl-codegen/src/wengert.rs`). Call `plan_memory(...)` with sensible defaults (empty `activation_live`, empty `extra_live_at_end`), then call `build(...)` with an empty `phase_markers` vec (phase markers as a follow-up — left as an enhancement to avoid scope creep).

- [ ] **Step 4: Run — expect pass**

- [ ] **Step 5: Extend `profile_cmd.rs` tests**

```rust
#[test]
fn memory_flag_appends_timeline() {
    let args = ProfileArgs { memory: true, ..sample_args() };
    let out = run_profile(&args).unwrap();
    assert!(out.contains("=== Memory Timeline"));
    assert!(out.contains("Peak:"));
    // Per-op table still present — --memory is additive.
    assert!(out.contains("TOTAL") || out.contains("Time"));
}
```

Run and iterate until green.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/profiling/memory_timeline.rs crates/nsl-codegen/src/profiling/mod.rs \
        crates/nsl-cli/src/profile_glue.rs crates/nsl-codegen/tests/profiling_memory_timeline.rs \
        crates/nsl-cli/tests/profile_cmd.rs
git commit -m "feat(profiling): --memory flag renders HBM usage timeline"
```

---

## Task 6: Codegen instrumentation (`profile: bool` + manifest)

**Files:**
- Create: `crates/nsl-codegen/src/profiling/instrument.rs`
- Modify: the codegen kernel-emit site. Find it via `grep -rn "cuLaunchKernel\|launch_kernel\|emit_kernel" crates/nsl-codegen/src/`. Most likely `crates/nsl-codegen/src/backend_ptx.rs` or `kernel.rs`.
- Modify: codegen entry (e.g. `compile_to_binary(...)` — grep for the function the CLI calls from `Run`) to accept a `profile: bool` parameter.
- Test: `crates/nsl-codegen/tests/profiling_instrument.rs`

- [ ] **Step 1: Write failing test**

```rust
use nsl_codegen::profiling::instrument::{ManifestBuilder, KernelEntry, SourceSpanJson};

#[test]
fn manifest_assigns_dense_ids() {
    let mut mb = ManifestBuilder::new("h100-sxm", "bf16");
    let id0 = mb.record_kernel("fused_attn", SourceSpanJson { file: "m.nsl".into(), start_line: 42, end_line: 48 }, 9.2, 83_900_000, 12_582_912);
    let id1 = mb.record_kernel("gate_proj", SourceSpanJson { file: "m.nsl".into(), start_line: 50, end_line: 50 }, 3.4, 23_100_000, 12_058_624);
    assert_eq!(id0, 0);
    assert_eq!(id1, 1);
    let m = mb.finish();
    assert_eq!(m.kernels.len(), 2);
}

#[test]
fn manifest_records_eliminated_ops_separately() {
    let mut mb = ManifestBuilder::new("h100-sxm", "bf16");
    mb.record_eliminated("chunk", SourceSpanJson { file: "m.nsl".into(), start_line: 45, end_line: 45 }, "fused into kernel 0");
    let m = mb.finish();
    assert!(m.kernels.is_empty());
    assert_eq!(m.eliminated_ops.len(), 1);
    assert_eq!(m.eliminated_ops[0].op_name, "chunk");
}

#[test]
fn manifest_serializes_to_expected_json_shape() {
    let mut mb = ManifestBuilder::new("h100-sxm", "bf16");
    mb.record_kernel("x", SourceSpanJson { file: "m.nsl".into(), start_line: 1, end_line: 1 }, 1.0, 10, 20);
    let s = serde_json::to_string(&mb.finish()).unwrap();
    assert!(s.contains("\"target_gpu\":\"h100-sxm\""));
    assert!(s.contains("\"kernels\""));
    assert!(s.contains("\"eliminated_ops\""));
}
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `instrument.rs`**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpanJson {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelEntry {
    pub kernel_id: u32,
    pub op_name: String,
    pub source_span: SourceSpanJson,
    pub predicted_us: f64,
    pub predicted_flops: u64,
    pub predicted_hbm_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminatedOp {
    pub op_name: String,
    pub source_span: SourceSpanJson,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub target_gpu: String,
    pub dtype: String,
    pub kernels: Vec<KernelEntry>,
    pub eliminated_ops: Vec<EliminatedOp>,
}

pub struct ManifestBuilder {
    inner: Manifest,
    next_id: u32,
}

impl ManifestBuilder {
    pub fn new(target_gpu: &str, dtype: &str) -> Self {
        Self {
            inner: Manifest {
                target_gpu: target_gpu.into(),
                dtype: dtype.into(),
                kernels: vec![],
                eliminated_ops: vec![],
            },
            next_id: 0,
        }
    }
    pub fn record_kernel(&mut self, op_name: &str, span: SourceSpanJson, us: f64, flops: u64, hbm: u64) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.inner.kernels.push(KernelEntry {
            kernel_id: id,
            op_name: op_name.into(),
            source_span: span,
            predicted_us: us, predicted_flops: flops, predicted_hbm_bytes: hbm,
        });
        id
    }
    pub fn record_eliminated(&mut self, op_name: &str, span: SourceSpanJson, reason: &str) {
        self.inner.eliminated_ops.push(EliminatedOp {
            op_name: op_name.into(), source_span: span, reason: reason.into(),
        });
    }
    pub fn finish(self) -> Manifest { self.inner }
}
```

Add `pub mod instrument;` to `profiling/mod.rs`.

- [ ] **Step 4: Run — expect 3 passed**

- [ ] **Step 5: Thread `profile: bool` through codegen**

a. Locate the codegen entry function (`grep -rn "pub fn compile" crates/nsl-codegen/src/`). Add a new boolean parameter `profile: bool` threaded from CLI → driver → codegen. Default it to `false` at every other caller.

b. At each GPU kernel-launch emit site, when `profile == true`:
   - Before the launch: emit a call to the runtime hook `nsl_profile_kernel_begin(kernel_id)` (external C symbol — see Task 7 for the Rust side).
   - After the launch: emit `nsl_profile_kernel_end(kernel_id)`.
   - Call `manifest.record_kernel(op_name, span_json_from(ast_node.span, source_map), predicted_us, predicted_flops, predicted_hbm)` using the prediction already computed by `walk_ops`. The predictions are looked up by matching the op's AST span against the `Vec<OpCost>` from the walker.

c. For every op elided by fusion, call `manifest.record_eliminated(...)`.

d. After codegen completes, serialize the manifest and write to `<out>.nsl-profile.json`.

- [ ] **Step 6: Add an integration test gated on CUDA feature**

```rust
#[cfg(feature = "cuda")]
#[test]
fn compile_with_profile_writes_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let out = tmp.path().join("tiny");
    nsl_codegen::build(/*file*/ "crates/nsl-cli/tests/fixtures/tiny_transformer.nsl",
                       /*out*/ &out, /*profile*/ true, /*target*/ "h100", /*dtype*/ "bf16").unwrap();
    let manifest = std::fs::read_to_string(out.with_extension("nsl-profile.json")).unwrap();
    let m: nsl_codegen::profiling::instrument::Manifest = serde_json::from_str(&manifest).unwrap();
    assert!(!m.kernels.is_empty());
}
```

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/profiling/instrument.rs \
        crates/nsl-codegen/src/profiling/mod.rs \
        crates/nsl-codegen/src/backend_ptx.rs  \
        crates/nsl-codegen/src/lib.rs  \
        crates/nsl-codegen/tests/profiling_instrument.rs
git commit -m "feat(codegen): emit kernel-launch hooks + profile manifest when profile flag is set"
```

---

## Task 7: Runtime collector (`cudaEvent_t` + bounded drain + JSON output)

**Files:**
- Create: `crates/nsl-runtime/src/profiler/mod.rs`
- Create: `crates/nsl-runtime/src/profiler/collector.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` (`pub mod profiler;` behind `#[cfg(feature = "cuda")]`)
- Test: `crates/nsl-runtime/tests/profiler_collector.rs`

- [ ] **Step 1: Write failing test (no-CUDA path — uses injectable clock)**

```rust
use nsl_runtime::profiler::collector::{Collector, ClockSource};

struct FakeClock(std::cell::Cell<f64>);
impl ClockSource for FakeClock {
    fn elapsed_us(&self, _start: u64, _end: u64) -> f64 { self.0.get() }
}

#[test]
fn aggregates_across_many_pairs_without_growing_unbounded() {
    let clock = FakeClock(std::cell::Cell::new(1.0));
    let mut c = Collector::new_with_clock(Box::new(clock));
    for i in 0..10_000 {
        c.begin(/*kernel_id*/ 0, /*raw_start*/ i);
        c.end(/*kernel_id*/ 0, /*raw_end*/ i);
    }
    let agg = c.snapshot().into_iter().find(|a| a.kernel_id == 0).unwrap();
    assert_eq!(agg.count, 10_000);
    assert!((agg.sum_us - 10_000.0).abs() < 1e-3);
    // Internal buffer stays bounded: after snapshot, drain is empty.
    assert!(c.drain_queue_len(0) <= 64);
}

#[test]
fn flush_writes_actual_json_in_expected_shape() {
    let clock = FakeClock(std::cell::Cell::new(2.5));
    let mut c = Collector::new_with_clock(Box::new(clock));
    c.begin(3, 0); c.end(3, 1);
    let tmp = tempfile::NamedTempFile::new().unwrap();
    c.flush_to(tmp.path()).unwrap();
    let s = std::fs::read_to_string(tmp.path()).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert!(v["aggregates"][0]["kernel_id"] == 3);
    assert!(v["aggregates"][0]["count"] == 1);
}
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `collector.rs`**

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub trait ClockSource: Send {
    fn elapsed_us(&self, start: u64, end: u64) -> f64;
}

#[cfg(feature = "cuda")]
pub struct CudaClock;
#[cfg(feature = "cuda")]
impl ClockSource for CudaClock {
    fn elapsed_us(&self, start: u64, end: u64) -> f64 {
        // cudaEventElapsedTime returns ms; start/end are cudaEvent_t cast to u64.
        // Call the cudarc FFI here, then convert ms -> us.
        unsafe {
            let mut ms: f32 = 0.0;
            cudarc::driver::sys::lib()
                .cuEventElapsedTime(&mut ms as *mut _,
                                    start as cudarc::driver::sys::CUevent,
                                    end as cudarc::driver::sys::CUevent)
                .result().unwrap();
            ms as f64 * 1000.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregate {
    pub kernel_id: u32,
    pub count: u64,
    pub sum_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub sum_sq_us: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ActualReport {
    pub aggregates: Vec<Aggregate>,
}

const DRAIN_CAP: usize = 64;

pub struct Collector {
    clock: Box<dyn ClockSource>,
    in_flight: HashMap<u32, u64>,                    // kernel_id -> start event handle
    drains: HashMap<u32, Vec<(u64, u64)>>,           // kernel_id -> pending (start, end)
    aggregates: HashMap<u32, Aggregate>,
}

impl Collector {
    pub fn new_with_clock(clock: Box<dyn ClockSource>) -> Self {
        Self { clock, in_flight: HashMap::new(), drains: HashMap::new(), aggregates: HashMap::new() }
    }
    pub fn begin(&mut self, kernel_id: u32, start_handle: u64) {
        self.in_flight.insert(kernel_id, start_handle);
    }
    pub fn end(&mut self, kernel_id: u32, end_handle: u64) {
        if let Some(start) = self.in_flight.remove(&kernel_id) {
            let q = self.drains.entry(kernel_id).or_default();
            q.push((start, end_handle));
            if q.len() >= DRAIN_CAP { self.drain(kernel_id); }
        }
    }
    fn drain(&mut self, kernel_id: u32) {
        let Some(q) = self.drains.get_mut(&kernel_id) else { return };
        let agg = self.aggregates.entry(kernel_id).or_insert(Aggregate {
            kernel_id, count: 0, sum_us: 0.0,
            min_us: f64::INFINITY, max_us: 0.0, sum_sq_us: 0.0,
        });
        for (s, e) in q.drain(..) {
            let us = self.clock.elapsed_us(s, e);
            agg.count += 1;
            agg.sum_us += us;
            agg.sum_sq_us += us * us;
            if us < agg.min_us { agg.min_us = us; }
            if us > agg.max_us { agg.max_us = us; }
        }
    }
    pub fn drain_queue_len(&self, kernel_id: u32) -> usize {
        self.drains.get(&kernel_id).map_or(0, |v| v.len())
    }
    pub fn snapshot(&mut self) -> Vec<Aggregate> {
        let ids: Vec<_> = self.drains.keys().copied().collect();
        for id in ids { self.drain(id); }
        self.aggregates.values().cloned().collect()
    }
    pub fn flush_to(&mut self, path: &Path) -> std::io::Result<()> {
        let report = ActualReport { aggregates: self.snapshot() };
        let s = serde_json::to_string_pretty(&report)?;
        std::fs::write(path, s)
    }
}
```

- [ ] **Step 4: Add FFI hooks**

```rust
// crates/nsl-runtime/src/profiler/ffi.rs
use std::cell::RefCell;
thread_local! { static COLLECTOR: RefCell<Option<super::collector::Collector>> = RefCell::new(None); }

#[no_mangle]
pub extern "C" fn nsl_profile_kernel_begin(kernel_id: u32) { /* create event, call begin */ }
#[no_mangle]
pub extern "C" fn nsl_profile_kernel_end(kernel_id: u32) { /* create event, call end */ }
#[no_mangle]
pub extern "C" fn nsl_profile_flush(path_ptr: *const u8, path_len: usize) -> i32 { /* ... */ 0 }
```

Concrete CUDA event creation uses `cudarc::driver::sys::cuEventCreate` and `cuEventRecord` on the current stream. See existing patterns in `crates/nsl-runtime/src/cuda*`.

- [ ] **Step 5: Run — expect 2 passed (host-only tests)**

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/profiler/ crates/nsl-runtime/src/lib.rs crates/nsl-runtime/tests/profiler_collector.rs
git commit -m "feat(runtime): bounded CUDA-event collector + flush to actual JSON"
```

---

## Task 8: `nsl run --monitor` — predicted-vs-actual + source-mapped kernel view

**Files:**
- Create: `crates/nsl-cli/src/monitor.rs`
- Modify: `crates/nsl-cli/src/main.rs` (add `monitor: bool` on `Run` variant; compile with profile enabled when set; after child process exits, invoke the renderer)
- Test: `crates/nsl-cli/tests/monitor.rs`

- [ ] **Step 1: Write failing test for comparison + source view**

```rust
use nsl_cli::monitor::{render_comparison, render_source_view, MonitorInputs};
use nsl_codegen::profiling::instrument::{Manifest, KernelEntry, EliminatedOp, SourceSpanJson};
use nsl_runtime::profiler::collector::{ActualReport, Aggregate};

fn mk_manifest() -> Manifest {
    Manifest {
        target_gpu: "h100-sxm".into(), dtype: "bf16".into(),
        kernels: vec![
            KernelEntry { kernel_id: 0, op_name: "fused_attn".into(),
                source_span: SourceSpanJson { file: "m.nsl".into(), start_line: 42, end_line: 48 },
                predicted_us: 9.2, predicted_flops: 0, predicted_hbm_bytes: 0 },
        ],
        eliminated_ops: vec![
            EliminatedOp { op_name: "chunk".into(),
                source_span: SourceSpanJson { file: "m.nsl".into(), start_line: 45, end_line: 45 },
                reason: "fused into kernel 0".into() },
        ],
    }
}

fn mk_actual(us: f64) -> ActualReport {
    ActualReport { aggregates: vec![
        Aggregate { kernel_id: 0, count: 100, sum_us: us * 100.0,
                    min_us: us, max_us: us, sum_sq_us: us * us * 100.0 },
    ] }
}

#[test]
fn comparison_flags_large_divergence() {
    let out = render_comparison(&mk_manifest(), &mk_actual(11.5));  // +25%
    assert!(out.contains("fused_attn"));
    assert!(out.contains("+25"));  // approximate percentage
    assert!(out.contains("❌") || out.contains("⚠"));
}

#[test]
fn comparison_small_delta_no_warning() {
    let out = render_comparison(&mk_manifest(), &mk_actual(9.5));   // +3.3%
    assert!(!out.contains("❌"));
}

#[test]
fn source_view_marks_eliminated_lines() {
    let src = "line1\nline2\n..line42..forward..\n..line43..\n..line44..\n..line45 chunk..\n..line46..\n..line47..\n..line48 end..";
    let out = render_source_view(&mk_manifest(), &mk_actual(9.2), src, "m.nsl");
    assert!(out.contains("fused_attn"));
    assert!(out.contains("eliminated"));
    assert!(out.contains("45"));
}
```

- [ ] **Step 2: Run — expect fail**

- [ ] **Step 3: Implement `monitor.rs`**

```rust
use nsl_codegen::profiling::instrument::{Manifest, KernelEntry, EliminatedOp};
use nsl_runtime::profiler::collector::ActualReport;

pub struct MonitorInputs {
    pub manifest: Manifest,
    pub actual: ActualReport,
    pub source: String,
    pub file: String,
}

pub fn render_comparison(manifest: &Manifest, actual: &ActualReport) -> String {
    let mut out = String::from("=== Predicted vs Actual Performance ===\n\n");
    out.push_str(&format!("{:<20} {:>10} {:>10} {:>10}\n", "Operation", "Predicted", "Actual", "Δ%"));
    for k in &manifest.kernels {
        let agg = actual.aggregates.iter().find(|a| a.kernel_id == k.kernel_id);
        let (actual_us, delta_pct) = match agg {
            Some(a) if a.count > 0 => {
                let mean = a.sum_us / a.count as f64;
                let d = (mean - k.predicted_us) / k.predicted_us * 100.0;
                (mean, d)
            }
            _ => (f64::NAN, f64::NAN),
        };
        let mark = if delta_pct.abs() > 20.0 { "❌" }
                   else if delta_pct.abs() > 5.0 { "⚠" }
                   else { "" };
        out.push_str(&format!("{:<20} {:>8.1}μs {:>8.1}μs {:>+8.1}% {}\n",
            k.op_name, k.predicted_us, actual_us, delta_pct, mark));
        if delta_pct.abs() > 20.0 {
            out.push_str(&format!("   → {}\n", likely_cause(k, delta_pct)));
        }
    }
    out
}

fn likely_cause(k: &KernelEntry, _delta: f64) -> String {
    // Simple heuristics (spec §4.4):
    match k.op_name.as_str() {
        s if s.contains("matmul") || s.contains("proj") =>
            "tile-size misalignment; check inner dim % 128".into(),
        s if s.contains("attn") =>
            "SMEM pressure / bank conflicts; try --csha-tile=64".into(),
        _ => "cause unknown; rerun with --target-profile=detailed".into(),
    }
}

pub fn render_source_view(m: &Manifest, actual: &ActualReport, src: &str, file: &str) -> String {
    let mut out = String::from("\n=== Source-Mapped Kernel View ===\n\n");
    let lines: Vec<&str> = src.lines().collect();

    // Group kernels + eliminated ops by (file, start_line).
    let mut annots: Vec<(u32, String)> = vec![];
    for k in &m.kernels {
        if k.source_span.file != file { continue; }
        let agg = actual.aggregates.iter().find(|a| a.kernel_id == k.kernel_id);
        let actual_us = agg.map(|a| a.sum_us / a.count.max(1) as f64).unwrap_or(f64::NAN);
        for line in k.source_span.start_line..=k.source_span.end_line {
            annots.push((line, format!("← {} ({:.1}μs actual)", k.op_name, actual_us)));
        }
    }
    for elim in &m.eliminated_ops {
        if elim.source_span.file != file { continue; }
        for line in elim.source_span.start_line..=elim.source_span.end_line {
            annots.push((line, format!("← eliminated ({})", elim.reason)));
        }
    }
    annots.sort_by_key(|(l, _)| *l);

    let (first, last) = (annots.first().map(|x| x.0).unwrap_or(1),
                         annots.last().map(|x| x.0).unwrap_or(lines.len() as u32));
    for (i, line) in lines.iter().enumerate() {
        let n = (i + 1) as u32;
        if n < first || n > last { continue; }
        let ann = annots.iter().find(|(ln, _)| *ln == n).map(|(_, a)| a.as_str()).unwrap_or("");
        out.push_str(&format!("{:>4} | {:<60} {}\n", n, line, ann));
    }
    out
}

pub fn run_monitor(file: &std::path::Path, manifest_path: &std::path::Path,
                   actual_path: &std::path::Path) -> Result<String, String> {
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(manifest_path).map_err(|e| e.to_string())?
    ).map_err(|e| e.to_string())?;
    let actual: ActualReport = match std::fs::read_to_string(actual_path) {
        Ok(s) => serde_json::from_str(&s).map_err(|e| e.to_string())?,
        Err(_) => {
            eprintln!("warning: actual JSON missing — run may have crashed; showing predictions only");
            ActualReport { aggregates: vec![] }
        }
    };
    let src = std::fs::read_to_string(file).map_err(|e| e.to_string())?;
    let mut out = render_comparison(&manifest, &actual);
    out.push_str(&render_source_view(&manifest, &actual, &src, file.to_str().unwrap()));
    Ok(out)
}
```

Add `pub mod monitor;` to `crates/nsl-cli/src/lib.rs`.

- [ ] **Step 4: Wire the `--monitor` flag**

Add `monitor: bool` to the existing `Run` variant (it already has `profile: bool`; treat `--monitor` as a superset — it implies `profile: true` *plus* post-run rendering). In the `Run` match arm:

```rust
if monitor {
    // Set profile=true so codegen emits hooks and manifest.
    compile_and_run_with_profile(&file, /*profile*/ true)?;
    let manifest_path = file.with_extension("nsl-profile.json");
    let actual_path = file.with_extension("nsl-profile-actual.json");
    println!("{}", nsl_cli::monitor::run_monitor(&file, &manifest_path, &actual_path)?);
    return Ok(());
}
```

- [ ] **Step 5: Run monitor tests**

```bash
cargo test -p nsl-cli --test monitor
```

Expected: 3 passed.

- [ ] **Step 6: End-to-end smoke (CUDA-only)**

```bash
cargo run -p nsl-cli --features cuda -- run --monitor tests/fixtures/tiny_transformer.nsl
```

Expected: predicted-vs-actual table + source-mapped view printed. No crashes.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-cli/src/monitor.rs crates/nsl-cli/src/main.rs crates/nsl-cli/tests/monitor.rs
git commit -m "feat(cli): nsl run --monitor renders predicted-vs-actual + source view"
```

---

## Task 9: Final review + workspace check

- [ ] **Step 1: Run full test suite**

```bash
cargo test --workspace
```

Expected: green. All new tests pass; no regressions.

- [ ] **Step 2: Clippy pass**

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Fix any warnings in new code. Leave pre-existing warnings alone.

- [ ] **Step 3: Manual acceptance checklist**

For each, verify by running and eyeballing output:

```bash
cargo run -p nsl-cli -- check --shapes tests/fixtures/tiny_transformer.nsl
cargo run -p nsl-cli -- profile tests/fixtures/tiny_transformer.nsl
cargo run -p nsl-cli -- profile --memory tests/fixtures/tiny_transformer.nsl
cargo run -p nsl-cli -- profile --json tests/fixtures/tiny_transformer.nsl
cargo run -p nsl-cli --features cuda -- run --monitor tests/fixtures/tiny_transformer.nsl
```

Each should produce the output style in the PDF §3.1 / §3.2 / §4.1 / §4.2 / §5.

- [ ] **Step 4: Final commit / open PR**

```bash
git log --oneline main..HEAD
gh pr create --title "Phase 1: NSL Dev Tools — shape debug, profile, memory timeline, monitor" --body-file docs/superpowers/specs/2026-04-12-nsl-dev-tools-phase1-design.md
```

---

## Self-Review

**Spec coverage:**

- Feature 1 `nsl check --shapes` → Task 2. ✅
- Feature 2 `nsl profile` (walker + renderer + `--json` + entry selection + `--batch`/`--seq`/`--dim`) → Tasks 3 + 4. ✅
- Feature 3 `nsl profile --memory` (additive) → Task 5. ✅
- Feature 4 `nsl run --monitor` (codegen instrumentation + runtime collector + predicted-vs-actual table + source-mapped view + bounded drain queue) → Tasks 6, 7, 8. ✅
- Non-goals (no WGGO explain, no training health monitor, no `@inspect`) — respected, not in any task. ✅

**Placeholder scan:** The walker contains `todo!()` for shape-extraction helpers whose exact implementation depends on the typed-AST API. This is called out explicitly in Task 3 Step 8 with guidance on what to implement. Not a drive-by placeholder — a known boundary that the implementer resolves by reading `nsl-semantic`. All other steps have concrete code or concrete commands.

**Type consistency:** `ProfileReport`, `OpCost`, `Manifest`, `Aggregate`, `ActualReport`, `SourceSpanJson`, `EntryKind` are defined once (Tasks 1, 6, 7) and referenced consistently afterward. `ShapeEnv` API (`new`, `with_defaults`, `set`, `resolve`, `parse_dim_flag`) is consistent across Tasks 3 and 4. Collector methods (`begin`, `end`, `snapshot`, `flush_to`, `drain_queue_len`) are consistent across test and impl.

**Scope:** Each task produces testable software on its own; the implementer can stop at the end of any task and have a working, merged piece.
